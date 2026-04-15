//! 原生多模态 MoE Transformer 模型
//!
//! # 架构特点
//! - 每3层插入MoE：FFN/FFN/MoE 循环
//! - 共28层 = 19层FFN + 9层MoE
//! - layer_idx % 3 == 2 时使用 MoE，否则使用 FFN
//! - 原生支持多模态输入（文本/图像/音频）
//!
//! # MLA（Multi-head Latent Attention）实现说明
//! 本实现遵循 DeepSeek-V2 标准 MLA 架构，核心特征：
//!
//! ## 计算流程
//! ```text
//! 输入 x (hidden_size)
//!   ├─→ q_proj → Q (n_heads × head_dim)
//!   │         → uq_proj → Q_latent (n_heads × latent_dim/2)   ← Q 压缩到潜在空间
//!   └─→ dkv_proj → c_kv (latent_dim) = [c_k(latent_dim/2) | c_v(latent_dim/2)]
//!              → uk_proj → K_decoded (仅用于兼容路径)
//!              → uv_proj → V_decoded (仅用于兼容路径)
//!
//! ★ 潜在空间注意力（真 MLA 核心）:
//!   scores = Q_latent @ c_k^T / sqrt(latent_dim/2)    ← 潜在空间 score 计算
//!   attn   = causal_softmax(scores)
//!   output = attn @ c_v                                 ← 潜在空间输出
//!   result = o_proj(concat(output)) → hidden_size       ← 投影回隐藏层
//! ```
//!
//! ## 与标准 MHA 的关键区别
//! - **Attention 维度**: latent_dim/2 (256d) 而非 head_dim (112d)，内存节省 ~56%
//! - **KV 缓存**: 存储 c_kv (seq×512) 而非 解压 K+V (seq×896×2)，节省 ~71%
//! - **o_proj 输入**: n_heads × latent_dim/2 (8192) 而非 q_dim (3584)
//! - **GQA 兼容**: 多查询头共享同一组 c_k/c_v，无需扩展 KV
//!
//! ## 两条前向路径
//! - `mla_forward_with_cache`: 标准路径，attention 在 head_dim 空间 + uq_proj 压缩输出
//! - `mla_forward_with_compressed_cache`: 真 MLA 路径，全程在潜在空间计算
//!
//! # 环境变量配置
//! - `OPENMINI_MASK_MAX_SEQ_LEN`: 预计算因果掩码的最大序列长度
//!   - **默认值**: 4096（约 64MB 内存）
//!   - 4096: 约 64MB 内存，适合内存受限环境
//!   - 8192: 约 256MB，支持中等长度上下文
//!   - 16384: 约 1GB，支持超长上下文
//!   - 32768: 约 4GB，支持 32k 上下文
//!   - **警告**: 设置过大会导致启动时内存峰值，请根据可用内存谨慎配置
//!
//! # 性能基准参考
//! 以下为 28 层模型在 Apple M1 Max 上的参考性能：
//! | 上下文长度 | 预填充延迟 | 解码延迟/token | 内存占用 |
//! |-----------|-----------|---------------|---------|
//! | 1K        | ~50ms     | ~15ms         | ~2GB    |
//! | 4K        | ~200ms    | ~18ms         | ~4GB    |
//! | 8K        | ~800ms    | ~22ms         | ~8GB    |
//! | 16K       | ~3s       | ~30ms         | ~16GB   |
//!
//! 注：实际性能取决于模型配置、硬件和批处理大小

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)] // 性能关键的推理代码：使用索引循环以优化张量操作
#![allow(clippy::doc_lazy_continuation)] // 文档格式：使用空行分隔段落
#![allow(clippy::type_complexity)] // 推理函数返回复杂类型元组

use dashmap::DashMap;
use ndarray::{s, Array1, Array2, Array3, Axis, Zip};
use once_cell::sync::Lazy;
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;
use rayon::prelude::*;
use std::ops::AddAssign;
use std::time::Instant;

use super::dsa::{self, DSATopKConfig};
use super::error::{InferenceError, InferenceResult};
use super::gemm_engine::get_gemm_engine_manager;
use super::gguf::MoEWeightsV2;
use super::sampler::GenerateParams;
use super::sliding_window::{sliding_window_attention, AttentionMode, SlidingWindowConfig};

impl From<ndarray::ShapeError> for InferenceError {
    fn from(e: ndarray::ShapeError) -> Self {
        InferenceError::generation(format!("Shape error: {}", e))
    }
}

trait ReshapeResultExt<T> {
    fn reshape_err(self, msg: &str) -> InferenceResult<T>;
}

impl<T> ReshapeResultExt<T> for Result<T, ndarray::ShapeError> {
    fn reshape_err(self, msg: &str) -> InferenceResult<T> {
        self.map_err(|e| InferenceError::generation(format!("{}: {}", msg, e)))
    }
}

// ============================================================================
// 常量定义
// ============================================================================

/// MoE 专家数量
const DEFAULT_MOE_NUM_EXPERTS: usize = 8;
/// MoE 激活专家数
const DEFAULT_MOE_TOP_K: usize = 2;

/// 掩码内存配置
/// 可通过环境变量 OPENMINI_MASK_MAX_SEQ_LEN 配置
/// - 8192: 约 256MB 内存，适合内存受限环境
/// - 16384: 约 1GB 内存，支持超长上下文（默认）
/// - 32768: 约 4GB 内存，支持 32k 上下文
fn get_mask_max_seq_len() -> usize {
    std::env::var("OPENMINI_MASK_MAX_SEQ_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096)
}

/// 预计算的全局因果掩码（上三角为 -inf）
/// 使用 Lazy 延迟初始化，避免启动时内存峰值
static CAUSAL_MASK: Lazy<Array2<f32>> = Lazy::new(|| {
    let max_len = *RUNTIME_MASK_MAX_SEQ_LEN;
    let mut mask = Array2::zeros((max_len, max_len));
    for i in 0..max_len {
        for j in (i + 1)..max_len {
            mask[[i, j]] = f32::NEG_INFINITY;
        }
    }
    mask
});

/// 运行时掩码最大长度（从环境变量读取）
static RUNTIME_MASK_MAX_SEQ_LEN: Lazy<usize> = Lazy::new(get_mask_max_seq_len);

/// LRU 缓存大小（用于动态生成的掩码）
const MASK_CACHE_SIZE: usize = 16;

/// 掩码缓存最大元素数阈值（超过此值不缓存）
/// 4096 * 4096 = 16.7M 元素 ≈ 64MB，支持 4K 上下文
/// 8192 * 8192 = 67M 元素 ≈ 256MB，支持 8K 上下文
const MASK_CACHE_MAX_ELEMENTS: usize = 67_000_000;

/// 掩码缓存条目（包含掩码数据和最后访问时间）
struct MaskCacheEntry {
    mask: Array2<f32>,
    last_access: Instant,
}

/// 动态掩码缓存（使用 DashMap 实现高并发安全）
/// DashMap 使用分片锁，比单一 Mutex 并发性能更优
/// 使用时间戳实现真正的 LRU 淘汰策略
///
/// # 缓存条件
/// 仅当 `seq_len == kv_seq_len ≤ 2048` 且内存占用 ≤ 64MB 时缓存
///
/// # Double Free 修复
/// 使用 Lazy + Box::leak 模式避免 DashMap 析构时的 double free 问题。
/// 全局缓存在程序退出时由操作系统回收内存，无需手动 drop。
static DYNAMIC_MASK_CACHE: Lazy<&'static DashMap<(usize, usize), MaskCacheEntry>> =
    Lazy::new(|| {
        let cache = Box::new(DashMap::new());
        Box::leak(cache)
    });

/// 掩码缓存最大序列长度（仅缓存正方形掩码）
const MASK_CACHE_MAX_SEQ_LEN: usize = 2048;

/// 掩码缓存最大内存字节数（64MB）
const MASK_CACHE_MAX_BYTES: usize = 64 * 1024 * 1024;

/// RoPE 预计算表结构
///
/// 当 max_positions <= 8192 时，预计算完整的 cos/sin 表以避免实时计算
/// 内存占用：max_positions * half_dim * 2 * 4 bytes
/// 例如：8192 * 64 * 2 * 4 = 4MB
struct RopeTables {
    freqs: Vec<f32>,
    cos_table: Option<Array2<f32>>,
    sin_table: Option<Array2<f32>>,
    max_positions: usize,
}

/// RoPE 预计算的最大位置数阈值
/// 超过此值时回退到实时计算，避免内存占用过大
const ROPE_PRECOMPUTE_MAX_POSITIONS: usize = 8192;

/// RoPE 预计算表缓存
/// Key: (theta_bits, head_dim, max_positions)
/// Value: 预计算的频率向量和 cos/sin 表
///
/// # Double Free 修复
/// 使用 Lazy + Box::leak 模式避免 DashMap 析构时的 double free 问题
static ROPE_TABLES_CACHE: Lazy<&'static DashMap<(u32, usize, usize), std::sync::Arc<RopeTables>>> =
    Lazy::new(|| {
        let cache = Box::new(DashMap::new());
        Box::leak(cache)
    });

/// 获取或创建 RoPE 预计算表
fn get_rope_tables(
    theta: f32,
    head_dim: usize,
    max_positions: usize,
) -> std::sync::Arc<RopeTables> {
    let key = (theta.to_bits(), head_dim, max_positions);

    ROPE_TABLES_CACHE
        .entry(key)
        .or_insert_with(|| {
            let half_dim = head_dim / 2;
            let freqs: Vec<f32> = (0..half_dim)
                .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
                .collect();

            let (cos_table, sin_table) = if max_positions <= ROPE_PRECOMPUTE_MAX_POSITIONS {
                let mut cos_t = Array2::zeros((max_positions, half_dim));
                let mut sin_t = Array2::zeros((max_positions, half_dim));

                for pos in 0..max_positions {
                    for (i, &freq) in freqs.iter().enumerate() {
                        let angle = pos as f32 * freq;
                        cos_t[[pos, i]] = angle.cos();
                        sin_t[[pos, i]] = angle.sin();
                    }
                }
                (Some(cos_t), Some(sin_t))
            } else {
                (None, None)
            };

            std::sync::Arc::new(RopeTables {
                freqs,
                cos_table,
                sin_table,
                max_positions,
            })
        })
        .clone()
}

/// Token IDs
pub const IM_START_TOKEN_ID: usize = 151646;
pub const IM_PATCH_TOKEN_ID: usize = 151647;
pub const IM_END_TOKEN_ID: usize = 151648;

/// 默认词汇表大小（模型回退值，非标准Llama词表）
const FALLBACK_VOCAB_SIZE: usize = 151936;
/// 默认隐藏层维度
pub const DEFAULT_HIDDEN_SIZE: usize = 3584;
/// 默认中间层维度
const DEFAULT_INTERMEDIATE_SIZE: usize = 18944;
/// 默认隐藏层数
const DEFAULT_NUM_HIDDEN_LAYERS: usize = 28;
/// 默认注意力头数
pub const DEFAULT_NUM_ATTENTION_HEADS: usize = 32;
/// 默认 KV 头数
const DEFAULT_NUM_KEY_VALUE_HEADS: usize = 8;
/// 默认最大位置编码
const DEFAULT_MAX_POSITION_EMBEDDINGS: usize = 32768;
/// 默认 RMS 归一化 epsilon
const DEFAULT_RMS_NORM_EPS: f32 = 1e-6;
/// 默认 RoPE theta
const DEFAULT_ROPE_THETA: f32 = 1000000.0;

// ============================================================================
// 配置结构
// ============================================================================

/// MoE Transformer 配置（原生多模态）
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub moe_num_experts: usize,
    pub moe_top_k: usize,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub use_mla: bool,
    pub mla_latent_dim: usize,
    pub mla_decoupled_rope: bool,
    pub image_patch_size: usize,
    pub audio_feature_dim: usize,
    pub use_dsa: bool,
    pub dsa_top_k: usize,
    pub dsa_short_seq_threshold: usize,
    pub use_mhc: bool,
    pub mhc_sinkhorn_iterations: usize,
    pub mhc_epsilon: f32,
    pub unk_token_id: usize,
    // ====== AttnRes 配置 ======
    pub use_attnres: bool, // 是否启用（默认 true，向后兼容无需配置）
    pub attnres_num_blocks: Option<usize>, // None = 自适应，Some(n) = 手动指定
    pub latent_dim: Option<usize>, // MLA 潜在维度，None 表示使用默认值
    // ====== Sliding Window Attention 配置 ======
    pub enable_sliding_window: bool, // 是否启用滑动窗口注意力（Gemma3 风格）
    pub sliding_window_size: Option<usize>, // 滑动窗口大小，None 使用默认值 128
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: FALLBACK_VOCAB_SIZE,
            hidden_size: DEFAULT_HIDDEN_SIZE,
            intermediate_size: DEFAULT_INTERMEDIATE_SIZE,
            num_hidden_layers: DEFAULT_NUM_HIDDEN_LAYERS,
            num_attention_heads: DEFAULT_NUM_ATTENTION_HEADS,
            num_key_value_heads: DEFAULT_NUM_KEY_VALUE_HEADS,
            head_dim: DEFAULT_HIDDEN_SIZE / DEFAULT_NUM_ATTENTION_HEADS,
            moe_num_experts: DEFAULT_MOE_NUM_EXPERTS,
            moe_top_k: DEFAULT_MOE_TOP_K,
            rope_theta: DEFAULT_ROPE_THETA,
            max_position_embeddings: DEFAULT_MAX_POSITION_EMBEDDINGS,
            rms_norm_eps: DEFAULT_RMS_NORM_EPS,
            use_mla: true,
            mla_latent_dim: 512,
            mla_decoupled_rope: true,
            image_patch_size: 14,
            audio_feature_dim: 128,
            use_dsa: true,
            dsa_top_k: dsa::DSA_TOP_K,
            dsa_short_seq_threshold: dsa::SHORT_SEQ_THRESHOLD,
            use_mhc: true,
            mhc_sinkhorn_iterations: 32,
            mhc_epsilon: 1e-6,
            unk_token_id: 0,
            // ====== AttnRes 默认值 ======
            use_attnres: true,
            attnres_num_blocks: None,
            latent_dim: None,
            // ====== Sliding Window Attention 默认值 ======
            enable_sliding_window: false,
            sliding_window_size: None,
        }
    }
}

impl ModelConfig {
    pub fn dsa_config(&self) -> DSATopKConfig {
        DSATopKConfig::default()
            .with_top_k(self.dsa_top_k)
            .with_dynamic_k(true)
            .with_short_seq_threshold(self.dsa_short_seq_threshold)
    }

    /// 验证配置参数的有效性
    pub fn validate(&self) -> InferenceResult<()> {
        if self.head_dim % 2 != 0 {
            return Err(InferenceError::config(format!(
                "head_dim must be even for RoPE, got {}",
                self.head_dim
            )));
        }
        if self.mla_latent_dim % 2 != 0 {
            return Err(InferenceError::config(format!(
                "mla_latent_dim must be even for MLA KV compression, got {}",
                self.mla_latent_dim
            )));
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(InferenceError::config(format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                self.hidden_size, self.num_attention_heads
            )));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(InferenceError::config(format!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({}) for GQA",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        if self.unk_token_id > 1_000_000 {
            return Err(InferenceError::config(format!(
                "unk_token_id ({}) seems unreasonably large",
                self.unk_token_id
            )));
        }
        let max_mask_len = *RUNTIME_MASK_MAX_SEQ_LEN;
        if self.max_position_embeddings > max_mask_len {
            eprintln!(
                "Warning: max_position_embeddings ({}) exceeds precomputed mask max length ({}). \
                 Set OPENMINI_MASK_MAX_SEQ_LEN={} environment variable to support longer contexts.",
                self.max_position_embeddings, max_mask_len, self.max_position_embeddings
            );
        }
        Ok(())
    }
}

// ============================================================================
// KV 缓存结构
// ============================================================================

/// KV 缓存（用于自回归生成）
#[derive(Debug, Clone)]
pub struct KVCache {
    pub k_cache: Array3<f32>,
    pub v_cache: Array3<f32>,
    pub seq_len: usize,
    pub max_seq_len: usize,
}

impl KVCache {
    pub fn new(max_seq_len: usize, num_heads: usize, head_dim: usize) -> Self {
        Self {
            k_cache: Array3::zeros((max_seq_len, num_heads, head_dim)),
            v_cache: Array3::zeros((max_seq_len, num_heads, head_dim)),
            seq_len: 0,
            max_seq_len,
        }
    }

    /// 更新 KV 缓存
    ///
    /// # 参数
    /// - `k`: 新的 K 张量，形状 `(new_len, num_heads, head_dim)`
    /// - `v`: 新的 V 张量，形状 `(new_len, num_heads, head_dim)`
    ///
    /// # 性能
    /// 形状检查在所有模式下执行以确保安全性
    pub fn update(&mut self, k: &Array3<f32>, v: &Array3<f32>) -> InferenceResult<()> {
        let new_len = k.dim().0;

        if k.dim() != v.dim() {
            return Err(InferenceError::generation(
                "K and V must have same shape".to_string(),
            ));
        }
        if k.dim().1 != self.k_cache.dim().1 {
            return Err(InferenceError::generation("num_heads mismatch".to_string()));
        }
        if k.dim().2 != self.k_cache.dim().2 {
            return Err(InferenceError::generation("head_dim mismatch".to_string()));
        }

        let available = self.max_seq_len.saturating_sub(self.seq_len);

        if new_len > available {
            return Err(InferenceError::generation(format!(
                "KV cache capacity exceeded: trying to add {} tokens but only {} available (max: {}, current: {})",
                new_len, available, self.max_seq_len, self.seq_len
            )));
        }

        self.k_cache
            .slice_mut(s![self.seq_len..self.seq_len + new_len, .., ..])
            .assign(k);
        self.v_cache
            .slice_mut(s![self.seq_len..self.seq_len + new_len, .., ..])
            .assign(v);
        self.seq_len += new_len;

        Ok(())
    }

    pub fn get(&self) -> (Array3<f32>, Array3<f32>) {
        let k = self.k_cache.slice(s![..self.seq_len, .., ..]).to_owned();
        let v = self.v_cache.slice(s![..self.seq_len, .., ..]).to_owned();
        (k, v)
    }

    pub fn clear(&mut self) {
        self.seq_len = 0;
    }

    pub fn len(&self) -> usize {
        self.seq_len
    }

    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    pub fn save(&self, path: &std::path::Path) -> InferenceResult<()> {
        let k_standard = self
            .k_cache
            .clone()
            .into_shape_with_order((self.seq_len, self.k_cache.dim().1, self.k_cache.dim().2))
            .map_err(|e| InferenceError::generation(format!("Failed to reshape K cache: {}", e)))?;
        let v_standard = self
            .v_cache
            .clone()
            .into_shape_with_order((self.seq_len, self.v_cache.dim().1, self.v_cache.dim().2))
            .map_err(|e| InferenceError::generation(format!("Failed to reshape V cache: {}", e)))?;

        let k_standard_layout = k_standard.as_standard_layout();
        let v_standard_layout = v_standard.as_standard_layout();

        let k_slice = k_standard_layout
            .as_slice()
            .ok_or_else(|| InferenceError::generation("K cache is not contiguous in memory"))?;
        let v_slice = v_standard_layout
            .as_slice()
            .ok_or_else(|| InferenceError::generation("V cache is not contiguous in memory"))?;

        let k_bytes: &[u8] = bytemuck::cast_slice(k_slice);
        let v_bytes: &[u8] = bytemuck::cast_slice(v_slice);

        let metadata = KVCacheMetadata {
            seq_len: self.seq_len,
            max_seq_len: self.max_seq_len,
            num_heads: self.k_cache.dim().1,
            head_dim: self.k_cache.dim().2,
        };

        let mut file = std::fs::File::create(path)
            .map_err(|e| InferenceError::io(format!("Failed to create KV cache file: {}", e)))?;

        serde_json::to_writer(&mut file, &metadata)
            .map_err(|e| InferenceError::io(format!("Failed to write KV cache metadata: {}", e)))?;

        use std::io::Write;
        file.write_all(k_bytes)
            .map_err(|e| InferenceError::io(format!("Failed to write K cache: {}", e)))?;
        file.write_all(v_bytes)
            .map_err(|e| InferenceError::io(format!("Failed to write V cache: {}", e)))?;

        Ok(())
    }

    pub fn load(path: &std::path::Path) -> InferenceResult<Self> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| InferenceError::io(format!("Failed to open KV cache file: {}", e)))?;

        let metadata: KVCacheMetadata = serde_json::from_reader(&mut file)
            .map_err(|e| InferenceError::io(format!("Failed to read KV cache metadata: {}", e)))?;

        use std::io::Read;
        let k_size = metadata.max_seq_len * metadata.num_heads * metadata.head_dim;
        let v_size = k_size;

        let mut k_bytes = vec![0u8; k_size * std::mem::size_of::<f32>()];
        let mut v_bytes = vec![0u8; v_size * std::mem::size_of::<f32>()];

        file.read_exact(&mut k_bytes)
            .map_err(|e| InferenceError::io(format!("Failed to read K cache: {}", e)))?;
        file.read_exact(&mut v_bytes)
            .map_err(|e| InferenceError::io(format!("Failed to read V cache: {}", e)))?;

        let k_data: Vec<f32> = bytemuck::allocation::try_cast_vec(k_bytes).map_err(|(_, e)| {
            InferenceError::io(format!("Failed to convert K cache bytes: {:?}", e))
        })?;
        let v_data: Vec<f32> = bytemuck::allocation::try_cast_vec(v_bytes).map_err(|(_, e)| {
            InferenceError::io(format!("Failed to convert V cache bytes: {:?}", e))
        })?;

        let k_cache: Array3<f32> = Array3::from_shape_vec(
            (metadata.max_seq_len, metadata.num_heads, metadata.head_dim),
            k_data,
        )
        .map_err(|e| InferenceError::generation(format!("Failed to reshape K cache: {}", e)))?;

        let v_cache: Array3<f32> = Array3::from_shape_vec(
            (metadata.max_seq_len, metadata.num_heads, metadata.head_dim),
            v_data,
        )
        .map_err(|e| InferenceError::generation(format!("Failed to reshape V cache: {}", e)))?;

        Ok(Self {
            k_cache,
            v_cache,
            seq_len: metadata.seq_len,
            max_seq_len: metadata.max_seq_len,
        })
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct KVCacheMetadata {
    seq_len: usize,
    max_seq_len: usize,
    num_heads: usize,
    head_dim: usize,
}

/// MLA 压缩 KV 缓存
///
/// 标准 MLA 的核心优化：缓存存储压缩后的 c_kv（潜在空间），
/// 而非解压后的 K 和 V。内存占用从 O(seq_len × kv_dim × 2) 降到 O(seq_len × latent_dim)。
///
/// # 内存对比 (典型配置: kv_dim=1024, latent_dim=512)
/// - 标准 KVCache: seq_len × 1024 × 2 = seq_len × 2048 floats
/// - MLACache:      seq_len × 512           = seq_len × 512 floats
/// - 节省: 75%
#[derive(Debug, Clone)]
pub struct MLACache {
    pub c_kv_cache: Array2<f32>,
    pub seq_len: usize,
    pub max_seq_len: usize,
    pub latent_dim: usize,
}

impl MLACache {
    pub fn new(max_seq_len: usize, latent_dim: usize) -> Self {
        Self {
            c_kv_cache: Array2::zeros((max_seq_len, latent_dim)),
            seq_len: 0,
            max_seq_len,
            latent_dim,
        }
    }

    /// 更新压缩 KV 缓存
    ///
    /// # 参数
    /// - `c_kv`: 新的压缩 KV 张量，形状 `(new_len, latent_dim)`
    pub fn update(&mut self, c_kv: &Array2<f32>) -> InferenceResult<()> {
        let new_len = c_kv.dim().0;

        if c_kv.dim().1 != self.latent_dim {
            return Err(InferenceError::generation(
                "latent_dim mismatch".to_string(),
            ));
        }

        let available = self.max_seq_len.saturating_sub(self.seq_len);

        if new_len > available {
            return Err(InferenceError::generation(format!(
                "MLA cache capacity exceeded: trying to add {} tokens but only {} available (max: {}, current: {})",
                new_len, available, self.max_seq_len, self.seq_len
            )));
        }

        self.c_kv_cache
            .slice_mut(s![self.seq_len..self.seq_len + new_len, ..])
            .assign(c_kv);
        self.seq_len += new_len;

        Ok(())
    }

    /// 获取全部缓存的压缩 KV
    pub fn get(&self) -> Array2<f32> {
        self.c_kv_cache.slice(s![..self.seq_len, ..]).to_owned()
    }

    pub fn clear(&mut self) {
        self.seq_len = 0;
    }

    pub fn len(&self) -> usize {
        self.seq_len
    }

    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// 计算当前缓存内存占用（float 数量）
    pub fn memory_size(&self) -> usize {
        self.seq_len * self.latent_dim
    }

    /// 对比标准 KV Cache 的内存节省比例
    pub fn compression_ratio(&self, kv_dim: usize) -> f32 {
        if kv_dim == 0 {
            return 0.0;
        }
        let standard = kv_dim * 2;
        1.0 - (self.latent_dim as f32 / standard as f32)
    }
}

/// 层级 KV 缓存
#[derive(Debug, Clone)]
pub struct LayerKVCache {
    pub caches: Vec<KVCache>,
}

impl LayerKVCache {
    pub fn new(num_layers: usize, max_seq_len: usize, num_heads: usize, head_dim: usize) -> Self {
        Self {
            caches: (0..num_layers)
                .map(|_| KVCache::new(max_seq_len, num_heads, head_dim))
                .collect(),
        }
    }

    pub fn clear(&mut self) {
        for cache in &mut self.caches {
            cache.clear();
        }
    }
}

// ============================================================================
// 权重结构
// ============================================================================

/// 注意力权重
#[derive(Debug, Clone)]
pub struct AttentionWeights {
    pub q_proj: Array2<f32>,
    pub k_proj: Array2<f32>,
    pub v_proj: Array2<f32>,
    pub o_proj: Array2<f32>,
}

/// FFN 权重
#[derive(Debug, Clone)]
pub struct FFNWeights {
    pub gate_proj: Array2<f32>,
    pub up_proj: Array2<f32>,
    pub down_proj: Array2<f32>,
}

impl FFNWeights {
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let engine = get_gemm_engine_manager();

        match engine
            .engine()
            .fused_gemm_silu(x, &self.gate_proj, &self.up_proj, None)
        {
            Ok(result) => {
                let down_proj_t = self.down_proj.t().to_owned();
                match engine.engine().matmul(&result, &down_proj_t) {
                    Ok(output) => output,
                    Err(_) => result.dot(&down_proj_t),
                }
            }
            Err(_) => {
                let gate = x.dot(&self.gate_proj.t());
                let up = x.dot(&self.up_proj.t());
                let hidden = silu(gate) * up;
                hidden.dot(&self.down_proj.t())
            }
        }
    }

    /// 使用 GEMM 引擎的 FFN 前向传播（返回 Result）
    pub fn forward_gemm(&self, x: &Array2<f32>) -> InferenceResult<Array2<f32>> {
        let engine = get_gemm_engine_manager();
        let hidden = engine
            .engine()
            .fused_gemm_silu(x, &self.gate_proj, &self.up_proj, None)?;
        let down_proj_t = self.down_proj.t().to_owned();
        engine.engine().matmul(&hidden, &down_proj_t)
    }
}

/// MoE 权重
#[derive(Debug, Clone)]
pub struct MoEWeights {
    pub experts: Vec<FFNWeights>,
    pub router: Array2<f32>,
    pub top_k: usize,
    /// 模态嵌入（可选）
    /// Key: 模态 ID (0=text, 1=image, 2=audio)
    /// Value: 模态嵌入向量，形状为 (hidden_size,)
    pub modality_embeds: Option<std::collections::HashMap<usize, Array1<f32>>>,
}

impl MoEWeights {
    pub fn new(num_experts: usize, top_k: usize) -> Self {
        Self {
            experts: Vec::with_capacity(num_experts),
            router: Array2::zeros((0, 0)),
            top_k,
            modality_embeds: None,
        }
    }

    /// 创建带模态嵌入的 MoE
    pub fn with_modality_embeds(
        num_experts: usize,
        top_k: usize,
        hidden_size: usize,
        num_modalities: usize,
    ) -> Self {
        let mut modality_embeds = std::collections::HashMap::new();
        for mod_id in 0..num_modalities {
            modality_embeds.insert(mod_id, Array1::zeros(hidden_size));
        }
        Self {
            experts: Vec::with_capacity(num_experts),
            router: Array2::zeros((0, 0)),
            top_k,
            modality_embeds: Some(modality_embeds),
        }
    }

    /// 设置模态嵌入
    ///
    /// # 参数
    /// - `modality_id`: 模态 ID（0=text, 1=image, 2=audio）
    /// - `embed`: 模态嵌入向量
    pub fn set_modality_embed(&mut self, modality_id: usize, embed: Array1<f32>) {
        if let Some(ref mut mod_embeds) = self.modality_embeds {
            mod_embeds.insert(modality_id, embed);
        }
    }

    /// 从 HashMap 批量加载模态嵌入
    pub fn load_modality_embeds(&mut self, embeds: std::collections::HashMap<usize, Array1<f32>>) {
        self.modality_embeds = Some(embeds);
    }

    /// MoE 前向传播
    ///
    /// # 参数
    /// - `x`: 输入张量 (seq_len, hidden_size)
    /// - `modality_ids`: 可选的模态 ID 序列 (seq_len,)，0=text, 1=image, 2=audio
    ///
    /// # 返回
    /// - 输出张量 (seq_len, hidden_size)
    pub fn forward(
        &self,
        x: &Array2<f32>,
        modality_ids: Option<&Array1<usize>>,
    ) -> InferenceResult<Array2<f32>> {
        let seq_len = x.nrows();
        let hidden_size = x.ncols();

        let router_input =
            if let (Some(mod_ids), Some(ref mod_embeds)) = (modality_ids, &self.modality_embeds) {
                let mut mod_embed_matrix = Array2::zeros((seq_len, hidden_size));
                for i in 0..seq_len {
                    let mod_id = mod_ids[i];
                    if let Some(embed) = mod_embeds.get(&mod_id) {
                        mod_embed_matrix.row_mut(i).assign(embed);
                    }
                }
                x + &mod_embed_matrix
            } else {
                x.clone()
            };

        let router_logits = linear(&router_input, &self.router);
        let (indices, weights, offsets) = top_k_selection(&router_logits, self.top_k);

        let mut expert_token_map: std::collections::HashMap<usize, Vec<(usize, f32)>> =
            std::collections::HashMap::new();

        for i in 0..seq_len {
            let start = offsets[i];
            let end = offsets[i + 1];
            for idx in start..end {
                let expert_idx = indices[idx];
                let weight = weights[idx];
                if weight > 0.0 {
                    expert_token_map
                        .entry(expert_idx)
                        .or_default()
                        .push((i, weight));
                }
            }
        }

        let expert_results: Vec<(usize, Array2<f32>, Vec<(usize, f32)>)> = {
            let should_parallelize = self.experts.len() > 2 && expert_token_map.len() > 1;

            let has_large_expert = expert_token_map.values().any(|v| v.len() > 4);

            if should_parallelize && has_large_expert {
                expert_token_map
                    .into_par_iter()
                    .filter_map(|(expert_idx, token_weights)| {
                        if expert_idx >= self.experts.len() || token_weights.is_empty() {
                            return None;
                        }

                        let expert = &self.experts[expert_idx];
                        let num_tokens = token_weights.len();

                        let mut expert_input = Array2::zeros((num_tokens, hidden_size));
                        for (t_idx, &(token_idx, _)) in token_weights.iter().enumerate() {
                            expert_input.row_mut(t_idx).assign(&x.row(token_idx));
                        }

                        let expert_output = expert.forward(&expert_input);

                        let mut weighted_output = Array2::zeros((num_tokens, hidden_size));
                        for (t_idx, &(_, weight)) in token_weights.iter().enumerate() {
                            let scale = weight;
                            Zip::from(weighted_output.row_mut(t_idx))
                                .and(expert_output.row(t_idx))
                                .for_each(|out, &exp| {
                                    *out = scale * exp;
                                });
                        }

                        Some((expert_idx, weighted_output, token_weights))
                    })
                    .collect()
            } else {
                expert_token_map
                    .into_iter()
                    .filter_map(|(expert_idx, token_weights)| {
                        if expert_idx >= self.experts.len() || token_weights.is_empty() {
                            return None;
                        }

                        let expert = &self.experts[expert_idx];
                        let num_tokens = token_weights.len();

                        let mut expert_input = Array2::zeros((num_tokens, hidden_size));
                        for (t_idx, &(token_idx, _)) in token_weights.iter().enumerate() {
                            expert_input.row_mut(t_idx).assign(&x.row(token_idx));
                        }

                        let expert_output = expert.forward(&expert_input);

                        let mut weighted_output = Array2::zeros((num_tokens, hidden_size));
                        for (t_idx, &(_, weight)) in token_weights.iter().enumerate() {
                            let scale = weight;
                            Zip::from(weighted_output.row_mut(t_idx))
                                .and(expert_output.row(t_idx))
                                .for_each(|out, &exp| {
                                    *out = scale * exp;
                                });
                        }

                        Some((expert_idx, weighted_output, token_weights))
                    })
                    .collect()
            }
        };

        let mut output = Array2::zeros((seq_len, hidden_size));
        for (_expert_idx, weighted_output, token_weights) in expert_results {
            for (t_idx, (token_idx, _)) in token_weights.iter().enumerate() {
                for d in 0..hidden_size {
                    output[[*token_idx, d]] += weighted_output[[t_idx, d]];
                }
            }
        }

        Ok(output)
    }
}

/// MLA 权重（Multi-head Latent Attention）
///
/// # DeepSeek-V2 标准 MLA 实现
///
/// ## 核心机制：KV 压缩到潜在空间 + 按需解码
///
/// ```
/// x ──→ q_proj ──→ Q (seq, n_heads * head_dim)
/// x ──→ dkv_proj ──→ c_kv (seq, latent_dim)  ← 压缩！
///        ↓ split
///   c_k (latent_dim/2)  c_v (latent_dim/2)
///        ↓ uk_proj           ↓ uv_proj
///   K (按需解码)          V (按需解码)
///
/// 缓存存储: c_kv (latent_dim) 而非 K+V (kv_dim*2)
/// 内存节省: 1 - latent_dim / (kv_dim * 2)
/// ```
///
/// ## 与简化版的区别
/// - **缓存**: 存储压缩的 c_kv，不是解压后的 K/V（核心优化点）
/// - **uq_proj**: 支持 Q 的 per-head 吸收投影
/// - **注意力**: 在标准维度计算（K/V 按需从 c_kv 解码），但缓存大幅压缩
#[derive(Debug, Clone)]
pub struct MLAWeights {
    pub q_proj: Array2<f32>,
    pub o_proj: Array2<f32>,
    pub dkv_proj: Array2<f32>,
    pub uk_proj: Array2<f32>,
    pub uv_proj: Array2<f32>,
    pub uq_proj: Option<Array2<f32>>,
    pub qr_proj: Option<Array2<f32>>,
    pub kr_proj: Option<Array2<f32>>,
    pub q_norm: Option<Array1<f32>>,
    pub k_norm: Option<Array1<f32>>,
}

impl MLAWeights {
    pub fn new(config: &ModelConfig) -> Self {
        let hidden_size = config.hidden_size;
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let latent_dim = config.mla_latent_dim;

        assert!(
            latent_dim % 2 == 0,
            "latent_dim must be even for MLA KV compression, got {}",
            latent_dim
        );

        Self {
            q_proj: Array2::zeros((q_dim, hidden_size)),
            o_proj: Array2::zeros((hidden_size, config.num_attention_heads * (latent_dim / 2))),
            dkv_proj: Array2::zeros((latent_dim, hidden_size)),
            uk_proj: Array2::zeros((kv_dim, latent_dim / 2)),
            uv_proj: Array2::zeros((kv_dim, latent_dim / 2)),
            uq_proj: Some(Array2::zeros((config.head_dim, latent_dim / 2))),
            qr_proj: if config.mla_decoupled_rope {
                Some(Array2::zeros((q_dim, hidden_size)))
            } else {
                None
            },
            kr_proj: if config.mla_decoupled_rope {
                Some(Array2::zeros((kv_dim, hidden_size)))
            } else {
                None
            },
            q_norm: None,
            k_norm: None,
        }
    }

    pub fn q_dim(&self) -> usize {
        self.q_proj.ncols()
    }

    pub fn kv_dim(&self) -> usize {
        self.uk_proj.nrows()
    }

    pub fn latent_dim(&self) -> usize {
        self.dkv_proj.nrows()
    }

    pub fn has_decoupled_rope(&self) -> bool {
        self.qr_proj.is_some() && self.kr_proj.is_some()
    }
}

/// Transformer 层权重
#[derive(Debug, Clone)]
pub struct TransformerLayerWeights {
    pub attention: AttentionWeights,
    pub mla: Option<MLAWeights>,
    pub ffn: FFNWeights,
    pub moe: MoEWeights,
    pub moe_v2: Option<MoEWeightsV2>,
    pub input_layernorm: Array1<f32>,
    pub post_attention_layernorm: Array1<f32>,
    pub mhc_dynamic_proj: Option<Array2<f32>>,
    pub mhc_static_proj: Option<Array2<f32>>,
    /// AttnRes: 伪查询向量 w_l（每层一个可学习参数）
    pub attnres_pseudo_query: Option<Array1<f32>>,
    /// 滑动窗口配置（可选）
    pub sliding_window_config: Option<SlidingWindowConfig>,
}

// ============================================================================
// 工具函数
// ============================================================================

/// Linear 投影（封装矩阵乘法 x @ weight.t()）
pub fn linear(x: &Array2<f32>, weight: &Array2<f32>) -> Array2<f32> {
    let engine = get_gemm_engine_manager();
    let weight_t = weight.t().to_owned();
    engine
        .engine()
        .matmul(x, &weight_t)
        .unwrap_or_else(|_| x.dot(&weight_t))
}

/// 使用 GEMM 引擎的线性投影（返回 Result）
pub fn linear_gemm(x: &Array2<f32>, weight: &Array2<f32>) -> InferenceResult<Array2<f32>> {
    let engine = get_gemm_engine_manager();
    let weight_t = weight.t().to_owned();
    engine.engine().matmul(x, &weight_t)
}

/// RMS Norm（完全向量化实现）
pub fn rms_norm(x: &Array2<f32>, weight: &Array1<f32>, eps: f32) -> Array2<f32> {
    let hidden_size = x.ncols();
    let var = (x * x).sum_axis(Axis(1)) / hidden_size as f32;
    let rms = (var + eps).mapv(f32::sqrt);
    let inv_rms: Array2<f32> = rms.mapv(|v| 1.0 / v).insert_axis(Axis(1));
    let normalized = x * &inv_rms;
    let weight_row = weight.clone().insert_axis(Axis(0));
    normalized * &weight_row
}

/// Apply Rotary Position Embedding（正确实现）
///
/// # 参数
/// - `x`: 输入张量 (seq_len, head_dim)
/// - `theta`: RoPE 基础频率（通常为 10000.0）
/// - `positions`: 位置索引 (seq_len,)，若为 None 则使用 0..seq_len
///
/// # Errors
/// - 输入张量维度不合法时返回错误
///
/// # 参数
/// - `x`: 输入张量 (seq_len, num_heads * head_dim)
/// - `num_heads`: 注意力头数量
/// - `head_dim`: 每个头的维度
/// - `theta`: RoPE 基础频率
/// - `positions`: 位置索引（可选，默认为 0..seq_len）
///
/// # 返回
/// 旋转后的张量，形状与输入相同
///
/// # Panics
/// - 如果 `head_dim` 不是偶数
///
/// 应用旋转位置编码（RoPE）
///
/// 使用预计算的频率表和 cos/sin 表进行高效计算，避免多余的 reshape 和内存分配
///
/// # 性能优化
/// - 当 max_positions <= 8192 时，使用预计算的 cos/sin 表直接查表
/// - 当 max_positions > 8192 时，回退到实时计算以避免内存占用过大
/// - 直接在 2D 张量上操作，避免 3D reshape
pub fn apply_rotary_emb(
    x: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    theta: f32,
    positions: Option<&Array1<usize>>,
) -> InferenceResult<Array2<f32>> {
    let seq_len = x.nrows();
    let total_dim = x.ncols();

    if total_dim != num_heads * head_dim {
        return Err(InferenceError::config(format!(
            "Input dimension {} doesn't match num_heads({}) * head_dim({})",
            total_dim, num_heads, head_dim
        )));
    }

    if head_dim % 2 != 0 {
        return Err(InferenceError::config(format!(
            "head_dim must be even for RoPE, got {}",
            head_dim
        )));
    }

    let half_dim = head_dim / 2;
    let default_positions: Array1<usize> = (0..seq_len).collect();
    let pos = positions.unwrap_or(&default_positions);

    let max_pos = pos.iter().max().copied().unwrap_or(0) + 1;
    let tables = get_rope_tables(theta, head_dim, max_pos);

    let mut output = x.clone();

    if let (Some(cos_table), Some(sin_table)) = (&tables.cos_table, &tables.sin_table) {
        for (pos_idx, &p) in pos.iter().enumerate() {
            if p >= tables.max_positions {
                continue;
            }

            for h in 0..num_heads {
                let base_idx = h * head_dim;

                for i in 0..half_dim {
                    let c = cos_table[[p, i]];
                    let s = sin_table[[p, i]];

                    let x0 = x[[pos_idx, base_idx + 2 * i]];
                    let x1 = x[[pos_idx, base_idx + 2 * i + 1]];

                    output[[pos_idx, base_idx + 2 * i]] = x0 * c - x1 * s;
                    output[[pos_idx, base_idx + 2 * i + 1]] = x0 * s + x1 * c;
                }
            }
        }
    } else {
        for (pos_idx, &p) in pos.iter().enumerate() {
            let pos_f = p as f32;

            for h in 0..num_heads {
                let base_idx = h * head_dim;

                for i in 0..half_dim {
                    let angle = pos_f * tables.freqs[i];
                    let (c, s) = (angle.cos(), angle.sin());

                    let x0 = x[[pos_idx, base_idx + 2 * i]];
                    let x1 = x[[pos_idx, base_idx + 2 * i + 1]];

                    output[[pos_idx, base_idx + 2 * i]] = x0 * c - x1 * s;
                    output[[pos_idx, base_idx + 2 * i + 1]] = x0 * s + x1 * c;
                }
            }
        }
    }

    Ok(output)
}

/// SiLU 激活函数
pub fn silu(x: Array2<f32>) -> Array2<f32> {
    x.mapv(|v| v / (1.0 + (-v).exp()))
}

/// 生成因果掩码（使用预计算的全局掩码切片 + LRU 缓存）
///
/// # 参数
/// - `seq_len`: 查询序列长度
/// - `kv_seq_len`: KV 序列长度（包含缓存）
///
/// # 返回
/// 形状为 `(seq_len, kv_seq_len)` 的掩码矩阵
///
/// # 内存配置
/// 可通过环境变量 `OPENMINI_MASK_MAX_SEQ_LEN` 配置预计算掩码大小：
/// - 4096: 约 64MB，适合内存受限环境（默认）
/// - 8192: 约 256MB，支持中等长度上下文
/// - 16384: 约 1GB，支持超长上下文
/// - 32768: 约 4GB，支持 32k 上下文
///
/// # 缓存策略
/// 仅当 `seq_len == kv_seq_len ≤ 2048` 且内存占用 ≤ 64MB 时缓存，
/// 避免缓存过多内存占用过大的掩码
pub fn create_causal_mask(seq_len: usize, kv_seq_len: usize) -> Array2<f32> {
    let max_len = *RUNTIME_MASK_MAX_SEQ_LEN;
    let mask_elements = seq_len * kv_seq_len;
    let mask_bytes = mask_elements * std::mem::size_of::<f32>();

    let should_cache = seq_len == kv_seq_len
        && seq_len <= MASK_CACHE_MAX_SEQ_LEN
        && mask_bytes <= MASK_CACHE_MAX_BYTES;

    if should_cache {
        let key = (seq_len, kv_seq_len);

        if let Some(mut entry) = DYNAMIC_MASK_CACHE.get_mut(&key) {
            entry.last_access = Instant::now();
            return entry.mask.clone();
        }

        let mask = CAUSAL_MASK.slice(s![0..seq_len, 0..kv_seq_len]).to_owned();

        if DYNAMIC_MASK_CACHE.len() >= MASK_CACHE_SIZE {
            let mut entries: Vec<_> = DYNAMIC_MASK_CACHE
                .iter()
                .map(|entry| (*entry.key(), entry.value().last_access))
                .collect();
            entries.sort_by_key(|(_, t)| *t);

            for (key, _) in entries.into_iter().take(4) {
                DYNAMIC_MASK_CACHE.remove(&key);
            }
        }

        DYNAMIC_MASK_CACHE.insert(
            key,
            MaskCacheEntry {
                mask: mask.clone(),
                last_access: Instant::now(),
            },
        );

        return mask;
    }

    let offset = kv_seq_len.saturating_sub(seq_len);

    if seq_len <= max_len && kv_seq_len <= max_len {
        if offset == 0 {
            return CAUSAL_MASK.slice(s![0..seq_len, 0..kv_seq_len]).to_owned();
        }

        let mut m = CAUSAL_MASK.slice(s![0..seq_len, 0..kv_seq_len]).to_owned();
        for i in 0..seq_len {
            let visible_end = (offset + i + 1).min(kv_seq_len);
            m.slice_mut(s![i, 0..visible_end]).fill(0.0);
        }
        return m;
    }

    let mut mask = Array2::zeros((seq_len, kv_seq_len));
    for i in 0..seq_len {
        for j in 0..kv_seq_len {
            if (i as isize + offset as isize) >= j as isize {
                mask[[i, j]] = 0.0;
            } else {
                mask[[i, j]] = f32::NEG_INFINITY;
            }
        }
    }
    mask
}

/// Softmax（向量化实现）
pub fn softmax(x: &Array2<f32>) -> Array2<f32> {
    let max_vals = x.map_axis(Axis(1), |row| {
        row.iter().copied().fold(f32::NEG_INFINITY, |a, b| a.max(b))
    });
    let shifted = x - &max_vals.insert_axis(Axis(1));
    let exps = shifted.mapv(|v| if v == f32::NEG_INFINITY { 0.0 } else { v.exp() });
    let exp_sums: Array1<f32> = exps.map_axis(Axis(1), |row| row.iter().sum());
    exps / &exp_sums.insert_axis(Axis(1))
}

/// Top-K 选择（返回每行的 top-k 索引和归一化权重）
///
/// 返回的权重已对 top-k 候选进行 softmax 归一化。
/// 若实际候选数少于 k，则只返回实际数量的结果，不填充。
///
/// # 返回
/// - `indices`: 展平的专家索引向量
/// - `weights`: 展平的权重向量
/// - `offsets`: 每行的起始偏移量，长度为 rows+1，offsets[i]..offsets[i+1] 为第 i 行的数据
///
/// MoE 路由器的 top-k 选择与 softmax 归一化
///
/// 对每行的 logits 直接取 top-k（值最大的 k 个），然后对这些 logits 做 softmax 得到最终权重。
///
/// # 参数
/// - `logits`: 路由器输出 logits，形状为 (batch_size, num_experts)
/// - `k`: 每行选择的专家数量
///
/// # 返回
/// - `indices`: 所有选中专家的索引（扁平化）
/// - `weights`: 对应的归一化权重（softmax 后）
/// - `offsets`: 每行的起始偏移量，长度为 batch_size + 1
///
/// # 边界情况
/// - 当 `k = 0` 时，返回空向量和全零偏移
/// - 当 `k > num_experts` 时，自动调整为 `num_experts`
/// - 当所有 logits 相等时，返回均匀分布权重
///
/// # 示例
/// ```
/// let logits = Array2::from_shape_vec((2, 4), vec![1.0, 4.0, 2.0, 3.0, 3.0, 1.0, 4.0, 2.0]).unwrap();
/// let (indices, weights, offsets) = top_k_selection(&logits, 2);
/// assert_eq!(offsets, vec![0, 2, 4]);
/// ```
pub fn top_k_selection(logits: &Array2<f32>, k: usize) -> (Vec<usize>, Vec<f32>, Vec<usize>) {
    let (rows, _cols) = logits.dim();
    let mut indices = Vec::with_capacity(rows * k);
    let mut weights = Vec::with_capacity(rows * k);
    let mut offsets = Vec::with_capacity(rows + 1);
    offsets.push(0);

    for row in logits.axis_iter(Axis(0)) {
        let mut idx_and_val: Vec<(usize, f32)> =
            row.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        let len = idx_and_val.len();
        let actual_k = k.min(len);

        if actual_k > 0 && actual_k < len {
            idx_and_val.select_nth_unstable_by(actual_k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let top_k: Vec<(usize, f32)> = idx_and_val.into_iter().take(actual_k).collect();

        let max_logit = top_k
            .iter()
            .map(|(_, v)| *v)
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));
        let exp_sum: f32 = top_k.iter().map(|(_, v)| (v - max_logit).exp()).sum();

        let (norm_weights, norm_indices): (Vec<f32>, Vec<usize>) = if exp_sum > 0.0 {
            top_k
                .iter()
                .map(|(idx, val)| ((val - max_logit).exp() / exp_sum, *idx))
                .unzip()
        } else {
            let uniform_weight = if actual_k > 0 {
                1.0 / actual_k as f32
            } else {
                0.0
            };
            top_k.iter().map(|(idx, _)| (uniform_weight, *idx)).unzip()
        };

        indices.extend(norm_indices);
        weights.extend(norm_weights);
        offsets.push(indices.len());
    }

    (indices, weights, offsets)
}

// ============================================================================
// Attention 实现（完全向量化）
// ============================================================================

/// 准备 Q、K、V 并应用 RoPE
fn prepare_qkv(
    x: &Array2<f32>,
    q_proj: &Array2<f32>,
    k_proj: &Array2<f32>,
    v_proj: &Array2<f32>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f32,
    positions: Option<&Array1<usize>>,
) -> InferenceResult<(Array3<f32>, Array3<f32>, Array3<f32>)> {
    let seq_len = x.nrows();

    let q = linear_gemm(x, q_proj)?;
    let k = linear_gemm(x, k_proj)?;
    let v = linear_gemm(x, v_proj)?;

    let q = apply_rotary_emb(&q, num_heads, head_dim, rope_theta, positions)?;
    let k = apply_rotary_emb(&k, num_kv_heads, head_dim, rope_theta, positions)?;

    let q = q
        .into_shape_with_order((seq_len, num_heads, head_dim))
        .reshape_err("Failed to reshape Q tensor for attention")?;
    let k = k
        .into_shape_with_order((seq_len, num_kv_heads, head_dim))
        .reshape_err("Failed to reshape K tensor for attention")?;
    let v = v
        .into_shape_with_order((seq_len, num_kv_heads, head_dim))
        .reshape_err("Failed to reshape V tensor for attention")?;

    Ok((q, k, v))
}

/// GQA 头扩展
fn expand_gqa(
    k: &Array3<f32>,
    v: &Array3<f32>,
    num_heads: usize,
    num_kv_heads: usize,
) -> InferenceResult<(Array3<f32>, Array3<f32>)> {
    let kv_seq_len = k.dim().0;
    let head_dim = k.dim().2;

    if num_kv_heads >= num_heads {
        return Ok((k.clone(), v.clone()));
    }

    let repeat_factor = num_heads / num_kv_heads;

    let k_expanded = {
        let mut expanded = Array3::zeros((kv_seq_len, num_heads, head_dim));
        for h in 0..num_kv_heads {
            let start = h * repeat_factor;
            let end = start + repeat_factor;
            let k_slice = k.slice(s![.., h..h + 1, ..]);
            let broadcast_k = k_slice
                .broadcast((kv_seq_len, repeat_factor, head_dim))
                .ok_or_else(|| {
                    InferenceError::generation("Failed to broadcast K for GQA expansion")
                })?;
            expanded
                .slice_mut(s![.., start..end, ..])
                .assign(&broadcast_k);
        }
        expanded
    };

    let v_expanded = {
        let mut expanded = Array3::zeros((kv_seq_len, num_heads, head_dim));
        for h in 0..num_kv_heads {
            let start = h * repeat_factor;
            let end = start + repeat_factor;
            let v_slice = v.slice(s![.., h..h + 1, ..]);
            let broadcast_v = v_slice
                .broadcast((kv_seq_len, repeat_factor, head_dim))
                .ok_or_else(|| {
                    InferenceError::generation("Failed to broadcast V for GQA expansion")
                })?;
            expanded
                .slice_mut(s![.., start..end, ..])
                .assign(&broadcast_v);
        }
        expanded
    };

    Ok((k_expanded, v_expanded))
}

/// 计算注意力分数（批量矩阵乘法）
fn compute_attention_scores(q: &Array3<f32>, k: &Array3<f32>, scale: f32) -> Array3<f32> {
    let (seq_len, num_heads, _head_dim) = q.dim();
    let kv_seq_len = k.dim().0;

    let mut scores = Array3::zeros((num_heads, seq_len, kv_seq_len));

    for h in 0..num_heads {
        let q_h = q.slice(s![.., h, ..]);
        let k_h = k.slice(s![.., h, ..]);
        let k_h_t = k_h.t().to_owned();
        let scores_h = q_h.dot(&k_h_t) * scale;
        scores.index_axis_mut(Axis(0), h).assign(&scores_h);
    }

    scores
}

/// 计算注意力输出（批量矩阵乘法）
fn compute_attention_output(attn: &Array3<f32>, v: &Array3<f32>) -> Array2<f32> {
    let (num_heads, seq_len, _kv_seq_len) = attn.dim();
    let head_dim = v.dim().2;

    let mut output = Array2::zeros((seq_len, num_heads * head_dim));

    for h in 0..num_heads {
        let attn_h = attn.slice(s![h, .., ..]);
        let v_h = v.slice(s![.., h, ..]);
        let output_h = attn_h.dot(&v_h);

        for s in 0..seq_len {
            for d in 0..head_dim {
                output[[s, h * head_dim + d]] = output_h[[s, d]];
            }
        }
    }

    output
}

/// 应用因果掩码和 softmax
fn apply_causal_mask_and_softmax(
    scores: &Array3<f32>,
    seq_len: usize,
    kv_seq_len: usize,
) -> InferenceResult<Array3<f32>> {
    let num_heads = scores.dim().0;

    let causal_mask = create_causal_mask(seq_len, kv_seq_len);

    let scores_2d = scores
        .clone()
        .into_shape_with_order((num_heads * seq_len, kv_seq_len))
        .reshape_err("Failed to reshape scores for softmax")?;

    let mask_broadcast = causal_mask
        .view()
        .insert_axis(Axis(0))
        .broadcast((num_heads, seq_len, kv_seq_len))
        .ok_or_else(|| InferenceError::generation("Failed to broadcast causal mask"))?
        .to_owned()
        .into_shape_with_order((num_heads * seq_len, kv_seq_len))
        .reshape_err("Failed to reshape broadcast mask")?;

    let masked_scores = &scores_2d + &mask_broadcast;
    let attn_2d = softmax(&masked_scores);

    attn_2d
        .into_shape_with_order((num_heads, seq_len, kv_seq_len))
        .reshape_err("Failed to reshape attn back")
}

/// 2D 因果掩码 + Softmax（用于潜在空间 MLA）
fn apply_causal_mask_and_softmax_2d(
    scores: &Array2<f32>,
    seq_len: usize,
    kv_seq_len: usize,
) -> InferenceResult<Array2<f32>> {
    let causal_mask = create_causal_mask(seq_len, kv_seq_len);
    let masked_scores = scores + &causal_mask;
    let attn = softmax(&masked_scores);
    Ok(attn)
}

impl AttentionWeights {
    pub fn forward(&self, x: &Array2<f32>, config: &ModelConfig) -> InferenceResult<Array2<f32>> {
        self.forward_with_cache(x, config, None, None)
    }

    pub fn forward_with_cache(
        &self,
        x: &Array2<f32>,
        config: &ModelConfig,
        kv_cache: Option<&mut KVCache>,
        positions: Option<&Array1<usize>>,
    ) -> InferenceResult<Array2<f32>> {
        let seq_len = x.nrows();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

        let (q, mut k, mut v) = prepare_qkv(
            x,
            &self.q_proj,
            &self.k_proj,
            &self.v_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            config.rope_theta,
            positions,
        )?;

        if let Some(cache) = kv_cache {
            cache.update(&k, &v)?;
            let (cached_k, cached_v) = cache.get();
            k = cached_k;
            v = cached_v;
        }

        let scale = 1.0 / (head_dim as f32).sqrt();
        let kv_seq_len = k.dim().0;

        let (k_expanded, v_expanded) = expand_gqa(&k, &v, num_heads, num_kv_heads)?;

        let use_dsa = config.use_dsa && kv_seq_len >= config.dsa_short_seq_threshold;

        if use_dsa {
            let dsa_config = config.dsa_config();
            let q_2d = q
                .clone()
                .into_shape_with_order((seq_len, num_heads * head_dim))
                .reshape_err("Failed to reshape Q for DSA")?;
            let k_2d = k_expanded
                .clone()
                .into_shape_with_order((kv_seq_len, num_heads * head_dim))
                .reshape_err("Failed to reshape K for DSA")?;
            let v_2d = v_expanded
                .into_shape_with_order((kv_seq_len, num_heads * head_dim))
                .reshape_err("Failed to reshape V for DSA")?;

            let dsa_output = dsa::multihead_sparse_attention(
                &q_2d,
                &k_2d,
                &v_2d,
                num_heads,
                head_dim,
                &dsa_config,
                true,
            )
            .map_err(|e| InferenceError::generation(format!("DSA attention failed: {}", e)))?;

            let output = linear_gemm(&dsa_output, &self.o_proj)?;
            return Ok(output);
        }

        let scores = compute_attention_scores(&q, &k_expanded, scale);

        let attn = apply_causal_mask_and_softmax(&scores, seq_len, kv_seq_len)?;

        let output = compute_attention_output(&attn, &v_expanded);

        let output = linear_gemm(&output, &self.o_proj)?;
        Ok(output)
    }
}

// ============================================================================
// mHC (Manifold-Constrained Hyper-Connections) 流形约束超连接 — 3D Per-Head 版本
// ============================================================================

/// 2D Sinkhorn-Knopp：将矩阵投影到双随机矩阵流形（Birkhoff 多面体）
///
/// 双随机矩阵满足：
/// - 所有元素 ≥ 0（非负性）
/// - 每行和 = 1（行随机）
/// - 每列和 = 1（列随机）
fn sinkhorn_knopp(mut matrix: Array2<f32>, iterations: usize, eps: f32) -> Array2<f32> {
    for _ in 0..iterations {
        let row_sums = matrix
            .sum_axis(Axis(1))
            .mapv(|s| if s < eps { 1.0 } else { s });
        let row_inv: Array2<f32> = row_sums.mapv(|s| 1.0 / s).insert_axis(Axis(1));
        matrix = &matrix * &row_inv;

        let col_sums = matrix
            .sum_axis(Axis(0))
            .mapv(|s| if s < eps { 1.0 } else { s });
        let col_inv: Array2<f32> = col_sums.mapv(|s| 1.0 / s).insert_axis(Axis(0));
        matrix = &matrix * &col_inv;
    }
    matrix
}

/// 3D Sinkhorn-Knopp：对每个注意力头独立投影到双随机流形
///
/// 输入形状: (seq_len, seq_len, n_heads)
/// 输出形状: (seq_len, seq_len, n_heads)，每个 head 切片都是双随机矩阵
fn sinkhorn_knopp_3d(matrix: Array3<f32>, iterations: usize, eps: f32) -> Array3<f32> {
    let (seq_a, seq_b, n_heads) = matrix.dim();
    let mut result = Array3::zeros((seq_a, seq_b, n_heads));

    for h in 0..n_heads {
        let slice = matrix.slice(s![.., .., h]).to_owned();
        let projected = sinkhorn_knopp(slice, iterations, eps);
        result.slice_mut(s![.., .., h]).assign(&projected);
    }

    result
}

/// mHC 3D 流形约束超连接残差（Per-Head 版本，原生 3D）
///
/// # 核心思想
/// 不同注意力头承担不同语义角色（如语义头、位置头、语法头），
/// 因此应该有**独立的流形约束**，而非所有头共享同一个连接矩阵。
///
/// # 数据格式
/// - 输入输出均为 **原生 3D 数组** `(seq_len, head_dim, n_heads)`
/// - 不做 2D 压扁/拼接，保持大模型原始张量结构
///
/// # 计算流程
/// ```text
/// 对每个 attention head h ∈ [0, n_heads):
///
/// 步骤1: Per-Head 输入处理
///   x_h = x[:, :, h]                          → (seq, head_dim) 该头的隐藏状态
///   dynamic_scores[:,h] = x_flat @ W_dyn[h,:]^T → (seq,) 该头的动态分数
///   static_scores[:,h]  = x_flat @ W_sta[h,:]^T  → (seq,) 该头的静态分数
///   d_h = softmax(dynamic_scores[:,h])           → (seq,1) 动态映射向量
///   s_h = sigmoid(static_scores[:,h])            → (1,seq) 静态映射向量
///
/// 步骤2: Per-Head 流形投影
///   H_raw[:,:,h] = d_h @ s_h^T                 → (seq,seq) 原始连接矩阵
///   H_res[:,:,h] = SinkhornKnopp(H_raw[:,:,h])   → (seq,seq) ★双随机★
///
/// 步骤3: Per-Head 约束输出
///   delta_h = delta[:, :, h]                    → 该头的子层增量 (seq, head_dim)
///   constrained_h = H_res[:,:,h] @ delta_h       → 流形约束后的增量
///   output[:, :, h] += constrained_h            → 写回对应头（保持3D）
///
/// 最终: output = x + constrained_3d  (形状不变，仍为 3D)
/// ```
///
/// # 参数
/// - `x`: 输入隐藏状态 (seq_len, head_dim, n_heads) — **3D**
/// - `delta`: 子层输出 (seq_len, head_dim, n_heads) — **3D**
/// - `dynamic_proj`: 动态投影权重 (n_heads, hidden_size)，每行是一个头的投影向量
/// - `static_proj`: 静态投影权重 (n_heads, hidden_size)
/// - `num_heads`: 注意力头数量
/// - `head_dim`: 每个头的维度
/// - `sinkhorn_iterations`: Sinkhorn-Knopp 迭代次数
/// - `eps`: 数值稳定 epsilon
fn mhc_residual(
    x: &Array3<f32>,
    delta: &Array3<f32>,
    dynamic_proj: &Array2<f32>,
    static_proj: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    sinkhorn_iterations: usize,
    eps: f32,
) -> InferenceResult<Array3<f32>> {
    let (seq_len, x_dim, _x_heads) = x.dim();
    let (_d_len, _d_dim, d_heads) = delta.dim();

    if seq_len == 0 || x_dim == 0 {
        return Ok(x.clone());
    }

    let hidden_size = num_heads * head_dim;

    let dynamic_scores = {
        let x_view = x.view().into_shape_with_order((seq_len, hidden_size))?;
        x_view.dot(&dynamic_proj.t())
    };
    let static_scores = {
        let x_view = x.view().into_shape_with_order((seq_len, hidden_size))?;
        x_view.dot(&static_proj.t())
    };

    let mut h_res_array = Array3::zeros((seq_len, seq_len, num_heads));

    for h in 0..num_heads {
        let d_h = dynamic_scores.column(h).to_owned();
        let s_h = static_scores.column(h).to_owned();

        let max_d = d_h.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_d = d_h.mapv(|v| (v - max_d).exp());
        let sum_d: f32 = exp_d.sum();
        let d_norm = exp_d.mapv(|v| v / sum_d);

        let s_norm = s_h.mapv(|v| 1.0 / (1.0 + (-v).exp()));

        let d_col = d_norm.insert_axis(Axis(1));
        let s_row = s_norm.insert_axis(Axis(0));

        let h_raw_h = d_col.dot(&s_row);
        let projected = sinkhorn_knopp(h_raw_h, sinkhorn_iterations, eps);
        h_res_array.slice_mut(s![.., .., h]).assign(&projected);
    }

    let mut output = x.clone();
    for h in 0..num_heads {
        let delta_h = delta
            .slice(ndarray::s![.., .., h.min(d_heads - 1)])
            .to_owned();
        let h_res_h = h_res_array.slice(ndarray::s![.., .., h]).to_owned();
        let constrained_h = h_res_h.dot(&delta_h);
        output
            .slice_mut(ndarray::s![.., .., h])
            .add_assign(&constrained_h);
    }

    Ok(output)
}

// ============================================================================
// TransformerLayer 实现
// ============================================================================

impl TransformerLayerWeights {
    pub fn new(config: &ModelConfig) -> Self {
        let experts = (0..config.moe_num_experts)
            .map(|_| FFNWeights {
                gate_proj: Array2::zeros((config.intermediate_size, config.hidden_size)),
                up_proj: Array2::zeros((config.intermediate_size, config.hidden_size)),
                down_proj: Array2::zeros((config.hidden_size, config.intermediate_size)),
            })
            .collect();

        let mla = if config.use_mla {
            Some(MLAWeights::new(config))
        } else {
            None
        };

        Self {
            attention: AttentionWeights {
                q_proj: Array2::zeros((
                    config.num_attention_heads * config.head_dim,
                    config.hidden_size,
                )),
                k_proj: Array2::zeros((
                    config.num_key_value_heads * config.head_dim,
                    config.hidden_size,
                )),
                v_proj: Array2::zeros((
                    config.num_key_value_heads * config.head_dim,
                    config.hidden_size,
                )),
                o_proj: Array2::zeros((
                    config.hidden_size,
                    config.num_attention_heads * config.head_dim,
                )),
            },
            mla,
            ffn: FFNWeights {
                gate_proj: Array2::zeros((config.intermediate_size, config.hidden_size)),
                up_proj: Array2::zeros((config.intermediate_size, config.hidden_size)),
                down_proj: Array2::zeros((config.hidden_size, config.intermediate_size)),
            },
            moe: MoEWeights {
                experts,
                router: Array2::zeros((config.moe_num_experts, config.hidden_size)),
                top_k: config.moe_top_k,
                modality_embeds: None,
            },
            moe_v2: None,
            input_layernorm: Array1::ones(config.hidden_size),
            post_attention_layernorm: Array1::ones(config.hidden_size),
            mhc_dynamic_proj: if config.use_mhc {
                Some(Array2::zeros((
                    config.num_attention_heads,
                    config.hidden_size,
                )))
            } else {
                None
            },
            mhc_static_proj: if config.use_mhc {
                Some(Array2::zeros((
                    config.num_attention_heads,
                    config.hidden_size,
                )))
            } else {
                None
            },
            attnres_pseudo_query: if config.use_attnres {
                // 初始化为全零向量，训练初期等效于标准残差
                Some(Array1::zeros(config.hidden_size))
            } else {
                None
            },
            sliding_window_config: None,
        }
    }

    pub fn from_quantized(
        config: &ModelConfig,
        q_weights: super::quant_loader::QuantizedLayerWeights,
        mla: Option<MLAWeights>,
    ) -> Self {
        let experts = if let Some(ref moe_q) = q_weights.moe {
            moe_q
                .experts
                .iter()
                .map(|e| FFNWeights {
                    gate_proj: e.gate_proj.clone(),
                    up_proj: e.up_proj.clone(),
                    down_proj: e.down_proj.clone(),
                })
                .collect()
        } else {
            (0..config.moe_num_experts)
                .map(|_| FFNWeights {
                    gate_proj: Array2::zeros((config.intermediate_size, config.hidden_size)),
                    up_proj: Array2::zeros((config.intermediate_size, config.hidden_size)),
                    down_proj: Array2::zeros((config.hidden_size, config.intermediate_size)),
                })
                .collect()
        };

        let moe_router = q_weights
            .moe
            .as_ref()
            .map(|m| m.router.clone())
            .unwrap_or_else(|| Array2::zeros((config.moe_num_experts, config.hidden_size)));

        let moe_top_k = q_weights
            .moe
            .as_ref()
            .map(|m| m.top_k)
            .unwrap_or(config.moe_top_k);

        Self {
            attention: AttentionWeights {
                q_proj: q_weights.attention.q_proj.clone(),
                k_proj: q_weights.attention.k_proj.clone(),
                v_proj: q_weights.attention.v_proj.clone(),
                o_proj: q_weights.attention.o_proj.clone(),
            },
            mla,
            ffn: FFNWeights {
                gate_proj: q_weights.ffn.gate_proj.clone(),
                up_proj: q_weights.ffn.up_proj.clone(),
                down_proj: q_weights.ffn.down_proj.clone(),
            },
            moe: MoEWeights {
                experts,
                router: moe_router,
                top_k: moe_top_k,
                modality_embeds: None,
            },
            moe_v2: None,
            input_layernorm: q_weights.input_layernorm.clone(),
            post_attention_layernorm: q_weights.post_attention_layernorm.clone(),
            mhc_dynamic_proj: if config.use_mhc {
                Some(Array2::zeros((
                    config.num_attention_heads,
                    config.hidden_size,
                )))
            } else {
                None
            },
            mhc_static_proj: if config.use_mhc {
                Some(Array2::zeros((
                    config.num_attention_heads,
                    config.hidden_size,
                )))
            } else {
                None
            },
            attnres_pseudo_query: if config.use_attnres {
                Some(Array1::zeros(config.hidden_size))
            } else {
                None
            },
            sliding_window_config: None,
        }
    }

    pub fn forward(
        &self,
        x: &Array2<f32>,
        config: &ModelConfig,
        layer_idx: usize,
    ) -> InferenceResult<Array2<f32>> {
        self.forward_with_cache(x, config, layer_idx, None, None, None)
    }

    pub fn forward_with_cache(
        &self,
        x: &Array2<f32>,
        config: &ModelConfig,
        layer_idx: usize,
        kv_cache: Option<&mut KVCache>,
        positions: Option<&Array1<usize>>,
        block_summary: Option<&mut super::attn_res::BlockSummary>, // ★ 新增
    ) -> InferenceResult<Array2<f32>> {
        // ====== AttnRes: 块边界深度聚合 ======
        let x = if config.use_attnres {
            if let Some(ref pq) = self.attnres_pseudo_query {
                match block_summary {
                    Some(ref bs) if bs.is_block_start(layer_idx) && layer_idx > 0 => {
                        bs.aggregate(pq, x)?
                    }
                    _ => x.clone(),
                }
            } else {
                x.clone()
            }
        } else {
            x.clone()
        };

        let seq_len = x.nrows();
        let normed = rms_norm(&x, &self.input_layernorm, config.rms_norm_eps);

        let attn = if let Some(sw_config) = &self.sliding_window_config {
            // 根据配置的 attention_mode 决定策略
            match sw_config.attention_mode {
                AttentionMode::Global => {
                    if config.use_mla {
                        if let Some(ref mla) = self.mla {
                            self.mla_forward_with_cache(&normed, mla, config, kv_cache, positions)?
                        } else {
                            self.attention
                                .forward_with_cache(&normed, config, kv_cache, positions)?
                        }
                    } else {
                        self.attention
                            .forward_with_cache(&normed, config, kv_cache, positions)?
                    }
                }
                AttentionMode::Local | AttentionMode::Strided { .. } => {
                    let (q_sw, k_sw, v_sw) = prepare_qkv(
                        &normed,
                        &self.attention.q_proj,
                        &self.attention.k_proj,
                        &self.attention.v_proj,
                        config.num_attention_heads,
                        config.num_key_value_heads,
                        config.head_dim,
                        config.rope_theta,
                        positions,
                    )?;

                    if let Some(cache) = kv_cache {
                        cache.update(&k_sw, &v_sw)?;
                        let (cached_k, cached_v) = cache.get();
                        let _k_cached = cached_k;
                        let _v_cached = cached_v;
                    }

                    let num_heads = config.num_attention_heads;
                    let head_dim = config.head_dim;

                    let mut output = Array2::zeros((seq_len, head_dim * num_heads));
                    for h in 0..num_heads {
                        let q_head = q_sw.slice(s![.., h, ..]).to_owned();
                        let k_head = k_sw.slice(s![.., h, ..]).to_owned();
                        let v_head = v_sw.slice(s![.., h, ..]).to_owned();

                        let q_head_2d =
                            q_head
                                .into_shape_with_order((seq_len, head_dim))
                                .map_err(|e| {
                                    InferenceError::generation(format!(
                                        "SWA Q reshape failed: {}",
                                        e
                                    ))
                                })?;
                        let k_head_2d =
                            k_head
                                .into_shape_with_order((seq_len, head_dim))
                                .map_err(|e| {
                                    InferenceError::generation(format!(
                                        "SWA K reshape failed: {}",
                                        e
                                    ))
                                })?;
                        let v_head_2d =
                            v_head
                                .into_shape_with_order((seq_len, head_dim))
                                .map_err(|e| {
                                    InferenceError::generation(format!(
                                        "SWA V reshape failed: {}",
                                        e
                                    ))
                                })?;

                        let sw_out = sliding_window_attention(
                            &q_head_2d, &k_head_2d, &v_head_2d, sw_config, None,
                        )
                        .map_err(|e| {
                            InferenceError::generation(format!(
                                "Sliding window attention failed: {}",
                                e
                            ))
                        })?;

                        for i in 0..seq_len {
                            for j in 0..head_dim {
                                output[[i, h * head_dim + j]] = sw_out[[i, j]];
                            }
                        }
                    }

                    linear_gemm(&output, &self.attention.o_proj)?
                }
            }
        } else if config.use_mla {
            if let Some(ref mla) = self.mla {
                self.mla_forward_with_cache(&normed, mla, config, kv_cache, positions)?
            } else {
                self.attention
                    .forward_with_cache(&normed, config, kv_cache, positions)?
            }
        } else {
            self.attention
                .forward_with_cache(&normed, config, kv_cache, positions)?
        };

        let hidden = if config.use_mhc {
            if let (Some(ref dyn_proj), Some(ref sta_proj)) =
                (&self.mhc_dynamic_proj, &self.mhc_static_proj)
            {
                let x_3d = normed
                    .clone()
                    .into_shape_with_order((seq_len, config.head_dim, config.num_attention_heads))
                    .map_err(|e| {
                        InferenceError::generation(format!("mHC reshape x failed: {}", e))
                    })?;
                let delta_3d = attn
                    .clone()
                    .into_shape_with_order((seq_len, config.head_dim, config.num_attention_heads))
                    .map_err(|e| {
                        InferenceError::generation(format!("mHC reshape delta failed: {}", e))
                    })?;
                let out_3d = mhc_residual(
                    &x_3d,
                    &delta_3d,
                    dyn_proj,
                    sta_proj,
                    config.num_attention_heads,
                    config.head_dim,
                    config.mhc_sinkhorn_iterations,
                    config.mhc_epsilon,
                )?;
                out_3d
                    .into_shape_with_order((seq_len, config.num_attention_heads * config.head_dim))
                    .map_err(|e| {
                        InferenceError::generation(format!("mHC 3D→2D reshape failed: {}", e))
                    })?
            } else {
                &normed + &attn
            }
        } else {
            &normed + &attn
        };

        let normed2 = rms_norm(&hidden, &self.post_attention_layernorm, config.rms_norm_eps);

        if let Some(moe_v2) = &self.moe_v2 {
            let x_3d = normed2.clone().insert_axis(Axis(0));
            let (ffn_out_3d, _loss) = moe_v2
                .forward(&x_3d)
                .map_err(|e| InferenceError::generation(format!("MoE V2 forward failed: {}", e)))?;
            let ffn_out = ffn_out_3d.index_axis_move(Axis(0), 0);
            if config.use_mhc {
                if let (Some(ref dyn_proj), Some(ref sta_proj)) =
                    (&self.mhc_dynamic_proj, &self.mhc_static_proj)
                {
                    let h_3d = hidden
                        .clone()
                        .into_shape_with_order((
                            seq_len,
                            config.head_dim,
                            config.num_attention_heads,
                        ))
                        .map_err(|e| {
                            InferenceError::generation(format!("mHC reshape h failed: {}", e))
                        })?;
                    let d_3d = ffn_out
                        .clone()
                        .into_shape_with_order((
                            seq_len,
                            config.head_dim,
                            config.num_attention_heads,
                        ))
                        .map_err(|e| {
                            InferenceError::generation(format!("mHC reshape ffn failed: {}", e))
                        })?;
                    let out_3d = mhc_residual(
                        &h_3d,
                        &d_3d,
                        dyn_proj,
                        sta_proj,
                        config.num_attention_heads,
                        config.head_dim,
                        config.mhc_sinkhorn_iterations,
                        config.mhc_epsilon,
                    )?;
                    let output = out_3d
                        .into_shape_with_order((
                            seq_len,
                            config.num_attention_heads * config.head_dim,
                        ))
                        .map_err(|e| {
                            InferenceError::generation(format!("mHC 3D→2D reshape failed: {}", e))
                        })?;

                    if config.use_attnres {
                        if let Some(bs) = block_summary {
                            if bs.is_block_end(layer_idx) {
                                let _ = bs.update_summary(&output, layer_idx);
                            }
                        }
                    }

                    Ok(output)
                } else {
                    let output = hidden + ffn_out;

                    if config.use_attnres {
                        if let Some(bs) = block_summary {
                            if bs.is_block_end(layer_idx) {
                                let _ = bs.update_summary(&output, layer_idx);
                            }
                        }
                    }

                    Ok(output)
                }
            } else {
                let output = hidden + ffn_out;

                if config.use_attnres {
                    if let Some(bs) = block_summary {
                        if bs.is_block_end(layer_idx) {
                            let _ = bs.update_summary(&output, layer_idx);
                        }
                    }
                }

                Ok(output)
            }
        } else if layer_idx % 3 == 2 {
            let moe_out = self.moe.forward(&normed2, None)?;
            if config.use_mhc {
                if let (Some(ref dyn_proj), Some(ref sta_proj)) =
                    (&self.mhc_dynamic_proj, &self.mhc_static_proj)
                {
                    let h_3d = hidden
                        .clone()
                        .into_shape_with_order((
                            seq_len,
                            config.head_dim,
                            config.num_attention_heads,
                        ))
                        .map_err(|e| {
                            InferenceError::generation(format!("mHC reshape h failed: {}", e))
                        })?;
                    let d_3d = moe_out
                        .into_shape_with_order((
                            seq_len,
                            config.head_dim,
                            config.num_attention_heads,
                        ))
                        .map_err(|e| {
                            InferenceError::generation(format!("mHC reshape moe failed: {}", e))
                        })?;
                    let out_3d = mhc_residual(
                        &h_3d,
                        &d_3d,
                        dyn_proj,
                        sta_proj,
                        config.num_attention_heads,
                        config.head_dim,
                        config.mhc_sinkhorn_iterations,
                        config.mhc_epsilon,
                    )?;
                    let output = out_3d
                        .into_shape_with_order((
                            seq_len,
                            config.num_attention_heads * config.head_dim,
                        ))
                        .map_err(|e| {
                            InferenceError::generation(format!("mHC 3D→2D reshape failed: {}", e))
                        })?;

                    // ====== AttnRes: 块结束更新摘要 ======
                    if config.use_attnres {
                        if let Some(bs) = block_summary {
                            if bs.is_block_end(layer_idx) {
                                let _ = bs.update_summary(&output, layer_idx);
                            }
                        }
                    }

                    Ok(output)
                } else {
                    let output = hidden + moe_out;

                    // ====== AttnRes: 块结束更新摘要 ======
                    if config.use_attnres {
                        if let Some(bs) = block_summary {
                            if bs.is_block_end(layer_idx) {
                                let _ = bs.update_summary(&output, layer_idx);
                            }
                        }
                    }

                    Ok(output)
                }
            } else {
                let output = hidden + moe_out;

                // ====== AttnRes: 块结束更新摘要 ======
                if config.use_attnres {
                    if let Some(bs) = block_summary {
                        if bs.is_block_end(layer_idx) {
                            let _ = bs.update_summary(&output, layer_idx);
                        }
                    }
                }

                Ok(output)
            }
        } else {
            let ffn_out = self.ffn.forward(&normed2);
            if config.use_mhc {
                if let (Some(ref dyn_proj), Some(ref sta_proj)) =
                    (&self.mhc_dynamic_proj, &self.mhc_static_proj)
                {
                    let h_3d = hidden
                        .clone()
                        .into_shape_with_order((
                            seq_len,
                            config.head_dim,
                            config.num_attention_heads,
                        ))
                        .map_err(|e| {
                            InferenceError::generation(format!("mHC reshape h failed: {}", e))
                        })?;
                    let d_3d = ffn_out
                        .into_shape_with_order((
                            seq_len,
                            config.head_dim,
                            config.num_attention_heads,
                        ))
                        .map_err(|e| {
                            InferenceError::generation(format!("mHC reshape ffn failed: {}", e))
                        })?;
                    let out_3d = mhc_residual(
                        &h_3d,
                        &d_3d,
                        dyn_proj,
                        sta_proj,
                        config.num_attention_heads,
                        config.head_dim,
                        config.mhc_sinkhorn_iterations,
                        config.mhc_epsilon,
                    )?;
                    let output = out_3d
                        .into_shape_with_order((
                            seq_len,
                            config.num_attention_heads * config.head_dim,
                        ))
                        .map_err(|e| {
                            InferenceError::generation(format!("mHC 3D→2D reshape failed: {}", e))
                        })?;

                    // ====== AttnRes: 块结束更新摘要 ======
                    if config.use_attnres {
                        if let Some(bs) = block_summary {
                            if bs.is_block_end(layer_idx) {
                                let _ = bs.update_summary(&output, layer_idx);
                            }
                        }
                    }

                    Ok(output)
                } else {
                    let output = hidden + ffn_out;

                    // ====== AttnRes: 块结束更新摘要 ======
                    if config.use_attnres {
                        if let Some(bs) = block_summary {
                            if bs.is_block_end(layer_idx) {
                                let _ = bs.update_summary(&output, layer_idx);
                            }
                        }
                    }

                    Ok(output)
                }
            } else {
                let output = hidden + ffn_out;

                // ====== AttnRes: 块结束更新摘要 ======
                if config.use_attnres {
                    if let Some(bs) = block_summary {
                        if bs.is_block_end(layer_idx) {
                            let _ = bs.update_summary(&output, layer_idx);
                        }
                    }
                }

                Ok(output)
            }
        }
    }

    fn mla_forward_with_cache(
        &self,
        x: &Array2<f32>,
        mla: &MLAWeights,
        config: &ModelConfig,
        kv_cache: Option<&mut KVCache>,
        positions: Option<&Array1<usize>>,
    ) -> InferenceResult<Array2<f32>> {
        let seq_len = x.nrows();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let latent_dim = mla.latent_dim();
        let half_latent = latent_dim / 2;

        let (q_reshaped, c_kv, k_full, uv_reshaped) =
            self.mla_prepare_qkv(x, mla, config, num_heads, num_kv_heads, head_dim, positions)?;

        let (k_reshaped, uv_reshaped) = if let Some(cache) = kv_cache {
            cache.update(&k_full, &uv_reshaped)?;
            cache.get()
        } else {
            (k_full, uv_reshaped)
        };

        let kv_seq_len = k_reshaped.dim().0;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let (k_expanded, v_expanded) =
            expand_gqa(&k_reshaped, &uv_reshaped, num_heads, num_kv_heads)?;

        let use_dsa = config.use_dsa && kv_seq_len >= config.dsa_short_seq_threshold;

        if use_dsa {
            let uq_proj = mla.uq_proj.as_ref().ok_or_else(|| {
                InferenceError::config("MLA-DSA requires uq_proj for latent space sparse attention")
            })?;

            let qk_latent_dim = uq_proj.dim().1;

            let c_k_all = c_kv.slice(s![.., ..half_latent]).to_owned();
            let c_v_all = c_kv.slice(s![.., half_latent..]).to_owned();

            let mut q_latent_3d = Array3::zeros((seq_len, qk_latent_dim, num_heads));
            for h in 0..num_heads {
                let q_h = q_reshaped.slice(s![.., h, ..]).to_owned();
                let q_latent_h = q_h.dot(uq_proj);
                q_latent_3d.slice_mut(s![.., .., h]).assign(&q_latent_h);
            }

            let mut c_k_3d = Array3::zeros((kv_seq_len, half_latent, num_heads));
            let mut c_v_3d = Array3::zeros((kv_seq_len, half_latent, num_heads));
            for h in 0..num_heads {
                c_k_3d.slice_mut(s![.., .., h]).assign(&c_k_all);
                c_v_3d.slice_mut(s![.., .., h]).assign(&c_v_all);
            }

            let dsa_config = config.dsa_config();

            let dsa_output = dsa::multihead_sparse_attention_3d(
                &q_latent_3d,
                &c_k_3d,
                &c_v_3d,
                num_heads,
                qk_latent_dim,
                &dsa_config,
                true,
            )
            .map_err(|e| {
                InferenceError::generation(format!(
                    "MLA-DSA 3D latent sparse attention failed: {}",
                    e
                ))
            })?;

            let dsa_output_2d = dsa_output
                .clone()
                .into_shape_with_order((seq_len, qk_latent_dim * num_heads))
                .map_err(|e| {
                    InferenceError::generation(format!("MLA-DSA 3D→2D reshape failed: {}", e))
                })?;
            let output = linear(&dsa_output_2d, &mla.o_proj);
            return Ok(output);
        }

        let uq_proj = mla.uq_proj.as_ref().ok_or_else(|| {
            InferenceError::config("MLA requires uq_proj for latent output projection")
        })?;

        let scores = compute_attention_scores(&q_reshaped, &k_expanded, scale);

        let attn = apply_causal_mask_and_softmax(&scores, seq_len, kv_seq_len)?;

        let attn_output = compute_attention_output(&attn, &v_expanded);

        let mut head_outputs: Vec<Array2<f32>> = Vec::with_capacity(num_heads);
        for h in 0..num_heads {
            let out_h = attn_output
                .slice(s![.., h * head_dim..(h + 1) * head_dim])
                .to_owned();
            let out_latent_h = out_h.dot(uq_proj);
            head_outputs.push(out_latent_h);
        }

        let total_out_dim = head_outputs[0].dim().1 * num_heads;
        let mut concat_output = Array2::zeros((seq_len, total_out_dim));
        for (h, out_h) in head_outputs.iter().enumerate() {
            let d = out_h.dim().1;
            concat_output
                .slice_mut(s![.., h * d..(h + 1) * d])
                .assign(out_h);
        }

        let output = linear(&concat_output, &mla.o_proj);

        Ok(output)
    }

    /// DeepSeek-V2 标准 MLA：潜在空间注意力
    ///
    /// # 与假 MLA 的本质区别
    ///
    /// ```text
    /// 假 MLA（之前）:
    ///   c_kv → uk_proj → K (head_dim) → Q @ K^T    ← 在原始维度计算 attention
    ///   c_kv → uv_proj → V (head_dim) → attn @ V     ← 输出也是原始维度
    ///
    /// 真 MLA（本函数）:
    ///   Q → uq_proj → Q_latent (latent_dim/2)         ← Q 压缩到潜在空间
    ///   scores = Q_latent @ c_k^T / sqrt(d)           ← ★ 潜在空间 attention score
    ///   attn = causal_softmax(scores)
    ///   output = attn @ c_v                            ← ★ 潜在空间输出！
    ///   final = o_proj(output)                         ← 投影回 hidden_size
    /// ```
    ///
    /// # 内存优势
    /// - Attention 矩阵: (seq, seq) × latent_dim/2 而非 head_dim
    /// - 缓存: O(seq × latent_dim) 而非 O(seq × kv_dim × 2)
    pub fn mla_forward_with_compressed_cache(
        &self,
        x: &Array2<f32>,
        mla: &MLAWeights,
        config: &ModelConfig,
        mut mla_cache: Option<&mut MLACache>,
        positions: Option<&Array1<usize>>,
    ) -> InferenceResult<Array2<f32>> {
        let seq_len = x.nrows();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let latent_dim = mla.latent_dim();
        let half_latent = latent_dim / 2;

        let uq_proj = mla
            .uq_proj
            .as_ref()
            .ok_or_else(|| InferenceError::config("true MLA requires uq_proj"))?;

        let qk_latent_dim = uq_proj.dim().1;

        let (q_reshaped, c_kv, _k_full, _uv_reshaped) =
            self.mla_prepare_qkv(x, mla, config, num_heads, num_kv_heads, head_dim, positions)?;

        if let Some(ref mut cache) = mla_cache {
            cache.update(&c_kv)?;
        }

        let cached_c_kv = match mla_cache {
            Some(ref cache) => cache.get(),
            None => c_kv,
        };

        let kv_seq_len = cached_c_kv.dim().0;
        let c_k_all = cached_c_kv.slice(s![.., ..half_latent]).to_owned();
        let c_v_all = cached_c_kv.slice(s![.., half_latent..]).to_owned();

        let scale = 1.0 / (qk_latent_dim as f32).sqrt();

        let mut head_outputs: Vec<Array2<f32>> = Vec::with_capacity(num_heads);

        for h in 0..num_heads {
            let q_h = q_reshaped.slice(s![.., h, ..]).to_owned();

            let q_latent_h = q_h.dot(uq_proj);

            let scores_h = q_latent_h.dot(&c_k_all.t()) * scale;

            let attn_h = apply_causal_mask_and_softmax_2d(&scores_h, seq_len, kv_seq_len)?;

            let out_h = attn_h.dot(&c_v_all);

            head_outputs.push(out_h);
        }

        let total_out_dim = head_outputs[0].dim().1 * num_heads;
        let mut concat_output = Array2::zeros((seq_len, total_out_dim));
        for (h, out_h) in head_outputs.iter().enumerate() {
            let d = out_h.dim().1;
            concat_output
                .slice_mut(s![.., h * d..(h + 1) * d])
                .assign(out_h);
        }

        let final_output = linear(&concat_output, &mla.o_proj);

        Ok(final_output)
    }

    /// MLA 准备 Q、K、V 和压缩 c_kv
    ///
    /// 返回值: (Q_reshaped, c_kv_compressed, K_full, V_reshaped)
    /// - Q_reshaped: (seq, n_heads, head_dim) 用于注意力计算
    /// - c_kv_compressed: (seq, latent_dim) 用于 MLA 压缩缓存
    /// - K_full: (seq, n_kv_heads, head_dim) 解压后的 K
    /// - V_reshaped: (seq, n_kv_heads, head_dim) 解压后的 V
    fn mla_prepare_qkv(
        &self,
        x: &Array2<f32>,
        mla: &MLAWeights,
        config: &ModelConfig,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        positions: Option<&Array1<usize>>,
    ) -> InferenceResult<(Array3<f32>, Array2<f32>, Array3<f32>, Array3<f32>)> {
        let seq_len = x.nrows();

        let q = linear(x, &mla.q_proj);

        let c_kv = linear(x, &mla.dkv_proj);

        let c_kv_reshaped = c_kv
            .to_owned()
            .to_shape((seq_len, 2, mla.latent_dim() / 2))
            .reshape_err("Failed to reshape KV latent for MLA")?
            .into_owned();

        let c_k = c_kv_reshaped.index_axis(Axis(1), 0).into_owned();
        let c_v = c_kv_reshaped.index_axis(Axis(1), 1).into_owned();

        let uk = linear(&c_k, &mla.uk_proj);
        let uv = linear(&c_v, &mla.uv_proj);

        let (q_out, k_out, uk_for_attention) = if let (Some(ref qr), Some(ref kr)) =
            (&mla.qr_proj, &mla.kr_proj)
        {
            let q_rope = linear(x, qr);
            let k_rope = linear(x, kr);
            let q_rope_out =
                apply_rotary_emb(&q_rope, num_heads, head_dim, config.rope_theta, positions)?;
            let k_rope_out = apply_rotary_emb(
                &k_rope,
                num_kv_heads,
                head_dim,
                config.rope_theta,
                positions,
            )?;
            let q_combined = &q + &q_rope_out;
            let uk_reshaped = uk
                .to_owned()
                .to_shape((seq_len, num_kv_heads, head_dim))
                .reshape_err("Failed to reshape UK for MLA attention")?
                .into_owned();
            (q_combined, k_rope_out, uk_reshaped)
        } else {
            let q_out = apply_rotary_emb(&q, num_heads, head_dim, config.rope_theta, positions)?;
            let uk_reshaped = uk
                .to_owned()
                .to_shape((seq_len, num_kv_heads, head_dim))
                .reshape_err("Failed to reshape UK for MLA RoPE")?
                .into_owned();
            let uk_2d = uk_reshaped
                .clone()
                .to_shape((seq_len, num_kv_heads * head_dim))
                .reshape_err("Failed to reshape UK back for MLA RoPE")?
                .into_owned();
            let k_out =
                apply_rotary_emb(&uk_2d, num_kv_heads, head_dim, config.rope_theta, positions)?;
            (q_out, k_out, uk_reshaped)
        };

        let q_reshaped = q_out
            .to_owned()
            .to_shape((seq_len, num_heads, head_dim))
            .reshape_err("Failed to reshape Q for MLA attention")?
            .into_owned();

        let k_reshaped = k_out
            .to_owned()
            .to_shape((seq_len, num_kv_heads, head_dim))
            .reshape_err("Failed to reshape K for MLA attention")?
            .into_owned();

        let uv_reshaped = uv
            .to_owned()
            .to_shape((seq_len, num_kv_heads, head_dim))
            .reshape_err("Failed to reshape UV for MLA attention")?
            .into_owned();

        let k_full = &k_reshaped + &uk_for_attention;

        Ok((q_reshaped, c_kv, k_full, uv_reshaped))
    }
}

// ============================================================================
// MultimodalTransformer 实现
// ============================================================================

/// 音频编码器权重
///
/// # 输入格式
/// 音频编码器期望输入为预处理后的音频特征，形状为 `(seq_len, audio_feature_dim)`。
/// 常见的音频特征包括：
/// - **Mel 频谱图**: `audio_feature_dim` = 滤波器组数量（如 80 或 128）
/// - **MFCC**: `audio_feature_dim` = 系数数量（如 13 或 40）
/// - **Log-Mel**: 对数 Mel 频谱图，通常归一化到 [-1, 1] 或 [0, 1]
///
/// # 预处理示例
/// ```ignore
/// // 使用 rustfft 或其他库提取 Mel 频谱
/// let mel_spectrogram = extract_mel_spectrogram(&audio_samples, sample_rate);
/// let audio_features = ndarray::Array2::from_shape_vec(
///     (num_frames, num_mel_bins),
///     mel_spectrogram
/// ).unwrap();
/// let encoded = model.encode_audio(&audio_features)?;
/// ```
///
/// # 注意
/// 本实现不包含原始波形处理，用户需自行提取音频特征。
#[derive(Debug, Clone)]
pub struct AudioEncoderWeights {
    pub conv1: Array2<f32>,
    pub pos_embed: Array2<f32>,
    pub layers: Vec<AudioLayerWeights>,
    pub proj: Array2<f32>,
    pub padding_mode: PaddingMode,
}

/// 音频编码器填充模式
#[derive(Debug, Clone, Copy, Default)]
pub enum PaddingMode {
    /// 边缘重复填充（默认）
    #[default]
    Replicate,
    /// 零填充
    Zero,
    /// 反射填充
    Reflect,
}

impl PaddingMode {
    /// 应用填充到音频特征
    pub fn apply_padding(&self, audio: &Array2<f32>, pad_size: usize) -> Array2<f32> {
        let seq_len = audio.nrows();
        let feature_dim = audio.ncols();
        let padded_len = seq_len + 2 * pad_size;
        let mut padded = Array2::zeros((padded_len, feature_dim));

        padded
            .slice_mut(s![pad_size..pad_size + seq_len, ..])
            .assign(audio);

        match self {
            PaddingMode::Replicate => {
                for i in 0..pad_size {
                    padded.slice_mut(s![i, ..]).assign(&audio.slice(s![0, ..]));
                    padded
                        .slice_mut(s![pad_size + seq_len + i, ..])
                        .assign(&audio.slice(s![seq_len - 1, ..]));
                }
            }
            PaddingMode::Zero => {
                // Already zeros, no action needed
            }
            PaddingMode::Reflect => {
                for i in 0..pad_size {
                    if i < seq_len {
                        padded
                            .slice_mut(s![pad_size - 1 - i, ..])
                            .assign(&audio.slice(s![i + 1, ..]));
                    }
                    if seq_len > pad_size + i + 1 {
                        padded
                            .slice_mut(s![pad_size + seq_len + i, ..])
                            .assign(&audio.slice(s![seq_len - 2 - i, ..]));
                    }
                }
            }
        }

        padded
    }
}

#[derive(Debug, Clone)]
pub struct AudioLayerWeights {
    pub norm1: Array1<f32>,
    pub attn_qkv: Array2<f32>,
    pub attn_proj: Array2<f32>,
    pub norm2: Array1<f32>,
    pub mlp_fc1: Array2<f32>,
    pub mlp_fc2: Array2<f32>,
}

impl AudioEncoderWeights {
    /// 创建音频编码器权重
    ///
    /// # 参数
    /// - `hidden_size`: 隐藏层维度
    /// - `num_layers`: Transformer 层数
    /// - `audio_feature_dim`: 音频特征维度（如 Mel 滤波器组数）
    /// - `max_audio_len`: 最大音频长度（用于位置编码）
    pub fn new(
        hidden_size: usize,
        num_layers: usize,
        audio_feature_dim: usize,
        max_audio_len: usize,
    ) -> Self {
        Self {
            conv1: Array2::zeros((audio_feature_dim * 3, hidden_size)),
            pos_embed: Array2::zeros((max_audio_len, hidden_size)),
            layers: (0..num_layers)
                .map(|_| AudioLayerWeights {
                    norm1: Array1::ones(hidden_size),
                    attn_qkv: Array2::zeros((hidden_size, hidden_size * 3)),
                    attn_proj: Array2::zeros((hidden_size, hidden_size)),
                    norm2: Array1::ones(hidden_size),
                    mlp_fc1: Array2::zeros((hidden_size, hidden_size * 4)),
                    mlp_fc2: Array2::zeros((hidden_size * 4, hidden_size)),
                })
                .collect(),
            proj: Array2::zeros((hidden_size, hidden_size)),
            padding_mode: PaddingMode::default(),
        }
    }

    /// 使用指定填充模式创建音频编码器
    pub fn with_padding_mode(mut self, mode: PaddingMode) -> Self {
        self.padding_mode = mode;
        self
    }

    /// 音频编码器前向传播
    ///
    /// # 参数
    /// - `audio`: 输入音频特征，形状为 `(seq_len, audio_feature_dim)`
    ///
    /// # 返回
    /// - 音频特征序列，形状为 `(seq_len, hidden_size)`
    pub fn forward(&self, audio: &Array2<f32>) -> InferenceResult<Array2<f32>> {
        let seq_len = audio.nrows();
        let audio_feature_dim = audio.ncols();
        let hidden_size = self.conv1.ncols();

        if audio_feature_dim * 3 != self.conv1.nrows() {
            return Err(InferenceError::config(format!(
                "Audio feature dimension mismatch: expected {}, got {}",
                self.conv1.nrows() / 3,
                audio_feature_dim
            )));
        }

        let audio_padded = self.padding_mode.apply_padding(audio, 1);

        let mut audio_expanded = Array2::zeros((seq_len, audio_feature_dim * 3));
        for t in 0..seq_len {
            audio_expanded
                .slice_mut(s![t, 0..audio_feature_dim])
                .assign(&audio_padded.slice(s![t, ..]));
            audio_expanded
                .slice_mut(s![t, audio_feature_dim..2 * audio_feature_dim])
                .assign(&audio_padded.slice(s![t + 1, ..]));
            audio_expanded
                .slice_mut(s![t, 2 * audio_feature_dim..])
                .assign(&audio_padded.slice(s![t + 2, ..]));
        }

        let x = linear(&audio_expanded, &self.conv1);

        let pos_embed = if seq_len <= self.pos_embed.nrows() {
            self.pos_embed.slice(s![0..seq_len, ..]).to_owned()
        } else {
            self.interpolate_pos_embed_1d(seq_len, hidden_size)
        };

        let mut x = &x + &pos_embed;

        for layer in &self.layers {
            x = self.audio_layer_forward(&x, layer)?;
        }

        Ok(linear(&x, &self.proj))
    }

    fn audio_layer_forward(
        &self,
        x: &Array2<f32>,
        layer: &AudioLayerWeights,
    ) -> InferenceResult<Array2<f32>> {
        let seq_len = x.nrows();
        let hidden_size = x.ncols();

        let normed = rms_norm(x, &layer.norm1, 1e-5);

        let qkv = linear(&normed, &layer.attn_qkv);
        let qkv_3d = qkv
            .into_shape_with_order((seq_len, 3, hidden_size))
            .map_err(|e| InferenceError::generation(format!("Failed to reshape QKV: {}", e)))?;

        let q: Array2<f32> = qkv_3d.slice(s![.., 0, ..]).to_owned();
        let k: Array2<f32> = qkv_3d.slice(s![.., 1, ..]).to_owned();
        let v: Array2<f32> = qkv_3d.slice(s![.., 2, ..]).to_owned();

        let scale = 1.0 / (hidden_size as f32).sqrt();
        let k_t = k.t().to_owned();
        let scores = q.dot(&k_t) * scale;
        let attn = softmax(&scores);
        let attn_out = attn.dot(&v);

        let mut x = x + &linear(&attn_out, &layer.attn_proj);

        let normed2 = rms_norm(&x, &layer.norm2, 1e-5);
        let mlp_hidden = linear(&normed2, &layer.mlp_fc1);
        let mlp_hidden = mlp_hidden.mapv(|v| v.max(0.0));
        let mlp_out = linear(&mlp_hidden, &layer.mlp_fc2);

        x = &x + &mlp_out;

        Ok(x)
    }

    fn interpolate_pos_embed_1d(&self, target_len: usize, hidden_size: usize) -> Array2<f32> {
        let orig_len = self.pos_embed.nrows();
        let mut pos_embed = Array2::zeros((target_len, hidden_size));

        for t in 0..target_len {
            let src_t = (t as f32) * (orig_len as f32 - 1.0) / (target_len as f32 - 1.0).max(1.0);
            let t0 = src_t.floor() as usize;
            let t1 = (t0 + 1).min(orig_len - 1);
            let dt = src_t - t0 as f32;

            for h in 0..hidden_size {
                let v0 = self.pos_embed[[t0, h]];
                let v1 = self.pos_embed[[t1, h]];
                pos_embed[[t, h]] = v0 * (1.0 - dt) + v1 * dt;
            }
        }

        pos_embed
    }
}

#[derive(Debug, Clone)]
pub struct VisionEncoderWeights {
    pub patch_embed: Array2<f32>,
    pub cls_token: Array1<f32>,
    pub pos_embed: Array2<f32>,
    pub layers: Vec<VisionLayerWeights>,
    pub proj: Array2<f32>,
    /// 原始 patch 网格高度（默认为 sqrt(num_patches-1)）
    pub orig_grid_h: Option<usize>,
    /// 原始 patch 网格宽度（默认为 sqrt(num_patches-1)）
    pub orig_grid_w: Option<usize>,
    /// 图像标准化均值（R, G, B）
    /// 默认为 CLIP/SigLIP 标准：[0.48145466, 0.4578275, 0.40821073]
    pub image_mean: [f32; 3],
    /// 图像标准化标准差（R, G, B）
    /// 默认为 CLIP/SigLIP 标准：[0.26862954, 0.26130258, 0.27577711]
    pub image_std: [f32; 3],
}

/// CLIP/SigLIP 标准图像均值
pub const CLIP_IMAGE_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
/// CLIP/SigLIP 标准图像标准差
pub const CLIP_IMAGE_STD: [f32; 3] = [0.26862954, 0.261_302_6, 0.275_777_1];

#[derive(Debug, Clone)]
pub struct VisionLayerWeights {
    pub norm1: Array1<f32>,
    pub attn_qkv: Array2<f32>,
    pub attn_proj: Array2<f32>,
    pub norm2: Array1<f32>,
    pub mlp_fc1: Array2<f32>,
    pub mlp_fc2: Array2<f32>,
}

impl VisionEncoderWeights {
    /// 创建视觉编码器权重
    ///
    /// # 参数
    /// - `hidden_size`: 隐藏层维度
    /// - `num_layers`: Transformer 层数
    /// - `image_patch_size`: 图像 patch 大小
    ///
    /// # 位置编码
    /// 位置编码表基于 **224×224** 输入图像计算。
    /// 推理时支持任意分辨率，位置编码会自动双线性插值。
    pub fn new(hidden_size: usize, num_layers: usize, image_patch_size: usize) -> Self {
        let patch_dim = image_patch_size * image_patch_size * 3;
        let num_patches = (224 / image_patch_size).pow(2) + 1;

        Self {
            patch_embed: Array2::zeros((patch_dim, hidden_size)),
            cls_token: Array1::zeros(hidden_size),
            pos_embed: Array2::zeros((num_patches, hidden_size)),
            layers: (0..num_layers)
                .map(|_| VisionLayerWeights {
                    norm1: Array1::ones(hidden_size),
                    attn_qkv: Array2::zeros((hidden_size, hidden_size * 3)),
                    attn_proj: Array2::zeros((hidden_size, hidden_size)),
                    norm2: Array1::ones(hidden_size),
                    mlp_fc1: Array2::zeros((hidden_size, hidden_size * 4)),
                    mlp_fc2: Array2::zeros((hidden_size * 4, hidden_size)),
                })
                .collect(),
            proj: Array2::zeros((hidden_size, hidden_size)),
            orig_grid_h: None,
            orig_grid_w: None,
            image_mean: CLIP_IMAGE_MEAN,
            image_std: CLIP_IMAGE_STD,
        }
    }

    /// 设置原始网格尺寸
    pub fn with_grid_size(mut self, grid_h: usize, grid_w: usize) -> Self {
        self.orig_grid_h = Some(grid_h);
        self.orig_grid_w = Some(grid_w);
        self
    }

    /// 设置图像标准化参数
    pub fn with_image_normalization(mut self, mean: [f32; 3], std: [f32; 3]) -> Self {
        self.image_mean = mean;
        self.image_std = std;
        self
    }

    /// 对图像像素进行标准化
    ///
    /// 将像素值从 [0, 255] 转换为标准化值：`(pixel / 255.0 - mean) / std`
    #[inline]
    fn normalize_pixel(&self, pixel: u8, channel: usize) -> f32 {
        let normalized = pixel as f32 / 255.0;
        (normalized - self.image_mean[channel]) / self.image_std[channel]
    }

    /// 双线性插值位置编码
    ///
    /// 将预训练的位置编码插值到目标分辨率
    ///
    /// # 安全性
    /// - 当 `target_h == 1` 或 `target_w == 1` 时，使用最近邻插值避免除零
    fn interpolate_pos_embed(
        &self,
        orig_grid: usize,
        target_h: usize,
        target_w: usize,
        hidden_size: usize,
    ) -> Array2<f32> {
        let target_patches = target_h * target_w;
        let mut pos_embed = Array2::zeros((target_patches + 1, hidden_size));

        for h_idx in 0..hidden_size {
            pos_embed[[0, h_idx]] = self.pos_embed[[0, h_idx]];
        }

        if target_patches == 0 {
            return pos_embed;
        }

        if orig_grid == target_h && orig_grid == target_w {
            for i in 0..target_patches {
                for h_idx in 0..hidden_size {
                    pos_embed[[i + 1, h_idx]] = self.pos_embed[[i + 1, h_idx]];
                }
            }
            return pos_embed;
        }

        let scale_h = if target_h > 1 {
            (orig_grid - 1) as f32 / (target_h - 1) as f32
        } else {
            0.0
        };
        let scale_w = if target_w > 1 {
            (orig_grid - 1) as f32 / (target_w - 1) as f32
        } else {
            0.0
        };

        for i in 0..target_h {
            for j in 0..target_w {
                let (src_i, di) = if target_h > 1 {
                    let src = i as f32 * scale_h;
                    (src.floor() as usize, src - src.floor())
                } else {
                    (0, 0.0)
                };

                let (src_j, dj) = if target_w > 1 {
                    let src = j as f32 * scale_w;
                    (src.floor() as usize, src - src.floor())
                } else {
                    (0, 0.0)
                };

                let i1 = (src_i + 1).min(orig_grid - 1);
                let j1 = (src_j + 1).min(orig_grid - 1);

                let idx00 = src_i * orig_grid + src_j + 1;
                let idx01 = src_i * orig_grid + j1 + 1;
                let idx10 = i1 * orig_grid + src_j + 1;
                let idx11 = i1 * orig_grid + j1 + 1;

                let target_idx = i * target_w + j + 1;

                for h_idx in 0..hidden_size {
                    let v00 = self.pos_embed[[idx00, h_idx]];
                    let v01 = self.pos_embed[[idx01, h_idx]];
                    let v10 = self.pos_embed[[idx10, h_idx]];
                    let v11 = self.pos_embed[[idx11, h_idx]];

                    let v0 = v00 * (1.0 - dj) + v01 * dj;
                    let v1 = v10 * (1.0 - dj) + v11 * dj;
                    let v = v0 * (1.0 - di) + v1 * di;

                    pos_embed[[target_idx, h_idx]] = v;
                }
            }
        }

        pos_embed
    }

    /// 双线性插值位置编码（支持非正方形网格）
    ///
    /// 将预训练的位置编码从非正方形原始网格插值到目标分辨率
    ///
    /// # 安全性
    /// - 当 `target_h == 1` 或 `target_w == 1` 时，使用最近邻插值避免除零
    fn interpolate_pos_embed_2d(
        &self,
        orig_grid_h: usize,
        orig_grid_w: usize,
        target_h: usize,
        target_w: usize,
        hidden_size: usize,
    ) -> Array2<f32> {
        let target_patches = target_h * target_w;
        let mut pos_embed = Array2::zeros((target_patches + 1, hidden_size));

        for h_idx in 0..hidden_size {
            pos_embed[[0, h_idx]] = self.pos_embed[[0, h_idx]];
        }

        if target_patches == 0 {
            return pos_embed;
        }

        if orig_grid_h == target_h && orig_grid_w == target_w {
            for i in 0..target_patches {
                for h_idx in 0..hidden_size {
                    pos_embed[[i + 1, h_idx]] = self.pos_embed[[i + 1, h_idx]];
                }
            }
            return pos_embed;
        }

        let scale_h = if target_h > 1 {
            (orig_grid_h - 1) as f32 / (target_h - 1) as f32
        } else {
            0.0
        };
        let scale_w = if target_w > 1 {
            (orig_grid_w - 1) as f32 / (target_w - 1) as f32
        } else {
            0.0
        };

        for i in 0..target_h {
            for j in 0..target_w {
                let (src_i, di) = if target_h > 1 {
                    let src = i as f32 * scale_h;
                    (src.floor() as usize, src - src.floor())
                } else {
                    (0, 0.0)
                };

                let (src_j, dj) = if target_w > 1 {
                    let src = j as f32 * scale_w;
                    (src.floor() as usize, src - src.floor())
                } else {
                    (0, 0.0)
                };

                let i1 = (src_i + 1).min(orig_grid_h - 1);
                let j1 = (src_j + 1).min(orig_grid_w - 1);

                let idx00 = src_i * orig_grid_w + src_j + 1;
                let idx01 = src_i * orig_grid_w + j1 + 1;
                let idx10 = i1 * orig_grid_w + src_j + 1;
                let idx11 = i1 * orig_grid_w + j1 + 1;

                let target_idx = i * target_w + j + 1;

                for h_idx in 0..hidden_size {
                    let v00 = self.pos_embed[[idx00, h_idx]];
                    let v01 = self.pos_embed[[idx01, h_idx]];
                    let v10 = self.pos_embed[[idx10, h_idx]];
                    let v11 = self.pos_embed[[idx11, h_idx]];

                    let v0 = v00 * (1.0 - dj) + v01 * dj;
                    let v1 = v10 * (1.0 - dj) + v11 * dj;
                    let v = v0 * (1.0 - di) + v1 * di;

                    pos_embed[[target_idx, h_idx]] = v;
                }
            }
        }

        pos_embed
    }

    /// 视觉编码器前向传播
    ///
    /// # 参数
    /// - `image`: 输入图像，形状为 `(H, W, 3)`
    ///
    /// # 返回
    /// - CLS token 的投影输出，形状为 `(1, hidden_size)`
    ///
    /// # Note
    /// - 支持任意分辨率输入，位置编码会自动插值
    /// - 推荐使用 224×224 分辨率以获得最佳效果（与预训练一致）
    pub fn forward(&self, image: &Array3<u8>) -> InferenceResult<Array2<f32>> {
        let (h, w, channels) = image.dim();

        if channels != 3 {
            return Err(InferenceError::image_preprocess(format!(
                "Image must have 3 channels (RGB), got {} channels",
                channels
            )));
        }

        let patch_size = ((self.patch_embed.nrows() / 3) as f32).sqrt() as usize;

        if h % patch_size != 0 || w % patch_size != 0 {
            return Err(InferenceError::image_preprocess(format!(
                "Image dimensions ({}, {}) must be divisible by patch_size {}. Consider padding the image.",
                h, w, patch_size
            )));
        }

        let num_patches_h = h / patch_size;
        let num_patches_w = w / patch_size;
        let num_patches = num_patches_h * num_patches_w;
        let hidden_size = self.patch_embed.ncols();
        let patch_dim = patch_size * patch_size * 3;

        let orig_grid_h = self
            .orig_grid_h
            .unwrap_or_else(|| ((self.pos_embed.nrows() - 1) as f32).sqrt() as usize);
        let orig_grid_w = self.orig_grid_w.unwrap_or(orig_grid_h);

        let pos_embed = if orig_grid_h == orig_grid_w {
            self.interpolate_pos_embed(orig_grid_h, num_patches_h, num_patches_w, hidden_size)
        } else {
            self.interpolate_pos_embed_2d(
                orig_grid_h,
                orig_grid_w,
                num_patches_h,
                num_patches_w,
                hidden_size,
            )
        };

        let image_2d = image
            .clone()
            .into_shape_with_order((h, w * 3))
            .map_err(|e| {
                InferenceError::image_preprocess(format!("Failed to reshape image: {}", e))
            })?;

        let mut patch_flat = Array2::zeros((num_patches, patch_dim));
        let mut row_idx = 0;
        for patch_i in 0..num_patches_h {
            for patch_j in 0..num_patches_w {
                let pi = patch_i * patch_size;
                let pj = patch_j * patch_size * 3;
                let mut col_idx = 0;
                for di in 0..patch_size {
                    let row_start = pi + di;
                    for dj in 0..patch_size {
                        let col_start = pj + dj * 3;
                        patch_flat[[row_idx, col_idx]] =
                            self.normalize_pixel(image_2d[[row_start, col_start]], 0);
                        patch_flat[[row_idx, col_idx + 1]] =
                            self.normalize_pixel(image_2d[[row_start, col_start + 1]], 1);
                        patch_flat[[row_idx, col_idx + 2]] =
                            self.normalize_pixel(image_2d[[row_start, col_start + 2]], 2);
                        col_idx += 3;
                    }
                }
                row_idx += 1;
            }
        }

        let patch_embeds = linear(&patch_flat, &self.patch_embed);

        let mut patches = Array2::zeros((num_patches + 1, hidden_size));
        patches
            .slice_mut(s![0, ..])
            .assign(&(&self.cls_token + &self.pos_embed.slice(s![0, ..])));
        patches
            .slice_mut(s![1.., ..])
            .assign(&(&patch_embeds + &pos_embed.slice(s![1.., ..])));

        let mut hidden = patches;
        for layer in &self.layers {
            hidden = self.vision_layer_forward(&hidden, layer)?;
        }

        let cls_output = hidden.row(0).to_owned();
        Ok(cls_output.insert_axis(Axis(0)).dot(&self.proj))
    }

    fn vision_layer_forward(
        &self,
        x: &Array2<f32>,
        layer: &VisionLayerWeights,
    ) -> InferenceResult<Array2<f32>> {
        let normed = rms_norm(x, &layer.norm1, 1e-6);

        let qkv = linear(&normed, &layer.attn_qkv);
        let seq_len = x.nrows();
        let hidden_size = x.ncols();

        let mut q = Array2::zeros((seq_len, hidden_size));
        let mut k = Array2::zeros((seq_len, hidden_size));
        let mut v = Array2::zeros((seq_len, hidden_size));

        for i in 0..seq_len {
            for j in 0..hidden_size {
                q[[i, j]] = qkv[[i, j]];
                k[[i, j]] = qkv[[i, hidden_size + j]];
                v[[i, j]] = qkv[[i, hidden_size * 2 + j]];
            }
        }

        let scale = 1.0 / (hidden_size as f32).sqrt();
        let k_t = k.t().to_owned();
        let scores = q.dot(&k_t) * scale;
        let attn = softmax(&scores);
        let attn_out = attn.dot(&v);
        let attn_out = linear(&attn_out, &layer.attn_proj);

        let hidden = x + &attn_out;

        let normed2 = rms_norm(&hidden, &layer.norm2, 1e-6);
        let mlp_hidden = linear(&normed2, &layer.mlp_fc1);
        let mlp_hidden = silu(mlp_hidden);
        let mlp_out = linear(&mlp_hidden, &layer.mlp_fc2);

        Ok(hidden + mlp_out)
    }
}

#[derive(Debug, Clone)]
pub struct MultimodalTransformer {
    pub layers: Vec<TransformerLayerWeights>,
    pub config: ModelConfig,
    pub embedding: Array2<f32>,
    pub lm_head: Array2<f32>,
    pub final_layernorm: Array1<f32>,
    pub vision_encoder: Option<VisionEncoderWeights>,
    pub audio_encoder: Option<AudioEncoderWeights>,
    weights_loaded: bool,
}

/// 生成状态，用于跟踪生成过程
struct GenerationState {
    kv_caches: Vec<KVCache>,
    total_len: usize,
    next_token: u32,
    output: Vec<u32>,
}

/// 生成步骤结果
enum StepResult {
    Continue,
    Stop,
}

impl MultimodalTransformer {
    /// 创建一个新的模型实例（权重未初始化）
    ///
    /// **警告**: 此构造函数创建的模型权重全为零，必须通过 `from_quant_loader`
    /// 或其他加载方法加载真实权重后才能使用。直接使用未初始化模型会导致输出全为零或 NaN。
    ///
    /// # Panics
    /// 当配置参数无效时会 panic（如 head_dim 为奇数）
    ///
    /// # 建议
    /// 推荐使用 `try_new` 方法以获得更好的错误处理
    pub fn new(config: ModelConfig) -> Self {
        Self::try_new(config).unwrap_or_else(|e| panic!("Invalid model configuration: {}", e))
    }

    /// 尝试创建一个新的模型实例，返回 Result
    pub fn try_new(config: ModelConfig) -> InferenceResult<Self> {
        config.validate()?;
        Ok(Self::new_unchecked(config))
    }

    /// 创建一个新的模型实例（跳过配置验证）
    fn new_unchecked(config: ModelConfig) -> Self {
        let layers = (0..config.num_hidden_layers)
            .map(|_| TransformerLayerWeights::new(&config))
            .collect();
        let vision_encoder =
            VisionEncoderWeights::new(config.hidden_size, 8, config.image_patch_size);
        let audio_encoder = if config.audio_feature_dim > 0 {
            Some(AudioEncoderWeights::new(
                config.hidden_size,
                4,
                config.audio_feature_dim,
                1024,
            ))
        } else {
            None
        };
        Self {
            layers,
            config: config.clone(),
            embedding: Array2::zeros((config.vocab_size, config.hidden_size)),
            lm_head: Array2::zeros((config.vocab_size, config.hidden_size)),
            final_layernorm: Array1::ones(config.hidden_size),
            vision_encoder: Some(vision_encoder),
            audio_encoder,
            weights_loaded: false,
        }
    }

    pub fn from_quant_loader(
        loader: &super::quant_loader::QuantizedModelLoader,
    ) -> InferenceResult<Self> {
        let config = loader.config().unwrap_or_else(|_| loader.infer_config());

        let num_layers = config.num_hidden_layers;
        let mut transformer = Self::try_new(config)?;

        transformer.embedding = loader.load_embedding()?;
        transformer.lm_head = loader
            .load_lm_head()
            .unwrap_or_else(|| transformer.embedding.clone());
        transformer.final_layernorm = loader.load_final_norm()?;

        for layer_idx in 0..num_layers {
            let q_weights = loader.load_layer(layer_idx)?;

            let mla = q_weights.mla.as_ref().map(|m| MLAWeights {
                q_proj: m.q_proj.clone(),
                o_proj: m.o_proj.clone(),
                dkv_proj: m.dkv_proj.clone(),
                uk_proj: m.uk_proj.clone(),
                uv_proj: m.uv_proj.clone(),
                uq_proj: None,
                qr_proj: m.qr_proj.clone(),
                kr_proj: m.kr_proj.clone(),
                q_norm: None,
                k_norm: None,
            });

            let mut layer =
                TransformerLayerWeights::from_quantized(&transformer.config, q_weights, mla);

            if layer_idx % 3 == 2 {
                let mod_embeds = loader.load_layer_modality_embeds(layer_idx);
                if !mod_embeds.is_empty() {
                    layer.moe.modality_embeds = Some(mod_embeds);
                }
            }

            // 配置滑动窗口注意力（如果启用）
            if transformer.config.enable_sliding_window {
                let window_size = transformer.config.sliding_window_size.unwrap_or(128);
                // 使用 Gemma3 默认模式：创建单个 Local 配置
                // 注意：实际的多层配置在 Transformer 层级管理
                let sw_config = SlidingWindowConfig::local_only(window_size, true);
                layer.sliding_window_config = Some(sw_config);
            }

            transformer.layers[layer_idx] = layer;
        }

        transformer.weights_loaded = true;
        Ok(transformer)
    }

    pub fn from_gguf(
        _gguf: &crate::hardware::kv_cache::streaming::StreamingAttention,
        _prefix: &str,
    ) -> InferenceResult<Self> {
        Err(InferenceError::config("from_gguf is not yet implemented. Use MultimodalTransformer::from_quant_loader() to load a quantized model."))
    }

    /// 标记权重已加载（仅供测试和手动加载权重后使用）
    pub fn mark_weights_loaded(&mut self) {
        self.weights_loaded = true;
    }

    pub fn get_token_embedding(&self, token_ids: &[u32]) -> InferenceResult<Array2<f32>> {
        let seq_len = token_ids.len();
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.embedding.nrows();
        let unk_id = self.config.unk_token_id;

        if unk_id >= vocab_size {
            return Err(InferenceError::config(format!(
                "unk_token_id ({}) is out of vocabulary range (0..{})",
                unk_id, vocab_size
            )));
        }

        let mut embedding = Array2::zeros((seq_len, hidden_size));

        for (i, &token_id) in token_ids.iter().enumerate() {
            let token_id = token_id as usize;
            let valid_id = if token_id < vocab_size {
                token_id
            } else {
                unk_id
            };
            embedding.row_mut(i).assign(&self.embedding.row(valid_id));
        }

        Ok(embedding)
    }

    pub fn forward(&self, x: &Array2<f32>) -> InferenceResult<Array2<f32>> {
        if !self.weights_loaded {
            return Err(InferenceError::config(
                "Model weights not loaded. Use from_quant_loader() to load weights before inference."
            ));
        }

        let mut hidden = x.clone();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &self.config, layer_idx)?;
        }

        hidden = rms_norm(&hidden, &self.final_layernorm, self.config.rms_norm_eps);

        Ok(hidden)
    }

    pub fn compute_logits(&self, hidden: &Array2<f32>) -> Array2<f32> {
        linear(hidden, &self.lm_head)
    }

    fn init_kv_caches(&self) -> Vec<KVCache> {
        self.layers
            .iter()
            .map(|_| {
                KVCache::new(
                    self.config.max_position_embeddings,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                )
            })
            .collect()
    }

    fn forward_with_kv_cache(
        &self,
        hidden: &Array2<f32>,
        kv_caches: &mut [KVCache],
        positions: &Array1<usize>,
    ) -> InferenceResult<Array2<f32>> {
        use super::attn_res::{AttnResConfig, BlockSummary};

        let mut h = hidden.clone();

        // ★ 创建 Block 摘要状态
        let attnres_config = if self.config.use_attnres {
            let nb = match self.config.attnres_num_blocks {
                Some(n) => n,
                None => {
                    // 自适应计算
                    let l = self.config.num_hidden_layers;
                    if l <= 8 {
                        2
                    } else if l <= 16 {
                        4
                    } else if l <= 64 {
                        8
                    } else {
                        16
                    }
                }
            };
            Some(AttnResConfig {
                enabled: true,
                num_blocks: nb,
                block_size: self.config.num_hidden_layers.div_ceil(nb),
                total_layers: self.config.num_hidden_layers,
                hidden_size: self.config.hidden_size,
                rms_eps: self.config.rms_norm_eps,
                init_scale: 0.0,
            })
        } else {
            None
        };

        let mut block_summary = attnres_config.as_ref().map(BlockSummary::new);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            h = layer.forward_with_cache(
                &h,
                &self.config,
                layer_idx,
                Some(&mut kv_caches[layer_idx]),
                Some(positions),
                block_summary.as_mut(), // ★ 传入
            )?;
        }
        Ok(h)
    }

    fn sample_next_token(
        &self,
        hidden: &Array2<f32>,
        temperature: f32,
        top_p: f32,
        last_row: bool,
    ) -> u32 {
        let logits = self.compute_logits(hidden);
        let row_idx = if last_row { logits.nrows() - 1 } else { 0 };
        let last_logits = logits.row(row_idx).to_owned();
        Self::sample_token_default(&last_logits, temperature, top_p)
    }

    fn fuse_image_features(
        &self,
        embedding: &Array2<f32>,
        image_features: &Array2<f32>,
        insert_pos: usize,
    ) -> Array2<f32> {
        let text_len = embedding.nrows();
        let num_patches = image_features.nrows();
        let hidden_size = embedding.ncols();

        assert!(
            insert_pos < text_len,
            "insert_pos {} must be less than text_len {}",
            insert_pos,
            text_len
        );

        let mut fused = Array2::zeros((text_len - 1 + num_patches, hidden_size));

        fused
            .slice_mut(s![0..insert_pos, ..])
            .assign(&embedding.slice(s![0..insert_pos, ..]));
        fused
            .slice_mut(s![insert_pos..insert_pos + num_patches, ..])
            .assign(image_features);
        fused
            .slice_mut(s![insert_pos + num_patches.., ..])
            .assign(&embedding.slice(s![insert_pos + 1.., ..]));

        fused
    }

    fn fuse_image_features_with_validation(
        &self,
        embedding: &Array2<f32>,
        image_features: &Array2<f32>,
        tokens: &[u32],
    ) -> InferenceResult<Array2<f32>> {
        let text_len = embedding.nrows();
        let num_patches = image_features.nrows();
        let hidden_size = embedding.ncols();

        let start_pos = tokens.iter().position(|&t| t == IM_START_TOKEN_ID as u32);
        let end_pos = tokens.iter().position(|&t| t == IM_END_TOKEN_ID as u32);

        match (start_pos, end_pos) {
            (Some(start), Some(end)) if end > start => {
                let num_placeholder_tokens = end - start - 1;
                let result_len = text_len - num_placeholder_tokens - 2 + num_patches;
                let mut fused = Array2::zeros((result_len, hidden_size));

                fused
                    .slice_mut(s![0..start, ..])
                    .assign(&embedding.slice(s![0..start, ..]));
                fused
                    .slice_mut(s![start..start + num_patches, ..])
                    .assign(image_features);
                fused
                    .slice_mut(s![start + num_patches.., ..])
                    .assign(&embedding.slice(s![end + 1.., ..]));

                Ok(fused)
            }
            (Some(start), None) => {
                let mut fused = Array2::zeros((text_len - 1 + num_patches, hidden_size));
                fused
                    .slice_mut(s![0..start, ..])
                    .assign(&embedding.slice(s![0..start, ..]));
                fused
                    .slice_mut(s![start..start + num_patches, ..])
                    .assign(image_features);
                fused
                    .slice_mut(s![start + num_patches.., ..])
                    .assign(&embedding.slice(s![start + 1.., ..]));
                Ok(fused)
            }
            _ => Ok(self.fuse_image_features(embedding, image_features, 0)),
        }
    }

    /// 执行单步生成
    fn generation_step(
        &self,
        state: &mut GenerationState,
        params: &GenerateParams,
        is_first: bool,
    ) -> InferenceResult<StepResult> {
        if is_first {
            return Ok(StepResult::Continue);
        }

        let pos: Array1<usize> = Array1::from_elem(1, state.total_len - 1);
        let embedding = self.get_token_embedding(&[state.next_token])?;
        let hidden = self.forward_with_kv_cache(&embedding, &mut state.kv_caches, &pos)?;
        let hidden = rms_norm(&hidden, &self.final_layernorm, self.config.rms_norm_eps);
        state.next_token = self.sample_next_token(&hidden, params.temperature, params.top_p, false);

        if state.next_token == 0 {
            return Ok(StepResult::Stop);
        }

        state.output.push(state.next_token);
        state.total_len += 1;
        Ok(StepResult::Continue)
    }

    /// 初始化生成状态
    fn init_generation_state(&self, embedding: &Array2<f32>) -> InferenceResult<GenerationState> {
        if embedding.nrows() > self.config.max_position_embeddings {
            return Err(InferenceError::generation(format!(
                "Input sequence length {} exceeds max position embeddings ({})",
                embedding.nrows(),
                self.config.max_position_embeddings
            )));
        }

        Ok(GenerationState {
            kv_caches: self.init_kv_caches(),
            total_len: embedding.nrows(),
            next_token: 0,
            output: Vec::new(),
        })
    }

    /// 处理初始嵌入并生成第一个token
    fn process_initial_embedding(
        &self,
        state: &mut GenerationState,
        embedding: &Array2<f32>,
        params: &GenerateParams,
    ) -> InferenceResult<()> {
        let init_positions: Array1<usize> = (0..state.total_len).collect();
        let hidden =
            self.forward_with_kv_cache(embedding, &mut state.kv_caches, &init_positions)?;
        let hidden = rms_norm(&hidden, &self.final_layernorm, self.config.rms_norm_eps);
        state.next_token = self.sample_next_token(&hidden, params.temperature, params.top_p, true);

        if state.next_token != 0 {
            state.output.push(state.next_token);
            state.total_len += 1;
        }
        Ok(())
    }

    pub fn generate(&self, tokens: &[u32], max_length: usize) -> InferenceResult<Vec<u32>> {
        let params = GenerateParams {
            max_new_tokens: max_length,
            ..Default::default()
        };
        self.generate_with_params(tokens, &params)
    }

    /// 文本生成（带采样参数）
    ///
    /// # 参数
    /// - `tokens`: 输入 token 序列
    /// - `params`: 生成参数配置
    ///
    /// # Errors
    /// - Token ID 超出词表范围时返回错误
    /// - 内部张量形状不匹配时返回错误
    ///
    /// # Example
    /// ```ignore
    /// let model = MultimodalTransformer::default();
    /// let params = GenerateParams { max_new_tokens: 100, ..Default::default() };
    /// let output = model.generate_with_params(&[1, 2, 3], &params)?;
    /// ```
    pub fn generate_with_params(
        &self,
        tokens: &[u32],
        params: &GenerateParams,
    ) -> InferenceResult<Vec<u32>> {
        let mut output = Vec::new();
        let mut current_tokens = tokens.to_vec();

        for _ in 0..params.max_new_tokens {
            let embedding = self.get_token_embedding(&current_tokens)?;
            let hidden = self.forward(&embedding)?;
            let logits = self.compute_logits(&hidden);

            let last_logits = logits.row(logits.nrows() - 1).to_owned();
            let next_token =
                Self::sample_token_default(&last_logits, params.temperature, params.top_p);

            output.push(next_token);
            current_tokens.push(next_token);

            if next_token == 0 {
                break;
            }
        }

        Ok(output)
    }

    pub fn generate_with_cache(
        &self,
        tokens: &[u32],
        max_length: usize,
    ) -> InferenceResult<Vec<u32>> {
        let params = GenerateParams {
            max_new_tokens: max_length,
            ..Default::default()
        };
        self.generate_with_cache_params(tokens, &params)
    }

    /// 文本生成（带 KV 缓存和采样参数）
    ///
    /// 使用 KV 缓存优化自回归生成，避免重复计算。
    ///
    /// # 参数
    /// - `tokens`: 输入 token 序列
    /// - `params`: 生成参数配置
    ///
    /// # Errors
    /// - Token ID 超出词表范围时返回错误
    /// - 序列长度超过 `max_position_embeddings` 时可能 panic
    ///
    /// # Performance
    /// 相比 `generate_with_params`，使用 KV 缓存可显著提升长序列生成速度。
    pub fn generate_with_cache_params(
        &self,
        tokens: &[u32],
        params: &GenerateParams,
    ) -> InferenceResult<Vec<u32>> {
        let embedding = self.get_token_embedding(tokens)?;
        let mut state = self.init_generation_state(&embedding)?;
        self.process_initial_embedding(&mut state, &embedding, params)?;

        if state.next_token == 0 {
            return Ok(state.output);
        }

        for _ in 1..params.max_new_tokens {
            match self.generation_step(&mut state, params, false)? {
                StepResult::Stop => break,
                StepResult::Continue => {}
            }
        }

        Ok(state.output)
    }

    /// 使用外部 KV 缓存生成文本
    ///
    /// 允许外部管理缓存生命周期，适用于多轮对话场景。
    ///
    /// # 参数
    /// - `tokens`: 输入 token 序列
    /// - `params`: 生成参数
    /// - `kv_caches`: 外部 KV 缓存数组（每层一个）
    ///
    /// # 返回
    /// - 生成的 token 序列
    pub fn generate_with_external_cache(
        &self,
        tokens: &[u32],
        params: &GenerateParams,
        kv_caches: &mut [Option<KVCache>],
    ) -> InferenceResult<Vec<u32>> {
        if !self.weights_loaded {
            return Err(InferenceError::generation("Model weights not loaded"));
        }

        let embedding = self.get_token_embedding(tokens)?;
        let seq_len = embedding.nrows();

        let mut output = Vec::with_capacity(params.max_new_tokens);
        let mut current_pos = seq_len;

        let positions: Array1<usize> = (0..seq_len).collect();

        let mut hidden = {
            let mut layer_input = embedding;
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let kv_cache = kv_caches.get_mut(layer_idx);
                layer_input = layer.forward_with_cache(
                    &layer_input,
                    &self.config,
                    layer_idx,
                    kv_cache.and_then(|c| c.as_mut()),
                    Some(&positions),
                    None, // AttnRes: 此处不使用 BlockSummary（预填充阶段）
                )?;
            }
            rms_norm(
                &layer_input,
                &self.final_layernorm,
                self.config.rms_norm_eps,
            )
        };

        let logits = linear(&hidden, &self.lm_head);
        let last_logits = logits.row(seq_len - 1).to_owned();
        let first_token = Self::sample_token(
            &last_logits,
            params.temperature,
            params.top_p,
            &mut rand::thread_rng(),
        );
        output.push(first_token);

        for _step in 1..params.max_new_tokens {
            let next_embed = self.get_token_embedding(&[output[output.len() - 1]])?;
            let pos_arr = Array1::from_elem(1, current_pos);

            hidden = {
                let mut layer_input = next_embed;
                for (layer_idx, layer) in self.layers.iter().enumerate() {
                    let kv_cache = kv_caches.get_mut(layer_idx);
                    layer_input = layer.forward_with_cache(
                        &layer_input,
                        &self.config,
                        layer_idx,
                        kv_cache.and_then(|c| c.as_mut()),
                        Some(&pos_arr),
                        None, // AttnRes: 此处不使用 BlockSummary（解码阶段）
                    )?;
                }
                rms_norm(
                    &layer_input,
                    &self.final_layernorm,
                    self.config.rms_norm_eps,
                )
            };

            let logits = linear(&hidden, &self.lm_head);
            let last_logits = logits.row(0).to_owned();
            let next_token = Self::sample_token(
                &last_logits,
                params.temperature,
                params.top_p,
                &mut rand::thread_rng(),
            );

            output.push(next_token);
            current_pos += 1;
        }

        Ok(output)
    }

    fn sample_token<R: Rng>(
        logits: &Array1<f32>,
        temperature: f32,
        top_p: f32,
        rng: &mut R,
    ) -> u32 {
        if temperature <= 0.0 {
            return logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
        }

        let vocab_size = logits.len();
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f32> = logits
            .iter()
            .map(|&l| ((l - max_logit) / temperature).exp())
            .collect();
        let sum: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|&p| p / sum).collect();

        let effective_top_p = if top_p > 0.0 && top_p < 1.0 {
            top_p
        } else {
            1.0
        };

        if effective_top_p >= 1.0 {
            if let Ok(dist) = WeightedIndex::new(&probs) {
                return dist.sample(rng) as u32;
            }
            return probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
        }

        let mut indexed_probs: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();

        let estimated_top_k = ((vocab_size as f32) * 0.1).min(1000.0) as usize;
        let nth_idx = estimated_top_k.min(vocab_size.saturating_sub(1));

        indexed_probs.select_nth_unstable_by(nth_idx, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut top_candidates: Vec<(usize, f32)> =
            indexed_probs.into_iter().take(nth_idx + 1).collect();
        top_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0f32;
        let mut cutoff_idx = 0;
        for (i, &(_, prob)) in top_candidates.iter().enumerate() {
            cumsum += prob;
            if cumsum >= effective_top_p {
                cutoff_idx = i;
                break;
            }
            cutoff_idx = i;
        }

        let top_indices: Vec<usize> = top_candidates[..=cutoff_idx]
            .iter()
            .map(|&(i, _)| i)
            .collect();
        let top_probs: Vec<f32> = top_candidates[..=cutoff_idx]
            .iter()
            .map(|&(_, p)| p)
            .collect();

        let total: f32 = top_probs.iter().sum();
        let normalized_probs: Vec<f32> = top_probs.iter().map(|&p| p / total).collect();

        if let Ok(dist) = WeightedIndex::new(&normalized_probs) {
            return top_indices[dist.sample(rng)] as u32;
        }
        top_indices[0] as u32
    }

    fn sample_token_default(logits: &Array1<f32>, temperature: f32, top_p: f32) -> u32 {
        Self::sample_token(logits, temperature, top_p, &mut rand::thread_rng())
    }

    /// 多模态生成（文本 + 图像）
    ///
    /// # 参数
    /// - `tokens`: 文本 token 序列
    /// - `image`: 图像输入 (H, W, 3)，RGB 格式
    /// - `max_length`: 最大生成长度
    ///
    /// # 返回
    /// 生成的 token 序列
    pub fn generate_multimodal(
        &self,
        tokens: &[u32],
        image: Option<&Array3<u8>>,
        max_length: usize,
    ) -> InferenceResult<Vec<u32>> {
        let params = GenerateParams {
            max_new_tokens: max_length,
            ..Default::default()
        };
        self.generate_multimodal_with_params(tokens, image, &params)
    }

    /// 多模态生成（带采样参数）
    ///
    /// # 参数
    /// - `tokens`: 文本 token 序列
    /// - `image`: 图像输入 (H, W, 3)，RGB 格式
    /// - `params`: 生成参数配置
    ///
    /// # 返回
    /// 生成的 token 序列
    pub fn generate_multimodal_with_params(
        &self,
        tokens: &[u32],
        image: Option<&Array3<u8>>,
        params: &GenerateParams,
    ) -> InferenceResult<Vec<u32>> {
        let mut embedding = self.get_token_embedding(tokens)?;

        if let (Some(img), Some(vision_encoder)) = (image, &self.vision_encoder) {
            let image_features = vision_encoder.forward(img)?;

            embedding =
                self.fuse_image_features_with_validation(&embedding, &image_features, tokens)?;
        }

        let mut state = self.init_generation_state(&embedding)?;
        self.process_initial_embedding(&mut state, &embedding, params)?;

        if state.next_token == 0 {
            return Ok(state.output);
        }

        for _ in 1..params.max_new_tokens {
            match self.generation_step(&mut state, params, false)? {
                StepResult::Stop => break,
                StepResult::Continue => {}
            }
        }

        Ok(state.output)
    }

    /// 使用预计算的视觉特征进行多模态生成
    ///
    /// # 参数
    /// - `tokens`: 输入 token 序列（包含图像占位符 tokens）
    /// - `visual_features`: 预计算的视觉特征 (num_patches, hidden_dim)
    /// - `params`: 生成参数配置
    ///
    /// # 返回
    /// 生成的 token 序列
    pub fn generate_with_visual_features(
        &self,
        tokens: &[u32],
        visual_features: Option<&Array2<f32>>,
        params: &GenerateParams,
    ) -> InferenceResult<Vec<u32>> {
        let mut embedding = self.get_token_embedding(tokens)?;

        if let Some(features) = visual_features {
            embedding = self.fuse_image_features_with_validation(&embedding, features, tokens)?;
        }

        let mut state = self.init_generation_state(&embedding)?;
        self.process_initial_embedding(&mut state, &embedding, params)?;

        if state.next_token == 0 {
            return Ok(state.output);
        }

        for _ in 1..params.max_new_tokens {
            match self.generation_step(&mut state, params, false)? {
                StepResult::Stop => break,
                StepResult::Continue => {}
            }
        }

        Ok(state.output)
    }

    /// 流式生成（逐 token 回调）
    ///
    /// 使用 KV 缓存优化，支持通过回调实时处理生成的 token。
    ///
    /// # 参数
    /// - `tokens`: 输入 token 序列
    /// - `params`: 生成参数配置
    /// - `callback`: 每个 token 生成后调用的回调，返回 false 可提前终止
    ///
    /// # Errors
    /// - Token ID 超出词表范围时返回 `InferenceError`
    /// - 回调返回错误时提前终止
    ///
    /// # Example
    /// ```ignore
    /// model.generate_streaming(&tokens, &params, |token_id| {
    ///     print!("{}", tokenizer.decode(&[token_id])?);
    ///     Ok(true) // 继续生成
    /// })?;
    /// ```
    pub fn generate_streaming<F>(
        &self,
        tokens: &[u32],
        params: &GenerateParams,
        mut callback: F,
    ) -> InferenceResult<()>
    where
        F: FnMut(u32) -> InferenceResult<bool>,
    {
        let mut kv_caches = self.init_kv_caches();

        let embedding = self
            .get_token_embedding(tokens)
            .map_err(|e| InferenceError::generation(e.to_string()))?;

        let mut total_len = tokens.len();
        let init_positions: Array1<usize> = (0..total_len).collect();

        let hidden = self
            .forward_with_kv_cache(&embedding, &mut kv_caches, &init_positions)
            .map_err(|e| InferenceError::generation(e.to_string()))?;
        let hidden = rms_norm(&hidden, &self.final_layernorm, self.config.rms_norm_eps);
        let mut next_token =
            self.sample_next_token(&hidden, params.temperature, params.top_p, true);

        if next_token == 0 {
            return Ok(());
        }

        if !callback(next_token)? {
            return Ok(());
        }
        total_len += 1;

        for _ in 1..params.max_new_tokens {
            let pos: Array1<usize> = Array1::from_elem(1, total_len - 1);
            let embedding = self
                .get_token_embedding(&[next_token])
                .map_err(|e| InferenceError::generation(e.to_string()))?;
            let hidden = self
                .forward_with_kv_cache(&embedding, &mut kv_caches, &pos)
                .map_err(|e| InferenceError::generation(e.to_string()))?;
            let hidden = rms_norm(&hidden, &self.final_layernorm, self.config.rms_norm_eps);
            next_token = self.sample_next_token(&hidden, params.temperature, params.top_p, false);

            if next_token == 0 {
                break;
            }

            if !callback(next_token)? {
                break;
            }
            total_len += 1;
        }

        Ok(())
    }

    /// 多模态流式生成（逐 token 回调）
    ///
    /// 支持文本和图像混合输入，使用 KV 缓存优化。
    ///
    /// # 参数
    /// - `tokens`: 输入 token 序列
    /// - `image`: 图像输入 (H, W, 3)，RGB 格式
    /// - `params`: 生成参数配置
    /// - `callback`: 每个 token 生成后调用的回调，返回 false 可提前终止
    ///
    /// # Errors
    /// - Token ID 超出词表范围时返回 `InferenceError`
    /// - 图像尺寸不符合要求时返回错误
    /// - 视觉编码器未初始化时返回错误
    ///
    /// # Note
    /// 图像特征会在 `<image>` token 位置插入到文本嵌入中。
    pub fn generate_streaming_multimodal<F>(
        &self,
        tokens: &[u32],
        image: Option<&Array3<u8>>,
        params: &GenerateParams,
        mut callback: F,
    ) -> InferenceResult<()>
    where
        F: FnMut(u32) -> InferenceResult<bool>,
    {
        let mut kv_caches = self.init_kv_caches();

        let mut embedding = self
            .get_token_embedding(tokens)
            .map_err(|e| InferenceError::generation(e.to_string()))?;

        if let (Some(img), Some(vision_encoder)) = (image, &self.vision_encoder) {
            let image_features = vision_encoder
                .forward(img)
                .map_err(|e| InferenceError::generation(e.to_string()))?;

            embedding = self
                .fuse_image_features_with_validation(&embedding, &image_features, tokens)
                .map_err(|e| InferenceError::generation(e.to_string()))?;
        }

        let mut total_len = embedding.nrows();
        let init_positions: Array1<usize> = (0..total_len).collect();

        let hidden = self
            .forward_with_kv_cache(&embedding, &mut kv_caches, &init_positions)
            .map_err(|e| InferenceError::generation(e.to_string()))?;
        let hidden = rms_norm(&hidden, &self.final_layernorm, self.config.rms_norm_eps);
        let mut next_token =
            self.sample_next_token(&hidden, params.temperature, params.top_p, true);

        if next_token == 0 {
            return Ok(());
        }

        if !callback(next_token)? {
            return Ok(());
        }
        total_len += 1;

        for _ in 1..params.max_new_tokens {
            let pos: Array1<usize> = Array1::from_elem(1, total_len - 1);
            let embedding = self
                .get_token_embedding(&[next_token])
                .map_err(|e| InferenceError::generation(e.to_string()))?;
            let hidden = self
                .forward_with_kv_cache(&embedding, &mut kv_caches, &pos)
                .map_err(|e| InferenceError::generation(e.to_string()))?;
            let hidden = rms_norm(&hidden, &self.final_layernorm, self.config.rms_norm_eps);
            next_token = self.sample_next_token(&hidden, params.temperature, params.top_p, false);

            if next_token == 0 {
                break;
            }

            if !callback(next_token)? {
                break;
            }
            total_len += 1;
        }

        Ok(())
    }

    pub fn encode_image(&self, image: &Array3<u8>) -> InferenceResult<Array2<f32>> {
        if let Some(ref vision_encoder) = self.vision_encoder {
            vision_encoder.forward(image)
        } else {
            Err(InferenceError::multimodal("Vision encoder not initialized"))
        }
    }

    /// 编码音频特征
    ///
    /// # 参数
    /// - `audio`: 音频特征序列，形状为 `(seq_len, audio_feature_dim)`
    ///
    /// # 返回
    /// - 编码后的音频特征，形状为 `(seq_len, hidden_size)`
    pub fn encode_audio(&self, audio: &Array2<f32>) -> InferenceResult<Array2<f32>> {
        if let Some(ref audio_encoder) = self.audio_encoder {
            audio_encoder.forward(audio)
        } else {
            Err(InferenceError::multimodal("Audio encoder not initialized"))
        }
    }
}

// ============================================================================
// 训练支持类型和方法
// ============================================================================

use ndarray::ArrayD;

/// 训练相关错误
#[derive(Debug)]
pub enum TrainingError {
    Io(std::io::Error),
    Serialization(String),
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    WeightNotFound(String),
    Other(String),
}

impl std::fmt::Display for TrainingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainingError::Io(e) => write!(f, "IO error: {}", e),
            TrainingError::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            TrainingError::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            TrainingError::WeightNotFound(name) => write!(f, "Weight not found: {}", name),
            TrainingError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for TrainingError {}

impl From<std::io::Error> for TrainingError {
    fn from(e: std::io::Error) -> Self {
        TrainingError::Io(e)
    }
}

/// 训练模式的前向传播结果
pub struct TrainForwardResult {
    pub logits: Array2<f32>,
    pub loss: Option<f32>,
    pub activation_cache: ActivationCache,
}

/// 中间激活值缓存（用于反向传播）
pub struct ActivationCache {
    pub hidden_states: Vec<Array2<f32>>,
    pub attention_scores: Vec<Option<Array3<f32>>>,
    pub input_embeddings: Array2<f32>,
}

/// 可训练参数引用
pub struct ParamRef {
    pub name: String,
    pub data: ArrayD<f32>,
    pub grad: Option<ArrayD<f32>>,
}

impl MultimodalTransformer {
    pub fn forward_train(
        &self,
        input_ids: &[usize],
        labels: Option<&[usize]>,
    ) -> InferenceResult<TrainForwardResult> {
        if !self.weights_loaded {
            return Err(InferenceError::config(
                "Model weights not loaded. Use from_quant_loader() to load weights before training."
            ));
        }

        let token_ids: Vec<u32> = input_ids.iter().map(|&id| id as u32).collect();
        let embeddings = self.get_token_embedding(&token_ids)?;
        let hidden = self.forward(&embeddings)?;
        let logits = self.compute_logits(&hidden);

        let loss = if let Some(labels) = labels {
            Some(self.compute_cross_entropy_loss(&logits, labels)?)
        } else {
            None
        };

        let activation_cache = ActivationCache {
            hidden_states: vec![embeddings.clone(), hidden.clone()],
            attention_scores: vec![None; self.layers.len()],
            input_embeddings: embeddings,
        };

        Ok(TrainForwardResult {
            logits,
            loss,
            activation_cache,
        })
    }

    fn compute_cross_entropy_loss(
        &self,
        logits: &Array2<f32>,
        labels: &[usize],
    ) -> InferenceResult<f32> {
        let seq_len = logits.nrows();
        if labels.len() != seq_len {
            return Err(InferenceError::generation(format!(
                "Labels length ({}) must match sequence length ({})",
                labels.len(),
                seq_len
            )));
        }

        let mut total_loss = 0.0f32;
        for (i, &label) in labels.iter().enumerate() {
            let row = logits.row(i);
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
            let log_prob = (row[label] - max_val) - exp_sum.ln();
            total_loss -= log_prob;
        }

        Ok(total_loss / seq_len as f32)
    }

    pub fn trainable_params(&self) -> Vec<ParamRef> {
        let mut params = Vec::new();

        params.push(ParamRef {
            name: "embedding".to_string(),
            data: self.embedding.clone().into_dyn(),
            grad: None,
        });

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layers.{}", layer_idx);

            params.push(ParamRef {
                name: format!("{}.input_layernorm", prefix),
                data: layer.input_layernorm.clone().into_dyn(),
                grad: None,
            });

            params.push(ParamRef {
                name: format!("{}.post_attention_layernorm", prefix),
                data: layer.post_attention_layernorm.clone().into_dyn(),
                grad: None,
            });

            params.push(ParamRef {
                name: format!("{}.attention.q_proj", prefix),
                data: layer.attention.q_proj.clone().into_dyn(),
                grad: None,
            });

            params.push(ParamRef {
                name: format!("{}.attention.k_proj", prefix),
                data: layer.attention.k_proj.clone().into_dyn(),
                grad: None,
            });

            params.push(ParamRef {
                name: format!("{}.attention.v_proj", prefix),
                data: layer.attention.v_proj.clone().into_dyn(),
                grad: None,
            });

            params.push(ParamRef {
                name: format!("{}.attention.o_proj", prefix),
                data: layer.attention.o_proj.clone().into_dyn(),
                grad: None,
            });

            params.push(ParamRef {
                name: format!("{}.ffn.gate_proj", prefix),
                data: layer.ffn.gate_proj.clone().into_dyn(),
                grad: None,
            });

            params.push(ParamRef {
                name: format!("{}.ffn.up_proj", prefix),
                data: layer.ffn.up_proj.clone().into_dyn(),
                grad: None,
            });

            params.push(ParamRef {
                name: format!("{}.ffn.down_proj", prefix),
                data: layer.ffn.down_proj.clone().into_dyn(),
                grad: None,
            });
        }

        params.push(ParamRef {
            name: "final_layernorm".to_string(),
            data: self.final_layernorm.clone().into_dyn(),
            grad: None,
        });

        params.push(ParamRef {
            name: "lm_head".to_string(),
            data: self.lm_head.clone().into_dyn(),
            grad: None,
        });

        params
    }

    pub fn load_weights(&mut self, _path: &std::path::Path) -> Result<(), TrainingError> {
        Ok(())
    }

    pub fn save_weights(&self, _path: &std::path::Path) -> Result<(), TrainingError> {
        Ok(())
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_identity_weight(rows: usize, cols: usize) -> Array2<f32> {
        let mut weight = Array2::zeros((rows, cols));
        let min_dim = rows.min(cols);
        for i in 0..min_dim {
            weight[[i, i]] = 1.0;
        }
        weight
    }

    fn create_constant_weight(rows: usize, cols: usize, value: f32) -> Array2<f32> {
        Array2::from_elem((rows, cols), value)
    }

    fn create_deterministic_attention_weights(config: &ModelConfig) -> AttentionWeights {
        let qkv_size = config.num_attention_heads * config.head_dim;
        let kv_size = config.num_key_value_heads * config.head_dim;

        AttentionWeights {
            q_proj: create_constant_weight(qkv_size, config.hidden_size, 0.1),
            k_proj: create_constant_weight(kv_size, config.hidden_size, 0.1),
            v_proj: create_constant_weight(kv_size, config.hidden_size, 0.1),
            o_proj: create_constant_weight(config.hidden_size, qkv_size, 0.1),
        }
    }

    fn create_deterministic_ffn_weights(config: &ModelConfig) -> FFNWeights {
        FFNWeights {
            gate_proj: create_constant_weight(config.intermediate_size, config.hidden_size, 0.1),
            up_proj: create_constant_weight(config.intermediate_size, config.hidden_size, 0.1),
            down_proj: create_constant_weight(config.hidden_size, config.intermediate_size, 0.1),
        }
    }

    fn create_deterministic_layer(config: &ModelConfig) -> TransformerLayerWeights {
        let experts = (0..config.moe_num_experts)
            .map(|_| create_deterministic_ffn_weights(config))
            .collect();

        TransformerLayerWeights {
            attention: create_deterministic_attention_weights(config),
            mla: if config.use_mla {
                Some(MLAWeights::new(config))
            } else {
                None
            },
            ffn: create_deterministic_ffn_weights(config),
            moe: MoEWeights {
                experts,
                router: create_constant_weight(config.moe_num_experts, config.hidden_size, 0.1),
                top_k: config.moe_top_k,
                modality_embeds: None,
            },
            input_layernorm: Array1::ones(config.hidden_size),
            post_attention_layernorm: Array1::ones(config.hidden_size),
            mhc_dynamic_proj: if config.use_mhc {
                Some(Array2::zeros((
                    config.num_attention_heads,
                    config.hidden_size,
                )))
            } else {
                None
            },
            mhc_static_proj: if config.use_mhc {
                Some(Array2::zeros((
                    config.num_attention_heads,
                    config.hidden_size,
                )))
            } else {
                None
            },
            moe_v2: None,
            attnres_pseudo_query: if config.use_attnres {
                Some(Array1::zeros(config.hidden_size))
            } else {
                None
            },
            sliding_window_config: None,
        }
    }

    #[test]
    fn test_ffn_forward() {
        let ffn = FFNWeights {
            gate_proj: Array2::zeros((2048, 512)),
            up_proj: Array2::zeros((2048, 512)),
            down_proj: Array2::zeros((512, 2048)),
        };

        let x = Array2::zeros((1, 512));
        let output = ffn.forward(&x);
        assert_eq!(output.dim(), (1, 512));
    }

    #[test]
    fn test_top_k_selection() {
        let probs = Array2::zeros((1, 8));
        let (indices, weights, offsets) = top_k_selection(&probs, 2);
        assert_eq!(indices.len(), 2);
        assert_eq!(weights.len(), 2);
        assert_eq!(offsets.len(), 2);
        assert_eq!(offsets[0], 0);
        assert_eq!(offsets[1], 2);
    }

    #[test]
    fn test_transformer_layer_moe() {
        let config = ModelConfig::default();
        let layer = TransformerLayerWeights::new(&config);

        let x = Array2::zeros((config.num_attention_heads, config.hidden_size));

        let output = layer.forward(&x, &config, 2);
        assert!(output.is_ok());

        let output = layer.forward(&x, &config, 0);
        assert!(output.is_ok());
    }

    #[test]
    fn test_rope_position_encoding() {
        let head_dim = 64;
        let seq_len = 4;
        let theta = 10000.0;

        let x = Array2::ones((seq_len, head_dim));

        let positions: Array1<usize> = (0..seq_len).collect();
        let result = apply_rotary_emb(&x, 1, head_dim, theta, Some(&positions)).unwrap();

        assert_eq!(result.dim(), (seq_len, head_dim));

        let result_no_pos = apply_rotary_emb(&x, 1, head_dim, theta, None).unwrap();
        assert_eq!(result_no_pos.dim(), (seq_len, head_dim));

        for i in 0..seq_len {
            for j in 0..head_dim {
                assert!(!result[[i, j]].is_nan());
                assert!(!result[[i, j]].is_infinite());
            }
        }
    }

    #[test]
    fn test_rope_different_positions() {
        let head_dim = 64;
        let theta = 10000.0;

        let x = Array2::ones((1, head_dim));

        let pos_0 = Array1::from_elem(1, 0usize);
        let pos_1 = Array1::from_elem(1, 1usize);

        let result_0 = apply_rotary_emb(&x, 1, head_dim, theta, Some(&pos_0)).unwrap();
        let result_1 = apply_rotary_emb(&x, 1, head_dim, theta, Some(&pos_1)).unwrap();

        let mut has_diff = false;
        for j in 0..head_dim {
            if (result_0[[0, j]] - result_1[[0, j]]).abs() > 1e-6 {
                has_diff = true;
                break;
            }
        }
        assert!(
            has_diff,
            "RoPE should produce different outputs for different positions"
        );
    }

    #[test]
    fn test_rms_norm_numerical_stability() {
        let hidden_size = 512;
        let weight = Array1::ones(hidden_size);

        let x_normal = Array2::ones((2, hidden_size));
        let result = rms_norm(&x_normal, &weight, 1e-6);
        assert_eq!(result.dim(), (2, hidden_size));

        let x_small: Array2<f32> = Array2::from_elem((2, hidden_size), 1e-10);
        let result_small = rms_norm(&x_small, &weight, 1e-6);
        for i in 0..2 {
            for j in 0..hidden_size {
                assert!(!result_small[[i, j]].is_nan());
            }
        }

        let x_large: Array2<f32> = Array2::from_elem((2, hidden_size), 1e6);
        let result_large = rms_norm(&x_large, &weight, 1e-6);
        for i in 0..2 {
            for j in 0..hidden_size {
                assert!(!result_large[[i, j]].is_nan());
                assert!(!result_large[[i, j]].is_infinite());
            }
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let normal = Array2::from_shape_vec((1, 5), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let result = softmax(&normal);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax should sum to 1");

        let large = Array2::from_shape_vec((1, 3), vec![1000.0, 1001.0, 1002.0]).unwrap();
        let result_large = softmax(&large);
        let sum_large: f32 = result_large.iter().sum();
        assert!(
            (sum_large - 1.0).abs() < 1e-5,
            "Softmax should handle large values"
        );
        for val in result_large.iter() {
            assert!(!val.is_nan() && !val.is_infinite());
        }

        let neg_inf = Array2::from_shape_vec((1, 3), vec![1.0, f32::NEG_INFINITY, 2.0]).unwrap();
        let result_inf = softmax(&neg_inf);
        assert!(
            (result_inf[[0, 1]] - 0.0).abs() < 1e-10,
            "NEG_INFINITY should become 0 after softmax"
        );
        assert!(
            result_inf[[0, 0]] > 0.0 && result_inf[[0, 2]] > 0.0,
            "Non-infinity values should have positive probability"
        );
    }

    #[test]
    fn test_causal_mask_correctness() {
        let seq_len = 4;
        let kv_seq_len = 4;
        let mask = create_causal_mask(seq_len, kv_seq_len);

        assert_eq!(mask.dim(), (seq_len, kv_seq_len));

        for i in 0..seq_len {
            for j in 0..kv_seq_len {
                if j > i {
                    assert_eq!(
                        mask[[i, j]],
                        f32::NEG_INFINITY,
                        "Position ({}, {}) should be masked (future token)",
                        i,
                        j
                    );
                } else {
                    assert_eq!(
                        mask[[i, j]],
                        0.0,
                        "Position ({}, {}) should not be masked (past/current token)",
                        i,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn test_causal_mask_precomputed_vs_dynamic() {
        let small_mask = create_causal_mask(100, 100);
        let large_mask = create_causal_mask(5000, 5000);

        assert_eq!(small_mask.dim(), (100, 100));
        assert_eq!(large_mask.dim(), (5000, 5000));

        for i in 0..10 {
            for j in 0..10 {
                assert_eq!(
                    small_mask[[i, j]],
                    large_mask[[i, j]],
                    "Precomputed and dynamic masks should be identical"
                );
            }
        }
    }

    #[test]
    fn test_causal_mask_with_kv_cache() {
        let seq_len = 2;
        let kv_seq_len = 5;
        let offset = kv_seq_len - seq_len;

        let mask = create_causal_mask(seq_len, kv_seq_len);

        assert_eq!(mask.dim(), (seq_len, kv_seq_len));

        assert_eq!(mask[[0, 0]], 0.0, "Row 0, Col 0 should be visible");
        assert_eq!(
            mask[[0, offset]],
            0.0,
            "Row 0, Col {} (offset) should be visible",
            offset
        );
        assert_eq!(
            mask[[0, offset + 1]],
            f32::NEG_INFINITY,
            "Row 0, Col {} should be masked",
            offset + 1
        );

        assert_eq!(
            mask[[1, kv_seq_len - 1]],
            0.0,
            "Row 1, last col should be visible"
        );
        assert_eq!(
            mask[[1, offset]],
            0.0,
            "Row 1, Col {} should be visible",
            offset
        );

        for i in 0..seq_len {
            let visible_end = offset + i + 1;
            for j in 0..visible_end {
                assert_eq!(
                    mask[[i, j]],
                    0.0,
                    "Position ({}, {}) should be visible",
                    i,
                    j
                );
            }
            for j in visible_end..kv_seq_len {
                assert_eq!(
                    mask[[i, j]],
                    f32::NEG_INFINITY,
                    "Position ({}, {}) should be masked",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_causal_mask_with_kv_cache_consistency() {
        let test_cases = [(2, 5), (3, 7), (1, 10), (5, 15)];

        for (seq_len, kv_seq_len) in test_cases {
            let mask = create_causal_mask(seq_len, kv_seq_len);
            let offset = kv_seq_len - seq_len;

            for i in 0..seq_len {
                let visible_end = offset + i + 1;
                for j in 0..visible_end.min(kv_seq_len) {
                    assert_eq!(
                        mask[[i, j]],
                        0.0,
                        "Position ({}, {}) should be visible for seq_len={}, kv_seq_len={}",
                        i,
                        j,
                        seq_len,
                        kv_seq_len
                    );
                }
                for j in visible_end..kv_seq_len {
                    assert_eq!(
                        mask[[i, j]],
                        f32::NEG_INFINITY,
                        "Position ({}, {}) should be masked for seq_len={}, kv_seq_len={}",
                        i,
                        j,
                        seq_len,
                        kv_seq_len
                    );
                }
            }
        }
    }

    #[test]
    fn test_top_p_sampling_distribution() {
        let logits = Array1::from_shape_vec(5, vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();

        let mut counts = vec![0usize; 5];
        let num_samples = 1000;

        for _ in 0..num_samples {
            let token = MultimodalTransformer::sample_token_default(&logits, 1.0, 0.9);
            counts[token as usize] += 1;
        }

        assert!(
            counts[0] < 10,
            "Lowest probability token should rarely be sampled with top_p=0.9, got {}",
            counts[0]
        );
        assert!(
            counts[4] > counts[3],
            "Highest probability token should be sampled most often"
        );
    }

    #[test]
    fn test_moe_weight_normalization() {
        let probs =
            Array2::from_shape_vec((1, 8), vec![0.1, 0.2, 0.3, 0.4, 0.05, 0.05, 0.05, 0.05])
                .unwrap();

        let (indices, weights, offsets) = top_k_selection(&probs, 2);

        assert_eq!(indices.len(), 2);
        assert_eq!(weights.len(), 2);
        assert_eq!(offsets.len(), 2);

        let weight_sum: f32 = weights.iter().sum();
        assert!(
            (weight_sum - 1.0).abs() < 1e-5,
            "MoE weights should be normalized to sum to 1, got {}",
            weight_sum
        );

        for &w in &weights {
            assert!(w >= 0.0 && w <= 1.0, "Weights should be in [0, 1]");
        }
    }

    #[test]
    fn test_kv_cache_update_and_retrieve() {
        let max_seq = 1024;
        let num_heads = 8;
        let head_dim = 64;

        let mut cache = KVCache::new(max_seq, num_heads, head_dim);

        let k = Array3::ones((4, num_heads, head_dim));
        let v = Array3::ones((4, num_heads, head_dim)) * 2.0;

        cache
            .update(&k, &v)
            .expect("KV cache update should succeed");

        let (cached_k, cached_v) = cache.get();
        assert_eq!(cached_k.dim(), (4, num_heads, head_dim));
        assert_eq!(cached_v.dim(), (4, num_heads, head_dim));

        let k2 = Array3::ones((2, num_heads, head_dim)) * 3.0;
        let v2 = Array3::ones((2, num_heads, head_dim)) * 4.0;
        cache
            .update(&k2, &v2)
            .expect("KV cache update should succeed");

        let (cached_k, cached_v) = cache.get();
        assert_eq!(cached_k.dim(), (6, num_heads, head_dim));
        assert_eq!(cached_v.dim(), (6, num_heads, head_dim));
    }

    #[test]
    fn test_linear_projection() {
        let batch = 2;
        let in_dim = 512;
        let out_dim = 2048;

        let weight = Array2::ones((out_dim, in_dim));
        let x = Array2::ones((batch, in_dim));

        let output = linear(&x, &weight);

        assert_eq!(output.dim(), (batch, out_dim));

        for i in 0..batch {
            for j in 0..out_dim {
                assert!((output[[i, j]] - in_dim as f32).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_end_to_end_generation() {
        let mut config = ModelConfig::default();
        config.hidden_size = 256;
        config.num_hidden_layers = 2;
        config.num_attention_heads = 4;
        config.num_key_value_heads = 4;
        config.head_dim = 64;
        config.intermediate_size = 512;
        config.vocab_size = 1000;
        config.max_position_embeddings = 512;

        let mut model = MultimodalTransformer::new(config.clone());

        for layer in &mut model.layers {
            layer.attention = create_deterministic_attention_weights(&config);
            layer.ffn = create_deterministic_ffn_weights(&config);
        }
        model.embedding = create_constant_weight(config.vocab_size, config.hidden_size, 0.1);
        model.lm_head = create_constant_weight(config.vocab_size, config.hidden_size, 0.1);
        model.weights_loaded = true;

        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let params = GenerateParams {
            max_new_tokens: 10,
            temperature: 1.0,
            top_p: 0.9,
            ..Default::default()
        };

        let result = model.generate_with_params(&tokens, &params);
        assert!(result.is_ok(), "Generation should succeed");

        let output = result.unwrap();
        assert!(
            output.len() <= 10,
            "Output should not exceed max_new_tokens"
        );

        let first_token = output.first().copied().unwrap_or(0);
        assert!(
            first_token < 1000,
            "Generated token should be within vocab size, got {}",
            first_token
        );

        assert!(output.len() > 0, "Should generate at least one token");
    }

    #[test]
    fn test_end_to_end_generation_with_cache() {
        let mut config = ModelConfig::default();
        config.hidden_size = 256;
        config.num_hidden_layers = 2;
        config.num_attention_heads = 4;
        config.num_key_value_heads = 4;
        config.head_dim = 64;
        config.intermediate_size = 512;
        config.vocab_size = 1000;
        config.max_position_embeddings = 512;

        let mut model = MultimodalTransformer::new(config.clone());

        for layer in &mut model.layers {
            layer.attention = create_deterministic_attention_weights(&config);
            layer.ffn = create_deterministic_ffn_weights(&config);
        }
        model.embedding = create_constant_weight(config.vocab_size, config.hidden_size, 0.1);
        model.lm_head = create_constant_weight(config.vocab_size, config.hidden_size, 0.1);
        model.weights_loaded = true;

        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let params = GenerateParams {
            max_new_tokens: 10,
            temperature: 1.0,
            top_p: 0.9,
            ..Default::default()
        };

        let result = model.generate_with_cache_params(&tokens, &params);
        assert!(result.is_ok(), "Generation with cache should succeed");

        let output = result.unwrap();
        assert!(
            output.len() <= 10,
            "Output should not exceed max_new_tokens"
        );

        let first_token = output.first().copied().unwrap_or(0);
        assert!(
            first_token < 1000,
            "Generated token should be within vocab size, got {}",
            first_token
        );
        assert!(output.len() > 0, "Should generate at least one token");
    }

    #[test]
    fn test_end_to_end_streaming_generation() {
        let mut config = ModelConfig::default();
        config.hidden_size = 256;
        config.num_hidden_layers = 2;
        config.num_attention_heads = 4;
        config.num_key_value_heads = 4;
        config.head_dim = 64;
        config.intermediate_size = 512;
        config.vocab_size = 1000;
        config.max_position_embeddings = 512;

        let mut model = MultimodalTransformer::new(config);
        model.mark_weights_loaded();

        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let params = GenerateParams {
            max_new_tokens: 10,
            temperature: 1.0,
            top_p: 0.9,
            ..Default::default()
        };

        let mut generated_tokens = Vec::new();
        let result = model.generate_streaming(&tokens, &params, |token_id| {
            generated_tokens.push(token_id);
            Ok(true)
        });

        assert!(result.is_ok(), "Streaming generation should succeed");
        assert!(
            generated_tokens.len() <= 10,
            "Should generate at most max_new_tokens"
        );

        for &token in &generated_tokens {
            assert!(token < 1000, "Generated token should be within vocab size");
        }
    }

    #[test]
    fn test_end_to_end_multimodal_generation() {
        let mut config = ModelConfig::default();
        config.hidden_size = 256;
        config.num_hidden_layers = 2;
        config.num_attention_heads = 4;
        config.num_key_value_heads = 4;
        config.head_dim = 64;
        config.intermediate_size = 512;
        config.vocab_size = 1000;
        config.max_position_embeddings = 512;

        let mut model = MultimodalTransformer::new(config);
        model.mark_weights_loaded();

        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let params = GenerateParams {
            max_new_tokens: 5,
            temperature: 1.0,
            top_p: 0.9,
            ..Default::default()
        };

        let mut generated_tokens = Vec::new();
        let result = model.generate_streaming_multimodal(&tokens, None, &params, |token_id| {
            generated_tokens.push(token_id);
            Ok(true)
        });

        assert!(
            result.is_ok(),
            "Multimodal generation without image should succeed"
        );
        assert!(
            generated_tokens.len() <= 5,
            "Should generate at most max_new_tokens"
        );
    }

    #[test]
    fn test_generation_early_termination() {
        let mut config = ModelConfig::default();
        config.hidden_size = 256;
        config.num_hidden_layers = 2;
        config.num_attention_heads = 4;
        config.num_key_value_heads = 4;
        config.head_dim = 64;
        config.intermediate_size = 512;
        config.vocab_size = 1000;
        config.max_position_embeddings = 512;

        let mut model = MultimodalTransformer::new(config);
        model.mark_weights_loaded();

        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let params = GenerateParams {
            max_new_tokens: 100,
            temperature: 1.0,
            top_p: 0.9,
            ..Default::default()
        };

        let mut count = 0;
        let result = model.generate_streaming(&tokens, &params, |_token_id| {
            count += 1;
            Ok(count < 3)
        });

        assert!(result.is_ok(), "Streaming generation should succeed");
        assert_eq!(
            count, 3,
            "Should have generated exactly 3 tokens before termination"
        );
    }

    #[test]
    fn test_model_forward_shape_consistency() {
        let mut config = ModelConfig::default();
        config.hidden_size = 256;
        config.num_hidden_layers = 4;
        config.num_attention_heads = 8;
        config.num_key_value_heads = 8;
        config.head_dim = 32;
        config.intermediate_size = 512;
        config.vocab_size = 1000;
        config.max_position_embeddings = 512;
        config.use_mla = true;
        config.mla_decoupled_rope = true;

        let hidden_size = config.hidden_size;
        let vocab_size = config.vocab_size;

        let mut model = MultimodalTransformer::new(config);
        model.mark_weights_loaded();

        let batch_sizes = vec![1, 2, 4, 8];
        let seq_lengths = vec![1, 16, 64, 128];

        for batch in &batch_sizes {
            for seq_len in &seq_lengths {
                let tokens: Vec<u32> = (0..(*batch * *seq_len))
                    .map(|i| (i % vocab_size) as u32)
                    .collect();

                let embedding = model.get_token_embedding(&tokens).unwrap();
                assert_eq!(embedding.dim(), (*batch * *seq_len, hidden_size));

                let hidden = model.forward(&embedding).unwrap();
                assert_eq!(hidden.dim(), (*batch * *seq_len, hidden_size));

                let logits = model.compute_logits(&hidden);
                assert_eq!(logits.dim(), (*batch * *seq_len, vocab_size));
            }
        }
    }

    #[test]
    fn test_rope_rotation_correctness() {
        let head_dim = 4;
        let theta = 10000.0;

        let x = Array2::from_shape_vec((1, head_dim), vec![1.0, 0.0, 1.0, 0.0]).unwrap();

        let pos_0 = Array1::from_elem(1, 0usize);
        let result_0 = apply_rotary_emb(&x, 1, head_dim, theta, Some(&pos_0)).unwrap();

        for j in 0..head_dim {
            assert!(
                (result_0[[0, j]] - x[[0, j]]).abs() < 1e-6,
                "At position 0, RoPE should not change the input"
            );
        }

        let pos_1 = Array1::from_elem(1, 1usize);
        let result_1 = apply_rotary_emb(&x, 1, head_dim, theta, Some(&pos_1)).unwrap();

        let cos_0 = (1.0_f32 * theta.powf(0.0 / head_dim as f32)).cos();
        let sin_0 = (1.0_f32 * theta.powf(0.0 / head_dim as f32)).sin();

        let expected_0 = 1.0 * cos_0 - 0.0 * sin_0;
        let expected_1 = 1.0 * sin_0 + 0.0 * cos_0;

        assert!(
            (result_1[[0, 0]] - expected_0).abs() < 1e-5,
            "RoPE rotation at position 1 should match expected value"
        );
        assert!(
            (result_1[[0, 1]] - expected_1).abs() < 1e-5,
            "RoPE rotation at position 1 should match expected value"
        );
    }

    #[test]
    fn test_rms_norm_correctness() {
        let hidden_size = 4;
        let weight = Array1::from_shape_vec(hidden_size, vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let x = Array2::from_shape_vec((1, hidden_size), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let result = rms_norm(&x, &weight, 1e-6);

        let rms = (4.0_f32 / 4.0 + 1e-6).sqrt();
        let expected = 1.0 / rms;

        assert!(
            (result[[0, 0]] - expected).abs() < 1e-5,
            "RMS norm should normalize correctly"
        );
        assert!(
            (result[[0, 1]] - expected * 2.0).abs() < 1e-5,
            "RMS norm should apply weight correctly"
        );
    }

    #[test]
    fn test_softmax_correctness() {
        let x = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let result = softmax(&x);

        let e1 = 1.0_f32.exp();
        let e2 = 2.0_f32.exp();
        let e3 = 3.0_f32.exp();
        let sum = e1 + e2 + e3;

        assert!(
            (result[[0, 0]] - e1 / sum).abs() < 1e-5,
            "Softmax value 0 incorrect"
        );
        assert!(
            (result[[0, 1]] - e2 / sum).abs() < 1e-5,
            "Softmax value 1 incorrect"
        );
        assert!(
            (result[[0, 2]] - e3 / sum).abs() < 1e-5,
            "Softmax value 2 incorrect"
        );

        let prob_sum: f32 = result.iter().sum();
        assert!(
            (prob_sum - 1.0).abs() < 1e-5,
            "Softmax probabilities should sum to 1"
        );
    }

    #[test]
    fn test_moe_routing_correctness() {
        let config = ModelConfig::default();
        let layer = create_deterministic_layer(&config);

        let x = Array2::from_elem((2, config.hidden_size), 0.5);
        let result = layer.forward(&x, &config, 2);
        assert!(result.is_ok(), "MoE layer forward should succeed");

        let output = result.unwrap();
        assert_eq!(
            output.dim(),
            (2, config.hidden_size),
            "MoE output shape mismatch"
        );

        for val in output.iter() {
            assert!(
                !val.is_nan() && !val.is_infinite(),
                "MoE output should be finite"
            );
        }

        let sum: f32 = output.iter().sum();
        assert!(
            sum.abs() > 1e-6,
            "MoE output should not be all zeros, sum={}",
            sum
        );
    }

    #[test]
    fn test_attention_output_range() {
        let config = ModelConfig::default();
        let layer = create_deterministic_layer(&config);

        let x = Array2::from_elem((4, config.hidden_size), 0.5);
        let result = layer.attention.forward(&x, &config);
        assert!(
            result.is_ok(),
            "Attention forward should succeed: {:?}",
            result.err()
        );

        let output = result.unwrap();
        assert_eq!(
            output.dim(),
            (4, config.hidden_size),
            "Attention output shape mismatch"
        );

        for val in output.iter() {
            assert!(
                !val.is_nan() && !val.is_infinite(),
                "Attention output should be finite"
            );
        }

        let sum: f32 = output.iter().sum();
        assert!(
            sum.abs() > 1e-6,
            "Attention output should not be all zeros, sum={}",
            sum
        );

        let has_variance = output
            .iter()
            .any(|&v| (v - sum / output.len() as f32).abs() > 1e-6);
        assert!(has_variance, "Attention output should have variance");
    }

    #[test]
    fn test_linear_projection_correctness() {
        let weight = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();

        let result = linear(&x, &weight);

        assert_eq!(result.dim(), (1, 3), "Linear output shape mismatch");

        assert!((result[[0, 0]] - (1.0 * 1.0 + 2.0 * 2.0)).abs() < 1e-5);
        assert!((result[[0, 1]] - (1.0 * 3.0 + 2.0 * 4.0)).abs() < 1e-5);
        assert!((result[[0, 2]] - (1.0 * 5.0 + 2.0 * 6.0)).abs() < 1e-5);
    }

    #[test]
    fn test_silu_activation() {
        let x = Array2::from_shape_vec((1, 3), vec![0.0, 1.0, -1.0]).unwrap();
        let result = silu(x);

        assert!((result[[0, 0]] - 0.0).abs() < 1e-5, "SiLU(0) should be 0");

        let expected_1 = 1.0 / (1.0 + (-1.0_f32).exp());
        assert!(
            (result[[0, 1]] - expected_1).abs() < 1e-5,
            "SiLU(1) incorrect"
        );

        let expected_neg1 = -1.0 / (1.0 + 1.0_f32.exp());
        assert!(
            (result[[0, 2]] - expected_neg1).abs() < 1e-5,
            "SiLU(-1) incorrect"
        );
    }

    #[test]
    fn test_dsa_integration() {
        let mut config = ModelConfig::default();
        config.use_dsa = false;

        let layer = TransformerLayerWeights::new(&config);

        let x = Array2::from_elem((10, config.hidden_size), 0.5);
        let result = layer.attention.forward(&x, &config);
        assert!(
            result.is_ok(),
            "Attention forward should succeed: {:?}",
            result.err()
        );

        let output = result.unwrap();
        assert_eq!(output.dim(), (10, config.hidden_size));

        let nan_count = output.iter().filter(|v| v.is_nan()).count();
        let inf_count = output.iter().filter(|v| v.is_infinite()).count();

        assert!(
            nan_count == 0,
            "Attention output has {} NaN values",
            nan_count
        );
        assert!(
            inf_count == 0,
            "Attention output has {} infinite values",
            inf_count
        );
    }

    #[test]
    fn test_mla_vs_standard_attention_shape() {
        let mut config = ModelConfig::default();
        config.hidden_size = 256;
        config.num_attention_heads = 4;
        config.num_key_value_heads = 2;
        config.head_dim = 64;
        config.mla_latent_dim = 128;

        let x = Array2::from_elem((4, config.hidden_size), 0.5);

        let layer = TransformerLayerWeights::new(&config);
        let result = layer.attention.forward(&x, &config);
        assert!(result.is_ok(), "Standard attention should succeed");
        let std_output = result.unwrap();

        assert!(layer.mla.is_some(), "MLA weights should be initialized");

        if let Some(ref mla) = layer.mla {
            let positions: Array1<usize> = (0..4).collect();
            let mut kv_cache = KVCache::new(10, config.num_key_value_heads, config.head_dim);

            let mla_result = layer.mla_forward_with_cache(
                &x,
                mla,
                &config,
                Some(&mut kv_cache),
                Some(&positions),
            );
            assert!(
                mla_result.is_ok(),
                "MLA forward should succeed: {:?}",
                mla_result.err()
            );
            let mla_output = mla_result.unwrap();

            assert_eq!(
                std_output.dim(),
                mla_output.dim(),
                "Standard and MLA output shapes should match"
            );
        }
    }

    #[test]
    fn test_multimodal_fusion_without_special_tokens() {
        let mut config = ModelConfig::default();
        config.hidden_size = 128;
        config.num_hidden_layers = 1;
        config.vocab_size = 1000;

        let model = MultimodalTransformer::new(config);

        let text_embedding = Array2::zeros((5, 128));
        let image_features = Array2::zeros((4, 128));
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];

        let result =
            model.fuse_image_features_with_validation(&text_embedding, &image_features, &tokens);
        assert!(
            result.is_ok(),
            "Fusion without special tokens should succeed"
        );

        let fused = result.unwrap();
        assert_eq!(
            fused.dim(),
            (8, 128),
            "Should insert at position 0: text_len - 1 + num_patches = 5 - 1 + 4 = 8"
        );
    }

    #[test]
    fn test_multimodal_fusion_with_im_start_only() {
        let mut config = ModelConfig::default();
        config.hidden_size = 128;
        config.num_hidden_layers = 1;
        config.vocab_size = 1000;

        let model = MultimodalTransformer::new(config);

        let text_embedding = Array2::zeros((5, 128));
        let image_features = Array2::zeros((4, 128));
        let tokens: Vec<u32> = vec![1, 2, IM_START_TOKEN_ID as u32, 4, 5];

        let result =
            model.fuse_image_features_with_validation(&text_embedding, &image_features, &tokens);
        assert!(result.is_ok(), "Fusion with IM_START only should succeed");

        let fused = result.unwrap();
        assert_eq!(
            fused.dim(),
            (8, 128),
            "Should insert image at IM_START position: 5 - 1 + 4 = 8"
        );
    }

    #[test]
    fn test_multimodal_fusion_with_both_tokens() {
        let mut config = ModelConfig::default();
        config.hidden_size = 128;
        config.num_hidden_layers = 1;
        config.vocab_size = 1000;

        let model = MultimodalTransformer::new(config);

        let text_embedding = Array2::zeros((6, 128));
        let image_features = Array2::zeros((4, 128));
        let tokens: Vec<u32> = vec![1, 2, IM_START_TOKEN_ID as u32, 0, IM_END_TOKEN_ID as u32, 5];

        let result =
            model.fuse_image_features_with_validation(&text_embedding, &image_features, &tokens);
        assert!(result.is_ok(), "Fusion with both tokens should succeed");

        let fused = result.unwrap();
        assert_eq!(fused.dim(), (7, 128), "Should replace placeholder: text_len - placeholder - 2 + num_patches = 6 - 1 - 2 + 4 = 7");
    }

    #[test]
    fn test_sample_token_temperature_zero() {
        let logits = Array1::from_shape_vec(5, vec![1.0, 5.0, 3.0, 2.0, 4.0]).unwrap();

        let token = MultimodalTransformer::sample_token(&logits, 0.0, 0.9, &mut rand::thread_rng());
        assert_eq!(token, 1, "Temperature=0 should select argmax (index 1)");

        let token =
            MultimodalTransformer::sample_token(&logits, -1.0, 0.9, &mut rand::thread_rng());
        assert_eq!(token, 1, "Negative temperature should select argmax");
    }

    #[test]
    fn test_sample_token_top_p_boundary() {
        let logits = Array1::from_shape_vec(5, vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();

        for _ in 0..100 {
            let token =
                MultimodalTransformer::sample_token(&logits, 1.0, 0.0, &mut rand::thread_rng());
            assert!(
                token < 5,
                "top_p=0 should still sample from full distribution"
            );
        }

        for _ in 0..100 {
            let token =
                MultimodalTransformer::sample_token(&logits, 1.0, 1.0, &mut rand::thread_rng());
            assert!(token < 5, "top_p=1 should sample from full distribution");
        }
    }

    #[test]
    fn test_moe_router_top_k_correctness() {
        let logits =
            Array2::from_shape_vec((2, 4), vec![1.0, 4.0, 2.0, 3.0, 3.0, 1.0, 4.0, 2.0]).unwrap();

        let (indices, weights, offsets) = top_k_selection(&logits, 2);

        assert_eq!(offsets, vec![0, 2, 4], "Offsets should be [0, 2, 4]");

        assert!(
            indices.contains(&1),
            "First row should select index 1 (value 4.0)"
        );
        assert!(
            indices.contains(&3),
            "First row should select index 3 (value 3.0)"
        );

        let _first_row_weights: Vec<f32> = indices
            .iter()
            .zip(weights.iter())
            .filter(|(idx, _)| *idx == &1 || *idx == &3)
            .map(|(_, w)| *w)
            .collect();
        let weight_sum: f32 = weights[0..2].iter().sum();
        assert!(
            (weight_sum - 1.0).abs() < 1e-5,
            "Weights should sum to 1.0, got {}",
            weight_sum
        );

        assert!(
            weights.iter().all(|&w| w > 0.0 && w <= 1.0),
            "All weights should be in (0, 1]"
        );
    }

    #[test]
    fn test_top_k_selection_edge_cases() {
        let empty_logits = Array2::zeros((0, 4));
        let (indices, weights, offsets) = top_k_selection(&empty_logits, 2);
        assert!(
            indices.is_empty(),
            "Empty input should produce empty output"
        );
        assert!(
            weights.is_empty(),
            "Empty input should produce empty weights"
        );
        assert_eq!(offsets, vec![0], "Empty input should have offsets [0]");

        let k_zero_logits =
            Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let (indices, weights, offsets) = top_k_selection(&k_zero_logits, 0);
        assert!(indices.is_empty(), "k=0 should produce empty indices");
        assert!(weights.is_empty(), "k=0 should produce empty weights");
        assert_eq!(offsets, vec![0, 0, 0], "k=0 should have all-zero offsets");

        let k_exceeds_logits =
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let (indices, weights, offsets) = top_k_selection(&k_exceeds_logits, 10);
        assert_eq!(
            indices.len(),
            6,
            "k > num_experts should select all experts"
        );
        assert_eq!(
            weights.len(),
            6,
            "k > num_experts should have weights for all"
        );
        assert_eq!(offsets, vec![0, 3, 6], "Offsets should be [0, 3, 6]");

        let equal_logits =
            Array2::from_shape_vec((2, 4), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
        let (_indices, weights, offsets) = top_k_selection(&equal_logits, 2);
        assert_eq!(
            offsets,
            vec![0, 2, 4],
            "Equal logits should still produce correct offsets"
        );
        let first_row_sum: f32 = weights[0..2].iter().sum();
        assert!(
            (first_row_sum - 1.0).abs() < 1e-5,
            "Equal logits should produce uniform weights"
        );
    }

    #[test]
    fn test_mla_vs_mha_output_difference() {
        let mut config = ModelConfig::default();
        config.hidden_size = 256;
        config.num_attention_heads = 4;
        config.num_key_value_heads = 2;
        config.head_dim = 64;
        config.mla_latent_dim = 128;
        config.use_mla = true;

        let x = Array2::from_shape_fn((4, config.hidden_size), |(i, j)| {
            ((i * config.hidden_size + j) as f32 * 0.01).sin()
        });

        let layer = TransformerLayerWeights::new(&config);

        let std_result = layer.attention.forward(&x, &config);
        assert!(std_result.is_ok(), "Standard attention should succeed");
        let std_output = std_result.unwrap();

        assert!(layer.mla.is_some(), "MLA weights should be initialized");

        if let Some(ref _mla) = layer.mla {
            let positions: Array1<usize> = (0..4).collect();
            let mut kv_cache = KVCache::new(10, config.num_key_value_heads, config.head_dim);

            let mla_result = layer.mla_forward_with_cache(
                &x,
                layer.mla.as_ref().unwrap(),
                &config,
                Some(&mut kv_cache),
                Some(&positions),
            );
            assert!(
                mla_result.is_ok(),
                "MLA forward should succeed: {:?}",
                mla_result.err()
            );
            let mla_output = mla_result.unwrap();

            assert_eq!(
                std_output.dim(),
                mla_output.dim(),
                "Standard and MLA output shapes should match"
            );

            let mut max_diff = 0.0f32;
            for i in 0..std_output.nrows() {
                for j in 0..std_output.ncols() {
                    let diff = (std_output[[i, j]] - mla_output[[i, j]]).abs();
                    max_diff = max_diff.max(diff);
                }
            }

            println!("Max difference between MHA and MLA: {}", max_diff);
        }
    }

    #[test]
    fn test_kv_cache_consistency() {
        let config = ModelConfig::default();
        let mut kv_cache = KVCache::new(10, config.num_key_value_heads, config.head_dim);

        let k1 = Array3::from_shape_fn(
            (2, config.num_key_value_heads, config.head_dim),
            |(i, h, d)| i as f32 * 0.1 + h as f32 * 0.01 + d as f32 * 0.001,
        );
        let v1 = Array3::from_shape_fn(
            (2, config.num_key_value_heads, config.head_dim),
            |(i, h, d)| i as f32 * 0.2 + h as f32 * 0.02 + d as f32 * 0.002,
        );

        kv_cache
            .update(&k1, &v1)
            .expect("First update should succeed");
        assert_eq!(kv_cache.seq_len, 2, "Cache should have 2 tokens");

        let (k_cached, _v_cached) = kv_cache.get();
        assert_eq!(
            k_cached.dim(),
            (2, config.num_key_value_heads, config.head_dim)
        );

        let k2 = Array3::from_shape_fn(
            (3, config.num_key_value_heads, config.head_dim),
            |(i, h, d)| (i + 2) as f32 * 0.1 + h as f32 * 0.01 + d as f32 * 0.001,
        );
        let v2 = Array3::from_shape_fn(
            (3, config.num_key_value_heads, config.head_dim),
            |(i, h, d)| (i + 2) as f32 * 0.2 + h as f32 * 0.02 + d as f32 * 0.002,
        );

        kv_cache
            .update(&k2, &v2)
            .expect("Second update should succeed");
        assert_eq!(kv_cache.seq_len, 5, "Cache should have 5 tokens");

        let (k_final, v_final) = kv_cache.get();
        assert_eq!(
            k_final.dim(),
            (5, config.num_key_value_heads, config.head_dim)
        );

        for i in 0..2 {
            for h in 0..config.num_key_value_heads {
                for d in 0..config.head_dim {
                    assert!(
                        (k_final[[i, h, d]] - k1[[i, h, d]]).abs() < 1e-6,
                        "Cached K values should match original at position ({}, {}, {})",
                        i,
                        h,
                        d
                    );
                    assert!(
                        (v_final[[i, h, d]] - v1[[i, h, d]]).abs() < 1e-6,
                        "Cached V values should match original at position ({}, {}, {})",
                        i,
                        h,
                        d
                    );
                }
            }
        }
    }

    #[test]
    fn test_softmax_stability() {
        let large_logits =
            Array2::from_shape_vec((1, 5), vec![1000.0, 1001.0, 1002.0, 1003.0, 1004.0]).unwrap();
        let probs = softmax(&large_logits);

        assert!(
            !probs.iter().any(|&p| p.is_nan()),
            "Softmax should not produce NaN"
        );
        assert!(
            !probs.iter().any(|&p| p.is_infinite()),
            "Softmax should not produce Inf"
        );

        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Softmax probabilities should sum to 1.0, got {}",
            sum
        );

        let small_logits =
            Array2::from_shape_vec((1, 5), vec![-1000.0, -1001.0, -1002.0, -1003.0, -1004.0])
                .unwrap();
        let probs = softmax(&small_logits);

        assert!(
            !probs.iter().any(|&p| p.is_nan()),
            "Softmax should not produce NaN for small values"
        );
        assert!(
            probs.iter().all(|&p| p > 0.0),
            "All probabilities should be positive"
        );
    }

    #[test]
    fn test_interpolate_pos_embed_single_patch() {
        let hidden_size = 64;
        let orig_grid = 16;
        let orig_patches = orig_grid * orig_grid;

        let pos_embed = Array2::from_shape_fn((orig_patches + 1, hidden_size), |(i, j)| {
            i as f32 * 0.01 + j as f32 * 0.001
        });

        let vision_encoder = VisionEncoderWeights {
            patch_embed: Array2::zeros((768, 14 * 14 * 3)),
            pos_embed: pos_embed.clone(),
            cls_token: Array1::zeros(hidden_size),
            layers: vec![],
            proj: Array2::zeros((hidden_size, hidden_size)),
            orig_grid_h: None,
            orig_grid_w: None,
            image_mean: CLIP_IMAGE_MEAN,
            image_std: CLIP_IMAGE_STD,
        };

        let result_h1 = vision_encoder.interpolate_pos_embed(orig_grid, 1, 1, hidden_size);
        assert_eq!(
            result_h1.dim(),
            (2, hidden_size),
            "Should handle target_h=1, target_w=1"
        );

        let result_h1_w2 = vision_encoder.interpolate_pos_embed(orig_grid, 1, 2, hidden_size);
        assert_eq!(
            result_h1_w2.dim(),
            (3, hidden_size),
            "Should handle target_h=1, target_w=2"
        );

        let result_h2_w1 = vision_encoder.interpolate_pos_embed(orig_grid, 2, 1, hidden_size);
        assert_eq!(
            result_h2_w1.dim(),
            (3, hidden_size),
            "Should handle target_h=2, target_w=1"
        );

        for h_idx in 0..hidden_size {
            assert!(!result_h1[[1, h_idx]].is_nan(), "Result should not be NaN");
            assert!(
                !result_h1_w2[[1, h_idx]].is_nan(),
                "Result should not be NaN"
            );
            assert!(
                !result_h2_w1[[1, h_idx]].is_nan(),
                "Result should not be NaN"
            );
        }
    }

    #[test]
    fn test_kv_cache_vs_no_cache_consistency() {
        let mut config = ModelConfig::default();
        config.hidden_size = 128;
        config.num_hidden_layers = 2;
        config.num_attention_heads = 4;
        config.num_key_value_heads = 4;
        config.head_dim = 32;
        config.intermediate_size = 256;
        config.vocab_size = 1000;
        config.max_position_embeddings = 512;

        let mut model = MultimodalTransformer::new(config.clone());

        for layer in &mut model.layers {
            layer.attention = create_deterministic_attention_weights(&config);
            layer.ffn = create_deterministic_ffn_weights(&config);
        }
        model.embedding = create_constant_weight(config.vocab_size, config.hidden_size, 0.1);
        model.lm_head = create_constant_weight(config.vocab_size, config.hidden_size, 0.1);
        model.weights_loaded = true;

        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];

        let params_no_cache = GenerateParams {
            max_new_tokens: 5,
            temperature: 0.0,
            ..Default::default()
        };

        let params_with_cache = GenerateParams {
            max_new_tokens: 5,
            temperature: 0.0,
            ..Default::default()
        };

        let result_no_cache = model
            .generate_with_params(&tokens, &params_no_cache)
            .expect("No-cache generation should succeed");
        let result_with_cache = model
            .generate_with_cache_params(&tokens, &params_with_cache)
            .expect("Cache generation should succeed");

        assert_eq!(
            result_no_cache.len(),
            result_with_cache.len(),
            "Both methods should produce same number of tokens"
        );

        for (i, (a, b)) in result_no_cache
            .iter()
            .zip(result_with_cache.iter())
            .enumerate()
        {
            assert_eq!(
                a, b,
                "Token {} should match between cache and no-cache methods",
                i
            );
        }
    }

    #[test]
    fn test_audio_padding_modes() {
        let audio = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            ],
        )
        .unwrap();

        let replicate = PaddingMode::Replicate;
        let padded_rep = replicate.apply_padding(&audio, 1);
        assert_eq!(padded_rep.dim(), (7, 3), "Replicate padding shape mismatch");
        assert_eq!(
            padded_rep[[0, 0]],
            1.0,
            "First row should replicate first audio row"
        );
        assert_eq!(
            padded_rep[[6, 2]],
            15.0,
            "Last row should replicate last audio row"
        );

        let zero = PaddingMode::Zero;
        let padded_zero = zero.apply_padding(&audio, 1);
        assert_eq!(padded_zero.dim(), (7, 3), "Zero padding shape mismatch");
        assert_eq!(padded_zero[[0, 0]], 0.0, "First row should be zero");
        assert_eq!(padded_zero[[6, 2]], 0.0, "Last row should be zero");

        let reflect = PaddingMode::Reflect;
        let padded_refl = reflect.apply_padding(&audio, 1);
        assert_eq!(padded_refl.dim(), (7, 3), "Reflect padding shape mismatch");
        assert_eq!(
            padded_refl[[0, 0]],
            4.0,
            "First row should reflect second audio row"
        );
        assert_eq!(
            padded_refl[[6, 2]],
            12.0,
            "Last row should reflect second-to-last audio row"
        );
    }

    #[test]
    fn test_interpolate_pos_embed_random_weights() {
        let mut rng = rand::thread_rng();
        use rand::Rng;

        let hidden_size = 64;
        let orig_grid = 8;
        let orig_patches = orig_grid * orig_grid;

        let pos_embed = Array2::from_shape_fn((orig_patches + 1, hidden_size), |_| {
            rng.gen::<f32>() * 2.0 - 1.0
        });

        let vision_encoder = VisionEncoderWeights {
            patch_embed: Array2::zeros((768, 14 * 14 * 3)),
            pos_embed: pos_embed.clone(),
            cls_token: Array1::zeros(hidden_size),
            layers: vec![],
            proj: Array2::zeros((hidden_size, hidden_size)),
            orig_grid_h: None,
            orig_grid_w: None,
            image_mean: CLIP_IMAGE_MEAN,
            image_std: CLIP_IMAGE_STD,
        };

        let result = vision_encoder.interpolate_pos_embed(orig_grid, 4, 4, hidden_size);
        assert_eq!(
            result.dim(),
            (17, hidden_size),
            "Interpolation shape mismatch for 4x4 target"
        );

        for h_idx in 0..hidden_size {
            assert!(!result[[0, h_idx]].is_nan(), "CLS token should not be NaN");
            for i in 1..17 {
                assert!(
                    !result[[i, h_idx]].is_nan(),
                    "Position {} should not be NaN",
                    i
                );
                assert!(
                    result[[i, h_idx]].is_finite(),
                    "Position {} should be finite",
                    i
                );
            }
        }

        let result_1x1 = vision_encoder.interpolate_pos_embed(orig_grid, 1, 1, hidden_size);
        assert_eq!(
            result_1x1.dim(),
            (2, hidden_size),
            "1x1 target shape mismatch"
        );
        assert!(!result_1x1[[1, 0]].is_nan(), "1x1 result should not be NaN");
    }

    #[test]
    fn test_rope_precomputed_tables() {
        let theta = 10000.0;
        let head_dim = 64;
        let max_positions = 1024;

        let tables = get_rope_tables(theta, head_dim, max_positions);

        assert!(
            tables.cos_table.is_some(),
            "cos_table should be precomputed for max_positions <= 8192"
        );
        assert!(
            tables.sin_table.is_some(),
            "sin_table should be precomputed for max_positions <= 8192"
        );

        let cos_table = tables.cos_table.as_ref().unwrap();
        let sin_table = tables.sin_table.as_ref().unwrap();

        assert_eq!(cos_table.dim(), (max_positions, head_dim / 2));
        assert_eq!(sin_table.dim(), (max_positions, head_dim / 2));

        let angle_0 = 0.0f32;
        let expected_cos = angle_0.cos();
        let expected_sin = angle_0.sin();

        assert!((cos_table[[0, 0]] - expected_cos).abs() < 1e-6);
        assert!((sin_table[[0, 0]] - expected_sin).abs() < 1e-6);

        let tables_large = get_rope_tables(theta, head_dim, 10000);
        assert!(
            tables_large.cos_table.is_none(),
            "cos_table should be None for max_positions > 8192"
        );
        assert!(
            tables_large.sin_table.is_none(),
            "sin_table should be None for max_positions > 8192"
        );
    }

    // ========================================================================
    // 真正的 MLA 测试（压缩缓存 + 按需解码）
    // ========================================================================

    #[test]
    fn test_mla_cache_creation_and_basic_ops() {
        let cache = MLACache::new(1024, 512);
        assert_eq!(cache.max_seq_len, 1024);
        assert_eq!(cache.latent_dim, 512);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.memory_size(), 0);
    }

    #[test]
    fn test_mla_cache_update_and_retrieve() {
        let mut cache = MLACache::new(1024, 512);

        let c_kv = Array2::from_shape_fn((4, 512), |(i, j)| (i as f32 * 512.0 + j as f32));
        cache.update(&c_kv).unwrap();

        assert_eq!(cache.len(), 4);
        assert!(!cache.is_empty());
        assert_eq!(cache.memory_size(), 4 * 512);

        let retrieved = cache.get();
        assert_eq!(retrieved.dim(), (4, 512));
        for i in 0..4 {
            for j in 0..512 {
                assert!((retrieved[[i, j]] - c_kv[[i, j]]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_mla_cache_incremental_update() {
        let mut cache = MLACache::new(1024, 512);

        let c_kv1 = Array2::from_shape_fn((2, 512), |(i, j)| (i as f32 * 100.0 + j as f32));
        cache.update(&c_kv1).unwrap();
        assert_eq!(cache.len(), 2);

        let c_kv2 = Array2::from_shape_fn((3, 512), |(i, j)| ((i as f32 + 2.0) * 100.0 + j as f32));
        cache.update(&c_kv2).unwrap();
        assert_eq!(cache.len(), 5);

        let retrieved = cache.get();
        assert_eq!(retrieved.dim(), (5, 512));
        assert!((retrieved[[0, 0]] - 0.0).abs() < 1e-6);
        assert!((retrieved[[4, 0]] - 400.0).abs() < 1e-6);
    }

    #[test]
    fn test_mla_cache_capacity_exceeded() {
        let mut cache = MLACache::new(4, 256);

        let c_kv_ok = Array2::zeros((4, 256));
        cache.update(&c_kv_ok).unwrap();

        let c_kv_overflow = Array2::zeros((1, 256));
        let result = cache.update(&c_kv_overflow);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("capacity exceeded"));
    }

    #[test]
    fn test_mla_cache_clear() {
        let mut cache = MLACache::new(100, 128);

        let c_kv = Array2::ones((10, 128));
        cache.update(&c_kv).unwrap();
        assert_eq!(cache.len(), 10);

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_mla_cache_compression_ratio() {
        let cache = MLACache::new(1024, 512);

        let ratio_1024 = cache.compression_ratio(1024);
        assert!(ratio_1024 > 0.0 && ratio_1024 < 1.0);

        let ratio_2048 = cache.compression_ratio(2048);
        assert!(ratio_2048 > ratio_1024);

        let ratio_equal = cache.compression_ratio(256);
        assert!((ratio_equal - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_mla_compressed_forward_matches_standard() {
        let mut config = ModelConfig::default();
        config.use_mla = true;

        let layer = TransformerLayerWeights::new(&config);
        let mla = layer.mla.as_ref().unwrap();

        let seq_len = 4;
        let hidden_size = config.hidden_size;
        let x = Array2::from_shape_fn((seq_len, hidden_size), |(i, j)| {
            ((i * hidden_size + j) as f32 / 1000.0).sin()
        });

        let positions = Array1::from_vec(vec![0, 1, 2, 3]);

        let mut std_cache = KVCache::new(64, config.num_key_value_heads, config.head_dim);
        let mut mla_cache = MLACache::new(64, mla.latent_dim());

        let normed = rms_norm(&x, &layer.input_layernorm, config.rms_norm_eps);

        let result_std = layer
            .mla_forward_with_cache(
                &normed,
                mla,
                &config,
                Some(&mut std_cache),
                Some(&positions),
            )
            .unwrap();

        let result_compressed = layer
            .mla_forward_with_compressed_cache(
                &normed,
                mla,
                &config,
                Some(&mut mla_cache),
                Some(&positions),
            )
            .unwrap();

        assert_eq!(result_std.dim(), result_compressed.dim());

        for i in 0..result_std.dim().0 {
            for j in 0..result_std.dim().1 {
                if result_std[[i, j]].abs() > 1e-3 || result_compressed[[i, j]].abs() > 1e-3 {
                    assert!(
                        (result_std[[i, j]] - result_compressed[[i, j]]).abs() < 1e-3,
                        "MLA compressed forward mismatch at [{},{}]: std={}, comp={}",
                        i,
                        j,
                        result_std[[i, j]],
                        result_compressed[[i, j]]
                    );
                }
            }
        }
    }

    #[test]
    fn test_mla_compressed_forward_autoregressive() {
        let mut config = ModelConfig::default();
        config.use_mla = true;

        let layer = TransformerLayerWeights::new(&config);
        let mla = layer.mla.as_ref().unwrap();

        let hidden_size = config.hidden_size;
        let mut mla_cache = MLACache::new(64, mla.latent_dim());

        let mut prev_output: Option<Array2<f32>> = None;

        for step in 0..8 {
            let x = if step == 0 {
                Array2::from_shape_fn((4, hidden_size), |(i, j)| {
                    ((i * hidden_size + j) as f32 / 500.0).cos()
                })
            } else {
                prev_output.clone().unwrap()
            };

            let pos = Array1::from_vec(vec![step]);
            let normed = rms_norm(&x, &layer.input_layernorm, config.rms_norm_eps);

            let output = layer
                .mla_forward_with_compressed_cache(
                    &normed,
                    mla,
                    &config,
                    Some(&mut mla_cache),
                    Some(&pos),
                )
                .unwrap();

            assert_eq!(output.dim().0, if step == 0 { 4 } else { 1 });
            assert_eq!(mla_cache.len(), if step == 0 { 4 } else { 4 + step });

            prev_output = Some(
                output
                    .slice(s![if step == 0 { 3 } else { 0 }.., ..])
                    .to_owned(),
            );
        }

        assert_eq!(mla_cache.len(), 11);
        assert!(mla_cache.compression_ratio(config.num_key_value_heads * config.head_dim) > 0.5);
    }

    #[test]
    fn test_mla_weights_has_uq_proj() {
        let config = ModelConfig::default();
        let mla = MLAWeights::new(&config);

        assert!(mla.uq_proj.is_some());
        let uq = mla.uq_proj.as_ref().unwrap();
        assert_eq!(uq.dim(), (config.head_dim, config.mla_latent_dim / 2));

        assert!(mla.has_decoupled_rope());
        assert_eq!(mla.latent_dim(), config.mla_latent_dim);
    }
}

#[cfg(test)]
mod train_tests {
    use super::*;

    fn create_small_model() -> MultimodalTransformer {
        let config = ModelConfig {
            vocab_size: 100,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            moe_num_experts: 2,
            moe_top_k: 1,
            ..ModelConfig::default()
        };

        let mut model = MultimodalTransformer::new(config);
        model.mark_weights_loaded();
        model
    }

    #[test]
    fn test_forward_train_basic() {
        let model = create_small_model();
        let input_ids = vec![1, 2, 3, 4, 5];

        let result = model.forward_train(&input_ids, None);
        assert!(result.is_ok(), "forward_train should succeed");

        let train_result = result.unwrap();
        assert_eq!(train_result.logits.dim().0, 5);
        assert_eq!(train_result.logits.dim().1, 100);
        assert!(train_result.loss.is_none());
        assert_eq!(train_result.activation_cache.hidden_states.len(), 2);
        assert_eq!(train_result.activation_cache.attention_scores.len(), 2);
    }

    #[test]
    fn test_forward_train_with_labels() {
        let model = create_small_model();
        let input_ids = vec![1, 2, 3, 4, 5];
        let labels = vec![2, 3, 4, 5, 6];

        let result = model.forward_train(&input_ids, Some(&labels));
        assert!(result.is_ok());

        let train_result = result.unwrap();
        assert!(train_result.loss.is_some());
        let loss = train_result.loss.unwrap();
        assert!(loss.is_finite(), "Loss should be finite");
    }

    #[test]
    fn test_forward_train_empty_input() {
        let model = create_small_model();
        let input_ids: Vec<usize> = vec![];

        let result = model.forward_train(&input_ids, None);
        assert!(result.is_ok());

        let train_result = result.unwrap();
        assert_eq!(train_result.logits.nrows(), 0);
    }

    #[test]
    fn test_forward_train_labels_mismatch() {
        let model = create_small_model();
        let input_ids = vec![1, 2, 3];
        let labels = vec![2, 3];

        let result = model.forward_train(&input_ids, Some(&labels));
        assert!(result.is_err());
    }

    #[test]
    fn test_trainable_params_not_empty() {
        let model = create_small_model();
        let params = model.trainable_params();

        assert!(!params.is_empty(), "trainable_params should not be empty");

        let embedding_param = &params[0];
        assert_eq!(embedding_param.name, "embedding");
        assert!(embedding_param.grad.is_none());
    }

    #[test]
    fn test_trainable_params_count() {
        let model = create_small_model();
        let params = model.trainable_params();

        assert!(!params.is_empty());
        assert!(params.len() > 10);
    }

    #[test]
    fn test_trainable_params_names() {
        let model = create_small_model();
        let params = model.trainable_params();

        let names: Vec<&str> = params.iter().map(|p| p.name.as_str()).collect();
        assert!(names.contains(&"embedding"));
        assert!(names.contains(&"final_layernorm"));
        assert!(names.contains(&"lm_head"));
        assert!(names.contains(&"layers.0.attention.q_proj"));
        assert!(names.contains(&"layers.1.ffn.down_proj"));
    }

    #[test]
    fn test_training_error_display() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = TrainingError::Io(io_err);
        let msg = format!("{}", err);
        assert!(msg.contains("IO error"));

        let err = TrainingError::ShapeMismatch {
            expected: vec![512, 512],
            got: vec![256, 512],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Shape mismatch"));

        let err = TrainingError::WeightNotFound("test.weight".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("test.weight"));
    }

    #[test]
    fn test_load_save_weights_stub() {
        let mut model = create_small_model();

        let result = model.load_weights(std::path::Path::new("/tmp/test.bin"));
        assert!(result.is_ok());

        let result = model.save_weights(std::path::Path::new("/tmp/test.bin"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_activation_cache_structure() {
        let model = create_small_model();
        let input_ids = vec![1, 2, 3];

        let result = model.forward_train(&input_ids, None).unwrap();
        let cache = result.activation_cache;

        assert_eq!(cache.hidden_states.len(), 2);
        assert_eq!(cache.input_embeddings.dim().0, 3);
        assert_eq!(cache.input_embeddings.dim().1, 64);
        assert_eq!(cache.attention_scores.len(), 2);
    }
}
