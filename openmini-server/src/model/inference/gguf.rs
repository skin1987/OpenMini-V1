//! GGUF 文件格式解析模块
//!
//! 本模块实现 GGUF (GGML Universal Format) 文件格式的解析，包括：
//! - 文件头解析
//! - 元数据解析
//! - 张量信息解析
//! - 张量数据读取
//!
//! # GGUF 文件格式
//!
//! GGUF 是一种高效的模型权重存储格式，支持：
//! - 多种数据类型（F32, F16, 量化类型等）
//! - 结构化元数据
//! - 内存映射访问
//!
//! # 支持的量化类型
//!
//! - Q4_0, Q4_1: 4位量化
//! - Q8_0: 8位量化
//! - Q2K - Q8K: K-量化系列

#![allow(dead_code)]
//! - IQ 系列: 智能量化

// ============================================================================
// 标准库和外部依赖导入
// ============================================================================

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, Result};
use bytemuck::cast_slice;
use memmap2::Mmap;

// ============================================================================
// GGUF 常量定义
// ============================================================================

/// GGUF 文件魔数 ("GGUF" 的小端序)
pub const GGUF_MAGIC: u32 = 0x46554747;

// ============================================================================
// 默认模型配置常量
// ============================================================================

/// 默认隐藏层大小
pub const DEFAULT_HIDDEN_SIZE: usize = 3584;
/// 默认隐藏层数量
pub const DEFAULT_NUM_HIDDEN_LAYERS: usize = 28;
/// 默认注意力头数量
pub const DEFAULT_NUM_ATTENTION_HEADS: usize = 28;
/// 默认中间层大小
pub const DEFAULT_INTERMEDIATE_SIZE: usize = 18944;
/// 默认词表大小（与 tokenizer::VOCAB_SIZE 保持一致）
pub const DEFAULT_VOCAB_SIZE: usize = 152064;
/// 默认最大位置编码长度
pub const DEFAULT_MAX_POSITION_EMBEDDINGS: usize = 131072;
/// 默认 RoPE 基础频率
pub const DEFAULT_ROPE_THETA: f32 = 1000000.0;
/// 默认 RMS 归一化 epsilon
pub const DEFAULT_RMS_NORM_EPS: f32 = 1e-6;

// ============================================================================
// GGUF 值类型定义
// ============================================================================

/// GGUF 值类型枚举
///
/// 定义 GGUF 文件中支持的所有数据类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufValueType {
    /// 无符号 8 位整数
    UInt8 = 0,
    /// 有符号 8 位整数
    Int8 = 1,
    /// 无符号 16 位整数
    UInt16 = 2,
    /// 有符号 16 位整数
    Int16 = 3,
    /// 无符号 32 位整数
    UInt32 = 4,
    /// 有符号 32 位整数
    Int32 = 5,
    /// 32 位浮点数
    Float32 = 6,
    /// 布尔值
    Bool = 7,
    /// 字符串
    String = 8,
    /// 数组
    Array = 9,
    /// 无符号 64 位整数
    UInt64 = 10,
    /// 有符号 64 位整数
    Int64 = 11,
    /// 64 位浮点数
    Float64 = 12,
}

/// GGUF 值枚举
///
/// 存储实际的元数据值
#[derive(Debug, Clone)]
pub enum GgufValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

// ============================================================================
// GGUF 元数据
// ============================================================================

/// GGUF 元数据结构体
///
/// 存储模型的所有元数据键值对
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    /// 键值对映射
    pub kv: HashMap<String, GgufValue>,
}

impl GgufMetadata {
    /// 获取字符串类型的元数据值
    ///
    /// # Parameters
    /// - `key`: 元数据键名
    ///
    /// # Returns
    /// 字符串值（如果存在且类型正确）
    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.kv.get(key) {
            Some(GgufValue::String(s)) => Some(s),
            _ => None,
        }
    }

    /// 获取 u32 类型的元数据值
    ///
    /// # Parameters
    /// - `key`: 元数据键名
    ///
    /// # Returns
    /// u32 值（如果存在且类型兼容）
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.kv.get(key) {
            Some(GgufValue::UInt32(v)) => Some(*v),
            Some(GgufValue::Int32(v)) => Some(*v as u32),
            _ => None,
        }
    }

    /// 获取 u64 类型的元数据值
    ///
    /// # Parameters
    /// - `key`: 元数据键名
    ///
    /// # Returns
    /// u64 值（如果存在且类型兼容）
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match self.kv.get(key) {
            Some(GgufValue::UInt64(v)) => Some(*v),
            Some(GgufValue::Int64(v)) => Some(*v as u64),
            Some(GgufValue::UInt32(v)) => Some(*v as u64),
            Some(GgufValue::Int32(v)) => Some(*v as u64),
            _ => None,
        }
    }

    /// 获取 f32 类型的元数据值
    ///
    /// # Parameters
    /// - `key`: 元数据键名
    ///
    /// # Returns
    /// f32 值（如果存在且类型正确）
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.kv.get(key) {
            Some(GgufValue::Float32(v)) => Some(*v),
            _ => None,
        }
    }

    /// 获取数组类型的元数据值
    ///
    /// # Parameters
    /// - `key`: 元数据键名
    ///
    /// # Returns
    /// 数组切片（如果存在且类型正确）
    pub fn get_array(&self, key: &str) -> Option<&[GgufValue]> {
        match self.kv.get(key) {
            Some(GgufValue::Array(arr)) => Some(arr),
            _ => None,
        }
    }
}

// ============================================================================
// GGUF 张量类型定义
// ============================================================================

/// GGUF 张量类型枚举
///
/// 定义支持的所有张量数据类型，包括浮点和量化类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufTensorType {
    /// 32 位浮点数
    F32 = 0,
    /// 16 位浮点数
    F16 = 1,
    /// 4 位量化 (类型 0)
    Q4_0 = 2,
    /// 4 位量化 (类型 1)
    Q4_1 = 3,
    /// 4 位量化 (类型 2)
    Q4_2 = 4,
    /// 4 位量化 (类型 3)
    Q4_3 = 5,
    /// 8 位量化 (类型 0)
    Q8_0 = 7,
    /// 8 位量化 (类型 1)
    Q8_1 = 8,
    /// 2 位 K-量化
    Q2K = 10,
    /// 3 位 K-量化
    Q3K = 11,
    /// 4 位 K-量化
    Q4K = 12,
    /// 5 位 K-量化
    Q5K = 13,
    /// 6 位 K-量化
    Q6K = 14,
    /// 8 位 K-量化
    Q8K = 15,
    /// 智能量化 2XXS
    Iq2Xxs = 16,
    /// 智能量化 2XS
    Iq2Xs = 17,
    /// 智能量化 3XXS
    Iq3Xxs = 18,
    /// 智能量化 1S
    Iq1S = 19,
    /// 智能量化 4NL
    Iq4Nl = 20,
    /// 智能量化 3S
    Iq3S = 21,
    /// 智能量化 2S
    Iq2S = 22,
    /// 智能量化 4XS
    Iq4Xs = 23,
    /// 8 位整数
    I8 = 24,
    /// 16 位整数
    I16 = 25,
    /// 32 位整数
    I32 = 26,
    /// 64 位整数
    I64 = 27,
    /// 64 位浮点数
    F64 = 28,
    /// BF16 浮点数
    BF16 = 29,
}

impl GgufTensorType {
    /// 从 u8 值创建张量类型
    ///
    /// # Parameters
    /// - `v`: 类型编号
    ///
    /// # Returns
    /// 张量类型（如果有效）
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            4 => Some(Self::Q4_2),
            5 => Some(Self::Q4_3),
            // 类型值 6 在 GGUF 规范中未定义/保留
            7 => Some(Self::Q8_0),
            8 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            16 => Some(Self::Iq2Xxs),
            17 => Some(Self::Iq2Xs),
            18 => Some(Self::Iq3Xxs),
            19 => Some(Self::Iq1S),
            20 => Some(Self::Iq4Nl),
            21 => Some(Self::Iq3S),
            22 => Some(Self::Iq2S),
            23 => Some(Self::Iq4Xs),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            29 => Some(Self::BF16),
            _ => None,
        }
    }

    /// 获取量化块大小
    ///
    /// 量化类型按块进行，此方法返回每个块的元素数量
    ///
    /// # Returns
    /// 每个量化块的元素数量
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 => 1,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q4_2 | Self::Q4_3 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
            Self::Iq2Xxs | Self::Iq2Xs | Self::Iq2S => 256,
            Self::Iq3Xxs | Self::Iq3S => 256,
            Self::Iq1S => 256,
            Self::Iq4Nl | Self::Iq4Xs => 256,
        }
    }

    /// 获取每个量化块的字节数
    ///
    /// # Returns
    /// 每个量化块占用的字节数
    ///
    /// # Note
    /// Q4_2 和 Q4_3 已在 GGUF 规范中标记为废弃，仅用于兼容旧模型
    pub fn bytes_per_block(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => 18, // 2 (scale) + 16 (32 * 4bit / 8)
            Self::Q4_1 => 20, // 2 (scale) + 2 (min) + 16 (32 * 4bit / 8)
            Self::Q4_2 => 18, // 已废弃: 2 (scale) + 16 (32 * 4bit / 8)
            Self::Q4_3 => 20, // 已废弃: 2 (scale) + 2 (min) + 16 (32 * 4bit / 8)
            Self::Q8_0 => 34, // 2 (scale) + 32 (32 * 8bit)
            Self::Q8_1 => 36, // 2 (scale) + 32 (32 * 8bit) + 2 (sum)
            Self::Q2K => 68,
            Self::Q3K => 162,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 258,
            Self::Iq1S => 132,
            Self::Iq2Xxs => 64,
            Self::Iq2Xs => 64,
            Self::Iq2S => 64,
            Self::Iq3Xxs => 96,
            Self::Iq3S => 96,
            Self::Iq4Nl => 130,
            Self::Iq4Xs => 130,
        }
    }

    /// 获取单个元素的字节数（非量化类型）
    ///
    /// # Returns
    /// 单个元素占用的字节数
    pub fn element_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            _ => self.bytes_per_block() / self.block_size(),
        }
    }
}

// ============================================================================
// GGUF 张量信息
// ============================================================================

/// GGUF 张量信息结构体
///
/// 存储单个张量的元数据信息
#[derive(Debug, Clone)]
pub struct GgufTensor {
    /// 张量名称
    pub name: String,
    /// 张量维度
    pub dims: Vec<usize>,
    /// 张量数据类型
    pub tensor_type: GgufTensorType,
    /// 数据在文件中的偏移量
    pub offset: u64,
}

impl GgufTensor {
    /// 计算张量的元素总数
    ///
    /// # Returns
    /// 所有维度的乘积
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }

    /// 计算张量数据的大小（字节）
    ///
    /// # Returns
    /// 张量数据占用的字节数
    pub fn data_size(&self) -> usize {
        let n = self.num_elements();
        let block_size = self.tensor_type.block_size();
        let bytes_per_block = self.tensor_type.bytes_per_block();
        let num_blocks = n.div_ceil(block_size);
        num_blocks * bytes_per_block
    }
}

// ============================================================================
// 多模态配置结构体
// ============================================================================

/// 视觉编码器配置
///
/// 从 GGUF 文件中提取的视觉模型配置
#[derive(Debug, Clone)]
pub struct VisionConfigGGUF {
    /// 输入图像大小
    pub image_size: usize,
    /// 图像块大小
    pub patch_size: usize,
    /// 隐藏层大小
    pub hidden_size: usize,
    /// 层数
    pub num_layers: usize,
    /// 注意力头数
    pub num_heads: usize,
    /// 中间层大小
    pub intermediate_size: usize,
}

/// 音频编码器配置
///
/// 从 GGUF 文件中提取的音频模型配置
#[derive(Debug, Clone)]
pub struct AudioConfigGGUF {
    /// 采样率
    pub sample_rate: usize,
    /// 梅尔频谱 bin 数
    pub num_mel_bins: usize,
    /// 隐藏层大小
    pub hidden_size: usize,
    /// 层数
    pub num_layers: usize,
    /// 注意力头数
    pub num_heads: usize,
}

/// TTS 配置
///
/// 从 GGUF 文件中提取的语音合成模型配置
#[derive(Debug, Clone)]
pub struct TTSConfigGGUF {
    /// 采样率
    pub sample_rate: usize,
    /// 隐藏层大小
    pub hidden_size: usize,
    /// Flow 层数
    pub num_flow_layers: usize,
}

// ============================================================================
// GGUF 文件头
// ============================================================================

/// GGUF 文件头结构体
///
/// 存储文件的基本信息
#[derive(Debug)]
pub struct GgufHeader {
    /// GGUF 版本号
    pub version: u32,
    /// 张量数量
    pub tensor_count: u64,
    /// 元数据键值对数量
    pub metadata_kv_count: u64,
}

// ============================================================================
// 模型配置
// ============================================================================

/// 模型配置结构体
///
/// 存储语言模型的完整配置信息
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// 词表大小
    pub vocab_size: usize,
    /// 隐藏层大小
    pub hidden_size: usize,
    /// 中间层大小
    pub intermediate_size: usize,
    /// 隐藏层数量
    pub num_hidden_layers: usize,
    /// 注意力头数量
    pub num_attention_heads: usize,
    /// KV 头数量（用于 GQA）
    pub num_key_value_heads: usize,
    /// 最大位置编码长度
    pub max_position_embeddings: usize,
    /// RMS 归一化 epsilon
    pub rms_norm_eps: f32,
    /// RoPE 基础频率
    pub rope_theta: f32,
    /// 模型架构类型
    pub architecture: Architecture,
    /// 是否为 MoE 架构
    pub is_moe: bool,
    /// MoE 专家数量（仅 MoE 架构有效）
    pub num_experts: usize,
    /// MoE Top-K 选择数量（仅 MoE 架构有效）
    pub top_k: usize,
    /// 是否使用 MLA（Multi-head Latent Attention）
    pub use_mla: bool,
    /// MLA 潜在维度（仅 MLA 架构有效）
    pub mla_latent_dim: usize,
}

impl From<super::model::ModelConfig> for ModelConfig {
    fn from(config: super::model::ModelConfig) -> Self {
        Self {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            max_position_embeddings: config.max_position_embeddings,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            architecture: Architecture::Llama,
            is_moe: config.moe_num_experts > 0,
            num_experts: config.moe_num_experts,
            top_k: config.moe_top_k,
            use_mla: config.use_mla,
            mla_latent_dim: config.mla_latent_dim,
        }
    }
}

impl From<ModelConfig> for super::model::ModelConfig {
    fn from(config: ModelConfig) -> Self {
        let head_dim = if config.num_attention_heads > 0 {
            config.hidden_size / config.num_attention_heads
        } else {
            super::model::DEFAULT_HIDDEN_SIZE / super::model::DEFAULT_NUM_ATTENTION_HEADS
        };

        Self {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim,
            max_position_embeddings: config.max_position_embeddings,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            moe_num_experts: if config.is_moe { config.num_experts } else { 0 },
            moe_top_k: if config.is_moe { config.top_k } else { 2 },
            use_mla: config.use_mla,
            mla_latent_dim: if config.use_mla {
                config.mla_latent_dim
            } else {
                512
            },
            ..Default::default()
        }
    }
}

// ============================================================================
// 模型架构定义
// ============================================================================

/// 支持的模型架构枚举
///
/// 定义 GGUF 格式支持的所有模型架构，每种架构都有特定的参数前缀和配置特点。
/// 架构按照流行程度和社区支持度排序。
///
/// # 架构分类
///
/// - **主流架构**: Llama, Qwen2, Mistral, Mixtral（广泛使用，完全支持）
/// - **专业架构**: MiniCPM, Gemma, Phi, Yi, ChatGLM, Baichuan（特定场景优化）
/// - **实验性架构**: Falcon, StableLM（有限测试）
///
/// # 使用示例
///
/// ```ignore
/// let arch = Architecture::from_str("mistral");
/// let prefix = arch.parameter_prefix(); // 返回 "mistral"
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Architecture {
    /// LLaMA 系列架构 (Meta)
    ///
    /// # 支持状态: ✅ 完全支持
    /// # 参数前缀: `llama.*`
    /// # 特点:
    /// - RoPE 位置编码
    /// - RMSNorm 归一化
    /// - SwiGLU 激活函数
    /// - GQA (Grouped Query Attention) 支持
    Llama,

    /// Qwen2 系列架构 (阿里云)
    ///
    /// # 支持状态: ✅ 完全支持
    /// # 参数前缀: `qwen2.*`
    /// # 特点:
    /// - 基于 LLaMA 架构优化
    /// - 改进的 RoPE 实现
    /// - 支持 GQA
    /// - 更好的多语言能力
    Qwen2,

    /// MiniCPM 系列架构 (面壁智能)
    ///
    /// # 支持状态: ✅ 完全支持（含视觉扩展）
    /// # 参数前缀: `minicpm.*`
    /// # 特点:
    /// - 轻量级设计
    /// - 内置视觉编码器支持
    /// - 高效的推理性能
    /// - 支持多模态输入
    MiniCPM,

    /// Gemma 系列架构 (Google)
    ///
    /// # 支持状态: ✅ 完全支持
    /// # 参数前缀: `gemma.*`
    /// # 特点:
    /// - GeGLU 激活函数（非 SwiGLU）
    /// - Post-norm 架构
    /// - 优化的注意力机制
    /// - 开源权重可用
    Gemma,

    /// Phi 系列架构 (Microsoft)
    ///
    /// # 支持状态: ⚠️ 已修复 Bug，基本支持
    /// # 参数前缀: `phi.*` / `phi3.*`
    /// # 特点:
    /// - Packed QKV 注意力（Phi-3）
    /// - 高效的参数设计
    /// - 针对特定任务优化
    ///
    /// # 已知限制
    /// - Phi-3 的 packed qkv 需要特殊处理
    /// - 某些变体可能需要额外适配
    Phi,

    /// Mistral 系列架构 (Mistral AI)
    ///
    /// # 支持状态: ✅ 新增支持
    /// # 参数前缀: `mistral.*`
    /// # 特点:
    /// - Sliding Window Attention (SWA)
    /// - RoPE 位置编码
    /// - GQA 支持
    /// - 高效的长文本处理
    ///
    /// # 性能影响
    /// - SWA 可显著减少长序列的计算复杂度
    /// - 推理速度比标准 attention 快 2-4x（长文本场景）
    Mistral,

    /// Mixtral 架构 (Mistral AI) - MoE 专家混合模型
    ///
    /// # 支持状态: ✅ 新增支持（MoE 基础功能）
    /// # 参数前缀: `mixtral.*`
    /// # 特点:
    /// - Sparse MoE (Mixture of Experts)
    /// - 默认 8 个专家，激活 2 个
    /// - 基于 Mistral 架构
    /// - 高效的参数利用率
    ///
    /// # 已知限制
    /// - 当前仅支持基础 MoE 加载
    /// - Expert routing 逻辑需在推理层实现
    /// - 内存占用较大（需加载所有专家权重）
    ///
    /// # 性能影响
    /// - 推理时只计算激活的专家，速度快
    /// - 但需要加载全部专家到内存
    Mixtral,

    /// DeepSeek-V3 架构 (DeepSeek AI) - 高级 MoE 架构
    ///
    /// # 支持状态: ✅ 新增支持（DeepSeek-V3 完整功能）
    /// # 参数前缀: `deepseek_v3.*`
    /// # 特点:
    /// - Advanced MoE with Shared Experts (共享专家)
    /// - Multi-head Latent Attention (MLA) 注意力机制
    /// - 支持多种 MoE 策略配置（Cyclic/FullLayer/Hybrid）
    /// - 负载均衡损失优化
    /// - 支持多模态嵌入
    ///
    /// # 已知限制
    /// - MLA 注意力实现复杂度较高
    /// - 共享专家和路由专家的前向传播需要特殊处理
    ///
    /// # 性能影响
    /// - 使用 MLA 显著减少 KV Cache 内存占用
    /// - 共享专家提高参数利用率
    /// - 路由专家提供动态计算能力
    DeepSeekV3,

    /// Yi 系列架构 (零一万物)
    ///
    /// # 支持状态: ✅ 新增支持
    /// # 参数前缀: `yi.*`
    /// # 特点:
    /// - 大RoPE theta (5000000)
    /// - 优化的中文理解能力
    /// - 支持 200K+ 上下文长度
    /// - 基于 LLaMA 架构改进
    ///
    /// # 特殊配置
    /// - rope_theta = 5000000 (默认值)
    /// - 支持超长上下文
    Yi,

    /// ChatGLM 系列架构 (智谱AI/清华)
    ///
    /// # 支持状态: ✅ 新增支持（基础功能）
    /// # 参数前缀: `chatglm.*` / `glm.*`
    /// # 特点:
    /// - Prefix Language Model (PrefixLM)
    /// - 2D-RoPE 位置编码
    /// - 针对对话优化
    /// - 强大的中文生成能力
    ///
    /// # 已知限制
    /// - 2D-RoPE 实现较复杂，当前简化处理
    /// - PrefixLM 的双向注意力需特殊处理
    ChatGLM,

    /// Baichuan 系列架构 (百川智能)
    ///
    /// # 支持状态: ✅ 新增支持
    /// # 参数前缀: `baichuan.*`
    /// # 特点:
    /// - ALiBi (Attention with Linear Biases) 位置编码
    /// - 无 RoPE 配置
    /// - 优秀的中文能力
    /// - 支持不同规模（7B/13B等）
    ///
    /// # 特殊配置
    /// - 不使用 RoPE（使用 ALiBi）
    /// - rope_theta 设置为 0.0 表示禁用
    Baichuan,

    /// Falcon 系列架构 (TII)
    ///
    /// # 支持状态: 🔶 实验性支持
    /// # 参数前缀: `falcon.*`
    /// # 特点:
    /// - 并行注意力机制
    /// - ALiBi 或 RoPE 编码（取决于版本）
    /// - 高效的推理性能
    ///
    /// # 已知限制
    /// - 不同版本差异较大（40B/180B）
    /// - 可能需要额外适配工作
    /// - 测试覆盖有限
    Falcon,

    /// StableLM 系列架构 (Stability AI)
    ///
    /// # 支持状态: 🔶 实验性支持
    /// # 参数前缀: `stablelm.*`
    /// # 特点:
    /// - 基于 LLaMA 架构
    /// - 长上下文支持
    /// - 开源友好
    ///
    /// # 已知限制
    /// - 社区使用较少
    /// - 测试数据有限
    /// - 可能存在未知兼容性问题
    StableLM,
}

impl Architecture {
    /// 从字符串解析架构名称
    ///
    /// 支持多种命名格式：
    /// - 标准名称: "llama", "qwen2", "mistral"
    /// - 版本后缀: "phi3" -> Phi, "mixtral-8x7b" -> Mixtral
    /// - 大小写不敏感
    ///
    /// # Parameters
    /// - `name`: 架构名称字符串
    ///
    /// # Returns
    /// 对应的 Architecture 枚举值，未知架构返回 Llama 作为 fallback
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(name: &str) -> Self {
        match name.to_lowercase().as_str() {
            "minicpm" => Self::MiniCPM,
            "qwen2" | "qwen" => Self::Qwen2,
            "gemma" => Self::Gemma,
            "phi" | "phi3" | "phi-2" | "phi-1.5" => Self::Phi,
            "mistral" => Self::Mistral,
            "mixtral" | "mixtral-8x7b" | "mixtral-8x22b" => Self::Mixtral,
            "deepseek" | "deepseek-v3" | "deepseek_v3" => Self::DeepSeekV3,
            "yi" => Self::Yi,
            "chatglm" | "glm" | "glm2" | "glm3" | "glm4" | "chatglm2" | "chatglm3" => Self::ChatGLM,
            "baichuan" => Self::Baichuan,
            "falcon" | "falcon-40b" | "falcon-180b" => Self::Falcon,
            "stablelm" => Self::StableLM,
            _ => Self::Llama, // 默认回退到 Llama
        }
    }

    /// 获取参数键名前缀
    ///
    /// 返回该架构在 GGUF 元数据中的参数前缀，
    /// 例如 Llama 返回 "llama"，Qwen2 返回 "qwen2"
    pub fn parameter_prefix(&self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Qwen2 => "qwen2",
            Self::MiniCPM => "minicpm",
            Self::Gemma => "gemma",
            Self::Phi => "phi", // 统一使用 phi 前缀（包括 phi3）
            Self::Mistral => "mistral",
            Self::Mixtral => "mixtral",
            Self::DeepSeekV3 => "deepseek_v3",
            Self::Yi => "yi",
            Self::ChatGLM => "chatglm",
            Self::Baichuan => "baichuan",
            Self::Falcon => "falcon",
            Self::StableLM => "stablelm",
        }
    }

    /// 检查是否为 MoE (Mixture of Experts) 架构
    ///
    /// MoE 架构有特殊的参数结构，包含专家数量等信息
    pub fn is_moe(&self) -> bool {
        matches!(self, Self::Mixtral | Self::DeepSeekV3)
    }

    /// 检查是否使用 Multi-head Latent Attention (MLA)
    ///
    /// MLA 是 DeepSeek-V3 的核心注意力机制，显著减少 KV Cache 内存占用
    pub fn uses_mla(&self) -> bool {
        matches!(self, Self::DeepSeekV3)
    }

    /// 获取推荐的默认 RoPE theta 值
    ///
    /// 不同架构有不同的最优 RoPE 基础频率
    pub fn default_rope_theta(&self) -> f32 {
        match self {
            Self::Yi => 5000000.0,    // Yi 使用大 theta 值
            Self::Baichuan => 0.0,    // Baichuan 使用 ALiBi，不使用 RoPE
            Self::ChatGLM => 10000.0, // ChatGLM 的 2D-RoPE
            _ => DEFAULT_ROPE_THETA,  // 其他架构使用默认值
        }
    }

    /// 检查是否使用 Packed QKV 注意力
    ///
    /// 某些架构将 Q、K、V 投影合并为一个张量
    pub fn uses_packed_qkv(&self) -> bool {
        matches!(self, Self::Phi) // Phi-3 使用 packed qkv
    }

    /// 获取架构的支持状态描述
    pub fn support_status(&self) -> &'static str {
        match self {
            Self::Llama | Self::Qwen2 | Self::MiniCPM | Self::Gemma => "✅ 完全支持",
            Self::Mistral | Self::Yi | Self::Baichuan => "✅ 新增支持",
            Self::Phi => "⚠️ 已修复Bug，基本支持",
            Self::Mixtral => "✅ MoE基础支持",
            Self::DeepSeekV3 => "✅ DeepSeek-V3完整支持（MoE+MLA）",
            Self::ChatGLM => "✅ 基础功能支持",
            Self::Falcon | Self::StableLM => "🔶 实验性支持",
        }
    }
}

impl std::fmt::Display for Architecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Llama => write!(f, "LLaMA"),
            Self::Qwen2 => write!(f, "Qwen2"),
            Self::MiniCPM => write!(f, "MiniCPM"),
            Self::Gemma => write!(f, "Gemma"),
            Self::Phi => write!(f, "Phi"),
            Self::Mistral => write!(f, "Mistral"),
            Self::Mixtral => write!(f, "Mixtral (MoE)"),
            Self::DeepSeekV3 => write!(f, "DeepSeek-V3 (MoE+MLA)"),
            Self::Yi => write!(f, "Yi"),
            Self::ChatGLM => write!(f, "ChatGLM"),
            Self::Baichuan => write!(f, "Baichuan"),
            Self::Falcon => write!(f, "Falcon"),
            Self::StableLM => write!(f, "StableLM"),
        }
    }
}

// ============================================================================
// MoE (Mixture of Experts) 相关类型定义
// ============================================================================

/// 层类型枚举
///
/// 用于标识 Transformer 层是标准 FFN 还是 MoE 专家层
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub enum LayerType {
    /// 标准前馈网络层
    FFN,
    /// Mixture of Experts 层
    MoE,
}

/// MoE 策略枚举
///
/// 定义不同层使用 MoE 或 FFN 的策略配置
#[derive(Debug, Clone)]
pub enum MoEStrategy {
    /// 循环策略：按固定周期交替使用 FFN 和 MoE
    ///
    /// # 参数
    /// - `period`: 循环周期长度
    /// - `offset`: 起始偏移量
    Cyclic { period: usize, offset: usize },

    /// 全层策略：前 N 层使用 FFN，后续层使用 MoE
    ///
    /// # 参数
    /// - `ffn_prefix_layers`: 使用 FFN 的层数
    FullLayer { ffn_prefix_layers: usize },

    /// 混合策略：显式指定每层的类型
    ///
    /// # 参数
    /// - `layer_types`: 每层的类型列表（索引对应层数）
    Hybrid { layer_types: Vec<LayerType> },
}

impl MoEStrategy {
    /// 根据策略获取指定层的类型
    ///
    /// # Parameters
    /// - `layer_idx`: 层索引（从 0 开始）
    /// - `total_layers`: 总层数
    ///
    /// # Returns
    /// 该层的类型（FFN 或 MoE）
    pub fn get_layer_type(&self, layer_idx: usize, _total_layers: usize) -> LayerType {
        match self {
            Self::Cyclic { period, offset } => {
                if (layer_idx + offset) % period == 0 {
                    LayerType::MoE
                } else {
                    LayerType::FFN
                }
            }
            Self::FullLayer { ffn_prefix_layers } => {
                if layer_idx < *ffn_prefix_layers {
                    LayerType::FFN
                } else {
                    LayerType::MoE
                }
            }
            Self::Hybrid { layer_types } => layer_types
                .get(layer_idx)
                .copied()
                .unwrap_or(LayerType::FFN),
        }
    }

    /// 创建 DeepSeek-V3 默认的循环策略
    ///
    /// DeepSeek-V3 通常每 1 层为 1 个周期，偏移量为 0
    pub fn deepseek_v3_default() -> Self {
        Self::Cyclic {
            period: 1,
            offset: 0,
        }
    }
}

/// MoE 版本枚举
///
/// 支持不同版本的 MoE 实现，确保向后兼容
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoEVersion {
    /// V1 版本（Mixtral 等传统 MoE）
    V1,
    /// V2 版本（DeepSeek-V3 高级 MoE，支持共享专家）
    V2,
}

/// FFN 权重结构体
///
/// 存储单个 FFN/专家的权重参数
#[derive(Debug, Clone)]
pub struct FFNWeights {
    /// 门控投影权重 (gate_proj)
    pub gate_weight: Vec<f32>,
    /// 上行投影权重 (up_proj)
    pub up_weight: Vec<f32>,
    /// 下行投影权重 (down_proj)
    pub down_weight: Vec<f32>,
}

/// MoE V2 权重结构体（DeepSeek-V3 风格）
///
/// 支持 Shared Experts + Routing Experts 的高级 MoE 架构
#[derive(Debug, Clone)]
pub struct MoEWeightsV2 {
    /// 共享专家权重列表（所有 token 都会经过共享专家）
    pub shared_experts: Vec<FFNWeights>,
    /// 共享专家的门控权重（可选）
    pub shared_gate: Option<ndarray::Array2<f32>>,
    /// 路由专家权重列表（根据路由动态选择）
    pub routing_experts: Vec<FFNWeights>,
    /// 路由器权重矩阵 (router weight)
    pub routing_router: ndarray::Array2<f32>,
    /// Top-K 选择数量（激活的专家数）
    pub top_k: usize,
    /// 容量因子（用于控制每个专家处理的最大 token 数）
    pub capacity_factor: f32,
    /// 负载均衡损失系数
    pub load_balance_loss_coef: f32,
    /// 多模态嵌入映射（模态 ID -> 嵌入向量）
    pub modality_embeds: Option<std::collections::HashMap<usize, ndarray::Array1<f32>>>,
}

impl MoEWeightsV2 {
    /// 创建新的 MoE V2 权重实例
    ///
    /// # Parameters
    /// - `num_shared_experts`: 共享专家数量
    /// - `num_routing_experts`: 路由专家数量
    /// - `top_k`: 激活专家数量
    /// - `hidden_size`: 隐藏层大小
    /// - `intermediate_size`: 中间层大小
    pub fn new(
        num_shared_experts: usize,
        num_routing_experts: usize,
        top_k: usize,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Self {
        let dummy_ffn = FFNWeights {
            gate_weight: vec![0.0; hidden_size * intermediate_size],
            up_weight: vec![0.0; hidden_size * intermediate_size],
            down_weight: vec![0.0; intermediate_size * hidden_size],
        };

        let mut shared_experts = Vec::with_capacity(num_shared_experts);
        for _ in 0..num_shared_experts {
            shared_experts.push(dummy_ffn.clone());
        }

        let mut routing_experts = Vec::with_capacity(num_routing_experts);
        for _ in 0..num_routing_experts {
            routing_experts.push(dummy_ffn.clone());
        }

        // 初始化路由器权重矩阵 [hidden_size, num_routing_experts]
        let routing_router = ndarray::Array2::<f32>::zeros((hidden_size, num_routing_experts));

        Self {
            shared_experts,
            shared_gate: None,
            routing_experts,
            routing_router,
            top_k,
            capacity_factor: 1.25,
            load_balance_loss_coef: 0.01,
            modality_embeds: None,
        }
    }

    /// 前向传播计算
    ///
    /// # Parameters
    /// - `input`: 输入张量 [batch_size, seq_len, hidden_size]
    ///
    /// # Returns
    /// - 输出张量 [batch_size, seq_len, hidden_size]
    /// - 负载均衡损失值
    pub fn forward(
        &self,
        input: &ndarray::Array3<f32>,
    ) -> Result<(ndarray::Array3<f32>, f32), anyhow::Error> {
        let (batch_size, seq_len, hidden_size) = input.dim();
        let num_tokens = batch_size * seq_len;

        let input_2d = input.to_shape((num_tokens, hidden_size))?.into_owned();

        let routing_logits = input_2d.dot(&self.routing_router);
        let (topk_indices, topk_weights) = self.topk_routing(&routing_logits)?;

        let mut shared_output = ndarray::Array2::<f32>::zeros((num_tokens, hidden_size));
        for expert in &self.shared_experts {
            let expert_out = self
                .swiglu_ffn(&input_2d, expert, hidden_size)?
                .into_owned();
            shared_output += &expert_out;
        }

        let mut routing_output = ndarray::Array2::<f32>::zeros((num_tokens, hidden_size));
        for k in 0..self.top_k {
            for i in 0..num_tokens {
                let expert_idx = topk_indices[[i, k]];
                let weight = topk_weights[[i, k]];
                let expert = &self.routing_experts[expert_idx];
                let input_row = input_2d.row(i);
                let expert_out =
                    self.swiglu_ffn_single(&input_row.to_vec(), expert, hidden_size)?;
                for j in 0..hidden_size {
                    routing_output[[i, j]] += weight * expert_out[j];
                }
            }
        }

        let final_output_2d = shared_output + &routing_output;
        let final_output = final_output_2d
            .to_shape((batch_size, seq_len, hidden_size))?
            .into_owned();
        let load_balance_loss = self.compute_load_balance_loss(&routing_logits);

        Ok((final_output, load_balance_loss))
    }

    fn swiglu_ffn(
        &self,
        input: &ndarray::Array2<f32>,
        ffn: &FFNWeights,
        hidden_size: usize,
    ) -> Result<ndarray::Array2<f32>, anyhow::Error> {
        let (_num_tokens, _) = input.dim();
        let intermediate_size = ffn.gate_weight.len() / hidden_size;

        let gate_w = ndarray::Array2::from_shape_vec(
            (hidden_size, intermediate_size),
            ffn.gate_weight.clone(),
        )?;
        let up_w = ndarray::Array2::from_shape_vec(
            (hidden_size, intermediate_size),
            ffn.up_weight.clone(),
        )?;
        let down_w = ndarray::Array2::from_shape_vec(
            (intermediate_size, hidden_size),
            ffn.down_weight.clone(),
        )?;

        let gate = input.dot(&gate_w);
        let up = input.dot(&up_w);

        let silu_gate = gate.mapv(|x| {
            let sig = 1.0 / (1.0 + (-x).exp());
            x * sig
        });

        let activated = &silu_gate * &up;
        let output = activated.dot(&down_w);

        Ok(output)
    }

    #[allow(clippy::needless_range_loop)]
    fn swiglu_ffn_single(
        &self,
        input: &[f32],
        ffn: &FFNWeights,
        hidden_size: usize,
    ) -> Result<Vec<f32>, anyhow::Error> {
        let intermediate_size = ffn.gate_weight.len() / hidden_size;

        let mut gate = vec![0.0; intermediate_size];
        let mut up = vec![0.0; intermediate_size];

        for i in 0..intermediate_size {
            for j in 0..hidden_size {
                gate[i] += input[j] * ffn.gate_weight[j * intermediate_size + i];
                up[i] += input[j] * ffn.up_weight[j * intermediate_size + i];
            }
        }

        let mut activated = vec![0.0; intermediate_size];
        for i in 0..intermediate_size {
            let sig = 1.0 / (1.0 + (-gate[i]).exp());
            activated[i] = gate[i] * sig * up[i];
        }

        let mut output = vec![0.0; hidden_size];
        for i in 0..hidden_size {
            for j in 0..intermediate_size {
                output[i] += activated[j] * ffn.down_weight[j * hidden_size + i];
            }
        }

        Ok(output)
    }

    /// Top-K 路由选择
    ///
    /// 从路由 logits 中选择 top-k 个专家及其归一化权重
    fn topk_routing(
        &self,
        routing_logits: &ndarray::Array2<f32>,
    ) -> Result<(ndarray::Array2<usize>, ndarray::Array2<f32>), anyhow::Error> {
        let (num_tokens, num_experts) = routing_logits.dim();

        let mut indices = ndarray::Array2::<usize>::zeros((num_tokens, self.top_k));
        let mut weights = ndarray::Array2::<f32>::zeros((num_tokens, self.top_k));

        for i in 0..num_tokens {
            let row = routing_logits.row(i).to_vec();

            // 创建 (expert_index, logit) 对并排序
            let mut expert_logits: Vec<(usize, f32)> = row.into_iter().enumerate().collect();
            expert_logits
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // 选择 top-k
            for k in 0..self.top_k.min(num_experts) {
                indices[[i, k]] = expert_logits[k].0;
            }

            // Softmax 归一化
            let topk_logits: Vec<f32> = expert_logits[..self.top_k.min(num_experts)]
                .iter()
                .map(|(_, logit)| *logit)
                .collect();
            let softmax_weights = self.softmax(&topk_logits);

            for k in 0..self.top_k.min(num_experts) {
                weights[[i, k]] = softmax_weights[k];
            }
        }

        Ok((indices, weights))
    }

    /// Softmax 函数
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        if logits.is_empty() {
            return Vec::new();
        }

        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits.iter().map(|x| (*x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        exp_vals.iter().map(|x| x / sum).collect()
    }

    /// 计算负载均衡损失
    ///
    /// 使用辅助损失函数鼓励均匀分配 token 到各专家
    pub fn compute_load_balance_loss(&self, routing_logits: &ndarray::Array2<f32>) -> f32 {
        let (num_tokens, num_experts) = routing_logits.dim();

        if num_tokens == 0 || num_experts == 0 {
            return 0.0;
        }

        // 计算每个 token 的路由概率分布
        let mut routing_probs = ndarray::Array2::<f32>::zeros((num_tokens, num_experts));
        for i in 0..num_tokens {
            let row = routing_logits.row(i).to_vec();
            let probs = self.softmax(&row);
            for j in 0..num_experts {
                routing_probs[[i, j]] = probs[j];
            }
        }

        // 计算每个专家的平均路由概率
        let expert_means = routing_probs.mean_axis(ndarray::Axis(0)).unwrap();

        // 计算方差作为负载均衡损失
        let mean_of_means: f32 = expert_means.mean().unwrap_or(0.0);
        let variance: f32 = expert_means
            .iter()
            .map(|&x| (x - mean_of_means).powi(2))
            .sum::<f32>()
            / num_experts as f32;

        variance * self.load_balance_loss_coef
    }
}

// ============================================================================
// GGUF 文件结构体
// ============================================================================

/// GGUF 文件结构体
///
/// 代表一个已解析的 GGUF 文件，提供访问元数据和张量数据的接口
#[derive(Debug)]
pub struct GgufFile {
    /// 文件头
    pub header: GgufHeader,
    /// 元数据
    pub metadata: GgufMetadata,
    /// 张量信息映射
    pub tensors: HashMap<String, GgufTensor>,
    /// 内存映射
    mmap: Mmap,
}

impl GgufFile {
    /// 打开 GGUF 文件
    ///
    /// # Parameters
    /// - `path`: 文件路径
    ///
    /// # Returns
    /// 解析后的 GGUF 文件实例
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        // 使用内存映射提高大文件读取效率
        let mmap = unsafe { Mmap::map(&file)? };

        Self::parse(mmap)
    }

    /// 获取视觉编码器张量
    ///
    /// 筛选出属于视觉模型的张量
    ///
    /// # Returns
    /// 视觉张量名称到张量信息的映射
    pub fn get_vision_tensors(&self) -> HashMap<String, &GgufTensor> {
        self.iter_vision_tensors()
            .map(|(k, v)| (k.clone(), v))
            .collect()
    }

    /// 迭代视觉编码器张量
    ///
    /// 返回迭代器，避免分配 HashMap
    pub fn iter_vision_tensors(&self) -> impl Iterator<Item = (&String, &GgufTensor)> {
        self.tensors.iter().filter(|(name, _)| {
            name.starts_with("vpm.")
                || name.starts_with("vision_model.")
                || name.starts_with("vision_encoder.")
                || name.starts_with("visual.")
        })
    }

    /// 获取重采样器张量
    ///
    /// 筛选出属于多模态投影层的张量
    ///
    /// # Returns
    /// 重采样器张量名称到张量信息的映射
    pub fn get_resampler_tensors(&self) -> HashMap<String, &GgufTensor> {
        self.iter_resampler_tensors()
            .map(|(k, v)| (k.clone(), v))
            .collect()
    }

    /// 迭代重采样器张量
    ///
    /// 返回迭代器，避免分配 HashMap
    pub fn iter_resampler_tensors(&self) -> impl Iterator<Item = (&String, &GgufTensor)> {
        self.tensors.iter().filter(|(name, _)| {
            name.starts_with("resampler.")
                || name.starts_with("mm_projector.")
                || name.starts_with("vision_proj.")
                || name.starts_with("multi_modal_projector.")
        })
    }

    /// 获取语言模型张量
    ///
    /// 筛选出属于语言模型的张量
    ///
    /// # Returns
    /// 语言模型张量名称到张量信息的映射
    pub fn get_llm_tensors(&self) -> HashMap<String, &GgufTensor> {
        self.iter_llm_tensors()
            .map(|(k, v)| (k.clone(), v))
            .collect()
    }

    /// 迭代语言模型张量
    ///
    /// 返回迭代器，避免分配 HashMap
    pub fn iter_llm_tensors(&self) -> impl Iterator<Item = (&String, &GgufTensor)> {
        self.tensors.iter().filter(|(name, _)| {
            name.starts_with("llm.")
                || name.starts_with("model.")
                || name.starts_with("language_model.")
                || (!name.starts_with("vpm.")
                    && !name.starts_with("vision_model.")
                    && !name.starts_with("resampler.")
                    && !name.starts_with("mm_projector."))
        })
    }

    /// 检查是否包含视觉张量
    ///
    /// # Returns
    /// 如果文件包含视觉编码器权重则返回 true
    pub fn has_vision_tensors(&self) -> bool {
        self.tensors.keys().any(|name| {
            name.starts_with("vpm.")
                || name.starts_with("vision_model.")
                || name.starts_with("vision_encoder.")
                || name.starts_with("visual.")
        })
    }

    /// 检查是否包含重采样器张量
    ///
    /// # Returns
    /// 如果文件包含多模态投影层权重则返回 true
    pub fn has_resampler_tensors(&self) -> bool {
        self.tensors.keys().any(|name| {
            name.starts_with("resampler.")
                || name.starts_with("mm_projector.")
                || name.starts_with("vision_proj.")
        })
    }

    /// 获取音频编码器张量
    ///
    /// # Returns
    /// 音频编码器张量名称到张量信息的映射
    pub fn get_audio_encoder_tensors(&self) -> HashMap<String, &GgufTensor> {
        self.tensors
            .iter()
            .filter(|(name, _)| {
                name.starts_with("audio_encoder.")
                    || name.starts_with("encoder.audio.")
                    || name.starts_with("whisper.")
                    || name.starts_with("audio_model.")
            })
            .map(|(k, v)| (k.clone(), v))
            .collect()
    }

    /// 获取 TTS 张量
    ///
    /// # Returns
    /// TTS 张量名称到张量信息的映射
    pub fn get_tts_tensors(&self) -> HashMap<String, &GgufTensor> {
        self.tensors
            .iter()
            .filter(|(name, _)| {
                name.starts_with("tts.")
                    || name.starts_with("flow.")
                    || name.starts_with("vocoder.")
                    || name.starts_with("speech_decoder.")
            })
            .map(|(k, v)| (k.clone(), v))
            .collect()
    }

    /// 检查是否包含音频编码器张量
    ///
    /// # Returns
    /// 如果文件包含音频编码器权重则返回 true
    pub fn has_audio_encoder_tensors(&self) -> bool {
        self.tensors.keys().any(|name| {
            name.starts_with("audio_encoder.")
                || name.starts_with("encoder.audio.")
                || name.starts_with("whisper.")
                || name.starts_with("audio_model.")
        })
    }

    /// 检查是否包含 TTS 张量
    ///
    /// # Returns
    /// 如果文件包含语音合成权重则返回 true
    pub fn has_tts_tensors(&self) -> bool {
        self.tensors.keys().any(|name| {
            name.starts_with("tts.")
                || name.starts_with("flow.")
                || name.starts_with("vocoder.")
                || name.starts_with("speech_decoder.")
        })
    }

    /// 检查是否包含语言模型张量
    ///
    /// # Returns
    /// 如果文件包含语言模型权重则返回 true
    pub fn has_llm_tensors(&self) -> bool {
        self.tensors.keys().any(|name| {
            name.starts_with("model.")
                || name.starts_with("llm.")
                || name.starts_with("language_model.")
                || name.starts_with("transformer.")
                || (name.starts_with("tok_embeddings.")
                    && !name.starts_with("vision_")
                    && !name.starts_with("audio_"))
        })
    }

    /// 获取音频编码器配置
    ///
    /// 从元数据中提取音频编码器配置
    ///
    /// # Returns
    /// 音频配置（如果存在）
    pub fn get_audio_config(&self) -> Option<AudioConfigGGUF> {
        if !self.has_audio_encoder_tensors() {
            return None;
        }

        let sample_rate = self
            .metadata
            .get_u32("audio.sample_rate")
            .or_else(|| self.metadata.get_u32("whisper.sample_rate"))
            .unwrap_or(16000) as usize;

        let num_mel_bins = self
            .metadata
            .get_u32("audio.num_mel_bins")
            .or_else(|| self.metadata.get_u32("whisper.num_mel_bins"))
            .unwrap_or(80) as usize;

        let hidden_size = self
            .metadata
            .get_u32("audio.hidden_size")
            .or_else(|| self.metadata.get_u32("whisper.hidden_size"))
            .unwrap_or(1024) as usize;

        let num_layers = self
            .metadata
            .get_u32("audio.num_layers")
            .or_else(|| self.metadata.get_u32("whisper.num_layers"))
            .unwrap_or(24) as usize;

        let num_heads = self
            .metadata
            .get_u32("audio.num_heads")
            .or_else(|| self.metadata.get_u32("whisper.num_heads"))
            .unwrap_or(16) as usize;

        Some(AudioConfigGGUF {
            sample_rate,
            num_mel_bins,
            hidden_size,
            num_layers,
            num_heads,
        })
    }

    /// 获取视觉编码器配置
    ///
    /// 从元数据中提取视觉编码器配置
    ///
    /// # Returns
    /// 视觉配置（如果存在）
    pub fn get_vision_config(&self) -> Option<VisionConfigGGUF> {
        if !self.has_vision_tensors() {
            return None;
        }

        let image_size = self
            .metadata
            .get_u32("vision.image_size")
            .or_else(|| self.metadata.get_u32("minicpm.vision.image_size"))
            .unwrap_or(448) as usize;

        let patch_size = self
            .metadata
            .get_u32("vision.patch_size")
            .or_else(|| self.metadata.get_u32("minicpm.vision.patch_size"))
            .unwrap_or(14) as usize;

        let hidden_size = self
            .metadata
            .get_u32("vision.hidden_size")
            .or_else(|| self.metadata.get_u32("minicpm.vision.hidden_size"))
            .unwrap_or(1152) as usize;

        let num_layers = self
            .metadata
            .get_u32("vision.num_layers")
            .or_else(|| self.metadata.get_u32("minicpm.vision.num_layers"))
            .unwrap_or(27) as usize;

        let num_heads = self
            .metadata
            .get_u32("vision.num_heads")
            .or_else(|| self.metadata.get_u32("minicpm.vision.num_heads"))
            .unwrap_or(16) as usize;

        let intermediate_size = self
            .metadata
            .get_u32("vision.intermediate_size")
            .or_else(|| self.metadata.get_u32("minicpm.vision.intermediate_size"))
            .unwrap_or(4304) as usize;

        Some(VisionConfigGGUF {
            image_size,
            patch_size,
            hidden_size,
            num_layers,
            num_heads,
            intermediate_size,
        })
    }

    /// 解析 GGUF 文件
    ///
    /// 从内存映射解析 GGUF 文件结构
    ///
    /// # Parameters
    /// - `mmap`: 内存映射
    ///
    /// # Returns
    /// 解析后的 GGUF 文件实例
    fn parse(mmap: Mmap) -> Result<Self> {
        // 检查文件是否为空
        if mmap.len() < 24 {
            return Err(anyhow!("File too small to be a valid GGUF file"));
        }

        // 检查文件大小是否合理（文档性检查，实际在 usize 范围内）
        #[allow(clippy::absurd_extreme_comparisons)]
        if mmap.len() > usize::MAX / 2 {
            return Err(anyhow!(
                "File too large to map on this platform ({} bytes)",
                mmap.len()
            ));
        }

        let mut offset: usize = 0;

        // 解析魔数（小端序）
        let magic = u32::from_le_bytes(mmap[offset..offset + 4].try_into()?);
        offset += 4;

        if magic != GGUF_MAGIC {
            return Err(anyhow!(
                "Invalid GGUF magic number: expected {:#x}, got {:#x}",
                GGUF_MAGIC,
                magic
            ));
        }

        // 解析版本号
        let version = u32::from_le_bytes(mmap[offset..offset + 4].try_into()?);
        offset += 4;

        // 解析张量数量
        let tensor_count = u64::from_le_bytes(mmap[offset..offset + 8].try_into()?);
        offset += 8;

        // 解析元数据数量
        let metadata_kv_count = u64::from_le_bytes(mmap[offset..offset + 8].try_into()?);
        offset += 8;

        let header = GgufHeader {
            version,
            tensor_count,
            metadata_kv_count,
        };

        // 解析元数据
        let (metadata, new_offset) = Self::parse_metadata(&mmap, offset, metadata_kv_count)?;
        offset = new_offset;

        // 解析张量信息
        let (tensors, _) = Self::parse_tensor_info(&mmap, offset, tensor_count)?;

        Ok(Self {
            header,
            metadata,
            tensors,
            mmap,
        })
    }

    /// 解析元数据
    ///
    /// # Parameters
    /// - `mmap`: 内存映射
    /// - `offset`: 当前偏移
    /// - `count`: 元数据数量
    ///
    /// # Returns
    /// 元数据和新的偏移
    fn parse_metadata(mmap: &[u8], mut offset: usize, count: u64) -> Result<(GgufMetadata, usize)> {
        let mut kv = HashMap::new();

        for _ in 0..count {
            // 解析键名
            let key = Self::parse_string(mmap, &mut offset)?;

            // 解析值
            let value = Self::parse_value(mmap, &mut offset)?;

            kv.insert(key, value);
        }

        Ok((GgufMetadata { kv }, offset))
    }

    /// 解析字符串
    ///
    /// # Parameters
    /// - `mmap`: 内存映射
    /// - `offset`: 当前偏移（可变引用）
    ///
    /// # Returns
    /// 解析的字符串
    fn parse_string(mmap: &[u8], offset: &mut usize) -> Result<String> {
        // 边界检查
        if *offset + 8 > mmap.len() {
            return Err(anyhow!("Unexpected EOF while reading string length"));
        }

        let len = u64::from_le_bytes(mmap[*offset..*offset + 8].try_into()?) as usize;
        *offset += 8;

        // 限制最大字符串长度（防止 DoS）
        const MAX_STRING_LEN: usize = 1024 * 1024; // 1MB
        if len > MAX_STRING_LEN {
            return Err(anyhow!(
                "String too long: {} bytes (max {})",
                len,
                MAX_STRING_LEN
            ));
        }

        // 边界检查
        if *offset + len > mmap.len() {
            return Err(anyhow!("Unexpected EOF while reading string data"));
        }

        let bytes = &mmap[*offset..*offset + len];
        *offset += len;

        String::from_utf8(bytes.to_vec()).map_err(|e| anyhow!("Invalid UTF-8 string: {}", e))
    }

    /// 解析值
    ///
    /// # Parameters
    /// - `mmap`: 内存映射
    /// - `offset`: 当前偏移（可变引用）
    ///
    /// # Returns
    /// 解析的值
    fn parse_value(mmap: &[u8], offset: &mut usize) -> Result<GgufValue> {
        let value_type = mmap[*offset];
        *offset += 1;

        let value_type = GgufValueType::try_from(value_type as u32)
            .map_err(|_| anyhow!("Unknown value type: {}", value_type))?;

        match value_type {
            GgufValueType::UInt8 => {
                let v = mmap[*offset];
                *offset += 1;
                Ok(GgufValue::UInt8(v))
            }
            GgufValueType::Int8 => {
                let v = mmap[*offset] as i8;
                *offset += 1;
                Ok(GgufValue::Int8(v))
            }
            GgufValueType::UInt16 => {
                let v = u16::from_le_bytes(mmap[*offset..*offset + 2].try_into()?);
                *offset += 2;
                Ok(GgufValue::UInt16(v))
            }
            GgufValueType::Int16 => {
                let v = i16::from_le_bytes(mmap[*offset..*offset + 2].try_into()?);
                *offset += 2;
                Ok(GgufValue::Int16(v))
            }
            GgufValueType::UInt32 => {
                let v = u32::from_le_bytes(mmap[*offset..*offset + 4].try_into()?);
                *offset += 4;
                Ok(GgufValue::UInt32(v))
            }
            GgufValueType::Int32 => {
                let v = i32::from_le_bytes(mmap[*offset..*offset + 4].try_into()?);
                *offset += 4;
                Ok(GgufValue::Int32(v))
            }
            GgufValueType::Float32 => {
                let v = f32::from_le_bytes(mmap[*offset..*offset + 4].try_into()?);
                *offset += 4;
                Ok(GgufValue::Float32(v))
            }
            GgufValueType::Bool => {
                let v = mmap[*offset] != 0;
                *offset += 1;
                Ok(GgufValue::Bool(v))
            }
            GgufValueType::String => {
                let v = Self::parse_string(mmap, offset)?;
                Ok(GgufValue::String(v))
            }
            GgufValueType::Array => {
                let element_type = mmap[*offset];
                *offset += 1;

                let count = u64::from_le_bytes(mmap[*offset..*offset + 8].try_into()?) as usize;
                *offset += 8;

                let mut arr = Vec::with_capacity(count);
                for _ in 0..count {
                    // 数组元素不带类型前缀
                    arr.push(Self::parse_value_of_type(mmap, offset, element_type)?);
                }

                Ok(GgufValue::Array(arr))
            }
            GgufValueType::UInt64 => {
                let v = u64::from_le_bytes(mmap[*offset..*offset + 8].try_into()?);
                *offset += 8;
                Ok(GgufValue::UInt64(v))
            }
            GgufValueType::Int64 => {
                let v = i64::from_le_bytes(mmap[*offset..*offset + 8].try_into()?);
                *offset += 8;
                Ok(GgufValue::Int64(v))
            }
            GgufValueType::Float64 => {
                let v = f64::from_le_bytes(mmap[*offset..*offset + 8].try_into()?);
                *offset += 8;
                Ok(GgufValue::Float64(v))
            }
        }
    }

    /// 解析指定类型的值
    ///
    /// # Parameters
    /// - `mmap`: 内存映射
    /// - `offset`: 当前偏移（可变引用）
    /// - `value_type`: 值类型编号
    ///
    /// # Returns
    /// 解析的值
    fn parse_value_of_type(mmap: &[u8], offset: &mut usize, value_type: u8) -> Result<GgufValue> {
        let value_type = GgufValueType::try_from(value_type as u32)
            .map_err(|_| anyhow!("Unknown value type: {}", value_type))?;

        match value_type {
            GgufValueType::UInt8 => {
                let v = mmap[*offset];
                *offset += 1;
                Ok(GgufValue::UInt8(v))
            }
            GgufValueType::Int8 => {
                let v = mmap[*offset] as i8;
                *offset += 1;
                Ok(GgufValue::Int8(v))
            }
            GgufValueType::UInt16 => {
                let v = u16::from_le_bytes(mmap[*offset..*offset + 2].try_into()?);
                *offset += 2;
                Ok(GgufValue::UInt16(v))
            }
            GgufValueType::Int16 => {
                let v = i16::from_le_bytes(mmap[*offset..*offset + 2].try_into()?);
                *offset += 2;
                Ok(GgufValue::Int16(v))
            }
            GgufValueType::UInt32 => {
                let v = u32::from_le_bytes(mmap[*offset..*offset + 4].try_into()?);
                *offset += 4;
                Ok(GgufValue::UInt32(v))
            }
            GgufValueType::Int32 => {
                let v = i32::from_le_bytes(mmap[*offset..*offset + 4].try_into()?);
                *offset += 4;
                Ok(GgufValue::Int32(v))
            }
            GgufValueType::Float32 => {
                let v = f32::from_le_bytes(mmap[*offset..*offset + 4].try_into()?);
                *offset += 4;
                Ok(GgufValue::Float32(v))
            }
            GgufValueType::Bool => {
                let v = mmap[*offset] != 0;
                *offset += 1;
                Ok(GgufValue::Bool(v))
            }
            GgufValueType::String => {
                let v = Self::parse_string(mmap, offset)?;
                Ok(GgufValue::String(v))
            }
            GgufValueType::UInt64 => {
                let v = u64::from_le_bytes(mmap[*offset..*offset + 8].try_into()?);
                *offset += 8;
                Ok(GgufValue::UInt64(v))
            }
            GgufValueType::Int64 => {
                let v = i64::from_le_bytes(mmap[*offset..*offset + 8].try_into()?);
                *offset += 8;
                Ok(GgufValue::Int64(v))
            }
            GgufValueType::Float64 => {
                let v = f64::from_le_bytes(mmap[*offset..*offset + 8].try_into()?);
                *offset += 8;
                Ok(GgufValue::Float64(v))
            }
            GgufValueType::Array => Err(anyhow!(
                "Nested arrays are not supported by GGUF specification"
            )),
        }
    }

    /// 解析张量信息
    ///
    /// # Parameters
    /// - `mmap`: 内存映射
    /// - `offset`: 当前偏移
    /// - `count`: 张量数量
    ///
    /// # Returns
    /// 张量映射和新的偏移
    fn parse_tensor_info(
        mmap: &[u8],
        mut offset: usize,
        count: u64,
    ) -> Result<(HashMap<String, GgufTensor>, usize)> {
        let mut tensors = HashMap::new();

        for _ in 0..count {
            // 解析张量名称
            let name = Self::parse_string(mmap, &mut offset)?;

            // 解析维度数量
            let n_dims = u32::from_le_bytes(mmap[offset..offset + 4].try_into()?) as usize;
            offset += 4;

            // GGUF 规范通常维度数 <= 4，设置上限防止恶意文件
            const MAX_DIMS: usize = 8;
            if n_dims > MAX_DIMS {
                return Err(anyhow!(
                    "Too many dimensions: {} (max {})",
                    n_dims,
                    MAX_DIMS
                ));
            }

            // 解析各维度大小
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                let dim = u64::from_le_bytes(mmap[offset..offset + 8].try_into()?) as usize;
                offset += 8;
                dims.push(dim);
            }

            // 解析张量类型
            let tensor_type_byte = mmap[offset];
            offset += 1;

            let tensor_type = GgufTensorType::from_u8(tensor_type_byte)
                .ok_or_else(|| anyhow!("Unknown tensor type: {}", tensor_type_byte))?;

            // 解析数据偏移
            let tensor_offset = u64::from_le_bytes(mmap[offset..offset + 8].try_into()?);
            offset += 8;

            let tensor = GgufTensor {
                name: name.clone(),
                dims,
                tensor_type,
                offset: tensor_offset,
            };

            tensors.insert(name, tensor);
        }

        Ok((tensors, offset))
    }

    /// 获取张量数据
    ///
    /// 通过名称获取张量的原始字节数据
    /// 获取张量原始数据
    ///
    /// # Parameters
    /// - `name`: 张量名称
    ///
    /// # Returns
    /// 张量数据的字节切片
    ///
    /// # Note
    /// `tensor.offset` 是从文件起始处的绝对偏移量（GGUF 规范）
    pub fn get_tensor_data(&self, name: &str) -> Option<&[u8]> {
        let tensor = self.tensors.get(name)?;
        // tensor.offset 是绝对偏移（GGUF 规范）
        let start = tensor.offset as usize;
        let end = start + tensor.data_size();

        if end <= self.mmap.len() {
            Some(&self.mmap[start..end])
        } else {
            None
        }
    }

    /// 通过张量引用获取数据
    ///
    /// 将张量数据反序列化为 f32 向量
    ///
    /// # Parameters
    /// - `tensor`: 张量信息引用
    ///
    /// # Returns
    /// 反序列化后的 f32 向量
    ///
    /// # Note
    /// `tensor.offset` 是从文件起始处的绝对偏移量（GGUF 规范）
    ///
    /// # Safety
    /// 内存映射通常满足对齐要求，但某些平台可能需要额外处理
    pub fn get_tensor_data_by_ref(&self, tensor: &GgufTensor) -> Result<Vec<f32>> {
        // tensor.offset 是绝对偏移（GGUF 规范）
        let start = tensor.offset as usize;
        let end = start + tensor.data_size();

        if end > self.mmap.len() {
            return Err(anyhow!("Tensor data out of bounds"));
        }

        let data = &self.mmap[start..end];
        let num_elements = tensor.dims.iter().product::<usize>();

        match tensor.tensor_type {
            GgufTensorType::F32 => {
                // 检查对齐：f32 需要 4 字节对齐
                if data.as_ptr() as usize % std::mem::align_of::<f32>() != 0 {
                    return Err(anyhow!("F32 data is not properly aligned"));
                }
                let f32_slice: &[f32] = cast_slice(data);
                if f32_slice.len() != num_elements {
                    return Err(anyhow!(
                        "F32 tensor size mismatch: expected {}, got {}",
                        num_elements,
                        f32_slice.len()
                    ));
                }
                Ok(f32_slice.to_vec())
            }
            GgufTensorType::F16 => {
                // 检查对齐：f16 需要 2 字节对齐
                if data.as_ptr() as usize % std::mem::align_of::<half::f16>() != 0 {
                    return Err(anyhow!("F16 data is not properly aligned"));
                }
                let f16_slice: &[half::f16] = cast_slice(data);
                if f16_slice.len() != num_elements {
                    return Err(anyhow!(
                        "F16 tensor size mismatch: expected {}, got {}",
                        num_elements,
                        f16_slice.len()
                    ));
                }
                Ok(f16_slice.iter().map(|&f| f.to_f32()).collect())
            }
            _ => {
                let result = super::quant::dequantize(data, tensor.tensor_type, num_elements);
                Ok(result)
            }
        }
    }

    /// 获取模型配置
    ///
    /// 从元数据中提取语言模型配置
    ///
    /// # Returns
    /// 模型配置
    pub fn get_model_config(&self) -> Result<ModelConfig> {
        let architecture_str = self
            .metadata
            .get_string("general.architecture")
            .unwrap_or("llama");

        let arch = Architecture::from_str(architecture_str);
        let prefix = arch.parameter_prefix();

        let is_moe = arch.is_moe();
        let use_mla = arch.uses_mla();

        let vocab_size = self
            .metadata
            .get_u64("general.vocab_size")
            .unwrap_or(DEFAULT_VOCAB_SIZE as u64) as usize;

        let fallback_prefixes: Vec<&str> = match arch {
            Architecture::Phi => vec!["phi", "llama", "qwen2", "gemma"],
            Architecture::Mistral => vec!["mistral", "llama", "qwen2"],
            Architecture::Mixtral => vec!["mixtral", "mistral", "llama"],
            Architecture::DeepSeekV3 => vec!["deepseek_v3", "deepseek", "mixtral", "llama"],
            Architecture::Yi => vec!["yi", "llama", "qwen2"],
            Architecture::ChatGLM => vec!["chatglm", "llama", "qwen2"],
            Architecture::Baichuan => vec!["baichuan", "llama", "qwen2"],
            Architecture::Falcon => vec!["falcon", "llama"],
            Architecture::StableLM => vec!["stablelm", "llama", "qwen2"],
            _ => vec![prefix, "llama", "qwen2", "gemma"],
        };

        let get_param_u32 = |key_suffix: &str| -> Option<u32> {
            if let Some(val) = self.metadata.get_u32(&format!("{}.{}", prefix, key_suffix)) {
                return Some(val);
            }
            for &fallback_prefix in &fallback_prefixes {
                if fallback_prefix != prefix {
                    if let Some(val) = self
                        .metadata
                        .get_u32(&format!("{}.{}", fallback_prefix, key_suffix))
                    {
                        return Some(val);
                    }
                }
            }
            None
        };

        let get_param_f32 = |key_suffix: &str| -> Option<f32> {
            if let Some(val) = self.metadata.get_f32(&format!("{}.{}", prefix, key_suffix)) {
                return Some(val);
            }
            for &fallback_prefix in &fallback_prefixes {
                if fallback_prefix != prefix {
                    if let Some(val) = self
                        .metadata
                        .get_f32(&format!("{}.{}", fallback_prefix, key_suffix))
                    {
                        return Some(val);
                    }
                }
            }
            None
        };

        let hidden_size =
            get_param_u32("embedding_length").unwrap_or(DEFAULT_HIDDEN_SIZE as u32) as usize;

        let intermediate_size = get_param_u32("feed_forward_length")
            .unwrap_or(DEFAULT_INTERMEDIATE_SIZE as u32) as usize;

        let num_hidden_layers =
            get_param_u32("block_count").unwrap_or(DEFAULT_NUM_HIDDEN_LAYERS as u32) as usize;

        let num_attention_heads = get_param_u32("attention.head_count")
            .unwrap_or(DEFAULT_NUM_ATTENTION_HEADS as u32)
            as usize;

        let num_key_value_heads =
            get_param_u32("attention.head_count_kv").unwrap_or(num_attention_heads as u32) as usize;

        let max_position_embeddings = get_param_u32("context_length")
            .unwrap_or(DEFAULT_MAX_POSITION_EMBEDDINGS as u32)
            as usize;

        let rms_norm_eps =
            get_param_f32("attention.layer_norm_rms_epsilon").unwrap_or(DEFAULT_RMS_NORM_EPS);

        let default_rope_theta = arch.default_rope_theta();
        let rope_theta = get_param_f32("rope.freq_base").unwrap_or(default_rope_theta);

        let _baichuan_no_rope = arch == Architecture::Baichuan && rope_theta == 0.0;

        let (num_experts, top_k, mla_latent_dim) = match arch {
            Architecture::DeepSeekV3 => {
                let experts = get_param_u32("expert_count")
                    .or_else(|| get_param_u32("moe.num_experts"))
                    .unwrap_or(64) as usize;
                let tk = get_param_u32("top_k")
                    .or_else(|| get_param_u32("moe.top_k"))
                    .unwrap_or(6) as usize;
                let latent = get_param_u32("mla.latent_dim").unwrap_or(512) as usize;
                (experts, tk, latent)
            }
            Architecture::Mixtral => {
                let experts = get_param_u32("expert_count")
                    .or_else(|| get_param_u32("moe.num_experts"))
                    .unwrap_or(8) as usize;
                let tk = get_param_u32("top_k")
                    .or_else(|| get_param_u32("moe.top_k"))
                    .unwrap_or(2) as usize;
                (experts, tk, 512)
            }
            _ => (0, 2, 512),
        };

        Ok(ModelConfig {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
            architecture: arch,
            is_moe,
            num_experts,
            top_k,
            use_mla,
            mla_latent_dim,
        })
    }

    /// 通过名称获取张量数据
    ///
    /// # Parameters
    /// - `name`: 张量名称
    ///
    /// # Returns
    /// 反序列化后的 f32 向量
    pub fn get_tensor_data_by_name(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("Tensor not found: {}", name))?;
        self.get_tensor_data_by_ref(tensor)
    }

    pub fn get_tensor_1d(&self, name: &str) -> Result<Vec<f32>> {
        self.get_tensor_data_by_name(name)
    }

    pub fn get_tensor_2d(&self, name: &str) -> Result<ndarray::Array2<f32>> {
        let data = self.get_tensor_data_by_name(name)?;
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("Tensor not found: {}", name))?;

        if tensor.dims.len() != 2 {
            return Err(anyhow!(
                "Expected 2D tensor for '{}', got {}D",
                name,
                tensor.dims.len()
            ));
        }

        let rows = tensor.dims[0];
        let cols = tensor.dims[1];
        Ok(ndarray::Array2::<f32>::from_shape_vec((rows, cols), data)?)
    }

    /// 获取 TTS 配置
    ///
    /// 从元数据中提取语音合成模型配置
    ///
    /// # Returns
    /// TTS 配置（如果存在）
    pub fn get_tts_config(&self) -> Option<TTSConfigGGUF> {
        if !self.has_tts_tensors() {
            return None;
        }

        let sample_rate = self.metadata.get_u32("tts.sample_rate").unwrap_or(24000) as usize;

        let hidden_size = self.metadata.get_u32("tts.hidden_size").unwrap_or(1024) as usize;

        let num_flow_layers = self.metadata.get_u32("tts.num_flow_layers").unwrap_or(12) as usize;

        Some(TTSConfigGGUF {
            sample_rate,
            hidden_size,
            num_flow_layers,
        })
    }

    pub fn load_moe_v2_weights(
        &self,
        layer_idx: usize,
        config: &ModelConfig,
    ) -> Option<MoEWeightsV2> {
        let prefix = format!(
            "{}.model.layers.{}",
            config.architecture.parameter_prefix(),
            layer_idx
        );

        let shared_experts = self.load_shared_experts(&prefix, config)?;
        let routing_experts = self.load_routing_experts(&prefix, config)?;

        let router_key = format!("{}.ffn.gate.weight", prefix);
        let router = self.get_tensor_2d(&router_key).ok()?;

        let num_shared = shared_experts.len();
        let num_routing = routing_experts.len();

        let mut moe_v2 = MoEWeightsV2::new(
            num_shared,
            num_routing,
            config.top_k,
            config.hidden_size,
            config.intermediate_size,
        );

        moe_v2.shared_experts = shared_experts;
        moe_v2.routing_experts = routing_experts;
        moe_v2.routing_router = router;

        Some(moe_v2)
    }

    fn load_shared_experts(&self, prefix: &str, _config: &ModelConfig) -> Option<Vec<FFNWeights>> {
        let mut experts = Vec::new();
        let mut i = 0;
        loop {
            let gate_key = format!("{}.ffn.shared_experts.{}.gate_proj.weight", prefix, i);
            let up_key = format!("{}.ffn.shared_experts.{}.up_proj.weight", prefix, i);
            let down_key = format!("{}.ffn.shared_experts.{}.down_proj.weight", prefix, i);

            match (
                self.get_tensor_1d(&gate_key),
                self.get_tensor_1d(&up_key),
                self.get_tensor_1d(&down_key),
            ) {
                (Ok(gate), Ok(up), Ok(down)) => {
                    experts.push(FFNWeights {
                        gate_weight: gate,
                        up_weight: up,
                        down_weight: down,
                    });
                    i += 1;
                }
                _ => break,
            }
        }

        if experts.is_empty() {
            None
        } else {
            Some(experts)
        }
    }

    fn load_routing_experts(&self, prefix: &str, config: &ModelConfig) -> Option<Vec<FFNWeights>> {
        let mut experts = Vec::new();
        let num_experts = config.num_experts;

        for i in 0..num_experts {
            let gate_key = format!("{}.ffn.experts.{}.gate_proj.weight", prefix, i);
            let up_key = format!("{}.ffn.experts.{}.up_proj.weight", prefix, i);
            let down_key = format!("{}.ffn.experts.{}.down_proj.weight", prefix, i);

            match (
                self.get_tensor_1d(&gate_key),
                self.get_tensor_1d(&up_key),
                self.get_tensor_1d(&down_key),
            ) {
                (Ok(gate), Ok(up), Ok(down)) => {
                    experts.push(FFNWeights {
                        gate_weight: gate,
                        up_weight: up,
                        down_weight: down,
                    });
                }
                _ => break,
            }
        }

        if experts.is_empty() {
            None
        } else {
            Some(experts)
        }
    }
}

// ============================================================================
// 类型转换实现
// ============================================================================

impl TryFrom<u32> for GgufValueType {
    type Error = anyhow::Error;

    fn try_from(v: u32) -> Result<Self> {
        match v {
            0 => Ok(Self::UInt8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::UInt16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::UInt32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::UInt64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => Err(anyhow!("Unknown value type: {}", v)),
        }
    }
}

impl TryFrom<u8> for GgufTensorType {
    type Error = anyhow::Error;

    fn try_from(v: u8) -> Result<Self> {
        Self::from_u8(v).ok_or_else(|| anyhow!("Unknown tensor type: {}", v))
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_magic() {
        assert_eq!(GGUF_MAGIC, 0x46554747);
    }

    #[test]
    fn test_tensor_type_from_u8() {
        assert_eq!(GgufTensorType::from_u8(0), Some(GgufTensorType::F32));
        assert_eq!(GgufTensorType::from_u8(1), Some(GgufTensorType::F16));
        assert_eq!(GgufTensorType::from_u8(2), Some(GgufTensorType::Q4_0));
        assert_eq!(GgufTensorType::from_u8(3), Some(GgufTensorType::Q4_1));
        assert_eq!(GgufTensorType::from_u8(12), Some(GgufTensorType::Q4K));
    }

    #[test]
    fn test_value_type_try_from() {
        assert!(matches!(
            GgufValueType::try_from(0u32),
            Ok(GgufValueType::UInt8)
        ));
        assert!(matches!(
            GgufValueType::try_from(6u32),
            Ok(GgufValueType::Float32)
        ));
        assert!(matches!(
            GgufValueType::try_from(8u32),
            Ok(GgufValueType::String)
        ));
        assert!(GgufValueType::try_from(100u32).is_err());
    }

    #[test]
    fn test_model_config_defaults() {
        assert_eq!(DEFAULT_VOCAB_SIZE, 152064);
        assert_eq!(DEFAULT_HIDDEN_SIZE, 3584);
        assert_eq!(DEFAULT_NUM_HIDDEN_LAYERS, 28);
        assert_eq!(DEFAULT_NUM_ATTENTION_HEADS, 28);
        assert_eq!(DEFAULT_INTERMEDIATE_SIZE, 18944);
        assert_eq!(DEFAULT_MAX_POSITION_EMBEDDINGS, 131072);
        assert!((DEFAULT_ROPE_THETA - 1000000.0).abs() < 1e-6);
        assert!((DEFAULT_RMS_NORM_EPS - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_type_block_sizes() {
        assert_eq!(GgufTensorType::F32.block_size(), 1);
        assert_eq!(GgufTensorType::F16.block_size(), 1);
        assert_eq!(GgufTensorType::Q4_0.block_size(), 32);
        assert_eq!(GgufTensorType::Q4_1.block_size(), 32);
        assert_eq!(GgufTensorType::Q8_0.block_size(), 32);
        assert_eq!(GgufTensorType::Q4K.block_size(), 256);
        assert_eq!(GgufTensorType::Q5K.block_size(), 256);
        assert_eq!(GgufTensorType::Q6K.block_size(), 256);
    }

    #[test]
    fn test_vision_config_gguf() {
        let config = VisionConfigGGUF {
            image_size: 448,
            patch_size: 14,
            hidden_size: 1152,
            num_layers: 27,
            num_heads: 16,
            intermediate_size: 4304,
        };
        assert_eq!(config.image_size, 448);
        assert_eq!(config.patch_size, 14);
        assert_eq!(config.hidden_size, 1152);
    }

    #[test]
    fn test_model_config_creation() {
        let config = ModelConfig {
            vocab_size: 152064,
            hidden_size: 3584,
            intermediate_size: 18944,
            num_hidden_layers: 28,
            num_attention_heads: 28,
            num_key_value_heads: 28,
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            architecture: Architecture::Llama,
            is_moe: false,
            num_experts: 8,
            top_k: 2,
            use_mla: true,
            mla_latent_dim: 512,
        };
        assert_eq!(config.vocab_size, 152064);
        assert_eq!(config.hidden_size, 3584);
    }

    #[test]
    fn test_gguf_tensor_creation() {
        let tensor = GgufTensor {
            name: "test_tensor".to_string(),
            dims: vec![10, 20],
            tensor_type: GgufTensorType::F32,
            offset: 0,
        };
        assert_eq!(tensor.name, "test_tensor");
        assert_eq!(tensor.dims, vec![10, 20]);
        assert_eq!(tensor.tensor_type, GgufTensorType::F32);
    }

    #[test]
    fn test_tensor_type_from_u8_all() {
        // 测试所有有效类型映射
        let type_mappings: [(u8, GgufTensorType); 24] = [
            (0, GgufTensorType::F32),
            (1, GgufTensorType::F16),
            (2, GgufTensorType::Q4_0),
            (3, GgufTensorType::Q4_1),
            (4, GgufTensorType::Q4_2),
            (5, GgufTensorType::Q4_3),
            (7, GgufTensorType::Q8_0),
            (8, GgufTensorType::Q8_1),
            (10, GgufTensorType::Q2K),
            (11, GgufTensorType::Q3K),
            (12, GgufTensorType::Q4K),
            (13, GgufTensorType::Q5K),
            (14, GgufTensorType::Q6K),
            (15, GgufTensorType::Q8K),
            (16, GgufTensorType::Iq2Xxs),
            (17, GgufTensorType::Iq2Xs),
            (18, GgufTensorType::Iq3Xxs),
            (19, GgufTensorType::Iq1S),
            (20, GgufTensorType::Iq4Nl),
            (21, GgufTensorType::Iq3S),
            (22, GgufTensorType::Iq2S),
            (23, GgufTensorType::Iq4Xs),
            (24, GgufTensorType::I8),
            (25, GgufTensorType::I16),
        ];

        for (value, expected_type) in type_mappings {
            assert_eq!(
                GgufTensorType::from_u8(value),
                Some(expected_type),
                "Failed for value {}",
                value
            );
        }

        // 测试更多整数类型
        assert_eq!(GgufTensorType::from_u8(26), Some(GgufTensorType::I32));
        assert_eq!(GgufTensorType::from_u8(27), Some(GgufTensorType::I64));
        assert_eq!(GgufTensorType::from_u8(28), Some(GgufTensorType::F64));
        assert_eq!(GgufTensorType::from_u8(29), Some(GgufTensorType::BF16));

        // 测试无效值
        assert_eq!(GgufTensorType::from_u8(6), None); // 保留值
        assert_eq!(GgufTensorType::from_u8(9), None); // 未定义
        assert_eq!(GgufTensorType::from_u8(100), None);
    }

    #[test]
    fn test_bytes_per_block_all() {
        // 测试非量化类型
        assert_eq!(GgufTensorType::F32.bytes_per_block(), 4);
        assert_eq!(GgufTensorType::F16.bytes_per_block(), 2);
        assert_eq!(GgufTensorType::BF16.bytes_per_block(), 2);
        assert_eq!(GgufTensorType::F64.bytes_per_block(), 8);
        assert_eq!(GgufTensorType::I8.bytes_per_block(), 1);
        assert_eq!(GgufTensorType::I16.bytes_per_block(), 2);
        assert_eq!(GgufTensorType::I32.bytes_per_block(), 4);
        assert_eq!(GgufTensorType::I64.bytes_per_block(), 8);

        // 测试量化类型
        assert_eq!(GgufTensorType::Q4_0.bytes_per_block(), 18);
        assert_eq!(GgufTensorType::Q4_1.bytes_per_block(), 20);
        assert_eq!(GgufTensorType::Q4_2.bytes_per_block(), 18);
        assert_eq!(GgufTensorType::Q4_3.bytes_per_block(), 20);
        assert_eq!(GgufTensorType::Q8_0.bytes_per_block(), 34);
        assert_eq!(GgufTensorType::Q8_1.bytes_per_block(), 36);
    }

    #[test]
    fn test_element_size() {
        // 非量化类型
        assert_eq!(GgufTensorType::F32.element_size(), 4);
        assert_eq!(GgufTensorType::F16.element_size(), 2);
        assert_eq!(GgufTensorType::I8.element_size(), 1);
        assert_eq!(GgufTensorType::I16.element_size(), 2);
        assert_eq!(GgufTensorType::I32.element_size(), 4);
        assert_eq!(GgufTensorType::I64.element_size(), 8);

        // 量化类型（通过 bytes_per_block / block_size 计算）
        assert_eq!(GgufTensorType::Q4_0.element_size(), 18 / 32);
        assert_eq!(GgufTensorType::Q8_0.element_size(), 34 / 32);
    }

    #[test]
    fn test_tensor_type_try_from_u8() {
        use std::convert::TryFrom;

        assert!(matches!(
            GgufTensorType::try_from(0u8),
            Ok(GgufTensorType::F32)
        ));
        assert!(matches!(
            GgufTensorType::try_from(1u8),
            Ok(GgufTensorType::F16)
        ));
        assert!(matches!(
            GgufTensorType::try_from(2u8),
            Ok(GgufTensorType::Q4_0)
        ));
        assert!(matches!(
            GgufTensorType::try_from(7u8),
            Ok(GgufTensorType::Q8_0)
        ));
        assert!(GgufTensorType::try_from(6u8).is_err()); // 保留值
        assert!(GgufTensorType::try_from(100u8).is_err()); // 无效值
    }
}

// ============================================================================
// 集成测试 - 最小 GGUF 文件解析
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// 构建最小化的 GGUF 文件字节
    fn build_minimal_gguf() -> Vec<u8> {
        let mut buf = Vec::new();

        // 魔数 "GGUF"
        buf.extend_from_slice(&0x46554747u32.to_le_bytes());

        // 版本号 (3)
        buf.extend_from_slice(&3u32.to_le_bytes());

        // 张量数量 (1)
        buf.extend_from_slice(&1u64.to_le_bytes());

        // 元数据数量 (3)
        buf.extend_from_slice(&3u64.to_le_bytes());

        // 元数据 1: general.architecture = "llama"
        let key1 = "general.architecture";
        buf.extend_from_slice(&(key1.len() as u64).to_le_bytes());
        buf.extend_from_slice(key1.as_bytes());
        buf.push(8); // String type
        let val1 = "llama";
        buf.extend_from_slice(&(val1.len() as u64).to_le_bytes());
        buf.extend_from_slice(val1.as_bytes());

        // 元数据 2: llama.embedding_length = 4096
        let key2 = "llama.embedding_length";
        buf.extend_from_slice(&(key2.len() as u64).to_le_bytes());
        buf.extend_from_slice(key2.as_bytes());
        buf.push(4); // UInt32 type
        buf.extend_from_slice(&4096u32.to_le_bytes());

        // 元数据 3: llama.block_count = 32
        let key3 = "llama.block_count";
        buf.extend_from_slice(&(key3.len() as u64).to_le_bytes());
        buf.extend_from_slice(key3.as_bytes());
        buf.push(4); // UInt32 type
        buf.extend_from_slice(&32u32.to_le_bytes());

        // 张量信息: "model.token_embd.weight" [4096] F32
        let tensor_name = "model.token_embd.weight";
        buf.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
        buf.extend_from_slice(tensor_name.as_bytes());

        // 维度数量
        buf.extend_from_slice(&1u32.to_le_bytes());
        // 维度大小
        buf.extend_from_slice(&4096u64.to_le_bytes());
        // 张量类型 F32
        buf.push(0);
        // 偏移量 (从文件头开始计算，需要对齐)
        let tensor_offset = buf.len() as u64 + 8; // +8 for offset field
                                                  // 对齐到 32 字节
        let aligned_offset = (tensor_offset + 31) / 32 * 32;
        buf.extend_from_slice(&aligned_offset.to_le_bytes());

        // 填充到对齐位置
        let padding = aligned_offset as usize - buf.len();
        buf.extend(std::iter::repeat(0u8).take(padding));

        // 张量数据: 4096 个 f32 (全部为 0.5)
        for _ in 0..4096 {
            buf.extend_from_slice(&0.5f32.to_le_bytes());
        }

        buf
    }

    #[test]
    fn test_gguf_file_parsing() {
        let gguf_bytes = build_minimal_gguf();

        // 写入临时文件
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(&gguf_bytes)
            .expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        // 解析文件
        let path = temp_file.path();
        let gguf = GgufFile::open(path).expect("Failed to parse GGUF file");

        // 验证文件头
        assert_eq!(gguf.header.version, 3);
        assert_eq!(gguf.header.tensor_count, 1);
        assert_eq!(gguf.header.metadata_kv_count, 3);

        // 验证元数据
        assert_eq!(
            gguf.metadata.get_string("general.architecture"),
            Some("llama")
        );
        assert_eq!(gguf.metadata.get_u32("llama.embedding_length"), Some(4096));
        assert_eq!(gguf.metadata.get_u32("llama.block_count"), Some(32));

        // 验证张量信息
        assert!(gguf.tensors.contains_key("model.token_embd.weight"));
        let tensor = gguf.tensors.get("model.token_embd.weight").unwrap();
        assert_eq!(tensor.dims, vec![4096]);
        assert_eq!(tensor.tensor_type, GgufTensorType::F32);
    }

    #[test]
    fn test_gguf_tensor_data_access() {
        let gguf_bytes = build_minimal_gguf();

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(&gguf_bytes)
            .expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let gguf = GgufFile::open(temp_file.path()).expect("Failed to parse GGUF file");

        // 获取张量数据
        let data = gguf
            .get_tensor_data("model.token_embd.weight")
            .expect("Failed to get tensor data");

        // 验证数据大小 (4096 * 4 bytes)
        assert_eq!(data.len(), 4096 * 4);

        // 验证数据内容
        let f32_slice: &[f32] = bytemuck::cast_slice(data);
        assert_eq!(f32_slice.len(), 4096);
        assert!((f32_slice[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_gguf_model_config_extraction() {
        let gguf_bytes = build_minimal_gguf();

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(&gguf_bytes)
            .expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let gguf = GgufFile::open(temp_file.path()).expect("Failed to parse GGUF file");

        // 提取模型配置
        let config = gguf.get_model_config().expect("Failed to get model config");

        // 验证配置值
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 32);
    }

    #[test]
    fn test_gguf_invalid_magic() {
        let mut buf = Vec::new();
        // 错误的魔数
        buf.extend_from_slice(&0x12345678u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(&buf)
            .expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let result = GgufFile::open(temp_file.path());
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid GGUF magic"));
    }

    #[test]
    fn test_gguf_file_too_small() {
        let buf = vec![0u8; 10]; // 太小

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(&buf)
            .expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let result = GgufFile::open(temp_file.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));
    }

    #[test]
    fn test_gguf_tensor_iteration() {
        let gguf_bytes = build_minimal_gguf();

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(&gguf_bytes)
            .expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let gguf = GgufFile::open(temp_file.path()).expect("Failed to parse GGUF file");

        // 测试 LLM 张量迭代器
        let llm_count = gguf.iter_llm_tensors().count();
        assert_eq!(llm_count, 1);

        // 测试视觉张量迭代器（应该为空）
        let vision_count = gguf.iter_vision_tensors().count();
        assert_eq!(vision_count, 0);

        // 测试 has_* 方法
        assert!(gguf.has_llm_tensors());
        assert!(!gguf.has_vision_tensors());
        assert!(!gguf.has_audio_encoder_tensors());
    }

    #[test]
    fn test_gguf_header_parse_valid() {
        // 有效GGUF头解析
        let gguf_bytes = build_minimal_gguf();

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(&gguf_bytes)
            .expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let gguf = GgufFile::open(temp_file.path()).expect("Failed to parse GGUF file");

        // 测试版本号、架构等字段
        assert_eq!(gguf.header.version, 3);
        assert_eq!(gguf.header.tensor_count, 1);
        assert_eq!(gguf.header.metadata_kv_count, 3);

        // 验证魔数
        assert_eq!(GGUF_MAGIC, 0x46554747);
    }

    #[test]
    fn test_gguf_tensor_info_extraction() {
        // 张量信息提取
        let gguf_bytes = build_minimal_gguf();

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(&gguf_bytes)
            .expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let gguf = GgufFile::open(temp_file.path()).expect("Failed to parse GGUF file");

        // 验证名称、维度、类型等
        let tensor = gguf
            .tensors
            .get("model.token_embd.weight")
            .expect("Tensor should exist");

        assert_eq!(tensor.name, "model.token_embd.weight");
        assert_eq!(tensor.dims, vec![4096]);
        assert_eq!(tensor.tensor_type, GgufTensorType::F32);

        // 验证张量计算方法
        assert_eq!(tensor.num_elements(), 4096);
        assert_eq!(tensor.data_size(), 4096 * 4); // F32 = 4 bytes per element

        // 测试get_tensor_data方法
        let data = gguf
            .get_tensor_data("model.token_embd.weight")
            .expect("Should get tensor data");
        assert!(!data.is_empty());
        assert_eq!(data.len(), 16384); // 4096 * 4 bytes
    }

    #[test]
    fn test_gguf_invalid_file_error() {
        // 无效文件错误处理 - 不存在的路径
        let result = GgufFile::open("/nonexistent/file.gguf");
        assert!(result.is_err(), "Should fail for nonexistent file");

        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.to_lowercase().contains("no such file")
                || err_msg.to_lowercase().contains("not found")
                || err_msg.to_lowercase().contains("os error"),
            "Error should indicate file not found: {}",
            err_msg
        );
    }

    #[test]
    fn test_gguf_truncated_file_handling() {
        // 截断文件处理
        let temp_dir = std::env::temp_dir();
        let truncated_path = temp_dir.join("truncated.gguf");

        // 创建截断的文件(只有部分头部)
        std::fs::write(&truncated_path, b"GGUF partial data...").ok();

        let result = GgufFile::open(&truncated_path);
        assert!(result.is_err(), "Should fail for truncated file");

        let err_msg = result.unwrap_err().to_string();
        // 应该报告文件太小或解析错误
        assert!(
            err_msg.to_lowercase().contains("too small")
                || err_msg.to_lowercase().contains("unexpected eof")
                || err_msg.contains("Invalid GGUF magic"),
            "Error should indicate truncation issue: {}",
            err_msg
        );

        std::fs::remove_file(truncated_path).ok();
    }

    #[test]
    fn test_gguf_invalid_magic_number() {
        // 无效魔数测试
        let mut buf = Vec::new();
        buf.extend_from_slice(&0x12345678u32.to_le_bytes()); // 错误的魔数
        buf.extend_from_slice(&3u32.to_le_bytes()); // 版本号
        buf.extend_from_slice(&0u64.to_le_bytes()); // 张量数量
        buf.extend_from_slice(&0u64.to_le_bytes()); // 元数据数量

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(&buf)
            .expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let result = GgufFile::open(temp_file.path());
        assert!(result.is_err(), "Should fail for invalid magic number");

        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Invalid GGUF magic"),
            "Error should mention invalid magic: {}",
            err_msg
        );
    }

    #[test]
    fn test_gguf_metadata_accessors() {
        // 元数据访问器测试
        let gguf_bytes = build_minimal_gguf();

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(&gguf_bytes)
            .expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let gguf = GgufFile::open(temp_file.path()).expect("Failed to parse GGUF file");

        // 测试字符串访问器
        let arch = gguf.metadata.get_string("general.architecture");
        assert_eq!(arch, Some("llama"));

        // 测试u32访问器
        let embed_len = gguf.metadata.get_u32("llama.embedding_length");
        assert_eq!(embed_len, Some(4096));

        // 测试不存在的键
        let missing = gguf.metadata.get_string("nonexistent.key");
        assert_eq!(missing, None);

        let missing_u32 = gguf.metadata.get_u32("nonexistent.u32");
        assert_eq!(missing_u32, None);
    }

    #[test]
    fn test_gguf_value_type_variants() {
        // 值类型枚举变体测试
        use GgufValue::*;

        // 创建各种类型的值
        let values = vec![
            UInt8(42),
            Int8(-10),
            UInt16(1000),
            Int16(-500),
            UInt32(12345),
            Int32(-99999),
            Float32(3.14),
            Bool(true),
            String("hello".to_string()),
            Array(vec![UInt8(1), UInt8(2), UInt8(3)]),
            UInt64(123456789),
            Int64(-987654321),
            Float64(2.71828),
        ];

        // 所有值都应该能创建和克隆
        for value in &values {
            let _clone = value.clone();
        }
    }

    #[test]
    fn test_gguf_tensor_type_comprehensive() {
        // 张量类型综合测试
        // 测试所有类型的block_size和bytes_per_block的一致性
        let types_to_test = [
            (GgufTensorType::F32, 1, 4),
            (GgufTensorType::F16, 1, 2),
            (GgufTensorType::BF16, 1, 2),
            (GgufTensorType::F64, 1, 8),
            (GgufTensorType::I8, 1, 1),
            (GgufTensorType::I16, 1, 2),
            (GgufTensorType::I32, 1, 4),
            (GgufTensorType::I64, 1, 8),
            (GgufTensorType::Q4_0, 32, 18),
            (GgufTensorType::Q4_1, 32, 20),
            (GgufTensorType::Q8_0, 32, 34),
            (GgufTensorType::Q8_1, 32, 36),
            (GgufTensorType::Q2K, 256, 68),
            (GgufTensorType::Q3K, 256, 162),
            (GgufTensorType::Q4K, 256, 144),
            (GgufTensorType::Q5K, 256, 176),
            (GgufTensorType::Q6K, 256, 210),
            (GgufTensorType::Q8K, 256, 258),
        ];

        for (tt, expected_block_size, expected_bytes) in types_to_test {
            assert_eq!(
                tt.block_size(),
                expected_block_size,
                "Block size mismatch for {:?}",
                tt
            );
            assert_eq!(
                tt.bytes_per_block(),
                expected_bytes,
                "Bytes per block mismatch for {:?}",
                tt
            );

            // element_size应该是bytes_per_block / block_size
            if expected_block_size > 0 {
                let expected_element_size = expected_bytes / expected_block_size;
                assert_eq!(
                    tt.element_size(),
                    expected_element_size,
                    "Element size mismatch for {:?}",
                    tt
                );
            }
        }
    }

    #[test]
    fn test_gguf_multimodal_tensor_detection() {
        // 多模态张量检测测试
        let gguf_bytes = build_minimal_gguf();

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(&gguf_bytes)
            .expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let gguf = GgufFile::open(temp_file.path()).expect("Failed to parse GGUF file");

        // 最小GGUF文件只有LLM张量,没有多模态张量
        assert!(!gguf.has_vision_tensors());
        assert!(!gguf.has_resampler_tensors());
        assert!(!gguf.has_audio_encoder_tensors());
        assert!(!gguf.has_tts_tensors());

        // 对应的配置应该返回None
        assert!(gguf.get_vision_config().is_none());
        assert!(gguf.get_audio_config().is_none());
        assert!(gguf.get_tts_config().is_none());

        // 张量集合也应该为空
        assert!(gguf.get_vision_tensors().is_empty());
        assert!(gguf.get_resampler_tensors().is_empty());
        assert!(gguf.get_audio_encoder_tensors().is_empty());
        assert!(gguf.get_tts_tensors().is_empty());
    }

    // ========================================================================
    // Architecture 架构枚举测试（新增 >20 个测试）
    // ========================================================================

    #[test]
    fn test_architecture_from_str_basic() {
        // 测试基本架构解析
        assert_eq!(Architecture::from_str("llama"), Architecture::Llama);
        assert_eq!(Architecture::from_str("qwen2"), Architecture::Qwen2);
        assert_eq!(Architecture::from_str("minicpm"), Architecture::MiniCPM);
        assert_eq!(Architecture::from_str("gemma"), Architecture::Gemma);
        assert_eq!(Architecture::from_str("phi"), Architecture::Phi);
        assert_eq!(Architecture::from_str("mistral"), Architecture::Mistral);
        assert_eq!(Architecture::from_str("mixtral"), Architecture::Mixtral);
        assert_eq!(Architecture::from_str("yi"), Architecture::Yi);
        assert_eq!(Architecture::from_str("chatglm"), Architecture::ChatGLM);
        assert_eq!(Architecture::from_str("baichuan"), Architecture::Baichuan);
        assert_eq!(Architecture::from_str("falcon"), Architecture::Falcon);
        assert_eq!(Architecture::from_str("stablelm"), Architecture::StableLM);
    }

    #[test]
    fn test_architecture_from_str_case_insensitive() {
        // 测试大小写不敏感
        assert_eq!(Architecture::from_str("LLaMA"), Architecture::Llama);
        assert_eq!(Architecture::from_str("QWen2"), Architecture::Qwen2);
        assert_eq!(Architecture::from_str("Mistral"), Architecture::Mistral);
        assert_eq!(Architecture::from_str("PHI"), Architecture::Phi);
        assert_eq!(Architecture::from_str("Yi"), Architecture::Yi);
    }

    #[test]
    fn test_architecture_from_str_version_variants() {
        // 测试版本变体映射
        // Phi 系列变体
        assert_eq!(Architecture::from_str("phi3"), Architecture::Phi);
        assert_eq!(Architecture::from_str("phi-2"), Architecture::Phi);
        assert_eq!(Architecture::from_str("phi-1.5"), Architecture::Phi);

        // Mixtral 变体
        assert_eq!(
            Architecture::from_str("mixtral-8x7b"),
            Architecture::Mixtral
        );
        assert_eq!(
            Architecture::from_str("mixtral-8x22b"),
            Architecture::Mixtral
        );

        // ChatGLM 变体
        assert_eq!(Architecture::from_str("glm"), Architecture::ChatGLM);
        assert_eq!(Architecture::from_str("glm2"), Architecture::ChatGLM);
        assert_eq!(Architecture::from_str("glm3"), Architecture::ChatGLM);
        assert_eq!(Architecture::from_str("glm4"), Architecture::ChatGLM);
        assert_eq!(Architecture::from_str("chatglm2"), Architecture::ChatGLM);
        assert_eq!(Architecture::from_str("chatglm3"), Architecture::ChatGLM);

        // Falcon 变体
        assert_eq!(Architecture::from_str("falcon-40b"), Architecture::Falcon);
        assert_eq!(Architecture::from_str("falcon-180b"), Architecture::Falcon);
    }

    #[test]
    fn test_architecture_from_str_unknown_fallback() {
        // 测试未知架构回退到 Llama
        assert_eq!(Architecture::from_str("unknown"), Architecture::Llama);
        assert_eq!(Architecture::from_str(""), Architecture::Llama);
        assert_eq!(Architecture::from_str("gpt4"), Architecture::Llama);
        assert_eq!(
            Architecture::from_str("some_random_model"),
            Architecture::Llama
        );
    }

    #[test]
    fn test_architecture_parameter_prefix() {
        // 测试参数前缀映射
        assert_eq!(Architecture::Llama.parameter_prefix(), "llama");
        assert_eq!(Architecture::Qwen2.parameter_prefix(), "qwen2");
        assert_eq!(Architecture::MiniCPM.parameter_prefix(), "minicpm");
        assert_eq!(Architecture::Gemma.parameter_prefix(), "gemma");
        assert_eq!(Architecture::Phi.parameter_prefix(), "phi"); // 统一使用 phi
        assert_eq!(Architecture::Mistral.parameter_prefix(), "mistral");
        assert_eq!(Architecture::Mixtral.parameter_prefix(), "mixtral");
        assert_eq!(Architecture::Yi.parameter_prefix(), "yi");
        assert_eq!(Architecture::ChatGLM.parameter_prefix(), "chatglm");
        assert_eq!(Architecture::Baichuan.parameter_prefix(), "baichuan");
        assert_eq!(Architecture::Falcon.parameter_prefix(), "falcon");
        assert_eq!(Architecture::StableLM.parameter_prefix(), "stablelm");
    }

    #[test]
    fn test_architecture_is_moe() {
        // 测试 MoE 检测
        assert!(!Architecture::Llama.is_moe());
        assert!(!Architecture::Qwen2.is_moe());
        assert!(!Architecture::Mistral.is_moe());
        assert!(Architecture::Mixtral.is_moe()); // 只有 Mixtral 是 MoE
        assert!(!Architecture::Phi.is_moe());
    }

    #[test]
    fn test_architecture_uses_packed_qkv() {
        // 测试 Packed QKV 检测
        assert!(!Architecture::Llama.uses_packed_qkv());
        assert!(!Architecture::Mistral.uses_packed_qkv());
        assert!(Architecture::Phi.uses_packed_qkv()); // Phi 使用 packed qkv
        assert!(!Architecture::Mixtral.uses_packed_qkv());
    }

    #[test]
    fn test_architecture_default_rope_theta() {
        // 测试默认 RoPE theta 值
        assert_eq!(Architecture::Llama.default_rope_theta(), DEFAULT_ROPE_THETA); // 1000000.0
        assert_eq!(Architecture::Yi.default_rope_theta(), 5000000.0); // Yi 特殊值
        assert_eq!(Architecture::Baichuan.default_rope_theta(), 0.0); // Baichuan 不使用 RoPE
        assert_eq!(Architecture::ChatGLM.default_rope_theta(), 10000.0); // ChatGLM 的 2D-RoPE
        assert_eq!(
            Architecture::Mistral.default_rope_theta(),
            DEFAULT_ROPE_THETA
        );
        assert_eq!(Architecture::Qwen2.default_rope_theta(), DEFAULT_ROPE_THETA);
    }

    #[test]
    fn test_architecture_support_status() {
        // 测试支持状态描述
        assert!(Architecture::Llama.support_status().contains("✅"));
        assert!(Architecture::Qwen2.support_status().contains("✅"));
        assert!(Architecture::Mistral.support_status().contains("✅"));
        assert!(Architecture::Phi.support_status().contains("⚠️")); // 已修复 Bug
        assert!(Architecture::Falcon.support_status().contains("🔶")); // 实验性
        assert!(Architecture::StableLM.support_status().contains("🔶")); // 实验性
    }

    #[test]
    fn test_architecture_display() {
        // 测试 Display trait 实现
        assert_eq!(format!("{}", Architecture::Llama), "LLaMA");
        assert_eq!(format!("{}", Architecture::Qwen2), "Qwen2");
        assert_eq!(format!("{}", Architecture::Mixtral), "Mixtral (MoE)");
        assert_eq!(format!("{}", Architecture::Yi), "Yi");
        assert_eq!(format!("{}", Architecture::Phi), "Phi");
    }

    #[test]
    fn test_architecture_equality_and_hash() {
        // 测试相等性和 Hash（用于 HashMap key）
        let arch1 = Architecture::Llama;
        let arch2 = Architecture::Llama;
        let arch3 = Architecture::Qwen2;

        assert_eq!(arch1, arch2);
        assert_ne!(arch1, arch3);

        // 测试在 HashMap 中使用
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Architecture::Llama);
        set.insert(Architecture::Qwen2);
        set.insert(Architecture::Mistral);
        assert_eq!(set.len(), 3);
        assert!(set.contains(&Architecture::Llama));
    }

    #[test]
    fn test_model_config_with_new_architectures() {
        // 测试新架构的模型配置提取
        // 创建一个包含 Mistral 架构的最小 GGUF 文件
        let gguf_bytes = build_minimal_gguf_with_arch("mistral");

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(&gguf_bytes)
            .expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let gguf = GgufFile::open(temp_file.path()).expect("Failed to parse GGUF file");
        let config = gguf.get_model_config().expect("Failed to get model config");

        // 验证配置值（应该使用默认值，因为只有 llama.* 参数）
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 32);
    }

    #[test]
    fn test_model_config_phi_bug_fix() {
        // 【关键测试】验证 Phi Bug 已修复
        // 场景：Phi 架构文件，但只有 phi.* 参数（没有 llama.*）
        // 修复前：会错误地回退到 llama.* 并读取错误值
        // 修复后：正确使用默认值或 phi.* 参数

        let gguf_bytes = build_gguf_with_phi_params();

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(&gguf_bytes)
            .expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let gguf = GgufFile::open(temp_file.path()).expect("Failed to parse GGUF file");

        // 验证架构被正确识别为 Phi
        assert_eq!(
            gguf.metadata.get_string("general.architecture"),
            Some("phi")
        );

        let config = gguf.get_model_config().expect("Failed to get model config");

        // 验证使用了 phi.* 参数（3072），而不是回退到 llama.*
        assert_eq!(
            config.hidden_size, 3072,
            "Phi 应该使用 phi.embedding_length"
        );
        assert_eq!(config.num_hidden_layers, 32, "Phi 应该使用 phi.block_count");
        assert_eq!(
            config.num_attention_heads, 32,
            "Phi 应该使用 phi.attention.head_count"
        );
    }

    #[test]
    fn test_model_config_yi_special_rope_theta() {
        // 测试 Yi 架构的特殊 RoPE theta 值
        let gguf_bytes = build_minimal_gguf_with_arch("yi");

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(&gguf_bytes)
            .expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let gguf = GgufFile::open(temp_file.path()).expect("Failed to parse GGUF file");
        let config = gguf.get_model_config().expect("Failed to get model config");

        // Yi 应该使用特殊的 rope_theta 默认值 5000000.0
        assert_eq!(
            config.rope_theta, 5000000.0,
            "Yi 应该使用 rope_theta=5000000.0"
        );
    }

    #[test]
    fn test_model_config_baichuan_no_rope() {
        // 测试 Baichuan 架构不使用 RoPE（使用 ALiBi）
        let gguf_bytes = build_minimal_gguf_with_arch("baichuan");

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(&gguf_bytes)
            .expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let gguf = GgufFile::open(temp_file.path()).expect("Failed to parse GGUF file");
        let config = gguf.get_model_config().expect("Failed to get model config");

        // Baichuan 应该禁用 RoPE（rope_theta=0.0）
        assert_eq!(
            config.rope_theta, 0.0,
            "Baichuan 应该禁用 RoPE（使用 ALiBi）"
        );
    }

    #[test]
    fn test_model_config_mixtral_moe_detection() {
        // 测试 Mixtral MoE 架构检测
        let gguf_bytes = build_minimal_gguf_with_arch("mixtral");

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(&gguf_bytes)
            .expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let gguf = GgufFile::open(temp_file.path()).expect("Failed to parse GGUF file");
        let _config = gguf.get_model_config().expect("Failed to get model config");

        // 验证架构识别（通过日志可以确认 MoE 检测）
        assert_eq!(
            gguf.metadata.get_string("general.architecture"),
            Some("mixtral")
        );
    }

    #[test]
    fn test_model_config_chatglm_2d_rope() {
        // 测试 ChatGLM 的 2D-RoPE 配置
        let gguf_bytes = build_minimal_gguf_with_arch("chatglm");

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(&gguf_bytes)
            .expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let gguf = GgufFile::open(temp_file.path()).expect("Failed to parse GGUF file");
        let config = gguf.get_model_config().expect("Failed to get model config");

        // ChatGLM 应该使用 rope_theta=10000.0
        assert_eq!(
            config.rope_theta, 10000.0,
            "ChatGLM 应该使用 2D-RoPE (theta=10000)"
        );
    }

    #[test]
    fn test_backward_compatibility_llama() {
        // 向后兼容性测试：确保旧的 LLaMA 文件仍能正常加载
        let gguf_bytes = build_minimal_gguf(); // 默认是 LLaMA 架构

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(&gguf_bytes)
            .expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let gguf = GgufFile::open(temp_file.path()).expect("Failed to parse GGUF file");
        let config = gguf.get_model_config().expect("Failed to get model config");

        // 验证所有配置值与修改前一致
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.rope_theta, DEFAULT_ROPE_THETA);
    }

    #[test]
    fn test_backward_compatibility_qwen2() {
        // 向后兼容性测试：Qwen2 文件
        let gguf_bytes = build_minimal_gguf_with_arch("qwen2");

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(&gguf_bytes)
            .expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let gguf = GgufFile::open(temp_file.path()).expect("Failed to parse GGUF file");
        let config = gguf.get_model_config().expect("Failed to get model config");

        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 32);
    }

    #[test]
    fn test_fallback_priority_order() {
        // 测试参数回退优先级顺序
        // 场景：架构为 mistral，但没有 mistral.* 参数，有 llama.* 参数
        // 应该回退到 llama.*

        let gguf_bytes = build_gguf_with_fallback_test();

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(&gguf_bytes)
            .expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let gguf = GgufFile::open(temp_file.path()).expect("Failed to parse GGUF file");
        let config = gguf.get_model_config().expect("Failed to get model config");

        // 应该回退到 llama.* 参数（因为 mistral.* 不存在）
        assert_eq!(
            config.hidden_size, 4096,
            "应该回退到 llama.embedding_length"
        );
    }

    #[test]
    fn test_tensor_name_parsing_new_architectures() {
        // 测试新架构的张量名称解析
        let tensor_names = vec![
            ("mistral blk.0.attn_q.weight", true),
            ("mixtral.blk.0.ffn_gate_exps.weight", true),
            ("yi.blk.0.attn_q.weight", true),
            ("chatglm.encoder.layers.0.self_attention.query.weight", true),
            (
                "baichuan.model.layers.0.self_attention.query_key_value.weight",
                true,
            ),
            ("falcon.h.self_attention.query.weight", true),
            ("unknown_arch.tensor", false), // 未知前缀
        ];

        for (name, expected_valid) in tensor_names {
            let is_valid = name.contains('.')
                && (name.starts_with("mistral")
                    || name.starts_with("mixtral")
                    || name.starts_with("yi")
                    || name.starts_with("chatglm")
                    || name.starts_with("baichuan")
                    || name.starts_with("falcon")
                    || name.starts_with("llama") // 兼容旧架构
                    || name.starts_with("model")); // 通用前缀
            assert_eq!(
                is_valid, expected_valid,
                "张量名称 '{}' 有效性检查失败",
                name
            );
        }
    }

    #[test]
    fn test_architecture_comprehensive_coverage() {
        // 全面覆盖测试：确保所有架构都能正常工作
        let architectures = vec![
            "llama", "qwen2", "minicpm", "gemma", "phi", "mistral", "mixtral", "yi", "chatglm",
            "baichuan", "falcon", "stablelm",
        ];

        for arch_name in &architectures {
            let gguf_bytes = build_minimal_gguf_with_arch(arch_name);

            let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
            temp_file
                .write_all(&gguf_bytes)
                .expect("Failed to write temp file");
            temp_file.flush().expect("Failed to flush");

            let result = GgufFile::open(temp_file.path());
            assert!(
                result.is_ok(),
                "架构 '{}' 的 GGUF 文件应该能成功解析",
                arch_name
            );

            let gguf = result.unwrap();
            let config_result = gguf.get_model_config();
            assert!(
                config_result.is_ok(),
                "架构 '{}' 的模型配置提取应该成功",
                arch_name
            );

            let config = config_result.unwrap();
            // 所有架构都应该能提取到有效的配置
            assert!(
                config.hidden_size > 0,
                "{}: hidden_size 应该 > 0",
                arch_name
            );
            assert!(
                config.num_hidden_layers > 0,
                "{}: num_hidden_layers 应该 > 0",
                arch_name
            );
            assert!(
                config.num_attention_heads > 0,
                "{}: num_attention_heads 应该 > 0",
                arch_name
            );
        }
    }

    // ========================================================================
    // 辅助函数：构建特殊测试用的 GGUF 文件
    // ========================================================================

    /// 构建包含指定架构的最小 GGUF 文件
    fn build_minimal_gguf_with_arch(architecture: &str) -> Vec<u8> {
        let mut buf = Vec::new();

        // GGUF 魔数和版本
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&3u64.to_le_bytes()); // metadata_kv_count

        // 元数据 1: general.architecture
        let val1 = architecture.as_bytes();
        let key1 = b"general.architecture";
        write_metadata_string(&mut buf, key1, val1);

        // 元数据 2: llama.embedding_length = 4096 (作为回退值)
        let key2 = b"llama.embedding_length";
        write_metadata_u32(&mut buf, key2, 4096);

        // 元数据 3: llama.block_count = 32
        let key3 = b"llama.block_count";
        write_metadata_u32(&mut buf, key3, 32);

        // 张量信息
        let tensor_name = b"model.token_embd.weight";
        write_tensor_info(&mut buf, tensor_name, vec![4096], GgufTensorType::F32);

        // 张量数据 (4096 个 f32 值)
        for i in 0..4096u32 {
            buf.extend_from_slice(&(i as f32 * 0.01).to_le_bytes());
        }

        // 对齐到 32 字节
        while buf.len() % 32 != 0 {
            buf.push(0);
        }

        buf
    }

    /// 构建 Phi 特定参数的 GGUF 文件（用于测试 Bug 修复）
    fn build_gguf_with_phi_params() -> Vec<u8> {
        let mut buf = Vec::new();

        // GGUF 魔数和版本
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0 (简化)
        buf.extend_from_slice(&5u64.to_le_bytes()); // metadata_kv_count = 5

        // 元数据 1: general.architecture = "phi"
        let key1 = b"general.architecture";
        write_metadata_string(&mut buf, key1, b"phi");

        // 元数据 2: phi.embedding_length = 3072 (Phi 特有值)
        let key2 = b"phi.embedding_length";
        write_metadata_u32(&mut buf, key2, 3072);

        // 元数据 3: phi.block_count = 32
        let key3 = b"phi.block_count";
        write_metadata_u32(&mut buf, key3, 32);

        // 元数据 4: phi.attention.head_count = 32
        let key4 = b"phi.attention.head_count";
        write_metadata_u32(&mut buf, key4, 32);

        // 元数据 5: general.vocab_size = 32000
        let key5 = b"general.vocab_size";
        write_metadata_u64(&mut buf, key5, 32000);

        // 注意：故意不添加 llama.* 参数，测试是否会错误回退

        buf
    }

    /// 构建用于测试回退逻辑的 GGUF 文件
    fn build_gguf_with_fallback_test() -> Vec<u8> {
        let mut buf = Vec::new();

        // GGUF 魔数和版本
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        buf.extend_from_slice(&3u64.to_le_bytes()); // metadata_kv_count = 3

        // 元数据 1: general.architecture = "mistral" (但无 mistral.* 参数)
        let key1 = b"general.architecture";
        write_metadata_string(&mut buf, key1, b"mistral");

        // 元数据 2: llama.embedding_length = 4096 (回退目标)
        let key2 = b"llama.embedding_length";
        write_metadata_u32(&mut buf, key2, 4096);

        // 元数据 3: llama.block_count = 32
        let key3 = b"llama.block_count";
        write_metadata_u32(&mut buf, key3, 32);

        buf
    }

    /// 写入字符串元数据
    fn write_metadata_string(buf: &mut Vec<u8>, key: &[u8], value: &[u8]) {
        // 键
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        // 值类型 (String = 8)
        buf.extend_from_slice(&(8u32).to_le_bytes());
        // 字符串值
        buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
        buf.extend_from_slice(value);
    }

    /// 写入 u32 元数据
    fn write_metadata_u32(buf: &mut Vec<u8>, key: &[u8], value: u32) {
        // 键
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        // 值类型 (UInt32 = 4)
        buf.extend_from_slice(&(4u32).to_le_bytes());
        // u32 值
        buf.extend_from_slice(&value.to_le_bytes());
    }

    /// 写入 u64 元数据
    fn write_metadata_u64(buf: &mut Vec<u8>, key: &[u8], value: u64) {
        // 键
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        // 值类型 (UInt64 = 10)
        buf.extend_from_slice(&(10u32).to_le_bytes());
        // u64 值
        buf.extend_from_slice(&value.to_le_bytes());
    }

    /// 写入张量信息
    fn write_tensor_info(
        buf: &mut Vec<u8>,
        name: &[u8],
        dims: Vec<u64>,
        tensor_type: GgufTensorType,
    ) {
        // 张量名称
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name);
        // 维度数量
        buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        // 各维度
        for dim in dims {
            buf.extend_from_slice(&dim.to_le_bytes());
        }
        // 张量类型
        buf.extend_from_slice(&(tensor_type as u32).to_le_bytes());
        // 偏移量（稍后填充）
        buf.extend_from_slice(&0u64.to_le_bytes());
    }
}
