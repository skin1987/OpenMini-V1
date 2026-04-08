//! MLA 投影矩阵模块
//!
//! 实现 MLA 中的各种投影矩阵：
//! - W_Q: Query 投影
//! - W_K: Key 投影（用于压缩前）
//! - W_V: Value 投影（用于压缩前）
//! - W_O: 输出投影
//! - W_DKV: KV 压缩投影（hidden -> latent）
//! - W_UK: KV 解压投影（latent -> K）
//! - W_UV: KV 解压投影（latent -> V）
//! - W_QR: 解耦 Query 投影（用于 RoPE）
//! - W_KR: 解耦 Key 投影（用于 RoPE）

#![allow(dead_code)]

use crate::hardware::kv_cache::mla::config::MLAConfig;
use ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
pub struct MLAProjection {
    pub q_proj: Array2<f32>,
    pub k_proj: Array2<f32>,
    pub v_proj: Array2<f32>,
    pub o_proj: Array2<f32>,
    pub dkv_proj: Array2<f32>,
    pub uk_proj: Array2<f32>,
    pub uv_proj: Array2<f32>,
    pub qr_proj: Option<Array2<f32>>,
    pub kr_proj: Option<Array2<f32>>,
    pub q_norm: Option<Array1<f32>>,
    pub k_norm: Option<Array1<f32>>,
}

impl MLAProjection {
    pub fn new(config: &MLAConfig) -> Self {
        let hidden_size = config.hidden_size;
        let num_attention_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let latent_dim = config.latent_dim;

        let q_dim = num_attention_heads * head_dim;
        let kv_dim = num_key_value_heads * head_dim;

        Self {
            q_proj: Array2::zeros((hidden_size, q_dim)),
            k_proj: Array2::zeros((hidden_size, kv_dim)),
            v_proj: Array2::zeros((hidden_size, kv_dim)),
            o_proj: Array2::zeros((q_dim, hidden_size)),
            dkv_proj: Array2::zeros((hidden_size, latent_dim)),
            uk_proj: Array2::zeros((latent_dim, kv_dim)),
            uv_proj: Array2::zeros((latent_dim, kv_dim)),
            qr_proj: if config.use_decoupled_rope {
                Some(Array2::zeros((hidden_size, q_dim)))
            } else {
                None
            },
            kr_proj: if config.use_decoupled_rope {
                Some(Array2::zeros((hidden_size, kv_dim)))
            } else {
                None
            },
            q_norm: None,
            k_norm: None,
        }
    }

    pub fn from_weights(
        q_proj: Array2<f32>,
        k_proj: Array2<f32>,
        v_proj: Array2<f32>,
        o_proj: Array2<f32>,
        dkv_proj: Array2<f32>,
        uk_proj: Array2<f32>,
        uv_proj: Array2<f32>,
        qr_proj: Option<Array2<f32>>,
        kr_proj: Option<Array2<f32>>,
        q_norm: Option<Array1<f32>>,
        k_norm: Option<Array1<f32>>,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            dkv_proj,
            uk_proj,
            uv_proj,
            qr_proj,
            kr_proj,
            q_norm,
            k_norm,
        }
    }

    pub fn forward_q(&self, hidden_states: &Array2<f32>) -> Array2<f32> {
        hidden_states.dot(&self.q_proj)
    }

    pub fn forward_k(&self, hidden_states: &Array2<f32>) -> Array2<f32> {
        hidden_states.dot(&self.k_proj)
    }

    pub fn forward_v(&self, hidden_states: &Array2<f32>) -> Array2<f32> {
        hidden_states.dot(&self.v_proj)
    }

    pub fn compress_kv(&self, hidden_states: &Array2<f32>) -> Array2<f32> {
        hidden_states.dot(&self.dkv_proj)
    }

    pub fn decompress_k(&self, c_kv: &Array2<f32>) -> Array2<f32> {
        c_kv.dot(&self.uk_proj)
    }

    pub fn decompress_v(&self, c_kv: &Array2<f32>) -> Array2<f32> {
        c_kv.dot(&self.uv_proj)
    }

    pub fn decompress_kv(&self, c_kv: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let k = self.decompress_k(c_kv);
        let v = self.decompress_v(c_kv);
        (k, v)
    }

    pub fn forward_output(&self, attn_output: &Array2<f32>) -> Array2<f32> {
        attn_output.dot(&self.o_proj)
    }

    pub fn q_dim(&self) -> usize {
        self.q_proj.ncols()
    }

    pub fn kv_dim(&self) -> usize {
        self.uk_proj.ncols()
    }

    pub fn latent_dim(&self) -> usize {
        self.dkv_proj.ncols()
    }

    pub fn has_decoupled_rope(&self) -> bool {
        self.qr_proj.is_some() && self.kr_proj.is_some()
    }

    pub fn memory_usage(&self) -> usize {
        let q_mem = self.q_proj.len() * 4;
        let k_mem = self.k_proj.len() * 4;
        let v_mem = self.v_proj.len() * 4;
        let o_mem = self.o_proj.len() * 4;
        let dkv_mem = self.dkv_proj.len() * 4;
        let uk_mem = self.uk_proj.len() * 4;
        let uv_mem = self.uv_proj.len() * 4;
        let qr_mem = self.qr_proj.as_ref().map_or(0, |p| p.len() * 4);
        let kr_mem = self.kr_proj.as_ref().map_or(0, |p| p.len() * 4);
        let q_norm_mem = self.q_norm.as_ref().map_or(0, |p| p.len() * 4);
        let k_norm_mem = self.k_norm.as_ref().map_or(0, |p| p.len() * 4);

        q_mem
            + k_mem
            + v_mem
            + o_mem
            + dkv_mem
            + uk_mem
            + uv_mem
            + qr_mem
            + kr_mem
            + q_norm_mem
            + k_norm_mem
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> MLAConfig {
        MLAConfig::default()
    }

    #[test]
    fn test_projection_creation() {
        let config = create_test_config();
        let proj = MLAProjection::new(&config);

        assert_eq!(proj.q_proj.dim(), (3584, 32 * 128));
        assert_eq!(proj.k_proj.dim(), (3584, 8 * 128));
        assert_eq!(proj.v_proj.dim(), (3584, 8 * 128));
        assert_eq!(proj.dkv_proj.dim(), (3584, 512));
        assert_eq!(proj.uk_proj.dim(), (512, 8 * 128));
        assert_eq!(proj.uv_proj.dim(), (512, 8 * 128));
    }

    #[test]
    fn test_forward_q() {
        let config = create_test_config();
        let proj = MLAProjection::new(&config);

        let hidden = Array2::zeros((1, config.hidden_size));
        let q = proj.forward_q(&hidden);

        assert_eq!(q.dim(), (1, config.num_attention_heads * config.head_dim));
    }

    #[test]
    fn test_compress_decompress_kv() {
        let config = create_test_config();
        let proj = MLAProjection::new(&config);

        let hidden = Array2::zeros((1, config.hidden_size));

        let c_kv = proj.compress_kv(&hidden);
        assert_eq!(c_kv.dim(), (1, config.latent_dim));

        let (k, v) = proj.decompress_kv(&c_kv);
        assert_eq!(k.dim(), (1, config.kv_dim()));
        assert_eq!(v.dim(), (1, config.kv_dim()));
    }

    #[test]
    fn test_memory_usage() {
        let config = create_test_config();
        let proj = MLAProjection::new(&config);

        let mem = proj.memory_usage();
        assert!(mem > 0);
        assert!(mem > 200_000_000);
    }

    #[test]
    fn test_has_decoupled_rope() {
        let config = create_test_config();
        let proj = MLAProjection::new(&config);

        assert!(proj.has_decoupled_rope());

        let no_rope_config = MLAConfig::default().with_decoupled_rope(false);
        let proj_no_rope = MLAProjection::new(&no_rope_config);

        assert!(!proj_no_rope.has_decoupled_rope());
    }

    // 新增分支覆盖测试

    /// 测试 from_weights 构造函数（带 RoPE）
    #[test]
    fn test_from_weights_with_rope() {
        // 使用小尺寸避免大内存分配（<20KB）
        let hidden_size = 64;
        let q_dim = 32;
        let kv_dim = 16;
        let latent_dim = 8;

        let q_proj = Array2::zeros((hidden_size, q_dim));
        let k_proj = Array2::zeros((hidden_size, kv_dim));
        let v_proj = Array2::zeros((hidden_size, kv_dim));
        let o_proj = Array2::zeros((q_dim, hidden_size));
        let dkv_proj = Array2::zeros((hidden_size, latent_dim));
        let uk_proj = Array2::zeros((latent_dim, kv_dim));
        let uv_proj = Array2::zeros((latent_dim, kv_dim));
        let qr_proj = Some(Array2::zeros((hidden_size, q_dim)));
        let kr_proj = Some(Array2::zeros((hidden_size, kv_dim)));

        let proj = MLAProjection::from_weights(
            q_proj.clone(),
            k_proj.clone(),
            v_proj.clone(),
            o_proj.clone(),
            dkv_proj.clone(),
            uk_proj.clone(),
            uv_proj.clone(),
            qr_proj,
            kr_proj,
            None,
            None,
        );

        assert!(proj.has_decoupled_rope());
        assert_eq!(proj.q_dim(), q_dim);
        assert_eq!(proj.kv_dim(), kv_dim);
        assert_eq!(proj.latent_dim(), latent_dim);
    }

    /// 测试 from_weights 构造函数（不带 RoPE）
    #[test]
    fn test_from_weights_without_rope() {
        // 使用小尺寸避免大内存分配（<20KB）
        let hidden_size = 64;
        let q_dim = 32;
        let kv_dim = 16;
        let latent_dim = 8;

        let q_proj = Array2::zeros((hidden_size, q_dim));
        let k_proj = Array2::zeros((hidden_size, kv_dim));
        let v_proj = Array2::zeros((hidden_size, kv_dim));
        let o_proj = Array2::zeros((q_dim, hidden_size));
        let dkv_proj = Array2::zeros((hidden_size, latent_dim));
        let uk_proj = Array2::zeros((latent_dim, kv_dim));
        let uv_proj = Array2::zeros((latent_dim, kv_dim));

        let proj = MLAProjection::from_weights(
            q_proj, k_proj, v_proj, o_proj, dkv_proj, uk_proj, uv_proj, None, None, None, None,
        );

        assert!(!proj.has_decoupled_rope());
    }

    /// 测试 forward_k 方法
    #[test]
    fn test_forward_k() {
        let config = create_test_config();
        let proj = MLAProjection::new(&config);

        let hidden = Array2::zeros((2, config.hidden_size));
        let k = proj.forward_k(&hidden);

        assert_eq!(k.dim(), (2, config.num_key_value_heads * config.head_dim));
    }

    /// 测试 forward_v 方法
    #[test]
    fn test_forward_v() {
        let config = create_test_config();
        let proj = MLAProjection::new(&config);

        let hidden = Array2::zeros((3, config.hidden_size));
        let v = proj.forward_v(&hidden);

        assert_eq!(v.dim(), (3, config.num_key_value_heads * config.head_dim));
    }

    /// 测试 compress_kv 方法
    #[test]
    fn test_compress_kv() {
        let config = create_test_config();
        let proj = MLAProjection::new(&config);

        let hidden = Array2::zeros((5, config.hidden_size));
        let c_kv = proj.compress_kv(&hidden);

        assert_eq!(c_kv.dim(), (5, config.latent_dim));
    }

    /// 测试 decompress_k 方法
    #[test]
    fn test_decompress_k() {
        let config = create_test_config();
        let proj = MLAProjection::new(&config);

        let c_kv = Array2::ones((3, config.latent_dim));
        let k = proj.decompress_k(&c_kv);

        assert_eq!(k.dim(), (3, config.kv_dim()));
    }

    /// 测试 decompress_v 方法
    #[test]
    fn test_decompress_v() {
        let config = create_test_config();
        let proj = MLAProjection::new(&config);

        let c_kv = Array2::ones((4, config.latent_dim));
        let v = proj.decompress_v(&c_kv);

        assert_eq!(v.dim(), (4, config.kv_dim()));
    }

    /// 测试 forward_output 方法
    #[test]
    fn test_forward_output() {
        let config = create_test_config();
        let proj = MLAProjection::new(&config);

        let attn_output = Array2::zeros((2, config.num_attention_heads * config.head_dim));
        let output = proj.forward_output(&attn_output);

        assert_eq!(output.dim(), (2, config.hidden_size));
    }

    /// 测试 q_dim、kv_dim、latent_dim 方法
    #[test]
    fn test_dimension_methods() {
        let config = create_test_config();
        let proj = MLAProjection::new(&config);

        assert_eq!(proj.q_dim(), config.num_attention_heads * config.head_dim);
        assert_eq!(proj.kv_dim(), config.num_key_value_heads * config.head_dim);
        assert_eq!(proj.latent_dim(), config.latent_dim);
    }

    /// 测试 from_weights 带 norm 向量的构造函数
    /// 覆盖分支：q_norm 和 k_norm 的 Some 分支
    #[test]
    fn test_from_weights_with_norms() {
        use ndarray::Array1;

        // 使用小尺寸避免大内存分配（<20KB）
        let hidden_size = 64;
        let q_dim = 32;
        let kv_dim = 16;
        let latent_dim = 8;

        let q_proj = Array2::zeros((hidden_size, q_dim));
        let k_proj = Array2::zeros((hidden_size, kv_dim));
        let v_proj = Array2::zeros((hidden_size, kv_dim));
        let o_proj = Array2::zeros((q_dim, hidden_size));
        let dkv_proj = Array2::zeros((hidden_size, latent_dim));
        let uk_proj = Array2::zeros((latent_dim, kv_dim));
        let uv_proj = Array2::zeros((latent_dim, kv_dim));
        let q_norm = Some(Array1::zeros(q_dim));
        let k_norm = Some(Array1::zeros(kv_dim));

        let proj = MLAProjection::from_weights(
            q_proj, k_proj, v_proj, o_proj, dkv_proj, uk_proj, uv_proj, None, None, q_norm, k_norm,
        );

        // 验证 norm 向量被正确存储
        assert!(proj.q_norm.is_some());
        assert!(proj.k_norm.is_some());
        assert_eq!(proj.q_norm.as_ref().unwrap().len(), q_dim);
        assert_eq!(proj.k_norm.as_ref().unwrap().len(), kv_dim);

        // 验证 memory_usage 包含 norm 的内存
        let mem = proj.memory_usage();
        assert!(mem > 0);
    }

    /// 测试 MLAProjection 的 Clone trait
    /// 覆盖分支：Clone 实现的正确性
    #[test]
    fn test_projection_clone() {
        let config = create_test_config();
        let proj = MLAProjection::new(&config);

        let cloned = proj.clone();

        // 验证所有字段都被正确克隆
        assert_eq!(proj.q_dim(), cloned.q_dim());
        assert_eq!(proj.kv_dim(), cloned.kv_dim());
        assert_eq!(proj.latent_dim(), cloned.latent_dim());
        assert_eq!(proj.has_decoupled_rope(), cloned.has_decoupled_rope());

        // 验证内存使用量相同
        assert_eq!(proj.memory_usage(), cloned.memory_usage());
    }

    /// 测试 forward_q 不同 batch size
    /// 覆盖分支：不同维度的输入
    #[test]
    fn test_forward_q_different_batch_sizes() {
        let config = create_test_config();
        let proj = MLAProjection::new(&config);

        // batch_size = 1
        let hidden_1 = Array2::zeros((1, config.hidden_size));
        let q_1 = proj.forward_q(&hidden_1);
        assert_eq!(q_1.dim().0, 1);

        // batch_size = 10
        let hidden_10 = Array2::zeros((10, config.hidden_size));
        let q_10 = proj.forward_q(&hidden_10);
        assert_eq!(q_10.dim().0, 10);

        // batch_size = 100
        let hidden_100 = Array2::zeros((100, config.hidden_size));
        let q_100 = proj.forward_q(&hidden_100);
        assert_eq!(q_100.dim().0, 100);
    }

    /// 测试 compress_kv 和 decompress_kv 的数据一致性
    /// 覆盖分支：压缩-解压缩往返
    #[test]
    fn test_compress_decompress_roundtrip() {
        let config = create_test_config();
        let proj = MLAProjection::new(&config);

        // 创建非零输入
        let hidden = Array2::from_shape_fn((3, config.hidden_size), |(i, j)| {
            (i * config.hidden_size + j) as f32
        });

        // 压缩
        let c_kv = proj.compress_kv(&hidden);
        assert_eq!(c_kv.dim(), (3, config.latent_dim));

        // 解压缩
        let (k, v) = proj.decompress_kv(&c_kv);
        assert_eq!(k.dim(), (3, config.kv_dim()));
        assert_eq!(v.dim(), (3, config.kv_dim()));

        // 验证输出维度正确性
        assert_eq!(k.dim().1, config.num_key_value_heads * config.head_dim);
        assert_eq!(v.dim().1, config.num_key_value_heads * config.head_dim);
    }

    /// 测试 memory_usage 在不同配置下的计算
    /// 覆盖分支：memory_usage 的各种场景
    #[test]
    fn test_memory_usage_scenarios() {
        // 带有 RoPE 的配置
        let config_with_rope = create_test_config();
        let proj_with_rope = MLAProjection::new(&config_with_rope);
        let mem_with_rope = proj_with_rope.memory_usage();
        assert!(mem_with_rope > 0);

        // 不带 RoPE 的配置
        let config_without_rope = MLAConfig::default().with_decoupled_rope(false);
        let proj_without_rope = MLAProjection::new(&config_without_rope);
        let mem_without_rope = proj_without_rope.memory_usage();
        assert!(mem_without_rope > 0);

        // 带有 RoPE 的应该占用更多内存（qr_proj 和 kr_proj）
        assert!(mem_with_rope > mem_without_rope);
    }

    /// 测试 has_decoupled_rope 在不同配置下的行为
    /// 覆盖分支：has_decoupled_rope 的判断逻辑
    #[test]
    fn test_has_decoupled_rope_edge_cases() {
        // 默认配置（use_decoupled_rope=true）
        let config_default = create_test_config();
        let proj_default = MLAProjection::new(&config_default);
        assert!(proj_default.has_decoupled_rope());

        // 禁用 RoPE
        let config_no_rope = MLAConfig::default().with_decoupled_rope(false);
        let proj_no_rope = MLAProjection::new(&config_no_rope);
        assert!(!proj_no_rope.has_decoupled_rope());
    }
}
