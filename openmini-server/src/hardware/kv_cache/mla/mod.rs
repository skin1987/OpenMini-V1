//! MLA 多头潜在注意力模块
//!
//! 通过低秩分解压缩 KV 缓存，实现约 93.3% 的压缩率
//!
//! # 核心原理
//!
//! MLA (Multi-head Latent Attention) 通过将高维 KV 投影压缩到低维潜在空间：
//!
//! ```text
//! 标准 Attention:
//!   K = x @ W_K  (seq_len, num_kv_heads × head_dim)
//!   V = x @ W_V  (seq_len, num_kv_heads × head_dim)
//!
//! MLA:
//!   c_KV = x @ W_DKV  (seq_len, latent_dim)  // 压缩
//!   K = c_KV @ W_UK  // 解压
//!   V = c_KV @ W_UV  // 解压
//! ```

pub mod config;
pub mod projection;
pub mod latent_cache;
pub mod attention;

pub use config::MLAConfig;
pub use projection::MLAProjection;
pub use latent_cache::MLALatentCache;
pub use attention::{MLAAttention, RoPECache, mla_attention_forward};
