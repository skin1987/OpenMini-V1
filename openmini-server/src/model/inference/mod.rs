//! OpenMini 推理引擎模块
//!
//! 本模块提供原生 MoE Transformer 推理功能：
//! - MoE (Mixture of Experts) 架构
//! - MLA (Multi-head Latent Attention)
//! - 支持 GGUF 模型加载
//! - 多模态推理（文本 + 图像）

pub mod gguf;
pub mod model;
pub mod quant;
pub mod quant_simd;
pub mod quant_loader;
pub mod context;
pub mod memory;
pub mod engine;
pub mod dsa;
pub mod mtp;
pub mod sampler;
pub mod tokenizer;
pub mod generator;
pub mod inference;
pub mod image_preprocess;
pub mod error;
pub mod flash_attention_3;
pub mod speculative_decoding_v2;
pub mod continuous_batching;
pub mod attn_res;

// 重导出常用类型
pub use error::{InferenceError, InferenceResult};
pub use image_preprocess::{ImagePreprocessor, ImagePreprocessorConfig};

pub use inference::InferenceEngine;
