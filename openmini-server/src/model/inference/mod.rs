//! OpenMini 推理引擎模块
//!
//! 本模块提供原生 MoE Transformer 推理功能：
//! - MoE (Mixture of Experts) 架构
//! - MLA (Multi-head Latent Attention)
//! - 支持 GGUF 模型加载
//! - 多模态推理（文本 + 图像）

pub mod attn_res;
pub mod context;
pub mod continuous_batching;
pub mod dsa;
pub mod engine;
pub mod error;
pub mod flash_attention_3;
pub mod generator;
pub mod gguf;
pub mod image_preprocess;
pub mod inference;
pub mod memory;
pub mod model;
pub mod mtp;
pub mod quant;
pub mod quant_loader;
pub mod quant_simd;
pub mod sampler;
pub mod speculative_decoding_v2;
pub mod tokenizer;

// 重导出常用类型
pub use error::{InferenceError, InferenceResult};
pub use image_preprocess::{ImagePreprocessor, ImagePreprocessorConfig};

pub use inference::InferenceEngine;
