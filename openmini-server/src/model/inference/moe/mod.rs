//! MoE (Mixture of Experts) 模块
//!
//! 提供混合专家模型的实现和优化策略：
//! - LongCat: 双分支长上下文优化 MoE 架构
//! - BlockFFN: Chunk级MoE稀疏优化模块

pub mod blockffn;
pub mod longcat;

// 重导出常用类型
pub use blockffn::{
    BlockFFN, BlockFFNConfig, ClsStatistics, FFNChunk, ReLURMSNormRouter,
    SdCompatibilityResult, SdFastPathConfig,
};
pub use longcat::{
    ExpertFFN, ExpertWeights, FusionLayer, FusionStrategy, LongCatConfig, LongCatMoE,
    LongCatStats, LightweightExpert, MlaCompressor, StandardMoEFormat, ZeroExpert,
};
