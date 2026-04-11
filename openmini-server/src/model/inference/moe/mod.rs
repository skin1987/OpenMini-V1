//! MoE (Mixture of Experts) 模块
//!
//! 提供混合专家模型的实现和优化策略：
//! - LongCat: 双分支长上下文优化 MoE 架构

pub mod longcat;

// 重导出常用类型
pub use longcat::{
    ExpertFFN, ExpertWeights, FusionLayer, FusionStrategy, LongCatConfig, LongCatMoE,
    LongCatStats, LightweightExpert, MlaCompressor, StandardMoEFormat, ZeroExpert,
};
