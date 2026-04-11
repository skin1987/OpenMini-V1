//! OpenMini Server - 高性能 LLM 推理服务器
//!
//! 本库提供完整的推理服务功能：
//! - 模型加载与推理
//! - GPU/CPU 自适应加速
//! - gRPC 服务接口
//! - 多模态支持
//! - TypeScript 类型自动生成（用于前端类型安全）

// Clippy 配置：允许预期的 cfg 条件值
#![allow(unexpected_cfgs)]
#![allow(clippy::too_many_arguments)]

pub mod benchmark;
pub mod config;
pub mod db;
pub mod distributed;
pub mod enterprise;
pub mod error;
pub mod hardware;
pub mod kernel;
pub mod logging;
pub mod model;
pub mod monitoring;
pub mod rl;
pub mod service;
pub mod training;
/// TypeScript 类型导出模块
///
/// 提供自动生成 TypeScript 类型定义的功能，
/// 确保前后端 API 类型一致性。
///
/// # 使用方式
///
/// ```rust,ignore
/// use openmini_server::types;
///
/// // 导出所有 TypeScript 类型到前端目录
/// types::export_all();
/// ```
pub mod types;
