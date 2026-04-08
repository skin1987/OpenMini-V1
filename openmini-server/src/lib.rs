//! OpenMini Server - 高性能 LLM 推理服务器
//!
//! 本库提供完整的推理服务功能：
//! - 模型加载与推理
//! - GPU/CPU 自适应加速
//! - gRPC 服务接口
//! - 多模态支持

// 允许文档缺失（内部实现库，文档可后续补充）
#![allow(missing_docs)]
// 允许未使用代码（公共API保留）
#![allow(dead_code)]

pub mod config;
pub mod hardware;
pub mod kernel;
pub mod logging;
pub mod model;
pub mod monitoring;
pub mod rl;
pub mod service;
