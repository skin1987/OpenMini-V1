//! Worker 模块入口
//!
//! 导出 Worker 进程相关的类型和函数

#![allow(dead_code)]

pub mod async_pool;
pub mod pool;
#[allow(clippy::module_inception)]
pub mod worker;

pub use async_pool::{AsyncInferencePool, InferenceTask};
// Worker, get_worker_id, is_worker_process 可按需启用:
// pub use worker::{get_worker_id, is_worker_process, Worker};
