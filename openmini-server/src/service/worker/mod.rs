//! Worker 模块入口
//!
//! 导出 Worker 进程相关的类型和函数

#![allow(dead_code)]

pub mod pool;
pub mod worker;

pub use pool::WorkerPool;
pub use worker::{Worker, get_worker_id, is_worker_process};
