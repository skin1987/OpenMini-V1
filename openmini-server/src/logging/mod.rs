//! OpenMini 结构化日志系统
//!
//! 提供JSON格式的结构化日志，支持：
//! - 多级别日志 (trace/debug/info/warn/error)
//! - 结构化字段 (request_id, model_name, latency_ms等)
//! - 日志采样 (避免高频日志刷屏)
//! - 上下文传播 (span/tracing)
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use openmini_server::logging::init_json_logging;
//! use openmini_server::logging::{RequestFields, ModelFields};
//! use tracing::{info_span, info};
//!
//! // 初始化JSON日志
//! init_json_logging("info");
//!
//! // 使用结构化字段记录请求日志
//! let span = info_span!("request",
//!     request_id = %RequestFields::new("req-001"),
//!     model_name = %ModelFields::new("llama-3-8b"),
//!     latency_ms = 42
//! );
//! ```

pub mod json_logger;
pub mod structured_fields;
pub mod log_sampler;

pub use json_logger::init_json_logging;
pub use structured_fields::*;
