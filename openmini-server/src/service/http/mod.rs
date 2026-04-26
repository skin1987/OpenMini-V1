//! HTTP REST API 服务模块
//!
//! 提供基于 axum 的 HTTP REST API 实现，支持：
//! - Chat Completion API（兼容 OpenAI 格式）
//! - 图像理解 API
//! - TTS / STT 语音 API
//! - 健康检查和监控端点
//! - Server-Sent Events (SSE) 流式输出
//!
//! ## 功能特性
//!
//! - **RESTful 设计**: 遵循 REST 规范的 API 设计
//! - **流式支持**: SSE 协议实现流式聊天输出
//! - **中间件**: 日志、CORS、超时、请求体大小限制
//! - **优雅关闭**: 支持 SIGTERM/SIGINT 信号的 graceful shutdown
//! - **Prometheus 集成**: 导出监控指标用于系统观测
//!
//! ## 使用示例
//!
//! ```ignore
//! use openmini_server::service::http::start_http_server;
//!
//! #[tokio::main]
//! async fn main() {
//!     let handle = start_http_server("0.0.0.0:8080", None).await.unwrap();
//!     handle.await.unwrap();
//! }
//! ```

pub mod types;
pub mod handlers;
pub mod middleware;
pub mod server;
pub mod sensitive_filter;
pub mod inference_types;
pub mod inference_handlers;

// 重新导出常用类型和函数
pub use types::*;
pub use handlers::{AppState, health_check, list_models};
pub use inference_types::*;
pub use inference_handlers::{InferenceState, PipelineStatsResponse};
pub use server::{HttpConfig, start_http_server, wait_for_http_server_shutdown};

/// HTTP API 版本常量
pub const API_VERSION: &str = "v1";

/// 默认 HTTP 端口
pub const DEFAULT_HTTP_PORT: u16 = 8080;
