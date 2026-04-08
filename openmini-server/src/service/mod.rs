//! 服务模块入口
//!
//! 导出服务层相关的子模块和类型

pub mod server;
pub mod thread;
pub mod worker;

#[cfg(feature = "grpc")]
pub mod grpc {
    pub mod client;
    pub mod server;
    pub mod types;
}

/// HTTP REST API 服务模块
///
/// 提供基于 axum 的 HTTP REST API 实现，支持聊天、图像理解、语音等功能。
pub mod http {
    pub mod handlers;
    pub mod middleware;
    pub mod server;
    pub mod types;
}
