//! HTTP 服务器启动和管理
//!
//! 提供 HTTP REST API 服务器的启动、配置和优雅关闭功能。

use std::net::SocketAddr;
use std::time::Duration;

use axum::Json;
use tokio::signal;
use tracing::{info, warn, error, debug};

use super::handlers::AppState;
use super::middleware;

use crate::config::settings::ServerConfig;

/// HTTP 服务器配置
#[derive(Debug, Clone)]
pub struct HttpConfig {
    /// 监听地址
    pub host: String,
    /// 监听端口
    pub port: u16,
    /// 请求超时时间（毫秒）
    pub request_timeout_ms: u64,
    /// 最大请求体大小（字节）
    pub max_body_size: usize,
    /// CORS 允许的源列表（None 表示允许所有）
    pub cors_allowed_origins: Option<Vec<String>>,
    /// 是否启用 metrics 端点
    pub enable_metrics: bool,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            request_timeout_ms: 60000,
            max_body_size: 10 * 1024 * 1024, // 10MB
            cors_allowed_origins: None,       // 开发模式：允许所有
            enable_metrics: true,
        }
    }
}

impl From<&ServerConfig> for HttpConfig {
    fn from(config: &ServerConfig) -> Self {
        Self {
            host: config.server.host.clone(),
            port: config.server.port + 1000, // 默认在 gRPC 端口 + 1000
            request_timeout_ms: config.server.request_timeout_ms,
            max_body_size: 10 * 1024 * 1024,
            cors_allowed_origins: None,
            enable_metrics: true,
        }
    }
}

/// 启动 HTTP REST API 服务器
///
/// 使用 axum 框架启动 OpenMini HTTP 服务，支持：
/// - 完整的 REST API 端点（聊天、图像理解、TTS/STT 等）
/// - Server-Sent Events (SSE) 流式输出
/// - Prometheus 监控指标导出
/// - 优雅关闭（graceful shutdown）处理 SIGTERM/SIGINT 信号
/// - 可配置的 CORS、超时、请求体大小限制
///
/// # 参数
/// - `addr`: 监听地址，格式为 "host:port"（如 "0.0.0.0:8080"）
/// - `config`: 可选的服务器配置，用于自定义服务行为
///
/// # 返回
/// - `Ok(JoinHandle<()>)`: 返回服务器的任务句柄，可用于等待服务结束
/// - `Err`: 地址解析失败或服务器启动失败
///
/// # 示例
///
/// ```ignore
/// let handle = start_http_server("0.0.0.0:8080", None).await?;
/// handle.await?; // 等待服务器结束
/// ```
pub async fn start_http_server(
    addr: &str,
    config: Option<&HttpConfig>,
) -> Result<tokio::task::JoinHandle<()>, Box<dyn std::error::Error + Send + Sync>> {
    info!(address = %addr, "正在启动 HTTP REST API 服务器");

    // 解析地址
    let socket_addr: SocketAddr = addr
        .parse()
        .map_err(|e| format!("无效的地址格式 '{}': {}", addr, e))?;

    // 从配置中提取设置
    let http_config = config.cloned().unwrap_or_default();

    debug!(
        host = %http_config.host,
        port = http_config.port,
        timeout_ms = http_config.request_timeout_ms,
        max_body_size_mb = http_config.max_body_size / (1024 * 1024),
        enable_metrics = http_config.enable_metrics,
        "HTTP 服务器配置"
    );

    // 创建应用状态
    let state = AppState::new();

    // 构建路由
    let router = build_routes(state, http_config.enable_metrics);

    // 应用中间件
    let router_with_middleware = middleware::apply_middlewares(
        router,
        http_config.cors_allowed_origins.clone(),
        http_config.request_timeout_ms,
        http_config.max_body_size,
    );

    // 创建 shutdown 信号通道
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::watch::channel(());

    // 注册信号处理器（SIGTERM, SIGINT）
    let signal_shutdown_tx = shutdown_tx.clone();
    tokio::spawn(async move {
        #[cfg(unix)]
        {
            if let Ok(mut sigterm) = signal::unix::signal(signal::unix::SignalKind::terminate()) {
                info!("注册 SIGTERM 信号处理器");
                sigterm.recv().await;
                info!("收到 SIGTERM 信号，开始优雅关闭...");
            } else {
                warn!("无法注册 SIGTERM 处理器");
            }
        }

        #[cfg(not(unix))]
        {
            info!("等待 Ctrl+C 信号...");
        }

        match signal::ctrl_c().await {
            Ok(()) => {
                info!("收到 SIGINT/Ctrl-C 信号，开始优雅关闭...");
            }
            Err(e) => {
                error!("监听 Ctrl-C 失败: {}", e);
            }
        }

        if signal_shutdown_tx.send(()).is_err() {
            warn!("关闭信号发送失败（接收端已关闭）");
        }
    });

    // 启动 HTTP 服务器任务
    let server_handle = tokio::spawn(async move {
        info!(
            address = %socket_addr,
            "HTTP 服务器正在绑定端口..."
        );

        let listener = match tokio::net::TcpListener::bind(socket_addr).await {
            Ok(listener) => listener,
            Err(e) => {
                error!(error = %e.to_string(), address = %socket_addr, "HTTP 服务器绑定失败");
                return;
            }
        };

        info!(
            address = %socket_addr,
            "HTTP 服务器绑定成功"
        );

        // 启动 axum 服务器并监听 shutdown 信号
        let server_result = axum::serve(listener, router_with_middleware)
            .with_graceful_shutdown(async move {
                shutdown_rx.changed().await.ok();
                info!("收到关闭信号，停止接受新连接...");

                // 给予现有连接一些时间完成处理
                tokio::time::sleep(Duration::from_secs(5)).await;
                info!("开始强制关闭剩余连接...");
            })
            .await;

        match server_result {
            Ok(()) => {
                info!("HTTP 服务器正常关闭");
            }
            Err(e) => {
                error!(error = %e.to_string(), "HTTP 服务器错误退出");
            }
        }
    });

    // 等待一小段时间确保服务器已启动
    tokio::time::sleep(Duration::from_millis(100)).await;

    info!(
        address = %addr,
        "HTTP REST API 服务器启动成功（PID: {}）",
        std::process::id()
    );

    Ok(server_handle)
}

/// 构建 API 路由
///
/// 注册所有 REST API 端点到路由器。
fn build_routes(state: AppState, enable_metrics: bool) -> axum::Router {
    use super::handlers::*;

    let mut api_routes = axum::Router::new()
        // 聊天相关
        .route("/chat", axum::routing::post(chat_completion))
        .route("/chat/stream", axum::routing::post(chat_completion_stream))
        // 图像理解
        .route("/image/understand", axum::routing::post(image_understand))
        // 语音相关
        .route("/tts", axum::routing::post(text_to_speech))
        .route("/stt", axum::routing::post(speech_to_text))
        // 健康检查和监控
        .route("/health", axum::routing::get(health_check))
        .route("/models", axum::routing::get(list_models));

    // 可选：启用 Prometheus metrics 端点
    if enable_metrics {
        api_routes = api_routes.route("/metrics", axum::routing::get(metrics));
    }

    // 将所有路由挂载到 /api/v1 前缀下
    let app = axum::Router::new()
        .nest("/api/v1", api_routes)
        .with_state(state);

    // 根路径返回欢迎信息
    let root_route = axum::Router::new().route("/", axum::routing::get(|| async {
        (
            axum::http::StatusCode::OK,
            Json(serde_json::json!({
                "name": "OpenMini Server",
                "version": env!("CARGO_PKG_VERSION"),
                "docs": "/api/v1",
                "status": "running"
            }))
        )
    }));

    app.merge(root_route)
}

/// 等待 HTTP 服务器优雅关闭
///
/// 阻塞当前线程直到服务器完成所有请求处理并关闭。
///
/// # 参数
/// - `server_handle`: 由 start_http_server 返回的任务句柄
///
/// # 返回
/// - `Ok(())`: 服务器正常关闭
/// - `Err`: 服务器异常退出
pub async fn wait_for_http_server_shutdown(
    server_handle: tokio::task::JoinHandle<()>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!("等待 HTTP 服务器关闭...");

    match server_handle.await {
        Ok(result) => {
            info!("HTTP 服务器任务完成");
            Ok(result)
        }
        Err(e) => {
            if e.is_cancelled() {
                warn!("HTTP 服务器任务被取消");
                Ok(())
            } else {
                error!(error = %e.to_string(), "HTTP 服务器任务 panic");
                Err(format!("服务器任务异常: {}", e).into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_start_http_server_success() {
        // 使用随机端口避免冲突
        let result = start_http_server("127.0.0.1:0", None).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_http_config_default() {
        let config = HttpConfig::default();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 8080);
        assert_eq!(config.request_timeout_ms, 60000);
        assert_eq!(config.max_body_size, 10 * 1024 * 1024);
        assert!(config.cors_allowed_origins.is_none());
        assert!(config.enable_metrics);
    }

    #[test]
    fn test_http_config_from_server_config() {
        let server_config = ServerConfig::default();
        let http_config = HttpConfig::from(&server_config);
        assert_eq!(http_config.host, server_config.server.host);
        assert_eq!(http_config.port, server_config.port + 1000);
        assert_eq!(http_config.request_timeout_ms, server_config.server.request_timeout_ms);
    }

    #[tokio::test]
    async fn test_start_http_server_invalid_address() {
        let result = start_http_server("invalid-address", None).await;
        assert!(result.is_err());
        let err_msg = format!("{}", result.err().unwrap());
        assert!(err_msg.contains("无效的地址格式"));
    }
}
