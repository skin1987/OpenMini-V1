//! HTTP 中间件
//!
//! 提供请求日志、CORS、超时控制等中间件功能。

use axum::{
    http::{Request, Method},
    middleware::Next,
    response::Response,
    body::Body,
    Router,
};
use tower_http::{
    cors::{CorsLayer, Any},
    timeout::TimeoutLayer,
    limit::RequestBodyLimitLayer,
};
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// 请求日志中间件
///
/// 记录每个请求的方法、路径、状态码和处理时间。
pub async fn logging_middleware(
    req: Request<Body>,
    next: Next,
) -> Response {
    let start = Instant::now();
    let method = req.method().clone();
    let uri = req.uri().clone();
    let path = uri.path().to_string();

    // 获取客户端IP（从 X-Forwarded-For 或直接连接）
    let client_ip = req
        .headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown")
        .split(',')
        .next()
        .unwrap_or("unknown")
        .trim()
        .to_string();

    // 继续处理请求
    let response = next.run(req).await;

    // 计算耗时和记录日志
    let duration = start.elapsed();
    let status = response.status();

    match status.as_u16() {
        200..=299 => info!(
            method = %method,
            path = %path,
            status = status.as_u16(),
            duration_ms = duration.as_millis() as u64,
            client_ip = %client_ip,
            "Request completed"
        ),
        400..=499 => warn!(
            method = %method,
            path = %path,
            status = status.as_u16(),
            duration_ms = duration.as_millis() as u64,
            client_ip = %client_ip,
            "Client error"
        ),
        _ => warn!(
            method = %method,
            path = %path,
            status = status.as_u16(),
            duration_ms = duration.as_millis() as u64,
            client_ip = %client_ip,
            "Server error"
        ),
    }

    response
}

/// CORS 配置中间件
///
/// 支持可配置的跨域资源共享策略。
pub fn cors_layer(allowed_origins: Option<Vec<String>>) -> CorsLayer {
    match allowed_origins {
        Some(origins) if !origins.is_empty() => {
            // 自定义允许的源列表
            tower_http::cors::CorsLayer::new()
                .allow_origin(
                    origins.iter()
                        .map(|o| o.parse().unwrap())
                        .collect::<Vec<_>>(),
                )
                .allow_methods([
                    Method::GET,
                    Method::POST,
                    Method::PUT,
                    Method::DELETE,
                    Method::OPTIONS,
                ])
                .allow_headers(Any)
                .max_age(Duration::from_secs(86400))
        }
        _ => {
            // 允许所有来源（开发模式）
            tower_http::cors::CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any)
                .max_age(Duration::from_secs(86400))
        }
    }
}

/// 请求体大小限制中间件
///
/// 限制请求体的最大大小，防止大文件上传攻击。
pub fn request_body_limit_layer(max_size: usize) -> RequestBodyLimitLayer {
    RequestBodyLimitLayer::new(max_size)
}

/// 请求超时中间件
///
/// 设置请求处理的最大时间限制。
pub fn timeout_layer(duration: Duration) -> TimeoutLayer {
    TimeoutLayer::new(duration)
}

/// 创建完整的中间件栈
///
/// 组合所有中间件并应用到路由器。
///
/// # 参数
/// - `router`: axum 路由器
/// - `cors_origins`: 可选的 CORS 允许源列表（None 表示允许所有）
/// - `request_timeout_ms`: 请求超时时间（毫秒）
/// - `max_body_size`: 最大请求体大小（字节）
///
/// # 返回
/// 应用中间件后的路由器
pub fn apply_middlewares(
    router: Router,
    cors_origins: Option<Vec<String>>,
    request_timeout_ms: u64,
    max_body_size: usize,
) -> Router {
    router
        // 1. 请求日志（最外层，记录所有请求）
        .layer(axum::middleware::from_fn(logging_middleware))
        // 2. CORS 支持
        .layer(cors_layer(cors_origins))
        // 3. 请求体大小限制
        .layer(request_body_limit_layer(max_body_size))
        // 4. 请求超时
        .layer(timeout_layer(Duration::from_millis(request_timeout_ms)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cors_layer_with_origins() {
        let layer = cors_layer(Some(vec![
            "http://localhost:3000".to_string(),
            "https://example.com".to_string(),
        ]));
        let _ = layer;
        // 验证层创建成功（不panic即可）
    }

    #[test]
    fn test_cors_layer_allow_all() {
        let layer = cors_layer(None);
        let _ = layer;
    }

    #[test]
    fn test_request_body_limit_layer() {
        let layer = request_body_limit_layer(10 * 1024 * 1024); // 10MB
        let _ = layer;
    }

    #[test]
    fn test_timeout_layer() {
        let layer = timeout_layer(Duration::from_secs(30));
        let _ = layer;
    }
}
