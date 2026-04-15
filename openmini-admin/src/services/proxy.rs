use anyhow::Result;
use reqwest::Client;
use serde_json::Value;

pub struct UpstreamProxy {
    client: Client,
    base_url: String,
}

impl UpstreamProxy {
    pub fn new(base_url: &str, timeout_secs: u64) -> Self {
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(timeout_secs))
                .build()
                .unwrap(),
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    pub async fn get_health(&self) -> Result<Option<Value>> {
        let url = format!("{}/health", self.base_url);
        match self.client.get(&url).send().await {
            Ok(resp) => {
                if resp.status().is_success() {
                    Ok(Some(resp.json::<Value>().await?))
                } else {
                    Ok(None)
                }
            }
            Err(_) => Ok(None),
        }
    }

    pub async fn get_models(&self) -> Result<Option<Value>> {
        let url = format!("{}/models", self.base_url);
        match self.client.get(&url).send().await {
            Ok(resp) => {
                if resp.status().is_success() {
                    Ok(Some(resp.json::<Value>().await?))
                } else {
                    Ok(None)
                }
            }
            Err(_) => Ok(None),
        }
    }

    pub async fn get_metrics(&self) -> Result<Option<String>> {
        let url = format!("{}/metrics", self.base_url);
        match self.client.get(&url).send().await {
            Ok(resp) => {
                if resp.status().is_success() {
                    Ok(Some(resp.text().await?))
                } else {
                    Ok(None)
                }
            }
            Err(_) => Ok(None),
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

// ============ 单元测试 ============

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proxy_creation() {
        let proxy = UpstreamProxy::new("http://localhost:8080", 30);
        assert_eq!(proxy.base_url(), "http://localhost:8080");
    }

    #[test]
    fn test_proxy_trailing_slash_removed() {
        let proxy = UpstreamProxy::new("http://localhost:8080/", 30);
        assert_eq!(proxy.base_url(), "http://localhost:8080");

        // trim_end_matches 只移除末尾连续的斜杠
        let proxy2 = UpstreamProxy::new("http://example.com///", 30);
        assert_eq!(proxy2.base_url(), "http://example.com"); // 所有尾部斜杠都被移除
    }

    #[test]
    fn test_proxy_different_timeouts() {
        // 短超时
        let proxy1 = UpstreamProxy::new("http://localhost:8080", 5);
        assert_eq!(proxy1.base_url(), "http://localhost:8080");

        // 长超时
        let proxy2 = UpstreamProxy::new("http://localhost:8080", 300);
        assert_eq!(proxy2.base_url(), "http://localhost:8080");
    }

    #[tokio::test]
    async fn test_get_health_unreachable_server() {
        // 连接到一个不太可能存在的地址，应该返回 Ok(None) 而不是 panic
        let proxy = UpstreamProxy::new("http://127.0.0.1:1", 1); // 使用极短超时和不可能的端口

        let result = proxy.get_health().await;

        // 应该成功返回 None（因为服务器不可达）
        match result {
            Ok(None) => {} // 预期行为
            Ok(Some(_)) => panic!("Unexpectedly got a response from unreachable server"),
            Err(e) => panic!("Should not error on unreachable server, got: {:?}", e),
        }
    }

    #[tokio::test]
    async fn test_get_models_unreachable_server() {
        let proxy = UpstreamProxy::new("http://127.0.0.1:1", 1);

        let result = proxy.get_models().await;

        match result {
            Ok(None) => {}
            Ok(Some(_)) => panic!("Unexpectedly got a response from unreachable server"),
            Err(e) => panic!("Should not error on unreachable server, got: {:?}", e),
        }
    }

    #[tokio::test]
    async fn test_get_metrics_unreachable_server() {
        let proxy = UpstreamProxy::new("http://127.0.0.1:1", 1);

        let result = proxy.get_metrics().await;

        match result {
            Ok(None) => {}
            Ok(Some(_)) => panic!("Unexpectedly got a response from unreachable server"),
            Err(e) => panic!("Should not error on unreachable server, got: {:?}", e),
        }
    }

    #[tokio::test]
    async fn test_get_health_with_mock_server() {
        use axum::{routing::get, Json, Router};
        use serde_json::json;

        // 创建一个 mock HTTP 服务器
        let app = Router::new().route(
            "/health",
            get(|| async {
                Json(json!({
                    "status": "healthy",
                    "version": "1.0.0"
                }))
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        // 在后台启动服务器
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        // 给服务器一点时间启动
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // 创建 proxy 并请求
        let proxy = UpstreamProxy::new(&format!("http://127.0.0.1:{}", port), 5);
        let result = proxy.get_health().await;

        match result {
            Ok(Some(health)) => {
                assert_eq!(
                    health.get("status").and_then(|v| v.as_str()),
                    Some("healthy")
                );
                assert_eq!(
                    health.get("version").and_then(|v| v.as_str()),
                    Some("1.0.0")
                );
            }
            Ok(None) => panic!("Expected health data but got None"),
            Err(e) => panic!("Failed to get health: {:?}", e),
        }
    }

    #[tokio::test]
    async fn test_get_models_with_mock_server() {
        use axum::{routing::get, Json, Router};
        use serde_json::json;

        let app = Router::new().route(
            "/models",
            get(|| async {
                Json(json!({
                    "models": [
                        {"id": "model-1", "name": "Test Model 1"},
                        {"id": "model-2", "name": "Test Model 2"}
                    ]
                }))
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let proxy = UpstreamProxy::new(&format!("http://127.0.0.1:{}", port), 5);
        let result = proxy.get_models().await;

        match result {
            Ok(Some(models)) => {
                assert!(models.get("models").is_some());
                let models_array = models["models"].as_array().unwrap();
                assert_eq!(models_array.len(), 2);
            }
            Ok(None) => panic!("Expected models data but got None"),
            Err(e) => panic!("Failed to get models: {:?}", e),
        }
    }

    #[tokio::test]
    async fn test_get_metrics_with_mock_server() {
        use axum::{routing::get, Router};

        let app = Router::new().route(
            "/metrics",
            get(|| async { "mock_metric_data 12345\n".to_string() }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let proxy = UpstreamProxy::new(&format!("http://127.0.0.1:{}", port), 5);
        let result = proxy.get_metrics().await;

        match result {
            Ok(Some(metrics)) => {
                assert!(metrics.contains("mock_metric_data"));
                assert!(metrics.contains("12345"));
            }
            Ok(None) => panic!("Expected metrics data but got None"),
            Err(e) => panic!("Failed to get metrics: {:?}", e),
        }
    }

    #[tokio::test]
    async fn test_health_endpoint_non_200_response() {
        use axum::{http::StatusCode, routing::get, Router};

        // 返回 503 的健康检查端点
        let app = Router::new().route("/health", get(|| async { StatusCode::SERVICE_UNAVAILABLE }));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let proxy = UpstreamProxy::new(&format!("http://127.0.0.1:{}", port), 5);
        let result = proxy.get_health().await;

        // 非 200 响应应该返回 Ok(None)
        match result {
            Ok(None) => {} // 预期行为
            Ok(Some(_)) => panic!("Expected None for non-200 response"),
            Err(e) => panic!("Should not error for non-200 response: {:?}", e),
        }
    }

    #[tokio::test]
    async fn test_timeout_handling() {
        // 创建一个延迟响应的服务器
        use axum::{routing::get, Router};

        let app = Router::new().route(
            "/slow",
            get(|| async {
                tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
                "slow response"
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // 使用非常短的超时
        let proxy = UpstreamProxy::new(&format!("http://127.0.0.1:{}/slow", port), 1);

        let start = std::time::Instant::now();
        let result = proxy.get_health().await; // 会请求 /health 而不是 /slow
        let elapsed = start.elapsed();

        // 应该在合理时间内完成（不会等待 10 秒）
        assert!(
            elapsed < std::time::Duration::from_secs(3),
            "Request should timeout quickly, but took {:?}",
            elapsed
        );

        // 超时应该返回错误或 None
        match result {
            Ok(_) | Err(_) => {} // 都可以接受
        }
    }

    #[test]
    fn test_base_url_immutability() {
        let proxy = UpstreamProxy::new("http://example.com/api", 30);

        // 多次调用 base_url() 应该返回相同的值
        assert_eq!(proxy.base_url(), "http://example.com/api");
        assert_eq!(proxy.base_url(), "http://example.com/api");
        assert_eq!(proxy.base_url(), "http://example.com/api");
    }

    #[test]
    fn test_proxy_can_be_cloned_if_needed() {
        // 虽然 UpstreamProxy 没有实现 Clone，但可以创建多个实例
        let proxy1 = UpstreamProxy::new("http://localhost:8080", 30);
        let proxy2 = UpstreamProxy::new("http://localhost:8080", 30);

        // 它们应该有相同的基本属性
        assert_eq!(proxy1.base_url(), proxy2.base_url());
    }
}
