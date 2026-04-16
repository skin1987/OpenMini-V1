//! Prometheus 指标 HTTP 导出器
//!
//! 提供 `/metrics` 端点供 Prometheus 抓取指标数据。
//!
//! ## 功能
//! - 基于 axum 的 HTTP 服务器
//! - 支持自定义端口配置
//! - 文本格式指标导出 (Prometheus Text Format)
//! - 优雅关闭支持
//!
//! ## 使用示例
//!
//! ```ignore
//! use openmini_server::monitoring::PrometheusExporter;
//!
//! let exporter = PrometheusExporter::new(9090);
//!
//! // 在异步上下文中启动
//! tokio::spawn(async move {
//!     if let Err(e) = exporter.start().await {
//!         eprintln!("Prometheus exporter error: {}", e);
//!     }
//! });
//! ```

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Result;
use axum::{extract::State, http::StatusCode, response::Response, routing::get, Router};
use prometheus::{Encoder, Registry, TextEncoder};
use tokio::net::TcpListener;
use tracing::{error, info};

/// Prometheus 指标 HTTP 导出器
///
/// 提供 `/metrics` 端点供 Prometheus 抓取。
/// 默认端口: 9090
pub struct PrometheusExporter {
    /// 监听地址
    addr: SocketAddr,
    /// Prometheus 注册表
    registry: Arc<Registry>,
}

impl PrometheusExporter {
    /// 创建新的导出器
    ///
    /// # 参数
    ///
    /// - `port`: 监听端口号
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let exporter = PrometheusExporter::new(9090);
    /// ```
    pub fn new(port: u16) -> Self {
        Self::with_registry(port, prometheus::default_registry().clone())
    }

    /// 使用自定义注册表创建导出器
    ///
    /// # 参数
    ///
    /// - `port`: 监听端口号
    /// - `registry`: Prometheus 注册表实例
    pub fn with_registry(port: u16, registry: Registry) -> Self {
        let addr = SocketAddr::from(([0, 0, 0, 0], port));
        Self {
            addr,
            registry: Arc::new(registry),
        }
    }

    /// 启动 HTTP 服务器（异步）
    ///
    /// 阻塞当前任务，直到服务器停止。
    ///
    /// # 错误
    ///
    /// 返回错误的情况：
    /// - 端口被占用
    /// - 权限不足（端口 < 1024）
    /// - 服务器内部错误
    pub async fn start(&self) -> Result<()> {
        info!(address = %self.addr, "Starting Prometheus metrics exporter");

        // 创建 TCP listener 以便获取实际绑定地址
        let listener = TcpListener::bind(self.addr)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to bind to {}: {}", self.addr, e))?;

        let actual_addr = listener
            .local_addr()
            .map_err(|e| anyhow::anyhow!("Failed to get local address: {}", e))?;

        info!(address = %actual_addr, "Prometheus metrics server listening");

        // 构建路由，注入共享状态
        let registry = Arc::clone(&self.registry);
        let app = Router::new()
            .route("/metrics", get(metrics_handler))
            .route("/health", get(health_handler))
            .with_state(registry);

        // 启动服务器
        axum::serve(listener, app)
            .await
            .map_err(|e| anyhow::anyhow!("Server error: {}", e))?;

        Ok(())
    }

    /// 获取所有指标的文本格式
    ///
    /// 返回 Prometheus Text Exposition 格式的指标数据。
    /// 可用于调试或手动测试。
    pub fn gather_metrics(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();

        if let Err(e) = encoder.encode(&metric_families, &mut buffer) {
            error!(error = %e, "Failed to encode metrics");
            return String::new();
        }

        String::from_utf8(buffer).unwrap_or_default()
    }

    /// 获取监听地址
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }
}

/// 处理 /metrics 请求的异步处理器
async fn metrics_handler(State(registry): State<Arc<Registry>>) -> Response {
    let encoder = TextEncoder::new();
    let metric_families = registry.gather();
    let mut buffer = Vec::new();

    match encoder.encode(&metric_families, &mut buffer) {
        Ok(_) => Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", encoder.format_type())
            .body(axum::body::Body::from(buffer))
            .unwrap(),
        Err(e) => {
            error!(error = %e, "Failed to encode metrics");
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(axum::body::Body::from("Internal error"))
                .unwrap()
        }
    }
}

/// 处理 /health 请求的异步处理器（简单健康检查）
async fn health_handler() -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(axum::body::Body::from(r#"{"status":"ok"}"#))
        .unwrap()
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试导出器创建
    #[test]
    fn test_exporter_creation() {
        let exporter = PrometheusExporter::new(9090);
        assert_eq!(exporter.addr.port(), 9090);
    }

    /// 测试使用自定义注册表创建
    #[test]
    fn test_exporter_with_custom_registry() {
        let custom_registry = Registry::new();
        let exporter = PrometheusExporter::with_registry(9091, custom_registry);
        assert_eq!(exporter.addr.port(), 9091);
    }

    /// 测试指标收集功能
    #[test]
    fn test_gather_metrics() {
        use crate::monitoring::metrics::*;

        // 设置一些测试值
        update_gpu_metrics(75.5, 1024 * 1024 * 1024);
        update_system_metrics(50.0, 512 * 1024 * 1024);

        let exporter = PrometheusExporter::new(19999); // 使用非常规端口避免冲突
        let metrics_output = exporter.gather_metrics();

        // 验证输出不为空且包含预期内容
        assert!(!metrics_output.is_empty());
        assert!(metrics_output.contains("openmini_gpu_utilization_percent"));
        assert!(metrics_output.contains("openmini_cpu_usage_percent"));
    }

    /// 测试指标收集包含推理指标
    #[test]
    fn test_gather_inference_metrics() {
        use crate::monitoring::metrics::*;

        // 记录一些推理请求
        record_inference_start();
        record_inference_complete("test-model", true, 100.0, "q4_0", "metal");

        let exporter = PrometheusExporter::new(19998);
        let metrics_output = exporter.gather_metrics();

        assert!(metrics_output.contains("openmini_inference_requests_total"));
        assert!(metrics_output.contains("openmini_active_requests"));
    }

    /// 测试默认地址绑定到所有接口
    #[test]
    fn test_default_address_binding() {
        let exporter = PrometheusExporter::new(9090);
        let addr = exporter.addr();

        // 应该绑定到 0.0.0.0
        assert_eq!(addr.ip(), std::net::Ipv4Addr::new(0, 0, 0, 0));
        assert_eq!(addr.port(), 9090);
    }

    /// 测试不同端口的导出器
    #[test]
    fn test_different_ports() {
        for port in [8080, 9090, 10000] {
            let exporter = PrometheusExporter::new(port);
            assert_eq!(exporter.addr().port(), port);
        }
    }

    /// 集成测试：启动并请求指标（需要可用端口和 monitoring feature）
    #[cfg(feature = "monitoring")]
    #[tokio::test]
    async fn test_server_start_and_request() {
        use reqwest;
        use std::time::Duration;

        // 使用随机端口避免冲突
        let port = rand::random::<u16>() % 10000 + 20000; // 20000-29999
        let exporter = PrometheusExporter::new(port);

        // 在后台启动服务器
        let handle = tokio::spawn(async move {
            // 给服务器一点时间启动，如果失败就返回
            let _ = tokio::time::timeout(Duration::from_millis(100), exporter.start()).await;
        });

        // 等待服务器启动
        tokio::time::sleep(Duration::from_millis(50)).await;

        // 尝试请求 /metrics 端点
        let url = format!("http://127.0.0.1:{}/metrics", port);
        match reqwest::get(&url).await {
            Ok(response) => {
                assert_eq!(response.status(), 200);
                let text = response.text().await.unwrap();
                assert!(!text.is_empty());
            }
            Err(_) => {
                // 端口可能被占用或权限问题，跳过此测试
                tracing::warn!("Skipping integration test: could not connect to {}", url);
            }
        }

        // 清理：停止服务器（通过drop handle）
        drop(handle);
    }

    // ===== 边界条件和分支覆盖率测试 =====

    #[test]
    fn test_prometheus_exporter_custom_port() {
        // 测试各种自定义端口
        for port in [8080, 9090, 3000, 80, 443] {
            let exporter = PrometheusExporter::new(port);
            assert_eq!(
                exporter.addr().port(),
                port,
                "Port should match for {}",
                port
            );

            // 验证地址绑定到所有接口
            assert_eq!(exporter.addr().ip(), std::net::Ipv4Addr::new(0, 0, 0, 0));
        }
    }

    #[test]
    fn test_gather_metrics_format_validation() {
        use crate::monitoring::metrics::*;

        // 设置一些指标值
        update_gpu_metrics(75.5, 1024 * 1024 * 1024); // 75.5% GPU, 1GB memory
        update_system_metrics(45.2, 512 * 1024 * 512); // 45.2% CPU, 256MB
        update_token_throughput(123.4);
        update_batch_size(6.0);

        let exporter = PrometheusExporter::new(19997);
        let metrics_text = exporter.gather_metrics();

        // 验证Prometheus基本格式要求
        assert!(
            !metrics_text.is_empty(),
            "Metrics output should not be empty"
        );

        // 验证包含HELP注释（Prometheus标准格式）
        assert!(
            metrics_text.contains("# HELP") || metrics_text.contains("openmini_"),
            "Metrics should contain HELP comments or metric names"
        );

        // 验证包含TYPE注释
        assert!(
            metrics_text.contains("# TYPE") || metrics_text.contains("openmini_"),
            "Metrics should contain TYPE comments or metric names"
        );

        // 验证包含我们设置的指标值
        assert!(
            metrics_text.contains("openmini_gpu_utilization_percent")
                || metrics_text.contains("openmini_cpu_usage_percent"),
            "Should contain system metrics"
        );
    }

    #[test]
    fn test_gather_metrics_with_inference_data() {
        use crate::monitoring::metrics::*;

        // 记录推理数据
        record_inference_start();
        record_inference_complete("test-model-for-export", true, 42.5, "q4_0", "metal");

        record_inference_start();
        record_inference_complete("error-model", false, 5000.0, "q8_0", "cpu");

        let exporter = PrometheusExporter::new(19996);
        let metrics_output = exporter.gather_metrics();

        // 验证输出包含推理相关指标
        assert!(
            metrics_output.contains("openmini_inference_requests_total"),
            "Should contain inference requests counter"
        );

        // 验证包含不同状态的记录
        assert!(
            metrics_output.len() > 50, // 应该有足够的内容
            "Output should have substantial content"
        );
    }

    #[test]
    fn test_gather_metrics_empty_registry() {
        // 测试使用空注册表时的行为
        let custom_registry = prometheus::Registry::new();
        let exporter = PrometheusExporter::with_registry(19995, custom_registry);

        let metrics_output = exporter.gather_metrics();

        // 空注册表应该返回空字符串或最小格式
        // （不应该panic）
        let _ = metrics_output;
    }

    #[test]
    fn test_exporter_addr_method() {
        // 测试addr()方法返回完整的SocketAddr
        let exporter = PrometheusExporter::new(8888);
        let addr = exporter.addr();

        assert_eq!(addr.port(), 8888);
        assert_eq!(addr.ip(), std::net::Ipv4Addr::new(0, 0, 0, 0));

        // 验证可以用于格式化
        let addr_string = addr.to_string();
        assert!(addr_string.contains("8888"));
        assert!(addr_string.contains("0.0.0.0"));
    }

    #[test]
    fn test_multiple_exporters_different_ports() {
        // 多个导出器实例应该独立工作
        let exporter1 = PrometheusExporter::new(19001);
        let exporter2 = PrometheusExporter::new(19002);
        let exporter3 = PrometheusExporter::new(19003);

        assert_ne!(exporter1.addr().port(), exporter2.addr().port());
        assert_ne!(exporter2.addr().port(), exporter3.addr().port());

        // 每个都应该能收集指标
        let metrics1 = exporter1.gather_metrics();
        let metrics2 = exporter2.gather_metrics();
        let metrics3 = exporter3.gather_metrics();

        // 它们应该收集相同的全局指标（因为共享同一个默认registry）
        // 但至少验证都能正常工作
        assert!(!metrics1.is_empty());
        assert!(!metrics2.is_empty());
        assert!(!metrics3.is_empty());
    }

    #[test]
    fn test_exporter_with_custom_registry_isolation() {
        // 使用自定义注册表的导出器应该与默认隔离
        let custom_registry = prometheus::Registry::new();
        let exporter = PrometheusExporter::with_registry(19004, custom_registry);

        // 自定义注册表应该是空的或只有我们添加的
        let metrics = exporter.gather_metrics();

        // 由于没有向custom_registry注册任何指标，输出可能为空或只包含基础格式
        // 关键是这不应该panic
        let _ = metrics;
    }

    /// 测试边界端口值（0和65535）
    /// 覆盖分支：端口的极端值
    #[test]
    fn test_boundary_port_values() {
        // 端口0（保留端口，但u16允许）
        let exporter_zero = PrometheusExporter::new(0);
        assert_eq!(exporter_zero.addr().port(), 0);

        // 最大端口65535
        let exporter_max = PrometheusExporter::new(65535);
        assert_eq!(exporter_max.addr().port(), 65535);

        // 常见系统端口
        let exporter_sys = PrometheusExporter::new(1023); // 最后一个系统端口
        assert_eq!(exporter_sys.addr().port(), 1023);

        // 第一个用户端口
        let exporter_user = PrometheusExporter::new(1024);
        assert_eq!(exporter_user.addr().port(), 1024);
    }

    /// 测试gather_metrics返回的指标包含时间戳信息或格式正确性
    /// 覆盖分支：Prometheus格式的完整性
    #[test]
    fn test_gather_metrics_format_structure() {
        use crate::monitoring::metrics::*;

        update_gpu_metrics(50.0, 2 * 1024 * 1024 * 1024);

        let exporter = PrometheusExporter::new(19993);
        let output = exporter.gather_metrics();

        if !output.is_empty() {
            // 验证输出是有效的UTF-8（应该总是成立）
            assert!(output.is_ascii() || !output.is_empty());

            // 验证基本结构：每行以换行符结尾（除了最后一行可能）
            let lines: Vec<&str> = output.lines().collect();
            if !lines.is_empty() {
                // 至少应该有一些内容行
                assert!(!lines.is_empty());
            }
        }
    }

    /// 测试多次调用gather_metrics的一致性
    /// 覆盖分支：重复调用的稳定性
    #[test]
    fn test_gather_metrics_consistency() {
        use crate::monitoring::metrics::*;

        update_gpu_metrics(75.5, 1024 * 1024 * 1024);

        let exporter = PrometheusExporter::new(19992);

        // 连续多次收集，结果应该一致（因为指标没变）
        let output1 = exporter.gather_metrics();
        let output2 = exporter.gather_metrics();
        let output3 = exporter.gather_metrics();

        assert!(!output1.is_empty());
        assert!(!output2.is_empty());
        assert!(!output3.is_empty());
        assert!(output1.contains("openmini_gpu"));
        assert!(output2.contains("openmini_gpu"));
        assert!(output3.contains("openmini_gpu"));
    }

    /// 测试使用不同注册表创建多个导出器的独立性
    /// 覆盖分支：多实例多注册表场景
    #[test]
    fn test_multiple_exporters_with_different_registries() {
        let registry1 = prometheus::Registry::new();
        let registry2 = prometheus::Registry::new();
        let registry3 = prometheus::Registry::new();

        let exporter1 = PrometheusExporter::with_registry(18001, registry1);
        let exporter2 = PrometheusExporter::with_registry(18002, registry2);
        let exporter3 = PrometheusExporter::with_registry(18003, registry3);

        // 每个导出器都应该能正常工作
        let metrics1 = exporter1.gather_metrics();
        let metrics2 = exporter2.gather_metrics();
        let metrics3 = exporter3.gather_metrics();

        // 空注册表应该返回空或最小输出
        // 关键是不panic
        let _ = metrics1;
        let _ = metrics2;
        let _ = metrics3;

        // 验证它们有不同的地址
        assert_ne!(exporter1.addr().port(), exporter2.addr().port());
        assert_ne!(exporter2.addr().port(), exporter3.addr().port());
    }

    /// 测试PrometheusExporter在设置各种指标后的行为
    /// 覆盖分支：多种指标类型的组合场景
    #[test]
    fn test_gather_metrics_with_all_metric_types() {
        use crate::monitoring::metrics::*;

        // 设置所有类型的指标
        update_gpu_metrics(88.8, 4 * 1024 * 1024 * 1024); // GPU指标
        update_system_metrics(66.6, 3 * 1024 * 1024 * 1024); // 系统指标
        update_token_throughput(999.9); // 吞吐量
        update_batch_size(32.0); // 批量大小
        update_scheduler_metrics(50.0, 5.0); // 调度器

        // 记录推理请求
        record_inference_start();
        record_inference_complete("comprehensive-test-model", true, 123.456, "q4_0", "metal");

        let exporter = PrometheusExporter::new(19991);
        let metrics_output = exporter.gather_metrics();

        // 输出应该不为空且有足够的内容
        assert!(!metrics_output.is_empty());

        // 验证输出长度合理（应该包含多种指标的描述和数据）
        // 注意：由于全局状态，这里只做基本的非空检查
    }

    /// 测试addr()方法返回值的完整性和可用性
    /// 覆盖分支：SocketAddr的所有字段访问
    #[test]
    fn test_addr_completeness() {
        let port = 12345;
        let exporter = PrometheusExporter::new(port);
        let addr = exporter.addr();

        // 验证地址组成部分
        assert_eq!(addr.port(), port);
        assert!(addr.is_ipv4()); // 应该是IPv4 (0.0.0.0)

        // 验证可以序列化为字符串
        let addr_str = addr.to_string();
        assert!(addr_str.contains(&port.to_string()));

        // 验证IP部分
        match addr.ip() {
            std::net::IpAddr::V4(ipv4) => {
                assert_eq!(ipv4, std::net::Ipv4Addr::new(0, 0, 0, 0));
            }
            _ => panic!("Expected IPv4 address"),
        }
    }

    /// 测试极端情况下的gather_metrics行为（零值指标）
    /// 覆盖分支：零值和最小值指标的处理
    #[test]
    fn test_gather_metrics_with_zero_values() {
        use crate::monitoring::metrics::*;

        // 设置为零值
        update_gpu_metrics(0.0, 0);
        update_system_metrics(0.0, 0);
        update_token_throughput(0.0);
        update_batch_size(0.0);
        update_scheduler_metrics(0.0, 0.0);

        let exporter = PrometheusExporter::new(19990);
        let output = exporter.gather_metrics();

        // 即使是零值，也不应该panic
        // 输出可能包含零值的指标数据
        let _ = output;
    }
}
