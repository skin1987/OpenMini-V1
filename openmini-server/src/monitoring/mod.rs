//! OpenMini 监控系统
//!
//! 提供 Prometheus 指标导出、性能统计、健康检查等功能。
//!
//! ## 功能
//! - Prometheus metrics 导出 (HTTP /metrics 端点)
//! - 推理性能指标 (延迟、吞吐量、GPU利用率)
//! - 系统资源指标 (内存、CPU、GPU)
//! - 自定义业务指标
//!
//! ## 使用示例
//!
//! ```ignore
//! use openmini_server::monitoring::{PrometheusExporter, HealthChecker};
//!
//! // 启动 Prometheus 指标导出服务
//! let exporter = PrometheusExporter::new(9090);
//! tokio::spawn(async move {
//!     exporter.start().await.unwrap();
//! });
//!
//! // 执行健康检查
//! let checker = HealthChecker::new();
//! let status = checker.check().await;
//! ```

pub mod business_metrics;
pub mod health_check;
pub mod metrics;
pub mod prometheus_exporter;

// 重新导出常用类型
pub use health_check::HealthChecker;
