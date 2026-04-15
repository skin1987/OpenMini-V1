//! OpenMini Benchmark Framework
//!
//! 高性能 LLM 推理基准测试与 CI 性能回归检测框架
//!
//! ## 功能特性
//! - 多维度性能指标采集 (TTFT/TPOT/TBTL/Throughput/Memory)
//! - 预定义测试场景 (短/中/长/超长上下文、并发、突发流量)
//! - 多格式结果导出 (JSON/CSV/Prometheus)
//! - 性能回归自动检测与报告
//! - CI/CD 集成支持
//!
//! ## 使用示例
//!
//! ```ignore
//! use openmini_server::benchmark::{
//!     BenchmarkConfig, ModelBenchmark, Scenario, OutputFormat
//! };
//!
//! let config = BenchmarkConfig::default()
//!     .with_model_path("models/llama-7b.gguf")
//!     .with_scenario(Scenario::MediumContext);
//!
//! let benchmark = ModelBenchmark::new(config);
//! let results = benchmark.run().await?;
//! benchmark.export(&results, OutputFormat::Json)?;
//! ```

pub mod config;
pub mod export;
pub mod metrics;
pub mod runner;
pub mod scenarios;

pub mod ci {
    pub mod regression;
}

pub use ci::regression::*;
pub use config::*;
pub use export::*;
pub use metrics::*;
pub use runner::*;
pub use scenarios::*;
