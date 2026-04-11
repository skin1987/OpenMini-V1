//! Benchmark Configuration Module
//!
//! 提供基准测试的完整配置，包括模型参数、硬件设置、推理参数等

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[derive(Default)]
pub enum DeviceType {
    #[default]
    CPU,
    GPU,
    CUDA,
    Metal,
}


impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceType::CPU => write!(f, "cpu"),
            DeviceType::GPU => write!(f, "gpu"),
            DeviceType::CUDA => write!(f, "cuda"),
            DeviceType::Metal => write!(f, "metal"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[derive(Default)]
pub enum OutputFormat {
    #[default]
    Json,
    Csv,
    Prometheus,
    Human,
}


impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputFormat::Json => write!(f, "json"),
            OutputFormat::Csv => write!(f, "csv"),
            OutputFormat::Prometheus => write!(f, "prometheus"),
            OutputFormat::Human => write!(f, "human"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub memory_total_gb: f64,
    pub gpu_model: Option<String>,
    pub gpu_memory_gb: Option<f64>,
    pub os_info: String,
}

impl Default for HardwareInfo {
    fn default() -> Self {
        HardwareInfo {
            cpu_model: "Unknown".to_string(),
            cpu_cores: num_cpus::get(),
            memory_total_gb: 0.0,
            gpu_model: None,
            gpu_memory_gb: None,
            os_info: std::env::consts::OS.to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub model_path: PathBuf,
    pub model_name: String,
    pub quantization: String,

    pub device: DeviceType,
    pub num_threads: usize,
    pub gpu_memory_fraction: f32,

    pub batch_sizes: Vec<usize>,
    pub sequence_lengths: Vec<usize>,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,

    pub num_warmup_runs: usize,
    pub num_benchmark_runs: usize,
    pub timeout_seconds: u64,

    pub output_format: OutputFormat,
    pub output_path: Option<PathBuf>,
    pub baseline_path: Option<PathBuf>,

    pub commit_hash: String,
    pub build_timestamp: String,
    pub hardware_info: HardwareInfo,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        BenchmarkConfig {
            model_path: PathBuf::from("models/model.gguf"),
            model_name: "default-model".to_string(),
            quantization: "Q4_K_M".to_string(),

            device: DeviceType::default(),
            num_threads: num_cpus::get(),
            gpu_memory_fraction: 0.9,

            batch_sizes: vec![1],
            sequence_lengths: vec![512, 1024, 2048, 4096, 8192, 16384, 32768],
            max_new_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,

            num_warmup_runs: 2,
            num_benchmark_runs: 5,
            timeout_seconds: 300,

            output_format: OutputFormat::Json,
            output_path: None,
            baseline_path: None,

            commit_hash: option_env!("GIT_COMMIT_HASH").unwrap_or("unknown").to_string(),
            build_timestamp: chrono::Utc::now().to_rfc3339(),
            hardware_info: HardwareInfo::default(),
        }
    }
}

impl BenchmarkConfig {
    pub fn new(model_path: impl Into<PathBuf>, model_name: impl Into<String>) -> Self {
        BenchmarkConfig {
            model_path: model_path.into(),
            model_name: model_name.into(),
            ..Default::default()
        }
    }

    pub fn with_device(mut self, device: DeviceType) -> Self {
        self.device = device;
        self
    }

    pub fn with_quantization(mut self, quantization: impl Into<String>) -> Self {
        self.quantization = quantization.into();
        self
    }

    pub fn with_batch_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.batch_sizes = sizes;
        self
    }

    pub fn with_sequence_lengths(mut self, lengths: Vec<usize>) -> Self {
        self.sequence_lengths = lengths;
        self
    }

    pub fn with_max_new_tokens(mut self, tokens: usize) -> Self {
        self.max_new_tokens = tokens;
        self
    }

    pub fn with_num_runs(mut self, warmup: usize, benchmark: usize) -> Self {
        self.num_warmup_runs = warmup;
        self.num_benchmark_runs = benchmark;
        self
    }

    pub fn with_output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = format;
        self
    }

    pub fn with_output_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.output_path = Some(path.into());
        self
    }

    pub fn with_baseline_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.baseline_path = Some(path.into());
        self
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.model_path.exists() {
            return Err(format!("Model path does not exist: {}", self.model_path.display()));
        }

        if self.batch_sizes.is_empty() {
            return Err("Batch sizes cannot be empty".to_string());
        }

        if self.sequence_lengths.is_empty() {
            return Err("Sequence lengths cannot be empty".to_string());
        }

        if self.num_benchmark_runs == 0 {
            return Err("Number of benchmark runs must be > 0".to_string());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.quantization, "Q4_K_M");
        assert_eq!(config.device, DeviceType::CPU);
        assert_eq!(config.sequence_lengths.len(), 7);
        assert_eq!(config.num_warmup_runs, 2);
        assert_eq!(config.num_benchmark_runs, 5);
    }

    #[test]
    fn test_config_builder() {
        let config = BenchmarkConfig::new("test/model.gguf", "test-model")
            .with_device(DeviceType::CUDA)
            .with_quantization("FP16")
            .with_batch_sizes(vec![1, 2, 4])
            .with_num_runs(3, 10);

        assert_eq!(config.model_name, "test-model");
        assert_eq!(config.device, DeviceType::CUDA);
        assert_eq!(config.quantization, "FP16");
        assert_eq!(config.batch_sizes, vec![1, 2, 4]);
        assert_eq!(config.num_warmup_runs, 3);
        assert_eq!(config.num_benchmark_runs, 10);
    }

    #[test]
    fn test_config_validation() {
        let config = BenchmarkConfig::default();
        let result = config.validate();
        assert!(result.is_err());

        let mut config = BenchmarkConfig::default();
        config.batch_sizes = vec![];
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_serialization() {
        let config = BenchmarkConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: BenchmarkConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.model_name, deserialized.model_name);
    }

    #[test]
    fn test_device_type_display() {
        assert_eq!(DeviceType::CPU.to_string(), "cpu");
        assert_eq!(DeviceType::CUDA.to_string(), "cuda");
        assert_eq!(DeviceType::Metal.to_string(), "metal");
    }

    #[test]
    fn test_output_format_display() {
        assert_eq!(OutputFormat::Json.to_string(), "json");
        assert_eq!(OutputFormat::Csv.to_string(), "csv");
        assert_eq!(OutputFormat::Prometheus.to_string(), "prometheus");
        assert_eq!(OutputFormat::Human.to_string(), "human");
    }
}
