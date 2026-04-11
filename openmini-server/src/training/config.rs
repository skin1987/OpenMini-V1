//! 训练配置系统
//!
//! 支持从 TOML 文件加载完整的训练配置。

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub mode: TrainingMode,
    
    #[serde(default)]
    pub model: ModelConfig,
    
    #[serde(default)]
    pub hyperparams: HyperParams,
    
    #[serde(default)]
    pub optimizer: OptimizerConfig,
    
    #[serde(default)]
    pub scheduler: SchedulerConfig,
    
    #[serde(default)]
    pub data: DataConfig,
    
    #[serde(default)]
    pub checkpoint: CheckpointConfig,
    
    #[serde(default)]
    pub logging: LoggingConfig,
    
    #[serde(default)]
    pub early_stopping: EarlyStoppingConfig,
    
    #[serde(default)]
    pub sft: Option<SFTConfig>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            mode: TrainingMode::CausalLM,
            model: ModelConfig::default(),
            hyperparams: HyperParams::default(),
            optimizer: OptimizerConfig::default(),
            scheduler: SchedulerConfig::default(),
            data: DataConfig::default(),
            checkpoint: CheckpointConfig::default(),
            logging: LoggingConfig::default(),
            early_stopping: EarlyStoppingConfig::default(),
            sft: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingMode {
    #[serde(rename = "causal_lm")]
    CausalLM,
    #[serde(rename = "continue_pretrain")]
    ContinuePretrain,
    #[serde(rename = "sft")]
    SFT,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name_or_path: String,
    #[serde(default = "default_dtype")]
    pub dtype: String,
}

fn default_dtype() -> String { "fp32".to_string() }

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name_or_path: "./models/base_model".to_string(),
            dtype: "fp32".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperParams {
    #[serde(default = "default_epochs")]
    pub num_train_epochs: usize,
    #[serde(default = "default_batch_size")]
    pub per_device_train_batch_size: usize,
    #[serde(default = "default_grad_accum")]
    pub gradient_accumulation_steps: usize,
    #[serde(default = "default_lr")]
    pub learning_rate: f64,
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,
    #[serde(default = "default_max_grad_norm")]
    pub max_grad_norm: f64,
    #[serde(default = "default_max_seq_len")]
    pub max_seq_length: usize,
    #[serde(default = "default_label_smoothing")]
    pub label_smoothing_factor: f64,
}

fn default_epochs() -> usize { 10 }
fn default_batch_size() -> usize { 8 }
fn default_grad_accum() -> usize { 1 }
fn default_lr() -> f64 { 1e-4 }
fn default_weight_decay() -> f64 { 0.01 }
fn default_max_grad_norm() -> f64 { 1.0 }
fn default_max_seq_len() -> usize { 2048 }
fn default_label_smoothing() -> f64 { 0.0 }

impl Default for HyperParams {
    fn default() -> Self {
        Self {
            num_train_epochs: 10,
            per_device_train_batch_size: 8,
            gradient_accumulation_steps: 1,
            learning_rate: 1e-4,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            max_seq_length: 2048,
            label_smoothing_factor: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    #[serde(default = "default_opt_type")]
    pub r#type: String,
    #[serde(default = "default_beta1")]
    pub beta1: f64,
    #[serde(default = "default_beta2")]
    pub beta2: f64,
    #[serde(default = "default_eps")]
    pub epsilon: f64,
}

fn default_opt_type() -> String { "adamw".to_string() }
fn default_beta1() -> f64 { 0.9 }
fn default_beta2() -> f64 { 0.999 }
fn default_eps() -> f64 { 1e-8 }

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            r#type: "adamw".to_string(),
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    #[serde(default = "default_sched_type")]
    pub r#type: String,
    #[serde(default = "default_warmup_ratio")]
    pub warmup_ratio: f64,
    #[serde(default)]
    pub warmup_steps: Option<usize>,
}

fn default_sched_type() -> String { "cosine".to_string() }
fn default_warmup_ratio() -> f64 { 0.1 }

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            r#type: "cosine".to_string(),
            warmup_ratio: 0.1,
            warmup_steps: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    pub train_path: PathBuf,
    #[serde(default)]
    pub validation_path: Option<PathBuf>,
    #[serde(default = "default_format")]
    pub format: String,
    #[serde(default = "default_val_split")]
    pub validation_split: f64,
    #[serde(default = "default_shuffle")]
    pub shuffle: bool,
    #[serde(default = "default_num_workers")]
    pub num_workers: usize,
}

fn default_format() -> String { "jsonl".to_string() }
fn default_val_split() -> f64 { 0.1 }
fn default_shuffle() -> bool { true }
fn default_num_workers() -> usize { 4 }

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            train_path: PathBuf::from("./data/train.jsonl"),
            validation_path: None,
            format: "jsonl".to_string(),
            validation_split: 0.1,
            shuffle: true,
            num_workers: 4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    pub output_dir: PathBuf,
    #[serde(default = "default_save_strategy")]
    pub save_strategy: String,
    #[serde(default = "default_save_steps")]
    pub save_steps: usize,
    #[serde(default = "default_save_limit")]
    pub save_total_limit: usize,
    #[serde(default)]
    pub resume_from_checkpoint: Option<String>,
}

fn default_save_strategy() -> String { "steps".to_string() }
fn default_save_steps() -> usize { 500 }
fn default_save_limit() -> usize { 5 }

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("./checkpoints"),
            save_strategy: "steps".to_string(),
            save_steps: 500,
            save_total_limit: 5,
            resume_from_checkpoint: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    #[serde(default = "default_log_steps")]
    pub log_every_n_steps: usize,
    pub logging_dir: PathBuf,
    #[serde(default)]
    pub save_metrics_history: bool,
}

fn default_log_steps() -> usize { 10 }

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            log_every_n_steps: 10,
            logging_dir: PathBuf::from("./logs"),
            save_metrics_history: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    #[serde(default = "default_es_enabled")]
    pub enabled: bool,
    #[serde(default = "default_patience")]
    pub patience: usize,
    #[serde(default = "default_min_delta")]
    pub min_delta: f64,
}

fn default_es_enabled() -> bool { true }
fn default_patience() -> usize { 5 }
fn default_min_delta() -> f64 { 0.001 }

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            patience: 5,
            min_delta: 0.001,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SFTConfig {
    pub prompt_template: String,
    #[serde(default)]
    pub mask_prompt_loss: bool,
    #[serde(default)]
    pub max_prompt_length: Option<usize>,
    #[serde(default)]
    pub max_response_length: Option<usize>,
}

impl TrainingConfig {
    /// 从 TOML 文件加载配置
    pub fn from_file(path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)
            .map_err(ConfigError::Io)?;
        
        // 解析完整 TOML 文件
        let value: toml::Value = toml::from_str(&content)
            .map_err(|e| ConfigError::ParseError(e.to_string()))?;
        
        // 提取 [training] 段
        let training_value = value.get("training")
            .ok_or_else(|| ConfigError::MissingField("missing [training] section".to_string()))?;
        
        // 将 [training] 段转换为 TrainingConfig
        let config: TrainingConfig = training_value.clone().try_into()
            .map_err(|e| ConfigError::ParseError(e.to_string()))?;
        
        config.validate()?;
        
        Ok(config)
    }
    
    /// 验证配置有效性
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.hyperparams.num_train_epochs == 0 {
            return Err(ConfigError::InvalidValue(
                "num_train_epochs must be > 0".to_string()
            ));
        }
        
        if self.hyperparams.per_device_train_batch_size == 0 {
            return Err(ConfigError::InvalidValue(
                "per_device_train_batch_size must be > 0".to_string()
            ));
        }
        
        if self.hyperparams.learning_rate <= 0.0 {
            return Err(ConfigError::InvalidValue(
                "learning_rate must be > 0".to_string()
            ));
        }
        
        if !["adamw", "sgd"].contains(&self.optimizer.r#type.as_str()) {
            return Err(ConfigError::InvalidValue(format!(
                "Invalid optimizer type: {} (must be 'adamw' or 'sgd')",
                self.optimizer.r#type
            )));
        }
        
        if !["cosine", "linear", "constant"].contains(&self.scheduler.r#type.as_str()) {
            return Err(ConfigError::InvalidValue(format!(
                "Invalid scheduler type: {}",
                self.scheduler.r#type
            )));
        }
        
        Ok(())
    }
    
    /// 计算有效 batch size（考虑梯度累积）
    pub fn effective_batch_size(&self) -> usize {
        self.hyperparams.per_device_train_batch_size * self.hyperparams.gradient_accumulation_steps
    }
    
    /// 计算总训练步数（估算）
    pub fn estimate_total_steps(&self, dataset_size: usize) -> u64 {
        let steps_per_epoch = dataset_size / self.effective_batch_size();
        steps_per_epoch as u64 * self.hyperparams.num_train_epochs as u64
    }
}

#[derive(Debug)]
pub enum ConfigError {
    Io(std::io::Error),
    ParseError(String),
    InvalidValue(String),
    MissingField(String),
}

impl From<std::io::Error> for ConfigError {
    fn from(err: std::io::Error) -> Self { Self::Io(err) }
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::ParseError(s) => write!(f, "Parse error: {}", s),
            Self::InvalidValue(s) => write!(f, "Invalid value: {}", s),
            Self::MissingField(s) => write!(f, "Missing field: {}", s),
        }
    }
}

impl std::error::Error for ConfigError {}

// 单元测试
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_default_config_validity() {
        let config = TrainingConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.mode, TrainingMode::CausalLM);
        assert_eq!(config.hyperparams.learning_rate, 1e-4);
    }
    
    #[test]
    fn test_invalid_learning_rate() {
        let mut config = TrainingConfig::default();
        config.hyperparams.learning_rate = -1.0;
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_effective_batch_size() {
        let mut config = TrainingConfig::default();
        config.hyperparams.per_device_train_batch_size = 8;
        config.hyperparams.gradient_accumulation_steps = 4;
        assert_eq!(config.effective_batch_size(), 32);
    }
    
    #[test]
    fn test_estimate_total_steps() {
        let config = TrainingConfig::default();
        let total = config.estimate_total_steps(10000);
        assert!(total > 0);
    }
    
    #[test]
    fn test_toml_parsing() {
        let tmp = TempDir::new().unwrap();
        let config_path = tmp.path().join("config.toml");
        
        let toml_content = r#"
[training]
mode = "sft"

[training.model]
name_or_path = "./model"
dtype = "fp32"

[training.hyperparams]
num_train_epochs = 5
learning_rate = 3e-5

[training.sft]
prompt_template = "Instruction: {instruction}\nResponse:"
mask_prompt_loss = true
"#;
        
        std::fs::write(&config_path, toml_content).unwrap();
        
        let config = TrainingConfig::from_file(&config_path).unwrap();
        assert_eq!(config.mode, TrainingMode::SFT);
        assert_eq!(config.hyperparams.num_train_epochs, 5);
        assert!(config.sft.is_some());
    }
}
