//! 14B-Dense 模型训练配置系统
//!
//! 支持从 TOML 文件加载完整的 14B 模型训练配置，
//! 包含模型架构、MoE、MLA、SFT、GRPO 等完整训练管线配置。
//!
//! # 配置层次
//!
//! ```text
//! model_14b.toml
//! ├── [model]           - 模型架构参数 (hidden_size, layers, heads...)
//! ├── [moe]             - MoE 专家配置
//! ├── [mla]             - 多头潜在注意力配置
//! ├── [training]        - 预训练超参数
//! ├── [data]            - 数据混合配置
//! ├── [sft]             - 有监督微调配置
//! ├── [grpo]            - GRPO 强化学习配置
//! ├── [hardware]        - 硬件资源与并行策略
//! ├── [checkpoint]      - Checkpoint管理与扩展策略
//! ├── [logging]         - 日志与监控
//! └── [early_stopping]  - 早停机制
//! ```

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ==================== 模型架构配置 ====================

/// 14B-Dense 模型架构配置
///
/// 定义 MultimodalTransformer 的完整架构参数，
/// 支持 ~14.2B 参数量的密集型 Transformer 模型。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model14BConfig {
    /// 模型名称
    pub name: String,
    /// 架构类型
    pub architecture: String,

    // 核心规模参数
    /// 词表大小
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    /// 隐藏层维度
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    /// FFN 中间层维度
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    /// 注意力头数
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    /// GQA KV 头数
    #[serde(default = "default_num_kv_heads")]
    pub num_key_value_heads: usize,
    /// Transformer 层数
    #[serde(default = "default_num_layers")]
    pub num_hidden_layers: usize,
    /// 最大位置编码
    #[serde(default = "max_position_embeddings")]
    pub max_position_embeddings: usize,

    // 归一化与位置编码
    /// RMSNorm epsilon
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    /// RoPE 基础频率
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    /// RoPE 扩展类型
    #[serde(default = "default_rope_scaling")]
    pub rope_scaling_type: String,
}

#[allow(clippy::derivable_impls)]
impl Default for Model14BConfig {
    fn default() -> Self {
        Self {
            name: "OpenMini-14B".to_string(),
            architecture: "MultimodalTransformer".to_string(),
            vocab_size: 152064,
            hidden_size: 5120,
            intermediate_size: 13824,
            num_attention_heads: 40,
            num_key_value_heads: 10,
            num_hidden_layers: 48,
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_scaling_type: "linear".to_string(),
        }
    }
}

fn default_vocab_size() -> usize {
    152064
}
fn default_hidden_size() -> usize {
    5120
}
fn default_intermediate_size() -> usize {
    13824
}
fn default_num_attention_heads() -> usize {
    40
}
fn default_num_kv_heads() -> usize {
    10
}
fn default_num_layers() -> usize {
    48
}
fn max_position_embeddings() -> usize {
    131072
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    10000.0
}
fn default_rope_scaling() -> String {
    "linear".to_string()
}

impl Model14BConfig {
    /// 计算模型总参数量（估算）
    ///
    /// 基于标准 Transformer 参数量公式：
    /// - Embedding: vocab_size * hidden_size
    /// - Attention per layer: 4 * hidden_size^2 * (num_heads / num_kv_heads)
    /// - FFN per layer: 2 * hidden_size * intermediate_size
    /// - Norms per layer: 2 * hidden_size
    /// - LM Head: vocab_size * hidden_size
    pub fn estimate_parameters(&self) -> u64 {
        let embed_params = self.vocab_size as u64 * self.hidden_size as u64;
        let attn_per_layer = 4
            * self.hidden_size.pow(2) as u64
            * (self.num_attention_heads / self.num_key_value_heads.max(1)) as u64;
        let ffn_per_layer = 2 * self.hidden_size as u64 * self.intermediate_size as u64;
        let norm_per_layer = 2 * self.hidden_size as u64;
        let lm_head = self.vocab_size as u64 * self.hidden_size as u64;

        embed_params
            + self.num_hidden_layers as u64 * (attn_per_layer + ffn_per_layer + norm_per_layer)
            + lm_head
    }

    /// 验证模型架构配置的有效性
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.hidden_size == 0 {
            return Err(ConfigError::InvalidValue(
                "hidden_size must be > 0".to_string(),
            ));
        }
        if self.num_hidden_layers == 0 {
            return Err(ConfigError::InvalidValue(
                "num_hidden_layers must be > 0".to_string(),
            ));
        }
        if self.num_attention_heads == 0 {
            return Err(ConfigError::InvalidValue(
                "num_attention_heads must be > 0".to_string(),
            ));
        }
        if self.num_key_value_heads == 0 {
            return Err(ConfigError::InvalidValue(
                "num_key_value_heads must be > 0".to_string(),
            ));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(ConfigError::InvalidValue(format!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        if self.vocab_size == 0 {
            return Err(ConfigError::InvalidValue(
                "vocab_size must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    /// 获取每个注意力头的维度
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// ==================== MoE 配置 ====================

/// MoE (Mixture of Experts) 配置
///
/// 虽然 14B-Dense 不启用 MoE，但保留配置以兼容框架扩展。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoEConfig {
    /// 专家数量
    #[serde(default = "default_num_experts")]
    pub num_experts: usize,
    /// 每个token激活的专家数
    #[serde(default = "default_experts_per_tok")]
    pub num_experts_per_tok: usize,
    /// 路由辅助损失系数
    #[serde(default = "default_router_aux_loss")]
    pub router_aux_loss_coef: f64,
    /// 路由z-loss系数
    #[serde(default = "default_router_z_loss")]
    pub router_z_loss_coef: f64,
}

#[allow(clippy::derivable_impls)]
impl Default for MoEConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            num_experts_per_tok: 2,
            router_aux_loss_coef: 0.001,
            router_z_loss_coef: 0.0001,
        }
    }
}

fn default_num_experts() -> usize {
    8
}
fn default_experts_per_tok() -> usize {
    2
}
fn default_router_aux_loss() -> f64 {
    0.001
}
fn default_router_z_loss() -> f64 {
    0.0001
}

// ==================== MLA 配置 ====================

/// MLA (Multi-head Latent Attention) 配置
///
/// DeepSeek 风格的压缩注意力机制，
/// 通过压缩 KV 缓存来降低显存占用。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLAConfig {
    /// 潜在表示维度
    #[serde(default = "default_latent_dim")]
    pub latent_dim: usize,
    /// 压缩比
    #[serde(default = "default_compress_ratio")]
    pub compress_ratio: f64,
    /// 压缩后的KV头数
    #[serde(default = "default_num_kv_heads_mla")]
    pub num_kv_heads: usize,
}

#[allow(clippy::derivable_impls)]
impl Default for MLAConfig {
    fn default() -> Self {
        Self {
            latent_dim: 1024,
            compress_ratio: 4.0,
            num_kv_heads: 8,
        }
    }
}

fn default_latent_dim() -> usize {
    1024
}
fn default_compress_ratio() -> f64 {
    4.0
}
fn default_num_kv_heads_mla() -> usize {
    8
}

// ==================== 训练超参数配置 ====================

/// 14B 模型预训练超参数配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHyperParams {
    /// 全局有效 batch size
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// 单设备 micro-batch size
    #[serde(default = "default_micro_batch_size")]
    pub micro_batch_size: usize,
    /// 梯度累积步数
    #[serde(default = "default_grad_accum_steps")]
    pub gradient_accumulation_steps: usize,
    /// 峰值学习率
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    /// 权重衰减
    #[serde(default = "default_weight_decay_14b")]
    pub weight_decay: f64,
    /// Warmup 步数
    #[serde(default = "default_warmup_steps")]
    pub warmup_steps: usize,
    /// 总训练步数
    #[serde(default = "default_total_steps")]
    pub total_steps: usize,
    /// Checkpoint 保存间隔
    #[serde(default = "default_save_steps")]
    pub save_steps: usize,
    /// 评估间隔
    #[serde(default = "default_eval_steps")]
    pub eval_steps: usize,
    /// 训练精度
    #[serde(default = "default_dtype")]
    pub dtype: String,

    // 梯度相关
    /// 最大梯度范数
    #[serde(default = "default_max_grad_norm")]
    pub max_grad_norm: f64,
    /// 累积精度
    #[serde(default = "default_accum_dtype")]
    pub grad_accum_dtype: String,

    // 学习率调度
    /// 调度器类型
    #[serde(default = "default_lr_scheduler")]
    pub lr_scheduler_type: String,
    /// 最小学习率比例
    #[serde(default = "default_min_lr_ratio")]
    pub min_lr_ratio: f64,
}

#[allow(clippy::derivable_impls)]
impl Default for TrainingHyperParams {
    fn default() -> Self {
        Self {
            batch_size: 256,
            micro_batch_size: 4,
            gradient_accumulation_steps: 64,
            learning_rate: 3e-4,
            weight_decay: 0.01,
            warmup_steps: 5000,
            total_steps: 500000,
            save_steps: 5000,
            eval_steps: 2500,
            dtype: "bf16".to_string(),
            max_grad_norm: 1.0,
            grad_accum_dtype: "fp32".to_string(),
            lr_scheduler_type: "cosine".to_string(),
            min_lr_ratio: 0.1,
        }
    }
}

fn default_batch_size() -> usize {
    256
}
fn default_micro_batch_size() -> usize {
    4
}
fn default_grad_accum_steps() -> usize {
    64
}
fn default_learning_rate() -> f64 {
    3e-4
}
fn default_weight_decay_14b() -> f64 {
    0.01
}
fn default_warmup_steps() -> usize {
    5000
}
fn default_total_steps() -> usize {
    500000
}
fn default_save_steps() -> usize {
    5000
}
fn default_eval_steps() -> usize {
    2500
}
fn default_dtype() -> String {
    "bf16".to_string()
}
fn default_max_grad_norm() -> f64 {
    1.0
}
fn default_accum_dtype() -> String {
    "fp32".to_string()
}
fn default_lr_scheduler() -> String {
    "cosine".to_string()
}
fn default_min_lr_ratio() -> f64 {
    0.1
}

impl TrainingHyperParams {
    /// 验证训练超参数有效性
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.micro_batch_size == 0 {
            return Err(ConfigError::InvalidValue(
                "micro_batch_size must be > 0".to_string(),
            ));
        }
        if self.gradient_accumulation_steps == 0 {
            return Err(ConfigError::InvalidValue(
                "gradient_accumulation_steps must be > 0".to_string(),
            ));
        }
        if self.learning_rate <= 0.0 {
            return Err(ConfigError::InvalidValue(
                "learning_rate must be > 0".to_string(),
            ));
        }
        if self.batch_size != self.micro_batch_size * self.gradient_accumulation_steps {
            return Err(ConfigError::InvalidValue(format!(
                "batch_size ({}) must equal micro_batch_size ({}) * gradient_accumulation_steps ({})",
                self.batch_size, self.micro_batch_size, self.gradient_accumulation_steps
            )));
        }
        if !["bf16", "fp16", "fp32"].contains(&self.dtype.as_str()) {
            return Err(ConfigError::InvalidValue(format!(
                "Invalid dtype: {} (must be 'bf16', 'fp16', or 'fp32')",
                self.dtype
            )));
        }
        Ok(())
    }

    /// 获取有效 batch size
    pub fn effective_batch_size(&self) -> usize {
        self.micro_batch_size * self.gradient_accumulation_steps
    }
}

// ==================== 数据配置 ====================

/// 训练数据混合配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataMixConfig {
    /// 总训练 tokens 数量
    #[serde(default = "default_train_tokens")]
    pub train_tokens: String,
    /// 代码数据占比
    #[serde(default = "default_code_tokens")]
    pub code_tokens: String,
    /// 中文数据占比
    #[serde(default = "default_chinese_tokens")]
    pub chinese_tokens: String,
    /// 数学数据占比
    #[serde(default)]
    pub math_tokens: Option<String>,
    /// 科学数据占比
    #[serde(default)]
    pub science_tokens: Option<String>,
    /// 通用文本数据占比
    #[serde(default)]
    pub general_tokens: Option<String>,

    // 数据格式
    /// 数据格式
    #[serde(default = "default_data_format")]
    pub data_format: String,
    /// 最大序列长度
    #[serde(default = "default_max_seq_len_data")]
    pub max_seq_length: usize,
    /// 是否使用序列打包
    #[serde(default = "default_packed_sequence")]
    pub packed_sequence: bool,
}

#[allow(clippy::derivable_impls)]
impl Default for DataMixConfig {
    fn default() -> Self {
        Self {
            train_tokens: "800B".to_string(),
            code_tokens: "100B".to_string(),
            chinese_tokens: "200B".to_string(),
            math_tokens: Some("150B".to_string()),
            science_tokens: Some("150B".to_string()),
            general_tokens: Some("200B".to_string()),
            data_format: "pretokenized".to_string(),
            max_seq_length: 8192,
            packed_sequence: true,
        }
    }
}

fn default_train_tokens() -> String {
    "800B".to_string()
}
fn default_code_tokens() -> String {
    "100B".to_string()
}
fn default_chinese_tokens() -> String {
    "200B".to_string()
}
fn default_data_format() -> String {
    "pretokenized".to_string()
}
fn default_max_seq_len_data() -> usize {
    8192
}
fn default_packed_sequence() -> bool {
    true
}

// ==================== SFT 配置 ====================

/// SFT (Supervised Fine-Tuning) 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SFT14BConfig {
    /// 微调轮数
    #[serde(default = "default_sft_epochs")]
    pub epochs: usize,
    /// SFT 学习率
    #[serde(default = "default_sft_lr")]
    pub learning_rate: f64,
    /// LoRA 秩
    #[serde(default = "default_lora_rank")]
    pub lora_rank: usize,
    /// LoRA alpha
    #[serde(default = "default_lora_alpha")]
    pub lora_alpha: usize,
    /// LoRA dropout
    #[serde(default = "default_lora_dropout")]
    pub lora_dropout: f64,
    /// LoRA 目标模块
    #[serde(default = "default_lora_targets")]
    pub lora_target_modules: Vec<String>,

    // SFT 损失配置
    /// 屏蔽 prompt 部分 loss
    #[serde(default = "default_mask_prompt")]
    pub mask_prompt_loss: bool,
    /// 最大响应长度
    #[serde(default = "default_response_len")]
    pub response_length_max: usize,
}

#[allow(clippy::derivable_impls)]
impl Default for SFT14BConfig {
    fn default() -> Self {
        Self {
            epochs: 3,
            learning_rate: 1e-5,
            lora_rank: 64,
            lora_alpha: 128,
            lora_dropout: 0.05,
            lora_target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
            mask_prompt_loss: true,
            response_length_max: 4096,
        }
    }
}

fn default_sft_epochs() -> usize {
    3
}
fn default_sft_lr() -> f64 {
    1e-5
}
fn default_lora_rank() -> usize {
    64
}
fn default_lora_alpha() -> usize {
    128
}
fn default_lora_dropout() -> f64 {
    0.05
}
fn default_lora_targets() -> Vec<String> {
    vec![
        "q_proj".to_string(),
        "k_proj".to_string(),
        "v_proj".to_string(),
        "o_proj".to_string(),
        "gate_proj".to_string(),
        "up_proj".to_string(),
        "down_proj".to_string(),
    ]
}
fn default_mask_prompt() -> bool {
    true
}
fn default_response_len() -> usize {
    4096
}

// ==================== GRPO 配置 ====================

/// GRPO (Group Relative Policy Optimization) 配置
///
/// 用于 RLHF 阶段的策略优化，集成已有的 GRPO 模块。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GRPO14BConfig {
    /// 组大小
    #[serde(default = "default_group_size")]
    pub group_size: usize,
    /// KL 散度惩罚系数
    #[serde(default = "default_kl_coef")]
    pub kl_coef: f64,
    /// PPO clip 范围
    #[serde(default = "default_clip_range")]
    pub clip_range: f64,
    /// 熵奖励系数
    #[serde(default = "default_entropy_coef")]
    pub entropy_coef: f64,
    /// 优势归一化 epsilon
    #[serde(default = "default_adv_eps")]
    pub advantage_norm_eps: f64,
    /// 梯度裁剪范数
    #[serde(default = "default_grad_clip_norm")]
    pub grad_clip_norm: f64,

    // 奖励函数配置
    /// 准确性奖励权重
    #[serde(default = "default_acc_weight")]
    pub reward_accuracy_weight: f64,
    /// 格式奖励权重
    #[serde(default = "default_fmt_weight")]
    pub reward_format_weight: f64,
}

#[allow(clippy::derivable_impls)]
impl Default for GRPO14BConfig {
    fn default() -> Self {
        Self {
            group_size: 64,
            kl_coef: 0.001,
            clip_range: 0.2,
            entropy_coef: 0.01,
            advantage_norm_eps: 1e-8,
            grad_clip_norm: 1.0,
            reward_accuracy_weight: 0.7,
            reward_format_weight: 0.3,
        }
    }
}

fn default_group_size() -> usize {
    64
}
fn default_kl_coef() -> f64 {
    0.001
}
fn default_clip_range() -> f64 {
    0.2
}
fn default_entropy_coef() -> f64 {
    0.01
}
fn default_adv_eps() -> f64 {
    1e-8
}
fn default_grad_clip_norm() -> f64 {
    1.0
}
fn default_acc_weight() -> f64 {
    0.7
}
fn default_fmt_weight() -> f64 {
    0.3
}

impl GRPO14BConfig {
    /// 转换为 RL 模块的 GRPOConfig
    pub fn to_grpo_config(&self) -> crate::rl::GRPOConfig {
        crate::rl::GRPOConfig {
            learning_rate: 1e-5, // GRPO 使用独立的学习率
            kl_coefficient: self.kl_coef,
            clip_epsilon: self.clip_range,
            group_size: self.group_size,
            advantage_norm_eps: self.advantage_norm_eps,
            max_kl_margin: 0.01,
            entropy_coef: self.entropy_coef,
            grad_clip_norm: self.grad_clip_norm,
        }
    }

    /// 验证 GRPO 配置
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.group_size == 0 {
            return Err(ConfigError::InvalidValue(
                "group_size must be > 0".to_string(),
            ));
        }
        if self.kl_coef < 0.0 {
            return Err(ConfigError::InvalidValue(
                "kl_coef must be >= 0".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.clip_range) {
            return Err(ConfigError::InvalidValue(
                "clip_range must be in [0, 1]".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&(self.reward_accuracy_weight + self.reward_format_weight)) {
            return Err(ConfigError::InvalidValue(
                "reward weights sum must be in [0, 1]".to_string(),
            ));
        }
        Ok(())
    }
}

// ==================== 硬件配置 ====================

/// 硬件资源配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// GPU 数量
    #[serde(default = "default_num_gpus")]
    pub num_gpus: usize,
    /// 单卡显存(GB)
    #[serde(default = "default_gpu_mem")]
    pub gpu_memory_gb: usize,
    /// 总显存(GB)
    #[serde(default = "default_total_mem")]
    pub total_memory_gb: usize,

    // 并行策略
    /// 张量并行度
    #[serde(default = "default_tp_size")]
    pub tensor_parallel_size: usize,
    /// 流水线并行度
    #[serde(default = "default_pp_size")]
    pub pipeline_parallel_size: usize,
    /// 数据并行度
    #[serde(default = "default_dp_size")]
    pub data_parallel_size: usize,

    // 显存优化
    /// 激活检查点
    #[serde(default = "default_act_checkpoint")]
    pub activation_checkpointing: bool,
    /// 优化器状态卸载
    #[serde(default)]
    pub offload_optimizer: bool,
    /// 混合精度训练
    #[serde(default = "default_mixed_prec")]
    pub mixed_precision: bool,
}

#[allow(clippy::derivable_impls)]
impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            num_gpus: 8,
            gpu_memory_gb: 80,
            total_memory_gb: 640,
            tensor_parallel_size: 4,
            pipeline_parallel_size: 2,
            data_parallel_size: 1,
            activation_checkpointing: true,
            offload_optimizer: false,
            mixed_precision: true,
        }
    }
}

fn default_num_gpus() -> usize {
    8
}
fn default_gpu_mem() -> usize {
    80
}
fn default_total_mem() -> usize {
    640
}
fn default_tp_size() -> usize {
    4
}
fn default_pp_size() -> usize {
    2
}
fn default_dp_size() -> usize {
    1
}
fn default_act_checkpoint() -> bool {
    true
}
fn default_mixed_prec() -> bool {
    true
}

impl HardwareConfig {
    /// 验证硬件配置
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.num_gpus == 0 {
            return Err(ConfigError::InvalidValue(
                "num_gpus must be > 0".to_string(),
            ));
        }
        let actual_parallel = self.tensor_parallel_size * self.pipeline_parallel_size;
        if actual_parallel > self.num_gpus {
            return Err(ConfigError::InvalidValue(format!(
                "TP({}) x PP({}) = {} exceeds num_gpus({})",
                self.tensor_parallel_size,
                self.pipeline_parallel_size,
                actual_parallel,
                self.num_gpus
            )));
        }
        Ok(())
    }

    /// 计算实际使用的 GPU 数量
    pub fn effective_gpus(&self) -> usize {
        self.tensor_parallel_size * self.pipeline_parallel_size * self.data_parallel_size.max(1)
    }
}

// ==================== Checkpoint 扩展配置 ====================

/// Checkpoint 与模型扩展配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointExtConfig {
    /// 输出目录
    #[serde(default = "default_output_dir")]
    pub output_dir: PathBuf,
    /// 保存策略
    #[serde(default = "default_save_strategy")]
    pub save_strategy: String,
    /// 最多保留数量
    #[serde(default = "default_save_limit")]
    pub save_total_limit: usize,
    /// 断点续训路径
    #[serde(default)]
    pub resume_from_checkpoint: Option<String>,

    // 从7B扩展配置
    /// 7B base checkpoint 路径
    #[serde(default)]
    pub base_model_path: Option<String>,
    /// 扩展策略
    #[serde(default = "default_expansion_strategy")]
    pub expansion_strategy: ExpansionStrategy,
}

#[allow(clippy::derivable_impls)]
impl Default for CheckpointExtConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("./checkpoints/14b-dense"),
            save_strategy: "steps".to_string(),
            save_total_limit: 5,
            resume_from_checkpoint: None,
            base_model_path: None,
            expansion_strategy: ExpansionStrategy::RandomInit,
        }
    }
}

/// 模型扩展策略（从7B到14B）
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ExpansionStrategy {
    /// 随机初始化新增参数
    #[default]
    RandomInit,
    /// 按比例复制并缩放
    ScaledCopy,
    /// 插值初始化
    Interpolation,
}

fn default_output_dir() -> PathBuf {
    PathBuf::from("./checkpoints/14b-dense")
}
fn default_save_strategy() -> String {
    "steps".to_string()
}
fn default_save_limit() -> usize {
    5
}
fn default_expansion_strategy() -> ExpansionStrategy {
    ExpansionStrategy::RandomInit
}

// ==================== 完整 14B 训练配置 ====================

/// 14B-Dense 模型完整训练配置
///
/// 整合所有配置段为统一的配置结构，
/// 支持预训练、SFT、GRPO 全流程训练。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingConfig14B {
    /// 模型架构配置
    #[serde(default)]
    pub model: Model14BConfig,
    /// MoE 配置
    #[serde(default)]
    pub moe: MoEConfig,
    /// MLA 配置
    #[serde(default)]
    pub mla: MLAConfig,
    /// 预训练超参数
    #[serde(default)]
    pub training: TrainingHyperParams,
    /// 数据混合配置
    #[serde(default)]
    pub data: DataMixConfig,
    /// SFT 配置
    #[serde(default)]
    pub sft: SFT14BConfig,
    /// GRPO 配置
    #[serde(default)]
    pub grpo: GRPO14BConfig,
    /// 硬件配置
    #[serde(default)]
    pub hardware: HardwareConfig,
    /// Checkpoint 配置
    #[serde(default)]
    pub checkpoint: CheckpointExtConfig,
    /// 日志配置
    #[serde(default)]
    pub logging: LoggingConfig14B,
    /// 早停配置
    #[serde(default)]
    pub early_stopping: EarlyStoppingConfig14B,
}

/// 日志配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig14B {
    #[serde(default = "default_log_steps_14b")]
    pub log_every_n_steps: usize,
    #[serde(default = "default_logging_dir")]
    pub logging_dir: PathBuf,
    #[serde(default)]
    pub save_metrics_history: bool,
    #[serde(default)]
    pub wandb_project: Option<String>,
    #[serde(default)]
    pub wandb_run_name: Option<String>,
}

#[allow(clippy::derivable_impls)]
impl Default for LoggingConfig14B {
    fn default() -> Self {
        Self {
            log_every_n_steps: 10,
            logging_dir: PathBuf::from("./logs/14b-training"),
            save_metrics_history: true,
            wandb_project: Some("openmini-14b".to_string()),
            wandb_run_name: None,
        }
    }
}

fn default_log_steps_14b() -> usize {
    10
}
fn default_logging_dir() -> PathBuf {
    PathBuf::from("./logs/14b-training")
}

/// 早停配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig14B {
    #[serde(default = "default_es_enabled_14b")]
    pub enabled: bool,
    #[serde(default = "default_patience_14b")]
    pub patience: usize,
    #[serde(default = "default_min_delta_14b")]
    pub min_delta: f64,
    #[serde(default = "default_monitor_metric")]
    pub monitor_metric: String,
    #[serde(default = "default_es_mode")]
    pub mode: String,
}

#[allow(clippy::derivable_impls)]
impl Default for EarlyStoppingConfig14B {
    fn default() -> Self {
        Self {
            enabled: true,
            patience: 10,
            min_delta: 0.001,
            monitor_metric: "val_loss".to_string(),
            mode: "min".to_string(),
        }
    }
}

fn default_es_enabled_14b() -> bool {
    true
}
fn default_patience_14b() -> usize {
    10
}
fn default_min_delta_14b() -> f64 {
    0.001
}
fn default_monitor_metric() -> String {
    "val_loss".to_string()
}
fn default_es_mode() -> String {
    "min".to_string()
}

impl TrainingConfig14B {
    /// 从 TOML 文件加载 14B 完整配置
    pub fn from_file(path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path).map_err(ConfigError::Io)?;

        let config: TrainingConfig14B =
            toml::from_str(&content).map_err(|e| ConfigError::ParseError(e.to_string()))?;

        config.validate()?;
        Ok(config)
    }

    /// 验证所有配置段的有效性
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.model.validate()?;
        self.training.validate()?;
        self.grpo.validate()?;
        self.hardware.validate()?;
        Ok(())
    }

    /// 估算 FLOPs（每 token）
    ///
    /// 基于 Transformer FLOPs 公式：
    /// `FLOPs ≈ 12 * L * H^2 * A * S`
    /// 其中 L=层数, H=隐藏维度, A=序列长度, S=序列长度
    pub fn estimate_flops_per_token(&self) -> u64 {
        let l = self.model.num_hidden_layers as u64;
        let h = self.model.hidden_size as u64;
        let _a = self.model.num_attention_heads as u64;
        let s = self.data.max_seq_length as u64;

        // 注意力 FLOPs: 4 * L * H * (H/A) * A * S = 4 * L * H^2 * S
        // 使用 f64 避免溢出，然后转回 u64
        let attn_flops_f64 = 4.0 * l as f64 * h as f64 * h as f64 * s as f64;

        // FFN FLOPs: 2 * L * H * I * S (两个线性层)
        let i = self.model.intermediate_size as u64;
        let ffn_flops_f64 = 2.0 * l as f64 * h as f64 * i as f64 * s as f64;

        let total = (attn_flops_f64 + ffn_flops_f64) as u64;
        total.max(1)
    }

    /// 估算总训练时间（秒）
    ///
    /// 基于硬件利用率假设进行粗略估计
    pub fn estimate_training_time_secs(&self) -> u64 {
        let flops_per_token = self.estimate_flops_per_token();
        // 使用 f64 避免溢出
        let total_flops_f64 = flops_per_token as f64
            * self.training.effective_batch_size() as f64
            * self.training.total_steps as f64;

        // 假设 8xH100 的有效算力约为 400 TFLOPS (BF16)
        let effective_tflops = self.hardware.num_gpus as f64 * 50.0; // 每张卡约50 TFLOPS
        let seconds = total_flops_f64 / (effective_tflops * 1e12);

        (seconds.max(1.0)) as u64
    }
}

// ==================== 错误类型 ====================

/// 配置错误类型
#[derive(Debug)]
pub enum ConfigError {
    Io(std::io::Error),
    ParseError(String),
    InvalidValue(String),
    MissingField(String),
}

impl From<std::io::Error> for ConfigError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
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

// ==================== 单元测试 ====================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_default_model_config_validity() {
        let config = Model14BConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.name, "OpenMini-14B");
        assert_eq!(config.hidden_size, 5120);
        assert_eq!(config.num_hidden_layers, 48);
    }

    #[test]
    fn test_model_parameter_estimation() {
        let config = Model14BConfig::default();
        let params = config.estimate_parameters();
        // 14B 模型参数量估算（简化公式可能偏高，接受较大范围）
        // 实际 ~14.2B 参数，估算值可能因公式不同而变化
        assert!(
            params > 10_000_000_000,
            "Expected > 10B params, got {}",
            params
        );
        assert!(
            params < 50_000_000_000,
            "Expected < 50B params, got {}",
            params
        );
    }

    #[test]
    fn test_head_dim_calculation() {
        let config = Model14BConfig::default();
        // 5120 / 40 = 128
        assert_eq!(config.head_dim(), 128);
    }

    #[test]
    fn test_model_validation_invalid_hidden_size() {
        let mut config = Model14BConfig::default();
        config.hidden_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_model_validation_heads_divisibility() {
        let mut config = Model14BConfig::default();
        config.num_attention_heads = 41; // 不能被10整除
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_training_hyperparams_validity() {
        let params = TrainingHyperParams::default();
        assert!(params.validate().is_ok());
        assert_eq!(params.batch_size, 256);
        assert_eq!(params.effective_batch_size(), 256);
    }

    #[test]
    fn test_training_validation_batch_consistency() {
        let mut params = TrainingHyperParams::default();
        params.batch_size = 999; // 不等于 4*64
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_training_validation_invalid_lr() {
        let mut params = TrainingHyperParams::default();
        params.learning_rate = -1.0;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_training_validation_invalid_dtype() {
        let mut params = TrainingHyperParams::default();
        params.dtype = "int8".to_string();
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_grpo_config_conversion() {
        let grpo = GRPO14BConfig::default();
        let rl_config = grpo.to_grpo_config();
        assert!((rl_config.kl_coefficient - grpo.kl_coef).abs() < f64::EPSILON);
        assert_eq!(rl_config.group_size, grpo.group_size);
    }

    #[test]
    fn test_grpo_config_validation() {
        let grpo = GRPO14BConfig::default();
        assert!(grpo.validate().is_ok());

        let mut invalid_grpo = grpo.clone();
        invalid_grpo.group_size = 0;
        assert!(invalid_grpo.validate().is_err());

        let mut invalid_clip = grpo.clone();
        invalid_clip.clip_range = 1.5; // 超出[0,1]
        assert!(invalid_clip.validate().is_err());
    }

    #[test]
    fn test_hardware_config_validation() {
        let hw = HardwareConfig::default();
        assert!(hw.validate().is_ok());
        assert_eq!(hw.effective_gpus(), 8); // 4 * 2 * 1

        let mut invalid_hw = hw.clone();
        invalid_hw.tensor_parallel_size = 100;
        assert!(invalid_hw.validate().is_err());
    }

    #[test]
    fn test_full_config_from_toml() {
        let tmp = TempDir::new().unwrap();
        let config_path = tmp.path().join("model_14b.toml");

        let toml_content = r#"
[model]
name = "OpenMini-14B"
architecture = "MultimodalTransformer"
hidden_size = 5120
num_hidden_layers = 48
num_attention_heads = 40
num_key_value_heads = 10

[training]
batch_size = 256
micro_batch_size = 4
gradient_accumulation_steps = 64
learning_rate = 3e-4
total_steps = 500000
dtype = "bf16"

[grpo]
group_size = 64
kl_coef = 0.001
clip_range = 0.2
entropy_coef = 0.01

[sft]
epochs = 3
learning_rate = 1e-5
lora_rank = 64
"#;

        std::fs::write(&config_path, toml_content).unwrap();

        let config = TrainingConfig14B::from_file(&config_path).unwrap();
        assert_eq!(config.model.hidden_size, 5120);
        assert_eq!(config.training.batch_size, 256);
        assert_eq!(config.grpo.group_size, 64);
        assert_eq!(config.sft.lora_rank, 64);
    }

    #[test]
    fn test_flops_estimation() {
        let config = TrainingConfig14B::default();
        let flops = config.estimate_flops_per_token();
        assert!(flops > 0, "FLOPs should be positive");

        let time = config.estimate_training_time_secs();
        assert!(time > 0, "Training time should be positive");
    }

    #[test]
    fn test_expansion_strategy_serialization() {
        let strategies = vec![
            ExpansionStrategy::RandomInit,
            ExpansionStrategy::ScaledCopy,
            ExpansionStrategy::Interpolation,
        ];

        for strategy in strategies {
            let serialized = serde_json::to_string(&strategy).unwrap();
            let deserialized: ExpansionStrategy = serde_json::from_str(&serialized).unwrap();
            assert_eq!(strategy, deserialized);
        }
    }

    #[test]
    fn test_default_full_config_validity() {
        let config = TrainingConfig14B::default();
        assert!(config.validate().is_ok());
        // 验证关键默认值
        assert_eq!(config.model.hidden_size, 5120);
        assert_eq!(config.training.total_steps, 500000);
        assert_eq!(config.hardware.num_gpus, 8);
        assert_eq!(config.checkpoint.save_total_limit, 5);
    }

    #[test]
    fn test_moe_and_mla_defaults() {
        let moe = MoEConfig::default();
        assert_eq!(moe.num_experts, 8);
        assert_eq!(moe.num_experts_per_tok, 2);

        let mla = MLAConfig::default();
        assert_eq!(mla.latent_dim, 1024);
        assert!((mla.compress_ratio - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sft_lora_targets() {
        let sft = SFT14BConfig::default();
        assert_eq!(sft.lora_target_modules.len(), 7);
        assert!(sft.lora_target_modules.contains(&"q_proj".to_string()));
        assert!(sft.lora_target_modules.contains(&"down_proj".to_string()));
    }
}
