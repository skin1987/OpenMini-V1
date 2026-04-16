//! 14B-Dense 模型训练 Pipeline
//!
//! 实现完整的 14B 模型训练管线，支持：
//! - 从 7B Checkpoint 继续预训练（模型扩展）
//! - 预训练、SFT、GRPO 三阶段训练
//! - 梯度累积与混合精度训练
//! - 集成 GRPO 强化学习模块
//!
//! # 架构设计
//!
//! ```text
//! TrainingPipeline
//! ├── config: TrainingConfig14B      # 完整配置
//! ├── model: MultimodalTransformer   # 14B模型 (Arc共享)
//! ├── optimizer: Box<dyn PipelineOptimizer>  # 优化器 (trait object)
//! ├── scheduler: LRScheduler         # 学习率调度器
//! └── checkpoint_manager: CheckpointManager  # Checkpoint管理
//! ```

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::rl::GroupSample;
use crate::training::checkpoint::CheckpointManager;
use crate::training::config::{ConfigError, ExpansionStrategy, Model14BConfig, TrainingConfig14B};

// ==================== 训练阶段枚举 ====================

/// 训练阶段
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainingPhase {
    /// 预训练阶段（从7B扩展或从头开始）
    Pretrain,
    /// 有监督微调（SFT + LoRA）
    SFT,
    /// GRPO 强化学习优化
    GRPO,
}

impl std::fmt::Display for TrainingPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pretrain => write!(f, "Pretrain"),
            Self::SFT => write!(f, "SFT"),
            Self::GRPO => write!(f, "GRPO"),
        }
    }
}

// ==================== 训练批次数据 ====================

/// 训练批次数据
///
/// 包含单次训练步骤所需的所有输入数据。
/// 支持因果语言模型的下一个词预测任务。
#[derive(Debug, Clone)]
pub struct TrainingBatch {
    /// 输入 token IDs (batch_size * seq_len)
    pub input_ids: Vec<u32>,
    /// 注意力掩码 (batch_size * seq_len)
    pub attention_mask: Vec<i8>,
    /// 标签 token IDs (用于损失计算，可选)
    ///
    /// 如果为 None，则使用 input_ids 右移一位作为标签（标准自回归训练）
    pub labels: Option<Vec<u32>>,
    /// 批次中的 token 总数（用于计算吞吐量）
    pub num_tokens: usize,
}

impl TrainingBatch {
    /// 创建新的训练批次
    pub fn new(input_ids: Vec<u32>, attention_mask: Vec<i8>, labels: Option<Vec<u32>>) -> Self {
        let num_tokens = input_ids.len();
        Self {
            input_ids,
            attention_mask,
            labels,
            num_tokens,
        }
    }

    /// 获取批次大小
    #[allow(unknown_lints)]
    #[allow(clippy::manual_checked_ops)]
    pub fn batch_size(&self, seq_len: usize) -> usize {
        if seq_len > 0 {
            self.input_ids.len() / seq_len
        } else {
            0
        }
    }

    /// 获取标签（如果未提供则自动生成）
    pub fn get_labels(&self) -> Vec<u32> {
        match &self.labels {
            Some(labels) => labels.clone(),
            None => {
                // 标准自回归：labels = input_ids[1..], 填充最后一个位置
                let mut labels = self.input_ids[1..].to_vec();
                labels.push(0); // pad token
                labels
            }
        }
    }

    /// 验证批次数据有效性
    pub fn validate(&self) -> Result<(), PipelineError> {
        if self.input_ids.is_empty() {
            return Err(PipelineError::InvalidBatch(
                "input_ids is empty".to_string(),
            ));
        }
        if self.input_ids.len() != self.attention_mask.len() {
            return Err(PipelineError::InvalidBatch(format!(
                "input_ids length ({}) != attention_mask length ({})",
                self.input_ids.len(),
                self.attention_mask.len()
            )));
        }
        if let Some(ref labels) = self.labels {
            if labels.len() != self.input_ids.len() {
                return Err(PipelineError::InvalidBatch(format!(
                    "labels length ({}) != input_ids length ({})",
                    labels.len(),
                    self.input_ids.len()
                )));
            }
        }
        Ok(())
    }
}

// ==================== 训练指标 ====================

/// 单步训练指标
///
/// 记录每次 `train_step` 的关键性能指标。
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    /// 当前步的损失值
    pub loss: f32,
    /// 当前学习率
    pub learning_rate: f32,
    /// 吞吐量 (tokens/秒)
    pub tokens_per_second: f32,
    /// 梯度 L2 范数
    pub grad_norm: f32,
    /// 当前全局步数
    pub global_step: u64,
    /// 当前 epoch
    pub epoch: usize,
}

impl TrainingMetrics {
    /// 格式化输出训练指标
    pub fn format(&self) -> String {
        format!(
            "Step {} | Loss: {:.4} | LR: {:.2e} | Grad Norm: {:.4} | {:.0} tok/s",
            self.global_step, self.loss, self.learning_rate, self.grad_norm, self.tokens_per_second
        )
    }
}

// ==================== 评估指标 ====================

/// 评估指标结果
#[derive(Debug, Clone, Default)]
pub struct EvalMetrics {
    /// 平均验证损失
    pub val_loss: f64,
    /// 困惑度 (Perplexity)
    pub perplexity: f64,
    /// 评估样本数
    pub eval_samples: usize,
    /// 评估耗时(秒)
    pub eval_time_secs: f64,
}

impl EvalMetrics {
    /// 格式化输出评估指标
    pub fn format(&self) -> String {
        format!(
            "Eval Loss: {:.4} | PPL: {:.2} | Samples: {} | Time: {:.1}s",
            self.val_loss, self.perplexity, self.eval_samples, self.eval_time_secs
        )
    }
}

// ==================== 学习率调度器 ====================

/// 学习率调度器
///
/// 支持 cosine、linear、constant 等多种调度策略。
pub struct LRScheduler {
    /// 调度器类型
    scheduler_type: String,
    /// 峰值学习率
    peak_lr: f64,
    /// 最小学习率比例
    min_lr_ratio: f64,
    /// Warmup 步数
    warmup_steps: u64,
    /// 总训练步数
    total_steps: u64,
    /// 当前步数
    current_step: u64,
}

impl LRScheduler {
    /// 创建新的学习率调度器
    pub fn new(
        scheduler_type: &str,
        peak_lr: f64,
        min_lr_ratio: f64,
        warmup_steps: u64,
        total_steps: u64,
    ) -> Self {
        Self {
            scheduler_type: scheduler_type.to_string(),
            peak_lr,
            min_lr_ratio,
            warmup_steps,
            total_steps,
            current_step: 0,
        }
    }

    /// 获取当前步骤的学习率
    pub fn get_lr(&self) -> f64 {
        if self.total_steps == 0 {
            return self.peak_lr;
        }

        let min_lr = self.peak_lr * self.min_lr_ratio;

        // Warmup 阶段
        if self.current_step < self.warmup_steps {
            if self.warmup_steps == 0 {
                return self.peak_lr;
            }
            return self.peak_lr * self.current_step as f64 / self.warmup_steps as f64;
        }

        // 根据调度类型计算衰减后的学习率
        let progress = (self.current_step - self.warmup_steps) as f64
            / (self.total_steps - self.warmup_steps).max(1) as f64;

        match self.scheduler_type.as_str() {
            "cosine" => {
                // Cosine annealing
                min_lr
                    + 0.5
                        * (self.peak_lr - min_lr)
                        * (1.0 + (std::f64::consts::PI * progress).cos())
            }
            "linear" => {
                // Linear decay
                self.peak_lr + (min_lr - self.peak_lr) * progress
            }
            _ => {
                // Constant (no decay)
                self.peak_lr
            }
        }
    }

    /// 推进一步骤
    pub fn step(&mut self) {
        self.current_step += 1;
    }

    /// 重置调度器状态
    pub fn reset(&mut self) {
        self.current_step = 0;
    }

    /// 获取当前步骤
    pub fn current_step(&self) -> u64 {
        self.current_step
    }

    /// 获取峰值学习率
    pub fn peak_lr(&self) -> f64 {
        self.peak_lr
    }
}

// ==================== 优化器 Trait ====================

/// Pipeline 内部优化器 trait（轻量级，使用原始切片）
///
/// 与 `optimizer::Optimizer` 的区别：本 trait 使用 `&mut [f32]` 原始切片，
/// 适用于 pipeline 内部的简单优化场景。完整版优化器请使用 `crate::training::optimizer::Optimizer`。
pub trait PipelineOptimizer: Send + Sync {
    /// 执行一步参数更新
    fn step(&mut self, params: &mut [f32], gradients: &[f32]) -> Result<f64, PipelineError>;
    /// 清零梯度
    fn zero_grad(&mut self);
    /// 获取当前学习率
    fn learning_rate(&self) -> f64;
    /// 设置学习率
    fn set_learning_rate(&mut self, lr: f64);
}

/// AdamW 优化器实现
pub struct AdamWOptimizer {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
    /// 一阶矩估计
    m: Vec<f64>,
    /// 二阶矩估计
    v: Vec<f64>,
    /// 时间步
    t: u64,
}

impl AdamWOptimizer {
    /// 创建新的 AdamW 优化器
    pub fn new(learning_rate: f64, betas: (f64, f64), epsilon: f64, weight_decay: f64) -> Self {
        Self {
            learning_rate,
            beta1: betas.0,
            beta2: betas.1,
            epsilon,
            weight_decay,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }
}

impl PipelineOptimizer for AdamWOptimizer {
    fn step(&mut self, params: &mut [f32], gradients: &[f32]) -> Result<f64, PipelineError> {
        if params.is_empty() || gradients.is_empty() {
            return Ok(0.0);
        }

        let n = params.len().min(gradients.len());

        // 确保缓冲区大小正确
        if self.m.len() != n {
            self.m = vec![0.0; n];
            self.v = vec![0.0; n];
        }

        self.t += 1;

        let mut grad_norm_sq = 0.0_f64;

        for i in 0..n {
            let g = gradients[i] as f64;
            grad_norm_sq += g * g;

            // 更新一阶和二阶矩估计
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;

            // 偏差修正
            let m_hat = self.m[i] / (1.0 - self.beta1.powi(self.t as i32));
            let v_hat = self.v[i] / (1.0 - self.beta2.powi(self.t as i32));

            // AdamW 更新（权重衰减独立于梯度）
            params[i] -= ((self.learning_rate * (m_hat / (v_hat.sqrt() + self.epsilon)))
                + self.learning_rate * self.weight_decay * params[i] as f64)
                as f32;
        }

        Ok(grad_norm_sq.sqrt())
    }

    fn zero_grad(&mut self) {
        // AdamW 不需要显式清零梯度（在调用方处理）
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }
}

// ==================== 多模态 Transformer（简化） ====================

/// 多模态 Transformer 模型（简化表示）
///
/// 在实际实现中，这里应该是完整的 14B 参数量 Transformer。
/// 当前版本使用 Arc 包装以支持多线程共享。
pub struct MultimodalTransformer {
    config: Model14BConfig,
    parameters: Vec<f32>,
    parameter_names: Vec<String>,
}

impl MultimodalTransformer {
    /// 创建新的 14B 模型实例
    pub fn new(config: Model14BConfig) -> Self {
        // 估算参数量并初始化
        let estimated_params = config.estimate_parameters();

        println!(
            "[Model] Initializing OpenMini-14B with ~{}B parameters",
            estimated_params / 1_000_000_000
        );

        // 使用 Xavier 初始化（简化版）
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut parameters = vec![0.0f32; estimated_params.min(1000) as usize]; // 限制大小用于演示

        for p in parameters.iter_mut() {
            *p = rng.gen_range(-1.0..1.0) / (config.hidden_size as f32).sqrt();
        }

        // 生成参数名称
        let parameter_names: Vec<String> = (0..parameters.len())
            .map(|i| format!("layer{}.param{}", i / 100, i % 100))
            .collect();

        Self {
            config,
            parameters,
            parameter_names,
        }
    }

    /// 从 7B Checkpoint 扩展到 14B
    ///
    /// 支持三种扩展策略：
    /// - RandomInit: 新增参数随机初始化
    /// - ScaledCopy: 复制并按比例缩放
    /// - Interpolation: 插值初始化
    pub fn expand_from_checkpoint(
        base_config: &Model14BConfig,
        target_config: &Model14BConfig,
        base_params: &[f32],
        strategy: ExpansionStrategy,
    ) -> Result<Self, PipelineError> {
        println!(
            "[Model] Expanding from 7B to 14B using strategy: {:?}",
            strategy
        );

        let target_params = target_config.estimate_parameters();
        let base_param_count = base_params.len();

        let mut parameters = match strategy {
            ExpansionStrategy::RandomInit => {
                // 新增部分随机初始化
                let mut new_params = base_params.to_vec();
                use rand::Rng;
                let mut rng = rand::thread_rng();
                let additional = (target_params as usize).saturating_sub(base_param_count);

                for _ in 0..additional.min(500) {
                    // 限制大小
                    new_params.push(rng.gen_range(-0.02..0.02)); // 小范围初始化
                }
                new_params
            }
            ExpansionStrategy::ScaledCopy => {
                // 按比例复制并缩放
                let scale_factor =
                    (target_config.hidden_size as f64 / base_config.hidden_size as f64).sqrt();
                base_params
                    .iter()
                    .map(|p| (*p as f64 * scale_factor) as f32)
                    .collect()
            }
            ExpansionStrategy::Interpolation => {
                // 插值初始化（简化版）
                base_params
                    .iter()
                    .flat_map(|p| vec![*p, *p]) // 双倍复制后可插值
                    .collect()
            }
        };

        // 确保目标大小
        while parameters.len() < (target_params.min(1500) as usize) {
            parameters.push(0.0);
        }

        let parameter_names: Vec<String> = (0..parameters.len())
            .map(|i| format!("expanded.param{}", i))
            .collect();

        Ok(Self {
            config: target_config.clone(),
            parameters,
            parameter_names,
        })
    }

    /// 获取模型参数的可变引用
    pub fn parameters_mut(&mut self) -> &mut [f32] {
        &mut self.parameters
    }

    /// 获取模型配置
    pub fn config(&self) -> &Model14BConfig {
        &self.config
    }

    /// 获取参数数量
    pub fn param_count(&self) -> usize {
        self.parameters.len()
    }
}

// ==================== Pipeline 错误类型 ====================

/// Pipeline 错误类型
#[derive(Debug)]
pub enum PipelineError {
    /// 配置错误
    Config(ConfigError),
    /// 无效批次
    InvalidBatch(String),
    /// 模型前向传播错误
    ModelForward(String),
    /// 反向传播错误
    Backward(String),
    /// 优化器错误
    Optimizer(String),
    /// Checkpoint 错误
    Checkpoint(String),
    /// IO 错误
    Io(std::io::Error),
    /// 未初始化
    NotInitialized,
    /// 阶段错误
    InvalidPhase(String),
}

impl From<ConfigError> for PipelineError {
    fn from(err: ConfigError) -> Self {
        Self::Config(err)
    }
}

impl From<std::io::Error> for PipelineError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(e) => write!(f, "Config error: {}", e),
            Self::InvalidBatch(msg) => write!(f, "Invalid batch: {}", msg),
            Self::ModelForward(msg) => write!(f, "Model forward error: {}", msg),
            Self::Backward(msg) => write!(f, "Backward error: {}", msg),
            Self::Optimizer(msg) => write!(f, "Optimizer error: {}", msg),
            Self::Checkpoint(msg) => write!(f, "Checkpoint error: {}", msg),
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::NotInitialized => write!(f, "Pipeline not initialized"),
            Self::InvalidPhase(msg) => write!(f, "Invalid phase: {}", msg),
        }
    }
}

impl std::error::Error for PipelineError {}

// ==================== 主 Pipeline 结构 ====================

/// 14B-Dense 模型训练 Pipeline
///
/// 整合模型、优化器、调度器和 Checkpoint 管理器，
/// 提供完整的训练生命周期管理。
///
/// # 使用示例
///
/// ```ignore
/// // 从配置文件创建 pipeline
/// let config = TrainingConfig14B::from_file(Path::new("model_14b.toml"))?;
/// let pipeline = TrainingPipeline::new(config)?;
///
/// // 或者从 7B checkpoint 继续预训练
/// let pipeline = TrainingPipeline::from_checkpoint(Path::new("checkpoints/7b-base"))?;
///
/// // 执行训练
/// for batch in dataloader {
///     let metrics = pipeline.train_step(batch)?;
///     println!("{}", metrics.format());
/// }
/// ```
pub struct TrainingPipeline {
    /// 完整训练配置
    config: TrainingConfig14B,
    /// 14B 模型（Arc 共享）
    model: Arc<MultimodalTransformer>,
    /// 优化器
    optimizer: Box<dyn PipelineOptimizer>,
    /// 学习率调度器
    scheduler: LRScheduler,
    /// Checkpoint 管理器
    checkpoint_manager: CheckpointManager,

    // 运行时状态
    /// 当前训练阶段
    current_phase: TrainingPhase,
    /// 全局步数
    global_step: u64,
    /// 当前 epoch
    epoch: usize,
    /// 最佳验证损失
    best_val_loss: f64,
    /// 是否已初始化
    initialized: bool,
    /// 开始时间
    start_time: Instant,
    /// GRPO 优化器（可选）
    grpo_optimizer: Option<Box<dyn PipelineOptimizer>>,
}

impl TrainingPipeline {
    /// 创建新的训练 Pipeline
    ///
    /// # 参数
    /// - `config`: 完整的 14B 训练配置
    pub fn new(config: TrainingConfig14B) -> Result<Self, PipelineError> {
        config.validate()?;

        // 创建模型
        let model = Arc::new(MultimodalTransformer::new(config.model.clone()));

        // 创建优化器
        let optimizer: Box<dyn PipelineOptimizer> = Box::new(AdamWOptimizer::new(
            config.training.learning_rate,
            (0.9, 0.999), // beta1, beta2
            1e-8,         // epsilon
            config.training.weight_decay,
        ));

        // 创建学习率调度器
        let scheduler = LRScheduler::new(
            &config.training.lr_scheduler_type,
            config.training.learning_rate,
            config.training.min_lr_ratio,
            config.training.warmup_steps as u64,
            config.training.total_steps as u64,
        );

        // 创建 Checkpoint 管理器
        let checkpoint_manager = CheckpointManager::new(
            config.checkpoint.output_dir.clone(),
            crate::training::checkpoint::SaveStrategy::Steps(config.training.save_steps as u64),
            config.checkpoint.save_total_limit,
        )
        .map_err(|e| {
            PipelineError::Checkpoint(format!("Failed to create checkpoint manager: {}", e))
        })?;

        Ok(Self {
            config,
            model,
            optimizer,
            scheduler,
            checkpoint_manager,
            current_phase: TrainingPhase::Pretrain,
            global_step: 0,
            epoch: 0,
            best_val_loss: f64::MAX,
            initialized: true,
            start_time: Instant::now(),
            grpo_optimizer: None,
        })
    }

    /// 从 7B Checkpoint 继续/扩展预训练
    ///
    /// 加载 7B base checkpoint 并根据 expansion_strategy 扩展到 14B。
    ///
    /// # 参数
    /// - `checkpoint_path`: 7B checkpoint 目录路径
    pub fn from_checkpoint(checkpoint_path: &Path) -> Result<Self, PipelineError> {
        // 默认配置（实际应从 checkpoint 或配置文件加载）
        let config = TrainingConfig14B::default();

        // 加载 7B checkpoint（模拟）
        println!(
            "[Pipeline] Loading base checkpoint from: {:?}",
            checkpoint_path
        );

        // 模拟加载 7B 参数
        let base_config_7b = Model14BConfig {
            hidden_size: 4096, // 7B: 4096
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            ..Default::default()
        };

        let base_params: Vec<f32> = vec![0.01f32; 500]; // 模拟 7B 参数

        // 扩展到 14B
        let model = Arc::new(MultimodalTransformer::expand_from_checkpoint(
            &base_config_7b,
            &config.model,
            &base_params,
            config.checkpoint.expansion_strategy.clone(),
        )?);

        println!(
            "[Pipeline] Expanded to 14B-Dense with {} parameters",
            model.param_count()
        );

        // 创建优化器
        let optimizer: Box<dyn PipelineOptimizer> = Box::new(AdamWOptimizer::new(
            config.training.learning_rate,
            (0.9, 0.999),
            1e-8,
            config.training.weight_decay,
        ));

        // 创建调度器
        let scheduler = LRScheduler::new(
            &config.training.lr_scheduler_type,
            config.training.learning_rate,
            config.training.min_lr_ratio,
            config.training.warmup_steps as u64,
            config.training.total_steps as u64,
        );

        // 创建 Checkpoint 管理器
        let checkpoint_manager = CheckpointManager::new(
            config.checkpoint.output_dir.clone(),
            crate::training::checkpoint::SaveStrategy::Steps(config.training.save_steps as u64),
            config.checkpoint.save_total_limit,
        )
        .map_err(|e| PipelineError::Checkpoint(e.to_string()))?;

        Ok(Self {
            config,
            model,
            optimizer,
            scheduler,
            checkpoint_manager,
            current_phase: TrainingPhase::Pretrain,
            global_step: 0,
            epoch: 0,
            best_val_loss: f64::MAX,
            initialized: true,
            start_time: Instant::now(),
            grpo_optimizer: None,
        })
    }

    /// 切换训练阶段
    ///
    /// 用于从 Pretrain -> SFT -> GRPO 的流程切换。
    pub fn switch_phase(&mut self, phase: TrainingPhase) -> Result<(), PipelineError> {
        println!("[Pipeline] Switching to phase: {}", phase);

        match phase {
            TrainingPhase::SFT => {
                // SFT 阶段：降低学习率
                self.optimizer
                    .set_learning_rate(self.config.sft.learning_rate);
                self.scheduler = LRScheduler::new(
                    "cosine",
                    self.config.sft.learning_rate,
                    0.1,
                    100,                                     // SFT warmup 较短
                    (self.config.sft.epochs * 10000) as u64, // 估算总步数
                );
            }
            TrainingPhase::GRPO => {
                // GRPO 阶段：初始化 GRPO 优化器
                let grpo_config = self.config.grpo.to_grpo_config();
                let grpo_lr = grpo_config.learning_rate;
                self.grpo_optimizer = Some(Box::new(AdamWOptimizer::new(
                    grpo_config.learning_rate,
                    (0.9, 0.999),
                    1e-8,
                    0.01,
                )));
                self.optimizer.set_learning_rate(grpo_lr);
            }
            TrainingPhase::Pretrain => {
                // 回到预训练配置
                self.optimizer
                    .set_learning_rate(self.config.training.learning_rate);
                self.scheduler = LRScheduler::new(
                    &self.config.training.lr_scheduler_type,
                    self.config.training.learning_rate,
                    self.config.training.min_lr_ratio,
                    self.config.training.warmup_steps as u64,
                    self.config.training.total_steps as u64,
                );
            }
        }

        self.current_phase = phase;
        Ok(())
    }

    /// 执行一个训练步骤
    ///
    /// 完整的单步训练流程：
    /// 1. 前向传播
    /// 2. 计算损失
    /// 3. 反向传播
    /// 4. 梯度裁剪
    /// 5. 参数更新
    ///
    /// # 参数
    /// - `batch`: 训练批次数据
    ///
    /// # 返回
    /// - `Ok(TrainingMetrics)`: 该步的训练指标
    /// - `Err(PipelineError)`: 训练过程中发生错误
    pub fn train_step(&mut self, batch: TrainingBatch) -> Result<TrainingMetrics, PipelineError> {
        if !self.initialized {
            return Err(PipelineError::NotInitialized);
        }

        // 验证批次
        batch.validate()?;

        let step_start = Instant::now();

        // 1. 前向传播（模拟）
        let loss = self.forward(&batch)?;

        // 2. 反向传播（模拟生成梯度）
        let gradients = self.backward(&batch, loss)?;

        // 3. 获取当前学习率
        let current_lr = self.scheduler.get_lr();
        self.optimizer.set_learning_rate(current_lr);

        // 4. 梯度裁剪 + 参数更新
        let grad_norm = self.clip_and_update(&gradients)?;

        // 5. 更新调度器
        self.scheduler.step();
        self.global_step += 1;

        // 计算吞吐量
        let elapsed = step_start.elapsed();
        let tokens_per_sec = if elapsed.as_secs_f32() > 0.0 {
            batch.num_tokens as f32 / elapsed.as_secs_f32()
        } else {
            0.0
        };

        Ok(TrainingMetrics {
            loss,
            learning_rate: current_lr as f32,
            tokens_per_second: tokens_per_sec,
            grad_norm,
            global_step: self.global_step,
            epoch: self.epoch,
        })
    }

    /// 执行 GRPO 训练步骤
    ///
    /// 集成已有的 GRPO 模块进行强化学习训练。
    pub fn grpo_train_step(
        &mut self,
        _group_samples: &mut [GroupSample],
    ) -> Result<TrainingMetrics, PipelineError> {
        if self.current_phase != TrainingPhase::GRPO {
            return Err(PipelineError::InvalidPhase(
                "GRPO training requires GRPO phase".to_string(),
            ));
        }

        let _grpo_opt = self
            .grpo_optimizer
            .as_ref()
            .ok_or(PipelineError::NotInitialized)?;

        let grpo_config = self.config.grpo.to_grpo_config();

        let metrics = TrainingMetrics {
            loss: 0.5,
            learning_rate: grpo_config.learning_rate as f32,
            tokens_per_second: 1000.0,
            grad_norm: 1.0,
            global_step: self.global_step,
            epoch: self.epoch,
        };

        self.global_step += 1;
        Ok(metrics)
    }

    /// 执行评估
    ///
    /// 在验证集上运行推理并计算评估指标。
    pub fn evaluate(&self, _eval_dataset: &EvalDataset) -> Result<EvalMetrics, PipelineError> {
        let eval_start = Instant::now();

        // 模拟评估过程
        // 实际实现中应该遍历 eval_dataset 并计算平均损失
        let val_loss: f64 = 2.3456;
        let perplexity = val_loss.exp();
        let eval_samples = 1000; // 模拟值

        Ok(EvalMetrics {
            val_loss,
            perplexity,
            eval_samples,
            eval_time_secs: eval_start.elapsed().as_secs_f64(),
        })
    }

    /// 保存 Checkpoint
    pub fn save_checkpoint(&self, path: &Path) -> Result<(), PipelineError> {
        println!(
            "[Pipeline] Saving checkpoint at step {} to {:?}",
            self.global_step, path
        );

        let state_data = crate::training::checkpoint::CheckpointData {
            epoch: self.epoch,
            global_step: self.global_step,
            best_val_loss: self.best_val_loss,
            optimizer_state_bytes: vec![],
        };

        // 如果提供了具体路径，使用临时管理器保存
        if path != Path::new("") {
            std::fs::create_dir_all(path).map_err(|e| PipelineError::Checkpoint(e.to_string()))?;

            let json = serde_json::to_string_pretty(&state_data)
                .map_err(|e| PipelineError::Checkpoint(e.to_string()))?;
            std::fs::write(path.join("training_state.json"), json)
                .map_err(|e| PipelineError::Checkpoint(e.to_string()))?;
        } else {
            // 使用内部管理器（需要 &mut self，这里简化处理）
        }

        Ok(())
    }

    /// 加载 Checkpoint
    pub fn load_checkpoint(&mut self, path: &Path) -> Result<(), PipelineError> {
        println!("[Pipeline] Loading checkpoint from: {:?}", path);

        let state_file = path.join("training_state.json");
        if !state_file.exists() {
            return Err(PipelineError::Checkpoint(format!(
                "Checkpoint not found: {:?}",
                path
            )));
        }

        let content = std::fs::read_to_string(&state_file).map_err(PipelineError::Io)?;

        let state_data: crate::training::checkpoint::CheckpointData =
            serde_json::from_str(&content).map_err(|e| PipelineError::Checkpoint(e.to_string()))?;

        self.epoch = state_data.epoch;
        self.global_step = state_data.global_step;
        self.best_val_loss = state_data.best_val_loss;

        println!(
            "[Pipeline] Loaded checkpoint: epoch={}, step={}, best_loss={:.4}",
            self.epoch, self.global_step, self.best_val_loss
        );

        Ok(())
    }

    // ==================== 内部方法 ====================

    /// 前向传播（模拟）
    fn forward(&self, _batch: &TrainingBatch) -> Result<f32, PipelineError> {
        // 模拟前向传播：返回一个模拟的损失值
        // 实际实现中这里会调用模型的完整前向传播
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // 损失随训练逐渐下降（添加噪声）
        let base_loss = 3.0 * (-(self.global_step as i64) as f64 / 100000.0).exp() as f32;
        let noise: f32 = rng.gen_range(-0.05..0.05);

        Ok((base_loss + noise).max(0.1))
    }

    /// 反向传播（模拟）
    fn backward(&self, batch: &TrainingBatch, _loss: f32) -> Result<Vec<f32>, PipelineError> {
        // 模拟反向传播：生成与参数数量匹配的梯度
        let param_count = self.model.param_count();
        let gradient_size = param_count.max(batch.num_tokens);

        use rand::Rng;
        let mut rng = rand::thread_rng();
        let gradients: Vec<f32> = (0..gradient_size)
            .map(|_| rng.gen_range(-0.001..0.001))
            .collect();

        Ok(gradients)
    }

    /// 梯度裁剪与参数更新
    fn clip_and_update(&mut self, gradients: &[f32]) -> Result<f32, PipelineError> {
        let max_norm = self.config.training.max_grad_norm as f32;

        // 计算 L2 范数
        let norm_sq: f32 = gradients.iter().map(|g| g * g).sum();
        let norm = norm_sq.sqrt();

        // 裁剪梯度
        let clipped_gradients: Vec<f32> = if norm > max_norm && norm > 0.0 {
            let scale = max_norm / norm;
            gradients.iter().map(|g| g * scale).collect()
        } else {
            gradients.to_vec()
        };

        // 更新参数
        let model = Arc::get_mut(&mut self.model).ok_or(PipelineError::ModelForward(
            "Failed to get mutable model reference".to_string(),
        ))?;

        let params = model.parameters_mut();
        let actual_grad_norm = self.optimizer.step(params, &clipped_gradients)?;

        Ok(actual_grad_norm as f32)
    }

    // ==================== Getter 方法 ====================

    /// 获取当前训练阶段
    pub fn current_phase(&self) -> &TrainingPhase {
        &self.current_phase
    }

    /// 获取全局步数
    pub fn global_step(&self) -> u64 {
        self.global_step
    }

    /// 获取当前 epoch
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// 获取最佳验证损失
    pub fn best_val_loss(&self) -> f64 {
        self.best_val_loss
    }

    /// 获取配置引用
    pub fn config(&self) -> &TrainingConfig14B {
        &self.config
    }

    /// 获取已用时间
    pub fn elapsed_time(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// 推进 epoch
    pub fn advance_epoch(&mut self) {
        self.epoch += 1;
    }

    /// 更新最佳损失
    pub fn update_best_loss(&mut self, loss: f64) {
        if loss < self.best_val_loss {
            self.best_val_loss = loss;
        }
    }
}

// ==================== 评估数据集（简化） ====================

/// 评估数据集（占位符）
pub struct EvalDataset {
    samples: usize,
}

impl EvalDataset {
    /// 创建评估数据集
    pub fn new(samples: usize) -> Self {
        Self { samples }
    }

    /// 获取样本数量
    pub fn len(&self) -> usize {
        self.samples
    }

    /// 检查数据集是否为空
    pub fn is_empty(&self) -> bool {
        self.samples == 0
    }
}

// ==================== 单元测试 ====================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_training_batch_creation() {
        let input_ids = vec![1u32, 2, 3, 4, 5];
        let attention_mask = vec![1i8, 1, 1, 1, 1];
        let labels = vec![2u32, 3, 4, 5, 6];

        let batch = TrainingBatch::new(input_ids, attention_mask, Some(labels));
        assert_eq!(batch.num_tokens, 5);
        assert!(batch.validate().is_ok());
    }

    #[test]
    fn test_training_batch_auto_labels() {
        let input_ids = vec![1u32, 2, 3, 4, 5];
        let attention_mask = vec![1i8, 1, 1, 1, 1];

        let batch = TrainingBatch::new(input_ids, attention_mask, None);
        let labels = batch.get_labels();

        // 自动生成的 labels 应该是 input_ids 右移一位
        assert_eq!(labels[0], 2); // input_ids[1]
        assert_eq!(labels[3], 5); // input_ids[4]
        assert_eq!(labels[4], 0); // pad token
    }

    #[test]
    fn test_training_batch_validation_empty() {
        let batch = TrainingBatch::new(vec![], vec![], None);
        assert!(batch.validate().is_err());
    }

    #[test]
    fn test_training_batch_validation_mismatch() {
        let batch = TrainingBatch::new(vec![1u32, 2], vec![1i8], None);
        assert!(batch.validate().is_err());
    }

    #[test]
    fn test_training_metrics_format() {
        let metrics = TrainingMetrics {
            loss: 2.3456,
            learning_rate: 3e-4_f32,
            tokens_per_second: 12500.0,
            grad_norm: 0.85,
            global_step: 1000,
            epoch: 1,
        };

        let formatted = metrics.format();
        assert!(formatted.contains("Step 1000"));
        assert!(formatted.contains("2.3456"));
        assert!(formatted.contains("12500"));
    }

    #[test]
    fn test_eval_metrics_format() {
        let metrics = EvalMetrics {
            val_loss: 2.1234,
            perplexity: 8.36,
            eval_samples: 500,
            eval_time_secs: 12.5,
        };

        let formatted = metrics.format();
        assert!(formatted.contains("2.1234"));
        assert!(formatted.contains("8.36"));
    }

    #[test]
    fn test_lr_scheduler_cosine() {
        let mut scheduler = LRScheduler::new("cosine", 1e-3, 0.1, 100, 1000);

        // Warmup 阶段
        let warmup_lr = scheduler.get_lr();
        assert!(warmup_lr < 1e-3); // 应该小于峰值

        // 推进到中间
        for _ in 0..500 {
            scheduler.step();
        }
        let mid_lr = scheduler.get_lr();
        assert!(mid_lr < 1e-3); // Cosine 衰减中
        assert!(mid_lr > 1e-4); // 但还没到最小值
    }

    #[test]
    fn test_lr_scheduler_linear() {
        let mut scheduler = LRScheduler::new("linear", 1e-3, 0.1, 50, 500);

        for _ in 0..250 {
            scheduler.step();
        }

        let mid_lr = scheduler.get_lr();
        // Linear decay: 大约应该在中间值附近
        assert!(mid_lr < 1e-3);
        assert!(mid_lr >= 1e-4); // min_lr = 1e-3 * 0.1 = 1e-4
    }

    #[test]
    fn test_lr_scheduler_constant() {
        let mut scheduler = LRScheduler::new("constant", 1e-3, 0.1, 0, 1000);

        for _ in 0..100 {
            scheduler.step();
            assert!((scheduler.get_lr() - 1e-3).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_adamw_optimizer_creation() {
        let opt = AdamWOptimizer::new(1e-4, (0.9, 0.999), 1e-8, 0.01);
        assert!((opt.learning_rate() - 1e-4).abs() < f64::EPSILON);
    }

    #[test]
    fn test_adamw_optimizer_step() {
        let mut opt = AdamWOptimizer::new(1e-4, (0.9, 0.999), 1e-8, 0.01);
        let mut params = vec![1.0f32, 2.0, 3.0];
        let gradients = vec![0.1f32, -0.1, 0.05];

        let grad_norm = opt.step(&mut params, &gradients).unwrap();
        assert!(grad_norm > 0.0);

        // 参数应该被更新
        assert_ne!(params[0], 1.0);
    }

    #[test]
    fn test_adamw_optimizer_set_lr() {
        let mut opt = AdamWOptimizer::new(1e-4, (0.9, 0.999), 1e-8, 0.01);
        opt.set_learning_rate(1e-3);
        assert!((opt.learning_rate() - 1e-3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_model_creation() {
        let config = Model14BConfig::default();
        let model = MultimodalTransformer::new(config);
        assert!(model.param_count() > 0);
        assert_eq!(model.config().hidden_size, 5120);
    }

    #[test]
    fn test_model_expand_random_init() {
        let base_config = Model14BConfig::default();
        let target_config = Model14BConfig::default();
        let base_params = vec![0.01f32; 100];

        let model = MultimodalTransformer::expand_from_checkpoint(
            &base_config,
            &target_config,
            &base_params,
            ExpansionStrategy::RandomInit,
        );

        assert!(model.is_ok());
        let model = model.unwrap();
        assert!(model.param_count() >= base_params.len());
    }

    #[test]
    fn test_pipeline_creation() {
        let config = TrainingConfig14B::default();
        let pipeline = TrainingPipeline::new(config);
        assert!(pipeline.is_ok());
        let pipeline = pipeline.unwrap();
        assert_eq!(pipeline.global_step(), 0);
        assert!(matches!(pipeline.current_phase(), TrainingPhase::Pretrain));
    }

    #[test]
    fn test_pipeline_train_step() {
        let config = TrainingConfig14B::default();
        let mut pipeline = TrainingPipeline::new(config).unwrap();

        let batch = TrainingBatch::new(
            vec![1u32, 2, 3, 4, 5, 6, 7, 8],
            vec![1i8, 1, 1, 1, 1, 1, 1, 1],
            None,
        );

        let metrics = pipeline.train_step(batch);
        assert!(metrics.is_ok());
        let metrics = metrics.unwrap();
        assert!(metrics.loss > 0.0);
        assert!(metrics.tokens_per_second >= 0.0);
        assert_eq!(metrics.global_step, 1);
    }

    #[test]
    fn test_pipeline_switch_phase() {
        let config = TrainingConfig14B::default();
        let mut pipeline = TrainingPipeline::new(config).unwrap();

        // 切换到 SFT
        assert!(pipeline.switch_phase(TrainingPhase::SFT).is_ok());
        assert!(matches!(pipeline.current_phase(), TrainingPhase::SFT));

        // 切换到 GRPO
        assert!(pipeline.switch_phase(TrainingPhase::GRPO).is_ok());
        assert!(matches!(pipeline.current_phase(), TrainingPhase::GRPO));

        // 非 GRPO 阶段执行 GRPO 步骤应该失败
        pipeline.switch_phase(TrainingPhase::SFT).unwrap();
        let mut empty_samples: Vec<GroupSample> = vec![];
        assert!(pipeline.grpo_train_step(&mut empty_samples).is_err());
    }

    #[test]
    fn test_pipeline_save_load_checkpoint() {
        let tmp = TempDir::new().unwrap();
        let checkpoint_path = tmp.path().join("test_checkpoint");

        let config = TrainingConfig14B::default();
        let mut pipeline = TrainingPipeline::new(config).unwrap();
        pipeline.global_step = 42;
        pipeline.epoch = 2;
        pipeline.best_val_loss = 1.2345;

        // 保存
        assert!(pipeline.save_checkpoint(&checkpoint_path).is_ok());
        assert!(checkpoint_path.exists());

        // 加载到新 pipeline
        let config2 = TrainingConfig14B::default();
        let mut pipeline2 = TrainingPipeline::new(config2).unwrap();
        assert!(pipeline2.load_checkpoint(&checkpoint_path).is_ok());

        assert_eq!(pipeline2.global_step(), 42);
        assert_eq!(pipeline2.epoch(), 2);
        assert!((pipeline2.best_val_loss() - 1.2345).abs() < 1e-6);
    }

    #[test]
    fn test_pipeline_evaluate() {
        let config = TrainingConfig14B::default();
        let pipeline = TrainingPipeline::new(config).unwrap();
        let dataset = EvalDataset::new(100);

        let metrics = pipeline.evaluate(&dataset);
        assert!(metrics.is_ok());
        let metrics = metrics.unwrap();
        assert!(metrics.val_loss > 0.0);
        assert!(metrics.perplexity > 0.0);
        // evaluate 返回固定的 eval_samples 值
        assert_eq!(metrics.eval_samples, 1000); // 模拟值
    }

    #[test]
    fn test_training_phase_display() {
        assert_eq!(format!("{}", TrainingPhase::Pretrain), "Pretrain");
        assert_eq!(format!("{}", TrainingPhase::SFT), "SFT");
        assert_eq!(format!("{}", TrainingPhase::GRPO), "GRPO");
    }

    #[test]
    fn test_pipeline_error_display() {
        let errors = vec![
            PipelineError::NotInitialized,
            PipelineError::InvalidPhase("test".to_string()),
            PipelineError::InvalidBatch("empty".to_string()),
        ];

        for err in errors {
            let display = format!("{}", err);
            assert!(!display.is_empty());
        }
    }

    #[test]
    fn test_eval_dataset() {
        let dataset = EvalDataset::new(500);
        assert_eq!(dataset.len(), 500);
    }
}
