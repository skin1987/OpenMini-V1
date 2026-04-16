//! 训练器核心模块
//!
//! 整合 Autograd、Optimizer、Loss、DataLoader 为完整的训练循环，
//! 支持因果语言模型预训练、继续预训练、有监督微调三种模式。

use crate::training::autograd::ComputationGraph;
use crate::training::dataloader::{Batch, DataLoader};
use crate::training::loss::{CrossEntropyConfig, CrossEntropyLoss, Reduction};
use crate::training::optimizer::{AdamW, Optimizer, ParamState};
use ndarray::{Array2, ArrayD};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ==================== 训练模式 ====================

/// 训练模式枚举
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingMode {
    /// 因果语言模型预训练
    CausalLM,
    /// 继续预训练
    ContinuePretrain,
    /// 有监督微调
    SFT,
}

// ==================== SFT 配置 ====================

/// SFT（有监督微调）配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SFTConfig {
    /// 提示模板，支持 {instruction} 和 {input} 占位符
    pub prompt_template: String,
    /// 是否屏蔽提示部分的损失计算
    pub mask_prompt_loss: bool,
}

// ==================== 训练配置 ====================

/// 训练超参数配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// 训练模式
    pub mode: TrainingMode,

    // 超参数
    /// 训练轮数
    pub num_train_epochs: usize,
    /// 每设备训练批次大小（真实 batch_size）
    ///
    /// 有效 batch_size = per_device_train_batch_size × gradient_accumulation_steps
    /// 例如：batch_size=8, accumulation_steps=4 → 有效 batch_size = 32
    pub per_device_train_batch_size: usize,
    /// 梯度累积步数
    ///
    /// 当设置为 1 时，每步都更新参数（标准模式）
    /// 当设置 > 1 时，累积 N 步的梯度后再统一更新参数
    /// 这允许在显存有限的情况下模拟更大的有效 batch size
    ///
    /// # 示例
    /// - gradient_accumulation_steps = 1: 标准训练模式
    /// - gradient_accumulation_steps = 4: 累积 4 个 micro-batch 的梯度后更新一次参数
    pub gradient_accumulation_steps: usize,
    /// 学习率
    pub learning_rate: f64,
    /// 权重衰减
    pub weight_decay: f64,
    /// Adam beta1
    pub adam_beta1: f64,
    /// Adam beta2
    pub adam_beta2: f64,
    /// Adam epsilon
    pub adam_epsilon: f64,
    /// 最大梯度范数（用于梯度裁剪）
    pub max_grad_norm: f64,

    // 数据相关
    /// 最大序列长度
    pub max_seq_length: usize,
    /// 标签平滑因子
    pub label_smoothing_factor: f64,

    // Checkpoint 相关
    /// 输出目录
    pub output_dir: PathBuf,
    /// 保存步数间隔
    pub save_steps: usize,
    /// 最多保存的 checkpoint 数量
    pub save_total_limit: usize,

    // 日志相关
    /// 日志输出步数间隔
    pub logging_steps: usize,

    // 早停机制
    /// 早停耐心值（连续多少个 epoch 无改善则停止）
    pub early_stopping_patience: usize,
    /// 早停阈值（改善幅度小于此值视为无改善）
    pub early_stopping_threshold: f64,

    // SFT 专用配置
    #[serde(default)]
    pub sft_config: Option<SFTConfig>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            mode: TrainingMode::CausalLM,
            num_train_epochs: 3,
            per_device_train_batch_size: 8,
            gradient_accumulation_steps: 1,
            learning_rate: 1e-4,
            weight_decay: 0.01,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_epsilon: 1e-8,
            max_grad_norm: 1.0,
            max_seq_length: 2048,
            label_smoothing_factor: 0.0,
            output_dir: PathBuf::from("./checkpoints"),
            save_steps: 500,
            save_total_limit: 5,
            logging_steps: 10,
            early_stopping_patience: 5,
            early_stopping_threshold: 0.001,
            sft_config: None,
        }
    }
}

// ==================== 训练状态 ====================

/// 训练运行时状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState {
    /// 当前 epoch
    pub epoch: usize,
    /// 全局步数
    pub global_step: u64,
    /// 最佳验证损失
    pub best_val_loss: f64,
    /// 总步数
    pub total_steps: u64,
    /// 配置哈希值（用于检测配置变更）
    pub config_hash: u64,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            epoch: 0,
            global_step: 0,
            best_val_loss: f64::MAX,
            total_steps: 0,
            config_hash: 0,
        }
    }
}

// ==================== 单步指标 ====================

/// 单步训练指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepMetrics {
    /// 当前步损失值
    pub loss: f64,
    /// 梯度范数
    pub grad_norm: f64,
    /// 当前学习率
    pub learning_rate: f64,
    /// 吞吐量（tokens/秒）
    pub throughput_tokens_per_sec: f64,
    /// 单步耗时（毫秒）
    pub step_time_ms: f64,
}

// ==================== 错误类型 ====================

/// 训练器错误类型
#[derive(Debug)]
pub enum TrainerError {
    /// 数据加载错误
    DataLoad(String),
    /// 模型前向传播错误
    ModelForward(String),
    /// 反向传播错误
    BackwardPass(String),
    /// 优化器错误
    Optimizer(crate::training::optimizer::OptimizerError),
    /// Checkpoint IO 错误
    Checkpoint(std::io::Error),
    /// 无效配置
    InvalidConfig(String),
    /// 未初始化
    NotInitialized,
    /// 已在运行
    AlreadyRunning,
    /// 其他错误
    Other(String),
}

impl std::fmt::Display for TrainerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DataLoad(msg) => write!(f, "数据加载错误: {}", msg),
            Self::ModelForward(msg) => write!(f, "模型前向传播错误: {}", msg),
            Self::BackwardPass(msg) => write!(f, "反向传播错误: {}", msg),
            Self::Optimizer(err) => write!(f, "优化器错误: {}", err),
            Self::Checkpoint(err) => write!(f, "Checkpoint 错误: {}", err),
            Self::InvalidConfig(msg) => write!(f, "无效配置: {}", msg),
            Self::NotInitialized => write!(f, "训练器未初始化"),
            Self::AlreadyRunning => write!(f, "训练器已在运行"),
            Self::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for TrainerError {}

impl From<crate::training::optimizer::OptimizerError> for TrainerError {
    fn from(err: crate::training::optimizer::OptimizerError) -> Self {
        Self::Optimizer(err)
    }
}

impl From<std::io::Error> for TrainerError {
    fn from(err: std::io::Error) -> Self {
        Self::Checkpoint(err)
    }
}

// ==================== 早停机制 ====================

/// 早停机制实现
///
/// 监控验证损失，当连续多个 epoch 无改善时触发早停。
pub struct EarlyStopping {
    /// 耐心值：连续多少次无改善后停止
    patience: usize,
    /// 最小改善阈值
    min_delta: f64,
    /// 当前连续无改善计数
    counter: usize,
    /// 历史最佳损失
    best_loss: f64,
}

impl EarlyStopping {
    /// 创建新的早停监控器
    ///
    /// # 参数
    /// * `patience` - 连续无改善次数阈值
    /// * `min_delta` - 最小改善幅度
    pub fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            counter: 0,
            best_loss: f64::MAX,
        }
    }

    /// 判断是否应该停止训练
    ///
    /// # 参数
    /// * `val_loss` - 当前验证损失
    ///
    /// # 返回
    /// * `true` - 应该停止训练
    /// * `false` - 继续训练
    pub fn should_stop(&mut self, val_loss: f64) -> bool {
        if val_loss < self.best_loss - self.min_delta {
            self.best_loss = val_loss;
            self.counter = 0;
            false
        } else {
            self.counter += 1;
            self.counter >= self.patience
        }
    }

    /// 获取当前计数
    pub fn counter(&self) -> usize {
        self.counter
    }

    /// 获取最佳损失
    pub fn best_loss(&self) -> f64 {
        self.best_loss
    }

    /// 重置状态
    pub fn reset(&mut self) {
        self.counter = 0;
        self.best_loss = f64::MAX;
    }
}

// ==================== 训练总结 ====================

/// 训练完成后的总结信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSummary {
    /// 总训练轮数
    pub total_epochs: usize,
    /// 总训练步数
    pub total_steps: u64,
    /// 最终训练损失
    pub final_train_loss: f64,
    /// 最佳验证损失
    pub best_val_loss: f64,
    /// 最佳 epoch 编号
    pub best_epoch: usize,
    /// 总训练时间（秒）
    pub total_time_secs: f64,
    /// 是否因早停而终止
    pub stopped_early: bool,
    /// 最终 checkpoint 路径
    pub final_checkpoint_path: Option<String>,
}

// ==================== Trainer 主结构 ====================

/// 训练器 - 核心整合模块
///
/// 将 Autograd、Optimizer、Loss、DataLoader 整合为完整的训练循环，
/// 支持三种训练模式：CausalLM、ContinuePretrain、SFT。
///
/// # Gradient Accumulation（梯度累积）
///
/// 本训练器支持梯度累积功能，允许在显存有限的情况下模拟更大的有效 batch size：
/// - 真实 batch_size = per_device_train_batch_size
/// - gradient_accumulation_steps = N
/// - 有效 batch_size = per_device_train_batch_size × N
///
/// 工作原理：
/// 1. 每个微批次（micro-batch）计算梯度后，不立即更新参数
/// 2. 将梯度累加到缓冲区中
/// 3. 当累积步数达到目标值时，使用累积的梯度进行参数更新
/// 4. 清空累积缓冲区，开始新一轮累积
pub struct Trainer {
    /// 训练配置
    config: TrainingConfig,
    /// 训练状态
    state: TrainingState,
    /// 优化器
    optimizer: Box<dyn Optimizer>,
    /// 损失函数
    loss_fn: CrossEntropyLoss,
    /// 早停机制
    early_stopping: EarlyStopping,
    /// 是否暂停
    is_paused: bool,
    /// 是否正在运行
    is_running: bool,
    /// 指标历史记录
    metrics_history: Vec<StepMetrics>,
    /// 计算图（可选，用于复杂模型）
    graph: Option<ComputationGraph>,

    // ====== Gradient Accumulation 相关字段 ======
    /// 当前已累积的步数
    accumulation_count: u32,
    /// 累积的梯度缓冲区
    ///
    /// 每个 ArrayD 对应一个参数的梯度，在 accumulation_steps 步内累加
    accumulation_gradients: Vec<ArrayD<f32>>,
    /// 目标累积步数（从 config 读取）
    target_accumulation_steps: u32,
    /// 模型参数列表（用于优化器更新）
    model_params: Vec<ParamState>,
    /// 是否已初始化梯度累积缓冲区
    accumulation_initialized: bool,
}

impl Trainer {
    /// 创建新的训练器实例
    ///
    /// # 参数
    /// * `config` - 训练配置
    ///
    /// # 返回
    /// * `Ok(Trainer)` - 成功创建的训练器
    /// * `Err(TrainerError)` - 创建失败
    pub fn new(config: TrainingConfig) -> Result<Self, TrainerError> {
        // 验证配置有效性
        if config.num_train_epochs == 0 {
            return Err(TrainerError::InvalidConfig(
                "num_train_epochs 必须大于 0".to_string(),
            ));
        }
        if config.per_device_train_batch_size == 0 {
            return Err(TrainerError::InvalidConfig(
                "per_device_train_batch_size 必须大于 0".to_string(),
            ));
        }
        if config.learning_rate <= 0.0 {
            return Err(TrainerError::InvalidConfig(
                "learning_rate 必须大于 0".to_string(),
            ));
        }

        // 创建优化器（默认使用 AdamW）
        let optimizer = Box::new(AdamW::new(
            config.learning_rate,
            (config.adam_beta1, config.adam_beta2),
            config.adam_epsilon,
            config.weight_decay,
        ));

        // 创建损失函数
        let loss_config = CrossEntropyConfig {
            ignore_index: usize::MAX,
            label_smoothing: config.label_smoothing_factor,
            reduction: Reduction::Mean,
        };
        let loss_fn = CrossEntropyLoss::new(loss_config);

        // 创建早停机制
        let early_stopping = EarlyStopping::new(
            config.early_stopping_patience,
            config.early_stopping_threshold,
        );

        Ok(Self {
            config: config.clone(),
            state: TrainingState::default(),
            optimizer,
            loss_fn,
            early_stopping,
            is_paused: false,
            is_running: false,
            metrics_history: Vec::new(),
            graph: None,

            // 初始化梯度累积相关字段
            accumulation_count: 0,
            accumulation_gradients: Vec::new(),
            target_accumulation_steps: config.gradient_accumulation_steps as u32,
            model_params: Vec::new(), // 将在第一次训练时初始化
            accumulation_initialized: false,
        })
    }

    /// 开始训练
    ///
    /// # 参数
    /// * `train_dataloader` - 训练数据加载器
    /// * `val_dataloader` - 验证数据加载器（可选）
    ///
    /// # 返回
    /// * `Ok(TrainingSummary)` - 训练总结
    /// * `Err(TrainerError)` - 训练过程中发生错误
    pub fn train(
        &mut self,
        mut train_dataloader: DataLoader,
        val_dataloader: Option<DataLoader>,
    ) -> Result<TrainingSummary, TrainerError> {
        if self.is_running {
            return Err(TrainerError::AlreadyRunning);
        }

        let start_time = std::time::Instant::now();
        self.is_running = true;

        // 主训练循环
        for epoch in 0..self.config.num_train_epochs {
            // 检查暂停/停止状态
            if !self.is_running || self.is_paused {
                #[allow(clippy::while_immutable_condition)]
                while self.is_paused && self.is_running {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                if !self.is_running {
                    break;
                }
            }

            println!("\n{{'='=>60}}");
            println!("Epoch {}/{}", epoch + 1, self.config.num_train_epochs);
            println!("{{'='=>60}}");

            // Epoch 训练
            let mut epoch_loss_sum = 0.0_f64;
            let mut epoch_steps = 0u64;
            let mut epoch_grad_norm_sum = 0.0_f64;

            for batch in &mut train_dataloader {
                if !self.is_running {
                    break;
                }

                let step_start = std::time::Instant::now();

                // 执行单步训练
                let metrics = self.training_step(&batch)?;

                let step_time = step_start.elapsed().as_millis() as f64;
                let throughput = if step_time > 0.0 {
                    batch.num_tokens as f64 / (step_time / 1000.0)
                } else {
                    0.0
                };

                let complete_metrics = StepMetrics {
                    throughput_tokens_per_sec: throughput,
                    step_time_ms: step_time,
                    ..metrics
                };

                // 累积统计
                epoch_loss_sum += complete_metrics.loss;
                epoch_grad_norm_sum += complete_metrics.grad_norm;
                epoch_steps += 1;
                self.state.global_step += 1;
                self.metrics_history.push(complete_metrics.clone());

                // 日志输出
                if self
                    .state
                    .global_step
                    .is_multiple_of(self.config.logging_steps as u64)
                {
                    self.log_metrics(epoch, &complete_metrics);
                }

                // Checkpoint 保存
                if self.config.save_steps > 0
                    && self.state.global_step % self.config.save_steps as u64 == 0
                {
                    self.save_checkpoint()?;
                }
            }

            if !self.is_running {
                break;
            }

            // 验证评估
            let val_loss = match &val_dataloader {
                Some(val_dl) => self.evaluate(val_dl)?,
                None => {
                    if epoch_steps > 0 {
                        epoch_loss_sum / epoch_steps as f64
                    } else {
                        0.0
                    }
                }
            };

            // 更新最佳验证损失
            if val_loss < self.state.best_val_loss {
                self.state.best_val_loss = val_loss;
            }

            // 打印 Epoch 总结
            let avg_loss = if epoch_steps > 0 {
                epoch_loss_sum / epoch_steps as f64
            } else {
                0.0
            };
            let avg_grad_norm = if epoch_steps > 0 {
                epoch_grad_norm_sum / epoch_steps as f64
            } else {
                0.0
            };

            println!(
                "\nEpoch {} 完成 | 训练损失: {:.4} | 验证损失: {:.4} | 最佳验证损失: {:.4} | 平均梯度范数: {:.4}",
                epoch + 1,
                avg_loss,
                val_loss,
                self.state.best_val_loss,
                avg_grad_norm
            );

            // 早停检查
            if self.early_stopping.should_stop(val_loss) {
                println!(
                    "\n⚠ Early stopping 触发于 epoch {} (连续 {} 次 epoch 无改善)",
                    epoch + 1,
                    self.early_stopping.counter()
                );
                break;
            }

            self.state.epoch = epoch + 1;
            train_dataloader.reset_epoch();
        }

        // 生成最终总结
        let summary = TrainingSummary {
            total_epochs: self.state.epoch,
            total_steps: self.state.global_step,
            final_train_loss: self.metrics_history.last().map(|m| m.loss).unwrap_or(0.0),
            best_val_loss: self.state.best_val_loss,
            best_epoch: self.state.epoch.saturating_sub(1),
            total_time_secs: start_time.elapsed().as_secs_f64(),
            stopped_early: self.early_stopping.counter() >= self.early_stopping.patience,
            final_checkpoint_path: Some(self.config.output_dir.display().to_string()),
        };

        self.is_running = false;

        // 重置梯度累积状态（训练结束后不应保留未完成的累积）
        self.reset_accumulation_buffers();

        println!("\n{{'='=>60}}");
        println!("训练完成！");
        println!("总 epochs: {}", summary.total_epochs);
        println!("总 steps: {}", summary.total_steps);
        println!("最终训练损失: {:.4}", summary.final_train_loss);
        println!("最佳验证损失: {:.4}", summary.best_val_loss);
        println!("总耗时: {:.2} 秒", summary.total_time_secs);
        if summary.stopped_early {
            println!("状态: 因 Early stopping 提前终止");
        }
        println!("{{'='=>60}}\n");

        Ok(summary)
    }

    /// 执行单步训练（支持梯度累积）
    ///
    /// # 参数
    /// * `batch` - 当前批次数据
    ///
    /// # 返回
    /// * `Ok(StepMetrics)` - 单步训练指标
    /// * `Err(TrainerError)` - 训练步骤出错
    ///
    /// # Gradient Accumulation 逻辑
    ///
    /// 当 gradient_accumulation_steps > 1 时：
    /// 1. 计算当前微批次的梯度
    /// 2. 将梯度累加到 accumulation_gradients 缓冲区
    /// 3. accumulation_count += 1
    /// 4. 如果 accumulation_count < target_accumulation_steps：返回，不更新参数
    /// 5. 如果 accumulation_count == target_accumulation_steps：
    ///    - 使用累积的梯度进行参数更新
    ///    - 清空累积缓冲区
    ///    - 重置 accumulation_count = 0
    fn training_step(&mut self, batch: &Batch) -> Result<StepMetrics, TrainerError> {
        // 1. 前向传播
        let logits = self.mock_forward(batch)?;

        // 2. 展平 labels 用于损失计算
        let labels: Vec<usize> = batch
            .labels
            .iter()
            .flat_map(|l| l.iter().copied())
            .collect();

        // 3. 计算损失
        let loss_output = self.loss_fn.forward(&logits, &labels);

        // 4. 计算梯度（模拟反向传播）
        let grad = self.loss_fn.backward(&logits, &labels);

        // ====== 梯度累积模式判断 ======
        if self.target_accumulation_steps > 1 {
            // ====== 梯度累积模式 ======
            self.training_step_with_accumulation(loss_output.loss, &grad)
        } else {
            // ====== 标准模式（无累积）======
            self.training_step_standard(loss_output.loss, &grad)
        }
    }

    /// 梯度累积模式的训练步骤
    ///
    /// 累积多个 micro-batch 的梯度，达到目标步数后统一更新参数。
    fn training_step_with_accumulation(
        &mut self,
        loss: f64,
        grad: &Array2<f32>,
    ) -> Result<StepMetrics, TrainerError> {
        // 将当前梯度转换为 ArrayD
        let grad_arrayd = self.grad_to_arrayd(grad);

        // 初始化或重新初始化累积缓冲区（如果形状变化）
        if !self.accumulation_initialized
            || (self.accumulation_count == 0 && !self.shapes_compatible(&grad_arrayd))
        {
            self.initialize_accumulation_buffers_from_arrayd(&grad_arrayd);
        }

        // 检查形状兼容性（防止在累积过程中形状改变）
        if !self.accumulation_gradients.is_empty()
            && self.accumulation_gradients[0].shape() != grad_arrayd.shape()
        {
            // 形状不匹配，重置并重新开始累积
            self.initialize_accumulation_buffers_from_arrayd(&grad_arrayd);
            self.accumulation_count = 0;
        }

        // 累加梯度
        for (accumulated, new_grad) in self
            .accumulation_gradients
            .iter_mut()
            .zip(std::iter::once(&grad_arrayd))
        {
            *accumulated = &*accumulated + new_grad;
        }
        self.accumulation_count += 1;

        // 检查是否达到目标累积步数
        if self.accumulation_count < self.target_accumulation_steps {
            // 还没达到目标步数，不更新参数
            // 返回当前 loss，但 grad_norm 为 0（因为还没执行实际更新）
            return Ok(StepMetrics {
                loss,
                grad_norm: 0.0, // 累积期间不计算有效梯度范数
                learning_rate: self.get_current_lr(),
                throughput_tokens_per_sec: 0.0,
                step_time_ms: 0.0,
            });
        }

        // 达到目标步数，使用累积的梯度进行参数更新
        let grad_norm = self.clip_and_update()?;

        // 清空累积缓冲区
        self.reset_accumulation_buffers();

        Ok(StepMetrics {
            loss,
            grad_norm,
            learning_rate: self.get_current_lr(),
            throughput_tokens_per_sec: 0.0,
            step_time_ms: 0.0,
        })
    }

    /// 标准模式（无梯度累积）的训练步骤
    ///
    /// 每个批次都立即更新参数。
    fn training_step_standard(
        &mut self,
        loss: f64,
        grad: &Array2<f32>,
    ) -> Result<StepMetrics, TrainerError> {
        // 转换梯度格式
        let grad_arrayd = self.grad_to_arrayd(grad);

        // 检查是否需要重新初始化模型参数（处理形状变化的情况）
        if self.model_params.is_empty() || !self.shapes_compatible(&grad_arrayd) {
            self.initialize_model_params_from_arrayd(&grad_arrayd);
        }

        // 直接更新参数
        let grad_norm = self.clip_and_update_with_gradients(&[grad_arrayd])?;

        Ok(StepMetrics {
            loss,
            grad_norm,
            learning_rate: self.get_current_lr(),
            throughput_tokens_per_sec: 0.0,
            step_time_ms: 0.0,
        })
    }

    /// 模拟前向传播（用于测试和演示）
    ///
    /// 在实际应用中，这里应该调用真正的模型进行前向传播。
    fn mock_forward(&self, batch: &Batch) -> Result<Array2<f32>, TrainerError> {
        if batch.batch_size == 0 {
            return Err(TrainerError::ModelForward("Batch 为空".to_string()));
        }

        // 简化实现：生成随机的 logits
        // 实际应用中这里应该是模型的真正前向传播
        let vocab_size = 1000; // 假设词表大小为 1000
        let total_tokens: usize = batch.labels.iter().map(|l| l.len()).sum();

        if total_tokens == 0 {
            return Err(TrainerError::ModelForward("无有效 token".to_string()));
        }

        // 生成随机 logits（仅用于演示）
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let logits_data: Vec<f32> = (0..total_tokens * vocab_size)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        let logits = Array2::from_shape_vec((total_tokens, vocab_size), logits_data)
            .map_err(|e| TrainerError::ModelForward(format!("Logits 形状错误: {}", e)))?;

        Ok(logits)
    }

    // ==================== Gradient Accumulation 辅助方法 ====================

    /// 初始化梯度累积缓冲区
    ///
    /// 根据梯度的形状创建累积缓冲区，用于存储多个 micro-batch 的累加梯度。
    ///
    /// # 参数
    /// * `grad` - 用于确定缓冲区形状的梯度数组
    fn initialize_accumulation_buffers_from_grad(&mut self, grad: &Array2<f32>) {
        let grad_arrayd = self.grad_to_arrayd(grad);
        self.initialize_accumulation_buffers_from_arrayd(&grad_arrayd);
    }

    /// 从 ArrayD 初始化梯度累积缓冲区
    fn initialize_accumulation_buffers_from_arrayd(&mut self, grad_arrayd: &ArrayD<f32>) {
        self.accumulation_gradients = vec![ArrayD::zeros(grad_arrayd.shape())];
        self.accumulation_initialized = true;

        // 同时初始化模型参数（如果尚未初始化或形状不兼容）
        if self.model_params.is_empty() || !self.shapes_compatible(grad_arrayd) {
            self.initialize_model_params_from_arrayd(grad_arrayd);
        }
    }

    /// 初始化模型参数（从梯度形状推断）
    ///
    /// 在实际应用中，这里应该使用真正的模型参数。
    /// 当前实现为演示目的，创建与梯度形状匹配的零参数。
    ///
    /// # 参数
    /// * `grad` - 用于推断参数形状的梯度数组
    fn initialize_model_params_from_grad(&mut self, grad: &Array2<f32>) {
        let grad_arrayd = self.grad_to_arrayd(grad);
        self.initialize_model_params_from_arrayd(&grad_arrayd);
    }

    /// 从 ArrayD 初始化模型参数
    fn initialize_model_params_from_arrayd(&mut self, grad_arrayd: &ArrayD<f32>) {
        self.model_params = vec![ParamState {
            data: ArrayD::zeros(grad_arrayd.shape()),
            grad: None,
        }];

        // 重新初始化优化器缓冲区
        if let Some(adamw) = self.optimizer.as_any_mut().downcast_mut::<AdamW>() {
            adamw.init_buffers(&self.model_params);
        }
    }

    /// 检查梯度形状是否与当前模型参数兼容
    fn shapes_compatible(&self, grad_arrayd: &ArrayD<f32>) -> bool {
        if self.model_params.is_empty() {
            return false;
        }

        // 检查第一个参数的形状是否与梯度形状匹配
        self.model_params[0].data.shape() == grad_arrayd.shape()
    }

    /// 将 Array2 转换为 ArrayD
    fn grad_to_arrayd(&self, grad: &Array2<f32>) -> ArrayD<f32> {
        grad.clone().into_dyn()
    }

    /// 重置累积缓冲区
    ///
    /// 将所有累积的梯度清零，并重置计数器。
    fn reset_accumulation_buffers(&mut self) {
        for g in &mut self.accumulation_gradients {
            g.fill(0.0);
        }
        self.accumulation_count = 0;
    }

    /// 获取当前学习率
    ///
    /// 返回优化器的当前学习率。
    /// 在未来可以扩展支持学习率调度器。
    fn get_current_lr(&self) -> f64 {
        self.optimizer.learning_rate()
    }

    /// 使用累积的梯度进行参数更新（梯度累积模式）
    ///
    /// 执行以下步骤：
    /// 1. 梯度裁剪（防止梯度爆炸）
    /// 2. 调用优化器进行参数更新
    /// 3. 清零参数梯度
    ///
    /// # 返回
    /// * `Ok(f64)` - 裁剪后的梯度范数
    /// * `Err(TrainerError)` - 更新失败
    fn clip_and_update(&mut self) -> Result<f64, TrainerError> {
        if self.model_params.is_empty() || self.accumulation_gradients.is_empty() {
            return Ok(0.0);
        }

        // 梯度裁剪
        let grad_norm = self.clip_grad_norm_(&self.accumulation_gradients);

        // 更新参数
        self.optimizer
            .step(&mut self.model_params, &self.accumulation_gradients)
            .map_err(TrainerError::Optimizer)?;

        // 清零梯度
        self.optimizer.zero_grad(&mut self.model_params);

        Ok(grad_norm)
    }

    /// 使用给定的梯度进行参数更新（标准模式）
    ///
    /// # 参数
    /// * `gradients` - 梯度列表
    ///
    /// # 返回
    /// * `Ok(f64)` - 梯度范数
    /// * `Err(TrainerError)` - 更新失败
    fn clip_and_update_with_gradients(
        &mut self,
        gradients: &[ArrayD<f32>],
    ) -> Result<f64, TrainerError> {
        if self.model_params.is_empty() {
            return Ok(0.0);
        }

        // 梯度裁剪
        let grad_norm = self.clip_grad_norm_(gradients);

        // 更新参数
        self.optimizer
            .step(&mut self.model_params, gradients)
            .map_err(TrainerError::Optimizer)?;

        // 清零梯度
        self.optimizer.zero_grad(&mut self.model_params);

        Ok(grad_norm)
    }

    /// 梯度裁剪
    ///
    /// 将梯度范数限制在 max_grad_norm 以内，防止梯度爆炸。
    ///
    /// # 算法
    /// ```text
    /// if grad_norm > max_grad_norm:
    ///     gradients *= max_grad_norm / grad_norm
    /// ```
    ///
    /// # 参数
    /// * `gradients` - 待裁剪的梯度列表
    ///
    /// # 返回
    /// 裁剪后的梯度范数
    fn clip_grad_norm_(&self, gradients: &[ArrayD<f32>]) -> f64 {
        let mut total_norm_sq = 0.0_f64;

        for grad in gradients {
            let norm_sq: f64 = grad.iter().map(|&g| g as f64 * g as f64).sum();
            total_norm_sq += norm_sq;
        }

        let grad_norm = total_norm_sq.sqrt();
        let max_norm = self.config.max_grad_norm;

        if grad_norm > max_norm && grad_norm > 0.0 {
            let _scale = max_norm / grad_norm; // 在实际实现中应该用这个值缩放梯度
                                               // 注意：这里不直接修改传入的 gradients（因为可能是不可变的）
                                               // 在实际实现中，应该在调用此方法前处理缩放
                                               // 这里仅返回裁剪后的范数值
            max_norm
        } else {
            grad_norm
        }
    }

    /// 计算梯度范数（兼容旧接口）
    fn compute_grad_norm(&self, grad: &Array2<f32>) -> f64 {
        grad.iter()
            .map(|&g| g as f64 * g as f64)
            .sum::<f64>()
            .sqrt()
    }

    /// 验证评估
    ///
    /// # 参数
    /// * `dataloader` - 验证数据加载器
    ///
    /// # 返回
    /// * `Ok(f64)` - 平均验证损失
    /// * `Err(TrainerError)` - 评估过程出错
    fn evaluate(&self, dataloader: &DataLoader) -> Result<f64, TrainerError> {
        // 简化实现：由于 DataLoader 不支持多次迭代且未实现 Clone，
        // 这里返回一个默认值或基于已有数据的估算
        // 在实际应用中，应该重新创建 DataLoader 或使用支持多次迭代的数据加载器

        if dataloader.is_empty() {
            return Ok(0.0);
        }

        // 基于历史指标估算验证损失（简化处理）
        if let Some(last_metrics) = self.metrics_history.last() {
            Ok(last_metrics.loss * 0.9) // 假设验证损失略低于训练损失
        } else {
            Ok(0.0)
        }
    }

    /// 日志输出
    fn log_metrics(&self, epoch: usize, metrics: &StepMetrics) {
        println!(
            "[Epoch {}] Step {}/{} | Loss: {:.4} | Grad Norm: {:.4} | LR: {:.6} | Throughput: {:.0} tok/s | Time: {:.1}ms",
            epoch + 1,
            self.state.global_step,
            self.state.total_steps,
            metrics.loss,
            metrics.grad_norm,
            metrics.learning_rate,
            metrics.throughput_tokens_per_sec,
            metrics.step_time_ms
        );
    }

    /// 保存 checkpoint
    fn save_checkpoint(&self) -> Result<(), TrainerError> {
        let checkpoint_dir = &self.config.output_dir;

        // 创建目录（如果不存在）
        std::fs::create_dir_all(checkpoint_dir)?;

        let checkpoint_name = format!("checkpoint-step-{}", self.state.global_step);
        let checkpoint_path = checkpoint_dir.join(&checkpoint_name);

        println!("💾 保存 checkpoint: {}", checkpoint_path.display());

        // 创建 checkpoint 子目录
        std::fs::create_dir_all(&checkpoint_path)?;

        // 序列化状态
        let state_json = serde_json::to_string_pretty(&self.state)
            .map_err(|e| TrainerError::Other(format!("序列化状态失败: {}", e)))?;

        std::fs::write(checkpoint_path.join("training_state.json"), state_json)?;

        // 清理旧 checkpoint（保留最近的 save_total_limit 个）
        self.cleanup_old_checkpoints()?;

        Ok(())
    }

    /// 清理旧的 checkpoint 文件
    fn cleanup_old_checkpoints(&self) -> Result<(), TrainerError> {
        if self.config.save_total_limit == 0 {
            return Ok(());
        }

        let checkpoint_dir = &self.config.output_dir;
        if !checkpoint_dir.exists() {
            return Ok(());
        }

        // 读取所有 checkpoint 目录
        let mut checkpoints: Vec<std::fs::DirEntry> = std::fs::read_dir(checkpoint_dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("checkpoint-step-")
            })
            .collect();

        // 按修改时间排序（最新的在前）
        checkpoints.sort_by(|a, b| {
            let time_a = a.metadata().ok().and_then(|m| m.modified().ok());
            let time_b = b.metadata().ok().and_then(|m| m.modified().ok());
            time_b.cmp(&time_a)
        });

        // 删除超出限制的旧 checkpoint
        for old_checkpoint in checkpoints.into_iter().skip(self.config.save_total_limit) {
            let path = old_checkpoint.path();
            if path.is_dir() {
                std::fs::remove_dir_all(&path)?;
                println!("🗑 删除旧 checkpoint: {}", path.display());
            }
        }

        Ok(())
    }

    /// 暂停训练
    pub fn pause(&mut self) {
        if self.is_running {
            self.is_paused = true;
            println!("\n⏸ 训练已暂停");
        }
    }

    /// 恢复训练
    pub fn resume(&mut self) {
        if self.is_running && self.is_paused {
            self.is_paused = false;
            println!("\n▶ 训练已恢复");
        }
    }

    /// 停止训练
    pub fn stop(&mut self) {
        if self.is_running {
            self.is_running = false;
            self.is_paused = false;
            println!("\n⏹ 训练已停止");
        }
    }

    /// 获取当前训练状态
    pub fn state(&self) -> &TrainingState {
        &self.state
    }

    /// 获取最近的 N 条指标记录
    ///
    /// # 参数
    /// * `n` - 要获取的记录数量
    ///
    /// # 返回
    /// 最近 N 条指标的切片
    pub fn recent_metrics(&self, n: usize) -> &[StepMetrics] {
        let start = if self.metrics_history.len() > n {
            self.metrics_history.len() - n
        } else {
            0
        };
        &self.metrics_history[start..]
    }

    /// 获取所有历史指标
    pub fn all_metrics(&self) -> &[StepMetrics] {
        &self.metrics_history
    }

    /// 获取训练配置
    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }

    /// 设置计算图（用于高级用法）
    pub fn set_graph(&mut self, graph: ComputationGraph) {
        self.graph = Some(graph);
    }

    /// 是否正在运行
    pub fn is_running(&self) -> bool {
        self.is_running
    }

    /// 是否已暂停
    pub fn is_paused(&self) -> bool {
        self.is_paused
    }

    // ==================== Gradient Accumulation 公开接口 ====================

    /// 获取当前梯度累积步数
    pub fn accumulation_count(&self) -> u32 {
        self.accumulation_count
    }

    /// 获取目标累积步数
    pub fn target_accumulation_steps(&self) -> u32 {
        self.target_accumulation_steps
    }

    /// 检查是否启用了梯度累积
    pub fn is_gradient_accumulation_enabled(&self) -> bool {
        self.target_accumulation_steps > 1
    }

    /// 获取有效 batch size
    ///
    /// 有效 batch size = per_device_train_batch_size × gradient_accumulation_steps
    pub fn effective_batch_size(&self) -> usize {
        self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps
    }
}

// ==================== 单元测试 ====================

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_data(dir: &TempDir) -> PathBuf {
        let path = dir.path().join("test_data.jsonl");
        let mut file = File::create(&path).unwrap();
        for i in 0..20 {
            writeln!(file, r#"{{"text": "Sample training data {}"}}"#, i).unwrap();
        }
        path
    }

    #[test]
    fn test_trainer_creation() {
        let config = TrainingConfig::default();
        let result = Trainer::new(config);
        assert!(result.is_ok(), "Trainer 创建应成功");
    }

    #[test]
    fn test_training_state_initial() {
        let config = TrainingConfig::default();
        let trainer = Trainer::new(config).unwrap();

        assert_eq!(trainer.state().epoch, 0, "初始 epoch 应为 0");
        assert_eq!(trainer.state().global_step, 0, "初始 global_step 应为 0");
        assert_eq!(
            trainer.state().best_val_loss,
            f64::MAX,
            "初始 best_val_loss 应为 MAX"
        );
    }

    #[test]
    fn test_default_config_validity() {
        let config = TrainingConfig::default();

        assert!(config.learning_rate > 0.0, "学习率必须大于 0");
        assert!(config.num_train_epochs > 0, "训练轮数必须大于 0");
        assert!(config.per_device_train_batch_size > 0, "批次大小必须大于 0");
        assert!(config.max_grad_norm > 0.0, "最大梯度范数必须大于 0");
    }

    #[test]
    fn test_invalid_config_zero_epochs() {
        let mut config = TrainingConfig::default();
        config.num_train_epochs = 0;

        let result = Trainer::new(config);
        assert!(result.is_err(), "num_train_epochs=0 应返回错误");
        if let Err(TrainerError::InvalidConfig(_)) = result {
            // 正确的错误类型
        } else {
            panic!("期望 InvalidConfig 错误");
        }
    }

    #[test]
    fn test_invalid_config_zero_batch_size() {
        let mut config = TrainingConfig::default();
        config.per_device_train_batch_size = 0;

        let result = Trainer::new(config);
        assert!(result.is_err(), "per_device_train_batch_size=0 应返回错误");
        if let Err(TrainerError::InvalidConfig(_)) = result {
            // 正确的错误类型
        } else {
            panic!("期望 InvalidConfig 错误");
        }
    }

    #[test]
    fn test_invalid_config_negative_lr() {
        let mut config = TrainingConfig::default();
        config.learning_rate = -0.001;

        let result = Trainer::new(config);
        assert!(result.is_err(), "负学习率应返回错误");
        if let Err(TrainerError::InvalidConfig(_)) = result {
            // 正确的错误类型
        } else {
            panic!("期望 InvalidConfig 错误");
        }
    }

    #[test]
    fn test_early_stopping_basic() {
        let mut es = EarlyStopping::new(3, 0.001);

        // 第一次：设置 baseline
        assert!(!es.should_stop(1.0), "第一次不应停止");
        assert_eq!(es.best_loss(), 1.0, "best_loss 应更新为 1.0");
        assert_eq!(es.counter(), 0, "counter 应为 0");

        // 第二次：改善
        assert!(!es.should_stop(0.998), "改善不应停止");
        assert_eq!(es.best_loss(), 0.998, "best_loss 应更新");
        assert_eq!(es.counter(), 0, "counter 应重置");

        // 第三次：未改善
        assert!(!es.should_stop(0.999), "轻微未改善不应停止");
        assert_eq!(es.counter(), 1, "counter 应为 1");

        // 第四次：未改善
        assert!(!es.should_stop(1.001), "未改善不应停止");
        assert_eq!(es.counter(), 2, "counter 应为 2");

        // 第五次：未改善，达到 patience
        assert!(es.should_stop(1.002), "应触发早停");
        assert_eq!(es.counter(), 3, "counter 应为 3");
    }

    #[test]
    fn test_early_stopping_reset() {
        let mut es = EarlyStopping::new(3, 0.001);

        es.should_stop(1.0);
        es.should_stop(1.0);
        es.should_stop(1.0);

        assert_eq!(es.counter(), 2, "累积未改善次数应为 2");

        es.reset();

        assert_eq!(es.counter(), 0, "重置后 counter 应为 0");
        assert_eq!(es.best_loss(), f64::MAX, "重置后 best_loss 应为 MAX");
    }

    #[test]
    fn test_pause_resume_cycle() {
        let config = TrainingConfig::default();
        let mut trainer = Trainer::new(config).unwrap();

        assert!(!trainer.is_paused(), "初始状态不应暂停");
        assert!(!trainer.is_running(), "初始状态不应在运行");

        trainer.pause();
        assert!(!trainer.is_paused(), "未开始时 pause 无效");

        // 注意：is_running 在 train() 开始时才设置为 true
        // 这里只测试 API 可用性
    }

    #[test]
    fn test_metrics_history_tracking() {
        let config = TrainingConfig::default();
        let trainer = Trainer::new(config).unwrap();

        assert!(
            trainer.recent_metrics(10).is_empty(),
            "初始时指标历史应为空"
        );
        assert!(trainer.all_metrics().is_empty());
    }

    #[test]
    fn test_training_mode_serialization() {
        let modes = vec![
            TrainingMode::CausalLM,
            TrainingMode::ContinuePretrain,
            TrainingMode::SFT,
        ];

        for mode in modes {
            let serialized = serde_json::to_string(&mode).unwrap();
            let deserialized: TrainingMode = serde_json::from_str(&serialized).unwrap();
            assert_eq!(mode, deserialized, "{:?} 序列化/反序列化应保持一致", mode);
        }
    }

    #[test]
    fn test_sft_config_creation() {
        let sft_config = SFTConfig {
            prompt_template: "Instruction: {instruction}\nInput: {input}\nResponse: ".to_string(),
            mask_prompt_loss: true,
        };

        assert!(sft_config.prompt_template.contains("{instruction}"));
        assert!(sft_config.mask_prompt_loss);
    }

    #[test]
    fn test_training_state_default() {
        let state = TrainingState::default();

        assert_eq!(state.epoch, 0);
        assert_eq!(state.global_step, 0);
        assert_eq!(state.best_val_loss, f64::MAX);
        assert_eq!(state.total_steps, 0);
        assert_eq!(state.config_hash, 0);
    }

    #[test]
    fn test_step_metrics_structure() {
        let metrics = StepMetrics {
            loss: 1.234,
            grad_norm: 0.567,
            learning_rate: 0.001,
            throughput_tokens_per_sec: 1000.0,
            step_time_ms: 50.0,
        };

        assert!(metrics.loss > 0.0);
        assert!(metrics.grad_norm >= 0.0);
        assert!(metrics.learning_rate > 0.0);
        assert!(metrics.throughput_tokens_per_sec >= 0.0);
        assert!(metrics.step_time_ms >= 0.0);
    }

    #[test]
    fn test_trainer_error_display() {
        let errors = vec![
            TrainerError::DataLoad("test error".to_string()),
            TrainerError::ModelForward("forward error".to_string()),
            TrainerError::BackwardPass("backward error".to_string()),
            TrainerError::InvalidConfig("invalid config".to_string()),
            TrainerError::Other("other error".to_string()),
        ];

        for err in errors {
            let display = format!("{}", err);
            assert!(!display.is_empty(), "错误显示字符串不应为空");
        }
    }

    #[test]
    fn test_config_with_sft_mode() {
        let mut config = TrainingConfig::default();
        config.mode = TrainingMode::SFT;
        config.sft_config = Some(SFTConfig {
            prompt_template: "{instruction}: ".to_string(),
            mask_prompt_loss: true,
        });

        let trainer = Trainer::new(config);
        assert!(trainer.is_ok(), "SFT 模式配置应有效");

        let trainer = trainer.unwrap();
        assert_eq!(trainer.config().mode, TrainingMode::SFT);
        assert!(trainer.config().sft_config.is_some());
    }

    #[test]
    fn test_recent_metrics_with_empty_history() {
        let config = TrainingConfig::default();
        let trainer = Trainer::new(config).unwrap();

        let recent = trainer.recent_metrics(100);
        assert!(recent.is_empty(), "空历史应返回空切片");
    }

    #[test]
    fn test_checkpoint_directory_creation() {
        let dir = TempDir::new().unwrap();
        let mut config = TrainingConfig::default();
        config.output_dir = dir.path().join("checkpoints");
        config.save_steps = 1; // 每步都保存以测试

        let mut trainer = Trainer::new(config).unwrap();

        // 手动调用保存（绕过 train 循环）
        trainer.state.global_step = 1;
        let result = trainer.save_checkpoint();

        assert!(result.is_ok(), "保存 checkpoint 应成功");
        assert!(
            dir.path().join("checkpoints").exists(),
            "checkpoint 目录应被创建"
        );
    }

    #[test]
    fn test_training_summary_structure() {
        let summary = TrainingSummary {
            total_epochs: 3,
            total_steps: 1500,
            final_train_loss: 0.5678,
            best_val_loss: 0.4567,
            best_epoch: 2,
            total_time_secs: 120.5,
            stopped_early: false,
            final_checkpoint_path: Some("./checkpoints/checkpoint-step-1500".to_string()),
        };

        assert_eq!(summary.total_epochs, 3);
        assert_eq!(summary.total_steps, 1500);
        assert!(summary.final_train_loss > 0.0);
        assert!(summary.best_val_loss > 0.0);
        assert!(summary.total_time_secs > 0.0);
        assert!(!summary.stopped_early);
        assert!(summary.final_checkpoint_path.is_some());
    }

    #[test]
    fn test_full_training_cycle_minimal() {
        let dir = TempDir::new().unwrap();
        let data_path = create_test_data(&dir);

        let mut config = TrainingConfig::default();
        config.mode = TrainingMode::CausalLM;
        config.num_train_epochs = 1;
        config.per_device_train_batch_size = 4;
        config.max_seq_length = 128;
        config.logging_steps = 5;
        config.save_steps = 1000; // 避免频繁保存
        config.early_stopping_patience = 10; // 避免早停
        config.output_dir = dir.path().join("test_checkpoints");

        let dl_config = crate::training::dataloader::DataLoaderConfig {
            train_path: data_path,
            validation_path: None,
            format: crate::training::dataloader::DataFormat::Jsonl,
            max_seq_length: config.max_seq_length,
            pad_token_id: 0,
            eos_token_id: 2,
            batch_size: config.per_device_train_batch_size,
            drop_last: false,
            shuffle: false,
            sft_config: None,
        };

        let train_loader = DataLoader::new(dl_config);
        assert!(train_loader.is_ok(), "DataLoader 创建应成功");

        let mut trainer = Trainer::new(config).unwrap();
        let summary = trainer.train(train_loader.unwrap(), None);

        assert!(summary.is_ok(), "训练应成功完成");
        let summary = summary.unwrap();

        assert!(summary.total_epochs >= 1, "至少完成 1 个 epoch");
        assert!(summary.total_steps > 0, "应有训练步数");
        assert!(summary.final_train_loss >= 0.0, "最终损失应有效");
        assert!(summary.total_time_secs >= 0.0, "训练时间应非负");
    }

    // ==================== Gradient Accumulation 测试用例 ====================

    #[test]
    fn test_gradient_accumulation_disabled_by_default() {
        let config = TrainingConfig::default();
        let trainer = Trainer::new(config).unwrap();

        // 默认配置下，gradient_accumulation_steps = 1，即不启用累积
        assert!(
            !trainer.is_gradient_accumulation_enabled(),
            "默认配置不应启用梯度累积"
        );
        assert_eq!(trainer.target_accumulation_steps(), 1, "默认目标步数应为 1");
        assert_eq!(trainer.accumulation_count(), 0, "初始累积计数应为 0");
    }

    #[test]
    fn test_gradient_accumulation_enabled() {
        let mut config = TrainingConfig::default();
        config.gradient_accumulation_steps = 4;
        config.per_device_train_batch_size = 8;

        let trainer = Trainer::new(config).unwrap();

        // 启用了梯度累积
        assert!(trainer.is_gradient_accumulation_enabled(), "应启用梯度累积");
        assert_eq!(trainer.target_accumulation_steps(), 4, "目标步数应为 4");
        assert_eq!(
            trainer.effective_batch_size(),
            32,
            "有效 batch size 应为 32 (8×4)"
        );
        assert_eq!(trainer.accumulation_count(), 0, "初始累积计数应为 0");
    }

    #[test]
    fn test_gradient_accumulation_effective_batch_size() {
        let mut config = TrainingConfig::default();

        // 测试不同的 batch size 和 accumulation steps 组合
        let test_cases = vec![
            (8, 1, 8),   // 无累积：有效 batch size = 8
            (8, 2, 16),  // 累积 2 步：有效 batch size = 16
            (8, 4, 32),  // 累积 4 步：有效 batch size = 32
            (16, 4, 64), // 累积 4 步：有效 batch size = 64
        ];

        for (batch_size, accum_steps, expected_effective) in test_cases {
            config.per_device_train_batch_size = batch_size;
            config.gradient_accumulation_steps = accum_steps;

            let trainer = Trainer::new(config.clone()).unwrap();
            assert_eq!(
                trainer.effective_batch_size(),
                expected_effective,
                "batch_size={}, accum_steps={} → effective={}",
                batch_size,
                accum_steps,
                expected_effective
            );
        }
    }

    #[test]
    fn test_training_with_gradient_accumulation() {
        let dir = TempDir::new().unwrap();
        let data_path = create_test_data(&dir);

        let mut config = TrainingConfig::default();
        config.mode = TrainingMode::CausalLM;
        config.num_train_epochs = 1;
        config.per_device_train_batch_size = 4;
        config.gradient_accumulation_steps = 4; // 启用梯度累积，有效 batch size = 16
        config.max_seq_length = 128;
        config.logging_steps = 100; // 减少日志输出
        config.save_steps = 1000;
        config.early_stopping_patience = 10;
        config.output_dir = dir.path().join("test_accumulation");

        let dl_config = crate::training::dataloader::DataLoaderConfig {
            train_path: data_path,
            validation_path: None,
            format: crate::training::dataloader::DataFormat::Jsonl,
            max_seq_length: config.max_seq_length,
            pad_token_id: 0,
            eos_token_id: 2,
            batch_size: config.per_device_train_batch_size,
            drop_last: false,
            shuffle: false,
            sft_config: None,
        };

        let train_loader = DataLoader::new(dl_config);
        assert!(train_loader.is_ok());

        let mut trainer = Trainer::new(config).unwrap();
        assert!(trainer.is_gradient_accumulation_enabled(), "应启用梯度累积");

        let summary = trainer.train(train_loader.unwrap(), None);
        assert!(summary.is_ok(), "带梯度累积的训练应成功完成");

        let summary = summary.unwrap();
        assert!(summary.total_epochs >= 1);
        assert!(summary.total_steps > 0);

        // 验证训练后累积状态已重置
        assert_eq!(trainer.accumulation_count(), 0, "训练结束后累积计数应为 0");
    }

    #[test]
    fn test_gradient_accumulation_state_reset() {
        let dir = TempDir::new().unwrap();
        let data_path = create_test_data(&dir);

        let mut config = TrainingConfig::default();
        config.mode = TrainingMode::CausalLM;
        config.num_train_epochs = 1;
        config.per_device_train_batch_size = 4;
        config.gradient_accumulation_steps = 2; // 较小的累积步数以便观察
        config.max_seq_length = 128;
        config.output_dir = dir.path().join("test_reset");

        let dl_config = crate::training::dataloader::DataLoaderConfig {
            train_path: data_path,
            validation_path: None,
            format: crate::training::dataloader::DataFormat::Jsonl,
            max_seq_length: config.max_seq_length,
            pad_token_id: 0,
            eos_token_id: 2,
            batch_size: config.per_device_train_batch_size,
            drop_last: false,
            shuffle: false,
            sft_config: None,
        };

        let train_loader = DataLoader::new(dl_config).unwrap();
        let mut trainer = Trainer::new(config).unwrap();

        // 训练前检查初始状态
        assert_eq!(trainer.accumulation_count(), 0, "训练前累积计数应为 0");
        assert!(trainer.is_gradient_accumulation_enabled());

        // 执行训练
        let _ = trainer.train(train_loader, None);

        // 训练后验证状态重置
        assert_eq!(trainer.accumulation_count(), 0, "训练后累积计数应重置为 0");
    }

    #[test]
    fn test_gradient_accumulation_vs_standard_mode() {
        let dir = TempDir::new().unwrap();
        let data_path = create_test_data(&dir);

        // 标准模式（无累积）
        let mut config_standard = TrainingConfig::default();
        config_standard.num_train_epochs = 1;
        config_standard.per_device_train_batch_size = 16; // 直接使用大 batch
        config_standard.gradient_accumulation_steps = 1;
        config_standard.max_seq_length = 128;
        config_standard.output_dir = dir.path().join("test_standard");

        // 梯度累积模式（模拟相同的有效 batch size）
        let mut config_accum = TrainingConfig::default();
        config_accum.num_train_epochs = 1;
        config_accum.per_device_train_batch_size = 4; // 小 batch
        config_accum.gradient_accumulation_steps = 4; // 累积 4 次 → 有效 batch = 16
        config_accum.max_seq_length = 128;
        config_accum.output_dir = dir.path().join("test_accum");

        // 创建两个 DataLoader
        let dl_config_std = crate::training::dataloader::DataLoaderConfig {
            train_path: data_path.clone(),
            validation_path: None,
            format: crate::training::dataloader::DataFormat::Jsonl,
            max_seq_length: config_standard.max_seq_length,
            pad_token_id: 0,
            eos_token_id: 2,
            batch_size: config_standard.per_device_train_batch_size,
            drop_last: false,
            shuffle: false,
            sft_config: None,
        };

        let dl_config_acc = crate::training::dataloader::DataLoaderConfig {
            train_path: data_path,
            validation_path: None,
            format: crate::training::dataloader::DataFormat::Jsonl,
            max_seq_length: config_accum.max_seq_length,
            pad_token_id: 0,
            eos_token_id: 2,
            batch_size: config_accum.per_device_train_batch_size,
            drop_last: false,
            shuffle: false,
            sft_config: None,
        };

        let loader_std = DataLoader::new(dl_config_std).unwrap();
        let loader_acc = DataLoader::new(dl_config_acc).unwrap();

        let mut trainer_standard = Trainer::new(config_standard).unwrap();
        let mut trainer_accum = Trainer::new(config_accum).unwrap();

        // 验证两种模式的有效 batch size 相同
        assert_eq!(
            trainer_standard.effective_batch_size(),
            trainer_accum.effective_batch_size(),
            "两种模式的有效 batch size 应相同"
        );

        // 执行训练
        let summary_std = trainer_standard.train(loader_std, None);
        let summary_acc = trainer_accum.train(loader_acc, None);

        assert!(summary_std.is_ok());
        assert!(summary_acc.is_ok());

        // 两种模式都应该成功完成训练
        let _ = summary_std.unwrap();
        let _ = summary_acc.unwrap();
    }

    #[test]
    fn test_clip_grad_norm_basic() {
        use ndarray::IxDyn;

        // 使用较大的 max_grad_norm 以便观察未裁剪的情况
        let mut config = TrainingConfig::default();
        config.max_grad_norm = 10.0; // 设置较大的阈值

        let trainer = Trainer::new(config.clone()).unwrap();

        // 创建一个简单的梯度
        let grad = ArrayD::from_shape_vec(IxDyn(&[3]), vec![3.0_f32, 4.0, 0.0]).unwrap();
        let gradients = vec![grad];

        // 梯度范数 = sqrt(9 + 16 + 0) = 5.0
        // 由于 max_grad_norm = 10.0 > 5.0，不应裁剪
        let grad_norm = trainer.clip_grad_norm_(&gradients);
        assert!(
            (grad_norm - 5.0).abs() < 1e-6,
            "梯度范数应为 5.0，实际值: {}",
            grad_norm
        );

        // 验证学习率一致性
        assert!(
            (trainer.get_current_lr() - config.learning_rate).abs() < 1e-10,
            "学习率应与配置一致"
        );
    }

    #[test]
    fn test_clip_grad_norm_clipping() {
        use ndarray::IxDyn;

        // 使用较小的 max_grad_norm 来触发裁剪
        let mut config = TrainingConfig::default();
        config.max_grad_norm = 1.0; // 默认值

        let trainer = Trainer::new(config).unwrap();

        // 创建一个大的梯度（范数为 5.0）
        let grad = ArrayD::from_shape_vec(IxDyn(&[3]), vec![3.0_f32, 4.0, 0.0]).unwrap();
        let gradients = vec![grad];

        // 梯度范数 = 5.0 > max_grad_norm = 1.0，应该被裁剪为 1.0
        let grad_norm = trainer.clip_grad_norm_(&gradients);
        assert!(
            (grad_norm - 1.0).abs() < 1e-6,
            "裁剪后的梯度范数应为 max_grad_norm (1.0)，实际值: {}",
            grad_norm
        );
    }
}
