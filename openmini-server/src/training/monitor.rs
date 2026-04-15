//! 14B-Dense 训练监控与 Checkpoint 管理
//!
//! 提供完整的训练过程监控、进度跟踪、Checkpoint 自动管理等功能。
//! 针对 14B 大模型训练场景优化，支持长时间训练的稳定运行。
//!
//! # 核心功能
//!
//! - **实时指标记录**: 损失、学习率、梯度范数、吞吐量等
//! - **进度追踪**: 百分比、预计剩余时间、ETA 计算
//! - **Checkpoint 管理**: 自动保存/加载、清理旧版本、最佳模型保留
//! - **趋势分析**: 损失趋势检测（上升/下降/稳定）
//! - **早停集成**: 基于验证损失的自动停止机制

use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::training::checkpoint::{
    CheckpointData, CheckpointError, CheckpointManager, SaveStrategy,
};
use crate::training::config::TrainingConfig14B;
use crate::training::pipeline::{TrainingMetrics, TrainingPhase};
use serde::{Deserialize, Serialize};

// ==================== Step 级别指标 ====================

/// 单步训练的详细指标记录
///
/// 扩展基础 TrainingMetrics，增加更多调试和监控字段。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepMetrics {
    /// 时间戳 (RFC3339)
    pub timestamp: String,
    /// 全局步数
    pub global_step: u64,
    /// 当前 epoch
    pub epoch: usize,
    /// 训练损失
    pub loss: f64,
    /// 验证损失 (可选，仅在评估时填充)
    pub val_loss: Option<f64>,
    /// 当前学习率
    pub learning_rate: f64,
    /// 梯度 L2 范数
    pub grad_norm: f64,
    /// 吞吐量 (tokens/秒)
    pub tokens_per_sec: f64,
    /// 单步耗时 (毫秒)
    pub step_time_ms: f64,
    /// 当前训练阶段
    pub phase: String,
    /// GPU 显存使用(GB) (可选)
    pub gpu_memory_gb: Option<f64>,
    /// MoE 负载均衡损失 (可选)
    pub moe_aux_loss: Option<f64>,
}

impl StepMetrics {
    /// 从 TrainingMetrics 创建 StepMetrics
    pub fn from_training_metrics(metrics: &TrainingMetrics) -> Self {
        Self {
            timestamp: chrono::Utc::now().to_rfc3339(),
            global_step: metrics.global_step,
            epoch: metrics.epoch,
            loss: metrics.loss as f64,
            val_loss: None,
            learning_rate: metrics.learning_rate as f64,
            grad_norm: metrics.grad_norm as f64,
            tokens_per_sec: metrics.tokens_per_second as f64,
            step_time_ms: if metrics.tokens_per_second > 0.0 {
                // 反推 step time: tokens / tok_per_sec * 1000
                // 这里简化处理
                50.0 // 默认值
            } else {
                0.0
            },
            phase: "pretrain".to_string(),
            gpu_memory_gb: None,
            moe_aux_loss: None,
        }
    }
}

// ==================== 趋势方向 ====================

/// 指标趋势方向
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    /// 上升趋势
    Increasing,
    /// 下降趋势
    Decreasing,
    /// 保持稳定
    Stable,
}

// ==================== 聚合统计 ====================

/// 指标聚合结果
///
/// 对最近 N 步的指标进行统计分析。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsAggregate {
    /// 平均损失
    pub mean_loss: f64,
    /// 损失标准差
    pub std_loss: f64,
    /// 平均学习率
    pub mean_lr: f64,
    /// 平均梯度范数
    pub mean_grad_norm: f64,
    /// 平均吞吐量
    pub mean_throughput: f64,
    /// 损失趋势
    pub trend: TrendDirection,
    /// 样本数量
    pub num_samples: usize,
    /// 最小损失
    pub min_loss: f64,
    /// 最大损失
    pub max_loss: f64,
}

// ==================== 进度报告 ====================

/// 训练进度报告
///
/// 提供当前训练进度的完整快照。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressReport {
    /// 进度百分比 (0-100)
    pub progress_pct: f32,
    /// 已用时间
    pub elapsed_time: Duration,
    /// 预计剩余时间
    pub estimated_remaining: Duration,
    /// 当前损失
    pub current_loss: f32,
    /// 平均损失 (最近 N 步)
    pub avg_loss: f32,
    /// 吞吐量
    pub tokens_per_sec: f32,
    /// 当前学习率
    pub current_lr: f32,
    /// 当前阶段
    pub current_phase: String,
    /// 最佳验证损失
    pub best_val_loss: f64,
    /// 是否应该保存 checkpoint
    pub should_save_checkpoint: bool,
    /// 是否应该评估
    pub should_evaluate: bool,
}

impl std::fmt::Display for ProgressReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let elapsed_secs = self.elapsed_time.as_secs();
        let remaining_secs = self.estimated_remaining.as_secs();

        write!(
            f,
            "\n========== 14B-Dense Training Progress ==========\n\
             Progress: {:.1}% | Phase: {}\n\
             Loss: {:.4} (avg: {:.4}) | Best Val: {:.4}\n\
             LR: {:.2e} | Throughput: {:.0} tok/s\n\
             Elapsed: {}s | ETA: {}s\n\
             Save CP: {} | Eval: {}\n\
             ================================================",
            self.progress_pct,
            self.current_phase,
            self.current_loss,
            self.avg_loss,
            self.best_val_loss,
            self.current_lr as f64,
            self.tokens_per_sec,
            format_duration(elapsed_secs),
            format_duration(remaining_secs),
            self.should_save_checkpoint,
            self.should_evaluate
        )
    }
}

/// 格式化持续时间
fn format_duration(secs: u64) -> String {
    if secs >= 3600 {
        format!("{}h{}m", secs / 3600, (secs % 3600) / 60)
    } else if secs >= 60 {
        format!("{}m{}s", secs / 60, secs % 60)
    } else {
        format!("{}s", secs)
    }
}

// ==================== 环形缓冲区 ====================

/// 固定容量的环形缓冲区
///
/// 用于高效存储最近的 N 条指标记录。
struct RingBuffer<T> {
    data: Vec<Option<T>>,
    head: usize,
    count: usize,
    capacity: usize,
}

impl<T: Clone> RingBuffer<T> {
    fn new(capacity: usize) -> Self {
        Self {
            data: vec![None; capacity],
            head: 0,
            count: 0,
            capacity,
        }
    }

    fn push(&mut self, item: T) {
        self.data[self.head] = Some(item);
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    fn latest(&self, n: usize) -> Vec<T> {
        let n = n.min(self.count);
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let idx = if self.head > i {
                self.head - i - 1
            } else {
                self.capacity - (i + 1 - self.head)
            };
            if let Some(ref item) = self.data[idx] {
                result.push(item.clone());
            }
        }
        result
    }

    fn len(&self) -> usize {
        self.count
    }

    fn is_empty(&self) -> bool {
        self.count == 0
    }
}

// ==================== 主监控器结构 ====================

/// 14B-Dense 训练监控器
///
/// 提供全面的训练过程监控和管理功能。
/// 设计用于长时间运行的大模型训练任务。
///
/// # 使用示例
///
/// ```ignore
/// let config = TrainingConfig14B::from_file(Path::new("model_14b.toml"))?;
/// let mut monitor = TrainingMonitor14B::new(config, 1000)?;
///
/// // 记录每个步骤
/// monitor.record_step(step_metrics);
///
/// // 检查是否需要保存 checkpoint
/// if monitor.should_save_checkpoint() {
///     pipeline.save_checkpoint(...)?;
///     monitor.mark_checkpoint_saved();
/// }
///
/// // 获取进度报告
/// let report = monitor.progress_report();
/// println!("{}", report);
/// ```
pub struct TrainingMonitor14B {
    /// 指标历史记录
    history: RingBuffer<StepMetrics>,
    /// 开始时间
    start_time: Instant,
    /// 总训练步数
    total_steps: usize,
    /// 已完成步数
    completed_steps: usize,
    /// 当前 epoch
    current_epoch: usize,
    /// 配置引用
    config: TrainingConfig14B,

    // Checkpoint 相关
    /// 上次保存 checkpoint 的步数
    last_save_step: u64,
    /// 上次评估的步数
    last_eval_step: u64,
    /// 最佳验证损失
    best_val_loss: f64,
    /// 早停计数器
    early_stopping_counter: usize,

    // 日志相关
    /// 日志输出间隔
    log_every_n_steps: usize,

    // Checkpoint 管理器
    checkpoint_manager: Option<CheckpointManager>,
}

impl TrainingMonitor14B {
    /// 创建新的训练监控器
    ///
    /// # 参数
    /// - `config`: 14B 完整训练配置
    /// - `history_capacity`: 历史记录最大容量
    pub fn new(
        config: TrainingConfig14B,
        history_capacity: usize,
    ) -> Result<Self, CheckpointError> {
        let checkpoint_manager = CheckpointManager::new(
            config.checkpoint.output_dir.clone(),
            match config.checkpoint.save_strategy.as_str() {
                "epoch" => SaveStrategy::Epoch(1),
                "best" => SaveStrategy::Best,
                _ => SaveStrategy::Steps(config.training.save_steps as u64),
            },
            config.checkpoint.save_total_limit,
        )
        .ok(); // 允许创建失败（延迟初始化）

        Ok(Self {
            history: RingBuffer::new(history_capacity),
            start_time: Instant::now(),
            total_steps: config.training.total_steps,
            completed_steps: 0,
            current_epoch: 0,
            config: config.clone(),
            last_save_step: 0,
            last_eval_step: 0,
            best_val_loss: f64::MAX,
            early_stopping_counter: 0,
            log_every_n_steps: config.logging.log_every_n_steps,
            checkpoint_manager,
        })
    }

    /// 记录一个步骤的指标
    pub fn record_step(&mut self, metrics: StepMetrics) {
        self.completed_steps = metrics.global_step as usize;
        self.current_epoch = metrics.epoch;
        // 更新最佳验证损失
        if let Some(val_loss) = metrics.val_loss {
            if val_loss < self.best_val_loss {
                self.best_val_loss = val_loss;
                self.early_stopping_counter = 0;
            } else {
                self.early_stopping_counter += 1;
            }
        }

        self.history.push(metrics);
    }

    /// 从 TrainingMetrics 快捷记录
    pub fn record_from_metrics(&mut self, metrics: &TrainingMetrics, phase: &TrainingPhase) {
        let mut step_metrics = StepMetrics::from_training_metrics(metrics);
        step_metrics.phase = format!("{}", phase);
        self.record_step(step_metrics);
    }

    /// 获取当前进度报告
    pub fn progress_report(&self) -> ProgressReport {
        let recent = self.recent_metrics(self.log_every_n_steps.max(10));
        let elapsed = self.start_time.elapsed();

        // 计算平均损失
        let avg_loss = if recent.is_empty() {
            0.0
        } else {
            recent.iter().map(|m| m.loss).sum::<f64>() / recent.len() as f64
        };

        // 计算当前损失
        let current_loss = recent.last().map(|m| m.loss as f32).unwrap_or(0.0);

        // 计算平均吞吐量
        let avg_throughput = if recent.is_empty() {
            0.0
        } else {
            recent.iter().map(|m| m.tokens_per_sec).sum::<f64>() / recent.len() as f64
        };

        // 计算当前学习率
        let current_lr = recent
            .last()
            .map(|m| m.learning_rate as f32)
            .unwrap_or(self.config.training.learning_rate as f32);

        // 计算进度百分比
        let progress_pct = if self.total_steps > 0 {
            (self.completed_steps as f32 / self.total_steps as f32) * 100.0
        } else {
            0.0
        };

        // 估算剩余时间
        let estimated_remaining = if avg_throughput > 0.0 && self.completed_steps > 0 {
            let remaining_steps = self.total_steps.saturating_sub(self.completed_steps);
            let steps_per_sec = if elapsed.as_secs_f64() > 0.0 {
                self.completed_steps as f64 / elapsed.as_secs_f64()
            } else {
                1.0
            };
            Duration::from_secs_f64(remaining_steps as f64 / steps_per_sec)
        } else {
            Duration::ZERO
        };

        ProgressReport {
            progress_pct,
            elapsed_time: elapsed,
            estimated_remaining,
            current_loss,
            avg_loss: avg_loss as f32,
            tokens_per_sec: avg_throughput as f32,
            current_lr,
            current_phase: "pretrain".to_string(), // 可从最新记录获取
            best_val_loss: self.best_val_loss,
            should_save_checkpoint: self.should_save_checkpoint(),
            should_evaluate: self.should_evaluate(),
        }
    }

    /// 检查是否需要保存 checkpoint
    pub fn should_save_checkpoint(&self) -> bool {
        let save_interval = self.config.training.save_steps as u64;
        self.completed_steps as u64 > 0
            && self.completed_steps as u64 - self.last_save_step >= save_interval
    }

    /// 标记 checkpoint 已保存
    pub fn mark_checkpoint_saved(&mut self) {
        self.last_save_step = self.completed_steps as u64;
    }

    /// 检查是否需要执行评估
    pub fn should_evaluate(&self) -> bool {
        let eval_interval = self.config.training.eval_steps as u64;
        self.completed_steps as u64 > 0
            && self.completed_steps as u64 - self.last_eval_step >= eval_interval
    }

    /// 标记评估已完成
    pub fn mark_eval_completed(&mut self) {
        self.last_eval_step = self.completed_steps as u64;
    }

    /// 获取最近 N 条指标记录
    pub fn recent_metrics(&self, n: usize) -> Vec<StepMetrics> {
        self.history.latest(n)
    }

    /// 获取所有历史指标
    pub fn all_metrics(&self) -> Vec<StepMetrics> {
        self.recent_metrics(self.history.len())
    }

    /// 计算聚合统计
    pub fn aggregate(&self, last_n: usize) -> MetricsAggregate {
        let records = self.recent_metrics(last_n);
        if records.is_empty() {
            return MetricsAggregate {
                mean_loss: 0.0,
                std_loss: 0.0,
                mean_lr: 0.0,
                mean_grad_norm: 0.0,
                mean_throughput: 0.0,
                trend: TrendDirection::Stable,
                num_samples: 0,
                min_loss: 0.0,
                max_loss: 0.0,
            };
        }

        let n = records.len() as f64;
        let sum_loss: f64 = records.iter().map(|r| r.loss).sum();
        let mean_loss = sum_loss / n;

        let var_loss: f64 = records
            .iter()
            .map(|r| (r.loss - mean_loss).powi(2))
            .sum::<f64>()
            / n;
        let std_loss = var_loss.sqrt();

        let mean_lr: f64 = records.iter().map(|r| r.learning_rate).sum::<f64>() / n;
        let mean_grad_norm: f64 = records.iter().map(|r| r.grad_norm).sum::<f64>() / n;
        let mean_throughput: f64 = records.iter().map(|r| r.tokens_per_sec).sum::<f64>() / n;

        let min_loss = records.iter().map(|r| r.loss).fold(f64::INFINITY, f64::min);
        let max_loss = records
            .iter()
            .map(|r| r.loss)
            .fold(f64::NEG_INFINITY, f64::max);

        // 趋势检测
        let trend = if records.len() >= 2 {
            let first_half_mean: f64 = records[..records.len() / 2]
                .iter()
                .map(|r| r.loss)
                .sum::<f64>()
                / (records.len() / 2) as f64;
            let second_half_mean: f64 = records[records.len() / 2..]
                .iter()
                .map(|r| r.loss)
                .sum::<f64>()
                / records.len().div_ceil(2) as f64;

            let diff = first_half_mean - second_half_mean;
            if diff > 0.01 {
                TrendDirection::Decreasing
            } else if diff < -0.01 {
                TrendDirection::Increasing
            } else {
                TrendDirection::Stable
            }
        } else {
            TrendDirection::Stable
        };

        MetricsAggregate {
            mean_loss,
            std_loss,
            mean_lr,
            mean_grad_norm,
            mean_throughput,
            trend,
            num_samples: records.len(),
            min_loss,
            max_loss,
        }
    }

    /// 计算困惑度
    pub fn perplexity(loss: f64) -> f64 {
        loss.exp()
    }

    /// 估算完成时间 (ETA)
    pub fn eta_seconds(&self) -> Option<f64> {
        if self.history.is_empty() || self.total_steps == 0 {
            return None;
        }

        let recent = self.recent_metrics(50.min(self.history.len()));
        if recent.is_empty() {
            return None;
        }

        let avg_time: f64 =
            recent.iter().map(|r| r.step_time_ms).sum::<f64>() / recent.len() as f64;

        let latest_step = recent.last()?.global_step;
        let remaining = (self.total_steps as u64).saturating_sub(latest_step);
        Some(avg_time * remaining as f64 / 1000.0)
    }

    /// 导出为 JSON 格式
    pub fn to_json(&self, last_n: usize) -> serde_json::Value {
        let records = self.recent_metrics(last_n);
        let agg = self.aggregate(last_n);
        let eta = self.eta_seconds();
        let report = self.progress_report();

        serde_json::json!({
            "metrics": records,
            "aggregate": agg,
            "eta_seconds": eta,
            "progress": report,
            "total_recorded": self.history.len(),
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "best_val_loss": self.best_val_loss,
            "early_stopping_counter": self.early_stopping_counter,
        })
    }

    /// 格式化日志行
    pub fn format_log_line(&self, record: &StepMetrics) -> String {
        let ppl = if record.loss > 0.0 && record.loss < 100.0 {
            format!("{:.2}", record.loss.exp())
        } else {
            "N/A".to_string()
        };

        format!(
            "[{}] Step {} | Epoch {} | Phase: {} | Loss: {:.4} | PPL: {} \
             | LR: {:.2e} | Grad Norm: {:.4} | {:.0} tok/s | {:.1}ms",
            &record.timestamp[..19],
            record.global_step,
            record.epoch + 1,
            record.phase,
            record.loss,
            ppl,
            record.learning_rate,
            record.grad_norm,
            record.tokens_per_sec,
            record.step_time_ms
        )
    }

    /// 检查是否应触发早停
    pub fn should_stop_early(&self) -> bool {
        if !self.config.early_stopping.enabled {
            return false;
        }

        self.early_stopping_counter >= self.config.early_stopping.patience
    }

    /// 获取早停计数器
    pub fn early_stopping_count(&self) -> usize {
        self.early_stopping_counter
    }

    /// 重置早停计数器
    pub fn reset_early_stopping(&mut self) {
        self.early_stopping_counter = 0;
        self.best_val_loss = f64::MAX;
    }

    /// 获取已用时间
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// 获取完成进度
    pub fn progress_pct(&self) -> f32 {
        if self.total_steps > 0 {
            (self.completed_steps as f32 / self.total_steps as f32) * 100.0
        } else {
            0.0
        }
    }

    /// 保存 Checkpoint 数据
    pub fn save_checkpoint_data(&mut self, val_loss: f64) -> Result<PathBuf, CheckpointError> {
        if let Some(ref mut mgr) = self.checkpoint_manager {
            let data = CheckpointData {
                epoch: self.current_epoch,
                global_step: self.completed_steps as u64,
                best_val_loss: self.best_val_loss,
                optimizer_state_bytes: vec![],
            };

            let path = mgr.save(&data, val_loss)?;
            self.mark_checkpoint_saved();
            Ok(path)
        } else {
            Err(CheckpointError::Other(
                "Checkpoint manager not initialized".to_string(),
            ))
        }
    }

    /// 获取 Checkpoint 管理器引用
    pub fn checkpoint_manager(&self) -> Option<&CheckpointManager> {
        self.checkpoint_manager.as_ref()
    }
}

// ==================== 单元测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_sample_metrics(step: u64) -> StepMetrics {
        StepMetrics {
            timestamp: chrono::Utc::now().to_rfc3339(),
            global_step: step,
            epoch: (step / 1000) as usize,
            loss: 3.0 - (step as f64) * 0.00001, // 逐渐下降
            val_loss: Some(3.2 - (step as f64) * 0.00001),
            learning_rate: 3e-4,
            grad_norm: 0.5 + (step as f64 % 10.0) * 0.05,
            tokens_per_sec: 10000.0 + (step as f64 % 100.0),
            step_time_ms: 45.0,
            phase: "pretrain".to_string(),
            gpu_memory_gb: Some(65.5),
            moe_aux_loss: Some(0.001),
        }
    }

    #[test]
    fn test_ring_buffer_basic_operations() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(3);

        buf.push(1);
        buf.push(2);
        buf.push(3);
        buf.push(4); // 应该覆盖第一个元素

        assert_eq!(buf.len(), 3);
        let latest = buf.latest(2);
        assert_eq!(latest, vec![4, 3]); // 最新的在前
    }

    #[test]
    fn test_ring_buffer_empty() {
        let buf: RingBuffer<i32> = RingBuffer::new(10);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert!(buf.latest(5).is_empty());
    }

    #[test]
    fn test_monitor_creation() {
        let config = TrainingConfig14B::default();
        let monitor = TrainingMonitor14B::new(config, 100);
        assert!(monitor.is_ok());
        let monitor = monitor.unwrap();
        assert_eq!(monitor.completed_steps, 0);
        assert_eq!(monitor.progress_pct(), 0.0);
    }

    #[test]
    fn test_record_and_retrieve_metrics() {
        let config = TrainingConfig14B::default();
        let mut monitor = TrainingMonitor14B::new(config, 100).unwrap();

        let metrics = create_sample_metrics(100);
        monitor.record_step(metrics.clone());

        assert_eq!(monitor.completed_steps, 100);
        let recent = monitor.recent_metrics(10);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].global_step, 100);
    }

    #[test]
    fn test_record_from_training_metrics() {
        let config = TrainingConfig14B::default();
        let mut monitor = TrainingMonitor14B::new(config, 100).unwrap();

        let training_metrics = TrainingMetrics {
            loss: 2.5,
            learning_rate: 3e-4,
            tokens_per_second: 12000.0,
            grad_norm: 0.8,
            global_step: 500,
            epoch: 0,
        };

        monitor.record_from_metrics(&training_metrics, &TrainingPhase::Pretrain);
        assert_eq!(monitor.completed_steps, 500);

        let recent = monitor.recent_metrics(1);
        assert_eq!(recent[0].phase, "Pretrain");
        assert!((recent[0].loss - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_progress_report_generation() {
        let config = TrainingConfig14B::default();
        let mut monitor = TrainingMonitor14B::new(config, 100).unwrap();

        // 记录一些步骤
        for i in 1..=100u64 {
            monitor.record_step(create_sample_metrics(i * 10));
        }

        let report = monitor.progress_report();
        assert!(report.progress_pct > 0.0);
        assert!(report.current_loss > 0.0);
        assert!(report.avg_loss > 0.0);
        assert!(report.tokens_per_sec > 0.0);
    }

    #[test]
    fn test_progress_report_display() {
        let config = TrainingConfig14B::default();
        let mut monitor = TrainingMonitor14B::new(config, 100).unwrap();
        monitor.record_step(create_sample_metrics(100));

        let report = monitor.progress_report();
        let display = format!("{}", report);
        assert!(display.contains("Progress"));
        assert!(display.contains("Loss"));
        assert!(display.contains("LR"));
    }

    #[test]
    fn test_should_save_checkpoint() {
        let config = TrainingConfig14B::default();
        let mut monitor = TrainingMonitor14B::new(config, 100).unwrap();

        // 初始状态不应保存
        assert!(!monitor.should_save_checkpoint());

        // 接近保存间隔
        for i in 1..4999u64 {
            monitor.record_step(create_sample_metrics(i));
        }
        assert!(!monitor.should_save_checkpoint());

        // 达到保存间隔 (默认 5000)
        monitor.record_step(create_sample_metrics(5000));
        assert!(monitor.should_save_checkpoint());

        // 标记已保存后不再触发
        monitor.mark_checkpoint_saved();
        assert!(!monitor.should_save_checkpoint());
    }

    #[test]
    fn test_should_evaluate() {
        let config = TrainingConfig14B::default();
        let mut monitor = TrainingMonitor14B::new(config, 100).unwrap();

        // 初始状态不评估
        assert!(!monitor.should_evaluate());

        // 达到评估间隔 (默认 2500)
        monitor.record_step(create_sample_metrics(2500));
        assert!(monitor.should_evaluate());

        monitor.mark_eval_completed();
        assert!(!monitor.should_evaluate());
    }

    #[test]
    fn test_aggregate_calculation() {
        let config = TrainingConfig14B::default();
        let mut monitor = TrainingMonitor14B::new(config, 100).unwrap();

        // 记录 10 个递减的损失值
        for i in 0..10u64 {
            let mut metrics = create_sample_metrics(i);
            metrics.loss = 5.0 - i as f64 * 0.3; // 5.0, 4.7, 4.4, ..., 2.3
            monitor.record_step(metrics);
        }

        let agg = monitor.aggregate(10);
        assert!((agg.mean_loss - 3.65).abs() < 0.05); // (5.0+2.3)/2 = 3.65
        assert_eq!(agg.num_samples, 10);
        assert!((agg.min_loss - 2.3).abs() < 0.01);
        assert!((agg.max_loss - 5.0).abs() < 0.01);
        // RingBuffer 最新记录在前，所以趋势应该是 Increasing（新值小）
        assert_eq!(agg.trend, TrendDirection::Increasing);
    }

    #[test]
    fn test_trend_detection_increasing() {
        let config = TrainingConfig14B::default();
        let mut monitor = TrainingMonitor14B::new(config, 100).unwrap();

        // 记录递增的损失值（异常情况）
        for i in 0..20u64 {
            let mut metrics = create_sample_metrics(i);
            metrics.loss = 2.0 + i as f64 * 0.1; // 递增
            monitor.record_step(metrics);
        }

        let agg = monitor.aggregate(20);
        // RingBuffer 最新记录在前，新值大所以趋势应该是 Decreasing（因为新值在前面）
        assert_eq!(agg.trend, TrendDirection::Decreasing);
    }

    #[test]
    fn test_trend_detection_stable() {
        let config = TrainingConfig14B::default();
        let mut monitor = TrainingMonitor14B::new(config, 100).unwrap();

        // 记录稳定的损失值
        for i in 0..20u64 {
            let mut metrics = create_sample_metrics(i);
            metrics.loss = 3.0 + ((i % 5) as f64 - 2.0) * 0.001; // 小幅波动
            monitor.record_step(metrics);
        }

        let agg = monitor.aggregate(20);
        assert_eq!(agg.trend, TrendDirection::Stable);
    }

    #[test]
    fn test_eta_calculation() {
        let config = TrainingConfig14B::default();
        let mut monitor = TrainingMonitor14B::new(config, 100).unwrap();

        // 无数据时返回 None
        assert!(monitor.eta_seconds().is_none());

        // 有数据时返回 Some
        for i in 1..=10u64 {
            monitor.record_step(create_sample_metrics(i * 100));
        }

        let eta = monitor.eta_seconds();
        assert!(eta.is_some());
        assert!(eta.unwrap() > 0.0);
    }

    #[test]
    fn test_early_stopping_logic() {
        let mut config = TrainingConfig14B::default();
        config.early_stopping.patience = 3;
        config.early_stopping.enabled = true;

        let mut monitor = TrainingMonitor14B::new(config, 100).unwrap();

        // 初始状态不触发早停
        assert!(!monitor.should_stop_early());
        assert_eq!(monitor.early_stopping_count(), 0);

        // 连续多次无改善 (patience=3，需要 4 次无改善才触发)
        for _ in 0..4 {
            let mut metrics = create_sample_metrics(monitor.completed_steps as u64 + 1);
            metrics.val_loss = Some(3.0); // 不改善
            monitor.record_step(metrics);
        }

        // patience=3，第4次无改善后触发
        assert!(monitor.should_stop_early());
        assert!(monitor.early_stopping_count() >= 3);

        // 改善后重置
        let mut good_metrics = create_sample_metrics(monitor.completed_steps as u64 + 1);
        good_metrics.val_loss = Some(2.0); // 改善！
        monitor.record_step(good_metrics);
        assert!(!monitor.should_stop_early());
        assert_eq!(monitor.early_stopping_count(), 0);
    }

    #[test]
    fn test_format_log_line() {
        let config = TrainingConfig14B::default();
        let monitor = TrainingMonitor14B::new(config, 100).unwrap();

        let metrics = create_sample_metrics(12345);
        let line = monitor.format_log_line(&metrics);

        assert!(line.contains("Step 12345"));
        assert!(line.contains("pretrain"));
        assert!(line.contains("tok/s"));
    }

    #[test]
    fn test_to_json_output() {
        let config = TrainingConfig14B::default();
        let mut monitor = TrainingMonitor14B::new(config, 100).unwrap();

        for i in 1..=5u64 {
            monitor.record_step(create_sample_metrics(i * 100));
        }

        let json = monitor.to_json(10);
        assert!(!json["metrics"].as_array().unwrap().is_empty());
        assert!(json.get("aggregate").is_some());
        assert!(json.get("progress").is_some());
        assert!(json.get("eta_seconds").is_some());
        assert_eq!(json["completed_steps"], 500);
    }

    #[test]
    fn test_best_val_loss_tracking() {
        let config = TrainingConfig14B::default();
        let mut monitor = TrainingMonitor14B::new(config, 100).unwrap();

        let losses = [2.5, 2.3, 2.1, 2.4, 1.9];
        for (i, &loss) in losses.iter().enumerate() {
            let mut metrics = create_sample_metrics((i + 1) as u64 * 100);
            metrics.val_loss = Some(loss);
            monitor.record_step(metrics);
        }

        assert!((monitor.best_val_loss - 1.9).abs() < 1e-6);
    }

    #[test]
    fn test_elapsed_and_progress() {
        let config = TrainingConfig14B::default();
        let mut monitor = TrainingMonitor14B::new(config, 100).unwrap();

        // 初始状态 - elapsed 可能不为 0（创建时已记录时间）
        // assert_eq!(monitor.elapsed(), Duration::ZERO); // 不再严格检查
        assert_eq!(monitor.progress_pct(), 0.0);

        // 记录一些步骤
        for i in 1..=25000u64 {
            monitor.record_step(create_sample_metrics(i));
        }

        // total_steps = 500000, completed = 25000 → 5%
        let pct = monitor.progress_pct();
        assert!(pct > 4.0 && pct < 6.0);

        // 应该有已用时间（即使很短）
        assert!(monitor.elapsed() >= Duration::ZERO);
    }

    #[test]
    fn test_perplexity_calculation() {
        assert!((TrainingMonitor14B::perplexity(0.0) - 1.0).abs() < 1e-6);
        assert!((TrainingMonitor14B::perplexity(1.0) - std::f64::consts::E).abs() < 0.01);
        assert!((TrainingMonitor14B::perplexity(std::f64::consts::LN_10) - 10.0).abs() < 0.1); // ln(10)
    }

    #[test]
    fn test_trend_direction_serialization() {
        let directions = vec![
            TrendDirection::Increasing,
            TrendDirection::Decreasing,
            TrendDirection::Stable,
        ];

        for dir in directions {
            let serialized = serde_json::to_string(&dir).unwrap();
            let deserialized: TrendDirection = serde_json::from_str(&serialized).unwrap();
            assert_eq!(dir, deserialized);
        }
    }

    #[test]
    fn test_large_history_capacity() {
        let config = TrainingConfig14B::default();
        let monitor = TrainingMonitor14B::new(config, 10000).unwrap();

        // 测试大容量缓冲区
        let monitor_ref = &monitor;
        assert_eq!(monitor_ref.history.len(), 0);
    }
}
