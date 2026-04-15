//! EXO动态策略调整器
//!
//! 基于实时性能监控的动态并行策略调整系统，支持：
//! - 实时性能指标收集和分析
//! - 自动策略效果评估和瓶颈检测
//! - 在线策略切换和迁移
//! - 调整决策学习和优化
//!
//! # 架构概述
//!
//! ```text
//! ExoDynamicAdjuster
//! ├── PerformanceMonitor (性能监控器)
//! │   ├── LatencyTracker (延迟跟踪器)
//! │   ├── ThroughputTracker (吞吐量跟踪器)
//! │   ├── ResourceUsageTracker (资源使用跟踪器)
//! │   └── ErrorRateTracker (错误率跟踪器)
//! ├── StrategyEvaluator (策略评估器)
//! │   ├── BottleneckDetector (瓶颈检测器)
//! │   ├── PerformanceAnalyzer (性能分析器)
//! │   └── AlternativeStrategyFinder (替代策略查找器)
//! ├── AdjustmentExecutor (调整执行器)
//! │   ├── StrategyMigrator (策略迁移器)
//! │   ├── StateSynchronizer (状态同步器)
//! │   └── RollbackManager (回滚管理器)
//! └── DecisionLearner (决策学习器)
//!     ├── AdjustmentHistory (调整历史)
//!     └── PerformanceModelUpdater (性能模型更新器)
//! ```
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use openmini_server::model::inference::exo_dynamic_adjuster::{
//!     ExoDynamicAdjuster, PerformanceMetrics, AdjustmentDecision
//! };
//!
//! // 创建动态调整器
//! let mut adjuster = ExoDynamicAdjuster::new();
//!
//! // 收集性能指标
//! let metrics = PerformanceMetrics {
//!     latency_ms: 120.5,
//!     throughput_tps: 85.3,
//!     memory_usage_gb: 12.7,
//!     error_rate: 0.01,
//!     timestamp: SystemTime::now(),
//! };
//!
//! // 评估是否需要调整
//! let decision = adjuster.evaluate_and_adjust(
//!     &metrics,
//!     &current_strategy,
//!     &device_topology,
//! )?;
//!
//! match decision {
//!     AdjustmentDecision::NoChange => println!("策略无需调整"),
//!     AdjustmentDecision::StrategyChange(new_strategy) => {
//!         println!("切换到新策略: {:?}", new_strategy);
//!     }
//!     AdjustmentDecision::ParameterTuning(params) => {
//!         println!("调整参数: {:?}", params);
//!     }
//! }
//! ```

use std::collections::HashMap;
use std::time::SystemTime;

use log::{debug, info, warn};

use super::{DeviceTopology, ExoParallelStrategyEngine, ParallelStrategyDecision, StrategyConfig};
use crate::model::inference::distributed_inference_config::ParallelStrategy;

// ==================== 核心类型定义 ====================

/// 性能指标
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// 推理延迟 (毫秒)
    pub latency_ms: f32,

    /// 吞吐量 (tokens/秒)
    pub throughput_tps: f32,

    /// 内存使用量 (GB)
    pub memory_usage_gb: f32,

    /// 错误率 (0-1)
    pub error_rate: f32,

    /// CPU使用率 (0-1)
    pub cpu_usage: f32,

    /// GPU使用率 (0-1)，如果没有GPU则为None
    pub gpu_usage: Option<f32>,

    /// 网络带宽使用率 (0-1)
    pub network_bandwidth_usage: f32,

    /// 收集时间戳
    pub timestamp: SystemTime,

    /// 批次大小
    pub batch_size: usize,

    /// 模型大小 (GB)
    pub model_size_gb: f32,

    /// 附加元数据
    pub metadata: HashMap<String, String>,
}

impl PerformanceMetrics {
    /// 创建新的性能指标
    pub fn new(
        latency_ms: f32,
        throughput_tps: f32,
        memory_usage_gb: f32,
        batch_size: usize,
        model_size_gb: f32,
    ) -> Self {
        Self {
            latency_ms,
            throughput_tps,
            memory_usage_gb,
            error_rate: 0.0,
            cpu_usage: 0.0,
            gpu_usage: None,
            network_bandwidth_usage: 0.0,
            timestamp: SystemTime::now(),
            batch_size,
            model_size_gb,
            metadata: HashMap::new(),
        }
    }

    /// 计算性能评分 (越高越好)
    pub fn performance_score(&self) -> f32 {
        // 简单评分公式：基于延迟和吞吐量
        let latency_score = if self.latency_ms > 0.0 {
            1000.0 / self.latency_ms // 延迟越低，分数越高
        } else {
            0.0
        };

        let throughput_score = self.throughput_tps;

        let memory_efficiency = if self.memory_usage_gb > 0.0 {
            self.throughput_tps / self.memory_usage_gb
        } else {
            0.0
        };

        // 加权平均
        (latency_score * 0.4 + throughput_score * 0.4 + memory_efficiency * 0.2).max(0.0)
    }

    /// 检查是否满足服务等级协议 (SLA)
    pub fn meets_sla(&self, target_latency_ms: f32, min_throughput_tps: f32) -> bool {
        self.latency_ms <= target_latency_ms && self.throughput_tps >= min_throughput_tps
    }
}

/// 调整决策
#[derive(Debug, Clone)]
pub enum AdjustmentDecision {
    /// 无需调整
    NoChange { reason: String, confidence: f32 },

    /// 切换到新策略
    StrategyChange {
        new_strategy: ParallelStrategyDecision,
        reason: String,
        expected_improvement: f32, // 预期改进百分比
        migration_cost: f32,       // 迁移成本评分 (0-1)
    },

    /// 调整策略参数
    ParameterTuning {
        new_config: StrategyConfig,
        reason: String,
        expected_improvement: f32,
    },

    /// 回滚到之前的策略
    Rollback {
        previous_strategy: ParallelStrategyDecision,
        reason: String,
    },
}

/// 瓶颈类型
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BottleneckType {
    /// 计算瓶颈 (CPU/GPU)
    Compute,

    /// 内存瓶颈
    Memory,

    /// 网络通信瓶颈
    Network,

    /// 输入/输出瓶颈
    Io,

    /// 策略配置瓶颈
    Strategy,

    /// 未知瓶颈
    Unknown,
}

/// 瓶颈分析结果
#[derive(Debug, Clone)]
pub struct BottleneckAnalysis {
    /// 瓶颈类型
    pub bottleneck_type: BottleneckType,

    /// 置信度 (0-1)
    pub confidence: f32,

    /// 严重程度 (0-1)
    pub severity: f32,

    /// 原因描述
    pub reason: String,

    /// 建议的解决方案
    pub suggested_solutions: Vec<String>,
}

/// 调整器配置
#[derive(Debug, Clone)]
pub struct AdjusterConfig {
    /// 性能监控间隔 (秒)
    pub monitoring_interval_sec: u64,

    /// 调整评估窗口大小 (样本数)
    pub evaluation_window_size: usize,

    /// 性能下降阈值 (百分比)
    pub performance_degradation_threshold: f32,

    /// 最小改进阈值 (百分比)
    pub min_improvement_threshold: f32,

    /// 最大调整频率 (秒)
    pub max_adjustment_frequency_sec: u64,

    /// 是否启用自动调整
    pub auto_adjustment_enabled: bool,

    /// 是否允许策略切换
    pub strategy_switching_enabled: bool,

    /// 是否允许参数调整
    pub parameter_tuning_enabled: bool,

    /// 是否启用回滚机制
    pub rollback_enabled: bool,

    /// SLA要求：目标延迟 (毫秒)
    pub target_latency_ms: f32,

    /// SLA要求：最小吞吐量 (tokens/秒)
    pub min_throughput_tps: f32,
}

impl Default for AdjusterConfig {
    fn default() -> Self {
        Self {
            monitoring_interval_sec: 10,
            evaluation_window_size: 10,
            performance_degradation_threshold: 20.0, // 20%性能下降
            min_improvement_threshold: 10.0,         // 至少10%改进
            max_adjustment_frequency_sec: 60,
            auto_adjustment_enabled: true,
            strategy_switching_enabled: true,
            parameter_tuning_enabled: true,
            rollback_enabled: true,
            target_latency_ms: 100.0,
            min_throughput_tps: 50.0,
        }
    }
}

// ==================== EXO动态策略调整器 ====================

/// EXO动态策略调整器
pub struct ExoDynamicAdjuster {
    /// 配置
    config: AdjusterConfig,

    /// 并行策略引擎
    strategy_engine: ExoParallelStrategyEngine,

    /// 性能指标历史记录
    performance_history: Vec<PerformanceMetrics>,

    /// 策略决策历史记录
    strategy_history: Vec<ParallelStrategyDecision>,

    /// 调整决策历史记录
    adjustment_history: Vec<AdjustmentDecision>,

    /// 当前策略
    current_strategy: Option<ParallelStrategyDecision>,

    /// 最后调整时间
    last_adjustment_time: Option<SystemTime>,

    /// 瓶颈分析历史记录
    bottleneck_history: Vec<BottleneckAnalysis>,

    /// 性能基线
    performance_baseline: Option<PerformanceMetrics>,
}

impl ExoDynamicAdjuster {
    /// 创建新的动态调整器
    pub fn new() -> Self {
        Self {
            config: AdjusterConfig::default(),
            strategy_engine: ExoParallelStrategyEngine::new(),
            performance_history: Vec::new(),
            strategy_history: Vec::new(),
            adjustment_history: Vec::new(),
            current_strategy: None,
            last_adjustment_time: None,
            bottleneck_history: Vec::new(),
            performance_baseline: None,
        }
    }

    /// 使用自定义配置创建动态调整器
    pub fn with_config(config: AdjusterConfig) -> Self {
        Self {
            config,
            strategy_engine: ExoParallelStrategyEngine::new(),
            performance_history: Vec::new(),
            strategy_history: Vec::new(),
            adjustment_history: Vec::new(),
            current_strategy: None,
            last_adjustment_time: None,
            bottleneck_history: Vec::new(),
            performance_baseline: None,
        }
    }

    /// 更新性能指标
    pub fn update_performance_metrics(&mut self, metrics: PerformanceMetrics) {
        // 添加到历史记录
        self.performance_history.push(metrics.clone());

        // 保持历史记录大小在评估窗口内
        if self.performance_history.len() > self.config.evaluation_window_size * 2 {
            self.performance_history
                .drain(0..self.performance_history.len() - self.config.evaluation_window_size);
        }

        // 设置性能基线（如果没有）
        if self.performance_baseline.is_none() {
            self.performance_baseline = Some(metrics.clone());
            info!(
                "设置性能基线: 延迟={:.2}ms, 吞吐量={:.2}tps",
                metrics.latency_ms, metrics.throughput_tps
            );
        }
    }

    /// 设置当前策略
    pub fn set_current_strategy(&mut self, strategy: ParallelStrategyDecision) {
        self.current_strategy = Some(strategy.clone());
        self.strategy_history.push(strategy);

        // 保持策略历史记录大小
        if self.strategy_history.len() > 20 {
            self.strategy_history.drain(0..10);
        }
    }

    /// 评估是否需要调整
    pub fn evaluate_and_adjust(
        &mut self,
        metrics: &PerformanceMetrics,
        device_topology: &DeviceTopology,
    ) -> Result<Option<AdjustmentDecision>, String> {
        // 检查调整频率限制
        if let Some(last_time) = self.last_adjustment_time {
            let elapsed = last_time.elapsed().map(|d| d.as_secs()).unwrap_or(u64::MAX);

            if elapsed < self.config.max_adjustment_frequency_sec {
                debug!("调整频率限制，上次调整 {} 秒前", elapsed);
                return Ok(None);
            }
        }

        // 检查自动调整是否启用
        if !self.config.auto_adjustment_enabled {
            return Ok(None);
        }

        // 更新性能指标
        self.update_performance_metrics(metrics.clone());

        // 分析瓶颈
        let bottleneck = self.analyze_bottleneck(metrics);

        // 检查是否满足SLA
        let meets_sla = metrics.meets_sla(
            self.config.target_latency_ms,
            self.config.min_throughput_tps,
        );

        // 检查性能下降
        let performance_degraded = self.check_performance_degradation(metrics);

        // 如果有瓶颈或性能下降，考虑调整
        if bottleneck.severity > 0.5 || performance_degraded || !meets_sla {
            info!(
                "检测到调整需求: 瓶颈={:?}({:.2}), SLA={}, 性能下降={}",
                bottleneck.bottleneck_type, bottleneck.severity, meets_sla, performance_degraded
            );

            let decision = self.generate_adjustment_decision(
                metrics,
                device_topology,
                &bottleneck,
                !meets_sla,
                performance_degraded,
            )?;

            if let Some(decision) = decision {
                // 记录调整时间
                self.last_adjustment_time = Some(SystemTime::now());
                self.adjustment_history.push(decision.clone());

                return Ok(Some(decision));
            }
        }

        Ok(None)
    }

    /// 分析性能瓶颈
    fn analyze_bottleneck(&self, metrics: &PerformanceMetrics) -> BottleneckAnalysis {
        let mut bottleneck_type = BottleneckType::Unknown;
        let mut confidence = 0.0;
        let mut severity = 0.0;
        let mut reason = String::new();
        let mut suggested_solutions = Vec::new();

        // 分析计算瓶颈
        if metrics.cpu_usage > 0.9 || metrics.gpu_usage.map(|u| u > 0.9).unwrap_or(false) {
            bottleneck_type = BottleneckType::Compute;
            confidence = 0.8;
            severity = metrics.cpu_usage.max(metrics.gpu_usage.unwrap_or(0.0));
            reason = format!(
                "计算资源使用率过高: CPU={:.1}%, GPU={:.1}%",
                metrics.cpu_usage * 100.0,
                metrics.gpu_usage.unwrap_or(0.0) * 100.0
            );
            suggested_solutions.push("增加计算资源".to_string());
            suggested_solutions.push("优化计算密集型操作".to_string());
            suggested_solutions.push("考虑使用张量并行".to_string());
        }
        // 分析内存瓶颈
        else if metrics.memory_usage_gb > 16.0 {
            // 假设16GB为阈值
            bottleneck_type = BottleneckType::Memory;
            confidence = 0.7;
            severity = (metrics.memory_usage_gb / 32.0).min(1.0); // 假设32GB为最大
            reason = format!("内存使用量过高: {:.1}GB", metrics.memory_usage_gb);
            suggested_solutions.push("减少批次大小".to_string());
            suggested_solutions.push("启用梯度检查点".to_string());
            suggested_solutions.push("考虑使用流水线并行".to_string());
        }
        // 分析网络瓶颈
        else if metrics.network_bandwidth_usage > 0.8 {
            bottleneck_type = BottleneckType::Network;
            confidence = 0.6;
            severity = metrics.network_bandwidth_usage;
            reason = format!(
                "网络带宽使用率过高: {:.1}%",
                metrics.network_bandwidth_usage * 100.0
            );
            suggested_solutions.push("优化通信模式".to_string());
            suggested_solutions.push("减少通信频率".to_string());
            suggested_solutions.push("考虑使用混合并行".to_string());
        }
        // 分析策略瓶颈
        else if let Some(current_strategy) = &self.current_strategy {
            let predicted_latency = current_strategy.predicted_latency_ms;
            let actual_latency = metrics.latency_ms;

            if actual_latency > predicted_latency * 1.5 {
                // 实际延迟比预测高50%
                bottleneck_type = BottleneckType::Strategy;
                confidence = 0.75;
                severity = (actual_latency / predicted_latency - 1.0).min(1.0);
                reason = format!(
                    "策略性能不达预期: 预测={:.2}ms, 实际={:.2}ms",
                    predicted_latency, actual_latency
                );
                suggested_solutions.push("重新评估并行策略".to_string());
                suggested_solutions.push("调整策略参数".to_string());
            }
        }

        // 如果没有检测到明显瓶颈，检查整体性能
        if bottleneck_type == BottleneckType::Unknown && self.performance_history.len() >= 5 {
            let recent_performance: Vec<f32> = self
                .performance_history
                .iter()
                .rev()
                .take(5)
                .map(|m| m.performance_score())
                .collect();

            let avg_recent: f32 =
                recent_performance.iter().sum::<f32>() / recent_performance.len() as f32;

            if let Some(baseline) = &self.performance_baseline {
                let baseline_score = baseline.performance_score();

                if avg_recent < baseline_score * 0.8 {
                    // 性能下降20%
                    bottleneck_type = BottleneckType::Unknown;
                    confidence = 0.5;
                    severity = 1.0 - (avg_recent / baseline_score);
                    reason = format!(
                        "整体性能下降: 基线={:.1}, 近期={:.1}",
                        baseline_score, avg_recent
                    );
                    suggested_solutions.push("全面性能分析".to_string());
                    suggested_solutions.push("系统健康检查".to_string());
                }
            }
        }

        BottleneckAnalysis {
            bottleneck_type,
            confidence,
            severity,
            reason,
            suggested_solutions,
        }
    }

    /// 检查性能下降
    fn check_performance_degradation(&self, metrics: &PerformanceMetrics) -> bool {
        if let Some(baseline) = &self.performance_baseline {
            let baseline_score = baseline.performance_score();
            let current_score = metrics.performance_score();

            let degradation = if baseline_score > 0.0 {
                (baseline_score - current_score) / baseline_score * 100.0
            } else {
                0.0
            };

            degradation > self.config.performance_degradation_threshold
        } else {
            false
        }
    }

    /// 生成调整决策
    fn generate_adjustment_decision(
        &mut self,
        metrics: &PerformanceMetrics,
        device_topology: &DeviceTopology,
        bottleneck: &BottleneckAnalysis,
        sla_violation: bool,
        performance_degraded: bool,
    ) -> Result<Option<AdjustmentDecision>, String> {
        // 根据瓶颈类型生成决策
        match bottleneck.bottleneck_type {
            BottleneckType::Compute => {
                // 计算瓶颈：考虑张量并行或增加计算资源
                if self.config.strategy_switching_enabled {
                    self.generate_compute_bottleneck_decision(metrics, device_topology, bottleneck)
                } else {
                    Ok(None)
                }
            }

            BottleneckType::Memory => {
                // 内存瓶颈：考虑流水线并行或减少批次大小
                if self.config.strategy_switching_enabled || self.config.parameter_tuning_enabled {
                    self.generate_memory_bottleneck_decision(metrics, device_topology, bottleneck)
                } else {
                    Ok(None)
                }
            }

            BottleneckType::Network => {
                // 网络瓶颈：优化通信或调整并行策略
                if self.config.parameter_tuning_enabled {
                    self.generate_network_bottleneck_decision(metrics, device_topology, bottleneck)
                } else {
                    Ok(None)
                }
            }

            BottleneckType::Strategy => {
                // 策略瓶颈：重新评估策略
                if self.config.strategy_switching_enabled {
                    self.generate_strategy_bottleneck_decision(metrics, device_topology, bottleneck)
                } else {
                    Ok(None)
                }
            }

            _ => {
                // 未知瓶颈或SLA违规：全面重新评估
                if sla_violation || performance_degraded {
                    self.generate_sla_violation_decision(metrics, device_topology, bottleneck)
                } else {
                    Ok(None)
                }
            }
        }
    }

    /// 生成计算瓶颈决策
    fn generate_compute_bottleneck_decision(
        &mut self,
        metrics: &PerformanceMetrics,
        device_topology: &DeviceTopology,
        bottleneck: &BottleneckAnalysis,
    ) -> Result<Option<AdjustmentDecision>, String> {
        // 检查当前策略是否为张量并行
        let is_tensor_parallel = self
            .current_strategy
            .as_ref()
            .map(|s| s.strategy == ParallelStrategy::TensorParallel)
            .unwrap_or(false);

        if !is_tensor_parallel {
            // 尝试切换到张量并行
            match self.strategy_engine.select_optimal_strategy(
                device_topology,
                metrics.model_size_gb,
                metrics.batch_size,
                Some(self.config.target_latency_ms),
            ) {
                Ok(new_strategy) => {
                    // 检查新策略是否为张量并行
                    if new_strategy.strategy == ParallelStrategy::TensorParallel {
                        let improvement = self.estimate_improvement(
                            self.current_strategy.as_ref(),
                            &new_strategy,
                            metrics,
                        );

                        if improvement >= self.config.min_improvement_threshold {
                            return Ok(Some(AdjustmentDecision::StrategyChange {
                                new_strategy,
                                reason: format!("计算瓶颈: {}", bottleneck.reason),
                                expected_improvement: improvement,
                                migration_cost: 0.3,
                            }));
                        }
                    }
                }
                Err(e) => warn!("选择张量并行策略失败: {}", e),
            }
        }

        // 如果已经是张量并行，尝试增加并行度
        if is_tensor_parallel && self.config.parameter_tuning_enabled {
            if let Some(strategy) = self.current_strategy.as_ref() {
                let current_tp_degree = strategy.config.tp_degree;
                let max_possible_tp_degree = device_topology.device_count();

                if current_tp_degree < max_possible_tp_degree {
                    let mut new_config = strategy.config.clone();
                    new_config.tp_degree = (current_tp_degree * 2).min(max_possible_tp_degree);

                    if new_config.tp_degree > current_tp_degree {
                        return Ok(Some(AdjustmentDecision::ParameterTuning {
                            new_config,
                            reason: format!("增加张量并行度以缓解计算瓶颈: {}", bottleneck.reason),
                            expected_improvement: 15.0, // 粗略估计
                        }));
                    }
                }
            }
        }

        Ok(None)
    }

    /// 生成内存瓶颈决策
    fn generate_memory_bottleneck_decision(
        &mut self,
        metrics: &PerformanceMetrics,
        device_topology: &DeviceTopology,
        bottleneck: &BottleneckAnalysis,
    ) -> Result<Option<AdjustmentDecision>, String> {
        // 检查当前策略是否为流水线并行
        let is_pipeline_parallel = self
            .current_strategy
            .as_ref()
            .map(|s| s.strategy == ParallelStrategy::PipelineParallel)
            .unwrap_or(false);

        // 首先尝试参数调整：减少批次大小或启用检查点
        if self.config.parameter_tuning_enabled {
            if let Some(strategy) = self.current_strategy.as_ref() {
                let mut new_config = strategy.config.clone();
                let mut changes = Vec::new();

                // 减少批次大小（如果大于1）
                if new_config.micro_batch_size > 1 {
                    new_config.micro_batch_size = (new_config.micro_batch_size / 2).max(1);
                    changes.push(format!("微批次大小减少到 {}", new_config.micro_batch_size));
                }

                // 启用梯度检查点
                if !new_config.gradient_checkpointing {
                    new_config.gradient_checkpointing = true;
                    changes.push("启用梯度检查点".to_string());
                }

                // 启用激活检查点
                if !new_config.activation_checkpointing {
                    new_config.activation_checkpointing = true;
                    changes.push("启用激活检查点".to_string());
                }

                if !changes.is_empty() {
                    return Ok(Some(AdjustmentDecision::ParameterTuning {
                        new_config,
                        reason: format!(
                            "内存优化: {}; 调整: {}",
                            bottleneck.reason,
                            changes.join(", ")
                        ),
                        expected_improvement: 20.0, // 粗略估计
                    }));
                }
            }
        }

        // 尝试切换到流水线并行
        if !is_pipeline_parallel && self.config.strategy_switching_enabled {
            match self.strategy_engine.select_optimal_strategy(
                device_topology,
                metrics.model_size_gb,
                metrics.batch_size,
                Some(self.config.target_latency_ms),
            ) {
                Ok(new_strategy) => {
                    // 检查新策略是否为流水线并行
                    if new_strategy.strategy == ParallelStrategy::PipelineParallel {
                        let improvement = self.estimate_improvement(
                            self.current_strategy.as_ref(),
                            &new_strategy,
                            metrics,
                        );

                        if improvement >= self.config.min_improvement_threshold {
                            return Ok(Some(AdjustmentDecision::StrategyChange {
                                new_strategy,
                                reason: format!("内存瓶颈: {}", bottleneck.reason),
                                expected_improvement: improvement,
                                migration_cost: 0.4,
                            }));
                        }
                    }
                }
                Err(e) => warn!("选择流水线并行策略失败: {}", e),
            }
        }

        Ok(None)
    }

    /// 生成网络瓶颈决策
    fn generate_network_bottleneck_decision(
        &mut self,
        _metrics: &PerformanceMetrics,
        _device_topology: &DeviceTopology,
        bottleneck: &BottleneckAnalysis,
    ) -> Result<Option<AdjustmentDecision>, String> {
        // 网络瓶颈优化：调整通信参数
        if self.config.parameter_tuning_enabled {
            if let Some(strategy) = self.current_strategy.as_ref() {
                let mut new_config = strategy.config.clone();

                // 增加通信优化级别
                if new_config.communication_optimization_level < 3 {
                    new_config.communication_optimization_level += 1;

                    return Ok(Some(AdjustmentDecision::ParameterTuning {
                        new_config,
                        reason: format!("网络优化: {}", bottleneck.reason),
                        expected_improvement: 10.0, // 粗略估计
                    }));
                }
            }
        }

        Ok(None)
    }

    /// 生成策略瓶颈决策
    fn generate_strategy_bottleneck_decision(
        &mut self,
        metrics: &PerformanceMetrics,
        device_topology: &DeviceTopology,
        bottleneck: &BottleneckAnalysis,
    ) -> Result<Option<AdjustmentDecision>, String> {
        if self.config.strategy_switching_enabled {
            // 全面重新评估策略
            match self.strategy_engine.select_optimal_strategy(
                device_topology,
                metrics.model_size_gb,
                metrics.batch_size,
                Some(self.config.target_latency_ms * 0.8), // 更严格的目标
            ) {
                Ok(new_strategy) => {
                    // 检查新策略是否与当前策略不同
                    let is_different = self
                        .current_strategy
                        .as_ref()
                        .map(|s| {
                            s.strategy != new_strategy.strategy
                                || s.config.tp_degree != new_strategy.config.tp_degree
                                || s.config.pp_degree != new_strategy.config.pp_degree
                        })
                        .unwrap_or(true);

                    if is_different {
                        let improvement = self.estimate_improvement(
                            self.current_strategy.as_ref(),
                            &new_strategy,
                            metrics,
                        );

                        if improvement >= self.config.min_improvement_threshold {
                            return Ok(Some(AdjustmentDecision::StrategyChange {
                                new_strategy,
                                reason: format!("策略优化: {}", bottleneck.reason),
                                expected_improvement: improvement,
                                migration_cost: 0.5,
                            }));
                        }
                    }
                }
                Err(e) => warn!("重新评估策略失败: {}", e),
            }
        }

        Ok(None)
    }

    /// 生成SLA违规决策
    fn generate_sla_violation_decision(
        &mut self,
        metrics: &PerformanceMetrics,
        device_topology: &DeviceTopology,
        bottleneck: &BottleneckAnalysis,
    ) -> Result<Option<AdjustmentDecision>, String> {
        // SLA违规：尝试找到任何能满足SLA的策略
        match self.strategy_engine.select_optimal_strategy(
            device_topology,
            metrics.model_size_gb,
            metrics.batch_size,
            Some(self.config.target_latency_ms),
        ) {
            Ok(new_strategy) => {
                // 检查新策略是否能满足SLA
                if new_strategy.predicted_latency_ms <= self.config.target_latency_ms {
                    return Ok(Some(AdjustmentDecision::StrategyChange {
                        new_strategy,
                        reason: format!("SLA违规: {}", bottleneck.reason),
                        expected_improvement: 30.0, // 粗略估计
                        migration_cost: 0.6,
                    }));
                }
            }
            Err(e) => warn!("寻找SLA兼容策略失败: {}", e),
        }

        Ok(None)
    }

    /// 估计改进百分比
    fn estimate_improvement(
        &self,
        current_strategy: Option<&ParallelStrategyDecision>,
        new_strategy: &ParallelStrategyDecision,
        metrics: &PerformanceMetrics,
    ) -> f32 {
        if let Some(current) = current_strategy {
            let current_score = if current.predicted_latency_ms > 0.0 {
                1000.0 / current.predicted_latency_ms
            } else {
                0.0
            };

            let new_score = if new_strategy.predicted_latency_ms > 0.0 {
                1000.0 / new_strategy.predicted_latency_ms
            } else {
                0.0
            };

            if current_score > 0.0 {
                ((new_score - current_score) / current_score * 100.0_f32).max(0.0_f32)
            } else {
                0.0
            }
        } else {
            // 没有当前策略，使用实际性能作为基准
            let current_score = metrics.performance_score();
            let new_score = if new_strategy.predicted_latency_ms > 0.0 {
                1000.0 / new_strategy.predicted_latency_ms
            } else {
                0.0
            };

            if current_score > 0.0 {
                ((new_score - current_score) / current_score * 100.0).max(0.0)
            } else {
                0.0
            }
        }
    }

    /// 获取调整历史记录
    pub fn get_adjustment_history(&self) -> &[AdjustmentDecision] {
        &self.adjustment_history
    }

    /// 获取性能历史记录
    pub fn get_performance_history(&self) -> &[PerformanceMetrics] {
        &self.performance_history
    }

    /// 获取当前策略
    pub fn get_current_strategy(&self) -> Option<&ParallelStrategyDecision> {
        self.current_strategy.as_ref()
    }

    /// 获取配置
    pub fn get_config(&self) -> &AdjusterConfig {
        &self.config
    }

    /// 更新配置
    pub fn update_config(&mut self, config: AdjusterConfig) {
        self.config = config;
    }

    /// 重置性能基线
    pub fn reset_baseline(&mut self) {
        self.performance_baseline = None;
        info!("性能基线已重置");
    }

    /// 执行回滚
    pub fn perform_rollback(&mut self) -> Result<Option<AdjustmentDecision>, String> {
        if !self.config.rollback_enabled {
            return Err("回滚机制未启用".to_string());
        }

        // 查找之前的策略
        if self.strategy_history.len() >= 2 {
            let previous_strategy = self.strategy_history[self.strategy_history.len() - 2].clone();

            return Ok(Some(AdjustmentDecision::Rollback {
                previous_strategy,
                reason: "手动触发回滚".to_string(),
            }));
        }

        Err("没有可回滚的策略历史记录".to_string())
    }
}

// ==================== 模块导出 ====================
// 类型已经在顶层定义为pub，不需要重复导出
