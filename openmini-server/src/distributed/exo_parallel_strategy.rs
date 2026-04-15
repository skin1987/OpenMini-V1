//! EXO自动并行策略引擎
//!
//! 基于EXO架构的拓扑感知自动并行策略选择系统，支持：
//! - 自动并行策略选择（张量并行、流水线并行、混合并行）
//! - 拓扑感知模型切分和资源分配
//! - 实时性能预测和优化
//! - 动态策略调整和故障恢复
//!
//! # 架构概述
//!
//! ```text
//! ExoParallelStrategyEngine
//! ├── TopologyAnalyzer (拓扑分析器)
//! │   ├── DeviceCapabilityProfiler (设备能力分析)
//! │   ├── NetworkLatencyProfiler (网络延迟分析)
//! │   └── ResourceAvailabilityChecker (资源可用性检查)
//! ├── StrategySelector (策略选择器)
//! │   ├── TensorParallelOptimizer (张量并行优化器)
//! │   ├── PipelineParallelOptimizer (流水线并行优化器)
//! │   └── HybridParallelOptimizer (混合并行优化器)
//! ├── PerformancePredictor (性能预测器)
//! │   ├── LatencyEstimator (延迟估计器)
//! │   ├── ThroughputEstimator (吞吐量估计器)
//! │   └── MemoryUsageEstimator (内存使用估计器)
//! └── DynamicAdjuster (动态调整器)
//!     ├── LoadMonitor (负载监控器)
//!     └── StrategyReevaluator (策略重评估器)
//! ```
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use openmini_server::model::inference::exo_parallel_strategy::{
//!     ExoParallelStrategyEngine, ParallelStrategyDecision, DeviceTopology
//! };
//!
//! // 创建设备拓扑信息
//! let topology = DeviceTopology::from_devices(devices);
//!
//! // 创建并行策略引擎
//! let strategy_engine = ExoParallelStrategyEngine::new();
//!
//! // 根据拓扑选择最优并行策略
//! let decision = strategy_engine.select_optimal_strategy(
//!     &topology,
//!     model_size_gb,
//!     batch_size,
//!     latency_budget_ms,
//! )?;
//!
//! println!("推荐策略: {:?}", decision.strategy);
//! println!("预计延迟: {:.2}ms", decision.predicted_latency_ms);
//! ```

use std::collections::HashMap;

use log::{info, warn};

use super::discovery::{DeviceInfo, DeviceResources};
use super::exo_communication::{DeviceCapabilities, NetworkTopology};
use crate::model::inference::distributed_inference_config::ParallelStrategy;

// ==================== 核心类型定义 ====================

/// 设备拓扑信息
#[derive(Debug, Clone)]
pub struct DeviceTopology {
    /// 设备列表
    pub devices: Vec<DeviceInfo>,

    /// 设备能力映射
    pub capabilities: HashMap<String, DeviceCapabilities>,

    /// 设备资源映射
    pub resources: HashMap<String, DeviceResources>,

    /// 网络拓扑
    pub network_topology: NetworkTopology,

    /// 设备间延迟矩阵 (纳秒)
    pub latency_matrix: HashMap<(String, String), u64>,

    /// 设备间带宽矩阵 (Gbps)
    pub bandwidth_matrix: HashMap<(String, String), f32>,
}

impl DeviceTopology {
    /// 从设备列表创建设备拓扑
    pub fn from_devices(devices: Vec<DeviceInfo>) -> Self {
        let capabilities: HashMap<String, DeviceCapabilities> = devices
            .iter()
            .map(|device| (device.device_id.clone(), device.capabilities.clone()))
            .collect();

        let resources: HashMap<String, DeviceResources> = devices
            .iter()
            .map(|device| (device.device_id.clone(), device.resources.clone()))
            .collect();

        // 简化网络拓扑（实际应从网络拓扑管理器获取）
        let network_topology = NetworkTopology {
            devices: Vec::new(),
            links: Vec::new(),
            updated_at: std::time::SystemTime::now(),
            version: 1,
        };

        // 简化延迟和带宽矩阵（实际应测量）
        let mut latency_matrix = HashMap::new();
        let mut bandwidth_matrix = HashMap::new();

        for i in 0..devices.len() {
            for j in 0..devices.len() {
                let device_i = &devices[i].device_id;
                let device_j = &devices[j].device_id;

                if i == j {
                    latency_matrix.insert((device_i.clone(), device_j.clone()), 0);
                    bandwidth_matrix.insert((device_i.clone(), device_j.clone()), f32::INFINITY);
                } else {
                    // 简化估计：基于设备类型和位置
                    let latency = estimate_latency(&devices[i], &devices[j]);
                    let bandwidth = estimate_bandwidth(&devices[i], &devices[j]);

                    latency_matrix.insert((device_i.clone(), device_j.clone()), latency);
                    bandwidth_matrix.insert((device_i.clone(), device_j.clone()), bandwidth);
                }
            }
        }

        Self {
            devices,
            capabilities,
            resources,
            network_topology,
            latency_matrix,
            bandwidth_matrix,
        }
    }

    /// 获取设备数量
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// 获取设备ID列表
    pub fn device_ids(&self) -> Vec<String> {
        self.devices.iter().map(|d| d.device_id.clone()).collect()
    }

    /// 获取两个设备间的延迟
    pub fn get_latency(&self, device_a: &str, device_b: &str) -> u64 {
        *self
            .latency_matrix
            .get(&(device_a.to_string(), device_b.to_string()))
            .or_else(|| {
                self.latency_matrix
                    .get(&(device_b.to_string(), device_a.to_string()))
            })
            .unwrap_or(&1_000_000) // 默认1毫秒
    }

    /// 获取两个设备间的带宽
    pub fn get_bandwidth(&self, device_a: &str, device_b: &str) -> f32 {
        *self
            .bandwidth_matrix
            .get(&(device_a.to_string(), device_b.to_string()))
            .or_else(|| {
                self.bandwidth_matrix
                    .get(&(device_b.to_string(), device_a.to_string()))
            })
            .unwrap_or(&1.0) // 默认1 Gbps
    }

    /// 获取设备计算能力评分
    pub fn get_compute_score(&self, device_id: &str) -> u8 {
        self.capabilities
            .get(device_id)
            .map(|cap| cap.compute_score)
            .unwrap_or(50)
    }

    /// 获取设备总内存
    pub fn get_total_memory_gb(&self, device_id: &str) -> f32 {
        self.resources
            .get(device_id)
            .map(|res| res.total_memory_bytes as f32 / (1024.0 * 1024.0 * 1024.0))
            .unwrap_or(8.0) // 默认8GB
    }

    /// 获取设备是否支持RDMA
    pub fn supports_rdma(&self, device_id: &str) -> bool {
        self.capabilities
            .get(device_id)
            .map(|cap| cap.supports_rdma)
            .unwrap_or(false)
    }

    /// 获取设备是否支持MLX
    pub fn supports_mlx(&self, device_id: &str) -> bool {
        self.capabilities
            .get(device_id)
            .map(|cap| cap.supports_mlx)
            .unwrap_or(false)
    }
}

/// 并行策略决策
#[derive(Debug, Clone)]
pub struct ParallelStrategyDecision {
    /// 选择的并行策略
    pub strategy: ParallelStrategy,

    /// 策略配置参数
    pub config: StrategyConfig,

    /// 设备分配映射
    pub device_assignment: HashMap<String, DeviceRole>,

    /// 预计推理延迟 (毫秒)
    pub predicted_latency_ms: f32,

    /// 预计内存使用 (GB)
    pub predicted_memory_gb: f32,

    /// 预计吞吐量 (tokens/秒)
    pub predicted_throughput_tps: f32,

    /// 策略置信度 (0-1)
    pub confidence: f32,
}

/// 策略配置参数
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// 张量并行度 (仅适用于张量并行和混合并行)
    pub tp_degree: usize,

    /// 流水线并行度 (仅适用于流水线并行和混合并行)
    pub pp_degree: usize,

    /// 序列并行度 (仅适用于序列并行)
    pub sp_degree: usize,

    /// 微批次大小 (仅适用于流水线并行)
    pub micro_batch_size: usize,

    /// 是否启用梯度检查点 (内存优化)
    pub gradient_checkpointing: bool,

    /// 是否启用激活检查点 (内存优化)
    pub activation_checkpointing: bool,

    /// 通信优化级别 (0-3)
    pub communication_optimization_level: u8,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            tp_degree: 1,
            pp_degree: 1,
            sp_degree: 1,
            micro_batch_size: 1,
            gradient_checkpointing: false,
            activation_checkpointing: false,
            communication_optimization_level: 1,
        }
    }
}

/// 设备角色
#[derive(Debug, Clone)]
pub enum DeviceRole {
    /// 主节点 (协调节点)
    Master,

    /// 张量并行工作节点
    TensorWorker {
        tp_rank: usize,
        tp_group: Vec<String>,
    },

    /// 流水线并行工作节点
    PipelineWorker { pp_rank: usize, pp_stage: usize },

    /// 混合并行工作节点
    HybridWorker {
        tp_rank: usize,
        pp_rank: usize,
        tp_group: Vec<String>,
    },

    /// 序列并行工作节点
    SequenceWorker {
        sp_rank: usize,
        sequence_chunk: usize,
    },
}

/// 策略选择错误
#[derive(Debug, thiserror::Error)]
pub enum StrategyError {
    #[error("不支持的设备拓扑: {0}")]
    UnsupportedTopology(String),

    #[error("资源不足: {0}")]
    InsufficientResources(String),

    #[error("性能预测失败: {0}")]
    PerformancePredictionFailed(String),

    #[error("策略配置无效: {0}")]
    InvalidConfiguration(String),

    #[error("设备发现失败: {0}")]
    DeviceDiscoveryFailed(String),
}

// ==================== EXO并行策略引擎 ====================

/// EXO并行策略引擎
pub struct ExoParallelStrategyEngine {
    /// 策略历史记录
    strategy_history: Vec<ParallelStrategyDecision>,

    /// 性能模型
    performance_model: PerformanceModel,

    /// 是否启用实时调整
    realtime_adjustment_enabled: bool,
}

impl ExoParallelStrategyEngine {
    /// 创建新的并行策略引擎
    pub fn new() -> Self {
        Self {
            strategy_history: Vec::new(),
            performance_model: PerformanceModel::new(),
            realtime_adjustment_enabled: true,
        }
    }

    /// 选择最优并行策略
    pub fn select_optimal_strategy(
        &mut self,
        topology: &DeviceTopology,
        model_size_gb: f32,
        batch_size: usize,
        latency_budget_ms: Option<f32>,
    ) -> Result<ParallelStrategyDecision, StrategyError> {
        info!(
            "开始选择并行策略，设备数: {}, 模型大小: {:.2}GB",
            topology.device_count(),
            model_size_gb
        );

        // 检查设备数量
        if topology.device_count() == 0 {
            return Err(StrategyError::DeviceDiscoveryFailed(
                "未发现任何设备".to_string(),
            ));
        }

        // 评估每种可行策略
        let mut candidate_decisions = Vec::new();

        // 评估张量并行
        if let Ok(decision) = self.evaluate_tensor_parallel(topology, model_size_gb, batch_size) {
            candidate_decisions.push(decision);
        }

        // 评估流水线并行
        if let Ok(decision) = self.evaluate_pipeline_parallel(topology, model_size_gb, batch_size) {
            candidate_decisions.push(decision);
        }

        // 评估混合并行
        if let Ok(decision) = self.evaluate_hybrid_parallel(topology, model_size_gb, batch_size) {
            candidate_decisions.push(decision);
        }

        // 评估序列并行
        if let Ok(decision) = self.evaluate_sequence_parallel(topology, model_size_gb, batch_size) {
            candidate_decisions.push(decision);
        }

        if candidate_decisions.is_empty() {
            return Err(StrategyError::UnsupportedTopology(
                "没有找到适合当前拓扑的并行策略".to_string(),
            ));
        }

        // 根据延迟预算过滤
        let filtered_decisions: Vec<ParallelStrategyDecision> =
            if let Some(budget) = latency_budget_ms {
                candidate_decisions
                    .iter()
                    .filter(|d| d.predicted_latency_ms <= budget)
                    .cloned()
                    .collect()
            } else {
                candidate_decisions.clone()
            };

        if filtered_decisions.is_empty() {
            warn!("没有策略满足延迟预算 {:.2}ms", latency_budget_ms.unwrap());
            // 返回延迟最小的策略
            let best_decision = candidate_decisions
                .iter()
                .min_by(|a, b| {
                    a.predicted_latency_ms
                        .partial_cmp(&b.predicted_latency_ms)
                        .unwrap()
                })
                .cloned()
                .unwrap();

            return Ok(best_decision);
        }

        // 选择最优策略（延迟最小，置信度最高）
        let best_decision = filtered_decisions
            .into_iter()
            .max_by(|a, b| {
                // 首先比较置信度，然后比较延迟
                let confidence_cmp = a.confidence.partial_cmp(&b.confidence).unwrap();
                if confidence_cmp == std::cmp::Ordering::Equal {
                    b.predicted_latency_ms
                        .partial_cmp(&a.predicted_latency_ms)
                        .unwrap()
                } else {
                    confidence_cmp
                }
            })
            .unwrap();

        // 保存到历史记录
        self.strategy_history.push(best_decision.clone());

        info!(
            "选择策略: {:?}, 预计延迟: {:.2}ms, 置信度: {:.2}",
            best_decision.strategy, best_decision.predicted_latency_ms, best_decision.confidence
        );

        Ok(best_decision)
    }

    /// 评估张量并行策略
    fn evaluate_tensor_parallel(
        &self,
        topology: &DeviceTopology,
        model_size_gb: f32,
        batch_size: usize,
    ) -> Result<ParallelStrategyDecision, StrategyError> {
        let device_count = topology.device_count();

        // 张量并行度必须是2的幂且不超过设备数量
        let max_tp_degree = device_count.next_power_of_two().min(device_count);

        if max_tp_degree < 2 {
            return Err(StrategyError::UnsupportedTopology(
                "张量并行需要至少2个设备".to_string(),
            ));
        }

        // 计算最优TP度数（基于设备计算能力和内存）
        let optimal_tp_degree = self.calculate_optimal_tp_degree(topology, model_size_gb);

        if optimal_tp_degree < 2 {
            return Err(StrategyError::UnsupportedTopology(
                "计算出的最优张量并行度小于2".to_string(),
            ));
        }

        // 创建设备分配
        let mut device_assignment = HashMap::new();
        let device_ids = topology.device_ids();

        // 选择计算能力最强的设备作为主节点
        let master_device = device_ids
            .iter()
            .max_by_key(|id| topology.get_compute_score(id))
            .unwrap()
            .clone();

        device_assignment.insert(master_device.clone(), DeviceRole::Master);

        // 为TP组分配设备
        let tp_group: Vec<String> = device_ids
            .iter()
            .filter(|id| **id != master_device)
            .take(optimal_tp_degree - 1)
            .cloned()
            .collect();

        for (i, device_id) in tp_group.iter().enumerate() {
            device_assignment.insert(
                device_id.clone(),
                DeviceRole::TensorWorker {
                    tp_rank: i + 1,
                    tp_group: tp_group.clone(),
                },
            );
        }

        // 预测性能
        let predicted_latency = self.performance_model.predict_tensor_parallel_latency(
            topology,
            model_size_gb,
            batch_size,
            optimal_tp_degree,
        );

        let predicted_memory = model_size_gb / optimal_tp_degree as f32;

        // 计算置信度
        let confidence = self.calculate_tp_confidence(topology, optimal_tp_degree);

        let decision = ParallelStrategyDecision {
            strategy: ParallelStrategy::TensorParallel,
            config: StrategyConfig {
                tp_degree: optimal_tp_degree,
                pp_degree: 1,
                sp_degree: 1,
                micro_batch_size: 1,
                gradient_checkpointing: false,
                activation_checkpointing: false,
                communication_optimization_level: 2,
            },
            device_assignment,
            predicted_latency_ms: predicted_latency,
            predicted_memory_gb: predicted_memory,
            predicted_throughput_tps: (batch_size as f32 * 1000.0) / predicted_latency.max(1.0),
            confidence,
        };

        Ok(decision)
    }

    /// 评估流水线并行策略
    fn evaluate_pipeline_parallel(
        &self,
        topology: &DeviceTopology,
        model_size_gb: f32,
        batch_size: usize,
    ) -> Result<ParallelStrategyDecision, StrategyError> {
        let device_count = topology.device_count();

        if device_count < 2 {
            return Err(StrategyError::UnsupportedTopology(
                "流水线并行需要至少2个设备".to_string(),
            ));
        }

        // 计算最优PP度数（基于设备内存和网络带宽）
        let optimal_pp_degree = self.calculate_optimal_pp_degree(topology, model_size_gb);

        // 创建设备分配
        let mut device_assignment = HashMap::new();
        let device_ids = topology.device_ids();

        // 选择计算能力最强的设备作为主节点
        let master_device = device_ids
            .iter()
            .max_by_key(|id| topology.get_compute_score(id))
            .unwrap()
            .clone();

        device_assignment.insert(master_device.clone(), DeviceRole::Master);

        // 为PP阶段分配设备
        let pp_devices: Vec<String> = device_ids
            .iter()
            .filter(|id| **id != master_device)
            .take(optimal_pp_degree)
            .cloned()
            .collect();

        for (i, device_id) in pp_devices.iter().enumerate() {
            device_assignment.insert(
                device_id.clone(),
                DeviceRole::PipelineWorker {
                    pp_rank: i,
                    pp_stage: i,
                },
            );
        }

        // 预测性能
        let predicted_latency = self.performance_model.predict_pipeline_parallel_latency(
            topology,
            model_size_gb,
            batch_size,
            optimal_pp_degree,
        );

        let predicted_memory = model_size_gb / optimal_pp_degree as f32;

        // 计算置信度
        let confidence = self.calculate_pp_confidence(topology, optimal_pp_degree);

        let decision = ParallelStrategyDecision {
            strategy: ParallelStrategy::PipelineParallel,
            config: StrategyConfig {
                tp_degree: 1,
                pp_degree: optimal_pp_degree,
                sp_degree: 1,
                micro_batch_size: batch_size.max(1).min(4), // 微批次大小
                gradient_checkpointing: true,
                activation_checkpointing: true,
                communication_optimization_level: 3,
            },
            device_assignment,
            predicted_latency_ms: predicted_latency,
            predicted_memory_gb: predicted_memory,
            predicted_throughput_tps: (batch_size as f32 * 1000.0) / predicted_latency.max(1.0),
            confidence,
        };

        Ok(decision)
    }

    /// 评估混合并行策略
    fn evaluate_hybrid_parallel(
        &self,
        topology: &DeviceTopology,
        model_size_gb: f32,
        batch_size: usize,
    ) -> Result<ParallelStrategyDecision, StrategyError> {
        let device_count = topology.device_count();

        if device_count < 4 {
            return Err(StrategyError::UnsupportedTopology(
                "混合并行需要至少4个设备".to_string(),
            ));
        }

        // 计算最优TP和PP度数
        let optimal_tp_degree = self.calculate_optimal_tp_degree(topology, model_size_gb / 2.0);
        let optimal_pp_degree = self.calculate_optimal_pp_degree(topology, model_size_gb / 2.0);

        let total_devices_needed = optimal_tp_degree * optimal_pp_degree;

        if total_devices_needed > device_count {
            return Err(StrategyError::InsufficientResources(format!(
                "混合并行需要{}个设备，但只有{}个",
                total_devices_needed, device_count
            )));
        }

        // 简化的设备分配（实际应更智能）
        let mut device_assignment = HashMap::new();
        let device_ids = topology.device_ids();

        // 选择主节点
        let master_device = device_ids[0].clone();
        device_assignment.insert(master_device, DeviceRole::Master);

        // 分配TP组和PP阶段
        let mut device_index = 1;

        for pp_rank in 0..optimal_pp_degree {
            let mut tp_group = Vec::new();

            for tp_rank in 0..optimal_tp_degree {
                if device_index < device_ids.len() {
                    let device_id = device_ids[device_index].clone();

                    device_assignment.insert(
                        device_id.clone(),
                        DeviceRole::HybridWorker {
                            tp_rank,
                            pp_rank,
                            tp_group: Vec::new(), // 将在后续填充
                        },
                    );

                    tp_group.push(device_id);
                    device_index += 1;
                }
            }

            // 更新TP组信息
            for device_id in &tp_group {
                if let Some(DeviceRole::HybridWorker {
                    tp_rank, pp_rank, ..
                }) = device_assignment.get_mut(device_id)
                {
                    *device_assignment.get_mut(device_id).unwrap() = DeviceRole::HybridWorker {
                        tp_rank: *tp_rank,
                        pp_rank: *pp_rank,
                        tp_group: tp_group.clone(),
                    };
                }
            }
        }

        // 预测性能
        let predicted_latency = self.performance_model.predict_hybrid_parallel_latency(
            topology,
            model_size_gb,
            batch_size,
            optimal_tp_degree,
            optimal_pp_degree,
        );

        let predicted_memory = model_size_gb / (optimal_tp_degree * optimal_pp_degree) as f32;

        // 计算置信度
        let confidence =
            self.calculate_hybrid_confidence(topology, optimal_tp_degree, optimal_pp_degree);

        let decision = ParallelStrategyDecision {
            strategy: ParallelStrategy::HybridParallel,
            config: StrategyConfig {
                tp_degree: optimal_tp_degree,
                pp_degree: optimal_pp_degree,
                sp_degree: 1,
                micro_batch_size: batch_size.max(1).min(2),
                gradient_checkpointing: true,
                activation_checkpointing: true,
                communication_optimization_level: 3,
            },
            device_assignment,
            predicted_latency_ms: predicted_latency,
            predicted_memory_gb: predicted_memory,
            predicted_throughput_tps: (batch_size as f32 * 1000.0) / predicted_latency.max(1.0),
            confidence,
        };

        Ok(decision)
    }

    /// 评估序列并行策略
    fn evaluate_sequence_parallel(
        &self,
        topology: &DeviceTopology,
        model_size_gb: f32,
        batch_size: usize,
    ) -> Result<ParallelStrategyDecision, StrategyError> {
        let device_count = topology.device_count();

        if device_count < 2 {
            return Err(StrategyError::UnsupportedTopology(
                "序列并行需要至少2个设备".to_string(),
            ));
        }

        // 序列并行度受限于批次大小和设备数量
        let optimal_sp_degree = device_count.min(batch_size);

        // 简化的设备分配
        let mut device_assignment = HashMap::new();
        let device_ids = topology.device_ids();

        let master_device = device_ids[0].clone();
        device_assignment.insert(master_device, DeviceRole::Master);

        for (i, device_id) in device_ids
            .iter()
            .skip(1)
            .take(optimal_sp_degree)
            .enumerate()
        {
            device_assignment.insert(
                device_id.clone(),
                DeviceRole::SequenceWorker {
                    sp_rank: i,
                    sequence_chunk: batch_size / optimal_sp_degree,
                },
            );
        }

        // 预测性能
        let predicted_latency = self.performance_model.predict_sequence_parallel_latency(
            topology,
            model_size_gb,
            batch_size,
            optimal_sp_degree,
        );

        let predicted_memory = model_size_gb;

        // 计算置信度
        let confidence = self.calculate_sp_confidence(topology, optimal_sp_degree);

        let decision = ParallelStrategyDecision {
            strategy: ParallelStrategy::SequenceParallel,
            config: StrategyConfig {
                tp_degree: 1,
                pp_degree: 1,
                sp_degree: optimal_sp_degree,
                micro_batch_size: 1,
                gradient_checkpointing: false,
                activation_checkpointing: false,
                communication_optimization_level: 1,
            },
            device_assignment,
            predicted_latency_ms: predicted_latency,
            predicted_memory_gb: predicted_memory,
            predicted_throughput_tps: (batch_size as f32 * 1000.0) / predicted_latency.max(1.0),
            confidence,
        };

        Ok(decision)
    }

    /// 计算最优张量并行度
    fn calculate_optimal_tp_degree(&self, topology: &DeviceTopology, model_size_gb: f32) -> usize {
        let device_count = topology.device_count();

        // 简单启发式：基于设备内存和计算能力
        let mut candidate_degrees = Vec::new();

        for degree in 2..=device_count {
            if device_count % degree == 0 {
                // 要求设备数量可整除
                // 检查设备内存是否足够
                let memory_per_device = model_size_gb / degree as f32;
                let mut sufficient_memory = true;

                for device_id in topology.device_ids() {
                    let device_memory_gb = topology.get_total_memory_gb(&device_id);
                    if device_memory_gb < memory_per_device * 1.2 {
                        // 20% 余量
                        sufficient_memory = false;
                        break;
                    }
                }

                if sufficient_memory {
                    candidate_degrees.push(degree);
                }
            }
        }

        // 选择最大的可行度数
        candidate_degrees.into_iter().max().unwrap_or(1)
    }

    /// 计算最优流水线并行度
    fn calculate_optimal_pp_degree(&self, topology: &DeviceTopology, model_size_gb: f32) -> usize {
        let device_count = topology.device_count();

        // 简单启发式：基于设备内存和网络带宽
        let mut candidate_degrees = Vec::new();

        for degree in 2..=device_count {
            let memory_per_device = model_size_gb / degree as f32;
            let mut feasible = true;

            // 检查设备内存
            for device_id in topology.device_ids() {
                let device_memory_gb = topology.get_total_memory_gb(&device_id);
                if device_memory_gb < memory_per_device * 1.3 {
                    // 30% 余量
                    feasible = false;
                    break;
                }
            }

            if feasible {
                candidate_degrees.push(degree);
            }
        }

        // 选择适中的度数
        candidate_degrees.into_iter().max().unwrap_or(1).min(8)
    }

    /// 计算张量并行置信度
    fn calculate_tp_confidence(&self, topology: &DeviceTopology, tp_degree: usize) -> f32 {
        let device_count = topology.device_count();

        // 基于设备均匀性计算置信度
        let mut compute_scores = Vec::new();
        let mut memory_sizes = Vec::new();

        for device_id in topology.device_ids() {
            compute_scores.push(topology.get_compute_score(&device_id) as f32);
            memory_sizes.push(topology.get_total_memory_gb(&device_id));
        }

        let compute_variance = variance(&compute_scores);
        let memory_variance = variance(&memory_sizes);

        // 方差越小，置信度越高
        let compute_confidence = 1.0 / (1.0 + compute_variance);
        let memory_confidence = 1.0 / (1.0 + memory_variance);

        // 设备数量是否匹配TP度数
        let degree_confidence = if device_count >= tp_degree { 0.9 } else { 0.3 };

        (compute_confidence * 0.4 + memory_confidence * 0.4 + degree_confidence * 0.2).min(1.0)
    }

    /// 计算流水线并行置信度
    fn calculate_pp_confidence(&self, topology: &DeviceTopology, pp_degree: usize) -> f32 {
        let device_count = topology.device_count();

        // 检查设备间带宽
        let mut bandwidths = Vec::new();
        let device_ids = topology.device_ids();

        for i in 0..device_ids.len() {
            for j in (i + 1)..device_ids.len() {
                bandwidths.push(topology.get_bandwidth(&device_ids[i], &device_ids[j]));
            }
        }

        let avg_bandwidth = bandwidths.iter().sum::<f32>() / bandwidths.len() as f32;

        // 带宽越高，置信度越高
        let bandwidth_confidence = (avg_bandwidth / 10.0).min(1.0); // 10 Gbps为理想值

        // 设备数量是否匹配PP度数
        let degree_confidence = if device_count >= pp_degree { 0.8 } else { 0.2 };

        (bandwidth_confidence * 0.7 + degree_confidence * 0.3).min(1.0)
    }

    /// 计算混合并行置信度
    fn calculate_hybrid_confidence(
        &self,
        topology: &DeviceTopology,
        tp_degree: usize,
        pp_degree: usize,
    ) -> f32 {
        let tp_confidence = self.calculate_tp_confidence(topology, tp_degree);
        let pp_confidence = self.calculate_pp_confidence(topology, pp_degree);

        (tp_confidence * 0.5 + pp_confidence * 0.5).min(1.0)
    }

    /// 计算序列并行置信度
    fn calculate_sp_confidence(&self, topology: &DeviceTopology, sp_degree: usize) -> f32 {
        let device_count = topology.device_count();

        // 序列并行对设备均匀性要求较低
        let degree_confidence = if device_count >= sp_degree { 0.9 } else { 0.4 };

        // 检查设备计算能力
        let mut compute_scores = Vec::new();
        for device_id in topology.device_ids() {
            compute_scores.push(topology.get_compute_score(&device_id) as f32);
        }

        let compute_variance = variance(&compute_scores);
        let compute_confidence = 1.0 / (1.0 + compute_variance);

        (degree_confidence * 0.6 + compute_confidence * 0.4).min(1.0)
    }
}

/// 性能模型
struct PerformanceModel {
    /// 基准性能数据
    baseline_performance: HashMap<String, f32>,
}

impl PerformanceModel {
    fn new() -> Self {
        Self {
            baseline_performance: HashMap::new(),
        }
    }

    fn predict_tensor_parallel_latency(
        &self,
        topology: &DeviceTopology,
        model_size_gb: f32,
        _batch_size: usize,
        tp_degree: usize,
    ) -> f32 {
        // 简化预测模型
        let base_latency_ms = model_size_gb * 10.0; // 基础延迟

        // TP加速比（理想情况）
        let speedup = tp_degree as f32 * 0.8; // 80% 效率

        // 通信开销
        let communication_overhead = if topology.supports_rdma(&topology.device_ids()[0]) {
            0.1
        } else if topology.supports_mlx(&topology.device_ids()[0]) {
            0.15
        } else {
            0.3
        };

        let communication_cost = (tp_degree - 1) as f32 * communication_overhead;

        // 总延迟
        let latency = (base_latency_ms / speedup) * (1.0 + communication_cost);

        latency.max(1.0)
    }

    fn predict_pipeline_parallel_latency(
        &self,
        topology: &DeviceTopology,
        model_size_gb: f32,
        _batch_size: usize,
        pp_degree: usize,
    ) -> f32 {
        // 流水线并行延迟模型
        let base_latency_ms = model_size_gb * 15.0;

        // 流水线气泡开销
        let bubble_overhead = 0.3 * (pp_degree - 1) as f32;

        // 通信开销
        let mut avg_bandwidth = 0.0;
        let device_ids = topology.device_ids();
        let mut count = 0;

        for i in 0..device_ids.len() {
            for j in (i + 1)..device_ids.len() {
                avg_bandwidth += topology.get_bandwidth(&device_ids[i], &device_ids[j]);
                count += 1;
            }
        }

        avg_bandwidth /= count.max(1) as f32;
        let bandwidth_factor = (1.0 / avg_bandwidth).min(2.0);

        let latency = base_latency_ms * (1.0 + bubble_overhead) * bandwidth_factor;

        latency.max(1.0)
    }

    fn predict_hybrid_parallel_latency(
        &self,
        topology: &DeviceTopology,
        model_size_gb: f32,
        batch_size: usize,
        tp_degree: usize,
        pp_degree: usize,
    ) -> f32 {
        // 混合并行延迟模型
        let tp_latency =
            self.predict_tensor_parallel_latency(topology, model_size_gb, batch_size, tp_degree);
        let pp_latency =
            self.predict_pipeline_parallel_latency(topology, model_size_gb, batch_size, pp_degree);

        // 加权平均
        (tp_latency * 0.6 + pp_latency * 0.4).max(1.0)
    }

    fn predict_sequence_parallel_latency(
        &self,
        _topology: &DeviceTopology,
        model_size_gb: f32,
        _batch_size: usize,
        sp_degree: usize,
    ) -> f32 {
        // 序列并行延迟模型
        let base_latency_ms = model_size_gb * 8.0;

        // 序列拆分加速比
        let speedup = sp_degree as f32 * 0.9; // 90% 效率

        base_latency_ms / speedup.max(1.0)
    }
}

/// 计算方差
fn variance(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;

    variance
}

/// 估计设备间延迟
fn estimate_latency(device_a: &DeviceInfo, device_b: &DeviceInfo) -> u64 {
    // 简化估计
    if device_a.device_type == device_b.device_type {
        500_000 // 0.5毫秒
    } else {
        1_000_000 // 1毫秒
    }
}

/// 估计设备间带宽
fn estimate_bandwidth(device_a: &DeviceInfo, device_b: &DeviceInfo) -> f32 {
    // 简化估计
    if device_a.capabilities.supports_rdma && device_b.capabilities.supports_rdma {
        40.0 // RDMA带宽
    } else if device_a.capabilities.supports_mlx && device_b.capabilities.supports_mlx {
        25.0 // MLX带宽
    } else {
        1.0 // 标准以太网
    }
}

// ==================== 模块导出 ====================
// 类型已经在顶层定义为pub，不需要重复导出
