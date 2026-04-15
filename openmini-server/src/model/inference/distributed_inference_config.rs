//! 分布式推理配置系统
//!
//! 支持大规模模型分布式推理的完整配置，
//! 包含模型并行、多节点通信、负载均衡等核心组件。
//!
//! # 配置层次
//!
//! ```text
//! distributed_inference_config.rs
//! ├── DistributedInferenceConfig   - 完整分布式推理配置
//! │   ├── ModelParallelConfig      - 模型并行策略配置
//! │   ├── NodeCommunicationConfig  - 多节点通信配置
//! │   ├── LoadBalancingConfig      - 负载均衡配置
//! │   └── PerformanceTuningConfig  - 性能调优配置
//! │
//! ├── ParallelStrategy             - 并行策略枚举
//! └── CommunicationBackend         - 通信后端枚举
//! ```
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use openmini_server::model::inference::distributed_inference_config::{
//!     DistributedInferenceConfig, ModelParallelConfig, ParallelStrategy,
//! };
//!
//! // 创建针对 70B 模型的分布式推理配置
//! let config = DistributedInferenceConfig::for_70b_model();
//! config.validate()?;
//!
//! // 估算推理延迟和吞吐量
//! let metrics = config.estimate_performance_metrics(1024); // batch_size=1024
//! ```
//!

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

// ==================== 并行策略配置 ====================

/// 模型并行策略
///
/// 定义模型在多个计算设备间的并行方式。
/// 推理场景通常优先考虑延迟，而非训练中的吞吐量。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[allow(clippy::enum_variant_names)]
pub enum ParallelStrategy {
    /// 张量并行 (Tensor Parallelism)
    /// 将单个层拆分为多个设备
    /// 优势: 层内并行，低延迟
    /// 适用: 单节点多 GPU 场景
    TensorParallel,

    /// 流水线并行 (Pipeline Parallelism)
    /// 将模型层按深度分配到多个设备
    /// 优势: 支持超大模型，显存效率高
    /// 缺点: 有流水线气泡
    PipelineParallel,

    /// 张量+流水线混合并行 (3D Parallelism)
    /// 结合 TP 和 PP 的优势
    /// 推荐: 大规模多节点推理
    HybridParallel,

    /// 序列并行 (Sequence Parallelism)
    /// 将输入序列拆分到多个设备
    /// 优势: 支持超长序列推理
    /// 适用: 长文本生成场景
    SequenceParallel,
}

impl fmt::Display for ParallelStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TensorParallel => write!(f, "tensor_parallel"),
            Self::PipelineParallel => write!(f, "pipeline_parallel"),
            Self::HybridParallel => write!(f, "hybrid_parallel"),
            Self::SequenceParallel => write!(f, "sequence_parallel"),
        }
    }
}

impl Default for ParallelStrategy {
    fn default() -> Self {
        Self::TensorParallel // 推理场景默认使用张量并行（低延迟）
    }
}

/// 通信后端选择
///
/// 定义多节点/多设备间的通信实现。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommunicationBackend {
    /// 使用 NCCL (NVIDIA Collective Communications Library)
    /// 支持: CUDA 设备间高速通信
    /// 适用: NVIDIA GPU 集群
    Nccl,

    /// 使用 RCCL (ROCm Collective Communications Library)
    /// 支持: AMD GPU 设备间通信
    /// 适用: AMD GPU 集群
    Rccl,

    /// 使用 Gloo (Facebook Collective Communication Library)
    /// 支持: CPU/GPU 跨平台通信
    /// 适用: 异构计算环境
    Gloo,

    /// 使用 MPI (Message Passing Interface)
    /// 支持: 超大规模跨节点通信
    /// 适用: HPC 集群
    Mpi,
}

impl fmt::Display for CommunicationBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nccl => write!(f, "nccl"),
            Self::Rccl => write!(f, "rccl"),
            Self::Gloo => write!(f, "gloo"),
            Self::Mpi => write!(f, "mpi"),
        }
    }
}

impl Default for CommunicationBackend {
    fn default() -> Self {
        Self::Nccl // 默认使用 NCCL（NVIDIA GPU 最优）
    }
}

// ==================== 模型并行配置 ====================

/// 模型并行配置
///
/// 控制模型在多个计算设备上的并行策略和参数。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParallelConfig {
    /// 并行策略选择
    #[serde(default)]
    pub strategy: ParallelStrategy,

    /// 张量并行度 (Tensor Parallelism Degree)
    /// 每个层的并行拆分数量
    /// 推荐: 1-8，通常为 GPU 数量的约数
    #[serde(default = "default_tp_degree")]
    pub tp_degree: usize,

    /// 流水线并行度 (Pipeline Parallelism Degree)
    /// 模型深度的流水线阶段数
    /// 推荐: 1-64，根据模型层数和节点数确定
    #[serde(default = "default_pp_degree")]
    pub pp_degree: usize,

    /// 序列并行拆分大小
    /// 每个设备处理的序列长度
    /// 推荐: 32-512，根据序列长度和显存确定
    #[serde(default = "default_sequence_chunk_size")]
    pub sequence_chunk_size: usize,

    /// 是否启用权重分片
    /// 将模型权重分片存储到不同设备
    /// 优势: 支持超大模型，显存效率高
    /// 缺点: 增加通信开销
    #[serde(default = "default_weight_sharding")]
    pub weight_sharding: bool,

    /// 是否启用激活检查点 (Activation Checkpointing)
    /// 通过重计算减少激活显存
    /// 优势: 显存效率提升 3-5 倍
    /// 缺点: 增加 30-40% 计算开销
    #[serde(default = "default_activation_checkpointing")]
    pub activation_checkpointing: bool,

    /// 微批次大小 (Micro-batch Size)
    /// 流水线并行中的微批次大小
    /// 推荐: 1-16，平衡吞吐和延迟
    #[serde(default = "default_micro_batch_size")]
    pub micro_batch_size: usize,
}

impl Default for ModelParallelConfig {
    fn default() -> Self {
        Self {
            strategy: ParallelStrategy::default(),
            tp_degree: default_tp_degree(),
            pp_degree: default_pp_degree(),
            sequence_chunk_size: default_sequence_chunk_size(),
            weight_sharding: default_weight_sharding(),
            activation_checkpointing: default_activation_checkpointing(),
            micro_batch_size: default_micro_batch_size(),
        }
    }
}

/// 为 70B 模型优化的模型并行配置
impl ModelParallelConfig {
    pub fn for_70b_model() -> Self {
        Self {
            strategy: ParallelStrategy::HybridParallel,
            tp_degree: 8,
            pp_degree: 8,
            sequence_chunk_size: 128,
            weight_sharding: true,
            activation_checkpointing: true,
            micro_batch_size: 4,
        }
    }

    /// 为 14B 模型优化的模型并行配置
    pub fn for_14b_model() -> Self {
        Self {
            strategy: ParallelStrategy::TensorParallel,
            tp_degree: 4,
            pp_degree: 1,
            sequence_chunk_size: 256,
            weight_sharding: false,
            activation_checkpointing: false,
            micro_batch_size: 8,
        }
    }

    /// 为 7B 模型优化的模型并行配置
    pub fn for_7b_model() -> Self {
        Self {
            strategy: ParallelStrategy::TensorParallel,
            tp_degree: 2,
            pp_degree: 1,
            sequence_chunk_size: 512,
            weight_sharding: false,
            activation_checkpointing: false,
            micro_batch_size: 16,
        }
    }
}

// ==================== 节点通信配置 ====================

/// 节点通信配置
///
/// 控制多节点/多设备间的通信行为。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCommunicationConfig {
    /// 通信后端选择
    #[serde(default)]
    pub backend: CommunicationBackend,

    /// 是否启用通信压缩
    /// 使用梯度压缩或参数压缩减少通信量
    /// 优势: 减少 50-80% 通信带宽
    /// 缺点: 增加少量计算开销
    #[serde(default = "default_compression_enabled")]
    pub compression_enabled: bool,

    /// 压缩比 (0.1-1.0)
    /// 1.0 表示无压缩，0.1 表示压缩到 10%
    #[serde(default = "default_compression_ratio")]
    pub compression_ratio: f32,

    /// AllReduce 操作的 bucket 大小 (bytes)
    /// 较大的 bucket 减少通信次数但增加延迟
    /// 较小的 bucket 降低延迟但增加通信开销
    #[serde(default = "default_reduce_bucket_size")]
    pub reduce_bucket_size: usize,

    /// AllGather 操作的 bucket 大小 (bytes)
    #[serde(default = "default_allgather_bucket_size")]
    pub allgather_bucket_size: usize,

    /// 是否启用异步通信
    /// 在计算过程中异步执行通信操作
    /// 优势: 隐藏通信延迟
    /// 缺点: 增加实现复杂度
    #[serde(default = "default_async_communication")]
    pub async_communication: bool,

    /// 通信超时时间 (毫秒)
    /// 通信操作的最大等待时间
    #[serde(default = "default_communication_timeout_ms")]
    pub communication_timeout_ms: u64,
}

impl Default for NodeCommunicationConfig {
    fn default() -> Self {
        Self {
            backend: CommunicationBackend::default(),
            compression_enabled: default_compression_enabled(),
            compression_ratio: default_compression_ratio(),
            reduce_bucket_size: default_reduce_bucket_size(),
            allgather_bucket_size: default_allgather_bucket_size(),
            async_communication: default_async_communication(),
            communication_timeout_ms: default_communication_timeout_ms(),
        }
    }
}

// ==================== 负载均衡配置 ====================

/// 负载均衡策略
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// 轮询调度 (Round Robin)
    /// 依次分配请求到各个节点
    RoundRobin,

    /// 最少连接 (Least Connections)
    /// 分配请求到当前连接最少的节点
    LeastConnections,

    /// 基于延迟的负载均衡
    /// 根据节点响应时间动态分配
    LatencyBased,

    /// 一致性哈希 (Consistent Hashing)
    /// 相同请求总是路由到相同节点
    /// 优势: 缓存友好
    ConsistentHashing,
}

impl fmt::Display for LoadBalancingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RoundRobin => write!(f, "round_robin"),
            Self::LeastConnections => write!(f, "least_connections"),
            Self::LatencyBased => write!(f, "latency_based"),
            Self::ConsistentHashing => write!(f, "consistent_hashing"),
        }
    }
}

impl Default for LoadBalancingStrategy {
    fn default() -> Self {
        Self::RoundRobin
    }
}

/// 负载均衡配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingConfig {
    /// 负载均衡策略
    #[serde(default)]
    pub strategy: LoadBalancingStrategy,

    /// 健康检查间隔 (毫秒)
    /// 定期检查节点健康状态
    #[serde(default = "default_health_check_interval_ms")]
    pub health_check_interval_ms: u64,

    /// 故障节点重试次数
    /// 请求失败后的重试次数
    #[serde(default = "default_failure_retry_count")]
    pub failure_retry_count: usize,

    /// 故障转移超时 (毫秒)
    /// 故障转移操作的最大等待时间
    #[serde(default = "default_failover_timeout_ms")]
    pub failover_timeout_ms: u64,

    /// 是否启用请求排队
    /// 当所有节点忙时排队等待
    #[serde(default = "default_request_queuing")]
    pub request_queuing: bool,

    /// 最大队列长度
    /// 请求队列的最大长度
    #[serde(default = "default_max_queue_length")]
    pub max_queue_length: usize,
}

impl Default for LoadBalancingConfig {
    fn default() -> Self {
        Self {
            strategy: LoadBalancingStrategy::default(),
            health_check_interval_ms: default_health_check_interval_ms(),
            failure_retry_count: default_failure_retry_count(),
            failover_timeout_ms: default_failover_timeout_ms(),
            request_queuing: default_request_queuing(),
            max_queue_length: default_max_queue_length(),
        }
    }
}

// ==================== 性能调优配置 ====================

/// 性能调优配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTuningConfig {
    /// 推理批次大小
    /// 单次推理处理的样本数
    /// 推荐: 1-128，根据显存和延迟需求确定
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// KV 缓存最大序列长度
    /// KV 缓存的最大序列长度限制
    #[serde(default = "default_max_sequence_length")]
    pub max_sequence_length: usize,

    /// 是否启用 KV 缓存分片
    /// 将 KV 缓存分片到多个设备
    /// 优势: 支持更长序列
    #[serde(default = "default_kv_cache_sharding")]
    pub kv_cache_sharding: bool,

    /// 是否启用预填充优化
    /// 对 prompt 进行预填充优化
    /// 优势: 减少首 token 延迟
    #[serde(default = "default_prefill_optimization")]
    pub prefill_optimization: bool,

    /// 预填充并行度
    /// prompt 预填充的并行度
    #[serde(default = "default_prefill_parallelism")]
    pub prefill_parallelism: usize,

    /// 是否启用增量解码优化
    /// 优化生成阶段的增量解码
    #[serde(default = "default_incremental_decoding")]
    pub incremental_decoding: bool,

    /// 最大并行解码数量
    /// 同时解码的序列数量
    #[serde(default = "default_max_parallel_decode")]
    pub max_parallel_decode: usize,
}

impl Default for PerformanceTuningConfig {
    fn default() -> Self {
        Self {
            batch_size: default_batch_size(),
            max_sequence_length: default_max_sequence_length(),
            kv_cache_sharding: default_kv_cache_sharding(),
            prefill_optimization: default_prefill_optimization(),
            prefill_parallelism: default_prefill_parallelism(),
            incremental_decoding: default_incremental_decoding(),
            max_parallel_decode: default_max_parallel_decode(),
        }
    }
}

// ==================== 主配置结构 ====================

/// 分布式推理配置错误
#[derive(Debug, Error)]
pub enum DistributedInferenceConfigError {
    #[error("配置验证失败: {0}")]
    Validation(String),

    #[error("无效的并行度配置: TP={tp}, PP={pp}, 总GPU={gpus}")]
    InvalidParallelism { tp: usize, pp: usize, gpus: usize },

    #[error("序列长度超过限制: {length} > {max}")]
    SequenceLengthExceeded { length: usize, max: usize },
}

/// 完整分布式推理配置
///
/// 整合模型并行、节点通信、负载均衡、性能调优为统一配置，
/// 用于驱动大规模分布式推理服务。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedInferenceConfig {
    /// 模型并行配置
    #[serde(default)]
    pub model_parallel: ModelParallelConfig,

    /// 节点通信配置
    #[serde(default)]
    pub node_communication: NodeCommunicationConfig,

    /// 负载均衡配置
    #[serde(default)]
    pub load_balancing: LoadBalancingConfig,

    /// 性能调优配置
    #[serde(default)]
    pub performance_tuning: PerformanceTuningConfig,

    // 全局硬件参数
    /// 总 GPU 数量
    #[serde(default = "default_total_gpus")]
    pub total_gpus: usize,

    /// 每个 GPU 的显存 (GB)
    #[serde(default = "default_gpu_memory_gb")]
    pub gpu_memory_gb: usize,

    /// 节点间网络带宽 (Gbps)
    #[serde(default = "default_network_bandwidth_gbps")]
    pub network_bandwidth_gbps: f32,

    /// 模型总参数量 (用于性能估算)
    #[serde(default = "default_model_parameters")]
    pub model_parameters: u64,
}

impl Default for DistributedInferenceConfig {
    fn default() -> Self {
        Self {
            model_parallel: ModelParallelConfig::default(),
            node_communication: NodeCommunicationConfig::default(),
            load_balancing: LoadBalancingConfig::default(),
            performance_tuning: PerformanceTuningConfig::default(),
            total_gpus: default_total_gpus(),
            gpu_memory_gb: default_gpu_memory_gb(),
            network_bandwidth_gbps: default_network_bandwidth_gbps(),
            model_parameters: default_model_parameters(),
        }
    }
}

impl DistributedInferenceConfig {
    /// 创建针对 70B 模型优化的分布式推理配置
    ///
    /// 基于 64x H100 SXM5 集群预设最优参数
    pub fn for_70b_model() -> Self {
        Self {
            model_parallel: ModelParallelConfig::for_70b_model(),
            node_communication: NodeCommunicationConfig::default(),
            load_balancing: LoadBalancingConfig::default(),
            performance_tuning: PerformanceTuningConfig {
                batch_size: 4,
                max_sequence_length: 8192,
                kv_cache_sharding: true,
                prefill_optimization: true,
                prefill_parallelism: 4,
                incremental_decoding: true,
                max_parallel_decode: 8,
            },
            total_gpus: 64,
            gpu_memory_gb: 80,
            network_bandwidth_gbps: 400.0, // NVLink + InfiniBand
            model_parameters: 70_500_000_000,
        }
    }

    /// 创建针对 14B 模型优化的分布式推理配置
    pub fn for_14b_model() -> Self {
        Self {
            model_parallel: ModelParallelConfig::for_14b_model(),
            node_communication: Default::default(),
            load_balancing: Default::default(),
            performance_tuning: PerformanceTuningConfig {
                batch_size: 16,
                max_sequence_length: 4096,
                kv_cache_sharding: false,
                prefill_optimization: true,
                prefill_parallelism: 2,
                incremental_decoding: true,
                max_parallel_decode: 16,
            },
            total_gpus: 4,
            gpu_memory_gb: 80,
            network_bandwidth_gbps: 200.0,
            model_parameters: 14_000_000_000,
        }
    }

    /// 创建针对 7B 模型优化的分布式推理配置
    pub fn for_7b_model() -> Self {
        Self {
            model_parallel: ModelParallelConfig::for_7b_model(),
            node_communication: Default::default(),
            load_balancing: Default::default(),
            performance_tuning: PerformanceTuningConfig {
                batch_size: 32,
                max_sequence_length: 4096,
                kv_cache_sharding: false,
                prefill_optimization: true,
                prefill_parallelism: 1,
                incremental_decoding: true,
                max_parallel_decode: 32,
            },
            total_gpus: 2,
            gpu_memory_gb: 80,
            network_bandwidth_gbps: 100.0,
            model_parameters: 7_000_000_000,
        }
    }

    /// 验证配置的合法性
    pub fn validate(&self) -> Result<(), DistributedInferenceConfigError> {
        // 检查并行度配置
        let total_parallel = self.model_parallel.tp_degree * self.model_parallel.pp_degree;
        if total_parallel > self.total_gpus {
            return Err(DistributedInferenceConfigError::InvalidParallelism {
                tp: self.model_parallel.tp_degree,
                pp: self.model_parallel.pp_degree,
                gpus: self.total_gpus,
            });
        }

        // 检查批次大小
        if self.performance_tuning.batch_size == 0 {
            return Err(DistributedInferenceConfigError::Validation(
                "批次大小必须大于 0".to_string(),
            ));
        }

        // 检查序列长度限制
        if self.performance_tuning.max_sequence_length == 0 {
            return Err(DistributedInferenceConfigError::Validation(
                "最大序列长度必须大于 0".to_string(),
            ));
        }

        Ok(())
    }

    /// 估算推理性能指标
    ///
    /// # 参数
    ///
    /// - `sequence_length`: 输入序列长度
    ///
    /// # 返回值
    ///
    /// 性能指标元组: (延迟_ms, 吞吐量_tokens_per_sec)
    pub fn estimate_performance_metrics(&self, sequence_length: usize) -> (f64, f64) {
        if sequence_length > self.performance_tuning.max_sequence_length {
            // 超出限制，返回保守估计
            return (1000.0, 10.0);
        }

        // 简化的性能估算模型
        let base_latency_ms = match self.model_parameters {
            p if p > 50_000_000_000 => 200.0, // 50B+
            p if p > 10_000_000_000 => 100.0, // 10-50B
            _ => 50.0,                        // <10B
        };

        let parallel_factor =
            (self.model_parallel.tp_degree * self.model_parallel.pp_degree) as f64;
        let latency_ms =
            base_latency_ms * (sequence_length as f64 / 512.0) / parallel_factor.sqrt();

        let batch_size = self.performance_tuning.batch_size as f64;
        let throughput_tokens_per_sec = (batch_size * 1000.0) / latency_ms.max(1.0);

        (latency_ms, throughput_tokens_per_sec)
    }

    /// 估算显存需求 (GB)
    pub fn estimate_memory_requirements_gb(&self) -> f64 {
        let param_memory_gb = (self.model_parameters as f64 * 2.0) / 1_000_000_000.0; // BF16 精度
        let kv_cache_memory_gb = (self.performance_tuning.max_sequence_length as f64
            * self.performance_tuning.batch_size as f64
            * self.model_parallel.tp_degree as f64
            * 2.0)
            / 1_000_000_000.0;

        let activation_memory_gb = if self.model_parallel.activation_checkpointing {
            param_memory_gb * 0.1 // 激活检查点大幅减少激活显存
        } else {
            param_memory_gb * 0.3 // 常规激活显存
        };

        (param_memory_gb + kv_cache_memory_gb + activation_memory_gb)
            / (self.model_parallel.tp_degree * self.model_parallel.pp_degree) as f64
    }
}

// ==================== 默认值函数 ====================

fn default_tp_degree() -> usize {
    1
}
fn default_pp_degree() -> usize {
    1
}
fn default_sequence_chunk_size() -> usize {
    256
}
fn default_weight_sharding() -> bool {
    false
}
fn default_activation_checkpointing() -> bool {
    false
}
fn default_micro_batch_size() -> usize {
    1
}

fn default_compression_enabled() -> bool {
    false
}
fn default_compression_ratio() -> f32 {
    0.5
}
fn default_reduce_bucket_size() -> usize {
    500_000_000
} // 500MB
fn default_allgather_bucket_size() -> usize {
    500_000_000
} // 500MB
fn default_async_communication() -> bool {
    true
}
fn default_communication_timeout_ms() -> u64 {
    30_000
} // 30秒

fn default_health_check_interval_ms() -> u64 {
    5_000
} // 5秒
fn default_failure_retry_count() -> usize {
    3
}
fn default_failover_timeout_ms() -> u64 {
    10_000
} // 10秒
fn default_request_queuing() -> bool {
    true
}
fn default_max_queue_length() -> usize {
    1000
}

fn default_batch_size() -> usize {
    1
}
fn default_max_sequence_length() -> usize {
    2048
}
fn default_kv_cache_sharding() -> bool {
    false
}
fn default_prefill_optimization() -> bool {
    true
}
fn default_prefill_parallelism() -> usize {
    1
}
fn default_incremental_decoding() -> bool {
    true
}
fn default_max_parallel_decode() -> usize {
    4
}

fn default_total_gpus() -> usize {
    1
}
fn default_gpu_memory_gb() -> usize {
    24
} // RTX 4090 级别
fn default_network_bandwidth_gbps() -> f32 {
    100.0
} // PCIe 4.0 x16
fn default_model_parameters() -> u64 {
    7_000_000_000
} // 7B

// ==================== 单元测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DistributedInferenceConfig::default();
        assert_eq!(config.total_gpus, 1);
        assert_eq!(config.model_parallel.tp_degree, 1);
        assert_eq!(config.model_parallel.pp_degree, 1);
    }

    #[test]
    fn test_70b_model_config() {
        let config = DistributedInferenceConfig::for_70b_model();
        assert_eq!(config.model_parameters, 70_500_000_000);
        assert_eq!(config.total_gpus, 64);
        assert_eq!(config.model_parallel.tp_degree, 8);
        assert_eq!(config.model_parallel.pp_degree, 8);
        assert!(config.model_parallel.weight_sharding);
        assert!(config.performance_tuning.kv_cache_sharding);
    }

    #[test]
    fn test_config_validation() {
        let mut config = DistributedInferenceConfig::default();
        config.model_parallel.tp_degree = 2;
        config.model_parallel.pp_degree = 2;
        config.total_gpus = 3; // 2*2=4 > 3，应该验证失败

        let result = config.validate();
        assert!(result.is_err());

        // 修复配置
        config.total_gpus = 4;
        let result = config.validate();
        assert!(result.is_ok());
    }

    #[test]
    fn test_performance_estimation() {
        let config = DistributedInferenceConfig::for_7b_model();
        let (latency, throughput) = config.estimate_performance_metrics(512);

        assert!(latency > 0.0);
        assert!(throughput > 0.0);
    }

    #[test]
    fn test_memory_estimation() {
        let config_7b = DistributedInferenceConfig::for_7b_model();
        let config_70b = DistributedInferenceConfig::for_70b_model();

        let mem_7b = config_7b.estimate_memory_requirements_gb();
        let mem_70b = config_70b.estimate_memory_requirements_gb();

        // 70B 模型总显存需求应该大于 7B 模型总显存需求
        let total_mem_7b = mem_7b * config_7b.total_gpus as f64;
        let total_mem_70b = mem_70b * config_70b.total_gpus as f64;
        assert!(
            total_mem_70b > total_mem_7b,
            "70B model total memory {} GB should be greater than 7B model total memory {} GB",
            total_mem_70b,
            total_mem_7b
        );
    }

    #[test]
    fn test_serialization() {
        let config = DistributedInferenceConfig::for_14b_model();

        // 序列化为 JSON
        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.is_empty());

        // 反序列化
        let deserialized: DistributedInferenceConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.model_parameters, config.model_parameters);
    }
}
