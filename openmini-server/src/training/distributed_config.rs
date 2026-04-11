//! 70B-Dense 分布式训练配置系统
//!
//! 支持大规模 70B 模型训练的完整分布式配置，
//! 包含 FSDP、DeepSpeed、数据并行、混合精度等核心组件。
//!
//! # 配置层次
//!
//! ```text
//! distributed_config.rs
//! ├── DistributedTrainingConfig   - 完整分布式训练配置
//! │   ├── FsdpConfig              - FSDP (Fully Sharded Data Parallel) 配置
//! │   ├── DeepSpeedConfig         - DeepSpeed 集成配置
//! │   ├── DataParallelConfig      - 数据并行策略配置
//! │   └── AmpConfig               - 混合精度训练 (AMP) 配置
//! │
//! ├── ShardingStrategy           - FSDP 分片策略枚举
//! └── MemoryEstimator            - 显存估算工具
//! ```
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use openmini_server::training::distributed_config::{
//!     DistributedTrainingConfig, FsdpConfig, ShardingStrategy,
//! };
//!
//! // 创建 70B 模型的 FSDP 训练配置
//! let config = DistributedTrainingConfig::for_70b_dense();
//! config.validate()?;
//!
//! // 估算训练显存需求
//! let memory_gb = config.estimate_training_memory_gb(70_500_000_000);
//! assert!(memory_gb > 1000.0); // 70B 需要 >1TB 显存
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

// ==================== FSDP 配置 ====================

/// FSDP (Fully Sharded Data Parallel) 分片策略
///
/// 定义模型参数和优化器状态在多 GPU 间的分片方式。
/// 70B 模型必须使用 FULL_SHARD 或 HYBRID_SHARD 才能适配显存。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardingStrategy {
    /// 完全分片: 参数、梯度、优化器状态全部分片
    /// 显存效率最高，通信开销最大
    /// 推荐: 70B 模型在 GPU < 80GB 时使用
    FullShard,

    /// 梯度和优化器状态分片: 参数不分片
    /// 通信较少，但每卡需要完整模型副本
    /// 适用: 模型能放入单卡显存时
    ShardGradOp,

    /// 混合分片: 结合前两者优点
    /// 参数在 TP 组内复制，跨组分片
    /// 推荐: 70B 模型在 H100 集群上的最佳选择
    HybridShard,
}

impl fmt::Display for ShardingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FullShard => write!(f, "full_shard"),
            Self::ShardGradOp => write!(f, "shard_grad_op"),
            Self::HybridShard => write!(f, "hybrid_shard"),
        }
    }
}

impl Default for ShardingStrategy {
    fn default() -> Self {
        Self::HybridShard // 70B 默认使用混合分片
    }
}

/// FSDP (Fully Sharded Data Parallel) 配置
///
/// PyTorch FSDP 的参数化配置，控制模型分片和内存管理行为。
/// 对于 70B Dense 模型，FSDP 是必需的显存优化技术。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FsdpConfig {
    /// 分片策略选择
    #[serde(default)]
    pub sharding_strategy: ShardingStrategy,

    /// 是否将 CPU 作为卸载后端
    ///
    /// 当为 true 时，优化器状态卸载到 CPU 内存，
    /// 以 GPU<->CPU 带宽换取显存空间。
    /// NVLink 高速互联下建议关闭 (false)，PCIe 场景建议开启 (true)
    #[serde(default = "default_offload_params")]
    pub offload_params: bool,

    /// 是否限制 all_gather 操作的内存峰值
    ///
    /// 开启后会限制前向/反向传播中的 all_gather 峰值显存，
    /// 但可能略微降低训练速度 (~5-10%)
    #[serde(default = "default_limit_all_gathers")]
    pub limit_all_gathers: bool,

    /// 每个所有者进程的最小参数数量 (用于负载均衡)
    ///
    /// 控制参数分片的粒度。值越小分片越均匀，
    /// 但增加元数据和通信开销。
    /// 推荐: 70B 使用 1M-10M 范围
    #[serde(default = "default_min_params")]
    pub min_num_params: usize,

    /// 是否启用激活检查点与 FSDP 的协同优化
    ///
    /// 开启后，FSDP 会自动管理激活检查点的卸载时机，
    /// 进一步减少峰值显存 ~15-20%
    #[serde(default = "default_activation_checkpointing_fsdp")]
    pub activation_checkpointing: bool,

    /// 是否使用 torch.compile 加速
    ///
    /// PyTorch 2.0+ 的编译优化，可提升 10-20% 吞吐
    /// 注意: 可能增加编译时间和调试难度
    #[serde(default)]
    pub use_torch_compile: bool,
}

impl Default for FsdpConfig {
    fn default() -> Self {
        Self {
            sharding_strategy: ShardingStrategy::HybridShard,
            offload_params: false,     // H100 NVLink 不需要卸载
            limit_all_gathers: true,    // 限制显存峰值
            min_num_params: 5_000_000,  // 5M 参数/片
            activation_checkpointing: true,
            use_torch_compile: false,   // 稳定性优先
        }
    }
}

fn default_offload_params() -> bool { false }
fn default_limit_all_gathers() -> bool { true }
fn default_min_params() -> usize { 5_000_000 }
fn default_activation_checkpointing_fsdp() -> bool { true }

impl FsdpConfig {
    /// 创建适合 70B-Dense 模型的 FSDP 配置
    ///
    /// 针对 64x H100 集群优化的默认配置：
    /// - HybridShard 平衡显存和通信
    /// - 关闭 CPU 卸载 (NVLink 900GB/s 带宽足够)
    /// - 开启激活检查点协同
    pub fn for_70b_dense() -> Self {
        Self {
            sharding_strategy: ShardingStrategy::HybridShard,
            offload_params: false,
            limit_all_gathers: true,
            min_num_params: 10_000_000, // 70B 用更大的分片粒度
            activation_checkpointing: true,
            use_torch_compile: false,
        }
    }

    /// 创建适合 14B-Dense 模型的 FSDP 配置
    pub fn for_14b_dense() -> Self {
        Self {
            sharding_strategy: ShardingStrategy::FullShard, // 14B 可用完全分片
            offload_params: false,
            limit_all_gathers: false, // 14B 显存压力小
            min_num_params: 1_000_000,
            activation_checkpointing: true,
            use_torch_compile: false,
        }
    }

    /// 验证 FSDP 配置有效性
    pub fn validate(&self) -> Result<(), DistributedTrainingError> {
        if self.min_num_params == 0 {
            return Err(DistributedTrainingError::InvalidConfiguration(
                "min_num_params must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    /// 估算 FSDP 显存节省比例
    ///
    /// 返回相对于 DDP (无分片) 的显存占用比例 (0.0-1.0)
    pub fn memory_reduction_ratio(&self) -> f64 {
        match self.sharding_strategy {
            ShardingStrategy::FullShard => {
                // 完全分片: 优化器状态 / N + 梯度 / N + 参数 / N
                // 相比 DDP 节省约 1/N 的优化器和梯度显存
                if self.offload_params {
                    0.15 // 极端情况下仅保留 15%
                } else {
                    0.35 // 正常情况约 35%
                }
            }
            ShardingStrategy::ShardGradOp => {
                // 仅分片梯度和优化器
                0.55 // 约 55%
            }
            ShardingStrategy::HybridShard => {
                // 混合模式介于两者之间
                if self.offload_params {
                    0.25
                } else {
                    0.42 // 70B 推荐配置约 42%
                }
            }
        }
    }
}

// ==================== DeepSpeed 配置 ====================

/// DeepSpeed 训练阶段
///
/// DeepSpeed 提供三种并行优化级别:
/// - Stage 1: 梯度分片 (类似 ZeRO-1)
/// - Stage 2: 梯度 + 优化器状态分片 (ZeRO-2)
/// - Stage 3: 全部分片包括参数 (ZeRO-3)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeepSpeedStage {
    /// Stage 1: 仅梯度分片
    Stage1 = 1,
    /// Stage 2: 梯度 + 优化器状态分片
    Stage2 = 2,
    /// Stage 3: 全部分片 (推荐用于 70B)
    Stage3 = 3,
}

impl Default for DeepSpeedStage {
    fn default() -> Self {
        Self::Stage3 // 70B 默认使用 Stage 3
    }
}

impl fmt::Display for DeepSpeedStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stage1 => write!(f, "stage1"),
            Self::Stage2 => write!(f, "stage2"),
            Self::Stage3 => write!(f, "stage3"),
        }
    }
}

/// DeepSpeed 集成配置
///
/// Microsoft DeepSpeed 的参数化配置，
/// 提供 ZeRO 优化、通信压缩等高级功能。
/// 可与 FSDP 替代或配合使用。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSpeedConfig {
    /// DeepSpeed 训练阶段 (1/2/3)
    #[serde(default)]
    pub stage: DeepSpeedStage,

    /// 是否将优化器状态卸载到 CPU/NVMe
    ///
    /// 卸载目标选项:
    /// - `false`: 不卸载 (GPU 显存充足时)
    /// - `"cpu"`: 卸载到 CPU 内存 (PCIe 带宽限制)
    /// - `"nvme"`: 卸载到 NVMe SSD (大容量但高延迟)
    #[serde(default)]
    pub offload_optimizer: Option<String>,

    /// AllReduce 操作的 bucket 大小 (bytes)
    ///
    /// 较大的 bucket 减少通信次数但增加延迟，
    /// 较小的 bucket 降低延迟但增加通信开销。
    /// 推荐: 70B 使用 500M-2G 范围
    #[serde(default = "default_reduce_bucket")]
    pub reduce_bucket_size: usize,

    /// AllGather 操作的 bucket 大小 (bytes)
    #[serde(default = "default_allgather_bucket")]
    pub allgather_bucket_size: usize,

    /// 是否启用梯度通信压缩
    ///
    /// 使用 1-bit Compressor 或 PowerSGD 减少通信量，
    /// 在带宽受限场景下提升 20-40% 吞吐
    #[serde(default)]
    pub gradient_compression: bool,

    /// 是否启用梯度累积通信重叠
    ///
    /// 在梯度累积步内异步执行通信，
    /// 有效隐藏通信延迟
    #[serde(default = "default_overlap_comm")]
    pub overlap_comm: bool,

    /// 每个节点是否使用本地梯度同步
    ///
    /// 同节点内使用 NCCL P2P 或 SHARED MEMORY，
    /// 跨节点使用标准 AllReduce
    #[serde(default)]
    pub reduce_scatter: bool,

    /// 自定义配置文件路径 (可选)
    ///
    /// 如果提供，将忽略其他字段并加载该文件
    #[serde(default)]
    pub config_file: Option<String>,
}

impl Default for DeepSpeedConfig {
    fn default() -> Self {
        Self {
            stage: DeepSpeedStage::Stage3,
            offload_optimizer: None,       // 不卸载
            reduce_bucket_size: 500_000_000, // 500 MB
            allgather_bucket_size: 500_000_000,
            gradient_compression: false,    // NVLink 下不需要
            overlap_comm: true,             // 推荐开启
            reduce_scatter: true,           // 推荐开启
            config_file: None,
        }
    }
}

fn default_reduce_bucket() -> usize { 500_000_000 }
fn default_allgather_bucket() -> usize { 500_000_000 }
fn default_overlap_comm() -> bool { true }

impl DeepSpeedConfig {
    /// 创建适合 70B-Dense 的 DeepSpeed 配置
    pub fn for_70b_dense() -> Self {
        Self {
            stage: DeepSpeedStage::Stage3,
            offload_optimizer: Some("cpu".to_string()), // 70B 可选 CPU 卸载
            reduce_bucket_size: 2_000_000_000,          // 2GB bucket
            allgather_bucket_size: 2_000_000_000,
            gradient_compression: false,
            overlap_comm: true,
            reduce_scatter: true,
            config_file: None,
        }
    }

    /// 创建适合 14B-Dense 的 DeepSpeed 配置
    pub fn for_14b_dense() -> Self {
        Self {
            stage: DeepSpeedStage::Stage2, // 14B 用 Stage2 足够
            offload_optimizer: None,
            reduce_bucket_size: 200_000_000,
            allgather_bucket_size: 200_000_000,
            gradient_compression: false,
            overlap_comm: true,
            reduce_scatter: true,
            config_file: None,
        }
    }

    /// 验证 DeepSpeed 配置有效性
    pub fn validate(&self) -> Result<(), DistributedTrainingError> {
        match self.stage {
            DeepSpeedStage::Stage1 | DeepSpeedStage::Stage2 | DeepSpeedStage::Stage3 => {}
        }

        if let Some(ref offload) = self.offload_optimizer {
            let valid_offloads = ["cpu", "nvme", "none"];
            if !valid_offloads.contains(&offload.as_str()) {
                return Err(DistributedTrainingError::InvalidConfiguration(format!(
                    "Invalid offload_optimizer: {} (must be 'cpu', 'nvme', or 'none')",
                    offload
                )));
            }
        }

        if self.reduce_bucket_size == 0 || self.allgather_bucket_size == 0 {
            return Err(DistributedTrainingError::InvalidConfiguration(
                "Bucket sizes must be > 0".to_string(),
            ));
        }

        Ok(())
    }

    /// 获取实际的 ZeRO 优化阶段数
    pub fn zero_stage(&self) -> u8 {
        self.stage as u8
    }
}

// ==================== 数据并行配置 ====================

/// 数据并行策略配置
///
/// 控制 DDP/DDP+FSDP/ZeRO 等数据并行方式的行为参数。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataParallelConfig {
    /// 数据并行度 (进程数/GPU 数)
    ///
    /// 通常等于 `num_gpus / (tp_degree * pp_degree)`
    /// 当使用 ZeRO/FSDP 时可以更大
    #[serde(default = "default_dp_degree")]
    pub dp_degree: usize,

    /// 梯度同步模式
    ///
    /// - `"allreduce"`: 标准 AllReduce (DDP)
    /// - `"reduce_scatter"`: ReduceScatter (更高效)
    /// - `"bucketed"`: 分桶 AllReduce (平衡延迟和吞吐)
    #[serde(default = "default_sync_mode")]
    pub sync_mode: String,

    /// 是否使用梯度累积进行虚拟增大 batch size
    ///
    /// 当 micro_batch_size 受显存限制时，
    /// 通过多次累积达到目标 batch size
    #[serde(default = "default_use_grad_accum")]
    pub use_gradient_accumulation: bool,

    /// 梯度累积步数
    #[serde(default = "default_accum_steps_dp")]
    pub accumulation_steps: usize,

    /// 是否在梯度累积期间不同步 (减少通信)
    ///
    /// 仅在最后一步才执行 AllReduce，
    /// 中间步数只做局部累积
    #[serde(default = "deferred_sync_default")]
    pub deferred_sync: bool,

    /// 检查点频率 (步数)
    ///
    /// 数据并行容错: 定期保存 DP rank 0 的状态
    #[serde(default = "default_ckpt_freq")]
    pub checkpoint_freq_steps: usize,
}

impl Default for DataParallelConfig {
    fn default() -> Self {
        Self {
            dp_degree: 1,
            sync_mode: "allreduce".to_string(),
            use_gradient_accumulation: true,
            accumulation_steps: 64,
            deferred_sync: true,
            checkpoint_freq_steps: 10000,
        }
    }
}

fn default_dp_degree() -> usize { 1 }
fn default_sync_mode() -> String { "allreduce".to_string() }
fn default_use_grad_accum() -> bool { true }
fn default_accum_steps_dp() -> usize { 64 }
fn deferred_sync_default() -> bool { true }
fn default_ckpt_freq() -> usize { 10000 }

impl DataParallelConfig {
    /// 创建 70B-Dense 数据并行配置
    pub fn for_70b_dense(num_gpus: usize, tp: usize, pp: usize) -> Self {
        let dp = num_gpus / (tp * pp).max(1);
        Self {
            dp_degree: dp.max(1),
            sync_mode: "reduce_scatter".to_string(), // 70B 用 ReduceScatter
            use_gradient_accumulation: true,
            accumulation_steps: 64,
            deferred_sync: true,
            checkpoint_freq_steps: 10000,
        }
    }

    /// 创建 14B-Dense 数据并行配置
    pub fn for_14b_dense(num_gpus: usize, tp: usize, pp: usize) -> Self {
        let dp = num_gpus / (tp * pp).max(1);
        Self {
            dp_degree: dp.max(1),
            sync_mode: "allreduce".to_string(), // 14B 用标准 AllReduce
            use_gradient_accumulation: true,
            accumulation_steps: 64,
            deferred_sync: true,
            checkpoint_freq_steps: 5000,
        }
    }

    /// 验证数据并行配置
    pub fn validate(&self) -> Result<(), DistributedTrainingError> {
        if self.dp_degree == 0 {
            return Err(DistributedTrainingError::InvalidConfiguration(
                "dp_degree must be > 0".to_string(),
            ));
        }
        if !["allreduce", "reduce_scatter", "bucketed"].contains(&self.sync_mode.as_str()) {
            return Err(DistributedTrainingError::InvalidConfiguration(format!(
                "Invalid sync_mode: {}", self.sync_mode
            )));
        }
        if self.use_gradient_accumulation && self.accumulation_steps == 0 {
            return Err(DistributedTrainingError::InvalidConfiguration(
                "accumulation_steps must be > 0 when gradient accumulation is enabled".to_string(),
            ));
        }
        Ok(())
    }

    /// 计算有效 batch size
    pub fn effective_batch_size(&self, micro_batch_size: usize) -> usize {
        if self.use_gradient_accumulation {
            micro_batch_size * self.accumulation_steps
        } else {
            micro_batch_size * self.dp_degree
        }
    }
}

// ==================== AMP (混合精度) 配置 ====================

/// 混合精度训练 (AMP) 配置
///
/// 控制自动混合精度的行为，包括:
/// - 权重/激活的数据类型
/// - Loss scaling 策略
/// - 特定层的精度覆盖
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmpConfig {
    /// 主训练精度 (权重和激活)
    ///
    /// 支持: "bf16" | "fp16" | "fp32"
    /// 推荐 70B 使用 bf16 (H100 原生支持，动态范围更大)
    #[serde(default = "default_amp_dtype")]
    pub dtype: String,

    /// 优化器状态精度
    ///
    /// 通常保持 fp32 以保证数值稳定性
    #[serde(default = "default_opt_dtype")]
    pub optimizer_dtype: String,

    /// Loss Scaling 类型
    ///
    /// - `"dynamic"`: 动态调整 scale factor (推荐 fp16)
    /// - `"static"`: 固定 scale factor (推荐 bf16)
    /// - `"none"`: 不使用 loss scaling (fp32)
    #[serde(default = "default_loss_scale_type")]
    pub loss_scale: String,

    /// 初始 loss scale (dynamic 模式下的起始值)
    #[serde(default = "default_init_loss_scale")]
    pub init_loss_scale: f64,

    /// Loss scale 增长因子 (每次连续 inf-free step 后乘以该值)
    #[serde(default = "default_loss_scale_growth")]
    pub loss_scale_growth_factor: f64,

    /// Loss scale 缩减因子 (遇到 inf 时除以该值)
    #[serde(default = "default_loss_scale_backoff")]
    pub loss_scale_backoff_factor: f64,

    /// 连续多少次 inf 后触发缩减
    #[serde(default = "default_loss_scale_window")]
    pub loss_scale_window: usize,

    /// 是否对特定层强制使用 FP32
    ///
    /// 如 LayerNorm、Softmax 等数值敏感操作
    #[serde(default = "default_force_fp32_layers")]
    pub force_fp32_layers: bool,

    /// 是否启用 TF32 (TensorFloat32) 加速
    ///
    /// TF32 有 FP32 的范围和 FP16 的速度 (Ampere+ GPU)
    /// 对矩阵运算几乎无损，推荐开启
    #[serde(default = "default_enable_tf32")]
    pub enable_tf32: bool,
}

impl Default for AmpConfig {
    fn default() -> Self {
        Self {
            dtype: "bf16".to_string(),
            optimizer_dtype: "fp32".to_string(),
            loss_scale: "static".to_string(), // bf16 用 static 即可
            init_loss_scale: 1.0,
            loss_scale_growth_factor: 2.0,
            loss_scale_backoff_factor: 0.5,
            loss_scale_window: 1000,
            force_fp32_layers: true,
            enable_tf32: true,
        }
    }
}

fn default_amp_dtype() -> String { "bf16".to_string() }
fn default_opt_dtype() -> String { "fp32".to_string() }
fn default_loss_scale_type() -> String { "static".to_string() }
fn default_init_loss_scale() -> f64 { 1.0 }
fn default_loss_scale_growth() -> f64 { 2.0 }
fn default_loss_scale_backoff() -> f64 { 0.5 }
fn default_loss_scale_window() -> usize { 1000 }
fn default_force_fp32_layers() -> bool { true }
fn default_enable_tf32() -> bool { true }

impl AmpConfig {
    /// 创建 70B-Dense AMP 配置 (BF16 优先)
    pub fn for_70b_dense() -> Self {
        Self {
            dtype: "bf16".to_string(),
            optimizer_dtype: "fp32".to_string(),
            loss_scale: "static".to_string(),
            init_loss_scale: 1.0,
            loss_scale_growth_factor: 2.0,
            loss_scale_backoff_factor: 0.5,
            loss_scale_window: 2000, // 70B 更保守
            force_fp32_layers: true,
            enable_tf32: true,
        }
    }

    /// 创建 14B-Dense AMP 配置
    pub fn for_14b_dense() -> Self {
        Self {
            dtype: "bf16".to_string(),
            optimizer_dtype: "fp32".to_string(),
            loss_scale: "static".to_string(),
            init_loss_scale: 1.0,
            loss_scale_growth_factor: 2.0,
            loss_scale_backoff_factor: 0.5,
            loss_scale_window: 1000,
            force_fp32_layers: true,
            enable_tf32: true,
        }
    }

    /// 验证 AMP 配置有效性
    pub fn validate(&self) -> Result<(), DistributedTrainingError> {
        if !["bf16", "fp16", "fp32"].contains(&self.dtype.as_str()) {
            return Err(DistributedTrainingError::InvalidConfiguration(format!(
                "Invalid dtype: {} (must be 'bf16', 'fp16', or 'fp32')",
                self.dtype
            )));
        }
        if !["dynamic", "static", "none"].contains(&self.loss_scale.as_str()) {
            return Err(DistributedTrainingError::InvalidConfiguration(format!(
                "Invalid loss_scale: {}", self.loss_scale
            )));
        }
        if self.init_loss_scale <= 0.0 {
            return Err(DistributedTrainingError::InvalidConfiguration(
                "init_loss_scale must be > 0".to_string(),
            ));
        }
        if !(0.0..1.0).contains(&self.loss_scale_backoff_factor) {
            return Err(DistributedTrainingError::InvalidConfiguration(
                "loss_scale_backoff_factor must be in (0, 1)".to_string(),
            ));
        }
        Ok(())
    }

    /// 获取每个参数的字节数
    pub fn bytes_per_param(&self) -> usize {
        match self.dtype.as_str() {
            "bf16" | "fp16" => 2,
            "fp32" => 4,
            _ => 4, // default to fp32
        }
    }

    /// 获取优化器状态每个参数的字节数
    pub fn optimizer_bytes_per_param(&self) -> usize {
        match self.optimizer_dtype.as_str() {
            "bf16" | "fp16" => 2,
            "fp32" => 4,
            _ => 4,
        }
    }
}

// ==================== 完整分布式训练配置 ====================

/// 分布式训练错误类型
#[derive(Debug, Error)]
pub enum DistributedTrainingError {
    /// 配置验证失败
    #[error("Configuration validation failed: {0}")]
    InvalidConfiguration(String),

    /// 显存估算失败
    #[error("Memory estimation failed: {0}")]
    MemoryEstimationError(String),

    /// 并行度计算失败
    #[error("Parallelism calculation failed: {0}")]
    ParallelismError(String),
}

/// 70B-Dense 完整分布式训练配置
///
/// 整合 FSDP、DeepSpeed、数据并行、混合精度为统一配置，
/// 用于驱动大规模分布式训练管线。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedTrainingConfig {
    /// FSDP 配置 (PyTorch 原生)
    #[serde(default)]
    pub fsdp: FsdpConfig,

    /// DeepSpeed 配置 (可选替代方案)
    #[serde(default)]
    pub deepspeed: DeepSpeedConfig,

    /// 数据并行配置
    #[serde(default)]
    pub data_parallel: DataParallelConfig,

    /// 混合精度配置
    #[serde(default)]
    pub amp: AmpConfig,

    // 全局硬件参数
    /// 总 GPU 数量
    #[serde(default = "default_total_gpus")]
    pub num_gpus: usize,

    /// 张量并行度
    #[serde(default = "default_tp")]
    pub tp_degree: usize,

    /// 流水线并行度
    #[serde(default = "default_pp")]
    pub pp_degree: usize,

    /// 每个 GPU 的显存 (GB)
    #[serde(default = "default_gpu_mem_gb")]
    pub gpu_memory_gb: usize,

    /// 模型总参数量 (用于显存估算)
    #[serde(default = "default_model_params")]
    pub model_parameters: u64,
}

impl Default for DistributedTrainingConfig {
    fn default() -> Self {
        Self {
            fsdp: FsdpConfig::default(),
            deepspeed: DeepSpeedConfig::default(),
            data_parallel: DataParallelConfig::default(),
            amp: AmpConfig::default(),
            num_gpus: 64,
            tp_degree: 8,
            pp_degree: 8,
            gpu_memory_gb: 80,
            model_parameters: 70_500_000_000, // 70.5B
        }
    }
}

fn default_total_gpus() -> usize { 64 }
fn default_tp() -> usize { 8 }
fn default_pp() -> usize { 8 }
fn default_gpu_mem_gb() -> usize { 80 }
fn default_model_params() -> u64 { 70_500_000_000 }

impl DistributedTrainingConfig {
    /// 创建针对 70B-Dense 模型优化的分布式配置
    ///
    /// 基于 64x H100 SXM5 集群预设最优参数:
    /// - TP=8, PP=8, DP=1 (纯 3D 并行)
    /// - FSDP HybridShard + ZeRO-3 兼容
    /// - BF16 混合精度
    pub fn for_70b_dense() -> Self {
        Self {
            fsdp: FsdpConfig::for_70b_dense(),
            deepspeed: DeepSpeedConfig::for_70b_dense(),
            data_parallel: DataParallelConfig::for_70b_dense(64, 8, 8),
            amp: AmpConfig::for_70b_dense(),
            num_gpus: 64,
            tp_degree: 8,
            pp_degree: 8,
            gpu_memory_gb: 80,
            model_parameters: 70_500_000_000,
        }
    }

    /// 创建针对 14B-Dense 模型的分布式配置
    pub fn for_14b_dense() -> Self {
        Self {
            fsdp: FsdpConfig::for_14b_dense(),
            deepspeed: DeepSpeedConfig::for_14b_dense(),
            data_parallel: DataParallelConfig::for_14b_dense(8, 4, 2),
            amp: AmpConfig::for_14b_dense(),
            num_gpus: 8,
            tp_degree: 4,
            pp_degree: 2,
            gpu_memory_gb: 80,
            model_parameters: 14_200_000_000,
        }
    }

    /// 验证所有配置段的有效性
    pub fn validate(&self) -> Result<(), DistributedTrainingError> {
        // 验证各子配置
        self.fsdp.validate()?;
        self.deepspeed.validate()?;
        self.data_parallel.validate()?;
        self.amp.validate()?;

        // 验证全局约束
        if self.num_gpus == 0 {
            return Err(DistributedTrainingError::InvalidConfiguration(
                "num_gpus must be > 0".to_string(),
            ));
        }

        let total_parallel = self.tp_degree * self.pp_degree;
        if total_parallel > self.num_gpus {
            return Err(DistributedTrainingError::InvalidConfiguration(format!(
                "TP({}) x PP({}) = {} exceeds num_gpus({})",
                self.tp_degree, self.pp_degree, total_parallel, self.num_gpus
            )));
        }

        if self.model_parameters == 0 {
            return Err(DistributedTrainingError::InvalidConfiguration(
                "model_parameters must be > 0".to_string(),
            ));
        }

        // 验证 DP 度计算一致性
        let expected_dp = self.num_gpus / total_parallel;
        if self.data_parallel.dp_degree != expected_dp && total_parallel <= self.num_gpus {
            // 允许手动设置不同的 DP 度，但发出警告
            log::warn!(
                "dp_degree={} differs from num_gpus/(tp*pp)={}/({}*{})={}",
                self.data_parallel.dp_degree,
                self.num_gpus,
                self.tp_degree,
                self.pp_degree,
                expected_dp
            );
        }

        Ok(())
    }

    /// 计算实际数据并行度
    pub fn effective_dp_degree(&self) -> usize {
        let total_tp_pp = self.tp_degree * self.pp_degree;
        if total_tp_pp == 0 {
            return 1;
        }
        self.num_gpus / total_tp_pp
    }

    /// 估算训练总显存需求 (GB)
    ///
    /// 基于 70B Dense 模型的经验公式:
    /// - 模型权重: params * bytes_per_param
    /// - 优化器状态: params * optimizer_bytes * 2 (AdamW momentum+variance)
    /// - 梯度: params * grad_bytes_per_param
    /// - 激活: 取决于序列长度和 batch size (粗略估计)
    /// - KV Cache 和临时缓冲: 额外开销
    pub fn estimate_training_memory_gb(&self) -> Result<f64, DistributedTrainingError> {
        let params = self.model_parameters as f64;
        let param_bytes = self.amp.bytes_per_param() as f64;
        let opt_bytes = self.optimizer_state_bytes_per_param() as f64;

        // 模型权重 (考虑 TP 分片)
        let weight_memory_gb =
            params * param_bytes / (1024.0 * 1024.0 * 1024.0) / self.tp_degree as f64;

        // AdamW 优化器状态 (momentum + variance, FP32)
        // 考虑 FSDP/ZeRO 分片
        let reduction = self.fsdp.memory_reduction_ratio();
        let optimizer_memory_gb =
            params * opt_bytes * 2.0 / (1024.0 * 1024.0 * 1024.0) * reduction;

        // 梯度 (BF16)
        let gradient_memory_gb =
            params * 2.0 / (1024.0 * 1024.0 * 1024.0) * reduction;

        // 激活显存 (粗略估计: 与 hidden_size^2 * layers * seq_len 成正比)
        // 70B: hidden=8192, layers=80, 假设 seq=8192, batch=8, checkpointing enabled
        let activation_estimate_gb = self.estimate_activation_memory_gb();

        // KV Cache 和临时缓冲 (MLA 压缩后较小)
        let kv_cache_gb = self.estimate_kv_cache_gb();

        // 框架开销 (PyTorch runtime, CUDA contexts, etc.)
        let overhead_gb = 2.0; // 固定开销

        let total_gb = weight_memory_gb + optimizer_memory_gb + gradient_memory_gb
            + activation_estimate_gb + kv_cache_gb + overhead_gb;

        Ok(total_gb)
    }

    /// 估算激活显存 (GB)
    ///
    /// 基于简化公式: O(L * H^2 * S * B * bytes)
    /// 考虑梯度检查点的节省效果
    fn estimate_activation_memory_gb(&self) -> f64 {
        let h = 8192.0_f64; // hidden_size
        let l = 80.0_f64;    // num_layers
        let s = 8192.0_f64;  // seq_length (假设)
        let b = 8.0_f64;     // micro_batch_size

        // 完整激活 (不考虑 checkpointing)
        let full_activation_bytes = l * h * h * s * b * 2.0; // BF16

        // 激活检查点通常节省 60-80% 显存
        let checkpoint_savings = if self.fsdp.activation_checkpointing {
            0.25 // 保留 25%
        } else {
            1.0 // 100%
        };

        // TP 分片
        let tp_sharding = 1.0 / self.tp_degree as f64;

        (full_activation_bytes * checkpoint_savings * tp_sharding) / (1024.0 * 1024.0 * 1024.0)
    }

    /// 估算 MLA 压缩后的 KV Cache (GB)
    fn estimate_kv_cache_gb(&self) -> f64 {
        // MLA: latent_dim=2048 (vs full hidden=8192), 80 layers
        let latent_dim = 2048.0_f64;
        let num_layers = 80.0_f64;
        let avg_seq_len = 4096.0_f64; // 平均推理/训练序列长度

        // K + V, each with latent_dim, BF16
        let kv_bytes = num_layers * latent_dim * avg_seq_len * 2.0 * 2.0;
        kv_bytes / (1024.0 * 1024.0 * 1024.0)
    }

    /// 获取优化器状态每个参数的字节数
    pub fn optimizer_state_bytes_per_param(&self) -> usize {
        self.amp.optimizer_bytes_per_param()
    }

    /// 估算推理显存需求 (GB)
    ///
    /// 仅包含模型权重 + KV Cache + 运行时开销
    pub fn estimate_inference_memory_gb(&self, precision: &str) -> Result<f64, DistributedTrainingError> {
        let params = self.model_parameters as f64;
        let bytes_per_param = match precision {
            "fp16" | "bf16" => 2.0_f64,
            "int8" => 1.0_f64,
            "int4" => 0.5_f64,
            "fp32" => 4.0_f64,
            _ => {
                return Err(DistributedTrainingError::MemoryEstimationError(format!(
                    "Unsupported precision: {}", precision
                )))
            }
        };

        let model_weight_gb = params * bytes_per_param / (1024.0 * 1024.0 * 1024.0);
        let kv_cache_gb = self.estimate_kv_cache_gb();
        let overhead_gb = 2.0;

        Ok(model_weight_gb + kv_cache_gb + overhead_gb)
    }

    /// 检查当前硬件是否满足训练需求
    pub fn check_hardware_fit(&self) -> HardwareFitResult {
        match self.estimate_training_memory_gb() {
            Ok(required_gb) => {
                let available_gb = self.num_gpus as f64 * self.gpu_memory_gb as f64;
                let per_gpu_required = required_gb / self.num_gpus as f64;
                let fits = per_gpu_required <= self.gpu_memory_gb as f64;

                HardwareFitResult {
                    fits,
                    required_memory_gb: required_gb,
                    available_memory_gb: available_gb,
                    per_gpu_required_gb: per_gpu_required,
                    per_gpu_available_gb: self.gpu_memory_gb as f64,
                    utilization_pct: (per_gpu_required / self.gpu_memory_gb as f64) * 100.0,
                    error: None,
                }
            }
            Err(e) => HardwareFitResult {
                fits: false,
                required_memory_gb: 0.0,
                available_memory_gb: self.num_gpus as f64 * self.gpu_memory_gb as f64,
                per_gpu_required_gb: 0.0,
                per_gpu_available_gb: self.gpu_memory_gb as f64,
                utilization_pct: 0.0,
                error: Some(e.to_string()),
            },
        }
    }

    /// 计算 70B vs 14B 的资源对比
    pub fn compare_with_14b(&self) -> ModelComparison {
        let config_14b = Self::for_14b_dense();

        ModelComparison {
            params_70b: self.model_parameters,
            params_14b: config_14b.model_parameters,
            param_ratio: self.model_parameters as f64 / config_14b.model_parameters as f64,

            gpus_70b: self.num_gpus,
            gpus_14b: config_14b.num_gpus,
            gpu_ratio: self.num_gpus as f64 / config_14b.num_gpus as f64,

            memory_70b_gb: self.estimate_training_memory_gb().unwrap_or(0.0),
            memory_14b_gb: config_14b.estimate_training_memory_gb().unwrap_or(0.0),

            tp_70b: self.tp_degree,
            tp_14b: config_14b.tp_degree,

            pp_70b: self.pp_degree,
            pp_14b: config_14b.pp_degree,
        }
    }
}

/// 硬件适配性检查结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareFitResult {
    /// 是否满足显存需求
    pub fits: bool,
    /// 所需总显存 (GB)
    pub required_memory_gb: f64,
    /// 可用总显存 (GB)
    pub available_memory_gb: f64,
    /// 每卡所需显存 (GB)
    pub per_gpu_required_gb: f64,
    /// 每卡可用显存 (GB)
    pub per_gpu_available_gb: f64,
    /// 显存利用率 (%)
    pub utilization_pct: f64,
    /// 错误信息 (如果有)
    #[serde(default)]
    pub error: Option<String>,
}

/// 模型对比结果 (70B vs 14B)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    /// 70B 参数量
    pub params_70b: u64,
    /// 14B 参数量
    pub params_14b: u64,
    /// 参数量比值
    pub param_ratio: f64,
    /// 70B GPU 数量
    pub gpus_70b: usize,
    /// 14B GPU 数量
    pub gpus_14b: usize,
    /// GPU 数量比值
    pub gpu_ratio: f64,
    /// 70B 估算训练显存 (GB)
    pub memory_70b_gb: f64,
    /// 14B 估算训练显存 (GB)
    pub memory_14b_gb: f64,
    /// 70B 张量并行度
    pub tp_70b: usize,
    /// 14B 张量并行度
    pub tp_14b: usize,
    /// 70B 流水线并行度
    pub pp_70b: usize,
    /// 14B 流水线并行度
    pub pp_14b: usize,
}

impl fmt::Display for ModelComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "OpenMini-70B vs OpenMini-14B Comparison:")?;
        writeln!(f, "  Parameters:  {:.1}B vs {:.1}B ({:.2}x)",
            self.params_70b as f64 / 1e9,
            self.params_14b as f64 / 1e9,
            self.param_ratio)?;
        writeln!(f,  "  GPUs:        {} vs {} ({:.1}x)",
            self.gpus_70b, self.gpus_14b, self.gpu_ratio)?;
        writeln!(f,  "  Est. Memory: {:.1} GB vs {:.1} GB ({:.2}x)",
            self.memory_70b_gb, self.memory_14b_gb,
            self.memory_70b_gb / self.memory_14b_gb.max(1.0))?;
        writeln!(f,  "  Parallelism: TP={},PP={} vs TP={},PP={}",
            self.tp_70b, self.pp_70b, self.tp_14b, self.pp_14b)
    }
}

// ==================== 单元测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    // ========== FSDP 配置测试 ==========

    #[test]
    fn test_fsdp_default_config_validity() {
        let config = FsdpConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.sharding_strategy, ShardingStrategy::HybridShard);
        assert!(!config.offload_params);
        assert!(config.limit_all_gathers);
        assert!(config.activation_checkpointing);
    }

    #[test]
    fn test_fsdp_for_70b_dense() {
        let config = FsdpConfig::for_70b_dense();
        assert_eq!(config.sharding_strategy, ShardingStrategy::HybridShard);
        assert_eq!(config.min_num_params, 10_000_000); // 70B 更大粒度
        assert!(!config.offload_params); // H100 不需要卸载
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_fsdp_for_14b_dense() {
        let config = FsdpConfig::for_14b_dense();
        assert_eq!(config.sharding_strategy, ShardingStrategy::FullShard);
        assert_eq!(config.min_num_params, 1_000_000);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_fsdp_invalid_min_params() {
        let mut config = FsdpConfig::default();
        config.min_num_params = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_sharding_strategy_display() {
        assert_eq!(ShardingStrategy::FullShard.to_string(), "full_shard");
        assert_eq!(ShardingStrategy::ShardGradOp.to_string(), "shard_grad_op");
        assert_eq!(ShardingStrategy::HybridShard.to_string(), "hybrid_shard");
    }

    #[test]
    fn test_fsdp_memory_reduction_ratios() {
        let full = FsdpConfig {
            sharding_strategy: ShardingStrategy::FullShard,
            offload_params: true,
            ..Default::default()
        };
        assert!(full.memory_reduction_ratio() < 0.2);

        let hybrid = FsdpConfig {
            sharding_strategy: ShardingStrategy::HybridShard,
            offload_params: false,
            ..Default::default()
        };
        let ratio = hybrid.memory_reduction_ratio();
        assert!(ratio > 0.35 && ratio < 0.50);
    }

    // ========== DeepSpeed 配置测试 ==========

    #[test]
    fn test_deepspeed_default_config() {
        let config = DeepSpeedConfig::default();
        assert_eq!(config.stage, DeepSpeedStage::Stage3);
        assert!(config.overlap_comm);
        assert!(config.reduce_scatter);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_deepspeed_for_70b_dense() {
        let config = DeepSpeedConfig::for_70b_dense();
        assert_eq!(config.stage, DeepSpeedStage::Stage3);
        assert_eq!(config.offload_optimizer, Some("cpu".to_string()));
        assert_eq!(config.reduce_bucket_size, 2_000_000_000);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_deepspeed_stage_display() {
        assert_eq!(DeepSpeedStage::Stage1.to_string(), "stage1");
        assert_eq!(DeepSpeedStage::Stage2.to_string(), "stage2");
        assert_eq!(DeepSpeedStage::Stage3.to_string(), "stage3");
    }

    #[test]
    fn test_deepspeed_invalid_offload() {
        let mut config = DeepSpeedConfig::default();
        config.offload_optimizer = Some("invalid".to_string());
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_deepspeed_zero_stage() {
        let config = DeepSpeedConfig::default();
        assert_eq!(config.zero_stage(), 3);
    }

    // ========== 数据并行配置测试 ==========

    #[test]
    fn test_data_parallel_default() {
        let config = DataParallelConfig::default();
        assert_eq!(config.dp_degree, 1);
        assert!(config.use_gradient_accumulation);
        assert_eq!(config.accumulation_steps, 64);
        assert!(config.deferred_sync);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_data_parallel_for_70b() {
        let config = DataParallelConfig::for_70b_dense(64, 8, 8);
        assert_eq!(config.dp_degree, 1); // 64/(8*8)=1
        assert_eq!(config.sync_mode, "reduce_scatter");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_data_parallel_effective_batch_size() {
        let config = DataParallelConfig::default();
        assert_eq!(config.effective_batch_size(8), 512); // 8*64
    }

    #[test]
    fn test_data_parallel_invalid_sync_mode() {
        let mut config = DataParallelConfig::default();
        config.sync_mode = "invalid".to_string();
        assert!(config.validate().is_err());
    }

    // ========== AMP 配置测试 ==========

    #[test]
    fn test_amp_default_config() {
        let config = AmpConfig::default();
        assert_eq!(config.dtype, "bf16");
        assert_eq!(config.optimizer_dtype, "fp32");
        assert_eq!(config.loss_scale, "static");
        assert!(config.force_fp32_layers);
        assert!(config.enable_tf32);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_amp_for_70b_dense() {
        let config = AmpConfig::for_70b_dense();
        assert_eq!(config.dtype, "bf16");
        assert_eq!(config.loss_scale_window, 2000); // 70B 更保守
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_amp_bytes_per_param() {
        let config = AmpConfig::default();
        assert_eq!(config.bytes_per_param(), 2); // bf16
        assert_eq!(config.optimizer_bytes_per_param(), 4); // fp32

        let fp32_config = AmpConfig {
            dtype: "fp32".to_string(),
            ..Default::default()
        };
        assert_eq!(fp32_config.bytes_per_param(), 4);
    }

    #[test]
    fn test_amp_invalid_dtype() {
        let mut config = AmpConfig::default();
        config.dtype = "int8".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_amp_invalid_loss_scale_backoff() {
        let mut config = AmpConfig::default();
        config.loss_scale_backoff_factor = 1.5; // Must be in (0, 1)
        assert!(config.validate().is_err());
    }

    // ========== 完整分布式配置测试 ==========

    #[test]
    fn test_distributed_config_default() {
        let config = DistributedTrainingConfig::default();
        assert_eq!(config.num_gpus, 64);
        assert_eq!(config.tp_degree, 8);
        assert_eq!(config.pp_degree, 8);
        assert_eq!(config.model_parameters, 70_500_000_000);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_distributed_config_for_70b_dense() {
        let config = DistributedTrainingConfig::for_70b_dense();
        assert!(config.validate().is_ok());

        // 验证关键 70B 参数
        assert_eq!(config.fsdp.sharding_strategy, ShardingStrategy::HybridShard);
        assert_eq!(config.deepspeed.stage, DeepSpeedStage::Stage3);
        assert_eq!(config.amp.dtype, "bf16");

        // 验证并行度
        assert_eq!(config.effective_dp_degree(), 1); // 64/(8*8)=1
    }

    #[test]
    fn test_distributed_config_for_14b_dense() {
        let config = DistributedTrainingConfig::for_14b_dense();
        assert!(config.validate().is_ok());
        assert_eq!(config.num_gpus, 8);
        assert_eq!(config.tp_degree, 4);
        assert_eq!(config.pp_degree, 2);
        assert_eq!(config.model_parameters, 14_200_000_000);
    }

    #[test]
    fn test_validate_exceeds_gpus() {
        let mut config = DistributedTrainingConfig::default();
        config.tp_degree = 100; // 远超 64 GPUs
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_zero_gpus() {
        let mut config = DistributedTrainingConfig::default();
        config.num_gpus = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_effective_dp_degree_calculation() {
        let config = DistributedTrainingConfig::for_70b_dense();
        assert_eq!(config.effective_dp_degree(), 1);

        let custom = DistributedTrainingConfig {
            num_gpus: 32,
            tp_degree: 4,
            pp_degree: 2,
            ..Default::default()
        };
        assert_eq!(custom.effective_dp_degree(), 4); // 32/(4*2)=4
    }

    // ========== 显存估算测试 ==========

    #[test]
    fn test_estimate_training_memory_70b() {
        let config = DistributedTrainingConfig::for_70b_dense();
        let memory_gb = config.estimate_training_memory_gb().unwrap();

        // 70B 训练应该需要大量显存 (>500 GB 总计)
        assert!(memory_gb > 500.0, "70B training should need >500GB, got {:.1}", memory_gb);
        // 但不应该超过物理极限 (64 * 80GB = 5120GB)
        assert!(memory_gb < 6000.0, "70B training should need <6000GB, got {:.1}", memory_gb);
    }

    #[test]
    fn test_estimate_training_memory_14b() {
        let config = DistributedTrainingConfig::for_14b_dense();
        let memory_gb = config.estimate_training_memory_gb().unwrap();

        // 14B 训练应该比 70B 少很多
        assert!(memory_gb < 200.0, "14B training should need <200GB, got {:.1}", memory_gb);
        // 但也应该有合理的大小
        assert!(memory_gb > 30.0, "14B training should need >30GB, got {:.1}", memory_gb);
    }

    #[test]
    fn test_estimate_inference_memory_by_precision() {
        let config = DistributedTrainingConfig::for_70b_dense();

        let fp16_mem = config.estimate_inference_memory_gb("bf16").unwrap();
        let int8_mem = config.estimate_inference_memory_gb("int8").unwrap();
        let int4_mem = config.estimate_inference_memory_gb("int4").unwrap();

        // FP16 应该最大
        assert!(fp16_mem > int8_mem);
        assert!(int8_mem > int4_mem);

        // 70B FP16 推理应该在 140GB 左右
        assert!(fp16_mem > 120.0 && fp16_mem < 180.0,
            "70B BF16 inference should be 120-180GB, got {:.1}", fp16_mem);

        // INT4 应该显著更小
        assert!(int4_mem < 60.0,
            "70B INT4 inference should be <60GB, got {:.1}", int4_mem);
    }

    #[test]
    fn test_estimate_inference_invalid_precision() {
        let config = DistributedTrainingConfig::for_70b_dense();
        assert!(config.estimate_inference_memory_gb("invalid").is_err());
    }

    // ========== 硬件适配性测试 ==========

    #[test]
    fn test_hardware_fit_70b_on_h100_cluster() {
        let config = DistributedTrainingConfig::for_70b_dense();
        let result = config.check_hardware_fit();

        // 64x H100 80GB 应该能够运行 70B (with FSDP/ZeRO)
        // 但可能利用率较高
        assert!(result.utilization_pct > 50.0, "Should have reasonable utilization");
        assert!(result.per_gpu_required_gb > 10.0, "Each GPU needs significant memory");
    }

    #[test]
    fn test_hardware_fit_result_fields() {
        let config = DistributedTrainingConfig::for_70b_dense();
        let result = config.check_hardware_fit();

        assert!(result.available_memory_gb > 0.0);
        assert!(result.required_memory_gb > 0.0);
        assert!(result.per_gpu_available_gb > 0.0);
    }

    // ========== 70B vs 14B 对比测试 ==========

    #[test]
    fn test_compare_70b_vs_14b() {
        let config_70b = DistributedTrainingConfig::for_70b_dense();
        let comparison = config_70b.compare_with_14b();

        // 70B 参数量应该是 14B 的 ~5 倍
        assert!(comparison.param_ratio > 4.0 && comparison.param_ratio < 6.0,
            "Param ratio should be ~5x, got {:.2}", comparison.param_ratio);

        // 70B GPU 数量应该是 14B 的 8 倍
        assert!((comparison.gpu_ratio - 8.0).abs() < 0.1,
            "GPU ratio should be 8x, got {:.1}", comparison.gpu_ratio);

        // 70B 显存应该远大于 14B
        assert!(comparison.memory_70b_gb > comparison.memory_14b_gb * 3.0,
            "70B should need significantly more memory than 14B");

        // 打印对比信息 (可在测试输出中查看)
        let display = format!("{}", comparison);
        assert!(display.contains("70B"));
        assert!(display.contains("14B"));
    }

    #[test]
    fn test_comparison_display_format() {
        let config = DistributedTrainingConfig::for_70b_dense();
        let comparison = config.compare_with_14b();
        let display = format!("{}", comparison);

        // 验证输出格式包含关键字段
        assert!(display.contains("Parameters:"));
        assert!(display.contains("GPUs:"));
        assert!(display.contains("Est. Memory:"));
        assert!(display.contains("Parallelism:"));
    }

    // ========== 序列化/反序列化测试 ==========

    #[test]
    fn test_serialize_deserialize_full_config() {
        let config = DistributedTrainingConfig::for_70b_dense();

        // JSON 序列化
        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.is_empty());

        // JSON 反序列化
        let deserialized: DistributedTrainingConfig =
            serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.num_gpus, config.num_gpus);
        assert_eq!(deserialized.tp_degree, config.tp_degree);
        assert_eq!(deserialized.fsdp.sharding_strategy, config.fsdp.sharding_strategy);

        // TOML 序列化 (如果可用)
        #[cfg(feature = "toml_serialization")]
        {
            let toml_str = toml::to_string(&config).unwrap();
            let from_toml: DistributedTrainingConfig = toml::from_str(&toml_str).unwrap();
            assert_eq!(from_toml.model_parameters, config.model_parameters);
        }
    }

    #[test]
    fn test_serialize_sharding_strategy() {
        let strategies = vec![
            ShardingStrategy::FullShard,
            ShardingStrategy::ShardGradOp,
            ShardingStrategy::HybridShard,
        ];

        for strategy in strategies {
            let json = serde_json::to_string(&strategy).unwrap();
            let deserialized: ShardingStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(strategy, deserialized);
        }
    }

    // ========== 边界条件测试 ==========

    #[test]
    fn test_single_gpu_config() {
        let config = DistributedTrainingConfig {
            num_gpus: 1,
            tp_degree: 1,
            pp_degree: 1,
            model_parameters: 7_000_000_000, // 7B
            ..Default::default()
        };
        // 单卡小模型应该有效
        assert!(config.validate().is_ok());
        assert_eq!(config.effective_dp_degree(), 1);
    }

    #[test]
    fn test_large_scale_config() {
        // 测试超大规模配置 (未来 1T 参数模型)
        let config = DistributedTrainingConfig {
            num_gpus: 1024,
            tp_degree: 32,
            pp_degree: 16,
            gpu_memory_gb: 80,
            model_parameters: 1_000_000_000_000, // 1T parameters
            ..Default::default()
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.effective_dp_degree(), 2); // 1024/(32*16)=2
    }
}
