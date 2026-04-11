//! 分布式推理配置模块
//!
//! 定义分布式系统的核心配置参数，包括：
//! - 集群拓扑（world_size, rank, tp_degree）
//! - 通信后端选择（NCCL/Gloo/MPI/Local）
//! - GPU 设备分配
//! - 性能调优选项
//!
//! ## 配置层次
//!
//! ```
//! DistributedConfig
//! ├── 集群配置 (world_size, rank, tp_degree)
//! ├── 通信配置 (backend, master_addr, master_port)
//! ├── 设备配置 (cuda_device_ids)
//! └── 性能配置 (micro_batch_size, overlap_comm_compute)
//! ```
//!
//! ## 使用示例
//!
//! ```rust,ignore
//! use openmini_server::distributed::config::DistributedConfig;
//!
//! // 本地测试模式（2卡模拟）
//! let config = DistributedConfig::for_local_testing(2);
//! config.validate()?;
//!
//! // 生产环境配置
//! let config = DistributedConfig {
//!     world_size: 4,
//!     rank: 0,
//!     tp_degree: 4,
//!     ..Default::default()
//! };
//! ```

use log::{info, warn};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// 分布式错误类型 (错误码前缀: DIS)
///
/// 涵盖分布式系统中的配置、通信、张量并行等错误。
#[derive(Debug, Error)]
pub enum DistributedError {
    /// 配置验证失败 (DIS001)
    #[error("Configuration validation failed: {0}")]
    ConfigValidation(String),

    /// 通信操作失败 (DIS002)
    #[error("Communication operation failed: {0}")]
    Communication(String),

    /// 张量并行操作失败 (DIS003)
    #[error("Tensor parallel operation failed: {0}")]
    TensorParallel(String),

    /// 路由器操作失败 (DIS004)
    #[error("Router operation failed: {0}")]
    Router(String),

    /// 不支持的操作 (DIS005)
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}

/// 通信后端类型
///
/// 支持多种通信后端以适应不同部署场景：
/// - **NCCL**: NVIDIA GPU 高性能通信（推荐用于多卡训练/推理）
/// - **Gloo**: CPU 通用通信（适合无GPU或跨节点场景）
/// - **MPI**: 标准MPI接口（兼容HPC集群）
/// - **Local**: 单机模拟模式（开发测试用，始终可用）
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CommBackend {
    /// NVIDIA NCCL (需要CUDA)
    NCCL,

    /// Facebook Gloo (CPU通用)
    Gloo,

    /// MPI (HPC集群)
    MPI,

    /// 本地模拟（单机测试）
    Local,
}

impl std::fmt::Display for CommBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NCCL => write!(f, "nccl"),
            Self::Gloo => write!(f, "gloo"),
            Self::MPI => write!(f, "mpi"),
            Self::Local => write!(f, "local"),
        }
    }
}

impl Default for CommBackend {
    fn default() -> Self {
        Self::Local
    }
}

/// 分布式推理核心配置
///
/// 封装了启动分布式推理所需的所有参数。
/// 通过 `validate()` 方法确保配置的一致性和有效性。
///
/// # 性能目标
///
/// | 配置项 | 推荐值 | 说明 |
/// |--------|--------|------|
/// | `tp_degree` | 2-4 | 2卡~1.9x, 4卡~3.6x 加速比 |
/// | `micro_batch_size` | 1-8 | 较小值降低显存峰值 |
/// | `overlap_comm_compute` | true | 通信计算重叠提升10-20%吞吐 |
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// 世界大小（总进程数/总GPU数）
    ///
    /// 范围: [1, 64]
    /// 对于张量并行，通常等于tp_degree
    pub world_size: usize,

    /// 当前进程排名（0-based）
    ///
    /// 范围: [0, world_size-1]
    /// rank=0 通常作为主进程负责协调
    pub rank: usize,

    /// 张量并行度（使用的GPU数量）
    ///
    /// 必须能整除 world_size
    /// 支持 1, 2, 4, 8 等值（推荐2-4）
    pub tp_degree: usize,

    /// 主节点地址（用于进程间初始化握手）
    ///
    /// 格式: IP地址或主机名
    /// 默认: "127.0.0.1" (本地测试)
    pub master_addr: String,

    /// 主节点端口
    ///
    /// 范围: [1024, 65535]
    /// 默认: 29500 (NCCL默认端口范围起始)
    pub master_port: u16,

    /// 通信后端类型
    ///
    /// 选择合适的后端以获得最佳性能：
    /// - 多GPU: NCCL
    /// - CPU集群: Gloo
    /// - HPC: MPI
    /// - 开发测试: Local
    pub backend: CommBackend,

    /// CUDA设备ID列表
    ///
    /// 指定当前进程可见的GPU设备。
    /// 长度应等于 tp_degree / (world_size / tp_degree) 或 1
    ///
    /// 示例: [0, 1] 表示使用GPU 0和1
    pub cuda_device_ids: Vec<usize>,

    /// 微批次大小
    ///
    /// 用于流水线并行的微批次切分。
    /// 较小值可降低显存使用，但可能增加通信开销。
    ///
    /// 推荐: 1-8（根据模型大小和显存调整）
    pub micro_batch_size: usize,

    /// 是否启用通信计算重叠
    ///
    /// 当为true时，通信操作与计算异步执行，
    /// 可提升整体吞吐量10-20%。
    ///
    /// 注意: 需要底层后端支持异步操作
    pub overlap_comm_compute: bool,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            rank: 0,
            tp_degree: 1,
            master_addr: "127.0.0.1".to_string(),
            master_port: 29500,
            backend: CommBackend::Local,
            cuda_device_ids: vec![0],
            micro_batch_size: 1,
            overlap_comm_compute: false,
        }
    }
}

impl DistributedConfig {
    /// 创建本地测试配置
    ///
    /// 用于开发和单元测试的便捷方法。
    /// 自动配置合理的默认值，使用 Local 后端。
    ///
    /// # 参数
    ///
    /// * `num_gpus` - 模拟的GPU数量（1-8）
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// // 创建2卡本地测试配置
    /// let config = DistributedConfig::for_local_testing(2);
    /// assert_eq!(config.world_size, 2);
    /// assert_eq!(config.tp_degree, 2);
    /// assert_eq!(config.backend, CommBackend::Local);
    /// ```
    pub fn for_local_testing(num_gpus: usize) -> Self {
        info!("Creating local testing config for {} GPUs", num_gpus);

        let device_ids: Vec<usize> = (0..num_gpus).collect();

        Self {
            world_size: num_gpus,
            rank: 0, // 默认rank 0，测试时可修改
            tp_degree: num_gpus,
            master_addr: "127.0.0.1".to_string(),
            master_port: 29500,
            backend: CommBackend::Local,
            cuda_device_ids: device_ids,
            micro_batch_size: 1,
            overlap_comm_compute: false,
        }
    }

    /// 验证配置的有效性
    ///
    /// 检查所有配置参数的一致性和合法性。
    /// 应在初始化分布式环境前调用。
    ///
    /// # 验证规则
    ///
    /// 1. `world_size >= 1`
    /// 2. `rank < world_size`
    /// 3. `tp_degree >= 1 && tp_degree <= world_size`
    /// 4. `world_size % tp_degree == 0` (tp_degree必须整除world_size)
    /// 5. `master_port > 1024`
    /// 6. `cuda_device_ids` 非空且所有ID唯一
    /// 7. `micro_batch_size >= 1`
    ///
    /// # 错误
    ///
    /// 返回 [`DistributedError::ConfigValidation`] 如果任何验证失败。
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// let mut config = DistributedConfig::for_local_testing(4);
    /// config.rank = 5; // 无效：超过world_size
    /// assert!(config.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), DistributedError> {
        info!("Validating distributed configuration");

        // 规则1: world_size >= 1
        if self.world_size == 0 {
            return Err(DistributedError::ConfigValidation(
                "world_size must be >= 1".to_string(),
            ));
        }

        // 规则2: rank < world_size
        if self.rank >= self.world_size {
            return Err(DistributedError::ConfigValidation(format!(
                "rank ({}) must be < world_size ({})",
                self.rank, self.world_size
            )));
        }

        // 规则3: tp_degree >= 1 && tp_degree <= world_size
        if self.tp_degree == 0 || self.tp_degree > self.world_size {
            return Err(DistributedError::ConfigValidation(format!(
                "tp_degree ({}) must be in range [1, {}]",
                self.tp_degree, self.world_size
            )));
        }

        // 规则4: tp_degree必须整除world_size
        if self.world_size % self.tp_degree != 0 {
            return Err(DistributedError::ConfigValidation(format!(
                "tp_degree ({}) must divide world_size ({}) evenly",
                self.tp_degree, self.world_size
            )));
        }

        // 规则5: master_port > 1024
        if self.master_port <= 1024 {
            return Err(DistributedError::ConfigValidation(format!(
                "master_port ({}) must be > 1024",
                self.master_port
            )));
        }

        // 规则6: cuda_device_ids非空且唯一
        if self.cuda_device_ids.is_empty() {
            return Err(DistributedError::ConfigValidation(
                "cuda_device_ids must not be empty".to_string(),
            ));
        }

        let unique_devices: std::collections::HashSet<_> =
            self.cuda_device_ids.iter().collect();
        if unique_devices.len() != self.cuda_device_ids.len() {
            return Err(DistributedError::ConfigValidation(
                "cuda_device_ids must contain unique values".to_string(),
            ));
        }

        // 规则7: micro_batch_size >= 1
        if self.micro_batch_size == 0 {
            return Err(DistributedError::ConfigValidation(
                "micro_batch_size must be >= 1".to_string(),
            ));
        }

        // 警告检查（不阻止运行但记录日志）
        if self.world_size > 8 && self.backend == CommBackend::Local {
            warn!(
                "Large world_size ({}) with Local backend may have performance issues",
                self.world_size
            );
        }

        if !self.overlap_comm_compute && self.world_size > 2 {
            info!(
                "Consider enabling overlap_comm_compute for better performance with {} GPUs",
                self.world_size
            );
        }

        info!(
            "Configuration validated successfully: world_size={}, rank={}, tp_degree={}, backend={}",
            self.world_size, self.rank, self.tp_degree, self.backend
        );

        Ok(())
    }

    /// 获取每个TP组的进程数
    ///
    /// 当存在多个TP组时（如DP+TP混合并行），
    /// 返回每个TP组内的进程数。
    pub fn processes_per_tp_group(&self) -> usize {
        self.world_size / self.tp_degree
    }

    /// 获取当前进程所在的TP组ID
    pub fn tp_group_id(&self) -> usize {
        self.rank / self.processes_per_tp_group()
    }

    /// 获取当前进程在TP组内的本地rank
    pub fn local_rank(&self) -> usize {
        self.rank % self.tp_degree
    }

    /// 是否为主进程（rank 0）
    pub fn is_master(&self) -> bool {
        self.rank == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DistributedConfig::default();
        assert_eq!(config.world_size, 1);
        assert_eq!(config.rank, 0);
        assert_eq!(config.tp_degree, 1);
        assert_eq!(config.backend, CommBackend::Local);
    }

    #[test]
    fn test_local_testing_config_2gpus() {
        let config = DistributedConfig::for_local_testing(2);
        assert_eq!(config.world_size, 2);
        assert_eq!(config.tp_degree, 2);
        assert_eq!(config.cuda_device_ids, vec![0, 1]);
        assert_eq!(config.backend, CommBackend::Local);
    }

    #[test]
    fn test_local_testing_config_4gpus() {
        let config = DistributedConfig::for_local_testing(4);
        assert_eq!(config.world_size, 4);
        assert_eq!(config.tp_degree, 4);
        assert_eq!(config.cuda_device_ids.len(), 4);
    }

    #[test]
    fn test_validate_valid_config() {
        let config = DistributedConfig::for_local_testing(2);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_invalid_rank() {
        let mut config = DistributedConfig::for_local_testing(2);
        config.rank = 3; // 超过world_size
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_tp_degree() {
        let mut config = DistributedConfig::for_local_testing(4);
        config.tp_degree = 3; // 不能整除world_size
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_zero_world_size() {
        let mut config = DistributedConfig::default();
        config.world_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_invalid_port() {
        let mut config = DistributedConfig::default();
        config.master_port = 80; // <= 1024
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_empty_device_ids() {
        let mut config = DistributedConfig::default();
        config.cuda_device_ids = vec![];
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_duplicate_device_ids() {
        let mut config = DistributedConfig::default();
        config.cuda_device_ids = vec![0, 0]; // 重复
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_comm_backend_display() {
        assert_eq!(CommBackend::NCCL.to_string(), "nccl");
        assert_eq!(CommBackend::Gloo.to_string(), "gloo");
        assert_eq!(CommBackend::MPI.to_string(), "mpi");
        assert_eq!(CommBackend::Local.to_string(), "local");
    }

    #[test]
    fn test_helper_methods() {
        let config = DistributedConfig::for_local_testing(4);

        // is_master
        assert!(config.is_master());

        // processes_per_tp_group
        assert_eq!(config.processes_per_tp_group(), 1);

        // tp_group_id and local_rank for rank 0
        assert_eq!(config.tp_group_id(), 0);
        assert_eq!(config.local_rank(), 0);

        // 测试其他rank
        let mut config_rank2 = config.clone();
        config_rank2.rank = 2;
        assert!(!config_rank2.is_master());
        assert_eq!(config_rank2.local_rank(), 2);
    }

    #[test]
    fn test_serialize_deserialize() {
        let config = DistributedConfig::for_local_testing(2);
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: DistributedConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.world_size, config.world_size);
        assert_eq!(deserialized.tp_degree, config.tp_degree);
        assert_eq!(deserialized.backend, config.backend);
    }
}
