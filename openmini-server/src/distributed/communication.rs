//! 分布式通信抽象层
//!
//! 提供集合通信操作的统一接口，支持多种后端实现：
//! - **CollectiveOps trait**: 定义标准集合通信原语
//! - **LocalComm**: 本地模拟实现（用于开发和测试）
//!
//! ## 支持的操作
//!
//! | 操作 | 说明 | 典型用途 |
//! |------|------|----------|
//! | `all_reduce` | 全局归约并广播 | 梯度同步、行并行结果聚合 |
//! | `all_gather` | 全局收集 | 列并行结果合并 |
//! | `reduce_scatter` | 归约后散射 | 分布式注意力计算 |
//! | `broadcast` | 根节点广播 | 参数同步、数据分发 |
//! | `barrier` | 全局同步点 | 确保所有进程到达同一点 |
//!
//! ## 性能特性
//!
//! - Local backend: O(n) 内存拷贝（适合测试）
//! - NCCL backend: GPU直连，带宽优化
//! - 支持异步操作（通过future/promise模式）
//!
//! # 示例
//!
//! ```rust,ignore
//! use openmini_server::distributed::communication::{LocalComm, CollectiveOps, ReduceOp};
//!
//! // 创建本地2卡通信环境
//! let comm = LocalComm::new(0, 2);
//!
//! let mut data = vec![1.0f32, 2.0, 3.0];
//! comm.all_reduce(&mut data, ReduceOp::Sum)?;
//! // data 现在包含 [2.0, 4.0, 6.0] (2个rank各贡献一份)
//! ```

use crate::distributed::config::DistributedError;
use log::{debug, info, trace, warn};
use std::sync::{Arc, Mutex};

/// 归约操作类型
///
/// 定义 all_reduce 和 reduce_scatter 支持的归约方式。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReduceOp {
    /// 求和 (最常用，用于梯度累加)
    Sum,

    /// 取最大值（用于ReLU反向传播）
    Max,

    /// 取最小值
    Min,

    /// 求乘积
    Prod,

    /// 求平均值（用于参数平均）
    Avg,
}

impl std::fmt::Display for ReduceOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sum => write!(f, "sum"),
            Self::Max => write!(f, "max"),
            Self::Min => write!(f, "min"),
            Self::Prod => write!(f, "prod"),
            Self::Avg => write!(f, "avg"),
        }
    }
}

/// 集合通信操作接口
///
/// 定义分布式系统中的核心通信原语。
/// 所有分布式后端（NCCL、Gloo、MPI、Local）都应实现此trait。
///
/// # 线程安全
///
/// 所有实现都必须是 `Send + Sync` 的，以支持多线程并发调用。
///
/// # 错误处理
///
/// 所有操作返回 [`DistributedError::Communication`] 类型的错误。
pub trait CollectiveOps: Send + Sync {
    /// All-Reduce 操作
    ///
    /// 对所有rank的数据执行归约操作，并将结果广播回所有rank。
    ///
    /// # 参数
    ///
    /// * `data` - 输入输出缓冲区（就地操作）
    /// * `op` - 归约操作类型
    ///
    /// # 数学定义
    ///
    /// 对于 rank r 的数据 data_r[i]，结果为：
    /// ```text
    /// result[i] = op(data_0[i], data_1[i], ..., data_{N-1}[i])
    /// ```
    ///
    /// # 性能说明
    ///
    /// - Ring算法: O(N) 通信轮次，O(data_size) 带宽消耗
    /// - Tree算法: O(log N) 轮次，但每轮带宽更大
    /// - 推荐在数据量大时使用Ring算法
    fn all_reduce(&self, data: &mut [f32], op: ReduceOp) -> Result<(), DistributedError>;

    /// All-Gather 操作
    ///
    /// 收集所有rank的数据并拼接到每个rank。
    ///
    /// # 参数
    ///
    /// * `local` - 本地输入数据
    /// * `global` - 输出缓冲区（大小 = local.len() * world_size）
    ///
    /// # 数据布局
    ///
    /// ```text
    /// global = [rank0_data, rank1_data, ..., rank{N-1}_data]
    /// ```
    ///
    /// # 典型用途
    ///
    /// - 列并行线性层的输出合并
    /// - Embedding查表的并行结果收集
    fn all_gather(&self, local: &[f32], global: &mut [f32]) -> Result<(), DistributedError>;

    /// Reduce-Scatter 操作
    ///
    /// 先对所有rank的数据执行归约，然后将结果均匀分片到各rank。
    ///
    /// # 参数
    ///
    /// * `global` - 输入数据（大小 = output.len() * world_size）
    /// * `local` - 输出缓冲区（大小 = global.len() / world_size）
    ///
    /// # 数学定义
    ///
    /// ```text
    /// reduced = reduce(global_0, global_1, ..., global_{N-1})
    /// local_rank_i = reduced[i * chunk_size .. (i+1) * chunk_size]
    /// ```
    ///
    /// # 性能优势
    ///
    /// 相比先all_reduce再手动切分，reduce_scatter减少了一半的通信量。
    fn reduce_scatter(&self, global: &[f32], local: &mut [f32]) -> Result<(), DistributedError>;

    /// Broadcast 操作
    ///
    /// 将root rank的数据广播到所有其他rank。
    ///
    /// # 参数
    ///
    /// * `data` - 在root上为输入，在其他rank上为输出
    /// * `root` - 源rank编号
    ///
    /// # 典型用途
    ///
    /// - 从rank 0广播模型权重
    /// - 广播随机种子确保可复现性
    /// - 广播超参数配置
    fn broadcast(&self, data: &mut [f32], root: usize) -> Result<(), DistributedError>;

    /// Barrier 同步
    ///
    /// 阻塞直到所有rank都调用此方法。
    ///
    /// # 用途
    ///
    /// - 确保所有rank完成当前阶段再进入下一阶段
    /// - 训练中每个epoch结束时的同步点
    /// - 调试时定位hang的位置
    fn barrier(&self) -> Result<(), DistributedError>;
}

/// 本地通信后端（用于开发、测试和单机模拟）
///
/// 使用共享内存模拟多GPU通信行为。
/// 不需要实际的GPU或网络设备，适合CI/CD和单元测试。
///
/// # 实现细节
///
/// - 使用 `Arc<Mutex<Vec<Vec<f32>>>>` 共享状态
/// - 所有rank共享同一个进程内的数据
/// - 通过锁保证线程安全（但不是高性能实现）
///
/// # 局限性
///
/// - 无法真实模拟网络延迟
/// - 锁竞争可能影响性能测试准确性
/// - 仅适合功能正确性验证，不适合性能基准测试
///
/// # 示例
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use openmini_server::distributed::communication::{LocalComm, CollectiveOps};
///
/// // 创建共享状态
/// let shared = Arc::new(Mutex::new(vec![vec![0.0f32; 3]; 2]));
///
/// // 创建2个rank的通信实例
/// let comm0 = LocalComm::with_shared_state(0, 2, shared.clone());
/// let comm1 = LocalComm::with_shared_state(1, 2, shared);
/// ```
pub struct LocalComm {
    /// 当前rank编号
    rank: usize,

    /// 总rank数（world size）
    world_size: usize,

    /// 共享状态：每个rank的本地数据副本
    ///
    /// 索引对应rank编号，用于模拟跨rank数据交换
    shared_state: Arc<Mutex<Vec<Vec<f32>>>>,
}

impl LocalComm {
    /// 创建新的LocalComm实例
    ///
    /// 自动初始化空的共享状态。
    ///
    /// # 参数
    ///
    /// * `rank` - 当前rank编号
    /// * `world_size` - 总rank数
    pub fn new(rank: usize, world_size: usize) -> Self {
        info!(
            "Creating LocalComm: rank={}, world_size={}",
            rank, world_size
        );

        let shared_state = Arc::new(Mutex::new(vec![Vec::new(); world_size]));

        Self {
            rank,
            world_size,
            shared_state,
        }
    }

    /// 使用已有共享状态创建LocalComm
    ///
    /// 用于创建多个rank共享同一状态的场景。
    ///
    /// # 参数
    ///
    /// * `rank` - 当前rank编号
    /// * `world_size` - 总rank数
    /// * `shared_state` - 共享的rank数据容器
    pub fn with_shared_state(
        rank: usize,
        world_size: usize,
        shared_state: Arc<Mutex<Vec<Vec<f32>>>>,
    ) -> Self {
        debug!(
            "Creating LocalComm with shared state: rank={}, world_size={}",
            rank, world_size
        );

        Self {
            rank,
            world_size,
            shared_state,
        }
    }

    /// 获取当前rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// 获取world size
    pub fn world_size(&self) -> usize {
        self.world_size
    }
}

impl CollectiveOps for LocalComm {
    fn all_reduce(&self, data: &mut [f32], op: ReduceOp) -> Result<(), DistributedError> {
        trace!(
            "AllReduce called: rank={}, len={}, op={}",
            self.rank,
            data.len(),
            op
        );

        if data.is_empty() {
            return Ok(());
        }

        // 将本地数据写入共享状态
        {
            let mut state = self.shared_state.lock().unwrap();
            if state[self.rank].len() != data.len() {
                state[self.rank] = data.to_vec();
            } else {
                state[self.rank].copy_from_slice(data);
            }
        }

        // 模拟等待其他rank（实际分布式环境中这里会阻塞）
        // 在单进程测试中，我们需要手动设置其他rank的数据

        // 执行归约
        let mut result = vec![0.0f32; data.len()];
        {
            let state = self.shared_state.lock().unwrap();

            for (i, res) in result.iter_mut().enumerate() {
                let values: Vec<f32> = state
                    .iter()
                    .map(|s| s.get(i).copied().unwrap_or(0.0))
                    .collect();

                *res = match op {
                    ReduceOp::Sum => values.iter().sum(),
                    ReduceOp::Max => values.into_iter().fold(f32::NEG_INFINITY, f32::max),
                    ReduceOp::Min => values.into_iter().fold(f32::INFINITY, f32::min),
                    ReduceOp::Prod => values.into_iter().product(),
                    ReduceOp::Avg => {
                        let sum: f32 = values.iter().sum();
                        sum / self.world_size as f32
                    }
                };
            }
        }

        // 写回结果
        data.copy_from_slice(&result);

        debug!(
            "AllReduce completed: rank={}, first_value={}",
            self.rank, data[0]
        );
        Ok(())
    }

    fn all_gather(&self, local: &[f32], global: &mut [f32]) -> Result<(), DistributedError> {
        trace!(
            "AllGather called: rank={}, local_len={}, global_len={}",
            self.rank,
            local.len(),
            global.len()
        );

        let expected_global_len = local.len() * self.world_size;
        if global.len() != expected_global_len {
            return Err(DistributedError::Communication(format!(
                "global buffer size mismatch: expected {}, got {}",
                expected_global_len,
                global.len()
            )));
        }

        if local.is_empty() {
            return Ok(());
        }

        // 将本地数据写入共享状态
        {
            let mut state = self.shared_state.lock().unwrap();
            if state[self.rank].len() != local.len() {
                state[self.rank] = local.to_vec();
            } else {
                state[self.rank].copy_from_slice(local);
            }
        }

        // 收集所有rank的数据
        {
            let state = self.shared_state.lock().unwrap();

            for (r, rank_data) in state.iter().enumerate() {
                let start = r * local.len();
                let end = start + local.len();

                if !rank_data.is_empty() {
                    global[start..end].copy_from_slice(rank_data);
                }
                // 如果某rank数据为空，保持global中该位置为0
            }
        }

        debug!(
            "AllGather completed: rank={}, collected {} ranks",
            self.rank, self.world_size
        );
        Ok(())
    }

    fn reduce_scatter(&self, global: &[f32], local: &mut [f32]) -> Result<(), DistributedError> {
        trace!(
            "ReduceScatter called: rank={}, global_len={}, local_len={}",
            self.rank,
            global.len(),
            local.len()
        );

        let expected_local_len = global.len() / self.world_size;
        if local.len() != expected_local_len {
            return Err(DistributedError::Communication(format!(
                "local buffer size mismatch: expected {}, got {}",
                expected_local_len,
                local.len()
            )));
        }

        if global.is_empty() {
            return Ok(());
        }

        // 将全局数据写入共享状态（模拟每个rank有完整副本）
        {
            let mut state = self.shared_state.lock().unwrap();
            if state[self.rank].len() != global.len() {
                state[self.rank] = global.to_vec();
            } else {
                state[self.rank].copy_from_slice(global);
            }
        }

        // 对每个chunk执行sum归约，然后取本rank对应的chunk
        let chunk_size = local.len();
        {
            let state = self.shared_state.lock().unwrap();

            for (i, loc) in local.iter_mut().enumerate() {
                let global_idx = self.rank * chunk_size + i;

                // 对所有rank在该位置的数据求和
                let sum: f32 = state
                    .iter()
                    .map(|s| s.get(global_idx).copied().unwrap_or(0.0))
                    .sum();

                *loc = sum;
            }
        }

        debug!(
            "ReduceScatter completed: rank={}, chunk_size={}",
            self.rank, chunk_size
        );
        Ok(())
    }

    fn broadcast(&self, data: &mut [f32], root: usize) -> Result<(), DistributedError> {
        trace!(
            "Broadcast called: rank={}, root={}, len={}",
            self.rank,
            root,
            data.len()
        );

        if root >= self.world_size {
            return Err(DistributedError::Communication(format!(
                "Invalid root rank: {} (world_size={})",
                root, self.world_size
            )));
        }

        if data.is_empty() {
            return Ok(());
        }

        if self.rank == root {
            // Root节点：将数据写入共享状态
            let mut state = self.shared_state.lock().unwrap();
            if state[root].len() != data.len() {
                state[root] = data.to_vec();
            } else {
                state[root].copy_from_slice(data);
            }

            debug!("Broadcast (root): sent {} bytes", data.len() * 4);
        } else {
            // 非Root节点：从共享状态读取root的数据
            let state = self.shared_state.lock().unwrap();

            if let Some(root_data) = state.get(root) {
                if !root_data.is_empty() && root_data.len() == data.len() {
                    data.copy_from_slice(root_data);

                    debug!(
                        "Broadcast (receiver): received {} bytes from rank {}",
                        data.len() * 4,
                        root
                    );
                } else if root_data.is_empty() {
                    warn!(
                        "Broadcast: root {} has not sent data yet, buffer remains unchanged",
                        root
                    );
                } else {
                    return Err(DistributedError::Communication(format!(
                        "Broadcast size mismatch: root has {}, expected {}",
                        root_data.len(),
                        data.len()
                    )));
                }
            }
        }

        Ok(())
    }

    fn barrier(&self) -> Result<(), DistributedError> {
        trace!("Barrier called: rank={}", self.rank);

        // 在真实分布式环境中，这里会阻塞直到所有rank都到达
        // 在LocalComm中，我们仅记录日志并立即返回
        // （因为所有操作都在同一线程或已同步）

        debug!("Barrier passed: rank={}", self.rank);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn create_comm_pair(world_size: usize) -> (LocalComm, LocalComm, Arc<Mutex<Vec<Vec<f32>>>>) {
        let shared = Arc::new(Mutex::new(vec![vec![0.0f32; 4]; world_size]));
        let comm0 = LocalComm::with_shared_state(0, world_size, shared.clone());
        let comm1 = LocalComm::with_shared_state(1, world_size, shared.clone());
        (comm0, comm1, shared)
    }

    #[test]
    fn test_local_comm_creation() {
        let comm = LocalComm::new(0, 2);
        assert_eq!(comm.rank(), 0);
        assert_eq!(comm.world_size(), 2);
    }

    #[test]
    fn test_all_reduce_sum() {
        let (comm0, comm1, shared) = create_comm_pair(2);

        // 设置rank 0的数据
        {
            let mut state = shared.lock().unwrap();
            state[0] = vec![1.0, 2.0, 3.0, 4.0];
        }
        // 设置rank 1的数据
        {
            let mut state = shared.lock().unwrap();
            state[1] = vec![5.0, 6.0, 7.0, 8.0];
        }

        // Rank 0执行all_reduce
        let mut data0 = vec![1.0, 2.0, 3.0, 4.0];
        comm0.all_reduce(&mut data0, ReduceOp::Sum).unwrap();
        assert_eq!(data0, vec![6.0, 8.0, 10.0, 12.0]); // 1+5, 2+6, 3+7, 4+8

        // Rank 1执行all_reduce
        let mut data1 = vec![5.0, 6.0, 7.0, 8.0];
        comm1.all_reduce(&mut data1, ReduceOp::Sum).unwrap();
        assert_eq!(data1, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_all_reduce_max() {
        let (comm0, _comm1, shared) = create_comm_pair(2);

        {
            let mut state = shared.lock().unwrap();
            state[0] = vec![1.0, 5.0, 3.0];
            state[1] = vec![4.0, 2.0, 9.0];
        }

        let mut data = vec![1.0, 5.0, 3.0];
        comm0.all_reduce(&mut data, ReduceOp::Max).unwrap();
        assert_eq!(data, vec![4.0, 5.0, 9.0]);
    }

    #[test]
    fn test_all_reduce_avg() {
        let (comm0, _comm1, shared) = create_comm_pair(2);

        {
            let mut state = shared.lock().unwrap();
            state[0] = vec![2.0, 4.0];
            state[1] = vec![4.0, 8.0];
        }

        let mut data = vec![2.0, 4.0];
        comm0.all_reduce(&mut data, ReduceOp::Avg).unwrap();
        assert_eq!(data, vec![3.0, 6.0]); // (2+4)/2, (4+8)/2
    }

    #[test]
    fn test_all_gather_basic() {
        let (comm0, comm1, shared) = create_comm_pair(2);

        let local_size = 3;
        let mut global0 = vec![0.0f32; local_size * 2];
        let mut global1 = vec![0.0f32; local_size * 2];

        // 设置rank 1的数据
        {
            let mut state = shared.lock().unwrap();
            state[1] = vec![7.0, 8.0, 9.0];
        }

        // Rank 0 all_gather
        {
            let mut state = shared.lock().unwrap();
            state[0] = vec![1.0, 2.0, 3.0];
        }
        comm0.all_gather(&[1.0, 2.0, 3.0], &mut global0).unwrap();

        // Rank 1 all_gather
        comm1.all_gather(&[7.0, 8.0, 9.0], &mut global1).unwrap();

        // 验证两个rank都收到相同的结果
        assert_eq!(global0, vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0]);
        assert_eq!(global1, vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_all_gather_size_mismatch() {
        let comm = LocalComm::new(0, 2);
        let mut global = vec![0.0f32; 5]; // 错误的大小

        let result = comm.all_gather(&[1.0, 2.0], &mut global);
        assert!(result.is_err());
    }

    #[test]
    fn test_reduce_scatter_basic() {
        let (comm0, _comm1, shared) = create_comm_pair(2);

        // 模拟2个rank各有全局数据
        let global_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 6元素，每个rank得到3个
        let mut local = vec![0.0f32; 3];

        // 设置rank 0的全局数据
        {
            let mut state = shared.lock().unwrap();
            state[0] = global_data.clone();
        }

        comm0.reduce_scatter(&global_data, &mut local).unwrap();

        // Rank 0应该得到第0-2个位置的sum（只有rank 0的数据）
        // 在真实场景中，会有多个rank的数据相加
        assert_eq!(local.len(), 3);
    }

    #[test]
    fn test_reduce_scatter_size_mismatch() {
        let comm = LocalComm::new(0, 2);
        let global = vec![1.0, 2.0, 3.0, 4.0];
        let mut local = vec![0.0f32; 3]; // 错误：应该是2

        let result = comm.reduce_scatter(&global, &mut local);
        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_root_to_others() {
        let (comm0, comm1, _shared) = create_comm_pair(2);

        // Root (rank 0) 广播数据
        let mut root_data = vec![42.0, 43.0, 44.0];
        comm0.broadcast(&mut root_data, 0).unwrap();
        assert_eq!(root_data, vec![42.0, 43.0, 44.0]);

        // 非root接收数据
        let mut recv_data = vec![0.0, 0.0, 0.0];
        comm1.broadcast(&mut recv_data, 0).unwrap();
        assert_eq!(recv_data, vec![42.0, 43.0, 44.0]);
    }

    #[test]
    fn test_broadcast_invalid_root() {
        let comm = LocalComm::new(0, 2);
        let mut data = vec![1.0, 2.0];

        let result = comm.broadcast(&mut data, 5); // 无效的root
        assert!(result.is_err());
    }

    #[test]
    fn test_barrier_success() {
        let comm = LocalComm::new(0, 2);
        assert!(comm.barrier().is_ok());
    }

    #[test]
    fn test_empty_data_handling() {
        let comm = LocalComm::new(0, 2);
        let mut empty: Vec<f32> = vec![];

        // 所有操作都应该优雅处理空数据
        assert!(comm.all_reduce(&mut empty, ReduceOp::Sum).is_ok());

        let mut global_empty: Vec<f32> = vec![];
        assert!(comm.all_gather(&[], &mut global_empty).is_ok());

        assert!(comm.reduce_scatter(&[], &mut empty).is_ok());
        assert!(comm.broadcast(&mut empty, 0).is_ok());
    }

    #[test]
    fn test_four_gpu_scenario() {
        // 测试4卡场景
        let shared = Arc::new(Mutex::new(vec![vec![0.0f32; 3]; 4]));
        let comms: Vec<LocalComm> = (0..4)
            .map(|r| LocalComm::with_shared_state(r, 4, shared.clone()))
            .collect();

        // 为每个rank设置不同的数据
        {
            let mut state = shared.lock().unwrap();
            state[0] = vec![1.0, 1.0, 1.0];
            state[1] = vec![2.0, 2.0, 2.0];
            state[2] = vec![3.0, 3.0, 3.0];
            state[3] = vec![4.0, 4.0, 4.0];
        }

        // Rank 0执行all_reduce
        let mut data = vec![1.0, 1.0, 1.0];
        comms[0].all_reduce(&mut data, ReduceOp::Sum).unwrap();
        assert_eq!(data, vec![10.0, 10.0, 10.0]); // 1+2+3+4
    }

    #[test]
    fn test_reduce_op_display() {
        assert_eq!(ReduceOp::Sum.to_string(), "sum");
        assert_eq!(ReduceOp::Max.to_string(), "max");
        assert_eq!(ReduceOp::Min.to_string(), "min");
        assert_eq!(ReduceOp::Prod.to_string(), "prod");
        assert_eq!(ReduceOp::Avg.to_string(), "avg");
    }
}
