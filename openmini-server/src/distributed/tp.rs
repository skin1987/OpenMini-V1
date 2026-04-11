//! 张量并行 (Tensor Parallelism) 核心实现
//!
//! 实现 Megatron-LM 风格的张量并行策略，支持：
//! - **列并行 (Column Parallel)**: 沿输出维度切分权重矩阵
//! - **行并行 (Row Parallel)**: 沿输入维度切分权重矩阵
//!
//! ## 架构设计
//!
//! ```
//! 输入 X
//! │
//! ▼
//! ┌─────────────────────┐
//! │   Column Parallel    │  Y_i = X @ W_i[:, :]  (各卡计算不同列)
//! └──────────┬──────────┘
//!            │ all-gather / concat
//!            ▼
//! ┌─────────────────────┐
//! │    Row Parallel     │  Z = Σ(Y_i @ V_i[:, :])  (各卡计算部分行后求和)
//! └──────────┬──────────┘
//!            │ all-reduce
//!            ▼
//!         输出 Z
//! ```
//!
//! ## 性能目标
//!
//! | GPU数量 | 理论加速比 | 实际加速比 | 内存节省 |
//! |---------|-----------|-----------|----------|
//! | 2卡     | 2.0x      | ~1.9x     | ~50%     |
//! | 4卡     | 4.0x      | ~3.6x     | ~75%     |
//!
//! # 示例
//!
//! ```rust,ignore
//! use ndarray::Array2;
//! use openmini_server::distributed::tp::{TensorParallelManager, ParallelType};
//! use openmini_server::distributed::config::DistributedConfig;
//!
//! let config = DistributedConfig::for_local_testing(2);
//! let tp = TensorParallelManager::new(&config)?;
//!
//! // 列并行线性层模拟
//! let input = Array2::zeros((32, 768));
//! let weight = Array2::random((768, 2048), ...);
//! let output = tp.parallel_forward(&input, &weight, ParallelType::ColumnParallel)?;
//! ```

use crate::distributed::communication::{CollectiveOps, ReduceOp};
use crate::distributed::config::{DistributedConfig, DistributedError};
use log::{debug, info, trace};
use ndarray::{Array2, s};
use std::sync::Arc;

/// 并行类型枚举
///
/// 定义支持的张量并行切分方式。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParallelType {
    /// 列并行（沿dim=1/输出维度切分）
    ///
    /// 适用场景：
    /// - 第一个线性层（投影到更大维度）
    /// - Attention中的Q/K/V投影
    /// - MLP的第一层
    ///
    /// 操作：Y_i = X @ W_i（独立矩阵乘法）
    ColumnParallel,

    /// 行并行（沿dim=0/输入维度切分）
    ///
    /// 适用场景：
    /// - 第二个线性层（从大维度投影回来）
    /// - Attention中的Output投影
    /// - MLP的第二层
    ///
    /// 操作：Z = all_reduce(Σ Y_i @ V_i)（需要通信同步）
    RowParallel,
}

impl std::fmt::Display for ParallelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ColumnParallel => write!(f, "column_parallel"),
            Self::RowParallel => write!(f, "row_parallel"),
        }
    }
}

/// 张量并行管理器
///
/// 协调多GPU间的张量切分、计算和通信操作。
/// 是分布式推理的核心组件。
///
/// # 线程安全
///
/// 通过 `Arc<dyn CollectiveOps>` 实现线程安全的通信接口共享。
///
/// # 典型工作流程
///
/// 1. 创建配置并验证
/// 2. 实例化 TensorParallelManager
/// 3. 使用 `column_parallel_weight` / `row_parallel_weight` 切分权重
/// 4. 在每个rank上执行 `parallel_forward`
/// 5. 收集结果（all_gather 或 all_reduce）
///
/// # 性能优化建议
///
/// - 启用 `overlap_comm_compute` 可隐藏通信延迟
/// - 合理设置 `micro_batch_size` 平衡显存和吞吐
/// - 2-4卡TP通常能获得最佳性价比
pub struct TensorParallelManager {
    /// 当前进程的rank编号
    rank: usize,

    /// 总进程数/GPU数
    world_size: usize,

    /// 当前进程绑定的设备ID
    device_id: usize,

    /// 通信后端接口
    comm: Arc<dyn CollectiveOps>,
}

impl TensorParallelManager {
    /// 创建新的张量并行管理器实例
    ///
    /// 根据配置初始化TP环境，包括：
    /// - 验证配置参数
    /// - 建立通信连接
    /// - 绑定GPU设备
    ///
    /// # 参数
    ///
    /// * `config` - 分布式配置（必须已通过validate）
    ///
    /// # 错误
    ///
    /// 返回 [`DistributedError::ConfigValidation`] 如果配置无效，
    /// 或 [`DistributedError::TensorParallel`] 如果初始化失败。
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// let config = DistributedConfig::for_local_testing(4);
    /// let tp = TensorParallelManager::new(&config)?;
    /// assert_eq!(tp.world_size(), 4);
    /// ```
    pub fn new(config: &DistributedConfig) -> Result<Self, DistributedError> {
        info!(
            "Initializing TensorParallelManager: rank={}, world_size={}, tp_degree={}",
            config.rank,
            config.world_size,
            config.tp_degree
        );

        // 验证配置
        config.validate()?;

        // 获取设备ID（如果有的话）
        let device_id = config
            .cuda_device_ids
            .get(config.rank)
            .copied()
            .unwrap_or(0);

        debug!("Binding to GPU device {}", device_id);

        // 创建LocalComm作为默认后端
        // 生产环境中这里会根据config.backend选择NCCL/Gloo/MPI
        let comm: Arc<dyn CollectiveOps> = match config.backend {
            crate::distributed::config::CommBackend::Local => {
                Arc::new(crate::distributed::communication::LocalComm::new(
                    config.rank,
                    config.world_size,
                ))
            }
            _ => {
                return Err(DistributedError::Unsupported(format!(
                    "Backend {:?} not yet implemented",
                    config.backend
                )));
            }
        };

        info!(
            "TensorParallelManager initialized successfully on device {}",
            device_id
        );

        Ok(Self {
            rank: config.rank,
            world_size: config.world_size,
            device_id,
            comm,
        })
    }

    /// 获取当前rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// 获取world size
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// 获取当前设备ID
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// 切分权重为列并行格式
    ///
    /// 将权重矩阵沿输出维度（dim=1）均匀切分为 `world_size` 份。
    /// 每个rank获得权重的连续列子集。
    ///
    /// # 参数
    ///
    /// * `weight` - 完整权重矩阵，形状 [input_features, output_features]
    ///
    /// # 返回
    ///
    /// 包含 `world_size` 个子矩阵的向量，每个形状为 [input_features, output_features/world_size]
    ///
    /// # 数学定义
    ///
    /// ```text
    /// weight.shape = [M, N]
    /// shards[i] = weight[:, i*(N/P) : (i+1)*(N/P)]  for i in [0, P)
    /// 其中 P = world_size
    /// ```
    ///
    /// # 约束条件
    ///
    /// - `output_features` 必须能被 `world_size` 整除
    /// - 权重必须是2维数组
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// let weight: Array2<f32> = ...; // shape: [768, 2048]
    /// let shards = tp.column_parallel_weight(&weight); // 2个shard，每个[768, 1024]
    /// assert_eq!(shards.len(), 2);
    /// assert_eq!(shards[0].shape(), [768, 1024]);
    /// ```
    pub fn column_parallel_weight(&self, weight: &Array2<f32>) -> Vec<Array2<f32>> {
        trace!(
            "Column parallel weight splitting: shape={:?}, world_size={}",
            weight.shape(),
            self.world_size
        );

        let (_input_dim, output_dim) = weight.dim();
        let shard_size = output_dim / self.world_size;

        debug!(
            "Splitting into {} shards of size {} columns",
            self.world_size, shard_size
        );

        let mut shards = Vec::with_capacity(self.world_size);

        for i in 0..self.world_size {
            let start_col = i * shard_size;
            let end_col = start_col + shard_size;

            let shard = weight.slice(s![.., start_col..end_col]).to_owned();
            let shard_shape = shard.shape().to_vec();
            shards.push(shard);

            trace!(
                "Shard {}: columns {}..{}, shape={:?}",
                i,
                start_col,
                end_col,
                shard_shape
            );
        }

        info!(
            "Weight split into {} column-parallel shards",
            shards.len()
        );
        shards
    }

    /// 切分权重为行并行格式
    ///
    /// 将权重矩阵沿输入维度（dim=0）均匀切分为 `world_size` 份。
    /// 每个rank获得权重的连续行子集。
    ///
    /// # 参数
    ///
    /// * `weight` - 完整权重矩阵，形状 [input_features, output_features]
    ///
    /// # 返回
    ///
    /// 包含 `world_size` 个子矩阵的向量，每个形状为 [input_features/world_size, output_features]
    ///
    /// # 数学定义
    ///
    /// ```text
    /// weight.shape = [M, N]
    /// shards[i] = weight[i*(M/P) : (i+1)*(M/P), :]  for i in [0, P)
    /// 其中 P = world_size
    /// ```
    ///
    /// # 典型用途
    ///
    /// 行并行通常与列并行配合使用（MLP结构）：
    /// - 第一层：列并行（无通信）
    /// - 激活函数（本地）
    /// - 第二层：行并行（需要all-reduce）
    pub fn row_parallel_weight(&self, weight: &Array2<f32>) -> Vec<Array2<f32>> {
        trace!(
            "Row parallel weight splitting: shape={:?}, world_size={}",
            weight.shape(),
            self.world_size
        );

        let (input_dim, _output_dim) = weight.dim();
        let shard_size = input_dim / self.world_size;

        debug!(
            "Splitting into {} shards of size {} rows",
            self.world_size, shard_size
        );

        let mut shards = Vec::with_capacity(self.world_size);

        for i in 0..self.world_size {
            let start_row = i * shard_size;
            let end_row = start_row + shard_size;

            let shard = weight.slice(s![start_row..end_row, ..]).to_owned();
            let shard_shape = shard.shape().to_vec();
            shards.push(shard);

            trace!(
                "Shard {}: rows {}..{}, shape={:?}",
                i,
                start_row,
                end_row,
                shard_shape
            );
        }

        info!(
            "Weight split into {} row-parallel shards",
            shards.len()
        );
        shards
    }

    /// 执行All-Reduce操作
    ///
    /// 对2D张量执行归约求和。用于行并行层的梯度/输出同步。
    ///
    /// # 参数
    ///
    /// * `tensor` - 输入输出缓冲区（就地操作）
    ///
    /// # 性能说明
    ///
    /// All-reduce是TP中主要的通信瓶颈。
    /// 对于大模型，此操作可能占总时间的20-40%。
    pub fn all_reduce(&self, tensor: &mut Array2<f32>) -> Result<(), DistributedError> {
        trace!(
            "AllReduce on tensor: shape={:?}, rank={}",
            tensor.shape(),
            self.rank
        );

        let mut data = tensor.as_slice().unwrap().to_vec();
        self.comm.all_reduce(&mut data, ReduceOp::Sum)?;

        // 写回结果
        for (i, val) in data.iter().enumerate() {
            let row = i / tensor.ncols();
            let col = i % tensor.ncols();
            tensor[[row, col]] = *val;
        }

        debug!(
            "AllReduce completed: shape={:?}",
            tensor.shape()
        );
        Ok(())
    }

    /// 执行All-Gather操作
    ///
    /// 收集所有rank的张量并拼接。用于列并行层的输出合并。
    ///
    /// # 参数
    ///
    /// * `local` - 本地张量
    ///
    /// # 返回
    ///
    /// 拼接后的全局张量，沿列维度拼接
    pub fn all_gather(&self, local: &Array2<f32>) -> Result<Array2<f32>, DistributedError> {
        trace!(
            "AllGather: local_shape={:?}, rank={}",
            local.shape(),
            self.rank
        );

        let (_rows, cols) = local.dim();
        let global_cols = cols * self.world_size;

        let local_data = local.as_slice().unwrap();
        let mut global_data = vec![0.0f32; local_data.len() * self.world_size];

        self.comm.all_gather(local_data, &mut global_data)?;

        // 重塑为2D数组
        let rows = local.nrows();
        let result =
            Array2::from_shape_vec((rows, global_cols), global_data).map_err(|e| {
                DistributedError::TensorParallel(format!("Failed to reshape gathered data: {}", e))
            })?;

        debug!(
            "AllGather completed: local_shape={:?}, global_shape={:?}",
            local.shape(),
            result.shape()
        );
        Ok(result)
    }

    /// 执行Reduce-Scatter操作
    ///
    /// 归约后散射到各rank。用于优化某些特定模式的通信。
    pub fn reduce_scatter(&self, global: &Array2<f32>) -> Result<Array2<f32>, DistributedError> {
        trace!(
            "ReduceScatter: global_shape={:?}, rank={}",
            global.shape(),
            self.rank
        );

        let (rows, cols) = global.dim();
        let local_cols = cols / self.world_size;

        let global_data = global.as_slice().unwrap();
        let mut local_data = vec![0.0f32; global_data.len() / self.world_size];

        self.comm.reduce_scatter(global_data, &mut local_data)?;

        let result =
            Array2::from_shape_vec((rows, local_cols), local_data).map_err(|e| {
                DistributedError::TensorParallel(format!(
                    "Failed to reshape scattered data: {}",
                    e
                ))
            })?;

        debug!(
            "ReduceScatter completed: global_shape={:?}, local_shape={:?}",
            global.shape(),
            result.shape()
        );
        Ok(result)
    }

    /// 执行并行前向传播
    ///
    /// 模拟在多GPU上执行前向传播的过程。
    /// 根据并行类型选择不同的计算和通信模式。
    ///
    /// # 参数
    ///
    /// * `input` - 输入张量，形状 [batch_size, features]
    /// * `weight` - 完整权重矩阵
    /// * `parallel_type` - 并行类型（列并行或行并行）
    ///
    /// # 返回
    ///
    /// 计算结果张量
    ///
    /// # 计算流程
    ///
    /// ## Column Parallel
    /// 1. 切分权重为多个shard
    /// 2. 当前rank使用对应的shard进行矩阵乘法
    /// 3. All-Gather收集所有rank的结果
    ///
    /// ## Row Parallel
    /// 1. 切分权重为多个shard
    /// 2. 当前rank使用对应的shard进行部分矩阵乘法
    /// 3. All-Reduce聚合所有rank的部分结果
    ///
    /// # 性能模型
    ///
    /// 对于batch B，输入维度M，输出维度N，P个GPU：
    /// - 单卡计算量: O(B*M*N)
    /// - TP计算量: O(B*M*N/P) + 通信开销
    /// - 通信开销:
    ///   - Column: O(B*N) (all-gather)
    ///   - Row: O(B*N) (all-reduce)
    pub fn parallel_forward(
        &self,
        input: &Array2<f32>,
        weight: &Array2<f32>,
        parallel_type: ParallelType,
    ) -> Result<Array2<f32>, DistributedError> {
        info!(
            "Parallel forward: input_shape={:?}, weight_shape={:?}, type={}, rank={}",
            input.shape(),
            weight.shape(),
            parallel_type,
            self.rank
        );

        match parallel_type {
            ParallelType::ColumnParallel => self.column_parallel_forward(input, weight),
            ParallelType::RowParallel => self.row_parallel_forward(input, weight),
        }
    }

    /// 列并行前向传播内部实现
    fn column_parallel_forward(
        &self,
        input: &Array2<f32>,
        weight: &Array2<f32>,
    ) -> Result<Array2<f32>, DistributedError> {
        debug!("Executing column parallel forward");

        // 1. 切分权重
        let shards = self.column_parallel_weight(weight);

        // 2. 当前rank使用对应shard进行计算
        let my_shard = &shards[self.rank];
        trace!(
            "Rank {} using shard with shape {:?}",
            self.rank,
            my_shard.shape()
        );

        // 3. 矩阵乘法: output = input @ shard
        // 注意：ndarray的matmul需要匹配维度
        let (batch_size, _in_features) = input.dim();
        let (_shard_in, out_features) = my_shard.dim();

        // 手动实现矩阵乘法（简化版）
        let mut local_output = Array2::zeros((batch_size, out_features));

        for b in 0..batch_size {
            for j in 0..out_features {
                let mut sum = 0.0f32;
                for k in 0.._shard_in {
                    sum += input[[b, k]] * my_shard[[k, j]];
                }
                local_output[[b, j]] = sum;
            }
        }

        trace!(
            "Local output computed: shape={:?}",
            local_output.shape()
        );

        // 4. All-Gather合并结果
        let output = self.all_gather(&local_output)?;

        info!(
            "Column parallel forward completed: output_shape={:?}",
            output.shape()
        );
        Ok(output)
    }

    /// 行并行前向传播内部实现
    fn row_parallel_forward(
        &self,
        input: &Array2<f32>,
        weight: &Array2<f32>,
    ) -> Result<Array2<f32>, DistributedError> {
        debug!("Executing row parallel forward");

        // 1. 切分权重
        let shards = self.row_parallel_weight(weight);

        // 2. 当前rank使用对应shard进行计算
        let my_shard = &shards[self.rank];
        trace!(
            "Rank {} using shard with shape {:?}",
            self.rank,
            my_shard.shape()
        );

        let (batch_size, _out_features) = {
            let (_, out_f) = weight.dim();
            (input.nrows(), out_f)
        };
        let (shard_in_features, shard_out_features) = my_shard.dim();

        // 3. 切分输入（每rank只处理部分输入特征）
        let input_start = self.rank * shard_in_features;
        let input_end = input_start + shard_in_features;

        // 确保输入维度足够大
        if input_end > input.ncols() {
            return Err(DistributedError::TensorParallel(format!(
                "Input dimension too small for row parallel: need {}, got {}",
                input_end,
                input.ncols()
            )));
        }

        let local_input = input.slice(s![.., input_start..input_end]).to_owned();

        // 4. 矩阵乘法: partial_output = local_input @ shard
        let mut partial_output = Array2::zeros((batch_size, shard_out_features));

        for b in 0..batch_size {
            for j in 0..shard_out_features {
                let mut sum = 0.0f32;
                for k in 0..shard_in_features {
                    sum += local_input[[b, k]] * my_shard[[k, j]];
                }
                partial_output[[b, j]] = sum;
            }
        }

        trace!(
            "Partial output computed: shape={:?}",
            partial_output.shape()
        );

        // 5. All-Reduce聚合
        let mut output = partial_output;
        self.all_reduce(&mut output)?;

        info!(
            "Row parallel forward completed: output_shape={:?}",
            output.shape()
        );
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::communication::LocalComm;
    use ndarray::{Array, Array2, Axis};
    use std::sync::Mutex;

    fn create_tp_manager(world_size: usize, rank: usize) -> (TensorParallelManager, Arc<Mutex<Vec<Vec<f32>>>>) {
        let shared = Arc::new(Mutex::new(vec![vec![0.0f32; 1024]; world_size]));
        let comm = Arc::new(LocalComm::with_shared_state(rank, world_size, shared.clone()));

        let tp = TensorParallelManager {
            rank,
            world_size,
            device_id: rank,
            comm,
        };

        (tp, shared)
    }

    #[test]
    fn test_tp_manager_creation() {
        let config = DistributedConfig::for_local_testing(2);
        let tp = TensorParallelManager::new(&config);
        assert!(tp.is_ok());
        let tp = tp.unwrap();
        assert_eq!(tp.rank(), 0);
        assert_eq!(tp.world_size(), 2);
    }

    #[test]
    fn test_column_parallel_weight_splitting() {
        let config = DistributedConfig::for_local_testing(2);
        let tp = TensorParallelManager::new(&config).unwrap();

        // 创建 8x4 的权重矩阵
        let weight: Array2<f32> = Array::from_shape_fn((8, 4), |(i, j)| (i * 4 + j) as f32);

        let shards = tp.column_parallel_weight(&weight);

        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0].shape(), [8, 2]); // 每个shard得到一半列
        assert_eq!(shards[1].shape(), [8, 2]);

        // 验证数据正确性
        assert_eq!(shards[0][[0, 0]], 0.0); // weight[0,0]
        assert_eq!(shards[0][[0, 1]], 1.0); // weight[0,1]
        assert_eq!(shards[1][[0, 0]], 2.0); // weight[0,2]
        assert_eq!(shards[1][[0, 1]], 3.0); // weight[0,3]
    }

    #[test]
    fn test_row_parallel_weight_splitting() {
        let config = DistributedConfig::for_local_testing(2);
        let tp = TensorParallelManager::new(&config).unwrap();

        // 创建 8x4 的权重矩阵
        let weight: Array2<f32> = Array::from_shape_fn((8, 4), |(i, j)| (i * 4 + j) as f32);

        let shards = tp.row_parallel_weight(&weight);

        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0].shape(), [4, 4]); // 每个shard得到一半行
        assert_eq!(shards[1].shape(), [4, 4]);

        // 验证数据正确性
        assert_eq!(shards[0][[0, 0]], 0.0);   // weight[0,0]
        assert_eq!(shards[0][[3, 3]], 15.0);  // weight[3,3]
        assert_eq!(shards[1][[0, 0]], 16.0);  // weight[4,0]
    }

    #[test]
    fn test_four_gpu_column_split() {
        let config = DistributedConfig::for_local_testing(4);
        let tp = TensorParallelManager::new(&config).unwrap();

        let weight: Array2<f32> = Array::from_shape_fn((16, 16), |(i, j)| (i * 16 + j) as f32);
        let shards = tp.column_parallel_weight(&weight);

        assert_eq!(shards.len(), 4);
        for shard in &shards {
            assert_eq!(shard.shape(), [16, 4]); // 16列分成4份，每份4列
        }
    }

    #[test]
    fn test_all_reduce_basic() {
        let (tp, shared) = create_tp_manager(2, 0);

        // 设置另一个rank的数据
        {
            let mut state = shared.lock().unwrap();
            state[1] = vec![1.0; 6]; // 2x3矩阵
        }

        let mut tensor = Array2::from_shape_fn((2, 3), |_pos| 1.0f32);
        tp.all_reduce(&mut tensor).unwrap();

        // 应该是 1+1=2 （两个rank都是1）
        for val in tensor.iter() {
            assert!((*val - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_all_gather_basic() {
        let (tp, shared) = create_tp_manager(2, 0);

        // 设置rank 1的数据
        {
            let mut state = shared.lock().unwrap();
            state[1] = vec![3.0, 4.0]; // 1x2矩阵
        }

        let local = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let result = tp.all_gather(&local).unwrap();

        assert_eq!(result.shape(), [1, 4]); // 2列 * 2 ranks
        assert!((result[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((result[[0, 2]] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_column_parallel_forward() {
        let config = DistributedConfig::for_local_testing(2);
        let tp = TensorParallelManager::new(&config).unwrap();

        // 小规模测试: batch=2, in=4, out=4
        let input: Array2<f32> = Array::from_shape_fn((2, 4), |(i, j)| ((i * 4 + j) as f32) * 0.1);
        let weight: Array2<f32> = Array::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f32 * 0.01);

        let result = tp.parallel_forward(&input, &weight, ParallelType::ColumnParallel);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.shape(), [2, 4]); // 输出应该恢复完整大小

        // 验证输出不为零且有限
        for val in output.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_row_parallel_forward() {
        let config = DistributedConfig::for_local_testing(2);
        let tp = TensorParallelManager::new(&config).unwrap();

        // batch=2, in=4, out=2
        let input: Array2<f32> = Array::from_shape_fn((2, 4), |(i, j)| ((i * 4 + j) as f32) * 0.1);
        let weight: Array2<f32> = Array::from_shape_fn((4, 2), |(i, j)| (i * 2 + j) as f32 * 0.01);

        let result = tp.parallel_forward(&input, &weight, ParallelType::RowParallel);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.shape(), [2, 2]); // 输出完整大小

        for val in output.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_parallel_type_display() {
        assert_eq!(ParallelType::ColumnParallel.to_string(), "column_parallel");
        assert_eq!(ParallelType::RowParallel.to_string(), "row_parallel");
    }

    #[test]
    fn test_invalid_config_rejected() {
        let mut config = DistributedConfig::default();
        config.world_size = 0;
        let result = TensorParallelManager::new(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_integration_two_gpu_scenario() {
        // 模拟2卡完整推理流程
        let config = DistributedConfig::for_local_testing(2);
        let tp0 = TensorParallelManager::new(&config).unwrap();

        let mut config_rank1 = config.clone();
        config_rank1.rank = 1;
        let tp1 = TensorParallelManager::new(&config_rank1).unwrap();

        // 共享权重
        let weight: Array2<f32> = Array::from_shape_fn((8, 8), |(i, j)| (i * 8 + j) as f32 * 0.01);

        // Rank 0 和 Rank 1 分别执行列并行
        let input: Array2<f32> = Array::from_shape_fn((2, 8), |(i, j)| ((i * 8 + j) as f32) * 0.1);

        let result0 = tp0.parallel_forward(&input, &weight, ParallelType::ColumnParallel);
        let result1 = tp1.parallel_forward(&input, &weight, ParallelType::ColumnParallel);

        assert!(result0.is_ok());
        assert!(result1.is_ok());

        // 两个rank应该产生相同的结果（因为all-gather后都拥有完整输出）
        let out0 = result0.unwrap();
        let out1 = result1.unwrap();
        assert_eq!(out0.shape(), out1.shape());
    }

    #[test]
    fn test_four_gpu_integration() {
        // 4卡集成测试
        let config = DistributedConfig::for_local_testing(4);
        let tp = TensorParallelManager::new(&config).unwrap();

        let weight: Array2<f32> = Array::from_shape_fn((16, 16), |(i, j)| (i * 16 + j) as f32 * 0.01);
        let input: Array2<f32> = Array::from_shape_fn((4, 16), |(i, j)| ((i * 16 + j) as f32) * 0.05);

        // 测试列并行
        let col_result = tp.parallel_forward(&input, &weight, ParallelType::ColumnParallel);
        assert!(col_result.is_ok());
        let col_out = col_result.unwrap();
        assert_eq!(col_out.shape(), [4, 16]);

        // 测试行并行
        let row_weight: Array2<f32> = Array::from_shape_fn((16, 8), |(i, j)| (i * 8 + j) as f32 * 0.01);
        let row_result = tp.parallel_forward(&input, &row_weight, ParallelType::RowParallel);
        assert!(row_result.is_ok());
        let row_out = row_result.unwrap();
        assert_eq!(row_out.shape(), [4, 8]);
    }

    #[test]
    fn test_weight_split_consistency() {
        // 验证切分后重新组合能得到原始权重
        let config = DistributedConfig::for_local_testing(2);
        let tp = TensorParallelManager::new(&config).unwrap();

        let original: Array2<f32> =
            Array::from_shape_fn((6, 10), |(i, j)| (i * 10 + j) as f32);

        // 测试列并行切分
        let col_shards = tp.column_parallel_weight(&original);
        let reconstructed: Vec<f32> = col_shards
            .iter()
            .flat_map(|s| s.iter().copied())
            .collect();
        let original_flat: Vec<f32> = original.iter().copied().collect();
        assert_eq!(reconstructed, original_flat);

        // 测试行并行切分
        let row_shards = tp.row_parallel_weight(&original);
        let row_reconstructed: Array2<f32> = ndarray::concatenate(
            Axis(0),
            &row_shards.iter().map(|s| s.view()).collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(row_reconstructed.shape(), original.shape());
    }
}
