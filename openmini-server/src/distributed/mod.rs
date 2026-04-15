//! 分布式推理模块 (Distributed Inference)
//!
//! 提供2-4卡张量并行的完整实现，包括：
//!
//! ## 模块结构
//!
//! ```
//! distributed/
//! ├── config.rs        # 分布式配置与验证
//! ├── communication.rs # 通信抽象层 (CollectiveOps trait)
//! ├── tp.rs            # 张量并行核心 (Column/Row Parallel)
//! └── router.rs        # 负载均衡路由器
//! ```
//!
//! ## 快速开始
//!
//! ```rust,ignore
//! use openmini_server::distributed::{
//!     config::DistributedConfig,
//!     tp::{TensorParallelManager, ParallelType},
//!     router::{DistributedRouter, LoadBalancingPolicy},
//! };
//!
//! // 1. 配置分布式环境
//! let config = DistributedConfig::for_local_testing(2);
//! config.validate()?;
//!
//! // 2. 创建张量并行管理器
//! let tp = TensorParallelManager::new(&config)?;
//!
//! // 3. 创建请求路由器
//! let mut router = DistributedRouter::new(2, LoadBalancingPolicy::LeastLoaded);
//!
//! // 4. 执行并行推理
//! let output = tp.parallel_forward(&input, &weight, ParallelType::ColumnParallel)?;
//! ```
//!
//! ## 性能目标
//!
//! | GPU数量 | 理论加速比 | 实际加速比 | 内存效率 |
//! |---------|-----------|-----------|----------|
//! | 2卡TP   | 2.0x      | ~1.9x     | ~50%节省 |
//! | 4卡TP   | 4.0x      | ~3.6x     | ~75%节省 |
//!
//! ## 架构特点
//!
//! - **Local Backend**: 始终可用，用于开发和测试
//! - **可扩展后端**: 支持NCCL/Gloo/MPI（接口已定义）
//! - **零依赖启动**: Local模式无需GPU或网络设备
//! - **完整测试覆盖**: 单元测试+集成测试

// 模块声明
pub mod communication;
pub mod config;
pub mod discovery;
pub mod exo_communication;
pub mod exo_dynamic_adjuster;
pub mod exo_parallel_strategy;
pub mod router;
pub mod tp;

// 公共导出
pub use communication::{CollectiveOps, LocalComm, ReduceOp};
pub use config::{CommBackend, DistributedConfig, DistributedError};
pub use router::{
    DistributedRouter, InferenceRequest, InferenceResponse, LoadBalancingPolicy, WorkerHealth,
    WorkerId, WorkerStatus,
};
pub use tp::{ParallelType, TensorParallelManager};

pub use exo_communication::{
    CommunicationConfig, CommunicationError, CommunicationLatencyStats, CommunicationTuningParams,
    DeviceCapabilities, DeviceType, ExoBackendType, ExoCommunicationBackend,
    ExoCommunicationManager, LinkType, NetworkDevice, NetworkLink, NetworkTopology,
    PerformanceStats,
};

pub use discovery::{
    DeviceInfo, DeviceResources, DeviceState, DeviceVersion, DiscoveryBackend,
    DiscoveryBackendType, DiscoveryConfig, DiscoveryError, DiscoveryEvent, DiscoveryStats,
    ExoDeviceDiscovery, PeerRegistry, TopologyManager,
};

pub use exo_parallel_strategy::{
    DeviceRole, DeviceTopology, ExoParallelStrategyEngine, ParallelStrategyDecision,
    StrategyConfig, StrategyError,
};

pub use exo_dynamic_adjuster::{
    AdjustmentDecision, BottleneckType, ExoDynamicAdjuster, PerformanceMetrics,
};

#[cfg(test)]
mod integration_tests {
    //! 集成测试：验证多模块协作和端到端场景

    use super::*;
    use ndarray::{Array, Array2, Axis};

    /// 测试2卡TP完整推理流程
    #[test]
    fn test_two_gpu_end_to_end() {
        // 初始化配置
        let config = DistributedConfig::for_local_testing(2);
        assert!(config.validate().is_ok());

        // 创建TP管理器
        let tp = TensorParallelManager::new(&config);
        assert!(tp.is_ok());
        let tp = tp.unwrap();
        assert_eq!(tp.world_size(), 2);

        // 创建输入和权重
        let input: Array2<f32> = Array::from_shape_fn((4, 8), |(i, j)| ((i * 8 + j) as f32) * 0.1);
        let weight: Array2<f32> =
            Array::from_shape_fn((8, 16), |(i, j)| (i * 16 + j) as f32 * 0.01);

        // 执行列并行前向传播
        let result = tp.parallel_forward(&input, &weight, ParallelType::ColumnParallel);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.shape(), [4, 16]); // 输出维度应恢复为完整大小

        // 验证输出有效性
        for val in output.iter() {
            assert!(val.is_finite());
            assert!(!val.is_nan());
        }
    }

    /// 测试4卡TP完整推理流程
    #[test]
    fn test_four_gpu_end_to_end() {
        // 4卡配置
        let config = DistributedConfig::for_local_testing(4);
        assert!(config.validate().is_ok());

        let tp = TensorParallelManager::new(&config).unwrap();
        assert_eq!(tp.world_size(), 4);

        // 更大的输入输出维度
        let input: Array2<f32> =
            Array::from_shape_fn((8, 32), |(i, j)| ((i * 32 + j) as f32) * 0.05);
        let weight_col: Array2<f32> =
            Array::from_shape_fn((32, 64), |(i, j)| (i * 64 + j) as f32 * 0.01);
        let weight_row: Array2<f32> =
            Array::from_shape_fn((64, 16), |(i, j)| (i * 16 + j) as f32 * 0.01);

        // 测试列并行
        let col_result = tp.parallel_forward(&input, &weight_col, ParallelType::ColumnParallel);
        assert!(col_result.is_ok());
        let col_out = col_result.unwrap();
        assert_eq!(col_out.shape(), [8, 64]);

        // 测试行并行
        let row_result = tp.parallel_forward(&input, &weight_row, ParallelType::RowParallel);
        assert!(row_result.is_ok());
        let row_out = row_result.unwrap();
        assert_eq!(row_out.shape(), [8, 16]);
    }

    /// 测试TP + Router协作场景
    #[test]
    fn test_tp_with_router_integration() {
        // 创建2卡TP环境
        let config = DistributedConfig::for_local_testing(2);
        let _tp = TensorParallelManager::new(&config).unwrap();

        // 创建2worker路由器
        let mut router = DistributedRouter::new(2, LoadBalancingPolicy::RoundRobin);

        // 设置不同的负载以测试负载均衡
        router.update_worker_load(WorkerId(0), 30);
        router.update_worker_load(WorkerId(1), 70);

        // 分发多个请求
        let mut dispatched = Vec::new();
        for i in 0..6 {
            let req = InferenceRequest::new(
                format!("integ_req_{}", i),
                vec![1, 2, 3, 4, 5, 6, 7, 8],
                256,
            );
            let wid = router.dispatch(req).unwrap();
            dispatched.push(wid);
        }

        // Round-robin应该均匀分布
        let count_w0 = dispatched.iter().filter(|&&w| w == WorkerId(0)).count();
        let count_w1 = dispatched.iter().filter(|&&w| w == WorkerId(1)).count();
        assert_eq!(count_w0, 3); // 各3个
        assert_eq!(count_w1, 3);

        // 收集所有结果
        for wid in &dispatched {
            let resp = router.collect_result(*wid).unwrap();
            assert!(resp.success);
        }

        // 健康检查
        let statuses = router.health_check();
        assert_eq!(statuses.len(), 2);
        for s in &statuses {
            assert_eq!(s.total_requests, 3); // 每个worker处理了3个请求
        }
    }

    /// 测试不同负载均衡策略的对比
    #[test]
    fn test_load_balancing_policies_comparison() {
        let num_workers = 4;

        // RoundRobin策略
        let mut rr_router = DistributedRouter::new(num_workers, LoadBalancingPolicy::RoundRobin);
        let rr_selections: Vec<WorkerId> = (0..8)
            .map(|_| rr_router.dispatch(create_test_request()).unwrap())
            .collect();

        // 验证轮询分布
        for i in 0..8 {
            assert_eq!(rr_selections[i], WorkerId(i % num_workers));
        }

        // LeastLoaded策略
        let mut ll_router = DistributedRouter::new(num_workers, LoadBalancingPolicy::LeastLoaded);

        // 设置不均匀负载
        ll_router.update_worker_load(WorkerId(0), 90);
        ll_router.update_worker_load(WorkerId(1), 80);
        ll_router.update_worker_load(WorkerId(2), 10); // 最低
        ll_router.update_worker_load(WorkerId(3), 50);

        // 多次选择都应该选择Worker 2
        for _ in 0..5 {
            let selected = ll_router.dispatch(create_test_request()).unwrap();
            assert_eq!(selected, WorkerId(2));
        }
    }

    #[test]
    fn test_weight_split_reconstruct_consistency() {
        let config = DistributedConfig::for_local_testing(4);
        let tp = TensorParallelManager::new(&config).unwrap();

        let original_weight: Array2<f32> =
            Array::from_shape_fn((16, 32), |(i, j)| ((i * 32 + j) as f32));

        let col_shards = tp.column_parallel_weight(&original_weight);
        assert_eq!(col_shards.len(), 4);

        let total_elements: usize = col_shards.iter().map(|s| s.len()).sum();
        assert_eq!(total_elements, original_weight.len());

        let col_reconstructed: Array2<f32> = ndarray::concatenate(
            Axis(1),
            &col_shards.iter().map(|s| s.view()).collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(col_reconstructed, original_weight);

        let row_shards = tp.row_parallel_weight(&original_weight);
        assert_eq!(row_shards.len(), 4);

        let row_total_elements: usize = row_shards.iter().map(|s| s.len()).sum();
        assert_eq!(row_total_elements, original_weight.len());

        let row_reconstructed: Array2<f32> = ndarray::concatenate(
            Axis(0),
            &row_shards.iter().map(|s| s.view()).collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(row_reconstructed, original_weight);
    }

    /// 测试通信操作的正确性验证
    #[test]
    fn test_communication_operations_correctness() {
        use std::sync::{Arc, Mutex};

        let world_size = 3;
        let shared = Arc::new(Mutex::new(vec![vec![0.0f32; 6]; world_size]));

        // 为每个rank设置已知数据
        {
            let mut state = shared.lock().unwrap();
            state[0] = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            state[1] = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
            state[2] = vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0];
        }

        // Rank 0执行all_reduce sum
        let comm0 = LocalComm::with_shared_state(0, world_size, shared.clone());
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        comm0.all_reduce(&mut data, ReduceOp::Sum).unwrap();

        // 验证sum结果：1+2+3=6, 2+4+6=12, ...
        assert_eq!(data, vec![6.0, 12.0, 18.0, 24.0, 30.0, 36.0]);

        // Rank 0执行all_gather
        let local_data = vec![10.0, 20.0];
        let mut global_data = vec![0.0f32; local_data.len() * world_size];

        // 先设置其他rank的数据
        {
            let mut state = shared.lock().unwrap();
            state[0] = local_data.clone();
            state[1] = vec![30.0, 40.0];
            state[2] = vec![50.0, 60.0];
        }

        comm0.all_gather(&local_data, &mut global_data).unwrap();

        // 验证gather结果
        assert_eq!(global_data, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    }

    /// 辅助函数：创建测试请求
    fn create_test_request() -> InferenceRequest {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);

        InferenceRequest::new(format!("test_{}", id), vec![1, 2, 3], 100)
    }

    /// 测试错误处理的完整性
    #[test]
    fn test_error_handling_comprehensive() {
        // 测试无效配置
        let mut bad_config = DistributedConfig::default();
        bad_config.world_size = 0;
        assert!(bad_config.validate().is_err());

        // 测试无效rank
        let mut bad_rank_config = DistributedConfig::for_local_testing(2);
        bad_rank_config.rank = 5; // 超过world_size
        assert!(bad_rank_config.validate().is_err());

        // 测试无效tp_degree
        let mut bad_tp_config = DistributedConfig::for_local_testing(4);
        bad_tp_config.tp_degree = 3; // 不能整除world_size
        assert!(bad_tp_config.validate().is_err());

        // 测试通信错误（broadcast到无效root）
        let comm = LocalComm::new(0, 2);
        let mut data = vec![1.0, 2.0];
        assert!(comm.broadcast(&mut data, 99).is_err());

        // 测试all_gather大小不匹配
        let mut wrong_size_global = vec![0.0f32; 99]; // 错误的大小
        assert!(comm
            .all_gather(&[1.0, 2.0], &mut wrong_size_global)
            .is_err());
    }

    /// 测试边界条件和极端情况
    #[test]
    fn test_edge_cases_and_boundary_conditions() {
        // 最小配置（1卡）
        let single_config = DistributedConfig::for_local_testing(1);
        assert!(single_config.validate().is_ok());
        let single_tp = TensorParallelManager::new(&single_config).unwrap();
        assert_eq!(single_tp.world_size(), 1);

        // 1卡的列并行应该返回完整结果
        let input: Array2<f32> = Array::from_shape_fn((2, 4), |(i, j)| (i * 4 + j) as f32);
        let weight: Array2<f32> = Array::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f32);
        let result = single_tp
            .parallel_forward(&input, &weight, ParallelType::ColumnParallel)
            .unwrap();
        assert_eq!(result.shape(), [2, 4]);

        // 空数据处理
        let empty_input: Array2<f32> = Array::from_shape_vec((0, 4), vec![]).unwrap();
        let empty_weight: Array2<f32> = Array::from_shape_vec((4, 0), vec![]).unwrap();
        // 空输入/权重的行为取决于实现，这里仅确保不panic
        let _empty_result =
            single_tp.parallel_forward(&empty_input, &empty_weight, ParallelType::ColumnParallel);

        // Router最小配置（1 worker）
        let mut single_router = DistributedRouter::new(1, LoadBalancingPolicy::RoundRobin);
        let req = InferenceRequest::new("single".to_string(), vec![1], 10);
        let wid = single_router.dispatch(req).unwrap();
        assert_eq!(wid, WorkerId(0));
        let resp = single_router.collect_result(wid).unwrap();
        assert!(resp.success);
    }
}
