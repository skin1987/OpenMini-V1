#![cfg(feature = "distributed")]
//! EXO架构集成测试
//!
//! 验证EXO架构各组件协同工作：
//! 1. 设备发现和拓扑构建
//! 2. 并行策略选择引擎
//! 3. 动态策略调整器
//! 4. 通信后端集成
//!
//! # 测试场景
//!
//! 1. **设备拓扑构建测试**: 验证从设备列表构建拓扑的能力
//! 2. **并行策略选择测试**: 验证基于拓扑选择最优策略的能力
//! 3. **动态调整测试**: 验证性能监控和策略调整的能力
//! 4. **端到端集成测试**: 验证整个EXO架构的协同工作

use std::collections::HashMap;
use std::time::SystemTime;

use openmini_server::distributed::{
    AdjustmentDecision, DeviceCapabilities, DeviceInfo, DeviceResources, DeviceState,
    DeviceTopology, DeviceType, DeviceVersion, ExoDynamicAdjuster, ExoParallelStrategyEngine,
    LinkType, NetworkDevice, NetworkLink, NetworkTopology, PerformanceMetrics,
};
use openmini_server::model::inference::distributed_inference_config::ParallelStrategy;

// ==================== 测试辅助函数 ====================

/// 创建设备信息
fn create_device_info(device_id: &str, device_type: DeviceType, compute_score: u8) -> DeviceInfo {
    DeviceInfo {
        device_id: device_id.to_string(),
        device_type,
        capabilities: DeviceCapabilities {
            supports_rdma: device_type == DeviceType::AppleGpu,
            supports_mlx: device_type == DeviceType::AppleGpu,
            supports_thunderbolt: true,
            max_memory_bandwidth_gbs: if device_type == DeviceType::AppleGpu {
                80.0
            } else {
                25.0
            },
            compute_score,
        },
        resources: DeviceResources {
            total_memory_bytes: if device_type == DeviceType::AppleGpu {
                16 * 1024 * 1024 * 1024 // 16GB
            } else {
                32 * 1024 * 1024 * 1024 // 32GB
            },
            available_memory_bytes: if device_type == DeviceType::AppleGpu {
                12 * 1024 * 1024 * 1024 // 12GB
            } else {
                24 * 1024 * 1024 * 1024 // 24GB
            },
            cpu_cores: if device_type == DeviceType::AppleGpu {
                8
            } else {
                16
            },
            cpu_utilization: 0.1,
            gpu_info: None,
            storage_info: openmini_server::distributed::discovery::StorageInfo {
                total_storage_bytes: 1024 * 1024 * 1024 * 1024, // 1TB
                available_storage_bytes: 512 * 1024 * 1024 * 1024, // 512GB
                storage_type: "NVMe SSD".to_string(),
                performance_score: 90,
            },
            network_bandwidth_gbps: 10.0,
        },
        network_addresses: vec![format!(
            "192.168.1.{}:8080",
            device_id.split('-').last().unwrap_or("1")
        )],
        metadata: HashMap::from([
            ("os".to_string(), "macOS 15.0".to_string()),
            ("architecture".to_string(), "arm64".to_string()),
        ]),
        first_seen: SystemTime::now(),
        last_seen: SystemTime::now(),
        version: DeviceVersion {
            software_version: "0.1.0".to_string(),
            protocol_version: "1.0".to_string(),
            hardware_version: "M3".to_string(),
            firmware_version: "1.0".to_string(),
        },
    }
}

/// 创建异构设备拓扑
fn create_heterogeneous_topology() -> DeviceTopology {
    // 创建包含不同类型和能力的设备
    let devices = vec![
        create_device_info("device-1-master", DeviceType::Cpu, 80),
        create_device_info("device-2-gpu", DeviceType::AppleGpu, 95),
        create_device_info("device-3-cpu", DeviceType::Cpu, 70),
        create_device_info("device-4-gpu", DeviceType::AppleGpu, 90),
        create_device_info("device-5-cpu", DeviceType::Cpu, 75),
    ];

    DeviceTopology::from_devices(devices)
}

/// 创建同构设备拓扑
fn create_homogeneous_topology() -> DeviceTopology {
    // 创建具有相似能力的设备
    let devices = vec![
        create_device_info("gpu-1", DeviceType::AppleGpu, 90),
        create_device_info("gpu-2", DeviceType::AppleGpu, 92),
        create_device_info("gpu-3", DeviceType::AppleGpu, 88),
        create_device_info("gpu-4", DeviceType::AppleGpu, 91),
    ];

    DeviceTopology::from_devices(devices)
}

// ==================== 集成测试 ====================

#[test]
fn test_device_topology_creation() {
    // 测试设备拓扑创建
    let topology = create_heterogeneous_topology();

    // 验证设备数量
    assert_eq!(topology.device_count(), 5, "设备数量应为5");

    // 验证设备ID
    let device_ids = topology.device_ids();
    assert!(device_ids.contains(&"device-1-master".to_string()));
    assert!(device_ids.contains(&"device-2-gpu".to_string()));

    // 验证设备能力
    let compute_score_1 = topology.get_compute_score("device-1-master");
    let compute_score_2 = topology.get_compute_score("device-2-gpu");

    assert!(
        compute_score_2 > compute_score_1,
        "GPU设备应有更高的计算评分"
    );

    // 验证RDMA支持
    let supports_rdma_1 = topology.supports_rdma("device-1-master");
    let supports_rdma_2 = topology.supports_rdma("device-2-gpu");

    assert!(!supports_rdma_1, "CPU设备不应支持RDMA");
    assert!(supports_rdma_2, "GPU设备应支持RDMA");

    // 验证内存大小
    let memory_gb_1 = topology.get_total_memory_gb("device-1-master");
    let memory_gb_2 = topology.get_total_memory_gb("device-2-gpu");

    assert!(memory_gb_1 > 30.0, "CPU设备应有超过30GB内存");
    assert!(memory_gb_2 > 15.0, "GPU设备应有超过15GB内存");

    println!("设备拓扑创建测试通过: {}个设备", topology.device_count());
}

#[test]
fn test_parallel_strategy_selection_heterogeneous() {
    // 测试异构设备拓扑的策略选择
    let topology = create_heterogeneous_topology();
    let mut strategy_engine = ExoParallelStrategyEngine::new();

    // 测试不同模型大小和批次大小的策略选择
    let test_cases = vec![
        (10.0, 1, 100.0), // 大模型，小批次，宽松延迟预算
        (3.0, 4, 50.0),   // 中等模型，中等批次，中等延迟预算
        (1.0, 8, 20.0),   // 小模型，大批次，严格延迟预算
    ];

    for (model_size_gb, batch_size, latency_budget_ms) in test_cases {
        let result = strategy_engine.select_optimal_strategy(
            &topology,
            model_size_gb,
            batch_size,
            Some(latency_budget_ms),
        );

        assert!(
            result.is_ok(),
            "策略选择应成功: 模型={}GB, 批次={}, 延迟预算={}ms",
            model_size_gb,
            batch_size,
            latency_budget_ms
        );

        let decision = result.unwrap();

        // 验证决策的基本属性
        assert!(decision.predicted_latency_ms > 0.0, "预测延迟应为正数");
        assert!(decision.predicted_memory_gb > 0.0, "预测内存使用应为正数");
        assert!(
            decision.confidence >= 0.0 && decision.confidence <= 1.0,
            "置信度应在0-1之间"
        );

        // 验证设备分配
        assert!(!decision.device_assignment.is_empty(), "设备分配不应为空");

        // 验证策略配置
        match decision.strategy {
            ParallelStrategy::TensorParallel => {
                assert!(decision.config.tp_degree >= 2, "张量并行度至少为2");
            }
            ParallelStrategy::PipelineParallel => {
                assert!(decision.config.pp_degree >= 2, "流水线并行度至少为2");
                assert!(decision.config.micro_batch_size >= 1, "微批次大小至少为1");
            }
            ParallelStrategy::HybridParallel => {
                assert!(
                    decision.config.tp_degree >= 2,
                    "混合并行的张量并行度至少为2"
                );
                assert!(
                    decision.config.pp_degree >= 2,
                    "混合并行的流水线并行度至少为2"
                );
            }
            ParallelStrategy::SequenceParallel => {
                assert!(decision.config.sp_degree >= 2, "序列并行度至少为2");
            }
            _ => {} // 其他策略
        }

        println!(
            "异构拓扑策略选择测试通过: 策略={:?}, 延迟={:.2}ms, 置信度={:.2}",
            decision.strategy, decision.predicted_latency_ms, decision.confidence
        );
    }
}

#[test]
fn test_parallel_strategy_selection_homogeneous() {
    // 测试同构设备拓扑的策略选择
    let topology = create_homogeneous_topology();
    let mut strategy_engine = ExoParallelStrategyEngine::new();

    // 同构设备通常更适合张量并行
    let result = strategy_engine.select_optimal_strategy(
        &topology,
        8.0,        // 8GB模型
        2,          // 批次大小2
        Some(80.0), // 80ms延迟预算
    );

    assert!(result.is_ok(), "同构拓扑策略选择应成功");

    let decision = result.unwrap();

    // 同构GPU设备通常推荐张量并行
    // 注意：这只是期望，实际取决于算法
    println!(
        "同构拓扑策略选择结果: 策略={:?}, 并行度={}, 延迟={:.2}ms",
        decision.strategy,
        decision.config.tp_degree.max(decision.config.pp_degree),
        decision.predicted_latency_ms
    );

    // 验证至少有一个设备被分配了角色
    assert!(!decision.device_assignment.is_empty(), "设备分配不应为空");

    // 验证预测性能指标合理
    assert!(decision.predicted_latency_ms > 0.0, "预测延迟应为正数");
    assert!(decision.predicted_memory_gb > 0.0, "预测内存使用应为正数");
    assert!(
        decision.predicted_throughput_tps >= 0.0,
        "预测吞吐量不应为负数"
    );
}

#[test]
fn test_dynamic_adjuster_basic_functionality() {
    // 测试动态调整器的基本功能
    let mut adjuster = ExoDynamicAdjuster::new();
    let topology = create_heterogeneous_topology();

    // 首先选择一个初始策略
    let mut strategy_engine = ExoParallelStrategyEngine::new();
    let initial_decision = strategy_engine
        .select_optimal_strategy(
            &topology,
            5.0,         // 5GB模型
            4,           // 批次大小4
            Some(100.0), // 100ms延迟预算
        )
        .expect("初始策略选择应成功");

    // 设置当前策略
    adjuster.set_current_strategy(initial_decision.clone());

    // 创建性能指标
    let metrics = PerformanceMetrics::new(
        120.5, // 延迟120.5ms (略高于预测)
        85.3,  // 吞吐量85.3 tokens/秒
        8.7,   // 内存使用8.7GB
        4,     // 批次大小4
        5.0,   // 模型大小5GB
    );

    // 评估是否需要调整
    let adjustment_result = adjuster.evaluate_and_adjust(&metrics, &topology);

    assert!(adjustment_result.is_ok(), "调整评估应成功");

    let adjustment_decision = adjustment_result.unwrap();

    // 调整决策可能为None（如果不需要调整）或Some（如果需要调整）
    match adjustment_decision {
        Some(decision) => {
            // 验证调整决策的基本属性
            match decision {
                AdjustmentDecision::NoChange { reason, confidence } => {
                    assert!(!reason.is_empty(), "原因不应为空");
                    assert!(confidence >= 0.0 && confidence <= 1.0, "置信度应在0-1之间");
                    println!("动态调整器决定不调整: {}, 置信度={:.2}", reason, confidence);
                }
                AdjustmentDecision::StrategyChange {
                    new_strategy,
                    reason,
                    expected_improvement,
                    migration_cost,
                } => {
                    assert!(!reason.is_empty(), "原因不应为空");
                    assert!(expected_improvement >= 0.0, "预期改进应为非负数");
                    assert!(
                        migration_cost >= 0.0 && migration_cost <= 1.0,
                        "迁移成本应在0-1之间"
                    );
                    println!(
                        "动态调整器决定切换策略: {}, 预期改进={:.1}%, 迁移成本={:.2}",
                        reason, expected_improvement, migration_cost
                    );
                }
                AdjustmentDecision::ParameterTuning {
                    new_config,
                    reason,
                    expected_improvement,
                } => {
                    assert!(!reason.is_empty(), "原因不应为空");
                    assert!(expected_improvement >= 0.0, "预期改进应为非负数");
                    println!(
                        "动态调整器决定调整参数: {}, 预期改进={:.1}%",
                        reason, expected_improvement
                    );
                }
                AdjustmentDecision::Rollback {
                    previous_strategy,
                    reason,
                } => {
                    assert!(!reason.is_empty(), "原因不应为空");
                    println!("动态调整器决定回滚: {}", reason);
                }
            }
        }
        None => {
            println!("动态调整器评估后认为无需调整");
        }
    }

    // 验证性能历史记录
    let performance_history = adjuster.get_performance_history();
    assert!(!performance_history.is_empty(), "性能历史记录不应为空");

    // 验证调整历史记录（可能为空）
    let adjustment_history = adjuster.get_adjustment_history();
    println!("调整历史记录数量: {}", adjustment_history.len());

    // 验证当前策略
    let current_strategy = adjuster.get_current_strategy();
    assert!(current_strategy.is_some(), "当前策略不应为None");
}

#[test]
fn test_dynamic_adjuster_sla_violation() {
    // 测试SLA违规时的调整决策
    let mut adjuster = ExoDynamicAdjuster::new();
    let topology = create_heterogeneous_topology();

    // 设置严格的SLA配置
    let mut config = adjuster.get_config().clone();
    config.target_latency_ms = 50.0; // 严格的目标延迟
    config.min_throughput_tps = 100.0; // 高的最小吞吐量
    config.performance_degradation_threshold = 10.0; // 敏感的性能下降阈值
    adjuster.update_config(config);

    // 选择一个初始策略
    let mut strategy_engine = ExoParallelStrategyEngine::new();
    let initial_decision = strategy_engine
        .select_optimal_strategy(
            &topology,
            7.0,        // 7GB模型
            2,          // 批次大小2
            Some(50.0), // 50ms延迟预算
        )
        .expect("初始策略选择应成功");

    adjuster.set_current_strategy(initial_decision);

    // 创建严重违反SLA的性能指标
    let metrics = PerformanceMetrics::new(
        180.0, // 严重超时的延迟180ms
        30.0,  // 很低的吞吐量30 tokens/秒
        10.5,  // 内存使用10.5GB
        2,     // 批次大小2
        7.0,   // 模型大小7GB
    );

    // 评估是否需要调整
    let adjustment_result = adjuster.evaluate_and_adjust(&metrics, &topology);
    assert!(adjustment_result.is_ok(), "SLA违规评估应成功");

    let adjustment_decision = adjustment_result.unwrap();

    // 严重SLA违规时更可能产生调整决策
    if let Some(decision) = adjustment_decision {
        println!("SLA违规触发调整决策: {:?}", decision);

        // 验证决策合理性
        match decision {
            AdjustmentDecision::NoChange { reason, .. } => {
                println!("SLA违规但决定不调整: {}", reason);
                // 这种情况可能发生，但通常SLA违规应该触发调整
            }
            AdjustmentDecision::StrategyChange {
                reason,
                expected_improvement,
                ..
            } => {
                assert!(!reason.is_empty(), "原因不应为空");
                assert!(expected_improvement > 0.0, "预期改进应为正数");
                println!(
                    "SLA违规触发策略切换: {}, 预期改进={:.1}%",
                    reason, expected_improvement
                );
            }
            _ => {} // 其他决策类型
        }
    }

    println!("SLA违规测试完成");
}

#[test]
fn test_performance_metrics_calculation() {
    // 测试性能指标计算
    let metrics = PerformanceMetrics::new(
        85.3,  // 延迟85.3ms
        120.5, // 吞吐量120.5 tokens/秒
        6.8,   // 内存使用6.8GB
        4,     // 批次大小4
        3.5,   // 模型大小3.5GB
    );

    // 计算性能评分
    let score = metrics.performance_score();
    assert!(score > 0.0, "性能评分应为正数");

    // 测试SLA检查
    let meets_sla = metrics.meets_sla(100.0, 50.0); // 目标延迟100ms, 最小吞吐量50tps
    assert!(meets_sla, "指标应满足较宽松的SLA");

    let fails_sla = metrics.meets_sla(50.0, 150.0); // 严格的目标延迟50ms, 高吞吐量150tps
    assert!(!fails_sla, "指标不应满足严格的SLA");

    println!(
        "性能指标测试通过: 评分={:.2}, 满足宽松SLA={}, 满足严格SLA={}",
        score, meets_sla, fails_sla
    );
}

#[test]
fn test_exo_architecture_integration() {
    // EXO架构端到端集成测试
    println!("开始EXO架构端到端集成测试...");

    // 1. 创建设备拓扑
    let topology = create_heterogeneous_topology();
    println!("步骤1: 创建设备拓扑完成, {}个设备", topology.device_count());

    // 2. 选择并行策略
    let mut strategy_engine = ExoParallelStrategyEngine::new();
    let strategy_decision = strategy_engine
        .select_optimal_strategy(
            &topology,
            6.0,         // 6GB模型
            8,           // 批次大小8
            Some(120.0), // 120ms延迟预算
        )
        .expect("并行策略选择应成功");

    println!(
        "步骤2: 选择并行策略完成, 策略={:?}, 预测延迟={:.2}ms, 置信度={:.2}",
        strategy_decision.strategy,
        strategy_decision.predicted_latency_ms,
        strategy_decision.confidence
    );

    // 3. 初始化动态调整器
    let mut adjuster = ExoDynamicAdjuster::new();
    adjuster.set_current_strategy(strategy_decision.clone());

    println!("步骤3: 初始化动态调整器完成");

    // 4. 模拟性能监控和调整循环
    let performance_scenarios = vec![
        (95.0, 110.5, 7.2), // 良好性能
        (135.0, 85.3, 7.5), // 性能下降
        (155.0, 65.8, 7.8), // 严重性能下降
        (115.0, 95.7, 7.3), // 恢复
    ];

    for (i, (latency, throughput, memory)) in performance_scenarios.iter().enumerate() {
        let metrics = PerformanceMetrics::new(
            *latency,
            *throughput,
            *memory,
            8,   // 批次大小8
            6.0, // 模型大小6GB
        );

        println!(
            "步骤4.{}: 性能指标 - 延迟={:.1}ms, 吞吐量={:.1}tps, 内存={:.1}GB",
            i + 1,
            latency,
            throughput,
            memory
        );

        let adjustment_result = adjuster.evaluate_and_adjust(&metrics, &topology);
        assert!(adjustment_result.is_ok(), "调整评估应成功");

        if let Ok(Some(decision)) = adjustment_result {
            println!("      调整决策: {:?}", decision);
        } else {
            println!("      无需调整");
        }
    }

    // 5. 验证最终状态
    let performance_history = adjuster.get_performance_history();
    let adjustment_history = adjuster.get_adjustment_history();
    let current_strategy = adjuster.get_current_strategy();

    assert!(!performance_history.is_empty(), "性能历史记录不应为空");
    assert!(current_strategy.is_some(), "当前策略不应为None");

    println!("步骤5: 验证最终状态完成");
    println!("      性能历史记录数量: {}", performance_history.len());
    println!("      调整历史记录数量: {}", adjustment_history.len());
    println!("      当前策略: {:?}", current_strategy.unwrap().strategy);

    println!("EXO架构端到端集成测试通过!");
}

#[test]
fn test_error_handling() {
    // 测试错误处理
    let mut strategy_engine = ExoParallelStrategyEngine::new();

    // 测试空设备拓扑
    let empty_topology = DeviceTopology::from_devices(vec![]);
    let result = strategy_engine.select_optimal_strategy(&empty_topology, 5.0, 4, Some(100.0));

    assert!(result.is_err(), "空设备拓扑应返回错误");

    // 测试不合理的参数
    let topology = create_heterogeneous_topology();
    let result = strategy_engine.select_optimal_strategy(
        &topology,
        1000.0,    // 不合理的巨大模型
        1000,      // 不合理的巨大批次
        Some(1.0), // 不合理的严格延迟预算
    );

    // 注意：这个可能成功也可能失败，取决于算法实现
    // 我们只验证函数执行不崩溃
    println!("错误处理测试完成");
}

// ==================== 性能基准测试 ====================

#[test]
#[ignore = "性能基准测试，仅在需要时运行"]
fn benchmark_strategy_selection_performance() {
    // 策略选择性能基准测试
    use std::time::Instant;

    let topology = create_heterogeneous_topology();
    let mut strategy_engine = ExoParallelStrategyEngine::new();

    let start = Instant::now();
    let iterations = 100;

    for i in 0..iterations {
        let model_size = 1.0 + (i as f32 % 10.0); // 1-10GB模型
        let batch_size = 1 + (i % 8); // 1-8批次大小

        let _ =
            strategy_engine.select_optimal_strategy(&topology, model_size, batch_size, Some(100.0));
    }

    let elapsed = start.elapsed();
    let avg_time_ms = elapsed.as_millis() as f64 / iterations as f64;

    println!(
        "策略选择性能基准: {}次迭代, 平均时间={:.2}ms",
        iterations, avg_time_ms
    );

    // 性能要求：平均选择时间应小于50ms
    assert!(
        avg_time_ms < 50.0,
        "策略选择平均时间应小于50ms, 实际={:.2}ms",
        avg_time_ms
    );
}
