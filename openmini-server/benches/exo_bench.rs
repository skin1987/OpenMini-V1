//! EXO 架构性能基准测试套件
//!
//! # 测试目标
//!
//! 测量 EXO 分布式架构关键组件的性能：
//! - 设备拓扑分析和策略选择算法性能
//! - 动态调整器决策性能
//! - 性能指标计算效率
//! - 拓扑感知并行化开销
//!
//! # 运行方式
//!
//! ```bash
//! # 运行 EXO 基准测试（需要 nightly Rust）
//! cargo +nightly bench --package openmini-server --bench exo_bench
//!
//! # 运行特定基准测试组
//! cargo +nightly bench --package openmini-server --bench exo_bench -- strategy_selection
//!
//! # 生成 HTML 报告
//! cargo +nightly bench --package openmini-server --bench exo_bench -- --save-baseline exo
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use openmini_server::distributed::{
    AdjustmentDecision, DeviceCapabilities, DeviceInfo, DeviceResources, DeviceTopology,
    DeviceType, ExoDynamicAdjuster, ExoParallelStrategyEngine, LinkType, NetworkDevice,
    NetworkLink, NetworkTopology, ParallelStrategyDecision, PerformanceMetrics,
};
use openmini_server::model::inference::distributed_inference_config::ParallelStrategy;
use std::time::Duration;

/// 创建测试用的设备拓扑
fn create_test_topology(num_devices: usize, heterogeneous: bool) -> DeviceTopology {
    let mut devices = Vec::new();
    let mut network_topology = NetworkTopology::new();

    for i in 0..num_devices {
        let device_type = if heterogeneous && i % 2 == 0 {
            DeviceType::AppleGpu
        } else {
            DeviceType::Cpu
        };

        let device_info = DeviceInfo {
            device_id: format!("device-{}", i),
            device_type,
            device_state: openmini_server::distributed::DeviceState::Online,
            device_version: "1.0.0".to_string(),
            capabilities: DeviceCapabilities {
                supports_rdma: device_type == DeviceType::AppleGpu,
                supports_mlx: device_type == DeviceType::AppleGpu,
                max_memory_bandwidth_gbs: if device_type == DeviceType::AppleGpu {
                    80.0
                } else {
                    25.0
                },
                total_memory_bytes: if device_type == DeviceType::AppleGpu {
                    32 * 1024 * 1024 * 1024 // 32 GB
                } else {
                    16 * 1024 * 1024 * 1024 // 16 GB
                },
                available_memory_bytes: if device_type == DeviceType::AppleGpu {
                    24 * 1024 * 1024 * 1024 // 24 GB
                } else {
                    12 * 1024 * 1024 * 1024 // 12 GB
                },
                cpu_cores: if device_type == DeviceType::AppleGpu {
                    8
                } else {
                    16
                },
                cpu_frequency_ghz: if device_type == DeviceType::AppleGpu {
                    3.2
                } else {
                    2.8
                },
                network_bandwidth_gbps: 10.0,
                network_latency_ms: if device_type == DeviceType::AppleGpu {
                    0.1
                } else {
                    0.5
                },
                rdma_bandwidth_gbps: if device_type == DeviceType::AppleGpu {
                    100.0
                } else {
                    0.0
                },
                mlx_performance_score: if device_type == DeviceType::AppleGpu {
                    95.0
                } else {
                    30.0
                },
                power_efficiency: if device_type == DeviceType::AppleGpu {
                    0.9
                } else {
                    0.7
                },
            },
            resources: DeviceResources {
                cpu_utilization: 0.3,
                memory_utilization: 0.4,
                network_utilization: 0.1,
                gpu_utilization: if device_type == DeviceType::AppleGpu {
                    Some(0.5)
                } else {
                    None
                },
                temperature: if device_type == DeviceType::AppleGpu {
                    Some(65.0)
                } else {
                    None
                },
                power_consumption_watts: if device_type == DeviceType::AppleGpu {
                    Some(45.0)
                } else {
                    None
                },
            },
            last_heartbeat: std::time::SystemTime::now(),
            gpu_info: None,
        };

        devices.push(device_info);

        // 创建网络设备并添加到拓扑中
        let network_device = NetworkDevice {
            device_id: format!("device-{}", i),
            ip_address: format!("192.168.1.{}", i + 100),
            mac_address: format!("00:11:22:33:44:{:02x}", i),
            supported_link_types: vec![LinkType::Thunderbolt, LinkType::Ethernet],
            max_bandwidth_gbps: 10.0,
            latency_ms: 0.5,
            is_rdma_capable: device_type == DeviceType::AppleGpu,
        };

        network_topology.add_device(network_device.clone());

        // 添加一些网络连接（环状拓扑）
        if i > 0 {
            let link = NetworkLink {
                from_device_id: format!("device-{}", i - 1),
                to_device_id: format!("device-{}", i),
                link_type: LinkType::Thunderbolt,
                bandwidth_gbps: 10.0,
                latency_ms: 0.1,
                reliability: 0.99,
                is_rdma_enabled: true,
                mlx_optimized: device_type == DeviceType::AppleGpu,
            };
            network_topology.add_link(link);
        }
    }

    // 闭合环
    if num_devices > 1 {
        let link = NetworkLink {
            from_device_id: format!("device-{}", num_devices - 1),
            to_device_id: "device-0".to_string(),
            link_type: LinkType::Thunderbolt,
            bandwidth_gbps: 10.0,
            latency_ms: 0.1,
            reliability: 0.99,
            is_rdma_enabled: true,
            mlx_optimized: true,
        };
        network_topology.add_link(link);
    }

    DeviceTopology::new(devices, network_topology)
}

/// 基准测试：策略选择性能（不同设备数量）
fn bench_strategy_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("strategy_selection");
    group.measurement_time(Duration::from_secs(10));

    for num_devices in [2, 4, 8, 16].iter() {
        let topology = create_test_topology(*num_devices, true);
        let strategy_engine = ExoParallelStrategyEngine::new();

        group.bench_with_input(
            BenchmarkId::new("heterogeneous", num_devices),
            num_devices,
            |b, _| {
                b.iter(|| {
                    let decision = strategy_engine.select_optimal_strategy(
                        black_box(&topology),
                        black_box(ParallelStrategy::TensorParallel),
                        black_box(1000),
                        black_box(100),
                    );
                    black_box(decision);
                });
            },
        );

        let homogeneous_topology = create_test_topology(*num_devices, false);
        group.bench_with_input(
            BenchmarkId::new("homogeneous", num_devices),
            num_devices,
            |b, _| {
                b.iter(|| {
                    let decision = strategy_engine.select_optimal_strategy(
                        black_box(&homogeneous_topology),
                        black_box(ParallelStrategy::TensorParallel),
                        black_box(1000),
                        black_box(100),
                    );
                    black_box(decision);
                });
            },
        );
    }

    group.finish();
}

/// 基准测试：动态调整器决策性能
fn bench_dynamic_adjuster(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamic_adjuster");
    group.measurement_time(Duration::from_secs(10));

    // 创建不同规模的拓扑
    for num_devices in [2, 4, 8].iter() {
        let topology = create_test_topology(*num_devices, true);
        let adjuster = ExoDynamicAdjuster::new(Default::default());

        // 创建性能指标
        let metrics = PerformanceMetrics {
            inference_latency_ms: 150.0,
            throughput_tokens_per_second: 1000.0,
            memory_usage_percentage: 75.0,
            network_bandwidth_utilization: 60.0,
            gpu_utilization: Some(85.0),
            power_efficiency: 0.8,
            sla_violation_count: 2,
            total_requests: 1000,
            successful_requests: 980,
            avg_batch_size: 32.0,
            model_size_mb: 3500.0,
            current_strategy: ParallelStrategy::TensorParallel,
            timestamp: std::time::SystemTime::now(),
        };

        group.bench_with_input(
            BenchmarkId::new("adjustment_decision", num_devices),
            num_devices,
            |b, _| {
                b.iter(|| {
                    let decision = adjuster.analyze_and_decide(
                        black_box(&topology),
                        black_box(&metrics),
                        black_box(&metrics), // 使用相同的历史指标
                    );
                    black_box(decision);
                });
            },
        );
    }

    group.finish();
}

/// 基准测试：设备拓扑分析性能
fn bench_topology_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("topology_analysis");
    group.measurement_time(Duration::from_secs(10));

    for num_devices in [2, 4, 8, 16, 32].iter() {
        let topology = create_test_topology(*num_devices, true);

        group.bench_with_input(
            BenchmarkId::from_parameter(num_devices),
            num_devices,
            |b, _| {
                b.iter(|| {
                    // 分析拓扑属性
                    let compute_capacity = topology.total_compute_capacity();
                    let memory_capacity = topology.total_memory_capacity();
                    let network_capacity = topology.network_capacity();
                    let is_heterogeneous = topology.is_heterogeneous();
                    let bottleneck_analysis = topology.identify_bottlenecks();

                    black_box(compute_capacity);
                    black_box(memory_capacity);
                    black_box(network_capacity);
                    black_box(is_heterogeneous);
                    black_box(bottleneck_analysis);
                });
            },
        );
    }

    group.finish();
}

/// 基准测试：性能指标计算性能
fn bench_performance_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_metrics");
    group.measurement_time(Duration::from_secs(5));

    let metrics = PerformanceMetrics {
        inference_latency_ms: 150.0,
        throughput_tokens_per_second: 1000.0,
        memory_usage_percentage: 75.0,
        network_bandwidth_utilization: 60.0,
        gpu_utilization: Some(85.0),
        power_efficiency: 0.8,
        sla_violation_count: 2,
        total_requests: 1000,
        successful_requests: 980,
        avg_batch_size: 32.0,
        model_size_mb: 3500.0,
        current_strategy: ParallelStrategy::TensorParallel,
        timestamp: std::time::SystemTime::now(),
    };

    group.bench_function("calculate_performance_score", |b| {
        b.iter(|| {
            let score = metrics.calculate_performance_score();
            black_box(score);
        });
    });

    group.bench_function("check_sla_compliance", |b| {
        b.iter(|| {
            let compliance = metrics.check_sla_compliance(100.0, 800.0, 90.0);
            black_box(compliance);
        });
    });

    group.bench_function("calculate_efficiency", |b| {
        b.iter(|| {
            let efficiency = metrics.calculate_efficiency();
            black_box(efficiency);
        });
    });

    group.finish();
}

/// 基准测试：端到端 EXO 工作流性能
fn bench_exo_workflow(c: &mut Criterion) {
    let mut group = c.benchmark_group("exo_workflow");
    group.measurement_time(Duration::from_secs(15));

    let topology = create_test_topology(4, true);
    let strategy_engine = ExoParallelStrategyEngine::new();
    let adjuster = ExoDynamicAdjuster::new(Default::default());

    group.bench_function("complete_workflow", |b| {
        b.iter(|| {
            // 步骤1：选择初始策略
            let strategy_decision = strategy_engine.select_optimal_strategy(
                black_box(&topology),
                black_box(ParallelStrategy::TensorParallel),
                black_box(1000),
                black_box(100),
            );

            // 步骤2：模拟性能指标
            let metrics = PerformanceMetrics {
                inference_latency_ms: if strategy_decision.predicted_latency_ms > 0.0 {
                    strategy_decision.predicted_latency_ms * 1.2 // 模拟性能下降
                } else {
                    150.0
                },
                throughput_tokens_per_second: strategy_decision.predicted_throughput_tps * 0.8,
                memory_usage_percentage: 75.0,
                network_bandwidth_utilization: 60.0,
                gpu_utilization: Some(85.0),
                power_efficiency: 0.8,
                sla_violation_count: 2,
                total_requests: 1000,
                successful_requests: 980,
                avg_batch_size: 32.0,
                model_size_mb: 3500.0,
                current_strategy: strategy_decision.selected_strategy,
                timestamp: std::time::SystemTime::now(),
            };

            // 步骤3：动态调整决策
            let adjustment_decision = adjuster.analyze_and_decide(
                black_box(&topology),
                black_box(&metrics),
                black_box(&metrics),
            );

            black_box(strategy_decision);
            black_box(metrics);
            black_box(adjustment_decision);
        });
    });

    group.finish();
}

criterion_group!(
    name = exo_benches;
    config = Criterion::default()
        .sample_size(100)
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(10))
        .noise_threshold(0.05);
    targets = bench_strategy_selection, bench_dynamic_adjuster, bench_topology_analysis, bench_performance_metrics, bench_exo_workflow
);

criterion_main!(exo_benches);
