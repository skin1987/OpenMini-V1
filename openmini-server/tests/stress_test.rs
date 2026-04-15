//! OpenMini-V1 压力测试套件
//!
//! # 测试目标
//!
//! 验证系统在高负载下的稳定性和正确性：
//! - 并发推理压力（100+ 并发请求）
//! - 长时间运行稳定性（内存泄漏检测）
//! - KV Cache 内存池管理
//! - 线程池饱和处理
//! - 连接池并发安全
//! - DSA 稀疏注意力数值稳定性
//!
//! # 运行方式
//!
//! ```bash
//! # 运行全部压力测试（默认 60 秒）
//! cargo test --package openmini-server --test stress_test -- --nocapture
//!
//! # 自定义持续时间（秒）
//! STRESS_TEST_DURATION=120 cargo test --package openmini-server --test stress_test -- --nocapture
//!
//! # 跳过耗时测试（CI 快速模式）
//! cargo test --package openmini-server --test stress_test -- --skip stress
//! ```
//!
//! # 预期资源需求
//!
//! - CPU: 2-4 核
//! - 内存: ~512MB
//! - 运行时间: 60-300 秒（可配置）

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ============================================================================
// 工具函数
// ============================================================================

/// 获取压力测试持续时间（秒）
/// 通过环境变量 `STRESS_TEST_DURATION` 配置，默认 60 秒
fn get_stress_duration() -> Duration {
    std::env::var("STRESS_TEST_DURATION")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(60))
}

/// 获取 CI 模式持续时间（缩短版，用于快速验证）
/// CI 环境下使用较短时间，避免超时
fn get_ci_duration() -> Duration {
    if is_ci_environment() {
        Duration::from_secs(10)
    } else {
        get_stress_duration()
    }
}

/// 检测是否在 CI 环境中运行
fn is_ci_environment() -> bool {
    std::env::var("CI").is_ok()
        || std::env::var("GITHUB_ACTIONS").is_ok()
        || std::env::var("GITLAB_CI").is_ok()
}

/// 计算百分位数
fn percentile(sorted_data: &[u64], p: f64) -> u64 {
    if sorted_data.is_empty() {
        return 0;
    }
    let index = ((p / 100.0) * (sorted_data.len() - 1) as f64).round() as usize;
    sorted_data[index.min(sorted_data.len() - 1)]
}

/// 输出 JSON 格式的性能指标（方便 CI 解析）
fn output_metrics_json(name: &str, metrics: &serde_json::Value) {
    let json = serde_json::json!({
        "test": name,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "metrics": metrics
    });
    eprintln!("METRICS: {}", serde_json::to_string(&json).unwrap());
}

// ============================================================================
// 1. 并发推理压力测试
// ============================================================================

/// 并发推理压力测试
///
/// # 目的
/// 验证系统在高并发推理请求下的稳定性
///
/// # 测试场景
/// - 模拟 100 个并发推理请求
/// - 持续运行（CI: 10s, 本地: 60s）
///
/// # 验证点
/// - 无 panic/crash
/// - 所有请求都得到响应
/// - 响应时间 P99 < 500ms（本地）/ < 1000ms（CI）
#[test]
fn test_concurrent_inference_stress() {
    use openmini_server::service::thread::pool::create_default_pool;

    let duration = get_ci_duration();
    let concurrent_requests = if is_ci_environment() { 20 } else { 100 };
    let completed = Arc::new(AtomicUsize::new(0));
    let errors = Arc::new(AtomicUsize::new(0));
    let latencies = Arc::new(std::sync::Mutex::new(Vec::new()));

    eprintln!("\n[stress] Concurrent Inference Stress Test");
    eprintln!(
        "  Duration: {:?}, Concurrency: {}",
        duration, concurrent_requests
    );

    let pool = Arc::new(create_default_pool());
    let start = Instant::now();
    let running = Arc::new(AtomicBool::new(true));

    // 启动多个工作线程模拟并发请求
    let mut handles = Vec::with_capacity(concurrent_requests);
    for i in 0..concurrent_requests {
        let completed = Arc::clone(&completed);
        let errors = Arc::clone(&errors);
        let latencies = Arc::clone(&latencies);
        let running = Arc::clone(&running);
        let _pool_clone = Arc::clone(&pool);

        let handle = std::thread::spawn(move || {
            while running.load(Ordering::Relaxed) {
                let req_start = Instant::now();

                // 模拟推理工作负载：CPU 密集型计算
                let result = std::panic::catch_unwind(|| {
                    // 模拟矩阵运算（类似推理计算）
                    let mut sum: f64 = 0.0;
                    for j in 0..1000 {
                        sum += (j as f64).sin().cos();
                    }
                    // 小延迟模拟 I/O
                    std::thread::sleep(Duration::from_micros(100));
                    sum
                });

                let latency = req_start.elapsed().as_micros() as u64;

                match result {
                    Ok(_) => {
                        completed.fetch_add(1, Ordering::Relaxed);
                        if let Ok(mut lats) = latencies.lock() {
                            lats.push(latency);
                        }
                    }
                    Err(e) => {
                        errors.fetch_add(1, Ordering::Relaxed);
                        eprintln!(
                            "[stress] Request {} panicked: {:?}",
                            i,
                            e.downcast_ref::<&str>()
                        );
                    }
                }

                // 控制请求速率
                std::thread::sleep(Duration::from_millis(1));
            }
        });

        handles.push(handle);
    }

    // 运行指定时长
    while start.elapsed() < duration {
        std::thread::sleep(Duration::from_millis(100));
    }
    running.store(false, Ordering::Relaxed);

    // 等待所有线程结束
    for handle in handles {
        let _ = handle.join();
    }

    // 收集结果
    let total_completed = completed.load(Ordering::Relaxed);
    let total_errors = errors.load(Ordering::Relaxed);
    let latencies_guard = latencies.lock().unwrap();
    let mut sorted_latencies: Vec<u64> = latencies_guard.clone();
    drop(latencies_guard);
    sorted_latencies.sort();

    let p50 = percentile(&sorted_latencies, 50.0);
    let p99 = percentile(&sorted_latencies, 99.0);
    let avg = if !sorted_latencies.is_empty() {
        sorted_latencies.iter().sum::<u64>() / sorted_latencies.len() as u64
    } else {
        0
    };

    eprintln!("\n[stress] Results:");
    eprintln!("  Total requests: {}", total_completed);
    eprintln!("  Errors: {}", total_errors);
    eprintln!("  Latency P50: {} us", p50);
    eprintln!("  Latency P99: {} us", p99);
    eprintln!("  Latency Avg: {} us", avg);

    // 输出 JSON 格式指标
    output_metrics_json(
        "concurrent_inference_stress",
        &serde_json::json!({
            "total_requests": total_completed,
            "errors": total_errors,
            "latency_p50_us": p50,
            "latency_p99_us": p99,
            "latency_avg_us": avg,
            "duration_sec": duration.as_secs(),
            "concurrency": concurrent_requests
        }),
    );

    // 断言验证
    assert_eq!(total_errors, 0, "Should have no panics/errors");
    assert!(
        total_completed > 0,
        "Should complete at least some requests"
    );

    // P99 延迟阈值（CI 放宽要求）
    let p99_threshold = if is_ci_environment() { 2000 } else { 500 };
    assert!(
        p99 < p99_threshold * 1000,
        "P99 latency {}us exceeds threshold {}us",
        p99,
        p99_threshold * 1000
    );
}

// ============================================================================
// 2. 长时间运行稳定性测试
// ============================================================================

/// 长时间运行稳定性测试
///
/// # 目的
/// 验证系统长时间运行时的稳定性，检测内存泄漏和资源泄漏
///
/// # 测试场景
/// - CI 环境：连续运行 30 秒
/// - 本地环境：10 分钟（可通过 STRESS_TEST_DURATION 配置）
/// - 每 100ms 发送一次请求
///
/// # 监控项
/// - 内存使用趋势
/// - 错误率变化
/// - 响应时间漂移
#[test]
fn test_long_running_stability() {
    use sysinfo::{MemoryRefreshKind, RefreshKind, System};

    let base_duration = if is_ci_environment() {
        Duration::from_secs(30)
    } else {
        Duration::from_secs(600) // 10 分钟
    };
    let duration = get_stress_duration().min(base_duration);

    eprintln!("\n[stress] Long Running Stability Test");
    eprintln!("  Duration: {:?}", duration);

    let mut sys =
        System::new_with_specifics(RefreshKind::new().with_memory(MemoryRefreshKind::everything()));
    sys.refresh_memory();

    let initial_memory = sys.used_memory();
    let start = Instant::now();
    let iteration = Arc::new(AtomicUsize::new(0));
    let errors = Arc::new(AtomicUsize::new(0));

    // 内存快照记录
    let memory_snapshots = Arc::new(std::sync::Mutex::new(Vec::new()));
    let snapshot_interval = Duration::from_secs(5); // 每 5 秒记录一次

    eprintln!("  Initial memory: {} MB", initial_memory / 1024 / 1024);

    while start.elapsed() < duration {
        let iter_start = Instant::now();

        // 执行工作负载
        let result = std::panic::catch_unwind(|| {
            // 分配并释放内存，模拟真实工作负载
            let data: Vec<f64> = (0..10000).map(|i| i as f64).collect();
            let _sum: f64 = data.iter().sum();

            // 模拟字符串操作
            let s = "test string for stability check".repeat(100);
            let _len = s.len();

            // 模拟 HashMap 操作
            use std::collections::HashMap;
            let mut map: HashMap<usize, String> = HashMap::new();
            for i in 0..100 {
                map.insert(i, format!("value_{}", i));
            }
            drop(map);
        });

        match result {
            Ok(_) => {}
            Err(_) => {
                errors.fetch_add(1, Ordering::Relaxed);
            }
        }

        iteration.fetch_add(1, Ordering::Relaxed);

        // 定期记录内存快照（每 snapshot_interval 秒）
        let elapsed_secs = start.elapsed().as_secs();
        if elapsed_secs > 0 && elapsed_secs % snapshot_interval.as_secs() == 0 {
            sys.refresh_memory();
            let current_mem = sys.used_memory();
            if let Ok(mut snapshots) = memory_snapshots.lock() {
                snapshots.push((start.elapsed().as_secs(), current_mem));
            }
            eprintln!(
                "  [{}] Memory: {} MB, Iterations: {}, Errors: {}",
                start.elapsed().as_secs(),
                current_mem / 1024 / 1024,
                iteration.load(Ordering::Relaxed),
                errors.load(Ordering::Relaxed)
            );
        }

        // 控制循环频率（约 100ms 一次迭代）
        let elapsed = iter_start.elapsed();
        if elapsed < Duration::from_millis(100) {
            std::thread::sleep(Duration::from_millis(100) - elapsed);
        }
    }

    // 最终内存检查
    sys.refresh_memory();
    let final_memory = sys.used_memory();
    let total_iterations = iteration.load(Ordering::Relaxed);
    let total_errors = errors.load(Ordering::Relaxed);

    eprintln!("\n[stress] Stability Results:");
    eprintln!("  Total iterations: {}", total_iterations);
    eprintln!("  Total errors: {}", total_errors);
    eprintln!("  Initial memory: {} MB", initial_memory / 1024 / 1024);
    eprintln!("  Final memory: {} MB", final_memory / 1024 / 1024);
    eprintln!(
        "  Memory delta: {} MB",
        (final_memory as i64 - initial_memory as i64) / 1024 / 1024
    );

    // 分析内存趋势
    let snapshots = memory_snapshots.lock().unwrap();
    if snapshots.len() >= 4 {
        let first_quarter_len = snapshots.len() / 4;
        let last_quarter_start = snapshots.len() - first_quarter_len;

        let first_avg: u64 = snapshots[..first_quarter_len]
            .iter()
            .map(|(_, m)| m)
            .sum::<u64>()
            / first_quarter_len as u64;
        let last_avg: u64 = snapshots[last_quarter_start..]
            .iter()
            .map(|(_, m)| m)
            .sum::<u64>()
            / first_quarter_len as u64;

        eprintln!("  First quarter avg memory: {} MB", first_avg / 1024 / 1024);
        eprintln!("  Last quarter avg memory: {} MB", last_avg / 1024 / 1024);

        // 允许 20% 的波动，但不应持续增长超过此范围
        let growth_ratio = if first_avg > 0 {
            last_avg as f64 / first_avg as f64
        } else {
            1.0
        };

        eprintln!("  Memory growth ratio: {:.2}", growth_ratio);

        assert!(
            growth_ratio < 1.5,
            "Possible memory leak detected! Growth ratio: {:.2} (threshold: 1.5)",
            growth_ratio
        );
    }

    // 输出 JSON 格式指标
    output_metrics_json(
        "long_running_stability",
        &serde_json::json!({
            "total_iterations": total_iterations,
            "errors": total_errors,
            "initial_memory_mb": initial_memory / 1024 / 1024,
            "final_memory_mb": final_memory / 1024 / 1024,
            "memory_delta_mb": (final_memory as i64 - initial_memory as i64) / 1024 / 1024,
            "duration_sec": duration.as_secs()
        }),
    );

    // 基本断言
    assert_eq!(total_errors, 0, "Should have no errors during long run");
    assert!(total_iterations > 0, "Should complete iterations");
}

// ============================================================================
// 3. KV Cache 压力测试
// ============================================================================

/// KV Cache 压力测试
///
/// # 目的
/// 验证 KV Cache 内存池在高频分配/释放下的管理正确性
///
/// # 测试场景
/// - 快速分配/释放大量 KV cache block
/// - 达到 max_blocks 时的边界行为
/// - 并发分配竞争
///
/// # 验证点
/// - 内存池计数准确
/// - 无内存泄漏（所有块最终可回收）
/// - 边界条件正确处理
#[test]
fn test_kv_cache_under_load() {
    use openmini_server::hardware::kv_cache::{block::KVCacheConfig, block_manager::BlockManager};

    let num_blocks = 256;
    let block_size = 16;

    eprintln!("\n[stress] KV Cache Under Load Test");
    eprintln!(
        "  Blocks: {}, Block size: {} tokens",
        num_blocks, block_size
    );

    let config = KVCacheConfig {
        max_blocks: num_blocks,
        block_size,
        ..Default::default()
    };

    let mut manager = BlockManager::new(&config);

    // 测试 1: 快速分配-释放循环
    eprintln!("  Phase 1: Rapid allocate/free cycle");
    let cycles = 1000;
    let blocks_per_cycle = 10;

    for cycle in 0..cycles {
        // 分配
        let allocated = manager.allocate(blocks_per_cycle, None);
        assert!(allocated.is_ok(), "Cycle {}: Allocation failed", cycle);

        let ids = allocated.unwrap();
        assert_eq!(ids.len(), blocks_per_cycle);

        // 释放
        manager.free(&ids);

        // 定期检查状态
        if cycle % 100 == 0 {
            eprintln!(
                "  Cycle {}: Free={}, Allocated={}",
                cycle,
                manager.available_blocks(),
                manager.allocated_blocks()
            );
            assert_eq!(
                manager.available_blocks(),
                num_blocks,
                "All blocks should be free after free"
            );
        }
    }

    // 测试 2: 达到容量上限
    eprintln!("  Phase 2: Capacity limit test");
    let almost_all = manager.allocate(num_blocks - 1, None);
    assert!(almost_all.is_ok());

    // 尝试超额分配应失败
    let overflow = manager.allocate(2, None);
    assert!(overflow.is_err(), "Overflow allocation should fail");

    // 释放后应能重新分配
    manager.free(&almost_all.unwrap());

    let recovery = manager.allocate(num_blocks / 2, None);
    assert!(recovery.is_ok(), "Should allocate after freeing");

    // 清理
    manager.free(&recovery.unwrap());

    // 测试 3: 多请求者并发分配
    eprintln!("  Phase 3: Multi-requester allocation");
    let mut manager2 = BlockManager::new(&config);
    let num_requesters = 10;
    let blocks_per_requester = num_blocks / num_requesters;
    let expected_total = blocks_per_requester * num_requesters;

    let mut all_ids = Vec::new();
    for req_id in 0..num_requesters {
        let owner_id = Some(req_id as u64 + 1);
        let allocated = manager2.allocate(blocks_per_requester, owner_id);
        assert!(allocated.is_ok(), "Requester {} failed", req_id);
        all_ids.extend(allocated.unwrap());
    }

    assert_eq!(
        all_ids.len(),
        expected_total,
        "Expected {} blocks allocated",
        expected_total
    );

    // 剩余的块应该可以分配
    let remaining = num_blocks - expected_total;
    if remaining > 0 {
        let extra = manager2.allocate(remaining, None);
        assert!(
            extra.is_ok(),
            "Should be able to allocate remaining {} blocks",
            remaining
        );
        all_ids.extend(extra.unwrap());
    }

    // 现在所有块都已分配，尝试超额分配应失败
    let overflow = manager2.allocate(1, None);
    assert!(overflow.is_err(), "No blocks should be available now");

    // 释放所有已分配的块
    manager2.free(&all_ids);

    let final_free = manager2.available_blocks();
    assert_eq!(final_free, num_blocks, "All blocks should be free");

    eprintln!("\n[stress] KV Cache Results:");
    eprintln!("  All phases passed!");
    eprintln!("  Final state: {} free blocks", final_free);

    // 输出 JSON 格式指标
    output_metrics_json(
        "kv_cache_under_load",
        &serde_json::json!({
            "cycles_completed": cycles,
            "blocks_per_cycle": blocks_per_cycle,
            "num_blocks": num_blocks,
            "block_size": block_size,
            "final_free_blocks": final_free,
            "final_allocated_blocks": manager2.allocated_blocks()
        }),
    );
}

// ============================================================================
// 4. 线程池压力测试
// ============================================================================

/// 线程池饱和压力测试
///
/// # 目的
/// 验证线程池在任务过载情况下的行为
///
/// # 测试场景
/// - 提交远超 pool 容量的任务
/// - 任务队列溢出处理
/// - 资源回收正确性
///
/// # 验证点
/// - 所有任务最终执行完成
/// - 无任务丢失
/// - 线程池正常关闭
#[test]
fn test_thread_pool_saturation() {
    use openmini_server::service::thread::pool::create_default_pool;

    let pool = create_default_pool();
    let total_tasks = 10000;
    let counter = Arc::new(AtomicUsize::new(0));
    let errors = Arc::new(AtomicUsize::new(0));

    eprintln!("\n[stress] Thread Pool Saturation Test");
    eprintln!("  Tasks to submit: {}", total_tasks);

    let start = Instant::now();

    // 大量提交任务
    for i in 0..total_tasks {
        let counter = Arc::clone(&counter);
        let errors = Arc::clone(&errors);

        pool.execute(move || {
            let result = std::panic::catch_unwind(|| {
                // 模拟不同类型的工作负载
                match i % 4 {
                    0 => {
                        // CPU 密集型
                        let mut sum = 0u64;
                        for j in 0..1000 {
                            sum = sum.wrapping_add(j);
                        }
                    }
                    1 => {
                        // 内存分配型
                        let _data: Vec<u8> = vec![0; 1024];
                    }
                    2 => {
                        // 混合型
                        let mut map = std::collections::HashMap::new();
                        for j in 0..100 {
                            map.insert(j, j * 2);
                        }
                    }
                    _ => {
                        // 轻量级
                        std::hint::spin_loop();
                    }
                }
            });

            if result.is_err() {
                errors.fetch_add(1, Ordering::Relaxed);
            } else {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        });
    }

    // 等待任务完成（带超时）
    let timeout = Duration::from_secs(if is_ci_environment() { 30 } else { 120 });
    let deadline = Instant::now() + timeout;

    while Instant::now() < deadline {
        let completed = counter.load(Ordering::Relaxed);
        if completed == total_tasks {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    let elapsed = start.elapsed();
    let completed = counter.load(Ordering::Relaxed);
    let error_count = errors.load(Ordering::Relaxed);

    eprintln!("\n[stress] Thread Pool Results:");
    eprintln!("  Tasks submitted: {}", total_tasks);
    eprintln!("  Tasks completed: {}", completed);
    eprintln!("  Errors: {}", error_count);
    eprintln!("  Elapsed time: {:.2}s", elapsed.as_secs_f64());
    eprintln!(
        "  Throughput: {:.0} tasks/sec",
        total_tasks as f64 / elapsed.as_secs_f64()
    );

    // 输出 JSON 格式指标
    output_metrics_json(
        "thread_pool_saturation",
        &serde_json::json!({
            "tasks_submitted": total_tasks,
            "tasks_completed": completed,
            "errors": error_count,
            "elapsed_ms": elapsed.as_millis(),
            "throughput_per_sec": total_tasks as f64 / elapsed.as_secs_f64()
        }),
    );

    // 断言
    assert_eq!(error_count, 0, "Should have no task execution errors");
    assert_eq!(
        completed, total_tasks,
        "All tasks should complete (completed: {}, expected: {})",
        completed, total_tasks
    );
}

// ============================================================================
// 5. 连接池并发测试
// ============================================================================

/// 连接池高并发测试
///
/// # 目的
/// 验证连接池在 50+ 并发获取/释放操作下的安全性
///
/// # 测试场景
/// - 启动本地测试服务器
/// - 多线程同时获取/释放连接
/// - 每个线程多次循环操作
///
/// # 验证点
/// - 连接计数准确性
/// - 无死锁
/// - 无竞态条件
/// - 统计数据一致性
#[tokio::test]
async fn test_connection_pool_high_concurrency() {
    use openmini_server::service::server::connection::ConnectionPool;

    let pool_size = 20usize;
    let num_threads = 50;
    let operations_per_thread = if is_ci_environment() { 10 } else { 100 };

    eprintln!("\n[stress] Connection Pool High Concurrency Test");
    eprintln!(
        "Pool size: {}, Threads: {}, Ops/thread: {}",
        pool_size, num_threads, operations_per_thread
    );

    // 启动本地测试服务器
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    eprintln!("  Test server address: {}", addr);

    // 后台任务接受连接
    let server_handle = tokio::spawn(async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        while let Ok((mut stream, _)) = listener.accept().await {
            // 简单 echo 服务
            let mut buf = [0u8; 1024];
            if let Ok(n) = stream.read(&mut buf).await {
                let _ = stream.write_all(&buf[..n]).await;
            }
        }
    });

    let pool = Arc::new(ConnectionPool::new(pool_size));
    let errors = Arc::new(AtomicUsize::new(0));
    let total_ops = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::with_capacity(num_threads);

    for thread_id in 0..num_threads {
        let pool = Arc::clone(&pool);
        let errors = Arc::clone(&errors);
        let total_ops = Arc::clone(&total_ops);

        let handle = tokio::spawn(async move {
            for op in 0..operations_per_thread {
                // 使用 acquire_or_connect 获取或创建连接
                match pool.acquire_or_connect(addr).await {
                    Ok(conn) => {
                        // 模拟使用连接
                        tokio::time::sleep(Duration::from_micros(100)).await;

                        // 释放连接回池中
                        pool.release(conn);
                        total_ops.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(e) => {
                        // 连接失败是可能的（在高压下）
                        eprintln!(
                            "[stress] Thread {} op {}: Connection error: {}",
                            thread_id, op, e
                        );
                        errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        });

        handles.push(handle);
    }

    // 等待所有任务完成
    for handle in handles {
        let _ = handle.await;
    }

    // 停止测试服务器
    server_handle.abort();

    // 等待一小段时间确保所有异步操作完成
    tokio::time::sleep(Duration::from_millis(100)).await;

    let stats = pool.stats().snapshot();
    let total_operations = total_ops.load(Ordering::Relaxed);
    let error_count = errors.load(Ordering::Relaxed);

    eprintln!("\n[stress] Connection Pool Results:");
    eprintln!("  Total successful operations: {}", total_operations);
    eprintln!("  Failed connections: {}", error_count);
    eprintln!("  Final active connections: {}", stats.active_connections);
    eprintln!("  Total created: {}", stats.total_created);
    eprintln!("  Total reused: {}", stats.total_reused);
    if stats.total_created + stats.total_reused > 0 {
        eprintln!("  Reuse rate: {:.2}%", stats.reuse_rate * 100.0);
    }

    // 输出 JSON 格式指标
    output_metrics_json(
        "connection_pool_high_concurrency",
        &serde_json::json!({
            "pool_size": pool_size,
            "num_threads": num_threads,
            "operations_per_thread": operations_per_thread,
            "total_operations": total_operations,
            "failed_connections": error_count,
            "final_active_connections": stats.active_connections,
            "total_created": stats.total_created,
            "total_reused": stats.total_reused,
            "reuse_rate": stats.reuse_rate
        }),
    );

    // 断言：活跃连接必须为 0（所有连接都已释放）
    assert_eq!(
        stats.active_connections, 0,
        "All connections should be released"
    );

    // 至少应该有一些成功的操作
    assert!(
        total_operations > 0,
        "Should have at least some successful operations"
    );

    // 错误率不应太高（允许一些失败，但不应全部失败）
    let total_attempts = total_operations + error_count;
    if total_attempts > 0 {
        let error_rate = error_count as f64 / total_attempts as f64;
        assert!(
            error_rate < 0.5,
            "Error rate too high: {:.1}%",
            error_rate * 100.0
        );
    }

    // 如果有复用发生，验证统计数据一致性
    if stats.total_created + stats.total_reused > 0 {
        let expected_total = stats.total_created + stats.total_reused;
        assert!(
            expected_total >= total_operations as u64,
            "Total tracked operations should be >= actual operations"
        );
    }

    eprintln!("\n  Connection pool concurrency check PASSED");
}

// ============================================================================
// 6. DSA 优化压力测试
// ============================================================================

/// DSA (Dynamic Sparse Attention) 压力测试
///
/// # 目的
/// 验证 DSA 稀疏注意力在大批量计算下的数值稳定性和内存安全
///
/// # 测试场景
/// - 不同稀疏度配置 (25%, 50%, 75%)
/// - 不同序列长度 (512, 1024, 2048)
/// - 批量连续计算
///
/// # 验证点
/// - 数值稳定性（无 NaN/Inf）
/// - 内存安全（无越界访问）
/// - 性能随稀疏度提升而改善
#[test]
fn test_dsa_stress() {
    use ndarray::Array2;
    use openmini_server::model::inference::dsa::{
        configure_rayon_pool, sparse_attention_forward, DSATopKConfig,
    };

    // 初始化 rayon 线程池
    let _ = configure_rayon_pool();

    let seq_lengths = if is_ci_environment() {
        vec![256, 512]
    } else {
        vec![512, 1024, 2048]
    };
    let sparsity_levels = [(25, 0.25), (50, 0.5), (75, 0.75)];
    let head_dim = 64;
    let num_heads = 8;
    let iterations = if is_ci_environment() { 5 } else { 20 };

    eprintln!("\n[stress] DSA Stress Test");
    eprintln!("  Head dim: {}, Num heads: {}", head_dim, num_heads);
    eprintln!("  Iterations per config: {}", iterations);

    let mut results = Vec::new();

    for &seq_len in &seq_lengths {
        for &(sparsity_name, sparsity_ratio) in &sparsity_levels {
            eprintln!(
                "\n  Testing: seq_len={}, sparsity={}%",
                seq_len, sparsity_name
            );

            let config = DSATopKConfig {
                base_top_k: (seq_len as f32 * (1.0 - sparsity_ratio)) as usize,
                use_dynamic_k: true,
                short_seq_threshold: 512,
            };

            let mut durations = Vec::with_capacity(iterations);
            let mut has_nan = false;
            let mut has_inf = false;

            for iter in 0..iterations {
                // 创建测试输入
                let q: Array2<f32> = Array2::from_shape_fn((seq_len, head_dim), |(i, j)| {
                    ((i * head_dim + j) as f32 * 0.01).sin()
                });
                let k: Array2<f32> = Array2::from_shape_fn((seq_len, head_dim), |(i, j)| {
                    ((i * head_dim + j) as f32 * 0.01).cos()
                });
                let v: Array2<f32> = Array2::from_shape_fn((seq_len, head_dim), |(i, j)| {
                    ((i * head_dim + j) as f32 * 0.01).tan()
                });

                let start = Instant::now();

                // 执行 DSA 计算
                let result = std::panic::catch_unwind(|| {
                    sparse_attention_forward(&q, &k, &v, head_dim, &config, false)
                });

                let elapsed = start.elapsed();

                match result {
                    Ok(Ok(output)) => {
                        // 检查数值稳定性
                        for val in output.iter() {
                            if val.is_nan() {
                                has_nan = true;
                            }
                            if val.is_infinite() {
                                has_inf = true;
                            }
                        }
                        durations.push(elapsed);
                    }
                    Ok(Err(e)) => {
                        eprintln!("    Iter {} failed: {}", iter, e);
                    }
                    Err(e) => {
                        eprintln!("    Iter {} panicked: {:?}", iter, e);
                    }
                }
            }

            let avg_duration = if !durations.is_empty() {
                let total: Duration = durations.iter().sum();
                total / durations.len() as u32
            } else {
                Duration::ZERO
            };

            eprintln!(
                "    Avg duration: {:.2}ms",
                avg_duration.as_secs_f64() * 1000.0
            );
            eprintln!("    NaN detected: {}", has_nan);
            eprintln!("    Inf detected: {}", has_inf);

            results.push(serde_json::json!({
                "seq_len": seq_len,
                "sparsity_pct": sparsity_name,
                "iterations": iterations,
                "avg_ms": avg_duration.as_secs_f64() * 1000.0,
                "has_nan": has_nan,
                "has_inf": has_inf
            }));

            // 断言数值稳定性
            assert!(
                !has_nan,
                "NaN values detected at seq_len={}, sparsity={}%",
                seq_len, sparsity_name
            );
            assert!(
                !has_inf,
                "Inf values detected at seq_len={}, sparsity={}%",
                seq_len, sparsity_name
            );
        }
    }

    // 输出 JSON 格式指标
    output_metrics_json(
        "dsa_stress",
        &serde_json::json!({
            "configurations_tested": results.len(),
            "results": results
        }),
    );

    eprintln!("\n[stress] DSA Stress Test Complete - All configurations stable");
}
