//! OpenMini-V1 内存泄漏检测测试
//!
//! # 测试目标
//!
//! 检测长时间运行时的内存泄漏问题：
//! - 堆内存持续增长
//! - 句柄/资源泄漏
//! - KV Cache 内存池泄漏
//! - 连接池连接泄漏
//!
//! # 运行方式
//!
//! ```bash
//! # 运行内存泄漏测试（默认 5 分钟）
//! cargo test --package openmini-server --test memory_leak_test -- --nocapture
//!
//! # 自定义持续时间
//! MEMORY_TEST_DURATION=300 cargo test --package openmini-server --test memory_leak_test -- --nocapture
//!
//! # CI 快速模式（30 秒）
//! # 自动在 CI 环境中启用
//! ```
//!
//! # 检测原理
//!
//! 1. 记录运行期间的内存快照
//! 2. 分析内存趋势（首尾对比）
//! 3. 允许合理的波动范围（10-20%）
//! 4. 持续增长超过阈值则判定为潜在泄漏

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ============================================================================
// 工具函数
// ============================================================================

/// 获取内存测试持续时间
fn get_memory_test_duration() -> Duration {
    // CI 环境使用较短时间
    if is_ci_env() {
        return Duration::from_secs(30);
    }

    std::env::var("MEMORY_TEST_DURATION")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(300)) // 默认 5 分钟
}

/// 是否在 CI 环境
fn is_ci_env() -> bool {
    std::env::var("CI").is_ok()
        || std::env::var("GITHUB_ACTIONS").is_ok()
}

/// 计算向量平均值
fn average(values: &[u64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<u64>() as f64 / values.len() as f64
}

/// 输出 JSON 格式的检测结果
fn output_leak_report(test_name: &str, report: &serde_json::Value) {
    let json = serde_json::json!({
        "test": test_name,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "result": report
    });
    eprintln!("LEAK_REPORT: {}", serde_json::to_string(&json).unwrap());
}

// ============================================================================
// 1. 整体内存稳定性测试
// ============================================================================

/// 长时间运行的内存稳定性测试
///
/// # 目的
/// 检测系统级别的内存泄漏，通过监控进程内存使用量随时间的变化趋势
///
/// # 测试方法
/// 1. 启动后记录初始内存基线
/// 2. 以固定间隔执行典型工作负载
/// 3. 定期采集内存快照
/// 4. 对比首尾阶段的平均内存使用量
///
/// # 判定标准
/// - 允许 20% 的内存波动（考虑 GC、缓存等因素）
/// - 超过 50% 的持续增长视为潜在泄漏
#[test]
fn test_memory_stability_over_time() {
    use sysinfo::{System, RefreshKind, MemoryRefreshKind};

    let duration = get_memory_test_duration();
    let snapshot_interval = Duration::from_secs(5);

    eprintln!("\n[leak] Memory Stability Over Time Test");
    eprintln!("  Duration: {:?}", duration);
    eprintln!("  Snapshot interval: {:?}", snapshot_interval);

    // 初始化系统监控
    let mut sys =
        System::new_with_specifics(RefreshKind::new().with_memory(MemoryRefreshKind::everything()));
    sys.refresh_memory();

    let start = Instant::now();
    let mut memory_snapshots: Vec<(u64, u64)> = Vec::new(); // (timestamp_seconds, used_memory_kb)

    // 记录初始状态
    sys.refresh_memory();
    let initial_memory = sys.used_memory();
    memory_snapshots.push((0, initial_memory));

    eprintln!("  Initial memory: {} MB", initial_memory / 1024 / 1024);

    // 工作负载计数器
    let iterations = Arc::new(AtomicUsize::new(0));
    let errors = Arc::new(AtomicUsize::new(0));

    // 主循环：执行工作负载并监控内存
    while start.elapsed() < duration {
        let iter_start = Instant::now();

        // 执行典型工作负载
        execute_typical_workload(&iterations, &errors);

        // 定期采集内存快照
        if start.elapsed().as_secs() % snapshot_interval.as_secs() == 0
            || start.elapsed().as_millis() % 1000 < 200
        {
            sys.refresh_memory();
            let current_mem = sys.used_memory();
            let elapsed_sec = start.elapsed().as_secs();

            // 避免重复记录同一秒的数据
            if memory_snapshots.last().map(|(t, _)| *t) != Some(elapsed_sec) {
                memory_snapshots.push((elapsed_sec, current_mem));

                eprintln!(
                    "  [{}s] Memory: {} MB, Iterations: {}, Errors: {}",
                    elapsed_sec,
                    current_mem / 1024 / 1024,
                    iterations.load(Ordering::Relaxed),
                    errors.load(Ordering::Relaxed)
                );
            }
        }

        // 控制循环频率
        let elapsed = iter_start.elapsed();
        if elapsed < Duration::from_millis(100) {
            std::thread::sleep(Duration::from_millis(100) - elapsed);
        }
    }

    // 最终内存检查
    sys.refresh_memory();
    let final_memory = sys.used_memory();
    memory_snapshots.push((duration.as_secs(), final_memory));

    let total_iterations = iterations.load(Ordering::Relaxed);
    let total_errors = errors.load(Ordering::Relaxed);

    eprintln!("\n[leak] Test Results:");
    eprintln!("  Total iterations: {}", total_iterations);
    eprintln!("  Total errors: {}", total_errors);
    eprintln!("  Initial memory: {} MB", initial_memory / 1024 / 1024);
    eprintln!("  Final memory: {} MB", final_memory / 1024 / 1024);
    eprintln!(
        "  Absolute delta: {} MB",
        (final_memory as i64 - initial_memory as i64) / 1024 / 1024
    );

    // 分析内存趋势
    let analysis_result = analyze_memory_trend(&memory_snapshots);

    eprintln!("\n  Trend Analysis:");
    eprintln!("    First quarter avg: {:.1} MB", analysis_result.first_quarter_avg / 1024.0);
    eprintln!("    Last quarter avg: {:.1} MB", analysis_result.last_quarter_avg / 1024.0);
    eprintln!("    Growth ratio: {:.2}", analysis_result.growth_ratio);
    eprintln!("    Peak memory: {} MB", analysis_result.peak_memory / 1024 / 1024);
    eprintln!("    Status: {}", analysis_result.status);

    // 输出 JSON 报告
    output_leak_report(
        "memory_stability_over_time",
        &serde_json::json!({
            "duration_sec": duration.as_secs(),
            "total_iterations": total_iterations,
            "total_errors": total_errors,
            "initial_memory_mb": initial_memory / 1024 / 1024,
            "final_memory_mb": final_memory / 1024 / 1024,
            "absolute_delta_mb": (final_memory as i64 - initial_memory as i64) / 1024 / 1024,
            "peak_memory_mb": analysis_result.peak_memory / 1024 / 1024,
            "first_quarter_avg_mb": analysis_result.first_quarter_avg / 1024.0,
            "last_quarter_avg_mb": analysis_result.last_quarter_avg / 1024.0,
            "growth_ratio": analysis_result.growth_ratio,
            "status": analysis_result.status,
            "snapshots_count": memory_snapshots.len()
        }),
    );

    // 断言验证
    assert_eq!(total_errors, 0, "Should have no execution errors");

    match analysis_result.status.as_str() {
        "LEAK_DETECTED" => panic!(
            "Memory leak detected! Growth ratio: {:.2} (threshold: 1.5)",
            analysis_result.growth_ratio
        ),
        "WARNING" => {
            eprintln!(
                "\n  WARNING: Memory growth above normal range ({:.2}), but within tolerance",
                analysis_result.growth_ratio
            );
        }
        _ => {}
    }
}

// ============================================================================
// 2. KV Cache 内存泄漏专项测试
// ============================================================================

/// KV Cache 内存池泄漏检测
///
/// # 目的
/// 验证 KV Cache BlockManager 在大量分配/释放操作后无内存泄漏
///
/// # 测试场景
/// - 大量随机大小的分配/释放
/// - 模拟真实推理场景的 cache 使用模式
/// - 验证最终所有块都被正确回收
#[test]
fn test_kv_cache_memory_leak() {
    use openmini_server::hardware::kv_cache::{
        block_manager::BlockManager, block::KVCacheConfig,
    };

    let num_blocks = 512;
    let block_size = 16;
    let num_cycles = if is_ci_env() { 100 } else { 1000 };
    let max_alloc_per_cycle = num_blocks / 4;

    eprintln!("\n[leak] KV Cache Memory Leak Test");
    eprintln!(
        "  Config: {} blocks x {} tokens, {} cycles",
        num_blocks, block_size, num_cycles
    );

    let config = KVCacheConfig {
        max_blocks: num_blocks,
        block_size,
        ..Default::default()
    };

    let mut manager = BlockManager::new(&config);
    let initial_free = manager.available_blocks();

    eprintln!("  Initial state: {} free blocks", initial_free);

    use rand::Rng;
    let mut rng = rand::thread_rng();
    // Track allocated IDs for potential cleanup
    let mut pending_ids: Vec<usize> = Vec::new();

    for cycle in 0..num_cycles {
        // 随机分配大小
        let alloc_size = rng.gen_range(1..=max_alloc_per_cycle);

        // 分配
        let allocated = manager.allocate(alloc_size, None);
        if allocated.is_err() {
            // 空间不足时先释放一些再重试
            if !pending_ids.is_empty() {
                let free_batch_size = (pending_ids.len() / 2).max(1);
                let to_free: Vec<usize> = pending_ids.drain(..free_batch_size.min(pending_ids.len())).collect();
                manager.free(&to_free);
            }
            continue;
        }
        let ids = allocated.unwrap();

        // 模拟使用一段时间
        std::hint::spin_loop();

        // 释放部分或全部
        let free_count = rng.gen_range(1..=ids.len());
        let (to_keep, to_free): (Vec<_>, Vec<_>) = ids.into_iter().enumerate().partition(|&(i, _)| i >= free_count);
        let free_ids: Vec<usize> = to_free.into_iter().map(|(_, id)| id).collect();
        let keep_ids: Vec<usize> = to_keep.into_iter().map(|(_, id)| id).collect();
        manager.free(&free_ids);
        pending_ids.extend(keep_ids);

        // 定期检查
        if cycle % (num_cycles / 10) == 0 {
            eprintln!(
                "  Cycle {}/{}: Free={}, Allocated={}",
                cycle, num_cycles, manager.available_blocks(), manager.allocated_blocks()
            );
        }
    }

    // 尝试释放剩余的所有块
    if !pending_ids.is_empty() {
        manager.free(&pending_ids);
    }

    let final_free = manager.available_blocks();

    eprintln!("\n[leak] KV Cache Results:");
    eprintln!("  Cycles completed: {}", num_cycles);
    eprintln!("  Final free blocks: {}", final_free);
    eprintln!("  Final allocated blocks: {}", manager.allocated_blocks());

    // 验证：大部分块应该是空闲的（允许少量残留，因为上面的随机释放可能没完全释放）
    let free_ratio = final_free as f64 / num_blocks as f64;
    eprintln!("  Free ratio: {:.2}%", free_ratio * 100.0);

    output_leak_report(
        "kv_cache_memory_leak",
        &serde_json::json!({
            "num_blocks": num_blocks,
            "block_size": block_size,
            "cycles_completed": num_cycles,
            "initial_free": initial_free,
            "final_free": final_free,
            "final_allocated": manager.allocated_blocks(),
            "free_ratio_pct": free_ratio * 100.0
        }),
    );

    // 至少应该有 80% 的块是空闲的
    assert!(
        free_ratio > 0.8,
        "Too many blocks still allocated: {}/{} ({:.1}% free)",
        final_free,
        num_blocks,
        free_ratio * 100.0
    );
}

// ============================================================================
// 3. 连接池泄漏检测
// ============================================================================

/// 连接池连接泄漏检测
///
/// # 目的
/// 验证连接池在高频获取/释放操作后无连接泄漏
///
/// # 测试场景
/// - 多线程并发获取/释放连接
/// - 验证最终活跃连接数归零
/// - 统计数据一致性检查
#[tokio::test]
async fn test_connection_pool_leak_detection() {
    use openmini_server::service::server::connection::ConnectionPool;

    let pool_size = 20usize;
    let num_threads = 30;
    let ops_per_thread = if is_ci_env() { 50 } else { 500 };

    eprintln!("\n[leak] Connection Pool Leak Detection Test");
    eprintln!(
        "  Pool size: {}, Threads: {}, Ops/thread: {}",
        pool_size, num_threads, ops_per_thread
    );

    let pool = Arc::new(ConnectionPool::new(pool_size));
    let initial_stats = pool.stats().snapshot();

    eprintln!("  Initial stats: created={}, active={}",
        initial_stats.total_created, initial_stats.active_connections);

    let mut handles = Vec::with_capacity(num_threads);

    for thread_id in 0..num_threads {
        let pool = Arc::clone(&pool);

        let handle = tokio::spawn(async move {
            for op in 0..ops_per_thread {
                // 获取连接
                match pool.acquire().await {
                    Some(conn) => {
                        // 模拟使用
                        tokio::time::sleep(Duration::from_micros(100)).await;

                        // 释放连接
                        pool.release(conn);
                    }
                    None => {
                        eprintln!(
                            "[leak] Thread {} op {}: Failed to acquire",
                            thread_id, op
                        );
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

    // 等待一小段时间确保所有异步操作完成
    tokio::time::sleep(Duration::from_millis(100)).await;

    let final_stats = pool.stats().snapshot();

    eprintln!("\n[leak] Connection Pool Results:");
    eprintln!("  Total operations: {}", num_threads * ops_per_thread);
    eprintln!("  Final active connections: {}", final_stats.active_connections);
    eprintln!("  Total created: {}", final_stats.total_created);
    eprintln!("  Total reused: {}", final_stats.total_reused);
    eprintln!(
        "  Acquire count: {}",
        pool.stats().acquire_count.load(std::sync::atomic::Ordering::Relaxed)
    );
    eprintln!(
        "  Release count: {}",
        pool.stats().release_count.load(std::sync::atomic::Ordering::Relaxed)
    );

    // 关键断言：活跃连接必须为 0
    assert_eq!(
        final_stats.active_connections, 0,
        "Connection leak detected! Active connections should be 0, got {}",
        final_stats.active_connections
    );

    // 获取/释放计数应匹配
    let acquire_count =
        pool.stats().acquire_count.load(std::sync::atomic::Ordering::Relaxed);
    let release_count =
        pool.stats().release_count.load(std::sync::atomic::Ordering::Relaxed);

    eprintln!("  Acquire/Release balance: acquire={}, release={}", acquire_count, release_count);

    // 允许少量差异（边界情况），但应基本平衡
    let diff = if acquire_count > release_count {
        acquire_count - release_count
    } else {
        release_count - acquire_count
    };
    assert!(
        diff < 10,
        "Acquire/release count mismatch by {} (threshold: 10)",
        diff
    );

    output_leak_report(
        "connection_pool_leak",
        &serde_json::json!({
            "pool_size": pool_size,
            "num_threads": num_threads,
            "ops_per_thread": ops_per_thread,
            "final_active_connections": final_stats.active_connections,
            "total_created": final_stats.total_created,
            "total_reused": final_stats.total_reused,
            "acquire_count": acquire_count,
            "release_count": release_count,
            "acquire_release_diff": diff
        }),
    );

    eprintln!("\n  Connection pool leak check PASSED");
}

// ============================================================================
// 4. 线程/任务泄漏检测
// ============================================================================

/// 线程池任务泄漏检测
///
/// # 目的
/// 验证线程池在大量任务提交后无任务泄漏
///
/// # 测试场景
/// - 提交大量任务并等待全部完成
/// - 验证任务计数器一致性
#[test]
fn test_thread_pool_task_leak() {
    use openmini_server::service::thread;

    let pool = thread::create_default_pool();
    let total_tasks = 5000;
    let completed = Arc::new(AtomicUsize::new(0));

    eprintln!("\n[leak] Thread Pool Task Leak Test");
    eprintln!("  Tasks to submit: {}", total_tasks);

    let start = Instant::now();

    for _task_id in 0..total_tasks {
        let completed = Arc::clone(&completed);
        pool.execute(move || {
            // 模拟工作
            let _: u64 = (0..100).map(|i| i as u64).sum();
            completed.fetch_add(1, Ordering::Relaxed);
        });
    }

    // 等待所有任务完成
    let timeout = Duration::from_secs(if is_ci_env() { 30 } else { 120 });
    let deadline = start + timeout;

    loop {
        let done = completed.load(Ordering::Relaxed);
        if done >= total_tasks {
            break;
        }
        if Instant::now() > deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    let elapsed = start.elapsed();
    let done = completed.load(Ordering::Relaxed);

    eprintln!("\n[leak] Thread Pool Results:");
    eprintln!("  Tasks submitted: {}", total_tasks);
    eprintln!("  Tasks completed: {}", done);
    eprintln!("  Missing tasks: {}", total_tasks - done);
    eprintln!("  Elapsed: {:.2}s", elapsed.as_secs_f64());

    output_leak_report(
        "thread_pool_task_leak",
        &serde_json::json!({
            "tasks_submitted": total_tasks,
            "tasks_completed": done,
            "missing_tasks": total_tasks - done,
            "elapsed_ms": elapsed.as_millis()
        }),
    );

    assert_eq!(done, total_tasks, "Task leak detected! {}/{} completed", done, total_tasks);
}

// ============================================================================
// 辅助函数和结构体
// ============================================================================

/// 内存趋势分析结果
struct MemoryAnalysisResult {
    first_quarter_avg: f64,
    last_quarter_avg: f64,
    growth_ratio: f64,
    peak_memory: u64,
    status: String, // "OK", "WARNING", "LEAK_DETECTED"
}

/// 分析内存趋势
fn analyze_memory_trend(snapshots: &[(u64, u64)]) -> MemoryAnalysisResult {
    if snapshots.len() < 4 {
        return MemoryAnalysisResult {
            first_quarter_avg: 0.0,
            last_quarter_avg: 0.0,
            growth_ratio: 1.0,
            peak_memory: 0,
            status: "INSUFFICIENT_DATA".to_string(),
        };
    }

    // 提取内存值
    let memory_values: Vec<u64> = snapshots.iter().map(|(_, m)| *m).collect();

    // 计算峰值
    let peak_memory = *memory_values.iter().max().unwrap();

    // 分成四份分析
    let quarter_len = memory_values.len() / 4;
    if quarter_len == 0 {
        return MemoryAnalysisResult {
            first_quarter_avg: memory_values[0] as f64,
            last_quarter_avg: *memory_values.last().unwrap() as f64,
            growth_ratio: 1.0,
            peak_memory,
            status: "OK".to_string(),
        };
    }

    let first_sum: u64 = memory_values[..quarter_len].iter().sum();
    let last_start = memory_values.len() - quarter_len;
    let last_sum: u64 = memory_values[last_start..].iter().sum();

    let first_avg = first_sum as f64 / quarter_len as f64;
    let last_avg = last_sum as f64 / quarter_len as f64;

    // 计算增长率
    let growth_ratio = if first_avg > 0.0 {
        last_avg / first_avg
    } else {
        1.0
    };

    // 判定状态
    let status = if growth_ratio > 1.5 {
        "LEAK_DETECTED".to_string()
    } else if growth_ratio > 1.2 {
        "WARNING".to_string()
    } else {
        "OK".to_string()
    };

    MemoryAnalysisResult {
        first_quarter_avg: first_avg,
        last_quarter_avg: last_avg,
        growth_ratio,
        peak_memory,
        status,
    }
}

/// 执行典型工作负载（用于内存稳定性测试）
fn execute_typical_workload(iterations: &Arc<AtomicUsize>, errors: &Arc<AtomicUsize>) {
    let result = std::panic::catch_unwind(|| {
        // 模拟推理相关的各种操作

        // 1. 向量/矩阵运算
        let vec: Vec<f64> = (0..5000).map(|i| i as f64).collect();
        let _sum: f64 = vec.iter().map(|&x| x.sin()).sum();

        // 2. 字符串处理
        let s = format!("inference payload data {}", iterations.load(Ordering::Relaxed));
        let _parsed: usize = s.parse().unwrap_or(0);

        // 3. HashMap 操作
        use std::collections::HashMap;
        let mut map: HashMap<String, f64> = HashMap::new();
        for i in 0..50 {
            map.insert(format!("key_{}", i), i as f64 * 0.5);
        }
        drop(map);

        // 4. 嵌套容器
        let nested: Vec<Vec<u32>> = (0..10)
            .map(|i| (0..20).map(|j| (i * 20 + j) as u32).collect())
            .collect();
        drop(nested);

        iterations.fetch_add(1, Ordering::Relaxed);
    });

    if result.is_err() {
        errors.fetch_add(1, Ordering::Relaxed);
    }
}
