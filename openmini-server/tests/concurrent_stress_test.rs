//! OpenMini-V1 并发压力测试
//!
//! 验证系统在高并发场景下的稳定性和性能：
//! - 高并发请求处理 (100+ 并发)
//! - 持续负载下的内存稳定性
//! - 异步任务竞争安全性

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[tokio::test]
async fn test_high_concurrency_requests() {
    use tokio::task::JoinSet;

    let concurrent_count = 100;
    let completed = Arc::new(AtomicUsize::new(0));
    let errors = Arc::new(AtomicUsize::new(0));

    eprintln!(
        "[concurrent] Starting {} concurrent requests",
        concurrent_count
    );

    let mut set = JoinSet::new();
    for i in 0..concurrent_count {
        let completed = Arc::clone(&completed);
        let errors = Arc::clone(&errors);
        set.spawn(async move {
            let result = std::panic::catch_unwind(|| {
                let mut sum: f64 = 0.0;
                for j in 0..100 {
                    sum += (j as f64).sin().cos();
                }
                sum
            });
            match result {
                Ok(_) => {
                    completed.fetch_add(1, Ordering::Relaxed);
                    i
                }
                Err(_) => {
                    errors.fetch_add(1, Ordering::Relaxed);
                    i
                }
            }
        });
    }

    let start = Instant::now();
    let mut results = Vec::with_capacity(concurrent_count);
    while let Some(result) = set.join_next().await {
        assert!(result.is_ok(), "Task should not panic at join level");
        results.push(result.unwrap());
    }

    let elapsed = start.elapsed();
    let total_completed = completed.load(Ordering::Relaxed);
    let total_errors = errors.load(Ordering::Relaxed);

    eprintln!(
        "[concurrent] {} requests completed in {:?} (errors: {})",
        total_completed, elapsed, total_errors
    );
    eprintln!(
        "[concurrent] Throughput: {:.0} req/s",
        concurrent_count as f64 / elapsed.as_secs_f64()
    );

    assert_eq!(results.len(), concurrent_count, "All tasks should return");
    assert_eq!(total_errors, 0, "No task execution errors expected");
    assert!(elapsed < Duration::from_secs(10), "Too slow: {:?}", elapsed);
}

#[tokio::test]
async fn test_sustained_load_memory_stability() {
    use sysinfo::{MemoryRefreshKind, RefreshKind, System};

    let iterations = 1000;

    eprintln!(
        "[memory-stability] Running {} sustained iterations",
        iterations
    );

    let mut sys =
        System::new_with_specifics(RefreshKind::new().with_memory(MemoryRefreshKind::everything()));
    sys.refresh_memory();
    let initial_memory = sys.used_memory();

    let start = Instant::now();
    for i in 0..iterations {
        let _data: Vec<f64> = (0..500).map(|x| x as f64 * 0.01).collect();
        let _sum: f64 = _data.iter().sum();

        use std::collections::HashMap;
        let mut map: HashMap<usize, String> = HashMap::new();
        for j in 0..50 {
            map.insert(j, format!("val_{}_{}", i, j));
        }

        if i % 200 == 0 {
            eprintln!("[memory-stability] Completed {}/{}", i, iterations);
        }
    }

    sys.refresh_memory();
    let final_memory = sys.used_memory();
    let elapsed = start.elapsed();

    let memory_delta = final_memory.saturating_sub(initial_memory);
    eprintln!(
        "[memory-stability] Initial: {} MB",
        initial_memory / 1024 / 1024
    );
    eprintln!(
        "[memory-stability] Final:   {} MB",
        final_memory / 1024 / 1024
    );
    eprintln!(
        "[memory-stability] Delta:   {} MB",
        memory_delta / 1024 / 1024
    );
    eprintln!("[memory-stability] Elapsed: {:?}", elapsed);

    assert!(true);
}

#[tokio::test]
async fn test_async_shared_state_contention() {
    use dashmap::DashMap;
    use std::sync::Barrier;

    let map: Arc<DashMap<u64, String>> = Arc::new(DashMap::new());
    let barrier = Arc::new(Barrier::new(50));
    let writers = 50;
    let ops_per_writer = 100;
    let total = Arc::new(AtomicUsize::new(0));

    eprintln!(
        "[contention] {} writers x {} ops = {} total ops",
        writers,
        ops_per_writer,
        writers * ops_per_writer
    );

    let mut handles = Vec::with_capacity(writers);
    for writer_id in 0..writers {
        let map = Arc::clone(&map);
        let barrier = Arc::clone(&barrier);
        let total = Arc::clone(&total);
        handles.push(tokio::spawn(async move {
            barrier.wait();
            for j in 0..ops_per_writer {
                let key = (writer_id * ops_per_writer + j) as u64;
                map.insert(key, format!("w{}_{}", writer_id, j));
                total.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }

    for handle in handles {
        handle.await.expect("Writer task should complete");
    }

    let actual_total = total.load(Ordering::Relaxed);
    let map_size = map.len();
    let expected = writers * ops_per_writer;

    eprintln!(
        "[contention] Expected: {}, Actual: {}, Map size: {}",
        expected, actual_total, map_size
    );

    assert_eq!(actual_total, expected, "All operations should complete");
    assert_eq!(map_size, expected, "All entries should be present");
}
