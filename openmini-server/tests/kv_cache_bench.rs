//! KV Cache 性能基准测试
//!
//! 测试指标：
//! - 内存分配/释放性能
//! - 批处理吞吐量
//! - 前缀缓存命中率
//! - 调度器延迟

#[cfg(test)]
mod performance_tests {
    use std::time::{Duration, Instant};
    
    use openmini_server::hardware::kv_cache::{
        BatchScheduler, GenerationRequest,
        PagedKVCache, PrefixCache, PrefixHash,
    };
    use ndarray::Array2;

    fn create_test_cache() -> PagedKVCache {
        PagedKVCache::with_capacity(10000, 16)
    }

    #[test]
    fn test_allocation_performance() {
        let mut cache = create_test_cache();
        
        let start = Instant::now();
        for i in 0..1000 {
            let _ = cache.allocate_slots(i, 32);
        }
        let elapsed = start.elapsed();
        
        println!("Allocation of 1000 requests (32 tokens each): {:?}", elapsed);
        println!("Per request: {:?}", elapsed / 1000);
        
        assert!(elapsed < Duration::from_millis(100), "Allocation too slow");
    }

    #[test]
    fn test_memory_utilization() {
        let mut cache = PagedKVCache::with_capacity(100, 16);
        
        for i in 0..50 {
            cache.allocate_slots(i, 16).unwrap();
        }
        
        let utilization = cache.utilization();
        println!("Memory utilization with 50/100 blocks: {:.2}%", utilization * 100.0);
        
        assert!(utilization > 0.49 && utilization < 0.51);
    }

    #[test]
    fn test_batch_throughput() {
        let kv_cache = create_test_cache();
        let mut scheduler = BatchScheduler::with_kv_cache(kv_cache);
        
        for i in 0..100 {
            let request = GenerationRequest::new(i, vec![1, 2, 3, 4, 5], 100);
            scheduler.add_request(request);
        }
        
        let start = Instant::now();
        let mut total_scheduled = 0;
        
        for _ in 0..100 {
            let scheduled = scheduler.schedule();
            total_scheduled += scheduled.len();
        }
        
        let elapsed = start.elapsed();
        println!("Scheduled {} requests in {:?}", total_scheduled, elapsed);
        
        let throughput = total_scheduled as f64 / elapsed.as_secs_f64();
        println!("Throughput: {:.2} requests/second", throughput);
    }

    #[test]
    fn test_prefix_cache_hit_rate() {
        let mut cache = PrefixCache::default_config();
        
        for i in 0..100 {
            let tokens: Vec<u32> = (i..i+16).collect();
            let hash = PrefixHash::from_tokens(&tokens);
            let _ = cache.insert(hash, vec![i as usize], 16);
        }
        
        for i in 0..50 {
            let tokens: Vec<u32> = (i..i+16).collect();
            let hash = PrefixHash::from_tokens(&tokens);
            cache.lookup(hash);
        }
        
        for i in 100..150 {
            let tokens: Vec<u32> = (i..i+16).collect();
            let hash = PrefixHash::from_tokens(&tokens);
            cache.lookup(hash);
        }
        
        let stats = cache.stats();
        println!("Hit rate: {:.2}%", stats.hit_rate * 100.0);
        println!("Hits: {}, Misses: {}", stats.hits, stats.misses);
        
        assert!(stats.hit_rate > 0.4, "Hit rate too low");
    }

    #[test]
    fn test_kv_write_read_latency() {
        let mut cache = create_test_cache();
        cache.allocate_slots(1, 128).unwrap();
        
        let k = Array2::ones((128, 128));
        let v = Array2::ones((128, 128));
        
        let write_start = Instant::now();
        cache.write_kv(1, 0, 0, &k, &v).unwrap();
        let write_time = write_start.elapsed();
        
        let read_start = Instant::now();
        let _ = cache.read_kv(1, 0);
        let read_time = read_start.elapsed();
        
        println!("Write latency (128 tokens): {:?}", write_time);
        println!("Read latency (128 tokens): {:?}", read_time);
        
        assert!(write_time < Duration::from_secs(1));
        assert!(read_time < Duration::from_secs(1));
    }

    #[test]
    fn test_concurrent_requests() {
        let mut cache = create_test_cache();
        
        let start = Instant::now();
        
        for i in 0..100 {
            cache.allocate_slots(i, 64).unwrap();
        }
        
        let alloc_time = start.elapsed();
        
        let write_start = Instant::now();
        let k = Array2::ones((64, 128));
        let v = Array2::ones((64, 128));
        
        for i in 0..100 {
            for layer in 0..4 {
                cache.write_kv(i, layer, 0, &k, &v).unwrap();
            }
        }
        
        let write_time = write_start.elapsed();
        
        println!("Allocation time: {:?}", alloc_time);
        println!("Write time (100 requests x 4 layers): {:?}", write_time);
        println!("Active requests: {}", cache.num_active_requests());
    }

    #[test]
    fn test_fork_performance() {
        let mut cache = create_test_cache();
        
        cache.allocate_slots(1, 64).unwrap();
        
        let start = Instant::now();
        
        for i in 2..50 {
            cache.fork_request(1, i).unwrap();
        }
        
        let elapsed = start.elapsed();
        println!("Fork 49 requests from 1 source: {:?}", elapsed);
        println!("Per fork: {:?}", elapsed / 49);
    }
    
    #[test]
    fn test_block_allocation_benchmark() {
        let iterations = 100;
        let mut total_time = Duration::ZERO;
        
        for _ in 0..iterations {
            let mut cache = create_test_cache();
            
            let start = Instant::now();
            for i in 0..100 {
                let _ = cache.allocate_slots(i, 32);
            }
            total_time += start.elapsed();
        }
        
        let avg_time = total_time / iterations;
        println!("Avg allocation time (100 requests): {:?}", avg_time);
    }
    
    #[test]
    fn test_prefix_hash_benchmark() {
        let tokens: Vec<u32> = (0..128).collect();
        let iterations = 10000;
        
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = PrefixHash::from_tokens(&tokens);
        }
        let elapsed = start.elapsed();
        
        println!("Prefix hash (128 tokens) x {}: {:?}", iterations, elapsed);
        println!("Per hash: {:?}", elapsed / iterations);
    }
}

#[cfg(test)]
mod stress_tests {
    use std::time::Instant;
    
    use openmini_server::hardware::kv_cache::{
        BatchScheduler, GenerationRequest, PagedKVCache,
    };

    #[test]
    fn test_stress_allocation() {
        let mut cache = PagedKVCache::with_capacity(1000, 16);
        
        let start = Instant::now();
        let mut allocated = 0;
        
        for i in 0..10000 {
            if cache.allocate_slots(i, 16).is_ok() {
                allocated += 1;
            }
        }
        
        let elapsed = start.elapsed();
        println!("Allocated {} requests in {:?}", allocated, elapsed);
        println!("Final utilization: {:.2}%", cache.utilization() * 100.0);
    }

    #[test]
    fn test_stress_scheduler() {
        let kv_cache = PagedKVCache::with_capacity(500, 16);
        let mut scheduler = BatchScheduler::with_kv_cache(kv_cache);
        
        let start = Instant::now();
        
        for i in 0..1000 {
            let request = GenerationRequest::new(i, vec![1, 2, 3, 4, 5], 100);
            scheduler.add_request(request);
        }
        
        let mut iterations = 0;
        while scheduler.has_pending() && iterations < 10000 {
            scheduler.schedule();
            iterations += 1;
        }
        
        let elapsed = start.elapsed();
        let stats = scheduler.stats();
        
        println!("Iterations: {}", iterations);
        println!("Completed: {}", stats.completed);
        println!("Time: {:?}", elapsed);
    }
}
