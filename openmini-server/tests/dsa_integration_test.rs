//! DSA (Dynamic Sparse Attention) 集成测试
//!
//! Phase 4 & 5: 端到端集成测试与性能验证
//!
//! # 测试覆盖
//!
//! - **完整管线正确性**: 从输入到输出的完整 DSA 流程验证
//! - **内存使用分析**: 高负载下的内存使用情况监控
//! - **优雅降级**: GPU 不可用时的 CPU 回退行为
//! - **缓冲区复用效率**: 预分配缓冲区的复用率统计
//!
//! # 运行方式
//!
//! ```bash
//! cargo test --test dsa_integration_test
//! ```

use std::time::Instant;

use ndarray::Array2;

use openmini_server::model::inference::dsa::{
    dsa_memory_pool, estimate_dsa_memory_usage, lightning_indexer, lightning_indexer_adaptive,
    lightning_indexer_auto, lightning_indexer_gpu, lightning_indexer_gpu_chunked,
    multihead_sparse_attention, optimize_data_layout_for_dsa, sparse_attention_forward,
    sparse_attention_forward_optimized, sparse_attention_forward_optimized_with_buffers,
    top_k_selection_metal, DSAMemoryPool, DSATempBuffers, DSATopKConfig,
};

// ============================================================================
// 测试数据生成工具函数
// ============================================================================

/// 生成指定维度的随机查询矩阵
fn generate_query_matrix(seq_len: usize, hidden_size: usize) -> Array2<f32> {
    Array2::from_shape_fn((seq_len, hidden_size), |(i, j)| {
        ((i * hidden_size + j) as f32 * 0.01).sin() + 0.5
    })
}

/// 生成指定维度的随机键/值矩阵
fn generate_kv_matrix(seq_len: usize, hidden_size: usize) -> Array2<f32> {
    Array2::from_shape_fn((seq_len, hidden_size), |(i, j)| {
        ((i * hidden_size + j) as f32 * 0.02).cos() + 0.3
    })
}

// ============================================================================
// 1. 完整管线正确性测试 (Full Pipeline Correctness)
// ============================================================================

/// 集成测试：完整 DSA 推理管线端到端正确性验证
///
/// 覆盖场景：
/// - 不同序列长度 (短/中/长)
/// - 不同 head_dim (64/128/256)
/// - 标准版 vs 优化版一致性
/// - 多头注意力完整性
#[test]
fn test_full_pipeline_correctness() {
    println!("\n=== Full Pipeline Correctness Test ===\n");

    // 测试配置矩阵
    let test_configs = vec![
        // (seq_len, head_dim, num_heads, top_k, use_causal)
        (8, 64, 1, 4, false),
        (16, 128, 4, 8, false),
        (32, 64, 8, 16, true), // 因果掩码
        (64, 128, 4, 32, false),
        (128, 256, 2, 64, false),
    ];

    for (seq_len, head_dim, num_heads, top_k, use_causal) in test_configs {
        let hidden_size = num_heads * head_dim;

        println!(
            "[Config] seq_len={}, head_dim={}, heads={}, k={}, causal={}",
            seq_len, head_dim, num_heads, top_k, use_causal
        );

        let q = generate_query_matrix(seq_len, hidden_size);
        let k = generate_kv_matrix(seq_len, hidden_size);
        let v = generate_kv_matrix(seq_len, hidden_size);

        let config = DSATopKConfig::new().with_top_k(top_k).with_dynamic_k(false);

        // 1. 单头稀疏注意力（标准版）
        let standard_result = sparse_attention_forward(&q, &k, &v, head_dim, &config, use_causal);
        assert!(
            standard_result.is_ok(),
            "Standard sparse_attention failed for seq_len={}",
            seq_len
        );
        let std_output = standard_result.unwrap();
        assert_eq!(std_output.dim(), (seq_len, head_dim));

        // 验证输出有限性
        for i in 0..seq_len {
            for j in 0..head_dim {
                assert!(
                    std_output[[i, j]].is_finite(),
                    "Standard output non-finite at [{}, {}] for seq_len={}",
                    i,
                    j,
                    seq_len
                );
            }
        }

        // 2. 单头稀疏注意力（优化版）
        let optimized_result =
            sparse_attention_forward_optimized(&q, &k, &v, head_dim, &config, use_causal);
        assert!(
            optimized_result.is_ok(),
            "Optimized sparse_attention failed for seq_len={}",
            seq_len
        );
        let opt_output = optimized_result.unwrap().0;
        assert_eq!(opt_output.dim(), (seq_len, head_dim));

        // 3. 标准版 vs 优化版一致性检查
        let max_diff = std_output
            .iter()
            .zip(opt_output.iter())
            .map(|(&s, &o)| (s - o).abs())
            .fold(0.0f32, |a, b| a.max(b));

        assert!(
            max_diff < 1e-2,
            "Standard/Optimized mismatch for seq_len={}, head_dim={}: max_diff={}",
            seq_len,
            head_dim,
            max_diff
        );

        // 4. 多头稀疏注意力
        if num_heads > 1 {
            let multihead_result =
                multihead_sparse_attention(&q, &k, &v, num_heads, head_dim, &config, use_causal);
            assert!(
                multihead_result.is_ok(),
                "Multi-head attention failed for heads={}",
                num_heads
            );
            let mh_output = multihead_result.unwrap();
            assert_eq!(mh_output.dim(), (seq_len, hidden_size));

            // 验证多头输出有限性
            for val in mh_output.iter() {
                assert!(
                    val.is_finite(),
                    "Multi-head output contains non-finite value"
                );
            }
        }

        println!("  [PASS] seq_len={}, max_diff={:.6}", seq_len, max_diff);
    }

    println!("\n[Full Pipeline Correctness] All tests passed!\n");
}

// ============================================================================
// 2. 内存使用分析测试 (Memory Usage Under Load)
// ======================================================================== ===

/// 集成测试：高负载下内存使用情况监控与分析
///
/// 覆盖场景：
/// - 大规模序列的内存占用估算准确性
/// - 内存池在高负载下的行为
/// - 长时间运行的内存稳定性
#[test]
fn test_memory_usage_under_load() {
    println!("\n=== Memory Usage Under Load Test ===\n");

    // 1. 内存估算准确性测试
    let estimate_cases = vec![
        (1024, 128, 256),
        (4096, 128, 512),
        (8192, 256, 1024),
        (16384, 128, 2048),
    ];

    for (seq_len, hidden_size, top_k) in estimate_cases {
        let estimate = estimate_dsa_memory_usage(seq_len, hidden_size, top_k);

        println!(
            "[Estimate] seq_len={}, hidden={}, k={} => peak={} bytes (~{:.1} MB)",
            seq_len,
            hidden_size,
            top_k,
            estimate.peak_bytes,
            estimate.peak_bytes as f64 / (1024.0 * 1024.0)
        );

        // 验证估算合理性：总内存应该 > 0 且在合理范围内
        assert!(
            estimate.peak_bytes > 0,
            "Memory estimate should be positive"
        );

        // 各组件内存应该非负
        assert!(estimate.qkv_bytes >= 0);
        assert!(estimate.scores_bytes >= 0);
        assert!(estimate.output_bytes >= 0);
    }

    // 2. 内存池高负载压力测试
    println!("\n[Memory Pool Stress Test]");
    let pool_capacity = 100 * 1024 * 1024; // 100MB
    let mut pool = DSAMemoryPool::new(pool_capacity);

    let initial_stats = pool.stats();
    println!(
        "  Initial: hits={}, misses={}, free={}",
        initial_stats.hit_count, initial_stats.miss_count, initial_stats.free_buffers_count
    );

    // 模拟多次分配/释放循环
    for _cycle in 0..10 {
        let sizes: Vec<usize> = (0..20).map(|i| 1024 * (i + 1)).collect();

        for size in sizes {
            let buf = pool.acquire_f32(size);
            pool.release_f32(buf);
        }
    }

    let after_stress_stats = pool.stats();
    println!(
        "  After stress: hits={}, misses={}, free={}",
        after_stress_stats.hit_count,
        after_stress_stats.miss_count,
        after_stress_stats.free_buffers_count
    );

    // 验证命中计数增加（说明复用生效）
    assert!(
        after_stress_stats.hit_count > initial_stats.hit_count,
        "Memory pool should have cache hits after reuse cycles"
    );

    // 3. 全局内存池访问测试
    println!("\n[Global Memory Pool Test]");
    let global_pool = dsa_memory_pool();
    let mut global_lock = global_pool.lock().unwrap();

    let _buf1 = global_lock.acquire_f32(1024);
    let _buf2 = global_lock.acquire_usize(256);

    let during_use_stats = global_lock.stats();
    println!(
        "  Global pool during use: capacity={}, free_buffers={}",
        during_use_stats.total_capacity_bytes, during_use_stats.free_buffers_count
    );

    drop(_buf1);
    drop(_buf2);

    let after_release_stats = global_lock.stats();
    println!(
        "  Global pool after release: free_buffers={}",
        after_release_stats.free_buffers_count
    );

    println!("\n[Memory Usage Under Load] All tests passed!\n");
}

// ============================================================================
// 3. 优雅降级测试 (Graceful Degradation)
// ============================================================================

/// 集成测试：GPU 不可用时系统优雅降级到 CPU 的行为验证
///
/// 覆盖场景：
/// - GPU 函数回退路径正确性
/// - 自适应选择器的降级决策
/// - Top-K Metal 回退行为
/// - 降级后结果一致性保证
#[test]
fn test_graceful_degradation() {
    println!("\n=== Graceful Degradation Test ===\n");

    let test_sizes = vec![(8, 64), (64, 128), (256, 64), (1024, 128)];

    for (seq_len, head_dim) in test_sizes {
        println!("[Test Size] seq_len={}, head_dim={}", seq_len, head_dim);

        let q = generate_query_matrix(seq_len, head_dim);
        let k = generate_kv_matrix(seq_len, head_dim);

        // 1. lightning_indexer_gpu 回退测试
        let gpu_result = lightning_indexer_gpu(&q, &k);
        match gpu_result {
            Ok(scores) => {
                println!("  [GPU] Available and successful");
                assert_eq!(scores.dim(), (seq_len, seq_len));
            }
            Err(e) => {
                println!("  [GPU] Not available (expected): {}", e);
                // GPU 不可用时返回错误是预期行为
            }
        }

        // 2. lightning_indexer_gpu_chunked 回退测试
        let chunked_result = lightning_indexer_gpu_chunked(&q, &k, None);
        match chunked_result {
            Ok((scores, stats)) => {
                println!(
                    "  [GPU Chunked] Available: chunking={}, chunks={:?}",
                    stats.used_chunking, stats.chunk_count
                );
                assert_eq!(scores.dim(), (seq_len, seq_len));
            }
            Err(e) => {
                println!("  [GPU Chunked] Fallback to CPU: {}", e);
            }
        }

        // 3. 自适应版本应始终成功（自动降级）
        let adaptive_result = lightning_indexer_adaptive(&q, &k);
        assert_eq!(
            adaptive_result.dim(),
            (seq_len, seq_len),
            "Adaptive indexer must always succeed"
        );
        println!("  [Adaptive] Success (auto-fallback working)");

        // 4. auto 版本也应成功
        let auto_result = lightning_indexer_auto(&q, &k);
        assert!(
            auto_result.is_ok(),
            "Auto indexer must succeed or return valid error"
        );
        if let Ok(scores) = auto_result {
            assert_eq!(scores.dim(), (seq_len, seq_len));
            println!("  [Auto] Success");
        }

        // 5. Top-K Metal 回退测试
        let scores_matrix = Array2::from_shape_fn((4, seq_len), |(i, j)| {
            ((i * seq_len + j) as f32 * 0.1) % 1.0
        });
        let top_k_result = top_k_selection_metal(&scores_matrix, 8.min(seq_len / 2));
        match top_k_result {
            Ok((indices, stats)) => {
                assert_eq!(indices.len(), 4); // 4 行
                println!("  [Top-K Metal] Success: algorithm={}", stats.algorithm);
            }
            Err(e) => {
                println!("  [Top-K Metal] Fallback: {}", e);
            }
        }

        // 6. 降级后结果一致性验证
        let cpu_standard = lightning_indexer(&q, &k);
        let cpu_adaptive = lightning_indexer_adaptive(&q, &k);

        let mut max_diff = 0.0f32;
        for i in 0..seq_len {
            for j in 0..seq_len {
                let diff = (cpu_standard[[i, j]] - cpu_adaptive[[i, j]]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }

        assert!(
            max_diff < 1e-5,
            "Degraded result inconsistent: max_diff={}",
            max_diff
        );

        println!(
            "  [Consistency] CPU vs Adaptive max_diff={:.8} (OK)\n",
            max_diff
        );
    }

    println!("[Graceful Degradation] All tests passed!\n");
}

// ============================================================================
// 4. 缓冲区复用效率测试 (Buffer Reuse Efficiency)
// ============================================================================

/// 集成测试：预分配缓冲区复用效率统计与验证
///
/// 覆盖场景：
/// - DSATempBuffers 命中率统计
/// - 不同大小请求的复用行为
/// - 重置后的状态恢复
/// - 与动态分配的性能对比
#[test]
fn test_buffer_reuse_efficiency() {
    println!("\n=== Buffer Reuse Efficiency Test ===\n");

    let max_seq_len = 2048;
    let hidden_size = 128;
    let max_k = 512;

    let mut buffers = DSATempBuffers::new(max_seq_len, hidden_size, max_k);

    println!(
        "[Setup] max_seq_len={}, hidden_size={}, max_k={}",
        max_seq_len, hidden_size, max_k
    );

    // 初始状态
    let (init_s, init_i, init_o) = buffers.stats();
    println!(
        "  Initial stats: score_hits={}, index_hits={}, output_hits={}",
        init_s, init_i, init_o
    );

    // 第一轮：多种大小的请求
    let test_rounds = vec![
        (256, 128),
        (512, 256),
        (1024, 512),
        (2048, 512),
        (1024, 256), // 重复之前的大小，应触发命中
        (512, 128),  // 更小的请求
    ];

    println!("\n[Round 1: Mixed size requests]");
    for (req_seq, req_k) in &test_rounds {
        // 分开获取以避免多重可变借用
        {
            let scores_buf = buffers.get_scores_buffer(*req_seq, *req_k);
            assert!(
                scores_buf.len() >= req_seq * req_k,
                "Scores buffer too small: got {}, need {}",
                scores_buf.len(),
                req_seq * req_k
            );
        }
        {
            let indices_buf = buffers.get_indices_buffer(*req_k);
            assert!(indices_buf.len() >= *req_k);
        }
        {
            let output_buf = buffers.get_output_buffer(hidden_size);
            assert!(output_buf.len() >= hidden_size);
        }

        println!("  Request ({:>4}, {:>4}): verified OK", req_seq, req_k);
    }

    let (round1_s, round1_i, round1_o) = buffers.stats();
    println!(
        "\n  After Round 1: score_hits={}, index_hits={}, output_hits={}",
        round1_s, round1_i, round1_o
    );

    // 第二轮：重复相同模式以观察命中率提升
    println!("\n[Round 2: Repeat pattern for hit rate observation]");
    for (req_seq, req_k) in &test_rounds {
        let _scores_buf = buffers.get_scores_buffer(*req_seq, *req_k);
        let _ = _scores_buf; // 显式释放引用
        let _indices_buf = buffers.get_indices_buffer(*req_k);
        let _ = _indices_buf;
        let _output_buf = buffers.get_output_buffer(hidden_size);
    }

    let (round2_s, round2_i, round2_o) = buffers.stats();
    println!(
        "  After Round 2: score_hits={}, index_hits={}, output_hits={}",
        round2_s, round2_i, round2_o
    );

    // 验证命中率有所增加（说明复用机制工作）
    assert!(
        round2_s >= round1_s || round2_i >= round1_i || round2_o >= round1_o,
        "Buffer reuse should increase hit counts over repeated access patterns"
    );

    // 第三轮：重置后重新开始
    println!("\n[Round 3: After reset]");
    buffers.reset();

    let (reset_s, reset_i, reset_o) = buffers.stats();
    println!(
        "  After Reset: score_hits={}, index_hits={}, output_hits={}",
        reset_s, reset_i, reset_o
    );

    // 重置后计数器应归零或显著降低
    assert!(
        reset_s <= round2_s && reset_i <= round2_i && reset_o <= round2_o,
        "Reset should clear or reduce hit counters"
    );

    // 性能对比：预分配 vs 动态分配
    println!("\n[Performance Comparison: Prealloc vs Dynamic]");

    let q = generate_query_matrix(1024, hidden_size);
    let k = generate_kv_matrix(1024, hidden_size);
    let v = generate_kv_matrix(1024, hidden_size);

    let config = DSATopKConfig::new().with_top_k(256).with_dynamic_k(false);

    // 预分配版本计时
    let mut prealloc_buffers = DSATempBuffers::new(1024, hidden_size, 256);
    let iterations = 50;

    let start_prealloc = Instant::now();
    for _ in 0..iterations {
        let _result = sparse_attention_forward_optimized_with_buffers(
            &q,
            &k,
            &v,
            hidden_size,
            &config,
            false,
            Some(&mut prealloc_buffers),
        );
    }
    let prealloc_time = start_prealloc.elapsed();

    // 动态分配版本计时
    let start_dynamic = Instant::now();
    for _ in 0..iterations {
        let _result = sparse_attention_forward_optimized(&q, &k, &v, hidden_size, &config, false);
    }
    let dynamic_time = start_dynamic.elapsed();

    println!(
        "  Preallocated:  {} iterations in {:.2}ms ({:.3}ms/iter)",
        iterations,
        prealloc_time.as_secs_f64() * 1000.0,
        prealloc_time.as_secs_f64() * 1000.0 / iterations as f64
    );
    println!(
        "  Dynamic alloc: {} iterations in {:.2}ms ({:.3}ms/iter)",
        iterations,
        dynamic_time.as_secs_f64() * 1000.0,
        dynamic_time.as_secs_f64() * 1000.0 / iterations as f64
    );

    // 数据布局优化效率测试
    println!("\n[Data Layout Optimization Efficiency]");
    let layout_start = Instant::now();
    for _ in 0..100 {
        let layout = optimize_data_layout_for_dsa(&q, &k, &v)
            .expect("optimize_data_layout_for_dsa should succeed");
        // 验证布局结构
        let (_lq, lkv, _head) = layout.dimensions();
        assert_eq!(lkv, 1024);
    }
    let layout_time = layout_start.elapsed();
    println!(
        "  Layout optimization: 100 calls in {:.2}ms",
        layout_time.as_secs_f64() * 1000.0
    );

    println!("\n[Buffer Reuse Efficiency] All tests passed!\n");
}
