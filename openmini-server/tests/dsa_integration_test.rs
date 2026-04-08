//! DSA (Dynamic Sparse Attention) 集成测试
//!
//! Phase 4 & 5: 端到端集成测试与性能验证

use std::time::Instant;

use ndarray::Array2;

use openmini_server::model::inference::dsa::{
    dsa_memory_pool, estimate_dsa_memory_usage, lightning_indexer, lightning_indexer_adaptive,
    lightning_indexer_auto, lightning_indexer_gpu, lightning_indexer_gpu_chunked,
    multihead_sparse_attention, optimize_data_layout_for_dsa, sparse_attention_forward,
    sparse_attention_forward_optimized, top_k_selection_metal, DSAMemoryPool, DSATopKConfig,
};

fn generate_query_matrix(seq_len: usize, hidden_size: usize) -> Array2<f32> {
    Array2::from_shape_fn((seq_len, hidden_size), |(i, j)| {
        ((i * hidden_size + j) as f32 * 0.01).sin() + 0.5
    })
}

fn generate_kv_matrix(seq_len: usize, hidden_size: usize) -> Array2<f32> {
    Array2::from_shape_fn((seq_len, hidden_size), |(i, j)| {
        ((i * hidden_size + j) as f32 * 0.02).cos() + 0.3
    })
}

#[test]
fn test_full_pipeline_correctness() {
    println!("\n=== Full Pipeline Correctness Test ===\n");

    let test_configs = vec![
        (8, 64, 1, 4, false),
        (16, 128, 1, 8, false),
        (32, 64, 1, 4, true),       // 因果掩码: 使用较小 top_k 避免全 mask
        (64, 128, 4, 32, false),      // 多头: hidden_size=512
        (128, 256, 2, 64, false),     // 多头: hidden_size=512
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

        if num_heads == 1 {
            let standard_result = sparse_attention_forward(&q, &k, &v, head_dim, &config, use_causal);
            assert!(
                standard_result.is_ok(),
                "Standard sparse_attention failed for seq_len={}",
                seq_len
            );
            let std_output = standard_result.unwrap();
            assert_eq!(std_output.dim(), (seq_len, head_dim));

            for i in 0..seq_len {
                for j in 0..head_dim {
                    assert!(
                        std_output[[i, j]].is_finite(),
                        "Standard output non-finite at [{}, {}] for seq_len={}",
                        i, j, seq_len
                    );
                }
            }

            let optimized_result =
                sparse_attention_forward_optimized(&q, &k, &v, head_dim, &config, use_causal);
            assert!(
                optimized_result.is_ok(),
                "Optimized sparse_attention failed for seq_len={}",
                seq_len
            );
            let opt_output = optimized_result.unwrap().0;
            assert_eq!(opt_output.dim(), (seq_len, head_dim));

            let max_diff = std_output
                .iter()
                .zip(opt_output.iter())
                .map(|(&s, &o)| (s - o).abs())
                .fold(0.0f32, |a, b| a.max(b));

            assert!(
                max_diff < 1e-2,
                "Standard/Optimized mismatch: max_diff={}", max_diff
            );

            println!("  [PASS] Single-head seq_len={}, max_diff={:.6}", seq_len, max_diff);
        } else {
            let multihead_result =
                multihead_sparse_attention(&q, &k, &v, num_heads, head_dim, &config, use_causal);
            assert!(
                multihead_result.is_ok(),
                "Multi-head attention failed for heads={}", num_heads
            );
            let mh_output = multihead_result.unwrap();
            assert_eq!(mh_output.dim(), (seq_len, hidden_size));

            for val in mh_output.iter() {
                assert!(val.is_finite(), "Multi-head output contains non-finite value");
            }

            println!("  [PASS] Multi-head heads={}, hidden_size={}", num_heads, hidden_size);
        }
    }

    println!("\n[Full Pipeline Correctness] All tests passed!\n");
}

#[test]
fn test_memory_usage_under_load() {
    println!("\n=== Memory Usage Under Load Test ===\n");

    let estimate_cases = vec![
        (1024, 128, 256),
        (2048, 256, 512),
        (4096, 128, 128),
        (8192, 256, 256),
    ];

    for (seq_len, head_dim, top_k) in estimate_cases {
        let estimate = estimate_dsa_memory_usage(seq_len, head_dim, top_k);

        println!(
            "[Estimate] seq_len={}, head_dim={}, k={} => {:.2} MB",
            seq_len, head_dim, top_k,
            estimate.qkv_bytes as f64 / (1024.0 * 1024.0)
        );

        assert!(estimate.qkv_bytes > 0);
        assert!(estimate.scores_bytes > 0);
        assert!(estimate.output_bytes > 0);
    }

    println!("\n[Memory Pool Stress Test]");
    let mut pool = DSAMemoryPool::new(1024 * 1024);
    let mut handles = Vec::new();

    for _ in 0..10 {
        let buf = pool.acquire_f32(100000);
        handles.push(buf);
    }

    drop(handles);

    println!("\n[Global Memory Pool Test]");
    let global_pool = dsa_memory_pool();
    let mut global_lock = global_pool.lock().unwrap();

    let _buf1 = global_lock.acquire_f32(1024);
    let _buf2 = global_lock.acquire_usize(256);

    drop(_buf1);
    drop(_buf2);

    println!("[Memory Usage Under Load] All tests passed!\n");
}

#[test]
fn test_graceful_degradation() {
    println!("\n=== Graceful Degradation Test ===\n");

    let test_sizes = vec![(8, 64), (64, 128), (256, 64), (1024, 128)];

    for (seq_len, head_dim) in test_sizes {
        println!("[Test Size] seq_len={}, head_dim={}", seq_len, head_dim);

        let q = generate_query_matrix(seq_len, head_dim);
        let k = generate_kv_matrix(seq_len, head_dim);

        let gpu_result = lightning_indexer_gpu(&q, &k);
        match gpu_result {
            Ok(scores) => {
                println!("  [GPU] Available and successful");
                assert_eq!(scores.dim(), (seq_len, seq_len));
            }
            Err(e) => {
                println!("  [GPU] Not available (expected): {}", e);
            }
        }

        let chunked_result = lightning_indexer_gpu_chunked(&q, &k, None);
        match chunked_result {
            Ok((scores, stats)) => {
                println!("  [GPU Chunked] Available: chunks={:?}", stats.chunk_count);
                assert_eq!(scores.dim(), (seq_len, seq_len));
            }
            Err(e) => {
                println!("  [GPU Chunked] Fallback to CPU: {}", e);
            }
        }

        let adaptive_result = lightning_indexer_adaptive(&q, &k);
        assert_eq!(
            adaptive_result.dim(),
            (seq_len, seq_len),
            "Adaptive indexer must always succeed"
        );
        println!("  [Adaptive] Success (auto-fallback working)");

        let auto_result = lightning_indexer_auto(&q, &k);
        match auto_result {
            Ok(scores) => {
                assert_eq!(scores.dim(), (seq_len, seq_len));
                println!("  [Auto] Success");
            }
            Err(e) => {
                if seq_len < 1024 {
                    panic!("Auto indexer should succeed for small matrices: {}", e);
                }
                println!("  [Auto] Failed (acceptable): {}", e);
            }
        }

        let scores_matrix = Array2::from_shape_fn((4, seq_len), |(i, j)| {
            ((i * seq_len + j) as f32 * 0.1) % 1.0
        });
        let top_k_result = top_k_selection_metal(&scores_matrix, 8.min(seq_len / 2));
        match top_k_result {
            Ok((indices, _stats)) => {
                assert_eq!(indices.len(), 4);
                println!("  [Top-K Metal] Success");
            }
            Err(e) => {
                println!("  [Top-K Metal] Fallback: {}", e);
            }
        }

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

        assert!(max_diff < 1e-5, "Degraded result inconsistent: max_diff={}", max_diff);

        println!("  [Consistency] CPU vs Adaptive OK\n");
    }

    println!("[Graceful Degradation] All tests passed!\n");
}
