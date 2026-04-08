//! DSA (Dynamic Sparse Attention) 性能基准测试
//!
//! 使用 Criterion 框架测试不同序列长度下的 CPU vs GPU 性能对比，
//! 以及 lightning_indexer 各变体的吞吐量。
//!
//! # 测试覆盖
//!
//! - **CPU vs GPU 对比**: 不同序列长度下 CPU 和 GPU 的性能差异
//! - **分块处理**: lightning_indexer_chunked 的分块效率
//! - **自适应选择**: lightning_indexer_adaptive 的策略选择正确性
//! - **批量处理**: 批量矩阵乘法的吞吐量提升
//! - **缓存优化**: 缓存友好版本的加速效果
//!
//! # 运行方式
//!
//! ```bash
//! cargo bench --bench dsa_bench
//! ```
//!
//! # 输出解读
//!
//! - `time`: 平均执行时间（越低越好）
//! - `throughput`: 吞吐量（越高越好）
//! - `slope`: 时间复杂度趋势

use std::time::Duration;

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use ndarray::Array2;

use openmini_server::model::inference::dsa::{
    lightning_indexer,
    lightning_indexer_adaptive,
    lightning_indexer_auto,
    lightning_indexer_cache_optimized,
    lightning_indexer_chunked,
    DSAMemoryPool,
    DSATempBuffers,
    lightning_indexer_gpu,
    lightning_indexer_gpu_adaptive_stats,
    lightning_indexer_gpu_chunked,
    sparse_attention_forward,
    sparse_attention_forward_optimized,
    sparse_attention_forward_optimized_with_buffers,
    DSATopKConfig,
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

/// 生成指定维度的随机键矩阵
fn generate_key_matrix(seq_len: usize, hidden_size: usize) -> Array2<f32> {
    Array2::from_shape_fn((seq_len, hidden_size), |(i, j)| {
        ((i * hidden_size + j) as f32 * 0.02).cos() + 0.3
    })
}

// ============================================================================
// 1. CPU vs GPU 性能对比基准测试
// ============================================================================

/// CPU Lightning Indexer 基准测试（不同序列长度）
fn bench_lightning_indexer_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("lightning_indexer_cpu");
    let hidden_size = 128;

    for seq_len in [256, 512, 1024, 2048, 4096].iter() {
        let q = generate_query_matrix(*seq_len, hidden_size);
        let k = generate_key_matrix(*seq_len, hidden_size);

        group.bench_with_input(
            BenchmarkId::new("cpu", seq_len),
            seq_len,
            |b, &_s| {
                b.iter(|| {
                    let result = lightning_indexer(black_box(&q), black_box(&k));
                    black_box(result)
                })
            },
        );

        // 设置合理的采样时间
        group.measurement_time(Duration::from_secs(3));
        group.throughput(Throughput::Elements(*seq_len as u64));
    }

    group.finish();
}

/// GPU Lightning Indexer 基准测试（不同序列长度）
fn bench_lightning_indexer_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("lightning_indexer_gpu");
    let hidden_size = 128;

    for seq_len in [256, 512, 1024, 2048, 4096].iter() {
        let q = generate_query_matrix(*seq_len, hidden_size);
        let k = generate_key_matrix(*seq_len, hidden_size);

        group.bench_with_input(
            BenchmarkId::new("gpu", seq_len),
            seq_len,
            |b, &_s| {
                b.iter(|| {
                    // GPU 可能不可用，捕获错误但不影响基准测试
                    let _result = lightning_indexer_gpu(black_box(&q), black_box(&k));
                })
            },
        );

        group.measurement_time(Duration::from_secs(3));
        group.throughput(Throughput::Elements(*seq_len as u64));
    }

    group.finish();
}

// ============================================================================
// 2. 分块处理基准测试
// ============================================================================

/// 分块 Lightning Indexer 基准测试
fn bench_lightning_indexer_chunked(c: &mut Criterion) {
    let mut group = c.benchmark_group("lightning_indexer_chunked");
    let hidden_size = 128;
    let seq_len = 4096; // 使用较大序列展示分块效果

    let q = generate_query_matrix(seq_len, hidden_size);
    let k = generate_key_matrix(seq_len, hidden_size);

    for chunk_size in [512, 1024, 2048, 4096].iter() {
        group.bench_with_input(
            BenchmarkId::new("chunked", chunk_size),
            chunk_size,
            |b, cs| {
                b.iter(|| {
                    let result =
                        lightning_indexer_chunked(black_box(&q), black_box(&k), black_box(*cs));
                    black_box(result)
                })
            },
        );
    }

    // 对比：不分块的标准版本
    group.bench_function("standard", |b| {
        b.iter(|| {
            let result = lightning_indexer(black_box(&q), black_box(&k));
            black_box(result)
        })
    });

    group.measurement_time(Duration::from_secs(3));
    group.finish();
}

// ============================================================================
// 3. 自适应选择器基准测试
// ============================================================================

/// 自适应 Lightning Indexer 基准测试
fn bench_lightning_indexer_adaptive(c: &mut Criterion) {
    let mut group = c.benchmark_group("lightning_indexer_adaptive");
    let hidden_size = 128;

    for seq_len in [128, 512, 1024, 2048, 8192].iter() {
        let q = generate_query_matrix(*seq_len, hidden_size);
        let k = generate_key_matrix(*seq_len, hidden_size);

        group.bench_with_input(
            BenchmarkId::new("adaptive", seq_len),
            seq_len,
            |b, &_s| {
                b.iter(|| {
                    let result = lightning_indexer_adaptive(black_box(&q), black_box(&k));
                    black_box(result)
                })
            },
        );
    }

    group.measurement_time(Duration::from_secs(3));
    group.finish();
}

/// 自动选择 Lightning Indexer 基准测试（增强版）
fn bench_lightning_indexer_auto(c: &mut Criterion) {
    let mut group = c.benchmark_group("lightning_indexer_auto");
    let hidden_size = 128;

    for seq_len in [64, 256, 1024, 4096, 16384].iter() {
        let q = generate_query_matrix(*seq_len, hidden_size);
        let k = generate_key_matrix(*seq_len, hidden_size);

        group.bench_with_input(
            BenchmarkId::new("auto", seq_len),
            seq_len,
            |b, &_s| {
                b.iter(|| {
                    let _result = lightning_indexer_auto(black_box(&q), black_box(&k));
                    // auto 返回 Result，忽略错误用于基准测试
                })
            },
        );
    }

    group.measurement_time(Duration::from_secs(3));
    group.finish();
}

// ============================================================================
// 4. 缓存优化版本基准测试
// ============================================================================

/// 缓存优化版 Lightning Indexer 基准测试
fn bench_lightning_indexer_cache_optimized(c: &mut Criterion) {
    let mut group = c.benchmark_group("lightning_indexer_cache_optimized");
    let hidden_size = 128;

    for seq_len in [256, 1024, 4096, 8192].iter() {
        let q = generate_query_matrix(*seq_len, hidden_size);
        let k = generate_key_matrix(*seq_len, hidden_size);

        group.bench_with_input(
            BenchmarkId::new("cache_opt", seq_len),
            seq_len,
            |b, &_s| {
                b.iter(|| {
                    let result =
                        lightning_indexer_cache_optimized(black_box(&q), black_box(&k));
                    black_box(result)
                })
            },
        );
    }

    group.measurement_time(Duration::from_secs(3));
    group.finish();
}

// ============================================================================
// 5. GPU 增强版基准测试（带统计信息）
// ======================================================================== ===

/// GPU Chunked 版本基准测试（带统计）
fn bench_lightning_indexer_gpu_chunked_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("lightning_indexer_gpu_chunked_with_stats");
    let hidden_size = 128;

    for seq_len in [512, 2048, 8192].iter() {
        let q = generate_query_matrix(*seq_len, hidden_size);
        let k = generate_key_matrix(*seq_len, hidden_size);

        group.bench_with_input(
            BenchmarkId::new("gpu_chunked_stats", seq_len),
            seq_len,
            |b, &_s| {
                b.iter(|| {
                    let _result =
                        lightning_indexer_gpu_chunked(black_box(&q), black_box(&k), None);
                })
            },
        );
    }

    group.measurement_time(Duration::from_secs(3));
    group.finish();
}

/// 自适应 GPU 版本基准测试（带统计）
fn bench_lightning_indexer_gpu_adaptive_stats_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("lightning_indexer_gpu_adaptive_stats");
    let hidden_size = 128;

    for seq_len in [64, 512, 2048, 16384].iter() {
        let q = generate_query_matrix(*seq_len, hidden_size);
        let k = generate_key_matrix(*seq_len, hidden_size);

        group.bench_with_input(
            BenchmarkId::new("gpu_adaptive_stats", seq_len),
            seq_len,
            |b, &_s| {
                b.iter(|| {
                    let _result =
                        lightning_indexer_gpu_adaptive_stats(black_box(&q), black_box(&k));
                })
            },
        );
    }

    group.measurement_time(Duration::from_secs(3));
    group.finish();
}

// ============================================================================
// 6. 综合对比基准测试
// ============================================================================

/// CPU 标准版 vs 缓存优化版 对比
fn bench_cpu_standard_vs_cache_optimized(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_comparison");
    let hidden_size = 128;
    let seq_len = 2048;

    let q = generate_query_matrix(seq_len, hidden_size);
    let k = generate_key_matrix(seq_len, hidden_size);

    group.bench_function("standard_cpu", |b| {
        b.iter(|| {
            let result = lightning_indexer(black_box(&q), black_box(&k));
            black_box(result)
        })
    });

    group.bench_function("cache_optimized_cpu", |b| {
        b.iter(|| {
            let result = lightning_indexer_cache_optimized(black_box(&q), black_box(&k));
            black_box(result)
        })
    });

    group.measurement_time(Duration::from_secs(5));
    group.finish();
}

// ============================================================================
// Phase 5: 性能验证基准测试 (Performance Validation Benchmarks)
// ============================================================================

// 7. 内存池 acquire/release 吞吐量基准
fn bench_memory_pool_acquire_release(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pool_throughput");

    for buffer_size in [64, 256, 1024, 4096].iter() {
        group.bench_with_input(
            BenchmarkId::new("acquire_release_f32", buffer_size),
            buffer_size,
            |b, &size| {
                let mut pool = DSAMemoryPool::new(50 * 1024 * 1024); // 50MB pool
                b.iter(|| {
                    let buf = pool.acquire_f32(black_box(size));
                    // 模拟使用：写入数据
                    black_box(&buf);
                    pool.release_f32(buf);
                })
            },
        );
    }

    // usize 缓冲区吞吐
    for buffer_size in [16, 64, 256, 1024].iter() {
        group.bench_with_input(
            BenchmarkId::new("acquire_release_usize", buffer_size),
            buffer_size,
            |b, &size| {
                let mut pool = DSAMemoryPool::new(10 * 1024 * 1024); // 10MB pool
                b.iter(|| {
                    let idx = pool.acquire_usize(black_box(size));
                    black_box(&idx);
                    pool.release_usize(idx);
                })
            },
        );
    }

    // Guarded API 吞吐
    group.bench_function("acquire_release_guarded_f32_256", |b| {
        let mut pool = DSAMemoryPool::new(10 * 1024 * 1024);
        b.iter(|| {
            let _guard = pool.acquire_f32_guarded(black_box(256));
            // guard 在此处自动 Drop 并归还缓冲区
        })
    });

    group.measurement_time(Duration::from_secs(3));
    group.finish();
}

// 8. 端到端标准版 vs 优化版对比基准
fn bench_e2e_standard_vs_optimized(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_standard_vs_optimized");

    for seq_len in [256, 1024, 4096].iter() {
        let head_dim = 128;
        let top_k = (*seq_len / 4).min(512);

        let q = generate_query_matrix(*seq_len, head_dim);
        let k = generate_key_matrix(*seq_len, head_dim);
        let v = generate_key_matrix(*seq_len, head_dim); // V 与 K 同维度

        let config = DSATopKConfig::new().with_top_k(top_k).with_dynamic_k(false);

        // 标准版 sparse_attention_forward
        group.bench_function(BenchmarkId::new("standard_e2e", seq_len), |b| {
            b.iter(|| {
                let result = sparse_attention_forward(
                    black_box(&q),
                    black_box(&k),
                    black_box(&v),
                    black_box(head_dim),
                    &config,
                    black_box(false),
                );
                black_box(result.is_ok())
            })
        });

        // 优化版 sparse_attention_forward_optimized
        group.bench_function(BenchmarkId::new("optimized_e2e", seq_len), |b| {
            b.iter(|| {
                let result = sparse_attention_forward_optimized(
                    black_box(&q),
                    black_box(&k),
                    black_box(&v),
                    black_box(head_dim),
                    &config,
                    black_box(false),
                );
                black_box(result.is_ok())
            })
        });
    }

    group.measurement_time(Duration::from_secs(5));
    group.finish();
}

// 9. 不同 head_dim 性能对比基准 (64/128/256)
fn bench_head_dim_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("head_dim_comparison");
    let seq_len = 1024;

    for &head_dim in [64usize, 128, 256].iter() {
        let q = generate_query_matrix(seq_len, head_dim);
        let k = generate_key_matrix(seq_len, head_dim);

        // Lightning Indexer 不同 head_dim
        group.bench_function(BenchmarkId::new("lightning_indexer", head_dim), |b| {
            b.iter(|| {
                let result = lightning_indexer(black_box(&q), black_box(&k));
                black_box(result)
            })
        });

        // 分块版本不同 head_dim
        group.bench_function(BenchmarkId::new("chunked_indexer", head_dim), |b| {
            b.iter(|| {
                let result =
                    lightning_indexer_chunked(black_box(&q), black_box(&k), black_box(512));
                black_box(result)
            })
        });
    }

    group.measurement_time(Duration::from_secs(3));
    group.finish();
}

// 10. 预分配 vs 动态分配性能对比基准
fn bench_prealloc_vs_dynamic(c: &mut Criterion) {
    let mut group = c.benchmark_group("prealloc_vs_dynamic");
    let seq_len = 2048;
    let hidden_size = 128;
    let max_k = 512;

    let q = generate_query_matrix(seq_len, hidden_size);
    let k = generate_key_matrix(seq_len, hidden_size);
    let v = generate_key_matrix(seq_len, hidden_size);

    // 预分配缓冲区版本
    let mut prealloc_buffers = DSATempBuffers::new(seq_len, hidden_size, max_k);

    group.bench_function("prealloc_optimized", |b| {
        b.iter(|| {
            let config = DSATopKConfig::new().with_top_k(max_k).with_dynamic_k(false);
            let _result = sparse_attention_forward_optimized_with_buffers(
                black_box(&q),
                black_box(&k),
                black_box(&v),
                black_box(hidden_size),
                &config,
                black_box(false),
                black_box(Some(&mut prealloc_buffers)),
            );
        })
    });

    // 动态分配版本（无预分配）
    group.bench_function("dynamic_optimized", |b| {
        b.iter(|| {
            let config = DSATopKConfig::new().with_top_k(max_k).with_dynamic_k(false);
            let _result = sparse_attention_forward_optimized(
                black_box(&q),
                black_box(&k),
                black_box(&v),
                black_box(hidden_size),
                &config,
                black_box(false),
            );
        })
    });

    // 标准版作为基线参考
    group.bench_function("baseline_standard", |b| {
        b.iter(|| {
            let config = DSATopKConfig::new().with_top_k(max_k).with_dynamic_k(false);
            let _result = sparse_attention_forward(
                black_box(&q),
                black_box(&k),
                black_box(&v),
                black_box(hidden_size),
                &config,
                black_box(false),
            );
        })
    });

    group.measurement_time(Duration::from_secs(5));
    group.finish();
}

// ============================================================================
// Criterion 配置和注册（更新：包含 Phase 5 新增基准）
// ============================================================================

criterion_group!(
    benches,
    // Phase 1-3 基准测试
    bench_lightning_indexer_cpu,
    bench_lightning_indexer_gpu,
    bench_lightning_indexer_chunked,
    bench_lightning_indexer_adaptive,
    bench_lightning_indexer_auto,
    bench_lightning_indexer_cache_optimized,
    bench_lightning_indexer_gpu_chunked_stats,
    bench_lightning_indexer_gpu_adaptive_stats_bench,
    bench_cpu_standard_vs_cache_optimized,
    // Phase 5 新增性能验证基准测试
    bench_memory_pool_acquire_release,
    bench_e2e_standard_vs_optimized,
    bench_head_dim_comparison,
    bench_prealloc_vs_dynamic,
);

criterion_main!(benches);
