//! 高性能推理 Pipeline 基准测试
//!
//! 对比不同注意力策略的性能：
//! - FlashAttention-3 (AMLA优化) vs Standard Attention
//! - 不同序列长度：短(64-512) / 中(1K-4K) / 长(8K-16K)
//! - 批量推理吞吐量
//!
//! # 运行方式
//!
//! ```bash
//! cargo bench --package openmini-server --bench high_perf_pipeline_bench
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array2;

use openmini_server::model::inference::high_performance_pipeline::{
    HighPerfPipelineConfig, HighPerformancePipeline,
};

// ============================================================================
// 辅助函数
// ============================================================================

fn create_test_data(
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let hidden_dim = num_heads * head_dim;

    let q = Array2::from_shape_fn((seq_len, hidden_dim), |(i, j)| {
        ((i as f32 * 0.01 + j as f32 * 0.02) % 10.0 - 5.0).sin()
    });
    let k = Array2::from_shape_fn((seq_len, hidden_dim), |(i, j)| {
        ((i as f32 * 0.015 + j as f32 * 0.01) % 8.0 - 4.0).cos()
    });
    let v = Array2::from_shape_fn((seq_len, hidden_dim), |(i, j)| {
        ((i + j) as f32 * 0.005).tanh()
    });

    (q, k, v)
}

fn create_fa3_pipeline(num_heads: usize, head_dim: usize) -> HighPerformancePipeline {
    let config = HighPerfPipelineConfig {
        num_heads,
        head_dim,
        num_kv_heads: num_heads,
        enable_fa3: true,
        enable_mla: false,
        enable_streaming: false,
        fa3_block_size: 128,
        ..Default::default()
    };

    HighPerformancePipeline::new(config).unwrap()
}

fn create_standard_pipeline(num_heads: usize, head_dim: usize) -> HighPerformancePipeline {
    let config = HighPerfPipelineConfig {
        num_heads,
        head_dim,
        num_kv_heads: num_heads,
        enable_fa3: false,
        enable_mla: false,
        enable_streaming: false,
        ..Default::default()
    };

    HighPerformancePipeline::new(config).unwrap()
}

// ============================================================================
// Benchmark 1: 短序列性能对比 (64-512 tokens)
// ============================================================================

fn bench_short_sequence_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("short_seq_comparison");

    let num_heads = 4;
    let head_dim = 32;

    for seq_len in [64usize, 128, 256, 512].iter() {
        let (q, k, v) = create_test_data(*seq_len, num_heads, head_dim);

        group.throughput(Throughput::Elements(*seq_len as u64));

        group.bench_with_input(
            BenchmarkId::new(format!("FA3_{}", seq_len), seq_len),
            seq_len,
            |b, _| {
                let mut pipeline = create_fa3_pipeline(num_heads, head_dim);
                b.iter(|| {
                    pipeline
                        .forward(black_box(&q), black_box(&k), black_box(&v))
                        .unwrap()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("Standard_{}", seq_len), seq_len),
            seq_len,
            |b, _| {
                let mut pipeline = create_standard_pipeline(num_heads, head_dim);
                b.iter(|| {
                    pipeline
                        .forward(black_box(&q), black_box(&k), black_box(&v))
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark 2: 中等序列性能对比 (1K-4K tokens)
// ============================================================================

fn bench_medium_sequence_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("medium_seq_comparison");

    let num_heads = 8;
    let head_dim = 64;

    for seq_len in [1024usize, 2048, 4096].iter() {
        let (q, k, v) = create_test_data(*seq_len, num_heads, head_dim);

        group.throughput(Throughput::Elements(*seq_len as u64));
        group.sample_size(20);

        group.bench_with_input(
            BenchmarkId::new(format!("FA3_{}", seq_len), seq_len),
            seq_len,
            |b, _| {
                let mut pipeline = create_fa3_pipeline(num_heads, head_dim);
                b.iter(|| {
                    pipeline
                        .forward(black_box(&q), black_box(&k), black_box(&v))
                        .unwrap()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("Standard_{}", seq_len), seq_len),
            seq_len,
            |b, _| {
                let mut pipeline = create_standard_pipeline(num_heads, head_dim);
                b.iter(|| {
                    pipeline
                        .forward(black_box(&q), black_box(&k), black_box(&v))
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark 3: 长序列性能 (8K-16K tokens)
// ============================================================================

fn bench_long_sequence(c: &mut Criterion) {
    let mut group = c.benchmark_group("long_sequence");

    let num_heads = 8;
    let head_dim = 64;

    for seq_len in [8192usize, 16384].iter() {
        let (q, k, v) = create_test_data(*seq_len, num_heads, head_dim);

        group.throughput(Throughput::Elements(*seq_len as u64));
        group.sample_size(5);

        group.bench_with_input(
            BenchmarkId::new(format!("Standard_{}", seq_len), seq_len),
            seq_len,
            |b, _| {
                let mut pipeline = create_standard_pipeline(num_heads, head_dim);
                b.iter(|| {
                    pipeline
                        .forward(black_box(&q), black_box(&k), black_box(&v))
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark 4: 不同模型规模配置
// ============================================================================

fn bench_model_scale_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_scale_comparison");

    let seq_len = 256;

    let configs = vec![
        ("7B", HighPerfPipelineConfig::for_7b_model()),
        ("13B", HighPerfPipelineConfig::for_13b_model()),
        ("70B", HighPerfPipelineConfig::for_70b_model()),
    ];

    for (name, config) in &configs {
        let (q, k, v) = create_test_data(seq_len, config.num_heads, config.head_dim);

        group.throughput(Throughput::Elements(seq_len as u64));

        group.bench_with_input(BenchmarkId::new(name.to_string(), name), name, |b, _| {
            let mut pipeline = HighPerformancePipeline::new(config.clone()).unwrap();
            b.iter(|| {
                pipeline
                    .forward(black_box(&q), black_box(&k), black_box(&v))
                    .unwrap()
            });
        });
    }

    group.finish();
}

// ============================================================================
// Benchmark 5: 批量推理吞吐量
// ============================================================================

fn bench_batch_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_throughput");

    let batch_sizes = [1usize, 4, 8, 16];
    let seq_len = 64;
    let num_heads = 4;
    let head_dim = 32;

    for batch_size in &batch_sizes {
        let mut queries = Vec::new();
        let mut keys = Vec::new();
        let mut values = Vec::new();

        for _ in 0..*batch_size {
            let (q, k, v) = create_test_data(seq_len, num_heads, head_dim);
            queries.push(q);
            keys.push(k);
            values.push(v);
        }

        let q_refs: Vec<&Array2<f32>> = queries.iter().collect();
        let k_refs: Vec<&Array2<f32>> = keys.iter().collect();
        let v_refs: Vec<&Array2<f32>> = values.iter().collect();

        let total_elements = (*batch_size * seq_len) as u64;

        group.throughput(Throughput::Elements(total_elements));

        group.bench_with_input(
            BenchmarkId::new(format!("batch_{}", batch_size), batch_size),
            batch_size,
            |b, _| {
                let mut pipeline = create_fa3_pipeline(num_heads, head_dim);
                b.iter(|| pipeline.batch_forward(&q_refs, &k_refs, &v_refs).unwrap());
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark 6: KV Cache 操作开销
// ============================================================================

fn bench_kv_cache_operations(c: &mut Criterion) {
    use openmini_server::hardware::kv_cache::block::KVCacheConfig;
    use openmini_server::hardware::kv_cache::paged_cache::PagedKVCache;

    let mut group = c.benchmark_group("kv_cache_operations");

    let config = KVCacheConfig {
        num_layers: 32,
        num_heads: 32,
        head_dim: 128,
        max_blocks: 1024,
        block_size: 16,
        dtype_size: 2,
        enable_prefix_cache: true,
        enable_swap: false,
    };

    group.bench_function("create_paged_kv", |b| {
        b.iter(|| PagedKVCache::new(config.clone()));
    });

    let cache = PagedKVCache::new(config);

    group.bench_function("cache_info_query", |b| {
        b.iter(|| {
            black_box(cache.available_blocks());
            black_box(cache.utilization());
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark 7: Pipeline 创建开销
// ============================================================================

fn bench_pipeline_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_creation");

    group.bench_function("fa3_pipeline", |b| {
        b.iter(|| create_fa3_pipeline(32, 128));
    });

    group.bench_function("standard_pipeline", |b| {
        b.iter(|| create_standard_pipeline(32, 128));
    });

    group.bench_function("7b_model_config", |b| {
        b.iter(|| HighPerfPipelineConfig::for_7b_model());
    });

    group.bench_function("13b_model_config", |b| {
        b.iter(|| HighPerfPipelineConfig::for_13b_model());
    });

    group.bench_function("70b_model_config", |b| {
        b.iter(|| HighPerfPipelineConfig::for_70b_model());
    });

    group.finish();
}

// ============================================================================
// 主入口
// ============================================================================

criterion_group!(
    benches,
    bench_short_sequence_comparison,
    bench_medium_sequence_comparison,
    bench_long_sequence,
    bench_model_scale_comparison,
    bench_batch_throughput,
    bench_kv_cache_operations,
    bench_pipeline_creation,
);
criterion_main!(benches);
