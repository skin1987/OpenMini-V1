//! OpenMini-V1 性能基准测试套件
//!
//! # 测试目标
//!
//! 建立性能基线，用于回归检测：
//! - 推理延迟和吞吐量
//! - 内存分配效率
//! - 序列化/反序列化性能
//! - 并发操作开销
//!
//! # 运行方式
//!
//! ```bash
//! # 运行所有基准测试（需要 nightly Rust）
//! cargo +nightly bench --package openmini-server
//!
//! # 运行特定基准测试组
//! cargo +nightly bench --package openmini-server --bench performance_bench -- inference
//!
//! # 生成 HTML 报告（criterion 默认输出到 target/criterion/）
//! cargo +nightly bench --package openmini-server -- --save-baseline main
//! ```
//!
//! # 输出位置
//!
//! 基准测试报告默认生成在 `target/criterion/report/` 目录下

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

// ============================================================================
// 1. 推理延迟基准测试
// ============================================================================

/// 推理延迟基准 - 单次推理
///
/// 测试不同序列长度下的单次推理延迟
fn bench_inference_latency(c: &mut Criterion) {
    let seq_lengths = [128usize, 512, 1024, 2048];

    let mut group = c.benchmark_group("inference_latency");
    // CI 环境使用较短测量时间
    if is_ci_env() {
        group.measurement_time(Duration::from_secs(5));
        group.sample_size(20);
    }

    for &seq_len in &seq_lengths {
        group.bench_with_input(
            BenchmarkId::new("single_inference", seq_len),
            &seq_len,
            |b, &len| {
                b.iter(|| {
                    // 模拟推理计算：矩阵乘法 + 激活函数
                    simulate_inference(black_box(len));
                });
            },
        );
    }
    group.finish();
}

/// 批量推理吞吐量基准
///
/// 测试不同 batch size 下的吞吐量（tokens/sec）
fn bench_batch_throughput(c: &mut Criterion) {
    let batch_sizes = [1usize, 4, 8, 16];
    let seq_len = 512;

    let mut group = c.benchmark_group("batch_throughput");
    if is_ci_env() {
        group.measurement_time(Duration::from_secs(5));
        group.sample_size(20);
    }

    for &batch_size in &batch_sizes {
        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &(batch_size, seq_len),
            |b, &(bs, sl)| {
                b.iter(|| {
                    for _ in 0..bs {
                        black_box(simulate_inference(sl));
                    }
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// 2. 内存分配基准测试
// ============================================================================

/// KV Cache 内存分配基准
///
/// 测试 KV Cache 的内存分配/释放性能
fn bench_kv_cache_allocation(c: &mut Criterion) {
    use openmini_server::hardware::kv_cache::{block::KVCacheConfig, block_manager::BlockManager};

    let configs = [(64, 16), (256, 32), (1024, 64)]; // (num_blocks, block_size)

    let mut group = c.benchmark_group("kv_cache_allocation");
    if is_ci_env() {
        group.measurement_time(Duration::from_secs(3));
        group.sample_size(15);
    }

    for &(num_blocks, block_size) in &configs {
        group.bench_with_input(
            BenchmarkId::new("allocate_free", format!("{}x{}", num_blocks, block_size)),
            &(num_blocks, block_size),
            |b, &(nb, bs)| {
                b.iter(|| {
                    let config = KVCacheConfig {
                        max_blocks: nb,
                        block_size: bs,
                        ..Default::default()
                    };
                    let mut manager = BlockManager::new(&config);

                    // 分配一半的块
                    let allocated = manager.allocate(nb / 2, None).unwrap();
                    // 释放所有块
                    manager.free(&allocated);

                    black_box(&manager);
                });
            },
        );
    }
    group.finish();
}

/// 张量创建/销毁基准
///
/// 测试 ndarray 张量的创建和销毁开销
fn bench_tensor_operations(c: &mut Criterion) {
    let sizes = [1000, 10000, 100000]; // 元素数量

    let mut group = c.benchmark_group("tensor_operations");
    if is_ci_env() {
        group.measurement_time(Duration::from_secs(3));
        group.sample_size(15);
    }

    // 张量创建
    for &size in &sizes {
        group.bench_with_input(BenchmarkId::new("create", size), &size, |b, &s| {
            b.iter(|| {
                let arr: Vec<f32> = (0..s).map(|i| i as f32).collect();
                black_box(arr);
            });
        });

        // 张量运算
        group.bench_with_input(BenchmarkId::new("elementwise_ops", size), &size, |b, &s| {
            let data: Vec<f32> = (0..s).map(|i| i as f32).collect();
            b.iter(|| {
                let result: Vec<f32> = data.iter().map(|&x| x.sin().cos()).collect();
                black_box(result);
            });
        });
    }
    group.finish();
}

// ============================================================================
// 3. 序列化/反序列化基准测试
// ============================================================================

/// JSON 序列化基准
///
/// 测试 serde_json 的序列化/反序列化性能
fn bench_json_serialization(c: &mut Criterion) {
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, Clone)]
    struct TestPayload {
        id: u64,
        tokens: Vec<u32>,
        logits: Vec<f32>,
        metadata: String,
    }

    fn create_payload(size: usize) -> TestPayload {
        TestPayload {
            id: 42,
            tokens: (0..size as u32).collect(),
            logits: (0..size).map(|i| i as f32 * 0.01).collect(),
            metadata: "benchmark test payload".repeat(10),
        }
    }

    let sizes = [128, 512, 2048];

    let mut group = c.benchmark_group("serialization");
    if is_ci_env() {
        group.measurement_time(Duration::from_secs(3));
        group.sample_size(15);
    }

    for &size in &sizes {
        let payload = create_payload(size);

        // JSON 序列化
        group.bench_with_input(
            BenchmarkId::new("json_serialize", size),
            &payload,
            |b, p| {
                b.iter(|| {
                    let json = serde_json::to_string(black_box(p)).unwrap();
                    black_box(json);
                });
            },
        );

        // JSON 反序列化
        let json_str = serde_json::to_string(&payload).unwrap();
        group.bench_with_input(
            BenchmarkId::new("json_deserialize", size),
            &json_str,
            |b, s| {
                b.iter(|| {
                    let p: TestPayload = serde_json::from_str(black_box(s)).unwrap();
                    black_box(p);
                });
            },
        );

        // Bincode 序列化（二进制格式）
        group.bench_with_input(
            BenchmarkId::new("bincode_serialize", size),
            &payload,
            |b, p| {
                b.iter(|| {
                    let bytes = bincode::serialize(black_box(p)).unwrap();
                    black_box(bytes);
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// 4. 并发性能基准测试
// ============================================================================

/// 线程池任务调度基准
///
/// 测试线程池的任务提交和执行延迟
fn bench_thread_pool_scheduling(c: &mut Criterion) {
    

    let task_counts = [100, 1000, 10000];

    let mut group = c.benchmark_group("thread_pool");
    if is_ci_env() {
        group.measurement_time(Duration::from_secs(5));
        group.sample_size(20);
    }

    for &count in &task_counts {
        group.bench_with_input(
            BenchmarkId::new("submit_and_wait", count),
            &count,
            |b, &cnt| {
                b.iter(|| {
                    let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
                    let mut handles = Vec::with_capacity(cnt);

                    for _ in 0..cnt {
                        let counter = std::sync::Arc::clone(&counter);
                        handles.push(std::thread::spawn(move || {
                            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        }));
                    }

                    for handle in handles {
                        let _ = handle.join();
                    }

                    black_box(counter.load(std::sync::atomic::Ordering::SeqCst));
                });
            },
        );
    }
    group.finish();
}

/// 锁竞争基准
///
/// 测试 Mutex/RwLock 在高竞争场景下的性能
fn bench_lock_contention(c: &mut Criterion) {
    use parking_lot::{Mutex, RwLock};
    use std::sync::Arc;

    let thread_counts = [2, 4, 8];
    let ops_per_thread = 1000;

    let mut group = c.benchmark_group("lock_contention");
    if is_ci_env() {
        group.measurement_time(Duration::from_secs(5));
        group.sample_size(15);
    }

    // Mutex 竞争
    for &threads in &thread_counts {
        group.bench_with_input(
            BenchmarkId::new("mutex_contention", threads),
            &(threads, ops_per_thread),
            |b, &(t, ops)| {
                b.iter(|| {
                    let data = Arc::new(Mutex::new(0usize));
                    let mut handles = Vec::with_capacity(t);

                    for _ in 0..t {
                        let data = Arc::clone(&data);
                        handles.push(std::thread::spawn(move || {
                            for _ in 0..ops {
                                let mut guard = data.lock();
                                *guard += 1;
                                drop(guard);
                            }
                        }));
                    }

                    for h in handles {
                        h.join().unwrap();
                    }

                    black_box(*data.lock());
                });
            },
        );
    }

    // RwLock 读多写少场景
    for &threads in &thread_counts {
        group.bench_with_input(
            BenchmarkId::new("rwlock_read_heavy", threads),
            &(threads, ops_per_thread),
            |b, &(t, ops)| {
                b.iter(|| {
                    let data = Arc::new(RwLock::new(Vec::<u8>::with_capacity(1024)));
                    let mut handles = Vec::with_capacity(t);

                    for i in 0..t {
                        let data = Arc::clone(&data);
                        handles.push(std::thread::spawn(move || {
                            for j in 0..ops {
                                if j % 100 == 0 && i == 0 {
                                    // 偶尔写入
                                    let mut w = data.write();
                                    w.push(i as u8);
                                    drop(w);
                                } else {
                                    // 主要读取
                                    let r = data.read();
                                    black_box(r.len());
                                    drop(r);
                                }
                            }
                        }));
                    }

                    for h in handles {
                        h.join().unwrap();
                    }

                    black_box(data.read().len());
                });
            },
        );
    }
    group.finish();
}

/// Channel 通信基准
///
/// 测试 crossbeam channel 的通信延迟
fn bench_channel_communication(c: &mut Criterion) {
    use crossbeam::channel::bounded;

    let message_sizes = [64, 1024, 16384]; // 消息大小（字节）

    let mut group = c.benchmark_group("channel");
    if is_ci_env() {
        group.measurement_time(Duration::from_secs(3));
        group.sample_size(15);
    }

    for &msg_size in &message_sizes {
        group.bench_with_input(
            BenchmarkId::new("bounded_send_recv", msg_size),
            &msg_size,
            |b, &ms| {
                b.iter(|| {
                    let (tx, rx) = bounded::<Vec<u8>>(100);
                    let msg: Vec<u8> = (0..ms).map(|i| i as u8).collect();

                    for _ in 0..1000 {
                        tx.send(msg.clone()).unwrap();
                        let received = rx.recv().unwrap();
                        black_box(received);
                    }
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// 5. DSA 计算基准测试
// ============================================================================

/// DSA 稀疏注意力计算基准
///
/// 不同稀疏度配置下的 DSA 性能表现
fn bench_dsa_computation(c: &mut Criterion) {
    use ndarray::Array2;
    use openmini_server::model::inference::dsa::{
        configure_rayon_pool, sparse_attention_forward, DSATopKConfig,
    };

    let _ = configure_rayon_pool();

    let seq_len = 512;
    let head_dim = 64;
    let sparsity_levels = [("25%", 0.25), ("50%", 0.5), ("75%", 0.75)];

    let mut group = c.benchmark_group("dsa_computation");
    if is_ci_env() {
        group.measurement_time(Duration::from_secs(5));
        group.sample_size(15);
    }

    // 预生成测试数据
    let q: Array2<f32> = Array2::from_shape_fn((seq_len, head_dim), |(i, j)| {
        ((i * head_dim + j) as f32 * 0.01).sin()
    });
    let k: Array2<f32> = Array2::from_shape_fn((seq_len, head_dim), |(i, j)| {
        ((i * head_dim + j) as f32 * 0.01).cos()
    });
    let v: Array2<f32> = Array2::from_shape_fn((seq_len, head_dim), |(i, j)| {
        ((i * head_dim + j) as f32 * 0.01).tan()
    });

    for &(name, sparsity) in &sparsity_levels {
        let config = DSATopKConfig {
            base_top_k: (seq_len as f32 * (1.0 - sparsity)) as usize,
            use_dynamic_k: true,
            short_seq_threshold: 512,
        };

        group.bench_with_input(
            BenchmarkId::new("sparse_attention", name),
            &config,
            |b, cfg| {
                b.iter(|| {
                    let result =
                        sparse_attention_forward(&q, &k, &v, head_dim, cfg, false).unwrap();
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

/// 量化 SIMD 操作性能基准
///
/// 测试不同量化格式的反量化性能：
/// - Q4_0, Q4_1, Q8_0 量化格式
/// - 不同数据大小：1K, 4K, 16K, 64K 元素
/// - SIMD 加速 vs 标量实现对比
fn bench_quant_simd_performance(c: &mut Criterion) {
    use openmini_server::model::inference::gguf::GgufTensorType;
    use openmini_server::model::inference::quant_simd::{dequantize_simd, safe_dequantize};

    // 测试的量化类型
    let quant_types = [
        ("Q4_0", GgufTensorType::Q4_0),
        ("Q4_1", GgufTensorType::Q4_1),
        ("Q8_0", GgufTensorType::Q8_0),
    ];

    // 测试数据大小（元素数量）
    let data_sizes = [1024, 4096, 16384, 65536];

    let mut group = c.benchmark_group("quant_simd_performance");
    if is_ci_env() {
        group.measurement_time(Duration::from_secs(3));
        group.sample_size(10);
    }

    // 为每个量化类型和数据大小生成测试数据
    for &(type_name, tensor_type) in &quant_types {
        for &size in &data_sizes {
            // 根据量化类型计算所需的字节数
            let bytes_per_element = tensor_type.element_size();
            let total_bytes = size * bytes_per_element;

            // 生成随机测试数据
            let mut test_data = vec![0u8; total_bytes];
            for (i, byte) in test_data.iter_mut().enumerate() {
                *byte = (i % 256) as u8;
            }

            // 基准测试安全反量化函数
            group.bench_with_input(
                BenchmarkId::new(format!("safe_dequantize_{}", type_name), size),
                &(test_data.clone(), tensor_type, size),
                |b, (data, t_type, n)| {
                    b.iter(|| {
                        let result = safe_dequantize(black_box(data), *t_type, *n);
                        let _ = black_box(result); // Ignore Result for benchmark
                    });
                },
            );

            // 基准测试直接 SIMD 反量化（无安全检查）
            group.bench_with_input(
                BenchmarkId::new(format!("dequantize_simd_{}", type_name), size),
                &(test_data.clone(), tensor_type, size),
                |b, (data, t_type, n)| {
                    b.iter(|| {
                        let result = dequantize_simd(black_box(data), *t_type, *n);
                        black_box(result);
                    });
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 模拟推理计算
///
/// 执行类似 LLM 推理的计算模式：矩阵运算 + 非线性激活
fn simulate_inference(seq_len: usize) -> f64 {
    // 模拟注意力分数计算
    let mut sum = 0.0f64;
    let head_dim = 64;

    for i in 0..seq_len.min(1000) {
        for j in 0..head_dim {
            // Q @ K^T 类似操作
            let q_val = ((i * head_dim + j) as f64 * 0.01).sin();
            let k_val = ((j * seq_len + i) as f64 * 0.01).cos();
            sum += q_val * k_val;
        }
    }

    // Softmax 近似
    sum = (sum / seq_len as f64).exp();

    // 模拟激活函数
    sum.tanh()
}

/// 检测是否在 CI 环境运行
fn is_ci_env() -> bool {
    std::env::var("CI").is_ok() || std::env::var("GITHUB_ACTIONS").is_ok()
}

// ============================================================================
// 注册基准测试组
// ============================================================================

criterion_group!(
    benches,
    // 推理性能
    bench_inference_latency,
    bench_batch_throughput,
    // 内存分配
    bench_kv_cache_allocation,
    bench_tensor_operations,
    // 序列化
    bench_json_serialization,
    // 并发性能
    bench_thread_pool_scheduling,
    bench_lock_contention,
    bench_channel_communication,
    // DSA 计算
    bench_dsa_computation,
    // 量化 SIMD 性能
    bench_quant_simd_performance,
);

criterion_main!(benches);
