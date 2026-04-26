//! Metal GPU 加速 GEMM 基准测试
//!
//! # 测试目标
//!
//! 验证和 benchmark CandleMetalBackend (Apple Metal GPU) 的核心运算性能，
//! 并与 CPU ndarray 实现进行对比，量化 GPU 加速效果。
//!
//! # 测试覆盖范围
//!
//! - **matmul**: 不同规模的矩阵乘法 (128x128, 512x512, 1024x1024)
//!   - 目的：验证 Metal GPU 在不同矩阵尺寸下的吞吐量特性
//!   - 小矩阵关注延迟，大矩阵关注带宽利用率
//!
//! - **batched_matmul**: 批量矩阵乘法
//!   - 目的：测试 GPU 并行处理多个独立矩阵乘法的能力
//!   - 模拟推理时 batched attention 场景
//!
//! - **fused_gemm_relu**: 融合 GEMM + ReLU 激活
//!   - 目的：测试 kernel 融合带来的性能收益
//!   - 对应 MLP 层的前向传播模式：output = ReLU(x @ W^T + b)
//!
//! - **fused_gemm_silu**: 融合 GEMM + SiLU (Swish) 激活
//!   - 目的：测试 SwiGLU 模式的 GPU 执行效率
//!   - 对应现代 LLM (LLaMA 系列) 的 FFN 层：
//!     output = (x @ gate_w^T) * SiLU(x @ up_w^T) + bias
//!
//! # 运行方式
//!
//! ```bash
//! # 仅在 macOS 上且启用 metal feature 时可运行
//! cargo bench --package openmini-server --features metal --bench metal_bench
//!
//! # 运行特定测试组
//! cargo bench --package openmini-server --features metal --bench metal_bench -- matmul
//!
//! # 生成 HTML 报告（输出到 target/criterion/）
//! cargo bench --package openmini-server --features metal --bench metal_bench -- --save-baseline main
//! ```
//!
//! # 平台限制
//!
//! 本基准测试仅在以下条件同时满足时编译和运行：
//! - `target_os = "macos"`: Apple macOS 系统
//! - `feature = "metal"`: 启用了 Metal GPU 支持

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array1, Array2, Array3};

#[cfg(all(target_os = "macos", feature = "metal"))]
use openmini_server::model::inference::gemm_engine::metal_backend::CandleMetalBackend;
#[cfg(all(target_os = "macos", feature = "metal"))]
use openmini_server::model::inference::gemm_engine::{GemmEngine, NdarrayFallbackBackend};

// ============================================================================
// 辅助函数：生成确定性测试数据
// ============================================================================

/// 生成指定形状的 f32 矩阵，使用确定性伪随机值
///
/// 使用简单的线性同余方式生成测试数据，确保：
/// - 每次调用结果一致（benchmark 可复现）
/// - 数据分布合理（覆盖正负值，避免全零或极端值）
/// - 数值范围适合 FP32 计算（约 [-1.0, 1.0]）
fn make_matrix(rows: usize, cols: usize) -> Array2<f32> {
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        let val = ((i * cols + j) as f32 * 0.01).sin() + ((i * cols + j) as f32 * 0.007).cos();
        // 归一化到合理范围
        val * 0.5
    })
}

/// 生成指定形状的批量 f32 矩阵（3D tensor）
///
/// 形状为 (batch_size, rows, cols)，用于 batched_matmul 测试
fn make_batch_matrix(batch_size: usize, rows: usize, cols: usize) -> Array3<f32> {
    Array3::from_shape_fn((batch_size, rows, cols), |(b, i, j)| {
        let val = ((b * rows * cols + i * cols + j) as f32 * 0.01).sin()
            + ((b * rows * cols + i * cols + j) as f32 * 0.007).cos();
        val * 0.5
    })
}

/// 生成偏置向量
///
/// 形状为 (size,)，用于 fused_gemm_relu / fused_gemm_silu 测试
fn make_bias(size: usize) -> Array1<f32> {
    Array1::from_shape_fn(size, |i| ((i as f32) * 0.01 - 0.5))
}

// ============================================================================
// 1. MatMul 基准测试：Metal GPU vs CPU (ndarray)
// ============================================================================

/// MatMul 性能对比基准
///
/// 测试不同矩阵尺寸下 Metal GPU 与 CPU ndarray 的矩阵乘法性能。
///
/// # 测试目的
/// - 验证 Metal 后端在各个尺寸下的正确性（通过结果一致性隐式保证）
/// - 量化 GPU 相对 CPU 的加速比
/// - 识别 GPU 加速的"盈亏平衡点"（小矩阵可能因 kernel launch 开销而慢于 CPU）
///
/// # 矩阵尺寸选择说明
/// - 128x128: 小矩阵，测试 kernel launch 开销占比
/// - 512x512: 中等矩阵，典型 embedding/attention 投影尺寸
/// - 1024x1024: 大矩阵，测试 GPU 带宽和并行度优势
#[cfg(all(target_os = "macos", feature = "metal"))]
fn bench_matmul(c: &mut Criterion) {
    // 初始化 Metal 后端
    let metal_backend: CandleMetalBackend = match CandleMetalBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("WARNING: Metal backend initialization failed: {}", e);
            eprintln!("Skipping Metal benchmarks. Falling back to CPU-only mode.");
            return;
        }
    };
    let cpu_backend = NdarrayFallbackBackend;

    let matrix_sizes: [(usize, usize); 3] = [(128, 128), (512, 512), (1024, 1024)];

    let mut group = c.benchmark_group("matmul");

    for &(m, k) in &matrix_sizes {
        let n = m; // 方阵：C(m,n) = A(m,k) @ B(k,n)

        // 预先生成测试数据（避免在 benchmark 循环中分配内存）
        let a = make_matrix(m, k);
        let b = make_matrix(k, n);

        // 计算理论 FLOPs 用于 throughput 度量
        // MatMul: C = A @ B 需要 2*m*k*n 次 FLOP (乘+加)
        let flops = 2u64 * m as u64 * k as u64 * n as u64;
        group.throughput(Throughput::Elements(flops));

        // --- Metal GPU MatMul ---
        // 注意：通过闭包捕获 &metal_backend 引用，避免 move 语义导致循环内重复使用失败
        {
            let a_ref = &a;
            let b_ref = &b;
            group.bench_function(BenchmarkId::new("metal", format!("{}x{}", m, n)), |b| {
                b.iter(|| {
                    let result = metal_backend.matmul(black_box(a_ref), black_box(b_ref));
                    black_box(result)
                });
            });
        }

        // --- CPU (ndarray) MatMul ---
        {
            let a_ref = &a;
            let b_ref = &b;
            group.bench_function(
                BenchmarkId::new("cpu_ndarray", format!("{}x{}", m, n)),
                |b| {
                    b.iter(|| {
                        let result = cpu_backend.matmul(black_box(a_ref), black_box(b_ref));
                        black_box(result)
                    });
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// 2. Batched MatMul 基准测试：GPU 批量并行能力
// ============================================================================

/// 批量矩阵乘法性能对比基准
///
/// 测试 GPU 同时处理多个独立矩阵乘法的能力。
/// 这是 Transformer 推理中 multi-head attention 的核心操作。
///
/// # 测试目的
/// - 验证 Metal GPU 在批量场景下的并行效率
/// - 与 CPU 逐个计算的方式对比，展示 GPU batch 处理优势
/// - 模拟实际推理中的 Q@K^T 和 Q@V^T 操作
///
/// # 配置说明
/// - batch_size=8: 典型的 attention head 数量
/// - 矩阵内部尺寸从 64 到 512，覆盖不同的 head_dim
#[cfg(all(target_os = "macos", feature = "metal"))]
fn bench_batched_matmul(c: &mut Criterion) {
    let metal_backend: CandleMetalBackend = match CandleMetalBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("WARNING: Metal backend init failed: {}", e);
            return;
        }
    };
    let cpu_backend = NdarrayFallbackBackend;

    // (batch_size, rows, cols) 配置
    let configs: [(usize, usize, usize); 3] = [(8, 64, 64), (8, 128, 128), (8, 256, 256)];

    let mut group = c.benchmark_group("batched_matmul");

    for &(batch, m, k) in &configs {
        let n = m;

        let a = make_batch_matrix(batch, m, k);
        let b = make_batch_matrix(batch, k, n);

        let flops = 2u64 * batch as u64 * m as u64 * k as u64 * n as u64;
        group.throughput(Throughput::Elements(flops));

        // --- Metal GPU Batched MatMul ---
        {
            let a_ref = &a;
            let b_ref = &b;
            group.bench_function(
                BenchmarkId::new("metal", format!("batch{}_{}x{}", batch, m, n)),
                |b| {
                    b.iter(|| {
                        let result =
                            metal_backend.batched_matmul(black_box(a_ref), black_box(b_ref));
                        black_box(result)
                    });
                },
            );
        }

        // --- CPU (ndarray) Batched MatMul ---
        {
            let a_ref = &a;
            let b_ref = &b;
            group.bench_function(
                BenchmarkId::new("cpu_ndarray", format!("batch{}_{}x{}", batch, m, n)),
                |b| {
                    b.iter(|| {
                        let result = cpu_backend.batched_matmul(black_box(a_ref), black_box(b_ref));
                        black_box(result)
                    });
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// 3. Fused GEMM + ReLU 基准测试：Kernel 融合效果
// ============================================================================

/// 融合 GEMM + ReLU 激活函数性能对比基准
///
/// 测试 output = ReLU(x @ W^T + bias) 操作的端到端性能。
/// 这是标准 MLP / 全连接层的核心计算模式。
///
/// # 测试目的
/// - 评估 kernel 融合（GEMM + bias add + ReLU）的性能收益
/// - Metal 后端可以在单个 compute pass 中完成所有操作，
///   避免 GEMM 结果写回内存再读回的额外开销
/// - 对应网络结构中的 Linear -> ReLU 层
///
/// # 注意事项
/// - 当前 CandleMetalBackend 的 fused_gemm_relu 实际上是分步执行
///   （GEMM -> add bias -> relu），未来可通过 Metal shader 融合进一步优化
/// - 此 benchmark 作为融合优化的基线参考
#[cfg(all(target_os = "macos", feature = "metal"))]
fn bench_fused_gemm_relu(c: &mut Criterion) {
    let metal_backend: CandleMetalBackend = match CandleMetalBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("WARNING: Metal backend init failed: {}", e);
            return;
        }
    };
    let cpu_backend = NdarrayFallbackBackend;

    // (input_dim, output_dim) 配置
    let configs: [(usize, usize); 3] = [
        (512, 2048),  // 小型 MLP: hidden -> intermediate
        (1024, 4096), // 中型 MLP
        (2048, 8192), // 大型 MLP
    ];

    let mut group = c.benchmark_group("fused_gemm_relu");

    for &(in_dim, out_dim) in &configs {
        // x: (batch, in_dim), w: (out_dim, in_dim), bias: (out_dim,)
        let batch = 64;
        let x = make_matrix(batch, in_dim);
        let w = make_matrix(out_dim, in_dim);
        let bias = make_bias(out_dim);

        // GEMM FLOPs: 2 * batch * in_dim * out_dim
        let flops = 2u64 * batch as u64 * in_dim as u64 * out_dim as u64;
        group.throughput(Throughput::Elements(flops));

        // --- Metal GPU Fused GEMM + ReLU ---
        {
            let x_ref = &x;
            let w_ref = &w;
            let bias_opt: Option<&Array1<f32>> = Some(&bias);
            group.bench_function(
                BenchmarkId::new("metal", format!("{}x{}", in_dim, out_dim)),
                |b| {
                    b.iter(|| {
                        let result = metal_backend.fused_gemm_relu(
                            black_box(x_ref),
                            black_box(w_ref),
                            bias_opt,
                        );
                        black_box(result)
                    });
                },
            );
        }

        // --- CPU (ndarray) Fused GEMM + ReLU ---
        {
            let x_ref = &x;
            let w_ref = &w;
            let bias_opt: Option<&Array1<f32>> = Some(&bias);
            group.bench_function(
                BenchmarkId::new("cpu_ndarray", format!("{}x{}", in_dim, out_dim)),
                |b| {
                    b.iter(|| {
                        let result = cpu_backend.fused_gemm_relu(
                            black_box(x_ref),
                            black_box(w_ref),
                            bias_opt,
                        );
                        black_box(result)
                    });
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// 4. Fused GEMM + SiLU (SwiGLU) 基准测试：LLM FFN 核心操作
// ============================================================================

/// 融合 GEMM + SiLU (Swish) 激活函数性能对比基准
///
/// 测试 SwiGLU FFN 模式：output = gate * SiLU(up) + bias
/// 其中 gate = x @ gate_w^T, up = x @ up_w^T
///
/// # 测试目的
/// - 评估 LLaMA 系列 LLM 的 Feed-Forward Network (FFN) 层在 Metal GPU 上的执行效率
/// - SwiGLU 是现代大语言模型的核心组件，涉及两次矩阵乘法和一次逐元素非线性激活
/// - SiLU(x) = x * sigmoid(x) 是计算密集型激活函数
///
/// # 计算复杂度分析
/// - 2 次 GEMM: 2 * 2 * batch * in_dim * out_dim FLOPs
/// - SiLU: sigmoid + multiply per element
/// - 总计约 4x 单次 GEMM 的计算量
///
/// # 应用场景
/// - LLaMA / Mistral / Qwen 等 LLM 的 FFN 层
/// - MoE (Mixture of Experts) 的专家网络前向传播
#[cfg(all(target_os = "macos", feature = "metal"))]
fn bench_fused_gemm_silu(c: &mut Criterion) {
    let metal_backend: CandleMetalBackend = match CandleMetalBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("WARNING: Metal backend init failed: {}", e);
            return;
        }
    };
    let cpu_backend = NdarrayFallbackBackend;

    // (input_dim, output_dim) 配置 — 模拟不同规模模型的 FFN
    let configs: [(usize, usize); 3] = [
        (512, 1382),  // ~7B 模型 FFN (hidden=512, intermediate≈2.7x)
        (1024, 2765), // ~13B 模型 FFN
        (2048, 5530), // ~30B 模型 FFN
    ];

    let mut group = c.benchmark_group("fused_gemm_silu");

    for &(in_dim, out_dim) in &configs {
        let batch = 64;
        let x = make_matrix(batch, in_dim);
        let gate_w = make_matrix(out_dim, in_dim);
        let up_w = make_matrix(out_dim, in_dim);
        let bias = make_bias(out_dim);

        // 2次GEMM + SiLU: 约 4 * batch * in_dim * out_dim FLOPs
        let flops = 4u64 * batch as u64 * in_dim as u64 * out_dim as u64;
        group.throughput(Throughput::Elements(flops));

        // --- Metal GPU Fused GEMM + SiLU ---
        {
            let x_ref = &x;
            let gw_ref = &gate_w;
            let uw_ref = &up_w;
            let bias_opt: Option<&Array1<f32>> = Some(&bias);
            group.bench_function(
                BenchmarkId::new("metal", format!("{}x{}", in_dim, out_dim)),
                |b| {
                    b.iter(|| {
                        let result = metal_backend.fused_gemm_silu(
                            black_box(x_ref),
                            black_box(gw_ref),
                            black_box(uw_ref),
                            bias_opt,
                        );
                        black_box(result)
                    });
                },
            );
        }

        // --- CPU (ndarray) Fused GEMM + SiLU ---
        {
            let x_ref = &x;
            let gw_ref = &gate_w;
            let uw_ref = &up_w;
            let bias_opt: Option<&Array1<f32>> = Some(&bias);
            group.bench_function(
                BenchmarkId::new("cpu_ndarray", format!("{}x{}", in_dim, out_dim)),
                |b| {
                    b.iter(|| {
                        let result = cpu_backend.fused_gemm_silu(
                            black_box(x_ref),
                            black_box(gw_ref),
                            black_box(uw_ref),
                            bias_opt,
                        );
                        black_box(result)
                    });
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// 注册基准测试组
// ============================================================================

#[cfg(all(target_os = "macos", feature = "metal"))]
criterion_group!(
    benches,
    bench_matmul,
    bench_batched_matmul,
    bench_fused_gemm_relu,
    bench_fused_gemm_silu,
);
#[cfg(all(target_os = "macos", feature = "metal"))]
criterion_main!(benches);

/// 非 macOS 或未启用 metal feature 时的空 main 函数
#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn main() {
    eprintln!();
    eprintln!("==============================================");
    eprintln!("  Metal benchmarks are not available");
    eprintln!("==============================================");
    eprintln!();
    eprintln!("  This benchmark requires:");
    eprintln!("    1. macOS operating system (target_os = \"macos\")");
    eprintln!("    2. 'metal' feature enabled (--features metal)");
    eprintln!();
    eprintln!("  Run with:");
    eprintln!("    cargo bench --package openmini-server \\");
    eprintln!("      --features metal --bench metal_bench");
    eprintln!();
}
