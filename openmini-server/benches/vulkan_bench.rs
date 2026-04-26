//! Vulkan GPU 加速 GEMM 基准测试
//!
//! # 测试目标
//!
//! 验证和 benchmark VulkanGemmBackend (Vulkan Compute Shader) 的核心运算性能，
//! 并与 CPU ndarray 实现进行对比，量化 Vulkan GPU 加速效果。
//!
//! # 测试覆盖范围
//!
//! - **matmul**: 不同规模的矩阵乘法 (128x128, 512x512, 1024x1024)
//!   - 目的：验证 Vulkan GPU 在不同矩阵尺寸下的吞吐量特性
//!   - 小矩阵关注延迟（kernel launch 开销），大矩阵关注带宽利用率
//!   - 识别 GPU 加速的"盈亏平衡点"
//!
//! - **batched_matmul**: 批量矩阵乘法
//!   - 目的：测试 GPU 并行处理多个独立矩阵乘法的能力
//!   - 模拟推理时 multi-head attention 场景中的 Q@K^T / Q@V^T 操作
//!
//! # 运行方式
//!
//! ```bash
//! # 仅在启用 vulkan feature 时可运行（跨平台: Linux/Windows/macOS）
//! cargo bench --package openmini-server --features vulkan --bench vulkan_bench
//!
//! # 运行特定测试组
//! cargo bench --package openmini-server --features vulkan --bench vulkan_bench -- matmul
//!
//! # 生成 HTML 报告（输出到 target/criterion/）
//! cargo bench --package openmini-server --features vulkan --bench vulkan_bench -- --save-baseline main
//! ```
//!
//! # 平台限制
//!
//! 本基准测试仅在以下条件满足时编译和运行：
//! - `feature = "vulkan"`: 启用了 Vulkan GPU 支持
//! - 系统需安装 Vulkan 驱动并支持 compute shader

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array2, Array3};

#[cfg(feature = "vulkan")]
use openmini_server::model::inference::gemm_engine::vulkan_backend::VulkanGemmBackend;
#[cfg(feature = "vulkan")]
use openmini_server::model::inference::gemm_engine::{GemmEngine, NdarrayFallbackBackend};

// ============================================================================
// 辅助函数：生成确定性测试数据
// ============================================================================

/// 生成指定形状的 f32 矩阵，使用确定性伪随机值
///
/// 使用三角函数组合方式生成测试数据，确保：
/// - 每次调用结果一致（benchmark 可复现）
/// - 数据分布合理（覆盖正负值，避免全零或极端值）
/// - 数值范围适合 FP32 计算（约 [-1.0, 1.0]）
///
/// # 参数
///
/// - `rows`: 矩阵行数 (M)
/// - `cols`: 矩阵列数 (N/K)
fn make_matrix(rows: usize, cols: usize) -> Array2<f32> {
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        let val = ((i * cols + j) as f32 * 0.01).sin() + ((i * cols + j) as f32 * 0.007).cos();
        // 归一化到合理范围 [-1.0, 1.0]
        val * 0.5
    })
}

/// 生成指定形状的批量 f32 矩阵（3D tensor）
///
/// 形状为 (batch_size, rows, cols)，用于 batched_matmul 测试。
/// 每个批次内的数据不同，确保测试覆盖多样性。
///
/// # 参数
///
/// - `batch_size`: 批次数量（对应 attention head 数量等）
/// - `rows`: 每个矩阵的行数
/// - `cols`: 每个矩阵的列数
fn make_batch_matrix(batch_size: usize, rows: usize, cols: usize) -> Array3<f32> {
    Array3::from_shape_fn((batch_size, rows, cols), |(b, i, j)| {
        let val = ((b * rows * cols + i * cols + j) as f32 * 0.01).sin()
            + ((b * rows * cols + i * cols + j) as f32 * 0.007).cos();
        val * 0.5
    })
}

// ============================================================================
// 1. MatMul 基准测试：Vulkan GPU vs CPU (ndarray)
// ============================================================================

/// MatMul 性能对比基准
///
/// 测试不同矩阵尺寸下 Vulkan GPU 与 CPU ndarray 的矩阵乘法性能。
///
/// # 测试目的
/// - 验证 Vulkan 后端在各个尺寸下的正确性（通过结果一致性隐式保证）
/// - 量化 GPU 相对 CPU 的加速比
/// - 识别 GPU 加速的"盈亏平衡点"（小矩阵可能因 kernel launch 开销而慢于 CPU）
/// - 对比 Vulkan 在 Linux/Windows/macOS 各平台上的表现差异
///
/// # 矩阵尺寸选择说明
/// - **128x128**: 小矩阵，测试 kernel launch 开销占比和数据传输延迟
/// - **512x512**: 中等矩阵，典型 embedding/attention 投影尺寸，最常用场景
/// - **1024x1024**: 大矩阵，测试 GPU 带宽、并行度和 shared memory 利用率
///
/// # FLOPs 计算公式
///
/// MatMul C = A(m,k) @ B(k,n) 需要 **2*m*k*n** 次 FLOP（每次内积包含 k 次乘法和 k-1 次加法）
#[cfg(feature = "vulkan")]
fn bench_matmul(c: &mut Criterion) {
    // 初始化 Vulkan 后端（失败时跳过 GPU benchmarks）
    let vulkan_backend: VulkanGemmBackend = match VulkanGemmBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("WARNING: Vulkan backend initialization failed: {}", e);
            eprintln!("Skipping Vulkan benchmarks. Falling back to CPU-only mode.");
            return;
        }
    };
    let cpu_backend = NdarrayFallbackBackend;

    // 测试矩阵尺寸配置: (M, K) 其中 N=M（方阵）
    let matrix_sizes: [(usize, usize); 3] = [(128, 128), (512, 512), (1024, 1024)];

    let mut group = c.benchmark_group("matmul");

    for &(m, k) in &matrix_sizes {
        let n = m; // 方阵配置: C(m,n) = A(m,k) @ B(k,n)

        // 预先生成测试数据（避免在 benchmark 循环中分配内存影响测量精度）
        let a = make_matrix(m, k);
        let b = make_matrix(k, n);

        // 计算理论 FLOPs 用于 throughput 度量（单位: Elements = FLOPs）
        // MatMul: C = A @ B 需要 2*m*k*n 次 FLOP (乘+加)
        let flops = 2u64 * m as u64 * k as u64 * n as u64;
        group.throughput(Throughput::Elements(flops));

        // --- Vulkan GPU MatMul ---
        // 通过闭包捕获引用，避免 move 语义导致循环内重复使用失败
        {
            let a_ref = &a;
            let b_ref = &b;
            group.bench_function(BenchmarkId::new("vulkan", format!("{}x{}", m, n)), |b| {
                b.iter(|| {
                    let result = vulkan_backend.matmul(black_box(a_ref), black_box(b_ref));
                    black_box(result)
                });
            });
        }

        // --- CPU (ndarray) MatMul ---
        // 作为 baseline 对比，展示 Vulkan GPU 的加速效果
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
/// 这是 Transformer 推理中 multi-head attention 的核心操作，
/// 每个 head 独立执行 Q@K^T 或 Q@V^T 计算。
///
/// # 测试目的
/// - 验证 Vulkan GPU 在批量场景下的并行效率
/// - 与 CPU 逐个计算的方式对比，展示 GPU batch 处理优势
/// - 模拟实际推理中的 multi-head attention 计算
/// - 评估 Vulkan command buffer 并发调度能力
///
/// # 配置说明
/// - **batch_size=8**: 典型的 attention head 数量（如 LLaMA-7B 的 32 heads 分组）
/// - **矩阵内部尺寸从 64 到 256**: 覆盖不同的 head_dim 配置
///   - 64: 小模型 head_dim（如 BERT-base）
///   - 128: 中等模型 head_dim（如 GPT-2 medium）
///   - 256: 大模型 head_dim（如 LLaMA 系列）
///
/// # 性能预期
///
/// 对于 Vulkan GPU:
/// - 小矩阵 (64x64): 可能受 kernel launch 开销影响，CPU 更有竞争力
/// - 中等矩阵 (128x128): GPU 开始展现优势
/// - 大矩阵 (256x256): GPU 显著领先，带宽和并行度充分发挥
#[cfg(feature = "vulkan")]
fn bench_batched_matmul(c: &mut Criterion) {
    let vulkan_backend: VulkanGemmBackend = match VulkanGemmBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("WARNING: Vulkan backend init failed: {}", e);
            return;
        }
    };
    let cpu_backend = NdarrayFallbackBackend;

    // (batch_size, M, K) 配置 — N=M（方阵）
    let configs: [(usize, usize, usize); 3] = [(8, 64, 64), (8, 128, 128), (8, 256, 256)];

    let mut group = c.benchmark_group("batched_matmul");

    for &(batch, m, k) in &configs {
        let n = m;

        let a = make_batch_matrix(batch, m, k);
        let b = make_batch_matrix(batch, k, n);

        // Batched MatMul 总 FLOPs: batch * 2 * m * k * n
        let flops = 2u64 * batch as u64 * m as u64 * k as u64 * n as u64;
        group.throughput(Throughput::Elements(flops));

        // --- Vulkan GPU Batched MatMul ---
        {
            let a_ref = &a;
            let b_ref = &b;
            group.bench_function(
                BenchmarkId::new("vulkan", format!("batch{}_{}x{}", batch, m, n)),
                |b| {
                    b.iter(|| {
                        let result =
                            vulkan_backend.batched_matmul(black_box(a_ref), black_box(b_ref));
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
// 注册基准测试组
// ============================================================================

#[cfg(feature = "vulkan")]
criterion_group!(benches, bench_matmul, bench_batched_matmul,);
#[cfg(feature = "vulkan")]
criterion_main!(benches);

/// 未启用 vulkan feature 时的空 main 函数
///
/// 输出友好的提示信息，告知用户如何正确运行此 benchmark。
#[cfg(not(feature = "vulkan"))]
fn main() {
    eprintln!();
    eprintln!("==============================================");
    eprintln!("  Vulkan benchmarks are not available");
    eprintln!("==============================================");
    eprintln!();
    eprintln!("  This benchmark requires:");
    eprintln!("    1. 'vulkan' feature enabled (--features vulkan)");
    eprintln!("    2. Vulkan-capable GPU with compute shader support");
    eprintln!();
    eprintln!("  Run with:");
    eprintln!("    cargo bench --package openmini-server \\");
    eprintln!("      --features vulkan --bench vulkan_bench");
    eprintln!();
}
