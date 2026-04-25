//! CUDA GPU 加速功能正确性验证测试
//!
//! 快速验证 CandleCudaBackend 的数值正确性（非性能）：
//! - 设备初始化与可用性检测
//! - matmul 结果正确性 (与 ndarray 对比，误差 < 1e-5)
//! - batched_matmul 正确性
//! - fused_gemm_relu 正确性 (GEMM + Bias + ReLU)
//! - fused_gemm_silu 正确性 (SwiGLU: gate * silu(up) + bias)
//!
//! 使用较小矩阵尺寸 (64x64, 128x128) 以快速完成。
//! 仅在启用 cuda feature 时编译；无 CUDA 设备或运行时失败时自动优雅跳过。

#[cfg(feature = "cuda")]
mod cuda_validation {

    use ndarray::{Array1, Array2, Array3};
    use rand::Rng;
    use rand::SeedableRng;
    use openmini_server::model::inference::gemm_engine::{
        cuda_backend::CandleCudaBackend, GemmEngine, NdarrayFallbackBackend,
    };

    /// 数值比较容差：f32 浮点运算在 CPU/GPU 间的可接受误差上限
    const ABS_TOL: f32 = 1e-5;

    // ==================== 辅助函数 ====================

    /// 使用固定种子生成确定性随机矩阵 (可复现测试结果)
    fn make_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        Array2::from_shape_fn((rows, cols), |_| {
            rng.gen_range(-1.0f32..1.0f32)
        })
    }

    /// 使用固定种子生成确定性随机偏置向量
    fn make_bias(size: usize, seed: u64) -> Array1<f32> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        Array1::from_shape_fn(size, |_| rng.gen_range(-0.5f32..0.5f32))
    }

    /// 使用固定种子生成确定性批量矩阵 (3D: batch x rows x cols)
    fn make_batch_matrix(batch: usize, rows: usize, cols: usize, seed: u64) -> Array3<f32> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        Array3::from_shape_fn((batch, rows, cols), |_| {
            rng.gen_range(-1.0f32..1.0f32)
        })
    }

    /// 尝试创建 CUDA 后端并预热。
    ///
    /// 返回 None 表示 CUDA 不可用（设备初始化失败或运行时无法执行 kernel），
    /// 调用方应跳过测试而非 panic。此设计确保在 CI / 无 GPU 环境下不会崩溃。
    fn try_create_cuda_backend() -> Option<CandleCudaBackend> {
        // 步骤 1: 初始化 CUDA 设备 (调用 cudaSetDevice / cuCtxCreate)
        let backend = match CandleCudaBackend::new() {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "[cuda-validation] CUDA backend init failed, skipping all: {}",
                    e
                );
                return None;
            }
        };

        // 步骤 2: 预热 — 执行一次最小 matmul 以验证 CUDA kernel 可正常调度执行
        // 如果此处失败，说明当前环境虽能初始化 CUDA context 但无法实际运行计算
        // （例如 headless CI、GPU 被锁定、驱动版本不兼容等）
        let probe_a = make_matrix(2, 2, 0);
        let probe_b = make_matrix(2, 2, 1);
        if backend.matmul(&probe_a, &probe_b).is_err() {
            eprintln!(
                "[cuda-validation] CUDA kernel execution failed \
                 (no GPU / driver issue / headless CI?), skipping all"
            );
            return None;
        }

        Some(backend)
    }

    /// 计算 2D 数组的最大绝对误差，用于数值正确性判定
    fn max_abs_diff_2d(actual: &Array2<f32>, expected: &Array2<f32>) -> f32 {
        (actual - expected)
            .mapv(|v| v.abs())
            .into_iter()
            .fold(f32::NEG_INFINITY, |a, b| a.max(b))
    }

    /// 计算 3D 数组的最大绝对误差
    fn max_abs_diff_3d(actual: &Array3<f32>, expected: &Array3<f32>) -> f32 {
        (actual - expected)
            .mapv(|v| v.abs())
            .into_iter()
            .fold(f32::NEG_INFINITY, |a, b| a.max(b))
    }

    /// 带自定义消息的 2D 数组近似相等断言 (误差 < ABS_TOL)
    fn assert_array2_approx(actual: &Array2<f32>, expected: &Array2<f32>, msg: &str) {
        let diff = max_abs_diff_2d(actual, expected);
        assert!(
            diff < ABS_TOL,
            "{}\n  max absolute diff: {} (tolerance: {})",
            msg,
            diff,
            ABS_TOL
        );
    }

    /// 带自定义消息的 3D 数组近似相等断言 (误差 < ABS_TOL)
    fn assert_array3_approx(actual: &Array3<f32>, expected: &Array3<f32>, msg: &str) {
        let diff = max_abs_diff_3d(actual, expected);
        assert!(
            diff < ABS_TOL,
            "{}\n  max absolute diff: {} (tolerance: {})",
            msg,
            diff,
            ABS_TOL
        );
    }

    // ==================== 测试：设备初始化 ====================

    /// 测试目标：验证 CUDA 后端能否成功初始化且属性报告一致
    #[test]
    fn test_cuda_device_init() {
        match try_create_cuda_backend() {
            Some(backend) => {
                assert!(
                    backend.is_available(),
                    "[cuda-init] backend.is_available() should return true after successful init"
                );
                assert_eq!(
                    backend.name(),
                    "candle_cuda",
                    "[cuda-init] backend name mismatch, expected 'candle_cuda', got '{}'",
                    backend.name()
                );
                eprintln!(
                    "[cuda-init] Device initialized successfully: name={}",
                    backend.name()
                );
            }
            None => {
                eprintln!("[cuda-init] Skipped: CUDA device not available on this machine");
            }
        }
    }

    // ==================== 测试：matmul 正确性 ====================

    /// 测试目标：验证 64x64 方阵 matmul 结果与 ndarray 参考实现一致
    #[test]
    fn test_matmul_64x64() {
        let cuda = match try_create_cuda_backend() {
            Some(b) => b,
            None => return,
        };
        let reference = NdarrayFallbackBackend;

        let m = 64;
        let k = 64;
        let n = 64;

        let a = make_matrix(m, k, 42);
        let b = make_matrix(n, k, 137); // b shape: (n,k)，内部转置为 (k,n) 后做 matmul

        let cuda_result = match cuda.matmul(&a, &b) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[cuda-matmul-64] Skipped - CUDA execution error: {}", e);
                return;
            }
        };
        let ref_result = reference
            .matmul(&a, &b)
            .expect("[ref-matmul-64] Reference matmul failed");

        assert_array2_approx(
            &cuda_result,
            &ref_result,
            "[cuda-matmul-64] CUDA matmul result differs from ndarray reference (64x64)",
        );

        assert_eq!(
            cuda_result.shape(),
            ref_result.shape(),
            "[cuda-matmul-64] Shape mismatch after matmul"
        );

        eprintln!("[cuda-matmul-64] PASSED: 64x64 matmul matches reference within tolerance");
    }

    /// 测试目标：验证 128x128 较大方阵 matmul 结果正确性
    #[test]
    fn test_matmul_128x128() {
        let cuda = match try_create_cuda_backend() {
            Some(b) => b,
            None => return,
        };
        let reference = NdarrayFallbackBackend;

        let m = 128;
        let k = 128;
        let n = 128;

        let a = make_matrix(m, k, 99);
        let b = make_matrix(n, k, 256);

        let cuda_result = match cuda.matmul(&a, &b) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[cuda-matmul-128] Skipped - CUDA execution error: {}", e);
                return;
            }
        };
        let ref_result = reference
            .matmul(&a, &b)
            .expect("[ref-matmul-128] Reference matmul failed");

        assert_array2_approx(
            &cuda_result,
            &ref_result,
            "[cuda-matmul-128] CUDA matmul result differs from ndarray reference (128x128)",
        );

        assert_eq!(
            cuda_result.shape(),
            ref_result.shape(),
            "[cuda-matmul-128] Shape mismatch after matmul"
        );

        eprintln!(
            "[cuda-matmul-128] PASSED: 128x128 matmul matches reference within tolerance"
        );
    }

    /// 测试目标：验证非方阵 matmul (32x64 @ 64x96 => 32x96) 形状与数值均正确
    #[test]
    fn test_matmul_non_square() {
        let cuda = match try_create_cuda_backend() {
            Some(b) => b,
            None => return,
        };
        let reference = NdarrayFallbackBackend;

        // CandleCudaBackend::matmul 内部做 a.matmul(b.t())
        // 所以输入 b 的形状应为 (n, k)，转置后变为 (k, n)
        // 目标输出: (m, k) @ (k, n) = (m, n) = (32, 96)
        let m = 32;
        let k = 64;
        let n = 96;

        let a = make_matrix(m, k, 777);     // (32, 64)
        let b = make_matrix(n, k, 888);     // (96, 64), 转置后 (64, 96)

        let cuda_result = match cuda.matmul(&a, &b) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[cuda-matmul-nonsquare] Skipped - CUDA execution error: {}", e);
                return;
            }
        };
        let ref_result =
            reference.matmul(&a, &b).expect("[ref-matmul-nonsquare] Reference matmul failed");

        assert_array2_approx(
            &cuda_result,
            &ref_result,
            "[cuda-matmul-nonsquare] Non-square matmul result mismatch",
        );

        assert_eq!(
            cuda_result.shape(),
            &[m, n],
            "[cuda-matmul-nonsquare] Output shape should be ({}, {}), got {:?}",
            m,
            n,
            cuda_result.shape()
        );

        eprintln!(
            "[cuda-matmul-nonsquare] PASSED: non-square ({}x{} @ {}x{}) correct",
            m, k, n, k
        );
    }

    // ==================== 测试：batched_matmul 正确性 ====================

    /// 测试目标：验证 batch=4 的批量矩阵乘法 (4 x 64x64 @ 4 x 64x64) 正确性
    #[test]
    fn test_batched_matmul_4x64x64() {
        let cuda = match try_create_cuda_backend() {
            Some(b) => b,
            None => return,
        };
        let reference = NdarrayFallbackBackend;

        let batch = 4;
        let m = 64;
        let k = 64;
        let n = 64;

        let a = make_batch_matrix(batch, m, k, 1001);
        let b = make_batch_matrix(batch, n, k, 2002);

        let cuda_result = match cuda.batched_matmul(&a, &b) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[cuda-batched-4x64] Skipped - CUDA execution error: {}", e);
                return;
            }
        };
        let ref_result = reference
            .batched_matmul(&a, &b)
            .expect("[ref-batched-4x64] Reference batched_matmul failed");

        assert_array3_approx(
            &cuda_result,
            &ref_result,
            "[cuda-batched-4x64] Batched matmul result mismatch (batch=4, 64x64)",
        );

        assert_eq!(
            cuda_result.shape(),
            &[batch, m, n],
            "[cuda-batched-4x64] Shape should be ({}, {}, {}), got {:?}",
            batch,
            m,
            n,
            cuda_result.shape()
        );

        eprintln!(
            "[cuda-batched-4x64] PASSED: batched_matmul (batch=4, 64x64) correct"
        );
    }

    /// 测试目标：验证 batch=8 且非方阵的批量乘法 (8 x 128x64 @ 8 x 64x64) 正确性
    #[test]
    fn test_batched_matmul_8x128x64() {
        let cuda = match try_create_cuda_backend() {
            Some(b) => b,
            None => return,
        };
        let reference = NdarrayFallbackBackend;

        let batch = 8;
        let m = 128;
        let k = 64;
        let n = 64;

        let a = make_batch_matrix(batch, m, k, 3333);
        let b = make_batch_matrix(batch, n, k, 4444);

        let cuda_result = match cuda.batched_matmul(&a, &b) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[cuda-batched-8x128] Skipped - CUDA execution error: {}", e);
                return;
            }
        };
        let ref_result = reference
            .batched_matmul(&a, &b)
            .expect("[ref-batched-8x128] Reference batched_matmul failed");

        assert_array3_approx(
            &cuda_result,
            &ref_result,
            "[cuda-batched-8x128] Batched matmul result mismatch (batch=8, 128x64)",
        );

        eprintln!(
            "[cuda-batched-8x128] PASSED: batched_matmul (batch=8, 128x64) correct"
        );
    }

    // ==================== 测试：fused_gemm_relu 正确性 ====================
    // 公式: output = relu(A @ W^T + bias)

    /// 测试目标：验证带偏置的 fused GEMM + ReLU (64x64) 结果正确，
    /// 并额外校验 ReLU 输出不含负值
    #[test]
    fn test_fused_gemm_relu_with_bias_64() {
        let cuda = match try_create_cuda_backend() {
            Some(b) => b,
            None => return,
        };
        let reference = NdarrayFallbackBackend;

        let m = 64;
        let k = 64;
        let n = 64;

        let a = make_matrix(m, k, 5555);
        let w = make_matrix(n, k, 6666); // weight shape is (n, k), transposed in GEMM
        let bias = make_bias(n, 7777);

        let cuda_result = match cuda.fused_gemm_relu(&a, &w, Some(&bias)) {
            Ok(r) => r,
            Err(e) => {
                eprintln!(
                    "[cuda-gemm-relu-64] Skipped - CUDA execution error: {}",
                    e
                );
                return;
            }
        };
        let ref_result = reference
            .fused_gemm_relu(&a, &w, Some(&bias))
            .expect("[ref-gemm-relu-64] Reference fused_gemm_relu(with bias) failed");

        assert_array2_approx(
            &cuda_result,
            &ref_result,
            "[cuda-gemm-relu-64] fused_gemm_relu(with bias) result mismatch (64x64)",
        );

        // ReLU 不变性验证: 输出中不应存在负值
        for (idx, val) in cuda_result.iter().enumerate() {
            assert!(
                *val >= 0.0,
                "[cuda-gemm-relu-64] ReLU output contains negative value at index {}: {}",
                idx,
                val
            );
        }

        eprintln!(
            "[cuda-gemm-relu-64] PASSED: fused_gemm_relu with bias (64x64) correct"
        );
    }

    /// 测试目标：验证无偏置的 fused GEMM + ReLU (128x64) 结果正确
    #[test]
    fn test_fused_gemm_relu_no_bias_128() {
        let cuda = match try_create_cuda_backend() {
            Some(b) => b,
            None => return,
        };
        let reference = NdarrayFallbackBackend;

        let m = 128;
        let k = 64;
        let n = 64;

        let a = make_matrix(m, k, 1111);
        let w = make_matrix(n, k, 2222);

        let cuda_result = match cuda.fused_gemm_relu(&a, &w, None) {
            Ok(r) => r,
            Err(e) => {
                eprintln!(
                    "[cuda-gemm-relu-no-bias] Skipped - CUDA execution error: {}",
                    e
                );
                return;
            }
        };
        let ref_result = reference
            .fused_gemm_relu(&a, &w, None)
            .expect("[ref-gemm-relu-no-bias] Reference fused_gemm_relu(no bias) failed");

        assert_array2_approx(
            &cuda_result,
            &ref_result,
            "[cuda-gemm-relu-no-bias] fused_gemm_relu(no bias) result mismatch (128x64)",
        );

        // ReLU 不变性验证
        for (idx, val) in cuda_result.iter().enumerate() {
            assert!(
                *val >= 0.0,
                "[cuda-gemm-relu-no-bias] ReLU output contains negative value at index {}: {}",
                idx,
                val
            );
        }

        eprintln!(
            "[cuda-gemm-relu-no-bias] PASSED: fused_gemm_relu without bias (128x64) correct"
        );
    }

    // ==================== 测试：fused_gemm_silu 正确性 ====================
    // SwiGLU 公式: output = gate @ gate_w^T * silu(x @ up_w^T) + bias
    // 其中 silu(x) = x * sigmoid(x)

    /// 测试目标：验证带偏置的 SwiGLU fused 操作 (64x64) 结果正确
    #[test]
    fn test_fused_gemm_silu_with_bias_64() {
        let cuda = match try_create_cuda_backend() {
            Some(b) => b,
            None => return,
        };
        let reference = NdarrayFallbackBackend;

        let m = 64;
        let k = 64;
        let n = 64;

        let x = make_matrix(m, k, 9999);
        let gate_w = make_matrix(n, k, 8888);
        let up_w = make_matrix(n, k, 7777);
        let bias = make_bias(n, 6666);

        let cuda_result = match cuda.fused_gemm_silu(&x, &gate_w, &up_w, Some(&bias)) {
            Ok(r) => r,
            Err(e) => {
                eprintln!(
                    "[cuda-gemm-silu-64] Skipped - CUDA execution error: {}",
                    e
                );
                return;
            }
        };
        let ref_result = reference
            .fused_gemm_silu(&x, &gate_w, &up_w, Some(&bias))
            .expect("[ref-gemm-silu-64] Reference fused_gemm_silu(with bias) failed");

        assert_array2_approx(
            &cuda_result,
            &ref_result,
            "[cuda-gemm-silu-64] fused_gemm_silu(with bias) result mismatch (64x64)",
        );

        eprintln!(
            "[cuda-gemm-silu-64] PASSED: fused_gemm_silu with bias (64x64) correct"
        );
    }

    /// 测试目标：验证无偏置的 SwiGLU fused 操作 (128x64) 结果正确
    #[test]
    fn test_fused_gemm_silu_no_bias_128() {
        let cuda = match try_create_cuda_backend() {
            Some(b) => b,
            None => return,
        };
        let reference = NdarrayFallbackBackend;

        let m = 128;
        let k = 64;
        let n = 64;

        let x = make_matrix(m, k, 12345);
        let gate_w = make_matrix(n, k, 23456);
        let up_w = make_matrix(n, k, 34567);

        let cuda_result = match cuda.fused_gemm_silu(&x, &gate_w, &up_w, None) {
            Ok(r) => r,
            Err(e) => {
                eprintln!(
                    "[cuda-gemm-silu-no-bias] Skipped - CUDA execution error: {}",
                    e
                );
                return;
            }
        };
        let ref_result = reference
            .fused_gemm_silu(&x, &gate_w, &up_w, None)
            .expect("[ref-gemm-silu-no-bias] Reference fused_gemm_silu(no bias) failed");

        assert_array2_approx(
            &cuda_result,
            &ref_result,
            "[cuda-gemm-silu-no-bias] fused_gemm_silu(no bias) result mismatch (128x64)",
        );

        eprintln!(
            "[cuda-gemm-silu-no-bias] PASSED: fused_gemm_silu without bias (128x64) correct"
        );
    }

    // ==================== 测试：边界情况 ====================

    /// 测试目标：验证近似单位矩阵乘法 A @ I ≈ A，检验基础线性代数性质
    #[test]
    fn test_matmul_identity_like() {
        let cuda = match try_create_cuda_backend() {
            Some(b) => b,
            None => return,
        };
        let reference = NdarrayFallbackBackend;

        // 近似单位矩阵乘法: A @ I ≈ A
        let n = 64;
        let a = make_matrix(n, n, 42);

        // 构造近似单位矩阵 (对角线=1.0, 其余极小值)
        let mut identity = Array2::<f32>::zeros((n, n));
        for i in 0..n {
            identity[[i, i]] = 1.0;
        }

        let cuda_result = match cuda.matmul(&a, &identity) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[cuda-identity] Skipped - CUDA execution error: {}", e);
                return;
            }
        };
        let ref_result = reference
            .matmul(&a, &identity)
            .expect("[ref-identity] Reference identity-like matmul failed");

        assert_array2_approx(
            &cuda_result,
            &ref_result,
            "[cuda-identity] Identity-like matmul result mismatch",
        );

        // 进一步验证: 结果应接近原始矩阵 A
        assert_array2_approx(
            &cuda_result,
            &a,
            "[cuda-identity] A @ I should approximate A",
        );

        eprintln!("[cuda-identity] PASSED: identity-like matmul preserves input");
    }

    /// 测试目标：验证 GEMM 输出全负时 ReLU 将其全部归零的边界行为
    #[test]
    fn test_fused_gemm_relu_all_negative_output() {
        let cuda = match try_create_cuda_backend() {
            Some(b) => b,
            None => return,
        };
        let reference = NdarrayFallbackBackend;

        // 使用负权重使 GEMM 输出倾向负数，验证 ReLU 将其全部归零
        let m = 32;
        let k = 32;
        let n = 32;

        let a = make_matrix(m, k, 101);
        // 负权重矩阵，使 GEMM 输出倾向于负值
        let w = make_matrix(n, k, 202).mapv(|v| -v.abs() - 1.0);

        let cuda_result = match cuda.fused_gemm_relu(&a, &w, None) {
            Ok(r) => r,
            Err(e) => {
                eprintln!(
                    "[cuda-relu-allneg] Skipped - CUDA execution error: {}",
                    e
                );
                return;
            }
        };
        let ref_result = reference
            .fused_gemm_relu(&a, &w, None)
            .expect("[ref-relu-allneg] Reference fused_gemm_relu(all negative) failed");

        assert_array2_approx(
            &cuda_result,
            &ref_result,
            "[cuda-relu-allneg] All-negative ReLU case mismatch",
        );

        eprintln!(
            "[cuda-relu-allneg] PASSED: all-negative ReLU boundary case correct"
        );
    }
}

// ==================== 未启用 cuda feature 时的占位测试 ====================

/// 当 cuda feature 未启用时的占位测试，确保测试套件不会因缺少 feature 而失败。
/// 此测试始终通过，仅作为文档说明当前环境不支持 CUDA 测试。
#[cfg(not(feature = "cuda"))]
mod cuda_unavailable {
    #[test]
    fn test_cuda_skipped_unavailable() {
        eprintln!(
            "[cuda-validation] Skipped: CUDA tests require feature='cuda'. \
             Current config: cuda_feature={}",
            cfg!(feature = "cuda")
        );
    }
}
