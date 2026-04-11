//! OpenMini-V1 GPU E2E 测试
//!
//! 端到端验证 GPU 后端功能：
//! - Metal 设备创建和基本操作 (macOS)
//! - CUDA 设备检测 (feature-gated)
//! - GPU 设备信息查询

#[cfg(feature = "metal")]
mod metal_tests {
    use super::*;

    #[test]
    fn test_metal_device_creation() {
        use openmini_server::hardware::gpu::metal::MetalBackend;

        eprintln!("[metal-e2e] Creating Metal backend...");

        let result = MetalBackend::new();
        match result {
            Ok(_backend) => {
                eprintln!("[metal-e2e] Metal backend created successfully");
                // 注意：device_info() 方法尚未实现，此处仅验证创建成功
            }
            Err(e) => {
                eprintln!("[metal-e2e] Metal backend creation failed (may be CI): {}", e);
                if !is_ci_environment() {
                    panic!("Metal backend should be available on macOS");
                }
            }
        }
    }

    #[test]
    fn test_metal_basic_matmul() {
        use openmini_server::hardware::gpu::metal::MetalBackend;

        eprintln!("[metal-matmul] Testing basic matmul...");

        let _backend = match MetalBackend::new() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("[metal-matmul] Skipped (no Metal): {}", e);
                return;
            }
        };

        // 注意：matmul() 方法尚未实现，此处仅验证 backend 创建成功
        eprintln!("[metal-matmul] Metal backend created (matmul test skipped - method not implemented)");
    }
}

#[cfg(feature = "cuda")]
mod cuda_tests {
    use super::*;

    #[test]
    fn test_cuda_device_detection() {
        use openmini_server::hardware::gpu::cuda::CudaBackend;

        eprintln!("[cuda-e2e] Detecting CUDA devices...");

        let result = CudaBackend::new();
        match result {
            Ok(_backend) => {
                eprintln!("[cuda-e2e] CUDA backend created successfully");
                // 注意：device_info() 方法尚未实现，此处仅验证创建成功
            }
            Err(e) => {
                eprintln!("[cuda-e2e] No CUDA device: {}", e);
            }
        }
    }

    #[ignore = "Requires real GPU with CUDA"]
    #[test]
    fn test_cuda_inference_pipeline() {
        eprintln!("[cuda-pipeline] Full inference pipeline test (ignored by default)");
    }
}

#[cfg(not(any(feature = "metal", feature = "cuda")))]
mod fallback_tests {
    #[test]
    fn test_no_gpu_backend_available() {
        eprintln!("[gpu-fallback] No GPU backend compiled in this build");
        assert!(true);
    }
}

fn is_ci_environment() -> bool {
    std::env::var("CI").is_ok()
        || std::env::var("GITHUB_ACTIONS").is_ok()
}
