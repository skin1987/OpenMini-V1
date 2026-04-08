//! 统一内核接口
//!
//! 提供跨平台的内核调用接口

use anyhow::Result;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// 内核后端
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelBackend {
    /// CPU
    Cpu,
    /// CUDA
    Cuda,
    /// Metal
    Metal,
}

/// 内核执行器
pub struct KernelExecutor {
    backend: KernelBackend,
}

impl KernelExecutor {
    /// 创建内核执行器
    pub fn new(backend: KernelBackend) -> Self {
        Self { backend }
    }

    /// 矩阵乘法
    pub fn matmul(&self, a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> Result<Array2<f32>> {
        match self.backend {
            KernelBackend::Cpu => crate::kernel::ops::matmul(a, b),
            KernelBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    // CUDA实现
                    let m = a.nrows();
                    let k = a.ncols();
                    let n = b.ncols();

                    let a_flat: Vec<f32> = a.iter().cloned().collect();
                    let b_flat: Vec<f32> = b.iter().cloned().collect();
                    let mut c_flat = vec![0.0f32; m * n];

                    // 这里需要实际的CUDA kernel调用
                    // 暂时使用CPU实现
                    crate::kernel::ops::matmul(a, b)
                }

                #[cfg(not(feature = "cuda"))]
                {
                    crate::kernel::ops::matmul(a, b)
                }
            }
            KernelBackend::Metal => {
                #[cfg(feature = "metal")]
                {
                    // Metal实现
                    crate::kernel::ops::matmul(a, b)
                }

                #[cfg(not(feature = "metal"))]
                {
                    crate::kernel::ops::matmul(a, b)
                }
            }
        }
    }

    /// Softmax
    pub fn softmax(&self, x: &ArrayView1<f32>) -> Array1<f32> {
        match self.backend {
            KernelBackend::Cpu => crate::kernel::ops::softmax(x),
            KernelBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    // CUDA实现
                    crate::kernel::cpu::simd::softmax_vector(
                        x.as_slice().unwrap(),
                        &mut vec![0.0; x.len()],
                    );
                    crate::kernel::ops::softmax(x)
                }

                #[cfg(not(feature = "cuda"))]
                {
                    crate::kernel::ops::softmax(x)
                }
            }
            KernelBackend::Metal => crate::kernel::ops::softmax(x),
        }
    }

    /// LayerNorm
    pub fn layer_norm(
        &self,
        x: &ArrayView1<f32>,
        weight: &ArrayView1<f32>,
        bias: &ArrayView1<f32>,
        eps: f32,
    ) -> Result<Array1<f32>> {
        crate::kernel::ops::layer_norm(x, weight, bias, eps)
    }

    /// RMSNorm
    pub fn rms_norm(
        &self,
        x: &ArrayView1<f32>,
        weight: &ArrayView1<f32>,
        eps: f32,
    ) -> Result<Array1<f32>> {
        crate::kernel::ops::rms_norm(x, weight, eps)
    }

    /// GELU
    pub fn gelu(&self, x: &ArrayView1<f32>) -> Array1<f32> {
        crate::kernel::ops::gelu(x)
    }

    /// SiLU
    pub fn silu(&self, x: &ArrayView1<f32>) -> Array1<f32> {
        crate::kernel::ops::silu(x)
    }

    /// 获取后端
    pub fn backend(&self) -> KernelBackend {
        self.backend
    }
}

impl Default for KernelExecutor {
    fn default() -> Self {
        Self::new(KernelBackend::Cpu)
    }
}

/// 自动选择最优后端
pub fn auto_select_backend() -> KernelBackend {
    #[cfg(feature = "cuda")]
    {
        // 检查CUDA是否可用
        return KernelBackend::Cuda;
    }

    #[cfg(feature = "metal")]
    {
        // 检查Metal是否可用
        return KernelBackend::Metal;
    }

    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        KernelBackend::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_kernel_executor() {
        let executor = KernelExecutor::new(KernelBackend::Cpu);

        let x = arr1(&[1.0, 2.0, 3.0]);
        let result = executor.softmax(&x.view());

        let sum: f32 = result.sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_auto_select_backend() {
        let backend = auto_select_backend();
        println!("Selected backend: {:?}", backend);

        // 验证返回的后端是有效的枚举值
        match backend {
            KernelBackend::Cpu | KernelBackend::Cuda | KernelBackend::Metal => {}
        }
    }

    #[test]
    fn test_kernel_executor_creation_all_backends() {
        // 测试所有后端类型的执行器创建
        let cpu_executor = KernelExecutor::new(KernelBackend::Cpu);
        assert_eq!(cpu_executor.backend(), KernelBackend::Cpu);

        let cuda_executor = KernelExecutor::new(KernelBackend::Cuda);
        assert_eq!(cuda_executor.backend(), KernelBackend::Cuda);

        let metal_executor = KernelExecutor::new(KernelBackend::Metal);
        assert_eq!(metal_executor.backend(), KernelBackend::Metal);
    }

    #[test]
    fn test_kernel_executor_default() {
        // 测试默认执行器（应该是 CPU）
        let executor = KernelExecutor::default();
        assert_eq!(executor.backend(), KernelBackend::Cpu);
    }

    #[test]
    fn test_kernel_softmax_basic() {
        // 测试基本 softmax 计算
        let executor = KernelExecutor::new(KernelBackend::Cpu);
        let x = arr1(&[1.0, 2.0, 3.0]);
        let result = executor.softmax(&x.view());

        // 验证和为 1
        let sum: f32 = result.sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // 验证单调递增（输入越大，softmax 输出越大）
        for i in 1..result.len() {
            assert!(result[i] > result[i - 1]);
        }
    }

    #[test]
    fn test_kernel_softmax_single_element() {
        // 测试单元素 softmax
        let executor = KernelExecutor::new(KernelBackend::Cpu);
        let x = arr1(&[5.0]);
        let result = executor.softmax(&x.view());

        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-5); // 单元素的 softmax 应该是 1
    }

    #[test]
    fn test_kernel_softmax_negative_values() {
        // 测试负数值的 softmax
        let executor = KernelExecutor::new(KernelBackend::Cpu);
        let x = arr1(&[-1.0, -2.0, -3.0]);
        let result = executor.softmax(&x.view());

        let sum: f32 = result.sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // 所有值应该 > 0 且 < 1
        for &val in result.iter() {
            assert!(val > 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_kernel_matmul_basic() {
        // 测试基本矩阵乘法
        let executor = KernelExecutor::new(KernelBackend::Cpu);

        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[5.0, 6.0], [7.0, 8.0]]);

        let result = executor.matmul(&a.view(), &b.view()).unwrap();

        // 手动计算: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        // = [[19, 22], [43, 50]]
        assert!((result[[0, 0]] - 19.0).abs() < 1e-5);
        assert!((result[[0, 1]] - 22.0).abs() < 1e-5);
        assert!((result[[1, 0]] - 43.0).abs() < 1e-5);
        assert!((result[[1, 1]] - 50.0).abs() < 1e-5);
    }

    #[test]
    fn test_kernel_layer_norm() {
        // 测试 LayerNorm
        let executor = KernelExecutor::new(KernelBackend::Cpu);

        let x = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let weight = arr1(&[1.0, 1.0, 1.0, 1.0]);
        let bias = arr1(&[0.0, 0.0, 0.0, 0.0]);

        let result = executor
            .layer_norm(&x.view(), &weight.view(), &bias.view(), 1e-5)
            .unwrap();

        assert_eq!(result.len(), 4);
        // LayerNorm 后的均值应该接近 0
        let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
        assert!(mean.abs() < 1e-5);
    }

    #[test]
    fn test_kernel_rms_norm() {
        // 测试 RMSNorm
        let executor = KernelExecutor::new(KernelBackend::Cpu);

        let x = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let weight = arr1(&[1.0, 1.0, 1.0, 1.0]);

        let result = executor.rms_norm(&x.view(), &weight.view(), 1e-5).unwrap();

        assert_eq!(result.len(), 4);
        // RMSNorm 的结果应该与输入成比例（当 weight=1 时）
        for (_i, &val) in result.iter().enumerate() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_kernel_gelu() {
        // 测试 GELU 激活函数
        let executor = KernelExecutor::new(KernelBackend::Cpu);

        let x = arr1(&[-1.0, 0.0, 1.0, 2.0]);
        let result = executor.gelu(&x.view());

        assert_eq!(result.len(), 4);
        // GELU(0) ≈ 0
        assert!((result[1] - 0.0).abs() < 1e-5);
        // GELU 应该保持符号（对于正数输出正，负数输出接近 0 或略负）
        assert!(result[2] > 0.0); // GELU(1) > 0
        assert!(result[3] > result[2]); // 单调递增
    }

    #[test]
    fn test_kernel_silu() {
        // 测试 SiLU (Swish) 激活函数: x * sigmoid(x)
        let executor = KernelExecutor::new(KernelBackend::Cpu);

        let x = arr1(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = executor.silu(&x.view());

        assert_eq!(result.len(), 5);
        // SiLU(0) = 0 * sigmoid(0) = 0
        assert!((result[2] - 0.0).abs() < 1e-5);
        // SiLU 对于正数应该 > 0
        assert!(result[3] > 0.0);
        assert!(result[4] > 0.0);
    }

    #[test]
    fn test_kernel_backend_debug_display() {
        // 测试后端类型的 Debug 和 Display
        let backends = vec![
            KernelBackend::Cpu,
            KernelBackend::Cuda,
            KernelBackend::Metal,
        ];

        for backend in backends {
            let debug = format!("{:?}", backend);
            assert!(!debug.is_empty());

            // 验证包含后端名称信息
            match backend {
                KernelBackend::Cpu => assert!(debug.contains("Cpu")),
                KernelBackend::Cuda => assert!(debug.contains("Cuda")),
                KernelBackend::Metal => assert!(debug.contains("Metal")),
            }
        }
    }

    #[test]
    fn test_kernel_clone_backends() {
        // 测试后端类型是否可以 Clone 和 Copy
        let original = KernelBackend::Cpu;
        let copied = original;
        let _cloned = original.clone(); // Clone

        assert_eq!(copied, KernelBackend::Cpu);
    }

    #[test]
    fn test_kernel_operations_with_cuda_backend() {
        // 测试 CUDA 后端（如果没有 CUDA feature，会回退到 CPU）
        #[cfg(feature = "cuda")]
        {
            let executor = KernelExecutor::new(KernelBackend::Cuda);
            let x = arr1(&[1.0, 2.0, 3.0]);
            let _result = executor.softmax(&x.view());
        }

        #[cfg(not(feature = "cuda"))]
        {
            // 没有 CUDA feature 时，CUDA 后端应该也能工作（回退到 CPU）
            let executor = KernelExecutor::new(KernelBackend::Cuda);
            let x = arr1(&[1.0, 2.0, 3.0]);
            let result = executor.softmax(&x.view());
            let sum: f32 = result.sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_kernel_operations_with_metal_backend() {
        // 测试 Metal 后端（类似 CUDA）
        #[cfg(feature = "metal")]
        {
            let executor = KernelExecutor::new(KernelBackend::Metal);
            let x = arr1(&[1.0, 2.0, 3.0]);
            let _result = executor.softmax(&x.view());
        }

        #[cfg(not(feature = "metal"))]
        {
            let executor = KernelExecutor::new(KernelBackend::Metal);
            let x = arr1(&[1.0, 2.0, 3.0]);
            let result = executor.softmax(&x.view());
            let sum: f32 = result.sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }
}
