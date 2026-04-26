use ndarray::{Array1, Array2, Array3, s};

use super::{GemmBackendType, GemmEngine};
use crate::hardware::gpu::vulkan::VulkanGpu;
use crate::hardware::gpu::vulkan_compute::matrix_multiply_gpu;
use crate::model::inference::error::{InferenceError, InferenceResult};

pub struct VulkanGemmBackend {
    gpu: VulkanGpu,
}

impl VulkanGemmBackend {
    pub fn new() -> Result<Self, String> {
        let gpu = VulkanGpu::new(None).map_err(|e| format!("Vulkan GPU init failed: {}", e))?;
        Ok(Self { gpu })
    }

    fn array2_to_vec(a: &Array2<f32>) -> Vec<f32> {
        if a.is_standard_layout() {
            a.as_slice().unwrap().to_vec()
        } else {
            a.as_standard_layout().as_slice().unwrap().to_vec()
        }
    }
}

impl GemmEngine for VulkanGemmBackend {
    fn name(&self) -> &'static str {
        "vulkan_compute"
    }

    fn backend_type(&self) -> GemmBackendType {
        GemmBackendType::Vulkan
    }

    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> InferenceResult<Array2<f32>> {
        let a_data = Self::array2_to_vec(a);
        let b_data = Self::array2_to_vec(b);
        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[1];

        let result = matrix_multiply_gpu(&self.gpu, &a_data, &b_data, m, k, n)
            .map_err(|e| InferenceError::generation(format!("Vulkan matmul failed: {}", e)))?;

        Array2::from_shape_vec((m, n), result)
            .map_err(|e| InferenceError::generation(format!("Shape conversion failed: {}", e)))
    }

    fn batched_matmul(&self, a: &Array3<f32>, b: &Array3<f32>) -> InferenceResult<Array3<f32>> {
        if a.shape()[0] != b.shape()[0] {
            return Err(InferenceError::config(format!(
                "Batch dimension mismatch: {} vs {}",
                a.shape()[0],
                b.shape()[0]
            )));
        }

        let batch_size = a.shape()[0];
        let m = a.shape()[1];
        let k = a.shape()[2];
        let n = b.shape()[2];

        let mut results = Vec::with_capacity(batch_size * m * n);

        for i in 0..batch_size {
            let a_slice = a.slice(s![i, .., ..]);
            let b_slice = b.slice(s![i, .., ..]);

            let a_data = if a_slice.is_standard_layout() {
                a_slice.as_slice().unwrap().to_vec()
            } else {
                a_slice.as_standard_layout().as_slice().unwrap().to_vec()
            };
            let b_data = if b_slice.is_standard_layout() {
                b_slice.as_slice().unwrap().to_vec()
            } else {
                b_slice.as_standard_layout().as_slice().unwrap().to_vec()
            };

            let result = matrix_multiply_gpu(&self.gpu, &a_data, &b_data, m, k, n)
                .map_err(|e| InferenceError::generation(format!("Batch matmul[{}] failed: {}", i, e)))?;

            results.extend(result);
        }

        Array3::from_shape_vec((batch_size, m, n), results)
            .map_err(|e| InferenceError::generation(format!("Shape conversion failed: {}", e)))
    }

    fn fused_gemm_relu(
        &self,
        a: &Array2<f32>,
        w: &Array2<f32>,
        bias: Option<&Array1<f32>>,
    ) -> InferenceResult<Array2<f32>> {
        let mut result = self.matmul(a, w)?;

        if let Some(bias) = bias {
            for mut row in result.rows_mut() {
                row += bias;
            }
        }

        result.mapv_inplace(|x| x.max(0.0));
        Ok(result)
    }

    fn fused_gemm_silu(
        &self,
        x: &Array2<f32>,
        gate_w: &Array2<f32>,
        up_w: &Array2<f32>,
        bias: Option<&Array1<f32>>,
    ) -> InferenceResult<Array2<f32>> {
        let gate_result = self.matmul(x, gate_w)?;
        let up_result = self.matmul(x, up_w)?;

        let silu = gate_result.mapv(|x| x / (1.0 + (-x).exp()));
        let mut result = &up_result * &silu;

        if let Some(bias) = bias {
            for mut row in result.rows_mut() {
                row += bias;
            }
        }

        Ok(result)
    }

    fn is_available(&self) -> bool {
        cfg!(feature = "vulkan")
    }
}
