use candle_core::{Device, DType, Result as CandleResult, Tensor};
use ndarray::{Array1, Array2, Array3};

use super::{GemmBackendType, GemmEngine};
use crate::model::inference::error::{InferenceError, InferenceResult};

pub struct CandleMetalBackend {
    device: Device,
}

impl CandleMetalBackend {
    pub fn new() -> Result<Self, String> {
        let device = Device::new_metal(0).map_err(|e| format!("Metal device init failed: {}", e))?;
        Ok(Self { device })
    }

    fn array2_to_tensor(&self, a: &Array2<f32>) -> CandleResult<Tensor> {
        let shape = a.shape();
        let data = if a.is_standard_layout() {
            a.as_slice().unwrap().to_vec()
        } else {
            a.as_standard_layout().as_slice().unwrap().to_vec()
        };
        Tensor::from_vec(data, shape, &self.device)
    }

    fn tensor_to_array2(t: &Tensor) -> InferenceResult<Array2<f32>> {
        let vals: Vec<f32> = t
            .to_vec2::<f32>()
            .map_err(|e| InferenceError::generation(format!("Tensor to vec2 failed: {}", e)))?
            .into_iter()
            .flatten()
            .collect();
        let shape = t.dims().to_vec();
        Array2::from_shape_vec((shape[0], shape[1]), vals)
            .map_err(|e| InferenceError::generation(format!("Shape conversion failed: {}", e)))
    }

    fn array3_to_tensor(&self, a: &Array3<f32>) -> CandleResult<Tensor> {
        let shape = a.shape();
        let data = if a.is_standard_layout() {
            a.as_slice().unwrap().to_vec()
        } else {
            a.as_standard_layout().as_slice().unwrap().to_vec()
        };
        Tensor::from_vec(data, shape, &self.device)
    }

    fn tensor_to_array3(t: &Tensor) -> InferenceResult<Array3<f32>> {
        let vals: Vec<f32> = t
            .to_vec3::<f32>()
            .map_err(|e| InferenceError::generation(format!("Tensor to vec3 failed: {}", e)))?
            .into_iter()
            .flatten()
            .flatten()
            .collect();
        let shape = t.dims().to_vec();
        Array3::from_shape_vec((shape[0], shape[1], shape[2]), vals)
            .map_err(|e| InferenceError::generation(format!("Shape conversion failed: {}", e)))
    }
}

impl GemmEngine for CandleMetalBackend {
    fn name(&self) -> &'static str {
        "candle_metal"
    }

    fn backend_type(&self) -> GemmBackendType {
        GemmBackendType::CandleMetal
    }

    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> InferenceResult<Array2<f32>> {
        let a_tensor = self
            .array2_to_tensor(a)
            .map_err(|e| InferenceError::generation(format!("Array to tensor failed: {}", e)))?;
        let b_tensor = self
            .array2_to_tensor(b)
            .map_err(|e| InferenceError::generation(format!("Array to tensor failed: {}", e)))?;
        let result = a_tensor
            .matmul(
                &b_tensor
                    .t()
                    .map_err(|e| InferenceError::generation(format!("Transpose failed: {}", e)))?,
            )
            .map_err(|e| InferenceError::generation(format!("MatMul failed: {}", e)))?;
        Self::tensor_to_array2(&result)
    }

    fn batched_matmul(&self, a: &Array3<f32>, b: &Array3<f32>) -> InferenceResult<Array3<f32>> {
        if a.shape()[0] != b.shape()[0] {
            return Err(InferenceError::config(format!(
                "Batch dimension mismatch: {} vs {}",
                a.shape()[0],
                b.shape()[0]
            )));
        }

        let a_tensor = self
            .array3_to_tensor(a)
            .map_err(|e| InferenceError::generation(format!("Array to tensor failed: {}", e)))?;
        let b_tensor = self
            .array3_to_tensor(b)
            .map_err(|e| InferenceError::generation(format!("Array to tensor failed: {}", e)))?;

        let b_transposed = b_tensor
            .transpose(0, 2)
            .map_err(|e| InferenceError::generation(format!("Transpose failed: {}", e)))?;

        let result = a_tensor
            .matmul(&b_transposed)
            .map_err(|e| InferenceError::generation(format!("Batched MatMul failed: {}", e)))?;

        Self::tensor_to_array3(&result)
    }

    fn fused_gemm_relu(
        &self,
        a: &Array2<f32>,
        w: &Array2<f32>,
        bias: Option<&Array1<f32>>,
    ) -> InferenceResult<Array2<f32>> {
        let a_tensor = self
            .array2_to_tensor(a)
            .map_err(|e| InferenceError::generation(format!("Array to tensor failed: {}", e)))?;
        let w_tensor = self
            .array2_to_tensor(w)
            .map_err(|e| InferenceError::generation(format!("Array to tensor failed: {}", e)))?;

        let mut result = a_tensor
            .matmul(
                &w_tensor
                    .t()
                    .map_err(|e| InferenceError::generation(format!("Transpose failed: {}", e)))?,
            )
            .map_err(|e| InferenceError::generation(format!("GEMM failed: {}", e)))?;

        if let Some(b) = bias {
            let bias_data = b.as_slice().unwrap().to_vec();
            let bias_shape = vec![b.len()];
            let bias_tensor = Tensor::from_vec(bias_data, bias_shape, &self.device)
                .map_err(|e| InferenceError::generation(format!("Bias tensor creation failed: {}", e)))?;
            result = (result + bias_tensor)
                .map_err(|e| InferenceError::generation(format!("Bias addition failed: {}", e)))?;
        }

        result = result
            .relu()
            .map_err(|e| InferenceError::generation(format!("ReLU failed: {}", e)))?;

        Self::tensor_to_array2(&result)
    }

    fn fused_gemm_silu(
        &self,
        x: &Array2<f32>,
        gate_w: &Array2<f32>,
        up_w: &Array2<f32>,
        bias: Option<&Array1<f32>>,
    ) -> InferenceResult<Array2<f32>> {
        let x_tensor = self
            .array2_to_tensor(x)
            .map_err(|e| InferenceError::generation(format!("Array to tensor failed: {}", e)))?;
        let gate_w_tensor = self
            .array2_to_tensor(gate_w)
            .map_err(|e| InferenceError::generation(format!("Array to tensor failed: {}", e)))?;
        let up_w_tensor = self
            .array2_to_tensor(up_w)
            .map_err(|e| InferenceError::generation(format!("Array to tensor failed: {}", e)))?;

        let gate = x_tensor
            .matmul(
                &gate_w_tensor
                    .t()
                    .map_err(|e| InferenceError::generation(format!("Transpose failed: {}", e)))?,
            )
            .map_err(|e| InferenceError::generation(format!("Gate GEMM failed: {}", e)))?;
        let up = x_tensor
            .matmul(
                &up_w_tensor
                    .t()
                    .map_err(|e| InferenceError::generation(format!("Transpose failed: {}", e)))?,
            )
            .map_err(|e| InferenceError::generation(format!("Up GEMM failed: {}", e)))?;

        let one_tensor = Tensor::ones(up.dims().to_vec(), DType::F32, &self.device)
            .map_err(|e| InferenceError::generation(format!("Create ones tensor failed: {}", e)))?;
        let neg_up = up
            .neg()
            .map_err(|e| InferenceError::generation(format!("Negation failed: {}", e)))?;
        let exp_neg_up = neg_up
            .exp()
            .map_err(|e| InferenceError::generation(format!("Exp failed: {}", e)))?;
        let exp_neg_up_plus_one = (exp_neg_up + one_tensor.clone())
            .map_err(|e| InferenceError::generation(format!("Addition failed: {}", e)))?;
        let sigmoid_up = one_tensor
            .div(&exp_neg_up_plus_one)
            .map_err(|e| InferenceError::generation(format!("Division failed: {}", e)))?;
        let silu_up = up
            .mul(&sigmoid_up)
            .map_err(|e| InferenceError::generation(format!("SiLU multiplication failed: {}", e)))?;

        let mut result = gate
            .mul(&silu_up)
            .map_err(|e| InferenceError::generation(format!("Final multiplication failed: {}", e)))?;

        if let Some(b) = bias {
            let bias_data = b.as_slice().unwrap().to_vec();
            let bias_shape = vec![b.len()];
            let bias_tensor = Tensor::from_vec(bias_data, bias_shape, &self.device)
                .map_err(|e| InferenceError::generation(format!("Bias tensor creation failed: {}", e)))?;
            result = (result + bias_tensor)
                .map_err(|e| InferenceError::generation(format!("Bias addition failed: {}", e)))?;
        }

        Self::tensor_to_array2(&result)
    }

    fn is_available(&self) -> bool {
        true
    }
}
