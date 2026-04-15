use ndarray::{Array1, Array2, Array3};
use std::sync::OnceLock;

#[cfg(feature = "candle")]
use candle_core::{DType, Device, Result as CandleResult, Tensor};

#[cfg(feature = "cuda")]
mod cuda_backend;
#[cfg(feature = "metal")]
mod metal_backend;

use crate::model::inference::error::{InferenceError, InferenceResult};

#[derive(Debug, Clone, Copy)]
pub enum GemmBackendType {
    Ndarray,
    CandleCpuBlas,
    CandleCuda,
    CandleMetal,
}

impl std::fmt::Display for GemmBackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ndarray => write!(f, "ndarray"),
            Self::CandleCpuBlas => write!(f, "candle-cpu-blas"),
            Self::CandleCuda => write!(f, "candle-cuda"),
            Self::CandleMetal => write!(f, "candle-metal"),
        }
    }
}

pub trait GemmEngine: Send + Sync {
    fn name(&self) -> &'static str;
    fn backend_type(&self) -> GemmBackendType;
    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> InferenceResult<Array2<f32>>;
    fn batched_matmul(&self, a: &Array3<f32>, b: &Array3<f32>) -> InferenceResult<Array3<f32>>;
    fn fused_gemm_relu(
        &self,
        a: &Array2<f32>,
        w: &Array2<f32>,
        bias: Option<&Array1<f32>>,
    ) -> InferenceResult<Array2<f32>>;
    fn fused_gemm_silu(
        &self,
        x: &Array2<f32>,
        gate_w: &Array2<f32>,
        up_w: &Array2<f32>,
        bias: Option<&Array1<f32>>,
    ) -> InferenceResult<Array2<f32>>;
    fn is_available(&self) -> bool;
}

pub struct NdarrayFallbackBackend;

impl GemmEngine for NdarrayFallbackBackend {
    fn name(&self) -> &'static str {
        "ndarray_fallback"
    }

    fn backend_type(&self) -> GemmBackendType {
        GemmBackendType::Ndarray
    }

    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> InferenceResult<Array2<f32>> {
        Ok(a.dot(b))
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
        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let a_slice = a.slice(ndarray::s![i, .., ..]);
            let b_slice = b.slice(ndarray::s![i, .., ..]);
            results.push(a_slice.to_owned().dot(&b_slice.to_owned()));
        }

        Array3::from_shape_vec(
            (batch_size, results[0].shape()[0], results[0].shape()[1]),
            results.into_iter().flatten().collect(),
        )
        .map_err(|e| InferenceError::generation(e.to_string()))
    }

    fn fused_gemm_relu(
        &self,
        a: &Array2<f32>,
        w: &Array2<f32>,
        bias: Option<&Array1<f32>>,
    ) -> InferenceResult<Array2<f32>> {
        let mut result = a.dot(w);
        if let Some(b) = bias {
            result += b;
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
        let gate = x.dot(gate_w);
        let up = x.dot(up_w);
        let mut result = gate
            * up.mapv(|x| {
                let sig = 1.0 / (1.0 + (-x).exp());
                x * sig
            });
        if let Some(b) = bias {
            result += b;
        }
        Ok(result)
    }

    fn is_available(&self) -> bool {
        true
    }
}

#[cfg(feature = "candle")]
pub struct CandleCpuBlasBackend {
    device: Device,
}

#[cfg(feature = "candle")]
impl Default for CandleCpuBlasBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "candle")]
impl CandleCpuBlasBackend {
    pub fn new() -> Self {
        Self {
            device: Device::Cpu,
        }
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

#[cfg(feature = "candle")]
impl GemmEngine for CandleCpuBlasBackend {
    fn name(&self) -> &'static str {
        "candle_cpu_blas"
    }

    fn backend_type(&self) -> GemmBackendType {
        GemmBackendType::CandleCpuBlas
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
            let bias_tensor =
                Tensor::from_vec(bias_data, bias_shape, &self.device).map_err(|e| {
                    InferenceError::generation(format!("Bias tensor creation failed: {}", e))
                })?;
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
        let silu_up = up.mul(&sigmoid_up).map_err(|e| {
            InferenceError::generation(format!("SiLU multiplication failed: {}", e))
        })?;

        let mut result = gate.mul(&silu_up).map_err(|e| {
            InferenceError::generation(format!("Final multiplication failed: {}", e))
        })?;

        if let Some(b) = bias {
            let bias_data = b.as_slice().unwrap().to_vec();
            let bias_shape = vec![b.len()];
            let bias_tensor =
                Tensor::from_vec(bias_data, bias_shape, &self.device).map_err(|e| {
                    InferenceError::generation(format!("Bias tensor creation failed: {}", e))
                })?;
            result = (result + bias_tensor)
                .map_err(|e| InferenceError::generation(format!("Bias addition failed: {}", e)))?;
        }

        Self::tensor_to_array2(&result)
    }

    fn is_available(&self) -> bool {
        true
    }
}

struct GemmEngineManagerInner {
    engine: Box<dyn GemmEngine>,
}

impl GemmEngineManagerInner {
    pub fn new() -> Self {
        let engine: Box<dyn GemmEngine> = Self::try_create_cuda_backend()
            .or_else(Self::try_create_metal_backend)
            .or_else(|| Self::try_create_blas_backend())
            .unwrap_or_else(|| Box::new(NdarrayFallbackBackend));

        let backend_name = engine.name();
        tracing::info!("GEMM engine initialized: {}", backend_name);

        Self { engine }
    }

    #[cfg(feature = "cuda")]
    fn try_create_cuda_backend() -> Option<Box<dyn GemmEngine>> {
        use crate::model::inference::gemm_engine::cuda_backend::CandleCudaBackend;
        match CandleCudaBackend::new() {
            Ok(b) if b.is_available() => Some(Box::new(b)),
            _ => None,
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn try_create_cuda_backend() -> Option<Box<dyn GemmEngine>> {
        None
    }

    #[cfg(feature = "metal")]
    fn try_create_metal_backend() -> Option<Box<dyn GemmEngine>> {
        use crate::model::inference::gemm_engine::metal_backend::CandleMetalBackend;
        match CandleMetalBackend::new() {
            Ok(b) if b.is_available() => Some(Box::new(b)),
            _ => None,
        }
    }

    #[cfg(not(feature = "metal"))]
    fn try_create_metal_backend() -> Option<Box<dyn GemmEngine>> {
        None
    }

    #[cfg(feature = "candle")]
    fn try_create_blas_backend() -> Option<Box<dyn GemmEngine>> {
        match std::panic::catch_unwind(|| {
            let backend = CandleCpuBlasBackend::new();
            backend.is_available()
        }) {
            Ok(true) => Some(Box::new(CandleCpuBlasBackend::new())),
            _ => None,
        }
    }

    #[cfg(not(feature = "candle"))]
    fn try_create_blas_backend() -> Option<Box<dyn GemmEngine>> {
        None
    }
}

pub struct GemmEngineManager {
    inner: GemmEngineManagerInner,
}

impl Default for GemmEngineManager {
    fn default() -> Self {
        Self::new()
    }
}

impl GemmEngineManager {
    pub fn new() -> Self {
        Self {
            inner: GemmEngineManagerInner::new(),
        }
    }

    pub fn engine(&self) -> &dyn GemmEngine {
        self.inner.engine.as_ref()
    }

    pub fn current_backend(&self) -> GemmBackendType {
        self.inner.engine.backend_type()
    }
}

static GEMM_ENGINE_MANAGER: OnceLock<GemmEngineManager> = OnceLock::new();

pub fn get_gemm_engine_manager() -> &'static GemmEngineManager {
    GEMM_ENGINE_MANAGER.get_or_init(GemmEngineManager::new)
}
