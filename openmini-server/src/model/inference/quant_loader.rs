//! 量化权重加载模块
//!
//! 提供从 GGUF 文件加载量化权重并转换为模型权重结构的功能
//!
//! # 功能
//!
//! - 从 GGUF 张量加载量化权重
//! - 自动反量化为 f32 格式
//! - 权重结构映射到模型层

#![allow(dead_code)]

use std::path::Path;

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2, ShapeBuilder};

use super::gguf::{GgufFile, GgufTensor, GgufTensorType};
use super::model::ModelConfig;
use super::quant_simd::get_optimal_threads;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct QuantizedWeights {
    pub data: Array2<f32>,
    pub tensor_type: GgufTensorType,
    pub original_shape: Vec<usize>,
}

impl QuantizedWeights {
    pub fn new(data: Array2<f32>, tensor_type: GgufTensorType) -> Self {
        let shape = data.shape().to_vec();
        Self {
            data,
            tensor_type,
            original_shape: shape,
        }
    }

    #[allow(dead_code)]
    pub fn num_elements(&self) -> usize {
        self.data.len()
    }
}

pub struct QuantizedWeightLoader {
    gguf: GgufFile,
    #[allow(dead_code)]
    num_threads: usize,
}

impl QuantizedWeightLoader {
    #[allow(dead_code)]
    pub fn open<P: AsRef<Path>>(path: P, use_simd: bool) -> Result<Self> {
        let gguf = GgufFile::open(path)?;

        let num_threads = if use_simd {
            get_optimal_threads(1024)
        } else {
            1
        };

        Ok(Self { gguf, num_threads })
    }

    pub fn load_tensor(&self, name: &str) -> Result<QuantizedWeights> {
        let tensor = self
            .gguf
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("Tensor not found: {}", name))?;

        self.load_tensor_data(tensor)
    }

    pub fn load_tensor_data(&self, tensor: &GgufTensor) -> Result<QuantizedWeights> {
        let data = self.gguf.get_tensor_data_by_ref(tensor)?;

        let dims = &tensor.dims;
        let shape = if dims.len() == 1 {
            [dims[0], 1]
        } else if dims.is_empty() {
            [1, 1]
        } else {
            [dims[0], dims[1]]
        };

        let arr = Array2::from_shape_vec((shape[0], shape[1]).f(), data)
            .map_err(|e| anyhow!("Failed to create array: {}", e))?;

        Ok(QuantizedWeights::new(arr, tensor.tensor_type))
    }

    #[allow(dead_code)]
    pub fn load_attention_weights(&self, layer_idx: usize) -> Result<AttentionWeightArrays> {
        let prefix = format!("model.layers.{}.attention", layer_idx);

        let q_proj = self
            .load_tensor(&format!("{}.q_proj", prefix))
            .map(|w| w.data)?;
        let k_proj = self
            .load_tensor(&format!("{}.k_proj", prefix))
            .map(|w| w.data)?;
        let v_proj = self
            .load_tensor(&format!("{}.v_proj", prefix))
            .map(|w| w.data)?;
        let o_proj = self
            .load_tensor(&format!("{}.o_proj", prefix))
            .map(|w| w.data)?;

        Ok(AttentionWeightArrays {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        })
    }

    #[allow(dead_code)]
    pub fn load_ffn_weights(&self, layer_idx: usize) -> Result<FfnWeightArrays> {
        let prefix = format!("model.layers.{}.ffn", layer_idx);

        let gate_proj = self
            .load_tensor(&format!("{}.gate_proj", prefix))
            .map(|w| w.data)?;
        let up_proj = self
            .load_tensor(&format!("{}.up_proj", prefix))
            .map(|w| w.data)?;
        let down_proj = self
            .load_tensor(&format!("{}.down_proj", prefix))
            .map(|w| w.data)?;

        Ok(FfnWeightArrays {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    #[allow(dead_code)]
    pub fn load_mla_weights(&self, layer_idx: usize) -> Result<MlaWeightArrays> {
        let prefix = format!("model.layers.{}.attention.mla", layer_idx);

        let q_proj = self
            .load_tensor(&format!("{}.q_proj", prefix))
            .map(|w| w.data)?;
        let o_proj = self
            .load_tensor(&format!("{}.o_proj", prefix))
            .map(|w| w.data)?;
        let dkv_proj = self
            .load_tensor(&format!("{}.dkv_proj", prefix))
            .map(|w| w.data)?;
        let uk_proj = self
            .load_tensor(&format!("{}.uk_proj", prefix))
            .map(|w| w.data)?;
        let uv_proj = self
            .load_tensor(&format!("{}.uv_proj", prefix))
            .map(|w| w.data)?;

        let qr_proj = self
            .try_load_tensor(&format!("{}.qr_proj", prefix))
            .map(|w| w.data);
        let kr_proj = self
            .try_load_tensor(&format!("{}.kr_proj", prefix))
            .map(|w| w.data);

        Ok(MlaWeightArrays {
            q_proj,
            o_proj,
            dkv_proj,
            uk_proj,
            uv_proj,
            qr_proj,
            kr_proj,
        })
    }

    pub fn try_load_tensor(&self, name: &str) -> Option<QuantizedWeights> {
        self.load_tensor(name).ok()
    }

    #[allow(dead_code)]
    pub fn load_moe_weights(&self, layer_idx: usize) -> Result<MoEWeightArrays> {
        let prefix = format!("model.layers.{}.moe", layer_idx);

        let num_experts = self.get_meta::<u32>("moe.num_experts").unwrap_or(8) as usize;
        let top_k = self.get_meta::<u32>("moe.top_k").unwrap_or(2) as usize;

        let mut experts = Vec::with_capacity(num_experts);
        for expert_idx in 0..num_experts {
            let expert_prefix = format!("{}.experts.{}", prefix, expert_idx);
            let gate_proj = self
                .load_tensor_with_fallback(
                    &format!("{}.gate_proj", expert_prefix),
                    layer_idx,
                    expert_idx,
                )
                .map(|w| w.data)?;
            let up_proj = self
                .load_tensor_with_fallback(
                    &format!("{}.up_proj", expert_prefix),
                    layer_idx,
                    expert_idx,
                )
                .map(|w| w.data)?;
            let down_proj = self
                .load_tensor_with_fallback(
                    &format!("{}.down_proj", expert_prefix),
                    layer_idx,
                    expert_idx,
                )
                .map(|w| w.data)?;
            experts.push(FfnWeightArrays {
                gate_proj,
                up_proj,
                down_proj,
            });
        }

        let router = self
            .load_tensor(&format!("{}.router", prefix))
            .map(|w| w.data)?;

        Ok(MoEWeightArrays {
            experts,
            router,
            top_k,
        })
    }

    fn load_tensor_with_fallback(
        &self,
        name: &str,
        layer_idx: usize,
        expert_idx: usize,
    ) -> Result<QuantizedWeights> {
        if let Ok(tensor) = self.load_tensor(name) {
            return Ok(tensor);
        }

        let fallback_names = vec![
            format!(
                "deepseek_v3.model.layers.{}.ffn.experts.{}.gate_proj",
                layer_idx, expert_idx
            ),
            format!(
                "deepseek_v3.model.layers.{}.ffn.experts.{}.up_proj",
                layer_idx, expert_idx
            ),
            format!(
                "deepseek_v3.model.layers.{}.ffn.experts.{}.down_proj",
                layer_idx, expert_idx
            ),
            format!(
                "deepseek_v3.model.layers.{}.ffn.experts.{}",
                layer_idx, expert_idx
            ),
        ];

        for fallback_name in &fallback_names {
            if name.contains("gate_proj") && fallback_name.contains("gate_proj")
                || name.contains("up_proj") && fallback_name.contains("up_proj")
                || name.contains("down_proj") && fallback_name.contains("down_proj")
            {
                if let Ok(tensor) = self.load_tensor(fallback_name) {
                    return Ok(tensor);
                }
            }
        }

        self.load_tensor(name)
    }

    pub fn load_embedding_weights(&self) -> Result<Array2<f32>> {
        self.load_tensor("model.embed_tokens").map(|w| w.data)
    }

    pub fn load_norm_weights(&self, name: &str) -> Result<Array1<f32>> {
        let weights = self.load_tensor(name)?;
        let len = weights.data.len();
        let data = weights
            .data
            .into_shape_with_order((len,))
            .map_err(|e| anyhow!("{}", e))?;
        Ok(data)
    }

    pub(crate) fn get_meta<T: FromGgufMeta>(&self, key: &str) -> Option<T> {
        T::from_gguf_meta(&self.gguf.metadata, key)
    }

    #[allow(dead_code)]
    pub fn list_tensors(&self) -> Vec<String> {
        self.gguf.tensors.keys().cloned().collect()
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.gguf.tensors.contains_key(name)
    }

    #[allow(dead_code)]
    pub fn tensor_type(&self, name: &str) -> Option<GgufTensorType> {
        self.gguf.tensors.get(name).map(|t| t.tensor_type)
    }

    /// 从 GGUF 元数据获取模型配置
    pub fn get_model_config(&self) -> Result<ModelConfig> {
        let gguf_config = self.gguf.get_model_config()?;
        Ok(gguf_config.into())
    }
}

#[allow(dead_code)]
pub struct AttentionWeightArrays {
    pub q_proj: Array2<f32>,
    pub k_proj: Array2<f32>,
    pub v_proj: Array2<f32>,
    pub o_proj: Array2<f32>,
}

#[allow(dead_code)]
#[derive(Clone)]
pub struct FfnWeightArrays {
    pub gate_proj: Array2<f32>,
    pub up_proj: Array2<f32>,
    pub down_proj: Array2<f32>,
}

#[allow(dead_code)]
pub struct MlaWeightArrays {
    pub q_proj: Array2<f32>,
    pub o_proj: Array2<f32>,
    pub dkv_proj: Array2<f32>,
    pub uk_proj: Array2<f32>,
    pub uv_proj: Array2<f32>,
    pub qr_proj: Option<Array2<f32>>,
    pub kr_proj: Option<Array2<f32>>,
}

#[allow(dead_code)]
pub struct MoEWeightArrays {
    pub experts: Vec<FfnWeightArrays>,
    pub router: Array2<f32>,
    pub top_k: usize,
}

pub(crate) trait FromGgufMeta: Sized {
    fn from_gguf_meta(
        metadata: &crate::model::inference::gguf::GgufMetadata,
        key: &str,
    ) -> Option<Self>;
}

impl FromGgufMeta for u32 {
    fn from_gguf_meta(
        metadata: &crate::model::inference::gguf::GgufMetadata,
        key: &str,
    ) -> Option<Self> {
        metadata.get_u32(key)
    }
}

impl FromGgufMeta for usize {
    fn from_gguf_meta(
        metadata: &crate::model::inference::gguf::GgufMetadata,
        key: &str,
    ) -> Option<Self> {
        metadata.get_u64(key).map(|v| v as usize)
    }
}

impl FromGgufMeta for f32 {
    fn from_gguf_meta(
        metadata: &crate::model::inference::gguf::GgufMetadata,
        key: &str,
    ) -> Option<Self> {
        metadata.get_f32(key)
    }
}

#[allow(dead_code)]
pub struct WeightBiasArrays {
    pub q_bias: Option<Array1<f32>>,
    pub k_bias: Option<Array1<f32>>,
    pub v_bias: Option<Array1<f32>>,
    pub o_bias: Option<Array1<f32>>,
}

impl WeightBiasArrays {
    #[allow(dead_code)]
    pub fn load_from_loader(loader: &QuantizedWeightLoader, layer_idx: usize) -> Result<Self> {
        let prefix = format!("model.layers.{}.attention", layer_idx);

        let q_bias = loader
            .try_load_tensor(&format!("{}.q_bias", prefix))
            .and_then(|w| {
                let len = w.data.len();
                w.data.into_shape_with_order((len,)).ok()
            });
        let k_bias = loader
            .try_load_tensor(&format!("{}.k_bias", prefix))
            .and_then(|w| {
                let len = w.data.len();
                w.data.into_shape_with_order((len,)).ok()
            });
        let v_bias = loader
            .try_load_tensor(&format!("{}.v_bias", prefix))
            .and_then(|w| {
                let len = w.data.len();
                w.data.into_shape_with_order((len,)).ok()
            });
        let o_bias = loader
            .try_load_tensor(&format!("{}.o_bias", prefix))
            .and_then(|w| {
                let len = w.data.len();
                w.data.into_shape_with_order((len,)).ok()
            });

        Ok(Self {
            q_bias,
            k_bias,
            v_bias,
            o_bias,
        })
    }
}

pub struct QuantizedModelLoader {
    loader: QuantizedWeightLoader,
}

impl QuantizedModelLoader {
    #[allow(dead_code)]
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let loader = QuantizedWeightLoader::open(path, true)?;
        Ok(Self { loader })
    }

    pub fn load_layer(&self, layer_idx: usize) -> Result<QuantizedLayerWeights> {
        let attention = QuantizedAttentionWeights::load_from_loader(&self.loader, layer_idx)?;
        let ffn = QuantizedFfnWeights::load_from_loader(&self.loader, layer_idx)?;

        let input_layernorm = self
            .loader
            .load_norm_weights(&format!("model.layers.{}.input_layernorm", layer_idx))?;
        let post_attention_layernorm = self.loader.load_norm_weights(&format!(
            "model.layers.{}.post_attention_layernorm",
            layer_idx
        ))?;

        let has_mla = self
            .loader
            .has_tensor(&format!("model.layers.{}.attention.mla.q_proj", layer_idx));
        let has_moe = self
            .loader
            .has_tensor(&format!("model.layers.{}.moe.router", layer_idx));

        let mla = if has_mla {
            Some(QuantizedMlaWeights::load_from_loader(
                &self.loader,
                layer_idx,
            )?)
        } else {
            None
        };

        let moe = if has_moe {
            Some(QuantizedMoEWeights::load_from_loader(
                &self.loader,
                layer_idx,
            )?)
        } else {
            None
        };

        Ok(QuantizedLayerWeights {
            attention,
            mla,
            ffn,
            moe,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn load_embedding(&self) -> Result<Array2<f32>> {
        self.loader.load_embedding_weights()
    }

    pub fn load_final_norm(&self) -> Result<Array1<f32>> {
        self.loader.load_norm_weights("model.final_layernorm")
    }

    pub fn num_layers(&self) -> usize {
        self.loader
            .get_meta::<u32>("model.num_hidden_layers")
            .unwrap_or(28) as usize
    }

    pub fn load_lm_head(&self) -> Option<Array2<f32>> {
        self.loader.try_load_tensor("lm_head").map(|w| w.data)
    }

    pub fn has_lm_head(&self) -> bool {
        self.loader.has_tensor("lm_head")
    }

    pub fn load_modality_embeds(&self) -> std::collections::HashMap<usize, Array1<f32>> {
        let mut embeds = std::collections::HashMap::new();

        for mod_id in 0..3 {
            let tensor_name = format!("model.modality_embed.{}", mod_id);
            if let Some(tensor) = self.loader.try_load_tensor(&tensor_name) {
                let hidden_size = tensor.data.nrows();
                if tensor.data.ncols() == 1 {
                    embeds.insert(mod_id, tensor.data.column(0).to_owned());
                } else if tensor.data.nrows() == 1 {
                    embeds.insert(mod_id, tensor.data.row(0).to_owned());
                } else {
                    let flat: Vec<f32> = tensor.data.iter().cloned().collect();
                    if flat.len() == hidden_size {
                        embeds.insert(mod_id, Array1::from_vec(flat));
                    }
                }
            }
        }

        embeds
    }

    pub fn load_layer_modality_embeds(
        &self,
        layer_idx: usize,
    ) -> std::collections::HashMap<usize, Array1<f32>> {
        let mut embeds = std::collections::HashMap::new();

        for mod_id in 0..3 {
            let tensor_name = format!("model.layers.{}.moe.modality_embed.{}", layer_idx, mod_id);
            if let Some(tensor) = self.loader.try_load_tensor(&tensor_name) {
                if tensor.data.ncols() == 1 {
                    embeds.insert(mod_id, tensor.data.column(0).to_owned());
                } else if tensor.data.nrows() == 1 {
                    embeds.insert(mod_id, tensor.data.row(0).to_owned());
                }
            }
        }

        embeds
    }

    /// 从 GGUF 元数据获取完整模型配置
    pub fn config(&self) -> Result<ModelConfig> {
        self.loader.get_model_config()
    }

    /// 从加载的权重推断模型配置（当元数据不完整时使用）
    #[allow(clippy::field_reassign_with_default)]
    pub fn infer_config(&self) -> ModelConfig {
        let mut config = ModelConfig::default();

        config.num_hidden_layers = self.num_layers();

        if let Ok(embedding) = self.load_embedding() {
            config.vocab_size = embedding.nrows();
            config.hidden_size = embedding.ncols();
        }

        if let Ok(layer) = self.load_layer(0) {
            let q_proj_rows = layer.attention.q_proj.nrows();
            let q_proj_cols = layer.attention.q_proj.ncols();

            if q_proj_cols == config.hidden_size {
                let num_heads = q_proj_rows / config.head_dim;
                config.num_attention_heads = num_heads;
            }

            let k_proj_rows = layer.attention.k_proj.nrows();
            config.num_key_value_heads = k_proj_rows / config.head_dim;

            if let Some(ref mla) = layer.mla {
                config.mla_latent_dim = mla.dkv_proj.nrows();
                config.use_mla = true;
            }

            if layer.moe.is_some() {
                config.moe_num_experts = 8;
                config.moe_top_k = 2;
            }
        }

        config
    }
}

pub struct QuantizedLayerWeights {
    pub attention: QuantizedAttentionWeights,
    pub mla: Option<QuantizedMlaWeights>,
    pub ffn: QuantizedFfnWeights,
    pub moe: Option<QuantizedMoEWeights>,
    pub input_layernorm: Array1<f32>,
    pub post_attention_layernorm: Array1<f32>,
}

pub struct QuantizedAttentionWeights {
    pub q_proj: Array2<f32>,
    pub k_proj: Array2<f32>,
    pub v_proj: Array2<f32>,
    pub o_proj: Array2<f32>,
}

impl QuantizedAttentionWeights {
    pub fn load_from_loader(loader: &QuantizedWeightLoader, layer_idx: usize) -> Result<Self> {
        let prefix = format!("model.layers.{}.attention", layer_idx);

        let q_proj = loader.load_tensor(&format!("{}.q_proj", prefix))?.data;
        let k_proj = loader.load_tensor(&format!("{}.k_proj", prefix))?.data;
        let v_proj = loader.load_tensor(&format!("{}.v_proj", prefix))?.data;
        let o_proj = loader.load_tensor(&format!("{}.o_proj", prefix))?.data;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        })
    }
}

pub struct QuantizedFfnWeights {
    pub gate_proj: Array2<f32>,
    pub up_proj: Array2<f32>,
    pub down_proj: Array2<f32>,
}

impl QuantizedFfnWeights {
    pub fn load_from_loader(loader: &QuantizedWeightLoader, layer_idx: usize) -> Result<Self> {
        let prefix = format!("model.layers.{}.ffn", layer_idx);

        let gate_proj = loader.load_tensor(&format!("{}.gate_proj", prefix))?.data;
        let up_proj = loader.load_tensor(&format!("{}.up_proj", prefix))?.data;
        let down_proj = loader.load_tensor(&format!("{}.down_proj", prefix))?.data;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

pub struct QuantizedMlaWeights {
    pub q_proj: Array2<f32>,
    pub o_proj: Array2<f32>,
    pub dkv_proj: Array2<f32>,
    pub uk_proj: Array2<f32>,
    pub uv_proj: Array2<f32>,
    pub qr_proj: Option<Array2<f32>>,
    pub kr_proj: Option<Array2<f32>>,
}

impl QuantizedMlaWeights {
    pub fn load_from_loader(loader: &QuantizedWeightLoader, layer_idx: usize) -> Result<Self> {
        let prefix = format!("model.layers.{}.attention.mla", layer_idx);

        let q_proj = loader.load_tensor(&format!("{}.q_proj", prefix))?.data;
        let o_proj = loader.load_tensor(&format!("{}.o_proj", prefix))?.data;
        let dkv_proj = loader.load_tensor(&format!("{}.dkv_proj", prefix))?.data;
        let uk_proj = loader.load_tensor(&format!("{}.uk_proj", prefix))?.data;
        let uv_proj = loader.load_tensor(&format!("{}.uv_proj", prefix))?.data;

        let qr_proj = loader
            .try_load_tensor(&format!("{}.qr_proj", prefix))
            .map(|w| w.data);
        let kr_proj = loader
            .try_load_tensor(&format!("{}.kr_proj", prefix))
            .map(|w| w.data);

        Ok(Self {
            q_proj,
            o_proj,
            dkv_proj,
            uk_proj,
            uv_proj,
            qr_proj,
            kr_proj,
        })
    }
}

pub struct QuantizedMoEWeights {
    pub experts: Vec<QuantizedFfnWeights>,
    pub router: Array2<f32>,
    pub top_k: usize,
}

impl QuantizedMoEWeights {
    pub fn load_from_loader(loader: &QuantizedWeightLoader, layer_idx: usize) -> Result<Self> {
        let prefix = format!("model.layers.{}.moe", layer_idx);

        let num_experts = loader.get_meta::<u32>("moe.num_experts").unwrap_or(8) as usize;
        let top_k = loader.get_meta::<u32>("moe.top_k").unwrap_or(2) as usize;

        let mut experts = Vec::with_capacity(num_experts);
        for expert_idx in 0..num_experts {
            let expert_prefix = format!("{}.experts.{}", prefix, expert_idx);
            let gate_proj = loader
                .load_tensor(&format!("{}.gate_proj", expert_prefix))?
                .data;
            let up_proj = loader
                .load_tensor(&format!("{}.up_proj", expert_prefix))?
                .data;
            let down_proj = loader
                .load_tensor(&format!("{}.down_proj", expert_prefix))?
                .data;
            experts.push(QuantizedFfnWeights {
                gate_proj,
                up_proj,
                down_proj,
            });
        }

        let router = loader.load_tensor(&format!("{}.router", prefix))?.data;

        Ok(Self {
            experts,
            router,
            top_k,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_weights_creation() {
        let data = Array2::zeros((10, 20));
        let weights = QuantizedWeights::new(data, GgufTensorType::F32);
        assert_eq!(weights.num_elements(), 200);
        assert_eq!(weights.original_shape, vec![10, 20]);
    }

    #[test]
    fn test_attention_weight_arrays_creation() {
        let arr = Array2::zeros((10, 10));
        let weights = AttentionWeightArrays {
            q_proj: arr.clone(),
            k_proj: arr.clone(),
            v_proj: arr.clone(),
            o_proj: arr,
        };
        assert_eq!(weights.q_proj.len(), 100);
    }

    #[test]
    fn test_ffn_weight_arrays_creation() {
        let arr = Array2::zeros((10, 10));
        let weights = FfnWeightArrays {
            gate_proj: arr.clone(),
            up_proj: arr.clone(),
            down_proj: arr,
        };
        assert_eq!(weights.gate_proj.len(), 100);
    }

    /// 测试：QuantizedWeights的Clone特性
    #[test]
    fn test_quantized_weights_clone() {
        let data = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let weights = QuantizedWeights::new(data.clone(), GgufTensorType::F32);

        let cloned = weights.clone();
        assert_eq!(cloned.num_elements(), weights.num_elements());
        assert_eq!(cloned.original_shape, weights.original_shape);
        assert_eq!(cloned.tensor_type, weights.tensor_type);
    }

    /// 测试：QuantizedWeights创建时记录正确的原始形状
    #[test]
    fn test_quantized_weights_shape_tracking() {
        // 测试1维数组
        let data_1d = Array2::from_shape_vec((10, 1), vec![0.0; 10]).unwrap();
        let weights_1d = QuantizedWeights::new(data_1d, GgufTensorType::F32);
        assert_eq!(weights_1d.original_shape, vec![10, 1]);

        // 测试2维数组
        let data_2d = Array2::from_shape_vec((5, 8), vec![0.0; 40]).unwrap();
        let weights_2d = QuantizedWeights::new(data_2d, GgufTensorType::F32);
        assert_eq!(weights_2d.original_shape, vec![5, 8]);
    }

    /// 测试：MlaWeightArrays创建（包含可选字段）
    #[test]
    fn test_mla_weight_arrays_creation() {
        let arr = Array2::zeros((10, 10));
        let mla_weights = MlaWeightArrays {
            q_proj: arr.clone(),
            o_proj: arr.clone(),
            dkv_proj: arr.clone(),
            uk_proj: arr.clone(),
            uv_proj: arr.clone(),
            qr_proj: None,
            kr_proj: None,
        };

        assert_eq!(mla_weights.q_proj.len(), 100);
        assert!(mla_weights.qr_proj.is_none());
        assert!(mla_weights.kr_proj.is_none());
    }

    /// 测试：MlaWeightArrays包含可选字段的情况
    #[test]
    fn test_mla_weight_arrays_with_optional_fields() {
        let arr = Array2::zeros((10, 10));
        let mla_weights = MlaWeightArrays {
            q_proj: arr.clone(),
            o_proj: arr.clone(),
            dkv_proj: arr.clone(),
            uk_proj: arr.clone(),
            uv_proj: arr.clone(),
            qr_proj: Some(arr.clone()),
            kr_proj: Some(arr),
        };

        assert!(mla_weights.qr_proj.is_some());
        assert!(mla_weights.kr_proj.is_some());
        assert_eq!(mla_weights.qr_proj.unwrap().len(), 100);
    }

    /// 测试：MoEWeightArrays创建（包含专家列表）
    #[test]
    fn test_moe_weight_arrays_creation() {
        let expert1 = FfnWeightArrays {
            gate_proj: Array2::zeros((10, 10)),
            up_proj: Array2::zeros((10, 10)),
            down_proj: Array2::zeros((10, 10)),
        };
        let expert2 = FfnWeightArrays {
            gate_proj: Array2::zeros((10, 10)),
            up_proj: Array2::zeros((10, 10)),
            down_proj: Array2::zeros((10, 10)),
        };
        let router = Array2::zeros((4, 8));

        let moe_weights = MoEWeightArrays {
            experts: vec![expert1, expert2],
            router,
            top_k: 2,
        };

        assert_eq!(moe_weights.experts.len(), 2);
        assert_eq!(moe_weights.top_k, 2);
        assert_eq!(moe_weights.router.shape(), [4, 8]);
    }

    /// 测试：WeightBiasArrays创建（所有bias为None）
    #[test]
    fn test_weight_bias_arrays_all_none() {
        let bias_arrays = WeightBiasArrays {
            q_bias: None,
            k_bias: None,
            v_bias: None,
            o_bias: None,
        };

        assert!(bias_arrays.q_bias.is_none());
        assert!(bias_arrays.k_bias.is_none());
        assert!(bias_arrays.v_bias.is_none());
        assert!(bias_arrays.o_bias.is_none());
    }

    /// 测试：WeightBiasArrays创建（部分bias有值）
    #[test]
    fn test_weight_bias_arrays_partial_bias() {
        let q_bias = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let o_bias = Array1::from_vec(vec![0.4, 0.5, 0.6]);

        let bias_arrays = WeightBiasArrays {
            q_bias: Some(q_bias),
            k_bias: None,
            v_bias: None,
            o_bias: Some(o_bias),
        };

        assert!(bias_arrays.q_bias.is_some());
        assert!(bias_arrays.k_bias.is_none());
        assert!(bias_arrays.v_bias.is_none());
        assert!(bias_arrays.o_bias.is_some());
        assert_eq!(bias_arrays.q_bias.unwrap().len(), 3);
    }

    /// 测试：FromGgufMeta trait的u32实现（模拟元数据查找）
    #[test]
    fn test_from_gguf_meta_u32() {
        // 注意：这个测试需要实际的GgufMetadata实例
        // 这里我们只验证trait签名和基本行为
        // 实际测试需要mock或真实的GGUF文件
        let _key = "test_key";
        // 由于GgufMetadata不是公开的，我们只能验证编译通过
        // 实际测试需要集成测试
    }

    /// 测试：FromGgufMeta trait的usize实现
    #[test]
    fn test_from_gguf_meta_usize() {
        // 验证usize从u64转换的实现
        // 实际测试需要真实的GgufMetadata实例
        let _key = "model.num_hidden_layers";
        // 集成测试中验证
    }

    /// 测试：FromGgufMeta trait的f32实现
    #[test]
    fn test_from_gguf_meta_f32() {
        // 验证f32元数据提取
        // 实际测试需要真实的GgufMetadata实例
        let _key = "model.temperature";
        // 集成测试中验证
    }

    /// 测试：QuantizedLayerWeights创建（包含所有可选字段）
    #[test]
    fn test_quantized_layer_weights_creation() {
        // 创建注意力权重
        let _attention = QuantizedAttentionWeights {
            q_proj: Array2::zeros((64, 128)),
            k_proj: Array2::zeros((32, 128)),
            v_proj: Array2::zeros((32, 128)),
            o_proj: Array2::zeros((64, 128)),
        };

        // 创建FFN权重
        let _ffn = QuantizedFfnWeights {
            gate_proj: Array2::zeros((256, 128)),
            up_proj: Array2::zeros((256, 128)),
            down_proj: Array2::zeros((128, 256)),
        };

        let input_layernorm = Array1::from_vec(vec![0.1; 128]);
        let post_attention_layernorm = Array1::from_vec(vec![0.2; 128]);

        // 测试无MLA和MoE的情况
        let _layer_no_optional = QuantizedLayerWeights {
            attention: QuantizedAttentionWeights {
                q_proj: Array2::zeros((64, 128)),
                k_proj: Array2::zeros((32, 128)),
                v_proj: Array2::zeros((32, 128)),
                o_proj: Array2::zeros((64, 128)),
            },
            mla: None,
            ffn: QuantizedFfnWeights {
                gate_proj: Array2::zeros((256, 128)),
                up_proj: Array2::zeros((256, 128)),
                down_proj: Array2::zeros((128, 256)),
            },
            moe: None,
            input_layernorm,
            post_attention_layernorm,
        };

        // 测试有MLA和MoE的情况
        let _mla = Some(QuantizedMlaWeights {
            q_proj: Array2::zeros((64, 64)),
            o_proj: Array2::zeros((64, 64)),
            dkv_proj: Array2::zeros((32, 64)),
            uk_proj: Array2::zeros((32, 64)),
            uv_proj: Array2::zeros((32, 64)),
            qr_proj: None,
            kr_proj: None,
        });

        // 创建8个专家（不使用clone）
        let _moe = Some(QuantizedMoEWeights {
            experts: vec![
                QuantizedFfnWeights {
                    gate_proj: Array2::zeros((64, 128)),
                    up_proj: Array2::zeros((64, 128)),
                    down_proj: Array2::zeros((128, 64)),
                },
                QuantizedFfnWeights {
                    gate_proj: Array2::zeros((64, 128)),
                    up_proj: Array2::zeros((64, 128)),
                    down_proj: Array2::zeros((128, 64)),
                },
                QuantizedFfnWeights {
                    gate_proj: Array2::zeros((64, 128)),
                    up_proj: Array2::zeros((64, 128)),
                    down_proj: Array2::zeros((128, 64)),
                },
                QuantizedFfnWeights {
                    gate_proj: Array2::zeros((64, 128)),
                    up_proj: Array2::zeros((64, 128)),
                    down_proj: Array2::zeros((128, 64)),
                },
                QuantizedFfnWeights {
                    gate_proj: Array2::zeros((64, 128)),
                    up_proj: Array2::zeros((64, 128)),
                    down_proj: Array2::zeros((128, 64)),
                },
                QuantizedFfnWeights {
                    gate_proj: Array2::zeros((64, 128)),
                    up_proj: Array2::zeros((64, 128)),
                    down_proj: Array2::zeros((128, 64)),
                },
                QuantizedFfnWeights {
                    gate_proj: Array2::zeros((64, 128)),
                    up_proj: Array2::zeros((64, 128)),
                    down_proj: Array2::zeros((128, 64)),
                },
                QuantizedFfnWeights {
                    gate_proj: Array2::zeros((64, 128)),
                    up_proj: Array2::zeros((64, 128)),
                    down_proj: Array2::zeros((128, 64)),
                },
            ],
            router: Array2::zeros((8, 16)),
            top_k: 2,
        });

        // 验证可以创建完整的layer weights结构
        let _layer_with_optional = QuantizedLayerWeights {
            attention: QuantizedAttentionWeights {
                q_proj: Array2::zeros((64, 128)),
                k_proj: Array2::zeros((32, 128)),
                v_proj: Array2::zeros((32, 128)),
                o_proj: Array2::zeros((64, 128)),
            },
            mla: _mla,
            ffn: QuantizedFfnWeights {
                gate_proj: Array2::zeros((256, 128)),
                up_proj: Array2::zeros((256, 128)),
                down_proj: Array2::zeros((128, 256)),
            },
            moe: _moe,
            input_layernorm: Array1::from_vec(vec![0.1; 128]),
            post_attention_layernorm: Array1::from_vec(vec![0.2; 128]),
        };
    }

    /// 测试：FfnWeightArrays的Clone特性
    #[test]
    fn test_ffn_weight_arrays_clone() {
        // 注意：FfnWeightArrays 实现了 Clone（通过 derive），但 QuantizedFfnWeights 没有
        // 这里测试 FfnWeightArrays 的 Clone
        let original = FfnWeightArrays {
            gate_proj: Array2::from_shape_vec((3, 4), vec![1.0; 12]).unwrap(),
            up_proj: Array2::from_shape_vec((3, 4), vec![2.0; 12]).unwrap(),
            down_proj: Array2::from_shape_vec((4, 3), vec![3.0; 12]).unwrap(),
        };

        let cloned = original.clone();

        assert_eq!(cloned.gate_proj.shape(), original.gate_proj.shape());
        assert_eq!(cloned.up_proj.shape(), original.up_proj.shape());
        assert_eq!(cloned.down_proj.shape(), original.down_proj.shape());

        // 验证数据一致性（使用迭代器比较）
        let gate_diff: f32 = cloned
            .gate_proj
            .iter()
            .zip(original.gate_proj.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>();
        assert!(gate_diff < 1e-10, "gate_proj数据不一致");

        let up_diff: f32 = cloned
            .up_proj
            .iter()
            .zip(original.up_proj.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>();
        assert!(up_diff < 1e-10, "up_proj数据不一致");
    }

    /// 测试：MlaWeightArrays的完整结构验证
    #[test]
    fn test_mla_weight_arrays_complete_structure() {
        // 测试所有必需字段都存在且维度正确
        let mla = QuantizedMlaWeights {
            q_proj: Array2::zeros((512, 256)),        // query投影
            o_proj: Array2::zeros((512, 256)),        // output投影
            dkv_proj: Array2::zeros((128, 256)),      // decompressed KV投影
            uk_proj: Array2::zeros((128, 256)),       // uncompressed KV投影
            uv_proj: Array2::zeros((128, 256)),       // uncompressed V投影
            qr_proj: Some(Array2::zeros((256, 256))), // 可选：compressed Q恢复
            kr_proj: Some(Array2::zeros((256, 256))), // 可选：compressed K恢复
        };

        assert_eq!(mla.q_proj.shape(), [512, 256]);
        assert_eq!(mla.o_proj.shape(), [512, 256]);
        assert_eq!(mla.dkv_proj.shape(), [128, 256]);
        assert_eq!(mla.uk_proj.shape(), [128, 256]);
        assert_eq!(mla.uv_proj.shape(), [128, 256]);

        // 可选字段应该存在
        assert!(mla.qr_proj.is_some());
        assert!(mla.kr_proj.is_some());
        assert_eq!(mla.qr_proj.as_ref().unwrap().shape(), [256, 256]);
    }

    /// 测试：MoEWeightArrays的不同专家数量配置
    #[test]
    fn test_moe_weight_arrays_various_expert_counts() {
        for num_experts in [1usize, 4, 8, 16, 32] {
            let experts: Vec<FfnWeightArrays> = (0..num_experts)
                .map(|_| FfnWeightArrays {
                    gate_proj: Array2::zeros((64, 32)),
                    up_proj: Array2::zeros((64, 32)),
                    down_proj: Array2::zeros((32, 64)),
                })
                .collect();

            let moe = MoEWeightArrays {
                experts,
                router: Array2::zeros((num_experts, 8)),
                top_k: if num_experts > 2 { 2 } else { 1 },
            };

            assert_eq!(moe.experts.len(), num_experts);
            assert_eq!(moe.router.nrows(), num_experts);

            // top_k不应该超过专家数量
            assert!(
                moe.top_k <= num_experts,
                "top_k={} 不应超过 expert_count={}",
                moe.top_k,
                num_experts
            );
        }
    }

    /// 测试：各种权重结构体的Debug特性输出
    #[test]
    fn test_weight_structs_debug_output() {
        // QuantizedWeights Debug（实现Debug的结构体）
        let weights = QuantizedWeights::new(Array2::ones((2, 2)), GgufTensorType::F32);
        let debug_str = format!("{:?}", weights);
        assert!(!debug_str.is_empty());
        assert!(debug_str.contains("QuantizedWeights"));
    }

    /// 测试：边界情况 - 空数组和单元素数组
    #[test]
    fn test_edge_cases_empty_and_single_element() {
        // 单元素权重
        let single_data = Array2::from_shape_vec((1, 1), vec![42.0]).unwrap();
        let weights = QuantizedWeights::new(single_data.clone(), GgufTensorType::F32);
        assert_eq!(weights.num_elements(), 1);
        assert_eq!(weights.original_shape, vec![1, 1]);

        // 注意力权重的最小有效尺寸
        let min_attn = AttentionWeightArrays {
            q_proj: Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
            k_proj: Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
            v_proj: Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
            o_proj: Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
        };
        assert_eq!(min_attn.q_proj.len(), 1);

        // FFN权重的最小有效尺寸
        let min_ffn = FfnWeightArrays {
            gate_proj: Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
            up_proj: Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
            down_proj: Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
        };
        assert_eq!(min_ffn.gate_proj.len(), 1);
    }

    /// 测试：权重数据的一致性验证（形状匹配）
    #[test]
    fn test_weight_consistency_validation() {
        // 创建一致的注意力权重（Q和O相同行数，K和V相同行数）
        let consistent_attn = AttentionWeightArrays {
            q_proj: Array2::zeros((64, 128)), // 64个query向量，每个128维
            k_proj: Array2::zeros((32, 128)), // 32个key向量
            v_proj: Array2::zeros((32, 128)), // 32个value向量
            o_proj: Array2::zeros((64, 128)), // 64个output向量
        };

        // 验证维度关系
        assert_eq!(
            consistent_attn.q_proj.ncols(),
            consistent_attn.k_proj.ncols()
        ); // head_dim一致
        assert_eq!(
            consistent_attn.k_proj.nrows(),
            consistent_attn.v_proj.nrows()
        ); // kv_len一致
        assert_eq!(
            consistent_attn.q_proj.nrows(),
            consistent_attn.o_proj.nrows()
        ); // query_len一致

        // 创建一致的FFN权重
        let consistent_ffn = FfnWeightArrays {
            gate_proj: Array2::zeros((256, 128)), // intermediate_size x hidden_size
            up_proj: Array2::zeros((256, 128)),   // intermediate_size x hidden_size
            down_proj: Array2::zeros((128, 256)), // hidden_size x intermediate_size
        };

        // 验证FFN维度关系
        assert_eq!(
            consistent_ffn.gate_proj.nrows(),
            consistent_ffn.up_proj.nrows()
        ); // intermediate_size
        assert_eq!(
            consistent_ffn.gate_proj.ncols(),
            consistent_ffn.down_proj.nrows()
        ); // hidden_size
        assert_eq!(
            consistent_ffn.gate_proj.nrows(),
            consistent_ffn.down_proj.ncols()
        ); // intermediate_size
    }

    /// 测试：不同GgufTensorType的处理
    #[test]
    fn test_different_tensor_types() {
        let data = Array2::zeros((4, 4));

        // 测试不同的tensor类型（只测试存在的类型）
        for tensor_type in [
            GgufTensorType::F32,
            GgufTensorType::F16,
            GgufTensorType::Q4_0,
            GgufTensorType::Q4_1,
            GgufTensorType::Q8_0,
        ] {
            let weights = QuantizedWeights::new(data.clone(), tensor_type);
            assert_eq!(weights.tensor_type, tensor_type);
            assert_eq!(weights.num_elements(), 16);
        }
    }

    /// 测试：大尺寸权重的处理能力
    #[test]
    fn test_large_weight_dimensions() {
        // 模拟大型模型权重维度
        let large_q: Array2<f32> = Array2::zeros((4096, 4096)); // 典型的LLM Q投影
        let large_k: Array2<f32> = Array2::zeros((1024, 4096)); // GQA时K更小
        let _large_v: Array2<f32> = Array2::zeros((1024, 4096));
        let _large_o: Array2<f32> = Array2::zeros((4096, 4096));

        let _large_attn = AttentionWeightArrays {
            q_proj: large_q,
            k_proj: large_k,
            v_proj: Array2::zeros((1024, 4096)),
            o_proj: Array2::zeros((4096, 4096)),
        };

        // 验证元素数量（使用len()而不是num_elements()）
        assert_eq!(_large_attn.q_proj.len(), 4096 * 4096); // 16M元素
        assert_eq!(_large_attn.k_proj.len(), 1024 * 4096); // 4M元素

        // 大型FFN
        let _large_ffn = FfnWeightArrays {
            gate_proj: Array2::zeros((11008, 4096)), // Llama-style FFN
            up_proj: Array2::zeros((11008, 4096)),
            down_proj: Array2::zeros((4096, 11008)),
        };

        assert_eq!(_large_ffn.gate_proj.len(), 11008 * 4096);
    }

    /// 测试：权重数值范围和统计特性
    #[test]
    fn test_weight_value_statistics() {
        // 创建已知分布的权重
        let data = Array2::from_shape_vec((100, 50), {
            (0..5000)
                .map(|idx| {
                    let i = idx / 50;
                    let j = idx % 50;
                    ((i * j) as f32 % 10.0) - 5.0 // 范围 [-5, 5]
                })
                .collect::<Vec<f32>>()
        })
        .unwrap();

        let weights = QuantizedWeights::new(data.clone(), GgufTensorType::F32);

        // 统计信息
        let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean_val = data.iter().sum::<f32>() / (data.len() as f32);

        assert!(min_val >= -5.0 && min_val <= 5.0);
        assert!(max_val >= -5.0 && max_val <= 5.0);
        assert!(mean_val >= -5.0 && mean_val <= 5.0);

        // 元素数量正确
        assert_eq!(weights.num_elements(), 5000); // 100 * 50

        // 形状记录正确
        assert_eq!(weights.original_shape, vec![100, 50]);
    }
}
