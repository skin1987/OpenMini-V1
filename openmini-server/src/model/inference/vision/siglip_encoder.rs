use crate::model::inference::error::{InferenceError, InferenceResult};
use ndarray::{Array1, Array2, Array3};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SigLIPEncoderConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub intermediate_size: usize,
    pub num_patches: usize,
}

impl Default for SigLIPEncoderConfig {
    fn default() -> Self {
        let image_size = 896;
        let patch_size = 14;
        Self {
            image_size,
            patch_size,
            hidden_size: 1152,
            num_hidden_layers: 26,
            num_attention_heads: 16,
            num_channels: 3,
            intermediate_size: 4304,
            num_patches: (image_size / patch_size).pow(2),
        }
    }
}

pub struct ViTTransformerLayer {
    pub attn_norm: Array1<f32>,
    pub q_proj: Array2<f32>,
    pub k_proj: Array2<f32>,
    pub v_proj: Array2<f32>,
    pub o_proj: Array2<f32>,
    pub ffn_norm: Array1<f32>,
    pub gate_proj: Array2<f32>,
    pub up_proj: Array2<f32>,
    pub down_proj: Array2<f32>,
}

impl ViTTransformerLayer {
    pub fn from_gguf_weights(
        weights: &HashMap<String, Vec<f32>>,
        prefix: &str,
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
    ) -> InferenceResult<Self> {
        Ok(Self {
            attn_norm: load_1d_weight(weights, &format!("{}.layer_norm_1.weight", prefix), hidden_size)?,
            q_proj: load_2d_weight(weights, &format!("{}.self_attn.q_proj.weight", prefix), hidden_size, hidden_size)?,
            k_proj: load_2d_weight(weights, &format!("{}.self_attn.k_proj.weight", prefix), hidden_size, hidden_size)?,
            v_proj: load_2d_weight(weights, &format!("{}.self_attn.v_proj.weight", prefix), hidden_size, hidden_size)?,
            o_proj: load_2d_weight(weights, &format!("{}.self_attn.o_proj.weight", prefix), hidden_size, hidden_size)?,
            ffn_norm: load_1d_weight(weights, &format!("{}.layer_norm_2.weight", prefix), hidden_size)?,
            gate_proj: load_2d_weight(weights, &format!("{}.mlp.gate_proj.weight", prefix), intermediate_size, hidden_size)?,
            up_proj: load_2d_weight(weights, &format!("{}.mlp.up_proj.weight", prefix), intermediate_size, hidden_size)?,
            down_proj: load_2d_weight(weights, &format!("{}.mlp.down_proj.weight", prefix), hidden_size, intermediate_size)?,
        })
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let (seq_len, hidden_size) = x.dim();

        let normalized = self.layer_norm(x, &self.attn_norm);

        let q = self.matmul(&normalized, &self.q_proj);
        let k = self.matmul(&normalized, &self.k_proj);
        let v = self.matmul(&normalized, &self.v_proj);

        let attn_output = self.multi_head_attention(&q, &k, &v, seq_len, hidden_size);

        let attn_out = self.matmul(&attn_output, &self.o_proj);

        let residual = x + &attn_out;

        let ffn_normalized = self.layer_norm(&residual, &self.ffn_norm);

        let gate = self.matmul(&ffn_normalized, &self.gate_proj);
        let up = self.matmul(&ffn_normalized, &self.up_proj);

        let activated = self.silu(&gate) * &up;

        let ffn_output = self.matmul(&activated, &self.down_proj);

        residual + &ffn_output
    }

    fn layer_norm(&self, x: &Array2<f32>, weight: &Array1<f32>) -> Array2<f32> {
        let (seq_len, hidden_size) = x.dim();
        let eps = 1e-6;

        let mut result = Array2::<f32>::zeros(x.raw_dim());

        for i in 0..seq_len {
            let row = x.row(i);
            let sum: f32 = row.iter().sum();
            let mean = sum / hidden_size as f32;
            let var = row.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / hidden_size as f32;
            let std = (var + eps).sqrt();

            for j in 0..hidden_size {
                result[[i, j]] = weight[j] * (x[[i, j]] - mean) / std;
            }
        }

        result
    }

    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        let (m, k1) = a.dim();
        let (_k2, n) = b.dim();

        let mut result = Array2::<f32>::zeros((m, n));

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..k1 {
                    sum += a[[i, k]] * b[[k, j]];
                }
                result[[i, j]] = sum;
            }
        }

        result
    }

    fn multi_head_attention(
        &self,
        q: &Array2<f32>,
        k: &Array2<f32>,
        v: &Array2<f32>,
        seq_len: usize,
        hidden_size: usize,
    ) -> Array2<f32> {
        let head_dim = hidden_size / 16;
        let scale = (head_dim as f32).powf(-0.5);

        let mut output = Array2::<f32>::zeros((seq_len, hidden_size));

        for head in 0..16 {
            let start = head * head_dim;
            let end = start + head_dim;

            let q_head = q.slice(ndarray::s![.., start..end]);
            let k_head = k.slice(ndarray::s![.., start..end]);
            let v_head = v.slice(ndarray::s![.., start..end]);

            let scores = self.compute_attention_scores(&q_head, &k_head, seq_len, head_dim, scale);
            let attn_weights = self.softmax(&scores, seq_len);

            for i in 0..seq_len {
                for j in 0..seq_len {
                    for d in 0..head_dim {
                        output[[i, start + d]] += attn_weights[[i, j]] * v_head[[j, d]];
                    }
                }
            }
        }

        output
    }

    fn compute_attention_scores(
        &self,
        q: &ndarray::ArrayView2<f32>,
        k: &ndarray::ArrayView2<f32>,
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Array2<f32> {
        let mut scores = Array2::<f32>::zeros((seq_len, seq_len));

        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut sum = 0.0;
                for d in 0..head_dim {
                    sum += q[[i, d]] * k[[j, d]];
                }
                scores[[i, j]] = sum * scale;
            }
        }

        scores
    }

    fn softmax(&self, x: &Array2<f32>, seq_len: usize) -> Array2<f32> {
        let mut result = Array2::<f32>::zeros((seq_len, seq_len));

        for i in 0..seq_len {
            let max_val = x.row(i).iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0;

            for j in 0..seq_len {
                let exp_val = (x[[i, j]] - max_val).exp();
                result[[i, j]] = exp_val;
                exp_sum += exp_val;
            }

            for j in 0..seq_len {
                result[[i, j]] /= exp_sum;
            }
        }

        result
    }

    fn silu(&self, x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| v * (1.0 / (1.0 + (-v).exp())))
    }
}

pub struct SigLIPEncoder {
    config: SigLIPEncoderConfig,
    patch_embedding: Array2<f32>,
    position_embedding: Array2<f32>,
    layers: Vec<ViTTransformerLayer>,
    layernorm: Array1<f32>,
    projection: Option<Array2<f32>>,
}

impl SigLIPEncoder {
    pub fn from_gguf_weights(
        config: &SigLIPEncoderConfig,
        weights: &HashMap<String, Vec<f32>>,
        prefix: &str,
    ) -> InferenceResult<Self> {
        let patch_emb_key = format!("{}.embeddings.patch_embedding.weight", prefix);
        let patch_embedding = load_2d_weight(&weights, &patch_emb_key, config.hidden_size, config.patch_size * config.patch_size * 3)?;

        let pos_emb_key = format!("{}.embeddings.position_embedding.weight", prefix);
        let position_embedding = load_2d_weight(&weights, &pos_emb_key, config.num_patches + 1, config.hidden_size)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer_prefix = format!("{}.encoder.layers.{}", prefix, i);
            layers.push(ViTTransformerLayer::from_gguf_weights(
                &weights,
                &layer_prefix,
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
            )?);
        }

        let ln_key = format!("{}.post_layernorm.weight", prefix);
        let layernorm = load_1d_weight(&weights, &ln_key, config.hidden_size)?;

        let proj_key = format!("{}.visual_projection.weight", prefix);
        let projection = if weights.contains_key(&proj_key) {
            Some(load_2d_weight(&weights, &proj_key, config.hidden_size, config.hidden_size)?)
        } else {
            None
        };

        Ok(Self {
            config: config.clone(),
            patch_embedding,
            position_embedding,
            layers,
            layernorm,
            projection,
        })
    }

    pub fn new(
        config: SigLIPEncoderConfig,
        weights: SigLIPWeights,
    ) -> InferenceResult<Self> {
        let num_patches = (config.image_size / config.patch_size).pow(2);

        let expected_patch_dim = config.patch_size.pow(2) * config.num_channels;
        if weights.patch_embedding.dim() != (config.hidden_size, expected_patch_dim) {
            return Err(InferenceError::config(format!(
                "patch_embedding dim mismatch: expected ({}, {}), got {:?}",
                config.hidden_size, expected_patch_dim, weights.patch_embedding.dim()
            )));
        }

        if weights.position_embedding.dim() != (num_patches + 1, config.hidden_size) {
            return Err(InferenceError::config(format!(
                "position_embedding dim mismatch: expected ({}, {}), got {:?}",
                num_patches + 1,
                config.hidden_size,
                weights.position_embedding.dim()
            )));
        }

        if weights.layers.len() != config.num_hidden_layers {
            return Err(InferenceError::config(format!(
                "Expected {} transformer layers, got {}",
                config.num_hidden_layers,
                weights.layers.len()
            )));
        }

        Ok(Self {
            config,
            patch_embedding: weights.patch_embedding,
            position_embedding: weights.position_embedding,
            layers: weights.layers,
            layernorm: weights.layernorm,
            projection: weights.projection,
        })
    }

    pub fn encode(&self, image: &Array3<u8>) -> InferenceResult<Array2<f32>> {
        let shape = image.shape();
        let (h, w, c) = (shape[0], shape[1], shape[2]);

        if c != self.config.num_channels {
            return Err(InferenceError::image_preprocess(format!(
                "Expected {} channels, got {}",
                self.config.num_channels, c
            )));
        }

        if h != self.config.image_size || w != self.config.image_size {
            return Err(InferenceError::image_preprocess(format!(
                "Image size must be {}x{}, got {}x{}",
                self.config.image_size, self.config.image_size, h, w
            )));
        }

        let image_f32: Array3<f32> = image.mapv(|v| v as f32 / 255.0);
        let patches = self.extract_patches(&image_f32);

        let mut hidden_states = self.patch_embed(&patches);
        hidden_states = hidden_states + &self.position_embedding;

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states);
        }

        hidden_states = self.final_layer_norm(&hidden_states);

        if let Some(ref proj) = self.projection {
            hidden_states = self.apply_projection(&hidden_states, proj);
        }

        Ok(hidden_states)
    }

    fn extract_patches(&self, image: &Array3<f32>) -> Array3<f32> {
        let patch_size = self.config.patch_size;
        let num_patches_h = self.config.image_size / patch_size;
        let num_patches_w = self.config.image_size / patch_size;
        let patch_dim = patch_size.pow(2) * self.config.num_channels;

        let mut patches = Array3::<f32>::zeros((num_patches_h * num_patches_w, patch_dim, 1));

        for py in 0..num_patches_h {
            for px in 0..num_patches_w {
                let patch_idx = py * num_patches_w + px;
                let mut pixel_idx = 0;

                for y in 0..patch_size {
                    for x in 0..patch_size {
                        for c in 0..self.config.num_channels {
                            patches[[patch_idx, pixel_idx, 0]] =
                                image[[py * patch_size + y, px * patch_size + x, c]];
                            pixel_idx += 1;
                        }
                    }
                }
            }
        }

        patches
    }

    fn patch_embed(&self, patches: &Array3<f32>) -> Array2<f32> {
        let (num_patches, patch_dim, _) = patches.dim();

        let flat_patches = patches.slice(ndarray::s![.., .., 0]).to_owned();

        let mut hidden = Array2::<f32>::zeros((num_patches + 1, self.config.hidden_size));

        for i in 0..num_patches {
            for j in 0..self.config.hidden_size {
                let mut sum = 0.0;
                for k in 0..patch_dim {
                    sum += flat_patches[[i, k]] * self.patch_embedding[[j, k]];
                }
                hidden[[i, j]] = sum;
            }
        }

        hidden
    }

    fn final_layer_norm(&self, x: &Array2<f32>) -> Array2<f32> {
        let (seq_len, hidden_size) = x.dim();
        let eps = 1e-6;

        let mut result = Array2::<f32>::zeros(x.raw_dim());

        for i in 0..seq_len {
            let row = x.row(i);
            let sum: f32 = row.iter().sum();
            let mean = sum / hidden_size as f32;
            let var = row
                .iter()
                .map(|&v| (v - mean).powi(2))
                .sum::<f32>()
                / hidden_size as f32;
            let std = (var + eps).sqrt();

            for j in 0..hidden_size {
                result[[i, j]] = self.layernorm[j] * (x[[i, j]] - mean) / std;
            }
        }

        result
    }

    fn apply_projection(&self, x: &Array2<f32>, proj: &Array2<f32>) -> Array2<f32> {
        let (seq_len, hidden_size) = x.dim();
        let (_, proj_dim) = proj.dim();

        let mut result = Array2::<f32>::zeros((seq_len, proj_dim));

        for i in 0..seq_len {
            for j in 0..proj_dim {
                let mut sum = 0.0;
                for k in 0..hidden_size {
                    sum += x[[i, k]] * proj[[k, j]];
                }
                result[[i, j]] = sum;
            }
        }

        result
    }

    pub fn config(&self) -> &SigLIPEncoderConfig {
        &self.config
    }

    pub fn num_patches(&self) -> usize {
        (self.config.image_size / self.config.patch_size).pow(2)
    }
}

pub struct SigLIPWeights {
    pub patch_embedding: Array2<f32>,
    pub position_embedding: Array2<f32>,
    pub layers: Vec<ViTTransformerLayer>,
    pub layernorm: Array1<f32>,
    pub projection: Option<Array2<f32>>,
}

fn load_1d_weight(
    weights: &HashMap<String, Vec<f32>>,
    key: &str,
    size: usize,
) -> InferenceResult<Array1<f32>> {
    match weights.get(key) {
        Some(data) if data.len() == size => Ok(Array1::from_vec(data.clone())),
        Some(data) => Err(InferenceError::WeightLoadError {
            message: format!("{}: expected {} elements, got {}", key, size, data.len()),
            source: None,
        }),
        None => Err(InferenceError::MissingWeight { name: key.to_string() }),
    }
}

fn load_2d_weight(
    weights: &HashMap<String, Vec<f32>>,
    key: &str,
    rows: usize,
    cols: usize,
) -> InferenceResult<Array2<f32>> {
    match weights.get(key) {
        Some(data) if data.len() == rows * cols => {
            Ok(Array2::from_shape_vec((rows, cols), data.clone()).unwrap())
        }
        Some(data) => Err(InferenceError::weight_load(format!(
            "{}: expected {}x{}={} elements, got {}", key, rows, cols, rows * cols, data.len()
        ))),
        None => Err(InferenceError::weight_load(format!("Missing weight: {}", key))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SigLIPEncoderConfig::default();
        assert_eq!(config.image_size, 896);
        assert_eq!(config.patch_size, 14);
        assert_eq!(config.hidden_size, 1152);
        assert_eq!(config.num_hidden_layers, 26);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_channels, 3);
        assert_eq!(config.intermediate_size, 4304);
    }

    #[test]
    fn test_vit_layer_forward() {
        let seq_len = 4;
        let hidden_size = 8;
        let intermediate = 16;

        let layer = create_test_vit_layer(hidden_size, intermediate);
        let input = Array2::<f32>::zeros((seq_len, hidden_size));
        let output = layer.forward(&input);

        assert_eq!(output.dim(), (seq_len, hidden_size));
    }

    #[test]
    fn test_siglip_encoder_creation() {
        let config = SigLIPEncoderConfig::default();
        let weights = create_test_weights(&config);

        let encoder = SigLIPEncoder::new(config.clone(), weights);
        assert!(encoder.is_ok());

        let encoder = encoder.unwrap();
        assert_eq!(encoder.num_patches(), 4096);
    }

    #[test]
    fn test_siglip_encode() {
        let config = SigLIPEncoderConfig {
            image_size: 56,
            patch_size: 14,
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_channels: 3,
            intermediate_size: 128,
            num_patches: 16, // (56/14) * (56/14) = 16
        };

        let weights = create_test_weights(&config);
        let encoder = SigLIPEncoder::new(config, weights).unwrap();

        let image = Array3::<u8>::zeros((56, 56, 3));
        let features = encoder.encode(&image);

        assert!(features.is_ok());
        let feat = features.unwrap();
        assert_eq!(feat.dim(), (17, 64)); // 16 patches + 1 CLS token
    }

    #[test]
    fn test_invalid_image_channels() {
        let config = SigLIPEncoderConfig {
            image_size: 56,
            patch_size: 14,
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_channels: 3,
            intermediate_size: 128,
            num_patches: 16,
        };

        let weights = create_test_weights(&config);
        let encoder = SigLIPEncoder::new(config, weights).unwrap();

        let invalid_image = Array3::<u8>::zeros((56, 56, 4)); // 4 channels instead of 3
        let result = encoder.encode(&invalid_image);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_image_size() {
        let config = SigLIPEncoderConfig {
            image_size: 56,
            patch_size: 14,
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_channels: 3,
            intermediate_size: 128,
            num_patches: 16,
        };

        let weights = create_test_weights(&config);
        let encoder = SigLIPEncoder::new(config, weights).unwrap();

        let wrong_size_image = Array3::<u8>::zeros((224, 224, 3)); // wrong size
        let result = encoder.encode(&wrong_size_image);
        assert!(result.is_err());
    }

    fn create_test_vit_layer(hidden_size: usize, intermediate: usize) -> ViTTransformerLayer {
        ViTTransformerLayer {
            attn_norm: Array1::<f32>::ones(hidden_size),
            q_proj: Array2::<f32>::from_shape_fn((hidden_size, hidden_size), |_| 0.01),
            k_proj: Array2::<f32>::from_shape_fn((hidden_size, hidden_size), |_| 0.01),
            v_proj: Array2::<f32>::from_shape_fn((hidden_size, hidden_size), |_| 0.01),
            o_proj: Array2::<f32>::from_shape_fn((hidden_size, hidden_size), |_| 0.01),
            ffn_norm: Array1::<f32>::ones(hidden_size),
            gate_proj: Array2::<f32>::from_shape_fn((hidden_size, intermediate), |_| 0.01),
            up_proj: Array2::<f32>::from_shape_fn((hidden_size, intermediate), |_| 0.01),
            down_proj: Array2::<f32>::from_shape_fn((intermediate, hidden_size), |_| 0.01),
        }
    }

    fn create_test_weights(config: &SigLIPEncoderConfig) -> SigLIPWeights {
        let num_patches = (config.image_size / config.patch_size).pow(2);
        let patch_dim = config.patch_size.pow(2) * config.num_channels;

        let layers: Vec<ViTTransformerLayer> = (0..config.num_hidden_layers)
            .map(|_| create_test_vit_layer(config.hidden_size, config.intermediate_size))
            .collect();

        SigLIPWeights {
            patch_embedding: Array2::<f32>::from_shape_fn(
                (config.hidden_size, patch_dim),
                |_| 0.01,
            ),
            position_embedding: Array2::<f32>::from_shape_fn(
                (num_patches + 1, config.hidden_size),
                |_| 0.01,
            ),
            layers,
            layernorm: Array1::<f32>::ones(config.hidden_size),
            projection: None,
        }
    }
}
