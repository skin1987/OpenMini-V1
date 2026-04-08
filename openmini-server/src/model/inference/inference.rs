//! 推理引擎 - 高级推理接口
//!
//! 提供统一的推理引擎接口，整合模型权重、分词器、采样器。
//! 支持文本生成和多模态推理。

#![allow(dead_code)]

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};

use rayon::prelude::*;

use super::error::{InferenceError, InferenceResult};
use super::image_preprocess::{ImagePreprocessor, ImagePreprocessorConfig};
use super::model::ModelConfig;
use super::model::MultimodalTransformer;
use super::quant_loader::QuantizedWeightLoader;
use super::sampler::GenerateParams;
use super::tokenizer::Tokenizer;

/// 推理统计信息
#[derive(Debug, Clone, Default)]
pub struct InferenceStats {
    /// 总推理时间（毫秒）
    pub inference_time_ms: u64,
    /// 总推理时间（秒，浮点精度）
    pub inference_time_secs: f64,
    /// 总 token 数量（prompt + generated）
    pub total_tokens: usize,
    /// Prompt token 数量
    pub prompt_tokens: usize,
    /// 生成的 token 数量
    pub generated_tokens: usize,
    /// 首 token 生成时间（TTFT，毫秒）
    pub time_to_first_token_ms: Option<u64>,
    /// 每 token 平均生成时间（TPOT，毫秒）
    pub time_per_output_token_ms: Option<f64>,
    /// 总体 token 处理速度
    ///
    /// **注意**：此指标包含 prompt 处理时间。
    /// 对于长 prompt 或短生成场景，该值可能低估实际生成速度。
    ///
    /// 如需精确测量生成速度，应在生成过程中记录首 token 时间。
    ///
    /// 计算公式：(prompt_tokens + generated_tokens) / inference_time_secs
    pub tokens_per_second: f32,
}

impl InferenceStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// 创建带时间统计的推理统计
    ///
    /// # 参数
    /// - `inference_time_ms`: 总推理时间（毫秒）
    /// - `prompt_tokens`: prompt token 数量
    /// - `generated_tokens`: 生成的 token 数量
    pub fn with_timing(
        inference_time_ms: u64,
        prompt_tokens: usize,
        generated_tokens: usize,
    ) -> Self {
        let total_tokens = prompt_tokens + generated_tokens;
        let inference_time_secs = inference_time_ms as f64 / 1000.0;
        let tokens_per_second = if inference_time_ms > 0 {
            (total_tokens as f32) / (inference_time_ms as f32 / 1000.0)
        } else {
            0.0
        };

        Self {
            inference_time_ms,
            inference_time_secs,
            total_tokens,
            prompt_tokens,
            generated_tokens,
            time_to_first_token_ms: None,
            time_per_output_token_ms: None,
            tokens_per_second,
        }
    }

    /// 创建带高精度时间统计的推理统计
    ///
    /// # 参数
    /// - `inference_time_secs`: 总推理时间（秒，浮点精度）
    /// - `prompt_tokens`: prompt token 数量
    /// - `generated_tokens`: 生成的 token 数量
    /// - `time_to_first_token_ms`: 首 token 生成时间（TTFT）
    pub fn with_timing_high_precision(
        inference_time_secs: f64,
        prompt_tokens: usize,
        generated_tokens: usize,
        time_to_first_token_ms: Option<u64>,
    ) -> Self {
        let total_tokens = prompt_tokens + generated_tokens;
        let inference_time_ms = (inference_time_secs * 1000.0) as u64;
        let tokens_per_second = if inference_time_secs > 0.0 {
            total_tokens as f32 / inference_time_secs as f32
        } else {
            0.0
        };

        let time_per_output_token_ms = if generated_tokens > 0 && time_to_first_token_ms.is_some() {
            let ttft = time_to_first_token_ms.unwrap() as f64;
            let decode_time = inference_time_secs * 1000.0 - ttft;
            Some(decode_time / generated_tokens as f64)
        } else {
            None
        };

        Self {
            inference_time_ms,
            inference_time_secs,
            total_tokens,
            prompt_tokens,
            generated_tokens,
            time_to_first_token_ms,
            time_per_output_token_ms,
            tokens_per_second,
        }
    }
}

/// 推理引擎
///
/// 整合所有推理组件，提供统一的推理接口。
pub struct InferenceEngine {
    /// 多模态模型
    model: Arc<MultimodalTransformer>,
    /// 分词器
    tokenizer: Arc<Tokenizer>,
    /// 模型配置
    config: ModelConfig,
    /// 图像预处理器
    image_preprocessor: ImagePreprocessor,
}

impl InferenceEngine {
    /// 从 GGUF 文件创建推理引擎
    ///
    /// # 参数
    /// - `path`: GGUF 文件路径
    ///
    /// # 返回
    /// 成功返回推理引擎实例
    ///
    /// # 错误
    /// - 文件不存在或格式错误
    /// - 必需的元数据缺失（如层数、隐藏大小）
    ///
    /// # 注意
    /// 当前实现会：
    /// 1. 解析 GGUF 文件元数据获取模型配置
    /// 2. 加载嵌入层和语言模型头权重
    /// 3. 加载所有 Transformer 层权重
    ///
    /// # 限制
    /// - 视觉编码器权重尚未加载
    /// - MLA/MoE 权重需要额外处理
    /// - 缺失的可选权重会打印警告但不会失败
    pub fn from_gguf(path: &Path) -> Result<Self, InferenceError> {
        let loader = QuantizedWeightLoader::open(path, true)
            .map_err(|e| InferenceError::model_file(e.to_string(), path))?;

        // 从 GGUF 元数据提取模型配置
        let config = loader
            .get_model_config()
            .map_err(|e| InferenceError::config(e.to_string()))?;

        // 创建模型结构
        let mut model = MultimodalTransformer::new(config.clone());

        // 加载嵌入层权重（必需）
        match loader.load_embedding_weights() {
            Ok(embedding) => model.embedding = embedding,
            Err(e) => warn!("Failed to load embedding weights: {}", e),
        }

        // 加载语言模型头权重（可选，某些模型共享嵌入层）
        match loader.load_tensor("lm_head") {
            Ok(lm_head) => model.lm_head = lm_head.data,
            Err(_) => {
                // 尝试共享嵌入层
                info!("lm_head not found, will use tied embeddings");
            }
        }

        // 加载最终层归一化权重（可选）
        match loader.load_norm_weights("model.norm") {
            Ok(final_norm) => model.final_layernorm = final_norm,
            Err(_) => debug!("model.norm not found, using default"),
        }

        // 加载各层权重
        let mut loaded_layers = 0;
        for layer_idx in 0..config.num_hidden_layers {
            if let Some(layer) = model.layers.get_mut(layer_idx) {
                let mut layer_loaded = false;

                // 加载注意力权重
                match loader.load_attention_weights(layer_idx) {
                    Ok(attn_weights) => {
                        layer.attention.q_proj = attn_weights.q_proj;
                        layer.attention.k_proj = attn_weights.k_proj;
                        layer.attention.v_proj = attn_weights.v_proj;
                        layer.attention.o_proj = attn_weights.o_proj;
                        layer_loaded = true;
                    }
                    Err(e) => {
                        warn!("Layer {} attention weights not found: {}", layer_idx, e);
                    }
                }

                // 加载 FFN 权重
                match loader.load_ffn_weights(layer_idx) {
                    Ok(ffn_weights) => {
                        layer.ffn.gate_proj = ffn_weights.gate_proj;
                        layer.ffn.up_proj = ffn_weights.up_proj;
                        layer.ffn.down_proj = ffn_weights.down_proj;
                        layer_loaded = true;
                    }
                    Err(e) => {
                        warn!("Layer {} FFN weights not found: {}", layer_idx, e);
                    }
                }

                // 加载层归一化权重
                let input_norm_name = format!("model.layers.{}.input_layernorm", layer_idx);
                match loader.load_norm_weights(&input_norm_name) {
                    Ok(input_norm) => {
                        layer.input_layernorm = input_norm;
                        layer_loaded = true;
                    }
                    Err(_) => {}
                }

                let post_attn_norm_name =
                    format!("model.layers.{}.post_attention_layernorm", layer_idx);
                match loader.load_norm_weights(&post_attn_norm_name) {
                    Ok(post_norm) => {
                        layer.post_attention_layernorm = post_norm;
                        layer_loaded = true;
                    }
                    Err(_) => {}
                }

                if layer_loaded {
                    loaded_layers += 1;
                }
            }
        }

        info!(
            "Loaded {}/{} transformer layers from GGUF file",
            loaded_layers, config.num_hidden_layers
        );

        if loaded_layers < config.num_hidden_layers {
            warn!(
                "Only {}/{} layers loaded - model may not work correctly!",
                loaded_layers, config.num_hidden_layers
            );
        }

        let tokenizer = Tokenizer::new();

        // 创建图像预处理器
        let image_preprocessor = ImagePreprocessor::new(ImagePreprocessorConfig::default());

        Ok(Self {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            config,
            image_preprocessor,
        })
    }

    /// 文本生成
    pub fn generate(&self, prompt: &str, params: &GenerateParams) -> InferenceResult<String> {
        let tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| InferenceError::tokenization(e.to_string()))?;

        let generated_ids = self
            .model
            .generate_with_params(&tokens, params)
            .map_err(|e| InferenceError::generation(e.to_string()))?;

        let generated_text = self
            .tokenizer
            .decode(&generated_ids)
            .map_err(|e| InferenceError::tokenization(e.to_string()))?;
        Ok(generated_text)
    }

    /// 多模态推理（文本 + 图像）
    ///
    /// # 参数
    /// - `prompt`: 文本提示
    /// - `image`: 图像输入 (H, W, 3)，RGB 格式，u8 类型
    /// - `params`: 生成参数
    ///
    /// # 图像预处理
    /// - 自动将图像调整到 224×224 像素（双线性插值）
    /// - 像素值归一化由模型内部处理（u8 → f32 / 255.0）
    /// - 支持任意尺寸输入，无需手动预处理
    ///
    /// # 返回
    /// 生成的文本
    pub fn generate_with_image(
        &self,
        prompt: &str,
        image: &ndarray::Array3<u8>,
        params: &GenerateParams,
    ) -> InferenceResult<String> {
        // 预处理图像（自动 resize 到目标尺寸）
        let processed = self
            .image_preprocessor
            .resize_only(image)
            .map_err(|e| InferenceError::image_preprocess(e.to_string()))?;

        let tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| InferenceError::tokenization(e.to_string()))?;

        // 调用多模态生成（传入预处理后的图像）
        let generated_ids = self
            .model
            .generate_multimodal_with_params(&tokens, Some(&processed), params)
            .map_err(|e| InferenceError::generation(e.to_string()))?;

        let generated_text = self
            .tokenizer
            .decode(&generated_ids)
            .map_err(|e| InferenceError::tokenization(e.to_string()))?;
        Ok(generated_text)
    }

    /// 多模态推理（文本 + 图像 + 音频）
    ///
    /// # 参数
    /// - `prompt`: 文本提示
    /// - `image`: 图像输入 (H, W, 3)，RGB 格式（会自动预处理）
    /// - `audio`: 音频特征 (可选)
    /// - `params`: 生成参数
    ///
    /// # 返回
    /// 生成的文本
    pub fn generate_multimodal(
        &self,
        prompt: &str,
        image: Option<&ndarray::Array3<u8>>,
        audio: Option<&ndarray::Array2<f32>>,
        params: &GenerateParams,
    ) -> InferenceResult<String> {
        // 音频特征暂未实现
        if audio.is_some() {
            warn!("generate_multimodal: audio features not yet implemented, ignoring");
        }

        // 预处理图像（如果存在）
        let processed_image = if let Some(img) = image {
            Some(
                self.image_preprocessor
                    .resize_only(img)
                    .map_err(|e| InferenceError::image_preprocess(e.to_string()))?,
            )
        } else {
            None
        };

        let tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| InferenceError::tokenization(e.to_string()))?;

        // 调用多模态生成（带采样参数）
        let generated_ids = self
            .model
            .generate_multimodal_with_params(&tokens, processed_image.as_ref(), params)
            .map_err(|e| InferenceError::generation(e.to_string()))?;

        let generated_text = self
            .tokenizer
            .decode(&generated_ids)
            .map_err(|e| InferenceError::tokenization(e.to_string()))?;
        Ok(generated_text)
    }

    /// 文本生成（带统计信息）
    pub fn generate_with_stats(
        &self,
        prompt: &str,
        params: &GenerateParams,
    ) -> InferenceResult<(String, InferenceStats)> {
        let start_time = Instant::now();

        let tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| InferenceError::tokenization(e.to_string()))?;
        let prompt_tokens = tokens.len();

        let generated_ids = self
            .model
            .generate_with_params(&tokens, params)
            .map_err(|e| InferenceError::generation(e.to_string()))?;
        let generated_tokens = generated_ids.len();

        let generated_text = self
            .tokenizer
            .decode(&generated_ids)
            .map_err(|e| InferenceError::tokenization(e.to_string()))?;

        let elapsed_ms = start_time.elapsed().as_millis() as u64;
        let stats = InferenceStats::with_timing(elapsed_ms, prompt_tokens, generated_tokens);

        Ok((generated_text, stats))
    }

    /// 批处理推理
    ///
    /// 注意：并行处理时，每个任务共享模型和分词器引用。
    /// 确保 Tokenizer 实现了 Send + Sync。
    pub fn batch_generate(
        &self,
        prompts: &[&str],
        params: &GenerateParams,
    ) -> InferenceResult<Vec<String>> {
        let results: Vec<String> = prompts
            .par_iter()
            .map(|prompt| {
                let tokens = self
                    .tokenizer
                    .encode(prompt)
                    .map_err(|e| InferenceError::tokenization(e.to_string()))?;

                let generated_ids = self
                    .model
                    .generate_with_params(&tokens, params)
                    .map_err(|e| InferenceError::generation(e.to_string()))?;

                let generated_text = self
                    .tokenizer
                    .decode(&generated_ids)
                    .map_err(|e| InferenceError::tokenization(e.to_string()))?;
                Ok(generated_text)
            })
            .collect::<Result<Vec<_>, InferenceError>>()?;

        Ok(results)
    }

    /// 批处理推理（带统计信息）
    ///
    /// 注意：并行处理时，每个任务共享模型和分词器引用。
    /// 确保 Tokenizer 实现了 Send + Sync。
    pub fn batch_generate_with_stats(
        &self,
        prompts: &[&str],
        params: &GenerateParams,
    ) -> InferenceResult<(Vec<String>, Vec<InferenceStats>)> {
        let results_and_stats: Vec<(String, InferenceStats)> = prompts
            .par_iter()
            .map(|prompt| {
                let start_time = Instant::now();

                let tokens = self
                    .tokenizer
                    .encode(prompt)
                    .map_err(|e| InferenceError::tokenization(e.to_string()))?;
                let prompt_tokens = tokens.len();

                let generated_ids = self
                    .model
                    .generate_with_params(&tokens, params)
                    .map_err(|e| InferenceError::generation(e.to_string()))?;
                let generated_tokens = generated_ids.len();

                let generated_text = self
                    .tokenizer
                    .decode(&generated_ids)
                    .map_err(|e| InferenceError::tokenization(e.to_string()))?;

                let elapsed_ms = start_time.elapsed().as_millis() as u64;
                let stats =
                    InferenceStats::with_timing(elapsed_ms, prompt_tokens, generated_tokens);

                Ok((generated_text, stats))
            })
            .collect::<Result<Vec<_>, InferenceError>>()?;

        let mut results = Vec::with_capacity(results_and_stats.len());
        let mut stats_list = Vec::with_capacity(results_and_stats.len());
        for (text, stats) in results_and_stats {
            results.push(text);
            stats_list.push(stats);
        }

        Ok((results, stats_list))
    }
}

/// 流式生成器
pub struct StreamGenerator {
    model: Arc<MultimodalTransformer>,
    tokenizer: Arc<Tokenizer>,
    image_preprocessor: ImagePreprocessor,
}

impl StreamGenerator {
    pub fn new(model: MultimodalTransformer, tokenizer: Tokenizer) -> Self {
        Self {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            image_preprocessor: ImagePreprocessor::new(ImagePreprocessorConfig::default()),
        }
    }

    pub fn from_engine(engine: &InferenceEngine) -> Self {
        Self {
            model: Arc::clone(&engine.model),
            tokenizer: Arc::clone(&engine.tokenizer),
            image_preprocessor: ImagePreprocessor::new(ImagePreprocessorConfig::default()),
        }
    }

    /// 流式生成（真正的逐 token 回调）
    ///
    /// 使用模型的 `generate_streaming` 方法，在生成每个 token 后立即回调，
    /// 实现真正的低延迟流式输出。
    ///
    /// # 回调返回值
    /// - `Ok(true)`: 继续生成
    /// - `Ok(false)`: 提前终止生成（如检测到停止词）
    /// - `Err(_)`: 发生错误，终止生成
    pub fn stream_generate<F>(
        &self,
        prompt: &str,
        params: &GenerateParams,
        mut callback: F,
    ) -> InferenceResult<InferenceStats>
    where
        F: FnMut(&str) -> InferenceResult<bool>,
    {
        let start_time = Instant::now();

        let tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| InferenceError::tokenization(e.to_string()))?;
        let prompt_tokens = tokens.len();

        let mut generated_tokens = 0;

        self.model
            .generate_streaming(&tokens, params, |token_id| {
                generated_tokens += 1;
                let token_text = self
                    .tokenizer
                    .decode(&[token_id])
                    .map_err(|e| InferenceError::tokenization(e.to_string()))?;
                callback(&token_text)
            })
            .map_err(|e| InferenceError::generation(e.to_string()))?;

        let elapsed_ms = start_time.elapsed().as_millis() as u64;
        let stats = InferenceStats::with_timing(elapsed_ms, prompt_tokens, generated_tokens);

        Ok(stats)
    }

    /// 流式生成（带图像输入）
    ///
    /// # 参数
    /// - `prompt`: 文本提示
    /// - `image`: 图像输入 (H, W, 3)，RGB 格式，u8 类型
    /// - `params`: 生成参数
    /// - `callback`: 每个 token 生成后调用的回调
    ///
    /// # 图像预处理
    /// - 自动将图像调整到 224×224 像素（双线性插值）
    /// - 像素值归一化由模型内部处理（u8 → f32 / 255.0）
    /// - 支持任意尺寸输入，无需手动预处理
    ///
    /// # 回调返回值
    /// - `Ok(true)`: 继续生成
    /// - `Ok(false)`: 提前终止生成
    /// - `Err(_)`: 发生错误，终止生成
    ///
    /// # 返回
    /// 推理统计信息
    pub fn stream_generate_with_image<F>(
        &self,
        prompt: &str,
        image: &ndarray::Array3<u8>,
        params: &GenerateParams,
        mut callback: F,
    ) -> InferenceResult<InferenceStats>
    where
        F: FnMut(&str) -> InferenceResult<bool>,
    {
        let start_time = Instant::now();

        // 预处理图像（自动 resize 到目标尺寸）
        let processed = self
            .image_preprocessor
            .resize_only(image)
            .map_err(|e| InferenceError::image_preprocess(e.to_string()))?;

        let tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| InferenceError::tokenization(e.to_string()))?;
        let prompt_tokens = tokens.len();

        let mut generated_tokens = 0;

        // 使用多模态流式生成
        self.model
            .generate_streaming_multimodal(&tokens, Some(&processed), params, |token_id| {
                generated_tokens += 1;
                let token_text = self
                    .tokenizer
                    .decode(&[token_id])
                    .map_err(|e| InferenceError::tokenization(e.to_string()))?;
                callback(&token_text)
            })
            .map_err(|e| InferenceError::generation(e.to_string()))?;

        let elapsed_ms = start_time.elapsed().as_millis() as u64;
        let stats = InferenceStats::with_timing(elapsed_ms, prompt_tokens, generated_tokens);

        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_stats_with_timing() {
        let stats = InferenceStats::with_timing(1000, 50, 100);

        assert_eq!(stats.inference_time_ms, 1000);
        assert!((stats.inference_time_secs - 1.0).abs() < 0.001);
        assert_eq!(stats.prompt_tokens, 50);
        assert_eq!(stats.generated_tokens, 100);
        assert_eq!(stats.total_tokens, 150);
        assert!((stats.tokens_per_second - 150.0).abs() < 0.01);
        assert!(stats.time_to_first_token_ms.is_none());
        assert!(stats.time_per_output_token_ms.is_none());
    }

    #[test]
    fn test_inference_stats_with_timing_zero_time() {
        let stats = InferenceStats::with_timing(0, 10, 20);

        assert_eq!(stats.inference_time_ms, 0);
        assert_eq!(stats.inference_time_secs, 0.0);
        assert_eq!(stats.total_tokens, 30);
        assert_eq!(stats.tokens_per_second, 0.0);
    }

    #[test]
    fn test_inference_stats_with_timing_high_precision() {
        let stats = InferenceStats::with_timing_high_precision(1.5, 50, 100, Some(200));

        assert_eq!(stats.inference_time_ms, 1500);
        assert!((stats.inference_time_secs - 1.5).abs() < 0.001);
        assert_eq!(stats.total_tokens, 150);
        assert_eq!(stats.time_to_first_token_ms, Some(200));
        assert!(stats.time_per_output_token_ms.is_some());
        let tpot = stats.time_per_output_token_ms.unwrap();
        assert!((tpot - 13.0).abs() < 0.1);
    }

    #[test]
    fn test_inference_stats_default() {
        let stats = InferenceStats::default();

        assert_eq!(stats.inference_time_ms, 0);
        assert_eq!(stats.inference_time_secs, 0.0);
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.prompt_tokens, 0);
        assert_eq!(stats.generated_tokens, 0);
        assert_eq!(stats.tokens_per_second, 0.0);
        assert!(stats.time_to_first_token_ms.is_none());
        assert!(stats.time_per_output_token_ms.is_none());
    }

    #[test]
    fn test_inference_stats_new() {
        let stats = InferenceStats::new();

        assert_eq!(stats.inference_time_ms, 0);
        assert_eq!(stats.inference_time_secs, 0.0);
        assert_eq!(stats.total_tokens, 0);
    }

    /// 测试：高精度统计中TPOT的计算精度（time_per_output_token_ms）
    #[test]
    fn test_inference_stats_tpot_calculation() {
        // 场景：推理时间2秒，prompt=10，generated=20，TTFT=500ms
        // TPOT = (2000ms - 500ms) / 20 = 75ms/token
        let stats = InferenceStats::with_timing_high_precision(2.0, 10, 20, Some(500));

        assert_eq!(stats.generated_tokens, 20);
        let tpot = stats.time_per_output_token_ms.expect("应该有TPOT");
        assert!((tpot - 75.0).abs() < 0.1, "TPOT应为75.0ms，实际: {}", tpot);
    }

    /// 测试：高精度统计中无TTFT时TPOT为None（边界条件）
    #[test]
    fn test_inference_stats_no_ttft_no_tpot() {
        // 当没有TTFT或生成token数为0时，TPOT应该是None
        let stats = InferenceStats::with_timing_high_precision(
            1.0, 10, 20, None, // 无TTFT
        );

        assert!(stats.time_to_first_token_ms.is_none());
        assert!(
            stats.time_per_output_token_ms.is_none(),
            "无TTFT时TPOT应为None"
        );
    }

    /// 测试：零生成token时的统计信息（边界条件）
    #[test]
    fn test_inference_stats_zero_generated_tokens() {
        let stats = InferenceStats::with_timing(1000, 50, 0); // 只处理prompt

        assert_eq!(stats.total_tokens, 50);
        assert_eq!(stats.generated_tokens, 0);
        assert!((stats.tokens_per_second - 50.0).abs() < 0.01);
    }

    /// 测试：InferenceStats的Clone特性（用于传递和复制统计信息）
    #[test]
    fn test_inference_stats_clone() {
        let stats = InferenceStats::with_timing_high_precision(3.14, 100, 200, Some(300));

        let cloned = stats.clone();
        assert_eq!(cloned.inference_time_ms, stats.inference_time_ms);
        assert!((cloned.inference_time_secs - stats.inference_time_secs).abs() < 1e-10);
        assert_eq!(cloned.prompt_tokens, stats.prompt_tokens);
        assert_eq!(cloned.generated_tokens, stats.generated_tokens);
        assert_eq!(cloned.total_tokens, stats.total_tokens);
        assert_eq!(cloned.time_to_first_token_ms, stats.time_to_first_token_ms);
        assert_eq!(
            cloned.time_per_output_token_ms,
            stats.time_per_output_token_ms
        );
        assert!((cloned.tokens_per_second - stats.tokens_per_second).abs() < 1e-5);
    }

    /// 测试：极小时间值的tokens_per_second精度（数值稳定性）
    #[test]
    fn test_inference_stats_very_small_time() {
        // 极短时间（1毫秒），大量token
        let stats = InferenceStats::with_timing(1, 100, 100);

        assert_eq!(stats.inference_time_ms, 1);
        assert!((stats.inference_time_secs - 0.001).abs() < 0.0001);
        // tokens_per_second 应该非常大但不应溢出
        assert!(
            stats.tokens_per_second > 100000.0,
            "TPS应大于100K，实际: {:.2}",
            stats.tokens_per_second
        );
    }

    /// 测试：Debug trait实现（确保所有字段都被正确格式化）
    #[test]
    fn test_inference_stats_debug() {
        let stats = InferenceStats::with_timing(1234, 56, 78);
        let debug_str = format!("{:?}", stats);

        // 验证Debug输出包含关键字段
        assert!(debug_str.contains("1234"), "应包含inference_time_ms");
        assert!(debug_str.contains("56"), "应包含prompt_tokens");
        assert!(debug_str.contains("78"), "应包含generated_tokens");
    }

    /// 测试：高精度模式下时间转换的舍入误差（毫秒转秒再转回）
    #[test]
    fn test_inference_stats_time_conversion_rounding() {
        // 使用非整数秒数测试转换精度
        let original_secs = 1.234567;
        let stats = InferenceStats::with_timing_high_precision(original_secs, 10, 20, None);

        // 验证毫秒转换正确（1234ms）
        assert_eq!(stats.inference_time_ms, 1234);

        // 验证秒数保持原始精度（1.234s）
        assert!((stats.inference_time_secs - original_secs).abs() < 0.001);
    }

    /// 测试：高精度统计中零生成token但有TTFT时的TPOT处理
    #[test]
    fn test_inference_stats_zero_generated_with_ttft() {
        // 当生成token数为0但提供TTFT时，TPOT应该是None（避免除以零）
        let stats = InferenceStats::with_timing_high_precision(
            1.0,
            10,
            0, // 零生成token
            Some(500),
        );

        assert_eq!(stats.generated_tokens, 0);
        assert!(stats.time_to_first_token_ms.is_some());
        assert!(
            stats.time_per_output_token_ms.is_none(),
            "零生成token时TPOT应为None，即使有TTFT"
        );
    }

    /// 测试：极端时间值的处理（极长时间运行）
    #[test]
    fn test_inference_stats_very_long_duration() {
        // 模拟长时间运行的推理（1小时）
        let hours_in_ms = 3600u64 * 1000; // 1小时=3,600,000毫秒

        let stats = InferenceStats::with_timing(
            hours_in_ms,
            10000, // 10K prompt tokens
            50000, // 50K generated tokens
        );

        assert_eq!(stats.total_tokens, 60000);
        assert!((stats.inference_time_secs - 3600.0).abs() < 1.0);

        // TPS应该在合理范围内 (60000 / 3600 ≈ 16.67)
        assert!(
            (stats.tokens_per_second - 16.67).abs() < 0.1,
            "长时间运行TPS计算错误: {}",
            stats.tokens_per_second
        );
    }

    /// 测试：tokens_per_second的各种边界条件
    #[test]
    fn test_inference_stats_tps_boundaries() {
        // 边界1: 只有prompt token，无生成token
        {
            let stats = InferenceStats::with_timing(1000, 100, 0);
            assert_eq!(stats.tokens_per_second, 100.0); // 100 / 1s
        }

        // 边界2: 单个token在极短时间内
        {
            let stats = InferenceStats::with_timing(1, 0, 1);
            assert!(
                stats.tokens_per_second > 999.0,
                "单tokenTPS应约等于1000: {}",
                stats.tokens_per_second
            );
        }

        // 边界3: 大量token在合理时间内
        {
            let stats = InferenceStats::with_timing(10000, 1000, 9000);
            let expected_tps = 10000.0 / 10.0; // 10000 tokens / 10s
            assert!((stats.tokens_per_second - expected_tps).abs() < 1.0);
        }
    }

    /// 测试：InferenceStats的Default和New方法一致性
    #[test]
    fn test_inference_stats_default_new_consistency() {
        let default_stats = InferenceStats::default();
        let new_stats = InferenceStats::new();

        // Default和New应该产生相同的结果
        assert_eq!(default_stats.inference_time_ms, new_stats.inference_time_ms);
        assert_eq!(
            default_stats.inference_time_secs,
            new_stats.inference_time_secs
        );
        assert_eq!(default_stats.total_tokens, new_stats.total_tokens);
        assert_eq!(default_stats.prompt_tokens, new_stats.prompt_tokens);
        assert_eq!(default_stats.generated_tokens, new_stats.generated_tokens);
        assert_eq!(default_stats.tokens_per_second, new_stats.tokens_per_second);
    }

    /// 测试：高精度统计中TTFT等于总时间时的情况
    #[test]
    fn test_inference_stats_ttft_equals_total_time() {
        // TTFT等于总推理时间（意味着所有时间都在首token上）
        let stats = InferenceStats::with_timing_high_precision(
            2.0,        // 2秒总时间
            10,         // 10 prompt tokens
            5,          // 5 generated tokens
            Some(2000), // TTFT = 2000ms = 总时间
        );

        // TPOT应该接近0或很小（解码时间几乎为0）
        if let Some(tpot) = stats.time_per_output_token_ms {
            assert!(
                tpot.abs() < 1.0 || tpot < 0.01,
                "TTFT=总时间时TPOT应接近0: {}",
                tpot
            );
        }
    }

    /// 测试：InferenceStats字段独立性（修改一个不影响其他）
    #[test]
    fn test_inference_stats_field_independence() {
        let stats1 = InferenceStats::with_timing(1000, 50, 50);
        let stats2 = InferenceStats::with_timing(2000, 100, 50); // 调整generated_tokens使TPS不同

        // 确保两个独立的实例不互相影响
        assert_ne!(stats1.inference_time_ms, stats2.inference_time_ms);
        assert_ne!(stats1.total_tokens, stats2.total_tokens);
        assert_ne!(stats1.tokens_per_second, stats2.tokens_per_second);
    }

    /// 测试：毫秒到秒的转换精度（大数值）
    #[test]
    fn test_inference_stats_large_millisecond_conversion() {
        // 测试大毫秒值转换精度
        let large_ms = u64::MAX / 2; // 接近u64最大值的一半
        let stats = InferenceStats::with_timing(large_ms, 1000, 9000);

        // 转换后的秒数应该非常大但不溢出
        assert!(stats.inference_time_secs > 0.0);
        assert!(stats.inference_time_secs.is_finite());

        // TPS应该仍然可计算
        assert!(stats.tokens_per_second > 0.0);
        assert!(stats.tokens_per_second.is_finite());
    }

    /// 测试：不同精度模式下的结果一致性比较
    #[test]
    fn test_inference_stats_precision_modes_comparison() {
        // 相同输入使用两种不同的创建方式
        let timing_ms = 1500u64;
        let prompt_tokens = 30usize;
        let generated_tokens = 70usize;

        // 方式1: with_timing（毫秒精度）
        let stats1 = InferenceStats::with_timing(timing_ms, prompt_tokens, generated_tokens);

        // 方式2: with_timing_high_precision（从ms转换回去）
        let timing_secs = timing_ms as f64 / 1000.0;
        let stats2 = InferenceStats::with_timing_high_precision(
            timing_secs,
            prompt_tokens,
            generated_tokens,
            None,
        );

        // 基本字段应该一致或非常接近
        assert_eq!(stats1.inference_time_ms, stats2.inference_time_ms);
        assert!((stats1.inference_time_secs - stats2.inference_time_secs).abs() < 0.001);
        assert_eq!(stats1.total_tokens, stats2.total_tokens);
        assert!((stats1.tokens_per_second - stats2.tokens_per_second).abs() < 0.01);
    }

    /// 测试：只有prompt token且时间为零的极端情况
    #[test]
    fn test_inference_stats_prompt_only_zero_time() {
        let stats = InferenceStats::with_timing(0, 5, 0);

        assert_eq!(stats.total_tokens, 5);
        assert_eq!(stats.generated_tokens, 0);
        assert_eq!(stats.prompt_tokens, 5);
        assert_eq!(stats.tokens_per_second, 0.0); // 除以零保护
        assert!(stats.time_to_first_token_ms.is_none());
        assert!(stats.time_per_output_token_ms.is_none());
    }
}
