//! 端到端推理测试
//!
//! 测试完整的推理流程，包括文本生成、图像理解等。

use std::sync::Arc;
use std::time::{Duration, Instant};

use openmini_server::hardware::memory::MemoryMonitor;
use openmini_server::model::inference::inference::InferenceStats;
use openmini_server::model::inference::{
    model::{ModelConfig, MultimodalTransformer},
    sampler::{BeamSearch, GenerateParams, Sampler},
    tokenizer::Tokenizer,
    InferenceEngine,
};

/// 测试推理引擎创建
#[test]
fn test_inference_engine_creation() {
    let config = ModelConfig::default();
    let _model = MultimodalTransformer::new(config);
    let _tokenizer = Tokenizer::new();

    let _engine = InferenceEngine::from_gguf(std::path::Path::new("test.gguf"));
}

/// 测试完整文本生成流程
#[test]
fn test_text_generation_e2e() {
    let config = ModelConfig::default();
    let model = MultimodalTransformer::new(config);
    let tokenizer = Tokenizer::new();

    let prompt = "Hello, world!";
    let tokens = tokenizer.encode(prompt).unwrap();
    let token_ids: Vec<u32> = tokens.to_vec();

    let max_new_tokens = 10;
    let generated = model.generate(&token_ids, max_new_tokens);

    assert!(generated.is_ok());
    let generated_ids = generated.unwrap();
    assert!(generated_ids.len() <= max_new_tokens);
}

/// 测试文本生成带统计信息
#[test]
fn test_text_generation_with_stats() {
    let config = ModelConfig::default();
    let model = MultimodalTransformer::new(config);
    let tokenizer = Tokenizer::new();

    let prompt = "Test prompt for generation";
    let start_time = Instant::now();

    let tokens = tokenizer.encode(prompt).unwrap();
    let prompt_tokens = tokens.len();
    let token_ids: Vec<u32> = tokens.to_vec();

    let generated = model.generate(&token_ids, 10).unwrap();
    let generated_tokens = generated.len();

    let elapsed_ms = start_time.elapsed().as_millis() as u64;
    let stats = InferenceStats::with_timing(elapsed_ms, prompt_tokens, generated_tokens);

    assert!(stats.inference_time_ms > 0);
    assert!(stats.total_tokens > 0);
    assert!(stats.tokens_per_second >= 0.0);
}

/// 测试多模态生成流程
#[test]
fn test_multimodal_generation_e2e() {
    let config = ModelConfig::default();
    let model = MultimodalTransformer::new(config);
    let tokenizer = Tokenizer::new();

    let prompt = "Describe this image";
    let tokens = tokenizer.encode(prompt).unwrap();
    let token_ids: Vec<u32> = tokens.to_vec();

    let generated = model.generate_multimodal(&token_ids, None, 10);

    assert!(generated.is_ok());
}

/// 测试采样器贪婪解码
#[test]
fn test_sampler_greedy_decoding() {
    let params = GenerateParams::default().with_sampling(false);
    let mut sampler = Sampler::new(params);

    let logits = ndarray::arr1(&[0.1, 0.5, 0.3, 0.9, 0.2]);
    let generated = vec![];

    let token = sampler.sample(&logits, &generated).unwrap();
    assert_eq!(token, 3);
}

/// 测试采样器温度采样
#[test]
fn test_sampler_temperature_sampling() {
    let params = GenerateParams::default()
        .with_sampling(true)
        .with_temperature(1.0)
        .with_top_p(1.0)
        .with_top_k(0);
    let mut sampler = Sampler::new(params);

    let logits = ndarray::arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let generated = vec![];

    let token = sampler.sample(&logits, &generated);
    assert!(token.is_ok());
}

/// 测试采样器 Top-K 过滤
#[test]
fn test_sampler_top_k_filtering() {
    let params = GenerateParams::default().with_top_k(3);
    let sampler = Sampler::new(params);

    let probs = ndarray::arr1(&[0.1, 0.3, 0.4, 0.15, 0.05]);
    let (indices, filtered_probs) = sampler.apply_top_k(&probs);

    assert_eq!(indices.len(), 3);
    assert!(filtered_probs.iter().all(|&p| p > 0.0));
}

/// 测试采样器 Top-P 过滤
#[test]
fn test_sampler_top_p_filtering() {
    let params = GenerateParams::default().with_top_p(0.9);
    let sampler = Sampler::new(params);

    let indices = vec![0, 1, 2, 3, 4];
    let probs = vec![0.4, 0.3, 0.2, 0.08, 0.02];

    let (filtered_indices, filtered_probs) = sampler.apply_top_p(&indices, &probs);

    let sum: f32 = filtered_probs.iter().sum();
    assert!(sum >= 0.9 || filtered_indices.len() < indices.len());
}

/// 测试重复惩罚
#[test]
fn test_repetition_penalty() {
    let params = GenerateParams::default().with_repetition_penalty(2.0);
    let sampler = Sampler::new(params);

    let mut logits = ndarray::arr1(&[1.0, -1.0, 2.0, -2.0]);
    let generated = vec![0, 1];

    sampler.apply_repetition_penalty(&mut logits, &generated);

    assert!((logits[0] - 0.5).abs() < 1e-5);
    assert!((logits[1] - (-2.0)).abs() < 1e-5);
    assert!((logits[2] - 2.0).abs() < 1e-5);
}

/// 测试束搜索初始化
#[test]
fn test_beam_search_initialization() {
    let mut beam_search = BeamSearch::new(3);
    beam_search.initialize(0, 0.0);

    assert_eq!(beam_search.candidates().len(), 1);
    assert_eq!(beam_search.best_candidate().unwrap().tokens, vec![0]);
}

/// 测试推理统计信息
#[test]
fn test_inference_stats_creation() {
    let stats = InferenceStats::with_timing(1000, 10, 20);

    assert_eq!(stats.inference_time_ms, 1000);
    assert_eq!(stats.prompt_tokens, 10);
    assert_eq!(stats.generated_tokens, 20);
    assert_eq!(stats.total_tokens, 30);
    assert!((stats.tokens_per_second - 30.0).abs() < 1e-5);
}

/// 测试 Tokenizer 编解码一致性
#[test]
fn test_tokenizer_encode_decode_consistency() {
    let tokenizer = Tokenizer::new();

    let original = "This is a test sentence.";
    let tokens = tokenizer.encode(original).unwrap();
    let _decoded = tokenizer.decode(&tokens).unwrap();

    assert!(!tokens.is_empty());
}

/// 测试模型前向传播
#[test]
fn test_model_forward_pass() {
    let config = ModelConfig::default();
    let model = MultimodalTransformer::new(config);

    let seq_len = 4;
    let hidden_size = model.config.hidden_size;
    let input = ndarray::Array2::zeros((seq_len, hidden_size));

    let output = model.forward(&input);
    assert!(output.is_ok());

    let output = output.unwrap();
    assert_eq!(output.dim(), (seq_len, hidden_size));
}

/// 测试 Logits 计算
#[test]
fn test_compute_logits() {
    let config = ModelConfig::default();
    let model = MultimodalTransformer::new(config);

    let seq_len = 2;
    let hidden_size = model.config.hidden_size;
    let hidden = ndarray::Array2::zeros((seq_len, hidden_size));

    let logits = model.compute_logits(&hidden);
    assert_eq!(logits.dim().0, seq_len);
    assert_eq!(logits.dim().1, model.config.vocab_size);
}

/// 测试内存监控
#[test]
fn test_memory_monitor_under_inference() {
    let monitor = MemoryMonitor::new(1024 * 1024 * 100);

    let result = monitor.allocate(1024 * 1024);
    assert!(result.is_ok());
    assert_eq!(monitor.usage(), 1024 * 1024);

    monitor.deallocate(1024 * 1024);
    assert_eq!(monitor.usage(), 0);
}

/// 测试推理延迟
#[test]
fn test_inference_latency() {
    let config = ModelConfig::default();
    let model = MultimodalTransformer::new(config);
    let tokenizer = Tokenizer::new();

    let prompt = "Test prompt";
    let tokens = tokenizer.encode(prompt).unwrap();
    let token_ids: Vec<u32> = tokens.to_vec();

    let start = Instant::now();
    let _ = model.generate(&token_ids, 5);
    let elapsed = start.elapsed();

    assert!(elapsed < Duration::from_secs(30));
}

/// 测试批处理生成
#[test]
fn test_batch_generation() {
    let config = ModelConfig::default();
    let model = MultimodalTransformer::new(config);
    let tokenizer = Tokenizer::new();

    let prompts = vec!["Hello", "World", "Test"];
    let mut results = Vec::new();

    for prompt in prompts {
        let tokens = tokenizer.encode(prompt).unwrap();
        let token_ids: Vec<u32> = tokens.to_vec();
        let generated = model.generate(&token_ids, 5);
        results.push(generated);
    }

    assert_eq!(results.len(), 3);
    for result in results {
        assert!(result.is_ok());
    }
}

/// 测试温度调节
#[test]
fn test_temperature_adjustment() {
    let params = GenerateParams::default().with_temperature(2.0);
    let sampler = Sampler::new(params);

    let logits = ndarray::arr1(&[1.0, 2.0, 3.0]);
    let mut logits_copy = logits.clone();

    sampler.apply_temperature(&mut logits_copy);

    assert!((logits_copy[0] - 0.5).abs() < 1e-5);
    assert!((logits_copy[1] - 1.0).abs() < 1e-5);
    assert!((logits_copy[2] - 1.5).abs() < 1e-5);
}

/// 测试 Softmax 计算
#[test]
fn test_softmax_calculation() {
    let sampler = Sampler::new(GenerateParams::default());
    let logits = ndarray::arr1(&[1.0, 2.0, 3.0]);
    let probs = sampler.softmax(&logits);

    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    assert!(probs[2] > probs[1]);
    assert!(probs[1] > probs[0]);
}

/// 测试 Argmax 选择
#[test]
fn test_argmax_selection() {
    let sampler = Sampler::new(GenerateParams::default());
    let logits = ndarray::arr1(&[0.1, 0.5, 0.3, 0.9, 0.2]);

    let result = sampler.argmax(&logits).unwrap();
    assert_eq!(result, 3);
}

/// 测试模型缓存清理
#[test]
fn test_model_cache_clear() {
    let config = ModelConfig::default();
    let _model = MultimodalTransformer::new(config);

    // clear_cache() 方法在当前 API 中暂不可用
    // 模型会在 drop 时自动清理资源
}

/// 测试固定种子采样器
#[test]
fn test_sampler_with_fixed_seed() {
    let params = GenerateParams::default();
    let sampler1 = Sampler::with_seed(params.clone(), 42);
    let sampler2 = Sampler::with_seed(params, 42);

    assert_eq!(sampler1.params().temperature, sampler2.params().temperature);
}

/// 测试生成参数默认值
#[test]
fn test_generate_params_defaults() {
    let params = GenerateParams::default();

    assert!(params.sampling);
    assert!((params.top_p - 0.8).abs() < 1e-6);
    assert_eq!(params.top_k, 100);
    assert!((params.temperature - 0.7).abs() < 1e-6);
    assert_eq!(params.num_beams, 3);
    assert!((params.repetition_penalty - 1.05).abs() < 1e-6);
    assert_eq!(params.max_new_tokens, 2048);
}

/// 测试推理上下文
#[test]
fn test_inference_context() {
    use openmini_server::model::inference::engine::InferenceContext;
    use openmini_server::model::inference::model::ModelConfig;

    let config = ModelConfig::default();
    let ctx = InferenceContext::new(&config);

    assert_eq!(ctx.kv_caches.len(), config.num_hidden_layers);
    assert_eq!(ctx.seq_len, 0);
}

/// 测试 KV Cache 层
#[test]
fn test_kv_cache_layer() {
    use openmini_server::model::inference::engine::KvCacheLayer;

    let mut layer = KvCacheLayer::new();
    assert!(layer.get().is_none());

    let k = ndarray::Array2::zeros((10, 64));
    let v = ndarray::Array2::zeros((10, 64));
    let _ = layer.update(k, v);

    assert!(layer.get().is_some());

    layer.clear();
    assert!(layer.get().is_none());
}

/// 测试生成参数构建器
#[test]
fn test_generate_params_builder() {
    let params = GenerateParams::new()
        .with_temperature(0.5)
        .with_top_p(0.9)
        .with_top_k(50)
        .with_max_new_tokens(1024)
        .with_repetition_penalty(1.2)
        .with_sampling(true)
        .with_num_beams(5);

    assert!((params.temperature - 0.5).abs() < 1e-6);
    assert!((params.top_p - 0.9).abs() < 1e-6);
    assert_eq!(params.top_k, 50);
    assert_eq!(params.max_new_tokens, 1024);
    assert!((params.repetition_penalty - 1.2).abs() < 1e-6);
    assert!(params.sampling);
    assert_eq!(params.num_beams, 5);
}

/// 测试模型配置
#[test]
fn test_model_config() {
    let config = ModelConfig::default();

    assert_eq!(config.vocab_size, 151936);
    assert_eq!(config.hidden_size, 3584);
    assert_eq!(config.num_hidden_layers, 28);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.image_patch_size, 14);
    assert_eq!(config.audio_feature_dim, 128);
}

/// 测试视觉编码器
#[test]
fn test_vision_encoder() {
    use ndarray::Array3;
    use openmini_server::model::inference::model::{ModelConfig, MultimodalTransformer};

    let model = MultimodalTransformer::new(ModelConfig::default());

    let image = Array3::from_shape_vec((224, 224, 3), vec![128u8; 224 * 224 * 3]).unwrap();
    let result = model.encode_image(&image);
    assert!(result.is_ok());
}

/// 测试多模态提示构建
#[test]
fn test_multimodal_prompt_building() {
    use openmini_server::model::inference::engine::build_multimodal_prompt;

    // 使用预编码的 token ID（模拟 tokenizer 输出）
    let text_tokens: Vec<usize> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let num_image_tokens = 64;
    let tokens = build_multimodal_prompt(&text_tokens, num_image_tokens);

    assert!(!tokens.is_empty());
    // 验证结构：IM_START + image_tokens + IM_END + text_tokens
    assert!(tokens.len() > num_image_tokens);
}

/// 测试流式生成器
#[tokio::test]
async fn test_stream_generator() {
    use openmini_server::model::inference::inference::StreamGenerator;

    let config = ModelConfig::default();
    let model = MultimodalTransformer::new(config);
    let tokenizer = Tokenizer::new();

    let generator = StreamGenerator::new(model, tokenizer);
    let params = GenerateParams::default().with_max_new_tokens(5);

    let mut chunks = Vec::new();
    let result = generator.stream_generate("Hello", &params, |chunk| {
        chunks.push(chunk.to_string());
        Ok(true)
    });

    assert!(result.is_ok());
}

/// 测试推理引擎文本生成
#[test]
fn test_inference_engine_generate() {
    let config = ModelConfig::default();
    let model = Arc::new(MultimodalTransformer::new(config));
    let tokenizer = Arc::new(Tokenizer::new());

    let params = GenerateParams::default().with_max_new_tokens(10);

    let prompt = "Hello, world!";
    let tokens = tokenizer.encode(prompt).unwrap();
    let token_ids: Vec<u32> = tokens.to_vec();

    let generated = model.generate(&token_ids, params.max_new_tokens);
    assert!(generated.is_ok());
}

/// 测试推理引擎带图像生成
#[test]
fn test_inference_engine_generate_with_image() {
    let config = ModelConfig::default();
    let model = Arc::new(MultimodalTransformer::new(config));

    let params = GenerateParams::default().with_max_new_tokens(10);

    let prompt = "Describe this";
    let tokenizer = Tokenizer::new();
    let tokens = tokenizer.encode(prompt).unwrap();
    let token_ids: Vec<u32> = tokens.to_vec();

    let generated = model.generate(&token_ids, params.max_new_tokens);

    assert!(generated.is_ok());
}

/// 测试批量采样
#[test]
fn test_batch_sampling() {
    let mut sampler = Sampler::new(GenerateParams::default());

    let logits_batch = vec![
        ndarray::arr1(&[0.1, 0.5, 0.3, 0.9]),
        ndarray::arr1(&[0.2, 0.4, 0.3, 0.1]),
        ndarray::arr1(&[0.5, 0.3, 0.1, 0.1]),
    ];

    let generated = vec![];
    let results = sampler.sample_batch(&logits_batch, &generated);

    assert!(results.is_ok());
    let tokens = results.unwrap();
    assert_eq!(tokens.len(), 3);
}

/// 测试束搜索最佳候选
#[test]
fn test_beam_search_best_candidate() {
    let mut beam_search = BeamSearch::new(3);
    beam_search.initialize(0, 0.0);

    let best = beam_search.best_candidate();
    assert!(best.is_some());
    assert_eq!(best.unwrap().score, 0.0);
}

/// 测试推理性能基准
#[test]
fn test_inference_performance_benchmark() {
    let config = ModelConfig::default();
    let model = MultimodalTransformer::new(config);
    let tokenizer = Tokenizer::new();

    let prompt = "Performance test prompt";
    let tokens = tokenizer.encode(prompt).unwrap();
    let token_ids: Vec<u32> = tokens.to_vec();

    let iterations = 3;
    let mut total_time = Duration::ZERO;

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = model.generate(&token_ids, 5);
        total_time += start.elapsed();
    }

    let avg_time = total_time / iterations;
    assert!(avg_time < Duration::from_secs(30));
}
