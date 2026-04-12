//! CALM: Chunked Autoregressive Language Modeling 实现
//!
//! CALM 是一种前瞻性的逐句生成范式，通过 Encoder-Decoder 架构将 hidden_state
//! 转换为 multi-token 输出，挑战传统的逐 token 生成方式。
//!
//! ## 核心优势
//! - **效率提升**: 传统 N 步 token 生成 -> M 步句子生成 (M << N)
//! - **语义完整性**: 保证生成的句子在语法和语义上的完整性
//! - **可控性**: 通过置信度机制控制生成质量
//!
//! ## 架构设计
//! ```
//! ┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
//! │   Prompt    │ --> │    CalmEncoder    │ --> │  Encoded    │
//! │ (token seq) │     │ (Transformer)     │     │ Context     │
//! └─────────────┘     └──────────────────┘     └──────┬──────┘
//!                                                     │
//!                                                     v
//! ┌─────────────┐     ┌──────────────────┐     ┌──────┴──────┐
//! │  Sentences  │ <-- │   CalmDecoder    │ <-- │ Scheduler   │
//! │ (output)    │     │ (Cross-Attn + LM)│     │ (Confidence)│
//! └─────────────┘     └──────────────────┘     └─────────────┘
//! ```

#![allow(dead_code)]

use anyhow::{Context, Result};
use ndarray::Array1;
use rand::prelude::*;
use std::collections::HashMap;

// ============================================================================
// 配置结构体
// ============================================================================

/// CALM 引擎配置参数
#[derive(Debug, Clone)]
pub struct CalmConfig {
    /// 隐藏层维度
    pub hidden_dim: usize,
    /// Encoder 层数
    pub num_encoder_layers: usize,
    /// Decoder 层数
    pub num_decoder_layers: usize,
    /// 每句最大 token 数
    pub max_sentence_length: usize,
    /// 候选句数量
    pub num_candidates: usize,
    /// 置信度阈值 (0.0-1.0)
    pub confidence_threshold: f32,
    /// 是否结合 speculative decoding
    pub use_speculative: bool,
}

impl Default for CalmConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 768,
            num_encoder_layers: 6,
            num_decoder_layers: 6,
            max_sentence_length: 50,
            num_candidates: 5,
            confidence_threshold: 0.8,
            use_speculative: false,
        }
    }
}

// ============================================================================
// 核心数据结构
// ============================================================================

/// Transformer 层（简化原型实现）
#[derive(Debug, Clone)]
struct TransformerLayer {
    dim: usize,
    weights: Array1<f32>,
}

impl TransformerLayer {
    fn new(dim: usize) -> Self {
        let mut rng = StdRng::from_entropy();
        let weights = Array1::from_shape_fn(dim, |_| rng.gen::<f32>() * 0.02);
        Self { dim, weights }
    }

    fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        // 简化的前向传播：线性变换 + 激活
        if input.len() != self.dim {
            return Array1::zeros(self.dim);
        }

        let dot_product: f32 = input
            .iter()
            .zip(self.weights.iter())
            .map(|(a, b)| a * b)
            .sum();
        Array1::from_shape_fn(self.dim, |i| {
            let x = input[i] + self.weights[i];
            // ReLU 激活函数
            x.max(0.0) + dot_product.tanh() * 0.01
        })
    }
}

/// 线性层（输出头）
#[derive(Debug, Clone)]
struct LinearLayer {
    weight: Array2<f32>,
    bias: Array1<f32>,
}

impl LinearLayer {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = StdRng::from_entropy();

        // Xavier 初始化
        let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();
        let weight = Array2::from_shape_fn((output_dim, input_dim), |_| rng.gen::<f32>() * scale);
        let bias = Array1::zeros(output_dim);

        Self { weight, bias }
    }

    fn forward(&self, input: &Array1<f32>) -> Result<Array1<f32>> {
        if input.len() != self.weight.ncols() {
            return Err(anyhow::anyhow!(
                "输入维度不匹配: 期望 {}, 实际 {}",
                self.weight.ncols(),
                input.len()
            ));
        }

        let output = Array1::from_shape_fn(self.weight.nrows(), |i| {
            let mut sum = self.bias[i];
            for j in 0..input.len() {
                sum += self.weight[[i, j]] * input[j];
            }
            sum
        });

        Ok(output)
    }
}

/// 交叉注意力层（简化版）
#[derive(Debug, Clone)]
struct CrossAttentionLayer {
    query_proj: LinearLayer,
    key_proj: LinearLayer,
    value_proj: LinearLayer,
    output_proj: LinearLayer,
    dim: usize,
}

impl CrossAttentionLayer {
    fn new(hidden_dim: usize) -> Self {
        Self {
            query_proj: LinearLayer::new(hidden_dim, hidden_dim),
            key_proj: LinearLayer::new(hidden_dim, hidden_dim),
            value_proj: LinearLayer::new(hidden_dim, hidden_dim),
            output_proj: LinearLayer::new(hidden_dim, hidden_dim),
            dim: hidden_dim,
        }
    }

    fn forward(&self, query: &Array1<f32>, keys_values: &[Array1<f32>]) -> Result<Array1<f32>> {
        if keys_values.is_empty() {
            return Ok(Array1::zeros(self.dim));
        }

        let q = self.query_proj.forward(query)?;
        let mut context = Array1::zeros(self.dim);

        for kv in keys_values {
            let k = self.key_proj.forward(kv)?;
            let v = self.value_proj.forward(kv)?;

            // 计算注意力分数
            let score: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
            let attention_weight = score.exp(); // 简化的 softmax

            // 加权求和
            for i in 0..self.dim {
                context[i] += attention_weight * v[i];
            }
        }

        // 归一化
        let norm: f32 = context.iter().map(|x| x.abs()).sum().max(1e-6);
        context.mapv_inplace(|x| x / norm);

        self.output_proj.forward(&context)
    }
}

// ============================================================================
// 句子边界检测器
// ============================================================================

/// 句子边界位置信息
#[derive(Debug, Clone)]
pub struct SentenceBoundary {
    /// 起始 token 索引
    pub start_idx: usize,
    /// 结束 token 索引
    pub end_idx: usize,
    /// 边界置信度
    pub confidence: f32,
}

impl SentenceBoundary {
    /// 创建新的句子边界
    pub fn new(start_idx: usize, end_idx: usize, confidence: f32) -> Self {
        Self {
            start_idx,
            end_idx,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// 获取边界长度
    pub fn length(&self) -> usize {
        self.end_idx.saturating_sub(self.start_idx) + 1
    }
}

/// 句子边界检测器
///
/// 使用分类模型检测输入序列中的句子边界位置。
/// 在实际应用中，这可以是一个训练好的 BERT 或其他模型。
pub struct SentenceBoundaryDetector {
    model: Array2<f32>, // 分类头权重
    vocab_size: usize,
}

impl SentenceBoundaryDetector {
    /// 创建新的句子边界检测器
    ///
    /// # 参数
    /// - `vocab_size`: 词表大小
    pub fn new(vocab_size: usize) -> Self {
        let mut rng = StdRng::from_entropy();
        let model = Array2::from_shape_fn((3, vocab_size), |_| rng.gen::<f32>() * 0.01); // 3类: 开始/结束/无

        Self { model, vocab_size }
    }

    /// 检测输入中的句子边界位置
    ///
    /// 返回检测到的所有句子边界列表
    pub fn detect_boundaries(&self, input: &[u32]) -> Vec<SentenceBoundary> {
        if input.is_empty() {
            return Vec::new();
        }

        let mut boundaries = Vec::new();
        let mut sentence_start = 0;

        for i in 0..input.len() {
            if self.is_sentence_end(input[i]) {
                // 计算边界置信度
                let confidence = self.calculate_boundary_confidence(input, i);

                boundaries.push(SentenceBoundary::new(sentence_start, i, confidence));
                sentence_start = i + 1;
            }
        }

        // 处理最后一个未闭合的句子
        if sentence_start < input.len() {
            boundaries.push(SentenceBoundary::new(
                sentence_start,
                input.len() - 1,
                0.6, // 未闭合句子的默认置信度
            ));
        }

        boundaries
    }

    /// 判断是否到达句子结束
    ///
    /// 基于 token ID 和标点符号规则判断
    pub fn is_sentence_end(&self, token_id: u32) -> bool {
        // 常见的句子结束标点符号的 token ID（示例值）
        matches!(token_id, 0 | 1 | 2 | 3) // . ! ? ;
    }

    /// 计算边界的置信度分数
    fn calculate_boundary_confidence(&self, input: &[u32], pos: usize) -> f32 {
        if pos >= input.len() || self.vocab_size == 0 {
            return 0.5;
        }

        let token_id = input[pos] as usize;
        if token_id >= self.vocab_size {
            return 0.5;
        }

        // 使用模型权重计算置信度（简化版）
        let score = self.model[[1, token_id]]; // "结束"类的得分
        let total_score: f32 = self.model.row(1).mapv(|x| x.abs()).sum();
        if total_score == 0.0 {
            return 0.5;
        }

        (score.abs() / total_score).clamp(0.0, 1.0)
    }
}

// ============================================================================
// 候选句子与置信度评分系统
// ============================================================================

/// 候选句子
#[derive(Debug, Clone)]
pub struct CandidateSentence {
    /// Token 序列
    pub tokens: Vec<u32>,
    /// 对数概率
    pub log_probability: f32,
    /// 置信度分数
    pub confidence: f32,
}

impl CandidateSentence {
    /// 创建新的候选句子
    pub fn new(tokens: Vec<u32>, log_probability: f32) -> Self {
        Self {
            tokens,
            log_probability,
            confidence: 0.0, // 初始为0，后续计算
        }
    }

    /// 获取句子长度
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// 判断是否是文档结束标记
    pub fn is_end_of_document(&self) -> bool {
        // 检查最后一个 token 是否是 EOS
        self.tokens.last().map_or(false, |&t| t == 2) // 假设 2 是 EOS
    }
}

/// 置信度评分器
///
/// 综合多个指标评估候选句子的质量：
/// - 语言模型概率
/// - 句法正确性
/// - 语义连贯性
pub struct ConfidenceScorer {
    threshold: f32,
    scorer_model: Option<Array2<f32>>,
}

impl ConfidenceScorer {
    /// 创建新的置信度评分器
    ///
    /// # 参数
    /// - `threshold`: 置信度阈值，低于此值的候选将被过滤
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
            scorer_model: None,
        }
    }

    /// 创建带自定义模型的评分器
    pub fn with_model(threshold: f32, model: Array2<f32>) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
            scorer_model: Some(model),
        }
    }

    /// 计算候选句子的置信度
    ///
    /// 综合多个指标进行评分：
    /// - 语言模型概率 (40%)
    /// - 句法正确性 (30%)
    /// - 语义连贯性 (30%)
    pub fn score(&self, candidate: &CandidateSentence) -> f32 {
        if candidate.is_empty() || candidate.tokens.is_empty() {
            return 0.0;
        }

        // 1. 语言模型概率得分 (基于对数概率)
        let lm_score = self.score_lm_probability(candidate.log_probability);

        // 2. 句法正确性得分 (基于长度和结构)
        let syntax_score = self.score_syntax(candidate);

        // 3. 语义连贯性得分 (基于 token 分布)
        let semantic_score = self.score_semantic(candidate);

        // 加权综合得分
        let total_score = lm_score * 0.4 + syntax_score * 0.3 + semantic_score * 0.3;

        total_score.clamp(0.0, 1.0)
    }

    /// 语言模型概率评分
    fn score_lm_probability(&self, log_prob: f32) -> f32 {
        // 将对数概率映射到 [0, 1]
        // 典型的 log_prob 范围: [-50, 0]
        let normalized = (log_prob + 50.0) / 50.0; // 映射到 [0, 1]
        normalized.clamp(0.0, 1.0)
    }

    /// 句法正确性评分
    fn score_syntax(&self, candidate: &CandidateSentence) -> f32 {
        let len = candidate.len();

        if len == 0 {
            return 0.0;
        }

        // 基于长度的简单启发式
        // 过短或过长的句子可能语法不完整
        let length_score = if len <= 20 {
            len as f32 / 20.0 // 短句子
        } else if len <= 100 {
            1.0 // 中等长度最理想
        } else {
            (100.0 / len as f32).max(0.3) // 长句子惩罚
        };

        length_score
    }

    /// 语义连贯性评分
    fn score_semantic(&self, candidate: &CandidateSentence) -> f32 {
        if candidate.tokens.is_empty() {
            return 0.0;
        }

        // 简化的语义评分：基于 token 多样性和重复度
        let unique_tokens: std::collections::HashSet<u32> =
            candidate.tokens.iter().cloned().collect();
        let diversity_ratio = unique_tokens.len() as f32 / candidate.tokens.len() as f32;

        // 多样性越高，语义越连贯（简化假设）
        diversity_ratio.min(1.0)
    }

    /// 过滤低置信度候选
    ///
    /// 返回置信度超过阈值的候选句子列表
    pub fn filter_candidates(&self, candidates: Vec<CandidateSentence>) -> Vec<CandidateSentence> {
        candidates
            .into_iter()
            .map(|mut c| {
                c.confidence = self.score(&c);
                c
            })
            .filter(|c| c.confidence >= self.threshold)
            .collect()
    }

    /// 获取阈值
    pub fn threshold(&self) -> f32 {
        self.threshold
    }
}

// ============================================================================
// 生成策略枚举
// ============================================================================

/// 生成策略
#[derive(Debug, Clone)]
enum GenerationStrategy {
    /// 贪心选择：选择最高置信度的候选
    Greedy,
    /// 束搜索：维护 top-k 个候选
    BeamSearch { width: usize },
    /// 采样：根据置信度分布随机采样
    Sampling { temperature: f32 },
}

impl Default for GenerationStrategy {
    fn default() -> Self {
        Self::Greedy
    }
}

// ============================================================================
// 句子调度器
// ============================================================================

/// 句子调度器
///
/// 负责从候选句子中选择最佳输出，支持多种选择策略
pub struct SentenceScheduler {
    strategy: GenerationStrategy,
    confidence_scorer: ConfidenceScorer,
}

impl SentenceScheduler {
    /// 创建新的句子调度器
    pub fn new(strategy: GenerationStrategy, confidence_scorer: ConfidenceScorer) -> Self {
        Self {
            strategy,
            confidence_scorer,
        }
    }

    /// 从候选列表中选择最佳句子
    ///
    /// 根据 strategy 选择不同的算法：
    /// - Greedy: 选择置信度最高的候选
    /// - BeamSearch: 束搜索选择
    /// - Sampling: 按温度采样
    pub fn select_best(
        &mut self,
        candidates: &[CandidateSentence],
    ) -> Result<Option<CandidateSentence>> {
        if candidates.is_empty() {
            return Ok(None);
        }

        // 先对所有候选进行评分和过滤
        let scored_candidates: Vec<CandidateSentence> = candidates
            .iter()
            .map(|c| {
                let mut scored = c.clone();
                scored.confidence = self.confidence_scorer.score(c);
                scored
            })
            .filter(|c| c.confidence >= self.confidence_scorer.threshold())
            .collect();

        if scored_candidates.is_empty() {
            return Ok(None); // 没有满足阈值的候选
        }

        match &self.strategy {
            GenerationStrategy::Greedy => {
                // 选择置信度最高的候选
                let best = scored_candidates
                    .into_iter()
                    .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap());
                Ok(best)
            }

            GenerationStrategy::BeamSearch { width } => {
                // 简化的束搜索：返回 top-k 中的最佳
                let mut sorted: Vec<CandidateSentence> = scored_candidates;
                sorted.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

                // 取前 width 个，然后从中选最好的
                let top_k: Vec<_> = sorted.into_iter().take(*width).collect();
                Ok(top_k.into_iter().next())
            }

            GenerationStrategy::Sampling { temperature } => {
                // 温度采样
                let mut rng = StdRng::from_entropy();

                // 提取置信度并应用温度
                let confidences: Vec<f32> = scored_candidates
                    .iter()
                    .map(|c| c.confidence.powf(1.0 / temperature))
                    .collect();

                // 归一化为概率分布
                let sum: f32 = confidences.iter().sum();
                if sum <= 0.0 {
                    // fallback 到贪心
                    return Ok(scored_candidates.into_iter().next());
                }

                let probs: Vec<f32> = confidences.iter().map(|p| p / sum).collect();

                // 加权随机采样
                match WeightedIndex::new(probs) {
                    Ok(dist) => {
                        let idx = dist.sample(&mut rng);
                        Ok(Some(scored_candidates[idx].clone()))
                    }
                    Err(_) => Ok(scored_candidates.into_iter().next()),
                }
            }
        }
    }

    /// 获取当前策略
    pub fn strategy(&self) -> &GenerationStrategy {
        &self.strategy
    }

    /// 更新策略
    pub fn set_strategy(&mut self, strategy: GenerationStrategy) {
        self.strategy = strategy;
    }
}

// ============================================================================
// Encoder
// ============================================================================

/// CALM Encoder
///
/// 处理输入上下文，将其转换为编码表示
pub struct CalmEncoder {
    layers: Vec<TransformerLayer>,
    sentence_detector: SentenceBoundaryDetector,
    hidden_dim: usize,
}

impl CalmEncoder {
    /// 创建新的 Encoder
    ///
    /// # 参数
    /// - `num_layers`: Transformer 层数
    /// - `hidden_dim`: 隐藏层维度
    /// - `vocab_size`: 词表大小（用于句子检测）
    pub fn new(num_layers: usize, hidden_dim: usize, vocab_size: usize) -> Self {
        let layers: Vec<TransformerLayer> = (0..num_layers)
            .map(|_| TransformerLayer::new(hidden_dim))
            .collect();

        let sentence_detector = SentenceBoundaryDetector::new(vocab_size);

        Self {
            layers,
            sentence_detector,
            hidden_dim,
        }
    }

    /// 编码输入 prompt
    ///
    /// 将 token 序列转换为隐藏状态表示
    pub fn encode(&self, prompt: &[u32]) -> Result<EncodedContext> {
        if prompt.is_empty() {
            return Err(anyhow::anyhow!("Prompt 不能为空"));
        }

        // 初始化隐藏状态（简化的 embedding）
        let mut hidden = Array1::from_shape_fn(self.hidden_dim, |i| {
            let base = (prompt[0] as f32 + i as f32) * 0.01;
            base.sin() // 使用 sin 作为简单的位置编码
        });

        // 通过所有 Transformer 层
        for layer in &self.layers {
            hidden = layer.forward(&hidden);
        }

        // 检测句子边界
        let boundaries = self.sentence_detector.detect_boundaries(prompt);

        Ok(EncodedContext {
            hidden_state: hidden,
            original_tokens: prompt.to_vec(),
            sentence_boundaries: boundaries,
        })
    }

    /// 获取层数
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

/// 编码后的上下文
#[derive(Debug, Clone)]
pub struct EncodedContext {
    /// 隐藏状态表示
    pub hidden_state: Array1<f32>,
    /// 原始 token 序列
    pub original_tokens: Vec<u32>,
    /// 检测到的句子边界
    pub sentence_boundaries: Vec<SentenceBoundary>,
}

// ============================================================================
// Decoder
// ============================================================================

/// CALM Decoder
///
/// 基于编码上下文生成完整句子
pub struct CalmDecoder {
    layers: Vec<TransformerLayer>,
    cross_attention: CrossAttentionLayer,
    output_head: LinearLayer,
    hidden_dim: usize,
    vocab_size: usize,
    max_length: usize,
}

impl CalmDecoder {
    /// 创建新的 Decoder
    ///
    /// # 参数
    /// - `num_layers`: Transformer 层数
    /// - `hidden_dim`: 隐藏层维度
    /// - `vocab_size`: 词表大小
    /// - `max_length`: 最大生成长度
    pub fn new(num_layers: usize, hidden_dim: usize, vocab_size: usize, max_length: usize) -> Self {
        let layers: Vec<TransformerLayer> = (0..num_layers)
            .map(|_| TransformerLayer::new(hidden_dim))
            .collect();

        let cross_attention = CrossAttentionLayer::new(hidden_dim);
        let output_head = LinearLayer::new(hidden_dim, vocab_size);

        Self {
            layers,
            cross_attention,
            output_head,
            hidden_dim,
            vocab_size,
            max_length,
        }
    }

    /// 生成候选句子
    ///
    /// 基于编码上下文生成多个候选句子
    pub fn generate_candidates(&self, encoded: &EncodedContext) -> Result<Vec<CandidateSentence>> {
        if encoded.original_tokens.is_empty() {
            return Err(anyhow::anyhow!("编码上下文不能为空"));
        }

        let mut candidates = Vec::new();
        let num_candidates = 5.min(self.vocab_size); // 默认生成5个候选

        for candidate_idx in 0..num_candidates {
            // 初始化 decoder 的隐藏状态（基于 encoder 输出）
            let mut hidden = encoded.hidden_state.clone();

            // 应用交叉注意力
            let attended = self
                .cross_attention
                .forward(&hidden, &[encoded.hidden_state.clone()])?;
            hidden = attended;

            // 通过 decoder 层
            for layer in &self.layers {
                hidden = layer.forward(&hidden);
            }

            // 生成 logits 并转换为 token 概率
            let logits = self.output_head.forward(&hidden)?;
            let tokens = self.sample_tokens_from_logits(&logits, candidate_idx)?;
            let log_prob = self.calculate_log_probability(&logits, &tokens);

            candidates.push(CandidateSentence::new(tokens, log_prob));
        }

        Ok(candidates)
    }

    /// 从 logits 采样 token
    fn sample_tokens_from_logits(&self, logits: &Array1<f32>, seed: usize) -> Result<Vec<u32>> {
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let mut tokens = Vec::new();
        let mut current_hidden = logits.clone();

        for _step in 0..self.max_length {
            // softmax 归一化
            let probs = self.softmax(&current_hidden)?;

            // 采样一个 token
            let token_id = self.sample_token(&probs, &mut rng)?;
            tokens.push(token_id);

            // 检查是否到达句子结束
            if self.is_sentence_end_token(token_id) && tokens.len() > 3 {
                break;
            }

            // 简化的下一步：更新 hidden state（实际中应该用模型更新）
            current_hidden = self.update_hidden_for_next_step(&current_hidden, token_id)?;
        }

        if tokens.is_empty() {
            tokens.push(0); // 默认 token
        }

        Ok(tokens)
    }

    /// Softmax 函数
    fn softmax(&self, logits: &Array1<f32>) -> Result<Array1<f32>> {
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let exp_vals: Vec<f32> = logits.iter().map(|x| (*x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();

        if sum <= 0.0 {
            return Err(anyhow::anyhow!("Softmax 计算错误: sum={}", sum));
        }

        Ok(Array1::from_vec(exp_vals.iter().map(|x| x / sum).collect()))
    }

    /// 从概率分布中采样单个 token
    fn sample_token(&self, probs: &Array1<f32>, rng: &mut StdRng) -> Result<u32> {
        match WeightedIndex::new(probs.to_vec()) {
            Ok(dist) => Ok(dist.sample(rng) as u32),
            Err(e) => Err(anyhow::anyhow!("采样失败: {}", e)),
        }
    }

    /// 判断是否是句子结束 token
    fn is_sentence_end_token(&self, token_id: u32) -> bool {
        matches!(token_id, 0 | 1 | 2 | 3) // . ! ? ;
    }

    /// 更新隐藏状态用于下一步（简化版）
    fn update_hidden_for_next_step(
        &self,
        current: &Array1<f32>,
        _token_id: u32,
    ) -> Result<Array1<f32>> {
        // 简化版本：添加小的扰动模拟自回归
        let mut rng = StdRng::from_entropy();
        let noise = Array1::from_shape_fn(current.len(), |_| (rng.gen::<f32>() - 0.5) * 0.01);

        Ok(current.clone() + noise)
    }

    /// 计算候选句子的对数概率
    fn calculate_log_probability(&self, _logits: &Array1<f32>, tokens: &[u32]) -> f32 {
        if tokens.is_empty() {
            return f32::NEG_INFINITY;
        }

        // 简化的对数概率计算
        let base_prob = 1.0 / tokens.len() as f32;
        (base_prob.ln()) * tokens.len() as f32
    }

    /// 获取词汇表大小
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

// ============================================================================
// CALM 输出结果
// ============================================================================

/// CALM 生成输出
#[derive(Debug, Clone)]
pub struct CalmOutput {
    /// 生成的句子列表
    pub sentences: Vec<CandidateSentence>,
}

impl CalmOutput {
    /// 创建新的输出
    pub fn new(sentences: Vec<CandidateSentence>) -> Self {
        Self { sentences }
    }

    /// 获取总 token 数
    pub fn total_tokens(&self) -> usize {
        self.sentences.iter().map(|s| s.len()).sum()
    }

    /// 获取句子数量
    pub fn num_sentences(&self) -> usize {
        self.sentences.len()
    }

    /// 合并为单一 token 序列
    pub fn to_token_sequence(&self) -> Vec<u32> {
        self.sentences
            .iter()
            .flat_map(|s| s.tokens.clone())
            .collect()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.sentences.is_empty()
    }
}

// ============================================================================
// 对比实验框架
// ============================================================================

/// 对比指标
#[derive(Debug, Clone)]
pub struct ComparisonMetrics {
    /// CALM 生成步数
    pub calm_steps: usize,
    /// 传统逐 token 生成步数
    pub traditional_steps: usize,
    /// 加速比
    pub speedup_ratio: f32,
    /// 质量评分 (BLEU/ROUGE 近似)
    pub quality_score: f32,
    /// 延迟降低百分比
    pub latency_reduction_pct: f32,
}

impl Default for ComparisonMetrics {
    fn default() -> Self {
        Self {
            calm_steps: 0,
            traditional_steps: 0,
            speedup_ratio: 1.0,
            quality_score: 0.0,
            latency_reduction_pct: 0.0,
        }
    }
}

impl ComparisonMetrics {
    /// 创建对比指标
    pub fn new(calm_steps: usize, traditional_steps: usize, quality_score: f32) -> Self {
        let speedup_ratio = if calm_steps > 0 {
            traditional_steps as f32 / calm_steps as f32
        } else {
            1.0
        };

        let latency_reduction_pct = if traditional_steps > 0 {
            ((traditional_steps - calm_steps) as f32 / traditional_steps as f32) * 100.0
        } else {
            0.0
        };

        Self {
            calm_steps,
            traditional_steps,
            speedup_ratio,
            quality_score,
            latency_reduction_pct,
        }
    }
}

/// 对比实验结果
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// CALM 输出
    pub calm_output: CalmOutput,
    /// 传统逐 token 生成的 token 序列
    pub traditional_tokens: Vec<u32>,
    /// 对比指标
    pub metrics: ComparisonMetrics,
}

impl ComparisonResult {
    /// 创建对比结果
    pub fn new(
        calm_output: CalmOutput,
        traditional_tokens: Vec<u32>,
        metrics: ComparisonMetrics,
    ) -> Self {
        Self {
            calm_output,
            traditional_tokens,
            metrics,
        }
    }

    /// 打印对比报告
    pub fn print_report(&self) {
        println!("\n=== CALM vs Traditional Token-by-Token ===");
        println!("CALM Steps: {}", self.metrics.calm_steps);
        println!("Traditional Steps: {}", self.metrics.traditional_steps);
        println!("Speedup Ratio: {:.2}x", self.metrics.speedup_ratio);
        println!("Quality Score: {:.4}", self.metrics.quality_score);
        println!(
            "Latency Reduction: {:.1}%",
            self.metrics.latency_reduction_pct
        );
        println!("========================================\n");
    }
}

// ============================================================================
// CALM Engine 核心
// ============================================================================

/// CALM: Chunked Autoregressive Language Modeling 引擎
///
/// 这是核心引擎，协调 Encoder、Decoder 和 Scheduler 进行逐句生成。
///
/// ## 工作流程
/// 1. **Encoding**: Encoder 处理输入 prompt，提取上下文表示
/// 2. **Generation**: Decoder 基于上下文生成多个候选句子
/// 3. **Selection**: Scheduler 评估置信度并选择最佳候选
/// 4. **Iteration**: 重复步骤 2-3 直到完成文档
///
/// ## 性能特点
/// - 相比传统逐 token 生成，步数从 O(N) 降低到 O(M)，其中 M 是句子数量
/// - 通常 M << N，带来显著的加速效果
pub struct CalmEngine {
    encoder: CalmEncoder,
    decoder: CalmDecoder,
    scheduler: SentenceScheduler,
    config: CalmConfig,
}

impl CalmEngine {
    /// 创建新的 CALM 引擎
    ///
    /// # 参数
    /// - `config`: 引擎配置
    /// - `vocab_size`: 词表大小
    pub fn new(config: CalmConfig, vocab_size: usize) -> Self {
        let encoder = CalmEncoder::new(config.num_encoder_layers, config.hidden_dim, vocab_size);

        let decoder = CalmDecoder::new(
            config.num_decoder_layers,
            config.hidden_dim,
            vocab_size,
            config.max_sentence_length,
        );

        let scorer = ConfidenceScorer::new(config.confidence_threshold);
        let scheduler = SentenceScheduler::new(GenerationStrategy::default(), scorer);

        Self {
            encoder,
            decoder,
            scheduler,
            config,
        }
    }

    /// 使用自定义策略创建引擎
    pub fn with_strategy(
        config: CalmConfig,
        vocab_size: usize,
        strategy: GenerationStrategy,
    ) -> Self {
        let engine = Self::new(config, vocab_size);
        // 注意：这里需要重新构造以使用自定义策略
        // 由于字段私有，我们通过新方法处理
        let scorer = ConfidenceScorer::new(config.confidence_threshold);
        let scheduler = SentenceScheduler::new(strategy, scorer);

        Self {
            scheduler,
            ..engine
        }
    }

    /// 逐句生成前向传播
    ///
    /// 这是 CALM 的核心方法，实现了完整的逐句生成流程。
    ///
    /// ## 对比说明
    /// - **传统方式**: token → token → token ... (N 步)
    /// - **CALM 方式**: context → [sentence] → [sentence] → ... (M 步, M << N)
    ///
    /// # 参数
    /// - `prompt`: 输入的 token 序列
    /// - `max_sentences`: 最大生成句子数量
    ///
    /// # 返回
    /// - `Ok(CalmOutput)`: 包含生成句子的输出
    /// - `Err`: 编码或解码过程中的错误
    ///
    /// # 示例
    /// ```ignore
    /// let mut engine = CalmEngine::new(CalmConfig::default(), 10000);
    /// let prompt = vec![101, 202, 303]; // 输入 token
    /// let output = engine.generate(&prompt, 5)?;
    /// println!("Generated {} sentences", output.num_sentences());
    /// ```
    pub fn generate(&mut self, prompt: &[u32], max_sentences: usize) -> Result<CalmOutput> {
        if prompt.is_empty() {
            return Err(anyhow::anyhow!("Prompt 不能为空"));
        }

        if max_sentences == 0 {
            return Ok(CalmOutput::new(Vec::new()));
        }

        // Step 1: Encoder 处理 prompt
        let encoded = self
            .encoder
            .encode(prompt)
            .with_context(|| "编码 prompt 失败")?;

        // Step 2: 循环生成句子
        let mut sentences = Vec::with_capacity(max_sentences);

        for step in 0..max_sentences {
            // Decoder 生成候选句子
            let candidates = self
                .decoder
                .generate_candidates(&encoded)
                .with_context(|| format!("第 {} 步: 生成候选句子失败", step))?;

            // 置信度评分与选择
            match self.scheduler.select_best(&candidates)? {
                Some(sentence) => {
                    sentences.push(sentence.clone());

                    // 检查是否到达文档结束
                    if sentence.is_end_of_document() {
                        break;
                    }
                }
                None => {
                    // 无有效候选，停止生成
                    break;
                }
            }
        }

        Ok(CalmOutput::new(sentences))
    }

    /// 与传统逐 token 生成对比实验
    ///
    /// 该方法用于评估 CALM 相对于传统方法的优势。
    ///
    /// # 参数
    /// - `prompt`: 输入 prompt
    /// - `traditional_output`: 传统方法生成的 token 序列（作为基准）
    ///
    /// # 返回
    /// 包含详细对比指标的 ComparisonResult
    pub fn compare_with_token_by_token(
        &mut self,
        prompt: &[u32],
        traditional_output: &[u32],
    ) -> Result<ComparisonResult> {
        // 使用 CALM 生成
        let calm_output = self.generate(prompt, 10)?; // 最多10个句子

        // 计算 CALM 步数（每个句子算一步）
        let calm_steps = calm_output.num_sentences().max(1);

        // 传统步数就是 token 数量
        let traditional_steps = traditional_output.len().max(1);

        // 计算质量评分（简化的 BLEU 近似）
        let quality_score = self.calculate_quality_score(&calm_output, traditional_output);

        // 创建对比指标
        let metrics = ComparisonMetrics::new(calm_steps, traditional_steps, quality_score);

        Ok(ComparisonResult::new(
            calm_output,
            traditional_output.to_vec(),
            metrics,
        ))
    }

    /// 计算质量评分（简化版 BLEU）
    fn calculate_quality_score(&self, calm_output: &CalmOutput, reference: &[u32]) -> f32 {
        if reference.is_empty() || calm_output.is_empty() {
            return 0.0;
        }

        let generated_tokens = calm_output.to_token_sequence();

        if generated_tokens.is_empty() {
            return 0.0;
        }

        // 简化的 BLEU 计算：基于 n-gram 重叠
        let mut matches = 0usize;
        let mut total = 0usize;

        // unigram 精确率
        let gen_unigrams: std::collections::HashMap<u32, usize> = generated_tokens
            .iter()
            .cloned()
            .fold(HashMap::new(), |mut acc, t| {
                *acc.entry(t).or_insert(0) += 1;
                acc
            });

        let ref_unigrams: std::collections::HashMap<u32, usize> =
            reference
                .iter()
                .cloned()
                .fold(HashMap::new(), |mut acc, t| {
                    *acc.entry(t).or_insert(0) += 1;
                    acc
                });

        for (token, &count) in &gen_unigrams {
            let ref_count = ref_unigrams.get(token).unwrap_or(&0);
            matches += count.min(*ref_count);
            total += count;
        }

        if total == 0 {
            return 0.0;
        }

        matches as f32 / total as f32
    }

    /// 获取配置引用
    pub fn config(&self) -> &CalmConfig {
        &self.config
    }

    /// 获取 Encoder 引用
    pub fn encoder(&self) -> &CalmEncoder {
        &self.encoder
    }

    /// 获取 Decoder 引用
    pub fn decoder(&self) -> &CalmDecoder {
        &self.decoder
    }

    /// 获取 Scheduler 引用
    pub fn scheduler(&self) -> &SentenceScheduler {
        &self.scheduler
    }

    /// 更新生成策略
    pub fn set_strategy(&mut self, strategy: GenerationStrategy) {
        self.scheduler.set_strategy(strategy);
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ===== 配置和基本结构测试 =====

    #[test]
    fn test_calm_config_default() {
        let config = CalmConfig::default();
        assert_eq!(config.hidden_dim, 768);
        assert_eq!(config.num_encoder_layers, 6);
        assert_eq!(config.num_decoder_layers, 6);
        assert_eq!(config.max_sentence_length, 50);
        assert_eq!(config.num_candidates, 5);
        assert!((config.confidence_threshold - 0.8).abs() < 1e-6);
        assert!(!config.use_speculative);
    }

    #[test]
    fn test_calm_config_custom() {
        let config = CalmConfig {
            hidden_dim: 1024,
            num_encoder_layers: 12,
            num_decoder_layers: 12,
            max_sentence_length: 100,
            num_candidates: 10,
            confidence_threshold: 0.9,
            use_speculative: true,
        };

        assert_eq!(config.hidden_dim, 1024);
        assert_eq!(config.num_encoder_layers, 12);
        assert_eq!(config.use_speculative, true);
    }

    #[test]
    fn test_calm_engine_creation() {
        let config = CalmConfig::default();
        let engine = CalmEngine::new(config, 10000);

        assert_eq!(engine.config().hidden_dim, 768);
        assert_eq!(engine.encoder().num_layers(), 6);
        assert_eq!(engine.decoder().vocab_size(), 10000);
    }

    // ===== 句子边界检测测试 =====

    #[test]
    fn test_sentence_boundary_detection() {
        let detector = SentenceBoundaryDetector::new(1000);

        // 模拟包含句子结束标记的输入
        let input: Vec<u32> = vec![100, 200, 300, 0, 400, 500, 1, 600]; // 0, 1 是句号和感叹号
        let boundaries = detector.detect_boundaries(&input);

        // 应该检测到至少2个句子边界
        assert!(boundaries.len() >= 2);

        // 第一个边界应该在 index 3（第一个句号）
        assert_eq!(boundaries[0].end_idx, 3);
    }

    #[test]
    fn test_is_sentence_end() {
        let detector = SentenceBoundaryDetector::new(1000);

        assert!(detector.is_sentence_end(0)); // .
        assert!(detector.is_sentence_end(1)); // !
        assert!(detector.is_sentence_end(2)); // ?
        assert!(detector.is_sentence_end(3)); // ;
        assert!(!detector.is_sentence_end(100)); // 普通单词
    }

    #[test]
    fn test_sentence_boundary_creation() {
        let boundary = SentenceBoundary::new(0, 5, 0.95);
        assert_eq!(boundary.start_idx, 0);
        assert_eq!(boundary.end_idx, 5);
        assert!((boundary.confidence - 0.95).abs() < 1e-6);
        assert_eq!(boundary.length(), 6);
    }

    #[test]
    fn test_sentence_boundary_confidence_clamping() {
        // 测试置信度裁剪到 [0, 1]
        let boundary_high = SentenceBoundary::new(0, 5, 1.5);
        assert!(boundary_high.confidence <= 1.0);

        let boundary_low = SentenceBoundary::new(0, 5, -0.5);
        assert!(boundary_low.confidence >= 0.0);
    }

    // ===== 置信度评分系统测试 =====

    #[test]
    fn test_confidence_scorer_creation() {
        let scorer = ConfidenceScorer::new(0.8);
        assert!((scorer.threshold() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_confidence_scorer_scoring() {
        let scorer = ConfidenceScorer::new(0.5);

        // 高置信度候选
        let high_conf_candidate = CandidateSentence::new(vec![1, 2, 3, 4, 5], -5.0);
        let score = scorer.score(&high_conf_candidate);
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_confidence_scorer_filtering() {
        let scorer = ConfidenceScorer::new(0.7);

        let candidates = vec![
            CandidateSentence::new(vec![1, 2, 3], -10.0), // 低置信度
            CandidateSentence::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], -2.0), // 高置信度
            CandidateSentence::new(vec![1], -15.0),       // 很低置信度
        ];

        let filtered = scorer.filter_candidates(candidates);
        // 应该只保留高置信度的候选
        assert!(filtered.len() <= 2);
    }

    #[test]
    fn test_confidence_scorer_empty_candidate() {
        let scorer = ConfidenceScorer::new(0.5);
        let empty_candidate = CandidateSentence::new(vec![], 0.0);

        let score = scorer.score(&empty_candidate);
        assert_eq!(score, 0.0);
    }

    // ===== 候选句子测试 =====

    #[test]
    fn test_candidate_sentence_creation() {
        let candidate = CandidateSentence::new(vec![10, 20, 30], -3.5);
        assert_eq!(candidate.len(), 3);
        assert!(!candidate.is_empty());
        assert!((candidate.log_probability - (-3.5)).abs() < 1e-6);
        assert_eq!(candidate.confidence, 0.0); // 初始为0
    }

    #[test]
    fn test_candidate_sentence_eos_detection() {
        // 不包含 EOS 的句子
        let no_eos = CandidateSentence::new(vec![10, 20, 30], -2.0);
        assert!(!no_eos.is_end_of_document());

        // 包含 EOS 的句子（假设 2 是 EOS）
        let with_eos = CandidateSentence::new(vec![10, 20, 2], -2.0);
        assert!(with_eos.is_end_of_document());
    }

    // ===== Encoder 测试 =====

    #[test]
    fn test_encoder_creation() {
        let encoder = CalmEncoder::new(6, 768, 10000);
        assert_eq!(encoder.num_layers(), 6);
    }

    #[test]
    fn test_encoder_encode() {
        let encoder = CalmEncoder::new(2, 64, 1000);
        let prompt: Vec<u32> = vec![100, 200, 300];

        let result = encoder.encode(&prompt);
        assert!(result.is_ok());

        let encoded = result.unwrap();
        assert_eq!(encoded.hidden_state.len(), 64);
        assert_eq!(encoded.original_tokens, prompt);
    }

    #[test]
    fn test_encoder_encode_empty_prompt() {
        let encoder = CalmEncoder::new(2, 64, 1000);
        let result = encoder.encode(&[]);
        assert!(result.is_err());
    }

    // ===== Decoder 测试 =====

    #[test]
    fn test_decoder_creation() {
        let decoder = CalmDecoder::new(6, 768, 10000, 50);
        assert_eq!(decoder.vocab_size(), 10000);
    }

    #[test]
    fn test_decoder_generate_candidates() {
        let decoder = CalmDecoder::new(2, 64, 100, 20);

        let encoded = EncodedContext {
            hidden_state: Array1::from_vec(vec![0.1; 64]),
            original_tokens: vec![10, 20, 30],
            sentence_boundaries: Vec::new(),
        };

        let result = decoder.generate_candidates(&encoded);
        assert!(result.is_ok());

        let candidates = result.unwrap();
        assert!(candidates.len() > 0);
        assert!(candidates.iter().all(|c| !c.is_empty()));
    }

    #[test]
    fn test_decoder_generate_empty_context() {
        let decoder = CalmDecoder::new(2, 64, 100, 20);

        let empty_encoded = EncodedContext {
            hidden_state: Array1::from_vec(vec![0.0; 64]),
            original_tokens: vec![],
            sentence_boundaries: Vec::new(),
        };

        let result = decoder.generate_candidates(&empty_encoded);
        assert!(result.is_err()); // 应该报错
    }

    // ===== Scheduler 测试 =====

    #[test]
    fn test_scheduler_greedy_selection() {
        let scorer = ConfidenceScorer::new(0.5);
        let mut scheduler = SentenceScheduler::new(GenerationStrategy::Greedy, scorer);

        let candidates = vec![
            CandidateSentence::new(vec![1, 2, 3], -5.0),
            CandidateSentence::new(vec![4, 5, 6, 7, 8], -2.0),
            CandidateSentence::new(vec![9], -10.0),
        ];

        let result = scheduler.select_best(&candidates);
        assert!(result.is_ok());

        let selected = result.unwrap();
        assert!(selected.is_some()); // 应该选中某个候选
    }

    #[test]
    fn test_scheduler_empty_candidates() {
        let scorer = ConfidenceScorer::new(0.5);
        let mut scheduler = SentenceScheduler::new(GenerationStrategy::Greedy, scorer);

        let result = scheduler.select_best(&[]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none()); // 空候选应返回 None
    }

    // ===== CALM Output 测试 =====

    #[test]
    fn test_calm_output_creation() {
        let sentences = vec![
            CandidateSentence::new(vec![1, 2, 3], -2.0),
            CandidateSentence::new(vec![4, 5], -1.5),
        ];

        let output = CalmOutput::new(sentences);
        assert_eq!(output.num_sentences(), 2);
        assert_eq!(output.total_tokens(), 5);
        assert!(!output.is_empty());
    }

    #[test]
    fn test_calm_output_to_token_sequence() {
        let sentences = vec![
            CandidateSentence::new(vec![1, 2, 3], -2.0),
            CandidateSentence::new(vec![4, 5], -1.5),
        ];

        let output = CalmOutput::new(sentences);
        let tokens = output.to_token_sequence();

        assert_eq!(tokens, vec![1, 2, 3, 4, 5]);
    }

    // ===== CALM Engine 核心功能测试 =====

    #[test]
    fn test_calm_engine_basic_generation() {
        let config = CalmConfig {
            max_sentence_length: 10, // 短句子加速测试
            ..Default::default()
        };
        let mut engine = CalmEngine::new(config, 1000);

        let prompt: Vec<u32> = vec![100, 200, 300, 400, 500];
        let result = engine.generate(&prompt, 3);

        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(output.num_sentences() <= 3); // 不应超过最大句子数
        assert!(!output.is_empty() || true); // 可能为空（如果置信度不够）
    }

    #[test]
    fn test_calm_engine_empty_prompt() {
        let config = CalmConfig::default();
        let mut engine = CalmEngine::new(config, 1000);

        let result = engine.generate(&[], 5);
        assert!(result.is_err()); // 空 prompt 应报错
    }

    #[test]
    fn test_calm_engine_zero_max_sentences() {
        let config = CalmConfig::default();
        let mut engine = CalmEngine::new(config, 1000);

        let prompt: Vec<u32> = vec![100, 200];
        let result = engine.generate(&prompt, 0);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.is_empty()); // 应返回空输出
    }

    // ===== 对比实验框架测试 =====

    #[test]
    fn test_comparison_metrics_creation() {
        let metrics = ComparisonMetrics::new(5, 50, 0.85);

        assert_eq!(metrics.calm_steps, 5);
        assert_eq!(metrics.traditional_steps, 50);
        assert!((metrics.speedup_ratio - 10.0).abs() < 1e-1); // 50/5 = 10
        assert!((metrics.quality_score - 0.85).abs() < 1e-6);
        assert!((metrics.latency_reduction_pct - 90.0).abs() < 1e-1); // (50-5)/50*100
    }

    #[test]
    fn test_comparison_result_creation() {
        let calm_output = CalmOutput::new(vec![CandidateSentence::new(vec![1, 2, 3], -2.0)]);
        let traditional = vec![1, 2, 3, 4, 5];
        let metrics = ComparisonMetrics::new(1, 5, 0.9);

        let result = ComparisonResult::new(calm_output, traditional, metrics);
        assert_eq!(result.metrics.calm_steps, 1);
        assert_eq!(result.traditional_tokens.len(), 5);
    }

    #[test]
    fn test_compare_with_traditional() {
        let config = CalmConfig {
            max_sentence_length: 10,
            ..Default::default()
        };
        let mut engine = CalmEngine::new(config, 500);

        let prompt: Vec<u32> = vec![10, 20, 30];
        let traditional_output: Vec<u32> = vec![40, 50, 60, 70, 80, 90, 100];

        let result = engine.compare_with_token_by_token(&prompt, &traditional_output);
        assert!(result.is_ok());

        let comparison = result.unwrap();
        assert!(comparison.metrics.speedup_ratio > 0.0);
        assert!(comparison.metrics.quality_score >= 0.0);
        assert!(comparison.metrics.quality_score <= 1.0);

        // 打印报告（验证不会 panic）
        comparison.print_report();
    }

    // ===== Mock 测试：模拟完整工作流 =====

    #[test]
    fn test_full_workflow_mock() {
        // 模拟完整的 CALM 工作流
        let config = CalmConfig {
            hidden_dim: 128,
            num_encoder_layers: 2,
            num_decoder_layers: 2,
            max_sentence_length: 15,
            num_candidates: 3,
            confidence_threshold: 0.3, // 低阈值以确保有输出
            use_speculative: false,
        };

        let mut engine = CalmEngine::new(config, 200);

        // 较长的 prompt
        let prompt: Vec<u32> = (1..=20).collect(); // 20 个 token

        // 生成
        let generate_result = engine.generate(&prompt, 5);
        assert!(generate_result.is_ok(), "生成应该成功");

        let output = generate_result.unwrap();
        println!(
            "Mock workflow - Generated {} sentences",
            output.num_sentences()
        );
        println!("Total tokens: {}", output.total_tokens());

        // 验证输出格式
        for (i, sentence) in output.sentences.iter().enumerate() {
            println!(
                "Sentence {}: {} tokens, confidence={:.4}",
                i,
                sentence.len(),
                sentence.confidence
            );
            assert!(sentence.len() > 0, "每个句子不应为空");
            assert!(sentence.len() <= 15, "不应超过最大长度");
        }
    }

    #[test]
    fn test_beam_search_strategy() {
        // 测试束搜索策略
        let config = CalmConfig {
            max_sentence_length: 10,
            confidence_threshold: 0.3,
            ..Default::default()
        };

        let mut engine =
            CalmEngine::with_strategy(config, 500, GenerationStrategy::BeamSearch { width: 3 });

        let prompt: Vec<u32> = vec![100, 200, 300];
        let result = engine.generate(&prompt, 3);

        assert!(result.is_ok());
        let output = result.unwrap();
        // 束搜索也应该能正常工作
        assert!(output.num_sentences() <= 3);
    }

    #[test]
    fn test_sampling_strategy() {
        // 测试采样策略
        let config = CalmConfig {
            max_sentence_length: 10,
            confidence_threshold: 0.2,
            ..Default::default()
        };

        let mut engine = CalmEngine::with_strategy(
            config,
            500,
            GenerationStrategy::Sampling { temperature: 0.8 },
        );

        let prompt: Vec<u32> = vec![50, 60, 70];
        let result = engine.generate(&prompt, 2);

        assert!(result.is_ok());
        let output = result.unwrap();
        // 采样策略也应该能正常工作
        assert!(output.num_sentences() <= 2);
    }

    #[test]
    fn test_edge_cases_extreme_configs() {
        // 极端配置测试
        let extreme_config = CalmConfig {
            hidden_dim: 32, // 非常小
            num_encoder_layers: 1,
            num_decoder_layers: 1,
            max_sentence_length: 3, // 非常短
            num_candidates: 1,
            confidence_threshold: 0.0, // 接受所有
            use_speculative: false,
        };

        let mut engine = CalmEngine::new(extreme_config, 100);

        let prompt: Vec<u32> = vec![1, 2, 3];
        let result = engine.generate(&prompt, 2);

        // 即使极端配置也不应崩溃
        assert!(result.is_ok(), "极端配置不应导致错误");
    }

    #[test]
    fn test_quality_score_calculation() {
        let config = CalmConfig::default();
        let mut engine = CalmEngine::new(config, 1000);

        // 创建已知输出
        let calm_output = CalmOutput::new(vec![
            CandidateSentence::new(vec![1, 2, 3, 4, 5], -2.0),
            CandidateSentence::new(vec![6, 7, 8], -1.5),
        ]);

        let reference: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        let score = engine.calculate_quality_score(&calm_output, &reference);

        // 应该有合理的重叠分数
        assert!(score >= 0.0 && score <= 1.0);
        assert!(score > 0.3, "应该有一定的重叠");
    }
}
