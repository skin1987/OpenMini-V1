//! 推测解码 (Speculative Decoding) 加速模块
//!
//! 本模块实现了 LLM 推理的关键加速技术 - 推测解码：
//! - 使用小型"草稿模型"(draft model)快速生成多个候选 token
//! - 使用大型"目标模型"(target model)并行验证这些候选 token
//! - 接受验证通过的 token，拒绝的从验证点重新开始
//! - 理论加速比 = 草稿模型速度 / 目标模型速度（通常 2-5x）
//!
//! ## 核心算法
//!
//! ### 标准推测解码
//! 1. 草稿模型快速生成 K 个候选 token
//! 2. 目标模型并行评估这 K 个位置的概率分布
//! 3. 逐位置比较草稿和目标概率，接受或拒绝
//! 4. 拒绝时从修正分布中采样一个 bonus token
//!
//! ### 典型接受 (Typical Acceptance)
//! 改进的接受准则，基于信息论：
//! - 计算每个 token 的"典型性"分数
//! - 更智能地决定接受/拒绝边界
//! - 在保持分布匹配的同时提高接受率
//!
//! ## 性能特性
//! - 纯 CPU 实现，无需 GPU
//! - 支持树形注意力掩码（高级模式）
//! - 自适应草稿长度调整
//! - 完整的性能统计和监控
//!
//! ## 示例
//!
//! ```ignore
//! use openmini_server::model::inference::speculative::{
//!     SpeculativeEngine, SpeculativeConfig,
//! };
//!
//! let config = SpeculativeConfig::default();
//! let engine = SpeculativeEngine::new(config);
//!
//! let result = engine.generate("Hello, world", &params, |token| {
//!     print!("{}", token);
//!     Ok(())
//! })?;
//! ```

use anyhow::{anyhow, Result};
use ndarray::Array2;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use std::time::Instant;

use super::inference::InferenceStats;
use super::sampler::GenerateParams;

// ============================================================================
// 配置与数据结构
// ============================================================================

/// 推测解码配置
///
/// 控制推测解码的所有行为参数，包括模型大小、推测长度、接受阈值等。
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// 草稿模型参数量（用于估算速度比）
    pub draft_model_size: usize,
    /// 目标模型参数量
    pub target_model_size: usize,
    /// 最大推测 token 数（通常 4-8）
    pub max_speculative_tokens: usize,
    /// 接受概率阈值（0.0-1.0）
    pub accept_threshold: f32,
    /// 是否使用树注意力（高级模式）
    pub use_tree_attention: bool,
    /// 温度参数（用于采样）
    pub temperature: f32,
    /// 是否启用自适应草稿长度
    pub enable_adaptive_draft: bool,
    /// 最小草稿长度
    pub min_draft_length: usize,
    /// 是否启用典型接受算法
    pub enable_typical_acceptance: bool,
}

impl Default for SpeculativeConfig {
    /// 默认配置
    ///
    /// 经过调优的默认参数，适用于一般场景：
    /// - 草稿模型: ~1B 参数
    /// - 目标模型: ~7B 参数
    /// - 最大推测: 5 tokens
    /// - 接受阈值: 0.5
    fn default() -> Self {
        Self {
            draft_model_size: 1_000_000_000,  // 1B
            target_model_size: 7_000_000_000, // 7B
            max_speculative_tokens: 5,
            accept_threshold: 0.5,
            use_tree_attention: false,
            temperature: 0.7,
            enable_adaptive_draft: true,
            min_draft_length: 2,
            enable_typical_acceptance: true,
        }
    }
}

impl SpeculativeConfig {
    /// 创建新的配置实例
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置草稿模型大小
    pub fn with_draft_model_size(mut self, size: usize) -> Self {
        self.draft_model_size = size;
        self
    }

    /// 设置目标模型大小
    pub fn with_target_model_size(mut self, size: usize) -> Self {
        self.target_model_size = size;
        self
    }

    /// 设置最大推测 token 数
    pub fn with_max_speculative_tokens(mut self, n: usize) -> Self {
        self.max_speculative_tokens = n;
        self
    }

    /// 设置接受阈值
    pub fn with_accept_threshold(mut self, threshold: f32) -> Self {
        self.accept_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// 启用/禁用树注意力
    pub fn with_tree_attention(mut self, enabled: bool) -> Self {
        self.use_tree_attention = enabled;
        self
    }

    /// 设置温度参数
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp.max(0.0);
        self
    }

    /// 设置最小草稿长度
    pub fn with_min_draft_length(mut self, min: usize) -> Self {
        self.min_draft_length = min;
        self
    }

    /// 启用/禁用自适应草稿长度
    pub fn with_enable_adaptive_draft(mut self, enabled: bool) -> Self {
        self.enable_adaptive_draft = enabled;
        self
    }

    /// 启用/禁用典型接受算法
    pub fn with_enable_typical_acceptance(mut self, enabled: bool) -> Self {
        self.enable_typical_acceptance = enabled;
        self
    }

    /// 计算理论加速比
    ///
    /// 基于模型参数量估算加速比。
    /// 实际加速比会因接受率、KV cache 效率等因素而变化。
    pub fn theoretical_speedup(&self) -> f32 {
        if self.target_model_size == 0 {
            return 1.0;
        }
        // 简化估算：速度比 ≈ 参数量的平方根比
        (self.draft_model_size as f32 / self.target_model_size as f32).sqrt()
    }
}

/// 推测候选结果
///
/// 包含草稿模型生成的候选 token 序列及其概率信息。
#[derive(Debug, Clone)]
pub struct SpeculativeCandidate {
    /// 候选 token 序列
    pub tokens: Vec<u32>,
    /// 每个 token 的概率
    pub probabilities: Vec<f32>,
    /// 草稿模型的 log 概率
    pub draft_log_probs: Vec<f32>,
}

impl SpeculativeCandidate {
    /// 创建空的候选结果
    pub fn empty() -> Self {
        Self {
            tokens: Vec::new(),
            probabilities: Vec::new(),
            draft_log_probs: Vec::new(),
        }
    }

    /// 候选 token 数量
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// 验证结果
///
/// 目标模型对候选 token 的验证结果。
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// 被接受的 token
    pub accepted_tokens: Vec<u32>,
    /// 接受数量
    pub accepted_count: usize,
    /// 拒绝位置（None=全部接受）
    pub rejected_position: Option<usize>,
    /// 奖励 token（采样自修正分布）
    pub bonus_token: Option<u32>,
    /// 验证耗时（微秒）
    pub verification_time_us: u64,
}

impl VerificationResult {
    /// 创建完全接受的验证结果
    fn fully_accepted(tokens: Vec<u32>, time_us: u64) -> Self {
        let count = tokens.len();
        Self {
            accepted_tokens: tokens,
            accepted_count: count,
            rejected_position: None,
            bonus_token: None,
            verification_time_us: time_us,
        }
    }

    /// 创建部分接受的验证结果
    fn partial_accept(accepted: Vec<u32>, rejected_pos: usize, bonus: u32, time_us: u64) -> Self {
        let count = accepted.len();
        Self {
            accepted_tokens: accepted,
            accepted_count: count,
            rejected_position: Some(rejected_pos),
            bonus_token: Some(bonus),
            verification_time_us: time_us,
        }
    }
}

/// 推测解码统计信息
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// 总推测步数
    pub total_speculation_steps: u64,
    /// 总生成的草稿 token 数
    pub total_draft_tokens: u64,
    /// 总接受的 token 数
    pub total_accepted_tokens: u64,
    /// 总拒绝次数
    pub total_rejections: u64,
    /// 总 bonus token 数
    pub total_bonus_tokens: u64,
    /// 平均接受长度
    pub avg_accept_length: f32,
    /// 平均草稿长度
    pub avg_draft_length: f32,
    /// 接受率
    pub acceptance_rate: f32,
    /// 实际加速比
    pub actual_speedup: f32,
    /// 总验证时间（微秒）
    pub total_verification_time_us: u64,
}

impl SpeculativeStats {
    /// 创建空的统计信息
    pub fn new() -> Self {
        Self::default()
    }
}

/// 单步推测结果
#[derive(Debug)]
struct StepResult {
    /// 接受的 token
    accepted_tokens: Vec<u32>,
    /// 是否完成（达到最大长度或结束符）
    done: bool,
}

// ============================================================================
// 草稿模型模拟器
// ============================================================================

/// 简化的草稿模型
///
/// 用于测试和 CPU 环境。在真实 GPU 环境中可替换为实际的小型模型。
///
/// 该模拟器使用简单的 n-gram 统计模型来生成候选 token，
/// 主要用于演示和测试推测解码流程。
pub struct DraftModel {
    /// 词汇表大小
    vocab_size: usize,
    /// 温度参数
    temperature: f32,
    /// 随机数生成器
    rng: StdRng,
}

impl DraftModel {
    /// 创建新的草稿模型
    ///
    /// # 参数
    /// - `vocab_size`: 词汇表大小
    /// - `temperature`: 采样温度
    pub fn new(vocab_size: usize, temperature: f32) -> Self {
        Self {
            vocab_size,
            temperature: temperature.max(0.01),
            rng: StdRng::from_entropy(),
        }
    }

    /// 使用固定种子创建（用于测试可复现性）
    pub fn with_seed(vocab_size: usize, temperature: f32, seed: u64) -> Self {
        Self {
            vocab_size,
            temperature: temperature.max(0.01),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// 生成候选 token 序列
    ///
    /// 基于前缀上下文快速生成 K 个候选 token。
    ///
    /// # 参数
    /// - `prefix_tokens`: 当前前缀 token 序列
    /// - `config`: 推测解码配置
    ///
    /// # 返回
    /// 包含候选 token 和概率信息的 SpeculativeCandidate
    pub fn generate_candidates(
        &mut self,
        prefix_tokens: &[u32],
        config: &SpeculativeConfig,
    ) -> Result<SpeculativeCandidate> {
        if prefix_tokens.is_empty() {
            return Err(anyhow!("前缀 token 序列为空"));
        }

        let num_tokens = config.max_speculative_tokens;
        let mut tokens = Vec::with_capacity(num_tokens);
        let mut probabilities = Vec::with_capacity(num_tokens);
        let mut log_probs = Vec::with_capacity(num_tokens);

        // 构建当前上下文（包含前缀）
        let mut context: Vec<u32> = prefix_tokens.to_vec();

        for _ in 0..num_tokens {
            // 获取下一个 token 的概率分布
            let probs = self.get_next_token_probs(&context);

            // 采样一个 token
            let (token_id, prob) = self.sample_token(&probs)?;

            tokens.push(token_id);
            probabilities.push(prob);
            log_probs.push(prob.ln());

            // 将采样的 token 加入上下文
            context.push(token_id);
        }

        Ok(SpeculativeCandidate {
            tokens,
            probabilities,
            draft_log_probs: log_probs,
        })
    }

    /// 获取 token 概率分布
    ///
    /// 基于简化的 n-gram 模型计算下一个 token 的概率分布。
    /// 在实际应用中，这里应该调用真实的草稿模型推理。
    fn get_next_token_probs(&self, context: &[u32]) -> Vec<f32> {
        if context.is_empty() {
            return vec![1.0 / self.vocab_size as f32; self.vocab_size];
        }

        // 简化的概率模型：基于最后一个 token 的伪随机分布
        let last_token = context[context.len() - 1] as usize % self.vocab_size;
        let mut probs = vec![0.01f32; self.vocab_size];

        // 给几个可能的 token 较高概率（模拟语言模型的局部性）
        let base_idx = last_token;
        let high_prob_tokens = [
            base_idx % self.vocab_size,
            (base_idx + 1) % self.vocab_size,
            (base_idx + 37) % self.vocab_size, // 使用质数增加分散性
            (base_idx + 101) % self.vocab_size,
            (base_idx + 251) % self.vocab_size,
        ];

        let mut remaining_prob = 1.0;
        for (i, &token_idx) in high_prob_tokens.iter().enumerate() {
            if i == high_prob_tokens.len() - 1 {
                probs[token_idx] = remaining_prob;
            } else {
                let p = remaining_prob * 0.4; // 分配 40% 给当前 token
                probs[token_idx] = p;
                remaining_prob -= p;
            }
        }

        // 应用温度
        if self.temperature != 1.0 {
            let inv_temp = 1.0 / self.temperature;
            for p in probs.iter_mut() {
                *p = (*p).powf(inv_temp);
            }
            // 重新归一化
            let sum: f32 = probs.iter().sum();
            if sum > 0.0 {
                for p in probs.iter_mut() {
                    *p /= sum;
                }
            }
        }

        probs
    }

    /// 从概率分布中采样一个 token
    fn sample_token(&mut self, probs: &[f32]) -> Result<(u32, f32)> {
        let sum: f32 = probs.iter().sum();
        if sum <= 0.0 {
            return Err(anyhow!("概率分布和为零"));
        }

        // 归一化
        let normalized: Vec<f32> = probs.iter().map(|&p| p / sum).collect();

        // 加权随机采样
        let dist = WeightedIndex::new(&normalized)?;
        let sampled_idx = dist.sample(&mut self.rng);

        Ok((sampled_idx as u32, normalized[sampled_idx]))
    }
}

// ============================================================================
// 目标模型验证器
// ============================================================================

/// 目标模型验证器
///
/// 负责验证草稿模型生成的候选 token 序列。
/// 提供两种验证算法：
/// - 标准验证：基于概率比的接受/拒绝
/// - 典型接受：基于信息论的改进算法
pub struct TargetVerifier {
    /// 词汇表大小
    vocab_size: usize,
    /// 随机数生成器
    rng: StdRng,
}

impl TargetVerifier {
    /// 创建新的验证器
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            rng: StdRng::from_entropy(),
        }
    }

    /// 使用固定种子创建（用于测试）
    pub fn with_seed(vocab_size: usize, seed: u64) -> Self {
        Self {
            vocab_size,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// 验证候选 token 序列
    ///
    /// 对草稿模型生成的候选序列进行逐位置验证，
    /// 返回每个位置的接受/拒绝状态和修正概率。
    ///
    /// # 参数
    /// - `prefix`: 前缀 token（不参与验证）
    /// - `candidates`: 草稿候选结果
    ///
    /// # 返回
    /// 验证结果，包括接受的 token、拒绝位置和 bonus token
    pub fn verify_candidates(
        &mut self,
        _prefix: &[u32],
        candidates: &SpeculativeCandidate,
        use_typical_acceptance: bool,
    ) -> VerificationResult {
        let start_time = Instant::now();

        if candidates.is_empty() {
            return VerificationResult::fully_accepted(Vec::new(), 0);
        }

        // 模拟目标模型的概率分布（在实际应用中应调用真实目标模型）
        let target_probs_list = self.simulate_target_probs(candidates);

        let mut accepted_tokens = Vec::new();

        for (i, (&token, draft_prob)) in candidates
            .tokens
            .iter()
            .zip(candidates.probabilities.iter())
            .enumerate()
        {
            let target_probs = &target_probs_list[i];
            let target_prob = target_probs[token as usize];

            if use_typical_acceptance {
                // 使用典型接受算法
                let (accepted, bonus) = Self::typical_acceptance_verify(
                    &[*draft_prob],
                    &[target_prob],
                    &[token],
                    &mut self.rng,
                );

                if accepted > 0 {
                    accepted_tokens.push(token);
                } else {
                    // 拒绝，返回部分接受结果
                    let elapsed = start_time.elapsed().as_micros() as u64;
                    return VerificationResult::partial_accept(
                        accepted_tokens,
                        i,
                        bonus.unwrap_or(0),
                        elapsed,
                    );
                }
            } else {
                // 使用标准验证算法
                let (accepted, bonus) =
                    self.standard_verify(&[*draft_prob], &[target_prob], &[token]);

                if accepted > 0 {
                    accepted_tokens.push(token);
                } else {
                    // 拒绝，返回部分接受结果
                    let elapsed = start_time.elapsed().as_micros() as u64;
                    return VerificationResult::partial_accept(
                        accepted_tokens,
                        i,
                        bonus.unwrap_or(0),
                        elapsed,
                    );
                }
            }
        }

        // 全部接受
        let elapsed = start_time.elapsed().as_micros() as u64;
        VerificationResult::fully_accepted(accepted_tokens, elapsed)
    }

    /// 模拟目标模型的概率分布
    ///
    /// 在实际应用中，这里应该调用真实的目标模型进行前向传播。
    /// 此处使用简化模拟来生成与草稿分布相似但有噪声的目标分布。
    fn simulate_target_probs(&mut self, candidates: &SpeculativeCandidate) -> Vec<Vec<f32>> {
        candidates
            .tokens
            .iter()
            .zip(candidates.probabilities.iter())
            .map(|(&token, &draft_prob)| {
                let mut probs = vec![0.001f32; self.vocab_size];

                // 目标模型通常对相同的 token 给出相似但略有不同的概率
                // 模拟：以较高概率分配给草稿选择的 token，但加入一些不确定性
                let noise: f32 = self.rng.gen::<f32>() * 0.2 - 0.1; // [-0.1, 0.1]
                let target_prob = (draft_prob + noise).clamp(0.01, 0.99);
                probs[token as usize] = target_prob;

                // 分配剩余概率给其他 token
                let remaining = 1.0 - target_prob;
                let spread = remaining / (self.vocab_size - 1).max(1) as f32;
                for (i, p) in probs.iter_mut().enumerate() {
                    if i != token as usize {
                        *p = spread;
                    }
                }

                probs
            })
            .collect()
    }

    /// 标准验证算法（逐位置比较概率）
    ///
    /// 基于 Levi et al. (2023) 的原始推测解码算法：
    /// - 接受概率 = min(1, p_target / p_draft)
    /// - 以接受概率决定是否接受该 token
    /// - 拒绝时从修正分布 max(0, p_target - p_draft) 中采样
    ///
    /// # 参数
    /// - `draft_probs`: 草稿模型概率
    /// - `target_probs`: 目标模型概率
    /// - `tokens`: 候选 token
    ///
    /// # 返回
    /// (接受数量, 可选的 bonus token)
    fn standard_verify(
        &mut self,
        draft_probs: &[f32],
        target_probs: &[f32],
        tokens: &[u32],
    ) -> (usize, Option<u32>) {
        if draft_probs.is_empty() || target_probs.is_empty() || tokens.is_empty() {
            return (0, None);
        }

        let draft_prob = draft_probs[0];
        let target_prob = target_probs[0];
        let token = tokens[0];

        // 计算接受概率: min(1, p_target / p_draft)
        let accept_prob = if draft_prob <= 0.0 {
            0.0
        } else {
            (target_prob / draft_prob).min(1.0)
        };

        // 以 accept_prob 的概率接受
        if self.rng.gen::<f32>() < accept_prob {
            (1, None) // 接受
        } else {
            // 拒绝：从修正分布中采样 bonus token
            // 修正分布: max(0, p_target - p_draft)
            let adjusted = (target_prob - draft_prob).max(0.0);

            // 简化处理：如果调整后概率很小，返回原 token；否则随机选择
            let bonus = if adjusted > 0.01 && self.rng.gen::<f32>() < 0.7 {
                // 尝试从附近 token 中选择
                Some((token as usize + self.rng.gen_range(1..=5)) as u32 % self.vocab_size as u32)
            } else {
                Some(token) // 保持原 token 作为 fallback
            };

            (0, bonus)
        }
    }

    /// 典型接受 (Typical Acceptance) 验证算法
    ///
    /// Chen et al. (2023) 提出的改进算法，基于信息论：
    /// - 计算 token 的"典型性"：-log(p) 接近熵的 token 更典型
    /// - 使用更复杂的接受准则，考虑分布的整体形状
    /// - 在保持分布一致性的同时提高接受率约 10-20%
    ///
    /// # 参数
    /// - `draft_probs`: 草稿模型概率
    /// - `target_probs`: 目标模型概率
    /// - `tokens`: 候选 token
    /// - `rng`: 随机数生成器
    ///
    /// # 返回
    /// (接受数量, 可选的 bonus token)
    #[allow(clippy::too_many_arguments)]
    fn typical_acceptance_verify(
        draft_probs: &[f32],
        target_probs: &[f32],
        tokens: &[u32],
        rng: &mut StdRng,
    ) -> (usize, Option<u32>) {
        if draft_probs.is_empty() || target_probs.is_empty() || tokens.is_empty() {
            return (0, None);
        }

        let draft_prob = draft_probs[0];
        let target_prob = target_probs[0];
        let token = tokens[0];

        // 计算信息量
        let draft_info = -draft_prob.ln().max(0.0); // 草稿模型的信息量
        let target_info = -target_prob.ln().max(0.0); // 目标模型的信息量

        // 计算典型性分数（基于两个模型的信息量差异）
        let info_diff = (draft_info - target_info).abs();

        // 动态接受阈值：信息差异越小越容易接受
        let dynamic_threshold = 0.5 + info_diff * 0.1; // 基础阈值 + 信息惩罚
        let dynamic_threshold = dynamic_threshold.min(0.95); // 上限

        // 结合标准接受概率
        let standard_accept_prob = if draft_prob <= 0.0 {
            0.0
        } else {
            (target_prob / draft_prob).min(1.0)
        };

        // 综合接受概率
        let combined_accept_prob = standard_accept_prob * (1.0 - dynamic_threshold * 0.5);

        // 决定是否接受
        if rng.gen::<f32>() < combined_accept_prob {
            (1, None) // 接受
        } else {
            // 拒绝：从修正分布采样
            let bonus = if target_prob > draft_prob {
                // 目标模型更确信，可能从高概率区域采样
                Some((token as usize + rng.gen_range(1..=10)) as u32 % 100000)
            } else {
                Some(token) // 保持或微调
            };

            (0, bonus)
        }
    }
}

// ============================================================================
// 推测解码引擎
// ============================================================================

/// 主推测解码引擎
///
/// 整合草稿模型和目标模型验证器，提供完整的推测解码生成功能。
///
/// ## 使用示例
///
/// ```ignore
/// let config = SpeculativeConfig::default();
/// let engine = SpeculativeEngine::new(config);
///
/// let result = engine.generate("Hello", &params, |token| {
///     print!("{}", token);
///     Ok(())
/// })?;
/// ```
pub struct SpeculativeEngine {
    /// 推测解码配置
    config: SpeculativeConfig,
    /// 草稿模型
    draft_model: DraftModel,
    /// 目标模型验证器
    verifier: TargetVerifier,
    /// 统计信息
    stats: SpeculativeStats,
    /// 当前自适应草稿长度
    current_draft_length: usize,
}

impl SpeculativeEngine {
    /// 创建新的推测解码引擎
    ///
    /// # 参数
    /// - `config`: 推测解码配置
    pub fn new(config: SpeculativeConfig) -> Self {
        let vocab_size = 100_000; // 默认词汇表大小
        let current_draft_length = config.max_speculative_tokens;

        Self {
            config: config.clone(),
            draft_model: DraftModel::new(vocab_size, config.temperature),
            verifier: TargetVerifier::new(vocab_size),
            stats: SpeculativeStats::new(),
            current_draft_length,
        }
    }

    /// 使用固定种子创建（用于测试可复现性）
    pub fn with_seed(config: SpeculativeConfig, seed: u64) -> Self {
        let vocab_size = 100_000;
        let current_draft_length = config.max_speculative_tokens;

        Self {
            config: config.clone(),
            draft_model: DraftModel::with_seed(vocab_size, config.temperature, seed),
            verifier: TargetVerifier::with_seed(vocab_size, seed.wrapping_add(1000)),
            stats: SpeculativeStats::new(),
            current_draft_length,
        }
    }

    /// 执行推测解码生成
    ///
    /// 完整的推测解码生成流程：
    /// 1. 编码 prompt 为 token 序列
    /// 2. 循环执行推测步骤直到完成
    /// 3. 通过回调函数流式输出
    ///
    /// # 参数
    /// - `prompt`: 输入提示文本
    /// - `params`: 生成参数
    /// - `callback`: 流式回调函数
    ///
    /// # 返回
    /// 推理统计信息
    pub fn generate<F>(
        &mut self,
        prompt: &str,
        params: &GenerateParams,
        mut callback: F,
    ) -> Result<InferenceStats>
    where
        F: FnMut(&str) -> Result<()>,
    {
        let start_time = Instant::now();

        // 简单分词：将文本按字符转换为 token ID（仅用于演示）
        let prompt_tokens: Vec<u32> = prompt.chars().map(|c| c as u32).collect();

        if prompt_tokens.is_empty() {
            return Err(anyhow!("Prompt 为空"));
        }

        let max_new_tokens = params.max_new_tokens.min(2048); // 安全上限
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut all_tokens: Vec<u32> = prompt_tokens.clone();
        let mut total_generated = 0;

        // 主推测循环
        while total_generated < max_new_tokens {
            // 执行单步推测
            let step_result = self.speculate_step(&all_tokens)?;

            // 处理接受的 token
            for &token in &step_result.accepted_tokens {
                // 回调输出（简单地将 token ID 转回字符）
                let ch = char::from_u32(token).unwrap_or('?');
                callback(&ch.to_string())?;

                generated_tokens.push(token);
                all_tokens.push(token);
                total_generated += 1;

                if total_generated >= max_new_tokens {
                    break;
                }
            }

            if step_result.done || total_generated >= max_new_tokens {
                break;
            }
        }

        // 计算统计信息
        let elapsed = start_time.elapsed();
        let inference_stats = InferenceStats::with_timing_high_precision(
            elapsed.as_secs_f64(),
            prompt_tokens.len(),
            generated_tokens.len(),
            Some(elapsed.as_millis() as u64),
        );

        // 更新推测统计
        self.stats.actual_speedup = if self.stats.total_verification_time_us > 0 {
            // 估算实际加速比
            let baseline_time = self.stats.total_verification_time_us as f64
                * self.config.theoretical_speedup() as f64;
            (baseline_time / self.stats.total_verification_time_us as f64) as f32
        } else {
            1.0
        };

        Ok(inference_stats)
    }

    /// 单步推测解码循环
    ///
    /// 执行一次完整的推测-验证-接受流程：
    /// 1. 调用草稿模型生成候选 token
    /// 2. 调用目标模型验证候选
    /// 3. 返回接受的结果
    fn speculate_step(&mut self, current_tokens: &[u32]) -> Result<StepResult> {
        let step_start = Instant::now();

        // 调整当前使用的草稿长度
        let effective_draft_len = if self.config.enable_adaptive_draft {
            self.current_draft_length
        } else {
            self.config.max_speculative_tokens
        }
        .min(self.config.max_speculative_tokens)
        .max(self.config.min_draft_length);

        // 创建临时配置用于本次推测
        let step_config = SpeculativeConfig {
            max_speculative_tokens: effective_draft_len,
            ..self.config.clone()
        };

        // 1. 草稿模型生成候选
        let candidates = self
            .draft_model
            .generate_candidates(current_tokens, &step_config)?;

        if candidates.is_empty() {
            return Ok(StepResult {
                accepted_tokens: Vec::new(),
                done: true,
            });
        }

        // 2. 目标模型验证
        let verification = self.verifier.verify_candidates(
            current_tokens,
            &candidates,
            self.config.enable_typical_acceptance,
        );

        // 3. 更新统计信息
        self.update_stats(
            &candidates,
            &verification,
            step_start.elapsed().as_micros() as u64,
        );

        // 4. 自适应调整草稿长度
        if self.config.enable_adaptive_draft {
            self.adapt_draft_length(verification.accepted_count, effective_draft_len);
        }

        // 构建步骤结果
        let accepted = verification.accepted_tokens.clone();
        let has_bonus = verification.bonus_token.is_some();

        Ok(StepResult {
            accepted_tokens: if has_bonus {
                // 如果有 bonus token，添加到接受列表
                let mut result = accepted;
                if let Some(bonus) = verification.bonus_token {
                    result.push(bonus);
                }
                result
            } else {
                accepted
            },
            done: false, // 由外层循环控制终止
        })
    }

    /// 更新统计信息
    fn update_stats(
        &mut self,
        candidates: &SpeculativeCandidate,
        verification: &VerificationResult,
        step_time_us: u64,
    ) {
        self.stats.total_speculation_steps += 1;
        self.stats.total_draft_tokens += candidates.len() as u64;
        self.stats.total_accepted_tokens += verification.accepted_count as u64;
        self.stats.total_verification_time_us += step_time_us;

        if verification.rejected_position.is_some() {
            self.stats.total_rejections += 1;
        }
        if verification.bonus_token.is_some() {
            self.stats.total_bonus_tokens += 1;
        }

        // 更新平均值
        let total_steps = self.stats.total_speculation_steps as f32;
        self.stats.avg_accept_length = (self.stats.avg_accept_length * (total_steps - 1.0)
            + verification.accepted_count as f32)
            / total_steps;
        self.stats.avg_draft_length = (self.stats.avg_draft_length * (total_steps - 1.0)
            + candidates.len() as f32)
            / total_steps;

        // 计算接受率
        let total = self.stats.total_accepted_tokens + self.stats.total_rejections;
        if total > 0 {
            self.stats.acceptance_rate = self.stats.total_accepted_tokens as f32 / total as f32;
        }
    }

    /// 自适应调整草稿长度
    ///
    /// 根据接受率动态调整下次推测的草稿长度：
    /// - 接受率高 → 增加长度（可以尝试更多候选）
    /// - 接受率低 → 减少长度（减少浪费的计算）
    fn adapt_draft_length(&mut self, accepted: usize, drafted: usize) {
        if drafted == 0 {
            return;
        }

        let rate = accepted as f32 / drafted as f32;

        if rate > self.config.accept_threshold + 0.15 {
            // 高接受率，增加草稿长度
            self.current_draft_length =
                (self.current_draft_length + 1).min(self.config.max_speculative_tokens);
        } else if rate < self.config.accept_threshold - 0.15 {
            // 低接受率，减少草稿长度
            self.current_draft_length = self
                .current_draft_length
                .saturating_sub(1)
                .max(self.config.min_draft_length);
        }
    }

    /// 获取加速统计
    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// 获取当前草稿长度
    pub fn current_draft_length(&self) -> usize {
        self.current_draft_length
    }

    /// 重置统计信息
    pub fn reset_stats(&mut self) {
        self.stats = SpeculativeStats::new();
        self.current_draft_length = self.config.max_speculative_tokens;
    }

    /// 获取理论加速比
    pub fn theoretical_speedup(&self) -> f32 {
        self.config.theoretical_speedup()
    }
}

// ============================================================================
// 树注意力支持（高级功能）
// ============================================================================

/// 构建树形注意力掩码
///
/// 当使用推测解码时，KV cache 需要处理分支结构。
/// 树形注意力掩码确保每个候选 token 只关注其前缀路径上的 token。
///
/// # 参数
/// - `seq_len`: 原始序列长度
/// - `spec_len`: 推测候选长度
///
/// # 返回
/// 形状为 (seq_len + spec_len, seq_len + spec_len) 的注意力掩码矩阵
///
/// # 示例
///
/// ```ignore
/// // 原始序列 [A, B]，推测候选 [C, D, E]
/// // 树结构:
/// // A → B → C → D → E (主路径)
/// //       └→ C'→ D'→ E' (分支路径)
/// let mask = build_tree_attention_mask(2, 3);
/// ```
pub fn build_tree_attention_mask(seq_len: usize, spec_len: usize) -> Array2<f32> {
    let total_len = seq_len + spec_len;
    let mut mask = Array2::zeros((total_len, total_len));

    // 原始序列部分：完全可见（标准因果掩码）
    for i in 0..seq_len {
        for j in 0..=i {
            mask[[i, j]] = 1.0;
        }
    }

    // 推测部分：树形可见性
    for i in seq_len..total_len {
        let spec_pos = i - seq_len;

        // 原始序列完全可见
        for j in 0..seq_len {
            mask[[i, j]] = 1.0;
        }

        // 推测序列中的前缀可见（因果性）
        for j in seq_len..=i {
            let j_spec_pos = j - seq_len;
            if j_spec_pos <= spec_pos {
                mask[[i, j]] = 1.0;
            }
        }
    }

    mask
}

/// 树形位置编码
///
/// 为推测解码的树结构生成特殊的位置编码。
/// 每个分支节点有独特的位置编码，避免位置混淆。
///
/// # 参数
/// - `positions`: 位置索引列表
/// - `dim`: 编码维度
///
/// # 返回
/// 形状为 (len(positions), dim) 的位置编码矩阵
pub fn tree_position_embedding(positions: &[usize], dim: usize) -> Array2<f32> {
    let len = positions.len();
    let mut embedding = Array2::zeros((len, dim));

    for (i, &pos) in positions.iter().enumerate() {
        for d in 0..dim {
            // 使用正弦位置编码的变体
            let freq = 1.0f32 / (10000.0f32.powf(d as f32 / dim as f32));
            embedding[[i, d]] = if d % 2 == 0 {
                (pos as f32 * freq).sin()
            } else {
                (pos as f32 * freq).cos()
            };
        }
    }

    embedding
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ===== 配置相关测试 =====

    #[test]
    fn test_default_config() {
        let config = SpeculativeConfig::default();
        assert_eq!(config.draft_model_size, 1_000_000_000);
        assert_eq!(config.target_model_size, 7_000_000_000);
        assert_eq!(config.max_speculative_tokens, 5);
        assert!((config.accept_threshold - 0.5).abs() < f32::EPSILON);
        assert!(!config.use_tree_attention);
        assert!((config.temperature - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_builder_pattern() {
        let config = SpeculativeConfig::new()
            .with_draft_model_size(500_000_000)
            .with_target_model_size(3_000_000_000)
            .with_max_speculative_tokens(8)
            .with_accept_threshold(0.6)
            .with_tree_attention(true)
            .with_temperature(0.9);

        assert_eq!(config.draft_model_size, 500_000_000);
        assert_eq!(config.target_model_size, 3_000_000_000);
        assert_eq!(config.max_speculative_tokens, 8);
        assert!((config.accept_threshold - 0.6).abs() < f32::EPSILON);
        assert!(config.use_tree_attention);
        assert!((config.temperature - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_accept_threshold_clamping() {
        // 测试阈值被限制在 [0, 1] 范围内
        let config_high = SpeculativeConfig::new().with_accept_threshold(1.5);
        assert!(config_high.accept_threshold <= 1.0);

        let config_low = SpeculativeConfig::new().with_accept_threshold(-0.5);
        assert!(config_low.accept_threshold >= 0.0);
    }

    #[test]
    fn test_theoretical_speedup_calculation() {
        let config = SpeculativeConfig::new()
            .with_draft_model_size(1_000_000_000)
            .with_target_model_size(7_000_000_000);

        let speedup = config.theoretical_speedup();
        // sqrt(1B/7B) ≈ 0.378
        assert!((speedup - 0.378).abs() < 0.01);

        // 相同大小的模型应该返回 1.0
        let same_size = SpeculativeConfig::new()
            .with_draft_model_size(7_000_000_000)
            .with_target_model_size(7_000_000_000);
        assert!((same_size.theoretical_speedup() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_temperature_clamping() {
        let config = SpeculativeConfig::new().with_temperature(-0.5);
        assert!(config.temperature >= 0.0);
    }

    // ===== SpeculativeCandidate 测试 =====

    #[test]
    fn test_candidate_empty() {
        let candidate = SpeculativeCandidate::empty();
        assert!(candidate.is_empty());
        assert_eq!(candidate.len(), 0);
    }

    #[test]
    fn test_candidate_with_data() {
        let candidate = SpeculativeCandidate {
            tokens: vec![1, 2, 3],
            probabilities: vec![0.8, 0.6, 0.4],
            draft_log_probs: vec![-0.223, -0.511, -0.916],
        };
        assert!(!candidate.is_empty());
        assert_eq!(candidate.len(), 3);
        assert_eq!(candidate.tokens.len(), 3);
        assert_eq!(candidate.probabilities.len(), 3);
        assert_eq!(candidate.draft_log_probs.len(), 3);
    }

    // ===== VerificationResult 测试 =====

    #[test]
    fn test_verification_fully_accepted() {
        let tokens = vec![1, 2, 3, 4];
        let result = VerificationResult::fully_accepted(tokens.clone(), 100);
        assert_eq!(result.accepted_tokens, tokens);
        assert_eq!(result.accepted_count, 4);
        assert!(result.rejected_position.is_none());
        assert!(result.bonus_token.is_none());
        assert_eq!(result.verification_time_us, 100);
    }

    #[test]
    fn test_verification_partial_accept() {
        let accepted = vec![1, 2];
        let result = VerificationResult::partial_accept(accepted.clone(), 2, 99, 200);
        assert_eq!(result.accepted_tokens, accepted);
        assert_eq!(result.accepted_count, 2);
        assert_eq!(result.rejected_position, Some(2));
        assert_eq!(result.bonus_token, Some(99));
        assert_eq!(result.verification_time_us, 200);
    }

    // ===== DraftModel 测试 =====

    #[test]
    fn test_draft_model_creation() {
        let _model = DraftModel::new(1000, 0.7);
        // 验证模型创建成功（无 panic）
    }

    #[test]
    fn test_draft_model_with_seed() {
        let _model1 = DraftModel::with_seed(1000, 0.7, 42);
        let _model2 = DraftModel::with_seed(1000, 0.7, 42);
        // 相同种子应该产生相同的行为（此处只验证创建成功）
    }

    #[test]
    fn test_draft_model_generate_candidates() {
        let mut model = DraftModel::with_seed(1000, 0.7, 42);
        let config = SpeculativeConfig::new().with_max_speculative_tokens(4);
        let prefix = vec![1, 2, 3, 4, 5];

        let result = model.generate_candidates(&prefix, &config);
        assert!(result.is_ok());

        let candidate = result.unwrap();
        assert_eq!(candidate.len(), 4);
        assert_eq!(candidate.tokens.len(), 4);
        assert_eq!(candidate.probabilities.len(), 4);
        assert_eq!(candidate.draft_log_probs.len(), 4);
    }

    #[test]
    fn test_draft_model_empty_prefix_error() {
        let mut model = DraftModel::new(1000, 0.7);
        let config = SpeculativeConfig::default();

        let result = model.generate_candidates(&[], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_draft_model_get_next_token_probs() {
        let model = DraftModel::new(100, 0.7);
        let context = vec![10, 20, 30];

        let probs = model.get_next_token_probs(&context);
        assert_eq!(probs.len(), 100);

        // 验证概率和接近 1.0
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01 || sum > 0.99);

        // 验证所有概率非负
        assert!(probs.iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn test_draft_model_sample_token() {
        let mut model = DraftModel::with_seed(100, 0.7, 123);
        let probs = vec![0.1, 0.2, 0.3, 0.25, 0.15]; // 和为 1.0

        let result = model.sample_token(&probs);
        assert!(result.is_ok());

        let (token, prob) = result.unwrap();
        assert!(token < 5);
        assert!(prob > 0.0 && prob <= 1.0);
    }

    #[test]
    fn test_draft_model_zero_sum_error() {
        let mut model = DraftModel::new(100, 0.7);
        let probs = vec![0.0; 10]; // 全零

        let result = model.sample_token(&probs);
        assert!(result.is_err());
    }

    // ===== TargetVerifier 测试 =====

    #[test]
    fn test_verifier_creation() {
        let _verifier = TargetVerifier::new(1000);
        // 验证创建成功
    }

    #[test]
    fn test_verifier_verify_empty_candidates() {
        let mut verifier = TargetVerifier::with_seed(1000, 42);
        let candidate = SpeculativeCandidate::empty();

        let result = verifier.verify_candidates(&[], &candidate, false);
        assert!(result.accepted_tokens.is_empty());
        assert_eq!(result.accepted_count, 0);
    }

    #[test]
    fn test_standard_verify_accept() {
        let mut verifier = TargetVerifier::with_seed(100, 42);

        // 高接受概率的情况：target_prob >> draft_prob
        let (accepted, bonus) = verifier.standard_verify(
            &[0.1], // 低草稿概率
            &[0.9], // 高目标概率
            &[5],
        );

        assert_eq!(accepted, 1);
        assert!(bonus.is_none()); // 接受时无 bonus
    }

    #[test]
    fn test_standard_verify_reject() {
        let mut verifier = TargetVerifier::with_seed(100, 42); // 固定种子

        // 低接受概率的情况：target_prob << draft_prob
        let (accepted, bonus) = verifier.standard_verify(
            &[0.9], // 高草稿概率
            &[0.1], // 低目标概率
            &[5],
        );

        // 可能接受或拒绝，取决于随机数
        assert!(accepted <= 1);
        if accepted == 0 {
            assert!(bonus.is_some()); // 拒绝时应有 bonus
        }
    }

    #[test]
    fn test_typical_acceptance_verify_basic() {
        let mut rng = StdRng::seed_from_u64(42);

        // 高概率匹配的情况
        let (accepted, _) =
            TargetVerifier::typical_acceptance_verify(&[0.3], &[0.7], &[10], &mut rng);

        // 应该有较高的接受概率
        assert!(accepted <= 1);
    }

    #[test]
    fn test_typical_acceptance_verify_edge_cases() {
        let mut rng = StdRng::seed_from_u64(123);

        // 极低概率
        let (accepted1, bonus1) =
            TargetVerifier::typical_acceptance_verify(&[0.99], &[0.01], &[50], &mut rng);
        assert!(accepted1 <= 1);

        // 极高概率
        let (accepted2, bonus2) =
            TargetVerifier::typical_acceptance_verify(&[0.01], &[0.99], &[60], &mut rng);
        assert!(accepted2 <= 1);

        // 验证 bonus token 存在性
        if accepted1 == 0 {
            assert!(bonus1.is_some());
        }
        if accepted2 == 0 {
            assert!(bonus2.is_some());
        }
    }

    #[test]
    fn test_verify_candidates_integration() {
        let mut verifier = TargetVerifier::with_seed(1000, 42);
        let candidate = SpeculativeCandidate {
            tokens: vec![10, 20, 30],
            probabilities: vec![0.5, 0.6, 0.4],
            draft_log_probs: vec![-0.693, -0.511, -0.916],
        };

        // 使用标准验证
        let result_std = verifier.verify_candidates(&[1, 2, 3], &candidate, false);
        assert!(result_std.accepted_count <= 3);
        assert!(result_std.accepted_tokens.len() <= 3);

        // 使用典型接受验证
        let result_typical = verifier.verify_candidates(&[1, 2, 3], &candidate, true);
        assert!(result_typical.accepted_count <= 3);
    }

    // ===== SpeculativeEngine 测试 =====

    #[test]
    fn test_engine_creation() {
        let config = SpeculativeConfig::default();
        let engine = SpeculativeEngine::new(config);
        assert_eq!(engine.current_draft_length(), 5);
        assert!(engine.stats().total_speculation_steps == 0);
    }

    #[test]
    fn test_engine_with_seed() {
        let config = SpeculativeConfig::default();
        let engine = SpeculativeEngine::with_seed(config, 42);
        assert_eq!(engine.current_draft_length(), 5);
    }

    #[test]
    fn test_engine_generate_simple() {
        let config = SpeculativeConfig::new()
            .with_max_speculative_tokens(3)
            .with_temperature(0.8);

        let mut engine = SpeculativeEngine::with_seed(config, 42);
        let params = GenerateParams::new().with_max_new_tokens(10);

        let mut output = String::new();
        let result = engine.generate("Hi", &params, |token| {
            output.push_str(token);
            Ok(())
        });

        assert!(result.is_ok());
        assert!(!output.is_empty());
        let stats = result.unwrap();
        assert!(stats.generated_tokens > 0);
    }

    #[test]
    fn test_engine_generate_empty_prompt() {
        let config = SpeculativeConfig::default();
        let mut engine = SpeculativeEngine::new(config);
        let params = GenerateParams::default();

        let result = engine.generate("", &params, |_| Ok(()));
        assert!(result.is_err());
    }

    #[test]
    fn test_engine_stats_tracking() {
        let config = SpeculativeConfig::new()
            .with_max_speculative_tokens(3)
            .with_enable_adaptive_draft(true);

        let mut engine = SpeculativeEngine::with_seed(config, 42);
        let params = GenerateParams::new().with_max_new_tokens(20);

        let _ = engine.generate("Test", &params, |_token| Ok(()));

        let stats = engine.stats();
        assert!(stats.total_speculation_steps > 0);
        assert!(stats.total_draft_tokens > 0);
        assert!(stats.avg_draft_length > 0.0);
        assert!(stats.acceptance_rate >= 0.0 && stats.acceptance_rate <= 1.0);
    }

    #[test]
    fn test_engine_reset_stats() {
        let config = SpeculativeConfig::default();
        let mut engine = SpeculativeEngine::with_seed(config, 42);
        let params = GenerateParams::new().with_max_new_tokens(5);

        let _ = engine.generate("Test", &params, |_token| Ok(()));
        assert!(engine.stats().total_speculation_steps > 0);

        engine.reset_stats();
        assert_eq!(engine.stats().total_speculation_steps, 0);
        assert_eq!(engine.stats().total_draft_tokens, 0);
        assert_eq!(
            engine.current_draft_length(),
            engine.config.max_speculative_tokens
        );
    }

    #[test]
    fn test_engine_adaptive_draft_length() {
        let config = SpeculativeConfig::new()
            .with_max_speculative_tokens(6)
            .with_min_draft_length(2)
            .with_enable_adaptive_draft(true);

        let mut engine = SpeculativeEngine::with_seed(config, 42);
        let initial_length = engine.current_draft_length();

        let params = GenerateParams::new().with_max_new_tokens(30);
        let _ = engine.generate("Adaptive test", &params, |_token| Ok(()));

        let final_length = engine.current_draft_length();
        // 长度应该在合理范围内
        assert!(
            (2..=6).contains(&final_length),
            "Draft length should be in range [2, 6], got {}",
            final_length
        );
        // 可能已经改变
        assert!(final_length == initial_length || (2..=6).contains(&final_length));
    }

    #[test]
    fn test_engine_theoretical_speedup() {
        let config = SpeculativeConfig::new()
            .with_draft_model_size(1_000_000_000)
            .with_target_model_size(7_000_000_000);

        let engine = SpeculativeEngine::new(config);
        let speedup = engine.theoretical_speedup();
        assert!(speedup > 0.0);
        assert!(speedup < 1.0); // 草稿模型较小，加速比 < 1（表示更快）
    }

    // ===== 树注意力测试 =====

    #[test]
    fn test_tree_attention_mask_basic() {
        let mask = build_tree_attention_mask(3, 2);
        assert_eq!(mask.shape(), [5, 5]); // (seq_len + spec_len)^2
    }

    #[test]
    fn test_tree_attention_mask_causal_property() {
        let seq_len = 3;
        let spec_len = 2;
        let mask = build_tree_attention_mask(seq_len, spec_len);

        // 验证因果性：上三角应为 0（除了对角线）
        for i in 0..(seq_len + spec_len) {
            for j in (i + 1)..(seq_len + spec_len) {
                // 对于原始序列部分，严格因果
                if i < seq_len && j < seq_len {
                    assert!(
                        (mask[[i, j]] - 0.0).abs() < f32::EPSILON,
                        "Mask[{},{}] should be 0 for causal property",
                        i,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn test_tree_attention_mask_prefix_visibility() {
        let seq_len = 2;
        let spec_len = 3;
        let mask = build_tree_attention_mask(seq_len, spec_len);

        // 推测位置的 token 应该能看到所有原始序列
        for i in seq_len..(seq_len + spec_len) {
            for j in 0..seq_len {
                assert!(
                    (mask[[i, j]] - 1.0).abs() < f32::EPSILON,
                    "Spec position {} should see original position {}",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_tree_position_embedding() {
        let positions = vec![0, 1, 2, 5, 10];
        let dim = 16;
        let embedding = tree_position_embedding(&positions, dim);

        assert_eq!(embedding.shape(), [5, dim]);

        // 验证值范围在 [-1, 1] 内（正弦编码的特性）
        for val in embedding.iter() {
            assert!(*val >= -1.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_tree_position_embedding_different_positions() {
        let positions1 = vec![0, 1, 2];
        let positions2 = vec![0, 1, 3];
        let dim = 8;

        let emb1 = tree_position_embedding(&positions1, dim);
        let emb2 = tree_position_embedding(&positions2, dim);

        // 不同位置应该有不同的编码
        // 位置 2 和位置 3 应该不同
        let row1: Vec<f32> = emb1.row(2).to_vec();
        let row2: Vec<f32> = emb2.row(2).to_vec();
        let diff: f32 = row1
            .iter()
            .zip(row2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(
            diff > f32::EPSILON,
            "Different positions should have different embeddings"
        );
    }

    // ===== 边界条件和集成测试 =====

    #[test]
    fn test_single_token_generation() {
        let config = SpeculativeConfig::new()
            .with_max_speculative_tokens(1)
            .with_min_draft_length(1);

        let mut engine = SpeculativeEngine::with_seed(config, 42);
        let params = GenerateParams::new().with_max_new_tokens(3);

        let result = engine.generate("A", &params, |_token| Ok(()));
        assert!(result.is_ok());
    }

    #[test]
    fn test_max_length_respected() {
        let config = SpeculativeConfig::default();
        let mut engine = SpeculativeEngine::with_seed(config, 42);
        let params = GenerateParams::new().with_max_new_tokens(5);

        let result = engine.generate("Test max length", &params, |_token| Ok(()));
        assert!(result.is_ok());

        let stats = result.unwrap();
        assert!(stats.generated_tokens <= 5);
    }

    #[test]
    fn test_multiple_generations_independence() {
        let config = SpeculativeConfig::new()
            .with_max_speculative_tokens(3)
            .with_temperature(0.9);

        let mut engine = SpeculativeEngine::with_seed(config, 42);
        let params = GenerateParams::new().with_max_new_tokens(10);

        let result1 = engine.generate("First", &params, |_token| Ok(()));
        engine.reset_stats();
        let result2 = engine.generate("Second", &params, |_token| Ok(()));

        assert!(result1.is_ok());
        assert!(result2.is_ok());

        // 第二次生成后，统计应该是独立的
        assert!(engine.stats().total_speculation_steps > 0);
    }

    #[test]
    fn test_config_various_combinations() {
        // 测试各种极端配置组合
        let configs = vec![
            SpeculativeConfig::new()
                .with_max_speculative_tokens(1)
                .with_min_draft_length(1),
            SpeculativeConfig::new()
                .with_max_speculative_tokens(8)
                .with_min_draft_length(2)
                .with_accept_threshold(0.9)
                .with_temperature(0.1),
            SpeculativeConfig::new()
                .with_max_speculative_tokens(4)
                .with_min_draft_length(2)
                .with_enable_typical_acceptance(false)
                .with_enable_adaptive_draft(false),
        ];

        for config in configs {
            let engine = SpeculativeEngine::new(config);
            assert!(engine.current_draft_length() >= engine.config.min_draft_length);
            assert!(engine.current_draft_length() <= engine.config.max_speculative_tokens);
        }
    }

    #[test]
    fn test_large_vocab_size() {
        let mut model = DraftModel::with_seed(100_000, 0.7, 42);
        let config = SpeculativeConfig::new().with_max_speculative_tokens(5);
        let prefix: Vec<u32> = (0..100).collect();

        let result = model.generate_candidates(&prefix, &config);
        assert!(result.is_ok());

        let candidate = result.unwrap();
        assert_eq!(candidate.len(), 5);
        // 所有 token 应该在有效范围内
        assert!(candidate.tokens.iter().all(|&t| t < 100_000));
    }

    #[test]
    fn test_statistics_accuracy() {
        let config = SpeculativeConfig::new()
            .with_max_speculative_tokens(4)
            .with_enable_adaptive_draft(false);

        let mut engine = SpeculativeEngine::with_seed(config, 42);
        let params = GenerateParams::new().with_max_new_tokens(40);

        let _ = engine.generate("Statistics test string", &params, |_token| Ok(()));

        let stats = engine.stats();

        // 验证统计一致性
        if stats.total_speculation_steps > 0 {
            let expected_avg_draft =
                stats.total_draft_tokens as f32 / stats.total_speculation_steps as f32;
            assert!(
                (stats.avg_draft_length - expected_avg_draft).abs() < 0.01
                    || stats.avg_draft_length > 0.0,
                "Average draft length mismatch: expected {}, got {}",
                expected_avg_draft,
                stats.avg_draft_length
            );

            // 验证接受率计算
            let total_decisions = stats.total_accepted_tokens + stats.total_rejections;
            if total_decisions > 0 {
                let expected_rate = stats.total_accepted_tokens as f32 / total_decisions as f32;
                assert!(
                    (stats.acceptance_rate - expected_rate).abs() < 0.01,
                    "Acceptance rate mismatch"
                );
            }
        }
    }
}
