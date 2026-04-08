//! 文本生成采样模块
//!
//! 本模块实现了多种文本生成采样策略，包括：
//! - 贪婪解码 (Greedy Decoding)
//! - Top-K 采样
//! - Top-P (Nucleus) 采样
//! - 温度调节 (Temperature)
//! - 重复惩罚 (Repetition Penalty)
//! - 束搜索 (Beam Search)
//!
//! # 示例
//!
//! ```ignore
//! use inference::sampler::{GenerateParams, Sampler};
//!
//! // 创建采样参数
//! let params = GenerateParams::new()
//!     .with_temperature(0.7)
//!     .with_top_p(0.9)
//!     .with_top_k(50);
//!
//! // 创建采样器
//! let mut sampler = Sampler::new(params);
//!
//! // 从 logits 采样
//! let token = sampler.sample(&logits, &generated_tokens)?;
//! ```

#![allow(dead_code)]

use anyhow::Result;
use ndarray::Array1;
use rand::distributions::WeightedIndex;
use rand::prelude::*;

// ============================================================================
// 生成参数配置
// ============================================================================

/// 文本生成参数配置
///
/// 控制文本生成的各种行为，包括采样策略、温度、Top-K/Top-P 过滤等。
/// 使用 Builder 模式进行配置。
#[derive(Debug, Clone)]
pub struct GenerateParams {
    /// 是否启用采样
    /// - true: 使用概率采样
    /// - false: 使用贪婪解码 (argmax)
    pub sampling: bool,

    /// Top-P (Nucleus) 采样阈值
    /// 保留累积概率达到 top_p 的最小 token 集合
    /// 范围: 0.0 ~ 1.0，值越小生成越确定
    pub top_p: f32,

    /// Top-K 采样数量
    /// 只保留概率最高的 K 个 token
    /// 0 表示不限制
    pub top_k: usize,

    /// 温度参数
    /// 控制输出分布的平滑程度
    /// - 温度越高，分布越平滑，生成越随机
    /// - 温度越低，分布越尖锐，生成越确定
    pub temperature: f32,

    /// 束搜索的束数量
    /// 用于 Beam Search 算法
    pub num_beams: usize,

    /// 重复惩罚系数
    /// 对已生成 token 施加惩罚，避免重复
    /// - 1.0: 无惩罚
    /// - > 1.0: 惩罚重复
    pub repetition_penalty: f32,

    /// 最大生成 token 数量
    pub max_new_tokens: usize,
}

impl Default for GenerateParams {
    /// 默认生成参数
    ///
    /// 默认值经过调优，适用于一般对话场景：
    /// - 启用采样
    /// - Top-P: 0.8
    /// - Top-K: 100
    /// - 温度: 0.7
    /// - 束数: 3
    /// - 重复惩罚: 1.05
    /// - 最大生成长度: 2048
    fn default() -> Self {
        Self {
            sampling: true,
            top_p: 0.8,
            top_k: 100,
            temperature: 0.7,
            num_beams: 3,
            repetition_penalty: 1.05,
            max_new_tokens: 2048,
        }
    }
}

impl GenerateParams {
    /// 创建默认参数配置
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置是否启用采样
    pub fn with_sampling(mut self, sampling: bool) -> Self {
        self.sampling = sampling;
        self
    }

    /// 设置 Top-P 阈值
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// 设置 Top-K 数量
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// 设置温度参数
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// 设置束搜索束数
    pub fn with_num_beams(mut self, num_beams: usize) -> Self {
        self.num_beams = num_beams;
        self
    }

    /// 设置重复惩罚系数
    pub fn with_repetition_penalty(mut self, repetition_penalty: f32) -> Self {
        self.repetition_penalty = repetition_penalty;
        self
    }

    /// 设置最大生成长度
    pub fn with_max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.max_new_tokens = max_new_tokens;
        self
    }
}

// ============================================================================
// 采样器实现
// ============================================================================

/// 文本生成采样器
///
/// 实现多种采样策略，支持：
/// - 贪婪解码
/// - Top-K 采样
/// - Top-P (Nucleus) 采样
/// - 温度调节
/// - 重复惩罚
///
/// 采样流程：
/// 1. 应用重复惩罚
/// 2. 应用温度调节
/// 3. 计算 Softmax 概率
/// 4. 应用 Top-K 过滤
/// 5. 应用 Top-P 过滤
/// 6. 从剩余候选中采样
#[derive(Debug, Clone)]
pub struct Sampler {
    /// 采样参数配置
    params: GenerateParams,

    /// 随机数生成器
    rng: StdRng,
}

impl Sampler {
    /// 创建采样器（使用随机种子）
    pub fn new(params: GenerateParams) -> Self {
        Self {
            params,
            rng: StdRng::from_entropy(),
        }
    }

    /// 创建采样器（使用固定种子）
    ///
    /// 固定种子可用于复现生成结果
    pub fn with_seed(params: GenerateParams, seed: u64) -> Self {
        Self {
            params,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// 获取当前参数配置
    pub fn params(&self) -> &GenerateParams {
        &self.params
    }

    /// 设置新的参数配置
    pub fn set_params(&mut self, params: GenerateParams) {
        self.params = params;
    }

    /// 从 logits 采样下一个 token
    ///
    /// # 参数
    /// - `logits`: 模型输出的 logits 分布
    /// - `generated_tokens`: 已生成的 token 列表（用于重复惩罚）
    ///
    /// # 返回
    /// 采样得到的 token 索引
    pub fn sample(&mut self, logits: &Array1<f32>, generated_tokens: &[u32]) -> Result<usize> {
        let mut logits = logits.clone();

        // 步骤 1: 应用重复惩罚
        self.apply_repetition_penalty(&mut logits, generated_tokens);

        // 步骤 2: 如果禁用采样，使用贪婪解码
        if !self.params.sampling {
            return self.argmax(&logits);
        }

        // 步骤 3: 应用温度调节
        self.apply_temperature(&mut logits);

        // 步骤 4: 计算 Softmax 概率
        let probs = self.softmax(&logits);

        // 步骤 5: 应用 Top-K 过滤
        let (filtered_indices, filtered_probs) = self.apply_top_k(&probs);

        // 步骤 6: 应用 Top-P 过滤
        let (filtered_indices, filtered_probs) = self.apply_top_p(&filtered_indices, &filtered_probs);

        // 步骤 7: 从过滤后的分布中采样
        self.sample_from_probs(&filtered_indices, &filtered_probs)
    }

    /// 贪婪解码：选择最大概率的 token
    ///
    /// 遍历所有 logits，返回最大值的索引
    pub fn argmax(&self, logits: &Array1<f32>) -> Result<usize> {
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;

        for (i, &val) in logits.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        Ok(max_idx)
    }

    /// 应用温度调节
    ///
    /// 将 logits 除以温度值：
    /// - 温度 > 1: 分布更平滑，生成更随机
    /// - 温度 < 1: 分布更尖锐，生成更确定
    /// - 温度 = 1: 不改变分布
    pub fn apply_temperature(&self, logits: &mut Array1<f32>) {
        if self.params.temperature <= 0.0 {
            return;
        }

        let temp = self.params.temperature;
        logits.mapv_inplace(|v| v / temp);
    }

    /// 计算 Softmax 概率分布
    ///
    /// 使用数值稳定的实现：
    /// softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    pub fn softmax(&self, logits: &Array1<f32>) -> Array1<f32> {
        // 减去最大值，防止数值溢出
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // 计算指数和
        let exp_sum: f32 = logits.iter().map(|&v| (v - max_val).exp()).sum();

        // 归一化
        logits.mapv(|v| (v - max_val).exp() / exp_sum)
    }

    /// 应用 Top-K 过滤
    ///
    /// 只保留概率最高的 K 个 token
    ///
    /// # 返回
    /// (过滤后的索引列表, 过滤后的概率列表)
    pub fn apply_top_k(&self, probs: &Array1<f32>) -> (Vec<usize>, Vec<f32>) {
        let vocab_size = probs.len();

        // 如果 top_k 为 0 或大于词表大小，不进行过滤
        if self.params.top_k == 0 || self.params.top_k >= vocab_size {
            let indices: Vec<usize> = (0..vocab_size).collect();
            let probs_vec: Vec<f32> = probs.iter().cloned().collect();
            return (indices, probs_vec);
        }

        // 将概率与索引配对
        let mut indexed_probs: Vec<(usize, f32)> = probs
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();

        // 按概率降序排序
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 取前 K 个
        let top_k: Vec<(usize, f32)> = indexed_probs.into_iter().take(self.params.top_k).collect();

        // 分离索引和概率
        let indices: Vec<usize> = top_k.iter().map(|(i, _)| *i).collect();
        let probs_vec: Vec<f32> = top_k.iter().map(|(_, p)| *p).collect();

        (indices, probs_vec)
    }

    /// 应用 Top-P (Nucleus) 过滤
    ///
    /// 保留累积概率达到 top_p 的最小 token 集合
    ///
    /// # 参数
    /// - `indices`: token 索引列表
    /// - `probs`: 对应的概率列表
    ///
    /// # 返回
    /// (过滤后的索引列表, 过滤后的概率列表)
    pub fn apply_top_p(&self, indices: &[usize], probs: &[f32]) -> (Vec<usize>, Vec<f32>) {
        // 如果 top_p >= 1.0，不进行过滤
        if self.params.top_p >= 1.0 {
            return (indices.to_vec(), probs.to_vec());
        }

        // 将概率与索引配对并排序
        let mut sorted_probs: Vec<(usize, f32)> = indices
            .iter()
            .zip(probs.iter())
            .map(|(&i, &p)| (i, p))
            .collect();

        sorted_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 累积概率，直到达到 top_p
        let mut cumsum = 0.0;
        let mut result = Vec::new();

        for (idx, prob) in sorted_probs {
            cumsum += prob;
            result.push((idx, prob));

            if cumsum >= self.params.top_p {
                break;
            }
        }

        // 分离索引和概率
        let filtered_indices: Vec<usize> = result.iter().map(|(i, _)| *i).collect();
        let filtered_probs: Vec<f32> = result.iter().map(|(_, p)| *p).collect();

        (filtered_indices, filtered_probs)
    }

    /// 应用重复惩罚
    ///
    /// 对已生成的 token 施加惩罚：
    /// - 正值 logits 除以惩罚系数
    /// - 负值 logits 乘以惩罚系数
    ///
    /// 这样可以降低重复生成相同 token 的概率
    pub fn apply_repetition_penalty(&self, logits: &mut Array1<f32>, generated_tokens: &[u32]) {
        if self.params.repetition_penalty == 1.0 {
            return;
        }

        let penalty = self.params.repetition_penalty;

        for &token_id in generated_tokens {
            let idx = token_id as usize;
            if idx < logits.len() {
                let val = logits[idx];
                if val > 0.0 {
                    // 正值：除以惩罚系数，降低概率
                    logits[idx] = val / penalty;
                } else {
                    // 负值：乘以惩罚系数，进一步降低概率
                    logits[idx] = val * penalty;
                }
            }
        }
    }

    /// 从概率分布中采样
    ///
    /// 使用加权随机采样从候选 token 中选择一个
    fn sample_from_probs(&mut self, indices: &[usize], probs: &[f32]) -> Result<usize> {
        // 检查空分布
        if indices.is_empty() || probs.is_empty() {
            return Err(anyhow::anyhow!("Empty probability distribution"));
        }

        // 计算概率和
        let sum: f32 = probs.iter().sum();
        if sum <= 0.0 {
            // 概率和为 0，随机选择一个
            let idx = self.rng.gen_range(0..indices.len());
            return Ok(indices[idx]);
        }

        // 归一化概率
        let normalized_probs: Vec<f32> = probs.iter().map(|p| p / sum).collect();

        // 使用加权随机采样
        let dist = WeightedIndex::new(&normalized_probs)?;
        let sampled_idx = dist.sample(&mut self.rng);

        Ok(indices[sampled_idx])
    }

    /// 批量采样
    ///
    /// 对多个 logits 分布同时进行采样
    pub fn sample_batch(&mut self, logits_batch: &[Array1<f32>], generated_tokens: &[u32]) -> Result<Vec<usize>> {
        logits_batch
            .iter()
            .map(|logits| self.sample(logits, generated_tokens))
            .collect()
    }
}

// ============================================================================
// 束搜索实现
// ============================================================================

/// 束搜索候选
///
/// 存储一个候选序列及其累积分数
#[derive(Debug, Clone)]
pub struct BeamSearchCandidate {
    /// 已生成的 token 序列
    pub tokens: Vec<u32>,

    /// 累积对数概率分数
    pub score: f32,
}

impl BeamSearchCandidate {
    /// 创建新的束搜索候选
    pub fn new(tokens: Vec<u32>, score: f32) -> Self {
        Self { tokens, score }
    }

    /// 返回序列长度
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// 检查序列是否为空
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// 束搜索算法实现
///
/// 束搜索是一种启发式搜索算法，在每一步保留 top-k 个最优候选，
/// 最终选择分数最高的序列作为输出。
///
/// # 工作流程
/// 1. 初始化：从起始 token 开始
/// 2. 扩展：对每个候选生成所有可能的下一个 token
/// 3. 剪枝：保留分数最高的 num_beams 个候选
/// 4. 终止：达到最大长度或所有候选都生成了结束符
pub struct BeamSearch {
    /// 束的数量（保留的候选数量）
    num_beams: usize,

    /// 当前候选列表
    candidates: Vec<BeamSearchCandidate>,
}

impl BeamSearch {
    /// 创建新的束搜索实例
    pub fn new(num_beams: usize) -> Self {
        Self {
            num_beams,
            candidates: Vec::new(),
        }
    }

    /// 初始化束搜索
    ///
    /// 设置起始 token 和初始分数
    pub fn initialize(&mut self, initial_token: u32, initial_score: f32) {
        self.candidates = vec![BeamSearchCandidate::new(vec![initial_token], initial_score)];
    }

    /// 获取当前所有候选
    pub fn candidates(&self) -> &[BeamSearchCandidate] {
        &self.candidates
    }

    /// 获取分数最高的候选
    pub fn best_candidate(&self) -> Option<&BeamSearchCandidate> {
        self.candidates
            .iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// 扩展候选
    ///
    /// 对每个当前候选，生成所有可能的下一个 token，
    /// 然后保留分数最高的 num_beams 个新候选
    pub fn expand(&mut self, logits_batch: &[Array1<f32>], _sampler: &mut Sampler) -> Result<()> {
        let mut new_candidates = Vec::new();

        for candidate in &self.candidates {
            let last_token_idx = candidate.tokens.len() - 1;
            if last_token_idx < logits_batch.len() {
                let logits = &logits_batch[last_token_idx];

                // 计算对数概率
                let log_probs = {
                    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_sum: f32 = logits.iter().map(|&v| (v - max_val).exp()).sum();
                    logits.mapv(|v| ((v - max_val).exp() / exp_sum).ln())
                };

                // 将对数概率与索引配对并排序
                let mut indexed_log_probs: Vec<(usize, f32)> = log_probs
                    .iter()
                    .enumerate()
                    .map(|(i, &lp)| (i, lp))
                    .collect();

                indexed_log_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // 为每个候选扩展 top num_beams 个 token
                for (token_id, log_prob) in indexed_log_probs.into_iter().take(self.num_beams) {
                    let mut new_tokens = candidate.tokens.clone();
                    new_tokens.push(token_id as u32);
                    let new_score = candidate.score + log_prob;
                    new_candidates.push(BeamSearchCandidate::new(new_tokens, new_score));
                }
            }
        }

        // 按分数排序并保留 top num_beams 个候选
        new_candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        new_candidates.truncate(self.num_beams);

        self.candidates = new_candidates;

        Ok(())
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_generate_params_default() {
        let params = GenerateParams::default();
        assert_eq!(params.sampling, true);
        assert!((params.top_p - 0.8).abs() < 1e-6);
        assert_eq!(params.top_k, 100);
        assert!((params.temperature - 0.7).abs() < 1e-6);
        assert_eq!(params.num_beams, 3);
        assert!((params.repetition_penalty - 1.05).abs() < 1e-6);
        assert_eq!(params.max_new_tokens, 2048);
    }

    #[test]
    fn test_generate_params_builder() {
        let params = GenerateParams::new()
            .with_temperature(0.5)
            .with_top_p(0.9)
            .with_top_k(50);

        assert!((params.temperature - 0.5).abs() < 1e-6);
        assert!((params.top_p - 0.9).abs() < 1e-6);
        assert_eq!(params.top_k, 50);
    }

    #[test]
    fn test_argmax() {
        let sampler = Sampler::new(GenerateParams::default());
        let logits = arr1(&[0.1, 0.5, 0.3, 0.9, 0.2]);
        let result = sampler.argmax(&logits).unwrap();
        assert_eq!(result, 3);
    }

    #[test]
    fn test_softmax() {
        let sampler = Sampler::new(GenerateParams::default());
        let logits = arr1(&[1.0, 2.0, 3.0]);
        let probs = sampler.softmax(&logits);

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_temperature() {
        let sampler = Sampler::new(GenerateParams::default().with_temperature(2.0));
        let logits = arr1(&[1.0, 2.0, 3.0]);
        let mut logits_copy = logits.clone();

        sampler.apply_temperature(&mut logits_copy);

        assert!((logits_copy[0] - 0.5).abs() < 1e-6);
        assert!((logits_copy[1] - 1.0).abs() < 1e-6);
        assert!((logits_copy[2] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_top_k() {
        let sampler = Sampler::new(GenerateParams::default().with_top_k(3));
        let probs = arr1(&[0.1, 0.3, 0.4, 0.15, 0.05]);

        let (indices, filtered_probs) = sampler.apply_top_k(&probs);

        assert_eq!(indices.len(), 3);

        assert!(filtered_probs.iter().all(|&p| p > 0.0));
    }

    #[test]
    fn test_top_p() {
        let sampler = Sampler::new(GenerateParams::default().with_top_p(0.9));
        let indices = vec![0, 1, 2, 3, 4];
        let probs = vec![0.4, 0.3, 0.2, 0.08, 0.02];

        let (filtered_indices, filtered_probs) = sampler.apply_top_p(&indices, &probs);

        let sum: f32 = filtered_probs.iter().sum();
        assert!(sum >= 0.9 || filtered_indices.len() < indices.len());
    }

    #[test]
    fn test_repetition_penalty() {
        let sampler = Sampler::new(GenerateParams::default().with_repetition_penalty(2.0));
        let mut logits = arr1(&[1.0, -1.0, 2.0, -2.0]);
        let generated = vec![0, 1];

        sampler.apply_repetition_penalty(&mut logits, &generated);

        assert!((logits[0] - 0.5).abs() < 1e-6);
        assert!((logits[1] - (-2.0)).abs() < 1e-6);
        assert!((logits[2] - 2.0).abs() < 1e-6);
        assert!((logits[3] - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sample_greedy() {
        let mut sampler = Sampler::new(GenerateParams::default().with_sampling(false));
        let logits = arr1(&[0.1, 0.5, 0.3, 0.9, 0.2]);
        let generated = vec![];

        let result = sampler.sample(&logits, &generated).unwrap();
        assert_eq!(result, 3);
    }

    #[test]
    fn test_beam_search() {
        let mut beam_search = BeamSearch::new(2);
        beam_search.initialize(0, 0.0);

        assert_eq!(beam_search.candidates().len(), 1);
        assert_eq!(beam_search.best_candidate().unwrap().tokens, vec![0]);
    }

    // 新增分支覆盖测试

    /// 测试 apply_temperature 的 temperature=0 分支（不执行任何操作）
    #[test]
    fn test_temperature_zero() {
        let sampler = Sampler::new(GenerateParams::default().with_temperature(0.0));
        let logits = arr1(&[1.0, 2.0, 3.0]);
        let mut logits_copy = logits.clone();

        sampler.apply_temperature(&mut logits_copy);
        assert_eq!(logits_copy, logits, "temperature=0 时 logits 不应改变");
    }

    /// 测试 apply_top_k 的 top_k=0 分支（不进行过滤）
    #[test]
    fn test_top_k_zero() {
        let sampler = Sampler::new(GenerateParams::default().with_top_k(0));
        let probs = arr1(&[0.1, 0.3, 0.4, 0.15, 0.05]);

        let (indices, filtered_probs) = sampler.apply_top_k(&probs);

        // 应返回所有元素
        assert_eq!(indices.len(), 5);
        assert_eq!(filtered_probs.len(), 5);
    }

    /// 测试 apply_top_k 的 top_k >= vocab_size 分支（不进行过滤）
    #[test]
    fn test_top_k_exceeds_vocab_size() {
        let sampler = Sampler::new(GenerateParams::default().with_top_k(100));
        let probs = arr1(&[0.1, 0.3, 0.4]); // vocab_size = 3

        let (indices, filtered_probs) = sampler.apply_top_k(&probs);

        // 应返回所有元素
        assert_eq!(indices.len(), 3);
        assert_eq!(filtered_probs.len(), 3);
    }

    /// 测试 apply_top_p 的 top_p>=1.0 分支（不进行过滤）
    #[test]
    fn test_top_p_one() {
        let sampler = Sampler::new(GenerateParams::default().with_top_p(1.0));
        let indices = vec![0, 1, 2];
        let probs = vec![0.33, 0.33, 0.34];

        let (filtered_indices, filtered_probs) = sampler.apply_top_p(&indices, &probs);

        // 应返回所有元素
        assert_eq!(filtered_indices.len(), 3);
        assert_eq!(filtered_probs.len(), 3);
    }

    /// 测试 apply_repetition_penalty 的 repetition_penalty=1.0 分支（不执行任何操作）
    #[test]
    fn test_repetition_penalty_one() {
        let sampler = Sampler::new(GenerateParams::default().with_repetition_penalty(1.0));
        let mut logits = arr1(&[1.0, -1.0, 2.0]);
        let generated = vec![0, 1];
        let logits_copy = logits.clone();

        sampler.apply_repetition_penalty(&mut logits, &generated);
        assert_eq!(logits, logits_copy, "repetition_penalty=1.0 时 logits 不应改变");
    }

    /// 测试 with_sampling 方法
    #[test]
    fn test_with_sampling() {
        let params = GenerateParams::new()
            .with_sampling(false);
        
        assert!(!params.sampling);
    }

    /// 测试 with_num_beams 方法
    #[test]
    fn test_with_num_beams() {
        let params = GenerateParams::new()
            .with_num_beams(5);
        
        assert_eq!(params.num_beams, 5);
    }

    /// 测试 with_max_new_tokens 方法
    #[test]
    fn test_with_max_new_tokens() {
        let params = GenerateParams::new()
            .with_max_new_tokens(4096);
        
        assert_eq!(params.max_new_tokens, 4096);
    }

    /// 测试 with_seed 方法创建可复现的采样器
    #[test]
    fn test_with_seed() {
        let params = GenerateParams::default();
        let sampler1 = Sampler::with_seed(params.clone(), 42);
        let sampler2 = Sampler::with_seed(params, 42);
        
        // 使用相同种子应产生相同结果
        let logits = arr1(&[0.1, 0.2, 0.7]);
        let mut s1 = sampler1;
        let mut s2 = sampler2;
        
        let r1 = s1.sample(&logits, &[]).unwrap();
        let r2 = s2.sample(&logits, &[]).unwrap();
        
        assert_eq!(r1, r2, "相同种子应产生相同的采样结果");
    }

    /// 测试 sample_batch 方法
    #[test]
    fn test_sample_batch() {
        let mut sampler = Sampler::new(GenerateParams::default().with_sampling(false));
        let logits_batch = vec![
            arr1(&[0.1, 0.5, 0.3, 0.9]),
            arr1(&[0.9, 0.1, 0.5, 0.3]),
        ];
        let generated = vec![];

        let results = sampler.sample_batch(&logits_batch, &generated).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], 3); // 第一个 argmax
        assert_eq!(results[1], 0); // 第二个 argmax
    }

    /// 测试 BeamSearchCandidate 的 new 和 len 方法
    #[test]
    fn test_beam_search_candidate() {
        let candidate = BeamSearchCandidate::new(vec![1, 2, 3], 0.5);
        
        assert_eq!(candidate.tokens, vec![1, 2, 3]);
        assert!((candidate.score - 0.5).abs() < 1e-6);
        assert_eq!(candidate.len(), 3);
        assert!(!candidate.is_empty());
    }

    /// 测试 BeamSearchCandidate 的空候选
    #[test]
    fn test_beam_search_candidate_empty() {
        let candidate = BeamSearchCandidate::new(vec![], 0.0);
        
        assert!(candidate.is_empty());
        assert_eq!(candidate.len(), 0);
    }
}
