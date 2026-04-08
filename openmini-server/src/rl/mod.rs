//! GRPO 强化学习模块
//!
//! 实现 Group Relative Policy Optimization 算法，用于提升 RL 训练稳定性和效率
//!
//! # 核心特性
//! - 组内相对奖励机制（无需 Critic 网络）
//! - 无偏 KL 估计器（修正 K3 偏差）
//! - Off-Policy 序列掩码
//! - Keep Routing（专家路由保持）
//! - Keep Sampling Mask（采样掩码保持）

pub mod actor;
pub mod grpo;
pub mod keep_routing;
pub mod keep_sampling_mask;
pub mod reward;

use std::collections::HashMap;

/// 简化的张量类型，用于 RL 模块内的数值计算
#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![0.0; size],
            shape: shape.to_vec(),
        }
    }

    pub fn from_slice(slice: &[f32]) -> Self {
        Self {
            data: slice.to_vec(),
            shape: vec![slice.len()],
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn size(&self, dim: usize) -> usize {
        self.shape.get(dim).copied().unwrap_or(1)
    }

    pub fn dim(&self) -> usize {
        self.shape.len()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.data.iter()
    }

    pub fn mean(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().sum::<f32>() / self.data.len() as f32
    }

    pub fn std(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let mean = self.mean();
        let variance =
            self.data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / self.data.len() as f32;
        variance.sqrt()
    }

    pub fn softmax(&self) -> Self {
        if self.data.is_empty() {
            return Self::zeros(&self.shape);
        }
        let max_val = self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = self.data.iter().map(|x| (x - max_val).exp()).sum();
        let softmax_data: Vec<f32> = self
            .data
            .iter()
            .map(|x| (x - max_val).exp() / exp_sum)
            .collect();
        Self {
            data: softmax_data,
            shape: self.shape.clone(),
        }
    }

    pub fn clamp(&self, min: f32, max: f32) -> Self {
        Self {
            data: self.data.iter().map(|x| x.clamp(min, max)).collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn abs_sum(&self) -> f32 {
        self.data.iter().map(|x| x.abs()).sum()
    }
}

impl From<Vec<f32>> for Tensor {
    fn from(data: Vec<f32>) -> Self {
        Self {
            data,
            shape: vec![1],
        }
    }
}

/// GRPO 算法配置
#[derive(Debug, Clone)]
pub struct GRPOConfig {
    pub learning_rate: f64,
    pub kl_coefficient: f64,
    pub clip_epsilon: f64,
    pub group_size: usize,
    pub advantage_norm_eps: f64,
    pub max_kl_margin: f64,
    pub entropy_coef: f64,
    pub grad_clip_norm: f64,
}

impl Default for GRPOConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-5,
            kl_coefficient: 0.1,
            clip_epsilon: 0.2,
            group_size: 4,
            advantage_norm_eps: 1e-8,
            max_kl_margin: 0.01,
            entropy_coef: 0.01,
            grad_clip_norm: 1.0,
        }
    }
}

/// 组采样结果
///
/// 包含同一 prompt 生成的多个候选响应的完整信息
#[derive(Debug, Clone)]
pub struct GroupSample {
    pub prompt_ids: Tensor,
    pub response_ids: Tensor,
    pub log_probs: Tensor,
    pub old_log_probs: Tensor,
    pub rewards: Vec<f64>,
    pub advantage: Option<Tensor>,
    pub prompt_length: usize,
    pub response_length: usize,
    pub router_cache: Option<RouterCache>,
    pub sampling_mask: Option<SamplingMask>,
}

/// 路由器缓存 - Keep Routing
///
/// 保存推理时选择的专家路由，训练时强制使用相同路由
#[derive(Debug, Clone)]
pub struct RouterCache {
    pub expert_indices: Tensor,
    pub routing_weights: Tensor,
    pub layer_idx: usize,
}

/// 采样掩码 - Keep Sampling Mask
///
/// 保存采样截断位置，确保训练与推理动作空间一致
#[derive(Debug, Clone)]
pub struct SamplingMask {
    pub mask: Tensor,
    pub valid_lengths: Vec<usize>,
}

/// 奖励计算结果
#[derive(Debug, Clone)]
pub struct RewardResult {
    pub total_reward: f64,
    pub accuracy_reward: f64,
    pub format_reward: f64,
    pub is_correct: bool,
    pub details: HashMap<String, f64>,
}

impl RewardResult {
    pub fn new(total_reward: f64, is_correct: bool) -> Self {
        Self {
            total_reward,
            accuracy_reward: 0.0,
            format_reward: 0.0,
            is_correct,
            details: HashMap::new(),
        }
    }

    pub fn with_details(mut self, details: HashMap<String, f64>) -> Self {
        self.details = details;
        self
    }
}

#[allow(dead_code)]
impl GroupSample {
    pub fn new(
        prompt_ids: Tensor,
        response_ids: Tensor,
        log_probs: Tensor,
        old_log_probs: Tensor,
        rewards: Vec<f64>,
    ) -> Self {
        let prompt_length = prompt_ids.dim();
        let response_length = response_ids.dim();
        Self {
            prompt_ids,
            response_ids,
            log_probs,
            old_log_probs,
            rewards,
            advantage: None,
            prompt_length,
            response_length,
            router_cache: None,
            sampling_mask: None,
        }
    }

    pub fn with_router_cache(mut self, cache: RouterCache) -> Self {
        self.router_cache = Some(cache);
        self
    }

    pub fn with_sampling_mask(mut self, mask: SamplingMask) -> Self {
        self.sampling_mask = Some(mask);
        self
    }

    pub fn set_advantage(&mut self, advantage: Tensor) {
        self.advantage = Some(advantage);
    }
}
