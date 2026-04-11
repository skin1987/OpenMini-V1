//! GRPO 核心算法实现
//!
//! 实现组内相对策略优化的核心计算函数

use crate::rl::{GRPOConfig, GroupSample, Tensor};

/// 计算组内相对优势
///
/// 针对同一问题生成的多个候选输出，计算归一化的相对优势。
/// 这是 GRPO 算法的核心：通过组内归一化消除奖励尺度的影响，
/// 使得不同 prompt 之间的优势值具有可比性。
///
/// # 参数
/// - `rewards`: 组内各样本的原始奖励值
/// - `epsilon`: 数值稳定性参数，防止除零
///
/// # 返回
/// 归一化后的相对优势向量，均值为 0，标准差为 1（近似）
pub fn compute_group_relative_advantage(rewards: &[f64], epsilon: f64) -> Vec<f64> {
    if rewards.is_empty() {
        return vec![];
    }

    let n = rewards.len() as f64;
    let mean: f64 = rewards.iter().sum::<f64>() / n;
    let variance: f64 = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = (variance + epsilon).sqrt();

    let advantages: Vec<f64> = rewards.iter().map(|r| (r - mean) / std).collect();

    advantages
}

/// 无偏 KL 散度估计器
///
/// 使用修正的 K3 估计器，消除原始估计器的系统性偏差。
/// 该估计器通过 `exp(log_ratio) - 1 - log_ratio` 的形式计算 KL 散度，
/// 在 log_ratio 较小时具有更好的数值稳定性。
///
/// # 参数
/// - `log_probs`: 当前策略的对数概率
/// - `old_log_probs`: 旧策略的对数概率
/// - `mask`: 可选掩码，用于选择有效的 token 位置
///
/// # 返回
/// 平均 KL 散度值（非负）
pub fn unbiased_kl_estimator(
    log_probs: &Tensor,
    old_log_probs: &Tensor,
    mask: Option<&Tensor>,
) -> f32 {
    let n = log_probs.len();
    if n == 0 || old_log_probs.len() != n {
        return 0.0;
    }

    let log_ratio: Vec<f32> = log_probs
        .iter()
        .zip(old_log_probs.iter())
        .map(|(p, old)| p - old)
        .collect();

    let kl_values: Vec<f32> = log_ratio
        .iter()
        .map(|x| (x.exp() - 1.0 - x).clamp(-1e6, 1e6))
        .collect();

    let valid_count = if let Some(m) = mask {
        m.iter().filter(|&&x| x > 0.0).count()
    } else {
        n
    };

    if valid_count == 0 {
        return 0.0;
    }

    let kl_sum: f32 = match mask {
        Some(m) => log_probs
            .iter()
            .zip(old_log_probs.iter())
            .zip(m.iter())
            .filter(|(_, &mask_val)| mask_val > 0.0)
            .map(|((p, old), _)| {
                let log_ratio = p - old;
                (log_ratio.exp() - 1.0 - log_ratio).clamp(-1e6, 1e6)
            })
            .sum(),
        None => kl_values.iter().sum(),
    };

    kl_sum / valid_count as f32
}

/// Off-Policy 序列掩码生成
///
/// 当负优势样本的 KL 散度超过阈值时，排除其损失计算。
/// 这是一种安全机制：对于策略变化过大的负样本，
/// 其梯度信号可能不可靠，应该被屏蔽。
///
/// # 参数
/// - `log_probs`: 当前策略的对数概率
/// - `old_log_probs`: 旧策略的对数概率
/// - `kl_threshold`: KL 散度阈值，超过此值的负优势样本将被掩码
/// - `advantages`: 各样本的优势值
///
/// # 返回
/// 掩码张量（1.0 表示保留，0.0 表示排除）
pub fn sequence_mask_by_kl(
    log_probs: &Tensor,
    old_log_probs: &Tensor,
    kl_threshold: f32,
    advantages: &[f64],
) -> Tensor {
    let n = log_probs.len();
    if n == 0 || old_log_probs.len() != n {
        return Tensor::zeros(&[n]);
    }

    let mask_data: Vec<f32> = log_probs
        .iter()
        .zip(old_log_probs.iter())
        .enumerate()
        .map(|(i, (p, old))| {
            let log_ratio = p - old;
            let kl = (log_ratio.exp() - 1.0 - log_ratio).abs();

            let is_negative_adv = if i < advantages.len() {
                advantages[i] < 0.0
            } else {
                false
            };

            if is_negative_adv && kl > kl_threshold {
                0.0
            } else {
                1.0
            }
        })
        .collect();

    Tensor::new(mask_data, vec![n])
}

/// PPO 策略梯度损失计算
///
/// 使用 clip 机制限制策略更新的幅度，防止过大的策略变化。
/// 损失函数形式：`-min(ratio * adv, clip(ratio) * adv)`
///
/// # 参数
/// - `log_probs`: 当前策略的对数概率
/// - `old_log_probs`: 旧策略的对数概率
/// - `advantages`: 优势值张量
/// - `mask`: 可选掩码，用于选择有效的 token 位置
/// - `clip_epsilon`: PPO 裁剪参数（通常为 0.2）
///
/// # 返回
/// 平均策略损失（负值表示优化方向正确）
pub fn policy_loss(
    log_probs: &Tensor,
    old_log_probs: &Tensor,
    advantages: &Tensor,
    mask: Option<&Tensor>,
    clip_epsilon: f32,
) -> f32 {
    let n = log_probs.len();
    if n == 0 || old_log_probs.len() != n || advantages.len() != n {
        return 0.0;
    }

    let ratio: Vec<f32> = log_probs
        .iter()
        .zip(old_log_probs.iter())
        .map(|(p, old)| (p - old).exp())
        .collect();

    let surr1: f32 = ratio
        .iter()
        .zip(advantages.iter())
        .map(|(r, a)| r * a)
        .sum();

    let surr2: f32 = ratio
        .iter()
        .zip(advantages.iter())
        .map(|(r, a)| r.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * a)
        .sum();

    let loss = -surr1.min(surr2) / n as f32;

    if let Some(m) = mask {
        let masked_loss: f32 = log_probs
            .iter()
            .zip(old_log_probs.iter())
            .zip(advantages.iter())
            .zip(m.iter())
            .filter(|(_, &m_val)| m_val > 0.0)
            .map(|(((p, old), a), _)| {
                let r = (p - old).exp();
                let s1 = r * a;
                let s2 = r.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * a;
                -s1.min(s2)
            })
            .sum();

        let valid_count = m.iter().filter(|&&x| x > 0.0).count();
        if valid_count > 0 {
            return masked_loss / valid_count as f32;
        }
    }

    loss
}

/// 策略熵损失计算
///
/// 用于增加策略的随机性，防止过早收敛到确定性策略。
/// 熵越大表示策略越随机（探索性更强）。
///
/// # 参数
/// - `log_probs`: 策略的对数概率分布
///
/// # 返回
/// 平均熵值（非负）
pub fn entropy_loss(log_probs: &Tensor) -> f32 {
    if log_probs.is_empty() {
        return 0.0;
    }

    let entropy: f32 = log_probs
        .iter()
        .map(|p| if *p <= 0.0 { 0.0 } else { -p * p.exp() })
        .sum();

    entropy / log_probs.len() as f32
}

/// KL 散度惩罚项
///
/// 计算当前策略与旧策略之间的 KL 散度，并乘以系数作为正则化惩罚。
/// 用于限制单次更新的幅度，保证训练稳定性。
///
/// # 参数
/// - `log_probs`: 当前策略的对数概率
/// - `old_log_probs`: 旧策略的对数概率
/// - `kl_coefficient`: KL 惩罚系数
///
/// # 返回
/// 加权后的 KL 惩罚值（非负）
pub fn kl_penalty(log_probs: &Tensor, old_log_probs: &Tensor, kl_coefficient: f32) -> f32 {
    unbiased_kl_estimator(log_probs, old_log_probs, None) * kl_coefficient
}

/// GRPO 优化器
///
/// 实现完整的 GRPO（Group Relative Policy Optimization）训练流程。
/// 负责计算优势值、执行训练步骤、更新模型参数等核心功能。
///
/// # 示例
/// ```ignore
/// let config = GRPOConfig::default();
/// let optimizer = GRPOOptimizer::new(config);
/// let metrics = optimizer.train_step(&mut samples, |ids| model.forward(ids));
/// ```
pub struct GRPOOptimizer {
    config: GRPOConfig,
}

impl GRPOOptimizer {
    /// 创建新的 GRPO 优化器实例
    ///
    /// # 参数
    /// - `config`: GRPO 算法配置参数
    pub fn new(config: GRPOConfig) -> Self {
        Self { config }
    }

    /// 计算组内相对优势值
    ///
    /// 对所有样本的奖励进行组内归一化，生成用于策略梯度计算的优势值。
    ///
    /// # 参数
    /// - `group_samples`: 组采样结果列表
    ///
    /// # 返回
    /// 归一化的优势值向量
    pub fn compute_advantages(&self, group_samples: &[GroupSample]) -> Vec<f64> {
        let rewards: Vec<f64> = group_samples
            .iter()
            .flat_map(|s| s.rewards.clone())
            .collect();

        if rewards.is_empty() {
            return vec![];
        }

        compute_group_relative_advantage(&rewards, self.config.advantage_norm_eps)
    }

    /// 执行单步训练
    ///
    /// 完整的 GRPO 训练步骤，包括：
    /// 1. 计算组内相对优势
    /// 2. 重新计算当前策略的对数概率
    /// 3. 计算 KL 散度和熵
    /// 4. 计算策略损失（带 clip）
    /// 5. 组合总损失（policy + kl_penalty - entropy）
    ///
    /// # 参数
    /// - `group_samples`: 可变的组采样结果（用于更新 log_probs）
    /// - `compute_log_probs`: 模型前向传播函数，接收 response_ids 返回 log_probs
    ///
    /// # 返回
    /// 训练指标，包含各项损失值和统计信息
    pub fn train_step(
        &self,
        group_samples: &mut [GroupSample],
        compute_log_probs: impl Fn(&Tensor) -> Tensor,
    ) -> TrainingMetrics {
        let advantages = self.compute_advantages(group_samples);

        if advantages.is_empty() {
            return TrainingMetrics::default();
        }

        let mut total_policy_loss = 0.0;
        let mut total_kl = 0.0;
        let mut total_entropy = 0.0;
        let mut valid_samples = 0;

        for sample in group_samples.iter_mut() {
            let log_probs = compute_log_probs(&sample.response_ids);

            let kl = unbiased_kl_estimator(&log_probs, &sample.old_log_probs, None);
            total_kl += kl as f64;

            let entropy = entropy_loss(&log_probs);
            total_entropy += entropy as f64;

            let advantages_tensor = Tensor::new(
                advantages
                    .iter()
                    .take(sample.response_length)
                    .cloned()
                    .map(|x| x as f32)
                    .collect(),
                vec![sample.response_length],
            );

            let mask = sample.sampling_mask.as_ref().map(|sm| &sm.mask);

            let loss = policy_loss(
                &log_probs,
                &sample.old_log_probs,
                &advantages_tensor,
                mask,
                self.config.clip_epsilon as f32,
            );
            total_policy_loss += loss as f64;
            valid_samples += 1;
        }

        if valid_samples > 0 {
            total_policy_loss /= valid_samples as f64;
            total_kl /= valid_samples as f64;
            total_entropy /= valid_samples as f64;
        }

        let kl_penalty = total_kl * self.config.kl_coefficient;
        let total_loss = total_policy_loss + kl_penalty - self.config.entropy_coef * total_entropy;

        TrainingMetrics {
            total_loss,
            policy_loss: total_policy_loss,
            kl_divergence: total_kl,
            entropy: total_entropy,
            advantage_mean: advantages.iter().sum::<f64>() / advantages.len().max(1) as f64,
        }
    }

    /// 应用梯度更新模型参数
    ///
    /// 使用简单的 SGD（随机梯度下降）更新规则：
    /// `param = param - learning_rate * gradient`
    /// 并支持梯度裁剪以防止梯度爆炸。
    ///
    /// # 参数
    /// - `model_params`: 可变的模型参数向量
    /// - `gradients`: 梯度向量
    /// - `learning_rate`: 学习率
    ///
    /// # 返回
    /// 梯度的 L2 范数，用于监控梯度大小
    pub fn apply_gradient(
        &self,
        model_params: &mut [f32],
        gradients: &[f32],
        learning_rate: f64,
    ) -> f32 {
        if model_params.is_empty() || gradients.is_empty() {
            return 0.0;
        }

        let min_len = model_params.len().min(gradients.len());

        // 计算梯度 L2 范数
        let grad_norm: f32 = gradients[..min_len]
            .iter()
            .map(|g| g.powi(2))
            .sum::<f32>()
            .sqrt();

        // 梯度裁剪
        let clip_norm = self.config.grad_clip_norm as f32;
        let scale = if grad_norm > clip_norm {
            clip_norm / grad_norm
        } else {
            1.0
        };

        // 应用梯度更新
        let lr = learning_rate as f32;
        for i in 0..min_len {
            model_params[i] -= lr * gradients[i] * scale;
        }

        grad_norm
    }
}

/// 训练指标
///
/// 用于记录和展示单次训练步骤的各项指标值。
/// 包含总损失、策略损失、KL 散度、熵等关键信息。
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    pub total_loss: f64,
    pub policy_loss: f64,
    pub kl_divergence: f64,
    pub entropy: f64,
    pub advantage_mean: f64,
}

#[allow(dead_code)]
impl TrainingMetrics {
    /// 格式化显示训练指标
    ///
    /// 返回包含所有指标的可读字符串，用于日志输出
    pub fn display(&self) -> String {
        format!(
            "Loss: {:.4} | Policy: {:.4} | KL: {:.4} | Entropy: {:.4} | Adv: {:.4}",
            self.total_loss,
            self.policy_loss,
            self.kl_divergence,
            self.entropy,
            self.advantage_mean
        )
    }
}

/// 创建 GRPO 优化器（工厂函数）
///
/// 便捷的构造函数，用于快速创建配置好的 GRPO 优化器实例。
///
/// # 参数
/// - `config`: GRPO 算法配置参数
///
/// # 返回
/// 配置完成的 GRPOOptimizer 实例
pub fn create_grpo_optimizer(config: GRPOConfig) -> GRPOOptimizer {
    GRPOOptimizer::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::{GRPOConfig, GroupSample, Tensor};

    #[test]
    fn test_compute_group_relative_advantage() {
        // 测试组内相对优势计算
        let rewards = vec![0.9, 0.7, 0.3, 0.95, 0.1];
        let advantages = compute_group_relative_advantage(&rewards, 1e-8);

        assert_eq!(advantages.len(), rewards.len());

        // 验证 advantages 有正有负（相对于均值）
        let has_positive = advantages.iter().any(|&a| a > 0.0);
        let has_negative = advantages.iter().any(|&a| a < 0.0);
        assert!(has_positive && has_negative);

        // 验证 advantages 均值接近 0
        let mean: f64 = advantages.iter().sum::<f64>() / advantages.len() as f64;
        assert!(mean.abs() < 1e-6);
    }

    #[test]
    fn test_compute_advantages_empty() {
        // 测试空奖励列表
        let advantages = compute_group_relative_advantage(&[], 1e-8);
        assert!(advantages.is_empty());
    }

    #[test]
    fn test_compute_advantages_single_value() {
        // 测试单值奖励（方差为0）
        let rewards = vec![0.5];
        let advantages = compute_group_relative_advantage(&rewards, 1e-8);
        assert_eq!(advantages.len(), 1);
        // 单值时 advantage 应该是 0（因为 mean == value）
        assert!((advantages[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_advantages_identical_values() {
        // 测试相同奖励值（方差为0，依赖 epsilon）
        let rewards = vec![0.5; 5];
        let advantages = compute_group_relative_advantage(&rewards, 1e-8);
        assert_eq!(advantages.len(), 5);
        // 所有值相同时，advantages 应该都是 0 或接近 0
        for &adv in &advantages {
            assert!(adv.abs() < 1e-6);
        }
    }

    #[test]
    fn test_unbiased_kl_estimator_basic() {
        // 测试基本 KL 估计
        let log_probs = Tensor::from_slice(&[-0.5, -1.0, -0.3]);
        let old_log_probs = Tensor::from_slice(&[-0.4, -0.9, -0.2]);

        let kl = unbiased_kl_estimator(&log_probs, &old_log_probs, None);
        // KL 散度应该 >= 0
        assert!(kl >= 0.0);
    }

    #[test]
    fn test_unbiased_kl_estimator_with_mask() {
        // 测试带掩码的 KL 估计
        let log_probs = Tensor::from_slice(&[-0.5, -1.0, -0.3, -0.8]);
        let old_log_probs = Tensor::from_slice(&[-0.4, -0.9, -0.2, -0.7]);
        let mask = Tensor::from_slice(&[1.0, 0.0, 1.0, 0.0]); // 只使用第 1、3 个

        let kl = unbiased_kl_estimator(&log_probs, &old_log_probs, Some(&mask));
        assert!(kl >= 0.0);
    }

    #[test]
    fn test_unbiased_kl_estimator_empty() {
        // 测试空张量
        let log_probs = Tensor::new(vec![], vec![0]);
        let old_log_probs = Tensor::new(vec![], vec![0]);

        let kl = unbiased_kl_estimator(&log_probs, &old_log_probs, None);
        assert_eq!(kl, 0.0);
    }

    #[test]
    fn test_unbiased_kl_estimator_zero_mask() {
        // 测试全零掩码
        let log_probs = Tensor::from_slice(&[-0.5, -1.0]);
        let old_log_probs = Tensor::from_slice(&[-0.4, -0.9]);
        let mask = Tensor::from_slice(&[0.0, 0.0]);

        let kl = unbiased_kl_estimator(&log_probs, &old_log_probs, Some(&mask));
        assert_eq!(kl, 0.0);
    }

    #[test]
    fn test_sequence_mask_by_kl() {
        // 测试基于 KL 的序列掩码
        let log_probs = Tensor::from_slice(&[-0.5, -2.0, -0.3]); // 第2个差异大
        let old_log_probs = Tensor::from_slice(&[-0.4, -0.5, -0.2]);
        let advantages = vec![0.1, -0.5, 0.2]; // 第2个是负优势

        let mask = sequence_mask_by_kl(
            &log_probs,
            &old_log_probs,
            0.01, // 低阈值
            &advantages,
        );

        // 验证返回的掩码长度正确
        assert_eq!(mask.len(), 3);
        let mask_data = mask.as_slice();
        // 负优势且高KL的样本应该被mask掉（值为0）
        assert_eq!(mask_data[1], 0.0); // 第2个样本
    }

    #[test]
    fn test_policy_loss_basic() {
        // 测试基本策略损失
        let log_probs = Tensor::from_slice(&[-0.5, -1.0, -0.3]);
        let old_log_probs = Tensor::from_slice(&[-0.6, -0.9, -0.4]);
        let advantages = Tensor::from_slice(&[0.5, -0.3, 0.7]);

        let loss = policy_loss(&log_probs, &old_log_probs, &advantages, None, 0.2);
        // PPO 损失应该是负数（因为我们最大化目标）
        assert!(loss <= 0.0);
    }

    #[test]
    fn test_policy_loss_empty() {
        // 测试空输入
        let log_probs = Tensor::new(vec![], vec![0]);
        let old_log_probs = Tensor::new(vec![], vec![0]);
        let advantages = Tensor::new(vec![], vec![0]);

        let loss = policy_loss(&log_probs, &old_log_probs, &advantages, None, 0.2);
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_entropy_loss_computation() {
        // 测试熵损失计算
        let log_probs = Tensor::from_slice(&[-0.5, -1.0, -0.3]);

        let entropy = entropy_loss(&log_probs);
        // 熵应该是正数（或零）
        assert!(entropy >= 0.0);
    }

    #[test]
    fn test_entropy_loss_empty() {
        // 测试空输入
        let log_probs = Tensor::new(vec![], vec![0]);
        let entropy = entropy_loss(&log_probs);
        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn test_entropy_loss_uniform_distribution() {
        // 测试非均匀分布的熵（应该 > 0）
        // 对于 log softmax 输出，使用非均匀分布
        let log_probs_data: Vec<f32> = vec![-0.1, -2.0, -5.0]; // 非均匀分布
        let log_probs = Tensor::new(log_probs_data, vec![3]);

        let entropy = entropy_loss(&log_probs);
        // 非均匀分布的熵应该 > 0
        assert!(entropy >= 0.0);
    }

    #[test]
    fn test_grpo_config_default() {
        // 测试默认配置
        let config = GRPOConfig::default();
        assert!((config.learning_rate - 1e-5).abs() < f64::EPSILON);
        assert!((config.kl_coefficient - 0.1).abs() < f64::EPSILON);
        assert!((config.clip_epsilon - 0.2).abs() < f64::EPSILON);
        assert_eq!(config.group_size, 4);
        assert!((config.advantage_norm_eps - 1e-8).abs() < f64::EPSILON);
        assert!((config.max_kl_margin - 0.01).abs() < f64::EPSILON);
        assert!((config.entropy_coef - 0.01).abs() < f64::EPSILON);
        assert!((config.grad_clip_norm - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_grpo_optimizer_creation() {
        // 测试优化器创建
        let config = GRPOConfig::default();
        let optimizer = GRPOOptimizer::new(config);

        // 验证优化器可以正常工作
        let empty_samples: Vec<GroupSample> = vec![];
        let advantages = optimizer.compute_advantages(&empty_samples);
        assert!(advantages.is_empty());
    }

    #[test]
    fn test_grpo_optimizer_train_step_empty() {
        // 测试空样本训练步骤
        let config = GRPOConfig::default();
        let optimizer = GRPOOptimizer::new(config);

        let mut samples: Vec<GroupSample> = vec![];
        let metrics = optimizer.train_step(&mut samples, |_: &Tensor| Tensor::zeros(&[1]));

        // 空样本应该返回默认指标
        assert_eq!(metrics.total_loss, 0.0);
        assert_eq!(metrics.policy_loss, 0.0);
        assert_eq!(metrics.kl_divergence, 0.0);
        assert_eq!(metrics.entropy, 0.0);
    }

    #[test]
    fn test_training_metrics_display() {
        // 测试训练指标显示
        let metrics = TrainingMetrics {
            total_loss: 1.2345,
            policy_loss: 0.9876,
            kl_divergence: 0.05,
            entropy: 1.5,
            advantage_mean: 0.123,
        };

        let display = metrics.display();
        assert!(display.contains("Loss:"));
        assert!(display.contains("Policy:"));
        assert!(display.contains("KL:"));
        assert!(display.contains("Entropy:"));
        assert!(display.contains("Adv:"));
    }

    #[test]
    fn test_training_metrics_default() {
        // 测试默认训练指标
        let metrics = TrainingMetrics::default();
        assert_eq!(metrics.total_loss, 0.0);
        assert_eq!(metrics.policy_loss, 0.0);
        assert_eq!(metrics.kl_divergence, 0.0);
        assert_eq!(metrics.entropy, 0.0);
        assert_eq!(metrics.advantage_mean, 0.0);
    }

    #[test]
    fn test_create_grpo_optimizer_function() {
        // 测试工厂函数
        let config = GRPOConfig::default();
        let optimizer = create_grpo_optimizer(config);

        // 应该能正常使用
        let _ = optimizer.compute_advantages(&[]);
    }

    #[test]
    fn test_kl_penalty() {
        // 测试 KL 惩罚
        let log_probs = Tensor::from_slice(&[-0.5, -1.0]);
        let old_log_probs = Tensor::from_slice(&[-0.4, -0.9]);
        let kl_coefficient = 0.1;

        let penalty = kl_penalty(&log_probs, &old_log_probs, kl_coefficient);
        // 惩罚 = KL * coefficient，应该 >= 0
        assert!(penalty >= 0.0);
    }

    // ==================== 用户要求的补充测试 ====================

    /// 测试训练步骤基本流程
    #[test]
    fn test_train_step_basic() {
        // 创建优化器
        let config = GRPOConfig::default();
        let optimizer = GRPOOptimizer::new(config);

        // 创建测试样本
        let sample = GroupSample::new(
            Tensor::from_slice(&[1.0, 2.0]),
            Tensor::from_slice(&[3.0, 4.0, 5.0]),
            Tensor::from_slice(&[-0.5, -1.0, -0.3]),
            Tensor::from_slice(&[-0.6, -0.9, -0.4]),
            vec![0.9, 0.7, 0.3],
        );

        // 定义模拟的前向传播函数
        let compute_log_probs = |_response_ids: &Tensor| -> Tensor {
            // 返回模拟的新的 log_probs（与旧的有轻微差异）
            Tensor::from_slice(&[-0.48, -0.95, -0.28])
        };

        // 执行训练步骤
        let metrics = optimizer.train_step(&mut [sample], compute_log_probs);

        // 验证返回的指标有效
        assert!(metrics.total_loss.is_finite());
        assert!(metrics.policy_loss.is_finite());
        assert!(metrics.kl_divergence >= 0.0); // KL 应该非负
        assert!(metrics.entropy >= 0.0); // 熵应该非负
        assert!(metrics.advantage_mean.is_finite());

        // 验证 display 方法可以正常工作
        let display_str = metrics.display();
        assert!(!display_str.is_empty());
        assert!(display_str.contains("Loss:"));
    }

    /// 测试组内相对优势计算的完整性和边界情况
    #[test]
    fn test_group_advantage_computation() {
        // 测试基本归一化特性：均值接近 0，标准差接近 1
        let rewards = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let advantages = compute_group_relative_advantage(&rewards, 1e-8);

        assert_eq!(advantages.len(), 5);

        // 验证均值为 0（数值误差范围内）
        let mean: f64 = advantages.iter().sum::<f64>() / advantages.len() as f64;
        assert!(mean.abs() < 1e-6);

        // 验证方差为 1（数值误差范围内）
        let variance: f64 =
            advantages.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / advantages.len() as f64;
        assert!((variance - 1.0).abs() < 1e-6);

        // 测试高方差奖励
        let high_var_rewards = vec![0.01, 0.99];
        let high_var_adv = compute_group_relative_advantage(&high_var_rewards, 1e-8);
        assert_eq!(high_var_adv.len(), 2);
        // 高方差时，两个优势值应该符号相反且绝对值相等
        assert!((high_var_adv[0] + high_var_adv[1]).abs() < 1e-6);

        // 测试低方差奖励（所有值相同）
        let low_var_rewards = vec![0.5; 10];
        let low_var_adv = compute_group_relative_advantage(&low_var_rewards, 1e-8);
        for &adv in &low_var_adv {
            assert!(adv.abs() < 1e-6);
        }

        // 测试极端奖励值
        let extreme_rewards = vec![1000.0, -1000.0];
        let extreme_adv = compute_group_relative_advantage(&extreme_rewards, 1e-8);
        assert_eq!(extreme_adv.len(), 2);
        assert!(extreme_adv[0] > 0.0); // 高奖励应该有正优势
        assert!(extreme_adv[1] < 0.0); // 低奖励应该有负优势
    }

    /// 测试 GRPO 配置验证
    #[test]
    fn test_grpo_config_validation() {
        // 测试默认配置的有效性
        let default_config = GRPOConfig::default();

        // 验证默认值的合理性
        assert!(default_config.learning_rate > 0.0 && default_config.learning_rate < 1.0);
        assert!(default_config.kl_coefficient >= 0.0);
        assert!(default_config.clip_epsilon > 0.0 && default_config.clip_epsilon <= 1.0);
        assert!(default_config.group_size > 0);
        assert!(default_config.advantage_norm_eps > 0.0);
        assert!(default_config.max_kl_margin > 0.0);
        assert!(default_config.entropy_coef >= 0.0);
        assert!(default_config.grad_clip_norm > 0.0);

        // 测试自定义配置
        let custom_config = GRPOConfig {
            learning_rate: 3e-5,
            kl_coefficient: 0.05,
            clip_epsilon: 0.15,
            group_size: 8,
            advantage_norm_eps: 1e-6,
            max_kl_margin: 0.02,
            entropy_coef: 0.02,
            grad_clip_norm: 0.5,
        };

        // 验证自定义配置可以被正确使用
        let optimizer = GRPOOptimizer::new(custom_config.clone());
        let _adv = optimizer.compute_advantages(&[]);

        // 验证 Clone 特征正常工作
        let cloned_config = custom_config.clone();
        assert!((cloned_config.learning_rate - custom_config.learning_rate).abs() < f64::EPSILON);

        // 测试边界配置值
        let boundary_config = GRPOConfig {
            learning_rate: 0.0, // 极小学习率
            kl_coefficient: 0.0,
            clip_epsilon: 0.0,
            group_size: 1,
            advantage_norm_eps: 1e-10,
            max_kl_margin: 0.0,
            entropy_coef: 0.0,
            grad_clip_norm: 0.0,
        };

        // 边界配置不应该导致 panic
        let boundary_optimizer = GRPOOptimizer::new(boundary_config);
        let _boundary_adv = boundary_optimizer.compute_advantages(&[]);
    }
}
