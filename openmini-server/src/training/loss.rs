use ndarray::{Array2, ArrayD};
use serde::{Deserialize, Serialize};

/// 损失 reduction 类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Reduction {
    Mean,
    Sum,
    None,
}

/// CrossEntropyLoss 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossEntropyConfig {
    pub ignore_index: usize,
    pub label_smoothing: f64,
    pub reduction: Reduction,
}

impl Default for CrossEntropyConfig {
    fn default() -> Self {
        Self {
            ignore_index: usize::MAX,
            label_smoothing: 0.0,
            reduction: Reduction::Mean,
        }
    }
}

/// 损失计算结果
pub struct LossOutput {
    pub loss: f64,
    pub individual_losses: Option<ArrayD<f64>>,
    pub num_valid_tokens: usize,
}

pub struct CrossEntropyLoss {
    config: CrossEntropyConfig,
}

impl CrossEntropyLoss {
    pub fn new(config: CrossEntropyConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(CrossEntropyConfig::default())
    }

    /// 计算 Cross Entropy Loss
    ///
    /// # Arguments
    /// * `logits` - 模型输出 [N, V] N=样本数, V=vocab_size
    /// * `labels` - 目标标签 [N] 每个样本的目标 token ID
    ///
    /// # Returns
    /// LossOutput 包含标量损失和其他信息
    pub fn forward(&self, logits: &Array2<f32>, labels: &[usize]) -> LossOutput {
        let (n_samples, vocab_size) = logits.dim();
        let epsilon = 1e-12_f32;

        // 1. Softmax
        let probs = softmax(logits);

        // 2. 过滤 ignore_index
        let valid_indices: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, &label)| label != self.config.ignore_index)
            .map(|(i, _)| i)
            .collect();

        if valid_indices.is_empty() {
            return LossOutput {
                loss: 0.0,
                individual_losses: None,
                num_valid_tokens: 0,
            };
        }

        let num_valid = valid_indices.len();

        // 3. 计算逐样本损失
        let mut individual_losses = Vec::with_capacity(n_samples);

        for (i, &label) in labels.iter().enumerate() {
            if label == self.config.ignore_index {
                individual_losses.push(0.0);
                continue;
            }

            if label >= vocab_size {
                panic!("Label {} exceeds vocabulary size {}", label, vocab_size);
            }

            let sample_loss = if self.config.label_smoothing > 0.0 {
                self.compute_smoothed_loss(&probs, i, label, vocab_size)
            } else {
                let prob = probs[[i, label]].max(epsilon);
                -(prob.ln() as f64)
            };

            individual_losses.push(sample_loss);
        }

        // 4. 根据 reduction 计算最终损失
        let loss = match self.config.reduction {
            Reduction::Mean => {
                let sum: f64 = individual_losses.iter().sum();
                sum / num_valid as f64
            }
            Reduction::Sum => individual_losses.iter().sum(),
            Reduction::None => 0.0,
        };

        // 5. 构建 individual_losses 数组（如果 reduction=None）
        let individual_array = if self.config.reduction == Reduction::None {
            let arr = ndarray::Array1::from(individual_losses);
            Some(arr.into_dyn())
        } else {
            None
        };

        LossOutput {
            loss,
            individual_losses: individual_array,
            num_valid_tokens: num_valid,
        }
    }

    fn compute_smoothed_loss(
        &self,
        probs: &Array2<f32>,
        sample_idx: usize,
        label: usize,
        vocab_size: usize,
    ) -> f64 {
        let smoothing = self.config.label_smoothing as f32;
        let epsilon = 1e-12_f32;

        let mut loss = 0.0_f64;
        for k in 0..vocab_size {
            let prob = probs[[sample_idx, k]].max(epsilon);
            let log_prob = prob.ln() as f64;

            let true_prob = if k == label {
                1.0 - smoothing as f64
            } else {
                smoothing as f64 / vocab_size as f64
            };

            loss -= true_prob * log_prob;
        }

        loss
    }

    /// 计算梯度（对 logits 的梯度）
    ///
    /// 用于反向传播时计算 d_loss/d_logits
    pub fn backward(&self, logits: &Array2<f32>, labels: &[usize]) -> Array2<f32> {
        let (n_samples, vocab_size) = logits.dim();

        // 1. 计算 softmax
        let probs = softmax(logits);

        // 2. 初始化梯度矩阵
        let mut grad = Array2::<f32>::zeros((n_samples, vocab_size));

        // 3. 对每个样本计算梯度
        for (i, &label) in labels.iter().enumerate() {
            if label == self.config.ignore_index || label >= vocab_size {
                continue;
            }

            if self.config.label_smoothing > 0.0 {
                let smoothing = self.config.label_smoothing as f32;
                for k in 0..vocab_size {
                    let true_dist = if k == label {
                        1.0 - smoothing
                    } else {
                        smoothing / vocab_size as f32
                    };
                    grad[[i, k]] = probs[[i, k]] - true_dist;
                }
            } else {
                grad[[i, label]] = probs[[i, label]] - 1.0;
                for k in 0..vocab_size {
                    if k != label {
                        grad[[i, k]] = probs[[i, k]];
                    }
                }
            }
        }

        // 4. 根据 reduction 归一化
        let valid_count = labels
            .iter()
            .filter(|&&l| l != self.config.ignore_index && l < vocab_size)
            .count()
            .max(1);

        if self.config.reduction == Reduction::Mean {
            grad.mapv_inplace(|x| x / valid_count as f32);
        }

        grad
    }
}

/// 数值稳定的 softmax（沿最后一维）
pub fn softmax(logits: &Array2<f32>) -> Array2<f32> {
    let (n, v) = logits.dim();
    let mut result = Array2::<f32>::zeros((n, v));

    for i in 0..n {
        // 减去最大值保证数值稳定
        let max_val = logits
            .row(i)
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        let exp_vals: Vec<f32> = (0..v).map(|j| (logits[[i, j]] - max_val).exp()).collect();

        let sum_exp: f32 = exp_vals.iter().sum();

        for j in 0..v {
            result[[i, j]] = exp_vals[j] / sum_exp;
        }
    }

    result
}

/// Log Softmax（更稳定的版本）
pub fn log_softmax(logits: &Array2<f32>) -> Array2<f32> {
    let (n, v) = logits.dim();
    let mut result = Array2::<f32>::zeros((n, v));

    for i in 0..n {
        let max_val = logits
            .row(i)
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        let exp_sum: f32 = (0..v).map(|j| (logits[[i, j]] - max_val).exp()).sum();

        let log_sum_exp = exp_sum.ln();

        for j in 0..v {
            result[[i, j]] = logits[[i, j]] - max_val - log_sum_exp;
        }
    }

    result
}

/// One-hot 编码
pub fn one_hot(labels: &[usize], num_classes: usize) -> Array2<f32> {
    let n = labels.len();
    let mut result = Array2::<f32>::zeros((n, num_classes));

    for (i, &label) in labels.iter().enumerate() {
        if label < num_classes {
            result[[i, label]] = 1.0;
        }
    }

    result
}

/// 从 CrossEntropy Loss 计算 Perplexity
///
/// PPL = exp(loss)
/// 是评估语言模型生成质量的常用指标
pub fn perplexity(loss: f64) -> f64 {
    loss.exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn create_test_logits() -> Array2<f32> {
        array![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0],]
    }

    fn create_test_labels() -> Vec<usize> {
        vec![2, 2, 0]
    }

    #[test]
    fn test_cross_entropy_basic() {
        let loss_fn = CrossEntropyLoss::with_defaults();
        let logits = create_test_logits();
        let labels = create_test_labels();

        let output = loss_fn.forward(&logits, &labels);

        assert!(output.loss > 0.0, "Loss should be positive");
        assert_eq!(output.num_valid_tokens, 3);
    }

    #[test]
    fn test_cross_entropy_perfect_prediction() {
        let loss_fn = CrossEntropyLoss::with_defaults();

        // 高置信度的正确预测
        let logits = array![[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
        let labels = vec![0, 1, 2];

        let output = loss_fn.forward(&logits, &labels);

        assert!(
            output.loss < 0.01,
            "Perfect prediction should have near-zero loss, got {}",
            output.loss
        );
    }

    #[test]
    fn test_cross_entropy_with_ignore_index() {
        let config = CrossEntropyConfig {
            ignore_index: usize::MAX,
            ..Default::default()
        };
        let loss_fn = CrossEntropyLoss::new(config);

        let logits = array![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]];
        let labels = vec![2, usize::MAX]; // 第二个是 padding

        let output = loss_fn.forward(&logits, &labels);

        assert_eq!(output.num_valid_tokens, 1);
        assert!(output.loss > 0.0);
    }

    #[test]
    fn test_label_smoothing_effect() {
        let config_no_smooth = CrossEntropyConfig {
            label_smoothing: 0.0,
            ..Default::default()
        };
        let config_with_smooth = CrossEntropyConfig {
            label_smoothing: 0.1,
            ..Default::default()
        };

        let loss_fn_no_smooth = CrossEntropyLoss::new(config_no_smooth);
        let loss_fn_with_smooth = CrossEntropyLoss::new(config_with_smooth);

        let logits = array![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]];
        let labels = vec![2, 2];

        let output_no_smooth = loss_fn_no_smooth.forward(&logits, &labels);
        let output_with_smooth = loss_fn_with_smooth.forward(&logits, &labels);

        assert!(
            output_with_smooth.loss > output_no_smooth.loss,
            "Label smoothing should increase loss"
        );
        assert!(
            output_with_smooth.loss - output_no_smooth.loss < 0.5,
            "Smoothing should not increase loss too much"
        );
    }

    #[test]
    fn test_reduction_modes() {
        let logits = array![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]];
        let labels = vec![2, 2];

        // Mean
        let config_mean = CrossEntropyConfig {
            reduction: Reduction::Mean,
            ..Default::default()
        };
        let output_mean = CrossEntropyLoss::new(config_mean).forward(&logits, &labels);
        assert!(output_mean.individual_losses.is_none());

        // Sum
        let config_sum = CrossEntropyConfig {
            reduction: Reduction::Sum,
            ..Default::default()
        };
        let output_sum = CrossEntropyLoss::new(config_sum).forward(&logits, &labels);
        assert!(
            output_sum.loss > output_mean.loss,
            "Sum should be >= mean for positive losses"
        );

        // None
        let config_none = CrossEntropyConfig {
            reduction: Reduction::None,
            ..Default::default()
        };
        let output_none = CrossEntropyLoss::new(config_none).forward(&logits, &labels);
        assert!(output_none.individual_losses.is_some());
        assert_eq!(output_none.individual_losses.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_backward_gradient_shape() {
        let loss_fn = CrossEntropyLoss::with_defaults();
        let logits = create_test_logits();
        let labels = create_test_labels();

        let grad = loss_fn.backward(&logits, &labels);

        assert_eq!(grad.shape(), logits.shape());
    }

    #[test]
    fn test_backward_gradient_correctness() {
        let loss_fn = CrossEntropyLoss::with_defaults();
        let logits = array![[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]];
        let labels = vec![2, 2];

        let grad = loss_fn.backward(&logits, &labels);

        // 使用有限差分法验证数值梯度
        let eps = 1e-4_f32;
        let mut numerical_grad = Array2::<f32>::zeros((2, 3));

        for i in 0..2 {
            for j in 0..3 {
                let mut logits_plus = logits.clone();
                logits_plus[[i, j]] += eps;
                let loss_plus = loss_fn.forward(&logits_plus, &labels).loss;

                let mut logits_minus = logits.clone();
                logits_minus[[i, j]] -= eps;
                let loss_minus = loss_fn.forward(&logits_minus, &labels).loss;

                numerical_grad[[i, j]] = ((loss_plus - loss_minus) / (2.0 * eps as f64)) as f32;
            }
        }

        // 验证解析梯度和数值梯度的差异
        let mut max_error = 0.0_f32;
        for i in 0..2 {
            for j in 0..3 {
                let error = (grad[[i, j]] - numerical_grad[[i, j]]).abs();
                if error > max_error {
                    max_error = error;
                }
            }
        }

        assert!(max_error < 1e-3, "Gradient error too large: {}", max_error);
    }

    #[test]
    fn test_perplexity_calculation() {
        assert!((perplexity(0.0) - 1.0).abs() < 1e-6);
        assert!((perplexity(1.0) - std::f64::consts::E).abs() < 1e-6);
        // perplexity(x) = exp(x), so perplexity(2.0) = e^2
        assert!((perplexity(2.0) - std::f64::consts::E.powi(2)).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // 大数值输入
        let large_logits = array![[1000.0, 1001.0, 1002.0]];
        let probs = softmax(&large_logits);

        for &p in probs.iter() {
            assert!(p.is_finite(), "Softmax should handle large values");
            assert!(
                (0.0..=1.0).contains(&p),
                "Probabilities should be in [0, 1]"
            );
        }

        // 负数值输入
        let neg_logits = array![[-1000.0, -999.0, -998.0]];
        let neg_probs = softmax(&neg_logits);

        for &p in neg_probs.iter() {
            assert!(p.is_finite(), "Softmax should handle negative values");
        }
    }

    #[test]
    fn test_log_softmax_vs_manual() {
        let logits = array![[1.0, 2.0, 3.0]];

        let ls = log_softmax(&logits);
        let sm = softmax(&logits);

        for i in 0..3 {
            let manual_log = sm[[0, i]].ln();
            assert!(
                (ls[[0, i]] - manual_log).abs() < 1e-5,
                "Log softmax should match log of softmax"
            );
        }
    }

    #[test]
    fn test_one_hot_encoding() {
        let labels = vec![0, 2, 1];
        let encoded = one_hot(&labels, 3);

        assert_eq!(encoded.shape(), &[3, 3]);
        assert!((encoded[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((encoded[[0, 1]] - 0.0).abs() < 1e-6);
        assert!((encoded[[1, 2]] - 1.0).abs() < 1e-6);
        assert!((encoded[[2, 1]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_input_handling() {
        let loss_fn = CrossEntropyLoss::with_defaults();
        let logits: Array2<f32> = Array2::zeros((0, 3));
        let labels: Vec<usize> = vec![];

        let output = loss_fn.forward(&logits, &labels);

        assert_eq!(output.loss, 0.0);
        assert_eq!(output.num_valid_tokens, 0);
    }

    #[test]
    #[should_panic(expected = "Label")]
    fn test_vocabulary_size_mismatch() {
        let loss_fn = CrossEntropyLoss::with_defaults();
        let logits = array![[1.0, 2.0, 3.0]]; // vocab_size=3
        let labels = vec![5]; // label > vocab_size

        loss_fn.forward(&logits, &labels);
    }

    #[test]
    fn test_all_ignored_labels() {
        let config = CrossEntropyConfig {
            ignore_index: usize::MAX,
            ..Default::default()
        };
        let loss_fn = CrossEntropyLoss::new(config);

        let logits = array![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]];
        let labels = vec![usize::MAX, usize::MAX];

        let output = loss_fn.forward(&logits, &labels);

        assert_eq!(output.loss, 0.0);
        assert_eq!(output.num_valid_tokens, 0);
    }
}
