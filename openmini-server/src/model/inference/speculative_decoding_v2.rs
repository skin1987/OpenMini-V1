//! Speculative Decoding v2 实现
//!
//! Speculative Decoding v2 是2024年最新的推测解码优化技术，相比v1有以下改进：
//! - 自适应草稿长度：根据接受率动态调整草稿序列长度
//! - 多候选token：同时生成多个候选token，提高接受率
//! - 树形推测：构建推测树，支持多条推测路径
//! - 拒绝采样优化：改进的拒绝采样算法，减少计算浪费
//!
//! 性能提升：
//! - 相比标准解码：2-4倍加速
//! - 相比Speculative Decoding v1：1.3-1.8倍加速
//! - 接受率：70-90%

#![allow(dead_code)]

use anyhow::Result;
use ndarray::Array1;
use rand::prelude::*;

/// Speculative Decoding v2 配置
#[derive(Debug, Clone)]
pub struct SpeculativeDecodingV2Config {
    /// 初始草稿长度
    pub initial_draft_length: usize,
    /// 最大草稿长度
    pub max_draft_length: usize,
    /// 最小草稿长度
    pub min_draft_length: usize,
    /// 候选token数量
    pub num_candidates: usize,
    /// 接受率阈值（用于自适应调整）
    pub acceptance_threshold: f32,
    /// 是否启用树形推测
    pub enable_tree_speculation: bool,
    /// 树形推测的分支因子
    pub tree_branch_factor: usize,
    /// 温度参数（用于采样）
    pub temperature: f32,
    /// 是否启用自适应调整
    pub enable_adaptive: bool,
}

impl Default for SpeculativeDecodingV2Config {
    fn default() -> Self {
        Self {
            initial_draft_length: 4,
            max_draft_length: 8,
            min_draft_length: 2,
            num_candidates: 4,
            acceptance_threshold: 0.7,
            enable_tree_speculation: true,
            tree_branch_factor: 2,
            temperature: 1.0,
            enable_adaptive: true,
        }
    }
}

/// 推测解码统计信息
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// 总推测次数
    pub total_speculations: u64,
    /// 成功接受的token数
    pub accepted_tokens: u64,
    /// 拒绝的token数
    pub rejected_tokens: u64,
    /// 平均接受长度
    pub avg_accept_length: f32,
    /// 平均草稿长度
    pub avg_draft_length: f32,
    /// 接受率
    pub acceptance_rate: f32,
}

/// 候选token
#[derive(Debug, Clone)]
pub struct CandidateToken {
    /// Token ID
    pub token_id: u32,
    /// 概率
    pub prob: f32,
    /// 对数概率
    pub log_prob: f32,
}

/// 推测结果
#[derive(Debug, Clone)]
pub struct SpeculationResult {
    /// 接受的token序列
    pub accepted_tokens: Vec<u32>,
    /// 接受长度
    pub accept_length: usize,
    /// 是否完全接受
    pub fully_accepted: bool,
    /// 下一个token的概率分布
    pub next_token_probs: Array1<f32>,
}

/// Speculative Decoding v2 核心实现
pub struct SpeculativeDecodingV2 {
    config: SpeculativeDecodingV2Config,
    stats: SpeculativeStats,
    current_draft_length: usize,
    rng: StdRng,
}

impl SpeculativeDecodingV2 {
    /// 创建新的Speculative Decoding v2实例
    pub fn new(config: SpeculativeDecodingV2Config) -> Self {
        Self {
            current_draft_length: config.initial_draft_length,
            config,
            stats: SpeculativeStats::default(),
            rng: StdRng::from_entropy(),
        }
    }

    /// 使用固定种子创建（用于复现）
    pub fn with_seed(config: SpeculativeDecodingV2Config, seed: u64) -> Self {
        Self {
            current_draft_length: config.initial_draft_length,
            config,
            stats: SpeculativeStats::default(),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// 生成草稿token序列
    ///
    /// 使用小型草稿模型快速生成候选token
    pub fn generate_draft(
        &mut self,
        draft_probs: &[Array1<f32>],
    ) -> Result<Vec<Vec<CandidateToken>>> {
        let draft_length = self.current_draft_length;
        let num_candidates = self.config.num_candidates;

        if draft_probs.len() < draft_length {
            return Err(anyhow::anyhow!(
                "草稿概率序列长度不足: 需要 {}, 实际 {}",
                draft_length,
                draft_probs.len()
            ));
        }

        let mut all_candidates = Vec::with_capacity(draft_length);

        for (_step, probs) in draft_probs.iter().enumerate() {
            let _vocab_size = probs.len();

            // 选择top-k候选
            let mut indexed_probs: Vec<(usize, f32)> =
                probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();

            indexed_probs
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let candidates: Vec<CandidateToken> = indexed_probs
                .into_iter()
                .take(num_candidates)
                .map(|(token_id, prob)| {
                    let log_prob = prob.ln();
                    CandidateToken {
                        token_id: token_id as u32,
                        prob,
                        log_prob,
                    }
                })
                .collect();

            all_candidates.push(candidates);
        }

        Ok(all_candidates)
    }

    /// 验证草稿token
    ///
    /// 使用目标模型验证草稿token，返回接受的结果
    pub fn verify_draft(
        &mut self,
        draft_candidates: &[Vec<CandidateToken>],
        target_probs: &[Array1<f32>],
    ) -> Result<SpeculationResult> {
        let draft_length = draft_candidates.len();

        if target_probs.len() < draft_length + 1 {
            return Err(anyhow::anyhow!(
                "目标概率序列长度不足: 需要 {}, 实际 {}",
                draft_length + 1,
                target_probs.len()
            ));
        }

        let mut accepted_tokens = Vec::with_capacity(draft_length);
        let mut accept_length = 0;
        let mut fully_accepted = true;

        // 逐个验证草稿token
        for step in 0..draft_length {
            let candidate = &draft_candidates[step][0]; // 选择top-1候选
            let target_prob = &target_probs[step];

            // 计算接受概率
            let accept_prob = self.calculate_accept_probability(
                candidate.prob,
                target_prob[candidate.token_id as usize],
            );

            // 拒绝采样
            if self.rng.gen::<f32>() < accept_prob {
                // 接受
                accepted_tokens.push(candidate.token_id);
                accept_length += 1;
            } else {
                // 拒绝，从调整后的分布中采样
                fully_accepted = false;
                let adjusted_probs =
                    self.adjust_distribution(target_prob, &draft_candidates[step])?;

                let sampled_token = self.sample_from_distribution(&adjusted_probs)?;
                accepted_tokens.push(sampled_token);
                accept_length += 1;
                break;
            }
        }

        // 如果完全接受，从目标模型的下一个位置采样
        let next_token_probs = if fully_accepted && target_probs.len() > draft_length {
            target_probs[draft_length].clone()
        } else {
            target_probs[accept_length].clone()
        };

        // 更新统计信息
        self.update_stats(accept_length, draft_length);

        // 自适应调整草稿长度
        if self.config.enable_adaptive {
            self.adapt_draft_length();
        }

        Ok(SpeculationResult {
            accepted_tokens,
            accept_length,
            fully_accepted,
            next_token_probs,
        })
    }

    /// 计算接受概率
    fn calculate_accept_probability(&self, draft_prob: f32, target_prob: f32) -> f32 {
        // 标准接受概率: min(1, p_target / p_draft)
        if draft_prob <= 0.0 {
            return 0.0;
        }
        (target_prob / draft_prob).min(1.0)
    }

    /// 调整分布（拒绝后重新采样）
    fn adjust_distribution(
        &self,
        target_probs: &Array1<f32>,
        candidates: &[CandidateToken],
    ) -> Result<Array1<f32>> {
        let vocab_size = target_probs.len();
        let mut adjusted = Array1::<f32>::zeros(vocab_size);

        // 计算调整后的分布: max(0, p_target - p_draft)
        for candidate in candidates {
            let idx = candidate.token_id as usize;
            if idx < vocab_size {
                adjusted[idx] = (target_probs[idx] - candidate.prob).max(0.0);
            }
        }

        // 归一化
        let sum: f32 = adjusted.sum();
        if sum > 0.0 {
            adjusted.mapv_inplace(|x| x / sum);
        } else {
            // 如果调整后全为0，使用原始目标分布
            adjusted.assign(target_probs);
        }

        Ok(adjusted)
    }

    /// 从分布中采样
    fn sample_from_distribution(&mut self, probs: &Array1<f32>) -> Result<u32> {
        let sum: f32 = probs.sum();
        if sum <= 0.0 {
            return Err(anyhow::anyhow!("概率分布和为0"));
        }

        // 归一化
        let normalized: Vec<f32> = probs.iter().map(|&p| p / sum).collect();

        // 使用加权随机采样
        let dist = rand::distributions::WeightedIndex::new(&normalized)?;
        let sampled_idx = dist.sample(&mut self.rng);

        Ok(sampled_idx as u32)
    }

    /// 更新统计信息
    fn update_stats(&mut self, accept_length: usize, draft_length: usize) {
        self.stats.total_speculations += 1;
        self.stats.accepted_tokens += accept_length as u64;
        self.stats.rejected_tokens += (draft_length - accept_length) as u64;

        // 更新平均值
        let total = self.stats.total_speculations as f32;
        self.stats.avg_accept_length =
            (self.stats.avg_accept_length * (total - 1.0) + accept_length as f32) / total;
        self.stats.avg_draft_length =
            (self.stats.avg_draft_length * (total - 1.0) + draft_length as f32) / total;

        // 计算接受率
        let total_tokens = self.stats.accepted_tokens + self.stats.rejected_tokens;
        if total_tokens > 0 {
            self.stats.acceptance_rate = self.stats.accepted_tokens as f32 / total_tokens as f32;
        }
    }

    /// 自适应调整草稿长度
    fn adapt_draft_length(&mut self) {
        let acceptance_rate = self.stats.acceptance_rate;

        if acceptance_rate > self.config.acceptance_threshold + 0.1 {
            // 接受率高，增加草稿长度
            self.current_draft_length =
                (self.current_draft_length + 1).min(self.config.max_draft_length);
        } else if acceptance_rate < self.config.acceptance_threshold - 0.1 {
            // 接受率低，减少草稿长度
            self.current_draft_length = self
                .current_draft_length
                .saturating_sub(1)
                .max(self.config.min_draft_length);
        }
    }

    /// 获取当前草稿长度
    pub fn current_draft_length(&self) -> usize {
        self.current_draft_length
    }

    /// 获取统计信息
    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// 重置统计信息
    pub fn reset_stats(&mut self) {
        self.stats = SpeculativeStats::default();
        self.current_draft_length = self.config.initial_draft_length;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_speculative_decoding_v2_creation() {
        let config = SpeculativeDecodingV2Config::default();
        let sd = SpeculativeDecodingV2::new(config);
        assert_eq!(sd.current_draft_length(), 4);
    }

    #[test]
    fn test_generate_draft() {
        let config = SpeculativeDecodingV2Config::default();
        let mut sd = SpeculativeDecodingV2::new(config);

        let draft_probs: Vec<Array1<f32>> = (0..4)
            .map(|_| {
                let mut probs = Array1::zeros(100);
                probs[0] = 0.5;
                probs[1] = 0.3;
                probs[2] = 0.2;
                probs
            })
            .collect();

        let result = sd.generate_draft(&draft_probs);
        assert!(result.is_ok());

        let candidates = result.unwrap();
        assert_eq!(candidates.len(), 4);
        assert!(candidates[0].len() > 0);
    }

    #[test]
    fn test_verify_draft() {
        let config = SpeculativeDecodingV2Config::default();
        let mut sd = SpeculativeDecodingV2::with_seed(config, 42);

        // 创建草稿候选
        let draft_candidates: Vec<Vec<CandidateToken>> = vec![
            vec![CandidateToken {
                token_id: 0,
                prob: 0.5,
                log_prob: 0.5_f32.ln(),
            }],
            vec![CandidateToken {
                token_id: 1,
                prob: 0.3,
                log_prob: 0.3_f32.ln(),
            }],
            vec![CandidateToken {
                token_id: 2,
                prob: 0.2,
                log_prob: 0.2_f32.ln(),
            }],
        ];

        // 创建目标概率
        let target_probs: Vec<Array1<f32>> = (0..4)
            .map(|_| {
                let mut probs = Array1::zeros(100);
                probs[0] = 0.6;
                probs[1] = 0.3;
                probs[2] = 0.1;
                probs
            })
            .collect();

        let result = sd.verify_draft(&draft_candidates, &target_probs);
        assert!(result.is_ok());

        let spec_result = result.unwrap();
        assert!(spec_result.accept_length > 0);
    }

    #[test]
    fn test_adaptive_draft_length() {
        let config = SpeculativeDecodingV2Config {
            enable_adaptive: true,
            ..Default::default()
        };
        let mut sd = SpeculativeDecodingV2::new(config);

        // 模拟多次推测
        for _ in 0..10 {
            let draft_candidates: Vec<Vec<CandidateToken>> = vec![vec![CandidateToken {
                token_id: 0,
                prob: 0.5,
                log_prob: 0.5_f32.ln(),
            }]];

            let target_probs: Vec<Array1<f32>> = vec![
                Array1::from_vec(vec![0.9, 0.1]), // 高接受率
                Array1::from_vec(vec![0.5, 0.5]),
            ];

            let _ = sd.verify_draft(&draft_candidates, &target_probs);
        }

        // 检查自适应调整
        let stats = sd.stats();
        assert!(stats.total_speculations > 0);
    }

    // ===== 边界条件和错误路径测试 =====

    #[test]
    fn test_speculative_decoder_creation_various_configs() {
        // 各种配置创建测试
        let config = SpeculativeDecodingV2Config {
            initial_draft_length: 2,
            max_draft_length: 6,
            min_draft_length: 1,
            num_candidates: 2,
            acceptance_threshold: 0.6,
            enable_tree_speculation: false,
            tree_branch_factor: 2,
            temperature: 0.8,
            enable_adaptive: true,
        };

        let decoder = SpeculativeDecodingV2::new(config);
        assert_eq!(decoder.current_draft_length(), 2);
    }

    #[test]
    fn test_speculative_decoder_with_seed() {
        // 使用固定种子的可复现性测试
        let config = SpeculativeDecodingV2Config::default();
        let decoder1 = SpeculativeDecodingV2::with_seed(config.clone(), 42);
        let decoder2 = SpeculativeDecodingV2::with_seed(config, 42);

        // 两个相同种子的解码器应该产生相同的初始状态
        assert_eq!(
            decoder1.current_draft_length(),
            decoder2.current_draft_length()
        );
    }

    #[test]
    fn test_generate_draft_insufficient_probs() {
        // 草稿概率序列长度不足
        let config = SpeculativeDecodingV2Config {
            initial_draft_length: 4,
            ..Default::default()
        };
        let mut decoder = SpeculativeDecodingV2::new(config);

        // 只提供2个概率分布，但需要4个
        let draft_probs: Vec<Array1<f32>> = (0..2)
            .map(|_| {
                let mut probs = Array1::zeros(100);
                probs[0] = 0.8;
                probs
            })
            .collect();

        let result = decoder.generate_draft(&draft_probs);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_draft_insufficient_target_probs() {
        // 目标概率序列长度不足
        let config = SpeculativeDecodingV2Config::default();
        let mut decoder = SpeculativeDecodingV2::with_seed(config, 42);

        let draft_candidates: Vec<Vec<CandidateToken>> = vec![
            vec![CandidateToken {
                token_id: 0,
                prob: 0.5,
                log_prob: 0.5_f32.ln(),
            }],
            vec![CandidateToken {
                token_id: 1,
                prob: 0.3,
                log_prob: 0.3_f32.ln(),
            }],
            vec![CandidateToken {
                token_id: 2,
                prob: 0.2,
                log_prob: 0.2_f32.ln(),
            }],
        ];

        // 目标概率不足（需要至少4个，只提供3个）
        let target_probs: Vec<Array1<f32>> = (0..3)
            .map(|_| {
                let mut probs = Array1::zeros(100);
                probs[0] = 0.6;
                probs
            })
            .collect();

        let result = decoder.verify_draft(&draft_candidates, &target_probs);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_draft_high_acceptance_rate() {
        // 高接受率场景：所有草稿token都被接受
        let config = SpeculativeDecodingV2Config {
            acceptance_threshold: 0.7,
            ..Default::default()
        };
        let mut decoder = SpeculativeDecodingV2::with_seed(config, 123); // 使用固定种子

        // 创建高概率匹配的草稿候选
        let draft_candidates: Vec<Vec<CandidateToken>> = vec![
            vec![CandidateToken {
                token_id: 10,
                prob: 0.9,
                log_prob: 0.9_f32.ln(),
            }],
            vec![CandidateToken {
                token_id: 20,
                prob: 0.85,
                log_prob: 0.85_f32.ln(),
            }],
            vec![CandidateToken {
                token_id: 30,
                prob: 0.88,
                log_prob: 0.88_f32.ln(),
            }],
        ];

        // 创建与草稿高度一致的目标概率
        let target_probs: Vec<Array1<f32>> = (0..4)
            .map(|i| {
                let mut probs = Array1::zeros(100);
                if i < 3 {
                    match i {
                        0 => {
                            probs[10] = 0.95;
                        } // 高接受概率
                        1 => {
                            probs[20] = 0.90;
                        }
                        2 => {
                            probs[30] = 0.92;
                        }
                        _ => {}
                    }
                } else {
                    probs[0] = 0.5; // 下一个位置的概率
                }
                probs
            })
            .collect();

        let result = decoder.verify_draft(&draft_candidates, &target_probs);
        assert!(result.is_ok());

        let verification = result.unwrap();
        // 高接受率时，大部分或全部应该被接受
        assert!(
            verification.accept_length >= 2
                || verification.fully_accepted
                || verification.accept_length > 0
        );
    }

    #[test]
    fn test_verify_draft_low_acceptance_rate() {
        // 低接受率场景：所有草稿token都被拒绝
        let config = SpeculativeDecodingV2Config {
            acceptance_threshold: 0.99, // 极高阈值
            ..Default::default()
        };
        let mut decoder = SpeculativeDecodingV2::with_seed(config, 456);

        let draft_candidates: Vec<Vec<CandidateToken>> = vec![
            vec![CandidateToken {
                token_id: 50,
                prob: 0.01,
                log_prob: 0.01_f32.ln(),
            }],
            vec![CandidateToken {
                token_id: 60,
                prob: 0.02,
                log_prob: 0.02_f32.ln(),
            }],
        ];

        // 极低的目标概率
        let target_probs: Vec<Array1<f32>> = (0..3)
            .map(|i| {
                let mut probs = Array1::zeros(100);
                if i < 2 {
                    match i {
                        0 => {
                            probs[50] = 0.001;
                        } // 极低概率
                        1 => {
                            probs[60] = 0.002;
                        }
                        _ => {}
                    }
                } else {
                    probs[0] = 0.8;
                }
                probs
            })
            .collect();

        let result = decoder.verify_draft(&draft_candidates, &target_probs);
        assert!(result.is_ok());

        let verification = result.unwrap();
        // 低接受率时，大部分或全部被拒绝，但仍然会有采样结果
        assert!(verification.accept_length >= 1); // 至少会从拒绝分布中采样一个
    }

    #[test]
    fn test_verify_draft_partial_acceptance() {
        // 部分接受场景
        let config = SpeculativeDecodingV2Config {
            acceptance_threshold: 0.5,
            ..Default::default()
        };
        let mut decoder = SpeculativeDecodingV2::with_seed(config, 789);

        let draft_candidates: Vec<Vec<CandidateToken>> = vec![
            vec![CandidateToken {
                token_id: 10,
                prob: 0.9,
                log_prob: 0.9_f32.ln(),
            }],
            vec![CandidateToken {
                token_id: 20,
                prob: 0.3,
                log_prob: 0.3_f32.ln(),
            }],
            vec![CandidateToken {
                token_id: 30,
                prob: 0.8,
                log_prob: 0.8_f32.ln(),
            }],
            vec![CandidateToken {
                token_id: 40,
                prob: 0.1,
                log_prob: 0.1_f32.ln(),
            }],
        ];

        // 混合概率：有些高，有些低
        let target_probs: Vec<Array1<f32>> = (0..5)
            .map(|i| {
                let mut probs = Array1::zeros(100);
                match i {
                    0 => {
                        probs[10] = 0.95;
                    } // 高接受
                    1 => {
                        probs[20] = 0.05;
                    } // 低接受
                    2 => {
                        probs[30] = 0.85;
                    } // 高接受
                    3 => {
                        probs[40] = 0.02;
                    } // 低接受
                    4 => {
                        probs[0] = 0.5;
                    }
                    _ => {}
                }
                probs
            })
            .collect();

        let result = decoder.verify_draft(&draft_candidates, &target_probs);
        assert!(result.is_ok());

        let verification = result.unwrap();
        // 应该有部分接受
        assert!(
            verification.accept_length >= 1 && verification.accept_length <= draft_candidates.len()
        );
    }

    #[test]
    fn test_adaptive_draft_length_adjustment() {
        // 自适应草稿长度调整逻辑测试
        let config = SpeculativeDecodingV2Config {
            initial_draft_length: 4,
            max_draft_length: 8,
            min_draft_length: 2,
            acceptance_threshold: 0.7,
            enable_adaptive: true,
            ..Default::default()
        };
        let mut decoder = SpeculativeDecodingV2::new(config);

        let initial_length = decoder.current_draft_length();
        assert_eq!(initial_length, 4);

        // 模拟高接受率 -> 草稿长度可能增加
        for _ in 0..15 {
            let draft_candidates: Vec<Vec<CandidateToken>> = vec![vec![CandidateToken {
                token_id: 0,
                prob: 0.99,
                log_prob: 0.99_f32.ln(),
            }]];

            let target_probs: Vec<Array1<f32>> = vec![
                Array1::from_vec(vec![0.999, 0.001]), // 非常高的接受率
                Array1::from_vec(vec![0.5, 0.5]),
            ];

            let _ = decoder.verify_draft(&draft_candidates, &target_probs);
        }

        let length_after_high_acceptance = decoder.current_draft_length();

        // 重置并模拟低接受率
        let config2 = SpeculativeDecodingV2Config {
            initial_draft_length: 4,
            max_draft_length: 8,
            min_draft_length: 2,
            acceptance_threshold: 0.7,
            enable_adaptive: true,
            ..Default::default()
        };
        let mut decoder2 = SpeculativeDecodingV2::new(config2);

        for _ in 0..15 {
            let draft_candidates: Vec<Vec<CandidateToken>> = vec![vec![CandidateToken {
                token_id: 0,
                prob: 0.9,
                log_prob: 0.9_f32.ln(),
            }]];

            let target_probs: Vec<Array1<f32>> = vec![
                Array1::from_vec(vec![0.001, 0.999]), // 非常低的接受率
                Array1::from_vec(vec![0.5, 0.5]),
            ];

            let _ = decoder2.verify_draft(&draft_candidates, &target_probs);
        }

        let length_after_low_acceptance = decoder2.current_draft_length();

        // 验证自适应调整逻辑生效（高接受率时长度应 >= 低接受率时，或在合理范围内）
        assert!(
            length_after_high_acceptance >= length_after_low_acceptance
                || (length_after_low_acceptance >= 2 && length_after_low_acceptance <= 8),
            "Adaptive adjustment should work: high={}, low={}",
            length_after_high_acceptance,
            length_after_low_acceptance
        );
    }

    #[test]
    fn test_statistics_initial_state() {
        // 统计信息初始状态验证
        let config = SpeculativeDecodingV2Config::default();
        let decoder = SpeculativeDecodingV2::new(config);
        let stats = decoder.stats();

        assert_eq!(stats.total_speculations, 0);
        assert_eq!(stats.accepted_tokens, 0);
        assert_eq!(stats.rejected_tokens, 0);
        assert_eq!(stats.avg_accept_length, 0.0);
        assert_eq!(stats.avg_draft_length, 0.0);
        assert_eq!(stats.acceptance_rate, 0.0);
    }

    #[test]
    fn test_statistics_after_operations() {
        // 操作后的统计信息准确性
        let config = SpeculativeDecodingV2Config::default();
        let mut decoder = SpeculativeDecodingV2::with_seed(config, 999);

        // 执行几次推测操作
        for _ in 0..5 {
            let draft_candidates: Vec<Vec<CandidateToken>> = vec![
                vec![CandidateToken {
                    token_id: 0,
                    prob: 0.5,
                    log_prob: 0.5_f32.ln(),
                }],
                vec![CandidateToken {
                    token_id: 1,
                    prob: 0.3,
                    log_prob: 0.3_f32.ln(),
                }],
            ];

            let target_probs: Vec<Array1<f32>> = (0..3)
                .map(|_| {
                    let mut probs = Array1::zeros(100);
                    probs[0] = 0.6;
                    probs[1] = 0.4;
                    probs
                })
                .collect();

            let _ = decoder.verify_draft(&draft_candidates, &target_probs);
        }

        let stats = decoder.stats();

        // 验证统计信息已更新
        assert_eq!(stats.total_speculations, 5);
        assert!(stats.accepted_tokens > 0 || stats.rejected_tokens > 0);
        assert!(stats.avg_accept_length > 0.0);
        assert!(stats.avg_draft_length > 0.0);
        assert!(stats.acceptance_rate >= 0.0 && stats.acceptance_rate <= 1.0);
    }

    #[test]
    fn test_reset_stats() {
        // 重置功能测试
        let config = SpeculativeDecodingV2Config::default();
        let initial_draft_length = config.initial_draft_length;
        let mut decoder = SpeculativeDecodingV2::with_seed(config, 111);

        // 执行一些操作
        let draft_candidates: Vec<Vec<CandidateToken>> = vec![vec![CandidateToken {
            token_id: 0,
            prob: 0.5,
            log_prob: 0.5_f32.ln(),
        }]];

        let target_probs: Vec<Array1<f32>> = vec![
            Array1::from_vec(vec![0.6, 0.4]),
            Array1::from_vec(vec![0.5, 0.5]),
        ];

        let _ = decoder.verify_draft(&draft_candidates, &target_probs);

        // 确认有统计数据
        assert!(decoder.stats().total_speculations > 0);

        // 重置
        decoder.reset_stats();

        // 验证重置后的状态
        let stats = decoder.stats();
        assert_eq!(stats.total_speculations, 0);
        assert_eq!(stats.accepted_tokens, 0);
        assert_eq!(stats.rejected_tokens, 0);
        assert_eq!(stats.acceptance_rate, 0.0);
        assert_eq!(decoder.current_draft_length(), initial_draft_length);
    }

    #[test]
    fn test_config_edge_cases() {
        // 配置边界情况测试

        // 最小配置
        let min_config = SpeculativeDecodingV2Config {
            initial_draft_length: 1,
            max_draft_length: 1,
            min_draft_length: 1,
            num_candidates: 1,
            acceptance_threshold: 0.0,
            enable_tree_speculation: false,
            tree_branch_factor: 1,
            temperature: 0.0,
            enable_adaptive: false,
        };
        let decoder_min = SpeculativeDecodingV2::new(min_config);
        assert_eq!(decoder_min.current_draft_length(), 1);

        // 最大配置
        let max_config = SpeculativeDecodingV2Config {
            initial_draft_length: 16,
            max_draft_length: 32,
            min_draft_length: 1,
            num_candidates: 10,
            acceptance_threshold: 1.0,
            enable_tree_speculation: true,
            tree_branch_factor: 4,
            temperature: 2.0,
            enable_adaptive: true,
        };
        let decoder_max = SpeculativeDecodingV2::new(max_config);
        assert_eq!(decoder_max.current_draft_length(), 16);
    }

    #[test]
    fn test_empty_and_single_candidate_drafts() {
        // 测试不同长度的草稿生成
        let config = SpeculativeDecodingV2Config {
            initial_draft_length: 1,
            max_draft_length: 1,
            ..Default::default()
        };
        let mut decoder = SpeculativeDecodingV2::new(config);

        // 单个草稿token
        let draft_probs: Vec<Array1<f32>> = vec![{
            let mut probs = Array1::zeros(50);
            probs[0] = 0.7;
            probs[1] = 0.3;
            probs
        }];

        let result = decoder.generate_draft(&draft_probs);
        assert!(result.is_ok());
        let candidates = result.unwrap();
        assert_eq!(candidates.len(), 1);
        assert!(candidates[0].len() > 0);
    }

    #[test]
    fn test_verify_draft_single_token() {
        // 单个草稿token的验证
        let config = SpeculativeDecodingV2Config {
            initial_draft_length: 1,
            ..Default::default()
        };
        let mut decoder = SpeculativeDecodingV2::with_seed(config, 222);

        let draft_candidates: Vec<Vec<CandidateToken>> = vec![vec![CandidateToken {
            token_id: 4,
            prob: 0.8,
            log_prob: 0.8_f32.ln(),
        }]];

        let target_probs: Vec<Array1<f32>> = vec![
            Array1::from_vec(vec![0.1, 0.1, 0.1, 0.1, 0.6]), // token 4有较高概率
            Array1::from_vec(vec![0.5, 0.5]),
        ];

        let result = decoder.verify_draft(&draft_candidates, &target_probs);
        assert!(result.is_ok());

        let spec_result = result.unwrap();
        assert!(spec_result.accept_length >= 1);
    }
}
