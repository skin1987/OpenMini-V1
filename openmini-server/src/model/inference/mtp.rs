//! MTP (Multi-Token Prediction) 并行解码模块
//!
//! 实现多候选token并行预测和投机采样验证。

#![allow(dead_code)]

use anyhow::Result;
use ndarray::Array2;

/// MTP 配置
#[derive(Debug, Clone)]
pub struct MtpConfig {
    /// 候选token数量
    pub num_candidates: usize,
    /// 是否启用投机采样
    pub speculative_sampling: bool,
    /// 验证阈值
    pub acceptance_threshold: f32,
    /// 最大连续接受数
    pub max_consecutive_accepts: usize,
}

impl Default for MtpConfig {
    fn default() -> Self {
        Self {
            num_candidates: 4,
            speculative_sampling: true,
            acceptance_threshold: 0.8,
            max_consecutive_accepts: 8,
        }
    }
}

/// MTP 候选token
#[derive(Debug, Clone)]
pub struct MtpCandidate {
    /// token ID
    pub token_id: usize,
    /// 概率
    pub probability: f32,
    /// 验证分数
    pub verification_score: f32,
}

/// MTP 解码器
#[derive(Debug, Clone)]
pub struct MtpDecoder {
    /// 配置
    config: MtpConfig,
    /// 候选缓存
    candidates: Vec<MtpCandidate>,
    /// 连续接受计数
    consecutive_accepts: usize,
    /// 总生成token数
    total_generated: usize,
    /// 总接受token数
    total_accepted: usize,
}

impl MtpDecoder {
    /// 创建新的 MTP 解码器
    pub fn new(config: MtpConfig) -> Self {
        Self {
            config,
            candidates: Vec::new(),
            consecutive_accepts: 0,
            total_generated: 0,
            total_accepted: 0,
        }
    }
    
    /// 使用默认配置创建
    pub fn default_decoder() -> Self {
        Self::new(MtpConfig::default())
    }
    
    /// 生成候选tokens
    pub fn generate_candidates(&mut self, logits: &Array2<f32>) -> Result<Vec<MtpCandidate>> {
        let (seq_len, _vocab_size) = logits.dim();
        let last_logits = logits.row(seq_len - 1).to_owned();
        
        // 并行计算 top-k 候选
        let mut indexed: Vec<(usize, f32)> = last_logits
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let candidates: Vec<MtpCandidate> = indexed
            .into_iter()
            .take(self.config.num_candidates)
            .map(|(token_id, probability)| MtpCandidate {
                token_id,
                probability,
                verification_score: 0.0,
            })
            .collect();
        
        self.candidates = candidates.clone();
        self.total_generated += candidates.len();
        
        Ok(candidates)
    }
    
    /// 验证候选tokens
    pub fn verify_candidates(
        &mut self,
        candidates: &[MtpCandidate],
        target_logits: &Array2<f32>,
    ) -> Result<Vec<MtpCandidate>> {
        if !self.config.speculative_sampling {
            return Ok(candidates.to_vec());
        }
        
        let (seq_len, _vocab_size) = target_logits.dim();
        let target_probs = target_logits.row(seq_len - 1);
        
        let verified: Vec<MtpCandidate> = candidates
            .iter()
            .map(|c| {
                let target_prob = target_probs.get(c.token_id).copied().unwrap_or(0.0);
                let verification_score = if target_prob > 0.0 {
                    c.probability / target_prob
                } else {
                    0.0
                };
                
                MtpCandidate {
                    token_id: c.token_id,
                    probability: c.probability,
                    verification_score,
                }
            })
            .collect();
        
        Ok(verified)
    }
    
    /// 选择最优候选
    pub fn select_best_candidate(&mut self, candidates: &[MtpCandidate]) -> Option<MtpCandidate> {
        let best = candidates
            .iter()
            .filter(|c| c.verification_score >= self.config.acceptance_threshold)
            .max_by(|a, b| a.verification_score.partial_cmp(&b.verification_score).unwrap());
        
        if best.is_some() {
            self.consecutive_accepts += 1;
            self.total_accepted += 1;
            
            if self.consecutive_accepts >= self.config.max_consecutive_accepts {
                self.consecutive_accepts = 0;
            }
        } else {
            self.consecutive_accepts = 0;
        }
        
        best.cloned()
    }
    
    /// 批量生成候选
    pub fn batch_generate_candidates(
        &mut self,
        batch_logits: &[Array2<f32>],
    ) -> Result<Vec<Vec<MtpCandidate>>> {
        batch_logits
            .iter()
            .map(|logits| self.generate_candidates(logits))
            .collect()
    }
    
    /// 获取接受率
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_generated == 0 {
            0.0
        } else {
            self.total_accepted as f32 / self.total_generated as f32
        }
    }
    
    /// 重置统计
    pub fn reset_stats(&mut self) {
        self.consecutive_accepts = 0;
        self.total_generated = 0;
        self.total_accepted = 0;
    }
    
    /// 获取配置
    pub fn config(&self) -> &MtpConfig {
        &self.config
    }
    
    /// 更新配置
    pub fn set_config(&mut self, config: MtpConfig) {
        self.config = config;
    }
}

/// 投机采样器
#[derive(Debug, Clone)]
pub struct SpeculativeSampler {
    /// MTP 解码器
    decoder: MtpDecoder,
    /// 候选序列
    candidate_sequence: Vec<usize>,
    /// 验证序列
    verification_sequence: Vec<usize>,
}

impl SpeculativeSampler {
    /// 创建新的投机采样器
    pub fn new(config: MtpConfig) -> Self {
        Self {
            decoder: MtpDecoder::new(config),
            candidate_sequence: Vec::new(),
            verification_sequence: Vec::new(),
        }
    }
    
    /// 生成候选序列
    pub fn generate_candidate_sequence(&mut self, logits: &Array2<f32>) -> Result<Vec<usize>> {
        let candidates = self.decoder.generate_candidates(logits)?;
        
        self.candidate_sequence = candidates.iter().map(|c| c.token_id).collect();
        
        Ok(self.candidate_sequence.clone())
    }
    
    /// 验证并接受tokens
    pub fn verify_and_accept(
        &mut self,
        target_logits: &Array2<f32>,
    ) -> Result<Vec<usize>> {
        let candidates: Vec<MtpCandidate> = self.candidate_sequence
            .iter()
            .map(|&token_id| MtpCandidate {
                token_id,
                probability: 1.0,
                verification_score: 0.0,
            })
            .collect();
        
        let verified = self.decoder.verify_candidates(&candidates, target_logits)?;
        
        let accepted: Vec<usize> = verified
            .iter()
            .filter(|c| c.verification_score >= self.decoder.config().acceptance_threshold)
            .map(|c| c.token_id)
            .collect();
        
        self.verification_sequence = accepted.clone();
        
        Ok(accepted)
    }
    
    /// 获取解码器
    pub fn decoder(&self) -> &MtpDecoder {
        &self.decoder
    }
    
    /// 获取解码器（可变）
    pub fn decoder_mut(&mut self) -> &mut MtpDecoder {
        &mut self.decoder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mtp_config_default() {
        let config = MtpConfig::default();
        assert_eq!(config.num_candidates, 4);
        assert!(config.speculative_sampling);
    }

    #[test]
    fn test_mtp_decoder_creation() {
        let decoder = MtpDecoder::default_decoder();
        assert_eq!(decoder.acceptance_rate(), 0.0);
    }

    #[test]
    fn test_generate_candidates() {
        let mut decoder = MtpDecoder::default_decoder();
        let logits = Array2::from_shape_fn((1, 100), |(_i, j)| (j as f32) / 100.0);

        let candidates = decoder.generate_candidates(&logits).unwrap();
        assert_eq!(candidates.len(), 4);

        // 最高概率的应该在候选中
        assert!(candidates.iter().any(|c| c.token_id >= 96));
    }

    #[test]
    fn test_verify_candidates() {
        let mut decoder = MtpDecoder::default_decoder();
        let logits = Array2::from_shape_fn((1, 100), |(_i, j)| (j as f32) / 100.0);

        let candidates = decoder.generate_candidates(&logits).unwrap();
        let verified = decoder.verify_candidates(&candidates, &logits).unwrap();

        assert_eq!(verified.len(), candidates.len());
    }

    #[test]
    fn test_acceptance_rate() {
        let mut decoder = MtpDecoder::default_decoder();

        // 模拟一些生成和接受
        decoder.total_generated = 10;
        decoder.total_accepted = 7;

        let rate = decoder.acceptance_rate();
        assert!((rate - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_speculative_sampler() {
        let sampler = SpeculativeSampler::new(MtpConfig::default());
        assert_eq!(sampler.decoder().acceptance_rate(), 0.0);
    }

    // ========== 新增测试开始 ==========

    /// 测试select_best_candidate：所有候选都满足阈值的情况
    #[test]
    fn test_select_best_candidate_all_pass() {
        let mut decoder = MtpDecoder::new(MtpConfig {
            num_candidates: 3,
            speculative_sampling: true,
            acceptance_threshold: 0.5, // 低阈值
            max_consecutive_accepts: 10,
        });

        let candidates = vec![
            MtpCandidate { token_id: 1, probability: 0.9, verification_score: 0.95 },
            MtpCandidate { token_id: 2, probability: 0.8, verification_score: 0.85 },
            MtpCandidate { token_id: 3, probability: 0.7, verification_score: 0.75 },
        ];

        let best = decoder.select_best_candidate(&candidates).unwrap();

        // 应该选择verification_score最高的候选
        assert_eq!(best.token_id, 1); // score=0.95最高
        assert!((best.verification_score - 0.95).abs() < 1e-6);
    }

    /// 测试select_best_candidate：没有候选满足阈值的情况
    #[test]
    fn test_select_best_candidate_none_pass() {
        let mut decoder = MtpDecoder::new(MtpConfig {
            num_candidates: 3,
            speculative_sampling: true,
            acceptance_threshold: 0.99, // 极高阈值
            max_consecutive_accepts: 10,
        });

        let candidates = vec![
            MtpCandidate { token_id: 1, probability: 0.9, verification_score: 0.5 },
            MtpCandidate { token_id: 2, probability: 0.8, verification_score: 0.3 },
            MtpCandidate { token_id: 3, probability: 0.7, verification_score: 0.1 },
        ];

        let best = decoder.select_best_candidate(&candidates);

        // 所有候选都不满足阈值，应返回None
        assert!(best.is_none());
    }

    /// 测试verify_candidates关闭投机采样的情况
    #[test]
    fn test_verify_candidates_no_speculative() {
        let mut decoder = MtpDecoder::new(MtpConfig {
            num_candidates: 3,
            speculative_sampling: false, // 关闭投机采样
            acceptance_threshold: 0.8,
            max_consecutive_accepts: 10,
        });

        let candidates = vec![
            MtpCandidate { token_id: 1, probability: 0.9, verification_score: 0.0 },
            MtpCandidate { token_id: 2, probability: 0.8, verification_score: 0.0 },
        ];

        let target_logits = Array2::from_shape_fn((1, 50), |(_, j)| j as f32 / 50.0);

        // 关闭投机采样时，应该直接返回原始候选不做验证
        let verified = decoder.verify_candidates(&candidates, &target_logits).unwrap();

        assert_eq!(verified.len(), candidates.len());
        // verification_score应该保持为0（未验证）
        for c in &verified {
            assert!((c.verification_score - 0.0).abs() < 1e-6);
        }
    }

    /// 测试batch_generate_candidates批量生成
    #[test]
    fn test_batch_generate_candidates() {
        let mut decoder = MtpDecoder::default_decoder();

        // 创建3个不同的logits批次
        let batch_logits = vec![
            Array2::from_shape_fn((1, 100), |(_, j)| (j as f32) / 100.0),
            Array2::from_shape_fn((1, 100), |(_, j)| (99 - j) as f32 / 100.0),
            Array2::from_shape_fn((1, 100), |(_, _j)| 0.5),
        ];

        let results = decoder.batch_generate_candidates(&batch_logits).unwrap();

        // 应该返回3个结果
        assert_eq!(results.len(), 3);

        // 每个结果都应该有num_candidates个候选
        for candidates in &results {
            assert_eq!(candidates.len(), 4); // 默认num_candidates=4
        }
    }

    /// 测试reset_stats重置统计信息
    #[test]
    fn test_reset_stats() {
        let mut decoder = MtpDecoder::default_decoder();

        // 模拟一些操作
        decoder.total_generated = 20;
        decoder.total_accepted = 15;
        decoder.consecutive_accepts = 5;

        // 重置统计
        decoder.reset_stats();

        // 验证所有计数器归零
        assert_eq!(decoder.consecutive_accepts, 0);
        assert_eq!(decoder.total_generated, 0);
        assert_eq!(decoder.total_accepted, 0);

        // 接受率应为0（避免除零）
        assert!((decoder.acceptance_rate() - 0.0).abs() < 1e-6);
    }

    /// 测试set_config动态更新配置
    #[test]
    fn test_set_config_dynamically() {
        let mut decoder = MtpDecoder::default_decoder();

        // 验证初始配置
        assert_eq!(decoder.config().num_candidates, 4);
        assert!(decoder.config().speculative_sampling);

        // 更新配置
        let new_config = MtpConfig {
            num_candidates: 8,
            speculative_sampling: false,
            acceptance_threshold: 0.95,
            max_consecutive_accepts: 20,
        };
        decoder.set_config(new_config);

        // 验证新配置生效
        assert_eq!(decoder.config().num_candidates, 8);
        assert!(!decoder.config().speculative_sampling);
        assert!((decoder.config().acceptance_threshold - 0.95).abs() < 1e-6);
        assert_eq!(decoder.config().max_consecutive_accepts, 20);
    }

    /// 测试连续接受计数达到上限时自动重置
    #[test]
    fn test_consecutive_accepts_reset_on_limit() {
        let mut decoder = MtpDecoder::new(MtpConfig {
            num_candidates: 3,
            speculative_sampling: true,
            acceptance_threshold: 0.5,
            max_consecutive_accepts: 3, // 连续接受3次后重置
        });

        let good_candidates = vec![
            MtpCandidate { token_id: 1, probability: 0.9, verification_score: 0.95 },
        ];

        // 第一次接受
        decoder.select_best_candidate(&good_candidates);
        assert_eq!(decoder.consecutive_accepts, 1);

        // 第二次接受
        decoder.select_best_candidate(&good_candidates);
        assert_eq!(decoder.consecutive_accepts, 2);

        // 第三次接受（达到上限）
        decoder.select_best_candidate(&good_candidates);
        // 达到max_consecutive_accepts后应该重置为0
        assert_eq!(decoder.consecutive_accepts, 0);
    }

    /// 测试SpeculativeSampler完整流程
    #[test]
    fn test_speculative_sampler_full_flow() {
        let mut sampler = SpeculativeSampler::new(MtpConfig {
            num_candidates: 3,
            speculative_sampling: true,
            acceptance_threshold: 0.5,
            max_consecutive_accepts: 10,
        });

        // 生成候选序列
        let logits = Array2::from_shape_fn((1, 50), |(_, j)| (j as f32) / 50.0);
        let candidate_sequence = sampler.generate_candidate_sequence(&logits).unwrap();

        // 应该返回num_candidates个token ID
        assert_eq!(candidate_sequence.len(), 3);

        // 验证并接受tokens（使用相同的logits作为目标）
        let accepted = sampler.verify_and_accept(&logits).unwrap();

        // 接受的数量取决于验证分数是否超过阈值
        // 至少应该返回一个Vec（可能为空）
        assert!(accepted.len() <= candidate_sequence.len());

        // 验证内部状态已更新
        let rate = sampler.decoder().acceptance_rate();
        assert!(rate >= 0.0 && rate <= 1.0);
    }

    /// 测试verify_candidates中的边界条件：target_prob=0
    #[test]
    fn test_verify_candidates_zero_target_prob() {
        let mut decoder = MtpDecoder::default_decoder();

        let candidates = vec![
            MtpCandidate { token_id: 999, probability: 0.9, verification_score: 0.0 }, // 不在目标分布中
            MtpCandidate { token_id: 0, probability: 0.1, verification_score: 0.0 },   // 在目标分布中但概率低
        ];

        // 创建一个只有前几个token有概率的目标分布
        let target_logits = Array2::from_shape_fn((1, 100), |(_, j)| {
            if j < 10 { 1.0 } else { 0.0 } // 只有0-9有概率
        });

        let verified = decoder.verify_candidates(&candidates, &target_logits).unwrap();

        // token_id=999不在目标分布中，verification_score应为0
        let candidate_999 = verified.iter().find(|c| c.token_id == 999).unwrap();
        assert!((candidate_999.verification_score - 0.0).abs() < 1e-6);

        // token_id=0在目标分布中，verification_score应该>0
        let candidate_0 = verified.iter().find(|c| c.token_id == 0).unwrap();
        assert!(candidate_0.verification_score > 0.0);
    }

    /// 测试MtpConfig自定义配置
    #[test]
    fn test_mtp_config_custom() {
        let custom_config = MtpConfig {
            num_candidates: 10,
            speculative_sampling: false,
            acceptance_threshold: 0.99,
            max_consecutive_accepts: 100,
        };

        assert_eq!(custom_config.num_candidates, 10);
        assert!(!custom_config.speculative_sampling);
        assert!((custom_config.acceptance_threshold - 0.99).abs() < 1e-6);
        assert_eq!(custom_config.max_consecutive_accepts, 100);
    }
}
