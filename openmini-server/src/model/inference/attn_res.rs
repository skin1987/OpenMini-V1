//! Attention Residuals (AttnRes) 实现
//!
//! 基于 Kimi 团队 2026 论文 "Attention Residuals" 的 Block 版本。
//! 用深度方向 Softmax 注意力替代固定等权残差累加，
//! 解决 PreNorm Dilution 问题，提升深层网络训练稳定性与推理质量。
//!
//! # 核心公式
//! ```text
//! h_l = x + Σ_{k=0}^{N-1} α_k · summary[k]
//! α_k = softmax( w_l · RMSNorm(summary[k]) )
//! ```
//!
//! # 与现有 mHC 的关系
//! - mHC 处理层内多流表征（Per-Head 流形约束）
//! - AttnRes 处理跨块深度聚合（Softmax 加权检索）
//! - 两者正交可组合使用

#![allow(dead_code)]

use super::error::InferenceError;
use super::error::InferenceResult;
use ndarray::{Array1, Array2, Axis};

/// Block AttnRes 配置
#[derive(Debug, Clone)]
pub struct AttnResConfig {
    /// 是否启用（默认 true）
    pub enabled: bool,
    /// Block 数量（0 = 自适应）
    pub num_blocks: usize,
    /// 每个 Block 的层数（0 = 自动计算 = ceil(L / num_blocks)）
    pub block_size: usize,
    /// 总层数
    pub total_layers: usize,
    /// 隐藏维度
    pub hidden_size: usize,
    /// RMSNorm epsilon
    pub rms_eps: f32,
    /// 伪查询向量初始化尺度（0.0 = 零初始化）
    pub init_scale: f32,
}

impl AttnResConfig {
    /// 根据层数自动计算最优配置
    pub fn auto(num_layers: usize, hidden_size: usize, rms_eps: f32) -> Self {
        // 自适应策略：
        // - L <= 8:   2 块（每块 ~4 层）
        // - L <= 16:  4 块（每块 ~4 层）
        // - L <= 32:  8 块（每块 ~4 层）← 论文推荐甜蜜点
        // - L <= 64:  8 块（每块 ~8 层）
        // - L > 64:   16 块（每块 ~8 层）
        let num_blocks = if num_layers <= 8 {
            2
        } else if num_layers <= 16 {
            4
        } else if num_layers <= 64 {
            8
        } else {
            16
        };

        let block_size = num_layers.div_ceil(num_blocks);

        Self {
            enabled: true,
            num_blocks,
            block_size,
            total_layers: num_layers,
            hidden_size,
            rms_eps,
            init_scale: 0.0,
        }
    }

    /// 禁用 AttnRes（回退到标准残差）
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            num_blocks: 0,
            block_size: 0,
            total_layers: 0,
            hidden_size: 0,
            rms_eps: 1e-6,
            init_scale: 0.0,
        }
    }
}

/// Block 摘要状态（跨层维护的跨块注意力状态）
///
/// 维护每个 Block 的摘要向量，用于跨块深度聚合。
/// 内存开销: O(N × d)，N=block数，d=hidden_size
pub struct BlockSummary {
    summaries: Vec<Array2<f32>>,
    block_size: usize,
    num_blocks: usize,
    current_block: usize,
    hidden_size: usize,
    rms_eps: f32,
    total_layers: usize,
}

impl BlockSummary {
    pub fn new(config: &AttnResConfig) -> Self {
        let mut summaries = Vec::with_capacity(config.num_blocks);
        for _ in 0..config.num_blocks {
            summaries.push(Array2::zeros((1, config.hidden_size)));
        }
        Self {
            summaries,
            block_size: config.block_size,
            num_blocks: config.num_blocks,
            current_block: 0,
            hidden_size: config.hidden_size,
            rms_eps: config.rms_eps,
            total_layers: config.total_layers,
        }
    }

    /// 获取当前层所属的 Block 索引
    #[inline]
    pub fn current_block_index(&self, layer_idx: usize) -> usize {
        (layer_idx / self.block_size).min(self.num_blocks - 1)
    }

    /// 判断当前层是否为 Block 第一层
    #[inline]
    pub fn is_block_start(&self, layer_idx: usize) -> bool {
        layer_idx == 0 || layer_idx % self.block_size == 0
    }

    /// 判断当前层是否为 Block 最后一层
    #[inline]
    pub fn is_block_end(&self, layer_idx: usize) -> bool {
        (layer_idx + 1) % self.block_size == 0 || layer_idx + 1 == self.total_layers
    }

    /// 更新当前 Block 摘要（在 Block 最后一层调用）
    /// 使用 mean-pooling 将 (seq, hidden) → (1, hidden) 作为摘要
    pub fn update_summary(
        &mut self,
        layer_output: &Array2<f32>,
        layer_idx: usize,
    ) -> InferenceResult<()> {
        let block_idx = self.current_block_index(layer_idx);
        if block_idx < self.summaries.len() {
            let (_seq_len, _) = layer_output.dim();
            let pooled: Array1<f32> = layer_output.mean_axis(Axis(0)).ok_or_else(|| {
                InferenceError::generation("AttnRes mean_pool failed: empty axis".to_string())
            })?;
            self.summaries[block_idx] = pooled.insert_axis(Axis(0));
        }
        Ok(())
    }

    /// 跨块聚合：用伪查询向量对历史 Block 摘要做 Softmax 加权聚合
    ///
    /// 数学: h_agg = Σ_k α_k · summary[k], where α = softmax(w · RMSNorm(summary))
    /// 返回: x + h_agg（保持残差模式）
    pub fn aggregate(
        &self,
        pseudo_query: &Array1<f32>,
        x: &Array2<f32>,
    ) -> InferenceResult<Array2<f32>> {
        let n = self.summaries.len();
        if n == 0 {
            return Ok(x.clone());
        }

        // Step 1: 对每个摘要做 RMSNorm 并计算 attention score
        let mut scores = Vec::with_capacity(n);
        for summary in &self.summaries {
            if summary.nrows() == 0 || summary.ncols() == 0 {
                scores.push(f32::NEG_INFINITY);
                continue;
            }
            let normed = rms_norm_vector(&summary.row(0).to_owned(), self.rms_eps)?;
            let score = pseudo_query.dot(&normed);
            scores.push(score);
        }

        // Step 2: Softmax 归一化
        let alphas = softmax_1d(&scores);

        // Step 3: 加权聚合
        let (seq_len, hidden_size) = x.dim();
        let mut aggregated = Array2::zeros((seq_len, hidden_size));
        for (k, summary) in self.summaries.iter().enumerate() {
            if k < alphas.len() && summary.nrows() > 0 {
                let weight = alphas[k];
                aggregated += &(summary.to_owned() * weight);
            }
        }

        // Step 4: 残差连接
        Ok(x + &aggregated)
    }
}

/// RMSNorm 向量版本：(x / sqrt(mean(x²) + eps))
fn rms_norm_vector(x: &Array1<f32>, eps: f32) -> InferenceResult<Array1<f32>> {
    let n = x.len();
    if n == 0 {
        return Err(InferenceError::generation(
            "rms_norm_vector: empty input".to_string(),
        ));
    }
    let sq_mean: f32 = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    let norm = (sq_mean + eps).sqrt();
    Ok(x.mapv(|v| v / norm))
}

/// 1D Softmax（数值稳定版）
fn softmax_1d(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return vec![];
    }
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    exps.iter().map(|e| e * inv_sum).collect()
}

/// Block AttnRes 跨块聚合（独立函数，供外部调用）
pub fn block_attnres_aggregate(
    x: &Array2<f32>,
    summaries: &[Array2<f32>],
    pseudo_query: &Array1<f32>,
    rms_eps: f32,
) -> InferenceResult<Array2<f32>> {
    let n = summaries.len();
    if n == 0 {
        return Ok(x.clone());
    }

    let mut scores = Vec::with_capacity(n);
    for summary in summaries {
        if summary.nrows() == 0 {
            scores.push(f32::NEG_INFINITY);
            continue;
        }
        let normed = rms_norm_vector(&summary.row(0).to_owned(), rms_eps)?;
        scores.push(pseudo_query.dot(&normed));
    }

    let alphas = softmax_1d(&scores);
    let (seq_len, hidden_size) = x.dim();
    let mut aggregated = Array2::zeros((seq_len, hidden_size));
    for (k, summary) in summaries.iter().enumerate() {
        if k < alphas.len() && summary.nrows() > 0 {
            aggregated += &(summary.to_owned() * alphas[k]);
        }
    }
    Ok(x + &aggregated)
}

/// 两阶段推理缓存（生产环境优化）
///
/// 利用伪查询向量 w_l 是固定参数的特性，
/// 在 Prefill 阶段一次性预计算所有跨块权重，
/// Decode 阶段直接查表，避免重复计算。
pub struct TwoStageAttnResCache {
    cross_weights: Vec<Vec<f32>>,
    valid: bool,
    num_blocks: usize,
}

impl TwoStageAttnResCache {
    pub fn new(num_blocks: usize) -> Self {
        let mut cross_weights = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            cross_weights.push(vec![0.0f32; num_blocks]);
        }
        Self {
            cross_weights,
            valid: false,
            num_blocks,
        }
    }

    /// 预计算阶段：一次性算好所有 Block 边界的跨块权重
    pub fn precompute(
        &mut self,
        block_summaries: &[Array2<f32>],
        layer_pseudo_queries: &[&Array1<f32>],
        rms_eps: f32,
    ) -> InferenceResult<()> {
        let n = block_summaries.len();
        if n != self.num_blocks {
            return Err(InferenceError::generation(format!(
                "Block count mismatch: expected {}, got {}",
                self.num_blocks, n
            )));
        }

        for (i, pq) in layer_pseudo_queries.iter().enumerate() {
            if i >= n {
                break;
            }
            let mut scores = Vec::with_capacity(n);
            for summary in block_summaries {
                if summary.nrows() == 0 {
                    scores.push(f32::NEG_INFINITY);
                    continue;
                }
                let normed = rms_norm_vector(&summary.row(0).to_owned(), rms_eps)?;
                scores.push(pq.dot(&normed));
            }
            self.cross_weights[i] = softmax_1d(&scores);
        }
        self.valid = true;
        Ok(())
    }

    /// 查询阶段：O(N) 加权聚合（无需重复 Softmax）
    pub fn query(
        &self,
        x: &Array2<f32>,
        target_block_idx: usize,
        block_summaries: &[Array2<f32>],
    ) -> InferenceResult<Array2<f32>> {
        if !self.valid || target_block_idx >= self.cross_weights.len() {
            return Err(InferenceError::generation(
                "TwoStageCache: not initialized or invalid index".to_string(),
            ));
        }

        let alphas = &self.cross_weights[target_block_idx];
        let (seq_len, hidden_size) = x.dim();
        let mut aggregated = Array2::zeros((seq_len, hidden_size));
        for (k, summary) in block_summaries.iter().enumerate() {
            if k < alphas.len() && summary.nrows() > 0 {
                aggregated += &(summary.to_owned() * alphas[k]);
            }
        }
        Ok(x + &aggregated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ==================== A. 基础功能测试 ====================

    #[test]
    fn test_attnres_config_auto() {
        // 测试不同层数的自适应配置
        let config_4 = AttnResConfig::auto(4, 256, 1e-6);
        assert!(config_4.enabled);
        assert_eq!(config_4.num_blocks, 2); // L<=8 → 2 blocks
        assert_eq!(config_4.block_size, 2); // ceil(4/2)=2

        let config_12 = AttnResConfig::auto(12, 256, 1e-6);
        assert_eq!(config_12.num_blocks, 4); // L<=16 → 4 blocks
        assert_eq!(config_12.block_size, 3); // ceil(12/4)=3

        let config_24 = AttnResConfig::auto(24, 256, 1e-6);
        assert_eq!(config_24.num_blocks, 8); // L<=32 → 8 blocks
        assert_eq!(config_24.block_size, 3); // ceil(24/8)=3

        let config_48 = AttnResConfig::auto(48, 256, 1e-6);
        assert_eq!(config_48.num_blocks, 8); // L<=64 → 8 blocks
        assert_eq!(config_48.block_size, 6); // ceil(48/8)=6

        let config_96 = AttnResConfig::auto(96, 256, 1e-6);
        assert_eq!(config_96.num_blocks, 16); // L>64 → 16 blocks
        assert_eq!(config_96.block_size, 6); // ceil(96/16)=6
    }

    #[test]
    fn test_attnres_config_disabled() {
        let config = AttnResConfig::disabled();
        assert!(!config.enabled);
        assert_eq!(config.num_blocks, 0);
        assert_eq!(config.block_size, 0);
        assert_eq!(config.hidden_size, 0);
    }

    #[test]
    fn test_block_summary_new() {
        let config = AttnResConfig::auto(12, 256, 1e-6);
        let bs = BlockSummary::new(&config);

        assert_eq!(bs.num_blocks, 4);
        assert_eq!(bs.block_size, 3);
        assert_eq!(bs.summaries.len(), 4);
        assert_eq!(bs.hidden_size, 256);

        // 所有摘要应初始化为零
        for summary in &bs.summaries {
            assert_eq!(summary.dim(), (1, 256));
            assert!(summary.iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn test_block_summary_block_index() {
        let config = AttnResConfig::auto(12, 256, 1e-6);
        let bs = BlockSummary::new(&config);

        // block_size=3, 所以:
        // layer 0,1,2 → block 0
        // layer 3,4,5 → block 1
        // layer 6,7,8 → block 2
        // layer 9,10,11 → block 3
        assert_eq!(bs.current_block_index(0), 0);
        assert_eq!(bs.current_block_index(1), 0);
        assert_eq!(bs.current_block_index(2), 0);
        assert_eq!(bs.current_block_index(3), 1);
        assert_eq!(bs.current_block_index(6), 2);
        assert_eq!(bs.current_block_index(9), 3);
        assert_eq!(bs.current_block_index(11), 3);
    }

    #[test]
    fn test_block_summary_is_start_end() {
        let config = AttnResConfig::auto(12, 256, 1e-6);
        let bs = BlockSummary::new(&config);

        // Block 边界检查
        assert!(bs.is_block_start(0)); // 第一层总是 start
        assert!(!bs.is_block_start(1));
        assert!(bs.is_block_start(3)); // block 1 开始
        assert!(bs.is_block_start(6)); // block 2 开始
        assert!(bs.is_block_start(9)); // block 3 开始

        assert!(bs.is_block_end(2)); // block 0 结束
        assert!(bs.is_block_end(5)); // block 1 结束
        assert!(bs.is_block_end(8)); // block 2 结束
        assert!(bs.is_block_end(11)); // 最后一层也是 end
    }

    // ==================== B. 数学正确性测试 ====================

    #[test]
    fn test_aggregate_single_block() {
        // 单块时应退化为恒等映射（因为只有一个非零摘要，权重=1.0）
        let config = AttnResConfig::auto(4, 64, 1e-6);
        let mut bs = BlockSummary::new(&config);

        // 设置一个非零摘要
        let mut summary_data = vec![0.0f32; 64];
        summary_data[0] = 1.0;
        bs.summaries[0] = Array2::from_shape_vec((1, 64), summary_data).unwrap();

        let x = Array2::from_shape_fn((2, 64), |(i, j)| (i * 64 + j) as f32);
        let pq = Array1::zeros(64);

        let result = bs.aggregate(&pq, &x).unwrap();
        // 单块时应该返回 x + summary（权重=1.0）
        assert_eq!(result.dim(), x.dim());
    }

    #[test]
    fn test_aggregate_softmax_normalization() {
        // 测试 Softmax 权重和严格等于 1
        let config = AttnResConfig::auto(8, 32, 1e-6);
        let mut bs = BlockSummary::new(&config);

        // 设置不同的摘要值
        for (i, summary) in bs.summaries.iter_mut().enumerate() {
            let mut data = vec![0.0f32; 32];
            data[0] = (i + 1) as f32; // 不同摘要有不同值
            *summary = Array2::from_shape_vec((1, 32), data).unwrap();
        }

        let x = Array2::zeros((1, 32));
        let pq = Array1::from_vec(vec![1.0; 32]);

        let _result = bs.aggregate(&pq, &x).unwrap();
        // 通过内部验证：手动计算 softmax 权重和
        let scores: Vec<f32> = bs
            .summaries
            .iter()
            .map(|s| {
                let normed = rms_norm_vector(&s.row(0).to_owned(), 1e-6).unwrap();
                pq.dot(&normed)
            })
            .collect();

        let alphas = softmax_1d(&scores);
        let weight_sum: f32 = alphas.iter().sum();
        assert!(
            (weight_sum - 1.0).abs() < 1e-5,
            "Softmax weights should sum to 1.0, got {}",
            weight_sum
        );
    }

    #[test]
    fn test_aggregate_zero_input() {
        let config = AttnResConfig::auto(4, 32, 1e-6);
        let mut bs = BlockSummary::new(&config);

        // 设置非零摘要
        let data = vec![1.0f32; 32];
        bs.summaries[0] = Array2::from_shape_vec((1, 32), data).unwrap();

        let x = Array2::zeros((2, 32)); // 零输入
        let pq = Array1::ones(32);

        let result = bs.aggregate(&pq, &x).unwrap();
        // 结果应该是 0 + weighted_summary ≈ 非零值
        assert!(result.iter().any(|&v| v.abs() > 0.01));
    }

    #[test]
    fn test_aggregate_deterministic() {
        let config = AttnResConfig::auto(4, 16, 1e-6);
        let mut bs = BlockSummary::new(&config);

        // 设置确定性的摘要
        let data = vec![0.5f32; 16];
        bs.summaries[0] = Array2::from_shape_vec((1, 16), data.clone()).unwrap();

        let x = Array2::from_shape_fn((1, 16), |(_, j)| j as f32);
        let pq = Array1::from_vec(vec![0.1; 16]);

        let result1 = bs.aggregate(&pq, &x).unwrap();
        let result2 = bs.aggregate(&pq, &x).unwrap();

        // 相同输入必须产生相同输出
        assert_eq!(result1.dim(), result2.dim());
        for i in 0..result1.nrows() {
            for j in 0..result1.ncols() {
                assert_abs_diff_eq!(result1[[i, j]], result2[[i, j]], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_rms_norm_vector_correctness() {
        // 手动计算 RMSNorm 验证
        let x = Array1::from_vec(vec![3.0, 4.0]); // sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
        let result = rms_norm_vector(&x, 1e-6).unwrap();

        let expected_norm: f64 = ((9.0_f64 + 16.0_f64) / 2.0_f64 + 1e-6_f64).sqrt();
        assert_abs_diff_eq!(result[0], (3.0_f64 / expected_norm) as f32, epsilon = 1e-5);
        assert_abs_diff_eq!(result[1], (4.0_f64 / expected_norm) as f32, epsilon = 1e-5);

        // 测试零向量（不应崩溃）
        let zero = Array1::zeros(4);
        let zero_result = rms_norm_vector(&zero, 1e-6).unwrap();
        assert!(zero_result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_softmax_1d_properties() {
        // 测试 Softmax 性质
        let scores = vec![1.0, 2.0, 3.0];
        let probs = softmax_1d(&scores);

        // 性质1: 长度相同
        assert_eq!(probs.len(), 3);

        // 性质2: 所有值为正
        assert!(probs.iter().all(|&p| p >= 0.0));

        // 性质3: 和为 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // 性质4: 最大值对应最大概率
        let max_idx = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(max_idx, probs[2]); // scores[2]=3.0 最大

        // 测试空输入
        let empty_probs = softmax_1d(&[]);
        assert!(empty_probs.is_empty());

        // 测试单元素
        let single_probs = softmax_1d(&[5.0]);
        assert_eq!(single_probs.len(), 1);
        assert_abs_diff_eq!(single_probs[0], 1.0, epsilon = 1e-5);
    }

    // ==================== C. 边界条件测试 ====================

    #[test]
    fn test_empty_summaries() {
        let x = Array2::from_shape_fn((2, 16), |(i, j)| (i * 16 + j) as f32);
        let summaries: Vec<Array2<f32>> = vec![];
        let pq = Array1::ones(16);

        let result = block_attnres_aggregate(&x, &summaries, &pq, 1e-6).unwrap();
        // 空摘要列表应返回原始输入
        assert_eq!(result.dim(), x.dim());
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert_abs_diff_eq!(result[[i, j]], x[[i, j]], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_single_element_summary() {
        let x = Array2::ones((1, 4));
        let mut summary = Array2::zeros((1, 4));
        summary[[0, 0]] = 2.0;

        let summaries = vec![summary];
        let pq = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);

        let result = block_attnres_aggregate(&x, &summaries, &pq, 1e-6).unwrap();
        assert_eq!(result.dim(), (1, 4));
        // 应该返回 x + weighted_summary
    }

    #[test]
    fn test_large_num_blocks() {
        // Stress test: 大量 Block
        let num_blocks = 64;
        let hidden = 32;
        let mut summaries = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            let mut s = Array2::zeros((1, hidden));
            s[[0, 0]] = i as f32;
            summaries.push(s);
        }

        let x = Array2::ones((2, hidden));
        let pq = Array1::ones(hidden);

        let result = block_attnres_aggregate(&x, &summaries, &pq, 1e-6).unwrap();
        assert_eq!(result.dim(), (2, hidden));
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_very_small_hidden_dim() {
        // 极小隐藏维度（1维）
        let config = AttnResConfig::auto(4, 1, 1e-6);
        let mut bs = BlockSummary::new(&config);

        let data = vec![5.0f32];
        bs.summaries[0] = Array2::from_shape_vec((1, 1), data).unwrap();

        let x = Array2::from_shape_vec((1, 1), vec![3.0]).unwrap();
        let pq = Array1::from_vec(vec![1.0]);

        let result = bs.aggregate(&pq, &x).unwrap();
        assert_eq!(result.dim(), (1, 1));
        assert!(result[[0, 0]].is_finite());
    }

    #[test]
    fn test_very_large_sequence() {
        // 长序列测试
        let seq_len = 1024;
        let hidden = 16;
        let config = AttnResConfig::auto(4, hidden, 1e-6);
        let mut bs = BlockSummary::new(&config);

        // 设置摘要
        let data = vec![1.0f32; hidden];
        bs.summaries[0] = Array2::from_shape_vec((1, hidden), data).unwrap();

        let x = Array2::from_shape_fn((seq_len, hidden), |(i, j)| (i as f32 * 0.01 + j as f32));
        let pq = Array1::ones(hidden);

        let result = bs.aggregate(&pq, &x).unwrap();
        assert_eq!(result.dim(), (seq_len, hidden));
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    // ==================== D. 生产环境测试 ====================

    #[test]
    fn test_two_stage_cache_precompute_and_query() {
        let num_blocks = 4;
        let hidden = 16;
        let mut cache = TwoStageAttnResCache::new(num_blocks);

        // 准备数据
        let mut summaries = Vec::with_capacity(num_blocks);
        let mut queries = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            let mut s = Array2::zeros((1, hidden));
            s[[0, 0]] = (i + 1) as f32;
            summaries.push(s);

            let q = Array1::from_vec(vec![1.0; hidden]);
            queries.push(q);
        }

        let query_refs: Vec<&Array1<f32>> = queries.iter().collect();

        // 预计算
        cache.precompute(&summaries, &query_refs, 1e-6).unwrap();
        assert!(cache.valid);

        // 查询
        let x = Array2::ones((2, hidden));
        let result = cache.query(&x, 0, &summaries).unwrap();
        assert_eq!(result.dim(), (2, hidden));
        assert!(result.iter().all(|&v| v.is_finite()));

        // 查询其他 block
        let result2 = cache.query(&x, 2, &summaries).unwrap();
        assert_eq!(result2.dim(), (2, hidden));
    }

    #[test]
    fn test_two_stage_cache_invalid_access() {
        let mut cache = TwoStageAttnResCache::new(4);
        let x = Array2::zeros((1, 8));

        // 未初始化时访问应报错
        let summaries_empty: Vec<Array2<f32>> = vec![];
        let result = cache.query(&x, 0, &summaries_empty);
        assert!(result.is_err());

        // 越界索引应报错（先正常初始化）
        let mut summaries = Vec::with_capacity(4);
        for _ in 0..4 {
            summaries.push(Array2::zeros((1, 8)));
        }
        cache.precompute(&summaries, &[], 1e-6).unwrap();
        let result2 = cache.query(&x, 10, &summaries);
        assert!(result2.is_err());
    }

    #[test]
    fn test_two_stage_cache_matches_direct() {
        // 验证两阶段缓存结果与直接计算一致
        let num_blocks = 4;
        let hidden = 32;
        let mut cache = TwoStageAttnResCache::new(num_blocks);

        // 准备确定性数据
        let mut summaries = Vec::with_capacity(num_blocks);
        let mut queries = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            let s = Array2::from_shape_fn((1, hidden), |(_, j)| (i * hidden + j) as f32 * 0.1);
            summaries.push(s);

            let q = Array1::from_shape_fn(hidden, |j| (j as f32 * 0.05));
            queries.push(q);
        }

        let query_refs: Vec<&Array1<f32>> = queries.iter().collect();
        cache.precompute(&summaries, &query_refs, 1e-6).unwrap();

        // 对每个 block 进行对比
        for block_idx in 0..num_blocks {
            let x = Array2::from_shape_fn((3, hidden), |(i, j)| {
                ((block_idx * 100 + i * hidden + j) as f32) * 0.01
            });

            // 直接计算
            let direct_result =
                block_attnres_aggregate(&x, &summaries, &queries[block_idx], 1e-6).unwrap();

            // 缓存查询
            let cached_result = cache.query(&x, block_idx, &summaries).unwrap();

            // 对比结果（容差 1e-5）
            assert_eq!(direct_result.dim(), cached_result.dim());
            for i in 0..direct_result.nrows() {
                for j in 0..direct_result.ncols() {
                    assert_abs_diff_eq!(
                        direct_result[[i, j]],
                        cached_result[[i, j]],
                        epsilon = 1e-5
                    );
                }
            }
        }
    }

    // ==================== E. 与 mHC 组合兼容性 ====================

    #[test]
    fn test_mhc_attnres_ordering() {
        // 验证 mHC 和 AttnRes 可以按顺序组合使用
        // 模拟场景：先执行 mHC（层内约束），再执行 AttnRes（跨块聚合）

        let hidden = 64;
        let seq_len = 2;

        // 模拟 mHC 输出（假设经过 Sinkhorn-Knopp 正则化后的表征）
        let mhc_output = Array2::from_shape_fn((seq_len, hidden), |(i, j)| {
            // 模拟经过归一化的输出
            let base = (i * hidden + j) as f32;
            base / (base.abs() + 1.0) // 归一化到 [-1, 1]
        });

        // 配置 AttnRes (使用 24 层 → 8 个 blocks)
        let config = AttnResConfig::auto(24, hidden, 1e-6);
        let mut bs = BlockSummary::new(&config);

        // 模拟前几个 block 已更新摘要
        for block_idx in 0..3 {
            let mut summary_data = vec![0.0f32; hidden];
            for j in 0..hidden {
                summary_data[j] = (block_idx * hidden + j) as f32 * 0.1;
            }
            bs.summaries[block_idx] = Array2::from_shape_vec((1, hidden), summary_data).unwrap();
        }

        // 执行 AttnRes 聚合
        let pseudo_query = Array1::ones(hidden);
        let final_output = bs.aggregate(&pseudo_query, &mhc_output).unwrap();

        // 验证输出维度和数值有效性
        assert_eq!(final_output.dim(), (seq_len, hidden));
        assert!(final_output.iter().all(|&v| v.is_finite()));

        // 验证残差连接确实生效（输出 ≠ 纯 mhc_output）
        let is_different = final_output
            .iter()
            .zip(mhc_output.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            is_different,
            "AttnRes should modify the output via residual connection"
        );
    }
}
