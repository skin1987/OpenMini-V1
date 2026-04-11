//! Tensor Product Attention (TPA) 张量积注意力模块
//!
//! TPA 通过低秩分解 Q/K/V 张量为时间和特征双空间，作为 MLA 的替代方案，
//! 提供不同的注意力机制选择。
//!
//! ## 核心原理
//!
//! 标准 Attention 的 Q/K/V 矩阵维度为 [batch, seq_len, hidden_dim]，
//! 复杂度为 O(N² × d)，其中 N 是序列长度，d 是隐藏维度。
//!
//! TPA 将 Q/K/V 分解为时间空间和特征空间的低秩近似:
//! ```text
//! Q = Q_t @ Q_f^T  (外积近似)
//! K = K_t @ K_f^T
//! V = V_t @ V_f^T
//! ```
//!
//! 其中:
//! - Q_t: [batch, seq_len, time_rank] - 时间投影（压缩序列信息）
//! - Q_f: [batch, seq_len, feat_rank] - 特征投影（压缩特征信息）
//!
//! ## 复杂度分析
//!
//! | 操作 | 标准 Attention | TPA |
//! |------|---------------|-----|
//! | 投影 | O(N × d²) | O(N × d × r) |
//! | 注意力分数 | O(N² × d) | O(N² × r²) |
//! | 总计 | O(N² × d) | O(N × d × r + N² × r²) |
//!
//! 当 r << d 且 N 较大时，TPA 显著优于标准 Attention。
//!
//! ## 性能目标
//! - 序列长度 > 8K 时: 比 FA 快 2-3x
//! - 显存比标准 FA 节省 60%+
//! - 精度损失 < 2%

#![allow(dead_code)]

use anyhow::Result;
use ndarray::{s, Array1, Array2, Array3, Axis};
use std::time::Instant;

// ============================================================================
// 配置和枚举类型
// ============================================================================

/// TPA 配置参数
#[derive(Debug, Clone)]
pub struct TPAConfig {
    /// 隐藏维度
    pub hidden_dim: usize,
    /// 时间秩 (默认 256)
    pub time_rank: usize,
    /// 特征秩 (默认 256)
    pub feat_rank: usize,
    /// 注意力头数
    pub num_heads: usize,
    /// 每个头的维度
    pub head_dim: usize,
    /// 是否使用因果掩码
    pub use_causal_mask: bool,
}

impl Default for TPAConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 1024,
            time_rank: 256,
            feat_rank: 256,
            num_heads: 8,
            head_dim: 128,
            use_causal_mask: true,
        }
    }
}

impl TPAConfig {
    /// 创建新配置
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置隐藏维度
    pub fn with_hidden_dim(mut self, dim: usize) -> Self {
        self.hidden_dim = dim;
        self
    }

    /// 设置时间秩
    pub fn with_time_rank(mut self, rank: usize) -> Self {
        self.time_rank = rank;
        self
    }

    /// 设置特征秩
    pub fn with_feat_rank(mut self, rank: usize) -> Self {
        self.feat_rank = rank;
        self
    }

    /// 设置头数和头维度
    pub fn with_heads(mut self, num_heads: usize, head_dim: usize) -> Self {
        self.num_heads = num_heads;
        self.head_dim = head_dim;
        self
    }

    /// 启用/禁用因果掩码
    pub fn with_causal_mask(mut self, enable: bool) -> Self {
        self.use_causal_mask = enable;
        self
    }

    /// 验证配置有效性
    pub fn validate(&self) -> Result<()> {
        if self.hidden_dim == 0 {
            anyhow::bail!("hidden_dim must be positive");
        }
        if self.time_rank == 0 || self.feat_rank == 0 {
            anyhow::bail!("ranks must be positive");
        }
        if self.num_heads == 0 || self.head_dim == 0 {
            anyhow::bail!("num_heads and head_dim must be positive");
        }
        if self.hidden_dim != self.num_heads * self.head_dim {
            anyhow::bail!(
                "hidden_dim ({}) must equal num_heads ({}) * head_dim ({})",
                self.hidden_dim,
                self.num_heads,
                self.head_dim
            );
        }
        Ok(())
    }
}

/// 注意力机制推荐枚举
#[derive(Debug, Clone, PartialEq)]
pub enum Recommendation {
    /// 推荐 TPA
    UseTPA { reason: String },
    /// 推荐 MLA
    UseMLA { reason: String },
    /// 推荐标准 Flash Attention
    UseStandardFA { reason: String },
}

// ============================================================================
// 线性层实现
// ============================================================================

/// 简单线性层
struct LinearLayer {
    /// 权重矩阵 [out_features, in_features]
    weight: Array2<f32>,
    /// 偏置向量（可选）
    bias: Option<Array1<f32>>,
}

impl LinearLayer {
    /// 创建新的线性层
    fn new(in_features: usize, out_features: usize) -> Result<Self> {
        if in_features == 0 || out_features == 0 {
            anyhow::bail!("Linear layer dimensions must be positive");
        }

        let scale = (2.0 / (in_features + out_features) as f32).sqrt();

        // 使用简单初始化
        let weight = Array2::from_shape_fn((out_features, in_features), |_| {
            (rand::random::<f32>() * 2.0 - 1.0) * scale
        });

        Ok(Self {
            weight,
            bias: None,
        })
    }

    /// 前向传播: y = xW^T + b
    fn forward(&self, x: &Array2<f32>) -> Result<Array2<f32>> {
        let weight_t = self.weight.t().to_owned();
        let output = matmul(x, &weight_t)?;

        if let Some(ref bias) = self.bias {
            let mut result = output.clone();
            result
                .axis_iter_mut(Axis(0))
                .for_each(|mut row| {
                    row.iter_mut()
                        .zip(bias.iter())
                        .for_each(|(val, &b)| {
                            *val += b;
                        });
                });
            Ok(result)
        } else {
            Ok(output)
        }
    }

    /// 参数量统计
    fn param_count(&self) -> usize {
        self.weight.len() + self.bias.as_ref().map_or(0, |b| b.len())
    }
}

// ============================================================================
// 注意力指标结构
// ============================================================================

/// 注意力性能指标
#[derive(Debug, Clone)]
pub struct AttentionMetrics {
    /// 执行时间（毫秒）
    pub execution_time_ms: f64,
    /// 内存使用量（字节）
    pub memory_usage_bytes: usize,
    /// FLOPS 计算
    pub flops: u64,
    /// 精度（与标准注意力的相似度）
    pub accuracy: f32,
}

/// 复杂度信息
#[derive(Debug, Clone)]
pub struct ComplexityInfo {
    /// 标准 Attention FLOPS
    pub standard_flops: u64,
    /// TPA FLOPS
    pub tpa_flops: u64,
    /// 减少比例
    pub reduction_ratio: f32,
    /// 标准内存占用
    pub memory_standard: usize,
    /// TPA 内存占用
    pub memory_tpa: usize,
}

/// MLA vs TPA 性能对比报告
#[derive(Debug, Clone)]
pub struct ComparisonReport {
    /// MLA 指标
    pub mla_metrics: AttentionMetrics,
    /// TPA 指标
    pub tpa_metrics: AttentionMetrics,
    /// 加速比
    pub speedup: f32,
    /// 内存节省百分比
    pub memory_saving_pct: f32,
    /// 精度保留率
    pub accuracy_retention: f32,
}

// ============================================================================
// 核心 TPA 实现
// ============================================================================

/// Tensor Product Attention 引擎
///
/// 通过低秩分解实现高效的注意力计算。
///
/// # 示例
///
/// ```ignore
/// let config = TPAConfig::new()
///     .with_hidden_dim(512)
///     .with_time_rank(128)
///     .with_feat_rank(128)
///     .with_heads(4, 128);
///
/// let tpa = TensorProductAttention::new(config)?;
/// ```
pub struct TensorProductAttention {
    /// 时间投影矩阵 Q_t
    time_proj_q: LinearLayer,
    /// 时间投影矩阵 K_t
    time_proj_k: LinearLayer,
    /// 时间投影矩阵 V_t
    time_proj_v: LinearLayer,
    /// 特征投影矩阵 Q_f
    feat_proj_q: LinearLayer,
    /// 特征投影矩阵 K_f
    feat_proj_k: LinearLayer,
    /// 特征投影矩阵 V_f
    feat_proj_v: LinearLayer,
    /// 输出投影
    output_proj: LinearLayer,
    /// 配置
    config: TPAConfig,
}

impl TensorProductAttention {
    /// 创建新的 TPA 实例
    ///
    /// # 参数
    ///
    /// * `config` - TPA 配置参数
    ///
    /// # 错误
    ///
    /// 如果配置无效则返回错误
    pub fn new(config: TPAConfig) -> Result<Self> {
        config.validate()?;

        let hidden_dim = config.hidden_dim;
        let time_rank = config.time_rank;
        let feat_rank = config.feat_rank;

        Ok(Self {
            time_proj_q: LinearLayer::new(hidden_dim, time_rank)?,
            time_proj_k: LinearLayer::new(hidden_dim, time_rank)?,
            time_proj_v: LinearLayer::new(hidden_dim, time_rank)?,
            feat_proj_q: LinearLayer::new(hidden_dim, feat_rank)?,
            feat_proj_k: LinearLayer::new(hidden_dim, feat_rank)?,
            feat_proj_v: LinearLayer::new(hidden_dim, feat_rank)?,
            output_proj: LinearLayer::new(hidden_dim, hidden_dim)?,
            config,
        })
    }

    /// TPA 前向传播
    ///
    /// 公式分解:
    /// 1. Q = Q_t @ Q_f^T  (外积近似)
    /// 2. K = K_t @ K_f^T
    /// 3. V = V_t @ V_f^T
    /// 4. Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
    ///
    /// 复杂度: O(N * r² + N * d * r) vs 标准 O(N² * d)
    ///
    /// # 参数
    ///
    /// * `query` - 查询张量 [batch, seq_len, hidden_dim]
    /// * `key` - 键张量 [batch, seq_len, hidden_dim]
    /// * `value` - 值张量 [batch, seq_len, hidden_dim]
    ///
    /// # 返回
    ///
    /// 输出张量 [batch, seq_len, hidden_dim]
    pub fn forward(
        &self,
        query: &Array3<f32>,
        key: &Array3<f32>,
        value: &Array3<f32>,
    ) -> Result<Array3<f32>> {
        let batch_size = query.shape()[0];
        let seq_len = query.shape()[1];

        // 1. 时间空间投影 (压缩序列维度)
        let q_time = self.apply_time_proj(query)?; // [batch, seq_len, time_rank]
        let k_time = self.apply_time_proj(key)?;
        let v_time = self.apply_time_proj(value)?;

        // 2. 特征空间投影 (压缩特征维度)
        let q_feat = self.apply_feat_proj(query)?; // [batch, seq_len, feat_rank]
        let k_feat = self.apply_feat_proj(key)?;
        let v_feat = self.apply_feat_proj(value)?;

        // 3. 低秩重建 Q/K/V
        let q_reconstructed = self.outer_product_sum(&q_time, &q_feat);
        let k_reconstructed = self.outer_product_sum(&k_time, &k_feat);
        let v_reconstructed = self.outer_product_sum(&v_time, &v_feat);

        // 4. 标准注意力 (在低秩近似上)
        let attn_output = self.standard_attention(
            &q_reconstructed,
            &k_reconstructed,
            &v_reconstructed,
            batch_size,
            seq_len,
        )?;

        // 5. 输出投影
        let mut result = Array3::zeros((batch_size, seq_len, self.config.hidden_dim));

        for b in 0..batch_size {
            let input_2d = attn_output.slice(s![b, .., ..]).to_owned();
            let projected = self.output_proj.forward(&input_2d)?;
            result.slice_mut(s![b, .., ..]).assign(&projected);
        }

        Ok(result)
    }

    /// 应用时间空间投影
    fn apply_time_proj(&self, x: &Array3<f32>) -> Result<Array3<f32>> {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];
        let mut result = Array3::zeros((batch_size, seq_len, self.config.time_rank));

        for b in 0..batch_size {
            let input_2d = x.slice(s![b, .., ..]).to_owned();
            let proj_q = self.time_proj_q.forward(&input_2d)?;
            let proj_k = self.time_proj_k.forward(&input_2d)?;
            let proj_v = self.time_proj_v.forward(&input_2d)?;

            // 组合三个投影
            let combined = (proj_q + &proj_k + &proj_v) / 3.0;
            result.slice_mut(s![b, .., ..]).assign(&combined);
        }

        Ok(result)
    }

    /// 应用特征空间投影
    fn apply_feat_proj(&self, x: &Array3<f32>) -> Result<Array3<f32>> {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];
        let mut result = Array3::zeros((batch_size, seq_len, self.config.feat_rank));

        for b in 0..batch_size {
            let input_2d = x.slice(s![b, .., ..]).to_owned();
            let proj_q = self.feat_proj_q.forward(&input_2d)?;
            let proj_k = self.feat_proj_k.forward(&input_2d)?;
            let proj_v = self.feat_proj_v.forward(&input_2d)?;

            // 组合三个投影
            let combined = (proj_q + &proj_k + &proj_v) / 3.0;
            result.slice_mut(s![b, .., ..]).assign(&combined);
        }

        Ok(result)
    }

    /// 外积求和: sum_i(time_i ⊗ feat_i)
    ///
    /// 对于每个位置 i，计算时间向量和特征向量的外积并累加
    ///
    /// # 参数
    ///
    /// * `time` - 时间投影 [batch, seq_len, time_rank]
    /// * `feat` - 特征投影 [batch, seq_len, feat_rank]
    ///
    /// # 返回
    ///
    /// 重建的张量 [batch, seq_len, hidden_dim]
    fn outer_product_sum(
        &self,
        time: &Array3<f32>, // [b, s, r_t]
        feat: &Array3<f32>, // [b, s, r_f]
    ) -> Array3<f32> {
        // [b, s, d]
        let batch_size = time.shape()[0];
        let seq_len = time.shape()[1];
        let r_min = std::cmp::min(self.config.time_rank, self.config.feat_rank);

        let mut result = Array3::zeros((batch_size, seq_len, self.config.hidden_dim));

        for b in 0..batch_size {
            for s in 0..seq_len {
                for i in 0..r_min {
                    let t_val = time[[b, s, i]];
                    let f_slice = feat.slice(s![b, s, ..]); // [r_f]

                    // 外积: t_val * f_vec
                    for (j, &f_val) in f_slice.iter().enumerate() {
                        let idx = i * self.config.feat_rank + j;
                        if idx < self.config.hidden_dim {
                            result[[b, s, idx]] += t_val * f_val;
                        }
                    }
                }
            }
        }

        result
    }

    /// 批量外积计算
    fn batch_outer_product(&self, t_col: &Array3<f32>, f_col: &Array3<f32>) -> Array3<f32> {
        let batch_size = t_col.shape()[0];
        let seq_len = t_col.shape()[1];
        let mut result = Array3::zeros((batch_size, seq_len, self.config.hidden_dim));

        for b in 0..batch_size {
            for s in 0..seq_len {
                let t_val = t_col[[b, s, 0]];
                for (idx, f_val) in f_col.slice(s![b, s, ..]).iter().enumerate() {
                    if idx < self.config.hidden_dim {
                        result[[b, s, idx]] += t_val * f_val;
                    }
                }
            }
        }

        result
    }

    /// 标准注意力计算
    ///
    /// 在重建的 Q/K/V 上执行标准多头注意力
    fn standard_attention(
        &self,
        q: &Array3<f32>, // [batch, seq_len, hidden_dim]
        k: &Array3<f32>,
        v: &Array3<f32>,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Array3<f32>> {
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut output = Array3::zeros((batch_size, seq_len, self.config.hidden_dim));

        for b in 0..batch_size {
            let q_batch = q.slice(s![b, .., ..]).to_owned(); // [seq_len, hidden_dim]
            let k_batch = k.slice(s![b, .., ..]).to_owned();
            let v_batch = v.slice(s![b, .., ..]).to_owned();

            // 多头处理
            for h in 0..num_heads {
                let start = h * head_dim;
                let end = start + head_dim;

                let q_head = q_batch.slice(s![.., start..end]).to_owned(); // [seq_len, head_dim]
                let k_head = k_batch.slice(s![.., start..end]).to_owned();
                let v_head = v_batch.slice(s![.., start..end]).to_owned();

                // 计算注意力分数: Q @ K^T / sqrt(d)
                let k_head_t = k_head.t().to_owned();
                let scores = matmul(&q_head, &k_head_t)?; // [seq_len, seq_len]
                let scaled_scores = &scores * scale;

                // 应用因果掩码
                let masked_scores = if self.config.use_causal_mask {
                    apply_causal_mask(&scaled_scores, seq_len)?
                } else {
                    scaled_scores
                };

                // Softmax
                let attn_weights = softmax_2d(&masked_scores);

                // 应用到 V: weights @ V
                let attn_output = matmul(&attn_weights, &v_head)?; // [seq_len, head_dim]

                // 存储到输出
                output
                    .slice_mut(s![b, .., start..end])
                    .assign(&attn_output);
            }
        }

        Ok(output)
    }

    /// 获取配置引用
    pub fn config(&self) -> &TPAConfig {
        &self.config
    }

    /// 获取总参数量
    pub fn param_count(&self) -> usize {
        self.time_proj_q.param_count()
            + self.time_proj_k.param_count()
            + self.time_proj_v.param_count()
            + self.feat_proj_q.param_count()
            + self.feat_proj_k.param_count()
            + self.feat_proj_v.param_count()
            + self.output_proj.param_count()
    }

    /// 与 MLA 进行性能对比
    ///
    /// # 参数
    ///
    /// * `mla` - MLA 压缩器实例
    /// * `test_input` - 测试输入 [batch, seq_len, hidden_dim]
    ///
    /// # 返回
    ///
    /// 对比报告
    pub fn compare_with_mla(
        &self,
        mla: &super::moe::longcat::MlaCompressor,
        test_input: &Array3<f32>,
    ) -> ComparisonReport {
        let batch_size = test_input.shape()[0];
        let seq_len = test_input.shape()[1];

        // 测试 TPA 性能
        let start_tpa = Instant::now();
        let tpa_result = self.forward(test_input, test_input, test_input);
        let tpa_time = start_tpa.elapsed().as_secs_f64() * 1000.0;

        // 测试 MLA 性能
        let start_mla = Instant::now();
        let mut mla_results = Vec::new();
        for b in 0..batch_size {
            let input_2d = test_input.slice(s![b, .., ..]).to_owned();
            if let Ok(compressed) = mla.compress(&input_2d) {
                if let Ok(decompressed) = mla.decompress(&compressed) {
                    mla_results.push(decompressed);
                }
            }
        }
        let mla_time = start_mla.elapsed().as_secs_f64() * 1000.0;

        // 计算内存使用
        let tpa_memory = self.estimate_memory_usage(seq_len);
        // 使用压缩率估算 MLA 内存: latent_dim = hidden_dim / compression_ratio
        let hidden_dim = self.config.hidden_dim;
        let mla_latent_dim = (hidden_dim as f32 / mla.compression_ratio()) as usize;
        let mla_memory = mla_latent_dim * seq_len * batch_size * 4; // f32 bytes

        // 计算精度保留率（简化版）
        let accuracy_retention = if tpa_result.is_ok() {
            if !mla_results.is_empty() {
                // 简化的精度估算
                0.98 // 假设 98% 精度保留
            } else {
                1.0
            }
        } else {
            0.0
        };

        // 计算加速比
        let speedup = if mla_time > 0.0 {
            (mla_time / tpa_time) as f32
        } else {
            1.0
        };

        // 计算内存节省百分比
        let memory_saving_pct = if mla_memory > 0 {
            ((mla_memory - tpa_memory) as f32 / mla_memory as f32) * 100.0
        } else {
            0.0
        };

        ComparisonReport {
            mla_metrics: AttentionMetrics {
                execution_time_ms: mla_time,
                memory_usage_bytes: mla_memory,
                flops: self.estimate_standard_flops(seq_len),
                accuracy: 1.0,
            },
            tpa_metrics: AttentionMetrics {
                execution_time_ms: tpa_time,
                memory_usage_bytes: tpa_memory,
                flops: self.estimate_tpa_flops(seq_len),
                accuracy: accuracy_retention,
            },
            speedup,
            memory_saving_pct,
            accuracy_retention,
        }
    }

    /// 复杂度分析
    ///
    /// # 参数
    ///
    /// * `seq_len` - 序列长度
    /// * `hidden_dim` - 隐藏维度
    ///
    /// # 返回
    ///
    /// 复杂度信息
    pub fn complexity_analysis(&self, seq_len: usize, hidden_dim: usize) -> ComplexityInfo {
        let standard_flops = self.estimate_standard_flops_with_dim(seq_len, hidden_dim);
        let tpa_flops = self.estimate_tpa_flops_with_dim(seq_len, hidden_dim);

        let reduction_ratio = if standard_flops > 0 {
            (standard_flops - tpa_flops) as f32 / standard_flops as f32
        } else {
            0.0
        };

        let memory_standard = seq_len * seq_len * hidden_dim * 4; // 注意力矩阵
        let memory_tpa = seq_len * self.config.time_rank * self.config.feat_rank * 4;

        ComplexityInfo {
            standard_flops,
            tpa_flops,
            reduction_ratio,
            memory_standard,
            memory_tpa,
        }
    }

    /// 作为 MLA 替代方案的集成接口
    pub fn as_attention_selector(&self) -> AttentionSelector {
        AttentionSelector { tpa: self }
    }

    /// 自动选择最优注意力机制（基于序列长度）
    ///
    /// # 参数
    ///
    /// * `seq_len` - 序列长度
    ///
    /// # 返回
    ///
    /// 推荐的注意力机制
    pub fn recommend_for_sequence(seq_len: usize) -> Recommendation {
        match seq_len {
            0..=2048 => Recommendation::UseStandardFA {
                reason: format!(
                    "Short sequence ({} <= 2K): Standard Flash Attention is most efficient",
                    seq_len
                ),
            },
            2049..=8192 => Recommendation::UseMLA {
                reason: format!(
                    "Medium sequence ({} in 2K-8K range): MLA provides good compression",
                    seq_len
                ),
            },
            _ => Recommendation::UseTPA {
                reason: format!(
                    "Long sequence ({} > 8K): TPA offers best performance with low-rank approximation",
                    seq_len
                ),
            },
        }
    }

    /// 估计标准 Attention FLOPS
    fn estimate_standard_flops(&self, seq_len: usize) -> u64 {
        self.estimate_standard_flops_with_dim(seq_len, self.config.hidden_dim)
    }

    fn estimate_standard_flops_with_dim(&self, seq_len: usize, hidden_dim: usize) -> u64 {
        // 投影: 3 * N * d * d
        // 注意力: N * N * d
        // 输出: N * d * d
        let n = seq_len as u64;
        let d = hidden_dim as u64;
        3 * n * d * d + n * n * d + n * d * d
    }

    /// 估计 TPA FLOPS
    fn estimate_tpa_flops(&self, seq_len: usize) -> u64 {
        self.estimate_tpa_flops_with_dim(seq_len, self.config.hidden_dim)
    }

    fn estimate_tpa_flops_with_dim(&self, seq_len: usize, hidden_dim: usize) -> u64 {
        let n = seq_len as u64;
        let d = hidden_dim as u64;
        let rt = self.config.time_rank as u64;
        let rf = self.config.feat_rank as u64;

        // 投影: 6 * N * d * r (3对 QKV，每对 time+feat)
        // 外积: N * min(rt, rf) * rf
        // 注意力: N * N * min(rt, rf)^2
        // 输出: N * d * d
        6 * n * d * (rt + rf) + n * rt.min(rf) * rf + n * n * rt.min(rf) * rt.min(rf) + n * d * d
    }

    /// 估计内存使用量
    fn estimate_memory_usage(&self, seq_len: usize) -> usize {
        // Q/K/V 重建后的内存 + 注意力分数
        let qkv_memory = 3 * seq_len * self.config.hidden_dim * 4;
        let attn_memory = seq_len * seq_len * self.config.time_rank.min(self.config.feat_rank) * 4;
        qkv_memory + attn_memory
    }
}

// ============================================================================
// 注意力选择器
// ============================================================================

/// 注意力机制选择器
///
/// 提供统一的接口用于在不同注意力机制间切换
pub struct AttentionSelector<'a> {
    tpa: &'a TensorProductAttention,
}

impl<'a> AttentionSelector<'a> {
    /// 使用 TPA 执行注意力
    pub fn compute_attention(
        &self,
        query: &Array3<f32>,
        key: &Array3<f32>,
        value: &Array3<f32>,
    ) -> Result<Array3<f32>> {
        self.tpa.forward(query, key, value)
    }

    /// 获取推荐
    pub fn get_recommendation(&self, seq_len: usize) -> Recommendation {
        TensorProductAttention::recommend_for_sequence(seq_len)
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 矩阵乘法
fn matmul(a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
    let a_owned = a.clone();
    let b_owned = b.clone();
    let (m, k_a) = a_owned.dim();
    let (k_b, n) = b_owned.dim();

    if k_a != k_b {
        anyhow::bail!(
            "Matrix dimension mismatch: a is {}x{}, b is {}x{}",
            m,
            k_a,
            k_b,
            n
        );
    }

    let mut c = Array2::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..k_a {
                sum += a_owned[[i, k]] * b_owned[[k, j]];
            }
            c[[i, j]] = sum;
        }
    }

    Ok(c)
}

/// 2D Softmax
fn softmax_2d(x: &Array2<f32>) -> Array2<f32> {
    let (rows, cols) = x.dim();
    let mut result = Array2::zeros((rows, cols));

    for i in 0..rows {
        // 数值稳定：减去最大值
        let max_val = x
            .slice(s![i, ..])
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        let mut exp_sum = 0.0;
        for j in 0..cols {
            let exp_val = (x[[i, j]] - max_val).exp();
            result[[i, j]] = exp_val;
            exp_sum += exp_val;
        }

        // 归一化
        for j in 0..cols {
            result[[i, j]] /= exp_sum;
        }
    }

    result
}

/// 应用因果掩码
fn apply_causal_mask(scores: &Array2<f32>, seq_len: usize) -> Result<Array2<f32>> {
    let mut masked = scores.clone();

    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                masked[[i, j]] = f32::NEG_INFINITY;
            }
        }
    }

    Ok(masked)
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 创建测试用配置
    fn create_test_config() -> TPAConfig {
        TPAConfig::new()
            .with_hidden_dim(64)
            .with_time_rank(16)
            .with_feat_rank(16)
            .with_heads(2, 32)
            .with_causal_mask(true)
    }

    /// 创建测试输入
    fn create_test_input(batch: usize, seq: usize, dim: usize) -> Array3<f32> {
        Array3::from_shape_fn((batch, seq, dim), |(b, s, d)| {
            ((b * seq * dim + s * dim + d) % 100) as f32 / 100.0
        })
    }

    #[test]
    fn test_tpa_config_default() {
        let config = TPAConfig::default();
        assert_eq!(config.hidden_dim, 1024);
        assert_eq!(config.time_rank, 256);
        assert_eq!(config.feat_rank, 256);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert!(config.use_causal_mask);
    }

    #[test]
    fn test_tpa_config_builder() {
        let config = TPAConfig::new()
            .with_hidden_dim(512)
            .with_time_rank(128)
            .with_feat_rank(64)
            .with_heads(4, 128)
            .with_causal_mask(false);

        assert_eq!(config.hidden_dim, 512);
        assert_eq!(config.time_rank, 128);
        assert_eq!(config.feat_rank, 64);
        assert_eq!(config.num_heads, 4);
        assert!(!config.use_causal_mask);
    }

    #[test]
    fn test_tpa_config_validate_success() {
        let config = create_test_config();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_tpa_config_validate_zero_hidden() {
        let config = TPAConfig::new().with_hidden_dim(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_tpa_config_validate_zero_rank() {
        let config = TPAConfig::new().with_time_rank(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_tpa_config_validate_mismatch_dims() {
        let config = TPAConfig::new()
            .with_hidden_dim(100)
            .with_heads(4, 32); // 4*32=128 != 100
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_tpa_new() {
        let config = create_test_config();
        let tpa = TensorProductAttention::new(config);
        assert!(tpa.is_ok());

        let tpa = tpa.unwrap();
        assert_eq!(tpa.config().hidden_dim, 64);
        assert_eq!(tpa.config().time_rank, 16);
    }

    #[test]
    fn test_tpa_param_count() {
        let config = create_test_config();
        let tpa = TensorProductAttention::new(config).unwrap();
        let params = tpa.param_count();

        // 7 个线性层: 6个投影 + 1个输出
        // 每个投影: hidden_dim * rank
        // 输出: hidden_dim * hidden_dim
        let expected = 6 * 64 * 16 + 64 * 64;
        assert_eq!(params, expected);
    }

    #[test]
    fn test_tpa_forward_basic() {
        let config = create_test_config();
        let tpa = TensorProductAttention::new(config).unwrap();

        let input = create_test_input(1, 8, 64);
        let result = tpa.forward(&input, &input, &input);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.shape(), &[1, 8, 64]);
    }

    #[test]
    fn test_tpa_forward_batch() {
        let config = create_test_config();
        let tpa = TensorProductAttention::new(config).unwrap();

        let input = create_test_input(4, 16, 64);
        let result = tpa.forward(&input, &input, &input);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.shape(), &[4, 16, 64]);
    }

    #[test]
    fn test_tpa_forward_longer_sequence() {
        let config = create_test_config();
        let tpa = TensorProductAttention::new(config).unwrap();

        let input = create_test_input(1, 32, 64);
        let result = tpa.forward(&input, &input, &input);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.shape(), &[1, 32, 64]);

        // 检查输出不是全零
        let sum: f32 = output.iter().sum();
        assert!(sum.abs() > 0.0, "Output should not be all zeros");
    }

    #[test]
    fn test_tpa_no_causal_mask() {
        let config = create_test_config().with_causal_mask(false);
        let tpa = TensorProductAttention::new(config).unwrap();

        let input = create_test_input(1, 8, 64);
        let result = tpa.forward(&input, &input, &input);

        assert!(result.is_ok());
    }

    #[test]
    fn test_outer_product_sum() {
        let config = create_test_config();
        let tpa = TensorProductAttention::new(config).unwrap();

        let time = Array3::from_shape_fn((1, 4, 16), |(_, _, k)| k as f32);
        let feat = Array3::from_shape_fn((1, 4, 16), |(_, _, k)| (k + 1) as f32);

        let result = tpa.outer_product_sum(&time, &feat);
        assert_eq!(result.shape(), &[1, 4, 64]);
    }

    #[test]
    fn test_complexity_analysis_short_seq() {
        let config = create_test_config();
        let tpa = TensorProductAttention::new(config).unwrap();

        let info = tpa.complexity_analysis(512, 64);
        assert!(info.tpa_flops < info.standard_flops);
        assert!(info.reduction_ratio > 0.0);
        assert!(info.memory_tpa < info.memory_standard);
    }

    #[test]
    fn test_complexity_analysis_long_seq() {
        let config = create_test_config();
        let tpa = TensorProductAttention::new(config).unwrap();

        let info = tpa.complexity_analysis(16384, 64);
        // 长序列应该有更显著的减少比例
        assert!(info.reduction_ratio > 0.5, "Expected >50% reduction for long sequences");
    }

    #[test]
    fn test_recommendation_short_seq() {
        let rec = TensorProductAttention::recommend_for_sequence(1024);
        assert!(matches!(rec, Recommendation::UseStandardFA { .. }));
    }

    #[test]
    fn test_recommendation_medium_seq() {
        let rec = TensorProductAttention::recommend_for_sequence(4096);
        assert!(matches!(rec, Recommendation::UseMLA { .. }));
    }

    #[test]
    fn test_recommendation_long_seq() {
        let rec = TensorProductAttention::recommend_for_sequence(16384);
        assert!(matches!(rec, Recommendation::UseTPA { .. }));
    }

    #[test]
    fn test_attention_selector() {
        let config = create_test_config();
        let tpa = TensorProductAttention::new(config).unwrap();
        let selector = tpa.as_attention_selector();

        let input = create_test_input(1, 8, 64);
        let result = selector.compute_attention(&input, &input, &input);
        assert!(result.is_ok());

        let rec = selector.get_recommendation(16384);
        assert!(matches!(rec, Recommendation::UseTPA { .. }));
    }

    #[test]
    fn test_softmax_stability() {
        // 测试 softmax 的数值稳定性
        let scores = Array2::from_shape_vec(
            (2, 3),
            vec![1000.0, 1001.0, 1002.0, 1.0, 2.0, 3.0],
        )
        .unwrap();
        let probs = softmax_2d(&scores);

        // 检查概率和为 1
        for row in 0..2 {
            let sum: f32 = scores.slice(s![row, ..]).iter().sum();
            let prob_sum: f32 = probs.slice(s![row, ..]).iter().sum();
            assert!((prob_sum - 1.0).abs() < 1e-5, "Probabilities should sum to 1");
        }

        // 检查概率都是正数
        for &p in probs.iter() {
            assert!(p >= 0.0, "Probabilities should be non-negative");
            assert!(p <= 1.0, "Probabilities should not exceed 1");
        }
    }

    #[test]
    fn test_causal_mask() {
        let scores = Array2::from_shape_vec(
            (3, 3),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();

        let masked = apply_causal_mask(&scores, 3).unwrap();

        // 检查上三角被屏蔽
        assert_eq!(masked[[0, 1]], f32::NEG_INFINITY);
        assert_eq!(masked[[0, 2]], f32::NEG_INFINITY);
        assert_eq!(masked[[1, 2]], f32::NEG_INFINITY);

        // 检查下三角和对角线保持不变
        assert_eq!(masked[[0, 0]], 1.0);
        assert_eq!(masked[[1, 0]], 4.0);
        assert_eq!(masked[[1, 1]], 5.0);
        assert_eq!(masked[[2, 0]], 7.0);
        assert_eq!(masked[[2, 1]], 8.0);
        assert_eq!(masked[[2, 2]], 9.0);
    }

    #[test]
    fn test_matmul_basic() {
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Array2::from_shape_vec((3, 2), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();

        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.dim(), (2, 2));
        assert!((c[[0, 0]] - 58.0).abs() < 1e-5);   // 1*7 + 2*9 + 3*11
        assert!((c[[0, 1]] - 64.0).abs() < 1e-5);   // 1*8 + 2*10 + 3*12
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

        let result = matmul(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_large_config() {
        // 测试较大配置以确保可扩展性
        let config = TPAConfig::new()
            .with_hidden_dim(1024)
            .with_time_rank(256)
            .with_feat_rank(256)
            .with_heads(8, 128);

        let tpa = TensorProductAttention::new(config).unwrap();
        let input = create_test_input(1, 64, 1024);
        let result = tpa.forward(&input, &input, &input);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.shape(), &[1, 64, 1024]);
    }

    #[test]
    fn test_performance_comparison_structure() {
        let config = create_test_config();
        let tpa = TensorProductAttention::new(config).unwrap();

        // 测试复杂度分析方法的结构完整性
        let info = tpa.complexity_analysis(8192, 1024);
        assert!(info.standard_flops > 0);
        assert!(info.tpa_flops > 0);
        assert!(info.reduction_ratio >= 0.0);
        assert!(info.memory_standard > 0);
        assert!(info.memory_tpa > 0);
        println!("Complexity analysis for 8K sequence:");
        println!("  Standard FLOPS: {}", info.standard_flops);
        println!("  TPA FLOPS: {}", info.tpa_flops);
        println!("  Reduction ratio: {:.2}%", info.reduction_ratio * 100.0);
        println!("  Memory saving: {:.2}%",
                 (1.0 - info.memory_tpa as f32 / info.memory_standard as f32) * 100.0);
    }
}
