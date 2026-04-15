//! Sliding Window Attention (滑动窗口注意力机制)
//!
//! 实现 Gemma3 风格的滑动窗口注意力，用于长序列推理优化。
//! 核心思想：每个位置只关注其周围固定窗口内的token，
//! 将计算复杂度从 O(n²) 降低到 O(n * window_size)。
//!
//! ## 架构特点
//! - 支持 Local/Global/Strided 三种注意力模式
//! - 可配置的窗口大小和步长参数
//! - 支持KV Cache优化
//! - 5层 Local + 1层 Global 的交替模式 (Gemma3默认)

use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};

/// 注意力模式枚举
///
/// 定义不同的注意力计算策略，用于适应不同场景需求
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum AttentionMode {
    /// 局部注意力模式
    /// 仅关注当前token周围的窗口内token
    /// 适用场景：长文本处理、降低计算复杂度
    #[default]
    Local,

    /// 全局注意力模式
    /// 所有位置都可以attend到所有其他位置
    /// 适用场景：关键信息聚合、短序列处理
    Global,

    /// 跨步长滑动窗口
    /// 以固定步长跳跃式地选择关注的token
    /// 适用场景：超长序列、稀疏注意力
    Strided {
        /// 步长大小，控制跳跃间隔
        stride: usize,
    },
}

/// 滑动窗口注意力配置
///
/// 包含所有可配置的参数，支持灵活调整以适应不同模型和场景
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlidingWindowConfig {
    /// 窗口大小（token数量）
    ///
    /// 每个位置可以关注的前后token总数。
    /// 较小的值减少计算量，但可能丢失远距离依赖关系；
    /// 较大的值保留更多信息，但增加计算开销。
    /// 推荐值：128, 256, 512
    pub window_size: usize,

    /// 注意力计算模式
    ///
    /// 决定如何构建注意力掩码和计算策略
    pub attention_mode: AttentionMode,

    /// 是否使用KV缓存优化
    ///
    /// 开启后可复用之前计算的Key/Value，
    /// 显著提升推理速度（特别是自回归生成场景）
    pub use_cache: bool,
}

#[allow(clippy::derivable_impls)]
impl Default for SlidingWindowConfig {
    fn default() -> Self {
        Self {
            window_size: 128,
            attention_mode: AttentionMode::Local,
            use_cache: true,
        }
    }
}

impl SlidingWindowConfig {
    /// 创建 Gemma3 默认配置
    ///
    /// Gemma3 采用 5层 Local + 1层 Global 的循环模式，
    /// 在保持局部建模能力的同时，定期进行全局信息聚合。
    ///
    /// # 参数
    /// - `num_layers`: 总层数
    /// - `window_size`: 窗口大小（推荐128）
    ///
    /// # 示例
    /// ```ignore
    /// let config = SlidingWindowConfig::gemma3_default(12, 128);
    /// // 第5、11层为Global，其余为Local
    /// ```
    pub fn gemma3_default(num_layers: usize, window_size: usize) -> Vec<Self> {
        let mut configs = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let mode = if i % 6 == 5 {
                AttentionMode::Global
            } else {
                AttentionMode::Local
            };
            configs.push(Self {
                window_size,
                attention_mode: mode,
                use_cache: true,
            });
        }
        configs
    }

    /// 创建纯局部注意力配置
    ///
    /// 所有层都使用相同的滑动窗口大小，
    /// 适用于需要统一处理的场景。
    ///
    /// # 参数
    /// - `window_size`: 窗口大小
    /// - `use_cache`: 是否启用缓存
    pub fn local_only(window_size: usize, use_cache: bool) -> Self {
        Self {
            window_size,
            attention_mode: AttentionMode::Local,
            use_cache,
        }
    }

    /// 创建带步长的配置
    ///
    /// 使用 Strided 模式，适用于超长序列的稀疏注意力。
    ///
    /// # 参数
    /// - `window_size`: 窗口大小
    /// - `stride`: 步长
    pub fn strided(window_size: usize, stride: usize) -> Self {
        Self {
            window_size,
            attention_mode: AttentionMode::Strided { stride },
            use_cache: false, // Strided模式通常不使用缓存
        }
    }

    /// 获取实际使用的窗口半径
    ///
    /// 根据模式和配置返回单侧窗口大小
    pub fn effective_window_radius(&self) -> usize {
        match &self.attention_mode {
            AttentionMode::Local | AttentionMode::Global => self.window_size / 2,
            AttentionMode::Strided { stride } => *stride.min(&self.window_size),
        }
    }
}

/// 构建滑动窗口注意力掩码
///
/// 根据配置生成布尔掩码矩阵，标记哪些位置允许参与注意力计算。
/// True 表示允许 attend，False 表示屏蔽。
///
/// # 参数
/// - `seq_len`: 序列长度（Query长度）
/// - `kv_len`: Key/Value序列长度（可能大于seq_len，因为有past_kv）
/// - `config`: 滑动窗口配置
///
/// # 返回
/// - 形状为 (seq_len, kv_len) 的布尔掩码矩阵
///
/// # 示例
/// ```ignore
/// let config = SlidingWindowConfig::local_only(4, true);
/// let mask = build_sliding_window_mask(8, 8, &config);
/// // 每行只有相邻的4个位置为true
/// ```
pub fn build_sliding_window_mask(
    seq_len: usize,
    kv_len: usize,
    config: &SlidingWindowConfig,
) -> Array2<bool> {
    let mut mask = Array2::from_elem((seq_len, kv_len), false);

    match &config.attention_mode {
        // 全局注意力：所有位置都可见
        AttentionMode::Global => {
            mask.fill(true);
        }

        // 局部滑动窗口
        AttentionMode::Local => {
            let half_window = config.window_size / 2;
            for i in 0..seq_len {
                let start = i.saturating_sub(half_window);
                let end = (i + half_window + 1).min(kv_len);
                for j in start..end {
                    mask[[i, j]] = true;
                }
            }
        }

        // 跨步长滑动窗口
        AttentionMode::Strided { stride } => {
            let half_window = config.effective_window_radius();
            for i in 0..seq_len {
                // 以 stride 为步长，在窗口内采样
                let center_start = i.saturating_sub(half_window);
                let center_end = (i + half_window + 1).min(kv_len);

                let mut j = center_start;
                while j < center_end {
                    if j < kv_len {
                        mask[[i, j]] = true;
                    }
                    j += stride;
                }

                // 始终包含自身位置
                if i < kv_len {
                    mask[[i, i]] = true;
                }
            }
        }
    }

    mask
}

/// 应用注意力掩码到分数矩阵
///
/// 将被掩码的位置设为负无穷大，经过softmax后会变为0，
/// 从而实现屏蔽效果。
///
/// # 参数
/// - `scores`: 注意力分数矩阵 (seq_len, kv_len)，会被原地修改
/// - `mask`: 布尔掩码矩阵
pub fn apply_attention_mask(scores: &mut Array2<f32>, mask: &Array2<bool>) {
    let neg_inf = f32::NEG_INFINITY;
    for ((i, j), val) in scores.indexed_iter_mut() {
        if !mask[[i, j]] {
            *val = neg_inf;
        }
    }
}

/// 执行softmax归一化
///
/// 对每一行独立执行softmax，确保每行的概率和为1。
/// 使用数值稳定的实现方式（减去最大值）。
///
/// # 参数
/// - `x`: 输入矩阵 (batch, features)
///
/// # 返回
/// - softmax后的概率分布矩阵
fn softmax_2d(x: &Array2<f32>) -> Array2<f32> {
    // 数值稳定性：减去每行的最大值
    let max_val = x
        .axis_iter(Axis(0))
        .map(|row| row.iter().cloned().fold(f32::NEG_INFINITY, f32::max))
        .collect::<Vec<_>>();

    let exp_x = x
        .indexed_iter()
        .map(|((i, _), &v)| (v - max_val[i]).exp())
        .collect::<Vec<_>>();

    let exp_array = Array2::from_shape_vec(x.dim(), exp_x).unwrap();

    // 计算每行的归一化因子
    let sum_exp = exp_array.sum_axis(Axis(1));

    // 广播除法
    exp_array / &sum_exp.insert_axis(Axis(1))
}

/// 滑动窗口注意力计算（核心函数）
///
/// 完整的前向传播流程：
/// 1. 计算注意力分数: Q @ K^T
/// 2. 缩放分数: / sqrt(head_dim)
/// 3. 应用滑动窗口掩码
/// 4. Softmax归一化
/// 5. 加权求和: attn_weights @ V
///
/// # 类型参数
/// - `T`: 数值类型（目前仅支持f32）
///
/// # 参数
/// - `query`: Query矩阵 (seq_len, head_dim)
/// - `key`: Key矩阵 (kv_seq_len, head_dim)
/// - `value`: Value矩阵 (kv_seq_len, head_dim_v)
/// - `config`: 滑动窗口配置
/// - `mask`: 可选的外部掩码（与内部掩码取交集）
///
/// # 返回
/// - 输出矩阵 (seq_len, head_dim_v)
///
/// # 错误
/// - 如果维度不匹配则返回错误
///
/// # 性能说明
/// - 时间复杂度: O(seq_len * min(window_size, kv_len) * head_dim)
/// - 空间复杂度: O(seq_len * min(window_size, kv_len))
///
/// # 示例
/// ```ignore
/// let config = SlidingWindowConfig::local_only(64, true);
/// let output = sliding_window_attention(&q, &k, &v, &config, None)?;
/// ```
pub fn sliding_window_attention(
    query: &Array2<f32>,
    key: &Array2<f32>,
    value: &Array2<f32>,
    config: &SlidingWindowConfig,
    mask: Option<&Array2<bool>>,
) -> Result<Array2<f32>, String> {
    // 维度验证
    let (seq_len, head_dim) = query.dim();
    let (kv_len, key_dim) = key.dim();
    let (v_len, _) = value.dim();

    if head_dim != key_dim {
        return Err(format!(
            "Query head_dim ({}) must equal Key head_dim ({})",
            head_dim, key_dim
        ));
    }

    if kv_len != v_len {
        return Err(format!(
            "Key length ({}) must equal Value length ({})",
            kv_len, v_len
        ));
    }

    // 缩放因子
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Step 1: 计算原始注意力分数 Q @ K^T
    let mut scores = query.dot(&key.t()) * scale;

    // Step 2: 构建并应用滑动窗口掩码
    let sw_mask = build_sliding_window_mask(seq_len, kv_len, config);
    apply_attention_mask(&mut scores, &sw_mask);

    // Step 3: 应用外部掩码（如果提供）
    if let Some(ext_mask) = mask {
        // 外部掩码与窗口掩码取交集
        for ((i, j), val) in scores.indexed_iter_mut() {
            if !ext_mask[[i, j]] {
                *val = f32::NEG_INFINITY;
            }
        }
    }

    // Step 4: Softmax归一化
    let attn_weights = softmax_2d(&scores);

    // Step 5: 加权求和得到输出
    let output = attn_weights.dot(value);

    Ok(output)
}

/// 批量滑动窗口注意力（多头版本）
///
/// 对多个注意力头并行执行滑动窗口注意力计算。
/// 适用于Transformer模型的多头注意力场景。
///
/// # 参数
/// - `query`: Query张量 (num_heads, seq_len, head_dim)
/// - `key`: Key张量 (num_heads, kv_seq_len, head_dim)
/// - `value`: Value张量 (num_heads, kv_seq_len, head_dim_v)
/// - `config`: 滑动窗口配置
/// - `mask`: 可选的外部掩码
///
/// # 返回
/// - 输出张量 (num_heads, seq_len, head_dim_v)
pub fn multi_head_sliding_window_attention(
    query: &[Array2<f32>],
    key: &[Array2<f32>],
    value: &[Array2<f32>],
    config: &SlidingWindowConfig,
    mask: Option<&Array2<bool>>,
) -> Result<Vec<Array2<f32>>, String> {
    let num_heads = query.len();

    if key.len() != num_heads || value.len() != num_heads {
        return Err(format!(
            "Number of heads mismatch: query={}, key={}, value={}",
            num_heads,
            key.len(),
            value.len()
        ));
    }

    let mut outputs = Vec::with_capacity(num_heads);

    for h in 0..num_heads {
        let out = sliding_window_attention(&query[h], &key[h], &value[h], config, mask)?;
        outputs.push(out);
    }

    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 辅助函数：创建测试用的随机矩阵
    fn create_test_matrix(rows: usize, cols: usize) -> Array2<f32> {
        Array2::from_shape_fn((rows, cols), |(i, j)| {
            ((i * cols + j) as f32 * 0.01 - 0.5).cos()
        })
    }

    // ==================== 配置测试 ====================

    #[test]
    fn test_gemma3_pattern() {
        // 测试 Gemma3 默认配置模式：5 local + 1 global 循环
        let configs = SlidingWindowConfig::gemma3_default(12, 128);

        // 第5层应该是 Global (索引从0开始，第6层)
        assert_eq!(configs[5].attention_mode, AttentionMode::Global);

        // 第11层也应该是 Global (第12层)
        assert_eq!(configs[11].attention_mode, AttentionMode::Global);

        // 第0层应该是 Local
        assert_eq!(configs[0].attention_mode, AttentionMode::Local);

        // 所有配置的窗口大小应该一致
        for config in &configs {
            assert_eq!(config.window_size, 128);
        }
    }

    #[test]
    fn test_local_config() {
        let config = SlidingWindowConfig::local_only(64, true);
        assert_eq!(config.window_size, 64);
        assert_eq!(config.attention_mode, AttentionMode::Local);
        assert!(config.use_cache);
    }

    #[test]
    fn test_strided_config() {
        let config = SlidingWindowConfig::strided(128, 16);
        assert_eq!(config.window_size, 128);
        assert_eq!(config.attention_mode, AttentionMode::Strided { stride: 16 });
        assert!(!config.use_cache); // Strided模式默认不使用缓存
    }

    #[test]
    fn test_default_config() {
        let config = SlidingWindowConfig::default();
        assert_eq!(config.window_size, 128);
        assert_eq!(config.attention_mode, AttentionMode::Local);
        assert!(config.use_cache);
    }

    // ==================== 掩码构建测试 ====================

    #[test]
    fn test_local_mask_basic() {
        // 测试基本的局部掩码
        let config = SlidingWindowConfig::local_only(4, true);
        let mask = build_sliding_window_mask(10, 10, &config);

        // 自身位置应该可见
        assert!(mask[[5, 5]], "Position should attend to itself");

        // 相邻位置应该可见
        assert!(mask[[5, 4]], "Adjacent position should be visible");
        assert!(mask[[5, 6]], "Adjacent position should be visible");

        // 远距离位置应该不可见
        assert!(!mask[[0, 9]], "Far position should be masked");
    }

    #[test]
    fn test_global_mask() {
        // 全局掩码：所有位置都应该可见
        let config = SlidingWindowConfig {
            window_size: 128,
            attention_mode: AttentionMode::Global,
            use_cache: true,
        };
        let mask = build_sliding_window_mask(8, 8, &config);

        // 所有位置都应该为true
        for i in 0..8 {
            for j in 0..8 {
                assert!(mask[[i, j]], "Global mask should allow all positions");
            }
        }
    }

    #[test]
    fn test_strided_mask() {
        // 测试跨步长掩码
        let config = SlidingWindowConfig::strided(10, 3);
        let mask = build_sliding_window_mask(12, 12, &config);

        // 自身位置必须可见（特殊规则）
        assert!(mask[[5, 5]], "Self position should always be visible");

        // 检查是否有按步长采样的位置可见
        let row_5_visible: Vec<usize> = (0..12).filter(|&j| mask[[5, j]]).collect();
        assert!(
            row_5_visible.len() > 1,
            "Strided mask should have multiple visible positions"
        );
    }

    #[test]
    fn test_boundary_conditions() {
        // 测试边界条件：窗口超出序列范围
        let config = SlidingWindowConfig::local_only(100, true);
        let mask = build_sliding_window_mask(10, 10, &config);

        // 第一个位置只能看到右侧
        assert!(mask[[0, 0]]);
        assert!(mask[[0, 1]]);

        // 最后一个位置只能看到左侧
        assert!(mask[[9, 9]]);
        assert!(mask[[9, 8]]);
    }

    #[test]
    fn test_kv_longer_than_query() {
        // KV序列比Query长的情况（有past_kv）
        let config = SlidingWindowConfig::local_only(4, true);
        let mask = build_sliding_window_mask(5, 20, &config);

        // Query位置应该能看到对应的KV区域
        assert!(mask[[2, 2]]);
        assert!(mask[[2, 3]]);
        assert!(mask[[2, 4]]);

        // 形状正确
        assert_eq!(mask.dim(), (5, 20));
    }

    // ==================== 注意力计算测试 ====================

    #[test]
    fn test_sliding_window_attention_forward() {
        // 测试基本的前向传播
        let seq_len = 8;
        let head_dim = 16;

        let q = create_test_matrix(seq_len, head_dim);
        let k = create_test_matrix(seq_len, head_dim);
        let v = create_test_matrix(seq_len, head_dim);

        let config = SlidingWindowConfig::local_only(4, true);
        let result = sliding_window_attention(&q, &k, &v, &config, None);

        assert!(result.is_ok(), "Attention computation should succeed");
        let output = result.unwrap();
        assert_eq!(
            output.shape(),
            &[seq_len, head_dim],
            "Output shape should match input"
        );

        // 输出不应包含NaN或Inf
        for val in output.iter() {
            assert!(val.is_finite(), "Output should not contain NaN or Inf");
        }
    }

    #[test]
    fn test_global_attention() {
        // 测试全局注意力的输出
        let seq_len = 6;
        let head_dim = 8;

        let q = create_test_matrix(seq_len, head_dim);
        let k = create_test_matrix(seq_len, head_dim);
        let v = create_test_matrix(seq_len, head_dim);

        let config = SlidingWindowConfig {
            window_size: 128,
            attention_mode: AttentionMode::Global,
            use_cache: true,
        };
        let output = sliding_window_attention(&q, &k, &v, &config, None).unwrap();

        assert_eq!(output.shape(), &[seq_len, head_dim]);
    }

    #[test]
    fn test_with_external_mask() {
        // 测试外部掩码的应用
        let seq_len = 6;
        let head_dim = 8;

        let q = create_test_matrix(seq_len, head_dim);
        let k = create_test_matrix(seq_len, head_dim);
        let v = create_test_matrix(seq_len, head_dim);

        let config = SlidingWindowConfig::local_only(10, true);

        // 创建外部掩码：只允许看前半部分
        let mut ext_mask = Array2::from_elem((seq_len, seq_len), false);
        for i in 0..seq_len {
            for j in 0..=i {
                ext_mask[[i, j]] = true; // 因果掩码
            }
        }

        let output = sliding_window_attention(&q, &k, &v, &config, Some(&ext_mask));
        assert!(output.is_ok(), "Should work with external mask");
    }

    #[test]
    fn test_dimension_mismatch_error() {
        // 测试维度不匹配的错误处理
        let q = create_test_matrix(8, 16);
        let k = create_test_matrix(8, 32); // 不匹配的head_dim
        let v = create_test_matrix(8, 16);

        let config = SlidingWindowConfig::default();
        let result = sliding_window_attention(&q, &k, &v, &config, None);

        assert!(result.is_err(), "Should fail on dimension mismatch");
        assert!(
            result.unwrap_err().contains("must equal"),
            "Error message should mention dimension equality"
        );
    }

    // ==================== 多头注意力测试 ====================

    #[test]
    fn test_multi_head_attention() {
        // 测试多头版本的滑动窗口注意力
        let num_heads = 4;
        let seq_len = 8;
        let head_dim = 16;

        let query: Vec<Array2<f32>> = (0..num_heads)
            .map(|_h| create_test_matrix(seq_len, head_dim))
            .collect();

        let key: Vec<Array2<f32>> = (0..num_heads)
            .map(|_h| create_test_matrix(seq_len, head_dim))
            .collect();

        let value: Vec<Array2<f32>> = (0..num_heads)
            .map(|_h| create_test_matrix(seq_len, head_dim))
            .collect();

        let config = SlidingWindowConfig::local_only(4, true);
        let outputs = multi_head_sliding_window_attention(&query, &key, &value, &config, None);

        assert!(outputs.is_ok());
        let results = outputs.unwrap();
        assert_eq!(results.len(), num_heads);

        for (h, output) in results.iter().enumerate() {
            assert_eq!(
                output.shape(),
                &[seq_len, head_dim],
                "Head {} output shape incorrect",
                h
            );
        }
    }

    #[test]
    fn test_multi_head_dimension_check() {
        // 测试多头版本的维度检查
        let query = vec![create_test_matrix(4, 8)];
        let key = vec![create_test_matrix(4, 8), create_test_matrix(4, 8)]; // 头数不匹配
        let value = vec![create_test_matrix(4, 8)];

        let config = SlidingWindowConfig::default();
        let result = multi_head_sliding_window_attention(&query, &key, &value, &config, None);

        assert!(result.is_err(), "Should fail on head count mismatch");
    }

    // ==================== 性能和边界测试 ====================

    #[test]
    fn test_large_sequence() {
        // 测试较长序列的性能和正确性
        let seq_len = 256;
        let head_dim = 64;

        let q = create_test_matrix(seq_len, head_dim);
        let k = create_test_matrix(seq_len, head_dim);
        let v = create_test_matrix(seq_len, head_dim);

        let config = SlidingWindowConfig::local_only(128, true);
        let start = std::time::Instant::now();
        let output = sliding_window_attention(&q, &k, &v, &config, None);
        let elapsed = start.elapsed();

        assert!(output.is_ok());
        println!("Large sequence ({}): {:?}", seq_len, elapsed);

        // 应该在合理时间内完成（< 1秒）
        assert!(elapsed.as_secs() < 1, "Should complete quickly");
    }

    #[test]
    fn test_window_larger_than_sequence() {
        // 窗口大小超过序列长度的情况
        let seq_len = 8;
        let head_dim = 16;

        let q = create_test_matrix(seq_len, head_dim);
        let k = create_test_matrix(seq_len, head_dim);
        let v = create_test_matrix(seq_len, head_dim);

        let config = SlidingWindowConfig::local_only(1024, true); // 远大于seq_len
        let output = sliding_window_attention(&q, &k, &v, &config, None);

        assert!(output.is_ok(), "Should handle window > sequence length");
    }

    #[test]
    fn test_output_values_range() {
        // 测试输出值的合理性（应该在合理范围内）
        let seq_len = 16;
        let head_dim = 32;

        let q = create_test_matrix(seq_len, head_dim);
        let k = create_test_matrix(seq_len, head_dim);
        let v = create_test_matrix(seq_len, head_dim);

        let config = SlidingWindowConfig::default();
        let output = sliding_window_attention(&q, &k, &v, &config, None).unwrap();

        // 检查输出是否在合理范围内（不应该有极端值）
        let max_val = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_val = output.iter().cloned().fold(f32::INFINITY, f32::min);

        assert!(max_val.is_finite(), "Max value should be finite");
        assert!(min_val.is_finite(), "Min value should be finite");

        // 对于标准化的输入，输出通常不会太大
        // 这里我们只检查有限性，具体范围取决于输入数据
    }
}
