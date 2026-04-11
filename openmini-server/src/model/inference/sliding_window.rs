//! Sliding Window Attention (Gemma3 风格)
//! 
//! 5层 Local Sliding Window + 1层 Global Attention 交替模式

use ndarray::{Array2, Axis};

#[derive(Debug, Clone, Copy)]
pub enum AttentionMode {
    /// 全局注意力 - 所有位置都可以attend到所有其他位置
    Global,
    /// 滑动窗口注意力 - 只能attend到窗口内的位置
    SlidingWindow { window_size: usize },
}

pub struct SlidingWindowConfig {
    /// 总层数
    pub num_layers: usize,
    /// 滑动窗口大小
    pub window_size: usize,
    /// 层级attention模式列表
    pub layer_modes: Vec<AttentionMode>,
}

impl SlidingWindowConfig {
    /// 创建 Gemma3 默认配置: 5 local + 1 global 循环
    pub fn gemma3_default(num_layers: usize, window_size: usize) -> Self {
        let mut layer_modes = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            if i % 6 == 5 {
                layer_modes.push(AttentionMode::Global);
            } else {
                layer_modes.push(AttentionMode::SlidingWindow { window_size });
            }
        }
        Self { num_layers, window_size, layer_modes }
    }
    
    /// 获取指定层的attention模式
    pub fn get_layer_mode(&self, layer_idx: usize) -> AttentionMode {
        self.layer_modes.get(layer_idx).copied()
            .unwrap_or(AttentionMode::SlidingWindow { window_size: self.window_size })
    }
}

/// 构建滑动窗口注意力掩码
/// 
/// # 参数
/// - seq_len: 序列长度
/// - window_size: 窗口大小
/// 
/// # 返回
/// - (seq_len, seq_len) 的布尔掩码，True表示允许attend
pub fn build_sliding_window_mask(seq_len: usize, window_size: usize) -> Array2<bool> {
    let half_window = window_size / 2;
    let mut mask = Array2::from_elem((seq_len, seq_len), false);
    
    for i in 0..seq_len {
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(seq_len);
        for j in start..end {
            mask[[i, j]] = true;
        }
    }
    
    mask
}

/// 应用注意力掩码到分数矩阵
/// 
/// 将掩码外的位置设为负无穷大，softmax后变为0
pub fn apply_attention_mask(scores: &mut Array2<f32>, mask: &Array2<bool>) {
    let neg_inf = f32::NEG_INFINITY;
    for ((i, j), val) in scores.indexed_iter_mut() {
        if !mask[[i, j]] {
            *val = neg_inf;
        }
    }
}

/// 滑动窗口注意力计算
/// 
/// 完整的前向传播：Q @ K^T → apply mask → softmax → @ V
pub fn sliding_window_attention(
    q: &Array2<f32>,   // (seq_len, head_dim)
    k: &Array2<f32>,   // (kv_seq_len, head_dim)
    v: &Array2<f32>,   // (kv_seq_len, head_dim_v)
    window_size: usize,
    scale: f32,
) -> Array2<f32> {
    let seq_len = q.nrows();
    let kv_len = k.nrows();
    let _head_dim = q.ncols();
    
    let scale = 1.0 / (scale as f32).sqrt();
    
    let mut scores = q.dot(&k.t()) * scale;
    
    let mut mask = build_sliding_window_mask(seq_len, window_size);
    if kv_len > seq_len {
        mask = mask.clone();
    }
    apply_attention_mask(&mut scores, &mask);
    
    let attn_weights = softmax_2d(&scores);
    attn_weights.dot(v)
}

fn softmax_2d(x: &Array2<f32>) -> Array2<f32> {
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_x = x.mapv(|v| (v - max_val).exp());
    let sum_exp = exp_x.sum_axis(Axis(1));
    exp_x / &sum_exp.insert_axis(Axis(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gemma3_pattern() {
        let config = SlidingWindowConfig::gemma3_default(12, 128);
        assert!(matches!(config.get_layer_mode(5), AttentionMode::Global));
        assert!(matches!(config.get_layer_mode(11), AttentionMode::Global));
        assert!(matches!(config.get_layer_mode(0), AttentionMode::SlidingWindow { .. }));
    }
    
    #[test]
    fn test_sliding_window_mask() {
        let mask = build_sliding_window_mask(10, 4);
        assert!(mask[[0, 0]]);
        assert!(!mask[[0, 9]]);
        assert!(mask[[5, 4]]);
        assert!(mask[[5, 6]]);
    }
    
    #[test]
    fn test_sliding_window_attention_forward() {
        let q = Array2::from_shape_fn((8, 16), |(i, j)| (i * 16 + j) as f32 * 0.1);
        let k = Array2::from_shape_fn((8, 16), |(i, j)| (i * 16 + j + 1) as f32 * 0.1);
        let v = Array2::from_shape_fn((8, 16), |(i, j)| (i * 16 + j + 2) as f32 * 0.1);
        
        let output = sliding_window_attention(&q, &k, &v, 4, 16.0);
        assert_eq!(output.shape(), &[8, 16]);
    }
}
