//! OpenMini 核心算子库
//!
//! 提供高性能的基础算子实现

use anyhow::Result;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// 矩阵乘法
pub fn matmul(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> Result<Array2<f32>> {
    if a.ncols() != b.nrows() {
        return Err(anyhow::anyhow!(
            "矩阵维度不匹配: {}x{} 与 {}x{}",
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols()
        ));
    }

    Ok(a.dot(b))
}

/// 向量加法
pub fn add(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Result<Array1<f32>> {
    if a.len() != b.len() {
        return Err(anyhow::anyhow!(
            "向量长度不匹配: {} vs {}",
            a.len(),
            b.len()
        ));
    }

    Ok(a + b)
}

/// 向量乘法（逐元素）
pub fn mul(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Result<Array1<f32>> {
    if a.len() != b.len() {
        return Err(anyhow::anyhow!(
            "向量长度不匹配: {} vs {}",
            a.len(),
            b.len()
        ));
    }

    Ok(a * b)
}

/// Softmax
pub fn softmax(x: &ArrayView1<f32>) -> Array1<f32> {
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_x: Array1<f32> = x.mapv(|v| (v - max_val).exp());
    let sum = exp_x.sum();
    exp_x / sum
}

/// LayerNorm
pub fn layer_norm(
    x: &ArrayView1<f32>,
    weight: &ArrayView1<f32>,
    bias: &ArrayView1<f32>,
    eps: f32,
) -> Result<Array1<f32>> {
    if x.len() != weight.len() || x.len() != bias.len() {
        return Err(anyhow::anyhow!("维度不匹配"));
    }

    let mean = x.mean().unwrap_or(0.0);
    let var = x.var(0.0);
    let std = (var + eps).sqrt();

    let normalized = x.mapv(|v| (v - mean) / std);
    Ok(&normalized * weight + bias)
}

/// RMSNorm
pub fn rms_norm(x: &ArrayView1<f32>, weight: &ArrayView1<f32>, eps: f32) -> Result<Array1<f32>> {
    if x.len() != weight.len() {
        return Err(anyhow::anyhow!("维度不匹配"));
    }

    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    let rms = (sum_sq / x.len() as f32 + eps).sqrt();

    Ok(x / rms * weight)
}

/// GELU激活函数
pub fn gelu(x: &ArrayView1<f32>) -> Array1<f32> {
    x.mapv(|v| {
        let cdf =
            0.5 * (1.0 + (2.0 / std::f32::consts::PI).sqrt() * (v + 0.044715 * v.powi(3)).tanh());
        v * cdf
    })
}

/// SiLU激活函数
pub fn silu(x: &ArrayView1<f32>) -> Array1<f32> {
    x.mapv(|v| v / (1.0 + (-v).exp()))
}

/// RoPE位置编码
pub fn rotary_embedding(x: &mut Array2<f32>, positions: &[usize], theta: f32) -> Result<()> {
    let seq_len = x.nrows();
    let head_dim = x.ncols();

    if positions.len() != seq_len {
        return Err(anyhow::anyhow!("位置长度不匹配"));
    }

    for (i, &pos) in positions.iter().enumerate() {
        for j in (0..head_dim).step_by(2) {
            let freq = 1.0 / theta.powf(j as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let cos = angle.cos();
            let sin = angle.sin();

            let x0 = x[[i, j]];
            let x1 = if j + 1 < head_dim { x[[i, j + 1]] } else { 0.0 };

            x[[i, j]] = x0 * cos - x1 * sin;
            if j + 1 < head_dim {
                x[[i, j + 1]] = x0 * sin + x1 * cos;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;
    use ndarray::arr2;

    #[test]
    fn test_softmax() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let result = softmax(&x.view());
        let sum: f32 = result.sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gelu() {
        let x = arr1(&[0.0, 1.0, -1.0]);
        let result = gelu(&x.view());
        assert!(result[0].abs() < 1e-5);
    }

    /// 测试：矩阵乘法正常情况（核心功能分支）
    #[test]
    fn test_matmul_normal() {
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[5.0, 6.0], [7.0, 8.0]]);

        let result = matmul(&a.view(), &b.view()).expect("矩阵乘法应该成功");
        assert_eq!(result.shape(), [2, 2]);
        assert!((result[[0, 0]] - 19.0).abs() < 1e-5); // 1*5 + 2*7 = 19
        assert!((result[[0, 1]] - 22.0).abs() < 1e-5); // 1*6 + 2*8 = 22
        assert!((result[[1, 0]] - 43.0).abs() < 1e-5); // 3*5 + 4*7 = 43
        assert!((result[[1, 1]] - 50.0).abs() < 1e-5); // 3*6 + 4*8 = 50
    }

    /// 测试：矩阵维度不匹配时返回错误（错误路径）
    #[test]
    fn test_matmul_dimension_mismatch() {
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]); // 2x2
        let b = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]); // 3x3

        let result = matmul(&a.view(), &b.view());
        assert!(result.is_err(), "维度不匹配应该返回错误");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("维度不匹配"),
            "错误信息应包含维度不匹配提示"
        );
    }

    /// 测试：向量加法正常情况（逐元素相加）
    #[test]
    fn test_add_normal() {
        let a = arr1(&[1.0, 2.0, 3.0]);
        let b = arr1(&[4.0, 5.0, 6.0]);

        let result = add(&a.view(), &b.view()).expect("向量加法应该成功");
        assert_eq!(result.len(), 3);
        assert!((result[0] - 5.0).abs() < 1e-5);
        assert!((result[1] - 7.0).abs() < 1e-5);
        assert!((result[2] - 9.0).abs() < 1e-5);
    }

    /// 测试：向量加法长度不匹配返回错误（错误路径）
    #[test]
    fn test_add_length_mismatch() {
        let a = arr1(&[1.0, 2.0]);
        let b = arr1(&[1.0, 2.0, 3.0]);

        let result = add(&a.view(), &b.view());
        assert!(result.is_err(), "长度不匹配应该返回错误");
    }

    /// 测试：向量逐元素乘法正常情况
    #[test]
    fn test_mul_normal() {
        let a = arr1(&[2.0, 3.0, 4.0]);
        let b = arr1(&[5.0, 6.0, 7.0]);

        let result = mul(&a.view(), &b.view()).expect("向量乘法应该成功");
        assert!((result[0] - 10.0).abs() < 1e-5);
        assert!((result[1] - 18.0).abs() < 1e-5);
        assert!((result[2] - 28.0).abs() < 1e-5);
    }

    /// 测试：向量乘法长度不匹配返回错误
    #[test]
    fn test_mul_length_mismatch() {
        let a = arr1(&[1.0, 2.0, 3.0]);
        let b = arr1(&[1.0, 2.0]);

        let result = mul(&a.view(), &b.view());
        assert!(result.is_err(), "长度不匹配应该返回错误");
    }

    /// 测试：LayerNorm归一化功能（核心功能，数值稳定性验证）
    #[test]
    fn test_layer_norm_normal() {
        let x = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let weight = arr1(&[1.0, 1.0, 1.0, 1.0]);
        let bias = arr1(&[0.0, 0.0, 0.0, 0.0]);

        let result =
            layer_norm(&x.view(), &weight.view(), &bias.view(), 1e-5).expect("LayerNorm应该成功");

        // 验证结果均值接近0
        let mean = result.mean().unwrap();
        assert!(mean.abs() < 1e-5, "LayerNorm后均值应为0，实际: {}", mean);

        // 验证方差接近1
        let var = result.var(0.0);
        assert!(
            (var - 1.0).abs() < 0.01,
            "LayerNorm后方差应接近1，实际: {}",
            var
        );
    }

    /// 测试：LayerNorm维度不匹配返回错误
    #[test]
    fn test_layer_norm_dimension_mismatch() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let weight = arr1(&[1.0, 2.0]); // 长度不同
        let bias = arr1(&[0.0, 0.0, 0.0]);

        let result = layer_norm(&x.view(), &weight.view(), &bias.view(), 1e-5);
        assert!(result.is_err(), "维度不匹配应该返回错误");
    }

    /// 测试：RMSNorm归一化功能
    #[test]
    fn test_rms_norm_normal() {
        let x = arr1(&[1.0, 2.0, 2.0, 0.0]);
        let weight = arr1(&[1.0, 1.0, 1.0, 1.0]);

        let result = rms_norm(&x.view(), &weight.view(), 1e-5).expect("RMSNorm应该成功");

        // 验证RMS值：sqrt((1+4+4+0)/4) = sqrt(9/4) = 1.5
        // 归一化后的RMS应该为1
        let sum_sq: f32 = result.iter().map(|&v| v * v).sum();
        let rms = (sum_sq / result.len() as f32).sqrt();
        assert!(
            (rms - 1.0).abs() < 0.01,
            "RMSNorm后RMS应接近1，实际: {}",
            rms
        );
    }

    /// 测试：RMSNorm维度不匹配返回错误
    #[test]
    fn test_rms_norm_dimension_mismatch() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let weight = arr1(&[1.0, 2.0]); // 长度不同

        let result = rms_norm(&x.view(), &weight.view(), 1e-5);
        assert!(result.is_err(), "维度不匹配应该返回错误");
    }

    /// 测试：SiLU激活函数（x * sigmoid(x)）
    #[test]
    fn test_silu() {
        let x = arr1(&[0.0, 1.0, -1.0, 10.0]);
        let result = silu(&x.view());

        // SiLU(0) = 0 * sigmoid(0) = 0
        assert!(result[0].abs() < 1e-5);

        // SiLU(x) 对于正数应该是正的
        assert!(result[1] > 0.0 && result[1] < 1.0);

        // SiLU(-1) 应该是负的但大于-1
        assert!(result[2] > -1.0 && result[2] < 0.0);

        // SiLU(10) 应该接近10（因为sigmoid(10)≈1）
        assert!((result[3] - 10.0).abs() < 0.001);
    }

    /// 测试：RoPE位置编码（旋转位置嵌入）
    #[test]
    fn test_rotary_embedding_normal() {
        let mut x = Array2::zeros((2, 4)); // seq_len=2, head_dim=4
        x[[0, 0]] = 1.0;
        x[[0, 1]] = 0.0;
        x[[1, 0]] = 1.0;
        x[[1, 1]] = 0.0;

        let positions = vec![0, 1];
        let theta = 10000.0;

        let result = rotary_embedding(&mut x, &positions, theta);
        assert!(result.is_ok(), "RoPE应该成功执行");

        // 验证旋转后的值发生了变化（非零旋转）
        assert!(
            x[[0, 0]] != 1.0 || x[[0, 1]] != 0.0 || x[[1, 0]] != 1.0 || x[[1, 1]] != 0.0,
            "RoPE应该改变输入值"
        );
    }

    /// 测试：RoPE位置长度不匹配返回错误
    #[test]
    fn test_rotary_embedding_position_mismatch() {
        let mut x = Array2::zeros((3, 4)); // seq_len=3
        let positions = vec![0, 1]; // 只有2个位置

        let result = rotary_embedding(&mut x, &positions, 10000.0);
        assert!(result.is_err(), "位置长度不匹配应该返回错误");
    }

    /// 测试：softmax对全零向量的处理（数值稳定性边界条件）
    #[test]
    fn test_softmax_zeros() {
        let x = arr1(&[0.0, 0.0, 0.0]);
        let result = softmax(&x.view());

        // 全零向量softmax后应该是均匀分布
        for i in 0..3 {
            assert!((result[i] - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    /// 测试：softmax对极大值的数值稳定性（防止溢出）
    #[test]
    fn test_softmax_large_values() {
        let x = arr1(&[1000.0, 2000.0, 3000.0]);
        let result = softmax(&x.view());

        // 验证概率和为1
        let sum: f32 = result.sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // 对于极大值，最大值对应的概率应该趋近于1.0（其他值趋近于0）
        // 由于数值稳定性处理（减去max），不会溢出
        assert!(result[2] > 0.99, "最大值概率应接近1.0，实际: {}", result[2]);
        assert!(result[0] < 0.01, "最小值概率应接近0.0，实际: {}", result[0]);
    }
}
