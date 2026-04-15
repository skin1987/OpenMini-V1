//! AMP (Automatic Mixed Precision) - 自动混合精度训练
//!
//! 支持 FP16/BF16 前向传播 + FP32 参数更新，
//! 防止梯度下溢的 Loss Scaling 技术。

use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MixedPrecision {
    /// 不使用混合精度（默认）
    None,
    /// FP16 半精度浮点
    Fp16,
    /// BFloat16（Brain Float，保留指数范围）
    Bf16,
}

impl Default for MixedPrecision {
    fn default() -> Self {
        Self::None
    }
}

impl std::fmt::Display for MixedPrecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "fp32"),
            Self::Fp16 => write!(f, "fp16"),
            Self::Bf16 => write!(f, "bf16"),
        }
    }
}

/// AMP 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmpConfig {
    pub enabled: bool,
    pub dtype: MixedPrecision,

    /// Loss scaling 初始值
    #[serde(default = "default_initial_scale")]
    pub initial_scale: f64,

    /// Loss scaling 增长因子（每 step 无溢出时乘以此值）
    #[serde(default = "default_growth_factor")]
    pub growth_factor: f64,

    /// Loss scaling 回退因子（检测到溢出时除以此值）
    #[serde(default = "default_backoff_factor")]
    pub backoff_factor: f64,

    /// Loss scaling 最大值
    #[serde(default = "default_max_scale")]
    pub max_scale: f64,

    /// 连续无溢出步数后增长 loss scale
    #[serde(default = "default_growth_interval")]
    pub growth_interval: usize,
}

fn default_initial_scale() -> f64 {
    2.0_f64.powi(15)
} // 32768.0
fn default_growth_factor() -> f64 {
    2.0
}
fn default_backoff_factor() -> f64 {
    0.5
}
fn default_max_scale() -> f64 {
    2.0_f64.powi(24)
} // 16777216.0
fn default_growth_interval() -> usize {
    2000
}

impl Default for AmpConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            dtype: MixedPrecision::None,
            initial_scale: default_initial_scale(),
            growth_factor: default_growth_factor(),
            backoff_factor: default_backoff_factor(),
            max_scale: default_max_scale(),
            growth_interval: default_growth_interval(),
        }
    }
}

impl AmpConfig {
    pub fn fp16() -> Self {
        Self {
            enabled: true,
            dtype: MixedPrecision::Fp16,
            ..Default::default()
        }
    }

    pub fn bf16() -> Self {
        Self {
            enabled: true,
            dtype: MixedPrecision::Bf16,
            ..Default::default()
        }
    }
}

/// Loss Scaler - 动态调整 loss scale 防止下溢
pub struct LossScaler {
    current_scale: f64,
    config: AmpConfig,
    steps_since_last_overflow: usize,
    overflow_count: u64,
}

impl LossScaler {
    pub fn new(config: AmpConfig) -> Self {
        Self {
            current_scale: config.initial_scale,
            config,
            steps_since_last_overflow: 0,
            overflow_count: 0,
        }
    }

    /// 获取当前 scale
    pub fn scale(&self) -> f64 {
        self.current_scale
    }

    /// 缩放 loss
    pub fn scale_loss(&self, loss: f32) -> f32 {
        (loss as f64 * self.current_scale) as f32
    }

    /// 取消缩放梯度
    pub fn unscale_gradients(&self, gradients: &mut [ArrayD<f32>]) {
        let inv_scale = 1.0 / self.current_scale;
        for grad in gradients.iter_mut() {
            *grad = grad.mapv(|x| (x as f64 * inv_scale) as f32);
        }
    }

    /// 检查是否有溢出并更新 scale
    pub fn check_and_update(&mut self, has_overflow: bool) -> bool {
        if has_overflow {
            // 检测到溢出：降低 scale
            self.current_scale *= self.config.backoff_factor;
            self.current_scale = self.current_scale.max(1.0); // 最小为 1
            self.steps_since_last_overflow = 0;
            self.overflow_count += 1;

            // 返回 true 表示应该跳过本次更新
            true
        } else {
            // 无溢出：定期尝试增加 scale
            self.steps_since_last_overflow += 1;

            if self.steps_since_last_overflow >= self.config.growth_interval {
                let new_scale = self.current_scale * self.config.growth_factor;
                if new_scale <= self.config.max_scale {
                    self.current_scale = new_scale;
                }
                self.steps_since_last_overflow = 0;
            }

            false
        }
    }

    /// 重置状态
    pub fn reset(&mut self) {
        self.current_scale = self.config.initial_scale;
        self.steps_since_last_overflow = 0;
        self.overflow_count = 0;
    }

    /// 获取统计信息
    pub fn stats(&self) -> LossScalerStats {
        LossScalerStats {
            current_scale: self.current_scale,
            total_overflows: self.overflow_count,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossScalerStats {
    pub current_scale: f64,
    pub total_overflows: u64,
}

/// 类型转换工具函数
/// F32 → BF16（截断尾数，保留指数范围）
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 31) & 1;
    let exponent = (bits >> 23) & 0xFF;
    let mantissa = bits & 0x7FFFFF;

    // BF16: 1 bit sign, 8 bits exponent, 7 bits mantissa
    ((sign << 15) | (exponent << 7) | (mantissa >> 16)) as u16
}

/// BF16 → F32
pub fn bf16_to_f16(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exponent = ((bits >> 7) & 0xFF) as u32;
    let mantissa = (bits & 0x7F) as u32;

    // 扩展为 F32 的 23 位 mantissa
    let full_mantissa = mantissa << 16;

    f32::from_bits((sign << 31) | (exponent << 23) | full_mantissa)
}

/// F32 → FP16（标准 IEEE 754 半精度）
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 31) & 1;
    let mut exponent = (bits >> 23) & 0xFF;
    let mut mantissa = bits & 0x7FFFFF;

    // 处理特殊值
    if exponent == 255 {
        // Inf 或 NaN
        return ((sign << 15) | 0x7C00 | (mantissa >> 13)) as u16;
    }

    if exponent == 0 && mantissa == 0 {
        // ±0
        return (sign << 15) as u16;
    }

    // 舍入到 nearest even
    let round_bit = (mantissa >> 12) & 1;
    let sticky = (mantissa & 0xFFF) != 0;

    mantissa >>= 13; // 从 23 位降到 10 位

    if round_bit == 1 && (sticky || (mantissa & 1) == 1) {
        mantissa += 1;

        if mantissa >= 0x400 {
            // 进位到 exponent
            mantissa = 0;
            exponent += 1;

            if exponent >= 31 {
                // 溢出到 inf
                return ((sign << 15) | 0x7C00) as u16;
            }
        }
    }

    if exponent > 0 {
        exponent -= 112; // bias 调整 (127 → 15)
    } else {
        exponent = 0;
    }

    ((sign << 15) | (exponent << 10) | mantissa) as u16
}

/// FP16 → F32
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            // ±0
            f32::from_bits(sign << 31)
        } else {
            // 非规格化数
            f32::from_bits((sign << 31) | (exponent << 23) | (mantissa << 13))
        }
    } else if exponent == 31 {
        if mantissa == 0 {
            // ±inf
            f32::from_bits((sign << 31) | 0x7F800000)
        } else {
            // NaN
            f32::from_bits(0x7FC00000)
        }
    } else {
        // 规格化数
        f32::from_bits((sign << 31) | ((exponent + 112) << 23) | (mantissa << 13))
    }
}

/// 检查数组中是否有 inf/nan（表示溢出）
pub fn has_overflow(gradients: &[ArrayD<f32>]) -> bool {
    gradients.iter().any(|g| g.iter().any(|&v| !v.is_finite()))
}

// 单元测试
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amp_config_default() {
        let config = AmpConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.dtype, MixedPrecision::None);
    }

    #[test]
    fn test_amp_config_fp16() {
        let config = AmpConfig::fp16();
        assert!(config.enabled);
        assert_eq!(config.dtype, MixedPrecision::Fp16);
    }

    #[test]
    fn test_loss_scaler_basic() {
        let config = AmpConfig::fp16();
        let initial = config.initial_scale;
        let scaler = LossScaler::new(config);

        assert_eq!(scaler.scale(), initial);

        let scaled_loss = scaler.scale_loss(0.5);
        let expected = (0.5_f64 * initial) as f32;
        assert!((scaled_loss - expected).abs() < 1e-6);
    }

    #[test]
    fn test_loss_scaler_growth() {
        let config = AmpConfig {
            enabled: true,
            dtype: MixedPrecision::Fp16,
            growth_interval: 5,
            ..Default::default()
        };

        let mut scaler = LossScaler::new(config.clone());

        // 模拟 5 步无溢出
        for _ in 0..5 {
            let skip = scaler.check_and_update(false);
            assert!(!skip);
        }

        // scale 应该增长
        assert!(scaler.scale() > config.initial_scale);
    }

    #[test]
    fn test_loss_scaler_backoff_on_overflow() {
        let config = AmpConfig::default();
        let mut scaler = LossScaler::new(config);

        let original_scale = scaler.scale();

        // 检测到溢出
        let skip = scaler.check_and_update(true);
        assert!(skip); // 应该跳过更新

        // scale 应该降低
        assert!(scaler.scale() < original_scale);
    }

    #[test]
    fn test_fp16_conversion_roundtrip() {
        // 使用 FP16 可表示范围内的数值（最大约 65504）
        let original: Vec<f32> = vec![1.0, -1.0, 0.5, 3.14159, 100.0, 1000.0, 65000.0];

        for &val in &original {
            let bits = f32_to_f16(val);
            let recovered = f16_to_f32(bits);

            // 允许一定的精度损失（相对误差 < 1%）
            if val.is_finite() && recovered.is_finite() {
                let rel_error = ((val - recovered).abs() / val.abs().max(1e-30)).abs();
                assert!(
                    rel_error < 0.01,
                    "FP16 conversion error too large: {} -> {}",
                    val,
                    recovered
                );
            }
        }
    }

    #[test]
    fn test_bf16_preserves_large_values() {
        // BF16 能保持大数值的精度（范围与 F32 相同，只是精度较低）
        let test_values: Vec<f32> = vec![1000.0, 10000.0, 65000.0];

        for &val in &test_values {
            let bf16_bits = f32_to_bf16(val);
            let recovered = bf16_to_f16(bf16_bits);

            if val.is_finite() && recovered.is_finite() {
                let rel_error = (val - recovered).abs() / val.abs().max(1e-30);
                // BF16 精度约为 3-4 位有效数字，相对误差应 < 1%
                assert!(
                    rel_error < 0.01,
                    "BF16 should preserve large values well: {} -> {}",
                    val,
                    recovered
                );
            }
        }
    }

    #[test]
    fn test_has_overflow_detection() {
        let normal_grads = vec![ArrayD::from_shape_vec([3].as_ref(), vec![1.0, 2.0, 3.0]).unwrap()];
        assert!(!has_overflow(&normal_grads));

        let overflow_grads =
            vec![ArrayD::from_shape_vec([1].as_ref(), vec![f32::INFINITY]).unwrap()];
        assert!(has_overflow(&overflow_grads));

        let nan_grads = vec![ArrayD::from_shape_vec([1].as_ref(), vec![f32::NAN]).unwrap()];
        assert!(has_overflow(&nan_grads));
    }
}
