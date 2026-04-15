//! FP8 量化模块
//! 支持 E4M3 和 E5M2 两种 FP8 格式

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fp8Format {
    E4M3,
    E5M2,
}

pub struct Fp8Quantizer {
    format: Fp8Format,
}

impl Fp8Quantizer {
    pub fn new(format: Fp8Format) -> Self {
        Self { format }
    }

    pub fn quantize(&self, data: &[f32]) -> Vec<u8> {
        match self.format {
            Fp8Format::E4M3 => self.quantize_e4m3(data),
            Fp8Format::E5M2 => self.quantize_e5m2(data),
        }
    }

    pub fn dequantize(&self, data: &[u8]) -> Vec<f32> {
        match self.format {
            Fp8Format::E4M3 => self.dequantize_e4m3(data),
            Fp8Format::E5M2 => self.dequantize_e5m2(data),
        }
    }

    fn quantize_e4m3(&self, data: &[f32]) -> Vec<u8> {
        data.iter()
            .map(|&x| {
                if x == 0.0 {
                    return 0x00;
                }

                let sign = if x.is_sign_negative() { 0x80 } else { 0x00 };
                let abs_x = x.abs();

                let (exponent, mantissa) = if abs_x >= 448.0 {
                    (0x0F, 0x07)
                } else if abs_x < f32::MIN_POSITIVE {
                    (0x00, 0x00)
                } else {
                    let log2 = abs_x.log2();
                    let exp = (log2.floor() as i32 + 7).clamp(0, 15);
                    let scale = 2.0f32.powi(exp - 7);
                    let man = ((abs_x / scale) * 8.0).floor() as u8;
                    let man_clamped = man.min(7);
                    (exp as u8, man_clamped)
                };

                sign | (exponent << 3) | mantissa
            })
            .collect()
    }

    fn quantize_e5m2(&self, data: &[f32]) -> Vec<u8> {
        data.iter()
            .map(|&x| {
                if x == 0.0 {
                    return 0x00;
                }

                let sign = if x.is_sign_negative() { 0x80 } else { 0x00 };
                let abs_x = x.abs();

                let (exponent, mantissa) = if abs_x >= 57344.0 {
                    (0x1F, 0x03)
                } else if abs_x < f32::MIN_POSITIVE {
                    (0x00, 0x00)
                } else {
                    let log2 = abs_x.log2();
                    let exp = (log2.floor() as i32 + 15).clamp(0, 31);
                    let scale = 2.0f32.powi(exp - 15);
                    let man = ((abs_x / scale) * 4.0).floor() as u8;
                    let man_clamped = man.min(3);
                    (exp as u8, man_clamped)
                };

                sign | (exponent << 2) | mantissa
            })
            .collect()
    }

    fn dequantize_e4m3(&self, data: &[u8]) -> Vec<f32> {
        data.iter()
            .map(|&byte| {
                let sign_f: f32 = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
                let exponent = (byte >> 3) & 0x0F;
                let mantissa = byte & 0x07;

                if exponent == 0 && mantissa == 0 {
                    return 0.0;
                }

                let exp_val = exponent as i32 - 7;
                let man_val = if exponent == 0 {
                    mantissa as f32 / 8.0
                } else {
                    1.0 + (mantissa as f32) / 8.0
                };

                sign_f * man_val * 2.0f32.powi(exp_val)
            })
            .collect()
    }

    fn dequantize_e5m2(&self, data: &[u8]) -> Vec<f32> {
        data.iter()
            .map(|&byte| {
                let sign_f: f32 = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
                let exponent = (byte >> 2) & 0x1F;
                let mantissa = byte & 0x03;

                if exponent == 0 && mantissa == 0 {
                    return 0.0;
                }

                let exp_val = exponent as i32 - 15;
                let man_val = if exponent == 0 {
                    mantissa as f32 / 4.0
                } else {
                    1.0 + (mantissa as f32) / 4.0
                };

                sign_f * man_val * 2.0f32.powi(exp_val)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp8_e4m3_roundtrip() {
        let quantizer = Fp8Quantizer::new(Fp8Format::E4M3);
        let original = vec![0.0, 1.0, -1.0, 100.0, -100.0, 0.001, -0.001];
        let quantized = quantizer.quantize(&original);
        let dequantized = quantizer.dequantize(&quantized);

        for (orig, &deq) in original.iter().zip(dequantized.iter()) {
            let rel_error = if *orig != 0.0 {
                (orig - deq).abs() / orig.abs()
            } else {
                deq.abs()
            };
            assert!(
                rel_error < 0.15,
                "E4M3 roundtrip error too large: {} -> {}",
                orig,
                deq
            );
        }
    }

    #[test]
    fn test_fp8_e5m2_roundtrip() {
        let quantizer = Fp8Quantizer::new(Fp8Format::E5M2);
        let original = vec![0.0, 1.0, -1.0, 1000.0, -1000.0, 0.0001, -0.0001];
        let quantized = quantizer.quantize(&original);
        let dequantized = quantizer.dequantize(&quantized);

        for (orig, &deq) in original.iter().zip(dequantized.iter()) {
            let rel_error = if *orig != 0.0 {
                (orig - deq).abs() / orig.abs()
            } else {
                deq.abs()
            };
            assert!(
                rel_error < 0.25,
                "E5M2 roundtrip error too large: {} -> {}",
                orig,
                deq
            );
        }
    }

    #[test]
    fn test_zero_handling() {
        let q_e4m3 = Fp8Quantizer::new(Fp8Format::E4M3);
        let q_e5m2 = Fp8Quantizer::new(Fp8Format::E5M2);

        assert_eq!(q_e4m3.quantize(&[0.0])[0], 0x00);
        assert_eq!(q_e5m2.quantize(&[0.0])[0], 0x00);
        assert!((q_e4m3.dequantize(&[0x00])[0]).abs() < f32::EPSILON);
        assert!((q_e5m2.dequantize(&[0x00])[0]).abs() < f32::EPSILON);
    }
}
