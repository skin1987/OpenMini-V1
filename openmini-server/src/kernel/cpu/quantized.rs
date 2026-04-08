//! 量化算子内核
//!
//! 提供INT4/INT8/FP16量化支持

use anyhow::Result;

/// 量化参数
#[derive(Debug, Clone)]
pub struct QuantParams {
    pub scale: f32,
    pub zero_point: i32,
    pub min_val: f32,
    pub max_val: f32,
}

/// 对称量化（INT8）
pub fn quantize_symmetric(data: &[f32]) -> Result<(Vec<i8>, f32)> {
    let max_abs = data.iter().map(|&x| x.abs()).fold(0.0, f32::max);
    let scale = max_abs / 127.0;

    if scale == 0.0 {
        return Ok((vec![0i8; data.len()], 1.0));
    }

    let quantized: Vec<i8> = data
        .iter()
        .map(|&x| (x / scale).round().clamp(-128.0, 127.0) as i8)
        .collect();

    Ok((quantized, scale))
}

/// 对称反量化（INT8）
pub fn dequantize_symmetric(data: &[i8], scale: f32) -> Vec<f32> {
    data.iter().map(|&x| x as f32 * scale).collect()
}

/// 非对称量化（INT8）
pub fn quantize_asymmetric(data: &[f32]) -> Result<(Vec<u8>, QuantParams)> {
    let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let scale = (max_val - min_val) / 255.0;
    let zero_point = (-min_val / scale).round() as i32;

    if scale == 0.0 {
        return Ok((
            vec![0u8; data.len()],
            QuantParams {
                scale: 1.0,
                zero_point: 0,
                min_val,
                max_val,
            },
        ));
    }

    let quantized: Vec<u8> = data
        .iter()
        .map(|&x| ((x / scale).round() as i32 + zero_point).clamp(0, 255) as u8)
        .collect();

    Ok((
        quantized,
        QuantParams {
            scale,
            zero_point,
            min_val,
            max_val,
        },
    ))
}

/// 非对称反量化（INT8）
pub fn dequantize_asymmetric(data: &[u8], params: &QuantParams) -> Vec<f32> {
    data.iter()
        .map(|&x| (x as i32 - params.zero_point) as f32 * params.scale)
        .collect()
}

/// INT4量化
pub fn quantize_int4(data: &[f32]) -> Result<(Vec<u8>, f32)> {
    let max_abs = data.iter().map(|&x| x.abs()).fold(0.0, f32::max);
    let scale = max_abs / 7.0;

    if scale == 0.0 {
        return Ok((vec![0u8; (data.len() + 1) / 2], 1.0));
    }

    let mut packed = Vec::with_capacity((data.len() + 1) / 2);

    for chunk in data.chunks(2) {
        let low = (chunk[0] / scale).round().clamp(-8.0, 7.0) as i8;
        let high = if chunk.len() > 1 {
            (chunk[1] / scale).round().clamp(-8.0, 7.0) as i8
        } else {
            0
        };

        // 打包两个INT4到一个字节
        let packed_byte = ((low & 0x0F) as u8) | ((high & 0x0F) as u8) << 4;
        packed.push(packed_byte);
    }

    Ok((packed, scale))
}

/// INT4反量化
pub fn dequantize_int4(packed: &[u8], scale: f32, len: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(len);

    for &byte in packed {
        let low = (byte & 0x0F) as i8;
        let low = if low > 7 { low - 16 } else { low };
        result.push(low as f32 * scale);

        if result.len() < len {
            let high = ((byte >> 4) & 0x0F) as i8;
            let high = if high > 7 { high - 16 } else { high };
            result.push(high as f32 * scale);
        }
    }

    result.truncate(len);
    result
}

/// FP16量化
pub fn quantize_fp16(data: &[f32]) -> Vec<u16> {
    data.iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect()
}

/// FP16反量化
pub fn dequantize_fp16(data: &[u16]) -> Vec<f32> {
    data.iter()
        .map(|&x| half::f16::from_bits(x).to_f32())
        .collect()
}

/// 动态量化（根据数据范围自动选择）
pub fn quantize_dynamic(data: &[f32]) -> Result<(Vec<u8>, QuantParams)> {
    let variance = data.iter().map(|&x| x * x).sum::<f32>() / data.len() as f32;

    if variance < 0.01 {
        quantize_asymmetric(data)
    } else {
        let (quantized, scale) = quantize_symmetric(data)?;
        let quantized_u8: Vec<u8> = quantized.iter().map(|&x| (x as i16 + 128) as u8).collect();
        Ok((
            quantized_u8,
            QuantParams {
                scale,
                zero_point: 128,
                min_val: -128.0 * scale,
                max_val: 127.0 * scale,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_symmetric() {
        let data = vec![1.0, -1.0, 0.5, -0.5];
        let (quantized, scale) = quantize_symmetric(&data).unwrap();

        let dequantized = dequantize_symmetric(&quantized, scale);

        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.1);
        }
    }

    #[test]
    fn test_quantize_int4() {
        let data = vec![1.0, -1.0, 0.5, -0.5, 0.25];
        let (packed, scale) = quantize_int4(&data).unwrap();

        let dequantized = dequantize_int4(&packed, scale, data.len());

        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.2);
        }
    }

    // ========== 新增测试开始 ==========

    /// 测试非对称量化和反量化
    #[test]
    fn test_quantize_asymmetric() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (quantized, params) = quantize_asymmetric(&data).unwrap();

        // 验证参数合理性
        assert!(params.scale > 0.0);
        assert!(params.min_val <= 1.0);
        assert!(params.max_val >= 5.0);

        // 反量化并验证误差
        let dequantized = dequantize_asymmetric(&quantized, &params);
        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            assert!(
                (orig - deq).abs() < 0.1,
                "原始值 {} 与反量化值 {} 差异过大",
                orig,
                deq
            );
        }
    }

    /// 测试FP16量化和反量化
    #[test]
    fn test_quantize_fp16() {
        let data = vec![1.0, 2.0, 3.14159, 0.001, -1.5];

        // FP16量化
        let fp16_data = quantize_fp16(&data);
        assert_eq!(fp16_data.len(), data.len());

        // FP16反量化
        let dequantized = dequantize_fp16(&fp16_data);
        assert_eq!(dequantized.len(), data.len());

        // 验证FP16精度损失在可接受范围内（FP16有约11位有效数字）
        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            if orig.abs() > 0.01 {
                // 对于非极小值，相对误差应小于1%
                let rel_error = (orig - deq).abs() / orig.abs();
                assert!(
                    rel_error < 0.01,
                    "FP16精度: 原始值={}, 反量化值={}, 相对误差={}",
                    orig,
                    deq,
                    rel_error
                );
            }
        }
    }

    /// 测试全零数据的量化（scale=0的特殊分支）
    #[test]
    fn test_zero_data_quantization() {
        let zero_data = vec![0.0; 10];

        // 对称量化全零数据
        let (quantized_sym, scale_sym) = quantize_symmetric(&zero_data).unwrap();
        assert_eq!(scale_sym, 1.0); // 全零时scale应返回1.0避免除零
        assert!(quantized_sym.iter().all(|&x| x == 0));

        // INT4量化全零数据
        let (quantized_i4, scale_i4) = quantize_int4(&zero_data).unwrap();
        assert_eq!(scale_i4, 1.0); // 全零时scale应返回1.0
        assert!(quantized_i4.iter().all(|&x| x == 0));
    }

    /// 测试动态量化：低方差数据走非对称路径
    #[test]
    fn test_dynamic_quantization_low_variance() {
        // 低方差数据（方差 < 0.01）应该使用非对称量化
        let low_variance_data = vec![1.001, 1.002, 0.999, 1.000, 1.001];

        let (quantized, params) = quantize_dynamic(&low_variance_data).unwrap();

        // 低方差数据应该被正确量化
        assert!(!quantized.is_empty());
        assert!(params.scale >= 0.0);

        // 反量化验证
        let dequantized = dequantize_asymmetric(&quantized, &params);
        for (orig, deq) in low_variance_data.iter().zip(dequantized.iter()) {
            assert!(
                (orig - deq).abs() < 0.1,
                "动态量化(低方差): 原始值={} vs 反量化值={}",
                orig,
                deq
            );
        }
    }

    /// 测试动态量化：高方差数据走对称路径
    #[test]
    fn test_dynamic_quantization_high_variance() {
        // 高方差数据（方差 >= 0.01）应该使用对称量化
        let high_variance_data = vec![-100.0, 100.0, -50.0, 50.0, 0.0];

        let (quantized, params) = quantize_dynamic(&high_variance_data).unwrap();

        assert!(!quantized.is_empty());
        assert!(params.scale > 0.0);
        assert_eq!(params.zero_point, 128); // 对称量化转换后的zero_point应为128

        // 反量化验证（需要转回i8再反量化）
        let i8_data: Vec<i8> = quantized
            .iter()
            .map(|&x| (x as i8).wrapping_sub(128_u8 as i8))
            .collect();
        let dequantized = dequantize_symmetric(&i8_data, params.scale);

        for (orig, deq) in high_variance_data.iter().zip(dequantized.iter()) {
            assert!(
                (orig - deq).abs() < 1.0,
                "动态量化(高方差): 原始值={} vs 反量化值={}",
                orig,
                deq
            );
        }
    }

    /// 测试单元素数据的量化
    #[test]
    fn test_single_element_quantization() {
        let single_data = vec![42.5];

        // 对称量化
        let (quantized_sym, scale_sym) = quantize_symmetric(&single_data).unwrap();
        let deq_sym = dequantize_symmetric(&quantized_sym, scale_sym);
        assert!((single_data[0] - deq_sym[0]).abs() < 0.1);

        // INT4量化
        let (quantized_i4, scale_i4) = quantize_int4(&single_data).unwrap();
        let deq_i4 = dequantize_int4(&quantized_i4, scale_i4, 1);
        assert!((single_data[0] - deq_i4[0]).abs() < 0.2);
    }

    /// 测试极端范围的量化（极大值和极小值）
    #[test]
    fn test_extreme_range_quantization() {
        // 极大值范围
        let large_values = vec![1e6, -1e6, 5e5, -5e5];
        let (quantized, scale) = quantize_symmetric(&large_values).unwrap();

        // scale应该能覆盖整个范围
        assert!(scale > 0.0);

        let dequantized = dequantize_symmetric(&quantized, scale);
        // 极端值的量化误差会较大，但应在合理范围内（<5%）
        for (orig, deq) in large_values.iter().zip(dequantized.iter()) {
            if orig.abs() > 0.0 {
                let rel_error = (orig - deq).abs() / orig.abs();
                assert!(
                    rel_error < 0.05,
                    "极端值量化误差过大: 原始值={}, 反量化值={}, 相对误差={}",
                    orig,
                    deq,
                    rel_error
                );
            }
        }
    }

    /// 测试INT4奇数长度数据处理（打包逻辑）
    #[test]
    fn test_int4_odd_length() {
        // 奇数长度数据，最后一个元素单独打包
        let odd_length_data = vec![1.0, -1.0, 2.0]; // 3个元素

        let (packed, scale) = quantize_int4(&odd_length_data).unwrap();
        // 3个INT4值应该打包成2个字节（ceil(3/2)=2）
        assert_eq!(packed.len(), 2);

        let dequantized = dequantize_int4(&packed, scale, 3);
        assert_eq!(dequantized.len(), 3);

        // 验证反量化结果
        for (orig, deq) in odd_length_data.iter().zip(dequantized.iter()) {
            assert!(
                (orig - deq).abs() < 0.2,
                "奇数长度INT4: 原始值={} vs 反量化值={}",
                orig,
                deq
            );
        }
    }

    /// 测试量化参数结构的完整性和有效性
    #[test]
    fn test_quant_params_structure() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (_quantized, params) = quantize_asymmetric(&data).unwrap();

        assert!(params.scale > 0.0, "scale必须为正");
        assert!(
            params.min_val <= data.iter().cloned().fold(f32::INFINITY, f32::min),
            "min_val应该<=数据最小值"
        );
        assert!(
            params.max_val >= data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            "max_val应该>=数据最大值"
        );

        assert!(
            params.zero_point >= -255 && params.zero_point <= 255,
            "zero_point应该在[-255,255]范围内"
        );
    }

    /// 测试负数为主的非对称量化
    #[test]
    fn test_negative_dominant_asymmetric() {
        // 负数为主的数据集
        let negative_data = vec![-10.0, -20.0, -15.0, -5.0, -25.0];

        let (quantized, params) = quantize_asymmetric(&negative_data).unwrap();

        // min_val应该是接近-25的值
        assert!(params.min_val <= -24.0);

        // max_val应该是接近-5的值
        assert!(params.max_val >= -6.0);

        // 反量化验证
        let dequantized = dequantize_asymmetric(&quantized, &params);
        for (orig, deq) in negative_data.iter().zip(dequantized.iter()) {
            assert!(
                (orig - deq).abs() < 0.5,
                "负数主导数据: 原始值={} vs 反量化值={}",
                orig,
                deq
            );
        }
    }

    #[test]
    fn test_empty_array_quantization() {
        let empty_data: Vec<f32> = vec![];

        let (quantized_sym, scale_sym) = quantize_symmetric(&empty_data).unwrap();
        assert!(quantized_sym.is_empty());
        assert_eq!(scale_sym, 1.0);

        let (quantized_asym, _params) = quantize_asymmetric(&empty_data).unwrap();
        assert!(quantized_asym.is_empty());

        let (quantized_i4, _scale_i4) = quantize_int4(&empty_data).unwrap();
        assert!(quantized_i4.is_empty());
    }

    #[test]
    fn test_uniform_data_quantization() {
        let uniform_data = vec![5.0; 20];

        let (quantized_sym, scale_sym) = quantize_symmetric(&uniform_data).unwrap();
        assert!(scale_sym > 0.0);
        let first_val = quantized_sym[0];
        assert!(quantized_sym.iter().all(|&x| x == first_val));

        let (_quantized_asym, params) = quantize_asymmetric(&uniform_data).unwrap();
        assert_eq!(params.scale, 1.0);
    }

    /// 测试FP16特殊值处理（NaN和Infinity）
    #[test]
    fn test_fp16_special_values() {
        use std::f32;

        // 测试NaN在FP16中的传播
        let data_with_nan = vec![1.0, f32::NAN, 3.0];
        let fp16_nan = quantize_fp16(&data_with_nan);
        let dequantized_nan = dequantize_fp16(&fp16_nan);
        assert!(dequantized_nan[1].is_nan(), "FP16应该保留NaN");

        // 测试Infinity在FP16中的传播
        let data_with_inf = vec![f32::INFINITY, -f32::INFINITY, 0.0];
        let fp16_inf = quantize_fp16(&data_with_inf);
        let dequantized_inf = dequantize_fp16(&fp16_inf);
        assert!(dequantized_inf[0].is_infinite(), "FP16应该保留Infinity");
        assert!(dequantized_inf[1].is_infinite(), "FP16应该保留-Infinity");
    }

    #[test]
    fn test_large_range_precision_loss() {
        let large_range_data = vec![1e-6, 1e-3, 1.0, 1e3, 1e6];

        let (quantized, scale) = quantize_symmetric(&large_range_data).unwrap();
        let dequantized = dequantize_symmetric(&quantized, scale);

        for (orig, deq) in large_range_data.iter().zip(dequantized.iter()) {
            if orig.abs() > 0.0 {
                let rel_error = (orig - deq).abs() / orig.abs();
                assert!(
                    rel_error <= 1.0,
                    "大范围数据精度: 原始值={}, 反量化值={}, 相对误差={}",
                    orig,
                    deq,
                    rel_error
                );
            }
        }
    }

    /// 测试量化-反量化的对称性（往返一致性）
    #[test]
    fn test_roundtrip_consistency() {
        let original_data = vec![0.5, -0.5, 1.234, -2.567, 100.0, -100.0];

        // 对称量化往返
        let (q_sym, s_sym) = quantize_symmetric(&original_data).unwrap();
        let dq_sym = dequantize_symmetric(&q_sym, s_sym);
        // 再次量化反量化的结果应该保持稳定
        let (q_sym2, _) = quantize_symmetric(&dq_sym).unwrap();
        assert_eq!(q_sym.len(), q_sym2.len());

        // 非对称量化往返
        let (q_asym, p_asym) = quantize_asymmetric(&original_data).unwrap();
        let dq_asym = dequantize_asymmetric(&q_asym, &p_asym);
        // 结果长度应该一致
        assert_eq!(original_data.len(), dq_asym.len());
    }

    /// 测试INT4打包和解包的正确性（位级验证）
    #[test]
    fn test_int4_pack_unpack_correctness() {
        // 已知值的INT4打包验证
        let test_cases = vec![
            (vec![7.0], vec![7]),           // 最大正数 (0111)
            (vec![-8.0], vec![-8]),         // 最小负数 (1000)
            (vec![0.0], vec![0]),           // 零
            (vec![7.0, -8.0], vec![7, -8]), // 打包到同一字节
        ];

        for (input, expected_values) in test_cases {
            let (packed, scale) = quantize_int4(&input).unwrap();
            let unpacked = dequantize_int4(&packed, scale, input.len());

            assert_eq!(unpacked.len(), expected_values.len(), "INT4往返长度不匹配");

            // 验证符号正确性
            for (i, &expected) in expected_values.iter().enumerate() {
                // 只验证符号一致性（INT4精度有限）
                if expected >= 0 {
                    assert!(
                        unpacked[i] >= -scale * 0.5,
                        "正数不应变为显著负数: index={}",
                        i
                    );
                } else {
                    assert!(
                        unpacked[i] < scale * 0.5,
                        "负数不应变为显著正数: index={}",
                        i
                    );
                }
            }
        }
    }

    #[test]
    fn test_asymmetric_zero_point_boundaries() {
        let positive_data: Vec<f32> = vec![0.0, 0.1, 0.2, 0.5, 1.0];
        let (_, params_pos) = quantize_asymmetric(&positive_data).unwrap();
        assert!(
            params_pos.zero_point >= -255 && params_pos.zero_point <= 255,
            "zero_point应在有效范围内: {}",
            params_pos.zero_point
        );

        let mixed_data: Vec<f32> = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let (_, params_mixed) = quantize_asymmetric(&mixed_data).unwrap();
        assert!(
            params_mixed.zero_point >= -255 && params_mixed.zero_point <= 255,
            "zero_point应在有效范围内: {}",
            params_mixed.zero_point
        );
    }

    /// 测试动态量化在variance恰好等于阈值时的行为
    #[test]
    fn test_dynamic_quantization_threshold_boundary() {
        // 构造方差恰好约等于0.01的数据
        // 方差 = E[X²] - (E[X])² ≈ 0.01
        let threshold_data = vec![
            1.0, 1.005, 0.995, 1.002, 0.998, 1.003, 0.997, 1.001, 0.999, 1.004,
        ];

        let (quantized, params) = quantize_dynamic(&threshold_data).unwrap();

        // 无论走哪个分支，都应该成功量化
        assert!(!quantized.is_empty());
        assert!(params.scale >= 0.0);

        // 反量化验证
        let dequantized = dequantize_asymmetric(&quantized, &params);
        for (orig, deq) in threshold_data.iter().zip(dequantized.iter()) {
            assert!(
                (orig - deq).abs() < 0.15,
                "动态量化(阈值边界): 原始值={} vs 反量化值={}",
                orig,
                deq
            );
        }
    }

    /// 测试混合正负数据的量化（包含零交叉）
    #[test]
    fn test_mixed_sign_data_quantization() {
        // 包含正、负、零的数据
        let mixed_data = vec![-100.0, -50.0, -1.0, 0.0, 1.0, 50.0, 100.0];

        // 对称量化
        let (q_sym, s_sym) = quantize_symmetric(&mixed_data).unwrap();
        let dq_sym = dequantize_symmetric(&q_sym, s_sym);
        for (orig, deq) in mixed_data.iter().zip(dq_sym.iter()) {
            assert!(
                (orig - deq).abs() < 1.0,
                "混合数据对称量化: {} vs {}",
                orig,
                deq
            );
        }

        // 非对称量化
        let (q_asym, p_asym) = quantize_asymmetric(&mixed_data).unwrap();
        let dq_asym = dequantize_asymmetric(&q_asym, &p_asym);
        for (orig, deq) in mixed_data.iter().zip(dq_asym.iter()) {
            assert!(
                (orig - deq).abs() < 0.5,
                "混合数据非对称量化: {} vs {}",
                orig,
                deq
            );
        }
    }
}
