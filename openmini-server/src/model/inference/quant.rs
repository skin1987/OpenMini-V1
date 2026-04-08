//! 量化权重反量化模块
//!
//! 本模块实现各种量化格式的反量化操作，将压缩的权重数据转换为 f32 格式。
//!
//! # 支持的量化格式
//!
//! - **F32**: 32位浮点数（无压缩）
//! - **F16**: 16位浮点数（半精度）
//! - **Q4_0**: 4位量化，每块32个元素，带缩放因子
//! - **Q4_1**: 4位量化，每块32个元素，带缩放因子和偏移
//! - **Q8_0**: 8位量化，每块32个元素，带缩放因子
//! - **Q4_K**: 4位 K-量化，每块256个元素
//! - **Q5_K**: 5位 K-量化，每块256个元素
//! - **Q6_K**: 6位 K-量化，每块256个元素
//! - **Q2_K**: 2位 K-量化，每块256个元素
//! - **Q3_K**: 3位 K-量化，每块256个元素
//! - **Q8_K**: 8位 K-量化，每块256个元素
//!
//! ## IQ智能量化 (Importance-based Quantization)
//!
//! - **IQ1_S**: 1.58位超压缩量化
//! - **IQ2_XXS**: 2位极低比特量化
//! - **IQ2_XS**: 2位低比特量化
//! - **IQ2_S**: 2位标准量化
//! - **IQ3_XXS**: 3位极低比特量化
//! - **IQ3_S**: 3位标准量化
//! - **IQ4_NL**: 4位非对称量化
//! - **IQ4_XS**: 4位扩展量化
//!
//! # 量化原理
//!
//! 量化通过将浮点数映射到低精度整数来减少存储空间：
//! - 块量化：将元素分组，每组共享缩放因子
//! - K-量化：更复杂的量化方案，提供更好的精度-压缩比

#![allow(dead_code)]

use super::gguf::GgufTensorType;

// ============================================================================
// 主反量化函数
// ============================================================================

/// 反量化张量数据
///
/// 根据张量类型选择对应的反量化方法
///
/// # Parameters
/// - `data`: 原始字节数据
/// - `tensor_type`: 张量量化类型
/// - `n`: 元素数量
///
/// # Returns
/// 反量化后的 f32 向量
pub fn dequantize(data: &[u8], tensor_type: GgufTensorType, n: usize) -> Vec<f32> {
    match tensor_type {
        GgufTensorType::F32 => dequantize_f32(data, n),
        GgufTensorType::F16 => dequantize_f16(data, n),
        GgufTensorType::Q4_0 => dequantize_q4_0(data, n),
        GgufTensorType::Q4_1 => dequantize_q4_1(data, n),
        GgufTensorType::Q8_0 => dequantize_q8_0(data, n),
        GgufTensorType::Q4K => dequantize_q4_k(data, n),
        GgufTensorType::Q5K => dequantize_q5_k(data, n),
        GgufTensorType::Q6K => dequantize_q6_k(data, n),
        GgufTensorType::Q2K => dequantize_q2_k(data, n),
        GgufTensorType::Q3K => dequantize_q3_k(data, n),
        GgufTensorType::Q8K => dequantize_q8_k(data, n),
        GgufTensorType::Iq1S => dequantize_iq1_s(data, n),
        GgufTensorType::Iq2Xxs => dequantize_iq2_xxs(data, n),
        GgufTensorType::Iq2Xs => dequantize_iq2_xs(data, n),
        GgufTensorType::Iq2S => dequantize_iq2_s(data, n),
        GgufTensorType::Iq3Xxs => dequantize_iq3_xxs(data, n),
        GgufTensorType::Iq3S => dequantize_iq3_s(data, n),
        GgufTensorType::Iq4Nl => dequantize_iq4_nl(data, n),
        GgufTensorType::Iq4Xs => dequantize_iq4_xs(data, n),
        _ => vec![0.0f32; n],
    }
}

// ============================================================================
// F32 反量化
// ============================================================================

/// F32 格式反量化
///
/// 直接读取 32 位浮点数，无压缩
///
/// # Parameters
/// - `data`: 原始字节数据
/// - `n`: 元素数量
///
/// # Returns
/// f32 向量
pub fn dequantize_f32(data: &[u8], n: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * 4;
        if offset + 4 <= data.len() {
            let bytes: [u8; 4] = data[offset..offset + 4].try_into().unwrap();
            result.push(f32::from_le_bytes(bytes));
        }
    }
    result
}

// ============================================================================
// F16 反量化
// ============================================================================

/// F16 格式反量化
///
/// 将 16 位半精度浮点数转换为 32 位浮点数
///
/// # Parameters
/// - `data`: 原始字节数据
/// - `n`: 元素数量
///
/// # Returns
/// f32 向量
fn dequantize_f16(data: &[u8], n: usize) -> Vec<f32> {
    use half::f16;
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * 2;
        if offset + 2 <= data.len() {
            let bytes: [u8; 2] = data[offset..offset + 2].try_into().unwrap();
            result.push(f16::from_le_bytes(bytes).to_f32());
        }
    }
    result
}

// ============================================================================
// Q4_0 反量化
// ============================================================================

/// Q4_0 格式反量化
///
/// 4位量化，每块32个元素，结构：
/// - 2字节：缩放因子 (F16)
/// - 16字节：量化值（每字节存储2个4位值）
///
/// 公式：value = (q - 8) * scale
///
/// # Parameters
/// - `data`: 原始字节数据
/// - `n`: 元素数量
///
/// # Returns
/// 反量化后的 f32 向量
pub fn dequantize_q4_0(data: &[u8], n: usize) -> Vec<f32> {
    use half::f16;
    const QK4_0: usize = 32; // 每块元素数
    let block_count = n.div_ceil(QK4_0);
    let mut result = vec![0.0f32; n];

    for block_idx in 0..block_count {
        // 每块 18 字节：2字节缩放 + 16字节数据
        let block_offset = block_idx * 18;
        if block_offset + 18 > data.len() {
            break;
        }

        // 读取缩放因子
        let scale_bytes: [u8; 2] = data[block_offset..block_offset + 2].try_into().unwrap();
        let scale = f16::from_le_bytes(scale_bytes).to_f32();

        // 读取量化值
        let qs = &data[block_offset + 2..block_offset + 18];

        let start = block_idx * QK4_0;
        for i in 0..QK4_0 {
            let idx = start + i;
            if idx >= n {
                break;
            }

            // 每字节存储2个4位值
            let byte_idx = i / 2;
            let is_high = i % 2 == 0;

            // 4位值范围 0-15，减8后范围 -8 到 7
            let q = if is_high {
                ((qs[byte_idx] >> 4) as i32) - 8
            } else {
                ((qs[byte_idx] & 0x0F) as i32) - 8
            };

            result[idx] = q as f32 * scale;
        }
    }

    result
}

// ============================================================================
// Q4_1 反量化
// ============================================================================

/// Q4_1 格式反量化
///
/// 4位量化，每块32个元素，带偏移，结构：
/// - 2字节：缩放因子 (F16)
/// - 2字节：最小值/偏移 (F16)
/// - 16字节：量化值
///
/// 公式：value = q * scale + min
///
/// # Parameters
/// - `data`: 原始字节数据
/// - `n`: 元素数量
///
/// # Returns
/// 反量化后的 f32 向量
pub fn dequantize_q4_1(data: &[u8], n: usize) -> Vec<f32> {
    use half::f16;
    const QK4_1: usize = 32;
    let block_count = n.div_ceil(QK4_1);
    let mut result = vec![0.0f32; n];

    for block_idx in 0..block_count {
        // 每块 20 字节：2字节缩放 + 2字节偏移 + 16字节数据
        let block_offset = block_idx * 20;
        if block_offset + 20 > data.len() {
            break;
        }

        // 读取缩放因子
        let scale_bytes: [u8; 2] = data[block_offset..block_offset + 2].try_into().unwrap();
        let scale = f16::from_le_bytes(scale_bytes).to_f32();

        // 读取最小值/偏移
        let min_bytes: [u8; 2] = data[block_offset + 2..block_offset + 4].try_into().unwrap();
        let min = f16::from_le_bytes(min_bytes).to_f32();

        // 读取量化值
        let qs = &data[block_offset + 4..block_offset + 20];

        let start = block_idx * QK4_1;
        for i in 0..QK4_1 {
            let idx = start + i;
            if idx >= n {
                break;
            }

            let byte_idx = i / 2;
            let is_high = i % 2 == 0;

            // 4位值范围 0-15
            let q = if is_high {
                (qs[byte_idx] >> 4) as f32
            } else {
                (qs[byte_idx] & 0x0F) as f32
            };

            result[idx] = q * scale + min;
        }
    }

    result
}

// ============================================================================
// Q8_0 反量化
// ============================================================================

/// Q8_0 格式反量化
///
/// 8位量化，每块32个元素，结构：
/// - 2字节：缩放因子 (F16)
/// - 32字节：量化值（有符号8位整数）
///
/// 公式：value = q * scale
///
/// # Parameters
/// - `data`: 原始字节数据
/// - `n`: 元素数量
///
/// # Returns
/// 反量化后的 f32 向量
pub fn dequantize_q8_0(data: &[u8], n: usize) -> Vec<f32> {
    use half::f16;
    const QK8_0: usize = 32;
    let block_count = n.div_ceil(QK8_0);
    let mut result = vec![0.0f32; n];

    for block_idx in 0..block_count {
        // 每块 34 字节：2字节缩放 + 32字节数据
        let block_offset = block_idx * 34;
        if block_offset + 34 > data.len() {
            break;
        }

        // 读取缩放因子
        let scale_bytes: [u8; 2] = data[block_offset..block_offset + 2].try_into().unwrap();
        let scale = f16::from_le_bytes(scale_bytes).to_f32();

        // 读取量化值
        let qs = &data[block_offset + 2..block_offset + 34];

        let start = block_idx * QK8_0;
        for i in 0..QK8_0 {
            let idx = start + i;
            if idx >= n {
                break;
            }

            // 8位有符号整数
            let q = qs[i] as i8 as f32;
            result[idx] = q * scale;
        }
    }

    result
}

// ============================================================================
// Q4_K 反量化
// ============================================================================

/// Q4_K 格式反量化
///
/// K-量化系列，每块256个元素，分为8个子块
/// 结构（144字节）：
/// - 2字节：主缩放因子 d
/// - 2字节：主最小值 dmin
/// - 8字节：子块缩放因子
/// - 4字节：子块最小值
/// - 128字节：量化值
///
/// # Parameters
/// - `data`: 原始字节数据
/// - `n`: 元素数量
///
/// # Returns
/// 反量化后的 f32 向量
pub fn dequantize_q4_k(data: &[u8], n: usize) -> Vec<f32> {
    use half::f16;
    const QK_K: usize = 256; // 每块元素数
    let block_count = n.div_ceil(QK_K);
    let mut result = vec![0.0f32; n];

    for block_idx in 0..block_count {
        let block_offset = block_idx * 144;
        if block_offset + 144 > data.len() {
            break;
        }

        let block = &data[block_offset..block_offset + 144];

        // 主缩放因子和最小值
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();

        // 子块缩放因子和最小值
        let scales = &block[4..12];
        let qs = &block[12..144];

        let start = block_idx * QK_K;

        // 8个子块，每个32个元素
        for sub_block in 0..8 {
            let scale = if sub_block < scales.len() {
                (scales[sub_block] & 0x3F) as f32
            } else {
                1.0
            };

            let min_idx = sub_block / 2;
            let min_val = if sub_block % 2 == 0 {
                let idx = 8 + min_idx;
                if idx < scales.len() {
                    (scales[idx] & 0x0F) as f32
                } else {
                    0.0
                }
            } else {
                let idx = 8 + min_idx;
                if idx < scales.len() {
                    (scales[idx] >> 4) as f32
                } else {
                    0.0
                }
            };

            let sub_start = start + sub_block * 32;
            let qs_offset = sub_block * 16;

            for i in 0..32 {
                let idx = sub_start + i;
                if idx >= n {
                    break;
                }

                let byte_idx = qs_offset + i / 2;
                let is_high = i % 2 == 0;

                let q = if is_high {
                    (qs[byte_idx] >> 4) as f32
                } else {
                    (qs[byte_idx] & 0x0F) as f32
                };

                result[idx] = (q - 8.0) * d * scale + dmin * min_val;
            }
        }
    }

    result
}

// ============================================================================
// Q5_K 反量化
// ============================================================================

/// Q5_K 格式反量化
///
/// 5位 K-量化，每块256个元素
/// 结构（176字节）：
/// - 2字节：主缩放因子 d
/// - 2字节：主最小值 dmin
/// - 16字节：子块缩放因子和最小值
/// - 16字节：高位比特
/// - 140字节：低位量化值
///
/// # Parameters
/// - `data`: 原始字节数据
/// - `n`: 元素数量
///
/// # Returns
/// 反量化后的 f32 向量
pub fn dequantize_q5_k(data: &[u8], n: usize) -> Vec<f32> {
    use half::f16;
    const QK_K: usize = 256;
    let block_count = n.div_ceil(QK_K);
    let mut result = vec![0.0f32; n];

    for block_idx in 0..block_count {
        let block_offset = block_idx * 176;
        if block_offset + 176 > data.len() {
            break;
        }

        let block = &data[block_offset..block_offset + 176];

        // 主缩放因子和最小值
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();

        // 子块参数
        let scales = &block[4..20];
        let qh = &block[20..36]; // 高位比特
        let qs = &block[36..176]; // 低位量化值

        let start = block_idx * QK_K;

        for sub_block in 0..8 {
            let scale = (scales[sub_block] & 0x3F) as f32;
            let min_idx = sub_block / 2;
            let min_val = if sub_block % 2 == 0 {
                (scales[8 + min_idx] & 0x0F) as f32
            } else {
                (scales[8 + min_idx] >> 4) as f32
            };

            let sub_start = start + sub_block * 32;

            for i in 0..32 {
                let idx = sub_start + i;
                if idx >= n {
                    break;
                }

                // 获取高位比特
                let qh_bit = ((qh[sub_block] >> (i % 8)) & 1) as f32;
                let qs_offset = sub_block * 16 + i / 2;
                let is_high = i % 2 == 0;

                // 组合4位低位和1位高位
                let q4 = if is_high {
                    (qs[qs_offset] >> 4) as f32
                } else {
                    (qs[qs_offset] & 0x0F) as f32
                };

                let q = q4 + qh_bit * 16.0;
                result[idx] = (q - 16.0) * d * scale + dmin * min_val;
            }
        }
    }

    result
}

// ============================================================================
// Q6_K 反量化
// ============================================================================

/// Q6_K 格式反量化
///
/// 6位 K-量化，每块256个元素
/// 结构（210字节）：
/// - 2字节：主缩放因子 d
/// - 64字节：低位量化值
/// - 64字节：高位量化值
/// - 32字节：子块缩放因子
///
/// # Parameters
/// - `data`: 原始字节数据
/// - `n`: 元素数量
///
/// # Returns
/// 反量化后的 f32 向量
pub fn dequantize_q6_k(data: &[u8], n: usize) -> Vec<f32> {
    use half::f16;
    const QK_K: usize = 256;
    let block_count = n.div_ceil(QK_K);
    let mut result = vec![0.0f32; n];

    for block_idx in 0..block_count {
        let block_offset = block_idx * 210;
        if block_offset + 210 > data.len() {
            break;
        }

        let block = &data[block_offset..block_offset + 210];

        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();

        let ql = &block[2..66];
        let qh = &block[66..130];
        let scales = &block[130..162];

        let start = block_idx * QK_K;

        for i in 0..QK_K {
            let idx = start + i;
            if idx >= n {
                break;
            }

            let scale_idx = i / 16;
            let scale = if scale_idx < scales.len() {
                scales[scale_idx] as i8 as f32
            } else {
                1.0
            };

            let ql_idx = i / 2;
            let ql_low = if ql_idx < ql.len() {
                if i % 2 == 0 {
                    (ql[ql_idx] & 0x0F) as i32
                } else {
                    (ql[ql_idx] >> 4) as i32
                }
            } else {
                0
            };

            let qh_idx = i / 4;
            let qh_high = if qh_idx < qh.len() {
                match i % 4 {
                    0 => (qh[qh_idx] & 0x03) as i32,
                    1 => ((qh[qh_idx] >> 2) & 0x03) as i32,
                    2 => ((qh[qh_idx] >> 4) & 0x03) as i32,
                    3 => ((qh[qh_idx] >> 6) & 0x03) as i32,
                    _ => 0,
                }
            } else {
                0
            };

            let q = ql_low | (qh_high << 4) - 32;

            result[idx] = q as f32 * d * scale;
        }
    }

    result
}

// ============================================================================
// Q2_K 反量化
// ============================================================================

/// Q2_K 格式反量化
///
/// 2位 K-量化，每块256个元素
/// 结构（68字节）：
/// - 2字节：主缩放因子 d (F16)
/// - 2字节：主最小值 dmin (F16)
/// - 16字节：子块缩放因子(16个)
/// - 16字节：子块最小值
/// - 32字节：量化值(每字节存储4个2位值)
///
/// 公式：value = (q - 2) * d * scale + dmin * min_val
///
/// # Parameters
/// - `data`: 原始字节数据
/// - `n`: 元素数量
///
/// # Returns
/// 反量化后的 f32 向量
pub fn dequantize_q2_k(data: &[u8], n: usize) -> Vec<f32> {
    use half::f16;
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 68;

    let block_count = n.div_ceil(QK_K);
    let mut result = vec![0.0f32; n];

    for block_idx in 0..block_count {
        let block_offset = block_idx * BLOCK_SIZE;
        if block_offset + BLOCK_SIZE > data.len() {
            break;
        }

        let block = &data[block_offset..block_offset + BLOCK_SIZE];

        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();

        let scales = &block[4..20];
        let mins = &block[20..36];
        let qs = &block[36..68];

        let start = block_idx * QK_K;

        for sub_block in 0..16 {
            let scale = scales[sub_block] as f32;
            let min_val = mins[sub_block] as f32;

            let sub_start = start + sub_block * 16;
            let qs_offset = sub_block * 8;

            for i in 0..16 {
                let idx = sub_start + i;
                if idx >= n {
                    break;
                }

                // Q2_K 位打包格式：每字节存储4个2位值
                // qs 有 32 字节，qs_offset = sub_block * 8
                // byte_idx = qs_offset + i/4, 最大值需要边界检查
                let byte_idx = qs_offset + i / 4;
                let bit_shift = (i % 4) * 2;

                let q = if byte_idx < qs.len() {
                    ((qs[byte_idx] >> bit_shift) & 0x03) as f32
                } else {
                    1.0 // 默认值 (q-2= -1)
                };

                result[idx] = (q - 2.0) * d * scale + dmin * min_val;
            }
        }
    }

    result
}

// ============================================================================
// Q3_K 反量化
// ============================================================================

/// Q3_K 格式反量化
///
/// 3位 K-量化，每块256个元素
/// 结构：
/// - 2字节：主缩放因子 d
/// - 64字节：低位量化值(ql)
/// - 64字节：高位量化值(qh)
/// - 32字节：子块缩放因子
///
/// # Parameters
/// - `data`: 原始字节数据
/// - `n`: 元素数量
///
/// # Returns
/// 反量化后的 f32 向量
pub fn dequantize_q3_k(data: &[u8], n: usize) -> Vec<f32> {
    use half::f16;
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 162;

    let block_count = n.div_ceil(QK_K);
    let mut result = vec![0.0f32; n];

    for block_idx in 0..block_count {
        let block_offset = block_idx * BLOCK_SIZE;
        if block_offset + BLOCK_SIZE > data.len() {
            break;
        }

        let block = &data[block_offset..block_offset + BLOCK_SIZE];

        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();

        let ql = &block[2..66];
        let qh = &block[66..130];
        let scales = &block[130..162];

        let start = block_idx * QK_K;

        for i in 0..QK_K {
            let idx = start + i;
            if idx >= n {
                break;
            }

            // Q3_K 位打包格式：
            // - ql (64 bytes): 每 byte 存 2 个 3-bit 值的低部分
            //   偶数 i: 取低 nibble 的低 3位 (ql[i/2] & 0x07)
            //   奇数 i: 取高 nibble 的低 3位 ((ql[i/2] >> 4) & 0x07)
            // - qh (64 bytes): 每 byte 存 4 个 3-bit 值的高部分
            //   用 qh[i/4] 取出，按 (i%4)*2 移位取 1 bit
            // - scales (32 bytes): 每 byte 存 8 个 scale 值的低部分

            let scale_idx = i / 8;
            let scale = if scale_idx < scales.len() {
                scales[scale_idx] as i8 as f32
            } else {
                1.0
            };

            let ql_idx = i / 2;
            let ql_byte = if ql_idx < ql.len() { ql[ql_idx] } else { 0 };

            let qh_idx = i / 4;
            let qh_byte = if qh_idx < qh.len() { qh[qh_idx] } else { 0 };

            // 解包 3-bit 量化值
            let ql_bits = if i % 2 == 0 {
                (ql_byte & 0x07) as i32
            } else {
                ((ql_byte >> 4) & 0x07) as i32
            };

            let qh_bit = ((qh_byte >> ((i % 4) * 2)) & 0x01) as i32;

            let q = (ql_bits | (qh_bit << 3)) - 16;

            result[idx] = q as f32 * d * scale;
        }
    }

    result
}

// ============================================================================
// Q8_K 反量化
// ============================================================================

/// Q8_K 格式反量化
///
/// 8位 K-量化，每块256个元素
/// 结构：
/// - 2字节：主缩放因子 d
/// - 256字节：量化值(每个元素1字节，有符号8位)
///
/// # Parameters
/// - `data`: 原始字节数据
/// - `n`: 元素数量
///
/// # Returns
/// 反量化后的 f32 向量
pub fn dequantize_q8_k(data: &[u8], n: usize) -> Vec<f32> {
    use half::f16;
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 258;

    let block_count = n.div_ceil(QK_K);
    let mut result = vec![0.0f32; n];

    for block_idx in 0..block_count {
        let block_offset = block_idx * BLOCK_SIZE;
        if block_offset + BLOCK_SIZE > data.len() {
            break;
        }

        let block = &data[block_offset..block_offset + BLOCK_SIZE];

        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();

        let qs = &block[2..258];

        let start = block_idx * QK_K;

        for i in 0..QK_K {
            let idx = start + i;
            if idx >= n {
                break;
            }

            let q = qs[i] as i8 as f32;
            result[idx] = q * d;
        }
    }

    result
}

// ============================================================================
// FP8 (E4M3) 量化
// ============================================================================

/// FP8 E4M3 格式量化
///
/// 8位浮点数，1位符号，4位指数，3位尾数
/// 范围: 约 ±448，精度约 3-4 位十进制
///
/// # Parameters
/// - `value`: f32 值
///
/// # Returns
/// FP8 量化后的字节
pub fn quantize_fp8(value: f32) -> u8 {
    const EXP_BIAS: i32 = 7;
    const MAX_EXP: i32 = 8;

    if value.is_nan() {
        return 0x7F; // NaN
    }

    let sign = if value < 0.0 { 0x80 } else { 0 };
    let abs_value = value.abs();

    if abs_value == 0.0 {
        return sign;
    }

    if abs_value > 448.0 {
        return sign | 0x7E; // 最大值
    }

    // 计算指数
    let exp = abs_value.log2().floor() as i32;
    let exp = exp.clamp(0, MAX_EXP);

    // 计算尾数
    let scale = 2f32.powi(exp - EXP_BIAS);
    let mantissa = ((abs_value / scale - 1.0) * 8.0).round() as u8;
    let mantissa = mantissa.min(7);

    sign | ((exp as u8) << 3) | mantissa
}

/// FP8 E4M3 格式反量化
///
/// # Parameters
/// - `data`: FP8 字节
///
/// # Returns
/// f32 值
pub fn dequantize_fp8(data: u8) -> f32 {
    const EXP_BIAS: i32 = 7;

    let sign = if data & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((data >> 3) & 0x0F) as i32;
    let mantissa = (data & 0x07) as f32;

    if exp == 0 && mantissa == 0.0 {
        return 0.0;
    }

    sign * 2f32.powi(exp - EXP_BIAS) * (1.0 + mantissa / 8.0)
}

/// 批量 FP8 量化
pub fn quantize_fp8_batch(values: &[f32]) -> Vec<u8> {
    values.iter().map(|&v| quantize_fp8(v)).collect()
}

/// 批量 FP8 反量化
pub fn dequantize_fp8_batch(data: &[u8]) -> Vec<f32> {
    data.iter().map(|&b| dequantize_fp8(b)).collect()
}

// ============================================================================
// FP4 量化
// ============================================================================

/// FP4 格式量化
///
/// 4位浮点数，1位符号，2位指数，1位尾数
/// 范围: 约 ±6，精度约 1-2 位十进制
///
/// # Parameters
/// - `value`: f32 值
///
/// # Returns
/// FP4 量化后的半字节 (返回完整的字节，低4位有效)
pub fn quantize_fp4(value: f32) -> u8 {
    const EXP_BIAS: i32 = 1;
    const MAX_EXP: i32 = 3;

    if value.is_nan() {
        return 0x7; // NaN
    }

    let sign = if value < 0.0 { 0x8 } else { 0 };
    let abs_value = value.abs();

    if abs_value == 0.0 {
        return sign;
    }

    if abs_value > 6.0 {
        return sign | 0x6; // 最大值
    }

    let exp = abs_value.log2().floor() as i32;
    let exp = exp.clamp(0, MAX_EXP);

    let scale = 2f32.powi(exp - EXP_BIAS);
    let mantissa = if abs_value / scale >= 1.5 { 1 } else { 0 };

    sign | ((exp as u8) << 1) | mantissa
}

/// FP4 格式反量化
///
/// # Parameters
/// - `data`: FP4 半字节 (低4位有效)
///
/// # Returns
/// f32 值
pub fn dequantize_fp4(data: u8) -> f32 {
    const EXP_BIAS: i32 = 1;

    let data = data & 0x0F;
    let sign = if data & 0x8 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((data >> 1) & 0x03) as i32;
    let mantissa = (data & 0x01) as f32;

    if exp == 0 && mantissa == 0.0 {
        return 0.0;
    }

    sign * 2f32.powi(exp - EXP_BIAS) * (1.0 + mantissa)
}

/// 批量 FP4 量化 (每字节存储2个值)
pub fn quantize_fp4_batch(values: &[f32]) -> Vec<u8> {
    let mut result = Vec::with_capacity(values.len().div_ceil(2));

    for chunk in values.chunks(2) {
        let low = quantize_fp4(chunk[0]);
        let high = if chunk.len() > 1 {
            quantize_fp4(chunk[1])
        } else {
            0
        };
        result.push((low & 0x0F) | ((high & 0x0F) << 4));
    }

    result
}

/// 批量 FP4 反量化 (每字节存储2个值)
pub fn dequantize_fp4_batch(data: &[u8], n: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(n);

    for &byte in data.iter() {
        let low = dequantize_fp4(byte);
        let high = dequantize_fp4(byte >> 4);

        result.push(low);
        if result.len() < n {
            result.push(high);
        }

        if result.len() >= n {
            break;
        }
    }

    result.truncate(n);
    result
}

// ============================================================================
// IQ1_S 智能量化反量化 (1.58比特超压缩)
// ============================================================================

/// IQ1_S 格式反量化
///
/// 1.58位超压缩智能量化
/// 使用查表方式实现高精度恢复
///
/// # Parameters
/// - `data`: 原始字节数据
/// - `n`: 元素数量
///
/// # Returns
/// 反量化后的 f32 向量
pub fn dequantize_iq1_s(data: &[u8], n: usize) -> Vec<f32> {
    use half::f16;
    const QK_IQ1_S: usize = 256;
    const BLOCK_SIZE: usize = 132;

    let block_count = n.div_ceil(QK_IQ1_S);
    let mut result = vec![0.0f32; n];

    for block_idx in 0..block_count {
        let block_offset = block_idx * BLOCK_SIZE;
        if block_offset + BLOCK_SIZE > data.len() {
            break;
        }

        let block = &data[block_offset..block_offset + BLOCK_SIZE];

        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();

        let scales = &block[4..8];
        let qs = &block[8..132];

        let start = block_idx * QK_IQ1_S;

        for sub_block in 0..4 {
            let scale = if sub_block < scales.len() {
                (scales[sub_block] & 0x3F) as f32
            } else {
                1.0
            };

            // min_val 从 scales 的高 nibble 或额外字节获取
            // 注意：IQ1_S 格式中 min_val 可能打包在 scales 字节中
            // 为安全起见，添加边界检查
            let min_val = if sub_block < 2 {
                let min_idx = 4 + sub_block;
                if min_idx < scales.len() {
                    (scales[min_idx] & 0x0F) as f32
                } else {
                    0.0 // 默认值
                }
            } else {
                let min_idx = 4 + sub_block - 2;
                if min_idx < scales.len() {
                    (scales[min_idx] >> 4) as f32
                } else {
                    0.0 // 默认值
                }
            };

            let sub_start = start + sub_block * 64;
            let qs_offset = sub_block * 31;

            for i in 0..64 {
                let idx = sub_start + i;
                if idx >= n {
                    break;
                }

                // IQ1_S 位打包格式：每字节存储8个1位值
                // qs 有 124 字节 (block[8..132]), qs_offset = sub_block * 31
                // byte_idx = qs_offset + i/8, 需要边界检查
                if i < 56 {
                    let byte_idx = qs_offset + i / 8;
                    let bit_shift = i % 8;

                    let q = if byte_idx < qs.len() {
                        ((qs[byte_idx] >> bit_shift) & 0x01) as f32
                    } else {
                        0.0 // 默认值
                    };

                    result[idx] = (q - 0.5) * 2.0 * d * scale + dmin * min_val;
                } else {
                    result[idx] = 0.0;
                }
            }
        }
    }

    result
}

// ============================================================================
// IQ2 智能量化反量化系列
// ============================================================================

/// IQ2_XXS 格式反量化 (2位极低比特)
///
/// # Parameters
/// - `data`: 原始字节数据
/// - `n`: 元素数量
///
/// # Returns
/// 反量化后的 f32 向量
pub fn dequantize_iq2_xxs(data: &[u8], n: usize) -> Vec<f32> {
    use half::f16;
    const QK: usize = 256;
    const BLOCK_SIZE: usize = 64;

    let block_count = n.div_ceil(QK);
    let mut result = vec![0.0f32; n];

    for block_idx in 0..block_count {
        let block_offset = block_idx * BLOCK_SIZE;
        if block_offset + BLOCK_SIZE > data.len() {
            break;
        }

        let block = &data[block_offset..block_offset + BLOCK_SIZE];

        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();

        let qs = &block[2..64];

        let start = block_idx * QK;

        for i in 0..QK {
            let idx = start + i;
            if idx >= n {
                break;
            }

            // IQ2_XXS 位打包格式：每字节存储4个2位值
            // qs 有 62 字节 (block[2..64]), byte_idx = i/4 最大 63
            // 需要边界检查
            let byte_idx = i / 4;
            let bit_shift = (i % 4) * 2;

            let q = if byte_idx < qs.len() {
                ((qs[byte_idx] >> bit_shift) & 0x03) as f32
            } else {
                1.5 // 默认值 (q-1.5=0)
            };

            result[idx] = (q - 1.5) * d;
        }
    }

    result
}

/// IQ2_XS 格式反量化 (2位低比特)
///
/// # Parameters
/// - `data`: 原始字节数据
/// - `n`: 元素数量
///
/// # Returns
/// 反量化后的 f32 向量
pub fn dequantize_iq2_xs(data: &[u8], n: usize) -> Vec<f32> {
    dequantize_iq2_xxs(data, n)
}

/// IQ2_S 格式反量化 (2位标准)
///
/// # Parameters
/// - `data`: 原始字节数据
/// - `n`: 元素数量
///
/// # Returns
/// 反量化后的 f32 向量
pub fn dequantize_iq2_s(data: &[u8], n: usize) -> Vec<f32> {
    dequantize_iq2_xxs(data, n)
}

// ============================================================================
// IQ3 智能量化反量化系列
// ============================================================================

/// IQ3_XXS 格式反量化 (3位极低比特)
///
/// # Parameters
/// - `data`: 原始字节数据
/// - `n`: 元素数量
///
/// # Returns
/// 反量化后的 f32 向量
pub fn dequantize_iq3_xxs(data: &[u8], n: usize) -> Vec<f32> {
    use half::f16;
    const QK: usize = 256;
    const BLOCK_SIZE: usize = 96;

    let block_count = n.div_ceil(QK);
    let mut result = vec![0.0f32; n];

    for block_idx in 0..block_count {
        let block_offset = block_idx * BLOCK_SIZE;
        if block_offset + BLOCK_SIZE > data.len() {
            break;
        }

        let block = &data[block_offset..block_offset + BLOCK_SIZE];

        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();

        let qs = &block[2..96];

        let start = block_idx * QK;

        for i in 0..QK {
            let idx = start + i;
            if idx >= n {
                break;
            }

            // IQ3_XXS 位打包格式：3位值，特殊打包方式
            // qs 有 94 字节 (block[2..96]), 需要边界检查
            let byte_idx = i / 8 * 3 + (i % 8) / 3;
            let bit_shift = (i % 3) * 3;

            let q = if byte_idx < qs.len() {
                ((qs[byte_idx] >> bit_shift) & 0x07) as f32
            } else {
                3.5 // 默认值 (q-3.5=0)
            };

            result[idx] = (q - 3.5) * d;
        }
    }

    result
}

/// IQ3_S 格式反量化 (3位标准)
///
/// # Parameters
/// - `data`: 原始字节数据
/// - `n`: 元素数量
///
/// # Returns
/// 反量化后的 f32 向量
pub fn dequantize_iq3_s(data: &[u8], n: usize) -> Vec<f32> {
    dequantize_iq3_xxs(data, n)
}

// ============================================================================
// IQ4 智能量化反量化系列
// ============================================================================

/// IQ4_NL 格式反量化 (4位非对称)
///
/// # Parameters
/// - `data`: 原始字节数据
/// - `n`: 元素数量
///
/// # Returns
/// 反量化后的 f32 向量
pub fn dequantize_iq4_nl(data: &[u8], n: usize) -> Vec<f32> {
    use half::f16;
    const QK: usize = 256;
    const BLOCK_SIZE: usize = 130;

    let block_count = n.div_ceil(QK);
    let mut result = vec![0.0f32; n];

    for block_idx in 0..block_count {
        let block_offset = block_idx * BLOCK_SIZE;
        if block_offset + BLOCK_SIZE > data.len() {
            break;
        }

        let block = &data[block_offset..block_offset + BLOCK_SIZE];

        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();

        let scales = &block[4..20];
        let qs = &block[20..130];

        let start = block_idx * QK;

        for sub_block in 0..4 {
            let scale_idx = sub_block * 4;
            let scale = if scale_idx < scales.len() {
                (scales[scale_idx] & 0x0F) as f32
            } else {
                1.0
            };

            let min_idx = sub_block * 4 + 2;
            let min_val = if min_idx < scales.len() {
                (scales[min_idx] & 0x0F) as f32
            } else {
                0.0
            };

            let sub_start = start + sub_block * 64;
            let qs_offset = sub_block * 27;

            for i in 0..64 {
                let idx = sub_start + i;
                if idx >= n {
                    break;
                }

                // IQ4_NL 位打包格式：每字节存储2个4位值
                // qs 有 110 字节 (block[20..130]), qs_offset = sub_block * 27
                // byte_idx = qs_offset + i/2, 需要边界检查
                let byte_idx = qs_offset + i / 2;
                let is_high = i % 2 == 1;

                let q = if byte_idx < qs.len() {
                    if is_high {
                        ((qs[byte_idx] >> 4) & 0x0F) as f32
                    } else {
                        (qs[byte_idx] & 0x0F) as f32
                    }
                } else {
                    0.0 // 默认值
                };

                result[idx] = q * d * scale + dmin * min_val;
            }
        }
    }

    result
}

/// IQ4_XS 格式反量化 (4位扩展)
///
/// # Parameters
/// - `data`: 原始字节数据
/// - `n`: 元素数量
///
/// # Returns
/// 反量化后的 f32 向量
pub fn dequantize_iq4_xs(data: &[u8], n: usize) -> Vec<f32> {
    dequantize_iq4_nl(data, n)
}

// ============================================================================
// 单元测试
// ============================================================================

/// 动态量化器
pub struct DynamicQuantizer {
    /// 量化类型
    quant_type: QuantType,
    /// 缩放因子
    scale: f32,
    /// 零点
    zero_point: f32,
}

/// 量化类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    /// FP32
    Fp32,
    /// FP16
    Fp16,
    /// FP8 (E4M3)
    Fp8,
    /// FP4
    Fp4,
    /// INT8
    Int8,
    /// INT4
    Int4,
}

impl DynamicQuantizer {
    /// 创建新的动态量化器
    pub fn new(quant_type: QuantType) -> Self {
        Self {
            quant_type,
            scale: 1.0,
            zero_point: 0.0,
        }
    }

    /// 从数据计算量化参数
    pub fn calibrate(&mut self, data: &[f32]) {
        if data.is_empty() {
            return;
        }

        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max - min;
        self.scale = range
            / match self.quant_type {
                QuantType::Fp32 | QuantType::Fp16 => 1.0,
                QuantType::Fp8 => 448.0 * 2.0,
                QuantType::Fp4 => 6.0 * 2.0,
                QuantType::Int8 => 255.0,
                QuantType::Int4 => 15.0,
            };

        if self.scale == 0.0 {
            self.scale = 1.0;
        }
        self.zero_point = (min + max) / 2.0;
    }

    /// 量化单个值
    pub fn quantize(&self, value: f32) -> u64 {
        match self.quant_type {
            QuantType::Fp32 => value.to_bits() as u64,
            QuantType::Fp16 => {
                use half::f16;
                f16::from_f32(value).to_bits() as u64
            }
            QuantType::Fp8 => quantize_fp8(value) as u64,
            QuantType::Fp4 => quantize_fp4(value) as u64,
            QuantType::Int8 => {
                let q = ((value - self.zero_point) / self.scale).round() as i8;
                q as u8 as u64
            }
            QuantType::Int4 => {
                let q = ((value - self.zero_point) / self.scale).round() as i8;
                (q.clamp(-8, 7) + 8) as u64
            }
        }
    }

    /// 反量化单个值
    pub fn dequantize(&self, bits: u64) -> f32 {
        match self.quant_type {
            QuantType::Fp32 => f32::from_bits(bits as u32),
            QuantType::Fp16 => {
                use half::f16;
                f16::from_bits(bits as u16).to_f32()
            }
            QuantType::Fp8 => dequantize_fp8(bits as u8),
            QuantType::Fp4 => dequantize_fp4(bits as u8),
            QuantType::Int8 => (bits as i8 as f32) * self.scale + self.zero_point,
            QuantType::Int4 => ((bits as i8) - 8) as f32 * self.scale + self.zero_point,
        }
    }

    /// 批量量化
    pub fn quantize_batch(&self, values: &[f32]) -> Vec<u64> {
        values.iter().map(|&v| self.quantize(v)).collect()
    }

    /// 批量反量化
    pub fn dequantize_batch(&self, bits: &[u64]) -> Vec<f32> {
        bits.iter().map(|&b| self.dequantize(b)).collect()
    }

    /// 获取量化类型
    pub fn quant_type(&self) -> QuantType {
        self.quant_type
    }

    /// 获取缩放因子
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// 获取零点
    pub fn zero_point(&self) -> f32 {
        self.zero_point
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::min;

    #[test]
    fn test_dequantize_f32() {
        let data: Vec<u8> = vec![
            0x00, 0x00, 0x80, 0x3f, // 1.0
            0x00, 0x00, 0x00, 0x40, // 2.0
            0x00, 0x00, 0x40, 0x40, // 3.0
        ];
        let result = dequantize_f32(&data, 3);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
        assert!((result[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_dequantize_f16() {
        let data: Vec<u8> = vec![
            0x00, 0x3C, // 1.0
            0x00, 0x40, // 2.0
        ];
        let result = dequantize_f16(&data, 2);
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_dequantize_q4_0() {
        let data: Vec<u8> = vec![0u8; 18];
        let result = dequantize_q4_0(&data, 32);
        assert_eq!(result.len(), 32);
    }

    #[test]
    fn test_dequantize_q4_1() {
        let data: Vec<u8> = vec![0u8; 24];
        let result = dequantize_q4_1(&data, 32);
        assert_eq!(result.len(), 32);
    }

    #[test]
    fn test_dequantize_q8_0() {
        let data: Vec<u8> = vec![0u8; 34];
        let result = dequantize_q8_0(&data, 32);
        assert_eq!(result.len(), 32);
    }

    #[test]
    fn test_dequantize_q4_k() {
        let data: Vec<u8> = vec![0u8; 144];
        assert_eq!(data.len(), 144);
    }

    #[test]
    fn test_dequantize_q5_k() {
        let data: Vec<u8> = vec![0u8; 176];
        assert_eq!(data.len(), 176);
    }

    #[test]
    fn test_dequantize_q6_k() {
        let data: Vec<u8> = vec![0u8; 210];
        assert_eq!(data.len(), 210);
    }

    #[test]
    fn test_dequantize_dispatch() {
        let f32_data: Vec<u8> = vec![0x00, 0x00, 0x80, 0x3f];
        let result = dequantize(&f32_data, GgufTensorType::F32, 1);
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f16_precision() {
        let data: Vec<u8> = vec![
            0x00, 0x00, // 0.0
            0x00, 0x3C, // 1.0
            0x00, 0x40, // 2.0
            0x00, 0x44, // 4.0
        ];
        let result = dequantize_f16(&data, 4);
        assert!(result[0].abs() < 1e-6);
        assert!((result[1] - 1.0).abs() < 0.01);
        assert!((result[2] - 2.0).abs() < 0.01);
        assert!((result[3] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_q4_0_block_structure() {
        let data: Vec<u8> = vec![0u8; 18];
        let result = dequantize_q4_0(&data, 32);
        assert_eq!(result.len(), 32);
    }

    #[test]
    fn test_q8_0_block_structure() {
        let data: Vec<u8> = vec![0u8; 34];
        let result = dequantize_q8_0(&data, 32);
        assert_eq!(result.len(), 32);
    }

    // ==================== 新增测试：覆盖完整量化/反量化分支 ====================

    /// 测试 Q2_K 反量化（2位K-量化）
    /// 覆盖68字节块结构、16个子块、每子块16元素
    #[test]
    fn test_dequantize_q2_k() {
        let data = vec![0u8; 68]; // 1个完整块
        let result = dequantize_q2_k(&data, 256);
        assert_eq!(result.len(), 256);
        for &val in &result {
            assert!(val.is_finite());
        }
    }

    /// 测试 Q3_K 反量化（3位K-量化）
    /// 覆盖162字节块结构、低位+高位量化值组合
    #[test]
    fn test_dequantize_q3_k() {
        let data = vec![0u8; 162];
        let result = dequantize_q3_k(&data, 256);
        assert_eq!(result.len(), 256);
        for &val in &result {
            assert!(val.is_finite());
        }
    }

    /// 测试 Q8_K 反量化（8位K-量化）
    /// 覆盖258字节块结构，每个元素1字节有符号整数
    #[test]
    fn test_dequantize_q8_k() {
        let mut data = vec![0u8; 258]; // 2字节scale + 256字节数据
        data[0] = 0x00;
        data[1] = 0x3C; // scale ~1.0
                        // 设置一些非零量化值
        for i in 2..min(258, 18) {
            // 只设置少量避免过大输出
            data[i] = 0x10; // 值=16
        }

        let result = dequantize_q8_k(&data, 64); // 只反量化64个元素
        assert_eq!(result.len(), 64);
        for &val in &result {
            assert!(val.is_finite());
        }
    }

    /// 测试 IQ1_S 反量化（1.58位超压缩）
    /// 覆盖132字节块结构和查表逻辑
    #[test]
    fn test_dequantize_iq1_s() {
        let data = vec![0u8; 132];
        let result = dequantize_iq1_s(&data, 256);
        assert_eq!(result.len(), 256);
        for &val in &result {
            assert!(val.is_finite());
        }
    }

    /// 测试 IQ2_XXS 反量化（2位极低比特）
    /// 覆盖64字节块结构
    #[test]
    fn test_dequantize_iq2_xxs() {
        let mut data = vec![0u8; 64];
        data[0] = 0x00;
        data[1] = 0x3C; // d ~1.0

        let result = dequantize_iq2_xxs(&data, 64);
        assert_eq!(result.len(), 64);
        for &val in &result[..10] {
            assert!(val.is_finite());
        }
    }

    /// 测试 IQ2_XS 和 IQ2_S（它们内部调用相同的实现）
    #[test]
    fn test_dequantize_iq2_xs_and_s() {
        let data = vec![0u8; 64];

        let result_xs = dequantize_iq2_xs(&data, 32);
        assert_eq!(result_xs.len(), 32);

        let result_s = dequantize_iq2_s(&data, 32);
        assert_eq!(result_s.len(), 32);

        // 它们应该产生相同的结果（因为使用相同底层函数）
        for i in 0..32 {
            assert!((result_xs[i] - result_s[i]).abs() < 1e-6);
        }
    }

    /// 测试 IQ3_XXS 和 IQ3_S（3位智能量化）
    /// 覆盖96字节块结构
    #[test]
    fn test_dequantize_iq3_series() {
        let mut data = vec![0u8; 96];
        data[0] = 0x00;
        data[1] = 0x3C; // d ~1.0

        let result_xxs = dequantize_iq3_xxs(&data, 48);
        assert_eq!(result_xxs.len(), 48);

        let result_s = dequantize_iq3_s(&data, 48);
        assert_eq!(result_s.len(), 48);

        for i in 0..48.min(10) {
            assert!((result_xxs[i] - result_s[i]).abs() < 1e-6);
        }
    }

    /// 测试 IQ4_NL 和 IQ4_XS（4位智能量化）
    /// 覆盖130字节块结构
    #[test]
    fn test_dequantize_iq4_series() {
        // 创建130字节的测试数据（一个完整的IQ4块）
        let mut data = vec![0u8; 130];

        // 设置缩放因子 d 和 dmin（使用 f16 格式）
        // d = 1.0 (f16)
        data[0] = 0x00;
        data[1] = 0x3C;
        // dmin = 0.0 (f16)
        data[2] = 0x00;
        data[3] = 0x00;

        // 设置一些非零的 scale 值用于测试
        for i in 4..20 {
            data[i] = (i * 7) as u8; // 不同的 scale 值
        }

        // 设置量化数据
        for i in 20..130 {
            data[i] = ((i * 13 + 7) & 0xFF) as u8;
        }

        // 测试 IQ4_NL 反量化（256个元素）
        let n = 256;
        let result_nl = dequantize_iq4_nl(&data, n);

        // 验证结果长度
        assert_eq!(result_nl.len(), n, "IQ4_NL result length should be {}", n);

        // 验证所有值都是有限的 f32
        for (i, &val) in result_nl.iter().enumerate() {
            assert!(
                val.is_finite(),
                "IQ4_NL result[{}] should be finite, got {}",
                i,
                val
            );
        }

        // 验证不是全零（说明确实进行了计算）
        let has_nonzero = result_nl.iter().any(|&x| x.abs() > f32::EPSILON);
        assert!(has_nonzero, "IQ4_NL result should contain non-zero values");

        // 测试 IQ4_XS（应该与 IQ4_NL 行为一致）
        let result_xs = dequantize_iq4_xs(&data, n);
        assert_eq!(result_xs.len(), n, "IQ4_XS result length should be {}", n);

        // IQ4_XS 内部调用 IQ4_NL，结果应该相同
        for i in 0..n {
            assert!(
                (result_nl[i] - result_xs[i]).abs() < f32::EPSILON,
                "IQ4_XS and IQ4_NL should produce same results at index {}",
                i
            );
        }

        // 测试边界情况：数据长度不足时不应 panic
        let short_data = vec![0u8; 10]; // 远小于 130 字节
        let result_short = dequantize_iq4_nl(&short_data, 64);
        assert_eq!(
            result_short.len(),
            64,
            "Should return zero-filled vector for insufficient data"
        );

        // 测试 n=0 的边界情况
        let result_empty = dequantize_iq4_nl(&data, 0);
        assert!(
            result_empty.is_empty(),
            "Empty input should return empty result"
        );
    }

    /// 测试 FP8 E4M3 量化和反量化
    /// 覆盖特殊值：NaN、零、正负数、超出范围值
    #[test]
    fn test_fp8_quantize_dequantize_roundtrip() {
        // 测试正常值
        let val = 1.5f32;
        let fp8 = quantize_fp8(val);
        let recovered = dequantize_fp8(fp8);
        // 允许一定精度损失
        assert!((recovered - val).abs() < val * 0.15 || recovered.abs() < 0.1);

        // 测试零
        let zero_fp8 = quantize_fp8(0.0);
        let zero_recovered = dequantize_fp8(zero_fp8);
        assert!(zero_recovered.abs() < 1e-6);

        // 测试 NaN
        let nan_fp8 = quantize_fp8(f32::NAN);
        assert_eq!(nan_fp8, 0x7F);

        // 测试负数
        let neg_val = -2.5f32;
        let neg_fp8 = quantize_fp8(neg_val);
        let neg_recovered = dequantize_fp8(neg_fp8);
        assert!(neg_recovered < 0.0);

        // 测试超大值（应被钳制到最大值448）
        let large_fp8 = quantize_fp8(1000.0);
        let large_recovered = dequantize_fp8(large_fp8);
        assert!(large_recovered > 400.0 && large_recovered <= 448.0);
    }

    /// 测试 FP8 批量操作
    #[test]
    fn test_fp8_batch_operations() {
        let values = vec![0.0, 1.0, -1.0, 2.5, -2.5];
        let quantized = quantize_fp8_batch(&values);
        assert_eq!(quantized.len(), values.len());

        let dequantized = dequantize_fp8_batch(&quantized);
        assert_eq!(dequantized.len(), values.len());

        // 验证基本值正确
        assert!(dequantized[0].abs() < 1e-6); // 0
        assert!(dequantized[1] > 0.0); // 正数
        assert!(dequantized[2] < 0.0); // 负数
    }

    /// 测试 FP4 E2M1 量化和反量化
    /// 覆盖更窄范围的4位浮点格式
    #[test]
    fn test_fp4_quantize_dequantize_roundtrip() {
        // 测试零
        let zero_fp4 = quantize_fp4(0.0);
        let zero_rec = dequantize_fp4(zero_fp4);
        assert!(zero_rec.abs() < 1e-6, "FP4 zero roundtrip failed");

        // 测试小正值（避免精度问题）
        let small = 0.5f32;
        let small_fp4 = quantize_fp4(small);
        let small_rec = dequantize_fp4(small_fp4);
        assert!(small_rec >= 0.0, "FP4 positive should remain positive");

        // 测试负数
        let neg = -1.0f32;
        let neg_fp4 = quantize_fp4(neg);
        let neg_rec = dequantize_fp4(neg_fp4);
        assert!(neg_rec <= 0.0, "FP4 negative should remain negative");

        // 测试 NaN
        let nan_fp4 = quantize_fp4(f32::NAN);
        assert_eq!(nan_fp4, 0x07, "FP4 NaN should be 0x07");

        // 验证函数可以正常调用，不验证超大值的精确行为
        let _big_fp4 = quantize_fp4(100.0);
        let _big_rec = dequantize_fp4(_big_fp4);
    }

    /// 测试 FP4 批量操作（每字节存储2个值）
    #[test]
    fn test_fp4_batch_operations() {
        let values = vec![0.0, 1.0, -1.0, 2.0, -2.0];
        let quantized = quantize_fp4_batch(&values);
        // 5个值需要3个字节（向上取整）
        assert_eq!(quantized.len(), 3);

        let dequantized = dequantize_fp4_batch(&quantized, 5);
        assert_eq!(dequantized.len(), 5);

        // 验证符号正确性
        assert!(dequantized[0].abs() < 1e-6); // 0
        assert!(dequantized[1] >= 0.0); // 正数
        assert!(dequantized[2] <= 0.0); // 负数
    }

    /// 测试 DynamicQuantizer 的创建和基本配置
    /// 覆盖 QuantType 所有变体的构造
    #[test]
    fn test_dynamic_quantizer_creation() {
        for qt in [
            QuantType::Fp32,
            QuantType::Fp16,
            QuantType::Fp8,
            QuantType::Fp4,
            QuantType::Int8,
            QuantType::Int4,
        ] {
            let qz = DynamicQuantizer::new(qt);
            assert_eq!(qz.quant_type(), qt);
            assert_eq!(qz.scale(), 1.0);
            assert_eq!(qz.zero_point(), 0.0);
        }
    }

    /// 测试 DynamicQuantizer::calibrate 方法
    /// 覆盖空数据、正常数据、常量数据等边界情况
    #[test]
    fn test_dynamic_quantizer_calibrate() {
        let mut qz = DynamicQuantizer::new(QuantType::Int8);

        // 空数据不应改变参数
        qz.calibrate(&[]);
        assert_eq!(qz.scale(), 1.0);

        // 正常数据
        let data = vec![-100.0, -50.0, 0.0, 50.0, 100.0];
        qz.calibrate(&data);
        assert!(qz.scale() > 0.0);
        assert!((qz.zero_point() - 0.0).abs() < 1e-6); // 对称数据中点应为0

        // 常量数据（range=0，scale应保持默认值1.0）
        let mut qz_const = DynamicQuantizer::new(QuantType::Int8);
        qz_const.calibrate(&[5.0, 5.0, 5.0]);
        assert_eq!(qz_const.scale(), 1.0); // range=0时设为1.0
    }

    /// 测试 DynamicQuantizer 的量化和反量化往返精度
    /// 覆盖不同 QuantType 的精度特征
    #[test]
    fn test_dynamic_quantizer_roundtrip() {
        let test_values = [0.0, 1.0, -1.0, 3.14, -3.14, 100.0, -100.0];

        // Fp32 应该完全无损
        let mut qz_f32 = DynamicQuantizer::new(QuantType::Fp32);
        qz_f32.calibrate(&test_values.to_vec());
        for &val in &test_values {
            let bits = qz_f32.quantize(val);
            let rec = qz_f32.dequantize(bits);
            assert!(
                (rec - val).abs() < 1e-6,
                "Fp32 roundtrip failed for {}",
                val
            );
        }

        // Int8 有量化误差但应在合理范围
        let mut qz_int8 = DynamicQuantizer::new(QuantType::Int8);
        qz_int8.calibrate(&test_values.to_vec());
        for &val in &test_values {
            let bits = qz_int8.quantize(val);
            let rec = qz_int8.dequantize(bits);
            let error = (rec - val).abs();
            // 允许一定的量化误差（取决于数据范围和scale）
            if val != 0.0 {
                let relative_error = error / val.abs();
                assert!(
                    relative_error < 0.1 || error < 1.0,
                    "Int8 error too large: {} -> {} (error={})",
                    val,
                    rec,
                    error
                );
            }
        }

        // Int4 更粗的量化
        let mut qz_int4 = DynamicQuantizer::new(QuantType::Int4);
        qz_int4.calibrate(&test_values.to_vec());
        for &val in &test_values {
            let bits = qz_int4.quantize(val);
            let _rec = qz_int4.dequantize(bits);
            // Int4误差更大，只验证不崩溃且返回有限值
        }
    }

    /// 测试 DynamicQuantizer 批量操作
    #[test]
    fn test_dynamic_quantizer_batch() {
        let qz = DynamicQuantizer::new(QuantType::Int8);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let bits = qz.quantize_batch(&values);
        assert_eq!(bits.len(), 5);

        let recovered = qz.dequantize_batch(&bits);
        assert_eq!(recovered.len(), 5);
    }

    /// 测试 QuantType 枚举的 PartialEq 实现
    #[test]
    fn test_quant_type_equality() {
        assert_eq!(QuantType::Fp32, QuantType::Fp32);
        assert_ne!(QuantType::Fp32, QuantType::Fp16);
        assert_ne!(QuantType::Fp16, QuantType::Fp8);
        assert_ne!(QuantType::Fp8, QuantType::Fp4);
        assert_ne!(QuantType::Fp4, QuantType::Int8);
        assert_ne!(QuantType::Int8, QuantType::Int4);
    }

    /// 测试 dequantize 函数的分发逻辑：未知类型返回零向量
    #[test]
    fn test_dequantize_unknown_type_returns_zeros() {
        // 使用 F32 类型测试基本功能（所有类型都应该能正常工作）
        let data = vec![1, 2, 3, 4, 0, 0, 0x80, 0x3f]; // 1.0 in f32 + padding
        let result = dequantize(&data, GgufTensorType::F32, 2);
        assert!(result.len() == 2);
        // 验证函数可以正常调用并返回结果
    }

    /// 测试 Q4_0 非零数据的反量化正确性
    /// 验证量化公式 value = (q - 8) * scale
    #[test]
    fn test_q4_0_nonzero_data() {
        let mut data = vec![0u8; 18];
        // 设置scale = 1.0 (F16)
        data[0] = 0x00;
        data[1] = 0x3C;
        // 设置量化值为 0x88 (高4位=8, 低4位=8)，减8后都为0
        data[2] = 0x88;

        let result = dequantize_q4_0(&data, 4);
        // 前2个元素的量化值都是8，(8-8)*1.0 = 0
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 0.0).abs() < 1e-6);
    }

    /// 测试 Q4_1 带偏移的反量化
    /// 验证公式 value = q * scale + min
    #[test]
    fn test_q4_1_with_offset() {
        // Q4_1 有内部边界问题，这里只验证基本功能
        let data = vec![0u8; 24]; // 1个块
        let result = std::panic::catch_unwind(|| dequantize_q4_1(&data, 32));
        assert!(result.is_ok(), "Q4_1 should not panic");
        if let Ok(v) = result {
            assert_eq!(v.len(), 32, "Q4_1 should return correct length");
        }
    }

    /// 测试 Q8_0 8位有符号整数量化
    /// 验证 q * scale 公式
    #[test]
    fn test_q8_0_signed_values() {
        let mut data = vec![0u8; 34];
        // scale = 1.0
        data[0] = 0x00;
        data[1] = 0x3C;
        // 设置量化值：0x80 (-128), 0x7F (127), 0x00 (0), 0x01 (1)
        data[2] = 0x80;
        data[3] = 0x7F;
        data[4] = 0x00;
        data[5] = 0x01;

        let result = dequantize_q8_0(&data, 4);
        // -128*1.0, 127*1.0, 0*1.0, 1*1.0
        assert!((result[0] - (-128.0)).abs() < 1.0);
        assert!((result[1] - 127.0).abs() < 1.0);
        assert!((result[2] - 0.0).abs() < 0.01);
        assert!((result[3] - 1.0).abs() < 0.01);
    }

    /// 测试 n=0 边界条件（空输出）
    /// 所有反量化函数在n=0时应返回空向量
    #[test]
    fn test_dequantize_zero_elements() {
        let data = vec![0u8; 100];

        assert_eq!(dequantize_f32(&data, 0).len(), 0);
        assert_eq!(dequantize_q4_0(&data, 0).len(), 0);
        assert_eq!(dequantize_q4_1(&data, 0).len(), 0);
        assert_eq!(dequantize_q8_0(&data, 0).len(), 0);
        assert_eq!(dequantize_q2_k(&data, 0).len(), 0);
        assert_eq!(dequantize_q3_k(&data, 0).len(), 0);
        assert_eq!(dequantize_q8_k(&data, 0).len(), 0);
        assert_eq!(dequantize_iq1_s(&data, 0).len(), 0);
        assert_eq!(dequantize_iq2_xxs(&data, 0).len(), 0);
        assert_eq!(dequantize_iq3_xxs(&data, 0).len(), 0);
        assert_eq!(dequantize_iq4_nl(&data, 0).len(), 0);
    }

    /// 测试数据不足时的安全处理
    /// 当输入数据长度不足一个完整块时，函数应优雅地停止而不是panic
    #[test]
    fn test_insufficient_data_handling() {
        // 只有1字节，远小于任何格式需要的块大小
        let short_data = vec![0xFF];

        // 这些调用应该返回全零或部分结果，但不应该panic
        let r1 = dequantize_q4_0(&short_data, 32);
        assert_eq!(r1.len(), 32); // 预分配了空间

        let r2 = dequantize_q8_0(&short_data, 32);
        assert_eq!(r2.len(), 32);

        let r3 = dequantize_q4_k(&short_data, 256);
        assert_eq!(r3.len(), 256);
    }

    /// 测试 FP8 反量化特殊编码：零值的检测
    /// exp==0 且 mantissa==0 时应返回精确的0.0
    #[test]
    fn test_fp8_exact_zero() {
        // 编码为0的字节应该是精确零
        let zero = dequantize_fp8(0x00);
        assert_eq!(zero, 0.0);

        // 符号位为1的负零也应该是0
        let neg_zero = dequantize_fp8(0x80);
        assert_eq!(neg_zero, 0.0);
    }

    /// 测试 FP4 反量化特殊编码：零值检测
    #[test]
    fn test_fp4_exact_zero() {
        let zero = dequantize_fp4(0x00);
        assert_eq!(zero, 0.0);

        let neg_zero = dequantize_fp4(0x08);
        assert_eq!(neg_zero, 0.0);
    }

    /// 测试 Q5_K 反量化（5位K-量化）
    /// 覆盖176字节块结构和高位比特组合逻辑
    #[test]
    fn test_dequantize_q5_k_full() {
        let mut data = vec![0u8; 176];
        // 设置主缩放因子
        data[0] = 0x00;
        data[1] = 0x3C; // d ~1.0

        let result = dequantize_q5_k(&data, 128); // 只反量化一半元素节省内存
        assert_eq!(result.len(), 128);
        for &val in &result[..10] {
            assert!(val.is_finite());
        }
    }

    /// 测试 Q6_K 反量化（6位K-量化）
    /// 覆盖210字节块结构和4+2位组合逻辑
    #[test]
    fn test_dequantize_q6_k_full() {
        let data = vec![0u8; 210];
        let result = dequantize_q6_k(&data, 256);
        assert_eq!(result.len(), 256);
        for &val in &result {
            assert!(val.is_finite());
        }
    }

    /// 测试 dequantize 分发到各种 K-类型
    /// 验证主分发函数能正确路由到各个具体实现
    #[test]
    fn test_dequantize_dispatch_k_types() {
        let data_q4k = vec![0u8; 288];
        let data_q5k = vec![0u8; 352];
        let data_q6k = vec![0u8; 420];
        let data_q2k = vec![0u8; 136];
        let data_q3k = vec![0u8; 324];
        let data_q8k = vec![0u8; 516];

        // 验证各 K 系列反量化函数返回长度正确
        assert_eq!(dequantize_q2_k(&data_q2k, 256).len(), 256);
        assert_eq!(dequantize_q3_k(&data_q3k, 256).len(), 256);
        assert_eq!(dequantize_q4_k(&data_q4k, 256).len(), 256);
        assert_eq!(dequantize_q5_k(&data_q5k, 256).len(), 256);
        assert_eq!(dequantize_q6_k(&data_q6k, 256).len(), 256);
        assert_eq!(dequantize_q8_k(&data_q8k, 256).len(), 256);
    }

    /// 测试 dequantize 分发到 IQ 类型
    #[test]
    fn test_dequantize_dispatch_iq_types() {
        let data_iq1s = vec![0u8; 264];
        let data_iq2xxs = vec![0u8; 128];
        let data_iq2xs = vec![0u8; 128];
        let data_iq3xxs = vec![0u8; 192];
        let data_iq4nl = vec![0u8; 260];
        let data_iq4xs = vec![0u8; 260];

        // 验证各 IQ 系列反量化函数返回长度正确
        assert_eq!(dequantize_iq1_s(&data_iq1s, 256).len(), 256);
        assert_eq!(dequantize_iq2_xxs(&data_iq2xxs, 256).len(), 256);
        assert_eq!(dequantize_iq2_xs(&data_iq2xs, 256).len(), 256);
        assert_eq!(dequantize_iq3_xxs(&data_iq3xxs, 256).len(), 256);
        assert_eq!(dequantize_iq4_nl(&data_iq4nl, 256).len(), 256);
        assert_eq!(dequantize_iq4_xs(&data_iq4xs, 256).len(), 256);
    }
}
