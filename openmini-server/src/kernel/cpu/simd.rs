//! SIMD优化的CPU内核
//!
//! 支持AVX2/AVX512/NEON指令集

#![allow(clippy::needless_range_loop)] // CPU SIMD 内核：使用索引循环以优化性能

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
pub fn is_avx2_supported() -> bool {
    is_x86_feature_detected!("avx2")
}

#[cfg(target_arch = "x86_64")]
pub fn is_avx512_supported() -> bool {
    is_x86_feature_detected!("avx512f")
}

#[cfg(target_arch = "aarch64")]
pub fn is_neon_supported() -> bool {
    true // NEON在ARM64上总是可用
}

/// AVX2优化的向量加法
///
/// # Safety
/// - 调用者必须确保 CPU 支持 AVX2 指令集（使用 `is_avx2_supported()` 检查）
/// - `a`, `b`, `c` 必须长度相同，且长度是 8 的倍数或能正确处理剩余元素
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn add_avx2(a: &[f32], b: &[f32], c: &mut [f32]) {
    let len = a.len();
    let chunks = len / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c.as_mut_ptr().add(offset), vc);
    }

    // 处理剩余元素
    for i in (chunks * 8)..len {
        c[i] = a[i] + b[i];
    }
}

/// AVX2优化的向量乘法
///
/// # Safety
/// - 调用者必须确保 CPU 支持 AVX2 指令集（使用 `is_avx2_supported()` 检查）
/// - `a`, `b`, `c` 必须长度相同，且长度是 8 的倍数或能正确处理剩余元素
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn mul_avx2(a: &[f32], b: &[f32], c: &mut [f32]) {
    let len = a.len();
    let chunks = len / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(c.as_mut_ptr().add(offset), vc);
    }

    // 处理剩余元素
    for i in (chunks * 8)..len {
        c[i] = a[i] * b[i];
    }
}

/// AVX2优化的Softmax
///
/// # Safety
/// - 调用者必须确保 CPU 支持 AVX2 指令集（使用 `is_avx2_supported()` 检查）
/// - `x` 和 `out` 必须长度相同
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn softmax_avx2(x: &[f32], out: &mut [f32]) {
    let len = x.len();

    // 找最大值
    let mut max_val = f32::NEG_INFINITY;
    let chunks = len / 8;

    let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
    for i in 0..chunks {
        let offset = i * 8;
        let vx = _mm256_loadu_ps(x.as_ptr().add(offset));
        vmax = _mm256_max_ps(vmax, vx);
    }

    let mut temp = [0.0f32; 8];
    _mm256_storeu_ps(temp.as_mut_ptr(), vmax);
    for &v in &temp {
        max_val = max_val.max(v);
    }

    for i in (chunks * 8)..len {
        max_val = max_val.max(x[i]);
    }

    // 计算exp(x - max)
    let vmax = _mm256_set1_ps(max_val);
    let mut sum = 0.0f32;

    for i in 0..chunks {
        let offset = i * 8;
        let vx = _mm256_loadu_ps(x.as_ptr().add(offset));
        let vsub = _mm256_sub_ps(vx, vmax);

        // 近似exp计算（使用多项式近似）
        let vexp_result = {
            // exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24
            let one = _mm256_set1_ps(1.0);
            let two = _mm256_set1_ps(2.0);
            let six = _mm256_set1_ps(6.0);
            let twenty_four = _mm256_set1_ps(24.0);

            let x = vsub;
            let x2 = _mm256_mul_ps(x, x);
            let x3 = _mm256_mul_ps(x2, x);
            let x4 = _mm256_mul_ps(x3, x);

            let term1 = one;
            let term2 = x;
            let term3 = _mm256_div_ps(x2, two);
            let term4 = _mm256_div_ps(x3, six);
            let term5 = _mm256_div_ps(x4, twenty_four);

            let sum1 = _mm256_add_ps(term1, term2);
            let sum2 = _mm256_add_ps(sum1, term3);
            let sum3 = _mm256_add_ps(sum2, term4);
            _mm256_add_ps(sum3, term5)
        };
        _mm256_storeu_ps(out.as_mut_ptr().add(offset), vexp_result);

        let mut temp = [0.0f32; 8];
        _mm256_storeu_ps(temp.as_mut_ptr(), vexp_result);
        sum += temp.iter().sum::<f32>();
    }

    for i in (chunks * 8)..len {
        out[i] = (x[i] - max_val).exp();
        sum += out[i];
    }

    // 归一化
    let vsum = _mm256_set1_ps(sum);
    for i in 0..chunks {
        let offset = i * 8;
        let vout = _mm256_loadu_ps(out.as_ptr().add(offset));
        let vnorm = _mm256_div_ps(vout, vsum);
        _mm256_storeu_ps(out.as_mut_ptr().add(offset), vnorm);
    }

    for i in (chunks * 8)..len {
        out[i] /= sum;
    }
}

/// NEON优化的向量加法
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn add_neon(a: &[f32], b: &[f32], c: &mut [f32]) {
    use std::arch::aarch64::*;

    let len = a.len();
    let chunks = len / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a.as_ptr().add(offset));
        let vb = vld1q_f32(b.as_ptr().add(offset));
        let vc = vaddq_f32(va, vb);
        vst1q_f32(c.as_mut_ptr().add(offset), vc);
    }

    for i in (chunks * 4)..len {
        c[i] = a[i] + b[i];
    }
}

/// 自动选择最优的向量加法实现
pub fn add_vector(a: &[f32], b: &[f32], c: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_supported() {
            unsafe { add_avx2(a, b, c) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { add_neon(a, b, c) };
        return;
    }

    // 回退到标量实现
    for i in 0..a.len() {
        c[i] = a[i] + b[i];
    }
}

/// 自动选择最优的向量乘法实现
pub fn mul_vector(a: &[f32], b: &[f32], c: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_supported() {
            unsafe { mul_avx2(a, b, c) };
            return;
        }
    }

    // 回退到标量实现
    for i in 0..a.len() {
        c[i] = a[i] * b[i];
    }
}

/// 自动选择最优的Softmax实现
pub fn softmax_vector(x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_supported() {
            unsafe { softmax_avx2(x, out) };
            return;
        }
    }

    // 回退到标量实现
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for i in 0..x.len() {
        out[i] = (x[i] - max_val).exp();
        sum += out[i];
    }
    for i in 0..x.len() {
        out[i] /= sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_vector() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut c = vec![0.0; 8];

        add_vector(&a, &b, &mut c);

        for i in 0..8 {
            assert!((c[i] - 9.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_vector() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 4];

        softmax_vector(&x, &mut out);

        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // ========== 新增测试开始 ==========

    /// 测试向量乘法基本功能
    #[test]
    fn test_mul_vector_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let mut c = vec![0.0; 4];

        mul_vector(&a, &b, &mut c);

        // 验证逐元素乘法结果
        assert!((c[0] - 2.0).abs() < 1e-5); // 1*2
        assert!((c[1] - 6.0).abs() < 1e-5); // 2*3
        assert!((c[2] - 12.0).abs() < 1e-5); // 3*4
        assert!((c[3] - 20.0).abs() < 1e-5); // 4*5
    }

    /// 测试非对齐长度的向量操作（覆盖SIMD剩余元素处理分支）
    #[test]
    fn test_add_vector_non_aligned_length() {
        // 测试长度不是SIMD宽度倍数的情况（x86_64: 8, aarch64: 4）
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]; // 11个元素
        let b = vec![11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut c = vec![0.0; 11];

        add_vector(&a, &b, &mut c);

        // 所有元素应该正确相加
        for i in 0..11 {
            assert!(
                (c[i] - 12.0).abs() < 1e-5,
                "索引 {} 失败: 期望 12.0, 实际 {}",
                i,
                c[i]
            );
        }
    }

    /// 测试单元素向量的边界条件
    #[test]
    fn test_single_element_vectors() {
        let a = vec![42.0];
        let b = vec![8.0];
        let mut c = vec![0.0; 1];

        add_vector(&a, &b, &mut c);
        assert!((c[0] - 50.0).abs() < 1e-5);

        mul_vector(&a, &b, &mut c);
        assert!((c[0] - 336.0).abs() < 1e-5);
    }

    /// 测试包含零值和负数的向量操作
    #[test]
    fn test_zero_and_negative_values() {
        let a = vec![0.0, -1.0, 2.0, -3.0, 4.0, -5.0];
        let b = vec![10.0, -20.0, 30.0, -40.0, 50.0, -60.0];
        let mut c = vec![0.0; 6];

        add_vector(&a, &b, &mut c);

        assert!((c[0] - 10.0).abs() < 1e-5); // 0 + 10
        assert!((c[1] - (-21.0)).abs() < 1e-5); // -1 + (-20)
        assert!((c[2] - 32.0).abs() < 1e-5); // 2 + 30
        assert!((c[3] - (-43.0)).abs() < 1e-5); // -3 + (-40)
    }

    /// 测试极端数值（极大值、极小值、接近零）
    #[test]
    fn test_extreme_values() {
        use std::f32;

        let a = vec![f32::MAX, f32::MIN_POSITIVE, 1e-10, 1e10];
        let b = vec![f32::MAX, f32::MIN_POSITIVE, 1e-10, 1e10];
        let mut c = vec![0.0; 4];

        add_vector(&a, &b, &mut c);

        // 验证极大值相加（可能溢出到infinity）
        assert!(c[0].is_infinite()); // MAX + MAX = infinity

        // 验证极小正值
        assert!((c[1] - 2.0 * f32::MIN_POSITIVE).abs() < 1e-15);

        // 验证科学计数法数值
        assert!((c[2] - 2e-10).abs() < 1e-18);
        assert!((c[3] - 2e10).abs() < 1e-5);
    }

    /// 测试Softmax的单元素和全相同值情况
    #[test]
    fn test_softmax_edge_cases() {
        // 单元素softmax
        let x = vec![5.0];
        let mut out = vec![0.0; 1];
        softmax_vector(&x, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-5); // 单元素概率应为1

        // 全相同值的softmax（均匀分布）
        let x_same = vec![3.0, 3.0, 3.0, 3.0];
        let mut out_same = vec![0.0; 4];
        softmax_vector(&x_same, &mut out_same);
        let expected_prob = 0.25;
        for val in &out_same {
            assert!((*val - expected_prob).abs() < 1e-5);
        }
    }

    /// 测试Softmax的大数值稳定性（防止exp溢出）
    #[test]
    fn test_softmax_large_values_stability() {
        let x = vec![1000.0, 1001.0, 999.0, 998.0];
        let mut out = vec![0.0; 4];

        // 不应panic或产生NaN
        softmax_vector(&x, &mut out);

        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // 最大值应有最高概率
        let max_idx = out
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        assert_eq!(max_idx, 1); // 1001.0 应该是最大概率
    }

    /// 测试SIMD特性检测函数
    #[test]
    fn test_simd_feature_detection() {
        #[cfg(target_arch = "x86_64")]
        {
            // AVX2检测结果应该是布尔值
            let avx2_supported = is_avx2_supported();
            println!("AVX2 supported: {}", avx2_supported);

            let avx512_supported = is_avx512_supported();
            println!("AVX-512 supported: {}", avx512_supported);
        }

        #[cfg(target_arch = "aarch64")]
        {
            // ARM64上NEON总是可用
            assert!(is_neon_supported());
        }
    }

    /// 测试空数组输入的断言检查
    #[test]
    fn test_empty_array_handling() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let mut c: Vec<f32> = vec![];

        // 空数组不应该panic，应该正常处理
        add_vector(&a, &b, &mut c);
        assert!(c.is_empty());
    }

    /// 测试向量乘法的非对齐长度
    #[test]
    fn test_mul_vector_non_aligned() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]; // 7个元素
        let b = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        let mut c = vec![0.0; 7];

        mul_vector(&a, &b, &mut c);

        for i in 0..7 {
            assert!((c[i] - (i as f32 + 1.0) * 2.0).abs() < 1e-5);
        }
    }

    /// 测试长度不匹配时应该触发断言失败（panic）
    #[test]
    #[should_panic]
    fn test_add_vector_length_mismatch_panic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0]; // 长度不同
        let mut c = vec![0.0; 3];

        add_vector(&a, &b, &mut c); // 应该panic
    }

    /// 测试乘法向量长度不匹配时应该触发断言失败
    #[test]
    #[should_panic]
    fn test_mul_vector_length_mismatch_panic() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0]; // 长度不同
        let mut c = vec![0.0; 2];

        mul_vector(&a, &b, &mut c); // 应该panic
    }

    /// 测试Softmax向量长度不匹配时应该触发断言失败
    #[test]
    #[should_panic]
    fn test_softmax_vector_length_mismatch_panic() {
        let x = vec![1.0, 2.0, 3.0];
        let mut out = vec![0.0; 2]; // 长度不同

        softmax_vector(&x, &mut out); // 应该panic
    }

    /// 测试Softmax负数值稳定性
    #[test]
    fn test_softmax_negative_values() {
        let x = vec![-1000.0, -1001.0, -999.0, -998.0];
        let mut out = vec![0.0; 4];

        // 不应panic或产生NaN
        softmax_vector(&x, &mut out);

        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // 最大值应有最高概率（-998.0是最大的）
        let max_idx = out
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        assert_eq!(max_idx, 3);
    }

    /// 测试Softmix极小值（接近零但非零）
    #[test]
    fn test_softmax_very_small_values() {
        use std::f32;

        let x = vec![
            f32::MIN_POSITIVE,
            2.0 * f32::MIN_POSITIVE,
            3.0 * f32::MIN_POSITIVE,
            4.0 * f32::MIN_POSITIVE,
        ];
        let mut out = vec![0.0; 4];

        softmax_vector(&x, &mut out);

        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // 所有值都应该大于0
        for val in &out {
            assert!(*val > 0.0, "Softmax输出不应有零或负值");
        }
    }

    /// 测试向量操作的幂等性（重复应用结果一致）
    #[test]
    fn test_vector_operations_idempotency() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        // 第一次加法
        let mut c1 = vec![0.0; 5];
        add_vector(&a, &b, &mut c1);

        // 第二次加法（相同输入）
        let mut c2 = vec![0.0; 5];
        add_vector(&a, &b, &mut c2);

        // 结果应该完全一致
        for i in 0..5 {
            assert!((c1[i] - c2[i]).abs() < 1e-10, "加法幂等性在索引{}处失败", i);
        }
    }

    /// 测试大规模向量的正确性和性能
    #[test]
    fn test_large_vector_operations() {
        let size = 10000;
        let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();
        let b: Vec<f32> = (0..size).map(|i| (size - i) as f32 * 0.002).collect();
        let mut c = vec![0.0; size];

        add_vector(&a, &b, &mut c);

        // 验证前几个和最后几个元素
        assert!((c[0] - (0.0 + (size as f32) * 0.002)).abs() < 1e-2);
        assert!((c[size - 1] - ((size - 1) as f32 * 0.001 + 0.0)).abs() < 1e-2);

        // 验证中间元素
        let mid = size / 2;
        assert!((c[mid] - (mid as f32 * 0.001 + (size - mid) as f32 * 0.002)).abs() < 1e-2);
    }

    /// 测试包含NaN和Infinity的输入处理
    #[test]
    fn test_nan_and_infinity_handling() {
        use std::f32;

        let a = vec![1.0, f32::NAN, 3.0, f32::INFINITY];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let mut c = vec![0.0; 4];

        // NaN传播
        add_vector(&a, &b, &mut c);
        assert!(c[1].is_nan(), "NaN应该在加法中传播");
        assert!(c[3].is_infinite(), "Infinity应该在加法中传播");

        // 乘法中的NaN和Infinity
        let mut d = vec![0.0; 4];
        mul_vector(&a, &b, &mut d);
        assert!(d[1].is_nan(), "NaN应该在乘法中传播");
        assert!(d[3].is_infinite(), "Infinity应该在乘法中传播");
    }

    /// 测试Softmax输出的概率分布特性
    #[test]
    fn test_softmax_probability_distribution() {
        // 测试softmax输出的基本概率性质

        // 性质1: 所有输出应在(0, 1)范围内
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut out = vec![0.0; 5];
        softmax_vector(&x, &mut out);

        for val in &out {
            assert!(
                *val > 0.0 && *val < 1.0,
                "Softmax概率应在(0,1)范围内，实际: {}",
                val
            );
        }

        // 性质2: 和应为1
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // 性质3: 输入越大，输出概率越高（单调性）
        for i in 1..out.len() {
            assert!(
                out[i] > out[i - 1],
                "Softmax应保持单调递增: out[{}]={} <= out[{}]={}",
                i - 1,
                out[i - 1],
                i,
                out[i]
            );
        }
    }

    /// 测试不同的SIMD边界长度（覆盖各种余数情况）
    #[test]
    fn test_various_alignment_boundaries() {
        // 测试各种不是SIMD宽度的倍数的长度
        // x86_64 AVX2: 8个float, NEON: 4个float
        test_alignment_boundary(1); // 最小长度
        test_alignment_boundary(2);
        test_alignment_boundary(3);
        test_alignment_boundary(7); // 8-1
        test_alignment_boundary(9); // 8+1
        test_alignment_boundary(15); // 16-1 (AVX512)
        test_alignment_boundary(17); // 16+1
        test_alignment_boundary(31); // 32-1
        test_alignment_boundary(33); // 32+1
    }

    fn test_alignment_boundary(len: usize) {
        let a: Vec<f32> = (0..len).map(|_i| _i as f32).collect();
        let b: Vec<f32> = (0..len).map(|_i| 1.0).collect();
        let mut c = vec![0.0; len];

        add_vector(&a, &b, &mut c);

        for i in 0..len {
            assert!(
                (c[i] - (i as f32 + 1.0)).abs() < 1e-5,
                "长度{}的边界测试在索引{}处失败",
                len,
                i
            );
        }
    }

    /// 测试向量累加的数值精度
    #[test]
    fn test_vector_accumulation_precision() {
        // 累加大量小数，测试浮点精度
        let n = 10000;
        let value = 0.0001f32;
        let a: Vec<f32> = vec![value; n];
        let b: Vec<f32> = vec![value; n];
        let mut c = vec![0.0; n];

        add_vector(&a, &b, &mut c);

        // 每个元素应该是 0.0002
        for &val in &c {
            assert!(
                (val - 2.0 * value).abs() < 1e-10,
                "累加精度误差过大: 期望 {}, 实际 {}",
                2.0 * value,
                val
            );
        }
    }
}
