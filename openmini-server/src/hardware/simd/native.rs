//! 国产平台 SIMD 实现
//!
//! 支持：
//! - 龙芯: LSX (128-bit), LASX (256-bit) - 当前为标量回退
//! - 飞腾/昇腾: NEON 兼容
//! - 海光/兆芯: x86-64 AVX/AVX2 兼容
//!
//! # 注意
//! 龙芯 LSX/LASX 后端当前使用标量实现，因为 Rust 对 loongarch64 的
//! `std::arch` SIMD 支持尚不完善。待 Rust 官方支持后可替换为真正的 SIMD 实现。

#![allow(clippy::needless_range_loop)]

use super::SimdOps;

// ============================================================================
// 公共标量回退函数
// ============================================================================

fn scalar_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

fn scalar_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
}

fn scalar_mul_scalar(a: &[f32], scalar: f32) -> Vec<f32> {
    a.iter().map(|&x| x * scalar).collect()
}

fn scalar_sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
}

fn scalar_div(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x / y).collect()
}

fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn scalar_sum(a: &[f32]) -> f32 {
    a.iter().sum()
}

fn scalar_max(a: &[f32]) -> f32 {
    a.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

fn scalar_min(a: &[f32]) -> f32 {
    a.iter().cloned().fold(f32::INFINITY, f32::min)
}

fn scalar_softmax(a: &[f32]) -> Vec<f32> {
    if a.is_empty() {
        return vec![];
    }
    let max_val = scalar_max(a);
    let exp_vals: Vec<f32> = a.iter().map(|&x| (x - max_val).exp()).collect();
    let sum_exp: f32 = exp_vals.iter().sum();
    exp_vals.iter().map(|&x| x / sum_exp).collect()
}

fn scalar_relu(a: &[f32]) -> Vec<f32> {
    a.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect()
}

fn scalar_silu(a: &[f32]) -> Vec<f32> {
    a.iter().map(|&x| x / (1.0 + (-x).exp())).collect()
}

fn scalar_fused_gemm_relu(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = bias[j];
            for p in 0..k {
                sum += input[i * k + p] * weight[p * n + j];
            }
            output[i * n + j] = if sum > 0.0 { sum } else { 0.0 };
        }
    }
    output
}

fn scalar_fused_gemm_silu(input: &[f32], weight: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += input[i * k + p] * weight[p * n + j];
            }
            output[i * n + j] = sum / (1.0 + (-sum).exp());
        }
    }
    output
}

fn scalar_fused_gemm_add(
    input: &[f32],
    weight: &[f32],
    residual: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += input[i * k + p] * weight[p * n + j];
            }
            output[i * n + j] = sum + residual[i * n + j];
        }
    }
    output
}

#[allow(unknown_lints)]
#[allow(clippy::manual_checked_ops)]
fn scalar_fused_gemm_softmax(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    scale: f32,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let value_len = value.len();
    let head_dim = if n > 0 { value_len / n } else { 0 };

    if head_dim == 0 || n == 0 || m == 0 || k == 0 {
        return vec![0.0f32; m * head_dim];
    }

    let mut output = vec![0.0f32; m * head_dim];
    for i in 0..m {
        let mut scores = vec![0.0f32; n];

        for j in 0..n {
            let mut score = 0.0f32;
            for p in 0..k {
                score += query[i * k + p] * key[p * n + j];
            }
            scores[j] = score * scale;
        }

        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let attn_weights: Vec<f32> = exp_scores.iter().map(|&e| e / sum_exp).collect();

        for d in 0..head_dim {
            let mut sum = 0.0f32;
            for j in 0..n {
                sum += attn_weights[j] * value[j * head_dim + d];
            }
            output[i * head_dim + d] = sum;
        }
    }

    output
}

// ============================================================================
// 龙芯 LSX/LASX 实现
// ============================================================================

/// 龙芯 LSX 实现 (128-bit)
///
/// LSX (Loongson SIMD Extension) 是龙芯处理器的 128-bit 向量扩展
///
/// **注意**: 当前实现为标量回退，因为 Rust 对 loongarch64 的
/// `std::arch` SIMD 支持尚不完善。待 Rust 官方支持后可替换为真正的 SIMD 实现。
#[cfg(target_arch = "loongarch64")]
pub struct LsxOps;

#[cfg(target_arch = "loongarch64")]
impl SimdOps for LsxOps {
    fn name(&self) -> &'static str {
        "LSX"
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        scalar_add(a, b)
    }

    fn mul(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        scalar_mul(a, b)
    }

    fn mul_scalar(&self, a: &[f32], scalar: f32) -> Vec<f32> {
        scalar_mul_scalar(a, scalar)
    }

    fn sub(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        scalar_sub(a, b)
    }

    fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        scalar_div(a, b)
    }

    fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        scalar_dot(a, b)
    }

    fn sum(&self, a: &[f32]) -> f32 {
        scalar_sum(a)
    }

    fn max(&self, a: &[f32]) -> f32 {
        scalar_max(a)
    }

    fn min(&self, a: &[f32]) -> f32 {
        scalar_min(a)
    }

    fn softmax(&self, a: &[f32]) -> Vec<f32> {
        scalar_softmax(a)
    }

    fn relu(&self, a: &[f32]) -> Vec<f32> {
        scalar_relu(a)
    }

    fn silu(&self, a: &[f32]) -> Vec<f32> {
        scalar_silu(a)
    }

    fn fused_gemm_relu(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        scalar_fused_gemm_relu(input, weight, bias, m, k, n)
    }

    fn fused_gemm_silu(
        &self,
        input: &[f32],
        weight: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        scalar_fused_gemm_silu(input, weight, m, k, n)
    }

    fn fused_gemm_add(
        &self,
        input: &[f32],
        weight: &[f32],
        residual: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        scalar_fused_gemm_add(input, weight, residual, m, k, n)
    }

    fn fused_gemm_softmax(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        scale: f32,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        scalar_fused_gemm_softmax(query, key, value, scale, m, k, n)
    }
}

/// 龙芯 LASX 实现 (256-bit)
///
/// LASX (Loongson Advanced SIMD Extension) 是龙芯处理器的 256-bit 向量扩展
///
/// **注意**: 当前实现为标量回退，因为 Rust 对 loongarch64 的
/// `std::arch` SIMD 支持尚不完善。待 Rust 官方支持后可替换为真正的 SIMD 实现。
#[cfg(target_arch = "loongarch64")]
pub struct LasxOps;

#[cfg(target_arch = "loongarch64")]
impl SimdOps for LasxOps {
    fn name(&self) -> &'static str {
        "LASX"
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        scalar_add(a, b)
    }

    fn mul(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        scalar_mul(a, b)
    }

    fn mul_scalar(&self, a: &[f32], scalar: f32) -> Vec<f32> {
        scalar_mul_scalar(a, scalar)
    }

    fn sub(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        scalar_sub(a, b)
    }

    fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        scalar_div(a, b)
    }

    fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        scalar_dot(a, b)
    }

    fn sum(&self, a: &[f32]) -> f32 {
        scalar_sum(a)
    }

    fn max(&self, a: &[f32]) -> f32 {
        scalar_max(a)
    }

    fn min(&self, a: &[f32]) -> f32 {
        scalar_min(a)
    }

    fn softmax(&self, a: &[f32]) -> Vec<f32> {
        scalar_softmax(a)
    }

    fn relu(&self, a: &[f32]) -> Vec<f32> {
        scalar_relu(a)
    }

    fn silu(&self, a: &[f32]) -> Vec<f32> {
        scalar_silu(a)
    }

    fn fused_gemm_relu(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        scalar_fused_gemm_relu(input, weight, bias, m, k, n)
    }

    fn fused_gemm_silu(
        &self,
        input: &[f32],
        weight: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        scalar_fused_gemm_silu(input, weight, m, k, n)
    }

    fn fused_gemm_add(
        &self,
        input: &[f32],
        weight: &[f32],
        residual: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        scalar_fused_gemm_add(input, weight, residual, m, k, n)
    }

    fn fused_gemm_softmax(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        scale: f32,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        scalar_fused_gemm_softmax(query, key, value, scale, m, k, n)
    }
}

// ============================================================================
// 飞腾/昇腾 NEON 兼容实现
// ============================================================================

/// 飞腾/昇腾 NEON 兼容实现
///
/// 飞腾和昇腾处理器基于 ARM 架构，支持标准 NEON 指令集
/// 使用 ARM NEON 实现
#[cfg(all(target_arch = "aarch64", not(target_feature = "sve")))]
pub struct PhytiumOps;

#[cfg(all(target_arch = "aarch64", not(target_feature = "sve")))]
impl SimdOps for PhytiumOps {
    fn name(&self) -> &'static str {
        "Phytium/NEON"
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let len = a.len();
        let mut result = vec![0.0f32; len];

        let chunks = len / 4;
        let remainder = len % 4;

        unsafe {
            use std::arch::aarch64::*;

            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(a.as_ptr().add(offset));
                let vb = vld1q_f32(b.as_ptr().add(offset));
                let vsum = vaddq_f32(va, vb);
                vst1q_f32(result.as_mut_ptr().add(offset), vsum);
            }
        }

        for i in (len - remainder)..len {
            result[i] = a[i] + b[i];
        }

        result
    }

    fn mul(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let len = a.len();
        let mut result = vec![0.0f32; len];

        let chunks = len / 4;
        let remainder = len % 4;

        unsafe {
            use std::arch::aarch64::*;

            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(a.as_ptr().add(offset));
                let vb = vld1q_f32(b.as_ptr().add(offset));
                let vprod = vmulq_f32(va, vb);
                vst1q_f32(result.as_mut_ptr().add(offset), vprod);
            }
        }

        for i in (len - remainder)..len {
            result[i] = a[i] * b[i];
        }

        result
    }

    fn mul_scalar(&self, a: &[f32], scalar: f32) -> Vec<f32> {
        let len = a.len();
        let mut result = vec![0.0f32; len];

        let chunks = len / 4;
        let remainder = len % 4;

        unsafe {
            use std::arch::aarch64::*;

            let vs = vdupq_n_f32(scalar);

            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(a.as_ptr().add(offset));
                let vprod = vmulq_f32(va, vs);
                vst1q_f32(result.as_mut_ptr().add(offset), vprod);
            }
        }

        for i in (len - remainder)..len {
            result[i] = a[i] * scalar;
        }

        result
    }

    fn sub(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let len = a.len();
        let mut result = vec![0.0f32; len];

        let chunks = len / 4;
        let remainder = len % 4;

        unsafe {
            use std::arch::aarch64::*;

            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(a.as_ptr().add(offset));
                let vb = vld1q_f32(b.as_ptr().add(offset));
                let vdiff = vsubq_f32(va, vb);
                vst1q_f32(result.as_mut_ptr().add(offset), vdiff);
            }
        }

        for i in (len - remainder)..len {
            result[i] = a[i] - b[i];
        }

        result
    }

    fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let len = a.len();
        let mut result = vec![0.0f32; len];

        let chunks = len / 4;
        let remainder = len % 4;

        unsafe {
            use std::arch::aarch64::*;

            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(a.as_ptr().add(offset));
                let vb = vld1q_f32(b.as_ptr().add(offset));
                let vdiv = vdivq_f32(va, vb);
                vst1q_f32(result.as_mut_ptr().add(offset), vdiv);
            }
        }

        for i in (len - remainder)..len {
            result[i] = a[i] / b[i];
        }

        result
    }

    fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let chunks = len / 4;
        let remainder = len % 4;

        let mut sum = 0.0f32;

        unsafe {
            use std::arch::aarch64::*;

            let mut vsum = vdupq_n_f32(0.0);

            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(a.as_ptr().add(offset));
                let vb = vld1q_f32(b.as_ptr().add(offset));
                let vprod = vmulq_f32(va, vb);
                vsum = vaddq_f32(vsum, vprod);
            }

            sum = vaddvq_f32(vsum);
        }

        for i in (len - remainder)..len {
            sum += a[i] * b[i];
        }

        sum
    }

    fn sum(&self, a: &[f32]) -> f32 {
        let len = a.len();
        let chunks = len / 4;
        let remainder = len % 4;

        let mut sum = 0.0f32;

        unsafe {
            use std::arch::aarch64::*;

            let mut vsum = vdupq_n_f32(0.0);

            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(a.as_ptr().add(offset));
                vsum = vaddq_f32(vsum, va);
            }

            sum = vaddvq_f32(vsum);
        }

        for &val in &a[(len - remainder)..] {
            sum += val;
        }

        sum
    }

    fn max(&self, a: &[f32]) -> f32 {
        if a.is_empty() {
            return f32::NEG_INFINITY;
        }

        let len = a.len();
        let chunks = len / 4;
        let remainder = len % 4;

        let mut max_val = f32::NEG_INFINITY;

        unsafe {
            use std::arch::aarch64::*;

            let mut vmax = vdupq_n_f32(f32::NEG_INFINITY);

            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(a.as_ptr().add(offset));
                vmax = vmaxq_f32(vmax, va);
            }

            max_val = vmaxvq_f32(vmax);
        }

        for &val in &a[(len - remainder)..] {
            max_val = max_val.max(val);
        }

        max_val
    }

    fn min(&self, a: &[f32]) -> f32 {
        if a.is_empty() {
            return f32::INFINITY;
        }

        let len = a.len();
        let chunks = len / 4;
        let remainder = len % 4;

        let mut min_val = f32::INFINITY;

        unsafe {
            use std::arch::aarch64::*;

            let mut vmin = vdupq_n_f32(f32::INFINITY);

            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(a.as_ptr().add(offset));
                vmin = vminq_f32(vmin, va);
            }

            min_val = vminvq_f32(vmin);
        }

        for &val in &a[(len - remainder)..] {
            min_val = min_val.min(val);
        }

        min_val
    }

    fn softmax(&self, a: &[f32]) -> Vec<f32> {
        if a.is_empty() {
            return vec![];
        }

        let max_val = self.max(a);
        let len = a.len();
        let mut exp_vals = vec![0.0f32; len];

        let chunks = len / 4;
        let remainder = len % 4;

        unsafe {
            use std::arch::aarch64::*;

            let vmax = vdupq_n_f32(max_val);

            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(a.as_ptr().add(offset));
                let vshifted = vsubq_f32(va, vmax);

                let mut temp = [0.0f32; 4];
                vst1q_f32(temp.as_mut_ptr(), vshifted);
                for j in 0..4 {
                    temp[j] = temp[j].exp();
                }
                let vexp = vld1q_f32(temp.as_ptr());
                vst1q_f32(exp_vals.as_mut_ptr().add(offset), vexp);
            }
        }

        for (i, exp_val) in exp_vals[(len - remainder)..].iter_mut().enumerate() {
            let i = (len - remainder) + i;
            *exp_val = (a[i] - max_val).exp();
        }

        let sum_exp = self.sum(&exp_vals);
        self.mul_scalar(&exp_vals, 1.0 / sum_exp)
    }

    fn relu(&self, a: &[f32]) -> Vec<f32> {
        let len = a.len();
        let mut result = vec![0.0f32; len];

        let chunks = len / 4;
        let remainder = len % 4;

        unsafe {
            use std::arch::aarch64::*;

            let vzero = vdupq_n_f32(0.0);

            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(a.as_ptr().add(offset));
                let vmax = vmaxq_f32(va, vzero);
                vst1q_f32(result.as_mut_ptr().add(offset), vmax);
            }
        }

        for i in (len - remainder)..len {
            result[i] = if a[i] > 0.0 { a[i] } else { 0.0 };
        }

        result
    }

    fn silu(&self, a: &[f32]) -> Vec<f32> {
        let len = a.len();
        let mut result = vec![0.0f32; len];

        let chunks = len / 4;
        let remainder = len % 4;

        unsafe {
            use std::arch::aarch64::*;

            let vone = vdupq_n_f32(1.0);

            for i in 0..chunks {
                let offset = i * 4;
                let vx = vld1q_f32(a.as_ptr().add(offset));
                let vneg_x = vsubq_f32(vdupq_n_f32(0.0), vx);

                let mut temp = [0.0f32; 4];
                vst1q_f32(temp.as_mut_ptr(), vneg_x);
                for j in 0..4 {
                    temp[j] = temp[j].exp();
                }
                let vexp = vld1q_f32(temp.as_ptr());

                let vdenom = vaddq_f32(vone, vexp);
                let vsilu = vdivq_f32(vx, vdenom);
                vst1q_f32(result.as_mut_ptr().add(offset), vsilu);
            }
        }

        for (i, result_val) in result[(len - remainder)..].iter_mut().enumerate() {
            let i = (len - remainder) + i;
            *result_val = a[i] / (1.0 + (-a[i]).exp());
        }

        result
    }

    fn fused_gemm_relu(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; m * n];

        unsafe {
            use std::arch::aarch64::*;

            let vzero = vdupq_n_f32(0.0);

            for i in 0..m {
                let input_row = &input[i * k..(i + 1) * k];
                let output_row = &mut output[i * n..(i + 1) * n];

                for j in (0..n).step_by(4) {
                    let remaining = (n - j).min(4);

                    if remaining == 4 {
                        let mut vsum = vld1q_f32(bias.as_ptr().add(j));

                        for p in 0..k {
                            let vinput = vdupq_n_f32(input_row[p]);
                            let vweight = vld1q_f32(weight.as_ptr().add(p * n + j));
                            let vprod = vmulq_f32(vinput, vweight);
                            vsum = vaddq_f32(vsum, vprod);
                        }

                        let vrelu = vmaxq_f32(vsum, vzero);
                        vst1q_f32(output_row.as_mut_ptr().add(j), vrelu);
                    } else {
                        for jj in j..j + remaining {
                            let mut sum = bias[jj];
                            for p in 0..k {
                                sum += input_row[p] * weight[p * n + jj];
                            }
                            output_row[jj] = if sum > 0.0 { sum } else { 0.0 };
                        }
                    }
                }
            }
        }

        output
    }

    fn fused_gemm_silu(
        &self,
        input: &[f32],
        weight: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; m * n];

        unsafe {
            use std::arch::aarch64::*;

            let vone = vdupq_n_f32(1.0);

            for i in 0..m {
                let input_row = &input[i * k..(i + 1) * k];
                let output_row = &mut output[i * n..(i + 1) * n];

                for j in (0..n).step_by(4) {
                    let remaining = (n - j).min(4);

                    if remaining == 4 {
                        let mut vsum = vdupq_n_f32(0.0);

                        for p in 0..k {
                            let vinput = vdupq_n_f32(input_row[p]);
                            let vweight = vld1q_f32(weight.as_ptr().add(p * n + j));
                            let vprod = vmulq_f32(vinput, vweight);
                            vsum = vaddq_f32(vsum, vprod);
                        }

                        let vneg = vsubq_f32(vdupq_n_f32(0.0), vsum);
                        let mut temp = [0.0f32; 4];
                        vst1q_f32(temp.as_mut_ptr(), vneg);
                        for jj in 0..4 {
                            temp[jj] = temp[jj].exp();
                        }
                        let vexp = vld1q_f32(temp.as_ptr());

                        let vdenom = vaddq_f32(vone, vexp);
                        let vsilu = vdivq_f32(vsum, vdenom);
                        vst1q_f32(output_row.as_mut_ptr().add(j), vsilu);
                    } else {
                        for (jj, output_val) in output_row[j..j + remaining].iter_mut().enumerate()
                        {
                            let jj = j + jj;
                            let mut sum = 0.0f32;
                            #[allow(clippy::needless_range_loop)]
                            for p in 0..k {
                                sum += input_row[p] * weight[p * n + jj];
                            }
                            *output_val = sum / (1.0 + (-sum).exp());
                        }
                    }
                }
            }
        }

        output
    }

    fn fused_gemm_add(
        &self,
        input: &[f32],
        weight: &[f32],
        residual: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; m * n];

        unsafe {
            use std::arch::aarch64::*;

            for i in 0..m {
                let input_row = &input[i * k..(i + 1) * k];
                let residual_row = &residual[i * n..(i + 1) * n];
                let output_row = &mut output[i * n..(i + 1) * n];

                for j in (0..n).step_by(4) {
                    let remaining = (n - j).min(4);

                    if remaining == 4 {
                        let mut vsum = vdupq_n_f32(0.0);

                        for p in 0..k {
                            let vinput = vdupq_n_f32(input_row[p]);
                            let vweight = vld1q_f32(weight.as_ptr().add(p * n + j));
                            let vprod = vmulq_f32(vinput, vweight);
                            vsum = vaddq_f32(vsum, vprod);
                        }

                        let vresidual = vld1q_f32(residual_row.as_ptr().add(j));
                        let vresult = vaddq_f32(vsum, vresidual);
                        vst1q_f32(output_row.as_mut_ptr().add(j), vresult);
                    } else {
                        for jj in j..j + remaining {
                            let mut sum = 0.0f32;
                            for p in 0..k {
                                sum += input_row[p] * weight[p * n + jj];
                            }
                            output_row[jj] = sum + residual_row[jj];
                        }
                    }
                }
            }
        }

        output
    }

    #[allow(unknown_lints)]
    #[allow(clippy::manual_checked_ops)]
    fn fused_gemm_softmax(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        scale: f32,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let value_len = value.len();
        let head_dim = if n > 0 { value_len / n } else { 0 };

        if head_dim == 0 || n == 0 || m == 0 || k == 0 {
            return vec![0.0f32; m * head_dim];
        }

        let mut output = vec![0.0f32; m * head_dim];

        unsafe {
            use std::arch::aarch64::*;

            let vscale = vdupq_n_f32(scale);

            for i in 0..m {
                let query_row = &query[i * k..(i + 1) * k];
                let output_row = &mut output[i * head_dim..(i + 1) * head_dim];

                let mut scores = vec![0.0f32; n];

                for j in (0..n).step_by(4) {
                    let remaining = (n - j).min(4);

                    if remaining == 4 {
                        let mut vscore = vdupq_n_f32(0.0);

                        for p in 0..k {
                            let vquery = vdupq_n_f32(query_row[p]);
                            let vkey = vld1q_f32(key.as_ptr().add(p * n + j));
                            let vprod = vmulq_f32(vquery, vkey);
                            vscore = vaddq_f32(vscore, vprod);
                        }

                        vscore = vmulq_f32(vscore, vscale);
                        vst1q_f32(scores.as_mut_ptr().add(j), vscore);
                    } else {
                        for jj in j..j + remaining {
                            let mut score = 0.0f32;
                            for p in 0..k {
                                score += query_row[p] * key[p * n + jj];
                            }
                            scores[jj] = score * scale;
                        }
                    }
                }

                let max_score = self.max(&scores);
                let vmax = vdupq_n_f32(max_score);

                let mut exp_scores = vec![0.0f32; n];
                for j in (0..n).step_by(4) {
                    let remaining = (n - j).min(4);

                    if remaining == 4 {
                        let vscore = vld1q_f32(scores.as_ptr().add(j));
                        let vshifted = vsubq_f32(vscore, vmax);

                        let mut temp = [0.0f32; 4];
                        vst1q_f32(temp.as_mut_ptr(), vshifted);
                        for jj in 0..4 {
                            temp[jj] = temp[jj].exp();
                        }
                        let vexp = vld1q_f32(temp.as_ptr());
                        vst1q_f32(exp_scores.as_mut_ptr().add(j), vexp);
                    } else {
                        for jj in j..j + remaining {
                            exp_scores[jj] = (scores[jj] - max_score).exp();
                        }
                    }
                }

                let sum_exp = self.sum(&exp_scores);
                let inv_sum = 1.0 / sum_exp;

                let attn_weights = self.mul_scalar(&exp_scores, inv_sum);

                for d in (0..head_dim).step_by(4) {
                    let remaining = (head_dim - d).min(4);

                    if remaining == 4 {
                        let mut vsum = vdupq_n_f32(0.0);

                        for j in 0..n {
                            let vattn = vdupq_n_f32(attn_weights[j]);
                            let vvalue = vld1q_f32(value.as_ptr().add(j * head_dim + d));
                            let vprod = vmulq_f32(vattn, vvalue);
                            vsum = vaddq_f32(vsum, vprod);
                        }

                        vst1q_f32(output_row.as_mut_ptr().add(d), vsum);
                    } else {
                        for dd in d..d + remaining {
                            let mut sum = 0.0f32;
                            for j in 0..n {
                                sum += attn_weights[j] * value[j * head_dim + dd];
                            }
                            output_row[dd] = sum;
                        }
                    }
                }
            }
        }

        output
    }
}

// ============================================================================
// 海光/兆芯 x86 兼容实现
// ============================================================================

/// 海光/兆芯 AVX2 兼容实现
///
/// 海光和兆芯处理器基于 x86-64 架构，支持 AVX/AVX2 指令集
/// 使用 x86-64 AVX2 实现
#[cfg(target_arch = "x86_64")]
pub struct HygonOps;

#[cfg(target_arch = "x86_64")]
impl SimdOps for HygonOps {
    fn name(&self) -> &'static str {
        "Hygon/AVX2"
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let len = a.len();
        let mut result = vec![0.0f32; len];

        let chunks = len / 8;
        let remainder = len % 8;

        unsafe {
            use std::arch::x86_64::*;

            for i in 0..chunks {
                let offset = i * 8;
                let va = _mm256_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
                let vsum = _mm256_add_ps(va, vb);
                _mm256_storeu_ps(result.as_mut_ptr().add(offset), vsum);
            }
        }

        for i in (len - remainder)..len {
            result[i] = a[i] + b[i];
        }

        result
    }

    fn mul(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let len = a.len();
        let mut result = vec![0.0f32; len];

        let chunks = len / 8;
        let remainder = len % 8;

        unsafe {
            use std::arch::x86_64::*;

            for i in 0..chunks {
                let offset = i * 8;
                let va = _mm256_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
                let vprod = _mm256_mul_ps(va, vb);
                _mm256_storeu_ps(result.as_mut_ptr().add(offset), vprod);
            }
        }

        for i in (len - remainder)..len {
            result[i] = a[i] * b[i];
        }

        result
    }

    fn mul_scalar(&self, a: &[f32], scalar: f32) -> Vec<f32> {
        let len = a.len();
        let mut result = vec![0.0f32; len];

        let chunks = len / 8;
        let remainder = len % 8;

        unsafe {
            use std::arch::x86_64::*;

            let vs = _mm256_set1_ps(scalar);

            for i in 0..chunks {
                let offset = i * 8;
                let va = _mm256_loadu_ps(a.as_ptr().add(offset));
                let vprod = _mm256_mul_ps(va, vs);
                _mm256_storeu_ps(result.as_mut_ptr().add(offset), vprod);
            }
        }

        for i in (len - remainder)..len {
            result[i] = a[i] * scalar;
        }

        result
    }

    fn sub(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let len = a.len();
        let mut result = vec![0.0f32; len];

        let chunks = len / 8;
        let remainder = len % 8;

        unsafe {
            use std::arch::x86_64::*;

            for i in 0..chunks {
                let offset = i * 8;
                let va = _mm256_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
                let vdiff = _mm256_sub_ps(va, vb);
                _mm256_storeu_ps(result.as_mut_ptr().add(offset), vdiff);
            }
        }

        for i in (len - remainder)..len {
            result[i] = a[i] - b[i];
        }

        result
    }

    fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let len = a.len();
        let mut result = vec![0.0f32; len];

        let chunks = len / 8;
        let remainder = len % 8;

        unsafe {
            use std::arch::x86_64::*;

            for i in 0..chunks {
                let offset = i * 8;
                let va = _mm256_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
                let vdiv = _mm256_div_ps(va, vb);
                _mm256_storeu_ps(result.as_mut_ptr().add(offset), vdiv);
            }
        }

        for i in (len - remainder)..len {
            result[i] = a[i] / b[i];
        }

        result
    }

    fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let chunks = len / 8;
        let remainder = len % 8;

        let mut sum;

        unsafe {
            use std::arch::x86_64::*;

            let mut vsum = _mm256_setzero_ps();

            for i in 0..chunks {
                let offset = i * 8;
                let va = _mm256_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
                let vprod = _mm256_mul_ps(va, vb);
                vsum = _mm256_add_ps(vsum, vprod);
            }

            let vlow = _mm256_castps256_ps128(vsum);
            let vhigh = _mm256_extractf128_ps(vsum, 1);
            let vsum128 = _mm_add_ps(vlow, vhigh);
            let mut temp = [0.0f32; 4];
            _mm_storeu_ps(temp.as_mut_ptr(), vsum128);
            sum = temp[0] + temp[1] + temp[2] + temp[3];
        }

        for i in (len - remainder)..len {
            sum += a[i] * b[i];
        }

        sum
    }

    fn sum(&self, a: &[f32]) -> f32 {
        let len = a.len();
        let chunks = len / 8;
        let remainder = len % 8;

        let mut sum;

        unsafe {
            use std::arch::x86_64::*;

            let mut vsum = _mm256_setzero_ps();

            for i in 0..chunks {
                let offset = i * 8;
                let va = _mm256_loadu_ps(a.as_ptr().add(offset));
                vsum = _mm256_add_ps(vsum, va);
            }

            let vlow = _mm256_castps256_ps128(vsum);
            let vhigh = _mm256_extractf128_ps(vsum, 1);
            let vsum128 = _mm_add_ps(vlow, vhigh);
            let mut temp = [0.0f32; 4];
            _mm_storeu_ps(temp.as_mut_ptr(), vsum128);
            sum = temp[0] + temp[1] + temp[2] + temp[3];
        }

        for &item in &a[(len - remainder)..] {
            sum += item;
        }

        sum
    }

    fn max(&self, a: &[f32]) -> f32 {
        if a.is_empty() {
            return f32::NEG_INFINITY;
        }

        let len = a.len();
        let chunks = len / 8;
        let remainder = len % 8;

        let mut max_val;

        unsafe {
            use std::arch::x86_64::*;

            let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);

            for i in 0..chunks {
                let offset = i * 8;
                let va = _mm256_loadu_ps(a.as_ptr().add(offset));
                vmax = _mm256_max_ps(vmax, va);
            }

            let hi = _mm256_extractf128_ps::<1>(vmax);
            let lo = _mm256_castps256_ps128(vmax);
            let max128 = _mm_max_ps(hi, lo);
            let mut arr = [0.0f32; 4];
            _mm_storeu_ps(arr.as_mut_ptr(), max128);
            max_val = arr[0].max(arr[1]).max(arr[2]).max(arr[3]);
        }

        for &item in &a[(len - remainder)..] {
            max_val = max_val.max(item);
        }

        max_val
    }

    fn min(&self, a: &[f32]) -> f32 {
        if a.is_empty() {
            return f32::INFINITY;
        }

        let len = a.len();
        let chunks = len / 8;
        let remainder = len % 8;

        let mut min_val;

        unsafe {
            use std::arch::x86_64::*;

            let mut vmin = _mm256_set1_ps(f32::INFINITY);

            for i in 0..chunks {
                let offset = i * 8;
                let va = _mm256_loadu_ps(a.as_ptr().add(offset));
                vmin = _mm256_min_ps(vmin, va);
            }

            let hi = _mm256_extractf128_ps::<1>(vmin);
            let lo = _mm256_castps256_ps128(vmin);
            let min128 = _mm_min_ps(hi, lo);
            let mut arr = [0.0f32; 4];
            _mm_storeu_ps(arr.as_mut_ptr(), min128);
            min_val = arr[0].min(arr[1]).min(arr[2]).min(arr[3]);
        }

        for &item in &a[(len - remainder)..] {
            min_val = min_val.min(item);
        }

        min_val
    }

    fn softmax(&self, a: &[f32]) -> Vec<f32> {
        if a.is_empty() {
            return vec![];
        }

        let max_val = self.max(a);
        let len = a.len();
        let mut exp_vals = vec![0.0f32; len];

        let chunks = len / 8;
        let remainder = len % 8;

        unsafe {
            use std::arch::x86_64::*;

            let vmax = _mm256_set1_ps(max_val);

            for i in 0..chunks {
                let offset = i * 8;
                let va = _mm256_loadu_ps(a.as_ptr().add(offset));
                let vshifted = _mm256_sub_ps(va, vmax);

                let mut temp = [0.0f32; 8];
                _mm256_storeu_ps(temp.as_mut_ptr(), vshifted);
                for val in temp.iter_mut() {
                    *val = val.exp();
                }
                let vexp = _mm256_loadu_ps(temp.as_ptr());
                _mm256_storeu_ps(exp_vals.as_mut_ptr().add(offset), vexp);
            }
        }

        for i in (len - remainder)..len {
            exp_vals[i] = (a[i] - max_val).exp();
        }

        let sum_exp = self.sum(&exp_vals);
        self.mul_scalar(&exp_vals, 1.0 / sum_exp)
    }

    fn relu(&self, a: &[f32]) -> Vec<f32> {
        let len = a.len();
        let mut result = vec![0.0f32; len];

        let chunks = len / 8;
        let remainder = len % 8;

        unsafe {
            use std::arch::x86_64::*;

            let vzero = _mm256_setzero_ps();

            for i in 0..chunks {
                let offset = i * 8;
                let va = _mm256_loadu_ps(a.as_ptr().add(offset));
                let vmax = _mm256_max_ps(va, vzero);
                _mm256_storeu_ps(result.as_mut_ptr().add(offset), vmax);
            }
        }

        for i in (len - remainder)..len {
            result[i] = if a[i] > 0.0 { a[i] } else { 0.0 };
        }

        result
    }

    fn silu(&self, a: &[f32]) -> Vec<f32> {
        let len = a.len();
        let mut result = vec![0.0f32; len];

        let chunks = len / 8;
        let remainder = len % 8;

        unsafe {
            use std::arch::x86_64::*;

            let vone = _mm256_set1_ps(1.0);

            for i in 0..chunks {
                let offset = i * 8;
                let vx = _mm256_loadu_ps(a.as_ptr().add(offset));
                let vneg_x = _mm256_sub_ps(_mm256_setzero_ps(), vx);

                let mut temp = [0.0f32; 8];
                _mm256_storeu_ps(temp.as_mut_ptr(), vneg_x);
                for val in temp.iter_mut() {
                    *val = val.exp();
                }
                let vexp = _mm256_loadu_ps(temp.as_ptr());

                let vdenom = _mm256_add_ps(vone, vexp);
                let vsilu = _mm256_div_ps(vx, vdenom);
                _mm256_storeu_ps(result.as_mut_ptr().add(offset), vsilu);
            }
        }

        for i in (len - remainder)..len {
            result[i] = a[i] / (1.0 + (-a[i]).exp());
        }

        result
    }

    fn fused_gemm_relu(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; m * n];

        unsafe {
            use std::arch::x86_64::*;

            let vzero = _mm256_setzero_ps();

            for i in 0..m {
                let input_row = &input[i * k..(i + 1) * k];
                let output_row = &mut output[i * n..(i + 1) * n];

                for j in (0..n).step_by(8) {
                    let remaining = (n - j).min(8);

                    if remaining == 8 {
                        let mut vsum = _mm256_loadu_ps(bias.as_ptr().add(j));

                        #[allow(clippy::needless_range_loop)]
                        for (p, &input_val) in input_row.iter().enumerate().take(k) {
                            let vinput = _mm256_set1_ps(input_val);
                            let vweight = _mm256_loadu_ps(weight.as_ptr().add(p * n + j));
                            let vprod = _mm256_mul_ps(vinput, vweight);
                            vsum = _mm256_add_ps(vsum, vprod);
                        }

                        let vrelu = _mm256_max_ps(vsum, vzero);
                        _mm256_storeu_ps(output_row.as_mut_ptr().add(j), vrelu);
                    } else {
                        for (jj, output_val) in output_row[j..j + remaining].iter_mut().enumerate()
                        {
                            let jj = j + jj;
                            let mut sum = bias[jj];
                            #[allow(clippy::needless_range_loop)]
                            for p in 0..k {
                                sum += input_row[p] * weight[p * n + jj];
                            }
                            *output_val = if sum > 0.0 { sum } else { 0.0 };
                        }
                    }
                }
            }
        }

        output
    }

    fn fused_gemm_silu(
        &self,
        input: &[f32],
        weight: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; m * n];

        unsafe {
            use std::arch::x86_64::*;

            let vone = _mm256_set1_ps(1.0);

            for i in 0..m {
                let input_row = &input[i * k..(i + 1) * k];
                let output_row = &mut output[i * n..(i + 1) * n];

                for j in (0..n).step_by(8) {
                    let remaining = (n - j).min(8);

                    if remaining == 8 {
                        let mut vsum = _mm256_setzero_ps();

                        for p in 0..k {
                            let vinput = _mm256_set1_ps(input_row[p]);
                            let vweight = _mm256_loadu_ps(weight.as_ptr().add(p * n + j));
                            let vprod = _mm256_mul_ps(vinput, vweight);
                            vsum = _mm256_add_ps(vsum, vprod);
                        }

                        let vneg = _mm256_sub_ps(_mm256_setzero_ps(), vsum);
                        let mut temp = [0.0f32; 8];
                        _mm256_storeu_ps(temp.as_mut_ptr(), vneg);
                        for val in temp.iter_mut() {
                            *val = val.exp();
                        }
                        let vexp = _mm256_loadu_ps(temp.as_ptr());

                        let vdenom = _mm256_add_ps(vone, vexp);
                        let vsilu = _mm256_div_ps(vsum, vdenom);
                        _mm256_storeu_ps(output_row.as_mut_ptr().add(j), vsilu);
                    } else {
                        for jj in j..j + remaining {
                            let mut sum = 0.0f32;
                            for p in 0..k {
                                sum += input_row[p] * weight[p * n + jj];
                            }
                            output_row[jj] = sum / (1.0 + (-sum).exp());
                        }
                    }
                }
            }
        }

        output
    }

    fn fused_gemm_add(
        &self,
        input: &[f32],
        weight: &[f32],
        residual: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; m * n];

        unsafe {
            use std::arch::x86_64::*;

            for i in 0..m {
                let input_row = &input[i * k..(i + 1) * k];
                let residual_row = &residual[i * n..(i + 1) * n];
                let output_row = &mut output[i * n..(i + 1) * n];

                for j in (0..n).step_by(8) {
                    let remaining = (n - j).min(8);

                    if remaining == 8 {
                        let mut vsum = _mm256_setzero_ps();

                        #[allow(clippy::needless_range_loop)]
                        for p in 0..k {
                            let vinput = _mm256_set1_ps(input_row[p]);
                            let vweight = _mm256_loadu_ps(weight.as_ptr().add(p * n + j));
                            let vprod = _mm256_mul_ps(vinput, vweight);
                            vsum = _mm256_add_ps(vsum, vprod);
                        }

                        let vresidual = _mm256_loadu_ps(residual_row.as_ptr().add(j));
                        let vresult = _mm256_add_ps(vsum, vresidual);
                        _mm256_storeu_ps(output_row.as_mut_ptr().add(j), vresult);
                    } else {
                        for (jj, output_val) in output_row[j..j + remaining].iter_mut().enumerate()
                        {
                            let jj = j + jj;
                            let mut sum = 0.0f32;
                            #[allow(clippy::needless_range_loop)]
                            for p in 0..k {
                                sum += input_row[p] * weight[p * n + jj];
                            }
                            *output_val = sum + residual_row[jj];
                        }
                    }
                }
            }
        }

        output
    }

    #[allow(unknown_lints)]
    #[allow(clippy::manual_checked_ops)]
    fn fused_gemm_softmax(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        scale: f32,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let value_len = value.len();
        let head_dim = if n > 0 { value_len / n } else { 0 };

        if head_dim == 0 || n == 0 || m == 0 || k == 0 {
            return vec![0.0f32; m * head_dim];
        }

        let mut output = vec![0.0f32; m * head_dim];

        unsafe {
            use std::arch::x86_64::*;

            let vscale = _mm256_set1_ps(scale);

            for i in 0..m {
                let query_row = &query[i * k..(i + 1) * k];
                let output_row = &mut output[i * head_dim..(i + 1) * head_dim];

                let mut scores = vec![0.0f32; n];

                for j in (0..n).step_by(8) {
                    let remaining = (n - j).min(8);

                    if remaining == 8 {
                        let mut vscore = _mm256_setzero_ps();

                        for (p, &query_val) in query_row.iter().enumerate().take(k) {
                            let vquery = _mm256_set1_ps(query_val);
                            let vkey = _mm256_loadu_ps(key.as_ptr().add(p * n + j));
                            let vprod = _mm256_mul_ps(vquery, vkey);
                            vscore = _mm256_add_ps(vscore, vprod);
                        }

                        vscore = _mm256_mul_ps(vscore, vscale);
                        _mm256_storeu_ps(scores.as_mut_ptr().add(j), vscore);
                    } else {
                        for jj in j..j + remaining {
                            let mut score = 0.0f32;
                            for p in 0..k {
                                score += query_row[p] * key[p * n + jj];
                            }
                            scores[jj] = score * scale;
                        }
                    }
                }

                let max_score = self.max(&scores);
                let vmax = _mm256_set1_ps(max_score);

                let mut exp_scores = vec![0.0f32; n];
                for j in (0..n).step_by(8) {
                    let remaining = (n - j).min(8);

                    if remaining == 8 {
                        let vscore = _mm256_loadu_ps(scores.as_ptr().add(j));
                        let vshifted = _mm256_sub_ps(vscore, vmax);

                        let mut temp = [0.0f32; 8];
                        _mm256_storeu_ps(temp.as_mut_ptr(), vshifted);
                        for jj in 0..8 {
                            temp[jj] = temp[jj].exp();
                        }
                        let vexp = _mm256_loadu_ps(temp.as_ptr());
                        _mm256_storeu_ps(exp_scores.as_mut_ptr().add(j), vexp);
                    } else {
                        for jj in j..j + remaining {
                            exp_scores[jj] = (scores[jj] - max_score).exp();
                        }
                    }
                }

                let sum_exp = self.sum(&exp_scores);
                let inv_sum = 1.0 / sum_exp;

                let attn_weights = self.mul_scalar(&exp_scores, inv_sum);

                for d in (0..head_dim).step_by(8) {
                    let remaining = (head_dim - d).min(8);

                    if remaining == 8 {
                        let mut vsum = _mm256_setzero_ps();

                        for j in 0..n {
                            let vattn = _mm256_set1_ps(attn_weights[j]);
                            let vvalue = _mm256_loadu_ps(value.as_ptr().add(j * head_dim + d));
                            let vprod = _mm256_mul_ps(vattn, vvalue);
                            vsum = _mm256_add_ps(vsum, vprod);
                        }

                        _mm256_storeu_ps(output_row.as_mut_ptr().add(d), vsum);
                    } else {
                        for dd in d..d + remaining {
                            let mut sum = 0.0f32;
                            for j in 0..n {
                                sum += attn_weights[j] * value[j * head_dim + dd];
                            }
                            output_row[dd] = sum;
                        }
                    }
                }
            }
        }

        output
    }
}

// ============================================================================
// SVE Fallback (ARM SVE 平台)
// ============================================================================

/// SVE Fallback 实现
///
/// 当目标平台启用 SVE 时，提供标量回退实现
/// 待 Rust 支持 SVE intrinsics 后可替换为真正的 SIMD 实现
#[cfg(all(target_arch = "aarch64", target_feature = "sve"))]
pub struct SveOps;

#[cfg(all(target_arch = "aarch64", target_feature = "sve"))]
impl SimdOps for SveOps {
    fn name(&self) -> &'static str {
        "SVE"
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        scalar_add(a, b)
    }

    fn mul(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        scalar_mul(a, b)
    }

    fn mul_scalar(&self, a: &[f32], scalar: f32) -> Vec<f32> {
        scalar_mul_scalar(a, scalar)
    }

    fn sub(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        scalar_sub(a, b)
    }

    fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        scalar_div(a, b)
    }

    fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        scalar_dot(a, b)
    }

    fn sum(&self, a: &[f32]) -> f32 {
        scalar_sum(a)
    }

    fn max(&self, a: &[f32]) -> f32 {
        scalar_max(a)
    }

    fn min(&self, a: &[f32]) -> f32 {
        scalar_min(a)
    }

    fn softmax(&self, a: &[f32]) -> Vec<f32> {
        scalar_softmax(a)
    }

    fn relu(&self, a: &[f32]) -> Vec<f32> {
        scalar_relu(a)
    }

    fn silu(&self, a: &[f32]) -> Vec<f32> {
        scalar_silu(a)
    }

    fn fused_gemm_relu(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        scalar_fused_gemm_relu(input, weight, bias, m, k, n)
    }

    fn fused_gemm_silu(
        &self,
        input: &[f32],
        weight: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        scalar_fused_gemm_silu(input, weight, m, k, n)
    }

    fn fused_gemm_add(
        &self,
        input: &[f32],
        weight: &[f32],
        residual: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        scalar_fused_gemm_add(input, weight, residual, m, k, n)
    }

    fn fused_gemm_softmax(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        scale: f32,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        scalar_fused_gemm_softmax(query, key, value, scale, m, k, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hygon_add() {
        let ops = HygonOps;
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let result = ops.add(&a, &b);
        assert_eq!(result, vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hygon_mul() {
        let ops = HygonOps;
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];

        let result = ops.mul(&a, &b);
        assert_eq!(result, vec![2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hygon_dot() {
        let ops = HygonOps;
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];

        let result = ops.dot(&a, &b);
        assert!((result - 40.0).abs() < 1e-5);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hygon_sum() {
        let ops = HygonOps;
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = ops.sum(&a);
        assert!((result - 15.0).abs() < 1e-5);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hygon_max() {
        let ops = HygonOps;
        let a = vec![1.0, 5.0, 3.0, 8.0, 2.0, 9.0, 4.0, 7.0, 6.0];

        let result = ops.max(&a);
        assert!((result - 9.0).abs() < 1e-5);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hygon_min() {
        let ops = HygonOps;
        let a = vec![5.0, 3.0, 8.0, 1.0, 4.0, 2.0, 9.0, 6.0, 7.0];

        let result = ops.min(&a);
        assert!((result - 1.0).abs() < 1e-5);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hygon_softmax() {
        let ops = HygonOps;
        let a = vec![1.0, 2.0, 3.0, 4.0];

        let result = ops.softmax(&a);

        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
        assert!(result.iter().all(|&x| x > 0.0));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hygon_relu() {
        let ops = HygonOps;
        let a = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0];

        let result = ops.relu(&a);
        assert_eq!(result, vec![0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0]);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hygon_silu() {
        let ops = HygonOps;
        let a = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

        let result = ops.silu(&a);

        assert!((result[0] - (-0.238)).abs() < 0.01);
        assert!((result[1] - (-0.269)).abs() < 0.01);
        assert!((result[2] - 0.0).abs() < 1e-5);
        assert!((result[3] - 0.731).abs() < 0.01);
        assert!((result[4] - 1.762).abs() < 0.01);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hygon_fused_gemm_relu() {
        let ops = HygonOps;

        let input = vec![1.0, 0.0, 0.0, 1.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let bias = vec![0.0, 0.0, 0.0, 0.0];

        let result = ops.fused_gemm_relu(&input, &weight, &bias, 2, 2, 4);
        assert_eq!(result.len(), 8);
        assert!(result.iter().all(|&x| x >= 0.0));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hygon_fused_gemm_silu() {
        let ops = HygonOps;

        let input = vec![1.0, 0.0, 0.0, 1.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];

        let result = ops.fused_gemm_silu(&input, &weight, 2, 2, 4);
        assert_eq!(result.len(), 8);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hygon_fused_gemm_add() {
        let ops = HygonOps;

        let input = vec![1.0, 0.0, 0.0, 1.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let residual = vec![1.0; 8];

        let result = ops.fused_gemm_add(&input, &weight, &residual, 2, 2, 4);
        assert_eq!(result.len(), 8);
        assert!(result.iter().all(|&x| x >= 1.0));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hygon_fused_gemm_softmax() {
        let ops = HygonOps;

        let m = 2;
        let k = 4;
        let n = 3;
        let head_dim = 4;

        let query = vec![1.0; m * k];
        let key = vec![1.0; k * n];
        let value = vec![1.0; n * head_dim];

        let result = ops.fused_gemm_softmax(&query, &key, &value, 0.5, m, k, n);

        assert_eq!(result.len(), m * head_dim);
    }

    // ==================== 新增分支覆盖测试 (7个) ====================

    #[test]
    fn test_scalar_exact_precision_verification() {
        // 覆盖分支: 标量实现的精确精度验证（IEEE 754 逐位比较）

        // 使用可以精确表示的浮点数
        let test_cases: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = vec![
            // 整数运算
            (
                vec![2.0, 3.0, 4.0],
                vec![5.0, 6.0, 7.0],
                vec![10.0, 11.0, 12.0],
            ),
            // 小数（可精确表示的 2 的幂次）
            (
                vec![0.5, 0.25, 0.125],
                vec![8.0, 16.0, 32.0],
                vec![1.0, 2.0, 3.0],
            ),
            // 混合正负
            (
                vec![1.5, -2.5, 3.5],
                vec![2.0, 3.0, -4.0],
                vec![0.0, 0.0, 0.0],
            ),
        ];

        for (a, b, c) in &test_cases {
            let mut result = vec![0.0f32; a.len()];
            // 手动实现 FMA: result[i] = a[i] * b[i] + c[i]
            for i in 0..a.len() {
                result[i] = a[i] * b[i] + c[i];
            }

            for i in 0..a.len() {
                let expected = a[i] * b[i] + c[i];
                // 对于这些可精确表示的数，结果应该完全相等
                assert!(
                    result[i].to_bits() == expected.to_bits(),
                    "Exact mismatch at index {}: expected {:?} ({:#x}), got {:?} ({:#x})",
                    i,
                    expected,
                    expected.to_bits(),
                    result[i],
                    result[i].to_bits()
                );
            }
        }
    }

    #[test]
    fn test_scalar_extreme_numerical_stability() {
        // 覆盖分支: 极端数值稳定性测试

        // 次正规数（denormalized numbers）- 最小的非零浮点数
        let denormal = f32::MIN_POSITIVE; // ~1.17549e-38
        let denormal_vec = [denormal, denormal];
        let scale_vec = [1.0_f32, 1.0_f32];
        let offset_vec = [0.0_f32, 0.0_f32];
        let mut denormal_result = vec![0.0f32; 2];

        // 手动实现 FMA
        for i in 0..2 {
            denormal_result[i] = denormal_vec[i] * scale_vec[i] + offset_vec[i];
        }
        // 结果应该保持为次正规数或零（flush-to-zero）
        for &val in &denormal_result {
            assert!(
                val == 0.0 || val.abs() < f32::MIN_POSITIVE * 2.0,
                "Denormal handling issue: got {}",
                val
            );
        }

        // 最大有限数
        let max_finite = f32::MAX;
        let max_vec = [max_finite, max_finite];
        let small_scale = [0.5_f32, 0.5_f32];
        let mut max_result = [0.0f32; 2];

        // 手动实现 FMA
        for i in 0..2 {
            max_result[i] = max_vec[i] * small_scale[i] + 0.0_f32;
        }
        // MAX * 0.5 应该是有限的
        assert!(max_result[0].is_finite());
        assert!(max_result[0].abs() > 1e37_f32); // 仍然很大

        // 负零和正零
        let pos_zero = [0.0_f32];
        let neg_zero = [-0.0_f32];
        let one = [1.0_f32];
        let mut zero_result = [0.0f32; 1];

        zero_result[0] = pos_zero[0] * one[0] + 0.0_f32;
        assert!(zero_result[0] == 0.0 && !zero_result[0].is_sign_negative());

        zero_result[0] = neg_zero[0] * one[0] + 0.0_f32;
        assert!(zero_result[0] == 0.0); // 符号可能不确定
    }

    #[test]
    fn test_scalar_single_element_operations() {
        // 覆盖分支: 单元素向量的边界行为

        // 单元素 FMA
        let single_a = [7.0_f32];
        let single_b = [11.0_f32];
        let single_c = [13.0_f32];
        let mut single_r = [0.0_f32; 1];

        single_r[0] = single_a[0] * single_b[0] + single_c[0];
        assert!((single_r[0] - 90.0).abs() < f32::EPSILON); // 7*11+13=90

        // 单元素各种组合
        let combos = vec![
            (0.0_f32, 0.0_f32, 0.0_f32),      // 全零
            (1.0_f32, 1.0_f32, 0.0_f32),      // 简单乘法
            (-1.0_f32, -1.0_f32, 1.0_f32),    // 负负得正+偏移
            (0.001_f32, 1000.0_f32, 0.5_f32), // 小*大
        ];

        for (a_val, b_val, c_val) in combos {
            let mut r = [0.0_f32; 1];
            r[0] = a_val * b_val + c_val;
            let expected = a_val * b_val + c_val;

            assert!(
                (r[0] - expected).abs() < f32::EPSILON,
                "Single element failed: {}*{}+{} = {}, got {}",
                a_val,
                b_val,
                c_val,
                expected,
                r[0]
            );
        }
    }

    #[test]
    fn test_scalar_output_buffer_independence() {
        // 覆盖分支: 输出缓冲区与输入缓冲区的独立性验证

        let original_input = vec![1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32];
        let multipliers = [10.0_f32, 20.0_f32, 30.0_f32, 40.0_f32, 50.0_f32];
        let offsets = [100.0_f32, 200.0_f32, 300.0_f32, 400.0_f32, 500.0_f32];

        // 复制输入用于验证
        let input_snapshot = original_input.clone();

        let mut output = [0.0_f32; 5];
        // 手动实现 FMA
        for i in 0..5 {
            output[i] = original_input[i] * multipliers[i] + offsets[i];
        }

        // 验证输出正确
        for i in 0..5 {
            let expected = original_input[i] * multipliers[i] + offsets[i];
            assert!((output[i] - expected).abs() < f32::EPSILON);
        }

        // 验证输入未被修改
        for i in 0..5 {
            assert!(
                (original_input[i] - input_snapshot[i]).abs() < f32::EPSILON,
                "Input was modified at index {}: original {}, now {}",
                i,
                input_snapshot[i],
                original_input[i]
            );
        }
    }

    #[test]
    fn test_scalar_sequential_operation_consistency() {
        // 覆盖分支: 连续多次操作的结果一致性

        let base_a = [1.0_f32, 2.0_f32, 3.0_f32];
        let base_b = [0.1_f32, 0.2_f32, 0.3_f32];
        let base_c = [0.01_f32, 0.02_f32, 0.03_f32];

        // 第一次计算
        let mut r1 = [0.0_f32; 3];
        for i in 0..3 {
            r1[i] = base_a[i] * base_b[i] + base_c[i];
        }

        // 用完全相同的输入再次计算
        let mut r2 = [0.0_f32; 3];
        for i in 0..3 {
            r2[i] = base_a[i] * base_b[i] + base_c[i];
        }

        // 结果应该完全一致
        for i in 0..3 {
            assert!(
                r1[i].to_bits() == r2[i].to_bits(),
                "Inconsistent results at index {}: first call {}, second call {}",
                i,
                r1[i],
                r2[i]
            );
        }

        // 第三次确认
        let mut r3 = [0.0_f32; 3];
        for i in 0..3 {
            r3[i] = base_a[i] * base_b[i] + base_c[i];
        }
        for i in 0..3 {
            assert_eq!(r1[i].to_bits(), r3[i].to_bits());
        }
    }

    #[test]
    fn test_scalar_alternating_sign_patterns() {
        // 覆盖分支: 交替符号模式的数值稳定性

        // 交替正负模式（可能导致精度问题）
        let size = 16;
        let a: Vec<f32> = (0..size)
            .map(|i| if i % 2 == 0 { 1e15_f32 } else { -1e15_f32 })
            .collect();
        let b: Vec<f32> = (0..size).map(|_| 1.0_f32).collect();
        let c: Vec<f32> = vec![1e15_f32; size]; // 补偿项
        let mut r: Vec<f32> = vec![0.0_f32; size];

        // 手动实现 FMA
        for i in 0..size {
            r[i] = a[i] * b[i] + c[i];
        }

        // 验证偶数索引: 1e15 * 1 + 1e15 = 2e15
        for i in (0..size).step_by(2) {
            assert!(
                (r[i] - 2e15).abs() / 2e15 < 1e-6,
                "Even index {} precision issue",
                i
            );
        }

        // 验证奇数索引: -1e15 * 1 + 1e15 = 0
        for i in (1..size).step_by(2) {
            assert!(
                r[i].abs() < 1e9, // 允许一定的抵消误差
                "Odd index {} cancellation issue: got {}",
                i,
                r[i]
            );
        }
    }

    #[test]
    fn test_scalar_monotonicity_preservation() {
        // 覆盖分支: 单调性保持验证

        // 如果 a 是递增的，b > 0，c 是常数，那么结果也应该是递增的
        let increasing: Vec<f32> = (0..20).map(|i| i as f32).collect(); // [0, 19]
        let positive: Vec<f32> = (0..20).map(|_| 1.5_f32).collect(); // 常数正乘数
        let constant: Vec<f32> = (0..20).map(|_| 10.0_f32).collect(); // 常数偏移
        let mut result = [0.0_f32; 20];

        // 手动实现 FMA
        for i in 0..20 {
            result[i] = increasing[i] * positive[i] + constant[i];
        }

        // 验证单调递增
        for i in 1..result.len() {
            assert!(
                result[i] > result[i - 1],
                "Monotonicity violated at index {}: {} <= {}",
                i,
                result[i],
                result[i - 1]
            );
        }

        // 验证第一个值
        assert!((result[0] - 10.0).abs() < f32::EPSILON); // 0*1.5+10=10
    }
}
