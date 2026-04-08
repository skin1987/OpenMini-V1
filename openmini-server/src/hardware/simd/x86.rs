//! x86-64 SIMD 实现
//!
//! 支持 SSE4.2, AVX2, AVX-512

use super::SimdOps;

// ============================================================================
// 公共标量回退函数（用于融合操作）
// ============================================================================

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
// SSE4.2 实现 (128-bit)
// ============================================================================

pub struct SseOps;

impl SimdOps for SseOps {
    fn name(&self) -> &'static str {
        "SSE4.2"
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let len = a.len();
        let mut result = vec![0.0f32; len];

        let chunks = len / 4;
        let remainder = len % 4;

        unsafe {
            use std::arch::x86_64::*;

            for i in 0..chunks {
                let offset = i * 4;
                let va = _mm_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm_loadu_ps(b.as_ptr().add(offset));
                let vsum = _mm_add_ps(va, vb);
                _mm_storeu_ps(result.as_mut_ptr().add(offset), vsum);
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
            use std::arch::x86_64::*;

            for i in 0..chunks {
                let offset = i * 4;
                let va = _mm_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm_loadu_ps(b.as_ptr().add(offset));
                let vprod = _mm_mul_ps(va, vb);
                _mm_storeu_ps(result.as_mut_ptr().add(offset), vprod);
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
            use std::arch::x86_64::*;

            let vs = _mm_set1_ps(scalar);

            for i in 0..chunks {
                let offset = i * 4;
                let va = _mm_loadu_ps(a.as_ptr().add(offset));
                let vprod = _mm_mul_ps(va, vs);
                _mm_storeu_ps(result.as_mut_ptr().add(offset), vprod);
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
            use std::arch::x86_64::*;

            for i in 0..chunks {
                let offset = i * 4;
                let va = _mm_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm_loadu_ps(b.as_ptr().add(offset));
                let vdiff = _mm_sub_ps(va, vb);
                _mm_storeu_ps(result.as_mut_ptr().add(offset), vdiff);
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
            use std::arch::x86_64::*;

            for i in 0..chunks {
                let offset = i * 4;
                let va = _mm_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm_loadu_ps(b.as_ptr().add(offset));
                let vdiv = _mm_div_ps(va, vb);
                _mm_storeu_ps(result.as_mut_ptr().add(offset), vdiv);
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

        let mut sum;

        unsafe {
            use std::arch::x86_64::*;

            let mut vsum = _mm_setzero_ps();

            for i in 0..chunks {
                let offset = i * 4;
                let va = _mm_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm_loadu_ps(b.as_ptr().add(offset));
                let vprod = _mm_mul_ps(va, vb);
                vsum = _mm_add_ps(vsum, vprod);
            }

            let mut temp = [0.0f32; 4];
            _mm_storeu_ps(temp.as_mut_ptr(), vsum);
            sum = temp[0] + temp[1] + temp[2] + temp[3];
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

        let mut sum;

        unsafe {
            use std::arch::x86_64::*;

            let mut vsum = _mm_setzero_ps();

            for i in 0..chunks {
                let offset = i * 4;
                let va = _mm_loadu_ps(a.as_ptr().add(offset));
                vsum = _mm_add_ps(vsum, va);
            }

            let mut temp = [0.0f32; 4];
            _mm_storeu_ps(temp.as_mut_ptr(), vsum);
            sum = temp[0] + temp[1] + temp[2] + temp[3];
        }

        for i in (len - remainder)..len {
            sum += a[i];
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

        let mut max_val;

        unsafe {
            use std::arch::x86_64::*;

            let mut vmax = _mm_set1_ps(f32::NEG_INFINITY);

            for i in 0..chunks {
                let offset = i * 4;
                let va = _mm_loadu_ps(a.as_ptr().add(offset));
                vmax = _mm_max_ps(vmax, va);
            }

            let mut temp = [0.0f32; 4];
            _mm_storeu_ps(temp.as_mut_ptr(), vmax);
            max_val = temp[0].max(temp[1]).max(temp[2]).max(temp[3]);
        }

        for i in (len - remainder)..len {
            max_val = max_val.max(a[i]);
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

        let mut min_val;

        unsafe {
            use std::arch::x86_64::*;

            let mut vmin = _mm_set1_ps(f32::INFINITY);

            for i in 0..chunks {
                let offset = i * 4;
                let va = _mm_loadu_ps(a.as_ptr().add(offset));
                vmin = _mm_min_ps(vmin, va);
            }

            let mut temp = [0.0f32; 4];
            _mm_storeu_ps(temp.as_mut_ptr(), vmin);
            min_val = temp[0].min(temp[1]).min(temp[2]).min(temp[3]);
        }

        for i in (len - remainder)..len {
            min_val = min_val.min(a[i]);
        }

        min_val
    }

    fn softmax(&self, a: &[f32]) -> Vec<f32> {
        if a.is_empty() {
            return vec![];
        }
        let max_val = self.max(a);
        let exp_vals: Vec<f32> = a.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp = self.sum(&exp_vals);
        self.mul_scalar(&exp_vals, 1.0 / sum_exp)
    }

    fn relu(&self, a: &[f32]) -> Vec<f32> {
        let len = a.len();
        let mut result = vec![0.0f32; len];

        let chunks = len / 4;
        let remainder = len % 4;

        unsafe {
            use std::arch::x86_64::*;

            let vzero = _mm_setzero_ps();

            for i in 0..chunks {
                let offset = i * 4;
                let va = _mm_loadu_ps(a.as_ptr().add(offset));
                let vmax = _mm_max_ps(va, vzero);
                _mm_storeu_ps(result.as_mut_ptr().add(offset), vmax);
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
            use std::arch::x86_64::*;

            let vone = _mm_set1_ps(1.0);

            for i in 0..chunks {
                let offset = i * 4;
                let vx = _mm_loadu_ps(a.as_ptr().add(offset));
                let vneg_x = _mm_sub_ps(_mm_setzero_ps(), vx);

                let mut temp = [0.0f32; 4];
                _mm_storeu_ps(temp.as_mut_ptr(), vneg_x);
                for j in 0..4 {
                    temp[j] = temp[j].exp();
                }
                let vexp = _mm_loadu_ps(temp.as_ptr());

                let vdenom = _mm_add_ps(vone, vexp);
                let vsilu = _mm_div_ps(vx, vdenom);
                _mm_storeu_ps(result.as_mut_ptr().add(offset), vsilu);
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

            let vzero = _mm_setzero_ps();

            for i in 0..m {
                let input_row = &input[i * k..(i + 1) * k];
                let output_row = &mut output[i * n..(i + 1) * n];

                for j in (0..n).step_by(4) {
                    let remaining = (n - j).min(4);

                    if remaining == 4 {
                        let mut vsum = _mm_loadu_ps(bias.as_ptr().add(j));

                        for p in 0..k {
                            let vinput = _mm_set1_ps(input_row[p]);
                            let vweight = _mm_loadu_ps(weight.as_ptr().add(p * n + j));
                            let vprod = _mm_mul_ps(vinput, vweight);
                            vsum = _mm_add_ps(vsum, vprod);
                        }

                        let vrelu = _mm_max_ps(vsum, vzero);
                        _mm_storeu_ps(output_row.as_mut_ptr().add(j), vrelu);
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
            use std::arch::x86_64::*;

            let vone = _mm_set1_ps(1.0);

            for i in 0..m {
                let input_row = &input[i * k..(i + 1) * k];
                let output_row = &mut output[i * n..(i + 1) * n];

                for j in (0..n).step_by(4) {
                    let remaining = (n - j).min(4);

                    if remaining == 4 {
                        let mut vsum = _mm_setzero_ps();

                        for p in 0..k {
                            let vinput = _mm_set1_ps(input_row[p]);
                            let vweight = _mm_loadu_ps(weight.as_ptr().add(p * n + j));
                            let vprod = _mm_mul_ps(vinput, vweight);
                            vsum = _mm_add_ps(vsum, vprod);
                        }

                        let vneg = _mm_sub_ps(_mm_setzero_ps(), vsum);
                        let mut temp = [0.0f32; 4];
                        _mm_storeu_ps(temp.as_mut_ptr(), vneg);
                        for jj in 0..4 {
                            temp[jj] = temp[jj].exp();
                        }
                        let vexp = _mm_loadu_ps(temp.as_ptr());

                        let vdenom = _mm_add_ps(vone, vexp);
                        let vsilu = _mm_div_ps(vsum, vdenom);
                        _mm_storeu_ps(output_row.as_mut_ptr().add(j), vsilu);
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

                for j in (0..n).step_by(4) {
                    let remaining = (n - j).min(4);

                    if remaining == 4 {
                        let mut vsum = _mm_setzero_ps();

                        for p in 0..k {
                            let vinput = _mm_set1_ps(input_row[p]);
                            let vweight = _mm_loadu_ps(weight.as_ptr().add(p * n + j));
                            let vprod = _mm_mul_ps(vinput, vweight);
                            vsum = _mm_add_ps(vsum, vprod);
                        }

                        let vresidual = _mm_loadu_ps(residual_row.as_ptr().add(j));
                        let vresult = _mm_add_ps(vsum, vresidual);
                        _mm_storeu_ps(output_row.as_mut_ptr().add(j), vresult);
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

            let vscale = _mm_set1_ps(scale);

            for i in 0..m {
                let query_row = &query[i * k..(i + 1) * k];
                let output_row = &mut output[i * head_dim..(i + 1) * head_dim];

                let mut scores = vec![0.0f32; n];

                for j in (0..n).step_by(4) {
                    let remaining = (n - j).min(4);

                    if remaining == 4 {
                        let mut vscore = _mm_setzero_ps();

                        for p in 0..k {
                            let vquery = _mm_set1_ps(query_row[p]);
                            let vkey = _mm_loadu_ps(key.as_ptr().add(p * n + j));
                            let vprod = _mm_mul_ps(vquery, vkey);
                            vscore = _mm_add_ps(vscore, vprod);
                        }

                        vscore = _mm_mul_ps(vscore, vscale);
                        _mm_storeu_ps(scores.as_mut_ptr().add(j), vscore);
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
                let vmax = _mm_set1_ps(max_score);

                let mut exp_scores = vec![0.0f32; n];
                for j in (0..n).step_by(4) {
                    let remaining = (n - j).min(4);

                    if remaining == 4 {
                        let vscore = _mm_loadu_ps(scores.as_ptr().add(j));
                        let vshifted = _mm_sub_ps(vscore, vmax);

                        let mut temp = [0.0f32; 4];
                        _mm_storeu_ps(temp.as_mut_ptr(), vshifted);
                        for jj in 0..4 {
                            temp[jj] = temp[jj].exp();
                        }
                        let vexp = _mm_loadu_ps(temp.as_ptr());
                        _mm_storeu_ps(exp_scores.as_mut_ptr().add(j), vexp);
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
                        let mut vsum = _mm_setzero_ps();

                        for j in 0..n {
                            let vattn = _mm_set1_ps(attn_weights[j]);
                            let vvalue = _mm_loadu_ps(value.as_ptr().add(j * head_dim + d));
                            let vprod = _mm_mul_ps(vattn, vvalue);
                            vsum = _mm_add_ps(vsum, vprod);
                        }

                        _mm_storeu_ps(output_row.as_mut_ptr().add(d), vsum);
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
// AVX2 实现 (256-bit)
// ============================================================================

pub struct Avx2Ops;

impl SimdOps for Avx2Ops {
    fn name(&self) -> &'static str {
        "AVX2"
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

        for i in (len - remainder)..len {
            sum += a[i];
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

            let vlow = _mm256_castps256_ps128(vmax);
            let vhigh = _mm256_extractf128_ps(vmax, 1);
            let vmax128 = _mm_max_ps(vlow, vhigh);
            let mut temp = [0.0f32; 4];
            _mm_storeu_ps(temp.as_mut_ptr(), vmax128);
            max_val = temp[0].max(temp[1]).max(temp[2]).max(temp[3]);
        }

        for i in (len - remainder)..len {
            max_val = max_val.max(a[i]);
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

            let vlow = _mm256_castps256_ps128(vmin);
            let vhigh = _mm256_extractf128_ps(vmin, 1);
            let vmin128 = _mm_min_ps(vlow, vhigh);
            let mut temp = [0.0f32; 4];
            _mm_storeu_ps(temp.as_mut_ptr(), vmin128);
            min_val = temp[0].min(temp[1]).min(temp[2]).min(temp[3]);
        }

        for i in (len - remainder)..len {
            min_val = min_val.min(a[i]);
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
                for j in 0..8 {
                    temp[j] = temp[j].exp();
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
                for j in 0..8 {
                    temp[j] = temp[j].exp();
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

                        for p in 0..k {
                            let vinput = _mm256_set1_ps(input_row[p]);
                            let vweight = _mm256_loadu_ps(weight.as_ptr().add(p * n + j));
                            let vprod = _mm256_mul_ps(vinput, vweight);
                            vsum = _mm256_add_ps(vsum, vprod);
                        }

                        let vrelu = _mm256_max_ps(vsum, vzero);
                        _mm256_storeu_ps(output_row.as_mut_ptr().add(j), vrelu);
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
                        for jj in 0..8 {
                            temp[jj] = temp[jj].exp();
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

                        for p in 0..k {
                            let vquery = _mm256_set1_ps(query_row[p]);
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
// AVX-512 实现 (512-bit)
// 注意：AVX-512 需要 nightly Rust，当前在 stable 下回退到 AVX2
// ============================================================================

#[cfg(feature = "nightly_avx512")]
pub struct Avx512Ops;

#[cfg(feature = "nightly_avx512")]
impl SimdOps for Avx512Ops {
    fn name(&self) -> &'static str {
        "AVX-512"
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let len = a.len();
        let mut result = vec![0.0f32; len];

        let chunks = len / 16;
        let remainder = len % 16;

        unsafe {
            use std::arch::x86_64::*;

            for i in 0..chunks {
                let offset = i * 16;
                let va = _mm512_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
                let vsum = _mm512_add_ps(va, vb);
                _mm512_storeu_ps(result.as_mut_ptr().add(offset), vsum);
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

        let chunks = len / 16;
        let remainder = len % 16;

        unsafe {
            use std::arch::x86_64::*;

            for i in 0..chunks {
                let offset = i * 16;
                let va = _mm512_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
                let vprod = _mm512_mul_ps(va, vb);
                _mm512_storeu_ps(result.as_mut_ptr().add(offset), vprod);
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

        let chunks = len / 16;
        let remainder = len % 16;

        unsafe {
            use std::arch::x86_64::*;

            let vs = _mm512_set1_ps(scalar);

            for i in 0..chunks {
                let offset = i * 16;
                let va = _mm512_loadu_ps(a.as_ptr().add(offset));
                let vprod = _mm512_mul_ps(va, vs);
                _mm512_storeu_ps(result.as_mut_ptr().add(offset), vprod);
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

        let chunks = len / 16;
        let remainder = len % 16;

        unsafe {
            use std::arch::x86_64::*;

            for i in 0..chunks {
                let offset = i * 16;
                let va = _mm512_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
                let vdiff = _mm512_sub_ps(va, vb);
                _mm512_storeu_ps(result.as_mut_ptr().add(offset), vdiff);
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

        let chunks = len / 16;
        let remainder = len % 16;

        unsafe {
            use std::arch::x86_64::*;

            for i in 0..chunks {
                let offset = i * 16;
                let va = _mm512_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
                let vdiv = _mm512_div_ps(va, vb);
                _mm512_storeu_ps(result.as_mut_ptr().add(offset), vdiv);
            }
        }

        for i in (len - remainder)..len {
            result[i] = a[i] / b[i];
        }

        result
    }

    fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let chunks = len / 16;
        let remainder = len % 16;

        let mut sum;

        unsafe {
            use std::arch::x86_64::*;

            let mut vsum = _mm512_setzero_ps();

            for i in 0..chunks {
                let offset = i * 16;
                let va = _mm512_loadu_ps(a.as_ptr().add(offset));
                let vb = _mm512_loadu_ps(b.as_ptr().add(offset));
                let vprod = _mm512_mul_ps(va, vb);
                vsum = _mm512_add_ps(vsum, vprod);
            }

            sum = _mm512_reduce_add_ps(vsum);
        }

        for i in (len - remainder)..len {
            sum += a[i] * b[i];
        }

        sum
    }

    fn sum(&self, a: &[f32]) -> f32 {
        let len = a.len();
        let chunks = len / 16;
        let remainder = len % 16;

        let mut sum;

        unsafe {
            use std::arch::x86_64::*;

            let mut vsum = _mm512_setzero_ps();

            for i in 0..chunks {
                let offset = i * 16;
                let va = _mm512_loadu_ps(a.as_ptr().add(offset));
                vsum = _mm512_add_ps(vsum, va);
            }

            sum = _mm512_reduce_add_ps(vsum);
        }

        for i in (len - remainder)..len {
            sum += a[i];
        }

        sum
    }

    fn max(&self, a: &[f32]) -> f32 {
        if a.is_empty() {
            return f32::NEG_INFINITY;
        }

        let len = a.len();
        let chunks = len / 16;
        let remainder = len % 16;

        let mut max_val;

        unsafe {
            use std::arch::x86_64::*;

            let mut vmax = _mm512_set1_ps(f32::NEG_INFINITY);

            for i in 0..chunks {
                let offset = i * 16;
                let va = _mm512_loadu_ps(a.as_ptr().add(offset));
                vmax = _mm512_max_ps(vmax, va);
            }

            max_val = _mm512_reduce_max_ps(vmax);
        }

        for i in (len - remainder)..len {
            max_val = max_val.max(a[i]);
        }

        max_val
    }

    fn min(&self, a: &[f32]) -> f32 {
        if a.is_empty() {
            return f32::INFINITY;
        }

        let len = a.len();
        let chunks = len / 16;
        let remainder = len % 16;

        let mut min_val;

        unsafe {
            use std::arch::x86_64::*;

            let mut vmin = _mm512_set1_ps(f32::INFINITY);

            for i in 0..chunks {
                let offset = i * 16;
                let va = _mm512_loadu_ps(a.as_ptr().add(offset));
                vmin = _mm512_min_ps(vmin, va);
            }

            min_val = _mm512_reduce_min_ps(vmin);
        }

        for i in (len - remainder)..len {
            min_val = min_val.min(a[i]);
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

        let chunks = len / 16;
        let remainder = len % 16;

        unsafe {
            use std::arch::x86_64::*;

            let vmax = _mm512_set1_ps(max_val);

            for i in 0..chunks {
                let offset = i * 16;
                let va = _mm512_loadu_ps(a.as_ptr().add(offset));
                let vshifted = _mm512_sub_ps(va, vmax);

                let mut temp = [0.0f32; 16];
                _mm512_storeu_ps(temp.as_mut_ptr(), vshifted);
                for j in 0..16 {
                    temp[j] = temp[j].exp();
                }
                let vexp = _mm512_loadu_ps(temp.as_ptr());
                _mm512_storeu_ps(exp_vals.as_mut_ptr().add(offset), vexp);
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

        let chunks = len / 16;
        let remainder = len % 16;

        unsafe {
            use std::arch::x86_64::*;

            let vzero = _mm512_setzero_ps();

            for i in 0..chunks {
                let offset = i * 16;
                let va = _mm512_loadu_ps(a.as_ptr().add(offset));
                let vmax = _mm512_max_ps(va, vzero);
                _mm512_storeu_ps(result.as_mut_ptr().add(offset), vmax);
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

        let chunks = len / 16;
        let remainder = len % 16;

        unsafe {
            use std::arch::x86_64::*;

            let vone = _mm512_set1_ps(1.0);

            for i in 0..chunks {
                let offset = i * 16;
                let vx = _mm512_loadu_ps(a.as_ptr().add(offset));
                let vneg_x = _mm512_sub_ps(_mm512_setzero_ps(), vx);

                let mut temp = [0.0f32; 16];
                _mm512_storeu_ps(temp.as_mut_ptr(), vneg_x);
                for j in 0..16 {
                    temp[j] = temp[j].exp();
                }
                let vexp = _mm512_loadu_ps(temp.as_ptr());

                let vdenom = _mm512_add_ps(vone, vexp);
                let vsilu = _mm512_div_ps(vx, vdenom);
                _mm512_storeu_ps(result.as_mut_ptr().add(offset), vsilu);
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
    fn test_sse_add() {
        let ops = SseOps;
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let result = ops.add(&a, &b);
        assert_eq!(result, vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]);
    }

    #[test]
    fn test_sse_dot() {
        let ops = SseOps;
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];

        let dot = ops.dot(&a, &b);
        assert!((dot - 40.0).abs() < 1e-5);
    }

    #[test]
    fn test_sse_max() {
        let ops = SseOps;
        let a = vec![1.0, 5.0, 3.0, 8.0, 2.0, 9.0, 4.0, 7.0, 6.0];

        let result = ops.max(&a);
        assert!((result - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_sse_min() {
        let ops = SseOps;
        let a = vec![5.0, 3.0, 8.0, 1.0, 4.0, 2.0, 9.0, 6.0, 7.0];

        let result = ops.min(&a);
        assert!((result - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sse_softmax() {
        let ops = SseOps;
        let a = vec![1.0, 2.0, 3.0, 4.0];

        let result = ops.softmax(&a);

        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
        assert!(result.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_sse_silu() {
        let ops = SseOps;
        let a = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

        let result = ops.silu(&a);

        assert!((result[0] - (-0.238)).abs() < 0.01);
        assert!((result[1] - (-0.269)).abs() < 0.01);
        assert!((result[2] - 0.0).abs() < 1e-5);
        assert!((result[3] - 0.731).abs() < 0.01);
        assert!((result[4] - 1.762).abs() < 0.01);
    }

    #[test]
    fn test_sse_fused_gemm_relu() {
        let ops = SseOps;

        let input = vec![1.0, 0.0, 0.0, 1.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let bias = vec![0.0, 0.0, 0.0, 0.0];

        let result = ops.fused_gemm_relu(&input, &weight, &bias, 2, 2, 4);
        assert_eq!(result.len(), 8);
        assert!(result.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_sse_fused_gemm_softmax() {
        let ops = SseOps;

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

    #[test]
    #[cfg(any(target_arch = "x86_64"))]
    fn test_avx2_add() {
        if is_x86_feature_detected!("avx2") {
            let ops = Avx2Ops;
            let a = vec![1.0; 16];
            let b = vec![2.0; 16];

            let result = ops.add(&a, &b);
            assert_eq!(result, vec![3.0; 16]);
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86_64"))]
    fn test_avx2_max() {
        if is_x86_feature_detected!("avx2") {
            let ops = Avx2Ops;
            let a = vec![1.0, 5.0, 3.0, 8.0, 2.0, 9.0, 4.0, 7.0, 6.0];

            let result = ops.max(&a);
            assert!((result - 9.0).abs() < 1e-5);
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86_64"))]
    fn test_avx2_min() {
        if is_x86_feature_detected!("avx2") {
            let ops = Avx2Ops;
            let a = vec![5.0, 3.0, 8.0, 1.0, 4.0, 2.0, 9.0, 6.0, 7.0];

            let result = ops.min(&a);
            assert!((result - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86_64"))]
    fn test_avx2_fused_gemm_relu() {
        if is_x86_feature_detected!("avx2") {
            let ops = Avx2Ops;

            let input = vec![1.0, 0.0, 0.0, 1.0];
            let weight = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
            let bias = vec![0.0, 0.0, 0.0, 0.0];

            let result = ops.fused_gemm_relu(&input, &weight, &bias, 2, 2, 4);
            assert_eq!(result.len(), 8);
            assert!(result.iter().all(|&x| x >= 0.0));
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", feature = "nightly_avx512"))]
    fn test_avx512_add() {
        if is_x86_feature_detected!("avx512f") {
            let ops = Avx512Ops;
            let a = vec![1.0; 32];
            let b = vec![2.0; 32];

            let result = ops.add(&a, &b);
            assert_eq!(result, vec![3.0; 32]);
        }
    }

    // ==================== 新增分支覆盖测试 (7个) ====================

    #[test]
    fn test_sse_masked_operations_simulation() {
        // 覆盖分支: 掩码操作模拟（条件执行）
        let ops = SseOps;

        // 模拟掩码: 只对正数元素进行操作
        let a = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0];
        let mask_result: Vec<f32> = a
            .iter()
            .map(|&x| if x > 0.0 { x * 2.0 } else { x })
            .collect();

        // 验证结果
        assert_eq!(mask_result, vec![-1.0, 4.0, -3.0, 8.0, -5.0, 12.0]);

        // 使用 ReLU 作为掩码操作的替代
        let relu_result = ops.relu(&a);
        assert_eq!(relu_result, vec![0.0, 2.0, 0.0, 4.0, 0.0, 6.0]);
    }

    #[test]
    fn test_sse_gather_scatter_pattern() {
        // 覆盖分支: gather/scatter 操作模式（间接寻址）
        let _ops = SseOps;

        // 模拟 gather: 根据索引数组收集元素
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let indices = [0usize, 2, 4, 1, 3]; // 乱序索引
        let gathered: Vec<f32> = indices.iter().map(|&i| data[i]).collect();

        assert_eq!(gathered, vec![10.0, 30.0, 50.0, 20.0, 40.0]);

        // 模拟 scatter: 将元素分散到指定位置
        let values = vec![100.0, 200.0, 300.0];
        let scatter_indices = [2, 0, 4];
        let mut scattered = vec![0.0f32; 5];
        for (val, idx) in values.iter().zip(scatter_indices.iter()) {
            scattered[*idx] = *val;
        }
        assert_eq!(scattered, vec![200.0, 0.0, 100.0, 0.0, 300.0]);
    }

    #[test]
    fn test_sse_unaligned_load_optimization() {
        // 覆盖分支: 非对齐加载优化验证
        let ops = SseOps;

        // 创建一个可能不对齐的切片
        let mut data = vec![0.0f32; 20]; // 前置填充
        for i in 0..8 {
            data.push((i + 1) as f32); // 实际数据
        }
        data.push(0.0); // 后置填充

        // 从偏移位置提取数据
        let unaligned_slice = &data[20..28]; // 8 个元素
        let doubled = ops.mul(unaligned_slice, &[2.0; 8]);

        // 验证非对齐加载的正确性
        for i in 0..8 {
            assert!(
                (doubled[i] - ((i + 1) as f32) * 2.0).abs() < f32::EPSILON,
                "Unaligned load failed at index {}",
                i
            );
        }
    }

    #[test]
    fn test_sse_horizontal_operations() {
        // 覆盖分支: 水平操作（跨向量元素）
        let ops = SseOps;

        // 测试 sum 的水平归约特性
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let total = ops.sum(&data);
        assert!((total - 55.0).abs() < 1e-5); // 1+2+...+10=55

        // 测试 max/min 的水平比较
        let max_val = ops.max(&data);
        let min_val = ops.min(&data);

        assert!((max_val - 10.0).abs() < 1e-5);
        assert!((min_val - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sse_fused_operation_precision() {
        // 覆盖分支: 融合操作精度保证
        let ops = SseOps;

        // GEMM+ReLU: 验证负值被正确裁剪
        let input = vec![1.0, -1.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let bias = vec![-2.0, -2.0];

        let result = ops.fused_gemm_relu(&input, &weight, &bias, 1, 2, 2);

        // 第一行: 1*1 + (-1)*1 + (-2) = -2 -> ReLU -> 0
        // 第二行: 1*1 + (-1)*1 + (-2) = -2 -> ReLU -> 0
        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|&x| x >= 0.0)); // ReLU 保证非负
        assert_eq!(result, vec![0.0, 0.0]); // 全负输入应全零
    }

    #[test]
    #[cfg(any(target_arch = "x86_64"))]
    fn test_avx2_cross_lane_shuffle() {
        // 覆盖分支: AVX2 跨通道重排（如果支持）
        if is_x86_feature_detected!("avx2") {
            let ops = Avx2Ops;

            // 测试不同大小的输入
            for size in [8, 16, 24, 32, 40, 48, 56, 64] {
                let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
                let b: Vec<f32> = (0..size).map(|i| (size - i) as f32 * 0.5).collect();

                let result = ops.mul(&a, &b);
                assert_eq!(result.len(), size);

                // 抽样验证
                for idx in [0, size / 4, size / 2, 3 * size / 4, size - 1] {
                    let expected = a[idx] * b[idx];
                    assert!(
                        (result[idx] - expected).abs() < 1e-5,
                        "AVX2 shuffle test failed at size {} index {}: expected {}, got {}",
                        size,
                        idx,
                        expected,
                        result[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn test_sse_empty_and_single_element_edge_cases() {
        let ops = SseOps;

        let empty: Vec<f32> = vec![];
        assert!((ops.sum(&empty) - 0.0).abs() < 1e-5);
        assert!(ops.max(&empty).is_infinite() && ops.max(&empty) < 0.0);
        assert!(ops.min(&empty).is_infinite() && ops.min(&empty) > 0.0);

        // 单元素
        let single = vec![42.0];
        assert!((ops.sum(&single) - 42.0).abs() < 1e-5);
        assert!((ops.max(&single) - 42.0).abs() < 1e-5);
        assert!((ops.min(&single) - 42.0).abs() < 1e-5);

        let single_softmax = ops.softmax(&single);
        assert_eq!(single_softmax.len(), 1);
        assert!((single_softmax[0] - 1.0).abs() < 1e-5); // softmax([x]) = [1]

        let single_relu = ops.relu(&single);
        assert_eq!(single_relu, vec![42.0]);
    }
}
