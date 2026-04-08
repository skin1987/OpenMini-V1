//! NEON SIMD 后端实现
//!
//! 提供基于 ARM NEON 的 CPU 向量化加速

#![allow(dead_code)]

use anyhow::{bail, Result};
use rayon::prelude::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

const GEMM_BLOCK_SIZE: usize = 64;
const GEMV_PARALLEL_THRESHOLD: usize = 256;

/// NEON 后端
pub struct NeonBackend {
    has_neon: bool,
    num_threads: usize,
}

impl NeonBackend {
    pub fn new() -> Self {
        let has_neon = Self::check_neon();
        let num_threads = rayon::current_num_threads();
        Self { has_neon, num_threads }
    }

    fn check_neon() -> bool {
        #[cfg(target_arch = "aarch64")]
        {
            std::is_aarch64_feature_detected!("neon")
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            false
        }
    }

    pub fn is_available(&self) -> bool {
        self.has_neon
    }
    
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// 向量加法: z = a * x + y (使用 NEON)
    pub fn axpy(&self, alpha: f32, x: &[f32], y: &mut [f32]) -> Result<()> {
        if x.len() != y.len() {
            bail!("向量维度不匹配: x({}) y({})", x.len(), y.len());
        }

        if alpha == 0.0 {
            return Ok(());
        }

        let n = x.len();

        #[cfg(target_arch = "aarch64")]
        if self.has_neon {
            unsafe {
                let alpha_v = vdupq_n_f32(alpha);
                let chunks = n / 4;
                let remainder = n % 4;

                for i in 0..chunks {
                    let x_v = vld1q_f32(x.as_ptr().add(i * 4));
                    let y_v = vld1q_f32(y.as_ptr().add(i * 4));
                    let result = vmlaq_f32(y_v, alpha_v, x_v);
                    vst1q_f32(y.as_mut_ptr().add(i * 4), result);
                }

                for i in 0..remainder {
                    y[chunks * 4 + i] += alpha * x[chunks * 4 + i];
                }
            }
            return Ok(());
        }

        for i in 0..n {
            y[i] += alpha * x[i];
        }
        Ok(())
    }

    /// 向量点积 (使用 NEON)
    pub fn dot(&self, x: &[f32], y: &[f32]) -> Result<f32> {
        if x.len() != y.len() {
            bail!("向量维度不匹配: x({}) y({})", x.len(), y.len());
        }

        let n = x.len();

        #[cfg(target_arch = "aarch64")]
        if self.has_neon {
            unsafe {
                let mut sum_v = vdupq_n_f32(0.0);
                let chunks = n / 4;
                let remainder = n % 4;

                for i in 0..chunks {
                    let x_v = vld1q_f32(x.as_ptr().add(i * 4));
                    let y_v = vld1q_f32(y.as_ptr().add(i * 4));
                    sum_v = vmlaq_f32(sum_v, x_v, y_v);
                }

                let result = vgetq_lane_f32(sum_v, 0)
                    + vgetq_lane_f32(sum_v, 1)
                    + vgetq_lane_f32(sum_v, 2)
                    + vgetq_lane_f32(sum_v, 3);

                let mut sum = result;

                for i in 0..remainder {
                    sum += x[chunks * 4 + i] * y[chunks * 4 + i];
                }

                return Ok(sum);
            }
        }

        let sum: f32 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
        Ok(sum)
    }

    /// 向量缩放 (使用 NEON)
    pub fn scale(&self, alpha: f32, x: &mut [f32]) {
        if alpha == 1.0 {
            return;
        }

        let n = x.len();

        #[cfg(target_arch = "aarch64")]
        if self.has_neon {
            unsafe {
                let alpha_v = vdupq_n_f32(alpha);
                let chunks = n / 4;
                let remainder = n % 4;

                for i in 0..chunks {
                    let x_v = vld1q_f32(x.as_ptr().add(i * 4));
                    let result = vmulq_f32(alpha_v, x_v);
                    vst1q_f32(x.as_mut_ptr().add(i * 4), result);
                }

                for i in 0..remainder {
                    x[chunks * 4 + i] *= alpha;
                }
            }
            return;
        }

        x.iter_mut().for_each(|xi| *xi *= alpha);
    }

    /// 向量范数 (使用 NEON)
    pub fn nrm2(&self, x: &[f32]) -> f32 {
        let n = x.len();

        #[cfg(target_arch = "aarch64")]
        if self.has_neon {
            unsafe {
                let mut sum_v = vdupq_n_f32(0.0);
                let chunks = n / 4;
                let remainder = n % 4;

                for i in 0..chunks {
                    let x_v = vld1q_f32(x.as_ptr().add(i * 4));
                    sum_v = vmlaq_f32(sum_v, x_v, x_v);
                }

                let result = vgetq_lane_f32(sum_v, 0)
                    + vgetq_lane_f32(sum_v, 1)
                    + vgetq_lane_f32(sum_v, 2)
                    + vgetq_lane_f32(sum_v, 3);

                let mut sum = result;

                for i in 0..remainder {
                    sum += x[chunks * 4 + i] * x[chunks * 4 + i];
                }

                return sum.sqrt();
            }
        }

        let sum: f32 = x.iter().map(|&xi| xi * xi).sum();
        sum.sqrt()
    }
    
    /// 向量复制: y = x (使用 NEON)
    pub fn copy(&self, x: &[f32], y: &mut [f32]) -> Result<()> {
        if x.len() != y.len() {
            bail!("向量维度不匹配: x({}) y({})", x.len(), y.len());
        }
        
        let n = x.len();
        
        #[cfg(target_arch = "aarch64")]
        if self.has_neon {
            unsafe {
                let chunks = n / 4;
                let remainder = n % 4;
                
                for i in 0..chunks {
                    let x_v = vld1q_f32(x.as_ptr().add(i * 4));
                    vst1q_f32(y.as_mut_ptr().add(i * 4), x_v);
                }
                
                for i in 0..remainder {
                    y[chunks * 4 + i] = x[chunks * 4 + i];
                }
            }
            return Ok(());
        }
        
        y.copy_from_slice(x);
        Ok(())
    }
    
    /// 矩阵-向量乘法: y = alpha * A * x + beta * y (使用 NEON)
    pub fn gemv(
        &self,
        alpha: f32,
        a: &[f32],
        a_rows: usize,
        a_cols: usize,
        x: &[f32],
        beta: f32,
        y: &mut [f32],
    ) -> Result<()> {
        if a_rows != y.len() {
            bail!("矩阵-向量维度不匹配: A({}×{}) y({})", a_rows, a_cols, y.len());
        }
        if a_cols != x.len() {
            bail!("矩阵-向量维度不匹配: A({}×{}) x({})", a_rows, a_cols, x.len());
        }
        
        if beta == 0.0 {
            y.fill(0.0);
        } else if beta != 1.0 {
            self.scale(beta, y);
        }
        
        if alpha == 0.0 {
            return Ok(());
        }
        
        let m = a_rows;
        let n = a_cols;
        
        if m >= GEMV_PARALLEL_THRESHOLD {
            let alpha = alpha;
            let results: Vec<f32> = (0..m)
                .into_par_iter()
                .map(|i| {
                    let row = &a[i * n..(i + 1) * n];
                    Self::dot_row_neon(row, x, alpha)
                })
                .collect();
            
            for (i, &val) in results.iter().enumerate() {
                y[i] += val;
            }
        } else {
            #[cfg(target_arch = "aarch64")]
            if self.has_neon {
                unsafe {
                    for i in 0..m {
                        let row = &a[i * n..(i + 1) * n];
                        y[i] += alpha * Self::dot_neon_unsafe(row, x);
                    }
                }
                return Ok(());
            }
            
            for i in 0..m {
                let row = &a[i * n..(i + 1) * n];
                let mut sum = 0.0f32;
                for j in 0..n {
                    sum += row[j] * x[j];
                }
                y[i] += alpha * sum;
            }
        }
        
        Ok(())
    }
    
    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn dot_neon_unsafe(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let mut sum_v = vdupq_n_f32(0.0);
        let chunks = n / 4;
        let remainder = n % 4;
        
        for i in 0..chunks {
            let a_v = vld1q_f32(a.as_ptr().add(i * 4));
            let b_v = vld1q_f32(b.as_ptr().add(i * 4));
            sum_v = vmlaq_f32(sum_v, a_v, b_v);
        }
        
        let mut sum = vgetq_lane_f32(sum_v, 0)
            + vgetq_lane_f32(sum_v, 1)
            + vgetq_lane_f32(sum_v, 2)
            + vgetq_lane_f32(sum_v, 3);
        
        for i in 0..remainder {
            sum += a[chunks * 4 + i] * b[chunks * 4 + i];
        }
        
        sum
    }
    
    #[cfg(not(target_arch = "aarch64"))]
    #[inline]
    fn dot_neon_unsafe(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }
    
    #[cfg(target_arch = "aarch64")]
    #[inline]
    fn dot_row_neon(row: &[f32], x: &[f32], alpha: f32) -> f32 {
        unsafe {
            alpha * Self::dot_neon_unsafe(row, x)
        }
    }
    
    #[cfg(not(target_arch = "aarch64"))]
    #[inline]
    fn dot_row_neon(row: &[f32], x: &[f32], alpha: f32) -> f32 {
        alpha * row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum::<f32>()
    }
    
    /// 矩阵乘法: C = alpha * A * B + beta * C (使用 NEON + 分块)
    pub fn gemm(
        &self,
        alpha: f32,
        a: &[f32],
        a_rows: usize,
        a_cols: usize,
        b: &[f32],
        b_rows: usize,
        b_cols: usize,
        beta: f32,
        c: &mut [f32],
    ) -> Result<()> {
        let m = a_rows;
        let k = a_cols;
        let k2 = b_rows;
        let n = b_cols;
        
        if k != k2 {
            bail!("矩阵维度不匹配: A({}×{}) B({}×{})", m, k, k2, n);
        }
        if c.len() != m * n {
            bail!("输出矩阵大小不匹配: C({}) 期望({})", c.len(), m * n);
        }
        
        if beta == 0.0 {
            c.fill(0.0);
        } else if beta != 1.0 {
            self.scale(beta, c);
        }
        
        if alpha == 0.0 {
            return Ok(());
        }
        
        #[cfg(target_arch = "aarch64")]
        if self.has_neon {
            unsafe {
                self.gemm_neon_blocked(alpha, a, m, k, b, n, c);
            }
            return Ok(());
        }
        
        self.gemm_fallback(alpha, a, m, k, b, n, c);
        Ok(())
    }
    
    #[cfg(target_arch = "aarch64")]
    unsafe fn gemm_neon_blocked(
        &self,
        alpha: f32,
        a: &[f32],
        m: usize,
        k: usize,
        b: &[f32],
        n: usize,
        c: &mut [f32],
    ) {
        let block_size = GEMM_BLOCK_SIZE.min(m).min(n).min(k);
        
        if block_size == 0 {
            self.gemm_fallback(alpha, a, m, k, b, n, c);
            return;
        }
        
        for i_block in (0..m).step_by(block_size) {
            let i_end = (i_block + block_size).min(m);
            
            for j_block in (0..n).step_by(block_size) {
                let j_end = (j_block + block_size).min(n);
                
                for k_block in (0..k).step_by(block_size) {
                    let k_end = (k_block + block_size).min(k);
                    
                    for i in i_block..i_end {
                        for l in k_block..k_end {
                            let a_val = alpha * a[i * k + l];
                            let a_v = vdupq_n_f32(a_val);
                            
                            let mut j = j_block;
                            while j + 4 <= j_end {
                                let b_v = vld1q_f32(b.as_ptr().add(l * n + j));
                                let c_v = vld1q_f32(c.as_ptr().add(i * n + j));
                                let result = vmlaq_f32(c_v, a_v, b_v);
                                vst1q_f32(c.as_mut_ptr().add(i * n + j), result);
                                j += 4;
                            }
                            
                            for j_rem in j..j_end {
                                c[i * n + j_rem] += a_val * b[l * n + j_rem];
                            }
                        }
                    }
                }
            }
        }
    }
    
    #[cfg(not(target_arch = "aarch64"))]
    unsafe fn gemm_neon_blocked(
        &self,
        alpha: f32,
        a: &[f32],
        m: usize,
        k: usize,
        b: &[f32],
        n: usize,
        c: &mut [f32],
    ) {
        self.gemm_fallback(alpha, a, m, k, b, n, c);
    }
    
    fn gemm_fallback(
        &self,
        alpha: f32,
        a: &[f32],
        m: usize,
        k: usize,
        b: &[f32],
        n: usize,
        c: &mut [f32],
    ) {
        for i in 0..m {
            for l in 0..k {
                let a_val = alpha * a[i * k + l];
                for j in 0..n {
                    c[i * n + j] += a_val * b[l * n + j];
                }
            }
        }
    }
    
    /// Softmax: x_i = exp(x_i - max) / sum(exp(x_i - max)) (使用 NEON)
    pub fn softmax(&self, x: &mut [f32]) -> Result<()> {
        if x.is_empty() {
            return Ok(());
        }
        
        let n = x.len();
        
        let max_val = self.max_value(x);
        
        #[cfg(target_arch = "aarch64")]
        if self.has_neon {
            unsafe {
                let max_v = vdupq_n_f32(max_val);
                let chunks = n / 4;
                let remainder = n % 4;
                
                for i in 0..chunks {
                    let x_v = vld1q_f32(x.as_ptr().add(i * 4));
                    let sub_v = vsubq_f32(x_v, max_v);
                    vst1q_f32(x.as_mut_ptr().add(i * 4), sub_v);
                }
                
                for i in 0..remainder {
                    x[chunks * 4 + i] -= max_val;
                }
                
                for i in 0..chunks {
                    let x_v = vld1q_f32(x.as_ptr().add(i * 4));
                    let exp_v = Self::exp_neon(x_v);
                    vst1q_f32(x.as_mut_ptr().add(i * 4), exp_v);
                }
                
                for i in 0..remainder {
                    x[chunks * 4 + i] = (x[chunks * 4 + i]).exp();
                }
            }
        } else {
            for xi in x.iter_mut() {
                *xi = (*xi - max_val).exp();
            }
        }
        
        #[cfg(not(target_arch = "aarch64"))]
        for xi in x.iter_mut() {
            *xi = (*xi - max_val).exp();
        }
        
        let sum: f32 = if n > 1024 {
            x.par_iter().sum()
        } else {
            x.iter().sum()
        };
        
        if sum > 0.0 {
            self.scale(1.0 / sum, x);
        }
        
        Ok(())
    }
    
    fn max_value(&self, x: &[f32]) -> f32 {
        #[cfg(target_arch = "aarch64")]
        if self.has_neon && x.len() >= 4 {
            unsafe {
                let mut max_v = vld1q_f32(x.as_ptr());
                let chunks = x.len() / 4;
                
                for i in 1..chunks {
                    let x_v = vld1q_f32(x.as_ptr().add(i * 4));
                    max_v = vmaxq_f32(max_v, x_v);
                }
                
                let mut max = vgetq_lane_f32(max_v, 0)
                    .max(vgetq_lane_f32(max_v, 1))
                    .max(vgetq_lane_f32(max_v, 2))
                    .max(vgetq_lane_f32(max_v, 3));
                
                for i in (chunks * 4)..x.len() {
                    max = max.max(x[i]);
                }
                
                return max;
            }
        }
        
        x.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }
    
    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn exp_neon(x: float32x4_t) -> float32x4_t {
        let log2e = vdupq_n_f32(1.4426950408889634);
        let negln2hi = vdupq_n_f32(-0.693359375);
        let negln2lo = vdupq_n_f32(2.12194440e-4);
        let _one = vdupq_n_f32(1.0);
        let _half = vdupq_n_f32(0.5);
        
        let fx = vmulq_f32(x, log2e);
        let fx = vaddq_f32(fx, _half);
        let fx = vrndmq_f32(fx);
        
        let tmp = vmulq_f32(fx, negln2hi);
        let x = vaddq_f32(x, tmp);
        let tmp = vmulq_f32(fx, negln2lo);
        let x = vaddq_f32(x, tmp);
        
        let mut exp_x = vdupq_n_f32(1.0);
        let x2 = vmulq_f32(x, x);
        let x3 = vmulq_f32(x2, x);
        let x4 = vmulq_f32(x3, x);
        let x5 = vmulq_f32(x4, x);
        
        let c1 = vdupq_n_f32(1.0);
        let c2 = vdupq_n_f32(1.0 / 2.0);
        let c3 = vdupq_n_f32(1.0 / 6.0);
        let c4 = vdupq_n_f32(1.0 / 24.0);
        let c5 = vdupq_n_f32(1.0 / 120.0);
        
        exp_x = vmlaq_f32(exp_x, c5, x5);
        exp_x = vmlaq_f32(exp_x, c4, x4);
        exp_x = vmlaq_f32(exp_x, c3, x3);
        exp_x = vmlaq_f32(exp_x, c2, x2);
        exp_x = vmlaq_f32(exp_x, c1, x);
        
        let fx_int = vcvtq_s32_f32(fx);
        let fx_int = vaddq_s32(fx_int, vdupq_n_s32(127));
        let fx_int = vshlq_n_s32(fx_int, 23);
        let pow2n = vreinterpretq_f32_s32(fx_int);
        
        vmulq_f32(exp_x, pow2n)
    }
}

impl Default for NeonBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_backend_creation() {
        let backend = NeonBackend::new();
        println!("NEON: {}", backend.has_neon);
    }

    #[test]
    fn test_dot() {
        let backend = NeonBackend::new();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 1.0, 1.0, 1.0];
        let result = backend.dot(&x, &y).unwrap();
        assert!((result - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_scale() {
        let backend = NeonBackend::new();
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        backend.scale(2.0, &mut x);
        assert!((x[0] - 2.0).abs() < 1e-5);
        assert!((x[3] - 8.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_gemv() {
        let backend = NeonBackend::new();
        
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 2.0];
        let mut y = vec![0.0, 0.0, 0.0];
        
        backend.gemv(1.0, &a, 3, 2, &x, 0.0, &mut y).unwrap();
        
        assert!((y[0] - 5.0).abs() < 1e-5);
        assert!((y[1] - 11.0).abs() < 1e-5);
        assert!((y[2] - 17.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_gemm() {
        let backend = NeonBackend::new();
        
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 6];
        
        backend.gemm(1.0, &a, 2, 3, &b, 3, 2, 0.0, &mut c).unwrap();
        
        assert!((c[0] - 22.0).abs() < 1e-4);
        assert!((c[1] - 28.0).abs() < 1e-4);
        assert!((c[2] - 49.0).abs() < 1e-4);
        assert!((c[3] - 64.0).abs() < 1e-4);
    }
    
    #[test]
    fn test_softmax() {
        let backend = NeonBackend::new();
        
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        backend.softmax(&mut x).unwrap();
        
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        
        for &xi in &x {
            assert!(xi > 0.0 && xi < 1.0);
        }
    }
    
    #[test]
    fn test_nrm2() {
        let backend = NeonBackend::new();
        let x = vec![3.0, 4.0];
        let result = backend.nrm2(&x);
        assert!((result - 5.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_copy() {
        let backend = NeonBackend::new();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0; 4];
        backend.copy(&x, &mut y).unwrap();
        assert_eq!(x, y);
    }

    // 新增分支覆盖测试

    /// 测试 axpy 的 alpha=0 分支（不执行任何操作）
    #[test]
    fn test_axpy_zero_alpha() {
        let backend = NeonBackend::new();
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![4.0, 5.0, 6.0];
        let y_copy = y.clone();
        
        backend.axpy(0.0, &x, &mut y).unwrap();
        assert_eq!(y, y_copy, "alpha=0 时 y 不应改变");
    }

    /// 测试 axpy 的维度不匹配错误分支
    #[test]
    fn test_axpy_dimension_mismatch() {
        let backend = NeonBackend::new();
        let x = vec![1.0, 2.0];
        let mut y = vec![1.0, 2.0, 3.0];
        
        let result = backend.axpy(1.0, &x, &mut y);
        assert!(result.is_err(), "维度不匹配应返回错误");
    }

    /// 测试 dot 的维度不匹配错误分支
    #[test]
    fn test_dot_dimension_mismatch() {
        let backend = NeonBackend::new();
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0, 3.0];
        
        let result = backend.dot(&x, &y);
        assert!(result.is_err(), "维度不匹配应返回错误");
    }

    /// 测试 scale 的 alpha=1.0 分支（不执行任何操作）
    #[test]
    fn test_scale_alpha_one() {
        let backend = NeonBackend::new();
        let mut x = vec![1.0, 2.0, 3.0];
        let x_copy = x.clone();
        
        backend.scale(1.0, &mut x);
        assert_eq!(x, x_copy, "alpha=1.0 时 x 不应改变");
    }

    /// 测试 nrm2 的空向量分支
    #[test]
    fn test_nrm2_empty() {
        let backend = NeonBackend::new();
        let x: Vec<f32> = vec![];
        let result = backend.nrm2(&x);
        assert_eq!(result, 0.0, "空向量的范数应为 0");
    }

    /// 测试 copy 的维度不匹配错误分支
    #[test]
    fn test_copy_dimension_mismatch() {
        let backend = NeonBackend::new();
        let x = vec![1.0, 2.0];
        let mut y = vec![0.0; 3];
        
        let result = backend.copy(&x, &mut y);
        assert!(result.is_err(), "维度不匹配应返回错误");
    }

    /// 测试 gemv 的 alpha=0 分支（不执行矩阵-向量乘法）
    #[test]
    fn test_gemv_zero_alpha() {
        let backend = NeonBackend::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 2.0];
        let mut y = vec![10.0, 20.0, 30.0];
        let y_copy = y.clone();
        
        backend.gemv(0.0, &a, 3, 2, &x, 0.0, &mut y).unwrap();
        assert_eq!(y, y_copy, "alpha=0 时 y 不应改变");
    }

    /// 测试 gemv 的 beta=0 分支（清空 y）
    #[test]
    fn test_gemv_zero_beta() {
        let backend = NeonBackend::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 2.0];
        let mut y = vec![10.0, 20.0, 30.0];
        
        backend.gemv(0.0, &a, 3, 2, &x, 0.0, &mut y).unwrap();
        assert!(y.iter().all(|&v| v == 0.0), "beta=0 时 y 应被清空");
    }

    /// 测试 gemv 的 beta=1 分支（不缩放 y）
    #[test]
    fn test_gemv_beta_one() {
        let backend = NeonBackend::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 2.0];
        let mut y = vec![1.0, 1.0, 1.0];
        
        backend.gemv(1.0, &a, 3, 2, &x, 1.0, &mut y).unwrap();
        assert!((y[0] - 6.0).abs() < 1e-5, "y[0] 应为 1 + 5");
    }

    /// 测试 gemv 的维度不匹配错误分支
    #[test]
    fn test_gemv_dimension_mismatch() {
        let backend = NeonBackend::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];
        
        let result = backend.gemv(1.0, &a, 3, 2, &x, 0.0, &mut y);
        assert!(result.is_err(), "维度不匹配应返回错误");
    }

    /// 测试 gemm 的 alpha=0 分支（不执行矩阵乘法）
    #[test]
    fn test_gemm_zero_alpha() {
        let backend = NeonBackend::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut c = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let c_copy = c.clone();
        
        backend.gemm(0.0, &a, 2, 3, &b, 3, 2, 0.0, &mut c).unwrap();
        assert_eq!(c, c_copy, "alpha=0 时 c 不应改变");
    }

    /// 测试 gemm 的 beta=0 分支（清空 c）
    #[test]
    fn test_gemm_zero_beta() {
        let backend = NeonBackend::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut c = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        
        backend.gemm(0.0, &a, 2, 3, &b, 3, 2, 0.0, &mut c).unwrap();
        assert!(c.iter().all(|&v| v == 0.0), "beta=0 时 c 应被清空");
    }

    /// 测试 gemm 的维度不匹配错误分支
    #[test]
    fn test_gemm_dimension_mismatch() {
        let backend = NeonBackend::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 6];
        
        let result = backend.gemm(1.0, &a, 2, 3, &b, 4, 2, 0.0, &mut c);
        assert!(result.is_err(), "维度不匹配应返回错误");
    }

    /// 测试 gemm 的输出矩阵大小不匹配错误分支
    #[test]
    fn test_gemm_output_size_mismatch() {
        let backend = NeonBackend::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];
        
        let result = backend.gemm(1.0, &a, 2, 3, &b, 3, 2, 0.0, &mut c);
        assert!(result.is_err(), "输出矩阵大小不匹配应返回错误");
    }

    /// 测试 softmax 的空向量分支
    #[test]
    fn test_softmax_empty() {
        let backend = NeonBackend::new();
        let mut x: Vec<f32> = vec![];
        let result = backend.softmax(&mut x);
        assert!(result.is_ok(), "空向量应返回 Ok");
        assert!(x.is_empty(), "空向量应保持为空");
    }

    /// 测试 is_available 方法
    #[test]
    fn test_is_available() {
        let backend = NeonBackend::new();
        let available = backend.is_available();
        // 在非 ARM 架构上应返回 false
        #[cfg(not(target_arch = "aarch64"))]
        assert!(!available);
    }

    /// 测试 num_threads 方法
    #[test]
    fn test_num_threads() {
        let backend = NeonBackend::new();
        let threads = backend.num_threads();
        assert!(threads > 0, "线程数应大于 0");
    }

    /// 测试 Default trait 实现
    #[test]
    fn test_neon_backend_default() {
        let backend = NeonBackend::default();
        assert_eq!(backend.num_threads(), NeonBackend::new().num_threads());
    }
}
