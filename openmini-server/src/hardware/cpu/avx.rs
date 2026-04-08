//! AVX SIMD 后端实现
//!
//! 提供基于 AVX/AVX2 的 CPU 向量化加速
//!
//! ## 特性
//! - AVX: 256-bit SIMD (8 floats)
//! - AVX2: 增强的 256-bit SIMD
//! - FMA: 融合乘加指令
//! - 自动检测并选择最优路径
//! - 可配置分块大小
//!
//! ## 安全性
//! - 运行时检测 CPU 特性，避免非法指令崩溃
//! - FMA 检测独立于 AVX，确保兼容性

#![allow(dead_code)]

use anyhow::{bail, Result};
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// 默认 GEMM 分块大小 (L1 缓存优化)
const DEFAULT_GEMM_BLOCK_SIZE: usize = 64;

/// GEMV 并行阈值
const GEMV_PARALLEL_THRESHOLD: usize = 256;

/// GEMM 配置
#[derive(Debug, Clone)]
pub struct GemmConfig {
    /// 分块大小
    pub block_size: usize,
    /// 是否使用并行
    pub parallel: bool,
    /// 并行阈值
    pub parallel_threshold: usize,
}

impl Default for GemmConfig {
    fn default() -> Self {
        Self {
            block_size: DEFAULT_GEMM_BLOCK_SIZE,
            parallel: true,
            parallel_threshold: 1024,
        }
    }
}

/// AVX 后端
pub struct AvxBackend {
    has_avx: bool,
    has_avx2: bool,
    has_fma: bool,
    num_threads: usize,
    gemm_config: GemmConfig,
}

impl AvxBackend {
    /// 创建新的 AVX 后端
    pub fn new() -> Self {
        let has_avx = Self::check_avx();
        let has_avx2 = Self::check_avx2();
        let has_fma = Self::check_fma();
        let num_threads = rayon::current_num_threads();
        
        Self {
            has_avx,
            has_avx2,
            has_fma,
            num_threads,
            gemm_config: GemmConfig::default(),
        }
    }
    
    /// 使用自定义配置创建后端
    pub fn with_config(config: GemmConfig) -> Self {
        let mut backend = Self::new();
        backend.gemm_config = config;
        backend
    }

    fn check_avx() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("avx")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    fn check_avx2() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }
    
    fn check_fma() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("fma")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    pub fn is_available(&self) -> bool {
        self.has_avx
    }

    pub fn has_avx2(&self) -> bool {
        self.has_avx2
    }
    
    pub fn has_fma(&self) -> bool {
        self.has_fma
    }
    
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
    
    /// 获取 SIMD 宽度 (floats)
    pub fn simd_width(&self) -> usize {
        if self.has_avx { 8 } else { 1 }
    }

    /// 向量加法: z = a * x + y
    pub fn axpy(&self, alpha: f32, x: &[f32], y: &mut [f32]) -> Result<()> {
        if x.len() != y.len() {
            bail!("向量维度不匹配: x({}) y({})", x.len(), y.len());
        }

        if alpha == 0.0 {
            return Ok(());
        }

        let n = x.len();

        #[cfg(target_arch = "x86_64")]
        if self.has_avx {
            return self.axpy_avx(alpha, x, y);
        }

        for i in 0..n {
            y[i] += alpha * x[i];
        }
        Ok(())
    }
    
    #[cfg(target_arch = "x86_64")]
    fn axpy_avx(&self, alpha: f32, x: &[f32], y: &mut [f32]) -> Result<()> {
        unsafe {
            let n = x.len();
            let alpha_v = _mm256_set1_ps(alpha);
            let chunks = n / 8;
            let remainder = n % 8;

            if self.has_fma {
                for i in 0..chunks {
                    let x_v = _mm256_loadu_ps(x.as_ptr().add(i * 8));
                    let y_v = _mm256_loadu_ps(y.as_ptr().add(i * 8));
                    let result = _mm256_fmadd_ps(alpha_v, x_v, y_v);
                    _mm256_storeu_ps(y.as_mut_ptr().add(i * 8), result);
                }
            } else {
                for i in 0..chunks {
                    let x_v = _mm256_loadu_ps(x.as_ptr().add(i * 8));
                    let y_v = _mm256_loadu_ps(y.as_ptr().add(i * 8));
                    let mul = _mm256_mul_ps(alpha_v, x_v);
                    let result = _mm256_add_ps(mul, y_v);
                    _mm256_storeu_ps(y.as_mut_ptr().add(i * 8), result);
                }
            }

            for i in 0..remainder {
                y[chunks * 8 + i] += alpha * x[chunks * 8 + i];
            }
        }
        Ok(())
    }

    /// 向量点积
    pub fn dot(&self, x: &[f32], y: &[f32]) -> Result<f32> {
        if x.len() != y.len() {
            bail!("向量维度不匹配: x({}) y({})", x.len(), y.len());
        }

        #[cfg(target_arch = "x86_64")]
        if self.has_avx {
            return self.dot_avx(x, y);
        }

        let sum: f32 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
        Ok(sum)
    }
    
    #[cfg(target_arch = "x86_64")]
    fn dot_avx(&self, x: &[f32], y: &[f32]) -> Result<f32> {
        unsafe {
            let n = x.len();
            let mut sum_v = _mm256_setzero_ps();
            let chunks = n / 8;
            let remainder = n % 8;

            if self.has_fma {
                for i in 0..chunks {
                    let x_v = _mm256_loadu_ps(x.as_ptr().add(i * 8));
                    let y_v = _mm256_loadu_ps(y.as_ptr().add(i * 8));
                    sum_v = _mm256_fmadd_ps(x_v, y_v, sum_v);
                }
            } else {
                for i in 0..chunks {
                    let x_v = _mm256_loadu_ps(x.as_ptr().add(i * 8));
                    let y_v = _mm256_loadu_ps(y.as_ptr().add(i * 8));
                    let mul = _mm256_mul_ps(x_v, y_v);
                    sum_v = _mm256_add_ps(sum_v, mul);
                }
            }

            let mut result = [0.0f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), sum_v);
            let mut sum: f32 = result.iter().sum();

            for i in 0..remainder {
                sum += x[chunks * 8 + i] * y[chunks * 8 + i];
            }

            Ok(sum)
        }
    }

    /// 向量缩放
    pub fn scale(&self, alpha: f32, x: &mut [f32]) {
        if alpha == 1.0 {
            return;
        }

        #[cfg(target_arch = "x86_64")]
        if self.has_avx {
            self.scale_avx(alpha, x);
            return;
        }

        x.iter_mut().for_each(|xi| *xi *= alpha);
    }
    
    #[cfg(target_arch = "x86_64")]
    fn scale_avx(&self, alpha: f32, x: &mut [f32]) {
        unsafe {
            let n = x.len();
            let alpha_v = _mm256_set1_ps(alpha);
            let chunks = n / 8;
            let remainder = n % 8;

            for i in 0..chunks {
                let x_v = _mm256_loadu_ps(x.as_ptr().add(i * 8));
                let result = _mm256_mul_ps(alpha_v, x_v);
                _mm256_storeu_ps(x.as_mut_ptr().add(i * 8), result);
            }

            for i in 0..remainder {
                x[chunks * 8 + i] *= alpha;
            }
        }
    }

    /// 向量范数
    pub fn nrm2(&self, x: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if self.has_avx {
            return self.nrm2_avx(x);
        }

        let sum: f32 = x.iter().map(|&xi| xi * xi).sum();
        sum.sqrt()
    }
    
    #[cfg(target_arch = "x86_64")]
    fn nrm2_avx(&self, x: &[f32]) -> f32 {
        unsafe {
            let n = x.len();
            let mut sum_v = _mm256_setzero_ps();
            let chunks = n / 8;
            let remainder = n % 8;

            if self.has_fma {
                for i in 0..chunks {
                    let x_v = _mm256_loadu_ps(x.as_ptr().add(i * 8));
                    sum_v = _mm256_fmadd_ps(x_v, x_v, sum_v);
                }
            } else {
                for i in 0..chunks {
                    let x_v = _mm256_loadu_ps(x.as_ptr().add(i * 8));
                    let mul = _mm256_mul_ps(x_v, x_v);
                    sum_v = _mm256_add_ps(sum_v, mul);
                }
            }

            let mut result = [0.0f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), sum_v);
            let mut sum: f32 = result.iter().sum();

            for i in 0..remainder {
                sum += x[chunks * 8 + i] * x[chunks * 8 + i];
            }

            sum.sqrt()
        }
    }
    
    /// 向量复制
    pub fn copy(&self, x: &[f32], y: &mut [f32]) -> Result<()> {
        if x.len() != y.len() {
            bail!("向量维度不匹配: x({}) y({})", x.len(), y.len());
        }
        
        #[cfg(target_arch = "x86_64")]
        if self.has_avx {
            self.copy_avx(x, y);
            return Ok(());
        }
        
        y.copy_from_slice(x);
        Ok(())
    }
    
    #[cfg(target_arch = "x86_64")]
    fn copy_avx(&self, x: &[f32], y: &mut [f32]) {
        unsafe {
            let n = x.len();
            let chunks = n / 8;
            let remainder = n % 8;
            
            for i in 0..chunks {
                let x_v = _mm256_loadu_ps(x.as_ptr().add(i * 8));
                _mm256_storeu_ps(y.as_mut_ptr().add(i * 8), x_v);
            }
            
            for i in 0..remainder {
                y[chunks * 8 + i] = x[chunks * 8 + i];
            }
        }
    }
    
    /// 矩阵-向量乘法
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
        
        let use_parallel = self.gemm_config.parallel && m >= self.gemm_config.parallel_threshold;
        
        if use_parallel {
            let has_fma = self.has_fma;
            let results: Vec<f32> = (0..m)
                .into_par_iter()
                .map(|i| {
                    let row = &a[i * n..(i + 1) * n];
                    unsafe { alpha * Self::dot_avx_unsafe(row, x, has_fma) }
                })
                .collect();
            
            for (i, &val) in results.iter().enumerate() {
                y[i] += val;
            }
        } else {
            #[cfg(target_arch = "x86_64")]
            if self.has_avx {
                unsafe {
                    for i in 0..m {
                        let row = &a[i * n..(i + 1) * n];
                        y[i] += alpha * Self::dot_avx_unsafe(row, x, self.has_fma);
                    }
                }
                return Ok(());
            }
            
            for i in 0..m {
                let row = &a[i * n..(i + 1) * n];
                y[i] += alpha * row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum::<f32>();
            }
        }
        
        Ok(())
    }
    
    #[cfg(target_arch = "x86_64")]
    #[inline]
    unsafe fn dot_avx_unsafe(a: &[f32], b: &[f32], has_fma: bool) -> f32 {
        let n = a.len();
        let mut sum_v = _mm256_setzero_ps();
        let chunks = n / 8;
        let remainder = n % 8;
        
        if has_fma {
            for i in 0..chunks {
                let a_v = _mm256_loadu_ps(a.as_ptr().add(i * 8));
                let b_v = _mm256_loadu_ps(b.as_ptr().add(i * 8));
                sum_v = _mm256_fmadd_ps(a_v, b_v, sum_v);
            }
        } else {
            for i in 0..chunks {
                let a_v = _mm256_loadu_ps(a.as_ptr().add(i * 8));
                let b_v = _mm256_loadu_ps(b.as_ptr().add(i * 8));
                let mul = _mm256_mul_ps(a_v, b_v);
                sum_v = _mm256_add_ps(sum_v, mul);
            }
        }
        
        let mut result = [0.0f32; 8];
        _mm256_storeu_ps(result.as_mut_ptr(), sum_v);
        let mut sum: f32 = result.iter().sum();
        
        for i in 0..remainder {
            sum += a[chunks * 8 + i] * b[chunks * 8 + i];
        }
        
        sum
    }
    
    /// 矩阵乘法
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
        
        #[cfg(target_arch = "x86_64")]
        if self.has_avx {
            unsafe {
                self.gemm_avx_blocked(alpha, a, m, k, b, n, c);
            }
            return Ok(());
        }
        
        self.gemm_fallback(alpha, a, m, k, b, n, c);
        Ok(())
    }
    
    #[cfg(target_arch = "x86_64")]
    unsafe fn gemm_avx_blocked(
        &self,
        alpha: f32,
        a: &[f32],
        m: usize,
        k: usize,
        b: &[f32],
        n: usize,
        c: &mut [f32],
    ) {
        let block_size = self.gemm_config.block_size.min(m).min(n).min(k);
        
        if block_size == 0 {
            self.gemm_fallback(alpha, a, m, k, b, n, c);
            return;
        }
        
        if self.has_fma {
            for i_block in (0..m).step_by(block_size) {
                let i_end = (i_block + block_size).min(m);
                
                for j_block in (0..n).step_by(block_size) {
                    let j_end = (j_block + block_size).min(n);
                    
                    for k_block in (0..k).step_by(block_size) {
                        let k_end = (k_block + block_size).min(k);
                        
                        for i in i_block..i_end {
                            for l in k_block..k_end {
                                let a_val = alpha * a[i * k + l];
                                let a_v = _mm256_set1_ps(a_val);
                                
                                let mut j = j_block;
                                while j + 8 <= j_end {
                                    let b_v = _mm256_loadu_ps(b.as_ptr().add(l * n + j));
                                    let c_v = _mm256_loadu_ps(c.as_ptr().add(i * n + j));
                                    let result = _mm256_fmadd_ps(a_v, b_v, c_v);
                                    _mm256_storeu_ps(c.as_mut_ptr().add(i * n + j), result);
                                    j += 8;
                                }
                                
                                for j_rem in j..j_end {
                                    c[i * n + j_rem] += a_val * b[l * n + j_rem];
                                }
                            }
                        }
                    }
                }
            }
        } else {
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
                                let a_v = _mm256_set1_ps(a_val);
                                
                                let mut j = j_block;
                                while j + 8 <= j_end {
                                    let b_v = _mm256_loadu_ps(b.as_ptr().add(l * n + j));
                                    let c_v = _mm256_loadu_ps(c.as_ptr().add(i * n + j));
                                    let mul = _mm256_mul_ps(a_v, b_v);
                                    let result = _mm256_add_ps(mul, c_v);
                                    _mm256_storeu_ps(c.as_mut_ptr().add(i * n + j), result);
                                    j += 8;
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
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    unsafe fn gemm_avx_blocked(
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
    
    /// Softmax
    pub fn softmax(&self, x: &mut [f32]) -> Result<()> {
        if x.is_empty() {
            return Ok(());
        }
        
        let n = x.len();
        
        let max_val = self.max_value(x);
        
        #[cfg(target_arch = "x86_64")]
        if self.has_avx {
            unsafe {
                self.softmax_avx(x, max_val);
            }
        } else {
            for xi in x.iter_mut() {
                *xi = (*xi - max_val).exp();
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        for xi in x.iter_mut() {
            *xi = (*xi - max_val).exp();
        }
        
        let sum: f32 = if self.gemm_config.parallel && n >= self.gemm_config.parallel_threshold {
            x.par_iter().sum()
        } else {
            x.iter().sum()
        };
        
        if sum > 0.0 {
            self.scale(1.0 / sum, x);
        }
        
        Ok(())
    }
    
    #[cfg(target_arch = "x86_64")]
    unsafe fn softmax_avx(&self, x: &mut [f32], max_val: f32) {
        let n = x.len();
        let max_v = _mm256_set1_ps(max_val);
        let chunks = n / 8;
        let remainder = n % 8;
        
        for i in 0..chunks {
            let x_v = _mm256_loadu_ps(x.as_ptr().add(i * 8));
            let sub_v = _mm256_sub_ps(x_v, max_v);
            _mm256_storeu_ps(x.as_mut_ptr().add(i * 8), sub_v);
        }
        
        for i in 0..remainder {
            x[chunks * 8 + i] -= max_val;
        }
        
        for i in 0..chunks {
            let x_v = _mm256_loadu_ps(x.as_ptr().add(i * 8));
            let exp_v = Self::exp_ps256(x_v, self.has_fma);
            _mm256_storeu_ps(x.as_mut_ptr().add(i * 8), exp_v);
        }
        
        for i in 0..remainder {
            x[chunks * 8 + i] = (x[chunks * 8 + i]).exp();
        }
    }
    
    fn max_value(&self, x: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if self.has_avx && x.len() >= 8 {
            unsafe {
                let mut max_v = _mm256_loadu_ps(x.as_ptr());
                let chunks = x.len() / 8;
                
                for i in 1..chunks {
                    let x_v = _mm256_loadu_ps(x.as_ptr().add(i * 8));
                    max_v = _mm256_max_ps(max_v, x_v);
                }
                
                let mut result = [0.0f32; 8];
                _mm256_storeu_ps(result.as_mut_ptr(), max_v);
                let mut max = result[0];
                for &val in &result[1..8] {
                    max = max.max(val);
                }
                
                for i in (chunks * 8)..x.len() {
                    max = max.max(x[i]);
                }
                
                return max;
            }
        }
        
        x.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }
    
    #[cfg(target_arch = "x86_64")]
    #[inline]
    unsafe fn exp_ps256(x: __m256, has_fma: bool) -> __m256 {
        let log2e = _mm256_set1_ps(1.4426950408889634);
        let negln2hi = _mm256_set1_ps(-0.693359375);
        let negln2lo = _mm256_set1_ps(2.12194440e-4);
        let half = _mm256_set1_ps(0.5);
        
        let fx = _mm256_mul_ps(x, log2e);
        let fx = _mm256_add_ps(fx, half);
        let fx = _mm256_floor_ps(fx);
        
        let tmp = _mm256_mul_ps(fx, negln2hi);
        let x = _mm256_add_ps(x, tmp);
        let tmp = _mm256_mul_ps(fx, negln2lo);
        let x = _mm256_add_ps(x, tmp);
        
        let mut exp_x = _mm256_set1_ps(1.0);
        let x2 = _mm256_mul_ps(x, x);
        let x3 = _mm256_mul_ps(x2, x);
        let x4 = _mm256_mul_ps(x3, x);
        let x5 = _mm256_mul_ps(x4, x);
        
        let c1 = _mm256_set1_ps(1.0);
        let c2 = _mm256_set1_ps(1.0 / 2.0);
        let c3 = _mm256_set1_ps(1.0 / 6.0);
        let c4 = _mm256_set1_ps(1.0 / 24.0);
        let c5 = _mm256_set1_ps(1.0 / 120.0);
        
        if has_fma {
            exp_x = _mm256_fmadd_ps(c5, x5, exp_x);
            exp_x = _mm256_fmadd_ps(c4, x4, exp_x);
            exp_x = _mm256_fmadd_ps(c3, x3, exp_x);
            exp_x = _mm256_fmadd_ps(c2, x2, exp_x);
            exp_x = _mm256_fmadd_ps(c1, x, exp_x);
        } else {
            let mut tmp = _mm256_mul_ps(c5, x5);
            exp_x = _mm256_add_ps(exp_x, tmp);
            tmp = _mm256_mul_ps(c4, x4);
            exp_x = _mm256_add_ps(exp_x, tmp);
            tmp = _mm256_mul_ps(c3, x3);
            exp_x = _mm256_add_ps(exp_x, tmp);
            tmp = _mm256_mul_ps(c2, x2);
            exp_x = _mm256_add_ps(exp_x, tmp);
            tmp = _mm256_mul_ps(c1, x);
            exp_x = _mm256_add_ps(exp_x, tmp);
        }
        
        let fx_int = _mm256_cvttps_epi32(fx);
        let fx_int = _mm256_add_epi32(fx_int, _mm256_set1_epi32(127));
        let fx_int = _mm256_slli_epi32(fx_int, 23);
        let pow2n = _mm256_castsi256_ps(fx_int);
        
        _mm256_mul_ps(exp_x, pow2n)
    }
}

impl Default for AvxBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx_backend_creation() {
        let backend = AvxBackend::new();
        println!("AVX: {}, AVX2: {}, FMA: {}", 
            backend.has_avx, backend.has_avx2, backend.has_fma);
        println!("SIMD width: {}", backend.simd_width());
    }

    #[test]
    fn test_dot() {
        let backend = AvxBackend::new();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let result = backend.dot(&x, &y).unwrap();
        assert!((result - 36.0).abs() < 1e-5);
    }

    #[test]
    fn test_scale() {
        let backend = AvxBackend::new();
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        backend.scale(2.0, &mut x);
        assert!((x[0] - 2.0).abs() < 1e-5);
        assert!((x[7] - 16.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_gemv() {
        let backend = AvxBackend::new();
        
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
        let backend = AvxBackend::new();
        
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];
        
        backend.gemm(1.0, &a, 2, 3, &b, 3, 2, 0.0, &mut c).unwrap();
        
        assert!((c[0] - 22.0).abs() < 1e-4);
        assert!((c[1] - 28.0).abs() < 1e-4);
        assert!((c[2] - 49.0).abs() < 1e-4);
        assert!((c[3] - 64.0).abs() < 1e-4);
    }
    
    #[test]
    fn test_softmax() {
        let backend = AvxBackend::new();
        
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        backend.softmax(&mut x).unwrap();
        
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        
        for &xi in &x {
            assert!(xi > 0.0 && xi < 1.0);
        }
    }
    
    #[test]
    fn test_nrm2() {
        let backend = AvxBackend::new();
        let x = vec![3.0, 4.0];
        let result = backend.nrm2(&x);
        assert!((result - 5.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_copy() {
        let backend = AvxBackend::new();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut y = vec![0.0; 8];
        backend.copy(&x, &mut y).unwrap();
        assert_eq!(x, y);
    }
    
    #[test]
    fn test_gemm_config() {
        let config = GemmConfig {
            block_size: 32,
            parallel: false,
            parallel_threshold: 512,
        };
        let backend = AvxBackend::with_config(config);
        assert_eq!(backend.gemm_config.block_size, 32);
        assert!(!backend.gemm_config.parallel);
    }
    
    #[test]
    fn test_axpy() {
        let backend = AvxBackend::new();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut y = vec![1.0; 8];
        backend.axpy(2.0, &x, &mut y).unwrap();
        
        for i in 0..8 {
            assert!((y[i] - (1.0 + 2.0 * (i + 1) as f32)).abs() < 1e-5);
        }
    }

    // 新增分支覆盖测试

    /// 测试 axpy 的 alpha=0 分支（不执行任何操作）
    #[test]
    fn test_axpy_zero_alpha() {
        let backend = AvxBackend::new();
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![4.0, 5.0, 6.0];
        let y_copy = y.clone();
        
        backend.axpy(0.0, &x, &mut y).unwrap();
        assert_eq!(y, y_copy, "alpha=0 时 y 不应改变");
    }

    /// 测试 axpy 的维度不匹配错误分支
    #[test]
    fn test_axpy_dimension_mismatch() {
        let backend = AvxBackend::new();
        let x = vec![1.0, 2.0];
        let mut y = vec![1.0, 2.0, 3.0];
        
        let result = backend.axpy(1.0, &x, &mut y);
        assert!(result.is_err(), "维度不匹配应返回错误");
    }

    /// 测试 dot 的维度不匹配错误分支
    #[test]
    fn test_dot_dimension_mismatch() {
        let backend = AvxBackend::new();
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0, 3.0];
        
        let result = backend.dot(&x, &y);
        assert!(result.is_err(), "维度不匹配应返回错误");
    }

    /// 测试 scale 的 alpha=1.0 分支（不执行任何操作）
    #[test]
    fn test_scale_alpha_one() {
        let backend = AvxBackend::new();
        let mut x = vec![1.0, 2.0, 3.0];
        let x_copy = x.clone();
        
        backend.scale(1.0, &mut x);
        assert_eq!(x, x_copy, "alpha=1.0 时 x 不应改变");
    }

    /// 测试 nrm2 的空向量分支
    #[test]
    fn test_nrm2_empty() {
        let backend = AvxBackend::new();
        let x: Vec<f32> = vec![];
        let result = backend.nrm2(&x);
        assert_eq!(result, 0.0, "空向量的范数应为 0");
    }

    /// 测试 copy 的维度不匹配错误分支
    #[test]
    fn test_copy_dimension_mismatch() {
        let backend = AvxBackend::new();
        let x = vec![1.0, 2.0];
        let mut y = vec![0.0; 3];
        
        let result = backend.copy(&x, &mut y);
        assert!(result.is_err(), "维度不匹配应返回错误");
    }

    /// 测试 gemv 的 alpha=0 分支（不执行矩阵-向量乘法，但 beta 仍生效）
    #[test]
    fn test_gemv_zero_alpha() {
        let backend = AvxBackend::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 2.0];
        
        // 测试 alpha=0, beta=1: y 保持不变
        let mut y1 = vec![10.0, 20.0, 30.0];
        let y1_copy = y1.clone();
        backend.gemv(0.0, &a, 3, 2, &x, 1.0, &mut y1).unwrap();
        assert_eq!(y1, y1_copy, "alpha=0, beta=1 时 y 不应改变");
        
        // 测试 alpha=0, beta=0: y 被清空
        let mut y2 = vec![10.0, 20.0, 30.0];
        backend.gemv(0.0, &a, 3, 2, &x, 0.0, &mut y2).unwrap();
        assert!(y2.iter().all(|&v| v == 0.0), "alpha=0, beta=0 时 y 应被清空");
    }

    /// 测试 gemv 的 beta=0 分支（清空 y）
    #[test]
    fn test_gemv_zero_beta() {
        let backend = AvxBackend::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 2.0];
        let mut y = vec![10.0, 20.0, 30.0];
        
        backend.gemv(0.0, &a, 3, 2, &x, 0.0, &mut y).unwrap();
        assert!(y.iter().all(|&v| v == 0.0), "beta=0 时 y 应被清空");
    }

    /// 测试 gemv 的 beta!=0 且 beta!=1 分支（缩放 y）
    #[test]
    fn test_gemv_beta_scale() {
        let backend = AvxBackend::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 2.0];
        let mut y = vec![10.0, 20.0, 30.0];
        
        backend.gemv(0.0, &a, 3, 2, &x, 2.0, &mut y).unwrap();
        assert!((y[0] - 20.0).abs() < 1e-5, "y[0] 应为 10 * 2");
    }

    /// 测试 gemv 的维度不匹配错误分支
    #[test]
    fn test_gemv_dimension_mismatch() {
        let backend = AvxBackend::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];
        
        let result = backend.gemv(1.0, &a, 3, 2, &x, 0.0, &mut y);
        assert!(result.is_err(), "维度不匹配应返回错误");
    }

    /// 测试 gemm 的 alpha=0 分支（不执行矩阵乘法，但 beta 仍生效）
    #[test]
    fn test_gemm_zero_alpha() {
        let backend = AvxBackend::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // b: 3x2
        
        // 测试 alpha=0, beta=1: c 保持不变
        // a: 2x3, b: 3x2 => c 应该是 2x2 = 4
        let mut c1 = vec![10.0, 20.0, 30.0, 40.0];
        let c1_copy = c1.clone();
        backend.gemm(0.0, &a, 2, 3, &b, 3, 2, 1.0, &mut c1).unwrap();
        assert_eq!(c1, c1_copy, "alpha=0, beta=1 时 c 不应改变");
        
        // 测试 alpha=0, beta=2: c 被缩放
        let mut c2 = vec![10.0, 20.0, 30.0, 40.0];
        backend.gemm(0.0, &a, 2, 3, &b, 3, 2, 2.0, &mut c2).unwrap();
        assert!((c2[0] - 20.0).abs() < 1e-5, "alpha=0, beta=2 时 c[0] 应为 20");
    }

    /// 测试 gemm 的 beta=0 分支（清空 c）
    #[test]
    fn test_gemm_zero_beta() {
        let backend = AvxBackend::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // b: 3x2
        // a: 2x3, b: 3x2 => c 应该是 2x2 = 4
        let mut c = vec![10.0, 20.0, 30.0, 40.0]; 
        
        backend.gemm(0.0, &a, 2, 3, &b, 3, 2, 0.0, &mut c).unwrap();
        assert!(c.iter().all(|&v| v == 0.0), "beta=0 时 c 应被清空");
    }

    /// 测试 gemm 的维度不匹配错误分支（K != K2）
    #[test]
    fn test_gemm_k_mismatch() {
        let backend = AvxBackend::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 4x2
        let mut c = vec![0.0; 6]; // 2x3
        
        let result = backend.gemm(1.0, &a, 2, 3, &b, 4, 2, 0.0, &mut c);
        assert!(result.is_err(), "K 维度不匹配应返回错误");
    }

    /// 测试 gemm 的输出矩阵大小不匹配错误分支
    #[test]
    fn test_gemm_output_size_mismatch() {
        let backend = AvxBackend::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // a: 2x3, b: 3x2 => c 应该是 2x2 = 4
        let mut c = vec![0.0; 5]; // 大小不匹配（应为 4）
        
        let result = backend.gemm(1.0, &a, 2, 3, &b, 3, 2, 0.0, &mut c);
        assert!(result.is_err(), "输出矩阵大小不匹配应返回错误");
    }

    /// 测试 softmax 的空向量分支
    #[test]
    fn test_softmax_empty() {
        let backend = AvxBackend::new();
        let mut x: Vec<f32> = vec![];
        let result = backend.softmax(&mut x);
        assert!(result.is_ok(), "空向量应返回 Ok");
        assert!(x.is_empty(), "空向量应保持为空");
    }

    /// 测试 is_available 方法
    #[test]
    fn test_is_available() {
        let backend = AvxBackend::new();
        let _available = backend.is_available();
        // 在非 x86_64 架构上应返回 false
        #[cfg(not(target_arch = "x86_64"))]
        assert!(!available);
    }

    /// 测试 has_avx2 方法
    #[test]
    fn test_has_avx2() {
        let backend = AvxBackend::new();
        let _has_avx2 = backend.has_avx2(); // 仅在 x86_64 上有意义
    }

    /// 测试 has_fma 方法
    #[test]
    fn test_has_fma() {
        let backend = AvxBackend::new();
        let _has_fma = backend.has_fma(); // 仅在 x86_64 上有意义
    }

    /// 测试 num_threads 方法
    #[test]
    fn test_num_threads() {
        let backend = AvxBackend::new();
        let threads = backend.num_threads();
        assert!(threads > 0, "线程数应大于 0");
    }

    /// 测试 simd_width 方法
    #[test]
    fn test_simd_width() {
        let backend = AvxBackend::new();
        let width = backend.simd_width();
        if backend.has_avx {
            assert_eq!(width, 8, "AVX SIMD 宽度应为 8 (256-bit)");
        } else {
            assert_eq!(width, 1, "无 AVX 时 SIMD 宽度应为 1");
        }
    }

    /// 测试 Default trait 实现
    #[test]
    fn test_avx_backend_default() {
        let backend = AvxBackend::default();
        assert_eq!(backend.num_threads(), AvxBackend::new().num_threads());
    }
}
