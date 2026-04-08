//! CPU BLAS 后端实现
//!
//! 提供基于BLAS的CPU加速支持。
//! 支持：OpenBLAS, Intel MKL

#![allow(dead_code)]

use anyhow::{bail, Result};
use ndarray::Array2;
use rayon::prelude::*;

#[cfg(feature = "blas-openblas")]
use cblas::{Layout, Transpose};

#[cfg(feature = "blas-mkl")]
use cblas::{Layout, Transpose};

/// 并行阈值常量
const PARALLEL_THRESHOLD: usize = 1024;
const GEMM_PARALLEL_THRESHOLD: usize = 64;

/// BLAS 后端
pub struct BlasBackend {
    num_threads: usize,
}

impl BlasBackend {
    pub fn new() -> Self {
        let num_threads = rayon::current_num_threads();
        let backend = Self { num_threads };
        backend.init();
        backend
    }

    pub fn new_with_threads(num_threads: usize) -> Self {
        let backend = Self { num_threads };
        backend.init();
        backend
    }

    pub fn set_num_threads(&mut self, num_threads: usize) {
        self.num_threads = num_threads;
        self.set_blas_threads(num_threads);
    }

    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// 设置 BLAS 库内部线程数
    fn set_blas_threads(&self, _num_threads: usize) {
        #[cfg(feature = "blas-openblas")]
        {
            extern "C" {
                fn openblas_set_num_threads(num: i32);
            }
            unsafe {
                openblas_set_num_threads(_num_threads as i32);
            }
        }

        #[cfg(feature = "blas-mkl")]
        {
            extern "C" {
                fn mkl_set_num_threads(num: i32);
            }
            unsafe {
                mkl_set_num_threads(_num_threads as i32);
            }
        }
    }

    /// 初始化时设置 BLAS 线程数 (公开方法，可手动调用重新设置)
    pub fn init(&self) {
        self.set_blas_threads(self.num_threads);
    }
}

impl Default for BlasBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl BlasBackend {
    /// 矩阵乘法: C = alpha * A * B + beta * C
    /// 
    /// 参数:
    /// - alpha: A*B 的系数
    /// - a: M×K 矩阵 (行主序)
    /// - b: K×N 矩阵 (行主序)
    /// - beta: C 的系数
    /// - c: M×N 结果矩阵 (行主序)
    pub fn gemm(
        &self,
        alpha: f32,
        a: &Array2<f32>,
        b: &Array2<f32>,
        beta: f32,
        c: &mut Array2<f32>,
    ) -> Result<()> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        
        if k != k2 {
            bail!("矩阵维度不匹配: A({}×{}) B({}×{})", m, k, k2, n);
        }

        #[cfg(any(feature = "blas-openblas", feature = "blas-mkl"))]
        {
            let a_data = a.as_slice().ok_or_else(|| anyhow::anyhow!("矩阵 A 不是连续存储"))?;
            let b_data = b.as_slice().ok_or_else(|| anyhow::anyhow!("矩阵 B 不是连续存储"))?;
            let c_data = c.as_slice_mut().ok_or_else(|| anyhow::anyhow!("矩阵 C 不是连续存储"))?;

            unsafe {
                cblas::sgemm(
                    Layout::RowMajor,
                    Transpose::None,
                    Transpose::None,
                    m as i32,
                    n as i32,
                    k as i32,
                    alpha,
                    a_data,
                    k as i32,
                    b_data,
                    n as i32,
                    beta,
                    c_data,
                    n as i32,
                );
            }
            return Ok(());
        }

        #[cfg(not(any(feature = "blas-openblas", feature = "blas-mkl")))]
        {
            self.gemm_fallback(alpha, a, b, beta, c);
            Ok(())
        }
    }

    #[cfg(not(any(feature = "blas-openblas", feature = "blas-mkl")))]
    fn gemm_fallback(&self, alpha: f32, a: &Array2<f32>, b: &Array2<f32>, beta: f32, c: &mut Array2<f32>) {
        let (m, k) = a.dim();
        let (_, n) = c.dim();

        if beta == 0.0 {
            c.fill(0.0);
        } else {
            c.mapv_inplace(|x| x * beta);
        }

        if m >= GEMM_PARALLEL_THRESHOLD && n >= GEMM_PARALLEL_THRESHOLD {
            let a_rows: Vec<_> = a.rows().into_iter().collect();
            let b_view = b.view();
            
            let results: Vec<(usize, usize, f32)> = (0..m)
                .into_par_iter()
                .flat_map(|i| {
                    let a_row = &a_rows[i];
                    (0..n).into_par_iter().map(move |j| {
                        let mut sum = 0.0f32;
                        for k_idx in 0..k {
                            sum += a_row[k_idx] * b_view[[k_idx, j]];
                        }
                        (i, j, alpha * sum)
                    })
                })
                .collect();
            
            for (i, j, val) in results {
                c[[i, j]] += val;
            }
        } else {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for k_idx in 0..k {
                        sum += a[[i, k_idx]] * b[[k_idx, j]];
                    }
                    c[[i, j]] += alpha * sum;
                }
            }
        }
    }

    /// 矩阵-向量乘法: y = alpha * A * x + beta * y
    pub fn gemv(&self, alpha: f32, a: &Array2<f32>, x: &[f32], beta: f32, y: &mut [f32]) -> Result<()> {
        let (m, n) = a.dim();
        
        if n != x.len() {
            bail!("矩阵-向量维度不匹配: A({}×{}) x({})", m, n, x.len());
        }
        if m != y.len() {
            bail!("输出向量大小不匹配: y({}) 期望({})", y.len(), m);
        }

        #[cfg(any(feature = "blas-openblas", feature = "blas-mkl"))]
        {
            let a_data = a.as_slice().ok_or_else(|| anyhow::anyhow!("矩阵 A 不是连续存储"))?;
            unsafe {
                cblas::sgemv(
                    Layout::RowMajor,
                    Transpose::None,
                    m as i32,
                    n as i32,
                    alpha,
                    a_data,
                    n as i32,
                    x,
                    1,
                    beta,
                    y,
                    1,
                );
            }
            return Ok(());
        }

        #[cfg(not(any(feature = "blas-openblas", feature = "blas-mkl")))]
        {
            self.gemv_fallback(alpha, a, x, beta, y);
            Ok(())
        }
    }

    #[cfg(not(any(feature = "blas-openblas", feature = "blas-mkl")))]
    fn gemv_fallback(&self, alpha: f32, a: &Array2<f32>, x: &[f32], beta: f32, y: &mut [f32]) {
        let (m, n) = a.dim();

        if beta == 0.0 {
            y.fill(0.0);
        } else {
            y.iter_mut().for_each(|y_i| *y_i *= beta);
        }

        if m >= GEMM_PARALLEL_THRESHOLD {
            let a_rows: Vec<_> = a.rows().into_iter().collect();
            
            let results: Vec<(usize, f32)> = (0..m)
                .into_par_iter()
                .map(|i| {
                    let mut sum = 0.0f32;
                    for j in 0..n {
                        sum += a_rows[i][j] * x[j];
                    }
                    (i, alpha * sum)
                })
                .collect();
            
            for (i, val) in results {
                y[i] += val;
            }
        } else {
            for i in 0..m {
                let mut sum = 0.0f32;
                for j in 0..n {
                    sum += a[[i, j]] * x[j];
                }
                y[i] += alpha * sum;
            }
        }
    }

    /// 向量缩放: x = alpha * x
    pub fn scale(&self, alpha: f32, x: &mut [f32]) {
        if alpha == 1.0 {
            return;
        }
        
        if x.len() < PARALLEL_THRESHOLD {
            x.iter_mut().for_each(|xi| *xi *= alpha);
        } else {
            x.par_iter_mut().for_each(|xi| *xi *= alpha);
        }
    }

    /// 向量加法: y = alpha * x + y
    pub fn axpy(&self, alpha: f32, x: &[f32], y: &mut [f32]) -> Result<()> {
        if x.len() != y.len() {
            bail!("向量维度不匹配: x({}) y({})", x.len(), y.len());
        }
        
        if alpha == 0.0 {
            return Ok(());
        }

        if x.len() < PARALLEL_THRESHOLD {
            y.iter_mut().zip(x.iter()).for_each(|(yi, &xi)| {
                *yi += alpha * xi;
            });
        } else {
            y.par_iter_mut().zip(x.par_iter()).for_each(|(yi, &xi)| {
                *yi += alpha * xi;
            });
        }
        
        Ok(())
    }

    /// 向量点积: x · y
    pub fn dot(&self, x: &[f32], y: &[f32]) -> Result<f32> {
        if x.len() != y.len() {
            bail!("向量维度不匹配: x({}) y({})", x.len(), y.len());
        }
        
        if x.len() < PARALLEL_THRESHOLD {
            Ok(x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum())
        } else {
            Ok(x.par_iter().zip(y.par_iter()).map(|(&a, &b)| a * b).sum())
        }
    }

    /// 向量范数: ||x||_2
    pub fn nrm2(&self, x: &[f32]) -> f32 {
        let sum: f32 = if x.len() < PARALLEL_THRESHOLD {
            x.iter().map(|&xi| xi * xi).sum()
        } else {
            x.par_iter().map(|&xi| xi * xi).sum()
        };
        sum.sqrt()
    }

    /// 向量复制: y = x
    pub fn copy(&self, x: &[f32], y: &mut [f32]) -> Result<()> {
        if x.len() != y.len() {
            bail!("向量维度不匹配: x({}) y({})", x.len(), y.len());
        }
        y.copy_from_slice(x);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blas_backend_creation() {
        let backend = BlasBackend::new();
        assert!(backend.num_threads() > 0);
    }

    #[test]
    fn test_gemm() {
        let backend = BlasBackend::new();
        
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mut c = Array2::zeros((2, 2));
        
        backend.gemm(1.0, &a, &b, 0.0, &mut c).unwrap();
        
        assert!((c[[0, 0]] - 22.0).abs() < 1e-5);
        assert!((c[[0, 1]] - 28.0).abs() < 1e-5);
        assert!((c[[1, 0]] - 49.0).abs() < 1e-5);
        assert!((c[[1, 1]] - 64.0).abs() < 1e-5);
    }

    #[test]
    fn test_gemv() {
        let backend = BlasBackend::new();
        
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0, 0.0];
        
        backend.gemv(1.0, &a, &x, 0.0, &mut y).unwrap();
        
        assert!((y[0] - 14.0).abs() < 1e-5);
        assert!((y[1] - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_dot() {
        let backend = BlasBackend::new();
        
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        
        let result = backend.dot(&x, &y).unwrap();
        assert!((result - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_axpy() {
        let backend = BlasBackend::new();
        
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![4.0, 5.0, 6.0];
        
        backend.axpy(2.0, &x, &mut y).unwrap();
        
        assert!((y[0] - 6.0).abs() < 1e-5);
        assert!((y[1] - 9.0).abs() < 1e-5);
        assert!((y[2] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_nrm2() {
        let backend = BlasBackend::new();
        
        let x = vec![3.0, 4.0];
        let result = backend.nrm2(&x);
        
        assert!((result - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_scale() {
        let backend = BlasBackend::new();
        
        let mut x = vec![1.0, 2.0, 3.0];
        backend.scale(2.0, &mut x);
        
        assert!((x[0] - 2.0).abs() < 1e-5);
        assert!((x[1] - 4.0).abs() < 1e-5);
        assert!((x[2] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_copy() {
        let backend = BlasBackend::new();
        
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];
        
        backend.copy(&x, &mut y).unwrap();
        
        assert_eq!(x, y);
    }

    // 新增分支覆盖测试

    /// 测试 gemm 的维度不匹配错误分支
    #[test]
    fn test_gemm_dimension_mismatch() {
        let backend = BlasBackend::new();
        
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Array2::from_shape_vec((4, 2), vec![1.0; 8]).unwrap();
        let mut c = Array2::zeros((2, 2));
        
        let result = backend.gemm(1.0, &a, &b, 0.0, &mut c);
        assert!(result.is_err(), "维度不匹配应返回错误");
    }

    /// 测试 gemv 的维度不匹配错误分支（A 的列数 != x 长度）
    #[test]
    fn test_gemv_dimension_mismatch_x() {
        let backend = BlasBackend::new();
        
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let x = vec![1.0, 2.0]; // 应为 3
        let mut y = vec![0.0; 2];
        
        let result = backend.gemv(1.0, &a, &x, 0.0, &mut y);
        assert!(result.is_err(), "维度不匹配应返回错误");
    }

    /// 测试 gemv 的维度不匹配错误分支（A 的行数 != y 长度）
    #[test]
    fn test_gemv_dimension_mismatch_y() {
        let backend = BlasBackend::new();
        
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3]; // 应为 2
        
        let result = backend.gemv(1.0, &a, &x, 0.0, &mut y);
        assert!(result.is_err(), "维度不匹配应返回错误");
    }

    /// 测试 axpy 的 alpha=0 分支（不执行任何操作）
    #[test]
    fn test_axpy_zero_alpha() {
        let backend = BlasBackend::new();
        
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![4.0, 5.0, 6.0];
        let y_copy = y.clone();
        
        backend.axpy(0.0, &x, &mut y).unwrap();
        assert_eq!(y, y_copy, "alpha=0 时 y 不应改变");
    }

    /// 测试 axpy 的维度不匹配错误分支
    #[test]
    fn test_axpy_dimension_mismatch() {
        let backend = BlasBackend::new();
        
        let x = vec![1.0, 2.0];
        let mut y = vec![1.0, 2.0, 3.0];
        
        let result = backend.axpy(1.0, &x, &mut y);
        assert!(result.is_err(), "维度不匹配应返回错误");
    }

    /// 测试 dot 的维度不匹配错误分支
    #[test]
    fn test_dot_dimension_mismatch() {
        let backend = BlasBackend::new();
        
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0, 3.0];
        
        let result = backend.dot(&x, &y);
        assert!(result.is_err(), "维度不匹配应返回错误");
    }

    /// 测试 scale 的 alpha=1.0 分支（不执行任何操作）
    #[test]
    fn test_scale_alpha_one() {
        let backend = BlasBackend::new();
        
        let mut x = vec![1.0, 2.0, 3.0];
        let x_copy = x.clone();
        
        backend.scale(1.0, &mut x);
        assert_eq!(x, x_copy, "alpha=1.0 时 x 不应改变");
    }

    /// 测试 copy 的维度不匹配错误分支
    #[test]
    fn test_copy_dimension_mismatch() {
        let backend = BlasBackend::new();
        
        let x = vec![1.0, 2.0];
        let mut y = vec![0.0; 3];
        
        let result = backend.copy(&x, &mut y);
        assert!(result.is_err(), "维度不匹配应返回错误");
    }

    /// 测试 new_with_threads 构造函数
    #[test]
    fn test_new_with_threads() {
        let backend = BlasBackend::new_with_threads(8);
        assert_eq!(backend.num_threads(), 8);
    }

    /// 测试 set_num_threads 方法
    #[test]
    fn test_set_num_threads() {
        let mut backend = BlasBackend::new();
        backend.set_num_threads(16);
        assert_eq!(backend.num_threads(), 16);
    }

    /// 测试 Default trait 实现
    #[test]
    fn test_blas_backend_default() {
        let backend = BlasBackend::default();
        assert!(backend.num_threads() > 0);
    }
}
