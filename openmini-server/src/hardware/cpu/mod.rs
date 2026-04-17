//! CPU 硬件抽象层
//!
//! 提供CPU相关的硬件加速接口
//!
//! ## 模块
//!
//! - `blas`: BLAS 后端 (OpenBLAS, Intel MKL)
//! - `avx`: AVX SIMD 后端 (x86_64)
//! - `neon`: NEON SIMD 后端 (ARM64)
//!
//! ## 架构
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                      CpuOps Trait                        │
//! │  - dot, axpy, scale, nrm2, copy, gemm, gemv, softmax    │
//! └─────────────────────────────────────────────────────────┘
//!        △                △                △
//!        │                │                │
//! ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐
//! │ BlasBackend │  │  AvxBackend │  │ NeonBackend │
//! │  (OpenBLAS) │  │  (AVX/AVX2) │  │   (ARM64)   │
//! │  (Intel MKL)│  │  (AVX-512)  │  │             │
//! └─────────────┘  └─────────────┘  └─────────────┘
//! ```

pub mod blas;

#[cfg(target_arch = "x86_64")]
pub mod avx;

#[cfg(target_arch = "aarch64")]
pub mod neon;

use anyhow::{bail, Result};
use ndarray::Array2;

// ============================================================================
// CPU 后端类型
// ============================================================================

/// CPU 后端类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum CpuBackendType {
    /// BLAS 后端 (OpenBLAS 或 Intel MKL)
    Blas,
    /// AVX SIMD 后端 (x86_64)
    Avx,
    /// NEON SIMD 后端 (ARM64)
    Neon,
    /// 纯 Rust 后端 (回退)
    Rust,
}

impl std::fmt::Display for CpuBackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CpuBackendType::Blas => write!(f, "BLAS"),
            CpuBackendType::Avx => write!(f, "AVX"),
            CpuBackendType::Neon => write!(f, "NEON"),
            CpuBackendType::Rust => write!(f, "Rust"),
        }
    }
}

// ============================================================================
// CPU 操作 Trait
// ============================================================================

/// CPU 计算操作统一接口
///
/// 所有 CPU 后端都需要实现此 trait，提供统一的向量/矩阵操作接口
#[allow(dead_code)]
pub trait CpuOps: Send + Sync {
    /// 返回后端名称
    fn backend_name(&self) -> &'static str;

    /// 返回后端类型
    fn backend_type(&self) -> CpuBackendType;

    /// 向量点积: x · y
    fn dot(&self, x: &[f32], y: &[f32]) -> Result<f32>;

    /// 向量加法: y = alpha * x + y
    fn axpy(&self, alpha: f32, x: &[f32], y: &mut [f32]) -> Result<()>;

    /// 向量缩放: x = alpha * x
    fn scale(&self, alpha: f32, x: &mut [f32]);

    /// 向量范数: ||x||_2
    fn nrm2(&self, x: &[f32]) -> f32;

    /// 向量复制: y = x
    fn copy(&self, x: &[f32], y: &mut [f32]) -> Result<()>;

    /// 矩阵乘法: C = alpha * A * B + beta * C
    fn gemm(
        &self,
        alpha: f32,
        a: &Array2<f32>,
        b: &Array2<f32>,
        beta: f32,
        c: &mut Array2<f32>,
    ) -> Result<()>;

    /// 矩阵-向量乘法: y = alpha * A * x + beta * y
    fn gemv(&self, alpha: f32, a: &Array2<f32>, x: &[f32], beta: f32, y: &mut [f32]) -> Result<()>;

    /// Softmax: x_i = exp(x_i) / sum(exp(x))
    fn softmax(&self, x: &mut [f32]) -> Result<()>;

    /// 返回线程数
    fn num_threads(&self) -> usize;
}

// ============================================================================
// CPU 信息结构
// ============================================================================

/// CPU 详细信息
#[derive(Debug, Clone)]
pub struct CpuInfoDetail {
    /// 架构名称
    pub arch: String,
    /// 物理核心数
    pub physical_cores: usize,
    /// 逻辑核心数
    pub logical_cores: usize,
    /// SIMD 支持情况
    pub simd: SimdInfo,
    /// 推荐后端
    pub recommended_backend: CpuBackendType,
    /// Rayon 线程数
    pub rayon_threads: usize,
}

/// SIMD 信息
#[derive(Debug, Clone, Default)]
pub struct SimdInfo {
    /// AVX 支持
    pub avx: bool,
    /// AVX2 支持
    pub avx2: bool,
    /// AVX-512 支持
    pub avx512: bool,
    /// NEON 支持
    pub neon: bool,
    /// 最佳 SIMD 宽度 (位)
    pub best_width: usize,
}

impl std::fmt::Display for CpuInfoDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Architecture: {}", self.arch)?;
        writeln!(f, "Physical cores: {}", self.physical_cores)?;
        writeln!(f, "Logical cores: {}", self.logical_cores)?;
        writeln!(f, "SIMD: {}", self.simd)?;
        writeln!(f, "Recommended backend: {}", self.recommended_backend)?;
        write!(f, "Rayon threads: {}", self.rayon_threads)
    }
}

impl std::fmt::Display for SimdInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut features = Vec::new();
        if self.avx512 {
            features.push("AVX-512");
        }
        if self.avx2 {
            features.push("AVX2");
        }
        if self.avx {
            features.push("AVX");
        }
        if self.neon {
            features.push("NEON");
        }
        if features.is_empty() {
            write!(f, "None ({}-bit)", self.best_width)
        } else {
            write!(f, "{} ({}-bit)", features.join(", "), self.best_width)
        }
    }
}

// ============================================================================
// CPU 后端选择器
// ============================================================================

/// CPU 后端选择器
pub struct CpuBackend;

impl CpuBackend {
    /// 检测最优 CPU 后端
    ///
    /// 优先级: BLAS > AVX > NEON > Rust
    pub fn detect() -> CpuBackendType {
        #[cfg(any(feature = "blas-openblas", feature = "blas-mkl"))]
        {
            return CpuBackendType::Blas;
        }

        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx") {
                return CpuBackendType::Avx;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if detect_neon_support() {
                return CpuBackendType::Neon;
            }
        }

        CpuBackendType::Rust
    }

    /// 创建最优后端实例
    pub fn create() -> Box<dyn CpuOps> {
        let backend_type = Self::detect();
        Self::create_from_type(backend_type)
    }

    /// 根据指定类型创建后端实例
    pub fn create_from_type(backend_type: CpuBackendType) -> Box<dyn CpuOps> {
        match backend_type {
            #[cfg(any(feature = "blas-openblas", feature = "blas-mkl"))]
            CpuBackendType::Blas => Box::new(blas::BlasBackend::new()),

            #[cfg(target_arch = "x86_64")]
            CpuBackendType::Avx => Box::new(avx::AvxBackend::new()),

            #[cfg(target_arch = "aarch64")]
            CpuBackendType::Neon => Box::new(neon::NeonBackend::new()),

            _ => Box::new(RustBackend::new()),
        }
    }

    /// 检测是否有 AVX 支持
    #[allow(dead_code)]
    pub fn has_avx() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("avx")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// 检测是否有 AVX2 支持
    #[allow(dead_code)]
    pub fn has_avx2() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// 检测是否有 AVX-512 支持
    #[allow(dead_code)]
    pub fn has_avx512() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("avx512f")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// 检测是否有 NEON 支持
    #[allow(dead_code)]
    pub fn has_neon() -> bool {
        detect_neon_support()
    }

    /// 检测是否有 BLAS 支持
    #[allow(dead_code)]
    pub fn has_blas() -> bool {
        cfg!(any(feature = "blas-openblas", feature = "blas-mkl"))
    }

    /// 获取结构化 CPU 信息
    pub fn cpu_info() -> CpuInfoDetail {
        let arch = Self::get_arch();
        let physical_cores = num_cpus::get_physical();
        let logical_cores = num_cpus::get();
        let simd = Self::get_simd_info();
        let recommended_backend = Self::detect();
        let rayon_threads = rayon::current_num_threads();

        CpuInfoDetail {
            arch,
            physical_cores,
            logical_cores,
            simd,
            recommended_backend,
            rayon_threads,
        }
    }

    fn get_arch() -> String {
        #[cfg(target_arch = "x86_64")]
        {
            "x86_64".to_string()
        }

        #[cfg(target_arch = "aarch64")]
        {
            "aarch64".to_string()
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            "unknown".to_string()
        }
    }

    fn get_simd_info() -> SimdInfo {
        let mut info = SimdInfo::default();

        #[cfg(target_arch = "x86_64")]
        {
            info.avx = std::is_x86_feature_detected!("avx");
            info.avx2 = std::is_x86_feature_detected!("avx2");
            info.avx512 = std::is_x86_feature_detected!("avx512f");
            info.best_width = if info.avx512 {
                512
            } else if info.avx {
                256
            } else {
                128
            };
        }

        #[cfg(target_arch = "aarch64")]
        {
            info.neon = detect_neon_support();
            info.best_width = if info.neon { 128 } else { 0 };
        }

        info
    }
}

// ============================================================================
// 纯 Rust 回退后端
// ============================================================================

/// 纯 Rust 后端 (回退实现)
struct RustBackend {
    num_threads: usize,
}

impl RustBackend {
    fn new() -> Self {
        Self {
            num_threads: rayon::current_num_threads(),
        }
    }
}

impl CpuOps for RustBackend {
    fn backend_name(&self) -> &'static str {
        "Rust (fallback)"
    }

    fn backend_type(&self) -> CpuBackendType {
        CpuBackendType::Rust
    }

    fn dot(&self, x: &[f32], y: &[f32]) -> Result<f32> {
        if x.len() != y.len() {
            bail!("向量维度不匹配: x({}) y({})", x.len(), y.len());
        }
        Ok(x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum())
    }

    fn axpy(&self, alpha: f32, x: &[f32], y: &mut [f32]) -> Result<()> {
        if x.len() != y.len() {
            bail!("向量维度不匹配: x({}) y({})", x.len(), y.len());
        }
        for i in 0..y.len() {
            y[i] += alpha * x[i];
        }
        Ok(())
    }

    fn scale(&self, alpha: f32, x: &mut [f32]) {
        x.iter_mut().for_each(|xi| *xi *= alpha);
    }

    fn nrm2(&self, x: &[f32]) -> f32 {
        let sum: f32 = x.iter().map(|&xi| xi * xi).sum();
        sum.sqrt()
    }

    fn copy(&self, x: &[f32], y: &mut [f32]) -> Result<()> {
        if x.len() != y.len() {
            bail!("向量维度不匹配: x({}) y({})", x.len(), y.len());
        }
        y.copy_from_slice(x);
        Ok(())
    }

    fn gemm(
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

        if beta == 0.0 {
            c.fill(0.0);
        } else {
            c.mapv_inplace(|x| x * beta);
        }

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for k_idx in 0..k {
                    sum += a[[i, k_idx]] * b[[k_idx, j]];
                }
                c[[i, j]] += alpha * sum;
            }
        }

        Ok(())
    }

    fn gemv(&self, alpha: f32, a: &Array2<f32>, x: &[f32], beta: f32, y: &mut [f32]) -> Result<()> {
        let (m, n) = a.dim();

        if n != x.len() {
            bail!("矩阵-向量维度不匹配: A({}×{}) x({})", m, n, x.len());
        }
        if m != y.len() {
            bail!("输出向量大小不匹配: y({}) 期望({})", y.len(), m);
        }

        if beta == 0.0 {
            y.fill(0.0);
        } else {
            y.iter_mut().for_each(|yi| *yi *= beta);
        }

        for i in 0..m {
            let mut sum = 0.0f32;
            for j in 0..n {
                sum += a[[i, j]] * x[j];
            }
            y[i] += alpha * sum;
        }

        Ok(())
    }

    fn softmax(&self, x: &mut [f32]) -> Result<()> {
        if x.is_empty() {
            return Ok(());
        }

        let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;

        for xi in x.iter_mut() {
            *xi = (*xi - max_val).exp();
            sum += *xi;
        }

        if sum > 0.0 {
            for xi in x.iter_mut() {
                *xi /= sum;
            }
        }

        Ok(())
    }

    fn num_threads(&self) -> usize {
        self.num_threads
    }
}

// ============================================================================
// 为现有后端实现 CpuOps trait
// ============================================================================

#[cfg(any(feature = "blas-openblas", feature = "blas-mkl"))]
impl CpuOps for blas::BlasBackend {
    fn backend_name(&self) -> &'static str {
        #[cfg(feature = "blas-openblas")]
        {
            "OpenBLAS"
        }
        #[cfg(feature = "blas-mkl")]
        {
            "Intel MKL"
        }
    }

    fn backend_type(&self) -> CpuBackendType {
        CpuBackendType::Blas
    }

    fn dot(&self, x: &[f32], y: &[f32]) -> Result<f32> {
        self.dot(x, y)
    }

    fn axpy(&self, alpha: f32, x: &[f32], y: &mut [f32]) -> Result<()> {
        self.axpy(alpha, x, y)
    }

    fn scale(&self, alpha: f32, x: &mut [f32]) {
        self.scale(alpha, x);
    }

    fn nrm2(&self, x: &[f32]) -> f32 {
        self.nrm2(x)
    }

    fn copy(&self, x: &[f32], y: &mut [f32]) -> Result<()> {
        self.copy(x, y)
    }

    fn gemm(
        &self,
        alpha: f32,
        a: &Array2<f32>,
        b: &Array2<f32>,
        beta: f32,
        c: &mut Array2<f32>,
    ) -> Result<()> {
        self.gemm(alpha, a, b, beta, c)
    }

    fn gemv(&self, alpha: f32, a: &Array2<f32>, x: &[f32], beta: f32, y: &mut [f32]) -> Result<()> {
        self.gemv(alpha, a, x, beta, y)
    }

    fn softmax(&self, x: &mut [f32]) -> Result<()> {
        if x.is_empty() {
            return Ok(());
        }

        let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;

        for xi in x.iter_mut() {
            *xi = (*xi - max_val).exp();
            sum += *xi;
        }

        if sum > 0.0 {
            for xi in x.iter_mut() {
                *xi /= sum;
            }
        }

        Ok(())
    }

    fn num_threads(&self) -> usize {
        self.num_threads()
    }
}

#[cfg(target_arch = "x86_64")]
impl CpuOps for avx::AvxBackend {
    fn backend_name(&self) -> &'static str {
        if self.has_avx2() {
            "AVX2"
        } else {
            "AVX"
        }
    }

    fn backend_type(&self) -> CpuBackendType {
        CpuBackendType::Avx
    }

    fn dot(&self, x: &[f32], y: &[f32]) -> Result<f32> {
        self.dot(x, y)
    }

    fn axpy(&self, alpha: f32, x: &[f32], y: &mut [f32]) -> Result<()> {
        self.axpy(alpha, x, y)
    }

    fn scale(&self, alpha: f32, x: &mut [f32]) {
        self.scale(alpha, x);
    }

    fn nrm2(&self, x: &[f32]) -> f32 {
        self.nrm2(x)
    }

    fn copy(&self, x: &[f32], y: &mut [f32]) -> Result<()> {
        self.copy(x, y)
    }

    fn gemm(
        &self,
        alpha: f32,
        a: &Array2<f32>,
        b: &Array2<f32>,
        beta: f32,
        c: &mut Array2<f32>,
    ) -> Result<()> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();

        let a_slice = a
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("矩阵 A 不是连续存储"))?;
        let b_slice = b
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("矩阵 B 不是连续存储"))?;
        let c_slice = c
            .as_slice_mut()
            .ok_or_else(|| anyhow::anyhow!("矩阵 C 不是连续存储"))?;

        self.gemm(alpha, a_slice, m, k, b_slice, k2, n, beta, c_slice)
    }

    fn gemv(&self, alpha: f32, a: &Array2<f32>, x: &[f32], beta: f32, y: &mut [f32]) -> Result<()> {
        let (m, n) = a.dim();

        let a_slice = a
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("矩阵 A 不是连续存储"))?;

        self.gemv(alpha, a_slice, m, n, x, beta, y)
    }

    fn softmax(&self, x: &mut [f32]) -> Result<()> {
        self.softmax(x)
    }

    fn num_threads(&self) -> usize {
        self.num_threads()
    }
}

#[cfg(target_arch = "aarch64")]
impl CpuOps for neon::NeonBackend {
    fn backend_name(&self) -> &'static str {
        "NEON"
    }

    fn backend_type(&self) -> CpuBackendType {
        CpuBackendType::Neon
    }

    fn dot(&self, x: &[f32], y: &[f32]) -> Result<f32> {
        self.dot(x, y)
    }

    fn axpy(&self, alpha: f32, x: &[f32], y: &mut [f32]) -> Result<()> {
        self.axpy(alpha, x, y)
    }

    fn scale(&self, alpha: f32, x: &mut [f32]) {
        self.scale(alpha, x);
    }

    fn nrm2(&self, x: &[f32]) -> f32 {
        self.nrm2(x)
    }

    fn copy(&self, x: &[f32], y: &mut [f32]) -> Result<()> {
        self.copy(x, y)
    }

    fn gemm(
        &self,
        alpha: f32,
        a: &Array2<f32>,
        b: &Array2<f32>,
        beta: f32,
        c: &mut Array2<f32>,
    ) -> Result<()> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();

        let a_slice = a
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("矩阵 A 不是连续存储"))?;
        let b_slice = b
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("矩阵 B 不是连续存储"))?;
        let c_slice = c
            .as_slice_mut()
            .ok_or_else(|| anyhow::anyhow!("矩阵 C 不是连续存储"))?;

        self.gemm(alpha, a_slice, m, k, b_slice, k2, n, beta, c_slice)
    }

    fn gemv(&self, alpha: f32, a: &Array2<f32>, x: &[f32], beta: f32, y: &mut [f32]) -> Result<()> {
        let (m, n) = a.dim();

        let a_slice = a
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("矩阵 A 不是连续存储"))?;

        self.gemv(alpha, a_slice, m, n, x, beta, y)
    }

    fn softmax(&self, x: &mut [f32]) -> Result<()> {
        self.softmax(x)
    }

    fn num_threads(&self) -> usize {
        self.num_threads()
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_detect() {
        let backend_type = CpuBackend::detect();
        println!("Detected CPU backend: {:?}", backend_type);

        let has_blas = CpuBackend::has_blas();
        let has_avx = CpuBackend::has_avx();
        let has_neon = CpuBackend::has_neon();

        println!("BLAS: {}, AVX: {}, NEON: {}", has_blas, has_avx, has_neon);
    }

    #[test]
    fn test_cpu_info() {
        let info = CpuBackend::cpu_info();
        println!("{}", info);
        assert!(info.physical_cores > 0);
        assert!(info.logical_cores > 0);
    }

    #[test]
    fn test_create_backend() {
        let backend = CpuBackend::create();
        println!("Created backend: {}", backend.backend_name());
        assert!(!backend.backend_name().is_empty());
    }

    #[test]
    fn test_cpu_ops() {
        let backend = CpuBackend::create();

        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 1.0, 1.0, 1.0];

        let dot = backend.dot(&x, &y).unwrap();
        assert!((dot - 10.0).abs() < 1e-5);

        let nrm2 = backend.nrm2(&x);
        assert!((nrm2 - 5.4772255).abs() < 1e-5);

        let mut z = vec![0.0; 4];
        backend.copy(&x, &mut z).unwrap();
        assert_eq!(x, z);
    }

    #[test]
    fn test_softmax() {
        let backend = CpuBackend::create();

        let mut x = vec![1.0, 2.0, 3.0];
        backend.softmax(&mut x).unwrap();

        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // ========== 新增测试开始 ==========

    /// 测试CpuBackendType枚举的Display实现
    #[test]
    fn test_cpu_backend_type_display() {
        assert_eq!(format!("{}", CpuBackendType::Blas), "BLAS");
        assert_eq!(format!("{}", CpuBackendType::Avx), "AVX");
        assert_eq!(format!("{}", CpuBackendType::Neon), "NEON");
        assert_eq!(format!("{}", CpuBackendType::Rust), "Rust");
    }

    /// 测试axpy操作 (y = alpha * x + y)
    #[test]
    fn test_axpy_operation() {
        let backend = CpuBackend::create();

        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![10.0, 20.0, 30.0, 40.0];

        // y = 2.0 * x + y
        backend.axpy(2.0, &x, &mut y).unwrap();

        // 验证结果
        assert!((y[0] - 12.0).abs() < 1e-5); // 2*1 + 10
        assert!((y[1] - 24.0).abs() < 1e-5); // 2*2 + 20
        assert!((y[2] - 36.0).abs() < 1e-5); // 2*3 + 30
        assert!((y[3] - 48.0).abs() < 1e-5); // 2*4 + 40
    }

    /// 测试scale操作 (x = alpha * x)
    #[test]
    fn test_scale_operation() {
        let backend = CpuBackend::create();

        let mut x = vec![1.0, 2.0, 3.0, 4.0];

        // x = 3.0 * x
        backend.scale(3.0, &mut x);

        assert!((x[0] - 3.0).abs() < 1e-5);
        assert!((x[1] - 6.0).abs() < 1e-5);
        assert!((x[2] - 9.0).abs() < 1e-5);
        assert!((x[3] - 12.0).abs() < 1e-5);
    }

    /// 测试gemm矩阵乘法 (C = alpha * A * B + beta * C)
    #[test]
    fn test_gemm_operation() {
        let backend = CpuBackend::create();

        // A: 2x3矩阵
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // B: 3x2矩阵
        let b = Array2::from_shape_vec((3, 2), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();

        // C: 2x2零矩阵
        let mut c = Array2::from_elem((2, 2), 0.0);

        // C = 1.0 * A * B + 0.0 * C
        backend.gemm(1.0, &a, &b, 0.0, &mut c).unwrap();

        // 手动计算验证
        // C[0,0] = 1*7 + 2*9 + 3*11 = 58
        assert!((c[[0, 0]] - 58.0).abs() < 1e-4);
        // C[0,1] = 1*8 + 2*10 + 3*12 = 64
        assert!((c[[0, 1]] - 64.0).abs() < 1e-4);
        // C[1,0] = 4*7 + 5*9 + 6*11 = 139
        assert!((c[[1, 0]] - 139.0).abs() < 1e-4);
        // C[1,1] = 4*8 + 5*10 + 6*12 = 154
        assert!((c[[1, 1]] - 154.0).abs() < 1e-4);
    }

    /// 测试gemv矩阵-向量乘法 (y = alpha * A * x + beta * y)
    #[test]
    fn test_gemv_operation() {
        let backend = CpuBackend::create();

        // A: 2x3矩阵
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // x: 长度为3的向量
        let x = vec![1.0, 2.0, 3.0];

        // y: 长度为2的向量
        let mut y = vec![0.0; 2];

        // y = 1.0 * A * x + 0.0 * y
        backend.gemv(1.0, &a, &x, 0.0, &mut y).unwrap();

        // 手动计算验证
        // y[0] = 1*1 + 2*2 + 3*3 = 14
        assert!((y[0] - 14.0).abs() < 1e-4);
        // y[1] = 4*1 + 5*2 + 6*3 = 32
        assert!((y[1] - 32.0).abs() < 1e-4);
    }

    /// 测试dot操作的维度不匹配错误处理
    #[test]
    fn test_dot_dimension_mismatch() {
        let backend = CpuBackend::create();

        let x = vec![1.0, 2.0, 3.0]; // 长度3
        let y = vec![1.0, 2.0]; // 长度2

        let result = backend.dot(&x, &y);

        // 应该返回错误
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("维度不匹配"));
    }

    /// 测试空向量的操作
    #[test]
    fn test_empty_vector_operations() {
        let backend = CpuBackend::create();

        let x: Vec<f32> = vec![];
        let y: Vec<f32> = vec![];

        // 空向量点积应为0
        let dot = backend.dot(&x, &y).unwrap();
        assert!((dot - 0.0).abs() < 1e-6);

        // 空向量范数应为0
        let nrm2 = backend.nrm2(&x);
        assert!((nrm2 - 0.0).abs() < 1e-6);

        // 空向量softmax不应失败
        let mut empty_softmax: Vec<f32> = vec![];
        let result = backend.softmax(&mut empty_softmax);
        assert!(result.is_ok());
    }

    /// 测试CpuInfoDetail的Display trait输出格式
    #[test]
    fn test_cpu_info_detail_display() {
        let info = CpuBackend::cpu_info();
        let display_output = format!("{}", info);

        // 验证输出包含关键字段
        assert!(display_output.contains("Architecture:"));
        assert!(display_output.contains("Physical cores:"));
        assert!(display_output.contains("Logical cores:"));
        assert!(display_output.contains("SIMD:"));
        assert!(display_output.contains("Recommended backend:"));
        assert!(display_output.contains("Rayon threads:"));
    }

    /// 测试SimdInfo的Display trait（包含/不包含SIMD特性）
    #[test]
    fn test_simd_info_display() {
        let simd = CpuBackend::cpu_info().simd;
        let display_output = format!("{}", simd);

        // 应该包含位宽信息
        assert!(display_output.contains("bit"));

        // 根据平台可能包含或不包含具体SIMD特性名称
        if simd.avx || simd.avx2 || simd.avx512 || simd.neon {
            // 如果有任何SIMD支持，应该显示特性名称
            assert!(
                display_output.contains("AVX") || display_output.contains("NEON"),
                "SIMD显示应包含AVX或NEON"
            );
        } else {
            // 无SIMD时应显示"None"
            assert!(display_output.contains("None"));
        }
    }

    /// 测试num_threads()返回合理的线程数
    #[test]
    fn test_num_threads() {
        let backend = CpuBackend::create();
        let num_threads = backend.num_threads();

        // 线程数应该至少为1
        assert!(num_threads >= 1, "线程数应该>=1，实际={}", num_threads);

        // 线程数不应该超过逻辑核心数的合理范围（比如<=1024）
        assert!(num_threads <= 1024, "线程数应该在合理范围内");
    }

    /// 测试后端类型检测的一致性
    #[test]
    fn test_backend_type_consistency() {
        let detected_type = CpuBackend::detect();
        let backend = CpuBackend::create();

        // 检测到的类型和创建的后端类型应该一致
        assert_eq!(backend.backend_type(), detected_type);

        // 后端类型应该在已知范围内
        match detected_type {
            CpuBackendType::Blas
            | CpuBackendType::Avx
            | CpuBackendType::Neon
            | CpuBackendType::Rust => {}
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn detect_neon_support() -> bool {
    std::arch::aarch64::is_aarch64_feature_detected!("neon")
}

#[cfg(not(target_arch = "aarch64"))]
fn detect_neon_support() -> bool {
    false
}
