//! SIMD 加速的向量操作模块
//!
//! 提供高效的向量计算功能，支持：
//! - 余弦相似度计算（AVX2/NEON/标量）
//! - 向量嵌入计算（均值、归一化、L2 范数）
//! - 批量相似度计算与排序
//! - 运行时 CPU 特性检测与自动选择最优实现
//!
//! # 架构
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    SimdVectorOps                            │
//! ├─────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
//! │  │   AVX2      │  │   NEON      │  │   Scalar    │        │
//! │  │  (x86_64)   │  │  (aarch64)  │  │  (Fallback) │        │
//! │  └─────────────┘  └─────────────┘  └─────────────┘        │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use std::sync::OnceLock;

/// 全局 SIMD 能力缓存
///
/// 在首次访问时检测 CPU 的 SIMD 指令集支持情况并缓存结果，
/// 后续访问直接返回缓存值。
pub static SIMD_CAPS: OnceLock<SimdCapabilities> = OnceLock::new();

/// SIMD 指令集能力描述
///
/// 表示当前 CPU 支持的 SIMD 指令集特性，
/// 用于在运行时选择最优的向量运算实现。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimdCapabilities {
    /// 是否支持 AVX2 指令集（x86_64）
    pub has_avx2: bool,
    /// 是否支持 SSE4.2 指令集（x86_64）
    pub has_sse42: bool,
    /// 是否支持 NEON 指令集（aarch64/ARM）
    pub has_neon: bool,
    /// 是否支持 FMA（融合乘加）指令
    pub has_fma: bool,
}

impl SimdCapabilities {
    /// 检测当前 CPU 的 SIMD 指令集支持情况
    ///
    /// 根据目标架构在编译时选择对应的检测逻辑：
    /// - x86_64: 检测 AVX2、SSE4.2、FMA
    /// - aarch64: 检测 NEON
    /// - 其他: 所有特性均为 false
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                has_avx2: is_x86_feature_detected!("avx2"),
                has_sse42: is_x86_feature_detected!("sse4.2"),
                has_fma: is_x86_feature_detected!("fma"),
                has_neon: false,
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self {
                has_avx2: false,
                has_sse42: false,
                has_fma: false,
                has_neon: true,
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                has_avx2: false,
                has_sse42: false,
                has_neon: false,
                has_fma: false,
            }
        }
    }

    /// 获取全局缓存的 SIMD 能力实例
    ///
    /// 首次调用时执行检测并缓存，后续调用直接返回缓存值。
    /// 返回静态引用，生命周期为 `'static`。
    pub fn get() -> &'static Self {
        SIMD_CAPS.get_or_init(Self::detect)
    }

    /// 获取当前可用的最高 SIMD 级别
    ///
    /// 按优先级返回：AVX2 > NEON > SSE4.2 > Scalar
    pub fn simd_level(&self) -> SimdLevel {
        if self.has_avx2 {
            SimdLevel::Avx2
        } else if self.has_neon {
            SimdLevel::Neon
        } else if self.has_sse42 {
            SimdLevel::Sse42
        } else {
            SimdLevel::Scalar
        }
    }
}

impl Default for SimdCapabilities {
    fn default() -> Self {
        Self::detect()
    }
}

/// SIMD 加速级别枚举
///
/// 表示当前系统可用的 SIMD 指令集级别，
/// 用于选择最优的向量运算实现路径。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    /// 标量实现（无 SIMD 加速，作为 fallback）
    Scalar,
    /// SSE4.2 指令集（x86_64 基线支持）
    Sse42,
    /// AVX2 指令集（x86_64 高性能）
    Avx2,
    /// NEON 指令集（aarch64/ARM）
    Neon,
}

/// 相似度搜索结果
///
/// 表示向量相似度计算的结果，包含候选向量的 ID 和相似度分数。
/// 实现了 `Ord` trait，可按分数降序排序。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimilarityResult {
    /// 候选向量的索引 ID
    pub id: usize,
    /// 余弦相似度分数，范围 [-1.0, 1.0]
    pub score: f32,
}

impl Eq for SimilarityResult {}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for SimilarityResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl Ord for SimilarityResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self.score.is_nan(), other.score.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (false, false) => self.score.partial_cmp(&other.score).unwrap_or(std::cmp::Ordering::Equal),
        }
    }
}

/// SIMD 加速的向量运算器
///
/// 提供高性能的向量数学运算，根据 CPU 特性自动选择最优实现：
/// - AVX2 + FMA（x86_64，256 位宽）
/// - NEON（aarch64/ARM，128 位宽）
/// - 标量 fallback（其他架构）
///
/// # 示例
///
/// ```ignore
/// let ops = SimdVectorOps::new();
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// let dot = ops.dot_product(&a, &b);
/// ```
#[derive(Debug, Clone)]
pub struct SimdVectorOps {
    caps: SimdCapabilities,
}

impl Default for SimdVectorOps {
    fn default() -> Self {
        Self::new()
    }
}

impl SimdVectorOps {
    /// 创建使用默认 SIMD 能力的向量运算器
    ///
    /// 自动检测当前 CPU 的 SIMD 指令集支持情况。
    pub fn new() -> Self {
        Self {
            caps: *SimdCapabilities::get(),
        }
    }

    /// 使用指定的 SIMD 能力创建向量运算器
    ///
    /// # 参数
    /// - `caps`: SIMD 能力配置
    pub fn with_caps(caps: SimdCapabilities) -> Self {
        Self { caps }
    }

    /// 获取当前使用的 SIMD 加速级别
    pub fn simd_level(&self) -> SimdLevel {
        self.caps.simd_level()
    }

    /// 计算两个向量的点积（内积）
    ///
    /// 自动选择最优的 SIMD 实现路径。
    ///
    /// # 参数
    /// - `a`: 第一个向量
    /// - `b`: 第二个向量（长度必须与 `a` 相等）
    ///
    /// # 返回
    /// 点积结果，即 Σ(a[i] * b[i])
    ///
    /// # Panics
    /// 当两个向量长度不相等时 panic
    pub fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "向量长度必须相等");

        #[cfg(target_arch = "x86_64")]
        {
            if self.caps.has_avx2 && self.caps.has_fma {
                return unsafe { self.dot_product_avx2(a, b) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if self.caps.has_neon {
                return unsafe { self.dot_product_neon(a, b) };
            }
        }

        self.dot_product_scalar(a, b)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_product_avx2(&self, a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;

        let len = a.len();
        let chunks = len / 8;
        let remainder = len % 8;

        let mut sum = _mm256_setzero_ps();

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            sum = _mm256_fmadd_ps(va, vb, sum);
        }

        let vlow = _mm256_castps256_ps128(sum);
        let vhigh = _mm256_extractf128_ps(sum, 1);
        let vsum128 = _mm_add_ps(vlow, vhigh);

        let mut temp = [0.0f32; 4];
        _mm_storeu_ps(temp.as_mut_ptr(), vsum128);
        let mut result = temp[0] + temp[1] + temp[2] + temp[3];

        for i in 0..remainder {
            result += a[len - remainder + i] * b[len - remainder + i];
        }

        result
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn dot_product_neon(&self, a: &[f32], b: &[f32]) -> f32 {
        use std::arch::aarch64::*;

        let len = a.len();
        let chunks = len / 4;
        let remainder = len % 4;

        let mut sum = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));
            sum = vfmaq_f32(sum, va, vb);
        }

        let mut temp = [0.0f32; 4];
        vst1q_f32(temp.as_mut_ptr(), sum);
        let mut result = temp[0] + temp[1] + temp[2] + temp[3];

        for i in 0..remainder {
            result += a[len - remainder + i] * b[len - remainder + i];
        }

        result
    }

    fn dot_product_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    /// 计算向量的 L2 范数平方（即向量与自身的点积）
    ///
    /// # 参数
    /// - `v`: 输入向量
    ///
    /// # 返回
    /// Σ(v[i]²)
    pub fn l2_norm_squared(&self, v: &[f32]) -> f32 {
        self.dot_product(v, v)
    }

    /// 计算向量的 L2 范数（欧几里得范数）
    ///
    /// # 参数
    /// - `v`: 输入向量
    ///
    /// # 返回
    /// √(Σ(v[i]²))
    pub fn l2_norm(&self, v: &[f32]) -> f32 {
        self.l2_norm_squared(v).sqrt()
    }

    /// 计算两个向量的余弦相似度
    ///
    /// 公式：cos(θ) = (a · b) / (||a|| * ||b||)
    /// 结果范围：[-1.0, 1.0]，值越大表示向量越相似。
    /// 当任一向量为零向量时返回 0.0。
    ///
    /// # 参数
    /// - `a`: 第一个向量
    /// - `b`: 第二个向量
    ///
    /// # Panics
    /// 当两个向量长度不相等时 panic
    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "向量长度必须相等");

        let dot = self.dot_product(a, b);
        let norm_a = self.l2_norm(a);
        let norm_b = self.l2_norm(b);

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    /// 对向量进行 L2 归一化，返回新向量
    ///
    /// 将向量缩放为单位向量（L2 范数为 1）。
    /// 零向量原样返回。
    ///
    /// # 参数
    /// - `v`: 输入向量
    ///
    /// # 返回
    /// 归一化后的新向量
    pub fn normalize(&self, v: &[f32]) -> Vec<f32> {
        let norm = self.l2_norm(v);
        if norm == 0.0 {
            return v.to_vec();
        }

        self.mul_scalar(v, 1.0 / norm)
    }

    /// 原地 L2 归一化（就地修改向量，避免分配）
    ///
    /// 比 `normalize` 更高效，适用于需要修改原向量的场景。
    ///
    /// # 参数
    /// - `v`: 待归一化的向量（将被修改）
    pub fn normalize_in_place(&self, v: &mut [f32]) {
        let norm = self.l2_norm(v);
        if norm == 0.0 {
            return;
        }

        let inv_norm = 1.0 / norm;

        #[cfg(target_arch = "x86_64")]
        {
            if self.caps.has_avx2 {
                unsafe { self.mul_scalar_in_place_avx2(v, inv_norm) };
                return;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if self.caps.has_neon {
                unsafe { self.mul_scalar_in_place_neon(v, inv_norm) };
                return;
            }
        }

        for x in v.iter_mut() {
            *x *= inv_norm;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn mul_scalar_in_place_avx2(&self, v: &mut [f32], scalar: f32) {
        use std::arch::x86_64::*;

        let len = v.len();
        let chunks = len / 8;
        let remainder = len % 8;

        let scalar_vec = _mm256_set1_ps(scalar);

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(v.as_ptr().add(offset));
            let result = _mm256_mul_ps(va, scalar_vec);
            _mm256_storeu_ps(v.as_mut_ptr().add(offset), result);
        }

        for i in 0..remainder {
            v[len - remainder + i] *= scalar;
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn mul_scalar_in_place_neon(&self, v: &mut [f32], scalar: f32) {
        use std::arch::aarch64::*;

        let len = v.len();
        let chunks = len / 4;
        let remainder = len % 4;

        let scalar_vec = vdupq_n_f32(scalar);

        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(v.as_ptr().add(offset));
            let result = vmulq_f32(va, scalar_vec);
            vst1q_f32(v.as_mut_ptr().add(offset), result);
        }

        for i in 0..remainder {
            v[len - remainder + i] *= scalar;
        }
    }

    /// 计算向量的算术平均值
    ///
    /// # 参数
    /// - `v`: 输入向量
    ///
    /// # 返回
    /// 所有元素的平均值，空向量返回 0.0
    pub fn mean(&self, v: &[f32]) -> f32 {
        if v.is_empty() {
            return 0.0;
        }

        let sum = self.sum(v);
        sum / v.len() as f32
    }

    /// 计算向量所有元素的和
    ///
    /// 使用 SIMD 加速求和。
    ///
    /// # 参数
    /// - `v`: 输入向量
    ///
    /// # 返回
    /// Σ(v[i])
    pub fn sum(&self, v: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if self.caps.has_avx2 {
                return unsafe { self.sum_avx2(v) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if self.caps.has_neon {
                return unsafe { self.sum_neon(v) };
            }
        }

        v.iter().sum()
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn sum_avx2(&self, v: &[f32]) -> f32 {
        use std::arch::x86_64::*;

        let len = v.len();
        let chunks = len / 8;
        let remainder = len % 8;

        let mut sum = _mm256_setzero_ps();

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(v.as_ptr().add(offset));
            sum = _mm256_add_ps(sum, va);
        }

        let v_low = _mm256_castps256_ps128(sum);
        let vhigh = _mm256_extractf128_ps(sum, 1);
        let vsum128 = _mm_add_ps(v_low, vhigh);

        let mut temp = [0.0f32; 4];
        _mm_storeu_ps(temp.as_mut_ptr(), vsum128);
        let mut result = temp[0] + temp[1] + temp[2] + temp[3];

        for i in 0..remainder {
            result += v[len - remainder + i];
        }

        result
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn sum_neon(&self, v: &[f32]) -> f32 {
        use std::arch::aarch64::*;

        let len = v.len();
        let chunks = len / 4;
        let remainder = len % 4;

        let mut sum = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(v.as_ptr().add(offset));
            sum = vaddq_f32(sum, va);
        }

        let mut temp = [0.0f32; 4];
        vst1q_f32(temp.as_mut_ptr(), sum);
        let mut result = temp[0] + temp[1] + temp[2] + temp[3];

        for i in 0..remainder {
            result += v[len - remainder + i];
        }

        result
    }

    /// 向量标量乘法，返回新向量
    ///
    /// # 参数
    /// - `v`: 输入向量
    /// - `scalar`: 标量乘数
    ///
    /// # 返回
    /// 每个元素乘以 scalar 后的新向量
    pub fn mul_scalar(&self, v: &[f32], scalar: f32) -> Vec<f32> {
        #[cfg(target_arch = "x86_64")]
        {
            if self.caps.has_avx2 {
                return unsafe { self.mul_scalar_avx2(v, scalar) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if self.caps.has_neon {
                return unsafe { self.mul_scalar_neon(v, scalar) };
            }
        }

        v.iter().map(|&x| x * scalar).collect()
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn mul_scalar_avx2(&self, v: &[f32], scalar: f32) -> Vec<f32> {
        use std::arch::x86_64::*;

        let len = v.len();
        let mut result = vec![0.0f32; len];
        let chunks = len / 8;
        let remainder = len % 8;

        let vs = _mm256_set1_ps(scalar);

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(v.as_ptr().add(offset));
            let vr = _mm256_mul_ps(va, vs);
            _mm256_storeu_ps(result.as_mut_ptr().add(offset), vr);
        }

        for i in 0..remainder {
            result[len - remainder + i] = v[len - remainder + i] * scalar;
        }

        result
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn mul_scalar_neon(&self, v: &[f32], scalar: f32) -> Vec<f32> {
        use std::arch::aarch64::*;

        let len = v.len();
        let mut result = vec![0.0f32; len];
        let chunks = len / 4;
        let remainder = len % 4;

        let vs = vdupq_n_f32(scalar);

        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(v.as_ptr().add(offset));
            let vr = vmulq_f32(va, vs);
            vst1q_f32(result.as_mut_ptr().add(offset), vr);
        }

        for i in 0..remainder {
            result[len - remainder + i] = v[len - remainder + i] * scalar;
        }

        result
    }

    /// 逐元素向量加法，返回新向量
    ///
    /// # 参数
    /// - `a`: 第一个向量
    /// - `b`: 第二个向量（长度必须与 `a` 相等）
    ///
    /// # 返回
    /// 逐元素相加的结果向量
    ///
    /// # Panics
    /// 当两个向量长度不相等时 panic
    pub fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), b.len(), "向量长度必须相等");

        #[cfg(target_arch = "x86_64")]
        {
            if self.caps.has_avx2 {
                return unsafe { self.add_avx2(a, b) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if self.caps.has_neon {
                return unsafe { self.add_neon(a, b) };
            }
        }

        a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_avx2(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        use std::arch::x86_64::*;

        let len = a.len();
        let mut result = vec![0.0f32; len];
        let chunks = len / 8;
        let remainder = len % 8;

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            let vr = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(result.as_mut_ptr().add(offset), vr);
        }

        for i in 0..remainder {
            result[len - remainder + i] = a[len - remainder + i] + b[len - remainder + i];
        }

        result
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn add_neon(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        use std::arch::aarch64::*;

        let len = a.len();
        let mut result = vec![0.0f32; len];
        let chunks = len / 4;
        let remainder = len % 4;

        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));
            let vr = vaddq_f32(va, vb);
            vst1q_f32(result.as_mut_ptr().add(offset), vr);
        }

        for i in 0..remainder {
            result[len - remainder + i] = a[len - remainder + i] + b[len - remainder + i];
        }

        result
    }

    /// 批量计算查询向量与候选向量的余弦相似度
    ///
    /// # 参数
    /// - `query`: 查询向量
    /// - `candidates`: 候选向量列表
    ///
    /// # 返回
    /// 每个候选向量的相似度结果（含索引 ID 和分数）
    pub fn batch_cosine_similarity(
        &self,
        query: &[f32],
        candidates: &[Vec<f32>],
    ) -> Vec<SimilarityResult> {
        let query_norm = self.l2_norm(query);
        if query_norm == 0.0 {
            return candidates
                .iter()
                .enumerate()
                .map(|(id, _)| SimilarityResult { id, score: 0.0 })
                .collect();
        }

        candidates
            .iter()
            .enumerate()
            .map(|(id, candidate)| {
                self.compute_similarity_result(id, query, candidate, query_norm)
            })
            .collect()
    }

    /// 批量计算余弦相似度并返回 Top-K 结果（按分数降序）
    ///
    /// # 参数
    /// - `query`: 查询向量
    /// - `candidates`: 候选向量列表
    /// - `top_k`: 返回的最相似结果数量
    ///
    /// # 返回
    /// 按相似度降序排列的前 top_k 个结果
    pub fn batch_cosine_similarity_sorted(
        &self,
        query: &[f32],
        candidates: &[Vec<f32>],
        top_k: usize,
    ) -> Vec<SimilarityResult> {
        let mut results = self.batch_cosine_similarity(query, candidates);
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// 批量计算余弦相似度（使用自定义 ID）
    ///
    /// 与 `batch_cosine_similarity` 类似，但使用 `(id, vector)` 元组指定候选向量 ID。
    ///
    /// # 参数
    /// - `query`: 查询向量
    /// - `candidates`: 候选向量列表，每个元素为 (ID, 向量) 元组
    ///
    /// # 返回
    /// 每个候选向量的相似度结果
    pub fn batch_cosine_similarity_with_ids(
        &self,
        query: &[f32],
        candidates: &[(usize, Vec<f32>)],
    ) -> Vec<SimilarityResult> {
        let query_norm = self.l2_norm(query);
        if query_norm == 0.0 {
            return candidates
                .iter()
                .map(|(id, _)| SimilarityResult { id: *id, score: 0.0 })
                .collect();
        }

        candidates
            .iter()
            .map(|(id, candidate)| {
                self.compute_similarity_result(*id, query, candidate, query_norm)
            })
            .collect()
    }

    /// 批量计算余弦相似度（使用自定义 ID）并返回 Top-K 结果
    ///
    /// # 参数
    /// - `query`: 查询向量
    /// - `candidates`: 候选向量列表
    /// - `top_k`: 返回的最相似结果数量
    ///
    /// # 返回
    /// 按相似度降序排列的前 top_k 个结果
    pub fn batch_cosine_similarity_with_ids_sorted(
        &self,
        query: &[f32],
        candidates: &[(usize, Vec<f32>)],
        top_k: usize,
    ) -> Vec<SimilarityResult> {
        let mut results = self.batch_cosine_similarity_with_ids(query, candidates);
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    #[inline]
    fn compute_similarity_result(
        &self,
        id: usize,
        query: &[f32],
        candidate: &[f32],
        query_norm: f32,
    ) -> SimilarityResult {
        let norm = self.l2_norm(candidate);
        let score = if norm == 0.0 {
            0.0
        } else {
            self.dot_product(query, candidate) / (query_norm * norm)
        };
        SimilarityResult { id, score }
    }

    /// 计算两个向量之间的欧氏距离
    ///
    /// 公式：d(a, b) = √(Σ(a[i] - b[i])²)
    ///
    /// # 参数
    /// - `a`: 第一个向量
    /// - `b`: 第二个向量
    ///
    /// # 返回
    /// 两向量间的欧氏距离
    ///
    /// # Panics
    /// 当两个向量长度不相等时 panic
    pub fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "向量长度必须相等");

        #[cfg(target_arch = "x86_64")]
        {
            if self.caps.has_avx2 && self.caps.has_fma {
                return unsafe { self.euclidean_distance_avx2(a, b) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if self.caps.has_neon {
                return unsafe { self.euclidean_distance_neon(a, b) };
            }
        }

        self.euclidean_distance_scalar(a, b)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn euclidean_distance_avx2(&self, a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;

        let len = a.len();
        let chunks = len / 8;
        let remainder = len % 8;

        let mut sum = _mm256_setzero_ps();

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            let diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        let v_low = _mm256_castps256_ps128(sum);
        let vhigh = _mm256_extractf128_ps(sum, 1);
        let vsum128 = _mm_add_ps(v_low, vhigh);

        let mut temp = [0.0f32; 4];
        _mm_storeu_ps(temp.as_mut_ptr(), vsum128);
        let mut result = temp[0] + temp[1] + temp[2] + temp[3];

        for i in 0..remainder {
            let diff = a[len - remainder + i] - b[len - remainder + i];
            result += diff * diff;
        }

        result.sqrt()
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn euclidean_distance_neon(&self, a: &[f32], b: &[f32]) -> f32 {
        use std::arch::aarch64::*;

        let len = a.len();
        let chunks = len / 4;
        let remainder = len % 4;

        let mut sum = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));
            let diff = vsubq_f32(va, vb);
            sum = vfmaq_f32(sum, diff, diff);
        }

        let mut temp = [0.0f32; 4];
        vst1q_f32(temp.as_mut_ptr(), sum);
        let mut result = temp[0] + temp[1] + temp[2] + temp[3];

        for i in 0..remainder {
            let diff = a[len - remainder + i] - b[len - remainder + i];
            result += diff * diff;
        }

        result.sqrt()
    }

    fn euclidean_distance_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        let sum: f32 = a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum();
        sum.sqrt()
    }

    /// 计算多个 embedding 的均值向量
    ///
    /// 使用 SIMD 加速（AVX2/NEON）进行向量累加，
    /// 最后调用 `mul_scalar` 计算均值。
    ///
    /// # 参数
    /// - `embeddings`: embedding 向量切片
    ///
    /// # 返回
    /// 均值向量，如果输入为空则返回空向量
    pub fn compute_embedding_mean(&self, embeddings: &[Vec<f32>]) -> Vec<f32> {
        if embeddings.is_empty() {
            return Vec::new();
        }

        let dim = embeddings[0].len();
        let mut sum = vec![0.0f32; dim];

        #[cfg(target_arch = "x86_64")]
        {
            if self.caps.has_avx2 {
                unsafe { self.compute_embedding_mean_avx2(&embeddings, &mut sum) };
                let n = embeddings.len() as f32;
                return self.mul_scalar(&sum, 1.0 / n);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if self.caps.has_neon {
                unsafe { self.compute_embedding_mean_neon(&embeddings, &mut sum) };
                let n = embeddings.len() as f32;
                return self.mul_scalar(&sum, 1.0 / n);
            }
        }

        for embedding in embeddings {
            for (i, &val) in embedding.iter().enumerate() {
                sum[i] += val;
            }
        }

        let n = embeddings.len() as f32;
        self.mul_scalar(&sum, 1.0 / n)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn compute_embedding_mean_avx2(&self, embeddings: &[Vec<f32>], sum: &mut [f32]) {
        use std::arch::x86_64::*;

        let dim = sum.len();
        let chunks = dim / 8;
        let remainder = dim % 8;

        for embedding in embeddings {
            for i in 0..chunks {
                let offset = i * 8;
                let va = _mm256_loadu_ps(sum.as_ptr().add(offset));
                let vb = _mm256_loadu_ps(embedding.as_ptr().add(offset));
                let result = _mm256_add_ps(va, vb);
                _mm256_storeu_ps(sum.as_mut_ptr().add(offset), result);
            }

            for i in 0..remainder {
                sum[dim - remainder + i] += embedding[dim - remainder + i];
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn compute_embedding_mean_neon(&self, embeddings: &[Vec<f32>], sum: &mut [f32]) {
        use std::arch::aarch64::*;

        let dim = sum.len();
        let chunks = dim / 4;
        let remainder = dim % 4;

        for embedding in embeddings {
            for i in 0..chunks {
                let offset = i * 4;
                let va = vld1q_f32(sum.as_ptr().add(offset));
                let vb = vld1q_f32(embedding.as_ptr().add(offset));
                let result = vaddq_f32(va, vb);
                vst1q_f32(sum.as_mut_ptr().add(offset), result);
            }

            for i in 0..remainder {
                sum[dim - remainder + i] += embedding[dim - remainder + i];
            }
        }
    }

    /// 计算向量中的最大值
    ///
    /// # 参数
    /// - `v`: 输入向量
    ///
    /// # 返回
    /// 向量中的最大值，空向量返回 `-f32::INFINITY`
    pub fn max(&self, v: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if self.caps.has_avx2 {
                return unsafe { self.max_avx2(v) };
            }
        }

        v.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn max_avx2(&self, v: &[f32]) -> f32 {
        use std::arch::x86_64::*;

        let len = v.len();
        if len == 0 {
            return f32::NEG_INFINITY;
        }

        let chunks = len / 8;
        let remainder = len % 8;

        let mut max_val = _mm256_set1_ps(f32::NEG_INFINITY);

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(v.as_ptr().add(offset));
            max_val = _mm256_max_ps(max_val, va);
        }

        let v_low = _mm256_castps256_ps128(max_val);
        let vhigh = _mm256_extractf128_ps(max_val, 1);
        let vmax128 = _mm_max_ps(v_low, vhigh);

        let mut temp = [0.0f32; 4];
        _mm_storeu_ps(temp.as_mut_ptr(), vmax128);
        let mut result = temp[0].max(temp[1]).max(temp[2]).max(temp[3]);

        for i in 0..remainder {
            result = result.max(v[len - remainder + i]);
        }

        result
    }

    /// 计算向量中的最小值
    pub fn min(&self, v: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if self.caps.has_avx2 {
                return unsafe { self.min_avx2(v) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if self.caps.has_neon {
                return unsafe { self.min_neon(v) };
            }
        }

        v.iter().cloned().fold(f32::INFINITY, f32::min)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn min_avx2(&self, v: &[f32]) -> f32 {
        use std::arch::x86_64::*;

        let len = v.len();
        if len == 0 {
            return f32::INFINITY;
        }

        let chunks = len / 8;
        let remainder = len % 8;

        let mut min_val = _mm256_set1_ps(f32::INFINITY);

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(v.as_ptr().add(offset));
            min_val = _mm256_min_ps(min_val, va);
        }

        let v_low = _mm256_castps256_ps128(min_val);
        let vhigh = _mm256_extractf128_ps(min_val, 1);
        let vmin128 = _mm_min_ps(v_low, vhigh);

        let mut temp = [0.0f32; 4];
        _mm_storeu_ps(temp.as_mut_ptr(), vmin128);
        let mut result = temp[0].min(temp[1]).min(temp[2]).min(temp[3]);

        for i in 0..remainder {
            result = result.min(v[len - remainder + i]);
        }

        result
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn min_neon(&self, v: &[f32]) -> f32 {
        use std::arch::aarch64::*;

        let len = v.len();
        if len == 0 {
            return f32::INFINITY;
        }

        let chunks = len / 4;
        let remainder = len % 4;

        let mut min_val = vdupq_n_f32(f32::INFINITY);

        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(v.as_ptr().add(offset));
            min_val = vminq_f32(min_val, va);
        }

        let mut temp = [0.0f32; 4];
        vst1q_f32(temp.as_mut_ptr(), min_val);
        let mut result = temp[0].min(temp[1]).min(temp[2]).min(temp[3]);

        for i in 0..remainder {
            result = result.min(v[len - remainder + i]);
        }

        result
    }

    /// 计算 softmax 函数
    /// 
    /// 对输入向量进行 softmax 归一化，使用数值稳定实现
    pub fn softmax(&self, v: &[f32]) -> Vec<f32> {
        let max_val = self.max(v);
        let exp_vals: Vec<f32> = v.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp = self.sum(&exp_vals);
        self.mul_scalar(&exp_vals, 1.0 / sum_exp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_simd_capabilities_detect() {
        let caps = SimdCapabilities::detect();
        assert!(!caps.has_avx2 || !caps.has_neon);

        #[cfg(target_arch = "x86_64")]
        {
            assert!(caps.has_sse42 || caps.has_avx2 || caps.has_fma);
        }

        #[cfg(target_arch = "aarch64")]
        assert!(caps.has_neon);
    }

    #[test]
    fn test_dot_product_basic() {
        let ops = SimdVectorOps::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];

        let dot = ops.dot_product(&a, &b);
        assert!(approx_eq(dot, 40.0, 1e-5));
    }

    #[test]
    fn test_dot_product_large() {
        let ops = SimdVectorOps::new();
        let a: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..256).map(|i| (i + 1) as f32).collect();

        let dot = ops.dot_product(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        assert!(approx_eq(dot, expected, 1e-3));
    }

    #[test]
    fn test_l2_norm() {
        let ops = SimdVectorOps::new();
        let v = vec![3.0, 4.0];

        let norm = ops.l2_norm(&v);
        assert!(approx_eq(norm, 5.0, 1e-5));
    }

    #[test]
    fn test_cosine_similarity() {
        let ops = SimdVectorOps::new();
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![1.0, 1.0, 0.0];

        let sim_ab = ops.cosine_similarity(&a, &b);
        assert!(approx_eq(sim_ab, 0.0, 1e-5));

        let sim_ac = ops.cosine_similarity(&a, &c);
        assert!(approx_eq(sim_ac, 0.70710678, 1e-5));

        let sim_aa = ops.cosine_similarity(&a, &a);
        assert!(approx_eq(sim_aa, 1.0, 1e-5));
    }

    #[test]
    fn test_normalize() {
        let ops = SimdVectorOps::new();
        let v = vec![3.0, 4.0];

        let normalized = ops.normalize(&v);
        let norm = ops.l2_norm(&normalized);
        assert!(approx_eq(norm, 1.0, 1e-5));
    }

    #[test]
    fn test_normalize_zero_vector() {
        let ops = SimdVectorOps::new();
        let v = vec![0.0, 0.0, 0.0];

        let normalized = ops.normalize(&v);
        assert_eq!(normalized, v);
    }

    #[test]
    fn test_mean() {
        let ops = SimdVectorOps::new();
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mean = ops.mean(&v);
        assert!(approx_eq(mean, 3.0, 1e-5));
    }

    #[test]
    fn test_sum() {
        let ops = SimdVectorOps::new();
        let v: Vec<f32> = (1..=100).map(|i| i as f32).collect();

        let sum = ops.sum(&v);
        assert!(approx_eq(sum, 5050.0, 1e-3));
    }

    #[test]
    fn test_mul_scalar() {
        let ops = SimdVectorOps::new();
        let v = vec![1.0, 2.0, 3.0, 4.0];

        let result = ops.mul_scalar(&v, 2.0);
        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_add() {
        let ops = SimdVectorOps::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];

        let result = ops.add(&a, &b);
        assert_eq!(result, vec![3.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_batch_cosine_similarity() {
        let ops = SimdVectorOps::new();
        let query = vec![1.0, 0.0, 0.0];
        let candidates = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, 0.0],
        ];

        let results = ops.batch_cosine_similarity(&query, &candidates);

        assert_eq!(results.len(), 3);
        assert!(approx_eq(results[0].score, 1.0, 1e-5));
        assert!(approx_eq(results[1].score, 0.0, 1e-5));
        assert!(approx_eq(results[2].score, 0.70710678, 1e-5));
    }

    #[test]
    fn test_batch_cosine_similarity_sorted() {
        let ops = SimdVectorOps::new();
        let query = vec![1.0, 0.0, 0.0];
        let candidates = vec![
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.5, 0.5, 0.0],
        ];

        let results = ops.batch_cosine_similarity_sorted(&query, &candidates, 2);

        assert_eq!(results.len(), 2);
        assert!(approx_eq(results[0].score, 1.0, 1e-5));
        assert!(results[0].id == 1);
    }

    #[test]
    fn test_euclidean_distance() {
        let ops = SimdVectorOps::new();
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];

        let dist = ops.euclidean_distance(&a, &b);
        assert!(approx_eq(dist, 5.0, 1e-5));
    }

    #[test]
    fn test_compute_embedding_mean() {
        let ops = SimdVectorOps::new();
        let embeddings = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let mean = ops.compute_embedding_mean(&embeddings);
        assert_eq!(mean, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_softmax() {
        let ops = SimdVectorOps::new();
        let v = vec![1.0, 2.0, 3.0];

        let result = ops.softmax(&v);
        let sum: f32 = result.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-5));

        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_max() {
        let ops = SimdVectorOps::new();
        let v = vec![1.0, 5.0, 3.0, 9.0, 2.0];

        let max_val = ops.max(&v);
        assert!(approx_eq(max_val, 9.0, 1e-5));
    }

    #[test]
    fn test_min() {
        let ops = SimdVectorOps::new();
        let v = vec![5.0, 1.0, 3.0, 9.0, 2.0];

        let min_val = ops.min(&v);
        assert!(approx_eq(min_val, 1.0, 1e-5));
    }

    #[test]
    fn test_normalize_in_place() {
        let ops = SimdVectorOps::new();
        let mut v = vec![3.0, 4.0];

        ops.normalize_in_place(&mut v);
        let norm = ops.l2_norm(&v);
        assert!(approx_eq(norm, 1.0, 1e-5));
    }

    #[test]
    fn test_batch_cosine_similarity_with_ids() {
        let ops = SimdVectorOps::new();
        let query = vec![1.0, 0.0, 0.0];
        let candidates = vec![
            (100, vec![1.0, 0.0, 0.0]),
            (200, vec![0.0, 1.0, 0.0]),
            (300, vec![1.0, 1.0, 0.0]),
        ];

        let results = ops.batch_cosine_similarity_with_ids(&query, &candidates);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, 100);
        assert!(approx_eq(results[0].score, 1.0, 1e-5));
        assert_eq!(results[1].id, 200);
        assert!(approx_eq(results[1].score, 0.0, 1e-5));
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let ops = SimdVectorOps::new();
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];

        let sim = ops.cosine_similarity(&a, &b);
        assert!(approx_eq(sim, 0.0, 1e-5));
    }

    #[test]
    fn test_mean_empty() {
        let ops = SimdVectorOps::new();
        let v: Vec<f32> = vec![];

        let mean = ops.mean(&v);
        assert!(approx_eq(mean, 0.0, 1e-5));
    }

    #[test]
    fn test_compute_embedding_mean_empty() {
        let ops = SimdVectorOps::new();
        let embeddings: Vec<Vec<f32>> = vec![];

        let mean = ops.compute_embedding_mean(&embeddings);
        assert!(mean.is_empty());
    }

    #[test]
    fn test_similarity_result_ordering() {
        let r1 = SimilarityResult { id: 0, score: 0.5 };
        let r2 = SimilarityResult { id: 1, score: 0.8 };
        let r3 = SimilarityResult { id: 2, score: 0.3 };

        let mut results = vec![r1, r2, r3];
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        assert_eq!(results[0].id, 1);
        assert_eq!(results[1].id, 0);
        assert_eq!(results[2].id, 2);
    }

    #[test]
    fn test_large_vector_operations() {
        let ops = SimdVectorOps::new();
        let dim = 1024;
        let a: Vec<f32> = (0..dim).map(|i| (i as f32) / 100.0).collect();
        let b: Vec<f32> = (0..dim).map(|i| ((i + 1) as f32) / 100.0).collect();

        let dot = ops.dot_product(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        assert!(approx_eq(dot, expected, 1e-2));

        let norm_a = ops.l2_norm(&a);
        let expected_norm = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(approx_eq(norm_a, expected_norm, 1e-2));

        let sum = ops.sum(&a);
        let expected_sum: f32 = a.iter().sum();
        assert!(approx_eq(sum, expected_sum, 1e-2));
    }

    #[test]
    fn test_non_aligned_size() {
        let ops = SimdVectorOps::new();

        for size in [1, 3, 5, 7, 9, 11, 13, 15, 17, 31, 63, 127, 255] {
            let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

            let dot = ops.dot_product(&a, &b);
            let expected: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
            assert!(
                approx_eq(dot, expected, 1e-3),
                "Failed for size {}: {} vs {}",
                size,
                dot,
                expected
            );
        }
    }
}

#[cfg(test)]
/// 性能基准测试模块
pub mod benchmark {
    use super::*;
    use std::time::Instant;

    /// 点积性能基准测试
    /// 
    /// 返回 (SIMD 时间, 标量时间)
    pub fn benchmark_dot_product(dim: usize, iterations: usize) -> (f64, f64) {
        let ops = SimdVectorOps::new();
        let a: Vec<f32> = (0..dim).map(|i| (i as f32) / 100.0).collect();
        let b: Vec<f32> = (0..dim).map(|i| ((i + 1) as f32) / 100.0).collect();

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = ops.dot_product(&a, &b);
        }
        let simd_time = start.elapsed().as_secs_f64();

        let start = Instant::now();
        for _ in 0..iterations {
            let _: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        }
        let scalar_time = start.elapsed().as_secs_f64();

        (simd_time, scalar_time)
    }

    /// 余弦相似度性能基准测试
    /// 
    /// 返回 (SIMD 时间, 标量时间)
    pub fn benchmark_cosine_similarity(dim: usize, iterations: usize) -> (f64, f64) {
        let ops = SimdVectorOps::new();
        let a: Vec<f32> = (0..dim).map(|i| (i as f32) / 100.0).collect();
        let b: Vec<f32> = (0..dim).map(|i| ((i + 1) as f32) / 100.0).collect();

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = ops.cosine_similarity(&a, &b);
        }
        let simd_time = start.elapsed().as_secs_f64();

        let start = Instant::now();
        for _ in 0..iterations {
            let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
            let _ = if norm_a > 0.0 && norm_b > 0.0 {
                dot / (norm_a * norm_b)
            } else {
                0.0
            };
        }
        let scalar_time = start.elapsed().as_secs_f64();

        (simd_time, scalar_time)
    }
}
