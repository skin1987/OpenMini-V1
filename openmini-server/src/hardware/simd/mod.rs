//! SIMD 抽象层
//!
//! 提供跨平台的 SIMD 加速接口，支持：
//! - x86-64: SSE4.2, AVX, AVX2, AVX-512
//! - ARM: NEON, SVE, SVE2
//! - 国产: LSX, LASX (龙芯)

#![allow(dead_code)]

// ============================================================================
// SIMD 能力检测
// ============================================================================

/// SIMD 能力级别
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SimdLevel {
    /// 无 SIMD 支持
    None,
    /// 128-bit SIMD (SSE4.2 / NEON)
    Level128,
    /// 256-bit SIMD (AVX2 / SVE)
    Level256,
    /// 512-bit SIMD (AVX-512)
    Level512,
}

impl Default for SimdLevel {
    fn default() -> Self {
        Self::detect()
    }
}

impl SimdLevel {
    /// 检测当前 CPU 的 SIMD 能力
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                Self::Level512
            } else if is_x86_feature_detected!("avx2") {
                Self::Level256
            } else if is_x86_feature_detected!("sse4.2") {
                Self::Level128
            } else {
                Self::None
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON 在 AArch64 上总是可用
            #[cfg(target_feature = "sve")]
            {
                Self::Level256
            }
            #[cfg(not(target_feature = "sve"))]
            {
                Self::Level128
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::None
        }
    }

    /// 获取 SIMD 宽度（位）
    pub fn width_bits(&self) -> usize {
        match self {
            Self::None => 0,
            Self::Level128 => 128,
            Self::Level256 => 256,
            Self::Level512 => 512,
        }
    }

    /// 获取 SIMD 宽度（f32 元素数）
    pub fn width_f32(&self) -> usize {
        self.width_bits() / 32
    }
}

// ============================================================================
// SIMD 操作 Trait
// ============================================================================

/// SIMD 向量操作 Trait
pub trait SimdOps: Send + Sync {
    /// 返回 SIMD 实现的名称
    fn name(&self) -> &'static str;

    /// 向量加法
    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32>;

    /// 向量乘法
    fn mul(&self, a: &[f32], b: &[f32]) -> Vec<f32>;

    /// 向量乘标量
    fn mul_scalar(&self, a: &[f32], scalar: f32) -> Vec<f32>;

    /// 向量减法
    fn sub(&self, a: &[f32], b: &[f32]) -> Vec<f32>;

    /// 向量除法
    fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32>;

    /// 向量点积
    fn dot(&self, a: &[f32], b: &[f32]) -> f32;

    /// 向量求和
    fn sum(&self, a: &[f32]) -> f32;

    /// 向量最大值
    fn max(&self, a: &[f32]) -> f32;

    /// 向量最小值
    fn min(&self, a: &[f32]) -> f32;

    /// Softmax
    fn softmax(&self, a: &[f32]) -> Vec<f32>;

    /// ReLU
    fn relu(&self, a: &[f32]) -> Vec<f32>;

    /// SiLU (Swish)
    fn silu(&self, a: &[f32]) -> Vec<f32>;

    // =========================================================================
    // 融合算子 (Fused Operators)
    // =========================================================================

    /// 矩阵乘法 + ReLU 融合 (Gemm + ReLU)
    ///
    /// 计算: output = relu(input @ weight + bias)
    /// 融合优势: 减少一次中间结果的内存读写
    ///
    /// # 参数
    /// - `input`: 输入矩阵 [M x K]
    /// - `weight`: 权重矩阵 [K x N]
    /// - `bias`: 偏置向量 [N]
    /// - `m`: 输出行数 M
    /// - `k`: 输入维度 K
    /// - `n`: 输出维度 N
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

    /// 矩阵乘法 + SiLU 融合 (Gemm + SiLU)
    ///
    /// 计算: output = silu(input @ weight)
    /// 融合优势: 减少一次中间结果的内存读写
    fn fused_gemm_silu(
        &self,
        input: &[f32],
        weight: &[f32],
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
                output[i * n + j] = sum / (1.0 + (-sum).exp());
            }
        }
        output
    }

    /// 矩阵乘法 + Add 融合 (Gemm + Add，用于残差连接)
    ///
    /// 计算: output = input @ weight + residual
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

    /// 矩阵乘法 + Softmax 融合 (用于 Attention)
    ///
    /// 计算: output = softmax(query @ key^T * scale) @ value
    ///
    /// # 参数
    /// - `query`: 查询矩阵 [M x K]
    /// - `key`: 键矩阵 [K x N] (已转置存储)
    /// - `value`: 值矩阵 [N x head_dim]
    /// - `scale`: 缩放因子
    /// - `m`: 查询数量
    /// - `k`: 头维度
    /// - `n`: 键值数量
    ///
    /// # 返回
    /// 输出矩阵 [M x head_dim]
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
}

// ============================================================================
// 标量实现（回退）
// ============================================================================

/// 标量实现（无 SIMD）
pub struct ScalarOps;

impl SimdOps for ScalarOps {
    fn name(&self) -> &'static str {
        "scalar"
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
    }

    fn mul(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
    }

    fn mul_scalar(&self, a: &[f32], scalar: f32) -> Vec<f32> {
        a.iter().map(|&x| x * scalar).collect()
    }

    fn sub(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
    }

    fn div(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x / y).collect()
    }

    fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    fn sum(&self, a: &[f32]) -> f32 {
        a.iter().sum()
    }

    fn max(&self, a: &[f32]) -> f32 {
        a.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    fn min(&self, a: &[f32]) -> f32 {
        a.iter().cloned().fold(f32::INFINITY, f32::min)
    }

    fn softmax(&self, a: &[f32]) -> Vec<f32> {
        let max_val = self.max(a);
        let exp_vals: Vec<f32> = a.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        exp_vals.iter().map(|&x| x / sum_exp).collect()
    }

    fn relu(&self, a: &[f32]) -> Vec<f32> {
        a.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect()
    }

    fn silu(&self, a: &[f32]) -> Vec<f32> {
        a.iter().map(|&x| x / (1.0 + (-x).exp())).collect()
    }
}

// ============================================================================
// SIMD 工厂函数
// ============================================================================

/// 创建最优 SIMD 操作实例
pub fn create_simd_ops() -> Box<dyn SimdOps> {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "nightly_avx512")]
        {
            if is_x86_feature_detected!("avx512f") {
                return Box::new(Avx512Ops);
            }
        }

        if is_x86_feature_detected!("avx2") || is_x86_feature_detected!("sse4.2") {
            Box::new(PackedSimdOps)
        } else {
            Box::new(ScalarOps)
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        Box::new(NeonOps)
    }

    #[cfg(target_arch = "loongarch64")]
    {
        Box::new(LasxOps)
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "loongarch64"
    )))]
    {
        Box::new(ScalarOps)
    }
}

// ============================================================================
// x86-64 SIMD 实现
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub use x86::{Avx2Ops, SseOps};

#[cfg(all(target_arch = "x86_64", feature = "nightly_avx512"))]
pub use x86::Avx512Ops;

// ============================================================================
// ARM SIMD 实现
// ============================================================================

#[cfg(target_arch = "aarch64")]
mod arm;

#[cfg(target_arch = "aarch64")]
pub use arm::NeonOps;

// ============================================================================
// 国产平台 SIMD 实现
// ============================================================================

mod faster_impl;
mod native;

#[cfg(target_arch = "loongarch64")]
pub use native::{LasxOps, LsxOps};

#[cfg(all(target_arch = "aarch64", not(target_feature = "sve")))]
pub use native::PhytiumOps;

#[cfg(all(target_arch = "aarch64", target_feature = "sve"))]
pub use native::SveOps;

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub use native::HygonOps;

pub use faster_impl::PackedSimdOps;

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_level_detect() {
        let level = SimdLevel::detect();
        // 至少应该有某种 SIMD 支持
        #[cfg(target_arch = "x86_64")]
        assert!(level >= SimdLevel::Level128);

        #[cfg(target_arch = "aarch64")]
        assert_eq!(level, SimdLevel::Level128);
    }

    #[test]
    fn test_scalar_ops() {
        let ops = ScalarOps;

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];

        let sum = ops.add(&a, &b);
        assert_eq!(sum, vec![3.0, 5.0, 7.0, 9.0]);

        let product = ops.mul(&a, &b);
        assert_eq!(product, vec![2.0, 6.0, 12.0, 20.0]);

        let dot = ops.dot(&a, &b);
        assert!((dot - 40.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax() {
        let ops = ScalarOps;
        let a = vec![1.0, 2.0, 3.0];
        let result = ops.softmax(&a);

        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_create_simd_ops() {
        let ops = create_simd_ops();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];

        let sum = ops.add(&a, &b);
        assert_eq!(sum.len(), 4);
    }

    // ==================== 新增分支覆盖测试 ====================

    /// 测试 SimdLevel 各变体的 width_bits() 分支
    #[test]
    fn test_simd_level_width_bits_all_variants() {
        // 覆盖 None -> 0
        assert_eq!(SimdLevel::None.width_bits(), 0);
        // 覆盖 Level128 -> 128
        assert_eq!(SimdLevel::Level128.width_bits(), 128);
        // 覆盖 Level256 -> 256
        assert_eq!(SimdLevel::Level256.width_bits(), 256);
        // 覆盖 Level512 -> 512
        assert_eq!(SimdLevel::Level512.width_bits(), 512);
    }

    /// 测试 SimdLevel 各变体的 width_f32() 分支（依赖 width_bits）
    #[test]
    fn test_simd_level_width_f32_all_variants() {
        assert_eq!(SimdLevel::None.width_f32(), 0); // 0/32=0
        assert_eq!(SimdLevel::Level128.width_f32(), 4); // 128/32=4
        assert_eq!(SimdLevel::Level256.width_f32(), 8); // 256/32=8
        assert_eq!(SimdLevel::Level512.width_f32(), 16); // 512/32=16
    }

    /// 测试 SimdLevel 的 Default trait 实现（调用 detect()）
    #[test]
    fn test_simd_level_default() {
        let level = SimdLevel::default();
        // Default 应该调用 detect()，结果一致
        assert_eq!(level, SimdLevel::detect());
    }

    /// 测试 SimdLevel 的 Ord/PartialOrd 排序特性
    #[test]
    fn test_simd_level_ordering() {
        assert!(SimdLevel::None < SimdLevel::Level128);
        assert!(SimdLevel::Level128 < SimdLevel::Level256);
        assert!(SimdLevel::Level256 < SimdLevel::Level512);
        assert!(SimdLevel::None <= SimdLevel::None);
    }

    /// 测试 ScalarOps 的 sub/div/max/min 完整方法覆盖
    #[test]
    fn test_scalar_ops_arithmetic_full() {
        let ops = ScalarOps;
        let a = vec![10.0, 5.0, 3.0];
        let b = vec![3.0, 2.0, 1.0];

        // sub: 向量减法
        let diff = ops.sub(&a, &b);
        assert_eq!(diff, vec![7.0, 3.0, 2.0]);

        // div: 向量除法
        let quotient = ops.div(&a, &b);
        assert_eq!(quotient, vec![10.0 / 3.0, 2.5, 3.0]);

        // max: 向量最大值
        assert!((ops.max(&a) - 10.0).abs() < 1e-6);

        // min: 向量最小值
        assert!((ops.min(&a) - 3.0).abs() < 1e-6);

        // sum: 向量求和
        assert!((ops.sum(&a) - 18.0).abs() < 1e-6);
    }

    /// 测试 ScalarOps 的 mul_scalar 方法
    #[test]
    fn test_scalar_ops_mul_scalar() {
        let ops = ScalarOps;
        let a = vec![1.0, 2.0, 3.0, 4.0];

        // 正标量
        let result = ops.mul_scalar(&a, 2.0);
        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);

        // 零标量
        let zero_result = ops.mul_scalar(&a, 0.0);
        assert_eq!(zero_result, vec![0.0, 0.0, 0.0, 0.0]);

        // 负标量
        let neg_result = ops.mul_scalar(&a, -1.0);
        assert_eq!(neg_result, vec![-1.0, -2.0, -3.0, -4.0]);
    }

    /// 测试 ScalarOps 的 relu 和 silu 激活函数
    #[test]
    fn test_scalar_ops_activations() {
        let ops = ScalarOps;
        let a = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

        // ReLU: max(0, x)
        let relu_result = ops.relu(&a);
        assert_eq!(relu_result, vec![0.0, 0.0, 0.0, 1.0, 2.0]);

        // SiLU: x * sigmoid(x)
        let silu_result = ops.silu(&a);
        assert_eq!(silu_result.len(), 5);
        // SiLU(0) = 0 * sigmoid(0) = 0
        assert!((silu_result[2] - 0.0).abs() < 1e-6);
        // SiLU 对正负输入有不同行为
        assert!(silu_result[3] > 0.0); // 正数输入
        assert!(silu_result[0] < 0.0); // 负数输入
    }

    /// 测试空输入向量的边界条件
    #[test]
    fn test_scalar_ops_empty_input() {
        let ops = ScalarOps;
        let empty: Vec<f32> = Vec::new();

        // 空向量运算应返回空向量或安全默认值
        assert_eq!(ops.add(&empty, &empty), Vec::<f32>::new());
        assert_eq!(ops.mul(&empty, &empty), Vec::<f32>::new());
        assert_eq!(ops.relu(&empty), Vec::<f32>::new());
        assert_eq!(ops.softmax(&empty), Vec::<f32>::new());
        assert!((ops.sum(&empty) - 0.0).abs() < 1e-6);
    }

    /// 测试单元素向量的边界条件
    #[test]
    fn test_scalar_ops_single_element() {
        let ops = ScalarOps;
        let a = vec![5.0];
        let b = vec![3.0];

        assert_eq!(ops.add(&a, &b), vec![8.0]);
        assert_eq!(ops.mul(&a, &b), vec![15.0]);
        assert!((ops.dot(&a, &b) - 15.0).abs() < 1e-6);

        // 单元素 softmax 结果应为 [1.0]
        let sm = ops.softmax(&a);
        assert_eq!(sm, vec![1.0]);
    }

    /// 测试 fused_gemm_relu 融合算子（小矩阵，避免大内存分配）
    #[test]
    fn test_fused_gemm_relu_small() {
        let ops = ScalarOps;
        // m=2, k=2, n=2 => 输出 4 个 f32，远小于 20KB
        let input = vec![1.0, 2.0, 3.0, 4.0]; // [2x2]
        let weight = vec![0.5, 0.5, 0.5, 0.5]; // [2x2]
        let bias = vec![0.1, 0.1]; // [2]

        let output = ops.fused_gemm_relu(&input, &weight, &bias, 2, 2, 2);
        assert_eq!(output.len(), 4);
        // 验证 ReLU 行为：所有输出应该 >= 0
        for &v in &output {
            assert!(v >= 0.0, "ReLU output should be non-negative, got {}", v);
        }
    }

    /// 测试 fused_gemm_softmax 的零维度边界（提前返回分支）
    #[test]
    fn test_fused_gemm_softmax_zero_dims() {
        let ops = ScalarOps;
        let query = vec![1.0];
        let key = vec![1.0];
        let value = vec![1.0];

        // m=0 时应返回空向量（覆盖 head_dim==0 || m==0 分支）
        let output = ops.fused_gemm_softmax(&query, &key, &value, 0.1, 0, 1, 1);
        assert_eq!(output, vec![0.0; 0]); // m*head_dim = 0

        // n=0 时也应返回空向量
        let output2 = ops.fused_gemm_softmax(&query, &key, &value, 0.1, 1, 1, 0);
        assert!(output2.is_empty());
    }

    // ==================== 新增测试：达到 20+ 覆盖率 ====================

    /// 测试：fused_gemm_silu 融合算子（覆盖第163-175行）
    #[test]
    fn test_fused_gemm_silu_small() {
        let ops = ScalarOps;
        // 小矩阵: m=2, k=2, n=2
        let input = vec![1.0, 2.0, 3.0, 4.0]; // [2x2]
        let weight = vec![0.5, 0.5, 0.5, 0.5]; // [2x2]

        let output = ops.fused_gemm_silu(&input, &weight, 2, 2, 2);
        assert_eq!(output.len(), 4); // m*n = 4

        // SiLU 输出应该在 [0, 1) 范围内或接近
        for &v in &output {
            // SiLU 可以是负数，但应该是有限值
            assert!(v.is_finite(), "SiLU输出应为有限值，got {}", v);
        }
    }

    /// 测试：fused_gemm_add 融合算子 - 残差连接（覆盖第180-192行）
    #[test]
    fn test_fused_gemm_add_small() {
        let ops = ScalarOps;
        // m=2, k=2, n=2
        let input = vec![1.0, 2.0, 3.0, 4.0]; // [2x2]
        let weight = vec![0.1, 0.2, 0.3, 0.4]; // [2x2]
        let residual = vec![0.5, 0.5, 0.5, 0.5]; // [2x2] 残差

        let output = ops.fused_gemm_add(&input, &weight, &residual, 2, 2, 2);
        assert_eq!(output.len(), 4);

        // 验证残差被正确加到结果上
        // 手动计算第一个元素: (1*0.1 + 2*0.3) + 0.5 = 0.7 + 0.5 = 1.2
        assert!((output[0] - 1.2).abs() < 0.01, "残差连接计算错误");
    }

    /// 测试：ScalarOps::div 除零安全性行为（Rust f32 除零返回 inf 或 NaN）
    #[test]
    fn test_scalar_ops_div_by_zero() {
        let ops = ScalarOps;
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 0.0, 0.0]; // 全零

        let result = ops.div(&a, &b);

        // Rust f32 / 0.0 = ±inf（取决于符号），不应panic
        assert_eq!(result.len(), 3);
        for &v in &result {
            // 应该是无穷大或NaN，不是正常有限值
            assert!(
                v.is_infinite() || v.is_nan() || v == f32::INFINITY || v == f32::NEG_INFINITY,
                "除零应产生特殊浮点值"
            );
        }
    }

    /// 测试：softmax 数值稳定性 - 大输入值（避免溢出）
    #[test]
    fn test_softmax_large_values() {
        let ops = ScalarOps;
        // 大数值（接近 f32::MAX 会导致溢出）
        let large_vals = vec![1000.0, 1001.0, 1002.0];

        let result = ops.softmax(&large_vals);

        // 验证结果和为1.0（概率分布）
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax结果和应为1.0");

        // 验证所有值都在 (0, 1) 范围内
        for &v in &result {
            assert!(v > 0.0 && v < 1.0, "softmax输出应在(0,1)范围内");
        }

        // 最大值应对应最大概率
        let max_idx = result
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(max_idx, 2, "最大输入值应有最大概率");
    }

    /// 测试：softmax 数值稳定性 - 相同输入值（均匀分布）
    #[test]
    fn test_softmax_uniform_input() {
        let ops = ScalarOps;
        let uniform_vals = vec![1.0, 1.0, 1.0, 1.0];

        let result = ops.softmax(&uniform_vals);

        // 所有值相同，应该得到均匀分布
        let expected_prob = 0.25; // 1/4
        for &v in &result {
            assert!(
                (v - expected_prob).abs() < 1e-5,
                "均匀输入应产生均匀分布，expected {} got {}",
                expected_prob,
                v
            );
        }
    }

    /// 测试：create_simd_ops 返回实例的方法调用完整性
    #[test]
    fn test_create_simd_ops_full_interface() {
        let ops = create_simd_ops();

        // 验证返回的实例实现了所有 SimdOps trait 方法
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        // name 不应 panic
        let name = ops.name();
        assert!(!name.is_empty());

        // 基本运算
        assert_eq!(ops.add(&a, &b).len(), 3);
        assert_eq!(ops.mul(&a, &b).len(), 3);
        assert_eq!(ops.sub(&a, &b).len(), 3);

        // 激活函数
        assert_eq!(ops.relu(&a).len(), 3);
        assert_eq!(ops.silu(&a).len(), 3);
        assert_eq!(ops.softmax(&a).len(), 3);
    }

    /// 测试：SimdLevel 枚举的 PartialEq 和 Eq 特性
    #[test]
    fn test_simd_level_equality() {
        // 验证相等性比较
        assert_eq!(SimdLevel::None, SimdLevel::None);
        assert_eq!(SimdLevel::Level128, SimdLevel::Level128);
        assert_ne!(SimdLevel::None, SimdLevel::Level128);

        // 验证可以用于 HashMap key 或 HashSet（需要 Eq + Hash）
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(SimdLevel::None);
        set.insert(SimdLevel::Level128);
        assert_eq!(set.len(), 2);

        // 插入重复值不会增加数量
        set.insert(SimdLevel::None);
        assert_eq!(set.len(), 2);
    }
}
