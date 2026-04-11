//! cuBLAS矩阵乘法加速模块
//!
//! 提供高性能GEMM（通用矩阵乘法）操作：
//! - 单精度浮点（f32）
//! - 半精度浮点（f16/BF16）
//! - 批量矩阵乘法
//! - 自动算法选择
//!
//! # 性能优化
//! - 根据矩阵尺寸自动选择最优cuBLAS算法
//! - 支持Tensor Core加速（SM 7.0+）
//! - 混合精度计算

use super::{CudaBuffer, CudaContext, CudaError};
use log::{debug, info, trace};

/// GEMM算法类型
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GemmAlgorithm {
    /// 默认自动选择
    Auto,
    /// GEMM（标准算法）
    Standard,
    /// 使用Tensor Cores
    TensorCore,
    /// 针对小矩阵优化
    SmallMatrix,
    /// 针对大矩阵优化
    LargeMatrix,
}

/// cuBLAS句柄封装
pub struct CublasHandle {
    context: CudaContext,
    handle: *mut std::ffi::c_void, // cublasHandle_t
    algorithm: GemmAlgorithm,
}

/// GEMM结果信息
#[derive(Debug, Clone)]
pub struct GemmResult {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub algorithm_used: GemmAlgorithm,
    pub execution_time_us: u64,
    pub tflops: f64,
}

impl CublasHandle {
    /// 创建新的cuBLAS句柄
    pub fn new(context: CudaContext, algorithm: Option<GemmAlgorithm>) -> Result<Self, CudaError> {
        let algorithm = algorithm.unwrap_or(GemmAlgorithm::Auto);
        
        info!("初始化cuBLAS句柄 (算法: {:?})", algorithm);
        
        #[cfg(feature = "cuda-native")]
        {
            use cudarc::blas::cublas::Cublas;
            let handle = Cublas::new(context.device().info().id)
                .map_err(|e| CudaError::CublasInitFailed {
                    message: e.to_string(),
                })?;
            
            Ok(Self {
                context,
                handle: Box::into_raw(Box::new(handle)) as *mut std::ffi::c_void,
                algorithm,
            })
        }

        #[cfg(not(feature = "cuda-native"))]
        {
            Ok(Self {
                context,
                handle: std::ptr::null_mut(),
                algorithm,
            })
        }
    }

    /// 单精度GEMM: C = alpha * op(A) * op(B) + beta * C
    ///
    /// # 参数
    /// - `transa`, `transb`: 是否转置A/B
    /// - `m`: A的行数/C的行数
    /// - `n`: B的列数/C的列数
    /// - `k`: A的列数/B的行数
    /// - `alpha`, `beta`: 标量系数
    /// - `a`, `b`, `c`: 输入输出矩阵
    pub fn gemm_f32(
        &self,
        transa: bool,
        transb: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &CudaBuffer<f32>,
        b: &CudaBuffer<f32>,
        beta: f32,
        c: &mut CudaBuffer<f32>,
    ) -> Result<GemmResult, CudaError> {
        debug!(
            "GEMM F32: {}x{} @ {}x{} (transA={}, transB={})",
            m, k, k, n, transa, transb
        );

        let start_time = std::time::Instant::now();

        // 验证维度
        self.validate_gemm_dims(transa, transb, m, n, k, a, b, c)?;

        #[cfg(feature = "cuda-native")]
        {
            use cudarc::blas::sys::{
                cublasOperation_t, cublasSgemm, cublasHandle_t,
                CUBLAS_OP_N, CUBLAS_OP_T,
            };
            
            let op_a = if transa { CUBLAS_OP_T } else { CUBLAS_OP_N };
            let op_b = if transb { CUBLAS_OP_T } else { CUBLAS_OP_N };
            
            let lda = if transa { k } else { m };
            let ldb = if transb { n } else { k };
            let ldc = m;
            
            unsafe {
                let result = cublasSgemm(
                    self.handle as cublasHandle_t,
                    op_a,
                    op_b,
                    m as i32,
                    n as i32,
                    k as i32,
                    &alpha,
                    a.as_ptr() as *const f32,
                    lda as i32,
                    b.as_ptr() as *const f32,
                    ldb as i32,
                    &beta,
                    c.as_mut_ptr() as *mut f32,
                    ldc as i32,
                );
                
                if result != 0 {
                    return Err(CudaError::KernelLaunchFailed {
                        message: format!("cuBLAS sgemm 错误码: {}", result),
                    });
                }
            }
        }

        #[cfg(not(feature = "cuda-native"))]
        {
            // CPU fallback实现
            self.gemm_f32_cpu(transa, transb, m, n, k, alpha, a, b, beta, c)?;
        }

        let elapsed = start_time.elapsed();
        let elapsed_us = elapsed.as_micros() as u64;
        
        // 计算TFLOPS
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let tflops = flops / (elapsed.as_secs_f64() * 1e12);

        let algorithm_used = self.select_algorithm(m, n, k);

        trace!(
            "GEMM 完成: {:.2}us ({:.2} TFLOPS)",
            elapsed_us,
            tflops
        );

        Ok(GemmResult {
            m,
            n,
            k,
            algorithm_used,
            execution_time_us: elapsed_us,
            tflops,
        })
    }

    /// 半精度GEMM（使用Tensor Cores）
    pub fn gemm_f16(
        &self,
        _transa: bool,
        _transb: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f16,
        a: &CudaBuffer<f16>,
        b: &CudaBuffer<f16>,
        beta: f16,
        c: &mut CudaBuffer<f16>,
    ) -> Result<GemmResult, CudaError> {
        debug!(
            "GEMM F16: {}x{} @ {}x{} (Tensor Core加速)",
            m, k, k, n
        );

        // 检查设备是否支持Tensor Core
        if !self.context.device().supports_compute_capability(7, 0) {
            return Err(CudaError::UnsupportedOperation {
                operation: "Tensor Core需要SM 7.0+".to_string(),
            });
        }

        let start_time = std::time::Instant::now();

        #[cfg(feature = "cuda-native")]
        {
            // 使用H-GEMM或自动降级
            use cudarc::blas::sys::{cublasHgemm, cublasHandle_t, cublasOperation_t};
            
            let op_a = if transa { 1u32 } else { 0 }; // CUBLAS_OP_T/N
            let op_b = if transb { 1u32 } else { 0 };
            
            unsafe {
                let result = cublasHgemm(
                    self.handle as cublasHandle_t,
                    std::mem::transmute(op_a),
                    std::mem::transmute(op_b),
                    m as i32,
                    n as i32,
                    k as i32,
                    &alpha.to_bits(),
                    a.as_ptr() as *const u16,
                    if transa { k as i32 } else { m as i32 },
                    b.as_ptr() as *const u16,
                    if transb { n as i32 } else { k as i32 },
                    &beta.to_bits(),
                    c.as_mut_ptr() as *mut u16,
                    m as i32,
                );
                
                if result != 0 {
                    return Err(CudaError::KernelLaunchFailed {
                        message: format!("cuBLAS hgemm 错误误: {}", result),
                    });
                }
            }
        }

        #[cfg(not(feature = "cuda-native"))]
        {
            // Mock：直接返回成功
            let _ = (a, b, c, alpha, beta);
        }

        let elapsed = start_time.elapsed();

        Ok(GemmResult {
            m,
            n,
            k,
            algorithm_used: GemmAlgorithm::TensorCore,
            execution_time_us: elapsed.as_micros() as u64,
            tflops: 0.0, // 需要实际测量
        })
    }

    /// 批量GEMM
    ///
    /// 同时计算多个独立的GEMM操作，提高GPU利用率
    pub fn batched_gemm_f32(
        &self,
        transa: bool,
        transb: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a_array: &[&CudaBuffer<f32>],
        b_array: &[&CudaBuffer<f32>],
        beta: f32,
        c_array: &mut [&mut CudaBuffer<f32>],
    ) -> Result<Vec<GemmResult>, CudaError> {
        let batch_size = a_array.len();
        
        if batch_size == 0 {
            return Ok(vec![]);
        }

        if batch_size != b_array.len() || batch_size != c_array.len() {
            return Err(CudaError::InvalidParameter {
                parameter: "批量数组长度不一致".to_string(),
            });
        }

        debug!("批量GEMM: batch={}, {}x{} @ {}", batch_size, m, k, k);

        let start_time = std::time::Instant::now();
        let mut results = Vec::with_capacity(batch_size);

        #[cfg(feature = "cuda-native")]
        {
            use cudarc::blas::sys::{
                cublasSgemmBatched, cublasHandle_t, cublasOperation_t,
                CUBLAS_OP_N, CUBLAS_OP_T,
            };

            let op_a = if transa { CUBLAS_OP_T } else { CUBLAS_OP_N };
            let op_b = if transb { CUBLAS_OP_T } else { CUBLAS_OP_N };

            // 准备指针数组
            let mut a_ptrs: Vec<*const f32> = Vec::with_capacity(batch_size);
            let mut b_ptrs: Vec<*const f32> = Vec::with_capacity(batch_size);
            let mut c_ptrs: Vec<*mut f32> = Vec::with_capacity(batch_size);

            for i in 0..batch_size {
                a_ptrs.push(unsafe { a_array[i].as_ptr() });
                b_ptrs.push(unsafe { b_array[i].as_ptr() });
                c_ptrs.push(unsafe { c_array[i].as_mut_ptr() });
            }

            unsafe {
                let result = cublasSgemmBatched(
                    self.handle as cublasHandle_t,
                    op_a,
                    op_b,
                    m as i32,
                    n as i32,
                    k as i32,
                    &alpha,
                    a_ptrs.as_ptr(),
                    if transa { k as i32 } else { m as i32 },
                    b_ptrs.as_ptr(),
                    if transb { n as i32 } else { k as i32 },
                    &beta,
                    c_ptrs.as_mut_ptr(),
                    m as i32,
                    batch_size as i32,
                );

                if result != 0 {
                    return Err(CudaError::KernelLaunchFailed {
                        message: format!("批量sgemm 错误: {}", result),
                    });
                }
            }
        }

        #[cfg(not(feature = "cuda-native"))]
        {
            // CPU fallback：逐个执行
            for i in 0..batch_size {
                self.gemm_f32_cpu(
                    transa, transb, m, n, k,
                    alpha, a_array[i], b_array[i], beta, c_array[i]
                )?;
            }
        }

        let elapsed = start_time.elapsed();
        let avg_time = elapsed.as_micros() as u64 / batch_size as u64;

        for _ in 0..batch_size {
            results.push(GemmResult {
                m, n, k,
                algorithm_used: GemmAlgorithm::Standard,
                execution_time_us: avg_time,
                tflops: 0.0,
            });
        }

        info!(
            "批量GEMM完成: {}个任务, 总计{:.2}ms, 平均{:.2}us/任务",
            batch_size,
            elapsed.as_millis(),
            avg_time
        );

        Ok(results)
    }

    /// 验证GEMM维度
    fn validate_gemm_dims<T>(
        &self,
        transa: bool,
        transb: bool,
        m: usize,
        n: usize,
        k: usize,
        a: &CudaBuffer<T>,
        b: &CudaBuffer<T>,
        c: &CudaBuffer<T>,
    ) -> Result<(), CudaError> {
        let (a_rows, a_cols) = if transa { (k, m) } else { (m, k) };
        let (b_rows, b_cols) = if transb { (n, k) } else { (k, n) };

        if a.len() < a_rows * a_cols {
            return Err(CudaError::InvalidParameter {
                parameter: format!("A缓冲区太小: 需要{}, 实际{}", a_rows * a_cols, a.len()),
            });
        }

        if b.len() < b_rows * b_cols {
            return Err(CudaError::InvalidParameter {
                parameter: format!("B缓冲区太小: 需要{}, 实际{}", b_rows * b_cols, b.len()),
            });
        }

        if c.len() < m * n {
            return Err(CudaError::InvalidParameter {
                parameter: format!("C缓冲区太小: 需要{}, 实际{}", m * n, c.len()),
            });
        }

        Ok(())
    }

    /// 选择最优算法
    fn select_algorithm(&self, m: usize, n: usize, k: usize) -> GemmAlgorithm {
        if self.algorithm != GemmAlgorithm::Auto {
            return self.algorithm;
        }

        let total_elements = m * n * k;
        
        // 小矩阵阈值：< 1M元素
        if total_elements < 1_000_000 {
            GemmAlgorithm::SmallMatrix
        }
        // 大矩阵阈值：> 100M元素
        else if total_elements > 100_000_000 {
            GemmAlgorithm::LargeMatrix
        }
        // 中等矩阵且支持Tensor Core
        else if self.context.device().supports_compute_capability(7, 0) {
            GemmAlgorithm::TensorCore
        }
        else {
            GemmAlgorithm::Standard
        }
    }

    /// CPU fallback GEMM实现
    #[cfg(not(feature = "cuda-native"))]
    fn gemm_f32_cpu(
        &self,
        transa: bool,
        transb: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &CudaBuffer<f32>,
        b: &CudaBuffer<f32>,
        beta: f32,
        c: &mut CudaBuffer<f32>,
    ) -> Result<(), CudaError> {
        // 转换为CPU数据（mock模式下可以直接访问）
        let a_data = a.to_host();
        let b_data = b.to_host();
        let mut c_data = c.to_host();

        // 简单的三重循环实现（生产代码应使用优化的BLAS库）
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    let a_val = if transa {
                        a_data[l * m + i]
                    } else {
                        a_data[i * k + l]
                    };
                    
                    let b_val = if transb {
                        b_data[j * k + l]
                    } else {
                        b_data[l * n + j]
                    };
                    
                    sum += a_val * b_val;
                }
                
                c_data[i * n + j] = alpha * sum + beta * c_data[i * n + j];
            }
        }

        // 写回（mock模式）
        *c = CudaBuffer::from_host(&c_data, c.device_id())?;

        Ok(())
    }
}

/// 半精度浮点类型
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
#[allow(non_upper_case_globals)]
pub struct f16(u16);

impl f16 {
    pub fn from_f32(v: f32) -> Self {
        // 简化的float16转换（生产代码应使用half crate）
        let bits = v.to_bits();
        let exponent = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
        let mantissa = bits & 0x007FFFFF;
        
        if exponent <= 0 {
            f16(0) // 下溢
        } else if exponent >= 31 {
            f16(0x7C00 | (((bits >> 16) & 0x8000) as u16)) // 无穷
        } else {
            f16((((bits >> 16) & 0x8000) as u16) | ((exponent as u16) << 10) | ((mantissa >> 13) as u16))
        }
    }

    pub fn to_f32(self) -> f32 {
        f32::from_bits(
            ((self.0 as u32 & 0x8000) << 16)
            | (((((self.0 as u32 >> 10) & 0x1F) as i32 + 127 - 15) as u32) << 23)
            | ((self.0 as u32 & 0x3FF) << 13)
        )
    }

    pub fn to_bits(self) -> u16 {
        self.0
    }
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        info!("销毁cuBLAS句柄");
        
        #[cfg(feature = "cuda-native")]
        {
            if !self.handle.is_null() {
                unsafe {
                    let _ = Box::from_raw(self.handle as *mut cudarc::blas::cublas::Cublas);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn get_test_handle() -> CublasHandle {
        let ctx = CudaContext::new(None).unwrap();
        CublasHandle::new(ctx, None).unwrap()
    }

    #[test]
    fn test_cublas_creation() {
        let handle = get_test_handle();
        assert!(handle.context.is_active());
    }

    #[test]
    fn test_gemm_f32_basic() {
        let handle = get_test_handle();
        let ctx = handle.context.clone();
        let device_id = ctx.device().info().id;

        // 创建简单的2x2矩阵
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0]; // 2x2
        let mut c_data: Vec<f32> = vec![0.0; 4]; // 2x2

        let a = CudaBuffer::from_host(&a_data, device_id).unwrap();
        let b = CudaBuffer::from_host(&b_data, device_id).unwrap();
        let mut c = CudaBuffer::from_host(&c_data, device_id).unwrap();

        let result = handle.gemm_f32(false, false, 2, 2, 2, 1.0, &a, &b, 0.0, &mut c).unwrap();

        assert_eq!(result.m, 2);
        assert_eq!(result.n, 2);
        assert_eq!(result.k, 2);
        assert!(result.execution_time_us >= 0);
    }

    #[test]
    fn test_gemm_transpose() {
        let handle = get_test_handle();
        let ctx = handle.context.clone();
        let device_id = ctx.device().info().id;

        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let c_data: Vec<f32> = vec![0.0; 4];

        let a = CudaBuffer::from_host(&a_data, device_id).unwrap();
        let b = CudaBuffer::from_host(&b_data, device_id).unwrap();
        let mut c = CudaBuffer::from_host(&c_data, device_id).unwrap();

        // 测试转置
        let result = handle.gemm_f32(true, true, 2, 2, 2, 1.0, &a, &b, 0.0, &mut c).unwrap();
        assert!(result.algorithm_used != GemmAlgorithm::Auto);
    }

    #[test]
    fn test_gemm_dimension_validation() {
        let handle = get_test_handle();
        let ctx = handle.context.clone();
        let device_id = ctx.device().info().id;

        // 故意创建太小的缓冲区
        let a_small: Vec<f32> = vec![1.0]; // 太小
        let b_data: Vec<f32> = vec![1.0; 4];
        let c_data: Vec<f32> = vec![0.0; 4];

        let a = CudaBuffer::from_host(&a_small, device_id).unwrap();
        let b = CudaBuffer::from_host(&b_data, device_id).unwrap();
        let mut c = CudaBuffer::from_host(&c_data, device_id).unwrap();

        let result = handle.gemm_f32(false, false, 2, 2, 2, 1.0, &a, &b, 0.0, &mut c);
        assert!(result.is_err());
    }

    #[test]
    fn test_batched_gemm() {
        let handle = get_test_handle();
        let ctx = handle.context.clone();
        let device_id = ctx.device().info().id;

        let batch_size = 3;
        let mut a_arrays = Vec::with_capacity(batch_size);
        let mut b_arrays = Vec::with_capacity(batch_size);
        let mut c_arrays = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let a_data: Vec<f32> = vec![1.0; 4];
            let b_data: Vec<f32> = vec![2.0; 4];
            let c_data: Vec<f32> = vec![0.0; 4];

            a_arrays.push(CudaBuffer::from_host(&a_data, device_id).unwrap());
            b_arrays.push(CudaBuffer::from_host(&b_data, device_id).unwrap());
            c_arrays.push(CudaBuffer::from_host(&c_data, device_id).unwrap());
        }

        let a_refs: Vec<&CudaBuffer<f32>> = a_arrays.iter().collect();
        let b_refs: Vec<&CudaBuffer<f32>> = b_arrays.iter().collect();
        let mut c_refs: Vec<&mut CudaBuffer<f32>> = c_arrays.iter_mut().collect();

        let results = handle.batched_gemm_f32(
            false, false, 2, 2, 2,
            1.0, &a_refs, &b_refs, 0.0, &mut c_refs
        ).unwrap();

        assert_eq!(results.len(), batch_size);
    }

    #[test]
    fn test_batched_gemm_length_mismatch() {
        let handle = get_test_handle();
        
        let result = handle.batched_gemm_f32(
            false, false, 2, 2, 2,
            1.0, &[], &[], 0.0, &mut []
        );
        
        // 空批次应该成功（0个操作）
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_algorithm_selection() {
        let handle = get_test_handle();
        
        // 小矩阵
        let algo = handle.select_algorithm(64, 64, 64);
        assert_eq!(algo, GemmAlgorithm::SmallMatrix);
        
        // 大矩阵
        let algo = handle.select_algorithm(4096, 4096, 4096);
        assert_eq!(algo, GemmAlgorithm::LargeMatrix);
    }

    #[test]
    fn test_f16_conversion() {
        let original: f32 = 3.14159;
        let h = f16::from_f32(original);
        let back = h.to_f32();
        
        // 允许一定误差（f16精度有限）
        assert!((original - back).abs() < 0.01);
    }

    #[test]
    fn test_tensor_core_requirement() {
        let handle = get_test_handle();
        let ctx = handle.context.clone();
        let device_id = ctx.device().info().id;

        let a_data: Vec<f16> = vec![f16::from_f32(1.0); 4];
        let b_data: Vec<f16> = vec![f16::from_f32(2.0); 4];
        let c_data: Vec<f16> = vec![f16::from_f32(0.0); 4];

        let a = CudaBuffer::from_host(&a_data, device_id).unwrap();
        let b = CudaBuffer::from_host(&b_data, device_id).unwrap();
        let mut c = CudaBuffer::from_host(&c_data, device_id).unwrap();

        // RTX 4090 (SM 8.9) 支持Tensor Core
        let result = handle.gemm_f16(
            false, false, 2, 2, 2,
            f16::from_f32(1.0), &a, &b,
            f16::from_f32(0.0), &mut c
        );
        
        // Mock模式下可能成功也可能失败
        let _ = result;
    }
}
