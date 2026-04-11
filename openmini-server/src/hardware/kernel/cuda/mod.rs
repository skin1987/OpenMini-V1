//! CUDA GPU 加速模块 - 上下文管理与设备抽象
//!
//! 提供CUDA设备发现、上下文管理、RAII资源管理等核心功能。
//! 支持多GPU环境，自动选择最优设备。
//!
//! # 性能目标
//! - TTFT < 100ms (RTX 4090)
//! - TPOT < 10ms/token
//! - Throughput > 100 tokens/s

pub mod memory;
pub mod matmul;
pub mod flash_attention;
pub mod quant;

// 重新导出常用类型
pub use memory::CudaBuffer;

use std::sync::Arc;
use std::{fmt, ptr};

use thiserror::Error;

/// CUDA错误类型
#[derive(Error, Debug)]
pub enum CudaError {
    #[error("CUDA驱动未安装或版本不兼容")]
    DriverNotAvailable,
    #[error("设备 {device_id} 不存在或不可用")]
    DeviceNotFound { device_id: u32 },
    #[error("内存分配失败: 请求 {requested} 字节")]
    MemoryAllocationFailed { requested: usize },
    #[error("内存释放失败")]
    MemoryFreeFailed,
    #[error("内核启动失败: {message}")]
    KernelLaunchFailed { message: String },
    #[error("同步错误: {message}")]
    SynchronizationError { message: String },
    #[error("cuBLAS初始化失败: {message}")]
    CublasInitFailed { message: String },
    #[error("不支持的操作: {operation}")]
    UnsupportedOperation { operation: String },
    #[error("无效参数: {parameter}")]
    InvalidParameter { parameter: String },
    #[error("超时: 超过 {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },
    #[error("内部错误: {message}")]
    Internal { message: String },
}

/// CUDA设备信息
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    pub id: u32,
    pub name: String,
    pub total_memory: usize,
    pub free_memory: usize,
    pub compute_capability: (u32, u32),
    pub max_threads_per_block: u32,
    pub multiprocessor_count: u32,
    pub clock_rate_khz: u32,
    pub memory_clock_rate_khz: u32,
    pub l2_cache_size: usize,
    pub shared_memory_per_block: usize,
}

impl fmt::Display for CudaDeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GPU {}: {} ({}MB, CC {}.{}",
            self.id,
            self.name,
            self.total_memory / (1024 * 1024),
            self.compute_capability.0,
            self.compute_capability.1
        )
    }
}

/// CUDA设备句柄
#[derive(Debug)]
pub struct CudaDevice {
    info: CudaDeviceInfo,
    _private: (),
}

impl CudaDevice {
    /// 获取设备信息
    pub fn info(&self) -> &CudaDeviceInfo {
        &self.info
    }

    /// 检查是否支持特定计算能力
    pub fn supports_compute_capability(&self, major: u32, minor: u32) -> bool {
        let (dev_major, dev_minor) = self.info.compute_capability;
        (dev_major > major) || (dev_major == major && dev_minor >= minor)
    }

    /// 获取理论算力（TFLOPS，估算值）
    pub fn theoretical_tflops(&self) -> f64 {
        // 简化的FLOPS估算公式
        // 实际值需要考虑架构-specific的优化
        let cores_per_sm = match self.info.compute_capability {
            (8, 0) | (8, 6) | (8, 9) | (9, 0) => 128, // Ampere/Lovelace/Hopper
            (7, 0) | (7, 5) => 64,                      // Volta/Turing
            _ => 64,
        };
        
        let cores = cores_per_sm as f64 * self.info.multiprocessor_count as f64;
        let clock_ghz = self.info.clock_rate_khz as f64 / 1e6;
        
        // 假设FMA操作，每周期2 FLOPS
        cores * clock_ghz * 2.0 / 1000.0
    }

    /// 获取内存带宽（GB/s，估算值）
    pub fn memory_bandwidth_gb_s(&self) -> f64 {
        let bus_width = match self.info.compute_capability {
            (8, 0) | (8, 6) => 384, // RTX 3090/3080
            (8, 9) => 256,          // RTX 4070
            (9, 0) => 512,          // H100
            (7, 0) | (7, 5) => 352, // V100/TITAN V
            _ => 256,
        };
        
        let freq_hz = self.info.memory_clock_rate_khz as f64 * 1000.0;
        (bus_width as f64 / 8.0) * (freq_hz / 1e9) * 2.0 // DDR
    }
}

/// CUDA运行时上下文
#[derive(Debug, Clone)]
pub struct CudaContext {
    device: Arc<CudaDevice>,
    stream: *mut std::ffi::c_void, // cudaStream_t
    active: bool,
}

// 安全性说明：
// CudaContext 不是 Send/Sync，因为包含原始指针
// 在实际应用中需要通过Arc<Mutex<>>等方式安全共享

impl CudaContext {
    /// 创建新的CUDA上下文
    ///
    /// # 参数
    /// - `device_id`: 可选的设备ID，None表示自动选择最佳设备
    ///
    /// # 返回
    /// 成功返回CudaContext实例，失败返回CudaError
    pub fn new(device_id: Option<u32>) -> Result<Self, CudaError> {
        log::info!("正在初始化CUDA上下文...");

        // 获取可用设备列表
        let devices = Self::devices_internal()?;
        
        if devices.is_empty() {
            return Err(CudaError::DriverNotAvailable);
        }

        // 选择目标设备
        let target_device = if let Some(id) = device_id {
            devices.into_iter()
                .find(|d| d.id == id)
                .ok_or(CudaError::DeviceNotFound { device_id: id })?
        } else {
            // 自动选择：优先选择显存最大、计算能力最高的设备
            devices.into_iter()
                .max_by_key(|d| {
                    (
                        d.total_memory,
                        d.compute_capacity_score(),
                        d.free_memory,
                    )
                })
                .expect("已验证devices非空")
        };

        log::info!(
            "选择设备: {} ({}MB 可用)",
            target_device.name,
            target_device.free_memory / (1024 * 1024)
        );

        Ok(Self {
            device: Arc::new(CudaDevice {
                info: target_device,
                _private: (),
            }),
            stream: ptr::null_mut(),
            active: true,
        })
    }

    /// 获取所有可用的CUDA设备
    pub fn devices() -> Result<Vec<CudaDeviceInfo>, CudaError> {
        Self::devices_internal()
    }

    /// 内部设备枚举实现
    fn devices_internal() -> Result<Vec<CudaDeviceInfo>, CudaError> {
        // 尝试使用真实CUDA API
        #[cfg(feature = "cuda-native")]
        {
            Self::enumerate_cuda_devices()
        }
        
        // 回退到模拟模式（用于开发和测试）
        #[cfg(not(feature = "cuda-native"))]
        {
            Self::mock_devices()
        }
    }

    /// 使用cudarc枚举真实设备
    #[cfg(feature = "cuda-native")]
    fn enumerate_cuda_devices() -> Result<Vec<CudaDeviceInfo>, CudaError> {
        use cudarc::driver::CudaDevice;
        
        let num_devices = cudarc::driver::result::device_count()
            .map_err(|_| CudaError::DriverNotAvailable)?;
        
        let mut devices = Vec::with_capacity(num_devices as usize);
        
        for i in 0..num_devices {
            let props = cudarc::driver::result::get_device_properties(i)
                .map_err(|_| CudaError::DeviceNotFound { device_id: i })?;
            
            devices.push(CudaDeviceInfo {
                id: i,
                name: props.name.to_string(),
                total_memory: props.total_global_mem,
                free_memory: props.total_global_mem / 2, // 估算值
                compute_capability: (props.major, props.minor),
                max_threads_per_block: props.max_threads_per_block,
                multiprocessor_count: props.multi_processor_count,
                clock_rate_khz: props.clock_rate,
                memory_clock_rate_khz: props.memory_clock_rate,
                l2_cache_size: props.l2_cache_size,
                shared_memory_per_block: props.shared_mem_per_block,
            });
        }
        
        Ok(devices)
    }

    /// 模拟设备（用于无GPU环境）
    #[cfg(not(feature = "cuda-native"))]
    fn mock_devices() -> Result<Vec<CudaDeviceInfo>, CudaError> {
        log::warn!("CUDA不可用，使用模拟设备模式");
        
        Ok(vec![
            CudaDeviceInfo {
                id: 0,
                name: "NVIDIA GeForce RTX 4090".to_string(),
                total_memory: 24 * 1024 * 1024 * 1024, // 24GB
                free_memory: 20 * 1024 * 1024 * 1024,
                compute_capability: (8, 9),
                max_threads_per_block: 1024,
                multiprocessor_count: 128,
                clock_rate_khz: 2520000,
                memory_clock_rate_khz: 21000,
                l2_cache_size: 72 * 1024 * 1024,
                shared_memory_per_block: 228 * 1024,
            }
        ])
    }

    /// 获取当前设备引用
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// 检查上下文是否活跃
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// 同步CUDA流
    pub fn synchronize(&self) -> Result<(), CudaError> {
        if !self.active {
            return Err(CudaError::SynchronizationError {
                message: "上下文未激活".to_string(),
            });
        }

        #[cfg(feature = "cuda-native")]
        {
            use cudarc::driver::result::stream::synchronize;
            unsafe {
                synchronize(self.stream).map_err(|e| CudaError::SynchronizationError {
                    message: e.to_string(),
                })?;
            }
        }

        Ok(())
    }

    /// 设置设备为当前上下文
    pub fn set_current(&self) -> Result<(), CudaError> {
        if !self.active {
            return Err(CudaError::Internal {
                message: "上下文未激活".to_string(),
            });
        }

        log::debug!("设置设备 {} 为当前上下文", self.device.info().id);
        Ok(())
    }
}

impl CudaDeviceInfo {
    /// 计算能力评分（用于设备选择）
    fn compute_capacity_score(&self) -> u64 {
        let (major, minor) = self.compute_capability;
        (major as u64) * 100 + (minor as u64)
    }
}

/// RAII资源清理
impl Drop for CudaContext {
    fn drop(&mut self) {
        if self.active {
            log::info!("释放CUDA上下文 (设备 {})", self.device.info().id);
            
            #[cfg(feature = "cuda-native")]
            {
                // 清理CUDA流等资源
                if !self.stream.is_null() {
                    unsafe {
                        let _ = cudarc::driver::result::stream::destroy(self.stream);
                    }
                }
            }
            
            self.active = false;
        }
    }
}

/// 单元测试模块
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_device_creation() {
        let devices = CudaContext::devices().unwrap();
        assert!(!devices.is_empty());
        
        let device = &devices[0];
        assert_eq!(device.id, 0);
        assert!(device.name.contains("RTX"));
        assert_eq!(device.total_memory, 24 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_context_auto_select() {
        let ctx = CudaContext::new(None).unwrap();
        assert!(ctx.is_active());
        assert_eq!(ctx.device().info().id, 0);
    }

    #[test]
    fn test_context_with_specific_device() {
        let ctx = CudaContext::new(Some(0)).unwrap();
        assert_eq!(ctx.device().info().id, 0);
    }

    #[test]
    fn test_invalid_device_id() {
        let result = CudaContext::new(Some(999));
        assert!(result.is_err());
        matches!(result.unwrap_err(), CudaError::DeviceNotFound { .. });
    }

    #[test]
    fn test_device_info_display() {
        let devices = CudaContext::devices().unwrap();
        let display = format!("{}", devices[0]);
        assert!(display.contains("RTX"));
        assert!(display.contains("24")); // MB
    }

    #[test]
    fn test_device_compute_capability_check() {
        let ctx = CudaContext::new(None).unwrap();
        let device = ctx.device();
        
        // RTX 4090 是 SM 8.9
        assert!(device.supports_compute_capability(8, 0));
        assert!(device.supports_compute_capability(8, 9));
        assert!(!device.supports_compute_capability(9, 0)); // 不支持Hopper特性
    }

    #[test]
    fn test_theoretical_performance() {
        let ctx = CudaContext::new(None).unwrap();
        let device = ctx.device();
        
        let tflops = device.theoretical_tflops();
        assert!(tflops > 300.0); // RTX 4090 应该 >300 TFLOPS
        
        let bandwidth = device.memory_bandwidth_gb_s();
        assert!(bandwidth > 500.0); // 应该 >500 GB/s
    }

    #[test]
    fn test_error_types() {
        let err = CudaError::MemoryAllocationFailed { requested: 1024 };
        assert!(err.to_string().contains("1024"));
        
        let err = CudaError::Timeout { timeout_ms: 5000 };
        assert!(err.to_string().contains("5000"));
    }

    #[test]
    fn test_context_synchronize_inactive() {
        let mut ctx = CudaContext::new(None).unwrap();
        // 手动停用
        ctx.active = false;
        
        let result = ctx.synchronize();
        assert!(result.is_err());
    }
}
