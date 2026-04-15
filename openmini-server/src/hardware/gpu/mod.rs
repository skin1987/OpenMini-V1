//! GPU 抽象层模块
//!
//! 提供跨平台的 GPU 加速接口，支持：
//! - Metal (macOS/iOS)
//! - CUDA (NVIDIA)
//! - Vulkan (通用)
//!
//! ## 使用方式
//!
//! ```ignore
//! use openmini_server::hardware::gpu::{GpuBackend, GpuOps};
//!
//! let backend = GpuBackend::detect();
//! if let Some(gpu) = backend {
//!     let result = gpu.matmul(&a, &b);
//! }
//! ```

#![allow(dead_code)]

#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metal;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(feature = "vulkan")]
pub mod vulkan_compute;

use anyhow::Result;
use ndarray::Array2;

/// GPU 设备信息
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// 设备名称
    pub name: String,
    /// 显存大小（字节）
    pub memory_size: usize,
    /// 计算能力版本 (如 CUDA 7.5)
    pub compute_capability: Option<(u32, u32)>,
    /// 支持的特性
    pub features: Vec<String>,
}

/// GPU 操作 trait
#[allow(unused_variables)]
pub trait GpuOps: Send + Sync {
    /// 获取设备信息
    fn device_info(&self) -> &GpuDeviceInfo;

    /// 矩阵乘法: C = A @ B
    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>>;

    /// 批量矩阵乘法
    fn batch_matmul(&self, a: &[Array2<f32>], b: &[Array2<f32>]) -> Result<Vec<Array2<f32>>>;

    /// Softmax
    fn softmax(&self, input: &Array2<f32>) -> Result<Array2<f32>>;

    /// Layer Normalization
    fn layer_norm(
        &self,
        input: &Array2<f32>,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<Array2<f32>>;

    /// 注意力计算
    fn attention(
        &self,
        query: &Array2<f32>,
        key: &Array2<f32>,
        value: &Array2<f32>,
        mask: Option<&Array2<f32>>,
    ) -> Result<Array2<f32>>;

    /// 同步设备
    fn synchronize(&self) -> Result<()>;

    /// 获取可用显存
    fn available_memory(&self) -> Result<usize>;
}

/// GPU 后端类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackendType {
    /// Apple Metal
    Metal,
    /// NVIDIA CUDA
    Cuda,
    /// Vulkan
    Vulkan,
    /// 无 GPU
    None,
}

/// GPU 后端枚举
#[allow(dead_code)]
pub enum GpuBackend {
    /// Metal 后端
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal(crate::hardware::gpu::metal::MetalBackend),
    /// CUDA 后端
    #[cfg(feature = "cuda")]
    Cuda(crate::hardware::gpu::cuda::CudaBackend),
    /// Vulkan 后端
    #[cfg(feature = "vulkan")]
    Vulkan(crate::hardware::gpu::vulkan::VulkanBackend),
    /// 占位符 (无 GPU 后端时使用，不应被访问)
    #[cfg(not(any(
        all(target_os = "macos", feature = "metal"),
        feature = "cuda",
        feature = "vulkan"
    )))]
    #[doc(hidden)]
    _Phantom,
}

impl GpuBackend {
    /// 检测并创建最优 GPU 后端
    pub fn detect() -> Option<Self> {
        Self::from_target_device("auto")
    }

    /// 根据目标设备配置创建 GPU 后端
    ///
    /// # 参数
    ///
    /// - `target_device`: 目标设备字符串，支持 "auto", "cpu", "cuda", "metal", "vulkan"
    ///
    /// # 返回值
    ///
    /// - `Some(GpuBackend)`: 成功创建后端
    /// - `None`: 无可用后端或配置为 "cpu"
    pub fn from_target_device(target_device: &str) -> Option<Self> {
        match target_device {
            "cpu" => return None,
            "cuda" => {
                #[cfg(feature = "cuda")]
                {
                    if let Ok(cuda) = crate::hardware::gpu::cuda::CudaBackend::new() {
                        return Some(GpuBackend::Cuda(cuda));
                    }
                }
                return None;
            }
            "metal" => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    if let Ok(metal) = crate::hardware::gpu::metal::MetalBackend::new() {
                        return Some(GpuBackend::Metal(metal));
                    }
                }
                return None;
            }
            "vulkan" => {
                #[cfg(feature = "vulkan")]
                {
                    if let Ok(vulkan) = crate::hardware::gpu::vulkan::VulkanBackend::new() {
                        return Some(GpuBackend::Vulkan(vulkan));
                    }
                }
                return None;
            }
            _ => {} // "auto" 或其他值，继续自动检测
        }

        // 自动检测逻辑
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Ok(metal) = crate::hardware::gpu::metal::MetalBackend::new() {
                return Some(GpuBackend::Metal(metal));
            }
        }

        #[cfg(feature = "cuda")]
        {
            if let Ok(cuda) = crate::hardware::gpu::cuda::CudaBackend::new() {
                return Some(GpuBackend::Cuda(cuda));
            }
        }

        #[cfg(feature = "vulkan")]
        {
            if let Ok(vulkan) = crate::hardware::gpu::vulkan::VulkanBackend::new() {
                return Some(GpuBackend::Vulkan(vulkan));
            }
        }

        None
    }

    /// 获取后端类型
    pub fn backend_type(&self) -> GpuBackendType {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            GpuBackend::Metal(_) => GpuBackendType::Metal,
            #[cfg(feature = "cuda")]
            GpuBackend::Cuda(_) => GpuBackendType::Cuda,
            #[cfg(feature = "vulkan")]
            GpuBackend::Vulkan(_) => GpuBackendType::Vulkan,
            #[cfg(not(any(
                all(target_os = "macos", feature = "metal"),
                feature = "cuda",
                feature = "vulkan"
            )))]
            GpuBackend::_Phantom => unreachable!("GpuBackend::_Phantom cannot be called"),
        }
    }
}

#[allow(unused_variables)]
impl GpuOps for GpuBackend {
    fn device_info(&self) -> &GpuDeviceInfo {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            GpuBackend::Metal(metal) => metal.device_info(),
            #[cfg(feature = "cuda")]
            GpuBackend::Cuda(cuda) => cuda.device_info(),
            #[cfg(feature = "vulkan")]
            GpuBackend::Vulkan(vulkan) => vulkan.device_info(),
            #[cfg(not(any(
                all(target_os = "macos", feature = "metal"),
                feature = "cuda",
                feature = "vulkan"
            )))]
            GpuBackend::_Phantom => unreachable!("GpuBackend::_Phantom cannot be called"),
        }
    }

    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            GpuBackend::Metal(metal) => metal.matmul(a, b),
            #[cfg(feature = "cuda")]
            GpuBackend::Cuda(cuda) => cuda.matmul(a, b),
            #[cfg(feature = "vulkan")]
            GpuBackend::Vulkan(vulkan) => vulkan.matmul(a, b),
            #[cfg(not(any(
                all(target_os = "macos", feature = "metal"),
                feature = "cuda",
                feature = "vulkan"
            )))]
            GpuBackend::_Phantom => unreachable!("GpuBackend::_Phantom cannot be called"),
        }
    }

    fn batch_matmul(&self, a: &[Array2<f32>], b: &[Array2<f32>]) -> Result<Vec<Array2<f32>>> {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            GpuBackend::Metal(metal) => metal.batch_matmul(a, b),
            #[cfg(feature = "cuda")]
            GpuBackend::Cuda(cuda) => cuda.batch_matmul(a, b),
            #[cfg(feature = "vulkan")]
            GpuBackend::Vulkan(vulkan) => vulkan.batch_matmul(a, b),
            #[cfg(not(any(
                all(target_os = "macos", feature = "metal"),
                feature = "cuda",
                feature = "vulkan"
            )))]
            GpuBackend::_Phantom => unreachable!("GpuBackend::_Phantom cannot be called"),
        }
    }

    fn softmax(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            GpuBackend::Metal(metal) => metal.softmax(input),
            #[cfg(feature = "cuda")]
            GpuBackend::Cuda(cuda) => cuda.softmax(input),
            #[cfg(feature = "vulkan")]
            GpuBackend::Vulkan(vulkan) => vulkan.softmax(input),
            #[cfg(not(any(
                all(target_os = "macos", feature = "metal"),
                feature = "cuda",
                feature = "vulkan"
            )))]
            GpuBackend::_Phantom => unreachable!("GpuBackend::_Phantom cannot be called"),
        }
    }

    fn layer_norm(
        &self,
        input: &Array2<f32>,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<Array2<f32>> {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            GpuBackend::Metal(metal) => metal.layer_norm(input, gamma, beta, eps),
            #[cfg(feature = "cuda")]
            GpuBackend::Cuda(cuda) => cuda.layer_norm(input, gamma, beta, eps),
            #[cfg(feature = "vulkan")]
            GpuBackend::Vulkan(vulkan) => vulkan.layer_norm(input, gamma, beta, eps),
            #[cfg(not(any(
                all(target_os = "macos", feature = "metal"),
                feature = "cuda",
                feature = "vulkan"
            )))]
            GpuBackend::_Phantom => unreachable!("GpuBackend::_Phantom cannot be called"),
        }
    }

    fn attention(
        &self,
        query: &Array2<f32>,
        key: &Array2<f32>,
        value: &Array2<f32>,
        mask: Option<&Array2<f32>>,
    ) -> Result<Array2<f32>> {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            GpuBackend::Metal(metal) => metal.attention(query, key, value, mask),
            #[cfg(feature = "cuda")]
            GpuBackend::Cuda(cuda) => cuda.attention(query, key, value, mask),
            #[cfg(feature = "vulkan")]
            GpuBackend::Vulkan(vulkan) => vulkan.attention(query, key, value, mask),
            #[cfg(not(any(
                all(target_os = "macos", feature = "metal"),
                feature = "cuda",
                feature = "vulkan"
            )))]
            GpuBackend::_Phantom => unreachable!("GpuBackend::_Phantom cannot be called"),
        }
    }

    fn synchronize(&self) -> Result<()> {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            GpuBackend::Metal(metal) => metal.synchronize(),
            #[cfg(feature = "cuda")]
            GpuBackend::Cuda(cuda) => cuda.synchronize(),
            #[cfg(feature = "vulkan")]
            GpuBackend::Vulkan(vulkan) => vulkan.synchronize(),
            #[cfg(not(any(
                all(target_os = "macos", feature = "metal"),
                feature = "cuda",
                feature = "vulkan"
            )))]
            GpuBackend::_Phantom => unreachable!("GpuBackend::_Phantom cannot be called"),
        }
    }

    fn available_memory(&self) -> Result<usize> {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            GpuBackend::Metal(metal) => metal.available_memory(),
            #[cfg(feature = "cuda")]
            GpuBackend::Cuda(cuda) => cuda.available_memory(),
            #[cfg(feature = "vulkan")]
            GpuBackend::Vulkan(vulkan) => vulkan.available_memory(),
            #[cfg(not(any(
                all(target_os = "macos", feature = "metal"),
                feature = "cuda",
                feature = "vulkan"
            )))]
            GpuBackend::_Phantom => unreachable!("GpuBackend::_Phantom cannot be called"),
        }
    }
}

/// 创建 GPU 操作实例 (自动检测最优后端)
pub fn create_gpu_ops() -> Option<Box<dyn GpuOps>> {
    GpuBackend::detect().map(|b| Box::new(b) as Box<dyn GpuOps>)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_device_info_creation_and_debug() {
        // 测试GpuDeviceInfo的创建和Debug输出
        let info = GpuDeviceInfo {
            name: "Test GPU".to_string(),
            memory_size: 8192 * 1024 * 1024, // 8GB
            compute_capability: Some((7, 5)),
            features: vec!["CUDA".to_string(), "TensorCore".to_string()],
        };

        // 验证字段
        assert_eq!(info.name, "Test GPU");
        assert_eq!(info.memory_size, 8192 * 1024 * 1024);
        assert_eq!(info.compute_capability, Some((7, 5)));
        assert_eq!(info.features.len(), 2);
        assert!(info.features.contains(&"CUDA".to_string()));

        // 验证Debug格式化包含设备名称
        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("Test GPU"));
    }

    #[test]
    fn test_gpu_device_info_minimal() {
        // 测试最小配置的GpuDeviceInfo
        let info = GpuDeviceInfo {
            name: "Minimal GPU".to_string(),
            memory_size: 0,
            compute_capability: None,
            features: vec![],
        };

        assert_eq!(info.name, "Minimal GPU");
        assert_eq!(info.memory_size, 0);
        assert!(info.compute_capability.is_none());
        assert!(info.features.is_empty());
    }

    #[test]
    fn test_gpu_backend_type_variants() {
        // 测试所有GPU后端类型变体
        let types = [
            GpuBackendType::Metal,
            GpuBackendType::Cuda,
            GpuBackendType::Vulkan,
            GpuBackendType::None,
        ];

        for backend_type in &types {
            // 验证Debug trait实现（不应panic）
            let _ = format!("{:?}", backend_type);
        }

        // 验证相等性比较
        assert_eq!(GpuBackendType::Metal, GpuBackendType::Metal);
        assert_ne!(GpuBackendType::Metal, GpuBackendType::Cuda);
    }

    #[test]
    fn test_gpu_detect_returns_option() {
        // 测试detect方法返回Option类型（不panic即可）
        // 在没有实际GPU的环境中，应该返回None
        let result = GpuBackend::detect();

        // 无论是否有GPU，都不应该panic
        match result {
            Some(backend) => {
                // 如果检测到GPU，验证可以获取后端类型
                let _backend_type = backend.backend_type();
            }
            None => {
                // 没有GPU也是合法的结果
            }
        }
    }

    #[test]
    fn test_create_gpu_ops_returns_option() {
        // 测试create_gpu_ops工厂方法
        let result = create_gpu_ops();
        // 注意：测试环境可能没有 GPU，所以结果可能是 None
        // 这里仅验证函数不 panic
        let _ = result;
    }

    #[test]
    fn test_from_target_device() {
        // 测试各种目标设备配置
        let result = GpuBackend::from_target_device("cpu");
        assert!(result.is_none(), "cpu 配置应该返回 None");

        // 测试auto配置（应该尝试检测所有可用后端）
        let _auto_result = GpuBackend::from_target_device("auto");
        // 不检查具体值，因为测试环境可能没有GPU

        // 测试无效配置
        let _invalid_result = GpuBackend::from_target_device("invalid");
        // 无效配置应该回退到自动检测
    }

    // ==================== 新增测试开始 ====================

    /// 测试GpuDeviceInfo克隆功能
    /// 覆盖分支：Clone trait实现
    #[test]
    fn test_gpu_device_info_clone() {
        let info1 = GpuDeviceInfo {
            name: "Clone Test GPU".to_string(),
            memory_size: 4 * 1024 * 1024 * 1024,
            compute_capability: Some((8, 0)),
            features: vec!["Feature1".to_string(), "Feature2".to_string()],
        };

        let info2 = info1.clone();

        // 验证克隆后的值相等
        assert_eq!(info1.name, info2.name);
        assert_eq!(info1.memory_size, info2.memory_size);
        assert_eq!(info1.compute_capability, info2.compute_capability);
        assert_eq!(info1.features, info2.features);

        // 验证是独立副本
        let mut info3 = info1.clone();
        info3.name = "Modified GPU".to_string();
        assert_ne!(info1.name, info3.name);
    }

    /// 测试GpuDeviceInfo不同内存大小
    /// 覆盖分支：边界条件 - 各种内存大小
    #[test]
    fn test_gpu_device_info_various_memory_sizes() {
        let sizes = vec![
            0,
            1024,
            1024 * 1024,             // 1MB
            1024 * 1024 * 1024,      // 1GB
            8 * 1024 * 1024 * 1024,  // 8GB
            24 * 1024 * 1024 * 1024, // 24GB
            usize::MAX,              // 极大值
        ];

        for size in sizes {
            let info = GpuDeviceInfo {
                name: format!("GPU with {} bytes", size),
                memory_size: size,
                compute_capability: None,
                features: vec![],
            };
            assert_eq!(info.memory_size, size);
        }
    }

    /// 测试GpuDeviceInfo不同计算能力版本
    /// 覆盖分支：compute_capability字段的各种值
    #[test]
    fn test_gpu_device_info_compute_capabilities() {
        let capabilities = vec![
            None,
            Some((3, 0)),
            Some((5, 0)),
            Some((6, 0)),
            Some((7, 0)),
            Some((7, 5)),
            Some((8, 0)),
            Some((8, 6)),
            Some((9, 0)),
        ];

        for cap in capabilities {
            let info = GpuDeviceInfo {
                name: "Test GPU".to_string(),
                memory_size: 8 * 1024 * 1024 * 1024,
                compute_capability: cap,
                features: vec![],
            };
            assert_eq!(info.compute_capability, cap);
        }
    }

    /// 测试GpuDeviceInfo大量特性列表
    /// 覆盖分支：features字段的边界条件
    #[test]
    fn test_gpu_device_info_many_features() {
        let features: Vec<String> = (0..100).map(|i| format!("Feature_{}", i)).collect();

        let info = GpuDeviceInfo {
            name: "GPU with many features".to_string(),
            memory_size: 16 * 1024 * 1024 * 1024,
            compute_capability: Some((8, 0)),
            features: features.clone(),
        };

        assert_eq!(info.features.len(), 100);
        assert!(info.features.contains(&"Feature_42".to_string()));
        assert!(info.features.contains(&"Feature_99".to_string()));
    }

    /// 测试GpuDeviceInfo空特性列表
    /// 覆盖分支：features字段的空值
    #[test]
    fn test_gpu_device_info_empty_features() {
        let info = GpuDeviceInfo {
            name: "GPU with no features".to_string(),
            memory_size: 4 * 1024 * 1024 * 1024,
            compute_capability: None,
            features: vec![],
        };

        assert!(info.features.is_empty());
        assert_eq!(info.features.len(), 0);
    }

    /// 测试GpuBackendType相等性和不等性
    /// 覆盖分支：PartialEq trait的完整实现
    #[test]
    fn test_gpu_backend_type_equality() {
        // 测试所有类型的相等性
        let types = vec![
            GpuBackendType::Metal,
            GpuBackendType::Cuda,
            GpuBackendType::Vulkan,
            GpuBackendType::None,
        ];

        // 每个类型应该等于自己
        for t in &types {
            assert_eq!(*t, *t);
        }

        // 不同类型应该不相等
        for i in 0..types.len() {
            for j in (i + 1)..types.len() {
                assert_ne!(types[i], types[j]);
            }
        }
    }

    /// 测试GpuBackendType Copy trait
    /// 覆盖分支：Copy trait的行为
    #[test]
    fn test_gpu_backend_type_copy() {
        let original = GpuBackendType::Cuda;
        let copied = original;

        // Copy trait允许移动后原值仍然可用
        assert_eq!(original, GpuBackendType::Cuda);
        assert_eq!(copied, GpuBackendType::Cuda);
    }

    /// 测试GpuDeviceInfo特殊字符名称
    /// 覆盖分支：name字段的边界条件
    #[test]
    fn test_gpu_device_info_special_names() {
        let names = vec![
            "",
            "NVIDIA RTX 4090",
            "AMD Radeon RX 7900 XTX",
            "Apple M2 Ultra",
            "Intel Arc A770",
            "中文GPU名称",
            "GPU with \"quotes\"",
            "GPU with 'apostrophes'",
            "GPU\nwith\nnewlines",
            "GPU\twith\ttabs",
            "GPU with emoji 🎮",
        ];

        for name in names {
            let info = GpuDeviceInfo {
                name: name.to_string(),
                memory_size: 8 * 1024 * 1024 * 1024,
                compute_capability: None,
                features: vec![],
            };
            assert_eq!(info.name, name);
        }
    }

    /// 测试GpuBackendType::None变体
    /// 覆盖分支：None类型的独立存在性
    #[test]
    fn test_gpu_backend_type_none() {
        let none_type = GpuBackendType::None;

        // 验证Debug输出
        let debug_str = format!("{:?}", none_type);
        assert!(debug_str.contains("None"));

        // 验证相等性
        assert_eq!(none_type, GpuBackendType::None);
        assert_ne!(none_type, GpuBackendType::Metal);
        assert_ne!(none_type, GpuBackendType::Cuda);
        assert_ne!(none_type, GpuBackendType::Vulkan);
    }

    /// 测试GpuDeviceInfo的Display-like行为（通过Debug）
    /// 覆盖分支：完整字段序列化
    #[test]
    fn test_gpu_device_info_debug_completeness() {
        // 创建包含所有字段的实例
        let info = GpuDeviceInfo {
            name: "Test GPU Full".to_string(),
            memory_size: 16 * 1024 * 1024 * 1024,
            compute_capability: Some((9, 0)),
            features: vec![
                "RayTracing".to_string(),
                "DLSS".to_string(),
                "TensorCore".to_string(),
            ],
        };

        let debug_output = format!("{:?}", info);

        // 验证所有关键字段都出现在输出中
        assert!(debug_output.contains("Test GPU Full"));
        assert!(debug_output.contains("17179869184") || debug_output.contains("16"));
        assert!(debug_output.contains("9"));
        assert!(debug_output.contains("RayTracing"));
        assert!(debug_output.contains("DLSS"));
    }

    /// 测试GpuDeviceInfo零值和极大值组合
    /// 覆盖分支：极端参数组合
    #[test]
    fn test_gpu_device_info_extreme_combinations() {
        // 最小配置
        let min_config = GpuDeviceInfo {
            name: String::new(),
            memory_size: 0,
            compute_capability: None,
            features: vec![],
        };
        assert_eq!(min_config.memory_size, 0);
        assert!(min_config.name.is_empty());
        assert!(min_config.features.is_empty());
        assert!(min_config.compute_capability.is_none());

        // 极大配置
        let max_config = GpuDeviceInfo {
            name: "X".repeat(10000),
            memory_size: usize::MAX,
            compute_capability: Some((u32::MAX, u32::MAX)),
            features: (0..1000).map(|i| format!("F{}", i)).collect(),
        };
        assert_eq!(max_config.memory_size, usize::MAX);
        assert_eq!(max_config.name.len(), 10000);
        assert_eq!(max_config.features.len(), 1000);
        assert_eq!(max_config.compute_capability, Some((u32::MAX, u32::MAX)));
    }

    /// 测试GpuBackendType所有变体的唯一性
    /// 覆盖分支：枚举变体互不相同
    #[test]
    fn test_gpu_backend_type_all_distinct() {
        let types = [GpuBackendType::Metal,
            GpuBackendType::Cuda,
            GpuBackendType::Vulkan,
            GpuBackendType::None];

        // 验证任意两个不同变体都不相等
        for i in 0..types.len() {
            for j in (i + 1)..types.len() {
                assert_ne!(
                    types[i], types[j],
                    "{:?} should not equal {:?}",
                    types[i], types[j]
                );
            }
        }
    }

    /// 测试GpuDeviceInfo单元素特性列表
    /// 覆盖分支：features长度为1的情况
    #[test]
    fn test_gpu_device_info_single_feature() {
        let info = GpuDeviceInfo {
            name: "Single Feature GPU".to_string(),
            memory_size: 2 * 1024 * 1024 * 1024,
            compute_capability: Some((5, 0)),
            features: vec!["OnlyFeature".to_string()],
        };

        assert_eq!(info.features.len(), 1);
        assert!(info.features.contains(&"OnlyFeature".to_string()));
    }

    /// 测试GpuDeviceInfo Unicode名称处理
    /// 覆盖分支：Unicode字符在name中的正确存储
    #[test]
    fn test_gpu_device_info_unicode_names() {
        let unicode_names = vec![
            ("日本語GPU", "Japanese"),
            ("한국어GPU", "Korean"),
            ("РусскийGPU", "Russian"),
            ("العربيةGPU", "Arabic"),
            ("🚀🎮⚡", "Emoji only"),
        ];

        for (unicode_name, description) in unicode_names {
            let info = GpuDeviceInfo {
                name: unicode_name.to_string(),
                memory_size: 4 * 1024 * 1024 * 1024,
                compute_capability: None,
                features: vec![description.to_string()],
            };
            assert_eq!(info.name, unicode_name, "Failed for {}", description);
            assert!(info.features.contains(&description.to_string()));
        }
    }

    /// 测试GpuDeviceInfo特性列表包含特殊字符串
    /// 覆盖分支：features中包含特殊字符
    #[test]
    fn test_gpu_device_info_special_feature_strings() {
        let special_features = vec![
            "".to_string(),
            " ".to_string(),
            "\t".to_string(),
            "\n".to_string(),
            "feature-with-dashes".to_string(),
            "feature_with_underscores".to_string(),
            "feature.with.dots".to_string(),
            "FEATURE_UPPERCASE".to_string(),
            "feature lowercase".to_string(),
            "feature#special@chars!".to_string(),
            "中文特性".to_string(),
        ];

        let info = GpuDeviceInfo {
            name: "Special Features GPU".to_string(),
            memory_size: 8 * 1024 * 1024 * 1024,
            compute_capability: Some((8, 6)),
            features: special_features.clone(),
        };

        assert_eq!(info.features.len(), special_features.len());
        for feature in &special_features {
            assert!(
                info.features.contains(feature),
                "Should contain feature: {:?}",
                feature
            );
        }
    }

    /// 测试GpuBackendType可以用于模式匹配
    /// 覆盖分支：match语句的所有分支
    #[test]
    fn test_gpu_backend_type_pattern_matching() {
        let test_cases = vec![
            (GpuBackendType::Metal, "metal"),
            (GpuBackendType::Cuda, "cuda"),
            (GpuBackendType::Vulkan, "vulkan"),
            (GpuBackendType::None, "none"),
        ];

        for (backend_type, expected_str) in test_cases {
            let result = match backend_type {
                GpuBackendType::Metal => "metal",
                GpuBackendType::Cuda => "cuda",
                GpuBackendType::Vulkan => "vulkan",
                GpuBackendType::None => "none",
            };
            assert_eq!(
                result, expected_str,
                "Pattern match failed for {:?}",
                backend_type
            );
        }
    }
}
