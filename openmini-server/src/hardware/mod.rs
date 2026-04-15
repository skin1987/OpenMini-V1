//! 硬件自适应调度模块
//!
//! 提供硬件检测、能力分级和自适应调度功能。
//!
//! # 功能
//!
//! - **硬件检测**: 自动检测 CPU/GPU/内存能力
//! - **能力分级**: 根据硬件配置自动分级
//! - **自适应调度**: 根据硬件能力选择最优推理策略
//!
//! # 支持的平台
//!
//! ## 移动端
//! - 最低要求: iPhone 13 Pro Max (A15 芯片)
//! - 上限: 不限制
//!
//! ## PC/服务器
//! - 最低要求: MacBook Pro 2017 (i7-7700HQ)
//! - 上限: 不限制
//!
//! ## 国产硬件
//! - 华为昇腾 (ARM + NPU)
//! - 飞腾 (ARM)
//! - 龙芯 (LoongArch)
//! - 申威 (SW-64)
//! - 海光 (x86-64)
//! - 兆芯 (x86-64)

pub mod cpu;
pub mod detector;
pub mod device;
pub mod ess;
pub mod gpu;
pub mod hyperthreading;
pub mod kernel;
pub mod kv_cache;
pub mod load_monitor;
pub mod memory;
pub mod persistence;
pub mod profile;
pub mod resource_manager;
pub mod scheduler;
pub mod simd;

// 重导出公共接口
#[allow(unused_imports)]
pub use detector::{
    CacheInfo, CacheTopology, CoreType, CpuArch, CpuInfo, GpuInfo, GpuType, HardwareProfile,
    HyperthreadTopology, MemoryInfo, NumaNode, NumaTopology, PhysicalCore, SimdCapabilities,
};

#[allow(unused_imports)]
pub use profile::{DeviceType, HardwareClassifier, HardwareLevel, InferenceStrategy};

#[allow(unused_imports)]
pub use simd::{create_simd_ops, ScalarOps, SimdLevel, SimdOps};

#[allow(unused_imports)]
pub use scheduler::{
    AdaptiveScheduler, AttentionStrategy, ComputeDevice as SchedulerComputeDevice, InferenceConfig,
    MemoryStrategy, ParallelStrategy, ScheduleStrategy, TaskThresholds, UnifiedScheduler,
};

#[allow(unused_imports)]
pub use hyperthreading::{
    CoreSelectionStrategy, CpuAffinity, CpuAffinityError, HyperthreadEfficiency, TaskType,
    ThreadPoolConfig,
};

#[allow(unused_imports)]
pub use resource_manager::{
    get_resource_manager, HardwareResourceManager, MemoryType, ResourceAllocation, ResourceRequest,
};

#[allow(unused_imports)]
pub use load_monitor::{get_system_load, LoadAction, LoadMonitor, LoadThresholds, SystemLoad};

#[allow(unused_imports)]
pub use cpu::{CpuBackend, CpuBackendType, CpuInfoDetail, CpuOps, SimdInfo};

#[allow(unused_imports)]
pub use persistence::{
    CompressionManager, DatabaseConfig, DatabaseManager, EvictionAlgorithm, EvictionPolicy,
    KvSwapConfig, KvSwapManager, KvSwapStats, PersistenceError,
};

/// 快速检测硬件配置
pub fn detect_hardware() -> HardwareProfile {
    HardwareProfile::detect()
}

/// 快速获取硬件级别
#[allow(dead_code)]
pub fn get_hardware_level() -> HardwareLevel {
    let profile = HardwareProfile::detect();
    let classifier = HardwareClassifier::new(profile);
    classifier.level()
}

/// 快速获取推荐推理策略
#[allow(dead_code)]
pub fn get_recommended_strategy() -> InferenceStrategy {
    let profile = HardwareProfile::detect();
    let classifier = HardwareClassifier::new(profile);
    classifier.recommended_strategy()
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_internal {
    use super::*;

    #[test]
    fn test_detect_hardware() {
        let profile = detect_hardware();
        assert!(profile.cpu.physical_cores > 0);
        assert!(profile.memory.total_gb > 0);
    }

    #[test]
    fn test_get_hardware_level() {
        let level = get_hardware_level();
        assert!(level >= HardwareLevel::Entry);
    }

    #[test]
    fn test_get_recommended_strategy() {
        let strategy = get_recommended_strategy();
        assert!(strategy.use_simd);
    }

    // ==================== 新增分支覆盖测试 ====================

    /// 测试 detect_hardware 返回的 profile 各字段合理性
    #[test]
    fn test_detect_hardware_profile_fields() {
        let profile = detect_hardware();

        // CPU 字段验证
        assert!(profile.cpu.physical_cores >= 1);

        // 内存字段验证
        assert!(profile.memory.total_gb >= 1);

        // GPU 字段验证（使用 gpu 单数形式）
        let _gpu_info = &profile.gpu;

        // SIMD 能力通过独立 API 检测
        let _simd_level = SimdLevel::detect();
    }

    /// 测试 HardwareProfile::detect() 与 detect_hardware() 一致性
    #[test]
    fn test_detect_consistency() {
        let profile1 = detect_hardware();
        let profile2 = HardwareProfile::detect();
        // 两次检测应该返回相同的基本信息
        assert_eq!(profile1.cpu.physical_cores, profile2.cpu.physical_cores);
        assert_eq!(profile1.memory.total_gb, profile2.memory.total_gb);
    }

    /// 测试 HardwareClassifier 的 level() 和 recommended_strategy()
    #[test]
    fn test_classifier_api() {
        let profile = detect_hardware();
        let classifier = HardwareClassifier::new(profile);

        // level() 应该返回有效级别
        let level = classifier.level();
        assert!(level >= HardwareLevel::Entry);

        // recommended_strategy() 应返回包含 simd 的策略
        let strategy = classifier.recommended_strategy();
        // 验证策略字段可访问
        let _use_simd = strategy.use_simd;
        let _use_gpu = strategy.use_gpu;
    }

    /// 测试 get_hardware_level 返回值在合法范围内
    #[test]
    fn test_hardware_level_range() {
        let level = get_hardware_level();
        // 确保返回的是已知变体之一（通过 Debug 输出验证）
        let level_str = format!("{:?}", level);
        assert!(
            !level_str.is_empty(),
            "HardwareLevel should have a valid debug representation"
        );
    }

    /// 测试 get_recommended_strategy 返回完整字段
    #[test]
    fn test_recommended_strategy_fields() {
        let strategy = get_recommended_strategy();
        // 验证推荐策略的关键字段
        assert!(strategy.use_simd, "Recommended strategy should use SIMD");
        // use_gpu 可能为 true 或 false，取决于硬件
    }

    /// 测试子模块 re-export 的类型可用性
    #[test]
    fn test_submodule_reexports() {
        // 验证 detector 模块导出的类型可用
        let profile = HardwareProfile::detect();
        assert!(profile.cpu.physical_cores > 0);

        // 验证 simd 模块导出的类型可用
        let _simd_level = SimdLevel::detect();
        let _ops: Box<dyn SimdOps> = create_simd_ops();

        // 验证 scheduler 模块导出的类型可用
        let _strategy_type = MemoryStrategy::StandardArena;

        // 验证 cpu 模块导出的类型可用
        let _backend_type = CpuBackendType::Rust; // 纯 Rust 后端
    }

    /// 测试 CpuArch / GpuType 等枚举类型的可用性
    #[test]
    fn test_enum_types_existence() {
        // 通过 profile 访问这些类型
        let profile = detect_hardware();
        let arch = profile.cpu.arch;
        let _arch_debug = format!("{:?}", arch);

        // GPU 信息（GpuInfo 是单个结构体，不是 Vec）
        let gpu = &profile.gpu;
        let _gpu_type = gpu.gpu_type;
        let _gpu_debug = format!("{:?}", gpu);
    }

    /// 测试多次调用 API 的稳定性
    #[test]
    fn test_api_stability_multiple_calls() {
        // 连续调用多次应返回一致结果
        let level1 = get_hardware_level();
        let level2 = get_hardware_level();
        assert_eq!(level1, level2);

        let strategy1 = get_recommended_strategy();
        let strategy2 = get_recommended_strategy();
        assert_eq!(strategy1.use_simd, strategy2.use_simd);
    }

    // ==================== 新增分支覆盖率测试 ====================

    /// 测试：HardwareProfile - GPU字段详细验证（GpuInfo结构体完整性）
    #[test]
    fn test_hardware_profile_gpu_fields() {
        let profile = detect_hardware();

        let gpu = &profile.gpu;

        // 验证 GpuInfo 的关键字段可访问且合理
        let _gpu_type = gpu.gpu_type; // GpuType 枚举
        let _supports_metal = gpu.supports_metal; // bool
        let _supports_cuda = gpu.supports_cuda; // bool

        // 验证 Debug 输出
        let gpu_debug = format!("{:?}", gpu);
        assert!(!gpu_debug.is_empty());
    }

    /// 测试：MemoryInfo - available_gb 与 total_gb 关系（内存信息一致性）
    #[test]
    fn test_memory_info_consistency() {
        let profile = detect_hardware();

        let memory = &profile.memory;

        // 基本合理性检查
        assert!(memory.total_gb > 0, "Total memory should be > 0");
        assert!(
            memory.available_gb <= memory.total_gb,
            "Available ({}) should be <= Total ({})",
            memory.available_gb,
            memory.total_gb
        );
        assert!(memory.available_gb > 0, "Available memory should be > 0");
    }

    /// 测试：SimdCapabilities - SIMD能力字段详细验证
    #[test]
    fn test_simd_capabilities_fields() {
        let profile = detect_hardware();

        let simd = &profile.cpu.simd;

        // 验证所有SIMD字段可访问
        let _neon = simd.neon; // ARM NEON
        let _avx2 = simd.avx2; // x86 AVX2
        let _avx512 = simd.avx512; // x86 AVX-512

        // 至少应该支持一种SIMD（或者都不支持在某些虚拟化环境）
        // 这里只验证字段存在性，不强制要求支持某种SIMD
    }

    /// 测试：CpuArch枚举 - 所有变体的Debug输出和比较性
    #[test]
    fn test_cpu_arch_enum_completeness() {
        let profile = detect_hardware();
        let arch = profile.cpu.arch;

        // 验证当前架构的Debug输出
        let arch_str = format!("{:?}", arch);
        assert!(
            !arch_str.is_empty(),
            "CpuArch should have valid debug representation"
        );

        // 验证架构是已知类型之一（通过匹配确认）
        match arch {
            CpuArch::X86_64 => { /* x86-64 架构 */ }
            CpuArch::AArch64 => { /* ARM64 架构 */ }
            CpuArch::LoongArch => { /* LoongArch 架构 */ }
            _ => { /* 其他架构（Arm, Sw64, RiscV） */ }
        }
    }

    /// 测试：GpuType枚举 - 可用性和变体验证
    #[test]
    fn test_gpu_type_enum() {
        let profile = detect_hardware();
        let gpu_type = profile.gpu.gpu_type;

        // 验证 GpuType 是有效枚举值
        let gpu_type_str = format!("{:?}", gpu_type);
        assert!(!gpu_type_str.is_empty());

        // 验证不同 GpuType 变体可以比较（如果需要）
        // 注意：GpuType 可能没有实现 PartialEq，这里只验证 Debug
    }

    /// 测试：DeviceType枚举 - 完整的变体覆盖和Debug输出
    #[test]
    fn test_device_type_complete_variants() {
        // 创建所有 DeviceType 变体并验证
        let types = vec![DeviceType::Mobile, DeviceType::Desktop, DeviceType::Server];

        for device_type in &types {
            let debug_str = format!("{:?}", device_type);
            assert!(
                !debug_str.is_empty(),
                "{:?} should have debug output",
                device_type
            );
        }

        // 验证不等性
        assert_ne!(DeviceType::Mobile, DeviceType::Desktop);
        assert_ne!(DeviceType::Desktop, DeviceType::Server);
        assert_ne!(DeviceType::Mobile, DeviceType::Server);
    }

    /// 测试：多次创建 HardwareClassifier 的一致性和独立性
    #[test]
    fn test_classifier_creation_consistency() {
        let profile = detect_hardware();

        // 从同一profile创建多个classifier
        let classifier1 = HardwareClassifier::new(profile.clone());
        let classifier2 = HardwareClassifier::new(profile.clone());
        let classifier3 = HardwareClassifier::new(profile.clone());

        // 所有classifier应该返回相同的结果
        assert_eq!(classifier1.level(), classifier2.level());
        assert_eq!(classifier2.level(), classifier3.level());
        assert_eq!(classifier1.device_type(), classifier2.device_type());
        assert_eq!(
            classifier1.meets_requirements(),
            classifier2.meets_requirements()
        );

        // recommended_strategy 也应该一致
        let strat1 = classifier1.recommended_strategy();
        let strat2 = classifier2.recommended_strategy();
        assert_eq!(strat1.use_simd, strat2.use_simd);
        assert_eq!(strat1.use_gpu, strat2.use_gpu);
        assert_eq!(strat1.batch_size, strat2.batch_size);
    }

    /// 测试：InferenceStrategy 结构体 - 所有字段的边界值验证
    #[test]
    fn test_inference_strategy_boundary_values() {
        // Entry级别策略（最小配置）
        let profile_entry = create_minimal_profile();
        let classifier_entry = HardwareClassifier::new(profile_entry);
        let strategy_entry = classifier_entry.recommended_strategy();

        // 验证最小配置的字段范围
        assert!(strategy_entry.batch_size >= 1, "Batch size should be >= 1");
        assert!(
            strategy_entry.max_seq_len >= 2048,
            "Max seq len should be >= 2048"
        );
        assert!(
            strategy_entry.max_seq_len <= 32768,
            "Max seq len should be <= 32768"
        );

        // 验证布尔字段都是有效的bool值（字段可以被读取）
        let _ = strategy_entry.use_simd;
        let _ = strategy_entry.use_gpu;
        let _ = strategy_entry.use_dsa;
        let _ = strategy_entry.use_flash_attention;
    }

    /// 辅助函数：创建最小配置的HardwareProfile用于边界测试
    fn create_minimal_profile() -> HardwareProfile {
        use crate::hardware::detector::{
            CacheTopology, CpuArch, CpuInfo, GpuInfo, HardwareProfile, HyperthreadTopology,
            MemoryInfo, NumaTopology, SimdCapabilities,
        };

        HardwareProfile {
            cpu: CpuInfo {
                arch: CpuArch::X86_64,
                physical_cores: 1,
                logical_cores: 1,
                simd: SimdCapabilities::default(),
                name: String::from("Minimal CPU"),
                is_apple_silicon: false,
                backend_type: crate::hardware::cpu::CpuBackendType::Rust,
            },
            gpu: GpuInfo::default(),
            memory: MemoryInfo {
                total_gb: 1,
                available_gb: 1,
            },
            hyperthreading: HyperthreadTopology::default(),
            cache: CacheTopology::default(),
            numa: NumaTopology::default(),
        }
    }

    /// 测试：硬件模块公开API的完整链路测试（端到端验证）
    #[test]
    fn test_full_api_chain() {
        // 完整的检测 -> 分级 -> 策略推荐链路
        let profile = detect_hardware();

        // Step 1: 检测
        assert!(profile.cpu.physical_cores >= 1);
        assert!(profile.memory.total_gb >= 1);

        // Step 2: 分级
        let classifier = HardwareClassifier::new(profile.clone());
        let level = classifier.level();
        assert!(level >= HardwareLevel::Entry);
        assert!(level <= HardwareLevel::Server);

        // Step 3: 策略推荐
        let strategy = classifier.recommended_strategy();
        assert!(strategy.use_simd); // 所有级别都应该使用SIMD
        assert!(strategy.batch_size >= 1);
        assert!(strategy.max_seq_len >= 2048);

        // 验证策略与级别的对应关系（基本一致性检查）
        match level {
            HardwareLevel::Entry => {
                // Entry 级别可能不支持 GPU，这是预期行为
                let _gpu_supported = strategy.use_gpu;
                let _ = _gpu_supported; // Use variable to indicate GPU support is checked
            }
            HardwareLevel::Standard | HardwareLevel::Professional | HardwareLevel::Server => {
                // 更高级别应该有更强大的配置
                assert!(strategy.batch_size >= 1);
            }
        }
    }
}
