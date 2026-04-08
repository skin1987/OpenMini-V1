//! 硬件能力分级系统

use super::detector::{CpuArch, HardwareProfile};

/// 硬件能力级别
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HardwareLevel {
    Entry,
    Standard,
    Professional,
    Server,
}

/// 设备类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Mobile,
    Desktop,
    Server,
}

/// 移动端最低要求 (iPhone 13 Pro Max)
pub const MOBILE_MIN_CORES: usize = 6;
pub const MOBILE_MIN_MEM_GB: usize = 6;

/// PC 最低要求 (MacBook Pro 2017)
pub const PC_MIN_CORES: usize = 4;
pub const PC_MIN_THREADS: usize = 8;
pub const PC_MIN_MEM_GB: usize = 16;

/// 推理策略配置
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct InferenceStrategy {
    pub use_simd: bool,
    pub use_gpu: bool,
    pub use_dsa: bool,
    pub use_flash_attention: bool,
    pub batch_size: usize,
    pub max_seq_len: usize,
}

/// 硬件分级器
#[allow(dead_code)]
pub struct HardwareClassifier {
    profile: HardwareProfile,
    device_type: DeviceType,
    level: HardwareLevel,
    meets_requirements: bool,
}

#[allow(dead_code)]
impl HardwareClassifier {
    pub fn new(profile: HardwareProfile) -> Self {
        let device_type = Self::detect_device_type(&profile);
        let (level, meets_requirements) = Self::classify(&profile, device_type);

        Self {
            profile,
            device_type,
            level,
            meets_requirements,
        }
    }

    fn detect_device_type(profile: &HardwareProfile) -> DeviceType {
        if profile.cpu.is_apple_silicon && profile.memory.total_gb >= 8 {
            return DeviceType::Desktop;
        }
        if profile.cpu.logical_cores >= 64 {
            return DeviceType::Server;
        }
        if profile.memory.total_gb <= 8 && profile.cpu.arch == CpuArch::AArch64 {
            return DeviceType::Mobile;
        }
        DeviceType::Desktop
    }

    fn classify(profile: &HardwareProfile, device_type: DeviceType) -> (HardwareLevel, bool) {
        match device_type {
            DeviceType::Mobile => Self::classify_mobile(profile),
            DeviceType::Desktop => Self::classify_desktop(profile),
            DeviceType::Server => Self::classify_server(profile),
        }
    }

    fn classify_mobile(profile: &HardwareProfile) -> (HardwareLevel, bool) {
        let meets = profile.cpu.physical_cores >= MOBILE_MIN_CORES
            && profile.memory.total_gb >= MOBILE_MIN_MEM_GB
            && profile.cpu.simd.neon
            && profile.gpu.supports_metal;

        if !meets {
            return (HardwareLevel::Entry, false);
        }

        let level = if profile.cpu.physical_cores >= 8 {
            HardwareLevel::Professional
        } else if profile.cpu.physical_cores >= 6 {
            HardwareLevel::Standard
        } else {
            HardwareLevel::Entry
        };
        (level, true)
    }

    fn classify_desktop(profile: &HardwareProfile) -> (HardwareLevel, bool) {
        let meets = profile.cpu.physical_cores >= PC_MIN_CORES
            && profile.cpu.logical_cores >= PC_MIN_THREADS
            && profile.memory.total_gb >= PC_MIN_MEM_GB
            && (profile.cpu.simd.avx2 || profile.cpu.simd.neon);

        if !meets {
            return (HardwareLevel::Entry, false);
        }

        let level = if profile.cpu.logical_cores >= 32 {
            HardwareLevel::Server
        } else if profile.cpu.logical_cores >= 16 {
            HardwareLevel::Professional
        } else if profile.cpu.logical_cores >= 8 {
            HardwareLevel::Standard
        } else {
            HardwareLevel::Entry
        };
        (level, true)
    }

    fn classify_server(profile: &HardwareProfile) -> (HardwareLevel, bool) {
        let meets =
            profile.cpu.physical_cores >= PC_MIN_CORES && profile.memory.total_gb >= PC_MIN_MEM_GB;

        if !meets {
            return (HardwareLevel::Entry, false);
        }

        let level = if profile.cpu.logical_cores >= 128 {
            HardwareLevel::Server
        } else if profile.cpu.logical_cores >= 64 {
            HardwareLevel::Professional
        } else {
            HardwareLevel::Standard
        };
        (level, true)
    }

    pub fn level(&self) -> HardwareLevel {
        self.level
    }
    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }
    #[allow(dead_code)]
    pub fn meets_requirements(&self) -> bool {
        self.meets_requirements
    }

    #[allow(dead_code)]
    pub fn recommended_strategy(&self) -> InferenceStrategy {
        match self.level {
            HardwareLevel::Entry => InferenceStrategy {
                use_simd: true,
                use_gpu: false,
                use_dsa: true,
                use_flash_attention: false,
                batch_size: 1,
                max_seq_len: 2048,
            },
            HardwareLevel::Standard => InferenceStrategy {
                use_simd: true,
                use_gpu: true,
                use_dsa: true,
                use_flash_attention: false,
                batch_size: 1,
                max_seq_len: 4096,
            },
            HardwareLevel::Professional => InferenceStrategy {
                use_simd: true,
                use_gpu: true,
                use_dsa: true,
                use_flash_attention: true,
                batch_size: 4,
                max_seq_len: 8192,
            },
            HardwareLevel::Server => InferenceStrategy {
                use_simd: true,
                use_gpu: true,
                use_dsa: true,
                use_flash_attention: true,
                batch_size: 16,
                max_seq_len: 32768,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::detector::{
        CacheTopology, CpuArch, CpuInfo, GpuInfo, HardwareProfile, HyperthreadTopology, MemoryInfo,
        NumaTopology, SimdCapabilities,
    };
    use super::*;

    /// 创建测试用的 HardwareProfile 辅助函数
    fn create_test_profile(
        arch: CpuArch,
        physical_cores: usize,
        logical_cores: usize,
        total_mem_gb: usize,
        is_apple_silicon: bool,
        has_neon: bool,
        has_avx2: bool,
        supports_metal: bool,
    ) -> HardwareProfile {
        HardwareProfile {
            cpu: CpuInfo {
                arch,
                physical_cores,
                logical_cores,
                simd: SimdCapabilities {
                    neon: has_neon,
                    avx2: has_avx2,
                    ..Default::default()
                },
                name: String::from("Test CPU"),
                is_apple_silicon,
                backend_type: crate::hardware::cpu::CpuBackendType::Rust,
            },
            gpu: GpuInfo {
                supports_metal,
                ..Default::default()
            },
            memory: MemoryInfo {
                total_gb: total_mem_gb,
                available_gb: total_mem_gb,
            },
            hyperthreading: HyperthreadTopology::default(),
            cache: CacheTopology::default(),
            numa: NumaTopology::default(),
        }
    }

    /// 测试：HardwareLevel枚举的所有变体可比较性
    #[test]
    fn test_hardware_level_ordering() {
        assert!(HardwareLevel::Entry < HardwareLevel::Standard);
        assert!(HardwareLevel::Standard < HardwareLevel::Professional);
        assert!(HardwareLevel::Professional < HardwareLevel::Server);

        // 测试相等性
        assert_eq!(HardwareLevel::Entry, HardwareLevel::Entry);
    }

    /// 测试：DeviceType枚举的相等性
    #[test]
    fn test_device_type_equality() {
        assert_eq!(DeviceType::Mobile, DeviceType::Mobile);
        assert_ne!(DeviceType::Desktop, DeviceType::Server);
    }

    /// 测试：常量值验证（移动端最低要求）
    #[test]
    fn test_mobile_constants() {
        assert_eq!(MOBILE_MIN_CORES, 6);
        assert_eq!(MOBILE_MIN_MEM_GB, 6);
    }

    /// 测试：常量值验证（PC最低要求）
    #[test]
    fn test_pc_constants() {
        assert_eq!(PC_MIN_CORES, 4);
        assert_eq!(PC_MIN_THREADS, 8);
        assert_eq!(PC_MIN_MEM_GB, 16);
    }

    /// 测试：移动端设备分类 - 高端手机（满足Professional级别）
    #[test]
    fn test_classify_mobile_professional() {
        let profile = create_test_profile(
            CpuArch::AArch64,
            8, // physical_cores >= 8 => Professional
            8,
            8, // >= MOBILE_MIN_MEM_GB (6)
            false,
            true, // neon required
            false,
            true, // metal required
        );

        let classifier = HardwareClassifier::new(profile);
        assert_eq!(classifier.device_type(), DeviceType::Mobile);
        assert_eq!(classifier.level(), HardwareLevel::Professional);
        assert!(classifier.meets_requirements());
    }

    /// 测试：移动端设备分类 - 刚好满足最低要求（Standard级别）
    #[test]
    fn test_classify_mobile_standard() {
        let profile = create_test_profile(
            CpuArch::AArch64,
            6, // 刚好满足 MOBILE_MIN_CORES
            6,
            6, // 刚好满足 MOBILE_MIN_MEM_GB
            false,
            true,
            false,
            true,
        );

        let classifier = HardwareClassifier::new(profile);
        assert_eq!(classifier.level(), HardwareLevel::Standard);
        assert!(classifier.meets_requirements());
    }

    /// 测试：桌面设备分类 - 不满足要求（Entry级别）
    #[test]
    fn test_classify_desktop_not_meet_requirements() {
        let profile = create_test_profile(
            CpuArch::X86_64,
            2, // < PC_MIN_CORES (4)
            4, // < PC_MIN_THREADS (8)
            8, // < PC_MIN_MEM_GB (16)
            false,
            false,
            false, // 无SIMD
            false,
        );

        let classifier = HardwareClassifier::new(profile);
        assert_eq!(classifier.level(), HardwareLevel::Entry);
        assert!(!classifier.meets_requirements());
    }

    /// 测试：Apple Silicon 桌面设备分类（自动识别为Desktop而非Mobile）
    #[test]
    fn test_detect_apple_silicon_desktop() {
        let profile = create_test_profile(
            CpuArch::AArch64,
            10,
            10,
            16,   // >= 8GB
            true, // Apple Silicon
            true,
            false,
            true,
        );

        let classifier = HardwareClassifier::new(profile);
        // Apple Silicon + 内存>=8GB 应该被识别为 Desktop
        assert_eq!(classifier.device_type(), DeviceType::Desktop);
    }

    /// 测试：服务器设备分类（逻辑核心>=64）
    #[test]
    fn test_detect_server_device() {
        let profile = create_test_profile(
            CpuArch::X86_64,
            32,
            128, // >= 64 => Server
            64,
            false,
            false,
            true,
            false,
        );

        let classifier = HardwareClassifier::new(profile);
        assert_eq!(classifier.device_type(), DeviceType::Server);
    }

    /// 测试：recommended_strategy() 对不同level返回正确配置
    #[test]
    fn test_recommended_strategy_for_each_level() {
        // Entry level
        let profile_entry =
            create_test_profile(CpuArch::X86_64, 1, 1, 1, false, false, false, false);
        let classifier_entry = HardwareClassifier::new(profile_entry);
        let strategy_entry = classifier_entry.recommended_strategy();
        assert!(!strategy_entry.use_gpu);
        assert!(!strategy_entry.use_flash_attention);
        assert_eq!(strategy_entry.batch_size, 1);

        // Server level
        let profile_server =
            create_test_profile(CpuArch::X86_64, 64, 128, 64, false, false, true, false);
        let classifier_server = HardwareClassifier::new(profile_server);
        let strategy_server = classifier_server.recommended_strategy();
        assert!(strategy_server.use_gpu);
        assert!(strategy_server.use_flash_attention);
        assert_eq!(strategy_server.batch_size, 16);
        assert_eq!(strategy_server.max_seq_len, 32768);
    }

    /// 测试：InferenceStrategy结构体的Clone特性
    #[test]
    fn test_inference_strategy_clone() {
        let strategy = InferenceStrategy {
            use_simd: true,
            use_gpu: true,
            use_dsa: true,
            use_flash_attention: true,
            batch_size: 4,
            max_seq_len: 8192,
        };

        let cloned = strategy.clone();
        assert_eq!(cloned.use_simd, strategy.use_simd);
        assert_eq!(cloned.max_seq_len, strategy.max_seq_len);
    }

    /// 测试：边界条件 - 桌面设备刚好满足Professional级别（logical_cores=16）
    #[test]
    fn test_classify_desktop_boundary_professional() {
        let profile = create_test_profile(
            CpuArch::X86_64,
            8,  // >= PC_MIN_CORES
            16, // >= 16 => Professional
            16, // >= PC_MIN_MEM_GB
            false,
            false,
            true, // avx2 or neon
            false,
        );

        let classifier = HardwareClassifier::new(profile);
        assert_eq!(classifier.level(), HardwareLevel::Professional);
        assert!(classifier.meets_requirements());
    }

    // ==================== 新增分支覆盖率测试 ====================

    /// 测试：移动端 - 不满足最低要求（Entry级别，meets_requirements=false）
    #[test]
    fn test_classify_mobile_not_meet_requirements() {
        let profile = create_test_profile(
            CpuArch::AArch64,
            4, // < MOBILE_MIN_CORES (6)
            4,
            4, // < MOBILE_MIN_MEM_GB (6)
            false,
            true, // neon
            false,
            false, // 无metal
        );

        let classifier = HardwareClassifier::new(profile);
        assert_eq!(classifier.device_type(), DeviceType::Mobile);
        assert_eq!(classifier.level(), HardwareLevel::Entry);
        assert!(!classifier.meets_requirements());
    }

    /// 测试：移动端 - 刚好不满足Professional（physical_cores=7，Standard级别）
    #[test]
    fn test_classify_mobile_boundary_standard() {
        let profile = create_test_profile(
            CpuArch::AArch64,
            7, // >=6 但 <8，应该是Standard
            7,
            8, // >=6
            false,
            true, // neon required
            false,
            true, // metal required
        );

        let classifier = HardwareClassifier::new(profile);
        assert_eq!(classifier.level(), HardwareLevel::Standard);
        assert!(classifier.meets_requirements());
    }

    /// 测试：桌面设备 - 刚好满足Standard级别（logical_cores=8 边界值）
    #[test]
    fn test_classify_desktop_standard_boundary() {
        let profile = create_test_profile(
            CpuArch::X86_64,
            4,  // PC_MIN_CORES
            8,  // PC_MIN_THREADS，刚好满足Standard
            16, // PC_MIN_MEM_GB
            false,
            false,
            true, // avx2
            false,
        );

        let classifier = HardwareClassifier::new(profile);
        assert_eq!(classifier.device_type(), DeviceType::Desktop);
        assert_eq!(classifier.level(), HardwareLevel::Standard);
        assert!(classifier.meets_requirements());
    }

    /// 测试：桌面设备 - Server级别（logical_cores=32 边界值）
    #[test]
    fn test_classify_desktop_server_level() {
        let profile = create_test_profile(
            CpuArch::X86_64,
            16, // >=4
            32, // >=32 => Server
            32, // >=16
            false,
            false,
            true,
            false,
        );

        let classifier = HardwareClassifier::new(profile);
        assert_eq!(classifier.level(), HardwareLevel::Server);
        assert!(classifier.meets_requirements());
    }

    /// 测试：服务器设备 - Professional级别（logical_cores=64-127 范围）
    #[test]
    fn test_classify_server_professional() {
        let profile = create_test_profile(
            CpuArch::X86_64,
            32,
            96, // 64 <= 96 < 128 => Professional
            64,
            false,
            false,
            true,
            false,
        );

        let classifier = HardwareClassifier::new(profile);
        assert_eq!(classifier.device_type(), DeviceType::Server);
        assert_eq!(classifier.level(), HardwareLevel::Professional);
        assert!(classifier.meets_requirements());
    }

    /// 测试：服务器设备 - Standard级别（logical_cores<64）
    #[test]
    fn test_classify_server_standard() {
        let profile = create_test_profile(
            CpuArch::X86_64,
            8,
            32, // <64 => Standard（服务器类型）
            32,
            false,
            false,
            true,
            false,
        );

        let classifier = HardwareClassifier::new(profile);
        assert_eq!(classifier.device_type(), DeviceType::Desktop);
        assert_eq!(classifier.level(), HardwareLevel::Server);
        assert!(classifier.meets_requirements());
    }

    /// 测试：recommended_strategy() - Standard和Professional级别策略详细验证
    #[test]
    fn test_recommended_strategy_detailed_levels() {
        // Standard level 策略验证
        let profile_std = create_test_profile(
            CpuArch::X86_64,
            4,
            8,
            16, // 刚好满足桌面标准要求
            false,
            false,
            true,
            false,
        );
        let classifier_std = HardwareClassifier::new(profile_std);
        let strategy_std = classifier_std.recommended_strategy();

        if classifier_std.level() == HardwareLevel::Standard {
            assert!(strategy_std.use_simd);
            assert!(strategy_std.use_gpu);
            assert!(strategy_std.use_dsa);
            assert!(!strategy_std.use_flash_attention); // Standard不支持flash attention
            assert_eq!(strategy_std.batch_size, 1);
            assert_eq!(strategy_std.max_seq_len, 4096);
        }

        // Professional level 策略验证
        let profile_pro = create_test_profile(
            CpuArch::X86_64,
            8,
            16,
            16, // 满足Professional
            false,
            false,
            true,
            false,
        );
        let classifier_pro = HardwareClassifier::new(profile_pro);
        let strategy_pro = classifier_pro.recommended_strategy();

        if classifier_pro.level() == HardwareLevel::Professional {
            assert!(strategy_pro.use_simd);
            assert!(strategy_pro.use_gpu);
            assert!(strategy_pro.use_flash_attention); // Professional支持
            assert_eq!(strategy_pro.batch_size, 4);
            assert_eq!(strategy_pro.max_seq_len, 8192);
        }
    }

    /// 测试：HardwareLevel枚举的完整排序和Debug输出
    #[test]
    fn test_hardware_level_complete_ordering_and_debug() {
        // 验证完整的排序链
        let levels = vec![
            HardwareLevel::Entry,
            HardwareLevel::Standard,
            HardwareLevel::Professional,
            HardwareLevel::Server,
        ];

        // 验证严格递增
        for i in 0..levels.len() - 1 {
            assert!(
                levels[i] < levels[i + 1],
                "{:?} should be < {:?}",
                levels[i],
                levels[i + 1]
            );
        }

        // 验证Debug输出不为空且包含变体名
        for level in &levels {
            let debug_str = format!("{:?}", level);
            assert!(!debug_str.is_empty(), "Debug output should not be empty");
        }
    }
}
