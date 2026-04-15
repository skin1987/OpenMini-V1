use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    DesktopHighEnd,
    Laptop,
    AppleSilicon,
    MobileDevice,
    Embedded,
}

impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DesktopHighEnd => write!(f, "Desktop (High-End)"),
            Self::Laptop => write!(f, "Laptop"),
            Self::AppleSilicon => write!(f, "Apple Silicon"),
            Self::MobileDevice => write!(f, "Mobile Device"),
            Self::Embedded => write!(f, "Embedded Device"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub vram_gb: usize,
    pub is_dedicated: bool,
}

#[derive(Debug, Clone)]
pub struct DeviceProfile {
    pub device_type: DeviceType,
    pub total_memory_gb: usize,
    pub available_memory_gb: usize,
    pub cpu_cores: usize,
    pub cpu_physical_cores: usize,
    pub gpu_info: Option<GpuInfo>,
    pub os_name: String,
    pub arch: String,
}

impl DeviceProfile {
    pub fn detect() -> Self {
        let os_name = std::env::consts::OS.to_string();
        let arch = std::env::consts::ARCH.to_string();

        let (total_memory_gb, available_memory_gb) = Self::detect_memory();
        let (cpu_cores, cpu_physical_cores) = Self::detect_cpu();
        let gpu_info = Self::detect_gpu();
        let device_type = Self::classify_device(
            &arch,
            &os_name,
            total_memory_gb,
            cpu_physical_cores,
            &gpu_info,
        );

        DeviceProfile {
            device_type,
            total_memory_gb,
            available_memory_gb,
            cpu_cores,
            cpu_physical_cores,
            gpu_info,
            os_name,
            arch,
        }
    }

    fn detect_memory() -> (usize, usize) {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory_specifics(sysinfo::MemoryRefreshKind::new().with_ram());

        let total_kb = sys.total_memory();
        let available_kb = sys.available_memory();

        let total_gb = (total_kb / (1024 * 1024)) as usize;
        let available_gb = (available_kb / (1024 * 1024)) as usize;

        (total_gb.max(1), available_gb.max(1))
    }

    fn detect_cpu() -> (usize, usize) {
        let logical_cores = num_cpus::get();
        let physical_cores = num_cpus::get_physical();

        (logical_cores, physical_cores.max(1))
    }

    fn detect_gpu() -> Option<GpuInfo> {
        #[cfg(target_os = "macos")]
        {
            if std::env::consts::ARCH == "aarch64" {
                return Some(GpuInfo {
                    name: "Apple GPU".to_string(),
                    vram_gb: 0,
                    is_dedicated: false,
                });
            }
        }

        None
    }

    fn classify_device(
        arch: &str,
        os: &str,
        total_memory_gb: usize,
        physical_cores: usize,
        gpu: &Option<GpuInfo>,
    ) -> DeviceType {
        if arch == "aarch64" && os == "macos" {
            return DeviceType::AppleSilicon;
        }

        if total_memory_gb < 4 {
            return DeviceType::Embedded;
        }

        if os == "ios" || os == "android" {
            return DeviceType::MobileDevice;
        }

        if physical_cores >= 8
            && total_memory_gb >= 16
            && gpu.as_ref().map(|g| g.is_dedicated).unwrap_or(false)
        {
            return DeviceType::DesktopHighEnd;
        }

        DeviceType::Laptop
    }

    pub fn is_low_memory(&self) -> bool {
        self.available_memory_gb < 8
    }

    pub fn is_apple_silicon(&self) -> bool {
        self.device_type == DeviceType::AppleSilicon
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub gemm_backend: String,
    pub enable_arena: bool,
    pub enable_fp8_kv: bool,
    pub max_batch_size: usize,
    pub enable_dsa: bool,
    pub dsa_threshold: usize,
    pub arena_size_mb: usize,
    pub offload_layers: bool,
}

impl RuntimeConfig {
    pub fn for_device(profile: &DeviceProfile) -> Self {
        match profile.device_type {
            DeviceType::AppleSilicon => {
                if profile.total_memory_gb >= 32 {
                    RuntimeConfig {
                        gemm_backend: "metal".to_string(),
                        enable_arena: true,
                        enable_fp8_kv: false,
                        max_batch_size: 8,
                        enable_dsa: true,
                        dsa_threshold: 512,
                        arena_size_mb: 256,
                        offload_layers: false,
                    }
                } else if profile.total_memory_gb < 16 {
                    RuntimeConfig {
                        gemm_backend: "metal".to_string(),
                        enable_arena: true,
                        enable_fp8_kv: true,
                        max_batch_size: 1,
                        enable_dsa: false,
                        dsa_threshold: 128,
                        arena_size_mb: 64,
                        offload_layers: true,
                    }
                } else {
                    RuntimeConfig {
                        gemm_backend: "metal".to_string(),
                        enable_arena: true,
                        enable_fp8_kv: false,
                        max_batch_size: 4,
                        enable_dsa: true,
                        dsa_threshold: 256,
                        arena_size_mb: 128,
                        offload_layers: false,
                    }
                }
            }
            DeviceType::DesktopHighEnd => {
                if profile
                    .gpu_info
                    .as_ref()
                    .map(|g| g.is_dedicated)
                    .unwrap_or(false)
                {
                    RuntimeConfig {
                        gemm_backend: "cuda".to_string(),
                        enable_arena: true,
                        enable_fp8_kv: false,
                        max_batch_size: 16,
                        enable_dsa: true,
                        dsa_threshold: 1024,
                        arena_size_mb: 512,
                        offload_layers: false,
                    }
                } else {
                    RuntimeConfig {
                        gemm_backend: "cpu-blas".to_string(),
                        enable_arena: true,
                        enable_fp8_kv: false,
                        max_batch_size: 8,
                        enable_dsa: true,
                        dsa_threshold: 512,
                        arena_size_mb: 256,
                        offload_layers: false,
                    }
                }
            }
            DeviceType::Laptop => {
                if profile.total_memory_gb < 16 || profile.is_low_memory() {
                    RuntimeConfig {
                        gemm_backend: "cpu-blas".to_string(),
                        enable_arena: true,
                        enable_fp8_kv: true,
                        max_batch_size: 1,
                        enable_dsa: false,
                        dsa_threshold: 128,
                        arena_size_mb: 64,
                        offload_layers: true,
                    }
                } else {
                    RuntimeConfig {
                        gemm_backend: "cpu-blas".to_string(),
                        enable_arena: true,
                        enable_fp8_kv: false,
                        max_batch_size: 4,
                        enable_dsa: true,
                        dsa_threshold: 256,
                        arena_size_mb: 128,
                        offload_layers: false,
                    }
                }
            }
            DeviceType::MobileDevice => RuntimeConfig {
                gemm_backend: "ndarray".to_string(),
                enable_arena: true,
                enable_fp8_kv: true,
                max_batch_size: 1,
                enable_dsa: false,
                dsa_threshold: 64,
                arena_size_mb: 32,
                offload_layers: true,
            },
            DeviceType::Embedded => RuntimeConfig {
                gemm_backend: "ndarray".to_string(),
                enable_arena: true,
                enable_fp8_kv: true,
                max_batch_size: 1,
                enable_dsa: false,
                dsa_threshold: 32,
                arena_size_mb: 16,
                offload_layers: true,
            },
        }
    }

    pub fn to_map(&self) -> HashMap<&str, String> {
        let mut map = HashMap::new();
        map.insert("gemm_backend", self.gemm_backend.clone());
        map.insert("enable_arena", self.enable_arena.to_string());
        map.insert("enable_fp8_kv", self.enable_fp8_kv.to_string());
        map.insert("max_batch_size", self.max_batch_size.to_string());
        map.insert("enable_dsa", self.enable_dsa.to_string());
        map.insert("dsa_threshold", self.dsa_threshold.to_string());
        map.insert("arena_size_mb", self.arena_size_mb.to_string());
        map.insert("offload_layers", self.offload_layers.to_string());
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_device_profile() {
        let profile = DeviceProfile::detect();

        assert!(profile.cpu_cores > 0);
        assert!(profile.cpu_physical_cores > 0);
        assert!(profile.total_memory_gb > 0);
        assert!(profile.available_memory_gb > 0);
        assert!(!profile.os_name.is_empty());
        assert!(!profile.arch.is_empty());
    }

    #[test]
    fn test_runtime_config_for_device() {
        let profile = DeviceProfile::detect();
        let config = RuntimeConfig::for_device(&profile);

        assert!(!config.gemm_backend.is_empty());
        assert!(config.max_batch_size > 0);
        assert!(config.arena_size_mb > 0);
    }

    #[test]
    fn test_apple_silicon_detection() {
        let profile = DeviceProfile::detect();

        if std::env::consts::ARCH == "aarch64" && std::env::consts::OS == "macos" {
            assert_eq!(profile.device_type, DeviceType::AppleSilicon);
            assert!(profile.is_apple_silicon());
        }
    }

    #[test]
    fn test_low_memory_detection() {
        let profile = DeviceProfile::detect();
        let _is_low = profile.is_low_memory();
    }

    #[test]
    fn test_config_to_map() {
        let profile = DeviceProfile::detect();
        let config = RuntimeConfig::for_device(&profile);
        let map = config.to_map();

        assert_eq!(map.len(), 8);
        assert!(map.contains_key("gemm_backend"));
        assert!(map.contains_key("max_batch_size"));
    }

    #[test]
    fn test_device_type_display() {
        assert_eq!(
            format!("{}", DeviceType::DesktopHighEnd),
            "Desktop (High-End)"
        );
        assert_eq!(format!("{}", DeviceType::Laptop), "Laptop");
        assert_eq!(format!("{}", DeviceType::AppleSilicon), "Apple Silicon");
        assert_eq!(format!("{}", DeviceType::MobileDevice), "Mobile Device");
        assert_eq!(format!("{}", DeviceType::Embedded), "Embedded Device");
    }

    #[test]
    fn test_all_device_types_have_config() {
        let types = vec![
            DeviceType::DesktopHighEnd,
            DeviceType::Laptop,
            DeviceType::AppleSilicon,
            DeviceType::MobileDevice,
            DeviceType::Embedded,
        ];

        for device_type in types {
            let profile = DeviceProfile {
                device_type,
                total_memory_gb: 16,
                available_memory_gb: 12,
                cpu_cores: 8,
                cpu_physical_cores: 4,
                gpu_info: None,
                os_name: "test".to_string(),
                arch: "test".to_string(),
            };

            let config = RuntimeConfig::for_device(&profile);
            assert!(!config.gemm_backend.is_empty());
            assert!(config.max_batch_size >= 1);
        }
    }
}
