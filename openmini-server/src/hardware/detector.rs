//! 硬件检测模块
//!
//! 自动检测硬件能力，包括 CPU、GPU、内存和特殊加速器。
//! 支持国际主流硬件和国产硬件平台。
//! 支持超线程拓扑、缓存拓扑和NUMA节点检测。

use std::collections::HashMap;
use std::fmt;

use super::cpu::{CpuBackend, CpuBackendType};

#[cfg(target_os = "linux")]
use glob::glob;

// ============================================================================
// CPU 架构枚举
// ============================================================================

/// CPU 架构类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum CpuArch {
    /// x86-64 (Intel/AMD)
    X86_64,
    /// ARM 64-bit (Apple Silicon, 飞腾, 昇腾)
    AArch64,
    /// ARM 32-bit
    Arm,
    /// LoongArch (龙芯)
    LoongArch,
    /// SW-64 (申威)
    Sw64,
    /// RISC-V
    RiscV,
    /// 未知架构
    Unknown,
}

impl fmt::Display for CpuArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CpuArch::X86_64 => write!(f, "x86_64"),
            CpuArch::AArch64 => write!(f, "aarch64"),
            CpuArch::Arm => write!(f, "arm"),
            CpuArch::LoongArch => write!(f, "loongarch"),
            CpuArch::Sw64 => write!(f, "sw64"),
            CpuArch::RiscV => write!(f, "riscv"),
            CpuArch::Unknown => write!(f, "unknown"),
        }
    }
}

// ============================================================================
// SIMD 能力
// ============================================================================

/// SIMD 指令集支持
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct SimdCapabilities {
    /// SSE4.2 支持 (x86)
    pub sse42: bool,
    /// AVX 支持 (x86)
    pub avx: bool,
    /// AVX2 支持 (x86)
    pub avx2: bool,
    /// AVX-512 支持 (x86)
    pub avx512: bool,
    /// NEON 支持 (ARM)
    pub neon: bool,
    /// SVE 支持 (ARM)
    pub sve: bool,
    /// SVE2 支持 (ARM)
    pub sve2: bool,
    /// LSX 支持 (龙芯)
    pub lsx: bool,
    /// LASX 支持 (龙芯)
    pub lasx: bool,
}

impl SimdCapabilities {
    /// 检测当前 CPU 的 SIMD 能力
    pub fn detect() -> Self {
        let mut caps = Self::default();

        #[cfg(target_arch = "x86_64")]
        {
            caps.sse42 = is_x86_feature_detected!("sse4.2");
            caps.avx = is_x86_feature_detected!("avx");
            caps.avx2 = is_x86_feature_detected!("avx2");
            caps.avx512 = is_x86_feature_detected!("avx512f");
        }

        #[cfg(target_arch = "aarch64")]
        {
            caps.neon = true; // NEON 在 AArch64 上总是可用
            #[cfg(target_feature = "sve")]
            {
                caps.sve = true;
            }
            #[cfg(target_feature = "sve2")]
            {
                caps.sve2 = true;
            }
        }

        caps
    }

    /// 获取最佳 SIMD 宽度（位）
    #[allow(dead_code)]
    pub fn best_width(&self) -> usize {
        if self.avx512 {
            512
        } else if self.avx2 || self.avx || self.sve || self.sve2 || self.lasx {
            256
        } else if self.neon || self.sse42 || self.lsx {
            128
        } else {
            0
        }
    }

    /// 是否支持任何 SIMD
    #[allow(dead_code)]
    pub fn has_simd(&self) -> bool {
        self.sse42
            || self.avx
            || self.avx2
            || self.avx512
            || self.neon
            || self.sve
            || self.sve2
            || self.lsx
            || self.lasx
    }
}

// ============================================================================
// GPU 信息
// ============================================================================

/// GPU 类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum GpuType {
    /// Apple GPU (M1/M2/M3)
    Apple,
    /// Intel 集成显卡
    IntelIntegrated,
    /// AMD 显卡
    Amd,
    /// NVIDIA 显卡
    Nvidia,
    /// 华为昇腾 NPU
    Ascend,
    /// 其他/未知
    Unknown,
}

/// GPU 信息
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GpuInfo {
    /// GPU 类型
    pub gpu_type: GpuType,
    /// GPU 名称
    pub name: String,
    /// 显存大小 (MB)
    pub memory_mb: usize,
    /// 是否支持 Metal
    pub supports_metal: bool,
    /// 是否支持 CUDA
    pub supports_cuda: bool,
    /// 是否支持 Vulkan
    pub supports_vulkan: bool,
    /// GPU 计算能力 (GFLOPS)
    pub compute_flops: Option<f64>,
    /// 内存带宽 (GB/s)
    pub memory_bandwidth: Option<f64>,
    /// 计算单元数量
    pub compute_units: Option<usize>,
    /// GPU 频率 (MHz)
    pub gpu_frequency_mhz: Option<usize>,
}

impl Default for GpuInfo {
    fn default() -> Self {
        Self {
            gpu_type: GpuType::Unknown,
            name: String::new(),
            memory_mb: 0,
            supports_metal: false,
            supports_cuda: false,
            supports_vulkan: false,
            compute_flops: None,
            memory_bandwidth: None,
            compute_units: None,
            gpu_frequency_mhz: None,
        }
    }
}

// ============================================================================
// CPU 信息
// ============================================================================

/// CPU 信息
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CpuInfo {
    /// CPU 架构
    pub arch: CpuArch,
    /// 物理核心数
    pub physical_cores: usize,
    /// 逻辑核心数
    pub logical_cores: usize,
    /// SIMD 能力
    pub simd: SimdCapabilities,
    /// CPU 名称
    pub name: String,
    /// 是否为 Apple Silicon
    pub is_apple_silicon: bool,
    /// CPU 后端类型
    pub backend_type: CpuBackendType,
}

impl CpuInfo {
    /// 检测 CPU 信息
    pub fn detect() -> Self {
        let arch = Self::detect_arch();
        let physical_cores = num_cpus::get_physical();
        let logical_cores = num_cpus::get();
        let simd = SimdCapabilities::detect();
        let name = Self::detect_cpu_name();
        let is_apple_silicon = cfg!(target_arch = "aarch64") && cfg!(target_vendor = "apple");
        let backend_type = CpuBackend::detect();

        Self {
            arch,
            physical_cores,
            logical_cores,
            simd,
            name,
            is_apple_silicon,
            backend_type,
        }
    }

    /// 检测超线程信息
    #[allow(dead_code)]
    pub fn has_hyperthreading(&self) -> bool {
        self.logical_cores > self.physical_cores
    }

    /// 获取每个物理核心的逻辑核心数
    #[allow(dead_code)]
    pub fn threads_per_core(&self) -> usize {
        if self.physical_cores > 0 {
            self.logical_cores / self.physical_cores
        } else {
            1
        }
    }

    fn detect_arch() -> CpuArch {
        #[cfg(target_arch = "x86_64")]
        {
            CpuArch::X86_64
        }

        #[cfg(target_arch = "aarch64")]
        {
            CpuArch::AArch64
        }

        #[cfg(target_arch = "arm")]
        {
            CpuArch::Arm
        }

        #[cfg(all(
            not(target_arch = "x86_64"),
            not(target_arch = "aarch64"),
            not(target_arch = "arm")
        ))]
        {
            CpuArch::Unknown
        }
    }

    fn detect_cpu_name() -> String {
        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            if let Ok(output) = Command::new("sysctl")
                .arg("-n")
                .arg("machdep.cpu.brand_string")
                .output()
            {
                return String::from_utf8_lossy(&output.stdout).trim().to_string();
            }
        }

        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
                for line in content.lines() {
                    if line.starts_with("model name") {
                        if let Some(name) = line.split(':').nth(1) {
                            return name.trim().to_string();
                        }
                    }
                }
            }
        }

        "Unknown CPU".to_string()
    }
}

// ============================================================================
// 内存信息
// ============================================================================

/// 内存信息
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MemoryInfo {
    /// 总内存 (GB)
    pub total_gb: usize,
    /// 可用内存 (GB)
    pub available_gb: usize,
}

impl MemoryInfo {
    /// 检测内存信息
    pub fn detect() -> Self {
        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            let total =
                if let Ok(output) = Command::new("sysctl").arg("-n").arg("hw.memsize").output() {
                    String::from_utf8_lossy(&output.stdout)
                        .trim()
                        .parse::<u64>()
                        .unwrap_or(0)
                        / (1024 * 1024 * 1024)
                } else {
                    0
                };

            let available = if let Ok(output) = Command::new("vm_stat").output() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                let mut free_pages: u64 = 0;
                for line in output_str.lines() {
                    if line.starts_with("Pages free:") {
                        let num_str = line.split(':').nth(1).unwrap_or("0");
                        free_pages = num_str.trim().trim_end_matches('.').parse().unwrap_or(0);
                        break;
                    }
                }
                (free_pages * 4096) / (1024 * 1024 * 1024)
            } else {
                0
            };

            Self {
                total_gb: total as usize,
                available_gb: available as usize,
            }
        }

        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                let mut total_kb: u64 = 0;
                let mut available_kb: u64 = 0;

                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        let num_str = line.split(':').nth(1).unwrap_or("0");
                        total_kb = num_str
                            .trim()
                            .split_whitespace()
                            .next()
                            .unwrap_or("0")
                            .parse()
                            .unwrap_or(0);
                    }
                    if line.starts_with("MemAvailable:") {
                        let num_str = line.split(':').nth(1).unwrap_or("0");
                        available_kb = num_str
                            .trim()
                            .split_whitespace()
                            .next()
                            .unwrap_or("0")
                            .parse()
                            .unwrap_or(0);
                    }
                }

                return Self {
                    total_gb: (total_kb / 1024 / 1024) as usize,
                    available_gb: (available_kb / 1024 / 1024) as usize,
                };
            }

            Self {
                total_gb: 0,
                available_gb: 0,
            }
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            Self {
                total_gb: 0,
                available_gb: 0,
            }
        }
    }
}

// ============================================================================
// 超线程拓扑
// ============================================================================

/// 核心类型（用于异构架构如 Apple Silicon）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreType {
    /// 性能核
    Performance,
    /// 能效核
    Efficiency,
    /// 标准核心
    Standard,
}

/// 单个物理核心信息
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PhysicalCore {
    /// 物理核心 ID
    pub id: usize,
    /// 核心类型
    pub core_type: CoreType,
    /// 对应的逻辑核心 ID 列表
    pub logical_cores: Vec<usize>,
    /// L1 数据缓存 ID
    pub l1_data_cache_id: Option<usize>,
    /// L1 指令缓存 ID
    pub l1_inst_cache_id: Option<usize>,
    /// L2 缓存 ID
    pub l2_cache_id: Option<usize>,
    /// L3 缓存 ID
    pub l3_cache_id: Option<usize>,
    /// NUMA 节点 ID
    pub numa_node_id: Option<usize>,
}

/// 超线程拓扑信息
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct HyperthreadTopology {
    /// 物理核心列表
    pub physical_cores: Vec<PhysicalCore>,
    /// 逻辑核心到物理核心的映射
    pub logical_to_physical: HashMap<usize, usize>,
    /// 是否支持超线程
    pub has_hyperthreading: bool,
    /// 每个物理核心的线程数
    pub threads_per_core: usize,
    /// 性能核数量（异构架构）
    pub performance_cores: usize,
    /// 能效核数量（异构架构）
    pub efficiency_cores: usize,
}

impl Default for HyperthreadTopology {
    fn default() -> Self {
        Self {
            physical_cores: Vec::new(),
            logical_to_physical: HashMap::new(),
            has_hyperthreading: false,
            threads_per_core: 1,
            performance_cores: 0,
            efficiency_cores: 0,
        }
    }
}

impl HyperthreadTopology {
    /// 检测超线程拓扑
    pub fn detect() -> Self {
        #[cfg(target_os = "macos")]
        {
            Self::detect_macos()
        }

        #[cfg(target_os = "linux")]
        {
            Self::detect_linux()
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            Self::default()
        }
    }

    #[cfg(target_os = "macos")]
    fn detect_macos() -> Self {
        use std::process::Command;

        let physical_cores = num_cpus::get_physical();
        let logical_cores = num_cpus::get();
        let threads_per_core = if physical_cores > 0 {
            logical_cores / physical_cores
        } else {
            1
        };

        let mut topology = Self {
            physical_cores: Vec::new(),
            logical_to_physical: HashMap::new(),
            has_hyperthreading: threads_per_core > 1,
            threads_per_core,
            performance_cores: 0,
            efficiency_cores: 0,
        };

        let is_apple_silicon = cfg!(target_arch = "aarch64");

        if is_apple_silicon {
            if let Ok(output) = Command::new("sysctl")
                .arg("-n")
                .arg("hw.perflevel0.physicalcpu")
                .output()
            {
                topology.performance_cores = String::from_utf8_lossy(&output.stdout)
                    .trim()
                    .parse()
                    .unwrap_or(0);
            }

            if let Ok(output) = Command::new("sysctl")
                .arg("-n")
                .arg("hw.perflevel1.physicalcpu")
                .output()
            {
                topology.efficiency_cores = String::from_utf8_lossy(&output.stdout)
                    .trim()
                    .parse()
                    .unwrap_or(0);
            }

            let mut logical_id = 0;
            for i in 0..topology.performance_cores {
                let mut core = PhysicalCore {
                    id: i,
                    core_type: CoreType::Performance,
                    logical_cores: Vec::new(),
                    l1_data_cache_id: Some(i),
                    l1_inst_cache_id: Some(i),
                    l2_cache_id: Some(i),
                    l3_cache_id: Some(0),
                    numa_node_id: Some(0),
                };
                for _ in 0..threads_per_core {
                    core.logical_cores.push(logical_id);
                    topology.logical_to_physical.insert(logical_id, i);
                    logical_id += 1;
                }
                topology.physical_cores.push(core);
            }

            for i in 0..topology.efficiency_cores {
                let core_id = topology.performance_cores + i;
                let mut core = PhysicalCore {
                    id: core_id,
                    core_type: CoreType::Efficiency,
                    logical_cores: Vec::new(),
                    l1_data_cache_id: Some(core_id),
                    l1_inst_cache_id: Some(core_id),
                    l2_cache_id: Some(core_id),
                    l3_cache_id: Some(0),
                    numa_node_id: Some(0),
                };
                for _ in 0..threads_per_core {
                    core.logical_cores.push(logical_id);
                    topology.logical_to_physical.insert(logical_id, core_id);
                    logical_id += 1;
                }
                topology.physical_cores.push(core);
            }
        } else {
            for i in 0..physical_cores {
                let mut core = PhysicalCore {
                    id: i,
                    core_type: CoreType::Standard,
                    logical_cores: Vec::new(),
                    l1_data_cache_id: Some(i),
                    l1_inst_cache_id: Some(i),
                    l2_cache_id: Some(i),
                    l3_cache_id: Some(0),
                    numa_node_id: Some(0),
                };
                for j in 0..threads_per_core {
                    let logical_id = i * threads_per_core + j;
                    core.logical_cores.push(logical_id);
                    topology.logical_to_physical.insert(logical_id, i);
                }
                topology.physical_cores.push(core);
            }
        }

        topology
    }

    #[cfg(target_os = "linux")]
    fn detect_linux() -> Self {
        let physical_cores = num_cpus::get_physical();
        let logical_cores = num_cpus::get();
        let threads_per_core = if physical_cores > 0 {
            logical_cores / physical_cores
        } else {
            1
        };

        let mut topology = Self {
            physical_cores: Vec::new(),
            logical_to_physical: HashMap::new(),
            has_hyperthreading: threads_per_core > 1,
            threads_per_core,
            performance_cores: physical_cores,
            efficiency_cores: 0,
        };

        let core_map = Self::build_core_map();
        let cache_info = Self::parse_cache_topology();
        let numa_info = Self::parse_numa_topology();

        let mut sorted_core_ids: Vec<usize> = core_map.keys().cloned().collect();
        sorted_core_ids.sort();

        for core_id in sorted_core_ids {
            if let Some(logical_ids) = core_map.get(&core_id) {
                let mut core = PhysicalCore {
                    id: core_id,
                    core_type: CoreType::Standard,
                    logical_cores: logical_ids.clone(),
                    l1_data_cache_id: cache_info.get(&core_id).and_then(|c| c.l1_data_cache_id),
                    l1_inst_cache_id: cache_info.get(&core_id).and_then(|c| c.l1_inst_cache_id),
                    l2_cache_id: cache_info.get(&core_id).and_then(|c| c.l2_cache_id),
                    l3_cache_id: cache_info.get(&core_id).and_then(|c| c.l3_cache_id),
                    numa_node_id: numa_info.get(&core_id).copied(),
                };

                for &logical_id in logical_ids {
                    topology.logical_to_physical.insert(logical_id, core_id);
                }

                topology.physical_cores.push(core);
            }
        }

        topology
    }

    #[cfg(target_os = "linux")]
    fn build_core_map() -> HashMap<usize, Vec<usize>> {
        let mut core_map: HashMap<usize, Vec<usize>> = HashMap::new();

        for cpu_id in 0..num_cpus::get() {
            let core_path = format!("/sys/devices/system/cpu/cpu{}/topology/core_id", cpu_id);
            if let Ok(content) = std::fs::read_to_string(&core_path) {
                if let Ok(core_id) = content.trim().parse::<usize>() {
                    core_map.entry(core_id).or_default().push(cpu_id);
                }
            }
        }

        if core_map.is_empty() {
            for i in 0..num_cpus::get_physical() {
                core_map.insert(i, vec![i]);
            }
        }

        core_map
    }

    #[cfg(target_os = "linux")]
    fn parse_cache_topology() -> HashMap<usize, CacheInfo> {
        let mut cache_map: HashMap<usize, CacheInfo> = HashMap::new();

        for cpu_id in 0..num_cpus::get() {
            let mut info = CacheInfo::default();

            let l1d_path = format!("/sys/devices/system/cpu/cpu{}/cache/index0/id", cpu_id);
            if let Ok(content) = std::fs::read_to_string(&l1d_path) {
                info.l1_data_cache_id = content.trim().parse().ok();
            }

            let l1i_path = format!("/sys/devices/system/cpu/cpu{}/cache/index1/id", cpu_id);
            if let Ok(content) = std::fs::read_to_string(&l1i_path) {
                info.l1_inst_cache_id = content.trim().parse().ok();
            }

            let l2_path = format!("/sys/devices/system/cpu/cpu{}/cache/index2/id", cpu_id);
            if let Ok(content) = std::fs::read_to_string(&l2_path) {
                info.l2_cache_id = content.trim().parse().ok();
            }

            let l3_path = format!("/sys/devices/system/cpu/cpu{}/cache/index3/id", cpu_id);
            if let Ok(content) = std::fs::read_to_string(&l3_path) {
                info.l3_cache_id = content.trim().parse().ok();
            }

            let core_path = format!("/sys/devices/system/cpu/cpu{}/topology/core_id", cpu_id);
            if let Ok(content) = std::fs::read_to_string(&core_path) {
                if let Ok(core_id) = content.trim().parse::<usize>() {
                    cache_map.insert(core_id, info);
                }
            }
        }

        cache_map
    }

    #[cfg(target_os = "linux")]
    fn parse_numa_topology() -> HashMap<usize, usize> {
        let mut numa_map: HashMap<usize, usize> = HashMap::new();

        for cpu_id in 0..num_cpus::get() {
            let path = format!("/sys/devices/system/cpu/cpu{}/node*", cpu_id);
            if let Ok(entries) = glob::glob(&path) {
                for entry in entries.flatten() {
                    if let Some(name) = entry.file_name() {
                        if let Some(node_str) = name.to_str() {
                            if node_str.starts_with("node") {
                                if let Ok(node_id) = node_str[4..].parse::<usize>() {
                                    let core_path = format!(
                                        "/sys/devices/system/cpu/cpu{}/topology/core_id",
                                        cpu_id
                                    );
                                    if let Ok(content) = std::fs::read_to_string(&core_path) {
                                        if let Ok(core_id) = content.trim().parse::<usize>() {
                                            numa_map.insert(core_id, node_id);
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        numa_map
    }

    /// 获取物理核心数
    pub fn physical_core_count(&self) -> usize {
        self.physical_cores.len()
    }

    /// 获取逻辑核心数
    pub fn logical_core_count(&self) -> usize {
        self.logical_to_physical.len()
    }

    /// 获取指定逻辑核心对应的物理核心 ID
    #[allow(dead_code)]
    pub fn get_physical_core(&self, logical_id: usize) -> Option<usize> {
        self.logical_to_physical.get(&logical_id).copied()
    }

    /// 获取最优计算核心列表（优先使用性能核）
    pub fn get_optimal_compute_cores(&self) -> Vec<usize> {
        let mut cores: Vec<usize> = self
            .physical_cores
            .iter()
            .filter(|c| c.core_type == CoreType::Performance)
            .map(|c| c.id)
            .collect();

        if cores.is_empty() {
            cores = self.physical_cores.iter().map(|c| c.id).collect();
        }

        cores
    }

    /// 获取每个物理核心的首选逻辑核心（避免超线程竞争）
    pub fn get_primary_logical_cores(&self) -> Vec<usize> {
        self.physical_cores
            .iter()
            .filter_map(|c| c.logical_cores.first().copied())
            .collect()
    }
}

#[cfg(target_os = "linux")]
#[derive(Default)]
struct CacheInfo {
    l1_data_cache_id: Option<usize>,
    l1_inst_cache_id: Option<usize>,
    l2_cache_id: Option<usize>,
    l3_cache_id: Option<usize>,
}

// ============================================================================
// 缓存拓扑
// ============================================================================

/// CPU 缓存信息
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CacheInfo {
    /// 缓存级别
    pub level: usize,
    /// 缓存大小 (KB)
    pub size_kb: usize,
    /// 缓存行大小 (字节)
    pub line_size: usize,
    /// 关联度
    pub associativity: usize,
    /// 共享此缓存的核心列表
    pub shared_cores: Vec<usize>,
}

/// 缓存拓扑信息
#[derive(Debug, Clone, Default)]
pub struct CacheTopology {
    /// L1 数据缓存列表
    pub l1_data_caches: Vec<CacheInfo>,
    /// L1 指令缓存列表
    pub l1_inst_caches: Vec<CacheInfo>,
    /// L2 缓存列表
    pub l2_caches: Vec<CacheInfo>,
    /// L3 缓存列表
    pub l3_caches: Vec<CacheInfo>,
}

impl CacheTopology {
    /// 检测缓存拓扑
    pub fn detect() -> Self {
        #[cfg(target_os = "macos")]
        {
            Self::detect_macos()
        }

        #[cfg(target_os = "linux")]
        {
            Self::detect_linux()
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            Self::default()
        }
    }

    #[cfg(target_os = "macos")]
    fn detect_macos() -> Self {
        use std::process::Command;

        let mut topology = Self::default();
        let physical_cores = num_cpus::get_physical();

        let l1d_size = if let Ok(output) = Command::new("sysctl")
            .arg("-n")
            .arg("hw.l1dcachesize")
            .output()
        {
            String::from_utf8_lossy(&output.stdout)
                .trim()
                .parse()
                .unwrap_or(32768)
                / 1024
        } else {
            32
        };

        let l1i_size = if let Ok(output) = Command::new("sysctl")
            .arg("-n")
            .arg("hw.l1icachesize")
            .output()
        {
            String::from_utf8_lossy(&output.stdout)
                .trim()
                .parse()
                .unwrap_or(32768)
                / 1024
        } else {
            32
        };

        let l2_size = if let Ok(output) = Command::new("sysctl")
            .arg("-n")
            .arg("hw.l2cachesize")
            .output()
        {
            String::from_utf8_lossy(&output.stdout)
                .trim()
                .parse()
                .unwrap_or(262144)
                / 1024
        } else {
            256
        };

        let l3_size = if let Ok(output) = Command::new("sysctl")
            .arg("-n")
            .arg("hw.l3cachesize")
            .output()
        {
            String::from_utf8_lossy(&output.stdout)
                .trim()
                .parse()
                .unwrap_or(0)
                / 1024
        } else {
            0
        };

        let cacheline_size = if let Ok(output) = Command::new("sysctl")
            .arg("-n")
            .arg("hw.cachelinesize")
            .output()
        {
            String::from_utf8_lossy(&output.stdout)
                .trim()
                .parse()
                .unwrap_or(64)
        } else {
            64
        };

        for i in 0..physical_cores {
            topology.l1_data_caches.push(CacheInfo {
                level: 1,
                size_kb: l1d_size,
                line_size: cacheline_size,
                associativity: 8,
                shared_cores: vec![i],
            });

            topology.l1_inst_caches.push(CacheInfo {
                level: 1,
                size_kb: l1i_size,
                line_size: cacheline_size,
                associativity: 8,
                shared_cores: vec![i],
            });

            topology.l2_caches.push(CacheInfo {
                level: 2,
                size_kb: l2_size,
                line_size: cacheline_size,
                associativity: 8,
                shared_cores: vec![i],
            });
        }

        if l3_size > 0 {
            topology.l3_caches.push(CacheInfo {
                level: 3,
                size_kb: l3_size,
                line_size: cacheline_size,
                associativity: 16,
                shared_cores: (0..physical_cores).collect(),
            });
        }

        topology
    }

    #[cfg(target_os = "linux")]
    fn detect_linux() -> Self {
        let mut topology = Self::default();

        for index in 0..10 {
            let base_path = format!("/sys/devices/system/cpu/cpu0/cache/index{}", index);
            if !std::path::Path::new(&base_path).exists() {
                break;
            }

            let level_path = format!("{}/level", base_path);
            let level: usize = std::fs::read_to_string(&level_path)
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0);

            let size_path = format!("{}/size", base_path);
            let size_kb: usize = std::fs::read_to_string(&size_path)
                .ok()
                .and_then(|s| {
                    let s = s.trim();
                    if s.ends_with('K') {
                        s[..s.len() - 1].parse().ok()
                    } else {
                        s.parse().ok()
                    }
                })
                .unwrap_or(0);

            let line_size_path = format!("{}/coherency_line_size", base_path);
            let line_size: usize = std::fs::read_to_string(&line_size_path)
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(64);

            let type_path = format!("{}/type", base_path);
            let cache_type: String = std::fs::read_to_string(&type_path)
                .ok()
                .map(|s| s.trim().to_string())
                .unwrap_or_default();

            let shared_path = format!("{}/shared_cpu_list", base_path);
            let shared_cores = Self::parse_cpu_list(&shared_path);

            let cache_info = CacheInfo {
                level,
                size_kb,
                line_size,
                associativity: 8,
                shared_cores,
            };

            match (level, cache_type.as_str()) {
                (1, "Data") => topology.l1_data_caches.push(cache_info),
                (1, "Instruction") => topology.l1_inst_caches.push(cache_info),
                (1, "Unified") => {
                    topology.l1_data_caches.push(cache_info.clone());
                    topology.l1_inst_caches.push(cache_info);
                }
                (2, _) => topology.l2_caches.push(cache_info),
                (3, _) => topology.l3_caches.push(cache_info),
                _ => {}
            }
        }

        topology
    }

    #[cfg(target_os = "linux")]
    fn parse_cpu_list(path: &str) -> Vec<usize> {
        let mut cores = Vec::new();
        if let Ok(content) = std::fs::read_to_string(path) {
            for part in content.trim().split(',') {
                if part.contains('-') {
                    let range: Vec<&str> = part.split('-').collect();
                    if range.len() == 2 {
                        if let (Ok(start), Ok(end)) =
                            (range[0].parse::<usize>(), range[1].parse::<usize>())
                        {
                            for id in start..=end {
                                cores.push(id);
                            }
                        }
                    }
                } else if let Ok(id) = part.parse::<usize>() {
                    cores.push(id);
                }
            }
        }
        cores
    }

    /// 获取总 L3 缓存大小 (KB)
    #[allow(dead_code)]
    pub fn total_l3_size_kb(&self) -> usize {
        self.l3_caches.iter().map(|c| c.size_kb).sum()
    }

    /// 获取缓存行大小
    #[allow(dead_code)]
    pub fn cache_line_size(&self) -> usize {
        self.l1_data_caches
            .first()
            .map(|c| c.line_size)
            .unwrap_or(64)
    }
}

// ============================================================================
// NUMA 拓扑
// ============================================================================

/// NUMA 节点信息
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct NumaNode {
    /// 节点 ID
    pub id: usize,
    /// 节点上的 CPU 核心列表
    pub cpus: Vec<usize>,
    /// 节点内存大小 (MB)
    pub memory_mb: usize,
    /// 节点距离矩阵索引
    pub distance: Vec<usize>,
}

/// NUMA 拓扑信息
#[derive(Debug, Clone)]
#[allow(dead_code)]
#[derive(Default)]
pub struct NumaTopology {
    /// NUMA 节点列表
    pub nodes: Vec<NumaNode>,
    /// 是否为 NUMA 架构
    pub is_numa: bool,
}

impl NumaTopology {
    /// 检测 NUMA 拓扑
    pub fn detect() -> Self {
        #[cfg(target_os = "linux")]
        {
            Self::detect_linux()
        }

        #[cfg(target_os = "macos")]
        {
            Self::detect_macos()
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            Self::default()
        }
    }

    #[cfg(target_os = "macos")]
    fn detect_macos() -> Self {
        Self {
            nodes: vec![NumaNode {
                id: 0,
                cpus: (0..num_cpus::get()).collect(),
                memory_mb: 0,
                distance: vec![0],
            }],
            is_numa: false,
        }
    }

    #[cfg(target_os = "linux")]
    fn detect_linux() -> Self {
        let mut topology = Self::default();

        let node_dirs: Vec<_> = glob::glob("/sys/devices/system/node/node*")
            .ok()
            .into_iter()
            .flatten()
            .flatten()
            .collect();

        if node_dirs.is_empty() {
            topology.nodes.push(NumaNode {
                id: 0,
                cpus: (0..num_cpus::get()).collect(),
                memory_mb: 0,
                distance: vec![0],
            });
            return topology;
        }

        topology.is_numa = node_dirs.len() > 1;

        for node_dir in node_dirs {
            let node_name = node_dir.file_name().and_then(|n| n.to_str()).unwrap_or("");

            if !node_name.starts_with("node") {
                continue;
            }

            let node_id: usize = node_name[4..].parse().unwrap_or(0);

            let cpus_path = node_dir.join("cpulist");
            let cpus = Self::parse_cpu_list(&cpus_path);

            let mem_path = node_dir.join("meminfo");
            let memory_mb = std::fs::read_to_string(&mem_path)
                .ok()
                .and_then(|s| {
                    for line in s.lines() {
                        if line.contains("MemTotal") {
                            let parts: Vec<&str> = line.split_whitespace().collect();
                            if parts.len() >= 4 {
                                return parts[3].parse().ok();
                            }
                        }
                    }
                    None
                })
                .unwrap_or(0);

            let distance_path = node_dir.join("distance");
            let distance = std::fs::read_to_string(&distance_path)
                .ok()
                .map(|s| {
                    s.split_whitespace()
                        .filter_map(|d| d.parse().ok())
                        .collect()
                })
                .unwrap_or_default();

            topology.nodes.push(NumaNode {
                id: node_id,
                cpus,
                memory_mb,
                distance,
            });
        }

        topology.nodes.sort_by_key(|n| n.id);
        topology
    }

    #[cfg(target_os = "linux")]
    fn parse_cpu_list(path: &std::path::PathBuf) -> Vec<usize> {
        let mut cpus = Vec::new();
        if let Ok(content) = std::fs::read_to_string(path) {
            for part in content.trim().split(',') {
                if part.contains('-') {
                    let range: Vec<&str> = part.split('-').collect();
                    if range.len() == 2 {
                        if let (Ok(start), Ok(end)) =
                            (range[0].parse::<usize>(), range[1].parse::<usize>())
                        {
                            for id in start..=end {
                                cpus.push(id);
                            }
                        }
                    }
                } else if let Ok(id) = part.parse::<usize>() {
                    cpus.push(id);
                }
            }
        }
        cpus
    }

    /// 获取 NUMA 节点数量
    #[allow(dead_code)]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// 获取指定 CPU 所在的 NUMA 节点
    #[allow(dead_code)]
    pub fn get_node_for_cpu(&self, cpu_id: usize) -> Option<&NumaNode> {
        self.nodes.iter().find(|n| n.cpus.contains(&cpu_id))
    }

    /// 获取最优 NUMA 节点（内存最大）
    pub fn get_optimal_node(&self) -> Option<&NumaNode> {
        self.nodes.iter().max_by_key(|n| n.memory_mb)
    }
}

// ============================================================================
// 硬件配置
// ============================================================================

/// 完整硬件配置
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct HardwareProfile {
    /// CPU 信息
    pub cpu: CpuInfo,
    /// GPU 信息
    pub gpu: GpuInfo,
    /// 内存信息
    pub memory: MemoryInfo,
    /// 超线程拓扑
    pub hyperthreading: HyperthreadTopology,
    /// 缓存拓扑
    pub cache: CacheTopology,
    /// NUMA 拓扑
    pub numa: NumaTopology,
}

impl HardwareProfile {
    /// 检测硬件配置
    pub fn detect() -> Self {
        let cpu = CpuInfo::detect();
        let gpu = GpuInfo::detect_gpu();
        let memory = MemoryInfo::detect();
        let hyperthreading = HyperthreadTopology::detect();
        let cache = CacheTopology::detect();
        let numa = NumaTopology::detect();

        Self {
            cpu,
            gpu,
            memory,
            hyperthreading,
            cache,
            numa,
        }
    }

    /// 获取硬件摘要
    #[allow(dead_code)]
    pub fn summary(&self) -> String {
        let ht_info = if self.hyperthreading.has_hyperthreading {
            format!(
                " (HT: {} threads/core, {} P-cores, {} E-cores)",
                self.hyperthreading.threads_per_core,
                self.hyperthreading.performance_cores,
                self.hyperthreading.efficiency_cores
            )
        } else {
            String::new()
        };

        let numa_info = if self.numa.is_numa {
            format!(" [NUMA: {} nodes]", self.numa.node_count())
        } else {
            String::new()
        };

        format!(
            "CPU: {} ({} cores, {} threads, SIMD: {}-bit){}\n\
             GPU: {} ({} MB)\n\
             Memory: {} GB ({} GB available){}\n\
             Cache: L3 {} KB, Line {} bytes",
            self.cpu.name,
            self.cpu.physical_cores,
            self.cpu.logical_cores,
            self.cpu.simd.best_width(),
            ht_info,
            self.gpu.name,
            self.gpu.memory_mb,
            self.memory.total_gb,
            self.memory.available_gb,
            numa_info,
            self.cache.total_l3_size_kb(),
            self.cache.cache_line_size(),
        )
    }
}

impl GpuInfo {
    /// 检测 GPU 信息
    pub fn detect_gpu() -> Self {
        #[cfg(target_os = "macos")]
        {
            Self::detect_macos_gpu()
        }

        #[cfg(target_os = "linux")]
        {
            Self::detect_linux_gpu()
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            Self::default()
        }
    }

    #[cfg(target_os = "macos")]
    #[allow(clippy::field_reassign_with_default)]
    fn detect_macos_gpu() -> Self {
        use std::process::Command;

        let mut gpu = Self::default();
        gpu.supports_metal = true;

        if cfg!(target_arch = "aarch64") {
            gpu.gpu_type = GpuType::Apple;
            gpu.name = "Apple GPU".to_string();

            if let Ok(output) = Command::new("sysctl").arg("-n").arg("hw.memsize").output() {
                let total_mem: u64 = String::from_utf8_lossy(&output.stdout)
                    .trim()
                    .parse()
                    .unwrap_or(0);
                gpu.memory_mb = (total_mem / (1024 * 1024)) as usize;
            }

            // Apple Silicon GPU 计算能力估算
            // M1/M2/M3 系列约 2.6 TFLOPS, 带宽约 100-800 GB/s
            gpu.compute_flops = Some(2600.0); // GFLOPS
            gpu.memory_bandwidth = Some(100.0); // GB/s (保守估计)
            gpu.compute_units = Some(8); // GPU cores
            gpu.gpu_frequency_mhz = Some(1278); // MHz
        } else if let Ok(output) = Command::new("system_profiler")
            .arg("SPDisplaysDataType")
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            if output_str.contains("Intel") {
                gpu.gpu_type = GpuType::IntelIntegrated;
                gpu.name = "Intel GPU".to_string();
                gpu.compute_flops = Some(400.0);
                gpu.memory_bandwidth = Some(25.0);
            } else if output_str.contains("AMD") || output_str.contains("Radeon") {
                gpu.gpu_type = GpuType::Amd;
                gpu.name = "AMD GPU".to_string();
                gpu.compute_flops = Some(2000.0);
                gpu.memory_bandwidth = Some(200.0);
            }
        }

        gpu
    }

    #[cfg(target_os = "linux")]
    fn detect_linux_gpu() -> Self {
        let mut gpu = Self::default();

        // 检测 NVIDIA GPU
        if std::path::Path::new("/proc/driver/nvidia/version").exists() {
            gpu.gpu_type = GpuType::Nvidia;
            gpu.supports_cuda = true;
            gpu.name = "NVIDIA GPU".to_string();
        }

        // 检测华为昇腾 NPU
        if std::path::Path::new("/usr/local/Ascend").exists() {
            gpu.gpu_type = GpuType::Ascend;
            gpu.name = "Huawei Ascend NPU".to_string();
        }

        gpu
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_cpu() {
        let cpu = CpuInfo::detect();
        assert!(cpu.physical_cores > 0);
        assert!(cpu.logical_cores >= cpu.physical_cores);
    }

    #[test]
    fn test_detect_memory() {
        let memory = MemoryInfo::detect();
        assert!(memory.total_gb > 0);
    }

    #[test]
    fn test_simd_capabilities() {
        let simd = SimdCapabilities::detect();
        // 至少应该有某种 SIMD 支持
        #[cfg(target_arch = "x86_64")]
        assert!(simd.sse42);

        #[cfg(target_arch = "aarch64")]
        assert!(simd.neon);
    }

    #[test]
    fn test_hardware_profile() {
        let profile = HardwareProfile::detect();
        let summary = profile.summary();
        assert!(!summary.is_empty());
    }

    #[test]
    fn test_hyperthread_topology() {
        let topology = HyperthreadTopology::detect();
        assert!(topology.physical_core_count() > 0);
        assert!(topology.logical_core_count() > 0);
        assert!(topology.logical_core_count() >= topology.physical_core_count());

        // 测试逻辑核心到物理核心映射
        if topology.has_hyperthreading {
            assert!(topology.threads_per_core > 1);
        }

        // 测试获取首选逻辑核心
        let primary_cores = topology.get_primary_logical_cores();
        assert_eq!(primary_cores.len(), topology.physical_core_count());
    }

    #[test]
    fn test_cache_topology() {
        let cache = CacheTopology::detect();
        assert!(!cache.l1_data_caches.is_empty());
        assert!(!cache.l1_inst_caches.is_empty());

        // 验证缓存行大小合理
        let line_size = cache.cache_line_size();
        assert!(line_size >= 16 && line_size <= 256);
    }

    #[test]
    fn test_numa_topology() {
        let numa = NumaTopology::detect();
        assert!(numa.node_count() > 0);

        // 单节点系统也应该有至少一个节点
        let node = numa.get_optimal_node();
        assert!(node.is_some());
    }

    #[test]
    fn test_cpu_hyperthreading_info() {
        let cpu = CpuInfo::detect();

        // 测试超线程检测
        let has_ht = cpu.has_hyperthreading();
        let threads_per_core = cpu.threads_per_core();

        if has_ht {
            assert!(threads_per_core > 1);
        } else {
            assert_eq!(threads_per_core, 1);
        }
    }

    #[test]
    fn test_optimal_compute_cores() {
        let topology = HyperthreadTopology::detect();
        let optimal_cores = topology.get_optimal_compute_cores();

        // 应该返回至少一个核心
        assert!(!optimal_cores.is_empty());

        // 对于 Apple Silicon，应该优先返回性能核
        #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
        {
            if topology.performance_cores > 0 {
                assert_eq!(optimal_cores.len(), topology.performance_cores);
            }
        }
    }

    #[test]
    fn test_hardware_detector_basic_info() {
        // 基本硬件信息检测
        let profile = HardwareProfile::detect();

        // 不panic即可,字段可能有值也可能为None
        let _cpu_name = profile.cpu.name;
        let _cpu_cores = profile.cpu.physical_cores;
        let _total_memory = profile.memory.total_gb;

        // 验证基本信息不为空(或为0)
        assert!(!_cpu_name.is_empty() || _cpu_name == "Unknown CPU");
    }

    #[test]
    fn test_hardware_level_determination() {
        // 硬件级别判定 - 基于内存和核心数
        let profile = HardwareProfile::detect();

        // 简单的硬件级别判定逻辑
        let memory_gb = profile.memory.total_gb;
        let cores = profile.cpu.physical_cores;

        let level = match (memory_gb, cores) {
            (m, _) if m >= 64 => "Ultra",          // 64GB+ 内存
            (m, c) if m >= 32 && c >= 8 => "High", // 32GB+ 且 8+ 核
            (m, c) if m >= 16 && c >= 4 => "Mid",  // 16GB+ 且 4+ 核
            _ => "Low",                            // 其他
        };

        // 级别应该在有效范围内
        assert!(["Low", "Mid", "High", "Ultra"].contains(&level));

        // 打印硬件信息用于调试
        println!(
            "Hardware Level: {} ({} GB, {} cores)",
            level, memory_gb, cores
        );
    }

    #[test]
    fn test_simd_capabilities_detection() {
        // SIMD能力检测
        let simd = SimdCapabilities::detect();

        // 至少应该有某种SIMD支持(在x86_64或aarch64上)
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            // 应该至少有一种SIMD指令集
            let _has_any_simd = simd.sse42
                || simd.avx
                || simd.avx2
                || simd.avx512
                || simd.neon
                || simd.sve
                || simd.sve2
                || simd.lsx
                || simd.lasx;

            // x86_64 应该有 SSE4.2 或更高
            #[cfg(target_arch = "x86_64")]
            {
                assert!(simd.sse42, "x86_64 should have SSE4.2 support");
            }

            // aarch64 应该有 NEON
            #[cfg(target_arch = "aarch64")]
            {
                assert!(simd.neon, "aarch64 should have NEON support");
            }
        }

        // 测试 best_width 方法
        let width = simd.best_width();
        // 宽度应该是0, 128, 256, 或 512
        assert!(
            width == 0 || width == 128 || width == 256 || width == 512,
            "Invalid SIMD width: {}",
            width
        );

        // 测试 has_simd 方法
        let has_simd = simd.has_simd();
        assert_eq!(has_simd, width > 0);
    }

    #[test]
    fn test_cpu_arch_detection() {
        // CPU架构检测
        let cpu = CpuInfo::detect();

        // 验证架构被正确识别
        #[cfg(target_arch = "x86_64")]
        assert_eq!(cpu.arch, CpuArch::X86_64);

        #[cfg(target_arch = "aarch64")]
        assert_eq!(cpu.arch, CpuArch::AArch64);

        #[cfg(target_arch = "arm")]
        assert_eq!(cpu.arch, CpuArch::Arm);

        // 测试 Display trait
        let arch_str = format!("{}", cpu.arch);
        assert!(!arch_str.is_empty());
        assert!(!arch_str.contains("Unknown") || cpu.arch == CpuArch::Unknown);
    }

    #[test]
    fn test_cpu_core_relationships() {
        // CPU核心关系验证
        let cpu = CpuInfo::detect();

        // 物理核心应该 <= 逻辑核心
        assert!(
            cpu.physical_cores <= cpu.logical_cores,
            "Physical cores ({}) should be <= logical cores ({})",
            cpu.physical_cores,
            cpu.logical_cores
        );

        // 都应该大于0
        assert!(
            cpu.physical_cores > 0,
            "Should have at least 1 physical core"
        );
        assert!(cpu.logical_cores > 0, "Should have at least 1 logical core");

        // 测试超线程信息方法
        let has_ht = cpu.has_hyperthreading();
        let tpc = cpu.threads_per_core();

        if has_ht {
            assert!(
                tpc > 1,
                "With hyperthreading, threads per core should be > 1"
            );
        } else {
            assert_eq!(
                tpc, 1,
                "Without hyperthreading, threads per core should be 1"
            );
        }

        // Apple Silicon 特定检查
        #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
        {
            assert!(
                cpu.is_apple_silicon,
                "Should detect as Apple Silicon on macOS ARM"
            );
        }
    }

    #[test]
    fn test_gpu_info_detection() {
        // GPU信息检测
        let gpu = GpuInfo::detect_gpu();

        // 至少不应该panic
        let _gpu_type = gpu.gpu_type;
        let gpu_name = &gpu.name;
        let _memory_mb = gpu.memory_mb;

        // 在macOS aarch64上应该支持Metal
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            assert!(gpu.supports_metal, "Apple Silicon should support Metal");
            assert_eq!(gpu.gpu_type, GpuType::Apple);
        }

        // 如果有GPU,名称不应该为空(除非是默认值)
        if _memory_mb > 0 {
            assert!(
                !gpu_name.is_empty()
                    || *gpu_name == "Apple GPU"
                    || *gpu_name == "NVIDIA GPU"
                    || *gpu_name == "AMD GPU"
                    || *gpu_name == "Intel GPU"
                    || *gpu_name == "Huawei Ascend NPU"
            );
        }
    }

    #[test]
    fn test_memory_info_validation() {
        // 内存信息验证
        let memory = MemoryInfo::detect();

        // 总内存应该大于可用内存
        assert!(
            memory.total_gb >= memory.available_gb,
            "Total memory ({}) should be >= available memory ({})",
            memory.total_gb,
            memory.available_gb
        );

        // 总内存应该大于0(在实际系统上)
        // 注意:在某些容器环境中可能为0,所以只做基本检查
        if memory.total_gb > 0 {
            assert!(memory.available_gb <= memory.total_gb);
        }
    }

    #[test]
    fn test_topology_consistency_checks() {
        // 拓扑一致性检查
        let ht = HyperthreadTopology::detect();

        // 物理核心数应该匹配CPU检测
        let cpu = CpuInfo::detect();
        assert_eq!(
            ht.physical_core_count(),
            cpu.physical_cores,
            "Hyperthread topology physical cores should match CPU info"
        );

        // 逻辑核心数也应该匹配
        assert_eq!(
            ht.logical_core_count(),
            cpu.logical_cores,
            "Hyperthread topology logical cores should match CPU info"
        );

        // 测试首选逻辑核心数量
        let primary_cores = ht.get_primary_logical_cores();
        assert_eq!(
            primary_cores.len(),
            ht.physical_core_count(),
            "Primary logical cores count should equal physical core count"
        );

        // 所有首选核心应该在逻辑到物理映射中存在
        for &logical_id in &primary_cores {
            assert!(
                ht.logical_to_physical.contains_key(&logical_id),
                "Logical core {} should have a mapping to physical core",
                logical_id
            );
        }
    }

    #[test]
    fn test_cache_topology_basic() {
        // 缓存拓扑基本验证
        let cache = CacheTopology::detect();

        // L1数据缓存和指令缓存数量应该相等(通常)
        assert_eq!(
            cache.l1_data_caches.len(),
            cache.l1_inst_caches.len(),
            "L1 data and instruction cache counts should match"
        );

        // 缓存数量应该与物理核心数相关
        let ht = HyperthreadTopology::detect();
        if !cache.l1_data_caches.is_empty() {
            // 通常每个物理核心有L1缓存
            assert!(
                cache.l1_data_caches.len() >= ht.physical_core_count(),
                "Should have at least one L1 cache per physical core"
            );
        }

        // 测试辅助方法
        let line_size = cache.cache_line_size();
        assert!(
            line_size >= 32 && line_size <= 256,
            "Cache line size {} should be between 32 and 256 bytes",
            line_size
        );

        // L3缓存可能不存在(某些系统),但如果存在大小应该合理
        let l3_size = cache.total_l3_size_kb();
        if l3_size > 0 {
            assert!(l3_size >= 256, "L3 cache size should be at least 256 KB");
        }
    }

    #[test]
    fn test_numa_topology_basic() {
        // NUMA拓扑基本验证
        let numa = NumaTopology::detect();

        // 应该至少有一个节点
        assert!(numa.node_count() >= 1, "Should have at least 1 NUMA node");

        // 获取最优节点
        let optimal = numa.get_optimal_node();
        assert!(optimal.is_some(), "Should have an optimal node");

        // 验证节点结构
        for node in &numa.nodes {
            // 节点应该有CPU列表
            assert!(!node.cpus.is_empty(), "NUMA node should have CPUs");

            // 距离矩阵应该非空(如果存在)
            if !node.distance.is_empty() {
                assert!(
                    node.distance.contains(&0),
                    "Distance matrix should contain distance to self (0)"
                );
            }
        }

        // 单节点系统(is_numa=false)也应该正常工作
        if !numa.is_numa {
            assert_eq!(
                numa.node_count(),
                1,
                "Non-NUMA system should have exactly 1 node"
            );
        }
    }

    #[test]
    fn test_hardware_profile_summary() {
        // 硬件配置摘要生成
        let profile = HardwareProfile::detect();
        let summary = profile.summary();

        // 摘要不应为空
        assert!(!summary.is_empty());

        // 应该包含关键信息
        assert!(summary.contains("CPU:"), "Summary should contain 'CPU:'");
        assert!(summary.contains("GPU:"), "Summary should contain 'GPU:'");
        assert!(
            summary.contains("Memory:"),
            "Summary should contain 'Memory:'"
        );
        assert!(
            summary.contains("Cache:"),
            "Summary should contain 'Cache:'"
        );

        // 应该包含数值信息
        assert!(summary.contains("cores"), "Summary should mention cores");
        assert!(
            summary.contains("threads"),
            "Summary should mention threads"
        );
        assert!(
            summary.contains("GB"),
            "Summary should mention GB for memory"
        );
    }

    #[test]
    fn test_gpu_type_variants() {
        // GPU类型枚举测试
        let types = [
            GpuType::Apple,
            GpuType::IntelIntegrated,
            GpuType::Amd,
            GpuType::Nvidia,
            GpuType::Ascend,
            GpuType::Unknown,
        ];

        // 所有变体都应该能创建并比较
        for gpu_type in &types {
            let _clone = *gpu_type;
            assert_eq!(*gpu_type, _clone);
        }

        // PartialEq应该工作
        assert_ne!(GpuType::Nvidia, GpuType::Amd);
        assert_eq!(GpuType::Apple, GpuType::Apple);
    }

    #[test]
    fn test_core_type_variants() {
        // 核心类型枚举测试
        let types = [
            CoreType::Performance,
            CoreType::Efficiency,
            CoreType::Standard,
        ];

        for core_type in &types {
            let _clone = *core_type;
            assert_eq!(*core_type, _clone);
        }

        // 应该能正确比较
        assert_ne!(CoreType::Performance, CoreType::Efficiency);
        assert_eq!(CoreType::Standard, CoreType::Standard);
    }
}
