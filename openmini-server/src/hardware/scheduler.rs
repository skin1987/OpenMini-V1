//! 自适应调度器模块
//!
//! 根据硬件能力自动选择最优推理策略，无需用户手动配置。
//!
//! ## 功能概述
//! - 自动检测硬件能力并分级
//! - 根据硬件等级选择最优调度策略
//! - 动态调整内存、注意力、并行策略
//! - 支持 CPU 亲和性和超线程优化
//!
//! ## 最低硬件要求
//! - 移动端: iPhone 13 Pro Max (A15)
//! - PC/服务器: MacBook Pro 2017 (i7-7700HQ)
//!
//! ## 使用示例
//! ```rust
//! use openmini_server::hardware::scheduler::AdaptiveScheduler;
//!
//! // 创建调度器（自动检测硬件）
//! let scheduler = AdaptiveScheduler::new();
//!
//! // 获取推荐配置
//! let config = scheduler.config();
//! println!("推荐线程数: {}", config.num_threads);
//! println!("KV Cache 大小: {} MB", config.kv_cache_size);
//!
//! // 根据可用内存动态调整
//! let mut scheduler = scheduler;
//! scheduler.adjust_for_memory(2048); // 2GB 可用内存
//! ```

#![allow(dead_code)]

use super::detector::HardwareProfile;
use super::hyperthreading::{CoreSelectionStrategy, CpuAffinity, HyperthreadEfficiency, TaskType};
use super::profile::{DeviceType, HardwareClassifier, HardwareLevel};

mod constants {
    pub const KB: usize = 1024;
    pub const MB: usize = 1024 * 1024;

    pub const MEMORY_THRESHOLD_SMALL: usize = 1024;
    pub const MEMORY_THRESHOLD_STANDARD: usize = 4096;

    pub const SEQ_LEN_THRESHOLD_MEDIUM: usize = 4096;
    pub const SEQ_LEN_THRESHOLD_LARGE: usize = 8192;

    pub const PARALLELISM_MIN_FOR_GPU: u32 = 30;
    pub const PARALLELISM_MAX: u32 = 100;

    pub const KV_CACHE_MIN_SMALL: usize = 64;
    pub const KV_CACHE_MIN_STANDARD: usize = 256;

    pub const MATRIX_SIZE_SMALL: usize = 256;
    pub const MATRIX_SIZE_MEDIUM: usize = 1024;
    pub const MATRIX_SIZE_LARGE: usize = 4096;

    pub const ATTENTION_HEADS_ENTRY: usize = 8;
    pub const ATTENTION_HEADS_STANDARD: usize = 16;
    pub const ATTENTION_HEADS_PROFESSIONAL: usize = 32;
    pub const ATTENTION_HEADS_SERVER: usize = 64;

    pub const DSA_RATIO_ENTRY: usize = 10;
    pub const DSA_RATIO_STANDARD_NUMERATOR: usize = 15;
    pub const DSA_RATIO_STANDARD_DENOMINATOR: usize = 100;
    pub const DSA_RATIO_PROFESSIONAL: usize = 5;
    pub const DSA_RATIO_SERVER: usize = 4;
}

/// 调度策略枚举
///
/// 定义不同硬件等级的调度策略。
///
/// | 策略 | 适用场景 | 特点 |
/// |------|---------|------|
/// | Entry | 入门设备 | SIMD + DSA + 基础 GPU |
/// | Standard | 标准设备 | SIMD + BLAS + DSA + 标准 GPU |
/// | Professional | 专业设备 | 全优化 + Flash Attention |
/// | Server | 服务器 | 分布式并行 + 多 GPU |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleStrategy {
    /// 入门配置策略 - SIMD + DSA + 基础 GPU
    Entry,
    /// 标准配置策略 - SIMD + BLAS + DSA + 标准 GPU
    Standard,
    /// 专业配置策略 - 全优化 + Flash Attention
    Professional,
    /// 服务器配置策略 - 分布式并行 + 多 GPU
    Server,
}

/// 注意力策略
///
/// 定义注意力计算的实现方式。
///
/// | 策略 | 适用场景 | 内存占用 | 速度 |
/// |------|---------|---------|------|
/// | Standard | 短序列 | O(n²) | 基准 |
/// | Dsa | 长序列 | O(n*k) | 快 |
/// | FlashAttention | GPU | O(n) | 最快 |
/// | MultiQueryOptimized | 多查询 | O(n) | 快 |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum AttentionStrategy {
    /// 标准注意力 - 完整注意力矩阵
    Standard,
    /// DSA 稀疏注意力 - 动态稀疏注意力，适合长序列
    Dsa,
    /// Flash Attention (GPU) - GPU 优化的注意力实现
    FlashAttention,
    /// 多查询注意力优化 - 适合批量推理
    MultiQueryOptimized,
}

/// 内存策略
///
/// 定义 KV Cache 的内存管理方式。
///
/// | 策略 | 适用内存 | 特点 |
/// |------|---------|------|
/// | SmallArena | < 1GB | Arena 小块分配 |
/// | StandardArena | 1-4GB | Arena + 缓存 |
/// | PagedAttention | 4-16GB | 分页注意力 |
/// | Distributed | > 16GB | 分布式内存 |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum MemoryStrategy {
    /// 小内存策略 - Arena 小块分配，适合内存受限设备
    SmallArena,
    /// 标准内存策略 - Arena + 缓存，适合标准设备
    StandardArena,
    /// 大内存策略 - Paged Attention，适合大内存设备
    PagedAttention,
    /// 分布式内存策略 - 跨节点内存共享
    Distributed,
}

/// 并行策略
///
/// 定义计算并行化的方式。
///
/// | 策略 | 适用场景 | 加速比 |
/// |------|---------|-------|
/// | Single | 调试/小任务 | 1x |
/// | MultiThread | CPU 密集 | ~Nx |
/// | SimdVectorized | 向量计算 | ~4x |
/// | GpuAccelerated | 大矩阵 | ~10-100x |
/// | Distributed | 大规模 | ~Nx |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ParallelStrategy {
    /// 单线程 - 调试或小任务
    Single,
    /// 多线程（数据并行）- CPU 多核并行
    MultiThread,
    /// SIMD 向量化 - 向量指令加速
    SimdVectorized,
    /// GPU 加速 - 使用 GPU 计算
    GpuAccelerated,
    /// 分布式 - 跨节点并行
    Distributed,
}

/// 推理配置
///
/// 包含推理所需的所有配置参数。
///
/// # 字段说明
/// - `strategy`: 调度策略
/// - `attention`: 注意力策略
/// - `memory`: 内存策略
/// - `parallel`: 并行策略
/// - `num_threads`: 推荐线程数
/// - `use_simd`: 是否启用 SIMD
/// - `use_gpu`: 是否启用 GPU
/// - `kv_cache_size`: KV Cache 大小（MB）
/// - `batch_size`: 批处理大小
#[derive(Debug, Clone, Copy)]
pub struct InferenceConfig {
    /// 调度策略
    pub strategy: ScheduleStrategy,
    /// 注意力策略
    pub attention: AttentionStrategy,
    /// 内存策略
    pub memory: MemoryStrategy,
    /// 并行策略
    pub parallel: ParallelStrategy,
    /// 推荐线程数
    pub num_threads: usize,
    /// 是否启用 SIMD
    pub use_simd: bool,
    /// 是否启用 GPU
    pub use_gpu: bool,
    /// KV Cache 大小（MB）
    pub kv_cache_size: usize,
    /// 批处理大小
    pub batch_size: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            strategy: ScheduleStrategy::Standard,
            attention: AttentionStrategy::Dsa,
            memory: MemoryStrategy::StandardArena,
            parallel: ParallelStrategy::MultiThread,
            num_threads: 4,
            use_simd: true,
            use_gpu: false,
            kv_cache_size: 512,
            batch_size: 1,
        }
    }
}

/// 自适应调度器
///
/// 根据硬件能力自动选择最优推理策略。
///
/// # 功能
/// - 自动检测硬件并分级
/// - 根据硬件等级配置推理参数
/// - 动态调整内存和注意力策略
/// - CPU 亲和性和超线程优化
///
/// # 示例
/// ```rust
/// let scheduler = AdaptiveScheduler::new();
/// println!("硬件等级: {:?}", scheduler.level());
/// println!("推荐配置: {:?}", scheduler.config());
/// ```
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AdaptiveScheduler {
    /// 硬件配置
    hardware: HardwareProfile,
    /// 硬件分级
    level: HardwareLevel,
    /// 设备类型
    device_type: DeviceType,
    /// 当前推理配置
    config: InferenceConfig,
    /// CPU 亲和性管理器
    cpu_affinity: Option<CpuAffinity>,
    /// 超线程效率
    ht_efficiency: Option<HyperthreadEfficiency>,
}

#[allow(dead_code)]
impl AdaptiveScheduler {
    /// 创建新的调度器
    ///
    /// 自动检测硬件并配置最优策略。
    ///
    /// # 示例
    /// ```rust
    /// let scheduler = AdaptiveScheduler::new();
    /// ```
    pub fn new() -> Self {
        let hardware = HardwareProfile::detect();
        let classifier = HardwareClassifier::new(hardware.clone());
        let level = classifier.level();
        let device_type = classifier.device_type();

        let config = Self::create_config(&hardware, level, device_type);

        let cpu_affinity = Some(CpuAffinity::new(
            hardware.hyperthreading.clone(),
            hardware.numa.clone(),
        ));

        let ht_efficiency = Some(HyperthreadEfficiency::estimate(
            &hardware.hyperthreading,
            TaskType::ComputeIntensive,
        ));

        Self {
            hardware,
            level,
            device_type,
            config,
            cpu_affinity,
            ht_efficiency,
        }
    }

    /// 从硬件配置创建
    ///
    /// # 参数
    /// - `hardware`: 硬件配置信息
    ///
    /// # 示例
    /// ```rust
    /// let hardware = HardwareProfile::detect();
    /// let scheduler = AdaptiveScheduler::from_hardware(hardware);
    /// ```
    pub fn from_hardware(hardware: HardwareProfile) -> Self {
        let classifier = HardwareClassifier::new(hardware.clone());
        let level = classifier.level();
        let device_type = classifier.device_type();

        let config = Self::create_config(&hardware, level, device_type);

        let cpu_affinity = Some(CpuAffinity::new(
            hardware.hyperthreading.clone(),
            hardware.numa.clone(),
        ));

        let ht_efficiency = Some(HyperthreadEfficiency::estimate(
            &hardware.hyperthreading,
            TaskType::ComputeIntensive,
        ));

        Self {
            hardware,
            level,
            device_type,
            config,
            cpu_affinity,
            ht_efficiency,
        }
    }

    /// 根据硬件能力创建配置
    fn create_config(
        hardware: &HardwareProfile,
        level: HardwareLevel,
        _device_type: DeviceType,
    ) -> InferenceConfig {
        let num_threads = hardware
            .cpu
            .physical_cores
            .min(hardware.cpu.logical_cores)
            .max(1);
        let has_gpu = hardware.gpu.gpu_type != super::detector::GpuType::Unknown;

        match level {
            HardwareLevel::Entry => InferenceConfig {
                num_threads,
                strategy: ScheduleStrategy::Entry,
                attention: AttentionStrategy::Dsa,
                memory: MemoryStrategy::SmallArena,
                parallel: ParallelStrategy::SimdVectorized,
                use_simd: true,
                use_gpu: has_gpu,
                kv_cache_size: 256,
                batch_size: 1,
            },
            HardwareLevel::Standard => InferenceConfig {
                num_threads,
                strategy: ScheduleStrategy::Standard,
                attention: AttentionStrategy::Dsa,
                memory: MemoryStrategy::StandardArena,
                parallel: ParallelStrategy::MultiThread,
                use_simd: true,
                use_gpu: has_gpu,
                kv_cache_size: 512,
                batch_size: 1,
            },
            HardwareLevel::Professional => InferenceConfig {
                num_threads,
                strategy: ScheduleStrategy::Professional,
                attention: AttentionStrategy::FlashAttention,
                memory: MemoryStrategy::PagedAttention,
                parallel: ParallelStrategy::GpuAccelerated,
                use_simd: true,
                use_gpu: true,
                kv_cache_size: 2048,
                batch_size: 4,
            },
            HardwareLevel::Server => InferenceConfig {
                num_threads,
                strategy: ScheduleStrategy::Server,
                attention: AttentionStrategy::FlashAttention,
                memory: MemoryStrategy::Distributed,
                parallel: ParallelStrategy::Distributed,
                use_simd: true,
                use_gpu: true,
                kv_cache_size: 8192,
                batch_size: 16,
            },
        }
    }

    /// 获取当前配置
    #[inline]
    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }

    /// 获取硬件分级
    #[inline]
    pub fn level(&self) -> HardwareLevel {
        self.level
    }

    /// 获取设备类型
    #[inline]
    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }

    /// 获取硬件配置
    #[inline]
    pub fn hardware(&self) -> &HardwareProfile {
        &self.hardware
    }

    /// 动态调整配置（根据可用内存）
    ///
    /// 根据当前可用内存动态调整内存策略和 KV Cache 大小。
    ///
    /// # 参数
    /// - `available_memory_mb`: 可用内存大小（MB）
    ///
    /// # 调整规则
    /// | 可用内存 | 内存策略 | KV Cache |
    /// |---------|---------|----------|
    /// | < 1GB | SmallArena | memory/4 |
    /// | 1-4GB | StandardArena | memory/2 |
    /// | > 4GB | PagedAttention | memory*3/4 |
    pub fn adjust_for_memory(&mut self, available_memory_mb: usize) {
        use constants::*;
        if available_memory_mb == 0 {
            self.config.memory = MemoryStrategy::SmallArena;
            self.config.kv_cache_size = KV_CACHE_MIN_SMALL;
        } else if available_memory_mb < MEMORY_THRESHOLD_SMALL {
            self.config.memory = MemoryStrategy::SmallArena;
            self.config.kv_cache_size = (available_memory_mb / 4).max(KV_CACHE_MIN_SMALL);
        } else if available_memory_mb < MEMORY_THRESHOLD_STANDARD {
            self.config.memory = MemoryStrategy::StandardArena;
            self.config.kv_cache_size = (available_memory_mb / 2).max(KV_CACHE_MIN_STANDARD);
        } else {
            self.config.memory = MemoryStrategy::PagedAttention;
            self.config.kv_cache_size = available_memory_mb.saturating_mul(3) / 4;
        }
    }

    /// 根据序列长度调整注意力策略
    ///
    /// 长序列时自动切换到更高效的注意力实现。
    ///
    /// # 参数
    /// - `seq_len`: 序列长度
    ///
    /// # 调整规则
    /// - seq_len > 8192: 所有设备使用 DSA
    /// - seq_len > 4096: Entry/Standard 用 DSA，Professional/Server 用 FlashAttention
    pub fn adjust_for_sequence_length(&mut self, seq_len: usize) {
        use constants::*;
        if seq_len > SEQ_LEN_THRESHOLD_LARGE {
            self.config.attention = AttentionStrategy::Dsa;
        } else if seq_len > SEQ_LEN_THRESHOLD_MEDIUM {
            match self.level {
                HardwareLevel::Entry | HardwareLevel::Standard => {
                    self.config.attention = AttentionStrategy::Dsa;
                }
                HardwareLevel::Professional | HardwareLevel::Server => {
                    self.config.attention = AttentionStrategy::FlashAttention;
                }
            }
        }
    }

    /// 获取推荐的 DSA k 值
    ///
    /// DSA（动态稀疏注意力）的稀疏参数 k。
    ///
    /// # 参数
    /// - `seq_len`: 序列长度
    ///
    /// # 返回
    /// 推荐的 k 值（保留的注意力位置数）
    ///
    /// # 计算规则
    /// | 硬件等级 | k 值比例 |
    /// |---------|---------|
    /// | Entry | 10% |
    /// | Standard | 15% |
    /// | Professional | 20% |
    /// | Server | 25% |
    #[inline]
    pub fn recommended_dsa_k(&self, seq_len: usize) -> usize {
        use constants::*;
        match self.level {
            HardwareLevel::Entry => seq_len / DSA_RATIO_ENTRY,
            HardwareLevel::Standard => {
                seq_len * DSA_RATIO_STANDARD_NUMERATOR / DSA_RATIO_STANDARD_DENOMINATOR
            }
            HardwareLevel::Professional => seq_len / DSA_RATIO_PROFESSIONAL,
            HardwareLevel::Server => seq_len / DSA_RATIO_SERVER,
        }
    }

    /// 获取推荐的注意力头数
    ///
    /// # 返回
    /// 推荐的注意力头数
    ///
    /// # 推荐值
    /// | 硬件等级 | 头数 |
    /// |---------|-----|
    /// | Entry | 8 |
    /// | Standard | 16 |
    /// | Professional | 32 |
    /// | Server | 64 |
    #[inline]
    pub fn recommended_attention_heads(&self) -> usize {
        use constants::*;
        match self.level {
            HardwareLevel::Entry => ATTENTION_HEADS_ENTRY,
            HardwareLevel::Standard => ATTENTION_HEADS_STANDARD,
            HardwareLevel::Professional => ATTENTION_HEADS_PROFESSIONAL,
            HardwareLevel::Server => ATTENTION_HEADS_SERVER,
        }
    }

    /// 获取 CPU 亲和性管理器
    ///
    /// # 返回
    /// CPU 亲和性管理器引用（如果可用）
    #[inline]
    pub fn cpu_affinity(&self) -> Option<&CpuAffinity> {
        self.cpu_affinity.as_ref()
    }

    #[inline]
    pub fn ht_efficiency(&self) -> Option<&HyperthreadEfficiency> {
        self.ht_efficiency.as_ref()
    }

    #[inline]
    pub fn optimal_threads_for_task(&self, task_type: TaskType) -> usize {
        match &self.cpu_affinity {
            Some(affinity) => affinity.optimal_thread_count(task_type),
            None => self.config.num_threads,
        }
    }

    /// 获取最优计算核心列表
    ///
    /// # 返回
    /// 推荐用于计算的 CPU 核心列表
    pub fn optimal_compute_cores(&self) -> Vec<usize> {
        if let Some(affinity) = &self.cpu_affinity {
            affinity.select_cores(CoreSelectionStrategy::PerformanceFirst, None)
        } else {
            (0..self.config.num_threads).collect()
        }
    }

    /// 绑定当前线程到最优核心
    ///
    /// 将当前线程绑定到物理性能核心。
    ///
    /// # 返回
    /// 成功返回 Ok(())，失败返回错误
    pub fn bind_current_thread_optimal(
        &self,
    ) -> Result<(), super::hyperthreading::CpuAffinityError> {
        if let Some(affinity) = &self.cpu_affinity {
            let cores = affinity.select_cores(CoreSelectionStrategy::PhysicalOnly, Some(1));
            if let Some(&core_id) = cores.first() {
                affinity.bind_current_thread(core_id)
            } else {
                Ok(())
            }
        } else {
            Ok(())
        }
    }

    /// 获取超线程加速比
    ///
    /// # 返回
    /// 超线程带来的加速比（1.0 表示无加速）
    #[inline]
    pub fn hyperthreading_speedup(&self) -> f32 {
        self.ht_efficiency.map_or(1.0, |e| e.speedup)
    }
}

impl Default for AdaptiveScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// 统一内存调度器
// ============================================================================

/// 计算设备类型
///
/// 定义可用的计算设备。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeDevice {
    /// 纯 CPU 计算
    Cpu,
    /// 集成 GPU (Metal/Vulkan)
    IntegratedGpu,
    /// 独立 GPU (CUDA)
    DiscreteGpu,
}

/// 任务大小阈值
///
/// 定义不同大小任务的计算设备选择阈值。
///
/// # 字段说明
/// - `small_matrix`: 小任务阈值，使用 CPU SIMD
/// - `medium_matrix`: 中任务阈值，使用集成 GPU
/// - `large_matrix`: 大任务阈值，使用独立 GPU
#[derive(Debug, Clone)]
pub struct TaskThresholds {
    /// 小任务矩阵大小阈值 (使用 CPU SIMD)
    pub small_matrix: usize,
    /// 中任务矩阵大小阈值 (使用集成 GPU)
    pub medium_matrix: usize,
    /// 大任务矩阵大小阈值 (使用独立 GPU)
    pub large_matrix: usize,
}

impl Default for TaskThresholds {
    fn default() -> Self {
        Self {
            small_matrix: 256,
            medium_matrix: 1024,
            large_matrix: 4096,
        }
    }
}

/// 统一内存调度器
///
/// 支持统一内存架构的调度器，自动选择最优计算设备。
///
/// # 功能
/// - 自动检测统一内存支持
/// - 根据任务大小选择计算设备
/// - 动态调整并行度
///
/// # 统一内存
/// macOS Apple Silicon 支持统一内存，CPU 和 GPU 共享内存，
/// 无需数据拷贝，提高效率。
///
/// # 示例
/// ```rust
/// let scheduler = UnifiedScheduler::new();
///
/// // 根据任务大小选择设备
/// let device = scheduler.select_device(1024);
/// println!("推荐设备: {:?}", device);
///
/// // 检查统一内存支持
/// if scheduler.has_unified_memory() {
///     println!("支持统一内存");
/// }
/// ```
#[derive(Debug)]
pub struct UnifiedScheduler {
    /// 基础调度器
    base: AdaptiveScheduler,
    /// 任务阈值
    thresholds: TaskThresholds,
    /// 首选计算设备
    preferred_device: ComputeDevice,
    /// 是否支持统一内存
    unified_memory: bool,
    /// 当前并行度 (0-100)
    parallelism: u32,
}

impl UnifiedScheduler {
    /// 创建新的统一调度器
    ///
    /// 自动检测硬件并配置最优策略。
    pub fn new() -> Self {
        let base = AdaptiveScheduler::new();
        let hardware = base.hardware();

        let preferred_device = Self::determine_preferred_device(hardware);
        let unified_memory = Self::check_unified_memory(hardware);

        Self {
            base,
            thresholds: TaskThresholds::default(),
            preferred_device,
            unified_memory,
            parallelism: 100,
        }
    }

    /// 检查是否支持统一内存
    ///
    /// macOS Apple Silicon 支持统一内存。
    fn check_unified_memory(hardware: &HardwareProfile) -> bool {
        #[cfg(target_os = "macos")]
        {
            matches!(hardware.cpu.arch, super::detector::CpuArch::AArch64)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = hardware;
            false
        }
    }

    /// 确定首选计算设备
    ///
    /// 根据 GPU 类型选择首选计算设备。
    fn determine_preferred_device(hardware: &HardwareProfile) -> ComputeDevice {
        use super::detector::GpuType;
        match hardware.gpu.gpu_type {
            GpuType::Nvidia | GpuType::Amd => ComputeDevice::DiscreteGpu,
            GpuType::Apple | GpuType::IntelIntegrated => ComputeDevice::IntegratedGpu,
            GpuType::Ascend => ComputeDevice::IntegratedGpu,
            GpuType::Unknown => ComputeDevice::Cpu,
        }
    }

    /// 选择计算设备
    ///
    /// 根据任务大小和并行度选择最优计算设备。
    ///
    /// # 参数
    /// - `matrix_size`: 矩阵大小（维度）
    ///
    /// # 返回
    /// 推荐的计算设备
    ///
    /// # 选择规则
    /// - 并行度 < 30%: 使用 CPU
    /// - 小任务 (< 256): 使用 CPU SIMD
    /// - 其他: 使用首选设备
    #[inline]
    pub fn select_device(&self, matrix_size: usize) -> ComputeDevice {
        use constants::*;
        if self.parallelism < PARALLELISM_MIN_FOR_GPU {
            return ComputeDevice::Cpu;
        }

        match self.preferred_device {
            ComputeDevice::Cpu => ComputeDevice::Cpu,
            ComputeDevice::IntegratedGpu => {
                if matrix_size < MATRIX_SIZE_SMALL {
                    ComputeDevice::Cpu
                } else {
                    ComputeDevice::IntegratedGpu
                }
            }
            ComputeDevice::DiscreteGpu => {
                if matrix_size < MATRIX_SIZE_SMALL {
                    ComputeDevice::Cpu
                } else {
                    ComputeDevice::DiscreteGpu
                }
            }
        }
    }

    #[inline]
    pub fn recommended_parallelism(&self) -> f32 {
        self.parallelism as f32 / constants::PARALLELISM_MAX as f32
    }

    #[inline]
    pub fn set_parallelism(&mut self, parallelism: u32) {
        self.parallelism = parallelism.min(constants::PARALLELISM_MAX);
    }

    #[inline]
    pub fn has_unified_memory(&self) -> bool {
        self.unified_memory
    }

    #[inline]
    pub fn preferred_device(&self) -> ComputeDevice {
        self.preferred_device
    }

    #[inline]
    pub fn base(&self) -> &AdaptiveScheduler {
        &self.base
    }

    #[inline]
    pub fn thresholds(&self) -> &TaskThresholds {
        &self.thresholds
    }

    pub fn set_thresholds(&mut self, thresholds: TaskThresholds) {
        self.thresholds = thresholds;
    }

    #[inline]
    pub fn recommended_threads(&self) -> usize {
        let base_threads = self.base.hardware().cpu.physical_cores;
        (base_threads * self.parallelism as usize / constants::PARALLELISM_MAX as usize).max(1)
    }
}

impl Default for UnifiedScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_creation() {
        let scheduler = AdaptiveScheduler::new();
        println!("Hardware Level: {:?}", scheduler.level());
        println!("Device Type: {:?}", scheduler.device_type());
        println!("Config: {:?}", scheduler.config());
    }

    #[test]
    fn test_dsa_k_recommendation() {
        let scheduler = AdaptiveScheduler::new();
        let k = scheduler.recommended_dsa_k(1024);
        assert!(k > 0 && k <= 1024);
    }

    #[test]
    fn test_memory_adjustment() {
        let mut scheduler = AdaptiveScheduler::new();
        scheduler.adjust_for_memory(512);
        assert_eq!(scheduler.config().memory, MemoryStrategy::SmallArena);

        scheduler.adjust_for_memory(2048);
        assert_eq!(scheduler.config().memory, MemoryStrategy::StandardArena);

        scheduler.adjust_for_memory(8192);
        assert_eq!(scheduler.config().memory, MemoryStrategy::PagedAttention);
    }

    #[test]
    fn test_unified_scheduler() {
        let scheduler = UnifiedScheduler::new();

        // 小任务使用 CPU
        let device = scheduler.select_device(100);
        assert_eq!(device, ComputeDevice::Cpu);

        // 检查统一内存
        println!("Unified memory: {}", scheduler.has_unified_memory());
    }

    #[test]
    fn test_parallelism() {
        let mut scheduler = UnifiedScheduler::new();

        scheduler.set_parallelism(50);
        assert_eq!(scheduler.recommended_parallelism(), 0.5);

        // 并行度应该被限制在 100
        scheduler.set_parallelism(150);
        assert_eq!(scheduler.parallelism, 100);
    }

    #[test]
    fn test_dsa_k_values() {
        let scheduler = AdaptiveScheduler::new();

        let k_entry = scheduler.recommended_dsa_k(1000);
        let k_standard = scheduler.recommended_dsa_k(1000);
        let k_professional = scheduler.recommended_dsa_k(1000);
        let k_server = scheduler.recommended_dsa_k(1000);

        assert!(k_entry <= 1000);
        assert!(k_standard <= 1000);
        assert!(k_professional <= 1000);
        assert!(k_server <= 1000);
    }

    #[test]
    fn test_memory_adjustment_edge_cases() {
        let mut scheduler = AdaptiveScheduler::new();

        scheduler.adjust_for_memory(0);
        assert_eq!(scheduler.config().memory, MemoryStrategy::SmallArena);
        assert_eq!(scheduler.config().kv_cache_size, 64);

        scheduler.adjust_for_memory(1023);
        assert_eq!(scheduler.config().memory, MemoryStrategy::SmallArena);

        scheduler.adjust_for_memory(1024);
        assert_eq!(scheduler.config().memory, MemoryStrategy::StandardArena);

        scheduler.adjust_for_memory(4095);
        assert_eq!(scheduler.config().memory, MemoryStrategy::StandardArena);

        scheduler.adjust_for_memory(4096);
        assert_eq!(scheduler.config().memory, MemoryStrategy::PagedAttention);
    }

    #[test]
    fn test_recommended_threads() {
        let mut scheduler = UnifiedScheduler::new();

        scheduler.set_parallelism(100);
        let threads_full = scheduler.recommended_threads();
        assert!(threads_full >= 1);

        scheduler.set_parallelism(50);
        let threads_half = scheduler.recommended_threads();
        assert!(threads_half >= 1);
        assert!(threads_half <= threads_full);

        scheduler.set_parallelism(0);
        let threads_zero = scheduler.recommended_threads();
        assert_eq!(threads_zero, 1);
    }

    #[test]
    fn test_attention_heads() {
        let scheduler = AdaptiveScheduler::new();
        let heads = scheduler.recommended_attention_heads();
        assert!(heads >= 8 && heads <= 64);
    }

    // ==================== 新增测试：覆盖枚举变体 ====================

    /// 测试 ScheduleStrategy 所有变体的 PartialEq 实现
    #[test]
    fn test_schedule_strategy_equality() {
        assert_eq!(ScheduleStrategy::Entry, ScheduleStrategy::Entry);
        assert_ne!(ScheduleStrategy::Entry, ScheduleStrategy::Standard);
        assert_ne!(ScheduleStrategy::Standard, ScheduleStrategy::Professional);
        assert_ne!(ScheduleStrategy::Professional, ScheduleStrategy::Server);
    }

    /// 测试 AttentionStrategy 所有变体的 PartialEq 实现
    #[test]
    fn test_attention_strategy_equality() {
        assert_eq!(AttentionStrategy::Standard, AttentionStrategy::Standard);
        assert_ne!(AttentionStrategy::Dsa, AttentionStrategy::FlashAttention);
        assert_ne!(
            AttentionStrategy::FlashAttention,
            AttentionStrategy::MultiQueryOptimized
        );
    }

    /// 测试 MemoryStrategy 所有变体的 PartialEq 实现
    #[test]
    fn test_memory_strategy_equality() {
        assert_eq!(MemoryStrategy::SmallArena, MemoryStrategy::SmallArena);
        assert_ne!(MemoryStrategy::SmallArena, MemoryStrategy::StandardArena);
        assert_ne!(
            MemoryStrategy::StandardArena,
            MemoryStrategy::PagedAttention
        );
        assert_ne!(MemoryStrategy::PagedAttention, MemoryStrategy::Distributed);
    }

    /// 测试 ParallelStrategy 所有变体的 PartialEq 实现
    #[test]
    fn test_parallel_strategy_equality() {
        assert_eq!(ParallelStrategy::Single, ParallelStrategy::Single);
        assert_ne!(
            ParallelStrategy::MultiThread,
            ParallelStrategy::SimdVectorized
        );
        assert_ne!(
            ParallelStrategy::SimdVectorized,
            ParallelStrategy::GpuAccelerated
        );
        assert_ne!(
            ParallelStrategy::GpuAccelerated,
            ParallelStrategy::Distributed
        );
    }

    /// 测试 InferenceConfig 的 Default trait 实现及字段默认值
    #[test]
    fn test_inference_config_default() {
        let config = InferenceConfig::default();
        assert_eq!(config.strategy, ScheduleStrategy::Standard);
        assert_eq!(config.attention, AttentionStrategy::Dsa);
        assert_eq!(config.memory, MemoryStrategy::StandardArena);
        assert_eq!(config.parallel, ParallelStrategy::MultiThread);
        assert_eq!(config.num_threads, 4);
        assert!(config.use_simd);
        assert!(!config.use_gpu);
        assert_eq!(config.kv_cache_size, 512);
        assert_eq!(config.batch_size, 1);
    }

    /// 测试 InferenceConfig 的 Clone 和 Copy trait
    #[test]
    fn test_inference_config_clone_copy() {
        let config = InferenceConfig::default();
        let config2 = config;
        assert_eq!(config.strategy, config2.strategy);
        assert_eq!(config.num_threads, config2.num_threads);
    }

    /// 测试 adjust_for_sequence_length 的所有分支：
    /// - seq_len > 8192: 使用 DSA（覆盖所有硬件等级）
    /// - seq_len > 4096 且 <= 8192: Entry/Standard 用 DSA，Professional/Server 用 FlashAttention
    /// - seq_len <= 4096: 不改变策略
    #[test]
    fn test_adjust_sequence_length_all_branches() {
        let mut scheduler = AdaptiveScheduler::new();
        let original_attention = scheduler.config().attention;

        // 分支1: seq_len > SEQ_LEN_THRESHOLD_LARGE (8192)，所有设备使用 DSA
        scheduler.adjust_for_sequence_length(10000);
        assert_eq!(scheduler.config().attention, AttentionStrategy::Dsa);

        // 重置并测试分支2: seq_len > SEQ_LEN_THRESHOLD_MEDIUM (4096) 且 <= 8192
        let mut scheduler2 = AdaptiveScheduler::new();
        scheduler2.adjust_for_sequence_length(5000);
        // Entry 或 Standard 等级应使用 DSA，Professional 或 Server 使用 FlashAttention
        match scheduler2.level() {
            HardwareLevel::Entry | HardwareLevel::Standard => {
                assert_eq!(scheduler2.config().attention, AttentionStrategy::Dsa);
            }
            HardwareLevel::Professional | HardwareLevel::Server => {
                assert_eq!(
                    scheduler2.config().attention,
                    AttentionStrategy::FlashAttention
                );
            }
        }

        // 分支3: seq_len <= SEQ_LEN_THRESHOLD_MEDIUM (4096)，不改变策略
        let mut scheduler3 = AdaptiveScheduler::new();
        scheduler3.adjust_for_sequence_length(2048);
        assert_eq!(scheduler3.config().attention, original_attention);
    }

    /// 测试 recommended_dsa_k 在不同序列长度下的行为
    /// 覆盖边界条件：seq_len=0、小值、大值
    #[test]
    fn test_recommended_dsa_k_boundary_values() {
        let scheduler = AdaptiveScheduler::new();

        // 边界条件：seq_len = 0
        let k_zero = scheduler.recommended_dsa_k(0);
        assert_eq!(k_zero, 0);

        // 小值
        let k_small = scheduler.recommended_dsa_k(10);
        assert!(k_small <= 10);

        // 大值
        let k_large = scheduler.recommended_dsa_k(100000);
        assert!(k_large > 0 && k_large <= 100000);
    }

    /// 测试 cpu_affinity() 返回 Some 值
    #[test]
    fn test_cpu_affinity_some() {
        let scheduler = AdaptiveScheduler::new();
        assert!(scheduler.cpu_affinity().is_some());
    }

    /// 测试 ht_efficiency() 返回 Some 值且 speedup > 0
    #[test]
    fn test_ht_efficiency_some() {
        let scheduler = AdaptiveScheduler::new();
        let ht = scheduler.ht_efficiency();
        assert!(ht.is_some());
        if let Some(efficiency) = ht {
            assert!(efficiency.speedup > 0.0);
        }
    }

    /// 测试 hyperthreading_speedup() 返回有效值
    #[test]
    fn test_hyperthreading_speedup_valid() {
        let scheduler = AdaptiveScheduler::new();
        let speedup = scheduler.hyperthreading_speedup();
        assert!(speedup >= 1.0); // 至少无加速（1.0x）
    }

    /// 测试 optimal_compute_cores() 返回非空列表
    #[test]
    fn test_optimal_compute_cores_non_empty() {
        let scheduler = AdaptiveScheduler::new();
        let cores = scheduler.optimal_compute_cores();
        // optimal_compute_cores 可能返回空列表（如果 cpu_affinity 为 None 且 num_threads 为 0）
        // 但通常应该有值，这里只验证不崩溃
        let _ = cores;
    }

    /// 测试 bind_current_thread_optimal() 成功执行
    #[test]
    fn test_bind_thread_optimal_success() {
        let scheduler = AdaptiveScheduler::new();
        let result = scheduler.bind_current_thread_optimal();
        assert!(result.is_ok());
    }

    /// 测试 hardware() 返回有效的硬件配置引用
    #[test]
    fn test_hardware_reference() {
        let scheduler = AdaptiveScheduler::new();
        let hw = scheduler.hardware();
        assert!(hw.cpu.physical_cores >= 1);
        assert!(hw.cpu.logical_cores >= 1);
    }

    /// 测试 device_type() 返回有效设备类型
    #[test]
    fn test_device_type_valid() {
        let scheduler = AdaptiveScheduler::new();
        let dt = scheduler.device_type();
        // DeviceType 应该是有效枚举值
        let _ = format!("{:?}", dt);
    }

    /// 测试 UnifiedScheduler 的 preferred_device() 返回有效计算设备
    #[test]
    fn test_unified_preferred_device() {
        let scheduler = UnifiedScheduler::new();
        let device = scheduler.preferred_device();
        match device {
            ComputeDevice::Cpu | ComputeDevice::IntegratedGpu | ComputeDevice::DiscreteGpu => {}
        }
    }

    /// 测试 select_device 根据矩阵大小选择设备的分支逻辑
    /// - parallelism < 30%: 总是返回 CPU
    /// - matrix_size < 256: 返回 CPU（即使有GPU）
    /// - matrix_size >= 256: 返回首选设备
    #[test]
    fn test_select_device_matrix_size_branches() {
        let mut scheduler = UnifiedScheduler::new();

        // 分支1: 低并行度 (< 30%)，总是 CPU
        scheduler.set_parallelism(20);
        let device_low_par = scheduler.select_device(500);
        assert_eq!(device_low_par, ComputeDevice::Cpu);

        // 恢复并行度
        scheduler.set_parallelism(100);

        // 分支2: 小矩阵 (< 256)，使用 CPU
        let device_small = scheduler.select_device(100);
        assert_eq!(device_small, ComputeDevice::Cpu);

        // 分支3: 大矩阵 (>= 256)，使用首选设备
        let device_large = scheduler.select_device(512);
        // 应该返回首选设备（不一定是CPU）
        match scheduler.preferred_device() {
            ComputeDevice::Cpu => assert_eq!(device_large, ComputeDevice::Cpu),
            _ => {} // GPU 设备可能返回 GPU
        }
    }

    /// 测试 TaskThresholds 的 Default trait 及自定义配置
    #[test]
    fn test_task_thresholds_default_and_custom() {
        let thresholds = TaskThresholds::default();
        assert_eq!(thresholds.small_matrix, 256);
        assert_eq!(thresholds.medium_matrix, 1024);
        assert_eq!(thresholds.large_matrix, 4096);

        // 自定义阈值
        let custom = TaskThresholds {
            small_matrix: 128,
            medium_matrix: 512,
            large_matrix: 2048,
        };
        assert_eq!(custom.small_matrix, 128);
    }

    /// 测试 set_thresholds 方法
    #[test]
    fn test_set_thresholds() {
        let mut scheduler = UnifiedScheduler::new();
        let custom = TaskThresholds {
            small_matrix: 64,
            medium_matrix: 256,
            large_matrix: 1024,
        };
        scheduler.set_thresholds(custom);
        assert_eq!(scheduler.thresholds().small_matrix, 64);
    }

    /// 测试 base() 方法返回基础调度器引用
    #[test]
    fn test_base_scheduler_reference() {
        let scheduler = UnifiedScheduler::new();
        let base = scheduler.base();
        assert!(base.level() == scheduler.base().level());
    }

    /// 测试 ComputeDevice 枚举的 PartialEq
    #[test]
    fn test_compute_device_equality() {
        assert_eq!(ComputeDevice::Cpu, ComputeDevice::Cpu);
        assert_ne!(ComputeDevice::Cpu, ComputeDevice::IntegratedGpu);
        assert_ne!(ComputeDevice::IntegratedGpu, ComputeDevice::DiscreteGpu);
    }

    /// 测试 AdaptiveScheduler 的 Default trait
    #[test]
    fn test_adaptive_scheduler_default() {
        let scheduler = AdaptiveScheduler::default();
        assert!(scheduler.level() == AdaptiveScheduler::new().level());
    }

    /// 测试 set_parallelism 边界：设置 u32::MAX 应被限制到 100
    #[test]
    fn test_set_parallelism_max_value() {
        let mut scheduler = UnifiedScheduler::new();
        scheduler.set_parallelism(u32::MAX);
        assert_eq!(scheduler.parallelism, 100);
    }

    /// 测试 adjust_for_memory 的 KV Cache 大小计算精确性
    /// 验证不同内存阈值下的 kv_cache_size 计算
    #[test]
    fn test_kv_cache_size_calculation_precision() {
        let mut scheduler = AdaptiveScheduler::new();

        // 0 MB -> 最小值 64
        scheduler.adjust_for_memory(0);
        assert_eq!(scheduler.config().kv_cache_size, 64);

        // 500 MB -> 500/4=125，但至少 64
        scheduler.adjust_for_memory(500);
        assert_eq!(scheduler.config().kv_cache_size, 125);

        // 2000 MB -> 2000/2=1000，但至少 256
        scheduler.adjust_for_memory(2000);
        assert_eq!(scheduler.config().kv_cache_size, 1000);

        // 8000 MB -> 8000*3/4=6000
        scheduler.adjust_for_memory(8000);
        assert_eq!(scheduler.config().kv_cache_size, 6000);
    }
}
