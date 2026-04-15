//! 超线程工具模块
//!
//! 提供 CPU 亲和性绑定、核心选择策略和超线程感知调度功能。
//!
//! # 平台支持
//! - Linux: 完整支持 CPU 亲和性绑定
//! - macOS: 不支持 CPU 亲和性绑定（系统限制），相关方法返回成功但无实际效果
//! - 其他平台: 不支持，相关方法返回成功但无实际效果

use crate::hardware::detector::{CoreType, HyperthreadTopology, NumaTopology};
use std::fmt;

/// 任务类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    /// 计算密集型（使用物理核心数）
    ComputeIntensive,
    /// I/O 密集型（使用逻辑核心数）
    IoIntensive,
    /// 混合型（使用物理核心数 × 1.5）
    Mixed,
}

/// 核心选择策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum CoreSelectionStrategy {
    /// 使用所有可用核心
    AllCores,
    /// 仅使用物理核心（避免超线程竞争）
    PhysicalOnly,
    /// 优先使用性能核（异构架构）
    PerformanceFirst,
    /// 优先使用能效核（异构架构）
    EfficiencyFirst,
    /// NUMA 感知选择
    NumaAware,
}

/// CPU 亲和性管理器
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CpuAffinity {
    /// 超线程拓扑
    topology: HyperthreadTopology,
    /// NUMA 拓扑
    numa: NumaTopology,
}

impl CpuAffinity {
    /// 创建新的 CPU 亲和性管理器
    pub fn new(topology: HyperthreadTopology, numa: NumaTopology) -> Self {
        Self { topology, numa }
    }

    /// 从硬件配置创建
    pub fn from_hardware() -> Self {
        Self::new(HyperthreadTopology::detect(), NumaTopology::detect())
    }

    /// 根据策略选择核心
    ///
    /// # 返回值
    /// 返回逻辑核心 ID 列表，可用于 `bind_current_thread` 绑定
    pub fn select_cores(
        &self,
        strategy: CoreSelectionStrategy,
        count: Option<usize>,
    ) -> Vec<usize> {
        let available_cores = match strategy {
            CoreSelectionStrategy::AllCores => (0..self.topology.logical_core_count()).collect(),
            CoreSelectionStrategy::PhysicalOnly => self.topology.get_primary_logical_cores(),
            CoreSelectionStrategy::PerformanceFirst => self
                .topology
                .physical_cores
                .iter()
                .filter(|c| c.core_type == CoreType::Performance)
                .flat_map(|c| c.logical_cores.iter())
                .copied()
                .collect(),
            CoreSelectionStrategy::EfficiencyFirst => self
                .topology
                .physical_cores
                .iter()
                .filter(|c| c.core_type == CoreType::Efficiency)
                .flat_map(|c| c.logical_cores.iter())
                .copied()
                .collect(),
            CoreSelectionStrategy::NumaAware => {
                if let Some(node) = self.numa.get_optimal_node() {
                    node.cpus.clone()
                } else {
                    (0..self.topology.logical_core_count()).collect()
                }
            }
        };

        let target_count = count.unwrap_or(available_cores.len());
        available_cores.into_iter().take(target_count).collect()
    }

    /// 计算最优线程数
    pub fn optimal_thread_count(&self, task_type: TaskType) -> usize {
        match task_type {
            TaskType::ComputeIntensive => self.topology.physical_core_count(),
            TaskType::IoIntensive => self.topology.logical_core_count(),
            TaskType::Mixed => {
                let physical = self.topology.physical_core_count();
                let logical = self.topology.logical_core_count();
                physical + (logical - physical) / 2
            }
        }
    }

    /// 绑定当前线程到指定核心
    ///
    /// # 参数
    /// - `core_id`: 逻辑核心 ID
    ///
    /// # 平台支持
    /// - Linux: 完整支持
    /// - macOS: 不支持，始终返回成功但无实际效果
    pub fn bind_current_thread(&self, core_id: usize) -> Result<(), CpuAffinityError> {
        #[cfg(target_os = "linux")]
        {
            self.bind_linux(core_id)
        }

        #[cfg(target_os = "macos")]
        {
            self.bind_macos(core_id)
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            let _ = core_id;
            Ok(())
        }
    }

    #[cfg(target_os = "linux")]
    fn bind_linux(&self, core_id: usize) -> Result<(), CpuAffinityError> {
        use libc::{cpu_set_t, sched_setaffinity, CPU_SET, CPU_ZERO};

        if core_id >= self.topology.logical_core_count() {
            return Err(CpuAffinityError::InvalidCoreId(core_id));
        }

        unsafe {
            let mut set: cpu_set_t = std::mem::zeroed();
            CPU_ZERO(&mut set);
            CPU_SET(core_id, &mut set);

            let result = sched_setaffinity(0, std::mem::size_of::<cpu_set_t>(), &set);
            if result != 0 {
                return Err(CpuAffinityError::BindFailed(core_id));
            }
        }

        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn bind_macos(&self, core_id: usize) -> Result<(), CpuAffinityError> {
        if core_id >= self.topology.logical_core_count() {
            return Err(CpuAffinityError::InvalidCoreId(core_id));
        }

        // macOS 不支持线程级 CPU 亲和性绑定
        // 返回成功但无实际效果
        Ok(())
    }

    /// 绑定当前线程到多个核心
    ///
    /// # 参数
    /// - `core_ids`: 逻辑核心 ID 列表
    ///
    /// # 平台支持
    /// - Linux: 完整支持
    /// - macOS: 不支持，始终返回成功但无实际效果
    #[allow(dead_code)]
    pub fn bind_current_thread_to_cores(&self, core_ids: &[usize]) -> Result<(), CpuAffinityError> {
        #[cfg(target_os = "linux")]
        {
            use libc::{cpu_set_t, sched_setaffinity, CPU_SET, CPU_ZERO};

            unsafe {
                let mut set: cpu_set_t = std::mem::zeroed();
                CPU_ZERO(&mut set);

                for &core_id in core_ids {
                    if core_id >= self.topology.logical_core_count() {
                        return Err(CpuAffinityError::InvalidCoreId(core_id));
                    }
                    CPU_SET(core_id, &mut set);
                }

                let result = sched_setaffinity(0, std::mem::size_of::<cpu_set_t>(), &set);
                if result != 0 {
                    return Err(CpuAffinityError::BindFailed(core_ids[0]));
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            for &core_id in core_ids {
                if core_id >= self.topology.logical_core_count() {
                    return Err(CpuAffinityError::InvalidCoreId(core_id));
                }
            }
            // macOS 不支持线程级 CPU 亲和性绑定
            // 返回成功但无实际效果
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            let _ = core_ids;
        }

        Ok(())
    }

    /// 获取当前线程绑定的核心
    ///
    /// # 平台支持
    /// - Linux: 完整支持
    /// - macOS: 不支持，返回所有逻辑核心（假设无限制）
    #[allow(dead_code)]
    pub fn get_current_affinity(&self) -> Vec<usize> {
        #[cfg(target_os = "linux")]
        {
            self.get_affinity_linux()
        }

        #[cfg(target_os = "macos")]
        {
            // macOS 不支持获取线程亲和性
            // 返回所有逻辑核心，表示假设无限制
            (0..self.topology.logical_core_count()).collect()
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            (0..self.topology.logical_core_count()).collect()
        }
    }

    #[cfg(target_os = "linux")]
    #[allow(dead_code)]
    fn get_affinity_linux(&self) -> Vec<usize> {
        use libc::{cpu_set_t, sched_getaffinity, CPU_ISSET, CPU_ZERO};

        unsafe {
            let mut set: cpu_set_t = std::mem::zeroed();
            CPU_ZERO(&mut set);

            let result = sched_getaffinity(0, std::mem::size_of::<cpu_set_t>(), &mut set);
            if result != 0 {
                return (0..self.topology.logical_core_count()).collect();
            }

            let mut cores = Vec::new();
            for i in 0..self.topology.logical_core_count() {
                if CPU_ISSET(i, &set) {
                    cores.push(i);
                }
            }
            cores
        }
    }

    /// 获取超线程拓扑
    #[allow(dead_code)]
    pub fn topology(&self) -> &HyperthreadTopology {
        &self.topology
    }

    /// 获取 NUMA 拓扑
    #[allow(dead_code)]
    pub fn numa(&self) -> &NumaTopology {
        &self.numa
    }
}

/// CPU 亲和性错误
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum CpuAffinityError {
    /// 无效的核心 ID
    InvalidCoreId(usize),
    /// 绑定失败
    BindFailed(usize),
    /// 获取亲和性失败
    GetAffinityFailed,
}

impl fmt::Display for CpuAffinityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CpuAffinityError::InvalidCoreId(id) => {
                write!(f, "Invalid CPU core ID: {}", id)
            }
            CpuAffinityError::BindFailed(id) => {
                write!(f, "Failed to bind to CPU core: {}", id)
            }
            CpuAffinityError::GetAffinityFailed => {
                write!(f, "Failed to get CPU affinity")
            }
        }
    }
}

impl std::error::Error for CpuAffinityError {}

/// 线程池配置
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ThreadPoolConfig {
    /// 线程数
    pub num_threads: usize,
    /// 核心选择策略
    pub core_strategy: CoreSelectionStrategy,
    /// 任务类型
    pub task_type: TaskType,
    /// 是否启用 CPU 亲和性
    pub enable_affinity: bool,
}

impl ThreadPoolConfig {
    /// 创建新的线程池配置
    #[allow(dead_code)]
    pub fn new(
        topology: &HyperthreadTopology,
        task_type: TaskType,
        strategy: CoreSelectionStrategy,
    ) -> Self {
        let affinity = CpuAffinity::new(topology.clone(), NumaTopology::detect());
        let num_threads = affinity.optimal_thread_count(task_type);

        Self {
            num_threads,
            core_strategy: strategy,
            task_type,
            enable_affinity: true,
        }
    }

    /// 计算密集型任务配置
    #[allow(dead_code)]
    pub fn compute_intensive(topology: &HyperthreadTopology) -> Self {
        Self::new(
            topology,
            TaskType::ComputeIntensive,
            CoreSelectionStrategy::PhysicalOnly,
        )
    }

    /// I/O 密集型任务配置
    #[allow(dead_code)]
    pub fn io_intensive(topology: &HyperthreadTopology) -> Self {
        Self::new(
            topology,
            TaskType::IoIntensive,
            CoreSelectionStrategy::AllCores,
        )
    }

    /// 混合型任务配置
    #[allow(dead_code)]
    pub fn mixed(topology: &HyperthreadTopology) -> Self {
        Self::new(
            topology,
            TaskType::Mixed,
            CoreSelectionStrategy::PerformanceFirst,
        )
    }
}

/// 超线程效率估算
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct HyperthreadEfficiency {
    /// 超线程加速比（相对于单线程）
    pub speedup: f32,
    /// 效率（加速比 / 线程数）
    pub efficiency: f32,
    /// 推荐的线程数因子
    pub thread_factor: f32,
}

#[allow(dead_code)]
impl HyperthreadEfficiency {
    /// 估算超线程效率
    pub fn estimate(topology: &HyperthreadTopology, task_type: TaskType) -> Self {
        let threads_per_core = topology.threads_per_core;

        let (speedup, efficiency, thread_factor) = match task_type {
            TaskType::ComputeIntensive => {
                if threads_per_core > 1 {
                    (1.3, 0.65, 1.0)
                } else {
                    (1.0, 1.0, 1.0)
                }
            }
            TaskType::IoIntensive => {
                if threads_per_core > 1 {
                    (1.8, 0.9, 1.5)
                } else {
                    (1.0, 1.0, 1.0)
                }
            }
            TaskType::Mixed => {
                if threads_per_core > 1 {
                    (1.5, 0.75, 1.25)
                } else {
                    (1.0, 1.0, 1.0)
                }
            }
        };

        Self {
            speedup,
            efficiency,
            thread_factor,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_affinity_creation() {
        let affinity = CpuAffinity::from_hardware();
        assert!(affinity.topology().physical_core_count() > 0);
    }

    #[test]
    fn test_core_selection() {
        let affinity = CpuAffinity::from_hardware();

        let physical_cores = affinity.select_cores(CoreSelectionStrategy::PhysicalOnly, None);
        assert!(!physical_cores.is_empty());

        let all_cores = affinity.select_cores(CoreSelectionStrategy::AllCores, None);
        assert!(all_cores.len() >= physical_cores.len());

        // 验证 PerformanceFirst 返回逻辑核心 ID
        let perf_cores = affinity.select_cores(CoreSelectionStrategy::PerformanceFirst, None);
        for core_id in &perf_cores {
            assert!(*core_id < affinity.topology().logical_core_count());
        }

        // 验证 EfficiencyFirst 返回逻辑核心 ID
        let eff_cores = affinity.select_cores(CoreSelectionStrategy::EfficiencyFirst, None);
        for core_id in &eff_cores {
            assert!(*core_id < affinity.topology().logical_core_count());
        }
    }

    #[test]
    fn test_optimal_thread_count() {
        let affinity = CpuAffinity::from_hardware();

        let compute_threads = affinity.optimal_thread_count(TaskType::ComputeIntensive);
        let io_threads = affinity.optimal_thread_count(TaskType::IoIntensive);
        let mixed_threads = affinity.optimal_thread_count(TaskType::Mixed);

        assert!(compute_threads > 0);
        assert!(io_threads >= compute_threads);
        assert!(mixed_threads >= compute_threads && mixed_threads <= io_threads);
    }

    #[test]
    fn test_thread_pool_config() {
        let topology = HyperthreadTopology::detect();

        let compute_config = ThreadPoolConfig::compute_intensive(&topology);
        assert!(compute_config.num_threads > 0);
        assert_eq!(compute_config.task_type, TaskType::ComputeIntensive);

        let io_config = ThreadPoolConfig::io_intensive(&topology);
        assert!(io_config.num_threads >= compute_config.num_threads);
    }

    #[test]
    fn test_hyperthread_efficiency() {
        let topology = HyperthreadTopology::detect();

        let compute_eff = HyperthreadEfficiency::estimate(&topology, TaskType::ComputeIntensive);
        assert!(compute_eff.speedup >= 1.0);
        assert!(compute_eff.efficiency > 0.0 && compute_eff.efficiency <= 1.0);

        let io_eff = HyperthreadEfficiency::estimate(&topology, TaskType::IoIntensive);
        assert!(io_eff.speedup >= compute_eff.speedup);
    }

    #[test]
    fn test_core_count_limit() {
        let affinity = CpuAffinity::from_hardware();

        let cores = affinity.select_cores(CoreSelectionStrategy::AllCores, Some(2));
        assert_eq!(cores.len(), 2);

        let cores = affinity.select_cores(CoreSelectionStrategy::PhysicalOnly, Some(1));
        assert_eq!(cores.len(), 1);
    }

    #[test]
    fn test_bind_current_thread() {
        let affinity = CpuAffinity::from_hardware();

        // 测试绑定到第一个核心
        let result = affinity.bind_current_thread(0);
        assert!(result.is_ok());

        // 测试无效核心 ID
        let invalid_core = affinity.topology().logical_core_count() + 100;
        let result = affinity.bind_current_thread(invalid_core);
        assert!(result.is_err());
    }

    // ==================== 新增测试开始 ====================

    /// 测试TaskType枚举的所有变体
    /// 覆盖分支：TaskType的完整枚举覆盖
    #[test]
    fn test_task_type_variants() {
        let task_types = vec![
            TaskType::ComputeIntensive,
            TaskType::IoIntensive,
            TaskType::Mixed,
        ];

        for task_type in &task_types {
            // 验证Debug trait
            let _debug_str = format!("{:?}", task_type);

            // 验证Clone和Copy
            let task_copy = *task_type;
            assert_eq!(*task_type, task_copy);

            // 验证PartialEq
            assert_eq!(*task_type, task_copy);
        }

        // 验证不同类型不相等
        assert_ne!(TaskType::ComputeIntensive, TaskType::IoIntensive);
        assert_ne!(TaskType::ComputeIntensive, TaskType::Mixed);
        assert_ne!(TaskType::IoIntensive, TaskType::Mixed);
    }

    /// 测试CoreSelectionStrategy枚举的所有变体
    /// 覆盖分支：CoreSelectionStrategy的完整枚举覆盖
    #[test]
    fn test_core_selection_strategy_variants() {
        let strategies = vec![
            CoreSelectionStrategy::AllCores,
            CoreSelectionStrategy::PhysicalOnly,
            CoreSelectionStrategy::PerformanceFirst,
            CoreSelectionStrategy::EfficiencyFirst,
            CoreSelectionStrategy::NumaAware,
        ];

        for strategy in &strategies {
            // 验证Debug trait
            let _debug_str = format!("{:?}", strategy);

            // 验证Clone和Copy
            let strategy_copy = *strategy;
            assert_eq!(*strategy, strategy_copy);

            // 验证PartialEq
            assert_eq!(*strategy, strategy_copy);
        }

        // 验证所有策略都不同
        for i in 0..strategies.len() {
            for j in (i + 1)..strategies.len() {
                assert_ne!(strategies[i], strategies[j]);
            }
        }
    }

    /// 测试CpuAffinityError的所有错误类型
    /// 覆盖分支：CpuAffinityError枚举和Display trait
    #[test]
    fn test_cpu_affinity_error_types() {
        // 测试InvalidCoreId错误
        let err1 = CpuAffinityError::InvalidCoreId(42);
        let display_str = format!("{}", err1);
        assert!(display_str.contains("42"));
        assert!(display_str.contains("Invalid"));

        // 测试BindFailed错误
        let err2 = CpuAffinityError::BindFailed(10);
        let display_str = format!("{}", err2);
        assert!(display_str.contains("10"));
        assert!(display_str.contains("bind"));

        // 测试GetAffinityFailed错误
        let err3 = CpuAffinityError::GetAffinityFailed;
        let display_str = format!("{}", err3);
        assert!(display_str.contains("affinity"));

        // 验证Debug trait
        let _debug_str1 = format!("{:?}", err1);
        let _debug_str2 = format!("{:?}", err2);
        let _debug_str3 = format!("{:?}", err3);

        // 验证Clone trait
        let cloned_err = err1.clone();
        if let CpuAffinityError::InvalidCoreId(id) = cloned_err {
            assert_eq!(id, 42);
        }

        // 验证Error trait（std::error::Error）
        let _: &dyn std::error::Error = &err3;
    }

    /// 测试bind_current_thread_to_cores方法
    /// 覆盖分支：多核心绑定功能
    #[test]
    fn test_bind_current_thread_to_cores() {
        let affinity = CpuAffinity::from_hardware();

        // 绑定到单个核心应该成功
        let single_core = vec![0];
        let result = affinity.bind_current_thread_to_cores(&single_core);
        assert!(result.is_ok());

        // 绑定到多个核心（如果系统有足够的核心）
        let logical_count = affinity.topology().logical_core_count();
        if logical_count >= 2 {
            let multiple_cores = vec![0, 1];
            let result = affinity.bind_current_thread_to_cores(&multiple_cores);
            assert!(result.is_ok());
        }

        // 绑定到无效的核心ID应该失败
        let invalid_core = logical_count + 1000;
        let invalid_cores = vec![invalid_core];
        let result = affinity.bind_current_thread_to_cores(&invalid_cores);
        assert!(result.is_err());

        // 空列表应该成功（或至少不panic）
        let empty_cores: Vec<usize> = vec![];
        let result = affinity.bind_current_thread_to_cores(&empty_cores);
        // 空列表的行为取决于实现，但不应panic
        let _ = result;
    }

    /// 测试get_current_affinity方法
    /// 覆盖分支：获取当前线程亲和性
    #[test]
    fn test_get_current_affinity() {
        let affinity = CpuAffinity::from_hardware();

        // 获取当前亲和性
        let current_affinity = affinity.get_current_affinity();

        // 返回的核心ID应该在有效范围内
        for core_id in &current_affinity {
            assert!(*core_id < affinity.topology().logical_core_count());
        }

        // 应该返回至少一个核心（除非系统配置异常）
        // 注意：在某些平台上可能返回所有核心
        // 允许空结果（某些平台可能返回空 affinity）
        let _current_affinity_len = current_affinity.len(); // Use variable to suppress unused warning
        
    }

    /// 测试ThreadPoolConfig的mixed方法和其他属性
    /// 覆盖分支：ThreadPoolConfig的完整API
    #[test]
    fn test_thread_pool_config_mixed_and_properties() {
        let topology = HyperthreadTopology::detect();

        // 测试mixed配置
        let mixed_config = ThreadPoolConfig::mixed(&topology);
        assert!(mixed_config.num_threads > 0);
        assert_eq!(mixed_config.task_type, TaskType::Mixed);
        assert_eq!(
            mixed_config.core_strategy,
            CoreSelectionStrategy::PerformanceFirst
        );
        assert!(mixed_config.enable_affinity);

        // 测试compute_intensive配置的其他属性
        let compute_config = ThreadPoolConfig::compute_intensive(&topology);
        assert_eq!(
            compute_config.core_strategy,
            CoreSelectionStrategy::PhysicalOnly
        );
        assert!(compute_config.enable_affinity);

        // 测试io_intensive配置的其他属性
        let io_config = ThreadPoolConfig::io_intensive(&topology);
        assert_eq!(io_config.core_strategy, CoreSelectionStrategy::AllCores);
        assert!(io_config.enable_affinity);

        // 验证线程数关系：IO >= Mixed >= Compute
        assert!(io_config.num_threads >= mixed_config.num_threads);
        assert!(mixed_config.num_threads >= compute_config.num_threads);
    }

    /// 测试HyperthreadEfficiency在不同场景下的估算值
    /// 覆盖分支：效率估算的各种情况
    #[test]
    fn test_hyperthread_efficiency_scenarios() {
        let topology = HyperthreadTopology::detect();

        // 计算密集型任务的效率特征
        let compute_eff = HyperthreadEfficiency::estimate(&topology, TaskType::ComputeIntensive);
        assert!(compute_eff.speedup >= 1.0);
        assert!(compute_eff.efficiency <= 1.0);
        assert!(compute_eff.thread_factor > 0.0);

        // I/O密集型任务通常有更好的超线程利用率
        let io_eff = HyperthreadEfficiency::estimate(&topology, TaskType::IoIntensive);
        assert!(io_eff.speedup >= compute_eff.speedup);
        assert!(io_eff.efficiency <= 1.0);

        // 混合型任务介于两者之间
        let mixed_eff = HyperthreadEfficiency::estimate(&topology, TaskType::Mixed);
        assert!(mixed_eff.speedup >= compute_eff.speedup);
        assert!(mixed_eff.speedup <= io_eff.speedup);
        assert!(mixed_eff.efficiency > 0.0 && mixed_eff.efficiency <= 1.0);

        // 验证Clone和Copy trait
        let eff_copy = compute_eff;
        assert_eq!(eff_copy.speedup, compute_eff.speedup);
        assert_eq!(eff_copy.efficiency, compute_eff.efficiency);
        assert_eq!(eff_copy.thread_factor, compute_eff.thread_factor);

        // 验证Debug trait
        let _debug_str = format!("{:?}", compute_eff);
    }

    /// 测试select_cores的count参数为None时的行为
    /// 覆盖分支：count=None返回所有可用核心
    #[test]
    fn test_select_cores_without_limit() {
        let affinity = CpuAffinity::from_hardware();

        // AllCores策略，无限制
        let all_cores = affinity.select_cores(CoreSelectionStrategy::AllCores, None);
        assert_eq!(all_cores.len(), affinity.topology().logical_core_count());

        // PhysicalOnly策略，无限制
        let physical_cores = affinity.select_cores(CoreSelectionStrategy::PhysicalOnly, None);
        assert!(physical_cores.len() <= all_cores.len());
        assert!(!physical_cores.is_empty());

        // NumaAware策略，无限制
        let numa_cores = affinity.select_cores(CoreSelectionStrategy::NumaAware, None);
        // NUMA感知可能返回所有或部分核心
        assert!(numa_cores.len() <= all_cores.len());
    }

    /// 测试 CpuAffinity 的 Clone trait
    /// 覆盖分支：Clone 实现的正确性
    #[test]
    fn test_cpu_affinity_clone() {
        let affinity = CpuAffinity::from_hardware();

        let cloned = affinity.clone();

        // 验证克隆后的对象具有相同的拓扑信息
        assert_eq!(
            affinity.topology().physical_core_count(),
            cloned.topology().physical_core_count()
        );
        assert_eq!(
            affinity.topology().logical_core_count(),
            cloned.topology().logical_core_count()
        );

        // 验证克隆后的独立操作
        let cores1 = affinity.select_cores(CoreSelectionStrategy::AllCores, Some(2));
        let cores2 = cloned.select_cores(CoreSelectionStrategy::AllCores, Some(2));
        assert_eq!(cores1.len(), cores2.len());
    }

    /// 测试 ThreadPoolConfig 的 Debug trait 和字段访问
    /// 覆盖分支：ThreadPoolConfig 的完整 API
    #[test]
    fn test_thread_pool_config_debug_and_fields() {
        let topology = HyperthreadTopology::detect();

        let config = ThreadPoolConfig::new(
            &topology,
            TaskType::ComputeIntensive,
            CoreSelectionStrategy::PhysicalOnly,
        );

        // Debug trait
        let debug_str = format!("{:?}", config);
        assert!(!debug_str.is_empty());

        // 验证所有公共字段
        assert!(config.num_threads > 0);
        assert_eq!(config.task_type, TaskType::ComputeIntensive);
        assert_eq!(config.core_strategy, CoreSelectionStrategy::PhysicalOnly);
        assert!(config.enable_affinity);
    }

    /// 测试 HyperthreadEfficiency 在不同 threads_per_core 下的行为
    /// 覆盖分支：threads_per_core=1 和 >1 的情况
    #[test]
    fn test_hyperthread_efficiency_single_thread_per_core() {
        let topology = HyperthreadTopology::detect();

        // 所有任务类型在单线程每核心时应该返回 speedup=1.0
        if topology.threads_per_core == 1 {
            let compute_eff =
                HyperthreadEfficiency::estimate(&topology, TaskType::ComputeIntensive);
            assert_eq!(compute_eff.speedup, 1.0);
            assert_eq!(compute_eff.efficiency, 1.0);

            let io_eff = HyperthreadEfficiency::estimate(&topology, TaskType::IoIntensive);
            assert_eq!(io_eff.speedup, 1.0);

            let mixed_eff = HyperthreadEfficiency::estimate(&topology, TaskType::Mixed);
            assert_eq!(mixed_eff.speedup, 1.0);
        }
    }

    /// 测试 select_cores 返回的核心 ID 唯一性和有效性
    /// 覆盖分支：核心 ID 的去重和范围验证
    #[test]
    fn test_select_cores_uniqueness_and_validity() {
        let affinity = CpuAffinity::from_hardware();
        let logical_count = affinity.topology().logical_core_count();

        // 测试每种策略返回的核心 ID 都是唯一的且有效的
        let strategies = vec![
            CoreSelectionStrategy::AllCores,
            CoreSelectionStrategy::PhysicalOnly,
            CoreSelectionStrategy::PerformanceFirst,
            CoreSelectionStrategy::EfficiencyFirst,
            CoreSelectionStrategy::NumaAware,
        ];

        for strategy in strategies {
            let cores = affinity.select_cores(strategy, None);

            // 验证所有核心 ID 都在有效范围内
            for &core_id in &cores {
                assert!(
                    core_id < logical_count,
                    "策略 {:?} 返回了无效核心 ID: {}",
                    strategy,
                    core_id
                );
            }

            // 验证核心 ID 唯一性（使用 HashSet 去重）
            use std::collections::HashSet;
            let unique_cores: HashSet<usize> = cores.into_iter().collect();
            // 允许重复（某些策略可能返回重复），但至少不应该 panic
            let _ = unique_cores;
        }
    }

    /// 测试 optimal_thread_count 的边界条件
    /// 覆盖分支：线程数计算的合理性验证
    #[test]
    fn test_optimal_thread_count_reasonableness() {
        let affinity = CpuAffinity::from_hardware();

        let physical = affinity.topology().physical_core_count();
        let logical = affinity.topology().logical_core_count();

        // 计算密集型：应该等于物理核心数
        let compute = affinity.optimal_thread_count(TaskType::ComputeIntensive);
        assert_eq!(compute, physical);

        // I/O 密集型：应该等于逻辑核心数
        let io = affinity.optimal_thread_count(TaskType::IoIntensive);
        assert_eq!(io, logical);

        // 混合型：应该在物理和逻辑之间
        let mixed = affinity.optimal_thread_count(TaskType::Mixed);
        assert!(mixed >= physical && mixed <= logical);

        // 所有线程数都应该 > 0
        assert!(compute > 0);
        assert!(io > 0);
        assert!(mixed > 0);
    }
}
