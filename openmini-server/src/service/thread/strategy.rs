//! 并行策略模块
//!
//! 提供任务并行划分、负载均衡和动态调度功能。

#![allow(dead_code)]

use crate::hardware::{
    CoreSelectionStrategy, CpuAffinity, HyperthreadEfficiency, HyperthreadTopology, NumaTopology,
    TaskType,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// 并行策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelStrategy {
    /// 串行执行
    Serial,
    /// 数据并行
    DataParallel,
    /// 任务并行
    TaskParallel,
    /// 流水线并行
    PipelineParallel,
    /// 混合并行
    Hybrid,
}

/// 负载均衡策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalanceStrategy {
    /// 静态划分
    Static,
    /// 动态调度
    Dynamic,
    /// 工作窃取
    WorkStealing,
    /// 引导式调度
    Guided,
}

/// 并行执行配置
#[derive(Debug, Clone)]
pub struct ParallelExecutionConfig {
    /// 并行策略
    pub strategy: ParallelStrategy,
    /// 负载均衡策略
    pub load_balance: LoadBalanceStrategy,
    /// 并行度
    pub parallelism: usize,
    /// 最小任务粒度
    pub min_granularity: usize,
    /// 是否启用动态调整
    pub dynamic_adjustment: bool,
    /// 缓存行大小
    pub cache_line_size: usize,
}

impl Default for ParallelExecutionConfig {
    fn default() -> Self {
        let topology = HyperthreadTopology::detect();
        let numa = NumaTopology::detect();
        let affinity = CpuAffinity::new(topology.clone(), numa);

        Self {
            strategy: ParallelStrategy::DataParallel,
            load_balance: LoadBalanceStrategy::Dynamic,
            parallelism: affinity.optimal_thread_count(TaskType::ComputeIntensive),
            min_granularity: 64,
            dynamic_adjustment: true,
            cache_line_size: 64,
        }
    }
}

impl ParallelExecutionConfig {
    /// 创建计算密集型配置
    pub fn compute_intensive() -> Self {
        Self {
            strategy: ParallelStrategy::DataParallel,
            load_balance: LoadBalanceStrategy::Static,
            ..Default::default()
        }
    }

    /// 创建 I/O 密集型配置
    pub fn io_intensive() -> Self {
        Self {
            strategy: ParallelStrategy::TaskParallel,
            load_balance: LoadBalanceStrategy::Dynamic,
            ..Default::default()
        }
    }

    /// 创建混合型配置
    pub fn mixed() -> Self {
        Self {
            strategy: ParallelStrategy::Hybrid,
            load_balance: LoadBalanceStrategy::WorkStealing,
            ..Default::default()
        }
    }
}

/// 任务划分器
#[derive(Debug, Clone)]
pub struct TaskPartitioner {
    /// 配置
    config: ParallelExecutionConfig,
    /// 超线程拓扑
    topology: HyperthreadTopology,
    /// 效率估算
    efficiency: HyperthreadEfficiency,
}

impl TaskPartitioner {
    /// 创建新的任务划分器
    pub fn new(config: ParallelExecutionConfig) -> Self {
        let topology = HyperthreadTopology::detect();
        let efficiency = HyperthreadEfficiency::estimate(&topology, TaskType::ComputeIntensive);

        Self {
            config,
            topology,
            efficiency,
        }
    }

    /// 使用默认配置创建
    pub fn default_partitioner() -> Self {
        Self::new(ParallelExecutionConfig::default())
    }

    /// 划分任务范围
    pub fn partition_range(&self, total: usize) -> Vec<(usize, usize)> {
        if total == 0 {
            return Vec::new();
        }

        let num_workers = self.config.parallelism;
        let chunk_size = self.calculate_chunk_size(total, num_workers);

        let mut partitions = Vec::with_capacity(num_workers);
        let mut start = 0;

        while start < total {
            let end = (start + chunk_size).min(total);
            partitions.push((start, end));
            start = end;
        }

        partitions
    }

    /// 根据负载均衡策略计算块大小
    fn calculate_chunk_size(&self, total: usize, num_workers: usize) -> usize {
        if num_workers == 0 {
            return total;
        }

        match self.config.load_balance {
            LoadBalanceStrategy::Static => {
                let base_size = total / num_workers;
                let adjusted = (base_size as f32 * self.efficiency.thread_factor) as usize;
                adjusted.max(self.config.min_granularity)
            }
            LoadBalanceStrategy::Dynamic => {
                let base_size = total / (num_workers * 4);
                base_size.max(self.config.min_granularity)
            }
            LoadBalanceStrategy::WorkStealing => {
                let base_size = total / (num_workers * 2);
                base_size.max(self.config.min_granularity)
            }
            LoadBalanceStrategy::Guided => {
                let base_size = (total as f32 / num_workers as f32 * 0.5) as usize;
                base_size.max(self.config.min_granularity)
            }
        }
    }

    /// 划分任务到核心
    pub fn partition_to_cores(&self, total: usize) -> Vec<(usize, usize, usize)> {
        let partitions = self.partition_range(total);
        let affinity = CpuAffinity::new(self.topology.clone(), NumaTopology::detect());
        let cores = affinity.select_cores(
            self.config.strategy.to_core_strategy(),
            Some(partitions.len()),
        );

        partitions
            .into_iter()
            .enumerate()
            .map(|(i, (start, end))| {
                let core_id = cores.get(i).copied().unwrap_or(i);
                (start, end, core_id)
            })
            .collect()
    }

    /// 获取最优并行度
    pub fn optimal_parallelism(&self, task_size: usize) -> usize {
        if task_size < self.config.min_granularity {
            return 1;
        }

        let max_parallel = self.config.parallelism;
        let task_based = task_size / self.config.min_granularity;

        max_parallel.min(task_based)
    }

    /// 获取配置
    pub fn config(&self) -> &ParallelExecutionConfig {
        &self.config
    }

    /// 获取拓扑
    pub fn topology(&self) -> &HyperthreadTopology {
        &self.topology
    }
}

impl ParallelStrategy {
    /// 转换为核心选择策略
    pub fn to_core_strategy(self) -> CoreSelectionStrategy {
        match self {
            ParallelStrategy::Serial => CoreSelectionStrategy::PhysicalOnly,
            ParallelStrategy::DataParallel => CoreSelectionStrategy::PhysicalOnly,
            ParallelStrategy::TaskParallel => CoreSelectionStrategy::AllCores,
            ParallelStrategy::PipelineParallel => CoreSelectionStrategy::PerformanceFirst,
            ParallelStrategy::Hybrid => CoreSelectionStrategy::PerformanceFirst,
        }
    }
}

/// 动态调度器
pub struct DynamicScheduler {
    /// 当前任务索引
    current: Arc<AtomicUsize>,
    /// 总任务数
    total: usize,
    /// 块大小
    chunk_size: usize,
    /// 配置
    config: ParallelExecutionConfig,
}

impl DynamicScheduler {
    /// 创建新的动态调度器
    pub fn new(total: usize, config: ParallelExecutionConfig) -> Self {
        let num_workers = config.parallelism;
        let chunk_size = if num_workers > 0 {
            (total / (num_workers * 4)).max(config.min_granularity)
        } else {
            total
        };

        Self {
            current: Arc::new(AtomicUsize::new(0)),
            total,
            chunk_size,
            config,
        }
    }

    /// 获取下一个任务范围
    pub fn next_chunk(&self) -> Option<(usize, usize)> {
        loop {
            let start = self.current.load(Ordering::Relaxed);
            if start >= self.total {
                return None;
            }

            let end = (start + self.chunk_size).min(self.total);

            match self.current.compare_exchange_weak(
                start,
                end,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some((start, end)),
                Err(_) => continue,
            }
        }
    }

    /// 获取剩余任务数
    pub fn remaining(&self) -> usize {
        let current = self.current.load(Ordering::Relaxed);
        self.total.saturating_sub(current)
    }

    /// 重置调度器
    pub fn reset(&self) {
        self.current.store(0, Ordering::SeqCst);
    }
}

/// 引导式调度器
pub struct GuidedScheduler {
    /// 当前任务索引
    current: Arc<AtomicUsize>,
    /// 总任务数
    total: usize,
    /// 工作者数量
    num_workers: usize,
    /// 最小块大小
    min_chunk: usize,
}

impl GuidedScheduler {
    /// 创建新的引导式调度器
    pub fn new(total: usize, num_workers: usize, min_chunk: usize) -> Self {
        Self {
            current: Arc::new(AtomicUsize::new(0)),
            total,
            num_workers: num_workers.max(1),
            min_chunk,
        }
    }

    /// 获取下一个任务范围
    pub fn next_chunk(&self) -> Option<(usize, usize)> {
        loop {
            let start = self.current.load(Ordering::Relaxed);
            if start >= self.total {
                return None;
            }

            let remaining = self.total - start;
            let chunk_size = (remaining / self.num_workers).max(self.min_chunk);
            let end = (start + chunk_size).min(self.total);

            match self.current.compare_exchange_weak(
                start,
                end,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some((start, end)),
                Err(_) => continue,
            }
        }
    }

    /// 获取剩余任务数
    pub fn remaining(&self) -> usize {
        let current = self.current.load(Ordering::Relaxed);
        self.total.saturating_sub(current)
    }
}

/// 缓存友好的数据布局
#[derive(Debug, Clone)]
pub struct CacheFriendlyLayout {
    /// 缓存行大小
    pub cache_line_size: usize,
    /// L1 缓存大小 (字节)
    pub l1_cache_size: usize,
    /// L2 缓存大小 (字节)
    pub l2_cache_size: usize,
    /// L3 缓存大小 (字节)
    pub l3_cache_size: usize,
}

impl Default for CacheFriendlyLayout {
    fn default() -> Self {
        Self {
            cache_line_size: 64,
            l1_cache_size: 32 * 1024,
            l2_cache_size: 256 * 1024,
            l3_cache_size: 8 * 1024 * 1024,
        }
    }
}

impl CacheFriendlyLayout {
    /// 从硬件检测创建
    pub fn from_hardware() -> Self {
        let cache = crate::hardware::CacheTopology::detect();

        Self {
            cache_line_size: cache.cache_line_size(),
            l1_cache_size: cache
                .l1_data_caches
                .first()
                .map(|c| c.size_kb * 1024)
                .unwrap_or(32 * 1024),
            l2_cache_size: cache
                .l2_caches
                .first()
                .map(|c| c.size_kb * 1024)
                .unwrap_or(256 * 1024),
            l3_cache_size: cache.total_l3_size_kb() * 1024,
        }
    }

    /// 计算对齐到缓存行的大小
    pub fn align_to_cache_line(&self, size: usize) -> usize {
        let mask = self.cache_line_size - 1;
        (size + mask) & !mask
    }

    /// 计算适合 L1 缓存的块大小
    pub fn l1_friendly_block_size(&self, element_size: usize) -> usize {
        if element_size == 0 {
            return 0;
        }
        self.l1_cache_size / element_size / 2
    }

    /// 计算适合 L2 缓存的块大小
    pub fn l2_friendly_block_size(&self, element_size: usize) -> usize {
        if element_size == 0 {
            return 0;
        }
        self.l2_cache_size / element_size / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_config_defaults() {
        let config = ParallelExecutionConfig::default();
        assert!(config.parallelism > 0);
        assert!(config.min_granularity > 0);
    }

    #[test]
    fn test_task_partitioner() {
        let partitioner = TaskPartitioner::default_partitioner();

        let partitions = partitioner.partition_range(1000);
        assert!(!partitions.is_empty());

        let total: usize = partitions.iter().map(|(s, e)| e - s).sum();
        assert_eq!(total, 1000);
    }

    #[test]
    fn test_dynamic_scheduler() {
        let config = ParallelExecutionConfig::default();
        let scheduler = DynamicScheduler::new(100, config);

        let mut total = 0;
        while let Some((start, end)) = scheduler.next_chunk() {
            total += end - start;
        }
        assert_eq!(total, 100);
    }

    #[test]
    fn test_guided_scheduler() {
        let scheduler = GuidedScheduler::new(100, 4, 10);

        let mut total = 0;
        let mut chunks = 0;
        while let Some((start, end)) = scheduler.next_chunk() {
            total += end - start;
            chunks += 1;
        }
        assert_eq!(total, 100);
        assert!(chunks > 0);
    }

    #[test]
    fn test_cache_friendly_layout() {
        let layout = CacheFriendlyLayout::from_hardware();

        assert!(layout.cache_line_size >= 16);

        let aligned = layout.align_to_cache_line(100);
        assert_eq!(aligned % layout.cache_line_size, 0);
        assert!(aligned >= 100);
    }

    #[test]
    fn test_optimal_parallelism() {
        let partitioner = TaskPartitioner::default_partitioner();

        let small_parallelism = partitioner.optimal_parallelism(10);
        assert_eq!(small_parallelism, 1);

        let large_parallelism = partitioner.optimal_parallelism(10000);
        assert!(large_parallelism > 0);
    }

    #[test]
    fn test_partition_to_cores() {
        let partitioner = TaskPartitioner::default_partitioner();

        let partitions = partitioner.partition_to_cores(1000);
        assert!(!partitions.is_empty());

        for (start, end, core_id) in &partitions {
            assert!(*start < *end);
            // 核心ID应该是一个有效的索引
            // 不做严格检查，因为核心选择策略可能返回不同的值
            let _ = core_id;
        }
    }

    /// 测试ParallelStrategy枚举的所有变体
    #[test]
    fn test_parallel_strategy_variants() {
        use crate::hardware::CoreSelectionStrategy;

        let strategies = vec![
            ParallelStrategy::Serial,
            ParallelStrategy::DataParallel,
            ParallelStrategy::TaskParallel,
            ParallelStrategy::PipelineParallel,
            ParallelStrategy::Hybrid,
        ];

        // 验证每个策略都能转换为核心选择策略
        for strategy in strategies {
            let _core_strategy = strategy.to_core_strategy();
        }

        // 验证特定映射
        assert_eq!(
            ParallelStrategy::Serial.to_core_strategy(),
            CoreSelectionStrategy::PhysicalOnly
        );
        assert_eq!(
            ParallelStrategy::TaskParallel.to_core_strategy(),
            CoreSelectionStrategy::AllCores
        );
    }

    /// 测试LoadBalanceStrategy枚举
    #[test]
    fn test_load_balance_strategy_variants() {
        let strategies = vec![
            LoadBalanceStrategy::Static,
            LoadBalanceStrategy::Dynamic,
            LoadBalanceStrategy::WorkStealing,
            LoadBalanceStrategy::Guided,
        ];

        // 验证每个变体都可以使用
        for strategy in &strategies {
            let mut config = ParallelExecutionConfig::default();
            config.load_balance = *strategy;
            assert_eq!(config.load_balance, *strategy);
        }
    }

    /// 测试ParallelExecutionConfig的工厂方法
    #[test]
    fn test_config_factory_methods() {
        // 测试compute_intensive配置
        let compute_config = ParallelExecutionConfig::compute_intensive();
        assert_eq!(compute_config.strategy, ParallelStrategy::DataParallel);
        assert_eq!(compute_config.load_balance, LoadBalanceStrategy::Static);

        // 测试io_intensive配置
        let io_config = ParallelExecutionConfig::io_intensive();
        assert_eq!(io_config.strategy, ParallelStrategy::TaskParallel);
        assert_eq!(io_config.load_balance, LoadBalanceStrategy::Dynamic);

        // 测试mixed配置
        let mixed_config = ParallelExecutionConfig::mixed();
        assert_eq!(mixed_config.strategy, ParallelStrategy::Hybrid);
        assert_eq!(mixed_config.load_balance, LoadBalanceStrategy::WorkStealing);
    }

    /// 测试TaskPartitioner - 空范围（边界条件）
    #[test]
    fn test_partition_range_empty() {
        let partitioner = TaskPartitioner::default_partitioner();
        let partitions = partitioner.partition_range(0);

        assert!(partitions.is_empty());
    }

    /// 测试TaskPartitioner - 小范围（边界条件）
    #[test]
    fn test_partition_range_small() {
        let partitioner = TaskPartitioner::default_partitioner();
        let partitions = partitioner.partition_range(5);

        assert!(!partitions.is_empty());
        let total: usize = partitions.iter().map(|(s, e)| e - s).sum();
        assert_eq!(total, 5);
    }

    /// 测试DynamicScheduler - 空任务（边界条件）
    #[test]
    fn test_dynamic_scheduler_empty_tasks() {
        let config = ParallelExecutionConfig::default();
        let scheduler = DynamicScheduler::new(0, config);

        assert_eq!(scheduler.remaining(), 0);
        assert!(scheduler.next_chunk().is_none());
    }

    /// 测试DynamicScheduler - reset功能
    #[test]
    fn test_dynamic_scheduler_reset() {
        let config = ParallelExecutionConfig::default();
        let scheduler = DynamicScheduler::new(100, config);

        // 消费一些任务
        scheduler.next_chunk();
        scheduler.next_chunk();

        let remaining_before = scheduler.remaining();
        assert!(remaining_before < 100);

        // 重置调度器
        scheduler.reset();

        // 验证重置后可以重新获取任务
        assert_eq!(scheduler.remaining(), 100);
        assert!(scheduler.next_chunk().is_some());
    }

    /// 测试GuidedScheduler - 边界条件
    #[test]
    fn test_guided_scheduler_edge_cases() {
        // 0个工作者（应该至少为1）
        let scheduler1 = GuidedScheduler::new(50, 0, 10);
        assert!(scheduler1.next_chunk().is_some());

        // 0个任务
        let scheduler2 = GuidedScheduler::new(0, 4, 10);
        assert!(scheduler2.next_chunk().is_none());
        assert_eq!(scheduler2.remaining(), 0);
    }

    /// 测试CacheFriendlyLayout - 各种方法
    #[test]
    fn test_cache_friendly_layout_methods() {
        let layout = CacheFriendlyLayout::default();

        // 测试对齐到缓存行
        assert_eq!(layout.align_to_cache_line(0), 0);
        assert_eq!(layout.align_to_cache_line(1), 64);
        assert_eq!(layout.align_to_cache_line(64), 64);
        assert_eq!(layout.align_to_cache_line(65), 128);

        // 测试L1友好的块大小
        let l1_block = layout.l1_friendly_block_size(4); // f32大小
        assert!(l1_block > 0);

        // 测试element_size为0的情况
        assert_eq!(layout.l1_friendly_block_size(0), 0);
        assert_eq!(layout.l2_friendly_block_size(0), 0);
    }

    /// 测试并发安全性 - 多线程使用DynamicScheduler
    #[test]
    fn test_dynamic_scheduler_concurrent_safety() {
        use std::sync::{Arc, Barrier};
        use std::thread;

        let config = ParallelExecutionConfig::default();
        let scheduler = Arc::new(DynamicScheduler::new(1000, config));
        let barrier = Arc::new(Barrier::new(4));
        let mut handles = vec![];

        for _ in 0..4 {
            let sched_clone = Arc::clone(&scheduler);
            let barrier_clone = Arc::clone(&barrier);

            handles.push(thread::spawn(move || {
                barrier_clone.wait();

                let mut local_total = 0;
                while let Some((start, end)) = sched_clone.next_chunk() {
                    local_total += end - start;
                }
                local_total
            }));
        }

        let totals: Vec<usize> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let sum: usize = totals.iter().sum();

        // 所有线程处理的总和应该等于总任务数
        assert_eq!(sum, 1000);
    }
}
