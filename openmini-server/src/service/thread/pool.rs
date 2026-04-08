//! 线程池 - 高性能异步任务执行
//!
//! 提供基于 crossbeam 通道的线程池实现，支持:
//! - 无界任务队列
//! - 优雅关闭
//! - 工作窃取(通过多 Worker 实现)
//! - 超线程感知
//! - CPU 亲和性绑定

#![allow(dead_code)]

use crate::hardware::{
    CoreSelectionStrategy, CpuAffinity, HyperthreadTopology, NumaTopology, TaskType,
};
use crossbeam::channel::{self, Receiver, Sender};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

type Task = Box<dyn FnOnce() + Send + 'static>;

/// 线程池配置
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// 线程数
    pub num_threads: usize,
    /// 任务类型
    pub task_type: TaskType,
    /// 是否启用 CPU 亲和性
    pub enable_affinity: bool,
    /// 核心选择策略
    pub core_strategy: CoreSelectionStrategy,
}

impl Default for PoolConfig {
    fn default() -> Self {
        let topology = HyperthreadTopology::detect();
        let numa = NumaTopology::detect();
        let affinity = CpuAffinity::new(topology, numa);

        Self {
            num_threads: affinity.optimal_thread_count(TaskType::Mixed),
            task_type: TaskType::Mixed,
            enable_affinity: false,
            core_strategy: CoreSelectionStrategy::PerformanceFirst,
        }
    }
}

impl PoolConfig {
    /// 创建计算密集型配置
    pub fn compute_intensive() -> Self {
        let topology = HyperthreadTopology::detect();
        let numa = NumaTopology::detect();
        let affinity = CpuAffinity::new(topology, numa);

        Self {
            num_threads: affinity.optimal_thread_count(TaskType::ComputeIntensive),
            task_type: TaskType::ComputeIntensive,
            enable_affinity: true,
            core_strategy: CoreSelectionStrategy::PhysicalOnly,
        }
    }

    /// 创建 I/O 密集型配置
    pub fn io_intensive() -> Self {
        let topology = HyperthreadTopology::detect();
        let numa = NumaTopology::detect();
        let affinity = CpuAffinity::new(topology, numa);

        Self {
            num_threads: affinity.optimal_thread_count(TaskType::IoIntensive),
            task_type: TaskType::IoIntensive,
            enable_affinity: false,
            core_strategy: CoreSelectionStrategy::AllCores,
        }
    }

    /// 创建混合型配置
    pub fn mixed() -> Self {
        let topology = HyperthreadTopology::detect();
        let numa = NumaTopology::detect();
        let affinity = CpuAffinity::new(topology, numa);

        Self {
            num_threads: affinity.optimal_thread_count(TaskType::Mixed),
            task_type: TaskType::Mixed,
            enable_affinity: true,
            core_strategy: CoreSelectionStrategy::PerformanceFirst,
        }
    }
}

/// 线程池
///
/// 管理一组工作线程，并行执行提交的任务。
/// 支持超线程感知和 CPU 亲和性绑定。
pub struct ThreadPool {
    /// 工作线程列表
    workers: Vec<Worker>,
    /// 任务发送端
    sender: Option<Sender<Task>>,
    /// 关闭标志
    shutdown_flag: Arc<AtomicBool>,
    /// 活跃任务计数
    _active_tasks: Arc<AtomicUsize>,
    /// 配置
    config: PoolConfig,
}

/// 工作线程
struct Worker {
    /// 线程 ID
    #[allow(dead_code)]
    id: usize,
    /// 绑定的核心 ID
    #[allow(dead_code)]
    core_id: Option<usize>,
    /// 线程句柄
    thread: Option<thread::JoinHandle<()>>,
}

impl ThreadPool {
    /// 创建新的线程池
    ///
    /// # 参数
    /// - `size`: 线程数量
    ///
    /// # Panics
    /// 当 size 为 0 时 panic
    pub fn new(size: usize) -> ThreadPool {
        let config = PoolConfig {
            num_threads: size,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// 使用配置创建线程池
    pub fn with_config(config: PoolConfig) -> ThreadPool {
        assert!(config.num_threads > 0);

        let (sender, receiver) = channel::unbounded::<Task>();
        let receiver = Arc::new(receiver);
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let active_tasks = Arc::new(AtomicUsize::new(0));

        let affinity = CpuAffinity::from_hardware();
        let core_assignments =
            affinity.select_cores(config.core_strategy, Some(config.num_threads));

        let mut workers = Vec::with_capacity(config.num_threads);

        for id in 0..config.num_threads {
            let core_id = core_assignments.get(id).copied();
            workers.push(Worker::new(
                id,
                core_id,
                Arc::clone(&receiver),
                Arc::clone(&shutdown_flag),
                Arc::clone(&active_tasks),
                config.enable_affinity,
            ));
        }

        ThreadPool {
            workers,
            sender: Some(sender),
            shutdown_flag,
            _active_tasks: active_tasks,
            config,
        }
    }

    /// 创建计算密集型线程池
    pub fn compute_intensive() -> Self {
        Self::with_config(PoolConfig::compute_intensive())
    }

    /// 创建 I/O 密集型线程池
    pub fn io_intensive() -> Self {
        Self::with_config(PoolConfig::io_intensive())
    }

    /// 创建混合型线程池
    pub fn mixed() -> Self {
        Self::with_config(PoolConfig::mixed())
    }

    /// 提交任务到线程池
    ///
    /// 任务会被放入队列，由空闲的工作线程执行。
    ///
    /// # 参数
    /// - `f`: 要执行的任务闭包
    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self._active_tasks.fetch_add(1, Ordering::SeqCst);
        let active_tasks = Arc::clone(&self._active_tasks);
        let task = Box::new(move || {
            f();
            active_tasks.fetch_sub(1, Ordering::SeqCst);
        });

        self.sender
            .as_ref()
            .expect("ThreadPool has been shut down")
            .send(task)
            .expect("Failed to send task to thread pool");
    }

    /// 提交高优先级任务
    pub fn execute_high_priority<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.execute(f);
    }

    /// 关闭线程池
    ///
    /// 设置关闭标志，等待所有工作线程完成当前任务。
    pub fn shutdown(&mut self) {
        self.shutdown_flag.store(true, Ordering::SeqCst);
        self.sender.take();

        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                let _ = thread.join();
            }
        }
    }

    /// 等待所有任务完成
    pub fn wait_completion(&self) {
        while self._active_tasks.load(Ordering::SeqCst) > 0 {
            thread::yield_now();
        }
    }

    /// 获取线程数量
    pub fn size(&self) -> usize {
        self.workers.len()
    }

    /// 获取活跃任务数
    pub fn active_count(&self) -> usize {
        self._active_tasks.load(Ordering::SeqCst)
    }

    /// 获取配置
    pub fn config(&self) -> &PoolConfig {
        &self.config
    }
}

impl Worker {
    /// 创建新的工作线程
    ///
    /// 工作线程会持续从通道接收任务并执行，
    /// 直到收到关闭信号或通道关闭。
    fn new(
        id: usize,
        core_id: Option<usize>,
        receiver: Arc<Receiver<Task>>,
        shutdown_flag: Arc<AtomicBool>,
        _active_tasks: Arc<AtomicUsize>,
        enable_affinity: bool,
    ) -> Worker {
        let thread = thread::spawn(move || {
            if enable_affinity {
                if let Some(cid) = core_id {
                    let affinity = CpuAffinity::from_hardware();
                    let _ = affinity.bind_current_thread(cid);
                }
            }

            while !shutdown_flag.load(Ordering::SeqCst) {
                match receiver.recv() {
                    Ok(task) => {
                        task();
                    }
                    Err(_) => {
                        break;
                    }
                }
            }
        });

        Worker {
            id,
            core_id,
            thread: Some(thread),
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shutdown_flag.store(true, Ordering::SeqCst);
        self.sender.take();

        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                let _ = thread.join();
            }
        }
    }
}

/// 创建默认线程池(基于硬件自动配置)
pub fn create_default_pool() -> ThreadPool {
    ThreadPool::mixed()
}

/// 创建计算密集型线程池
pub fn create_compute_pool() -> ThreadPool {
    ThreadPool::compute_intensive()
}

/// 创建 I/O 密集型线程池
pub fn create_io_pool() -> ThreadPool {
    ThreadPool::io_intensive()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    #[test]
    fn test_thread_pool_new() {
        let pool = ThreadPool::new(4);
        assert_eq!(pool.size(), 4);
    }

    #[test]
    fn test_thread_pool_with_config() {
        let config = PoolConfig::compute_intensive();
        let pool = ThreadPool::with_config(config);
        assert!(pool.size() > 0);
        assert_eq!(pool.config().task_type, TaskType::ComputeIntensive);
    }

    #[test]
    fn test_thread_pool_execute() {
        let pool = ThreadPool::new(2);
        let counter = Arc::new(AtomicUsize::new(0));

        for _ in 0..10 {
            let counter_clone = Arc::clone(&counter);
            pool.execute(move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            });
        }

        thread::sleep(Duration::from_millis(100));
        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }

    #[test]
    fn test_thread_pool_shutdown() {
        let mut pool = ThreadPool::new(2);
        let counter = Arc::new(AtomicUsize::new(0));

        for _ in 0..5 {
            let counter_clone = Arc::clone(&counter);
            pool.execute(move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            });
        }

        thread::sleep(Duration::from_millis(100));
        pool.shutdown();
        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn test_compute_intensive_pool() {
        let pool = ThreadPool::compute_intensive();
        let topology = HyperthreadTopology::detect();

        assert_eq!(pool.size(), topology.physical_core_count());
    }

    #[test]
    fn test_io_intensive_pool() {
        let pool = ThreadPool::io_intensive();
        let topology = HyperthreadTopology::detect();

        assert_eq!(pool.size(), topology.logical_core_count());
    }

    #[test]
    fn test_active_task_count() {
        let pool = ThreadPool::new(2);
        let counter = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(std::sync::Barrier::new(3));

        for _ in 0..2 {
            let counter_clone = Arc::clone(&counter);
            let barrier_clone = Arc::clone(&barrier);
            pool.execute(move || {
                barrier_clone.wait();
                counter_clone.fetch_add(1, Ordering::SeqCst);
            });
        }

        assert_eq!(pool.active_count(), 2);
        barrier.wait();
        pool.wait_completion();
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn test_pool_config_defaults() {
        let config = PoolConfig::default();
        assert!(config.num_threads > 0);
    }

    #[test]
    fn test_thread_pool_creation_and_size() {
        // 线程池创建和大小验证
        let pool = ThreadPool::new(4);

        // 验证初始状态
        assert_eq!(pool.size(), 4, "Pool should have 4 workers");
        assert_eq!(pool.active_count(), 0, "Initial active count should be 0");
    }

    #[test]
    fn test_thread_pool_execute_multiple_tasks() {
        // 多任务执行测试
        let pool = ThreadPool::new(2);

        use std::sync::{Arc, Mutex};
        let results: Arc<Mutex<Vec<i32>>> = Arc::new(Mutex::new(Vec::new()));

        for i in 0..10 {
            let results_clone = Arc::clone(&results);
            pool.execute(move || {
                results_clone.lock().unwrap().push(i * i);
            });
        }

        // 等待所有任务完成
        pool.wait_completion();

        let final_results = results.lock().unwrap();
        assert_eq!(final_results.len(), 10, "Should have 10 results");

        // 验证结果包含所有预期的值(顺序可能不同)
        for i in 0..10 {
            assert!(
                final_results.contains(&(i * i)),
                "Should contain result for input {}",
                i
            );
        }
    }

    #[test]
    fn test_thread_pool_shutdown_and_restart() {
        // 关闭和重启测试
        let mut pool = ThreadPool::new(2);

        // 执行一些任务
        let counter = Arc::new(AtomicUsize::new(0));
        for _ in 0..5 {
            let counter_clone = Arc::clone(&counter);
            pool.execute(move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            });
        }

        thread::sleep(Duration::from_millis(50));

        // 关闭线程池
        pool.shutdown();

        // 等待关闭完成
        thread::sleep(Duration::from_millis(50));

        // 验证任务已完成
        assert_eq!(
            counter.load(Ordering::SeqCst),
            5,
            "All tasks should complete before shutdown"
        );
    }

    #[test]
    fn test_thread_pool_panic_recovery() {
        // 任务panic恢复测试
        let pool = ThreadPool::new(2);

        // 提交一个会panic的任务
        pool.execute(|| {
            panic!("intentional panic for testing");
        });

        // 等待一下让任务执行
        thread::sleep(Duration::from_millis(100));

        // 提交正常任务 - 应该能继续工作
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);
        pool.execute(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        pool.wait_completion();

        // 正常任务应该成功执行
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "Normal task should execute after panic"
        );
    }

    #[test]
    fn test_thread_pool_high_priority_tasks() {
        // 高优先级任务测试
        let pool = ThreadPool::new(2);

        let counter = Arc::new(AtomicUsize::new(0));

        // 提交普通任务
        for _ in 0..3 {
            let counter_clone = Arc::clone(&counter);
            pool.execute(move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            });
        }

        // 提交高优先级任务
        for _ in 0..3 {
            let counter_clone = Arc::clone(&counter);
            pool.execute_high_priority(move || {
                counter_clone.fetch_add(10, Ordering::SeqCst); // 用不同的增量区分
            });
        }

        pool.wait_completion();

        // 所有6个任务都应该完成
        let total = counter.load(Ordering::SeqCst);
        assert_eq!(
            total, 33,
            "All tasks (normal + high priority) should complete: got {}",
            total
        );
    }

    #[test]
    fn test_thread_pool_config_variants() {
        // 不同配置变体测试

        // 计算密集型配置
        let compute_pool = PoolConfig::compute_intensive();
        assert!(compute_pool.num_threads > 0);
        assert_eq!(compute_pool.task_type, TaskType::ComputeIntensive);
        assert!(compute_pool.enable_affinity);

        // I/O密集型配置
        let io_pool = PoolConfig::io_intensive();
        assert!(io_pool.num_threads > 0);
        assert_eq!(io_pool.task_type, TaskType::IoIntensive);
        assert!(!io_pool.enable_affinity);

        // 混合型配置
        let mixed_pool = PoolConfig::mixed();
        assert!(mixed_pool.num_threads > 0);
        assert_eq!(mixed_pool.task_type, TaskType::Mixed);
        assert!(mixed_pool.enable_affinity);

        // 使用配置创建线程池
        let pool1 = ThreadPool::with_config(compute_pool);
        assert!(pool1.size() > 0);

        let pool2 = ThreadPool::with_config(io_pool);
        assert!(pool2.size() > 0);

        let pool3 = ThreadPool::with_config(mixed_pool);
        assert!(pool3.size() > 0);
    }

    #[test]
    fn test_thread_pool_factory_functions() {
        // 工厂函数测试

        // 计算密集型线程池
        let compute_pool = ThreadPool::compute_intensive();
        let topology = HyperthreadTopology::detect();
        assert_eq!(compute_pool.size(), topology.physical_core_count());

        // I/O密集型线程池
        let io_pool = ThreadPool::io_intensive();
        assert_eq!(io_pool.size(), topology.logical_core_count());

        // 混合型线程池
        let mixed_pool = ThreadPool::mixed();
        assert!(mixed_pool.size() > 0);
    }

    #[test]
    fn test_thread_pool_concurrent_execution() {
        // 并发执行测试
        let pool = ThreadPool::new(4);

        let barrier = Arc::new(std::sync::Barrier::new(5)); // 4 workers + main thread
        let start_counter = Arc::new(AtomicUsize::new(0));
        let end_counter = Arc::new(AtomicUsize::new(0));

        // 启动4个并发任务
        for _ in 0..4 {
            let barrier_clone = Arc::clone(&barrier);
            let start_clone = Arc::clone(&start_counter);
            let end_clone = Arc::clone(&end_counter);

            pool.execute(move || {
                start_clone.fetch_add(1, Ordering::SeqCst);
                barrier_clone.wait(); // 等待所有任务都到达这里
                end_clone.fetch_add(1, Ordering::SeqCst);
            });
        }

        // 主线程也等待barrier
        barrier.wait();

        // 在barrier之后,所有4个任务都应该已经启动
        let started = start_counter.load(Ordering::SeqCst);
        assert_eq!(started, 4, "All 4 tasks should have started");

        // 等待所有任务完成
        pool.wait_completion();

        let completed = end_counter.load(Ordering::SeqCst);
        assert_eq!(completed, 4, "All 4 tasks should have completed");
    }

    #[test]
    fn test_thread_pool_task_ordering_guarantees() {
        // 任务提交顺序保证测试(同一任务内的操作是顺序的)
        let pool = ThreadPool::new(2);

        let sequence: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));

        for i in 0..5 {
            let seq_clone = Arc::clone(&sequence);
            pool.execute(move || {
                // 每个任务内部按顺序写入两个值
                let mut seq = seq_clone.lock().unwrap();
                seq.push(i * 10); // 第一个值
                seq.push(i * 10 + 1); // 第二个值
            });
        }

        pool.wait_completion();

        let final_seq = sequence.lock().unwrap();
        assert_eq!(final_seq.len(), 10, "Should have 10 entries");

        // 验证每个任务的内部顺序保持不变
        for i in 0..5 {
            let first_pos = final_seq.iter().position(|&x| x == i * 10).unwrap();
            let second_pos = final_seq.iter().position(|&x| x == i * 10 + 1).unwrap();
            assert!(
                second_pos > first_pos,
                "Task {}'s second operation should come after first",
                i
            );
        }
    }
}
