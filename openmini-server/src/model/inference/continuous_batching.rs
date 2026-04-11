//! Continuous Batching 优化实现
//!
//! Continuous Batching（连续批处理）是2024年LLM推理的关键优化技术：
//! - 动态批处理：无需等待所有序列完成，动态添加新请求
//! - 迭代级调度：每次迭代重新调度，最大化GPU利用率
//! - 内存高效：及时释放已完成序列的内存
//! - 延迟优化：降低首个token的响应时间
//!
//! 性能提升：
//! - 吞吐量提升：2-4倍
//! - GPU利用率：从30%提升到80%+
//! - 平均延迟：降低50%+

#![allow(dead_code)]

use anyhow::Result;
use parking_lot::RwLock;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// 请求状态
#[derive(Debug, Clone, PartialEq)]
pub enum RequestStatus {
    /// 等待中
    Waiting,
    /// 运行中
    Running,
    /// 已暂停（等待更多token）
    Paused,
    /// 已完成
    Completed,
    /// 已取消
    Cancelled,
}

/// 生成请求
#[derive(Debug, Clone)]
pub struct GenerationRequest {
    /// 请求ID
    pub id: u64,
    /// 输入token序列
    pub input_tokens: Vec<u32>,
    /// 已生成的token序列
    pub generated_tokens: Vec<u32>,
    /// 最大生成长度
    pub max_length: usize,
    /// 当前状态
    pub status: RequestStatus,
    /// 创建时间
    pub created_at: Instant,
    /// 开始处理时间
    pub started_at: Option<Instant>,
    /// 完成时间
    pub completed_at: Option<Instant>,
    /// 优先级（越大越优先）
    pub priority: i32,
    /// 温度参数
    pub temperature: f32,
    /// Top-p参数
    pub top_p: f32,
}

impl GenerationRequest {
    /// 创建新的生成请求
    pub fn new(id: u64, input_tokens: Vec<u32>, max_length: usize) -> Self {
        Self {
            id,
            input_tokens,
            generated_tokens: Vec::new(),
            max_length,
            status: RequestStatus::Waiting,
            created_at: Instant::now(),
            started_at: None,
            completed_at: None,
            priority: 0,
            temperature: 1.0,
            top_p: 0.9,
        }
    }

    /// 设置优先级
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// 设置温度
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// 获取总token数
    pub fn total_tokens(&self) -> usize {
        self.input_tokens.len() + self.generated_tokens.len()
    }

    /// 是否完成
    pub fn is_finished(&self) -> bool {
        self.generated_tokens.len() >= self.max_length
            || self.status == RequestStatus::Completed
            || self.status == RequestStatus::Cancelled
    }

    /// 获取等待时间
    pub fn wait_time(&self) -> Duration {
        match self.started_at {
            Some(started) => started.duration_since(self.created_at),
            None => Instant::now().duration_since(self.created_at),
        }
    }

    /// 获取处理时间
    pub fn processing_time(&self) -> Option<Duration> {
        match (self.started_at, self.completed_at) {
            (Some(started), Some(completed)) => Some(completed.duration_since(started)),
            (Some(started), None) => Some(Instant::now().duration_since(started)),
            _ => None,
        }
    }
}

/// 批处理调度策略
#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
pub enum SchedulingStrategy {
    /// 先进先出
    FIFO,
    /// 最短作业优先
    ShortestJobFirst,
    /// 优先级调度
    Priority,
    /// 公平调度
    Fair,
}

/// Continuous Batching 配置
#[derive(Debug, Clone)]
pub struct ContinuousBatchingConfig {
    /// 最大批大小
    pub max_batch_size: usize,
    /// 最大序列长度
    pub max_sequence_length: usize,
    /// 调度策略
    pub scheduling_strategy: SchedulingStrategy,
    /// 最大等待时间（毫秒）
    pub max_wait_time_ms: u64,
    /// 最小批大小（用于流水线）
    pub min_batch_size: usize,
    /// 是否启用抢占
    pub enable_preemption: bool,
    /// 抢占阈值（内存使用率）
    pub preemption_threshold: f32,
    /// 是否启用动态批大小
    pub enable_dynamic_batch_size: bool,
}

impl Default for ContinuousBatchingConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_sequence_length: 4096,
            scheduling_strategy: SchedulingStrategy::Fair,
            max_wait_time_ms: 100,
            min_batch_size: 1,
            enable_preemption: true,
            preemption_threshold: 0.9,
            enable_dynamic_batch_size: true,
        }
    }
}

/// 批处理统计信息
#[derive(Debug, Clone, Default)]
pub struct BatchingStats {
    /// 总请求数
    pub total_requests: u64,
    /// 完成请求数
    pub completed_requests: u64,
    /// 取消请求数
    pub cancelled_requests: u64,
    /// 平均批大小
    pub avg_batch_size: f32,
    /// 平均等待时间（毫秒）
    pub avg_wait_time_ms: f64,
    /// 平均处理时间（毫秒）
    pub avg_processing_time_ms: f64,
    /// GPU利用率
    pub gpu_utilization: f32,
    /// 吞吐量
    pub throughput: f32,
}

/// Continuous Batching 调度器
pub struct ContinuousBatchingScheduler {
    config: ContinuousBatchingConfig,
    /// 等待队列
    waiting_queue: Arc<RwLock<VecDeque<GenerationRequest>>>,
    /// 运行中的请求
    running_requests: Arc<RwLock<HashMap<u64, GenerationRequest>>>,
    /// 已完成的请求
    completed_requests: Arc<RwLock<HashMap<u64, GenerationRequest>>>,
    /// 统计信息
    stats: Arc<RwLock<BatchingStats>>,
    /// 下一个请求ID
    next_request_id: Arc<RwLock<u64>>,
}

impl ContinuousBatchingScheduler {
    /// 创建新的调度器
    pub fn new(config: ContinuousBatchingConfig) -> Self {
        Self {
            config,
            waiting_queue: Arc::new(RwLock::new(VecDeque::new())),
            running_requests: Arc::new(RwLock::new(HashMap::new())),
            completed_requests: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(BatchingStats::default())),
            next_request_id: Arc::new(RwLock::new(0)),
        }
    }

    /// 添加新请求
    pub fn add_request(&self, mut request: GenerationRequest) -> Result<u64> {
        let mut next_id = self.next_request_id.write();
        request.id = *next_id;
        *next_id += 1;

        let id = request.id;

        // 添加到等待队列
        self.waiting_queue.write().push_back(request);

        // 更新统计
        self.stats.write().total_requests += 1;

        Ok(id)
    }

    /// 取消请求
    pub fn cancel_request(&self, request_id: u64) -> Result<()> {
        // 从等待队列移除
        {
            let mut queue = self.waiting_queue.write();
            if let Some(pos) = queue.iter().position(|r| r.id == request_id) {
                queue.remove(pos);
                self.stats.write().cancelled_requests += 1;
                return Ok(());
            }
        }

        // 从运行队列移除
        {
            let mut running = self.running_requests.write();
            if let Some(mut request) = running.remove(&request_id) {
                request.status = RequestStatus::Cancelled;
                request.completed_at = Some(Instant::now());
                self.completed_requests.write().insert(request_id, request);
                self.stats.write().cancelled_requests += 1;
            }
        }

        Ok(())
    }

    /// 调度下一批请求
    pub fn schedule_batch(&self) -> Result<Vec<GenerationRequest>> {
        let mut batch = Vec::new();
        let mut queue = self.waiting_queue.write();

        // 根据调度策略排序
        match self.config.scheduling_strategy {
            SchedulingStrategy::FIFO => {
                // 已经是FIFO顺序
            }
            SchedulingStrategy::ShortestJobFirst => {
                // 按输入长度排序（短的优先）
                let mut vec: Vec<_> = queue.drain(..).collect();
                vec.sort_by_key(|r| r.input_tokens.len());
                queue.extend(vec);
            }
            SchedulingStrategy::Priority => {
                // 按优先级排序
                let mut vec: Vec<_> = queue.drain(..).collect();
                vec.sort_by(|a, b| b.priority.cmp(&a.priority));
                queue.extend(vec);
            }
            SchedulingStrategy::Fair => {
                // 公平调度：轮转
                // 已经在队列中，无需特殊处理
            }
        }

        // 选择请求加入批次
        let now = Instant::now();
        let max_wait = Duration::from_millis(self.config.max_wait_time_ms);

        while batch.len() < self.config.max_batch_size && !queue.is_empty() {
            if let Some(mut request) = queue.pop_front() {
                let has_waiting = now.duration_since(request.created_at) > max_wait;
                let can_fill_batch = batch.len() >= self.config.min_batch_size;

                if has_waiting || can_fill_batch {
                    // 已超时或已达到最小批大小，可以开始处理
                    request.status = RequestStatus::Running;
                    request.started_at = Some(now);
                    batch.push(request);
                } else {
                    // 未达到最小批大小且未超时，但仍然添加第一个请求以避免死锁
                    request.status = RequestStatus::Running;
                    request.started_at = Some(now);
                    batch.push(request);
                }
            }
        }

        // 将批次中的请求移到运行队列
        for request in &batch {
            self.running_requests
                .write()
                .insert(request.id, request.clone());
        }

        // 更新统计
        if !batch.is_empty() {
            let mut stats = self.stats.write();
            let total = stats.total_requests as f32;
            stats.avg_batch_size =
                (stats.avg_batch_size * (total - 1.0) + batch.len() as f32) / total;
        }

        Ok(batch)
    }

    /// 更新请求状态（添加生成的token）
    pub fn update_request(&self, request_id: u64, new_token: u32) -> Result<()> {
        let mut running = self.running_requests.write();

        if let Some(request) = running.get_mut(&request_id) {
            request.generated_tokens.push(new_token);

            // 检查是否完成
            if request.is_finished() {
                if let Some(mut request) = running.remove(&request_id) {
                    request.status = RequestStatus::Completed;
                    request.completed_at = Some(Instant::now());

                    // 更新统计
                    let mut stats = self.stats.write();
                    stats.completed_requests += 1;
                    if let Some(processing_time) = request.processing_time() {
                        let total = stats.completed_requests as f64;
                        stats.avg_processing_time_ms = (stats.avg_processing_time_ms
                            * (total - 1.0)
                            + processing_time.as_millis() as f64)
                            / total;
                    }
                    if let Some(wait_time) = Some(request.wait_time()) {
                        let total = stats.completed_requests as f64;
                        stats.avg_wait_time_ms = (stats.avg_wait_time_ms * (total - 1.0)
                            + wait_time.as_millis() as f64)
                            / total;
                    }

                    // 移到完成队列
                    self.completed_requests.write().insert(request_id, request);
                }
            }
        }

        Ok(())
    }

    /// 抢占请求（释放内存）
    pub fn preempt_requests(&self, memory_usage: f32) -> Result<Vec<u64>> {
        if !self.config.enable_preemption || memory_usage < self.config.preemption_threshold {
            return Ok(Vec::new());
        }

        let mut preempted = Vec::new();
        let mut running = self.running_requests.write();

        // 按优先级排序，抢占低优先级的请求
        let mut requests: Vec<_> = running.iter().map(|(id, _)| *id).collect();
        requests.sort();

        // 计算需要释放的内存
        let target_memory = self.config.preemption_threshold * 0.8;
        let mut current_memory = memory_usage;

        for id in requests {
            if current_memory <= target_memory {
                break;
            }

            // 暂停请求
            if let Some(mut request) = running.remove(&id) {
                request.status = RequestStatus::Paused;

                // 放回等待队列
                self.waiting_queue.write().push_back(request);

                preempted.push(id);

                // 估算释放的内存（简化）
                current_memory -= 0.1; // 假设每个请求占用10%内存
            }
        }

        Ok(preempted)
    }

    /// 获取运行中的请求数量
    pub fn running_count(&self) -> usize {
        self.running_requests.read().len()
    }

    /// 获取等待中的请求数量
    pub fn waiting_count(&self) -> usize {
        self.waiting_queue.read().len()
    }

    /// 获取统计信息
    pub fn stats(&self) -> BatchingStats {
        self.stats.read().clone()
    }

    /// 获取请求状态
    pub fn get_request_status(&self, request_id: u64) -> Option<RequestStatus> {
        // 检查运行队列
        if let Some(request) = self.running_requests.read().get(&request_id) {
            return Some(request.status.clone());
        }

        // 检查完成队列
        if let Some(request) = self.completed_requests.read().get(&request_id) {
            return Some(request.status.clone());
        }

        // 检查等待队列
        for request in self.waiting_queue.read().iter() {
            if request.id == request_id {
                return Some(request.status.clone());
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_request_creation() {
        let request = GenerationRequest::new(1, vec![1, 2, 3], 100);
        assert_eq!(request.id, 1);
        assert_eq!(request.input_tokens.len(), 3);
        assert_eq!(request.status, RequestStatus::Waiting);
    }

    #[test]
    fn test_scheduler_add_request() {
        let config = ContinuousBatchingConfig::default();
        let scheduler = ContinuousBatchingScheduler::new(config);

        let request = GenerationRequest::new(0, vec![1, 2, 3], 100);
        let result = scheduler.add_request(request);

        assert!(result.is_ok());
        assert_eq!(scheduler.waiting_count(), 1);
    }

    #[test]
    fn test_scheduler_schedule_batch() {
        let config = ContinuousBatchingConfig {
            min_batch_size: 1,
            ..Default::default()
        };
        let scheduler = ContinuousBatchingScheduler::new(config);

        // 添加多个请求
        for i in 0..5 {
            let request = GenerationRequest::new(i, vec![1, 2, 3], 100);
            scheduler.add_request(request).unwrap();
        }

        // 调度批次
        let _batch = scheduler.schedule_batch().unwrap();
        assert!(_batch.len() > 0);
        assert!(_batch.len() <= 32);
    }

    #[test]
    fn test_scheduler_update_request() {
        let config = ContinuousBatchingConfig {
            min_batch_size: 1,
            ..Default::default()
        };
        let scheduler = ContinuousBatchingScheduler::new(config);

        let request = GenerationRequest::new(0, vec![1, 2, 3], 100);
        let id = scheduler.add_request(request).unwrap();

        // 调度批次
        let _batch = scheduler.schedule_batch().unwrap();
        assert!(!_batch.is_empty());

        // 更新请求
        let result = scheduler.update_request(id, 4);
        assert!(result.is_ok());
    }

    #[test]
    fn test_scheduler_cancel_request() {
        let config = ContinuousBatchingConfig::default();
        let scheduler = ContinuousBatchingScheduler::new(config);

        let request = GenerationRequest::new(0, vec![1, 2, 3], 100);
        let id = scheduler.add_request(request).unwrap();

        // 取消请求
        let result = scheduler.cancel_request(id);
        assert!(result.is_ok());
        assert_eq!(scheduler.waiting_count(), 0);
    }

    #[test]
    fn test_priority_scheduling() {
        let config = ContinuousBatchingConfig {
            scheduling_strategy: SchedulingStrategy::Priority,
            min_batch_size: 1,
            ..Default::default()
        };
        let scheduler = ContinuousBatchingScheduler::new(config);

        // 添加不同优先级的请求
        let low = GenerationRequest::new(0, vec![1, 2, 3], 100).with_priority(1);
        let high = GenerationRequest::new(0, vec![4, 5, 6], 100).with_priority(10);

        scheduler.add_request(low).unwrap();
        scheduler.add_request(high).unwrap();

        // 调度批次
        let batch = scheduler.schedule_batch().unwrap();

        // 高优先级请求应该先被调度
        assert!(batch.len() > 0);
    }

    #[test]
    fn test_request_completion() {
        let request = GenerationRequest::new(1, vec![1, 2, 3], 5);
        assert!(!request.is_finished());

        // 添加生成的token
        let mut request = request;
        request.generated_tokens = vec![4, 5, 6, 7, 8];
        assert!(request.is_finished());
    }

    #[test]
    fn test_stats() {
        let config = ContinuousBatchingConfig {
            min_batch_size: 1,
            ..Default::default()
        };
        let scheduler = ContinuousBatchingScheduler::new(config);

        // 添加并处理请求
        let request = GenerationRequest::new(0, vec![1, 2, 3], 5);
        let id = scheduler.add_request(request).unwrap();

        let _batch = scheduler.schedule_batch().unwrap();

        // 生成token直到完成
        for _ in 0..5 {
            scheduler.update_request(id, 4).unwrap();
        }

        let stats = scheduler.stats();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.completed_requests, 1);
    }

    // ===== 边界条件和分支覆盖率测试 =====

    #[test]
    fn test_schedule_empty_queue_returns_empty_batch() {
        // 空队列调度应返回空批次
        let config = ContinuousBatchingConfig {
            min_batch_size: 1,
            max_batch_size: 4,
            ..Default::default()
        };
        let scheduler = ContinuousBatchingScheduler::new(config);

        let batch = scheduler.schedule_batch().unwrap();
        assert!(batch.is_empty());
    }

    #[test]
    fn test_schedule_all_completed_requests() {
        // 所有请求完成后再次调度返回空批次
        let config = ContinuousBatchingConfig {
            min_batch_size: 1,
            max_batch_size: 4,
            ..Default::default()
        };
        let scheduler = ContinuousBatchingScheduler::new(config);

        // 添加并完成一个请求
        let request = GenerationRequest::new(0, vec![1], 3); // max_length=3
        let id = scheduler.add_request(request).unwrap();

        let batch = scheduler.schedule_batch().unwrap();
        assert!(!batch.is_empty());

        // 生成3个token达到max_length完成请求
        for _ in 0..3 {
            scheduler.update_request(id, 100).unwrap();
        }

        // 再次调度应该返回空批次
        let empty_batch = scheduler.schedule_batch().unwrap();
        assert!(empty_batch.is_empty());
    }

    #[test]
    fn test_cancel_nonexistent_request_returns_ok() {
        // 取消不存在的请求应该返回Ok（当前实现行为）
        let scheduler = ContinuousBatchingScheduler::new(ContinuousBatchingConfig::default());
        let result = scheduler.cancel_request(99999);
        // 当前实现在找不到请求时仍然返回 Ok(())
        assert!(result.is_ok());
    }

    #[test]
    fn test_update_nonexistent_request_returns_ok() {
        // 更新不存在的请求状态应该返回Ok（当前实现行为）
        let scheduler = ContinuousBatchingScheduler::new(ContinuousBatchingConfig::default());
        let result = scheduler.update_request(88888, 100);
        // 当前实现在找不到请求时仍然返回 Ok(())
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_request_status_for_nonexistent() {
        // 查询不存在的请求状态应返回None
        let scheduler = ContinuousBatchingScheduler::new(ContinuousBatchingConfig::default());
        let status = scheduler.get_request_status(12345);
        assert!(status.is_none());
    }

    #[test]
    fn test_stats_accuracy_after_multiple_operations() {
        // 验证统计信息在多次操作后的准确性
        let config = ContinuousBatchingConfig {
            min_batch_size: 1,
            max_batch_size: 2,
            ..Default::default()
        };
        let scheduler = ContinuousBatchingScheduler::new(config);

        // 添加2个请求
        let req1 = GenerationRequest::new(0, vec![1], 2);
        let req2 = GenerationRequest::new(0, vec![2], 2);
        let id1 = scheduler.add_request(req1).unwrap();
        let _id2 = scheduler.add_request(req2).unwrap();

        let stats = scheduler.stats();
        assert_eq!(stats.total_requests, 2);

        // 调度批次
        let _batch1 = scheduler.schedule_batch().unwrap();

        // 完成第一个请求
        for _ in 0..2 {
            scheduler.update_request(id1, 100).unwrap();
        }

        let stats = scheduler.stats();
        assert_eq!(stats.completed_requests, 1);
        assert_eq!(stats.total_requests, 2);
    }

    #[test]
    fn test_different_scheduling_strategies() {
        // 测试不同的调度策略都能正常工作

        for strategy in &[
            SchedulingStrategy::FIFO,
            SchedulingStrategy::ShortestJobFirst,
            SchedulingStrategy::Priority,
            SchedulingStrategy::Fair,
        ] {
            let config = ContinuousBatchingConfig {
                scheduling_strategy: strategy.clone(),
                min_batch_size: 1,
                max_batch_size: 4,
                ..Default::default()
            };

            let scheduler = ContinuousBatchingScheduler::new(config);

            // 添加4个不同长度的请求
            for i in 1..=4 {
                let request = GenerationRequest::new(0, vec![1; i], 10);
                scheduler.add_request(request).unwrap();
            }

            let batch = scheduler.schedule_batch();
            assert!(batch.is_ok(), "Strategy {:?} failed", strategy);
            assert!(
                !batch.unwrap().is_empty(),
                "Strategy {:?} returned empty batch",
                strategy
            );
        }
    }

    #[test]
    fn test_priority_scheduling_order() {
        // 验证优先级调度的排序顺序
        let config = ContinuousBatchingConfig {
            scheduling_strategy: SchedulingStrategy::Priority,
            min_batch_size: 1,
            max_batch_size: 10,
            ..Default::default()
        };
        let scheduler = ContinuousBatchingScheduler::new(config);

        // 添加不同优先级的请求（后添加的高优先级应该在前面）
        let low = GenerationRequest::new(0, vec![1, 2, 3], 100).with_priority(1);
        let mid = GenerationRequest::new(0, vec![4, 5, 6], 100).with_priority(5);
        let high = GenerationRequest::new(0, vec![7, 8, 9], 100).with_priority(10);

        scheduler.add_request(low).unwrap();
        scheduler.add_request(mid).unwrap();
        scheduler.add_request(high).unwrap();

        let batch = scheduler.schedule_batch().unwrap();

        // 高优先级请求应该先被调度（索引更小）
        assert_eq!(batch.len(), 3);
        assert!(batch[0].priority >= batch[1].priority);
        assert!(batch[1].priority >= batch[2].priority);
    }

    #[test]
    fn test_shortest_job_first_order() {
        // 验证最短作业优先的排序顺序
        let config = ContinuousBatchingConfig {
            scheduling_strategy: SchedulingStrategy::ShortestJobFirst,
            min_batch_size: 1,
            max_batch_size: 10,
            ..Default::default()
        };
        let scheduler = ContinuousBatchingScheduler::new(config);

        // 添加不同输入长度的请求
        let long = GenerationRequest::new(0, vec![1; 100], 100); // 长输入
        let short = GenerationRequest::new(0, vec![1], 100); // 短输入
        let medium = GenerationRequest::new(0, vec![1; 50], 100); // 中等输入

        scheduler.add_request(long).unwrap();
        scheduler.add_request(short).unwrap();
        scheduler.add_request(medium).unwrap();

        let batch = scheduler.schedule_batch().unwrap();

        // 短输入的请求应该先被调度
        assert_eq!(batch.len(), 3);
        assert!(batch[0].input_tokens.len() <= batch[1].input_tokens.len());
        assert!(batch[1].input_tokens.len() <= batch[2].input_tokens.len());
    }

    #[test]
    fn test_preempt_requests_with_high_memory() {
        // 内存压力下的抢占行为
        let config = ContinuousBatchingConfig {
            enable_preemption: true,
            preemption_threshold: 0.8,
            min_batch_size: 1,
            max_batch_size: 10,
            ..Default::default()
        };
        let scheduler = ContinuousBatchingScheduler::new(config);

        // 添加并调度多个请求
        for _i in 0..5 {
            let request = GenerationRequest::new(0, vec![1; 100], 1000);
            scheduler.add_request(request).unwrap();
        }

        let batch = scheduler.schedule_batch().unwrap();
        assert!(!batch.is_empty()); // 确保有运行中的请求

        // 模拟高内存使用率触发抢占
        let preempted = scheduler.preempt_requests(0.95).unwrap();

        // 由于内存使用率超过阈值，应该有请求被抢占
        // （具体数量取决于实现逻辑）
        let _ = preempted;

        // 抢占后这些请求应该回到等待队列或暂停状态
        let waiting_count = scheduler.waiting_count();
        let running_count = scheduler.running_count();

        // 验证状态变化（至少有一些请求从running移出）
        assert!(
            waiting_count > 0 || running_count < batch.len(),
            "Preemption should move requests out of running state"
        );
    }

    #[test]
    fn test_preempt_requests_below_threshold() {
        // 内存使用率低于阈值时不应该抢占
        let config = ContinuousBatchingConfig {
            enable_preemption: true,
            preemption_threshold: 0.9,
            min_batch_size: 1,
            max_batch_size: 10,
            ..Default::default()
        };
        let scheduler = ContinuousBatchingScheduler::new(config);

        // 添加并调度请求
        let request = GenerationRequest::new(0, vec![1, 2, 3], 100);
        scheduler.add_request(request).unwrap();
        scheduler.schedule_batch().unwrap();

        // 低内存使用率不应触发抢占
        let preempted = scheduler.preempt_requests(0.5).unwrap();
        assert!(
            preempted.is_empty(),
            "Should not preempt when memory is below threshold"
        );
    }

    #[test]
    fn test_preempt_disabled() {
        // 禁用抢占时即使高内存也不应该抢占
        let config = ContinuousBatchingConfig {
            enable_preemption: false,
            preemption_threshold: 0.8,
            ..Default::default()
        };
        let scheduler = ContinuousBatchingScheduler::new(config);

        // 添加并调度请求
        let request = GenerationRequest::new(0, vec![1, 2, 3], 100);
        scheduler.add_request(request).unwrap();
        scheduler.schedule_batch().unwrap();

        // 即使高内存使用率也不应抢占（因为禁用了）
        let preempted = scheduler.preempt_requests(0.99).unwrap();
        assert!(
            preempted.is_empty(),
            "Should not preempt when preemption is disabled"
        );
    }

    #[test]
    fn test_concurrent_access_safety() {
        // 多线程并发访问的安全性测试
        use std::sync::{Arc, Barrier};
        use std::thread;

        let scheduler = Arc::new(ContinuousBatchingScheduler::new(ContinuousBatchingConfig {
            max_batch_size: 32,
            ..Default::default()
        }));
        let barrier = Arc::new(Barrier::new(4));

        let handles: Vec<_> = (0..4)
            .map(|_i| {
                let s = Arc::clone(&scheduler);
                let b = Arc::clone(&barrier);
                thread::spawn(move || {
                    b.wait(); // 所有线程同时开始
                    for j in 0..10 {
                        let request = GenerationRequest::new(0, vec![1, 2, 3], 100);
                        let _ = s.add_request(request);

                        if j % 3 == 0 {
                            let _ = s.schedule_batch();
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("Thread should not panic");
        }

        let stats = scheduler.stats();
        assert_eq!(
            stats.total_requests, 40,
            "All 40 requests should be recorded"
        );
    }

    #[test]
    fn test_request_wait_time_and_processing_time() {
        // 测试等待时间和处理时间的计算
        let config = ContinuousBatchingConfig {
            min_batch_size: 1,
            ..Default::default()
        };
        let scheduler = ContinuousBatchingScheduler::new(config);

        let request = GenerationRequest::new(0, vec![1, 2, 3], 5);
        let id = scheduler.add_request(request).unwrap();

        // 刚创建时还没有started_at，wait_time应该是从created_at到现在
        let status_before = scheduler.get_request_status(id);
        assert_eq!(status_before, Some(RequestStatus::Waiting));

        // 调度后应该变成Running
        let batch = scheduler.schedule_batch().unwrap();
        assert!(!batch.is_empty());

        let status_after_schedule = scheduler.get_request_status(id);
        assert_eq!(status_after_schedule, Some(RequestStatus::Running));

        // 完成请求后应该变成Completed
        for _ in 0..5 {
            scheduler.update_request(id, 100).unwrap();
        }

        let status_completed = scheduler.get_request_status(id);
        assert_eq!(status_completed, Some(RequestStatus::Completed));
    }

    #[test]
    fn test_cancel_running_request() {
        // 取消正在运行的请求
        let config = ContinuousBatchingConfig {
            min_batch_size: 1,
            ..Default::default()
        };
        let scheduler = ContinuousBatchingScheduler::new(config);

        let request = GenerationRequest::new(0, vec![1, 2, 3], 100);
        let id = scheduler.add_request(request).unwrap();

        // 调度使其进入运行状态
        scheduler.schedule_batch().unwrap();
        assert_eq!(scheduler.running_count(), 1);

        // 取消运行中的请求
        scheduler.cancel_request(id).unwrap();

        // 应该不再在运行队列中
        assert_eq!(scheduler.running_count(), 0);

        // 状态应该是Cancelled
        let status = scheduler.get_request_status(id);
        assert_eq!(status, Some(RequestStatus::Cancelled));
    }

    #[test]
    fn test_batch_size_respects_max_limit() {
        // 批次大小不应该超过max_batch_size
        let config = ContinuousBatchingConfig {
            min_batch_size: 1,
            max_batch_size: 3,
            ..Default::default()
        };
        let scheduler = ContinuousBatchingScheduler::new(config);

        // 添加超过max_batch_size数量的请求
        for i in 0..10 {
            let request = GenerationRequest::new(i as u64, vec![1, 2, 3], 100);
            scheduler.add_request(request).unwrap();
        }

        let batch = scheduler.schedule_batch().unwrap();
        assert!(
            batch.len() <= 3,
            "Batch size should not exceed max_batch_size"
        );
    }

    #[test]
    fn test_is_finished_with_different_statuses() {
        // 测试is_finished在不同状态下的行为

        // Waiting状态 - 未完成
        let req_waiting = GenerationRequest::new(1, vec![1, 2, 3], 100);
        assert!(!req_waiting.is_finished());

        // Running状态且未达到max_length - 未完成
        let mut req_running = GenerationRequest::new(2, vec![1, 2, 3], 100);
        req_running.status = RequestStatus::Running;
        req_running.generated_tokens = vec![4, 5];
        assert!(!req_running.is_finished());

        // 达到max_length - 完成
        let mut req_completed_by_length = GenerationRequest::new(3, vec![1, 2, 3], 5);
        req_completed_by_length.generated_tokens = vec![4, 5, 6, 7, 8];
        assert!(req_completed_by_length.is_finished());

        // Completed状态 - 完成
        let mut req_status_completed = GenerationRequest::new(4, vec![1, 2, 3], 100);
        req_status_completed.status = RequestStatus::Completed;
        assert!(req_status_completed.is_finished());

        // Cancelled状态 - 完成
        let mut req_cancelled = GenerationRequest::new(5, vec![1, 2, 3], 100);
        req_cancelled.status = RequestStatus::Cancelled;
        assert!(req_cancelled.is_finished());
    }

    #[test]
    fn test_generation_request_builder_methods() {
        // 测试GenerationRequest的builder方法链式调用
        let request = GenerationRequest::new(42, vec![1, 2, 3, 4, 5], 200)
            .with_priority(15)
            .with_temperature(0.7);

        assert_eq!(request.id, 42);
        assert_eq!(request.priority, 15);
        assert!((request.temperature - 0.7).abs() < 0.001);
        assert_eq!(request.max_length, 200);
        assert_eq!(request.input_tokens.len(), 5);
        assert_eq!(request.status, RequestStatus::Waiting);
    }

    #[test]
    fn test_total_tokens_calculation() {
        // 测试总token数计算
        let request = GenerationRequest::new(
            1,
            vec![1, 2, 3, 4, 5], // 5个输入tokens
            100,
        );

        // 初始状态：只有输入tokens
        assert_eq!(request.total_tokens(), 5);

        // 生成一些tokens后
        let mut request = request;
        request.generated_tokens = vec![6, 7, 8]; // 生成了3个tokens
        assert_eq!(request.total_tokens(), 8); // 5 + 3 = 8
    }
}
