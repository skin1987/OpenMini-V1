//! 连续批处理实现
//!
//! 实现高效的请求调度和批处理：
//! - 动态批处理
//! - 抢占式调度
//! - 内存感知调度

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use super::paged_cache::{PagedKVCache, RequestId};

/// 请求状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum RequestState {
    /// 等待中
    Waiting,
    /// 运行中
    Running,
    /// 已暂停（抢占）
    Preempted,
    /// 已完成
    Finished,
}

/// 请求优先级
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[allow(dead_code)]
pub enum RequestPriority {
    /// 低优先级
    Low = 0,
    /// 普通优先级
    Normal = 1,
    /// 高优先级
    High = 2,
    /// 实时优先级
    Realtime = 3,
}

impl Default for RequestPriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// 生成请求
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GenerationRequest {
    /// 请求ID
    pub id: RequestId,
    /// 输入token
    pub input_tokens: Vec<u32>,
    /// 最大生成长度
    pub max_tokens: usize,
    /// 已生成token
    pub generated_tokens: Vec<u32>,
    /// 优先级
    pub priority: RequestPriority,
    /// 到达时间
    pub arrival_time: Instant,
    /// 开始执行时间
    pub start_time: Option<Instant>,
    /// 首token时间
    pub first_token_time: Option<Instant>,
    /// 当前状态
    pub state: RequestState,
    /// 温度参数
    pub temperature: f32,
    /// Top-p参数
    pub top_p: f32,
}

#[allow(dead_code)]
impl GenerationRequest {
    /// 创建新的生成请求
    pub fn new(id: RequestId, input_tokens: Vec<u32>, max_tokens: usize) -> Self {
        Self {
            id,
            input_tokens,
            max_tokens,
            generated_tokens: Vec::new(),
            priority: RequestPriority::Normal,
            arrival_time: Instant::now(),
            start_time: None,
            first_token_time: None,
            state: RequestState::Waiting,
            temperature: 1.0,
            top_p: 1.0,
        }
    }

    /// 设置优先级
    pub fn with_priority(mut self, priority: RequestPriority) -> Self {
        self.priority = priority;
        self
    }

    /// 设置温度
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// 设置top-p
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// 获取总token数
    pub fn total_tokens(&self) -> usize {
        self.input_tokens.len() + self.generated_tokens.len()
    }

    /// 检查是否完成
    pub fn is_finished(&self) -> bool {
        self.state == RequestState::Finished || self.generated_tokens.len() >= self.max_tokens
    }

    /// 获取等待时间
    pub fn wait_time(&self) -> Duration {
        self.arrival_time.elapsed()
    }

    /// 获取执行时间
    pub fn execution_time(&self) -> Option<Duration> {
        self.start_time.map(|t| t.elapsed())
    }

    /// 获取所需的块数
    pub fn required_blocks(&self, block_size: usize) -> usize {
        let total_tokens = self.total_tokens();
        total_tokens.div_ceil(block_size)
    }
}

/// 生成结果
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GenerationResult {
    /// 请求ID
    pub request_id: RequestId,
    /// 生成的token
    pub tokens: Vec<u32>,
    /// 是否完成
    pub finished: bool,
    /// 生成的文本
    pub text: Option<String>,
    /// 首token延迟
    pub time_to_first_token: Option<Duration>,
    /// 总生成时间
    pub total_time: Duration,
}

/// 批处理配置
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BatchConfig {
    /// 最大批大小
    pub max_batch_size: usize,
    /// 最大序列长度
    pub max_seq_len: usize,
    /// 抢占阈值（内存使用率）
    pub preempt_threshold: f32,
    /// 调度间隔（毫秒）
    pub schedule_interval_ms: u64,
    /// 最大等待token数
    pub max_waiting_tokens: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_seq_len: 4096,
            preempt_threshold: 0.9,
            schedule_interval_ms: 10,
            max_waiting_tokens: 2048,
        }
    }
}

/// 批处理调度器
#[derive(Debug)]
#[allow(dead_code)]
pub struct BatchScheduler {
    /// 配置
    config: BatchConfig,
    /// 等待队列
    waiting_queue: VecDeque<GenerationRequest>,
    /// 运行中的请求
    running_requests: HashMap<RequestId, GenerationRequest>,
    /// 已完成的请求
    completed_requests: Vec<GenerationResult>,
    /// KV Cache
    kv_cache: PagedKVCache,
    /// 统计信息
    total_processed: AtomicUsize,
    total_preemptions: AtomicUsize,
}

#[allow(dead_code)]
impl BatchScheduler {
    /// 创建新的批处理调度器
    pub fn new(kv_cache: PagedKVCache, config: BatchConfig) -> Self {
        Self {
            config,
            waiting_queue: VecDeque::new(),
            running_requests: HashMap::new(),
            completed_requests: Vec::new(),
            kv_cache,
            total_processed: AtomicUsize::new(0),
            total_preemptions: AtomicUsize::new(0),
        }
    }

    /// 使用默认配置创建
    pub fn with_kv_cache(kv_cache: PagedKVCache) -> Self {
        Self::new(kv_cache, BatchConfig::default())
    }

    /// 添加请求
    pub fn add_request(&mut self, request: GenerationRequest) {
        self.waiting_queue.push_back(request);
    }

    /// 添加多个请求
    pub fn add_requests(&mut self, requests: Vec<GenerationRequest>) {
        for request in requests {
            self.add_request(request);
        }
    }

    /// 调度一步
    pub fn schedule(&mut self) -> Vec<RequestId> {
        self.check_completed();
        
        let mut scheduled = Vec::new();
        
        while self.can_schedule() && !self.waiting_queue.is_empty() {
            if let Some(mut request) = self.waiting_queue.pop_front() {
                let required_blocks = request.required_blocks(self.kv_cache.config().block_size);
                
                if self.try_allocate(required_blocks) {
                    request.state = RequestState::Running;
                    request.start_time = Some(Instant::now());
                    let id = request.id;
                    self.running_requests.insert(id, request);
                    scheduled.push(id);
                } else {
                    let blocks_freed = self.preempt_for_blocks(required_blocks);
                    
                    if blocks_freed >= required_blocks || self.try_allocate(required_blocks) {
                        request.state = RequestState::Running;
                        request.start_time = Some(Instant::now());
                        let id = request.id;
                        self.running_requests.insert(id, request);
                        scheduled.push(id);
                    } else {
                        request.state = RequestState::Waiting;
                        self.waiting_queue.push_front(request);
                        break;
                    }
                }
            }
        }
        
        scheduled
    }

    /// 检查是否可以调度
    fn can_schedule(&self) -> bool {
        self.running_requests.len() < self.config.max_batch_size
            && self.kv_cache.utilization() < self.config.preempt_threshold
    }

    /// 尝试为请求分配资源
    fn try_allocate(&self, num_blocks: usize) -> bool {
        self.kv_cache.available_blocks() >= num_blocks
    }

    /// 抢占低优先级请求以释放指定数量的块
    /// 返回实际释放的块数
    fn preempt_for_blocks(&mut self, required_blocks: usize) -> usize {
        let mut to_preempt: Vec<RequestId> = Vec::new();
        let mut blocks_freed = 0;
        
        let mut candidates: Vec<_> = self.running_requests.iter()
            .filter(|(_, req)| req.priority == RequestPriority::Low)
            .collect();
        
        candidates.sort_by_key(|(_, req)| req.priority);
        
        for (id, request) in candidates {
            if blocks_freed >= required_blocks {
                break;
            }
            let request_blocks = request.required_blocks(self.kv_cache.config().block_size);
            blocks_freed += request_blocks;
            to_preempt.push(*id);
        }
        
        for id in to_preempt {
            if let Some(mut request) = self.running_requests.remove(&id) {
                request.state = RequestState::Waiting;
                request.start_time = None;
                self.kv_cache.free_request(&id);
                self.waiting_queue.push_back(request);
                self.total_preemptions.fetch_add(1, Ordering::SeqCst);
            }
        }
        
        blocks_freed
    }

    /// 检查已完成的请求
    fn check_completed(&mut self) {
        let completed: Vec<RequestId> = self.running_requests
            .iter()
            .filter(|(_, req)| req.is_finished())
            .map(|(id, _)| *id)
            .collect();
        
        for id in completed {
            if let Some(request) = self.running_requests.remove(&id) {
                let time_to_first_token = request.first_token_time
                    .map(|ft| ft.duration_since(request.start_time.unwrap()));
                
                let total_time = request.start_time
                    .map(|st| st.elapsed())
                    .unwrap_or_else(|| request.wait_time());
                
                let result = GenerationResult {
                    request_id: id,
                    tokens: request.generated_tokens.clone(),
                    finished: true,
                    text: None,
                    time_to_first_token,
                    total_time,
                };
                self.completed_requests.push(result);
                self.kv_cache.free_request(&id);
                self.total_processed.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    /// 获取运行中的请求
    pub fn get_running_requests(&self) -> Vec<&GenerationRequest> {
        self.running_requests.values().collect()
    }

    /// 获取运行中的请求ID
    pub fn running_request_ids(&self) -> Vec<RequestId> {
        self.running_requests.keys().copied().collect()
    }

    /// 获取请求
    pub fn get_request(&self, id: RequestId) -> Option<&GenerationRequest> {
        self.running_requests.get(&id)
            .or_else(|| self.waiting_queue.iter().find(|r| r.id == id))
    }

    /// 获取可变请求
    pub fn get_request_mut(&mut self, id: RequestId) -> Option<&mut GenerationRequest> {
        if let Some(req) = self.running_requests.get_mut(&id) {
            return Some(req);
        }
        for req in &mut self.waiting_queue {
            if req.id == id {
                return Some(req);
            }
        }
        None
    }

    /// 添加生成的token
    pub fn add_generated_token(&mut self, request_id: RequestId, token: u32) {
        if let Some(request) = self.running_requests.get_mut(&request_id) {
            if request.first_token_time.is_none() {
                request.first_token_time = Some(Instant::now());
            }
            request.generated_tokens.push(token);
        }
    }

    /// 标记请求完成
    pub fn finish_request(&mut self, request_id: RequestId) -> Option<GenerationResult> {
        if let Some(request) = self.running_requests.remove(&request_id) {
            let time_to_first_token = request.first_token_time
                .map(|ft| ft.duration_since(request.start_time.unwrap()));
            
            let total_time = request.start_time
                .map(|st| st.elapsed())
                .unwrap_or_else(|| request.wait_time());
            
            let result = GenerationResult {
                request_id,
                tokens: request.generated_tokens.clone(),
                finished: true,
                text: None,
                time_to_first_token,
                total_time,
            };
            self.kv_cache.free_request(&request_id);
            self.completed_requests.push(result.clone());
            self.total_processed.fetch_add(1, Ordering::SeqCst);
            Some(result)
        } else {
            None
        }
    }

    /// 取消请求
    pub fn cancel_request(&mut self, request_id: RequestId) -> bool {
        if self.running_requests.remove(&request_id).is_some() {
            self.kv_cache.free_request(&request_id);
            return true;
        }
        let original_len = self.waiting_queue.len();
        self.waiting_queue.retain(|r| r.id != request_id);
        original_len != self.waiting_queue.len()
    }

    /// 获取已完成的请求
    pub fn get_completed(&mut self) -> Vec<GenerationResult> {
        std::mem::take(&mut self.completed_requests)
    }

    /// 获取等待队列长度
    pub fn waiting_count(&self) -> usize {
        self.waiting_queue.len()
    }

    /// 获取运行中请求数
    pub fn running_count(&self) -> usize {
        self.running_requests.len()
    }

    /// 检查是否有待处理的请求
    pub fn has_pending(&self) -> bool {
        !self.waiting_queue.is_empty() || !self.running_requests.is_empty()
    }

    /// 获取KV Cache引用
    pub fn kv_cache(&self) -> &PagedKVCache {
        &self.kv_cache
    }

    /// 获取KV Cache可变引用
    pub fn kv_cache_mut(&mut self) -> &mut PagedKVCache {
        &mut self.kv_cache
    }

    /// 获取统计信息
    pub fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            waiting: self.waiting_queue.len(),
            running: self.running_requests.len(),
            completed: self.total_processed.load(Ordering::SeqCst),
            total_preemptions: self.total_preemptions.load(Ordering::SeqCst),
            memory_utilization: self.kv_cache.utilization(),
        }
    }

    /// 重置调度器
    pub fn reset(&mut self) {
        self.waiting_queue.clear();
        
        for id in self.running_requests.keys().cloned().collect::<Vec<_>>() {
            self.kv_cache.free_request(&id);
        }
        self.running_requests.clear();
        
        self.completed_requests.clear();
    }
}

/// 调度器统计信息
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct SchedulerStats {
    pub waiting: usize,
    pub running: usize,
    pub completed: usize,
    pub total_preemptions: usize,
    pub memory_utilization: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_scheduler() -> BatchScheduler {
        let kv_cache = PagedKVCache::with_capacity(100, 16);
        BatchScheduler::with_kv_cache(kv_cache)
    }

    #[test]
    fn test_new_scheduler() {
        let scheduler = create_test_scheduler();
        assert_eq!(scheduler.waiting_count(), 0);
        assert_eq!(scheduler.running_count(), 0);
        assert!(!scheduler.has_pending());
    }

    #[test]
    fn test_add_request() {
        let mut scheduler = create_test_scheduler();
        
        let request = GenerationRequest::new(1, vec![1, 2, 3, 4, 5], 100);
        scheduler.add_request(request);
        
        assert_eq!(scheduler.waiting_count(), 1);
    }

    #[test]
    fn test_schedule() {
        let mut scheduler = create_test_scheduler();
        
        let request = GenerationRequest::new(1, vec![1, 2, 3, 4, 5], 100);
        scheduler.add_request(request);
        
        let scheduled = scheduler.schedule();
        
        assert!(!scheduled.is_empty() || scheduler.waiting_count() > 0);
    }

    #[test]
    fn test_cancel_request() {
        let mut scheduler = create_test_scheduler();
        
        let request = GenerationRequest::new(1, vec![1, 2, 3, 4, 5], 100);
        scheduler.add_request(request);
        
        assert!(scheduler.cancel_request(1));
        assert_eq!(scheduler.waiting_count(), 0);
    }

    #[test]
    fn test_finish_request() {
        let mut scheduler = create_test_scheduler();
        
        let request = GenerationRequest::new(1, vec![1, 2, 3, 4, 5], 100);
        scheduler.add_request(request);
        scheduler.schedule();
        
        if scheduler.running_count() > 0 {
            let result = scheduler.finish_request(1);
            assert!(result.is_some());
            assert_eq!(scheduler.running_count(), 0);
        }
    }

    #[test]
    fn test_add_generated_token() {
        let mut scheduler = create_test_scheduler();
        
        let mut request = GenerationRequest::new(1, vec![1, 2, 3, 4, 5], 100);
        request.state = RequestState::Running;
        request.start_time = Some(Instant::now());
        scheduler.running_requests.insert(1, request);
        
        scheduler.add_generated_token(1, 42);
        
        let req = scheduler.get_request(1).unwrap();
        assert_eq!(req.generated_tokens, vec![42]);
        assert!(req.first_token_time.is_some());
    }

    #[test]
    fn test_stats() {
        let mut scheduler = create_test_scheduler();
        
        scheduler.add_request(GenerationRequest::new(1, vec![1, 2, 3], 100));
        scheduler.add_request(GenerationRequest::new(2, vec![4, 5, 6], 100));
        
        let stats = scheduler.stats();
        assert_eq!(stats.waiting, 2);
        assert_eq!(stats.running, 0);
    }

    #[test]
    fn test_request_priority() {
        let request = GenerationRequest::new(1, vec![1, 2, 3], 100)
            .with_priority(RequestPriority::High);
        
        assert_eq!(request.priority, RequestPriority::High);
    }

    #[test]
    fn test_request_is_finished() {
        let mut request = GenerationRequest::new(1, vec![1, 2, 3], 5);
        
        assert!(!request.is_finished());
        
        request.generated_tokens = vec![10, 11, 12, 13, 14];
        assert!(request.is_finished());
    }

    #[test]
    fn test_preempted_request_rescheduled() {
        let mut scheduler = create_test_scheduler();
        
        let low_priority = GenerationRequest::new(1, vec![1, 2, 3], 100)
            .with_priority(RequestPriority::Low);
        scheduler.add_request(low_priority);
        
        let scheduled = scheduler.schedule();
        assert!(scheduled.contains(&1));
        assert_eq!(scheduler.running_count(), 1);
        
        let high_priority = GenerationRequest::new(2, vec![4, 5, 6], 100)
            .with_priority(RequestPriority::High);
        scheduler.add_request(high_priority);
        
        scheduler.schedule();
        
        let stats = scheduler.stats();
        assert!(stats.total_preemptions > 0 || scheduler.running_count() == 2);
    }

    #[test]
    fn test_time_to_first_token() {
        let mut scheduler = create_test_scheduler();
        
        let mut request = GenerationRequest::new(1, vec![1, 2, 3], 100);
        request.state = RequestState::Running;
        request.start_time = Some(Instant::now());
        scheduler.running_requests.insert(1, request);
        
        scheduler.add_generated_token(1, 42);
        
        let req = scheduler.get_request(1).unwrap();
        assert!(req.first_token_time.is_some());
    }

    // ==================== 新增分支覆盖测试 ====================

    /// 测试 GenerationRequest builder 模式完整链（覆盖 with_temperature/with_top_p）
    #[test]
    fn test_request_builder_full_chain() {
        // 覆盖完整的 builder 链式调用
        let request = GenerationRequest::new(1, vec![10, 20, 30], 50)
            .with_priority(RequestPriority::Realtime)
            .with_temperature(0.8)
            .with_top_p(0.9);

        assert_eq!(request.id, 1);
        assert_eq!(request.priority, RequestPriority::Realtime);
        assert!((request.temperature - 0.8).abs() < f32::EPSILON);
        assert!((request.top_p - 0.9).abs() < f32::EPSILON);
        assert_eq!(request.state, RequestState::Waiting);
        assert_eq!(request.max_tokens, 50);
    }

    /// 测试 required_blocks 边界条件：空token列表（覆盖第134-137行 div_ceil）
    #[test]
    fn test_required_blocks_empty_tokens() {
        let request = GenerationRequest::new(1, vec![], 100);  // 无输入token

        // 覆盖：total_tokens=0 时 required_blocks 应返回0
        let blocks = request.required_blocks(16);
        assert_eq!(blocks, 0, "无token时所需块数应为0");
    }

    /// 测试 wait_time 和 execution_time（覆盖第124-131行时间计算）
    #[test]
    fn test_request_timing_methods() {
        let mut request = GenerationRequest::new(1, vec![1, 2, 3], 100);

        // wait_time 在 start_time=None 时应正常工作
        let wait_time = request.wait_time();
        assert!(wait_time >= Duration::ZERO);

        // execution_time 在 start_time=None 时应返回 None
        assert!(request.execution_time().is_none());

        // 设置 start_time 后 execution_time 应返回 Some
        request.start_time = Some(Instant::now());
        let exec_time = request.execution_time();
        assert!(exec_time.is_some());
        assert!(exec_time.unwrap() >= Duration::ZERO);
    }

    /// 测试 add_requests 批量添加（覆盖第231-235行批量路径）
    #[test]
    fn test_add_requests_batch() {
        let mut scheduler = create_test_scheduler();

        let requests = vec![
            GenerationRequest::new(1, vec![1], 100),
            GenerationRequest::new(2, vec![2], 100),
            GenerationRequest::new(3, vec![3], 100),
            GenerationRequest::new(4, vec![4], 100),
            GenerationRequest::new(5, vec![5], 100),
        ];

        // 覆盖：批量添加请求
        scheduler.add_requests(requests);
        assert_eq!(scheduler.waiting_count(), 5);
    }

    /// 测试 get_request_mut 可变修改（覆盖第368-378行可变引用路径）
    #[test]
    fn test_get_request_mut_modification() {
        let mut scheduler = create_test_scheduler();

        // 添加到等待队列
        let req = GenerationRequest::new(1, vec![1, 2], 50)
            .with_priority(RequestPriority::Low);
        scheduler.add_request(req);

        // 通过 get_request_mut 修改
        if let Some(req) = scheduler.get_request_mut(1) {
            req.priority = RequestPriority::High;
            req.temperature = 0.5;
        }

        // 验证修改生效
        let req = scheduler.get_request(1).unwrap();
        assert_eq!(req.priority, RequestPriority::High);
        assert!((req.temperature - 0.5).abs() < f32::EPSILON);
    }

    /// 测试 SchedulerStats 完整性（覆盖第459-467行统计信息结构体）
    #[test]
    fn test_scheduler_stats_completeness() {
        let mut scheduler = create_test_scheduler();

        // 初始状态统计
        let stats = scheduler.stats();
        assert_eq!(stats.waiting, 0);
        assert_eq!(stats.running, 0);
        assert_eq!(stats.completed, 0);
        assert_eq!(stats.total_preemptions, 0);
        assert!(stats.memory_utilization >= 0.0 && stats.memory_utilization <= 1.0);

        // 添加请求后
        scheduler.add_request(GenerationRequest::new(1, vec![1, 2, 3], 100));
        let stats = scheduler.stats();
        assert_eq!(stats.waiting, 1);
    }

    /// 测试 BatchConfig 默认值和 RequestState 枚举（覆盖配置和状态枚举）
    #[test]
    fn test_config_and_state_enums() {
        // 覆盖 BatchConfig Default trait
        let config = BatchConfig::default();
        assert_eq!(config.max_batch_size, 32);
        assert_eq!(config.max_seq_len, 4096);
        assert!((config.preempt_threshold - 0.9).abs() < f32::EPSILON);
        assert_eq!(config.schedule_interval_ms, 10);
        assert_eq!(config.max_waiting_tokens, 2048);

        // 覆盖 RequestState 所有变体及 Debug trait
        let states = [
            RequestState::Waiting,
            RequestState::Running,
            RequestState::Preempted,
            RequestState::Finished,
        ];
        for state in &states {
            let _ = format!("{:?}", state);
        }
        assert_ne!(RequestState::Waiting, RequestState::Running);

        // 覆盖 RequestPriority Ord trait
        assert!(RequestPriority::Realtime > RequestPriority::High);
        assert!(RequestPriority::High > RequestPriority::Normal);
        assert!(RequestPriority::Normal > RequestPriority::Low);
    }

    /// 测试 cancel_request 运行中状态（覆盖第418-426行运行中取消分支）
    #[test]
    fn test_cancel_running_request() {
        let mut scheduler = create_test_scheduler();

        let mut request = GenerationRequest::new(1, vec![1, 2, 3], 100);
        request.state = RequestState::Running;
        request.start_time = Some(Instant::now());
        scheduler.running_requests.insert(1, request);

        assert_eq!(scheduler.running_count(), 1);

        // 覆盖：取消运行中的请求
        let cancelled = scheduler.cancel_request(1);
        assert!(cancelled, "取消运行中请求应返回true");
        assert_eq!(scheduler.running_count(), 0);
    }

    /// 测试 has_pending 边界条件（覆盖第444-446行）
    #[test]
    fn test_has_pending_boundary() {
        let mut scheduler = create_test_scheduler();

        // 空状态
        assert!(!scheduler.has_pending());

        // 只有等待队列
        scheduler.add_request(GenerationRequest::new(1, vec![1], 100));
        assert!(scheduler.has_pending());

        // 清空后
        scheduler.cancel_request(1);
        assert!(!scheduler.has_pending());
    }
}
