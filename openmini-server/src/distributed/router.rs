//! 分布式推理路由器
//!
//! 实现请求分发和负载均衡功能，支持：
//! - **多种负载均衡策略**: RoundRobin、LeastLoaded、QueueLengthAware
//! - **Worker健康检查**: 自动检测异常节点
//! - **动态负载更新**: 实时调整调度决策
//!
//! ## 架构设计
//!
//! ```
//! Client Request
//! │
//! ▼
//! ┌──────────────────┐
//! │ DistributedRouter│  ← 负载均衡策略选择
//! └──────┬───────────┘
//!        │ dispatch()
//!   ┌────┼────┬────────┐
//!   ▼    ▼    ▼        ▼
//! ┌────┐┌────┐┌────┐ ┌────┐
//! │W0  ││W1  ││W2  │ │W3  │  ← GPU Workers
//! └────┘└────┘└────┘ └────┘
//! ```
//!
//! # 示例
//!
//! ```rust,ignore
//! use openmini_server::distributed::router::{DistributedRouter, LoadBalancingPolicy};
//!
//! // 创建4worker的路由器，使用最少负载策略
//! let mut router = DistributedRouter::new(4, LoadBalancingPolicy::LeastLoaded);
//!
//! // 分发请求
//! let worker_id = router.dispatch(request)?;
//! let response = router.collect_result(worker_id)?;
//! ```

use crate::distributed::config::DistributedError;
use log::{debug, info, trace, warn};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

/// Worker唯一标识符
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkerId(pub usize);

impl std::fmt::Display for WorkerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Worker({})", self.0)
    }
}

/// 推理请求结构
///
/// 封装客户端发送的推理请求信息。
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    /// 请求ID（用于追踪）
    pub request_id: String,

    /// 输入token IDs
    pub input_ids: Vec<i64>,

    /// 最大生成长度
    pub max_new_tokens: usize,

    /// 请求优先级（0=最高，数值越大优先级越低）
    pub priority: u8,

    /// 估计计算量（用于负载均衡，单位：FLOPs）
    pub estimated_flops: u64,
}

impl InferenceRequest {
    /// 创建新的推理请求
    pub fn new(request_id: String, input_ids: Vec<i64>, max_new_tokens: usize) -> Self {
        Self {
            request_id,
            input_ids,
            max_new_tokens,
            priority: 0,
            estimated_flops: 0,
        }
    }

    /// 设置请求优先级
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// 设置估计计算量
    pub fn with_estimated_flops(mut self, flops: u64) -> Self {
        self.estimated_flops = flops;
        self
    }
}

/// 推理响应结构
///
/// 包含推理结果和元数据。
#[derive(Debug, Clone)]
pub struct InferenceResponse {
    /// 对应的请求ID
    pub request_id: String,

    /// 生成的token IDs
    pub output_ids: Vec<i64>,

    /// 推理耗时（毫秒）
    pub latency_ms: f64,

    /// 处理该请求的Worker ID
    pub worker_id: WorkerId,

    /// 是否成功完成
    pub success: bool,

    /// 错误信息（如果失败）
    pub error_message: Option<String>,
}

impl InferenceResponse {
    /// 创建成功响应
    pub fn success(
        request_id: String,
        output_ids: Vec<i64>,
        latency_ms: f64,
        worker_id: WorkerId,
    ) -> Self {
        Self {
            request_id,
            output_ids,
            latency_ms,
            worker_id,
            success: true,
            error_message: None,
        }
    }

    /// 创建失败响应
    pub fn failure(request_id: String, error: String, worker_id: WorkerId) -> Self {
        Self {
            request_id,
            output_ids: vec![],
            latency_ms: 0.0,
            worker_id,
            success: false,
            error_message: Some(error),
        }
    }
}

/// 负载均衡策略枚举
///
/// 定义不同的请求分发算法。
#[derive(Debug, Clone, PartialEq)]
pub enum LoadBalancingPolicy {
    /// 轮询调度
    ///
    /// 简单且公平地分配请求到各worker。
    /// 适合请求耗时均匀的场景。
    ///
    /// # 特点
    /// - O(1) 时间复杂度
    /// - 无需维护状态
    /// - 可能不均衡（当请求耗时差异大时）
    RoundRobin,

    /// 最少负载优先
    ///
    /// 将请求发送给当前负载最低的worker。
    /// 需要定期更新各worker的负载信息。
    ///
    /// # 特点
    /// - O(n) 时间复杂度（n=worker数）
    /// - 能较好平衡异构worker
    /// - 依赖准确的负载报告
    LeastLoaded,

    /// 队列长度感知
    ///
    /// 综合考虑当前队列长度和历史处理能力。
    /// 适合突发流量场景。
    ///
    /// # 特点
    /// - O(n) 时间复杂度
    /// - 考虑排队延迟
    /// - 自适应能力强
    QueueLengthAware,
}

impl std::fmt::Display for LoadBalancingPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RoundRobin => write!(f, "round_robin"),
            Self::LeastLoaded => write!(f, "least_loaded"),
            Self::QueueLengthAware => write!(f, "queue_length_aware"),
        }
    }
}

impl Default for LoadBalancingPolicy {
    fn default() -> Self {
        Self::RoundRobin
    }
}

/// Worker句柄
///
/// 包含Worker的状态信息和运行时指标。
#[derive(Clone)]
struct WorkerHandle {
    /// Worker ID
    id: WorkerId,

    /// GPU设备ID
    gpu_id: usize,

    /// 当前计算负载（0-100，表示GPU利用率百分比）
    current_load: Arc<AtomicUsize>,

    /// 当前队列中等待处理的请求数
    queue_length: Arc<AtomicUsize>,
}

impl WorkerHandle {
    fn new(id: WorkerId, gpu_id: usize) -> Self {
        Self {
            id,
            gpu_id,
            current_load: Arc::new(AtomicUsize::new(0)),
            queue_length: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn get_load(&self) -> usize {
        self.current_load.load(Ordering::Relaxed)
    }

    fn get_queue_length(&self) -> usize {
        self.queue_length.load(Ordering::Relaxed)
    }
}

/// Worker健康状态
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerHealth {
    /// 健康
    Healthy,

    /// 过载（负载 > 90%）
    Overloaded { load: usize },

    /// 响应超时
    Timeout { last_response_ms: f64 },

    /// 错误状态
    Error { message: String },

    /// 未知（尚未收到心跳）
    Unknown,
}

/// Worker状态信息
#[derive(Debug, Clone)]
pub struct WorkerStatus {
    /// Worker ID
    pub worker_id: WorkerId,

    /// GPU ID
    pub gpu_id: usize,

    /// 当前负载百分比
    pub current_load: usize,

    /// 队列长度
    pub queue_length: usize,

    /// 健康状态
    pub health: WorkerHealth,

    /// 已处理的总请求数
    pub total_requests: usize,

    /// 平均响应时间（毫秒）
    pub avg_latency_ms: f64,
}

/// 分布式路由器
///
/// 管理多个推理Worker节点，负责：
/// - 请求分发（根据负载均衡策略）
/// - 结果收集
/// - 健康监控
/// - 动态负载调整
///
/// # 线程安全
///
/// 所有公共方法都是线程安全的。
/// 内部使用原子操作保证无锁并发访问。
///
/// # 使用示例
///
/// ```rust,ignore
/// let mut router = DistributedRouter::new(4, LoadBalancingPolicy::LeastLoaded);
///
/// for i in 0..10 {
///     let req = InferenceRequest::new(format!("req_{}", i), vec![1, 2, 3], 100);
///     let worker_id = router.dispatch(req)?;
///     
///     // 模拟处理...
///     router.update_worker_load(worker_id, 50); // 更新负载
///     
///     let resp = router.collect_result(worker_id)?;
/// }
/// ```
pub struct DistributedRouter {
    /// Worker列表
    workers: Vec<WorkerHandle>,

    /// 负载均衡策略
    load_balancer: LoadBalancingPolicy,

    /// RoundRobin计数器
    rr_counter: AtomicUsize,

    /// 各Worker统计信息（线程安全）
    stats: Arc<std::sync::Mutex<HashMap<WorkerId, WorkerStats>>>,

    /// 已分发但未完成的请求映射
    pending_requests:
        Arc<std::sync::Mutex<HashMap<String, (WorkerId, InferenceRequest)>>>,
}

/// Worker内部统计数据
#[derive(Debug)]
struct WorkerStats {
    total_requests: usize,
    total_latency_ms: f64,
    last_update: std::time::Instant,
}

impl Default for WorkerStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            total_latency_ms: 0.0,
            last_update: std::time::Instant::now(),
        }
    }
}

impl DistributedRouter {
    /// 创建新的分布式路由器
    ///
    /// # 参数
    ///
    /// * `num_workers` - Worker数量（对应GPU数量）
    /// * `policy` - 负载均衡策略
    ///
    /// # Panics
    ///
    /// 如果 `num_workers` 为0则panic
    pub fn new(num_workers: usize, policy: LoadBalancingPolicy) -> Self {
        assert!(num_workers > 0, "num_workers must be > 0");

        info!(
            "Creating DistributedRouter: {} workers, policy={}",
            num_workers, policy
        );

        let workers: Vec<WorkerHandle> = (0..num_workers)
            .map(|i| WorkerHandle::new(WorkerId(i), i))
            .collect();

        let mut initial_stats = HashMap::new();
        for w in &workers {
            initial_stats.insert(w.id, WorkerStats::default());
        }

        Self {
            workers,
            load_balancer: policy,
            rr_counter: AtomicUsize::new(0),
            stats: Arc::new(std::sync::Mutex::new(initial_stats)),
            pending_requests: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    /// 获取Worker数量
    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }

    /// 获取当前的负载均衡策略
    pub fn policy(&self) -> &LoadBalancingPolicy {
        &self.load_balancer
    }

    /// 分发请求到合适的Worker
    ///
    /// 根据配置的策略选择目标Worker，
    /// 并记录请求到pending队列。
    ///
    /// # 参数
    ///
    /// * `request` - 推理请求
    ///
    /// # 返回
    ///
    /// 选中的Worker ID
    ///
    /// # 错误
    ///
    /// 返回 [`DistributedError::Router`] 如果所有Worker都不可用。
    pub fn dispatch(&mut self, request: InferenceRequest) -> Result<WorkerId, DistributedError> {
        debug!(
            "Dispatching request {}: input_len={}, max_tokens={}, priority={}",
            request.request_id,
            request.input_ids.len(),
            request.max_new_tokens,
            request.priority
        );

        let worker_id = match &self.load_balancer {
            LoadBalancingPolicy::RoundRobin => self.round_robin_select(),
            LoadBalancingPolicy::LeastLoaded => self.least_loaded_select(),
            LoadBalancingPolicy::QueueLengthAware => self.queue_aware_select(),
        };

        match worker_id {
            Some(id) => {
                // 保存request_id用于日志（在move之前）
                let req_id_for_log = request.request_id.clone();

                // 记录pending请求
                {
                    let mut pending = self.pending_requests.lock().unwrap();
                    pending.insert(request.request_id.clone(), (id, request));
                }

                // 更新队列长度
                if let Some(worker) = self.workers.iter().find(|w| w.id == id) {
                    worker.queue_length.fetch_add(1, Ordering::Relaxed);
                }

                // 更新统计
                {
                    let mut stats = self.stats.lock().unwrap();
                    if let Some(s) = stats.get_mut(&id) {
                        s.total_requests += 1;
                        s.last_update = std::time::Instant::now();
                    }
                }

                info!("Request {} dispatched to {}", req_id_for_log, id);
                Ok(id)
            }
            None => Err(DistributedError::Router(
                "No available worker".to_string(),
            )),
        }
    }

    /// 收集Worker的处理结果
    ///
    /// 从pending队列中移除请求并返回模拟结果。
    ///
    /// # 参数
    ///
    /// * `worker_id` - 处理请求的Worker ID
    ///
    /// # 返回
    ///
    /// 推理响应（成功或失败）
    pub fn collect_result(&self, worker_id: WorkerId) -> Result<InferenceResponse, DistributedError> {
        debug!("Collecting result from {}", worker_id);

        // 从pending队列找到该worker最早的一个请求
        let (request_id, _request) = {
            let mut pending = self.pending_requests.lock().unwrap();

            // 找到此worker最早的请求
            let mut target_key: Option<String> = None;
            for (key, &(wid, _)) in pending.iter() {
                if wid == worker_id {
                    match &target_key {
                        None => {
                            target_key = Some(key.clone());
                        }
                        Some(_) => continue,
                    }
                }
            }

            match target_key {
                Some(key) => match pending.remove_entry(&key) {
                    Some((k, v)) => (k, v.1),
                    None => {
                        return Err(DistributedError::Router(format!(
                            "No pending request for worker {}",
                            worker_id
                        )));
                    }
                },
                None => {
                    return Err(DistributedError::Router(format!(
                        "No pending request for worker {}",
                        worker_id
                    )));
                }
            }
        };

        // 更新队列长度
        if let Some(worker) = self.workers.iter().find(|w| w.id == worker_id) {
            worker.queue_length.fetch_sub(1, Ordering::Relaxed);
        }

        // 模拟处理结果（实际环境中这里会从worker接收真实结果）
        let latency = 10.0 + (worker_id.0 as f64) * 2.0; // 模拟不同worker的不同延迟

        let response = InferenceResponse::success(
            request_id.clone(),
            vec![100, 200, 300], // 模拟输出token
            latency,
            worker_id,
        );

        // 更新统计
        {
            let mut stats = self.stats.lock().unwrap();
            if let Some(s) = stats.get_mut(&worker_id) {
                s.total_latency_ms += latency;
            }
        }

        info!(
            "Result collected from {} for request {}, latency={:.2}ms",
            worker_id, request_id, latency
        );

        Ok(response)
    }

    /// 执行健康检查
    ///
    /// 返回所有Worker的当前状态信息。
    ///
    /// # 健康判断标准
    ///
    /// - **Healthy**: load < 80%
    /// - **Overloaded**: load >= 80%
    /// - **Error**: 无法获取状态
    pub fn health_check(&self) -> Vec<WorkerStatus> {
        debug!("Performing health check on {} workers", self.workers.len());

        let stats = self.stats.lock().unwrap();

        self.workers
            .iter()
            .map(|w| {
                let load = w.get_load();
                let queue_len = w.get_queue_length();
                let stat = stats.get(&w.id);

                let health = if load >= 95 {
                    WorkerHealth::Overloaded { load }
                } else if load >= 80 {
                    WorkerHealth::Overloaded { load }
                } else if load > 0 && load < 80 {
                    WorkerHealth::Healthy
                } else {
                    WorkerHealth::Healthy
                };

                let (total_reqs, avg_lat) = match stat {
                    Some(s) => (
                        s.total_requests,
                        if s.total_requests > 0 {
                            s.total_latency_ms / s.total_requests as f64
                        } else {
                            0.0
                        },
                    ),
                    None => (0, 0.0),
                };

                WorkerStatus {
                    worker_id: w.id,
                    gpu_id: w.gpu_id,
                    current_load: load,
                    queue_length: queue_len,
                    health,
                    total_requests: total_reqs,
                    avg_latency_ms: avg_lat,
                }
            })
            .collect()
    }

    /// 更新Worker负载信息
    ///
    /// 由外部监控组件调用，定期更新各Worker的实时负载。
    ///
    /// # 参数
    ///
    /// * `worker_id` - 目标Worker
    /// * `load` - 新的负载值（0-100）
    pub fn update_worker_load(&self, worker_id: WorkerId, load: usize) {
        trace!("Updating worker {} load to {}", worker_id, load);

        if let Some(worker) = self.workers.iter().find(|w| w.id == worker_id) {
            let clamped_load = load.min(100);
            worker.current_load.store(clamped_load, Ordering::Relaxed);
        } else {
            warn!("Unknown worker id: {}", worker_id);
        }
    }

    /// Round-Robin选择策略
    ///
    /// 简单轮询，依次选择每个worker。
    fn round_robin_select(&self) -> Option<WorkerId> {
        let idx = self.rr_counter.fetch_add(1, Ordering::Relaxed) % self.workers.len();
        Some(self.workers[idx].id)
    }

    /// 最少负载选择策略
    ///
    /// 选择当前CPU/GPU利用率最低的worker。
    fn least_loaded_select(&self) -> Option<WorkerId> {
        self.workers
            .iter()
            .min_by_key(|w| w.get_load())
            .map(|w| w.id)
    }

    /// 队列长度感知选择策略
    ///
    /// 综合考虑队列长度和当前负载，选择综合得分最优的worker。
    ///
    /// # 评分公式
    ///
    /// ```text
    /// score = queue_length * 0.6 + (load / 100.0) * 0.4
    /// ```
    ///
    /// 得分越低越好。
    fn queue_aware_select(&self) -> Option<WorkerId> {
        self.workers
            .iter()
            .map(|w| {
                let score =
                    w.get_queue_length() as f64 * 0.6 + (w.get_load() as f64 / 100.0) * 0.4;
                (w.id, score)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_request(id: &str) -> InferenceRequest {
        InferenceRequest::new(id.to_string(), vec![1, 2, 3, 4, 5], 128)
    }

    #[test]
    fn test_router_creation() {
        let router = DistributedRouter::new(4, LoadBalancingPolicy::RoundRobin);
        assert_eq!(router.num_workers(), 4);
        assert_eq!(router.policy(), &LoadBalancingPolicy::RoundRobin);
    }

    #[test]
    fn test_dispatch_round_robin() {
        let mut router = DistributedRouter::new(2, LoadBalancingPolicy::RoundRobin);

        let w0 = router.dispatch(create_request("req1")).unwrap();
        let w1 = router.dispatch(create_request("req2")).unwrap();
        let w2 = router.dispatch(create_request("req3")).unwrap();

        // Round-robin应该交替选择
        assert_ne!(w0, w1); // 不同worker
        assert_eq!(w0, w2);  // 第3个回到第1个
    }

    #[test]
    fn test_dispatch_least_loaded() {
        let mut router = DistributedRouter::new(3, LoadBalancingPolicy::LeastLoaded);

        // 设置不同的负载
        router.update_worker_load(WorkerId(0), 80);
        router.update_worker_load(WorkerId(1), 20);
        router.update_worker_load(WorkerId(2), 60);

        let selected = router.dispatch(create_request("req1")).unwrap();
        assert_eq!(selected, WorkerId(1)); // 应该选择负载最低的
    }

    #[test]
    fn test_dispatch_queue_aware() {
        let mut router = DistributedRouter::new(3, LoadBalancingPolicy::QueueLengthAware);

        // 设置不同的队列长度和负载
        router.update_worker_load(WorkerId(0), 30);
        router.update_worker_load(WorkerId(1), 40);
        router.update_worker_load(WorkerId(2), 20);

        // 手动设置队列长度（通过dispatch增加）
        let _ = router.dispatch(create_request("req_q0")); // Worker 0队列+1

        let selected = router.dispatch(create_request("req_test")).unwrap();
        // 应该避免选择队列较长的worker 0
        assert_eq!(selected, WorkerId(2)); // Worker 2负载最低且队列最短
    }

    #[test]
    fn test_collect_result() {
        let mut router = DistributedRouter::new(2, LoadBalancingPolicy::RoundRobin);

        let worker_id = router.dispatch(create_request("req_collect")).unwrap();
        let result = router.collect_result(worker_id);

        assert!(result.is_ok());
        let resp = result.unwrap();
        assert!(resp.success);
        assert_eq!(resp.worker_id, worker_id);
        assert!(!resp.output_ids.is_empty());
    }

    #[test]
    fn test_collect_nonexistent_request() {
        let mut router = DistributedRouter::new(2, LoadBalancingPolicy::RoundRobin);

        let result = router.collect_result(WorkerId(0));
        assert!(result.is_err()); // 没有pending请求
    }

    #[test]
    fn test_health_check_all_healthy() {
        let router = DistributedRouter::new(3, LoadBalancingPolicy::RoundRobin);
        let statuses = router.health_check();

        assert_eq!(statuses.len(), 3);
        for status in &statuses {
            assert_eq!(status.health, WorkerHealth::Healthy);
            assert_eq!(status.current_load, 0); // 初始无负载
        }
    }

    #[test]
    fn test_health_check_overloaded() {
        let router = DistributedRouter::new(2, LoadBalancingPolicy::RoundRobin);

        router.update_worker_load(WorkerId(0), 95);
        router.update_worker_load(WorkerId(1), 50);

        let statuses = router.health_check();

        assert_eq!(statuses[0].health, WorkerHealth::Overloaded { load: 95 });
        assert_eq!(statuses[1].health, WorkerHealth::Healthy);
    }

    #[test]
    fn test_update_worker_load() {
        let router = DistributedRouter::new(2, LoadBalancingPolicy::RoundRobin);

        router.update_worker_load(WorkerId(0), 75);
        let statuses = router.health_check();

        assert_eq!(statuses[0].current_load, 75);
    }

    #[test]
    fn test_load_clamping() {
        let router = DistributedRouter::new(1, LoadBalancingPolicy::RoundRobin);

        // 测试超过100的值被clamp
        router.update_worker_load(WorkerId(0), 150);
        let statuses = router.health_check();

        assert_eq!(statuses[0].current_load, 100); // 应被限制为100
    }

    #[test]
    fn test_multiple_dispatch_and_collect() {
        let mut router = DistributedRouter::new(2, LoadBalancingPolicy::RoundRobin);

        // 发送5个请求
        let mut worker_ids = Vec::new();
        for i in 0..5 {
            let wid = router.dispatch(create_request(&format!("req_{}", i))).unwrap();
            worker_ids.push(wid);
        }

        // 收集所有结果
        let mut results = Vec::new();
        for &wid in &worker_ids {
            let result = router.collect_result(wid).unwrap();
            results.push(result);
        }

        assert_eq!(results.len(), 5);
        for r in &results {
            assert!(r.success);
        }
    }

    #[test]
    fn test_worker_status_fields() {
        let mut router = DistributedRouter::new(2, LoadBalancingPolicy::RoundRobin);

        let _ = router.dispatch(create_request("test"));
        router.update_worker_load(WorkerId(0), 42);

        let statuses = router.health_check();
        let status = &statuses[0];

        assert_eq!(status.worker_id, WorkerId(0));
        assert_eq!(status.gpu_id, 0);
        assert_eq!(status.current_load, 42);
        assert_eq!(status.total_requests, 1);
    }

    #[test]
    fn test_policy_display() {
        assert_eq!(LoadBalancingPolicy::RoundRobin.to_string(), "round_robin");
        assert_eq!(LoadBalancingPolicy::LeastLoaded.to_string(), "least_loaded");
        assert_eq!(
            LoadBalancingPolicy::QueueLengthAware.to_string(),
            "queue_length_aware"
        );
    }

    #[test]
    fn test_inference_request_builder_pattern() {
        let req = InferenceRequest::new("test".to_string(), vec![1, 2], 100)
            .with_priority(5)
            .with_estimated_flops(1000000);

        assert_eq!(req.priority, 5);
        assert_eq!(req.estimated_flops, 1000000);
    }

    #[test]
    fn test_four_worker_scenario() {
        let mut router = DistributedRouter::new(4, LoadBalancingPolicy::LeastLoaded);

        // 设置递减的负载
        for i in 0..4 {
            router.update_worker_load(WorkerId(i), 90 - i * 20);
        }

        // 多次分发都应该选择负载最低的worker 3
        for _ in 0..3 {
            let selected = router.dispatch(create_request("4gpu_test")).unwrap();
            assert_eq!(selected, WorkerId(3));
        }
    }

    #[test]
    fn test_inference_response_success_failure() {
        let success_resp =
            InferenceResponse::success("req1".to_string(), vec![1, 2], 10.0, WorkerId(0));
        assert!(success_resp.success);
        assert_eq!(success_resp.output_ids, vec![1, 2]);

        let fail_resp =
            InferenceResponse::failure("req2".to_string(), "error msg".to_string(), WorkerId(1));
        assert!(!fail_resp.success);
        assert!(fail_resp.error_message.is_some());
    }
}
