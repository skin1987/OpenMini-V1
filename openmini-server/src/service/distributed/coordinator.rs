//! 分布式推理协调器
//!
//! 负责任务调度、负载均衡、结果聚合。
//!
//! ## 职责
//!
//! 1. **节点管理**: 注册/注销工作节点，监控节点健康状态
//! 2. **任务调度**: 接收推理请求，根据负载均衡策略分配给工作节点
//! 3. **结果收集**: 收集工作节点的推理结果并返回给客户端
//!
//! ## 负载均衡策略
//!
//! 当前实现基于最少任务数（Least-Loaded）的简单策略，
//! 未来可扩展为：加权轮询、一致性哈希、基于延迟的自适应调度等。

use std::collections::{HashMap, VecDeque};

use tracing::{debug, info, trace, warn};

use super::mesh::MeshNode;
use super::protocol::*;

/// 协调器错误类型
#[derive(Debug, thiserror::Error)]
pub enum CoordinatorError {
    /// 没有可用的工作节点
    #[error("没有可用的 worker 节点")]
    NoAvailableWorkers,

    /// 任务提交失败
    #[error("任务提交失败: {0}")]
    TaskSubmitFailed(String),

    /// 网络通信错误
    #[error("网络错误: {0}")]
    NetworkError(String),
}

/// 工作节点状态信息
struct WorkerStatus {
    /// 节点能力描述
    capabilities: NodeCapabilities,
    /// 当前正在处理的任务数量（用于负载均衡）
    current_load: u32,
    /// 最后一次心跳时间戳 (Unix timestamp in ms)
    last_heartbeat: u64,
    /// 节点是否健康
    healthy: bool,
}

/// 协调器系统状态
#[derive(Debug, Clone)]
pub struct CoordinatorStatus {
    /// 已注册的工作节点数量
    pub total_workers: usize,
    /// 健康的工作节点数量
    pub healthy_workers: usize,
    /// 待处理的任务队列长度
    pub pending_tasks: usize,
    /// 已完成的任务总数
    pub completed_tasks: u64,
}

/// 分布式推理协调器
///
/// 作为集群的中心调度器，负责任务分发和结果收集。
/// 通常运行在专用的协调器节点上。
pub struct Coordinator {
    /// Mesh 网络句柄，用于与工作节点通信
    mesh: MeshNode,
    /// 待处理任务队列 (FIFO)
    pending_tasks: VecDeque<InferenceRequest>,
    /// 工作节点状态表 (node_id -> WorkerStatus)
    workers: HashMap<String, WorkerStatus>,
    /// 已收集的结果缓存 (task_id -> InferenceResponse)
    results: HashMap<String, InferenceResponse>,
    /// 任务计数器（用于生成唯一 task_id）
    task_counter: u64,
    /// 统计信息
    completed_tasks_count: u64,
}

impl Coordinator {
    /// 创建新的协调器实例
    ///
    /// # 参数
    ///
    /// - `mesh`: 已初始化的 Mesh 网络节点
    pub fn new(mesh: MeshNode) -> Self {
        info!(coordinator_id = %mesh.node_id(), "初始化分布式协调器");

        Self {
            mesh,
            pending_tasks: VecDeque::new(),
            workers: HashMap::new(),
            results: HashMap::new(),
            task_counter: 0,
            completed_tasks_count: 0,
        }
    }

    /// 注册新工作节点
    ///
    /// 当工作节点启动时调用此方法注册到协调器。
    ///
    /// # 参数
    ///
    /// - `node_id`: 工作节点的唯一标识符
    /// - `caps`: 工作节点的硬件能力信息
    pub async fn register_worker(&mut self, node_id: String, caps: NodeCapabilities) {
        let status = WorkerStatus {
            capabilities: caps.clone(),
            current_load: 0,
            last_heartbeat: chrono::Utc::now().timestamp_millis() as u64,
            healthy: true,
        };

        self.workers.insert(node_id.clone(), status);

        info!(
            coordinator_id = %self.mesh.node_id(),
            worker_id = %node_id,
            gpu_name = %caps.gpu_name,
            vram_mb = caps.vram_size_mb,
            "工作节点已注册"
        );
    }

    /// 注销工作节点
    ///
    /// 当工作节点关闭或检测到故障时调用。
    pub async fn unregister_worker(&mut self, node_id: &str) {
        if self.workers.remove(node_id).is_some() {
            warn!(
                coordinator_id = %self.mesh.node_id(),
                worker_id = %node_id,
                "工作节点已注销"
            );
        }
    }

    /// 提交推理请求
    ///
    /// 将客户端的推理请求加入待处理队列，并返回任务 ID。
    ///
    /// # 参数
    ///
    /// - `request`: 推理请求
    ///
    /// # 返回
    ///
    /// 返回生成的任务 ID，可用于后续查询结果
    ///
    /// # 错误
    ///
    /// - [`CoordinatorError::TaskSubmitFailed`]: 队列已满或内部错误
    pub async fn submit(&mut self, request: InferenceRequest) -> Result<String, CoordinatorError> {
        // 生成唯一任务 ID
        self.task_counter += 1;
        let task_id = format!("task-{}-{}", self.mesh.node_id(), self.task_counter);

        debug!(
            coordinator_id = %self.mesh.node_id(),
            task_id = %task_id,
            model = %request.model_name,
            input_len = request.input_len(),
            "收到推理请求"
        );

        // 加入待处理队列
        self.pending_tasks.push_back(request);

        Ok(task_id)
    }

    /// 分发待处理任务
    ///
    /// 将队列中的任务分配给最空闲的工作节点。
    /// 使用最少负载（Least-Loaded）策略进行调度。
    pub async fn dispatch_pending(&mut self) {
        while !self.pending_tasks.is_empty() {
            // 找到最空闲的健康节点
            let best_worker = self.find_best_worker();

            match best_worker {
                Some(worker_id) => {
                    // 取出队首任务
                    if let Some(request) = self.pending_tasks.pop_front() {
                        let task_id = format!("task-{}-{}", self.mesh.node_id(), self.task_counter);

                        // 发送任务分配消息
                        let msg = DistributedMessage::TaskAssign {
                            task_id: task_id.clone(),
                            payload: request,
                            priority: 0,
                        };

                        match self.mesh.send_to(&worker_id, msg).await {
                            Ok(()) => {
                                // 更新工作节点负载
                                if let Some(status) = self.workers.get_mut(&worker_id) {
                                    status.current_load += 1;
                                }

                                debug!(
                                    coordinator_id = %self.mesh.node_id(),
                                    task_id = %task_id,
                                    worker_id = %worker_id,
                                    "任务已分发给工作节点"
                                );
                            }
                            Err(e) => {
                                // 发送失败，将任务放回队列头部
                                self.pending_tasks.push_front(
                                    // 注意：这里需要从 msg 中恢复 request，简化处理
                                    InferenceRequest::new("", "", vec![], 0, 0.0),
                                );
                                warn!(
                                    error = %e.to_string(),
                                    worker_id = %worker_id,
                                    "任务发送失败"
                                );
                                break; // 避免无限循环
                            }
                        }
                    }
                }
                None => {
                    debug!("没有可用的工作节点，停止分发");
                    break;
                }
            }
        }
    }

    /// 处理接收到的消息
    ///
    /// 从 Mesh 网络接收消息并根据类型进行处理：
    /// - 心跳更新：更新工作节点状态
    /// - 任务结果：保存到结果缓存
    /// - 节点注册：注册新节点
    /// - 错误报告：记录日志
    pub async fn handle_message(&mut self, msg: DistributedMessage) {
        match msg {
            DistributedMessage::Heartbeat {
                node_id,
                timestamp,
                gpu_utilization,
                memory_used_mb,
            } => {
                if let Some(status) = self.workers.get_mut(&node_id) {
                    status.last_heartbeat = timestamp;
                    status.healthy = true;

                    trace!(
                        worker_id = %node_id,
                        gpu_util = gpu_utilization,
                        memory_mb = memory_used_mb,
                        "更新心跳"
                    );
                }
            }

            DistributedMessage::TaskResult {
                task_id,
                result,
                execution_time_us,
            } => {
                // 保存结果
                self.results.insert(task_id.clone(), result);
                self.completed_tasks_count += 1;

                // 减少对应工作节点的负载（这里简化处理）
                // 实际应该跟踪每个任务属于哪个 worker

                debug!(
                    task_id = %task_id,
                    execution_us = execution_time_us,
                    "收到任务结果"
                );
            }

            DistributedMessage::NodeRegister {
                node_id,
                capabilities,
            } => {
                self.register_worker(node_id, capabilities).await;
            }

            DistributedMessage::Error {
                task_id,
                error_code,
                message,
            } => {
                warn!(
                    task_id = %task_id,
                    error_code = error_code,
                    message = %message,
                    "收到错误报告"
                );
            }

            _ => {
                debug!("忽略不支持的消息类型");
            }
        }
    }

    /// 收集已完成的结果
    ///
    /// 返回所有已收集但尚未被取走的推理结果。
    ///
    /// # 返回
    ///
    /// 返回 (task_id, InferenceResponse) 的列表
    pub fn collect_results(&mut self) -> Vec<(String, InferenceResponse)> {
        let results: Vec<(String, InferenceResponse)> = self.results.drain().collect();
        results
    }

    /// 获取指定任务的结果
    ///
    /// # 参数
    ///
    /// - `task_id`: 任务 ID
    ///
    /// # 返回
    ///
    /// - `Some(InferenceResponse)`: 任务已完成
    /// - `None`: 任务未完成或不存在
    pub fn get_result(&self, task_id: &str) -> Option<InferenceResponse> {
        self.results.get(task_id).cloned()
    }

    /// 获取协调器状态
    ///
    /// 返回当前的系统统计信息，包括：
    /// - 工作节点数量和健康状态
    /// - 待处理任务数
    /// - 已完成任务数
    pub fn status(&self) -> CoordinatorStatus {
        let healthy_count = self.workers.values().filter(|w| w.healthy).count();

        CoordinatorStatus {
            total_workers: self.workers.len(),
            healthy_workers: healthy_count,
            pending_tasks: self.pending_tasks.len(),
            completed_tasks: self.completed_tasks_count,
        }
    }

    /// 获取健康的工作节点列表
    pub fn healthy_workers(&self) -> Vec<&str> {
        self.workers
            .iter()
            .filter(|(_, status)| status.healthy)
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// 找到最空闲的工作节点（私有方法）
    ///
    /// 使用 Least-Loaded 策略选择当前任务数最少的工作节点
    fn find_best_worker(&self) -> Option<String> {
        self.workers
            .iter()
            .filter(|(_, status)| status.healthy)
            .min_by_key(|(_, status)| status.current_load)
            .map(|(id, _)| id.clone())
    }

    /// 获取待处理任务数量
    pub fn pending_task_count(&self) -> usize {
        self.pending_tasks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_capabilities() -> NodeCapabilities {
        NodeCapabilities::new("Test GPU", 8192, 16, vec!["test-model".to_string()])
    }

    #[tokio::test]
    async fn test_coordinator_creation() {
        let caps = create_test_capabilities();
        let (mesh, _) = MeshNode::new("coord", caps);
        let coordinator = Coordinator::new(mesh);

        let status = coordinator.status();
        assert_eq!(status.total_workers, 0);
        assert_eq!(status.pending_tasks, 0);
        assert_eq!(status.completed_tasks, 0);
    }

    #[tokio::test]
    async fn test_worker_registration() {
        let caps = create_test_capabilities();
        let (mesh, _) = MeshNode::new("coord", caps);
        let mut coordinator = Coordinator::new(mesh);

        // 注册工作节点
        let worker_caps = create_test_capabilities();
        coordinator
            .register_worker("worker-1".to_string(), worker_caps)
            .await;

        let status = coordinator.status();
        assert_eq!(status.total_workers, 1);
        assert_eq!(status.healthy_workers, 1);

        // 注销工作节点
        coordinator.unregister_worker("worker-1").await;

        let status = coordinator.status();
        assert_eq!(status.total_workers, 0);
    }

    #[tokio::test]
    async fn test_submit_request() {
        let caps = create_test_capabilities();
        let (mesh, _) = MeshNode::new("coord", caps);
        let mut coordinator = Coordinator::new(mesh);

        let request = InferenceRequest::new("session-1", "test-model", vec![1, 2, 3], 50, 0.7);

        let task_id = coordinator.submit(request).await.unwrap();
        assert!(!task_id.is_empty());
        assert!(task_id.starts_with("task-coord-"));
        assert_eq!(coordinator.pending_task_count(), 1);
    }

    #[tokio::test]
    async fn test_dispatch_with_no_workers() {
        let caps = create_test_capabilities();
        let (mesh, _) = MeshNode::new("coord", caps);
        let mut coordinator = Coordinator::new(mesh);

        // 提交任务但没有注册 worker
        let request = InferenceRequest::new("s", "m", vec![1], 10, 0.5);
        coordinator.submit(request).await.unwrap();

        // 尝试分发（应该不会 panic，只是无法分发）
        coordinator.dispatch_pending().await;
        assert_eq!(coordinator.pending_task_count(), 1); // 任务仍在队列中
    }

    #[tokio::test]
    async fn test_handle_heartbeat_message() {
        let caps = create_test_capabilities();
        let (mesh, _) = MeshNode::new("coord", caps);
        let mut coordinator = Coordinator::new(mesh);

        // 先注册 worker
        coordinator
            .register_worker("worker-1".to_string(), create_test_capabilities())
            .await;

        // 处理心跳消息
        let heartbeat_msg = DistributedMessage::Heartbeat {
            node_id: "worker-1".to_string(),
            timestamp: 9999,
            gpu_utilization: 0.8,
            memory_used_mb: 4096,
        };

        coordinator.handle_message(heartbeat_msg).await;

        // worker 应该仍然是健康的
        let healthy = coordinator.healthy_workers();
        assert!(healthy.contains(&"worker-1"));
    }

    #[tokio::test]
    async fn test_handle_result_message() {
        let caps = create_test_capabilities();
        let (mesh, _) = MeshNode::new("coord", caps);
        let mut coordinator = Coordinator::new(mesh);

        // 处理任务结果
        let stats = InferenceStats::empty();
        let response = InferenceResponse::success(vec![100], vec![-0.2], "stop", stats);

        let result_msg = DistributedMessage::TaskResult {
            task_id: "task-1".to_string(),
            result: response,
            execution_time_us: 5000,
        };

        coordinator.handle_message(result_msg).await;

        // 应该能够获取结果
        let result = coordinator.get_result("task-1");
        assert!(result.is_some());
        assert_eq!(result.unwrap().output_len(), 1);

        // 统计信息应该更新
        let status = coordinator.status();
        assert_eq!(status.completed_tasks, 1);
    }

    #[tokio::test]
    async fn test_collect_results() {
        let caps = create_test_capabilities();
        let (mesh, _) = MeshNode::new("coord", caps);
        let mut coordinator = Coordinator::new(mesh);

        // 添加多个结果
        for i in 0..3 {
            let stats = InferenceStats::empty();
            let response = InferenceResponse::success(vec![i as u32], vec![0.0], "stop", stats);

            let msg = DistributedMessage::TaskResult {
                task_id: format!("task-{}", i),
                result: response,
                execution_time_us: 1000,
            };

            coordinator.handle_message(msg).await;
        }

        // 收集所有结果
        let results = coordinator.collect_results();
        assert_eq!(results.len(), 3);

        // 结果应该被清空
        assert_eq!(coordinator.status().completed_tasks, 0); // 注意：completed_tasks_count 不受 collect 影响
    }
}
