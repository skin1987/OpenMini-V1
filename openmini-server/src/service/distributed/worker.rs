//! 分布式推理工作节点
//!
//! 在每个 GPU 节点上运行，执行实际的推理计算。
//!
//! ## 职责
//!
//! 1. **任务接收**: 从协调器接收推理任务
//! 2. **推理执行**: 调用本地推理引擎完成计算
//! 3. **结果返回**: 将推理结果发送回协调器
//! 4. **状态报告**: 定期发送心跳和资源使用情况
//!
//! ## 工作流程
//!
//! ```text
//! ┌──────────┐    TaskAssign     ┌──────────┐
//! │Coordinator│─────────────────▶│  Worker   │
//! │          │◀─────────────────│          │
//! │          │   TaskResult     │          │
//! │          │                  │ Inference │
//! │          │   Heartbeat      │  Engine   │
//! │          │◀─────────────────│          │
//! └──────────┘                  └──────────┘
//! ```

use std::collections::VecDeque;

use tokio::time::{interval, Duration};
use tracing::{debug, info, trace, warn};

use super::mesh::MeshNode;
use super::protocol::*;
use crate::error::AppError;

/// 分布式工作节点配置
#[derive(Debug, Clone)]
pub struct DistributedWorkerConfig {
    /// 心跳间隔（秒）
    pub heartbeat_interval_secs: u64,
    /// 最大并发任务数
    pub max_concurrent_tasks: usize,
}

impl Default for DistributedWorkerConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval_secs: 5,
            max_concurrent_tasks: 4,
        }
    }
}

/// 分布式推理工作节点
///
/// 运行在每个 GPU 节点上，负责：
/// - 接收并处理协调器分配的推理任务
/// - 执行实际的模型推理计算
/// - 向协调器报告健康状态和资源使用情况
pub struct WorkerNode {
    /// Mesh 网络句柄
    mesh: MeshNode,
    /// 本地任务队列（待处理）
    local_queue: VecDeque<DistributedMessage>,
    /// 是否正在处理任务
    processing: bool,
    /// 配置参数
    config: DistributedWorkerConfig,
    /// 统计信息：已处理的任务总数
    tasks_processed: u64,
}

impl WorkerNode {
    /// 创建新的工作节点实例
    ///
    /// # 参数
    ///
    /// - `mesh`: 已初始化并连接到协调器的 Mesh 网络节点
    /// - `config`: 可选的工作节点配置，默认值见 [`WorkerConfig::default`]
    pub fn new(mesh: MeshNode, config: Option<DistributedWorkerConfig>) -> Self {
        let config = config.unwrap_or_default();

        info!(
            worker_id = %mesh.node_id(),
            gpu_name = %mesh.capabilities().gpu_name,
            max_batch = mesh.capabilities().max_batch_size,
            "初始化工作节点"
        );

        Self {
            mesh,
            local_queue: VecDeque::new(),
            processing: false,
            config,
            tasks_processed: 0,
        }
    }

    /// 启动工作循环
    ///
    /// 主事件循环，持续运行直到收到停止信号或发生致命错误。
    /// 循环逻辑：
    ///
    /// 1. 接收来自协调器的消息
    /// 2. 处理任务分配、心跳请求等
    /// 3. 定期发送心跳
    /// 4. 执行本地推理任务
    pub async fn run(&mut self) -> Result<(), AppError> {
        info!(worker_id = %self.mesh.node_id(), "启动工作节点主循环");

        // 创建心跳定时器
        let mut heartbeat_interval =
            interval(Duration::from_secs(self.config.heartbeat_interval_secs));

        loop {
            tokio::select! {
                // 接收消息
                msg = self.mesh.recv() => {
                    match msg {
                        Some(message) => {
                            self.handle_message(message).await?;
                        }
                        None => {
                            warn!("通道已关闭，工作节点退出");
                            break;
                        }
                    }
                }

                // 心跳定时器触发
                _ = heartbeat_interval.tick() => {
                    self.send_heartbeat().await;
                }
            }

            // 尝试处理本地队列中的任务
            if !self.processing && !self.local_queue.is_empty() {
                if let Some(task_msg) = self.local_queue.pop_front() {
                    self.process_task(task_msg).await?;
                }
            }
        }

        Ok(())
    }

    /// 处理接收到的消息
    async fn handle_message(&mut self, msg: DistributedMessage) -> Result<(), AppError> {
        match msg {
            DistributedMessage::TaskAssign {
                task_id,
                payload,
                priority,
            } => {
                debug!(
                    worker_id = %self.mesh.node_id(),
                    task_id = %task_id,
                    priority = priority,
                    model = %payload.model_name,
                    "收到任务分配"
                );

                // 如果当前正在处理其他任务，加入队列等待
                if self.processing {
                    debug!(
                        worker_id = %self.mesh.node_id(),
                        task_id = %task_id,
                        "当前繁忙，任务加入队列"
                    );
                    self.local_queue.push_back(DistributedMessage::TaskAssign {
                        task_id,
                        payload,
                        priority,
                    });
                } else {
                    // 直接处理任务
                    let task_msg = DistributedMessage::TaskAssign {
                        task_id,
                        payload,
                        priority,
                    };
                    self.process_task(task_msg).await?;
                }
            }

            // 其他消息类型在此原型中暂不处理
            _ => {
                debug!(worker_id = %self.mesh.node_id(), "忽略不支持的消息类型");
            }
        }

        Ok(())
    }

    /// 处理单个推理任务
    ///
    /// 执行完整的推理流程：
    /// 1. 调用推理引擎进行前向传播
    /// 2. 收集输出 token 和统计信息
    /// 3. 构建响应消息
    /// 4. 发送结果给协调器
    async fn process_task(&mut self, task_msg: DistributedMessage) -> Result<(), AppError> {
        // 提取任务信息
        let (task_id, request) = match task_msg {
            DistributedMessage::TaskAssign {
                task_id, payload, ..
            } => (task_id, payload),
            _ => return Err(AppError::Internal("无效的任务消息".to_string())),
        };

        self.processing = true;

        debug!(
            worker_id = %self.mesh.node_id(),
            task_id = %task_id,
            "开始处理推理任务"
        );

        // 记录开始时间
        let start_time = std::time::Instant::now();

        // 执行推理（这里模拟实际推理过程）
        let response = self.process_inference(request.clone()).await;

        // 计算执行时间（微秒）
        let execution_time_us = start_time.elapsed().as_micros() as u64;

        match response {
            Ok(result) => {
                // 发送成功结果
                let result_msg = DistributedMessage::TaskResult {
                    task_id: task_id.clone(),
                    result,
                    execution_time_us,
                };

                if let Err(e) = self.mesh.send_to(/* coordinator_id */ "", result_msg).await {
                    // 在实际实现中，应该知道 coordinator 的 ID
                    warn!(
                        worker_id = %self.mesh.node_id(),
                        task_id = %task_id,
                        error = %e,
                        "发送结果失败"
                    );
                } else {
                    self.tasks_processed += 1;
                    debug!(
                        worker_id = %self.mesh.node_id(),
                        task_id = %task_id,
                        time_us = execution_time_us,
                        "任务完成"
                    );
                }
            }
            Err(e) => {
                // 发送错误报告
                let error_msg = DistributedMessage::Error {
                    task_id: task_id.clone(),
                    error_code: 500,
                    message: e.to_string(),
                };

                if let Err(send_err) = self.mesh.send_to("", error_msg).await {
                    warn!(
                        worker_id = %self.mesh.node_id(),
                        task_id = %task_id,
                        error = %send_err,
                        "发送错误报告失败"
                    );
                }

                warn!(
                    worker_id = %self.mesh.node_id(),
                    task_id = %task_id,
                    error = %e,
                    "任务执行失败"
                );
            }
        }

        self.processing = false;
        Ok(())
    }

    /// 执行实际的推理计算
    ///
    /// # 注意
    ///
    /// 这是原型实现，使用模拟数据代替真实的模型推理。
    /// 生产环境应替换为调用实际的 InferenceEngine。
    async fn process_inference(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, AppError> {
        debug!(
            worker_id = %self.mesh.node_id(),
            session = %request.session_id,
            model = %request.model_name,
            input_len = request.input_len(),
            max_tokens = request.max_tokens,
            "执行推理计算"
        );

        // 模拟推理延迟（在实际实现中替换为真实推理）
        tokio::time::sleep(Duration::from_millis(10)).await;

        // 模拟生成输出 token（实际应从模型获取）
        let output_tokens: Vec<u32> = (0..request.max_tokens.min(10))
            .map(|i| 100 + i as u32)
            .collect();

        // 模拟 logprobs
        let logprobs: Vec<f32> = output_tokens.iter().map(|_| -0.5).collect();

        // 模拟统计信息
        let stats = InferenceStats {
            ttft_ms: 5.0,             // 模拟首 token 延迟
            tokens_per_second: 100.0, // 模拟生成速度
            total_time_ms: 15.0,      // 模拟总时间
            gpu_memory_mb: 1024,      // 模拟显存占用
        };

        Ok(InferenceResponse::success(
            output_tokens,
            logprobs,
            "length", // 模拟结束原因
            stats,
        ))
    }

    /// 发送心跳消息
    ///
    /// 定期向协调器报告自身状态，包括：
    /// - 节点 ID
    /// - 当前时间戳
    /// - GPU 利用率（模拟值）
    /// - 内存使用量（模拟值）
    async fn send_heartbeat(&self) {
        let heartbeat = DistributedMessage::Heartbeat {
            node_id: self.mesh.node_id().to_string(),
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
            gpu_utilization: 0.7 + (rand::random::<f32>() * 0.2), // 模拟 GPU 使用率 70-90%
            memory_used_mb: 4096 + (rand::random::<u64>() % 2048), // 模拟内存使用
        };

        // 在实际实现中，应该发送给已知的 coordinator ID
        // 这里简化处理，假设 coordinator 已连接
        if let Err(e) = self.mesh.send_to(/* coordinator_id */ "", heartbeat).await {
            warn!(
                worker_id = %self.mesh.node_id(),
                error = %e,
                "发送心跳失败"
            );
        } else {
            trace!(
                worker_id = %self.mesh.node_id(),
                "心跳发送成功"
            );
        }
    }

    /// 获取工作节点统计信息
    pub fn stats(&self) -> WorkerStats {
        WorkerStats {
            node_id: self.mesh.node_id().to_string(),
            queue_length: self.local_queue.len(),
            is_processing: self.processing,
            tasks_processed: self.tasks_processed,
        }
    }
}

/// 工作节点运行时统计
#[derive(Debug, Clone)]
pub struct WorkerStats {
    /// 节点 ID
    pub node_id: String,
    /// 当前队列中的待处理任务数
    pub queue_length: usize,
    /// 是否正在处理任务
    pub is_processing: bool,
    /// 累计已处理的任务数
    pub tasks_processed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_mesh() -> (MeshNode, mpsc::Sender<DistributedMessage>) {
        let caps = NodeCapabilities::new("Test GPU", 8192, 16, vec!["test-model".to_string()]);
        MeshNode::new("worker-test", caps)
    }

    #[tokio::test]
    async fn test_worker_creation() {
        let (mesh, _) = create_test_mesh();
        let worker = WorkerNode::new(mesh, None);

        assert_eq!(worker.mesh.node_id(), "worker-test");
        assert!(!worker.processing);
        assert_eq!(worker.stats().queue_length, 0);
    }

    #[tokio::test]
    async fn test_worker_custom_config() {
        let (mesh, _) = create_test_mesh();
        let config = DistributedWorkerConfig {
            heartbeat_interval_secs: 10,
            max_concurrent_tasks: 8,
        };
        let worker = WorkerNode::new(mesh, Some(config));

        assert_eq!(worker.config.heartbeat_interval_secs, 10);
        assert_eq!(worker.config.max_concurrent_tasks, 8);
    }

    #[tokio::test]
    async fn test_process_inference() {
        let (mesh, _) = create_test_mesh();
        let worker = WorkerNode::new(mesh, None);

        let request =
            InferenceRequest::new("session-1", "test-model", vec![1, 2, 3, 4, 5], 20, 0.8);

        let result = worker.process_inference(request).await.unwrap();

        assert!(!result.output_tokens.is_empty());
        assert_eq!(result.output_len(), 20); // max_tokens = 20
        assert_eq!(result.finish_reason, "length");
        assert!(result.stats.ttft_ms > 0.0);
        assert!(result.stats.tokens_per_second > 0.0);
    }

    #[tokio::test]
    async fn test_worker_stats() {
        let (mesh, _) = create_test_mesh();
        let mut worker = WorkerNode::new(mesh, None);

        let stats = worker.stats();
        assert_eq!(stats.node_id, "worker-test");
        assert_eq!(stats.queue_length, 0);
        assert!(!stats.is_processing);
        assert_eq!(stats.tasks_processed, 0);
    }

    #[tokio::test]
    async fn test_handle_task_assign_message() {
        let (mut mesh, _) = create_test_mesh();
        let mut worker = WorkerNode::new(mesh, None);

        let request = InferenceRequest::new("s", "m", vec![1], 5, 0.5);
        let task_msg = DistributedMessage::TaskAssign {
            task_id: "task-1".to_string(),
            payload: request,
            priority: 1,
        };

        // 处理任务分配消息
        let result = worker.handle_message(task_msg).await;
        assert!(result.is_ok());

        // 任务应该被处理（因为 processing = false）
        // 注意：process_task 会尝试发送结果到 coordinator，
        // 但测试环境中没有连接 coordinator，所以可能会失败
        // 这里主要验证不 panic 且能正常处理
    }

    #[tokio::test]
    async fn test_task_queued_when_busy() {
        let (mut mesh, _) = create_test_mesh();
        let mut worker = WorkerNode::new(mesh, None);

        // 模拟工作节点正在处理任务
        worker.processing = true;

        let request = InferenceRequest::new("s", "m", vec![1], 5, 0.5);
        let task_msg = DistributedMessage::TaskAssign {
            task_id: "task-queued".to_string(),
            payload: request,
            priority: 0,
        };

        // 处理任务（应该被加入队列）
        worker.handle_message(task_msg).await.unwrap();

        assert_eq!(worker.stats().queue_length, 1);
    }
}
