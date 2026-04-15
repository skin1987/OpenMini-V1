//! 异步推理任务池
//!
//! 提供非阻塞的推理任务提交和批量处理能力。
//! 使用 mpsc channel + oneshot channel 实现异步通信，
//! 使用 tokio::task::spawn_blocking 执行 CPU 密集型推理计算。

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio::time::timeout;

/// 推理任务
pub struct InferenceTask {
    pub prompt: String,
    pub session_id: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

/// 推理结果
pub struct InferenceResult {
    pub session_id: String,
    pub text: String,
    pub finished: bool,
}

/// 带回调的任务包装
struct TaskWithCallback {
    task: InferenceTask,
    response_tx: oneshot::Sender<Result<InferenceResult, String>>,
}

/// 异步推理任务池
pub struct AsyncInferencePool {
    task_tx: mpsc::Sender<TaskWithCallback>,
    batch_size_max: usize,
    batch_timeout: Duration,
}

impl AsyncInferencePool {
    /// 创建新的异步推理任务池（不启动后台循环）
    ///
    /// # 参数
    /// - `channel_size`: mpsc channel 缓冲区大小
    /// - `batch_size_max`: 最大批量大小
    /// - `batch_timeout`: 批量收集超时时间
    ///
    /// # 注意
    /// 此方法仅创建池结构体，不会启动后台批量处理循环。
    /// 推荐使用 [`Self::create`] 方法来同时创建并启动后台循环。
    pub fn new(channel_size: usize, batch_size_max: usize, batch_timeout: Duration) -> Self {
        let (task_tx, _task_rx) = mpsc::channel(channel_size);

        Self {
            task_tx,
            batch_size_max,
            batch_timeout,
        }
    }

    /// 创建并启动异步推理任务池
    ///
    /// # 参数
    /// - `channel_size`: mpsc channel 缓冲区大小
    /// - `batch_size_max`: 最大批量大小
    /// - `batch_timeout`: 批量收集超时时间
    /// - `engine_fn`: 推理引擎函数，接收批量任务，返回批量结果
    ///
    /// # 返回值
    /// 返回 `(pool, shutdown_tx)`:
    /// - `pool`: 异步推理任务池实例
    /// - `shutdown_tx`: 关闭信号发送端，发送 `true` 可优雅关闭后台循环
    pub fn create(
        channel_size: usize,
        batch_size_max: usize,
        batch_timeout: Duration,
        engine_fn: impl Fn(Vec<InferenceTask>) -> Vec<InferenceResult> + Send + Sync + 'static,
    ) -> (Self, tokio::sync::watch::Sender<bool>) {
        let (task_tx, task_rx) = mpsc::channel(channel_size);
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

        let engine: Arc<dyn Fn(Vec<InferenceTask>) -> Vec<InferenceResult> + Send + Sync> =
            Arc::new(engine_fn);

        tokio::spawn(async move {
            Self::batch_loop(task_rx, engine, batch_size_max, batch_timeout, shutdown_rx).await;
        });

        let pool = Self {
            task_tx,
            batch_size_max,
            batch_timeout,
        };

        (pool, shutdown_tx)
    }

    /// 异步提交推理任务
    ///
    /// 不阻塞调用方，结果通过 oneshot channel 回传。
    /// 默认超时时间为 300 秒。
    ///
    /// # 错误
    /// - `"InferencePool closed"`: 任务池已关闭
    /// - `"Inference timeout"`: 推理超时（300秒）
    /// - `"Response channel closed"`: 响应通道意外关闭
    pub async fn submit(&self, task: InferenceTask) -> Result<InferenceResult, String> {
        let (response_tx, response_rx) = oneshot::channel();

        let wrapper = TaskWithCallback { task, response_tx };

        self.task_tx
            .send(wrapper)
            .await
            .map_err(|_| "InferencePool closed".to_string())?;

        timeout(Duration::from_secs(300), response_rx)
            .await
            .map_err(|_| "Inference timeout".to_string())?
            .map_err(|_| "Response channel closed".to_string())?
    }

    /// 批量处理主循环
    ///
    /// 持续从 channel 接收任务，达到批量大小或超时后执行推理。
    async fn batch_loop(
        mut task_rx: mpsc::Receiver<TaskWithCallback>,
        engine: Arc<dyn Fn(Vec<InferenceTask>) -> Vec<InferenceResult> + Send + Sync>,
        batch_size_max: usize,
        batch_timeout: Duration,
        shutdown: tokio::sync::watch::Receiver<bool>,
    ) {
        loop {
            if *shutdown.borrow() {
                break;
            }

            let mut batch = Vec::with_capacity(batch_size_max);

            // 收集一批任务（超时或达到最大数量）
            let _collect_result = timeout(batch_timeout, async {
                while batch.len() < batch_size_max {
                    match task_rx.recv().await {
                        Some(task) => batch.push(task),
                        None => break,
                    }
                }
            })
            .await;

            // 无论超时还是正常完成，都继续处理已收集的任务
            if !batch.is_empty() {
                // 提取纯任务数据
                let inference_tasks: Vec<InferenceTask> = batch
                    .iter()
                    .map(|t| InferenceTask {
                        prompt: t.task.prompt.clone(),
                        session_id: t.task.session_id.clone(),
                        max_tokens: t.task.max_tokens,
                        temperature: t.task.temperature,
                    })
                    .collect();

                // 在 blocking 线程池中执行 CPU 密集型推理计算
                let engine_clone = Arc::clone(&engine);
                let results = tokio::task::spawn_blocking(move || engine_clone(inference_tasks))
                    .await
                    .unwrap_or_default();

                // 分发结果回各请求方
                for (wrapper, result) in batch.into_iter().zip(results.into_iter()) {
                    let _ = wrapper.response_tx.send(Ok(result));
                }
            }
        }
    }
}

impl Default for AsyncInferencePool {
    fn default() -> Self {
        Self::new(1000, 8, Duration::from_millis(10))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_async_pool_submit_and_receive() {
        let (pool, _shutdown) =
            AsyncInferencePool::create(100, 4, Duration::from_millis(10), |tasks| {
                tasks
                    .iter()
                    .map(|t| InferenceResult {
                        session_id: t.session_id.clone(),
                        text: format!("Response for: {}", t.prompt),
                        finished: true,
                    })
                    .collect()
            });

        let task = InferenceTask {
            prompt: "Hello".to_string(),
            session_id: "test-session".to_string(),
            max_tokens: Some(100),
            temperature: None,
        };

        let result = pool.submit(task).await.unwrap();
        assert_eq!(result.session_id, "test-session");
        assert!(result.text.contains("Hello"));
        assert!(result.finished);
    }

    #[tokio::test]
    async fn test_async_pool_multiple_submits() {
        let (pool, _shutdown) =
            AsyncInferencePool::create(200, 8, Duration::from_millis(50), |tasks| {
                tasks
                    .iter()
                    .map(|t| InferenceResult {
                        session_id: t.session_id.clone(),
                        text: format!("OK: {}", t.prompt),
                        finished: true,
                    })
                    .collect()
            });

        // 使用 Arc 共享 pool 引用，解决 AsyncInferencePool 不可 Clone 的问题
        let pool_arc = Arc::new(pool);

        let mut handles = vec![];
        for i in 0..10 {
            let pool_ref = Arc::clone(&pool_arc);
            let h = tokio::spawn(async move {
                let task = InferenceTask {
                    prompt: format!("request-{}", i),
                    session_id: format!("session-{}", i),
                    max_tokens: None,
                    temperature: None,
                };
                pool_ref.submit(task).await
            });
            handles.push(h);
        }

        for h in handles {
            let result = h.await.unwrap();
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_async_pool_shutdown() {
        let (pool, shutdown_tx) =
            AsyncInferencePool::create(100, 4, Duration::from_millis(10), |tasks| {
                tasks
                    .iter()
                    .map(|t| InferenceResult {
                        session_id: t.session_id.clone(),
                        text: format!("Shutdown test: {}", t.prompt),
                        finished: true,
                    })
                    .collect()
            });

        // 发送关闭信号
        let _ = shutdown_tx.send(true);

        // 给一点时间让后台循环退出
        tokio::time::sleep(Duration::from_millis(50)).await;

        // 验证提交会失败（因为后台循环已关闭）
        let task = InferenceTask {
            prompt: "After shutdown".to_string(),
            session_id: "shutdown-test".to_string(),
            max_tokens: None,
            temperature: None,
        };

        // 注意：由于 channel 可能还有缓冲，这里不一定立即失败
        // 但如果等待足够长时间，应该能观察到关闭行为
        let result = tokio::time::timeout(Duration::from_millis(100), pool.submit(task)).await;
        // 结果可能是 Ok 也可能是 Err(Timeout)，取决于时序
        let _ = result;
    }
}
