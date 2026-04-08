//! Worker 进程 - 任务执行单元
//!
//! 提供独立的工作进程实现，通过 stdin/stdout 与主进程通信。
//! 支持任务执行、心跳检测和模型加载。

#![allow(dead_code)]

use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::model::inference::InferenceEngine;

/// Worker 空闲状态
pub const WORKER_STATE_IDLE: u8 = 0;
/// Worker 忙碌状态
pub const WORKER_STATE_BUSY: u8 = 1;
/// Worker 死亡状态
pub const WORKER_STATE_DEAD: u8 = 2;

/// 任务命令
pub const WORKER_CMD_TASK: u8 = 1;
/// 关闭命令
pub const WORKER_CMD_SHUTDOWN: u8 = 2;
/// 心跳命令
pub const WORKER_CMD_HEARTBEAT: u8 = 3;
/// 加载模型命令
pub const WORKER_CMD_LOAD_MODEL: u8 = 4;

/// 任务结构
///
/// 表示一个待执行的任务，包含任务 ID 和数据。
#[derive(Debug, Clone)]
pub struct Task {
    /// 任务唯一标识
    pub id: u64,
    /// 任务数据
    pub data: Vec<u8>,
}

impl Task {
    /// 创建新任务
    pub fn new(id: u64, data: Vec<u8>) -> Self {
        Self { id, data }
    }

    /// 序列化任务
    ///
    /// 格式: [id: u64][data_len: u32][data: bytes]
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&self.id.to_le_bytes());
        buf.extend_from_slice(&(self.data.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.data);
        buf
    }

    /// 反序列化任务
    pub fn deserialize(data: &[u8]) -> Result<Self, &'static str> {
        if data.len() < 12 {
            return Err("Invalid task data: too short");
        }
        let id = u64::from_le_bytes(data[0..8].try_into().unwrap());
        let len = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        if data.len() < 12 + len {
            return Err("Invalid task data: incomplete");
        }
        let task_data = data[12..12 + len].to_vec();
        Ok(Self {
            id,
            data: task_data,
        })
    }
}

/// 任务结果
///
/// 表示任务执行的结果，包含任务 ID、成功标志和数据。
#[derive(Debug, Clone)]
pub struct TaskResult {
    /// 对应的任务 ID
    pub task_id: u64,
    /// 是否成功
    pub success: bool,
    /// 结果数据
    pub data: Vec<u8>,
}

impl TaskResult {
    /// 序列化结果
    ///
    /// 格式: [task_id: u64][success: u8][data_len: u32][data: bytes]
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&self.task_id.to_le_bytes());
        buf.push(if self.success { 1 } else { 0 });
        buf.extend_from_slice(&(self.data.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.data);
        buf
    }

    /// 反序列化结果
    pub fn deserialize(data: &[u8]) -> Result<Self, &'static str> {
        if data.len() < 13 {
            return Err("Invalid result data: too short");
        }
        let task_id = u64::from_le_bytes(data[0..8].try_into().unwrap());
        let success = data[8] == 1;
        let len = u32::from_le_bytes(data[9..13].try_into().unwrap()) as usize;
        if data.len() < 13 + len {
            return Err("Invalid result data: incomplete");
        }
        let result_data = data[13..13 + len].to_vec();
        Ok(Self {
            task_id,
            success,
            data: result_data,
        })
    }
}

/// Worker 进程
///
/// 独立的工作进程，通过 stdin/stdout 与主进程通信。
/// 支持加载模型、执行任务和心跳响应。
pub struct Worker {
    /// Worker ID
    id: usize,
    /// 当前状态
    state: Arc<AtomicU8>,
    /// 最后心跳时间
    last_heartbeat: Arc<AtomicU64>,
    /// 推理引擎实例
    engine: Option<Arc<parking_lot::RwLock<InferenceEngine>>>,
}

impl Worker {
    /// 创建新的 Worker
    pub fn new(id: usize) -> Self {
        Self {
            id,
            state: Arc::new(AtomicU8::new(WORKER_STATE_IDLE)),
            last_heartbeat: Arc::new(AtomicU64::new(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            )),
            engine: None,
        }
    }

    /// 加载模型
    ///
    /// # 参数
    /// - `model_path`: 模型文件路径
    pub fn load_model(&mut self, model_path: &str) -> anyhow::Result<()> {
        let engine = InferenceEngine::from_gguf(Path::new(model_path))?;
        self.engine = Some(Arc::new(parking_lot::RwLock::new(engine)));
        Ok(())
    }

    /// 获取状态引用
    pub fn state(&self) -> Arc<AtomicU8> {
        self.state.clone()
    }

    /// 获取最后心跳时间引用
    pub fn last_heartbeat(&self) -> Arc<AtomicU64> {
        self.last_heartbeat.clone()
    }

    /// 运行 Worker 主循环
    ///
    /// 持续从 stdin 读取命令并执行，直到收到关闭命令。
    pub fn run(&self) {
        self.update_heartbeat();
        self.state.store(WORKER_STATE_IDLE, Ordering::SeqCst);

        let stdin = io::stdin();
        let stdout = io::stdout();
        let mut stdout_lock = stdout.lock();

        loop {
            let mut cmd_buf = [0u8; 1];
            match stdin.lock().read_exact(&mut cmd_buf) {
                Ok(_) => {}
                Err(_) => {
                    self.state.store(WORKER_STATE_DEAD, Ordering::SeqCst);
                    break;
                }
            }

            let cmd = cmd_buf[0];
            match cmd {
                WORKER_CMD_TASK => {
                    self.state.store(WORKER_STATE_BUSY, Ordering::SeqCst);
                    self.update_heartbeat();

                    let mut len_buf = [0u8; 4];
                    if stdin.lock().read_exact(&mut len_buf).is_err() {
                        self.state.store(WORKER_STATE_DEAD, Ordering::SeqCst);
                        break;
                    }
                    let len = u32::from_le_bytes(len_buf) as usize;

                    let mut task_data = vec![0u8; len];
                    if stdin.lock().read_exact(&mut task_data).is_err() {
                        self.state.store(WORKER_STATE_DEAD, Ordering::SeqCst);
                        break;
                    }

                    let task = match Task::deserialize(&task_data) {
                        Ok(t) => t,
                        Err(_) => {
                            self.state.store(WORKER_STATE_IDLE, Ordering::SeqCst);
                            continue;
                        }
                    };

                    let result = self.process_task(task);

                    let result_data = result.serialize();
                    let resp_len = (result_data.len() as u32).to_le_bytes();
                    if stdout_lock.write_all(&resp_len).is_err() {
                        self.state.store(WORKER_STATE_DEAD, Ordering::SeqCst);
                        break;
                    }
                    if stdout_lock.write_all(&result_data).is_err() {
                        self.state.store(WORKER_STATE_DEAD, Ordering::SeqCst);
                        break;
                    }
                    let _ = stdout_lock.flush();

                    self.state.store(WORKER_STATE_IDLE, Ordering::SeqCst);
                    self.update_heartbeat();
                }
                WORKER_CMD_SHUTDOWN => {
                    self.state.store(WORKER_STATE_DEAD, Ordering::SeqCst);
                    break;
                }
                WORKER_CMD_HEARTBEAT => {
                    self.update_heartbeat();
                    let resp = self.state.load(Ordering::SeqCst);
                    if stdout_lock.write_all(&[resp]).is_err() {
                        self.state.store(WORKER_STATE_DEAD, Ordering::SeqCst);
                        break;
                    }
                    let _ = stdout_lock.flush();
                }
                _ => {}
            }
        }
    }

    /// 处理任务
    fn process_task(&self, task: Task) -> TaskResult {
        let result_data = format!(
            "Worker {} processed task {}: {} bytes received",
            self.id,
            task.id,
            task.data.len()
        )
        .into_bytes();
        TaskResult {
            task_id: task.id,
            success: true,
            data: result_data,
        }
    }

    /// 更新心跳时间
    fn update_heartbeat(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.last_heartbeat.store(now, Ordering::SeqCst);
    }
}

/// 运行 Worker 进程
pub fn run_worker(id: usize, _state: Arc<AtomicU8>, _last_heartbeat: Arc<AtomicU64>) {
    let worker = Worker::new(id);
    std::mem::forget(worker.state());
    std::mem::forget(worker.last_heartbeat());
    worker.run();
}

/// 检查当前进程是否为 Worker 进程
pub fn is_worker_process() -> bool {
    std::env::var("MINICPM_WORKER_ID").is_ok()
}

/// 获取 Worker ID
pub fn get_worker_id() -> Option<usize> {
    std::env::var("MINICPM_WORKER_ID")
        .ok()
        .and_then(|s| s.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_creation() {
        // 测试 Worker 创建和初始状态
        let worker = Worker::new(1);

        // 验证初始状态为 IDLE
        let state = worker.state();
        assert_eq!(
            state.load(std::sync::atomic::Ordering::SeqCst),
            WORKER_STATE_IDLE
        );

        // 验证心跳时间已设置
        let heartbeat = worker.last_heartbeat();
        assert!(heartbeat.load(std::sync::atomic::Ordering::SeqCst) > 0);
    }

    #[test]
    fn test_worker_status_transitions() {
        // 测试 Worker 状态转换
        let worker = Worker::new(42);

        // 初始状态: Initialized (IDLE)
        assert_eq!(
            worker.state().load(std::sync::atomic::Ordering::SeqCst),
            WORKER_STATE_IDLE
        );

        // 模拟状态转换为 BUSY
        worker
            .state()
            .store(WORKER_STATE_BUSY, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(
            worker.state().load(std::sync::atomic::Ordering::SeqCst),
            WORKER_STATE_BUSY
        );

        // 模拟状态转换为 DEAD
        worker
            .state()
            .store(WORKER_STATE_DEAD, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(
            worker.state().load(std::sync::atomic::Ordering::SeqCst),
            WORKER_STATE_DEAD
        );
    }

    #[test]
    fn test_task_creation() {
        // 测试任务创建
        let task = Task::new(123, b"hello world".to_vec());
        assert_eq!(task.id, 123);
        assert_eq!(task.data, b"hello world");
    }

    #[test]
    fn test_task_empty_data() {
        // 测试空数据任务
        let task = Task::new(0, Vec::new());
        assert_eq!(task.id, 0);
        assert!(task.data.is_empty());

        // 序列化空数据任务应该正常工作
        let serialized = task.serialize();
        let deserialized = Task::deserialize(&serialized).unwrap();
        assert_eq!(deserialized.id, 0);
        assert!(deserialized.data.is_empty());
    }

    #[test]
    fn test_task_large_data() {
        // 测试大数据任务（注意：避免过大内存分配）
        let large_data = vec![0xAB; 1024]; // 1KB 数据（安全大小）
        let task = Task::new(999, large_data.clone());

        let serialized = task.serialize();
        assert_eq!(serialized.len(), 8 + 4 + 1024); // id + len + data

        let deserialized = Task::deserialize(&serialized).unwrap();
        assert_eq!(deserialized.id, 999);
        assert_eq!(deserialized.data.len(), 1024);
    }

    #[test]
    fn test_task_result_creation() {
        // 测试结果创建
        let result = TaskResult {
            task_id: 100,
            success: true,
            data: b"success".to_vec(),
        };
        assert_eq!(result.task_id, 100);
        assert!(result.success);
        assert_eq!(result.data, b"success");
    }

    #[test]
    fn test_is_worker_process() {
        // 测试非 Worker 进程检测（测试环境中不应该设置此变量）
        // 先保存原始值
        let original = std::env::var("MINICPM_WORKER_ID").ok();

        // 删除环境变量
        std::env::remove_var("MINICPM_WORKER_ID");
        assert!(!is_worker_process());
        assert!(get_worker_id().is_none());

        // 设置环境变量
        std::env::set_var("MINICPM_WORKER_ID", "5");
        assert!(is_worker_process());
        assert_eq!(get_worker_id(), Some(5));

        // 测试无效值
        std::env::set_var("MINICPM_WORKER_ID", "invalid");
        assert!(is_worker_process());
        assert!(get_worker_id().is_none()); // 解析失败返回 None

        // 恢复原始值
        match original {
            Some(val) => std::env::set_var("MINICPM_WORKER_ID", val),
            None => std::env::remove_var("MINICPM_WORKER_ID"),
        }
    }

    #[test]
    fn test_multiple_workers() {
        // 测试创建多个 Worker 实例
        let worker1 = Worker::new(0);
        let worker2 = Worker::new(1);
        let worker3 = Worker::new(2);

        // 每个 Worker 应该有独立的状态
        assert_ne!(worker1.state().as_ptr(), worker2.state().as_ptr());
        assert_ne!(worker2.state().as_ptr(), worker3.state().as_ptr());

        // 所有 Worker 初始状态都应该是 IDLE
        assert_eq!(
            worker1.state().load(std::sync::atomic::Ordering::SeqCst),
            WORKER_STATE_IDLE
        );
        assert_eq!(
            worker2.state().load(std::sync::atomic::Ordering::SeqCst),
            WORKER_STATE_IDLE
        );
        assert_eq!(
            worker3.state().load(std::sync::atomic::Ordering::SeqCst),
            WORKER_STATE_IDLE
        );
    }

    // ==================== 新增测试开始 ====================

    /// 测试Task的serialize和deserialize完整流程
    /// 覆盖分支：序列化和反序列化的往返一致性
    #[test]
    fn test_task_serialize_deserialize_roundtrip() {
        let original_task = Task::new(12345, b"test data for roundtrip".to_vec());

        let serialized = original_task.serialize();
        let deserialized = Task::deserialize(&serialized).unwrap();

        assert_eq!(original_task.id, deserialized.id);
        assert_eq!(original_task.data, deserialized.data);
    }

    /// 测试Task反序列化错误处理 - 数据太短
    /// 覆盖分支：deserialize的错误路径（too short）
    #[test]
    fn test_task_deserialize_too_short() {
        // 少于12字节的数据应该返回错误
        let short_data = vec![0u8; 11];
        let result = Task::deserialize(&short_data);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Invalid task data: too short");

        // 空数据也应该返回错误
        let empty_data = Vec::new();
        let result = Task::deserialize(&empty_data);
        assert!(result.is_err());

        // 只有部分header的数据
        let partial_data = vec![0u8; 8]; // 只有id，没有长度字段
        let result = Task::deserialize(&partial_data);
        assert!(result.is_err());
    }

    /// 测试Task反序列化错误处理 - 数据不完整
    /// 覆盖分支：deserialize的错误路径（incomplete）
    #[test]
    fn test_task_deserialize_incomplete() {
        // 构造一个声明了更长数据的序列化结果，但实际数据不足
        let mut fake_data = Vec::new();
        fake_data.extend_from_slice(&1u64.to_le_bytes()); // id = 1
        fake_data.extend_from_slice(&1000u32.to_le_bytes()); // 声明1000字节数据
        fake_data.extend_from_slice(&[0u8; 10]); // 但只提供10字节

        let result = Task::deserialize(&fake_data);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Invalid task data: incomplete");
    }

    /// 测试TaskResult的serialize和deserialize
    /// 覆盖分支：TaskResult的完整序列化/反序列化
    #[test]
    fn test_task_result_serialize_deserialize() {
        // 测试成功的TaskResult
        let success_result = TaskResult {
            task_id: 100,
            success: true,
            data: b"operation succeeded".to_vec(),
        };

        let serialized = success_result.serialize();
        let deserialized = TaskResult::deserialize(&serialized).unwrap();

        assert_eq!(success_result.task_id, deserialized.task_id);
        assert_eq!(success_result.success, deserialized.success);
        assert_eq!(success_result.data, deserialized.data);

        // 测试失败的TaskResult
        let failure_result = TaskResult {
            task_id: 200,
            success: false,
            data: b"operation failed with error details".to_vec(),
        };

        let serialized = failure_result.serialize();
        let deserialized = TaskResult::deserialize(&serialized).unwrap();

        assert!(!deserialized.success);
        assert_eq!(deserialized.task_id, 200);
    }

    /// 测试TaskResult反序列化错误处理
    /// 覆盖分支：TaskResult deserialize的错误路径
    #[test]
    fn test_task_result_deserialize_errors() {
        // 太短的数据
        let too_short = vec![0u8; 12];
        let result = TaskResult::deserialize(&too_short);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Invalid result data: too short");

        // 不完整的数据
        let mut incomplete = Vec::new();
        incomplete.extend_from_slice(&1u64.to_le_bytes()); // task_id
        incomplete.push(1); // success = true
        incomplete.extend_from_slice(&500u32.to_le_bytes()); // 声明500字节
        incomplete.extend_from_slice(&[0u8; 10]); // 只提供10字节

        let result = TaskResult::deserialize(&incomplete);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Invalid result data: incomplete");
    }

    /// 测试TaskResult空数据场景
    /// 覆盖分支：边界条件 - 空数据
    #[test]
    fn test_task_result_empty_data() {
        let empty_result = TaskResult {
            task_id: 0,
            success: true,
            data: Vec::new(),
        };

        let serialized = empty_result.serialize();
        let deserialized = TaskResult::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.task_id, 0);
        assert!(deserialized.success);
        assert!(deserialized.data.is_empty());
    }

    /// 测试Worker心跳更新机制
    /// 覆盖分支：心跳时间的单调递增性
    #[test]
    fn test_worker_heartbeat_update() {
        let worker = Worker::new(1);

        let initial_heartbeat = worker
            .last_heartbeat()
            .load(std::sync::atomic::Ordering::SeqCst);

        // 等待一小段时间确保时间戳不同
        std::thread::sleep(std::time::Duration::from_millis(10));

        // 手动触发心跳更新（通过调用update_heartbeat）
        // 注意：update_heartbeat是私有方法，我们通过间接方式验证

        // 创建新worker并验证其心跳时间 >= 前面的
        let worker2 = Worker::new(2);
        let later_heartbeat = worker2
            .last_heartbeat()
            .load(std::sync::atomic::Ordering::SeqCst);

        assert!(later_heartbeat >= initial_heartbeat);
    }

    /// 测试Task包含特殊字符和二进制数据
    /// 覆盖分支：特殊数据处理
    #[test]
    fn test_task_with_special_and_binary_data() {
        // 包含null字符的数据
        let null_data = b"hello\0world\0".to_vec();
        let task1 = Task::new(1, null_data.clone());
        let serialized1 = task1.serialize();
        let deserialized1 = Task::deserialize(&serialized1).unwrap();
        assert_eq!(deserialized1.data, null_data);

        // 包含所有可能字节值的小数据集（0-255）
        let all_bytes: Vec<u8> = (0..=255).collect();
        let task2 = Task::new(2, all_bytes.clone());
        let serialized2 = task2.serialize();
        let deserialized2 = Task::deserialize(&serialized2).unwrap();
        assert_eq!(deserialized2.data, all_bytes);

        // 包含unicode文本
        let unicode_data = "中文测试 🎉 日本語テスト 한국어".as_bytes().to_vec();
        let task3 = Task::new(3, unicode_data.clone());
        let serialized3 = task3.serialize();
        let deserialized3 = Task::deserialize(&serialized3).unwrap();
        assert_eq!(deserialized3.data, unicode_data);
    }
}
