//! 分布式推理通信协议
//!
//! 定义节点间的消息类型和序列化格式。
//! 所有消息使用 serde 进行序列化，支持 bincode 高效传输。

use serde::{Deserialize, Serialize};

/// 消息类型枚举
///
/// 定义分布式推理系统中所有节点间通信的消息类型，
/// 包括心跳检测、任务分配、结果返回等。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributedMessage {
    /// 心跳检测
    ///
    /// 工作节点定期向协调器发送心跳，报告自身状态
    Heartbeat {
        node_id: String,
        timestamp: u64,
        gpu_utilization: f32,
        memory_used_mb: u64,
    },

    /// 任务分配
    ///
    /// 协调器将推理任务分配给工作节点
    TaskAssign {
        task_id: String,
        payload: InferenceRequest,
        priority: u8,
    },

    /// 任务结果
    ///
    /// 工作节点完成推理后，将结果返回给协调器
    TaskResult {
        task_id: String,
        result: InferenceResponse,
        execution_time_us: u64,
    },

    /// 节点注册
    ///
    /// 新工作节点加入集群时发送注册信息
    NodeRegister {
        node_id: String,
        capabilities: NodeCapabilities,
    },

    /// 错误报告
    ///
    /// 任务执行失败时发送错误信息
    Error {
        task_id: String,
        error_code: u32,
        message: String,
    },
}

/// 推理请求
///
/// 包含完整的推理任务参数，从客户端传递到工作节点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// 会话 ID，用于关联同一对话的多次请求
    pub session_id: String,
    /// 模型名称
    pub model_name: String,
    /// 输入 token 序列（已 tokenize）
    pub input_tokens: Vec<u32>,
    /// 最大生成 token 数
    pub max_tokens: usize,
    /// 采样温度参数 (0.0-2.0)
    pub temperature: f32,
}

/// 推理响应
///
/// 包含推理结果和性能统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    /// 输出 token 序列
    pub output_tokens: Vec<u32>,
    /// 每个 token 的 log 概率
    pub logprobs: Vec<f32>,
    /// 结束原因 ("length" | "stop" | "eos")
    pub finish_reason: String,
    /// 推理性能统计
    pub stats: InferenceStats,
}

/// 推理统计信息
///
/// 记录单次推理的性能指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceStats {
    /// 首个 token 延迟 (ms)
    pub ttft_ms: f32,
    /// Token 生成速度 (tokens/s)
    pub tokens_per_second: f32,
    /// 总推理时间 (ms)
    pub total_time_ms: f32,
    /// GPU 显存占用 (MB)
    pub gpu_memory_mb: u64,
}

/// 节点能力描述
///
/// 描述工作节点的硬件能力和支持的模型列表
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    /// GPU 型号名称 (如 "NVIDIA A100", "Apple M1 Pro")
    pub gpu_name: String,
    /// GPU 显存大小 (MB)
    pub vram_size_mb: u64,
    /// 最大批处理大小
    pub max_batch_size: usize,
    /// 支持的模型名称列表
    pub supported_models: Vec<String>,
}

impl InferenceRequest {
    /// 创建新的推理请求
    pub fn new(
        session_id: impl Into<String>,
        model_name: impl Into<String>,
        input_tokens: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            model_name: model_name.into(),
            input_tokens,
            max_tokens,
            temperature,
        }
    }

    /// 获取输入序列长度
    pub fn input_len(&self) -> usize {
        self.input_tokens.len()
    }
}

impl InferenceResponse {
    /// 创建成功的推理响应
    pub fn success(
        output_tokens: Vec<u32>,
        logprobs: Vec<f32>,
        finish_reason: impl Into<String>,
        stats: InferenceStats,
    ) -> Self {
        Self {
            output_tokens,
            logprobs,
            finish_reason: finish_reason.into(),
            stats,
        }
    }

    /// 获取输出序列长度
    pub fn output_len(&self) -> usize {
        self.output_tokens.len()
    }
}

impl InferenceStats {
    /// 创建空的统计信息（用于测试）
    #[cfg(test)]
    pub fn empty() -> Self {
        Self {
            ttft_ms: 0.0,
            tokens_per_second: 0.0,
            total_time_ms: 0.0,
            gpu_memory_mb: 0,
        }
    }
}

impl NodeCapabilities {
    /// 创建节点能力描述
    pub fn new(
        gpu_name: impl Into<String>,
        vram_size_mb: u64,
        max_batch_size: usize,
        supported_models: Vec<String>,
    ) -> Self {
        Self {
            gpu_name: gpu_name.into(),
            vram_size_mb,
            max_batch_size,
            supported_models,
        }
    }

    /// 检查是否支持指定模型
    pub fn supports_model(&self, model_name: &str) -> bool {
        self.supported_models.contains(&model_name.to_string())
            || self.supported_models.contains(&"*".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_request_creation() {
        let request =
            InferenceRequest::new("session-001", "llama-7b", vec![1, 2, 3, 4, 5], 100, 0.7);

        assert_eq!(request.session_id, "session-001");
        assert_eq!(request.model_name, "llama-7b");
        assert_eq!(request.input_len(), 5);
        assert_eq!(request.max_tokens, 100);
        assert!((request.temperature - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_inference_response_creation() {
        let stats = InferenceStats::empty();
        let response =
            InferenceResponse::success(vec![10, 20, 30], vec![-0.5, -0.3, -0.1], "length", stats);

        assert_eq!(response.output_len(), 3);
        assert_eq!(response.finish_reason, "length");
        assert_eq!(response.logprobs.len(), 3);
    }

    #[test]
    fn test_node_capabilities_supports_model() {
        let caps = NodeCapabilities::new(
            "NVIDIA A100",
            80 * 1024, // 80GB
            32,
            vec!["llama-7b".to_string(), "llama-13b".to_string()],
        );

        assert!(caps.supports_model("llama-7b"));
        assert!(caps.supports_model("llama-13b"));
        assert!(!caps.supports_model("gpt-4"));

        // 通配符匹配
        let wildcard_caps = NodeCapabilities::new("GPU", 24 * 1024, 16, vec!["*".to_string()]);
        assert!(wildcard_caps.supports_model("any-model"));
    }

    #[test]
    fn test_message_serialization() {
        use bincode;

        let msg = DistributedMessage::Heartbeat {
            node_id: "worker-1".to_string(),
            timestamp: 1234567890,
            gpu_utilization: 0.85,
            memory_used_mb: 16384,
        };

        // 测试序列化和反序列化
        let encoded: Vec<u8> = bincode::serialize(&msg).expect("序列化失败");
        let decoded: DistributedMessage = bincode::deserialize(&encoded).expect("反序列化失败");

        match decoded {
            DistributedMessage::Heartbeat {
                node_id,
                timestamp,
                gpu_utilization,
                memory_used_mb,
            } => {
                assert_eq!(node_id, "worker-1");
                assert_eq!(timestamp, 1234567890);
                assert!((gpu_utilization - 0.85).abs() < f32::EPSILON);
                assert_eq!(memory_used_mb, 16384);
            }
            _ => panic!("消息类型不匹配"),
        }
    }

    #[test]
    fn test_task_assign_message() {
        let request = InferenceRequest::new("sess-1", "model-1", vec![1, 2], 50, 0.5);
        let msg = DistributedMessage::TaskAssign {
            task_id: "task-001".to_string(),
            payload: request,
            priority: 1,
        };

        match &msg {
            DistributedMessage::TaskAssign {
                task_id, priority, ..
            } => {
                assert_eq!(*task_id, "task-001");
                assert_eq!(*priority, 1);
            }
            _ => panic!("消息类型错误"),
        }
    }
}
