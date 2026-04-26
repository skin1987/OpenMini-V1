//! 高性能推理 API 类型定义
//!
//! 定义直接调用 HighPerformancePipeline 的 REST API 接口。

use serde::{Deserialize, Serialize};
use ts_rs::TS;

/// 推理计算请求
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct InferenceRequest {
    /// Query 张量 [seq_len, num_heads * head_dim]
    pub query: Vec<Vec<f32>>,

    /// Key 张量 [total_seq_len, num_kv_heads * head_dim]
    pub key: Vec<Vec<f32>>,

    /// Value 张量 [total_seq_len, num_kv_heads * head_dim]
    pub value: Vec<Vec<f32>>,

    /// 注意力头数（可选，默认使用模型配置）
    #[serde(default)]
    pub num_heads: Option<usize>,

    /// 每个头维度（可选，默认使用模型配置）
    #[serde(default)]
    pub head_dim: Option<usize>,
}

/// 推理计算响应
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct InferenceResponse {
    /// 请求ID
    pub id: String,

    /// 输出张量 [seq_len, num_heads * head_dim]
    pub output: Vec<Vec<f32>>,

    /// 输出维度信息
    pub output_shape: Vec<usize>,

    /// 使用的注意力策略
    pub strategy: String,

    /// 推理时间（毫秒）
    pub inference_time_ms: f64,

    /// 性能统计
    pub stats: Option<InferenceStats>,
}

/// 推理性能统计
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct InferenceStats {
    /// 总处理 token 数
    pub total_tokens: usize,

    /// 每秒 token 数
    pub tokens_per_second: f32,

    /// KV Cache 利用率
    pub kv_cache_utilization: f32,

    /// 已使用块数
    pub blocks_used: usize,
}

/// Pipeline 配置响应
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct PipelineConfigResponse {
    /// 最大序列长度
    pub max_seq_len: usize,

    /// 注意力头数
    pub num_heads: usize,

    /// 头维度
    pub head_dim: usize,

    /// KV头数
    pub num_kv_heads: usize,

    /// 层数
    pub num_layers: usize,

    /// 是否启用 FlashAttention-3
    pub enable_fa3: bool,

    /// 是否启用 MLA
    pub enable_mla: bool,

    /// 是否启用 Streaming
    pub enable_streaming: bool,

    /// 流式切换阈值
    pub streaming_threshold: usize,

    /// KV Cache 信息
    pub kv_cache: KvCacheInfo,
}

/// KV Cache 状态信息
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct KvCacheInfo {
    /// 可用块数
    pub available_blocks: usize,

    /// 已分配块数
    pub allocated_blocks: usize,

    /// 利用率 (0.0-1.0)
    pub utilization: f32,

    /// 块大小
    pub block_size: usize,

    /// 最大块数
    pub max_blocks: usize,
}

/// 批量推理请求
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct BatchInferenceRequest {
    /// 多组输入
    pub requests: Vec<InferenceRequest>,
}

/// 批量推理响应
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct BatchInferenceResponse {
    /// 请求ID
    pub id: String,

    /// 各样本输出
    pub outputs: Vec<InferenceResponse>,

    /// 总时间（毫秒）
    pub total_time_ms: f64,

    /// 平均吞吐量
    pub avg_tokens_per_second: f32,
}
