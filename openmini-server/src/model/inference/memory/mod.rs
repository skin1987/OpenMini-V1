//! 三级记忆网络（DMN）模块
//!
//! 提供三级记忆机制：
//! 1. 瞬时记忆：当前推理批次的 KV Cache
//! 2. 短期记忆：滑动窗口记忆，支持多轮对话
//! 3. 长期记忆：持久化知识库，支持向量检索
//!
//! # 架构
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    MemoryManager                            │
//! ├─────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
//! │  │   Instant   │  │   Short    │  │    Long     │        │
//! │  │   Memory    │  │   Term     │  │    Term     │        │
//! │  │  (KV Cache) │  │  Memory    │  │  Memory     │        │
//! │  │             │  │  (滑动窗口)│  │  (向量存储) │        │
//! │  └─────────────┘  └─────────────┘  └─────────────┘        │
//! └─────────────────────────────────────────────────────────────┘
//! ```

#![warn(missing_docs)]
#![allow(dead_code)]

use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// 记忆级别枚举
///
/// 区分三级记忆层次：
/// - `Instant`: 瞬时记忆，当前推理批次的 KV Cache
/// - `ShortTerm`: 短期记忆，滑动窗口机制
/// - `LongTerm`: 长期记忆，持久化向量存储
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryLevel {
    /// 瞬时记忆：当前批次
    Instant,
    /// 短期记忆：滑动窗口
    ShortTerm,
    /// 长期记忆：持久化
    LongTerm,
}

impl MemoryLevel {
    /// 获取记忆级别的字符串表示
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryLevel::Instant => "instant",
            MemoryLevel::ShortTerm => "short_term",
            MemoryLevel::LongTerm => "long_term",
        }
    }
}

/// 记忆系统配置
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// 瞬时记忆容量
    pub instant_capacity: usize,
    /// 短期记忆容量
    pub short_term_capacity: usize,
    /// 长期记忆容量
    pub long_term_capacity: usize,
    /// 压缩阈值
    pub compression_threshold: usize,
    /// 驱逐策略
    pub eviction_strategy: EvictionStrategy,
    /// 嵌入向量维度
    ///
    /// 预留接口：向量维度填充功能将在后续版本实现
    pub embedding_dim: Option<usize>,
    /// 嵌入向量填充策略
    ///
    /// 预留接口：向量维度填充功能将在后续版本实现
    pub padding_strategy: PaddingStrategy,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            instant_capacity: 4096,
            short_term_capacity: 1024,
            long_term_capacity: 10000,
            compression_threshold: 512,
            eviction_strategy: EvictionStrategy::LRU,
            embedding_dim: None,
            padding_strategy: PaddingStrategy::default(),
        }
    }
}

/// 驱逐策略枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum EvictionStrategy {
    /// 最近最少使用
    LRU,
    /// 最不经常使用
    LFU,
    /// 先进先出
    FIFO,
}

/// 嵌入向量填充策略
///
/// 用于将任意维度向量填充到模型所需维度
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaddingStrategy {
    /// 循环取模填充：embedding[[0, j]] = mean[[0, j % dim]]
    /// 可能引入周期性模式，但对相似度计算影响有限
    #[default]
    Cyclic,
    /// 零填充：超出原始维度的位置填充 0
    Zero,
}

/// 记忆项结构体
///
/// 存储单个记忆项的数据和元信息
///
/// # 字段说明
/// - `data`: 嵌入向量数据
/// - `timestamp`: 记忆产生时间（原始时间戳）
/// - `importance`: 重要性分数
/// - `session_id`: 所属会话 ID
#[derive(Debug, Clone)]
pub struct MemoryItem {
    /// 嵌入向量数据
    pub data: Array2<f32>,
    /// 记忆产生时间（原始时间戳）
    /// 注意：短期记忆内部使用 `write_timestamp` 计算时间衰减
    pub timestamp: u64,
    /// 重要性分数
    pub importance: f32,
    /// 所属会话 ID
    pub session_id: Option<u64>,
}

impl MemoryItem {
    /// 创建新的记忆项
    pub fn new(data: Array2<f32>, timestamp: u64) -> Self {
        Self {
            data,
            timestamp,
            importance: 1.0,
            session_id: None,
        }
    }

    /// 设置重要性分数
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance;
        self
    }

    /// 设置会话 ID
    pub fn with_session(mut self, session_id: u64) -> Self {
        self.session_id = Some(session_id);
        self
    }
}

pub mod hnsw;
pub mod instant;
pub mod long_term;
pub mod manager;
pub mod persistence;
pub mod short_term;
pub mod simd_ops;

/// 引擎桥接模块（内部实现）
///
/// 提供推理引擎与记忆系统之间的数据转换接口
pub(crate) mod engine_bridge;

/// 缓存桥接模块（内部实现）
///
/// 提供 KV Cache 与记忆系统之间的映射和同步接口
pub(crate) mod cache_bridge;

#[allow(unused_imports)]
pub use hnsw::HNSWIndex;
pub use instant::InstantMemory;
pub use long_term::LongTermMemory;
#[allow(unused_imports)]
pub use manager::{DMNMetrics, MemoryManager};
pub use short_term::ShortTermMemory;
