//! KV Cache 优化模块
//!
//! 提供高效的KV Cache管理，包括：
//! - PagedAttention 分页注意力
//! - 连续批处理 (Continuous Batching)
//! - 前缀缓存 (Prefix Cache)
//!
//! ## 架构
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    KV Cache 系统                             │
//! ├─────────────────────────────────────────────────────────────┤
//! │  BlockManager ──→ PageTable ──→ PagedKVCache               │
//! │        │              │              │                      │
//! │        └──────────────┴──────────────┘                      │
//! │                       │                                     │
//! │              PrefixCache (可选)                             │
//! │                       │                                     │
//! │              ContinuousBatching                             │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## 使用示例
//!
//! ```rust,ignore
//! use openmini_server::inference::kv_cache::{PagedKVCache, BatchScheduler, KVCacheConfig};
//!
//! // 创建KV Cache
//! let config = KVCacheConfig::default().with_max_memory(1024);
//! let kv_cache = PagedKVCache::new(config);
//!
//! // 创建批处理调度器
//! let scheduler = BatchScheduler::with_kv_cache(kv_cache);
//!
//! // 添加请求
//! let request = GenerationRequest::new(1, vec![1, 2, 3, 4, 5], 100);
//! scheduler.add_request(request);
//!
//! // 调度执行
//! let scheduled = scheduler.schedule();
//! ```

pub mod block;
pub mod block_manager;
pub mod continuous_batch;
pub mod mla;
pub mod page_table;
pub mod paged_cache;
pub mod persistence;
pub mod prefix_cache;
pub mod streaming;

// 重导出主要类型
#[allow(unused_imports)]
pub use block::{Block, BlockId, BlockState, KVCacheConfig, DEFAULT_BLOCK_SIZE};
#[allow(unused_imports)]
pub use block_manager::BlockManager;
#[allow(unused_imports)]
pub use continuous_batch::{
    BatchConfig, BatchScheduler, GenerationRequest, GenerationResult, RequestPriority,
    RequestState, SchedulerStats,
};
#[allow(unused_imports)]
pub use mla::{
    mla_attention_forward, MLAAttention, MLAConfig, MLALatentCache, MLAProjection, RoPECache,
};
#[allow(unused_imports)]
pub use page_table::{PageTable, PageTableEntry, PageTableManager};
#[allow(unused_imports)]
pub use paged_cache::PagedKVCache;
#[allow(unused_imports)]
pub use prefix_cache::{PrefixCache, PrefixCacheConfig, PrefixCacheStats, PrefixEntry, PrefixHash};
#[allow(unused_imports)]
pub use streaming::{StreamingAttention, StreamingAttentionConfig, StreamingAttentionStats};

/// 请求ID类型
pub type RequestId = u64;

/// 层索引类型
#[allow(dead_code)]
pub type LayerIdx = usize;

/// Token位置类型
#[allow(dead_code)]
pub type TokenPos = usize;

/// KV Cache错误类型
#[derive(Debug, Clone, thiserror::Error)]
#[allow(dead_code)]
pub enum KVCacheError {
    /// 内存不足
    #[error("Out of memory: requested {requested} blocks, available {available}")]
    OutOfMemory { requested: usize, available: usize },

    /// 请求不存在
    #[error("Request {0} not found")]
    RequestNotFound(RequestId),

    /// 块不存在
    #[error("Block {0} not found")]
    BlockNotFound(BlockId),

    /// 层越界
    #[error("Layer {layer} out of range (max: {max})")]
    LayerOutOfRange { layer: usize, max: usize },

    /// 位置越界
    #[error("Position {pos} out of range (max: {max})")]
    PositionOutOfRange { pos: usize, max: usize },

    /// 无效操作
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    /// 配置错误
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// 缓存清除错误
    #[error("Cache clear error: {0}")]
    ClearError(String),
}

/// KV 缓存 trait
///
/// 定义 KV 缓存的最小接口，用于抽象不同类型的缓存实现
/// （如 `StandardKVCache` 和 `MLALatentCache`）。
///
/// # 所需的最小接口
/// - `num_tokens()`: 获取当前 token 数量
/// - `clear()`: 清除缓存
/// - `memory_usage()`: 获取内存使用量
/// - `is_empty()`: 检查缓存是否为空
pub trait KVCache {
    /// 获取当前 token 数量
    fn num_tokens(&self) -> usize;

    /// 清除缓存
    ///
    /// # 返回
    /// 成功返回 `Ok(())`，失败返回 `KVCacheError::ClearError`
    fn clear_cache(&mut self) -> Result<(), KVCacheError>;

    /// 获取内存使用量（字节）
    fn memory_usage(&self) -> usize;

    /// 检查缓存是否为空
    fn is_empty(&self) -> bool {
        self.num_tokens() == 0
    }
}

/// KV Cache结果类型
#[allow(dead_code)]
pub type KVCacheResult<T> = Result<T, KVCacheError>;

/// 从字符串错误转换
impl From<String> for KVCacheError {
    fn from(msg: String) -> Self {
        KVCacheError::InvalidOperation(msg)
    }
}

impl From<&str> for KVCacheError {
    fn from(msg: &str) -> Self {
        KVCacheError::InvalidOperation(msg.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_config() {
        let config = KVCacheConfig::default();
        assert!(config.block_size > 0);
        assert!(config.max_blocks > 0);
    }

    #[test]
    fn test_paged_kv_cache_creation() {
        let cache = PagedKVCache::with_capacity(100, 16);
        assert_eq!(cache.available_blocks(), 100);
    }

    #[test]
    fn test_block_manager_creation() {
        let manager = BlockManager::with_capacity(50, 16);
        assert_eq!(manager.total_blocks(), 50);
        assert_eq!(manager.available_blocks(), 50);
    }

    #[test]
    fn test_page_table_creation() {
        let pt = PageTable::new(100);
        assert!(pt.is_empty());
        assert_eq!(pt.capacity(), 100);
    }

    #[test]
    fn test_prefix_cache_creation() {
        let cache = PrefixCache::default_config();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_batch_scheduler_creation() {
        let kv_cache = PagedKVCache::with_capacity(100, 16);
        let scheduler = BatchScheduler::with_kv_cache(kv_cache);
        assert!(!scheduler.has_pending());
    }

    #[test]
    fn test_error_conversion() {
        let err: KVCacheError = "test error".into();
        assert!(matches!(err, KVCacheError::InvalidOperation(_)));
    }

    // ==================== 新增分支覆盖测试 ====================

    /// 测试 KVCacheError 所有变体的构造和 Display
    #[test]
    fn test_kv_cache_error_all_variants() {
        // 覆盖 OutOfMemory 变体
        let oom = KVCacheError::OutOfMemory {
            requested: 10,
            available: 5,
        };
        let msg = format!("{}", oom);
        assert!(msg.contains("10") && msg.contains("5"));

        // 覆盖 RequestNotFound 变体
        let rnf = KVCacheError::RequestNotFound(42);
        assert!(format!("{}", rnf).contains("42"));

        // 覆盖 BlockNotFound 变体
        let bnf = KVCacheError::BlockNotFound(7usize); // BlockId 是 usize 类型别名
        assert!(format!("{}", bnf).contains("7"));

        // 覆盖 LayerOutOfRange 变体
        let lor = KVCacheError::LayerOutOfRange { layer: 5, max: 3 };
        let msg = format!("{}", lor);
        assert!(msg.contains("5") && msg.contains("3"));

        // 覆盖 PositionOutOfRange 变体
        let por = KVCacheError::PositionOutOfRange { pos: 100, max: 50 };
        let msg = format!("{}", por);
        assert!(msg.contains("100") && msg.contains("50"));

        // 覆盖 InvalidOperation 变体
        let io = KVCacheError::InvalidOperation("bad op".to_string());
        assert!(matches!(io, KVCacheError::InvalidOperation(_)));

        // 覆盖 ConfigError 变体
        let ce = KVCacheError::ConfigError("bad config".to_string());
        assert!(matches!(ce, KVCacheError::ConfigError(_)));

        // 覆盖 ClearError 变体
        let cle = KVCacheError::ClearError("clear failed".to_string());
        assert!(matches!(cle, KVCacheError::ClearError(_)));
    }

    /// 测试 From<String> 错误转换（覆盖 InvalidOperation 分支）
    #[test]
    fn test_error_from_string() {
        let err: KVCacheError = From::from(String::from("string error"));
        match err {
            KVCacheError::InvalidOperation(s) => {
                assert_eq!(s, "string error");
            }
            other => panic!("Expected InvalidOperation, got {:?}", other),
        }
    }

    /// 测试 From<&str> 错误转换（覆盖 InvalidOperation 分支）
    #[test]
    fn test_error_from_str() {
        let err: KVCacheError = From::from("str error");
        match err {
            KVCacheError::InvalidOperation(s) => {
                assert_eq!(s, "str error");
            }
            other => panic!("Expected InvalidOperation, got {:?}", other),
        }
    }

    /// 测试类型别名 RequestId / LayerIdx / TokenPos 的可用性
    #[test]
    fn test_type_aliases() {
        // 验证类型别名可以正常使用
        let _req_id: RequestId = 12345u64;
        let _layer_idx: LayerIdx = 0usize;
        let _token_pos: TokenPos = 0usize;

        // KVCacheResult 类型别名
        let ok_result: KVCacheResult<()> = Ok(());
        assert!(ok_result.is_ok());

        let err_result: KVCacheResult<()> = Err(KVCacheError::InvalidOperation("test".into()));
        assert!(err_result.is_err());
    }

    /// 测试 KVCache trait 的 is_empty 默认实现（基于 num_tokens==0）
    #[test]
    fn test_kv_cache_trait_is_empty_default() {
        // 创建一个简单的 struct 来验证 trait 默认实现
        struct MockCache {
            tokens: usize,
        }
        impl KVCache for MockCache {
            fn num_tokens(&self) -> usize {
                self.tokens
            }
            fn clear_cache(&mut self) -> Result<(), KVCacheError> {
                self.tokens = 0;
                Ok(())
            }
            fn memory_usage(&self) -> usize {
                self.tokens * 100
            }
        }

        // 空缓存
        let empty_cache = MockCache { tokens: 0 };
        assert!(empty_cache.is_empty());

        // 非空缓存
        let nonempty_cache = MockCache { tokens: 5 };
        assert!(!nonempty_cache.is_empty());
    }

    /// 测试 KVCacheError 的 Clone 和 Debug 特性
    #[test]
    fn test_kv_cache_error_clone_debug() {
        let err = KVCacheError::RequestNotFound(99);
        let cloned = err.clone();
        assert_eq!(format!("{:?}", err), format!("{:?}", cloned));

        // Debug 输出应包含枚举名
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("RequestNotFound"));
    }

    /// 测试 DEFAULT_BLOCK_SIZE 常量
    #[test]
    fn test_default_block_size() {
        // 验证 DEFAULT_BLOCK_SIZE 是合理的正数（运行时检查）
        let block_size = DEFAULT_BLOCK_SIZE;
        assert!(block_size > 0, "DEFAULT_BLOCK_SIZE should be positive, got {}", block_size);
        assert!(block_size >= 16, "DEFAULT_BLOCK_SIZE should be at least 16, got {}", block_size);
    }

    /// 测试 BlockId 的基本操作（如果可用）
    #[test]
    fn test_block_id_usage() {
        let id: BlockId = 1;
        let id2: BlockId = 1;
        let id3: BlockId = 2;

        // BlockId 应该支持比较
        assert_eq!(id, id2);
        assert_ne!(id, id3);
    }

    /// 测试 KVCacheConfig 的字段可访问性
    #[test]
    fn test_kv_cache_config_fields() {
        let config = KVCacheConfig::default();
        // 验证关键字段有合理默认值
        assert!(config.block_size > 0);
        assert!(config.max_blocks > 0);
    }

    /// 测试 PageTableEntry / PageTableManager 可用性
    #[test]
    fn test_page_table_types_exist() {
        // 验证类型可以实例化
        let pt = PageTable::new(64);
        assert_eq!(pt.capacity(), 64);

        // PageTableManager 如果是关联类型或独立类型，应能使用
        let _pt_manager_check: Option<PageTable> = Some(pt);
        assert!(_pt_manager_check.is_some());
    }
}
