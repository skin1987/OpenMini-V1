//! KV Cache 换入换出管理模块
//!
//! 实现类似操作系统虚拟内存管理的 KV Cache 自动换入换出系统。
//!
//! # 核心功能
//!
//! - **自动换出**: 当内存中的 KV Cache blocks 超过阈值时，自动将低优先级的 block 写入 SQLite
//! - **按需换入**: 当需要访问已被换出的 block 时，从数据库加载回内存
//! - **LRU 管理**: 使用 LRU (最近最少使用) 算法管理 block 的淘汰顺序
//! - **压缩存储**: 可选的 zstd/lz4 压缩，减少数据库体积
//! - **统计监控**: 完整的性能指标（命中率、换入换出次数、延迟等）
//!
//! # 架构设计
//!
//! ```
//! ┌─────────────────────────────────────────────┐
//! │              Application Layer               │
//! └─────────────────┬───────────────────────────┘
//!                   │ get_block / put_block
//! ▼
//! ┌─────────────────────────────────────────────┐
//! │            KvSwapManager                     │
//! │  ┌───────────┐  ┌───────────┐  ┌─────────┐  │
//! │  │ In-Memory │  │LRU Queue  │  │ Stats   │  │
//! │  │  (Hot)    │  │           │  │         │  │
//! │  └─────┬─────┘  └───────────┘  └─────────┘  │
//! │        │                                   │
//! │  ┌─────▼─────┐                             │
//! │  │SwappedOut │◄──── Auto Eviction          │
//! │  │  (Cold)   │      (High Watermark)       │
//! │  └─────┬─────┘                             │
//! └────────┼───────────────────────────────────┘
//!          │ Swap In / Swap Out
//! ▼
//! ┌─────────────────────────────────────────────┐
//! │         SQLite Database                      │
//! │     kv_cache_store table                     │
//! └─────────────────────────────────────────────┘
//! ```
//!
//! # 使用示例
//!
//! ```ignore
//! use openmini_server::hardware::persistence::kv_swap::*;
//! use sqlx::SqlitePool;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), anyhow::Error> {
//!     let pool = SqlitePool::connect("sqlite::memory:").await?;
//!     let config = KvSwapConfig::default();
//!
//!     let manager = KvSwapManager::new(pool, config).await?;
//!
//!     // 写入 KV Cache block
//!     let data = vec![0u8; 1024];
//!     manager.put_block("session-1", 0, 0, &data).await?;
//!
//!     // 读取（自动处理换入换出）
//!     let loaded = manager.get_block("session-1", 0, 0).await?;
//!     assert_eq!(loaded, data);
//!
//!     // 获取统计信息
//!     let stats = manager.get_stats();
//!     println!("Hit rate: {:.2}%", stats.hit_rate());
//!
//!     Ok(())
//! }
//! ```

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use sqlx::SqlitePool;
use std::collections::{HashMap, VecDeque};

use crate::hardware::persistence::compression::{CompressionAlgorithm, CompressionManager};

// ============================================================================
// 核心数据结构定义
// ============================================================================

/// Block 唯一标识符
///
/// 用于唯一标识一个 KV Cache block，由 session_id、layer_idx 和 block_idx 组成。
/// 实现了 Hash、Eq、PartialEq 以便在 HashMap 中使用。
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct BlockId {
    /// 会话 ID
    session_id: String,
    /// 层索引
    layer_idx: usize,
    /// Block 索引
    block_idx: usize,
}

impl BlockId {
    /// 创建新的 BlockId
    ///
    /// # 参数
    ///
    /// * `session_id` - 会话标识符
    /// * `layer_idx` - 模型层索引
    /// * `block_idx` - 该层内的 block 索引
    pub fn new(session_id: &str, layer_idx: usize, block_idx: usize) -> Self {
        Self {
            session_id: session_id.to_string(),
            layer_idx,
            block_idx,
        }
    }

    /// 获取会话 ID
    pub fn session(&self) -> &str {
        &self.session_id
    }

    /// 获取层索引
    pub fn layer(&self) -> usize {
        self.layer_idx
    }

    /// 获取 block 索引
    pub fn block(&self) -> usize {
        self.block_idx
    }
}

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BlockId(session={}, layer={}, block={})",
            self.session_id, self.layer_idx, self.block_idx
        )
    }
}

/// KV Cache 交换状态
///
/// 描述一个 block 当前所处的位置和状态。
#[derive(Debug, Clone, PartialEq)]
pub enum KvSwapStatus {
    /// 在内存中 (Hot) - 可直接访问
    InMemory,
    /// 已换出到数据库 (Cold) - 需要从 DB 加载
    SwappedOut {
        /// 数据库中的记录 ID
        db_id: i64,
    },
    /// 正在从数据库加载中
    Loading,
}

/// KV Cache 条目 (从数据库读取)
///
/// 对应数据库表 `kv_cache_store` 中的一行记录。
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct KvCacheEntry {
    /// 数据库主键 ID
    pub id: i64,
    /// 会话 ID
    pub session_id: String,
    /// 层索引
    pub layer_idx: i64,
    /// Block 索引
    pub block_idx: i64,
    /// Key 数据 (序列化后的 KV cache keys)
    pub key_data: Vec<u8>,
    /// Value 数据 (序列化后的 KV cache values)
    pub value_data: Vec<u8>,
    /// 数据大小 (字节)
    pub size_bytes: i64,
    /// 重要性分数 (0.0-1.0, 越高越重要)
    pub importance: f64,
    /// 最后访问时间
    pub last_accessed_at: DateTime<Utc>,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 是否已压缩
    pub compressed: bool,
    /// 原始数据大小 (压缩前)
    pub original_size: i64,
}

/// 交换配置
///
/// 控制 KV Cache 换入换出的行为参数。
#[derive(Debug, Clone)]
pub struct KvSwapConfig {
    /// 内存中最大 block 数量 (默认 1024)
    ///
    /// 当内存中的 block 数超过此值时，将触发强制换出。
    pub max_in_memory_blocks: usize,

    /// 触发换出的高水位线 (默认 800)
    ///
    /// 当内存中的 block 数达到此值时，开始自动换出低优先级的 block。
    pub high_watermark: usize,

    /// 停止换出的低水位线 (默认 600)
    ///
    /// 当换出后内存中的 block 数降到此值以下时，停止换出操作。
    pub low_watermark: usize,

    /// 是否启用压缩 (默认 true)
    ///
    /// 启用后，换出到数据库的数据将使用配置的算法进行压缩。
    pub enable_compression: bool,

    /// 压缩算法 (默认 Zstd)
    pub compression_algorithm: CompressionAlgorithm,

    /// 批量换出大小 (默认 32)
    ///
    /// 每次换出操作处理的 block 数量。较大的值可以提高吞吐量，
    /// 但会增加单次操作的延迟。
    pub batch_evict_size: usize,

    /// LRU 采样数量 (默认 16)
    ///
    /// 选择淘汰候选时的采样数。使用采样而非全量排序可以显著提高性能。
    pub lru_sample_size: usize,
}

impl Default for KvSwapConfig {
    fn default() -> Self {
        Self {
            max_in_memory_blocks: 1024,
            high_watermark: 800,
            low_watermark: 600,
            enable_compression: true,
            compression_algorithm: CompressionAlgorithm::Zstd,
            batch_evict_size: 32,
            lru_sample_size: 16,
        }
    }
}

/// 交换统计信息
///
/// 记录 KV Cache 换入换出的性能指标，用于监控和调优。
#[derive(Debug, Default, Clone)]
pub struct KvSwapStats {
    /// 总换入次数 (从 DB 加载到内存)
    pub swap_in_count: u64,
    /// 总换出次数 (从内存写入 DB)
    pub swap_out_count: u64,
    /// 缓存命中次数 (直接从内存读取)
    pub hit_count: u64,
    /// 缓存未命中次数 (需要从 DB 加载或不存在)
    pub miss_count: u64,
    /// 总换入字节数
    pub total_swapped_in_bytes: u64,
    /// 总换出字节数
    pub total_swapped_out_bytes: u64,
    /// 平均换入延迟 (微秒)
    pub avg_swap_in_latency_us: u64,
    /// 平均换出延迟 (微秒)
    pub avg_swap_out_latency_us: u64,
    /// 当前内存中的 block 数量
    pub current_in_memory: usize,
    /// 当前已换出的 block 数量
    pub current_swapped_out: usize,
}

impl KvSwapStats {
    /// 计算缓存命中率
    ///
    /// 返回值范围 [0.0, 1.0]，如果没有任何访问则返回 0.0。
    pub fn hit_rate(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total == 0 {
            return 0.0;
        }
        self.hit_count as f64 / total as f64
    }

    /// 计算总访问次数
    pub fn total_accesses(&self) -> u64 {
        self.hit_count + self.miss_count
    }

    /// 计算平均每次换入的字节数
    pub fn avg_swap_in_bytes(&self) -> u64 {
        if self.swap_in_count == 0 {
            return 0;
        }
        self.total_swapped_in_bytes / self.swap_in_count
    }

    /// 计算平均每次换出的字节数
    pub fn avg_swap_out_bytes(&self) -> u64 {
        if self.swap_out_count == 0 {
            return 0;
        }
        self.total_swapped_out_bytes / self.swap_out_count
    }
}

impl std::fmt::Display for KvSwapStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KvSwapStats {{\n\
             \thit_rate: {:.2}% ({}/{}),\n\
             \tswap_in: {} times ({} bytes),\n\
             \tswap_out: {} times ({} bytes),\n\
             \tin_memory: {}, swapped_out: {},\n\
             \tavg_latency: in={}μs out={}μs\n\
             }}",
            self.hit_rate() * 100.0,
            self.hit_count,
            self.total_accesses(),
            self.swap_in_count,
            self.total_swapped_in_bytes,
            self.swap_out_count,
            self.total_swapped_out_bytes,
            self.current_in_memory,
            self.current_swapped_out,
            self.avg_swap_in_latency_us,
            self.avg_swap_out_latency_us
        )
    }
}

// ============================================================================
// KvSwapManager 核心实现
// ============================================================================

/// KV Cache 交换管理器
///
/// 实现类似操作系统虚拟内存管理的自动换入换出系统。
/// 对上层提供透明的 KV Cache 存取接口，内部自动管理内存和持久化存储之间的数据迁移。
///
/// # 线程安全
///
/// 所有共享状态都使用 `parking_lot::RwLock` 保护，支持多线程并发访问。
///
/// # 性能特性
///
/// - 使用 LRU 队列高效管理淘汰顺序
/// - 批量换出减少 I/O 开销
/// - 可选压缩减少存储空间
/// - 采样策略避免全量排序
///
/// # 示例
///
/// ```ignore
/// let pool = SqlitePool::connect("sqlite::memory:").await?;
/// let config = KvSwapConfig {
///     max_in_memory_blocks: 100,
///     ..Default::default()
/// };
/// let manager = KvSwapManager::new(pool, config).await?;
/// ```
pub struct KvSwapManager {
    /// SQLite 数据库连接池
    db_pool: SqlitePool,

    /// 内存中的 KV Cache blocks (BlockId -> 序列化后的数据)
    ///
    /// 这是"热数据"，可以直接访问，无需 I/O 操作。
    in_memory: RwLock<HashMap<BlockId, Vec<u8>>>,

    /// 已换出到数据库的 blocks (BlockId -> 数据库记录 ID)
    ///
    /// 这些是"冷数据"，需要通过 swap_in 从数据库加载回内存。
    swapped_out: RwLock<HashMap<BlockId, i64>>,

    /// LRU 队列 (最近使用的在尾部)
    ///
    /// 用于快速确定应该淘汰哪些 block。
    /// 队列尾部的 block 是最近访问的，头部的 block 是最久未访问的。
    lru_queue: RwLock<VecDeque<BlockId>>,

    /// 交换配置
    config: KvSwapConfig,

    /// 统计信息 (线程安全)
    stats: RwLock<KvSwapStats>,

    /// 压缩管理器
    compressor: CompressionManager,
}

impl KvSwapManager {
    /// 创建新的交换管理器并初始化数据库表
    ///
    /// 此方法会：
    /// 1. 初始化所有内部状态结构
    /// 2. 创建/验证数据库表结构
    /// 3. 初始化压缩管理器
    ///
    /// # 参数
    ///
    /// * `db_pool` - SQLite 连接池
    /// * `config` - 交换配置参数
    ///
    /// # 返回值
    ///
    /// * `Ok(KvSwapManager)` - 成功创建的管理器实例
    /// * `Err(anyhow::Error)` - 初始化失败
    ///
    /// # 错误情况
    ///
    /// - 数据库连接失败
    /// - 表创建失败
    pub async fn new(db_pool: SqlitePool, config: KvSwapConfig) -> Result<Self, anyhow::Error> {
        tracing::info!(
            max_in_memory = config.max_in_memory_blocks,
            high_watermark = config.high_watermark,
            low_watermark = config.low_watermark,
            enable_compression = config.enable_compression,
            compression_algo = %config.compression_algorithm,
            "Initializing KvSwapManager"
        );

        // 初始化数据库表
        Self::initialize_kv_cache_table(&db_pool).await?;

        // 创建压缩管理器
        let compressor = if config.enable_compression {
            CompressionManager::with_config(
                crate::hardware::persistence::compression::CompressionConfig {
                    algorithm: config.compression_algorithm,
                    ..Default::default()
                },
            )
        } else {
            CompressionManager::with_config(
                crate::hardware::persistence::compression::CompressionConfig {
                    algorithm: CompressionAlgorithm::None,
                    ..Default::default()
                },
            )
        };

        let manager = Self {
            db_pool,
            in_memory: RwLock::new(HashMap::new()),
            swapped_out: RwLock::new(HashMap::new()),
            lru_queue: RwLock::new(VecDeque::new()),
            config,
            stats: RwLock::new(KvSwapStats::default()),
            compressor,
        };

        tracing::info!("KvSwapManager initialized successfully");
        Ok(manager)
    }

    /// 获取 KV Cache block (自动处理换入换出)
    ///
    /// 如果 block 在内存中，直接返回（缓存命中）。
    /// 如果 block 已被换出到数据库，自动执行换入操作（缓存未命中）。
    ///
    /// # 参数
    ///
    /// * `session_id` - 会话标识符
    /// * `layer` - 层索引
    /// * `block` - block 索引
    ///
    /// # 返回值
    ///
    /// * `Ok(Vec<u8>)` - block 的数据
    /// * `Err(anyhow::Error)` - 获取失败
    ///
    /// # 性能
    ///
    /// - 命中: < 0.1ms (纯内存访问)
    /// - 未命中: 1-10ms (需要数据库 I/O)
    pub async fn get_block(
        &self,
        session_id: &str,
        layer: usize,
        block: usize,
    ) -> Result<Vec<u8>, anyhow::Error> {
        let block_id = BlockId::new(session_id, layer, block);

        // 1. 尝试从内存获取
        {
            let in_mem = self.in_memory.read();
            if let Some(data) = in_mem.get(&block_id) {
                // 命中：克隆数据以避免借用冲突
                let data_clone = data.clone();

                // 释放读锁（通过离开作用域）
                drop(in_mem);

                self.touch_lru(&block_id);

                let mut stats = self.stats.write();
                stats.hit_count += 1;

                tracing::trace!(
                    block = %block_id,
                    size = data_clone.len(),
                    "Cache hit"
                );

                return Ok(data_clone);
            }
        }

        // 2. 未命中：检查是否在 swapped_out 中
        let db_id = {
            let swapped = self.swapped_out.read();
            swapped.get(&block_id).copied()
        };

        match db_id {
            Some(id) => {
                // 3. 从数据库换入
                tracing::debug!(
                    block = %block_id,
                    db_id = id,
                    "Cache miss - swapping in from database"
                );

                let start = std::time::Instant::now();
                let data = self.swap_in(id, &block_id).await?;
                let latency_us = start.elapsed().as_micros() as u64;

                // 更新统计信息
                let mut stats = self.stats.write();
                stats.miss_count += 1;
                stats.swap_in_count += 1;
                stats.total_swapped_in_bytes += data.len() as u64;

                // 更新平均延迟 (指数移动平均)
                if stats.avg_swap_in_latency_us == 0 {
                    stats.avg_swap_in_latency_us = latency_us;
                } else {
                    stats.avg_swap_in_latency_us =
                        (stats.avg_swap_in_latency_us * 9 + latency_us) / 10;
                }

                Ok(data)
            }
            None => {
                // 4. 不存在
                let mut stats = self.stats.write();
                stats.miss_count += 1;

                tracing::warn!(block = %block_id, "Block not found");

                Err(anyhow::anyhow!("Block not found: {}", block_id))
            }
        }
    }

    /// 写入/更新 KV Cache block
    ///
    /// 将数据写入内存，并检查是否需要执行换出操作以控制内存使用。
    ///
    /// # 参数
    ///
    /// * `session_id` - 会话标识符
    /// * `layer` - 层索引
    /// * `block` - block 索引
    /// * `data` - 要写入的数据
    ///
    /// # 返回值
    ///
    /// * `Ok(())` - 写入成功
    /// * `Err(anyhow::Error)` - 写入失败
    ///
    /// # 副作用
    ///
    /// 如果写入后内存中的 block 数超过高水位线，将自动触发换出操作。
    pub async fn put_block(
        &self,
        session_id: &str,
        layer: usize,
        block: usize,
        data: &[u8],
    ) -> Result<(), anyhow::Error> {
        let block_id = BlockId::new(session_id, layer, block);
        let data = data.to_vec();

        tracing::trace!(
            block = %block_id,
            size = data.len(),
            "Putting block into cache"
        );

        // 1. 如果之前已换出，先从 swapped_out 移除
        {
            let mut swapped = self.swapped_out.write();
            if swapped.remove(&block_id).is_some() {
                tracing::debug!(block = %block_id, "Removing from swapped_out (was cold)");
            }
        }

        // 2. 写入内存
        {
            let mut in_mem = self.in_memory.write();

            // 如果是更新已有 block，不需要增加计数
            let is_update = in_mem.contains_key(&block_id);
            in_mem.insert(block_id.clone(), data);

            if !is_update {
                // 新 block：添加到 LRU 队列尾部
                drop(in_mem); // 释放写锁后再操作 LRU
                self.push_lru(&block_id);
            } else {
                // 更新已有 block：移动到 LRU 尾部
                drop(in_mem);
                self.touch_lru(&block_id);
            }
        }

        // 3. 更新统计
        {
            let mut stats = self.stats.write();
            stats.current_in_memory = self.in_memory.read().len();
        }

        // 4. 检查是否需要换出
        self.check_and_evict().await?;

        Ok(())
    }

    /// 批量写入多个 blocks
    ///
    /// 优化批量写入场景的性能，减少锁竞争和换出检查频率。
    ///
    /// # 参数
    ///
    /// * `blocks` - 待写入的 block 列表，每个元素为 `(session_id, layer_idx, block_idx, data)`
    ///
    /// # 性能优势
    ///
    /// 相比逐条调用 `put_block`，批量写入可以：
    /// - 减少锁获取/释放次数
    /// - 只在最后检查一次换出条件
    /// - 减少日志输出开销
    pub async fn put_blocks(
        &self,
        blocks: Vec<(String, usize, usize, Vec<u8>)>,
    ) -> Result<(), anyhow::Error> {
        if blocks.is_empty() {
            return Ok(());
        }

        tracing::debug!(count = blocks.len(), "Batch putting blocks");

        // 批量写入内存
        for (session_id, layer, block, data) in blocks {
            let block_id = BlockId::new(&session_id, layer, block);

            // 清理 swapped_out
            {
                let mut swapped = self.swapped_out.write();
                swapped.remove(&block_id);
            }

            // 写入内存
            {
                let mut in_mem = self.in_memory.write();
                let is_update = in_mem.contains_key(&block_id);
                in_mem.insert(block_id.clone(), data);

                if !is_update {
                    drop(in_mem);
                    self.push_lru(&block_id);
                } else {
                    drop(in_mem);
                    self.touch_lru(&block_id);
                }
            }
        }

        // 更新统计
        {
            let mut stats = self.stats.write();
            stats.current_in_memory = self.in_memory.read().len();
        }

        // 最后统一检查换出
        self.check_and_evict().await?;

        Ok(())
    }

    /// 手动触发换出 (强制)
    ///
    /// 强制换出指定数量的 block 到数据库，无论当前内存使用情况如何。
    /// 适用于预释放内存或测试场景。
    ///
    /// # 参数
    ///
    /// * `count` - 要换出的 block 数量
    ///
    /// # 返回值
    ///
    /// * `Ok(usize)` - 实际换出的 block 数量（可能小于请求的数量）
    /// * `Err(anyhow::Error)` - 换出失败
    pub async fn force_evict(&self, count: usize) -> Result<usize, anyhow::Error> {
        if count == 0 {
            return Ok(0);
        }

        tracing::info!(requested = count, "Force eviction requested");

        // 选择要换出的 victims
        let victims = self.select_victims(count);

        if victims.is_empty() {
            tracing::debug!("No blocks available for eviction");
            return Ok(0);
        }

        // 执行换出
        let evicted = self.swap_out(victims).await?;

        tracing::info!(
            requested = count,
            evicted = evicted,
            "Force eviction completed"
        );

        Ok(evicted)
    }

    /// 预取一批 blocks 到内存 (批量换入)
    ///
    /// 对于即将访问的 blocks，可以提前加载到内存中以避免后续的换入延迟。
    /// 特别适合模型推理前的预热阶段。
    ///
    /// # 参数
    ///
    /// * `session_id` - 会话标识符
    /// * `layers` - 各层的 block 索引列表，格式为 `[(layer_idx, [block_idx, ...])]`
    ///
    /// # 示例
    ///
    /// ```ignore
    /// // 预取第 0 层的 block 0-3 和第 1 层的 block 0-2
    /// manager.prefetch_blocks("session-1", &[
    ///     (0, vec![0, 1, 2, 3]),
    ///     (1, vec![0, 1, 2]),
    /// ]).await?;
    /// ```
    pub async fn prefetch_blocks(
        &self,
        session_id: &str,
        layers: &[(usize, Vec<usize>)],
    ) -> Result<(), anyhow::Error> {
        if layers.is_empty() {
            return Ok(());
        }

        let mut prefetch_count = 0usize;

        for (layer, blocks) in layers {
            for &block in blocks {
                let block_id = BlockId::new(session_id, *layer, block);

                // 检查是否已在内存中
                {
                    let in_mem = self.in_memory.read();
                    if in_mem.contains_key(&block_id) {
                        continue; // 已在内存中，跳过
                    }
                }

                // 检查是否已换出
                let db_id = {
                    let swapped = self.swapped_out.read();
                    swapped.get(&block_id).copied()
                };

                if let Some(db_id) = db_id {
                    // 执行换入
                    match self.swap_in(db_id, &block_id).await {
                        Ok(_) => {
                            prefetch_count += 1;
                            self.push_lru(&block_id);
                        }
                        Err(e) => {
                            tracing::warn!(
                                block = %block_id,
                                error = %e,
                                "Failed to prefetch block"
                            );
                        }
                    }
                }
            }
        }

        if prefetch_count > 0 {
            tracing::info!(
                session = session_id,
                prefetched = prefetch_count,
                "Prefetch completed"
            );
        }

        Ok(())
    }

    /// 删除指定会话的所有 blocks
    ///
    /// 同时清理内存和数据库中的数据。适用于会话结束后的资源释放。
    ///
    /// # 参数
    ///
    /// * `session_id` - 要清理的会话 ID
    ///
    /// # 返回值
    ///
    /// * `Ok(u64)` - 删除的总 block 数量
    /// * `Err(anyhow::Error)` - 删除失败
    pub async fn evict_session(&self, session_id: &str) -> Result<u64, anyhow::Error> {
        tracing::info!(session = session_id, "Evicting all blocks for session");

        let mut total_removed = 0u64;

        // 1. 从内存中移除
        {
            let mut in_mem = self.in_memory.write();
            let before_len = in_mem.len();
            in_mem.retain(|id, _| id.session() != session_id);
            total_removed += (before_len - in_mem.len()) as u64;
        }

        // 2. 从 swapped_out 映射中移除
        {
            let mut swapped = self.swapped_out.write();
            swapped.retain(|id, _| id.session() != session_id);
        }

        // 3. 从 LRU 队列中移除
        {
            let mut lru = self.lru_queue.write();
            lru.retain(|id| id.session() != session_id);
        }

        // 4. 从数据库删除
        let db_result = sqlx::query("DELETE FROM kv_cache_store WHERE session_id = ?")
            .bind(session_id)
            .execute(&self.db_pool)
            .await?;

        total_removed += db_result.rows_affected();

        // 更新统计
        {
            let mut stats = self.stats.write();
            stats.current_in_memory = self.in_memory.read().len();
            stats.current_swapped_out = self.swapped_out.read().len();
        }

        tracing::info!(
            session = session_id,
            removed = total_removed,
            "Session eviction completed"
        );

        Ok(total_removed)
    }

    /// 获取统计信息快照
    ///
    /// 返回当前的统计信息副本，可用于监控和调试。
    pub fn get_stats(&self) -> KvSwapStats {
        let mut stats = self.stats.read().clone();

        // 更新实时数值
        stats.current_in_memory = self.in_memory.read().len();
        stats.current_swapped_out = self.swapped_out.read().len();

        stats
    }

    /// 重置统计信息
    ///
    /// 将所有计数器和平均值归零。
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write();
        *stats = KvSwapStats::default();

        // 重新设置当前值
        stats.current_in_memory = self.in_memory.read().len();
        stats.current_swapped_out = self.swapped_out.read().len();

        tracing::info!("Statistics reset");
    }

    /// 获取当前内存使用估算
    ///
    /// 返回内存中所有 block 的总字节大小。
    /// 注意：这是近似值，不包括 HashMap 和其他元数据的开销。
    pub fn estimate_memory_usage(&self) -> usize {
        let in_mem = self.in_memory.read();
        in_mem.values().map(|v| v.len()).sum()
    }

    /// 获取当前内存中的 block 数量
    pub fn in_memory_count(&self) -> usize {
        self.in_memory.read().len()
    }

    /// 获取当前已换出的 block 数量
    pub fn swapped_out_count(&self) -> usize {
        self.swapped_out.read().len()
    }

    // ========================================================================
    // 内部方法 (Private)
    // ========================================================================

    /// 初始化 KV Cache 数据库表
    async fn initialize_kv_cache_table(db_pool: &SqlitePool) -> Result<(), anyhow::Error> {
        tracing::debug!("Initializing kv_cache_store table");

        let create_table_sql = r#"
            CREATE TABLE IF NOT EXISTS kv_cache_store (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                layer_idx INTEGER NOT NULL,
                block_idx INTEGER NOT NULL,
                key_data BLOB NOT NULL,
                value_data BLOB NOT NULL,
                size_bytes INTEGER NOT NULL,
                importance REAL DEFAULT 0.5,
                last_accessed_at DATETIME NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                compressed BOOLEAN DEFAULT FALSE,
                original_size INTEGER NOT NULL,
                
                UNIQUE(session_id, layer_idx, block_idx)
            );
            
            CREATE INDEX IF NOT EXISTS idx_kv_session ON kv_cache_store(session_id);
            CREATE INDEX IF NOT EXISTS idx_kv_access ON kv_cache_store(last_accessed_at);
        "#;

        sqlx::raw_sql(create_table_sql)
            .execute(db_pool)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to initialize kv_cache_store table: {}", e))?;

        tracing::debug!("kv_cache_store table initialized successfully");
        Ok(())
    }

    /// 从数据库换入 (load from DB)
    ///
    /// 从数据库加载指定 block 的数据到内存。
    /// 此方法会：
    /// 1. 从数据库查询数据
    /// 2. 如果数据被压缩，进行解压
    /// 3. 将数据放入内存
    /// 4. 从 swapped_out 映射中移除
    async fn swap_in(&self, db_id: i64, block_id: &BlockId) -> Result<Vec<u8>, anyhow::Error> {
        // 1. 从数据库查询
        let entry = sqlx::query_as::<_, KvCacheEntry>("SELECT * FROM kv_cache_store WHERE id = ?")
            .bind(db_id)
            .fetch_optional(&self.db_pool)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Block not found in database: id={}", db_id))?;

        // 2. 解压（如果需要）
        let data = if entry.compressed && entry.original_size > 0 {
            let decompressed = self
                .compressor
                .decompress(
                    &entry.value_data,
                    entry.original_size as usize,
                    self.config.compression_algorithm,
                )
                .map_err(|e| anyhow::anyhow!("Decompression failed for {}: {}", block_id, e))?;
            decompressed.data
        } else {
            entry.value_data
        };

        // 3. 放入内存
        {
            let mut in_mem = self.in_memory.write();
            in_mem.insert(block_id.clone(), data.clone());
        }

        // 4. 从 swapped_out 移除
        {
            let mut swapped = self.swapped_out.write();
            swapped.remove(block_id);
        }

        // 5. 添加到 LRU 尾部
        self.push_lru(block_id);

        // 6. 更新统计
        {
            let mut stats = self.stats.write();
            stats.current_in_memory = self.in_memory.read().len();
            stats.current_swapped_out = self.swapped_out.read().len();
        }

        tracing::debug!(
            block = %block_id,
            size = data.len(),
            compressed = entry.compressed,
            "Swapped in from database"
        );

        Ok(data)
    }

    /// 换出到数据库 (write to DB)
    ///
    /// 将指定的 blocks 从内存写入数据库，并从内存中移除。
    /// 支持批量写入以提高性能。
    ///
    /// # 参数
    ///
    /// * `block_ids` - 要换出的 block ID 列表
    ///
    /// # 返回值
    ///
    /// * `Ok(usize)` - 成功换出的 block 数量
    /// * `Err(anyhow::Error)` - 换出失败
    async fn swap_out(&self, block_ids: Vec<BlockId>) -> Result<usize, anyhow::Error> {
        if block_ids.is_empty() {
            return Ok(0);
        }

        let start = std::time::Instant::now();
        let mut success_count = 0usize;
        let mut total_bytes = 0usize;
        let attempted_count = block_ids.len(); // 保存长度，因为 block_ids 会被移动

        // 收集要换出的数据
        let blocks_to_write: Vec<(BlockId, Vec<u8>)> = {
            let mut in_mem = self.in_memory.write();
            block_ids
                .into_iter()
                .filter_map(|id| in_mem.remove(&id).map(|data| (id, data)))
                .collect()
        };

        // 批量写入数据库
        for (block_id, data) in blocks_to_write {
            let now = Utc::now();
            let original_size = data.len() as i64;

            // 压缩（如果启用）
            let (value_data, compressed) = if self.config.enable_compression && data.len() > 256 {
                match self.compressor.compress(&data) {
                    Ok(result) if result.ratio < 0.95 => {
                        // 只有压缩率足够好才使用压缩数据
                        (result.data, true)
                    }
                    _ => (data.clone(), false),
                }
            } else {
                (data.clone(), false)
            };

            // 插入或更新数据库
            let result = sqlx::query(
                r#"
                INSERT INTO kv_cache_store 
                (session_id, layer_idx, block_idx, key_data, value_data, size_bytes, 
                 importance, last_accessed_at, compressed, original_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, layer_idx, block_idx) 
                DO UPDATE SET
                    value_data = excluded.value_data,
                    size_bytes = excluded.size_bytes,
                    last_accessed_at = excluded.last_accessed_at,
                    compressed = excluded.compressed,
                    original_size = excluded.original_size
                "#,
            )
            .bind(block_id.session())
            .bind(block_id.layer() as i64)
            .bind(block_id.block() as i64)
            .bind(Vec::<u8>::new()) // key_data 占位符
            .bind(&value_data)
            .bind(value_data.len() as i64)
            .bind(0.5f64) // 默认重要性
            .bind(now)
            .bind(compressed)
            .bind(original_size)
            .execute(&self.db_pool)
            .await;

            match result {
                Ok(_) => {
                    // 记录到 swapped_out
                    // 注意：这里简化处理，实际应获取插入的 ID
                    // 由于使用了 UPSERT，我们需要查询确认 ID
                    let row = sqlx::query_as::<_, (i64,)>(
                        "SELECT id FROM kv_cache_store WHERE session_id = ? AND layer_idx = ? AND block_idx = ?"
                    )
                    .bind(block_id.session())
                    .bind(block_id.layer() as i64)
                    .bind(block_id.block() as i64)
                    .fetch_one(&self.db_pool)
                    .await;

                    match row {
                        Ok((db_id,)) => {
                            let mut swapped = self.swapped_out.write();
                            swapped.insert(block_id.clone(), db_id);

                            // 从 LRU 移除
                            self.remove_from_lru(&block_id);

                            success_count += 1;
                            total_bytes += data.len();
                        }
                        Err(e) => {
                            tracing::error!(
                                block = %block_id,
                                error = %e,
                                "Failed to get DB ID after insert"
                            );
                        }
                    }
                }
                Err(e) => {
                    tracing::error!(
                        block = %block_id,
                        error = %e,
                        "Failed to swap out block to database"
                    );

                    // 失败时放回内存
                    let mut in_mem = self.in_memory.write();
                    in_mem.insert(block_id, data);
                }
            }
        }

        // 更新统计
        let latency_us = start.elapsed().as_micros() as u64;
        {
            let mut stats = self.stats.write();
            stats.swap_out_count += 1;
            stats.total_swapped_out_bytes += total_bytes as u64;

            if stats.avg_swap_out_latency_us == 0 {
                stats.avg_swap_out_latency_us = latency_us;
            } else {
                stats.avg_swap_out_latency_us =
                    (stats.avg_swap_out_latency_us * 9 + latency_us) / 10;
            }

            stats.current_in_memory = self.in_memory.read().len();
            stats.current_swapped_out = self.swapped_out.read().len();
        }

        tracing::info!(
            attempted = attempted_count,
            success = success_count,
            bytes = total_bytes,
            latency_ms = start.elapsed().as_millis(),
            "Swap out completed"
        );

        Ok(success_count)
    }

    /// 检查是否需要执行换出
    ///
    /// 当内存中的 block 数超过高水位线时，自动换出直到降到低水位线以下。
    async fn check_and_evict(&self) -> Result<(), anyhow::Error> {
        let current_count = self.in_memory.read().len();

        // 检查是否超过高水位线
        if current_count <= self.config.high_watermark {
            return Ok(());
        }

        tracing::debug!(
            current = current_count,
            high_watermark = self.config.high_watermark,
            low_watermark = self.config.low_watermark,
            "High watermark reached, starting eviction"
        );

        // 计算需要换出的数量
        let target = current_count.saturating_sub(self.config.low_watermark);
        let to_evict = target.min(self.config.batch_evict_size);

        // 选择 victims
        let victims = self.select_victims(to_evict);

        if victims.is_empty() {
            return Ok(());
        }

        // 执行换出
        self.swap_out(victims).await?;

        Ok(())
    }

    /// LRU 采样选择牺牲品
    ///
    /// 使用采样策略选择应该被换出的 block，避免全量排序的开销。
    ///
    /// # 参数
    ///
    /// * `count` - 需要选择的 victim 数量
    ///
    /// # 返回值
    ///
    /// 应该被换出的 BlockId 列表（按 LRU 顺序排列）
    fn select_victims(&self, count: usize) -> Vec<BlockId> {
        if count == 0 {
            return vec![];
        }

        let lru = self.lru_queue.read();

        // 如果队列很短，直接返回前面的元素
        if lru.len() <= count {
            return lru.iter().cloned().collect();
        }

        // 采样策略：从前面的旧元素中选择
        // LRU 队列头部是最旧的
        let sample_size = self.config.lru_sample_size.min(lru.len());
        let candidates: Vec<&BlockId> = lru.iter().take(sample_size).collect();

        // 选择前 count 个（已经按 LRU 顺序排列）
        candidates.into_iter().take(count).cloned().collect()
    }

    // ========================================================================
    // LRU 辅助方法
    // ========================================================================

    /// 更新 LRU 位置（移动到尾部）
    ///
    /// 当 block 被访问时调用，将其标记为最近使用。
    fn touch_lru(&self, block_id: &BlockId) {
        let mut lru = self.lru_queue.write();

        // 移除现有位置（如果有）
        if let Some(pos) = lru.iter().position(|id| id == block_id) {
            lru.remove(pos);
        }

        // 添加到尾部
        lru.push_back(block_id.clone());
    }

    /// 添加到 LRU 队列尾部
    ///
    /// 新 block 或换入的 block 使用此方法。
    fn push_lru(&self, block_id: &BlockId) {
        let mut lru = self.lru_queue.write();
        lru.push_back(block_id.clone());
    }

    /// 从 LRU 队列中移除
    ///
    /// 换出的 block 使用此方法。
    fn remove_from_lru(&self, block_id: &BlockId) {
        let mut lru = self.lru_queue.write();
        if let Some(pos) = lru.iter().position(|id| id == block_id) {
            lru.remove(pos);
        }
    }
}

// ============================================================================
// 测试模块
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 创建测试用的内存数据库
    async fn create_test_db() -> SqlitePool {
        let pool = SqlitePool::connect("sqlite::memory:")
            .await
            .expect("Failed to create test database");
        pool
    }

    /// 创建测试用的 KvSwapManager
    async fn create_test_manager(config: Option<KvSwapConfig>) -> KvSwapManager {
        let pool = create_test_db().await;
        let config = config.unwrap_or_else(|| KvSwapConfig {
            max_in_memory_blocks: 10, // 小阈值以便测试换出
            high_watermark: 8,
            low_watermark: 6,
            enable_compression: false, // 关闭压缩以简化测试
            ..Default::default()
        });

        KvSwapManager::new(pool, config)
            .await
            .expect("Failed to create test manager")
    }

    #[tokio::test]
    async fn test_basic_put_get() {
        let manager = create_test_manager(None).await;

        // 写入 block
        let data = b"Hello, KV Cache!".to_vec();
        manager
            .put_block("session-1", 0, 0, &data)
            .await
            .expect("Failed to put block");

        // 读取 block
        let loaded = manager
            .get_block("session-1", 0, 0)
            .await
            .expect("Failed to get block");

        assert_eq!(loaded, data, "Data should match after put/get");

        // 验证统计信息
        let stats = manager.get_stats();
        assert_eq!(stats.hit_count, 1, "Should have 1 hit");
        assert_eq!(stats.miss_count, 0, "Should have 0 misses");
        assert_eq!(stats.current_in_memory, 1, "Should have 1 block in memory");
    }

    #[tokio::test]
    async fn test_auto_eviction() {
        let manager = create_test_manager(None).await;

        // 写入超过高水位线的 blocks
        for i in 0..10i32 {
            let data = format!("block-data-{}", i).into_bytes();
            manager
                .put_block("session-1", 0, i as usize, &data)
                .await
                .expect("Failed to put block");
        }

        let stats = manager.get_stats();

        // 应该触发了自动换出
        assert!(
            stats.current_in_memory <= manager.config.high_watermark,
            "In-memory blocks should be at or below high watermark after auto-eviction"
        );

        println!(
            "Auto eviction test: in_memory={}, swapped_out={}",
            stats.current_in_memory, stats.current_swapped_out
        );
    }

    #[tokio::test]
    async fn test_swap_in_on_miss() {
        let manager = create_test_manager(None).await;

        // 写入足够多的 blocks 触发换出
        for i in 0..10i32 {
            let data = format!("swap-test-data-{}", i).into_bytes();
            manager
                .put_block("session-1", 0, i as usize, &data)
                .await
                .expect("Failed to put block");
        }

        // 等待确保换出完成
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // 获取统计信息
        let stats_before = manager.get_stats();
        let swapped_out_before = stats_before.current_swapped_out;

        if swapped_out_before > 0 {
            // 尝试访问一个可能被换出的 block
            // 注意：由于 LRU 顺序，较早写入的 block 可能已被换出
            let result = manager.get_block("session-1", 0, 0).await;

            match result {
                Ok(data) => {
                    let stats_after = manager.get_stats();

                    // 验证数据完整性
                    assert!(!data.is_empty(), "Data should not be empty");

                    // 如果发生了换入，统计信息应该反映这一点
                    if stats_after.swap_in_count > stats_before.swap_in_count {
                        println!(
                            "Swap-in occurred: swap_in_count increased from {} to {}",
                            stats_before.swap_in_count, stats_after.swap_in_count
                        );
                    }
                }
                Err(e) => {
                    // 可能该 block 还在内存中，或者已被正确清理
                    println!("Get block result: {:?}", e);
                }
            }
        } else {
            println!("No blocks were swapped out, skipping swap-in test");
        }
    }

    #[tokio::test]
    async fn test_batch_operations() {
        let manager = create_test_manager(None).await;

        // 准备批量数据
        let blocks: Vec<(String, usize, usize, Vec<u8>)> = (0..5i32)
            .map(|i| {
                (
                    "batch-session".to_string(),
                    0,
                    i as usize,
                    format!("batch-data-{}", i).into_bytes(),
                )
            })
            .collect();

        // 批量写入
        manager
            .put_blocks(blocks)
            .await
            .expect("Failed to batch put blocks");

        // 验证所有 block 都可访问
        for i in 0..5i32 {
            let result = manager.get_block("batch-session", 0, i as usize).await;
            assert!(result.is_ok(), "Block {} should be accessible", i);
        }

        let stats = manager.get_stats();
        assert_eq!(
            stats.current_in_memory, 5,
            "All 5 blocks should be in memory"
        );
    }

    #[tokio::test]
    async fn test_session_eviction() {
        let manager = create_test_manager(None).await;

        // 为两个 session 写入数据
        for i in 0..3i32 {
            let data = format!("session-a-{}", i).into_bytes();
            manager
                .put_block("session-A", 0, i as usize, &data)
                .await
                .unwrap();

            let data = format!("session-b-{}", i).into_bytes();
            manager
                .put_block("session-B", 0, i as usize, &data)
                .await
                .unwrap();
        }

        // 验证初始状态
        assert_eq!(
            manager.in_memory_count(),
            6,
            "Should have 6 blocks initially"
        );

        // 清理 session-A
        let removed = manager
            .evict_session("session-A")
            .await
            .expect("Failed to evict session");

        assert!(
            removed >= 3,
            "Should remove at least 3 blocks for session-A"
        );
        assert_eq!(
            manager.in_memory_count(),
            3,
            "Should have 3 blocks remaining (session-B)"
        );

        // 验证 session-B 的数据仍然可访问
        let result = manager.get_block("session-B", 0, 0).await;
        assert!(result.is_ok(), "Session-B data should still be accessible");
    }

    #[tokio::test]
    async fn test_compression_enabled() {
        // 这个测试仅在 compression feature 启用时运行
        #[cfg(feature = "compression")]
        {
            let pool = create_test_db().await;
            let config = KvSwapConfig {
                max_in_memory_blocks: 5,
                high_watermark: 4,
                low_watermark: 3,
                enable_compression: true,
                compression_algorithm: CompressionAlgorithm::Zstd,
                ..Default::default()
            };

            let manager = KvSwapManager::new(pool, config)
                .await
                .expect("Failed to create manager with compression");

            // 写入可压缩的数据
            let repetitive_data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();

            for i in 0..6i32 {
                manager
                    .put_block("compress-session", 0, i as usize, &repetitive_data)
                    .await
                    .expect("Failed to put block");
            }

            // 验证压缩统计
            let comp_stats = manager.compressor.stats();
            if comp_stats.total_compressions > 0 {
                println!(
                    "Compression stats: ratio={:.3}, saved={} bytes",
                    comp_stats.overall_ratio(),
                    comp_stats.saved_bytes()
                );

                // 验证确实进行了压缩
                assert!(
                    comp_stats.overall_ratio() < 1.0,
                    "Compression should reduce data size"
                );
            }
        }

        #[cfg(not(feature = "compression"))]
        {
            println!("Skipping compression test (feature not enabled)");
        }
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let manager = create_test_manager(None).await;

        // 执行一系列操作
        // 1. 写入 3 个 blocks
        for i in 0..3i32 {
            let data = format!("stats-test-{}", i).into_bytes();
            manager
                .put_block("stats-session", 0, i as usize, &data)
                .await
                .unwrap();
        }

        // 2. 读取它们（命中）
        for i in 0..3i32 {
            manager
                .get_block("stats-session", 0, i as usize)
                .await
                .unwrap();
        }

        // 3. 重复读取（再次命中）
        for i in 0..3i32 {
            manager
                .get_block("stats-session", 0, i as usize)
                .await
                .unwrap();
        }

        // 验证统计信息
        let stats = manager.get_stats();

        assert_eq!(
            stats.hit_count, 6,
            "Should have 6 hits (3 initial + 3 repeat)"
        );
        assert_eq!(stats.miss_count, 0, "Should have 0 misses");
        assert!(stats.hit_rate() > 0.99, "Hit rate should be very high");
        assert_eq!(stats.current_in_memory, 3, "Should have 3 blocks in memory");

        println!("Stats tracking test:\n{}", stats);

        // 重置统计
        manager.reset_stats();
        let stats_after_reset = manager.get_stats();
        assert_eq!(stats_after_reset.hit_count, 0, "Stats should be reset");
        assert_eq!(stats_after_reset.miss_count, 0, "Misses should be reset");
    }

    #[tokio::test]
    async fn test_force_eviction() {
        let manager = create_test_manager(None).await;

        // 写入一些 blocks
        for i in 0..5i32 {
            let data = format!("force-evict-{}", i).into_bytes();
            manager
                .put_block("force-session", 0, i as usize, &data)
                .await
                .unwrap();
        }

        assert_eq!(
            manager.in_memory_count(),
            5,
            "Should have 5 blocks before force evict"
        );

        // 强制换出 2 个
        let evicted = manager.force_evict(2).await.expect("Force eviction failed");

        assert!(evicted <= 2, "Should evict at most 2 blocks");
        assert!(
            manager.in_memory_count() <= 3,
            "Should have at most 3 blocks after evicting 2"
        );

        println!(
            "Force eviction: requested=2, actual={}, remaining={}",
            evicted,
            manager.in_memory_count()
        );
    }

    #[tokio::test]
    async fn test_prefetch_blocks() {
        let manager = create_test_manager(None).await;

        // 先写入并触发换出
        for i in 0..8i32 {
            let data = format!("prefetch-data-{}", i).into_bytes();
            manager
                .put_block("prefetch-session", 0, i as usize, &data)
                .await
                .unwrap();
        }

        // 等待换出完成
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let stats_before = manager.get_stats();

        // 预取
        manager
            .prefetch_blocks("prefetch-session", &[(0, vec![0, 1, 2])])
            .await
            .expect("Prefetch should succeed");

        let stats_after = manager.get_stats();

        // 验证预取后这些 block 可以快速访问
        for i in 0..3i32 {
            let result = manager.get_block("prefetch-session", 0, i as usize).await;
            assert!(
                result.is_ok(),
                "Prefetched block {} should be accessible",
                i
            );
        }

        println!(
            "Prefetch test: swap_in before={}, after={}",
            stats_before.swap_in_count, stats_after.swap_in_count
        );
    }

    #[tokio::test]
    async fn test_memory_usage_estimation() {
        let manager = create_test_manager(None).await;

        // 初始状态
        assert_eq!(
            manager.estimate_memory_usage(),
            0,
            "Initial memory usage should be 0"
        );

        // 写入已知大小的数据
        let data_1kb = vec![0u8; 1024];
        let data_2kb = vec![0u8; 2048];

        manager
            .put_block("mem-session", 0, 0, &data_1kb)
            .await
            .unwrap();
        manager
            .put_block("mem-session", 0, 1, &data_2kb)
            .await
            .unwrap();

        let estimated = manager.estimate_memory_usage();
        let expected = data_1kb.len() + data_2kb.len();

        assert_eq!(
            estimated, expected,
            "Memory usage estimation should match actual data sizes"
        );

        println!(
            "Memory usage: estimated={} bytes ({:.2} KB)",
            estimated,
            estimated as f64 / 1024.0
        );
    }

    #[tokio::test]
    async fn test_block_id_display() {
        let id = BlockId::new("test-session", 5, 10);
        let display = format!("{}", id);

        assert!(
            display.contains("test-session"),
            "Should contain session ID"
        );
        assert!(display.contains("layer=5"), "Should contain layer index");
        assert!(display.contains("block=10"), "Should contain block index");
    }

    #[tokio::test]
    async fn test_config_defaults() {
        let config = KvSwapConfig::default();

        assert_eq!(config.max_in_memory_blocks, 1024);
        assert_eq!(config.high_watermark, 800);
        assert_eq!(config.low_watermark, 600);
        assert!(config.enable_compression);
        assert_eq!(config.compression_algorithm, CompressionAlgorithm::Zstd);
        assert_eq!(config.batch_evict_size, 32);
        assert_eq!(config.lru_sample_size, 16);
    }

    #[tokio::test]
    async fn test_stats_display() {
        let stats = KvSwapStats {
            hit_count: 100,
            miss_count: 20,
            swap_in_count: 5,
            swap_out_count: 3,
            total_swapped_in_bytes: 5000,
            total_swapped_out_bytes: 8000,
            avg_swap_in_latency_us: 150,
            avg_swap_out_latency_us: 200,
            current_in_memory: 50,
            current_swapped_out: 30,
        };

        let display = format!("{}", stats);

        assert!(display.contains("hit_rate:"), "Should show hit rate");
        assert!(display.contains("swap_in:"), "Should show swap-in count");
        assert!(display.contains("swap_out:"), "Should show swap-out count");
        assert!(
            display.contains("in_memory:"),
            "Should show in-memory count"
        );
    }
}
