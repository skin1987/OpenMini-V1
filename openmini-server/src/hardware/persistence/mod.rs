//! 持久化存储管理模块
//!
//! 作为硬件资源管理层的一部分，负责将内存中的数据持久化到 SQLite 数据库，
//! 实现类似操作系统虚拟内存的换入换出机制。
//!
//! # 核心功能
//!
//! - **KV Cache 换入换出**: 当内存不足时自动将 KV Cache block 换出到数据库
//! - **模型权重分页**: 支持超大模型的按需加载
//! - **内存页交换**: 提供虚拟内存扩展能力
//! - **统一淘汰策略**: LRU/LFU/Adaptive 等多种算法
//!
//! # 架构定位
//!
//! ```
//! hardware/
//! ├── cpu/           ← CPU 资源管理
//! ├── gpu/           ← GPU 资源管理
//! ├── memory/        ← RAM 内存管理 (现有)
//! └── persistence/   ← 🔥 持久化存储管理 (本模块)
//!     ├── database.rs       # SQLite 连接池
//!     ├── kv_swap.rs        # KV Cache 换入换出
//!     ├── eviction_policy.rs # 统一淘汰策略
//!     └── compression.rs    # 数据压缩
//! ```
//!
//! # 设计理念
//!
//! 类似操作系统的虚拟内存管理系统：
//! - Hot 数据在 RAM (< 0.1ms 访问)
//! - Warm/Cold 数据在 SQLite (1-10ms 访问)
//! - 自动换入换出，对上层透明

pub mod compression;
pub mod database;
pub mod eviction_policy;
pub mod kv_swap;

pub use database::{DatabaseManager, DatabaseConfig, PersistenceError};
pub use compression::CompressionManager;
pub use eviction_policy::{
    EvictionAlgorithm, EvictionPolicy,
};
pub use kv_swap::{
    KvSwapConfig, KvSwapManager, KvSwapStats,
};

/// Persistence Layer 版本信息
pub const PERSISTENCE_VERSION: &str = "1.0.0";

/// 默认数据库文件名
pub const DEFAULT_DB_NAME: &str = "openmini_persistence.db";
