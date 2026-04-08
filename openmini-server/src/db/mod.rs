//! SQLite 数据库模块
//!
//! 提供完整的 SQLite 数据库支持，包括：
//! - 连接池管理（WAL 模式、性能优化）
//! - 会话表 CRUD 操作
//! - 消息表 CRUD 操作
//! - 记忆表 CRUD 操作（替代 HashMap 存储）
//!
//! # 使用示例
//!
//! ```ignore
//! use openmini_server::db::{DatabaseConfig, create_pool};
//!
//! let config = DatabaseConfig::default();
//! let pool = create_pool(&config).await?;
//! ```

pub mod memory;
pub mod message;
pub mod pool;
pub mod session;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// 数据库配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// 数据库文件路径
    pub path: PathBuf,
    /// 连接池大小
    pub pool_size: u32,
    /// 忙等待超时时间（毫秒）
    pub busy_timeout_ms: u64,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("data/openmini.db"),
            pool_size: 10,
            busy_timeout_ms: 5000,
        }
    }
}
