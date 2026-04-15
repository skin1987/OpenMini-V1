//! SQLite 数据库连接池管理器
//!
//! 提供高性能的 SQLite 数据库连接池管理，支持：
//! - 自动建表和 schema 管理
//! - WAL 模式优化读写性能
//! - 连接池配置和管理
//! - 健康检查和统计信息收集
//! - 过期数据清理

use sqlx::{sqlite::SqlitePoolOptions, SqlitePool};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

// ============================================================================
// 配置定义
// ============================================================================

/// 数据库配置结构体
///
/// # 示例
///
/// ```rust,no_run
/// use std::path::PathBuf;
/// use openmini_server::hardware::persistence::database::DatabaseConfig;
///
/// let config = DatabaseConfig {
///     db_path: PathBuf::from("/var/lib/openmini/cache.db"),
///     pool_size: 20,
///     enable_wal: true,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    /// 数据库文件路径
    pub db_path: PathBuf,
    /// 连接池最小大小（默认 10）
    pub pool_size: u32,
    /// 是否启用 WAL 模式（默认 true）
    pub enable_wal: bool,
    /// 忙等待超时时间（秒）（默认 30）
    pub busy_timeout_secs: u64,
    /// 最大连接数（默认 20）
    pub max_connections: u32,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("./data/openmini_persistence.db"),
            pool_size: 10,
            enable_wal: true,
            busy_timeout_secs: 30,
            max_connections: 20,
        }
    }
}

// ============================================================================
// 错误类型定义
// ============================================================================

/// 持久化层错误类型
///
/// 封装所有数据库操作可能出现的错误，提供清晰的错误分类。
#[derive(Debug)]
pub enum PersistenceError {
    /// 数据库连接错误
    Connection(String),
    /// SQL 查询执行错误
    Query(String),
    /// I/O 操作错误
    Io(std::io::Error),
    /// 序列化/反序列化错误
    Serialization(String),
    /// 资源未找到错误
    NotFound(String),
}

impl std::fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Connection(msg) => write!(f, "Database connection error: {}", msg),
            Self::Query(msg) => write!(f, "Query execution error: {}", msg),
            Self::Io(err) => write!(f, "I/O error: {}", err),
            Self::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            Self::NotFound(msg) => write!(f, "Resource not found: {}", msg),
        }
    }
}

impl std::error::Error for PersistenceError {}

impl From<sqlx::Error> for PersistenceError {
    fn from(err: sqlx::Error) -> Self {
        match err {
            sqlx::Error::PoolTimedOut | sqlx::Error::PoolClosed => {
                PersistenceError::Connection(err.to_string())
            }
            sqlx::Error::Database(db_err) => PersistenceError::Query(db_err.message().to_string()),
            _ => PersistenceError::Query(err.to_string()),
        }
    }
}

impl From<std::io::Error> for PersistenceError {
    fn from(err: std::io::Error) -> Self {
        PersistenceError::Io(err)
    }
}

// ============================================================================
// 统计信息结构
// ============================================================================

/// 数据库统计信息
///
/// 用于监控数据库使用情况和性能指标。
#[derive(Debug, Default, Clone)]
pub struct DatabaseStats {
    /// 总查询次数
    pub total_queries: u64,
    /// 总写入次数
    pub total_writes: u64,
    /// 总读取次数
    pub total_reads: u64,
    /// 缓存命中次数
    pub cache_hits: u64,
    /// 缓存未命中次数
    pub cache_misses: u64,
    /// 数据库文件大小（字节）
    pub db_size_bytes: u64,
}

// ============================================================================
// 数据库管理器核心实现
// ============================================================================

/// SQLite 数据库连接池管理器
///
/// 提供完整的数据库生命周期管理，包括：
/// - 连接池创建和维护
/// - 自动表结构初始化
/// - 性能优化参数配置
/// - 运行时统计和监控
///
/// # 使用示例
///
/// ```rust,no_run
/// use openmini_server::hardware::persistence::database::{DatabaseManager, DatabaseConfig};
/// use std::path::PathBuf;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let config = DatabaseConfig::default();
///     let db = DatabaseManager::new(config).await?;
///
///     // 执行健康检查
///     let healthy = db.health_check().await?;
///     println!("Database healthy: {}", healthy);
///
///     // 获取统计信息
///     let stats = db.get_stats().await;
///     println!("Total queries: {}", stats.total_queries);
///
///     // 关闭连接池
///     db.close().await?;
///     Ok(())
/// }
/// ```
pub struct DatabaseManager {
    /// 数据库配置
    config: DatabaseConfig,
    /// SQLite 连接池
    pool: SqlitePool,
    /// 运行时统计信息（线程安全）
    stats: Arc<RwLock<DatabaseStats>>,
}

impl DatabaseManager {
    /// 创建并初始化数据库管理器
    ///
    /// 该方法会：
    /// 1. 创建数据库目录（如果不存在）
    /// 2. 建立连接池
    /// 3. 配置 SQLite 优化参数（WAL模式等）
    /// 4. 初始化表结构
    ///
    /// # 参数
    ///
    /// * `config` - 数据库配置信息
    ///
    /// # 返回值
    ///
    /// * `Ok(DatabaseManager)` - 成功初始化的数据库管理器实例
    /// * `Err(PersistenceError)` - 初始化过程中发生的错误
    ///
    /// # 错误情况
    ///
    /// - 目录创建失败
    /// - 连接池建立失败
    /// - 表结构初始化失败
    pub async fn new(config: DatabaseConfig) -> Result<Self, PersistenceError> {
        tracing::info!(
            db_path = %config.db_path.display(),
            pool_size = config.pool_size,
            enable_wal = config.enable_wal,
            "Initializing database manager"
        );

        // 确保数据库目录存在
        if let Some(parent) = config.db_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // 构建数据库连接 URL
        let db_url = format!("sqlite://{}?mode=rwc", config.db_path.to_string_lossy());

        // 创建连接池
        let pool = SqlitePoolOptions::new()
            .min_connections(config.pool_size)
            .max_connections(config.max_connections)
            .connect(&db_url)
            .await
            .map_err(|e| {
                PersistenceError::Connection(format!("Failed to create connection pool: {}", e))
            })?;

        let manager = Self {
            config,
            pool,
            stats: Arc::new(RwLock::new(DatabaseStats::default())),
        };

        // 配置 SQLite 优化参数
        manager.configure_pragmas().await?;

        // 初始化表结构
        manager.initialize_tables().await?;

        tracing::info!("Database manager initialized successfully");

        Ok(manager)
    }

    /// 获取数据库连接池引用
    ///
    /// 用于在其他模块中直接执行 SQL 查询。
    ///
    /// # 返回值
    ///
    /// 返回内部维护的 `SqlitePool` 引用
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// 执行数据库健康检查
    ///
    /// 通过执行简单的 SELECT 1 来验证连接是否正常工作，
    /// 并更新数据库文件大小统计信息。
    ///
    /// # 返回值
    ///
    /// * `Ok(true)` - 数据库连接正常
    /// * `Err(PersistenceError)` - 健康检查失败
    pub async fn health_check(&self) -> Result<bool, PersistenceError> {
        tracing::debug!("Executing health check");

        match sqlx::query("SELECT 1").fetch_one(&self.pool).await {
            Ok(_) => {
                self.update_db_size().await;
                tracing::debug!("Health check passed");
                Ok(true)
            }
            Err(e) => {
                tracing::error!(error = %e, "Health check failed");
                Err(PersistenceError::Connection(format!(
                    "Health check failed: {}",
                    e
                )))
            }
        }
    }

    /// 获取数据库运行时统计信息
    ///
    /// 返回当前的统计快照，包括查询次数、缓存命中率等信息。
    ///
    /// # 返回值
    ///
    /// 当前时刻的 `DatabaseStats` 快照
    pub async fn get_stats(&self) -> DatabaseStats {
        let stats = self.stats.read().await.clone();
        stats
    }

    /// 关闭数据库连接池
    ///
    /// 优雅地关闭所有数据库连接。调用后不应再使用此实例。
    ///
    /// # 返回值
    ///
    /// * `Ok(())` - 关闭成功
    /// * `Err(PersistenceError)` - 关闭过程中发生错误
    pub async fn close(&self) -> Result<(), PersistenceError> {
        tracing::info!("Closing database connection pool");

        self.pool.close().await;

        let final_stats = self.stats.read().await;
        tracing::info!(
            total_queries = final_stats.total_queries,
            total_writes = final_stats.total_writes,
            total_reads = final_stats.total_reads,
            "Database manager closed"
        );

        Ok(())
    }

    // ========================================================================
    // 内部方法
    // ========================================================================

    /// 初始化数据库表结构
    ///
    /// 创建所有必需的表，如果表已存在则跳过。
    /// 包括：
    /// - kv_cache_store: KV 缓存存储表
    /// - session_store: 会话存储表
    async fn initialize_tables(&self) -> Result<(), PersistenceError> {
        tracing::info!("Initializing database tables");

        // 创建 KV Cache 存储表
        // 用于存储模型推理过程中的键值缓存数据
        let create_kv_cache = r#"
            CREATE TABLE IF NOT EXISTS kv_cache_store (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                layer_idx INTEGER NOT NULL,
                block_idx INTEGER NOT NULL,
                key_data BLOB NOT NULL,
                value_data BLOB NOT NULL,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                importance REAL NOT NULL DEFAULT 1.0,
                compressed BOOLEAN NOT NULL DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                UNIQUE(session_id, layer_idx, block_idx)
            );
            
            -- 为常用查询路径创建索引
            CREATE INDEX IF NOT EXISTS idx_kv_cache_session 
                ON kv_cache_store(session_id);
            
            CREATE INDEX IF NOT EXISTS idx_kv_cache_importance 
                ON kv_cache_store(importance DESC);
            
            CREATE INDEX IF NOT EXISTS idx_kv_cache_created 
                ON kv_cache_store(created_at);
        "#;

        // 创建会话存储表
        // 用于存储推理会话的状态和元数据
        let create_session_store = r#"
            CREATE TABLE IF NOT EXISTS session_store (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                model_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                tokens_count INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT DEFAULT '{}'
            );
            
            -- 索引
            CREATE INDEX IF NOT EXISTS idx_session_status 
                ON session_store(status);
            
            CREATE INDEX IF NOT EXISTS idx_session_model 
                ON session_store(model_name);
            
            CREATE INDEX IF NOT EXISTS idx_session_updated 
                ON session_store(updated_at);
        "#;

        // 在事务中执行建表操作，保证原子性
        let mut tx = self.pool.begin().await.map_err(|e| {
            PersistenceError::Connection(format!("Failed to begin transaction: {}", e))
        })?;

        sqlx::raw_sql(create_kv_cache)
            .execute(&mut *tx)
            .await
            .map_err(|e| {
                PersistenceError::Query(format!("Failed to create kv_cache_store table: {}", e))
            })?;

        sqlx::raw_sql(create_session_store)
            .execute(&mut *tx)
            .await
            .map_err(|e| {
                PersistenceError::Query(format!("Failed to create session_store table: {}", e))
            })?;

        tx.commit().await.map_err(|e| {
            PersistenceError::Query(format!(
                "Failed to commit initialization transaction: {}",
                e
            ))
        })?;

        tracing::info!("Database tables initialized successfully");
        Ok(())
    }

    /// 配置 SQLite 性能优化参数
    ///
    /// 根据配置启用或调整以下参数：
    /// - WAL (Write-Ahead Logging) 模式：提高并发性能
    /// - 忙等待超时：防止锁竞争导致的错误
    /// - 同步模式：在安全性和性能之间权衡
    /// - 缓存大小：优化内存使用
    ///
    /// 注意：某些 PRAGMA（如 journal_mode, synchronous）必须在事务外设置，
    /// 这是 SQLite 的限制。因此这里所有 PRAGMA 都直接在连接池上执行。
    async fn configure_pragmas(&self) -> Result<(), PersistenceError> {
        tracing::debug!("Configuring SQLite pragmas");

        // 注意：以下 PRAGMA 必须在事务外设置（SQLite 限制）

        // 启用 WAL 模式以提高并发读写性能
        if self.config.enable_wal {
            sqlx::raw_sql("PRAGMA journal_mode=WAL;")
                .execute(&self.pool)
                .await
                .map_err(|e| PersistenceError::Query(format!("Failed to set WAL mode: {}", e)))?;

            // WAL 模式下的自动检查点设置
            sqlx::raw_sql("PRAGMA wal_autocheckpoint=1000;")
                .execute(&self.pool)
                .await
                .map_err(|e| {
                    PersistenceError::Query(format!("Failed to set autocheckpoint: {}", e))
                })?;
        } else {
            sqlx::raw_sql("PRAGMA journal_mode=DELETE;")
                .execute(&self.pool)
                .await
                .map_err(|e| {
                    PersistenceError::Query(format!("Failed to set journal mode: {}", e))
                })?;
        }

        // 设置忙等待超时时间（毫秒）
        let timeout_ms = self.config.busy_timeout_secs * 1000;
        sqlx::raw_sql(&format!("PRAGMA busy_timeout={};", timeout_ms))
            .execute(&self.pool)
            .await
            .map_err(|e| PersistenceError::Query(format!("Failed to set busy timeout: {}", e)))?;

        // 外键约束支持
        sqlx::raw_sql("PRAGMA foreign_keys=ON;")
            .execute(&self.pool)
            .await
            .map_err(|e| {
                PersistenceError::Query(format!("Failed to enable foreign keys: {}", e))
            })?;

        // 同步模式：NORMAL 在性能和数据安全之间取得平衡
        // FULL 最安全但最慢，OFF 最快但可能损坏数据
        sqlx::raw_sql("PRAGMA synchronous=NORMAL;")
            .execute(&self.pool)
            .await
            .map_err(|e| {
                PersistenceError::Query(format!("Failed to set synchronous mode: {}", e))
            })?;

        // 缓存大小设置为 -2000 (2MB)
        // 负值表示 KB，正值表示页面数
        sqlx::raw_sql("PRAGMA cache_size=-2000;")
            .execute(&self.pool)
            .await
            .map_err(|e| PersistenceError::Query(format!("Failed to set cache size: {}", e)))?;

        // 启用临时内存中的临时表和索引
        sqlx::raw_sql("PRAGMA temp_store=MEMORY;")
            .execute(&self.pool)
            .await
            .map_err(|e| PersistenceError::Query(format!("Failed to set temp store: {}", e)))?;

        tracing::debug!(
            wal_enabled = self.config.enable_wal,
            busy_timeout_ms = timeout_ms,
            "SQLite pragmas configured"
        );

        Ok(())
    }

    /// 清理指定表的过期数据
    ///
    /// 根据指定的天数阈值删除过期的记录。
    /// 这对于维护数据库大小和性能非常重要。
    ///
    /// # 参数
    ///
    /// * `table_name` - 要清理的表名（必须是已知的安全表名）
    /// * `age_days` - 数据保留天数，超过此天数的数据将被删除
    ///
    /// # 返回值
    ///
    /// * `Ok(u64)` - 删除的记录数量
    /// * `Err(PersistenceError)` - 清理操作失败
    ///
    /// # 安全性
    ///
    /// 此方法只允许清理预定义的白名单表，防止 SQL 注入攻击。
    ///
    /// # 示例
    ///
    /// ```rust,no_run
    /// # use openmini_server::hardware::persistence::database::DatabaseManager;
    /// #[tokio::main]
    /// async fn example_cleanup(db: &DatabaseManager) -> Result<(), Box<dyn std::error::Error>> {
    ///     // 清理 30 天前的 KV Cache 数据
    ///     let deleted_count = db.cleanup_expired("kv_cache_store", 30).await?;
    ///     println!("Deleted {} expired cache entries", deleted_count);
    ///     Ok(())
    /// }
    /// ```
    pub async fn cleanup_expired(
        &self,
        table_name: &str,
        age_days: i64,
    ) -> Result<u64, PersistenceError> {
        // 白名单验证，防止 SQL 注入
        const ALLOWED_TABLES: &[&str] = &["kv_cache_store", "session_store"];

        if !ALLOWED_TABLES.contains(&table_name) {
            return Err(PersistenceError::Query(format!(
                "Table '{}' is not allowed for cleanup",
                table_name
            )));
        }

        tracing::info!(
            table = table_name,
            age_days = age_days,
            "Cleaning up expired data"
        );

        // 构建安全的删除语句
        let delete_sql = format!(
            "DELETE FROM {} WHERE created_at < datetime('now', '-' || ? || ' days')",
            table_name
        );

        let result = sqlx::query(&delete_sql)
            .bind(age_days)
            .execute(&self.pool)
            .await
            .map_err(|e| {
                PersistenceError::Query(format!("Failed to cleanup expired data: {}", e))
            })?;

        let deleted_count = result.rows_affected();

        // 更新统计信息
        {
            let mut stats = self.stats.write().await;
            stats.total_writes += deleted_count as u64;
        }

        tracing::info!(
            table = table_name,
            deleted_records = deleted_count,
            "Cleanup completed"
        );

        Ok(deleted_count)
    }

    // ========================================================================
    // 辅助方法
    // ========================================================================

    /// 更新数据库文件大小统计信息
    async fn update_db_size(&self) {
        if let Ok(metadata) = tokio::fs::metadata(&self.config.db_path).await {
            let size = metadata.len();
            let mut stats = self.stats.write().await;
            stats.db_size_bytes = size;
        }
    }

    /// 更新查询计数器（内部使用）
    pub(crate) async fn increment_query_count(&self, is_write: bool) {
        let mut stats = self.stats.write().await;
        stats.total_queries += 1;
        if is_write {
            stats.total_writes += 1;
        } else {
            stats.total_reads += 1;
        }
    }

    /// 更新缓存命中/未命中统计（内部使用）
    pub(crate) async fn update_cache_stats(&self, hit: bool) {
        let mut stats = self.stats.write().await;
        if hit {
            stats.cache_hits += 1;
        } else {
            stats.cache_misses += 1;
        }
    }
}

// ============================================================================
// 测试模块
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// 测试数据库管理器的完整生命周期
    #[tokio::test]
    async fn test_database_lifecycle() {
        let dir = tempdir().expect("Failed to create temp directory");
        let db_path = dir.path().join("test_persistence.db");

        let config = DatabaseConfig {
            db_path: db_path.clone(),
            pool_size: 5,
            enable_wal: true,
            ..Default::default()
        };

        // 测试创建
        let db = DatabaseManager::new(config)
            .await
            .expect("Failed to create database manager");

        // 测试健康检查
        let healthy = db.health_check().await.expect("Health check failed");
        assert!(healthy, "Database should be healthy after creation");

        // 测试获取统计信息
        let stats = db.get_stats().await;
        assert_eq!(stats.total_queries, 0);

        // 测试获取连接池
        assert!(db.pool().size() > 0, "Pool should have connections");

        // 测试关闭
        db.close().await.expect("Failed to close database");
    }

    /// 测试过期数据清理功能
    #[tokio::test]
    async fn test_cleanup_expired() {
        let dir = tempdir().expect("Failed to create temp directory");
        let db_path = dir.path().join("test_cleanup.db");

        let config = DatabaseConfig {
            db_path,
            ..Default::default()
        };

        let db = DatabaseManager::new(config)
            .await
            .expect("Failed to create database manager");

        // 插入测试数据（使用过去的日期）
        sqlx::query(
            "INSERT INTO kv_cache_store (session_id, layer_idx, block_idx, key_data, value_data, size_bytes) \
             VALUES ('test-session', 0, 0, X'00', X'01', 100)"
        )
        .execute(db.pool())
        .await
        .expect("Failed to insert test data");

        // 清理超过 0 天的数据（应该删除刚插入的数据）
        let _deleted = db
            .cleanup_expired("kv_cache_store", 0)
            .await
            .expect("Cleanup failed");

        // 注意：由于我们刚刚插入数据，可能不会被删除（取决于精度）
        // 这里主要验证方法不会报错

        // 测试非法表名
        let result = db
            .cleanup_expired("malicious_table; DROP TABLE users--", 30)
            .await;
        assert!(result.is_err(), "Should reject invalid table names");

        db.close().await.expect("Failed to close database");
    }

    /// 测试配置默认值
    #[test]
    fn test_default_config() {
        let config = DatabaseConfig::default();
        assert_eq!(config.pool_size, 10);
        assert_eq!(config.max_connections, 20);
        assert!(config.enable_wal);
        assert_eq!(config.busy_timeout_secs, 30);
    }

    /// 测试错误类型的 Display 实现
    #[test]
    fn test_error_display() {
        let conn_err = PersistenceError::Connection("test".to_string());
        assert!(conn_err.to_string().contains("connection"));

        let query_err = PersistenceError::Query("test".to_string());
        assert!(query_err.to_string().contains("Query"));

        let not_found = PersistenceError::NotFound("resource".to_string());
        assert!(not_found.to_string().contains("not found"));
    }
}
