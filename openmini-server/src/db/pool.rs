//! SQLite 连接池管理
//!
//! 提供连接池创建、配置和数据库迁移功能。
//! 使用 WAL 模式以获得更好的并发性能。

use crate::db::DatabaseConfig;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::SqlitePool;
use std::str::FromStr;

/// 创建 SQLite 连接池
///
/// # 参数
/// - `config`: 数据库配置
///
/// # 返回
/// 成功返回连接池实例
pub async fn create_pool(config: &DatabaseConfig) -> anyhow::Result<SqlitePool> {
    // 确保数据库目录存在
    if let Some(parent) = config.path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // 配置 SQLite 连接选项
    let options = SqliteConnectOptions::from_str(&format!("sqlite:{}?mode=rwc", config.path.display()))?
        .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
        .busy_timeout(std::time::Duration::from_millis(config.busy_timeout_ms))
        .synchronous(sqlx::sqlite::SqliteSynchronous::Normal)
        .foreign_keys(true);

    // 创建连接池
    let pool = SqlitePoolOptions::new()
        .max_connections(config.pool_size)
        .connect_with(options)
        .await?;

    tracing::info!(
        path = %config.path.display(),
        pool_size = config.pool_size,
        "SQLite 连接池创建成功"
    );

    // 执行数据库迁移
    run_migrations(&pool).await?;

    Ok(pool)
}

/// 运行数据库迁移
///
/// 创建所有必需的表和索引。
async fn run_migrations(pool: &SqlitePool) -> anyhow::Result<()> {
    tracing::info!("开始执行数据库迁移...");

    // 会话表
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL DEFAULT '',
            model       TEXT NOT NULL DEFAULT 'openmini-v1',
            status      TEXT NOT NULL DEFAULT 'active',
            metadata    TEXT,
            created_at  INTEGER NOT NULL,
            updated_at  INTEGER NOT NULL
        )
        "#,
    )
    .execute(pool)
    .await?;

    // 会话表索引
    sqlx::query(
        r#"
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)
        "#,
    )
    .execute(pool)
    .await?;

    sqlx::query(
        r#"
        CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)
        "#,
    )
    .execute(pool)
    .await?;

    // 消息表
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS messages (
            id            TEXT PRIMARY KEY,
            session_id    TEXT NOT NULL REFERENCES sessions(id),
            message_idx   INTEGER NOT NULL,
            role          TEXT NOT NULL,
            content       TEXT NOT NULL,
            media_type    TEXT DEFAULT 'text',
            media_data    BLOB,
            token_count   INTEGER DEFAULT 0,
            created_at    INTEGER NOT NULL,
            UNIQUE(session_id, message_idx)
        )
        "#,
    )
    .execute(pool)
    .await?;

    // 消息表索引
    sqlx::query(
        r#"
        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)
        "#,
    )
    .execute(pool)
    .await?;

    // 记忆表
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS memories (
            id           TEXT PRIMARY KEY,
            session_id   TEXT NOT NULL REFERENCES sessions(id),
            content      TEXT NOT NULL,
            importance   REAL NOT NULL DEFAULT 0.5,
            level        TEXT NOT NULL DEFAULT 'short_term',
            embedding    BLOB,
            created_at   INTEGER NOT NULL,
            expires_at   INTEGER
        )
        "#,
    )
    .execute(pool)
    .await?;

    // 记忆表索引
    sqlx::query(
        r#"
        CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id)
        "#,
    )
    .execute(pool)
    .await?;

    sqlx::query(
        r#"
        CREATE INDEX IF NOT EXISTS idx_memories_level ON memories(level)
        "#,
    )
    .execute(pool)
    .await?;

    tracing::info!("数据库迁移完成");

    Ok(())
}

/// 关闭连接池
///
/// 优雅地关闭所有连接。
pub async fn close_pool(pool: &SqlitePool) {
    tracing::info!("正在关闭数据库连接池...");
    pool.close().await;
    tracing::info!("数据库连接池已关闭");
}
