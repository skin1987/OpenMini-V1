//! 记忆表 CRUD 操作
//!
//! 提供记忆的创建、查询、更新和删除功能。
//! 用于替代现有的 HashMap<String, Vec<String>> 存储。

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use sqlx::SqlitePool;
use uuid::Uuid;

/// 记忆级别
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MemoryLevel {
    /// 即时记忆（当前对话）
    Instant,
    /// 短期记忆
    ShortTerm,
    /// 长期记忆
    LongTerm,
}

impl std::fmt::Display for MemoryLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryLevel::Instant => write!(f, "instant"),
            MemoryLevel::ShortTerm => write!(f, "short_term"),
            MemoryLevel::LongTerm => write!(f, "long_term"),
        }
    }
}

impl From<&str> for MemoryLevel {
    fn from(s: &str) -> Self {
        match s {
            "instant" => MemoryLevel::Instant,
            "long_term" => MemoryLevel::LongTerm,
            _ => MemoryLevel::ShortTerm,
        }
    }
}

/// 记忆结构体
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Memory {
    /// 记忆 ID（UUID）
    pub id: String,
    /// 会话 ID
    pub session_id: String,
    /// 内容（Vec<String> 的 JSON 序列化）
    pub content: String,
    /// 重要性评分 (0.0 - 1.0)
    pub importance: f64,
    /// 记忆级别
    pub level: String,
    /// 嵌入向量（二进制）
    pub embedding: Option<Vec<u8>>,
    /// 创建时间
    #[sqlx(rename = "created_at")]
    pub created_at: DateTime<Utc>,
    /// 过期时间
    #[sqlx(rename = "expires_at")]
    pub expires_at: Option<DateTime<Utc>>,
}

/// 新建记忆请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewMemory {
    /// 会话 ID
    pub session_id: String,
    /// 内容（字符串列表，将被序列化为 JSON）
    pub content: Vec<String>,
    /// 重要性评分（默认 0.5）
    pub importance: Option<f64>,
    /// 记忆级别（默认 short_term）
    pub level: Option<MemoryLevel>,
    /// 嵌入向量
    pub embedding: Option<Vec<u8>>,
    /// 过期时间
    pub expires_at: Option<DateTime<Utc>>,
}

impl Memory {
    /// 创建新记忆
    ///
    /// # 参数
    /// - `pool`: 数据库连接池
    /// - `new_memory`: 新记忆信息
    ///
    /// # 返回
    /// 成功返回创建的记忆
    pub async fn create(pool: &SqlitePool, new_memory: NewMemory) -> anyhow::Result<Self> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now();
        let content = serde_json::to_string(&new_memory.content)?;
        let importance = new_memory.importance.unwrap_or(0.5);
        let level = new_memory
            .level
            .unwrap_or(MemoryLevel::ShortTerm)
            .to_string();

        sqlx::query_as::<_, Memory>(
            r#"
            INSERT INTO memories (id, session_id, content, importance, level, embedding, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&id)
        .bind(&new_memory.session_id)
        .bind(&content)
        .bind(importance)
        .bind(&level)
        .bind(&new_memory.embedding)
        .bind(now)
        .bind(new_memory.expires_at)
        .fetch_one(pool)
        .await
        .map_err(|e| anyhow::anyhow!("创建记忆失败: {}", e))
    }

    /// 查询会话的所有记忆
    ///
    /// # 参数
    /// - `pool`: 数据库连接池
    /// - `session_id`: 会话 ID
    ///
    /// # 返回
    /// 成功返回记忆列表
    pub async fn find_by_session(pool: &SqlitePool, session_id: &str) -> anyhow::Result<Vec<Self>> {
        let memories = sqlx::query_as::<_, Memory>(
            r#"
            SELECT * FROM memories WHERE session_id = ? ORDER BY created_at DESC
            "#,
        )
        .bind(session_id)
        .fetch_all(pool)
        .await
        .map_err(|e| anyhow::anyhow!("查询会话记忆失败: {}", e))?;

        Ok(memories)
    }

    /// 查询指定级别的记忆
    ///
    /// # 参数
    /// - `pool`: 数据库连接池
    /// - `session_id`: 会话 ID
    /// - `level`: 记忆级别
    ///
    /// # 返回
    /// 成功返回匹配的记忆列表
    pub async fn find_by_level(
        pool: &SqlitePool,
        session_id: &str,
        level: MemoryLevel,
    ) -> anyhow::Result<Vec<Self>> {
        let memories = sqlx::query_as::<_, Memory>(
            r#"
            SELECT * FROM memories WHERE session_id = ? AND level = ? ORDER BY importance DESC
            "#,
        )
        .bind(session_id)
        .bind(level.to_string())
        .fetch_all(pool)
        .await
        .map_err(|e| anyhow::anyhow!("查询记忆失败: {}", e))?;

        Ok(memories)
    }

    /// 更新重要性评分
    ///
    /// # 参数
    /// - `pool`: 数据库连接池
    /// - `importance`: 新的重要性评分
    ///
    /// # 返回
    /// 成功返回更新后的记忆
    pub async fn update_importance(
        &self,
        pool: &SqlitePool,
        importance: f64,
    ) -> anyhow::Result<Self> {
        sqlx::query_as::<_, Memory>(
            r#"
            UPDATE memories SET importance = ? WHERE id = ?
            "#,
        )
        .bind(importance)
        .bind(&self.id)
        .fetch_one(pool)
        .await
        .map_err(|e| anyhow::anyhow!("更新记忆重要性失败: {}", e))
    }

    /// 删除记忆
    ///
    /// # 参数
    /// - `pool`: 数据库连接池
    ///
    /// # 返回
    /// 成功返回 ()
    pub async fn delete(&self, pool: &SqlitePool) -> anyhow::Result<()> {
        sqlx::query("DELETE FROM memories WHERE id = ?")
            .bind(&self.id)
            .execute(pool)
            .await
            .map_err(|e| anyhow::anyhow!("删除记忆失败: {}", e))?;

        tracing::info!(memory_id = %self.id, "记忆已删除");

        Ok(())
    }

    /// 清理过期记忆
    ///
    /// 删除所有已过期的记忆记录。
    ///
    /// # 参数
    /// - `pool`: 数据库连接池
    ///
    /// # 返回
    /// 成功返回删除的记录数
    pub async fn cleanup_expired(pool: &SqlitePool) -> anyhow::Result<u64> {
        let result =
            sqlx::query("DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?")
                .bind(Utc::now())
                .execute(pool)
                .await
                .map_err(|e| anyhow::anyhow!("清理过期记忆失败: {}", e))?;

        if result.rows_affected() > 0 {
            tracing::info!(deleted_count = result.rows_affected(), "已清理过期记忆");
        }

        Ok(result.rows_affected())
    }

    /// 解析内容为字符串列表
    ///
    /// 将存储的 JSON 字符串反序列化为 Vec<String>。
    ///
    /// # 返回
    /// 成功返回字符串列表
    pub fn parse_content(&self) -> anyhow::Result<Vec<String>> {
        serde_json::from_str(&self.content).map_err(|e| anyhow::anyhow!("解析记忆内容失败: {}", e))
    }
}
