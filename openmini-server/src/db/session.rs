//! 会话表 CRUD 操作
//!
//! 提供会话的创建、查询、更新和删除功能。

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use sqlx::SqlitePool;
use uuid::Uuid;

/// 会话状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SessionStatus {
    /// 活跃状态
    Active,
    /// 已关闭
    Closed,
    /// 已归档
    Archived,
}

impl std::fmt::Display for SessionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionStatus::Active => write!(f, "active"),
            SessionStatus::Closed => write!(f, "closed"),
            SessionStatus::Archived => write!(f, "archived"),
        }
    }
}

impl From<&str> for SessionStatus {
    fn from(s: &str) -> Self {
        match s {
            "closed" => SessionStatus::Closed,
            "archived" => SessionStatus::Archived,
            _ => SessionStatus::Active,
        }
    }
}

/// 会话结构体
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Session {
    /// 会话 ID（UUID）
    pub id: String,
    /// 用户 ID
    pub user_id: String,
    /// 模型名称
    pub model: String,
    /// 会话状态
    pub status: String,
    /// 元数据（JSON 格式）
    pub metadata: Option<String>,
    /// 创建时间
    #[sqlx(rename = "created_at")]
    pub created_at: DateTime<Utc>,
    /// 更新时间
    #[sqlx(rename = "updated_at")]
    pub updated_at: DateTime<Utc>,
}

/// 新建会话请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewSession {
    /// 用户 ID
    pub user_id: String,
    /// 模型名称
    pub model: Option<String>,
    /// 元数据
    pub metadata: Option<serde_json::Value>,
}

impl Session {
    /// 创建新会话
    ///
    /// # 参数
    /// - `pool`: 数据库连接池
    /// - `new_session`: 新会话信息
    ///
    /// # 返回
    /// 成功返回创建的会话
    pub async fn create(pool: &SqlitePool, new_session: NewSession) -> anyhow::Result<Self> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now();
        let model = new_session.model.unwrap_or_else(|| "openmini-v1".to_string());
        let metadata = new_session
            .metadata
            .map(|m| serde_json::to_string(&m).unwrap_or_default());

        sqlx::query_as::<_, Session>(
            r#"
            INSERT INTO sessions (id, user_id, model, status, metadata, created_at, updated_at)
            VALUES (?, ?, ?, 'active', ?, ?, ?)
            "#,
        )
        .bind(&id)
        .bind(&new_session.user_id)
        .bind(&model)
        .bind(&metadata)
        .bind(now)
        .bind(now)
        .fetch_one(pool)
        .await
        .map_err(|e| anyhow::anyhow!("创建会话失败: {}", e))
    }

    /// 根据 ID 查询会话
    ///
    /// # 参数
    /// - `pool`: 数据库连接池
    /// - `id`: 会话 ID
    ///
    /// # 返回
    /// 成功返回会话，不存在返回 None
    pub async fn find_by_id(pool: &SqlitePool, id: &str) -> anyhow::Result<Option<Self>> {
        let session = sqlx::query_as::<_, Session>(
            r#"
            SELECT * FROM sessions WHERE id = ?
            "#,
        )
        .bind(id)
        .fetch_optional(pool)
        .await
        .map_err(|e| anyhow::anyhow!("查询会话失败: {}", e))?;

        Ok(session)
    }

    /// 更新会话状态
    ///
    /// # 参数
    /// - `pool`: 数据库连接池
    /// - `status`: 新状态
    ///
    /// # 返回
    /// 成功返回更新后的会话
    pub async fn update_status(
        &self,
        pool: &SqlitePool,
        status: SessionStatus,
    ) -> anyhow::Result<Self> {
        let now = Utc::now();

        sqlx::query_as::<_, Session>(
            r#"
            UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?
            "#,
        )
        .bind(status.to_string())
        .bind(now)
        .bind(&self.id)
        .fetch_one(pool)
        .await
        .map_err(|e| anyhow::anyhow!("更新会话状态失败: {}", e))
    }

    /// 查询用户的所有会话
    ///
    /// # 参数
    /// - `pool`: 数据库连接池
    /// - `user_id`: 用户 ID
    ///
    /// # 返回
    /// 成功返回会话列表
    pub async fn list_by_user(pool: &SqlitePool, user_id: &str) -> anyhow::Result<Vec<Self>> {
        let sessions = sqlx::query_as::<_, Session>(
            r#"
            SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC
            "#,
        )
        .bind(user_id)
        .fetch_all(pool)
        .await
        .map_err(|e| anyhow::anyhow!("查询用户会话失败: {}", e))?;

        Ok(sessions)
    }

    /// 关闭会话
    ///
    /// # 参数
    /// - `pool`: 数据库连接池
    ///
    /// # 返回
    /// 成功返回更新后的会话
    pub async fn close(&self, pool: &SqlitePool) -> anyhow::Result<Self> {
        self.update_status(pool, SessionStatus::Closed).await
    }

    /// 删除会话
    ///
    /// # 参数
    /// - `pool`: 数据库连接池
    ///
    /// # 返回
    /// 成功返回 ()
    pub async fn delete(&self, pool: &SqlitePool) -> anyhow::Result<()> {
        // 先删除关联的消息
        sqlx::query("DELETE FROM messages WHERE session_id = ?")
            .bind(&self.id)
            .execute(pool)
            .await
            .map_err(|e| anyhow::anyhow!("删除会话消息失败: {}", e))?;

        // 删除关联的记忆
        sqlx::query("DELETE FROM memories WHERE session_id = ?")
            .bind(&self.id)
            .execute(pool)
            .await
            .map_err(|e| anyhow::anyhow!("删除会话记忆失败: {}", e))?;

        // 删除会话
        sqlx::query("DELETE FROM sessions WHERE id = ?")
            .bind(&self.id)
            .execute(pool)
            .await
            .map_err(|e| anyhow::anyhow!("删除会话失败: {}", e))?;

        tracing::info!(session_id = %self.id, "会话已删除");

        Ok(())
    }
}
