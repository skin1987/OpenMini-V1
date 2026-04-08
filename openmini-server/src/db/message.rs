//! 消息表 CRUD 操作
//!
//! 提供消息的创建、查询和删除功能。

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use sqlx::SqlitePool;
use uuid::Uuid;

/// 消息角色
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MessageRole {
    /// 用户消息
    User,
    /// 助手回复
    Assistant,
    /// 系统消息
    System,
}

impl std::fmt::Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageRole::User => write!(f, "user"),
            MessageRole::Assistant => write!(f, "assistant"),
            MessageRole::System => write!(f, "system"),
        }
    }
}

/// 媒体类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MediaType {
    /// 纯文本
    Text,
    /// 图像
    Image,
    /// 音频
    Audio,
    /// 视频
    Video,
}

impl std::fmt::Display for MediaType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MediaType::Text => write!(f, "text"),
            MediaType::Image => write!(f, "image"),
            MediaType::Audio => write!(f, "audio"),
            MediaType::Video => write!(f, "video"),
        }
    }
}

impl From<&str> for MediaType {
    fn from(s: &str) -> Self {
        match s {
            "image" => MediaType::Image,
            "audio" => MediaType::Audio,
            "video" => MediaType::Video,
            _ => MediaType::Text,
        }
    }
}

/// 消息结构体
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Message {
    /// 消息 ID（UUID）
    pub id: String,
    /// 会话 ID
    pub session_id: String,
    /// 消息序号
    pub message_idx: i64,
    /// 角色
    pub role: String,
    /// 内容
    pub content: String,
    /// 媒体类型
    pub media_type: String,
    /// 媒体数据（二进制）
    pub media_data: Option<Vec<u8>>,
    /// Token 数量
    pub token_count: i64,
    /// 创建时间
    #[sqlx(rename = "created_at")]
    pub created_at: DateTime<Utc>,
}

/// 新建消息请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewMessage {
    /// 会话 ID
    pub session_id: String,
    /// 角色
    pub role: MessageRole,
    /// 内容
    pub content: String,
    /// 媒体类型（默认 Text）
    pub media_type: Option<MediaType>,
    /// 媒体数据
    pub media_data: Option<Vec<u8>>,
    /// Token 数量
    pub token_count: Option<i64>,
}

impl Message {
    /// 创建新消息
    ///
    /// # 参数
    /// - `pool`: 数据库连接池
    /// - `new_message`: 新消息信息
    ///
    /// # 返回
    /// 成功返回创建的消息
    pub async fn create(pool: &SqlitePool, new_message: NewMessage) -> anyhow::Result<Self> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now();
        let media_type = new_message
            .media_type
            .unwrap_or(MediaType::Text)
            .to_string();
        let token_count = new_message.token_count.unwrap_or(0);

        // 获取当前会话的最大消息序号
        let max_idx: Option<(i64,)> = sqlx::query_as(
            r#"
            SELECT COALESCE(MAX(message_idx), -1) FROM messages WHERE session_id = ?
            "#,
        )
        .bind(&new_message.session_id)
        .fetch_optional(pool)
        .await
        .map_err(|e| anyhow::anyhow!("获取消息序号失败: {}", e))?;

        let message_idx = max_idx.map_or(0, |(idx,)| idx + 1);

        sqlx::query_as::<_, Message>(
            r#"
            INSERT INTO messages (id, session_id, message_idx, role, content, media_type, media_data, token_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&id)
        .bind(&new_message.session_id)
        .bind(message_idx)
        .bind(new_message.role.to_string())
        .bind(&new_message.content)
        .bind(&media_type)
        .bind(&new_message.media_data)
        .bind(token_count)
        .bind(now)
        .fetch_one(pool)
        .await
        .map_err(|e| anyhow::anyhow!("创建消息失败: {}", e))
    }

    /// 查询会话的所有消息
    ///
    /// # 参数
    /// - `pool`: 数据库连接池
    /// - `session_id`: 会话 ID
    ///
    /// # 返回
    /// 成功返回消息列表（按序号排序）
    pub async fn find_by_session(
        pool: &SqlitePool,
        session_id: &str,
    ) -> anyhow::Result<Vec<Self>> {
        let messages = sqlx::query_as::<_, Message>(
            r#"
            SELECT * FROM messages WHERE session_id = ? ORDER BY message_idx ASC
            "#,
        )
        .bind(session_id)
        .fetch_all(pool)
        .await
        .map_err(|e| anyhow::anyhow!("查询会话消息失败: {}", e))?;

        Ok(messages)
    }

    /// 查询最近的 N 条消息
    ///
    /// # 参数
    /// - `pool`: 数据库连接池
    /// - `session_id`: 会话 ID
    /// - `limit`: 返回数量限制
    ///
    /// # 返回
    /// 成功返回最近的消息列表
    pub async fn find_recent(
        pool: &SqlitePool,
        session_id: &str,
        limit: i64,
    ) -> anyhow::Result<Vec<Self>> {
        let messages = sqlx::query_as::<_, Message>(
            r#"
            SELECT * FROM messages WHERE session_id = ? ORDER BY message_idx DESC LIMIT ?
            "#,
        )
        .bind(session_id)
        .bind(limit)
        .fetch_all(pool)
        .await
        .map_err(|e| anyhow::anyhow!("查询最近消息失败: {}", e))?;

        Ok(messages)
    }

    /// 统计会话的消息数量
    ///
    /// # 参数
    /// - `pool`: 数据库连接池
    /// - `session_id`: 会话 ID
    ///
    /// # 返回
    /// 成功返回消息数量
    pub async fn count_by_session(
        pool: &SqlitePool,
        session_id: &str,
    ) -> anyhow::Result<i64> {
        let count: (i64,) = sqlx::query_as(
            r#"
            SELECT COUNT(*) FROM messages WHERE session_id = ?
            "#,
        )
        .bind(session_id)
        .fetch_one(pool)
        .await
        .map_err(|e| anyhow::anyhow!("统计消息数量失败: {}", e))?;

        Ok(count.0)
    }

    /// 删除会话的所有消息
    ///
    /// # 参数
    /// - `pool`: 数据库连接池
    /// - `session_id`: 会话 ID
    ///
    /// # 返回
    /// 成功返回 ()
    pub async fn delete_by_session(
        pool: &SqlitePool,
        session_id: &str,
    ) -> anyhow::Result<()> {
        let result = sqlx::query("DELETE FROM messages WHERE session_id = ?")
            .bind(session_id)
            .execute(pool)
            .await
            .map_err(|e| anyhow::anyhow!("删除会话消息失败: {}", e))?;

        tracing::info!(
            session_id = %session_id,
            deleted_count = result.rows_affected(),
            "会话消息已删除"
        );

        Ok(())
    }
}
