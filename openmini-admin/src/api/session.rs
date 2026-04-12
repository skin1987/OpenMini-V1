//! 会话监控 API 模块
//!
//! 提供活跃会话管理、会话详情查询、会话统计和会话终止功能。
//! 用于实时监控和管理用户会话状态。

use axum::{
    extract::{Path, Query, State},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use crate::error::AppError;
use crate::AppState;

// ==================== 数据结构定义 ====================

/// 会话信息结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub session_id: String,
    pub user_id: Option<i64>,
    pub username: Option<String>,
    pub model_id: Option<String>,
    pub status: String,
    pub created_at: String,
    pub last_activity_at: String,
    pub message_count: i64,
    pub total_tokens: i64,
}

/// 会话详情（包含消息历史）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDetail {
    #[serde(flatten)]
    pub base: SessionInfo,
    pub messages: Vec<SessionMessage>,
    pub metadata: HashMap<String, Value>,
}

/// 会话消息记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessage {
    pub id: i64,
    pub role: String,
    pub content: String,
    pub token_count: i64,
    pub created_at: String,
    pub latency_ms: Option<i64>,
}

/// 时间范围枚举
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeRange {
    LastHour,
    Last6Hours,
    Last24Hours,
    Last7Days,
    Last30Days,
}

impl TimeRange {
    /// 获取时间范围的 SQL WHERE 条件
    pub fn to_sql_condition(&self) -> String {
        match self {
            TimeRange::LastHour => "AND created_at >= datetime('now', '-1 hour')".to_string(),
            TimeRange::Last6Hours => "AND created_at >= datetime('now', '-6 hours')".to_string(),
            TimeRange::Last24Hours => "AND created_at >= datetime('now', '-1 day')".to_string(),
            TimeRange::Last7Days => "AND created_at >= datetime('now', '-7 days')".to_string(),
            TimeRange::Last30Days => "AND created_at >= datetime('now', '-30 days')".to_string(),
        }
    }
}

/// 会话统计数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStats {
    // 基础统计
    pub total_sessions: i64,
    pub active_sessions: i64,
    pub completed_sessions: i64,

    // Token 使用量统计
    pub total_input_tokens: i64,
    pub total_output_tokens: i64,
    pub avg_tokens_per_session: f64,

    // 性能指标
    pub avg_response_time_ms: f64,
    pub p50_response_time_ms: f64,
    pub p95_response_time_ms: f64,
    pub p99_response_time_ms: f64,

    // 时间范围
    pub time_range: String,
    pub calculated_at: String,
}

/// 分页查询参数
#[derive(Debug, Deserialize)]
pub struct PaginationParams {
    pub page: Option<u32>,
    pub page_size: Option<u32>,
    pub status: Option<String>,
    pub user_id: Option<i64>,
}

impl PaginationParams {
    /// 获取分页参数，带默认值和上限限制
    pub fn normalized(&self) -> (i64, i64) {
        let page = self.page.unwrap_or(1).max(1) as i64;
        let page_size = self.page_size.unwrap_or(20).min(100).max(1) as i64;
        (page, page_size)
    }

    /// 计算 OFFSET 值
    pub fn offset(&self) -> i64 {
        let (page, page_size) = self.normalized();
        (page - 1) * page_size
    }
}

// ==================== API 处理函数 ====================

/// 获取活跃会话列表（支持分页、过滤）
///
/// # 参数
/// - `page`: 页码（默认 1）
/// - `page_size`: 每页数量（默认 20，最大 100）
/// - `status`: 状态过滤（active/completed/error）
/// - `user_id`: 用户 ID 过滤
///
/// # 返回
/// 分页的会话列表及总数
pub async fn list_active_sessions(
    State(state): State<AppState>,
    Query(params): Query<PaginationParams>,
) -> Result<Json<Value>, AppError> {
    let (page, page_size) = params.normalized();

    // 构建动态查询条件
    let mut conditions = vec!["status = 'active'".to_string()];
    let mut bind_values: Vec<String> = vec![];

    if let Some(ref status) = params.status {
        if !status.is_empty() {
            conditions.push(format!("status = '{}'", status));
        }
    }

    if let Some(user_id) = params.user_id {
        conditions.push("user_id = ?".to_string());
        bind_values.push(user_id.to_string());
    }

    let where_clause = if conditions.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", conditions.join(" AND "))
    };

    // 查询会话列表
    let query_str = format!(
        "SELECT s.*, u.username \
         FROM sessions s \
         LEFT JOIN users u ON s.user_id = u.id \
         {} \
         ORDER BY s.last_activity_at DESC \
         LIMIT ? OFFSET ?",
        where_clause
    );

    let mut query = sqlx::query_as::<_, (String, Option<i64>, Option<String>, Option<String>, String, String, String, i64, i64)>(
        &query_str
    );

    for value in &bind_values {
        query = query.bind(value);
    }
    query = query.bind(page_size).bind(params.offset());

    let rows = query.fetch_all(&*state.pool).await?;

    // 构建会话信息列表
    let sessions: Vec<SessionInfo> = rows
        .into_iter()
        .map(|row| SessionInfo {
            session_id: row.0,
            user_id: row.1,
            username: row.2,
            model_id: row.3,
            status: row.4,
            created_at: row.5,
            last_activity_at: row.6,
            message_count: row.7,
            total_tokens: row.8,
        })
        .collect();

    // 查询总数
    let count_query = format!(
        "SELECT COUNT(*) FROM sessions {}",
        where_clause
    );

    let mut count_query_builder = sqlx::query_as::<_, (i64,)>(&count_query);
    for value in &bind_values {
        count_query_builder = count_query_builder.bind(value);
    }
    let (total,) = count_query_builder.fetch_one(&*state.pool).await?;

    Ok(Json(serde_json::json!({
        "items": sessions,
        "total": total,
        "page": page,
        "page_size": page_size,
        "has_more": (page * page_size) < total
    })))
}

/// 获取会话详情（含消息历史）
///
/// # 参数
/// - `session_id`: 会话 ID
///
/// # 返回
/// 会话基本信息及消息历史记录
pub async fn get_session_detail(
    Path(session_id): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    // 查询会话基本信息
    let session_row = sqlx::query_as::<_, (String, Option<i64>, Option<String>, Option<String>, String, String, String, i64, i64)>(
        "SELECT s.*, u.username \
         FROM sessions s \
         LEFT JOIN users u ON s.user_id = u.id \
         WHERE s.session_id = ?"
    )
    .bind(&session_id)
    .fetch_optional(&*state.pool)
    .await?
    .ok_or_else(|| AppError::NotFound(format!("会话不存在: {}", session_id)))?;

    let base_info = SessionInfo {
        session_id: session_row.0.clone(),
        user_id: session_row.1,
        username: session_row.2,
        model_id: session_row.3,
        status: session_row.4.clone(),
        created_at: session_row.5,
        last_activity_at: session_row.6,
        message_count: session_row.7,
        total_tokens: session_row.8,
    };

    // 查询消息历史（最多返回最近 100 条）
    let message_rows = sqlx::query_as::<_, (i64, String, String, i64, String, Option<i64>)>(
        "SELECT id, role, content, token_count, created_at, latency_ms \
         FROM session_messages \
         WHERE session_id = ? \
         ORDER BY created_at ASC \
         LIMIT 100"
    )
    .bind(&session_id)
    .fetch_all(&*state.pool)
    .await?;

    let messages: Vec<SessionMessage> = message_rows
        .into_iter()
        .map(|row| SessionMessage {
            id: row.0,
            role: row.1,
            content: row.2,
            token_count: row.3,
            created_at: row.4,
            latency_ms: row.5,
        })
        .collect();

    // 构建元数据
    let mut metadata = HashMap::new();
    metadata.insert(
        "message_limit_reached".to_string(),
        Value::Bool(messages.len() >= 100),
    );
    metadata.insert(
        "query_time".to_string(),
        Value::String(chrono::Utc::now().to_rfc3339()),
    );

    Ok(Json(serde_json::json!({
        "session": base_info,
        "messages": messages,
        "metadata": metadata
    })))
}

/// 获取会话统计信息（token使用量、响应时间等）
///
/// # 参数
/// - `time_range`: 时间范围（last_hour/last_6_hours/last_24_hours/last_7_days/last_30_days/custom）
///
/// # 返回
/// 会话统计数据，包括 token 用量和性能指标
pub async fn get_session_stats(
    State(state): State<AppState>,
    Query(time_range): Query<serde_json::Value>,
) -> Result<Json<Value>, AppError> {
    // 解析时间范围，默认为 last 24 hours
    let range: TimeRange = serde_json::from_value(time_range.clone())
        .unwrap_or(TimeRange::Last24Hours);

    let time_condition = range.to_sql_condition();
    let range_label = match &range {
        TimeRange::LastHour => "last_hour",
        TimeRange::Last6Hours => "last_6_hours",
        TimeRange::Last24Hours => "last_24_hours",
        TimeRange::Last7Days => "last_7_days",
        TimeRange::Last30Days => "last_30_days",
    };

    // 并行查询多个统计指标以提高性能
    // 预构建所有 SQL 查询字符串以避免临时值借用问题
    let total_sessions_sql = format!("SELECT COUNT(*) FROM sessions WHERE 1=1 {}", time_condition);
    let completed_sessions_sql = format!(
        "SELECT COUNT(*) FROM sessions WHERE status = 'completed' {}",
        time_condition
    );
    let tokens_sql = format!(
        "SELECT \
         COALESCE(SUM(input_tokens), 0), \
         COALESCE(SUM(output_tokens), 0) \
         FROM sessions \
         WHERE 1=1 {}",
        time_condition
    );
    let response_times_sql = format!(
        "SELECT \
         AVG(latency_ms), \
         (SELECT latency_ms FROM session_messages WHERE latency_ms > 0 {} ORDER BY latency_ms ASC LIMIT 1 OFFSET (SELECT COUNT(*)/2 FROM session_messages WHERE latency_ms > 0 {})), \
         (SELECT latency_ms FROM session_messages WHERE latency_ms > 0 {} ORDER BY latency_ms DESC LIMIT 1 OFFSET (SELECT COUNT()*95/100 FROM session_messages WHERE latency_ms > 0 {})), \
         (SELECT latency_ms FROM session_messages WHERE latency_ms > 0 {} ORDER BY latency_ms DESC LIMIT 1 OFFSET (SELECT COUNT()*99/100 FROM session_messages WHERE latency_ms > 0 {})) \
         FROM session_messages \
         WHERE role = 'assistant' AND latency_ms IS NOT NULL {}",
        time_condition, time_condition.replace("created_at", "m.created_at"),
        time_condition, time_condition.replace("created_at", "m.created_at"),
        time_condition, time_condition.replace("created_at", "m.created_at"),
        time_condition
    );

    let (
        total_sessions_result,
        active_sessions_result,
        completed_sessions_result,
        tokens_result,
        response_times_result,
    ) = tokio::try_join!(
        // 总会话数
        sqlx::query_as::<_, (i64,)>(&total_sessions_sql)
        .fetch_one(&*state.pool),

        // 活跃会话数
        sqlx::query_as::<_, (i64,)>(
            "SELECT COUNT(*) FROM sessions WHERE status = 'active'"
        )
        .fetch_one(&*state.pool),

        // 已完成会话数
        sqlx::query_as::<_, (i64,)>(&completed_sessions_sql)
        .fetch_one(&*state.pool),

        // Token 统计
        sqlx::query_as::<_, (Option<i64>, Option<i64>)>(&tokens_sql)
        .fetch_one(&*state.pool),

        // 响应时间统计（从消息表获取）
        sqlx::query_as::<_, (Option<f64>, Option<f64>, Option<f64>, Option<f64>)>(&response_times_sql)
        .fetch_one(&*state.pool),
    )?;

    let (total_sessions,) = total_sessions_result;
    let (active_sessions,) = active_sessions_result;
    let (completed_sessions,) = completed_sessions_result;
    let (input_tokens, output_tokens) = tokens_result;
    let (avg_latency, p50_latency, p95_latency, p99_latency) = response_times_result;

    // 计算平均每会话 token 数
    let avg_tokens_per_session = if total_sessions > 0 {
        (input_tokens.unwrap_or(0) + output_tokens.unwrap_or(0)) as f64 / total_sessions as f64
    } else {
        0.0
    };

    let stats = SessionStats {
        total_sessions,
        active_sessions,
        completed_sessions,
        total_input_tokens: input_tokens.unwrap_or(0),
        total_output_tokens: output_tokens.unwrap_or(0),
        avg_tokens_per_session,
        avg_response_time_ms: avg_latency.unwrap_or(0.0),
        p50_response_time_ms: p50_latency.unwrap_or(0.0),
        p95_response_time_ms: p95_latency.unwrap_or(0.0),
        p99_response_time_ms: p99_latency.unwrap_or(0.0),
        time_range: range_label.to_string(),
        calculated_at: chrono::Utc::now().to_rfc3339(),
    };

    Ok(Json(serde_json::json!(stats)))
}

/// 终止指定会话
///
/// # 参数
/// - `session_id`: 要终止的会话 ID
///
/// # 返回
/// 操作结果确认
pub async fn terminate_session(
    Path(session_id): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    // 检查会话是否存在且为活跃状态
    let current_status: Option<(String,)> = sqlx::query_as(
        "SELECT status FROM sessions WHERE session_id = ?"
    )
    .bind(&session_id)
    .fetch_optional(&*state.pool)
    .await?
    .map(|row| row);

    match current_status {
        Some((status,)) if status == "active" => {
            // 更新会话状态为 terminated
            let now = chrono::Utc::now().to_rfc3339();
            sqlx::query(
                "UPDATE sessions SET status = 'terminated', updated_at = ?, last_activity_at = ? WHERE session_id = ?"
            )
            .bind(&now)
            .bind(&now)
            .bind(&session_id)
            .execute(&*state.pool)
            .await?;

            tracing::info!(session_id = %session_id, "会话已终止");

            Ok(Json(serde_json::json!({
                "message": "会话已成功终止",
                "session_id": session_id,
                "status": "terminated",
                "terminated_at": now
            })))
        }
        Some((status,)) => Err(AppError::BadRequest(format!(
            "无法终止状态为 '{}' 的会话",
            status
        ))),
        None => Err(AppError::NotFound(format!("会话不存在: {}", session_id))),
    }
}

/// 批量清理过期会话
///
/// 清理超过指定天数未活动的会话
///
/// # Query 参数
/// - `days`: 过期天数（默认 30 天）
pub async fn cleanup_expired_sessions(
    State(state): State<AppState>,
    Query(params): Query<serde_json::Value>,
) -> Result<Json<Value>, AppError> {
    let days: i64 = params
        .get("days")
        .and_then(|v| v.as_i64())
        .unwrap_or(30);

    if days <= 0 {
        return Err(AppError::BadRequest("days 必须大于 0".to_string()));
    }

    let result = sqlx::query(
        "DELETE FROM sessions \
         WHERE status IN ('completed', 'terminated', 'error') \
         AND last_activity_at < datetime('now', '-' || ? || ' days')"
    )
    .bind(days)
    .execute(&*state.pool)
    .await?;

    let deleted_count = result.rows_affected();

    tracing::info!(days = days, deleted = deleted_count, "清理过期会话完成");

    Ok(Json(serde_json::json!({
        "message": "过期会话清理完成",
        "deleted_count": deleted_count,
        "expiry_days": days,
        "cleaned_at": chrono::Utc::now().to_rfc3339()
    })))
}

// ==================== 单元测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== 数据结构测试 ====================

    #[test]
    fn test_session_info_serialization() {
        let session = SessionInfo {
            session_id: "sess_abc123".to_string(),
            user_id: Some(1),
            username: Some("admin".to_string()),
            model_id: Some("qwen-14b".to_string()),
            status: "active".to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            last_activity_at: "2024-01-01T01:00:00Z".to_string(),
            message_count: 10,
            total_tokens: 1500,
        };

        let json = serde_json::to_value(&session).unwrap();
        assert_eq!(json["session_id"], "sess_abc123");
        assert_eq!(json["user_id"], 1);
        assert_eq!(json["username"], "admin");
        assert_eq!(json["status"], "active");
        assert_eq!(json["message_count"], 10);
        assert_eq!(json["total_tokens"], 1500);
    }

    #[test]
    fn test_session_message_serialization() {
        let message = SessionMessage {
            id: 1,
            role: "user".to_string(),
            content: "Hello, how are you?".to_string(),
            token_count: 6,
            created_at: "2024-01-01T00:05:00Z".to_string(),
            latency_ms: None,
        };

        let json = serde_json::to_value(&message).unwrap();
        assert_eq!(json["id"], 1);
        assert_eq!(json["role"], "user");
        assert_eq!(json["content"], "Hello, how are you?");
        assert_eq!(json["token_count"], 6);
        // Option 在 JSON 中序列化为 null
        assert_eq!(json.get("latency_ms").and_then(|v| v.as_null()), Some(()));
    }

    #[test]
    fn test_time_range_default() {
        // 测试从字符串反序列化（snake_case 格式）
        let range: TimeRange = serde_json::from_value(serde_json::json!("last24_hours")).unwrap();
        assert!(matches!(range, TimeRange::Last24Hours));
    }

    #[test]
    fn test_pagination_params_normalization() {
        // 测试正常值
        let params = PaginationParams {
            page: Some(2),
            page_size: Some(50),
            status: None,
            user_id: None,
        };
        let (page, size) = params.normalized();
        assert_eq!(page, 2);
        assert_eq!(size, 50);

        // 测试边界值：page_size 超过上限
        let params_exceeded = PaginationParams {
            page: Some(1),
            page_size: Some(200),
            status: None,
            user_id: None,
        };
        let (_, limited_size) = params_exceeded.normalized();
        assert_eq!(limited_size, 100); // 应该被限制为 100

        // 测试默认值
        let params_default = PaginationParams {
            page: None,
            page_size: None,
            status: None,
            user_id: None,
        };
        let (default_page, default_size) = params_default.normalized();
        assert_eq!(default_page, 1);
        assert_eq!(default_size, 20);
    }

    #[test]
    fn test_pagination_offset_calculation() {
        let params = PaginationParams {
            page: Some(3),
            page_size: Some(25),
            status: None,
            user_id: None,
        };
        assert_eq!(params.offset(), 50); // (3-1)*25 = 50
    }

    #[test]
    fn test_session_stats_structure() {
        let stats = SessionStats {
            total_sessions: 100,
            active_sessions: 15,
            completed_sessions: 80,
            total_input_tokens: 50000,
            total_output_tokens: 120000,
            avg_tokens_per_session: 1700.0,
            avg_response_time_ms: 250.5,
            p50_response_time_ms: 200.0,
            p95_response_time_ms: 800.0,
            p99_response_time_ms: 1500.0,
            time_range: "last_24_hours".to_string(),
            calculated_at: "2024-06-15T12:00:00Z".to_string(),
        };

        let json = serde_json::to_value(&stats).unwrap();
        assert_eq!(json["total_sessions"], 100);
        assert_eq!(json["active_sessions"], 15);
        assert_eq!(json["avg_tokens_per_session"], 1700.0);
        assert!((json["avg_response_time_ms"].as_f64().unwrap() - 250.5).abs() < 0.001);
    }
}
