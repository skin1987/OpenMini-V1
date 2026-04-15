#![allow(dead_code)]

use axum::{
    extract::{Path, Query, State},
    Json,
};
use bcrypt::{hash, DEFAULT_COST};
use chrono::Utc;
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use sqlx::Row;

use crate::db::models::UserRole;
use crate::error::AppError;
use crate::AppState;

// ============ 请求/响应结构体 ============

#[derive(Deserialize)]
pub struct UserQueryParams {
    pub page: Option<u64>,
    pub page_size: Option<u64>,
    #[allow(dead_code)]
    pub keyword: Option<String>,
    #[allow(dead_code)]
    pub role: Option<i32>,
    #[allow(dead_code)]
    pub status: Option<String>,
}

#[derive(Deserialize)]
pub struct CreateUserRequest {
    pub username: String,
    pub email: String,
    pub password: String,
    #[serde(default)]
    pub role: Option<i32>,
}

#[derive(Deserialize)]
pub struct UpdateUserRequest {
    pub email: Option<String>,
    pub role: Option<i32>,
}

#[derive(Deserialize)]
pub struct UpdateRoleRequest {
    pub role: i32,
}

#[derive(Deserialize)]
pub struct UpdateStatusRequest {
    pub status: String,
}

#[derive(Deserialize)]
pub struct ResetPasswordRequest {
    #[serde(default)]
    pub new_password: Option<String>,
}

#[derive(Deserialize)]
pub struct UpdateMyPasswordRequest {
    pub old_password: String,
    pub new_password: String,
    pub confirm_password: String,
}

#[derive(Serialize)]
pub struct PaginatedResponse<T: Serialize> {
    pub items: Vec<T>,
    pub total: u64,
    pub page: u64,
    pub page_size: u64,
}

#[derive(Serialize)]
pub struct UserInfoResponse {
    pub id: i64,
    pub username: String,
    pub email: String,
    pub role: String,
    pub status: String,
    pub last_login_at: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Serialize)]
pub struct UserDetailResponse {
    #[serde(flatten)]
    pub user: UserInfoResponse,
    pub login_history: Vec<LoginHistoryRecord>,
}

#[derive(Serialize)]
pub struct LoginHistoryRecord {
    pub login_time: Option<String>,
    pub ip_address: Option<String>,
}

#[derive(Serialize)]
pub struct ResetPasswordResponse {
    pub success: bool,
    pub message: String,
    pub temporary_password: Option<String>,
}

// ============ 辅助函数 ============

fn row_to_user_info(row: sqlx::sqlite::SqliteRow) -> UserInfoResponse {
    let is_active: bool = row.get("is_active");
    let role_i32: i32 = row.get("role");

    UserInfoResponse {
        id: row.get("id"),
        username: row.get("username"),
        email: row.get("email"),
        role: match UserRole::from(role_i32) {
            UserRole::Admin => "admin".to_string(),
            UserRole::Operator => "operator".to_string(),
            UserRole::Viewer => "viewer".to_string(),
        },
        status: if is_active { "active" } else { "disabled" }.to_string(),
        last_login_at: row.get("last_login_at"),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
    }
}

fn generate_temporary_password() -> String {
    let mut rng = thread_rng();
    (0..12)
        .map(|_| {
            let b = rng.gen::<u8>();
            match b % 3 {
                0 => (b'0' + (b % 10)) as char,
                1 => (b'a' + (b % 26)) as char,
                _ => (b'A' + (b % 26)) as char,
            }
        })
        .collect()
}

// ============ API 处理函数 ============

/// 获取用户列表（分页、搜索、筛选）
pub async fn list_users(
    State(state): State<AppState>,
    Query(params): Query<UserQueryParams>,
) -> Result<Json<PaginatedResponse<UserInfoResponse>>, AppError> {
    let page = params.page.unwrap_or(1);
    let size = params.page_size.unwrap_or(20);

    // 查询总数
    let count_row = sqlx::query("SELECT COUNT(*) FROM users")
        .fetch_one(&*state.pool)
        .await?;
    let total: i64 = count_row.get(0);

    // 查询数据（按创建时间倒序）
    let rows = sqlx::query("SELECT * FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?")
        .bind(size as i64)
        .bind(((page - 1) * size) as i64)
        .fetch_all(&*state.pool)
        .await?;

    let users: Vec<UserInfoResponse> = rows.into_iter().map(row_to_user_info).collect();

    Ok(Json(PaginatedResponse {
        items: users,
        total: total as u64,
        page,
        page_size: size,
    }))
}

/// 创建用户
pub async fn create_user(
    State(state): State<AppState>,
    Json(req): Json<CreateUserRequest>,
) -> Result<Json<UserInfoResponse>, AppError> {
    // 验证密码长度
    if req.password.len() < 6 {
        return Err(AppError::BadRequest("密码长度至少6位".into()));
    }

    // bcrypt 加密密码
    let password_hash = hash(&req.password, DEFAULT_COST)
        .map_err(|e| AppError::Internal(anyhow::anyhow!("密码加密失败: {}", e)))?;

    let now = Utc::now().to_rfc3339();
    let role = req.role.unwrap_or(UserRole::Operator as i32);

    let result = sqlx::query(
        "INSERT INTO users (username, email, password_hash, role, is_active, created_at, updated_at) VALUES (?, ?, ?, ?, 1, ?, ?)"
    )
    .bind(&req.username)
    .bind(&req.email)
    .bind(&password_hash)
    .bind(role)
    .bind(&now)
    .bind(&now)
    .execute(&*state.pool)
    .await?;

    let new_id = result.last_insert_rowid();

    // 查询刚创建的用户
    let row = sqlx::query("SELECT * FROM users WHERE id = ?")
        .bind(new_id)
        .fetch_one(&*state.pool)
        .await?;

    Ok(Json(row_to_user_info(row)))
}

/// 获取用户详情
pub async fn get_user_detail(
    Path(id): Path<i64>,
    State(state): State<AppState>,
) -> Result<Json<UserDetailResponse>, AppError> {
    let row = sqlx::query("SELECT * FROM users WHERE id = ?")
        .bind(id)
        .fetch_optional(&*state.pool)
        .await?
        .ok_or_else(|| AppError::NotFound("用户不存在".into()))?;

    let user_info = row_to_user_info(row);

    // TODO: 从审计日志表获取登录历史（如果有的话）
    let login_history = vec![LoginHistoryRecord {
        login_time: None,
        ip_address: None,
    }];

    Ok(Json(UserDetailResponse {
        user: user_info,
        login_history,
    }))
}

/// 更新用户信息
pub async fn update_user(
    Path(id): Path<i64>,
    State(state): State<AppState>,
    Json(req): Json<UpdateUserRequest>,
) -> Result<Json<UserInfoResponse>, AppError> {
    let now = Utc::now().to_rfc3339();

    // 动态构建更新语句
    let mut updates = Vec::new();
    if req.email.is_some() {
        updates.push("email = ?");
    }
    if req.role.is_some() {
        updates.push("role = ?");
    }
    updates.push("updated_at = ?");

    if updates.is_empty() {
        return Err(AppError::BadRequest("没有要更新的字段".into()));
    }

    let sql = format!(
        "UPDATE users SET {} WHERE id = ? RETURNING *",
        updates.join(", ")
    );

    let mut query = sqlx::query(&sql);
    if let Some(email) = &req.email {
        query = query.bind(email);
    }
    if let Some(role) = req.role {
        query = query.bind(role);
    }
    query = query.bind(&now);
    query = query.bind(id);

    let row = query.fetch_one(&*state.pool).await?;

    Ok(Json(row_to_user_info(row)))
}

/// 删除用户（硬删除）
pub async fn delete_user(
    Path(id): Path<i64>,
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, AppError> {
    // 检查是否存在
    let exists = sqlx::query("SELECT id FROM users WHERE id = ?")
        .bind(id)
        .fetch_optional(&*state.pool)
        .await?
        .is_some();

    if !exists {
        return Err(AppError::NotFound("用户不存在".into()));
    }

    // 执行删除
    sqlx::query("DELETE FROM users WHERE id = ?")
        .bind(id)
        .execute(&*state.pool)
        .await?;

    Ok(Json(serde_json::json!({
        "success": true,
        "message": "用户已成功删除"
    })))
}

/// 修改用户角色
pub async fn update_role(
    Path(id): Path<i64>,
    State(state): State<AppState>,
    Json(req): Json<UpdateRoleRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let now = Utc::now().to_rfc3339();

    // 验证角色值
    match req.role {
        0..=2 => {}
        _ => return Err(AppError::BadRequest("无效的角色值".into())),
    }

    let row = sqlx::query("UPDATE users SET role = ?, updated_at = ? WHERE id = ? RETURNING *")
        .bind(req.role)
        .bind(&now)
        .bind(id)
        .fetch_one(&*state.pool)
        .await?;

    let role_name = match UserRole::from(req.role) {
        UserRole::Admin => "管理员",
        UserRole::Operator => "操作员",
        UserRole::Viewer => "查看者",
    };

    Ok(Json(serde_json::json!({
        "success": true,
        "message": format!("角色已修改为：{}", role_name),
        "user": row_to_user_info(row)
    })))
}

/// 启用/禁用用户
pub async fn update_status(
    Path(id): Path<i64>,
    State(state): State<AppState>,
    Json(req): Json<UpdateStatusRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let now = Utc::now().to_rfc3339();

    let is_active = match req.status.to_lowercase().as_str() {
        "active" | "enabled" | "1" | "true" => true,
        "disabled" | "0" | "false" => false,
        _ => {
            return Err(AppError::BadRequest(
                "无效的状态值，应为 active 或 disabled".into(),
            ))
        }
    };

    let row =
        sqlx::query("UPDATE users SET is_active = ?, updated_at = ? WHERE id = ? RETURNING *")
            .bind(is_active)
            .bind(&now)
            .bind(id)
            .fetch_one(&*state.pool)
            .await?;

    let status_text = if is_active { "启用" } else { "禁用" };

    Ok(Json(serde_json::json!({
        "success": true,
        "message": format!("用户已{}", status_text),
        "user": row_to_user_info(row)
    })))
}

/// 重置密码（生成临时密码或使用指定密码）
pub async fn reset_password(
    Path(id): Path<i64>,
    State(state): State<AppState>,
    Json(req): Json<ResetPasswordRequest>,
) -> Result<Json<ResetPasswordResponse>, AppError> {
    let new_password = match req.new_password {
        Some(pwd) => {
            if pwd.len() < 6 {
                return Err(AppError::BadRequest("密码长度至少6位".into()));
            }
            pwd
        }
        None => generate_temporary_password(),
    };

    // bcrypt 加密新密码
    let password_hash = hash(&new_password, DEFAULT_COST)
        .map_err(|e| AppError::Internal(anyhow::anyhow!("密码加密失败: {}", e)))?;

    let now = Utc::now().to_rfc3339();

    sqlx::query("UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?")
        .bind(&password_hash)
        .bind(&now)
        .bind(id)
        .execute(&*state.pool)
        .await?;

    Ok(Json(ResetPasswordResponse {
        success: true,
        message: "密码重置成功".to_string(),
        temporary_password: Some(new_password),
    }))
}

/// 当前用户修改自己的密码
pub async fn update_my_password(
    State(state): State<AppState>,
    axum::Extension(auth_user): axum::Extension<crate::auth::jwt::Claims>,
    Json(req): Json<UpdateMyPasswordRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    // 验证两次输入的新密码是否一致
    if req.new_password != req.confirm_password {
        return Err(AppError::BadRequest("两次输入的密码不一致".into()));
    }

    // 验证新密码长度
    if req.new_password.len() < 6 {
        return Err(AppError::BadRequest("新密码长度至少6位".into()));
    }

    // 查询当前用户
    let row = sqlx::query("SELECT * FROM users WHERE id = ?")
        .bind(auth_user.sub)
        .fetch_one(&*state.pool)
        .await?;

    let current_hash: String = row.get("password_hash");

    // 验证旧密码
    let valid = bcrypt::verify(&req.old_password, &current_hash).unwrap_or(false);

    if !valid {
        return Err(AppError::BadRequest("旧密码错误".into()));
    }

    // 加密新密码
    let new_hash = hash(&req.new_password, DEFAULT_COST)
        .map_err(|e| AppError::Internal(anyhow::anyhow!("密码加密失败: {}", e)))?;

    let now = Utc::now().to_rfc3339();

    sqlx::query("UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?")
        .bind(&new_hash)
        .bind(&now)
        .bind(auth_user.sub)
        .execute(&*state.pool)
        .await?;

    Ok(Json(serde_json::json!({
        "success": true,
        "message": "密码修改成功"
    })))
}

// ============ 单元测试 ============

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::State;
    use std::sync::Arc;

    /// 创建用于测试的 SQLite 内存数据库连接池
    async fn init_test_pool() -> sqlx::SqlitePool {
        let pool = sqlx::SqlitePool::connect(":memory:").await.unwrap();

        // 执行迁移
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role INTEGER NOT NULL DEFAULT 0,
                is_active BOOLEAN NOT NULL DEFAULT 1,
                last_login_at DATETIME,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            "#,
        )
        .execute(&pool)
        .await
        .unwrap();

        pool
    }

    /// 创建测试用的 AppState（使用 mock proxy）
    async fn create_test_state(pool: &sqlx::SqlitePool) -> AppState {
        AppState {
            pool: Arc::new(pool.clone()),
            proxy: Arc::new(crate::services::UpstreamProxy::new(
                "http://localhost:8080",
                30,
            )),
            config: crate::config::AdminConfig::default(),
        }
    }

    /// 辅助函数：在数据库中插入一个测试用户，返回用户 ID
    async fn create_test_user(
        pool: &sqlx::SqlitePool,
        username: &str,
        email: &str,
        password: &str,
        role: i32,
    ) -> i64 {
        let password_hash = bcrypt::hash(password, bcrypt::DEFAULT_COST).unwrap();
        let result = sqlx::query(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
        )
        .bind(username)
        .bind(email)
        .bind(&password_hash)
        .bind(role)
        .execute(pool)
        .await
        .unwrap();
        result.last_insert_rowid()
    }

    // ==================== list_users 测试 ====================

    #[tokio::test]
    async fn test_list_users_empty() {
        let pool = init_test_pool().await;
        let state = create_test_state(&pool).await;

        let response = list_users(
            State(state),
            Query(UserQueryParams {
                page: Some(1),
                page_size: Some(20),
                keyword: None,
                role: None,
                status: None,
            }),
        )
        .await;

        assert!(response.is_ok());
        let json = response.unwrap();
        assert_eq!(json.items.len(), 0);
        assert_eq!(json.total, 0);
        assert_eq!(json.page, 1);
        assert_eq!(json.page_size, 20);
    }

    #[tokio::test]
    async fn test_list_users_with_data() {
        let pool = init_test_pool().await;
        let state = create_test_state(&pool).await;

        // 插入测试数据
        create_test_user(&pool, "user1", "user1@test.com", "password123", 0).await;
        create_test_user(&pool, "user2", "user2@test.com", "password123", 1).await;

        let response = list_users(
            State(state),
            Query(UserQueryParams {
                page: Some(1),
                page_size: Some(20),
                keyword: None,
                role: None,
                status: None,
            }),
        )
        .await;

        assert!(response.is_ok());
        let json = response.unwrap();
        assert_eq!(json.items.len(), 2);
        assert_eq!(json.total, 2);
    }

    #[tokio::test]
    async fn test_list_users_pagination() {
        let pool = init_test_pool().await;
        let state = create_test_state(&pool).await;

        // 插入 5 个用户
        for i in 0..5 {
            create_test_user(
                &pool,
                &format!("user{}", i),
                &format!("user{}@test.com", i),
                "password123",
                0,
            )
            .await;
        }

        // 第一页，每页 2 条
        let response = list_users(
            State(state),
            Query(UserQueryParams {
                page: Some(1),
                page_size: Some(2),
                keyword: None,
                role: None,
                status: None,
            }),
        )
        .await;

        assert!(response.is_ok());
        let json = response.unwrap();
        assert_eq!(json.items.len(), 2);
        assert_eq!(json.total, 5);
        assert_eq!(json.page, 1);

        // 第二页
        let state = create_test_state(&pool).await;
        let response = list_users(
            State(state),
            Query(UserQueryParams {
                page: Some(2),
                page_size: Some(2),
                keyword: None,
                role: None,
                status: None,
            }),
        )
        .await;

        assert!(response.is_ok());
        let json = response.unwrap();
        assert_eq!(json.items.len(), 2);
        assert_eq!(json.page, 2);
    }

    #[tokio::test]
    async fn test_list_users_default_params() {
        let pool = init_test_pool().await;
        let state = create_test_state(&pool).await;

        let response = list_users(
            State(state),
            Query(UserQueryParams {
                page: None,
                page_size: None,
                keyword: None,
                role: None,
                status: None,
            }),
        )
        .await;

        assert!(response.is_ok());
        let json = response.unwrap();
        // 默认应该是第 1 页，每页 20 条
        assert_eq!(json.page, 1);
        assert_eq!(json.page_size, 20);
    }

    // ==================== create_user 测试 ====================

    #[tokio::test]
    async fn test_create_user_success() {
        let pool = init_test_pool().await;
        let state = create_test_state(&pool).await;

        let request = CreateUserRequest {
            username: "testuser".to_string(),
            email: "test@example.com".to_string(),
            password: "securepass123".to_string(),
            role: Some(1),
        };

        let response = create_user(State(state), Json(request)).await;

        assert!(response.is_ok());
        let json = response.unwrap();
        assert_eq!(json.username, "testuser");
        assert_eq!(json.email, "test@example.com");
        assert_eq!(json.role, "operator");
        assert_eq!(json.status, "active");
    }

    #[tokio::test]
    async fn test_create_user_default_role() {
        let pool = init_test_pool().await;
        let state = create_test_state(&pool).await;

        let request = CreateUserRequest {
            username: "defrole".to_string(),
            email: "defrole@test.com".to_string(),
            password: "password123".to_string(),
            role: None, // 不指定角色，应该默认为 Operator
        };

        let response = create_user(State(state), Json(request)).await;

        assert!(response.is_ok());
        let json = response.unwrap();
        assert_eq!(json.role, "operator");
    }

    #[tokio::test]
    async fn test_create_user_short_password() {
        let pool = init_test_pool().await;
        let state = create_test_state(&pool).await;

        let request = CreateUserRequest {
            username: "shortpass".to_string(),
            email: "short@test.com".to_string(),
            password: "12345".to_string(), // 只有5位
            role: None,
        };

        let response = create_user(State(state), Json(request)).await;

        assert!(response.is_err());
        match response.err().unwrap() {
            AppError::BadRequest(msg) => assert!(msg.contains("密码长度至少6位")),
            other => panic!("Expected BadRequest error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_create_user_duplicate_username() {
        let pool = init_test_pool().await;
        create_test_user(&pool, "duplicate", "dup1@test.com", "password123", 0).await;
        let state = create_test_state(&pool).await;

        let request = CreateUserRequest {
            username: "duplicate".to_string(), // 重复的用户名
            email: "dup2@test.com".to_string(),
            password: "password123".to_string(),
            role: None,
        };

        let response = create_user(State(state), Json(request)).await;

        // 应该返回数据库错误（UNIQUE 约束冲突）
        assert!(response.is_err());
    }

    #[tokio::test]
    async fn test_create_user_duplicate_email() {
        let pool = init_test_pool().await;
        create_test_user(&pool, "unique1", "same@email.com", "password123", 0).await;
        let state = create_test_state(&pool).await;

        let request = CreateUserRequest {
            username: "unique2".to_string(),
            email: "same@email.com".to_string(), // 重复的邮箱
            password: "password123".to_string(),
            role: None,
        };

        let response = create_user(State(state), Json(request)).await;

        assert!(response.is_err());
    }

    // ==================== get_user_detail 测试 ====================

    #[tokio::test]
    async fn test_get_user_detail_success() {
        let pool = init_test_pool().await;
        let user_id =
            create_test_user(&pool, "detailuser", "detail@test.com", "password123", 0).await;
        let state = create_test_state(&pool).await;

        let response = get_user_detail(axum::extract::Path(user_id), State(state)).await;

        assert!(response.is_ok());
        let json = response.unwrap();
        assert_eq!(json.user.username, "detailuser");
        assert_eq!(json.user.email, "detail@test.com");
        assert_eq!(json.user.role, "admin");
        // 登录历史应该有记录（即使是空的）
        assert!(!json.login_history.is_empty());
    }

    #[tokio::test]
    async fn test_get_user_detail_not_found() {
        let pool = init_test_pool().await;
        let state = create_test_state(&pool).await;

        let response = get_user_detail(axum::extract::Path(99999i64), State(state)).await;

        assert!(response.is_err());
        match response.err().unwrap() {
            AppError::NotFound(msg) => assert!(msg.contains("用户不存在")),
            other => panic!("Expected NotFound error, got: {:?}", other),
        }
    }

    // ==================== update_user 测试 ====================

    #[tokio::test]
    async fn test_update_user_email_only() {
        let pool = init_test_pool().await;
        let user_id =
            create_test_user(&pool, "updateemail", "old@email.com", "password123", 0).await;
        let state = create_test_state(&pool).await;

        let request = UpdateUserRequest {
            email: Some("new@email.com".to_string()),
            role: None,
        };

        let response = update_user(axum::extract::Path(user_id), State(state), Json(request)).await;

        assert!(response.is_ok());
        let json = response.unwrap();
        assert_eq!(json.email, "new@email.com");
    }

    #[tokio::test]
    async fn test_update_user_role_only() {
        let pool = init_test_pool().await;
        let user_id =
            create_test_user(&pool, "updaterole", "role@test.com", "password123", 0).await;
        let state = create_test_state(&pool).await;

        let request = UpdateUserRequest {
            email: None,
            role: Some(2), // Viewer
        };

        let response = update_user(axum::extract::Path(user_id), State(state), Json(request)).await;

        assert!(response.is_ok());
        let json = response.unwrap();
        assert_eq!(json.role, "viewer");
    }

    #[tokio::test]
    async fn test_update_user_both_fields() {
        let pool = init_test_pool().await;
        let user_id =
            create_test_user(&pool, "updateboth", "both@test.com", "password123", 0).await;
        let state = create_test_state(&pool).await;

        let request = UpdateUserRequest {
            email: Some("updated@test.com".to_string()),
            role: Some(1),
        };

        let response = update_user(axum::extract::Path(user_id), State(state), Json(request)).await;

        assert!(response.is_ok());
        let json = response.unwrap();
        assert_eq!(json.email, "updated@test.com");
        assert_eq!(json.role, "operator");
    }

    #[tokio::test]
    async fn test_update_user_only_updated_at() {
        let pool = init_test_pool().await;
        let user_id =
            create_test_user(&pool, "onlyupdate", "onlyupdate@test.com", "password123", 0).await;
        let state = create_test_state(&pool).await;

        // 当 email 和 role 都是 None 时，仍然会更新 updated_at 字段
        let request = UpdateUserRequest {
            email: None,
            role: None,
        };

        let response = update_user(axum::extract::Path(user_id), State(state), Json(request)).await;

        // 这个请求会成功，因为 updated_at 总是被更新
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn test_update_user_not_found() {
        let pool = init_test_pool().await;
        let state = create_test_state(&pool).await;

        let request = UpdateUserRequest {
            email: Some("test@test.com".to_string()),
            role: None,
        };

        let response =
            update_user(axum::extract::Path(99999i64), State(state), Json(request)).await;

        assert!(response.is_err()); // 数据库错误（找不到行）
    }

    // ==================== delete_user 测试 ====================

    #[tokio::test]
    async fn test_delete_user_success() {
        let pool = init_test_pool().await;
        let user_id =
            create_test_user(&pool, "todelete", "delete@test.com", "password123", 0).await;
        let state = create_test_state(&pool).await;

        let response = delete_user(axum::extract::Path(user_id), State(state)).await;

        assert!(response.is_ok());
        let json = response.unwrap();
        assert_eq!(json.get("success").and_then(|v| v.as_bool()), Some(true));

        // 验证用户确实被删除了
        let exists: Option<(i64,)> = sqlx::query_as("SELECT id FROM users WHERE id = ?")
            .bind(user_id)
            .fetch_optional(&pool)
            .await
            .unwrap();
        assert!(exists.is_none());
    }

    #[tokio::test]
    async fn test_delete_user_not_found() {
        let pool = init_test_pool().await;
        let state = create_test_state(&pool).await;

        let response = delete_user(axum::extract::Path(99999i64), State(state)).await;

        assert!(response.is_err());
        match response.err().unwrap() {
            AppError::NotFound(msg) => assert!(msg.contains("用户不存在")),
            other => panic!("Expected NotFound error, got: {:?}", other),
        }
    }

    // ==================== update_role 测试 ====================

    #[tokio::test]
    async fn test_update_role_to_admin() {
        let pool = init_test_pool().await;
        let user_id = create_test_user(&pool, "torole", "role@test.com", "password123", 2).await; // 初始是 Viewer
        let state = create_test_state(&pool).await;

        let request = UpdateRoleRequest { role: 0 }; // 改为 Admin

        let response = update_role(axum::extract::Path(user_id), State(state), Json(request)).await;

        assert!(response.is_ok());
        let json = response.unwrap();
        assert_eq!(json.get("success").and_then(|v| v.as_bool()), Some(true));
        assert!(json
            .get("message")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("管理员"));
    }

    #[tokio::test]
    async fn test_update_role_to_operator() {
        let pool = init_test_pool().await;
        let user_id = create_test_user(&pool, "toop", "op@test.com", "password123", 0).await; // 初始是 Admin
        let state = create_test_state(&pool).await;

        let request = UpdateRoleRequest { role: 1 }; // 改为 Operator

        let response = update_role(axum::extract::Path(user_id), State(state), Json(request)).await;

        assert!(response.is_ok());
        let json = response.unwrap();
        assert!(json
            .get("message")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("操作员"));
    }

    #[tokio::test]
    async fn test_update_role_to_viewer() {
        let pool = init_test_pool().await;
        let user_id = create_test_user(&pool, "toview", "view@test.com", "password123", 0).await;
        let state = create_test_state(&pool).await;

        let request = UpdateRoleRequest { role: 2 }; // 改为 Viewer

        let response = update_role(axum::extract::Path(user_id), State(state), Json(request)).await;

        assert!(response.is_ok());
        let json = response.unwrap();
        assert!(json
            .get("message")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("查看者"));
    }

    #[tokio::test]
    async fn test_update_role_invalid_value() {
        let pool = init_test_pool().await;
        let user_id =
            create_test_user(&pool, "invalidrole", "invalid@test.com", "password123", 0).await;
        let state = create_test_state(&pool).await;

        let request = UpdateRoleRequest { role: 99 }; // 无效的角色值

        let response = update_role(axum::extract::Path(user_id), State(state), Json(request)).await;

        assert!(response.is_err());
        match response.err().unwrap() {
            AppError::BadRequest(msg) => assert!(msg.contains("无效的角色值")),
            other => panic!("Expected BadRequest error, got: {:?}", other),
        }
    }

    // ==================== update_status 测试 ====================

    #[tokio::test]
    async fn test_update_status_to_active() {
        let pool = init_test_pool().await;
        let user_id =
            create_test_user(&pool, "statusactive", "status@test.com", "password123", 0).await;
        let state = create_test_state(&pool).await;

        let request = UpdateStatusRequest {
            status: "active".to_string(),
        };

        let response =
            update_status(axum::extract::Path(user_id), State(state), Json(request)).await;

        assert!(response.is_ok());
        let json = response.unwrap();
        assert_eq!(json.get("success").and_then(|v| v.as_bool()), Some(true));
        assert!(json
            .get("message")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("启用"));
    }

    #[tokio::test]
    async fn test_update_status_to_disabled() {
        let pool = init_test_pool().await;
        let user_id =
            create_test_user(&pool, "statusdisable", "disable@test.com", "password123", 0).await;
        let state = create_test_state(&pool).await;

        let request = UpdateStatusRequest {
            status: "disabled".to_string(),
        };

        let response =
            update_status(axum::extract::Path(user_id), State(state), Json(request)).await;

        assert!(response.is_ok());
        let json = response.unwrap();
        assert!(json
            .get("message")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("禁用"));
    }

    #[tokio::test]
    async fn test_update_status_variants() {
        let pool = init_test_pool().await;

        // 测试各种有效的状态值
        for status in &["enabled", "1", "true"] {
            let user_id = create_test_user(
                &pool,
                &format!("status_{}", status),
                &format!("{}@test.com", status),
                "password123",
                0,
            )
            .await;
            let state = create_test_state(&pool).await;

            let request = UpdateStatusRequest {
                status: status.to_string(),
            };
            let response =
                update_status(axum::extract::Path(user_id), State(state), Json(request)).await;

            assert!(response.is_ok(), "Status '{}' should be valid", status);
        }
    }

    #[tokio::test]
    async fn test_update_status_invalid() {
        let pool = init_test_pool().await;
        let user_id = create_test_user(
            &pool,
            "statusinvalid",
            "invalidstatus@test.com",
            "password123",
            0,
        )
        .await;
        let state = create_test_state(&pool).await;

        let request = UpdateStatusRequest {
            status: "invalid_status".to_string(),
        };

        let response =
            update_status(axum::extract::Path(user_id), State(state), Json(request)).await;

        assert!(response.is_err());
        match response.err().unwrap() {
            AppError::BadRequest(msg) => assert!(msg.contains("无效的状态值")),
            other => panic!("Expected BadRequest error, got: {:?}", other),
        }
    }

    // ==================== reset_password 测试 ====================

    #[tokio::test]
    async fn test_reset_password_auto_generate() {
        let pool = init_test_pool().await;
        let user_id =
            create_test_user(&pool, "resetauto", "resetauto@test.com", "password123", 0).await;
        let state = create_test_state(&pool).await;

        let request = ResetPasswordRequest { new_password: None }; // 自动生成临时密码

        let response =
            reset_password(axum::extract::Path(user_id), State(state), Json(request)).await;

        assert!(response.is_ok());
        let json = response.unwrap();
        assert!(json.success);
        assert!(json.temporary_password.is_some());
        let temp_pass = json.temporary_password.clone().unwrap();
        assert_eq!(temp_pass.len(), 12); // 生成的临时密码长度应该是12位
    }

    #[tokio::test]
    async fn test_reset_password_custom_password() {
        let pool = init_test_pool().await;
        let user_id = create_test_user(
            &pool,
            "resetcustom",
            "resetcustom@test.com",
            "password123",
            0,
        )
        .await;
        let state = create_test_state(&pool).await;

        let new_pass = "newsecurepassword";
        let request = ResetPasswordRequest {
            new_password: Some(new_pass.to_string()),
        };

        let response =
            reset_password(axum::extract::Path(user_id), State(state), Json(request)).await;

        assert!(response.is_ok());
        let json = response.unwrap();
        assert!(json.success);
        assert_eq!(json.temporary_password.as_deref(), Some(new_pass));
    }

    #[tokio::test]
    async fn test_reset_password_short_custom_password() {
        let pool = init_test_pool().await;
        let user_id =
            create_test_user(&pool, "resetshort", "resetshort@test.com", "password123", 0).await;
        let state = create_test_state(&pool).await;

        let request = ResetPasswordRequest {
            new_password: Some("12345".to_string()),
        }; // 太短

        let response =
            reset_password(axum::extract::Path(user_id), State(state), Json(request)).await;

        assert!(response.is_err());
        match response.err().unwrap() {
            AppError::BadRequest(msg) => assert!(msg.contains("密码长度至少6位")),
            other => panic!("Expected BadRequest error, got: {:?}", other),
        }
    }

    // ==================== update_my_password 测试 ====================

    #[tokio::test]
    async fn test_update_my_password_success() {
        let pool = init_test_pool().await;
        let user_id = create_test_user(
            &pool,
            "mypassword",
            "mypassword@test.com",
            "oldpassword123",
            0,
        )
        .await;
        let state = create_test_state(&pool).await;

        let claims = crate::auth::jwt::Claims {
            sub: user_id,
            username: "mypassword".to_string(),
            role: "admin".to_string(),
            exp: (chrono::Utc::now() + chrono::Duration::hours(1)).timestamp(),
            iat: chrono::Utc::now().timestamp(),
        };

        let request = UpdateMyPasswordRequest {
            old_password: "oldpassword123".to_string(),
            new_password: "newpassword123".to_string(),
            confirm_password: "newpassword123".to_string(),
        };

        let response =
            update_my_password(State(state), axum::Extension(claims), Json(request)).await;

        assert!(response.is_ok());
        let json = response.unwrap();
        assert_eq!(json.get("success").and_then(|v| v.as_bool()), Some(true));
    }

    #[tokio::test]
    async fn test_update_my_password_wrong_old_password() {
        let pool = init_test_pool().await;
        let user_id =
            create_test_user(&pool, "wrongold", "wrongold@test.com", "correctpassword", 0).await;
        let state = create_test_state(&pool).await;

        let claims = crate::auth::jwt::Claims {
            sub: user_id,
            username: "wrongold".to_string(),
            role: "admin".to_string(),
            exp: (chrono::Utc::now() + chrono::Duration::hours(1)).timestamp(),
            iat: chrono::Utc::now().timestamp(),
        };

        let request = UpdateMyPasswordRequest {
            old_password: "wrongpassword".to_string(), // 错误的旧密码
            new_password: "newpassword123".to_string(),
            confirm_password: "newpassword123".to_string(),
        };

        let response =
            update_my_password(State(state), axum::Extension(claims), Json(request)).await;

        assert!(response.is_err());
        match response.err().unwrap() {
            AppError::BadRequest(msg) => assert!(msg.contains("旧密码错误")),
            other => panic!("Expected BadRequest error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_update_my_password_mismatch() {
        let pool = init_test_pool().await;
        let user_id =
            create_test_user(&pool, "mismatch", "mismatch@test.com", "password123", 0).await;
        let state = create_test_state(&pool).await;

        let claims = crate::auth::jwt::Claims {
            sub: user_id,
            username: "mismatch".to_string(),
            role: "admin".to_string(),
            exp: (chrono::Utc::now() + chrono::Duration::hours(1)).timestamp(),
            iat: chrono::Utc::now().timestamp(),
        };

        let request = UpdateMyPasswordRequest {
            old_password: "password123".to_string(),
            new_password: "newpass1".to_string(),
            confirm_password: "newpass2".to_string(), // 不一致
        };

        let response =
            update_my_password(State(state), axum::Extension(claims), Json(request)).await;

        assert!(response.is_err());
        match response.err().unwrap() {
            AppError::BadRequest(msg) => assert!(msg.contains("两次输入的密码不一致")),
            other => panic!("Expected BadRequest error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_update_my_password_too_short() {
        let pool = init_test_pool().await;
        let user_id =
            create_test_user(&pool, "tooshort", "tooshort@test.com", "password123", 0).await;
        let state = create_test_state(&pool).await;

        let claims = crate::auth::jwt::Claims {
            sub: user_id,
            username: "tooshort".to_string(),
            role: "admin".to_string(),
            exp: (chrono::Utc::now() + chrono::Duration::hours(1)).timestamp(),
            iat: chrono::Utc::now().timestamp(),
        };

        let request = UpdateMyPasswordRequest {
            old_password: "password123".to_string(),
            new_password: "12345".to_string(), // 太短
            confirm_password: "12345".to_string(),
        };

        let response =
            update_my_password(State(state), axum::Extension(claims), Json(request)).await;

        assert!(response.is_err());
        match response.err().unwrap() {
            AppError::BadRequest(msg) => assert!(msg.contains("新密码长度至少6位")),
            other => panic!("Expected BadRequest error, got: {:?}", other),
        }
    }

    // ==================== 辅助函数测试 ====================

    #[test]
    fn test_generate_temporary_password_length() {
        let password = generate_temporary_password();
        assert_eq!(password.len(), 12);
    }

    #[test]
    fn test_generate_temporary_password_uniqueness() {
        let pass1 = generate_temporary_password();
        let pass2 = generate_temporary_password();
        // 极小概率会相同，但通常应该不同
        assert_ne!(pass1, pass2, "Generated passwords should be unique");
    }

    #[test]
    fn test_generate_temporary_password_format() {
        let password = generate_temporary_password();
        // 密码应该只包含数字、大小写字母
        for c in password.chars() {
            assert!(
                c.is_ascii_alphanumeric(),
                "Password should only contain alphanumeric characters, found: '{}'",
                c
            );
        }
    }
}
