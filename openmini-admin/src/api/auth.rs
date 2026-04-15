use axum::{extract::State, Json};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use sqlx::Row;

use crate::auth::jwt::create_token;
use crate::db::models::User;
use crate::error::AppError;
use crate::AppState;

#[derive(Deserialize)]
pub struct LoginRequest {
    pub username: String,
    pub password: String,
}

#[derive(Serialize)]
pub struct LoginResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: i64,
    pub user: UserInfo,
}

#[derive(Serialize)]
pub struct UserInfo {
    pub id: i64,
    pub username: String,
    pub email: String,
    pub role: String,
}

fn row_to_user(row: sqlx::sqlite::SqliteRow) -> User {
    User {
        id: row.get("id"),
        username: row.get("username"),
        email: row.get("email"),
        password_hash: row.get("password_hash"),
        role: row.get("role"),
        is_active: row.get("is_active"),
        last_login_at: row.get("last_login_at"),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
    }
}

pub async fn login(
    State(state): State<AppState>,
    Json(req): Json<LoginRequest>,
) -> Result<Json<LoginResponse>, AppError> {
    let row = sqlx::query("SELECT * FROM users WHERE username = ? AND is_active = 1")
        .bind(&req.username)
        .fetch_optional(&*state.pool)
        .await?
        .ok_or_else(|| AppError::BadRequest("用户名或密码错误".into()))?;

    let user = row_to_user(row);

    let valid = bcrypt::verify(&req.password, &user.password_hash).unwrap_or(false);

    if !valid {
        return Err(AppError::BadRequest("用户名或密码错误".into()));
    }

    let now = Utc::now().to_rfc3339();
    sqlx::query("UPDATE users SET last_login_at = ? WHERE id = ?")
        .bind(now)
        .bind(user.id)
        .execute(&*state.pool)
        .await?;

    let secret = &state.config.jwt.secret;
    let hours = state.config.jwt.expiration_hours;
    let token = create_token(
        &user.id,
        &user.username,
        &user.role.to_string(),
        secret,
        hours,
    )?;

    Ok(Json(LoginResponse {
        access_token: token,
        token_type: "Bearer".into(),
        expires_in: (hours as i64) * 3600,
        user: UserInfo {
            id: user.id,
            username: user.username.clone(),
            email: user.email,
            role: match user.role {
                0 => "admin".to_string(),
                1 => "operator".to_string(),
                _ => "viewer".to_string(),
            },
        },
    }))
}
