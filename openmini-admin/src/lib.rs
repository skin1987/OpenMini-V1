pub mod api;
mod auth;
pub mod config;
pub mod db;
mod error;
pub mod services;

use std::sync::Arc;

use axum::Router;

#[derive(Clone)]
pub struct AppState {
    pub pool: Arc<sqlx::SqlitePool>,
    pub proxy: Arc<services::UpstreamProxy>,
    pub config: config::AdminConfig,
}

pub fn create_app(state: AppState) -> Router {
    Router::new()
        .merge(api::create_admin_routes(&state))
        .with_state(state)
        .layer(tower_http::cors::CorsLayer::permissive())
        .layer(tower_http::trace::TraceLayer::new_for_http())
}

pub async fn seed_admin_user(pool: &sqlx::SqlitePool) -> anyhow::Result<()> {
    let exists: Option<(i64,)> = sqlx::query_as("SELECT id FROM users WHERE username = 'admin'")
        .fetch_optional(pool)
        .await?;

    if exists.is_none() {
        let password_hash = bcrypt::hash("admin123", bcrypt::DEFAULT_COST)?;
        sqlx::query("INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)")
            .bind("admin")
            .bind("admin@openmini.local")
            .bind(password_hash)
            .bind(0i32)
            .execute(pool)
            .await?;
        tracing::info!("默认管理员账号已创建: admin / admin123");
    }

    Ok(())
}
