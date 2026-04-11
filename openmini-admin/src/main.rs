mod config;
mod db;
mod auth;
mod api;
mod services;
mod error;

use std::sync::Arc;
use tokio::signal;
use tracing::info;
use axum::Router;

#[derive(Clone)]
pub struct AppState {
    pub pool: Arc<sqlx::SqlitePool>,
    pub proxy: Arc<services::UpstreamProxy>,
    pub config: config::AdminConfig,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()).as_str(),
        )
        .init();

    let config = config::load_config().unwrap_or_default();
    info!("OpenMini Admin 启动中...");
    info!("配置: {}:{}", config.server.host, config.server.port);

    let pool = Arc::new(db::init_pool(&config.database.url).await?);
    info!("数据库连接成功");

    seed_admin_user(&pool).await?;

    let proxy = Arc::new(services::UpstreamProxy::new(
        &config.upstream.base_url,
        config.upstream.timeout_secs,
    ));

    let state = AppState { pool, proxy, config: config.clone() };

    let app = Router::new()
        .merge(api::create_admin_routes(&state))
        .with_state(state)
        .layer(tower_http::cors::CorsLayer::permissive())
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let addr = format!("{}:{}", config.server.host, config.server.port);
    info!("OpenMini Admin 监听于 {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn seed_admin_user(pool: &sqlx::SqlitePool) -> anyhow::Result<()> {
    let exists: Option<(i64,)> = sqlx::query_as("SELECT id FROM users WHERE username = 'admin'")
        .fetch_optional(pool)
        .await?;

    if exists.is_none() {
        let password_hash = bcrypt::hash("admin123", bcrypt::DEFAULT_COST)?;
        sqlx::query(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)"
        )
        .bind("admin")
        .bind("admin@openmini.local")
        .bind(password_hash)
        .bind(0i32)
        .execute(pool)
        .await?;
        info!("默认管理员账号已创建: admin / admin123");
    }

    Ok(())
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let ctrl_c = async {
            signal::ctrl_c().await.ok();
        };
        let terminate = async {
            let mut sig = signal::unix::signal(signal::unix::SignalKind::terminate())
                .expect("Failed to install signal handler");
            sig.recv().await;
        };
        tokio::select! {
            _ = ctrl_c => {},
            _ = terminate => {},
        }
    }
    #[cfg(not(unix))]
    {
        signal::ctrl_c().await.ok();
    }
    info!("收到关闭信号，正在优雅关闭...");
}
