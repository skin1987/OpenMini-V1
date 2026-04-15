use std::sync::Arc;
use tokio::signal;
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "info".into())
                .as_str(),
        )
        .init();

    let config = openmini_admin::config::load_config().unwrap_or_default();
    info!("OpenMini Admin 启动中...");
    info!("配置: {}:{}", config.server.host, config.server.port);

    let pool = Arc::new(openmini_admin::db::init_pool(&config.database.url).await?);
    info!("数据库连接成功");

    openmini_admin::seed_admin_user(&pool).await?;

    let proxy = Arc::new(openmini_admin::services::UpstreamProxy::new(
        &config.upstream.base_url,
        config.upstream.timeout_secs,
    ));

    let state = openmini_admin::AppState {
        pool,
        proxy,
        config: config.clone(),
    };

    let app = openmini_admin::create_app(state);

    let addr = format!("{}:{}", config.server.host, config.server.port);
    info!("OpenMini Admin 监听于 {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

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
