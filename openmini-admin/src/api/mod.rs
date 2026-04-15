pub mod alert;
pub mod apikey;
pub mod audit;
pub mod auth;
pub mod config;
pub mod metrics;
pub mod model;
pub mod service;
pub mod session;
pub mod user;

use crate::{auth::middleware::create_auth_middleware, AppState};
use axum::Router;

pub fn create_admin_routes(state: &AppState) -> Router<AppState> {
    let auth_mw = create_auth_middleware(state.config.jwt.secret.clone());

    let protected = Router::new()
        // ==================== 服务管理 ====================
        .route(
            "/admin/service/status",
            axum::routing::get(service::get_status),
        )
        .route(
            "/admin/service/workers",
            axum::routing::get(service::get_workers),
        )
        .route(
            "/admin/service/restart",
            axum::routing::post(service::restart_service),
        )
        .route(
            "/admin/service/stop",
            axum::routing::post(service::stop_service),
        )
        // ==================== 模型管理 ====================
        .route(
            "/admin/model",
            axum::routing::get(model::list_models).post(model::load_model),
        )
        .route(
            "/admin/model/:id",
            axum::routing::get(model::get_model_detail).put(model::update_model_config),
        )
        .route(
            "/admin/model/switch",
            axum::routing::post(model::switch_model),
        )
        .route(
            "/admin/model/:id/unload",
            axum::routing::post(model::unload_model),
        )
        .route(
            "/admin/model/:id/health",
            axum::routing::get(model::check_health),
        )
        // ==================== 会话监控 ====================
        .route(
            "/admin/sessions",
            axum::routing::get(session::list_active_sessions),
        )
        .route(
            "/admin/sessions/:session_id",
            axum::routing::get(session::get_session_detail),
        )
        .route(
            "/admin/sessions/stats",
            axum::routing::get(session::get_session_stats),
        )
        .route(
            "/admin/sessions/:session_id/terminate",
            axum::routing::post(session::terminate_session),
        )
        .route(
            "/admin/sessions/cleanup",
            axum::routing::post(session::cleanup_expired_sessions),
        )
        // ==================== 指标仪表盘 ====================
        .route(
            "/admin/metrics/system",
            axum::routing::get(metrics::system_metrics),
        )
        .route(
            "/admin/metrics/inference",
            axum::routing::get(metrics::inference_metrics),
        )
        .route(
            "/admin/metrics/history",
            axum::routing::get(metrics::metrics_history),
        )
        .route(
            "/admin/metrics/dashboard",
            axum::routing::get(metrics::dashboard_overview),
        )
        .route(
            "/admin/metrics/alerts/thresholds",
            axum::routing::get(metrics::alert_thresholds),
        )
        // ==================== API Key 管理 ====================
        .route(
            "/admin/apikeys",
            axum::routing::get(apikey::list_apikeys).post(apikey::create_apikey),
        )
        .route(
            "/admin/apikeys/:id",
            axum::routing::delete(apikey::delete_apikey),
        )
        .route(
            "/admin/apikeys/:id/toggle",
            axum::routing::patch(apikey::toggle_apikey),
        )
        .route(
            "/admin/apikeys/:id/usage",
            axum::routing::get(apikey::get_usage),
        )
        // ==================== 用户管理 ====================
        .route(
            "/admin/users",
            axum::routing::get(user::list_users).post(user::create_user),
        )
        .route(
            "/admin/users/:id",
            axum::routing::get(user::get_user_detail)
                .put(user::update_user)
                .delete(user::delete_user),
        )
        .route(
            "/admin/users/:id/role",
            axum::routing::put(user::update_role),
        )
        .route(
            "/admin/users/:id/status",
            axum::routing::put(user::update_status),
        )
        .route(
            "/admin/users/:id/password",
            axum::routing::put(user::reset_password),
        )
        .route(
            "/admin/users/me/password",
            axum::routing::put(user::update_my_password),
        )
        // ==================== 告警管理 ====================
        .route(
            "/admin/alerts/rules",
            axum::routing::get(alert::list_rules).post(alert::create_rule),
        )
        .route(
            "/admin/alerts/rules/:id",
            axum::routing::put(alert::update_rule).delete(alert::delete_rule),
        )
        .route(
            "/admin/alerts/rules/:id/toggle",
            axum::routing::patch(alert::toggle_rule),
        )
        .route(
            "/admin/alerts/records",
            axum::routing::get(alert::list_records),
        )
        .route(
            "/admin/alerts/records/:id/ack",
            axum::routing::patch(alert::acknowledge_alert),
        )
        .route(
            "/admin/alerts/records/:id/resolve",
            axum::routing::patch(alert::resolve_alert),
        )
        .route(
            "/admin/alerts/records/summary",
            axum::routing::get(alert::get_summary),
        )
        // ==================== 审计日志 ====================
        .route("/admin/audit/logs", axum::routing::get(audit::list_logs))
        .route("/admin/audit/stats", axum::routing::get(audit::get_stats))
        // ==================== 配置管理 ====================
        .route(
            "/admin/config",
            axum::routing::get(config::get_config).put(config::update_config),
        )
        .route(
            "/admin/config/reload",
            axum::routing::post(config::reload_config),
        )
        .route(
            "/admin/config/restart",
            axum::routing::post(config::apply_config_restart),
        )
        .route(
            "/admin/config/history",
            axum::routing::get(config::get_history),
        )
        .route(
            "/admin/config/validate",
            axum::routing::post(config::validate_config),
        )
        .route(
            "/admin/config/export",
            axum::routing::get(config::export_config),
        )
        .layer(axum::middleware::from_fn(auth_mw));

    Router::new()
        .route("/admin/auth/login", axum::routing::post(auth::login))
        .merge(protected)
}
