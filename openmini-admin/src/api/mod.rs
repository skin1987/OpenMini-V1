pub mod auth;
pub mod service;
pub mod model;
pub mod apikey;
pub mod alert;
pub mod audit;
pub mod config;
pub mod user;

use axum::Router;
use crate::{AppState, auth::middleware::create_auth_middleware};

pub fn create_admin_routes(state: &AppState) -> Router<AppState> {
    let auth_mw = create_auth_middleware(state.config.jwt.secret.clone());

    let protected = Router::new()
        .route("/admin/service/status", axum::routing::get(service::get_status))
        .route("/admin/service/workers", axum::routing::get(service::get_workers))
        .route("/admin/service/restart", axum::routing::post(service::restart_service))
        .route("/admin/service/stop", axum::routing::post(service::stop_service))
        .route("/admin/model", axum::routing::get(model::list_models).post(model::load_model))
        .route("/admin/model/switch", axum::routing::post(model::switch_model))
        .route("/admin/model/:id/unload", axum::routing::post(model::unload_model))
        .route("/admin/model/:id/health", axum::routing::get(model::check_health))
        .route("/admin/apikeys", axum::routing::get(apikey::list_apikeys).post(apikey::create_apikey))
        .route("/admin/apikeys/:id", axum::routing::delete(apikey::delete_apikey))
        .route("/admin/apikeys/:id/toggle", axum::routing::patch(apikey::toggle_apikey))
        .route("/admin/apikeys/:id/usage", axum::routing::get(apikey::get_usage))
        // 用户管理路由
        .route("/admin/users", axum::routing::get(user::list_users).post(user::create_user))
        .route("/admin/users/:id", axum::routing::get(user::get_user_detail).put(user::update_user).delete(user::delete_user))
        .route("/admin/users/:id/role", axum::routing::put(user::update_role))
        .route("/admin/users/:id/status", axum::routing::put(user::update_status))
        .route("/admin/users/:id/password", axum::routing::put(user::reset_password))
        .route("/admin/users/me/password", axum::routing::put(user::update_my_password))
        // 告警管理路由
        .route("/admin/alerts/rules", axum::routing::get(alert::list_rules).post(alert::create_rule))
        .route("/admin/alerts/rules/:id", axum::routing::put(alert::update_rule).delete(alert::delete_rule))
        .route("/admin/alerts/rules/:id/toggle", axum::routing::patch(alert::toggle_rule))
        .route("/admin/alerts/records", axum::routing::get(alert::list_records))
        .route("/admin/alerts/records/:id/ack", axum::routing::patch(alert::acknowledge_alert))
        .route("/admin/alerts/records/:id/resolve", axum::routing::patch(alert::resolve_alert))
        .route("/admin/alerts/records/summary", axum::routing::get(alert::get_summary))
        .route("/admin/audit/logs", axum::routing::get(audit::list_logs))
        .route("/admin/audit/stats", axum::routing::get(audit::get_stats))
        .route("/admin/config", axum::routing::get(config::get_config).put(config::update_config))
        .route("/admin/config/reload", axum::routing::post(config::reload_config))
        .route("/admin/config/history", axum::routing::get(config::get_history))
        .layer(axum::middleware::from_fn(auth_mw));

    Router::new()
        .route("/admin/auth/login", axum::routing::post(auth::login))
        .merge(protected)
}
