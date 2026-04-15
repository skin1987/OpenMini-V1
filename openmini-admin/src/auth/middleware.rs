use axum::{extract::Request, middleware::Next, response::Response};
use std::sync::Arc;

use super::jwt::verify_token;
use crate::error::AppError;

pub fn create_auth_middleware(
    jwt_secret: String,
) -> impl Clone
       + Fn(
    Request,
    Next,
)
    -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Response, AppError>> + Send>> {
    let secret = Arc::new(jwt_secret);
    move |mut req: Request, next: Next| {
        let secret = secret.clone();
        Box::pin(async move {
            let auth_header = req
                .headers()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("");

            let token = if let Some(stripped) = auth_header.strip_prefix("Bearer ") {
                stripped
            } else {
                return Err(AppError::Unauthorized);
            };

            match verify_token(token, &secret) {
                Ok(claims) => {
                    req.extensions_mut().insert(claims);
                    Ok(next.run(req).await)
                }
                Err(_) => Err(AppError::Unauthorized),
            }
        })
    }
}

// ============ 单元测试 ============

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::jwt::create_token;
    use axum::{
        body::Body,
        http::{Request as HttpRequest, StatusCode},
        routing::get,
        Router,
    };
    use tower::util::ServiceExt;

    /// 创建一个简单的测试路由
    fn create_test_app(secret: &str) -> Router {
        Router::new()
            .route("/protected", get(|| async { "protected data" }))
            .layer(axum::middleware::from_fn(create_auth_middleware(
                secret.to_string(),
            )))
    }

    #[tokio::test]
    async fn test_valid_token_passes() {
        let secret = "test_secret_123";
        let app = create_test_app(secret);

        // 创建有效的 JWT token
        let token = create_token(&1i64, "testuser", "admin", secret, 1).unwrap();

        // 构建请求
        let request = HttpRequest::builder()
            .uri("/protected")
            .header("authorization", format!("Bearer {}", token))
            .body(Body::empty())
            .unwrap();

        // 发送请求
        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_invalid_token_rejected() {
        let secret = "test_secret_123";
        let app = create_test_app(secret);

        // 使用无效的 token
        let request = HttpRequest::builder()
            .uri("/protected")
            .header("authorization", "Bearer invalid.token.here")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_expired_token_rejected() {
        let secret = "test_secret_123";
        let app = create_test_app(secret);

        // 手动创建一个已过期的 token（通过设置过去的过期时间）
        use chrono::{Duration, Utc};
        use jsonwebtoken::{encode, EncodingKey, Header};
        use serde::Serialize;

        #[derive(Debug, Serialize, Clone)]
        struct ExpiredClaims {
            pub sub: i64,
            pub username: String,
            pub role: String,
            pub exp: i64,
            pub iat: i64,
        }

        let expired_claims = ExpiredClaims {
            sub: 1,
            username: "testuser".to_string(),
            role: "admin".to_string(),
            exp: (Utc::now() - Duration::hours(1)).timestamp(), // 1小时前过期
            iat: (Utc::now() - Duration::hours(2)).timestamp(),
        };

        let expired_token = encode(
            &Header::default(),
            &expired_claims,
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .unwrap();

        let request = HttpRequest::builder()
            .uri("/protected")
            .header("authorization", format!("Bearer {}", expired_token))
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_missing_authorization_header() {
        let secret = "test_secret_123";
        let app = create_test_app(secret);

        // 不带 Authorization header 的请求
        let request = HttpRequest::builder()
            .uri("/protected")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_wrong_scheme_rejected() {
        let secret = "test_secret_123";
        let app = create_test_app(secret);

        // 使用错误的认证方案（不是 Bearer）
        let request = HttpRequest::builder()
            .uri("/protected")
            .header("authorization", "Basic dXNlcjpwYXNz") // Basic Auth 而不是 Bearer
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_empty_bearer_token() {
        let secret = "test_secret_123";
        let app = create_test_app(secret);

        // 空的 Bearer token
        let request = HttpRequest::builder()
            .uri("/protected")
            .header("authorization", "Bearer ")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_different_secret_rejected() {
        let correct_secret = "correct_secret";
        let wrong_secret = "wrong_secret";
        let app = create_test_app(correct_secret);

        // 用错误的密钥创建 token
        let token = create_token(&1i64, "testuser", "admin", wrong_secret, 1).unwrap();

        let request = HttpRequest::builder()
            .uri("/protected")
            .header("authorization", format!("Bearer {}", token))
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_claims_inserted_into_extensions() {
        use crate::auth::jwt::Claims;
        let secret = "test_secret_123";

        // 创建一个可以检查 claims 的路由
        let app = Router::new()
            .route(
                "/check-claims",
                get(|req: Request| async move {
                    // 尝试从 extensions 中获取 claims
                    match req.extensions().get::<Claims>() {
                        Some(claims) => format!("user_id: {}, role: {}", claims.sub, claims.role),
                        None => "no claims found".to_string(),
                    }
                }),
            )
            .layer(axum::middleware::from_fn(create_auth_middleware(
                secret.to_string(),
            )));

        let token = create_token(&42i64, "admin_user", "admin", secret, 1).unwrap();

        let request = HttpRequest::builder()
            .uri("/check-claims")
            .header("authorization", format!("Bearer {}", token))
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        // 读取响应体
        use axum::body;
        let body = body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8(body.to_vec()).unwrap();

        assert!(body_str.contains("user_id: 42"), "Should contain user ID");
        assert!(body_str.contains("role: admin"), "Should contain role");
    }
}
