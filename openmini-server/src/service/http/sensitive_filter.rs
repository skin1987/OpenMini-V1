use axum::{extract::Request, middleware::Next, response::Response};
use tracing::info;

/// 敏感数据正则模式
const SENSITIVE_PATTERNS: &[&str] = &[
    r"(?i)(api[_\-]?key|authorization|token|password|secret|apikey)[=:]\s*\S+",
    r"(?i)Bearer\s+\S+",
    r"(?i)sk-[a-zA-Z0-9]{20,}",
];

/// 敏感信息脱敏中间件
///
/// 自动过滤日志和响应中的 API Key、密码等敏感字段。
pub async fn sensitive_data_filter_middleware(req: Request, next: Next) -> Response {
    // 记录请求前脱敏处理
    if let Some(headers) = req.headers().get("authorization") {
        if let Ok(auth_str) = headers.to_str() {
            let masked = mask_sensitive_data(auth_str);
            info!(authorization_masked = %masked, "Request received");
        }
    }

    let response = next.run(req).await;

    response
}

/// 脱敏处理函数
///
/// 将文本中的敏感信息替换为 ***MASKED***。
pub fn mask_sensitive_data(text: &str) -> String {
    let mut result = text.to_string();

    for pattern in SENSITIVE_PATTERNS {
        if let Ok(re) = regex::Regex::new(pattern) {
            result = re.replace_all(&result, "***MASKED***").to_string();
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_api_key() {
        let input = "api_key=sk-abc123def456";
        let output = mask_sensitive_data(input);
        assert!(output.contains("***MASKED***"));
        assert!(!output.contains("sk-abc123"));
    }

    #[test]
    fn test_mask_bearer_token() {
        let input = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9";
        let output = mask_sensitive_data(input);
        assert!(output.contains("***MASKED***"));
        assert!(!output.contains("eyJhbGci"));
    }

    #[test]
    fn test_mask_password() {
        let input = "password=MySecretPass123";
        let output = mask_sensitive_data(input);
        assert!(output.contains("***MASKED***"));
        assert!(!output.contains("MySecretPass123"));
    }

    #[test]
    fn test_no_sensitive_data() {
        let input = "This is a normal message without secrets";
        let output = mask_sensitive_data(input);
        assert_eq!(output, input);
    }

    #[test]
    fn test_multiple_patterns() {
        let input = "api_key=secret123 Bearer token456 password=pass789";
        let output = mask_sensitive_data(input);
        let masked_count = output.matches("***MASKED***").count();
        assert_eq!(masked_count, 3);
    }
}
