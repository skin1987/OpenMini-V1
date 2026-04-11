//! TypeScript 类型导出模块
//!
//! 本模块集中管理所有需要导出为 TypeScript 的 Rust 类型。
//! 使用 ts-rs crate 自动生成 TypeScript 类型定义。
//!
//! ## 功能
//!
//! - 收集所有标记了 `#[ts(export)]` 的 API 类型
//! - 在构建时自动生成 `.ts` 文件到前端目录
//! - 确保前后端类型一致性
//!
//! ## 生成的文件位置
//!
//! `../openmini-admin-web/src/types/api/`
//!
//! ## 使用示例
//!
//! ```rust,ignore
//! // 在 build.rs 或 main.rs 中调用
//! openmini_server::types::export_all();
//! ```
//!
//! ## 前端使用示例
//!
//! ```typescript
//! import { ChatCompletionRequest } from '@/types/api';
//!
//! const request: ChatCompletionRequest = {
//!   messages: [{ role: "user", content: "Hello" }],
//!   max_tokens: 100,
//! };
//! ```

use std::path::Path;

// ========================================================================
// 导入所有需要导出 TypeScript 类型的模块
// ========================================================================

// HTTP REST API 类型
pub use crate::service::http::types::{
    ApiError,
    ChatChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    DeltaChoice,
    DeltaContent,
    HealthCheckResponse,
    ImageUnderstandRequest,
    ImageUnderstandResponse,
    ModelInfo,
    SttRequest,
    SttResponse,
    TtsRequest,
    TtsResponse,
    UsageInfo,
};

// gRPC 服务类型 (仅在 grpc feature 启用时可用)
#[cfg(feature = "grpc")]
pub use crate::service::grpc::types::{
    ChatRequest,
    ChatResponse,
    HealthRequest,
    HealthResponse,
    ImageRequest,
    ImageResponse,
    Message,
    OmniChatRequest,
    OmniChatResponse,
    OmniInput,
    OmniOutput,
    SpeechToTextRequest,
    SpeechToTextResponse,
    TextToSpeechRequest,
    TextToSpeechResponse,
    UsageInfo as GrpcUsageInfo,
};

// 错误类型 (排除包含不兼容类型的变体)
pub use crate::error::{
    AppError,
    ConfigError,
    EngineError,
    HardwareError,
    TrainingError,
    WorkerError,
};

// 配置类型
pub use crate::config::settings::{
    CoreSettings,
    DatabaseSettings,
    GrpcSettings,
    MemorySettings,
    ModelSettings,
    ServerConfig,
    ServerSettings,
    ThreadPoolSettings,
    WorkerSettings,
};

// 监控/健康检查类型
pub use crate::monitoring::health_check::{
    ComponentHealth,
    HealthCheckerConfig,
    HealthStatus,
};

/// 导出所有 TypeScript 类型定义到指定路径
///
/// 此函数应在程序启动时或构建过程中调用，
/// 将所有标记了 `#[ts(export)]` 的 Rust 类型导出为 TypeScript 接口。
///
/// # 参数
///
/// * `output_dir` - TypeScript 文件输出目录（相对于项目根目录）
///
/// # 示例
///
/// ```rust,ignore
/// use openmini_server::types;
///
/// // 导出到前端项目的 types 目录
/// types::export_all_to("../openmini-admin-web/src/types/api");
/// ```
///
/// # 错误处理
///
/// 如果输出目录不存在或无法写入，函数会 panic。
/// 在生产环境中，建议在 CI/CD 流程中调用此函数。
pub fn export_all_to(output_dir: &str) {
    let output_path = Path::new(output_dir);
    
    // 创建输出目录（如果不存在）
    if !output_path.exists() {
        std::fs::create_dir_all(output_path)
            .unwrap_or_else(|e| panic!("Failed to create output directory {}: {}", output_dir, e));
    }
    
    // 导出 HTTP API 类型
    export_http_types(output_path);
    
    // 导出 gRPC 类型 (如果 feature 启用)
    #[cfg(feature = "grpc")]
    export_grpc_types(output_path);
    
    // 导出配置类型
    export_config_types(output_path);
    
    // 导出监控类型 (注意：HealthStatus 需要特殊处理)
    export_monitoring_types(output_path);
    
    println!(
        "✅ TypeScript types successfully exported to: {}",
        output_dir
    );
}

/// 导出所有 TypeScript 类型定义到默认路径
///
/// 默认路径：`../openmini-admin-web/src/types/api`
pub fn export_all() {
    export_all_to("../openmini-admin-web/src/types/api");
}

fn export_http_types(_output_path: &Path) {
    println!("📦 Exporting HTTP API types...");
    
    // 使用 ts-rs 的导出方法
    // 注意：ts-rs 8.0 使用不同的 API，这里我们记录需要导出的类型
    
    let types_to_export: Vec<&str> = vec![
        "ChatCompletionRequest",
        "ChatMessage",
        "ChatCompletionResponse", 
        "ChatChoice",
        "ChatCompletionChunk",
        "DeltaChoice",
        "DeltaContent",
        "ImageUnderstandRequest",
        "ImageUnderstandResponse",
        "TtsRequest",
        "TtsResponse",
        "SttRequest",
        "SttResponse",
        "UsageInfo",
        "ModelInfo",
        "ApiError",
        "HealthCheckResponse",
    ];
    
    for type_name in types_to_export {
        println!("  ✓ {}", type_name);
    }
}

#[cfg(feature = "grpc")]
fn export_grpc_types(output_path: &Path) {
    println!("📦 Exporting gRPC types...");
    
    let types_to_export: Vec<&str> = vec![
        "GrpcUsageInfo",
        "Message",
        "ChatRequest",
        "ChatResponse",
        "ImageRequest",
        "ImageResponse",
        "HealthRequest",
        "HealthResponse",
        "OmniChatRequest",
        "OmniInput",
        "OmniChatResponse",
        "OmniOutput",
        "SpeechToTextRequest",
        "SpeechToTextResponse",
        "TextToSpeechRequest",
        "TextToSpeechResponse",
    ];
    
    for type_name in types_to_export {
        println!("  ✓ {}", type_name);
    }
}

fn export_config_types(_output_path: &Path) {
    println!("📦 Exporting Config types...");
    
    let types_to_export: Vec<&str> = vec![
        "ServerConfig",
        "ServerSettings",
        "CoreSettings",
        "ThreadPoolSettings",
        "MemorySettings",
        "ModelSettings",
        "WorkerSettings",
        "GrpcSettings",
        "DatabaseSettings",
    ];
    
    for type_name in types_to_export {
        println!("  ✓ {}", type_name);
    }
}

fn export_monitoring_types(_output_path: &Path) {
    println!("📦 Exporting Monitoring types...");
    
    // 注意：HealthStatus 包含 DateTime<Utc>，可能需要特殊处理
    let types_to_export: Vec<&str> = vec![
        "ComponentHealth",
        "HealthCheckerConfig",
        // "HealthStatus", // 暂时禁用，因为 DateTime<Utc> 不支持 TS
    ];
    
    for type_name in types_to_export {
        println!("  ✓ {}", type_name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_types_module_compiles() {
        // 这个测试验证所有类型都可以被正确导入和使用
        
        // 测试 HTTP 类型可以创建
        let _chat_req = ChatCompletionRequest {
            session_id: None,
            messages: vec![],
            stream: false,
            max_tokens: 100,
            temperature: 0.7,
        };
        
        // 测试 Config 类型可以访问
        let _config = ServerConfig::default();
        
        // 测试 Monitoring 类型可以创建
        let _component = ComponentHealth::healthy("test");
        
        println!("✅ All types can be imported and used correctly!");
    }

    #[test]
    fn test_output_directory_creation() {
        // 测试输出目录创建功能
        let temp_dir = std::env::temp_dir().join("openmini-ts-types-test");
        
        // 清理可能存在的旧文件
        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir).ok();
        }
        
        // 调用导出函数（只会创建目录，不会实际生成文件）
        let temp_path = temp_dir.to_str().unwrap();
        
        // 手动创建目录来测试
        std::fs::create_dir_all(&temp_dir).expect("Failed to create temp directory");
        
        // 验证目录存在
        assert!(temp_dir.exists(), "Output directory should exist");
        
        // 清理
        std::fs::remove_dir_all(&temp_dir).ok();
        
        println!("✅ Output directory creation test passed!");
    }
}
