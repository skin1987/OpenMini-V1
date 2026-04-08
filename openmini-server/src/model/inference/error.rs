//! 推理引擎错误类型
//!
//! 提供细粒度的错误分类，便于调用者区分不同类型的错误。
//!
//! # 错误类型说明
//!
//! | 错误类型 | 含义 | 典型场景 | 可恢复性 |
//! |----------|------|----------|----------|
//! | `ModelFileError` | 模型文件错误 | 文件不存在、格式错误、权限不足 | 致命 |
//! | `WeightLoadError` | 权重加载错误 | 权重解析失败、内存不足 | 致命 |
//! | `MissingWeight` | 权重缺失 | 必需的权重张量不存在 | 可恢复 |
//! | `ConfigError` | 配置错误 | 参数无效、缺少必需配置 | 致命 |
//! | `TokenizationError` | 分词错误 | 文本编码/解码失败 | 可恢复 |
//! | `GenerationError` | 生成错误 | 推理过程失败、采样错误 | 视情况 |
//! | `MultimodalError` | 多模态错误 | 图像/音频处理失败 | 视情况 |
//! | `ImagePreprocessError` | 图像预处理错误 | 尺寸不匹配、格式不支持 | 可恢复 |
//! | `MemoryError` | 内存错误 | 内存分配失败、OOM | 致命 |
//! | `IoError` | IO 错误 | 文件读写失败 | 视情况 |
//! | `Other` | 其他错误 | 兜底错误类型 | 视情况 |

#![allow(dead_code)]
//!
//! # 使用示例
//!
//! ```ignore
//! use openmini_server::model::inference::error::{InferenceError, InferenceResult};
//!
//! fn process_input(text: &str) -> InferenceResult<String> {
//!     if text.is_empty() {
//!         return Err(InferenceError::tokenization("Input text is empty"));
//!     }
//!     Ok(text.to_string())
//! }
//! ```

use std::path::PathBuf;

/// 推理引擎错误类型
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    /// 模型文件错误
    #[error("Model file error: {message} (path: {path:?})")]
    ModelFileError {
        message: String,
        path: PathBuf,
    },

    /// 权重加载错误
    #[error("Weight loading error: {message}")]
    WeightLoadError {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// 权重缺失错误
    #[error("Missing required weight: {name}")]
    MissingWeight {
        name: String,
    },

    /// 配置错误
    #[error("Configuration error: {message}")]
    ConfigError {
        message: String,
    },

    /// 分词错误
    #[error("Tokenization error: {message}")]
    TokenizationError {
        message: String,
    },

    /// 生成错误
    #[error("Generation error: {message}")]
    GenerationError {
        message: String,
    },

    /// 多模态错误
    #[error("Multimodal error: {message}")]
    MultimodalError {
        message: String,
    },

    /// 图像预处理错误
    #[error("Image preprocessing error: {message}")]
    ImagePreprocessError {
        message: String,
    },

    /// 内存错误
    #[error("Memory error: {message}")]
    MemoryError {
        message: String,
    },

    /// IO 错误
    #[error("IO error: {message}")]
    IoError {
        message: String,
        #[source]
        source: std::io::Error,
    },

    /// 其他错误
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

impl InferenceError {
    /// 创建模型文件错误
    pub fn model_file(message: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        Self::ModelFileError {
            message: message.into(),
            path: path.into(),
        }
    }

    /// 创建权重加载错误
    pub fn weight_load(message: impl Into<String>) -> Self {
        Self::WeightLoadError {
            message: message.into(),
            source: None,
        }
    }

    /// 创建权重加载错误（带源错误）
    pub fn weight_load_with_source(
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::WeightLoadError {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    /// 创建缺失权重错误
    pub fn missing_weight(name: impl Into<String>) -> Self {
        Self::MissingWeight {
            name: name.into(),
        }
    }

    /// 创建配置错误
    pub fn config(message: impl Into<String>) -> Self {
        Self::ConfigError {
            message: message.into(),
        }
    }

    /// 创建分词错误
    pub fn tokenization(message: impl Into<String>) -> Self {
        Self::TokenizationError {
            message: message.into(),
        }
    }

    /// 创建生成错误
    pub fn generation(message: impl Into<String>) -> Self {
        Self::GenerationError {
            message: message.into(),
        }
    }

    /// 创建多模态错误
    pub fn multimodal(message: impl Into<String>) -> Self {
        Self::MultimodalError {
            message: message.into(),
        }
    }

    /// 创建图像预处理错误
    pub fn image_preprocess(message: impl Into<String>) -> Self {
        Self::ImagePreprocessError {
            message: message.into(),
        }
    }

    /// 创建内存错误
    pub fn memory(message: impl Into<String>) -> Self {
        Self::MemoryError {
            message: message.into(),
        }
    }

    /// 创建 IO 错误
    pub fn io(message: impl Into<String>) -> Self {
        Self::IoError {
            message: message.into(),
            source: std::io::Error::new(std::io::ErrorKind::Other, "no source error"),
        }
    }

    /// 创建 IO 错误（带源错误）
    pub fn io_with_source(message: impl Into<String>, source: std::io::Error) -> Self {
        Self::IoError {
            message: message.into(),
            source,
        }
    }

    /// 检查是否为可恢复错误
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::TokenizationError { .. }
                | Self::ImagePreprocessError { .. }
                | Self::MissingWeight { .. }
        )
    }

    /// 检查是否为致命错误
    pub fn is_fatal(&self) -> bool {
        matches!(
            self,
            Self::ModelFileError { .. }
                | Self::WeightLoadError { .. }
                | Self::ConfigError { .. }
                | Self::MemoryError { .. }
        )
    }
}

/// 结果类型别名
pub type InferenceResult<T> = std::result::Result<T, InferenceError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = InferenceError::missing_weight("embedding");
        assert!(err.to_string().contains("embedding"));
    }

    #[test]
    fn test_error_classification() {
        let err = InferenceError::tokenization("test");
        assert!(err.is_recoverable());
        assert!(!err.is_fatal());

        let err = InferenceError::model_file("test", "/path/to/model");
        assert!(!err.is_recoverable());
        assert!(err.is_fatal());
    }

    #[test]
    fn test_all_error_variants_display_and_debug() {
        // 测试所有错误变体的 Display 和 Debug 实现
        use std::path::PathBuf;

        let errors: Vec<InferenceError> = vec![
            InferenceError::ModelFileError {
                message: "Model file not found".to_string(),
                path: PathBuf::from("/models/test.gguf"),
            },
            InferenceError::WeightLoadError {
                message: "Failed to load weights".to_string(),
                source: None,
            },
            InferenceError::MissingWeight {
                name: "embedding.weight".to_string(),
            },
            InferenceError::ConfigError {
                message: "Invalid batch size".to_string(),
            },
            InferenceError::TokenizationError {
                message: "Cannot encode empty text".to_string(),
            },
            InferenceError::GenerationError {
                message: "Generation timeout".to_string(),
            },
            InferenceError::MultimodalError {
                message: "Image processing failed".to_string(),
            },
            InferenceError::ImagePreprocessError {
                message: "Invalid image dimensions".to_string(),
            },
            InferenceError::MemoryError {
                message: "Out of memory".to_string(),
            },
            InferenceError::IoError {
                message: "Failed to read file".to_string(),
                source: std::io::Error::new(std::io::ErrorKind::NotFound, "not found"),
            },
            InferenceError::Other(anyhow::anyhow!("Generic error")),
        ];

        for error in errors {
            let display = format!("{}", error);
            let debug = format!("{:?}", error);

            // 验证非空
            assert!(!display.is_empty(), "Display should not be empty");
            assert!(!debug.is_empty(), "Debug should not be empty");

            // 验证 Debug 有足够的内容
            assert!(
                debug.len() > 5,
                "Debug output should have meaningful content, got: {}",
                debug
            );
        }
    }

    #[test]
    fn test_recoverable_errors() {
        // 测试所有可恢复的错误
        let recoverable_errors: Vec<InferenceError> = vec![
            InferenceError::tokenization("test"),
            InferenceError::image_preprocess("test"),
            InferenceError::missing_weight("test"),
        ];

        for err in &recoverable_errors {
            assert!(
                err.is_recoverable(),
                "{:?} should be recoverable",
                err
            );
            assert!(
                !err.is_fatal(),
                "{:?} should not be fatal",
                err
            );
        }
    }

    #[test]
    fn test_fatal_errors() {
        // 测试所有致命错误
        let fatal_errors: Vec<InferenceError> = vec![
            InferenceError::model_file("test", "/path"),
            InferenceError::weight_load("test"),
            InferenceError::config("test"),
            InferenceError::memory("OOM"),
        ];

        for err in &fatal_errors {
            assert!(
                !err.is_recoverable(),
                "{:?} should not be recoverable",
                err
            );
            assert!(
                err.is_fatal(),
                "{:?} should be fatal",
                err
            );
        }
    }

    #[test]
    fn test_factory_methods() {
        // 测试所有工厂方法

        // model_file
        let err = InferenceError::model_file("file missing", "/tmp/model.bin");
        let msg = format!("{}", err);
        assert!(msg.contains("file missing"));
        assert!(msg.contains("/tmp/model.bin"));

        // weight_load (无源)
        let err = InferenceError::weight_load("load failed");
        assert!(format!("{}", err).contains("load failed"));

        // weight_load_with_source (带源)
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "no permission");
        let err = InferenceError::weight_load_with_source("load failed with io", io_err);
        assert!(format!("{}", err).contains("load failed with io"));

        // missing_weight
        let err = InferenceError::missing_weight("layer.0.weight");
        assert!(format!("{}", err).contains("layer.0.weight"));

        // config
        let err = InferenceError::config("batch_size must be positive");
        assert!(format!("{}", err).contains("batch_size must be positive"));

        // tokenization
        let err = InferenceError::tokenization("empty input");
        assert!(format!("{}", err).contains("empty input"));

        // generation
        let err = InferenceError::generation("timeout after 30s");
        assert!(format!("{}", err).contains("timeout after 30s"));

        // multimodal
        let err = InferenceError::multimodal("unsupported image format");
        assert!(format!("{}", err).contains("unsupported image format"));

        // image_preprocess
        let err = InferenceError::image_preprocess("dimensions mismatch");
        assert!(format!("{}", err).contains("dimensions mismatch"));

        // memory
        let err = InferenceError::memory("allocated 4GB, need 8GB");
        assert!(format!("{}", err).contains("allocated 4GB, need 8GB"));

        // io (无源)
        let err = InferenceError::io("read error");
        assert!(format!("{}", err).contains("read error"));

        // io_with_source (带源)
        let io_err2 = std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "eof");
        let err = InferenceError::io_with_source("unexpected eof", io_err2);
        assert!(format!("{}", err).contains("unexpected eof"));
    }

    #[test]
    fn test_error_source_chain() {
        // 测试有 source 的错误的错误链
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = InferenceError::io_with_source("cannot open model", io_err);

        // 验证可以获取 source
        if let InferenceError::IoError { source, .. } = &err {
            assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
            assert_eq!(source.to_string(), "file not found");
        } else {
            panic!("Expected IoError variant");
        }
    }

    #[test]
    fn test_weight_load_error_with_source() {
        // 测试 WeightLoadError 的 source
        let custom_err = std::fmt::Error;
        let err = InferenceError::weight_load_with_source("custom error", custom_err);

        match &err {
            InferenceError::WeightLoadError { source: Some(s), .. } => {
                // source 存在
                let _ = s.to_string();
            }
            _ => panic!("Expected WeightLoadError with Some(source)"),
        }
    }

    #[test]
    fn test_other_error_from_anyhow() {
        // 测试从 anyhow::Error 转换
        let anyhow_err = anyhow::anyhow!("context: {}", "detailed error message");
        let inference_err: InferenceError = anyhow_err.into();

        assert!(format!("{}", inference_err).contains("detailed error message"));
    }

    #[test]
    fn test_model_file_error_contains_path() {
        // 验证 ModelFileError 包含路径信息
        let err = InferenceError::model_file(
            "test error",
            "/very/long/path/to/models/large-model-v2.gguf",
        );

        let msg = format!("{}", err);
        assert!(msg.contains("test error"));
        assert!(msg.contains("/very/long/path/to/models/large-model-v2.gguf"));
    }

    #[test]
    fn test_inference_result_type_alias() {
        // 测试结果类型别名
        let ok_result: InferenceResult<String> = Ok("success".to_string());
        assert!(ok_result.is_ok());

        let err_result: InferenceResult<String> = Err(InferenceError::tokenization("fail"));
        assert!(err_result.is_err());
    }

    #[test]
    fn test_error_send_sync_static() {
        // 验证 Error 满足 Send + Sync + 'static 约束（这是错误类型的常见要求）
        fn assert_send_sync<T: Send + Sync + 'static>() {}

        assert_send_sync::<InferenceError>();
    }

    #[test]
    fn test_multiple_same_type_errors() {
        // 测试同一类型的不同实例
        let err1 = InferenceError::tokenization("error 1");
        let err2 = InferenceError::tokenization("error 2");

        assert_ne!(format!("{}", err1), format!("{}", err2));
        assert!(err1.is_recoverable());
        assert!(err2.is_recoverable());
    }

    #[test]
    fn test_error_debug_format_comprehensive() {
        // 测试 Debug 格式的完整性
        let errors: Vec<InferenceError> = vec![
            InferenceError::ModelFileError {
                message: "test".to_string(),
                path: PathBuf::from("/test"),
            },
            InferenceError::MissingWeight {
                name: "test_weight".to_string(),
            },
        ];

        for err in errors {
            let debug = format!("{:#?}", err); // Pretty debug 格式
            assert!(!debug.is_empty());

            let simple_debug = format!("{:?}", err);
            assert!(!simple_debug.is_empty());
            // 验证 Debug 输出有足够的内容
            assert!(
                simple_debug.len() > 5,
                "Debug output should have meaningful content, got: {}",
                simple_debug
            );
        }
    }

    // ==================== 新增分支覆盖测试 (8个) ====================

    #[test]
    fn test_error_context_chain_multiple_levels() {
        // 覆盖分支: 多层错误链式包装

        // 创建一个多层错误链
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "config.toml");
        let inference_err = InferenceError::io_with_source("failed to load config", io_err);

        // 验证错误信息包含上下文
        let msg = format!("{}", inference_err);
        assert!(msg.contains("failed to load config"));

        // 验证 source() 可以获取底层错误
        if let InferenceError::IoError { source, .. } = &inference_err {
            let source_msg = format!("{}", source);
            assert!(source_msg.contains("config.toml"));
        }

        // 再次包装（模拟上层添加更多上下文）
        let top_level_msg = format!("in initialization: {}", inference_err);
        assert!(top_level_msg.contains("failed to load config"));
        assert!(top_level_msg.contains("in initialization"));
    }

    #[test]
    fn test_error_send_sync_in_multi_threaded_context() {
        // 覆盖分支: 在多线程环境中验证 Send + Sync
        use std::sync::mpsc::channel;

        let err1 = InferenceError::memory("OOM in thread 1");
        let err2 = InferenceError::tokenization("encoding failed in thread 2");

        let (tx, rx) = channel();

        // 在线程间发送错误
        std::thread::spawn(move || {
            tx.send(err1).unwrap();
            tx.send(err2).unwrap();
        });

        let received1 = rx.recv().unwrap();
        let received2 = rx.recv().unwrap();

        match (&received1, &received2) {
            (InferenceError::MemoryError { message: m1 }, InferenceError::TokenizationError { message: m2 }) => {
                assert!(m1.contains("thread 1"));
                assert!(m2.contains("thread 2"));
            }
            _ => panic!("Unexpected error variants"),
        }
    }

    #[test]
    fn test_all_factory_methods_return_correct_variants() {
        // 覆盖分支: 验证每个工厂方法返回正确的变体

        // model_file -> ModelFileError
        let e = InferenceError::model_file("msg", "/path");
        assert!(matches!(e, InferenceError::ModelFileError { .. }));

        // weight_load -> WeightLoadError (无源)
        let e = InferenceError::weight_load("msg");
        assert!(matches!(e, InferenceError::WeightLoadError { source: None, .. }));

        // weight_load_with_source -> WeightLoadError (有源)
        let e = InferenceError::weight_load_with_source("msg", std::fmt::Error);
        assert!(matches!(e, InferenceError::WeightLoadError { source: Some(_), .. }));

        // missing_weight -> MissingWeight
        let e = InferenceError::missing_weight("w");
        assert!(matches!(e, InferenceError::MissingWeight { .. }));

        // config -> ConfigError
        let e = InferenceError::config("c");
        assert!(matches!(e, InferenceError::ConfigError { .. }));

        // tokenization -> TokenizationError
        let e = InferenceError::tokenization("t");
        assert!(matches!(e, InferenceError::TokenizationError { .. }));

        // generation -> GenerationError
        let e = InferenceError::generation("g");
        assert!(matches!(e, InferenceError::GenerationError { .. }));

        // multimodal -> MultimodalError
        let e = InferenceError::multimodal("m");
        assert!(matches!(e, InferenceError::MultimodalError { .. }));

        // image_preprocess -> ImagePreprocessError
        let e = InferenceError::image_preprocess("i");
        assert!(matches!(e, InferenceError::ImagePreprocessError { .. }));

        // memory -> MemoryError
        let e = InferenceError::memory("mem");
        assert!(matches!(e, InferenceError::MemoryError { .. }));

        // io -> IoError (无源)
        let e = InferenceError::io("io");
        assert!(matches!(e, InferenceError::IoError { .. }));
        if let InferenceError::IoError { source, .. } = &e {
            assert_eq!(source.kind(), std::io::ErrorKind::Other);
        }

        // io_with_source -> IoError (有指定源)
        let custom_io = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let e = InferenceError::io_with_source("custom", custom_io);
        assert!(matches!(e, InferenceError::IoError { .. }));
        if let InferenceError::IoError { source, .. } = &e {
            assert_eq!(source.kind(), std::io::ErrorKind::PermissionDenied);
        }
    }

    #[test]
    fn test_error_classification_edge_cases() {
        // 覆盖边界: 边界分类验证

        // 所有可恢复错误的 is_recoverable 应该一致
        let recoverable = vec![
            InferenceError::tokenization("a"),
            InferenceError::image_preprocess("b"),
            InferenceError::missing_weight("c"),
        ];
        for r in recoverable {
            assert!(r.is_recoverable(), "{:?} should be recoverable", r);
            assert!(!r.is_fatal(), "{:?} should not be fatal", r);
        }

        // 所有致命错误的 is_fatal 应该一致
        let fatal = vec![
            InferenceError::model_file("a", "/p"),
            InferenceError::weight_load("b"),
            InferenceError::config("c"),
            InferenceError::memory("d"),
        ];
        for f in fatal {
            assert!(f.is_fatal(), "{:?} should be fatal", f);
            assert!(!f.is_recoverable(), "{:?} should not be recoverable", f);
        }

        // 中间状态错误（既不完全可恢复也不致命）
        let intermediate = vec![
            InferenceError::generation("timeout"),
            InferenceError::multimodal("unsupported format"),
            InferenceError::io("network error"),
        ];
        for i in intermediate {
            // 这些可能根据具体实现被分类为不同类别
            // 这里只验证它们不会 panic
            let _rec = i.is_recoverable();
            let _fat = i.is_fatal();
        }
    }

    #[test]
    fn test_model_file_error_path_preservation() {
        // 覆盖分支: ModelFileError 路径信息的完整保留
        let paths = vec![
            ("/simple/path/model.gguf", "simple"),
            ("/very/deep/nested/path/with/many/segments/model.bin", "deep"),
            ("/path/with spaces/and special.chars/model.gguf", "special"),
            ("/path/unicode/模型文件.gguf", "unicode"),
        ];

        for (path_str, desc) in paths {
            let err = InferenceError::model_file(format!("{} error", desc), path_str);
            let msg = format!("{}", err);

            assert!(msg.contains(desc));
            assert!(msg.contains(path_str));

            // 验证 path 字段完整保留
            if let InferenceError::ModelFileError { path, .. } = err {
                assert_eq!(path.to_string_lossy(), path_str);
            }
        }
    }

    #[test]
    fn test_io_error_source_chain_traversal() {
        // 覆盖分支: IoError 的 source() 链遍历

        // 创建带有特定 kind 的 IO 错误
        let kinds = vec![
            (std::io::ErrorKind::NotFound, "file not found"),
            (std::io::ErrorKind::PermissionDenied, "permission denied"),
            (std::io::ErrorKind::ConnectionRefused, "connection refused"),
            (std::io::ErrorKind::UnexpectedEof, "unexpected eof"),
            (std::io::ErrorKind::OutOfMemory, "out of memory"),
        ];

        for (kind, desc) in kinds {
            let io_err = std::io::Error::new(kind, desc);
            let inf_err = InferenceError::io_with_source(format!("IO: {}", desc), io_err);

            // 遍历 source 链
            if let InferenceError::IoError { source, .. } = &inf_err {
                assert_eq!(source.kind(), kind);

                // 使用 Display 格式化
                let source_msg = format!("{}", source);
                assert!(source_msg.contains(desc));
            }
        }
    }

    #[test]
    fn test_other_error_from_anyhow_comprehensive() {
        // 覆盖分支: 从 anyhow::Error 转换的多种场景
        use anyhow;

        // 简单字符串错误
        let simple = anyhow::anyhow!("simple error");
        let converted: InferenceError = simple.into();
        assert!(format!("{}", converted).contains("simple error"));

        // 带上下文的链式错误
        let chained = anyhow::anyhow!("base error")
            .context("layer 1 context")
            .context("layer 2 context");
        let converted2: InferenceError = chained.into();
        let msg = format!("{}", converted2);
        assert!(msg.contains("layer 2 context"));

        // 从其他标准错误转换（使用 String 错误）
        let std_err_str = "broken pipe".to_string();
        let anyhow_from_std = anyhow::anyhow!("{}", std_err_str);
        let converted3: InferenceError = anyhow_from_std.into();
        assert!(format!("{}", converted3).contains("broken pipe"));
    }
}
