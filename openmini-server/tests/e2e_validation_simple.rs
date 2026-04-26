//! 端到端推理验证测试 (简化版)
//!
//! 使用真实GGUF模型验证核心功能：
//! 1. GGUF文件解析
//! 2. 模型配置提取
//! 3. 权重加载验证
//! 4. 推理引擎初始化
//!
//! 运行方式:
//!   OPENMINI_MODEL_PATH=/path/to/model.gguf cargo test --package openmini-server --test e2e_validation_simple -- --nocapture

use std::path::{Path, PathBuf};

/// 获取测试模型路径
fn get_test_model_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("OPENMINI_MODEL_PATH") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return Some(p);
        }
    }

    let search_paths = vec![
        PathBuf::from("./models"),
        PathBuf::from("../models"),
        PathBuf::from("./test-models"),
    ];

    for base in &search_paths {
        if base.is_dir() {
            if let Ok(entries) = std::fs::read_dir(base) {
                for entry in entries.flatten() {
                    if entry.path().extension().map_or(false, |e| e == "gguf") {
                        return Some(entry.path());
                    }
                }
            }
        } else if base.extension().map_or(false, |e| e == "gguf") && base.exists() {
            return Some(base.clone());
        }
    }

    None
}

#[test]
fn test_gguf_file_parsing() {
    let model_path = match get_test_model_path() {
        Some(p) => p,
        None => {
            eprintln!("[SKIP] 未找到GGUF模型文件");
            eprintln!("设置 OPENMINI_MODEL_PATH 环境变量或运行: bash scripts/download_test_model.sh");
            return;
        }
    };

    println!("\n📁 测试模型: {:?}", model_path);

    assert!(model_path.exists(), "模型文件不存在");
    
    let metadata = std::fs::metadata(&model_path).expect("无法读取文件元数据");
    assert!(metadata.len() > 0, "模型文件为空");
    
    println!("   文件大小: {:.2} MB", metadata.len() as f64 / 1024.0 / 1024.0);
}

#[test]
fn test_gguf_header_parsing() {
    let model_path = match get_test_model_path() {
        Some(p) => p,
        None => return,
    };

    use openmini_server::model::inference::gguf::GgufFile;
    
    let start = std::time::Instant::now();
    let gguf_file = GgufFile::open(&model_path)
        .expect("无法打开GGUF文件");
    
    let elapsed = start.elapsed();
    
    println!("\n⏱️ GGUF解析时间: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    
    // 访问公开字段
    let header = &gguf_file.header;
    
    println!("\n📋 模型信息:");
    println!("   版本: {}", header.version);
    println!("   张量数量: {}", header.tensor_count);
    println!("   KV对数量: {}", header.metadata_kv_count);
    
    // 验证元数据
    let metadata = &gguf_file.metadata;
    
    if let Some(name) = metadata.get_string("general.name") {
        println!("   模型名称: {}", name);
    }
    
    if let Some(arch) = metadata.get_string("general.architecture") {
        println!("   架构: {}", arch);
    }

    assert!(header.version > 0, "无效的GGUF版本");
    assert!(header.tensor_count > 0, "无张量信息");
}

#[test]
fn test_tensor_info_loading() {
    let model_path = match get_test_model_path() {
        Some(p) => p,
        None => return,
    };

    use openmini_server::model::inference::gguf::GgufFile;
    
    let gguf_file = GgufFile::open(&model_path)
        .expect("无法打开GGUF文件");

    let tensors = &gguf_file.tensors;
    
    println!("\n📦 张量信息 (前10个):");
    for (i, (name, tensor)) in tensors.iter().take(10).enumerate() {
        println!("   [{}] {} - dims: {:?}, type: {:?}", 
            i + 1, 
            name,
            tensor.dims,
            tensor.tensor_type
        );
    }
    
    if tensors.len() > 10 {
        println!("   ... 共 {} 个张量", tensors.len());
    }

    let has_embedding = tensors.keys().any(|n| n.contains("token_embd") || n.contains("embed"));
    let has_output = tensors.keys().any(|n| n.contains("output") || n.contains("lm_head"));
    
    println!("\n✓ 包含embedding层: {}", has_embedding);
    println!("✓ 包含输出层: {}", has_output);

    assert!(tensors.len() > 0, "应该有至少一个张量");
}

#[test]
fn test_inference_engine_creation() {
    let model_path = match get_test_model_path() {
        Some(p) => p,
        None => return,
    };

    use openmini_server::model::inference::InferenceEngine;

    let start = std::time::Instant::now();
    
    match InferenceEngine::from_gguf(&model_path) {
        Ok(_engine) => {
            let elapsed = start.elapsed();
            
            println!("\n✅ 推理引擎创建成功!");
            println!("   加载时间: {:.2}s", elapsed.as_secs_f32());
        }
        Err(e) => {
            panic!("❌ 推理引擎创建失败: {}", e);
        }
    }
}

#[test]
fn test_tokenization() {
    use openmini_server::model::inference::tokenizer::Tokenizer;

    let tokenizer = Tokenizer::new();
    
    let test_cases = vec![
        ("Hello", 1),
        ("Hello world", 2),
        ("The quick brown fox", 4),
    ];

    println!("\n🔤 Tokenization测试:");

    for (text, expected_min_tokens) in test_cases {
        match tokenizer.encode(text) {
            Ok(tokens) => {
                let count = tokens.len();
                println!("   '{}' -> {} tokens ✓", text, count);
                assert!(count >= expected_min_tokens, 
                    "'{}' 应该至少有 {} tokens, 实际 {}", text, expected_min_tokens, count);
            }
            Err(e) => {
                panic!("Tokenization失败 '{}': {}", text, e);
            }
        }
    }
}

#[test]
fn test_model_config_extraction() {
    let model_path = match get_test_model_path() {
        Some(p) => p,
        None => return,
    };

    use openmini_server::model::inference::gguf::GgufFile;
    use openmini_server::model::inference::InferenceEngine;

    // 先验证GGUF可以解析
    let gguf_file = GgufFile::open(&model_path)
        .expect("无法打开GGUF文件");

    // 从元数据提取配置
    let metadata = &gguf_file.metadata;
    
    println!("\n⚙️ 模型配置 (从GGUF元数据):");
    
    if let Some(arch) = metadata.get_string("general.architecture") {
        println!("   架构: {}", arch);
    }
    
    if let Some(name) = metadata.get_string("general.name") {
        println!("   模型名称: {}", name);
    }
    
    if let Some(layers) = metadata.get_u32("llama.block_count") {
        println!("   层数: {}", layers);
    }
    
    if let Some(hidden) = metadata.get_u32("llama.embedding_length") {
        println!("   隐藏维度: {}", hidden);
    }

    // 然后验证引擎可以创建（验证权重加载）
    let _engine = InferenceEngine::from_gguf(&model_path)
        .expect("引擎创建失败");

    println!("\n✅ 模型配置提取和引擎创建成功!");
}
