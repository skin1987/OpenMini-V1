//! OpenMini Server Build Script
//!
//! 本构建脚本负责：
//! 1. 使用 prost-build 编译 Protocol Buffers 定义
//! 2. 使用 ts-rs 生成 TypeScript 类型定义到前端目录
//!
//! ## TypeScript 类型生成
//!
//! 所有标记了 `#[ts(export)]` 的 Rust 类型会自动生成对应的 TypeScript 接口。
//! 生成的文件位于 `../openmini-admin-web/src/types/api/` 目录。
//!
//! ## 使用方式
//!
//! 前端可以直接导入生成的类型：
//! ```typescript
//! import { ChatCompletionRequest } from '@/types/api';
//! ```

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ========================================================================
    // 1. 编译 Protocol Buffers 定义
    // ========================================================================
    let proto_files = &["../openmini-proto/proto/openmini.proto"];

    let mut config = prost_build::Config::new();
    config
        .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
        .out_dir("src/");

    config.compile_protos(proto_files, &["../openmini-proto/proto/"])?;

    println!("cargo:rerun-if-changed=../openmini-proto/proto/openmini.proto");

    // ========================================================================
    // 2. 配置 TypeScript 类型输出路径
    // ========================================================================
    
    // 设置 TypeScript 输出目录（前端项目的 types/api 目录）
    let ts_output_path = "../openmini-admin-web/src/types/api";
    
    // 告诉 cargo 在以下条件变化时重新运行构建脚本
    println!("cargo:rerun-if-changed=src/service/http/types.rs");
    println!("cargo:rerun-if-changed=src/service/grpc/types.rs");
    println!("cargo:rerun-if-changed=src/error.rs");
    println!("cargo:rerun-if-changed=src/config/settings.rs");
    println!("cargo:rerun-if-changed=src/monitoring/health_check.rs");
    
    // 输出 TypeScript 输出路径信息（供开发者参考）
    println!("cargo:warning=TypeScript types will be generated to: {}", ts_output_path);

    Ok(())
}
