//! Protobuf 构建脚本
//!
//! 使用 tonic-build 从 .proto 文件生成 Rust 代码。
//! 生成服务端和客户端代码。

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["proto/openmini.proto"], &["proto"])?;
    Ok(())
}
