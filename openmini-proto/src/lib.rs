//! 协议定义模块
//!
//! 通过 tonic 自动生成 gRPC 协议代码

pub mod proto {
    tonic::include_proto!("openmini");
}
