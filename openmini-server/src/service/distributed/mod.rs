//! 分布式推理模块
//!
//! 提供基于 Tokio Mesh 的多节点推理能力：
//! - 节点发现与注册
//! - 任务分发与收集
//! - 结果聚合
//!
//! ## 架构概述
//!
//! ```
//! ┌─────────────┐
//! │ Coordinator │ (协调器节点)
//! └──────┬──────┘
//!    mesh RPC
//! ┌─────┴─────┬──────────┐
//! ▼           ▼          ▼
//! ┌──────┐  ┌──────┐  ┌──────┐
//! │Worker│  │Worker│  │Worker│
//! │Node 1│  │Node 2│  │Node 3│
//! └──────┘  └──────┘  └──────┘
//! ```

pub mod mesh;
pub mod coordinator;
pub mod worker;
pub mod protocol;

pub use mesh::MeshNode;
pub use coordinator::Coordinator;
pub use worker::WorkerNode;
pub use protocol::*;
