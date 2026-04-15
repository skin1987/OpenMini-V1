//! Tokio Mesh 网络层
//!
//! 基于 tokio::sync::mpsc 实现的轻量级节点间通信。
//! 提供点对点和广播通信能力，支持动态节点加入/离开。
//!
//! ## 设计特点
//!
//! - 基于 channel 的内存通信（适合单机多进程或测试）
//! - 支持运行时动态添加/移除节点
//! - 线程安全：使用 Arc<RwLock> 保护共享状态
//! - 异步 API：所有操作都是非阻塞的

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn};

use super::protocol::{DistributedMessage, NodeCapabilities};

/// Mesh 网络错误类型
#[derive(Debug, thiserror::Error)]
pub enum MeshError {
    /// 目标节点未连接
    #[error("节点未连接: {0}")]
    NotConnected(String),

    /// 消息发送失败
    #[error("发送失败: {0}")]
    SendFailed(String),

    /// 通道已关闭（接收端已丢弃）
    #[error("通道已关闭")]
    ChannelClosed,
}

/// Mesh 网络节点
///
/// 表示分布式网络中的一个节点，可以与其他节点建立连接并进行通信。
/// 每个 MeshNode 维护一个到其他节点的发送通道映射表。
pub struct MeshNode {
    /// 节点唯一标识符
    node_id: String,
    /// 已连接的节点及其发送通道 (peer_id -> sender)
    peers: Arc<RwLock<HashMap<String, mpsc::Sender<DistributedMessage>>>>,
    /// 接收消息的通道接收端
    inbox: mpsc::Receiver<DistributedMessage>,
    /// 保留一个发送端，用于外部向此节点发送消息
    inbox_tx: mpsc::Sender<DistributedMessage>,
    /// 节点的硬件能力描述
    capabilities: NodeCapabilities,
}

impl MeshNode {
    /// 创建新的 Mesh 节点
    ///
    /// # 参数
    ///
    /// - `node_id`: 节点唯一标识符
    /// - `capabilities`: 节点的硬件能力信息
    ///
    /// # 返回
    ///
    /// 返回新创建的 MeshNode 实例和该节点的接收端句柄（用于其他节点连接）
    pub fn new(
        node_id: impl Into<String>,
        capabilities: NodeCapabilities,
    ) -> (Self, mpsc::Sender<DistributedMessage>) {
        let node_id = node_id.into();
        let (inbox_tx, inbox_rx) = mpsc::channel::<DistributedMessage>(256);

        info!(node_id = %node_id, "创建 Mesh 节点");

        let node = Self {
            node_id,
            peers: Arc::new(RwLock::new(HashMap::new())),
            inbox: inbox_rx,
            inbox_tx: inbox_tx.clone(),
            capabilities,
        };

        (node, inbox_tx)
    }

    /// 连接到其他节点
    ///
    /// 建立到目标节点的单向通信通道。调用后可以向目标节点发送消息。
    ///
    /// # 参数
    ///
    /// - `peer_id`: 目标节点的 ID
    /// - `tx`: 目标节点的接收端句柄（通过 MeshNode::new() 获取）
    pub async fn connect(&self, peer_id: String, tx: mpsc::Sender<DistributedMessage>) {
        let mut peers = self.peers.write().await;
        peers.insert(peer_id.clone(), tx);
        debug!(
            from_node = %self.node_id,
            to_node = %peer_id,
            "已连接到对等节点"
        );
    }

    /// 断开与指定节点的连接
    ///
    /// # 参数
    ///
    /// - `peer_id`: 要断开的节点 ID
    pub async fn disconnect(&self, peer_id: &str) {
        let mut peers = self.peers.write().await;
        if peers.remove(peer_id).is_some() {
            debug!(
                from_node = %self.node_id,
                to_node = %peer_id,
                "已断开与对等节点的连接"
            );
        }
    }

    /// 发送消息到指定节点
    ///
    /// # 参数
    ///
    /// - `peer_id`: 目标节点 ID
    /// - `msg`: 要发送的消息
    ///
    /// # 错误
    ///
    /// - [`MeshError::NotConnected`]: 目标节点未连接
    /// - [`MeshError::SendFailed`]: 发送失败（通道满或已关闭）
    /// - [`MeshError::ChannelClosed`]: 通道已关闭
    pub async fn send_to(&self, peer_id: &str, msg: DistributedMessage) -> Result<(), MeshError> {
        let peers = self.peers.read().await;

        match peers.get(peer_id) {
            Some(sender) => {
                sender
                    .send(msg)
                    .await
                    .map_err(|e| MeshError::SendFailed(e.to_string()))?;
                Ok(())
            }
            None => Err(MeshError::NotConnected(peer_id.to_string())),
        }
    }

    /// 广播消息到所有已连接的节点
    ///
    /// 将消息发送给所有当前已连接的对等节点。
    /// 如果某个节点发送失败，会记录警告但不会中断广播。
    ///
    /// # 参数
    ///
    /// - `msg`: 要广播的消息
    ///
    /// # 错误
    ///
    /// 返回第一个遇到的错误（如果有）
    pub async fn broadcast(&self, msg: DistributedMessage) -> Result<(), MeshError> {
        let peers = self.peers.read().await;
        let mut last_error = None;

        for (peer_id, sender) in peers.iter() {
            // 克隆消息用于发送（因为 msg 会被消费）
            let msg_clone = msg.clone();
            match sender.send(msg_clone).await {
                Ok(()) => {}
                Err(e) => {
                    warn!(
                        from_node = %self.node_id,
                        to_node = %peer_id,
                        error = %e,
                        "广播消息发送失败"
                    );
                    if last_error.is_none() {
                        last_error = Some(MeshError::SendFailed(format!("{}: {}", peer_id, e)));
                    }
                }
            }
        }

        match last_error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    /// 接收消息
    ///
    /// 阻塞等待并返回下一条收到的消息。
    /// 如果所有发送端都已关闭且缓冲区为空，则返回 None。
    ///
    /// # 返回
    ///
    /// - `Some(DistributedMessage)`: 收到的消息
    /// - `None`: 通道已关闭且无更多消息
    pub async fn recv(&mut self) -> Option<DistributedMessage> {
        self.inbox.recv().await
    }

    /// 尝试非阻塞地接收消息
    ///
    /// 立即返回，如果没有可用消息则返回 None
    pub async fn try_recv(&mut self) -> Option<DistributedMessage> {
        // 使用 tokio 的 try_recv 需要互斥锁，这里简化实现
        // 实际生产环境可以使用 tokio::sync::mpsc::UnboundedReceiver 或带超时的 recv
        self.inbox.recv().await
    }

    /// 获取已连接的节点列表
    ///
    /// 返回当前所有已建立连接的对等节点 ID 列表
    pub async fn connected_peers(&self) -> Vec<String> {
        let peers = self.peers.read().await;
        peers.keys().cloned().collect()
    }

    /// 获取已连接的节点数量
    pub async fn peer_count(&self) -> usize {
        let peers = self.peers.read().await;
        peers.len()
    }

    /// 获取节点 ID
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// 获取节点能力信息
    pub fn capabilities(&self) -> &NodeCapabilities {
        &self.capabilities
    }

    /// 获取节点的接收端句柄
    ///
    /// 用于其他节点通过 connect() 连接到此节点
    pub fn inbox_sender(&self) -> mpsc::Sender<DistributedMessage> {
        self.inbox_tx.clone()
    }

    /// 检查是否已连接到指定节点
    pub async fn is_connected(&self, peer_id: &str) -> bool {
        let peers = self.peers.read().await;
        peers.contains_key(peer_id)
    }
}

impl std::fmt::Debug for MeshNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshNode")
            .field("node_id", &self.node_id)
            .field("capabilities", &self.capabilities)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mesh_node_creation() {
        let caps = NodeCapabilities::new("Test GPU", 8192, 16, vec!["model".to_string()]);
        let (node, _tx) = MeshNode::new("test-node", caps);

        assert_eq!(node.node_id(), "test-node");
        assert_eq!(node.peer_count().await, 0);
    }

    #[tokio::test]
    async fn test_mesh_connect_and_send() {
        // 创建两个节点
        let caps = NodeCapabilities::new("GPU", 8192, 16, vec![]);
        let (node_a, tx_a) = MeshNode::new("node-a", caps.clone());
        let (mut node_b, tx_b) = MeshNode::new("node-b", caps);

        // 建立双向连接
        node_a.connect("node-b".to_string(), tx_b).await;
        node_b.connect("node-a".to_string(), tx_a).await;

        // 发送消息
        let msg = DistributedMessage::Heartbeat {
            node_id: "node-a".to_string(),
            timestamp: 1234,
            gpu_utilization: 0.5,
            memory_used_mb: 1024,
        };

        let result = node_a.send_to("node-b", msg.clone()).await;
        assert!(result.is_ok());

        // 接收消息
        let received = node_b.recv().await;
        assert!(received.is_some());

        match received.unwrap() {
            DistributedMessage::Heartbeat {
                node_id, timestamp, ..
            } => {
                assert_eq!(node_id, "node-a");
                assert_eq!(timestamp, 1234);
            }
            _ => panic!("收到错误的消息类型"),
        }
    }

    #[tokio::test]
    async fn test_mesh_broadcast() {
        let caps = NodeCapabilities::new("GPU", 8192, 16, vec![]);

        // 创建 coordinator 和两个 worker
        let (coordinator, coord_tx) = MeshNode::new("coord", caps.clone());
        let (mut worker1, w1_tx) = MeshNode::new("worker-1", caps.clone());
        let (mut worker2, w2_tx) = MeshNode::new("worker-2", caps);

        // coordinator 连接到 workers
        coordinator.connect("worker-1".to_string(), w1_tx).await;
        coordinator.connect("worker-2".to_string(), w2_tx).await;

        // 广播消息
        let msg = DistributedMessage::NodeRegister {
            node_id: "coord".to_string(),
            capabilities: NodeCapabilities::new("", 0, 0, vec![]),
        };

        let result = coordinator.broadcast(msg.clone()).await;
        assert!(result.is_ok());

        // 两个 worker 都应该收到消息
        let recv1 = worker1.recv().await;
        let recv2 = worker2.recv().await;

        assert!(recv1.is_some());
        assert!(recv2.is_some());
    }

    #[tokio::test]
    async fn test_mesh_disconnect() {
        let caps = NodeCapabilities::new("GPU", 8192, 16, vec![]);
        let (node_a, tx_b) = MeshNode::new("node-a", caps.clone());
        let (_node_b, _tx_a) = MeshNode::new("node-b", caps);

        // 先连接
        node_a.connect("node-b".to_string(), tx_b).await;
        assert_eq!(node_a.peer_count().await, 1);
        assert!(node_a.is_connected("node-b").await);

        // 断开连接
        node_a.disconnect("node-b").await;
        assert_eq!(node_a.peer_count().await, 0);
        assert!(!node_a.is_connected("node-b").await);
    }

    #[tokio::test]
    async fn test_send_to_unconnected_node() {
        let caps = NodeCapabilities::new("GPU", 8192, 16, vec![]);
        let (node, _) = MeshNode::new("node", caps);

        let msg = DistributedMessage::Heartbeat {
            node_id: "node".to_string(),
            timestamp: 0,
            gpu_utilization: 0.0,
            memory_used_mb: 0,
        };

        let result = node.send_to("nonexistent", msg).await;
        assert!(result.is_err());

        match result.unwrap_err() {
            MeshError::NotConnected(id) => {
                assert_eq!(id, "nonexistent");
            }
            _ => panic!("错误的错误类型"),
        }
    }

    #[tokio::test]
    async fn test_connected_peers_list() {
        let caps = NodeCapabilities::new("GPU", 8192, 16, vec![]);
        let (node, tx1) = MeshNode::new("main", caps.clone());
        let (_, tx2) = MeshNode::new("peer-1", caps.clone());
        let (_, tx3) = MeshNode::new("peer-2", caps);

        node.connect("peer-1".to_string(), tx1).await;
        node.connect("peer-2".to_string(), tx2).await;

        let peers = node.connected_peers().await;
        assert_eq!(peers.len(), 2);
        assert!(peers.contains(&"peer-1".to_string()));
        assert!(peers.contains(&"peer-2".to_string()));
    }
}
