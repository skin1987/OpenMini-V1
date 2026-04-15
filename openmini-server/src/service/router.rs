//! Core Router — Per-Core Actor 负载均衡器
//!
//! 将客户端请求分发到多个 PerCoreActor 实例。
//! 支持 RoundRobin / LeastConnections / ConsistentHash 三种策略。

use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::sync::mpsc;

use crate::service::core_actor::{CoreRequest, CoreResponse};

/// 负载均衡策略
#[derive(Debug, Clone, Copy)]
pub enum BalanceStrategy {
    /// 轮询
    RoundRobin,
    /// 最少连接
    LeastConnections,
    /// 一致性哈希（按 session_id）
    ConsistentHash(u32),
}

/// Per-Core Actor 的发送句柄
#[derive(Clone)]
pub struct ActorHandle {
    pub core_id: usize,
    pub tx: mpsc::Sender<CoreRequest>,
}

impl ActorHandle {
    pub fn new(core_id: usize, tx: mpsc::Sender<CoreRequest>) -> Self {
        Self { core_id, tx }
    }

    /// 获取当前队列深度（近似值）
    fn load(&self) -> usize {
        // mpsc Sender 不直接暴露容量信息，这里返回 0 作为占位
        // 实际生产环境可以通过包装层跟踪活跃请求数
        0
    }
}

/// Core Router — 请求路由分发器
pub struct CoreRouter {
    actors: Vec<ActorHandle>,
    strategy: BalanceStrategy,
    next_index: AtomicUsize,
}

impl CoreRouter {
    /// 创建新的 Router
    pub fn new(actors: Vec<ActorHandle>, strategy: BalanceStrategy) -> Self {
        let count = actors.len();
        assert!(count > 0, "Must have at least one actor");

        Self {
            actors,
            strategy,
            next_index: AtomicUsize::new(0),
        }
    }

    /// 分发请求到目标 Actor
    ///
    /// 返回 oneshot 接收端，用于获取推理结果
    pub async fn dispatch(
        &self,
        prompt: String,
        session_id: String,
    ) -> Result<mpsc::Receiver<CoreResponse>, String> {
        let target = self.select_target(&session_id);
        let actor = &self.actors[target];

        let (response_tx, response_rx) = mpsc::channel::<CoreResponse>(1);

        let request = CoreRequest {
            prompt,
            session_id: session_id.clone(),
            response_tx,
        };

        actor
            .tx
            .send(request)
            .await
            .map_err(|_| format!("Core-{} is closed", actor.core_id))?;

        Ok(response_rx)
    }

    /// 根据策略选择目标 Actor
    fn select_target(&self, session_id: &str) -> usize {
        match self.strategy {
            BalanceStrategy::RoundRobin => self.round_robin_select(),
            BalanceStrategy::LeastConnections => self.least_conn_select(),
            BalanceStrategy::ConsistentHash(virtual_nodes) => {
                self.consistent_hash_select(session_id, virtual_nodes)
            }
        }
    }

    /// 轮询选择
    fn round_robin_select(&self) -> usize {
        self.next_index.fetch_add(1, Ordering::Relaxed) % self.actors.len()
    }

    /// 最少连接选择
    fn least_conn_select(&self) -> usize {
        self.actors
            .iter()
            .enumerate()
            .min_by_key(|(_, a)| a.load())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// 一致性哈希选择
    fn consistent_hash_select(&self, key: &str, _virtual_nodes: u32) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        (hash as usize) % self.actors.len()
    }

    /// 获取 Actor 数量
    pub fn actor_count(&self) -> usize {
        self.actors.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_creation() {
        let (tx0, _) = mpsc::channel(100);
        let (tx1, _) = mpsc::channel(100);
        let actors = vec![ActorHandle::new(0, tx0), ActorHandle::new(1, tx1)];
        let router = CoreRouter::new(actors, BalanceStrategy::RoundRobin);
        assert_eq!(router.actor_count(), 2);
    }

    #[test]
    #[should_panic]
    fn test_router_empty_panics() {
        let router = CoreRouter::new(vec![], BalanceStrategy::RoundRobin);
        let _ = router;
    }

    #[test]
    fn test_round_robin_distribution() {
        let (tx0, _) = mpsc::channel(100);
        let (tx1, _) = mpsc::channel(100);
        let actors = vec![ActorHandle::new(0, tx0), ActorHandle::new(1, tx1)];
        let router = CoreRouter::new(actors.clone(), BalanceStrategy::RoundRobin);

        let mut counts = [0usize; 2];
        for _ in 0..100 {
            let idx = router.round_robin_select();
            counts[idx] += 1;
        }

        assert_eq!(counts[0], 50);
        assert_eq!(counts[1], 50);
    }

    #[test]
    fn test_consistent_hash_deterministic() {
        let (tx0, _) = mpsc::channel(100);
        let (tx1, _) = mpsc::channel(100);
        let actors = vec![ActorHandle::new(0, tx0), ActorHandle::new(1, tx1)];
        let router = CoreRouter::new(actors, BalanceStrategy::ConsistentHash(100));

        let idx1 = router.consistent_hash_select("session-abc", 100);
        let idx2 = router.consistent_hash_select("session-abc", 100);
        let _idx3 = router.consistent_hash_select("session-xyz", 100);

        assert_eq!(idx1, idx2); // 相同 key 总是映射到同一个 target
                                // idx3 可能与 idx1 不同（不强制要求）
    }
}
