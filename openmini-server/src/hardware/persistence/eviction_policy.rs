//! 统一淘汰策略模块
//!
//! 为所有持久化子系统提供一致的淘汰算法。
//!
//! # 支持的算法
//!
//! - **LRU**: 最近最少使用 (适合时间局部性强的场景)
//! - **LFU**: 最少经常使用 (适合热点数据稳定的场景)
//! - **FIFO**: 先进先出 (简单高效)
//! - **Adaptive**: 自适应 (根据访问模式自动选择)
//! - **ImportanceWeighted**: 重要性加权 (综合考虑多个因素)

use std::cmp::Ordering;

/// 淘汰算法类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EvictionAlgorithm {
    #[default]
    Lru,
    Lfu,
    Fifo,
    Adaptive,
    ImportanceWeighted,
}

impl std::fmt::Display for EvictionAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvictionAlgorithm::Lru => write!(f, "LRU"),
            EvictionAlgorithm::Lfu => write!(f, "LFU"),
            EvictionAlgorithm::Fifo => write!(f, "FIFO"),
            EvictionAlgorithm::Adaptive => write!(f, "Adaptive"),
            EvictionAlgorithm::ImportanceWeighted => write!(f, "ImportanceWeighted"),
        }
    }
}

/// 淘汰候选条目 trait
///
/// 所有需要参与淘汰的数据结构都需要实现此 trait
pub trait EvictionCandidate: Send + Sync {
    /// 唯一标识符
    fn id(&self) -> &str;

    /// 最后访问时间戳 (Unix 秒)
    fn last_access(&self) -> u64;

    /// 总访问次数
    fn access_count(&self) -> u64;

    /// 占用的内存大小 (字节)
    fn size(&self) -> usize;

    /// 重要性分数 (0.0-1.0, 越高越不应该被淘汰)
    fn importance(&self) -> f32;

    /// 创建时间戳 (Unix 秒)
    fn created_at(&self) -> u64;
}

/// 淘汰策略配置
#[derive(Debug, Clone)]
pub struct EvictionPolicyConfig {
    /// 使用的算法
    pub algorithm: EvictionAlgorithm,

    /// 最大容量 (0 表示无限制)
    pub max_capacity: usize,

    /// 高水位线 (达到此值开始淘汰)
    pub high_watermark_pct: f32,

    /// 低水位线 (降到此值停止淘汰)
    pub low_watermark_pct: f32,

    /// 重要性权重 (用于 ImportanceWeighted 算法)
    pub importance_weight: f32,

    /// 年龄权重 (用于 ImportanceWeighted 算法)
    pub age_weight: f32,

    /// 大小权重 (用于 ImportanceWeighted 算法)
    pub size_weight: f32,
}

impl Default for EvictionPolicyConfig {
    fn default() -> Self {
        Self {
            algorithm: EvictionAlgorithm::Lru,
            max_capacity: usize::MAX,
            high_watermark_pct: 0.8,
            low_watermark_pct: 0.6,
            importance_weight: 0.4,
            age_weight: 0.3,
            size_weight: 0.3,
        }
    }
}

/// 淘汰策略执行器
pub struct EvictionPolicy {
    config: EvictionPolicyConfig,
    current_usage: std::sync::atomic::AtomicUsize,
    stats: parking_lot::RwLock<EvictionStats>,
}

/// 淘汰统计信息
#[derive(Debug, Default, Clone)]
pub struct EvictionStats {
    /// 总淘汰次数
    pub total_evictions: u64,
    /// LRU 算法使用次数
    pub lru_evictions: u64,
    /// LFU 算法使用次数
    pub lfu_evictions: u64,
    /// FIFO 算法使用次数
    pub fifo_evictions: u64,
    /// Adaptive 算法使用次数
    pub adaptive_evictions: u64,
    /// ImportanceWeighted 算法使用次数
    pub importance_evictions: u64,
    /// 总释放内存量 (字节)
    pub total_freed_bytes: u64,
}

impl EvictionPolicy {
    /// 创建新的淘汰策略
    pub fn new(config: EvictionPolicyConfig) -> Self {
        Self {
            config,
            current_usage: std::sync::atomic::AtomicUsize::new(0),
            stats: parking_lot::RwLock::new(EvictionStats::default()),
        }
    }

    /// 获取当前使用的算法
    pub fn algorithm(&self) -> EvictionAlgorithm {
        self.config.algorithm
    }

    /// 动态切换算法
    pub fn set_algorithm(&mut self, algorithm: EvictionAlgorithm) {
        self.config.algorithm = algorithm;
    }

    /// 更新当前使用量
    pub fn update_usage(&self, usage: usize) {
        self.current_usage
            .store(usage, std::sync::atomic::Ordering::Relaxed);
    }

    /// 获取当前使用量
    pub fn current_usage(&self) -> usize {
        self.current_usage
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// 检查是否需要淘汰
    pub fn should_evict(&self) -> bool {
        let usage = self.current_usage();
        if self.config.max_capacity == 0 {
            return false;
        }

        let threshold = (self.config.max_capacity as f32 * self.config.high_watermark_pct) as usize;
        usage > threshold
    }

    /// 选择应该淘汰的候选者
    ///
    /// # 参数
    /// - candidates: 候选列表
    /// - count: 需要选择的数量
    ///
    /// # 返回
    /// 应该被淘汰的候选者列表（按优先级排序）
    pub fn select_victims<'a, T: EvictionCandidate>(
        &self,
        candidates: &'a [T],
        count: usize,
    ) -> Vec<&'a T> {
        if candidates.is_empty() || count == 0 {
            return vec![];
        }

        let victims = match self.config.algorithm {
            EvictionAlgorithm::Lru => self.select_lru(candidates, count),
            EvictionAlgorithm::Lfu => self.select_lfu(candidates, count),
            EvictionAlgorithm::Fifo => self.select_fifo(candidates, count),
            EvictionAlgorithm::Adaptive => self.select_adaptive(candidates, count),
            EvictionAlgorithm::ImportanceWeighted => self.select_importance(candidates, count),
        };

        // 更新统计
        let mut stats = self.stats.write();
        stats.total_evictions += 1;
        match self.config.algorithm {
            EvictionAlgorithm::Lru => stats.lru_evictions += 1,
            EvictionAlgorithm::Lfu => stats.lfu_evictions += 1,
            EvictionAlgorithm::Fifo => stats.fifo_evictions += 1,
            EvictionAlgorithm::Adaptive => stats.adaptive_evictions += 1,
            EvictionAlgorithm::ImportanceWeighted => stats.importance_evictions += 1,
        }
        stats.total_freed_bytes += victims.iter().map(|v| v.size()).sum::<usize>() as u64;

        victims
    }

    // === 具体算法实现 ===

    /// LRU: 选择最长时间未访问的
    fn select_lru<'a, T: EvictionCandidate>(
        &self,
        candidates: &'a [T],
        count: usize,
    ) -> Vec<&'a T> {
        let mut sorted: Vec<&T> = candidates.iter().collect();
        sorted.sort_by_key(|a| a.last_access());
        sorted.into_iter().take(count).collect()
    }

    /// LFU: 选择访问次数最少的
    fn select_lfu<'a, T: EvictionCandidate>(
        &self,
        candidates: &'a [T],
        count: usize,
    ) -> Vec<&'a T> {
        let mut sorted: Vec<&T> = candidates.iter().collect();
        sorted.sort_by_key(|a| a.access_count());
        sorted.into_iter().take(count).collect()
    }

    /// FIFO: 选择最早创建的
    fn select_fifo<'a, T: EvictionCandidate>(
        &self,
        candidates: &'a [T],
        count: usize,
    ) -> Vec<&'a T> {
        let mut sorted: Vec<&T> = candidates.iter().collect();
        sorted.sort_by_key(|a| a.created_at());
        sorted.into_iter().take(count).collect()
    }

    /// Adaptive: 根据访问模式自适应选择
    fn select_adaptive<'a, T: EvictionCandidate>(
        &self,
        candidates: &'a [T],
        count: usize,
    ) -> Vec<&'a T> {
        // 分析访问模式
        let now = current_timestamp();
        let mut scored: Vec<(f64, &T)> = candidates
            .iter()
            .map(|c| {
                let age = (now - c.last_access()) as f64;
                let freq = c.access_count() as f64;

                // 访问频率低且年龄大的优先淘汰
                let score = age / (freq.ln() + 1e-6);
                (score, c)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        scored.into_iter().take(count).map(|(_, c)| c).collect()
    }

    /// ImportanceWeighted: 综合考虑重要性、年龄、大小
    fn select_importance<'a, T: EvictionCandidate>(
        &self,
        candidates: &'a [T],
        count: usize,
    ) -> Vec<&'a T> {
        let now = current_timestamp();

        let mut scored: Vec<(f64, &T)> = candidates
            .iter()
            .map(|c| {
                let age = (now - c.last_access()) as f64; // 年龄越大越好淘汰
                let inv_importance = 1.0 / (c.importance() as f64 + 1e-6); // 重要性低的好淘汰
                let size_factor = c.size() as f64 / (1024.0 * 1024.0); // 小的好淘汰

                let score = age * self.config.age_weight as f64
                    + inv_importance * self.config.importance_weight as f64
                    + size_factor * self.config.size_weight as f64;
                (score, c)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        scored.into_iter().take(count).map(|(_, c)| c).collect()
    }

    /// 获取统计信息
    pub fn stats(&self) -> EvictionStats {
        self.stats.read().clone()
    }

    /// 重置统计信息
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write();
        *stats = EvictionStats::default();
    }
}

// 辅助函数
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// 单元测试
#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct TestCandidate {
        id: String,
        last_access: u64,
        access_count: u64,
        size: usize,
        importance: f32,
        created_at: u64,
    }

    impl EvictionCandidate for TestCandidate {
        fn id(&self) -> &str {
            &self.id
        }
        fn last_access(&self) -> u64 {
            self.last_access
        }
        fn access_count(&self) -> u64 {
            self.access_count
        }
        fn size(&self) -> usize {
            self.size
        }
        fn importance(&self) -> f32 {
            self.importance
        }
        fn created_at(&self) -> u64 {
            self.created_at
        }
    }

    #[test]
    fn test_lru_selects_oldest() {
        let policy = EvictionPolicy::new(EvictionPolicyConfig::default());

        let candidates = vec![
            TestCandidate {
                id: "a".into(),
                last_access: 100,
                access_count: 10,
                size: 1000,
                importance: 0.9,
                created_at: 50,
            },
            TestCandidate {
                id: "b".into(),
                last_access: 200,
                access_count: 5,
                size: 2000,
                importance: 0.5,
                created_at: 60,
            },
            TestCandidate {
                id: "c".into(),
                last_access: 300,
                access_count: 15,
                size: 500,
                importance: 0.7,
                created_at: 70,
            },
        ];

        let victims = policy.select_victims(&candidates, 1);
        assert_eq!(victims[0].id(), "a"); // 最旧的应该被选中
    }

    #[test]
    fn test_lfu_selects_least_frequent() {
        let mut config = EvictionPolicyConfig::default();
        config.algorithm = EvictionAlgorithm::Lfu;
        let policy = EvictionPolicy::new(config);

        let candidates = vec![
            TestCandidate {
                id: "a".into(),
                last_access: 200,
                access_count: 5,
                size: 1000,
                importance: 0.9,
                created_at: 50,
            },
            TestCandidate {
                id: "b".into(),
                last_access: 200,
                access_count: 1,
                size: 2000,
                importance: 0.5,
                created_at: 60,
            },
            TestCandidate {
                id: "c".into(),
                last_access: 200,
                access_count: 15,
                size: 500,
                importance: 0.7,
                created_at: 70,
            },
        ];

        let victims = policy.select_victims(&candidates, 1);
        assert_eq!(victims[0].id(), "b"); // 访问次数最少的
    }

    #[test]
    fn test_should_evict_logic() {
        let config = EvictionPolicyConfig {
            max_capacity: 1000,
            high_watermark_pct: 0.8,
            ..Default::default()
        };
        let policy = EvictionPolicy::new(config);

        policy.update_usage(700);
        assert!(!policy.should_evict()); // 70% < 80%

        policy.update_usage(850);
        assert!(policy.should_evict()); // 85% > 80%
    }

    #[test]
    fn test_algorithm_switching() {
        let mut policy = EvictionPolicy::new(EvictionPolicyConfig::default());
        assert_eq!(policy.algorithm(), EvictionAlgorithm::Lru);

        policy.set_algorithm(EvictionAlgorithm::Lfu);
        assert_eq!(policy.algorithm(), EvictionAlgorithm::Lfu);
    }
}
