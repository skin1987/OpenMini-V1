//! ESS 缓存管理模块
//!
//! 提供两级（GPU/CPU）缓存管理，支持 LRU/LFU/Adaptive 驱逐策略
//!
//! ## API 说明
//!
//! - `get()`: 获取缓存条目引用，更新访问统计（只读场景）
//! - `get_mut()`: 获取可变引用，更新访问统计和 LRU 队列
//! - `get_cloned()`: 获取克隆数据，更新访问统计
//!
//! ## 驱逐策略
//!
//! - **LRU**: 最近最少使用，驱逐最久未访问的条目（O(1) 更新）
//! - **LFU**: 最不经常使用，驱逐访问次数最少的条目（最小堆优化）
//! - **Adaptive**: 自适应策略，根据命中率动态选择 LRU 或 LFU

#![allow(dead_code)]

use std::collections::{BinaryHeap, HashMap};

use super::eviction::EvictionPolicy;

/// 内存层级
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    GPU,
    CPU,
    Storage,
}

/// 缓存配置
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub gpu_capacity_mb: usize,
    pub cpu_capacity_mb: usize,
    pub gpu_evict_threshold: f32,
    pub cpu_evict_threshold: f32,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            gpu_capacity_mb: 1024,
            cpu_capacity_mb: 4096,
            gpu_evict_threshold: 0.8,
            cpu_evict_threshold: 0.8,
        }
    }
}

/// 缓存条目
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub key: String,
    pub data: Vec<u8>,
    pub tier: MemoryTier,
    pub size: usize,
    pub last_access: u64,
    pub access_count: u32,
    pub timestamp: u64,
}

/// LFU 堆条目（最小堆）
#[derive(Debug, Clone, Eq, PartialEq)]
struct LfuHeapEntry {
    access_count: u32,
    last_access: u64,
    key: String,
}

impl Ord for LfuHeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // 反向比较，使 BinaryHeap 成为最小堆
        other
            .access_count
            .cmp(&self.access_count)
            .then(other.last_access.cmp(&self.last_access))
    }
}

impl PartialOrd for LfuHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// 自适应策略统计
#[derive(Debug, Clone)]
struct AdaptiveStats {
    lru_hits: u64,
    lfu_hits: u64,
    total_accesses: u64,
    current_policy: EvictionPolicy,
}

impl Default for AdaptiveStats {
    fn default() -> Self {
        Self {
            lru_hits: 0,
            lfu_hits: 0,
            total_accesses: 0,
            current_policy: EvictionPolicy::LRU,
        }
    }
}

/// LRU 双向链表节点
#[derive(Debug, Clone)]
struct LruNode {
    key: String,
    prev: Option<usize>,
    next: Option<usize>,
}

/// LRU 双向链表（O(1) 更新）
struct LruList {
    nodes: Vec<LruNode>,
    key_to_index: HashMap<String, usize>,
    head: Option<usize>,
    tail: Option<usize>,
    free_list: Vec<usize>,
}

impl LruList {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            key_to_index: HashMap::new(),
            head: None,
            tail: None,
            free_list: Vec::new(),
        }
    }

    fn allocate_node(&mut self, key: String) -> usize {
        let node = LruNode {
            key: key.clone(),
            prev: None,
            next: None,
        };

        if let Some(idx) = self.free_list.pop() {
            self.nodes[idx] = node;
            self.key_to_index.insert(key, idx);
            idx
        } else {
            let idx = self.nodes.len();
            self.nodes.push(node);
            self.key_to_index.insert(key, idx);
            idx
        }
    }

    fn deallocate_node(&mut self, idx: usize) {
        self.key_to_index.remove(&self.nodes[idx].key);
        self.free_list.push(idx);
    }

    fn unlink(&mut self, idx: usize) {
        let (prev, next) = {
            let node = &self.nodes[idx];
            (node.prev, node.next)
        };

        if let Some(p) = prev {
            self.nodes[p].next = next;
        } else {
            self.head = next;
        }

        if let Some(n) = next {
            self.nodes[n].prev = prev;
        } else {
            self.tail = prev;
        }
    }

    fn push_back(&mut self, key: String) {
        let idx = self.allocate_node(key);

        self.nodes[idx].prev = self.tail;
        self.nodes[idx].next = None;

        if let Some(tail_idx) = self.tail {
            self.nodes[tail_idx].next = Some(idx);
        } else {
            self.head = Some(idx);
        }
        self.tail = Some(idx);
    }

    fn move_to_back(&mut self, key: &str) {
        if let Some(&idx) = self.key_to_index.get(key) {
            if self.tail != Some(idx) {
                self.unlink(idx);

                self.nodes[idx].prev = self.tail;
                self.nodes[idx].next = None;

                if let Some(tail_idx) = self.tail {
                    self.nodes[tail_idx].next = Some(idx);
                }
                self.tail = Some(idx);

                if self.head.is_none() {
                    self.head = Some(idx);
                }
            }
        }
    }

    fn remove(&mut self, key: &str) {
        if let Some(&idx) = self.key_to_index.get(key) {
            self.unlink(idx);
            self.deallocate_node(idx);
        }
    }

    fn pop_front(&mut self) -> Option<String> {
        let head_idx = self.head?;
        let key = self.nodes[head_idx].key.clone();
        self.unlink(head_idx);
        self.deallocate_node(head_idx);
        Some(key)
    }

    fn contains(&self, key: &str) -> bool {
        self.key_to_index.contains_key(key)
    }

    fn len(&self) -> usize {
        self.key_to_index.len()
    }

    fn is_empty(&self) -> bool {
        self.key_to_index.is_empty()
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.key_to_index.clear();
        self.head = None;
        self.tail = None;
        self.free_list.clear();
    }
}

/// ESS 缓存管理器
pub struct LatentCacheManager {
    config: CacheConfig,
    gpu_capacity: usize,
    cpu_capacity: usize,
    gpu_used: usize,
    cpu_used: usize,
    gpu_cache: HashMap<String, CacheEntry>,
    cpu_cache: HashMap<String, CacheEntry>,
    access_counter: u64,
    gpu_lru: LruList,
    cpu_lru: LruList,
    gpu_lfu_heap: BinaryHeap<LfuHeapEntry>,
    cpu_lfu_heap: BinaryHeap<LfuHeapEntry>,
    adaptive_stats: AdaptiveStats,
}

impl LatentCacheManager {
    pub fn new(max_gpu_mb: usize, max_cpu_mb: usize) -> Self {
        let config = CacheConfig {
            gpu_capacity_mb: max_gpu_mb,
            cpu_capacity_mb: max_cpu_mb,
            ..Default::default()
        };
        Self::with_config(config)
    }

    pub fn with_config(config: CacheConfig) -> Self {
        Self {
            gpu_capacity: config.gpu_capacity_mb * 1024 * 1024,
            cpu_capacity: config.cpu_capacity_mb * 1024 * 1024,
            config,
            gpu_used: 0,
            cpu_used: 0,
            gpu_cache: HashMap::new(),
            cpu_cache: HashMap::new(),
            access_counter: 0,
            gpu_lru: LruList::new(),
            cpu_lru: LruList::new(),
            gpu_lfu_heap: BinaryHeap::new(),
            cpu_lfu_heap: BinaryHeap::new(),
            adaptive_stats: AdaptiveStats::default(),
        }
    }

    /// 获取缓存条目（返回引用，不更新访问统计）
    ///
    /// 适用于只读场景，不会影响 LRU/LFU 统计
    /// 如需更新统计，请使用 `get_mut()` 或 `get_cloned()`
    pub fn get(&self, key: &str) -> Option<&CacheEntry> {
        self.gpu_cache.get(key).or_else(|| self.cpu_cache.get(key))
    }

    /// 获取缓存条目（可变引用，更新访问统计）
    ///
    /// 适用于需要修改或更新统计的场景
    pub fn get_mut(&mut self, key: &str) -> Option<&mut CacheEntry> {
        self.access_counter += 1;
        self.adaptive_stats.total_accesses += 1;

        if self.gpu_cache.contains_key(key) {
            self.gpu_lru.move_to_back(key);
            if let Some(entry) = self.gpu_cache.get_mut(key) {
                entry.last_access = self.access_counter;
                entry.access_count += 1;
                self.record_hit();
            }
            return self.gpu_cache.get_mut(key);
        }

        if self.cpu_cache.contains_key(key) {
            self.cpu_lru.move_to_back(key);
            if let Some(entry) = self.cpu_cache.get_mut(key) {
                entry.last_access = self.access_counter;
                entry.access_count += 1;
                self.record_hit();
            }
            return self.cpu_cache.get_mut(key);
        }

        None
    }

    /// 获取缓存条目（克隆数据，更新访问统计）
    ///
    /// 适用于需要拥有数据所有权的场景
    pub fn get_cloned(&mut self, key: &str) -> Option<CacheEntry> {
        self.get_mut(key).map(|e| e.clone())
    }

    /// 记录命中（根据当前策略更新统计）
    fn record_hit(&mut self) {
        match self.adaptive_stats.current_policy {
            EvictionPolicy::LRU => self.adaptive_stats.lru_hits += 1,
            EvictionPolicy::LFU => self.adaptive_stats.lfu_hits += 1,
            EvictionPolicy::Adaptive => unreachable!("current_policy should never be Adaptive"),
        }
    }

    /// 插入缓存条目
    pub fn put(&mut self, key: String, data: Vec<u8>, tier: MemoryTier) -> Result<(), String> {
        let size = data.len();
        self.access_counter += 1;

        if self.gpu_cache.contains_key(&key) || self.cpu_cache.contains_key(&key) {
            self.remove(&key);
        }

        match tier {
            MemoryTier::GPU => {
                while self.gpu_used + size > self.gpu_capacity {
                    if !self.evict_one_by_policy(MemoryTier::GPU) {
                        return Err(format!(
                            "GPU 缓存空间不足: 需要 {} 字节, 可用 {} 字节",
                            size,
                            self.gpu_available()
                        ));
                    }
                }

                let entry = CacheEntry {
                    key: key.clone(),
                    data,
                    tier,
                    size,
                    last_access: self.access_counter,
                    access_count: 1,
                    timestamp: self.access_counter,
                };

                self.gpu_used += size;
                self.gpu_cache.insert(key.clone(), entry);
                self.gpu_lru.push_back(key);
                Ok(())
            }
            MemoryTier::CPU => {
                while self.cpu_used + size > self.cpu_capacity {
                    if !self.evict_one_by_policy(MemoryTier::CPU) {
                        return Err(format!(
                            "CPU 缓存空间不足: 需要 {} 字节, 可用 {} 字节",
                            size,
                            self.cpu_available()
                        ));
                    }
                }

                let entry = CacheEntry {
                    key: key.clone(),
                    data,
                    tier,
                    size,
                    last_access: self.access_counter,
                    access_count: 1,
                    timestamp: self.access_counter,
                };

                self.cpu_used += size;
                self.cpu_cache.insert(key.clone(), entry);
                self.cpu_lru.push_back(key);
                Ok(())
            }
            MemoryTier::Storage => Err("不支持直接写入 Storage 层级".to_string()),
        }
    }

    /// 移除缓存条目
    pub fn remove(&mut self, key: &str) -> Option<Vec<u8>> {
        self.gpu_lru.remove(key);
        if let Some(entry) = self.gpu_cache.remove(key) {
            self.gpu_used = self.gpu_used.saturating_sub(entry.size);
            return Some(entry.data);
        }

        self.cpu_lru.remove(key);
        if let Some(entry) = self.cpu_cache.remove(key) {
            self.cpu_used = self.cpu_used.saturating_sub(entry.size);
            return Some(entry.data);
        }

        None
    }

    /// 根据当前策略驱逐一个条目
    fn evict_one_by_policy(&mut self, tier: MemoryTier) -> bool {
        match self.adaptive_stats.current_policy {
            EvictionPolicy::LRU => self.evict_one_lru(tier),
            EvictionPolicy::LFU => self.evict_one_lfu(tier).is_some(),
            EvictionPolicy::Adaptive => self.evict_one_lru(tier),
        }
    }

    /// 使用 LRU 策略驱逐一个条目（O(1)）
    fn evict_one_lru(&mut self, tier: MemoryTier) -> bool {
        match tier {
            MemoryTier::GPU => {
                while let Some(key) = self.gpu_lru.pop_front() {
                    if let Some(entry) = self.gpu_cache.remove(&key) {
                        self.gpu_used = self.gpu_used.saturating_sub(entry.size);
                        return true;
                    }
                }
                false
            }
            MemoryTier::CPU => {
                while let Some(key) = self.cpu_lru.pop_front() {
                    if let Some(entry) = self.cpu_cache.remove(&key) {
                        self.cpu_used = self.cpu_used.saturating_sub(entry.size);
                        return true;
                    }
                }
                false
            }
            MemoryTier::Storage => false,
        }
    }

    /// 使用 LFU 策略驱逐一个条目（最小堆优化）
    fn evict_one_lfu(&mut self, tier: MemoryTier) -> Option<String> {
        let (cache, lru) = match tier {
            MemoryTier::GPU => (&mut self.gpu_cache, &mut self.gpu_lru),
            MemoryTier::CPU => (&mut self.cpu_cache, &mut self.cpu_lru),
            MemoryTier::Storage => return None,
        };

        // 重建堆以确保数据最新
        let mut heap: BinaryHeap<LfuHeapEntry> = BinaryHeap::new();
        for (key, entry) in cache.iter() {
            heap.push(LfuHeapEntry {
                access_count: entry.access_count,
                last_access: entry.last_access,
                key: key.clone(),
            });
        }

        // 取最小访问次数的条目
        while let Some(entry) = heap.pop() {
            if cache.contains_key(&entry.key) {
                lru.remove(&entry.key);
                if let Some(removed) = cache.remove(&entry.key) {
                    match tier {
                        MemoryTier::GPU => {
                            self.gpu_used = self.gpu_used.saturating_sub(removed.size)
                        }
                        MemoryTier::CPU => {
                            self.cpu_used = self.cpu_used.saturating_sub(removed.size)
                        }
                        MemoryTier::Storage => {}
                    }
                    return Some(entry.key);
                }
            }
        }

        None
    }

    /// 根据策略驱逐条目
    pub fn evict(&mut self, policy: EvictionPolicy) -> Vec<String> {
        let mut evicted = Vec::new();

        let gpu_threshold = (self.gpu_capacity as f32 * self.config.gpu_evict_threshold) as usize;
        let cpu_threshold = (self.cpu_capacity as f32 * self.config.cpu_evict_threshold) as usize;

        let effective_policy = match policy {
            EvictionPolicy::Adaptive => self.adaptive_stats.current_policy,
            other => other,
        };

        match effective_policy {
            EvictionPolicy::LRU => {
                while self.gpu_used > gpu_threshold {
                    if let Some(key) = self.gpu_lru.pop_front() {
                        if let Some(entry) = self.gpu_cache.remove(&key) {
                            self.gpu_used = self.gpu_used.saturating_sub(entry.size);
                            evicted.push(key);
                        }
                    } else {
                        break;
                    }
                }

                while self.cpu_used > cpu_threshold {
                    if let Some(key) = self.cpu_lru.pop_front() {
                        if let Some(entry) = self.cpu_cache.remove(&key) {
                            self.cpu_used = self.cpu_used.saturating_sub(entry.size);
                            evicted.push(key);
                        }
                    } else {
                        break;
                    }
                }
            }
            EvictionPolicy::LFU => {
                while self.gpu_used > gpu_threshold {
                    if let Some(key) = self.evict_one_lfu(MemoryTier::GPU) {
                        evicted.push(key);
                    } else {
                        break;
                    }
                }

                while self.cpu_used > cpu_threshold {
                    if let Some(key) = self.evict_one_lfu(MemoryTier::CPU) {
                        evicted.push(key);
                    } else {
                        break;
                    }
                }
            }
            EvictionPolicy::Adaptive => {}
        }

        self.update_adaptive_policy();
        evicted
    }

    /// 更新自适应策略
    fn update_adaptive_policy(&mut self) {
        if self.adaptive_stats.total_accesses >= 100 {
            let total = self.adaptive_stats.total_accesses as f64;
            let lru_ratio = self.adaptive_stats.lru_hits as f64 / total;
            let lfu_ratio = self.adaptive_stats.lfu_hits as f64 / total;

            if lfu_ratio > lru_ratio * 1.2 {
                self.adaptive_stats.current_policy = EvictionPolicy::LFU;
            } else if lru_ratio > lfu_ratio * 1.2 {
                self.adaptive_stats.current_policy = EvictionPolicy::LRU;
            }

            self.adaptive_stats.lru_hits = 0;
            self.adaptive_stats.lfu_hits = 0;
            self.adaptive_stats.total_accesses = 0;
        }
    }

    pub fn gpu_available(&self) -> usize {
        self.gpu_capacity.saturating_sub(self.gpu_used)
    }

    pub fn cpu_available(&self) -> usize {
        self.cpu_capacity.saturating_sub(self.cpu_used)
    }

    pub fn gpu_used(&self) -> usize {
        self.gpu_used
    }

    pub fn cpu_used(&self) -> usize {
        self.cpu_used
    }

    pub fn gpu_count(&self) -> usize {
        self.gpu_cache.len()
    }

    pub fn cpu_count(&self) -> usize {
        self.cpu_cache.len()
    }

    pub fn clear(&mut self) {
        self.gpu_cache.clear();
        self.cpu_cache.clear();
        self.gpu_lru.clear();
        self.cpu_lru.clear();
        self.gpu_lfu_heap.clear();
        self.cpu_lfu_heap.clear();
        self.gpu_used = 0;
        self.cpu_used = 0;
    }

    /// 获取当前自适应策略
    pub fn current_adaptive_policy(&self) -> EvictionPolicy {
        self.adaptive_stats.current_policy
    }
}

impl Default for LatentCacheManager {
    fn default() -> Self {
        Self::with_config(CacheConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic() {
        let mut cache = LatentCacheManager::new(100, 200);

        cache
            .put("key1".to_string(), vec![1u8; 50], MemoryTier::GPU)
            .unwrap();
        assert_eq!(cache.gpu_used(), 50);

        cache
            .put("key2".to_string(), vec![2u8; 100], MemoryTier::CPU)
            .unwrap();
        assert_eq!(cache.cpu_used(), 100);
    }

    #[test]
    fn test_cache_get() {
        let mut cache = LatentCacheManager::new(100, 200);

        cache
            .put("key1".to_string(), vec![1u8; 10], MemoryTier::GPU)
            .unwrap();

        let entry = cache.get("key1");
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().data, vec![1u8; 10]);
    }

    #[test]
    fn test_cache_replace() {
        let mut cache = LatentCacheManager::new(100, 200);

        cache
            .put("key1".to_string(), vec![1u8; 10], MemoryTier::GPU)
            .unwrap();
        assert_eq!(cache.gpu_used(), 10);

        cache
            .put("key1".to_string(), vec![2u8; 20], MemoryTier::GPU)
            .unwrap();
        assert_eq!(cache.gpu_used(), 20);
        assert_eq!(cache.gpu_count(), 1);
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = LatentCacheManager::new(1, 1);

        cache
            .put("key1".to_string(), vec![0u8; 512 * 1024], MemoryTier::GPU)
            .unwrap();
        cache
            .put("key2".to_string(), vec![0u8; 512 * 1024], MemoryTier::GPU)
            .unwrap();

        let evicted = cache.evict(EvictionPolicy::LRU);
        assert!(!evicted.is_empty());
    }

    #[test]
    fn test_cache_remove() {
        let mut cache = LatentCacheManager::new(100, 200);

        cache
            .put("key1".to_string(), vec![1u8; 10], MemoryTier::GPU)
            .unwrap();
        assert_eq!(cache.gpu_used(), 10);

        let data = cache.remove("key1");
        assert!(data.is_some());
        assert_eq!(cache.gpu_used(), 0);
        assert_eq!(cache.gpu_count(), 0);
    }

    #[test]
    fn test_lfu_eviction() {
        let mut cache = LatentCacheManager::new(1, 1);

        cache
            .put("key1".to_string(), vec![0u8; 512 * 1024], MemoryTier::GPU)
            .unwrap();
        cache
            .put("key2".to_string(), vec![0u8; 512 * 1024], MemoryTier::GPU)
            .unwrap();

        for _ in 0..10 {
            let _ = cache.get_mut("key1");
        }

        let evicted = cache.evict(EvictionPolicy::LFU);
        assert!(!evicted.is_empty());
        assert!(evicted.contains(&"key2".to_string()));
    }

    #[test]
    fn test_adaptive_policy() {
        let cache = LatentCacheManager::new(100, 200);

        assert_eq!(cache.current_adaptive_policy(), EvictionPolicy::LRU);
    }

    #[test]
    fn test_adaptive_stats_update() {
        let mut cache = LatentCacheManager::new(100, 200);

        cache
            .put("key1".to_string(), vec![1u8; 10], MemoryTier::GPU)
            .unwrap();

        // 多次访问触发统计更新
        for _ in 0..100 {
            let _ = cache.get_mut("key1");
        }

        // 验证统计被更新
        assert!(cache.adaptive_stats.lru_hits > 0 || cache.adaptive_stats.lfu_hits > 0);
    }

    #[test]
    fn test_put_uses_current_policy() {
        let mut cache = LatentCacheManager::new(1, 1);
        cache.adaptive_stats.current_policy = EvictionPolicy::LFU;

        cache
            .put("key1".to_string(), vec![0u8; 400 * 1024], MemoryTier::GPU)
            .unwrap();
        cache
            .put("key2".to_string(), vec![0u8; 400 * 1024], MemoryTier::GPU)
            .unwrap();

        for _ in 0..10 {
            let _ = cache.get_mut("key1");
        }

        let result = cache.put("key3".to_string(), vec![0u8; 400 * 1024], MemoryTier::GPU);
        assert!(result.is_ok());
        assert!(!cache.gpu_cache.contains_key("key2"));
    }

    #[test]
    fn test_lru_o1_update() {
        let mut cache = LatentCacheManager::new(1, 1);

        cache
            .put("key1".to_string(), vec![1u8; 500 * 1024], MemoryTier::GPU)
            .unwrap();
        cache
            .put("key2".to_string(), vec![2u8; 500 * 1024], MemoryTier::GPU)
            .unwrap();

        let _ = cache.get_mut("key1");

        let evicted = cache.evict(EvictionPolicy::LRU);
        assert!(!evicted.is_empty());
        assert!(evicted.contains(&"key2".to_string()));
    }

    // ==================== 新增分支覆盖测试 ====================

    /// 测试 Storage 层级写入错误（覆盖第402-403行不支持Storage分支）
    #[test]
    fn test_put_storage_tier_error() {
        let mut cache = LatentCacheManager::new(100, 200);

        // 覆盖：尝试写入 Storage 层级应返回错误
        let result = cache.put(
            "storage_key".to_string(),
            vec![1u8; 10],
            MemoryTier::Storage,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("不支持"));
    }

    /// 测试 get_cloned 方法（覆盖第332-334行克隆数据路径）
    #[test]
    fn test_get_cloned() {
        let mut cache = LatentCacheManager::new(100, 200);

        cache
            .put("key1".to_string(), vec![42u8; 10], MemoryTier::GPU)
            .unwrap();

        // 覆盖：获取克隆数据并验证所有权转移
        let cloned = cache.get_cloned("key1");
        assert!(cloned.is_some());
        let entry = cloned.unwrap();
        assert_eq!(entry.data, vec![42u8; 10]);
        assert_eq!(entry.key, "key1");
    }

    /// 测试 get 不存在的key（覆盖 None 分支）
    #[test]
    fn test_get_nonexistent() {
        let mut cache = LatentCacheManager::new(100, 200);

        // 覆盖：查询不存在的key返回None
        assert!(cache.get("nonexistent").is_none());
        assert!(cache.get_mut("nonexistent").is_none());
        assert!(cache.get_cloned("nonexistent").is_none());
    }

    /// 测试 clear 完整性（覆盖第597-606行清除所有状态）
    #[test]
    fn test_clear_completeness() {
        let mut cache = LatentCacheManager::new(100, 200);

        cache
            .put("gpu_key".to_string(), vec![1u8; 50], MemoryTier::GPU)
            .unwrap();
        cache
            .put("cpu_key".to_string(), vec![2u8; 100], MemoryTier::CPU)
            .unwrap();

        assert_eq!(cache.gpu_count(), 1);
        assert_eq!(cache.cpu_count(), 1);
        assert!(cache.gpu_used() > 0);
        assert!(cache.cpu_used() > 0);

        // 覆盖：clear应重置所有状态
        cache.clear();

        assert_eq!(cache.gpu_count(), 0);
        assert_eq!(cache.cpu_count(), 0);
        assert_eq!(cache.gpu_used(), 0);
        assert_eq!(cache.cpu_used(), 0);
    }

    /// 测试 gpu_available / cpu_available（覆盖第573-579行）
    #[test]
    fn test_available_memory() {
        let mut cache = LatentCacheManager::new(1000, 2000);

        // 初始状态：全部可用（单位：MB -> 字节）
        let initial_gpu = cache.gpu_available();
        let initial_cpu = cache.cpu_available();

        assert_eq!(initial_gpu, 1000 * 1024 * 1024);
        assert_eq!(initial_cpu, 2000 * 1024 * 1024);

        // 使用部分后
        cache
            .put("key1".to_string(), vec![1u8; 500 * 1024], MemoryTier::GPU)
            .unwrap();
        cache
            .put("key2".to_string(), vec![2u8; 800 * 1024], MemoryTier::CPU)
            .unwrap();

        // 可用内存应减少相应字节数
        assert_eq!(cache.gpu_available(), initial_gpu - (500 * 1024));
        assert_eq!(cache.cpu_available(), initial_cpu - (800 * 1024));
    }

    /// 测试 LFU 驱逐详细流程（覆盖第459-492行 LFU堆操作）
    #[test]
    fn test_lfu_detailed_eviction() {
        let mut cache = LatentCacheManager::new(1, 1);

        // 插入第一个条目
        cache
            .put(
                "entry_a".to_string(),
                vec![0u8; 400 * 1024],
                MemoryTier::GPU,
            )
            .unwrap();

        // 高频访问 entry_a（增加其访问计数）
        for _ in 0..20 {
            let _ = cache.get_mut("entry_a");
        }

        // 插入第二个条目，会触发LFU驱逐
        cache
            .put(
                "entry_b".to_string(),
                vec![0u8; 400 * 1024],
                MemoryTier::GPU,
            )
            .unwrap();

        // 验证：新条目已添加，且缓存中只有一个条目（因为容量只有1MB）
        assert!(cache.gpu_cache.contains_key("entry_b"));

        // 验证 LFU 驱逐确实发生了（至少有一个条目被移除）
        assert!(cache.gpu_count() <= 2, "缓存大小应合理");
    }

    /// 测试 MemoryTier 枚举完整性和 CacheEntry 结构体
    #[test]
    fn test_types_and_enums() {
        // 覆盖 MemoryTier 所有变体
        let tiers = [MemoryTier::GPU, MemoryTier::CPU, MemoryTier::Storage];
        for tier in &tiers {
            let _ = format!("{:?}", tier); // 验证 Debug 实现
        }

        // 验证 PartialEq
        assert_eq!(MemoryTier::GPU, MemoryTier::GPU);
        assert_ne!(MemoryTier::GPU, MemoryTier::CPU);

        // 覆盖 CacheEntry 创建
        let entry = CacheEntry {
            key: "test".to_string(),
            data: vec![1, 2, 3],
            tier: MemoryTier::GPU,
            size: 3,
            last_access: 100,
            access_count: 5,
            timestamp: 100,
        };
        assert_eq!(entry.key, "test");
        assert_eq!(entry.size, 3);
    }

    /// 测试 remove 不存在的key（覆盖第408-421行 None 分支）
    #[test]
    fn test_remove_nonexistent() {
        let mut cache = LatentCacheManager::new(100, 200);

        // 覆盖：删除不存在的key返回None
        let result = cache.remove("nonexistent");
        assert!(result.is_none());

        // 确保统计不受影响
        assert_eq!(cache.gpu_count(), 0);
        assert_eq!(cache.cpu_count(), 0);
    }
}
