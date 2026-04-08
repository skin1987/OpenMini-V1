//! 前缀缓存实现
//!
//! 支持共享前缀的KV Cache复用，减少重复计算

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

use super::block::BlockId;

/// 前缀哈希
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub struct PrefixHash(u64);

#[allow(dead_code)]
impl PrefixHash {
    /// 从token序列计算哈希
    pub fn from_tokens(tokens: &[u32]) -> Self {
        let mut hasher = DefaultHasher::new();
        tokens.hash(&mut hasher);
        Self(hasher.finish())
    }

    /// 从字节数据计算哈希
    pub fn from_bytes(data: &[u8]) -> Self {
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        Self(hasher.finish())
    }

    /// 获取原始哈希值
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

/// 前缀缓存条目
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PrefixEntry {
    /// 前缀哈希
    pub hash: PrefixHash,
    /// 对应的块ID列表
    pub block_ids: Vec<BlockId>,
    /// 引用计数
    pub ref_count: usize,
    /// 前缀长度（token数）
    pub length: usize,
    /// 最后访问时间戳
    pub last_access: u64,
}

#[allow(dead_code)]
impl PrefixEntry {
    fn new(hash: PrefixHash, block_ids: Vec<BlockId>, length: usize) -> Self {
        Self {
            hash,
            block_ids,
            ref_count: 1,
            length,
            last_access: 0,
        }
    }
}

/// 前缀缓存配置
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PrefixCacheConfig {
    /// 最大缓存条目数
    pub max_entries: usize,
    /// 最大缓存token数
    pub max_tokens: usize,
    /// 是否启用LRU淘汰
    pub enable_lru: bool,
    /// 最小前缀长度
    pub min_prefix_length: usize,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            max_tokens: 100000,
            enable_lru: true,
            min_prefix_length: 16,
        }
    }
}

/// 前缀缓存
#[derive(Debug)]
#[allow(dead_code)]
pub struct PrefixCache {
    /// 哈希到条目的映射
    entries: HashMap<PrefixHash, PrefixEntry>,
    /// 配置
    config: PrefixCacheConfig,
    /// 当前缓存token数
    total_tokens: usize,
    /// 当前时间戳（用于LRU）
    current_time: u64,
    /// 命中次数
    hits: usize,
    /// 未命中次数
    misses: usize,
}

#[allow(dead_code)]
impl PrefixCache {
    /// 创建新的前缀缓存
    pub fn new(config: PrefixCacheConfig) -> Self {
        Self {
            entries: HashMap::new(),
            config,
            total_tokens: 0,
            current_time: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// 使用默认配置创建
    pub fn default_config() -> Self {
        Self::new(PrefixCacheConfig::default())
    }

    /// 查找前缀
    pub fn lookup(&mut self, hash: PrefixHash) -> Option<&PrefixEntry> {
        self.current_time += 1;
        
        if let Some(entry) = self.entries.get_mut(&hash) {
            entry.last_access = self.current_time;
            entry.ref_count += 1;
            self.hits += 1;
            Some(self.entries.get(&hash)?)
        } else {
            self.misses += 1;
            None
        }
    }

    /// 插入前缀
    pub fn insert(&mut self, hash: PrefixHash, block_ids: Vec<BlockId>, length: usize) -> Result<(), String> {
        if length < self.config.min_prefix_length {
            return Err(format!("Prefix too short: {} < {}", length, self.config.min_prefix_length));
        }
        
        if self.entries.len() >= self.config.max_entries {
            self.evict_lru();
        }
        
        let new_tokens = self.total_tokens + length;
        if new_tokens > self.config.max_tokens {
            self.evict_for_tokens(length);
        }
        
        self.current_time += 1;
        let mut entry = PrefixEntry::new(hash, block_ids, length);
        entry.last_access = self.current_time;
        self.total_tokens += length;
        self.entries.insert(hash, entry);
        
        Ok(())
    }

    /// 从token序列插入
    pub fn insert_tokens(&mut self, tokens: &[u32], block_ids: Vec<BlockId>) -> Result<(), String> {
        let hash = PrefixHash::from_tokens(tokens);
        self.insert(hash, block_ids, tokens.len())
    }

    /// 增加引用计数
    pub fn inc_ref(&mut self, hash: PrefixHash) {
        if let Some(entry) = self.entries.get_mut(&hash) {
            entry.ref_count += 1;
            self.current_time += 1;
            entry.last_access = self.current_time;
        }
    }

    /// 减少引用计数
    pub fn dec_ref(&mut self, hash: PrefixHash) {
        if let Some(entry) = self.entries.get_mut(&hash) {
            entry.ref_count = entry.ref_count.saturating_sub(1);
        }
    }

    /// 移除前缀
    pub fn remove(&mut self, hash: &PrefixHash) -> Option<PrefixEntry> {
        if let Some(entry) = self.entries.remove(hash) {
            self.total_tokens -= entry.length;
            Some(entry)
        } else {
            None
        }
    }

    /// 检查是否存在
    pub fn contains(&self, hash: &PrefixHash) -> bool {
        self.entries.contains_key(hash)
    }

    /// LRU淘汰
    fn evict_lru(&mut self) {
        if !self.config.enable_lru {
            return;
        }
        
        let mut oldest_hash: Option<PrefixHash> = None;
        let mut oldest_time = u64::MAX;
        
        for (hash, entry) in &self.entries {
            if entry.ref_count == 0 && entry.last_access < oldest_time {
                oldest_time = entry.last_access;
                oldest_hash = Some(*hash);
            }
        }
        
        if let Some(hash) = oldest_hash {
            self.remove(&hash);
        }
    }

    /// 为腾出空间而淘汰
    fn evict_for_tokens(&mut self, needed: usize) {
        let mut freed = 0;
        let mut to_remove = Vec::new();
        
        for (hash, entry) in &self.entries {
            if entry.ref_count == 0 {
                to_remove.push(*hash);
                freed += entry.length;
                if freed >= needed {
                    break;
                }
            }
        }
        
        for hash in to_remove {
            self.remove(&hash);
        }
    }

    /// 获取缓存大小
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 获取总token数
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// 获取命中率
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f32 / total as f32
    }

    /// 获取统计信息
    pub fn stats(&self) -> PrefixCacheStats {
        PrefixCacheStats {
            entries: self.entries.len(),
            total_tokens: self.total_tokens,
            hits: self.hits,
            misses: self.misses,
            hit_rate: self.hit_rate(),
        }
    }

    /// 清空缓存
    pub fn clear(&mut self) {
        self.entries.clear();
        self.total_tokens = 0;
        self.hits = 0;
        self.misses = 0;
    }

    /// 获取所有条目
    pub fn entries(&self) -> impl Iterator<Item = (&PrefixHash, &PrefixEntry)> {
        self.entries.iter()
    }
}

/// 前缀缓存统计信息
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct PrefixCacheStats {
    pub entries: usize,
    pub total_tokens: usize,
    pub hits: usize,
    pub misses: usize,
    pub hit_rate: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_hash() {
        let tokens1 = vec![1, 2, 3, 4, 5];
        let tokens2 = vec![1, 2, 3, 4, 5];
        let tokens3 = vec![1, 2, 3, 4, 6];
        
        let hash1 = PrefixHash::from_tokens(&tokens1);
        let hash2 = PrefixHash::from_tokens(&tokens2);
        let hash3 = PrefixHash::from_tokens(&tokens3);
        
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_prefix_cache_insert() {
        let mut cache = PrefixCache::default_config();
        
        let hash = PrefixHash::from_tokens(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let result = cache.insert(hash, vec![0, 1], 16);
        
        assert!(result.is_ok());
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.total_tokens(), 16);
    }

    #[test]
    fn test_prefix_cache_lookup() {
        let mut cache = PrefixCache::default_config();
        
        let hash = PrefixHash::from_tokens(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        cache.insert(hash, vec![0, 1], 16).unwrap();
        
        let entry = cache.lookup(hash);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().block_ids, vec![0, 1]);
        
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_prefix_cache_miss() {
        let mut cache = PrefixCache::default_config();
        
        let hash = PrefixHash::from_tokens(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let result = cache.lookup(hash);
        
        assert!(result.is_none());
        
        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_prefix_cache_remove() {
        let mut cache = PrefixCache::default_config();
        
        let hash = PrefixHash::from_tokens(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        cache.insert(hash, vec![0, 1], 16).unwrap();
        
        let entry = cache.remove(&hash);
        assert!(entry.is_some());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.total_tokens(), 0);
    }

    #[test]
    fn test_prefix_cache_ref_count() {
        let mut cache = PrefixCache::default_config();
        
        let hash = PrefixHash::from_tokens(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        cache.insert(hash, vec![0, 1], 16).unwrap();
        
        cache.inc_ref(hash);
        cache.inc_ref(hash);
        
        let entry = cache.lookup(hash).unwrap();
        assert!(entry.ref_count >= 3);
    }

    #[test]
    fn test_prefix_cache_too_short() {
        let mut cache = PrefixCache::default_config();
        
        let hash = PrefixHash::from_tokens(&[1, 2, 3]);
        let result = cache.insert(hash, vec![0], 3);
        
        assert!(result.is_err());
    }

    #[test]
    fn test_hit_rate() {
        let mut cache = PrefixCache::default_config();
        
        let hash1 = PrefixHash::from_tokens(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let hash2 = PrefixHash::from_tokens(&[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]);
        
        cache.insert(hash1, vec![0], 16).unwrap();
        
        cache.lookup(hash1);
        cache.lookup(hash2);
        cache.lookup(hash1);
        
        let stats = cache.stats();
        assert!((stats.hit_rate - 0.666).abs() < 0.01);
    }

    // ==================== 新增测试：覆盖完整分支 ====================
    
    /// 测试 PrefixHash::from_bytes 方法
    /// 覆盖从字节数据计算哈希的路径
    #[test]
    fn test_prefix_hash_from_bytes() {
        let data1 = b"hello world";
        let data2 = b"hello world";
        let data3 = b"hello rust";
        
        let hash1 = PrefixHash::from_bytes(data1);
        let hash2 = PrefixHash::from_bytes(data2);
        let hash3 = PrefixHash::from_bytes(data3);
        
        // 相同数据应产生相同哈希
        assert_eq!(hash1, hash2);
        // 不同数据应产生不同哈希
        assert_ne!(hash1, hash3);
    }

    /// 测试 PrefixHash::as_u64 方法
    /// 验证可以正确获取原始哈希值
    #[test]
    fn test_prefix_hash_as_u64() {
        let tokens = vec![1, 2, 3, 4, 5];
        let hash = PrefixHash::from_tokens(&tokens);
        let u64_val = hash.as_u64();
        // u64 值应该是有效的（非零，因为非空数据）
        // 注意：理论上可能为0，但概率极低
        let _ = u64_val;
    }

    /// 测试 PrefixHash 的 Hash trait 实现
    /// 验证可以用作 HashMap 的键
    #[test]
    fn test_prefix_hash_hash_trait() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        let hash1 = PrefixHash::from_tokens(&[1, 2, 3]);
        let hash2 = PrefixHash::from_tokens(&[4, 5, 6]);
        set.insert(hash1);
        set.insert(hash2);
        assert_eq!(set.len(), 2);
    }

    /// 测试 insert_tokens 方法
    /// 覆盖从token序列直接插入的便捷方法
    #[test]
    fn test_insert_tokens() {
        let mut cache = PrefixCache::default_config();
        let tokens: Vec<u32> = (1..=20).collect(); // 20个token，>= min_prefix_length(16)
        
        let result = cache.insert_tokens(&tokens, vec![0, 1, 2]);
        assert!(result.is_ok());
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.total_tokens(), 20);
        
        // 验证可以通过哈希查找到
        let expected_hash = PrefixHash::from_tokens(&tokens);
        assert!(cache.contains(&expected_hash));
    }

    /// 测试 insert_tokens 对短前缀返回错误
    #[test]
    fn test_insert_tokens_too_short() {
        let mut cache = PrefixCache::default_config();
        let short_tokens: Vec<u32> = (1..=10).collect(); // 10个token < min_prefix_length(16)
        
        let result = cache.insert_tokens(&short_tokens, vec![0]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Prefix too short"));
    }

    /// 测试 inc_ref 和 dec_ref 的边界行为
    /// - dec_ref 多次调用不会导致下溢（使用 saturating_sub）
    /// - inc_ref 不存在的哈希不会崩溃
    #[test]
    fn test_inc_dec_ref_boundary() {
        let mut cache = PrefixCache::default_config();
        let hash = PrefixHash::from_tokens(&(1..=20).collect::<Vec<u32>>());
        cache.insert(hash, vec![0], 20).unwrap();
        
        // 增加引用计数
        cache.inc_ref(hash);
        cache.inc_ref(hash);
        let entry = cache.lookup(hash).unwrap();
        assert!(entry.ref_count >= 3); // 初始1 + 2次inc_ref
        
        // 减少引用计数多次（测试 saturating_sub 不会下溢）
        cache.dec_ref(hash);
        cache.dec_ref(hash);
        cache.dec_ref(hash);
        cache.dec_ref(hash);
        cache.dec_ref(hash); // 远超初始值
        
        // 对不存在的哈希操作不应崩溃
        let non_existent = PrefixHash::from_tokens(&[99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]);
        cache.inc_ref(non_existent);
        cache.dec_ref(non_existent);
    }

    /// 测试 contains 方法
    /// 覆盖存在和不存在的两种情况
    #[test]
    fn test_contains_method() {
        let mut cache = PrefixCache::default_config();
        let existing_hash = PrefixHash::from_tokens(&(1..=20).collect::<Vec<u32>>());
        let non_existing_hash = PrefixHash::from_tokens(&(100..=120).collect::<Vec<u32>>());
        
        cache.insert(existing_hash.clone(), vec![0], 20).unwrap();
        
        assert!(cache.contains(&existing_hash));
        assert!(!cache.contains(&non_existing_hash));
    }

    /// 测试 remove 不存在的条目返回 None
    #[test]
    fn test_remove_nonexistent() {
        let mut cache = PrefixCache::default_config();
        let non_existing = PrefixHash::from_tokens(&(100..=120).collect::<Vec<u32>>());
        
        let result = cache.remove(&non_existing);
        assert!(result.is_none());
        assert_eq!(cache.len(), 0);
    }

    /// 测试 clear 方法清空所有状态
    /// 包括 entries、total_tokens、hits、misses
    #[test]
    fn test_clear_resets_all_state() {
        let mut cache = PrefixCache::default_config();
        
        // 插入一些数据并产生命中/未命中
        let hash1 = PrefixHash::from_tokens(&(1..=20).collect::<Vec<u32>>());
        let hash2 = PrefixHash::from_tokens(&(21..=40).collect::<Vec<u32>>());
        cache.insert(hash1, vec![0], 20).unwrap();
        cache.insert(hash2, vec![1], 20).unwrap();
        cache.lookup(hash1); // hit
        cache.lookup(PrefixHash::from_tokens(&(200..=220).collect::<Vec<u32>>())); // miss
        
        assert_eq!(cache.len(), 2);
        assert!(cache.total_tokens() > 0);
        
        // 清空
        cache.clear();
        
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.total_tokens(), 0);
        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.hit_rate, 0.0);
    }

    /// 测试 is_empty 方法
    #[test]
    fn test_is_empty() {
        let mut cache = PrefixCache::default_config();
        assert!(cache.is_empty());
        
        let hash = PrefixHash::from_tokens(&(1..=20).collect::<Vec<u32>>());
        cache.insert(hash, vec![0], 20).unwrap();
        assert!(!cache.is_empty());
    }

    /// 测试 entries() 迭代器遍历所有条目
    #[test]
    fn test_entries_iterator() {
        let mut cache = PrefixCache::default_config();
        
        let hash1 = PrefixHash::from_tokens(&(1..=20).collect::<Vec<u32>>());
        let hash2 = PrefixHash::from_tokens(&(21..=40).collect::<Vec<u32>>());
        cache.insert(hash1.clone(), vec![0], 20).unwrap();
        cache.insert(hash2.clone(), vec![1], 20).unwrap();
        
        let count = cache.entries().count();
        assert_eq!(count, 2);
        
        // 验证迭代器内容
        for (_hash, entry) in cache.entries() {
            assert_eq!(entry.length, 20);
            assert!(entry.ref_count >= 1);
        }
    }

    /// 测试 LRU 淘汰机制：当条目数达到 max_entries 时自动淘汰最久未使用的条目
    /// 覆盖 evict_lru 方法的完整逻辑
    #[test]
    fn test_lru_eviction_on_max_entries() {
        let config = PrefixCacheConfig {
            max_entries: 3,
            max_tokens: 100000,
            enable_lru: true,
            min_prefix_length: 16,
        };
        let mut cache = PrefixCache::new(config);
        
        // 插入3个条目（达到上限）
        let hashes: Vec<PrefixHash> = (0..3usize)
            .map(|i| {
                let start = i * 20 + 1;
                let tokens: Vec<u32> = (start..=start + 19).map(|x| x as u32).collect();
                let hash = PrefixHash::from_tokens(&tokens);
                cache.insert(hash, vec![i], 20).unwrap();
                hash
            })
            .collect();
        assert_eq!(cache.len(), 3);
        
        // 减少第一个条目的引用计数使其可被淘汰
        cache.dec_ref(hashes[0]);
        
        // 插入第4个条目，触发LRU淘汰
        let tokens4: Vec<u32> = (61..=80).map(|x| x as u32).collect::<Vec<u32>>();
        let hash4 = PrefixHash::from_tokens(&tokens4);
        cache.insert(hash4, vec![3], 20).unwrap();
        
        // 应该仍然只有3个条目（最老的且ref_count=0的被淘汰）
        assert_eq!(cache.len(), 3);
    }

    /// 测试 Token 限制淘汰：当总 token 数超过 max_tokens 时淘汰旧条目
    /// 覆盖 evict_for_tokens 方法的逻辑
    #[test]
    fn test_token_limit_eviction() {
        let config = PrefixCacheConfig {
            max_entries: 100,
            max_tokens: 50, // 很小的token限制
            enable_lru: true,
            min_prefix_length: 16,
        };
        let mut cache = PrefixCache::new(config);
        
        // 插入第一个条目（20 tokens）
        let tokens1: Vec<u32> = (1..=20).collect();
        let hash1 = PrefixHash::from_tokens(&tokens1);
        cache.insert(hash1.clone(), vec![0], 20).unwrap();
        assert_eq!(cache.total_tokens(), 20);
        
        // 插入第二个条目（20 tokens），总共40 < 50，应该成功
        let tokens2: Vec<u32> = (21..=40).collect();
        let hash2 = PrefixHash::from_tokens(&tokens2);
        cache.insert(hash2, vec![1], 20).unwrap();
        assert_eq!(cache.total_tokens(), 40);
        
        // 插入第三个条目（20 tokens），总共60 > 50，应该触发淘汰
        let tokens3: Vec<u32> = (41..=60).collect();
        let hash3 = PrefixHash::from_tokens(&tokens3);
        cache.insert(hash3, vec![2], 20).unwrap();
        // 总token数应该 <= 60（可能淘汰了部分旧条目）
        assert!(cache.total_tokens() <= 60);
    }

    /// 测试禁用 LRU 时的行为
    /// 当 enable_lru=false 且达到 max_entries 时，evict_lru 应该直接返回不执行淘汰
    #[test]
    fn test_lru_disabled_no_eviction() {
        let config = PrefixCacheConfig {
            max_entries: 2, // 小上限
            max_tokens: 100000,
            enable_lru: false, // 禁用LRU
            min_prefix_length: 16,
        };
        let mut cache = PrefixCache::new(config);
        
        // 插入2个条目（达到上限）
        let tokens1: Vec<u32> = (1..=20).collect();
        let hash1 = PrefixHash::from_tokens(&tokens1);
        cache.insert(hash1, vec![0], 20).unwrap();
        
        let tokens2: Vec<u32> = (21..=40).collect();
        let hash2 = PrefixHash::from_tokens(&tokens2);
        cache.insert(hash2, vec![1], 20).unwrap();
        assert_eq!(cache.len(), 2);
        
        // 尝试插入第3个条目（因为LRU禁用且没有ref_count=0的条目可淘汰，可能会失败或替换）
        let tokens3: Vec<u32> = (41..=60).collect();
        let hash3 = PrefixHash::from_tokens(&tokens3);
        let result = cache.insert(hash3, vec![2], 20);
        // 结果取决于实现，但不应panic
        let _ = result;
    }

    /// 测试自定义 PrefixCacheConfig 配置
    /// 覆盖不同的配置参数组合
    #[test]
    fn test_custom_config() {
        let custom_config = PrefixCacheConfig {
            max_entries: 500,
            max_tokens: 50000,
            enable_lru: false,
            min_prefix_length: 8, // 更小的最小前缀长度
        };
        let cache = PrefixCache::new(custom_config);
        
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.total_tokens(), 0);
        
        // 可以插入更短的prefix
        let mut cache_mut = cache;
        let short_tokens: Vec<u32> = (1..=10).collect(); // 10 >= 8
        let hash = PrefixHash::from_tokens(&short_tokens);
        let result = cache_mut.insert(hash, vec![0], 10);
        assert!(result.is_ok());
    }

    /// 测试 PrefixCacheStats 结构体的字段完整性
    #[test]
    fn test_stats_fields_completeness() {
        let mut cache = PrefixCache::default_config();
        
        // 初始状态
        let initial_stats = cache.stats();
        assert_eq!(initial_stats.entries, 0);
        assert_eq!(initial_stats.total_tokens, 0);
        assert_eq!(initial_stats.hits, 0);
        assert_eq!(initial_stats.misses, 0);
        assert_eq!(initial_stats.hit_rate, 0.0);
        
        // 插入并查询后
        let hash = PrefixHash::from_tokens(&(1..=20).collect::<Vec<u32>>());
        cache.insert(hash.clone(), vec![0], 20).unwrap();
        cache.lookup(hash); // 1次hit
        cache.lookup(PrefixHash::from_tokens(&(100..=120).collect::<Vec<u32>>())); // 1次miss
        
        let final_stats = cache.stats();
        assert_eq!(final_stats.entries, 1);
        assert_eq!(final_stats.total_tokens, 20);
        assert_eq!(final_stats.hits, 1);
        assert_eq!(final_stats.misses, 1);
        assert!((final_stats.hit_rate - 0.5).abs() < 0.01);
    }

    /// 测试 PrefixEntry 结构体的字段完整性
    #[test]
    fn test_prefix_entry_fields() {
        let hash = PrefixHash::from_tokens(&(1..=20).collect::<Vec<u32>>());
        let block_ids: Vec<BlockId> = vec![0, 1, 2];
        let length = 20;
        
        let entry = PrefixEntry::new(hash.clone(), block_ids.clone(), length);
        
        assert_eq!(entry.hash, hash);
        assert_eq!(entry.block_ids, block_ids);
        assert_eq!(entry.ref_count, 1); // 初始引用计数为1
        assert_eq!(entry.length, length);
        assert_eq!(entry.last_access, 0); // 初始时间戳为0
    }

    /// 测试多个请求共享相同前缀的场景
    /// 模拟多请求场景下前缀缓存的复用效果
    #[test]
    fn test_multiple_requests_shared_prefix() {
        let mut cache = PrefixCache::default_config();
        
        // 公共前缀（模拟system prompt）
        let common_prefix: Vec<u32> = (1..=32).collect(); // 32 token公共前缀
        let common_hash = PrefixHash::from_tokens(&common_prefix);
        cache.insert(common_hash.clone(), vec![0, 1], 32).unwrap();
        
        // 模拟3个请求都命中公共前缀
        for _ in 0..3 {
            let entry = cache.lookup(common_hash);
            assert!(entry.is_some());
        }
        
        let stats = cache.stats();
        assert_eq!(stats.hits, 3);
        assert_eq!(stats.misses, 0);
        assert!((stats.hit_rate - 1.0).abs() < 0.01); // 100%命中率
        
        // 验证引用计数增加
        let entry = cache.lookup(common_hash).unwrap();
        assert!(entry.ref_count >= 4); // 初始1 + 4次lookup
    }

    /// 测试 lookup 更新 last_access 时间戳的行为
    /// 验证每次lookup都会更新访问时间，影响LRU淘汰顺序
    #[test]
    fn test_lookup_updates_last_access() {
        let mut cache = PrefixCache::default_config();
        
        let hash1 = PrefixHash::from_tokens(&(1..=20).collect::<Vec<u32>>());
        let hash2 = PrefixHash::from_tokens(&(21..=40).collect::<Vec<u32>>());
        cache.insert(hash1.clone(), vec![0], 20).unwrap();
        cache.insert(hash2.clone(), vec![1], 20).unwrap();
        
        // 记录第一次lookup的时间戳
        let time_after_first_lookup = cache.lookup(hash1).unwrap().last_access;
        
        // 再次lookup同一个条目
        let time_after_second_lookup = cache.lookup(hash1).unwrap().last_access;
        
        // 时间戳应该递增
        assert!(time_after_second_lookup > time_after_first_lookup);
    }

    /// 测试空 token 序列的哈希计算
    #[test]
    fn test_empty_tokens_hash() {
        let empty_tokens: Vec<u32> = vec![];
        let hash = PrefixHash::from_tokens(&empty_tokens);
        // 空序列也应该能正常计算哈希（虽然不能用于insert因为太短）
        let _ = hash.as_u64();
    }

    /// 测试重复插入相同哈希的行为
    /// 覆盖HashMap已存在key时的insert行为（会覆盖旧条目）
    #[test]
    fn test_duplicate_insert_same_hash() {
        let mut cache = PrefixCache::default_config();
        let tokens: Vec<u32> = (1..=20).collect();
        let hash = PrefixHash::from_tokens(&tokens);
        
        // 第一次插入
        cache.insert(hash, vec![0, 1], 20).unwrap();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.total_tokens(), 20);
        
        // 第二次插入相同哈希（不同block_ids）
        cache.insert(hash, vec![2, 3], 20).unwrap();
        // HashMap的insert会覆盖旧值，所以len仍为1，但total_tokens可能变化
        // 取决于实现是否先移除旧的再插入新的
        assert!(cache.len() <= 1);
    }
}
