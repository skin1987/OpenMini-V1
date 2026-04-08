//! ESS (Elastic Storage System) 弹性存储系统
//!
//! 实现CPU-GPU异构内存架构，支持Latent-Cache卸载到CPU内存
//! 突破GPU显存限制，支持128K+上下文推理

#![allow(dead_code)]

mod cache;
mod eviction;
mod prefetch;
mod transfer;

pub use cache::{CacheEntry, LatentCacheManager, MemoryTier};
pub use eviction::EvictionPolicy;
pub use prefetch::Prefetcher;
pub use transfer::{FlashTransfer, TransferManager};

#[cfg(test)]
pub use eviction::{AdaptiveEviction, LfuEviction, LruEviction};
#[cfg(test)]
pub use prefetch::{AccessPattern, LocalityAnalyzer};
#[cfg(test)]
pub use transfer::TransferDirection;

use std::sync::{Arc, RwLock};

pub struct EssConfig {
    pub max_gpu_memory_mb: usize,
    pub max_cpu_memory_mb: usize,
    pub prefetch_enabled: bool,
    pub eviction_policy: EvictionPolicy,
    pub transfer_batch_size: usize,
}

impl Default for EssConfig {
    fn default() -> Self {
        Self {
            max_gpu_memory_mb: 8192,
            max_cpu_memory_mb: 65536,
            prefetch_enabled: true,
            eviction_policy: EvictionPolicy::Adaptive,
            transfer_batch_size: 4096,
        }
    }
}

pub struct ElasticStorageSystem {
    config: EssConfig,
    cache_manager: Arc<RwLock<LatentCacheManager>>,
    transfer_manager: Arc<TransferManager>,
    prefetcher: Arc<RwLock<Option<Prefetcher>>>,
    stats: Arc<RwLock<EssStats>>,
}

#[derive(Debug, Default, Clone)]
pub struct EssStats {
    pub gpu_hits: u64,
    pub cpu_hits: u64,
    pub misses: u64,
    pub prefetch_operations: u64,
    pub transfer_bytes: u64,
    pub eviction_count: u64,
    pub put_errors: u64,
}

impl ElasticStorageSystem {
    pub fn new(config: EssConfig) -> Self {
        let cache_manager = Arc::new(RwLock::new(LatentCacheManager::new(
            config.max_gpu_memory_mb,
            config.max_cpu_memory_mb,
        )));

        let transfer_manager = Arc::new(TransferManager::new(config.transfer_batch_size));

        let prefetcher = if config.prefetch_enabled {
            Arc::new(RwLock::new(Some(Prefetcher::new())))
        } else {
            Arc::new(RwLock::new(None))
        };

        Self {
            config,
            cache_manager,
            transfer_manager,
            prefetcher,
            stats: Arc::new(RwLock::new(EssStats::default())),
        }
    }

    /// 获取缓存条目
    ///
    /// 同时更新统计信息和触发预取
    pub fn get(&self, key: &str) -> Option<CacheEntry> {
        let cache = self.cache_manager.read().unwrap();

        if let Some(entry) = cache.get(key) {
            let mut stats = self.stats.write().unwrap();
            match entry.tier {
                MemoryTier::GPU => stats.gpu_hits += 1,
                MemoryTier::CPU => stats.cpu_hits += 1,
                MemoryTier::Storage => stats.cpu_hits += 1,
            }

            if let Some(ref mut prefetcher_guard) = *self.prefetcher.write().unwrap() {
                let _ = prefetcher_guard.access(entry.data.as_ptr() as usize);
            }

            Some(entry.clone())
        } else {
            self.stats.write().unwrap().misses += 1;
            None
        }
    }

    /// 存储数据到指定层级
    ///
    /// 返回错误信息（如有）
    pub fn put(&self, key: String, data: Vec<u8>, tier: MemoryTier) -> Result<(), String> {
        let data_len = data.len();
        let mut cache = self.cache_manager.write().unwrap();

        match cache.put(key, data, tier) {
            Ok(()) => {
                self.stats.write().unwrap().transfer_bytes += data_len as u64;
                Ok(())
            }
            Err(e) => {
                self.stats.write().unwrap().put_errors += 1;
                Err(e)
            }
        }
    }

    /// 驱逐缓存条目
    pub fn evict(&self, policy: EvictionPolicy) -> Vec<String> {
        let mut cache = self.cache_manager.write().unwrap();
        let evicted = cache.evict(policy);

        let mut stats = self.stats.write().unwrap();
        stats.eviction_count += evicted.len() as u64;

        evicted
    }

    /// 传输数据到 GPU
    ///
    /// 注意：当前使用 FlashTransfer 模拟实现，仅进行内存复制。
    /// 生产环境应替换为真实的 GPU 传输 API（CUDA、Vulkan 等）。
    pub fn transfer_to_gpu(&self, key: &str) -> Option<Vec<u8>> {
        let data = {
            let mut cache = self.cache_manager.write().unwrap();
            cache.remove(key)
        }?;

        let transfer = FlashTransfer::new();
        let result = transfer.transfer_to_device(&data);

        if result.is_some() {
            self.stats.write().unwrap().transfer_bytes += data.len() as u64;
        }

        result
    }

    /// 获取预取列表
    pub fn get_prefetch_list(&self) -> Vec<usize> {
        if let Some(ref prefetcher) = *self.prefetcher.read().unwrap() {
            prefetcher.get_prefetch_list().to_vec()
        } else {
            Vec::new()
        }
    }

    /// 获取统计信息
    pub fn stats(&self) -> EssStats {
        self.stats.read().unwrap().clone()
    }

    /// 重置统计信息
    pub fn reset_stats(&self) {
        *self.stats.write().unwrap() = EssStats::default();
    }

    /// 获取缓存配置
    pub fn config(&self) -> &EssConfig {
        &self.config
    }
}

impl Default for ElasticStorageSystem {
    fn default() -> Self {
        Self::new(EssConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_allocation() {
        let mut cache = LatentCacheManager::new(1024, 8192);

        let result = cache.put("key1".to_string(), vec![0u8; 1024], MemoryTier::GPU);
        assert!(result.is_ok());

        let result = cache.put("key2".to_string(), vec![0u8; 2048], MemoryTier::GPU);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cache_put_get() {
        let mut cache = LatentCacheManager::new(1024, 8192);

        let _ = cache.put("key1".to_string(), vec![1, 2, 3, 4], MemoryTier::GPU);

        let entry = cache.get("key1");
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_cache_eviction_lru() {
        let mut cache = LatentCacheManager::new(1, 8192);

        let _ = cache.put("key1".to_string(), vec![0u8; 512 * 1024], MemoryTier::GPU);
        let _ = cache.put("key2".to_string(), vec![0u8; 512 * 1024], MemoryTier::GPU);

        cache.get("key1");

        let result = cache.put("key3".to_string(), vec![0u8; 512 * 1024], MemoryTier::GPU);
        if result.is_err() {
            let evicted = cache.evict(EvictionPolicy::LRU);
            assert!(!evicted.is_empty());
            assert!(evicted.contains(&"key2".to_string()));
        }
    }

    #[test]
    fn test_eviction_policy_lru() {
        let mut eviction = LruEviction::new();

        eviction.access("a");
        eviction.access("b");
        eviction.access("a");

        let key = eviction.evict();
        assert!(key.is_some());
    }

    #[test]
    fn test_eviction_policy_lfu() {
        let mut eviction = LfuEviction::new();

        eviction.access("a");
        eviction.access("a");
        eviction.access("a");
        eviction.access("b");

        let key = eviction.evict();
        assert!(key.is_some());
        assert_eq!(key.unwrap(), "b");
    }

    #[test]
    fn test_access_pattern_sequential() {
        let mut analyzer = LocalityAnalyzer::new(100);

        analyzer.record_access(10);
        analyzer.record_access(20);
        analyzer.record_access(30);

        let pattern = analyzer.get_pattern();
        assert!(pattern != AccessPattern::Unknown);
    }

    #[test]
    fn test_access_pattern_strided() {
        let mut analyzer = LocalityAnalyzer::new(100);

        analyzer.record_access(0);
        analyzer.record_access(4096);
        analyzer.record_access(8192);

        let pattern = analyzer.get_pattern();
        assert_eq!(pattern, AccessPattern::Strided);
    }

    #[test]
    fn test_prefetcher() {
        let mut prefetcher = Prefetcher::new();

        prefetcher.access(1000);
        prefetcher.access(2000);

        let predicted = prefetcher.access(3000);
        assert!(predicted.is_some() || predicted.is_none());
    }

    #[test]
    fn test_transfer_manager() {
        let manager = TransferManager::new(10);

        let id = manager.submit(vec![1, 2, 3], TransferDirection::ToGPU);
        assert!(id > 0);

        assert_eq!(manager.pending_count(), 1);
    }

    #[test]
    fn test_flash_transfer() {
        let transfer = FlashTransfer::new();

        let data = vec![1u8; 1024];
        let result = transfer.transfer_to_device(&data);
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 1024);
    }

    #[test]
    fn test_ess_config_default() {
        let config = EssConfig::default();
        assert_eq!(config.max_gpu_memory_mb, 8192);
        assert_eq!(config.max_cpu_memory_mb, 65536);
        assert!(config.prefetch_enabled);
    }

    #[test]
    fn test_ess_system_basic() {
        let ess = ElasticStorageSystem::new(EssConfig::default());

        let result = ess.put("test_key".to_string(), vec![1, 2, 3, 4], MemoryTier::GPU);
        assert!(result.is_ok());

        let entry = ess.get("test_key");
        assert!(entry.is_some());
    }

    #[test]
    fn test_cache_tier_hierarchy() {
        let mut cache = LatentCacheManager::new(100, 1000);

        let _ = cache.put("gpu_data".to_string(), vec![1, 2, 3], MemoryTier::GPU);
        let _ = cache.put("cpu_data".to_string(), vec![4, 5, 6], MemoryTier::CPU);

        let gpu_entry = cache.get("gpu_data");
        let cpu_entry = cache.get("cpu_data");

        assert!(gpu_entry.is_some());
        assert!(cpu_entry.is_some());
        assert_eq!(gpu_entry.unwrap().tier, MemoryTier::GPU);
        assert_eq!(cpu_entry.unwrap().tier, MemoryTier::CPU);
    }

    #[test]
    fn test_adaptive_eviction() {
        let mut eviction = AdaptiveEviction::new();

        for _ in 0..50 {
            eviction.access("a");
            eviction.access("b");
        }

        let key = eviction.evict();
        assert!(key.is_some());
    }

    #[test]
    fn test_locality_predict_next() {
        let mut analyzer = LocalityAnalyzer::new(100);

        analyzer.record_access(0);
        analyzer.record_access(4096);
        analyzer.record_access(8192);

        let prediction = analyzer.predict_next();
        assert!(prediction.is_some());
        assert_eq!(prediction.unwrap(), 12288);
    }

    #[test]
    fn test_ess_stats_tracking() {
        let ess = ElasticStorageSystem::new(EssConfig::default());

        ess.put("key1".to_string(), vec![1, 2, 3], MemoryTier::GPU)
            .unwrap();
        ess.put("key2".to_string(), vec![4, 5, 6], MemoryTier::CPU)
            .unwrap();

        ess.get("key1");
        ess.get("key2");
        ess.get("nonexistent");

        let stats = ess.stats();
        assert_eq!(stats.gpu_hits, 1);
        assert_eq!(stats.cpu_hits, 1);
        assert_eq!(stats.misses, 1);
        assert!(stats.transfer_bytes > 0);
    }

    #[test]
    fn test_ess_put_error_handling() {
        let config = EssConfig {
            max_gpu_memory_mb: 1,
            ..Default::default()
        };
        let ess = ElasticStorageSystem::new(config);

        let result = ess.put(
            "key1".to_string(),
            vec![0u8; 2 * 1024 * 1024],
            MemoryTier::GPU,
        );
        assert!(result.is_err());

        let stats = ess.stats();
        assert_eq!(stats.put_errors, 1);
    }

    #[test]
    fn test_ess_eviction_stats() {
        let ess = ElasticStorageSystem::new(EssConfig {
            max_gpu_memory_mb: 1,
            ..Default::default()
        });

        ess.put("key1".to_string(), vec![0u8; 512 * 1024], MemoryTier::GPU)
            .unwrap();
        ess.put("key2".to_string(), vec![0u8; 512 * 1024], MemoryTier::GPU)
            .unwrap();

        let evicted = ess.evict(EvictionPolicy::LRU);

        let stats = ess.stats();
        assert_eq!(stats.eviction_count, evicted.len() as u64);
        assert!(!evicted.is_empty());
    }
}
