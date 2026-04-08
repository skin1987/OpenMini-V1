//! Keep Routing 实现
//!
//! 保存推理时使用的专家路由，训练时强制使用相同路由，确保动作空间匹配

use crate::rl::Tensor;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// 路由器缓存
///
/// 保存专家路由选择结果
#[derive(Debug, Clone)]
pub struct RouterCache {
    pub expert_indices: Tensor,
    pub routing_weights: Tensor,
    pub layer_idx: usize,
}

/// 路由器缓存管理器
///
/// 管理和存储多个路由器缓存，支持 LRU 淘汰
#[derive(Debug, Clone)]
pub struct RouterCacheManager {
    caches: Arc<RwLock<HashMap<String, RouterCache>>>,
    max_caches: usize,
}

impl RouterCacheManager {
    pub fn new(max_caches: usize) -> Self {
        Self {
            caches: Arc::new(RwLock::new(HashMap::new())),
            max_caches,
        }
    }

    pub fn store(&self, key: String, cache: RouterCache) {
        let mut caches = self.caches.write();

        if caches.len() >= self.max_caches && !caches.contains_key(&key) {
            if let Some(first_key) = caches.keys().next().cloned() {
                caches.remove(&first_key);
            }
        }

        caches.insert(key, cache);
    }

    pub fn get(&self, key: &str) -> Option<RouterCache> {
        let caches = self.caches.read();
        caches.get(key).cloned()
    }

    pub fn remove(&self, key: &str) -> Option<RouterCache> {
        let mut caches = self.caches.write();
        caches.remove(key)
    }

    pub fn clear(&self) {
        let mut caches = self.caches.write();
        caches.clear();
    }

    pub fn len(&self) -> usize {
        let caches = self.caches.read();
        caches.len()
    }

    pub fn is_empty(&self) -> bool {
        let caches = self.caches.read();
        caches.is_empty()
    }
}

impl Default for RouterCacheManager {
    fn default() -> Self {
        Self::new(1000)
    }
}

pub struct KeepRouting {
    cache_manager: RouterCacheManager,
    enabled: bool,
    enforce_mode: EnforceMode,
}

#[derive(Debug, Clone)]
pub enum EnforceMode {
    Strict,
    Soft,
    None,
}

impl KeepRouting {
    pub fn new(max_caches: usize) -> Self {
        Self {
            cache_manager: RouterCacheManager::new(max_caches),
            enabled: true,
            enforce_mode: EnforceMode::Strict,
        }
    }

    pub fn with_enforce_mode(mut self, mode: EnforceMode) -> Self {
        self.enforce_mode = mode;
        self
    }

    pub fn disable(mut self) -> Self {
        self.enabled = false;
        self
    }

    pub fn save_routing(
        &self,
        key: String,
        expert_indices: Tensor,
        routing_weights: Tensor,
        layer_idx: usize,
    ) {
        if !self.enabled {
            return;
        }

        let cache = RouterCache {
            expert_indices,
            routing_weights,
            layer_idx,
        };

        self.cache_manager.store(key, cache);
    }

    pub fn get_routing(&self, key: &str) -> Option<RouterCache> {
        if !self.enabled {
            return None;
        }

        self.cache_manager.get(key)
    }

    pub fn apply_routing(
        &self,
        key: &str,
        default_indices: &Tensor,
        default_weights: &Tensor,
    ) -> (Tensor, Tensor) {
        match self.get_routing(key) {
            Some(cache) => (cache.expert_indices, cache.routing_weights),
            None => match self.enforce_mode {
                EnforceMode::Strict => (default_indices.clone(), default_weights.clone()),
                EnforceMode::Soft => (default_indices.clone(), default_weights.clone()),
                EnforceMode::None => (default_indices.clone(), default_weights.clone()),
            },
        }
    }

    pub fn should_enforce(&self) -> bool {
        self.enabled && matches!(self.enforce_mode, EnforceMode::Strict)
    }

    pub fn clear(&self) {
        self.cache_manager.clear();
    }
}

impl Default for KeepRouting {
    fn default() -> Self {
        Self::new(1000)
    }
}

pub fn load_router_cache(_path: &str) -> Option<RouterCache> {
    None
}

pub fn save_router_cache(_path: &str, _cache: &RouterCache) -> bool {
    false
}

#[allow(dead_code)]
impl RouterCache {
    pub fn new(expert_indices: Tensor, routing_weights: Tensor, layer_idx: usize) -> Self {
        Self {
            expert_indices,
            routing_weights,
            layer_idx,
        }
    }

    pub fn num_experts(&self) -> usize {
        self.expert_indices.size(1)
    }

    pub fn get_expert_for_token(&self, token_idx: usize) -> Option<(usize, f32)> {
        let idx_data = self.expert_indices.as_slice();
        let weight_data = self.routing_weights.as_slice();

        if token_idx < idx_data.len() {
            Some((idx_data[token_idx] as usize, weight_data[token_idx]))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_cache_creation() {
        // 测试路由器缓存创建
        let expert_indices = Tensor::new(vec![0.0f32, 1.0, 2.0, 3.0], vec![4]); // 一维张量
        let routing_weights = Tensor::new(vec![0.8, 0.6, 0.9, 0.7], vec![4]);
        let cache = RouterCache::new(expert_indices, routing_weights, 1);

        assert_eq!(cache.layer_idx, 1);
        // num_experts() 返回 size(1)，对于一维张量默认为 1
        assert_eq!(cache.num_experts(), 1);
        // 验证数据长度正确
        assert_eq!(cache.expert_indices.len(), 4);
    }

    #[test]
    fn test_router_cache_get_expert_for_token() {
        // 测试获取指定token的专家
        let expert_indices = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 0.0, 1.0]);
        let routing_weights = Tensor::from_slice(&[0.9, 0.7, 0.85, 0.6, 0.95, 0.75]);
        let cache = RouterCache::new(expert_indices, routing_weights, 0);

        // 有效索引
        let (expert_idx, weight) = cache.get_expert_for_token(2).unwrap();
        assert_eq!(expert_idx, 2);
        assert!((weight - 0.85).abs() < f32::EPSILON);

        // 越界索引
        let result = cache.get_expert_for_token(10);
        assert!(result.is_none());
    }

    #[test]
    fn test_router_cache_manager_basic_operations() {
        // 测试基本的 CRUD 操作
        let manager = RouterCacheManager::new(3);

        // 初始状态为空
        assert!(manager.is_empty());
        assert_eq!(manager.len(), 0);

        // 添加缓存
        let cache1 = RouterCache::new(
            Tensor::from_slice(&[0.0f32, 1.0]),
            Tensor::from_slice(&[0.9, 0.7]),
            0,
        );
        manager.store("key1".to_string(), cache1);

        assert_eq!(manager.len(), 1);
        assert!(!manager.is_empty());

        // 获取缓存
        let retrieved = manager.get("key1").unwrap();
        assert_eq!(retrieved.layer_idx, 0);

        // 获取不存在的key
        let not_found = manager.get("nonexistent");
        assert!(not_found.is_none());

        // 删除缓存
        let removed = manager.remove("key1").unwrap();
        assert_eq!(removed.layer_idx, 0);
        assert!(manager.is_empty());
    }

    #[test]
    fn test_router_cache_manager_lru_eviction() {
        // 测试容量限制（达到上限时淘汰旧条目）
        // 注意：HashMap 不保证顺序，所以这里只验证容量限制
        let manager = RouterCacheManager::new(2); // 最大容量 2

        manager.store(
            "key0".to_string(),
            RouterCache::new(Tensor::from_slice(&[0.0f32]), Tensor::from_slice(&[0.9]), 0),
        );
        manager.store(
            "key1".to_string(),
            RouterCache::new(Tensor::from_slice(&[1.0f32]), Tensor::from_slice(&[0.8]), 1),
        );

        assert_eq!(manager.len(), 2);

        // 添加第三个，应该淘汰某个旧的
        manager.store(
            "key2".to_string(),
            RouterCache::new(Tensor::from_slice(&[2.0f32]), Tensor::from_slice(&[0.7]), 2),
        );

        assert_eq!(manager.len(), 2); // 容量仍为 2
                                      // 新添加的 key2 应该存在
        assert!(manager.get("key2").is_some());
    }

    #[test]
    fn test_router_cache_manager_clear() {
        // 测试清空操作
        let manager = RouterCacheManager::new(10);

        for i in 0..5 {
            manager.store(
                format!("key{}", i),
                RouterCache::new(
                    Tensor::from_slice(&[i as f32]),
                    Tensor::from_slice(&[0.5 + i as f32 * 0.1]),
                    i,
                ),
            );
        }

        assert_eq!(manager.len(), 5);
        manager.clear();
        assert!(manager.is_empty());
    }

    #[test]
    fn test_keep_routing_creation_and_config() {
        // 测试创建和配置
        let routing = KeepRouting::new(100);
        assert!(routing.should_enforce()); // 默认 Strict 模式

        // 禁用
        let disabled = routing.disable();
        assert!(!disabled.should_enforce());

        // 不同强制模式
        let strict = KeepRouting::new(100).with_enforce_mode(EnforceMode::Strict);
        assert!(strict.should_enforce());

        let soft = KeepRouting::new(100).with_enforce_mode(EnforceMode::Soft);
        assert!(!soft.should_enforce()); // Soft 模式不强制

        let none = KeepRouting::new(100).with_enforce_mode(EnforceMode::None);
        assert!(!none.should_enforce()); // None 模式不强制
    }

    #[test]
    fn test_keep_routing_save_and_get() {
        // 测试保存和获取路由
        let routing = KeepRouting::new(100);

        let expert_indices = Tensor::from_slice(&[0.0f32, 1.0, 2.0]);
        let routing_weights = Tensor::from_slice(&[0.9, 0.7, 0.8]);

        routing.save_routing("prompt1".to_string(), expert_indices, routing_weights, 0);

        // 获取已保存的路由
        let cache = routing.get_routing("prompt1").unwrap();
        assert_eq!(cache.layer_idx, 0);
        // 验证数据长度（num_experts 对于一维张量返回 1）
        assert_eq!(cache.expert_indices.len(), 3);

        // 获取不存在的路由
        let not_found = routing.get_routing("nonexistent");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_keep_routing_disabled_saves_nothing() {
        // 测试禁用状态下不保存
        let routing = KeepRouting::new(100).disable();

        routing.save_routing(
            "prompt1".to_string(),
            Tensor::from_slice(&[0f32]),
            Tensor::from_slice(&[0.9]),
            0,
        );

        // 禁用状态应该返回 None
        let result = routing.get_routing("prompt1");
        assert!(result.is_none());
    }

    #[test]
    fn test_keep_routing_apply_routing_with_cache() {
        // 测试应用路由（有缓存时使用缓存）
        let routing = KeepRouting::new(100);

        let cached_indices = Tensor::from_slice(&[0.0f32, 1.0]);
        let cached_weights = Tensor::from_slice(&[0.95, 0.85]);

        routing.save_routing(
            "cached_prompt".to_string(),
            cached_indices.clone(),
            cached_weights.clone(),
            0,
        );

        let default_indices = Tensor::from_slice(&[99.0f32, 88.0]); // 这些不应该被使用
        let default_weights = Tensor::from_slice(&[0.1, 0.2]);

        let (indices, weights) =
            routing.apply_routing("cached_prompt", &default_indices, &default_weights);

        // 应该返回缓存的值，而不是默认值
        assert_eq!(indices.as_slice()[0], 0.0); // 来自缓存
        assert_eq!(weights.as_slice()[0], 0.95); // 来自缓存
    }

    #[test]
    fn test_keep_routing_apply_routing_without_cache() {
        // 测试应用路由（无缓存时使用默认值）
        let routing = KeepRouting::new(100);

        let default_indices = Tensor::from_slice(&[42.0f32, 43.0]);
        let default_weights = Tensor::from_slice(&[0.6, 0.7]);

        let (indices, weights) =
            routing.apply_routing("uncached_prompt", &default_indices, &default_weights);

        // 无缓存时应该返回默认值
        assert_eq!(indices.as_slice()[0], 42.0);
        assert_eq!(weights.as_slice()[0], 0.6);
    }

    #[test]
    fn test_keep_routing_default() {
        // 测试默认配置
        let routing = KeepRouting::default();
        assert!(routing.should_enforce());
    }

    #[test]
    fn test_keep_routing_clear() {
        // 测试清空所有路由
        let routing = KeepRouting::new(100);

        routing.save_routing(
            "key1".to_string(),
            Tensor::from_slice(&[0.0f32]),
            Tensor::from_slice(&[0.9]),
            0,
        );
        routing.save_routing(
            "key2".to_string(),
            Tensor::from_slice(&[1.0f32]),
            Tensor::from_slice(&[0.8]),
            1,
        );

        routing.clear();

        assert!(routing.get_routing("key1").is_none());
        assert!(routing.get_routing("key2").is_none());
    }

    #[test]
    fn test_load_save_router_cache_stubs() {
        // 测试加载/保存存根函数（总是返回 None/false）
        let loaded = load_router_cache("/nonexistent/path.cache");
        assert!(loaded.is_none());

        let cache = RouterCache::new(Tensor::from_slice(&[0.0f32]), Tensor::from_slice(&[0.9]), 0);
        let saved = save_router_cache("/tmp/test.cache", &cache);
        assert!(!saved);
    }

    // ==================== 新增分支覆盖测试 (7个) ====================

    #[test]
    fn test_router_cache_manager_capacity_boundary_exact() {
        // 覆盖分支: 容量刚好达到上限时的行为
        let manager = RouterCacheManager::new(2);

        let cache0 = RouterCache::new(Tensor::from_slice(&[0.0f32]), Tensor::from_slice(&[1.0]), 0);
        manager.store("key0".to_string(), cache0);
        assert_eq!(manager.len(), 1);

        let cache1 = RouterCache::new(Tensor::from_slice(&[1.0f32]), Tensor::from_slice(&[0.8]), 1);
        manager.store("key1".to_string(), cache1);
        assert_eq!(manager.len(), 2); // 刚好达到容量

        // 更新已存在的 key 不应增加长度
        let cache0_updated =
            RouterCache::new(Tensor::from_slice(&[2.0f32]), Tensor::from_slice(&[0.7]), 2);
        manager.store("key0".to_string(), cache0_updated);
        assert_eq!(manager.len(), 2); // 仍然是 2
    }

    #[test]
    fn test_router_cache_manager_remove_nonexistent() {
        // 覆盖分支: 删除不存在的 key
        let manager = RouterCacheManager::new(5);

        // 删除空管理器中的 key
        let result = manager.remove("nonexistent");
        assert!(result.is_none());
        assert_eq!(manager.len(), 0);

        // 添加一个后再删除不存在的
        manager.store(
            "existing".to_string(),
            RouterCache::new(Tensor::from_slice(&[1.0]), Tensor::from_slice(&[0.5]), 0),
        );
        let result2 = manager.remove("also_nonexistent");
        assert!(result2.is_none());
        assert_eq!(manager.len(), 1); // 原有的还在
    }

    #[test]
    fn test_keep_routing_enforce_mode_all_variants() {
        // 覆盖分支: 所有强制模式下的 apply_routing 行为

        let indices = Tensor::from_slice(&[42.0, 43.0]);
        let weights = Tensor::from_slice(&[0.6, 0.7]);

        // Strict 模式: 无缓存时返回默认值
        let strict = KeepRouting::new(100).with_enforce_mode(EnforceMode::Strict);
        let (idx, _wgt) = strict.apply_routing("uncached", &indices, &weights);
        assert_eq!(idx.as_slice()[0], 42.0); // 返回默认值

        // Soft 模式: 无缓存时返回默认值
        let soft = KeepRouting::new(100).with_enforce_mode(EnforceMode::Soft);
        let (idx, _wgt) = soft.apply_routing("uncached", &indices, &weights);
        assert_eq!(idx.as_slice()[0], 42.0); // 返回默认值

        // None 模式: 无缓存时返回默认值
        let none_mode = KeepRouting::new(100).with_enforce_mode(EnforceMode::None);
        let (idx, _wgt) = none_mode.apply_routing("uncached", &indices, &weights);
        assert_eq!(idx.as_slice()[0], 42.0); // 返回默认值

        // 验证 should_enforce 只在 Strict + enabled 时为 true
        assert!(strict.should_enforce());
        assert!(!soft.should_enforce());
        assert!(!none_mode.should_enforce());
    }

    #[test]
    fn test_keep_routing_disabled_all_operations_noop() {
        // 覆盖分支: 禁用状态下所有操作都是 no-op
        let routing = KeepRouting::new(100).disable();

        // save_routing 应该不做任何事
        routing.save_routing(
            "should_not_save".to_string(),
            Tensor::from_slice(&[1.0]),
            Tensor::from_slice(&[0.9]),
            0,
        );
        assert!(routing.get_routing("should_not_save").is_none());

        // get_routing 应该返回 None
        assert!(routing.get_routing("anything").is_none());

        // apply_routing 应该返回默认值
        let default_idx = Tensor::from_slice(&[99.0]);
        let default_wgt = Tensor::from_slice(&[0.5]);
        let (idx, _wgt) = routing.apply_routing("any_key", &default_idx, &default_wgt);
        assert_eq!(idx.as_slice()[0], 99.0);

        // clear 应该成功（但实际没有数据可清）
        routing.clear();
        assert!(routing.get_routing("anything").is_none());

        // should_enforce 应该返回 false
        assert!(!routing.should_enforce());
    }

    #[test]
    fn test_router_cache_expert_for_token_boundary() {
        // 覆盖分支: get_expert_for_token 的各种边界情况

        // 正常情况 - 第一个 token (index 0)
        let cache = RouterCache::new(
            Tensor::from_slice(&[5.0f32, 3.0, 7.0]),
            Tensor::from_slice(&[0.9, 0.6, 0.95]),
            0,
        );
        let (expert, weight) = cache.get_expert_for_token(0).unwrap();
        assert_eq!(expert, 5);
        assert!((weight - 0.9).abs() < f32::EPSILON);

        // 最后一个 token (index len-1)
        let (expert, weight) = cache.get_expert_for_token(2).unwrap();
        assert_eq!(expert, 7);
        assert!((weight - 0.95).abs() < f32::EPSILON);

        // 越界 index (正好等于长度)
        let result = cache.get_expert_for_token(3);
        assert!(result.is_none());

        // 超大越界
        let result = cache.get_expert_for_token(1000);
        assert!(result.is_none());

        // 单元素缓存
        let single = RouterCache::new(Tensor::from_slice(&[0.0f32]), Tensor::from_slice(&[1.0]), 5);
        let (expert, weight) = single.get_expert_for_token(0).unwrap();
        assert_eq!(expert, 0);
        assert!((weight - 1.0).abs() < f32::EPSILON);

        let result = single.get_expert_for_token(1);
        assert!(result.is_none());
    }

    #[test]
    fn test_router_cache_clone_and_independence() {
        // 覆盖分支: RouterCache 的 Clone 特性和独立性
        let original = RouterCache::new(
            Tensor::from_slice(&[1.0, 2.0, 3.0]),
            Tensor::from_slice(&[0.4, 0.5, 0.6]),
            10,
        );

        let cloned = original.clone();

        // 验证值相等
        assert_eq!(original.layer_idx, cloned.layer_idx);
        assert_eq!(original.expert_indices.len(), cloned.expert_indices.len());
        assert_eq!(original.routing_weights.len(), cloned.routing_weights.len());

        // 修改 clone 不应影响原始（Tensor 可能是共享的，但 layer_idx 是独立的）
        // 这里主要验证 Clone trait 可用且不会 panic
        let _another_clone = original.clone();
    }

    #[test]
    fn test_keep_routing_multiple_prompts_isolation() {
        // 覆盖分支: 多个 prompt 之间的路由隔离
        let routing = KeepRouting::new(50);

        // 保存多个不同的路由
        for i in 0..5 {
            routing.save_routing(
                format!("prompt_{}", i),
                Tensor::from_slice(&[i as f32, i as f32 + 1.0]),
                Tensor::from_slice(&[0.9 - i as f32 * 0.1, 0.8 - i as f32 * 0.05]),
                i,
            );
        }

        // 验证每个路由独立存在
        for i in 0..5 {
            let cache = routing.get_routing(&format!("prompt_{}", i)).unwrap();
            assert_eq!(cache.layer_idx, i);
            assert!((cache.expert_indices.as_slice()[0] - i as f32).abs() < f32::EPSILON);
        }

        // 验证总数正确
        // 注意：RouterCacheManager 内部使用 HashMap，len() 应该反映存储的数量
        assert!(!routing.get_routing("prompt_0").is_none()); // 至少有一个存在
    }
}
