//! 记忆管理器
//!
//! 统一管理三级记忆，提供读写接口和同步机制
//!
//! # 功能特性
//! - 统一检索 API：跨层级搜索和排序
//! - 跨层记忆协调：记忆提升、降级和巩固
//! - 并发访问优化：无锁读取，使用 dashmap
//! - 配置热更新：运行时配置变更
//! - 统计和监控：Prometheus 格式指标

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use ndarray::Array2;
use parking_lot::{Mutex, RwLock};

use super::simd_ops::SimdVectorOps;
#[allow(unused_imports)]
use super::{
    EvictionStrategy, InstantMemory, LongTermMemory, MemoryConfig, MemoryLevel, PaddingStrategy,
    ShortTermMemory,
};

/// 搜索结果项
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// 记忆数据
    pub data: Array2<f32>,
    /// 相似度分数
    pub score: f32,
    /// 来源层级
    pub level: MemoryLevel,
    /// 时间戳
    pub timestamp: u64,
    /// 重要性
    pub importance: f32,
}

impl SearchResult {
    /// 创建新的搜索结果
    ///
    /// # 参数
    /// - `data`: 记忆数据
    /// - `score`: 相似度分数（0.0-1.0）
    /// - `level`: 来源记忆层级
    /// - `timestamp`: 数据时间戳
    /// - `importance`: 重要性分数
    pub fn new(
        data: Array2<f32>,
        score: f32,
        level: MemoryLevel,
        timestamp: u64,
        importance: f32,
    ) -> Self {
        Self {
            data,
            score,
            level,
            timestamp,
            importance,
        }
    }
}

/// 搜索查询条件
#[derive(Debug, Clone)]
pub struct SearchQuery {
    /// 查询向量
    pub vector: Array2<f32>,
    /// 返回数量
    pub top_k: usize,
    /// 最小相似度阈值
    pub min_score: f32,
    /// 是否包含瞬时记忆
    pub include_instant: bool,
    /// 是否包含短期记忆
    pub include_short_term: bool,
    /// 是否包含长期记忆
    pub include_long_term: bool,
    /// 按重要性过滤的最小值
    pub min_importance: Option<f32>,
}

impl SearchQuery {
    /// 创建新的搜索查询
    ///
    /// 默认包含所有层级、最小相似度为 0.0
    ///
    /// # 参数
    /// - `vector`: 查询向量
    /// - `top_k`: 返回结果数量
    pub fn new(vector: Array2<f32>, top_k: usize) -> Self {
        Self {
            vector,
            top_k,
            min_score: 0.0,
            include_instant: true,
            include_short_term: true,
            include_long_term: true,
            min_importance: None,
        }
    }

    /// 设置最小相似度阈值
    ///
    /// # 参数
    /// - `min_score`: 最小相似度分数（0.0-1.0）
    ///
    /// # 返回值
    /// 修改后的 SearchQuery 实例（支持链式调用）
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = min_score;
        self
    }

    /// 设置要搜索的记忆层级
    ///
    /// # 参数
    /// - `instant`: 是否包含瞬时记忆
    /// - `short_term`: 是否包含短期记忆
    /// - `long_term`: 是否包含长期记忆
    ///
    /// # 返回值
    /// 修改后的 SearchQuery 实例（支持链式调用）
    pub fn with_levels(mut self, instant: bool, short_term: bool, long_term: bool) -> Self {
        self.include_instant = instant;
        self.include_short_term = short_term;
        self.include_long_term = long_term;
        self
    }

    /// 设置最小重要性过滤条件
    ///
    /// # 参数
    /// - `min_importance`: 最小重要性阈值
    ///
    /// # 返回值
    /// 修改后的 SearchQuery 实例（支持链式调用）
    pub fn with_min_importance(mut self, min_importance: f32) -> Self {
        self.min_importance = Some(min_importance);
        self
    }
}

/// DMN 统计信息
#[derive(Debug, Clone, Default)]
pub struct DMNStats {
    /// 瞬时记忆数量
    pub instant_count: usize,
    /// 短期记忆数量
    pub short_term_count: usize,
    /// 长期记忆数量
    pub long_term_count: usize,
    /// 总读取次数
    pub total_reads: u64,
    /// 总写入次数
    pub total_writes: u64,
    /// 总搜索次数
    pub total_searches: u64,
    /// 记忆提升次数
    pub promotions: u64,
    /// 记忆降级次数
    pub demotions: u64,
    /// 压缩次数
    pub compressions: u64,
    /// 缓存命中率
    pub cache_hit_rate: f32,
    /// 平均查询延迟（微秒）
    pub avg_query_latency_us: f64,
}

/// Prometheus 指标
#[derive(Debug, Clone, Default)]
pub struct DMNMetrics {
    /// 指标名称到值的映射
    pub metrics: HashMap<String, f64>,
}

impl DMNMetrics {
    /// 创建空的 DMN 指标实例
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }

    /// 插入指标数据
    ///
    /// # 参数
    /// - `name`: 指标名称
    /// - `value`: 指标值
    pub fn insert(&mut self, name: &str, value: f64) {
        self.metrics.insert(name.to_string(), value);
    }

    /// 转换为 Prometheus 格式
    pub fn to_prometheus_format(&self) -> String {
        let mut output = String::new();

        // 记忆计数指标
        output.push_str("# HELP dmn_memory_count Number of memories per level\n");
        output.push_str("# TYPE dmn_memory_count gauge\n");

        let level_mapping = [
            ("instant_memory_count", "instant"),
            ("short_term_memory_count", "short_term"),
            ("long_term_memory_count", "long_term"),
        ];

        for (key, level) in level_mapping.iter() {
            if let Some(value) = self.metrics.get(*key) {
                output.push_str(&format!(
                    "dmn_memory_count{{level=\"{}\"}} {}\n",
                    level, value
                ));
            }
        }

        // 缓存命中率
        output.push_str("# HELP dmn_cache_hit_rate Cache hit rate\n");
        output.push_str("# TYPE dmn_cache_hit_rate gauge\n");

        if let Some(value) = self.metrics.get("cache_hit_rate") {
            output.push_str(&format!("dmn_cache_hit_rate {}\n", value));
        }

        // 缓存大小
        output.push_str("# HELP dmn_cache_size Current cache size\n");
        output.push_str("# TYPE dmn_cache_size gauge\n");

        if let Some(value) = self.metrics.get("cache_size") {
            output.push_str(&format!("dmn_cache_size {}\n", value));
        }

        // 操作计数器
        output.push_str("# HELP dmn_operations_total Total operation counts\n");
        output.push_str("# TYPE dmn_operations_total counter\n");

        let op_mapping = [
            ("total_reads", "reads"),
            ("total_writes", "writes"),
            ("total_searches", "searches"),
            ("promotions", "promotions"),
            ("demotions", "demotions"),
            ("compressions", "compressions"),
        ];

        for (key, op) in op_mapping.iter() {
            if let Some(value) = self.metrics.get(*key) {
                output.push_str(&format!(
                    "dmn_operations_total{{operation=\"{}\"}} {}\n",
                    op, value
                ));
            }
        }

        output
    }
}

/// 配置变更回调类型
pub type ConfigChangeCallback = Box<dyn Fn(&MemoryConfig) + Send + Sync>;

/// 跨层记忆协调器
#[derive(Debug)]
pub struct MemoryCoordinator {
    /// 提升阈值（重要性）
    promotion_threshold: f32,
    /// 降级阈值（重要性）
    demotion_threshold: f32,
    /// 巩固间隔（秒）
    consolidation_interval_secs: u64,
    /// 上次巩固时间
    last_consolidation: AtomicU64,
    /// 提升计数
    promotion_count: AtomicU64,
    /// 降级计数
    demotion_count: AtomicU64,
}

impl Default for MemoryCoordinator {
    fn default() -> Self {
        Self {
            promotion_threshold: 0.7,
            demotion_threshold: 0.2,
            consolidation_interval_secs: 300,
            last_consolidation: AtomicU64::new(0),
            promotion_count: AtomicU64::new(0),
            demotion_count: AtomicU64::new(0),
        }
    }
}

impl MemoryCoordinator {
    /// 创建新的跨层记忆协调器
    ///
    /// # 参数
    /// - `promotion_threshold`: 提升到长期记忆的重要性阈值
    /// - `demotion_threshold`: 降级到短期记忆的重要性阈值
    /// - `consolidation_interval_secs`: 巩固操作的时间间隔（秒）
    pub fn new(
        promotion_threshold: f32,
        demotion_threshold: f32,
        consolidation_interval_secs: u64,
    ) -> Self {
        Self {
            promotion_threshold,
            demotion_threshold,
            consolidation_interval_secs,
            last_consolidation: AtomicU64::new(0),
            promotion_count: AtomicU64::new(0),
            demotion_count: AtomicU64::new(0),
        }
    }

    /// 检查是否需要巩固
    pub fn should_consolidate(&self) -> bool {
        let now = current_timestamp();
        let last = self.last_consolidation.load(Ordering::Relaxed);
        now.saturating_sub(last) >= self.consolidation_interval_secs
    }

    /// 标记巩固完成
    pub fn mark_consolidation(&self) {
        self.last_consolidation
            .store(current_timestamp(), Ordering::Relaxed);
    }

    /// 记录提升
    pub fn record_promotion(&self) {
        self.promotion_count.fetch_add(1, Ordering::Relaxed);
    }

    /// 记录降级
    pub fn record_demotion(&self) {
        self.demotion_count.fetch_add(1, Ordering::Relaxed);
    }

    /// 获取提升次数
    pub fn get_promotion_count(&self) -> u64 {
        self.promotion_count.load(Ordering::Relaxed)
    }

    /// 获取降级次数
    pub fn get_demotion_count(&self) -> u64 {
        self.demotion_count.load(Ordering::Relaxed)
    }

    /// 检查是否应该提升
    pub fn should_promote(&self, importance: f32) -> bool {
        importance >= self.promotion_threshold
    }

    /// 检查是否应该降级
    pub fn should_demote(&self, importance: f32) -> bool {
        importance <= self.demotion_threshold
    }
}

/// 增强版记忆管理器
pub struct MemoryManager {
    /// 瞬时记忆
    instant: Arc<InstantMemory>,
    /// 短期记忆（使用 RwLock）
    short_term: Arc<RwLock<ShortTermMemory>>,
    /// 长期记忆（使用 RwLock）
    long_term: Arc<RwLock<LongTermMemory>>,
    /// 配置
    config: RwLock<MemoryConfig>,
    /// 当前会话 ID
    current_session_id: AtomicU64,
    /// 跨层协调器
    coordinator: Arc<MemoryCoordinator>,
    /// 快速访问缓存（使用 DashMap 实现无锁读取）
    cache: DashMap<String, Array2<f32>>,
    /// 缓存容量
    cache_capacity: usize,
    /// 缓存命中次数
    cache_hits: AtomicU64,
    /// LRU 访问顺序（最久未使用在前）
    cache_lru: Mutex<VecDeque<String>>,
    /// 统计：读取次数
    read_count: AtomicU64,
    /// 统计：写入次数
    write_count: AtomicU64,
    /// 统计：搜索次数
    search_count: AtomicU64,
    /// 统计：压缩次数
    compression_count: AtomicU64,
    /// 配置变更回调
    config_callbacks: Mutex<Vec<ConfigChangeCallback>>,
    /// SIMD 向量操作实例
    simd_ops: SimdVectorOps,
}

/// 克隆说明：
///
/// - 克隆会复制缓存和 LRU 队列，但不会复制配置变更回调
/// - 克隆后的实例与原实例共享底层存储（instant、short_term、long_term）
/// - 统计计数器会复制当前值，但之后独立累加
/// - 如需完全共享状态，建议使用 `Arc<MemoryManager>` 而非克隆
impl Clone for MemoryManager {
    fn clone(&self) -> Self {
        Self {
            instant: Arc::clone(&self.instant),
            short_term: Arc::clone(&self.short_term),
            long_term: Arc::clone(&self.long_term),
            config: RwLock::new(self.config.read().clone()),
            current_session_id: AtomicU64::new(self.current_session_id.load(Ordering::Relaxed)),
            coordinator: Arc::clone(&self.coordinator),
            cache: self.cache.clone(),
            cache_capacity: self.cache_capacity,
            cache_hits: AtomicU64::new(self.cache_hits.load(Ordering::Relaxed)),
            cache_lru: Mutex::new(self.cache_lru.lock().clone()),
            read_count: AtomicU64::new(self.read_count.load(Ordering::Relaxed)),
            write_count: AtomicU64::new(self.write_count.load(Ordering::Relaxed)),
            search_count: AtomicU64::new(self.search_count.load(Ordering::Relaxed)),
            compression_count: AtomicU64::new(self.compression_count.load(Ordering::Relaxed)),
            config_callbacks: Mutex::new(Vec::new()),
            simd_ops: SimdVectorOps::new(),
        }
    }
}

impl fmt::Debug for MemoryManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryManager")
            .field("instant", &self.instant)
            .field("short_term", &"<RwLock<ShortTermMemory>>")
            .field("long_term", &"<RwLock<LongTermMemory>>")
            .field("config", &self.config.read())
            .field(
                "current_session_id",
                &self.current_session_id.load(Ordering::Relaxed),
            )
            .field("coordinator", &self.coordinator)
            .field("cache_size", &self.cache.len())
            .field("cache_capacity", &self.cache_capacity)
            .field("read_count", &self.read_count.load(Ordering::Relaxed))
            .field("write_count", &self.write_count.load(Ordering::Relaxed))
            .field("search_count", &self.search_count.load(Ordering::Relaxed))
            .field(
                "compression_count",
                &self.compression_count.load(Ordering::Relaxed),
            )
            .field(
                "config_callbacks",
                &format!("{} callbacks", self.config_callbacks.lock().len()),
            )
            .finish()
    }
}

impl MemoryManager {
    /// 创建新的记忆管理器
    ///
    /// 初始化三级记忆系统，包括缓存和统计计数器
    ///
    /// # 参数
    /// - `config`: 记忆系统配置
    pub fn new(config: MemoryConfig) -> Self {
        let cache_capacity = (config.short_term_capacity / 4).max(64);
        Self {
            instant: Arc::new(InstantMemory::new(&config)),
            short_term: Arc::new(RwLock::new(ShortTermMemory::new(&config))),
            long_term: Arc::new(RwLock::new(LongTermMemory::new(&config))),
            config: RwLock::new(config),
            current_session_id: AtomicU64::new(0),
            coordinator: Arc::new(MemoryCoordinator::default()),
            cache: DashMap::with_capacity(cache_capacity),
            cache_capacity,
            cache_hits: AtomicU64::new(0),
            cache_lru: Mutex::new(VecDeque::with_capacity(cache_capacity)),
            read_count: AtomicU64::new(0),
            write_count: AtomicU64::new(0),
            search_count: AtomicU64::new(0),
            compression_count: AtomicU64::new(0),
            config_callbacks: Mutex::new(Vec::new()),
            simd_ops: SimdVectorOps::new(),
        }
    }

    /// 使用自定义协调器创建记忆管理器
    ///
    /// 允许自定义跨层协调策略
    ///
    /// # 参数
    /// - `config`: 记忆系统配置
    /// - `coordinator`: 自定义的跨层协调器
    pub fn with_coordinator(config: MemoryConfig, coordinator: MemoryCoordinator) -> Self {
        let cache_capacity = (config.short_term_capacity / 4).max(64);
        Self {
            instant: Arc::new(InstantMemory::new(&config)),
            short_term: Arc::new(RwLock::new(ShortTermMemory::new(&config))),
            long_term: Arc::new(RwLock::new(LongTermMemory::new(&config))),
            config: RwLock::new(config),
            current_session_id: AtomicU64::new(0),
            coordinator: Arc::new(coordinator),
            cache: DashMap::with_capacity(cache_capacity),
            cache_capacity,
            cache_hits: AtomicU64::new(0),
            cache_lru: Mutex::new(VecDeque::with_capacity(cache_capacity)),
            read_count: AtomicU64::new(0),
            write_count: AtomicU64::new(0),
            search_count: AtomicU64::new(0),
            compression_count: AtomicU64::new(0),
            config_callbacks: Mutex::new(Vec::new()),
            simd_ops: SimdVectorOps::new(),
        }
    }

    /// 写入记忆
    pub fn write(&self, level: MemoryLevel, data: Array2<f32>) {
        self.write_count.fetch_add(1, Ordering::Relaxed);
        let timestamp = current_timestamp();

        match level {
            MemoryLevel::Instant => {
                self.instant.write(data, timestamp);
            }
            MemoryLevel::ShortTerm => {
                let mut stm = self.short_term.write();
                stm.write(
                    data,
                    timestamp,
                    Some(self.current_session_id.load(Ordering::Relaxed)),
                );
            }
            MemoryLevel::LongTerm => {
                let key = format!("mem_{}", timestamp);
                let mut ltm = self.long_term.write();
                ltm.write(key, data, timestamp, 1.0);
            }
        }
    }

    /// 写入记忆（带重要性）
    pub fn write_with_importance(&self, level: MemoryLevel, data: Array2<f32>, importance: f32) {
        self.write_count.fetch_add(1, Ordering::Relaxed);

        if level == MemoryLevel::LongTerm {
            let timestamp = current_timestamp();
            let key = format!("mem_{}", timestamp);
            let mut ltm = self.long_term.write();
            ltm.write(key, data, timestamp, importance);
        } else {
            self.write(level, data);
        }
    }

    /// 写入记忆（带键）
    pub fn write_with_key(
        &self,
        level: MemoryLevel,
        key: String,
        data: Array2<f32>,
        importance: f32,
    ) {
        self.write_count.fetch_add(1, Ordering::Relaxed);
        let timestamp = current_timestamp();

        // 缓存驱逐：当缓存超过容量时，逐个移除最久未使用的条目
        while self.cache.len() >= self.cache_capacity {
            self.evict_cache_entries();
        }

        match level {
            MemoryLevel::Instant => {
                self.instant.write(data.clone(), timestamp);
                self.cache.insert(key.clone(), data);
                self.update_lru(&key);
            }
            MemoryLevel::ShortTerm => {
                let mut stm = self.short_term.write();
                stm.write(
                    data.clone(),
                    timestamp,
                    Some(self.current_session_id.load(Ordering::Relaxed)),
                );
                self.cache.insert(key.clone(), data);
                self.update_lru(&key);
            }
            MemoryLevel::LongTerm => {
                let mut ltm = self.long_term.write();
                ltm.write(key.clone(), data.clone(), timestamp, importance);
                self.cache.insert(key.clone(), data);
                self.update_lru(&key);
            }
        }
    }

    /// 更新 LRU 顺序：将 key 移到最近使用位置
    fn update_lru(&self, key: &str) {
        let mut lru = self.cache_lru.lock();
        lru.retain(|k| k != key);
        lru.push_back(key.to_string());
    }

    /// 缓存驱逐：LRU 策略移除最久未使用的条目
    ///
    /// 每次只移除一个条目，避免并发驱逐时过度移除
    fn evict_cache_entries(&self) {
        let mut lru = self.cache_lru.lock();

        if let Some(key) = lru.pop_front() {
            self.cache.remove(&key);
        }
    }

    /// 读取记忆
    pub fn read(&self, level: MemoryLevel) -> Vec<Array2<f32>> {
        self.read_count.fetch_add(1, Ordering::Relaxed);

        match level {
            MemoryLevel::Instant => self.instant.read(),
            MemoryLevel::ShortTerm => {
                let stm = self.short_term.read();
                stm.read_all()
            }
            MemoryLevel::LongTerm => {
                let ltm = self.long_term.read();
                ltm.search_by_importance(ltm.len())
            }
        }
    }

    /// 读取最近 n 条记忆
    pub fn read_last(&self, level: MemoryLevel, n: usize) -> Vec<Array2<f32>> {
        self.read_count.fetch_add(1, Ordering::Relaxed);

        match level {
            MemoryLevel::Instant => self.instant.read_last(n),
            MemoryLevel::ShortTerm => {
                let stm = self.short_term.read();
                stm.read_last(n)
            }
            MemoryLevel::LongTerm => {
                let ltm = self.long_term.read();
                ltm.search_by_importance(n)
            }
        }
    }

    /// 无锁读取（从缓存）
    pub fn get_unchecked(&self, key: &str) -> Option<Array2<f32>> {
        if let Some(data) = self.cache.get(key) {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            self.update_lru(key);
            Some(data.clone())
        } else {
            None
        }
    }

    /// 从缓存获取（带更新）
    pub fn get_with_update(&self, key: &str, level: MemoryLevel) -> Option<Array2<f32>> {
        if let Some(data) = self.cache.get(key) {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            self.update_lru(key);
            return Some(data.clone());
        }

        self.read_count.fetch_add(1, Ordering::Relaxed);

        if level == MemoryLevel::LongTerm {
            let data = {
                let ltm = self.long_term.read();
                ltm.read(key)
            };

            if let Some(data) = data {
                self.cache.insert(key.to_string(), data.clone());
                self.update_lru(key);
                return Some(data);
            }
        }
        None
    }

    /// 搜索长期记忆
    pub fn search(&self, query: &Array2<f32>, top_k: usize) -> Vec<(Array2<f32>, f32)> {
        self.search_count.fetch_add(1, Ordering::Relaxed);

        let ltm = self.long_term.read();
        ltm.search(query, top_k)
            .into_iter()
            .map(|(idx, score)| {
                let data = ltm
                    .items
                    .get(idx)
                    .map(|i| i.data.clone())
                    .unwrap_or_else(|| Array2::zeros((0, 0)));
                (data, score)
            })
            .collect()
    }

    /// 统一搜索接口
    pub fn unified_search(&self, query: &Array2<f32>, top_k: usize) -> Vec<SearchResult> {
        self.search_count.fetch_add(1, Ordering::Relaxed);
        let mut results = self.search_all_levels(query, top_k, true, true, true);
        self.sort_search_results(&mut results);
        results.truncate(top_k);
        results
    }

    /// 跨层搜索
    pub fn search_all_levels(
        &self,
        query: &Array2<f32>,
        top_k: usize,
        include_instant: bool,
        include_short_term: bool,
        include_long_term: bool,
    ) -> Vec<SearchResult> {
        let mut results = Vec::new();

        if include_instant {
            let instant_results = self.search_instant(query, top_k);
            results.extend(instant_results);
        }

        if include_short_term {
            let short_term_results = self.search_short_term(query, top_k);
            results.extend(short_term_results);
        }

        if include_long_term {
            let long_term_results = self.search_long_term(query, top_k);
            results.extend(long_term_results);
        }

        results
    }

    /// 按查询条件搜索
    pub fn search_by_query(&self, query: &SearchQuery) -> Vec<SearchResult> {
        self.search_count.fetch_add(1, Ordering::Relaxed);

        let mut results = self.search_all_levels(
            &query.vector,
            query.top_k * 2,
            query.include_instant,
            query.include_short_term,
            query.include_long_term,
        );

        if query.min_score > 0.0 {
            results.retain(|r| r.score >= query.min_score);
        }

        if let Some(min_imp) = query.min_importance {
            results.retain(|r| r.importance >= min_imp);
        }

        self.sort_search_results(&mut results);
        results.truncate(query.top_k);
        results
    }

    /// 搜索瞬时记忆
    fn search_instant(&self, query: &Array2<f32>, top_k: usize) -> Vec<SearchResult> {
        let instant_data = self.instant.read();
        let mut scores: Vec<(Array2<f32>, f32)> = Vec::with_capacity(instant_data.len());

        for data in instant_data.into_iter() {
            let score = Self::compute_similarity(query, &data);
            scores.push((data, score));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
            .into_iter()
            .take(top_k)
            .map(|(data, score)| SearchResult::new(data, score, MemoryLevel::Instant, 0, 1.0))
            .collect()
    }

    /// 搜索短期记忆（优化锁持有时间）
    fn search_short_term(&self, query: &Array2<f32>, top_k: usize) -> Vec<SearchResult> {
        let items: Vec<Array2<f32>> = {
            let stm = self.short_term.read();
            stm.read_all().into_iter().collect()
        };

        let mut scores: Vec<(Array2<f32>, f32)> = Vec::with_capacity(items.len());
        for data in items {
            let score = Self::compute_similarity(query, &data);
            scores.push((data, score));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
            .into_iter()
            .take(top_k)
            .map(|(data, score)| SearchResult::new(data, score, MemoryLevel::ShortTerm, 0, 1.0))
            .collect()
    }

    /// 搜索长期记忆（优化锁持有时间）
    fn search_long_term(&self, query: &Array2<f32>, top_k: usize) -> Vec<SearchResult> {
        let ltm = self.long_term.read();
        ltm.search(query, top_k)
            .into_iter()
            .filter_map(|(idx, score)| {
                ltm.items.get(idx).map(|item| {
                    SearchResult::new(
                        item.data.clone(),
                        score,
                        MemoryLevel::LongTerm,
                        item.timestamp,
                        item.importance,
                    )
                })
            })
            .collect()
    }

    /// 排序搜索结果
    fn sort_search_results(&self, results: &mut [SearchResult]) {
        results.sort_by(|a, b| {
            let score_a = a.score * 0.6 + a.importance * 0.4;
            let score_b = b.score * 0.6 + b.importance * 0.4;
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// 计算相似度（使用 SIMD 加速）
    ///
    /// 要求两个矩阵的列数相同，否则返回 0.0
    fn compute_similarity(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        // 维度必须一致
        if a.ncols() != b.ncols() {
            return 0.0;
        }

        let simd_ops = SimdVectorOps::new();

        let a_slice: &[f32] = a.as_slice().unwrap_or(&[]);
        let b_slice: &[f32] = b.as_slice().unwrap_or(&[]);

        if a_slice.is_empty() || b_slice.is_empty() {
            return 0.0;
        }

        let min_len = a_slice.len().min(b_slice.len());

        simd_ops.cosine_similarity(&a_slice[..min_len], &b_slice[..min_len])
    }

    /// 提升到长期记忆
    ///
    /// 只有当 key 在缓存中存在时才能提升，否则返回 false
    pub fn promote_to_long_term(&self, key: &str, importance: f32) -> bool {
        if !self.coordinator.should_promote(importance) {
            return false;
        }

        // 必须从缓存中获取对应 key 的数据
        let data = match self.cache.get(key) {
            Some(cached) => cached.clone(),
            None => return false, // 找不到对应 key，提升失败
        };

        let timestamp = current_timestamp();
        let new_key = format!("promoted_{}_{}", key, timestamp);

        let mut ltm = self.long_term.write();
        ltm.write(new_key, data.clone(), timestamp, importance);
        drop(ltm);
        self.cache.insert(key.to_string(), data);
        self.update_lru(key);
        self.coordinator.record_promotion();
        true
    }

    /// 降级到短期记忆
    pub fn demote_to_short_term(&self, key: &str) -> bool {
        let (data, importance) = {
            let ltm = self.long_term.read();
            match ltm.read(key) {
                Some(d) => {
                    let imp = ltm.get_importance(key).unwrap_or(1.0);
                    (d, imp)
                }
                None => return false,
            }
        };

        if !self.coordinator.should_demote(importance) {
            return false;
        }

        let timestamp = current_timestamp();

        let mut stm = self.short_term.write();
        stm.write(data.clone(), timestamp, None);
        drop(stm);
        self.cache.insert(key.to_string(), data);
        self.update_lru(key);
        self.coordinator.record_demotion();
        true
    }

    /// 记忆巩固
    pub fn consolidate(&self) -> usize {
        if !self.coordinator.should_consolidate() {
            return 0;
        }

        let mut consolidated_count = 0;

        let important_memories = {
            let ltm = self.long_term.read();
            ltm.search_by_importance(10)
        };

        let mut stm = self.short_term.write();
        let timestamp = current_timestamp();
        for data in important_memories {
            stm.write(data, timestamp, None);
            consolidated_count += 1;
        }

        self.coordinator.mark_consolidation();
        consolidated_count
    }

    /// 强制巩固（忽略时间间隔）
    pub fn force_consolidate(&self) -> usize {
        let mut consolidated_count = 0;

        let important_memories = {
            let ltm = self.long_term.read();
            ltm.search_by_importance(10)
        };

        let mut stm = self.short_term.write();
        let timestamp = current_timestamp();
        for data in important_memories {
            stm.write(data, timestamp, None);
            consolidated_count += 1;
        }

        self.coordinator.mark_consolidation();
        consolidated_count
    }

    /// 巩固到短期记忆（兼容旧 API）
    pub fn consolidate_to_short_term(&self) {
        self.consolidate();
    }

    /// 开始会话
    pub fn start_session(&mut self) -> u64 {
        self.current_session_id.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// 结束会话
    pub fn end_session(&self) {
        self.consolidate();
    }

    /// 清空记忆
    pub fn clear(&self, level: Option<MemoryLevel>) {
        match level {
            Some(MemoryLevel::Instant) => {
                self.instant.clear();
            }
            Some(MemoryLevel::ShortTerm) => {
                let mut stm = self.short_term.write();
                stm.clear();
            }
            Some(MemoryLevel::LongTerm) => {
                let mut ltm = self.long_term.write();
                ltm.clear();
            }
            None => {
                self.instant.clear();
                let mut stm = self.short_term.write();
                stm.clear();
                let mut ltm = self.long_term.write();
                ltm.clear();
                self.cache.clear();
                self.cache_lru.lock().clear();
            }
        }
    }

    /// 获取配置
    pub fn get_config(&self) -> MemoryConfig {
        self.config.read().clone()
    }

    /// 更新配置
    pub fn update_config(&self, new_config: MemoryConfig) {
        {
            let mut config = self.config.write();
            *config = new_config.clone();
        }

        let callbacks = self.config_callbacks.lock();
        for callback in callbacks.iter() {
            callback(&new_config);
        }
    }

    /// 重载配置
    pub fn reload_config(&self) -> MemoryConfig {
        self.config.read().clone()
    }

    /// 注册配置变更回调
    pub fn register_config_callback(&self, callback: ConfigChangeCallback) {
        let mut callbacks = self.config_callbacks.lock();
        callbacks.push(callback);
    }

    /// 获取统计信息
    pub fn get_stats(&self) -> DMNStats {
        let instant_count = self.instant.read().len();
        let short_term_count = {
            let stm = self.short_term.read();
            stm.len()
        };
        let long_term_count = {
            let ltm = self.long_term.read();
            ltm.len()
        };

        let total_reads = self.read_count.load(Ordering::Relaxed);
        let total_writes = self.write_count.load(Ordering::Relaxed);
        let total_searches = self.search_count.load(Ordering::Relaxed);

        let cache_hits = self.cache_hits.load(Ordering::Relaxed);
        let total_cache_requests = cache_hits + total_reads;
        let cache_hit_rate = if total_cache_requests > 0 {
            cache_hits as f32 / total_cache_requests as f32
        } else {
            0.0
        };

        DMNStats {
            instant_count,
            short_term_count,
            long_term_count,
            total_reads,
            total_writes,
            total_searches,
            promotions: self.coordinator.get_promotion_count(),
            demotions: self.coordinator.get_demotion_count(),
            compressions: self.compression_count.load(Ordering::Relaxed),
            cache_hit_rate,
            avg_query_latency_us: 0.0,
        }
    }

    /// 获取 Prometheus 格式指标
    pub fn get_metrics(&self) -> DMNMetrics {
        let stats = self.get_stats();
        let mut metrics = DMNMetrics::new();

        metrics.insert("instant_memory_count", stats.instant_count as f64);
        metrics.insert("short_term_memory_count", stats.short_term_count as f64);
        metrics.insert("long_term_memory_count", stats.long_term_count as f64);
        metrics.insert("total_reads", stats.total_reads as f64);
        metrics.insert("total_writes", stats.total_writes as f64);
        metrics.insert("total_searches", stats.total_searches as f64);
        metrics.insert("promotions", stats.promotions as f64);
        metrics.insert("demotions", stats.demotions as f64);
        metrics.insert("compressions", stats.compressions as f64);
        metrics.insert("cache_hit_rate", stats.cache_hit_rate as f64);
        metrics.insert("cache_size", self.cache.len() as f64);

        metrics
    }

    /// 获取 Prometheus 格式字符串
    pub fn get_prometheus_metrics(&self) -> String {
        self.get_metrics().to_prometheus_format()
    }

    /// 获取瞬时记忆引用
    pub fn instant_memory(&self) -> &InstantMemory {
        &self.instant
    }

    /// 获取短期记忆 Arc
    pub fn short_term_memory(&self) -> Arc<RwLock<ShortTermMemory>> {
        Arc::clone(&self.short_term)
    }

    /// 获取长期记忆 Arc
    pub fn long_term_memory(&self) -> Arc<RwLock<LongTermMemory>> {
        Arc::clone(&self.long_term)
    }

    /// 获取协调器引用
    pub fn coordinator(&self) -> &MemoryCoordinator {
        &self.coordinator
    }

    /// 合并所有记忆
    ///
    /// 只合并维度一致的记忆，维度不一致的会被过滤掉
    pub fn merge_all_memory(&self) -> Array2<f32> {
        let mut all_data: Vec<Array2<f32>> = Vec::new();

        all_data.extend(self.read(MemoryLevel::Instant));
        all_data.extend(self.read(MemoryLevel::ShortTerm));
        all_data.extend(self.read(MemoryLevel::LongTerm));

        if all_data.is_empty() {
            return Array2::zeros((0, 0));
        }

        // 先确定基准维度
        let dim = all_data[0].ncols();

        // 先过滤维度不一致的数据，再计算总行数
        let all_data: Vec<Array2<f32>> = all_data
            .into_iter()
            .filter(|data| data.ncols() == dim)
            .collect();

        if all_data.is_empty() {
            return Array2::zeros((0, 0));
        }

        let total_rows: usize = all_data.iter().map(|m| m.nrows()).sum();

        let mut result = Array2::zeros((total_rows, dim));
        let mut offset = 0;
        for data in all_data {
            let rows = data.nrows();
            for i in 0..rows {
                for j in 0..dim {
                    result[[offset + i, j]] = data[[i, j]];
                }
            }
            offset += rows;
        }

        result
    }

    /// 缓存大小
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// 清理缓存
    pub fn clear_cache(&self) {
        self.cache.clear();
        self.cache_lru.lock().clear();
    }

    /// 智能压缩短期记忆
    pub fn compress_short_term(&self) -> usize {
        let mut stm = self.short_term.write();
        let removed = stm.compress_smart();
        self.compression_count.fetch_add(1, Ordering::Relaxed);
        removed.len()
    }

    /// 应用长期记忆衰减
    pub fn apply_long_term_decay(&self) {
        let mut ltm = self.long_term.write();
        ltm.apply_decay();
    }

    /// 自动遗忘低重要性记忆
    pub fn auto_forget(&self) -> usize {
        let mut ltm = self.long_term.write();
        ltm.auto_forget()
    }

    /// 构建长期记忆的 HNSW 索引
    pub fn build_long_term_index(&self) {
        let mut ltm = self.long_term.write();
        ltm.build_hnsw_index(None);
    }

    /// 持久化长期记忆
    pub fn persist_long_term(&self) -> std::io::Result<()> {
        let mut ltm = self.long_term.write();
        ltm.persist()
    }

    /// 恢复长期记忆
    pub fn restore_long_term(&self) -> std::io::Result<usize> {
        let mut ltm = self.long_term.write();
        ltm.restore()
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new(MemoryConfig::default())
    }
}

/// 获取当前时间戳
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_manager_write_read() {
        let manager = MemoryManager::default();
        let data = Array2::zeros((10, 512));

        manager.write(MemoryLevel::Instant, data.clone());
        let read = manager.read(MemoryLevel::Instant);

        assert!(!read.is_empty());
    }

    #[test]
    fn test_memory_manager_session() {
        let mut manager = MemoryManager::default();

        let session_id = manager.start_session();
        assert!(session_id > 0);

        let data = Array2::zeros((5, 512));
        manager.write(MemoryLevel::ShortTerm, data);

        manager.end_session();
    }

    #[test]
    fn test_memory_manager_clear() {
        let manager = MemoryManager::default();
        let data = Array2::zeros((10, 512));

        manager.write(MemoryLevel::Instant, data);
        manager.clear(None);

        assert!(manager.read(MemoryLevel::Instant).is_empty());
    }

    #[test]
    fn test_merge_all_memory() {
        let manager = MemoryManager::default();

        let data1 = Array2::ones((5, 128));
        let data2 = Array2::ones((3, 128));

        manager.write(MemoryLevel::Instant, data1);
        manager.write(MemoryLevel::ShortTerm, data2);

        let merged = manager.merge_all_memory();
        assert_eq!(merged.nrows(), 8);
    }

    #[test]
    fn test_unified_search() {
        let manager = MemoryManager::default();

        let data1 = Array2::ones((5, 128));
        let data2 = Array2::zeros((3, 128));

        manager.write(MemoryLevel::Instant, data1.clone());
        manager.write(MemoryLevel::ShortTerm, data2);

        let query = Array2::ones((1, 128));
        let results = manager.unified_search(&query, 5);

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_search_all_levels() {
        let manager = MemoryManager::default();

        let data = Array2::ones((5, 128));
        manager.write(MemoryLevel::Instant, data.clone());
        manager.write(MemoryLevel::ShortTerm, data.clone());

        let query = Array2::ones((1, 128));
        let results = manager.search_all_levels(&query, 10, true, true, false);

        assert!(!results.is_empty());
    }

    #[test]
    fn test_search_by_query() {
        let manager = MemoryManager::default();

        let data = Array2::ones((5, 128));
        manager.write(MemoryLevel::Instant, data);

        let query = SearchQuery::new(Array2::ones((1, 128)), 5)
            .with_min_score(0.5)
            .with_levels(true, false, false);

        let results = manager.search_by_query(&query);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_get_unchecked() {
        let manager = MemoryManager::default();

        let data = Array2::ones((5, 128));
        manager.write_with_key(
            MemoryLevel::LongTerm,
            "test_key".to_string(),
            data.clone(),
            1.0,
        );

        let cached = manager.get_unchecked("test_key");
        assert!(cached.is_some());
    }

    #[test]
    fn test_promote_to_long_term() {
        let coordinator = MemoryCoordinator::new(0.5, 0.2, 300);
        let manager = MemoryManager::with_coordinator(MemoryConfig::default(), coordinator);

        let data = Array2::ones((5, 128));
        manager.cache.insert("test_promote".to_string(), data);

        let result = manager.promote_to_long_term("test_promote", 0.8);
        assert!(result);

        let stats = manager.get_stats();
        assert_eq!(stats.promotions, 1);
    }

    #[test]
    fn test_consolidate() {
        let manager = MemoryManager::default();

        let data = Array2::ones((5, 128));
        manager.write_with_importance(MemoryLevel::LongTerm, data, 0.9);

        manager.force_consolidate();

        let short_term_data = manager.read(MemoryLevel::ShortTerm);
        assert!(!short_term_data.is_empty());
    }

    #[test]
    fn test_update_config() {
        let manager = MemoryManager::default();

        let new_config = MemoryConfig {
            instant_capacity: 8192,
            short_term_capacity: 2048,
            long_term_capacity: 20000,
            compression_threshold: 1024,
            eviction_strategy: EvictionStrategy::LFU,
            embedding_dim: None,
            padding_strategy: PaddingStrategy::default(),
        };

        manager.update_config(new_config.clone());
        let config = manager.get_config();

        assert_eq!(config.instant_capacity, 8192);
        assert_eq!(config.short_term_capacity, 2048);
    }

    #[test]
    fn test_config_callback() {
        let manager = MemoryManager::default();
        let called = Arc::new(AtomicU64::new(0));
        let called_clone = Arc::clone(&called);

        manager.register_config_callback(Box::new(move |_| {
            called_clone.fetch_add(1, Ordering::Relaxed);
        }));

        let new_config = MemoryConfig {
            instant_capacity: 8192,
            ..Default::default()
        };

        manager.update_config(new_config);

        assert_eq!(called.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_get_stats() {
        let manager = MemoryManager::default();

        let data = Array2::ones((5, 128));
        manager.write(MemoryLevel::Instant, data.clone());
        manager.write(MemoryLevel::ShortTerm, data.clone());
        manager.write(MemoryLevel::LongTerm, data);

        let stats = manager.get_stats();

        assert!(stats.instant_count > 0);
        assert!(stats.short_term_count > 0);
        assert!(stats.long_term_count > 0);
        assert!(stats.total_writes >= 3);
    }

    #[test]
    fn test_get_metrics() {
        let manager = MemoryManager::default();

        let data = Array2::ones((5, 128));
        manager.write(MemoryLevel::Instant, data);

        let metrics = manager.get_metrics();

        assert!(metrics.metrics.contains_key("instant_memory_count"));
        assert!(metrics.metrics.contains_key("total_writes"));
    }

    #[test]
    fn test_prometheus_format() {
        let manager = MemoryManager::default();

        let data = Array2::ones((5, 128));
        manager.write(MemoryLevel::Instant, data);

        let prometheus_output = manager.get_prometheus_metrics();

        assert!(prometheus_output.contains("# HELP dmn_memory_count"));
        assert!(prometheus_output.contains("# TYPE dmn_memory_count gauge"));
        assert!(prometheus_output.contains("dmn_memory_count{level=\"instant\"}"));
    }

    #[test]
    fn test_cache_operations() {
        let manager = MemoryManager::default();

        let data = Array2::ones((5, 128));
        manager.write_with_key(MemoryLevel::LongTerm, "key1".to_string(), data.clone(), 1.0);
        manager.write_with_key(MemoryLevel::LongTerm, "key2".to_string(), data.clone(), 1.0);

        assert_eq!(manager.cache_size(), 2);

        manager.clear_cache();
        assert_eq!(manager.cache_size(), 0);
    }

    #[test]
    fn test_lru_eviction() {
        let config = MemoryConfig {
            short_term_capacity: 256,
            ..Default::default()
        };
        let manager = MemoryManager::new(config);

        let data = Array2::ones((5, 128));

        for i in 0..300 {
            manager.write_with_key(
                MemoryLevel::LongTerm,
                format!("key{}", i),
                data.clone(),
                1.0,
            );
        }

        assert!(manager.cache_size() <= 64, "缓存大小应不超过容量");
    }

    #[test]
    fn test_cache_hit_rate() {
        let manager = MemoryManager::default();

        let data = Array2::ones((5, 128));
        manager.write_with_key(MemoryLevel::LongTerm, "key1".to_string(), data.clone(), 1.0);

        let stats = manager.get_stats();
        assert_eq!(stats.total_writes, 1);

        // 缓存命中
        manager.get_unchecked("key1");
        manager.get_unchecked("key1");
        manager.get_unchecked("key1");

        // 缓存未命中（会增加 read_count）
        manager.get_with_update("key2", MemoryLevel::LongTerm);

        let stats = manager.get_stats();
        // cache_hits 应该为 3，read_count 应该为 1
        assert!(stats.cache_hit_rate > 0.0, "缓存命中率应大于0");
    }

    #[test]
    fn test_memory_coordinator() {
        let coordinator = MemoryCoordinator::new(0.7, 0.2, 300);

        assert!(coordinator.should_promote(0.8));
        assert!(!coordinator.should_promote(0.5));

        assert!(coordinator.should_demote(0.1));
        assert!(!coordinator.should_demote(0.5));
    }

    #[test]
    fn test_search_result_sorting() {
        let manager = MemoryManager::default();

        let data1 = Array2::ones((5, 128));
        let data2 = Array2::zeros((5, 128));

        manager.write_with_importance(MemoryLevel::LongTerm, data1, 0.9);
        manager.write_with_importance(MemoryLevel::LongTerm, data2, 0.3);

        let query = Array2::ones((1, 128));
        let results = manager.unified_search(&query, 2);

        if results.len() >= 2 {
            assert!(results[0].importance >= results[1].importance);
        }
    }

    #[test]
    fn test_compress_short_term() {
        let config = MemoryConfig {
            short_term_capacity: 20,
            compression_threshold: 5,
            ..Default::default()
        };
        let manager = MemoryManager::new(config);

        for i in 0..10 {
            let data = Array2::from_shape_fn((5, 32), |(r, c)| (i + r + c) as f32 * 0.1);
            manager.write(MemoryLevel::ShortTerm, data);
        }

        let compressed = manager.compress_short_term();
        let _ = compressed;
    }

    #[test]
    fn test_write_with_key() {
        let manager = MemoryManager::default();

        let data = Array2::ones((5, 128));
        manager.write_with_key(
            MemoryLevel::LongTerm,
            "custom_key".to_string(),
            data.clone(),
            0.8,
        );

        let cached = manager.get_unchecked("custom_key");
        assert!(cached.is_some());
    }

    #[test]
    fn test_multiple_sessions() {
        let mut manager = MemoryManager::default();

        let session1 = manager.start_session();
        manager.write(MemoryLevel::ShortTerm, Array2::zeros((5, 128)));

        let session2 = manager.start_session();
        manager.write(MemoryLevel::ShortTerm, Array2::zeros((5, 128)));

        assert_ne!(session1, session2);
    }
}
