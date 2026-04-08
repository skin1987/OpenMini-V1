//! KV Cache 与记忆系统深度集成模块
//!
//! 本模块实现 KV Cache 与三级记忆系统之间的双向数据流转：
//! - 瞬时记忆到 KV Cache 映射
//! - 记忆预加载到 KV Cache
//! - KV Cache 到记忆的回写
//! - 内存共享优化
//!
//! # 架构
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    CacheBridge 系统                          │
//! ├─────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
//! │  │ CacheMapper │  │CachePreloader│ │ CacheWriter │        │
//! │  │  (映射层)   │  │  (预加载层) │  │  (回写层)   │        │
//! │  └─────────────┘  └─────────────┘  └─────────────┘        │
//! │         │                │                │                │
//! │         └────────────────┴────────────────┘                │
//! │                          │                                 │
//! │              SharedMemoryPool (内存共享)                    │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use bytes::Bytes;
use ndarray::Array2;
use parking_lot::RwLock;

use super::instant::{InstantMemory, ZeroCopyBuffer};
use super::long_term::LongTermMemory;
use super::short_term::ShortTermMemory;
use super::MemoryLevel;
use crate::hardware::kv_cache::paged_cache::PagedKVCache;

/// 映射统计信息
#[derive(Debug, Clone, Default)]
pub struct MappingStats {
    /// 成功映射次数
    pub successful_mappings: u64,
    /// 失败映射次数
    pub failed_mappings: u64,
    /// 同步次数
    pub sync_count: u64,
    /// 总映射数据量（字节）
    pub total_mapped_bytes: usize,
    /// 平均映射延迟（微秒）
    pub avg_mapping_latency_us: f64,
}

impl MappingStats {
    /// 计算映射成功率
    pub fn success_rate(&self) -> f32 {
        let total = self.successful_mappings + self.failed_mappings;
        if total == 0 {
            return 0.0;
        }
        self.successful_mappings as f32 / total as f32
    }
}

/// KV Cache 映射条目
#[derive(Debug, Clone)]
pub struct KVMappingEntry {
    /// 请求ID
    pub request_id: u64,
    /// 层索引
    pub layer: usize,
    /// 起始位置
    pub start_pos: usize,
    /// 长度
    pub length: usize,
    /// 时间戳
    pub timestamp: u64,
    /// 重要性权重
    pub importance: f32,
}

/// 瞬时记忆到 KV Cache 映射器
///
/// 负责将瞬时记忆层的数据映射到 KV Cache 中，
/// 实现零拷贝传输和高效数据流转
#[derive(Debug)]
pub struct CacheMapper {
    /// 映射表：记忆索引 -> KV映射条目
    mapping_table: RwLock<HashMap<usize, KVMappingEntry>>,
    /// 反向映射：请求ID -> 记忆索引
    reverse_mapping: RwLock<HashMap<u64, usize>>,
    /// 关联的 KV Cache
    kv_cache: RwLock<Option<Arc<RwLock<PagedKVCache>>>>,
    /// 关联的瞬时记忆
    instant_memory: RwLock<Option<Arc<InstantMemory>>>,
    /// 统计信息
    stats: RwLock<MappingStats>,
    /// 配置
    config: MapperConfig,
    /// 计数器
    mapping_counter: AtomicU64,
}

/// 映射器配置
#[derive(Debug, Clone)]
pub struct MapperConfig {
    /// 最大映射数量
    pub max_mappings: usize,
    /// 是否启用零拷贝
    pub enable_zero_copy: bool,
    /// 映射超时（毫秒）
    pub mapping_timeout_ms: u64,
    /// 是否自动同步
    pub auto_sync: bool,
}

impl Default for MapperConfig {
    fn default() -> Self {
        Self {
            max_mappings: 10000,
            enable_zero_copy: true,
            mapping_timeout_ms: 5000,
            auto_sync: true,
        }
    }
}

impl CacheMapper {
    /// 创建新的映射器
    pub fn new(config: MapperConfig) -> Self {
        Self {
            mapping_table: RwLock::new(HashMap::new()),
            reverse_mapping: RwLock::new(HashMap::new()),
            kv_cache: RwLock::new(None),
            instant_memory: RwLock::new(None),
            stats: RwLock::new(MappingStats::default()),
            config,
            mapping_counter: AtomicU64::new(0),
        }
    }

    /// 关联 KV Cache
    pub fn attach_kv_cache(&self, kv_cache: Arc<RwLock<PagedKVCache>>) {
        let mut cache = self.kv_cache.write();
        *cache = Some(kv_cache);
    }

    /// 关联瞬时记忆
    pub fn attach_instant_memory(&self, instant_memory: Arc<InstantMemory>) {
        let mut mem = self.instant_memory.write();
        *mem = Some(instant_memory);
    }

    /// 映射瞬时记忆到 KV Cache
    ///
    /// 将瞬时记忆层的数据映射到 KV Cache 的指定位置
    /// 支持零拷贝传输以提高性能
    pub fn map_instant_to_kv(
        &self,
        memory_index: usize,
        request_id: u64,
        layer: usize,
        start_pos: usize,
    ) -> Result<KVMappingEntry, String> {
        let start_time = std::time::Instant::now();

        let kv_cache_guard = self.kv_cache.read();
        let kv_cache = kv_cache_guard
            .as_ref()
            .ok_or_else(|| "KV Cache not attached".to_string())?;

        let instant_memory = self.instant_memory.read();
        let instant_mem = instant_memory
            .as_ref()
            .ok_or_else(|| "Instant memory not attached".to_string())?;

        let data = instant_mem.read_last(1);
        if data.is_empty() {
            let mut stats = self.stats.write();
            stats.failed_mappings += 1;
            return Err("No data in instant memory".to_string());
        }

        let data = if memory_index < data.len() {
            data[memory_index].clone()
        } else {
            data[0].clone()
        };

        let length = data.nrows();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut kv_cache = kv_cache.write();

        kv_cache
            .allocate_slots(request_id, length)
            .inspect_err(|_e| {
                let mut stats = self.stats.write();
                stats.failed_mappings += 1;
            })?;

        let v = data.clone();
        kv_cache.write_kv(request_id, layer, start_pos, &data, &v)?;

        let entry = KVMappingEntry {
            request_id,
            layer,
            start_pos,
            length,
            timestamp,
            importance: 1.0,
        };

        {
            let mut table = self.mapping_table.write();
            if table.len() >= self.config.max_mappings {
                let oldest_entry = table
                    .iter()
                    .min_by_key(|(_, e)| e.timestamp)
                    .map(|(k, v)| (*k, v.request_id));
                if let Some((key, request_id)) = oldest_entry {
                    table.remove(&key);
                    drop(table);
                    let mut reverse = self.reverse_mapping.write();
                    reverse.remove(&request_id);
                    let mut table = self.mapping_table.write();
                    table.insert(memory_index, entry.clone());
                } else {
                    table.insert(memory_index, entry.clone());
                }
            } else {
                table.insert(memory_index, entry.clone());
            }
        }

        {
            let mut reverse = self.reverse_mapping.write();
            reverse.insert(request_id, memory_index);
        }

        let elapsed = start_time.elapsed().as_micros() as f64;
        {
            let mut stats = self.stats.write();
            stats.successful_mappings += 1;
            stats.total_mapped_bytes += data.len() * std::mem::size_of::<f32>();
            stats.avg_mapping_latency_us =
                (stats.avg_mapping_latency_us * (stats.successful_mappings - 1) as f64 + elapsed)
                    / stats.successful_mappings as f64;
        }

        self.mapping_counter.fetch_add(1, Ordering::Relaxed);

        Ok(entry)
    }

    /// 批量映射瞬时记忆到 KV Cache
    pub fn map_batch(
        &self,
        entries: Vec<(usize, u64, usize, usize)>,
    ) -> Vec<Result<KVMappingEntry, String>> {
        entries
            .into_iter()
            .map(|(mem_idx, req_id, layer, start_pos)| {
                self.map_instant_to_kv(mem_idx, req_id, layer, start_pos)
            })
            .collect()
    }

    /// 同步瞬时记忆到 KV Cache
    ///
    /// 将瞬时记忆层的所有数据同步到 KV Cache
    pub fn sync_to_kv_cache(&self, request_id: u64, layer: usize) -> Result<usize, String> {
        let kv_cache_guard = self.kv_cache.read();
        let kv_cache = kv_cache_guard
            .as_ref()
            .ok_or_else(|| "KV Cache not attached".to_string())?;

        let instant_memory = self.instant_memory.read();
        let instant_mem = instant_memory
            .as_ref()
            .ok_or_else(|| "Instant memory not attached".to_string())?;

        let all_data = instant_mem.read();
        if all_data.is_empty() {
            return Ok(0);
        }

        let total_rows: usize = all_data.iter().map(|d| d.nrows()).sum();
        let dim = all_data[0].ncols();

        let mut combined = Array2::zeros((total_rows, dim));
        let mut offset = 0;
        for data in &all_data {
            let rows = data.nrows();
            for i in 0..rows {
                for j in 0..dim {
                    combined[[offset + i, j]] = data[[i, j]];
                }
            }
            offset += rows;
        }

        let mut kv_cache = kv_cache.write();

        if !kv_cache.contains_request(request_id) {
            kv_cache.allocate_slots(request_id, total_rows)?;
        }

        let v = combined.clone();
        kv_cache.write_kv(request_id, layer, 0, &combined, &v)?;

        {
            let mut stats = self.stats.write();
            stats.sync_count += 1;
        }

        Ok(total_rows)
    }

    /// 获取映射条目
    pub fn get_mapping(&self, memory_index: usize) -> Option<KVMappingEntry> {
        let table = self.mapping_table.read();
        table.get(&memory_index).cloned()
    }

    /// 通过请求ID获取映射
    pub fn get_mapping_by_request(&self, request_id: u64) -> Option<KVMappingEntry> {
        let reverse = self.reverse_mapping.read();
        reverse.get(&request_id).and_then(|&idx| {
            let table = self.mapping_table.read();
            table.get(&idx).cloned()
        })
    }

    /// 移除映射
    pub fn remove_mapping(&self, memory_index: usize) -> Option<KVMappingEntry> {
        let mut table = self.mapping_table.write();
        let entry = table.remove(&memory_index);

        if let Some(ref e) = entry {
            let mut reverse = self.reverse_mapping.write();
            reverse.remove(&e.request_id);
        }

        entry
    }

    /// 清空所有映射
    pub fn clear_mappings(&self) {
        let mut table = self.mapping_table.write();
        let mut reverse = self.reverse_mapping.write();
        table.clear();
        reverse.clear();
    }

    /// 获取映射数量
    pub fn mapping_count(&self) -> usize {
        self.mapping_table.read().len()
    }

    /// 获取统计信息
    pub fn stats(&self) -> MappingStats {
        self.stats.read().clone()
    }

    /// 重置统计
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write();
        *stats = MappingStats::default();
    }
}

impl Default for CacheMapper {
    fn default() -> Self {
        Self::new(MapperConfig::default())
    }
}

/// 预加载统计信息
#[derive(Debug, Clone, Default)]
pub struct PreloadStats {
    /// 成功预加载次数
    pub successful_preloads: u64,
    /// 失败预加载次数
    pub failed_preloads: u64,
    /// 预加载的总token数
    pub total_preloaded_tokens: usize,
    /// 缓存命中次数
    pub cache_hits: u64,
    /// 缓存未命中次数
    pub cache_misses: u64,
    /// 平均预加载延迟（微秒）
    pub avg_preload_latency_us: f64,
}

impl PreloadStats {
    /// 计算缓存命中率
    pub fn hit_rate(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            return 0.0;
        }
        self.cache_hits as f32 / total as f32
    }
}

/// 预加载器配置
#[derive(Debug, Clone)]
pub struct PreloaderConfig {
    /// 最大预加载数量
    pub max_preload_count: usize,
    /// 预加载批次大小
    pub batch_size: usize,
    /// 是否启用优先级预加载
    pub enable_priority: bool,
    /// 预加载超时（毫秒）
    pub preload_timeout_ms: u64,
}

impl Default for PreloaderConfig {
    fn default() -> Self {
        Self {
            max_preload_count: 1000,
            batch_size: 32,
            enable_priority: true,
            preload_timeout_ms: 10000,
        }
    }
}

/// 预加载任务
#[derive(Debug, Clone)]
pub struct PreloadTask {
    /// 任务ID
    pub task_id: u64,
    /// 源记忆层级
    pub source_level: MemoryLevel,
    /// 目标请求ID
    pub target_request_id: u64,
    /// 目标层
    pub target_layer: usize,
    /// 优先级（越高越优先）
    pub priority: f32,
    /// 状态
    pub status: PreloadStatus,
}

/// 预加载状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreloadStatus {
    /// 待处理
    Pending,
    /// 进行中
    InProgress,
    /// 已完成
    Completed,
    /// 失败
    Failed,
}

/// 记忆预加载器
///
/// 负责将短期记忆和长期记忆预加载到 KV Cache 中，
/// 支持按查询预加载和优先级调度
#[derive(Debug)]
pub struct CachePreloader {
    /// 关联的 KV Cache
    kv_cache: RwLock<Option<Arc<RwLock<PagedKVCache>>>>,
    /// 短期记忆引用
    short_term_memory: RwLock<Option<Arc<RwLock<ShortTermMemory>>>>,
    /// 长期记忆引用
    long_term_memory: RwLock<Option<Arc<RwLock<LongTermMemory>>>>,
    /// 预加载任务队列
    task_queue: RwLock<Vec<PreloadTask>>,
    /// 统计信息
    stats: RwLock<PreloadStats>,
    /// 配置
    config: PreloaderConfig,
    /// 任务计数器
    task_counter: AtomicU64,
}

impl CachePreloader {
    /// 创建新的预加载器
    pub fn new(config: PreloaderConfig) -> Self {
        Self {
            kv_cache: RwLock::new(None),
            short_term_memory: RwLock::new(None),
            long_term_memory: RwLock::new(None),
            task_queue: RwLock::new(Vec::new()),
            stats: RwLock::new(PreloadStats::default()),
            config,
            task_counter: AtomicU64::new(0),
        }
    }

    /// 关联 KV Cache
    pub fn attach_kv_cache(&self, kv_cache: Arc<RwLock<PagedKVCache>>) {
        let mut cache = self.kv_cache.write();
        *cache = Some(kv_cache);
    }

    /// 关联短期记忆
    pub fn attach_short_term_memory(&self, memory: Arc<RwLock<ShortTermMemory>>) {
        let mut mem = self.short_term_memory.write();
        *mem = Some(memory);
    }

    /// 关联长期记忆
    pub fn attach_long_term_memory(&self, memory: Arc<RwLock<LongTermMemory>>) {
        let mut mem = self.long_term_memory.write();
        *mem = Some(memory);
    }

    /// 预加载记忆到 KV Cache
    ///
    /// 将指定记忆层级的数据预加载到 KV Cache
    pub fn preload_memories(
        &self,
        source_level: MemoryLevel,
        request_id: u64,
        layer: usize,
        count: usize,
    ) -> Result<usize, String> {
        let start_time = std::time::Instant::now();

        let kv_cache_guard = self.kv_cache.read();
        let kv_cache = kv_cache_guard
            .as_ref()
            .ok_or_else(|| "KV Cache not attached".to_string())?;

        let memories: Vec<Array2<f32>> = match source_level {
            MemoryLevel::ShortTerm => {
                let short_term = self.short_term_memory.read();
                let mem = short_term
                    .as_ref()
                    .ok_or_else(|| "Short term memory not attached".to_string())?;
                let mem_guard = mem.read();
                mem_guard.read_last(count)
            }
            MemoryLevel::LongTerm => {
                let long_term = self.long_term_memory.read();
                let mem = long_term
                    .as_ref()
                    .ok_or_else(|| "Long term memory not attached".to_string())?;
                let mem_guard = mem.read();
                mem_guard.search_by_importance(count)
            }
            MemoryLevel::Instant => {
                return Err("Use CacheMapper for instant memory".to_string());
            }
        };

        if memories.is_empty() {
            return Ok(0);
        }

        let total_rows: usize = memories.iter().map(|m| m.nrows()).sum();
        let dim = memories[0].ncols();

        let mut combined = Array2::zeros((total_rows, dim));
        let mut offset = 0;
        for data in &memories {
            let rows = data.nrows();
            for i in 0..rows {
                for j in 0..dim {
                    combined[[offset + i, j]] = data[[i, j]];
                }
            }
            offset += rows;
        }

        let mut kv_cache = kv_cache.write();

        if !kv_cache.contains_request(request_id) {
            kv_cache
                .allocate_slots(request_id, total_rows)
                .map_err(|e| {
                    let mut stats = self.stats.write();
                    stats.failed_preloads += 1;
                    format!("Failed to allocate slots: {}", e)
                })?;
        }

        let v = combined.clone();
        kv_cache.write_kv(request_id, layer, 0, &combined, &v)?;

        let elapsed = start_time.elapsed().as_micros() as f64;
        {
            let mut stats = self.stats.write();
            stats.successful_preloads += 1;
            stats.total_preloaded_tokens += total_rows;
            stats.avg_preload_latency_us =
                (stats.avg_preload_latency_us * (stats.successful_preloads - 1) as f64 + elapsed)
                    / stats.successful_preloads as f64;
        }

        Ok(total_rows)
    }

    /// 按查询预加载记忆
    ///
    /// 根据查询向量从长期记忆中检索相关记忆并预加载
    pub fn preload_by_query(
        &self,
        query: &Array2<f32>,
        request_id: u64,
        layer: usize,
        top_k: usize,
    ) -> Result<usize, String> {
        let start_time = std::time::Instant::now();

        let kv_cache_guard = self.kv_cache.read();
        let kv_cache = kv_cache_guard
            .as_ref()
            .ok_or_else(|| "KV Cache not attached".to_string())?;

        let long_term = self.long_term_memory.read();
        let long_term_mem = long_term
            .as_ref()
            .ok_or_else(|| "Long term memory not attached".to_string())?;

        let results = {
            let mem_guard = long_term_mem.read();
            mem_guard.search(query, top_k)
        };

        if results.is_empty() {
            {
                let mut stats = self.stats.write();
                stats.cache_misses += 1;
            }
            return Ok(0);
        }

        {
            let mut stats = self.stats.write();
            stats.cache_hits += 1;
        }

        let memories: Vec<Array2<f32>> = {
            let mem_guard = long_term_mem.read();
            results
                .iter()
                .filter_map(|(idx, _)| mem_guard.items.get(*idx).map(|item| item.data.clone()))
                .collect()
        };

        if memories.is_empty() {
            return Ok(0);
        }

        let total_rows: usize = memories.iter().map(|m| m.nrows()).sum();
        let dim = memories[0].ncols();

        let mut combined = Array2::zeros((total_rows, dim));
        let mut offset = 0;
        for data in &memories {
            let rows = data.nrows();
            for i in 0..rows {
                for j in 0..dim {
                    combined[[offset + i, j]] = data[[i, j]];
                }
            }
            offset += rows;
        }

        let mut kv_cache = kv_cache.write();

        if !kv_cache.contains_request(request_id) {
            kv_cache
                .allocate_slots(request_id, total_rows)
                .map_err(|e| {
                    let mut stats = self.stats.write();
                    stats.failed_preloads += 1;
                    format!("Failed to allocate slots: {}", e)
                })?;
        }

        let v = combined.clone();
        kv_cache.write_kv(request_id, layer, 0, &combined, &v)?;

        let elapsed = start_time.elapsed().as_micros() as f64;
        {
            let mut stats = self.stats.write();
            stats.successful_preloads += 1;
            stats.total_preloaded_tokens += total_rows;
            stats.avg_preload_latency_us =
                (stats.avg_preload_latency_us * (stats.successful_preloads - 1) as f64 + elapsed)
                    / stats.successful_preloads as f64;
        }

        Ok(total_rows)
    }

    /// 添加预加载任务
    pub fn add_preload_task(
        &self,
        source_level: MemoryLevel,
        target_request_id: u64,
        target_layer: usize,
        priority: f32,
    ) -> u64 {
        let task_id = self.task_counter.fetch_add(1, Ordering::Relaxed);

        let task = PreloadTask {
            task_id,
            source_level,
            target_request_id,
            target_layer,
            priority,
            status: PreloadStatus::Pending,
        };

        let mut queue = self.task_queue.write();

        if queue.len() >= self.config.max_preload_count {
            queue.retain(|t| t.status != PreloadStatus::Completed);
        }

        if self.config.enable_priority {
            let pos = queue
                .iter()
                .position(|t| t.priority < priority)
                .unwrap_or(queue.len());
            queue.insert(pos, task);
        } else {
            queue.push(task);
        }

        task_id
    }

    /// 执行下一个预加载任务
    pub fn execute_next_task(&self) -> Option<Result<usize, String>> {
        let mut queue = self.task_queue.write();

        let task_idx = queue
            .iter()
            .position(|t| t.status == PreloadStatus::Pending)?;

        let task = &mut queue[task_idx];
        task.status = PreloadStatus::InProgress;
        let task_clone = task.clone();

        drop(queue);

        let result = self.preload_memories(
            task_clone.source_level,
            task_clone.target_request_id,
            task_clone.target_layer,
            self.config.batch_size,
        );

        let mut queue = self.task_queue.write();
        if let Some(t) = queue.get_mut(task_idx) {
            t.status = if result.is_ok() {
                PreloadStatus::Completed
            } else {
                PreloadStatus::Failed
            };
        }

        Some(result)
    }

    /// 批量执行预加载任务
    pub fn execute_batch(&self, batch_size: usize) -> Vec<Result<usize, String>> {
        (0..batch_size)
            .filter_map(|_| self.execute_next_task())
            .collect()
    }

    /// 获取待处理任务数量
    pub fn pending_task_count(&self) -> usize {
        self.task_queue
            .read()
            .iter()
            .filter(|t| t.status == PreloadStatus::Pending)
            .count()
    }

    /// 清空任务队列
    pub fn clear_tasks(&self) {
        self.task_queue.write().clear();
    }

    /// 获取统计信息
    pub fn stats(&self) -> PreloadStats {
        self.stats.read().clone()
    }

    /// 重置统计
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write();
        *stats = PreloadStats::default();
    }
}

impl Default for CachePreloader {
    fn default() -> Self {
        Self::new(PreloaderConfig::default())
    }
}

/// 回写统计信息
#[derive(Debug, Clone, Default)]
pub struct WritebackStats {
    /// 成功回写次数
    pub successful_writebacks: u64,
    /// 失败回写次数
    pub failed_writebacks: u64,
    /// 回写的总token数
    pub total_writeback_tokens: usize,
    /// 提取次数
    pub extraction_count: u64,
    /// 平均回写延迟（微秒）
    pub avg_writeback_latency_us: f64,
}

/// 回写器配置
#[derive(Debug, Clone)]
pub struct WriterConfig {
    /// 最大回写批次大小
    pub max_batch_size: usize,
    /// 是否启用重要性计算
    pub compute_importance: bool,
    /// 回写超时（毫秒）
    pub writeback_timeout_ms: u64,
    /// 是否自动清理已回写的 KV Cache
    pub auto_cleanup: bool,
}

impl Default for WriterConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            compute_importance: true,
            writeback_timeout_ms: 5000,
            auto_cleanup: true,
        }
    }
}

/// KV Cache 到记忆的回写器
///
/// 负责将 KV Cache 中的数据回写到记忆层，
/// 支持重要性计算和自动清理
#[derive(Debug)]
pub struct CacheWriter {
    /// 关联的 KV Cache
    kv_cache: RwLock<Option<Arc<RwLock<PagedKVCache>>>>,
    /// 短期记忆引用
    short_term_memory: RwLock<Option<Arc<RwLock<ShortTermMemory>>>>,
    /// 长期记忆引用
    long_term_memory: RwLock<Option<Arc<RwLock<LongTermMemory>>>>,
    /// 统计信息
    stats: RwLock<WritebackStats>,
    /// 配置
    config: WriterConfig,
}

impl CacheWriter {
    /// 创建新的回写器
    pub fn new(config: WriterConfig) -> Self {
        Self {
            kv_cache: RwLock::new(None),
            short_term_memory: RwLock::new(None),
            long_term_memory: RwLock::new(None),
            stats: RwLock::new(WritebackStats::default()),
            config,
        }
    }

    /// 关联 KV Cache
    pub fn attach_kv_cache(&self, kv_cache: Arc<RwLock<PagedKVCache>>) {
        let mut cache = self.kv_cache.write();
        *cache = Some(kv_cache);
    }

    /// 关联短期记忆
    pub fn attach_short_term_memory(&self, memory: Arc<RwLock<ShortTermMemory>>) {
        let mut mem = self.short_term_memory.write();
        *mem = Some(memory);
    }

    /// 关联长期记忆
    pub fn attach_long_term_memory(&self, memory: Arc<RwLock<LongTermMemory>>) {
        let mut mem = self.long_term_memory.write();
        *mem = Some(memory);
    }

    /// 回写 KV Cache 到记忆层
    ///
    /// 将指定请求的 KV Cache 数据回写到目标记忆层
    pub fn writeback_to_memory(
        &self,
        request_id: u64,
        layer: usize,
        target_level: MemoryLevel,
    ) -> Result<usize, String> {
        let start_time = std::time::Instant::now();

        let kv_cache_guard = self.kv_cache.read();
        let kv_cache = kv_cache_guard
            .as_ref()
            .ok_or_else(|| "KV Cache not attached".to_string())?;

        let (k, _v) = {
            let cache = kv_cache.read();
            cache
                .read_kv(request_id, layer)
                .ok_or_else(|| format!("No KV data for request {} layer {}", request_id, layer))?
        };

        let rows = k.nrows();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let importance = if self.config.compute_importance {
            Self::compute_importance(&k)
        } else {
            1.0
        };

        match target_level {
            MemoryLevel::ShortTerm => {
                let short_term = self.short_term_memory.read();
                let mem = short_term
                    .as_ref()
                    .ok_or_else(|| "Short term memory not attached".to_string())?;
                let mut mem_guard = mem.write();
                mem_guard.write(k, timestamp, Some(request_id));
            }
            MemoryLevel::LongTerm => {
                let long_term = self.long_term_memory.read();
                let mem = long_term
                    .as_ref()
                    .ok_or_else(|| "Long term memory not attached".to_string())?;
                let key = format!("kv_{}_{}", request_id, layer);
                let mut mem_guard = mem.write();
                mem_guard.write(key, k, timestamp, importance);
            }
            MemoryLevel::Instant => {
                return Err("Cannot writeback to instant memory".to_string());
            }
        }

        if self.config.auto_cleanup {
            let mut cache = kv_cache.write();
            cache.free_request(&request_id);
        }

        let elapsed = start_time.elapsed().as_micros() as f64;
        {
            let mut stats = self.stats.write();
            stats.successful_writebacks += 1;
            stats.total_writeback_tokens += rows;
            stats.avg_writeback_latency_us = (stats.avg_writeback_latency_us
                * (stats.successful_writebacks - 1) as f64
                + elapsed)
                / stats.successful_writebacks as f64;
        }

        Ok(rows)
    }

    /// 提取 KV Cache 数据到记忆
    ///
    /// 提取指定范围的 KV Cache 数据并存储到记忆层
    pub fn extract_kv_to_memory(
        &self,
        request_id: u64,
        layer: usize,
        start: usize,
        length: usize,
        target_level: MemoryLevel,
    ) -> Result<usize, String> {
        let start_time = std::time::Instant::now();

        let kv_cache_guard = self.kv_cache.read();
        let kv_cache = kv_cache_guard
            .as_ref()
            .ok_or_else(|| "KV Cache not attached".to_string())?;

        let (k, _v) = {
            let cache = kv_cache.read();
            cache
                .read_kv_range(request_id, layer, start, length)
                .ok_or_else(|| {
                    format!(
                        "No KV data for request {} layer {} range {}-{}",
                        request_id,
                        layer,
                        start,
                        start + length
                    )
                })?
        };

        let rows = k.nrows();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let importance = if self.config.compute_importance {
            Self::compute_importance(&k)
        } else {
            1.0
        };

        match target_level {
            MemoryLevel::ShortTerm => {
                let short_term = self.short_term_memory.read();
                let mem = short_term
                    .as_ref()
                    .ok_or_else(|| "Short term memory not attached".to_string())?;
                let mut mem_guard = mem.write();
                mem_guard.write(k, timestamp, Some(request_id));
            }
            MemoryLevel::LongTerm => {
                let long_term = self.long_term_memory.read();
                let mem = long_term
                    .as_ref()
                    .ok_or_else(|| "Long term memory not attached".to_string())?;
                let key = format!("kv_{}_{}_{}_{}", request_id, layer, start, length);
                let mut mem_guard = mem.write();
                mem_guard.write(key, k, timestamp, importance);
            }
            MemoryLevel::Instant => {
                return Err("Cannot extract to instant memory".to_string());
            }
        }

        let elapsed = start_time.elapsed().as_micros() as f64;
        {
            let mut stats = self.stats.write();
            stats.extraction_count += 1;
            stats.total_writeback_tokens += rows;
            stats.avg_writeback_latency_us =
                (stats.avg_writeback_latency_us * stats.extraction_count as f64 + elapsed)
                    / (stats.extraction_count + 1) as f64;
        }

        Ok(rows)
    }

    /// 批量回写
    pub fn writeback_batch(
        &self,
        requests: Vec<(u64, usize, MemoryLevel)>,
    ) -> Vec<Result<usize, String>> {
        requests
            .into_iter()
            .map(|(req_id, layer, level)| self.writeback_to_memory(req_id, layer, level))
            .collect()
    }

    /// 计算数据重要性
    fn compute_importance(data: &Array2<f32>) -> f32 {
        if data.is_empty() {
            return 0.0;
        }

        let sum: f32 = data.iter().map(|v| v.abs()).sum();
        let count = data.len() as f32;
        let mean = sum / count;

        let variance: f32 = data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / count;

        let normalized_variance = (variance.sqrt() / (mean.abs() + 1e-6)).min(1.0);
        let sparsity = data.iter().filter(|v| v.abs() < 1e-6).count() as f32 / count;

        normalized_variance * 0.6 + (1.0 - sparsity) * 0.4
    }

    /// 获取统计信息
    pub fn stats(&self) -> WritebackStats {
        self.stats.read().clone()
    }

    /// 重置统计
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write();
        *stats = WritebackStats::default();
    }
}

impl Default for CacheWriter {
    fn default() -> Self {
        Self::new(WriterConfig::default())
    }
}

/// 共享内存块
///
/// 引用计数由 `Bytes` 内部管理，无需额外计数
#[derive(Debug, Clone)]
pub struct SharedMemoryBlock {
    /// 块ID
    pub block_id: usize,
    /// 数据（Bytes 内部已有引用计数）
    pub data: Bytes,
    /// 数据形状
    pub shape: (usize, usize),
    /// 创建时间戳
    pub timestamp: u64,
}

impl SharedMemoryBlock {
    /// 创建新的共享内存块
    pub fn new(block_id: usize, data: Bytes, shape: (usize, usize), timestamp: u64) -> Self {
        Self {
            block_id,
            data,
            shape,
            timestamp,
        }
    }

    /// 获取数据大小（字节）
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// 转换为 Array2
    pub fn to_array(&self) -> Array2<f32> {
        let (rows, cols) = self.shape;
        let mut array = Array2::zeros((rows, cols));

        for i in 0..rows {
            for j in 0..cols {
                let offset = (i * cols + j) * std::mem::size_of::<f32>();
                if offset + 4 <= self.data.len() {
                    if let Ok(bytes) = self.data[offset..offset + 4].try_into() {
                        array[[i, j]] = f32::from_le_bytes(bytes);
                    }
                }
            }
        }

        array
    }
}

/// 内存池统计信息
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// 总分配块数
    pub total_blocks: usize,
    /// 活跃块数
    pub active_blocks: usize,
    /// 总内存使用（字节）
    pub total_memory_bytes: usize,
    /// 零拷贝传输次数
    pub zero_copy_transfers: u64,
    /// 缓存命中次数
    pub cache_hits: u64,
    /// 缓存未命中次数
    pub cache_misses: u64,
}

impl PoolStats {
    /// 计算缓存命中率
    pub fn hit_rate(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            return 0.0;
        }
        self.cache_hits as f32 / total as f32
    }

    /// 计算内存利用率
    pub fn utilization(&self) -> f32 {
        if self.total_blocks == 0 {
            return 0.0;
        }
        self.active_blocks as f32 / self.total_blocks as f32
    }
}

/// 内存池配置
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// 最大块数量
    pub max_blocks: usize,
    /// 单块最大大小（字节）
    pub max_block_size: usize,
    /// 是否启用缓存
    pub enable_cache: bool,
    /// 缓存过期时间（秒）
    pub cache_ttl_seconds: u64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_blocks: 10000,
            max_block_size: 1024 * 1024,
            enable_cache: true,
            cache_ttl_seconds: 3600,
        }
    }
}

/// 共享内存池
///
/// 提供零拷贝内存共享和高效数据传输
/// 使用 `Arc<SharedMemoryBlock>` 管理块，引用计数由 `Arc` 统一管理
#[derive(Debug)]
pub struct SharedMemoryPool {
    /// 内存块存储
    blocks: RwLock<HashMap<usize, Arc<SharedMemoryBlock>>>,
    /// 统计信息
    stats: RwLock<PoolStats>,
    /// 配置
    config: PoolConfig,
    /// 块计数器
    block_counter: AtomicUsize,
    /// 总内存使用
    total_memory: AtomicUsize,
}

impl SharedMemoryPool {
    /// 创建新的共享内存池
    pub fn new(config: PoolConfig) -> Self {
        Self {
            blocks: RwLock::new(HashMap::new()),
            stats: RwLock::new(PoolStats::default()),
            config,
            block_counter: AtomicUsize::new(0),
            total_memory: AtomicUsize::new(0),
        }
    }

    /// 分配共享内存块
    ///
    /// 分配指定大小的共享内存块，返回块ID
    pub fn allocate_shared(&self, data: &Array2<f32>) -> Result<usize, String> {
        let shape = (data.nrows(), data.ncols());
        let byte_size = shape.0 * shape.1 * std::mem::size_of::<f32>();

        if byte_size > self.config.max_block_size {
            return Err(format!(
                "Block size {} exceeds maximum {}",
                byte_size, self.config.max_block_size
            ));
        }

        let current_memory = self.total_memory.load(Ordering::Relaxed);
        if current_memory + byte_size > self.config.max_blocks * self.config.max_block_size {
            self.evict_lru();
        }

        let mut bytes = Vec::with_capacity(byte_size);
        for row in data.rows() {
            for &val in row.iter() {
                bytes.extend_from_slice(&val.to_le_bytes());
            }
        }

        let block_id = self.block_counter.fetch_add(1, Ordering::Relaxed);
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let block = SharedMemoryBlock::new(block_id, Bytes::from(bytes), shape, timestamp);

        {
            let mut blocks = self.blocks.write();
            blocks.insert(block_id, Arc::new(block));
        }

        self.total_memory.fetch_add(byte_size, Ordering::Relaxed);

        {
            let mut stats = self.stats.write();
            stats.total_blocks += 1;
            stats.active_blocks += 1;
            stats.total_memory_bytes += byte_size;
        }

        Ok(block_id)
    }

    /// 分配共享内存块（从 ZeroCopyBuffer）
    pub fn allocate_from_buffer(&self, buffer: ZeroCopyBuffer) -> Result<usize, String> {
        let shape = buffer.shape();
        let byte_size = buffer.memory_size();

        if byte_size > self.config.max_block_size {
            return Err(format!(
                "Block size {} exceeds maximum {}",
                byte_size, self.config.max_block_size
            ));
        }

        let current_memory = self.total_memory.load(Ordering::Relaxed);
        if current_memory + byte_size > self.config.max_blocks * self.config.max_block_size {
            self.evict_lru();
        }

        let block_id = self.block_counter.fetch_add(1, Ordering::Relaxed);

        let block = SharedMemoryBlock::new(
            block_id,
            buffer.as_bytes().clone(),
            shape,
            buffer.timestamp(),
        );

        {
            let mut blocks = self.blocks.write();
            blocks.insert(block_id, Arc::new(block));
        }

        self.total_memory.fetch_add(byte_size, Ordering::Relaxed);

        {
            let mut stats = self.stats.write();
            stats.total_blocks += 1;
            stats.active_blocks += 1;
            stats.total_memory_bytes += byte_size;
        }

        Ok(block_id)
    }

    /// 零拷贝传输
    ///
    /// 从源块传输数据到目标，无需数据拷贝
    /// 返回 `Arc<SharedMemoryBlock>`，引用计数由 `Arc` 管理
    pub fn zero_copy_transfer(&self, source_block_id: usize) -> Option<Arc<SharedMemoryBlock>> {
        let blocks = self.blocks.read();
        let block = match blocks.get(&source_block_id) {
            Some(b) => b,
            None => {
                drop(blocks);
                let mut stats = self.stats.write();
                stats.cache_misses += 1;
                return None;
            }
        };

        let cloned = Arc::clone(block);

        drop(blocks);

        {
            let mut stats = self.stats.write();
            stats.zero_copy_transfers += 1;
            stats.cache_hits += 1;
        }

        Some(cloned)
    }

    /// 批量零拷贝传输
    pub fn zero_copy_batch(&self, block_ids: &[usize]) -> Vec<Option<Arc<SharedMemoryBlock>>> {
        block_ids
            .iter()
            .map(|&id| self.zero_copy_transfer(id))
            .collect()
    }

    /// 获取块数据
    pub fn get_block(&self, block_id: usize) -> Option<Arc<SharedMemoryBlock>> {
        let blocks = self.blocks.read();
        blocks.get(&block_id).cloned()
    }

    /// 获取块数据为 Array2
    pub fn get_block_as_array(&self, block_id: usize) -> Option<Array2<f32>> {
        let blocks = self.blocks.read();
        blocks.get(&block_id).map(|b| b.to_array())
    }

    /// 释放块
    pub fn release_block(&self, block_id: usize) -> bool {
        let mut blocks = self.blocks.write();

        if let Some(block) = blocks.remove(&block_id) {
            let size = block.size();
            self.total_memory.fetch_sub(size, Ordering::Relaxed);

            {
                let mut stats = self.stats.write();
                stats.active_blocks -= 1;
                stats.total_memory_bytes -= size;
            }

            true
        } else {
            false
        }
    }

    /// 批量释放块
    pub fn release_batch(&self, block_ids: &[usize]) -> usize {
        block_ids
            .iter()
            .filter(|&&id| self.release_block(id))
            .count()
    }

    /// 驱逐最近最少使用的块
    ///
    /// 仅驱逐没有被外部引用的块（strong_count == 1 表示只有池持有）
    fn evict_lru(&self) {
        let mut blocks = self.blocks.write();

        if blocks.is_empty() {
            return;
        }

        let candidates: Vec<(usize, u64)> = blocks
            .iter()
            .filter(|(_, b)| Arc::strong_count(*b) == 1)
            .map(|(id, b)| (*id, b.timestamp))
            .collect();

        if candidates.is_empty() {
            return;
        }

        let oldest = candidates
            .into_iter()
            .min_by_key(|(_, ts)| *ts)
            .map(|(id, _)| id);

        if let Some(id) = oldest {
            if let Some(block) = blocks.remove(&id) {
                let size = block.size();
                self.total_memory.fetch_sub(size, Ordering::Relaxed);

                let mut stats = self.stats.write();
                stats.active_blocks -= 1;
                stats.total_memory_bytes -= size;
            }
        }
    }

    /// 清理过期块（仅清理无外部引用的）
    pub fn cleanup_expired(&self) -> usize {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let threshold = current_time.saturating_sub(self.config.cache_ttl_seconds);

        let mut blocks = self.blocks.write();
        let to_remove: Vec<usize> = blocks
            .iter()
            .filter(|(_, b)| b.timestamp < threshold && Arc::strong_count(*b) == 1)
            .map(|(id, _)| *id)
            .collect();

        let removed = to_remove.len();
        let mut freed_bytes = 0usize;

        for id in to_remove {
            if let Some(block) = blocks.remove(&id) {
                freed_bytes += block.size();
            }
        }

        self.total_memory.fetch_sub(freed_bytes, Ordering::Relaxed);

        {
            let mut stats = self.stats.write();
            stats.active_blocks -= removed;
            stats.total_memory_bytes -= freed_bytes;
        }

        removed
    }

    /// 清空所有块
    pub fn clear(&self) {
        let mut blocks = self.blocks.write();
        blocks.clear();

        self.total_memory.store(0, Ordering::Relaxed);

        let mut stats = self.stats.write();
        stats.active_blocks = 0;
        stats.total_memory_bytes = 0;
    }

    /// 获取活跃块数量
    pub fn active_block_count(&self) -> usize {
        self.blocks.read().len()
    }

    /// 获取总内存使用
    pub fn memory_usage(&self) -> usize {
        self.total_memory.load(Ordering::Relaxed)
    }

    /// 获取统计信息
    pub fn stats(&self) -> PoolStats {
        self.stats.read().clone()
    }

    /// 重置统计
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write();
        stats.zero_copy_transfers = 0;
        stats.cache_hits = 0;
        stats.cache_misses = 0;
    }
}

impl Default for SharedMemoryPool {
    fn default() -> Self {
        Self::new(PoolConfig::default())
    }
}

/// CacheBridge 集成管理器
///
/// 统一管理映射器、预加载器、回写器和内存池
#[derive(Debug)]
pub struct CacheBridge {
    /// 映射器
    pub mapper: CacheMapper,
    /// 预加载器
    pub preloader: CachePreloader,
    /// 回写器
    pub writer: CacheWriter,
    /// 共享内存池
    pub memory_pool: SharedMemoryPool,
    /// 配置
    config: CacheBridgeConfig,
}

/// CacheBridge 配置
#[derive(Debug, Clone, Default)]
pub struct CacheBridgeConfig {
    /// 映射器配置
    pub mapper_config: MapperConfig,
    /// 预加载器配置
    pub preloader_config: PreloaderConfig,
    /// 回写器配置
    pub writer_config: WriterConfig,
    /// 内存池配置
    pub pool_config: PoolConfig,
}

impl CacheBridge {
    /// 创建新的 CacheBridge
    pub fn new(config: CacheBridgeConfig) -> Self {
        Self {
            mapper: CacheMapper::new(config.mapper_config.clone()),
            preloader: CachePreloader::new(config.preloader_config.clone()),
            writer: CacheWriter::new(config.writer_config.clone()),
            memory_pool: SharedMemoryPool::new(config.pool_config.clone()),
            config,
        }
    }

    /// 关联所有组件到 KV Cache
    pub fn attach_kv_cache(&self, kv_cache: Arc<RwLock<PagedKVCache>>) {
        self.mapper.attach_kv_cache(Arc::clone(&kv_cache));
        self.preloader.attach_kv_cache(Arc::clone(&kv_cache));
        self.writer.attach_kv_cache(kv_cache);
    }

    /// 关联瞬时记忆
    pub fn attach_instant_memory(&self, instant_memory: Arc<InstantMemory>) {
        self.mapper.attach_instant_memory(instant_memory);
    }

    /// 关联短期记忆
    pub fn attach_short_term_memory(&self, short_term: Arc<RwLock<ShortTermMemory>>) {
        self.preloader
            .attach_short_term_memory(Arc::clone(&short_term));
        self.writer.attach_short_term_memory(short_term);
    }

    /// 关联长期记忆
    pub fn attach_long_term_memory(&self, long_term: Arc<RwLock<LongTermMemory>>) {
        self.preloader
            .attach_long_term_memory(Arc::clone(&long_term));
        self.writer.attach_long_term_memory(long_term);
    }

    /// 完整数据流：瞬时记忆 -> KV Cache -> 长期记忆
    pub fn full_flow(
        &self,
        memory_index: usize,
        request_id: u64,
        layer: usize,
    ) -> Result<usize, String> {
        let _mapping = self
            .mapper
            .map_instant_to_kv(memory_index, request_id, layer, 0)?;

        let tokens = self
            .writer
            .writeback_to_memory(request_id, layer, MemoryLevel::LongTerm)?;

        Ok(tokens)
    }

    /// 获取综合统计信息
    pub fn combined_stats(&self) -> CombinedStats {
        CombinedStats {
            mapping: self.mapper.stats(),
            preload: self.preloader.stats(),
            writeback: self.writer.stats(),
            pool: self.memory_pool.stats(),
        }
    }

    /// 重置所有统计
    pub fn reset_all_stats(&self) {
        self.mapper.reset_stats();
        self.preloader.reset_stats();
        self.writer.reset_stats();
        self.memory_pool.reset_stats();
    }
}

impl Default for CacheBridge {
    fn default() -> Self {
        Self::new(CacheBridgeConfig::default())
    }
}

/// 综合统计信息
#[derive(Debug, Clone)]
pub struct CombinedStats {
    /// 映射统计
    pub mapping: MappingStats,
    /// 预加载统计
    pub preload: PreloadStats,
    /// 回写统计
    pub writeback: WritebackStats,
    /// 内存池统计
    pub pool: PoolStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_kv_cache() -> Arc<RwLock<PagedKVCache>> {
        Arc::new(RwLock::new(PagedKVCache::with_capacity(100, 16)))
    }

    fn create_test_instant_memory() -> Arc<InstantMemory> {
        Arc::new(InstantMemory::default())
    }

    fn create_test_short_term_memory() -> Arc<RwLock<ShortTermMemory>> {
        Arc::new(RwLock::new(ShortTermMemory::default()))
    }

    fn create_test_long_term_memory() -> Arc<RwLock<LongTermMemory>> {
        Arc::new(RwLock::new(LongTermMemory::default()))
    }

    #[test]
    fn test_cache_mapper_creation() {
        let mapper = CacheMapper::default();
        assert_eq!(mapper.mapping_count(), 0);
    }

    #[test]
    fn test_cache_mapper_map_instant_to_kv() {
        let kv_cache = create_test_kv_cache();
        let instant_memory = create_test_instant_memory();

        let data = Array2::zeros((16, 128));
        instant_memory.write(data, 1);

        let mapper = CacheMapper::default();
        mapper.attach_kv_cache(kv_cache);
        mapper.attach_instant_memory(instant_memory);

        let result = mapper.map_instant_to_kv(0, 1, 0, 0);
        assert!(result.is_ok());

        let entry = result.unwrap();
        assert_eq!(entry.request_id, 1);
        assert_eq!(entry.layer, 0);

        assert_eq!(mapper.mapping_count(), 1);
    }

    #[test]
    fn test_cache_mapper_sync_to_kv_cache() {
        let kv_cache = create_test_kv_cache();
        let instant_memory = create_test_instant_memory();

        for i in 0..3 {
            let data = Array2::zeros((16, 128));
            instant_memory.write(data, i as u64);
        }

        let mapper = CacheMapper::default();
        mapper.attach_kv_cache(kv_cache);
        mapper.attach_instant_memory(instant_memory);

        let result = mapper.sync_to_kv_cache(1, 0);
        assert!(result.is_ok());

        let tokens = result.unwrap();
        assert_eq!(tokens, 48);
    }

    #[test]
    fn test_cache_mapper_stats() {
        let kv_cache = create_test_kv_cache();
        let instant_memory = create_test_instant_memory();

        let data = Array2::zeros((16, 128));
        instant_memory.write(data, 1);

        let mapper = CacheMapper::default();
        mapper.attach_kv_cache(kv_cache);
        mapper.attach_instant_memory(instant_memory);

        let _ = mapper.map_instant_to_kv(0, 1, 0, 0);

        let stats = mapper.stats();
        assert_eq!(stats.successful_mappings, 1);
        assert!(stats.total_mapped_bytes > 0);
    }

    #[test]
    fn test_cache_preloader_creation() {
        let preloader = CachePreloader::default();
        assert_eq!(preloader.pending_task_count(), 0);
    }

    #[test]
    fn test_cache_preloader_preload_memories() {
        let kv_cache = create_test_kv_cache();
        let short_term = create_test_short_term_memory();

        {
            let mut stm = short_term.write();
            let data = Array2::zeros((16, 128));
            stm.write(data, 1, None);
        }

        let preloader = CachePreloader::default();
        preloader.attach_kv_cache(kv_cache);
        preloader.attach_short_term_memory(short_term);

        let result = preloader.preload_memories(MemoryLevel::ShortTerm, 1, 0, 10);
        assert!(result.is_ok());

        let tokens = result.unwrap();
        assert_eq!(tokens, 16);
    }

    #[test]
    fn test_cache_preloader_preload_by_query() {
        let kv_cache = create_test_kv_cache();
        let long_term = create_test_long_term_memory();

        {
            let mut ltm = long_term.write();
            let data = Array2::zeros((16, 128));
            ltm.write("key1".to_string(), data, 1, 1.0);
        }

        let preloader = CachePreloader::default();
        preloader.attach_kv_cache(kv_cache);
        preloader.attach_long_term_memory(long_term);

        let query = Array2::zeros((16, 128));
        let result = preloader.preload_by_query(&query, 1, 0, 5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cache_preloader_task_queue() {
        let preloader = CachePreloader::default();

        let task_id = preloader.add_preload_task(MemoryLevel::ShortTerm, 1, 0, 1.0);
        assert_eq!(task_id, 0);
        assert_eq!(preloader.pending_task_count(), 1);

        preloader.add_preload_task(MemoryLevel::LongTerm, 2, 0, 0.5);
        assert_eq!(preloader.pending_task_count(), 2);

        preloader.clear_tasks();
        assert_eq!(preloader.pending_task_count(), 0);
    }

    #[test]
    fn test_cache_writer_creation() {
        let writer = CacheWriter::default();
        let stats = writer.stats();
        assert_eq!(stats.successful_writebacks, 0);
    }

    #[test]
    fn test_cache_writer_writeback_to_memory() {
        let kv_cache = create_test_kv_cache();
        let short_term = create_test_short_term_memory();

        {
            let mut cache = kv_cache.write();
            let _ = cache.allocate_slots(1, 16);
            let k = Array2::ones((16, 128));
            let v = Array2::ones((16, 128));
            let _ = cache.write_kv(1, 0, 0, &k, &v);
        }

        let writer = CacheWriter::default();
        writer.attach_kv_cache(kv_cache);
        writer.attach_short_term_memory(short_term);

        let result = writer.writeback_to_memory(1, 0, MemoryLevel::ShortTerm);
        assert!(result.is_ok());

        let tokens = result.unwrap();
        assert_eq!(tokens, 16);

        let stats = writer.stats();
        assert_eq!(stats.successful_writebacks, 1);
    }

    #[test]
    fn test_cache_writer_extract_kv_to_memory() {
        let kv_cache = create_test_kv_cache();
        let long_term = create_test_long_term_memory();

        {
            let mut cache = kv_cache.write();
            let _ = cache.allocate_slots(1, 16);
            let k = Array2::ones((16, 128));
            let v = Array2::ones((16, 128));
            let _ = cache.write_kv(1, 0, 0, &k, &v);
        }

        let writer = CacheWriter::default();
        writer.attach_kv_cache(kv_cache);
        writer.attach_long_term_memory(long_term);

        let result = writer.extract_kv_to_memory(1, 0, 0, 8, MemoryLevel::LongTerm);
        assert!(result.is_ok());

        let tokens = result.unwrap();
        assert_eq!(tokens, 8);

        let stats = writer.stats();
        assert_eq!(stats.extraction_count, 1);
    }

    #[test]
    fn test_shared_memory_block() {
        let data = Array2::from_shape_fn((4, 8), |(i, j)| (i * 8 + j) as f32);
        let buffer = ZeroCopyBuffer::from_array(&data, 123);

        let block = SharedMemoryBlock::new(
            0,
            buffer.as_bytes().clone(),
            buffer.shape(),
            buffer.timestamp(),
        );

        assert_eq!(block.block_id, 0);
        assert_eq!(block.shape, (4, 8));

        let recovered = block.to_array();
        for i in 0..4 {
            for j in 0..8 {
                assert!((recovered[[i, j]] - data[[i, j]]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_shared_memory_pool_creation() {
        let pool = SharedMemoryPool::default();
        assert_eq!(pool.active_block_count(), 0);
        assert_eq!(pool.memory_usage(), 0);
    }

    #[test]
    fn test_shared_memory_pool_allocate() {
        let pool = SharedMemoryPool::default();

        let data = Array2::from_shape_fn((16, 128), |(i, j)| (i + j) as f32);
        let result = pool.allocate_shared(&data);

        assert!(result.is_ok());
        let block_id = result.unwrap();

        assert_eq!(pool.active_block_count(), 1);
        assert!(pool.memory_usage() > 0);

        let block = pool.get_block(block_id);
        assert!(block.is_some());
    }

    #[test]
    fn test_shared_memory_pool_zero_copy() {
        let pool = SharedMemoryPool::default();

        let data = Array2::from_shape_fn((16, 128), |(i, j)| (i + j) as f32);
        let block_id = pool.allocate_shared(&data).unwrap();

        let transferred = pool.zero_copy_transfer(block_id);
        assert!(transferred.is_some());

        let stats = pool.stats();
        assert_eq!(stats.zero_copy_transfers, 1);
        assert_eq!(stats.cache_hits, 1);
    }

    #[test]
    fn test_shared_memory_pool_release() {
        let pool = SharedMemoryPool::default();

        let data = Array2::zeros((16, 128));
        let block_id = pool.allocate_shared(&data).unwrap();

        assert_eq!(pool.active_block_count(), 1);

        let released = pool.release_block(block_id);
        assert!(released);
        assert_eq!(pool.active_block_count(), 0);
    }

    #[test]
    fn test_shared_memory_pool_batch_operations() {
        let pool = SharedMemoryPool::default();

        let data = Array2::zeros((16, 128));
        let ids: Vec<usize> = (0..5)
            .map(|_| pool.allocate_shared(&data).unwrap())
            .collect();

        assert_eq!(pool.active_block_count(), 5);

        let transferred = pool.zero_copy_batch(&ids);
        assert_eq!(transferred.len(), 5);
        assert!(transferred.iter().all(|b| b.is_some()));

        let released = pool.release_batch(&ids);
        assert_eq!(released, 5);
        assert_eq!(pool.active_block_count(), 0);
    }

    #[test]
    fn test_cache_bridge_creation() {
        let bridge = CacheBridge::default();
        assert_eq!(bridge.mapper.mapping_count(), 0);
        assert_eq!(bridge.preloader.pending_task_count(), 0);
        assert_eq!(bridge.memory_pool.active_block_count(), 0);
    }

    #[test]
    fn test_cache_bridge_attach_components() {
        let bridge = CacheBridge::default();
        let kv_cache = create_test_kv_cache();
        let instant_memory = create_test_instant_memory();
        let short_term = create_test_short_term_memory();
        let long_term = create_test_long_term_memory();

        bridge.attach_kv_cache(kv_cache);
        bridge.attach_short_term_memory(short_term);
        bridge.attach_long_term_memory(long_term);
        bridge.attach_instant_memory(instant_memory);
    }

    #[test]
    fn test_cache_bridge_combined_stats() {
        let bridge = CacheBridge::default();
        let stats = bridge.combined_stats();

        assert_eq!(stats.mapping.successful_mappings, 0);
        assert_eq!(stats.preload.successful_preloads, 0);
        assert_eq!(stats.writeback.successful_writebacks, 0);
        assert_eq!(stats.pool.active_blocks, 0);
    }

    #[test]
    fn test_mapping_stats_success_rate() {
        let mut stats = MappingStats::default();
        assert_eq!(stats.success_rate(), 0.0);

        stats.successful_mappings = 8;
        stats.failed_mappings = 2;
        assert!((stats.success_rate() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_preload_stats_hit_rate() {
        let mut stats = PreloadStats::default();
        assert_eq!(stats.hit_rate(), 0.0);

        stats.cache_hits = 7;
        stats.cache_misses = 3;
        assert!((stats.hit_rate() - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_pool_stats_utilization() {
        let mut stats = PoolStats::default();
        assert_eq!(stats.utilization(), 0.0);

        stats.total_blocks = 10;
        stats.active_blocks = 7;
        assert!((stats.utilization() - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_mapper_config_default() {
        let config = MapperConfig::default();
        assert_eq!(config.max_mappings, 10000);
        assert!(config.enable_zero_copy);
        assert!(config.auto_sync);
    }

    #[test]
    fn test_preloader_config_default() {
        let config = PreloaderConfig::default();
        assert_eq!(config.max_preload_count, 1000);
        assert_eq!(config.batch_size, 32);
        assert!(config.enable_priority);
    }

    #[test]
    fn test_writer_config_default() {
        let config = WriterConfig::default();
        assert_eq!(config.max_batch_size, 64);
        assert!(config.compute_importance);
        assert!(config.auto_cleanup);
    }

    #[test]
    fn test_pool_config_default() {
        let config = PoolConfig::default();
        assert_eq!(config.max_blocks, 10000);
        assert_eq!(config.max_block_size, 1024 * 1024);
        assert!(config.enable_cache);
    }

    #[test]
    fn test_eviction_skips_blocks_with_external_refs() {
        let config = PoolConfig {
            max_blocks: 2,
            max_block_size: 64,
            enable_cache: true,
            cache_ttl_seconds: 3600,
        };
        let pool = SharedMemoryPool::new(config);

        let data1 = Array2::zeros((2, 2));
        let data2 = Array2::zeros((2, 2));
        let data3 = Array2::zeros((2, 2));
        let data4 = Array2::zeros((2, 2));

        let id1 = pool.allocate_shared(&data1).unwrap();
        let id2 = pool.allocate_shared(&data2).unwrap();

        assert_eq!(pool.active_block_count(), 2);

        let external_ref = pool.zero_copy_transfer(id1).unwrap();

        let id3 = pool.allocate_shared(&data3).unwrap();

        assert!(
            pool.get_block(id1).is_some(),
            "Block 1 should still exist (has external ref)"
        );
        assert!(
            pool.get_block(id2).is_some() || pool.get_block(id3).is_some(),
            "At least one other block should exist"
        );

        drop(external_ref);

        let id4 = pool.allocate_shared(&data4).unwrap();

        let has_id1 = pool.get_block(id1).is_some();
        let has_id2 = pool.get_block(id2).is_some();
        let has_id3 = pool.get_block(id3).is_some();
        let has_id4 = pool.get_block(id4).is_some();

        assert!(
            has_id1 || has_id2 || has_id3 || has_id4,
            "At least one block should exist"
        );
        assert!(has_id4, "Block 4 should exist");
    }

    #[test]
    fn test_arc_strong_count_tracking() {
        let pool = SharedMemoryPool::default();

        let data = Array2::zeros((4, 4));
        let block_id = pool.allocate_shared(&data).unwrap();

        {
            let blocks = pool.blocks.read();
            let block = blocks.get(&block_id).unwrap();
            assert_eq!(std::sync::Arc::strong_count(block), 1);
        }

        let external1 = pool.zero_copy_transfer(block_id).unwrap();
        {
            let blocks = pool.blocks.read();
            let block = blocks.get(&block_id).unwrap();
            assert_eq!(std::sync::Arc::strong_count(block), 2);
        }

        let external2 = pool.zero_copy_transfer(block_id).unwrap();
        {
            let blocks = pool.blocks.read();
            let block = blocks.get(&block_id).unwrap();
            assert_eq!(std::sync::Arc::strong_count(block), 3);
        }

        drop(external1);
        {
            let blocks = pool.blocks.read();
            let block = blocks.get(&block_id).unwrap();
            assert_eq!(std::sync::Arc::strong_count(block), 2);
        }

        drop(external2);
        {
            let blocks = pool.blocks.read();
            let block = blocks.get(&block_id).unwrap();
            assert_eq!(std::sync::Arc::strong_count(block), 1);
        }
    }

    #[test]
    fn test_cleanup_expired_skips_external_refs() {
        let config = PoolConfig {
            max_blocks: 100,
            max_block_size: 1024,
            enable_cache: true,
            cache_ttl_seconds: 0,
        };
        let pool = SharedMemoryPool::new(config);

        let data = Array2::zeros((4, 4));
        let block_id = pool.allocate_shared(&data).unwrap();

        let _external_ref = pool.zero_copy_transfer(block_id).unwrap();

        let removed = pool.cleanup_expired();
        assert_eq!(removed, 0, "Should not remove block with external ref");
        assert!(
            pool.get_block(block_id).is_some(),
            "Block should still exist"
        );
    }
}
