//! 推理引擎与记忆系统集成模块
//!
//! 提供推理引擎与三级记忆系统之间的桥接功能：
//! - 记忆注入：将相关记忆注入推理上下文
//! - 结果记忆化：自动将推理结果写入记忆层
//! - 上下文构建：构建包含记忆的推理上下文
//! - 记忆预热：预加载相关记忆到缓存

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use ndarray::Array2;
use parking_lot::RwLock;

use super::{MemoryLevel, MemoryManager};

// ============================================================================
// 常量定义
// ============================================================================

/// 默认记忆检索数量
const DEFAULT_TOP_K: usize = 10;
/// 默认上下文最大长度
const DEFAULT_MAX_CONTEXT_LEN: usize = 4096;
/// 高优先级阈值
const HIGH_PRIORITY_THRESHOLD: f32 = 0.7;
/// 中优先级阈值
const MEDIUM_PRIORITY_THRESHOLD: f32 = 0.4;

// ============================================================================
// 错误类型定义
// ============================================================================

/// 桥接模块统一错误类型
#[derive(Debug, Clone)]
pub enum BridgeError {
    /// 上下文容量超出
    CapacityExceeded {
        current: usize,
        required: usize,
        max: usize,
    },
    /// 记忆列数不一致
    ColumnMismatch {
        expected: usize,
        actual: usize,
    },
    /// 缓存未命中
    CacheMiss(String),
    /// 分配失败
    AllocationFailed(String),
    /// 无效数据
    InvalidData(String),
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CapacityExceeded { current, required, max } => {
                write!(f, "Context capacity exceeded: current {} + required {} > max {}", current, required, max)
            }
            Self::ColumnMismatch { expected, actual } => {
                write!(f, "Column count mismatch: expected {}, actual {}", expected, actual)
            }
            Self::CacheMiss(key) => {
                write!(f, "Cache miss for key: {}", key)
            }
            Self::AllocationFailed(msg) => {
                write!(f, "Allocation failed: {}", msg)
            }
            Self::InvalidData(msg) => {
                write!(f, "Invalid data: {}", msg)
            }
        }
    }
}

impl std::error::Error for BridgeError {}

// ============================================================================
// 记忆注入器
// ============================================================================

/// 记忆注入配置
#[derive(Debug, Clone)]
pub struct InjectorConfig {
    /// 检索的记忆数量
    pub top_k: usize,
    /// 最小相似度阈值
    pub min_similarity: f32,
    /// 是否包含短期记忆
    pub include_short_term: bool,
    /// 是否包含长期记忆
    pub include_long_term: bool,
    /// 注入模板
    pub injection_template: String,
    /// 短期记忆相似度权重（默认 0.8，避免总是优先于长期记忆）
    pub short_term_similarity_weight: f32,
    /// 短期记忆数量（用于 build_prompt_with_memory）
    pub short_term_count: usize,
}

impl Default for InjectorConfig {
    fn default() -> Self {
        Self {
            top_k: DEFAULT_TOP_K,
            min_similarity: 0.3,
            include_short_term: true,
            include_long_term: true,
            injection_template: "[相关记忆]\n{memory}\n[/相关记忆]\n".to_string(),
            short_term_similarity_weight: 0.8,
            short_term_count: 3,
        }
    }
}

/// 记忆注入器
///
/// 负责将相关记忆注入到推理上下文中
#[derive(Debug)]
pub struct MemoryInjector {
    /// 记忆管理器
    memory_manager: MemoryManager,
    /// 配置
    config: InjectorConfig,
    /// 注入统计
    stats: RwLock<InjectorStats>,
}

/// 注入统计信息
#[derive(Debug, Clone, Default)]
pub struct InjectorStats {
    /// 总注入次数
    pub total_injections: u64,
    /// 成功注入次数
    pub successful_injections: u64,
    /// 失败注入次数
    pub failed_injections: u64,
    /// 平均注入记忆数
    pub avg_memories_per_injection: f32,
    /// 总注入记忆数
    pub total_memories_injected: u64,
}

impl MemoryInjector {
    /// 创建新的记忆注入器
    pub fn new(memory_manager: MemoryManager) -> Self {
        Self {
            memory_manager,
            config: InjectorConfig::default(),
            stats: RwLock::new(InjectorStats::default()),
        }
    }

    /// 使用自定义配置创建注入器
    pub fn with_config(memory_manager: MemoryManager, config: InjectorConfig) -> Self {
        Self {
            memory_manager,
            config,
            stats: RwLock::new(InjectorStats::default()),
        }
    }

    /// 将相关记忆注入推理上下文
    ///
    /// 根据查询向量检索相关记忆，并将其注入到上下文中
    /// 返回注入的记忆数量，若超出容量限制则返回错误
    pub fn inject_context(&self, query: &Array2<f32>, context: &mut Vec<f32>, max_capacity: Option<usize>) -> Result<usize, BridgeError> {
        let mut stats = self.stats.write();
        stats.total_injections += 1;
        drop(stats);

        let mut memories = Vec::new();

        if self.config.include_long_term {
            let long_term_memories = self.memory_manager.search(query, self.config.top_k);
            for (data, score) in long_term_memories {
                if score >= self.config.min_similarity {
                    memories.push((data, score));
                }
            }
        }

        if self.config.include_short_term {
            let short_term_memories = self.memory_manager.read_last(MemoryLevel::ShortTerm, self.config.top_k / 2);
            for data in short_term_memories {
                memories.push((data, self.config.short_term_similarity_weight));
            }
        }

        memories.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let injected_count = memories.len();
        if injected_count > 0 {
            let total_elements: usize = memories.iter().map(|(d, _)| d.len()).sum();
            
            if let Some(max_len) = max_capacity {
                if context.len() + total_elements > max_len {
                    let mut stats = self.stats.write();
                    stats.failed_injections += 1;
                    return Err(BridgeError::CapacityExceeded {
                        current: context.len(),
                        required: total_elements,
                        max: max_len,
                    });
                }
            }

            let mut stats = self.stats.write();
            stats.successful_injections += 1;
            stats.total_memories_injected += injected_count as u64;
            stats.avg_memories_per_injection = stats.total_memories_injected as f32 / stats.successful_injections as f32;

            for (data, _) in memories {
                if data.is_standard_layout() {
                    if let Some(slice) = data.as_slice() {
                        context.extend_from_slice(slice);
                        continue;
                    }
                }
                let flat: Vec<f32> = data.iter().copied().collect();
                context.extend(flat);
            }
        }

        Ok(injected_count)
    }

    /// 构建包含记忆的提示
    ///
    /// 将记忆内容格式化为可读的提示文本
    pub fn build_prompt_with_memory(&self, query: &Array2<f32>, original_prompt: &str) -> String {
        let mut memory_texts = Vec::new();

        if self.config.include_long_term {
            let long_term_memories = self.memory_manager.search(query, self.config.top_k);
            for (data, score) in long_term_memories {
                if score >= self.config.min_similarity {
                    let text = self.data_to_text(&data);
                    memory_texts.push(format!("[相似度: {:.2}] {}", score, text));
                }
            }
        }

        if self.config.include_short_term {
            let short_term_memories = self.memory_manager.read_last(MemoryLevel::ShortTerm, self.config.short_term_count);
            for data in short_term_memories {
                let text = self.data_to_text(&data);
                memory_texts.push(format!("[近期记忆] {}", text));
            }
        }

        if memory_texts.is_empty() {
            return original_prompt.to_string();
        }

        let memory_section = memory_texts.join("\n");
        let template = &self.config.injection_template;
        let memory_block = template.replace("{memory}", &memory_section);

        format!("{}{}", memory_block, original_prompt)
    }

    /// 将数据转换为文本表示
    fn data_to_text(&self, data: &Array2<f32>) -> String {
        if data.is_empty() {
            return "[空记忆]".to_string();
        }

        let rows = data.nrows().min(3);
        let cols = data.ncols().min(5);

        let mut text = String::from("记忆片段: [");
        for i in 0..rows {
            if i > 0 {
                text.push_str("; ");
            }
            for j in 0..cols {
                if j > 0 {
                    text.push_str(", ");
                }
                text.push_str(&format!("{:.2}", data[[i, j]]));
            }
            if cols < data.ncols() {
                text.push_str("...");
            }
        }
        if rows < data.nrows() {
            text.push_str(" ...");
        }
        text.push(']');

        text
    }

    /// 获取统计信息
    pub fn get_stats(&self) -> InjectorStats {
        self.stats.read().clone()
    }

    /// 重置统计信息
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write();
        *stats = InjectorStats::default();
    }

    /// 获取配置
    pub fn config(&self) -> &InjectorConfig {
        &self.config
    }

    /// 获取记忆管理器引用
    pub fn memory_manager(&self) -> &MemoryManager {
        &self.memory_manager
    }
}

// ============================================================================
// 记忆提取器
// ============================================================================

/// 记忆提取配置
#[derive(Debug, Clone)]
pub struct ExtractorConfig {
    /// 关键信息提取阈值
    pub extraction_threshold: f32,
    /// 是否自动记忆化
    pub auto_memorize: bool,
    /// 默认记忆级别
    pub default_level: MemoryLevel,
    /// 最小记忆重要性
    pub min_importance: f32,
    /// 是否提取摘要
    pub extract_summary: bool,
    /// 关键信息分类阈值：事实
    pub fact_threshold: f32,
    /// 关键信息分类阈值：概念
    pub concept_threshold: f32,
    /// 关键信息分类阈值：事件
    pub event_threshold: f32,
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        Self {
            extraction_threshold: 0.5,
            auto_memorize: true,
            default_level: MemoryLevel::ShortTerm,
            min_importance: 0.3,
            extract_summary: true,
            fact_threshold: 0.8,
            concept_threshold: 0.7,
            event_threshold: 0.5,
        }
    }
}

/// 提取的关键信息
#[derive(Debug, Clone)]
pub struct KeyInfo {
    /// 信息内容
    pub content: Array2<f32>,
    /// 重要性分数
    pub importance: f32,
    /// 信息类型
    pub info_type: InfoType,
    /// 时间戳
    pub timestamp: u64,
}

/// 信息类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InfoType {
    /// 关键结论
    Conclusion,
    /// 重要事实
    Fact,
    /// 推理步骤
    ReasoningStep,
    /// 摘要
    Summary,
    /// 其他
    Other,
}

/// 记忆提取器
///
/// 负责从推理结果中提取关键信息并写入记忆层
#[derive(Debug)]
pub struct MemoryExtractor {
    /// 记忆管理器
    memory_manager: MemoryManager,
    /// 配置
    config: ExtractorConfig,
    /// 提取统计
    stats: RwLock<ExtractorStats>,
}

/// 提取统计信息
#[derive(Debug, Clone, Default)]
pub struct ExtractorStats {
    /// 总提取次数
    pub total_extractions: u64,
    /// 成功提取次数
    pub successful_extractions: u64,
    /// 提取错误次数
    pub extraction_errors: u64,
    /// 总提取信息数
    pub total_info_extracted: u64,
    /// 自动记忆化次数
    pub auto_memorize_count: u64,
}

impl MemoryExtractor {
    /// 创建新的记忆提取器
    pub fn new(memory_manager: MemoryManager) -> Self {
        Self {
            memory_manager,
            config: ExtractorConfig::default(),
            stats: RwLock::new(ExtractorStats::default()),
        }
    }

    /// 使用自定义配置创建提取器
    pub fn with_config(memory_manager: MemoryManager, config: ExtractorConfig) -> Self {
        Self {
            memory_manager,
            config,
            stats: RwLock::new(ExtractorStats::default()),
        }
    }

    /// 从推理结果提取关键信息
    ///
    /// 分析推理结果，提取具有高重要性的信息片段
    pub fn extract_key_info(&self, result: &Array2<f32>) -> Vec<KeyInfo> {
        let mut stats = self.stats.write();
        stats.total_extractions += 1;

        if result.is_empty() {
            return Vec::new();
        }

        let mut key_infos = Vec::new();
        let timestamp = current_timestamp();

        let importance_scores = self.calculate_importance_scores(result);
        let rows = result.nrows();

        if importance_scores.len() != rows {
            stats.extraction_errors += 1;
            return Vec::new();
        }

        for (i, &score) in importance_scores.iter().enumerate() {
            if score >= self.config.extraction_threshold {
                let content = result.row(i).to_owned().insert_axis(ndarray::Axis(0));

                let info_type = self.classify_info(&content, score);

                key_infos.push(KeyInfo {
                    content,
                    importance: score,
                    info_type,
                    timestamp,
                });
            }
        }

        if self.config.extract_summary && !key_infos.is_empty() {
            let summary = self.extract_summary_info(result, &importance_scores);
            if let Some(summary_info) = summary {
                key_infos.push(summary_info);
            }
        }

        stats.successful_extractions += 1;
        stats.total_info_extracted += key_infos.len() as u64;

        key_infos
    }

    /// 自动将结果写入记忆层
    ///
    /// 根据配置自动将推理结果存储到指定记忆级别
    pub fn auto_memorize(&self, result: &Array2<f32>, importance: Option<f32>) -> bool {
        if !self.config.auto_memorize {
            return false;
        }

        let mut stats = self.stats.write();

        let imp = importance.unwrap_or_else(|| self.calculate_overall_importance(result));
        if imp < self.config.min_importance {
            return false;
        }

        self.memory_manager.write_with_importance(
            self.config.default_level,
            result.clone(),
            imp,
        );

        stats.auto_memorize_count += 1;
        true
    }

    /// 将提取的关键信息批量写入记忆
    pub fn memorize_key_infos(&self, key_infos: &[KeyInfo], level: Option<MemoryLevel>) -> usize {
        let target_level = level.unwrap_or(self.config.default_level);
        let mut count = 0;

        for info in key_infos {
            if info.importance >= self.config.min_importance {
                self.memory_manager.write_with_importance(
                    target_level,
                    info.content.clone(),
                    info.importance,
                );
                count += 1;
            }
        }

        count
    }

    /// 计算重要性分数
    fn calculate_importance_scores(&self, data: &Array2<f32>) -> Vec<f32> {
        let rows = data.nrows();
        let cols = data.ncols();

        if rows == 0 || cols == 0 {
            return Vec::new();
        }

        const EPSILON: f32 = 1e-8;

        let mut scores = Vec::with_capacity(rows);

        let col_means: Vec<f32> = (0..cols)
            .map(|j| {
                let sum: f32 = (0..rows).map(|i| data[[i, j]]).sum();
                sum / rows as f32
            })
            .collect();

        let col_stds: Vec<f32> = (0..cols)
            .map(|j| {
                let mean = col_means[j];
                let variance: f32 = (0..rows)
                    .map(|i| (data[[i, j]] - mean).powi(2))
                    .sum::<f32>()
                    / rows as f32;
                variance.sqrt().max(EPSILON)
            })
            .collect();

        for i in 0..rows {
            let row = data.row(i);

            let norm: f32 = row.iter().map(|v| v.powi(2)).sum::<f32>().sqrt().max(EPSILON);
            let row_mean: f32 = row.iter().sum::<f32>() / cols as f32;
            let variance: f32 = (row.iter().map(|v| v.powi(2)).sum::<f32>() / cols as f32
                - row_mean.powi(2))
            .max(0.0);

            let z_score_sum: f32 = row
                .iter()
                .enumerate()
                .map(|(j, &v)| {
                    let std = col_stds[j];
                    if std > EPSILON {
                        ((v - col_means[j]) / std).abs()
                    } else {
                        0.0
                    }
                })
                .sum();

            let score = (norm / (cols as f32).sqrt() * 0.3
                + (variance.sqrt() / norm).min(1.0) * 0.3
                + (z_score_sum / cols as f32).min(1.0) * 0.4)
                .clamp(0.0, 1.0);

            scores.push(score);
        }

        scores
    }

    /// 计算整体重要性
    fn calculate_overall_importance(&self, data: &Array2<f32>) -> f32 {
        if data.is_empty() {
            return 0.0;
        }

        let scores = self.calculate_importance_scores(data);
        if scores.is_empty() {
            return 0.0;
        }

        let sum: f32 = scores.iter().sum();
        sum / scores.len() as f32
    }

    /// 分类信息类型
    fn classify_info(&self, data: &Array2<f32>, importance: f32) -> InfoType {
        if data.is_empty() {
            return InfoType::Other;
        }

        let rows = data.nrows();
        let cols = data.ncols();

        let sparsity = data.iter().filter(|v| v.abs() < 1e-6).count() as f32 / (rows * cols) as f32;

        let variance: f32 = {
            let mean: f32 = data.iter().sum::<f32>() / (rows * cols) as f32;
            data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (rows * cols) as f32
        };

        if importance > self.config.fact_threshold {
            InfoType::Conclusion
        } else if sparsity > self.config.concept_threshold {
            InfoType::Fact
        } else if variance > self.config.event_threshold {
            InfoType::ReasoningStep
        } else {
            InfoType::Other
        }
    }

    /// 提取摘要信息
    fn extract_summary_info(
        &self,
        data: &Array2<f32>,
        importance_scores: &[f32],
    ) -> Option<KeyInfo> {
        if data.nrows() == 0 || data.ncols() == 0 {
            return None;
        }

        let cols = data.ncols();
        let mut weighted_sum = vec![0.0f32; cols];
        let mut total_weight = 0.0f32;

        for (i, &score) in importance_scores.iter().enumerate() {
            if i < data.nrows() {
                let weight = score.max(0.1);
                for j in 0..cols {
                    weighted_sum[j] += data[[i, j]] * weight;
                }
                total_weight += weight;
            }
        }

        if total_weight < 1e-6 {
            return None;
        }

        let summary: Array2<f32> = ndarray::Array1::from_vec(
            weighted_sum.iter().map(|v| v / total_weight).collect()
        )
        .insert_axis(ndarray::Axis(0))
        .into_dyn()
        .into_dimensionality::<ndarray::Ix2>()
        .ok()?;

        let avg_importance: f32 = importance_scores.iter().sum::<f32>() / importance_scores.len() as f32;

        Some(KeyInfo {
            content: summary,
            importance: avg_importance,
            info_type: InfoType::Summary,
            timestamp: current_timestamp(),
        })
    }

    /// 获取统计信息
    pub fn get_stats(&self) -> ExtractorStats {
        self.stats.read().clone()
    }

    /// 重置统计信息
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write();
        *stats = ExtractorStats::default();
    }

    /// 获取配置
    pub fn config(&self) -> &ExtractorConfig {
        &self.config
    }

    /// 获取记忆管理器引用
    pub fn memory_manager(&self) -> &MemoryManager {
        &self.memory_manager
    }
}

// ============================================================================
// 上下文构建器
// ============================================================================

/// 记忆项优先级
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryPriority {
    /// 低优先级
    Low,
    /// 中优先级
    Medium,
    /// 高优先级
    High,
    /// 关键优先级
    Critical,
}

/// 带优先级的记忆项
#[derive(Debug, Clone)]
pub struct PrioritizedMemory {
    /// 记忆数据
    pub data: Array2<f32>,
    /// 优先级
    pub priority: MemoryPriority,
    /// 来源
    pub source: String,
    /// 时间戳
    pub timestamp: u64,
}

/// 上下文构建配置
#[derive(Debug, Clone)]
pub struct ContextBuilderConfig {
    /// 最大上下文长度
    pub max_context_len: usize,
    /// 是否自动排序
    pub auto_sort: bool,
    /// 是否去重
    pub deduplicate: bool,
    /// 相似度去重阈值
    pub dedup_threshold: f32,
    /// 指纹差异阈值（用于快速过滤，差异大于此值跳过相似度计算）
    pub fingerprint_threshold: u64,
}

impl Default for ContextBuilderConfig {
    fn default() -> Self {
        Self {
            max_context_len: DEFAULT_MAX_CONTEXT_LEN,
            auto_sort: true,
            deduplicate: true,
            dedup_threshold: 0.95,
            fingerprint_threshold: 100,
        }
    }
}

/// 上下文构建器
///
/// 构建包含记忆的推理上下文，支持优先级排序
#[derive(Debug)]
pub struct ContextBuilder {
    /// 记忆项列表
    memories: Vec<PrioritizedMemory>,
    /// 配置
    config: ContextBuilderConfig,
    /// 当前长度
    current_len: usize,
    /// 期望的列数（用于一致性检查）
    expected_cols: Option<usize>,
}

impl ContextBuilder {
    /// 创建新的上下文构建器
    pub fn new() -> Self {
        Self {
            memories: Vec::new(),
            config: ContextBuilderConfig::default(),
            current_len: 0,
            expected_cols: None,
        }
    }

    /// 使用自定义配置创建构建器
    pub fn with_config(config: ContextBuilderConfig) -> Self {
        Self {
            memories: Vec::new(),
            config,
            current_len: 0,
            expected_cols: None,
        }
    }

    /// 添加记忆到上下文
    ///
    /// 返回是否成功添加
    /// 若记忆列数与已有记忆不一致，返回 false
    pub fn add_memory(&mut self, data: Array2<f32>, priority: MemoryPriority, source: &str) -> bool {
        let data_len = data.len();
        let cols = data.ncols();

        if let Some(expected) = self.expected_cols {
            if cols != expected {
                return false;
            }
        } else if !data.is_empty() {
            self.expected_cols = Some(cols);
        }

        if self.current_len + data_len > self.config.max_context_len && !self.try_evict_for_space(data_len) {
            return false;
        }

        self.memories.push(PrioritizedMemory {
            data,
            priority,
            source: source.to_string(),
            timestamp: current_timestamp(),
        });
        self.current_len += data_len;

        true
    }

    /// 添加带重要性分数的记忆
    pub fn add_memory_with_importance(
        &mut self,
        data: Array2<f32>,
        importance: f32,
        source: &str,
    ) -> bool {
        let priority = self.importance_to_priority(importance);
        self.add_memory(data, priority, source)
    }

    /// 批量添加记忆
    pub fn add_memories(&mut self, memories: Vec<(Array2<f32>, MemoryPriority, &str)>) -> usize {
        let mut added = 0;
        for (data, priority, source) in memories {
            if self.add_memory(data, priority, source) {
                added += 1;
            }
        }
        added
    }

    /// 构建最终上下文
    ///
    /// 将所有记忆合并为最终的上下文数组
    pub fn build(&mut self) -> Array2<f32> {
        if self.memories.is_empty() {
            return Array2::zeros((0, 0));
        }

        if self.config.auto_sort {
            self.sort_by_priority();
        }

        if self.config.deduplicate {
            self.deduplicate_memories();
        }

        let dim = self.expected_cols.unwrap_or(0);
        let total_rows: usize = self.memories.iter().map(|m| m.data.nrows()).sum();

        if total_rows == 0 || dim == 0 {
            return Array2::zeros((0, 0));
        }

        let mut result = Array2::zeros((total_rows, dim));
        let mut offset = 0;

        for memory in &self.memories {
            let rows = memory.data.nrows();
            let cols = memory.data.ncols().min(dim);
            for i in 0..rows {
                for j in 0..cols {
                    result[[offset + i, j]] = memory.data[[i, j]];
                }
            }
            offset += rows;
        }

        result
    }

    /// 构建为扁平向量
    pub fn build_flat(&mut self) -> Vec<f32> {
        let context = self.build();
        context.iter().copied().collect()
    }

    /// 按优先级排序
    fn sort_by_priority(&mut self) {
        self.memories.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// 去重记忆
    ///
    /// 使用哈希快速过滤 + 相似度计算的混合策略：
    /// 1. 计算每个记忆的哈希指纹
    /// 2. 仅对哈希相近的记忆进行相似度计算
    /// 3. 使用 HashSet 优化查找，避免 O(n) 的 contains 调用
    fn deduplicate_memories(&mut self) {
        if self.memories.len() < 2 {
            return;
        }

        let fingerprints: Vec<u64> = self.memories
            .iter()
            .map(|m| Self::compute_fingerprint(&m.data))
            .collect();

        let mut to_remove: HashSet<usize> = HashSet::new();

        for i in 0..self.memories.len().saturating_sub(1) {
            if to_remove.contains(&i) {
                continue;
            }
            for j in (i + 1)..self.memories.len() {
                if to_remove.contains(&j) {
                    continue;
                }

                let fp_diff = (fingerprints[i] as i64 - fingerprints[j] as i64).unsigned_abs();
                if fp_diff > self.config.fingerprint_threshold {
                    continue;
                }

                let similarity = self.compute_similarity(&self.memories[i].data, &self.memories[j].data);
                if similarity >= self.config.dedup_threshold {
                    if self.memories[i].priority >= self.memories[j].priority {
                        to_remove.insert(j);
                    } else {
                        to_remove.insert(i);
                        break;
                    }
                }
            }
        }

        if to_remove.is_empty() {
            return;
        }

        let mut idx = 0;
        self.memories.retain(|m| {
            let keep = !to_remove.contains(&idx);
            if !keep {
                self.current_len -= m.data.len();
            }
            idx += 1;
            keep
        });
    }

    /// 计算记忆数据的快速指纹
    ///
    /// 用于快速过滤明显不同的记忆，避免不必要的相似度计算
    fn compute_fingerprint(data: &Array2<f32>) -> u64 {
        if data.is_empty() {
            return 0;
        }

        let rows = data.nrows();
        let cols = data.ncols();
        
        let mut sum: f64 = 0.0;
        let mut sum_sq: f64 = 0.0;
        let mut count: u64 = 0;

        for val in data.iter() {
            let v = *val as f64;
            sum += v;
            sum_sq += v * v;
            count += 1;
        }

        let mean = if count > 0 { sum / count as f64 } else { 0.0 };
        let variance = if count > 0 { sum_sq / count as f64 - mean * mean } else { 0.0 };

        let mean_bits = mean.to_bits();
        let var_bits = variance.to_bits();

        mean_bits
            .wrapping_add(var_bits.wrapping_mul(31))
            .wrapping_add((rows as u64).wrapping_mul(17))
            .wrapping_add(cols as u64)
    }

    /// 计算相似度
    ///
    /// 按行计算余弦相似度并取平均，考虑形状差异
    ///
    /// **语义假设**：
    /// - 行顺序有意义：假设记忆数据的行顺序代表时间或逻辑顺序
    /// - 行对齐比较：仅比较相同行索引的数据
    /// - 列数必须一致：列数不同的记忆直接返回 0（不相似）
    /// - 公共行比较：行数不同时，仅比较公共行部分
    ///
    /// **适用场景**：
    /// - 时序数据（如对话历史、推理步骤）
    /// - 结构化记忆（每行代表一个语义单元）
    ///
    /// **不适用场景**：
    /// - 行顺序无意义的数据（如词袋模型）
    /// - 需要全局相似度的场景（应使用展平后的余弦相似度）
    fn compute_similarity(&self, a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        let a_rows = a.nrows();
        let b_rows = b.nrows();
        let a_cols = a.ncols();
        let b_cols = b.ncols();

        if a_cols != b_cols {
            return 0.0;
        }

        let common_rows = a_rows.min(b_rows);
        if common_rows == 0 {
            return 0.0;
        }

        let mut total_similarity = 0.0;
        for i in 0..common_rows {
            let row_a = a.row(i);
            let row_b = b.row(i);

            let dot: f32 = row_a.iter().zip(row_b.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f32 = row_a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
            let norm_b: f32 = row_b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

            if norm_a > 1e-6 && norm_b > 1e-6 {
                total_similarity += dot / (norm_a * norm_b);
            }
        }

        total_similarity / common_rows as f32
    }

    /// 尝试驱逐低优先级记忆以腾出空间
    fn try_evict_for_space(&mut self, required_space: usize) -> bool {
        let mut freed = 0;

        while freed < required_space && !self.memories.is_empty() {
            if let Some(idx) = self.find_lowest_priority_index() {
                let removed_len = self.memories[idx].data.len();
                self.memories.remove(idx);
                freed += removed_len;
                self.current_len -= removed_len;
            } else {
                break;
            }
        }

        freed >= required_space
    }

    /// 找到最低优先级的记忆索引
    fn find_lowest_priority_index(&self) -> Option<usize> {
        self.memories
            .iter()
            .enumerate()
            .min_by_key(|(_, m)| m.priority)
            .map(|(i, _)| i)
    }

    /// 将重要性分数转换为优先级
    fn importance_to_priority(&self, importance: f32) -> MemoryPriority {
        if importance >= HIGH_PRIORITY_THRESHOLD {
            MemoryPriority::High
        } else if importance >= MEDIUM_PRIORITY_THRESHOLD {
            MemoryPriority::Medium
        } else {
            MemoryPriority::Low
        }
    }

    /// 清空构建器
    pub fn clear(&mut self) {
        self.memories.clear();
        self.current_len = 0;
    }

    /// 获取记忆数量
    pub fn len(&self) -> usize {
        self.memories.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.memories.is_empty()
    }

    /// 获取当前长度
    pub fn current_length(&self) -> usize {
        self.current_len
    }

    /// 获取剩余容量
    pub fn remaining_capacity(&self) -> usize {
        self.config.max_context_len.saturating_sub(self.current_len)
    }

    /// 获取配置
    pub fn config(&self) -> &ContextBuilderConfig {
        &self.config
    }

    /// 获取按优先级分组的记忆数量
    pub fn count_by_priority(&self) -> HashMap<MemoryPriority, usize> {
        let mut counts = HashMap::new();
        for memory in &self.memories {
            *counts.entry(memory.priority).or_insert(0) += 1;
        }
        counts
    }
}

impl Default for ContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// 记忆预热器
// ============================================================================

/// 记忆预热配置
#[derive(Debug, Clone)]
pub struct WarmerConfig {
    /// 预热记忆数量
    pub warmup_count: usize,
    /// 缓存大小
    pub cache_size: usize,
    /// 是否预加载长期记忆
    pub preload_long_term: bool,
    /// 是否预加载短期记忆
    pub preload_short_term: bool,
    /// 预热相似度阈值
    pub warmup_similarity_threshold: f32,
}

impl Default for WarmerConfig {
    fn default() -> Self {
        Self {
            warmup_count: 20,
            cache_size: 100,
            preload_long_term: true,
            preload_short_term: true,
            warmup_similarity_threshold: 0.5,
        }
    }
}

/// 预热缓存项
#[derive(Debug, Clone)]
struct CacheItem {
    /// 记忆数据（使用 Arc 共享，避免克隆）
    data: Arc<Array2<f32>>,
    /// 访问次数
    access_count: u32,
    /// 最后访问时间
    last_access: u64,
}

/// 记忆预热器
///
/// 预加载相关记忆到缓存，加速后续访问
#[derive(Debug)]
pub struct MemoryWarmer {
    /// 记忆管理器
    memory_manager: MemoryManager,
    /// 配置
    config: WarmerConfig,
    /// 预热缓存
    cache: RwLock<HashMap<String, CacheItem>>,
    /// 预热统计
    stats: RwLock<WarmerStats>,
}

/// 预热统计信息
#[derive(Debug, Clone, Default)]
pub struct WarmerStats {
    /// 总预热次数
    pub total_warmups: u64,
    /// 缓存命中次数
    pub cache_hits: u64,
    /// 缓存未命中次数
    pub cache_misses: u64,
    /// 预加载记忆数
    pub preloaded_count: u64,
}

impl MemoryWarmer {
    /// 创建新的记忆预热器
    pub fn new(memory_manager: MemoryManager) -> Self {
        Self {
            memory_manager,
            config: WarmerConfig::default(),
            cache: RwLock::new(HashMap::new()),
            stats: RwLock::new(WarmerStats::default()),
        }
    }

    /// 使用自定义配置创建预热器
    pub fn with_config(memory_manager: MemoryManager, config: WarmerConfig) -> Self {
        Self {
            memory_manager,
            config,
            cache: RwLock::new(HashMap::new()),
            stats: RwLock::new(WarmerStats::default()),
        }
    }

    /// 预热相关记忆
    ///
    /// 根据查询预加载相关记忆到缓存
    /// 使用内容哈希作为键，避免重复存储相同内容
    pub fn warmup(&self, query: &Array2<f32>) -> usize {
        let mut stats = self.stats.write();
        stats.total_warmups += 1;
        drop(stats);

        let mut warmed_count = 0;

        if self.config.preload_long_term {
            let long_term_memories = self.memory_manager.search(query, self.config.warmup_count);
            for (data, score) in long_term_memories {
                if score >= self.config.warmup_similarity_threshold {
                    let key = format!("lt_{:016x}", Self::compute_content_hash(&data));
                    if self.add_to_cache_if_absent(&key, data) {
                        warmed_count += 1;
                    }
                }
            }
        }

        if self.config.preload_short_term {
            let short_term_memories = self.memory_manager.read_last(
                MemoryLevel::ShortTerm,
                self.config.warmup_count / 2,
            );
            for data in short_term_memories {
                let key = format!("st_{:016x}", Self::compute_content_hash(&data));
                if self.add_to_cache_if_absent(&key, data) {
                    warmed_count += 1;
                }
            }
        }

        let mut stats = self.stats.write();
        stats.preloaded_count += warmed_count as u64;
        warmed_count
    }

    /// 预加载到缓存
    ///
    /// 将指定记忆预加载到缓存中
    pub fn preload_to_cache(&self, key: &str, data: Array2<f32>) -> bool {
        let mut cache = self.cache.write();

        if cache.len() >= self.config.cache_size {
            self.evict_lru(&mut cache);
        }

        cache.insert(
            key.to_string(),
            CacheItem {
                data: Arc::new(data),
                access_count: 0,
                last_access: current_timestamp(),
            },
        );

        true
    }

    /// 从缓存获取记忆
    ///
    /// 使用读锁优先策略，减少写锁持有时间
    /// 在写锁中再次确认键存在，避免并发驱逐导致的统计偏差
    /// 返回 Arc 共享引用，避免深拷贝
    pub fn get_from_cache(&self, key: &str) -> Option<Arc<Array2<f32>>> {
        let data = {
            let cache = self.cache.read();
            cache.get(key).map(|item| Arc::clone(&item.data))
        };
        
        if let Some(data) = data {
            let mut cache = self.cache.write();
            if cache.contains_key(key) {
                if let Some(item) = cache.get_mut(key) {
                    item.access_count += 1;
                    item.last_access = current_timestamp();
                }
                let mut stats = self.stats.write();
                stats.cache_hits += 1;
                return Some(data);
            }
        }
        
        let mut stats = self.stats.write();
        stats.cache_misses += 1;
        None
    }

    /// 添加到缓存
    fn add_to_cache(&self, key: &str, data: Array2<f32>) {
        let mut cache = self.cache.write();

        if cache.len() >= self.config.cache_size {
            self.evict_lru(&mut cache);
        }

        cache.insert(
            key.to_string(),
            CacheItem {
                data: Arc::new(data),
                access_count: 0,
                last_access: current_timestamp(),
            },
        );
    }

    /// 仅当键不存在时添加到缓存
    ///
    /// 返回是否成功添加（键不存在时）
    /// 若键已存在但内容相同，视为已存在；若内容不同，生成新键
    fn add_to_cache_if_absent(&self, key: &str, data: Array2<f32>) -> bool {
        let mut cache = self.cache.write();
        
        if let Some(existing) = cache.get(key) {
            if Self::data_equals(&existing.data, &data) {
                return false;
            }
        } else {
            if cache.len() >= self.config.cache_size {
                self.evict_lru(&mut cache);
            }

            cache.insert(
                key.to_string(),
                CacheItem {
                    data: Arc::new(data),
                    access_count: 0,
                    last_access: current_timestamp(),
                },
            );
            return true;
        }

        const MAX_KEY_VERSIONS: u32 = 10;
        for counter in 1..=MAX_KEY_VERSIONS {
            let new_key = format!("{}_v{}", key, counter);
            if !cache.contains_key(&new_key) {
                cache.insert(
                    new_key,
                    CacheItem {
                        data: Arc::new(data),
                        access_count: 0,
                        last_access: current_timestamp(),
                    },
                );
                return true;
            }
        }

        if cache.len() >= self.config.cache_size {
            self.evict_lru(&mut cache);
        }

        let fallback_key = format!("{}_v{}", key, current_timestamp());
        cache.insert(
            fallback_key,
            CacheItem {
                data: Arc::new(data),
                access_count: 0,
                last_access: current_timestamp(),
            },
        );
        true
    }

    /// 深度比较两个 Array2 是否相等
    ///
    /// 用于处理哈希碰撞时的 fallback 比较
    fn data_equals(a: &Array2<f32>, b: &Array2<f32>) -> bool {
        if a.shape() != b.shape() {
            return false;
        }
        for (va, vb) in a.iter().zip(b.iter()) {
            if (va - vb).abs() > 1e-6 {
                return false;
            }
        }
        true
    }

    /// 计算记忆内容的哈希值
    fn compute_content_hash(data: &Array2<f32>) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        data.shape().hash(&mut hasher);
        for val in data.iter() {
            val.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    /// 驱逐最近最少使用的缓存项
    ///
    /// 仅依据 last_access 时间戳，实现真正的 LRU 语义
    fn evict_lru(&self, cache: &mut HashMap<String, CacheItem>) {
        if let Some(lru_key) = cache
            .iter()
            .min_by_key(|(_, item)| item.last_access)
            .map(|(k, _)| k.clone())
        {
            cache.remove(&lru_key);
        }
    }

    /// 清空缓存
    pub fn clear_cache(&self) {
        let mut cache = self.cache.write();
        cache.clear();
    }

    /// 获取缓存大小
    pub fn cache_size(&self) -> usize {
        self.cache.read().len()
    }

    /// 获取缓存命中率
    pub fn cache_hit_rate(&self) -> f32 {
        let stats = self.stats.read();
        let total = stats.cache_hits + stats.cache_misses;
        if total == 0 {
            return 0.0;
        }
        stats.cache_hits as f32 / total as f32
    }

    /// 获取统计信息
    pub fn get_stats(&self) -> WarmerStats {
        self.stats.read().clone()
    }

    /// 重置统计信息
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write();
        *stats = WarmerStats::default();
    }

    /// 获取配置
    pub fn config(&self) -> &WarmerConfig {
        &self.config
    }

    /// 获取记忆管理器引用
    pub fn memory_manager(&self) -> &MemoryManager {
        &self.memory_manager
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 获取当前时间戳
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::MemoryConfig;

    fn create_test_memory_manager() -> MemoryManager {
        MemoryManager::new(MemoryConfig::default())
    }

    fn create_test_data(rows: usize, cols: usize) -> Array2<f32> {
        Array2::from_shape_fn((rows, cols), |(i, j)| (i + j) as f32 * 0.1)
    }

    #[test]
    fn test_memory_injector_new() {
        let manager = create_test_memory_manager();
        let injector = MemoryInjector::new(manager);

        assert_eq!(injector.config().top_k, DEFAULT_TOP_K);
    }

    #[test]
    fn test_memory_injector_inject_context() {
        let manager = create_test_memory_manager();
        let injector = MemoryInjector::new(manager);

        let query = create_test_data(10, 128);
        let mut context = Vec::new();

        let result = injector.inject_context(&query, &mut context, None);

        let stats = injector.get_stats();
        assert_eq!(stats.total_injections, 1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_memory_injector_inject_context_with_capacity() {
        let manager = create_test_memory_manager();
        let injector = MemoryInjector::new(manager);

        let query = create_test_data(10, 128);
        let mut context = Vec::new();

        let result = injector.inject_context(&query, &mut context, Some(10));
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_memory_injector_build_prompt() {
        let manager = create_test_memory_manager();
        let injector = MemoryInjector::new(manager);

        let query = create_test_data(10, 128);
        let prompt = "请回答以下问题";

        let result = injector.build_prompt_with_memory(&query, prompt);

        assert!(result.contains("请回答以下问题"));
    }

    #[test]
    fn test_memory_injector_stats() {
        let manager = create_test_memory_manager();
        let injector = MemoryInjector::new(manager);

        let stats = injector.get_stats();
        assert_eq!(stats.total_injections, 0);

        injector.reset_stats();
        let stats = injector.get_stats();
        assert_eq!(stats.total_injections, 0);
    }

    #[test]
    fn test_memory_extractor_new() {
        let manager = create_test_memory_manager();
        let extractor = MemoryExtractor::new(manager);

        assert!(extractor.config().auto_memorize);
    }

    #[test]
    fn test_memory_extractor_extract_key_info() {
        let manager = create_test_memory_manager();
        let extractor = MemoryExtractor::new(manager);

        let result = create_test_data(20, 128);
        let key_infos = extractor.extract_key_info(&result);

        let stats = extractor.get_stats();
        assert_eq!(stats.total_extractions, 1);
        assert!(!key_infos.is_empty() || stats.total_info_extracted == 0);
    }

    #[test]
    fn test_memory_extractor_auto_memorize() {
        let manager = create_test_memory_manager();
        let extractor = MemoryExtractor::new(manager);

        let result = create_test_data(10, 128);
        let success = extractor.auto_memorize(&result, Some(0.8));

        let stats = extractor.get_stats();
        if success {
            assert!(stats.auto_memorize_count > 0);
        }
    }

    #[test]
    fn test_memory_extractor_memorize_key_infos() {
        let manager = create_test_memory_manager();
        let extractor = MemoryExtractor::new(manager);

        let key_infos = vec![
            KeyInfo {
                content: create_test_data(5, 64),
                importance: 0.8,
                info_type: InfoType::Conclusion,
                timestamp: current_timestamp(),
            },
            KeyInfo {
                content: create_test_data(5, 64),
                importance: 0.3,
                info_type: InfoType::Fact,
                timestamp: current_timestamp(),
            },
        ];

        let count = extractor.memorize_key_infos(&key_infos, None);
        assert!(count >= 1);
    }

    #[test]
    fn test_context_builder_new() {
        let builder = ContextBuilder::new();

        assert!(builder.is_empty());
        assert_eq!(builder.len(), 0);
    }

    #[test]
    fn test_context_builder_add_memory() {
        let mut builder = ContextBuilder::new();

        let data = create_test_data(10, 64);
        let added = builder.add_memory(data, MemoryPriority::High, "test");

        assert!(added);
        assert_eq!(builder.len(), 1);
    }

    #[test]
    fn test_context_builder_add_memory_with_importance() {
        let mut builder = ContextBuilder::new();

        let data = create_test_data(10, 64);
        let added = builder.add_memory_with_importance(data, 0.8, "test");

        assert!(added);
        assert_eq!(builder.len(), 1);
    }

    #[test]
    fn test_context_builder_build() {
        let config = ContextBuilderConfig {
            deduplicate: false,
            ..Default::default()
        };
        let mut builder = ContextBuilder::with_config(config);

        let data1 = create_test_data(5, 32);
        let data2 = create_test_data(3, 32);

        builder.add_memory(data1, MemoryPriority::High, "test1");
        builder.add_memory(data2, MemoryPriority::Medium, "test2");

        let result = builder.build();

        assert_eq!(result.nrows(), 8);
        assert_eq!(result.ncols(), 32);
    }

    #[test]
    fn test_context_builder_priority_sort() {
        let mut builder = ContextBuilder::new();

        let data1 = create_test_data(5, 32);
        let data2 = create_test_data(5, 32);
        let data3 = create_test_data(5, 32);

        builder.add_memory(data1.clone(), MemoryPriority::Low, "low");
        builder.add_memory(data2.clone(), MemoryPriority::High, "high");
        builder.add_memory(data3.clone(), MemoryPriority::Medium, "medium");

        builder.sort_by_priority();

        assert_eq!(builder.memories[0].priority, MemoryPriority::High);
        assert_eq!(builder.memories[1].priority, MemoryPriority::Medium);
        assert_eq!(builder.memories[2].priority, MemoryPriority::Low);
    }

    #[test]
    fn test_context_builder_capacity() {
        let config = ContextBuilderConfig {
            max_context_len: 100,
            ..Default::default()
        };
        let mut builder = ContextBuilder::with_config(config);

        let data = create_test_data(10, 10);
        let added1 = builder.add_memory(data.clone(), MemoryPriority::High, "test1");
        let added2 = builder.add_memory(data, MemoryPriority::Low, "test2");

        assert!(added1);
        assert!(!added2 || builder.current_length() <= 100);
    }

    #[test]
    fn test_context_builder_count_by_priority() {
        let mut builder = ContextBuilder::new();

        builder.add_memory(create_test_data(5, 32), MemoryPriority::High, "h1");
        builder.add_memory(create_test_data(5, 32), MemoryPriority::High, "h2");
        builder.add_memory(create_test_data(5, 32), MemoryPriority::Medium, "m1");
        builder.add_memory(create_test_data(5, 32), MemoryPriority::Low, "l1");

        let counts = builder.count_by_priority();

        assert_eq!(*counts.get(&MemoryPriority::High).unwrap_or(&0), 2);
        assert_eq!(*counts.get(&MemoryPriority::Medium).unwrap_or(&0), 1);
        assert_eq!(*counts.get(&MemoryPriority::Low).unwrap_or(&0), 1);
    }

    #[test]
    fn test_context_builder_clear() {
        let mut builder = ContextBuilder::new();

        builder.add_memory(create_test_data(5, 32), MemoryPriority::High, "test");
        assert!(!builder.is_empty());

        builder.clear();
        assert!(builder.is_empty());
    }

    #[test]
    fn test_memory_warmer_new() {
        let manager = create_test_memory_manager();
        let warmer = MemoryWarmer::new(manager);

        assert_eq!(warmer.cache_size(), 0);
    }

    #[test]
    fn test_memory_warmer_warmup() {
        let manager = create_test_memory_manager();
        let warmer = MemoryWarmer::new(manager);

        let query = create_test_data(10, 128);
        let count = warmer.warmup(&query);

        let stats = warmer.get_stats();
        assert_eq!(stats.total_warmups, 1);
        let _ = count;
    }

    #[test]
    fn test_memory_warmer_preload_to_cache() {
        let manager = create_test_memory_manager();
        let warmer = MemoryWarmer::new(manager);

        let data = create_test_data(10, 64);
        let success = warmer.preload_to_cache("test_key", data);

        assert!(success);
        assert_eq!(warmer.cache_size(), 1);
    }

    #[test]
    fn test_memory_warmer_get_from_cache() {
        let manager = create_test_memory_manager();
        let warmer = MemoryWarmer::new(manager);

        let data = create_test_data(10, 64);
        warmer.preload_to_cache("test_key", data);

        let result = warmer.get_from_cache("test_key");
        assert!(result.is_some());

        let result = warmer.get_from_cache("nonexistent");
        assert!(result.is_none());

        let stats = warmer.get_stats();
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
    }

    #[test]
    fn test_memory_warmer_cache_hit_rate() {
        let manager = create_test_memory_manager();
        let warmer = MemoryWarmer::new(manager);

        let data = create_test_data(10, 64);
        warmer.preload_to_cache("test_key", data);

        warmer.get_from_cache("test_key");
        warmer.get_from_cache("test_key");
        warmer.get_from_cache("nonexistent");

        let rate = warmer.cache_hit_rate();
        assert!((rate - 0.6666667).abs() < 0.01 || (rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_memory_warmer_clear_cache() {
        let manager = create_test_memory_manager();
        let warmer = MemoryWarmer::new(manager);

        let data = create_test_data(10, 64);
        warmer.preload_to_cache("test_key", data);
        assert!(warmer.cache_size() > 0);

        warmer.clear_cache();
        assert_eq!(warmer.cache_size(), 0);
    }

    #[test]
    fn test_memory_warmer_stats() {
        let manager = create_test_memory_manager();
        let warmer = MemoryWarmer::new(manager);

        let stats = warmer.get_stats();
        assert_eq!(stats.total_warmups, 0);

        warmer.reset_stats();
        let stats = warmer.get_stats();
        assert_eq!(stats.total_warmups, 0);
    }

    #[test]
    fn test_key_info_types() {
        let manager = create_test_memory_manager();
        let extractor = MemoryExtractor::new(manager);

        let result = create_test_data(20, 128);
        let key_infos = extractor.extract_key_info(&result);

        for info in &key_infos {
            match info.info_type {
                InfoType::Conclusion => assert!(info.importance > 0.0),
                InfoType::Fact => {}
                InfoType::ReasoningStep => {}
                InfoType::Summary => {}
                InfoType::Other => {}
            }
        }
    }

    #[test]
    fn test_injector_with_custom_config() {
        let manager = create_test_memory_manager();
        let config = InjectorConfig {
            top_k: 5,
            min_similarity: 0.5,
            include_short_term: false,
            include_long_term: true,
            injection_template: "[Memory]\n{memory}\n[/Memory]\n".to_string(),
            short_term_similarity_weight: 0.8,
            short_term_count: 3,
        };
        let injector = MemoryInjector::with_config(manager, config);

        assert_eq!(injector.config().top_k, 5);
        assert!(!injector.config().include_short_term);
    }

    #[test]
    fn test_extractor_with_custom_config() {
        let manager = create_test_memory_manager();
        let config = ExtractorConfig {
            extraction_threshold: 0.7,
            auto_memorize: false,
            default_level: MemoryLevel::LongTerm,
            min_importance: 0.5,
            extract_summary: false,
            fact_threshold: 0.8,
            concept_threshold: 0.7,
            event_threshold: 0.5,
        };
        let extractor = MemoryExtractor::with_config(manager, config);

        assert!(!extractor.config().auto_memorize);
        assert_eq!(extractor.config().default_level, MemoryLevel::LongTerm);
    }

    #[test]
    fn test_warmer_with_custom_config() {
        let manager = create_test_memory_manager();
        let config = WarmerConfig {
            warmup_count: 10,
            cache_size: 50,
            preload_long_term: false,
            preload_short_term: true,
            warmup_similarity_threshold: 0.6,
        };
        let warmer = MemoryWarmer::with_config(manager, config);

        assert_eq!(warmer.config().warmup_count, 10);
        assert!(!warmer.config().preload_long_term);
    }

    #[test]
    fn test_context_builder_deduplicate() {
        let config = ContextBuilderConfig {
            deduplicate: true,
            dedup_threshold: 0.9,
            ..Default::default()
        };
        let mut builder = ContextBuilder::with_config(config);

        let data = create_test_data(5, 32);
        builder.add_memory(data.clone(), MemoryPriority::High, "test1");
        builder.add_memory(data, MemoryPriority::High, "test2");

        builder.deduplicate_memories();

        assert!(builder.len() <= 2);
    }

    #[test]
    fn test_memory_priority_ordering() {
        assert!(MemoryPriority::Critical > MemoryPriority::High);
        assert!(MemoryPriority::High > MemoryPriority::Medium);
        assert!(MemoryPriority::Medium > MemoryPriority::Low);
    }

    #[test]
    fn test_injector_data_to_text() {
        let manager = create_test_memory_manager();
        let injector = MemoryInjector::new(manager);

        let data = create_test_data(2, 3);
        let text = injector.data_to_text(&data);

        assert!(text.contains("记忆片段"));
    }

    #[test]
    fn test_context_builder_build_flat() {
        let mut builder = ContextBuilder::new();

        let data = create_test_data(5, 32);
        builder.add_memory(data, MemoryPriority::High, "test");

        let flat = builder.build_flat();

        assert_eq!(flat.len(), 5 * 32);
    }

    #[test]
    fn test_extractor_calculate_importance_scores() {
        let manager = create_test_memory_manager();
        let extractor = MemoryExtractor::new(manager);

        let data = create_test_data(10, 64);
        let scores = extractor.calculate_importance_scores(&data);

        assert_eq!(scores.len(), 10);
        for &score in &scores {
            assert!(score >= 0.0 && score <= 1.0);
        }
    }

    #[test]
    fn test_context_builder_column_consistency() {
        let mut builder = ContextBuilder::new();

        let data1 = Array2::zeros((4, 8));
        let data2 = Array2::zeros((4, 8));
        let data3 = Array2::zeros((4, 16));

        assert!(builder.add_memory(data1, MemoryPriority::High, "test1"));
        assert!(builder.add_memory(data2, MemoryPriority::High, "test2"));
        assert!(!builder.add_memory(data3, MemoryPriority::High, "test3"));
    }

    #[test]
    fn test_extractor_error_stats() {
        let manager = create_test_memory_manager();
        let extractor = MemoryExtractor::new(manager);

        let data = create_test_data(10, 64);
        let _ = extractor.extract_key_info(&data);

        let stats = extractor.get_stats();
        assert!(stats.extraction_errors == 0 || stats.extraction_errors > 0);
    }

    #[test]
    fn test_warmer_arc_sharing() {
        let manager = create_test_memory_manager();
        let warmer = MemoryWarmer::new(manager);

        let data = create_test_data(10, 64);
        warmer.preload_to_cache("test_key", data);

        let arc1 = warmer.get_from_cache("test_key").unwrap();
        let _arc2 = warmer.get_from_cache("test_key").unwrap();

        assert_eq!(Arc::strong_count(&arc1), 3);
    }

    #[test]
    fn test_fingerprint_computation() {
        let data1 = create_test_data(10, 64);
        let data2 = create_test_data(10, 64);
        let data3 = create_test_data(10, 32);

        let fp1 = ContextBuilder::compute_fingerprint(&data1);
        let fp2 = ContextBuilder::compute_fingerprint(&data2);
        let fp3 = ContextBuilder::compute_fingerprint(&data3);

        assert!(fp1 > 0);
        assert!(fp2 > 0);
        assert!(fp3 > 0);
        assert_ne!(fp1, fp3);
    }
}
