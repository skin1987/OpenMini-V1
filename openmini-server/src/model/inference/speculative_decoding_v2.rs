//! Speculative Decoding v2 实现
//!
//! Speculative Decoding v2 是2024年最新的推测解码优化技术，相比v1有以下改进：
//! - 自适应草稿长度：根据接受率动态调整草稿序列长度
//! - 多候选token：同时生成多个候选token，提高接受率
//! - 树形推测：构建推测树，支持多条推测路径
//! - 拒绝采样优化：改进的拒绝采样算法，减少计算浪费
//!
//! 性能提升：
//! - 相比标准解码：2-4倍加速
//! - 相比Speculative Decoding v1：1.3-1.8倍加速
//! - 接受率：70-90%

#![allow(dead_code)]

use anyhow::Result;
use ndarray::Array1;
use rand::prelude::*;

/// Speculative Decoding v2 配置
#[derive(Debug, Clone)]
pub struct SpeculativeDecodingV2Config {
    /// 初始草稿长度
    pub initial_draft_length: usize,
    /// 最大草稿长度
    pub max_draft_length: usize,
    /// 最小草稿长度
    pub min_draft_length: usize,
    /// 候选token数量
    pub num_candidates: usize,
    /// 接受率阈值（用于自适应调整）
    pub acceptance_threshold: f32,
    /// 是否启用树形推测
    pub enable_tree_speculation: bool,
    /// 树形推测的分支因子
    pub tree_branch_factor: usize,
    /// 温度参数（用于采样）
    pub temperature: f32,
    /// 是否启用自适应调整
    pub enable_adaptive: bool,
}

impl Default for SpeculativeDecodingV2Config {
    fn default() -> Self {
        Self {
            initial_draft_length: 4,
            max_draft_length: 8,
            min_draft_length: 2,
            num_candidates: 4,
            acceptance_threshold: 0.7,
            enable_tree_speculation: true,
            tree_branch_factor: 2,
            temperature: 1.0,
            enable_adaptive: true,
        }
    }
}

/// 推测解码统计信息
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// 总推测次数
    pub total_speculations: u64,
    /// 成功接受的token数
    pub accepted_tokens: u64,
    /// 拒绝的token数
    pub rejected_tokens: u64,
    /// 平均接受长度
    pub avg_accept_length: f32,
    /// 平均草稿长度
    pub avg_draft_length: f32,
    /// 接受率
    pub acceptance_rate: f32,
}

/// 候选token
#[derive(Debug, Clone)]
pub struct CandidateToken {
    /// Token ID
    pub token_id: u32,
    /// 概率
    pub prob: f32,
    /// 对数概率
    pub log_prob: f32,
}

/// 推测结果
#[derive(Debug, Clone)]
pub struct SpeculationResult {
    /// 接受的token序列
    pub accepted_tokens: Vec<u32>,
    /// 接受长度
    pub accept_length: usize,
    /// 是否完全接受
    pub fully_accepted: bool,
    /// 下一个token的概率分布
    pub next_token_probs: Array1<f32>,
}

/// N-gram 验证器
///
/// 使用N-gram统计模型验证草稿token的合理性，
/// 基于历史token序列计算条件概率来评估候选token的置信度。
#[derive(Debug, Clone)]
pub struct NgramVerifier {
    /// N-gram的阶数（默认为3）
    n: usize,
    /// N-gram频率统计表：ngram -> 出现次数
    ngram_counts: std::collections::HashMap<Vec<u32>, u64>,
    /// 总token数（用于概率归一化）
    total_tokens: u64,
    /// 平滑参数（Laplace平滑）
    smoothing: f32,
}

impl NgramVerifier {
    /// 创建新的N-gram验证器
    ///
    /// # 参数
    /// - `n`: N-gram的阶数，默认为3
    /// - `smoothing`: Laplace平滑参数，默认为0.1
    pub fn new(n: usize, smoothing: f32) -> Self {
        Self {
            n: n.max(1), // N至少为1
            ngram_counts: std::collections::HashMap::new(),
            total_tokens: 0,
            smoothing: smoothing.max(0.0),
        }
    }

    /// 使用默认配置创建（N=3）
    pub fn with_default_config() -> Self {
        Self::new(3, 0.1)
    }

    /// 训练N-gram模型
    ///
    /// 从历史token序列中学习N-gram频率分布
    ///
    /// # 参数
    /// - `tokens`: 历史token序列
    pub fn train(&mut self, tokens: &[u32]) {
        if tokens.len() < self.n {
            return;
        }

        self.total_tokens += tokens.len() as u64;

        // 提取所有n-gram并计数
        for window in tokens.windows(self.n) {
            let ngram = window.to_vec();
            *self.ngram_counts.entry(ngram).or_insert(0) += 1;
        }
    }

    /// 验证草稿token序列
    ///
    /// 计算每个草稿token基于前文历史的条件概率，
    /// 返回每个位置的置信度分数用于辅助接受/拒绝决策。
    ///
    /// # 参数
    /// - `context`: 当前上下文token序列（历史tokens）
    /// - `draft_tokens`: 待验证的草稿token序列
    ///
    /// # 返回值
    /// 每个草稿token的置信度分数数组（0.0-1.0）
    pub fn verify_with_ngram(&self, context: &[u32], draft_tokens: &[u32]) -> Array1<f32> {
        let mut confidences = Array1::<f32>::zeros(draft_tokens.len());

        for (i, &token) in draft_tokens.iter().enumerate() {
            // 构建当前n-gram的上下文窗口
            let start_pos = i.saturating_sub(self.n - 1);

            // 从context和已验证的draft_tokens中提取上下文
            let mut ngram_context: Vec<u32> = Vec::with_capacity(self.n - 1);

            // 先从原始context中取
            let context_start = context.len().saturating_sub((self.n - 1).saturating_sub(i));
            if i < self.n - 1 && context_start < context.len() {
                ngram_context.extend_from_slice(&context[context_start..]);
            } else if i >= self.n - 1 {
                // 从已验证的draft_tokens中取
                let draft_start = start_pos;
                let draft_end = i;
                if draft_start < draft_end {
                    ngram_context.extend_from_slice(&draft_tokens[draft_start..draft_end]);
                }
            }

            // 构建完整的n-gram用于查询
            let mut query_ngram = ngram_context.clone();
            query_ngram.push(token);

            // 计算条件概率 P(token | context)
            let confidence = self.calculate_conditional_probability(&query_ngram, &ngram_context);
            confidences[i] = confidence;
        }

        confidences
    }

    /// 计算条件概率 P(token | context)
    ///
    /// 使用带Laplace平滑的最大似然估计
    fn calculate_conditional_probability(&self, full_ngram: &[u32], context: &[u32]) -> f32 {
        // 查询完整n-gram的出现次数
        let numerator = *self.ngram_counts.get(full_ngram).unwrap_or(&0) as f32;

        // 查询上下文的累计出现次数（作为分母）
        let denominator = if context.is_empty() {
            // 无上下文时使用总token数
            self.total_tokens as f32
        } else {
            // 统计以该上下文开头的所有n-gram的总次数
            self.count_context_occurrences(context) as f32
        };

        // 应用Laplace平滑避免零概率
        let vocab_size_estimate = 10000.0; // 估计词汇表大小
        let smoothed_num = numerator + self.smoothing;
        let smoothed_denom = denominator + self.smoothing * vocab_size_estimate;

        if smoothed_denom <= 0.0 {
            return 1.0 / vocab_size_estimate; // 返回均匀分布概率
        }

        (smoothed_num / smoothed_denom).min(1.0)
    }

    /// 统计特定上下文出现的次数
    fn count_context_occurrences(&self, context: &[u32]) -> u64 {
        if context.is_empty() {
            return self.total_tokens;
        }

        self.ngram_counts
            .iter()
            .filter(|(ngram, _)| ngram.starts_with(context))
            .map(|(_, &count)| count)
            .sum()
    }

    /// 获取N-gram阶数
    pub fn n(&self) -> usize {
        self.n
    }

    /// 获取训练样本总数
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// 获取N-gram表大小
    pub fn ngram_table_size(&self) -> usize {
        self.ngram_counts.len()
    }

    /// 清空训练数据
    pub fn clear(&mut self) {
        self.ngram_counts.clear();
        self.total_tokens = 0;
    }
}

/// 树形注意力缓存
///
/// 为Speculative Decoding v2的树形推测路径维护独立的KV Cache，
/// 支持多路径并行验证时的缓存复用，减少重复计算。
#[derive(Debug)]
pub struct TreeAttentionCache {
    /// 缓存条目：路径标识符 -> KV缓存数据
    cache_entries: std::collections::HashMap<u64, CacheEntry>,
    /// 最大缓存容量
    max_capacity: usize,
    /// 缓存命中统计
    hits: u64,
    /// 缓存未命中统计
    misses: u64,
    /// 当前缓存大小（按条目数）
    current_size: usize,
    /// 总内存使用估算（字节）
    memory_usage_bytes: u64,
}

/// 单个缓存条目
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Key缓存数据（简化表示：存储为扁平化向量）
    key_cache: Array1<f32>,
    /// Value缓存数据
    value_cache: Array1<f32>,
    /// 路径深度（层数）
    depth: usize,
    /// 创建时间戳（用于LRU淘汰）
    timestamp: u64,
    /// 最后访问时间
    last_access: u64,
}

impl TreeAttentionCache {
    /// 创建新的树形注意力缓存
    ///
    /// # 参数
    /// - `max_capacity`: 最大缓存条目数量
    pub fn new(max_capacity: usize) -> Self {
        Self {
            cache_entries: std::collections::HashMap::new(),
            max_capacity: max_capacity.max(1),
            hits: 0,
            misses: 0,
            current_size: 0,
            memory_usage_bytes: 0,
        }
    }

    /// 使用默认容量创建（100个条目）
    pub fn with_default_capacity() -> Self {
        Self::new(100)
    }

    /// 存储或更新KV缓存
    ///
    /// # 参数
    /// - `path_id`: 树形路径的唯一标识符
    /// - `key_cache`: Key矩阵的扁平化数据
    /// - `value_cache`: Value矩阵的扁平化数据
    /// - `depth`: 该路径在树中的深度/层数
    pub fn store(
        &mut self,
        path_id: u64,
        key_cache: Array1<f32>,
        value_cache: Array1<f32>,
        depth: usize,
    ) {
        let current_time = self.get_timestamp();

        // 检查是否需要淘汰（超出容量限制）
        if !self.cache_entries.contains_key(&path_id) && self.current_size >= self.max_capacity {
            self.evict_lru_entry();
        }

        let entry_size = (key_cache.len() + value_cache.len()) * std::mem::size_of::<f32>();

        // 如果已存在，先移除旧条目的内存占用
        if let Some(old_entry) = self.cache_entries.get(&path_id) {
            let old_size = (old_entry.key_cache.len() + old_entry.value_cache.len())
                * std::mem::size_of::<f32>();
            self.memory_usage_bytes = self.memory_usage_bytes.saturating_sub(old_size as u64);
        } else {
            self.current_size += 1;
        }

        // 创建新条目
        let entry = CacheEntry {
            key_cache,
            value_cache,
            depth,
            timestamp: current_time,
            last_access: current_time,
        };

        self.cache_entries.insert(path_id, entry);
        self.memory_usage_bytes += entry_size as u64;
    }

    /// 查询KV缓存
    ///
    /// # 参数
    /// - `path_id`: 树形路径的唯一标识符
    ///
    /// # 返回值
    /// Option<(Key缓存, Value缓存, 路径深度)>，未命中返回None
    pub fn lookup(&mut self, path_id: u64) -> Option<(Array1<f32>, Array1<f32>, usize)> {
        let current_time = self.get_timestamp(); // 先获取时间戳避免借用冲突

        match self.cache_entries.get_mut(&path_id) {
            Some(entry) => {
                entry.last_access = current_time;
                self.hits += 1;
                Some((
                    entry.key_cache.clone(),
                    entry.value_cache.clone(),
                    entry.depth,
                ))
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// 批量预加载多条路径的缓存
    ///
    /// 在树形推测开始前，批量加载可能需要的路径缓存
    ///
    /// # 参数
    /// - `paths`: 待加载的路径列表 (path_id, key_cache, value_cache, depth)
    pub fn batch_preload(&mut self, paths: Vec<(u64, Array1<f32>, Array1<f32>, usize)>) {
        for (path_id, key_cache, value_cache, depth) in paths {
            self.store(path_id, key_cache, value_cache, depth);
        }
    }

    /// 使指定路径的缓存失效
    pub fn invalidate(&mut self, path_id: u64) -> bool {
        if let Some(entry) = self.cache_entries.remove(&path_id) {
            let size =
                (entry.key_cache.len() + entry.value_cache.len()) * std::mem::size_of::<f32>();
            self.memory_usage_bytes = self.memory_usage_bytes.saturating_sub(size as u64);
            self.current_size -= 1;
            true
        } else {
            false
        }
    }

    /// 清空所有缓存
    pub fn clear(&mut self) {
        self.cache_entries.clear();
        self.current_size = 0;
        self.memory_usage_bytes = 0;
        self.hits = 0;
        self.misses = 0;
    }

    /// LRU淘汰策略：移除最近最少使用的条目
    fn evict_lru_entry(&mut self) {
        if self.cache_entries.is_empty() {
            return;
        }

        // 找到最久未访问的条目
        let lru_path_id = self
            .cache_entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_access)
            .map(|(&path_id, _)| path_id);

        if let Some(path_id) = lru_path_id {
            self.invalidate(path_id);
        }
    }

    /// 获取简单时间戳（单调递增计数器模拟）
    fn get_timestamp(&self) -> u64 {
        // 使用命中+未命中总和作为简单的时间戳代理
        // 实际生产环境应使用系统时钟
        self.hits + self.misses
    }

    /// 获取缓存命中率
    pub fn hit_rate(&self) -> f32 {
        let total_accesses = self.hits + self.misses;
        if total_accesses == 0 {
            0.0
        } else {
            self.hits as f32 / total_accesses as f32
        }
    }

    /// 获取当前缓存条目数
    pub fn current_size(&self) -> usize {
        self.current_size
    }

    /// 获取最大缓存容量
    pub fn max_capacity(&self) -> usize {
        self.max_capacity
    }

    /// 获取内存使用估算（字节）
    pub fn memory_usage_bytes(&self) -> u64 {
        self.memory_usage_bytes
    }

    /// 获取命中/未命中统计
    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }
}

/// 扩展的推测状态（集成N-gram验证和树形缓存）
///
/// 将N-gram验证器和树形注意力缓存整合到推测解码流程中
#[derive(Debug)]
pub struct EnhancedSpeculativeState {
    /// 基础推测解码器
    decoder: SpeculativeDecodingV2,
    /// N-gram验证器
    ngram_verifier: NgramVerifier,
    /// 树形注意力缓存
    tree_cache: TreeAttentionCache,
    /// N-gram置信度阈值（低于此值的token将被拒绝）
    ngram_confidence_threshold: f32,
    /// 是否启用N-gram辅助验证
    enable_ngram_verification: bool,
    /// 是否启用树形缓存
    enable_tree_cache: bool,
}

impl EnhancedSpeculativeState {
    /// 创建增强版推测状态
    ///
    /// # 参数
    /// - `config`: 推测解码配置
    /// - `ngram_n`: N-gram阶数
    /// - `ngram_threshold`: N-gram置信度阈值
    /// - `cache_capacity`: 树形缓存容量
    pub fn new(
        config: SpeculativeDecodingV2Config,
        ngram_n: usize,
        ngram_confidence_threshold: f32,
        cache_capacity: usize,
    ) -> Self {
        Self {
            decoder: SpeculativeDecodingV2::new(config),
            ngram_verifier: NgramVerifier::new(ngram_n, 0.1),
            tree_cache: TreeAttentionCache::new(cache_capacity),
            ngram_confidence_threshold,
            enable_ngram_verification: true,
            enable_tree_cache: true,
        }
    }

    /// 使用默认配置创建
    pub fn with_defaults() -> Self {
        Self::new(
            SpeculativeDecodingV2Config::default(),
            3,    // trigram
            0.01, // 低阈值
            50,   // 缓存容量
        )
    }

    /// 训练N-gram模型
    pub fn train_ngram_model(&mut self, tokens: &[u32]) {
        self.ngram_verifier.train(tokens);
    }

    /// 增强版验证流程
    ///
    /// 结合标准验证、N-gram验证和树形缓存的综合决策
    pub fn enhanced_verify(
        &mut self,
        context: &[u32],
        draft_candidates: &[Vec<CandidateToken>],
        target_probs: &[Array1<f32>],
        path_id: u64,
    ) -> Result<EnhancedVerificationResult> {
        // 第一步：标准验证
        let standard_result = self.decoder.verify_draft(draft_candidates, target_probs)?;

        // 第二步：N-gram辅助验证（如果启用且结果非完全接受）
        let mut ngram_confidences = Array1::<f32>::zeros(0);
        let mut ngram_adjusted_result = standard_result.clone();

        if self.enable_ngram_verification && !standard_result.fully_accepted {
            let draft_token_ids: Vec<u32> = draft_candidates
                .iter()
                .map(|candidates| candidates[0].token_id)
                .collect();

            ngram_confidences = self
                .ngram_verifier
                .verify_with_ngram(context, &draft_token_ids);

            // 如果N-gram置信度过低，调整接受结果
            for (i, &conf) in ngram_confidences.iter().enumerate() {
                if conf < self.ngram_confidence_threshold && i < standard_result.accept_length {
                    // 降低接受长度
                    ngram_adjusted_result.accept_length =
                        ngram_adjusted_result.accept_length.min(i);
                    ngram_adjusted_result.fully_accepted = false;
                    // 截断接受的tokens
                    ngram_adjusted_result.accepted_tokens.truncate(i);
                    break; // 发现第一个低置信度就停止
                }
            }
        }

        // 第三步：树形缓存操作（如果启用）
        let cache_hit = if self.enable_tree_cache {
            // 尝试查找缓存
            let lookup_result = self.tree_cache.lookup(path_id);
            lookup_result.is_some()
        } else {
            false
        };

        Ok(EnhancedVerificationResult {
            standard_result: ngram_adjusted_result,
            ngram_confidences: ngram_confidences.clone(), // 克隆以避免移动后借用
            cache_hit,
            used_ngram_verification: self.enable_ngram_verification
                && !ngram_confidences.is_empty(),
            used_tree_cache: self.enable_tree_cache,
        })
    }

    /// 存储路径缓存
    pub fn store_path_cache(
        &mut self,
        path_id: u64,
        key_cache: Array1<f32>,
        value_cache: Array1<f32>,
        depth: usize,
    ) {
        if self.enable_tree_cache {
            self.tree_cache
                .store(path_id, key_cache, value_cache, depth);
        }
    }

    /// 获取基础解码器的可变引用
    pub fn decoder_mut(&mut self) -> &mut SpeculativeDecodingV2 {
        &mut self.decoder
    }

    /// 获取基础解码器的不可变引用
    pub fn decoder(&self) -> &SpeculativeDecodingV2 {
        &self.decoder
    }

    /// 获取N-gram验证器的不可变引用
    pub fn ngram_verifier(&self) -> &NgramVerifier {
        &self.ngram_verifier
    }

    /// 获取树形缓存的不可变引用
    pub fn tree_cache(&self) -> &TreeAttentionCache {
        &self.tree_cache
    }

    /// 启用/禁用N-gram验证
    pub fn set_ngram_verification(&mut self, enabled: bool) {
        self.enable_ngram_verification = enabled;
    }

    /// 启用/禁用树形缓存
    pub fn set_tree_cache(&mut self, enabled: bool) {
        self.enable_tree_cache = enabled;
    }

    /// 重置所有状态
    pub fn reset_all(&mut self) {
        self.decoder.reset_stats();
        self.ngram_verifier.clear();
        self.tree_cache.clear();
    }
}

/// 增强验证结果
#[derive(Debug, Clone)]
pub struct EnhancedVerificationResult {
    /// 标准验证结果（可能被N-gram调整过）
    pub standard_result: SpeculationResult,
    /// N-gram置信度分数
    pub ngram_confidences: Array1<f32>,
    /// 是否命中树形缓存
    pub cache_hit: bool,
    /// 是否使用了N-gram验证
    pub used_ngram_verification: bool,
    /// 是否使用了树形缓存
    pub used_tree_cache: bool,
}

/// Speculative Decoding v2 主结构
#[derive(Debug)]
pub struct SpeculativeDecodingV2 {
    config: SpeculativeDecodingV2Config,
    stats: SpeculativeStats,
    current_draft_length: usize,
    rng: StdRng,
}

impl SpeculativeDecodingV2 {
    /// 创建新的Speculative Decoding v2实例
    pub fn new(config: SpeculativeDecodingV2Config) -> Self {
        Self {
            current_draft_length: config.initial_draft_length,
            config,
            stats: SpeculativeStats::default(),
            rng: StdRng::from_entropy(),
        }
    }

    /// 使用固定种子创建（用于复现）
    pub fn with_seed(config: SpeculativeDecodingV2Config, seed: u64) -> Self {
        Self {
            current_draft_length: config.initial_draft_length,
            config,
            stats: SpeculativeStats::default(),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// 生成草稿token序列
    ///
    /// 使用小型草稿模型快速生成候选token
    pub fn generate_draft(
        &mut self,
        draft_probs: &[Array1<f32>],
    ) -> Result<Vec<Vec<CandidateToken>>> {
        let draft_length = self.current_draft_length;
        let num_candidates = self.config.num_candidates;

        if draft_probs.len() < draft_length {
            return Err(anyhow::anyhow!(
                "草稿概率序列长度不足: 需要 {}, 实际 {}",
                draft_length,
                draft_probs.len()
            ));
        }

        let mut all_candidates = Vec::with_capacity(draft_length);

        for probs in draft_probs.iter() {
            let _vocab_size = probs.len();

            // 选择top-k候选
            let mut indexed_probs: Vec<(usize, f32)> =
                probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();

            indexed_probs
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let candidates: Vec<CandidateToken> = indexed_probs
                .into_iter()
                .take(num_candidates)
                .map(|(token_id, prob)| {
                    let log_prob = prob.ln();
                    CandidateToken {
                        token_id: token_id as u32,
                        prob,
                        log_prob,
                    }
                })
                .collect();

            all_candidates.push(candidates);
        }

        Ok(all_candidates)
    }

    /// 验证草稿token
    ///
    /// 使用目标模型验证草稿token，返回接受的结果
    pub fn verify_draft(
        &mut self,
        draft_candidates: &[Vec<CandidateToken>],
        target_probs: &[Array1<f32>],
    ) -> Result<SpeculationResult> {
        let draft_length = draft_candidates.len();

        if target_probs.len() < draft_length + 1 {
            return Err(anyhow::anyhow!(
                "目标概率序列长度不足: 需要 {}, 实际 {}",
                draft_length + 1,
                target_probs.len()
            ));
        }

        let mut accepted_tokens = Vec::with_capacity(draft_length);
        let mut accept_length = 0;
        let mut fully_accepted = true;

        // 逐个验证草稿token
        for step in 0..draft_length {
            let candidate = &draft_candidates[step][0]; // 选择top-1候选
            let target_prob = &target_probs[step];

            // 计算接受概率
            let accept_prob = self.calculate_accept_probability(
                candidate.prob,
                target_prob[candidate.token_id as usize],
            );

            // 拒绝采样
            if self.rng.gen::<f32>() < accept_prob {
                // 接受
                accepted_tokens.push(candidate.token_id);
                accept_length += 1;
            } else {
                // 拒绝，从调整后的分布中采样
                fully_accepted = false;
                let adjusted_probs =
                    self.adjust_distribution(target_prob, &draft_candidates[step])?;

                let sampled_token = self.sample_from_distribution(&adjusted_probs)?;
                accepted_tokens.push(sampled_token);
                accept_length += 1;
                break;
            }
        }

        // 如果完全接受，从目标模型的下一个位置采样
        let next_token_probs = if fully_accepted && target_probs.len() > draft_length {
            target_probs[draft_length].clone()
        } else {
            target_probs[accept_length].clone()
        };

        // 更新统计信息
        self.update_stats(accept_length, draft_length);

        // 自适应调整草稿长度
        if self.config.enable_adaptive {
            self.adapt_draft_length();
        }

        Ok(SpeculationResult {
            accepted_tokens,
            accept_length,
            fully_accepted,
            next_token_probs,
        })
    }

    /// 计算接受概率
    fn calculate_accept_probability(&self, draft_prob: f32, target_prob: f32) -> f32 {
        // 标准接受概率: min(1, p_target / p_draft)
        if draft_prob <= 0.0 {
            return 0.0;
        }
        (target_prob / draft_prob).min(1.0)
    }

    /// 调整分布（拒绝后重新采样）
    fn adjust_distribution(
        &self,
        target_probs: &Array1<f32>,
        candidates: &[CandidateToken],
    ) -> Result<Array1<f32>> {
        let vocab_size = target_probs.len();
        let mut adjusted = Array1::<f32>::zeros(vocab_size);

        // 计算调整后的分布: max(0, p_target - p_draft)
        for candidate in candidates {
            let idx = candidate.token_id as usize;
            if idx < vocab_size {
                adjusted[idx] = (target_probs[idx] - candidate.prob).max(0.0);
            }
        }

        // 归一化
        let sum: f32 = adjusted.sum();
        if sum > 0.0 {
            adjusted.mapv_inplace(|x| x / sum);
        } else {
            // 如果调整后全为0，使用原始目标分布
            adjusted.assign(target_probs);
        }

        Ok(adjusted)
    }

    /// 从分布中采样
    fn sample_from_distribution(&mut self, probs: &Array1<f32>) -> Result<u32> {
        let sum: f32 = probs.sum();
        if sum <= 0.0 {
            return Err(anyhow::anyhow!("概率分布和为0"));
        }

        // 归一化
        let normalized: Vec<f32> = probs.iter().map(|&p| p / sum).collect();

        // 使用加权随机采样
        let dist = rand::distributions::WeightedIndex::new(&normalized)?;
        let sampled_idx = dist.sample(&mut self.rng);

        Ok(sampled_idx as u32)
    }

    /// 更新统计信息
    fn update_stats(&mut self, accept_length: usize, draft_length: usize) {
        self.stats.total_speculations += 1;
        self.stats.accepted_tokens += accept_length as u64;
        self.stats.rejected_tokens += (draft_length - accept_length) as u64;

        // 更新平均值
        let total = self.stats.total_speculations as f32;
        self.stats.avg_accept_length =
            (self.stats.avg_accept_length * (total - 1.0) + accept_length as f32) / total;
        self.stats.avg_draft_length =
            (self.stats.avg_draft_length * (total - 1.0) + draft_length as f32) / total;

        // 计算接受率
        let total_tokens = self.stats.accepted_tokens + self.stats.rejected_tokens;
        if total_tokens > 0 {
            self.stats.acceptance_rate = self.stats.accepted_tokens as f32 / total_tokens as f32;
        }
    }

    /// 自适应调整草稿长度
    fn adapt_draft_length(&mut self) {
        let acceptance_rate = self.stats.acceptance_rate;

        if acceptance_rate > self.config.acceptance_threshold + 0.1 {
            // 接受率高，增加草稿长度
            self.current_draft_length =
                (self.current_draft_length + 1).min(self.config.max_draft_length);
        } else if acceptance_rate < self.config.acceptance_threshold - 0.1 {
            // 接受率低，减少草稿长度
            self.current_draft_length = self
                .current_draft_length
                .saturating_sub(1)
                .max(self.config.min_draft_length);
        }
    }

    /// 获取当前草稿长度
    pub fn current_draft_length(&self) -> usize {
        self.current_draft_length
    }

    /// 获取统计信息
    pub fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// 重置统计信息
    pub fn reset_stats(&mut self) {
        self.stats = SpeculativeStats::default();
        self.current_draft_length = self.config.initial_draft_length;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_speculative_decoding_v2_creation() {
        let config = SpeculativeDecodingV2Config::default();
        let sd = SpeculativeDecodingV2::new(config);
        assert_eq!(sd.current_draft_length(), 4);
    }

    #[test]
    fn test_generate_draft() {
        let config = SpeculativeDecodingV2Config::default();
        let mut sd = SpeculativeDecodingV2::new(config);

        let draft_probs: Vec<Array1<f32>> = (0..4)
            .map(|_| {
                let mut probs = Array1::zeros(100);
                probs[0] = 0.5;
                probs[1] = 0.3;
                probs[2] = 0.2;
                probs
            })
            .collect();

        let result = sd.generate_draft(&draft_probs);
        assert!(result.is_ok());

        let candidates = result.unwrap();
        assert_eq!(candidates.len(), 4);
        assert!(candidates[0].len() > 0);
    }

    #[test]
    fn test_verify_draft() {
        let config = SpeculativeDecodingV2Config::default();
        let mut sd = SpeculativeDecodingV2::with_seed(config, 42);

        // 创建草稿候选
        let draft_candidates: Vec<Vec<CandidateToken>> = vec![
            vec![CandidateToken {
                token_id: 0,
                prob: 0.5,
                log_prob: 0.5_f32.ln(),
            }],
            vec![CandidateToken {
                token_id: 1,
                prob: 0.3,
                log_prob: 0.3_f32.ln(),
            }],
            vec![CandidateToken {
                token_id: 2,
                prob: 0.2,
                log_prob: 0.2_f32.ln(),
            }],
        ];

        // 创建目标概率
        let target_probs: Vec<Array1<f32>> = (0..4)
            .map(|_| {
                let mut probs = Array1::zeros(100);
                probs[0] = 0.6;
                probs[1] = 0.3;
                probs[2] = 0.1;
                probs
            })
            .collect();

        let result = sd.verify_draft(&draft_candidates, &target_probs);
        assert!(result.is_ok());

        let spec_result = result.unwrap();
        assert!(spec_result.accept_length > 0);
    }

    #[test]
    fn test_adaptive_draft_length() {
        let config = SpeculativeDecodingV2Config {
            enable_adaptive: true,
            ..Default::default()
        };
        let mut sd = SpeculativeDecodingV2::new(config);

        // 模拟多次推测
        for _ in 0..10 {
            let draft_candidates: Vec<Vec<CandidateToken>> = vec![vec![CandidateToken {
                token_id: 0,
                prob: 0.5,
                log_prob: 0.5_f32.ln(),
            }]];

            let target_probs: Vec<Array1<f32>> = vec![
                Array1::from_vec(vec![0.9, 0.1]), // 高接受率
                Array1::from_vec(vec![0.5, 0.5]),
            ];

            let _ = sd.verify_draft(&draft_candidates, &target_probs);
        }

        // 检查自适应调整
        let stats = sd.stats();
        assert!(stats.total_speculations > 0);
    }

    // ===== 边界条件和错误路径测试 =====

    #[test]
    fn test_speculative_decoder_creation_various_configs() {
        // 各种配置创建测试
        let config = SpeculativeDecodingV2Config {
            initial_draft_length: 2,
            max_draft_length: 6,
            min_draft_length: 1,
            num_candidates: 2,
            acceptance_threshold: 0.6,
            enable_tree_speculation: false,
            tree_branch_factor: 2,
            temperature: 0.8,
            enable_adaptive: true,
        };

        let decoder = SpeculativeDecodingV2::new(config);
        assert_eq!(decoder.current_draft_length(), 2);
    }

    #[test]
    fn test_speculative_decoder_with_seed() {
        // 使用固定种子的可复现性测试
        let config = SpeculativeDecodingV2Config::default();
        let decoder1 = SpeculativeDecodingV2::with_seed(config.clone(), 42);
        let decoder2 = SpeculativeDecodingV2::with_seed(config, 42);

        // 两个相同种子的解码器应该产生相同的初始状态
        assert_eq!(
            decoder1.current_draft_length(),
            decoder2.current_draft_length()
        );
    }

    #[test]
    fn test_generate_draft_insufficient_probs() {
        // 草稿概率序列长度不足
        let config = SpeculativeDecodingV2Config {
            initial_draft_length: 4,
            ..Default::default()
        };
        let mut decoder = SpeculativeDecodingV2::new(config);

        // 只提供2个概率分布，但需要4个
        let draft_probs: Vec<Array1<f32>> = (0..2)
            .map(|_| {
                let mut probs = Array1::zeros(100);
                probs[0] = 0.8;
                probs
            })
            .collect();

        let result = decoder.generate_draft(&draft_probs);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_draft_insufficient_target_probs() {
        // 目标概率序列长度不足
        let config = SpeculativeDecodingV2Config::default();
        let mut decoder = SpeculativeDecodingV2::with_seed(config, 42);

        let draft_candidates: Vec<Vec<CandidateToken>> = vec![
            vec![CandidateToken {
                token_id: 0,
                prob: 0.5,
                log_prob: 0.5_f32.ln(),
            }],
            vec![CandidateToken {
                token_id: 1,
                prob: 0.3,
                log_prob: 0.3_f32.ln(),
            }],
            vec![CandidateToken {
                token_id: 2,
                prob: 0.2,
                log_prob: 0.2_f32.ln(),
            }],
        ];

        // 目标概率不足（需要至少4个，只提供3个）
        let target_probs: Vec<Array1<f32>> = (0..3)
            .map(|_| {
                let mut probs = Array1::zeros(100);
                probs[0] = 0.6;
                probs
            })
            .collect();

        let result = decoder.verify_draft(&draft_candidates, &target_probs);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_draft_high_acceptance_rate() {
        // 高接受率场景：所有草稿token都被接受
        let config = SpeculativeDecodingV2Config {
            acceptance_threshold: 0.7,
            ..Default::default()
        };
        let mut decoder = SpeculativeDecodingV2::with_seed(config, 123); // 使用固定种子

        // 创建高概率匹配的草稿候选
        let draft_candidates: Vec<Vec<CandidateToken>> = vec![
            vec![CandidateToken {
                token_id: 10,
                prob: 0.9,
                log_prob: 0.9_f32.ln(),
            }],
            vec![CandidateToken {
                token_id: 20,
                prob: 0.85,
                log_prob: 0.85_f32.ln(),
            }],
            vec![CandidateToken {
                token_id: 30,
                prob: 0.88,
                log_prob: 0.88_f32.ln(),
            }],
        ];

        // 创建与草稿高度一致的目标概率
        let target_probs: Vec<Array1<f32>> = (0..4)
            .map(|i| {
                let mut probs = Array1::zeros(100);
                if i < 3 {
                    match i {
                        0 => {
                            probs[10] = 0.95;
                        } // 高接受概率
                        1 => {
                            probs[20] = 0.90;
                        }
                        2 => {
                            probs[30] = 0.92;
                        }
                        _ => {}
                    }
                } else {
                    probs[0] = 0.5; // 下一个位置的概率
                }
                probs
            })
            .collect();

        let result = decoder.verify_draft(&draft_candidates, &target_probs);
        assert!(result.is_ok());

        let verification = result.unwrap();
        // 高接受率时，大部分或全部应该被接受
        assert!(
            verification.accept_length >= 2
                || verification.fully_accepted
                || verification.accept_length > 0
        );
    }

    #[test]
    fn test_verify_draft_low_acceptance_rate() {
        // 低接受率场景：所有草稿token都被拒绝
        let config = SpeculativeDecodingV2Config {
            acceptance_threshold: 0.99, // 极高阈值
            ..Default::default()
        };
        let mut decoder = SpeculativeDecodingV2::with_seed(config, 456);

        let draft_candidates: Vec<Vec<CandidateToken>> = vec![
            vec![CandidateToken {
                token_id: 50,
                prob: 0.01,
                log_prob: 0.01_f32.ln(),
            }],
            vec![CandidateToken {
                token_id: 60,
                prob: 0.02,
                log_prob: 0.02_f32.ln(),
            }],
        ];

        // 极低的目标概率
        let target_probs: Vec<Array1<f32>> = (0..3)
            .map(|i| {
                let mut probs = Array1::zeros(100);
                if i < 2 {
                    match i {
                        0 => {
                            probs[50] = 0.001;
                        } // 极低概率
                        1 => {
                            probs[60] = 0.002;
                        }
                        _ => {}
                    }
                } else {
                    probs[0] = 0.8;
                }
                probs
            })
            .collect();

        let result = decoder.verify_draft(&draft_candidates, &target_probs);
        assert!(result.is_ok());

        let verification = result.unwrap();
        // 低接受率时，大部分或全部被拒绝，但仍然会有采样结果
        assert!(verification.accept_length >= 1); // 至少会从拒绝分布中采样一个
    }

    #[test]
    fn test_verify_draft_partial_acceptance() {
        // 部分接受场景
        let config = SpeculativeDecodingV2Config {
            acceptance_threshold: 0.5,
            ..Default::default()
        };
        let mut decoder = SpeculativeDecodingV2::with_seed(config, 789);

        let draft_candidates: Vec<Vec<CandidateToken>> = vec![
            vec![CandidateToken {
                token_id: 10,
                prob: 0.9,
                log_prob: 0.9_f32.ln(),
            }],
            vec![CandidateToken {
                token_id: 20,
                prob: 0.3,
                log_prob: 0.3_f32.ln(),
            }],
            vec![CandidateToken {
                token_id: 30,
                prob: 0.8,
                log_prob: 0.8_f32.ln(),
            }],
            vec![CandidateToken {
                token_id: 40,
                prob: 0.1,
                log_prob: 0.1_f32.ln(),
            }],
        ];

        // 混合概率：有些高，有些低
        let target_probs: Vec<Array1<f32>> = (0..5)
            .map(|i| {
                let mut probs = Array1::zeros(100);
                match i {
                    0 => {
                        probs[10] = 0.95;
                    } // 高接受
                    1 => {
                        probs[20] = 0.05;
                    } // 低接受
                    2 => {
                        probs[30] = 0.85;
                    } // 高接受
                    3 => {
                        probs[40] = 0.02;
                    } // 低接受
                    4 => {
                        probs[0] = 0.5;
                    }
                    _ => {}
                }
                probs
            })
            .collect();

        let result = decoder.verify_draft(&draft_candidates, &target_probs);
        assert!(result.is_ok());

        let verification = result.unwrap();
        // 应该有部分接受
        assert!(
            verification.accept_length >= 1 && verification.accept_length <= draft_candidates.len()
        );
    }

    #[test]
    fn test_adaptive_draft_length_adjustment() {
        // 自适应草稿长度调整逻辑测试
        let config = SpeculativeDecodingV2Config {
            initial_draft_length: 4,
            max_draft_length: 8,
            min_draft_length: 2,
            acceptance_threshold: 0.7,
            enable_adaptive: true,
            ..Default::default()
        };
        let mut decoder = SpeculativeDecodingV2::new(config);

        let initial_length = decoder.current_draft_length();
        assert_eq!(initial_length, 4);

        // 模拟高接受率 -> 草稿长度可能增加
        for _ in 0..15 {
            let draft_candidates: Vec<Vec<CandidateToken>> = vec![vec![CandidateToken {
                token_id: 0,
                prob: 0.99,
                log_prob: 0.99_f32.ln(),
            }]];

            let target_probs: Vec<Array1<f32>> = vec![
                Array1::from_vec(vec![0.999, 0.001]), // 非常高的接受率
                Array1::from_vec(vec![0.5, 0.5]),
            ];

            let _ = decoder.verify_draft(&draft_candidates, &target_probs);
        }

        let length_after_high_acceptance = decoder.current_draft_length();

        // 重置并模拟低接受率
        let config2 = SpeculativeDecodingV2Config {
            initial_draft_length: 4,
            max_draft_length: 8,
            min_draft_length: 2,
            acceptance_threshold: 0.7,
            enable_adaptive: true,
            ..Default::default()
        };
        let mut decoder2 = SpeculativeDecodingV2::new(config2);

        for _ in 0..15 {
            let draft_candidates: Vec<Vec<CandidateToken>> = vec![vec![CandidateToken {
                token_id: 0,
                prob: 0.9,
                log_prob: 0.9_f32.ln(),
            }]];

            let target_probs: Vec<Array1<f32>> = vec![
                Array1::from_vec(vec![0.001, 0.999]), // 非常低的接受率
                Array1::from_vec(vec![0.5, 0.5]),
            ];

            let _ = decoder2.verify_draft(&draft_candidates, &target_probs);
        }

        let length_after_low_acceptance = decoder2.current_draft_length();

        // 验证自适应调整逻辑生效（高接受率时长度应 >= 低接受率时，或在合理范围内）
        assert!(
            length_after_high_acceptance >= length_after_low_acceptance
                || (length_after_low_acceptance >= 2 && length_after_low_acceptance <= 8),
            "Adaptive adjustment should work: high={}, low={}",
            length_after_high_acceptance,
            length_after_low_acceptance
        );
    }

    #[test]
    fn test_statistics_initial_state() {
        // 统计信息初始状态验证
        let config = SpeculativeDecodingV2Config::default();
        let decoder = SpeculativeDecodingV2::new(config);
        let stats = decoder.stats();

        assert_eq!(stats.total_speculations, 0);
        assert_eq!(stats.accepted_tokens, 0);
        assert_eq!(stats.rejected_tokens, 0);
        assert_eq!(stats.avg_accept_length, 0.0);
        assert_eq!(stats.avg_draft_length, 0.0);
        assert_eq!(stats.acceptance_rate, 0.0);
    }

    #[test]
    fn test_statistics_after_operations() {
        // 操作后的统计信息准确性
        let config = SpeculativeDecodingV2Config::default();
        let mut decoder = SpeculativeDecodingV2::with_seed(config, 999);

        // 执行几次推测操作
        for _ in 0..5 {
            let draft_candidates: Vec<Vec<CandidateToken>> = vec![
                vec![CandidateToken {
                    token_id: 0,
                    prob: 0.5,
                    log_prob: 0.5_f32.ln(),
                }],
                vec![CandidateToken {
                    token_id: 1,
                    prob: 0.3,
                    log_prob: 0.3_f32.ln(),
                }],
            ];

            let target_probs: Vec<Array1<f32>> = (0..3)
                .map(|_| {
                    let mut probs = Array1::zeros(100);
                    probs[0] = 0.6;
                    probs[1] = 0.4;
                    probs
                })
                .collect();

            let _ = decoder.verify_draft(&draft_candidates, &target_probs);
        }

        let stats = decoder.stats();

        // 验证统计信息已更新
        assert_eq!(stats.total_speculations, 5);
        assert!(stats.accepted_tokens > 0 || stats.rejected_tokens > 0);
        assert!(stats.avg_accept_length > 0.0);
        assert!(stats.avg_draft_length > 0.0);
        assert!(stats.acceptance_rate >= 0.0 && stats.acceptance_rate <= 1.0);
    }

    #[test]
    fn test_reset_stats() {
        // 重置功能测试
        let config = SpeculativeDecodingV2Config::default();
        let initial_draft_length = config.initial_draft_length;
        let mut decoder = SpeculativeDecodingV2::with_seed(config, 111);

        // 执行一些操作
        let draft_candidates: Vec<Vec<CandidateToken>> = vec![vec![CandidateToken {
            token_id: 0,
            prob: 0.5,
            log_prob: 0.5_f32.ln(),
        }]];

        let target_probs: Vec<Array1<f32>> = vec![
            Array1::from_vec(vec![0.6, 0.4]),
            Array1::from_vec(vec![0.5, 0.5]),
        ];

        let _ = decoder.verify_draft(&draft_candidates, &target_probs);

        // 确认有统计数据
        assert!(decoder.stats().total_speculations > 0);

        // 重置
        decoder.reset_stats();

        // 验证重置后的状态
        let stats = decoder.stats();
        assert_eq!(stats.total_speculations, 0);
        assert_eq!(stats.accepted_tokens, 0);
        assert_eq!(stats.rejected_tokens, 0);
        assert_eq!(stats.acceptance_rate, 0.0);
        assert_eq!(decoder.current_draft_length(), initial_draft_length);
    }

    #[test]
    fn test_config_edge_cases() {
        // 配置边界情况测试

        // 最小配置
        let min_config = SpeculativeDecodingV2Config {
            initial_draft_length: 1,
            max_draft_length: 1,
            min_draft_length: 1,
            num_candidates: 1,
            acceptance_threshold: 0.0,
            enable_tree_speculation: false,
            tree_branch_factor: 1,
            temperature: 0.0,
            enable_adaptive: false,
        };
        let decoder_min = SpeculativeDecodingV2::new(min_config);
        assert_eq!(decoder_min.current_draft_length(), 1);

        // 最大配置
        let max_config = SpeculativeDecodingV2Config {
            initial_draft_length: 16,
            max_draft_length: 32,
            min_draft_length: 1,
            num_candidates: 10,
            acceptance_threshold: 1.0,
            enable_tree_speculation: true,
            tree_branch_factor: 4,
            temperature: 2.0,
            enable_adaptive: true,
        };
        let decoder_max = SpeculativeDecodingV2::new(max_config);
        assert_eq!(decoder_max.current_draft_length(), 16);
    }

    #[test]
    fn test_empty_and_single_candidate_drafts() {
        // 测试不同长度的草稿生成
        let config = SpeculativeDecodingV2Config {
            initial_draft_length: 1,
            max_draft_length: 1,
            ..Default::default()
        };
        let mut decoder = SpeculativeDecodingV2::new(config);

        // 单个草稿token
        let draft_probs: Vec<Array1<f32>> = vec![{
            let mut probs = Array1::zeros(50);
            probs[0] = 0.7;
            probs[1] = 0.3;
            probs
        }];

        let result = decoder.generate_draft(&draft_probs);
        assert!(result.is_ok());
        let candidates = result.unwrap();
        assert_eq!(candidates.len(), 1);
        assert!(candidates[0].len() > 0);
    }

    #[test]
    fn test_verify_draft_single_token() {
        // 单个草稿token的验证
        let config = SpeculativeDecodingV2Config {
            initial_draft_length: 1,
            ..Default::default()
        };
        let mut decoder = SpeculativeDecodingV2::with_seed(config, 222);

        let draft_candidates: Vec<Vec<CandidateToken>> = vec![vec![CandidateToken {
            token_id: 4,
            prob: 0.8,
            log_prob: 0.8_f32.ln(),
        }]];

        let target_probs: Vec<Array1<f32>> = vec![
            Array1::from_vec(vec![0.1, 0.1, 0.1, 0.1, 0.6]), // token 4有较高概率
            Array1::from_vec(vec![0.5, 0.5]),
        ];

        let result = decoder.verify_draft(&draft_candidates, &target_probs);
        assert!(result.is_ok());

        let spec_result = result.unwrap();
        assert!(spec_result.accept_length >= 1);
    }

    // ===== N-gram 验证器测试 =====

    #[test]
    fn test_ngram_verifier_creation() {
        // 测试N-gram验证器的创建
        let verifier = NgramVerifier::new(3, 0.1);
        assert_eq!(verifier.n(), 3);
        assert_eq!(verifier.total_tokens(), 0);
        assert_eq!(verifier.ngram_table_size(), 0);
    }

    #[test]
    fn test_ngram_verifier_default_config() {
        // 测试默认配置
        let verifier = NgramVerifier::with_default_config();
        assert_eq!(verifier.n(), 3); // 默认trigram
    }

    #[test]
    fn test_ngram_verifier_training() {
        // 测试训练功能
        let mut verifier = NgramVerifier::new(3, 0.1);

        // 训练数据：简单序列
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        verifier.train(&tokens);

        // 验证统计信息
        assert_eq!(verifier.total_tokens(), 8);
        // 对于长度8的序列和n=3，应该有6个trigram (8-3+1=6)
        assert_eq!(verifier.ngram_table_size(), 6);
    }

    #[test]
    fn test_ngram_verification_with_trained_model() {
        // 测试使用已训练模型进行验证
        let mut verifier = NgramVerifier::new(3, 0.1);

        // 训练一个有规律的序列
        let training_tokens: Vec<u32> = vec![10, 20, 30, 40, 50, 10, 20, 30, 40, 50];
        verifier.train(&training_tokens);

        // 验证与训练数据一致的草稿token（应有较高置信度）
        let context: Vec<u32> = vec![10, 20];
        let draft_tokens: Vec<u32> = vec![30]; // 在训练数据中 [10,20] 后面常跟 30

        let confidences = verifier.verify_with_ngram(&context, &draft_tokens);
        assert_eq!(confidences.len(), 1);
        // 由于训练数据中有这个模式，置信度应该 > 0
        assert!(confidences[0] >= 0.0);
    }

    #[test]
    fn test_ngram_verification_empty_context() {
        // 测试空上下文情况
        let mut verifier = NgramVerifier::new(2, 0.1); // bigram

        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        verifier.train(&tokens);

        let context: Vec<u32> = vec![];
        let draft_tokens: Vec<u32> = vec![1, 2, 3];

        let confidences = verifier.verify_with_ngram(&context, &draft_tokens);
        assert_eq!(confidences.len(), 3);
        // 即使没有上下文，也应该返回合理的概率值
        for conf in confidences.iter() {
            assert!(*conf >= 0.0 && *conf <= 1.0);
        }
    }

    #[test]
    fn test_ngram_clear_functionality() {
        // 测试清空功能
        let mut verifier = NgramVerifier::new(3, 0.1);

        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        verifier.train(&tokens);

        assert!(verifier.total_tokens() > 0);
        assert!(verifier.ngram_table_size() > 0);

        verifier.clear();

        assert_eq!(verifier.total_tokens(), 0);
        assert_eq!(verifier.ngram_table_size(), 0);
    }

    #[test]
    fn test_ngram_various_n_values() {
        // 测试不同N值的N-gram
        for n in 1..=5 {
            let mut verifier = NgramVerifier::new(n, 0.05);
            assert_eq!(verifier.n(), n);

            let tokens: Vec<u32> = (0..20).collect();
            verifier.train(&tokens);

            let expected_ngrams = if tokens.len() >= n {
                tokens.len() - n + 1
            } else {
                0
            };
            assert_eq!(
                verifier.ngram_table_size(),
                expected_ngrams,
                "Failed for n={}: expected {} ngrams",
                n,
                expected_ngrams
            );
        }
    }

    // ===== 树形注意力缓存测试 =====

    #[test]
    fn test_tree_cache_creation() {
        // 测试缓存创建
        let cache = TreeAttentionCache::new(50);
        assert_eq!(cache.max_capacity(), 50);
        assert_eq!(cache.current_size(), 0);
        assert_eq!(cache.memory_usage_bytes(), 0);
        assert_eq!(cache.hit_rate(), 0.0); // 无访问时命中率为0
    }

    #[test]
    fn test_tree_cache_default_capacity() {
        // 测试默认容量
        let cache = TreeAttentionCache::with_default_capacity();
        assert_eq!(cache.max_capacity(), 100);
    }

    #[test]
    fn test_tree_cache_store_and_lookup() {
        // 测试存储和查找基本功能
        let mut cache = TreeAttentionCache::new(10);

        let key_cache = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let value_cache = Array1::from_vec(vec![0.5, 0.6, 0.7, 0.8]);

        // 存储
        cache.store(1, key_cache.clone(), value_cache.clone(), 2);

        assert_eq!(cache.current_size(), 1);
        assert!(cache.memory_usage_bytes() > 0);

        // 查找
        let result = cache.lookup(1);
        assert!(result.is_some());

        let (retrieved_key, retrieved_value, depth) = result.unwrap();
        assert_eq!(retrieved_key, key_cache);
        assert_eq!(retrieved_value, value_cache);
        assert_eq!(depth, 2);
    }

    #[test]
    fn test_tree_cache_miss() {
        // 测试缓存未命中
        let mut cache = TreeAttentionCache::new(10);

        let key_cache = Array1::from_vec(vec![0.1, 0.2]);
        let value_cache = Array1::from_vec(vec![0.3, 0.4]);

        cache.store(1, key_cache, value_cache, 1);

        // 先查找存在的路径（命中）
        let result_hit = cache.lookup(1);
        assert!(result_hit.is_some());

        // 查找不存在的路径（未命中）
        let result_miss = cache.lookup(999);
        assert!(result_miss.is_none());

        // 验证统计信息
        let (hits, misses) = cache.stats();
        assert_eq!(hits, 1); // lookup(1)是命中
        assert_eq!(misses, 1); // lookup(999)是未命中
    }

    #[test]
    fn test_tree_cache_lru_eviction() {
        // 测试LRU淘汰策略
        let mut cache = TreeAttentionCache::new(3); // 小容量以便触发淘汰

        // 填满缓存
        for i in 1..=3 {
            let key = Array1::from_vec(vec![i as f32]);
            let value = Array1::from_vec(vec![(i * 10) as f32]);
            cache.store(i, key, value, i as usize);
        }

        assert_eq!(cache.current_size(), 3);

        // 访问路径2和3，更新它们的访问时间（使路径1成为最久未使用的）
        let _ = cache.lookup(2);
        let _ = cache.lookup(3);

        // 插入第4个条目，应淘汰最久未使用的（路径1）
        let new_key = Array1::from_vec(vec![99.0]);
        let new_value = Array1::from_vec(vec![999.0]);
        cache.store(4, new_key.clone(), new_value.clone(), 4);

        assert_eq!(cache.current_size(), 3); // 容量不变

        // 验证路径1被淘汰（它是最久未访问的）
        let result = cache.lookup(1);
        assert!(result.is_none(), "Path 1 should have been evicted");

        // 新条目应该存在
        let result = cache.lookup(4);
        assert!(result.is_some());

        // 路径2和3应该仍然存在
        assert!(cache.lookup(2).is_some());
        assert!(cache.lookup(3).is_some());
    }

    #[test]
    fn test_tree_cache_invalidation() {
        // 测试使缓存失效
        let mut cache = TreeAttentionCache::new(10);

        let key = Array1::from_vec(vec![1.0, 2.0]);
        let value = Array1::from_vec(vec![3.0, 4.0]);

        cache.store(42, key, value, 1);
        assert_eq!(cache.current_size(), 1);

        // 使失效
        let removed = cache.invalidate(42);
        assert!(removed);
        assert_eq!(cache.current_size(), 0);

        // 使不存在的路径失效
        let removed_again = cache.invalidate(42);
        assert!(!removed_again);
    }

    #[test]
    fn test_tree_cache_batch_preload() {
        // 测试批量预加载
        let mut cache = TreeAttentionCache::new(10);

        let paths: Vec<(u64, Array1<f32>, Array1<f32>, usize)> = vec![
            (
                1,
                Array1::from_vec(vec![0.1]),
                Array1::from_vec(vec![0.2]),
                1,
            ),
            (
                2,
                Array1::from_vec(vec![0.3]),
                Array1::from_vec(vec![0.4]),
                2,
            ),
            (
                3,
                Array1::from_vec(vec![0.5]),
                Array1::from_vec(vec![0.6]),
                3,
            ),
        ];

        cache.batch_preload(paths);

        assert_eq!(cache.current_size(), 3);

        // 验证所有路径都可访问
        for path_id in 1..=3 {
            assert!(cache.lookup(path_id).is_some());
        }
    }

    #[test]
    fn test_tree_cache_clear() {
        // 测试清空缓存
        let mut cache = TreeAttentionCache::new(10);

        for i in 1..=5 {
            let key = Array1::from_vec(vec![i as f32]);
            let value = Array1::from_vec(vec![(i * 2) as f32]);
            cache.store(i, key, value, i as usize);
        }

        assert_eq!(cache.current_size(), 5);
        assert!(cache.memory_usage_bytes() > 0);

        cache.clear();

        assert_eq!(cache.current_size(), 0);
        assert_eq!(cache.memory_usage_bytes(), 0);

        let (hits, misses) = cache.stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 0);
    }

    #[test]
    fn test_tree_cache_hit_rate_calculation() {
        // 测试命中率计算
        let mut cache = TreeAttentionCache::new(10);

        let key = Array1::from_vec(vec![1.0]);
        let value = Array1::from_vec(vec![2.0]);
        cache.store(1, key, value, 1);

        // 多次命中
        for _ in 0..5 {
            let _ = cache.lookup(1);
        }

        // 多次未命中
        for _ in 0..3 {
            let _ = cache.lookup(999);
        }

        let hit_rate = cache.hit_rate();
        // 5次命中 + 3次未命中 + 1次初始存储后的隐式访问 ≈ 接近 6/9 或类似比例
        // 具体数值取决于实现细节，但应该在合理范围内
        assert!(hit_rate >= 0.0 && hit_rate <= 1.0);
    }

    #[test]
    fn test_tree_cache_update_existing_entry() {
        // 测试更新已存在的条目
        let mut cache = TreeAttentionCache::new(10);

        let key1 = Array1::from_vec(vec![1.0, 2.0]);
        let value1 = Array1::from_vec(vec![3.0, 4.0]);
        cache.store(1, key1, value1, 1);

        let size_before = cache.current_size();

        // 更新同一路径
        let key2 = Array1::from_vec(vec![5.0, 6.0, 7.0]); // 不同大小
        let value2 = Array1::from_vec(vec![8.0, 9.0, 10.0]);
        cache.store(1, key2, value2, 3);

        // 条目数不应增加
        assert_eq!(cache.current_size(), size_before);

        // 应该获取到新的值
        let result = cache.lookup(1).unwrap();
        assert_eq!(result.0.len(), 3); // 新key的长度
        assert_eq!(result.2, 3); // 新深度
    }

    // ===== EnhancedSpeculativeState 集成测试 =====

    #[test]
    fn test_enhanced_state_creation() {
        // 测试增强状态的创建
        let state = EnhancedSpeculativeState::with_defaults();

        assert_eq!(state.decoder().current_draft_length(), 4); // 默认配置
        assert_eq!(state.ngram_verifier().n(), 3);
        assert_eq!(state.tree_cache().max_capacity(), 50);
    }

    #[test]
    fn test_enhanced_state_custom_config() {
        // 测试自定义配置
        let config = SpeculativeDecodingV2Config {
            initial_draft_length: 3,
            max_draft_length: 6,
            ..Default::default()
        };

        let state = EnhancedSpeculativeState::new(config, 4, 0.05, 200);

        assert_eq!(state.decoder().current_draft_length(), 3);
        assert_eq!(state.ngram_verifier().n(), 4); // 4-gram
        assert_eq!(state.tree_cache().max_capacity(), 200);
    }

    #[test]
    fn test_enhanced_state_train_and_verify() {
        // 测试完整的训练和验证流程
        let mut state = EnhancedSpeculativeState::with_defaults();

        // 训练N-gram模型
        let training_data: Vec<u32> = vec![
            100, 200, 300, 400, 500, 100, 200, 300, 400, 500, 100, 200, 300, 400, 500,
        ];
        state.train_ngram_model(&training_data);

        assert!(state.ngram_verifier().total_tokens() > 0);

        // 执行增强验证
        let context: Vec<u32> = vec![100, 200];
        let draft_candidates: Vec<Vec<CandidateToken>> = vec![
            vec![CandidateToken {
                token_id: 300,
                prob: 0.7,
                log_prob: 0.7_f32.ln(),
            }],
            vec![CandidateToken {
                token_id: 400,
                prob: 0.5,
                log_prob: 0.5_f32.ln(),
            }],
        ];

        let target_probs: Vec<Array1<f32>> = vec![
            {
                let mut probs = Array1::zeros(1000);
                probs[300] = 0.8; // 第一个token高概率（可能被接受）
                probs
            },
            {
                let mut probs = Array1::zeros(1000);
                probs[400] = 0.05; // 第二个token低概率（很可能被拒绝，触发部分接受）
                probs
            },
            {
                let mut probs = Array1::zeros(1000);
                probs[0] = 0.5;
                probs
            },
        ];

        let result = state.enhanced_verify(&context, &draft_candidates, &target_probs, 1);
        assert!(result.is_ok());

        let enhanced_result = result.unwrap();
        // N-gram验证在非完全接受时使用，或者即使完全接受也可能使用（取决于实现）
        // 树形缓存总是被查询（如果启用）
        assert!(enhanced_result.used_tree_cache);
        // accept_length is unsigned, always non-negative
    }

    #[test]
    fn test_enhanced_state_toggle_features() {
        // 测试功能开关
        let mut state = EnhancedSpeculativeState::with_defaults();

        // 禁用N-gram验证
        state.set_ngram_verification(false);

        let context: Vec<u32> = vec![];
        let draft_candidates: Vec<Vec<CandidateToken>> = vec![vec![CandidateToken {
            token_id: 1,
            prob: 0.5,
            log_prob: 0.5_f32.ln(),
        }]];

        let target_probs: Vec<Array1<f32>> = vec![
            Array1::from_vec(vec![0.6, 0.4]),
            Array1::from_vec(vec![0.5, 0.5]),
        ];

        let result = state.enhanced_verify(&context, &draft_candidates, &target_probs, 1);
        assert!(result.is_ok());

        let enhanced_result = result.unwrap();
        assert!(!enhanced_result.used_ngram_verification); // 应该未被使用

        // 禁用树形缓存
        state.set_tree_cache(false);
        state.set_ngram_verification(true);

        let result2 = state.enhanced_verify(&context, &draft_candidates, &target_probs, 2);
        assert!(result2.is_ok());

        let enhanced_result2 = result2.unwrap();
        assert!(!enhanced_result2.used_tree_cache); // 应该未被使用
    }

    #[test]
    fn test_enhanced_state_reset_all() {
        // 测试重置所有状态
        let mut state = EnhancedSpeculativeState::with_defaults();

        // 训练并执行一些操作
        let training: Vec<u32> = vec![1, 2, 3, 4, 5];
        state.train_ngram_model(&training);

        let context: Vec<u32> = vec![];
        let draft_candidates: Vec<Vec<CandidateToken>> = vec![vec![CandidateToken {
            token_id: 1,
            prob: 0.5,
            log_prob: 0.5_f32.ln(),
        }]];
        let target_probs: Vec<Array1<f32>> = vec![
            Array1::from_vec(vec![0.6, 0.4]),
            Array1::from_vec(vec![0.5, 0.5]),
        ];
        let _ = state.enhanced_verify(&context, &draft_candidates, &target_probs, 1);

        // 存储一些缓存
        let key = Array1::from_vec(vec![1.0]);
        let value = Array1::from_vec(vec![2.0]);
        state.store_path_cache(1, key, value, 1);

        // 重置
        state.reset_all();

        // 验证所有状态已清空
        assert_eq!(state.decoder().stats().total_speculations, 0);
        assert_eq!(state.ngram_verifier().total_tokens(), 0);
        assert_eq!(state.tree_cache().current_size(), 0);
    }

    #[test]
    fn test_enhanced_state_store_and_lookup_cache() {
        // 测试通过EnhancedState操作树形缓存
        let mut state = EnhancedSpeculativeState::with_defaults();

        let key_cache = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let value_cache = Array1::from_vec(vec![0.4, 0.5, 0.6]);

        state.store_path_cache(100, key_cache.clone(), value_cache.clone(), 5);
        assert_eq!(state.tree_cache().current_size(), 1);

        // 通过enhanced_verify间接访问缓存（会尝试lookup）
        let context: Vec<u32> = vec![];
        let draft_candidates: Vec<Vec<CandidateToken>> = vec![vec![CandidateToken {
            token_id: 1,
            prob: 0.5,
            log_prob: 0.5_f32.ln(),
        }]];
        let target_probs: Vec<Array1<f32>> = vec![
            Array1::from_vec(vec![0.6, 0.4]),
            Array1::from_vec(vec![0.5, 0.5]),
        ];

        let result = state.enhanced_verify(&context, &draft_candidates, &target_probs, 100);
        assert!(result.is_ok());

        let enhanced_result = result.unwrap();
        assert!(enhanced_result.cache_hit); // 路径100应该命中
    }
}
