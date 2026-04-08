//! HNSW (Hierarchical Navigable Small World) 向量索引
//!
//! 用于高效近似最近邻搜索的图索引结构
//!
//! # 核心概念
//!
//! - 多层图结构：上层稀疏，下层密集
//! - 插入时随机分配层数（指数分布）
//! - 搜索时从最高层开始，贪婪搜索到最近邻
//! - 连接数限制 M，保证图的稀疏性
//!
//! # 性能特点
//!
//! - 搜索复杂度：O(log(n))
//! - 空间复杂度：O(n * M * log(n))
//! - 支持增量更新，不阻塞读取
//!
//! # 邻居选择策略
//!
//! ## 简单策略 (Simple)
//! 按距离排序取前 m 个最近邻，速度快但可能导致邻居过于集中。
//!
//! ## 启发式策略 (Heuristic)
//! 考虑邻居的多样性，优先选择能增加覆盖范围的邻居。
//! 通过 `heuristic_diversity_factor` 参数控制多样性程度：
//!
//! | 因子值 | 效果 | 适用场景 |
//! |--------|------|----------|
//! | 0.5-0.6 | 较低多样性，邻居较集中 | 精度优先 |
//! | 0.7 (默认) | 平衡多样性与距离 | 通用场景 |
//! | 0.8-0.9 | 较高多样性，邻居分布广 | 召回优先 |
//!
//! 因子越大，越倾向于选择与已选点距离远的候选，增加图的连通性和召回率，
//! 但可能牺牲一定的精度。建议在 0.5-0.9 范围内调整。

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;

use parking_lot::RwLock;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

mod constants {
    pub const DEFAULT_M: usize = 16;
    pub const DEFAULT_EF_CONSTRUCTION: usize = 200;
    pub const DEFAULT_EF_SEARCH: usize = 50;
    pub const DEFAULT_SEED: u64 = 42;

    pub const MIN_LEVEL: usize = 4;
    pub const MAX_LEVEL: usize = 16;

    pub const DEFAULT_HEURISTIC_DIVERSITY_FACTOR: f32 = 0.7;

    pub const DISTANCE_UNROLL_FACTOR: usize = 4;
}

/// 邻居选择策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighborSelection {
    /// 简单策略：按距离排序取前 m 个
    Simple,
    /// 启发式策略：考虑多样性，优先选择能增加覆盖范围的邻居
    Heuristic,
}

/// HNSW 配置参数
#[derive(Debug, Clone, Copy)]
pub struct HNSWConfig {
    /// 每个节点的最大连接数
    pub m: usize,
    /// 构建时的搜索宽度
    pub ef_construction: usize,
    /// 搜索时的搜索宽度
    pub ef_search: usize,
    /// 最大层数（0 表示自动计算）
    pub max_level: usize,
    /// 随机种子
    pub seed: u64,
    /// 邻居选择策略
    pub neighbor_selection: NeighborSelection,
    /// 启发式邻居选择的多样性因子
    ///
    /// 控制启发式邻居选择时对多样性的偏好程度。
    /// 判断条件：`min_dist_to_selected >= candidate.distance * factor`
    ///
    /// - **值范围**: 0.5 - 0.9
    /// - **默认值**: 0.7
    /// - **较大值**: 倾向选择与已选点距离远的候选，增加图连通性
    /// - **较小值**: 倾向选择距离近的候选，提高精度
    pub heuristic_diversity_factor: f32,
}

impl Default for HNSWConfig {
    fn default() -> Self {
        use constants::*;
        Self {
            m: DEFAULT_M,
            ef_construction: DEFAULT_EF_CONSTRUCTION,
            ef_search: DEFAULT_EF_SEARCH,
            max_level: 0,
            seed: DEFAULT_SEED,
            neighbor_selection: NeighborSelection::Simple,
            heuristic_diversity_factor: DEFAULT_HEURISTIC_DIVERSITY_FACTOR,
        }
    }
}

/// HNSW 节点
#[derive(Debug, Clone)]
pub struct Node {
    /// 节点 ID
    pub id: usize,
    /// 向量数据
    pub vector: Vec<f32>,
    /// 各层的邻居列表（层号 -> 邻居 ID 列表）
    pub neighbors: HashMap<usize, Vec<usize>>,
}

impl Node {
    /// 创建新的 HNSW 节点
    ///
    /// # 参数
    /// - `id`: 节点唯一标识符
    /// - `vector`: 节点的向量数据
    #[inline]
    pub fn new(id: usize, vector: Vec<f32>) -> Self {
        Self {
            id,
            vector,
            neighbors: HashMap::new(),
        }
    }

    /// 获取指定层的邻居节点ID列表
    ///
    /// # 参数
    /// - `level`: 层号
    ///
    /// # 返回值
    /// 返回该层邻居ID的切片引用，如果没有邻居则返回空切片
    #[inline]
    pub fn get_neighbors(&self, level: usize) -> &[usize] {
        self.neighbors.get(&level).map_or(&[], |v| v.as_slice())
    }

    /// 向指定层添加一个邻居节点
    ///
    /// # 参数
    /// - `level`: 层号
    /// - `neighbor_id`: 要添加的邻居节点ID
    #[inline]
    pub fn add_neighbor(&mut self, level: usize, neighbor_id: usize) {
        self.neighbors.entry(level).or_default().push(neighbor_id);
    }

    /// 设置指定层的邻居列表（替换现有邻居）
    ///
    /// # 参数
    /// - `level`: 层号
    /// - `neighbors`: 新的邻居ID列表
    #[inline]
    pub fn set_neighbors(&mut self, level: usize, neighbors: Vec<usize>) {
        self.neighbors.insert(level, neighbors);
    }
}

/// HNSW 层
#[derive(Debug, Default)]
pub struct Layer {
    /// 该层的节点 ID 集合
    pub nodes: HashSet<usize>,
}

impl Layer {
    /// 创建新的空层
    #[inline]
    pub fn new() -> Self {
        Self {
            nodes: HashSet::new(),
        }
    }

    /// 向层中添加一个节点
    ///
    /// # 参数
    /// - `node_id`: 要添加的节点ID
    #[inline]
    pub fn add_node(&mut self, node_id: usize) {
        self.nodes.insert(node_id);
    }

    /// 从层中移除一个节点
    ///
    /// # 参数
    /// - `node_id`: 要移除的节点ID
    #[inline]
    pub fn remove_node(&mut self, node_id: usize) {
        self.nodes.remove(&node_id);
    }

    /// 检查层中是否包含指定节点
    ///
    /// # 参数
    /// - `node_id`: 要检查的节点ID
    ///
    /// # 返回值
    /// 如果包含返回 true，否则返回 false
    #[inline]
    pub fn contains(&self, node_id: usize) -> bool {
        self.nodes.contains(&node_id)
    }

    /// 获取层中的节点数量
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// 检查层是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// 搜索结果项
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchResult {
    /// 节点ID
    pub id: usize,
    /// 到查询向量的距离（欧氏距离平方）
    pub distance: f32,
}

impl Eq for SearchResult {}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// HNSW 索引主结构
#[derive(Debug)]
pub struct HNSWIndex {
    /// 配置参数
    config: HNSWConfig,
    /// 所有节点（节点 ID -> 节点）
    nodes: Arc<RwLock<HashMap<usize, Node>>>,
    /// 各层结构
    layers: Arc<RwLock<Vec<Layer>>>,
    /// 入口节点 ID
    entry_point: Arc<RwLock<Option<usize>>>,
    /// 最大层数
    max_level: Arc<RwLock<usize>>,
    /// 随机数生成器
    rng: Arc<RwLock<StdRng>>,
    /// 层数生成因子
    level_factor: f64,
    /// 向量维度（0 表示未设置）
    dimension: Arc<RwLock<usize>>,
}

impl HNSWIndex {
    /// 创建新的 HNSW 索引
    ///
    /// # 参数
    /// - `config`: HNSW 配置参数
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let config = HNSWConfig::default();
    /// let index = HNSWIndex::new(config);
    /// ```
    #[inline]
    pub fn new(config: HNSWConfig) -> Self {
        let level_factor = 1.0 / (config.m as f64).ln();
        let rng = StdRng::seed_from_u64(config.seed);

        Self {
            config,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            layers: Arc::new(RwLock::new(Vec::new())),
            entry_point: Arc::new(RwLock::new(None)),
            max_level: Arc::new(RwLock::new(0)),
            rng: Arc::new(RwLock::new(rng)),
            level_factor,
            dimension: Arc::new(RwLock::new(0)),
        }
    }

    /// 获取向量的维度
    ///
    /// # 返回值
    /// 如果尚未插入任何向量返回0，否则返回向量维度
    #[inline]
    pub fn dimension(&self) -> usize {
        *self.dimension.read()
    }

    /// 检查向量维度是否一致
    fn check_dimension(&self, vector: &[f32]) -> Result<(), String> {
        let mut dim = self.dimension.write();
        if *dim == 0 {
            *dim = vector.len();
            return Ok(());
        }
        if *dim != vector.len() {
            return Err(format!(
                "Dimension mismatch: expected {}, got {}",
                *dim,
                vector.len()
            ));
        }
        Ok(())
    }

    /// 计算两个向量的欧氏距离平方
    ///
    /// 使用循环展开优化，减少迭代器开销。
    /// 展开因子由 `DISTANCE_UNROLL_FACTOR` 常量定义（默认为 4）。
    #[inline]
    fn distance(v1: &[f32], v2: &[f32]) -> f32 {
        use constants::DISTANCE_UNROLL_FACTOR;

        let len = v1.len();
        let chunks = len / DISTANCE_UNROLL_FACTOR;
        let remainder = len % DISTANCE_UNROLL_FACTOR;

        let mut sum = 0.0f32;

        for i in 0..chunks {
            let offset = i * DISTANCE_UNROLL_FACTOR;
            let a0 = v1[offset];
            let a1 = v1[offset + 1];
            let a2 = v1[offset + 2];
            let a3 = v1[offset + 3];
            let b0 = v2[offset];
            let b1 = v2[offset + 1];
            let b2 = v2[offset + 2];
            let b3 = v2[offset + 3];

            let d0 = a0 - b0;
            let d1 = a1 - b1;
            let d2 = a2 - b2;
            let d3 = a3 - b3;

            sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        }

        for i in 0..remainder {
            let offset = chunks * DISTANCE_UNROLL_FACTOR + i;
            let d = v1[offset] - v2[offset];
            sum += d * d;
        }

        sum
    }

    fn random_level(&self) -> usize {
        use constants::*;
        let mut rng = self.rng.write();
        let random_value: f64 = rng.gen();
        let level = (-random_value.ln() * self.level_factor) as usize;

        let max_allowed = if self.config.max_level > 0 {
            self.config.max_level
        } else {
            let node_count = self.nodes.read().len();
            if node_count > 0 {
                ((node_count as f64).log2().ceil() as usize).clamp(MIN_LEVEL, MAX_LEVEL)
            } else {
                MAX_LEVEL
            }
        };

        level.min(max_allowed)
    }

    /// 贪婪搜索：从指定节点开始，找到距离查询向量最近的节点
    fn greedy_search(
        &self,
        query: &[f32],
        entry_id: usize,
        ef: usize,
        level: usize,
    ) -> Vec<SearchResult> {
        let nodes = self.nodes.read();

        let entry_node = match nodes.get(&entry_id) {
            Some(node) => node,
            None => return Vec::new(),
        };

        let entry_distance = Self::distance(query, &entry_node.vector);

        let mut visited = HashSet::with_capacity(ef * 2);
        visited.insert(entry_id);

        let mut candidates = BinaryHeap::with_capacity(ef);
        candidates.push(SearchResult {
            id: entry_id,
            distance: -entry_distance,
        });

        let mut results = BinaryHeap::with_capacity(ef + 1);
        results.push(SearchResult {
            id: entry_id,
            distance: entry_distance,
        });

        while let Some(candidate) = candidates.pop() {
            let candidate_dist = -candidate.distance;

            if let Some(worst) = results.peek() {
                if candidate_dist > worst.distance {
                    break;
                }
            }

            let neighbors = match nodes.get(&candidate.id) {
                Some(node) => node.get_neighbors(level).to_vec(),
                None => continue,
            };

            for neighbor_id in neighbors {
                if visited.contains(&neighbor_id) {
                    continue;
                }
                visited.insert(neighbor_id);

                let neighbor_node = match nodes.get(&neighbor_id) {
                    Some(node) => node,
                    None => continue,
                };

                let dist = Self::distance(query, &neighbor_node.vector);

                if results.len() < ef
                    || dist < results.peek().map(|r| r.distance).unwrap_or(f32::MAX)
                {
                    candidates.push(SearchResult {
                        id: neighbor_id,
                        distance: -dist,
                    });

                    results.push(SearchResult {
                        id: neighbor_id,
                        distance: dist,
                    });

                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut result_vec: Vec<SearchResult> = results.into_iter().collect();
        result_vec.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result_vec
    }

    fn select_neighbors(&self, candidates: &[SearchResult], m: usize, query: &[f32]) -> Vec<usize> {
        match self.config.neighbor_selection {
            NeighborSelection::Simple => {
                let mut sorted = candidates.to_vec();
                sorted.sort_by(|a, b| {
                    a.distance
                        .partial_cmp(&b.distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                sorted.iter().take(m).map(|r| r.id).collect()
            }
            NeighborSelection::Heuristic => self.select_neighbors_heuristic(candidates, m, query),
        }
    }

    fn select_neighbors_heuristic(
        &self,
        candidates: &[SearchResult],
        m: usize,
        _query: &[f32],
    ) -> Vec<usize> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let diversity_factor = self.config.heuristic_diversity_factor;

        let mut sorted = candidates.to_vec();
        sorted.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut selected: Vec<usize> = Vec::with_capacity(m);
        let mut selected_vectors: Vec<Vec<f32>> = Vec::with_capacity(m);

        let nodes = self.nodes.read();

        for candidate in &sorted {
            if selected.len() >= m {
                break;
            }

            let candidate_node = match nodes.get(&candidate.id) {
                Some(n) => n,
                None => continue,
            };

            let mut min_dist_to_selected = f32::MAX;
            for existing_vec in &selected_vectors {
                let dist = Self::distance(&candidate_node.vector, existing_vec);
                if dist < min_dist_to_selected {
                    min_dist_to_selected = dist;
                }
            }

            if min_dist_to_selected >= candidate.distance * diversity_factor {
                selected.push(candidate.id);
                selected_vectors.push(candidate_node.vector.clone());
            }
        }

        if selected.len() < m {
            for candidate in &sorted {
                if selected.len() >= m {
                    break;
                }
                if !selected.contains(&candidate.id) {
                    selected.push(candidate.id);
                }
            }
        }

        selected
    }

    /// 插入新向量
    ///
    /// # 错误
    /// 如果向量维度与已有数据不一致，将忽略该插入操作
    pub fn insert(&self, id: usize, vector: Vec<f32>) {
        if self.check_dimension(&vector).is_err() {
            return;
        }

        let node_level = self.random_level();
        let new_node = Node::new(id, vector.clone());

        {
            let mut nodes = self.nodes.write();
            nodes.insert(id, new_node);
        }

        let entry_point_id = {
            let entry = self.entry_point.read();
            *entry
        };

        if entry_point_id.is_none() {
            let mut layers = self.layers.write();
            while layers.len() <= node_level {
                layers.push(Layer::new());
            }
            layers[node_level].add_node(id);

            let mut entry = self.entry_point.write();
            *entry = Some(id);

            let mut max_level = self.max_level.write();
            *max_level = node_level;

            return;
        }

        let entry_id = match entry_point_id {
            Some(id) => id,
            None => return,
        };
        let current_max_level = {
            let max_level = self.max_level.read();
            *max_level
        };

        let mut current_entry = entry_id;

        for level in (node_level + 1..=current_max_level).rev() {
            let results = self.greedy_search(&vector, current_entry, 1, level);
            if let Some(nearest) = results.first() {
                current_entry = nearest.id;
            }
        }

        {
            let mut layers = self.layers.write();
            while layers.len() <= node_level {
                layers.push(Layer::new());
            }
        }

        for level in (0..=node_level).rev() {
            let ef = self.config.ef_construction;
            let candidates = self.greedy_search(&vector, current_entry, ef, level);

            let m = if level == 0 {
                self.config.m * 2
            } else {
                self.config.m
            };

            let neighbors = self.select_neighbors(&candidates, m, &vector);

            {
                let mut nodes = self.nodes.write();
                if let Some(node) = nodes.get_mut(&id) {
                    node.set_neighbors(level, neighbors.clone());
                }
            }

            {
                let mut layers = self.layers.write();
                layers[level].add_node(id);
            }

            for neighbor_id in neighbors {
                let neighbor_vector = {
                    let mut nodes = self.nodes.write();
                    if let Some(neighbor) = nodes.get_mut(&neighbor_id) {
                        neighbor.add_neighbor(level, id);
                        let neighbor_neighbors = neighbor.get_neighbors(level).to_vec();
                        let m_max = if level == 0 {
                            self.config.m * 2
                        } else {
                            self.config.m
                        };

                        if neighbor_neighbors.len() > m_max {
                            Some((neighbor.vector.clone(), m_max))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };

                if let Some((vector, m_max)) = neighbor_vector {
                    let results = self.greedy_search(&vector, neighbor_id, m_max, level);
                    let selected = self.select_neighbors(&results, m_max, &vector);

                    let mut nodes = self.nodes.write();
                    if let Some(n) = nodes.get_mut(&neighbor_id) {
                        n.set_neighbors(level, selected);
                    }
                }
            }

            if !candidates.is_empty() {
                current_entry = candidates[0].id;
            }
        }

        if node_level > current_max_level {
            let mut max_level = self.max_level.write();
            *max_level = node_level;

            let mut entry = self.entry_point.write();
            *entry = Some(id);
        }
    }

    /// 搜索最近邻
    ///
    /// # 参数
    /// - `query`: 查询向量
    /// - `k`: 返回的最近邻数量
    ///
    /// # 返回
    /// 如果维度不匹配或索引为空，返回空向量
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        if self.check_dimension(query).is_err() {
            return Vec::new();
        }

        let entry_id = {
            let entry = self.entry_point.read();
            match *entry {
                Some(id) => id,
                None => return Vec::new(),
            }
        };

        let max_level = {
            let max_level = self.max_level.read();
            *max_level
        };

        let mut current_entry = entry_id;

        for level in (1..=max_level).rev() {
            let results = self.greedy_search(query, current_entry, 1, level);
            if let Some(nearest) = results.first() {
                current_entry = nearest.id;
            }
        }

        let ef = self.config.ef_search.max(k);
        let mut results = self.greedy_search(query, current_entry, ef, 0);

        results.truncate(k);
        results
    }

    /// 获取索引中的节点总数
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.read().len()
    }

    /// 检查索引是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.read().is_empty()
    }

    /// 获取指定ID节点的向量数据
    ///
    /// # 参数
    /// - `id`: 节点ID
    ///
    /// # 返回值
    /// 如果节点存在返回 Some(向量)，否则返回 None
    #[inline]
    pub fn get_vector(&self, id: usize) -> Option<Vec<f32>> {
        self.nodes.read().get(&id).map(|node| node.vector.clone())
    }

    /// 获取 HNSW 配置的引用
    ///
    /// # 返回值
    /// 返回配置参数的不可变引用
    #[inline]
    pub fn config(&self) -> &HNSWConfig {
        &self.config
    }

    /// 清空索引
    pub fn clear(&self) {
        let mut nodes = self.nodes.write();
        nodes.clear();

        let mut layers = self.layers.write();
        layers.clear();

        let mut entry = self.entry_point.write();
        *entry = None;

        let mut max_level = self.max_level.write();
        *max_level = 0;

        let mut dimension = self.dimension.write();
        *dimension = 0;
    }
}

impl Default for HNSWIndex {
    fn default() -> Self {
        Self::new(HNSWConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vector(id: usize, dim: usize) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        if id < dim {
            v[id] = 1.0;
        }
        v
    }

    #[test]
    fn test_hnsw_insert_single() {
        let index = HNSWIndex::default();
        let vector = create_test_vector(0, 128);

        index.insert(0, vector);

        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_hnsw_insert_multiple() {
        let index = HNSWIndex::default();

        for i in 0..100 {
            let vector = create_test_vector(i, 128);
            index.insert(i, vector);
        }

        assert_eq!(index.len(), 100);
    }

    #[test]
    fn test_hnsw_search_basic() {
        let config = HNSWConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 50,
            max_level: 0,
            seed: 42,
            neighbor_selection: NeighborSelection::Simple,
            heuristic_diversity_factor: 0.7,
        };
        let index = HNSWIndex::new(config);

        for i in 0..50 {
            let vector = create_test_vector(i, 128);
            index.insert(i, vector);
        }

        let query = create_test_vector(0, 128);
        let results = index.search(&query, 5);

        assert!(!results.is_empty());
        assert!(results.len() <= 5);

        if !results.is_empty() {
            assert_eq!(results[0].id, 0);
        }
    }

    #[test]
    fn test_hnsw_search_accuracy() {
        let config = HNSWConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            max_level: 0,
            seed: 42,
            neighbor_selection: NeighborSelection::Simple,
            heuristic_diversity_factor: 0.7,
        };
        let index = HNSWIndex::new(config);

        for i in 0..100 {
            let mut vector = vec![0.0; 64];
            vector[i % 64] = 1.0;
            index.insert(i, vector);
        }

        let mut query = vec![0.0; 64];
        query[10] = 1.0;

        let results = index.search(&query, 10);

        assert!(!results.is_empty());

        let found = results.iter().any(|r| r.id == 10 || r.id == 10 + 64);
        assert!(found);
    }

    #[test]
    fn test_hnsw_get_vector() {
        let index = HNSWIndex::default();
        let vector = create_test_vector(5, 128);

        index.insert(5, vector.clone());

        let retrieved = index.get_vector(5);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), vector);

        let missing = index.get_vector(999);
        assert!(missing.is_none());
    }

    #[test]
    fn test_hnsw_clear() {
        let index = HNSWIndex::default();

        for i in 0..10 {
            let vector = create_test_vector(i, 64);
            index.insert(i, vector);
        }

        assert_eq!(index.len(), 10);

        index.clear();

        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_hnsw_random_level() {
        let index = HNSWIndex::default();

        let mut level_counts = HashMap::new();
        for _ in 0..1000 {
            let level = index.random_level();
            *level_counts.entry(level).or_insert(0) += 1;
        }

        assert!(level_counts.contains_key(&0));
        assert!(*level_counts.get(&0).unwrap() > 500);
    }

    #[test]
    fn test_hnsw_distance() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];
        let v3 = vec![1.0, 1.0, 0.0];

        let d12 = HNSWIndex::distance(&v1, &v2);
        let d13 = HNSWIndex::distance(&v1, &v3);

        assert!((d12 - 2.0).abs() < 1e-5);
        assert!((d13 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_hnsw_concurrent_read() {
        let index = Arc::new(HNSWIndex::default());

        for i in 0..100 {
            let vector = create_test_vector(i, 64);
            index.insert(i, vector);
        }

        let index_clone = Arc::clone(&index);
        let handle = std::thread::spawn(move || {
            let query = create_test_vector(0, 64);
            index_clone.search(&query, 5)
        });

        let results = handle.join().unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_node_neighbors() {
        let mut node = Node::new(0, vec![1.0, 0.0]);

        node.add_neighbor(0, 1);
        node.add_neighbor(0, 2);
        node.add_neighbor(1, 3);

        assert_eq!(node.get_neighbors(0), &[1, 2]);
        assert_eq!(node.get_neighbors(1), &[3]);
        assert!(node.get_neighbors(2).is_empty());

        node.set_neighbors(0, vec![4, 5]);
        assert_eq!(node.get_neighbors(0), &[4, 5]);
    }

    #[test]
    fn test_layer_operations() {
        let mut layer = Layer::new();

        assert!(layer.is_empty());

        layer.add_node(1);
        layer.add_node(2);
        layer.add_node(3);

        assert_eq!(layer.len(), 3);
        assert!(layer.contains(1));
        assert!(layer.contains(2));
        assert!(!layer.contains(4));

        layer.remove_node(2);
        assert_eq!(layer.len(), 2);
        assert!(!layer.contains(2));
    }

    #[test]
    fn test_search_result_ordering() {
        let r1 = SearchResult {
            id: 0,
            distance: 0.5,
        };
        let r2 = SearchResult {
            id: 1,
            distance: 0.3,
        };
        let r3 = SearchResult {
            id: 2,
            distance: 0.7,
        };

        let mut heap = BinaryHeap::new();
        heap.push(r1);
        heap.push(r2);
        heap.push(r3);

        let max = heap.pop().unwrap();
        assert_eq!(max.distance, 0.7);
    }

    #[test]
    fn test_config_default() {
        let config = HNSWConfig::default();

        assert_eq!(config.m, 16);
        assert_eq!(config.ef_construction, 200);
        assert_eq!(config.ef_search, 50);
        assert_eq!(config.max_level, 0);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_dimension_check() {
        let index = HNSWIndex::default();

        index.insert(0, vec![1.0, 0.0, 0.0]);
        assert_eq!(index.dimension(), 3);

        index.insert(1, vec![0.0, 1.0, 0.0]);
        assert_eq!(index.len(), 2);

        index.insert(2, vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn test_search_dimension_mismatch() {
        let index = HNSWIndex::default();

        for i in 0..10 {
            let vector = create_test_vector(i, 64);
            index.insert(i, vector);
        }

        let query = vec![0.0; 128];
        let results = index.search(&query, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_empty_index_search() {
        let index = HNSWIndex::default();

        let query = vec![0.0; 64];
        let results = index.search(&query, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_dimension_reset_after_clear() {
        let index = HNSWIndex::default();

        index.insert(0, vec![1.0; 64]);
        assert_eq!(index.dimension(), 64);

        index.clear();
        assert_eq!(index.dimension(), 0);

        index.insert(1, vec![1.0; 128]);
        assert_eq!(index.dimension(), 128);
    }

    #[test]
    fn test_heuristic_neighbor_selection() {
        let config = HNSWConfig {
            m: 8,
            ef_construction: 50,
            ef_search: 20,
            max_level: 0,
            seed: 42,
            neighbor_selection: NeighborSelection::Heuristic,
            heuristic_diversity_factor: 0.7,
        };
        let index = HNSWIndex::new(config);

        for i in 0..20 {
            let mut vector = vec![0.0; 32];
            vector[i % 32] = 1.0;
            index.insert(i, vector);
        }

        let query = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let results = index.search(&query, 5);

        assert!(!results.is_empty());
        assert!(results.len() <= 5);

        if let Some(first) = results.first() {
            assert!(first.distance < 0.5);
        }
    }

    #[test]
    fn test_heuristic_vs_simple() {
        let config_simple = HNSWConfig {
            m: 8,
            ef_construction: 50,
            ef_search: 20,
            max_level: 0,
            seed: 42,
            neighbor_selection: NeighborSelection::Simple,
            heuristic_diversity_factor: 0.7,
        };
        let config_heuristic = HNSWConfig {
            m: 8,
            ef_construction: 50,
            ef_search: 20,
            max_level: 0,
            seed: 42,
            neighbor_selection: NeighborSelection::Heuristic,
            heuristic_diversity_factor: 0.7,
        };

        let index_simple = HNSWIndex::new(config_simple);
        let index_heuristic = HNSWIndex::new(config_heuristic);

        for i in 0..50 {
            let mut vector = vec![0.0; 64];
            vector[i % 64] = 1.0;
            index_simple.insert(i, vector.clone());
            index_heuristic.insert(i + 100, vector);
        }

        let mut query = vec![0.0; 64];
        query[0] = 1.0;
        let results_simple = index_simple.search(&query, 5);
        let results_heuristic = index_heuristic.search(&query, 5);

        assert!(
            !results_simple.is_empty(),
            "Simple search should return results"
        );
        assert!(
            !results_heuristic.is_empty(),
            "Heuristic search should return results"
        );
    }
}
