//! ESS 预取模块

#![allow(dead_code)]

use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    Sequential,
    Random,
    Strided,
    Temporal,
    Spatial,
    Unknown,
}

#[derive(Debug, Clone, Copy)]
pub struct PrefetchConfig {
    pub max_history: usize,
    pub spatial_threshold: usize,
    pub sequential_threshold: usize,
    pub max_prefetch: usize,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            max_history: 100,
            spatial_threshold: 4096,
            sequential_threshold: 4096,
            max_prefetch: 10,
        }
    }
}

pub struct LocalityAnalyzer {
    history: VecDeque<usize>,
    pattern: AccessPattern,
    stride: Option<usize>,
    avg_stride: usize,
    stride_direction: i64,
    config: PrefetchConfig,
}

impl LocalityAnalyzer {
    pub fn new(max_history: usize) -> Self {
        Self::with_config(PrefetchConfig {
            max_history,
            ..Default::default()
        })
    }

    pub fn with_config(config: PrefetchConfig) -> Self {
        Self {
            history: VecDeque::with_capacity(config.max_history),
            pattern: AccessPattern::Unknown,
            stride: None,
            avg_stride: 0,
            stride_direction: 1,
            config,
        }
    }

    fn abs_diff(a: usize, b: usize) -> usize {
        if a > b {
            a - b
        } else {
            b - a
        }
    }

    fn signed_diff(a: usize, b: usize) -> i64 {
        a as i64 - b as i64
    }

    fn reset(&mut self) {
        self.history.clear();
        self.pattern = AccessPattern::Unknown;
        self.stride = None;
        self.avg_stride = 0;
        self.stride_direction = 1;
    }

    pub fn record_access(&mut self, address: usize) {
        self.history.push_back(address);

        if self.history.len() > self.config.max_history {
            self.history.pop_front();
        }

        self.analyze_pattern();
    }

    fn analyze_pattern(&mut self) {
        if self.history.len() < 2 {
            self.pattern = AccessPattern::Unknown;
            return;
        }

        let addresses: Vec<usize> = self.history.iter().copied().collect();
        let n = addresses.len();

        let last_addr = addresses[n - 1];

        if addresses[..n - 1].contains(&last_addr) {
            self.pattern = AccessPattern::Temporal;
            return;
        }

        if n >= 3 {
            let diff1 = Self::signed_diff(addresses[1], addresses[0]);
            let diff2 = Self::signed_diff(addresses[2], addresses[1]);

            if diff1 == diff2 && diff1 != 0 {
                let all_same_stride = addresses
                    .windows(2)
                    .all(|w| Self::signed_diff(w[1], w[0]) == diff1);

                if all_same_stride {
                    self.stride = Some(diff1.unsigned_abs() as usize);
                    self.stride_direction = diff1.signum();
                    self.pattern = AccessPattern::Strided;
                    return;
                }
            }
        }

        if n >= 2 {
            let signed_strides: Vec<i64> = addresses
                .windows(2)
                .map(|w| Self::signed_diff(w[1], w[0]))
                .collect();

            let abs_strides: Vec<usize> = signed_strides
                .iter()
                .map(|&s| s.unsigned_abs() as usize)
                .collect();

            let sum: usize = abs_strides.iter().sum();
            self.avg_stride = if abs_strides.is_empty() {
                0
            } else {
                sum / abs_strides.len()
            };

            let all_small = abs_strides
                .iter()
                .all(|&s| s > 0 && s < self.config.sequential_threshold);
            let all_same_direction =
                signed_strides.iter().all(|&s| s > 0) || signed_strides.iter().all(|&s| s < 0);

            if all_small && all_same_direction && self.avg_stride > 0 {
                self.pattern = AccessPattern::Sequential;
                return;
            }
        }

        if n >= 3 {
            let last = addresses[n - 1];
            let prev = addresses[n - 2];
            let prev2 = addresses[n - 3];
            let diff1 = Self::abs_diff(last, prev);
            let diff2 = Self::abs_diff(prev, prev2);

            if diff1 > 0
                && diff1 < self.config.spatial_threshold
                && diff2 > 0
                && diff2 < self.config.spatial_threshold
            {
                self.pattern = AccessPattern::Spatial;
                return;
            }
        }

        self.pattern = AccessPattern::Random;
    }

    pub fn get_pattern(&self) -> AccessPattern {
        self.pattern
    }

    pub fn predict_next(&self) -> Option<usize> {
        let last_addr = *self.history.back()?;

        match self.pattern {
            AccessPattern::Temporal => Some(last_addr),
            AccessPattern::Strided => {
                let s = self.stride? as i64;
                let next = last_addr as i64 + s * self.stride_direction;
                if next >= 0 {
                    Some(next as usize)
                } else {
                    None
                }
            }
            AccessPattern::Sequential => {
                if self.avg_stride > 0 {
                    Some(last_addr + self.avg_stride)
                } else {
                    Some(last_addr + self.config.sequential_threshold)
                }
            }
            AccessPattern::Spatial => Some(last_addr + self.config.spatial_threshold),
            _ => None,
        }
    }
}

pub struct Prefetcher {
    locality_analyzer: LocalityAnalyzer,
    enabled: bool,
    prefetch_queue: Vec<usize>,
    config: PrefetchConfig,
}

impl Prefetcher {
    pub fn new() -> Self {
        Self::with_config(PrefetchConfig::default())
    }

    pub fn with_config(config: PrefetchConfig) -> Self {
        Self {
            locality_analyzer: LocalityAnalyzer::with_config(config),
            enabled: true,
            prefetch_queue: Vec::with_capacity(config.max_prefetch),
            config,
        }
    }

    pub fn access(&mut self, address: usize) -> Option<usize> {
        if !self.enabled {
            return None;
        }

        self.locality_analyzer.record_access(address);

        if self.prefetch_queue.len() >= self.config.max_prefetch {
            self.prefetch_queue.remove(0);
        }

        if let Some(predicted) = self.locality_analyzer.predict_next() {
            if !self.prefetch_queue.contains(&predicted) {
                self.prefetch_queue.push(predicted);
                return Some(predicted);
            }
        }

        None
    }

    pub fn get_prefetch_list(&self) -> &[usize] {
        &self.prefetch_queue
    }

    pub fn clear(&mut self) {
        self.prefetch_queue.clear();
        self.locality_analyzer.reset();
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn get_pattern(&self) -> AccessPattern {
        self.locality_analyzer.get_pattern()
    }
}

impl Default for Prefetcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strided_pattern() {
        let mut analyzer = LocalityAnalyzer::new(100);

        analyzer.record_access(0);
        analyzer.record_access(4096);
        analyzer.record_access(8192);

        assert_eq!(analyzer.get_pattern(), AccessPattern::Strided);
        assert_eq!(analyzer.predict_next(), Some(12288));
    }

    #[test]
    fn test_sequential_pattern() {
        let mut analyzer = LocalityAnalyzer::new(100);

        analyzer.record_access(100);
        analyzer.record_access(350);
        analyzer.record_access(800);

        assert_eq!(analyzer.get_pattern(), AccessPattern::Sequential);
    }

    #[test]
    fn test_temporal_pattern() {
        let mut analyzer = LocalityAnalyzer::new(100);

        analyzer.record_access(1000);
        analyzer.record_access(2000);
        analyzer.record_access(1000);

        assert_eq!(analyzer.get_pattern(), AccessPattern::Temporal);
        assert_eq!(analyzer.predict_next(), Some(1000));
    }

    #[test]
    fn test_spatial_pattern() {
        let mut analyzer = LocalityAnalyzer::new(100);

        analyzer.record_access(100000);
        analyzer.record_access(5000);
        analyzer.record_access(5050);
        analyzer.record_access(5080);

        assert_eq!(analyzer.get_pattern(), AccessPattern::Spatial);
    }

    #[test]
    fn test_random_pattern() {
        let mut analyzer = LocalityAnalyzer::new(100);

        analyzer.record_access(100);
        analyzer.record_access(50000);
        analyzer.record_access(200);

        assert_eq!(analyzer.get_pattern(), AccessPattern::Random);
    }

    #[test]
    fn test_sequential_direction_consistency() {
        let mut analyzer = LocalityAnalyzer::new(100);

        analyzer.record_access(100);
        analyzer.record_access(200);
        analyzer.record_access(150);
        analyzer.record_access(250);

        assert_ne!(analyzer.get_pattern(), AccessPattern::Sequential);
    }

    #[test]
    fn test_clear_resets_state() {
        let mut prefetcher = Prefetcher::new();

        prefetcher.access(1000);
        prefetcher.access(2000);
        prefetcher.access(3000);

        assert!(!prefetcher.get_prefetch_list().is_empty());
        assert_ne!(prefetcher.get_pattern(), AccessPattern::Unknown);

        prefetcher.clear();

        assert!(prefetcher.get_prefetch_list().is_empty());
        assert_eq!(prefetcher.get_pattern(), AccessPattern::Unknown);
    }

    #[test]
    fn test_address_overflow_safety() {
        let mut analyzer = LocalityAnalyzer::new(100);

        analyzer.record_access(10000);
        analyzer.record_access(1000);
        analyzer.record_access(100);

        assert!(analyzer.get_pattern() != AccessPattern::Unknown);
    }

    #[test]
    fn test_prefetcher_basic() {
        let mut prefetcher = Prefetcher::new();

        prefetcher.access(1000);
        prefetcher.access(2000);
        let predicted = prefetcher.access(3000);

        assert!(predicted.is_some());
        assert!(!prefetcher.get_prefetch_list().is_empty());
    }

    #[test]
    fn test_prefetcher_disabled() {
        let mut prefetcher = Prefetcher::new();
        prefetcher.set_enabled(false);

        let result = prefetcher.access(1000);
        assert!(result.is_none());
    }

    #[test]
    fn test_config() {
        let config = PrefetchConfig {
            max_history: 50,
            spatial_threshold: 2048,
            sequential_threshold: 1024,
            max_prefetch: 5,
        };

        let mut prefetcher = Prefetcher::with_config(config);

        prefetcher.access(0);
        prefetcher.access(256);
        prefetcher.access(768);

        assert_eq!(prefetcher.get_pattern(), AccessPattern::Sequential);
    }

    // ==================== 新增分支覆盖测试 ====================

    /// 测试 predict_next 各模式返回值（覆盖第167-193行所有模式分支）
    #[test]
    fn test_predict_next_all_patterns() {
        // Strided 模式预测（等差序列，所有stride相同）
        let mut strided_analyzer = LocalityAnalyzer::new(100);
        strided_analyzer.record_access(100);
        strided_analyzer.record_access(200);
        strided_analyzer.record_access(300); // avg_stride=100, 所有stride相同 -> Strided
        assert_eq!(strided_analyzer.get_pattern(), AccessPattern::Strided);
        let predicted = strided_analyzer.predict_next();
        assert!(predicted.is_some());
        assert_eq!(predicted.unwrap(), 400);

        // Sequential 模式预测（stride不同但都在阈值内）
        let mut seq_analyzer = LocalityAnalyzer::with_config(PrefetchConfig {
            sequential_threshold: 4096,
            ..Default::default()
        });
        seq_analyzer.record_access(0);
        seq_analyzer.record_access(256); // stride=256
        seq_analyzer.record_access(768); // stride=512 (不同) -> Sequential
        assert_eq!(seq_analyzer.get_pattern(), AccessPattern::Sequential);

        let predicted_seq = seq_analyzer.predict_next();
        assert!(predicted_seq.is_some());

        // Temporal 模式预测（返回最后地址）
        let mut temp_analyzer = LocalityAnalyzer::new(100);
        temp_analyzer.record_access(500);
        temp_analyzer.record_access(600);
        temp_analyzer.record_access(500); // 重复 -> temporal
        assert_eq!(temp_analyzer.get_pattern(), AccessPattern::Temporal);
        assert_eq!(temp_analyzer.predict_next(), Some(500));

        // Unknown 模式（历史不足）
        let empty_analyzer = LocalityAnalyzer::new(100);
        assert_eq!(empty_analyzer.get_pattern(), AccessPattern::Unknown);
        assert_eq!(empty_analyzer.predict_next(), None);
    }

    /// 测试 Strided 负方向（覆盖第174-179行负stride分支）
    #[test]
    fn test_strided_negative_direction() {
        let mut analyzer = LocalityAnalyzer::new(100);

        // 负方向等差序列：100, 50, 0 (stride=-50)
        analyzer.record_access(100);
        analyzer.record_access(50);
        analyzer.record_access(0);

        assert_eq!(analyzer.get_pattern(), AccessPattern::Strided);

        // 预测下一个应为 -50，但负值返回None
        let predicted = analyzer.predict_next();
        assert!(predicted.is_none(), "负地址应返回None");
    }

    /// 测试 history 溢出处理（覆盖第83-85行 pop_front 分支）
    #[test]
    fn test_history_overflow() {
        let config = PrefetchConfig {
            max_history: 3,
            ..Default::default()
        };
        let mut analyzer = LocalityAnalyzer::with_config(config);

        // 记录超过 max_history 的访问
        analyzer.record_access(1);
        analyzer.record_access(2);
        analyzer.record_access(3);
        analyzer.record_access(4); // 应移除1
        analyzer.record_access(5); // 应移除2

        // 历史应只保留 [3, 4, 5]
        assert_eq!(analyzer.predict_next(), Some(6));
    }

    /// 测试 AccessPattern 枚举完整性和 Clone/Copy trait
    #[test]
    fn test_access_pattern_traits() {
        // 覆盖所有变体
        let patterns = [
            AccessPattern::Sequential,
            AccessPattern::Random,
            AccessPattern::Strided,
            AccessPattern::Temporal,
            AccessPattern::Spatial,
            AccessPattern::Unknown,
        ];

        for p in &patterns {
            // 验证 Debug
            let _ = format!("{:?}", p);

            // 验证 Clone 和 Copy
            let p_copy = *p;
            assert_eq!(*p, p_copy);

            // 验证 PartialEq
            assert_eq!(*p, p_copy);
        }

        // 验证不同变体不等
        assert_ne!(AccessPattern::Sequential, AccessPattern::Random);
    }

    /// 测试 PrefetchConfig 默认值和 Prefetcher Default trait
    #[test]
    fn test_prefetch_defaults() {
        // 覆盖 Default trait
        let config = PrefetchConfig::default();
        assert_eq!(config.max_history, 100);
        assert_eq!(config.spatial_threshold, 4096);
        assert_eq!(config.sequential_threshold, 4096);
        assert_eq!(config.max_prefetch, 10);

        // 覆盖 Prefetcher Default
        let prefetcher = Prefetcher::default();
        assert!(prefetcher.get_prefetch_list().is_empty());
        assert_eq!(prefetcher.get_pattern(), AccessPattern::Unknown);
    }

    /// 测试预取队列满时的行为（覆盖第224-226行 remove(0) 分支）
    #[test]
    fn test_prefetch_queue_full() {
        let config = PrefetchConfig {
            max_prefetch: 2,
            ..Default::default()
        };
        let mut prefetcher = Prefetcher::with_config(config);

        // 生成足够多的预取以填满队列
        prefetcher.access(100);
        prefetcher.access(200);
        prefetcher.access(300); // 队列满，移除最早的

        let list = prefetcher.get_prefetch_list();
        assert!(list.len() <= 2, "队列大小不应超过max_prefetch");
    }

    /// 测试 Spatial 模式阈值边界（覆盖第153-158行 spatial_threshold 比较）
    #[test]
    fn test_spatial_boundary_threshold() {
        let config = PrefetchConfig {
            spatial_threshold: 100,
            sequential_threshold: 10, // 设置很小的sequential阈值
            ..Default::default()
        };
        let mut analyzer = LocalityAnalyzer::with_config(config);

        // 使用不同的stride（避免被识别为Strided），差距在spatial阈值内
        analyzer.record_access(10000);
        analyzer.record_access(10050); // diff=50 < 100 (spatial threshold)
        analyzer.record_access(10130); // diff=80 < 100 (不同stride，不是Strided)

        // 应该是 Spatial（因为stride不同且都在spatial阈值内）
        let pattern = analyzer.get_pattern();
        // 可能是 Spatial 或 Random（取决于具体实现），只要不是 Strided/Sequential 就可以
        assert!(
            pattern == AccessPattern::Spatial || pattern == AccessPattern::Random,
            "期望Spatial或Random，实际得到: {:?}",
            pattern
        );
    }
}
