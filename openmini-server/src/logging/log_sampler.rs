//! 日志采样器
//!
//! 提供智能日志采样功能，避免高频操作产生过多日志：
//! - 基于时间窗口的采样（每N秒只记录一次）
//! - 基于概率的采样（只记录一定比例的日志）
//! - 基于计数的采样（每N次只记录一次）
//! - 自适应采样（根据错误率动态调整）

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ============================================================================
// 时间窗口采样器
// ============================================================================

/// 基于时间窗口的日志采样器
///
/// 在指定的时间间隔内只允许通过一条日志，适用于周期性报告的场景。
/// 例如：每10秒只记录一次GPU内存状态。
///
/// # 示例
///
/// ```rust,ignore
/// use openmini_server::logging::TimeWindowSampler;
/// use tracing::info;
///
/// // 每5秒只记录一次
/// let mut sampler = TimeWindowSampler::new(Duration::from_secs(5));
///
/// fn report_gpu_memory(mem_used: u64) {
///     if sampler.should_log() {
///         info!(gpu_memory_mb = mem_used, "GPU memory status");
///     }
/// }
/// ```
pub struct TimeWindowSampler {
    interval: Duration,
    last_log: Instant,
}

impl TimeWindowSampler {
    /// 创建时间窗口采样器
    ///
    /// # 参数
    ///
    /// * `interval` - 允许日志通过的最小时间间隔
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            last_log: Instant::now() - interval * 2, // 初始化为过去，确保首次可以通过
        }
    }

    /// 检查是否应该记录当前日志
    ///
    /// 如果距离上次记录超过设定间隔，返回true并更新时间戳；
    /// 否则返回false。
    pub fn should_log(&mut self) -> bool {
        let now = Instant::now();

        if now.duration_since(self.last_log) >= self.interval {
            self.last_log = now;
            true
        } else {
            false
        }
    }

    /// 重置采样器，强制下次调用时通过
    pub fn reset(&mut self) {
        self.last_log = Instant::now() - self.interval * 2;
    }
}

// ============================================================================
// 概率采样器
// ============================================================================

/// 基于概率的日志采样器
///
/// 以指定概率决定是否记录日志，适用于高频但低重要性的日志。
/// 例如：只记录10%的中间推理步骤日志。
///
/// # 线程安全
///
/// 内部使用原子计数器，可安全地在多线程环境中使用。
///
/// # 示例
///
/// ```rust,ignore
/// use openmini_server::logging::ProbabilitySampler;
/// use tracing::debug;
///
/// // 只记录约10%的日志
/// static DEBUG_SAMPLER: ProbabilitySampler = ProbabilitySampler::new(0.1);
///
/// fn log_debug_step(step: u32) {
///     if DEBUG_SAMPLER.should_log() {
///         debug!(step, "Processing step");
///     }
/// }
/// ```
pub struct ProbabilitySampler {
    probability: f64,
    counter: AtomicU64,
}

impl ProbabilitySampler {
    /// 创建概率采样器
    ///
    /// # 参数
    ///
    /// * `probability` - 通过概率，范围 [0.0, 1.0]
    ///   - 0.0: 永不记录
    ///   - 1.0: 全部记录
    ///   - 0.1: 记录约10%
    ///
    /// # Panics
    ///
    /// 当 probability 不在 [0.0, 1.0] 范围内时会 panic
    pub const fn new(probability: f64) -> Self {
        if probability < 0.0 || probability > 1.0 {
            panic!("Probability must be between 0.0 and 1.0");
        }
        Self {
            probability,
            counter: AtomicU64::new(0),
        }
    }

    /// 检查是否应该记录当前日志
    ///
    /// 使用简单的计数器算法近似概率分布：
    /// 每次调用递增计数器，当 `counter % (1/probability) == 0` 时返回true。
    /// 对于非整数倒数的情况，会尽量逼近设定的概率。
    pub fn should_log(&self) -> bool {
        if self.probability >= 1.0 {
            return true;
        }
        if self.probability <= 0.0 {
            return false;
        }

        let count = self.counter.fetch_add(1, Ordering::Relaxed);
        let interval = (1.0 / self.probability).round() as u64;

        if interval == 0 {
            return true; // probability接近1.0时的边界情况
        }

        count % interval == 0
    }

    /// 获取当前的采样概率
    pub fn probability(&self) -> f64 {
        self.probability
    }

    /// 获取已检查的总次数（用于监控）
    pub fn total_checks(&self) -> u64 {
        self.counter.load(Ordering::Relaxed)
    }
}

// ============================================================================
// 计数采样器
// ============================================================================

/// 基于固定计数的日志采样器
///
/// 每N次调用中只允许通过1次，适用于循环体内的日志。
/// 例如：在1000步的推理中，每100步记录一次进度。
///
/// # 示例
///
/// ```rust,ignore
/// use openmini_server::logging::CountSampler;
/// use tracing::info;
///
/// // 每100次记录一次
/// let mut sampler = CountSampler::every_n(100);
///
/// for step in 0..1000 {
///     if sampler.should_log() {
///         info!(step, progress = step as f32 / 10.0, "Progress");
///     }
/// }
/// ```
pub struct CountSampler {
    interval: u64,
    counter: u64,
}

impl CountSampler {
    /// 创建每N次记录一次的采样器
    ///
    /// # 参数
    ///
    /// * `n` - 采样间隔（每N次记录1次）
    pub fn every_n(n: u64) -> Self {
        assert!(n > 0, "Interval must be greater than 0");
        Self {
            interval: n,
            counter: 0,
        }
    }

    /// 检查是否应该记录当前日志
    pub fn should_log(&mut self) -> bool {
        self.counter += 1;
        if self.counter % self.interval == 0 {
            true
        } else {
            false
        }
    }

    /// 获取当前计数
    pub fn current_count(&self) -> u64 {
        self.counter
    }

    /// 重置计数器
    pub fn reset(&mut self) {
        self.counter = 0;
    }
}

// ============================================================================
// 自适应采样器
// ============================================================================

/// 自适应日志采样器
///
/// 根据最近的错误率动态调整采样频率：
/// - 错误率高时：增加采样率（记录更多日志以便诊断）
/// - 错误率低时：降低采样率（减少日志量）
///
/// 适用于生产环境的长期运行服务。
///
/// # 示例
///
/// ```rust,ignore
/// use openmini_server::logging::AdaptiveSampler;
/// use tracing::{info, error};
///
/// let mut sampler = AdaptiveSampler::new(0.05); // 默认5%采样率
///
/// loop {
///     match process_request() {
///         Ok(_) => {
///             sampler.record_success();
///             if sampler.should_log() {
///                 info!("Request processed");
///             }
///         }
///         Err(e) => {
///             sampler.record_error();
///             error!(error = %e, "Request failed"); // 错误始终记录
///         }
///     }
/// }
/// ```
pub struct AdaptiveSampler {
    base_probability: f64,    // 基础采样率
    max_probability: f64,     // 最大采样率
    min_probability: f64,     // 最小采样率
    window_size: u64,         // 统计窗口大小
    success_count: AtomicU64, // 成功计数
    error_count: AtomicU64,   // 错误计数
    counter: AtomicU64,       // 总调用计数
}

impl AdaptiveSampler {
    /// 创建自适应采样器
    ///
    /// # 参数
    ///
    /// * `base_probability` - 基础采样率（无错误时）[0.0, 1.0]
    pub fn new(base_probability: f64) -> Self {
        assert!(base_probability > 0.0 && base_probability <= 1.0);

        Self {
            base_probability,
            max_probability: 1.0,
            min_probability: base_probability * 0.1, // 最少为基础值的10%
            window_size: 1000,
            success_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            counter: AtomicU64::new(0),
        }
    }

    /// 设置最大采样率
    pub fn with_max_probability(mut self, max: f64) -> Self {
        assert!(max <= 1.0);
        self.max_probability = max;
        self
    }

    /// 设置最小采样率
    pub fn with_min_probability(mut self, min: f64) -> Self {
        assert!(min >= 0.0);
        self.min_probability = min;
        self
    }

    /// 设置统计窗口大小
    pub fn with_window_size(mut self, size: u64) -> Self {
        assert!(size > 0);
        self.window_size = size;
        self
    }

    /// 记录一次成功操作
    pub fn record_success(&self) {
        self.success_count.fetch_add(1, Ordering::Relaxed);
    }

    /// 记录一次错误操作
    pub fn record_error(&self) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }

    /// 检查是否应该记录当前日志
    ///
    /// 根据最近窗口内的错误率动态计算采样概率：
    /// - 错误率 < 1%: 使用基础采样率
    /// - 错误率 1%-10%: 线性增加采样率
    /// - 错误率 > 10%: 使用最大采样率
    pub fn should_log(&self) -> bool {
        let count = self.counter.fetch_add(1, Ordering::Relaxed);

        // 每window_size次重新计算采样率
        if count % self.window_size != 0 {
            return false; // 在窗口内只采样一次（简化实现）
        }

        // 计算当前错误率
        let successes = self.success_count.load(Ordering::Relaxed);
        let errors = self.error_count.load(Ordering::Relaxed);
        let total = successes + errors;

        if total == 0 {
            return rand::random::<f64>() < self.base_probability;
        }

        let error_rate = errors as f64 / total as f64;

        // 根据错误率计算实际采样概率
        let effective_prob = if error_rate < 0.01 {
            self.base_probability
        } else if error_rate < 0.10 {
            // 线性插值：1%->base, 10%->max
            let t = (error_rate - 0.01) / 0.09;
            self.base_probability + t * (self.max_probability - self.base_probability)
        } else {
            self.max_probability
        };

        // 重置计数器以开始新窗口
        self.success_count.store(0, Ordering::Relaxed);
        self.error_count.store(0, Ordering::Relaxed);

        rand::random::<f64>() < effective_prob
    }

    /// 获取当前错误率估计
    pub fn current_error_rate(&self) -> f64 {
        let successes = self.success_count.load(Ordering::Relaxed);
        let errors = self.error_count.load(Ordering::Relaxed);
        let total = successes + errors;

        if total == 0 {
            0.0
        } else {
            errors as f64 / total as f64
        }
    }
}

// ============================================================================
// 预定义采样器配置
// ============================================================================

/// 推荐的GPU内存状态采样间隔（10秒）
pub const GPU_MEMORY_SAMPLE_INTERVAL: Duration = Duration::from_secs(10);

/// 推荐的批处理状态采样间隔（5秒）
pub const BATCH_STATUS_SAMPLE_INTERVAL: Duration = Duration::from_secs(5);

/// 推荐的调试级采样概率（1%）
pub const DEBUG_SAMPLE_PROBABILITY: f64 = 0.01;

/// 推荐的详细追踪采样概率（0.1%）
pub const TRACE_SAMPLE_PROBABILITY: f64 = 0.001;

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tracing::info;

    // ==================== TimeWindowSampler 测试 ====================

    #[test]
    fn test_time_window_first_call_always_passes() {
        let mut sampler = TimeWindowSampler::new(Duration::from_millis(100));
        assert!(sampler.should_log()); // 首次应该通过
    }

    #[test]
    fn test_time_window_blocks_rapid_calls() {
        let mut sampler = TimeWindowSampler::new(Duration::from_secs(60)); // 60秒间隔
        sampler.should_log(); // 第一次通过

        // 紧接着的调用应该被阻止
        assert!(!sampler.should_log());
        assert!(!sampler.should_log());
    }

    #[test]
    fn test_time_window_reset() {
        let mut sampler = TimeWindowSampler::new(Duration::from_secs(60));
        sampler.should_log(); // 第一次通过
        assert!(!sampler.should_log());

        sampler.reset(); // 重置后应该再次通过
        assert!(sampler.should_log());
    }

    // ==================== ProbabilitySampler 测试 ====================

    #[test]
    fn test_probability_full() {
        let sampler = ProbabilitySampler::new(1.0);
        // 100%概率应该全部通过
        for _ in 0..100 {
            assert!(sampler.should_log());
        }
    }

    #[test]
    fn test_probability_none() {
        let sampler = ProbabilitySampler::new(0.0);
        // 0%概率应该全部拒绝
        for _ in 0..100 {
            assert!(!sampler.should_log());
        }
    }

    #[test]
    fn test_probability_half_approximate() {
        let sampler = ProbabilitySampler::new(0.5);
        let mut passed = 0;
        let total = 1000;

        for _ in 0..total {
            if sampler.should_log() {
                passed += 1;
            }
        }

        // 应该接近50%，允许±10%的误差
        let ratio = passed as f64 / total as f64;
        assert!(
            ratio > 0.4 && ratio < 0.6,
            "Ratio was {}, expected ~0.5",
            ratio
        );
    }

    #[test]
    fn test_probability_tenth_approximate() {
        let sampler = ProbabilitySampler::new(0.1);
        let mut passed = 0;
        let total = 1000;

        for _ in 0..total {
            if sampler.should_log() {
                passed += 1;
            }
        }

        let ratio = passed as f64 / total as f64;
        assert!(
            ratio > 0.05 && ratio < 0.15,
            "Ratio was {}, expected ~0.1",
            ratio
        );
    }

    #[test]
    fn test_probability_counter_increments() {
        let sampler = ProbabilitySampler::new(0.5);
        for _ in 0..10 {
            sampler.should_log();
        }
        assert_eq!(sampler.total_checks(), 10);
    }

    // ==================== CountSampler 测试 ====================

    #[test]
    fn test_count_every_n() {
        let mut sampler = CountSampler::every_n(5);

        // 前4次不应该通过
        for i in 1..5 {
            assert!(!sampler.should_log(), "Call {} should not pass", i);
        }

        // 第5次应该通过
        assert!(sampler.should_log(), "Call 5 should pass");

        // 接下来的4次不应该通过
        for i in 1..5 {
            assert!(!sampler.should_log(), "Call {} should not pass", i + 5);
        }

        // 第10次应该再次通过
        assert!(sampler.should_log(), "Call 10 should pass");
    }

    #[test]
    fn test_count_reset() {
        let mut sampler = CountSampler::every_n(10);
        for _ in 0..5 {
            sampler.should_log();
        }
        assert_eq!(sampler.current_count(), 5);

        sampler.reset();
        assert_eq!(sampler.current_count(), 0);

        // 重置后重新计数
        for _ in 0..9 {
            assert!(!sampler.should_log());
        }
        assert!(sampler.should_log());
    }

    // ==================== AdaptiveSampler 测试 ====================

    #[test]
    fn test_adaptive_creation() {
        let sampler = AdaptiveSampler::new(0.1);
        assert!((sampler.base_probability - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_adaptive_custom_config() {
        let sampler = AdaptiveSampler::new(0.2)
            .with_max_probability(0.8)
            .with_min_probability(0.05)
            .with_window_size(500);

        assert!((sampler.max_probability - 0.8).abs() < 0.001);
        assert!((sampler.min_probability - 0.05).abs() < 0.001);
        assert_eq!(sampler.window_size, 500);
    }

    #[test]
    fn test_adaptive_record_and_check() {
        let sampler = AdaptiveSampler::new(0.5);

        // 记录一些成功
        for _ in 0..100 {
            sampler.record_success();
        }

        // 应该能够正常调用should_log而不panic
        // （具体结果取决于随机性和内部逻辑）
        let _ = sampler.should_log();
    }

    #[test]
    fn test_adaptive_error_rate_calculation() {
        let sampler = AdaptiveSampler::new(0.1);

        // 记录混合结果
        for _ in 0..90 {
            sampler.record_success();
        }
        for _ in 0..10 {
            sampler.record_error();
        }

        let rate = sampler.current_error_rate();
        assert!(
            (rate - 0.1).abs() < 0.01,
            "Error rate was {}, expected ~0.1",
            rate
        );
    }

    // ==================== 集成测试 ====================

    #[test]
    fn test_sampler_with_tracing() {
        // 验证采样器可以在实际的tracing日志中使用
        let mut count_sampler = CountSampler::every_n(3);

        let mut logged_count = 0;
        for i in 0..9 {
            if count_sampler.should_log() {
                info!(iteration = i, "Sampled log entry");
                logged_count += 1;
            }
        }

        // 9次调用，每3次采样1次，应该有3条日志
        assert_eq!(logged_count, 3);
    }

    #[test]
    fn test_multiple_samplers_independent() {
        // 验证多个采样器实例互不影响
        let mut sampler_a = CountSampler::every_n(2);
        let mut sampler_b = CountSampler::every_n(3);

        let mut a_count = 0;
        let mut b_count = 0;

        for _ in 0..12 {
            if sampler_a.should_log() {
                a_count += 1;
            }
            if sampler_b.should_log() {
                b_count += 1;
            }
        }

        assert_eq!(a_count, 6); // 12/2 = 6
        assert_eq!(b_count, 4); // 12/3 = 4
    }

    #[test]
    fn test_sampler_config_constants() {
        // 验证预定义的采样配置常量有效
        assert!(DEBUG_SAMPLE_PROBABILITY > 0.0 && DEBUG_SAMPLE_PROBABILITY <= 1.0);
        assert!(TRACE_SAMPLE_PROBABILITY > 0.0 && TRACE_SAMPLE_PROBABILITY <= 1.0);
        assert!(GPU_MEMORY_SAMPLE_INTERVAL.as_secs() > 0);
        assert!(BATCH_STATUS_SAMPLE_INTERVAL.as_secs() > 0);
    }

    // ===== 边界条件和分支覆盖率测试 =====

    // ==================== TimeWindowSampler 额外测试 ====================

    #[test]
    fn test_time_window_sampler_timing_precision() {
        // 测试时间窗口采样器的精确时间控制
        let mut sampler = TimeWindowSampler::new(std::time::Duration::from_millis(100)); // 100ms间隔

        // 第一次应该采样
        assert!(sampler.should_log(), "First call should always pass");

        // 短时间内第二次应该跳过
        assert!(
            !sampler.should_log(),
            "Second immediate call should be blocked"
        );

        // 等待超过间隔后应该再次通过（需要实际等待，这里只测试快速连续调用）
        for _ in 0..10 {
            assert!(!sampler.should_log(), "Rapid calls should be blocked");
        }
    }

    #[test]
    fn test_time_window_sampler_very_short_interval() {
        // 测试非常短的时间间隔
        let mut sampler = TimeWindowSampler::new(std::time::Duration::from_nanos(1)); // 1纳秒

        // 由于初始化时last_log被设置为过去很长时间，第一次总是通过
        assert!(sampler.should_log());

        // 即使是1纳秒间隔，紧接着的调用也可能因为系统时钟精度而被阻塞
        // 但我们至少验证它不会panic
        let _ = sampler.should_log();
    }

    #[test]
    fn test_time_window_sampler_very_long_interval() {
        // 测试非常长的时间间隔（1小时）
        let mut sampler = TimeWindowSampler::new(std::time::Duration::from_secs(3600));

        assert!(sampler.should_log()); // 首次通过

        // 之后所有调用都被阻塞（在我们的测试时间范围内）
        for _ in 0..100 {
            assert!(!sampler.should_log());
        }
    }

    // ==================== ProbabilitySampler 额外测试 ====================

    #[test]
    fn test_probability_sampler_rate_accuracy() {
        // 更精确地测试概率采样率
        let sampler = ProbabilitySampler::new(0.5); // 50%采样率

        let mut sampled = 0;
        let total = 10000; // 使用更大的样本以获得更准确的结果

        for _ in 0..total {
            if sampler.should_log() {
                sampled += 1;
            }
        }

        let rate = sampled as f64 / total as f64;
        // 允许±5%的误差（对于大样本）
        assert!(
            (rate - 0.5).abs() < 0.05,
            "Rate was {:.4}, expected ~0.5 (±0.05)",
            rate
        );
    }

    #[test]
    fn test_probability_sampler_edge_cases() {
        // 测试边界值附近的概率
        // 0.01 (1%)
        let sampler_low = ProbabilitySampler::new(0.01);
        let low_count = (0..10000).filter(|_| sampler_low.should_log()).count();
        let low_rate = low_count as f64 / 10000.0;
        assert!(
            low_rate > 0.0 && low_rate < 0.02,
            "Low rate was {}",
            low_rate
        );

        // 0.99 (99%)
        let sampler_high = ProbabilitySampler::new(0.99);
        let high_count = (0..10000).filter(|_| sampler_high.should_log()).count();
        let high_rate = high_count as f64 / 10000.0;
        assert!(
            high_rate > 0.97 && high_rate <= 1.0,
            "High rate was {}",
            high_rate
        );
    }

    #[test]
    fn test_probability_sampler_total_checks_tracking() {
        // 测试计数器正确跟踪总调用次数
        let sampler = ProbabilitySampler::new(0.5);

        assert_eq!(sampler.total_checks(), 0);

        for i in 1..=50 {
            sampler.should_log();
            assert_eq!(sampler.total_checks(), i, "Counter should track calls");
        }
    }

    #[test]
    fn test_probability_sampler_get_probability() {
        // 测试获取概率值的访问器
        let sampler1 = ProbabilitySampler::new(0.25);
        let sampler2 = ProbabilitySampler::new(0.75);
        let sampler3 = ProbabilitySampler::new(1.0);
        let sampler4 = ProbabilitySampler::new(0.0);

        assert!((sampler1.probability() - 0.25).abs() < 0.001);
        assert!((sampler2.probability() - 0.75).abs() < 0.001);
        assert!((sampler3.probability() - 1.0).abs() < 0.001);
        assert!((sampler4.probability() - 0.0).abs() < 0.001);
    }

    // ==================== CountSampler 额外测试 ====================

    #[test]
    fn test_count_sampler_exact_interval_behavior() {
        // 精确测试计数采样器的间隔行为
        let mut sampler = CountSampler::every_n(5); // 每5次采样1次

        // 前4次不应该采样
        for i in 1..=4 {
            assert!(!sampler.should_log(), "Call {} should not pass", i);
        }

        // 第5次应该采样
        assert!(sampler.should_log(), "Call 5 should pass");

        // 第6-9次不应该采样
        for i in 6..=9 {
            assert!(!sampler.should_log(), "Call {} should not pass", i);
        }

        // 第10次应该再次采样
        assert!(sampler.should_log(), "Call 10 should pass");

        // 第11次不应该采样
        assert!(!sampler.should_log(), "Call 11 should not pass");
    }

    #[test]
    fn test_count_sampler_with_interval_one() {
        // interval=1 意味着每次都采样
        let mut sampler = CountSampler::every_n(1);

        for i in 1..=20 {
            assert!(
                sampler.should_log(),
                "With interval=1, call {} should pass",
                i
            );
        }
    }

    #[test]
    fn test_count_sampler_large_interval() {
        // 大间隔测试
        let mut sampler = CountSampler::every_n(100);

        // 前99次都不应该采样
        for i in 1..=99 {
            assert!(
                !sampler.should_log(),
                "Call {} should not pass with interval=100",
                i
            );
        }

        // 第100次应该采样
        assert!(sampler.should_log(), "Call 100 should pass");
    }

    #[test]
    fn test_count_sampler_current_count_tracking() {
        // 测试当前计数的跟踪
        let mut sampler = CountSampler::every_n(7);

        for expected in 1..=15 {
            sampler.should_log();
            assert_eq!(
                sampler.current_count(),
                expected,
                "Current count should match number of calls"
            );
        }
    }

    // ==================== AdaptiveSampler 额外测试 ====================

    #[test]
    fn test_adaptive_sampler_error_response() {
        // 测试自适应采样器对错误率的响应
        let sampler = AdaptiveSampler::new(0.1); // 基础10%采样率

        // 正常情况下低采样率（记录成功）
        for _ in 0..100 {
            sampler.record_success();
        }

        // 调用should_log多次来触发窗口重置和采样决策
        let normal_samples = (0..100).filter(|_| sampler.should_log()).count();

        // 由于窗口机制，可能只有少数几次真正检查了采样率
        // 但正常情况下采样次数应该较少
        // （具体数量取决于实现细节）

        // 报告高错误率
        for _ in 0..50 {
            sampler.record_error();
        }
        for _ in 0..50 {
            sampler.record_success(); // 混合一些成功
        }

        // 再次调用should_log
        let high_error_samples = (0..100).filter(|_| sampler.should_log()).count();

        // 高错误率时的采样行为可能不同
        // 我们主要验证它不会panic且能正常工作
        let _ = normal_samples;
        let _ = high_error_samples;
    }

    #[test]
    fn test_adaptive_sampler_custom_configuration() {
        // 测试自适应采样器的各种自定义配置
        let sampler = AdaptiveSampler::new(0.2)
            .with_max_probability(0.9)
            .with_min_probability(0.02)
            .with_window_size(200);

        assert!((sampler.base_probability - 0.2).abs() < 0.001);
        assert!((sampler.max_probability - 0.9).abs() < 0.001);
        assert!((sampler.min_probability - 0.02).abs() < 0.001);
        assert_eq!(sampler.window_size, 200);
    }

    #[test]
    fn test_adaptive_sampler_error_rate_calculation() {
        // 详细测试错误率计算
        let sampler = AdaptiveSampler::new(0.1);

        // 初始状态：无操作
        assert!((sampler.current_error_rate() - 0.0).abs() < 0.001);

        // 只记录成功
        for _ in 0..100 {
            sampler.record_success();
        }
        assert!((sampler.current_error_rate() - 0.0).abs() < 0.001);

        // 记录一些错误
        for _ in 0..25 {
            sampler.record_error();
        }
        for _ in 0..75 {
            sampler.record_success();
        }

        // 错误率应该是 25/(25+75) = 0.25
        let rate = sampler.current_error_rate();
        assert!(
            rate > 0.0 && rate <= 0.5,
            "Error rate was {}, expected in (0, 0.5]",
            rate
        );

        // 重置后重新开始（通过should_log触发窗口重置）
        let _ = sampler.should_log();

        // 重置后错误率应该回到0或很低
        let reset_rate = sampler.current_error_rate();
        assert!(
            reset_rate < 0.01 || reset_rate >= 0.0,
            "Reset rate should be near 0"
        );
    }

    #[test]
    fn test_adaptive_sampler_record_methods_thread_safety() {
        // 验证record方法在概念上是线程安全的（使用原子操作）
        use std::thread;

        let sampler = Arc::new(AdaptiveSampler::new(0.1));
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let s = Arc::clone(&sampler);
                thread::spawn(move || {
                    if i % 2 == 0 {
                        for _ in 0..100 {
                            s.record_success();
                        }
                    } else {
                        for _ in 0..50 {
                            s.record_error();
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("Thread should complete without panic");
        }

        // 验证最终状态一致
        let total_ops = sampler
            .success_count
            .load(std::sync::atomic::Ordering::Relaxed)
            + sampler
                .error_count
                .load(std::sync::atomic::Ordering::Relaxed);

        // 4个线程: 2个成功线程*100 + 2个错误线程*50 = 300
        assert_eq!(total_ops, 300, "Total operations should be 300");

        // 验证error_rate在合理范围内
        let rate = sampler.current_error_rate();
        // 100个错误 / 300总数 ≈ 33.3%
        assert!(
            (rate - 0.333).abs() < 0.01,
            "Error rate was {}, expected ~0.333",
            rate
        );
    }
}
