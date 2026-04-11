//! 内存监控器 - 内存使用追踪和限制
//!
//! 提供内存使用监控功能，支持内存限制和峰值追踪。
//! 使用原子操作实现线程安全。

#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

const MAX_MEMORY_GB: usize = 12;
const GB: usize = 1024 * 1024 * 1024;
const DEFAULT_WARNING_THRESHOLD: f64 = 70.0;
const DEFAULT_CRITICAL_THRESHOLD: f64 = 90.0;
const HISTORY_SIZE: usize = 100;

pub type AllocationCallback = Arc<dyn Fn(usize, usize) + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPressure {
    Normal,
    Warning,
    Critical,
}

#[derive(Debug, Clone)]
pub struct Snapshot {
    pub current_usage: usize,
    pub peak_usage: usize,
    pub max_memory: usize,
    pub usage_percent: f64,
    pub available: usize,
    pub pressure: MemoryPressure,
}

impl Snapshot {
    pub fn new(
        current_usage: usize,
        peak_usage: usize,
        max_memory: usize,
        pressure: MemoryPressure,
    ) -> Self {
        let usage_percent = if max_memory > 0 {
            (current_usage as f64 / max_memory as f64) * 100.0
        } else {
            0.0
        };
        let available = max_memory.saturating_sub(current_usage);
        Self {
            current_usage,
            peak_usage,
            max_memory,
            usage_percent,
            available,
            pressure,
        }
    }
}

/// 内存监控器
///
/// 线程安全的内存使用追踪器，支持:
/// - 内存分配/释放计数
/// - 峰值使用记录
/// - 内存限制检查
pub struct MemoryMonitor {
    max_memory: usize,
    current_usage: AtomicUsize,
    peak_usage: AtomicUsize,
    warning_threshold: RwLock<f64>,
    critical_threshold: RwLock<f64>,
    callbacks: RwLock<Vec<AllocationCallback>>,
    history: RwLock<VecDeque<f64>>,
}

impl MemoryMonitor {
    pub fn new(max_memory: usize) -> Self {
        Self {
            max_memory,
            current_usage: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
            warning_threshold: RwLock::new(DEFAULT_WARNING_THRESHOLD),
            critical_threshold: RwLock::new(DEFAULT_CRITICAL_THRESHOLD),
            callbacks: RwLock::new(Vec::new()),
            history: RwLock::new(VecDeque::with_capacity(HISTORY_SIZE)),
        }
    }

    /// 创建默认监控器(12GB 限制)
    pub fn default_monitor() -> Self {
        Self::new(MAX_MEMORY_GB * GB)
    }

    pub fn allocate(&self, size: usize) -> Result<(), String> {
        loop {
            let current = self.current_usage.load(Ordering::Relaxed);
            let new_usage = current + size;

            if new_usage > self.max_memory {
                return Err(format!(
                    "Memory limit exceeded: {} + {} > {}",
                    current, size, self.max_memory
                ));
            }

            match self.current_usage.compare_exchange_weak(
                current,
                new_usage,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.update_peak(new_usage);
                    self.trigger_callbacks(current, new_usage);
                    return Ok(());
                }
                Err(_) => continue,
            }
        }
    }

    pub fn deallocate(&self, size: usize) {
        loop {
            let current = self.current_usage.load(Ordering::Relaxed);
            let new_usage = current.saturating_sub(size);

            match self.current_usage.compare_exchange_weak(
                current,
                new_usage,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.record_history(current);
                    return;
                }
                Err(_) => continue,
            }
        }
    }

    /// 更新峰值使用量
    fn update_peak(&self, usage: usize) {
        loop {
            let peak = self.peak_usage.load(Ordering::Relaxed);
            if usage <= peak {
                return;
            }

            match self.peak_usage.compare_exchange_weak(
                peak,
                usage,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(_) => continue,
            }
        }
    }

    /// 检查是否还有可用内存
    pub fn check_limit(&self) -> bool {
        self.current_usage.load(Ordering::Relaxed) < self.max_memory
    }

    /// 获取当前使用量
    pub fn usage(&self) -> usize {
        self.current_usage.load(Ordering::Relaxed)
    }

    /// 获取峰值使用量
    pub fn peak_usage(&self) -> usize {
        self.peak_usage.load(Ordering::Relaxed)
    }

    /// 获取可用内存
    pub fn available(&self) -> usize {
        self.max_memory
            .saturating_sub(self.current_usage.load(Ordering::Relaxed))
    }

    /// 获取使用率百分比
    pub fn usage_percent(&self) -> f64 {
        if self.max_memory == 0 {
            return 0.0;
        }
        let usage = self.current_usage.load(Ordering::Relaxed);
        (usage as f64 / self.max_memory as f64) * 100.0
    }

    /// 获取最大内存限制
    pub fn max_memory(&self) -> usize {
        self.max_memory
    }

    /// 获取最大内存限制(GB)
    pub fn max_memory_gb(&self) -> f64 {
        self.max_memory as f64 / GB as f64
    }

    pub fn check_pressure(&self) -> MemoryPressure {
        let percent = self.usage_percent();
        let critical = self.critical_threshold.read().ok().map(|r| *r).unwrap_or(DEFAULT_CRITICAL_THRESHOLD);
        let warning = self.warning_threshold.read().ok().map(|r| *r).unwrap_or(DEFAULT_WARNING_THRESHOLD);

        if percent >= critical {
            MemoryPressure::Critical
        } else if percent >= warning {
            MemoryPressure::Warning
        } else {
            MemoryPressure::Normal
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        let pressure = self.check_pressure();
        Snapshot::new(self.usage(), self.peak_usage(), self.max_memory, pressure)
    }

    pub fn register_callback(&self, callback: AllocationCallback) {
        if let Ok(mut callbacks) = self.callbacks.write() {
            callbacks.push(callback);
        }
    }

    fn trigger_callbacks(&self, old_usage: usize, new_usage: usize) {
        let current_percent = self.usage_percent();
        let warning = self.warning_threshold.read().ok().map(|r| *r).unwrap_or(DEFAULT_WARNING_THRESHOLD);

        if current_percent >= warning {
            if let Ok(callbacks) = self.callbacks.read() {
                for callback in callbacks.iter() {
                    callback(old_usage, new_usage);
                }
            }
        }
    }

    fn record_history(&self, usage: usize) {
        if self.max_memory == 0 {
            return;
        }
        let percent = (usage as f64 / self.max_memory as f64) * 100.0;
        if let Ok(mut history) = self.history.write() {
            if history.len() >= HISTORY_SIZE {
                history.pop_front();
            }
            history.push_back(percent);
        }
    }

    pub fn auto_adjust_thresholds(&self) {
        let history = match self.history.read() {
            Ok(h) => h,
            Err(_) => return,
        };
        if history.len() < HISTORY_SIZE / 2 {
            return;
        }

        let avg_percent: f64 = history.iter().sum::<f64>() / history.len() as f64;
        let max_percent = history.iter().cloned().fold(0.0_f64, f64::max);

        if let (Ok(mut warning), Ok(mut critical)) = (self.warning_threshold.write(), self.critical_threshold.write()) {
            if avg_percent > 60.0 {
                *warning = (avg_percent + 10.0).min(85.0);
                *critical = (max_percent + 5.0).min(95.0);
            } else if avg_percent < 40.0 && *warning > 60.0 {
                *warning = 70.0;
                *critical = 90.0;
            }
        }
    }

    pub fn warning_threshold(&self) -> f64 {
        self.warning_threshold.read().ok().map(|r| *r).unwrap_or(DEFAULT_WARNING_THRESHOLD)
    }

    pub fn critical_threshold(&self) -> f64 {
        *self.critical_threshold.read().unwrap()
    }

    pub fn set_warning_threshold(&self, threshold: f64) {
        let mut warning = self.warning_threshold.write().unwrap();
        *warning = threshold.clamp(0.0, 100.0);
    }

    pub fn set_critical_threshold(&self, threshold: f64) {
        let mut critical = self.critical_threshold.write().unwrap();
        *critical = threshold.clamp(0.0, 100.0);
    }
}

impl Default for MemoryMonitor {
    fn default() -> Self {
        Self::default_monitor()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_monitor_new() {
        let monitor = MemoryMonitor::new(1024);
        assert_eq!(monitor.max_memory(), 1024);
        assert_eq!(monitor.usage(), 0);
        assert_eq!(monitor.available(), 1024);
    }

    #[test]
    fn test_memory_monitor_allocate() {
        let monitor = MemoryMonitor::new(1024);

        assert!(monitor.allocate(512).is_ok());
        assert_eq!(monitor.usage(), 512);
        assert_eq!(monitor.available(), 512);

        assert!(monitor.allocate(512).is_ok());
        assert_eq!(monitor.usage(), 1024);
        assert_eq!(monitor.available(), 0);
    }

    #[test]
    fn test_memory_monitor_limit_exceeded() {
        let monitor = MemoryMonitor::new(1024);

        assert!(monitor.allocate(512).is_ok());

        let result = monitor.allocate(1024);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Memory limit exceeded"));
    }

    #[test]
    fn test_memory_monitor_deallocate() {
        let monitor = MemoryMonitor::new(1024);

        monitor.allocate(512).unwrap();
        assert_eq!(monitor.usage(), 512);

        monitor.deallocate(256);
        assert_eq!(monitor.usage(), 256);
    }

    #[test]
    fn test_memory_monitor_peak_usage() {
        let monitor = MemoryMonitor::new(1024);

        monitor.allocate(512).unwrap();
        assert_eq!(monitor.peak_usage(), 512);

        monitor.allocate(256).unwrap();
        assert_eq!(monitor.peak_usage(), 768);

        monitor.deallocate(256);
        assert_eq!(monitor.peak_usage(), 768);
    }

    #[test]
    fn test_memory_monitor_usage_percent() {
        let monitor = MemoryMonitor::new(1000);

        monitor.allocate(250).unwrap();
        assert!((monitor.usage_percent() - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_memory_monitor_check_limit() {
        let monitor = MemoryMonitor::new(1024);

        assert!(monitor.check_limit());

        monitor.allocate(1024).unwrap();
        assert!(!monitor.check_limit());
    }

    #[test]
    fn test_memory_monitor_default() {
        let monitor = MemoryMonitor::default();
        assert_eq!(monitor.max_memory_gb(), 12.0);
    }

    #[test]
    fn test_memory_monitor_zero_max_memory() {
        let monitor = MemoryMonitor::new(0);

        // 使用率应为 0.0，不会除零
        assert_eq!(monitor.usage_percent(), 0.0);

        // 可用内存应为 0
        assert_eq!(monitor.available(), 0);

        // 分配应失败
        let result = monitor.allocate(1);
        assert!(result.is_err());
    }

    // ==================== 分支覆盖率补充测试 ====================

    #[test]
    fn test_memory_monitor_pressure_levels() {
        let monitor = MemoryMonitor::new(1000);

        // 初始状态：Normal压力
        assert_eq!(monitor.check_pressure(), MemoryPressure::Normal);

        // 分配到70%以下：Normal
        monitor.allocate(600).unwrap();
        assert_eq!(monitor.check_pressure(), MemoryPressure::Normal);

        // 分配到70-90%之间：Warning
        monitor.deallocate(600);
        monitor.allocate(750).unwrap(); // 75%
        assert_eq!(monitor.check_pressure(), MemoryPressure::Warning);

        // 分配到90%以上：Critical
        monitor.deallocate(750);
        monitor.allocate(950).unwrap(); // 95%
        assert_eq!(monitor.check_pressure(), MemoryPressure::Critical);
    }

    #[test]
    fn test_memory_monitor_snapshot() {
        let monitor = MemoryMonitor::new(1024);

        monitor.allocate(512).unwrap();

        let snapshot = monitor.snapshot();

        assert_eq!(snapshot.current_usage, 512);
        assert_eq!(snapshot.peak_usage, 512);
        assert_eq!(snapshot.max_memory, 1024);
        assert!((snapshot.usage_percent - 50.0).abs() < 0.01);
        assert_eq!(snapshot.available, 512);
    }

    #[test]
    fn test_memory_monitor_threshold_management() {
        let monitor = MemoryMonitor::new(1000);

        // 获取默认阈值
        assert!((monitor.warning_threshold() - 70.0).abs() < 0.01);
        assert!((monitor.critical_threshold() - 90.0).abs() < 0.01);

        // 设置自定义阈值
        monitor.set_warning_threshold(60.0);
        monitor.set_critical_threshold(80.0);

        assert!((monitor.warning_threshold() - 60.0).abs() < 0.01);
        assert!((monitor.critical_threshold() - 80.0).abs() < 0.01);

        // 测试边界值clamp（负值应该被clamp到0）
        monitor.set_warning_threshold(-10.0);
        assert_eq!(monitor.warning_threshold(), 0.0);

        // 超过100的值应该被clamp到100
        monitor.set_critical_threshold(150.0);
        assert_eq!(monitor.critical_threshold(), 100.0);
    }

    #[test]
    fn test_memory_monitor_callback_trigger() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let monitor = Arc::new(MemoryMonitor::new(1000));
        let callback_triggered = Arc::new(AtomicBool::new(false));
        let callback_clone = callback_triggered.clone();

        // 注册回调
        monitor.register_callback(Arc::new(move |_old: usize, _new: usize| {
            callback_clone.store(true, Ordering::SeqCst);
        }));

        // 分配超过warning阈值（默认70%）
        monitor.allocate(800).unwrap();

        // 回调应该被触发
        assert!(callback_triggered.load(Ordering::Relaxed));
    }

    #[test]
    fn test_memory_monitor_max_memory_gb() {
        let monitor = MemoryMonitor::new(2 * 1024 * 1024 * 1024); // 2GB

        assert!((monitor.max_memory_gb() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_memory_monitor_deallocate_does_not_affect_peak() {
        let monitor = MemoryMonitor::new(1024);

        monitor.allocate(800).unwrap();
        assert_eq!(monitor.peak_usage(), 800);

        monitor.deallocate(400);
        // 峰值不应该减少
        assert_eq!(monitor.peak_usage(), 800);
        // 当前使用量应该减少
        assert_eq!(monitor.usage(), 400);
    }

    // ==================== 新增测试：达到 20+ 覆盖率 ====================

    /// 测试：Snapshot::new 在 max_memory=0 时 usage_percent 为 0.0（避免除零）
    #[test]
    fn test_snapshot_zero_max_memory() {
        // 覆盖 Snapshot::new 第44-48行的 max_memory==0 分支
        let snapshot = Snapshot::new(100, 200, 0, MemoryPressure::Normal);

        assert_eq!(snapshot.current_usage, 100);
        assert_eq!(snapshot.peak_usage, 200);
        assert_eq!(snapshot.max_memory, 0);
        assert_eq!(snapshot.usage_percent, 0.0); // 除零保护
        assert_eq!(snapshot.available, 0); // saturating_sub
    }

    /// 测试：record_history 在 max_memory=0 时提前返回（不记录历史）
    #[test]
    fn test_record_history_skips_when_max_memory_zero() {
        let monitor = MemoryMonitor::new(0);

        // 尝试分配会失败，但我们可以直接测试 history 是否为空
        // max_memory=0 时 record_history 应该提前返回
        monitor.deallocate(100); // 这会调用 record_history

        let history = monitor.history.read().unwrap();
        assert!(history.is_empty(), "max_memory=0时不应记录历史");
    }

    /// 测试：auto_adjust_thresholds 在历史数据不足时跳过调整
    #[test]
    fn test_auto_adjust_insufficient_history() {
        let monitor = MemoryMonitor::new(1000);

        // 历史数据不足（< HISTORY_SIZE/2 = 50），不应调整阈值
        let original_warning = monitor.warning_threshold();
        let original_critical = monitor.critical_threshold();

        monitor.auto_adjust_thresholds();

        // 阈值应该保持不变
        assert!((monitor.warning_threshold() - original_warning).abs() < 0.01);
        assert!((monitor.critical_threshold() - original_critical).abs() < 0.01);
    }

    /// 测试：deallocate 大于当前使用量时的 saturating_sub 行为
    #[test]
    fn test_deallocate_more_than_usage() {
        let monitor = MemoryMonitor::new(1024);

        monitor.allocate(256).unwrap();
        assert_eq!(monitor.usage(), 256);

        // 释放超过当前使用量，不应出现负值（saturating_sub）
        monitor.deallocate(512);
        assert_eq!(monitor.usage(), 0); // 应饱和到0
    }

    /// 测试：回调在低于 warning 阈值时不触发
    #[test]
    fn test_callback_not_triggered_below_threshold() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let monitor = Arc::new(MemoryMonitor::new(1000));
        let callback_triggered = Arc::new(AtomicBool::new(false));
        let callback_clone = callback_triggered.clone();

        monitor.register_callback(Arc::new(move |_old: usize, _new: usize| {
            callback_clone.store(true, Ordering::SeqCst);
        }));

        // 分配 500 (50%)，低于默认 warning 阈值 70%
        monitor.allocate(500).unwrap();

        // 回调不应该被触发
        assert!(!callback_triggered.load(Ordering::Relaxed));
    }

    /// 测试：多次连续分配和释放的稳定性
    #[test]
    fn test_multiple_allocate_deallocate_cycles() {
        let monitor = MemoryMonitor::new(1024);

        for i in 0..10 {
            let size = (i + 1) * 50;
            monitor.allocate(size).unwrap();
            assert_eq!(monitor.usage(), size);

            monitor.deallocate(size / 2);
            assert_eq!(monitor.usage(), size / 2);

            // 重置为下一次循环
            monitor.deallocate(size / 2);
            assert_eq!(monitor.usage(), 0);
        }
    }

    /// 测试：Snapshot 的 pressure 字段正确性验证
    #[test]
    fn test_snapshot_pressure_field() {
        let monitor = MemoryMonitor::new(1000);

        // Normal 压力状态下的 snapshot
        let snapshot_normal = monitor.snapshot();
        assert_eq!(snapshot_normal.pressure, MemoryPressure::Normal);

        // Critical 压力状态下的 snapshot
        monitor.allocate(950).unwrap(); // 95%
        let snapshot_critical = monitor.snapshot();
        assert_eq!(snapshot_critical.pressure, MemoryPressure::Critical);
    }
}
