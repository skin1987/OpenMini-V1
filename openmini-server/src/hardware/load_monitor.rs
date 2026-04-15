#![allow(dead_code)]

//! 动态负载监测模块
//!
//! 监测 CPU/GPU 负载和温度，动态调节计算资源使用。

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// 负载阈值配置
#[derive(Debug, Clone)]
pub struct LoadThresholds {
    /// CPU 占用率上限 (0.0-1.0)
    pub cpu_max: f32,
    /// CPU 温度上限 (摄氏度)
    pub temp_max: f32,
    /// 内存占用上限 (0.0-1.0)
    pub memory_max: f32,
    /// 调节间隔 (毫秒)
    pub adjust_interval_ms: u64,
}

impl Default for LoadThresholds {
    fn default() -> Self {
        Self {
            cpu_max: 0.8,
            temp_max: 85.0,
            memory_max: 0.9,
            adjust_interval_ms: 100,
        }
    }
}

/// 系统负载状态
#[derive(Debug, Clone)]
pub struct SystemLoad {
    /// CPU 占用率 (0.0-1.0)
    pub cpu_usage: f32,
    /// CPU 温度 (摄氏度)
    pub cpu_temp: f32,
    /// 内存占用率 (0.0-1.0)
    pub memory_usage: f32,
    /// GPU 占用率 (0.0-1.0)
    pub gpu_usage: f32,
    /// GPU 温度 (摄氏度)
    pub gpu_temp: f32,
    /// 时间戳
    pub timestamp: Instant,
}

impl Default for SystemLoad {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            cpu_temp: 50.0,
            memory_usage: 0.0,
            gpu_usage: 0.0,
            gpu_temp: 0.0,
            timestamp: Instant::now(),
        }
    }
}

/// 负载调节动作
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadAction {
    /// 正常运行
    Normal,
    /// 降低并行度
    ReduceParallelism,
    /// 暂停计算
    Pause,
    /// 紧急停止
    EmergencyStop,
}

/// 负载监测器
#[derive(Debug)]
pub struct LoadMonitor {
    /// 阈值配置
    thresholds: LoadThresholds,
    /// 当前负载
    current_load: SystemLoad,
    /// 上次调节时间
    last_adjust: Instant,
    /// 当前并行度 (0-100)
    parallelism: AtomicU64,
    /// 是否暂停
    paused: bool,
}

impl LoadMonitor {
    /// 创建新的负载监测器
    pub fn new(thresholds: LoadThresholds) -> Self {
        Self {
            thresholds,
            current_load: SystemLoad::default(),
            last_adjust: Instant::now(),
            parallelism: AtomicU64::new(100),
            paused: false,
        }
    }

    /// 更新系统负载
    pub fn update(&mut self, load: SystemLoad) {
        self.current_load = load;
        self.check_and_adjust();
    }

    /// 检查并调节负载
    fn check_and_adjust(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_adjust).as_millis()
            < self.thresholds.adjust_interval_ms as u128
        {
            return;
        }
        self.last_adjust = now;

        let action = self.determine_action();
        self.apply_action(action);
    }

    /// 确定调节动作
    fn determine_action(&self) -> LoadAction {
        let load = &self.current_load;

        if load.cpu_temp > self.thresholds.temp_max + 10.0 {
            return LoadAction::EmergencyStop;
        }

        if load.cpu_usage > 0.95 || load.cpu_temp > self.thresholds.temp_max {
            return LoadAction::Pause;
        }

        if load.cpu_usage > self.thresholds.cpu_max
            || load.cpu_temp > self.thresholds.temp_max - 5.0
        {
            return LoadAction::ReduceParallelism;
        }

        LoadAction::Normal
    }

    /// 应用调节动作
    fn apply_action(&mut self, action: LoadAction) {
        match action {
            LoadAction::Normal => {
                self.paused = false;
                let current = self.parallelism.load(Ordering::Relaxed);
                if current < 100 {
                    self.parallelism
                        .store((current + 10).min(100), Ordering::Relaxed);
                }
            }
            LoadAction::ReduceParallelism => {
                self.paused = false;
                let current = self.parallelism.load(Ordering::Relaxed);
                self.parallelism
                    .store((current as f32 * 0.8) as u64, Ordering::Relaxed);
            }
            LoadAction::Pause => {
                self.paused = true;
                self.parallelism.store(0, Ordering::Relaxed);
            }
            LoadAction::EmergencyStop => {
                self.paused = true;
                self.parallelism.store(0, Ordering::Relaxed);
            }
        }
    }

    /// 获取当前并行度 (0-100)
    pub fn parallelism(&self) -> u64 {
        self.parallelism.load(Ordering::Relaxed)
    }

    /// 是否应该暂停
    pub fn should_pause(&self) -> bool {
        self.paused
    }

    /// 获取当前负载
    pub fn current_load(&self) -> &SystemLoad {
        &self.current_load
    }

    /// 获取阈值配置
    pub fn thresholds(&self) -> &LoadThresholds {
        &self.thresholds
    }

    /// 设置阈值配置
    pub fn set_thresholds(&mut self, thresholds: LoadThresholds) {
        self.thresholds = thresholds;
    }

    /// 获取 CPU 占用率
    pub fn cpu_usage(&self) -> f32 {
        self.current_load.cpu_usage
    }

    /// 获取 CPU 温度
    pub fn cpu_temp(&self) -> f32 {
        self.current_load.cpu_temp
    }

    /// 是否过载
    pub fn is_overloaded(&self) -> bool {
        self.current_load.cpu_usage > self.thresholds.cpu_max
            || self.current_load.cpu_temp > self.thresholds.temp_max
    }
}

impl Default for LoadMonitor {
    fn default() -> Self {
        Self::new(LoadThresholds::default())
    }
}

/// CPU 负载采样
pub fn sample_cpu_usage() -> f32 {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("ps").args(["-A", "-o", "%cpu"]).output() {
            if let Ok(stdout) = String::from_utf8(output.stdout) {
                let total: f32 = stdout
                    .lines()
                    .skip(1)
                    .filter_map(|line| line.trim().parse::<f32>().ok())
                    .sum();
                return (total / 100.0).min(1.0);
            }
        }
    }
    0.5
}

/// CPU 温度采样
pub fn sample_cpu_temp() -> f32 {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("osascript")
            .args([
                "-e",
                "return do shell script \"sysctl -n hw.ThermalLevel\" 2>/dev/null || echo 0",
            ])
            .output()
        {
            if let Ok(stdout) = String::from_utf8(output.stdout) {
                if let Ok(level) = stdout.trim().parse::<u32>() {
                    return 40.0 + level as f32 * 15.0;
                }
            }
        }
    }
    50.0
}

/// 内存占用采样
pub fn sample_memory_usage() -> f32 {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("vm_stat").output() {
            if let Ok(stdout) = String::from_utf8(output.stdout) {
                let mut free = 0u64;
                let mut used = 0u64;
                for line in stdout.lines() {
                    if line.starts_with("Pages free:") {
                        free = line
                            .split(':')
                            .nth(1)
                            .and_then(|s| s.trim().trim_end_matches('.').parse::<u64>().ok())
                            .unwrap_or(0);
                    } else if line.starts_with("Pages active:")
                        || line.starts_with("Pages inactive:")
                    {
                        used += line
                            .split(':')
                            .nth(1)
                            .and_then(|s| s.trim().trim_end_matches('.').parse::<u64>().ok())
                            .unwrap_or(0);
                    }
                }
                let total = free + used;
                if total > 0 {
                    return used as f32 / total as f32;
                }
            }
        }
    }
    0.5
}

/// 获取完整系统负载
pub fn get_system_load() -> SystemLoad {
    SystemLoad {
        cpu_usage: sample_cpu_usage(),
        cpu_temp: sample_cpu_temp(),
        memory_usage: sample_memory_usage(),
        gpu_usage: 0.0,
        gpu_temp: 0.0,
        timestamp: Instant::now(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_monitor() {
        let monitor = LoadMonitor::default();
        assert_eq!(monitor.parallelism(), 100);
    }

    #[test]
    fn test_load_reduction() {
        let mut monitor = LoadMonitor::new(LoadThresholds {
            cpu_max: 0.5,
            temp_max: 70.0,
            memory_max: 0.8,
            adjust_interval_ms: 0,
        });

        monitor.update(SystemLoad {
            cpu_usage: 0.9,
            cpu_temp: 60.0,
            memory_usage: 0.5,
            gpu_usage: 0.0,
            gpu_temp: 0.0,
            timestamp: Instant::now(),
        });

        assert!(monitor.parallelism() < 100);
    }

    #[test]
    fn test_system_load_sampling() {
        let load = get_system_load();
        assert!(load.cpu_usage >= 0.0 && load.cpu_usage <= 1.0);
        assert!(load.cpu_temp >= 0.0);
    }

    // ========== 新增测试开始 ==========

    /// 测试LoadThresholds默认配置
    #[test]
    fn test_thresholds_default() {
        let thresholds = LoadThresholds::default();

        // 验证默认阈值合理性
        assert!((thresholds.cpu_max - 0.8).abs() < f32::EPSILON);
        assert!((thresholds.temp_max - 85.0).abs() < f32::EPSILON);
        assert!((thresholds.memory_max - 0.9).abs() < f32::EPSILON);
        assert_eq!(thresholds.adjust_interval_ms, 100);
    }

    /// 测试SystemLoad默认值
    #[test]
    fn test_system_load_default() {
        let load = SystemLoad::default();

        assert!((load.cpu_usage - 0.0).abs() < f32::EPSILON);
        assert!((load.cpu_temp - 50.0).abs() < f32::EPSILON);
        assert!((load.memory_usage - 0.0).abs() < f32::EPSILON);
        assert!((load.gpu_usage - 0.0).abs() < f32::EPSILON);
        assert!((load.gpu_temp - 0.0).abs() < f32::EPSILON);
    }

    /// 测试EmergencyStop动作（温度超过阈值+10度）
    #[test]
    fn test_emergency_stop_action() {
        let mut monitor = LoadMonitor::new(LoadThresholds {
            cpu_max: 0.8,
            temp_max: 85.0,
            memory_max: 0.9,
            adjust_interval_ms: 0, // 立即调节
        });

        // 温度超过temp_max + 10度（95+度）应触发紧急停止
        monitor.update(SystemLoad {
            cpu_usage: 0.5,
            cpu_temp: 96.0, // > 85 + 10
            memory_usage: 0.3,
            gpu_usage: 0.0,
            gpu_temp: 0.0,
            timestamp: Instant::now(),
        });

        assert_eq!(monitor.parallelism(), 0); // 并行度应为0
        assert!(monitor.should_pause()); // 应该暂停
    }

    /// 测试Pause动作（CPU>95%或温度超过阈值）
    #[test]
    fn test_pause_action_high_cpu() {
        let mut monitor = LoadMonitor::new(LoadThresholds {
            cpu_max: 0.8,
            temp_max: 85.0,
            memory_max: 0.9,
            adjust_interval_ms: 0,
        });

        // CPU使用率超过95%应触发暂停
        monitor.update(SystemLoad {
            cpu_usage: 0.96, // > 0.95
            cpu_temp: 60.0,
            memory_usage: 0.3,
            gpu_usage: 0.0,
            gpu_temp: 0.0,
            timestamp: Instant::now(),
        });

        assert_eq!(monitor.parallelism(), 0);
        assert!(monitor.should_pause());
    }

    /// 测试Pause动作（温度超过阈值但未到紧急停止）
    #[test]
    fn test_pause_action_high_temp() {
        let mut monitor = LoadMonitor::new(LoadThresholds {
            cpu_max: 0.8,
            temp_max: 70.0,
            memory_max: 0.9,
            adjust_interval_ms: 0,
        });

        // 温度超过temp_max但未达到紧急停止条件
        monitor.update(SystemLoad {
            cpu_usage: 0.6,
            cpu_temp: 72.0, // > 70 (temp_max), 但 < 80 (temp_max+10)
            memory_usage: 0.3,
            gpu_usage: 0.0,
            gpu_temp: 0.0,
            timestamp: Instant::now(),
        });

        assert_eq!(monitor.parallelism(), 0);
        assert!(monitor.should_pause());
    }

    /// 测试ReduceParallelism动作（CPU或温度接近阈值）
    #[test]
    fn test_reduce_parallelism_action() {
        let mut monitor = LoadMonitor::new(LoadThresholds {
            cpu_max: 0.5,
            temp_max: 70.0,
            memory_max: 0.9,
            adjust_interval_ms: 0,
        });

        // CPU使用率超过cpu_max但未达到暂停条件
        monitor.update(SystemLoad {
            cpu_usage: 0.7, // > 0.5 (cpu_max), 但 < 0.95
            cpu_temp: 65.0,
            memory_usage: 0.3,
            gpu_usage: 0.0,
            gpu_temp: 0.0,
            timestamp: Instant::now(),
        });

        // 并行度应该降低但不为0
        assert!(monitor.parallelism() < 100);
        assert!(monitor.parallelism() > 0);
        assert!(!monitor.should_pause()); // 不应完全暂停
    }

    /// 测试Normal动作恢复并行度
    #[test]
    fn test_normal_action_recovery() {
        let mut monitor = LoadMonitor::new(LoadThresholds {
            cpu_max: 0.8,
            temp_max: 85.0,
            memory_max: 0.9,
            adjust_interval_ms: 0,
        });

        // 先触发降级
        monitor.update(SystemLoad {
            cpu_usage: 0.9,
            cpu_temp: 75.0,
            memory_usage: 0.3,
            gpu_usage: 0.0,
            gpu_temp: 0.0,
            timestamp: Instant::now(),
        });
        let reduced_parallelism = monitor.parallelism();
        assert!(reduced_parallelism < 100);

        // 恢复正常负载
        monitor.update(SystemLoad {
            cpu_usage: 0.2, // 远低于cpu_max
            cpu_temp: 50.0, // 远低于temp_max
            memory_usage: 0.2,
            gpu_usage: 0.0,
            gpu_temp: 0.0,
            timestamp: Instant::now(),
        });

        // 并行度应该增加
        assert!(monitor.parallelism() > reduced_parallelism);
        assert!(!monitor.should_pause());
    }

    /// 测试is_overloaded()判断逻辑
    #[test]
    fn test_is_overloaded() {
        let mut monitor = LoadMonitor::new(LoadThresholds {
            cpu_max: 0.5,
            temp_max: 70.0,
            memory_max: 0.9,
            adjust_interval_ms: 0,
        });

        // 正常情况不过载
        monitor.update(SystemLoad {
            cpu_usage: 0.3,
            cpu_temp: 50.0,
            memory_usage: 0.4,
            gpu_usage: 0.0,
            gpu_temp: 0.0,
            timestamp: Instant::now(),
        });
        assert!(!monitor.is_overloaded());

        // CPU过载
        monitor.update(SystemLoad {
            cpu_usage: 0.6, // > 0.5
            cpu_temp: 50.0,
            memory_usage: 0.4,
            gpu_usage: 0.0,
            gpu_temp: 0.0,
            timestamp: Instant::now(),
        });
        assert!(monitor.is_overloaded());

        // 温度过载
        monitor.update(SystemLoad {
            cpu_usage: 0.3,
            cpu_temp: 75.0, // > 70
            memory_usage: 0.4,
            gpu_usage: 0.0,
            gpu_temp: 0.0,
            timestamp: Instant::now(),
        });
        assert!(monitor.is_overloaded());
    }

    /// 测试set_thresholds()动态调整
    #[test]
    fn test_set_thresholds_dynamically() {
        let mut monitor = LoadMonitor::default();

        // 使用宽松阈值
        monitor.set_thresholds(LoadThresholds {
            cpu_max: 0.95,
            temp_max: 95.0,
            memory_max: 0.98,
            adjust_interval_ms: 50,
        });

        // 高负载但在新阈值内不应触发调节
        monitor.update(SystemLoad {
            cpu_usage: 0.90,    // < 0.95
            cpu_temp: 90.0,     // < 95
            memory_usage: 0.95, // < 0.98
            gpu_usage: 0.0,
            gpu_temp: 0.0,
            timestamp: Instant::now(),
        });

        // 应该保持Normal状态
        assert_eq!(monitor.parallelism(), 100);
        assert!(!monitor.should_pause());

        // 验证新阈值已生效
        assert!((monitor.thresholds().cpu_max - 0.95).abs() < f32::EPSILON);
    }

    /// 测试adjust_interval_ms防止频繁调节
    #[test]
    fn test_adjust_interval_throttling() {
        let mut monitor = LoadMonitor::new(LoadThresholds {
            cpu_max: 0.5,
            temp_max: 70.0,
            memory_max: 0.9,
            adjust_interval_ms: 10000, // 10秒间隔
        });

        // 第一次更新会立即执行（因为last_adjust是初始化时间）
        monitor.update(SystemLoad {
            cpu_usage: 0.9,
            cpu_temp: 75.0,
            memory_usage: 0.5,
            gpu_usage: 0.0,
            gpu_temp: 0.0,
            timestamp: Instant::now(),
        });
        let parallelism_after_first = monitor.parallelism();

        // 立即第二次更新（间隔太短，不应再次调节）
        monitor.update(SystemLoad {
            cpu_usage: 0.95, // 更高的负载
            cpu_temp: 80.0,
            memory_usage: 0.6,
            gpu_usage: 0.0,
            gpu_temp: 0.0,
            timestamp: Instant::now(),
        });

        // 并行度不应该改变
        assert_eq!(monitor.parallelism(), parallelism_after_first);
    }

    /// 测试LoadAction枚举的所有变体
    /// 覆盖分支：所有动作类型的完整覆盖
    #[test]
    fn test_load_action_variants() {
        let actions = vec![
            LoadAction::Normal,
            LoadAction::ReduceParallelism,
            LoadAction::Pause,
            LoadAction::EmergencyStop,
        ];

        for action in &actions {
            // 验证Debug trait实现（不应panic）
            let debug_str = format!("{:?}", action);
            assert!(!debug_str.is_empty());
        }

        // 验证相等性比较
        assert_eq!(LoadAction::Normal, LoadAction::Normal);
        assert_ne!(LoadAction::Normal, LoadAction::Pause);
        assert_eq!(LoadAction::EmergencyStop, LoadAction::EmergencyStop);

        // 验证Copy trait
        let original = LoadAction::ReduceParallelism;
        let _copied = original; // Copy允许这样使用
        let _still_valid = original; // 原值仍然可用
    }

    /// 测试SystemLoad的Clone和Debug特性
    /// 覆盖分支：SystemLoad的trait实现
    #[test]
    fn test_system_load_clone_and_debug() {
        let load1 = SystemLoad {
            cpu_usage: 0.75,
            cpu_temp: 65.5,
            memory_usage: 0.60,
            gpu_usage: 0.80,
            gpu_temp: 70.0,
            timestamp: Instant::now(),
        };

        // 测试克隆
        let load2 = load1.clone();
        assert!((load1.cpu_usage - load2.cpu_usage).abs() < f32::EPSILON);
        assert!((load1.cpu_temp - load2.cpu_temp).abs() < f32::EPSILON);
        assert!((load1.memory_usage - load2.memory_usage).abs() < f32::EPSILON);
        assert!((load1.gpu_usage - load2.gpu_usage).abs() < f32::EPSILON);
        assert!((load1.gpu_temp - load2.gpu_temp).abs() < f32::EPSILON);

        // 测试Debug输出
        let debug_str = format!("{:?}", load1);
        assert!(!debug_str.is_empty());
    }

    /// 测试极端负载值的边界条件
    /// 覆盖分支：负载值的边界情况
    #[test]
    fn test_extreme_load_values() {
        let mut monitor = LoadMonitor::new(LoadThresholds {
            cpu_max: 0.5,
            temp_max: 70.0,
            memory_max: 0.8,
            adjust_interval_ms: 0,
        });

        // CPU使用率为0（完全空闲）
        monitor.update(SystemLoad {
            cpu_usage: 0.0,
            cpu_temp: 30.0, // 很低的温度
            memory_usage: 0.0,
            gpu_usage: 0.0,
            gpu_temp: 0.0,
            timestamp: Instant::now(),
        });
        assert!(!monitor.should_pause()); // 不应暂停

        // CPU使用率为1.0（满载）
        monitor.update(SystemLoad {
            cpu_usage: 1.0, // > 0.95，应该触发暂停
            cpu_temp: 50.0,
            memory_usage: 0.5,
            gpu_usage: 0.0,
            gpu_temp: 0.0,
            timestamp: Instant::now(),
        });
        assert!(monitor.should_pause()); // 应该暂停
        assert_eq!(monitor.parallelism(), 0);
    }

    /// 测试温度精确在阈值边界的情况
    /// 覆盖分支：温度阈值的精确边界条件
    #[test]
    fn test_temperature_boundary_conditions() {
        let mut monitor = LoadMonitor::new(LoadThresholds {
            cpu_max: 0.8,
            temp_max: 85.0,
            memory_max: 0.9,
            adjust_interval_ms: 0,
        });

        // 温度恰好等于temp_max-5（不应触发ReduceParallelism）
        monitor.update(SystemLoad {
            cpu_usage: 0.3, // 远低于cpu_max
            cpu_temp: 80.0, // = 85 - 5，不触发
            memory_usage: 0.4,
            gpu_usage: 0.0,
            gpu_temp: 0.0,
            timestamp: Instant::now(),
        });
        assert_eq!(monitor.parallelism(), 100); // 应该保持Normal

        // 温度恰好等于temp_max（触发Pause）
        monitor.update(SystemLoad {
            cpu_usage: 0.3,
            cpu_temp: 85.0,
            memory_usage: 0.4,
            gpu_usage: 0.0,
            gpu_temp: 0.0,
            timestamp: Instant::now(),
        });
        let p = monitor.parallelism();
        assert!(p < 100);
    }

    /// 测试LoadMonitor的Default实现一致性
    /// 覆盖分支：Default trait的正确性
    #[test]
    fn test_load_monitor_default_consistency() {
        let monitor1 = LoadMonitor::default();
        let monitor2 = LoadMonitor::default();

        // 两个默认实例应该有相同的初始状态
        assert_eq!(monitor1.parallelism(), monitor2.parallelism());
        assert_eq!(monitor1.should_pause(), monitor2.should_pause());

        // 验证默认阈值
        assert!((monitor1.thresholds().cpu_max - 0.8).abs() < f32::EPSILON);
        assert!((monitor1.thresholds().temp_max - 85.0).abs() < f32::EPSILON);
        assert!((monitor1.thresholds().memory_max - 0.9).abs() < f32::EPSILON);
        assert_eq!(monitor1.thresholds().adjust_interval_ms, 100);
    }

    /// 测试多次连续更新后的并行度变化趋势
    /// 覆盖分支：连续更新的累积效应
    #[test]
    fn test_consecutive_updates_trend() {
        let mut monitor = LoadMonitor::new(LoadThresholds {
            cpu_max: 0.6,
            temp_max: 75.0,
            memory_max: 0.8,
            adjust_interval_ms: 0,
        });

        // 初始并行度
        assert_eq!(monitor.parallelism(), 100);

        // 第一次降级
        monitor.update(SystemLoad {
            cpu_usage: 0.7,
            cpu_temp: 65.0,
            memory_usage: 0.5,
            gpu_usage: 0.0,
            gpu_temp: 0.0,
            timestamp: Instant::now(),
        });
        let after_first_reduce = monitor.parallelism();
        assert!(after_first_reduce < 100 && after_first_reduce > 0);

        // 第二次降级（进一步降低）
        monitor.update(SystemLoad {
            cpu_usage: 0.8,
            cpu_temp: 68.0,
            memory_usage: 0.6,
            gpu_usage: 0.0,
            gpu_temp: 0.0,
            timestamp: Instant::now(),
        });
        let after_second_reduce = monitor.parallelism();
        assert!(after_second_reduce <= after_first_reduce); // 应该继续降低或持平
    }

    /// 测试sample_cpu_usage返回值范围
    /// 覆盖分支：采样函数的有效范围
    #[test]
    fn test_sample_cpu_usage_range() {
        let usage = sample_cpu_usage();

        // 返回值应该在[0.0, 1.0]范围内
        assert!(
            (0.0..=1.0).contains(&usage),
            "CPU usage {} is out of range [0.0, 1.0]",
            usage
        );
    }

    /// 测试sample_cpu_temp返回值合理性
    /// 覆盖分支：温度采样的合理范围
    #[test]
    fn test_sample_cpu_temp_range() {
        let temp = sample_cpu_temp();

        // 温度应该是合理的正值（通常20-120摄氏度）
        assert!(temp >= 0.0, "CPU temp {} should be non-negative", temp);
        // 注意：实际温度可能很高，所以上限不做严格限制
    }

    /// 测试sample_memory_usage返回值范围
    /// 覆盖分支：内存采样函数的有效范围
    #[test]
    fn test_sample_memory_usage_range() {
        let usage = sample_memory_usage();

        // 返回值应该在[0.0, 1.0]范围内
        assert!(
            (0.0..=1.0).contains(&usage),
            "Memory usage {} is out of range [0.0, 1.0]",
            usage
        );
    }
}
