//! Performance Regression Detection Module
//!
//! 提供性能回归自动检测功能：
//! - 加载基线数据
//! - 计算性能变化百分比
//! - PASS/FAIL/WARN 判定
//! - 生成回归报告

use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionThresholds {
    pub ttft_degradation_pct: f64,
    pub tpot_degradation_pct: f64,
    pub memory_growth_pct: f64,
    pub throughput_degradation_pct: f64,
}

impl Default for RegressionThresholds {
    fn default() -> Self {
        RegressionThresholds {
            ttft_degradation_pct: 5.0,
            tpot_degradation_pct: 10.0,
            memory_growth_pct: 20.0,
            throughput_degradation_pct: 10.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RegressionStatus {
    Pass,
    Warn,
    Fail,
}

impl std::fmt::Display for RegressionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegressionStatus::Pass => write!(f, "✅ PASS"),
            RegressionStatus::Warn => write!(f, "⚠️  WARN"),
            RegressionStatus::Fail => write!(f, "❌ FAIL"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComparison {
    pub metric_name: String,
    pub baseline_value: f64,
    pub current_value: f64,
    pub change_pct: f64,
    pub threshold: f64,
    pub status: RegressionStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioComparison {
    pub scenario_name: String,
    pub comparisons: Vec<MetricComparison>,
    pub overall_status: RegressionStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionReport {
    pub baseline_commit: String,
    pub current_commit: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub thresholds: RegressionThresholds,
    pub scenario_comparisons: Vec<ScenarioComparison>,
    pub overall_status: RegressionStatus,
    pub summary: String,
}

impl RegressionReport {
    pub fn new(baseline_commit: impl Into<String>, current_commit: impl Into<String>) -> Self {
        RegressionReport {
            baseline_commit: baseline_commit.into(),
            current_commit: current_commit.into(),
            timestamp: chrono::Utc::now(),
            thresholds: RegressionThresholds::default(),
            scenario_comparisons: Vec::new(),
            overall_status: RegressionStatus::Pass,
            summary: String::new(),
        }
    }

    pub fn with_thresholds(mut self, thresholds: RegressionThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    pub fn finalize(&mut self) {
        let pass_count = self
            .scenario_comparisons
            .iter()
            .filter(|sc| sc.overall_status == RegressionStatus::Pass)
            .count();
        let warn_count = self
            .scenario_comparisons
            .iter()
            .filter(|sc| sc.overall_status == RegressionStatus::Warn)
            .count();
        let fail_count = self
            .scenario_comparisons
            .iter()
            .filter(|sc| sc.overall_status == RegressionStatus::Fail)
            .count();

        if fail_count > 0 {
            self.overall_status = RegressionStatus::Fail;
        } else if warn_count > 0 {
            self.overall_status = RegressionStatus::Warn;
        } else {
            self.overall_status = RegressionStatus::Pass;
        }

        self.summary = format!(
            "Regression Test Result: {} | Pass: {} | Warn: {} | Fail: {}",
            self.overall_status, pass_count, warn_count, fail_count
        );
    }
}

pub struct RegressionChecker {
    thresholds: RegressionThresholds,
}

impl RegressionChecker {
    pub fn new() -> Self {
        RegressionChecker {
            thresholds: RegressionThresholds::default(),
        }
    }

    pub fn with_thresholds(thresholds: RegressionThresholds) -> Self {
        RegressionChecker { thresholds }
    }

    pub fn load_baseline(
        &self,
        path: &PathBuf,
    ) -> anyhow::Result<crate::benchmark::runner::BenchmarkResults> {
        let content = fs::read_to_string(path)?;
        let results: crate::benchmark::runner::BenchmarkResults = serde_json::from_str(&content)?;
        Ok(results)
    }

    pub fn save_baseline(
        &self,
        results: &crate::benchmark::runner::BenchmarkResults,
        path: &PathBuf,
    ) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(results)?;
        fs::write(path, content)?;
        println!("Baseline saved to: {}", path.display());
        Ok(())
    }

    pub fn compare(
        &self,
        baseline: &crate::benchmark::runner::BenchmarkResults,
        current: &crate::benchmark::runner::BenchmarkResults,
    ) -> anyhow::Result<RegressionReport> {
        let mut report = RegressionReport::new(
            baseline.config.commit_hash.clone(),
            current.config.commit_hash.clone(),
        );
        report.thresholds = self.thresholds.clone();

        for current_result in &current.results {
            let scenario_name = current_result.scenario.name().to_string();

            let baseline_result = baseline.results.iter().find(|r| {
                r.scenario.name() == current_result.scenario.name()
                    && r.batch_size == current_result.batch_size
                    && r.sequence_length == current_result.sequence_length
            });

            let mut comparisons = Vec::new();

            if let Some(baseline_r) = baseline_result {
                comparisons.push(self.compare_metric(
                    "TTFT (ms)",
                    baseline_r.metrics.ttft_ms.mean,
                    current_result.metrics.ttft_ms.mean,
                    self.thresholds.ttft_degradation_pct,
                    true,
                ));

                comparisons.push(self.compare_metric(
                    "TPOT (ms/token)",
                    baseline_r.metrics.tpot_ms_per_token.mean,
                    current_result.metrics.tpot_ms_per_token.mean,
                    self.thresholds.tpot_degradation_pct,
                    true,
                ));

                comparisons.push(self.compare_metric(
                    "Throughput (tokens/s)",
                    baseline_r.metrics.throughput_tokens_s,
                    current_result.metrics.throughput_tokens_s,
                    self.thresholds.throughput_degradation_pct,
                    true,
                ));

                comparisons.push(self.compare_metric(
                    "Memory Peak (MB)",
                    baseline_r.metrics.memory.peak_mb,
                    current_result.metrics.memory.peak_mb,
                    self.thresholds.memory_growth_pct,
                    false,
                ));
            }

            let overall_status = if comparisons
                .iter()
                .any(|c| c.status == RegressionStatus::Fail)
            {
                RegressionStatus::Fail
            } else if comparisons
                .iter()
                .any(|c| c.status == RegressionStatus::Warn)
            {
                RegressionStatus::Warn
            } else {
                RegressionStatus::Pass
            };

            report.scenario_comparisons.push(ScenarioComparison {
                scenario_name,
                comparisons,
                overall_status,
            });
        }

        report.finalize();
        Ok(report)
    }

    fn compare_metric(
        &self,
        name: &str,
        baseline_value: f64,
        current_value: f64,
        threshold: f64,
        lower_is_better: bool,
    ) -> MetricComparison {
        let change_pct = if baseline_value > 0.0 {
            ((current_value - baseline_value) / baseline_value.abs()) * 100.0
        } else {
            0.0
        };

        let status = if lower_is_better {
            if change_pct > threshold {
                RegressionStatus::Fail
            } else if change_pct > threshold * 0.5 {
                RegressionStatus::Warn
            } else {
                RegressionStatus::Pass
            }
        } else if change_pct > threshold {
            RegressionStatus::Fail
        } else if change_pct > threshold * 0.5 {
            RegressionStatus::Warn
        } else {
            RegressionStatus::Pass
        };

        MetricComparison {
            metric_name: name.to_string(),
            baseline_value,
            current_value,
            change_pct,
            threshold,
            status,
        }
    }

    pub fn generate_markdown_report(&self, report: &RegressionReport) -> String {
        let mut md = String::new();

        md.push_str("# 📊 Performance Regression Report\n\n");
        md.push_str(&format!(
            "**Overall Status:** {}\n\n",
            report.overall_status
        ));
        md.push_str(&format!(
            "- **Baseline Commit:** `{}`\n",
            report.baseline_commit
        ));
        md.push_str(&format!(
            "- **Current Commit:** `{}`\n",
            report.current_commit
        ));
        md.push_str(&format!(
            "- **Generated:** {}\n\n",
            report.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        ));

        md.push_str("## Summary\n\n");
        md.push_str(&format!("{}\n\n", report.summary));

        md.push_str("## Thresholds\n\n");
        md.push_str("| Metric | Threshold |\n");
        md.push_str("|--------|----------|\n");
        md.push_str(&format!(
            "| TTFT Degradation | {:.1}% |\n",
            report.thresholds.ttft_degradation_pct
        ));
        md.push_str(&format!(
            "| TPOT Degradation | {:.1}% |\n",
            report.thresholds.tpot_degradation_pct
        ));
        md.push_str(&format!(
            "| Memory Growth | {:.1}% |\n",
            report.thresholds.memory_growth_pct
        ));
        md.push_str(&format!(
            "| Throughput Degradation | {:.1}% |\n\n",
            report.thresholds.throughput_degradation_pct
        ));

        md.push_str("## Scenario Results\n\n");

        for sc in &report.scenario_comparisons {
            md.push_str(&format!(
                "### {} ({})\n\n",
                sc.scenario_name, sc.overall_status
            ));

            if !sc.comparisons.is_empty() {
                md.push_str("| Metric | Baseline | Current | Change | Status |\n");
                md.push_str("|--------|----------|---------|--------|--------|\n");

                for comp in &sc.comparisons {
                    let change_str = if comp.change_pct >= 0.0 {
                        format!("+{:.2}%", comp.change_pct)
                    } else {
                        format!("{:.2}%", comp.change_pct)
                    };

                    md.push_str(&format!(
                        "| {} | {:.2} | {:.2} | {} | {} |\n",
                        comp.metric_name,
                        comp.baseline_value,
                        comp.current_value,
                        change_str,
                        comp.status
                    ));
                }
            }

            md.push('\n');
        }

        md.push_str("---\n");
        md.push_str("*Report generated by OpenMini Benchmark Framework*\n");

        md
    }

    pub fn export_report(
        &self,
        report: &RegressionReport,
        format: &str,
        path: Option<&PathBuf>,
    ) -> anyhow::Result<String> {
        let output = match format {
            "markdown" => self.generate_markdown_report(report),
            "json" => serde_json::to_string_pretty(report)?,
            _ => return Err(anyhow::anyhow!("Unsupported format: {}", format)),
        };

        match path {
            Some(p) => {
                fs::write(p, &output)?;
                println!("Regression report exported to: {}", p.display());
            }
            None => {
                print!("{}", output);
            }
        }

        Ok(output)
    }
}

impl Default for RegressionChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::config::BenchmarkConfig;
    use crate::benchmark::metrics::BenchmarkMetrics;
    use crate::benchmark::runner::{BenchmarkResults, ScenarioResult};
    use crate::benchmark::scenarios::Scenario;

    fn create_baseline_results() -> BenchmarkResults {
        let mut config = BenchmarkConfig::new("test.gguf", "baseline-model");
        config.commit_hash = "abc123".to_string();

        let mut results = BenchmarkResults::new(config);

        let mut metrics = BenchmarkMetrics::default();
        metrics.ttft_ms.mean = 50.0;
        metrics.tpot_ms_per_token.mean = 10.0;
        metrics.throughput_tokens_s = 100.0;
        metrics.memory.peak_mb = 2000.0;

        results.add_result(ScenarioResult {
            scenario: Scenario::MediumContext,
            batch_size: 1,
            sequence_length: 2048,
            run_index: 0,
            metrics,
            duration_ms: 5000.0,
            success: true,
            error_message: None,
        });

        results
    }

    fn create_current_results_improved() -> BenchmarkResults {
        let mut config = BenchmarkConfig::new("test.gguf", "current-model");
        config.commit_hash = "def456".to_string();

        let mut results = BenchmarkResults::new(config);

        let mut metrics = BenchmarkMetrics::default();
        metrics.ttft_ms.mean = 45.0; // 改善10%（从50ms降到45ms）
        metrics.tpot_ms_per_token.mean = 9.0; // 改善10%（从10ms降到9ms）
        metrics.throughput_tokens_s = 104.0; // 改善4%（从100增加到104，<5% warn阈值避免误报）
        metrics.memory.peak_mb = 2050.0; // 增长2.5%（从2000到2050，<10% warn阈值）

        results.add_result(ScenarioResult {
            scenario: Scenario::MediumContext,
            batch_size: 1,
            sequence_length: 2048,
            run_index: 0,
            metrics,
            duration_ms: 4500.0,
            success: true,
            error_message: None,
        });

        results
    }

    fn create_current_results_regressed() -> BenchmarkResults {
        let mut config = BenchmarkConfig::new("test.gguf", "current-model");
        config.commit_hash = "def456".to_string();

        let mut results = BenchmarkResults::new(config);

        let mut metrics = BenchmarkMetrics::default();
        metrics.ttft_ms.mean = 60.0;
        metrics.tpot_ms_per_token.mean = 12.0;
        metrics.throughput_tokens_s = 80.0;
        metrics.memory.peak_mb = 2500.0;

        results.add_result(ScenarioResult {
            scenario: Scenario::MediumContext,
            batch_size: 1,
            sequence_length: 2048,
            run_index: 0,
            metrics,
            duration_ms: 6000.0,
            success: true,
            error_message: None,
        });

        results
    }

    #[test]
    fn test_regression_check_pass() {
        let checker = RegressionChecker::new();
        let baseline = create_baseline_results();
        let current = create_current_results_improved();

        let report = checker.compare(&baseline, &current).unwrap();
        assert_eq!(report.overall_status, RegressionStatus::Pass);
    }

    #[test]
    fn test_regression_check_fail() {
        let checker = RegressionChecker::new();
        let baseline = create_baseline_results();
        let current = create_current_results_regressed();

        let report = checker.compare(&baseline, &current).unwrap();
        assert_eq!(report.overall_status, RegressionStatus::Fail);
    }

    #[test]
    fn test_custom_thresholds() {
        let thresholds = RegressionThresholds {
            ttft_degradation_pct: 1.0,
            ..Default::default()
        };
        let checker = RegressionChecker::with_thresholds(thresholds);
        let baseline = create_baseline_results();
        let current = create_current_results_improved();

        let report = checker.compare(&baseline, &current).unwrap();
        assert!(matches!(
            report.overall_status,
            RegressionStatus::Pass | RegressionStatus::Warn
        ));
    }

    #[test]
    fn test_save_and_load_baseline() {
        let checker = RegressionChecker::new();
        let baseline = create_baseline_results();

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("baseline.json");

        checker.save_baseline(&baseline, &path).unwrap();
        let loaded = checker.load_baseline(&path).unwrap();

        assert_eq!(loaded.benchmark_id, baseline.benchmark_id);
        assert_eq!(loaded.results.len(), baseline.results.len());
    }

    #[test]
    fn test_markdown_report_generation() {
        let checker = RegressionChecker::new();
        let baseline = create_baseline_results();
        let current = create_current_results_improved();

        let report = checker.compare(&baseline, &current).unwrap();
        let markdown = checker.generate_markdown_report(&report);

        assert!(markdown.contains("# 📊 Performance Regression Report"));
        assert!(markdown.contains("Baseline Commit"));
        assert!(markdown.contains("MediumContext"));
        assert!(markdown.contains("PASS"));
    }

    #[test]
    fn test_regression_status_display() {
        assert_eq!(format!("{}", RegressionStatus::Pass), "✅ PASS");
        assert_eq!(format!("{}", RegressionStatus::Warn), "⚠️  WARN");
        assert_eq!(format!("{}", RegressionStatus::Fail), "❌ FAIL");
    }

    #[test]
    fn test_serialization() {
        let report = RegressionReport::new("abc123", "def456");
        let json = serde_json::to_string(&report).unwrap();
        let deserialized: RegressionReport = serde_json::from_str(&json).unwrap();
        assert_eq!(report.baseline_commit, deserialized.baseline_commit);
    }
}
