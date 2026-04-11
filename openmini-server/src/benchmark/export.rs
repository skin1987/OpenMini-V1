//! Results Export Module
//!
//! 支持多种格式的基准测试结果导出：
//! - JSON: 详细报告格式
//! - CSV: 便于 Excel 分析
//! - Prometheus: 监控系统集成
//! - Human: 可读性强的文本格式

use std::fs;
use std::path::PathBuf;

use crate::benchmark::config::OutputFormat;
use crate::benchmark::runner::BenchmarkResults;

pub trait Exporter: Send + Sync {
    fn export(&self, results: &BenchmarkResults, path: &Option<PathBuf>) -> anyhow::Result<()>;
    fn format_name(&self) -> &'static str;
}

pub struct JsonExporter;

impl Exporter for JsonExporter {
    fn export(&self, results: &BenchmarkResults, path: &Option<PathBuf>) -> anyhow::Result<()> {
        let output = serde_json::to_string_pretty(results)?;

        match path {
            Some(p) => {
                fs::write(p, output)?;
                println!("Results exported to JSON: {}", p.display());
            }
            None => {
                println!("{}", output);
            }
        }

        Ok(())
    }

    fn format_name(&self) -> &'static str {
        "JSON"
    }
}

pub struct CsvExporter;

impl Exporter for CsvExporter {
    fn export(&self, results: &BenchmarkResults, path: &Option<PathBuf>) -> anyhow::Result<()> {
        let mut output = String::new();

        output.push_str(
            "scenario,batch_size,seq_len,ttft_mean,ttft_p50,ttft_p95,ttft_p99,tpot_mean,tpot_std,throughput,memory_peak,memory_kv_cache,success,duration_ms\n",
        );

        for result in &results.results {
            output.push_str(&format!(
                "{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{},{}\n",
                result.scenario.name(),
                result.batch_size,
                result.sequence_length,
                result.metrics.ttft_ms.mean,
                result.metrics.ttft_ms.p50,
                result.metrics.ttft_ms.p95,
                result.metrics.ttft_ms.p99,
                result.metrics.tpot_ms_per_token.mean,
                result.metrics.tpot_ms_per_token.std,
                result.metrics.throughput_tokens_s,
                result.metrics.memory.peak_mb,
                result.metrics.memory.kv_cache_mb,
                result.success,
                result.duration_ms,
            ));
        }

        match path {
            Some(p) => {
                fs::write(p, output)?;
                println!("Results exported to CSV: {}", p.display());
            }
            None => {
                print!("{}", output);
            }
        }

        Ok(())
    }

    fn format_name(&self) -> &'static str {
        "CSV"
    }
}

pub struct PrometheusExporter;

impl Exporter for PrometheusExporter {
    fn export(&self, results: &BenchmarkResults, path: &Option<PathBuf>) -> anyhow::Result<()> {
        let mut output = String::new();

        output.push_str("# OpenMini Benchmark Results\n");
        output.push_str("# Generated at: ");
        output.push_str(&results.timestamp.to_rfc3339());
        output.push_str("\n\n");

        output.push_str("# HELP openmini_benchmark_ttft_ms Time to first token in milliseconds\n");
        output.push_str("# TYPE openmini_benchmark_ttft_ms summary\n");

        for result in &results.results {
            let scenario = result.scenario.name();
            output.push_str(&format!(
                "openmini_benchmark_ttft_ms{{scenario=\"{}\",quantile=\"0.5\"}} {:.2}\n",
                scenario, result.metrics.ttft_ms.p50
            ));
            output.push_str(&format!(
                "openmini_benchmark_ttft_ms{{scenario=\"{}\",quantile=\"0.95\"}} {:.2}\n",
                scenario, result.metrics.ttft_ms.p95
            ));
            output.push_str(&format!(
                "openmini_benchmark_ttft_ms{{scenario=\"{}\",quantile=\"0.99\"}} {:.2}\n",
                scenario, result.metrics.ttft_ms.p99
            ));
            output.push_str(&format!(
                "openmini_benchmark_ttft_ms_sum{{scenario=\"{}}} {:.2}\n",
                scenario, result.metrics.ttft_ms.mean
            ));
            output.push_str(&format!(
                "openmini_benchmark_ttft_ms_count{{scenario=\"{}}} {}\n",
                scenario, results.config.num_benchmark_runs
            ));
        }

        output.push_str("\n# HELP openmini_benchmark_tpot_ms Time per output token in milliseconds\n");
        output.push_str("# TYPE openmini_benchmark_tpot_ms gauge\n");

        for result in &results.results {
            output.push_str(&format!(
                "openmini_benchmark_tpot_ms{{scenario=\"{}\"}} {:.2}\n",
                result.scenario.name(),
                result.metrics.tpot_ms_per_token.mean
            ));
        }

        output.push_str("\n# HELP openmini_benchmark_throughput_tokens_s Tokens per second throughput\n");
        output.push_str("# TYPE openmini_benchmark_throughput_tokens_s gauge\n");

        for result in &results.results {
            output.push_str(&format!(
                "openmini_benchmark_throughput_tokens_s{{scenario=\"{}\"}} {:.2}\n",
                result.scenario.name(),
                result.metrics.throughput_tokens_s
            ));
        }

        output.push_str("\n# HELP openmini_benchmark_memory_mb Memory usage in MB\n");
        output.push_str("# TYPE openmini_benchmark_memory_mb gauge\n");

        for result in &results.results {
            output.push_str(&format!(
                "openmini_benchmark_memory_mb{{scenario=\"{}\",type=\"peak\"}} {:.2}\n",
                result.scenario.name(),
                result.metrics.memory.peak_mb
            ));
            output.push_str(&format!(
                "openmini_benchmark_memory_mb{{scenario=\"{}\",type=\"kv_cache\"}} {:.2}\n",
                result.scenario.name(),
                result.metrics.memory.kv_cache_mb
            ));
        }

        match path {
            Some(p) => {
                fs::write(p, output)?;
                println!("Results exported to Prometheus format: {}", p.display());
            }
            None => {
                print!("{}", output);
            }
        }

        Ok(())
    }

    fn format_name(&self) -> &'static str {
        "Prometheus"
    }
}

pub struct HumanExporter;

impl Exporter for HumanExporter {
    fn export(&self, results: &BenchmarkResults, path: &Option<PathBuf>) -> anyhow::Result<()> {
        let mut output = String::new();

        output.push_str("╔══════════════════════════════════════════════════════════╗\n");
        output.push_str("║              OpenMini Benchmark Results                   ║\n");
        output.push_str("╠══════════════════════════════════════════════════════════╣\n");
        output.push_str(&format!("║ Benchmark ID: {:^44} ║\n", results.benchmark_id));
        output.push_str(&format!("║ Timestamp:    {:^44} ║\n", results.timestamp.format("%Y-%m-%d %H:%M:%S UTC")));
        output.push_str(&format!("║ Model:        {:^44} ║\n", results.config.model_name));
        output.push_str(&format!("║ Quantization: {:^44} ║\n", results.config.quantization));
        output.push_str(&format!("║ Device:       {:^44} ║\n", results.config.device));
        output.push_str(&format!("║ Duration:     {:^43.2}s ║\n", results.total_duration_ms / 1000.0));
        output.push_str("╚══════════════════════════════════════════════════════════╝\n\n");

        output.push_str("┌──────────────────┬────────┬─────────┬──────────┬──────────┬─────────────┬──────────┐\n");
        output.push_str("│ Scenario         │ Batch  │ Seq Len │ TTFT(ms) │ TPOT(ms) │ Throughput  │ Memory   │\n");
        output.push_str("├──────────────────┼────────┼─────────┼──────────┼──────────┼─────────────┼──────────┤\n");

        for result in &results.results {
            output.push_str(&format!(
                "│ {:16} │ {:6} │ {:7} │ {:8.2} │ {:8.2} │ {:11.2} │ {:8.2} │\n",
                result.scenario.name(),
                result.batch_size,
                result.sequence_length,
                result.metrics.ttft_ms.mean,
                result.metrics.tpot_ms_per_token.mean,
                result.metrics.throughput_tokens_s,
                result.metrics.memory.peak_mb
            ));
        }

        output.push_str("└──────────────────┴────────┴─────────┴──────────┴──────────┴─────────────┴──────────┘\n");

        let success_count = results.results.iter().filter(|r| r.success).count();
        output.push_str(&format!("\n✓ Successful: {}/{}\n", success_count, results.results.len()));

        match path {
            Some(p) => {
                fs::write(p, output)?;
                println!("Results exported to human-readable format: {}", p.display());
            }
            None => {
                print!("{}", output);
            }
        }

        Ok(())
    }

    fn format_name(&self) -> &'static str {
        "Human"
    }
}

pub fn get_exporter(format: &OutputFormat) -> Box<dyn Exporter> {
    match format {
        OutputFormat::Json => Box::new(JsonExporter),
        OutputFormat::Csv => Box::new(CsvExporter),
        OutputFormat::Prometheus => Box::new(PrometheusExporter),
        OutputFormat::Human => Box::new(HumanExporter),
    }
}

pub fn export_results(
    results: &BenchmarkResults,
    format: &OutputFormat,
    path: Option<PathBuf>,
) -> anyhow::Result<()> {
    let exporter = get_exporter(format);
    exporter.export(results, &path)
}

pub fn export_all_formats(
    results: &BenchmarkResults,
    base_path: impl AsRef<std::path::Path>,
) -> anyhow::Result<Vec<PathBuf>> {
    let base = base_path.as_ref();
    let mut exported_paths = Vec::new();

    let formats = [
        (OutputFormat::Json, "results.json"),
        (OutputFormat::Csv, "results.csv"),
        (OutputFormat::Prometheus, "results.prom"),
        (OutputFormat::Human, "results.txt"),
    ];

    for (format, filename) in &formats {
        let path = base.join(filename);
        export_results(results, format, Some(path.clone()))?;
        exported_paths.push(path);
    }

    Ok(exported_paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::config::BenchmarkConfig;
    use crate::benchmark::runner::ScenarioResult;
    use crate::benchmark::scenarios::Scenario;

    fn create_sample_results() -> BenchmarkResults {
        let config = BenchmarkConfig::new("test.gguf", "test-model");
        let mut results = BenchmarkResults::new(config);

        let mut metrics = crate::benchmark::metrics::BenchmarkMetrics::default();
        metrics.ttft_ms.mean = 45.2;
        metrics.ttft_ms.p50 = 43.1;
        metrics.ttft_ms.p95 = 52.8;
        metrics.ttft_ms.p99 = 67.3;
        metrics.tpot_ms_per_token.mean = 12.3;
        metrics.tpot_ms_per_token.std = 2.1;
        metrics.throughput_tokens_s = 81.3;
        metrics.memory.peak_mb = 2048.0;
        metrics.memory.kv_cache_mb = 512.0;

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

    #[test]
    fn test_json_export() {
        let results = create_sample_results();
        let exporter = JsonExporter;
        let result = exporter.export(&results, &None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_csv_export() {
        let results = create_sample_results();
        let exporter = CsvExporter;
        let result = exporter.export(&results, &None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_prometheus_export() {
        let results = create_sample_results();
        let exporter = PrometheusExporter;
        let result = exporter.export(&results, &None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_human_export() {
        let results = create_sample_results();
        let exporter = HumanExporter;
        let result = exporter.export(&results, &None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_file_export() {
        let results = create_sample_results();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.json");

        let exporter = JsonExporter;
        let result = exporter.export(&results, &Some(path.clone()));
        assert!(result.is_ok());
        assert!(path.exists());
    }

    #[test]
    fn test_get_exporter() {
        let json_exporter = get_exporter(&OutputFormat::Json);
        assert_eq!(json_exporter.format_name(), "JSON");

        let csv_exporter = get_exporter(&OutputFormat::Csv);
        assert_eq!(csv_exporter.format_name(), "CSV");

        let prom_exporter = get_exporter(&OutputFormat::Prometheus);
        assert_eq!(prom_exporter.format_name(), "Prometheus");

        let human_exporter = get_exporter(&OutputFormat::Human);
        assert_eq!(human_exporter.format_name(), "Human");
    }

    #[test]
    fn test_export_all_formats() {
        let results = create_sample_results();
        let temp_dir = tempfile::tempdir().unwrap();

        let paths = export_all_formats(&results, temp_dir.path()).unwrap();
        assert_eq!(paths.len(), 4);

        for path in &paths {
            assert!(path.exists());
        }
    }
}
