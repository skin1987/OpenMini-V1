//! Core Metrics Collection Module
//!
//! 提供完整的性能指标采集功能，包括：
//! - TTFT (Time To First Token): 首token延迟
//! - TPOT (Time Per Output Token): 每输出token时间
//! - TBTL (Total Benchmarked Tokens per Latency): 总吞吐/延迟比
//! - Throughput: tokens/s 吞吐量
//! - Memory Usage: 内存占用
//! - GPU/CPU Utilization: 利用率

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

impl Default for LatencyMetrics {
    fn default() -> Self {
        LatencyMetrics {
            mean: 0.0,
            std: 0.0,
            min: f64::MAX,
            max: 0.0,
            p50: 0.0,
            p95: 0.0,
            p99: 0.0,
        }
    }
}

impl LatencyMetrics {
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return LatencyMetrics::default();
        }

        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p50 = percentile(&sorted, 50.0);
        let p95 = percentile(&sorted, 95.0);
        let p99 = percentile(&sorted, 99.0);

        LatencyMetrics {
            mean,
            std,
            min: sorted[0],
            max: sorted[sorted.len() - 1],
            p50,
            p95,
            p99,
        }
    }
}

fn percentile(sorted_data: &[f64], percent: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }
    let k = (percent / 100.0) * (sorted_data.len() as f64 - 1.0);
    let f = k.floor() as usize;
    let c = k.ceil() as usize;
    if f == c {
        sorted_data[f]
    } else {
        sorted_data[f] + (k - f as f64) * (sorted_data[c] - sorted_data[f])
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    pub peak_mb: f64,
    pub kv_cache_mb: f64,
    pub model_weights_mb: f64,
    pub allocated_mb: f64,
}

impl Default for MemoryMetrics {
    fn default() -> Self {
        MemoryMetrics {
            peak_mb: 0.0,
            kv_cache_mb: 0.0,
            model_weights_mb: 0.0,
            allocated_mb: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct UtilizationMetrics {
    pub cpu_percent: Option<f64>,
    pub gpu_percent: Option<f64>,
    pub gpu_memory_percent: Option<f64>,
    pub power_watts: Option<f64>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    pub ttft_ms: LatencyMetrics,
    pub tpot_ms_per_token: LatencyMetrics,
    pub tbtl: f64,
    pub throughput_tokens_s: f64,
    pub total_tokens_generated: usize,
    pub total_prompt_tokens: usize,
    pub memory: MemoryMetrics,
    pub utilization: UtilizationMetrics,
    pub inference_duration_ms: f64,
}

impl Default for BenchmarkMetrics {
    fn default() -> Self {
        BenchmarkMetrics {
            ttft_ms: LatencyMetrics::default(),
            tpot_ms_per_token: LatencyMetrics::default(),
            tbtl: 0.0,
            throughput_tokens_s: 0.0,
            total_tokens_generated: 0,
            total_prompt_tokens: 0,
            memory: MemoryMetrics::default(),
            utilization: UtilizationMetrics::default(),
            inference_duration_ms: 0.0,
        }
    }
}

impl BenchmarkMetrics {
    pub fn new() -> Self {
        BenchmarkMetrics::default()
    }

    pub fn calculate_tbtl(&mut self) {
        if self.inference_duration_ms > 0.0 && self.total_tokens_generated > 0 {
            self.tbtl = (self.total_tokens_generated as f64)
                / (self.inference_duration_ms / 1000.0);
        }
    }

    pub fn calculate_throughput(&mut self) {
        if self.inference_duration_ms > 0.0 {
            self.throughput_tokens_s =
                (self.total_tokens_generated as f64) / (self.inference_duration_ms / 1000.0);
        }
    }

    pub fn finalize(&mut self) {
        self.calculate_tbtl();
        self.calculate_throughput();
    }
}

pub struct MetricsCollector {
    ttft_samples: Vec<f64>,
    tpot_samples: Vec<f64>,
    start_time: std::time::Instant,
    first_token_time: Option<std::time::Instant>,
    token_count: usize,
    prompt_tokens: usize,
}

impl MetricsCollector {
    pub fn new(prompt_tokens: usize) -> Self {
        MetricsCollector {
            ttft_samples: Vec::new(),
            tpot_samples: Vec::new(),
            start_time: std::time::Instant::now(),
            first_token_time: None,
            token_count: 0,
            prompt_tokens,
        }
    }

    pub fn record_first_token(&mut self) {
        self.first_token_time = Some(std::time::Instant::now());
        let ttft_ms = self.first_token_time.unwrap().duration_since(self.start_time).as_millis() as f64;
        self.ttft_samples.push(ttft_ms);
    }

    pub fn record_token(&mut self) {
        self.token_count += 1;
        if let Some(first_time) = self.first_token_time {
            let elapsed = std::time::Instant::now().duration_since(first_time).as_millis() as f64;
            if self.token_count > 1 {
                let per_token = elapsed / (self.token_count - 1) as f64;
                self.tpot_samples.push(per_token);
            }
        }
    }

    pub fn collect(self) -> BenchmarkMetrics {
        let duration_ms = self.start_time.elapsed().as_millis() as f64;

        let mut metrics = BenchmarkMetrics {
            ttft_ms: LatencyMetrics::from_samples(&self.ttft_samples),
            tpot_ms_per_token: LatencyMetrics::from_samples(&self.tpot_samples),
            total_tokens_generated: self.token_count,
            total_prompt_tokens: self.prompt_tokens,
            inference_duration_ms: duration_ms,
            ..Default::default()
        };

        metrics.finalize();
        metrics
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        MetricsCollector::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_metrics_from_samples() {
        let samples = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let metrics = LatencyMetrics::from_samples(&samples);

        assert!((metrics.mean - 30.0).abs() < 0.001);
        assert_eq!(metrics.min, 10.0);
        assert_eq!(metrics.max, 50.0);
        assert!(metrics.std > 0.0);
    }

    #[test]
    fn test_latency_metrics_empty() {
        let metrics = LatencyMetrics::from_samples(&[]);
        assert_eq!(metrics.mean, 0.0);
        assert_eq!(metrics.min, f64::MAX);
    }

    #[test]
    fn test_percentile_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&data, 50.0), 3.0);
        assert!((percentile(&data, 95.0) - 4.8).abs() < 0.01);
    }

    #[test]
    fn test_benchmark_metrics_finalization() {
        let mut metrics = BenchmarkMetrics {
            total_tokens_generated: 100,
            inference_duration_ms: 1000.0,
            ..Default::default()
        };
        metrics.finalize();

        assert!((metrics.throughput_tokens_s - 100.0).abs() < 0.001);
        assert!((metrics.tbtl - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_metrics_collector_basic() {
        let mut collector = MetricsCollector::new(512);

        // 添加延迟模拟推理处理时间（在记录首token前）
        // 这样TTFT = record_first_token时间 - start_time 才会有正值
        std::thread::sleep(std::time::Duration::from_millis(10));

        // 记录首token时间（此时会立即计算TTFT）
        collector.record_first_token();

        // 添加延迟模拟token生成间隔
        std::thread::sleep(std::time::Duration::from_millis(10));

        for _ in 0..10 {
            collector.record_token();
            // 每个token间添加延迟确保TPOT计算有正值
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        let metrics = collector.collect();

        assert_eq!(metrics.total_tokens_generated, 10);
        assert_eq!(metrics.total_prompt_tokens, 512);
        // TTFT和TPOT应该有正值（因为我们添加了足够的延迟）
        assert!(metrics.ttft_ms.mean > 0.0, "TTFT should be positive after recording first token with delay");
        assert!(metrics.tpot_ms_per_token.mean > 0.0, "TPOT should be positive after recording tokens with delay");
    }

    #[test]
    fn test_memory_metrics_default() {
        let memory = MemoryMetrics::default();
        assert_eq!(memory.peak_mb, 0.0);
        assert_eq!(memory.kv_cache_mb, 0.0);
    }

    #[test]
    fn test_utilization_metrics_default() {
        let util = UtilizationMetrics::default();
        assert!(util.cpu_percent.is_none());
        assert!(util.gpu_percent.is_none());
    }

    #[test]
    fn test_serialization() {
        let metrics = BenchmarkMetrics::default();
        let json = serde_json::to_string(&metrics).unwrap();
        let deserialized: BenchmarkMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(metrics.total_tokens_generated, deserialized.total_tokens_generated);
    }
}
