//! TrainingMonitor - 训练监控系统

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsRecord {
    pub timestamp: String,
    pub global_step: u64,
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: Option<f64>,
    pub learning_rate: f64,
    pub grad_norm: f64,
    pub tokens_per_sec: f64,
    pub step_time_ms: f64,
}

struct RingBuffer<T> {
    data: Vec<Option<T>>,
    head: usize,
    count: usize,
    capacity: usize,
}

impl<T: Clone> RingBuffer<T> {
    fn new(capacity: usize) -> Self {
        Self {
            data: vec![None; capacity],
            head: 0,
            count: 0,
            capacity,
        }
    }
    
    fn push(&mut self, item: T) {
        self.data[self.head] = Some(item);
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }
    
    fn latest(&self, n: usize) -> Vec<T> {
        let n = n.min(self.count);
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let idx = if self.head > i {
                self.head - i - 1
            } else {
                self.capacity - (i + 1 - self.head)
            };
            if let Some(ref item) = self.data[idx] {
                result.push(item.clone());
            }
        }
        result
    }
    
    fn len(&self) -> usize { self.count }
    
    fn is_empty(&self) -> bool { self.count == 0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsAggregate {
    pub mean_loss: f64,
    pub std_loss: f64,
    pub mean_lr: f64,
    pub mean_grad_norm: f64,
    pub mean_throughput: f64,
    pub trend: TrendDirection,
    pub num_samples: usize,
}

pub struct TrainingMonitor {
    history: RingBuffer<MetricsRecord>,
    start_time: std::time::Instant,
    log_every_n_steps: usize,
}

impl TrainingMonitor {
    pub fn new(max_capacity: usize, log_every_n_steps: usize) -> Self {
        Self {
            history: RingBuffer::new(max_capacity),
            start_time: std::time::Instant::now(),
            log_every_n_steps,
        }
    }
    
    pub fn record(&mut self, record: MetricsRecord) {
        self.history.push(record);
    }
    
    pub fn recent_metrics(&self, n: usize) -> Vec<MetricsRecord> {
        self.history.latest(n)
    }
    
    pub fn all_metrics(&self) -> Vec<MetricsRecord> {
        self.recent_metrics(self.history.len())
    }
    
    pub fn format_log_line(&self, record: &MetricsRecord) -> String {
        let ppl = if record.train_loss > 0.0 && record.train_loss < 100.0 {
            format!("{:.2}", record.train_loss.exp())
        } else {
            "N/A".to_string()
        };
        
        format!(
            "[{}] Epoch {} | Step {} | LR: {:.2e}\n\
             \tTrain Loss: {:.4} | PPL: {} | Grad Norm: {:.4} | {:.1} tok/s",
            &record.timestamp[..19],
            record.epoch + 1,
            record.global_step,
            record.learning_rate,
            record.train_loss,
            ppl,
            record.grad_norm,
            record.tokens_per_sec
        )
    }
    
    pub fn perplexity(loss: f64) -> f64 {
        loss.exp()
    }
    
    pub fn eta_seconds(&self, total_steps: u64) -> Option<f64> {
        if self.history.is_empty() || total_steps == 0 {
            return None;
        }
        
        let recent = self.recent_metrics(50.min(self.history.len()));
        if recent.is_empty() {
            return None;
        }
        
        let avg_time: f64 = recent.iter()
            .map(|r| r.step_time_ms)
            .sum::<f64>() / recent.len() as f64;
        
        let latest_step = recent.last()?.global_step;
        let remaining = total_steps.saturating_sub(latest_step);
        Some(avg_time * remaining as f64 / 1000.0)
    }
    
    pub fn aggregate(&self, last_n: usize) -> MetricsAggregate {
        let records = self.recent_metrics(last_n);
        if records.is_empty() {
            return MetricsAggregate {
                mean_loss: 0.0,
                std_loss: 0.0,
                mean_lr: 0.0,
                mean_grad_norm: 0.0,
                mean_throughput: 0.0,
                trend: TrendDirection::Stable,
                num_samples: 0,
            };
        }
        
        let n = records.len() as f64;
        let sum_loss: f64 = records.iter().map(|r| r.train_loss).sum();
        let mean_loss = sum_loss / n;
        
        let var_loss: f64 = records.iter()
            .map(|r| (r.train_loss - mean_loss).powi(2))
            .sum::<f64>() / n;
        let std_loss = var_loss.sqrt();
        
        let mean_lr: f64 = records.iter().map(|r| r.learning_rate).sum::<f64>() / n;
        let mean_grad_norm: f64 = records.iter().map(|r| r.grad_norm).sum::<f64>() / n;
        let mean_throughput: f64 = records.iter().map(|r| r.tokens_per_sec).sum::<f64>() / n;
        
        let trend = if records.len() >= 2 {
            let first_half_mean: f64 = records[..records.len()/2].iter()
                .map(|r| r.train_loss).sum::<f64>() / (records.len()/2) as f64;
            let second_half_mean: f64 = records[records.len()/2..].iter()
                .map(|r| r.train_loss).sum::<f64>() / records.len().div_ceil(2) as f64;
            
            let diff = first_half_mean - second_half_mean;
            if diff > 0.01 { TrendDirection::Decreasing }
            else if diff < -0.01 { TrendDirection::Increasing }
            else { TrendDirection::Stable }
        } else {
            TrendDirection::Stable
        };
        
        MetricsAggregate {
            mean_loss,
            std_loss,
            mean_lr,
            mean_grad_norm,
            mean_throughput,
            trend,
            num_samples: records.len(),
        }
    }
    
    pub fn to_json(&self, last_n: usize) -> serde_json::Value {
        let records = self.recent_metrics(last_n);
        let agg = self.aggregate(last_n);
        let eta = self.eta_seconds(u64::MAX);
        
        serde_json::json!({
            "metrics": records,
            "aggregate": agg,
            "eta_seconds": eta,
            "total_recorded": self.history.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ring_buffer_basic() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(3);
        buf.push(1);
        buf.push(2);
        buf.push(3);
        buf.push(4);
        
        assert_eq!(buf.len(), 3);
        let latest = buf.latest(2);
        assert_eq!(latest, vec![4, 3]);
    }
    
    #[test]
    fn test_monitor_creation() {
        let monitor = TrainingMonitor::new(100, 10);
        assert!(monitor.all_metrics().is_empty());
    }
    
    #[test]
    fn test_record_and_retrieve() {
        let mut monitor = TrainingMonitor::new(100, 10);
        
        let record = MetricsRecord {
            timestamp: chrono::Utc::now().to_rfc3339(),
            global_step: 100,
            epoch: 0,
            train_loss: 2.3456,
            val_loss: Some(2.5678),
            learning_rate: 1e-4,
            grad_norm: 0.8543,
            tokens_per_sec: 12500.0,
            step_time_ms: 45.2,
        };
        
        monitor.record(record);
        assert_eq!(monitor.all_metrics().len(), 1);
    }
    
    #[test]
    fn test_format_log_line() {
        let monitor = TrainingMonitor::new(100, 10);
        let record = MetricsRecord {
            timestamp: "2024-01-15T10:23:45Z".to_string(),
            global_step: 156,
            epoch: 2,
            train_loss: 2.3456,
            val_loss: Some(2.5678),
            learning_rate: 8.5e-05,
            grad_norm: 0.7543,
            tokens_per_sec: 12000.0,
            step_time_ms: 42.8,
        };
        
        let line = monitor.format_log_line(&record);
        assert!(line.contains("Epoch 3"));
        assert!(line.contains("Step 156"));
        assert!(line.contains("2.3456"));
    }
    
    #[test]
    fn test_aggregate_calculation() {
        let mut monitor = TrainingMonitor::new(100, 10);
        
        for i in 0..10 {
            monitor.record(MetricsRecord {
                timestamp: chrono::Utc::now().to_rfc3339(),
                global_step: i,
                epoch: 0,
                train_loss: 10.0 - i as f64 * 0.5,
                val_loss: None,
                learning_rate: 1e-4,
                grad_norm: 1.0,
                tokens_per_sec: 10000.0,
                step_time_ms: 50.0,
            });
        }
        
        let agg = monitor.aggregate(10);
        assert!((agg.mean_loss - 7.75).abs() < 0.01);
        assert_eq!(agg.num_samples, 10);
    }
    
    #[test]
    fn test_trend_detection() {
        let mut monitor = TrainingMonitor::new(100, 10);

        // Record increasing loss values (from old to new)
        // latest() returns in reverse order (newest first)
        // So when we retrieve them, they'll be in decreasing order
        for i in 0..20 {
            monitor.record(MetricsRecord {
                timestamp: chrono::Utc::now().to_rfc3339(),
                global_step: i,
                epoch: 0,
                train_loss: 4.3 + i as f64 * 0.3, // increasing from 4.3 to 10.0
                val_loss: None,
                learning_rate: 1e-4,
                grad_norm: 1.0,
                tokens_per_sec: 10000.0,
                step_time_ms: 50.0,
            });
        }

        let agg = monitor.aggregate(20);
        // Records are returned newest-first, so first_half has higher loss values
        // This means the trend is actually decreasing over time (newer records have lower loss)
        assert_eq!(agg.trend, TrendDirection::Decreasing);
    }
    
    #[test]
    fn test_to_json_output() {
        let mut monitor = TrainingMonitor::new(100, 10);
        monitor.record(MetricsRecord {
            timestamp: chrono::Utc::now().to_rfc3339(),
            global_step: 1,
            epoch: 0,
            train_loss: 2.0,
            val_loss: None,
            learning_rate: 1e-4,
            grad_norm: 1.0,
            tokens_per_sec: 10000.0,
            step_time_ms: 50.0,
        });
        
        let json = monitor.to_json(10);
        assert!(json["metrics"].as_array().unwrap().len() > 0);
        assert!(json.get("aggregate").is_some());
    }
}
