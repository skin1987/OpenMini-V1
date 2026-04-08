export interface DashboardData {
  service_status: ServiceHealthSummary
  system_resources: ResourceMetrics
  inference_metrics: InferenceMetricsSummary
  recent_alerts: AlertSummary
  active_models: ActiveModelSummary[]
  throughput_trend: TrendDataPoint[]
}

export interface ServiceHealthSummary {
  status: 'healthy' | 'unhealthy' | 'degraded'
  uptime_percentage: number
  last_restart: string
  component_count: number
  healthy_components: number
}

export interface ResourceMetrics {
  timestamp: string
  cpu: CpuMetrics
  memory: MemoryMetrics
  gpu: GpuMetrics[]
  disk: DiskMetrics
  network: NetworkMetrics
}

export interface CpuMetrics {
  usage_percent: number
  cores: number
  load_average_1m: number
  load_average_5m: number
  load_average_15m: number
  process_count: number
  thread_count: number
}

export interface MemoryMetrics {
  total_gb: number
  used_gb: number
  free_gb: number
  usage_percent: number
  swap_total_gb: number
  swap_used_gb: number
  swap_usage_percent: number
}

export interface GpuMetrics {
  device_id: number
  name: string
  utilization_percent: number
  memory_total_mb: number
  memory_used_mb: number
  memory_free_mb: number
  memory_usage_percent: number
  temperature_celsius: number
  power_draw_watts: number
  power_limit_watts: number
  fan_speed_percent: number
}

export interface DiskMetrics {
  total_gb: number
  used_gb: number
  free_gb: number
  usage_percent: number
  read_iops: number
  write_iops: number
  read_throughput_mb_s: number
  write_throughput_mb_s: number
}

export interface NetworkMetrics {
  bytes_received: number
  bytes_sent: number
  packets_received: number
  packets_sent: number
  errors_in: number
  errors_out: number
  connections_active: number
}

export interface InferenceMetrics {
  timestamp: string
  requests_per_second: number
  avg_latency_ms: number
  p50_latency_ms: number
  p90_latency_ms: number
  p99_latency_ms: number
  tokens_per_second: number
  input_tokens_count: number
  output_tokens_count: number
  active_connections: number
  queue_length: number
  error_rate: number
  success_rate: number
  cache_hit_rate: number
}

export interface InferenceMetricsSummary extends InferenceMetrics {
  trend_direction: 'up' | 'down' | 'stable'
  trend_percent_change: number
}

export interface AlertSummary {
  total_active: number
  critical_count: number
  warning_count: number
  info_count: number
  resolved_today: number
  average_resolution_time_minutes: number
}

export interface ActiveModelSummary {
  id: string
  name: string
  status: string
  requests_count: number
  avg_latency_ms: number
}

export interface TrendDataPoint {
  timestamp: string
  value: number
}

export interface MetricsHistoryParams {
  metric_type: 'cpu' | 'memory' | 'gpu' | 'inference' | 'throughput' | 'latency'
  start_time: string
  end_time: string
  interval?: '1m' | '5m' | '15m' | '1h' | '1d'
  aggregation?: 'avg' | 'max' | 'min' | 'sum' | 'count'
}

export interface MetricsHistoryResponse {
  metric_type: string
  data_points: TrendDataPoint[]
  statistics: {
    min: number
    max: number
    avg: number
    current: number
  }
}
