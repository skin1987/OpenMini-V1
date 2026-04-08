import type { ModelStatus } from '../index'

export interface ModelInfo {
  id: string
  name: string
  display_name: string
  description?: string
  status: ModelStatus
  provider: string
  size?: number
  format: string
  path: string
  loaded_at?: string
  created_at: string
  updated_at: string
  metadata?: Record<string, any>
}

export interface ModelDetail extends ModelInfo {
  parameters: number
  quantization: string
  context_length: number
  max_tokens: number
  supported_features: string[]
  capabilities: ModelCapabilities
  resource_usage: ModelResourceUsage
  performance_stats?: ModelPerformanceStats
}

export interface ModelCapabilities {
  chat_completion: boolean
  embedding: boolean
  function_calling: boolean
  vision: boolean
  streaming: boolean
}

export interface ModelResourceUsage {
  gpu_memory_mb: number
  cpu_memory_mb: number
  gpu_utilization: number
  temperature?: number
}

export interface ModelPerformanceStats {
  avg_latency_ms: number
  p50_latency_ms: number
  p95_latency_ms: number
  p99_latency_ms: number
  throughput_tps: number
  total_requests: number
  success_rate: number
}

export interface LoadModelRequest {
  model_id: string
  options?: LoadModelOptions
}

export interface LoadModelOptions {
  gpu_layers?: number
  ctx_size?: number
  n_batch?: number
  n_gqa?: number
  rms_norm_eps?: number
}

export interface SwitchModelRequest {
  from_model_id: string
  to_model_id: string
  graceful_timeout_ms?: number
}

export interface ModelHealthStatus {
  model_id: string
  is_healthy: boolean
  last_check_time: string
  error_message?: string
  response_time_ms: number
}

export interface ModelRegistryItem {
  id: string
  name: string
  display_name: string
  description?: string
  provider: string
  format: string
  size_bytes: number
  download_url?: string
  checksum?: string
  tags: string[]
  compatible_versions: string[]
  created_at: string
  updated_at: string
}
