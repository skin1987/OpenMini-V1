import { get, post, put } from '@/utils/request'

// ==================== 服务管理 API ====================

export function getServiceStatus() {
  return get<any>('/admin/service/status')
}

export function getWorkerList() {
  return get<any>('/admin/service/workers')
}

export function restartService() {
  return post<any>('/admin/service/restart')
}

export function stopService() {
  return post<any>('/admin/service/stop')
}

// ==================== 模型管理 API ====================

/**
 * 获取模型列表（支持分页和过滤）
 * @param params 查询参数
 * - page: 页码（默认 1）
 * - page_size: 每页数量（默认 20，最大 100）
 * - status: 状态过滤（available/loading/unloaded/error）
 * - provider: 提供商过滤（local/huggingface/openai 等）
 * - format: 格式过滤（gguf/safetensors 等）
 * - search: 搜索关键词
 */
export function getModelList(params?: {
  page?: number
  page_size?: number
  status?: string
  provider?: string
  format?: string
  search?: string
}) {
  return get<any>('/admin/model', params)
}

/**
 * 获取模型详细信息
 * @param modelId 模型 ID 或名称
 */
export function getModelDetail(modelId: string) {
  return get<any>(`/admin/model/${modelId}`)
}

/**
 * 加载模型到内存/GPU
 * @param data 加载参数
 * - model_path: 模型文件路径
 * - config: 可选的模型配置参数
 */
export function loadModel(data: {
  model_path: string
  config?: {
    context_length?: number
    max_tokens?: number
    temperature?: number
    top_p?: number
    top_k?: number
    repeat_penalty?: number
    gpu_layers?: number
    n_batch?: number
    n_ctx?: number
    mlock?: boolean
    mmap?: boolean
  }
}) {
  return post<any>('/admin/model', data)
}

/**
 * 卸载指定模型
 * @param modelId 模型 ID
 */
export function unloadModel(modelId: string) {
  return post<any>(`/admin/model/${modelId}/unload`)
}

/**
 * 更新模型运行配置
 * @param modelId 模型 ID
 * @param config 配置更新字段（支持部分更新）
 */
export function updateModelConfig(
  modelId: string,
  config: {
    context_length?: number
    max_tokens?: number
    temperature?: number
    top_p?: number
    top_k?: number
    repeat_penalty?: number
    gpu_layers?: number
    n_batch?: number
    n_ctx?: number
    mlock?: boolean
    mmap?: boolean
  }
) {
  return put<any>(`/admin/model/${modelId}`, config)
}

/**
 * 切换当前活跃模型（热切换）
 * @param data 切换参数
 * - from_model_id: 当前模型 ID
 * - to_model_id: 目标模型 ID
 * - graceful_timeout_ms: 优雅切换超时时间（毫秒），可选
 */
export function switchModel(data: {
  from_model_id: string
  to_model_id: string
  graceful_timeout_ms?: number
}) {
  return post<any>('/admin/model/switch', data)
}

/**
 * 检查模型健康状态
 * @param modelId 模型 ID
 */
export function checkModelHealth(modelId: string) {
  return get<any>(`/admin/model/${modelId}/health`)
}

// ==================== 会话监控 API ====================

/**
 * 获取活跃会话列表（支持分页、过滤）
 * @param params 查询参数
 * - page: 页码（默认 1）
 * - page_size: 每页数量（默认 20，最大 100）
 * - status: 状态过滤（active/completed/error）
 * - user_id: 用户 ID 过滤
 */
export function getSessionList(params?: {
  page?: number
  page_size?: number
  status?: string
  user_id?: number
}) {
  return get<any>('/admin/sessions', params)
}

/**
 * 获取会话详情（含消息历史）
 * @param sessionId 会话 ID
 */
export function getSessionDetail(sessionId: string) {
  return get<any>(`/admin/sessions/${sessionId}`)
}

/**
 * 获取会话统计信息（token使用量、响应时间等）
 * @param timeRange 时间范围
 * - last_hour / last_6_hours / last_24_hours / last_7_days / last_30_days
 * - custom: 自定义范围 { start: string, end: string }
 */
export function getSessionStats(timeRange?: string | { start: string; end: string }) {
  return get<any>('/admin/sessions/stats', { time_range: timeRange })
}

/**
 * 终止指定会话
 * @param sessionId 会话 ID
 */
export function terminateSession(sessionId: string) {
  return post<any>(`/admin/sessions/${sessionId}/terminate`)
}

/**
 * 批量清理过期会话
 * @param days 过期天数（默认 30 天）
 */
export function cleanupExpiredSessions(days?: number) {
  return post<any>('/admin/sessions/cleanup', { days: days || 30 })
}

// ==================== 指标仪表盘 API ====================

/**
 * 获取系统资源指标（CPU/GPU/内存/磁盘）
 */
export function getSystemMetrics() {
  return get<any>('/admin/metrics/system')
}

/**
 * 获取推理性能指标（TTFT/TPOT/QPS/throughput）
 */
export function getInferenceMetrics() {
  return get<any>('/admin/metrics/inference')
}

/**
 * 获取历史趋势数据
 * @param metricType 指标类型
 * - cpu_usage / memory_usage / gpu_usage / gpu_memory / gpu_temperature
 * - requests_per_second / tokens_per_second / latency_avg / latency_p95 / queue_length
 * @param params 时间范围参数
 * - time_range: 时间范围（5m/15m/1h/6h/24h/7d/30d）
 * - interval_seconds: 自定义采样间隔（可选）
 */
export function getMetricsHistory(
  metricType: string,
  params?: {
    time_range?: string
    interval_seconds?: number
  }
) {
  return get<any>(`/admin/metrics/history`, { ...params, metric_type: metricType })
}

/**
 * 获取实时监控概览（聚合多个指标用于 Dashboard 首页）
 */
export function getDashboardOverview() {
  return get<any>('/admin/metrics/dashboard')
}

/**
 * 获取告警阈值配置
 */
export function getAlertThresholds() {
  return get<any>('/admin/metrics/alerts/thresholds')
}

// ==================== 配置管理 API ====================

/**
 * 获取当前服务器完整配置
 */
export function getConfig() {
  return get<any>('/admin/config')
}

/**
 * 部分更新配置（支持深度合并）
 * 只需提供要更改的字段，未提供的字段保持不变。
 * @param config 包含任意要更新的配置段和字段
 * - change_reason: 可选的变更原因说明
 */
export function updateConfig(config: {
  server?: {
    host?: string
    port?: number
    max_connections?: number
    request_timeout_secs?: number
    cors_enabled?: boolean
  }
  thread_pool?: {
    size?: number
    max_queue_size?: number
  }
  memory?: {
    max_memory_gb?: number
    model_memory_gb?: number
    cache_memory_gb?: number
    enable_swap?: boolean
  }
  model?: {
    path?: string
    quantization?: string
    context_length?: number
    gpu_layers?: number
    default_temperature?: number
  }
  worker?: {
    count?: number
    restart_on_failure?: boolean
    max_restarts?: number
  }
  grpc?: {
    enabled?: boolean
    max_message_size_mb?: number
    port?: number
  }
  change_reason?: string
}) {
  return put<any>('/admin/config', config)
}

/**
 * 热重载配置（从磁盘重新读取配置文件）
 */
export function reloadConfig() {
  return post<any>('/admin/config/reload')
}

/**
 * 获取配置变更历史记录
 * @param params 查询参数
 * - page: 页码（默认 1）
 * - page_size: 每页数量（默认 20）
 * - section: 按配置节过滤（server/model/memory 等）
 */
export function getConfigHistory(params?: {
  page?: number
  page_size?: number
  section?: string
}) {
  return get<any>('/admin/config/history', params)
}

/**
 * 重启服务以应用需要重启才能生效的配置变更
 */
export function applyConfigRestart() {
  return post<any>('/admin/config/restart')
}

/**
 * 验证配置值的有效性（不实际应用）
 * 用于在正式提交前检查配置是否合法
 * @param config 要验证的配置片段
 */
export function validateConfig(config: any) {
  return post<any>('/admin/config/validate', config)
}

/**
 * 导出当前配置为 TOML 格式（用于备份或迁移）
 */
export function exportConfig() {
  return get<any>('/admin/config/export')
}
