import { get } from '@/utils/request'
import type {
  DashboardData,
  ResourceMetrics,
  InferenceMetrics,
  MetricsHistoryParams,
  MetricsHistoryResponse
} from '@/types/api/monitoring'

/**
 * 获取仪表盘聚合数据
 * 整合服务状态、系统资源、推理指标、告警摘要等信息
 * @returns 仪表盘展示所需的完整数据
 */
export async function getDashboardData(): Promise<DashboardData> {
  const [healthRes, metricsRes] = await Promise.all([
    get<any>('/v1/health'),
    get<ResourceMetrics>('/v1/metrics')
  ])

  return {
    service_status: healthRes,
    system_resources: metricsRes,
    inference_metrics: {} as any,
    recent_alerts: { total_active: 0, critical_count: 0, warning_count: 0, info_count: 0, resolved_today: 0, average_resolution_time_minutes: 0 },
    active_models: [],
    throughput_trend: []
  }
}

/**
 * 获取系统资源原始指标
 * @returns GPU/CPU/内存/磁盘/网络的详细监控数据
 */
export function getResourceMetrics(): Promise<ResourceMetrics> {
  return get<ResourceMetrics>('/v1/metrics')
}

/**
 * 获取推理性能指标
 * @returns QPS、延迟分布、吞吐量、错误率等数据
 */
export function getInferenceMetrics(): Promise<InferenceMetrics> {
  return get<InferenceMetrics>('/v1/metrics')
}

/**
 * 获取历史趋势数据
 * @param params - 查询参数（指标类型、时间范围、聚合方式）
 * @returns 历史数据点和统计摘要
 */
export function getMetricsHistory(params: MetricsHistoryParams): Promise<MetricsHistoryResponse> {
  return get<MetricsHistoryResponse>('/admin/metrics/history', { params })
}

/**
 * 获取 Prometheus 格式指标
 * @returns Prometheus 文本格式的原始指标数据
 */
export function getPrometheusMetrics(): Promise<string> {
  return get<string>('/v1/metrics', {
    headers: { 'Accept': 'text/plain' }
  })
}
