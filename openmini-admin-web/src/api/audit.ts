import { get } from '@/utils/request'
import type {
  AuditLog,
  AuditQueryParams,
  AuditLogDetailResponse,
  AuditStatsParams,
  AuditStats,
  ExportAuditLogsParams,
  ExportAuditLogsResponse
} from '@/types/api/audit'

/**
 * 获取审计日志列表
 * @param params - 多维度过滤和分页条件
 * @returns 分页的审计日志列表
 */
export function getAuditLogs(params?: AuditQueryParams): Promise<{
  items: AuditLog[]
  total: number
  page: number
  page_size: number
}> {
  return get('/admin/audit/logs', { params })
}

/**
 * 获取审计日志详情
 * @param id - 日志 ID
 * @returns 日志完整信息（含请求体、响应体、堆栈跟踪等）
 */
export function getAuditLogDetail(id: string): Promise<AuditLogDetailResponse> {
  return get<AuditLogDetailResponse>(`/admin/audit/logs/${id}`)
}

/**
 * 获取审计统计数据
 * @param params - 统计周期、分组维度、过滤条件
 * @returns 多维度的统计分析数据
 */
export function getAuditStats(params: AuditStatsParams): Promise<AuditStats> {
  return get<AuditStats>('/admin/audit/stats', { params })
}

/**
 * 导出审计日志
 * @param params - 导出格式、过滤条件和日期格式
 * @returns 下载链接和文件信息
 */
export function exportAuditLogs(params: ExportAuditLogsParams): Promise<ExportAuditLogsResponse> {
  return get<ExportAuditLogsResponse>('/admin/audit/export', { params })
}
