import { get } from '@/utils/request'

export function getAuditLogs(params?: { page?: number; page_size?: number; action?: string }) {
  return get<any>('/admin/audit/logs', { params })
}

export function getAuditStats() {
  return get<any>('/admin/audit/stats')
}
