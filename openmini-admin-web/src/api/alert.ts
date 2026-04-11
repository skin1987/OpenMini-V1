import { get, post, put, del, patch } from '@/utils/request'

export function getAlertRules() {
  return get<any[]>('/admin/alerts/rules')
}

export function createAlertRule(data: any) {
  return post<any>('/admin/alerts/rules', data)
}

export function updateAlertRule(id: string | number, data: any) {
  return put<any>(`/admin/alerts/rules/${id}`, data)
}

export function deleteAlertRule(id: string | number) {
  return del(`/admin/alerts/rules/${id}`)
}

export function toggleAlertRule(id: string | number) {
  return patch<any>(`/admin/alerts/rules/${id}/toggle`)
}

export function getAlertRecords(params?: { page?: number; page_size?: number }) {
  return get<any>('/admin/alerts/records', { params })
}

export function acknowledgeAlert(id: string | number) {
  return patch<any>(`/admin/alerts/records/${id}/ack`)
}

export function resolveAlert(id: string | number) {
  return patch<any>(`/admin/alerts/records/${id}/resolve`)
}

export function getAlertSummary() {
  return get<any>('/admin/alerts/records/summary')
}
