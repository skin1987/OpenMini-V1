import { get } from '@/utils/request'

export async function getDashboardData() {
  const [serviceStatus, alertSummary] = await Promise.all([
    get<any>('/admin/service/status').catch(() => null),
    get<any>('/admin/alerts/records/summary').catch(() => null)
  ])

  return {
    service_status: serviceStatus,
    alert_summary: alertSummary
  }
}

export function getResourceMetrics() {
  return get<any>('/admin/service/status')
}
