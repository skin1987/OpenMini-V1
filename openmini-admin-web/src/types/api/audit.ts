export interface AuditLog {
  id: string
  timestamp: string
  user_id: string
  username: string
  action: AuditAction
  resource_type: ResourceType
  resource_id?: string
  details: AuditLogDetails
  ip_address: string
  user_agent: string
  result: 'success' | 'failure'
  error_message?: string
  duration_ms: number
  metadata?: Record<string, any>
}

export type AuditAction =
  | 'login'
  | 'logout'
  | 'create'
  | 'read'
  | 'update'
  | 'delete'
  | 'config_change'
  | 'model_load'
  | 'model_unload'
  | 'service_restart'
  | 'alert_trigger'
  | 'alert_acknowledge'
  | 'alert_resolve'
  | 'user_create'
  | 'user_update'
  | 'user_delete'
  | 'apikey_create'
  | 'apikey_revoke'
  | 'permission_change'
  | 'export_data'

export type ResourceType =
  | 'user'
  | 'api_key'
  | 'model'
  | 'config'
  | 'alert_rule'
  | 'audit_log'
  | 'system'
  | 'session'
  | 'unknown'

export interface AuditLogDetails {
  before?: Record<string, any>
  after?: Record<string, any>
  changes?: Array<{
    field: string
    old_value: any
    new_value: any
  }>
  request_params?: Record<string, any>
  response_status?: number
  additional_info?: string
}

export interface AuditQueryParams {
  page?: number
  page_size?: number
  keyword?: string
  user_id?: string
  action?: AuditAction
  resource_type?: ResourceType
  resource_id?: string
  result?: 'success' | 'failure'
  start_time?: string
  end_time?: string
  ip_address?: string
  sort_by?: 'timestamp' | 'action' | 'user_id' | 'duration_ms'
  sort_order?: 'asc' | 'desc'
}

export interface AuditLogDetailResponse extends AuditLog {
  full_request_body?: any
  full_response_body?: any
  stack_trace?: string
  related_logs?: AuditLog[]
}

export interface AuditStatsParams {
  start_time: string
  end_time: string
  group_by?: 'action' | 'resource_type' | 'user_id' | 'result' | 'hour' | 'day'
  filter_action?: AuditAction
  filter_resource_type?: ResourceType
  filter_user_id?: string
  filter_result?: 'success' | 'failure'
}

export interface PeriodInfo {
  start_time: string
  end_time: string
  total_days: number
}

export interface StatsOverview {
  total_events: number
  success_count: number
  failure_count: number
  success_rate: number
  unique_users: number
  unique_ips: number
  avg_duration_ms: number
  max_duration_ms: number
  events_per_minute: number
}

export interface StatBreakdown {
  category: string
  count: number
  percentage: number
  trend?: 'up' | 'down' | 'stable'
}

export interface UserActivityStat {
  user_id: string
  username: string
  event_count: number
  success_count: number
  failure_count: number
  last_action_time: string
  most_common_action: string
}

export interface HourlyStat {
  hour: number
  count: number
  success_count: number
  failure_count: number
}

export interface ErrorAnalysis {
  total_errors: number
  error_rate: number
  common_errors: CommonError[]
  error_by_action: StatBreakdown[]
}

export interface CommonError {
  error_message: string
  count: number
  percentage: number
  first_occurrence: string
  last_occurrence: string
}

export interface TrendDataPoint {
  timestamp: string
  count: number
}

export interface AuditStats {
  period: PeriodInfo
  overview: StatsOverview
  action_breakdown: StatBreakdown[]
  resource_type_breakdown: StatBreakdown[]
  user_activity: UserActivityStat[]
  hourly_distribution: HourlyStat[]
  error_analysis: ErrorAnalysis
  trend_data: TrendDataPoint[]
}

export interface ExportAuditLogsParams extends AuditQueryParams {
  format: 'csv' | 'json' | 'excel'
  include_details?: boolean
  date_format?: string
}

export interface ExportAuditLogsResponse {
  download_url: string
  filename: string
  file_size_bytes: number
  record_count: number
  expires_at: string
}
