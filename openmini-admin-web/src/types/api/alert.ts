import type { AlertSeverity, AlertStatus } from '../index'

export interface AlertRule {
  id: string
  name: string
  description?: string
  severity: AlertSeverity
  condition: AlertCondition
  enabled: boolean
  notification_channels: NotificationChannel[]
  cooldown_seconds: number
  created_by: string
  created_at: string
  updated_at: string
  last_triggered_at?: string
  trigger_count: number
}

export interface AlertCondition {
  metric: string
  operator: 'gt' | 'lt' | 'eq' | 'gte' | 'lte' | 'in' | 'not_in'
  threshold: number | string | boolean
  duration_seconds?: number
  labels?: Record<string, string>
}

export interface NotificationChannel {
  type: 'email' | 'webhook' | 'slack' | 'dingtalk' | 'wechat'
  config: Record<string, string>
  enabled: boolean
}

export interface CreateAlertRuleRequest {
  name: string
  description?: string
  severity: AlertSeverity
  condition: AlertCondition
  notification_channels: Omit<NotificationChannel, 'id'>[]
  cooldown_seconds?: number
}

export interface UpdateAlertRuleRequest extends Partial<CreateAlertRuleRequest> {
  enabled?: boolean
}

export interface AlertRecord {
  id: string
  rule_id: string
  rule_name: string
  severity: AlertSeverity
  status: AlertStatus
  message: string
  details?: Record<string, any>
  triggered_at: string
  acknowledged_at?: string
  acknowledged_by?: string
  resolved_at?: string
  resolved_by?: string
  labels?: Record<string, string>
}

export interface AlertQueryParams {
  page?: number
  page_size?: number
  keyword?: string
  severity?: AlertSeverity
  status?: AlertStatus
  rule_id?: string
  start_time?: string
  end_time?: string
  sort_by?: string
  sort_order?: 'asc' | 'desc'
}

export interface AcknowledgeAlertRequest {
  acknowledged_by: string
  comment?: string
}

export interface ResolveAlertRequest {
  resolved_by: string
  resolution_comment?: string
}

export interface ToggleAlertRuleResponse {
  success: boolean
  rule_id: string
  new_enabled_state: boolean
}

export interface TestAlertRequest {
  rule_id?: string
  channel_type: NotificationChannel['type']
  channel_config?: Record<string, string>
  test_message?: string
}

export interface TestAlertResponse {
  success: boolean
  message: string
  sent_at: string
  delivery_details?: Record<string, any>
}

export interface AlertSummaryData {
  total_rules: number
  active_rules: number
  total_records: number
  active_alerts: number
  critical_count: number
  warning_count: number
  info_count: number
  resolved_today: number
  unresolved_count: number
  avg_resolution_time_minutes: number
  most_triggered_rule?: string
  alerts_by_severity: Record<AlertSeverity, number>
  alerts_by_hour: { hour: number; count: number }[]
  recent_trend: 'increasing' | 'decreasing' | 'stable'
}
