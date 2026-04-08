export interface ApiKeyInfo {
  id: string
  key_prefix: string
  name: string
  description?: string
  created_by: string
  created_at: string
  last_used_at?: string
  expires_at?: string
  status: 'active' | 'disabled' | 'expired' | 'revoked'
  scopes: string[]
  rate_limit_rpm: number
  quota_limit?: ApiKeyQuota
  usage_current: ApiKeyUsageCurrent
  metadata?: Record<string, string>
}

export interface ApiKeyQuota {
  daily_requests: number
  monthly_requests: number
  daily_tokens: number
  monthly_tokens: number
}

export interface ApiKeyUsageCurrent {
  today_requests: number
  month_requests: number
  today_tokens: number
  month_tokens: number
  last_reset_date: string
}

export interface ApiKeyQueryParams {
  page?: number
  page_size?: number
  keyword?: string
  status?: 'active' | 'disabled' | 'expired' | 'revoked'
  created_by?: string
  sort_by?: 'name' | 'created_at' | 'last_used_at'
  sort_order?: 'asc' | 'desc'
}

export interface CreateApiKeyRequest {
  name: string
  description?: string
  expires_at?: string
  scopes?: string[]
  rate_limit_rpm?: number
  quota_limit?: Omit<ApiKeyQuota, never>
  metadata?: Record<string, string>
}

export interface CreateApiKeyResponse {
  id: string
  key: string
  key_prefix: string
  name: string
  created_at: string
  expires_at?: string
  warning: string
}

export interface ToggleApiKeyResponse {
  success: boolean
  api_key_id: string
  new_status: 'active' | 'disabled'
}

export interface DailyUsage {
  date: string
  requests: number
  tokens: number
  errors: number
}

export interface MonthlyUsageSummary {
  month: string
  total_requests: number
  total_tokens: number
  total_errors: number
  avg_daily_requests: number
  peak_day: string
}

export interface EndpointUsage {
  endpoint: string
  method: string
  request_count: number
  avg_latency_ms: number
}

export interface ErrorBreakdown {
  error_code: string
  count: number
  percentage: number
}

export interface ApiKeyUsage {
  api_key_id: string
  daily_usage: DailyUsage[]
  monthly_summary: MonthlyUsageSummary
  top_endpoints: EndpointUsage[]
  error_breakdown: ErrorBreakdown[]
}

export interface UpdateApiKeyQuotaRequest {
  daily_requests?: number
  monthly_requests?: number
  daily_tokens?: number
  monthly_tokens?: number
}
