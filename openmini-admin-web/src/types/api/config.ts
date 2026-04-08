export interface ConfigData {
  server: ServerConfig
  model: ModelConfig
  security: SecurityConfig
  monitoring: MonitoringConfig
  logging: LoggingConfig
  [key: string]: any
}

export interface ServerConfig {
  host: string
  port: number
  workers: number
  timeout: number
  cors: CorsConfig
  rate_limiting: RateLimitingConfig
}

export interface CorsConfig {
  enabled: boolean
  origins: string[]
  methods: string[]
  allowed_headers: string[]
  max_age: number
}

export interface RateLimitingConfig {
  enabled: boolean
  window_ms: number
  max_requests: number
}

export interface ModelConfig {
  default_model: string
  models_dir: string
  auto_load: boolean
  cache_enabled: boolean
  cache_size_mb: number
  default_params: DefaultModelParams
}

export interface DefaultModelParams {
  temperature: number
  top_p: number
  top_k: number
  max_tokens: number
  repeat_penalty: number
}

export interface SecurityConfig {
  auth_enabled: boolean
  jwt_secret_masked: string
  session_timeout_minutes: number
  max_login_attempts: number
  lockout_duration_minutes: number
  api_key_required: boolean
  allowed_ips?: string[]
}

export interface MonitoringConfig {
  enabled: boolean
  metrics_port: number
  prometheus_enabled: boolean
  alert_enabled: boolean
  retention_days: number
}

export interface LoggingConfig {
  level: 'debug' | 'info' | 'warn' | 'error'
  file_enabled: boolean
  console_enabled: boolean
  log_dir: string
  max_file_size_mb: number
  max_files: number
  json_format: boolean
}

export interface ConfigSchema {
  $schema: string
  title: string
  description: string
  type: 'object'
  properties: Record<string, SchemaProperty>
  required: string[]
}

export interface SchemaProperty {
  type: string
  description: string
  default?: any
  enum?: any[]
  minimum?: number
  maximum?: number
  pattern?: string
  format?: string
  properties?: Record<string, SchemaProperty>
  items?: SchemaProperty
  sensitive?: boolean
}

export interface UpdateConfigRequest {
  section: keyof ConfigData
  changes: Partial<ConfigData[keyof ConfigData]>
  comment?: string
  updated_by: string
}

export interface UpdateConfigResponse {
  success: boolean
  message: string
  applied_changes: string[]
  requires_restart: boolean
  reload_time?: string
}

export interface ReloadConfigResponse {
  success: boolean
  message: string
  reload_time: string
  config_hash: string
}

export interface ConfigHistoryParams {
  page?: number
  page_size?: number
  section?: string
  start_time?: string
  end_time?: string
  changed_by?: string
  sort_by?: 'timestamp' | 'section'
  sort_order?: 'asc' | 'desc'
}

export interface ConfigHistoryItem {
  id: string
  timestamp: string
  section: string
  action: 'create' | 'update' | 'delete' | 'reload'
  changes: ConfigChangeDetail[]
  changed_by: string
  comment?: string
  rollback_available: boolean
}

export interface ConfigChangeDetail {
  field: string
  old_value: any
  new_value: any
  sensitive: boolean
}
