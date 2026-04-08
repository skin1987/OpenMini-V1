// 通用 API 响应结构
export interface ApiResponse<T = any> {
  code: number
  message: string
  data: T
}

// 分页响应结构
export interface PaginatedResponse<T = any> {
  list: T[]
  total: number
  page: number
  pageSize: number
}

// 用户角色常量（使用 const 对象替代 enum 以兼容 erasableSyntaxOnly）
export const UserRole = {
  ADMIN: 'admin',
  OPERATOR: 'operator',
  VIEWER: 'viewer'
} as const

export type UserRole = (typeof UserRole)[keyof typeof UserRole]

// 登录表单数据
export interface LoginForm {
  username: string
  password: string
  remember?: boolean
}

// 登录响应数据
export interface LoginResponse {
  access_token: string
  refresh_token: string
  expires_in: number
  user_info: UserInfo
}

// 用户信息
export interface UserInfo {
  id: number | string
  username: string
  email: string
  role: UserRole
  avatar: string
  last_login_at: string
  created_at: string
}

// 模型状态常量（使用 const 对象替代 enum）
export const ModelStatus = {
  TRAINING: 'training',
  COMPLETED: 'completed',
  FAILED: 'failed',
  PENDING: 'pending'
} as const

export type ModelStatus = (typeof ModelStatus)[keyof typeof ModelStatus]

// 告警严重级别常量（使用 const 对象替代 enum）
export const AlertSeverity = {
  INFO: 'info',
  WARNING: 'warning',
  ERROR: 'error',
  CRITICAL: 'critical'
} as const

export type AlertSeverity = (typeof AlertSeverity)[keyof typeof AlertSeverity]

// 告警状态常量（使用 const 对象替代 enum）
export const AlertStatus = {
  ACTIVE: 'active',
  ACKNOWLEDGED: 'acknowledged',
  RESOLVED: 'resolved'
} as const

export type AlertStatus = (typeof AlertStatus)[keyof typeof AlertStatus]
