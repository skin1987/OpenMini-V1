import type { UserRole } from '../index'

export interface UserInfo {
  id: string
  username: string
  email: string
  display_name: string
  avatar_url?: string
  role: UserRole
  status: 'active' | 'disabled' | 'locked'
  last_login_at?: string
  created_at: string
  updated_at: string
  created_by?: string
  permissions: string[]
}

export interface UserQueryParams {
  page?: number
  page_size?: number
  keyword?: string
  role?: UserRole
  status?: 'active' | 'disabled' | 'locked'
  sort_by?: 'username' | 'email' | 'created_at' | 'last_login_at'
  sort_order?: 'asc' | 'desc'
}

export interface CreateUserRequest {
  username: string
  email: string
  password: string
  display_name: string
  role: UserRole
  avatar_url?: string
  permissions?: string[]
}

export interface UpdateUserRequest {
  email?: string
  display_name?: string
  avatar_url?: string
  role?: UserRole
  permissions?: string[]
}

export interface UpdateUserRoleRequest {
  role: UserRole
  updated_by: string
}

export interface UpdateUserStatusRequest {
  status: 'active' | 'disabled'
  reason?: string
  updated_by: string
}

export interface ResetPasswordResponse {
  success: boolean
  message: string
  temporary_password?: string
  expires_at?: string
}

export interface UpdateMyPasswordRequest {
  old_password: string
  new_password: string
  confirm_password: string
}

export interface UserDetailResponse extends UserInfo {
  login_history: UserLoginRecord[]
  activity_summary: UserActivitySummary
}

export interface UserLoginRecord {
  login_time: string
  ip_address: string
  user_agent: string
  location?: string
  status: 'success' | 'failed'
  failure_reason?: string
}

export interface UserActivitySummary {
  total_logins: number
  actions_this_month: number
  last_action_time: string
  most_used_feature: string
}
