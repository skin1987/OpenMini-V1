import { get, post, put, del } from '@/utils/request'
import type {
  UserInfo,
  UserQueryParams,
  CreateUserRequest,
  UpdateUserRequest,
  UpdateUserRoleRequest,
  UpdateUserStatusRequest,
  ResetPasswordResponse,
  UpdateMyPasswordRequest,
  UserDetailResponse
} from '@/types/api/user'
import type { UserRole } from '@/types'

/**
 * 获取用户列表
 * @param params - 分页、搜索、筛选条件
 * @returns 分页的用户列表
 */
export function getUserList(params?: UserQueryParams): Promise<{
  items: UserInfo[]
  total: number
  page: number
  page_size: number
}> {
  return get('/admin/users', { params })
}

/**
 * 创建用户
 * @param data - 用户信息（用户名、密码、角色等）
 * @returns 创建成功的新用户信息
 */
export function createUser(data: CreateUserRequest): Promise<UserInfo> {
  return post<UserInfo>('/admin/users', data)
}

/**
 * 获取用户详情
 * @param id - 用户 ID
 * @returns 用户完整信息（含登录历史、活动统计）
 */
export function getUserDetail(id: string): Promise<UserDetailResponse> {
  return get<UserDetailResponse>(`/admin/users/${id}`)
}

/**
 * 更新用户信息
 * @param id - 用户 ID
 * @param data - 需要更新的字段
 * @returns 更新后的用户信息
 */
export function updateUser(id: string, data: UpdateUserRequest): Promise<UserInfo> {
  return put<UserInfo>(`/admin/users/${id}`, data)
}

/**
 * 删除用户
 * @param id - 用户 ID
 * @returns 删除结果
 */
export function deleteUser(id: string): Promise<{ success: boolean; message: string }> {
  return del(`/admin/users/${id}`)
}

/**
 * 修改用户角色
 * @param id - 用户 ID
 * @param role - 新角色
 * @returns 操作结果
 */
export function updateUserRole(id: string, role: UserRole): Promise<{ success: boolean; message: string }> {
  return put(`/admin/users/${id}/role`, { role, updated_by: '' } as UpdateUserRoleRequest)
}

/**
 * 启用或禁用用户
 * @param id - 用户 ID
 * @param status - 目标状态（active/disabled）
 * @returns 操作结果
 */
export function updateUserStatus(
  id: string,
  status: 'active' | 'disabled'
): Promise<{ success: boolean; message: string }> {
  return put(`/admin/users/${id}/status`, { status, updated_by: '' } as UpdateUserStatusRequest)
}

/**
 * 重置用户密码
 * @param id - 用户 ID
 * @returns 重置结果（含临时密码）
 */
export function resetUserPassword(id: string): Promise<ResetPasswordResponse> {
  return put<ResetPasswordResponse>(`/admin/users/${id}/password`)
}

/**
 * 当前用户修改自己的密码
 * @param data - 旧密码和新密码
 * @returns 修改结果
 */
export function updateMyPassword(data: UpdateMyPasswordRequest): Promise<{ success: boolean; message: string }> {
  return put('/admin/users/me/password', data)
}
