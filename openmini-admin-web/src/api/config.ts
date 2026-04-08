import { get, post, put } from '@/utils/request'
import type {
  ConfigData,
  ConfigSchema,
  UpdateConfigRequest,
  UpdateConfigResponse,
  ReloadConfigResponse,
  ConfigHistoryParams,
  ConfigHistoryItem
} from '@/types/api/config'

/**
 * 获取当前配置（敏感字段已脱敏）
 * @returns 完整配置对象（密码等敏感字段显示为 ***）
 */
export function getConfig(): Promise<ConfigData> {
  return get<ConfigData>('/admin/config')
}

/**
 * 获取配置 JSON Schema
 * @returns 用于前端表单校验和动态生成的 Schema 定义
 */
export function getConfigSchema(): Promise<ConfigSchema> {
  return get<ConfigSchema>('/admin/config/schema')
}

/**
 * 更新配置
 * @param data - 要更新的配置段和变更内容
 * @returns 更新结果（是否需要重启才能生效）
 */
export function updateConfig(data: UpdateConfigRequest): Promise<UpdateConfigResponse> {
  return put<UpdateConfigResponse>('/admin/config', data)
}

/**
 * 热重载配置（无需重启服务）
 * @returns 重载结果和配置哈希值
 */
export function reloadConfig(): Promise<ReloadConfigResponse> {
  return post<ReloadConfigResponse>('/admin/config/reload')
}

/**
 * 获取配置变更历史
 * @param params - 时间范围、配置段、操作人等过滤条件
 * @returns 分页的变更历史记录
 */
export function getConfigHistory(params?: ConfigHistoryParams): Promise<{
  items: ConfigHistoryItem[]
  total: number
  page: number
  page_size: number
}> {
  return get('/admin/config/history', { params })
}
