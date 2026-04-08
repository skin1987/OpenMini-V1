import { get, post, put, del } from '@/utils/request'
import type {
  ApiKeyInfo,
  ApiKeyQueryParams,
  CreateApiKeyRequest,
  CreateApiKeyResponse,
  ToggleApiKeyResponse,
  ApiKeyUsage,
  UpdateApiKeyQuotaRequest
} from '@/types/api/apikey'

/**
 * 获取 API Key 列表
 * @param params - 分页、搜索、筛选条件
 * @returns 分页的 Key 列表
 */
export function getApiKeyList(params?: ApiKeyQueryParams): Promise<{
  items: ApiKeyInfo[]
  total: number
  page: number
  page_size: number
}> {
  return get('/admin/apikeys', { params })
}

/**
 * 创建 API Key
 * @param data - Key 配置（名称、权限范围、配额限制等）
 * @returns 创建成功的信息（包含完整密钥，仅此一次显示）
 */
export function createApiKey(data: CreateApiKeyRequest): Promise<CreateApiKeyResponse> {
  return post<CreateApiKeyResponse>('/admin/apikeys', data)
}

/**
 * 废弃（删除）API Key
 * @param id - Key ID
 * @returns 删除结果
 */
export function deleteApiKey(id: string): Promise<{ success: boolean; message: string }> {
  return del(`/admin/apikeys/${id}`)
}

/**
 * 启用或禁用 API Key
 * @param id - Key ID
 * @param enabled - 是否启用
 * @returns 操作结果和新状态
 */
export function toggleApiKey(id: string, enabled: boolean): Promise<ToggleApiKeyResponse> {
  return post<ToggleApiKeyResponse>(`/admin/apikeys/${id}/toggle`, { enabled })
}

/**
 * 获取 API Key 用量统计
 * @param id - Key ID
 * @returns 详细用量数据（按日/月统计、热门接口、错误分析）
 */
export function getApiKeyUsage(id: string): Promise<ApiKeyUsage> {
  return get<ApiKeyUsage>(`/admin/apikeys/${id}/usage`)
}

/**
 * 设置 API Key 配额限制
 * @param id - Key ID
 * @param data - 新配额设置（请求次数、Token 数量限制）
 * @returns 更新结果
 */
export function updateApiKeyQuota(
  id: string,
  data: UpdateApiKeyQuotaRequest
): Promise<{ success: boolean; message: string }> {
  return put(`/admin/apikeys/${id}/quota`, data)
}
