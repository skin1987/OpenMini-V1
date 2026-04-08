import { get, post } from '@/utils/request'
import type {
  ModelInfo,
  ModelDetail,
  LoadModelRequest,
  SwitchModelRequest,
  ModelHealthStatus,
  ModelRegistryItem
} from '@/types/api/model'

const BASE_URL = '/v1/models'

/**
 * 获取模型列表
 * @returns 所有已注册的模型基本信息列表
 */
export function getModelList(): Promise<ModelInfo[]> {
  return get<ModelInfo[]>(BASE_URL)
}

/**
 * 获取模型详情
 * @param id - 模型 ID
 * @returns 模型完整信息，包括性能统计、资源使用等
 */
export function getModelDetail(id: string): Promise<ModelDetail> {
  return get<ModelDetail>(`${BASE_URL}/${id}`)
}

/**
 * 加载模型到内存
 * @param data - 加载参数（模型 ID 和可选配置）
 * @returns 模型加载结果
 */
export function loadModel(data: LoadModelRequest): Promise<{ success: boolean; message: string; model_id: string }> {
  return post('/admin/models/load', data)
}

/**
 * 卸载模型释放内存
 * @param id - 模型 ID
 * @returns 卸载结果信息
 */
export function unloadModel(id: string): Promise<{ success: boolean; message: string }> {
  return post(`/admin/models/${id}/unload`)
}

/**
 * 热切换模型（无缝替换正在服务的模型）
 * @param data - 切换参数（源模型、目标模型、超时时间）
 * @returns 切换结果信息
 */
export function switchModel(data: SwitchModelRequest): Promise<{
  success: boolean
  message: string
  from_model_id: string
  to_model_id: string
  switch_time_ms: number
}> {
  return post('/admin/models/switch', data)
}

/**
 * 检查模型健康状态
 * @param id - 模型 ID
 * @returns 模型健康检查结果
 */
export function checkModelHealth(id: string): Promise<ModelHealthStatus> {
  return get<ModelHealthStatus>(`/admin/models/${id}/health`)
}

/**
 * 获取模型注册表
 * @returns 所有可用模型（包括未加载的）的注册信息
 */
export function getModelRegistry(): Promise<ModelRegistryItem[]> {
  return get<ModelRegistryItem[]>('/admin/models/registry')
}
