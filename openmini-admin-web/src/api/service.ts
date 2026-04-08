import { get, post } from '@/utils/request'
import type {
  ServiceStatus,
  ServiceInfo,
  WorkerInfo,
  RestartServiceResponse
} from '@/types/api/service'

const BASE_URL = '/v1'

/**
 * 获取服务健康状态
 * @returns 服务状态信息，包括各组件健康情况
 */
export function getServiceStatus(): Promise<ServiceStatus> {
  return get<ServiceStatus>(`${BASE_URL}/health`)
}

/**
 * 获取服务详细信息
 * @returns 服务版本、运行时长、资源使用等详细信息
 */
export function getServiceInfo(): Promise<ServiceInfo> {
  return get<ServiceInfo>(`${BASE_URL}/health`)
}

/**
 * 优雅重启服务
 * @returns 重启结果信息
 */
export function restartService(): Promise<RestartServiceResponse> {
  return post<RestartServiceResponse>('/admin/service/restart')
}

/**
 * 优雅停止服务
 * @returns 停止结果信息
 */
export function stopService(): Promise<{ success: boolean; message: string }> {
  return post('/admin/service/stop')
}

/**
 * 获取 Worker 列表
 * @returns 所有 Worker 的运行状态和资源使用情况
 */
export function getWorkerList(): Promise<WorkerInfo[]> {
  return get<WorkerInfo[]>('/admin/workers')
}

/**
 * 重启指定 Worker
 * @param id - Worker ID
 * @returns 重启结果信息
 */
export function restartWorker(id: number): Promise<{ success: boolean; message: string }> {
  return post(`/admin/workers/${id}/restart`)
}
