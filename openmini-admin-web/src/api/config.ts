import { get, post, put } from '@/utils/request'

export function getConfig() {
  return get<any>('/admin/config')
}

export function updateConfig(data: any) {
  return put<any>('/admin/config', data)
}

export function reloadConfig() {
  return post('/admin/config/reload')
}

export function getConfigHistory(params?: { page?: number; page_size?: number }) {
  return get<any>('/admin/config/history', { params })
}
