import { get, post, del, patch } from '@/utils/request'

export function getApiKeyList(params?: { page?: number; page_size?: number }) {
  return get<any>('/admin/apikeys', { params })
}

export function createApiKey(data: { name?: string; owner_id?: number }) {
  return post<any>('/admin/apikeys', data)
}

export function deleteApiKey(id: string | number) {
  return del(`/admin/apikeys/${id}`)
}

export function toggleApiKey(id: string | number) {
  return patch<any>(`/admin/apikeys/${id}/toggle`)
}

export function getApiKeyUsage(id: string | number) {
  return get<any>(`/admin/apikeys/${id}/usage`)
}
