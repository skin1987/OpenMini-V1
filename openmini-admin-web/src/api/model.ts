import { get, post } from '@/utils/request'

export function getModelList() {
  return get<any>('/admin/model')
}

export function loadModel(data: { path?: string }) {
  return post<any>('/admin/model', data)
}

export function unloadModel(id: string | number) {
  return post<any>(`/admin/model/${id}/unload`)
}

export function switchModel(data: { target: string }) {
  return post<any>('/admin/model/switch', data)
}

export function checkModelHealth(id: string | number) {
  return get<any>(`/admin/model/${id}/health`)
}
