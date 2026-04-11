import { get, post } from '@/utils/request'

export function getServiceStatus() {
  return get<any>('/admin/service/status')
}

export function getWorkerList() {
  return get<any>('/admin/service/workers')
}

export function restartService() {
  return post<any>('/admin/service/restart')
}

export function stopService() {
  return post<any>('/admin/service/stop')
}
