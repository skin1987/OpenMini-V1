import axios from 'axios'
import type { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios'
import { ElMessage } from 'element-plus'
import { useUserStore } from '@/stores/user'
import router from '@/router'

const service: AxiosInstance = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
})

service.interceptors.request.use(
  (config) => {
    const userStore = useUserStore()
    if (userStore.token) {
      config.headers.Authorization = `Bearer ${userStore.token}`
    }
    return config
  },
  (error) => {
    console.error('请求错误：', error)
    return Promise.reject(error)
  }
)

service.interceptors.response.use(
  (response: AxiosResponse) => {
    const res = response.data

    if (res.code !== 200 && res.code !== 0) {
      ElMessage.error(res.message || '请求失败')

      if (res.code === 401) {
        const userStore = useUserStore()
        userStore.resetState()
        router.push('/login')
      }

      if (res.code === 403) {
        ElMessage.error('权限不足')
      }

      return Promise.reject(new Error(res.message || '请求失败'))
    }

    return res
  },
  (error) => {
    console.error('响应错误：', error)

    if (error.response) {
      const status = error.response.status

      switch (status) {
        case 400:
          ElMessage.error('请求参数错误')
          break
        case 401:
          ElMessage.error('未授权，请重新登录')
          const userStore = useUserStore()
          userStore.resetState()
          router.push('/login')
          break
        case 403:
          ElMessage.error('权限不足')
          break
        case 404:
          ElMessage.error('请求地址不存在')
          break
        case 500:
          ElMessage.error('服务器内部错误')
          break
        default:
          ElMessage.error(`请求失败(${status})`)
      }
    } else {
      ElMessage.error('网络错误，请检查网络连接')
    }

    return Promise.reject(error)
  }
)

export function get<T = any>(url: string, config?: AxiosRequestConfig): Promise<T> {
  return service.get(url, config)
}

export function post<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
  return service.post(url, data, config)
}

export function put<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
  return service.put(url, data, config)
}

export function del<T = any>(url: string, config?: AxiosRequestConfig): Promise<T> {
  return service.delete(url, config)
}

export default service
