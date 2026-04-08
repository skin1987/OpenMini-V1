export interface ServiceStatus {
  status: 'healthy' | 'unhealthy' | 'degraded'
  timestamp: string
  uptime: number
  version: string
  components: ComponentStatus[]
}

export interface ComponentStatus {
  name: string
  status: 'up' | 'down' | 'degraded'
  message?: string
}

export interface ServiceInfo {
  version: string
  build_time: string
  start_time: string
  uptime: number
  hostname: string
  os: string
  arch: string
  node_version: string
  memory_usage: MemoryUsage
  cpu_usage: number
  worker_count: number
}

export interface MemoryUsage {
  rss: number
  heap_total: number
  heap_used: number
  external: number
  array_buffers: number
}

export interface WorkerInfo {
  id: number
  pid: number
  status: 'online' | 'offline' | 'starting' | 'stopping'
  memory_usage: MemoryUsage
  cpu_usage: number
  uptime: number
  request_count: number
  last_heartbeat: string
}

export interface RestartServiceResponse {
  success: boolean
  message: string
  restart_time: string
}
