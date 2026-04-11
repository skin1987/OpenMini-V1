<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Monitor, Cpu, Coin, TrendCharts } from '@element-plus/icons-vue'
import { getDashboardData } from '@/api/monitoring'

const loading = ref(false)
let refreshTimer: number | null = null

const gpuStats = ref({
  utilization: 0,
  memoryUsed: 0,
  memoryTotal: 0,
  temperature: 0
})

const cpuStats = ref({
  usage: 0,
  cores: 0,
  loadAvg: [0, 0, 0]
})

const memoryStats = ref({
  used: 0,
  total: 0,
  swapUsed: 0
})

const inferenceStats = ref({
  qps: 0,
  tokenThroughput: 0,
  activeConnections: 0,
  batchSize: 0,
  avgLatency: 0,
  p50: 0,
  p95: 0,
  p99: 0
})

const componentHealth = ref([
  { name: 'GPU', status: 'success' as 'success' | 'warning' },
  { name: 'Memory', status: 'success' as 'success' | 'warning' },
  { name: 'CPU', status: 'success' as 'success' | 'warning' },
  { name: 'Scheduler', status: 'success' as 'success' | 'warning' },
  { name: 'Model', status: 'success' as 'success' | 'warning' }
])

const recentAlerts = ref([
  { id: 1, level: 'info', message: '暂无告警信息', time: '-' }
])

const chartData = ref({
  xAxis: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'],
  series: [
    {
      name: 'GPU利用率',
      data: [0, 0, 0, 0, 0, 0, 0],
      color: '#409EFF'
    },
    {
      name: 'CPU使用率',
      data: [0, 0, 0, 0, 0, 0, 0],
      color: '#67C23A'
    }
  ]
})

const memoryChartData = ref({
  xAxis: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'],
  series: [
    {
      name: '内存使用',
      data: [0, 0, 0, 0, 0, 0, 0],
      color: '#E6A23C'
    }
  ]
})

const serviceStatus = ref({
  status: '-',
  upstream: '-',
  version: '-'
})

const alertSummary = ref({
  firing: 0,
  acknowledged: 0,
  resolved: 0
})

async function fetchDashboardData() {
  loading.value = true
  try {
    const data = await getDashboardData()

    if (data) {
      if (data.service_status) {
        serviceStatus.value = {
          status: data.service_status.status || '-',
          upstream: data.service_status.upstream || '-',
          version: data.service_status.version || '-'
        }

        if (data.service_status.gpu?.[0]) {
          const gpu = data.service_status.gpu[0]
          gpuStats.value = {
            utilization: gpu.utilization_percent || 0,
            memoryUsed: (gpu.memory_used_mb / 1024) || 0,
            memoryTotal: (gpu.memory_total_mb / 1024) || 0,
            temperature: gpu.temperature_celsius || 0
          }
        }

        if (data.service_status.cpu) {
          cpuStats.value = {
            usage: data.service_status.cpu.usage_percent || 0,
            cores: data.service_status.cpu.cores || 0,
            loadAvg: [
              data.service_status.cpu.load_average_1m || 0,
              data.service_status.cpu.load_average_5m || 0,
              data.service_status.cpu.load_average_15m || 0
            ]
          }
        }

        if (data.service_status.memory) {
          memoryStats.value = {
            used: data.service_status.memory.used_gb || 0,
            total: data.service_status.memory.total_gb || 0,
            swapUsed: data.service_status.memory.swap_used_gb || 0
          }
        }

        if (data.service_status.inference) {
          inferenceStats.value = {
            qps: data.service_status.inference.requests_per_second || 0,
            tokenThroughput: data.service_status.inference.tokens_per_second || 0,
            activeConnections: data.service_status.inference.active_connections || 0,
            batchSize: 8,
            avgLatency: data.service_status.inference.avg_latency_ms || 0,
            p50: data.service_status.inference.p50_latency_ms || 0,
            p95: data.service_status.inference.p90_latency_ms || 0,
            p99: data.service_status.inference.p99_latency_ms || 0
          }
        }

        componentHealth.value = [
          { name: 'GPU', status: data.service_status.status === 'healthy' ? 'success' : 'warning' },
          { name: 'Memory', status: data.service_status.memory?.usage_percent > 80 ? 'warning' : 'success' },
          { name: 'CPU', status: data.service_status.cpu?.usage_percent > 80 ? 'warning' : 'success' },
          { name: 'Scheduler', status: 'success' },
          { name: 'Model', status: 'success' }
        ]
      }

      if (data.alert_summary) {
        alertSummary.value = {
          firing: data.alert_summary.firing || data.alert_summary.total_active || 0,
          acknowledged: data.alert_summary.acknowledged || 0,
          resolved: data.alert_summary.resolved || data.alert_summary.resolved_today || 0
        }

        const alerts = []
        if (data.alert_summary.critical_count > 0) {
          alerts.push({ id: alerts.length + 1, level: 'danger', message: `${data.alert_summary.critical_count} 个严重告警`, time: '-' })
        }
        if (data.alert_summary.warning_count > 0) {
          alerts.push({ id: alerts.length + 1, level: 'warning', message: `${data.alert_summary.warning_count} 个警告`, time: '-' })
        }
        if (alerts.length > 0) {
          recentAlerts.value = alerts
        }
      }
    }
  } catch (error) {
    console.error('获取仪表盘数据失败:', error)
    ElMessage.error('数据加载失败，显示默认值')
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  fetchDashboardData()
  refreshTimer = window.setInterval(fetchDashboardData, 10000)
})

onUnmounted(() => {
  if (refreshTimer) {
    clearInterval(refreshTimer)
  }
})
</script>

<template>
  <div class="dashboard-container" v-loading="loading">
    <el-row :gutter="20" class="stat-row">
      <el-col :xs="24" :sm="12" :md="6">
        <StatCard
          title="GPU利用率"
          :value="gpuStats.utilization.toFixed(1)"
          suffix="%"
          :icon="Monitor"
          color="default"
          :trend="{ value: 2.3, isUp: true }"
        >
          <div class="progress-wrapper">
            <el-progress
              :percentage="gpuStats.utilization"
              :color="'#409EFF'"
              :stroke-width="8"
            />
            <span class="detail-text">显存: {{ gpuStats.memoryUsed.toFixed(1) }}GB / {{ gpuStats.memoryTotal.toFixed(1) }}GB</span>
          </div>
        </StatCard>
      </el-col>

      <el-col :xs="24" :sm="12" :md="6">
        <StatCard
          title="GPU显存"
          :value="gpuStats.memoryUsed.toFixed(1)"
          suffix="GB"
          :icon="Coin"
          color="success"
          :trend="{ value: 1.5, isUp: true }"
        >
          <div class="progress-wrapper">
            <el-progress
              :percentage="gpuStats.memoryTotal > 0 ? (gpuStats.memoryUsed / gpuStats.memoryTotal) * 100 : 0"
              :color="'#67C23A'"
              :stroke-width="8"
            />
            <span class="detail-text">温度: {{ gpuStats.temperature }}°C</span>
          </div>
        </StatCard>
      </el-col>

      <el-col :xs="24" :sm="12" :md="6">
        <StatCard
          title="CPU使用率"
          :value="cpuStats.usage.toFixed(1)"
          suffix="%"
          :icon="Cpu"
          color="warning"
          :trend="{ value: 0.8, isUp: false }"
        >
          <div class="progress-wrapper">
            <el-progress
              :percentage="cpuStats.usage"
              :color="'#E6A23C'"
              :stroke-width="8"
            />
            <span class="detail-text">核心数: {{ cpuStats.cores }}</span>
          </div>
        </StatCard>
      </el-col>

      <el-col :xs="24" :sm="12" :md="6">
        <StatCard
          title="内存使用量"
          :value="memoryStats.used.toFixed(1)"
          suffix="GB"
          :icon="TrendCharts"
          color="danger"
          :trend="{ value: 3.2, isUp: true }"
        >
          <div class="progress-wrapper">
            <el-progress
              :percentage="memoryStats.total > 0 ? (memoryStats.used / memoryStats.total) * 100 : 0"
              :color="'#F56C6C'"
              :stroke-width="8"
            />
            <span class="detail-text">总计: {{ memoryStats.total.toFixed(1) }}GB</span>
          </div>
        </StatCard>
      </el-col>
    </el-row>

    <el-row :gutter="20" class="chart-row">
      <el-col :span="16">
        <el-card shadow="hover">
          <template #header>
            <span>GPU & CPU 趋势</span>
          </template>
          <LineChart :data="chartData" :height="320" :area-style="true" />
        </el-card>
      </el-col>

      <el-col :span="8">
        <el-card shadow="hover">
          <template #header>
            <span>内存趋势</span>
          </template>
          <LineChart :data="memoryChartData" :height="320" :show-legend="false" :area-style="true" />
        </el-card>
      </el-col>
    </el-row>

    <el-card shadow="hover" class="inference-panel">
      <template #header>
        <span>推理性能面板</span>
      </template>
      <el-row :gutter="20">
        <el-col :span="6">
          <div class="metric-item">
            <div class="metric-label">QPS</div>
            <div class="metric-value">{{ inferenceStats.qps }}</div>
            <div class="metric-unit">请求/秒</div>
          </div>
        </el-col>
        <el-col :span="6">
          <div class="metric-item">
            <div class="metric-label">Token吞吐量</div>
            <div class="metric-value">{{ (inferenceStats.tokenThroughput / 1000).toFixed(1) }}K</div>
            <div class="metric-unit">tokens/秒</div>
          </div>
        </el-col>
        <el-col :span="6">
          <div class="metric-item">
            <div class="metric-label">活跃连接</div>
            <div class="metric-value">{{ inferenceStats.activeConnections }}</div>
            <div class="metric-unit">个连接</div>
          </div>
        </el-col>
        <el-col :span="6">
          <div class="metric-item">
            <div class="metric-label">批处理大小</div>
            <div class="metric-value">{{ inferenceStats.batchSize }}</div>
            <div class="metric-unit">平均</div>
          </div>
        </el-col>
      </el-row>

      <el-divider />

      <el-row :gutter="20">
        <el-col :span="8">
          <div class="latency-item">
            <span>P50 延迟</span>
            <strong>{{ inferenceStats.p50 }}ms</strong>
          </div>
        </el-col>
        <el-col :span="8">
          <div class="latency-item">
            <span>P95 延迟</span>
            <strong>{{ inferenceStats.p95 }}ms</strong>
          </div>
        </el-col>
        <el-col :span="8">
          <div class="latency-item">
            <span>P99 延迟</span>
            <strong>{{ inferenceStats.p99 }}ms</strong>
          </div>
        </el-col>
      </el-row>
    </el-card>

    <el-row :gutter="20" class="bottom-row">
      <el-col :span="10">
        <el-card shadow="hover">
          <template #header>
            <span>健康状态总览</span>
          </template>
          <div class="health-list">
            <div v-for="item in componentHealth" :key="item.name" class="health-item">
              <StatusBadge :status="item.status" :text="item.name" />
            </div>
          </div>
        </el-card>
      </el-col>

      <el-col :span="14">
        <el-card shadow="hover">
          <template #header>
            <span>最近告警</span>
          </template>
          <el-table :data="recentAlerts" size="small" stripe>
            <el-table-column prop="level" label="级别" width="80">
              <template #default="{ row }">
                <StatusBadge
                  :status="row.level === 'danger' ? 'danger' : row.level === 'warning' ? 'warning' : 'info'"
                  :text="row.level === 'danger' ? '严重' : row.level === 'warning' ? '警告' : '提示'"
                />
              </template>
            </el-table-column>
            <el-table-column prop="message" label="消息" show-overflow-tooltip />
            <el-table-column prop="time" label="时间" width="100" />
          </el-table>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style lang="scss" scoped>
.dashboard-container {
  .stat-row {
    margin-bottom: $spacing-lg;
  }

  .chart-row {
    margin-bottom: $spacing-lg;
  }

  .inference-panel {
    margin-bottom: $spacing-lg;

    .metric-item {
      text-align: center;
      padding: $spacing-md;

      .metric-label {
        font-size: $font-size-sm;
        color: $text-secondary;
        margin-bottom: $spacing-xs;
      }

      .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: $primary-color;
        line-height: 1.2;
      }

      .metric-unit {
        font-size: $font-size-xs;
        color: $text-placeholder;
        margin-top: 4px;
      }
    }

    .latency-item {
      text-align: center;
      padding: $spacing-sm;

      span {
        display: block;
        font-size: $font-size-sm;
        color: $text-secondary;
        margin-bottom: 4px;
      }

      strong {
        font-size: 22px;
        color: $primary-color;
      }
    }
  }

  .bottom-row {
    .health-list {
      display: flex;
      flex-wrap: wrap;
      gap: $spacing-md;
      padding: $spacing-md 0;
    }
  }

  .progress-wrapper {
    .detail-text {
      display: block;
      margin-top: $spacing-sm;
      font-size: $font-size-xs;
      color: $text-secondary;
    }
  }
}
</style>
