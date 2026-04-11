<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { RefreshRight } from '@element-plus/icons-vue'
import * as echarts from 'echarts'

const loading = ref(false)
const latencyChartRef = ref<HTMLElement>()
const throughputChartRef = ref<HTMLElement>()
let latencyInstance: echarts.ECharts | null = null
let throughputInstance: echarts.ECharts | null = null
let timer: ReturnType<typeof setInterval> | null = null

const inferenceStats = ref({
  qps: 156,
  avgLatency: 123,
  p50Latency: 89,
  p95Latency: 256,
  p99Latency: 892,
  tokenThroughput: 12400,
  activeConnections: 42,
  batchSize: 8,
  totalRequests: 1284567,
  totalTokens: 89234156,
  errorRate: 0.03,
  cacheHitRate: 94.5
})

function initCharts() {
  if (latencyChartRef.value) {
    latencyInstance = echarts.init(latencyChartRef.value)
    latencyInstance.setOption({
      tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
      legend: { data: ['P50', 'P95', 'P99'], bottom: 0 },
      grid: { top: 15, right: 15, bottom: 35, left: 55 },
      xAxis: { type: 'category', data: ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00'], boundaryGap: false },
      yAxis: { type: 'value', name: '延迟(ms)', axisLabel: { formatter: '{value}ms' } },
      series: [
        { name: 'P50', type: 'line', data: [72, 68, 75, 82, 95, 88, 85, 91, 87, 92, 86, 79], smooth: true, itemStyle: { color: '#67c23a' } },
        { name: 'P95', type: 'line', data: [210, 195, 230, 245, 280, 265, 250, 270, 255, 275, 240, 225], smooth: true, itemStyle: { color: '#e6a23c' } },
        { name: 'P99', type: 'line', data: [720, 680, 850, 920, 1100, 980, 920, 1050, 960, 1080, 890, 800], smooth: true, itemStyle: { color: '#f56c6c' }, areaStyle: { color: 'rgba(245,108,108,0.08)' } }
      ]
    })
  }

  if (throughputChartRef.value) {
    throughputInstance = echarts.init(throughputChartRef.value)
    throughputInstance.setOption({
      tooltip: { trigger: 'axis' },
      legend: { data: ['Token 吞吐量', '请求数'], bottom: 0 },
      grid: { top: 15, right: 15, bottom: 35, left: 55 },
      xAxis: { type: 'category', data: ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00'] },
      yAxis: [
        { type: 'value', name: 'tok/s', position: 'left' },
        { type: 'value', name: 'QPS', position: 'right' }
      ],
      series: [
        { name: 'Token 吞吐量', type: 'bar', data: [10200, 9800, 11500, 12100, 13800, 13200, 12500, 14000, 12900, 13500, 12000, 11800], itemStyle: { color: '#409eff', borderRadius: [2, 2, 0, 0] } },
        { name: '请求数', type: 'line', yAxisIndex: 1, data: [98, 95, 120, 135, 168, 155, 142, 175, 158, 165, 138, 128], smooth: true, itemStyle: { color: '#909399' } }
      ]
    })
  }
}

function refreshData() {
  inferenceStats.value.qps = Math.floor(Math.random() * 30 + 140)
  inferenceStats.value.avgLatency = Math.floor(Math.random() * 40 + 100)
  inferenceStats.value.tokenThroughput = Math.floor(Math.random() * 2000 + 11500)
  inferenceStats.value.activeConnections = Math.floor(Math.random() * 15 + 35)
}

onMounted(() => {
  initCharts()
  timer = setInterval(refreshData, 8000)
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  if (timer) clearInterval(timer)
  latencyInstance?.dispose()
  throughputInstance?.dispose()
  window.removeEventListener('resize', handleResize)
})

function handleResize() {
  latencyInstance?.resize()
  throughputInstance?.resize()
}
</script>

<template>
  <div class="inference-container">
    <PageHeader title="推理指标" subtitle="LLM 推理性能实时监控">

      <template #extra>
        <el-button :icon="RefreshRight" circle @click="refreshData" />
      </template>
    </PageHeader>

    <!-- 核心性能指标 -->
    <el-row :gutter="16" class="stat-row">
      <el-col :span="6">
        <StatCard title="QPS" :value="inferenceStats.qps" :trend="{ value: 5.2, isUp: true }" color="primary" />
      </el-col>
      <el-col :span="6">
        <StatCard title="平均延迟" :value="inferenceStats.avgLatency" suffix="ms" :trend="{ value: 3.1, isUp: false }" color="success" />
      </el-col>
      <el-col :span="6">
        <StatCard title="Token 吞吐" :value="(inferenceStats.tokenThroughput / 1000).toFixed(1)" suffix="K/s" :trend="{ value: 2.8, isUp: true }" color="warning" />
      </el-col>
      <el-col :span="6">
        <StatCard title="活跃连接" :value="inferenceStats.activeConnections" color="danger" />
      </el-col>
    </el-row>

    <!-- 延迟分布 -->
    <el-row :gutter="16" class="chart-row">
      <el-col :span="16">
        <el-card shadow="hover">
          <template #header><span>延迟分布趋势（P50/P95/P99）</span></template>
          <div ref="latencyChartRef" style="height: 320px" v-loading="loading" />
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card shadow="hover">
          <template #header><span>延迟分位数</span></template>
          <div class="latency-list">
            <div class="latency-item">
              <span class="label">P50</span>
              <el-progress :percentage="(inferenceStats.p50Latency / inferenceStats.p99Latency * 100)" :stroke-width="14" status="success" />
              <span class="value">{{ inferenceStats.p50Latency }}ms</span>
            </div>
            <div class="latency-item">
              <span class="label">P95</span>
              <el-progress :percentage="(inferenceStats.p95Latency / inferenceStats.p99Latency * 100)" :stroke-width="14" status="warning" />
              <span class="value">{{ inferenceStats.p95Latency }}ms</span>
            </div>
            <div class="latency-item">
              <span class="label">P99</span>
              <el-progress :percentage="100" :stroke-width="14" status="exception" />
              <span class="value">{{ inferenceStats.p99Latency }}ms</span>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 吞吐量图表 -->
    <el-row :gutter="16" class="chart-row">
      <el-col :span="16">
        <el-card shadow="hover">
          <template #header><span>吞吐量趋势（24h）</span></template>
          <div ref="throughputChartRef" style="height: 300px" />
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card shadow="hover">
          <template #header><span>推理统计</span></template>
          <div class="stats-grid">
            <div class="stat-item">
              <span class="label">总请求数</span>
              <span class="value">{{ inferenceStats.totalRequests.toLocaleString() }}</span>
            </div>
            <div class="stat-item">
              <span class="label">总 Token 数</span>
              <span class="value">{{ (inferenceStats.totalTokens / 1000000).toFixed(2) }}M</span>
            </div>
            <div class="stat-item">
              <span class="label">错误率</span>
              <span class="value error">{{ (inferenceStats.errorRate * 100).toFixed(2) }}%</span>
            </div>
            <div class="stat-item">
              <span class="label">缓存命中率</span>
              <span class="value success">{{ inferenceStats.cacheHitRate }}%</span>
            </div>
            <div class="stat-item">
              <span class="label">批处理大小</span>
              <span class="value">{{ inferenceStats.batchSize }}</span>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style lang="scss" scoped>
.inference-container {
  .stat-row { margin-bottom: 16px; }
  .chart-row { margin-bottom: 16px; }

  .latency-list {
    padding: 8px 0;
    .latency-item {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 20px;
      &:last-child { margin-bottom: 0; }
      .label { width: 36px; font-size: 13px; font-weight: 600; color: #606266; flex-shrink: 0; }
      .value { width: 56px; text-align: right; font-weight: bold; font-size: 14px; flex-shrink: 0; }
    }
  }

  .stats-grid {
    .stat-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 0;
      border-bottom: 1px solid #ebeef5;
      &:last-child { border-bottom: none; }
      .label { font-size: 13px; color: #909399; }
      .value { font-size: 16px; font-weight: 600; color: #303133; }
      .value.error { color: #f56c6c; }
      .value.success { color: #67c23a; }
    }
  }
}
</style>
