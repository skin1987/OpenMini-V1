<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'

const loading = ref(false)
const gpuChartRef = ref<HTMLElement>()
const cpuChartRef = ref<HTMLElement>()
const memoryChartRef = ref<HTMLElement>()
let gpuInstance: echarts.ECharts | null = null
let cpuInstance: echarts.ECharts | null = null
let memoryInstance: echarts.ECharts | null = null
let timer: ReturnType<typeof setInterval> | null = null

const resourceStats = ref({
  gpuUtilization: 78.5,
  gpuMemoryUsed: 18.2,
  gpuMemoryTotal: 24,
  cpuUtilization: 45.3,
  memoryUsed: 32.6,
  memoryTotal: 64,
  kvCacheUsed: 8.4,
  kvCacheTotal: 16,
  temperature: 68,
  powerDraw: 285
})

function generateTimeData(count: number) {
  const now = Date.now()
  return Array.from({ length: count }, (_, i) => new Date(now - (count - 1 - i) * 10000).toLocaleTimeString('zh-CN', { hour12: false }))
}

function initCharts() {
  if (gpuChartRef.value) {
    gpuInstance = echarts.init(gpuChartRef.value)
    gpuInstance.setOption({
      tooltip: { trigger: 'axis' },
      legend: { data: ['GPU 利用率', '显存使用'], bottom: 0 },
      grid: { top: 15, right: 15, bottom: 35, left: 50 },
      xAxis: { type: 'category', data: generateTimeData(30), axisLabel: { fontSize: 11 } },
      yAxis: [
        { type: 'value', name: '利用率(%)', min: 0, max: 100, axisLabel: { formatter: '{value}%' } },
        { type: 'value', name: '显存(GB)', min: 0, max: 24 }
      ],
      series: [
        { name: 'GPU 利用率', type: 'line', data: Array.from({ length: 30 }, () => Math.floor(Math.random() * 25 + 65)), smooth: true, itemStyle: { color: '#409eff' }, areaStyle: { color: 'rgba(64,158,255,0.1)' } },
        { name: '显存使用', type: 'line', yAxisIndex: 1, data: Array.from({ length: 30 }, () => +(Math.random() * 2 + 17).toFixed(1)), smooth: true, itemStyle: { color: '#e6a23c' } }
      ]
    })
  }

  if (cpuChartRef.value) {
    cpuInstance = echarts.init(cpuChartRef.value)
    cpuInstance.setOption({
      tooltip: { trigger: 'axis' },
      legend: { data: ['CPU 使用率'], bottom: 0 },
      grid: { top: 15, right: 15, bottom: 35, left: 50 },
      xAxis: { type: 'category', data: generateTimeData(30), axisLabel: { fontSize: 11 } },
      yAxis: { type: 'value', min: 0, max: 100, axisLabel: { formatter: '{value}%' } },
      series: [{ name: 'CPU 使用率', type: 'bar', data: Array.from({ length: 30 }, () => Math.floor(Math.random() * 20 + 35)), itemStyle: { color: '#67c23a', borderRadius: [2, 2, 0, 0] } }]
    })
  }

  if (memoryChartRef.value) {
    memoryInstance = echarts.init(memoryChartRef.value)
    memoryInstance.setOption({
      tooltip: { trigger: 'axis', formatter: (params: any) => {
        const d = params[0]
        return `${d.name}<br/>内存: ${d.data[1]}GB<br/>KV Cache: ${d.data[2]}GB`
      }},
      legend: { data: ['内存使用', 'KV Cache'], bottom: 0 },
      grid: { top: 15, right: 15, bottom: 35, left: 50 },
      xAxis: { type: 'category', data: generateTimeData(30), axisLabel: { fontSize: 11 } },
      yAxis: { type: 'value', name: 'GB', max: 64 },
      series: [
        { name: '内存使用', type: 'area', stack: 'total', data: Array.from({ length: 30 }, () => +(Math.random() * 3 + 31).toFixed(1)), itemStyle: { color: '#f56c6c' }, areaStyle: { color: 'rgba(245,108,108,0.3)' } },
        { name: 'KV Cache', type: 'area', stack: 'total', data: Array.from({ length: 30 }, () => +(Math.random() * 2 + 7).toFixed(1)), itemStyle: { color: '#909399' }, areaStyle: { color: 'rgba(144,147,153,0.3)' } }
      ]
    })
  }
}

function refreshData() {
  resourceStats.value.gpuUtilization = +(Math.random() * 10 + 73).toFixed(1)
  resourceStats.value.cpuUtilization = +(Math.random() * 15 + 40).toFixed(1)
  resourceStats.value.memoryUsed = +(Math.random() * 2 + 31.5).toFixed(1)
  resourceStats.value.temperature = Math.floor(Math.random() * 5 + 65)
  resourceStats.value.powerDraw = Math.floor(Math.random() * 20 + 275)

  if (gpuInstance) {
    const opt = gpuInstance.getOption()
    const newGpu = Math.floor(Math.random() * 25 + 65)
    const newMem = +(Math.random() * 2 + 17).toFixed(1)
    ;(opt?.series as any)?.[0]?.data?.shift()
    ;(opt?.series as any)?.[0]?.data?.push(newGpu)
    ;(opt?.series as any)?.[1]?.data?.shift()
    ;(opt?.series as any)?.[1]?.data?.push(+newMem)
    ;(opt?.xAxis as any)?.[0]?.data?.shift()
    ;(opt?.xAxis as any)?.[0]?.data?.push(new Date().toLocaleTimeString('zh-CN', { hour12: false }))
    gpuInstance.setOption(opt)
  }
}

onMounted(() => {
  initCharts()
  timer = setInterval(refreshData, 5000)
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  if (timer) clearInterval(timer)
  gpuInstance?.dispose()
  cpuInstance?.dispose()
  memoryInstance?.dispose()
  window.removeEventListener('resize', handleResize)
})

function handleResize() {
  gpuInstance?.resize()
  cpuInstance?.resize()
  memoryInstance?.resize()
}
</script>

<template>
  <div class="resources-container">
    <PageHeader title="资源监控" subtitle="系统硬件资源实时监控" />

    <!-- 核心指标卡片 -->
    <el-row :gutter="16" class="stat-row">
      <el-col :span="6">
        <StatCard title="GPU 利用率" :value="resourceStats.gpuUtilization" suffix="%" :trend="{ value: 2.1, isUp: true }" color="primary" />
      </el-col>
      <el-col :span="6">
        <StatCard title="显存使用" :value="resourceStats.gpuMemoryUsed" suffix="GB" :trend="{ value: 0.3, isUp: true }" color="warning">
          <template #footer>
            <el-progress :percentage="(resourceStats.gpuMemoryUsed / resourceStats.gpuMemoryTotal * 100)" :stroke-width="6" />
            <span class="footer-text">总计 {{ resourceStats.gpuMemoryTotal }} GB</span>
          </template>
        </StatCard>
      </el-col>
      <el-col :span="6">
        <StatCard title="CPU 使用率" :value="resourceStats.cpuUtilization" suffix="%" :trend="{ value: 1.2, isUp: false }" color="success" />
      </el-col>
      <el-col :span="6">
        <StatCard title="内存使用" :value="resourceStats.memoryUsed" suffix="GB" color="danger">
          <template #footer>
            <el-progress :percentage="(resourceStats.memoryUsed / resourceStats.memoryTotal * 100)" :stroke-width="6" status="warning" />
            <span class="footer-text">KV Cache: {{ resourceStats.kvCacheUsed }} / {{ resourceStats.kvCacheTotal }} GB</span>
          </template>
        </StatCard>
      </el-col>
    </el-row>

    <!-- 图表区域 -->
    <el-row :gutter="16" class="chart-row">
      <el-col :span="24">
        <el-card shadow="hover">
          <template #header><span>GPU & 显存趋势（最近 5 分钟）</span></template>
          <div ref="gpuChartRef" style="height: 320px" v-loading="loading" />
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="16" class="chart-row">
      <el-col :span="12">
        <el-card shadow="hover">
          <template #header><span>CPU 使用率</span></template>
          <div ref="cpuChartRef" style="height: 280px" />
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card shadow="hover">
          <template #header><span>内存分布（含 KV Cache）</span></template>
          <div ref="memoryChartRef" style="height: 280px" />
        </el-card>
      </el-col>
    </el-row>

    <!-- 硬件详情 -->
    <el-row :gutter="16" class="chart-row">
      <el-col :span="12">
        <el-card shadow="hover">
          <template #header><span>GPU 硬件信息</span></template>
          <el-descriptions :column="2" border>
            <el-descriptions-item label="型号">NVIDIA A100-SXM4-80GB</el-descriptions-item>
            <el-descriptions-item label="驱动版本">550.54.14</el-descriptions-item>
            <el-descriptions-item label="CUDA 版本">12.4</el-descriptions-item>
            <el-descriptions-item label="温度">{{ resourceStats.temperature }}°C</el-descriptions-item>
            <el-descriptions-item label="功耗">{{ resourceStats.powerDraw }}W / 400W</el-descriptions-item>
            <el-descriptions-item label="风扇转速">42%</el-descriptions-item>
          </el-descriptions>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card shadow="hover">
          <template #header><span>内存分配明细</span></template>
          <div ref="memoryPieRef" style="height: 260px" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style lang="scss" scoped>
.resources-container {
  .stat-row { margin-bottom: 16px; }
  .chart-row { margin-bottom: 16px; }
  .footer-text {
    display: block;
    margin-top: 6px;
    font-size: 12px;
    color: #909399;
  }
}
</style>
