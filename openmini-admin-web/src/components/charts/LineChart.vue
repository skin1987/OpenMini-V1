<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import * as echarts from 'echarts'

interface Props {
  data: {
    xAxis: string[]
    series: Array<{ name: string; data: number[]; color?: string }>
  }
  height?: number
  title?: string
  showLegend?: boolean
  smooth?: boolean
  areaStyle?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  height: 300,
  title: '',
  showLegend: true,
  smooth: true,
  areaStyle: false
})

const chartRef = ref<HTMLDivElement>()
let chartInstance: echarts.ECharts | null = null

function initChart() {
  if (!chartRef.value) return

  if (chartInstance) {
    chartInstance.dispose()
  }

  chartInstance = echarts.init(chartRef.value)
  updateChart()
}

function updateChart() {
  if (!chartInstance) return

  const option: echarts.EChartsOption = {
    title: props.title ? { text: props.title, left: 'center' } : undefined,
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross' }
    },
    legend: props.showLegend ? { bottom: 0 } : undefined,
    grid: {
      left: '3%',
      right: '4%',
      bottom: props.showLegend ? '15%' : '3%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: props.data.xAxis,
      axisLabel: { fontSize: 11 }
    },
    yAxis: {
      type: 'value',
      axisLabel: { fontSize: 11 }
    },
    series: props.data.series.map((s) => ({
      name: s.name,
      type: 'line' as const,
      data: s.data,
      smooth: props.smooth,
      areaStyle: props.areaStyle ? {} : undefined,
      itemStyle: s.color ? { color: s.color } : undefined,
      lineStyle: s.color ? { color: s.color } : undefined
    }))
  }

  chartInstance.setOption(option, true)
}

let resizeObserver: ResizeObserver | null = null

onMounted(() => {
  initChart()

  if (chartRef.value) {
    resizeObserver = new ResizeObserver(() => {
      chartInstance?.resize()
    })
    resizeObserver.observe(chartRef.value)
  }
})

onUnmounted(() => {
  resizeObserver?.disconnect()
  chartInstance?.dispose()
  chartInstance = null
})

watch(
  () => props.data,
  () => {
    updateChart()
  },
  { deep: true }
)

defineExpose({ refresh: updateChart })
</script>

<template>
  <div ref="chartRef" :style="{ height: `${height}px` }"></div>
</template>
