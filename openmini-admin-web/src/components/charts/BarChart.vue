<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import * as echarts from 'echarts'

interface Props {
  data: {
    xAxis: string[]
    series: Array<{ name: string; data: number[]; color?: string }>
  }
  height?: number
  horizontal?: boolean
  showLegend?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  height: 300,
  horizontal: false,
  showLegend: true
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
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' }
    },
    legend: props.showLegend ? { bottom: 0 } : undefined,
    grid: {
      left: '3%',
      right: '4%',
      bottom: props.showLegend ? '15%' : '3%',
      containLabel: true
    },
    xAxis: props.horizontal
      ? { type: 'value', axisLabel: { fontSize: 11 } }
      : { type: 'category', data: props.data.xAxis, axisLabel: { fontSize: 11 } },
    yAxis: props.horizontal
      ? { type: 'category', data: props.data.xAxis, axisLabel: { fontSize: 11 } }
      : { type: 'value', axisLabel: { fontSize: 11 } },
    series: props.data.series.map((s) => ({
      name: s.name,
      type: 'bar' as const,
      data: s.data,
      itemStyle: s.color ? { color: s.color } : undefined,
      barWidth: '40%'
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
