<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import * as echarts from 'echarts'

interface DataItem {
  name: string
  value: number
}

interface Props {
  data: DataItem[]
  height?: number
  title?: string
  showLegend?: boolean
  radius?: string[]
}

const props = withDefaults(defineProps<Props>(), {
  height: 300,
  title: '',
  showLegend: true,
  radius: () => ['40%', '70%']
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
      trigger: 'item',
      formatter: '{b}: {c} ({d}%)'
    },
    legend: props.showLegend ? { bottom: 0 } : undefined,
    series: [
      {
        type: 'pie',
        radius: props.radius,
        avoidLabelOverlap: true,
        itemStyle: {
          borderRadius: 6,
          borderColor: '#fff',
          borderWidth: 2
        },
        label: { show: true, formatter: '{b}\n{d}%' },
        data: props.data
      }
    ]
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
