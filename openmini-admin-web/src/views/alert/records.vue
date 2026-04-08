<script setup lang="ts">
import { ref, onMounted, nextTick, computed } from 'vue'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts'
import {
  getAlertRecords,
  acknowledgeAlert,
  resolveAlert,
  getAlertSummary
} from '@/api/alert'
import type { AlertRecord, AlertQueryParams, AcknowledgeAlertRequest, ResolveAlertRequest } from '@/types/api/alert'
import type { AlertSeverity, AlertStatus } from '@/types'

const loading = ref(false)
const recordList = ref<AlertRecord[]>([])
const total = ref(0)
const currentPage = ref(1)
const pageSize = ref(10)
const severityFilter = ref('')
const statusFilter = ref('')
const timeRange = ref('24h')
const keyword = ref('')
const customTimeRange = ref<[Date, Date] | null>(null)

const summaryData = ref<{
  firing: number
  acknowledged: number
  resolved: number
  total: number
}>({
  firing: 0,
  acknowledged: 0,
  resolved: 0,
  total: 0
})

let chartInstance: echarts.ECharts | null = null

const queryParams = computed<AlertQueryParams>(() => {
  const params: AlertQueryParams = {
    page: currentPage.value,
    page_size: pageSize.value,
    keyword: keyword.value || undefined,
    severity: (severityFilter.value as AlertSeverity) || undefined,
    status: (statusFilter.value as AlertStatus) || undefined
  }

  if (timeRange.value && timeRange.value !== 'custom') {
    const now = new Date()
    const rangeMap: Record<string, number> = {
      '1h': 3600000,
      '6h': 21600000,
      '24h': 86400000,
      '7d': 604800000,
      '30d': 2592000000
    }
    const ms = rangeMap[timeRange.value]
    if (ms) {
      params.start_time = new Date(now.getTime() - ms).toISOString()
      params.end_time = now.toISOString()
    }
  }

  if (timeRange.value === 'custom' && customTimeRange.value) {
    params.start_time = customTimeRange.value[0].toISOString()
    params.end_time = customTimeRange.value[1].toISOString()
  }

  return params
})

onMounted(() => {
  fetchRecords()
  renderTrendChart()
})

async function fetchRecords() {
  loading.value = true
  try {
    const res = await getAlertRecords(queryParams.value)
    recordList.value = res.items
    total.value = res.total

    // 计算统计摘要
    summaryData.value = {
      firing: res.items.filter(r => r.status === 'firing').length,
      acknowledged: res.items.filter(r => r.status === 'acknowledged').length,
      resolved: res.items.filter(r => r.status === 'resolved').length,
      total: res.total
    }
  } catch (error) {
    ElMessage.error('获取告警记录失败')
  } finally {
    loading.value = false
  }
}

function handleSearch() {
  currentPage.value = 1
  fetchRecords()
}

function handleFilterChange() {
  currentPage.value = 1
  fetchRecords()
}

function handleSizeChange(val: number) {
  pageSize.value = val
  currentPage.value = 1
  fetchRecords()
}

function handleCurrentChange(val: number) {
  currentPage.value = val
  fetchRecords()
}

async function handleAcknowledge(record: AlertRecord) {
  try {
    const data: AcknowledgeAlertRequest = {
      acknowledged_by: 'current_user',
      comment: ''
    }
    await acknowledgeAlert(record.id, data)
    ElMessage.success('已确认该告警')
    fetchRecords()
  } catch (error) {
    ElMessage.error('确认失败')
  }
}

async function handleResolve(record: AlertRecord) {
  try {
    const data: ResolveAlertRequest = {
      resolved_by: 'current_user',
      resolution_comment: ''
    }
    await resolveAlert(record.id, data)
    ElMessage.success('已解决该告警')
    fetchRecords()
  } catch (error) {
    ElMessage.error('解决失败')
  }
}

function getSeverityType(severity: string): 'danger' | 'warning' | 'info' {
  const map: Record<string, 'danger' | 'warning' | 'info'> = {
    critical: 'danger',
    warning: 'warning',
    info: 'info'
  }
  return map[severity] || 'info'
}

function getSeverityText(severity: string): string {
  const map: Record<string, string> = {
    critical: '严重',
    warning: '警告',
    info: '信息'
  }
  return map[severity] || severity
}

function getStatusType(status: string): 'danger' | 'warning' | 'success' | 'info' {
  const map: Record<string, 'danger' | 'warning' | 'success' | 'info'> = {
    firing: 'danger',
    acknowledged: 'warning',
    resolved: 'success'
  }
  return map[status] || 'info'
}

function getStatusText(status: string): string {
  const map: Record<string, string> = {
    firing: '触发中',
    acknowledged: '已确认',
    resolved: '已解决'
  }
  return map[status] || status
}

function formatTime(time: string) {
  return time.slice(0, 16).replace('T', ' ')
}

function calcDuration(triggeredAt: string): string {
  const start = new Date(triggeredAt)
  const now = new Date()
  const diffMs = now.getTime() - start.getTime()

  const hours = Math.floor(diffMs / 3600000)
  const minutes = Math.floor((diffMs % 3600000) / 60000)

  if (hours > 0) {
    return `${hours}小时${minutes > 0 ? minutes + '分钟' : ''}`
  }
  return `${minutes}分钟`
}

function getRowClassName({ row }: { row: AlertRecord }) {
  if (row.status === 'firing') return 'row-firing'
  if (row.status === 'acknowledged') return 'row-acknowledged'
  return 'row-resolved'
}

function renderTrendChart() {
  nextTick(() => {
    const chartDom = document.getElementById('trend-chart')
    if (!chartDom) return

    if (chartInstance) {
      chartInstance.dispose()
    }

    chartInstance = echarts.init(chartDom)

    // Mock 数据：近 7 天每日告警数量
    const dates = []
    const criticalData = []
    const warningData = []
    const infoData = []

    for (let i = 6; i >= 0; i--) {
      const date = new Date()
      date.setDate(date.getDate() - i)
      dates.push(`${date.getMonth() + 1}/${date.getDate()}`)

      // 随机生成 mock 数据
      criticalData.push(Math.floor(Math.random() * 5))
      warningData.push(Math.floor(Math.random() * 8) + 2)
      infoData.push(Math.floor(Math.random() * 6) + 1)
    }

    chartInstance.setOption({
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' }
      },
      legend: {
        data: ['严重', '警告', '信息'],
        bottom: 0
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '15%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: dates
      },
      yAxis: {
        type: 'value',
        name: '告警数量'
      },
      series: [
        {
          name: '严重',
          type: 'bar',
          stack: 'total',
          itemStyle: { color: '#f56c6c' },
          data: criticalData
        },
        {
          name: '警告',
          type: 'bar',
          stack: 'total',
          itemStyle: { color: '#e6a23c' },
          data: warningData
        },
        {
          name: '信息',
          type: 'bar',
          stack: 'total',
          itemStyle: { color: '#909399' },
          data: infoData
        }
      ]
    })
  })
}
</script>

<template>
  <div class="page-container">
    <!-- 页面标题 -->
    <div class="page-header">
      <h2>告警记录</h2>
    </div>

    <!-- 统计卡片区 -->
    <el-row :gutter="16" class="stats-row">
      <el-col :span="6">
        <el-card class="stat-card stat-firing" shadow="hover">
          <el-statistic title="触发中" :value="summaryData.firing" />
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card stat-acknowledged" shadow="hover">
          <el-statistic title="已确认" :value="summaryData.acknowledged" />
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card stat-resolved" shadow="hover">
          <el-statistic title="已解决" :value="summaryData.resolved" />
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card stat-total" shadow="hover">
          <el-statistic title="总数" :value="summaryData.total" />
        </el-card>
      </el-col>
    </el-row>

    <!-- 筛选工具栏 -->
    <el-card class="toolbar-card" shadow="never">
      <div class="toolbar">
        <el-select
          v-model="severityFilter"
          placeholder="级别筛选"
          clearable
          style="width: 120px"
          @change="handleFilterChange"
        >
          <el-option label="全部" value="" />
          <el-option label="严重" value="critical" />
          <el-option label="警告" value="warning" />
          <el-option label="信息" value="info" />
        </el-select>

        <el-select
          v-model="statusFilter"
          placeholder="状态筛选"
          clearable
          style="width: 120px"
          @change="handleFilterChange"
        >
          <el-option label="全部" value="" />
          <el-option label="触发中" value="firing" />
          <el-option label="已确认" value="acknowledged" />
          <el-option label="已解决" value="resolved" />
        </el-select>

        <el-select
          v-model="timeRange"
          placeholder="时间范围"
          style="width: 130px"
          @change="handleFilterChange"
        >
          <el-option label="最近 1 小时" value="1h" />
          <el-option label="最近 6 小时" value="6h" />
          <el-option label="最近 24 小时" value="24h" />
          <el-option label="最近 7 天" value="7d" />
          <el-option label="最近 30 天" value="30d" />
          <el-option label="自定义" value="custom" />
        </el-select>

        <el-date-picker
          v-if="timeRange === 'custom'"
          v-model="customTimeRange"
          type="datetimerange"
          range-separator="至"
          start-placeholder="开始时间"
          end-placeholder="结束时间"
          style="width: 380px"
          @change="handleFilterChange"
        />

        <el-input
          v-model="keyword"
          placeholder="关键词搜索"
          clearable
          style="width: 200px; margin-left: auto"
          @keyup.enter="handleSearch"
          @clear="handleSearch"
        >
          <template #prefix>
            <el-icon><Search /></el-icon>
          </template>
        </el-input>
      </div>
    </el-card>

    <!-- 告警记录表格 -->
    <el-card shadow="never" class="table-card">
      <el-table
        :data="recordList"
        v-loading="loading"
        stripe
        border
        style="width: 100%"
        :row-class-name="getRowClassName"
      >
        <el-table-column label="级别" width="90" align="center">
          <template #default="{ row }">
            <div class="severity-indicator">
              <span
                class="dot"
                :class="'dot-' + row.severity"
              ></span>
              <el-tag
                :type="getSeverityType(row.severity)"
                effect="dark"
                size="small"
              >
                {{ getSeverityText(row.severity) }}
              </el-tag>
            </div>
          </template>
        </el-table-column>

        <el-table-column label="规则名称" min-width="160">
          <template #default="{ row }">
            <el-link type="primary" :underline="false">
              {{ row.rule_name }}
            </el-link>
          </template>
        </el-table-column>

        <el-table-column label="消息内容" min-width="220">
          <template #default="{ row }">
            <el-tooltip :content="row.message" placement="top" :show-after="300">
              <span class="message-text">{{ row.message }}</span>
            </el-tooltip>
          </template>
        </el-table-column>

        <el-table-column label="当前值" width="100">
          <template #default="{ row }">
            <span
              :class="{ 'value-exceeded': Number(row.details?.current_value) > Number(row.details?.threshold) }"
            >
              {{ row.details?.current_value ?? '-' }}
            </span>
          </template>
        </el-table-column>

        <el-table-column label="触发时间" width="160">
          <template #default="{ row }">
            {{ formatTime(row.triggered_at) }}
          </template>
        </el-table-column>

        <el-table-column label="持续时长" width="110">
          <template #default="{ row }">
            {{ calcDuration(row.triggered_at) }}
          </template>
        </el-table-column>

        <el-table-column label="状态" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)" effect="dark" size="small">
              {{ getStatusText(row.status) }}
            </el-tag>
          </template>
        </el-table-column>

        <el-table-column label="确认人" width="100">
          <template #default="{ row }">
            {{ row.acknowledged_by || '-' }}
          </template>
        </el-table-column>

        <el-table-column label="操作" width="140" fixed="right">
          <template #default="{ row }">
            <template v-if="row.status === 'firing'">
              <el-button link type="primary" size="small" @click="handleAcknowledge(row)">
                确认
              </el-button>
            </template>
            <template v-else-if="row.status === 'acknowledged'">
              <el-button link type="success" size="small" @click="handleResolve(row)">
                解决
              </el-button>
            </template>
            <template v-else-if="row.status === 'resolved'">
              <div class="resolved-info">
                <div>{{ formatTime(row.resolved_at!) }}</div>
                <div class="resolver">解决人: {{ row.resolved_by }}</div>
              </div>
            </template>
          </template>
        </el-table-column>
      </el-table>

      <!-- 分页 -->
      <div class="pagination-wrapper">
        <el-pagination
          v-model:current-page="currentPage"
          v-model:page-size="pageSize"
          :total="total"
          :page-sizes="[10, 20, 50, 100]"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleSizeChange"
          @current-change="handleCurrentChange"
        />
      </div>
    </el-card>

    <!-- 告警趋势图 -->
    <el-card class="chart-card" shadow="never">
      <template #header>
        <span class="chart-title">近 7 天告警趋势</span>
      </template>
      <div id="trend-chart" style="width: 100%; height: 350px"></div>
    </el-card>
  </div>
</template>

<style lang="scss" scoped>
.page-container {
  padding: 20px;
  background: #f5f7fa;
  min-height: calc(100vh - 84px);
}

.page-header {
  margin-bottom: 16px;

  h2 {
    font-size: 22px;
    font-weight: 600;
    color: #303133;
    margin: 0;
  }
}

.stats-row {
  margin-bottom: 16px;

  .stat-card {
    text-align: center;

    :deep(.el-statistic__head) {
      font-size: 14px;
      color: #909399;
    }

    :deep(.el-statistic__content) {
      font-size: 28px;
      font-weight: bold;
    }
  }

  .stat-firing {
    border-top: 3px solid #f56c6c;

    :deep(.el-statistic__content) {
      color: #f56c6c;
    }
  }

  .stat-acknowledged {
    border-top: 3px solid #e6a23c;

    :deep(.el-statistic__content) {
      color: #e6a23c;
    }
  }

  .stat-resolved {
    border-top: 3px solid #67c23a;

    :deep(.el-statistic__content) {
      color: #67c23a;
    }
  }

  .stat-total {
    border-top: 3px solid #409eff;

    :deep(.el-statistic__content) {
      color: #409eff;
    }
  }
}

.toolbar-card {
  margin-bottom: 16px;

  .toolbar {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }
}

.table-card {
  margin-bottom: 16px;

  .pagination-wrapper {
    display: flex;
    justify-content: flex-end;
    padding-top: 16px;
  }

  :deep(.row-firing) {
    background-color: #fef0f0 !important;
  }

  :deep(.row-acknowledged) {
    background-color: #fdf6ec !important;
  }

  :deep(.row-resolved) {
    background-color: #f0f9eb !important;
  }

  .severity-indicator {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;

    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      display: inline-block;

      &.dot-critical {
        background-color: #f56c6c;
        box-shadow: 0 0 4px rgba(245, 108, 108, 0.5);
      }

      &.dot-warning {
        background-color: #e6a23c;
        box-shadow: 0 0 4px rgba(230, 162, 60, 0.5);
      }

      &.dot-info {
        background-color: #909399;
      }
    }
  }

  .message-text {
    display: inline-block;
    max-width: 200px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    cursor: pointer;
  }

  .value-exceeded {
    color: #f56c6c;
    font-weight: bold;
  }

  .resolved-info {
    font-size: 12px;
    color: #909399;
    text-align: left;

    .resolver {
      color: #67c23a;
    }
  }
}

.chart-card {
  .chart-title {
    font-size: 16px;
    font-weight: 600;
    color: #303133;
  }
}
</style>
