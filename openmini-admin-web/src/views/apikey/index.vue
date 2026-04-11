<script setup lang="ts">
import { ref, reactive, computed, onMounted, nextTick } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { CopyDocument, WarningFilled, Search, Plus } from '@element-plus/icons-vue'
import * as echarts from 'echarts'
import {
  getApiKeyList,
  createApiKey,
  deleteApiKey,
  toggleApiKey,
  getApiKeyUsage
} from '@/api/apikey'
import type {
  ApiKeyInfo,
  ApiKeyQueryParams,
  CreateApiKeyRequest,
  CreateApiKeyResponse,
  ApiKeyUsage
} from '@/types/api/apikey'

const loading = ref(false)
const apiKeyList = ref<ApiKeyInfo[]>([])
const total = ref(0)
const currentPage = ref(1)
const pageSize = ref(10)
const keyword = ref('')
const statusFilter = ref('')

const createDialogVisible = ref(false)
const resultDialogVisible = ref(false)
const editDialogVisible = ref(false)
const usageDrawerVisible = ref(false)

const createdKeyResult = ref<CreateApiKeyResponse | null>(null)
const currentEditKey = ref<ApiKeyInfo | null>(null)
const currentUsageData = ref<ApiKeyUsage | null>(null)
const currentUsageKeyId = ref('')

const createForm = reactive({
  name: '',
  expires_at: undefined as string | undefined,
  daily_requests: 0,
  monthly_tokens: 0
})

const editForm = reactive({
  name: '',
  expires_at: undefined as string | undefined,
  daily_requests: 0,
  monthly_tokens: 0
})

const queryParams = computed<ApiKeyQueryParams>(() => ({
  page: currentPage.value,
  page_size: pageSize.value,
  keyword: keyword.value || undefined,
  status: (statusFilter.value as any) || undefined
}))

onMounted(() => {
  fetchApiKeyList()
})

async function fetchApiKeyList() {
  loading.value = true
  try {
    const res = await getApiKeyList(queryParams.value)
    apiKeyList.value = res.items || res.data?.items || []
    total.value = res.total || res.data?.total || 0
  } catch (error) {
    ElMessage.error('获取 API Key 列表失败')
  } finally {
    loading.value = false
  }
}

function handleSearch() {
  currentPage.value = 1
  fetchApiKeyList()
}

function handleStatusChange() {
  currentPage.value = 1
  fetchApiKeyList()
}

function handleSizeChange(val: number) {
  pageSize.value = val
  currentPage.value = 1
  fetchApiKeyList()
}

function handleCurrentChange(val: number) {
  currentPage.value = val
  fetchApiKeyList()
}

function openCreateDialog() {
  createForm.name = ''
  createForm.expires_at = undefined
  createForm.daily_requests = 0
  createForm.monthly_tokens = 0
  createDialogVisible.value = true
}

async function handleCreate() {
  if (!createForm.name.trim()) {
    ElMessage.warning('请输入 Key 名称')
    return
  }

  try {
    const data: CreateApiKeyRequest = {
      name: createForm.name.trim(),
      expires_at: createForm.expires_at,
      quota_limit: {
        daily_requests: createForm.daily_requests || 0,
        monthly_requests: 0,
        daily_tokens: 0,
        monthly_tokens: createForm.monthly_tokens || 0
      }
    }
    const result = await createApiKey(data)
    createdKeyResult.value = result as CreateApiKeyResponse
    createDialogVisible.value = false
    resultDialogVisible.value = true
    ElMessage.success('API Key 创建成功')
  } catch (error) {
    ElMessage.error('创建 API Key 失败')
  }
}

function closeResultDialog() {
  resultDialogVisible.value = false
  createdKeyResult.value = null
  fetchApiKeyList()
}

async function copyToClipboard(text: string) {
  try {
    await navigator.clipboard.writeText(text)
    ElMessage.success('已复制到剪贴板')
  } catch (error) {
    ElMessage.error('复制失败，请手动复制')
  }
}

function openEditDialog(row: ApiKeyInfo) {
  currentEditKey.value = row
  editForm.name = row.name
  editForm.expires_at = row.expires_at
  editForm.daily_requests = row.quota_limit?.daily_requests || 0
  editForm.monthly_tokens = row.quota_limit?.monthly_tokens || 0
  editDialogVisible.value = true
}

async function handleEdit() {
  if (!currentEditKey.value || !editForm.name.trim()) {
    ElMessage.warning('请输入 Key 名称')
    return
  }

  try {
    await toggleApiKey(currentEditKey.value.id)
    ElMessage.success('更新成功')
    editDialogVisible.value = false
    fetchApiKeyList()
  } catch (error) {
    ElMessage.error('更新失败')
  }
}

async function handleToggleStatus(row: ApiKeyInfo) {
  try {
    await toggleApiKey(row.id)
    ElMessage.success(`已${row.status === 'active' ? '禁用' : '启用'}该 Key`)
    fetchApiKeyList()
  } catch (error) {
    ElMessage.error('操作失败')
  }
}

async function handleDelete(row: ApiKeyInfo) {
  try {
    await ElMessageBox.confirm(
      '确定要废弃该 API Key 吗？此操作不可恢复',
      '确认废弃',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    await deleteApiKey(row.id)
    ElMessage.success('已废弃该 Key')
    fetchApiKeyList()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error('删除失败')
    }
  }
}

async function openUsageDrawer(row: ApiKeyInfo) {
  currentUsageKeyId.value = row.id
  usageDrawerVisible.value = true
  try {
    const data = await getApiKeyUsage(row.id)
    currentUsageData.value = data
    nextTick(() => {
      renderUsageCharts(data)
    })
  } catch (error) {
    ElMessage.error('获取用量数据失败')
  }
}

let chartInstance: echarts.ECharts | null = null

function renderUsageCharts(data: ApiKeyUsage) {
  const chartDom = document.getElementById('usage-chart')
  if (!chartDom) return

  if (chartInstance) {
    chartInstance.dispose()
  }

  chartInstance = echarts.init(chartDom)
  const dates = data.daily_usage.map(item => item.date.slice(5))
  const requests = data.daily_usage.map(item => item.requests)
  const tokens = data.daily_usage.map(item => item.tokens / 1000)

  chartInstance.setOption({
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross' }
    },
    legend: {
      data: ['请求数', 'Token用量(K)'],
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
      data: dates,
      boundaryGap: false
    },
    yAxis: [
      {
        type: 'value',
        name: '请求数'
      },
      {
        type: 'value',
        name: 'Token(K)',
        position: 'right'
      }
    ],
    series: [
      {
        name: '请求数',
        type: 'line',
        smooth: true,
        data: requests,
        itemStyle: { color: '#409EFF' }
      },
      {
        name: 'Token用量(K)',
        type: 'line',
        smooth: true,
        yAxisIndex: 1,
        data: tokens,
        itemStyle: { color: '#67C23A' }
      }
    ]
  })
}

function formatRelative(time?: string) {
  if (!time) return '-'
  const now = new Date()
  const target = new Date(time)
  const diff = now.getTime() - target.getTime()

  const minutes = Math.floor(diff / 60000)
  const hours = Math.floor(diff / 3600000)
  const days = Math.floor(diff / 86400000)

  if (minutes < 1) return '刚刚'
  if (minutes < 60) return `${minutes}分钟前`
  if (hours < 24) return `${hours}小时前`
  if (days < 30) return `${days}天前`
  return time.slice(0, 10)
}

function getStatusType(status: string) {
  const map: Record<string, string> = {
    active: 'success',
    disabled: 'warning',
    expired: 'info',
    revoked: 'danger'
  }
  return map[status] || 'info'
}

function getStatusText(status: string) {
  const map: Record<string, string> = {
    active: '活跃',
    disabled: '已禁用',
    expired: '已过期',
    revoked: '已废弃'
  }
  return map[status] || status
}
</script>

<template>
  <div class="page-container">
    <div class="page-header">
      <h2>API Key 管理</h2>
    </div>

    <el-card class="toolbar-card" shadow="never">
      <div class="toolbar">
        <el-input
          v-model="keyword"
          placeholder="搜索 Key 名称"
          clearable
          style="width: 240px"
          @keyup.enter="handleSearch"
          @clear="handleSearch"
        >
          <template #prefix>
            <el-icon><Search /></el-icon>
          </template>
        </el-input>

        <el-select
          v-model="statusFilter"
          placeholder="状态筛选"
          clearable
          style="width: 160px"
          @change="handleStatusChange"
        >
          <el-option label="全部" value="" />
          <el-option label="活跃" value="active" />
          <el-option label="已禁用" value="disabled" />
          <el-option label="已过期" value="expired" />
        </el-select>

        <el-button type="primary" @click="openCreateDialog">
          <el-icon><Plus /></el-icon>
          创建 Key
        </el-button>
      </div>
    </el-card>

    <el-card shadow="never" class="table-card">
      <el-table
        :data="apiKeyList"
        v-loading="loading"
        stripe
        border
        style="width: 100%"
      >
        <el-table-column prop="name" label="Key 名称" min-width="120" />

        <el-table-column label="Key 前缀" min-width="180">
          <template #default="{ row }">
            <code class="key-prefix">{{ row.key_prefix }}</code>
            <el-button
              link
              type="primary"
              size="small"
              @click="copyToClipboard(row.key_prefix)"
            >
              <el-icon><CopyDocument /></el-icon>
            </el-button>
          </template>
        </el-table-column>

        <el-table-column prop="created_by" label="创建者" width="100" />

        <el-table-column prop="status" label="状态" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)" effect="dark">
              {{ getStatusText(row.status) }}
            </el-tag>
          </template>
        </el-table-column>

        <el-table-column label="配额" min-width="150">
          <template #default="{ row }">
            <div class="quota-info">
              <span>日请求: {{ row.quota_limit?.daily_requests || '∞' }}</span>
              <span>月Token: {{ row.quota_limit?.monthly_tokens || '∞' }}</span>
            </div>
          </template>
        </el-table-column>

        <el-table-column label="今日/月用量" min-width="140">
          <template #default="{ row }">
            <div class="usage-info">
              <div class="usage-item">
                <span>{{ row.usage_current.today_requests }}</span>
                <el-progress
                  :percentage="
                    row.quota_limit?.daily_requests
                      ? Math.min((row.usage_current.today_requests / row.quota_limit.daily_requests) * 100, 100)
                      : 0
                  "
                  :stroke-width="4"
                  style="flex: 1; margin-left: 8px"
                />
              </div>
              <div class="usage-item">
                <span>{{ row.usage_current.month_tokens }}</span>
                <el-progress
                  :percentage="
                    row.quota_limit?.monthly_tokens
                      ? Math.min((row.usage_current.month_tokens / row.quota_limit.monthly_tokens) * 100, 100)
                      : 0
                  "
                  :stroke-width="4"
                  style="flex: 1; margin-left: 8px"
                />
              </div>
            </div>
          </template>
        </el-table-column>

        <el-table-column label="创建时间" width="120">
          <template #default="{ row }">
            {{ formatRelative(row.created_at) }}
          </template>
        </el-table-column>

        <el-table-column label="最近使用" width="120">
          <template #default="{ row }">
            {{ row.last_used_at ? formatRelative(row.last_used_at) : '从未使用' }}
          </template>
        </el-table-column>

        <el-table-column label="操作" width="240" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="openEditDialog(row)">
              编辑
            </el-button>
            <el-button
              link
              :type="row.status === 'active' ? 'warning' : 'success'"
              size="small"
              @click="handleToggleStatus(row)"
            >
              {{ row.status === 'active' ? '禁用' : '启用' }}
            </el-button>
            <el-popconfirm
              title="确定要废弃该 Key 吗？"
              confirmButtonText="确定"
              cancelButtonText="取消"
              @confirm="handleDelete(row)"
            >
              <template #reference>
                <el-button link type="danger" size="small">废弃</el-button>
              </template>
            </el-popconfirm>
            <el-button link type="info" size="small" @click="openUsageDrawer(row)">
              查看用量
            </el-button>
          </template>
        </el-table-column>
      </el-table>

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

    <el-dialog
      v-model="createDialogVisible"
      title="创建 API Key"
      width="500px"
      :close-on-click-modal="false"
    >
      <el-form label-width="120px">
        <el-form-item label="名称" required>
          <el-input
            v-model="createForm.name"
            placeholder="请输入 Key 名称（最多50字符）"
            maxlength="50"
            show-word-limit
          />
        </el-form-item>

        <el-form-item label="过期时间">
          <el-date-picker
            v-model="createForm.expires_at"
            type="datetime"
            placeholder="不设置则永不过期"
            format="YYYY-MM-DD HH:mm:ss"
            value-format="YYYY-MM-DDTHH:mm:ssZ"
            style="width: 100%"
          />
        </el-form-item>

        <el-form-item label="每日请求配额">
          <el-input-number
            v-model="createForm.daily_requests"
            :min="0"
            :step="100"
            controls-position="right"
            style="width: 100%"
          />
          <div class="form-tip">0 表示无限制</div>
        </el-form-item>

        <el-form-item label="每月 Token 配额">
          <el-input-number
            v-model="createForm.monthly_tokens"
            :min="0"
            :step="10000"
            controls-position="right"
            style="width: 100%"
          />
          <div class="form-tip">0 表示无限制</div>
        </el-form-item>
      </el-form>

      <template #footer>
        <el-button @click="createDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="handleCreate">创建</el-button>
      </template>
    </el-dialog>

    <el-dialog
      v-model="resultDialogVisible"
      title="🎉 API Key 创建成功"
      width="560px"
      :close-on-click-modal="false"
      :show-close="false"
      :close-on-press-escape="false"
    >
      <div class="result-content">
        <el-alert
          type="error"
          :closable="false"
          show-icon
          style="margin-bottom: 20px"
        >
          <template #title>
            <WarningFilled style="margin-right: 4px" />
            请立即复制完整密钥，关闭后将无法再次查看！
          </template>
        </el-alert>

        <div class="key-display">
          <code class="full-key">{{ createdKeyResult?.key }}</code>
          <el-button
            type="primary"
            @click="copyToClipboard(createdKeyResult?.key || '')"
            style="margin-top: 12px"
          >
            <el-icon><CopyDocument /></el-icon>
            复制完整密钥
          </el-button>
        </div>

        <el-descriptions :column="1" border style="margin-top: 20px">
          <el-descriptions-item label="Key 前缀">
            {{ createdKeyResult?.key_prefix }}
          </el-descriptions-item>
          <el-descriptions-item label="名称">
            {{ createdKeyResult?.name }}
          </el-descriptions-item>
          <el-descriptions-item label="创建时间">
            {{ createdKeyResult?.created_at }}
          </el-descriptions-item>
        </el-descriptions>
      </div>

      <template #footer>
        <el-button type="primary" @click="closeResultDialog">我已复制，关闭</el-button>
      </template>
    </el-dialog>

    <el-dialog
      v-model="editDialogVisible"
      title="编辑 API Key"
      width="500px"
      :close-on-click-modal="false"
    >
      <el-form label-width="120px">
        <el-form-item label="名称" required>
          <el-input
            v-model="editForm.name"
            placeholder="请输入 Key 名称"
            maxlength="50"
            show-word-limit
          />
        </el-form-item>

        <el-form-item label="过期时间">
          <el-date-picker
            v-model="editForm.expires_at"
            type="datetime"
            placeholder="选择过期时间"
            format="YYYY-MM-DD HH:mm:ss"
            value-format="YYYY-MM-DDTHH:mm:ssZ"
            style="width: 100%"
          />
        </el-form-item>

        <el-form-item label="每日请求配额">
          <el-input-number
            v-model="editForm.daily_requests"
            :min="0"
            :step="100"
            controls-position="right"
            style="width: 100%"
          />
        </el-form-item>

        <el-form-item label="每月 Token 配额">
          <el-input-number
            v-model="editForm.monthly_tokens"
            :min="0"
            :step="10000"
            controls-position="right"
            style="width: 100%"
          />
        </el-form-item>
      </el-form>

      <template #footer>
        <el-button @click="editDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="handleEdit">保存</el-button>
      </template>
    </el-dialog>

    <el-drawer
      v-model="usageDrawerVisible"
      title="用量详情"
      direction="rtl"
      size="600px"
    >
      <div v-if="currentUsageData" class="usage-detail">
        <el-row :gutter="16" class="stats-row">
          <el-col :span="12">
            <el-statistic title="今日请求数" :value="currentUsageData.daily_usage[currentUsageData.daily_usage.length - 1]?.requests || 0">
              <template #suffix>
                <span class="quota-text"> / {{ currentEditKey?.quota_limit?.daily_requests || '∞' }}</span>
              </template>
            </el-statistic>
          </el-col>
          <el-col :span="12">
            <el-statistic title="本月 Token 用量" :value="currentUsageData.monthly_summary.total_tokens">
              <template #suffix>
                <span class="quota-text"> / {{ currentEditKey?.quota_limit?.monthly_tokens || '∞' }}</span>
              </template>
            </el-statistic>
          </el-col>
        </el-row>

        <div class="chart-section">
          <h4>近 7 天用量趋势</h4>
          <div id="usage-chart" style="width: 100%; height: 350px"></div>
        </div>

        <div class="history-section">
          <h4>热门接口调用统计</h4>
          <el-table :data="currentUsageData.top_endpoints" size="small" border>
            <el-table-column prop="endpoint" label="接口路径" />
            <el-table-column prop="method" label="方法" width="80" />
            <el-table-column prop="request_count" label="调用次数" width="100" />
            <el-table-column prop="avg_latency_ms" label="平均延迟(ms)" width="120" />
          </el-table>
        </div>
      </div>
    </el-drawer>
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

.toolbar-card {
  margin-bottom: 16px;

  .toolbar {
    display: flex;
    align-items: center;
    gap: 12px;
  }
}

.table-card {
  .pagination-wrapper {
    display: flex;
    justify-content: flex-end;
    padding-top: 16px;
  }
}

.key-prefix {
  font-family: 'Courier New', monospace;
  background: #f5f7fa;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 13px;
  color: #606266;
}

.quota-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
  color: #909399;
}

.usage-info {
  .usage-item {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
    font-size: 12px;

    span {
      min-width: 40px;
      text-align: right;
    }
  }
}

.form-tip {
  font-size: 12px;
  color: #909399;
  margin-top: 4px;
}

.result-content {
  .key-display {
    text-align: center;
    padding: 20px;
    background: #fafafa;
    border-radius: 8px;
  }

  .full-key {
    display: block;
    font-family: 'Courier New', monospace;
    font-size: 18px;
    font-weight: bold;
    color: #f56c6c;
    word-break: break-all;
    line-height: 1.6;
    padding: 12px;
    background: #fff;
    border: 2px dashed #f56c6c;
    border-radius: 4px;
  }
}

.usage-detail {
  .stats-row {
    margin-bottom: 24px;
  }

  .quota-text {
    font-size: 14px;
    color: #909399;
    margin-left: 4px;
  }

  .chart-section {
    margin-bottom: 24px;

    h4 {
      font-size: 16px;
      font-weight: 600;
      margin-bottom: 12px;
      color: #303133;
    }
  }

  .history-section {
    h4 {
      font-size: 16px;
      font-weight: 600;
      margin-bottom: 12px;
      color: #303133;
    }
  }
}
</style>
