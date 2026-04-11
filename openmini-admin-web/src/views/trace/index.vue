<script setup lang="ts">
import { ref, reactive } from 'vue'
import { Search, Filter, View } from '@element-plus/icons-vue'

const loading = ref(false)
const showAdvancedFilter = ref(false)

const filters = reactive({
  timeRange: null as [Date, Date] | null,
  interfaceType: '',
  statusCode: '',
  model: '',
  minLatency: null as number | null,
  requestId: ''
})

const requestList = ref([
  {
    id: 'req_a1b2c3d4e5f6',
    timestamp: '2026-04-09 14:32:15.234',
    interfaceType: 'chat/completions',
    statusCode: 200,
    latency: 123,
    promptTokens: 156,
    completionTokens: 428,
    totalTokens: 584,
    model: 'Qwen-14B-Chat'
  },
  {
    id: 'req_b2c3d4e5f6g7',
    timestamp: '2026-04-09 14:32:14.892',
    interfaceType: 'embeddings',
    statusCode: 200,
    latency: 45,
    promptTokens: 2048,
    completionTokens: 0,
    totalTokens: 2048,
    model: 'Llama-3-8B-Instruct'
  },
  {
    id: 'req_c3d4e5f6g7h8',
    timestamp: '2026-04-09 14:32:13.456',
    interfaceType: 'chat/completions',
    statusCode: 429,
    latency: 1523,
    promptTokens: 2048,
    completionTokens: 0,
    totalTokens: 2048,
    model: 'Qwen-14B-Chat'
  },
  {
    id: 'req_d4e5f6g7h8i9',
    timestamp: '2026-04-09 14:32:12.111',
    interfaceType: 'models/list',
    statusCode: 200,
    latency: 12,
    promptTokens: 0,
    completionTokens: 0,
    totalTokens: 0,
    model: '-'
  },
  {
    id: 'req_e5f6g7h8i9j0',
    timestamp: '2026-04-09 14:32:11.789',
    interfaceType: 'chat/completions',
    statusCode: 500,
    latency: 2345,
    promptTokens: 4096,
    completionTokens: 0,
    totalTokens: 4096,
    model: 'Yi-34B-Chat'
  }
])

const detailDrawer = ref(false)
const currentRequest = ref<any>(null)

const slowRequests = ref([
  { id: 'req_c3d4e5f6g7h8', latency: 1523, interfaceType: 'chat/completions', model: 'Qwen-14B-Chat', time: '14:32:14' },
  { id: 'req_e5f6g7h8i9j0', latency: 2345, interfaceType: 'chat/completions', model: 'Yi-34B-Chat', time: '14:32:11' },
  { id: 'req_f6g7h8i9j0k1', latency: 1876, interfaceType: 'chat/completions', model: 'Qwen-14B-Chat', time: '14:31:58' },
  { id: 'req_g7h8i9j0k1l2', latency: 1654, interfaceType: 'embeddings', model: 'Llama-3-8B-Instruct', time: '14:31:45' },
  { id: 'req_h8i9j0k1l2m3', latency: 1432, interfaceType: 'chat/completions', model: 'Baichuan2-7B-Chat', time: '14:31:22' }
])

const errorStats = ref([
  { code: 500, count: 12, desc: '服务器内部错误' },
  { code: 429, count: 45, desc: '请求频率超限' },
  { code: 401, count: 8, desc: '认证失败' },
  { code: 404, count: 3, desc: '接口不存在' }
])

const errorPieData = ref(errorStats.value.map(e => ({ name: `${e.code}`, value: e.count })))

function getStatusType(code: number) {
  if (code >= 200 && code < 300) return 'success'
  if (code >= 400 && code < 500) return 'warning'
  if (code >= 500) return 'danger'
  return 'info'
}

function getLatencyClass(latency: number) {
  if (latency <= 100) return ''
  if (latency <= 500) return 'latency-warning'
  return 'latency-danger'
}

function handleDetail(row: any) {
  currentRequest.value = row
  detailDrawer.value = true
}

function truncateId(id: string) {
  return id.slice(0, 12) + '...'
}

function getLatencyColor(latency: number) {
  if (latency <= 200) return '#67C23A'
  if (latency <= 1000) return '#E6A23C'
  return '#F56C6C'
}
</script>

<template>
  <div class="trace-container">
    <!-- 高级筛选工具栏 -->
    <el-card shadow="hover" class="filter-card">
      <div class="filter-header" @click="showAdvancedFilter = !showAdvancedFilter">
        <span>高级筛选</span>
        <el-icon :class="{ rotated: showAdvancedFilter }"><Filter /></el-icon>
      </div>
      <el-collapse-transition>
        <div v-show="showAdvancedFilter" class="filter-body">
          <el-row :gutter="16">
            <el-col :span="8">
              <el-date-picker v-model="filters.timeRange" type="datetimerange" range-separator="至" start-placeholder="开始" end-placeholder="结束" style="width: 100%" />
            </el-col>
            <el-col :span="4">
              <el-select v-model="filters.interfaceType" placeholder="接口类型" clearable style="width: 100%">
                <el-option label="Chat Completions" value="chat/completions" />
                <el-option label="Embeddings" value="embeddings" />
                <el-option label="Models" value="models" />
              </el-select>
            </el-col>
            <el-col :span="3">
              <el-select v-model="filters.statusCode" placeholder="状态码" clearable style="width: 100%">
                <el-option label="200" value="200" />
                <el-option label="400" value="400" />
                <el-option label="401" value="401" />
                <el-option label="429" value="429" />
                <el-option label="500" value="500" />
              </el-select>
            </el-col>
            <el-col :span="4">
              <el-input v-model="filters.model" placeholder="模型筛选" clearable />
            </el-col>
            <el-col :span="3">
              <el-input-number v-model="filters.minLatency" placeholder="最小延迟(ms)" :min="0" style="width: 100%" />
            </el-col>
            <el-col :span="4">
              <el-input v-model="filters.requestId" placeholder="请求ID搜索" clearable />
            </el-col>
          </el-row>
          <div class="filter-actions">
            <el-button type="primary" :icon="Search">搜索</el-button>
            <el-button>重置</el-button>
          </div>
        </div>
      </el-collapse-transition>
    </el-card>

    <!-- 请求列表 -->
    <el-card shadow="hover">
      <el-table :data="requestList" stripe v-loading="loading" default-sort="{ prop: 'timestamp', order: 'descending' }">
        <el-table-column label="ID" width="140" fixed>
          <template #default="{ row }">
            <code class="req-id">{{ truncateId(row.id) }}</code>
          </template>
        </el-table-column>
        <el-table-column prop="timestamp" label="时间" width="180" sortable />
        <el-table-column prop="interfaceType" label="接口" width="150">
          <template #default="{ row }">
            <el-tag size="small">{{ row.interfaceType }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="statusCode" label="状态码" width="90">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.statusCode)" size="small">{{ row.statusCode }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="latency" label="延迟(ms)" width="110" sortable>
          <template #default="{ row }">
            <span :class="getLatencyClass(row.latency)" :style="{ color: getLatencyColor(row.latency), fontWeight: row.latency > 1000 ? 'bold' : 'normal' }">
              {{ row.latency }}
            </span>
          </template>
        </el-table-column>
        <el-table-column prop="promptTokens" label="Prompt" width="90" />
        <el-table-column prop="completionTokens" label="Completion" width="100" />
        <el-table-column prop="totalTokens" label="Total" width="80" />
        <el-table-column prop="model" label="模型" min-width="160" />
        <el-table-column label="操作" width="80" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" link size="small" :icon="View" @click="handleDetail(row)">详情</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 请求详情抽屉 -->
    <el-drawer v-model="detailDrawer" title="请求详情" size="700px">
      <template v-if="currentRequest">
        <!-- 基本信息 -->
        <el-descriptions title="基本信息" :column="2" border class="detail-section">
          <el-descriptions-item label="请求ID">
            <code>{{ currentRequest.id }}</code>
          </el-descriptions-item>
          <el-descriptions-item label="时间">{{ currentRequest.timestamp }}</el-descriptions-item>
          <el-descriptions-item label="接口类型">
            <el-tag size="small">{{ currentRequest.interfaceType }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="状态码">
            <el-tag :type="getStatusType(currentRequest.statusCode)" size="small">{{ currentRequest.statusCode }}</el-tag>
          </el-descriptions-item>
        </el-descriptions>

        <!-- Token统计卡片 -->
        <el-row :gutter="16" class="token-stats">
          <el-col :span="8">
            <StatCard title="Prompt Tokens" :value="currentRequest.promptTokens" color="default" />
          </el-col>
          <el-col :span="8">
            <StatCard title="Completion Tokens" :value="currentRequest.completionTokens" color="success" />
          </el-col>
          <el-col :span="8">
            <StatCard title="Total Tokens" :value="currentRequest.totalTokens" color="warning" />
          </el-col>
        </el-row>

        <!-- 耗时分解时间线 -->
        <div class="timeline-section">
          <h4>耗时分解</h4>
          <BarChart
            :data="{
              xAxis: ['排队', '加载', '推理', '编码', '序列化'],
              series: [{ name: '耗时(ms)', data: [5, 23, 78, 12, 5], color: '#409EFF' }]
            }"
            :height="220"
            :horizontal="true"
            :show-legend="false"
          />
        </div>

        <!-- 请求/响应内容 -->
        <el-collapse>
          <el-collapse-item title="请求内容 (JSON)" name="request">
            <pre class="json-block">{ "model": "{{ currentRequest.model }}", "messages": [...], "max_tokens": 2048, "temperature": 0.7 }</pre>
          </el-collapse-item>
          <el-collapse-item title="响应内容 (JSON)" name="response">
            <pre class="json-block">{ "id": "chatcmpl-xxx", "object": "chat.completion", "choices": [...], "usage": { "prompt_tokens": {{ currentRequest.promptTokens }}, "completion_tokens": {{ currentRequest.completionTokens }}}}</pre>
          </el-collapse-item>
        </el-collapse>
      </template>
    </el-drawer>

    <!-- 慢请求Top20 -->
    <el-row :gutter="20" class="bottom-section">
      <el-col :span="14">
        <el-card shadow="hover">
          <template #header><span>慢请求 Top20</span></template>
          <el-table :data="slowRequests" stripe size="small">
            <el-table-column label="排名" width="60" type="index" />
            <el-table-column label="延迟(ms)" width="100" sortable>
              <template #default="{ row }">
                <span :style="{ color: getLatencyColor(row.latency), fontWeight: 'bold' }">{{ row.latency }}</span>
              </template>
            </el-table-column>
            <el-table-column prop="interfaceType" label="接口" />
            <el-table-column prop="model" label="模型" />
            <el-table-column prop="time" label="时间" width="100" />
          </el-table>
        </el-card>
      </el-col>

      <el-col :span="10">
        <el-card shadow="hover">
          <template #header><span>错误请求分析</span></template>
          <PieChart :data="errorPieData" :height="240" :title="''" />
          <el-divider />
          <el-table :data="errorStats" size="small">
            <el-table-column prop="code" label="状态码" width="80">
              <template #default="{ row }">
                <el-tag type="danger" size="small">{{ row.code }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="count" label="次数" width="70" />
            <el-table-column prop="desc" label="描述" />
          </el-table>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style lang="scss" scoped>
.trace-container {
  .filter-card {
    margin-bottom: $spacing-lg;

    .filter-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      cursor: pointer;
      font-weight: 600;

      .el-icon {
        transition: transform $transition-fast;

        &.rotated {
          transform: rotate(180deg);
        }
      }
    }

    .filter-body {
      margin-top: $spacing-md;

      .filter-actions {
        margin-top: $spacing-md;
        display: flex;
        gap: $spacing-sm;
      }
    }
  }

  .bottom-section {
    margin-top: $spacing-lg;
  }

  .req-id {
    font-family: $font-mono;
    font-size: 12px;
    background: #f0f2f5;
    padding: 2px 6px;
    border-radius: 3px;
  }

  .latency-warning {
    color: #E6A23C;
  }

  .latency-danger {
    color: #F56C6C;
    font-weight: bold;
  }

  .detail-section {
    margin-bottom: $spacing-lg;
  }

  .token-stats {
    margin-bottom: $spacing-lg;
  }

  .timeline-section {
    margin-bottom: $spacing-lg;

    h4 {
      margin-bottom: $spacing-md;
      color: $text-primary;
    }
  }

  .json-block {
    margin: 0;
    padding: $spacing-md;
    background: #1e1e1e;
    color: #d4d4d4;
    border-radius: $border-radius-sm;
    font-family: $font-mono;
    font-size: 12px;
    line-height: 1.6;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-all;
  }
}
</style>
