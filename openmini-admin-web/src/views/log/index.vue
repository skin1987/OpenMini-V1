<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Download, VideoPause, VideoPlay, Delete, Lock } from '@element-plus/icons-vue'

const loading = ref(false)
const keyword = ref('')
const timeRange = ref<[Date, Date] | null>(null)
const levels = ref<string[]>([])
const currentPage = ref(1)
const pageSize = ref(20)

const logList = ref([
  {
    id: 1,
    timestamp: '2026-04-09 14:32:15.234',
    level: 'INFO',
    source: 'http-handler',
    message: 'POST /v1/chat/completions 请求处理完成，耗时123ms',
    raw: '{"request_id":"req_abc123","method":"POST","path":"/v1/chat/completions","status":200,"duration_ms":123}'
  },
  {
    id: 2,
    timestamp: '2026-04-09 14:32:14.892',
    level: 'WARN',
    source: 'memory-manager',
    message: '内存使用率达到75%，建议关注',
    raw: '{"component":"memory","usage_percent":75,"threshold":80}'
  },
  {
    id: 3,
    timestamp: '2026-04-09 14:32:13.456',
    level: 'ERROR',
    source: 'model-loader',
    message: '模型加载失败：文件不存在 /models/invalid-model',
    raw: '{"error":"file_not_found","path":"/models/invalid-model"}'
  },
  {
    id: 4,
    timestamp: '2026-04-09 14:32:12.111',
    level: 'INFO',
    source: 'scheduler',
    message: '新连接建立，当前活跃连接数: 42',
    raw: '{"action":"connect","active_connections":42}'
  }
])

const realtimeLogs = ref<any[]>([])
const isStreaming = ref(true)
const isPaused = ref(false)
let streamTimer: number | null = null

function getLevelType(level: string) {
  const map: Record<string, string> = {
    INFO: 'info',
    WARN: 'warning',
    ERROR: 'danger',
    DEBUG: ''
  }
  return map[level] || ''
}

function getLevelColor(level: string) {
  const map: Record<string, string> = {
    INFO: '#909399',
    WARN: '#E6A23C',
    ERROR: '#F56C6C',
    DEBUG: '#409EFF'
  }
  return map[level] || ''
}

function handleSearch() {
  loading.value = true
  setTimeout(() => {
    loading.value = false
  }, 500)
}

function handleExport(format: string) {
  ElMessage.success(`导出${format.toUpperCase()}格式成功`)
}

function toggleStream() {
  isPaused.value = !isPaused.value
}

function clearLogs() {
  realtimeLogs.value = []
}

function generateRealtimeLog() {
  if (isPaused.value) return

  const levels = ['INFO', 'WARN', 'ERROR']
  const sources = ['http-handler', 'scheduler', 'memory-manager', 'gpu-monitor']
  const messages = [
    '请求处理完成，耗时98ms',
    'GPU温度: 72°C',
    '队列长度: 5',
    '批处理大小调整: 8→12',
    'Token生成速率: 124 tok/s'
  ]

  const newLog = {
    id: Date.now(),
    timestamp: new Date().toISOString().replace('T', ' ').slice(0, 23),
    level: levels[Math.floor(Math.random() * levels.length)],
    source: sources[Math.floor(Math.random() * sources.length)],
    message: messages[Math.floor(Math.random() * messages.length)]
  }

  realtimeLogs.value.unshift(newLog)
  if (realtimeLogs.value.length > 100) {
    realtimeLogs.value.pop()
  }
}

onMounted(() => {
  streamTimer = window.setInterval(generateRealtimeLog, 2000)
})

onUnmounted(() => {
  if (streamTimer) {
    clearInterval(streamTimer)
  }
})
</script>

<template>
  <div class="log-container">
    <!-- 筛选工具栏 -->
    <el-card shadow="hover" class="toolbar-card">
      <el-row :gutter="16" align="middle">
        <el-col :span="8">
          <el-date-picker
            v-model="timeRange"
            type="datetimerange"
            range-separator="至"
            start-placeholder="开始时间"
            end-placeholder="结束时间"
            style="width: 100%"
          />
        </el-col>
        <el-col :span="4">
          <el-select v-model="levels" multiple collapse-tags placeholder="日志级别" style="width: 100%">
            <el-option label="INFO" value="INFO" />
            <el-option label="WARN" value="WARN" />
            <el-option label="ERROR" value="ERROR" />
          </el-select>
        </el-col>
        <el-col :span="6">
          <el-input v-model="keyword" placeholder="关键词搜索..." clearable />
        </el-col>
        <el-col :span="3">
          <el-button type="primary" @click="handleSearch">查询</el-button>
        </el-col>
        <el-col :span="3">
          <el-dropdown @command="handleExport">
            <el-button :icon="Download">导出</el-button>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item command="csv">CSV格式</el-dropdown-item>
                <el-dropdown-item command="json">JSON格式</el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </el-col>
      </el-row>
    </el-card>

    <!-- 日志列表 -->
    <el-card shadow="hover">
      <el-table :data="logList" stripe v-loading="loading">
        <el-table-column prop="timestamp" label="时间戳" width="200" fixed />
        <el-table-column prop="level" label="级别" width="90">
          <template #default="{ row }">
            <el-tag :type="getLevelType(row.level)" size="small">
              {{ row.level }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="source" label="来源模块" width="140" />
        <el-table-column prop="message" label="消息内容" min-width="300" show-overflow-tooltip>
          <template #default="{ row }">
            <el-tooltip :content="row.message" placement="top">
              <span class="log-message">{{ row.message }}</span>
            </el-tooltip>
          </template>
        </el-table-column>
        <el-table-column type="expand">
          <template #default="{ row }">
            <pre class="log-raw">{{ row.raw }}</pre>
          </template>
        </el-table-column>
      </el-table>

      <el-pagination
        v-model:current-page="currentPage"
        v-model:page-size="pageSize"
        :total="1000"
        layout="total, sizes, prev, pager, next, jumper"
        style="margin-top: 16px; justify-content: flex-end"
      />
    </el-card>

    <!-- 实时日志流 -->
    <el-card shadow="hover" class="realtime-card">
      <template #header>
        <div class="realtime-header">
          <div class="header-left">
            <span>实时日志流</span>
            <StatusBadge :status="isStreaming ? 'success' : 'danger'" :text="isStreaming ? '已连接' : '已断开'" />
          </div>
          <div class="header-actions">
            <el-button :type="isPaused ? 'success' : 'warning'" size="small" @click="toggleStream">
              {{ isPaused ? '恢复' : '暂停' }}
            </el-button>
            <el-button size="small" @click="clearLogs">清屏</el-button>
          </div>
        </div>
      </template>
      <div class="log-stream" ref="logContainer">
        <div
          v-for="log in realtimeLogs"
          :key="log.id"
          class="log-line"
          :style="{ color: getLevelColor(log.level) }"
        >
          <span class="log-time">{{ log.timestamp }}</span>
          <span class="log-level">[{{ log.level }}]</span>
          <span class="log-source">[{{ log.source }}]</span>
          <span class="log-msg">{{ log.message }}</span>
        </div>
        <div v-if="!realtimeLogs.length" class="empty-log">
          暂无实时日志...
        </div>
      </div>
    </el-card>
  </div>
</template>

<style lang="scss" scoped>
.log-container {
  .toolbar-card {
    margin-bottom: $spacing-lg;
  }

  .realtime-card {
    margin-top: $spacing-lg;

    .realtime-header {
      display: flex;
      justify-content: space-between;
      align-items: center;

      .header-left {
        display: flex;
        align-items: center;
        gap: $spacing-sm;
      }

      .header-actions {
        display: flex;
        gap: $spacing-sm;
      }
    }

    .log-stream {
      background-color: #1e1e1e;
      border-radius: $border-radius-md;
      padding: $spacing-md;
      height: 350px;
      overflow-y: auto;
      font-family: $font-mono;
      font-size: 12px;
      line-height: 1.6;

      .log-line {
        padding: 2px 0;

        .log-time {
          color: #6a9955;
        }

        .log-level {
          font-weight: bold;
        }

        .log-source {
          color: #9cdcfe;
        }

        .log-msg {
          color: #d4d4d4;
        }
      }

      .empty-log {
        text-align: center;
        color: #666;
        padding: $spacing-xl 0;
      }

      &::-webkit-scrollbar {
        width: 8px;
        height: 8px;
      }

      &::-webkit-scrollbar-thumb {
        background: #444;
        border-radius: 4px;
      }
    }
  }

  .log-message {
    cursor: pointer;
  }

  .log-raw {
    margin: 0;
    padding: $spacing-md;
    background: #f5f7fa;
    border-radius: $border-radius-sm;
    font-family: $font-mono;
    font-size: 12px;
    max-height: 300px;
    overflow: auto;
  }
}
</style>
