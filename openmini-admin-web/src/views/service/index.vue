<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Refresh, VideoPause } from '@element-plus/icons-vue'
import { getServiceStatus, getWorkerList, restartService, stopService } from '@/api/service'

const loading = ref(false)

const serviceInfo = ref<Record<string, any>>({})

const workerList = ref<any[]>([])

const serviceStatus = ref('')

const confirmVisible = ref(false)
const actionType = ref('')

function getStatusType(status: string) {
  const map: Record<string, string> = {
    online: 'success',
    offline: 'danger',
    starting: 'warning',
    stopping: 'info',
    running: 'success',
    stopped: 'danger'
  }
  return map[status] || 'info'
}

function getStatusText(status: string) {
  const map: Record<string, string> = {
    online: '运行中',
    offline: '离线',
    starting: '启动中',
    stopping: '停止中',
    running: '运行中',
    stopped: '已停止'
  }
  return map[status] || status
}

async function fetchData() {
  loading.value = true
  try {
    const [statusRes, workersRes] = await Promise.all([
      getServiceStatus(),
      getWorkerList()
    ])
    if (statusRes.data) {
      const data = statusRes.data
      serviceInfo.value = {
        version: data.version || '-',
        listenAddress: data.upstream || data.listen_address || '-',
        uptime: data.uptime || '-',
        pid: data.pid || '-',
        connections: data.connections || 0
      }
      serviceStatus.value = data.status || ''
    }
    if (workersRes.data && workersRes.data.workers) {
      workerList.value = workersRes.data.workers.map((w: any, index: number) => ({
        id: w.id || index + 1,
        status: w.status || 'offline',
        pid: w.pid || 0,
        startTime: w.start_time || w.startTime || '-',
        restartCount: w.restart_count || w.restartCount || 0,
        cpuUsage: w.cpu_usage || w.cpuUsage || 0,
        memoryUsage: w.memory_usage || w.memoryUsage || 0
      }))
    }
  } catch (error) {
    ElMessage.error('获取服务状态失败')
  } finally {
    loading.value = false
  }
}

async function handleRestartWorker(row: any) {
  try {
    await restartService()
    ElMessage.success(`Worker-${row.id} 重启成功`)
    await fetchData()
  } catch (error) {
    ElMessage.error('重启失败')
  }
}

function showConfirm(action: string) {
  actionType.value = action
  confirmVisible.value = true
}

async function handleConfirm() {
  try {
    if (actionType.value === 'restart') {
      await restartService()
      ElMessage.success('服务优雅重启成功')
    } else if (actionType.value === 'stop') {
      await stopService()
      ElMessage.warning('服务已停止')
    }
    confirmVisible.value = false
    await fetchData()
  } catch (error) {
    ElMessage.error(actionType.value === 'restart' ? '重启失败' : '停止失败')
  }
}

onMounted(() => {
  fetchData()
})
</script>

<template>
  <div class="service-container">
    <!-- 服务状态总览 -->
    <el-card shadow="hover" class="info-card">
      <template #header>
        <div class="card-header">
          <span>服务状态总览</span>
          <StatusBadge status="success" text="运行正常" />
        </div>
      </template>
      <el-descriptions :column="5" border>
        <el-descriptions-item label="版本号">{{ serviceInfo.version }}</el-descriptions-item>
        <el-descriptions-item label="监听地址">{{ serviceInfo.listenAddress }}</el-descriptions-item>
        <el-descriptions-item label="运行时长">{{ serviceInfo.uptime }}</el-descriptions-item>
        <el-descriptions-item label="PID">{{ serviceInfo.pid }}</el-descriptions-item>
        <el-descriptions-item label="连接数">{{ serviceInfo.connections }}</el-descriptions-item>
      </el-descriptions>
    </el-card>

    <!-- Worker进程列表 -->
    <el-card shadow="hover" class="worker-card">
      <template #header>
        <span>Worker 进程列表</span>
      </template>
      <el-table :data="workerList" stripe v-loading="loading">
        <el-table-column prop="id" label="ID" width="60" />
        <el-table-column prop="status" label="状态" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)" size="small">
              {{ getStatusText(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="pid" label="PID" width="90" />
        <el-table-column prop="startTime" label="启动时间" />
        <el-table-column prop="restartCount" label="重启次数" width="100">
          <template #default="{ row }">
            <el-tag v-if="row.restartCount > 0" type="warning" size="small">
              {{ row.restartCount }}次
            </el-tag>
            <span v-else>-</span>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="120" fixed="right">
          <template #default="{ row }">
            <el-button
              type="primary"
              link
              size="small"
              :disabled="row.status !== 'online'"
              @click="handleRestartWorker(row)"
            >
              重启
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 服务控制区 -->
    <el-card shadow="hover" class="control-card">
      <template #header>
        <span>服务控制</span>
      </template>
      <div class="control-buttons">
        <el-popconfirm
          title="确定要优雅重启服务吗？当前连接将被逐步关闭"
          @confirm="showConfirm('restart')"
        >
          <template #reference>
            <el-button type="warning" :icon="Refresh">
              优雅重启
            </el-button>
          </template>
        </el-popconfirm>

        <el-popconfirm
          title="确定要停止服务吗？此操作将中断所有正在处理的请求！"
          @confirm="showConfirm('stop')"
        >
          <template #reference>
            <el-button type="danger" :icon="VideoPause">
              停止服务
            </el-button>
          </template>
        </el-popconfirm>
      </div>
    </el-card>

    <ConfirmDialog
      v-model:visible="confirmVisible"
      :title="actionType === 'restart' ? '确认重启' : '确认停止'"
      :content="actionType === 'restart' ? '服务将优雅重启，现有连接将逐步关闭' : '警告：此操作将立即停止所有服务，可能导致数据丢失'"
      :type="actionType === 'stop' ? 'danger' : 'warning'"
      @confirm="handleConfirm"
    />
  </div>
</template>

<style lang="scss" scoped>
.service-container {
  .info-card,
  .worker-card,
  .control-card {
    margin-bottom: $spacing-lg;
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .control-buttons {
    display: flex;
    gap: $spacing-md;
    padding: $spacing-md 0;
  }
}
</style>
