<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Edit, RefreshRight, Check, Close } from '@element-plus/icons-vue'

const isEditMode = ref(false)
const activeTab = ref('server')
const hotReloadVisible = ref(false)

const serverForm = reactive({
  host: '0.0.0.0',
  port: 8080,
  max_connections: 10000,
  request_timeout_ms: 30000
})

const threadPoolForm = reactive({
  size: 8,
  stack_size_kb: 512
})

const memoryForm = reactive({
  max_memory_gb: 64,
  model_memory_gb: 48,
  cache_memory_gb: 8,
  arena_size_mb: 256
})

const modelForm = reactive({
  path: '/models',
  quantization: 'INT8',
  context_length: 4096
})

const workerForm = reactive({
  count: 4,
  restart_on_failure: true,
  health_check_interval_ms: 5000
})

const grpcForm = reactive({
  max_message_size_mb: 16,
  keepalive_time_ms: 30000,
  keepalive_timeout_ms: 10000
})

const historyList = ref([
  { operator: 'admin', time: '2026-04-09 10:30:00', group: 'server', summary: '修改端口为8080' },
  { operator: 'admin', time: '2026-04-08 15:20:00', group: 'memory', summary: '增加模型内存至48GB' },
  { operator: 'operator', time: '2026-04-07 09:15:00', group: 'worker', summary: 'Worker数量调整为4' }
])

const memoryChartData = [
  { name: '模型内存', value: 48 },
  { name: '缓存内存', value: 8 },
  { name: '系统预留', value: 8 }
]

function handleSave() {
  ElMessage.success('配置保存成功')
  isEditMode.value = false
}

function handleCancel() {
  isEditMode.value = false
}

function handleHotReload() {
  hotReloadVisible.value = false
  ElMessage.success('配置热重载成功')
}
</script>

<template>
  <div class="config-container">
    <PageHeader title="配置管理" subtitle="管理系统运行参数和配置项">
      <template #extra>
        <el-switch
          v-model="isEditMode"
          active-text="编辑模式"
          inactive-text="查看模式"
          style="margin-right: 12px"
        />
        <el-popconfirm
          title="确定要热重载配置吗？部分配置需要重启服务生效"
          @confirm="handleHotReload"
        >
          <template #reference>
            <el-button type="warning" :icon="RefreshRight">热重载</el-button>
          </template>
        </el-popconfirm>
      </template>
    </PageHeader>

    <!-- 配置表单 -->
    <el-card shadow="hover">
      <el-tabs v-model="activeTab" tab-position="left" style="min-height: 400px">
        <el-tab-pane label="Server" name="server">
          <el-form :disabled="!isEditMode" label-width="160px">
            <el-form-item label="监听地址 (Host)">
              <el-input v-model="serverForm.host" :class="{ 'edit-highlight': isEditMode }" />
            </el-form-item>
            <el-form-item label="监听端口 (Port)">
              <el-input-number v-model="serverForm.port" :min="1" :max="65535" :class="{ 'edit-highlight': isEditMode }" />
            </el-form-item>
            <el-form-item label="最大连接数">
              <el-input-number v-model="serverForm.max_connections" :min="100" :step="100" :class="{ 'edit-highlight': isEditMode }" />
            </el-form-item>
            <el-form-item label="请求超时时间 (ms)">
              <el-input-number v-model="serverForm.request_timeout_ms" :min="1000" :step="1000" :class="{ 'edit-highlight': isEditMode }" />
            </el-form-item>
          </el-form>
        </el-tab-pane>

        <el-tab-pane label="线程池" name="thread_pool">
          <el-form :disabled="!isEditMode" label-width="160px">
            <el-form-item label="线程池大小">
              <el-input-number v-model="threadPoolForm.size" :min="1" :max="64" :class="{ 'edit-highlight': isEditMode }" />
            </el-form-item>
            <el-form-item label="栈大小 (KB)">
              <el-input-number v-model="threadPoolForm.stack_size_kb" :min="128" :step="64" :class="{ 'edit-highlight': isEditMode }" />
            </el-form-item>
          </el-form>
        </el-tab-pane>

        <el-tab-pane label="内存管理" name="memory">
          <el-row :gutter="24">
            <el-col :span="14">
              <el-form :disabled="!isEditMode" label-width="160px">
                <el-form-item label="最大内存 (GB)">
                  <el-input-number v-model="memoryForm.max_memory_gb" :min="1" :step="4" :class="{ 'edit-highlight': isEditMode }" />
                </el-form-item>
                <el-form-item label="模型内存 (GB)">
                  <el-input-number v-model="memoryForm.model_memory_gb" :min="1" :step="4" :class="{ 'edit-highlight': isEditMode }" />
                </el-form-item>
                <el-form-item label="缓存内存 (GB)">
                  <el-input-number v-model="memoryForm.cache_memory_gb" :min="0" :step="2" :class="{ 'edit-highlight': isEditMode }" />
                </el-form-item>
                <el-form-item label="Arena 大小 (MB)">
                  <el-input-number v-model="memoryForm.arena_size_mb" :min="64" :step="32" :class="{ 'edit-highlight': isEditMode }" />
                </el-form-item>
              </el-form>
            </el-col>
            <el-col :span="10">
              <div class="chart-wrapper">
                <h4>内存分配分布</h4>
                <PieChart :data="memoryChartData" :height="250" :show-legend="true" />
              </div>
            </el-col>
          </el-row>
        </el-tab-pane>

        <el-tab-pane label="模型配置" name="model">
          <el-form :disabled="!isEditMode" label-width="160px">
            <el-form-item label="模型路径">
              <el-input v-model="modelForm.path" :class="{ 'edit-highlight': isEditMode }" />
            </el-form-item>
            <el-form-item label="默认量化类型">
              <el-select v-model="modelForm.quantization" :class="{ 'edit-highlight': isEditMode }">
                <el-option label="INT4" value="INT4" />
                <el-option label="INT8" value="INT8" />
                <el-option label="FP16" value="FP16" />
                <el-option label="FP32" value="FP32" />
              </el-select>
            </el-form-item>
            <el-form-item label="默认上下文长度">
              <el-input-number v-model="modelForm.context_length" :min="256" :step="512" :class="{ 'edit-highlight': isEditMode }" />
            </el-form-item>
          </el-form>
        </el-tab-pane>

        <el-tab-pane label="Worker" name="worker">
          <el-form :disabled="!isEditMode" label-width="160px">
            <el-form-item label="Worker 数量">
              <el-input-number v-model="workerForm.count" :min="1" :max="16" :class="{ 'edit-highlight': isEditMode }" />
            </el-form-item>
            <el-form-item label="失败自动重启">
              <el-switch v-model="workerForm.restart_on_failure" :disabled="!isEditMode" />
            </el-form-item>
            <el-form-item label="健康检查间隔 (ms)">
              <el-input-number v-model="workerForm.health_check_interval_ms" :min="1000" :step="1000" :class="{ 'edit-highlight': isEditMode }" />
            </el-form-item>
          </el-form>
        </el-tab-pane>

        <el-tab-pane label="gRPC" name="grpc">
          <el-form :disabled="!isEditMode" label-width="160px">
            <el-form-item label="最大消息大小 (MB)">
              <el-input-number v-model="grpcForm.max_message_size_mb" :min="1" :max="128" :step="4" :class="{ 'edit-highlight': isEditMode }" />
            </el-form-item>
            <el-form-item label="Keepalive 时间 (ms)">
              <el-input-number v-model="grpcForm.keepalive_time_ms" :min="5000" :step="5000" :class="{ 'edit-highlight': isEditMode }" />
            </el-form-item>
            <el-form-item label="Keepalive 超时 (ms)">
              <el-input-number v-model="grpcForm.keepalive_timeout_ms" :min="1000" :step="1000" :class="{ 'edit-highlight': isEditMode }" />
            </el-form-item>
          </el-form>
        </el-tab-pane>
      </el-tabs>

      <div v-if="isEditMode" class="form-actions">
        <el-button @click="handleCancel">取消</el-button>
        <el-button type="primary" :icon="Check" @click="handleSave">保存配置</el-button>
      </div>
    </el-card>

    <!-- 配置变更历史 -->
    <el-card shadow="hover" class="history-card">
      <template #header><span>配置变更历史</span></template>
      <el-table :data="historyList" stripe size="small">
        <el-table-column prop="operator" label="操作人" width="120" />
        <el-table-column prop="time" label="时间" width="180" />
        <el-table-column prop="group" label="分组" width="120">
          <template #default="{ row }">
            <el-tag size="small">{{ row.group }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="summary" label="摘要" />
      </el-table>
    </el-card>
  </div>
</template>

<style lang="scss" scoped>
.config-container {
  .history-card {
    margin-top: $spacing-lg;
  }

  .form-actions {
    display: flex;
    justify-content: flex-end;
    gap: $spacing-sm;
    padding: $spacing-lg 0;
    border-top: 1px solid $border-color-lighter;
    margin-top: $spacing-lg;
  }

  .edit-highlight {
    :deep(.el-input__wrapper),
    :deep(.el-textarea__inner) {
      box-shadow: 0 0 0 2px var(--el-color-primary) inset !important;
    }
  }

  .chart-wrapper {
    padding: $spacing-md;
    background: #fafafa;
    border-radius: $border-radius-md;

    h4 {
      margin: 0 0 $spacing-md;
      color: $text-primary;
      font-size: 14px;
    }
  }
}
</style>
