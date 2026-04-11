<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Plus } from '@element-plus/icons-vue'
import { getModelList, loadModel, unloadModel, switchModel, checkModelHealth } from '@/api/model'

const loading = ref(false)
const keyword = ref('')
const statusFilter = ref('')
const dialogVisible = ref(false)
const drawerVisible = ref(false)
const currentModel = ref<any>(null)
const switchVisible = ref(false)

const modelList = ref<any[]>([])

const form = reactive({
  path: '',
  quantization: 'INT8',
  contextLength: 4096
})

async function fetchModelList() {
  loading.value = true
  try {
    const res: any = await getModelList()
    modelList.value = res.models || []
  } catch (error: any) {
    ElMessage.error(error.message || '获取模型列表失败')
  } finally {
    loading.value = false
  }
}

function getStatusType(status: string) {
  const map: Record<string, string> = {
    loaded: 'success',
    unloaded: 'info',
    loading: 'warning',
    error: 'danger'
  }
  return map[status] || 'info'
}

function getStatusText(status: string) {
  const map: Record<string, string> = {
    loaded: '已加载',
    unloaded: '未加载',
    loading: '加载中',
    error: '错误'
  }
  return map[status] || status
}

function getQuantType(type: string) {
  const map: Record<string, string> = {
    INT4: 'danger',
    INT8: 'warning',
    FP16: '',
    FP32: 'success'
  }
  return map[type] || ''
}

function handleLoadModel() {
  dialogVisible.value = true
}

async function submitLoadModel() {
  try {
    await loadModel({ path: form.path })
    ElMessage.success('模型加载任务已提交')
    dialogVisible.value = false
    await fetchModelList()
  } catch (error: any) {
    ElMessage.error(error.message || '模型加载失败')
  }
}

function handleDetail(row: any) {
  currentModel.value = row
  drawerVisible.value = true
}

async function handleToggleLoad(row: any) {
  if (row.status === 'loaded') {
    try {
      await unloadModel(row.id)
      ElMessage.success(`模型 ${row.name} 已卸载`)
      await fetchModelList()
    } catch (error: any) {
      ElMessage.error(error.message || '卸载模型失败')
    }
  } else {
    try {
      await loadModel({ path: row.path })
      ElMessage.success(`开始加载模型: ${row.name}`)
      await fetchModelList()
    } catch (error: any) {
      ElMessage.error(error.message || '加载模型失败')
    }
  }
}

async function handleHealthCheck(row: any) {
  try {
    const res: any = await checkModelHealth(row.id)
    if (res.healthy !== false) {
      ElMessage.success(`${row.name} 健康检查通过`)
    } else {
      ElMessage.warning(`${row.name} 健康检查异常`)
    }
  } catch (error: any) {
    ElMessage.error(error.message || '健康检查失败')
  }
}

async function handleSwitchModel(targetName?: string) {
  if (!targetName) {
    switchVisible.value = false
    return
  }
  try {
    await switchModel({ target: targetName })
    ElMessage.success('热切换完成')
    switchVisible.value = false
    await fetchModelList()
  } catch (error: any) {
    ElMessage.error(error.message || '热切换失败')
  }
}

onMounted(() => {
  fetchModelList()
})
</script>

<template>
  <div class="model-container">
    <el-card shadow="hover" class="toolbar-card">
      <div class="toolbar">
        <div class="toolbar-left">
          <el-input v-model="keyword" placeholder="搜索模型名称..." clearable style="width: 240px" />
          <el-select v-model="statusFilter" placeholder="状态筛选" clearable style="width: 140px">
            <el-option label="已加载" value="loaded" />
            <el-option label="未加载" value="unloaded" />
            <el-option label="加载中" value="loading" />
            <el-option label="错误" value="error" />
          </el-select>
        </div>
        <div class="toolbar-right">
          <el-button type="primary" :icon="Plus" @click="handleLoadModel">加载模型</el-button>
        </div>
      </div>
    </el-card>

    <el-card shadow="hover">
      <el-table :data="modelList" stripe v-loading="loading">
        <el-table-column prop="name" label="模型名称" min-width="180" />
        <el-table-column prop="path" label="路径" min-width="200" show-overflow-tooltip />
        <el-table-column prop="source" label="来源" width="100" />
        <el-table-column prop="size" label="大小" width="100" />
        <el-table-column prop="quantization" label="量化类型" width="110">
          <template #default="{ row }">
            <el-tag :type="getQuantType(row.quantization)" size="small">{{ row.quantization }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="contextLength" label="上下文长度" width="110">
          <template #default="{ row }">
            {{ row.contextLength?.toLocaleString() || '-' }}
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="100">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)" size="small">{{ getStatusText(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="280" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" link size="small" @click="handleDetail(row)">详情</el-button>
            <el-button :type="row.status === 'loaded' ? 'warning' : 'success'" link size="small" @click="handleToggleLoad(row)">
              {{ row.status === 'loaded' ? '卸载' : '加载' }}
            </el-button>
            <el-button type="info" link size="small" @click="handleHealthCheck(row)">健康</el-button>
            <el-button type="warning" link size="small" :disabled="row.status !== 'loaded'" @click="switchVisible = true">热切换</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog v-model="dialogVisible" title="加载模型" width="500px">
      <el-form :model="form" label-width="100px">
        <el-form-item label="模型路径">
          <el-input v-model="form.path" placeholder="/models/model-name" />
        </el-form-item>
        <el-form-item label="量化类型">
          <el-select v-model="form.quantization">
            <el-option label="INT4" value="INT4" />
            <el-option label="INT8" value="INT8" />
            <el-option label="FP16" value="FP16" />
            <el-option label="FP32" value="FP32" />
          </el-select>
        </el-form-item>
        <el-form-item label="上下文长度">
          <el-input-number v-model="form.contextLength" :min="512" :max="131072" :step="512" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="submitLoadModel">确认加载</el-button>
      </template>
    </el-dialog>

    <el-drawer v-model="drawerVisible" title="模型详情" size="450px">
      <el-descriptions v-if="currentModel" :column="1" border>
        <el-descriptions-item label="名称">{{ currentModel.name }}</el-descriptions-item>
        <el-descriptions-item label="路径">{{ currentModel.path }}</el-descriptions-item>
        <el-descriptions-item label="来源">{{ currentModel.source || '-' }}</el-descriptions-item>
        <el-descriptions-item label="大小">{{ currentModel.size }}</el-descriptions-item>
        <el-descriptions-item label="量化类型">{{ currentModel.quantization }}</el-descriptions-item>
        <el-descriptions-item label="上下文长度">{{ currentModel.contextLength?.toLocaleString() || '-' }}</el-descriptions-item>
        <el-descriptions-item label="状态">
          <el-tag :type="getStatusType(currentModel.status)" size="small">{{ getStatusText(currentModel.status) }}</el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="加载时间">{{ currentModel.loadTime || '-' }}</el-descriptions-item>
      </el-descriptions>
    </el-drawer>

    <ConfirmDialog v-model:visible="switchVisible" title="确认热切换" content="热切换将在不中断服务的情况下替换模型，切换过程中可能有短暂延迟" type="warning" @confirm="handleSwitchModel(currentModel?.name)" />
  </div>
</template>

<style lang="scss" scoped>
.model-container {
  .toolbar-card {
    margin-bottom: 16px;
  }

  .toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;

    .toolbar-left {
      display: flex;
      gap: 12px;
    }
  }
}
</style>
