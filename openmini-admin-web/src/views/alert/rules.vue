<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus, ArrowDown } from '@element-plus/icons-vue'
import {
  getAlertRules,
  createAlertRule,
  updateAlertRule,
  deleteAlertRule,
  toggleAlertRule
} from '@/api/alert'
import type { AlertRule, CreateAlertRuleRequest, UpdateAlertRuleRequest } from '@/types/api/alert'
import type { AlertSeverity } from '@/types'

const loading = ref(false)
const ruleList = ref<AlertRule[]>([])
const statusFilter = ref('')
const dialogVisible = ref(false)
const dialogTitle = ref('创建告警规则')
const isEdit = ref(false)
const currentEditRule = ref<AlertRule | null>(null)

const form = reactive({
  name: '',
  metric: '',
  operator: 'gt',
  threshold: 0,
  duration_seconds: 0,
  severity: 'warning' as AlertSeverity,
  notification_channels: [] as string[],
  webhook_url: '',
  description: ''
})

const metricOptions = [
  { label: 'GPU 利用率', value: 'gpu_utilization' },
  { label: '显存使用', value: 'memory' },
  { label: 'CPU 使用率', value: 'cpu' },
  { label: '推理延迟', value: 'latency' },
  { label: '错误率', value: 'error_rate' },
  { label: '调度队列长度', value: 'queue_length' },
  { label: 'KV Cache 使用', value: 'kv_cache' }
]

const operatorOptions = [
  { label: '大于 (>)', value: 'gt' },
  { label: '小于 (<)', value: 'lt' },
  { label: '大于等于 (≥)', value: 'gte' },
  { label: '小于等于 (≤)', value: 'lte' },
  { label: '等于 (=)', value: 'eq' },
  { label: '不等于 (≠)', value: 'ne' }
]

const channelOptions = [
  { label: 'Webhook', value: 'webhook' },
  { label: '邮件', value: 'email' },
  { label: '企业微信', value: 'wechat' },
  { label: '钉钉', value: 'dingtalk' },
  { label: 'Slack', value: 'slack' }
]

const templates = [
  {
    name: 'GPU 利用率过高',
    metric: 'gpu_utilization',
    operator: 'gt',
    threshold: 90,
    duration_seconds: 300,
    severity: 'critical' as AlertSeverity,
    emoji: '🔴'
  },
  {
    name: '显存使用率过高',
    metric: 'memory',
    operator: 'gt',
    threshold: 95,
    duration_seconds: 300,
    severity: 'critical' as AlertSeverity,
    emoji: '🔴'
  },
  {
    name: '推理延迟 P99 过高',
    metric: 'latency',
    operator: 'gt',
    threshold: 5000,
    duration_seconds: 180,
    severity: 'warning' as AlertSeverity,
    emoji: '🟡'
  },
  {
    name: '错误率过高',
    metric: 'error_rate',
    operator: 'gt',
    threshold: 5,
    duration_seconds: 180,
    severity: 'warning' as AlertSeverity,
    emoji: '🟡'
  },
  {
    name: '调度队列过长',
    metric: 'queue_length',
    operator: 'gt',
    threshold: 100,
    duration_seconds: 60,
    severity: 'warning' as AlertSeverity,
    emoji: '🟡'
  },
  {
    name: 'KV Cache 使用率过高',
    metric: 'kv_cache',
    operator: 'gt',
    threshold: 90,
    duration_seconds: 120,
    severity: 'info' as AlertSeverity,
    emoji: '🔵'
  }
]

const filteredRules = computed(() => {
  if (!statusFilter.value) return ruleList.value
  return ruleList.value.filter(r =>
    statusFilter.value === 'enabled' ? r.enabled : !r.enabled
  )
})

onMounted(() => {
  fetchRules()
})

async function fetchRules() {
  loading.value = true
  try {
    const res = await getAlertRules()
    ruleList.value = Array.isArray(res) ? res : []
  } catch (error) {
    ElMessage.error('获取告警规则失败')
  } finally {
    loading.value = false
  }
}

function handleStatusFilter() {}

function openCreateDialog() {
  isEdit.value = false
  dialogTitle.value = '创建告警规则'
  resetForm()
  dialogVisible.value = true
}

function openEditDialog(rule: AlertRule) {
  isEdit.value = true
  dialogTitle.value = '编辑告警规则'
  currentEditRule.value = rule
  form.name = rule.name
  form.metric = rule.condition.metric
  form.operator = rule.condition.operator
  form.threshold = Number(rule.condition.threshold)
  form.duration_seconds = rule.condition.duration_seconds || 0
  form.severity = rule.severity
  form.notification_channels = rule.notification_channels.map(ch => ch.type)
  form.webhook_url = rule.notification_channels.find(ch => ch.type === 'webhook')?.config?.url || ''
  form.description = rule.description || ''
  dialogVisible.value = true
}

function applyTemplate(template: typeof templates[0]) {
  isEdit.value = false
  dialogTitle.value = '创建告警规则'
  form.name = template.name
  form.metric = template.metric
  form.operator = template.operator
  form.threshold = template.threshold
  form.duration_seconds = template.duration_seconds
  form.severity = template.severity
  form.notification_channels = ['webhook']
  form.webhook_url = ''
  form.description = ''
  dialogVisible.value = true
}

function resetForm() {
  form.name = ''
  form.metric = ''
  form.operator = 'gt'
  form.threshold = 0
  form.duration_seconds = 0
  form.severity = 'warning'
  form.notification_channels = []
  form.webhook_url = ''
  form.description = ''
}

async function handleSubmit() {
  if (!form.name.trim()) {
    ElMessage.warning('请输入规则名称')
    return
  }
  if (!form.metric) {
    ElMessage.warning('请选择监控指标')
    return
  }

  try {
    const data: CreateAlertRuleRequest = {
      name: form.name.trim(),
      description: form.description || undefined,
      severity: form.severity,
      condition: {
        metric: form.metric,
        operator: form.operator as any,
        threshold: form.threshold,
        duration_seconds: form.duration_seconds || undefined
      },
      notification_channels: form.notification_channels.map(type => ({
        type: type as any,
        config: type === 'webhook' ? { url: form.webhook_url || '' } : {} as Record<string, string>,
        enabled: true
      })),
      cooldown_seconds: form.duration_seconds || undefined
    }

    if (isEdit.value && currentEditRule.value) {
      const updateData: UpdateAlertRuleRequest = { ...data }
      await updateAlertRule(currentEditRule.value.id, updateData)
      ElMessage.success('更新成功')
    } else {
      await createAlertRule(data)
      ElMessage.success('创建成功')
    }

    dialogVisible.value = false
    fetchRules()
  } catch (error) {
    ElMessage.error(isEdit.value ? '更新失败' : '创建失败')
  }
}

async function handleToggleEnabled(rule: AlertRule) {
  try {
    const result = await toggleAlertRule(rule.id)
    ElMessage.success(`已${result.new_enabled_state ? '启用' : '禁用'}该规则`)
    fetchRules()
  } catch (error) {
    ElMessage.error('操作失败')
  }
}

async function handleDelete(rule: AlertRule) {
  try {
    await ElMessageBox.confirm(
      `确定要删除规则「${rule.name}」吗？`,
      '确认删除',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    await deleteAlertRule(rule.id)
    ElMessage.success('删除成功')
    fetchRules()
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error('删除失败')
    }
  }
}

function getMetricLabel(metric: string): string {
  return metricOptions.find(opt => opt.value === metric)?.label || metric
}

function getOperatorLabel(operator: string): string {
  const map: Record<string, string> = {
    gt: '>',
    lt: '<',
    gte: '≥',
    lte: '≤',
    eq: '=',
    ne: '≠'
  }
  return map[operator] || operator
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

function formatDuration(seconds?: number): string {
  if (!seconds) return '-'
  const minutes = Math.floor(seconds / 60)
  const secs = seconds % 60
  if (minutes > 0) {
    return `${minutes}分${secs > 0 ? secs + '秒' : ''}`
  }
  return `${secs}秒`
}

function formatTime(time: string) {
  return time.slice(0, 16).replace('T', ' ')
}
</script>

<template>
  <div class="page-container">
    <div class="page-header">
      <h2>告警规则</h2>
    </div>

    <el-card class="toolbar-card" shadow="never">
      <div class="toolbar">
        <el-select
          v-model="statusFilter"
          placeholder="状态筛选"
          clearable
          style="width: 140px"
          @change="handleStatusFilter"
        >
          <el-option label="全部" value="" />
          <el-option label="启用" value="enabled" />
          <el-option label="禁用" value="disabled" />
        </el-select>

        <el-button type="primary" @click="openCreateDialog">
          <el-icon><Plus /></el-icon>
          创建规则
        </el-button>

        <el-dropdown trigger="click" @command="(cmd: typeof templates[0]) => applyTemplate(cmd)">
          <el-button type="success">
            模板快速创建<el-icon class="el-icon--right"><ArrowDown /></el-icon>
          </el-button>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item
                v-for="(tpl, index) in templates"
                :key="index"
                :command="tpl"
              >
                {{ tpl.emoji }} {{ tpl.name }}
                <span class="template-detail">
                  {{ getMetricLabel(tpl.metric) }} {{ getOperatorLabel(tpl.operator) }} {{ tpl.threshold }}
                  {{ tpl.duration_seconds ? `持续${formatDuration(tpl.duration_seconds)}` : '' }}
                  [{{ getSeverityText(tpl.severity) }}]
                </span>
              </el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
      </div>
    </el-card>

    <el-card shadow="never" class="table-card">
      <el-table
        :data="filteredRules"
        v-loading="loading"
        stripe
        border
        style="width: 100%"
      >
        <el-table-column prop="name" label="名称" min-width="140" />

        <el-table-column label="监控指标" width="130">
          <template #default="{ row }">
            <el-tag effect="plain" size="small">
              {{ getMetricLabel(row.condition.metric) }}
            </el-tag>
          </template>
        </el-table-column>

        <el-table-column label="条件" width="140">
          <template #default="{ row }">
            <span class="condition-text">
              {{ getOperatorLabel(row.condition.operator) }} {{ row.condition.threshold }}
            </span>
          </template>
        </el-table-column>

        <el-table-column label="持续时间" width="110">
          <template #default="{ row }">
            {{ formatDuration(row.condition.duration_seconds) }}
          </template>
        </el-table-column>

        <el-table-column label="级别" width="90">
          <template #default="{ row }">
            <el-tag :type="getSeverityType(row.severity)" effect="dark" size="small">
              {{ getSeverityText(row.severity) }}
            </el-tag>
          </template>
        </el-table-column>

        <el-table-column label="通知渠道" min-width="160">
          <template #default="{ row }">
            <el-tag
              v-for="ch in row.notification_channels"
              :key="ch.type"
              size="small"
              effect="plain"
              style="margin-right: 4px; margin-bottom: 2px"
            >
              {{ ch.type === 'webhook' ? 'Webhook' : ch.type === 'email' ? '邮件' : ch.type === 'wechat' ? '企微' : ch.type === 'dingtalk' ? '钉钉' : 'Slack' }}
            </el-tag>
          </template>
        </el-table-column>

        <el-table-column label="状态" width="90" align="center">
          <template #default="{ row }">
            <el-switch
              :model-value="row.enabled"
              @change="handleToggleEnabled(row)"
              active-text=""
              inactive-text=""
            />
          </template>
        </el-table-column>

        <el-table-column label="创建时间" width="160">
          <template #default="{ row }">
            {{ formatTime(row.created_at) }}
          </template>
        </el-table-column>

        <el-table-column label="操作" width="180" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="openEditDialog(row)">
              编辑
            </el-button>
            <el-popconfirm
              title="确定要删除该规则吗？"
              confirmButtonText="确定"
              cancelButtonText="取消"
              @confirm="handleDelete(row)"
            >
              <template #reference>
                <el-button link type="danger" size="small">删除</el-button>
              </template>
            </el-popconfirm>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog
      v-model="dialogVisible"
      :title="dialogTitle"
      width="600px"
      :close-on-click-modal="false"
    >
      <el-form label-width="110px" :model="form">
        <el-form-item label="规则名称" required>
          <el-input v-model="form.name" placeholder="请输入规则名称" maxlength="50" show-word-limit />
        </el-form-item>

        <el-form-item label="监控指标" required>
          <el-select v-model="form.metric" placeholder="请选择监控指标" style="width: 100%">
            <el-option
              v-for="opt in metricOptions"
              :key="opt.value"
              :label="opt.label"
              :value="opt.value"
            />
          </el-select>
        </el-form-item>

        <el-form-item label="条件" required>
          <el-select v-model="form.operator" style="width: 120px; margin-right: 8px">
            <el-option
              v-for="opt in operatorOptions"
              :key="opt.value"
              :label="opt.label"
              :value="opt.value"
            />
          </el-select>
          <el-input-number v-model="form.threshold" :min="0" controls-position="right" style="flex: 1" />
        </el-form-item>

        <el-form-item label="持续时间">
          <el-input-number
            v-model="form.duration_seconds"
            :min="0"
            :step="60"
            controls-position="right"
            style="width: 200px"
          />
          <span class="unit-text">秒（{{ formatDuration(form.duration_seconds) }}）</span>
        </el-form-item>

        <el-form-item label="告警级别" required>
          <el-radio-group v-model="form.severity">
            <el-radio value="critical" style="color: #f56c6c">严重</el-radio>
            <el-radio value="warning" style="color: #e6a23c">警告</el-radio>
            <el-radio value="info" style="color: #909399">信息</el-radio>
          </el-radio-group>
        </el-form-item>

        <el-form-item label="通知渠道">
          <el-checkbox-group v-model="form.notification_channels">
            <el-checkbox
              v-for="ch in channelOptions"
              :key="ch.value"
              :value="ch.value"
            >
              {{ ch.label }}
            </el-checkbox>
          </el-checkbox-group>
        </el-form-item>

        <el-form-item v-if="form.notification_channels.includes('webhook')" label="Webhook URL">
          <el-input
            v-model="form.webhook_url"
            placeholder="请输入 Webhook 地址"
            type="textarea"
            :rows="2"
          />
        </el-form-item>

        <el-form-item label="备注">
          <el-input
            v-model="form.description"
            placeholder="可选：添加规则说明"
            type="textarea"
            :rows="3"
            maxlength="200"
            show-word-limit
          />
        </el-form-item>
      </el-form>

      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="handleSubmit">
          {{ isEdit ? '保存' : '创建' }}
        </el-button>
      </template>
    </el-dialog>
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
  .condition-text {
    font-family: monospace;
    font-weight: 600;
    color: #409eff;
  }
}

.template-detail {
  margin-left: 8px;
  font-size: 12px;
  color: #909399;
}

.unit-text {
  margin-left: 8px;
  color: #909399;
  font-size: 13px;
}
</style>
