<script setup lang="ts">
import { WarningFilled } from '@element-plus/icons-vue'

interface Props {
  visible: boolean
  title?: string
  content: string
  confirmText?: string
  cancelText?: string
  type?: 'warning' | 'danger'
  loading?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  title: '确认操作',
  confirmText: '确认',
  cancelText: '取消',
  type: 'warning',
  loading: false
})

const emit = defineEmits<{
  (e: 'confirm'): void
  (e: 'cancel'): void
  (e: 'update:visible', value: boolean): void
}>()

function handleConfirm() {
  emit('confirm')
}

function handleCancel() {
  emit('cancel')
  emit('update:visible', false)
}
</script>

<template>
  <el-dialog
    :model-value="visible"
    :title="title"
    width="420px"
    :close-on-click-modal="false"
    :class="{ 'dialog-danger': type === 'danger' }"
    @update:model-value="(val: boolean) => emit('update:visible', val)"
  >
    <div class="dialog-content">
      <el-icon
        v-if="type === 'danger'"
        :size="48"
        :color="'#F56C6C'"
        class="dialog-icon"
      >
        <WarningFilled />
      </el-icon>
      <p>{{ content }}</p>
    </div>
    <template #footer>
      <el-button @click="handleCancel" :disabled="loading">
        {{ cancelText }}
      </el-button>
      <el-button
        :type="type === 'danger' ? 'danger' : 'primary'"
        :loading="loading"
        @click="handleConfirm"
      >
        {{ confirmText }}
      </el-button>
    </template>
  </el-dialog>
</template>

<style lang="scss" scoped>
.dialog-content {
  text-align: center;
  padding: $spacing-lg 0;

  .dialog-icon {
    margin-bottom: $spacing-md;
  }

  p {
    color: $text-regular;
    line-height: 1.6;
    margin: 0;
  }
}

:deep(.dialog-danger) {
  .el-dialog__header {
    background-color: #fef0f0;
    border-bottom: 1px solid #fde2e2;
  }
}
</style>
