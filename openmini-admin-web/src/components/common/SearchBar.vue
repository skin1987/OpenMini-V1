<script setup lang="ts">
import { Search, RefreshRight } from '@element-plus/icons-vue'

interface Props {
  modelValue: string
  placeholder?: string
  showSearch?: boolean
  showReset?: boolean
}

withDefaults(defineProps<Props>(), {
  placeholder: '请输入关键词搜索...',
  showSearch: true,
  showReset: true
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void
  (e: 'search'): void
  (e: 'reset'): void
}>()

function handleInput(e: Event) {
  emit('update:modelValue', (e.target as HTMLInputElement).value)
}

function handleKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter') {
    emit('search')
  }
}
</script>

<template>
  <div class="search-bar">
    <el-input
      :model-value="modelValue"
      :placeholder="placeholder"
      clearable
      :prefix-icon="Search"
      @input="handleInput"
      @keydown="handleKeydown"
    />
    <div class="search-actions">
      <el-button
        v-if="showSearch"
        type="primary"
        :icon="Search"
        @click="emit('search')"
      >
        搜索
      </el-button>
      <el-button
        v-if="showReset"
        :icon="RefreshRight"
        @click="emit('reset')"
      >
        重置
      </el-button>
    </div>
  </div>
</template>

<style lang="scss" scoped>
.search-bar {
  display: flex;
  gap: $spacing-sm;
  align-items: center;

  .search-actions {
    display: flex;
    gap: $spacing-xs;
  }
}
</style>
