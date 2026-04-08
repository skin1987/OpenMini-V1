<script setup lang="ts">
import { computed } from 'vue'
import { TrendCharts, ArrowUp, ArrowDown } from '@element-plus/icons-vue'

interface Props {
  title: string
  value: number | string
  suffix?: string
  prefix?: string
  icon?: any
  trend?: { value: number; isUp: boolean }
  color?: 'default' | 'success' | 'warning' | 'danger'
  loading?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  suffix: '',
  prefix: '',
  icon: undefined,
  trend: undefined,
  color: 'default',
  loading: false
})

const colorMap = {
  default: '#409EFF',
  success: '#67C23A',
  warning: '#E6A23C',
  danger: '#F56C6C'
}

const cardColor = computed(() => colorMap[props.color])
</script>

<template>
  <el-card shadow="hover" class="stat-card" v-loading="loading">
    <div class="stat-header">
      <div class="stat-title">
        <el-icon v-if="icon" :size="20" style="margin-right: 8px">
          <component :is="icon" />
        </el-icon>
        <span>{{ title }}</span>
      </div>
      <div v-if="trend" class="stat-trend" :class="{ up: trend.isUp, down: !trend.isUp }">
        <el-icon :size="14">
          <component :is="trend.isUp ? ArrowUp : ArrowDown" />
        </el-icon>
        <span>{{ Math.abs(trend.value) }}%</span>
      </div>
    </div>
    <div class="stat-value" :style="{ color: cardColor }">
      <span v-if="prefix">{{ prefix }}</span>
      {{ value }}
      <span v-if="suffix">{{ suffix }}</span>
    </div>
    <el-divider />
    <slot></slot>
    <template #footer v-if="$slots.footer">
      <slot name="footer"></slot>
    </template>
  </el-card>
</template>

<style lang="scss" scoped>
.stat-card {
  height: 100%;

  .stat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: $spacing-md;

    .stat-title {
      font-size: $font-size-sm;
      color: $text-secondary;
      display: flex;
      align-items: center;
    }

    .stat-trend {
      display: flex;
      align-items: center;
      gap: 4px;
      font-size: $font-size-xs;

      &.up { color: $success-color; }
      &.down { color: $danger-color; }
    }
  }

  .stat-value {
    font-size: 28px;
    font-weight: bold;
    line-height: 1.2;
  }

  :deep(.el-divider) {
    margin: $spacing-sm 0;
  }
}
</style>
