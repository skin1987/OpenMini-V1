<script setup lang="ts">
import { ArrowLeft } from '@element-plus/icons-vue'

interface Props {
  title: string
  subtitle?: string
  showBack?: boolean
}

withDefaults(defineProps<Props>(), {
  subtitle: '',
  showBack: false
})

const emit = defineEmits(['back'])
</script>

<template>
  <el-row class="page-header" align="middle">
    <el-col :span="16">
      <div class="header-left">
        <el-button
          v-if="showBack"
          text
          type="primary"
          :icon="ArrowLeft"
          @click="emit('back')"
          style="margin-right: 12px"
        >
          返回
        </el-button>
        <div class="title-group">
          <h2 class="title">{{ title }}</h2>
          <p v-if="subtitle" class="subtitle">{{ subtitle }}</p>
        </div>
      </div>
    </el-col>
    <el-col :span="8">
      <div class="header-extra">
        <slot name="extra"></slot>
      </div>
    </el-col>
  </el-row>
</template>

<style lang="scss" scoped>
.page-header {
  margin-bottom: $spacing-lg;
  padding: $spacing-md 0;

  .header-left {
    display: flex;
    align-items: center;

    .title-group {
      .title {
        margin: 0;
        font-size: 20px;
        color: $text-primary;
        font-weight: 600;
      }

      .subtitle {
        margin: 4px 0 0;
        font-size: $font-size-sm;
        color: $text-secondary;
      }
    }
  }

  .header-extra {
    display: flex;
    justify-content: flex-end;
    align-items: center;
  }
}
</style>
