<script setup lang="ts">
// 监控仪表盘 - 占位页面
// TODO: 实现监控仪表盘功能，展示系统概览、实时指标、图表等
</script>

<template>
  <div class="page-container">
    <el-card>
      <template #header>
        <span class="page-title">监控仪表盘</span>
      </template>
      <div class="placeholder-content">
        <el-empty description="模块开发中..." />
      </div>
    </el-card>
  </div>
</template>

<style lang="scss" scoped>
.page-container {
  .page-title {
    font-size: $font-size-lg;
    font-weight: bold;
    color: $text-primary;
  }

  .placeholder-content {
    min-height: 400px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
}
</style>
