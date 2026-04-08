<script setup lang="ts">
// 资源监控 - 占位页面
// TODO: 实现资源监控功能，包括 CPU、内存、GPU 使用率等系统资源监控
</script>

<template>
  <div class="page-container">
    <el-card>
      <template #header>
        <span class="page-title">资源监控</span>
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
