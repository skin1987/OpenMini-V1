<script setup lang="ts">
// 日志中心 - 占位页面
// TODO: 实现日志中心功能，包括日志查询、过滤、导出等
</script>

<template>
  <div class="page-container">
    <el-card>
      <template #header>
        <span class="page-title">日志中心</span>
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
