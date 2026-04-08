<script setup lang="ts">
// 用量分析 - 占位页面
// TODO: 实现用量分析功能，包括 API 调用统计、Token 用量、费用分析等
</script>

<template>
  <div class="page-container">
    <el-card>
      <template #header>
        <span class="page-title">用量分析</span>
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
