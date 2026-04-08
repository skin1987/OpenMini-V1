<script setup lang="ts">
// 服务管理 - 占位页面
// TODO: 实现服务管理功能，包括服务列表、启停、配置等操作
</script>

<template>
  <div class="page-container">
    <el-card>
      <template #header>
        <span class="page-title">服务管理</span>
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
