<script setup lang="ts">
// 配置管理 - 占位页面
// TODO: 实现配置管理功能，包括系统参数设置、环境变量管理等
</script>

<template>
  <div class="page-container">
    <el-card>
      <template #header>
        <span class="page-title">配置管理</span>
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
