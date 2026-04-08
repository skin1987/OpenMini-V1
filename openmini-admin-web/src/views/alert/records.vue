<script setup lang="ts">
// 告警记录 - 占位页面
// TODO: 实现告警记录查询功能，包括历史告警列表、详情查看、状态处理等
</script>

<template>
  <div class="page-container">
    <el-card>
      <template #header>
        <span class="page-title">告警记录</span>
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
