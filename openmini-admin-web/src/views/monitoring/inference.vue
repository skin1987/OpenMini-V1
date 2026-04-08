<script setup lang="ts">
// 推理指标 - 占位页面
// TODO: 实现推理指标监控功能，包括 QPS、延迟、吞吐量等性能指标
</script>

<template>
  <div class="page-container">
    <el-card>
      <template #header>
        <span class="page-title">推理指标</span>
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
