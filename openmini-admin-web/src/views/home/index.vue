<script setup lang="ts">
</script>

<template>
  <div class="home-container">
    <div class="welcome-section">
      <h1 class="welcome-title">OpenMini 运维管理平台</h1>
      <p class="welcome-desc">AI 模型服务一站式运维管理解决方案</p>
      <div class="quick-cards">
        <div class="quick-card" @click="$router.push('/dashboard')">
          <div class="card-icon" style="background: linear-gradient(135deg, #409eff, #66b1ff)">
            <el-icon :size="28"><Monitor /></el-icon>
          </div>
          <span>监控仪表盘</span>
        </div>
        <div class="quick-card" @click="$router.push('/model')">
          <div class="card-icon" style="background: linear-gradient(135deg, #67c23a, #85ce61)">
            <el-icon :size="28"><Box /></el-icon>
          </div>
          <span>模型管理</span>
        </div>
        <div class="quick-card" @click="$router.push('/alert/rules')">
          <div class="card-icon" style="background: linear-gradient(135deg, #e6a23c, #ebb563)">
            <el-icon :size="28"><Bell /></el-icon>
          </div>
          <span>告警中心</span>
        </div>
        <div class="quick-card" @click="$router.push('/usage')">
          <div class="card-icon" style="background: linear-gradient(135deg, #f56c6c, #f78989)">
            <el-icon :size="28"><DataAnalysis /></el-icon>
          </div>
          <span>用量分析</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped lang="scss">
.home-container {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: calc(100vh - 120px);
}

.welcome-section {
  text-align: center;

  .welcome-title {
    font-size: 32px;
    font-weight: 700;
    background: linear-gradient(135deg, #409eff, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 12px;
  }

  .welcome-desc {
    font-size: 16px;
    color: #909399;
    margin-bottom: 48px;
  }

  .quick-cards {
    display: flex;
    gap: 24px;
    justify-content: center;

    .quick-card {
      cursor: pointer;
      padding: 24px 20px;
      border-radius: 12px;
      background: #fff;
      box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
      transition: all 0.25s ease;
      width: 140px;
      text-align: center;

      &:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
      }

      .card-icon {
        width: 56px;
        height: 56px;
        border-radius: 14px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: #fff;
        margin-bottom: 12px;
      }

      span {
        font-size: 14px;
        font-weight: 500;
        color: #303133;
      }
    }
  }
}
</style>
