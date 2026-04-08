<script setup lang="ts">
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { User, Lock } from '@element-plus/icons-vue'

const router = useRouter()

// 表单数据
const loginForm = reactive({
  username: '',
  password: ''
})

// 加载状态
const loading = ref(false)

// 登录处理（占位）
const handleLogin = async () => {
  if (!loginForm.username || !loginForm.password) {
    return
  }
  loading.value = true
  // TODO: 实现登录逻辑，调用后端接口验证用户名密码
  setTimeout(() => {
    localStorage.setItem('token', 'mock-token')
    const redirect = (router.currentRoute.value.query.redirect as string) || '/'
    router.push(redirect)
    loading.value = false
  }, 1000)
}
</script>

<template>
  <div class="login-container">
    <div class="login-card">
      <h2 class="login-title">OpenMini 运维管理</h2>
      <p class="login-subtitle">AI 模型服务运维平台</p>

      <el-form :model="loginForm" class="login-form">
        <el-form-item prop="username">
          <el-input
            v-model="loginForm.username"
            placeholder="请输入用户名"
            size="large"
            :prefix-icon="User"
          />
        </el-form-item>

        <el-form-item prop="password">
          <el-input
            v-model="loginForm.password"
            type="password"
            placeholder="请输入密码"
            size="large"
            show-password
            :prefix-icon="Lock"
            @keyup.enter="handleLogin"
          />
        </el-form-item>

        <el-form-item>
          <el-button
            type="primary"
            size="large"
            class="login-btn"
            :loading="loading"
            @click="handleLogin"
          >
            登 录
          </el-button>
        </el-form-item>
      </el-form>
    </div>
  </div>
</template>

<style lang="scss" scoped>
.login-container {
  width: 100%;
  height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

  .login-card {
    width: 420px;
    padding: 40px;
    background-color: #fff;
    border-radius: $border-radius-lg;
    box-shadow: $box-shadow-dark;

    .login-title {
      text-align: center;
      font-size: $font-size-xl * 1.5;
      color: $text-primary;
      margin-bottom: $spacing-sm;
    }

    .login-subtitle {
      text-align: center;
      font-size: $font-size-base;
      color: $text-secondary;
      margin-bottom: $spacing-xl * 1.5;
    }

    .login-form {
      .login-btn {
        width: 100%;
      }
    }
  }
}
</style>
