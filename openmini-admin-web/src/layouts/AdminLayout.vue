<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  Fold, Expand, Monitor, SetUp, Box, Key, User,
  Cpu, TrendCharts, Bell, Document, Notebook,
  Setting, DataAnalysis, Connection
} from '@element-plus/icons-vue'

const route = useRoute()
const router = useRouter()

// 折叠状态，从 localStorage 恢复
const isCollapsed = ref(localStorage.getItem('sidebarCollapsed') === 'true')

// 监听折叠状态变化，持久化到 localStorage
watch(isCollapsed, (val) => {
  localStorage.setItem('sidebarCollapsed', String(val))
})

// 侧边栏宽度
const sidebarWidth = computed(() => isCollapsed.value ? '64px' : '240px')

// 菜单数据，按分组组织
const menuGroups = [
  {
    title: '监控中心',
    children: [
      { path: '/dashboard', title: '监控仪表盘', icon: Monitor }
    ]
  },
  {
    title: '服务管理',
    children: [
      { path: '/service', title: '服务管理', icon: SetUp },
      { path: '/model', title: '模型管理', icon: Box }
    ]
  },
  {
    title: '权限管理',
    children: [
      { path: '/apikey', title: 'API Key', icon: Key },
      { path: '/user', title: '用户管理', icon: User }
    ]
  },
  {
    title: '运维中心',
    children: [
      { path: '/alert/rules', title: '告警规则', icon: Bell },
      { path: '/alert/records', title: '告警记录', icon: Document },
      { path: '/log', title: '日志中心', icon: Notebook },
      { path: '/config', title: '配置管理', icon: Setting }
    ]
  },
  {
    title: '数据分析',
    children: [
      { path: '/monitoring/resources', title: '资源监控', icon: Cpu },
      { path: '/monitoring/inference', title: '推理指标', icon: TrendCharts },
      { path: '/usage', title: '用量分析', icon: DataAnalysis },
      { path: '/trace', title: '请求追踪', icon: Connection }
    ]
  }
]

// 当前激活的菜单项
const activeMenu = computed(() => route.path)

// 面包屑数据
const breadcrumbs = computed(() => {
  return route.matched.filter(item => item.meta?.title).map(item => ({
    title: item.meta.title as string
  }))
})

// 切换折叠状态
const toggleCollapse = () => {
  isCollapsed.value = !isCollapsed.value
}

// 全屏切换
const isFullscreen = ref(false)
const toggleFullscreen = () => {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen()
    isFullscreen.value = true
  } else {
    document.exitFullscreen()
    isFullscreen.value = false
  }
}

// 用户下拉菜单操作
const handleCommand = (command: string) => {
  switch (command) {
    case 'profile':
      console.log('个人信息')
      break
    case 'logout':
      localStorage.removeItem('token')
      router.push('/login')
      break
  }
}
</script>

<template>
  <div class="admin-layout">
    <!-- 左侧边栏 -->
    <aside class="sidebar" :class="{ collapsed: isCollapsed }" :style="{ width: sidebarWidth }">
      <!-- Logo 区域 -->
      <div class="logo-container">
        <h1 v-show="!isCollapsed" class="logo-text">OpenMini</h1>
        <span v-show="isCollapsed" class="logo-text-short">O</span>
      </div>

      <!-- 导航菜单 -->
      <el-menu
        :default-active="activeMenu"
        :collapse="isCollapsed"
        :collapse-transition="true"
        router
        class="sidebar-menu"
      >
        <!-- 遍历菜单分组 -->
        <template v-for="group in menuGroups" :key="group.title">
          <!-- 单个菜单项（无子菜单时直接显示） -->
          <template v-if="group.children.length === 1">
            <el-menu-item :index="group.children[0].path">
              <el-icon><component :is="group.children[0].icon" /></el-icon>
              <template #title>{{ group.children[0].title }}</template>
            </el-menu-item>
          </template>

          <!-- 多个子菜单时使用 el-sub-menu -->
          <el-sub-menu v-else :index="group.title">
            <template #title>
              <el-icon><component :is="group.children[0].icon" /></el-icon>
              <span>{{ group.title }}</span>
            </template>
            <el-menu-item
              v-for="child in group.children"
              :key="child.path"
              :index="child.path"
            >
              <el-icon><component :is="child.icon" /></el-icon>
              <template #title>{{ child.title }}</template>
            </el-menu-item>
          </el-sub-menu>
        </template>
      </el-menu>

      <!-- 底部折叠按钮 -->
      <div class="collapse-btn" @click="toggleCollapse">
        <el-icon :size="20">
          <Fold v-if="!isCollapsed" />
          <Expand v-else />
        </el-icon>
      </div>
    </aside>

    <!-- 右侧主区域 -->
    <div class="main-area" :style="{ marginLeft: sidebarWidth }">
      <!-- 顶部导航栏 -->
      <header class="header">
        <div class="header-left">
          <el-icon class="collapse-trigger" :size="20" @click="toggleCollapse">
            <Fold v-if="!isCollapsed" />
            <Expand v-else />
          </el-icon>
        </div>

        <div class="header-center">
          <el-breadcrumb separator="/">
            <el-breadcrumb-item v-for="(item, index) in breadcrumbs" :key="index">
              {{ item.title }}
            </el-breadcrumb-item>
          </el-breadcrumb>
        </div>

        <div class="header-right">
          <el-tooltip content="全屏切换" placement="bottom">
            <el-icon class="fullscreen-btn" :size="18" @click="toggleFullscreen">
              <FullScreen v-if="!isFullscreen" />
              <CopyDocument v-else />
            </el-icon>
          </el-tooltip>

          <el-dropdown trigger="click" @command="handleCommand">
            <div class="user-dropdown">
              <el-avatar :size="32" icon="UserFilled" />
              <span class="username">管理员</span>
            </div>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item command="profile">个人信息</el-dropdown-item>
                <el-dropdown-item command="logout" divided>退出登录</el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </div>
      </header>

      <!-- 主内容区域 -->
      <main class="main-content">
        <router-view v-slot="{ Component }">
          <transition name="fade-transform" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </main>
    </div>
  </div>
</template>

<style lang="scss" scoped>
.admin-layout {
  display: flex;
  height: 100vh;
  overflow: hidden;
}

// 侧边栏样式
.sidebar {
  position: fixed;
  left: 0;
  top: 0;
  bottom: 0;
  width: 240px;
  background-color: $bg-color;
  border-right: 1px solid $border-color-lighter;
  transition: width $transition-duration $transition-timing;
  display: flex;
  flex-direction: column;
  z-index: 1001;
  overflow: hidden;

  &.collapsed {
    width: 64px;
  }

  .logo-container {
    height: $header-height;
    display: flex;
    align-items: center;
    justify-content: center;
    border-bottom: 1px solid $border-color-lighter;

    .logo-text {
      font-size: $font-size-xl;
      font-weight: bold;
      color: $primary-color;
      white-space: nowrap;
    }

    .logo-text-short {
      font-size: $font-size-xl;
      font-weight: bold;
      color: $primary-color;
    }
  }

  .sidebar-menu {
    flex: 1;
    overflow-y: auto;
    overflow-x: hidden;
    border-right: none;
    @include custom-scrollbar;
  }

  .collapse-btn {
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    border-top: 1px solid $border-color-lighter;
    color: $text-secondary;
    transition: color $transition-duration;

    &:hover {
      color: $primary-color;
    }
  }
}

// 右侧主区域
.main-area {
  flex: 1;
  margin-left: 240px;
  transition: margin-left $transition-duration $transition-timing;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

// 顶部导航栏
.header {
  height: 56px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 $spacing-lg;
  background-color: #fff;
  border-bottom: 1px solid $border-color-lighter;
  box-shadow: $box-shadow-base;
  z-index: 1000;

  .header-left {
    display: flex;
    align-items: center;

    .collapse-trigger {
      cursor: pointer;
      color: $text-secondary;
      transition: color $transition-duration;

      &:hover {
        color: $primary-color;
      }
    }
  }

  .header-center {
    flex: 1;
    display: flex;
    justify-content: center;
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: $spacing-md;

    .fullscreen-btn {
      cursor: pointer;
      color: $text-secondary;
      transition: color $transition-duration;

      &:hover {
        color: $primary-color;
      }
    }

    .user-dropdown {
      display: flex;
      align-items: center;
      gap: $spacing-sm;
      cursor: pointer;

      .username {
        font-size: $font-size-base;
        color: $text-primary;
      }
    }
  }
}

// 主内容区域
.main-content {
  flex: 1;
  padding: $spacing-lg;
  overflow-y: auto;
  background-color: $bg-color-page;
  @include custom-scrollbar;
}

// 页面过渡动画
.fade-transform-enter-active,
.fade-transform-leave-active {
  transition: all $transition-duration $transition-timing;
}

.fade-transform-enter-from {
  opacity: 0;
  transform: translateX(-30px);
}

.fade-transform-leave-to {
  opacity: 0;
  transform: translateX(30px);
}
</style>
