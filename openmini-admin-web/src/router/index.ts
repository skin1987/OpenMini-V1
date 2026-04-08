import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'
import { useUserStore } from '@/stores/user'
import AdminLayout from '@/layouts/AdminLayout.vue'

const routes: RouteRecordRaw[] = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/login/index.vue'),
    meta: {
      title: '登录',
      requiresAuth: false
    }
  },
  {
    path: '/',
    component: AdminLayout,
    redirect: '/dashboard',
    meta: {
      requiresAuth: true
    },
    children: [
      {
        path: 'dashboard',
        name: 'Dashboard',
        component: () => import('@/views/dashboard/index.vue'),
        meta: { title: '监控仪表盘', icon: 'Monitor' }
      },
      {
        path: 'service',
        name: 'Service',
        component: () => import('@/views/service/index.vue'),
        meta: { title: '服务管理', icon: 'SetUp' }
      },
      {
        path: 'model',
        name: 'Model',
        component: () => import('@/views/model/index.vue'),
        meta: { title: '模型管理', icon: 'Box' }
      },
      {
        path: 'apikey',
        name: 'ApiKey',
        component: () => import('@/views/apikey/index.vue'),
        meta: { title: 'API Key', icon: 'Key' }
      },
      {
        path: 'user',
        name: 'User',
        component: () => import('@/views/user/index.vue'),
        meta: { title: '用户管理', icon: 'User' }
      },
      {
        path: 'monitoring/resources',
        name: 'Resources',
        component: () => import('@/views/monitoring/resources.vue'),
        meta: { title: '资源监控', icon: 'Cpu' }
      },
      {
        path: 'monitoring/inference',
        name: 'Inference',
        component: () => import('@/views/monitoring/inference.vue'),
        meta: { title: '推理指标', icon: 'TrendCharts' }
      },
      {
        path: 'alert/rules',
        name: 'AlertRules',
        component: () => import('@/views/alert/rules.vue'),
        meta: { title: '告警规则', icon: 'Bell' }
      },
      {
        path: 'alert/records',
        name: 'AlertRecords',
        component: () => import('@/views/alert/records.vue'),
        meta: { title: '告警记录', icon: 'Document' }
      },
      {
        path: 'log',
        name: 'Log',
        component: () => import('@/views/log/index.vue'),
        meta: { title: '日志中心', icon: 'Notebook' }
      },
      {
        path: 'config',
        name: 'Config',
        component: () => import('@/views/config/index.vue'),
        meta: { title: '配置管理', icon: 'Setting' }
      },
      {
        path: 'usage',
        name: 'Usage',
        component: () => import('@/views/usage/index.vue'),
        meta: { title: '用量分析', icon: 'DataAnalysis' }
      },
      {
        path: 'trace',
        name: 'Trace',
        component: () => import('@/views/trace/index.vue'),
        meta: { title: '请求追踪', icon: 'Connection' }
      }
    ]
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach((to, _from, next) => {
  const userStore = useUserStore()

  if (to.meta.requiresAuth !== false && !userStore.isLoggedIn) {
    next({
      path: '/login',
      query: { redirect: to.fullPath }
    })
  } else if (to.path === '/login' && userStore.isLoggedIn) {
    next({ path: '/' })
  } else {
    next()
  }
})

export default router
