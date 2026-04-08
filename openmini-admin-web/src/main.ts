import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'

import App from './App.vue'
import router from './router'
import './style.css'

import '@/styles/global.scss'
import StatCard from '@/components/common/StatCard.vue'
import StatusBadge from '@/components/common/StatusBadge.vue'
import PageHeader from '@/components/common/PageHeader.vue'
import SearchBar from '@/components/common/SearchBar.vue'
import ConfirmDialog from '@/components/common/ConfirmDialog.vue'

// 创建 Vue 应用实例
const app = createApp(App)

// 注册 Pinia 状态管理
app.use(createPinia())

// 注册 Vue Router
app.use(router)

// 注册 Element Plus UI 组件库
app.use(ElementPlus, { size: 'default', zIndex: 3000 })

// 全局注册所有 Element Plus 图标组件
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}

// 全局注册公共组件
app.component('StatCard', StatCard)
app.component('StatusBadge', StatusBadge)
app.component('PageHeader', PageHeader)
app.component('SearchBar', SearchBar)
app.component('ConfirmDialog', ConfirmDialog)

// 挂载应用
app.mount('#app')
