# Tasks

- [x] Task 1: 项目初始化与基础架构搭建
  - [x] 使用 Vite 6 创建 Vue 3 + TypeScript 项目 `openmini-admin-web`
  - [x] 安装核心依赖：element-plus, pinia, vue-router, axios, echarts, sass, @element-plus/icons-vue
  - [x] 配置 TypeScript、路径别名 (@/ → src/)、SCSS 全局变量
  - [x] 配置 ESLint + Prettier 代码规范
  - [x] 验证：`npm run dev` 可正常启动

- [x] Task 2: 布局系统与路由框架
  - [x] 实现 AdminLayout 组件（侧边栏 + 顶部 Header + 主内容区）
  - [x] 侧边栏导航菜单（含图标、折叠功能、路由高亮）
  - [x] 顶部 Header（面包屑、用户下拉菜单、全屏切换）
  - [x] Vue Router 路由配置（嵌套路由、懒加载）
  - [x] 路由守卫（未登录重定向登录页）
  - [x] 验证：页面布局正确渲染，路由切换正常

- [x] Task 3: 认证模块 (Login + JWT)
  - [x] 登录页面 UI（用户名/密码表单，Element Plus 表单校验）
  - [x] Pinia user store（Token 存储/获取/清除）
  - [x] Axios 拦截器（请求带 Token / 401 自动跳转登录 / 403 无权限提示）
  - [x] Token 刷新机制
  - [x] 自定义权限指令 v-permission
  - [x] 验证：可正常登录/退出，未登录访问页面自动跳转

- [x] Task 4: API 对接层封装
  - [x] Axios 实例创建（baseURL, timeout, interceptors）
  - [x] 封装各模块 API 调用：
    - service.ts — 服务管理相关接口
    - model.ts — 模型管理相关接口
    - monitoring.ts — 监控指标相关接口
    - alert.ts — 告警相关接口
    - user.ts — 用户管理接口
    - apikey.ts — API Key 管理接口
    - config.ts — 配置管理接口
    - audit.ts — 审计日志接口
  - [x] 通用请求/响应类型定义
  - [x] 错误处理统一封装

- [ ] Task 5: 监控仪表盘 (Dashboard)
  - [ ] Dashboard 页面骨架搭建
  - [ ] 系统资源面板组件（GPU/CPU/内存卡片 + ECharts 实时曲线）
  - [ ] 推理性能面板组件（QPS/延迟直方图/吞吐量/活跃连接数）
  - [ ] 健康状态总览组件（组件状态指示灯 + 服务信息卡片）
  - [ ] 最近告警摘要列表组件
  - [ ] 数据定时刷新机制（10s 间隔，可配置）
  - [ ] 验证：所有图表正确渲染并展示实时数据

- [ ] Task 6: 服务管理模块
  - [ ] 服务状态总览页（版本/地址/运行时长/连接数统计）
  - [ ] Worker 进程列表表格（ID/状态/PID/启动时间/重启次数）
  - [ ] 单个 Worker 重启操作（二次确认弹窗）
  - [ ] 服务优雅重启/停止按钮（admin 权限 + 二次确认）
  - [ ] 验证：服务状态正确展示，操作按钮按权限显示

- [ ] Task 7: 模型管理中心
  - [ ] 模型列表页（表格展示 + 搜索/筛选）
  - [ ] 模型加载/卸载/热切换操作按钮
  - [ ] 模型详情面板（参数/显存/KV Cache/加载时间）
  - [ ] 模型健康检查按钮及结果展示
  - [ ] 操作确认弹窗（热切换需二次确认）
  - [ ] 验证：模型 CRUD 操作完整可用

- [ ] Task 8: API Key 管理模块
  - [ ] Key 列表页（表格 + 状态标签 + 筛选）
  - [ ] 创建 Key 弹窗（名称/过期时间/配额设置）
  - [ ] Key 创建成功后一次性完整展示（可复制）
  - [ ] Key 启用/禁用/废弃操作
  - [ ] 用量统计面板（请求数/Token数/配额进度条/趋势图）
  - [ ] 验证：Key 全生命周期管理完整

- [ ] Task 9: 用户权限管理模块
  - [ ] 用户列表页（分页表格 + 搜索）
  - [ ] 创建/编辑用户弹窗（用户名/邮箱/密码/角色选择）
  - [ ] 角色修改操作
  - [ ] 用户启用/禁用/重置密码
  - [ ] 权限控制验证：viewer 角色隐藏编辑按钮
  - [ ] 验证：RBAC 三角色权限体系生效

- [ ] Task 10: 告警中心模块
  - [ ] 告警规则列表页（CRUD 表格 + 内置模板快速创建）
  - [ ] 规则创建/编辑弹窗（指标选择/条件/阈值/持续时间/级别/通知渠道）
  - [ ] 规则启停开关
  - [ ] 告警记录列表（筛选/分页/状态标签）
  - [ ] 告警确认/解决操作
  - [ ] 告警统计面板（各级别数量 + 趋势图）
  - [ ] 验证：告警规则触发和记录流转完整

- [ ] Task 11: 日志中心模块
  - [ ] 日志查询页（时间范围选择器/级别筛选/关键词搜索）
  - [ ] 日志列表（分页 + 级别颜色标识）
  - [ ] 实时日志流（SSE 连接 + 自动滚动 + 暂停恢复）
  - [ ] 日志导出功能（CSV/JSON 格式选择）
  - [ ] 验证：日志查询和实时流正常工作

- [ ] Task 12: 配置管理模块
  - [ ] 配置查看页（分组结构化表单展示，脱敏敏感字段）
  - [ ] 配置编辑模式（可编辑字段高亮）
  - [ ] 配置保存 + 热重载按钮
  - [ ] 配置变更历史列表（谁/时间/变更内容 diff）
  - [ ] 验证：配置查看/编辑/历史记录完整

- [ ] Task 13: 用量分析模块
  - [ ] Token 用量趋势图（日/周/月粒度切换）
  - [ ] Prompt vs Completion Tokens 对比图
  - [ ] 调用分布饼图（按模型/接口类型）
  - [ ] Top 排行榜（高频用户/API Key）
  - [ ] 成本估算报表（月度汇总）
  - [ ] 验证：图表数据正确渲染

- [ ] Task 14: 请求链路追踪模块
  - [ ] 请求列表页（ID/时间/接口/状态码/延迟/Token 数/模型，排序筛选分页）
  - [ ] 请求详情抽屉/弹窗（完整请求响应 + 耗时分解时间线）
  - [ ] 慢请求 Top N 排行
  - [ ] 错误请求列表及分类统计
  - [ ] 验证：请求追踪数据完整展示

- [ ] Task 15: 公共组件库
  - [ ] StatusBadge 状态标签组件（success/warning/danger/info 变体）
  - [ ] StatCard 统计卡片组件（数值 + 趋势箭头 + 图标）
  - [ ] DataTable 通用表格组件（排序/筛选/分页/批量操作封装）
  - [ ] ConfirmDialog 危险操作确认弹窗组件
  - [ ] SearchBar 通用搜索栏组件
  - [ ] PageHeader 页面标题 + 操作按钮区域组件
  - [ ] 验证：公共组件在各页面复用正常

- [ ] Task 16: 全局样式与主题
  - [ ] SCSS 变量体系（主题色/间距/圆角/阴影）
  - [ ] Element Plus 主题定制（覆盖 CSS 变量）
  - [ ] 暗色模式支持（CSS 变量切换）
  - [ ] 全局动画过渡效果
  - [ ] 响应式断点适配
  - [ ] 验证：视觉风格统一一致

# Task Dependencies

- [Task 2] depends on [Task 1]
- [Task 3] depends on [Task 2]
- [Task 4] depends on [Task 1]
- [Task 5] depends on [Task 3], [Task 4]
- [Task 6] depends on [Task 3], [Task 4]
- [Task 7] depends on [Task 3], [Task 4]
- [Task 8] depends on [Task 3], [Task 4]
- [Task 9] depends on [Task 3], [Task 4]
- [Task 10] depends on [Task 3], [Task 4]
- [Task 11] depends on [Task 3], [Task 4]
- [Task 12] depends on [Task 3], [Task 4]
- [Task 13] depends on [Task 4]
- [Task 14] depends on [Task 4]
- [Task 15] depends on [Task 1]
- [Task 16] depends on [Task 1]
- [Task 5~14] 可并行开发（依赖 Task 3+4 完成后）
- [Task 15, 16] 可与 Task 2 并行开发
