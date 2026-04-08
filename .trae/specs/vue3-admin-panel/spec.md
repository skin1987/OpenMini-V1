# Vue 3 大模型运维管理系统 Spec

## Why

OpenMini 项目当前只有 Rust 推理服务端（业务 API）和 Python 客户端，**缺少专业的后台运维管理界面**。需要构建一个功能丰富、专业的大模型运维管理系统，实现对推理服务的全方位监控、管理和运营。

## What Changes

- **新增**: `openmini-admin-web/` — Vue 3 + TypeScript + Vite 前端项目
- **新增**: 10+ 专业运维页面模块（监控仪表盘、服务管理、模型管理、API Key 管理、用户权限、告警中心、日志中心、配置管理、用量分析、请求追踪）
- **新增**: 与现有 `openmini-server` HTTP API 对接层
- **新增**: 实时数据推送（WebSocket/SSE）支持

## Impact

- Affected specs: 无（全新模块）
- Affected code:
  - `openmini-server/src/service/http/handlers.rs` — 现有 API 端点（只读对接，不修改）
  - `openmini-server/src/monitoring/metrics.rs` — Prometheus 指标对接
  - `openmini-server/src/monitoring/health_check.rs` — 健康检查对接
  - `config/server.toml` — 服务配置对接参考

## ADDED Requirements

### Requirement: 项目初始化与基础架构

系统 SHALL 使用以下技术栈初始化 Vue 3 项目：

- Vue 3.4+ (Composition API + `<script setup>`)
- Vite 6 构建工具
- TypeScript 5.x
- Element Plus UI 组件库
- Pinia 状态管理
- Vue Router 4（含权限路由守卫）
- Axios HTTP 客户端
- ECharts 5 图表库
- SCSS 样式预处理

#### Scenario: 项目创建成功

- **WHEN** 执行 `npm create vite@latest` 初始化项目并安装依赖
- **THEN** 项目可成功启动 (`npm run dev`)，默认端口 5173 可访问

### Requirement: 监控仪表盘 (Dashboard)

系统 SHALL 提供实时监控仪表盘页面，展示以下核心指标：

1. **系统资源面板**
   - GPU 利用率实时曲线（折线图）
   - GPU 显存使用量（进度条 + 数值）
   - CPU 使用率曲线
   - 内存使用量（含 KV Cache 占比）
2. **推理性能面板**
   - QPS（每秒请求数）实时数值 + 趋势
   - 推理延迟 P50/P95/P99 直方图
   - Token 吞吐量（tokens/s）
   - 活跃连接数 / 批处理大小
3. **健康状态总览**
   - 各组件状态指示灯（GPU/Memory/CPU/Scheduler/Model）
   - 服务运行时长 / 版本号
   - 最近告警摘要列表（最新 5 条）

#### Scenario: 仪表盘正常渲染

- **WHEN** 用户访问 Dashboard 页面
- **THEN** 所有图表组件正确加载，数据从 `/api/v1/health` 和 `/api/v1/metrics` 自动刷新（默认 10s 间隔）

### Requirement: 服务管理模块

系统 SHALL 提供服务管理功能：

1. **服务状态总览页**
   - 显示服务版本、监听地址/端口、运行时长
   - Worker 进程列表（ID/状态/PID/启动时间/重启次数）
   - 连接数统计（当前/峰值/最大限制）
2. **Worker 管理**
   - 查看每个 Worker 的详细状态
   - 单个 Worker 重启操作（需 operator 权限）
3. **服务控制**
   - 优雅重启按钮（需 admin 权限，二次确认）
   - 优雅停止按钮（需 admin 权限，二次确认）

#### Scenario: 服务状态查询

- **WHEN** 用户进入服务管理页面
- **THEN** 从 `/api/v1/health` 获取聚合健康状态并展示各组件详情

### Requirement: 模型管理中心

系统 SHALL 提供模型全生命周期管理：

1. **模型注册表**
   - 展示所有可用模型（扫描结果或手动注册）
   - 字段：名称/文件路径/大小/量化类型/上下文长度/状态(loaded/unloaded/error)
   - 支持搜索和筛选
2. **模型操作**
   - 加载模型到内存（operator 权限）
   - 卸载模型释放显存（operator 权限）
   - 热切换模型（无缝替换，需确认）
   - 模型健康检查按钮
3. **模型详情页**
   - 当前加载模型的详细参数
   - 显存占用、KV Cache 大小
   - 加载时间戳

#### Scenario: 模型列表展示

- **WHEN** 用户访问模型管理页面
- **THEN** 从 `/api/v1/models` 获取模型列表并以表格形式展示

### Requirement: API Key 管理模块

系统 SHALL 提供 API Key 全生命周期管理：

1. **Key 列表**
   - Key 名称/前缀(om-sk_xxxx)/创建者/状态/创建时间/过期时间/最近使用
   - 支持按状态筛选（活跃/已禁用/已过期）
2. **Key 操作**
   - 创建新 Key（自定义名称 + 过期时间 + 配额设置）
   - 禁用/启用 Key
   - 废弃撤销 Key
3. **用量统计**
   - 每个 Key 的今日请求数 / 本月 Token 用量
   - 配额使用率进度条
   - 用量趋势图（近 7 天/30 天）

#### Scenario: 创建 API Key

- **WHEN** 用户填写 Key 名称和可选配额后提交
- **THEN** 系统生成新 Key 并在界面上完整显示（仅此一次可复制）

### Requirement: 用户权限管理模块

系统 SHALL 提供 RBAC 用户管理：

1. **用户列表**
   - 用户名/邮箱/角色/状态/最后登录时间/创建时间
   - 分页 + 搜索
2. **用户操作**
   - 创建用户（用户名/邮箱/密码/角色分配）
   - 编辑用户信息
   - 修改用户角色
   - 启用/禁用用户
   - 重置密码
3. **三角色体系**
   - `admin`: 全部权限
   - `operator`: 服务/模型/Key/告警操作权限（不含用户管理和配置修改）
   - `viewer`: 只读查看权限

#### Scenario: 角色权限控制

- **WHEN** viewer 角色用户访问用户管理页面
- **THEN** 页面隐藏所有编辑/删除/创建按钮，仅显示只读表格

### Requirement: 告警中心模块

系统 SHALL 提供完整的告警管理能力：

1. **告警规则管理**
   - 规则列表：名称/指标/条件/阈值/持续时间/级别/状态
   - 内置规则模板：
     - GPU 利用率 > 90% 持续 5min → critical
     - 显存使用 > 95% → critical
     - 推理延迟 P99 > 5000ms → warning
     - 错误率 > 5% 持续 3min → warning
     - 调度队列长度 > 100 → warning
     - KV Cache 使用 > 90% → info
   - 创建/编辑/启停规则
2. **告警记录**
   - 记录列表：规则名/级别/消息/值/触发时间/状态(firing/acknowledged/resolved)
   - 操作：确认告警 / 解决告警
   - 筛选：按级别/状态/时间范围过滤
3. **告警统计**
   - 各级别告警数量统计
   - 近 24h/7d/30d 告警趋势

#### Scenario: 告警触发与通知

- **WHEN** 监控指标满足告警规则条件且持续超过阈值时间
- **THEN** 系统生成 firing 状态告警记录并在仪表盘高亮显示

### Requirement: 日志中心模块

系统 SHALL 提供结构化日志查看能力：

1. **日志查询**
   - 日志列表：时间戳/级别(INFO/WARN/ERROR)/来源模块/消息内容
   - 过滤条件：时间范围/日志级别/关键词搜索
   - 分页加载
2. **实时日志流**
   - SSE 实时推送最新日志
   - 可暂停/恢复
   - 自动滚动到底部
3. **日志导出**
   - 导出为 CSV / JSON 格式
   - 支持选择时间范围

#### Scenario: 日志查询

- **WHEN** 用户选择时间范围和过滤条件后点击查询
- **THEN** 返回匹配的日志记录并分页展示

### Requirement: 配置管理模块

系统 SHALL 提供在线配置管理：

1. **配置查看**
   - 以结构化表单展示当前 TOML 配置（脱敏处理敏感字段）
   - 按 server/thread_pool/memory/model/worker/grpc 分组展示
2. **配置编辑**
   - 在线编辑各配置项
   - JSON Schema 校验输入合法性
3. **配置热重载**
   - 一键重载配置按钮（admin 权限）
   - 变更历史记录（谁/什么时候/改了什么）

#### Scenario: 配置变更审计

- **WHEN** admin 用户修改了配置并提交
- **THEN** 记录变更历史包含旧值/新值/操作人/时间/原因

### Requirement: 用量分析模块

系统 SHALL 提供用量统计分析：

1. **Token 用量趋势**
   - 日/周/月粒度的 Token 用量折线图
   - Prompt Tokens vs Completion Tokens 对比
2. **调用分布**
   - 按模型/接口类型的调用次数分布饼图
   - Top 10 高频调用用户/API Key
3. **成本估算**
   - 基于 Token 用量的成本估算（可配置单价）
   - 月度成本报表

#### Scenario: 用量数据展示

- **WHEN** 用户访问用量分析页面
- **THEN** 自动加载最近 30 天的用量数据并渲染图表

### Requirement: 请求链路追踪模块

系统 SHALL 提供请求级别的追踪分析：

1. **请求列表**
   - 请求 ID / 时间 / 接口 / 状态码 / 延迟 / Token 数 / 模型
   - 排序/筛选/分页
2. **请求详情**
   - 完整请求/响应内容（格式化展示）
   - 时间线分解（排队/推理/编码耗时）
3. **慢请求分析**
   - Top N 慢请求排行
   - 错误请求列表及错误分类统计

#### Scenario: 请求详情查看

- **WHEN** 点击某条请求记录
- **THEN** 弹出详情面板展示完整请求信息和耗时分解

### Requirement: 认证与权限体系

系统 SHALL 实现安全的认证授权：

1. **登录认证**
   - 用户名 + 密码登录
   - JWT Token 存储（localStorage）
   - Token 自动刷新机制
   - 登录路由守卫（未登录跳转登录页）
2. **权限控制**
   - 菜单级权限（根据角色隐藏无权菜单）
   - 按钮级权限（自定义 directive `v-permission`）
   - API 请求级权限（403 自动提示无权限）
3. **会话管理**
   - 退出登录清除 Token
   - 多标签页同步登出
   - Token 过期自动跳转登录

#### Scenario: 未授权访问拦截

- **WHEN** 未登录用户直接访问管理页面 URL
- **THEN** 自动重定向到登录页，登录后跳回原页面

### Requirement: 布局与交互规范

系统 SHALL 遵循企业级后台管理系统的布局标准：

1. **整体布局**
   - 左侧固定宽度侧边栏导航（可折叠）
   - 顶部 Header（面包屑/用户头像/退出/全屏切换）
   - 主内容区域（自适应）
2. **响应式适配**
   - 支持 1280px+ 分辨率
   - 侧边栏折叠时自适应
3. **交互规范**
   - 所有异步操作显示 loading 状态
   - 操作成功/失败有 toast 提示
   - 危险操作（删除/重启/停止）需二次确认弹窗
   - 表格支持排序/筛选/批量操作

#### Scenario: 布局正常渲染

- **WHEN** 用户打开任意管理页面
- **THEN** 侧边栏导航 + 顶部 Header + 内容区域正确渲染，路由高亮当前菜单

## MODIFIED Requirements

无（全新模块，不修改现有代码）

## REMOVED Requirements

无
