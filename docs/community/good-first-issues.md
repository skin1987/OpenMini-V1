# Good First Issues

欢迎新手贡献者！以下是适合入门的任务，按难度排序。

## 🟢 入门级 (无需深度了解代码库)

### 文档改进类

- [ ] **#101**: 补充 `nsa.rs` 的 API 文档示例代码
  - 预计时间: 1小时
  - 涉及文件: `src/model/inference/nsa.rs`
  - 要求: 理解NSA基本概念

- [ ] **#102**: 创建 Quick Start Guide (中文版)
  - 预计时间: 2小时
  - 涉及文件: `docs/quickstart-zh.md`
  - 要求: 能运行项目

- [ ] **#103**: 改进错误消息的可读性
  - 预计时间: 2小时
  - 涉及文件: `src/model/inference/error.rs`
  - 要求: 了解Rust error handling

### 测试补充类

- [ ] **#201**: 为 `quant.rs` 添加边界值测试
  - 预计时间: 2小时
  - 涉及文件: `src/model/inference/quant.rs`
  - 要求: 了解量化基本原理

- [ ] **#202**: 为 `benchmark/config.rs` 添加序列化测试
  - 预计时间: 1小时
  - 涉及文件: `src/benchmark/config.rs`
  - 要求: 了解TOML/JSON

### 小Bug修复类

- [ ] **#301**: 修复 CLI help message 错别字
  - 预计时间: 30分钟
  - 涉及文件: `openmini-server/src/bin/openmini-server.rs`

- [ ] **#302**: 统一日志格式 (部分模块使用不同format)
  - 预计时间: 1小时
  - 涉及文件: 多个文件

## 🟡 初级 (需了解部分代码库)

### 新增语言/架构支持

- [ ] **#401**: 添加 Starling 模型的 GGUF 加载支持
  - 预计时间: 4小时
  - 参考: `gguf.rs` 中已有的架构实现
  - 要求: 理解GGUF格式

- [ ] **#402**: 添加中文 README.md
  - 预计时间: 2小时
  - 涉及文件: `README.md`

### 性能优化

- [ ] **#501**: 为 `continuous_batching.rs` 添加缓存友好的内存访问模式
  - 预计时间: 4小时
  - 要求: 了解CPU缓存机制

- [ ] **#502**: 实现 `tokenizer.rs` 的 LRU token cache
  - 预计时间: 3小时
  - 要求: 了解数据结构

## 🟠 中级 (可独立完成的feature)

### 新功能

- [ ] **#601**: 添加 Prometheus metrics endpoint
  - 预计时间: 8小时
  - 参考: 已有的 benchmark/metrics.rs

- [ ] **#602**: 实现 gRPC streaming 输出
  - 预计时间: 8小时
  - 参考: `service/grpc/server.rs`

- [ ] **#603**: 添加 WebUI 管理面板
  - 预计时间: 16小时
  - 前后端分离

## 如何开始

1. 选择一个 Issue
2. Comment "I'd like to work on this"
3. 等待 Maintainer 分配
4. 开始工作!

## 需要帮助?

- 在 Issue 下提问
- 加入 Discord: [OpenMini Community](https://discord.gg/openmini)
- 查看 [CONTRIBUTING.md](../CONTRIBUTING.md)
