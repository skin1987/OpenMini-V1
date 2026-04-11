# 贡献指南 (Contributing Guide)

感谢您对 OpenMini-V1 项目的关注！我们欢迎所有形式的贡献，包括但不限于代码提交、文档改进、问题报告和建议反馈。

## 🎯 项目简介

OpenMini-V1 是一个开源的大语言模型推理服务框架，致力于提供高效、易用、可扩展的本地化 AI 推理解决方案。我们的目标是让每个开发者都能轻松部署和使用大语言模型。

## 🤝 贡献理念

### 核心价值观

- **质量优先 (Quality First)**: 每一行代码都应经过深思熟虑，确保稳定性和可维护性
- **测试驱动 (Test-Driven Development)**: 新功能必须配备完善的单元测试和集成测试
- **文档先行 (Documentation First)**: API 变更和新功能必须有清晰的文档说明
- **社区协作 (Community Collaboration)**: 尊重每一位贡献者，保持开放包容的沟通氛围

### 贡献方式

1. **代码贡献**: 修复 Bug、添加新功能、性能优化
2. **文档贡献**: 改进文档、翻译文档、编写教程
3. **问题反馈**: 报告 Bug、提出功能建议、分享使用经验
4. **代码审查**: 帮助审查他人的 PR，提升代码质量

---

## 🛠️ 开发环境配置

### 系统要求

| 组件 | 最低版本 | 推荐版本 |
|------|---------|---------|
| Rust | 1.75+ | 最新 stable |
| Node.js | 18+ | 20 LTS |
| Python | 3.10+ | 3.11+ |
| CUDA (GPU) | 11.8+ | 12.1+ |
| 内存 | 16GB | 32GB+ |

### 环境安装步骤

#### 1. 安装 Rust 工具链

```bash
# 安装 rustup (如果尚未安装)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 添加到 PATH
source $HOME/.cargo/env

# 验证安装
rustc --version
cargo --version
```

#### 2. 安装 Node.js 和 pnpm

```bash
# 使用 nvm 安装 Node.js
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 20
nvm use 20

# 安装 pnpm
npm install -g pnpm
```

#### 3. 安装 Python 依赖

```bash
# 创建虚拟环境 (推荐)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装开发依赖
pip install openai httpx aiohttp pytest pytest-cov mypy ruff
```

#### 4. 克隆项目并构建

```bash
# 克隆仓库
git clone https://github.com/your-org/OpenMini-V1.git
cd OpenMini-V1

# 构建后端 (Rust)
cargo build --release

# 构建前端 (Vue)
cd frontend
pnpm install
pnpm build
cd ..

# 运行测试
cargo test
pnpm test
```

### IDE 配置推荐

#### VS Code (推荐)

安装以下扩展：

- **rust-analyzer**: Rust 语言支持
- **Volar**: Vue 3 开发支持
- **ESLint + Prettier**: 代码格式化和检查
- **Python**: Python 语言支持
- **Error Lens**: 内联错误显示

#### 配置文件

项目根目录包含以下配置文件：
- `.vscode/settings.json`: VS Code 工作区设置
- `.editorconfig`: 编辑器统一配置
- `.prettierrc`: Prettier 格式化规则

---

## 📝 代码规范

### Rust 代码规范

#### 命名约定

```rust
// 类型和 Trait: PascalCase
struct ChatCompletionRequest;
trait ModelLoader;

// 函数和方法: snake_case
fn create_chat_completion() {}
fn load_model_from_path() {}

// 常量: SCREAMING_SNAKE_CASE
const MAX_TOKEN_LIMIT: usize = 4096;

// 模块和文件: snake_case
mod chat_completions;
// 文件: chat_completions.rs
```

#### 错误处理

```rust
// ✅ 正确: 使用 thiserror 定义错误类型
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("模型未找到: {model_name}")]
    ModelNotFound { model_name: String },

    #[error("请求超时")]
    RequestTimeout,

    #[error("内部服务器错误: {0}")]
    Internal(String),
}

// ✅ 正确: 使用 ? 操作符传播错误
async fn process_request(req: Request) -> Result<Response, ApiError> {
    let validated = validate_request(req)?;
    let result = execute(validated).await?;
    Ok(result)
}

// ❌ 避免: 不要使用 unwrap() 在生产代码中
let value = option.unwrap(); // 危险!
```

#### Clippy 规则

项目强制执行以下 Clippy lint 规则：

```toml
# Cargo.toml 或 .clippy.toml
[lints.clippy]
# 正确性
all = "warn"
pedantic = "warn"

# 允许的例外 (需在 PR 中说明理由)
module_inception = "allow"
must_use_candidate = "allow"
```

运行 Clippy 检查：

```bash
# 检查所有 lint
cargo clippy --all-targets --all-features -- -D warnings

# 自动修复部分问题
cargo clippy --fix
```

### TypeScript/Vue 前端规范

#### TypeScript 规范

```typescript
// ✅ 使用接口定义类型
interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: Date;
}

// ✅ 使用 const 断言
const MODEL_OPTIONS = ['openmini-7b', 'openmini-13b'] as const;
type ModelName = typeof MODEL_OPTIONS[number];

// ❌ 避免使用 any
function processData(data: any) { ... } // 不推荐
function processData<T>(data: T): ProcessedData<T> { ... } // 推荐
```

#### Vue 组件规范

```vue
<template>
  <!-- 使用语义化的 HTML -->
  <section class="chat-container">
    <ChatMessage
      v-for="msg in messages"
      :key="msg.id"
      :message="msg"
      @retry="handleRetry"
    />
  </section>
</template>

<script setup lang="ts">
// Composition API + <script setup>
import { ref, computed } from 'vue';
import type { ChatMessage } from '@/types';

// Props 和 Emits 明确定义
const props = defineProps<{
  messages: ChatMessage[];
}>();

const emit = defineEmits<{
  retry: [messageId: string];
}>();
</script>

<style scoped>
/* 使用 scoped 避免样式污染 */
.chat-container {
  @apply flex flex-col gap-4;
}
</style>
```

### Commit Message 规范

采用 **Conventional Commits** 规范（中文描述）：

#### 格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

#### Type 列表

| Type | 描述 | 示例 |
|------|------|------|
| `feat` | 新功能 | `feat(chat): 添加流式输出支持` |
| `fix` | Bug 修复 | `fix(api): 修复长连接超时问题` |
| `docs` | 文档更新 | `docs(readme): 更新安装说明` |
| `style` | 代码格式 | `style(rust): 统一导入顺序` |
| `refactor` | 重构 | `refactor(loader): 重构模型加载逻辑` |
| `perf` | 性能优化 | `perf(inference): 减少 GPU 内存分配` |
| `test` | 测试相关 | `test(chat): 添加边界条件测试` |
| `chore` | 构建/工具 | `chore(deps): 更新依赖版本` |

#### 示例

```
feat(chat): 添加多轮对话上下文管理

- 实现对话历史自动截断逻辑
- 支持自定义上下文窗口大小
- 添加最大轮次限制配置

Closes #123
```

---

## 🧪 测试要求

### 单元测试覆盖率

- **最低要求**: >80% 行覆盖率
- **目标**: >90% 行覆盖率
- **关键路径**: 必须达到 100% 覆盖率（API 处理、安全验证）

#### Rust 测试示例

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_completion_validation() {
        // Arrange
        let request = ChatCompletionRequest::new("test-model", vec![
            Message::user("Hello")
        ]);

        // Act
        let result = validate_request(&request);

        // Assert
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_streaming_response() {
        // 异步测试示例
        let service = create_test_service().await;
        let response = service.stream_chat(request).await;

        assert!(response.is_ok());
        // 验证流式响应格式
    }
}
```

#### 运行测试

```bash
# 运行所有测试
cargo test

# 运行特定模块测试
cargo test --lib chat::

# 显示覆盖率
cargo tarpaulin --out Html

# 运行前端测试
pnpm test

# 运行 E2E 测试
pnpm test:e2e
```

### 集成测试编写规范

```rust
// tests/integration/chat_api.rs
use reqwest::Client;
use serde_json::json;

#[tokio::test]
async fn test_chat_completion_endpoint() {
    let client = Client::new();
    
    let response = client
        .post("http://localhost:8080/v1/chat/completions")
        .json(&json!({
            "model": "openmini-7b",
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .send()
        .await
        .expect("请求失败");

    assert_eq!(response.status(), 200);
    
    let body: serde_json::Value = response.json().await.unwrap();
    assert!(body["choices"].is_array());
}
```

### 性能基准测试

```rust
#[bench]
fn bench_token_generation(b: &mut Bencher) {
    let model = load_test_model();
    b.iter(|| {
        model.generate_tokens(black_box("测试输入"));
    });
}

// 运行基准测试
// cargo bench
```

---

## 🔀 PR 流程

### PR 标题格式

遵循 Conventional Commits 格式：

```
type(scope): description
```

**示例**:
- `feat(api): 添加 embedding 接口支持`
- `fix(ui): 修复移动端布局溢出问题`
- `docs(contributing): 补充测试指南章节`

### PR Checklist 模板

每次提交 PR 前，请确保完成以下检查：

```markdown
## PR Description

### 变更概述
[简要描述本次变更的内容和目的]

### 变更类型
- [ ] 🚀 新功能 (Feature)
- [ ] 🐛 Bug 修复 (Bug Fix)
- [ ] 📝 文档更新 (Documentation)
- [ ] 🎨 代码重构 (Refactoring)
- [ ] ⚡ 性能优化 (Performance)

### 测试情况
- [ ] 所有现有测试通过 (`cargo test` && `pnpm test`)
- [ ] 新增功能的单元测试覆盖率 >80%
- [ ] 集成测试已通过 (如适用)
- [ ] 手动测试已完成

### 代码质量
- [ ] 无新增 Clippy 警告 (`cargo clippy` 通过)
- [ ] 代码格式符合规范 (`cargo fmt` && `pnpm format`)
- [ ] 无 ESLint/Prettier 警告
- [ ] 大型 PR 已拆分为多个小 PR

### 文档
- [ ] API 文档已更新 (如适用)
- [ ] README/CHANGELOG 已更新 (如适用)
- [ ] 代码注释充分且清晰
- [ ] 用户可见变更的迁移指南 (如适用)

### 其他
- [ ] Commit message 符合规范
- [ ] 分支基于最新的 main 分支
- [ ] 合并冲突已解决
```

### Reviewer 分配规则

根据 PR 的修改范围自动分配审查者：

| 修改范围 | 审查者 | 说明 |
|---------|--------|------|
| 核心 API / Rust 后端 | @maintainer-core | 后端维护者 |
| 前端 UI / Vue 组件 | @maintainer-frontend | 前端维护者 |
| 文档 / 示例 | @maintainer-docs | 文档维护者 |
| CI/CD / DevOps | @maintainer-devops | DevOps 维护者 |

### 合并策略

本项目采用 **Squash Merge** 策略：

1. 保持 git 历史整洁
2. 每个 squash commit 对应一个完整的 feature/fix
3. Squash 后的 commit message 应清晰描述变更内容

---

## 👥 社区准则

### 行为准则

我们的项目遵循 [Contributor Covenant](https://www.contributorcovenant.org/) 行为准则 v2.1：

- **尊重他人**: 以建设性的方式交流，尊重不同观点
- **包容开放**: 欢迎不同背景的贡献者参与
- **专业协作**: 保持专业态度，专注于技术讨论
- **互助成长**: 帮助新人成长，分享知识和经验

**违规行为举报**: 请发送邮件至 [maintainers@openmini.ai](mailto:maintainers@openmini.ai)

### 沟通语言

- **主要语言**: 中文 (简体)
- **次要语言**: English (国际化交流)
- **Issue/PR**: 建议使用中英双语标题，方便全球开发者理解
- **代码注释**: 英文为主，复杂逻辑可用中文补充说明

### Issue / 讨论区使用规范

#### 提交 Issue 前检查

1. **搜索现有 Issue**: 避免重复提交
2. **使用模板**: 完整填写 Issue 模板
3. **提供复现信息**: 包含环境信息、日志、截图等
4. **最小化复现**: 提供最简复现步骤

#### Issue 标签分类

| 标签 | 用途 | 示例 |
|------|------|------|
| `bug` | Bug 报告 | 内存泄漏、崩溃 |
| `feature` | 功能需求 | 新模型支持、API 扩展 |
| `documentation` | 文档问题 | 缺失文档、错误说明 |
| `good first issue` | 新手友好 | 简单的 Bug 修复 |
| `help wanted` | 需要帮助 | 功能实现需要协助 |

#### Discussion 使用场景

- 💡 **想法分享**: 功能设计讨论、架构方案
- ❓ **使用帮助**: 部署问题、配置疑问
- 📢 **公告通知**: 版本发布、重要变更
- 🎉 **展示分享**: 使用案例、性能优化经验

---

## 📞 联系方式

- **GitHub Discussions**: [讨论区链接](https://github.com/your-org/OpenMini-V1/discussions)
- **Issue**: [问题追踪](https://github.com/your-org/OpenMini-V1/issues)
- **邮箱**: [dev@openmini.ai](mailto:dev@openmini.ai)

---

## 🙏 致谢

感谢每一位为 OpenMini-V1 做出贡献的开发者！您的每一行代码、每一个建议都在推动项目向前发展。

特别感谢：
- 核心贡献者团队
- 文档翻译志愿者
- 测试和反馈的用户

---

*最后更新: 2026-04-10*
