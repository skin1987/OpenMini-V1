# OpenMini-V1 分支保护规则配置指南

本文档提供完整的分支保护策略和代码审查流程配置指南。

---

## 📋 分支保护规则概览

### 推荐的分支结构

```
main (生产环境)
  └── develop (开发环境) [可选]
        ├── feature/* (新功能开发)
        ├── fix/* (Bug 修复)
        ├── perf/* (性能优化)
        └── hotfix/* (紧急修复，直接合并到 main)
```

---

## 🔒 main 分支保护规则（必须配置）

### 配置位置

GitHub Repository → Settings → Branches → Add branch protection rule

### 规则配置

#### 基本设置

| 设置项 | 推荐值 | 说明 |
|--------|-------|------|
| **Branch name pattern** | `main` | 保护主分支 |
| **Require a pull request before merging** | ✅ 启用 | 强制使用 PR 合并 |
| **Require approvals** | ✅ 启用 | 需要代码审查 |
| **Number of approvals required** | `1` 或 `2` | 至少 1-2 人审批 |
| **Dismiss stale pull request approvals when new commits are pushed** | ✅ 启用 | 新提交需重新审查 |
| **Require review from CODEOWNERS** | ⚠️ 可选 | 关键模块需要特定人审查 |

#### CI/CD 必须通过

| 设置项 | 推荐值 | 说明 |
|--------|-------|------|
| **Require status checks to pass before merging** | ✅ 启用 | 强制 CI 通过 |
| **Required status checks** | 选择以下检查项： | |

**必需的状态检查：**

- [x] `lint` - Lint 和格式检查
- [x] `build` - 构建验证（至少一个平台）
- [x] `test` - 测试套件

**可选状态检查：**

- [ ] `security-audit` - 安全审计（非阻塞）
- [ ] `benchmark` - 性能基准测试（仅 main 分支）

#### 其他限制

| 设置项 | 推荐值 | 说明 |
|--------|-------|------|
| **Require signed commits** | ⚠️ 可选 | 要求 GPG 签名（严格模式） |
| **Require linear history** | ✅ 推荐 | 使用 squash merge 保持历史整洁 |
| **Allow force pushes** | ❌ 禁止 | 防止历史被篡改 |
| **Allow deletions** | ❌ 禁止 | 防止主分支被误删 |
| **Include administrators** | ✅ 启用 | 管理员也受规则约束 |

---

## 🔧 develop 分支保护规则（可选）

如果使用 develop 分支作为集成分支：

### 规则配置

| 设置项 | 推荐值 | 说明 |
|--------|-------|------|
| **Branch name pattern** | `develop` | 保护开发分支 |
| **Require a pull request before merging** | ✅ 启用 | 同样需要 PR |
| **Number of approvals required** | `1` | 至少 1 人审批即可 |
| **Require status checks to pass before merging** | ✅ 启用 | lint + build 必须 pass |
| **Allow force pushes** | ❌ 禁止 | |
| **Allow deletions** | ❌ 禁止 | |

---

## 👥 CODEOWNERS 文件配置

创建 `.github/CODEOWNERS` 文件指定关键文件的审查者：

```gitowners
# 全局默认所有者
* @skin1987

# 核心推理引擎需要核心团队审查
/openmini-server/src/model/ @core-team
/openmini-server/src/kernel/ @gpu-team

# GPU 加速模块
/openmini-server/src/hardware/gpu/ @gpu-team

# 服务层
/openmini-server/src/service/ @backend-team

# Python 客户端
/openmini-client/ @python-team

# 协议定义
/openmini-proto/ @protocol-team

# CI/CD 配置
/.github/ @devops-team
```

---

## 🔄 合并策略建议

### 推荐的合并方式：Squash and Merge

**优点：**
- 保持主分支历史整洁
- 每个 commit 对应一个功能点
- 易于代码回溯和问题定位

**配置方法：**
1. Settings → General → Pull Requests
2. 勾选 "Allow squash merging"
3. 取消勾选 "Allow merge commits" 和 "Allow rebase merging"

### 分支命名规范

```bash
# 功能开发
feature/<module>-<description>
示例: feature/metal-batch-optimization

# Bug 修复
fix/<module>-<issue-number>
示例: fix/kv-cache-memory-leak-123

# 性能优化
perf/<module>-<metric>
示例: perf/inference-latency-reduction

# 紧急修复（直接合入 main）
hotfix/<critical-issue>
示例: hotfix/crash-on-startup

# 重构
refactor/<module>-<what>
示例: refactor/memory-management-cleanup
```

---

## ✅ PR 审查清单模板

每个 PR 应该包含以下内容：

### 提交前自查

- [ ] 代码已通过 `cargo fmt --all` 格式化
- [ ] 代码已通过 `cargo clippy --all-targets` 检查
- [ ] 所有测试通过 (`cargo test --workspace`)
- [ ] 新功能有对应的单元测试
- [ ] API 变更有文档更新
- [ ] 性能变更包含基准测试数据
- [ ] 无敏感信息泄露（密钥、密码等）
- [ ] Commit message 符合 Conventional Commits 规范

### Reviewer 审查要点

1. **代码质量**
   - 是否遵循项目编码规范？
   - 是否有适当的错误处理？
   - 是否有性能隐患？

2. **安全性**
   - 是否引入安全漏洞？
   - 输入是否经过验证？
   - 是否有注入风险？

3. **可维护性**
   - 代码是否易于理解？
   - 注释是否充分？
   - 是否过度复杂？

4. **测试覆盖**
   - 新代码是否有测试？
   - 边界情况是否覆盖？
   - 集成测试是否足够？

---

## 🚀 自动化工作流集成

### GitHub Actions 与分支保护的配合

我们的 CI/CD 工作流（`.github/workflows/ci-cd.yml`）已经配置好以下状态检查：

1. **lint** - 格式化和静态分析
2. **build** - 多平台构建验证
3. **test** - 单元测试和集成测试
4. **security-audit** - 依赖安全审计
5. **benchmark** - 性能基准测试（仅 main 分支）

这些状态检查会自动出现在 PR 页面，分支保护规则可以强制要求它们全部通过才能合并。

---

## 📊 监控和报告

### 推荐使用的 GitHub 功能

1. **Branch protection violations**
   - 在 Actions 中查看违规记录
   - 定期审查强制推送或绕过规则的记录

2. **Code review metrics**
   - 查看 Insights → Contributors 了解贡献情况
   - 使用 Pull requests 统计了解审查效率

3. **Dependency updates**
   - 开启 Dependabot 自动更新依赖
   - 配置安全告警通知

---

## 🔔 通知配置

建议开启以下通知：

- **PR 活动**：新 PR 创建、状态变化
- **Review 请求**：被请求审查时
- **CI 失败**：构建或测试失败
- **Security alerts**：安全漏洞发现
- **Dependabot alerts**：依赖安全问题

---

## 📝 快速配置命令行工具

如果你喜欢使用 CLI，可以使用 GitHub CLI (`gh`)：

```bash
# 安装 gh CLI
brew install gh  # macOS
# 或访问 https://cli.github.com/

# 登录
gh auth login

# 配置 main 分支保护规则
gh api repos/skin1987/OpenMini-V1/branches/main/protection \
  -X PUT \
  -f required_status_checks='{"strict":true,"contexts":["lint","build","test"]}' \
  -f enforce_admins=true \
  -f required_pull_request_reviews='{"dismiss_stale_reviews":true,"required_approving_review_count":1}' \
  -f restrictions=null \
  -f allow_force_pushes=false \
  -f allow_deletions=false
```

---

## 🎯 最佳实践总结

1. **始终从最新代码开始**：在创建分支前先 `git pull origin main`
2. **小步提交**：保持每次 PR 聚焦单一功能
3. **及时响应 Review**：收到反馈后尽快修改
4. **保持 PR 更新**：如有冲突及时 rebase
5. **写好 commit message**：遵循规范，便于追溯
6. **删除已合并分支**：保持仓库整洁

---

## 📚 相关资源

- [GitHub 官方文档 - 分支保护](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pullrequests/about-protected-branches)
- [Conventional Commits 规范](https://www.conventionalcommits.org/)
- [GitHub Actions 文档](https://docs.github.com/en/actions)

---

**最后更新时间**: 2026-04-09
**维护者**: OpenMini Team
