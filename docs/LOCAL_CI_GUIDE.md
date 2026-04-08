# OpenMini-V1 本地 CI/CD 开发指南

## 📖 关于 act (nektos/act)

**act** 是一个可以在本地运行 GitHub Actions 的工具，让你无需推送到 GitHub 就能测试 CI/CD 流水线。

### ✨ 核心优势

- ⚡ **极速反馈** - 本地运行，秒级响应（vs GitHub 的分钟级）
- 💰 **节省配额** - 免费版 2000 分钟/月，本地运行不消耗
- 🐛 **快速调试** - 即时看到错误日志，方便修复
- 🔒 **离线工作** - 无需网络连接也能测试

---

## 🚀 快速开始

### 前置条件

1. **安装 Docker Desktop** (macOS)
   ```bash
   # 下载并安装: https://www.docker.com/products/docker-desktop/
   ```

2. **配置 act 工具**（已下载到 `~/Downloads/`）

3. **解除 macOS 安全限制**
   ```bash
   xattr -d com.apple.quarantine ~/Downloads/act_Darwin_x86_64/act
   ```

### 使用方法

#### 方式一：使用便捷脚本（推荐）

```bash
cd /Users/apple/Desktop/OpenMini-V1

# 查看帮助
./scripts/local-ci.sh help

# 运行 Lint 检查
./scripts/local-ci.sh lint

# 运行测试套件
./scripts/local-ci.sh test

# 完整 CI 流水线
./scripts/local-ci.sh full

# 列出所有可用的 Jobs
./scripts/local-ci.sh list

# 模拟运行（不实际执行）
./scripts/local-ci.sh dry-run
```

#### 方式二：直接使用 act 命令

```bash
cd /Users/apple/Desktop/OpenMini-V1

# 查看所有可用的 jobs
~/Downloads/act_Darwin_x86_64/act -l

# 运行特定的 job
~/Downloads/act_Darwin_x86_64/act -j lint
~/Downloads/act_Darwin_x86_64/act -j build
~/Downloads/act_Darwin_x86_64/act -j test

# 运行完整工作流
~/Downloads/act_Darwin_x86_64/act

# 模拟运行（dry-run 模式）
~/Downloads/act_Darwin_x86_64/act -n -v

# 使用特定环境变量
~/Downloads/act_Darwin_x86_64/act --env-file .env.local
```

---

## 📋 可用的 CI Jobs

基于 `.github/workflows/ci-cd.yml` 配置：

| Job 名称 | 功能 | 运行时间 | 推荐场景 |
|---------|------|---------|---------|
| `lint` | Lint + 格式检查 | ~30 秒 | 每次提交前 |
| `build` | 多平台构建验证 | ~2-5 分钟 | 修改依赖后 |
| `test` | 单元+集成测试 | ~3-8 分钟 | 修改代码后 |
| `security-audit` | 安全审计 | ~1-2 分钟 | 定期检查 |
| `benchmark` | 性能基准测试 | ~5-10 分钟 | 仅 main 分支 |

---

## 🎯 典型工作流程

### 场景 1: 开发新功能时

```bash
# 1. 编写代码
vim src/some_module.rs

# 2. 运行 Lint 检查（快速验证）
./scripts/local-ci.sh lint

# 3. 如果通过，运行测试
./scripts/local-ci.sh test

# 4. 全部通过后提交
git add .
git commit -m "feat(module): 新功能描述"

# 5. 推送到 GitHub（触发真实 CI）
git push origin feature/new-feature
```

### 场景 2: 修复 CI 失败时

```bash
# 1. 查看 GitHub Actions 报错
#    （假设 lint job 失败）

# 2. 本地复现问题
./scripts/local-ci.sh lint

# 3. 根据错误信息修复代码
vim src/problematic_file.rs

# 4. 再次本地验证
./scripts/local-ci.sh lint

# 5. 确认无误后推送修复
git commit -m "fix(lint): 修复 clippy 警告"
git push
```

### 场景 3: 修改 CI 配置时

```bash
# 1. 编辑 CI 配置
vim .github/workflows/ci-cd.yml

# 2. 查看将运行的步骤（不执行）
./scripts/local-ci.sh dry-run

# 3. 确认配置正确后完整运行
./scripts/local-ci.sh full

# 4. 推送新配置
git add .github/workflows/ci-cd.yml
git commit -m "ci: 更新 CI 配置"
git push
```

---

## 🔧 高级配置

### 自定义环境变量

创建 `.env.local` 文件：

```env
DATABASE_URL=postgres://user:pass@localhost:5432/test_db
RUST_LOG=debug
CUSTOM_VAR=value
```

然后运行：
```bash
~/Downloads/act_Darwin_x86_64/act --env-file .env.local
```

### 使用 Secrets（敏感数据）

创建 `.secrets` 文件（不要提交到 Git！）：

```yaml
GITHUB_TOKEN: ghp_xxxxxxxxxxxx
DOCKER_PASSWORD: your_password
API_KEY: secret_key_here
```

运行时加载：
```bash
~/Downloads/act_Darwin_x86_64/act --secret-file .secrets
```

### 跳过某些 Jobs

```bash
# 只运行 lint 和 test，跳过其他
~/Downloads/act_Darwin_x86_64/act -j lint -j test
```

### 并行运行（性能优化）

```bash
# 并行执行所有 jobs（需要足够资源）
~/Downloads/act_Darwin_x86_64/act -p
```

---

## ⚠️ 常见问题与解决方案

### Q1: "permission denied" 错误

**原因**: macOS 安全限制阻止执行

**解决**:
```bash
xattr -d com.apple.quarantine ~/Downloads/act_Darwin_x86_64/act
chmod +x ~/Downloads/act_Darwin_x86_64/act
```

### Q2: Docker 未启动

**错误**: `Cannot connect to the Docker daemon`

**解决**:
1. 打开 Docker Desktop 应用
2. 等待 Docker 引擎启动完成（状态栏显示 "Docker is running"）
3. 重新运行命令

### Q3: 镜像拉取失败

**错误**: `Error: image not found`

**解决**:
```bash
# 手动拉取所需镜像
docker pull node:18-bullseye
docker pull rust:latest
```

### Q4: 内存不足

**错误**: `OOMKilled` 或系统变慢

**解决**:
1. 在 Docker Desktop 中增加内存限制（建议 4GB+）
2. 避免并行运行太多 jobs
3. 关闭不必要的应用释放内存

### Q5: 与 GitHub Actions 行为不一致

**原因**: 本地环境和 GitHub 环境有差异

**解决方案**:
1. 检查 `.github/workflows/ci-cd.yml` 中的 `runs-on` 配置
2. 确保本地 Docker 镜像版本正确
3. 使用 `-v` 参数查看详细日志对比差异

---

## 📊 性能对比

| 环境 | Lint | Build | Test | Full Pipeline |
|------|------|-------|------|---------------|
| **GitHub Actions** | ~30s | ~3min | ~5min | ~10min |
| **Local (act)** | ~15s | ~1min | ~2min | ~4min |
| **提升比例** | **2x** | **3x** | **2.5x** | **2.5x** |

*注：基于 MacBook Pro M1, 16GB RAM 测试*

---

## 🎓 最佳实践

### ✅ Do（推荐做法）

1. **每次提交前运行 lint**
   ```bash
   ./scripts/local-ci.sh lint
   ```

2. **Push 前运行完整测试**
   ```bash
   ./scripts/local-ci.sh full
   ```

3. **修改 CI 配置后立即测试**
   ```bash
   ./scripts/local-ci.sh dry-run  # 先预览
   ./scripts/local-ci.sh full     # 再实测
   ```

4. **定期运行安全审计**
   ```bash
   ./scripts/local-ci.sh security
   ```

### ❌ Don't（避免做法）

1. 不要在未测试的情况下直接 Push 到 main
2. 不要忽略本地 CI 的失败（即使 GitHub 可能通过）
3. 不要在低内存情况下并行运行所有 jobs
4. 不要将 `.secrets` 文件提交到 Git

---

## 📚 相关资源

- **act 官方仓库**: https://github.com/nektos/act
- **官方文档**: https://nektosact.com/
- **Docker 安装**: https://docs.docker.com/docker-for-mac/install/

---

## 💬 获取帮助

如果遇到问题：

1. 查看本指南的"常见问题"章节
2. 运行 `./scripts/local-ci.sh help`
3. 查看 act 官方文档
4. 在项目 Issues 中提问

---

**最后更新**: 2026-04-09  
**适用版本**: nektos/act v0.2.x+
