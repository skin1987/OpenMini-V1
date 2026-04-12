# Contributing to OpenMini-V1

感谢你对 OpenMini-V1 项目的关注！本文档将指导你如何参与贡献。

## 开发环境搭建

### 前置条件

- **Rust**: 1.82+ (stable)
- **Cargo**: 随 Rust 安装
- **Git**: 版本控制
- (可选) **CUDA**: 11.8+ (GPU开发)

### 快速开始

```bash
# 克隆仓库
git clone https://github.com/openmini/openmini-v1.git
cd openmini-v1

# 编译
cd openmini-server
cargo build --release

# 运行测试
cargo test

# 代码格式化
cargo fmt

# 静态检查
cargo clippy -- -D warnings
```

## 项目结构

```
openmini-v1/
├── openmini-server/          # 核心推理服务器
│   ├── src/
│   │   ├── model/inference/  # 推理引擎 (FA3, NSA, Kascade...)
│   │   ├── hardware/         # GPU/CPU/内存管理
│   │   ├── training/         # 训练管线
│   │   ├── benchmark/        # 性能基准
│   │   ├── distributed/      # 分布式推理
│   │   └── enterprise/       # 企业版功能
│   └── Cargo.toml
├── config/                   # 模型配置
└── docs/                     # 文档
```

## 代码规范

### Rustfmt

项目使用标准 `rustfmt` 格式化：

```bash
cargo fmt --all
```

### Clippy

提交前必须通过 clippy 检查：

```bash
cargo clippy -- -D warnings
```

### 命名约定

| 类型 | 约定 | 示例 |
|------|------|------|
| 结构体/枚举 | PascalCase | `FlashAttention3` |
| 函数/方法 | snake_case | `compute_top_k` |
| 常量 | UPPER_SNAKE_CASE | `MAX_SEQ_LEN` |
| 模块 | snake_case | `flash_attention_3` |

## 提交信息格式

使用 Conventional Commits 规范：

```
<type>(<scope>): <subject>

<body>
```

**Type**: feat, fix, docs, style, refactor, test, chore, perf

**示例**:
```
feat(inference): add NSA sparse attention implementation

- Implement TokenCompressor for global information preservation
- Add TopKSelector for key detail retention
- Integrate with MLA path selection logic
```

## PR 流程

1. **Fork** 并创建分支: `git checkout -b feature/my-feature`
2. **编码**: 遵循代码规范
3. **测试**: `cargo test` 全部通过
4. **格式化**: `cargo fmt && cargo clippy`
5. **提交**: 使用规范的 commit message
6. **Push** 并创建 PR

### PR 描述模板

```markdown
## 变更类型
- [ ] Bug修复
- [ ] 新功能
- [ ] 性能优化
- [ ] 文档更新

## 描述
简要描述你的变更

## 测试
- [ ] 单元测试已添加/更新
- [ ] 集成测试已通过
- [ ] 无回归问题

## 关联Issue
Closes #xxx
```

## 测试要求

### 新功能必须包含：

1. **单元测试**: 覆盖核心逻辑路径
2. **边界测试**: 处理极端输入
3. **集成测试**: 与现有模块的交互

### 测试命名规范：

```rust
#[test]
fn test_<module>_<function>_<scenario>() {
    // ✅ 正确
}

// 示例:
fn test_nsa_forward_long_sequence() {}
fn test_kascade_reuse_strategy_adaptive() {}
fn test_flash_attention_fp8_precision() {}
```

## Issue 标签说明

| 标签 | 含义 |
|------|------|
| `bug` | 缺陷报告 |
| `enhancement` | 新功能请求 |
| `good first issue` | 适合新手 |
| `help wanted` | 需要社区帮助 |
| `performance` | 性能优化 |
| `paper` | 论文实现 |

## 性能要求

对于性能敏感代码：

1. 使用 `criterion` crate 编写 benchmark
2. 在 PR 中提供 before/after 对比数据
3. 注释关键优化点

```rust
#[cfg(test)]
mod benchmarks {
    use criterion::{black_box, criterion_group, Criterion};

    fn bench_forward(c: &mut Criterion) {
        c.bench_function("nsa_forward", |b| {
            b.iter(|| nsa.forward(black_box(&input)));
        });
    }
}
```

## 文档要求

### 公共 API 必须包含：

1. 功能描述
2. 参数说明
3. 返回值说明
4. 使用示例
5. Panic 条件（如有）

```rust
/// NSA 三路稀疏注意力前向传播
///
/// # Arguments
/// * `q` - Query tensor [batch, seq_len, num_heads * head_dim]
/// * `k` - Key tensor
/// * `v` - Value tensor
///
/// # Returns
/// Attention output with same shape as q
///
/// # Example
/// ```
/// let output = nsa.forward(&q, &k, &v)?;
/// ```
pub fn forward(&self, q: &Array3<f32>, k: &Array3<f32>, v: &Array3<f32>) -> Result<Array3<f32>> { ... }
```

## 安全漏洞报告

发现安全问题时，请发送邮件至 security@openmini.ai，不要公开 Issue。

## 行为准则

- 尊重所有贡献者
- 建设性讨论技术方案
- 帮助新人成长
- 关注代码质量而非个人

## 联系方式

- **Discord**: [OpenMini Community](https://discord.gg/openmini)
- **GitHub Discussions**: [讨论区](https://github.com/openmini/openmini-v1/discussions)
- **Email**: dev@openmini.ai
