# OpenMini-V1 v1.2.0-beta.1 Release Notes

**Release Date**: 2026-04-09 | **Status**: 🧪 Beta Testing

## 📊 Quality Metrics
| Metric | Result | Status |
|--------|--------|--------|
| DSA Tests | **3/3 passed** | 🟢 |
| RL Tests | **105/105 passed** | 🟢 |
| Metal Tests | **23/23 passed** | 🟢 |
| Clippy Errors | **0** | 🟢 |
| Release Build | ✅ 5m31s | 🟢 |

## 🔧 Bug Fixes (3 issues)
- DSA Integration Tests: 维度匹配 + 因果掩码稳定性 + GPU降级
- Clippy Errors: LOG2_E + usize::MAX → 0 errors
- Field Access Path: ServerConfig.server.port

## 🆕 New Features
- Vue3 Admin Panel Framework
- Database Abstraction Layer
- Enhanced CI/CD Pipeline

## 🚀 Quick Start
```bash
git clone https://github.com/skin1987/OpenMini-V1.git && cd OpenMini-V1 && git checkout v1.2.0-beta.1
cargo build --release
./target/release/openmini-server --config config/server.toml
curl http://localhost:8080/health
```

Full details in CHANGELOG.md