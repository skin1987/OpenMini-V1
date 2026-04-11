# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0-beta.1] - 2026-04-09

### 🎉 Added
- **Vue3 Admin Panel**: 完整的管理面板框架
- **Database Abstraction Layer**: 数据库抽象层 (MemoryStore/SessionManager/MessagePool)
- **CI/CD Pipeline**: 完整的持续集成配置 (GitHub Actions + act)

### 🔧 Fixed
- **DSA Integration Tests** - 测试配置维度匹配、因果掩码数值稳定性、GPU降级处理 → **3/3 passed**
- **Clippy Errors** - LOG2_E常量 + usize::MAX比较 → **0 errors**
- **Field Access Path** - ServerConfig.port → ServerConfig.server.port

### 🧪 Tested
```
✅ Metal Runtime:    23/23 passed (100%)
✅ RL Module:       105/105 passed (100%)
✅ DSA Integration:   3/3 passed    (100%)
✅ Clippy Errors:   0 errors
✅ Release Build:    Success (5m31s)
```

---

## [1.1.0] - 2026-04-08

### 🎉 Initial Release
- High-Performance Inference Engine (Candle-based, Flash Attention 3, DSA)
- Multi-Hardware Support (CPU/CUDA/Metal/Vulkan)
- Reinforcement Learning Module (GRPO, Actor/Reward)
- Service Layer (gRPC/HTTP, Worker Pool)
- Monitoring & Observability (Prometheus, Health Check)

---

**Note**: This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).