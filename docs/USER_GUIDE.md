# OpenMini-V1 用户操作手册

**版本**: v1.2.0+  
**最后更新**: 2026-04-09  
**适用范围**: 所有用户（管理员、开发者、运维人员）

---

## 📖 目录

1. [快速入门](#1-快速入门)
2. [核心功能使用](#2-核心功能使用)
3. [Admin 管理面板](#3-admin-管理面板)
4. [高级功能](#4-高级功能)
5. [性能优化建议](#5-性能优化建议)
6. [故障排查](#6-故障排查)

---

## 1. 快速入门

### 1.1 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| **操作系统** | macOS 12+, Ubuntu 20.04+ | macOS 14+, Ubuntu 22.04+ |
| **CPU** | 4 核心 (支持 AVX2 或 NEON) | 8 核心以上 |
| **内存** | 8 GB RAM | 16 GB RAM (7B 模型) / 32 GB (13B+) |
| **存储** | 10 GB 可用空间 (SSD) | 50 GB NVMe SSD |
| **GPU** (可选) | NVIDIA GTX 1060+ (6GB) / Apple M1+ | RTX 3090 (24GB) / M2 Ultra |
| **Rust** | 1.75+ | 最新稳定版 |

### 1.2 安装步骤

#### 方式 A: 从源码编译 (推荐开发者)

```bash
# 1. 克隆仓库
git clone https://github.com/skin1987/OpenMini-V1.git
cd OpenMini-V1

# 2. 编译 Release 版本 (~5-6 分钟)
cargo build --release

# 3. 复制配置文件
cp config/server.toml.example config/server.toml

# 4. 启动服务
./target/release/openmini-server --config config/server.toml
```

#### 方式 B: Docker 部署 (推荐生产环境)

```bash
# 构建镜像
docker build -t openmini-server .

# 运行容器
docker run -d \
  --name openmini \
  -p 50051:50051 \
  -p 8080:8080 \
  -v ./models:/app/models \
  -v ./config:/app/config \
  openmini-server
```

#### 方式 C: Docker Compose (完整部署)

```bash
# 一键启动 (含 Admin 面板)
docker compose up -d

# 查看日志
docker compose logs -f openmini-server
```

### 1.3 首次启动配置

编辑 `config/server.toml`:

```toml
[server]
host = "0.0.0.0"          # 监听地址
port = 50051              # gRPC 端口
http_port = 8080          # HTTP REST 端口

[model]
name = "your-model.gguf" # 模型文件名
path = "/path/to/models"  # 模型目录

[hardware]
backend = "auto"          # auto | cpu | cuda | metal
```

启动后验证:

```bash
curl http://localhost:8080/health
# 预期输出: {"status":"healthy","version":"1.2.0-beta.1"}
```

---

## 2. 核心功能使用

### 2.1 模型推理

#### REST API 调用

**生成文本 (Completions API)**:

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "解释量子计算",
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "stop": ["\n\n"]
  }'
```

**响应示例**:

```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1677858245,
  "model": "your-model.gguf",
  "choices": [
    {
      "text": "量子计算是一种利用量子力学原理进行计算的技术...",
      "index": 0,
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 128,
    "total_tokens": 133
  }
}
```

**流式输出 (Server-Sent Events)**:

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "stream": true}'
```

#### gRPC 客户端 (Rust)

```rust
use tonic::Request;
use openmini_proto::inference_service_client::InferenceServiceClient;
use openmini_proto::{InferenceRequest, GenerateRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = InferenceServiceClient::connect("http://localhost:50051").await?;
    
    let request = Request::new(InferenceRequest {
        generate_request: Some(GenerateRequest {
            prompt: "What is machine learning?".into(),
            max_new_tokens: 128,
            temperature: 0.8,
            top_p: 0.95,
            ..Default::default()
        }),
        ..Default::default()
    });
    
    let response = client.generate(request).await?;
    println!("Response: {:?}", response.into_inner());
    
    Ok(())
}
```

#### Python SDK 使用

```python
from openmini_client import OpenMiniClient

client = OpenMiniClient(
    base_url="http://localhost:8080",
    api_key="your-api-key"
)

response = client.generate(
    prompt="Explain neural networks",
    max_tokens=256,
    temperature=0.7
)

print(f"Generated text: {response.text}")
print(f"Tokens used: {response.tokens_used}")
print(f"Latency: {response.latency_ms}ms")
```

### 2.2 配置管理

#### server.toml 完整参数说明

```toml
[server]
host = "0.0.0.0"              # 服务监听地址
port = 50051                   # gRPC API 端口
http_port = 8080               # REST API 端口
workers = 4                    # Worker 进程数量 (0=自动)
request_timeout = 300          # 请求超时时间(秒)
max_concurrent_requests = 100  # 最大并发请求数

[model]
name = "model.q4_0.gguf"      # GGUF 模型文件名
path = "./models"              # 模型文件目录
max_context_length = 4096      # 最大上下文长度
max_batch_size = 8             # 最大批处理大小

[hardware]
backend = "auto"               # 硬件后端: auto/cpu/cuda/metal
gpu_memory_fraction = 0.9      # GPU 内存使用比例 (0.0-1.0)

[inference]
temperature_default = 0.7      # 默认温度参数
top_p_default = 0.9            # 默认 Top-P 采样
repetition_penalty = 1.1       # 重复惩罚系数

[logging]
level = "info"                 # 日志级别: trace/debug/info/warn/error
format = "json"                # 输出格式: json/text
file = "./logs/openmini.log"   # 日志文件路径 (空=仅控制台)

[monitoring]
prometheus_enabled = true      # 启用 Prometheus 指标
metrics_port = 9090            # 指标暴露端口
health_check_path = "/health"  # 健康检查端点
```

#### 动态配置更新 (无需重启)

```bash
# 通过 Admin API 更新配置
curl -X PUT http://localhost:8080/admin/config \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "inference": {
      "temperature_default": 0.5
    }
  }'

# 重载配置 (从文件)
curl -X POST http://localhost:8080/admin/config/reload \
  -H "Authorization: Bearer <token>"
```

---

## 3. Admin 管理面板

### 3.1 访问和登录

**启动方式**:

```bash
# 后端服务 (端口 3001)
cd openmini-admin && cargo run

# 前端界面 (开发模式)
cd openmini-admin-web && npm run dev
# 访问: http://localhost:5173
```

**默认账号**:
- 用户名: `admin`
- 密码: `admin123`
- ⚠️ **首次登录请立即修改密码**

### 3.2 功能模块概览

| 模块 | 功能描述 | 权限要求 |
|------|---------|---------|
| **仪表盘** | 系统状态、资源监控、实时指标 | viewer+ |
| **用户管理** | 用户 CRUD、角色分配、密码重置 | admin |
| **API Key 管理** | 密钥创建/撤销、配额设置、用量统计 | admin/operator |
| **模型管理** | 模型加载/卸载、健康检查、切换 | admin/operator |
| **服务管理** | Worker 状态、重启/停止服务 | admin |
| **监控中心** | 推理性能图表、资源使用率 | viewer+ |
| **告警系统** | 规则配置、告警记录、确认/解决 | operator+ |
| **审计日志** | 操作记录查询、统计报表 | admin |
| **系统配置** | 参数编辑、历史版本回滚 | admin |

### 3.3 用户管理操作示例

#### 创建新用户

通过 Web 界面:
1. 进入「用户管理」页面
2. 点击「新建用户」
3. 填写信息：
   - 用户名: `operator1` (3-20字符)
   - 邮箱: `operator1@example.com`
   - 密码: `SecurePass123!` (≥6字符)
   - 角色: `operator`
4. 点击「保存」

通过 API:

```bash
curl -X POST http://localhost:3001/admin/users \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "operator1",
    "email": "operator1@example.com",
    "password": "SecurePass123!",
    "role": "operator"
  }'
```

#### 角色权限矩阵

| 权限 | admin | operator | viewer |
|------|-------|----------|--------|
| 用户管理 | ✅ 完全控制 | ❌ | ❌ |
| API Key 管理 | ✅ | ✅ 创建/查看自己的 | ✅ 查看 |
| 模型管理 | ✅ 卸载/加载 | ✅ 切换/查看 | ✅ 查看 |
| 服务管理 | ✅ 重启/停止 | ❌ | ❌ |
| 配置修改 | ✅ | ❌ | ❌ |
| 查看监控 | ✅ | ✅ | ✅ |
| 查看日志 | ✅ | ✅ | ✅ |

---

## 4. 高级功能

### 4.1 量化模型支持

OpenMini-V1 支持多种 GGUF 量化格式，在精度和性能间取得平衡:

| 格式 | 位宽 | 显存占用 (7B模型) | 推荐场景 |
|------|------|------------------|----------|
| **F16** | 16-bit | ~14 GB | 最高精度需求 |
| **Q8_0** | 8-bit | ~7 GB | 平衡选择 |
| **Q4_0** | 4-bit | ~3.5 GB | 资源受限环境 |
| **Q4_1** | 4-bit + 校正 | ~3.9 GB | 比 Q4_0 精度略优 |

**使用方法**:

只需将量化后的 `.gguf` 文件放入模型目录，无需额外配置：

```toml
[model]
name = "llama-2-7b-chat.Q4_0.gguf"  # 自动识别格式
path = "/models"
```

### 4.2 DSA 稀疏注意力优化

Dynamic Sparse Attention (DSA) 可显著降低长文本推理的内存和计算开销：

**启用方式** (默认开启):

```toml
[inference]
dsa_enabled = true           # 启用 DSA
dsa_sparsity = 0.5           # 稀疏度 (0.25-0.75)
dsa_top_k = 32               # 每个 query 的 top-k 注意力头数
```

**性能提升** (seq_len=2048, sparsity=50%):
- 内存使用: ↓ 40%
- 推理速度: ↑ 25%
- 输出质量: ≈ 98% 相似度 (vs 全注意力)

### 4.3 多硬件后端切换

根据运行环境自动或手动选择最优计算后端:

```bash
# 自动检测 (推荐)
[hardware]
backend = "auto"

# 强制指定
backend = "metal"     # Apple Silicon (M1/M2/M3)
backend = "cuda"      # NVIDIA GPU
backend = "cpu"       # CPU only (AVX2/NEON)
```

**各后端性能对比** (Llama 2 7B, seq_len=512):

| 后端 | 硬件 | Tokens/sec | 内存效率 |
|------|------|------------|----------|
| Metal | M2 Max (38核) | ~45 | ★★★★☆ |
| CUDA | RTX 3090 | ~85 | ★★★★★ |
| CPU AVX2 | i9-13900K | ~12 | ★★★☆☆ |
| CPU NEON | M2 (10核) | ~18 | ★★★★☆ |

### 4.4 批量推理与流式输出

**批量推理** (提高吞吐量):

```rust
// 同时处理多个请求
let requests = vec![
    GenerateRequest { prompt: "问题1".into(), ..Default::default() },
    GenerateRequest { prompt: "问题2".into(), ..Default::default() },
    // ...最多 max_batch_size 个
];

let responses = client.batch_generate(requests).await?;
```

**流式输出** (降低首字延迟):

```bash
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "写一首诗"}],
    "stream": true
  }'
```

输出 SSE 格式，每个 token 一个事件:

```
data: {"choices":[{"delta":{"content":"春"}}]}

data: {"choices":[{"delta":{"content":"眠"}}]}

data: [DONE]
```

---

## 5. 性能优化建议

### 5.1 硬件选择建议

**GPU 推荐** (按预算):

| 预算 | GPU | 适用模型 | 场景 |
|------|-----|---------|------|
| 入门 | RTX 3060 12GB | 7B Q4 | 个人开发 |
| 中端 | RTX 4090 24GB | 13B Q4/F16 | 小团队 |
| 高端 | A100 80GB | 70B+ F16 | 生产服务 |
| Mac | M2 Ultra 192GB | 34B F16 | 本地开发 |

**CPU 推荐**:
- x86: 支持 AVX-512 的 Intel Xeon 或 AMD EPYC
- ARM: Apple M-series (M1 Pro及以上)

### 5.2 参数调优指南

**吞吐量优先** (高并发 API 服务):

```toml
[server]
workers = 4                    # 根据 CPU 核心数调整
max_concurrent_requests = 200  # 提高并发上限

[model]
max_batch_size = 16            # 增大批次 (需要更多显存)

[inference]
temperature_default = 0.3      # 降低随机性可加速采样
```

**延迟优先** (实时对话):

```toml
[server]
workers = 1                    # 单进程减少调度延迟
max_concurrent_requests = 10   # 限制并发避免排队

[model]
max_batch_size = 1             # 关闭批处理

[inference]
cache_enabled = true           # 启用 KV Cache
```

**内存优化** (低资源环境):

```toml
[model]
name = "model.Q4_0.gguf"      # 使用 4-bit 量化
max_context_length = 2048      # 缩短上下文

[hardware]
gpu_memory_fraction = 0.8      # 保留部分显存给系统
```

### 5.3 内存管理最佳实践

1. **KV Cache 监控**: 定期检查 `/metrics` 中的 `kv_cache_blocks_used`
2. **及时释放**: 不活跃的会话自动超时释放 (默认 30min)
3. **批处理策略**: 小请求合并成批次减少碎片化
4. **量化选择**: 生产环境推荐 Q4_0/Q4_1，精度损失 <2%

---

## 6. 故障排查

### 6.1 常见错误及解决方案

#### 错误 1: 模型加载失败

**症状**:
```
Error: Failed to load model: Io(Os { code: 2, kind: NotFound })
```

**解决方案**:
1. 检查 `config/server.toml` 中 `[model].path` 是否正确
2. 确认模型文件存在于指定目录
3. 验证文件权限: `ls -lh models/`

#### 错误 2: CUDA Out of Memory

**症状**:
```
CUDA error: out of memory
```

**解决方案**:
1. 减小 `max_batch_size` (如 16→8→4)
2. 降低 `max_context_length` (如 4096→2048)
3. 使用更激进的量化 (F16→Q8_0→Q4_0)
4. 设置 `gpu_memory_fraction = 0.8`

#### 错误 3: 连接被拒绝

**症状**:
```
Connection refused (os error 61)
```

**解决方案**:
1. 确认服务已启动: `ps aux | grep openmini`
2. 检查端口占用: `lsof -i :50051`
3. 查看防火墙规则: `sudo ufw status`

#### 错误 4: 认证失败 (Admin Panel)

**症状**:
```
401 Unauthorized: Token expired or invalid
```

**解决方案**:
1. 重新登录获取新 token
2. 检查 JWT 密钥配置是否一致
3. 验证 token 未过期 (默认 24h 有效期)

### 6.2 日志分析与调试

**启用详细日志**:

```bash
RUST_LOG=debug ./openmini-server --config server.toml
```

**关键日志位置**:

| 信息类型 | 日志级别 | 示例关键词 |
|---------|---------|-----------|
| 请求处理 | INFO | `"method":"POST","path":"/v1/completions"` |
| 模型加载 | INFO | `"Loading model","file":"xxx.gguf"` |
| 性能指标 | INFO | `"latency_ms":150,"tokens_generated":64` |
| 错误详情 | ERROR | `"error":"Failed to allocate memory"` |
| 调试跟踪 | DEBUG | `"block_idx":42,"kv_cache_hit":true` |

**结构化日志解析** (JSON 格式):

```bash
# 提取所有错误日志
cat logs/openmini.log | jq 'select(.level == "ERROR")'

# 统计请求延迟分布
cat logs/openmini.log | jq -r '.latency_ms' | awk '{sum+=$1; count++} END {print sum/count}'
```

### 6.3 性能问题诊断

**步骤 1: 检查资源使用**

```bash
# CPU 和内存
top -p $(pgrep openmini-server)

# GPU 使用 (如适用)
nvidia-smi -l 1

# 网络连接数
ss -tn | grep :50051 | wc -l
```

**步骤 2: 分析 Prometheus 指标**

```bash
curl http://localhost:9090/metrics | grep -E "(openmini_|process_)"
```

关键指标:
- `openmini_inference_duration_seconds`: 推理延迟分布
- `openmini_requests_total`: 请求计数 (按状态码分类)
- `openmini_active_connections`: 当前活跃连接数
- `process_resident_memory_bytes`: 内存使用量

**步骤 3: 压力测试定位瓶颈**

```bash
# 运行内置压力测试
cargo test --package openmini-server --test stress_test -- --nocapture

# 或使用外部工具
wrk -t12 -c400 -d30s http://localhost:8080/health
```

---

## 📚 相关文档

- [API 完整参考](./API.md)
- [部署指南](./PRODUCTION_DEPLOYMENT.md)
- [故障排查手册](./TROUBLESHOOTING.md)
- [CHANGELOG](../CHANGELOG.md)
- [GitHub Issues](https://github.com/skin1987/OpenMini-V1/issues)

---

## 💡 获取帮助

- **文档**: https://github.com/skin1987/OpenMini-V1/wiki
- **Issues**: https://github.com/skin1987/OpenMini-V1/issues
- **Discussions**: https://github.com/skin1987/OpenMini-V1/discussions
- **Email**: support@openmini.ai (如有)

---

*最后更新: 2026-04-09 by OpenMini Team*
