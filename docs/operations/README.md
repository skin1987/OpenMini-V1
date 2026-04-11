# OpenMini 运维手册

本文档提供 OpenMini 推理服务在生产环境中的部署、监控、故障排查和维护指南。

---

## 目录

1. [部署指南](#1-部署指南)
2. [配置参考](#2-配置参考)
3. [环境变量](#3-环境变量)
4. [监控与告警设置](#4-监控与告警设置)
5. [故障排查指南](#5-故障排查指南)
6. [性能调优参数](#6-性能调优参数)
7. [内存优化建议](#7-内存优化建议)
8. [GPU 内存管理](#8-gpu-内存管理)
9. [维护操作流程](#9-维护操作流程)

---

## 1. 部署指南

### 1.1 系统要求

#### 硬件要求

| 组件 | 最低配置 | 推荐配置 | 生产环境 |
|------|---------|---------|---------|
| **CPU** | 4 核 (支持 AVX2) | 8 核 (AVX2) | 16+ 核 (AVX2/AVX-512) |
| **内存** | 16 GB RAM | 32 GB RAM | 64+ GB RAM |
| **GPU** (可选) | NVIDIA T4 (16GB) | NVIDIA A10G (24GB) | NVIDIA A100/H100 (40/80GB) |
| **存储** | 50 GB SSD | 200 GB NVMe SSD | 1 TB NVMe SSD |
| **网络** | 1 Gbps | 10 Gbps | 25+ Gbps |

#### 软件依赖

| 软件 | 版本要求 | 用途 |
|------|---------|------|
| **Rust toolchain** | >= 1.75.0 (stable) | 编译运行时 |
| **CUDA Toolkit** | >= 11.8 (GPU 推理) | CUDA kernel 编译与运行 |
| **cuDNN** | >= 8.6 (GPU 推理) | 深度学习加速库 |
| **TensorRT** (可选) | >= 8.5 | 模型优化与加速推理 |
| **Prometheus** | >= 2.40.0 | 指标采集 |
| **Grafana** | >= 9.0.0 | 可视化监控面板 |

#### 操作系统支持

- **Linux**: Ubuntu 20.04+/22.04, CentOS 8+, Amazon Linux 2023
- **macOS**: 13+ (Apple Silicon M1/M2/M3, Metal 后端)
- **Windows**: WSL2 (不推荐生产使用)

### 1.2 构建与安装

#### 从源码构建

```bash
# 1. 克隆仓库
git clone https://github.com/openmini/openmini-v1.git
cd openmini-v1

# 2. 安装 Rust (如果未安装)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 3. 安装系统依赖 (Ubuntu/Debian)
sudo apt update
sudo apt install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    libprotobuf-dev

# 4. 安装 CUDA 工具链 (如果使用 GPU)
# 参考: https://developer.nvidia.com/cuda-downloads

# 5. Release 编译 (优化性能)
cargo build --release -p openmini-server

# 6. 验证编译结果
./target/release/openmini-server --version
```

#### Docker 构建（推荐）

```bash
# 使用官方 Dockerfile
docker build -t openmini-server:latest .

# 运行容器
docker run -d \
    --name openmini \
    --gpus all \
    -p 8000:8000 \
    -p 9090:9090 \
    -v $(pwd)/config:/app/config \
    -v $(pwd)/models:/app/models \
    openmini-server:latest
```

### 1.3 快速启动

```bash
# 1. 准备配置文件
cp config/server.toml.example config/server.toml
# 编辑配置文件，填入模型路径等必要信息

# 2. 启动服务
./target/release/openmini-server --config config/server.toml

# 3. 验证服务健康
curl http://localhost:8000/health

# 4. 测试推理接口
curl -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "test", "prompt": "Hello", "max_tokens": 10}'
```

---

## 2. 配置参考

完整的配置文件位于 `config/server.toml`，以下是各配置项的详细说明：

### 2.1 服务器基础配置 (`[server]`)

```toml
[server]
# 监听地址 (默认: "0.0.0.0")
host = "0.0.0.0"

# HTTP 端口 (范围: 1-65535, 默认: 8000)
port = 8000

# Metrics 暴露端口 (Prometheus 抓取用, 默认: 9090)
metrics_port = 9090

# 工作线程数 (默认: CPU 核心数, 推荐: 4-16)
num_workers = 8

# 请求超时时间 (秒, 默认: 300)
request_timeout = 300

# 最大请求体大小 (MB, 默认: 10)
max_request_size_mb = 10

# 启用 CORS (开发环境设为 true)
cors_enabled = false
```

### 2.2 模型配置 (`[model]`)

```toml
[model]
# 模型名称 (用于日志和指标标识)
name = "llama-3-8b-instruct"

# 模型权重文件路径 (GGUF/Safetensors 格式)
path = "/models/llama-3-8b-Q4_K_M.gguf"

# Tokenizer 文件路径
tokenizer_path = "/models/tokenizer.json"

# 模型隐藏层维度 (必须与模型实际值一致)
hidden_size = 4096

# 注意力头数
num_attention_heads = 32

# 层数
num_hidden_layers = 32

# 词表大小
vocab_size = 128256

# 最大序列长度 (影响 KV Cache 大小)
max_seq_len = 8192
```

### 2.3 调度器配置 (`[scheduler]`)

```toml
[scheduler]
# 最大并发任务数 (默认: CPU 核心数)
# GPU 推理推荐: 8-16; CPU 推理推荐: 等于核心数
max_concurrent = 8

# 任务队列容量 (默认: 1000)
# 内存估算: 每个 ~1-10KB, 1000 队列 ≈ 1-10 MB
queue_capacity = 1000

# 连续批处理大小 (默认: 8)
# 较大值提高吞吐量但增加延迟
batch_size = 8

# 批处理等待超时 (毫秒, 默认: 5)
# 实时对话: 2-5ms; 离线批量: 20-50ms
batch_timeout_ms = 5
```

### 2.4 KV Cache 配置 (`[kv_cache]`)

```toml
[kv_cache]
# 启用分页机制 (内存不足时自动换出到 CPU)
paging = true

# 最大页数 (每页 64 tokens, 根据显存调整)
max_pages = 1024

# CPU Offload 上限 (字节, 0=无限制)
cpu_offload_bytes = 0

# 缓存淘汰策略: "lru" | "fifo" | "lfu"
eviction_policy = "lru"

# Prefix Cache 最大条目数 (共享前缀缓存)
prefix_cache_max_entries = 100
```

### 2.5 训练配置 (`[training]`) (可选)

```toml
[training]
# 学习率
learning_rate = 1e-5

# Batch size
train_batch_size = 4

# 梯度裁剪阈值
grad_clip_norm = 1.0

# 混合精度训练 (节省显存)
amp_enabled = true

# 检查点保存路径
checkpoint_dir = "/data/checkpoints"

# 检查点保存间隔 (步数)
save_every_n_steps = 500
```

### 2.6 日志配置 (`[logging]`)

```toml
[logging]
# 日志级别: "trace" | "debug" | "info" | "warn" | "error"
level = "info"

# 日志格式: "json" | "text" | "pretty"
format = "json"

# 日志文件路径 (空=仅输出到 stdout)
file = "/var/log/openmini/server.log"

# 日志轮转大小 (MB)
rotation_size_mb = 100

# 保留日志文件数
max_files = 10
```

---

## 3. 环境变量

除了配置文件，以下环境变量可覆盖或补充配置：

| 环境变量 | 说明 | 默认值 | 示例 |
|---------|------|--------|------|
| `OPENMINI_CONFIG_PATH` | 配置文件路径 | `./config/server.toml` | `/etc/openmini/server.toml` |
| `OPENMINI_LOG_LEVEL` | 日志级别覆盖 | (来自配置) | `debug` |
| `OPENMINI_MODEL_PATH` | 模型权重路径覆盖 | (来自配置) | `/models/new-model.gguf` |
| `OPENMINI_GPU_MEMORY_FRACTION` | GPU 显存使用比例 | 1.0 (全部) | `0.8` (使用 80%) |
| `OPENMINI_CUDA_VISIBLE_DEVICES` | 可见 GPU 列表 | 所有 GPU | `0,1` (使用前 2 张卡) |
| `RUST_LOG` | Rust 全局日志级别 | `info` | `openmini_server=debug,tokio=info` |
| `TOKENIZERS_PARALLELISM` | Tokenizer 并行度 | `true` | `false` (单线程模式) |

### 环境变量使用示例

```bash
# 开发调试模式
export RUST_LOG="openmini_server=debug"
export OPENMINI_LOG_LEVEL="debug"
./target/release/openmini-server

# 生产环境指定 GPU
export OPENMINI_CUDA_VISIBLE_DEVICES="0,1,2,3"
export OPENMINI_GPU_MEMORY_FRACTION="0.9"
./target/release/openmini-server --config /etc/openmini/server.toml
```

---

## 4. 监控与告警设置

### 4.1 Prometheus 抓取配置

在 Prometheus 的 `prometheus.yml` 中添加：

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # OpenMini 服务指标
  - job_name: 'openmini'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:9090']
        labels:
          env: production
          service: openmini-inference
          instance: 'server-01'

  # 多实例场景 (Kubernetes/Docker Swarm)
  - job_name: 'openmini-cluster'
    scrape_interval: 10s
    dns_sd_configs:
      - names:
          - 'tasks.openmini'
        type: 'A'
        port: 9090
```

### 4.2 Grafana Dashboard 导入

#### 方法 1: 导入 JSON 配置

1. 打开 Grafana → Dashboards → Import
2. 粘贴下方的 Dashboard JSON 或上传文件
3. 选择 Prometheus 数据源
4. 点击 Import

#### 方法 2: 使用 Dashboard ID (待发布)

我们计划在未来版本提供 Grafana 官方 Marketplace Dashboard ID。

#### 推荐面板布局

| 行 | 面板名称 | PromQL 查询 | 类型 |
|----|---------|------------|------|
| 1 | Token 吞吐量 (tokens/sec) | `sum(rate(openmini_inference_tokens_total{status="success"}[5m])) by (model)` | Graph |
| 1 | 请求 QPS | `sum(rate(openmini_request_duration_seconds_count[5m])) by (endpoint)` | Graph |
| 2 | P99 延迟 (秒) | `histogram_quantile(0.99, sum(rate(openmini_request_duration_seconds_bucket[5m])) by (le, endpoint))` | Graph |
| 2 | P50 延迟 (秒) | `histogram_quantile(0.50, sum(rate(...)) by (le, endpoint))` | Graph |
| 3 | KV Cache 内存 (GB) | `sum(openmini_kv_cache_usage_bytes) / 1024^3` | Gauge |
| 3 | Worker 队列深度 | `openmini_worker_queue_length` | Graph |
| 4 | 活跃连接数 | `openmini_active_connections` | Graph |
| 4 | 错误率 (%) | `rate(openmini_inference_tokens_total{status="error"}[5m]) / rate(openmini_inference_tokens_total[5m]) * 100` | Stat |

### 4.3 告警规则配置

创建 `alert_rules.yml` 并添加到 Prometheus 配置：

```yaml
groups:
  - name: openmini-alerts
    interval: 30s
    rules:
      # === P0 紧急告警 ===

      - alert: HighErrorRate
        expr: |
          rate(openmini_inference_tokens_total{status="error"}[5m])
          /
          rate(openmini_inference_tokens_total[5m])
          > 0.05
        for: 2m
        labels:
          severity: critical
          team: ml-platform
        annotations:
          summary: "OpenMini 错误率过高 ({{ $value | humanizePercentage }})"
          description: "错误率超过 5%，请立即检查服务状态"

      - alert: HighP99Latency
        expr: |
          histogram_quantile(0.99,
            sum(rate(openmini_request_duration_seconds_bucket[5m])) by (le)
          ) > 10
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "请求 P99 延迟过高 ({{ $value }}s)"
          description: "P99 延迟超过 10 秒，可能存在性能瓶颈"

      - alert: ModelNotLoaded
        expr: openmini_model_loaded != 1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "模型未加载或已卸载"
          description: "检测到模型处于未加载状态，服务不可用"

      # === P1 重要告警 ===

      - alert: WorkerQueueBacklog
        expr: openmini_worker_queue_length > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Worker 队列积压严重 ({{ $value }} 个任务)"
          description: "队列深度超过 100，建议扩容或限流"

      - alert: HighKVCacheUsage
        expr: sum(openmini_kv_cache_usage_bytes) / 1024^3 > 20
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "KV Cache 内存占用过高 ({{ $value }} GB)"
          description: "KV Cache 占用超过 20GB，考虑启用分页或减小 batch_size"

      - alert: HighActiveConnections
        expr: openmini_active_connections > 500
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "活跃连接数过多 ({{ $value }})"
          description: "连接数超过 500，检查是否有连接泄露"

      # === P2 一般告警 ===

      - alert: LowThroughput
        expr: sum(rate(openmini_inference_tokens_total[1h])) < 1000
        for: 30m
        labels:
          severity: info
        annotations:
          summary: "Token 吞吐量异常偏低 (< 1K tokens/s)"
          description: "当前吞吐量为 {{ $value }} tokens/s，低于基线"

      - alert: ElevatedLatency
        expr: |
          histogram_quantile(0.50,
            sum(rate(openmini_request_duration_seconds_bucket[5m])) by (le)
          ) > 2
        for: 15m
        labels:
          severity: info
        annotations:
          summary: "平均延迟升高 (P50 = {{ $value }}s)"
          description: "P50 延迟超过 2 秒，请关注性能趋势"
```

### 4.4 告警通知渠道配置 (AlertManager)

```yaml
# alertmanager.yml
route:
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 12h
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty-slack'

receivers:
  - name: 'pagerduty-slack'
    slack_configs:
      - channel: '#ml-platform-alerts'
        send_resolved: true
        title: '🚨 {{ .Status | toUpper }}: {{ .CommonLabels.alertname }}'
        text: "{{ range .Alerts }}{{ .Annotations.description }}\n{{ end }}"
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
        severity: critical
```

### 4.5 日志聚合配置

#### ELK Stack (Elasticsearch + Logstash + Kibana)

```conf
# logstash.conf
input {
  file {
    path => "/var/log/openmini/*.json"
    codec => json
    start_position => "beginning"
  }
}

filter {
  if [message] =~ /ERROR/ {
    mutate { add_tag => ["error"] }
  }

  date {
    match => ["timestamp", "ISO8601"]
  }
}

output {
  elasticsearch {
    hosts => ["http://elasticsearch:9200"]
    index => "openmini-%{+YYYY.MM.dd}"
  }
}
```

#### Loki + Promtail (轻量级方案)

```yaml
# promtail-config.yaml
scrape_configs:
  - job_name: openmini
    static_configs:
      - targets:
          - localhost
        labels:
          job: openmini
          __path__: /var/log/openmi/*.json
    pipeline_stages:
      - json:
          expressions:
            level: level
            message: message
            timestamp: timestamp
      - labels:
          level:
      - output:
```

---

## 5. 故障排查指南

### 5.1 常见错误及解决方案

| 错误现象 | 可能原因 | 解决方案 | 优先级 |
|---------|---------|---------|--------|
| **服务启动失败：模型加载错误** | 模型文件路径错误、格式不支持、权限问题 | 1. 检查 `model.path` 是否正确<br>2. 确认文件格式为 GGUF/Safetensors<br>3. 检查文件读取权限 (`ls -la`) | P0 |
| **CUDA out of memory** | 显存不足、batch_size 过大、KV Cache 过多 | 1. 减小 `batch_size` (如 8→4)<br>2. 减小 `max_seq_len`<br>3. 启用 `kv_cache.paging=true`<br>4. 使用量化模型 (Q4_K_M) | P0 |
| **请求超时 (504 Gateway Timeout)** | 队列积压、模型计算慢、网络延迟 | 1. 检查 `worker_queue_length` 指标<br>2. 减小 `max_concurrent` 或增大超时时间<br>3. 检查 GPU 利用率 (`nvidia-smi`) | P1 |
| **高错误率 (>5%)** | 输入数据格式错误、模型数值不稳定 | 1. 查看错误日志定位具体原因<br>2. 检查输入 prompt 长度是否超限<br>3. 验证 tokenizer 配置 | P0 |
| **KV Cache OOM** | 序列过长、并发请求多、缓存未释放 | 1. 启用分页: `kv_cache.paging=true`<br>2. 设置 `kv_cache.max_pages`<br>3. 重启服务清理缓存 | P1 |
| **响应格式错误 (JSON 解析失败)** | 流式输出中断、编码问题 | 1. 检查客户端是否正确处理 SSE (Server-Sent Events)<br>2. 确认 Content-Type 为 `application/json` 或 `text/event-stream`<br>3. 查看 HTTP 状态码 | P2 |
| **性能突然下降** | GPU 温度过高降频、CPU 竞争、磁盘 I/O 瓶颈 | 1. 检查 GPU 温度 (`nvidia-smi -q -g TEMPERATURE`)<br>2. 查看系统负载 (`top`, `htop`)<br>3. 检查磁盘 IO (`iostat -x 1`) | P1 |
| **连接被拒绝 (Connection Refused)** | 服务未启动、端口占用、防火墙拦截 | 1. 确认服务正在运行 (`ps aux \| grep openmini`)<br>2. 检查端口占用 (`netstat -tlnp \| grep 8000`)<br>3. 检查防火墙规则 (`iptables -L`) | P0 |

### 5.2 诊断命令速查

```bash
# ===== 服务状态检查 =====

# 检查进程是否运行
ps aux | grep openmini

# 检查端口监听
netstat -tlnp | grep -E '(8000|9090)'
# 或
ss -tlnp | grep -E '(8000|9090)'

# 健康检查端点
curl -s http://localhost:8000/health | jq .
# 预期输出: {"status":"healthy","model":"loaded","uptime":3600}

# ===== GPU 状态检查 =====

# GPU 概览
nvidia-smi

# 详细 GPU 信息 (温度、频率、功耗)
nvidia-smi -q -g TEMPERATURE,CLOCK,POWER

# GPU 进程占用
nvidia-smi pmon -c 1

# ===== 系统资源检查 =====

# CPU 和内存使用
top -bn1 | head -20
# 或
htop

# 磁盘空间
df -h
df -h /dev/shm  # 共享内存 (shm)

# 网络连接统计
ss -s

# ===== 日志查看 =====

# 实时查看日志 (JSON 格式)
tail -f /var/log/openmini/server.log | jq '{timestamp, level, message}'

# 过滤错误日志
grep '"level":"error"' /var/log/openmini/server.log | tail -100

# 统计最近 1 小时的错误数
awk '/"level":"error"/' /var/log/openmini/server.log | grep "$(date -d '1 hour ago' +%Y-%m-%dT%H)" | wc -l

# ===== Prometheus 指标查询 =====

# 直接访问 metrics 端点
curl -s http://localhost:9090/metrics | grep openmini

# 使用 promtool 验证规则
promtool test rules test_rules.yml

# ===== 性能分析 =====

# 使用 perf 分析 CPU 瓶颈 (Linux)
perf top -p $(pgrep -f openmini-server)

# 使用 strace 追踪系统调用 (谨慎使用，有性能开销)
strace -p $(pgrep -f openmini-server) -c -f
```

### 5.3 日志关键字搜索

```bash
# 搜索特定错误码
grep "ENG001\|HW001\|CFG001" /var/log/openmini/server.log

# 搜索 panic 或 unwrap 失败
grep -E "(panic|unwrap|expect.*failed)" /var/log/openmini/server.log

# 搜索超时相关
grep -i "timeout\|timed.out" /var/log/openmini/server.log

# 搜索内存分配失败
grep -i "out.of.memory\|allocation.failed\|oom" /var/log/openmini/server.log

# 搜索 CUDA 相关错误
grep -i "cuda\|gpu\|kernelfail" /var/log/openmini/server.log
```

---

## 6. 性能调优参数

### 6.1 吞吐量优化 (Tokens/sec ↑)

| 参数 | 当前值 | 优化建议 | 预期效果 |
|------|-------|---------|---------|
| `scheduler.batch_size` | 8 | → 16 (GPU) / → 4 (低延迟) | 吞吐量 +80% 或 延迟 -50% |
| `scheduler.max_concurrent` | 8 | → 16 (如果 GPU 利用率 < 70%) | 吞吐量 +50-100% |
| `model.quantization` | FP16 | → Q4_K_M (4-bit) | 显存 -75%, 吞吐量 +30% |
| `kv_cache.paging` | false | → true | 支持更长序列, 减少 OOM |
| `simd.enabled` | true | ✅ 保持开启 | Softmax 加速 4x (AVX2) |

### 6.2 延迟优化 (Time to First Token ↓)

| 参数 | 当前值 | 优化建议 | 预期效果 |
|------|-------|---------|---------|
| `scheduler.batch_timeout_ms` | 5 | → 2 (实时对话) | TTFT -60% |
| `scheduler.batch_size` | 8 | → 4 | TTFT -40% |
| `scheduler.queue_capacity` | 1000 | → 200 (减少排队) | 排队时间 -80% |
| `model.max_seq_len` | 8192 | → 2048 (短文本场景) | 内存占用 -75% |
| `prefetch.enabled` | false | → true (实验性) | 预取加速 +20% |

### 6.3 不同场景的推荐配置

#### 场景 A: 高吞吐离线处理 (Batch Inference)

```toml
[scheduler]
max_concurrent = 16
queue_capacity = 5000
batch_size = 32
batch_timeout_ms = 50

[model]
max_seq_len = 4096

[kv_cache]
paging = true
max_pages = 2048
```

**预期性能**: ~2000+ tokens/sec, P99 latency < 5s

#### 场景 B: 低延迟实时对话 (Chat API)

```toml
[scheduler]
max_concurrent = 4
queue_capacity = 200
batch_size = 4
batch_timeout_ms = 2

[model]
max_seq_len = 2048

[kv_cache]
paging = true
prefix_cache_max_entries = 500  # 启用前缀缓存加速重复查询
```

**预期性能**: TTFT < 100ms, P99 latency < 2s

#### 场景 C: 成本优化边缘设备 (Edge/IoT)

```toml
[scheduler]
max_concurrent = 2
queue_capacity = 50
batch_size = 2
batch_timeout_ms = 10

[model]
# 使用小型量化模型
path = "/models/tinyllama-1.1b-Q3_K_S.gguf"
max_seq_len = 1024

[kv_cache]
paging = true
max_pages = 256
cpu_offload_bytes = 536870912  # 512MB CPU offload
```

**预期性能**: 内存 < 2GB, 适合 Raspberry Pi 4/Jetson Nano

---

## 7. 内存优化建议

### 7.1 内存分布概览

```
┌─────────────────────────────────────────────────────┐
│                  总物理内存                          │
│  ┌───────────────────────────────────────────────┐ │
│  │              GPU 显存 (VRAM)                   │ │
│  │  ┌─────────┐ ┌──────────┐ ┌────────────────┐ │ │
│  │  │ 模型权重 │ │ KV Cache │ │ 激活值/临时     │ │ │
│  │  │ (~2-8GB)│ │ (~4-20GB)│ │ (~1-4GB)       │ │ │
│  │  └─────────┘ └──────────┘ └────────────────┘ │ │
│  └───────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────┐ │
│  │              系统 RAM                         │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │ │
│  │  │ 二进制/   │ │ 任务队列  │ │ CPU Offload  │ │ │
│  │  │ 运行时    │ │ (~10MB)  │ │ KV Cache     │ │ │
│  │  │ (~200MB) │ │           │ │ (~可选)       │ │ │
│  │  └──────────┘ └──────────┘ └──────────────┘ │ │
│  └───────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### 7.2 各组件内存占用估算

| 组件 | FP16 模型 | Q4_K_M 量化 | Q3_K_S 量化 |
|------|----------|-------------|-------------|
| **Llama-3-8B 权重** | ~16 GB | ~5 GB | ~4 GB |
| **Llama-3-70B 权重** | ~140 GB | ~40 GB | ~32 GB |
| **KV Cache (seq=4096)** | ~8 GB/层 | ~8 GB/层 | ~8 GB/层 |
| **激活值 (batch=8)** | ~2 GB | ~2 GB | ~2 GB |
| **运行时开销** | ~500 MB | ~500 MB | ~500 MB |

### 7.3 内存优化策略

#### 策略 1: 模型量化 (最有效)

```bash
# 将 FP16 模型转换为 Q4_K_M (节省 75% 显存)
python convert_hf_to_gguf.py \
    --model-name-or-path meta-llama/Llama-3-8B \
    --outfile llama-3-8b-Q4_K_M.gguf \
    --outtype q4_k_m
```

**效果**: 16GB → 5GB, 对精度影响 < 1%

#### 策略 2: KV Cache 分页

```toml
[kv_cache]
paging = true
max_pages = 512  # 限制最大页数
eviction_policy = "lru"
```

**效果**: 自动将不活跃的 KV Cache 页换出到 CPU 内存

#### 策略 3: 动态批处理调整

根据当前内存压力自动调整 batch_size：

```rust,ignore
// 伪代码示例
let gpu_memory_used = get_gpu_memory_usage();
let gpu_memory_total = get_gpu_memory_total();
let usage_ratio = gpu_memory_used as f64 / gpu_memory_total as f64;

if usage_ratio > 0.85 {
    // 高内存压力: 减小 batch
    scheduler.set_batch_size(4);
} else if usage_ratio < 0.6 {
    // 低内存压力: 增大 batch 提高吞吐
    scheduler.set_batch_size(16);
}
```

#### 策略 4: 激活值卸载 (Checkpointing)

对于超长序列，启用梯度/激活值 checkpointing：

```toml
[model]
activation_checkpointing = true  # 用计算换内存
```

**效果**: 显存减少 30-50%, 计算时间增加 20%

---

## 8. GPU 内存管理

### 8.1 监控 GPU 内存使用

```bash
# 实时监控 (每秒刷新)
watch -n 1 nvidia-smi

# 详细内存信息
nvidia-smi -q -d MEMORY

# Python 脚本持续记录 (用于绘图)
while true; do
    echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits,noheader)"
    sleep 1
done >> gpu_memory.log
```

### 8.2 常见 GPU 内存问题

#### 问题 1: CUDA OOM (Out of Memory)

**症状**: 日志中出现 `cudaErrorOutOfMemory` 或 `RuntimeError: CUDA error: out of memory`

**解决方案**:

1. **减小 batch_size**
   ```toml
   scheduler.batch_size = 4  # 从 8 降到 4
   ```

2. **启用 KV Cache 分页**
   ```toml
   kv_cache.paging = true
   kv_cache.cpu_offload_bytes = 1073741824  # 允许 1GB CPU offload
   ```

3. **使用量化模型**
   ```bash
   # FP16 → Q4_K_M (节省 75% 显存)
   ```

4. **清理 GPU 缓存**
   ```bash
   # 重启服务 (最彻底)
   sudo systemctl restart openmini
   
   # 或在代码中手动释放 (如果有 API)
   ```

#### 问题 2: 显存碎片化

**症状**: 总空闲显存足够，但无法分配大块连续内存

**解决方案**:

1. **定期重启服务** (每周或每月)
2. **使用内存池分配器** (CUDA Memory Pool)
   ```bash
   export CUDA_MEM_POOL=1
   ```
3. **预分配显存** (避免运行时分配)
   ```bash
   export CUDA_MALLOC_ASYNC=0  # 禁用异步分配
   export CUDA_PREALLOC=1      # 启用预分配
   ```

#### 问题 3: GPU 显存泄漏

**症状**: 显存使用量随时间持续增长，直到 OOM

**诊断方法**:

```bash
# 1. 监控显存增长趋势
watch -n 60 'nvidia-smi | grep openmini'

# 2. 使用 cuda-memcheck 检测泄漏
cuda-memcheck --leak-check full ./target/release/openmini-server

# 3. 检查是否有未释放的张量
# 在代码中添加日志打印分配/释放事件
```

**解决方案**:

1. **升级到最新版本** (可能已修复泄漏 bug)
2. **减少并发请求数** (`max_concurrent`)
3. **定期重启** 作为临时缓解措施

### 8.3 多 GPU 管理 (未来支持)

当前版本主要针对单 GPU 优化。多 GPU 支持规划中：

- **模型并行**: 将大模型分割到多张卡 (Megatron-LM style)
- **数据并行**: 多卡副本处理不同 batch (DDP)
- **张量并行**: 单层计算分散到多卡 (DeepSpeed style)

**临时方案**: 使用多个单卡实例 + 负载均衡

```nginx
# Nginx 负载均衡配置示例
upstream openmini_backend {
    server 10.0.0.1:8000;  # GPU 0
    server 10.0.0.2:8000;  # GPU 1
    server 10.0.0.3:8000;  # GPU 2
    server 10.0.0.4:8000;  # GPU 3
}

server {
    listen 80;
    
    location /v1/ {
        proxy_pass http://openmini_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 9. 维护操作流程

### 9.1 模型热更新 (Hot Reload)

**适用场景**: 更新模型权重、切换不同版本模型、A/B 测试

**前置条件**:

1. 新模型文件已准备好且格式正确
2. 新旧模型兼容相同的 tokenizer
3. 有足够的内存同时加载两个模型 (短暂重叠)

**操作步骤**:

```bash
#!/bin/bash
# model_hot_reload.sh

set -e

OLD_MODEL="/models/llama-3-8b-v1.gguf"
NEW_MODEL="/models/llama-3-8b-v2.gguf"
BACKUP_DIR="/models/backups/$(date +%Y%m%d_%H%M%S)"

echo "=== 开始模型热更新 ==="

# 1. 验证新模型文件存在且可读
if [ ! -f "$NEW_MODEL" ]; then
    echo "错误: 新模型文件不存在: $NEW_MODEL"
    exit 1
fi

echo "[1/5] 验证新模型文件... ✓"

# 2. 创建备份目录
mkdir -p "$BACKUP_DIR"
cp "$OLD_MODEL" "$BACKUP_DIR/"

echo "[2/5] 备份旧模型到 $BACKUP_DIR ... ✓"

# 3. 通过 API 触发热更新 (非破坏性)
# 先标记旧模型即将卸载
curl -s -X POST http://localhost:8000/admin/model/preunload \
    -H "Content-Type: application/json" \
    -d '{"model_path": "'"$OLD_MODEL"'"}'

sleep 5  # 等待当前请求完成

echo "[3/5] 通知服务准备切换... ✓"

# 4. 执行模型切换
curl -s -X POST http://localhost:8000/admin/model/load \
    -H "Content-Type: application/json" \
    -d '{"model_path": "'"$NEW_MODEL"'", "strategy": "hot_swap"}'

echo "[4/5] 加载新模型... ✓"

# 5. 验证新模型可用
sleep 10
HEALTH=$(curl -s http://localhost:8000/health)
echo "$HEALTH" | jq .

if echo "$HEALTH" | grep -q '"model_loaded":true'; then
    echo "[5/5] 验证通过! 热更新成功 ✓✓✓"
else
    echo "[5/5] ⚠️ 验证失败，正在回滚..."
    curl -s -X POST http://localhost:8000/admin/model/load \
        -H "Content-Type: application/json" \
        -d '{"model_path": "'"$OLD_MODEL"'"}'
    exit 1
fi

echo "=== 热更新完成 ==="
```

**注意事项**:

- 热更新期间会有短暂的不可用窗口 (~10-30 秒)
- 正在处理的请求会完成，新请求会排队等待
- 如果新模型加载失败，自动回滚到旧模型
- 建议在低峰期执行 (如凌晨 3-5 点)

### 9.2 KV Cache 备份与恢复

**适用场景**: 服务迁移、灾难恢复、长期会话保存

#### 备份 KV Cache

```bash
#!/bin/bash
# backup_kv_cache.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="/backups/kv_cache_${TIMESTAMP}.bin"

echo "开始备份 KV Cache..."

# 1. 通过 API 触发快照
curl -s -X POST http://localhost:8000/admin/kv-cache/snapshot \
    -H "Content-Type: application/json" \
    -d '{"output_path": "'"$BACKUP_FILE"'"}' \
    --output "$BACKUP_FILE.progress"

# 2. 等待快照完成
echo "快照生成中..."
sleep 30

# 3. 验证备份完整性
if [ -f "$BACKUP_FILE" ]; then
    SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    echo "备份完成: $BACKUP_FILE ($SIZE)"
else
    echo "备份失败!"
    exit 1
fi
```

#### 恢复 KV Cache

```bash
#!/bin/bash
# restore_kv_cache.sh

BACKUP_FILE=$1  # 从命令行参数获取备份文件路径

if [ -z "$BACKUP_FILE" ] || [ ! -f "$BACKUP_FILE" ]; then
    echo "用法: $0 <backup_file.bin>"
    exit 1
fi

echo "从 $BACKUP_FILE 恢复 KV Cache..."

# 1. 停止接受新请求
curl -s -X POST http://localhost:8000/admin/maintenance/drain \
    -H "Content-Type: application/json" \
    -d '{"grace_period_seconds": 60}'

echo "等待当前请求完成 (60秒)..."

# 2. 执行恢复
curl -s -X POST http://localhost:8000/admin/kv-cache/restore \
    -H "Content-Type: application/json" \
    -d '{"input_path": "'"$BACKUP_FILE"'"}'

echo "恢复中..."

# 3. 恢复服务
sleep 30
curl -s -X POST http://localhost:8000/admin/maintenance/resume

echo "恢复完成，服务已重新上线"
```

### 9.3 健康检查解读

#### 健康检查端点响应格式

```json
{
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00Z",
    "uptime_seconds": 86400,
    "version": "0.1.0",
    "components": {
        "model": {
            "status": "loaded",
            "name": "llama-3-8b-instruct",
            "memory_mb": 5120,
            "quantization": "Q4_K_M"
        },
        "gpu": {
            "status": "available",
            "device_name": "NVIDIA A10G",
            "memory_total_gb": 24,
            "memory_used_gb": 18,
            "utilization_percent": 75
        },
        "scheduler": {
            "status": "running",
            "active_tasks": 4,
            "queued_tasks": 12,
            "completed_total": 1234567
        },
        "kv_cache": {
            "status": "normal",
            "pages_used": 256,
            "pages_total": 512,
            "memory_mb": 4096
        }
    },
    "metrics_summary": {
        "requests_last_5min": 1500,
        "avg_latency_ms": 120,
        "p99_latency_ms": 350,
        "error_rate_percent": 0.02,
        "tokens_per_second": 2450
    }
}
```

#### 状态判断逻辑

| 字段 | 正常值 | 警告值 | 异常值 | 处理措施 |
|------|-------|--------|--------|---------|
| `status` | `"healthy"` | `"degraded"` | `"unhealthy"` | degraded: 监控观察; unhealthy: 立即排查 |
| `model.status` | `"loaded"` | `"loading"` | `"not_loaded"` | not_loaded: 检查模型路径和权限 |
| `gpu.utilization_percent` | 60-90% | < 30% or > 95% | 100% (瓶颈) | 低利用率: 增大 batch; 高利用率: 减小 batch |
| `scheduler.queued_tasks` | < 50 | 50-200 | > 200 | 增加容量或限流 |
| `kv_cache.pages_used/pages_total` | < 70% | 70-90% | > 90% | 启用分页或减小 max_seq_len |
| `error_rate_percent` | < 0.1% | 0.1-1% | > 1% | 检查日志定位错误原因 |
| `avg_latency_ms` | < 200 | 200-500 | > 500 | 检查队列积压和 GPU 状态 |

### 9.4 优雅关闭 (Graceful Shutdown)

**目标**: 在关闭过程中：
1. 停止接受新请求
2. 完成正在处理的请求 (最多等待 30 秒)
3. 保存必要的检查点和缓存
4. 释放所有资源 (GPU 内存、文件句柄等)
5. 退出进程 (exit code 0)

#### 手动触发优雅关闭

```bash
# 方式 1: SIGTERM 信号 (推荐)
kill -TERM $(pgrep -f openmini-server)

# 方式 2: 通过 API
curl -X POST http://localhost:8000/admin/shutdown \
    -H "Content-Type: application/json" \
    -d '{"grace_period_seconds": 30, "reason": "scheduled maintenance"}'

# 方式 3: Ctrl+C (前台运行时)
# 按 Ctrl+C 发送 SIGINT
```

#### 关闭过程监控

```bash
# 监控关闭进度
watch -n 1 'curl -s http://localhost:8000/health | jq ".components.scheduler.active_tasks, .status"'
```

预期输出变化:

```
# 关闭前
"active_tasks": 8, "status": "healthy"

# 关闭中 (draining)
"active_tasks": 4, "status": "degraded"

# 关闭完成 (所有请求已完成)
"active_tasks": 0, "status": "shutting_down"

# 服务停止 (HTTP 不再响应)
curl: (7) Failed to connect to localhost port 8000...
```

#### 强制关闭 (紧急情况)

**警告**: 强制关闭可能导致：
- 正在处理的请求丢失 (客户端收到错误)
- KV Cache 未保存 (需要重建)
- 检查点未写入 (训练进度丢失)

```bash
# 仅在正常关闭超时后使用!
kill -9 $(pgrep -f openmini-server)

# 或
kill -SIGKILL $(pgrep -f openmini-server)
```

### 9.5 定期维护检查清单

#### 每日检查 (自动化脚本)

```bash
#!/bin/bash
# daily_health_check.sh

echo "=== OpenMini 每日健康检查 ==="
DATE=$(date +%Y-%m-%d)
LOG_FILE="/var/log/openmini/daily_check_${DATE}.log"

# 1. 服务状态
echo "[1/8] 检查服务状态..." | tee -a $LOG_FILE
if pgrep -f openmini-server > /dev/null; then
    echo "  ✓ 服务运行中 (PID: $(pgrep -f openmini-server))" | tee -a $LOG_FILE
else
    echo "  ✗ 服务未运行!" | tee -a $LOG_FILE
    # 触发告警或自动重启
fi

# 2. 健康检查端点
echo "[2/8] 检查健康端点..." | tee -a $LOG_FILE
HEALTH=$(curl -sf http://localhost:8000/health 2>/dev/null || echo "{}")
STATUS=$(echo $HEALTH | jq -r '.status // "unknown"')
echo "  状态: $STATUS" | tee -a $LOG_FILE

# 3. GPU 状态
echo "[3/8] 检查 GPU..." | tee -a $LOG_FILE
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "N/A")
GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "N/A")
echo "  GPU 利用率: ${GPU_UTIL}%, 显存: ${GPU_MEM} MB" | tee -a $LOG_FILE

# 4. 磁盘空间
echo "[4/8] 检查磁盘空间..." | tee -a $LOG_FILE
DISK_USAGE=$(df -h / | awk 'NR==2{print $5}')
echo "  磁盘使用率: $DISK_USAGE" | tee -a $LOG_FILE

# 5. 错误日志统计
echo "[5/8] 统计近 24h 错误数..." | tee -a $LOG_FILE
ERROR_COUNT=$(grep '"level":"error"' /var/log/openmini/server.log 2>/dev/null | grep "$(date -d '24 hours ago' +%Y-%m-%dT%H)" | wc -l)
echo "  错误数: $ERROR_COUNT" | tee -a $LOG_FILE

# 6-8. 更多检查项...

echo "=== 检查完成 ===" | tee -a $LOG_FILE
```

#### 每周维护任务

- [ ] 清理旧日志文件 (保留最近 7 天)
- [ ] 检查磁盘空间并清理临时文件
- [ ] 备份 KV Cache 到远程存储
- [ ] 检查 Prometheus/Grafana 告警规则有效性
- [ ] 审计安全日志 (登录、API Key 使用)
- [ ] 检查依赖项更新 (Rust crates, CUDA driver)

#### 每月维护任务

- [ ] 全量备份模型权重和配置
- [ ] 性能基准测试并与上月对比
- [ ] 容量规划评估 (是否需要扩容)
- [ ] 安全补丁更新 (OS, CUDA, Rust toolchain)
- [ ] 文档更新 (本运维手册、API 文档)
- [ ] 灾难恢复演练 (模拟故障并验证恢复流程)

---

## 附录

### A. 常用命令速查

```bash
# 启动服务
./target/release/openmini-server --config config/server.toml

# 后台运行
nohup ./target/release/openmini-server --config config/server.toml > server.log 2>&1 &

# 停止服务 (优雅)
kill -TERM $(pgrep -f openmini-server)

# 查看版本
./target/release/openmini-server --version

# 查看帮助
./target/release/openmini-server --help
```

### B. 故障上报模板

当遇到需要社区帮助的问题时，请提供：

```markdown
## 问题描述
(简要描述问题现象)

## 环境信息
- OS: (uname -a)
- Rust version: (rustc --version)
- CUDA version: (nvcc --version)
- GPU型号: (nvidia-smi -L)
- OpenMini version: (openmini-server --version)

## 复现步骤
1.
2.
3.

## 期望行为
(描述你期望的正确行为)

## 实际行为
(描述实际发生的错误行为)

## 日志片段
(粘贴相关的错误日志，注意脱敏)

## 配置文件
(粘贴 server.toml 敏感部分用 *** 替代)
```

### C. 相关文档链接

- [API 文档](../API.md)
- [架构设计文档](../architecture/001-simplify-worker-pool.md)
- [WorkerPool 迁移指南](../migration_worker_pool_to_scheduler.md)
- [用户使用手册](../USER_GUIDE.md)
- [生产部署指南](../PRODUCTION_DEPLOYMENT.md)
- [GitHub Issues](https://github.com/openmini/openmini-v1/issues)

---

**文档版本**: v1.0.0  
**最后更新**: 2026-04-10  
**维护者**: OpenMini 运维团队  
**反馈渠道**: [GitHub Discussions](https://github.com/openmini/openmini-v1/discussions)
