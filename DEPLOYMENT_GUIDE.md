# OpenMini 部署指南

## 目录

- [系统要求](#系统要求)
- [安装步骤](#安装步骤)
- [配置说明](#配置说明)
- [启动命令](#启动命令)
- [验证部署](#验证部署)
- [常见问题](#常见问题)

---

## 系统要求

### 硬件要求

| 配置项 | 最低要求 | 推荐配置 |
|--------|----------|----------|
| CPU | 4 核 | 8 核+ |
| 内存 | 8 GB | 16 GB+ |
| 存储 | 10 GB | 20 GB+ SSD |
| GPU | 可选 | NVIDIA GPU (CUDA 12.0+) |

### 软件要求

| 软件 | 版本要求 |
|------|----------|
| Rust | 1.70+ |
| Python | 3.8+ (客户端) |
| Protoc | 3.20+ |
| CUDA | 12.0+ (GPU 加速) |

### 操作系统支持

- **macOS**: 10.15+ (Intel/Apple Silicon)
- **Linux**: Ubuntu 20.04+, CentOS 8+
- **Windows**: Windows 10+ (WSL2)

---

## 安装步骤

### 方式一: 从源码编译 (推荐)

#### 1. 安装 Rust

```bash
# macOS/Linux
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 验证安装
rustc --version
cargo --version
```

#### 2. 安装 Protocol Buffers

```bash
# macOS
brew install protobuf

# Ubuntu/Debian
sudo apt-get install -y protobuf-compiler

# CentOS/RHEL
sudo yum install -y protobuf-compiler

# 验证安装
protoc --version
```

#### 3. 克隆项目

```bash
git clone https://github.com/openmini/openmini-v1.git
cd openmini-v1
```

#### 4. 编译服务端

```bash
# CPU 版本
cargo build --release

# GPU 版本 (需要 CUDA)
cargo build --release --features cuda

# Metal 版本 (macOS)
cargo build --release --features metal
```

#### 5. 安装 Python 客户端

```bash
# 安装依赖
pip install grpcio grpcio-tools

# 生成 Python protobuf 文件
protoc --python_out=. --grpc_python_out=. \
    openmini-proto/proto/openmini.proto

# 安装客户端
cd openmini-client
pip install -e .
```

### 方式二: Docker 部署

#### 1. 构建 Docker 镜像

```bash
# CPU 版本
docker build -t openmini:latest .

# GPU 版本
docker build -t openmini:gpu --build-arg TARGET=gpu .
```

#### 2. 运行容器

```bash
# CPU 版本
docker run -d \
    --name openmini \
    -p 50051:50051 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/config:/app/config \
    --memory=12g \
    openmini:latest

# GPU 版本
docker run -d \
    --name openmini \
    --gpus all \
    -p 50051:50051 \
    -v $(pwd)/models:/app/models \
    openmini:gpu
```

### 方式三: 预编译二进制

```bash
# 下载最新版本
wget https://github.com/openmini/openmini-v1/releases/latest/download/openmini-server-linux-x86_64.tar.gz

# 解压
tar -xzf openmini-server-linux-x86_64.tar.gz

# 赋予执行权限
chmod +x openmini-server
```

---

## 配置说明

### 配置文件位置

默认配置文件: `config/server.toml`

### 完整配置示例

```toml
[server]
host = "0.0.0.0"           # 监听地址
port = 50051               # 监听端口
max_connections = 100      # 最大连接数
request_timeout_ms = 60000 # 请求超时(毫秒)

[thread_pool]
size = 4                   # 线程池大小 (建议设为 CPU 核心数)
stack_size_kb = 8192       # 线程栈大小(KB)

[memory]
max_memory_gb = 12         # 最大可用内存(GB)
model_memory_gb = 6        # 模型内存上限(GB)
cache_memory_gb = 4        # KV 缓存内存上限(GB)
arena_size_mb = 256        # Arena 分配器大小(MB)

[model]
path = "models/openmini-v1-q4_k_m.gguf"  # 模型文件路径
quantization = "Q4_K_M"    # 量化类型
context_length = 4096      # 上下文长度

[worker]
count = 3                  # Worker 进程数
restart_on_failure = true  # 失败时自动重启
health_check_interval_ms = 5000  # 健康检查间隔

[grpc]
max_message_size_mb = 100  # 最大消息大小(MB)
keepalive_time_ms = 30000  # Keepalive 时间
keepalive_timeout_ms = 10000  # Keepalive 超时
```

### 配置项详解

#### [server] - 服务器配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| host | string | "0.0.0.0" | 监听地址，"0.0.0.0" 表示所有网卡 |
| port | int | 50051 | gRPC 服务端口 |
| max_connections | int | 100 | 最大并发连接数 |
| request_timeout_ms | int | 60000 | 请求超时时间(毫秒) |

#### [thread_pool] - 线程池配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| size | int | CPU 核心数 | 工作线程数量 |
| stack_size_kb | int | 8192 | 每个线程的栈大小 |

#### [memory] - 内存配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| max_memory_gb | int | 12 | 服务最大可用内存 |
| model_memory_gb | int | 6 | 模型加载内存上限 |
| cache_memory_gb | int | 4 | KV 缓存内存上限 |
| arena_size_mb | int | 256 | 内存池 Arena 大小 |

**内存分配策略**:
- 0-4 GB: SmallArena (低内存设备)
- 5-8 GB: StandardArena (标准配置)
- 9-16 GB: PagedAttention (推荐配置)
- 16+ GB: Distributed (高配服务器)

#### [model] - 模型配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| path | string | "models/..." | GGUF 模型文件路径 |
| quantization | string | "Q4_K_M" | 量化类型 |
| context_length | int | 4096 | 最大上下文长度 |

**支持的量化类型**:
- `Q4_K_M`: 4-bit 量化，平衡质量和速度
- `Q5_K_M`: 5-bit 量化，更高质量
- `Q8_0`: 8-bit 量化，最高质量
- `FP16`: 半精度浮点，需要更多内存

#### [worker] - Worker 配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| count | int | 3 | Worker 进程数量 |
| restart_on_failure | bool | true | 失败时自动重启 |
| health_check_interval_ms | int | 5000 | 健康检查间隔 |

#### [grpc] - gRPC 配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| max_message_size_mb | int | 100 | 最大消息大小 |
| keepalive_time_ms | int | 30000 | Keepalive 间隔 |
| keepalive_timeout_ms | int | 10000 | Keepalive 超时 |

---

## 启动命令

### 基本启动

```bash
# 使用默认配置
./openmini-server

# 指定配置文件
./openmini-server --config config/server.toml

# 指定模型路径
./openmini-server --model models/openmini-v1-q4_k_m.gguf
```

### 后台运行

```bash
# 使用 nohup
nohup ./openmini-server > logs/server.log 2>&1 &

# 使用 systemd (推荐)
sudo systemctl start openmini
sudo systemctl enable openmini  # 开机自启
```

### Systemd 服务配置

创建文件 `/etc/systemd/system/openmini.service`:

```ini
[Unit]
Description=OpenMini Inference Server
After=network.target

[Service]
Type=simple
User=openmini
WorkingDirectory=/opt/openmini
ExecStart=/opt/openmini/openmini-server --config /opt/openmini/config/server.toml
Restart=on-failure
RestartSec=5
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

启动服务:

```bash
sudo systemctl daemon-reload
sudo systemctl start openmini
sudo systemctl status openmini
```

### Docker Compose

创建 `docker-compose.yml`:

```yaml
version: '3.8'

services:
  openmini:
    image: openmini:latest
    container_name: openmini
    ports:
      - "50051:50051"
    volumes:
      - ./models:/app/models
      - ./config:/app/config
    environment:
      - RUST_LOG=info
    deploy:
      resources:
        limits:
          memory: 12G
    restart: unless-stopped
```

启动:

```bash
docker-compose up -d
```

---

## 验证部署

### 1. 检查服务状态

```bash
# 检查进程
ps aux | grep openmini

# 检查端口
lsof -i :50051
netstat -tlnp | grep 50051
```

### 2. 健康检查

```bash
# 使用 grpcurl
grpcurl -plaintext localhost:50051 openmini.OpenMini/HealthCheck

# 使用 Python 客户端
python -c "
from openmini_client import OpenMiniClient
client = OpenMiniClient()
print('服务状态:', '正常' if client.health_check() else '异常')
"
```

### 3. 测试对话

```bash
# 使用 Python 测试
python chat.py

# 或使用提供的测试脚本
python -c "
from openmini_client import OpenMiniClient, Message

client = OpenMiniClient()
messages = [Message(role='user', content='你好')]

for resp in client.chat(messages):
    print(resp.token, end='')
print()
"
```

### 4. 性能测试

```bash
# 运行基准测试
cargo bench --package openmini-server

# 运行集成测试
cargo test --package openmini-server -- --test-threads=1
```

---

## 常见问题

### Q1: 端口被占用

**错误信息**: `Address already in use`

**解决方案**:
```bash
# 查找占用进程
lsof -i :50051

# 终止进程
kill -9 <PID>

# 或更换端口
./openmini-server --port 50052
```

### Q2: 内存不足

**错误信息**: `Cannot allocate memory` 或 `Out of memory`

**解决方案**:
- 减小 `memory.max_memory_gb` 配置
- 使用更大量化的模型 (如 Q4_K_M)
- 减少 `worker.count`
- 增加系统交换空间

### Q3: 模型加载失败

**错误信息**: `Failed to load model`

**解决方案**:
```bash
# 检查模型文件
ls -la models/

# 验证模型完整性
md5sum models/openmini-v1-q4_k_m.gguf

# 检查文件权限
chmod 644 models/*.gguf
```

### Q4: GPU 未被识别

**错误信息**: `CUDA not available`

**解决方案**:
```bash
# 检查 CUDA 安装
nvidia-smi
nvcc --version

# 检查 CUDA 库
ldconfig -p | grep cuda

# 重新编译 GPU 版本
cargo build --release --features cuda
```

### Q5: macOS 编译错误

**错误信息**: `linker 'cc' not found`

**解决方案**:
```bash
# 安装 Xcode 命令行工具
xcode-select --install

# 安装依赖
brew install openssl protobuf
```

### Q6: 连接超时

**错误信息**: `Deadline exceeded`

**解决方案**:
- 增加 `request_timeout_ms` 配置
- 检查网络连接
- 减少 `max_tokens` 参数

---

## 监控与日志

### 日志配置

```bash
# 设置日志级别
export RUST_LOG=info,openmini_server=debug

# 输出到文件
RUST_LOG=info ./openmini-server 2>&1 | tee logs/server.log
```

### 监控指标

服务提供以下监控指标:

- 请求数/秒
- 平均响应时间
- 内存使用率
- GPU 使用率 (如适用)
- 连接数

### Prometheus 集成

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'openmini'
    static_configs:
      - targets: ['localhost:9090']
```

---

## 升级指南

### 从旧版本升级

```bash
# 1. 停止服务
sudo systemctl stop openmini

# 2. 备份配置
cp -r config config.bak

# 3. 更新代码
git pull origin main

# 4. 重新编译
cargo build --release

# 5. 检查配置变更
diff config/server.toml config.bak/server.toml

# 6. 启动服务
sudo systemctl start openmini
```

---

## 相关链接

- [API 文档](docs/API.md)
- [GitHub 仓库](https://github.com/openmini/openmini-v1)
- [问题反馈](https://github.com/openmini/openmini-v1/issues)
