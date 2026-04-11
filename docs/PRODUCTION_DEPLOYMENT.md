# OpenMini-V1 生产环境部署指南

**版本**: v1.2.0+  
**适用场景**: 生产环境、企业级部署  
**难度等级**: 中级 (需要 Linux 系统管理基础)

---

## 📖 目录

1. [部署架构](#1-部署架构)
2. [环境准备](#2-环境准备)
3. [安装部署](#3-安装部署)
4. [配置详解](#4-配置详解)
5. [运维管理](#5-运维管理)
6. [安全加固](#6-安全加固)
7. [升级维护](#7-升级维护)
8. [高可用方案](#8-高可用方案)

---

## 1. 部署架构

### 1.1 推荐架构图 (单机部署)

```
┌─────────────────────────────────────────────────────┐
│                    客户端层                          │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│   │ Web App  │  │ Python   │  │ gRPC     │        │
│   │ (Admin)  │  │ Client   │  │ Client   │        │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│        │              │              │              │
└────────┼──────────────┼──────────────┼──────────────┘
         │              │              │
         ▼              ▼              ▼
┌─────────────────────────────────────────────────────┐
│                  反向代理 (Nginx)                    │
│   HTTPS:443 → HTTP:8080 (REST API)                 │
│   gRPC:50051 → 直连或通过 TCP 负载均衡               │
└──────────────────────┬──────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
┌─────────────────┐       ┌─────────────────┐
│  OpenMini Server │       │  Admin Service  │
│  :8080 / :50051  │       │  :3001           │
│                  │       │                  │
│  - 模型推理      │       │  - 用户管理      │
│  - KV Cache      │       │  - API Key 管理  │
│  - Worker Pool   │       │  - 监控面板      │
└────────┬──────────┘       └────────┬──────────┘
         │                           │
         └───────────┬───────────────┘
                     ▼
         ┌─────────────────────┐
         │    SQLite Database   │
         │  (admin.db)          │
         └─────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│   Model Files   │    │   Log Files      │
│   (/models/)    │    │   (/logs/)       │
└─────────────────┘    └─────────────────┘
```

### 1.2 组件说明

| 组件 | 用途 | 资源需求 |
|------|------|---------|
| **OpenMini Server** | 核心推理引擎 | 4-16 CPU, 8-64GB RAM, GPU(可选) |
| **Admin Service** | 管理后台 API | 1-2 CPU, 512MB-2GB RAM |
| **Nginx** | 反向代理 + TLS 终止 | 1 CPU, 256MB RAM |
| **SQLite** | 元数据存储 (用户/API Key) | <100MB |
| **Model Files** | GGUF 模型权重 | 3-50GB (取决于模型) |

### 1.3 网络拓扑

```
Internet
    │
    ├── Port 443 (HTTPS) → Nginx → Admin Panel (Vue3 SPA)
    │                              ↓
    │                         Admin API (:3001)
    │
    └── Port 50051 (gRPC) → OpenMini Server (直连，内网推荐)
    
Internal Network:
    - Admin API ↔ Server API (HTTP, 内网通信)
    - Prometheus → Server Metrics (:9090)
```

---

## 2. 环境准备

### 2.1 操作系统要求

**推荐发行版**:
- Ubuntu 22.04 LTS (长期支持至 2027 年)
- Debian 12 Bookworm
- CentOS Stream 9 / Rocky Linux 9

**内核要求**:
```bash
# 检查内核版本 (建议 ≥ 5.15)
uname -r
# 输出: 5.15.0-91-generic ✅
```

**系统优化**:

```bash
# 增加文件描述符限制
echo "* soft nofile 65535" >> /etc/security/limits.conf
echo "* hard nofile 65535" >> /etc/security/limits.conf

# 优化网络参数
cat >> /etc/sysctl.conf << EOF
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_tw_reuse = 1
net.core.netdev_max_backlog = 65535
EOF

sysctl -p
```

### 2.2 硬件规格建议

#### 推理服务器 (带 GPU)

| 规模 | QPS 目标 | CPU | 内存 | GPU | 存储 |
|------|----------|-----|------|-----|------|
| **小型** | 1-5 | 8 核 | 32 GB | RTX 3060 12GB | 100 GB SSD |
| **中型** | 5-20 | 16 核 | 64 GB | RTX 4090 24GB | 500 GB NVMe |
| **大型** | 20-100 | 32 核 | 128 GB | A100 80GB x2 | 1 TB NVMe |

#### 推理服务器 (纯 CPU)

| 规模 | QPS 目标 | CPU | 内存 | 存储 |
|------|----------|-----|------|------|
| **小型** | 0.5-2 | 16 核 AVX2 | 32 GB | 100 GB SSD |
| **中型** | 2-10 | 32 核 AVX-512 | 64 GB | 500 GB NVMe |
| **大型** | 10-30 | 64 核 AVX-512 | 128 GB | 1 TB NVMe |

### 2.3 依赖安装

#### Rust 工具链 (编译部署)

```bash
# 安装 Rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 验证版本 (≥ 1.75)
rustc --version
cargo --version
```

#### CUDA Toolkit (GPU 部署)

```bash
# NVIDIA 驱动 (Ubuntu)
sudo apt install nvidia-driver-535

# CUDA Toolkit 12.1
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run --silent

# 验证
nvidia-smi
nvcc --version
```

#### 其他依赖

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    sqlite3 \
    nginx \
    certbot python3-certbot-nginx

# 创建专用用户 (安全最佳实践)
sudo useradd -m -s /bin/bash openmini
sudo mkdir -p /opt/openmini
sudo chown openmini:openmini /opt/openmini
```

---

## 3. 安装部署

### 3.1 从源码编译

```bash
# 切换到专用用户
sudo su - openmini

# 克隆代码
cd /opt
git clone https://github.com/skin1987/OpenMini-V1.git
cd OpenMini-V1

# 编译 Release 版本 (~5-10 分钟)
cargo build --release

# 验证编译产物
ls -lh target/release/openmini-server
# 应该显示: ~15-50MB (取决于编译选项)
```

### 3.2 Docker 部署 (推荐)

#### 单容器部署

**Dockerfile 已包含在项目中**:

```bash
# 构建镜像
docker build -t openmini-server:v1.2.0 .

# 运行容器
docker run -d \
  --name openmini-server \
  --restart unless-stopped \
  -p 50051:50051 \
  -p 8080:8080 \
  -p 9090:9090 \
  -v /data/models:/app/models \
  -v /data/config:/app/config \
  -v /data/logs:/app/logs \
  -e RUST_LOG=info \
  openmini-server:v1.2.0
```

#### Docker Compose 编排

使用项目自带的 `docker-compose.yml`:

```yaml
version: '3.8'

services:
  openmini-server:
    build: .
    container_name: openmini-server
    ports:
      - "50051:50051"   # gRPC
      - "8080:8080"     # REST API
      - "9090:9090"     # Metrics
    volumes:
      - ./models:/app/models:ro
      - ./config:/app/config:ro
      - ./logs:/app/logs
      - ./data:/app/data  # SQLite 数据库持久化
    environment:
      - RUST_LOG=info
      - RUST_BACKTRACE=1
    deploy:
      resources:
        limits:
          memory: 32G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  openmini-admin:
    build: ./openmini-admin
    container_name: openmini-admin
    ports:
      - "3001:3001"
    volumes:
      - ./admin-data:/app/data
    depends_on:
      - openmini-server
    restart: unless-stopped

  # 可选: Prometheus 监控
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'
    restart: unless-stopped

volumes:
  prometheus-data:
```

**启动命令**:

```bash
# 首次启动 (后台运行)
docker compose up -d

# 查看日志
docker compose logs -f openmini-server

# 停止服务
docker compose down
```

### 3.3 二进制部署 (预编译版本)

如果提供预编译的二进制文件:

```bash
# 下载 (示例 URL)
wget https://github.com/skin1987/OpenMini-V1/releases/download/v1.2.0/openmini-server-linux-x86_64.tar.gz

# 解压
tar -xzf openmini-server-linux-x86_64.tar.gz
sudo mv openmini-server /usr/local/bin/
sudo chmod +x /usr/local/bin/openmini-server

# 验证
openmini-server --version
```

### 3.4 Systemd 服务配置

**创建服务文件** `/etc/systemd/system/openmini.service`:

```ini
[Unit]
Description=OpenMini-V1 Inference Server
After=network.target

[Service]
Type=simple
User=openmini
Group=openmini
WorkingDirectory=/opt/OpenMini-V1
ExecStart=/opt/OpenMini-V1/target/release/openmini-server --config /opt/OpenMini-V1/config/server.toml
Restart=on-failure
RestartSec=5

# 资源限制
LimitNOFILE=65535
MemoryMax=32G

# 环境变量
Environment="RUST_LOG=info"
Environment="RUST_BACKTRACE=1"

# 安全加固
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/OpenMini-V1/models
ReadWritePaths=/opt/OpenMini-V1/logs
ReadWritePaths=/opt/OpenMini-V1/data

[Install]
WantedBy=multi-user.target
```

**启用和管理服务**:

```bash
# 重载 systemd 配置
sudo systemctl daemon-reload

# 启用开机自启
sudo systemctl enable openmini

# 启动服务
sudo systemctl start openmini

# 查看状态
sudo systemctl status openmini

# 查看日志
sudo journalctl -u openmini -f

# 停止服务
sudo systemctl stop openmini
```

---

## 4. 配置详解

### 4.1 生产环境 server.toml

```toml
[server]
host = "0.0.0.0"
port = 50051
http_port = 8080
workers = 4                    # 根据 CPU 核心数设置
request_timeout = 120          # 生产环境缩短超时
max_concurrent_requests = 200  # 提高并发能力
graceful_shutdown_timeout = 30 # 优雅关闭等待时间

[model]
name = "llama-2-7b-chat.Q4_0.gguf"
path = "/data/models"
max_context_length = 4096
max_batch_size = 8             # 平衡吞吐量和延迟

[hardware]
backend = "auto"
gpu_memory_fraction = 0.85     # 保留部分显存给系统

[inference]
temperature_default = 0.7
top_p_default = 0.9
repetition_penalty = 1.1
dsa_enabled = true
dsa_sparsity = 0.5

[logging]
level = "info"                # 生产环境不建议 debug
format = "json"               # 结构化日志便于解析
file = "/var/log/openmini/server.log"
# 日志轮转由外部工具处理 (logrotate)

[monitoring]
prometheus_enabled = true
metrics_port = 9090
health_check_path = "/health"

[security]
# JWT 配置 (Admin API)
jwt_secret = "<生成强随机密钥>"
jwt_expiration_hours = 24
cors_allowed_origins = ["https://admin.yourdomain.com"]
rate_limit_requests = 1000    # 每分钟请求数
rate_limit_window_sec = 60
```

**生成 JWT 密钥**:

```bash
openssl rand -base64 64
# 输出类似: aB3xK9... (复制到配置文件)
```

### 4.2 Nginx 反向代理配置

**文件**: `/etc/nginx/sites-available/openmini`:

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    # SSL 证书 (Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # REST API 代理
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 超时设置 (长时间推理请求)
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;

        # 请求体大小限制 (批量请求可能较大)
        client_max_body_size 50m;
    }

    # gRPC 代理 (如需通过 HTTP 暴露)
    location /grpc/ {
        grpc_pass grpc://127.0.0.1:50051;
    }

    # Prometheus 指标 (仅内网访问)
    location /metrics {
        allow 10.0.0.0/8;      # 内网 IP
        allow 172.16.0.0/12;
        deny all;
        proxy_pass http://127.0.0.1:9090/metrics;
    }
}
```

**启用站点并获取证书**:

```bash
sudo ln -s /etc/nginx/sites-available/openmini /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Let's Encrypt 自动证书
sudo certbot --nginx -d api.yourdomain.com
```

### 4.3 日志轮转配置

**文件**: `/etc/logrotate.d/openmini`:

```
/var/log/openmini/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0644 openmini openmini
    postrotate
        systemctl kill -USR1 openmini 2>/dev/null || true
    endscript
}
```

---

## 5. 运维管理

### 5.1 服务健康检查

```bash
# 基础健康检查
curl http://localhost:8080/health
# 预期: {"status":"healthy","version":"1.2.0-beta.1"}

# 详细状态 (Admin API, 需要 token)
curl -H "Authorization: Bearer $TOKEN" http://localhost:3001/admin/service/status

# Prometheus 指标检查
curl http://localhost:9090/metrics | grep openmini_
```

### 5.2 监控告警 (Prometheus + Grafana)

**Prometheus 抓取配置** (`prometheus.yml`):

```yaml
scrape_jobs:
  - job_name: 'openmini'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
```

**关键告警规则** (`alerts.yml`):

```yaml
groups:
  - name: openmini_alerts
    rules:
      # 服务不可用
      - alert: OpenMiniDown
        expr: up{job="openmini"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "OpenMini 服务不可用"

      # 错误率过高
      - alert: HighErrorRate
        expr: |
          sum(rate(openmini_requests_total{status=~"5.."}[5m]))
          /
          sum(rate(openmini_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "API 错误率超过 5%"

      # P99 延迟过高
      - alert: HighLatencyP99
        expr: |
          histogram_quantile(0.99,
            rate(openmini_inference_duration_seconds_bucket[5m])
          ) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P99 推理延迟超过 10 秒"

      # 内存使用率过高
      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes / 1024 / 1024 / 1024 > 28
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "内存使用超过 28GB"
```

### 5.3 备份恢复

#### 数据库备份 (SQLite)

```bash
#!/bin/bash
# backup.sh - 每日备份脚本

BACKUP_DIR="/backups/openmini/$(date +%Y%m%d)"
DB_PATH="/opt/OpenMini-V1/data/admin.db"

mkdir -p "$BACKUP_DIR"

# SQLite 在线备份 (无需停机)
sqlite3 "$DB_PATH" ".backup '$BACKUP_DIR/admin.db.backup'"

# 压缩
gzip "$BACKUP_DIR/admin.db.backup"

# 保留最近 7 天的备份
find /backups/openmini -type f -mtime +7 -delete

echo "$(date): Backup completed to $BACKUP_DIR"
```

**定时任务** (crontab):

```bash
# 每天凌晨 2 点执行备份
0 2 * * * /opt/scripts/backup.sh >> /var/log/backup.log 2>&1
```

#### 配置版本控制

```bash
# 使用 Git 管理配置变更
cd /opt/OpenMini-V1/config
git init
git add server.toml
git commit -m "Initial production config"

# 每次修改后提交
git add -A && git commit -m "Updated batch_size to 16"
```

### 5.4 性能监控仪表板 (Grafana)

推荐导入以下面板 ID 或自行创建:

**核心指标**:
- **请求速率**: `sum(rate(openmini_requests_total[5m])) by (status)`
- **P50/P95/P99 延迟**: `histogram_quantile(0.5/0.95/0.99, rate(...))`
- **活跃连接数**: `openmini_active_connections`
- **KV Cache 使用率**: `kv_cache_blocks_used / kv_cache_blocks_total`
- **内存趋势**: `process_resident_memory_bytes`
- **GPU 利用率** (如有): `nvidia_gpu_utilization_gpu`

---

## 6. 安全加固

### 6.1 认证授权

**JWT 最佳实践**:
- 密钥长度 ≥ 64 字符 (随机生成)
- 过期时间 ≤ 24 小时
- 生产环境禁用测试账号 (`admin/admin123`)
- 定期轮换密钥 (每 90 天)

**RBAC 权限最小化原则**:
- 仅给操作员必要的权限
- API Key 设置合理的配额和过期时间
- 审计所有管理员操作

### 6.2 网络安全

**防火墙规则** (UFW):

```bash
# 默认拒绝入站
sudo ufw default deny incoming
sudo ufw default allow outgoing

# 允许 SSH
sudo ufw allow 22/tcp

# 允许 HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# 允许内部 gRPC (仅受信任 IP)
sudo ufw allow from 10.0.0.0/8 to any port 50051 proto tcp

# 启用防火墙
sudo ufw enable
```

**TLS 配置**:
- 强制 HTTPS (HTTP → 301 重定向)
- 使用 TLS 1.2+ (禁用 SSLv3, TLS 1.0/1.1)
- 定期更新证书 (Let's Encrypt 自动续期)
- 启用 HSTS 头部

### 6.3 审计日志

**启用详细审计**:

```toml
[audit]
enabled = true
log_all_requests = true       # 记录所有 API 请求
log_auth_events = true        # 登录/登出事件
log_config_changes = true     # 配置修改
retention_days = 90           # 保留 90 天
```

**定期审查**:
- 检查异常登录尝试
- 监控权限提升操作
- 分析失败请求模式 (可能的攻击)

---

## 7. 升级维护

### 7.1 滚动升级策略 (零停机)

**步骤**:

1. **准备新版本**
   ```bash
   git fetch origin
   git checkout v1.2.0-stable
   cargo build --release
   ```

2. **灰度发布** (多实例时)
   ```bash
   # 逐个替换实例
   for i in {1..3}; do
       docker compose -f docker-compose.yml up -d openmini-server-$i
       sleep 60  # 等待健康检查通过
       # 验证流量切换
       curl http://localhost:8080/health
   done
   ```

3. **验证**
   ```bash
   # 运行回归测试
   ./scripts/run_regression_tests.sh --quick
   
   # 检查错误日志
   tail -100 /var/log/openmini/server.log | grep ERROR
   ```

4. **回滚** (如发现问题)
   ```bash
   git checkout v1.1.0  # 回退到上一稳定版
   cargo build --release
   sudo systemctl restart openmini
   ```

### 7.2 数据迁移

**版本兼容性矩阵**:

| 从版本 | 到 v1.2.0 | 是否需要迁移 |
|--------|-----------|-------------|
| v1.1.0 | ✅ 是 | 自动 (数据库 schema 兼容) |
| v1.0.x | ⚠️ 可能 | 手动执行 migration |
| v0.x | ❌ 不支持 | 需要全新安装 |

**迁移前检查清单**:
- [ ] 备份数据库 (`admin.db`)
- [ ] 备份配置文件 (`server.toml`)
- [ ] 记录当前模型加载状态
- [ ] 通知用户计划维护窗口
- [ ] 准备回滚方案

### 7.3 性能基线对比

每次升级后运行基准测试，建立性能档案:

```bash
# 运行基准测试
cargo bench --package openmini-server

# 保存结果
cp target/criterion/report/ ./benchmarks/v1.2.0/

# 对比差异
# (使用 criterion 的 HTML 报告可视化)
```

**关注指标**:
- 平均延迟变化 < ±10%
- P99 延迟不退化
- 内存使用无异常增长
- 无新增错误类型

---

## 8. 高可用方案 (可选)

### 8.1 多实例负载均衡

**架构**:

```
Client → Nginx (LB) → Server-1 (:8080)
                      → Server-2 (:8081)
                      → Server-3 (:8082)
```

**Nginx upstream 配置**:

```nginx
upstream openmini_backend {
    least_conn;                    # 最少连接算法
    
    server 10.0.0.1:8080 weight=1 max_fails=3 fail_timeout=30s;
    server 10.0.0.2:8080 weight=1 max_fails=3 fail_timeout=30s;
    server 10.0.0.3:8080 weight=1 max_fails=3 fail_timeout=30s;
    
    keepalive 32;                  # 长连接池
}
```

**注意**:
- 每个 Server 实例独立加载模型 (显存 x N)
- 共享存储 (NFS/GlusterFS) 用于模型文件
- Session affinity 对话类应用必需

### 8.2 Kubernetes 部署 (高级)

**Deployment 示例**:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openmini-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: openmini-server
  template:
    metadata:
      labels:
        app: openmini-server
    spec:
      containers:
      - name: server
        image: openmini-server:v1.2.0
        ports:
        - containerPort: 8080
        - containerPort: 50051
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
          limits:
            memory: "32Gi"
            cpu: "16"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: models
          mountPath: /app/models
          readOnly: true
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: openmini-service
spec:
  selector:
    app: openmini-server
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  - name: grpc
    port: 50051
    targetPort: 50051
  type: LoadBalancer
```

---

## 📞 故障应急联系

在部署完成后，请保存以下信息:

| 项目 | 内容 |
|------|------|
| **服务地址** | https://api.yourdomain.com |
| **Admin 面板** | https://admin.yourdomain.com |
| **监控地址** | http://monitor.yourdomain.com:3000 |
| **告警通知** | PagerDuty / Slack / Email |
| **技术支持** | support@yourcompany.com |
| **紧急联系人** | DevOps on-call: +86-xxx-xxxx-xxxx |

---

## ✅ 部署检查清单

部署完成后，逐项验证:

- [ ] 服务启动成功 (`systemctl status openmini`)
- [ ] 健康检查正常 (`curl /health`)
- [ ] HTTPS 证书有效 (浏览器访问无警告)
- [ ] Admin 面板可登录 (非默认密码)
- [ ] 模型加载成功 (日志中无错误)
- [ ] Prometheus 指标可采集
- [ ] 告警规则已配置
- [ ] 日志轮转正常工作
- [ ] 备份脚本已验证
- [ ] 防火墙规则已生效
- [ ] 性能基准已记录

---

*文档版本: 1.0 | 最后更新: 2026-04-09*
