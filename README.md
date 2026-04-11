# <img src="https://img.shields.io/badge/version-v1.2.0--beta.1-orange" alt="version"> <img src="https://img.shields.io/badge/Rust-1.75+-blue" alt="Rust"> <img src="https://img.shields.io/badge/license-MIT-green" alt="license"> <img src="https://img.shields.io/badge/status-beta-yellow" alt="status"> **OpenMini-V1**

> High-Performance LLM Inference Server — Built in Rust, Powered by Candle

[Release](https://github.com/skin1987/OpenMini-V1/releases/tag/v1.2.0-beta.1) | [CHANGELOG](./CHANGELOG.md) | [RELEASE_NOTES](./RELEASE_NOTES.md) | [Docker Guide](#docker)

---

## Overview

OpenMini-V1 is a production-grade LLM inference server built entirely in Rust, leveraging the [Candle](https://github.com/huggingface/candle) ML framework for high-performance model execution. It supports multi-hardware acceleration (CPU / CUDA / Metal), advanced attention optimizations (DSA, Flash Attention 3), and includes a complete reinforcement learning training pipeline.

### Quality Metrics (v1.2.0-beta.1)

| Metric | Result |
|--------|--------|
| DSA Integration Tests | 3/3 passed |
| RL Module Tests | 105/105 passed |
| Metal GPU Tests | 23/23 passed |
| Clippy Errors | 0 errors |
| Release Build | Success (5m31s) |

---

## Features

- Multi-Hardware Acceleration: CPU (AVX/NEON/SIMD), CUDA, Metal (Apple Silicon), Vulkan (experimental)
- Advanced Attention: Dynamic Sparse Attention (DSA), Flash Attention 3, Paged KV Cache
- Quantization: GGUF loader, INT8/INT4 quantization, SIMD-accelerated inference
- Reinforcement Learning: GRPO algorithm, Actor-Reward architecture, Keep Routing & Sampling Mask
- Service Layer: gRPC gateway (tonic), HTTP REST API (axum), Worker process pool
- Monitoring: Prometheus metrics exporter, health check endpoints, structured logging
- Admin Panel: Vue3 + TypeScript management dashboard (`openmini-admin-web/`)
- Database Layer: MemoryStore, SessionManager, MessagePool abstraction
- CI/CD: GitHub Actions pipeline + local `act` runner support
- Docker: Multi-stage build with docker-compose one-click deployment

---

## Architecture

```
OpenMini-V1/
├── openmini-server/          Core inference engine (Rust)
│   ├── src/
│   │   ├── hardware/         CPU/GPU/Metal/CUDA backends
│   │   ├── model/inference/  DSA, FA3, GGUF, sampler, tokenizer
│   │   ├── rl/               GRPO, Actor, Reward, Keep Routing
│   │   ├── service/          gRPC, HTTP, Worker pool, Gateway
│   │   ├── db/               MemoryStore, SessionManager
│   │   └── monitoring/       Metrics, Health Check, Prometheus
│   └── tests/                Integration & benchmark tests
├── openmini-proto/           Protocol Buffers definitions
├── openmini-admin/           Admin API service (Rust/Tonic)
├── openmini-admin-web/       Vue3 admin dashboard
├── openmini-client/          Python client SDK
├── config/                   Server configuration
├── .github/workflows/        CI/CD pipelines
└── Dockerfile                Multi-stage container build
```

**Multi-process Architecture**: Main process manages Workers via process pool; each Worker runs independent model inference.

---

## Quick Start

### Prerequisites

- **Rust** 1.75+ ([install](https://rustup.rs))
- **Platform**: macOS (Apple Silicon) or Linux (x86_64)
- **Memory**: 8GB+ RAM (16GB recommended for 7B models)

### Build & Run

```bash
# Clone repository
git clone https://github.com/skin1987/OpenMini-V1.git && cd OpenMini-V1

# Build release binary (~5-6 minutes)
cargo build --release

# Configure server
cp config/server.toml.example config/server.toml

# Start server
./target/release/openmini-server --config config/server.toml
```

### Verify Installation

```bash
curl http://localhost:8080/health
# Expected: {"status":"healthy","version":"1.2.0-beta.1"}
```

### Usage Examples

```bash
# REST API - Generate text
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!", "max_tokens": 50, "temperature": 0.7}'

# View metrics (Prometheus format)
curl http://localhost:8080/metrics
```

See [examples/rust_client.rs](./examples/rust_client.rs) for a full gRPC client example.

---

## Configuration

Key options in `config/server.toml`:

```toml
[server]
host = "0.0.0.0"
port = 50051              # gRPC port
http_port = 8080          # HTTP REST port

[model]
name = "your-model.gguf"
path = "/path/to/models"

[hardware]
backend = "auto"          # auto | cpu | cuda | metal
```

---

## Docker

### Quick Deploy

```bash
# Build image
docker build -t openmini-server .

# Or use docker-compose (includes optional monitoring stack)
docker compose up -d
```

The image exposes:
- Port `50051` — gRPC API
- Port `8080` — HTTP REST API
- Port `9090` — Prometheus metrics (optional)

---

## Testing

```bash
# All tests
cargo test --workspace

# Specific test suites
cargo test --package openmini-server --test dsa_integration_test    # DSA: 3 tests
cargo test --package openmini-server --lib rl::                     # RL: 105 tests
cargo test --package openmini-server --lib hardware::gpu::metal::tests # Metal: 23 tests

# Clippy lint check
cargo clippy --package openmini-server --lib                        # 0 errors
```

---

## Roadmap

- [ ] v1.2.0-stable: Resolve remaining clippy warnings, fix quant_simd SIGSEGV
- [ ] Vulkan backend completion
- [ ] Distributed inference support
- [ ] Model parallelism (tensor/pipeline)
- [ ] Streaming output enhancements
- [ ] Full documentation site

See [CHANGELOG.md](./CHANGELOG.md) for complete version history.

---

## Contributing

We welcome contributions! Please see our templates:

- [Bug Report](/.github/ISSUE_TEMPLATE/bug_report.yml)
- [Feature Request](/.github/ISSUE_TEMPLATE/feature_request.yml)
- [PR Template](/.github/PULL_REQUEST_TEMPLATE.md)

**Development workflow**:

1. Fork and clone
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes with tests
4. Run `cargo clippy && cargo test`
5. Submit PR with description

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Candle](https://github.com/huggingface/candle) — ML inference framework
- [Tonic](https://github.com/hyperium/tonic) — gRPC framework
- [Axum](https://github.com/tokio-rs/axum) — Web framework
- The Rust community for outstanding tooling and ecosystem

---

<p align="center">
  <strong>OpenMini-V1</strong> — Built with speed, safety, and simplicity.
</p>
