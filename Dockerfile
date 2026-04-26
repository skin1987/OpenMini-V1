# Multi-stage Dockerfile for OpenMini-V1
# Supports: CPU-only, CUDA, Vulkan builds

# ============================================
# Stage 0: System dependencies (cached layer)
# ============================================
FROM rust:1.75-slim AS base
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Stage 1: CPU-only builder (default)
# ============================================
FROM base AS builder-cpu
COPY Cargo.toml Cargo.lock ./
COPY openmini-server/Cargo.toml ./openmini-server/
COPY openmini-proto/Cargo.toml ./openmini-proto/

RUN mkdir -p openmini-server/src openmini-proto/src && \
    touch openmini-server/src/lib.rs openmini-proto/src/lib.rs && \
    cargo build --release || true && \
    find target/release -name "*.d" -delete && \
    rm -rf openmini-server/src openmini-proto/src

COPY . .
ARG BUILD_FEATURES=""
RUN if [ -z "$BUILD_FEATURES" ]; then \
        cargo build --release; \
    else \
        cargo build --release --features "$BUILD_FEATURES"; \
    fi

# ============================================
# Stage 2: CUDA builder (NVIDIA GPU support)
# ============================================
FROM nvidia/cuda:12.4-devel-ubuntu22.04 AS builder-cuda
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    curl \
    git \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    . /root/.cargo/env && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

COPY Cargo.toml Cargo.lock ./
COPY openmini-server/Cargo.toml ./openmini-server/
COPY openmini-proto/Cargo.toml ./openmini-proto/

RUN mkdir -p openmini-server/src openmini-proto/src && \
    touch openmini-server/src/lib.rs openmini-proto/src/lib.rs && \
    cargo build --release --features cuda || true && \
    find target/release -name "*.d" -delete && \
    rm -rf openmini-server/src openmini-proto/src

COPY . .
ARG BUILD_FEATURES="cuda"
RUN cargo build --release --features "$BUILD_FEATURES"

# ============================================
# Stage 3: Vulkan builder (cross-platform GPU)
# ============================================
FROM base AS builder-vulkan
RUN apt-get update && apt-get install -y --no-install-recommends \
    libvulkan-dev \
    vulkan-tools \
    mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY openmini-server/Cargo.toml ./openmini-server/
COPY openmini-proto/Cargo.toml ./openmini-proto/

RUN mkdir -p openmini-server/src openmini-proto/src && \
    touch openmini-server/src/lib.rs openmini-proto/src/lib.rs && \
    cargo build --release --features vulkan || true && \
    find target/release -name "*.d" -delete && \
    rm -rf openmini-server/src openmini-proto/src

COPY . .
ARG BUILD_FEATURES="vulkan"
RUN cargo build --release --features "$BUILD_FEATURES"

# ============================================
# Stage 4: Runtime image (minimal)
# ============================================
FROM debian:bookworm-slim AS runtime
LABEL maintainer="OpenMini Team" \
      version="1.2.0-beta.1" \
      description="OpenMini-V1 AI Inference Server"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/* && \
    useradd -r -s /bin/false openmini && \
    mkdir -p /data/models /data/logs /data/cache /etc/openmini && \
    chown -R openmini:openmini /app /data /etc/openmini

WORKDIR /app

# Copy binary from CPU builder stage (default)
# To build with CUDA:   docker build --target runtime-cuda .
# To build with Vulkan: docker build --target runtime-vulkan .
COPY --from=builder-cpu /app/target/release/openmini-server /usr/local/bin/

# Copy default config
COPY config/server.toml.example /etc/openmini/server.toml.example

USER openmini

EXPOSE 50051 8080

# Enhanced health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENV RUST_LOG=info \
    OPENMINI_CONFIG_PATH=/etc/openmini/server.toml \
    OPENMINI_MODEL_PATH=/data/models \
    HARDWARE_BACKEND=auto

ENTRYPOINT ["openmini-server"]
CMD ["--config", "/etc/openmini/server.toml"]

# ============================================
# Stage 5: CUDA Runtime (with NVIDIA drivers)
# ============================================
FROM runtime AS runtime-cuda
FROM nvidia/cuda:12.4-runtime-ubuntu22.04 AS runtime-cuda-real

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/* && \
    useradd -r -s /bin/false openmini && \
    mkdir -p /data/models /data/logs /data/cache /etc/openmini && \
    chown -R openmini:openmini /app /data /etc/openmini

WORKDIR /app

COPY --from=builder-cuda /app/target/release/openmini-server /usr/local/bin/
COPY config/server.toml.example /etc/openmini/server.toml.example

USER openmini

EXPOSE 50051 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENV RUST_LOG=info \
    OPENMINI_CONFIG_PATH=/etc/openmini/server.toml \
    OPENMINI_MODEL_PATH=/data/models \
    HARDWARE_BACKEND=cuda

ENTRYPOINT ["openmini-server"]
CMD ["--config", "/etc/openmini/server.toml"]

# ============================================
# Stage 6: Vulkan Runtime (with Vulkan drivers)
# ============================================
FROM runtime AS runtime-vulkan
RUN apt-get update && apt-get install -y --no-install-recommends \
    libvulkan1 \
    mesa-vulkan-drivers \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*
