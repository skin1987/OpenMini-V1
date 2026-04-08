# Multi-stage Dockerfile for OpenMini-V1
# Optimized for production deployment with minimal image size

# ============================================================
# Stage 1: Builder - Compile dependencies and application
# ============================================================
FROM rust:1.75-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Copy Cargo files first (layer caching)
COPY Cargo.toml Cargo.lock ./
COPY openmini-server/Cargo.toml ./openmini-server/
COPY openmini-proto/Cargo.toml ./openmini-proto/

# Create dummy source files for dependency caching
RUN mkdir -p openmini-server/src openmini-proto/src && \
    touch openmini-server/src/lib.rs openmini-proto/src/lib.rs && \
    cargo build --release --manifest-path Cargo.toml || true && \
    find target/release -name "*.d" -delete && \
    rm -rf openmini-server/src openmini-proto/src

# Copy actual source code
COPY . .

# Build the application (with optional features)
ARG BUILD_FEATURES=""
RUN if [ -z "$BUILD_FEATURES" ]; then \
        cargo build --release; \
    else \
        cargo build --release --features "$BUILD_FEATURES"; \
    fi

# ============================================================
# Stage 2: Runtime - Minimal image for execution
# ============================================================
FROM debian:bookworm-slim AS runtime

LABEL maintainer="OpenMini Team <team@openmini.ai>"
LABEL version="1.2.0-beta.1"
LABEL description="High-performance LLM inference server"
LABEL org.opencontainers.image.source="https://github.com/skin1987/OpenMini-V1"

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -r -s /bin/false openmini

WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/target/release/openmini-server /usr/local/bin/

# Copy configuration files
COPY config/server.toml.example /etc/openmini/server.toml.example
COPY config/logging.yaml.example /etc/openmini/logging.yaml.example

# Create data directories
RUN mkdir -p /data/models /data/logs /data/cache && \
    chown -R openmini:openmini /app /data /etc/openmini

USER openmini

# Expose ports
EXPOSE 50051 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Environment variables
ENV RUST_LOG=info \
    OPENMINI_CONFIG_PATH=/etc/openmini/server.toml \
    OPENMINI_MODEL_PATH=/data/models \
    OPENMINI_LOG_DIR=/data/logs \
    OPENMINI_CACHE_DIR=/data/cache

# Default command
ENTRYPOINT ["openmini-server"]
CMD ["--config", "/etc/openmini/server.toml"]

# ============================================================
# Stage 3: Development (optional, for debugging)
# ============================================================
FROM builder AS development

# Install debug tools
RUN apt-get update && apt-get install -y \
    gdb \
    valgrind \
    strace \
    && rm -rf /var/lib/apt/lists/*

# Keep debug symbols
ENV RUSTFLAGS="-g"

CMD ["cargo", "run", "--package", "openmini-server", "--", "--config", "config/server.toml"]

# ============================================================
# Stage 4: Test runner (for CI/CD)
# ============================================================
FROM builder AS test

# Run full test suite as validation step
RUN cargo test --workspace --lib && \
    cargo test --package openmini-server --test dsa_integration_test && \
    cargo test --package openmini-server --lib rl:: && \
    echo "✅ All tests passed!"

# ============================================================
# Build Arguments (for customization)
# ============================================================
# Usage:
#   docker build --build-arg TARGETPLATFORM=linux/amd64 .
#   docker build --build-arg BUILD_FEATURES=cuda,metal .

ARG TARGETPLATFORM=linux/amd64
ARG BUILD_FEATURES=""
