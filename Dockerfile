FROM rust:1.75-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential pkg-config libssl-dev protobuf-compiler && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
COPY openmini-server/Cargo.toml ./openmini-server/
COPY openmini-proto/Cargo.toml ./openmini-proto/
RUN mkdir -p openmini-server/src openmini-proto/src && touch openmini-server/src/lib.rs openmini-proto/src/lib.rs && cargo build --release --manifest-path Cargo.toml || true && find target/release -name "*.d" -delete && rm -rf openmini-server/src openmini-proto/src
COPY . .
ARG BUILD_FEATURES=""
RUN if [ -z "$BUILD_FEATURES" ]; then cargo build --release; else cargo build --release --features "$BUILD_FEATURES"; fi

FROM debian:bookworm-slim AS runtime
LABEL maintainer="OpenMini Team <team@openmini.ai>" version="1.2.0-beta.1"
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libssl3 && rm -rf /var/lib/apt/lists/* && useradd -r -s /bin/false openmini
WORKDIR /app
COPY --from=builder /app/target/release/openmini-server /usr/local/bin/
COPY config/server.toml.example /etc/openmini/server.toml.example
RUN mkdir -p /data/models /data/logs /data/cache && chown -R openmini:openmini /app /data /etc/openmini
USER openmini
EXPOSE 50051 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 CMD curl -f http://localhost:8080/health || exit 1
ENV RUST_LOG=info OPENMINI_CONFIG_PATH=/etc/openmini/server.toml OPENMINI_MODEL_PATH=/data/models OPENMINI_LOG_DIR=/data/logs OPENMINI_CACHE_DIR=/data/cache
ENTRYPOINT ["openmini-server"]
CMD ["--config", "/etc/openmini/server.toml"]