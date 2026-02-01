# Rusteeze Docker Build
# Multi-stage build for minimal production image

# ============================================================================
# Stage 1: Build Environment
# ============================================================================
FROM rust:1.75-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create build directory
WORKDIR /build

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/

# Build release binary
RUN cargo build --release --bin rusteeze \
    && strip target/release/rusteeze

# ============================================================================
# Stage 2: Runtime Environment  
# ============================================================================
FROM debian:bookworm-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 rusteeze

# Create directories
RUN mkdir -p /app /data /models && \
    chown -R rusteeze:rusteeze /app /data /models

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/target/release/rusteeze /app/rusteeze

# Copy default configuration
COPY config.example.yaml /app/config.yaml

# Set ownership
RUN chown -R rusteeze:rusteeze /app

# Switch to non-root user
USER rusteeze

# Environment variables
ENV RUSTEEZE_HOST=0.0.0.0 \
    RUSTEEZE_PORT=8000 \
    LOG_LEVEL=info \
    HF_HOME=/data/huggingface

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
ENTRYPOINT ["/app/rusteeze"]
CMD ["serve"]

# ============================================================================
# Stage 3: CUDA Runtime (Optional GPU Support)
# ============================================================================
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS cuda-runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 rusteeze

# Create directories
RUN mkdir -p /app /data /models && \
    chown -R rusteeze:rusteeze /app /data /models

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/target/release/rusteeze /app/rusteeze

# Copy default configuration
COPY config.example.yaml /app/config.yaml

# Set ownership
RUN chown -R rusteeze:rusteeze /app

# Switch to non-root user
USER rusteeze

# Environment variables
ENV RUSTEEZE_HOST=0.0.0.0 \
    RUSTEEZE_PORT=8000 \
    LOG_LEVEL=info \
    HF_HOME=/data/huggingface \
    CUDA_VISIBLE_DEVICES=all

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
ENTRYPOINT ["/app/rusteeze"]
CMD ["serve"]
