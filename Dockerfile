# Transcode CLI Docker Image
# Multi-stage build for minimal final image size
#
# Usage:
#   docker build -t transcode .
#   docker run -v $(pwd):/data transcode -i /data/input.mp4 -o /data/output.mp4

# =============================================================================
# Stage 1: Builder - Compile with release optimizations and LTO
# =============================================================================
FROM rust:1.92-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    nasm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy workspace manifests first for better layer caching
COPY Cargo.toml Cargo.lock ./

# Copy all workspace member Cargo.toml files
COPY transcode-core/Cargo.toml transcode-core/Cargo.toml
COPY transcode-codecs/Cargo.toml transcode-codecs/Cargo.toml
COPY transcode-containers/Cargo.toml transcode-containers/Cargo.toml
COPY transcode-pipeline/Cargo.toml transcode-pipeline/Cargo.toml
COPY transcode/Cargo.toml transcode/Cargo.toml
COPY transcode-cli/Cargo.toml transcode-cli/Cargo.toml
COPY transcode-av1/Cargo.toml transcode-av1/Cargo.toml
COPY transcode-streaming/Cargo.toml transcode-streaming/Cargo.toml
COPY transcode-gpu/Cargo.toml transcode-gpu/Cargo.toml
COPY transcode-ai/Cargo.toml transcode-ai/Cargo.toml
COPY transcode-quality/Cargo.toml transcode-quality/Cargo.toml
COPY transcode-distributed/Cargo.toml transcode-distributed/Cargo.toml
COPY transcode-intel/Cargo.toml transcode-intel/Cargo.toml
COPY transcode-hwaccel/Cargo.toml transcode-hwaccel/Cargo.toml
COPY transcode-pertitle/Cargo.toml transcode-pertitle/Cargo.toml
COPY transcode-neural/Cargo.toml transcode-neural/Cargo.toml
COPY transcode-live/Cargo.toml transcode-live/Cargo.toml
COPY transcode-hdr/Cargo.toml transcode-hdr/Cargo.toml
COPY transcode-audio-ai/Cargo.toml transcode-audio-ai/Cargo.toml
COPY transcode-caption/Cargo.toml transcode-caption/Cargo.toml
COPY transcode-watermark/Cargo.toml transcode-watermark/Cargo.toml
COPY transcode-cloud/Cargo.toml transcode-cloud/Cargo.toml
COPY transcode-analytics/Cargo.toml transcode-analytics/Cargo.toml
COPY transcode-zerocopy/Cargo.toml transcode-zerocopy/Cargo.toml
COPY transcode-hevc/Cargo.toml transcode-hevc/Cargo.toml
COPY transcode-vp9/Cargo.toml transcode-vp9/Cargo.toml
COPY transcode-opus/Cargo.toml transcode-opus/Cargo.toml
COPY transcode-mkv/Cargo.toml transcode-mkv/Cargo.toml
COPY transcode-ts/Cargo.toml transcode-ts/Cargo.toml
COPY transcode-deinterlace/Cargo.toml transcode-deinterlace/Cargo.toml
COPY transcode-loudness/Cargo.toml transcode-loudness/Cargo.toml
COPY transcode-timecode/Cargo.toml transcode-timecode/Cargo.toml
COPY transcode-framerate/Cargo.toml transcode-framerate/Cargo.toml
COPY transcode-drm/Cargo.toml transcode-drm/Cargo.toml
COPY transcode-telemetry/Cargo.toml transcode-telemetry/Cargo.toml
COPY transcode-compat/Cargo.toml transcode-compat/Cargo.toml
COPY transcode-bench/Cargo.toml transcode-bench/Cargo.toml

# Create dummy source files for dependency caching
RUN mkdir -p transcode-core/src && echo "pub fn dummy() {}" > transcode-core/src/lib.rs && \
    mkdir -p transcode-codecs/src && echo "pub fn dummy() {}" > transcode-codecs/src/lib.rs && \
    mkdir -p transcode-containers/src && echo "pub fn dummy() {}" > transcode-containers/src/lib.rs && \
    mkdir -p transcode-pipeline/src && echo "pub fn dummy() {}" > transcode-pipeline/src/lib.rs && \
    mkdir -p transcode/src && echo "pub fn dummy() {}" > transcode/src/lib.rs && \
    mkdir -p transcode-cli/src && echo "fn main() {}" > transcode-cli/src/main.rs && \
    mkdir -p transcode-av1/src && echo "pub fn dummy() {}" > transcode-av1/src/lib.rs && \
    mkdir -p transcode-streaming/src && echo "pub fn dummy() {}" > transcode-streaming/src/lib.rs && \
    mkdir -p transcode-gpu/src && echo "pub fn dummy() {}" > transcode-gpu/src/lib.rs && \
    mkdir -p transcode-ai/src && echo "pub fn dummy() {}" > transcode-ai/src/lib.rs && \
    mkdir -p transcode-quality/src && echo "pub fn dummy() {}" > transcode-quality/src/lib.rs && \
    mkdir -p transcode-distributed/src && echo "pub fn dummy() {}" > transcode-distributed/src/lib.rs && \
    mkdir -p transcode-intel/src && echo "pub fn dummy() {}" > transcode-intel/src/lib.rs && \
    mkdir -p transcode-hwaccel/src && echo "pub fn dummy() {}" > transcode-hwaccel/src/lib.rs && \
    mkdir -p transcode-pertitle/src && echo "pub fn dummy() {}" > transcode-pertitle/src/lib.rs && \
    mkdir -p transcode-neural/src && echo "pub fn dummy() {}" > transcode-neural/src/lib.rs && \
    mkdir -p transcode-live/src && echo "pub fn dummy() {}" > transcode-live/src/lib.rs && \
    mkdir -p transcode-hdr/src && echo "pub fn dummy() {}" > transcode-hdr/src/lib.rs && \
    mkdir -p transcode-audio-ai/src && echo "pub fn dummy() {}" > transcode-audio-ai/src/lib.rs && \
    mkdir -p transcode-caption/src && echo "pub fn dummy() {}" > transcode-caption/src/lib.rs && \
    mkdir -p transcode-watermark/src && echo "pub fn dummy() {}" > transcode-watermark/src/lib.rs && \
    mkdir -p transcode-cloud/src && echo "pub fn dummy() {}" > transcode-cloud/src/lib.rs && \
    mkdir -p transcode-analytics/src && echo "pub fn dummy() {}" > transcode-analytics/src/lib.rs && \
    mkdir -p transcode-zerocopy/src && echo "pub fn dummy() {}" > transcode-zerocopy/src/lib.rs && \
    mkdir -p transcode-hevc/src && echo "pub fn dummy() {}" > transcode-hevc/src/lib.rs && \
    mkdir -p transcode-vp9/src && echo "pub fn dummy() {}" > transcode-vp9/src/lib.rs && \
    mkdir -p transcode-opus/src && echo "pub fn dummy() {}" > transcode-opus/src/lib.rs && \
    mkdir -p transcode-mkv/src && echo "pub fn dummy() {}" > transcode-mkv/src/lib.rs && \
    mkdir -p transcode-ts/src && echo "pub fn dummy() {}" > transcode-ts/src/lib.rs && \
    mkdir -p transcode-deinterlace/src && echo "pub fn dummy() {}" > transcode-deinterlace/src/lib.rs && \
    mkdir -p transcode-loudness/src && echo "pub fn dummy() {}" > transcode-loudness/src/lib.rs && \
    mkdir -p transcode-timecode/src && echo "pub fn dummy() {}" > transcode-timecode/src/lib.rs && \
    mkdir -p transcode-framerate/src && echo "pub fn dummy() {}" > transcode-framerate/src/lib.rs && \
    mkdir -p transcode-drm/src && echo "pub fn dummy() {}" > transcode-drm/src/lib.rs && \
    mkdir -p transcode-telemetry/src && echo "pub fn dummy() {}" > transcode-telemetry/src/lib.rs && \
    mkdir -p transcode-compat/src && echo "pub fn dummy() {}" > transcode-compat/src/lib.rs && \
    mkdir -p transcode-bench/src && echo "pub fn dummy() {}" > transcode-bench/src/lib.rs

# Build dependencies only (will be cached)
# Note: This may fail due to missing actual source, but dependencies will be cached
ENV CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse
RUN cargo build --release --package transcode-cli 2>/dev/null || true

# Copy actual source code
COPY transcode-core/ transcode-core/
COPY transcode-codecs/ transcode-codecs/
COPY transcode-containers/ transcode-containers/
COPY transcode-pipeline/ transcode-pipeline/
COPY transcode/ transcode/
COPY transcode-cli/ transcode-cli/
COPY transcode-av1/ transcode-av1/
COPY transcode-streaming/ transcode-streaming/
COPY transcode-gpu/ transcode-gpu/
COPY transcode-ai/ transcode-ai/
COPY transcode-quality/ transcode-quality/
COPY transcode-distributed/ transcode-distributed/
COPY transcode-intel/ transcode-intel/
COPY transcode-hwaccel/ transcode-hwaccel/
COPY transcode-pertitle/ transcode-pertitle/
COPY transcode-neural/ transcode-neural/
COPY transcode-live/ transcode-live/
COPY transcode-hdr/ transcode-hdr/
COPY transcode-audio-ai/ transcode-audio-ai/
COPY transcode-caption/ transcode-caption/
COPY transcode-watermark/ transcode-watermark/
COPY transcode-cloud/ transcode-cloud/
COPY transcode-analytics/ transcode-analytics/
COPY transcode-zerocopy/ transcode-zerocopy/
COPY transcode-hevc/ transcode-hevc/
COPY transcode-vp9/ transcode-vp9/
COPY transcode-opus/ transcode-opus/
COPY transcode-mkv/ transcode-mkv/
COPY transcode-ts/ transcode-ts/
COPY transcode-deinterlace/ transcode-deinterlace/
COPY transcode-loudness/ transcode-loudness/
COPY transcode-timecode/ transcode-timecode/
COPY transcode-framerate/ transcode-framerate/
COPY transcode-drm/ transcode-drm/
COPY transcode-telemetry/ transcode-telemetry/
COPY transcode-compat/ transcode-compat/
COPY transcode-bench/ transcode-bench/

# Build the CLI binary with release optimizations
# The workspace Cargo.toml has LTO enabled in [profile.release]:
#   lto = true, codegen-units = 1, opt-level = 3
RUN cargo build --release --package transcode-cli --locked && \
    strip /build/target/release/transcode

# =============================================================================
# Stage 2: Runtime - Minimal Debian slim image
# =============================================================================
FROM debian:bookworm-slim AS runtime

# Install runtime dependencies and tini for proper signal handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 -s /bin/sh transcode

# Copy the built binary from builder stage
COPY --from=builder /build/target/release/transcode /usr/local/bin/transcode

# Ensure binary is executable
RUN chmod +x /usr/local/bin/transcode

# Create data directory for volume mounting
RUN mkdir -p /data && chown transcode:transcode /data

# Switch to non-root user for security
USER transcode
WORKDIR /data

# Use tini as init for proper signal handling and zombie process reaping
# This ensures graceful shutdown when receiving SIGTERM/SIGINT
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/transcode"]

# Default to showing help when no arguments provided
CMD ["--help"]

# =============================================================================
# Stage 3: Development - Full Rust toolchain for development
# =============================================================================
FROM rust:1.92-bookworm AS development

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    nasm \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install useful Rust development tools
RUN rustup component add rustfmt clippy && \
    cargo install cargo-watch cargo-audit cargo-outdated

WORKDIR /app

# Mount source code as volume
VOLUME ["/app"]

CMD ["cargo", "watch", "-x", "check", "-x", "test"]

# =============================================================================
# Image Metadata
# =============================================================================
LABEL org.opencontainers.image.title="Transcode CLI"
LABEL org.opencontainers.image.description="Memory-safe, high-performance media transcoding tool"
LABEL org.opencontainers.image.vendor="Transcode Contributors"
LABEL org.opencontainers.image.licenses="MIT OR Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/transcode/transcode"
LABEL org.opencontainers.image.documentation="https://github.com/transcode/transcode#docker-usage"
