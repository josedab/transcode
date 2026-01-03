# Transcode Architecture

This document describes the high-level architecture of the Transcode codec library.

## Overview

Transcode is a modular, memory-safe video/audio transcoding library written in Rust. It's organized as a Cargo workspace with specialized crates for different concerns.

## Crate Dependency Diagram

```
                              ┌──────────────┐
                              │  transcode   │  (high-level API)
                              └──────┬───────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           │                         │                         │
           ▼                         ▼                         ▼
   ┌───────────────┐       ┌─────────────────┐       ┌─────────────────┐
   │transcode-codecs│      │transcode-containers│    │transcode-pipeline│
   └───────┬───────┘       └────────┬────────┘       └────────┬────────┘
           │                        │                         │
           └────────────────────────┼─────────────────────────┘
                                    │
                                    ▼
                           ┌────────────────┐
                           │ transcode-core │  (shared types)
                           └────────────────┘
```

## Core Crates

### transcode-core
Foundation crate providing shared types:
- `Frame` - Video frame with pixel data
- `Packet` - Encoded media packet
- `Timestamp`, `Duration`, `TimeBase` - Time handling
- `BitReader`, `BitWriter` - Bitstream I/O
- Error types and result aliases

### transcode-codecs
Video and audio codec implementations:
- **Video**: H.264/AVC decoder/encoder
- **Audio**: AAC decoder/encoder, MP3 decoder
- SIMD-optimized DSP operations
- Codec traits (`VideoEncoder`, `VideoDecoder`, `AudioEncoder`, `AudioDecoder`)

### transcode-containers
Container format support:
- **MP4/MOV**: Full demuxing and muxing
- Atom/box parsing and writing
- Stream metadata extraction

### transcode-pipeline
Transcoding orchestration:
- Pipeline builder pattern
- Filter chain support
- Progress reporting

## Extended Codec Crates

| Crate | Description |
|-------|-------------|
| `transcode-av1` | AV1 via rav1e (encode) and dav1d (decode) |
| `transcode-hevc` | H.265/HEVC codec |
| `transcode-vp9` | VP9 codec |
| `transcode-opus` | Opus audio codec |

## Container Crates

| Crate | Description |
|-------|-------------|
| `transcode-mkv` | Matroska/WebM container |
| `transcode-ts` | MPEG Transport Stream |
| `transcode-streaming` | HLS/DASH output |

## Processing Crates

| Crate | Description |
|-------|-------------|
| `transcode-gpu` | wgpu-based GPU compute |
| `transcode-ai` | Neural upscaling, denoising |
| `transcode-quality` | PSNR, SSIM, VMAF metrics |
| `transcode-deinterlace` | Deinterlacing filters |
| `transcode-framerate` | Frame rate conversion |
| `transcode-loudness` | Audio loudness normalization |
| `transcode-hdr` | HDR tone mapping |

## Infrastructure Crates

| Crate | Description |
|-------|-------------|
| `transcode-distributed` | Distributed transcoding |
| `transcode-intel` | Content intelligence |
| `transcode-telemetry` | Metrics and observability |
| `transcode-drm` | DRM encryption |
| `transcode-hwaccel` | Hardware acceleration |

## Platform Crates

| Crate | Description |
|-------|-------------|
| `transcode-cli` | Command-line interface |
| `transcode-python` | Python bindings (PyO3) |
| `transcode-wasm` | WebAssembly support |
| `transcode-compat` | FFmpeg compatibility layer |

## Data Flow

```
Input File
    │
    ▼
┌──────────┐     ┌────────┐     ┌─────────┐     ┌──────────┐
│ Demuxer  │────▶│Decoder │────▶│ Filters │────▶│ Encoder  │
└──────────┘     └────────┘     └─────────┘     └──────────┘
                                                      │
                                                      ▼
                                               ┌──────────┐
                                               │  Muxer   │
                                               └──────────┘
                                                      │
                                                      ▼
                                               Output File
```

## Design Principles

1. **Memory Safety**: Pure Rust with no unsafe blocks in core paths
2. **Modularity**: Each concern is a separate crate
3. **Zero-Copy**: Minimize data copying through careful lifetime management
4. **SIMD Acceleration**: Optimized DSP operations with runtime dispatch
5. **Async Support**: Optional async APIs via Tokio

## Error Handling

All crates use `thiserror` for error types with proper error chains:

```rust
use transcode_core::error::{Error, Result};

fn process() -> Result<()> {
    // Errors propagate with context
}
```

## Thread Safety

- Core types are `Send + Sync` where appropriate
- Pipeline supports parallel filter execution
- Distributed crate enables multi-node processing

## Feature Flags

The workspace uses feature flags for optional functionality:
- `async` - Async I/O support
- `simd` - SIMD optimizations (enabled by default)
- Hardware acceleration features per platform
