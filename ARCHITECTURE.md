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

## Professional Audio Crates

| Crate | Description | FFI Required |
|-------|-------------|--------------|
| `transcode-ac3` | AC-3/E-AC-3 (Dolby Digital) | Optional* |
| `transcode-dts` | DTS/DTS-HD/TrueHD audio | Optional* |
| `transcode-spatial` | Spatial/immersive audio | No |

*Native parsing available without FFI. Full encode/decode requires `ffi-ffmpeg` feature and appropriate licenses.

## Image/Cinema Crates

| Crate | Description | FFI Required |
|-------|-------------|--------------|
| `transcode-jpeg2000` | JPEG2000 (DCP/cinema) | Optional* |
| `transcode-prores` | Apple ProRes | No |
| `transcode-dnxhd` | DNxHD/DNxHR | No |
| `transcode-openexr` | OpenEXR HDR images | Optional |

*Native parsing available without FFI. Full encode/decode requires `ffi-openjpeg` feature.

## Legacy/Broadcast Crates

| Crate | Description | FFI Required |
|-------|-------------|--------------|
| `transcode-mpeg2` | MPEG-2 video | Optional* |
| `transcode-avi` | AVI container | No |
| `transcode-mxf` | MXF container | No |

*Native parsing available without FFI. Full encode/decode requires `ffi-ffmpeg` feature and appropriate licenses.

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

## FFI Wrapper Architecture

For patent-encumbered codecs, we use an FFI wrapper pattern that separates native parsing from licensed encode/decode functionality:

```
┌─────────────────────────────────────────────────────────────┐
│                    transcode-dts crate                       │
├─────────────────────────────────────────────────────────────┤
│  Native Layer (always available)                             │
│  ├─ Sync word detection (0x7FFE8001, 0xF8726FBA)            │
│  ├─ Header parsing (sample rate, channels, bit depth)       │
│  ├─ Format detection (Core, HD, MA, TrueHD, Atmos)          │
│  └─ Metadata extraction                                      │
├─────────────────────────────────────────────────────────────┤
│  FFI Layer (optional, requires ffi-ffmpeg feature)          │
│  ├─ Full audio decoding                                      │
│  ├─ Audio encoding                                           │
│  └─ User must obtain appropriate licenses                    │
└─────────────────────────────────────────────────────────────┘
```

### Codec Licensing Strategy

| Category | Implementation | Licensing |
|----------|----------------|-----------|
| **Royalty-Free** | Native Rust | Open source (MIT/Apache-2.0) |
| **Patent-Encumbered** | FFI Wrapper | User provides license |

**Royalty-Free Codecs (Native):**
- AV1, VP8, VP9, Opus, Vorbis, FLAC, Theora, FFV1
- ALAC (Apple allows implementation)
- ProRes (Apple open specification)
- DNxHD (Avid open specification)

**Patent-Encumbered (FFI Wrapper Only):**
- AC-3/E-AC-3 (Dolby licensing required)
- DTS/DTS-HD/TrueHD (DTS/Dolby licensing required)
- MPEG-2 (MPEG-LA licensing required)

### FFI Feature Flags

```toml
# Enable FFmpeg-based decode/encode for licensed codecs
transcode-dts = { version = "0.1", features = ["ffi-ffmpeg"] }
transcode-ac3 = { version = "0.1", features = ["ffi-ffmpeg"] }
transcode-mpeg2 = { version = "0.1", features = ["ffi-ffmpeg"] }

# Enable OpenJPEG for JPEG2000
transcode-jpeg2000 = { version = "0.1", features = ["ffi-openjpeg"] }
```

## Codec Trait Hierarchy

```
                    ┌─────────────┐
                    │ CodecInfo   │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
   ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
   │ VideoDecoder  │ │ VideoEncoder  │ │ AudioDecoder  │
   └───────────────┘ └───────────────┘ └───────────────┘
                                               │
                                       ┌───────┴───────┐
                                       ▼               ▼
                               ┌───────────────┐ ┌───────────────┐
                               │  DtsDecoder   │ │ Ac3Decoder    │
                               └───────────────┘ └───────────────┘
```

All decoders/encoders implement common traits for unified pipeline integration.
