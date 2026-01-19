# Architecture Overview

Transcode is organized as a modular workspace with specialized crates for different concerns.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              transcode (facade)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                           transcode-pipeline                                 │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│    │  Demux   │ →  │  Decode  │ →  │  Filter  │ →  │  Encode  │ → Mux     │
│    └──────────┘    └──────────┘    └──────────┘    └──────────┘           │
├─────────────────────────────────────────────────────────────────────────────┤
│  transcode-containers  │  transcode-codecs  │  transcode-core              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Crate Organization

### Core Layer

| Crate | Purpose |
|-------|---------|
| `transcode-core` | Fundamental types: frames, packets, timestamps, errors |
| `transcode-codecs` | Codec trait definitions and base implementations |
| `transcode-containers` | Container format demuxing and muxing |

### Codec Crates

Each major codec has its own crate:

- `transcode-av1` - AV1 (via rav1e/dav1d)
- `transcode-hevc` - H.265/HEVC
- `transcode-vp9`, `transcode-vp8` - VP9/VP8
- `transcode-opus`, `transcode-flac` - Audio codecs

### Processing Crates

| Crate | Purpose |
|-------|---------|
| `transcode-gpu` | GPU compute via wgpu |
| `transcode-ai` | Neural network enhancement |
| `transcode-quality` | PSNR, SSIM, VMAF metrics |

### Integration Crates

| Crate | Purpose |
|-------|---------|
| `transcode` | Public API facade |
| `transcode-cli` | Command-line interface |
| `transcode-python` | Python bindings (PyO3) |
| `transcode-wasm` | WebAssembly support |

## Data Flow

### Transcoding Pipeline

```
Input File
    │
    ▼
┌─────────────┐
│   Demuxer   │  ← Reads container, extracts packets
└──────┬──────┘
       │ Packet (compressed)
       ▼
┌─────────────┐
│   Decoder   │  ← Decompresses video/audio
└──────┬──────┘
       │ Frame (raw)
       ▼
┌─────────────┐
│   Filters   │  ← Scale, crop, color convert, etc.
└──────┬──────┘
       │ Frame (processed)
       ▼
┌─────────────┐
│   Encoder   │  ← Compresses to target codec
└──────┬──────┘
       │ Packet (compressed)
       ▼
┌─────────────┐
│    Muxer    │  ← Writes container format
└─────────────┘
       │
       ▼
Output File
```

### Frame Types

```rust
// Raw video frame
pub struct Frame {
    pub data: Vec<Vec<u8>>,      // Plane data (Y, U, V)
    pub width: u32,
    pub height: u32,
    pub pixel_format: PixelFormat,
    pub pts: Timestamp,
    pub key_frame: bool,
}

// Compressed packet
pub struct Packet<'a> {
    pub data: Cow<'a, [u8]>,
    pub pts: Timestamp,
    pub dts: Timestamp,
    pub flags: PacketFlags,
    pub stream_index: usize,
}
```

## Key Design Decisions

1. **Pure Rust**: No FFI to C libraries for core functionality
2. **Trait-based abstraction**: Codecs implement common traits
3. **Zero-copy where possible**: `Cow<'a, [u8]>` for packet data
4. **SIMD abstraction**: Runtime detection, platform-specific implementations
5. **Async-ready**: Pipeline uses tokio for I/O

See [Architecture Decision Records](../adr/index.md) for detailed rationale.

## Thread Model

```
┌─────────────────────────────────────────────────────────────────┐
│                        Main Thread                               │
│  ┌─────────────┐                                                │
│  │  Pipeline   │ ← Orchestrates processing                      │
│  └──────┬──────┘                                                │
│         │                                                        │
├─────────┼────────────────────────────────────────────────────────┤
│         ▼                                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Decode    │  │   Decode    │  │   Decode    │  ← Thread   │
│  │   Worker    │  │   Worker    │  │   Worker    │    Pool     │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐                               │
│  │   Encode    │  │   Encode    │  ← Encoding is               │
│  │   Worker    │  │   Worker    │    parallelized              │
│  └─────────────┘  └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

## Error Handling

Transcode uses a hierarchical error system:

```rust
pub enum Error {
    Codec(CodecError),
    Container(ContainerError),
    Bitstream(BitstreamError),
    Io(std::io::Error),
    WithContext { source: Box<Error>, context: String },
}

pub enum CodecError {
    Av1(Av1ErrorKind),
    Opus(OpusErrorKind),
    UnsupportedProfile { profile: u8 },
    // ...
}
```

Errors can be chained with context:

```rust
decoder.decode(packet)
    .context("Failed to decode frame")?
```
