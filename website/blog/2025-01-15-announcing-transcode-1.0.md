---
slug: announcing-transcode-1.0
title: Announcing Transcode 1.0 - Memory-Safe Video Transcoding for Rust
authors: [transcode-team]
tags: [release, rust]
---

We're excited to announce the release of **Transcode 1.0**, a memory-safe, high-performance codec library written entirely in Rust.

After two years of development and extensive testing, Transcode is ready for production use. This release marks a milestone in bringing modern language safety guarantees to video processing.

<!-- truncate -->

## Why We Built Transcode

Video processing has traditionally been dominated by C/C++ libraries like FFmpeg and GStreamer. While incredibly capable, these libraries carry inherent risks:

- **Memory safety vulnerabilities**: Buffer overflows and use-after-free bugs have led to countless CVEs in codec implementations
- **Complex integration**: Embedding C libraries in modern applications requires careful FFI management
- **Build complexity**: Dozens of system dependencies make reproducible builds challenging

Transcode addresses these issues by providing a pure Rust implementation with no unsafe code in the critical paths.

## What's in 1.0

### Production-Ready Codecs

- **Video**: H.264, H.265/HEVC, AV1 (via rav1e/dav1d), VP9, ProRes, DNxHD
- **Audio**: AAC, Opus, FLAC, AC3, DTS
- **Containers**: MP4, MKV, WebM, HLS, DASH

### Modern Features

- **SIMD optimization**: Automatic runtime detection of AVX2, AVX-512, and NEON
- **GPU acceleration**: wgpu-based compute shaders for color conversion and scaling
- **AI enhancement**: Neural upscaling, denoising, and frame interpolation
- **Distributed processing**: Built-in coordinator/worker architecture
- **Quality metrics**: PSNR, SSIM, MS-SSIM, and VMAF

### Multi-Platform Bindings

- Native Rust crate
- Python bindings via PyO3
- Node.js bindings via N-API
- WebAssembly with Web Workers
- C API for other languages

## Quick Start

```bash
# Install the CLI
cargo install transcode-cli

# Transcode a video
transcode -i input.mp4 -o output.mp4 --video-codec h264 --video-bitrate 5000

# Or use as a library
cargo add transcode
```

```rust
use transcode::{Transcoder, TranscodeOptions};

let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.mp4")
    .video_bitrate(5_000_000);

Transcoder::new(options)?.run()?;
```

## Performance

Our benchmarks show Transcode performs competitively with native C implementations while providing memory safety guarantees:

| Operation | Transcode | x264/x265 |
|-----------|-----------|-----------|
| H.264 encode (1080p, medium) | 78 fps | 72 fps |
| H.265 encode (1080p, medium) | 35 fps | 30 fps |
| H.264 decode (1080p) | 420 fps | 400 fps |

See our [benchmarks documentation](/docs/reference/benchmarks) for detailed numbers.

## What's Next

Our roadmap for 2025 includes:

- VVC/H.266 codec support
- Enhanced hardware acceleration (improved VAAPI, NVENC)
- Live streaming input
- Audio enhancement features

## Get Involved

Transcode is open source under MIT/Apache-2.0 dual license.

- [GitHub Repository](https://github.com/transcode/transcode)
- [Documentation](/docs/getting-started/installation)
- [Discord Community](https://discord.gg/transcode)

We welcome contributions! Check out our [contributing guide](/docs/advanced/contributing) to get started.

Thank you to everyone who contributed to making this release possible.
