---
sidebar_position: 0
title: Why Transcode?
description: How Transcode compares to FFmpeg, GStreamer, and other alternatives
---

# Why Transcode?

Transcode is a modern, memory-safe codec library built from the ground up in Rust. Here's how it compares to existing solutions and when you should consider using it.

## The Problem with Traditional Codec Libraries

Most video processing today relies on libraries written in C/C++ decades ago:

- **Memory safety vulnerabilities** - Buffer overflows, use-after-free, and other memory bugs are common in codec implementations
- **Complex dependencies** - FFmpeg requires dozens of system libraries and careful compilation
- **Difficult embedding** - Integrating C libraries into modern languages requires FFI, bindings, and careful memory management
- **Monolithic architecture** - All-or-nothing approach makes it hard to use just what you need

## Transcode's Approach

| Aspect | Transcode | FFmpeg | GStreamer |
|--------|-----------|--------|-----------|
| **Language** | Pure Rust | C | C |
| **Memory Safety** | Guaranteed | Manual | Manual |
| **Dependencies** | Minimal | Many system libs | Many plugins |
| **Binary Size** | ~5MB (core) | ~50MB+ | ~100MB+ |
| **Embedding** | Native crate | FFI required | FFI required |
| **WebAssembly** | First-class | Limited | Not supported |
| **Learning Curve** | Rust idioms | Complex API | Plugin system |

## When to Use Transcode

### Choose Transcode When:

**Building Rust applications**
- Native integration with no FFI overhead
- Cargo-based dependency management
- Type-safe API with compile-time checks

**Security is critical**
- Memory-safe implementation eliminates entire classes of vulnerabilities
- Safe to process untrusted media files
- No buffer overflow exploits possible

**Deploying to WebAssembly**
- First-class WASM support with Web Workers
- Browser-based video processing
- Edge computing scenarios

**Embedding in other languages**
- Clean Python bindings via PyO3
- Node.js bindings via N-API
- C API for other language FFI

**Minimizing dependencies**
- No system library requirements for core functionality
- Predictable builds across platforms
- Smaller container images

### Consider FFmpeg When:

- You need every codec ever created (including obscure ones)
- You're using shell scripting for video processing
- You have existing FFmpeg pipelines to maintain
- You need real-time capture from hardware devices

### Consider GStreamer When:

- You're building a media player application
- You need complex pipeline graphs with branching
- You're working in a GTK/GNOME environment
- You need hardware-specific plugins

## Feature Comparison

### Codec Support

| Codec | Transcode | FFmpeg | GStreamer |
|-------|-----------|--------|-----------|
| H.264/AVC | ✅ Native | ✅ Native | ✅ Plugin |
| H.265/HEVC | ✅ Native | ✅ Native | ✅ Plugin |
| AV1 | ✅ rav1e/dav1d | ✅ Multiple | ✅ Plugin |
| VP9 | ✅ Native | ✅ Native | ✅ Plugin |
| ProRes | ✅ Native | ✅ Native | ✅ Plugin |
| AAC | ✅ Native | ✅ Native | ✅ Plugin |
| Opus | ✅ Native | ✅ Native | ✅ Plugin |
| FLAC | ✅ Native | ✅ Native | ✅ Plugin |

### Platform Support

| Platform | Transcode | FFmpeg | GStreamer |
|----------|-----------|--------|-----------|
| Linux | ✅ | ✅ | ✅ |
| macOS | ✅ | ✅ | ✅ |
| Windows | ✅ | ✅ | ✅ |
| WebAssembly | ✅ | ⚠️ Limited | ❌ |
| iOS | ✅ | ⚠️ Complex | ⚠️ Complex |
| Android | ✅ | ⚠️ Complex | ⚠️ Complex |

### Modern Features

| Feature | Transcode | FFmpeg | GStreamer |
|---------|-----------|--------|-----------|
| GPU Compute (wgpu) | ✅ | ❌ | ⚠️ CUDA only |
| AI Enhancement | ✅ Built-in | ❌ External | ❌ External |
| Quality Metrics | ✅ PSNR/SSIM/VMAF | ⚠️ Filter-based | ⚠️ Plugin |
| Distributed Processing | ✅ Built-in | ❌ External | ❌ External |
| Content Intelligence | ✅ Built-in | ❌ | ❌ |

## Code Comparison

### Simple Transcode

**Transcode (Rust):**
```rust
use transcode::{Transcoder, TranscodeOptions};

let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.mp4")
    .video_bitrate(5_000_000);

Transcoder::new(options)?.run()?;
```

**FFmpeg (Command line):**
```bash
ffmpeg -i input.mp4 -b:v 5M output.mp4
```

**FFmpeg (C API):**
```c
// 200+ lines of boilerplate for format context,
// codec context, packet allocation, frame handling,
// error checking, and cleanup...
```

### Quality Assessment

**Transcode:**
```rust
use transcode_quality::{Psnr, Ssim, QualityAssessment};

let qa = QualityAssessment::new();
let report = qa.assess(&reference, &distorted)?;

println!("PSNR: {:.2} dB", report.psnr.average());
println!("SSIM: {:.4}", report.ssim.score);
```

**FFmpeg:**
```bash
ffmpeg -i reference.mp4 -i distorted.mp4 \
  -lavfi "[0:v][1:v]psnr" -f null -
# Then parse the log output manually
```

## Performance

Transcode is designed for performance:

- **SIMD optimization** - AVX2, AVX-512, NEON with runtime detection
- **Zero-copy pipelines** - Minimize memory allocations
- **Parallel processing** - Multi-threaded encoding/decoding
- **GPU acceleration** - wgpu compute shaders for filters

See our [benchmarks](/docs/reference/benchmarks) for detailed performance comparisons.

## Migration from FFmpeg

If you're coming from FFmpeg, here's a quick translation guide:

| FFmpeg | Transcode CLI | Transcode Rust |
|--------|---------------|----------------|
| `-i input.mp4` | `-i input.mp4` | `.input("input.mp4")` |
| `-c:v libx264` | `--video-codec h264` | `.video_codec("h264")` |
| `-b:v 5M` | `--video-bitrate 5000` | `.video_bitrate(5_000_000)` |
| `-crf 23` | `--crf 23` | `.video_crf(23)` |
| `-c:a aac` | `--audio-codec aac` | `.audio_codec("aac")` |
| `-vf scale=1920:1080` | `-F "scale=1920:1080"` | `.add_filter(Scale::new(1920, 1080))` |

## Getting Started

Ready to try Transcode?

```bash
# Install the CLI
cargo install transcode-cli

# Or add to your Rust project
cargo add transcode
```

Continue to [Installation](/docs/getting-started/installation) for detailed setup instructions.
