---
sidebar_position: 6
title: Changelog
description: Release history and version notes
---

# Changelog

All notable changes to Transcode are documented here. This project adheres to [Semantic Versioning](https://semver.org/).

For the complete list of releases, see [GitHub Releases](https://github.com/transcode/transcode/releases).

## [1.0.0] - 2025-01-15

The first stable release of Transcode, marking production-ready status for core functionality.

### Highlights

- **12+ video codecs** with full encode/decode support
- **9+ audio codecs** including lossless formats
- **9+ container formats** including streaming (HLS/DASH)
- **GPU acceleration** via wgpu compute shaders
- **AI enhancement** with neural upscaling and denoising
- **Distributed processing** with fault tolerance
- **Multi-platform** support: Rust, Python, Node.js, WebAssembly, Docker

### Added

#### Core
- `transcode-core` - Fundamental types (frames, packets, timestamps, errors)
- `transcode-codecs` - Base codec traits and H.264/AAC implementations
- `transcode-containers` - MP4/MOV container support
- `transcode-pipeline` - Pipeline orchestration system
- `transcode` - High-level facade API

#### Video Codecs
- H.264/AVC - Full encode/decode with all profiles
- H.265/HEVC - Main and Main10 profiles
- AV1 - Via rav1e (encode) and dav1d (decode)
- VP9/VP8 - WebM-compatible encoding
- ProRes - All variants (Proxy to 4444 XQ)
- DNxHD/DNxHR - Avid workflow support
- MPEG-2 - Broadcast compatibility
- FFV1 - Lossless archival
- Theora - Decode-only support
- Cineform - GoPro format support

#### Audio Codecs
- AAC - LC, HE-AAC, HE-AACv2
- Opus - Best-in-class quality/size ratio
- FLAC - Lossless compression
- AC3/E-AC3 - Dolby Digital
- DTS - Surround sound
- ALAC - Apple Lossless
- Vorbis - Open format
- PCM - Uncompressed audio
- MP3 - Decode only

#### Containers
- MP4/MOV - Full mux/demux
- MKV/WebM - Matroska support
- HLS - Adaptive streaming output
- DASH - MPEG-DASH output
- MPEG-TS - Broadcast transport
- AVI - Legacy support
- FLV - Flash video
- MXF - Professional broadcast

#### Processing
- `transcode-gpu` - GPU compute via wgpu
- `transcode-ai` - Neural upscaling, denoising, frame interpolation
- `transcode-quality` - PSNR, SSIM, MS-SSIM, VMAF metrics
- `transcode-intel` - Scene detection, content classification
- `transcode-distributed` - Coordinator/worker architecture

#### Platform Bindings
- `transcode-cli` - Full-featured command-line tool
- `transcode-python` - Python bindings via PyO3
- `transcode-node` - Node.js bindings via N-API
- `transcode-wasm` - WebAssembly with Web Workers
- `transcode-capi` - C API for FFI

#### Hardware Acceleration
- VideoToolbox (macOS) - Full hardware encoding/decoding
- VA-API (Linux) - Intel/AMD hardware support
- NVENC (NVIDIA) - GPU encoding

### Performance
- SIMD optimization for x86_64 (SSE4.2, AVX2, AVX-512)
- SIMD optimization for ARM64 (NEON)
- Zero-copy frame pools
- Multi-threaded encoding/decoding

---

## [0.9.0] - 2024-12-01 (Beta)

### Added
- Initial AV1 support via rav1e/dav1d
- HLS/DASH streaming output
- GPU acceleration foundation
- Distributed processing prototype

### Changed
- Refactored codec trait system
- Improved error messages

### Fixed
- Memory leak in H.264 decoder
- Timestamp handling for variable frame rate

---

## [0.8.0] - 2024-10-15 (Beta)

### Added
- H.265/HEVC codec support
- VP9 codec support
- Quality metrics (PSNR, SSIM)
- Python bindings

### Changed
- Improved pipeline performance
- Better SIMD detection

---

## [0.7.0] - 2024-08-01 (Alpha)

### Added
- Core transcoding pipeline
- H.264 encode/decode
- AAC encode/decode
- MP4 container support
- CLI tool

### Known Issues
- No hardware acceleration
- Limited container support

---

## Version Policy

### Stability Guarantees

**1.x releases** maintain backward compatibility:
- No breaking changes to public APIs within 1.x
- New features added in minor versions (1.1, 1.2, etc.)
- Bug fixes in patch versions (1.0.1, 1.0.2, etc.)

### Deprecation Process

1. Features are marked `#[deprecated]` with migration guidance
2. Deprecated features work for at least 2 minor versions
3. Removal happens in the next major version

### Support Timeline

| Version | Status | Support Until |
|---------|--------|---------------|
| 1.0.x | **Current** | Active development |
| 0.9.x | Maintenance | 2025-06-01 |
| 0.8.x | End of Life | - |
| 0.7.x | End of Life | - |

---

## Upgrading

### From 0.9.x to 1.0.0

Most code should work without changes. Key updates:

```rust
// Old (0.9.x)
use transcode::codec::H264Decoder;

// New (1.0.0)
use transcode_codecs::video::h264::H264Decoder;
```

### From 0.8.x to 0.9.x

The pipeline API was restructured:

```rust
// Old (0.8.x)
let pipeline = Pipeline::new(input, output);

// New (0.9.x+)
let pipeline = PipelineBuilder::new()
    .input(input)
    .output(output)
    .build()?;
```

---

## Contributing to Changelog

When submitting PRs, include changelog entries in the PR description:

```markdown
## Changelog

### Added
- New feature description

### Changed
- Changed behavior description

### Fixed
- Bug fix description
```

See [Contributing Guide](/docs/advanced/contributing) for more details.
