# Changelog

This page documents all notable changes to Transcode.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For the complete changelog, see [CHANGELOG.md](https://github.com/transcode/transcode/blob/main/CHANGELOG.md) in the repository.

## [1.0.0] - 2026-01-12

### Highlights

This is the initial stable release of Transcode, featuring:

- **Complete codec suite**: H.264, H.265/HEVC, AV1, VP9, VP8, and more
- **Audio support**: AAC, Opus, FLAC, MP3, AC-3, DTS
- **Container formats**: MP4, MKV, WebM, MPEG-TS, HLS, DASH
- **GPU acceleration**: via wgpu compute shaders
- **AI enhancement**: Neural upscaling, denoising, frame interpolation
- **Quality metrics**: PSNR, SSIM, MS-SSIM, VMAF
- **Distributed processing**: Multi-worker transcoding with fault tolerance
- **Platform bindings**: CLI, Python, WebAssembly

### Added

- Core types and utilities (`transcode-core`)
- Video codecs with SIMD optimization (AVX2, NEON)
- Audio codecs with psychoacoustic modeling
- Container format support (demuxing and muxing)
- Pipeline architecture for streaming processing
- Hardware acceleration support (VAAPI, VideoToolbox, NVENC)
- Content intelligence (scene detection, classification)
- HLS/DASH streaming output
- Python bindings via PyO3
- WebAssembly support with Web Workers
- Comprehensive test suite including fuzz testing
- Docker support for containerized deployment

### Documentation

- Architecture documentation
- API reference on docs.rs
- Examples for common use cases
- Contributing guide

## Upgrade Guide

### From 0.x to 1.0

If you were using a pre-release version:

1. Update your `Cargo.toml`:
   ```toml
   [dependencies]
   transcode = "1.0"
   ```

2. The `Transcoder` API is now stable. Key changes:
   - `TranscodeOptions` uses builder pattern
   - Error types are now hierarchical
   - Async operations use tokio runtime

3. See the [migration guide](./guides/migration-1.0.md) for detailed instructions.
