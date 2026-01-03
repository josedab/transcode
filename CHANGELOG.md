# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of the Transcode codec library
- Core types and utilities (`transcode-core`)
  - Bitstream reader/writer with Exp-Golomb support
  - Frame and sample buffer abstractions
  - Timestamp and time base handling
  - Memory pool implementations
- Video codecs (`transcode-codecs`)
  - H.264/AVC decoder and encoder
  - SIMD optimizations (AVX2 for x86_64, NEON for aarch64)
  - Runtime SIMD capability detection
- Audio codecs (`transcode-codecs`)
  - AAC decoder and encoder with ADTS support
  - MP3 decoder and encoder with psychoacoustic model
- Container formats (`transcode-containers`, `transcode-mkv`, `transcode-ts`)
  - MP4/MOV demuxer and muxer
  - ISOBMFF box parsing
  - MKV/WebM demuxer and muxer with full lacing support (Xiph, EBML, fixed-size)
  - MPEG-TS demuxer and muxer with PCR/PTS/DTS handling
- Additional video codecs
  - HEVC/H.265 decoder and encoder (`transcode-hevc`)
  - VP9 decoder and encoder (`transcode-vp9`)
  - AV1 codec via rav1e/dav1d (`transcode-av1`)
- Additional audio codecs
  - Opus decoder and encoder (`transcode-opus`)
- Video processing
  - Deinterlacing filters: bob, weave, yadif (`transcode-deinterlace`)
  - Frame rate conversion with motion interpolation (`transcode-framerate`)
  - HDR processing (`transcode-hdr`)
- Audio processing
  - Loudness normalization (EBU R128, ATSC A/85) (`transcode-loudness`)
- Content protection
  - DRM encryption support (Widevine, FairPlay, PlayReady) (`transcode-drm`)
- Streaming
  - HLS and DASH output (`transcode-streaming`)
  - Live streaming support (`transcode-live`)
- Professional features
  - Timecode support (SMPTE, drop-frame, NDF) (`transcode-timecode`)
  - Closed captioning and subtitles (`transcode-caption`)
  - Watermarking (`transcode-watermark`)
- AI/ML features
  - Neural network upscaling (`transcode-neural`, `transcode-ai`)
  - Audio AI processing (`transcode-audio-ai`)
  - Content intelligence and scene detection (`transcode-intel`, `transcode-intelligence`)
- Infrastructure
  - Distributed transcoding (`transcode-distributed`)
  - Cloud integration (`transcode-cloud`)
  - GPU acceleration via wgpu (`transcode-gpu`)
  - Hardware acceleration (`transcode-hwaccel`)
  - Telemetry and observability (`transcode-telemetry`)
  - Analytics (`transcode-analytics`)
  - Per-title encoding (`transcode-pertitle`)
- Quality metrics
  - PSNR, SSIM, MS-SSIM, VMAF approximation (`transcode-quality`)
- Interoperability
  - FFmpeg compatibility layer (`transcode-compat`)
  - WebAssembly support (`transcode-wasm`)
  - Zero-copy optimizations (`transcode-zerocopy`)
- Pipeline architecture (`transcode-pipeline`)
  - Async transcoding pipeline
  - Filter chain support
  - Frame rate conversion
- High-level API (`transcode`)
  - Simple transcoding interface
  - Format auto-detection
- CLI tool (`transcode-cli`)
  - Progress bar with colored output
  - SIMD capability display
  - Multiple output format support
- Python bindings (`transcode-python`)
  - PyO3-based Python module
  - Cross-platform wheel builds
- Comprehensive testing
  - Unit tests for all modules
  - Integration tests
  - Fuzz testing targets
  - Benchmarks with Criterion (`transcode-bench`)
- CI/CD infrastructure
  - GitHub Actions workflows
  - Multi-platform testing (Linux, macOS, Windows)
  - Code coverage with tarpaulin
  - Security auditing with cargo-deny
  - Docker support for containerized deployment

## [0.1.0] - Unreleased

Initial development release.

[Unreleased]: https://github.com/transcode/transcode/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/transcode/transcode/releases/tag/v0.1.0
