# Introduction

**Transcode** is a memory-safe, high-performance universal codec library written entirely in Rust. It provides comprehensive support for video and audio encoding/decoding without relying on FFmpeg or other system libraries.

## Why Transcode?

### Memory Safety First

Traditional codec libraries written in C/C++ are prone to buffer overflows and memory corruption vulnerabilities. Transcode eliminates these risks by leveraging Rust's ownership system and borrow checker.

### High Performance

Despite being memory-safe, Transcode doesn't compromise on performance:

- **SIMD Optimizations**: Automatic runtime detection and use of AVX2, SSE4.2 (x86_64) and NEON (ARM64)
- **GPU Acceleration**: Compute shaders via wgpu for color conversion, scaling, and effects
- **Zero-Copy Operations**: Minimize memory allocations in hot paths
- **Parallel Processing**: Multi-threaded encoding and decoding

### Pure Rust, No Dependencies

Transcode is implemented entirely in Rust with no FFmpeg or system library dependencies, making it:

- Easy to cross-compile for any target
- Simple to deploy (single binary)
- Auditable for security

### Comprehensive Format Support

| Category | Formats |
|----------|---------|
| Video Codecs | H.264, H.265/HEVC, AV1, VP9, VP8, ProRes, DNxHD, MPEG-2, FFV1 |
| Audio Codecs | AAC, MP3, Opus, FLAC, AC-3, DTS, ALAC, Vorbis, PCM |
| Containers | MP4, MKV, WebM, MPEG-TS, AVI, FLV, MXF, HLS, DASH |

## Features at a Glance

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRANSCODE                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │ Demuxer │→ │ Decoder │→ │ Filters │→ │ Encoder │→ Muxer    │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘            │
├─────────────────────────────────────────────────────────────────┤
│  GPU Acceleration │ AI Enhancement │ Quality Metrics            │
├─────────────────────────────────────────────────────────────────┤
│  Distributed Processing │ HLS/DASH Streaming │ Content Intel    │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Example

```rust
use transcode::{Transcoder, TranscodeOptions};

fn main() -> transcode::Result<()> {
    let options = TranscodeOptions::new()
        .input("input.mp4")
        .output("output.mp4")
        .video_bitrate(5_000_000)
        .audio_bitrate(128_000);

    let mut transcoder = Transcoder::new(options)?;
    transcoder.run()?;

    println!("Transcoding complete!");
    Ok(())
}
```

## Getting Help

- **Documentation**: You're reading it!
- **API Reference**: [docs.rs/transcode](https://docs.rs/transcode)
- **GitHub Issues**: [Report bugs](https://github.com/transcode/transcode/issues)
- **Discussions**: [Ask questions](https://github.com/transcode/transcode/discussions)

## License

Transcode is dual-licensed under MIT and Apache 2.0. You may choose either license.
