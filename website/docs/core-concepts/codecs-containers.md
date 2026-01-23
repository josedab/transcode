---
sidebar_position: 2
title: Codecs & Containers
description: Understanding video codecs, audio codecs, and container formats
---

# Codecs & Containers

Understanding the difference between codecs and containers is fundamental to working with video.

## What's the Difference?

- **Codec** (coder-decoder): Compresses and decompresses video/audio data
- **Container**: Packages multiple streams (video, audio, subtitles) into a single file

Think of it like this:
- The codec is the **language** (how data is encoded)
- The container is the **envelope** (how streams are packaged)

```
┌─────────────────────────────────────────┐
│           Container (MP4)               │
│  ┌─────────────────────────────────┐   │
│  │  Video Stream (H.264 codec)     │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │  Audio Stream (AAC codec)       │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │  Subtitle Stream (SRT)          │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

## Video Codecs

### H.264/AVC

The most widely supported video codec.

```rust
use transcode_codecs::video::h264::{H264Encoder, H264EncoderConfig};

let config = H264EncoderConfig {
    width: 1920,
    height: 1080,
    bitrate: 5_000_000,
    profile: H264Profile::High,
    level: H264Level::L4_1,
    ..Default::default()
};

let encoder = H264Encoder::new(config)?;
```

**Best for:**
- Maximum compatibility (plays everywhere)
- Streaming to older devices
- When decode performance matters

### H.265/HEVC

50% better compression than H.264 at similar quality.

```rust
use transcode_hevc::{HevcEncoder, HevcEncoderConfig};

let config = HevcEncoderConfig {
    width: 3840,
    height: 2160,
    bitrate: 15_000_000,
    preset: HevcPreset::Medium,
    ..Default::default()
};

let encoder = HevcEncoder::new(config)?;
```

**Best for:**
- 4K/UHD content
- Storage-constrained scenarios
- Modern devices and platforms

### AV1

Best compression available, ~30% better than HEVC.

```rust
use transcode_av1::{Av1Encoder, Av1Config};

let config = Av1Config {
    width: 1920,
    height: 1080,
    bitrate: 3_000_000,
    speed: 6,  // 0-10, higher = faster
    ..Default::default()
};

let encoder = Av1Encoder::new(config)?;
```

**Best for:**
- Web streaming (YouTube, Netflix use AV1)
- Bandwidth-constrained delivery
- When encode time isn't critical

### VP9

Google's codec, widely used for WebM.

```rust
use transcode_vp9::{Vp9Encoder, Vp9Config};

let encoder = Vp9Encoder::new(Vp9Config {
    width: 1920,
    height: 1080,
    bitrate: 4_000_000,
    ..Default::default()
})?;
```

**Best for:**
- WebM containers
- YouTube uploads
- Browser-based playback

### Professional Codecs

For broadcast and post-production:

| Codec | Use Case | Crate |
|-------|----------|-------|
| ProRes | Apple ecosystem, editing | `transcode-prores` |
| DNxHD/DNxHR | Avid workflows | `transcode-dnxhd` |
| FFV1 | Archival, lossless | `transcode-ffv1` |
| Cineform | GoPro footage | `transcode-cineform` |

## Audio Codecs

### AAC

Most compatible lossy audio codec.

```rust
use transcode_codecs::audio::aac::{AacEncoder, AacConfig};

let encoder = AacEncoder::new(AacConfig {
    sample_rate: 48000,
    channels: 2,
    bitrate: 128_000,
    profile: AacProfile::Lc,
})?;
```

### Opus

Best quality-to-size ratio for lossy audio.

```rust
use transcode_opus::{OpusEncoder, OpusConfig};

let encoder = OpusEncoder::new(OpusConfig {
    sample_rate: 48000,
    channels: 2,
    bitrate: 128_000,
    application: OpusApplication::Audio,
})?;
```

### FLAC

Lossless audio compression.

```rust
use transcode_flac::{FlacEncoder, FlacConfig};

let encoder = FlacEncoder::new(FlacConfig {
    sample_rate: 44100,
    channels: 2,
    bits_per_sample: 16,
    compression_level: 5,
})?;
```

### Codec Comparison

| Codec | Type | Quality | Size | Compatibility |
|-------|------|---------|------|---------------|
| AAC | Lossy | Good | Small | Excellent |
| Opus | Lossy | Excellent | Smallest | Good |
| FLAC | Lossless | Perfect | Large | Good |
| MP3 | Lossy | Fair | Small | Excellent |
| AC3 | Lossy | Good | Medium | DVD/Blu-ray |

## Container Formats

### MP4/MOV

The most universal container format.

```rust
use transcode_containers::mp4::{Mp4Muxer, Mp4Config};

let muxer = Mp4Muxer::new(output_path, Mp4Config {
    brand: Mp4Brand::Isom,
    fragment: false,
    ..Default::default()
})?;
```

**Supports:** H.264, H.265, AAC, AC3, subtitles

### MKV (Matroska)

Flexible container supporting almost any codec.

```rust
use transcode_mkv::{MkvMuxer, MkvConfig};

let muxer = MkvMuxer::new(output_path, MkvConfig::default())?;
```

**Supports:** Nearly all video/audio codecs, multiple subtitle tracks

### WebM

Web-optimized container (subset of MKV).

```rust
use transcode_webm::{WebmMuxer, WebmConfig};

let muxer = WebmMuxer::new(output_path, WebmConfig::default())?;
```

**Supports:** VP8, VP9, AV1, Vorbis, Opus

### HLS (HTTP Live Streaming)

Apple's adaptive streaming format.

```rust
use transcode_streaming::{HlsMuxer, HlsConfig};

let muxer = HlsMuxer::new("output/", HlsConfig {
    segment_duration: 6.0,
    playlist_type: PlaylistType::Vod,
    ..Default::default()
})?;
```

**Output structure:**
```
output/
├── playlist.m3u8
├── segment_000.ts
├── segment_001.ts
└── segment_002.ts
```

### DASH (Dynamic Adaptive Streaming)

Industry standard for adaptive streaming.

```rust
use transcode_streaming::{DashMuxer, DashConfig};

let muxer = DashMuxer::new("output/", DashConfig {
    segment_duration: 4.0,
    ..Default::default()
})?;
```

## Codec/Container Compatibility

Not all codecs work with all containers:

| Container | H.264 | H.265 | AV1 | VP9 | AAC | Opus |
|-----------|-------|-------|-----|-----|-----|------|
| MP4 | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| MKV | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| WebM | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ |
| HLS | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| DASH | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

## Choosing the Right Combination

### For Web Streaming

```rust
// Maximum compatibility
let options = TranscodeOptions::new()
    .video_codec("h264")
    .audio_codec("aac")
    .output("output.mp4");

// Modern browsers, best compression
let options = TranscodeOptions::new()
    .video_codec("av1")
    .audio_codec("opus")
    .output("output.webm");
```

### For Archival

```rust
// Lossless preservation
let options = TranscodeOptions::new()
    .video_codec("ffv1")
    .audio_codec("flac")
    .output("archive.mkv");
```

### For Editing

```rust
// Professional editing workflow
let options = TranscodeOptions::new()
    .video_codec("prores")
    .audio_codec("pcm")
    .output("edit.mov");
```

## Next Steps

- [Frames & Packets](/docs/core-concepts/frames-packets) - Understanding raw and compressed data
- [Pipeline](/docs/core-concepts/pipeline) - How to chain codecs together
- [Codec Matrix](/docs/reference/codecs-matrix) - Complete compatibility table
