# Transcoding Basics

This guide covers the fundamental concepts of video transcoding with Transcode.

## What is Transcoding?

Transcoding is the process of converting media from one format to another. This typically involves:

1. **Demuxing**: Extracting audio and video streams from a container
2. **Decoding**: Decompressing the streams into raw frames
3. **Processing**: Applying filters (scale, crop, color convert)
4. **Encoding**: Compressing frames with a target codec
5. **Muxing**: Packaging streams into an output container

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  Demux  │ →  │ Decode  │ →  │ Process │ →  │ Encode  │ →  │   Mux   │
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
   MP4/MKV       H.264          Scale         H.265           MP4
                 AAC           Denoise         Opus
```

## Key Concepts

### Codecs vs Containers

- **Codec**: Compression algorithm (H.264, HEVC, AAC, Opus)
- **Container**: File format that holds streams (MP4, MKV, WebM)

A container can hold multiple codecs:
```
MP4 Container
├── Video: H.264
├── Audio: AAC (English)
├── Audio: AC-3 (Spanish)
└── Subtitles: SRT
```

### Bitrate and Quality

**Bitrate** is how much data per second is used:
- Higher bitrate = better quality, larger file
- Lower bitrate = lower quality, smaller file

**CRF (Constant Rate Factor)** targets quality instead:
- Lower CRF = better quality (18 is visually lossless)
- Higher CRF = lower quality (28 is acceptable for streaming)

```rust
// Bitrate mode: target specific file size
let options = TranscodeOptions::new()
    .video_bitrate(5_000_000);  // 5 Mbps

// CRF mode: target quality level
let options = TranscodeOptions::new()
    .crf(23);  // Good quality, reasonable size
```

### Resolution and Aspect Ratio

Common resolutions:
- **4K/UHD**: 3840×2160
- **1080p/FHD**: 1920×1080
- **720p/HD**: 1280×720
- **480p/SD**: 854×480

```rust
// Scale to 1080p, maintaining aspect ratio
let options = TranscodeOptions::new()
    .width(1920)
    .height(1080);
```

### Frame Rate

Common frame rates:
- **24 fps**: Cinema
- **30 fps**: TV (NTSC)
- **60 fps**: Gaming, sports
- **25 fps**: TV (PAL)

```rust
// Convert to 30 fps
let options = TranscodeOptions::new()
    .fps(30.0);
```

## Basic Transcoding

### Copy Mode (Remuxing)

Change container without re-encoding:

```rust
let options = TranscodeOptions::new()
    .input("input.mkv")
    .output("output.mp4")
    .video_codec(Codec::Copy)
    .audio_codec(Codec::Copy);
```

This is fast because frames aren't decoded/encoded.

### Re-encoding

Convert to a different codec:

```rust
let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.mp4")
    .video_codec(Codec::H265)
    .audio_codec(Codec::Opus);
```

### Two-Pass Encoding

Better quality for target bitrate:

```rust
let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.mp4")
    .video_bitrate(5_000_000)
    .two_pass(true);
```

Pass 1 analyzes content, Pass 2 uses that data for optimal encoding.

## Common Workflows

### Web Optimization

For streaming on the web:

```rust
let options = TranscodeOptions::new()
    .input("source.mov")
    .output("web.mp4")
    .video_codec(Codec::H264)
    .video_bitrate(2_500_000)
    .audio_codec(Codec::Aac)
    .audio_bitrate(128_000)
    .faststart(true);  // Move moov atom for streaming
```

### Archive Quality

For long-term storage:

```rust
let options = TranscodeOptions::new()
    .input("source.mp4")
    .output("archive.mkv")
    .video_codec(Codec::H265)
    .crf(18)  // Near-lossless
    .preset(Preset::Slow);  // Better compression
```

### Mobile Delivery

For phones and tablets:

```rust
let options = TranscodeOptions::new()
    .input("source.mp4")
    .output("mobile.mp4")
    .width(1280)
    .height(720)
    .video_bitrate(1_500_000)
    .audio_bitrate(96_000);
```

## Understanding Presets

Presets trade encoding speed for compression efficiency:

| Preset | Speed | File Size | Use Case |
|--------|-------|-----------|----------|
| ultrafast | Fastest | Largest | Preview, testing |
| fast | Fast | Large | Quick exports |
| medium | Balanced | Medium | General use |
| slow | Slow | Small | Final delivery |
| veryslow | Slowest | Smallest | Archival |

```rust
let options = TranscodeOptions::new()
    .preset(Preset::Medium);  // Good balance
```

## Next Steps

- [Video Codecs](./video-codecs.md) - Codec-specific options
- [Quality Metrics](./quality-metrics.md) - Measuring output quality
- [GPU Acceleration](./gpu-acceleration.md) - Hardware encoding
