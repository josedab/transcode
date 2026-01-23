---
sidebar_position: 3
title: Codecs Matrix
description: Complete list of supported codecs and containers
---

# Codecs Matrix

Complete reference of supported video codecs, audio codecs, and container formats.

## Video Codecs

### Encoding Support

| Codec | Encode | Decode | Hardware | Notes |
|-------|:------:|:------:|:--------:|-------|
| **H.264/AVC** | ✅ | ✅ | ✅ | Most compatible |
| **H.265/HEVC** | ✅ | ✅ | ✅ | 50% smaller than H.264 |
| **AV1** | ✅ | ✅ | ⚠️ | Best compression, slower |
| **VP9** | ✅ | ✅ | ⚠️ | WebM standard |
| **VP8** | ✅ | ✅ | ❌ | Legacy WebM |
| **ProRes** | ✅ | ✅ | ❌ | Professional editing |
| **DNxHD/DNxHR** | ✅ | ✅ | ❌ | Avid workflows |
| **MJPEG** | ✅ | ✅ | ❌ | Motion JPEG |
| **MPEG-2** | ❌ | ✅ | ❌ | Decode only |
| **MPEG-4** | ❌ | ✅ | ❌ | Decode only |
| **Theora** | ❌ | ✅ | ❌ | Decode only |

### H.264/AVC

```rust
use transcode_codecs::h264::{H264Encoder, H264EncoderConfig, Profile, Level};

let config = H264EncoderConfig {
    profile: Profile::High,
    level: Level::L41,
    bitrate: 5_000_000,
    ..Default::default()
};
```

| Profile | Description | Use Case |
|---------|-------------|----------|
| `Baseline` | No B-frames, no CABAC | Mobile, video calls |
| `Main` | B-frames, CABAC | Broadcast |
| `High` | 8x8 transform | Streaming, Blu-ray |
| `High10` | 10-bit color | HDR content |

| Level | Max Resolution | Max Bitrate |
|-------|----------------|-------------|
| 3.0 | 720x480@30 | 10 Mbps |
| 3.1 | 1280x720@30 | 14 Mbps |
| 4.0 | 1920x1080@30 | 20 Mbps |
| 4.1 | 1920x1080@30 | 50 Mbps |
| 5.0 | 1920x1080@60 | 135 Mbps |
| 5.1 | 3840x2160@30 | 240 Mbps |

### H.265/HEVC

```rust
use transcode_codecs::h265::{H265Encoder, H265EncoderConfig};

let config = H265EncoderConfig {
    profile: Profile::Main,
    bitrate: 3_000_000,  // ~50% less than H.264
    ..Default::default()
};
```

| Profile | Description |
|---------|-------------|
| `Main` | 8-bit 4:2:0 |
| `Main10` | 10-bit 4:2:0 |
| `Main12` | 12-bit support |
| `MainStillPicture` | Still images |

### AV1

```rust
use transcode_av1::{Av1Encoder, Av1EncoderConfig};

let config = Av1EncoderConfig {
    speed: 6,        // 0-10, higher = faster
    bitrate: 2_500_000,
    tiles: 4,        // Parallel encoding
    ..Default::default()
};
```

| Speed | Quality | Use Case |
|-------|---------|----------|
| 0-2 | Highest | Final encoding |
| 3-5 | Good | Production |
| 6-8 | Medium | Real-time |
| 9-10 | Lower | Live streaming |

### VP9

```rust
use transcode_codecs::vp9::{Vp9Encoder, Vp9EncoderConfig};

let config = Vp9EncoderConfig {
    bitrate: 3_000_000,
    quality: Quality::Good,
    ..Default::default()
};
```

### ProRes

```rust
use transcode_codecs::prores::{ProResEncoder, ProResProfile};

let encoder = ProResEncoder::new(ProResProfile::ProRes422HQ)?;
```

| Profile | Bitrate (1080p30) | Use Case |
|---------|-------------------|----------|
| `ProRes422Proxy` | ~45 Mbps | Offline editing |
| `ProRes422LT` | ~100 Mbps | Light editing |
| `ProRes422` | ~150 Mbps | Standard editing |
| `ProRes422HQ` | ~220 Mbps | High quality |
| `ProRes4444` | ~330 Mbps | VFX, alpha channel |
| `ProRes4444XQ` | ~500 Mbps | Highest quality |

## Audio Codecs

### Encoding Support

| Codec | Encode | Decode | Lossy | Notes |
|-------|:------:|:------:|:-----:|-------|
| **AAC** | ✅ | ✅ | ✅ | MP4 standard |
| **Opus** | ✅ | ✅ | ✅ | Best quality/size |
| **MP3** | ✅ | ✅ | ✅ | Universal compatibility |
| **FLAC** | ✅ | ✅ | ❌ | Lossless |
| **ALAC** | ✅ | ✅ | ❌ | Apple lossless |
| **Vorbis** | ✅ | ✅ | ✅ | OGG container |
| **AC-3** | ❌ | ✅ | ✅ | Dolby Digital |
| **E-AC-3** | ❌ | ✅ | ✅ | Dolby Digital Plus |
| **DTS** | ❌ | ✅ | ✅ | Decode only |
| **PCM** | ✅ | ✅ | ❌ | Uncompressed |

### AAC

```rust
use transcode_codecs::aac::{AacEncoder, AacEncoderConfig, AacProfile};

let config = AacEncoderConfig {
    profile: AacProfile::Lc,
    bitrate: 128_000,
    sample_rate: 48000,
    channels: 2,
};
```

| Profile | Description | Bitrate Range |
|---------|-------------|---------------|
| `Lc` | Low Complexity | 64-320 kbps |
| `He` | High Efficiency | 32-128 kbps |
| `HeV2` | HE-AAC v2 | 24-64 kbps |
| `Ld` | Low Delay | Real-time |

### Opus

```rust
use transcode_codecs::opus::{OpusEncoder, OpusEncoderConfig};

let config = OpusEncoderConfig {
    bitrate: 128_000,
    sample_rate: 48000,
    channels: 2,
    application: Application::Audio,
};
```

| Bitrate | Quality | Use Case |
|---------|---------|----------|
| 32 kbps | Speech | VoIP |
| 64 kbps | Good | Podcasts |
| 128 kbps | Excellent | Music streaming |
| 256 kbps | Transparent | Archival |

### MP3

```rust
use transcode_codecs::mp3::{Mp3Encoder, Mp3EncoderConfig};

let config = Mp3EncoderConfig {
    bitrate: 320_000,
    mode: Mode::Vbr,
    quality: 2,  // 0-9, lower = better
};
```

### FLAC

```rust
use transcode_codecs::flac::{FlacEncoder, FlacEncoderConfig};

let config = FlacEncoderConfig {
    compression: 5,  // 0-8, higher = smaller
    sample_rate: 48000,
    bits_per_sample: 24,
    channels: 2,
};
```

## Container Formats

### Support Matrix

| Container | Read | Write | Video | Audio | Subtitles |
|-----------|:----:|:-----:|:-----:|:-----:|:---------:|
| **MP4** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **MKV** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **WebM** | ✅ | ✅ | VP8/VP9/AV1 | Opus/Vorbis | ✅ |
| **MOV** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **AVI** | ✅ | ⚠️ | ✅ | ✅ | ❌ |
| **MPEG-TS** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **FLV** | ✅ | ⚠️ | H.264 | AAC/MP3 | ❌ |
| **OGG** | ✅ | ✅ | Theora | Vorbis/Opus | ❌ |
| **WAV** | ✅ | ✅ | ❌ | PCM | ❌ |
| **MP3** | ✅ | ✅ | ❌ | MP3 | ❌ |
| **FLAC** | ✅ | ✅ | ❌ | FLAC | ❌ |
| **M4A** | ✅ | ✅ | ❌ | AAC/ALAC | ❌ |

### MP4

```rust
use transcode_containers::mp4::{Mp4Muxer, Mp4MuxerConfig};

let config = Mp4MuxerConfig {
    brand: Brand::Isom,
    faststart: true,     // Move moov to beginning
    fragment: false,     // Fragmented MP4
    ..Default::default()
};
```

**Supported codecs:**
- Video: H.264, H.265, AV1, VP9
- Audio: AAC, Opus, AC-3, ALAC, MP3

### MKV (Matroska)

```rust
use transcode_containers::mkv::{MkvMuxer, MkvMuxerConfig};

let config = MkvMuxerConfig {
    write_cues: true,    // Seek index
    ..Default::default()
};
```

**Supported codecs:** Nearly all video and audio codecs

### WebM

```rust
use transcode_containers::webm::{WebmMuxer, WebmMuxerConfig};

let config = WebmMuxerConfig::default();
```

**Supported codecs:**
- Video: VP8, VP9, AV1
- Audio: Vorbis, Opus

### HLS (HTTP Live Streaming)

```rust
use transcode_streaming::hls::{HlsMuxer, HlsConfig};

let config = HlsConfig {
    segment_duration: Duration::from_secs(6),
    playlist_type: PlaylistType::Vod,
    ..Default::default()
};
```

**Supported codecs:**
- Video: H.264, H.265
- Audio: AAC, AC-3

### DASH (Dynamic Adaptive Streaming)

```rust
use transcode_streaming::dash::{DashMuxer, DashConfig};

let config = DashConfig {
    segment_duration: Duration::from_secs(4),
    ..Default::default()
};
```

**Supported codecs:**
- Video: H.264, H.265, VP9, AV1
- Audio: AAC, Opus

## Pixel Formats

| Format | Bits | Chroma | Description |
|--------|------|--------|-------------|
| `Yuv420p` | 8 | 4:2:0 | Most common |
| `Yuv422p` | 8 | 4:2:2 | Broadcast |
| `Yuv444p` | 8 | 4:4:4 | No chroma subsampling |
| `Yuv420p10le` | 10 | 4:2:0 | HDR content |
| `Yuv422p10le` | 10 | 4:2:2 | Professional |
| `Rgb24` | 8 | RGB | Full color |
| `Rgba` | 8 | RGBA | With alpha |
| `Nv12` | 8 | 4:2:0 | Hardware acceleration |

## Hardware Acceleration

### Encoding

| Codec | NVENC | QSV | VideoToolbox | VA-API |
|-------|:-----:|:---:|:------------:|:------:|
| H.264 | ✅ | ✅ | ✅ | ✅ |
| H.265 | ✅ | ✅ | ✅ | ✅ |
| AV1 | ⚠️ | ⚠️ | ❌ | ⚠️ |
| VP9 | ❌ | ⚠️ | ❌ | ⚠️ |

### Decoding

| Codec | NVDEC | QSV | VideoToolbox | VA-API |
|-------|:-----:|:---:|:------------:|:------:|
| H.264 | ✅ | ✅ | ✅ | ✅ |
| H.265 | ✅ | ✅ | ✅ | ✅ |
| AV1 | ✅ | ✅ | ❌ | ✅ |
| VP9 | ✅ | ✅ | ✅ | ✅ |

## Codec Recommendations

### Web Delivery

| Use Case | Video | Audio | Container |
|----------|-------|-------|-----------|
| Maximum compatibility | H.264 | AAC | MP4 |
| Modern browsers | H.265/AV1 | Opus | MP4/WebM |
| Lowest bandwidth | AV1 | Opus | WebM |

### Professional

| Use Case | Video | Audio | Container |
|----------|-------|-------|-----------|
| Editing | ProRes 422 | PCM | MOV |
| Archive | ProRes 4444 | PCM | MOV |
| Exchange | DNxHD | PCM | MKV |

### Streaming

| Platform | Video | Audio | Format |
|----------|-------|-------|--------|
| Generic HLS | H.264 | AAC | HLS |
| High-end HLS | H.265 | AAC | HLS |
| Modern DASH | AV1 | Opus | DASH |

## Next Steps

- [Configuration](/docs/reference/configuration) - Full config reference
- [CLI Reference](/docs/reference/cli) - Command-line options
- [Hardware Acceleration](/docs/advanced/hardware-acceleration) - GPU encoding
