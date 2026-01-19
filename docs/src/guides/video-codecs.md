# Video Codecs

This guide covers the video codecs supported by Transcode and their configuration options.

## Codec Overview

| Codec | Encode | Decode | Best For |
|-------|--------|--------|----------|
| H.264/AVC | ✅ | ✅ | Compatibility, streaming |
| H.265/HEVC | ✅ | ✅ | 4K, HDR, efficiency |
| AV1 | ✅ | ✅ | Web, future-proof |
| VP9 | ✅ | ✅ | WebM, YouTube |
| VP8 | ✅ | ✅ | Legacy WebM |
| ProRes | ✅ | ✅ | Professional editing |
| DNxHD/HR | ✅ | ✅ | Broadcast, Avid |

## H.264/AVC

The most widely supported video codec. Use for maximum compatibility.

### Profiles

| Profile | Features | Use Case |
|---------|----------|----------|
| Baseline | No B-frames, CAVLC only | Mobile, video conferencing |
| Main | B-frames, CABAC | Standard streaming |
| High | 8x8 transform, custom quant | High quality |

```rust
use transcode::codecs::h264::{H264Encoder, Profile, Level};

let encoder = H264Encoder::new()
    .profile(Profile::High)
    .level(Level::L4_1)  // 1080p60
    .build()?;
```

### Recommended Settings

**Streaming (low latency)**:
```rust
let options = TranscodeOptions::new()
    .video_codec(Codec::H264)
    .preset(Preset::Fast)
    .tune(Tune::ZeroLatency)
    .video_bitrate(4_000_000);
```

**Quality (archival)**:
```rust
let options = TranscodeOptions::new()
    .video_codec(Codec::H264)
    .preset(Preset::Slow)
    .crf(18);
```

## H.265/HEVC

50% better compression than H.264 at same quality. Best for 4K and HDR.

### Profiles

| Profile | Bit Depth | HDR | Use Case |
|---------|-----------|-----|----------|
| Main | 8-bit | No | Standard content |
| Main 10 | 10-bit | Yes | HDR, wide color |
| Main 12 | 12-bit | Yes | Professional |

```rust
use transcode::codecs::hevc::{HevcEncoder, Profile};

let encoder = HevcEncoder::new()
    .profile(Profile::Main10)
    .build()?;
```

### Recommended Settings

**4K HDR**:
```rust
let options = TranscodeOptions::new()
    .video_codec(Codec::Hevc)
    .profile("main10")
    .crf(22)
    .preset(Preset::Medium);
```

## AV1

Royalty-free, excellent compression. Best for web delivery.

### Encoder Selection

Transcode uses:
- **rav1e** for encoding (Rust-native)
- **dav1d** for decoding (fastest AV1 decoder)

```rust
use transcode::codecs::av1::{Av1Encoder, Speed};

let encoder = Av1Encoder::new()
    .speed(Speed::Six)  // 0=slowest/best, 10=fastest
    .build()?;
```

### Recommended Settings

**Web streaming**:
```rust
let options = TranscodeOptions::new()
    .video_codec(Codec::Av1)
    .crf(30)  // AV1 CRF scale differs from H.264
    .speed(6);
```

**Note**: AV1 encoding is slower than H.264/HEVC but produces smaller files.

## VP9

Google's codec, widely used in YouTube and WebM.

```rust
let options = TranscodeOptions::new()
    .video_codec(Codec::Vp9)
    .video_bitrate(3_000_000)
    .quality(Quality::Good);
```

## ProRes

Apple's professional codec. Maintains quality during editing.

| Variant | Data Rate | Use Case |
|---------|-----------|----------|
| Proxy | ~45 Mbps | Offline editing |
| LT | ~102 Mbps | Lightweight |
| 422 | ~147 Mbps | Standard |
| HQ | ~220 Mbps | High quality |
| 4444 | ~330 Mbps | Alpha channel |

```rust
use transcode::codecs::prores::{ProResEncoder, ProResProfile};

let encoder = ProResEncoder::new()
    .profile(ProResProfile::Hq)
    .build()?;
```

## Comparison

### Quality vs File Size

At 1080p, targeting similar visual quality:

| Codec | Bitrate | Relative Size |
|-------|---------|---------------|
| H.264 | 8 Mbps | 100% |
| HEVC | 4 Mbps | 50% |
| AV1 | 3 Mbps | 38% |
| VP9 | 4 Mbps | 50% |

### Encoding Speed

Relative encoding time for 1080p (H.264 = 1.0x):

| Codec | Speed |
|-------|-------|
| H.264 | 1.0x |
| HEVC | 2-3x slower |
| AV1 | 5-10x slower |
| VP9 | 2-3x slower |

## Hardware Acceleration

See [GPU Acceleration](./gpu-acceleration.md) for hardware encoding options.

## Choosing a Codec

```
Need maximum compatibility?     → H.264
Need HDR or 4K efficiency?      → HEVC
Need royalty-free for web?      → AV1 or VP9
Need editing/post-production?   → ProRes or DNxHD
```
