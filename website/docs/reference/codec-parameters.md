# Codec Parameters Reference

This reference documents all configurable parameters for video and audio codecs in the Transcode library.

## H.264 (AVC) Encoder

### Configuration Structure

```rust
use transcode_codecs::video::h264::{H264EncoderConfig, H264Profile, H264Level};

let config = H264EncoderConfig::new(1920, 1080)
    .with_frame_rate(30)
    .with_crf(23)
    .with_preset(EncoderPreset::Medium);
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `width` | `u32` | required | Output width in pixels |
| `height` | `u32` | required | Output height in pixels |
| `frame_rate` | `(u32, u32)` | `(30, 1)` | Frame rate as numerator/denominator |
| `profile` | `H264Profile` | `Main` | H.264 profile (Baseline, Main, High) |
| `level` | `H264Level` | `4.0` | H.264 level |
| `rate_control` | `RateControlMode` | `Crf(23)` | Rate control mode |
| `preset` | `EncoderPreset` | `Medium` | Speed/quality tradeoff |
| `gop_size` | `u32` | `250` | Keyframe interval |
| `bframes` | `u8` | `2` | Number of B-frames between I and P |
| `ref_frames` | `u8` | `3` | Number of reference frames |
| `cabac` | `bool` | `true` | Use CABAC entropy coding |

### Rate Control Modes

```rust
pub enum RateControlMode {
    Crf(u8),                        // Quality-based (0-51, lower = better)
    Cbr(u32),                       // Constant bitrate (bits/second)
    Vbr { target: u32, max: u32 },  // Variable bitrate
    Cqp(u8),                        // Constant QP
}
```

| Mode | Use Case | Typical Values |
|------|----------|----------------|
| CRF | General encoding | 18-28 (23 default) |
| CBR | Streaming | 2-8 Mbps for 1080p |
| VBR | High quality | target=5M, max=8M |
| CQP | Testing | 18-28 |

### Presets

| Preset | Quality | Speed | Use Case |
|--------|---------|-------|----------|
| `Ultrafast` | Lowest | Fastest | Real-time, low latency |
| `Superfast` | Low | Very fast | Live streaming |
| `Veryfast` | Low | Fast | Live streaming |
| `Faster` | Medium-low | Fast | Fast transcoding |
| `Fast` | Medium | Above avg | Balanced speed |
| `Medium` | Good | Average | Default, balanced |
| `Slow` | High | Below avg | Quality focus |
| `Slower` | Higher | Slow | Archival |
| `Veryslow` | Highest | Very slow | Maximum quality |
| `Placebo` | Maximum | Impractical | Testing only |

### Profiles

| Profile | Features | Use Case |
|---------|----------|----------|
| `Baseline` | No B-frames, CAVLC only | Mobile, low latency |
| `Main` | B-frames, CABAC | Broadcast, general |
| `High` | 8x8 transform, custom quant | Blu-ray, high quality |
| `High10` | 10-bit color | HDR content |
| `High422` | 4:2:2 chroma | Professional |
| `High444` | 4:4:4 chroma | Lossless workflows |

### Threading Options

```rust
let config = H264EncoderConfig::new(1920, 1080)
    .with_threads(0)              // Auto-detect
    .with_slice_count(4)          // Slices per frame
    .with_lookahead_depth(40)     // B-frame decisions
    .with_slice_parallel(true)    // Enable slice parallelism
    .with_frame_parallel(true);   // Enable frame parallelism
```

---

## AV1 Encoder

### Configuration Structure

```rust
use transcode_av1::{Av1Config, Av1Preset, RateControlMode};

let config = Av1Config::new(3840, 2160)
    .with_preset(Av1Preset::Medium)
    .with_quality(28)
    .with_threads(0);
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `width` | `u32` | required | Output width (max 8192) |
| `height` | `u32` | required | Output height (max 4320) |
| `framerate_num` | `u32` | `30` | Frame rate numerator |
| `framerate_den` | `u32` | `1` | Frame rate denominator |
| `preset` | `Av1Preset` | `Medium` | Speed preset (0-10) |
| `rate_control` | `RateControlMode` | `CQ(28)` | Rate control |
| `bit_depth` | `u8` | `8` | 8 or 10 bit |
| `tile_cols_log2` | `u8` | `0` | Tile columns (power of 2) |
| `tile_rows_log2` | `u8` | `0` | Tile rows (power of 2) |
| `keyframe_interval` | `u32` | `240` | Max keyframe distance |
| `min_keyframe_interval` | `u32` | `12` | Min keyframe distance |
| `low_latency` | `bool` | `false` | Enable low latency mode |
| `threads` | `usize` | `0` | Thread count (0 = auto) |

### Rate Control Modes

```rust
pub enum RateControlMode {
    ConstantQuality { quantizer: u8 },  // CQ mode (0-63)
    Vbr { bitrate: u64 },               // Variable bitrate
    Cbr { bitrate: u64 },               // Constant bitrate
    TwoPassFirst,                       // First pass
    TwoPassSecond { stats: Vec<u8> },   // Second pass
}
```

| Mode | Use Case | Values |
|------|----------|--------|
| `ConstantQuality` | Default, general | 20-35 (28 default) |
| `Vbr` | Streaming | 2-20 Mbps for 4K |
| `Cbr` | Live streaming | Similar to VBR |
| `TwoPass` | Maximum efficiency | Best for archival |

### Presets

| Preset | Speed | Quality | Encoding Speed |
|--------|-------|---------|----------------|
| `Placebo` | 0 | Best | Impractical |
| `VerySlow` | 2 | Excellent | Very slow |
| `Slower` | 3 | Great | Slow |
| `Slow` | 4 | Good | Below avg |
| `Medium` | 6 | Good | Average |
| `Fast` | 7 | Acceptable | Fast |
| `Faster` | 8 | Lower | Faster |
| `VeryFast` | 9 | Low | Very fast |
| `UltraFast` | 10 | Lowest | Fastest |

### HDR Configuration

```rust
let config = Av1Config::new(3840, 2160)
    .with_hdr()  // Sets: 10-bit, BT.2020, PQ transfer
    .with_bit_depth(10)
    .with_content_type(ContentType::Film);
```

### Color Properties

| Parameter | Options | Default |
|-----------|---------|---------|
| `color_primaries` | Bt709, Bt2020, Smpte431, etc. | `Bt709` |
| `transfer` | Bt709, Smpte2084 (PQ), Hlg, etc. | `Bt709` |
| `matrix` | Bt709, Bt2020Ncl, Ictcp, etc. | `Bt709` |
| `full_range` | true/false | `false` |

### Content Type Hints

```rust
pub enum ContentType {
    Unknown,    // Mixed or unknown
    Film,       // Movie content
    Animation,  // Cartoon/anime
    Screen,     // Screen capture
    Gaming,     // Game footage
}
```

---

## AAC Encoder

### Configuration Structure

```rust
use transcode_codecs::audio::aac::{AacEncoderConfig, AacProfile};

let config = AacEncoderConfig {
    profile: AacProfile::Lc,
    sample_rate: 48000,
    channels: 2,
    bitrate: 192000,
    adts: true,
    quality: 0.5,
};
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `profile` | `AacProfile` | `Lc` | AAC profile |
| `sample_rate` | `u32` | `44100` | Sample rate in Hz |
| `channels` | `u8` | `2` | Number of channels (1-8) |
| `bitrate` | `u32` | `128000` | Target bitrate (bits/sec) |
| `adts` | `bool` | `true` | Include ADTS headers |
| `quality` | `f32` | `0.5` | Quality factor (0.0-1.0) |

### Profiles

| Profile | Description | Use Case |
|---------|-------------|----------|
| `Lc` | Low Complexity | Default, most compatible |
| `He` | High Efficiency (SBR) | Low bitrate streaming |
| `HeV2` | HE-AAC v2 (SBR+PS) | Very low bitrate stereo |
| `Ld` | Low Delay | Real-time communication |
| `Eld` | Enhanced Low Delay | Conferencing |

### Recommended Bitrates

| Quality | Stereo | 5.1 Surround |
|---------|--------|--------------|
| Low | 96 kbps | 256 kbps |
| Medium | 128 kbps | 384 kbps |
| Good | 192 kbps | 512 kbps |
| High | 256 kbps | 640 kbps |
| Transparent | 320 kbps | 768 kbps |

### Sample Rates

Supported: 8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000 Hz

---

## Common Patterns

### Quality-Based Encoding

```rust
// H.264 CRF mode
let h264_config = H264EncoderConfig::new(1920, 1080)
    .with_crf(18);  // High quality

// AV1 CQ mode
let av1_config = Av1Config::new(1920, 1080)
    .with_quality(24);  // High quality
```

### Streaming Configuration

```rust
// Low latency H.264
let config = H264EncoderConfig::new(1920, 1080)
    .with_preset(EncoderPreset::Veryfast)
    .with_bitrate(4_000_000)
    .with_frame_rate(30);

// No B-frames for lower latency
config.bframes = 0;
config.gop_size = 60;  // 2-second keyframe interval
```

### High Quality Archival

```rust
// H.264 archival
let h264 = H264EncoderConfig::new(1920, 1080)
    .with_preset(EncoderPreset::Slower)
    .with_crf(18);

// AV1 archival
let av1 = Av1Config::new(1920, 1080)
    .with_preset(Av1Preset::Slow)
    .with_quality(20);
```

### 4K HDR Configuration

```rust
let config = Av1Config::new(3840, 2160)
    .with_hdr()
    .with_preset(Av1Preset::Slow)
    .with_quality(26)
    .with_tiles(2, 2)  // 4x parallelism
    .with_threads(8);
```

---

## Quality Guidelines

### CRF/CQ Values

| Value | Visual Quality | File Size | Use Case |
|-------|---------------|-----------|----------|
| 0-15 | Visually lossless | Very large | Master archives |
| 16-18 | Excellent | Large | Personal archives |
| 19-22 | Very good | Medium | General high quality |
| 23-26 | Good | Medium-small | Default, balanced |
| 27-30 | Acceptable | Small | Streaming |
| 31-35 | Low | Very small | Previews |
| 36+ | Poor | Tiny | Thumbnails |

### Bitrate Recommendations

| Resolution | Low | Medium | High | Premium |
|------------|-----|--------|------|---------|
| 720p | 2 Mbps | 4 Mbps | 6 Mbps | 8 Mbps |
| 1080p | 4 Mbps | 6 Mbps | 10 Mbps | 15 Mbps |
| 4K | 15 Mbps | 25 Mbps | 40 Mbps | 60 Mbps |
| 8K | 50 Mbps | 80 Mbps | 120 Mbps | 150 Mbps |

*For AV1, reduce values by approximately 30-40%*

## Next Steps

- [Basic Transcoding](../guides/basic-transcoding.md) - Getting started
- [GPU Acceleration](../guides/gpu-acceleration.md) - Hardware encoding
- [Quality Metrics](../guides/quality-metrics.md) - Measuring output
