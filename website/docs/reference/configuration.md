---
sidebar_position: 4
title: Configuration
description: Complete configuration reference
---

# Configuration Reference

Complete reference for Transcode configuration options.

## Configuration File

Transcode reads configuration from TOML files in the following order:

1. `/etc/transcode/config.toml` (system-wide)
2. `~/.config/transcode/config.toml` (user)
3. `./transcode.toml` (project)
4. `TRANSCODE_CONFIG` environment variable

Later files override earlier ones.

## Full Configuration Example

```toml
# transcode.toml

[general]
threads = 0              # 0 = auto-detect
log_level = "info"       # trace, debug, info, warn, error
temp_dir = "/tmp/transcode"

[defaults]
# Default video settings
video_codec = "h264"
video_bitrate = 5000000
video_preset = "medium"
video_crf = 23
video_profile = "high"
video_level = "4.1"

# Default audio settings
audio_codec = "aac"
audio_bitrate = 128000
audio_sample_rate = 48000
audio_channels = 2

[video]
# H.264 specific
[video.h264]
preset = "medium"
profile = "high"
level = "4.1"
tune = "none"            # film, animation, grain, stillimage, fastdecode, zerolatency
rc_mode = "crf"          # crf, cbr, vbr
crf = 23
keyint = 250
bframes = 3
ref_frames = 4

[video.h265]
preset = "medium"
profile = "main"
crf = 28
keyint = 250

[video.av1]
speed = 6                # 0-10
crf = 30
tiles = 4
threads = 0

[video.vp9]
quality = "good"         # best, good, realtime
crf = 31
threads = 4

[audio]
[audio.aac]
profile = "lc"           # lc, he, he_v2
bitrate = 128000
sample_rate = 48000
channels = 2

[audio.opus]
bitrate = 128000
application = "audio"    # audio, voip, lowdelay
frame_size = 20          # ms

[audio.mp3]
bitrate = 320000
mode = "vbr"             # cbr, vbr
quality = 2              # 0-9 for VBR

[audio.flac]
compression = 5          # 0-8

[filters]
# Default filter settings
scale_mode = "lanczos"   # nearest, bilinear, bicubic, lanczos
deinterlace = false
denoise_strength = 0.0

[output]
# Output defaults
faststart = true         # Move moov atom for streaming
overwrite = "ask"        # ask, yes, no

[hls]
segment_duration = 6
playlist_type = "vod"    # vod, event, live
master_playlist = "master.m3u8"

[dash]
segment_duration = 4
manifest = "manifest.mpd"

[gpu]
enabled = false
backend = "auto"         # auto, vulkan, metal, dx12
device = 0               # GPU device index

[distributed]
coordinator_url = ""
worker_id = ""
max_concurrent_tasks = 2

[quality]
default_metrics = ["psnr", "ssim"]
vmaf_model = "vmaf_v0.6.1"
sample_interval = 1      # Every Nth frame
```

## Configuration Sections

### [general]

General application settings.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `threads` | int | 0 | Worker threads (0 = auto) |
| `log_level` | string | "info" | Log verbosity level |
| `temp_dir` | string | system | Temporary file directory |

```toml
[general]
threads = 8
log_level = "debug"
temp_dir = "/tmp/transcode"
```

### [defaults]

Default encoding parameters applied when not specified.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `video_codec` | string | "h264" | Default video codec |
| `video_bitrate` | int | 5000000 | Default bitrate (bps) |
| `video_preset` | string | "medium" | Encoding speed preset |
| `video_crf` | int | 23 | Constant Rate Factor |
| `audio_codec` | string | "aac" | Default audio codec |
| `audio_bitrate` | int | 128000 | Default audio bitrate |

### [video.h264]

H.264/AVC encoder settings.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `preset` | string | "medium" | Encoding preset |
| `profile` | string | "high" | H.264 profile |
| `level` | string | "4.1" | H.264 level |
| `tune` | string | "none" | Tuning preset |
| `rc_mode` | string | "crf" | Rate control mode |
| `crf` | int | 23 | Quality (0-51, lower = better) |
| `keyint` | int | 250 | Keyframe interval |
| `bframes` | int | 3 | Max B-frames |
| `ref_frames` | int | 4 | Reference frames |
| `cabac` | bool | true | Use CABAC entropy coding |
| `deblock` | bool | true | Deblocking filter |

**Presets (slowest to fastest):**
- `veryslow` - Best quality, slowest
- `slower`
- `slow`
- `medium` - Default balance
- `fast`
- `faster`
- `veryfast`
- `superfast`
- `ultrafast` - Fastest, lowest quality

**Profiles:**
- `baseline` - Most compatible, no B-frames
- `main` - Broadcast standard
- `high` - Best quality, most features

**Tune options:**
- `film` - High-quality film content
- `animation` - Animated content
- `grain` - Preserve film grain
- `stillimage` - Still image slideshow
- `fastdecode` - Low-latency decoding
- `zerolatency` - Real-time streaming

### [video.h265]

H.265/HEVC encoder settings.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `preset` | string | "medium" | Encoding preset |
| `profile` | string | "main" | HEVC profile |
| `crf` | int | 28 | Quality (0-51) |
| `keyint` | int | 250 | Keyframe interval |

**Profiles:**
- `main` - 8-bit 4:2:0
- `main10` - 10-bit 4:2:0
- `main12` - 12-bit support
- `main-intra` - Intra-only

### [video.av1]

AV1 encoder settings (rav1e).

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `speed` | int | 6 | Speed (0-10, higher = faster) |
| `crf` | int | 30 | Quality (0-63) |
| `tiles` | int | 4 | Tile columns for parallelism |
| `threads` | int | 0 | Encoding threads (0 = auto) |
| `keyint` | int | 240 | Keyframe interval |

### [video.vp9]

VP9 encoder settings.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `quality` | string | "good" | Quality mode |
| `crf` | int | 31 | Quality (0-63) |
| `speed` | int | 1 | CPU speed (0-8) |
| `threads` | int | 4 | Encoding threads |

### [audio.aac]

AAC encoder settings.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `profile` | string | "lc" | AAC profile |
| `bitrate` | int | 128000 | Bitrate in bps |
| `sample_rate` | int | 48000 | Sample rate in Hz |
| `channels` | int | 2 | Number of channels |

**Profiles:**
- `lc` - Low Complexity (most common)
- `he` - High Efficiency (HE-AAC)
- `he_v2` - High Efficiency v2

### [audio.opus]

Opus encoder settings.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `bitrate` | int | 128000 | Bitrate in bps |
| `application` | string | "audio" | Application type |
| `frame_size` | int | 20 | Frame size in ms |
| `complexity` | int | 10 | Encoder complexity (0-10) |

**Applications:**
- `audio` - Music and mixed content
- `voip` - Speech optimization
- `lowdelay` - Low latency

### [audio.mp3]

MP3 encoder settings.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `bitrate` | int | 320000 | Bitrate in bps |
| `mode` | string | "vbr" | Encoding mode |
| `quality` | int | 2 | VBR quality (0-9) |

### [audio.flac]

FLAC encoder settings.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `compression` | int | 5 | Compression level (0-8) |

### [filters]

Default video filter settings.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `scale_mode` | string | "lanczos" | Scaling algorithm |
| `deinterlace` | bool | false | Auto deinterlace |
| `denoise_strength` | float | 0.0 | Denoise amount (0-1) |

**Scale modes:**
- `nearest` - Nearest neighbor (fast, blocky)
- `bilinear` - Bilinear interpolation
- `bicubic` - Bicubic interpolation
- `lanczos` - Lanczos (best quality)

### [output]

Output file settings.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `faststart` | bool | true | Move moov atom for web |
| `overwrite` | string | "ask" | Overwrite behavior |
| `container` | string | "auto" | Output container format |

### [hls]

HLS streaming output settings.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `segment_duration` | int | 6 | Segment length in seconds |
| `playlist_type` | string | "vod" | Playlist type |
| `master_playlist` | string | "master.m3u8" | Master playlist name |
| `segment_filename` | string | "segment_%03d.ts" | Segment filename pattern |

**Playlist types:**
- `vod` - Video on demand
- `event` - Growing playlist
- `live` - Sliding window

### [dash]

DASH streaming output settings.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `segment_duration` | int | 4 | Segment length in seconds |
| `manifest` | string | "manifest.mpd" | Manifest filename |
| `segment_template` | string | "segment_$Number$.m4s" | Segment pattern |

### [gpu]

GPU acceleration settings.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | false | Enable GPU processing |
| `backend` | string | "auto" | GPU backend |
| `device` | int | 0 | GPU device index |
| `memory_limit` | int | 0 | Max GPU memory (0 = auto) |

**Backends:**
- `auto` - Auto-detect best
- `vulkan` - Vulkan (cross-platform)
- `metal` - Metal (macOS/iOS)
- `dx12` - DirectX 12 (Windows)

### [distributed]

Distributed processing settings.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `coordinator_url` | string | "" | Coordinator address |
| `worker_id` | string | "" | Worker identifier |
| `max_concurrent_tasks` | int | 2 | Max parallel tasks |

### [quality]

Quality metrics settings.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `default_metrics` | array | ["psnr", "ssim"] | Metrics to calculate |
| `vmaf_model` | string | "vmaf_v0.6.1" | VMAF model version |
| `sample_interval` | int | 1 | Frame sampling interval |

## Environment Variables

Environment variables override config file settings.

| Variable | Description | Example |
|----------|-------------|---------|
| `TRANSCODE_CONFIG` | Config file path | `/path/to/config.toml` |
| `TRANSCODE_THREADS` | Thread count | `8` |
| `TRANSCODE_LOG_LEVEL` | Log level | `debug` |
| `TRANSCODE_TEMP_DIR` | Temp directory | `/tmp/transcode` |
| `TRANSCODE_GPU` | Enable GPU | `true` |
| `COORDINATOR_URL` | Coordinator URL | `http://localhost:8080` |
| `WORKER_ID` | Worker ID | `worker-1` |

## Programmatic Configuration

```rust
use transcode::{Config, ConfigBuilder};

// Load from file
let config = Config::from_file("transcode.toml")?;

// Build programmatically
let config = ConfigBuilder::new()
    .threads(8)
    .log_level(LogLevel::Debug)
    .video_codec(VideoCodec::H264)
    .video_bitrate(5_000_000)
    .audio_codec(AudioCodec::Aac)
    .audio_bitrate(128_000)
    .build()?;

// Use with transcoder
let transcoder = Transcoder::with_config(options, config)?;
```

## Profiles

Create reusable encoding profiles:

```toml
# profiles.toml

[profile.web]
video_codec = "h264"
video_profile = "main"
video_level = "3.1"
video_bitrate = 2500000
audio_codec = "aac"
audio_bitrate = 128000

[profile.archive]
video_codec = "h265"
video_crf = 18
video_preset = "slow"
audio_codec = "flac"

[profile.mobile]
video_codec = "h264"
video_profile = "baseline"
video_bitrate = 1000000
width = 1280
height = 720
audio_codec = "aac"
audio_bitrate = 96000
```

Use profiles:

```bash
transcode -i input.mp4 -o output.mp4 --profile web
```

```rust
let options = TranscodeOptions::from_profile("web")?
    .input("input.mp4")
    .output("output.mp4");
```

## Next Steps

- [CLI Reference](/docs/reference/cli) - Command-line options
- [API Reference](/docs/reference/api) - Rust API documentation
- [Codecs Matrix](/docs/reference/codecs-matrix) - Supported codecs
