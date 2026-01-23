---
sidebar_position: 1
title: CLI Reference
description: Complete command-line interface reference
---

# CLI Reference

Complete reference for the Transcode command-line interface.

## Installation

```bash
# From crates.io
cargo install transcode-cli

# From source
git clone https://github.com/example/transcode
cd transcode
cargo install --path transcode-cli
```

## Basic Syntax

```bash
transcode [OPTIONS] -i <INPUT> -o <OUTPUT>
```

## Quick Examples

```bash
# Basic transcode
transcode -i input.mp4 -o output.mp4

# Set video bitrate
transcode -i input.mp4 -o output.mp4 --video-bitrate 5000

# Change codec
transcode -i input.mp4 -o output.av1 --video-codec av1

# Scale video
transcode -i input.mp4 -o output.mp4 --width 1920 --height 1080
```

## Global Options

| Option | Short | Description |
|--------|-------|-------------|
| `--help` | `-h` | Show help message |
| `--version` | `-V` | Show version |
| `--verbose` | `-v` | Increase verbosity (use multiple times) |
| `--quiet` | `-q` | Suppress output |
| `--json` | `-j` | Output JSON format |
| `--config <FILE>` | `-c` | Load config file |
| `--threads <N>` | `-t` | Number of threads (default: auto) |
| `--dry-run` | | Show what would be done |

## Input/Output Options

| Option | Short | Description |
|--------|-------|-------------|
| `--input <FILE>` | `-i` | Input file path (required) |
| `--output <FILE>` | `-o` | Output file path (required) |
| `--overwrite` | `-y` | Overwrite output without asking |
| `--no-overwrite` | `-n` | Never overwrite output |

### Multiple Inputs

```bash
# Concatenate multiple inputs
transcode -i input1.mp4 -i input2.mp4 -o combined.mp4 --concat

# Use different streams from different files
transcode -i video.mp4 -i audio.wav -o output.mp4 --map 0:v --map 1:a
```

## Video Options

| Option | Description | Default |
|--------|-------------|---------|
| `--video-codec <CODEC>` | Video codec | copy |
| `--video-bitrate <KBPS>` | Video bitrate in kbps | auto |
| `--crf <VALUE>` | Constant Rate Factor (0-51) | 23 |
| `--preset <PRESET>` | Encoding preset | medium |
| `--width <PIXELS>` | Output width | auto |
| `--height <PIXELS>` | Output height | auto |
| `--fps <RATE>` | Frame rate | auto |
| `--keyint <FRAMES>` | Keyframe interval | 250 |
| `--bframes <N>` | Max B-frames | 3 |
| `--profile <PROFILE>` | Codec profile | auto |
| `--level <LEVEL>` | Codec level | auto |
| `--pix-fmt <FORMAT>` | Pixel format | yuv420p |
| `--no-video` | Disable video | |

### Video Codecs

| Codec | Aliases | Description |
|-------|---------|-------------|
| `h264` | `avc`, `x264` | H.264/AVC |
| `h265` | `hevc`, `x265` | H.265/HEVC |
| `av1` | `libaom`, `rav1e` | AV1 |
| `vp9` | `libvpx-vp9` | VP9 |
| `prores` | `prores_ks` | Apple ProRes |
| `copy` | | Copy without re-encoding |

### Presets

| Preset | Speed | Quality |
|--------|-------|---------|
| `ultrafast` | Fastest | Lowest |
| `superfast` | | |
| `veryfast` | | |
| `faster` | | |
| `fast` | | |
| `medium` | Default | Default |
| `slow` | | |
| `slower` | | |
| `veryslow` | Slowest | Highest |

### Examples

```bash
# High quality H.264
transcode -i input.mp4 -o output.mp4 \
  --video-codec h264 \
  --crf 18 \
  --preset slow

# 4K AV1 encoding
transcode -i input.mp4 -o output.webm \
  --video-codec av1 \
  --width 3840 --height 2160 \
  --video-bitrate 15000

# Fast preview encode
transcode -i input.mp4 -o preview.mp4 \
  --video-codec h264 \
  --preset ultrafast \
  --crf 28
```

## Audio Options

| Option | Description | Default |
|--------|-------------|---------|
| `--audio-codec <CODEC>` | Audio codec | copy |
| `--audio-bitrate <KBPS>` | Audio bitrate in kbps | 128 |
| `--sample-rate <HZ>` | Sample rate | auto |
| `--channels <N>` | Number of channels | auto |
| `--no-audio` | Disable audio | |

### Audio Codecs

| Codec | Description |
|-------|-------------|
| `aac` | AAC (default for MP4) |
| `opus` | Opus (default for WebM) |
| `mp3` | MP3 |
| `flac` | FLAC (lossless) |
| `pcm` | Uncompressed PCM |
| `copy` | Copy without re-encoding |

### Examples

```bash
# High quality audio
transcode -i input.mp4 -o output.mp4 \
  --audio-codec aac \
  --audio-bitrate 320

# Extract audio only
transcode -i video.mp4 -o audio.mp3 \
  --no-video \
  --audio-codec mp3 \
  --audio-bitrate 320

# Lossless audio
transcode -i input.mp4 -o output.mkv \
  --audio-codec flac
```

## Scaling & Cropping

| Option | Description |
|--------|-------------|
| `--width <W>` | Output width (-1 for auto) |
| `--height <H>` | Output height (-1 for auto) |
| `--scale <MODE>` | Scale algorithm |
| `--crop <W:H:X:Y>` | Crop video |
| `--pad <W:H:X:Y:COLOR>` | Pad video |
| `--fit <W:H>` | Fit within dimensions (maintain aspect) |
| `--fill <W:H>` | Fill dimensions (crop to fit) |

### Scale Modes

| Mode | Description |
|------|-------------|
| `nearest` | Nearest neighbor |
| `bilinear` | Bilinear interpolation |
| `bicubic` | Bicubic interpolation |
| `lanczos` | Lanczos (best quality) |

### Examples

```bash
# Scale to 1080p (maintain aspect ratio)
transcode -i input.mp4 -o output.mp4 --height 1080 --width -1

# Scale to exact size
transcode -i input.mp4 -o output.mp4 --width 1920 --height 1080

# Crop center 720p from 1080p
transcode -i input.mp4 -o output.mp4 --crop 1280:720:320:180

# Fit within 720p box
transcode -i input.mp4 -o output.mp4 --fit 1280:720
```

## Time Options

| Option | Description |
|--------|-------------|
| `--start <TIME>` | Start time (HH:MM:SS or seconds) |
| `--end <TIME>` | End time |
| `--duration <TIME>` | Duration |
| `--to <TIME>` | Alias for --end |

### Examples

```bash
# Extract first 30 seconds
transcode -i input.mp4 -o clip.mp4 --duration 30

# Extract from 1:00 to 2:00
transcode -i input.mp4 -o clip.mp4 --start 1:00 --end 2:00

# Extract last 60 seconds
transcode -i input.mp4 -o clip.mp4 --start -60
```

## Filters

| Option | Description |
|--------|-------------|
| `--filter <FILTER>` | Add video filter |
| `--audio-filter <FILTER>` | Add audio filter |

### Video Filters

| Filter | Syntax | Description |
|--------|--------|-------------|
| `scale` | `scale=W:H` | Scale video |
| `crop` | `crop=W:H:X:Y` | Crop video |
| `fps` | `fps=RATE` | Change frame rate |
| `brightness` | `brightness=VALUE` | Adjust brightness (-1 to 1) |
| `contrast` | `contrast=VALUE` | Adjust contrast (0 to 2) |
| `saturation` | `saturation=VALUE` | Adjust saturation (0 to 2) |
| `sharpen` | `sharpen=STRENGTH` | Sharpen (0 to 1) |
| `blur` | `blur=RADIUS` | Gaussian blur |
| `denoise` | `denoise=STRENGTH` | Noise reduction |
| `deinterlace` | `deinterlace` | Remove interlacing |

### Audio Filters

| Filter | Syntax | Description |
|--------|--------|-------------|
| `volume` | `volume=VALUE` | Adjust volume (0 to 2) |
| `normalize` | `normalize` | Normalize audio level |
| `resample` | `resample=RATE` | Resample audio |

### Examples

```bash
# Chain multiple filters
transcode -i input.mp4 -o output.mp4 \
  --filter "scale=1920:1080" \
  --filter "brightness=0.1" \
  --filter "sharpen=0.3"

# Audio normalization
transcode -i input.mp4 -o output.mp4 \
  --audio-filter "normalize" \
  --audio-filter "volume=0.8"
```

## Streaming Output

### HLS

```bash
transcode -i input.mp4 --hls output/
```

| Option | Description | Default |
|--------|-------------|---------|
| `--hls <DIR>` | Output HLS to directory | |
| `--hls-time <SECS>` | Segment duration | 6 |
| `--hls-playlist <NAME>` | Playlist filename | playlist.m3u8 |
| `--hls-variant` | Add variant stream | |

```bash
# Multi-bitrate HLS
transcode -i input.mp4 --hls output/ \
  --hls-variant "1920x1080:5000k" \
  --hls-variant "1280x720:2500k" \
  --hls-variant "854x480:1000k"
```

### DASH

```bash
transcode -i input.mp4 --dash output/
```

| Option | Description | Default |
|--------|-------------|---------|
| `--dash <DIR>` | Output DASH to directory | |
| `--dash-segment <SECS>` | Segment duration | 4 |
| `--dash-manifest <NAME>` | Manifest filename | manifest.mpd |

## Quality Metrics

```bash
transcode quality <REFERENCE> <COMPRESSED>
```

| Option | Description |
|--------|-------------|
| `--metrics <LIST>` | Metrics to calculate (psnr,ssim,vmaf) |
| `--json` | Output as JSON |
| `--csv` | Output frame scores as CSV |

### Examples

```bash
# Compare quality
transcode quality original.mp4 compressed.mp4

# Specific metrics
transcode quality original.mp4 compressed.mp4 --metrics psnr,ssim

# Export frame-by-frame CSV
transcode quality original.mp4 compressed.mp4 --csv > quality.csv
```

## Media Info

```bash
transcode info <FILE>
```

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |
| `--streams` | Show stream details |

### Example Output

```
File: video.mp4
Duration: 00:02:30.50
Size: 150.5 MB
Container: MP4

Video Stream #0:
  Codec: H.264 (High Profile)
  Resolution: 1920x1080
  Frame Rate: 23.976 fps
  Bitrate: 8.5 Mbps
  Pixel Format: yuv420p

Audio Stream #0:
  Codec: AAC (LC)
  Sample Rate: 48000 Hz
  Channels: 2 (stereo)
  Bitrate: 128 kbps
```

## Batch Processing

```bash
# Process all MP4 files
transcode batch --input "*.mp4" --output-dir output/ --video-bitrate 5000

# With pattern replacement
transcode batch -i "input/*.avi" -o "output/{name}.mp4"
```

| Option | Description |
|--------|-------------|
| `--input <PATTERN>` | Input file pattern (glob) |
| `--output-dir <DIR>` | Output directory |
| `--parallel <N>` | Parallel jobs |
| `--continue` | Skip existing outputs |

## Hardware Acceleration

| Option | Description |
|--------|-------------|
| `--hwaccel <TYPE>` | Hardware acceleration type |
| `--hwaccel-device <DEV>` | Device to use |
| `--gpu` | Enable GPU processing |

### Types

| Type | Description |
|------|-------------|
| `auto` | Auto-detect |
| `cuda` | NVIDIA CUDA |
| `nvenc` | NVIDIA NVENC |
| `qsv` | Intel Quick Sync |
| `vaapi` | VA-API (Linux) |
| `videotoolbox` | macOS VideoToolbox |

### Examples

```bash
# NVIDIA hardware encoding
transcode -i input.mp4 -o output.mp4 \
  --hwaccel nvenc \
  --video-codec h264

# Intel Quick Sync
transcode -i input.mp4 -o output.mp4 \
  --hwaccel qsv \
  --video-codec h264
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TRANSCODE_THREADS` | Number of threads | CPU count |
| `TRANSCODE_LOG_LEVEL` | Log level | info |
| `TRANSCODE_CONFIG` | Config file path | |
| `TRANSCODE_TEMP_DIR` | Temporary directory | System temp |

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Input file not found |
| 4 | Output error |
| 5 | Codec error |
| 6 | User cancelled |

## Config File

```toml
# ~/.config/transcode/config.toml

[defaults]
video_codec = "h264"
video_bitrate = 5000
audio_codec = "aac"
audio_bitrate = 128
preset = "medium"

[paths]
temp_dir = "/tmp/transcode"

[performance]
threads = 8
gpu = true
```

## Shell Completion

```bash
# Bash
transcode completions bash > /etc/bash_completion.d/transcode

# Zsh
transcode completions zsh > ~/.zfunc/_transcode

# Fish
transcode completions fish > ~/.config/fish/completions/transcode.fish
```

## Next Steps

- [API Reference](/docs/reference/api) - Rust API documentation
- [Configuration](/docs/reference/configuration) - Full config options
- [Codecs Matrix](/docs/reference/codecs-matrix) - Supported codecs
