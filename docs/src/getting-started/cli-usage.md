# CLI Usage

The `transcode` command-line tool provides a powerful interface for video and audio transcoding.

## Basic Syntax

```bash
transcode [OPTIONS] -i <INPUT> -o <OUTPUT>
```

## Options Reference

### Input/Output

| Option | Description |
|--------|-------------|
| `-i, --input <FILE>` | Input file path (required) |
| `-o, --output <FILE>` | Output file path (required) |
| `-y, --overwrite` | Overwrite output file if it exists |
| `--no-video` | Disable video output |
| `--no-audio` | Disable audio output |

### Video Options

| Option | Description |
|--------|-------------|
| `--video-codec <CODEC>` | Video codec: h264, hevc, av1, vp9, copy |
| `--video-bitrate <BPS>` | Video bitrate in bits/second |
| `--crf <VALUE>` | Constant Rate Factor (0-51, lower = better) |
| `--preset <PRESET>` | Encoding preset: ultrafast, fast, medium, slow |
| `--width <PIXELS>` | Output width |
| `--height <PIXELS>` | Output height |
| `--fps <RATE>` | Output frame rate |

### Audio Options

| Option | Description |
|--------|-------------|
| `--audio-codec <CODEC>` | Audio codec: aac, opus, flac, mp3, copy |
| `--audio-bitrate <BPS>` | Audio bitrate in bits/second |
| `--sample-rate <HZ>` | Audio sample rate |
| `--channels <NUM>` | Number of audio channels |

### Advanced Options

| Option | Description |
|--------|-------------|
| `--two-pass` | Enable two-pass encoding |
| `--gpu` | Enable GPU acceleration |
| `--threads <NUM>` | Number of encoding threads |
| `--start <TIME>` | Start time (e.g., "00:01:30" or "90") |
| `--duration <TIME>` | Duration to transcode |

### Output Options

| Option | Description |
|--------|-------------|
| `--progress` | Show progress bar |
| `-v, --verbose` | Verbose output |
| `-q, --quiet` | Suppress non-error output |
| `--json` | Output statistics as JSON |

## Examples

### Basic Transcoding

```bash
# Convert MKV to MP4
transcode -i movie.mkv -o movie.mp4

# Re-encode with specific bitrate
transcode -i input.mp4 -o output.mp4 --video-bitrate 5000000
```

### Quality Control

```bash
# Use CRF for quality-based encoding
transcode -i input.mp4 -o output.mp4 --crf 23

# High quality (slower)
transcode -i input.mp4 -o output.mp4 --crf 18 --preset slow

# Fast encoding (lower quality)
transcode -i input.mp4 -o output.mp4 --preset ultrafast
```

### Resolution and Format

```bash
# Scale to 1080p
transcode -i input.mp4 -o output.mp4 --width 1920 --height 1080

# Convert to AV1
transcode -i input.mp4 -o output.mp4 --video-codec av1

# Extract audio only
transcode -i video.mp4 -o audio.aac --no-video
```

### Time Selection

```bash
# Start at 1 minute, transcode 30 seconds
transcode -i input.mp4 -o clip.mp4 --start 00:01:00 --duration 30

# Start at 90 seconds
transcode -i input.mp4 -o clip.mp4 --start 90
```

### Streaming Output

```bash
# Generate HLS
transcode -i input.mp4 -o output/ --format hls --segment-duration 6

# Generate DASH
transcode -i input.mp4 -o output/ --format dash
```

### GPU Acceleration

```bash
# Use GPU for encoding
transcode -i input.mp4 -o output.mp4 --gpu

# Specific hardware encoder
transcode -i input.mp4 -o output.mp4 --video-codec h264_nvenc
```

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Input file not found |
| 4 | Output file exists (use -y to overwrite) |
| 5 | Unsupported codec or format |
| 6 | Encoding error |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `TRANSCODE_THREADS` | Default thread count |
| `TRANSCODE_LOG` | Log level (error, warn, info, debug, trace) |
| `TRANSCODE_GPU` | Default GPU device |

## Configuration File

Create `~/.config/transcode/config.toml`:

```toml
[defaults]
video_codec = "h264"
audio_codec = "aac"
preset = "medium"

[presets.web]
video_bitrate = 2_500_000
audio_bitrate = 128_000
width = 1280
height = 720

[presets.archive]
video_codec = "hevc"
crf = 18
preset = "slow"
```

Use presets with:

```bash
transcode -i input.mp4 -o output.mp4 --use-preset web
```
