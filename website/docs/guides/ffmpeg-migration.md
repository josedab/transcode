---
sidebar_position: 8
title: FFmpeg Migration
description: Practical guide for migrating from FFmpeg to Transcode
---

# FFmpeg Migration Guide

This guide provides practical recipes for migrating common FFmpeg workflows to Transcode.

## Quick Reference

### CLI Command Mapping

| Task | FFmpeg | Transcode |
|------|--------|-----------|
| Basic transcode | `ffmpeg -i in.mp4 out.mp4` | `transcode -i in.mp4 -o out.mp4` |
| Set video codec | `-c:v libx264` | `--video-codec h264` |
| Set audio codec | `-c:a aac` | `--audio-codec aac` |
| Video bitrate | `-b:v 5M` | `--video-bitrate 5000` |
| Audio bitrate | `-b:a 128k` | `--audio-bitrate 128` |
| Quality (CRF) | `-crf 23` | `--quality 23` |
| Preset | `-preset slow` | `--preset slow` |
| Resolution | `-vf scale=1920:1080` | `--width 1920 --height 1080` |
| Frame rate | `-r 30` | `--framerate 30` |
| Seek start | `-ss 00:01:00` | `--start 1:00` |
| Duration | `-t 30` | `--duration 30` |
| Overwrite | `-y` | `--overwrite` |
| No audio | `-an` | `--no-audio` |
| No video | `-vn` | `--no-video` |
| Two-pass | `-pass 1` / `-pass 2` | `--two-pass` |
| Metadata | `-metadata title="..."` | `--metadata title="..."` |

## Common Workflows

### 1. Convert to H.264 MP4

**FFmpeg:**
```bash
ffmpeg -i input.mkv -c:v libx264 -crf 23 -c:a aac -b:a 128k output.mp4
```

**Transcode:**
```bash
transcode -i input.mkv -o output.mp4 --video-codec h264 --quality 23 --audio-codec aac --audio-bitrate 128
```

### 2. Convert to H.265/HEVC

**FFmpeg:**
```bash
ffmpeg -i input.mp4 -c:v libx265 -crf 28 -preset slow -c:a copy output.mp4
```

**Transcode:**
```bash
transcode -i input.mp4 -o output.mp4 --video-codec h265 --quality 28 --preset slow --audio-codec copy
```

### 3. Scale Video

**FFmpeg:**
```bash
ffmpeg -i input.mp4 -vf scale=1280:720 -c:a copy output.mp4
```

**Transcode:**
```bash
transcode -i input.mp4 -o output.mp4 --width 1280 --height 720 --audio-codec copy
```

### 4. Extract Clip

**FFmpeg:**
```bash
ffmpeg -ss 00:01:30 -i input.mp4 -t 30 -c copy output.mp4
```

**Transcode:**
```bash
transcode -i input.mp4 -o output.mp4 --start 1:30 --duration 30 --video-codec copy --audio-codec copy
```

### 5. Extract Audio Only

**FFmpeg:**
```bash
ffmpeg -i video.mp4 -vn -c:a libmp3lame -q:a 2 audio.mp3
```

**Transcode:**
```bash
transcode -i video.mp4 -o audio.mp3 --no-video --audio-codec mp3 --audio-quality 2
```

### 6. Add Audio to Video

**FFmpeg:**
```bash
ffmpeg -i video.mp4 -i audio.mp3 -c:v copy -c:a aac -shortest output.mp4
```

**Transcode:**
```bash
transcode -i video.mp4 --audio-input audio.mp3 -o output.mp4 --video-codec copy --audio-codec aac
```

### 7. Create GIF

**FFmpeg:**
```bash
ffmpeg -i input.mp4 -vf "fps=10,scale=320:-1" -loop 0 output.gif
```

**Transcode:**
```bash
transcode -i input.mp4 -o output.gif --framerate 10 --width 320 --format gif
```

### 8. Two-Pass Encoding

**FFmpeg:**
```bash
ffmpeg -i input.mp4 -c:v libx264 -b:v 2M -pass 1 -an -f null /dev/null
ffmpeg -i input.mp4 -c:v libx264 -b:v 2M -pass 2 -c:a aac output.mp4
```

**Transcode:**
```bash
transcode -i input.mp4 -o output.mp4 --video-codec h264 --video-bitrate 2000 --two-pass --audio-codec aac
```

### 9. Batch Processing

**FFmpeg (shell script):**
```bash
for f in *.avi; do
    ffmpeg -i "$f" -c:v libx264 -crf 23 "${f%.avi}.mp4"
done
```

**Transcode:**
```bash
transcode batch --input "*.avi" --output "{name}.mp4" --video-codec h264 --quality 23
```

Or in Rust:
```rust
use transcode::{Transcoder, TranscodeOptions};
use std::fs;

for entry in fs::read_dir(".")? {
    let path = entry?.path();
    if path.extension() == Some("avi".as_ref()) {
        let output = path.with_extension("mp4");
        TranscodeOptions::new()
            .input(&path)
            .output(&output)
            .video_codec("h264")
            .quality(23)
            .run()?;
    }
}
```

### 10. HLS Streaming

**FFmpeg:**
```bash
ffmpeg -i input.mp4 \
    -c:v libx264 -b:v 2M -g 60 -keyint_min 60 \
    -hls_time 6 -hls_list_size 0 \
    -hls_segment_filename 'segment_%03d.ts' \
    playlist.m3u8
```

**Transcode:**
```bash
transcode -i input.mp4 -o output/ --format hls \
    --video-codec h264 --video-bitrate 2000 \
    --hls-segment-duration 6 --hls-playlist playlist.m3u8
```

Or programmatically:
```rust
use transcode_streaming::{HlsConfig, HlsWriter};

let config = HlsConfig::new("output/")
    .with_segment_duration(6.0)
    .with_video_bitrate(2_000_000);

let mut writer = HlsWriter::new(config)?;
// ... write frames
writer.finalize()?;
```

## API Migration

### Opening Files

**FFmpeg (C):**
```c
AVFormatContext *fmt_ctx = NULL;
int ret = avformat_open_input(&fmt_ctx, "input.mp4", NULL, NULL);
if (ret < 0) {
    // Handle error
}
ret = avformat_find_stream_info(fmt_ctx, NULL);
```

**Transcode (Rust):**
```rust
use transcode_containers::mp4::Mp4Demuxer;

let demuxer = Mp4Demuxer::open("input.mp4")?;
println!("Streams: {:?}", demuxer.streams());
```

### Decoding Frames

**FFmpeg (C):**
```c
AVPacket *pkt = av_packet_alloc();
AVFrame *frame = av_frame_alloc();

while (av_read_frame(fmt_ctx, pkt) >= 0) {
    if (pkt->stream_index == video_stream_idx) {
        avcodec_send_packet(dec_ctx, pkt);
        while (avcodec_receive_frame(dec_ctx, frame) >= 0) {
            // Process frame
        }
    }
    av_packet_unref(pkt);
}
```

**Transcode (Rust):**
```rust
use transcode_codecs::video::h264::H264Decoder;

let mut demuxer = Mp4Demuxer::open("input.mp4")?;
let mut decoder = H264Decoder::new()?;

while let Some(packet) = demuxer.read_packet()? {
    if packet.stream_index == video_stream_idx {
        for frame in decoder.decode(&packet)? {
            // Process frame
        }
    }
}
```

### Encoding Frames

**FFmpeg (C):**
```c
avcodec_send_frame(enc_ctx, frame);
while (avcodec_receive_packet(enc_ctx, pkt) >= 0) {
    av_interleaved_write_frame(out_fmt_ctx, pkt);
    av_packet_unref(pkt);
}
```

**Transcode (Rust):**
```rust
use transcode_codecs::video::h264::H264Encoder;

let config = EncoderConfig::default()
    .with_bitrate(5_000_000)
    .with_preset(Preset::Medium);

let mut encoder = H264Encoder::new(config)?;

for packet in encoder.encode(&frame)? {
    muxer.write_packet(&packet)?;
}
```

### Progress Callbacks

**FFmpeg (C):**
```c
// FFmpeg doesn't have built-in progress callbacks
// You have to track packet timestamps manually
int64_t total_duration = fmt_ctx->duration;
int64_t current_pts = pkt->pts;
double progress = (double)current_pts / total_duration;
```

**Transcode (Rust):**
```rust
use transcode::{Transcoder, TranscodeOptions, Progress};

let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.mp4")
    .on_progress(|p: &Progress| {
        println!("{}% ({}/{} frames, {} fps)",
            p.percent, p.frame, p.total_frames, p.fps);
    });

Transcoder::new(options)?.run()?;
```

## C API for FFmpeg Users

If you're used to FFmpeg's C API, Transcode provides a C API with similar patterns:

```c
#include <transcode.h>

TranscodeContext *ctx = NULL;
TranscodeError err;

// Similar to avformat_open_input
err = transcode_open_input("input.mp4", &ctx);
if (err != TRANSCODE_SUCCESS) {
    fprintf(stderr, "Error: %s\n", transcode_error_string(err));
    return 1;
}

// Similar to av_read_frame
TranscodePacket *pkt = transcode_packet_alloc();
TranscodeFrame *frame = transcode_frame_alloc();

while ((err = transcode_read_packet(ctx, pkt)) == TRANSCODE_SUCCESS) {
    // Similar to avcodec_decode_video2
    if (transcode_decode_packet(ctx, pkt, frame) == TRANSCODE_SUCCESS) {
        // Process frame
    }
}

// Cleanup (similar to av_free, avformat_close_input)
transcode_frame_free(frame);
transcode_packet_free(pkt);
transcode_close(ctx);
```

## Filter Equivalents

| FFmpeg Filter | Transcode |
|---------------|-----------|
| `scale=W:H` | `ScaleFilter::new(W, H)` |
| `crop=W:H:X:Y` | `CropFilter::new(W, H, X, Y)` |
| `transpose=1` | `RotateFilter::new(90)` |
| `hflip` | `FlipFilter::horizontal()` |
| `vflip` | `FlipFilter::vertical()` |
| `fps=30` | `FramerateFilter::new(30)` |
| `eq=brightness=0.1` | `BrightnessFilter::new(0.1)` |
| `volume=2.0` | `VolumeFilter::new(2.0)` |
| `adelay=1000` | `AudioDelayFilter::new_ms(1000)` |

### Applying Filters

**FFmpeg:**
```bash
ffmpeg -i input.mp4 -vf "scale=1280:720,transpose=1" output.mp4
```

**Transcode CLI:**
```bash
transcode -i input.mp4 -o output.mp4 --filter "scale=1280:720" --filter "rotate=90"
```

**Transcode Rust:**
```rust
use transcode::{ScaleFilter, RotateFilter, FilterChain};

let mut chain = FilterChain::new();
chain.add(ScaleFilter::new(1280, 720));
chain.add(RotateFilter::new(90));

// Apply to frame
let processed = chain.apply(&frame)?;
```

## Common Pitfalls

### 1. Codec Names

FFmpeg uses library names (`libx264`), Transcode uses codec names (`h264`):

```bash
# FFmpeg
ffmpeg -c:v libx264 ...

# Transcode
transcode --video-codec h264 ...
```

### 2. Bitrate Units

FFmpeg accepts `5M`, Transcode uses kbps:

```bash
# FFmpeg: 5 Mbps
ffmpeg -b:v 5M ...

# Transcode: 5000 kbps = 5 Mbps
transcode --video-bitrate 5000 ...
```

### 3. Copy Codec

Both use `copy`, but syntax differs:

```bash
# FFmpeg
ffmpeg -c:v copy -c:a copy ...

# Transcode
transcode --video-codec copy --audio-codec copy ...
```

### 4. Null Output

```bash
# FFmpeg
ffmpeg -f null /dev/null

# Transcode
transcode --format null --output /dev/null
```

## Getting Help

- List available codecs: `transcode codecs`
- List presets: `transcode presets`
- File info: `transcode info file.mp4`
- System check: `transcode doctor`

## Next Steps

- [CLI Reference](/docs/reference/cli) - Full command documentation
- [API Reference](/docs/reference/api) - Rust API documentation
- [Benchmarks](/docs/reference/benchmarks) - Performance comparison
- [Transcode vs FFmpeg](/docs/comparison/vs-ffmpeg) - Detailed comparison
