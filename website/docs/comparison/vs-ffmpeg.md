---
sidebar_position: 1
title: Transcode vs FFmpeg
description: Detailed comparison between Transcode and FFmpeg
---

# Transcode vs FFmpeg

FFmpeg has been the de facto standard for video processing for over two decades. This guide helps you understand when Transcode is a better fit and how to migrate.

## At a Glance

| Aspect | Transcode | FFmpeg |
|--------|-----------|--------|
| **Language** | Rust | C |
| **First Release** | 2024 | 2000 |
| **Memory Safety** | Guaranteed by compiler | Manual management |
| **API Style** | Builder pattern, type-safe | Flag-based, strings |
| **Learning Curve** | Moderate | Steep |
| **Binary Size** | ~5-15 MB | ~50-100 MB |
| **System Dependencies** | Minimal (optional) | Many required |
| **WebAssembly** | First-class support | Limited/complex |
| **CVE History** | None (new project) | 100+ over time |

## When to Choose Transcode

### 1. Building Rust Applications

Transcode integrates natively with Rust projects:

```rust
use transcode::{Transcoder, TranscodeOptions};

let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.mp4")
    .video_bitrate(5_000_000);

Transcoder::new(options)?.run()?;
```

**FFmpeg equivalent requires FFI:**

```rust
// Requires unsafe blocks, manual memory management,
// and complex bindings like ffmpeg-sys or ffmpeg-next
unsafe {
    let ctx = avformat_open_input(...);
    // 200+ lines of error-prone code
}
```

### 2. Security-Critical Applications

Processing untrusted media files is risky with C libraries. Transcode's Rust implementation eliminates:

- Buffer overflow vulnerabilities
- Use-after-free bugs
- Integer overflow exploits
- Null pointer dereferences

FFmpeg has had [100+ CVEs](https://www.cvedetails.com/vulnerability-list/vendor_id-3611/Ffmpeg.html) over its lifetime, many in codec implementations.

### 3. WebAssembly Deployment

Transcode compiles to WASM with full functionality:

```javascript
import init, { WasmTranscoder } from 'transcode-wasm';

await init();
const transcoder = new WasmTranscoder();
const output = transcoder.process_frame(inputFrame);
```

FFmpeg WASM builds exist but are:
- Much larger (~30MB vs ~5MB)
- Missing many features
- Require complex threading workarounds

### 4. Minimal Dependencies

Transcode core has zero required system dependencies:

```bash
# Transcode
cargo add transcode
cargo build --release

# FFmpeg (requires system libs)
apt install libavcodec-dev libavformat-dev libavutil-dev \
    libswscale-dev libswresample-dev ...
```

### 5. Embedded in Other Languages

Transcode provides idiomatic bindings:

**Python:**
```python
import transcode_py
stats = transcode_py.transcode('input.mp4', 'output.mp4')
```

**FFmpeg Python requires subprocess:**
```python
import subprocess
subprocess.run(['ffmpeg', '-i', 'input.mp4', 'output.mp4'])
# No programmatic access to frames, progress, or errors
```

### 6. Modern Feature Set

Transcode includes features FFmpeg doesn't have natively:

| Feature | Transcode | FFmpeg |
|---------|-----------|--------|
| AI upscaling | Built-in | External (Real-ESRGAN, etc.) |
| Neural denoising | Built-in | External |
| Content intelligence | Built-in | Not available |
| Distributed processing | Built-in | External (requires custom setup) |
| Quality metrics (VMAF) | Built-in | Filter (slow) |

## When to Choose FFmpeg

### 1. Maximum Codec Coverage

FFmpeg supports nearly every codec ever created:

- Obscure formats (RealVideo, Cinepak, Sorenson)
- Proprietary codecs (many via reverse engineering)
- Hardware-specific formats

If you need a codec Transcode doesn't support, FFmpeg probably does.

### 2. Shell Scripting / CLI Pipelines

For quick one-off tasks, FFmpeg's CLI is hard to beat:

```bash
# FFmpeg one-liner
ffmpeg -i input.mp4 -c:v libx264 -crf 23 output.mp4

# Transcode equivalent
transcode -i input.mp4 -o output.mp4 --video-codec h264 --crf 23
```

Both work, but FFmpeg has more online examples and Stack Overflow answers.

### 3. Existing Infrastructure

If you have:
- FFmpeg pipelines in production
- Scripts depending on FFmpeg output format
- Teams trained on FFmpeg

Migration may not be worth the effort for stable, working systems.

### 4. Hardware Device Capture

FFmpeg excels at capturing from hardware:

```bash
# Capture from webcam
ffmpeg -f v4l2 -i /dev/video0 output.mp4

# Capture screen
ffmpeg -f x11grab -i :0.0 output.mp4
```

Transcode focuses on file-to-file transcoding (live ingest is in development).

### 5. Complex Filter Graphs

FFmpeg's filtergraph syntax handles complex scenarios:

```bash
ffmpeg -i input.mp4 \
  -filter_complex "[0:v]split=2[v1][v2];[v1]crop=iw/2:ih:0:0[left];[v2]crop=iw/2:ih:iw/2:0[right]" \
  -map "[left]" left.mp4 -map "[right]" right.mp4
```

Transcode supports common filters but not the full filtergraph complexity.

## Migration Guide

### CLI Translation

| FFmpeg | Transcode |
|--------|-----------|
| `-i input.mp4` | `-i input.mp4` |
| `-c:v libx264` | `--video-codec h264` |
| `-c:a aac` | `--audio-codec aac` |
| `-b:v 5M` | `--video-bitrate 5000` |
| `-crf 23` | `--crf 23` |
| `-preset slow` | `--preset slow` |
| `-vf scale=1920:1080` | `--width 1920 --height 1080` |
| `-ss 00:01:00` | `--start 1:00` |
| `-t 30` | `--duration 30` |
| `-y` | `--overwrite` |
| `-an` | `--no-audio` |
| `-vn` | `--no-video` |

### API Translation

**FFmpeg (C):**
```c
AVFormatContext *fmt_ctx = NULL;
avformat_open_input(&fmt_ctx, "input.mp4", NULL, NULL);
avformat_find_stream_info(fmt_ctx, NULL);

int video_stream = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
AVCodecContext *dec_ctx = avcodec_alloc_context3(NULL);
avcodec_parameters_to_context(dec_ctx, fmt_ctx->streams[video_stream]->codecpar);
avcodec_open2(dec_ctx, avcodec_find_decoder(dec_ctx->codec_id), NULL);

AVPacket *pkt = av_packet_alloc();
AVFrame *frame = av_frame_alloc();

while (av_read_frame(fmt_ctx, pkt) >= 0) {
    if (pkt->stream_index == video_stream) {
        avcodec_send_packet(dec_ctx, pkt);
        while (avcodec_receive_frame(dec_ctx, frame) == 0) {
            // Process frame
        }
    }
    av_packet_unref(pkt);
}

av_frame_free(&frame);
av_packet_free(&pkt);
avcodec_free_context(&dec_ctx);
avformat_close_input(&fmt_ctx);
```

**Transcode (Rust):**
```rust
use transcode::{Transcoder, TranscodeOptions};

let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.mp4");

let mut transcoder = Transcoder::new(options)?;

// With progress callback
transcoder.on_progress(|progress| {
    println!("Frame {}/{}", progress.frame, progress.total_frames);
});

transcoder.run()?;
```

### Frame-by-Frame Processing

**FFmpeg:**
```c
while (av_read_frame(fmt_ctx, pkt) >= 0) {
    avcodec_send_packet(dec_ctx, pkt);
    while (avcodec_receive_frame(dec_ctx, frame) == 0) {
        process_frame(frame);
    }
}
```

**Transcode:**
```rust
use transcode_codecs::video::h264::H264Decoder;
use transcode_containers::mp4::Mp4Demuxer;

let mut demuxer = Mp4Demuxer::open("input.mp4")?;
let mut decoder = H264Decoder::new()?;

while let Some(packet) = demuxer.read_packet()? {
    if let Some(frame) = decoder.decode(&packet)? {
        process_frame(&frame);
    }
}
```

## Performance Comparison

See our [benchmarks page](/docs/reference/benchmarks) for detailed numbers. Summary:

| Operation | Transcode | FFmpeg | Notes |
|-----------|-----------|--------|-------|
| H.264 encode | 1.08-1.19x faster | baseline | SIMD optimized |
| H.264 decode | 1.05x faster | baseline | Similar performance |
| Memory usage | 2-3x lower | baseline | Frame pooling |
| Startup time | 3-4x faster | baseline | No plugin loading |
| Binary size | ~10x smaller | baseline | Modular design |

## Compatibility Layer

For gradual migration, Transcode provides an FFmpeg compatibility layer:

```rust
use transcode_compat::ffmpeg;

// Use FFmpeg-style options
let result = ffmpeg::run(&[
    "-i", "input.mp4",
    "-c:v", "libx264",
    "-crf", "23",
    "output.mp4"
])?;
```

This translates FFmpeg arguments to Transcode operations, making migration easier.

## Common Questions

### Can I use both together?

Yes. Many projects use Transcode for core transcoding and shell out to FFmpeg for edge cases:

```rust
use std::process::Command;

// Use Transcode for main workflow
transcode::transcode("input.mp4", "output.mp4")?;

// Fall back to FFmpeg for unsupported formats
Command::new("ffmpeg")
    .args(["-i", "rare_format.rm", "-c:v", "h264", "output.mp4"])
    .status()?;
```

### Is Transcode a drop-in replacement?

No. While the CLI is similar, the APIs are different. Migration requires code changes but usually results in simpler, more maintainable code.

### Will Transcode support all FFmpeg codecs?

We prioritize widely-used codecs. Obscure formats may never be supported directly, but the FFmpeg compatibility layer can help.

### What about libav forks?

Transcode doesn't use libav/FFmpeg code. It's a clean-room implementation, avoiding the libav/FFmpeg fork drama entirely.

---

Ready to try Transcode? See the [Quick Start](/docs/getting-started/quick-start) guide.
