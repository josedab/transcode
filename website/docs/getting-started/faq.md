---
sidebar_position: 4
title: FAQ & Troubleshooting
description: Frequently asked questions and common issues
---

# FAQ & Troubleshooting

## Frequently Asked Questions

### General

#### Is Transcode a replacement for FFmpeg?

Transcode can replace FFmpeg for many common use cases, but they serve slightly different purposes:

- **Transcode** is a library-first tool designed for embedding in applications, with a CLI for convenience
- **FFmpeg** is a CLI-first tool with a complex C API for programmatic use

For new Rust projects, Transcode offers a cleaner integration. For shell scripting or legacy workflows, FFmpeg may still be appropriate. See [Why Transcode?](/docs/getting-started/why-transcode) for a detailed comparison.

#### Does Transcode use FFmpeg internally?

No. Transcode is written entirely in Rust with no FFmpeg dependency. Some optional features (like certain hardware encoders) may use system libraries, but the core is pure Rust.

#### What codecs are supported?

Transcode supports all major codecs:

**Video**: H.264, H.265/HEVC, AV1, VP9, VP8, ProRes, DNxHD, MPEG-2, and more
**Audio**: AAC, Opus, FLAC, MP3 (decode), AC3, DTS, and more
**Containers**: MP4, MKV, WebM, HLS, DASH, MPEG-TS, and more

See the [Codec Matrix](/docs/reference/codecs-matrix) for full details.

#### Is Transcode production-ready?

Yes. Core codecs (H.264, H.265, AV1, AAC, Opus) and containers (MP4, MKV) are production-ready and well-tested. Some experimental features are marked as such in the documentation.

#### What's the license?

Transcode is dual-licensed under MIT and Apache 2.0. You can choose whichever license works best for your project.

### Performance

#### Why is my encoding slow?

Check these common causes:

1. **Debug mode**: Ensure you're using `--release`:
   ```bash
   cargo build --release
   cargo run --release
   ```

2. **SIMD not detected**: Verify SIMD support:
   ```bash
   transcode info --simd
   ```

3. **Wrong preset**: Use faster presets for less critical content:
   ```bash
   transcode -i input.mp4 -o output.mp4 --preset fast
   ```

4. **CPU throttling**: Check if your CPU is thermal throttling

#### How do I enable GPU acceleration?

```rust
let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.mp4")
    .gpu_acceleration(true);
```

Or via CLI:
```bash
transcode -i input.mp4 -o output.mp4 --gpu
```

GPU acceleration requires the `gpu` feature and a compatible GPU. See [GPU Acceleration](/docs/guides/gpu-acceleration) for details.

#### How much memory does Transcode use?

Memory usage scales with resolution:

| Resolution | Typical Usage |
|------------|---------------|
| 720p | ~95 MB |
| 1080p | ~180 MB |
| 4K | ~580 MB |

Use frame pools to reduce memory:
```rust
let options = TranscodeOptions::new()
    .use_frame_pool(true);
```

### Installation

#### Build fails with "missing dav1d"

The AV1 decoder requires dav1d. Install it:

```bash
# Ubuntu/Debian
sudo apt install libdav1d-dev

# macOS
brew install dav1d

# Fedora
sudo dnf install dav1d-devel
```

#### Python bindings won't install

Ensure you have Rust installed:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then install with pip:
```bash
pip install transcode-py
```

If building from source:
```bash
cd transcode-python
pip install maturin
maturin develop
```

#### WebAssembly build fails

Install wasm-pack:
```bash
cargo install wasm-pack
```

Then build:
```bash
cd transcode-wasm
wasm-pack build --target web
```

### Usage

#### How do I transcode to a specific bitrate?

```rust
let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.mp4")
    .video_bitrate(5_000_000)  // 5 Mbps
    .audio_bitrate(128_000);   // 128 kbps
```

CLI:
```bash
transcode -i input.mp4 -o output.mp4 --video-bitrate 5000 --audio-bitrate 128
```

#### How do I extract just audio?

```rust
let options = TranscodeOptions::new()
    .input("video.mp4")
    .output("audio.aac")
    .video_codec("none");
```

CLI:
```bash
transcode -i video.mp4 -o audio.aac --no-video
```

#### How do I generate HLS/DASH streams?

```rust
use transcode_streaming::{HlsMuxer, StreamingConfig};

let config = StreamingConfig::default()
    .segment_duration(6.0);

let hls = HlsMuxer::new("output/", config)?;
```

CLI:
```bash
transcode -i input.mp4 --hls output/
```

See [Streaming Output](/docs/guides/streaming-output) for multi-bitrate variants.

#### How do I use Transcode in async code?

```rust
use transcode::AsyncTranscoder;

async fn transcode_async() -> Result<()> {
    let options = TranscodeOptions::new()
        .input("input.mp4")
        .output("output.mp4");

    let transcoder = AsyncTranscoder::new(options)?;
    transcoder.run().await?;

    Ok(())
}
```

---

## Troubleshooting

### Error Messages

#### "Unsupported codec"

The input file uses a codec not supported by Transcode. Check the [Codec Matrix](/docs/reference/codecs-matrix) for supported formats.

**Solution**: Convert with a supported codec or use the `copy` codec to pass through:
```bash
transcode -i input.mp4 -o output.mkv --video-codec copy
```

#### "BitstreamCorruption"

The input file is corrupted or uses non-standard encoding.

**Solutions**:
1. Try re-downloading the file
2. Use `--strict false` to attempt recovery:
   ```bash
   transcode -i corrupted.mp4 -o output.mp4 --strict false
   ```
3. Check if the file plays in VLC (which is more lenient)

#### "OutOfMemory"

The system ran out of memory during processing.

**Solutions**:
1. Enable frame pools:
   ```rust
   .use_frame_pool(true)
   ```
2. Reduce concurrent streams
3. Process in segments for very large files
4. Increase system swap space

#### "GpuNotAvailable"

GPU acceleration was requested but no compatible GPU was found.

**Solutions**:
1. Check GPU drivers are installed
2. Verify wgpu compatibility:
   ```bash
   transcode info --gpu
   ```
3. Fall back to CPU processing:
   ```rust
   .gpu_acceleration(false)
   ```

#### "HardwareEncoderNotFound"

Hardware encoding was requested but the encoder isn't available.

**Solutions**:
1. Install required drivers (NVIDIA, Intel, AMD)
2. Check hardware capabilities:
   ```bash
   transcode info --hwaccel
   ```
3. Use software encoding as fallback

### Common Issues

#### Output file is larger than input

This can happen when:
1. **Bitrate is higher**: Check your bitrate settings
2. **Less efficient codec**: e.g., converting H.265 → H.264
3. **CRF is too low**: Higher quality = larger files

**Solution**: Use `--crf 23` for balanced quality/size or set a target bitrate.

#### Audio/video sync issues

**Causes**:
- Variable frame rate input
- Corrupted timestamps
- Stream copy with incompatible containers

**Solutions**:
1. Force constant frame rate:
   ```bash
   transcode -i input.mp4 -o output.mp4 -F "fps=30"
   ```
2. Re-encode both streams (don't use `--copy`)

#### Colors look wrong after transcoding

**Causes**:
- Color space mismatch
- HDR to SDR conversion without tone mapping

**Solutions**:
1. Preserve color metadata:
   ```bash
   transcode -i input.mp4 -o output.mp4 --preserve-color
   ```
2. For HDR content, use tone mapping:
   ```bash
   transcode -i hdr.mp4 -o sdr.mp4 --tonemap
   ```

#### Transcoding stops at a specific frame

**Causes**:
- Corrupted frame in source
- Encoder state overflow

**Solutions**:
1. Skip the corrupted section:
   ```bash
   transcode -i input.mp4 -o output.mp4 --skip-errors
   ```
2. Transcode in segments around the corruption

### Platform-Specific Issues

#### macOS: "Library not loaded: libdav1d"

Install dav1d via Homebrew:
```bash
brew install dav1d
```

If using an M1/M2 Mac, ensure you're using the ARM version:
```bash
arch -arm64 brew install dav1d
```

#### Linux: "VAAPI initialization failed"

Install VA-API drivers:
```bash
# Intel
sudo apt install intel-media-va-driver

# AMD
sudo apt install mesa-va-drivers

# Verify
vainfo
```

#### Windows: "NVENC not found"

1. Update NVIDIA drivers to 470.x or later
2. Install CUDA Toolkit 11.0+
3. Ensure `nvEncodeAPI64.dll` is in your PATH

### Getting Help

If you're still stuck:

1. **Search existing issues**: [GitHub Issues](https://github.com/transcode/transcode/issues)
2. **Check discussions**: [GitHub Discussions](https://github.com/transcode/transcode/discussions)
3. **Join Discord**: [Transcode Discord](https://discord.gg/transcode)
4. **File a bug report** with:
   - Transcode version (`transcode --version`)
   - OS and architecture
   - Minimal reproduction steps
   - Input file details (codec, resolution, duration)
