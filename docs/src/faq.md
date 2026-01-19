# Frequently Asked Questions

## General

### What is Transcode?

Transcode is a memory-safe, high-performance universal codec library written entirely in Rust. It provides video and audio encoding/decoding without relying on FFmpeg or other C libraries.

### Why use Transcode instead of FFmpeg?

| Aspect | Transcode | FFmpeg |
|--------|-----------|--------|
| Memory Safety | Rust's guarantees | C/C++ (manual) |
| Dependencies | Pure Rust | Many system libs |
| Cross-compilation | Simple | Complex |
| Binary Size | ~50MB | ~100MB+ |
| Performance | Comparable | Mature optimizations |

Choose Transcode when:
- Memory safety is critical
- You need easy cross-compilation
- You want a single binary with no dependencies
- You're building a Rust application

Choose FFmpeg when:
- You need maximum compatibility
- You need codecs we don't support yet
- You're using established FFmpeg-based workflows

### Is Transcode production-ready?

Yes for core functionality. Check the [Feature Maturity](https://github.com/transcode/transcode#feature-maturity) table in the README for status of specific features.

## Installation

### What's the minimum Rust version?

Rust 1.75 or later is required.

### Do I need system dependencies?

The core library (H.264, AAC, MP4) has no system dependencies. Some optional features require:

- **AV1 decoding**: libdav1d
- **Hardware acceleration**: Platform-specific (VA-API, VideoToolbox, NVENC)

### How do I install the CLI tool?

```bash
cargo install transcode-cli
```

## Usage

### How do I transcode a video?

```rust
use transcode::{Transcoder, TranscodeOptions};

let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.mp4")
    .video_bitrate(5_000_000);

let mut transcoder = Transcoder::new(options)?;
transcoder.run()?;
```

### How do I use GPU acceleration?

```rust
let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.mp4")
    .gpu(true);
```

GPU acceleration requires the `gpu` feature and compatible hardware.

### Can I transcode without re-encoding?

Yes, use codec "copy":

```bash
transcode -i input.mkv -o output.mp4 --video-codec copy --audio-codec copy
```

### How do I extract audio from a video?

```bash
transcode -i video.mp4 -o audio.aac --no-video
```

## Codecs

### Which codecs are supported?

**Video**: H.264, H.265/HEVC, AV1, VP9, VP8, ProRes, DNxHD, MPEG-2, FFV1, Theora, Cineform

**Audio**: AAC, MP3, Opus, FLAC, AC-3, DTS, ALAC, Vorbis, PCM

### Why is MP3 encoding not supported?

MP3 is patent-encumbered in some jurisdictions. We support MP3 decoding but recommend Opus or AAC for encoding.

### Is VVC/H.266 supported?

VVC support is in progress. Check the [roadmap](https://github.com/transcode/transcode#roadmap) for status.

## Performance

### How fast is Transcode?

Performance is comparable to FFmpeg (+/- 15%) for most codecs. SIMD optimizations (AVX2, NEON) are automatically used when available.

### How do I check SIMD capabilities?

```rust
use transcode_codecs::detect_simd;

let caps = detect_simd();
println!("AVX2: {}", caps.avx2);
println!("Best: {}", caps.best_level());
```

### How do I improve encoding speed?

1. Use a faster preset: `--preset fast` or `--preset ultrafast`
2. Enable GPU acceleration: `--gpu`
3. Use more threads: `--threads 8`
4. Use CRF mode instead of bitrate mode

## Troubleshooting

### "Unsupported codec" error

Ensure the codec is enabled in your Cargo.toml features:

```toml
[dependencies]
transcode = { version = "1.0", features = ["av1", "hevc"] }
```

### "File not found" error

Check that the input path is correct and the file exists:

```rust
if !std::path::Path::new("input.mp4").exists() {
    eprintln!("Input file not found!");
}
```

### High memory usage

For large files, enable streaming mode:

```rust
let options = TranscodeOptions::new()
    .input("large.mp4")
    .output("output.mp4")
    .streaming(true);  // Process in chunks
```

### Corrupted output

1. Ensure the transcoding completed (check exit code)
2. Verify input file integrity
3. Try a different codec or container

## Contributing

### How do I contribute?

See [Contributing Guide](./contributing/development-setup.md).

### How do I report a bug?

Open an issue on GitHub with:
- Transcode version
- Operating system
- Steps to reproduce
- Expected vs actual behavior

### How do I request a feature?

Open a GitHub issue using the feature request template.
