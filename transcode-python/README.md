# transcode-py

Python bindings for the Transcode codec library - a memory-safe, high-performance media processing library written in Rust with PyO3.

## Features

- **Memory Safe**: Built on Rust, eliminating buffer overflow vulnerabilities
- **High Performance**: SIMD-optimized (SSE4.2, AVX2, AVX-512, NEON, SVE) with runtime detection
- **Simple API**: Easy-to-use Python interface for common transcoding tasks
- **Flexible Configuration**: Builder pattern for resolution, bitrate, codec, and threading options

## Installation

```bash
pip install transcode-py
```

### Building from Source

Requires Rust toolchain and maturin:

```bash
pip install maturin
cd transcode-python
maturin develop --release
```

## Quick Start

```python
import transcode_py

# Simple transcoding
stats = transcode_py.transcode("input.mp4", "output.mp4")
print(f"Processed {stats.frames_encoded} frames")
print(f"Compression ratio: {stats.compression_ratio:.2f}x")
```

## Advanced Usage

### Using Options Builder

```python
import transcode_py

# Configure transcoding options
options = (
    transcode_py.TranscodeOptions()
    .input("input.mp4")
    .output("output.mp4")
    .video_codec("h264")
    .audio_codec("aac")
    .video_bitrate(5_000_000)   # 5 Mbps
    .audio_bitrate(128_000)     # 128 kbps
    .resolution(1920, 1080)     # Output resolution
    .threads(4)                 # Encoding threads
    .overwrite(True)
)

# Create and run transcoder
transcoder = transcode_py.Transcoder(options)
stats = transcoder.run()

print(f"Input size: {stats.input_size} bytes")
print(f"Output size: {stats.output_size} bytes")
```

### Check SIMD Capabilities

```python
import transcode_py

# Detect available SIMD features
caps = transcode_py.detect_simd()
print(caps)  # SimdCapabilities(AVX2, SSE4.2, ...)
print(f"Best SIMD level: {caps.best_level()}")
print(f"Has SIMD acceleration: {caps.has_simd()}")
```

## API Reference

### Functions

- `transcode(input_path, output_path, video_codec=None, audio_codec=None, video_bitrate=None, audio_bitrate=None, overwrite=False)` - High-level transcoding
- `detect_simd()` - Detect CPU SIMD capabilities
- `version()` - Get library version

### Classes

- `TranscodeOptions` - Builder for transcoding configuration
- `Transcoder` - Transcoding engine
- `TranscodeStats` - Statistics (packets_processed, frames_decoded, frames_encoded, input_size, output_size, compression_ratio)
- `SimdCapabilities` - SIMD detection (sse42, avx2, avx512, fma, neon, sve)

## Requirements

- Python >= 3.8
- Rust toolchain (for building from source)

## License

MIT OR Apache-2.0
