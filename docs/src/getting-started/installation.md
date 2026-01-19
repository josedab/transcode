# Installation

Transcode can be used as a Rust library, command-line tool, or Python package.

## Rust Library

Add Transcode to your `Cargo.toml`:

```toml
[dependencies]
transcode = "1.0"
```

### Feature Flags

Transcode uses feature flags to enable optional functionality:

```toml
[dependencies]
transcode = { version = "1.0", features = ["gpu", "ai"] }
```

| Feature | Description |
|---------|-------------|
| `default` | Core codecs (H.264, AAC, MP3) and containers (MP4) |
| `gpu` | GPU acceleration via wgpu |
| `ai` | AI-based enhancement (upscaling, denoising) |
| `streaming` | HLS/DASH output support |
| `distributed` | Distributed transcoding |
| `full` | All features enabled |

## Command-Line Tool

### From crates.io

```bash
cargo install transcode-cli
```

### From Source

```bash
git clone https://github.com/transcode/transcode.git
cd transcode
cargo install --path transcode-cli
```

### Verify Installation

```bash
transcode --version
transcode --help
```

## Python Package

### From PyPI

```bash
pip install transcode-py
```

### From Source

```bash
git clone https://github.com/transcode/transcode.git
cd transcode/transcode-python
pip install maturin
maturin develop --release
```

### Verify Installation

```python
import transcode_py
print(f"Version: {transcode_py.version()}")
```

## Docker

### Pull from Registry

```bash
docker pull transcode/transcode:latest
```

### Build Locally

```bash
git clone https://github.com/transcode/transcode.git
cd transcode
docker build -t transcode .
```

### Run

```bash
docker run -v $(pwd):/data transcode -i /data/input.mp4 -o /data/output.mp4
```

## System Requirements

### Minimum Requirements

- **Rust**: 1.75 or later (for building from source)
- **Python**: 3.8 or later (for Python bindings)
- **Memory**: 512 MB RAM
- **Disk**: 100 MB for installation

### Optional Dependencies

Some features require additional system libraries:

#### AV1 Codec (dav1d decoder)

```bash
# Ubuntu/Debian
sudo apt install libdav1d-dev

# macOS
brew install dav1d

# Fedora
sudo dnf install dav1d-devel
```

#### Hardware Acceleration

See the [Hardware Acceleration Guide](../guides/gpu-acceleration.md) for platform-specific requirements.

## Verifying Your Installation

Run the SIMD detection example to verify your installation and see available optimizations:

```bash
cargo run --example simd_detection
```

Expected output:

```
SIMD Capabilities:
  SSE4.2: true
  AVX2: true
  AVX-512: false
  NEON: false
  Best level: AVX2
```

## Next Steps

- [Quick Start](./quick-start.md) - Your first transcoding job
- [CLI Usage](./cli-usage.md) - Command-line reference
- [Transcoding Basics](../guides/transcoding-basics.md) - Core concepts
