---
sidebar_position: 1
title: Installation
description: Install Transcode for Rust, Python, Node.js, or use the CLI
---

# Installation

Transcode is available for multiple platforms and languages. Choose the installation method that fits your workflow.

## Rust

Add Transcode to your `Cargo.toml`:

```toml
[dependencies]
transcode = "1.0"
```

Or use cargo add:

```bash
cargo add transcode
```

### Feature Flags

Transcode uses feature flags to enable optional functionality:

```toml
[dependencies]
transcode = { version = "1.0", features = ["gpu", "ai", "distributed"] }
```

| Feature | Description |
|---------|-------------|
| `default` | Core transcoding with H.264, AAC, MP4 support |
| `gpu` | GPU acceleration via wgpu |
| `ai` | AI enhancement (upscaling, denoising) |
| `distributed` | Distributed transcoding support |
| `streaming` | HLS/DASH output |
| `quality` | Quality metrics (PSNR, SSIM, VMAF) |
| `all` | Enable all features |

## Python

Install the Python bindings via pip:

```bash
pip install transcode-py
```

### Verify Installation

```python
import transcode_py

# Check version
print(transcode_py.__version__)

# Check SIMD capabilities
caps = transcode_py.detect_simd()
print(f"Best SIMD level: {caps.best_level()}")
```

## Node.js

Install the Node.js bindings:

```bash
npm install transcode
```

Or with yarn:

```bash
yarn add transcode
```

## WebAssembly

For browser-based transcoding:

```bash
npm install transcode-wasm
```

## CLI

Install the command-line tool:

```bash
cargo install transcode-cli
```

Verify the installation:

```bash
transcode --version
transcode --help
```

## Docker

Pull the official Docker image:

```bash
docker pull transcode/transcode:latest
```

Run a transcoding job:

```bash
docker run -v $(pwd):/data transcode/transcode \
  -i /data/input.mp4 \
  -o /data/output.mp4
```

## Building from Source

### Prerequisites

- Rust 1.75 or later
- Python 3.8+ (for Python bindings)

### Clone and Build

```bash
# Clone the repository
git clone https://github.com/transcode/transcode.git
cd transcode

# Build all crates
cargo build --release

# Run tests
cargo test

# Install CLI locally
cargo install --path transcode-cli
```

## Optional Dependencies

Some features require additional system libraries:

### AV1 Codec

The AV1 decoder uses dav1d via FFI:

```bash
# Ubuntu/Debian
sudo apt install pkg-config libdav1d-dev

# macOS
brew install dav1d

# Fedora/RHEL
sudo dnf install dav1d-devel
```

### Hardware Acceleration

For GPU-accelerated encoding:

**VA-API (Linux):**
```bash
sudo apt install libva-dev vainfo
```

**VideoToolbox (macOS):**
Included with macOS 10.13+.

**NVENC (NVIDIA):**
Requires NVIDIA driver 470.x+ and CUDA Toolkit 11.0+.

## Troubleshooting

### Common Issues

**"cargo: command not found"**

Install Rust via rustup:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**"pip: No module named transcode_py"**

Ensure you're using the correct Python environment:
```bash
python3 -m pip install transcode-py
```

**Build fails with missing system libraries**

Install the required dependencies for your platform as listed above.

## Next Steps

- [Quick Start](/docs/getting-started/quick-start) - Your first transcode in 2 minutes
- [First Transcode](/docs/getting-started/first-transcode) - Detailed walkthrough
- [CLI Reference](/docs/reference/cli) - Complete CLI documentation
