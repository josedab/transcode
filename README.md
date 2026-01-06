# Transcode

A memory-safe, high-performance universal codec library written in Rust.

[![CI](https://github.com/transcode/transcode/workflows/CI/badge.svg)](https://github.com/transcode/transcode/actions)
[![codecov](https://codecov.io/gh/transcode/transcode/branch/main/graph/badge.svg)](https://codecov.io/gh/transcode/transcode)
[![Crates.io](https://img.shields.io/crates/v/transcode.svg)](https://crates.io/crates/transcode)
[![Documentation](https://docs.rs/transcode/badge.svg)](https://docs.rs/transcode)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)

## Features

- **Memory Safe**: Built entirely in Rust, eliminating buffer overflow vulnerabilities common in C/C++ codec implementations
- **High Performance**: SIMD-optimized (AVX2, NEON) with automatic runtime detection
- **GPU Acceleration**: Compute shaders via wgpu for color conversion, scaling, and effects
- **AI Enhancement**: Neural network-based upscaling, denoising, and frame interpolation
- **Distributed Processing**: Scale across multiple workers with fault tolerance
- **Content Intelligence**: Scene detection and content classification for adaptive encoding
- **Quality Metrics**: PSNR, SSIM, MS-SSIM, and VMAF for perceptual quality assessment
- **Streaming Output**: HLS and DASH support with adaptive bitrate encoding
- **WebAssembly**: Run in browsers with Web Workers for parallel processing
- **Pure Rust**: No FFmpeg or system library dependencies
- **Cross-Platform**: Works on Linux, macOS, Windows, and WebAssembly
- **Python Bindings**: First-class Python support via PyO3

## Supported Formats

### Video Codecs
| Codec | Decode | Encode | Status |
|-------|--------|--------|--------|
| H.264/AVC | ✅ | ✅ | Complete |
| AV1 | ✅ | ✅ | Complete (via rav1e/dav1d) |
| H.265/HEVC | ✅ | ✅ | Complete |
| VP9 | ✅ | ✅ | Complete |
| VP8 | ✅ | ✅ | Complete |
| ProRes | ✅ | ✅ | Complete |
| DNxHD/DNxHR | ✅ | ✅ | Complete |
| MPEG-2 | ✅ | ✅ | Complete |
| FFV1 | ✅ | ✅ | Complete |
| Theora | ✅ | ❌ | Decode only |
| Cineform | ✅ | ✅ | Complete |
| VVC/H.266 | 🔄 | 🔄 | In Progress |

### Audio Codecs
| Codec | Decode | Encode | Status |
|-------|--------|--------|--------|
| AAC | ✅ | ✅ | Complete |
| MP3 | ✅ | ❌ | Decode only |
| Opus | ✅ | ✅ | Complete |
| FLAC | ✅ | ✅ | Complete |
| AC3/E-AC3 | ✅ | ✅ | Complete |
| DTS | ✅ | ✅ | Complete |
| ALAC | ✅ | ✅ | Complete |
| PCM | ✅ | ✅ | Complete |
| Vorbis | ✅ | ✅ | Complete |

### Container Formats
| Format | Demux | Mux | Status |
|--------|-------|-----|--------|
| MP4/MOV | ✅ | ✅ | Complete |
| HLS | ❌ | ✅ | Mux only |
| DASH | ❌ | ✅ | Mux only |
| MKV | ✅ | ✅ | Complete |
| WebM | ✅ | ✅ | Complete |
| MPEG-TS | ✅ | ✅ | Complete |
| AVI | ✅ | ✅ | Complete |
| FLV | ✅ | ✅ | Complete |
| MXF | ✅ | ✅ | Complete |

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode = "0.1"
```

### CLI Usage

```bash
# Install the CLI
cargo install transcode-cli

# Transcode a video
transcode -i input.mp4 -o output.mp4 --video-codec h264 --video-bitrate 5000000

# Show help
transcode --help
```

### Library Usage

```rust
use transcode::{Transcoder, TranscodeOptions};

fn main() -> transcode::Result<()> {
    // Configure transcoding options
    let options = TranscodeOptions::new()
        .input("input.mp4")
        .output("output.mp4")
        .video_bitrate(5_000_000)  // 5 Mbps
        .audio_bitrate(128_000)    // 128 kbps
        .overwrite(true);

    // Create and run transcoder
    let mut transcoder = Transcoder::new(options)?;
    transcoder.run()?;

    // Get statistics
    let stats = transcoder.stats();
    println!("Processed {} frames", stats.frames_encoded);
    println!("Compression ratio: {:.2}x", stats.compression_ratio());

    Ok(())
}
```

### Python Usage

```bash
pip install transcode-py
```

```python
import transcode_py

# Simple transcoding
stats = transcode_py.transcode('input.mp4', 'output.mp4')
print(f"Processed {stats.frames_encoded} frames")
print(f"Compression ratio: {stats.compression_ratio:.2f}x")

# Using the builder pattern
options = transcode_py.TranscodeOptions()
options = options.input('input.mp4')
options = options.output('output.mp4')
options = options.video_bitrate(5_000_000)
options = options.overwrite(True)

transcoder = transcode_py.Transcoder(options)
stats = transcoder.run()

# Check SIMD capabilities
caps = transcode_py.detect_simd()
print(f"Best SIMD level: {caps.best_level()}")
```

## Architecture

Transcode is organized as a modular workspace with 70+ specialized crates:

```
transcode/
├── Core
│   ├── transcode-core/           # Core types and utilities
│   ├── transcode-codecs/         # Base codec implementations (H.264, AAC, MP3)
│   ├── transcode-containers/     # Container formats (MP4)
│   └── transcode-pipeline/       # Pipeline orchestration
│
├── Video Codecs
│   ├── transcode-av1/            # AV1 (rav1e/dav1d)
│   ├── transcode-hevc/           # H.265/HEVC
│   ├── transcode-vp9/            # VP9
│   ├── transcode-vp8/            # VP8
│   ├── transcode-prores/         # Apple ProRes
│   ├── transcode-dnxhd/          # Avid DNxHD/DNxHR
│   ├── transcode-mpeg2/          # MPEG-2
│   ├── transcode-ffv1/           # FFV1 lossless
│   ├── transcode-cineform/       # GoPro Cineform
│   ├── transcode-theora/         # Theora
│   └── transcode-vvc/            # VVC/H.266 (in progress)
│
├── Audio Codecs
│   ├── transcode-opus/           # Opus
│   ├── transcode-flac/           # FLAC lossless
│   ├── transcode-ac3/            # Dolby AC-3/E-AC-3
│   ├── transcode-dts/            # DTS
│   ├── transcode-alac/           # Apple Lossless
│   ├── transcode-pcm/            # PCM audio
│   └── transcode-vorbis/         # Vorbis
│
├── Containers
│   ├── transcode-mkv/            # Matroska
│   ├── transcode-webm/           # WebM
│   ├── transcode-ts/             # MPEG-TS
│   ├── transcode-avi/            # AVI
│   ├── transcode-flv/            # Flash Video
│   └── transcode-mxf/            # MXF
│
├── Image Formats
│   ├── transcode-images/         # Image processing
│   ├── transcode-webp/           # WebP
│   ├── transcode-jpeg2000/       # JPEG 2000
│   ├── transcode-openexr/        # OpenEXR
│   └── transcode-tiff/           # TIFF
│
├── Processing
│   ├── transcode-gpu/            # GPU compute via wgpu
│   ├── transcode-ai/             # AI enhancement
│   ├── transcode-quality/        # Quality metrics (PSNR, SSIM, VMAF)
│   ├── transcode-deinterlace/    # Deinterlacing
│   ├── transcode-framerate/      # Frame rate conversion
│   ├── transcode-hdr/            # HDR processing
│   ├── transcode-resample/       # Audio resampling
│   ├── transcode-subtitle/       # Subtitle handling
│   ├── transcode-caption/        # Closed captions
│   └── transcode-watermark/      # Watermarking
│
├── Streaming & Distribution
│   ├── transcode-streaming/      # HLS/DASH output
│   ├── transcode-live/           # Live streaming
│   ├── transcode-distributed/    # Distributed processing
│   └── transcode-cloud/          # Cloud integration
│
├── Professional
│   ├── transcode-dolby/          # Dolby technologies
│   ├── transcode-spatial/        # Spatial audio
│   ├── transcode-loudness/       # Loudness metering
│   ├── transcode-timecode/       # Timecode handling
│   ├── transcode-drm/            # DRM support
│   └── transcode-pertitle/       # Per-title encoding
│
├── Intelligence
│   ├── transcode-intel/          # Content intelligence
│   ├── transcode-intelligence/   # Advanced analysis
│   ├── transcode-neural/         # Neural processing
│   └── transcode-analytics/      # Analytics
│
├── Platform Bindings
│   ├── transcode-cli/            # CLI
│   ├── transcode-python/         # Python bindings (PyO3)
│   ├── transcode-node/           # Node.js bindings
│   ├── transcode-wasm/           # WebAssembly
│   └── transcode-capi/           # C API
│
├── Hardware
│   ├── transcode-hwaccel/        # Hardware acceleration
│   └── transcode-zerocopy/       # Zero-copy transfers
│
├── Testing & Compatibility
│   ├── transcode-bench/          # Benchmarks
│   ├── transcode-conformance/    # Conformance tests
│   ├── transcode-compat/         # FFmpeg compatibility
│   └── transcode-telemetry/      # Telemetry
│
└── transcode/                    # Main library facade
```

### SIMD Optimization

The library automatically detects and uses the best available SIMD instruction set:

- **x86_64**: SSE4.2, AVX2, AVX-512, FMA
- **ARM64**: NEON, SVE
- **Fallback**: Scalar implementation for maximum compatibility

```rust
use transcode_codecs::detect_simd;

let caps = detect_simd();
println!("AVX2: {}", caps.avx2);
println!("NEON: {}", caps.neon);
println!("Best level: {}", caps.best_level());
```

## Performance

Benchmarks comparing Transcode against other implementations (1080p H.264 encoding):

| Library | Frames/sec | Memory Usage |
|---------|------------|--------------|
| Transcode (AVX2) | TBD | TBD |
| Transcode (NEON) | TBD | TBD |
| Transcode (Scalar) | TBD | TBD |

Run benchmarks locally:

```bash
cargo bench
```

## Building from Source

### Prerequisites

- Rust 1.75 or later
- Python 3.8+ (for Python bindings)

### Build

```bash
# Clone the repository
git clone https://github.com/transcode/transcode.git
cd transcode

# Build all crates
cargo build --release

# Run tests
cargo test

# Build Python bindings
cd transcode-python
pip install maturin
maturin develop
```

### Development

```bash
# Format code
cargo fmt

# Run clippy
cargo clippy --all-targets --all-features

# Generate documentation
cargo doc --open
```

## Examples

See the [transcode/examples/](transcode/examples/) directory for complete usage examples:

- `basic_transcode.rs` - Simple transcoding example
- `simd_detection.rs` - SIMD capability detection
- `codec_info.rs` - Querying codec capabilities
- `low_level_encode.rs` - Low-level encoder usage
- `quality_metrics.rs` - PSNR, SSIM, and MS-SSIM quality assessment
- `content_intelligence.rs` - Scene detection and content classification

Run an example:
```bash
cargo run --example quality_metrics
cargo run --example content_intelligence
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `cargo test`
5. Run linting: `cargo clippy`
6. Submit a pull request

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Acknowledgments

- The Rust community for excellent tooling and libraries
- [PyO3](https://pyo3.rs/) for seamless Python integration
- [Criterion](https://bheisler.github.io/criterion.rs/) for benchmarking

## Advanced Features

### GPU Acceleration

```rust
use transcode_gpu::{GpuContext, ColorConverter, GpuScaler};

// Initialize GPU context
let ctx = GpuContext::new().await?;

// Color space conversion on GPU
let converter = ColorConverter::new(&ctx)?;
let rgb_frame = converter.yuv_to_rgb(&yuv_frame).await?;

// GPU-accelerated scaling
let scaler = GpuScaler::new(&ctx, 3840, 2160)?; // 4K output
let upscaled = scaler.scale(&frame).await?;
```

### AI Enhancement

```rust
use transcode_ai::{Upscaler, Denoiser, FrameInterpolator};

// 4x upscaling with neural network
let upscaler = Upscaler::new(UpscaleMethod::Lanczos, 4)?;
let hd_frame = upscaler.process(&sd_frame)?;

// AI-based denoising
let denoiser = Denoiser::new(DenoiseMethod::Bilateral)?;
let clean_frame = denoiser.process(&noisy_frame)?;

// Frame interpolation for 60fps
let interpolator = FrameInterpolator::new(InterpolationMethod::OpticalFlow)?;
let mid_frame = interpolator.interpolate(&frame1, &frame2, 0.5)?;
```

### Quality Assessment

```rust
use transcode_quality::{Psnr, Ssim, QualityAssessment};

// Calculate PSNR
let psnr = Psnr::default();
let psnr_result = psnr.calculate(&reference, &distorted)?;
println!("PSNR: {:.2} dB", psnr_result.average());

// Calculate SSIM
let ssim = Ssim::default();
let ssim_result = ssim.calculate(&reference, &distorted)?;
println!("SSIM: {:.4}", ssim_result.score);

// Comprehensive quality assessment
let qa = QualityAssessment::new();
let report = qa.assess(&reference, &distorted)?;
```

### Content Intelligence

```rust
use transcode_intel::{VideoAnalyzer, SceneConfig, SceneDetector};

// Analyze video content
let mut analyzer = VideoAnalyzer::default();
let analysis = analyzer.analyze_sequence(&frames)?;

println!("Detected {} scenes", analysis.scene_count());
println!("Dominant content: {:?}", analysis.dominant_content);
println!("Average motion: {:.2}", analysis.avg_motion);

// Recommended bitrate factor based on content
let factor = analysis.recommended_bitrate_factor();
```

### Streaming Output

```rust
use transcode_streaming::{HlsMuxer, DashMuxer, StreamingConfig};

// HLS output with multiple quality levels
let config = StreamingConfig::default()
    .segment_duration(6.0)
    .playlist_type(PlaylistType::Event);

let hls = HlsMuxer::new("output/", config)?;
hls.write_segment(&segment)?;
hls.finish()?;

// DASH output
let dash = DashMuxer::new("output/", config)?;
dash.write_segment(&segment)?;
dash.generate_mpd()?;
```

### Distributed Transcoding

```rust
use transcode_distributed::{Coordinator, WorkerRunner, CoordinatorConfig};

// Start coordinator
let config = CoordinatorConfig::default();
let coordinator = Coordinator::new(config).await?;

// Submit a job
let job = coordinator.submit_job(
    vec![segment1, segment2, segment3],
    params,
).await?;

// Monitor progress
while let Some(event) = coordinator.next_event().await {
    match event {
        CoordinatorEvent::TaskCompleted { task_id, .. } => {
            println!("Task {} completed", task_id);
        }
        CoordinatorEvent::JobCompleted { job_id, .. } => {
            println!("Job {} finished!", job_id);
            break;
        }
        _ => {}
    }
}
```

### WebAssembly

```javascript
import init, { WasmTranscoder } from 'transcode-wasm';

await init();

// Create transcoder
const transcoder = new WasmTranscoder();
transcoder.set_input_format('h264');
transcoder.set_output_format('av1');

// Process frame
const outputFrame = transcoder.process_frame(inputFrame);
```

## Docker Usage

The transcode CLI is available as a Docker image for containerized deployments.

### Building the Docker Image

```bash
# Build the production image
docker build -t transcode .

# Build with a specific stage
docker build --target runtime -t transcode:latest .
docker build --target development -t transcode:dev .
```

### Running with Docker

```bash
# Basic transcoding
docker run -v $(pwd):/data transcode -i /data/input.mp4 -o /data/output.mp4

# With codec and bitrate options
docker run -v $(pwd):/data transcode \
    -i /data/input.mp4 \
    -o /data/output.mp4 \
    --video-codec h264 \
    --audio-codec aac \
    --video-bitrate 5000

# Show help
docker run transcode --help

# Interactive mode
docker run -it -v $(pwd):/data transcode -i /data/input.mp4 -o /data/output.mp4
```

### Using Docker Compose

```bash
# Build all images
docker compose build

# Run transcoding
docker compose run --rm transcode -i /data/input.mp4 -o /data/output.mp4

# Start development environment with hot reload
docker compose up dev

# Run tests
docker compose run --rm test

# Run linting and formatting checks
docker compose run --rm lint

# Run security audit
docker compose run --rm audit

# Start observability stack (Jaeger, Prometheus, Grafana)
docker compose --profile observability up
```

### Volume Mounting

The container expects input/output files to be accessible via the `/data` directory:

```bash
# Mount current directory
docker run -v $(pwd):/data transcode -i /data/video.mp4 -o /data/output.mp4

# Mount specific directories for input and output
docker run \
    -v /path/to/input:/input:ro \
    -v /path/to/output:/output:rw \
    transcode -i /input/video.mp4 -o /output/result.mp4
```

### Image Variants

| Image | Size | Description |
|-------|------|-------------|
| `transcode:latest` | ~50MB | Production runtime (Debian slim) |
| `transcode:dev` | ~2GB | Development with full Rust toolchain |
| `transcode:alpine` | ~30MB | Alpine-based minimal image |

### Signal Handling

The Docker image uses [tini](https://github.com/krallin/tini) as an init process to ensure proper signal handling:

- **SIGTERM/SIGINT**: Graceful shutdown, allowing current transcoding to complete
- **Zombie process reaping**: Prevents zombie processes from accumulating

### Security Features

- Runs as non-root user (`transcode`, UID 1000)
- Read-only root filesystem (when using docker-compose)
- No new privileges security option
- Minimal runtime dependencies

## Roadmap

### Completed
- [x] AV1 codec support (rav1e encoder, dav1d decoder)
- [x] H.265/HEVC codec support
- [x] VP9/VP8 codec support
- [x] ProRes, DNxHD, MPEG-2, FFV1, Cineform, Theora codecs
- [x] Opus, FLAC, AC3, DTS, ALAC, PCM, Vorbis audio codecs
- [x] MKV/WebM container support
- [x] MPEG-TS, AVI, FLV, MXF container support
- [x] WebAssembly support with Web Workers
- [x] HLS/DASH streaming output
- [x] GPU acceleration (wgpu)
- [x] Hardware acceleration (VAAPI, VideoToolbox, NVENC, QSV)
- [x] AI enhancement pipeline
- [x] Quality metrics (PSNR, SSIM, VMAF)
- [x] Distributed transcoding
- [x] Content intelligence
- [x] HDR processing
- [x] DRM support
- [x] Per-title encoding optimization
- [x] FFmpeg compatibility layer

### Planned
- [ ] VVC/H.266 codec support (in progress)
- [ ] Live streaming input (in progress)
- [ ] Audio enhancement (noise reduction, normalization)
