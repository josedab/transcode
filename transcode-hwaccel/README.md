# transcode-hwaccel

Hardware-accelerated video encoding and decoding for the transcode project.

## Overview

This crate provides a unified interface to hardware video acceleration APIs across platforms:

| Platform | API | Feature Flag |
|----------|-----|--------------|
| Linux | VA-API | `vaapi` |
| macOS | VideoToolbox | `videotoolbox` |
| NVIDIA GPUs | NVENC/NVDEC | `nvenc` |
| Intel | Quick Sync Video | `qsv` |
| AMD | Advanced Media Framework | `amf` |

## Supported Codecs

- H.264/AVC
- H.265/HEVC
- AV1
- VP9/VP8
- MPEG-2, VC-1, JPEG

## Key Types

### Detection

```rust
use transcode_hwaccel::{detect_accelerators, detect_best, HwAccelInfo};

// Detect all available accelerators (sorted by priority)
let accelerators = detect_accelerators();

// Get the best available accelerator
if let Some(best) = detect_best() {
    println!("Using: {}", best.accel_type.name());
}
```

### Encoder

```rust
use transcode_hwaccel::{HwEncoder, HwEncoderConfig, HwAccelType, HwPreset};

// Create encoder config
let config = HwEncoderConfig::h264(1920, 1080, 5_000_000)
    .with_preset(HwPreset::Medium)
    .with_gop_size(60);

// Create and use encoder
let mut encoder = HwEncoder::new(HwAccelType::VideoToolbox, config)?;
let packet = encoder.encode(&frame)?;
```

### Decoder

```rust
use transcode_hwaccel::{HwDecoder, HwDecoderConfig, HwAccelType};

let config = HwDecoderConfig::h264();
let mut decoder = HwDecoder::new(HwAccelType::Nvenc, config)?;
let frame = decoder.decode(&packet)?;
```

### Configuration Options

**Rate Control** (`HwRateControl`):
- `Cqp(qp)` - Constant QP
- `Cbr(bitrate)` - Constant bitrate
- `Vbr { target, max }` - Variable bitrate
- `Cq(quality)` - Constant quality

**Presets** (`HwPreset`): `Fastest`, `Fast`, `Medium`, `Slow`, `Slowest`

**Profiles** (`HwProfile`): `H264Baseline`, `H264Main`, `H264High`, `HevcMain`, `HevcMain10`, `Av1Main`

**Surface Formats** (`HwSurfaceFormat`): `Nv12`, `P010`, `Yuv420p`, `Yuv420p10`, `Rgba`, `Bgra`

## Capabilities

Query hardware capabilities before encoding:

```rust
use transcode_hwaccel::get_capabilities;

let caps = get_capabilities(HwAccelType::VideoToolbox)?;
println!("Max resolution: {}x{}", caps.max_width, caps.max_height);
println!("10-bit support: {}", caps.supports_10bit);
println!("HDR support: {}", caps.supports_hdr);
```

## Feature Flags

```toml
[dependencies]
transcode-hwaccel = { version = "0.1", features = ["nvenc", "async"] }
```

- `vaapi` - VA-API support (Linux)
- `videotoolbox` - VideoToolbox support (macOS)
- `nvenc` - NVIDIA NVENC/NVDEC support
- `qsv` - Intel Quick Sync support
- `amf` - AMD AMF support
- `async` - Async encoding/decoding via Tokio

## License

See the workspace root for license information.
