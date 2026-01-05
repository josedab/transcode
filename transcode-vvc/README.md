# transcode-vvc

VVC/H.266 (Versatile Video Coding) implementation for the transcode library.

## Overview

VVC offers approximately 50% better compression efficiency compared to HEVC/H.265 while maintaining similar visual quality. This crate provides a comprehensive implementation following the ITU-T H.266 specification.

## Features

### Core Capabilities
- **NAL Unit Parsing**: VPS, SPS, PPS, Picture Header, SEI units
- **Profile/Tier/Level**: Main10, Main10 4:4:4, Multilayer profiles
- **CABAC Entropy Coding**: VVC-specific context models
- **Transform/Quantization**: DCT-II, DST-VII, DCT-VIII (4x4 to 64x64)

### Advanced Coding Tools
- **Intra Prediction**: 67 modes + MIP, ISP, BDPCM
- **Inter Prediction**: MMVD, SMVD, Affine, GPM, CIIP, BDOF, DMVR
- **In-Loop Filtering**: Deblocking, SAO, ALF, CCALF, LMCS

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-vvc = { path = "../transcode-vvc" }
```

### Decoding

```rust
use transcode_vvc::{VvcDecoder, VvcDecoderConfig};

let mut decoder = VvcDecoder::new()?;

// Decode NAL units
for nal_data in nal_units {
    if let Some(frame) = decoder.decode_nal(&nal_data)? {
        println!("Decoded: {}x{}", frame.width, frame.height);
    }
}
```

### Encoding

```rust
use transcode_vvc::{VvcEncoder, VvcEncoderConfig, VvcPreset};

let config = VvcEncoderConfig::with_preset(VvcPreset::Medium)
    .with_resolution(1920, 1080)
    .with_framerate(30, 1);

let mut encoder = VvcEncoder::new(config)?;
let packets = encoder.encode(&frame)?;
```

### Configuration

```rust
use transcode_vvc::{VvcConfig, VvcProfile, VvcLevel, VvcTier};

let config = VvcConfig::new()
    .with_resolution(3840, 2160)
    .with_profile(VvcProfile::Main10)
    .with_level(VvcLevel::L5_1)
    .with_tier(VvcTier::Main)
    .with_bit_depth(10)
    .with_ctu_size(128);

config.validate()?;
```

## Profiles

| Profile | Description |
|---------|-------------|
| Main10 | 10-bit 4:2:0 |
| Main10 4:4:4 | 10-bit 4:4:4 |
| Multilayer Main10 | Multi-layer support |
| Still Picture | Single frame |

## Levels

| Level | Max Resolution | Max Sample Rate |
|-------|----------------|-----------------|
| 3.1 | 1280x720 | 720p60 |
| 4.0/4.1 | 2048x1080 | 1080p60 |
| 5.0/5.1/5.2 | 4096x2160 | 4K60 |
| 6.0/6.1/6.2 | 8192x4320 | 8K60 |
| 6.3 | 16384x8640 | 16K |

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
