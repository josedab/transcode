# transcode-ffv1

A pure Rust implementation of the FFV1 lossless video codec.

## Features

- **Lossless Compression**: Bit-perfect video preservation
- **Multiple Bit Depths**: 8, 9, 10, 12, 14, and 16-bit support
- **Color Spaces**: YCbCr (4:2:0, 4:2:2, 4:4:4) and RGB
- **Slice-Based Encoding**: Parallel encoding/decoding
- **CRC32 Integrity**: Data integrity checking (version 3+)
- **Archival Quality**: Ideal for video preservation

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-ffv1 = { path = "../transcode-ffv1" }
```

### Decoding

```rust
use transcode_ffv1::{Ffv1Decoder, Ffv1Config};

// Parse configuration from container
let config = Ffv1Config::from_record(&config_record)?;
let mut decoder = Ffv1Decoder::new(config)?;

// Decode frames
let frame = decoder.decode(&encoded_frame)?;
println!("Decoded: {}x{}", frame.width, frame.height);
```

### Configuration

```rust
use transcode_ffv1::{Ffv1Config, ColorSpace, ChromaSubsampling};

// Create configuration for 1080p 10-bit YUV 4:2:0
let config = Ffv1Config::new(1920, 1080, 10)?;

// Configure advanced options
let mut config = Ffv1Config::default();
config.width = 3840;
config.height = 2160;
config.bits_per_raw_sample = 10;
config.colorspace = ColorSpace::YCbCr;
config.chroma_subsampling = ChromaSubsampling::Yuv420;
config.num_h_slices = 4;
config.num_v_slices = 4;
```

### Encoding (with `encoder` feature)

```rust
use transcode_ffv1::{Ffv1Encoder, Ffv1EncoderConfig};

let config = Ffv1EncoderConfig::new(1920, 1080, 10);
let mut encoder = Ffv1Encoder::new(config)?;
let encoded = encoder.encode(&frame)?;
```

## Supported Versions

| Version | Features |
|---------|----------|
| 0 | Basic |
| 1 | Variable bit depth |
| 3 | Slices, CRC, multi-threading |

## Feature Flags

- `encoder`: Enable FFV1 encoding support

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
