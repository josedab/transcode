# transcode-cineform

A pure Rust implementation of the GoPro CineForm intermediate video codec.

## Features

- **Wavelet-Based Compression**: Haar wavelet transforms for high-quality editing
- **Multiple Quality Levels**: Proxy to Film Scan quality presets
- **10-bit/12-bit Support**: Professional bit depth handling
- **Alpha Channel**: Full RGBA support for 4444 variants
- **Editing-Friendly**: Low decode complexity for NLE workflows

## Supported Profiles

| Profile | Quality | Use Case |
|---------|---------|----------|
| Low (Proxy) | Lowest | Offline editing |
| Medium | Balanced | General editing |
| High | High | Finishing work |
| Film Scan 1 | Very High | High-end production |
| Film Scan 2 | Maximum | Digital cinema |

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-cineform = { path = "../transcode-cineform" }
```

### Decoding

```rust
use transcode_cineform::{CineformDecoder, CineformFrame};

let data = std::fs::read("video.cfhd")?;
let mut decoder = CineformDecoder::new();
let frame = decoder.decode_frame(&data)?;

println!("Frame: {}x{}", frame.width, frame.height);
println!("Profile: {:?}", frame.profile);
```

### Encoding

```rust
use transcode_cineform::{CineformEncoder, EncoderConfig, Quality};

let config = EncoderConfig::new(1920, 1080)
    .with_quality(Quality::High)
    .with_bit_depth(BitDepth::Bit10);

let mut encoder = CineformEncoder::new(config)?;
let encoded = encoder.encode_frame(&frame)?;
```

## Pixel Formats

- YUV 4:2:2 (standard)
- RGBA/BGRA (for 4444 profiles)
- 10-bit and 12-bit variants

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
