# transcode-prores

A pure Rust implementation of the Apple ProRes video codec.

## Features

- **All Profiles**: Proxy, LT, Standard, HQ, 4444, and 4444 XQ
- **10-bit/12-bit**: Professional bit depth support
- **Alpha Channel**: Full RGBA support for 4444 profiles
- **Slice-Based Decoding**: Parallel processing support
- **DCT Coefficient Decoding**: Huffman table support
- **Color Space Support**: Full colorimetry metadata

## Supported Profiles

| Profile | Quality | Chroma | Bit Depth |
|---------|---------|--------|-----------|
| ProRes 422 Proxy | Low | 4:2:2 | 10-bit |
| ProRes 422 LT | Medium | 4:2:2 | 10-bit |
| ProRes 422 | Standard | 4:2:2 | 10-bit |
| ProRes 422 HQ | High | 4:2:2 | 10-bit |
| ProRes 4444 | Very High | 4:4:4 | 12-bit |
| ProRes 4444 XQ | Maximum | 4:4:4 | 12-bit |

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-prores = { path = "../transcode-prores" }
```

### Decoding

```rust
use transcode_prores::{ProResDecoder, ProResFrame};

let data = std::fs::read("video.prores")?;
let mut decoder = ProResDecoder::new();
let frame = decoder.decode_frame(&data)?;

println!("Frame: {}x{}", frame.width, frame.height);
println!("Profile: {:?}", frame.profile);
```

### Encoding

```rust
use transcode_prores::{ProResEncoder, EncoderConfig, ProResProfile};

let config = EncoderConfig::new(1920, 1080)
    .with_profile(ProResProfile::Hq);

let mut encoder = ProResEncoder::new(config)?;
let encoded = encoder.encode_frame(&frame)?;
```

### Probing

```rust
use transcode_prores::{probe_prores, get_profile, get_dimensions};

if probe_prores(&data) {
    let profile = get_profile(&data)?;
    let (width, height) = get_dimensions(&data)?;
    println!("ProRes {} at {}x{}", profile, width, height);
}
```

## Color Space Support

- Color Primaries: BT.709, BT.601, P3, BT.2020
- Transfer Characteristics: SDR, HLG, PQ
- Matrix Coefficients: BT.709, BT.601, BT.2020

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
