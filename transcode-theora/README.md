# transcode-theora

A pure Rust implementation of the Theora video codec.

## Features

- **Royalty-Free**: Open, royalty-free codec from Xiph.Org
- **DCT-Based Compression**: Block-based lossy compression
- **Multiple Frame Types**: Intra, Predicted, and Golden frames
- **Huffman Entropy Coding**: Efficient coefficient encoding
- **YCbCr Support**: 4:2:0, 4:2:2, and 4:4:4 formats

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-theora = { path = "../transcode-theora" }
```

### Decoding

```rust
use transcode_theora::{TheoraDecoder, TheoraConfig};

let config = TheoraConfig::new(1920, 1080)?;
let mut decoder = TheoraDecoder::new(config)?;

// Decode frames
let frame = decoder.decode(&packet)?;
println!("Frame type: {:?}", frame.frame_type);
```

### Configuration

```rust
use transcode_theora::TheoraConfig;

let mut config = TheoraConfig::new(1280, 720)?;
config.set_framerate(30000, 1001);  // 29.97 fps
config.set_quality(48);              // 0-63 quality level
config.set_bitrate(2_000_000);       // Target bitrate
```

### Encoding (with `encoder` feature)

```rust
use transcode_theora::{TheoraEncoder, TheoraEncoderConfig};

let config = TheoraEncoderConfig::new(1280, 720)
    .with_quality(48)
    .with_keyframe_interval(250);

let mut encoder = TheoraEncoder::new(config)?;
let packet = encoder.encode(&frame)?;
```

## Pixel Formats

| Format | Subsampling |
|--------|-------------|
| YUV420 | 4:2:0 (default) |
| YUV422 | 4:2:2 |
| YUV444 | 4:4:4 |

## Color Spaces

- Unspecified
- ITU-R BT.601
- ITU-R BT.709

## Feature Flags

- `encoder`: Enable Theora encoding support

## Container Support

Theora is commonly used with:
- OGG container (native)
- Matroska (MKV)

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
