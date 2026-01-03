# transcode-vp9

VP9 video codec implementation for the transcode library.

A pure Rust implementation of the VP9 video codec with decoder support for profiles 0-3.

## Features

- **Profile Support**: Profiles 0, 1, 2, and 3 (8-bit, 10-bit, 12-bit)
- **Superblock Structure**: 64x64 superblocks with recursive partitioning
- **Reference Frames**: LAST, GOLDEN, ALTREF reference frame management
- **Transform Types**: DCT, ADST, and identity transforms (4x4 to 32x32)
- **Loop Filter**: Deblocking filter with delta and sharpness support
- **Segmentation**: Up to 8 segments with feature data
- **Boolean Entropy Coding**: Arithmetic coding for compressed data
- **Superframe Support**: Automatic parsing of multiple frames per packet

## Supported Color Spaces

- BT.601 (SD video)
- BT.709 (HD video)
- BT.2020 (UHD/HDR video)
- sRGB (screen content)

## Chroma Subsampling

- 4:2:0 (Profile 0/2)
- 4:2:2 and 4:4:4 (Profile 1/3)

## Key Types

### Decoder

- `Vp9Decoder` - Main decoder struct
- `Vp9DecoderConfig` - Configuration (threads, error concealment, loop filter)

### Frame Header

- `FrameHeader` - Parsed frame header with all parameters
- `Profile` - VP9 profile (0-3)
- `FrameType` - Keyframe or Inter
- `ColorSpace`, `ColorRange`, `ChromaSubsampling`
- `LoopFilterParams`, `QuantParams`, `SegmentationParams`

### Prediction

- `BlockSize` - Block sizes from 4x4 to 64x64
- `IntraMode` - DC, V, H, D45, D135, D117, D153, D207, D63, TM
- `InterMode` - ZeroMv, NearestMv, NearMv, NewMv
- `MotionVector` - Quarter-pel motion vectors
- `TxSize`, `TxType` - Transform sizes and types

### Entropy Coding

- `BoolDecoder` - Boolean arithmetic decoder
- `ProbabilityContext` - Probability tables for entropy coding

### Errors

- `Vp9Error` - Comprehensive error types with recovery info
- `Result<T>` - VP9-specific result type

## Usage

### Basic Decoding

```rust
use transcode_vp9::{Vp9Decoder, Vp9DecoderConfig};

let config = Vp9DecoderConfig::default();
let mut decoder = Vp9Decoder::new(config);

// Decode VP9 frame data
let frames = decoder.decode_frame(vp9_data)?;
for frame in frames {
    // Process decoded frame
    println!("Frame {}x{}", frame.width(), frame.height());
}
```

### Custom Configuration

```rust
use transcode_vp9::{Vp9Decoder, Vp9DecoderConfig};

let config = Vp9DecoderConfig {
    max_threads: 4,
    error_concealment: true,
    output_pool_size: 16,
    enable_loop_filter: true,
};
let mut decoder = Vp9Decoder::new(config);
```

### Checking Codec Info

```rust
use transcode_vp9::codec_info;

let info = codec_info();
assert_eq!(info.name, "vp9");
assert!(info.can_decode);
```

## VP9 Bitstream Structure

VP9 frames consist of:

1. **Uncompressed Header** - Fixed-length fields: frame type, dimensions, reference info, loop filter, quantization
2. **Compressed Header** - Probability updates using boolean entropy coding
3. **Tile Data** - Coded video data split into tiles for parallel decoding

## Optional Features

- `simd` - Enable SIMD optimizations (when available)

## License

See the workspace root for license information.
