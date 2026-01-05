# transcode-vp8

A pure Rust implementation of the VP8 video codec.

## Features

- **Royalty-Free**: Open codec developed by On2/Google
- **Baseline Decoding**: Full VP8 specification support
- **Multiple Reference Frames**: Last, Golden, and AltRef frames
- **Boolean Arithmetic Coder**: Efficient entropy coding
- **Loop Filtering**: In-loop deblocking filter
- **WebM Compatible**: Native WebM container support

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-vp8 = { path = "../transcode-vp8" }
```

### Decoding

```rust
use transcode_vp8::{Vp8Decoder, Vp8Frame};

let mut decoder = Vp8Decoder::new()?;

// Decode compressed frame
let frame = decoder.decode(&compressed_data)?;
println!("Frame: {}x{}", frame.width, frame.height);
println!("Frame type: {:?}", frame.frame_type);
```

### Encoding (with `encoder` feature)

```rust
use transcode_vp8::{Vp8Encoder, Vp8EncoderConfig, RateControlMode};

let config = Vp8EncoderConfig::new(1280, 720)
    .with_bitrate(2_000_000)
    .with_keyframe_interval(150)
    .with_rate_control(RateControlMode::Vbr);

let mut encoder = Vp8Encoder::new(config)?;
let packet = encoder.encode(&frame)?;
```

## Frame Types

| Type | Description |
|------|-------------|
| KeyFrame | Independent frame (I-frame) |
| InterFrame | Predicted frame (P-frame) |

## Prediction Modes

### Macroblock Modes (Luma)
- DC_PRED: DC prediction
- V_PRED: Vertical prediction
- H_PRED: Horizontal prediction
- TM_PRED: True motion prediction
- B_PRED: Sub-block prediction (16 modes)

### Chroma Modes
- DC_PRED, V_PRED, H_PRED, TM_PRED

## Feature Flags

- `encoder`: Enable VP8 encoding support

## Container Support

VP8 is commonly used with:
- WebM container (native)
- Matroska (MKV)
- RTMP/WebRTC streaming

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
