# transcode-core

Core types and utilities for the Transcode library.

## Overview

`transcode-core` provides fundamental types used throughout the Transcode ecosystem:

- Bitstream reading/writing utilities
- Frame and packet types
- Error handling primitives
- Common data structures

## Crate Features

```toml
[dependencies]
transcode-core = { version = "1.0", features = ["simd", "async"] }
```

| Feature | Description |
|---------|-------------|
| `simd` | Enable SIMD-optimized implementations |
| `async` | Enable async/await support |
| `serde` | Serialization support |

## Modules

### bitstream

Bit-level reading and writing:

```rust
use transcode_core::bitstream::BitReader;

let data = &[0b10110100, 0b11001010];
let mut reader = BitReader::new(data);

// Read individual bits
let bit = reader.read_bit()?;           // 1

// Read multiple bits
let value = reader.read_bits(4)?;       // 0b0110 = 6

// Read Exp-Golomb codes (H.264)
let ue = reader.read_ue()?;             // Unsigned
let se = reader.read_se()?;             // Signed

// Check remaining bits
println!("Remaining: {} bits", reader.remaining_bits());
```

### frame

Video frame representation:

```rust
use transcode_core::frame::{Frame, PixelFormat};

// Create frame
let frame = Frame::new(1920, 1080, PixelFormat::Yuv420p);

// Access planes
let y_plane = frame.plane(0);
let u_plane = frame.plane(1);
let v_plane = frame.plane(2);

// Get frame info
println!("{}x{} {:?}", frame.width(), frame.height(), frame.format());
println!("PTS: {}", frame.pts());
```

### sample

Audio sample representation:

```rust
use transcode_core::sample::{Sample, SampleBuffer, SampleFormat};

// Create sample buffer
let buffer = SampleBuffer::new(1024, 2, SampleFormat::F32);

// Access samples
let samples = buffer.samples();
let left_channel = buffer.channel(0);
let right_channel = buffer.channel(1);
```

### packet

Encoded data packets:

```rust
use transcode_core::packet::Packet;

let packet = Packet::builder()
    .data(encoded_data)
    .pts(0)
    .dts(0)
    .duration(1001)
    .keyframe(true)
    .build();

println!("Size: {} bytes", packet.data().len());
println!("Keyframe: {}", packet.is_keyframe());
```

### error

Error handling:

```rust
use transcode_core::error::{Error, Result, CodecError};

fn decode(data: &[u8]) -> Result<Frame> {
    if data.is_empty() {
        return Err(CodecError::BitstreamCorruption {
            message: "Empty input".into(),
        }.into());
    }
    // ...
}
```

## Types

### PixelFormat

Supported pixel formats:

```rust
pub enum PixelFormat {
    Yuv420p,    // Planar YUV 4:2:0
    Yuv422p,    // Planar YUV 4:2:2
    Yuv444p,    // Planar YUV 4:4:4
    Nv12,       // Semi-planar YUV 4:2:0
    Rgb24,      // Packed RGB
    Rgba32,     // Packed RGBA
    Bgra32,     // Packed BGRA
    Gray8,      // Grayscale
}
```

### SampleFormat

Audio sample formats:

```rust
pub enum SampleFormat {
    U8,         // Unsigned 8-bit
    S16,        // Signed 16-bit
    S32,        // Signed 32-bit
    F32,        // 32-bit float
    F64,        // 64-bit float
    S16p,       // Planar signed 16-bit
    F32p,       // Planar 32-bit float
}
```

### Rational

Rational numbers for timestamps:

```rust
use transcode_core::Rational;

let fps = Rational::new(30000, 1001);  // 29.97 fps
let timebase = Rational::new(1, 90000);

println!("FPS: {:.2}", fps.to_f64());
```

## Full API Reference

See the [rustdoc documentation](https://docs.rs/transcode-core) for complete API details.
