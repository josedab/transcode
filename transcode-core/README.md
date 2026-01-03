# transcode-core

Core types and utilities for the Transcode codec library.

This crate provides fundamental building blocks used across all Transcode components.

## Features

- **Error Handling** - Comprehensive error hierarchy with `Error`, `CodecError`, `ContainerError`, and `BitstreamError`
- **Bitstream I/O** - Bit-level reading/writing with Exp-Golomb support for H.264/H.265 parsing
- **Frame Buffers** - Video frame abstractions with multiple pixel formats (YUV, RGB, NV12, etc.)
- **Sample Buffers** - Audio sample handling with planar/interleaved formats
- **Packets** - Encoded media data containers with timestamps and side data
- **Timestamps** - Precise time representation with automatic time base conversion
- **Rational Numbers** - Exact fraction arithmetic for frame rates and time bases

## Key Types

| Type | Description |
|------|-------------|
| `Error`, `Result<T>` | Main error type and result alias |
| `BitReader`, `BitWriter` | Bit-level stream access |
| `Frame`, `FrameBuffer` | Decoded video frames |
| `Sample`, `SampleBuffer` | Decoded audio samples |
| `Packet` | Encoded media packets |
| `Timestamp`, `Duration`, `TimeBase` | Time handling |
| `PixelFormat`, `SampleFormat` | Media format descriptors |

## Usage

```rust
use transcode_core::{
    BitReader, Frame, Packet, Timestamp, TimeBase,
    PixelFormat, Result,
};

// Parse a bitstream
fn parse_header(data: &[u8]) -> Result<u32> {
    let mut reader = BitReader::new(data);
    let sync_word = reader.read_bits(8)?;
    let exp_golomb_val = reader.read_ue()?;
    Ok(exp_golomb_val)
}

// Create a video frame
let frame = Frame::new(1920, 1080, PixelFormat::Yuv420p, TimeBase::MPEG);

// Work with timestamps
let pts = Timestamp::from_millis(1000);
let seconds = pts.to_seconds(); // Some(1.0)

// Handle packets
let packet = Packet::new(encoded_data)
    .with_timestamps(pts, pts)
    .with_stream_index(0);
```

## Error Handling

```rust
use transcode_core::error::{Error, CodecError, Result};

fn decode(data: &[u8]) -> Result<Frame> {
    if data.is_empty() {
        return Err(CodecError::BitstreamCorruption { offset: 0 }.into());
    }
    // ...
}
```

## Documentation

See the [main Transcode documentation](../README.md) for the complete library overview.

## License

See the repository root for license information.
