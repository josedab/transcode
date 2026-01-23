---
sidebar_position: 3
title: Frames & Packets
description: Understanding raw frames and compressed packets
---

# Frames & Packets

The two fundamental data types in video processing are **frames** (decoded/raw data) and **packets** (encoded/compressed data).

## Video Frames

A frame represents a single image in a video sequence.

### Frame Structure

```rust
pub struct Frame {
    /// Frame width in pixels
    pub width: u32,
    /// Frame height in pixels
    pub height: u32,
    /// Pixel format (YUV420, RGB, etc.)
    pub pixel_format: PixelFormat,
    /// Color space (BT.709, BT.2020, etc.)
    pub color_space: ColorSpace,
    /// Color range (full or limited)
    pub color_range: ColorRange,
    /// Presentation timestamp
    pub timestamp: Timestamp,
    /// Raw pixel data
    pub data: FrameBuffer,
}
```

### Creating Frames

```rust
use transcode::{Frame, PixelFormat, ColorSpace};

// Create a blank frame
let frame = Frame::new(1920, 1080, PixelFormat::Yuv420p);

// Create from existing data
let frame = Frame::from_data(
    1920,
    1080,
    PixelFormat::Yuv420p,
    yuv_data,
)?;
```

### Pixel Formats

Video frames use different pixel formats depending on the stage of processing:

| Format | Description | Use Case |
|--------|-------------|----------|
| `Yuv420p` | YUV 4:2:0 planar | Most video codecs |
| `Yuv422p` | YUV 4:2:2 planar | Professional video |
| `Yuv444p` | YUV 4:4:4 planar | High quality |
| `Rgb24` | RGB 8-bit | Display, image processing |
| `Rgba32` | RGBA 8-bit | Compositing |
| `Nv12` | YUV 4:2:0 semi-planar | Hardware codecs |

```rust
// Convert between formats
let rgb_frame = frame.convert(PixelFormat::Rgb24)?;
```

### Planar vs. Packed

**Planar** formats store each color channel separately:

```
Y plane:  [Y Y Y Y Y Y Y Y ...]
U plane:  [U U U U ...]
V plane:  [V V V V ...]
```

**Packed** formats interleave channels:

```
[Y U Y V Y U Y V ...]  (YUYV)
[R G B R G B R G B ...]  (RGB24)
```

### Accessing Pixel Data

```rust
// Get plane data (for planar formats)
let y_plane = frame.plane(0);  // Luma
let u_plane = frame.plane(1);  // Chroma U
let v_plane = frame.plane(2);  // Chroma V

// Get stride (bytes per row)
let y_stride = frame.stride(0);

// Access specific pixel
let pixel = frame.get_pixel(x, y)?;
```

## Audio Samples

Audio is represented as sample buffers:

```rust
pub struct SampleBuffer {
    /// Number of samples per channel
    pub num_samples: usize,
    /// Number of channels
    pub channels: usize,
    /// Sample format
    pub format: SampleFormat,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Channel layout
    pub layout: ChannelLayout,
    /// Raw audio data
    pub data: Vec<f32>,
}
```

### Sample Formats

| Format | Description | Bits |
|--------|-------------|------|
| `S16` | Signed 16-bit integer | 16 |
| `S32` | Signed 32-bit integer | 32 |
| `F32` | 32-bit float | 32 |
| `F64` | 64-bit float | 64 |

```rust
use transcode::{SampleBuffer, SampleFormat, ChannelLayout};

let samples = SampleBuffer::new(
    1024,                    // samples per channel
    2,                       // stereo
    SampleFormat::F32,
    48000,                   // 48 kHz
    ChannelLayout::Stereo,
);
```

## Packets

Packets contain compressed (encoded) data.

### Packet Structure

```rust
pub struct Packet {
    /// Compressed data
    pub data: Vec<u8>,
    /// Presentation timestamp (when to display)
    pub pts: Timestamp,
    /// Decode timestamp (when to decode)
    pub dts: Option<Timestamp>,
    /// Packet flags
    pub flags: PacketFlags,
    /// Stream index in container
    pub stream_index: usize,
    /// Duration
    pub duration: Option<Duration>,
}
```

### Packet Flags

```rust
bitflags! {
    pub struct PacketFlags: u32 {
        const KEYFRAME = 0x0001;      // Can be decoded independently
        const CORRUPT = 0x0002;       // Data may be corrupt
        const DISCARD = 0x0004;       // Can be discarded
        const TRUSTED = 0x0008;       // Trusted keyframe
        const DISPOSABLE = 0x0010;    // Not needed for decoding others
    }
}

// Check if packet is a keyframe
if packet.flags.contains(PacketFlags::KEYFRAME) {
    println!("This is a keyframe");
}
```

### PTS vs DTS

- **PTS** (Presentation Timestamp): When the frame should be displayed
- **DTS** (Decode Timestamp): When the frame should be decoded

For codecs with B-frames, decode order differs from display order:

```
Decode order:  I0  P3  B1  B2  P6  B4  B5
Display order: I0  B1  B2  P3  B4  B5  P6
               ↑              ↑
              PTS=0         PTS=3
              DTS=0         DTS=1
```

## Timestamps

Timestamps use a time base for precision:

```rust
pub struct Timestamp {
    pub pts: i64,
    pub time_base: TimeBase,
}

pub struct TimeBase {
    pub num: i32,  // numerator
    pub den: i32,  // denominator
}
```

### Working with Timestamps

```rust
// Create timestamp
let ts = Timestamp::new(90000, TimeBase::new(1, 90000));  // 1 second

// Convert to seconds
let seconds = ts.as_seconds();  // 1.0

// Convert between time bases
let new_ts = ts.rescale(TimeBase::new(1, 1000));  // milliseconds
```

### Common Time Bases

| Time Base | Usage |
|-----------|-------|
| 1/90000 | MPEG-TS |
| 1/1000 | Milliseconds |
| 1/framerate | Frame-accurate |

## Frame Types

Video codecs use different frame types:

### I-Frame (Keyframe)

- Complete picture
- Can be decoded independently
- Largest size
- Required for seeking

### P-Frame (Predicted)

- Predicted from previous frames
- Smaller than I-frames
- Requires previous frames to decode

### B-Frame (Bidirectional)

- Predicted from previous AND future frames
- Smallest size
- Most complex to decode

```
GOP (Group of Pictures):
I   B   B   P   B   B   P   B   B   I
↑                                   ↑
Keyframe                        Keyframe
```

### Checking Frame Type

```rust
// After decoding
if frame.is_keyframe() {
    println!("Keyframe at {}", frame.timestamp.as_seconds());
}

// From packet (before decoding)
if packet.flags.contains(PacketFlags::KEYFRAME) {
    println!("Keyframe packet");
}
```

## Memory Management

### Frame Pools

Reuse frame buffers to reduce allocation:

```rust
use transcode_core::pool::FramePool;

// Create pool with 10 frames
let pool = FramePool::new(10, 1920, 1080, PixelFormat::Yuv420p);

// Get frame from pool (reuses buffer)
let frame = pool.get()?;

// Frame automatically returns to pool when dropped
```

### Zero-Copy Operations

Avoid copying when possible:

```rust
// Share data between frame and GPU texture
let texture = GpuTexture::from_frame_view(&frame)?;

// Decode directly into output frame
decoder.decode_into(&packet, &mut output_frame)?;
```

## Next Steps

- [Pipeline](/docs/core-concepts/pipeline) - Connecting decoders and encoders
- [GPU Acceleration](/docs/guides/gpu-acceleration) - Process frames on GPU
- [Quality Metrics](/docs/guides/quality-metrics) - Compare frame quality
