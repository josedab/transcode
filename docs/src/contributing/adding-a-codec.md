# Adding a Codec

This guide walks through adding a new codec to Transcode.

## Overview

Adding a codec involves:

1. Creating the module structure
2. Implementing codec traits
3. Adding bitstream parsing
4. Implementing encode/decode logic
5. Adding SIMD optimizations
6. Writing tests

## Module Structure

Create the codec module under the appropriate directory:

```
transcode-codecs/src/video/mycodec/
├── mod.rs          # Public API, re-exports
├── decoder.rs      # Decoder implementation
├── encoder.rs      # Encoder implementation (if applicable)
├── bitstream.rs    # Bitstream parsing utilities
├── types.rs        # Codec-specific types
└── simd/           # SIMD optimizations
    ├── mod.rs
    ├── x86_64.rs
    └── aarch64.rs
```

## Step 1: Define Types

```rust
// types.rs

use transcode_core::{PixelFormat, Rational};

/// Codec configuration
#[derive(Debug, Clone)]
pub struct MyCodecConfig {
    pub width: u32,
    pub height: u32,
    pub framerate: Rational,
    pub bitrate: Option<u32>,
}

/// Frame header information
#[derive(Debug)]
pub struct FrameHeader {
    pub frame_type: FrameType,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameType {
    Key,
    Inter,
}
```

## Step 2: Implement Bitstream Parsing

```rust
// bitstream.rs

use transcode_core::bitstream::BitReader;
use crate::error::{CodecError, Result};
use super::types::*;

pub fn parse_frame_header(reader: &mut BitReader) -> Result<FrameHeader> {
    // Read sync code
    let sync = reader.read_bits(24)?;
    if sync != 0x4D5943 {  // "MYC"
        return Err(CodecError::BitstreamCorruption {
            message: "Invalid sync code".into(),
        }.into());
    }

    // Read frame type
    let frame_type = if reader.read_bit()? {
        FrameType::Key
    } else {
        FrameType::Inter
    };

    // Read dimensions
    let width = reader.read_bits(16)? as u32;
    let height = reader.read_bits(16)? as u32;

    Ok(FrameHeader {
        frame_type,
        width,
        height,
    })
}
```

## Step 3: Implement Decoder

```rust
// decoder.rs

use transcode_core::{Frame, Packet, PixelFormat};
use transcode_codecs::traits::{VideoDecoder, CodecInfo};
use crate::error::Result;
use super::{bitstream, types::*};

pub struct MyCodecDecoder {
    config: Option<MyCodecConfig>,
    reference_frame: Option<Frame>,
}

impl MyCodecDecoder {
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: None,
            reference_frame: None,
        })
    }
}

impl CodecInfo for MyCodecDecoder {
    fn codec_id(&self) -> &str {
        "mycodec"
    }

    fn name(&self) -> &str {
        "My Custom Codec"
    }

    fn supported_formats(&self) -> &[PixelFormat] {
        &[PixelFormat::Yuv420p]
    }

    fn simd_features(&self) -> SimdFeatures {
        detect_simd()
    }
}

impl VideoDecoder for MyCodecDecoder {
    fn decode(&mut self, packet: &Packet) -> Result<Option<Frame>> {
        let mut reader = BitReader::new(packet.data());

        // Parse header
        let header = bitstream::parse_frame_header(&mut reader)?;

        // Initialize config on first frame
        if self.config.is_none() {
            self.config = Some(MyCodecConfig {
                width: header.width,
                height: header.height,
                framerate: Rational::new(30, 1),
                bitrate: None,
            });
        }

        // Allocate output frame
        let mut frame = Frame::new(
            header.width,
            header.height,
            PixelFormat::Yuv420p,
        );

        // Decode frame data
        match header.frame_type {
            FrameType::Key => {
                self.decode_key_frame(&mut reader, &mut frame)?;
                self.reference_frame = Some(frame.clone());
            }
            FrameType::Inter => {
                let reference = self.reference_frame.as_ref()
                    .ok_or(CodecError::NotReady {
                        reason: "No reference frame".into()
                    })?;
                self.decode_inter_frame(&mut reader, reference, &mut frame)?;
                self.reference_frame = Some(frame.clone());
            }
        }

        Ok(Some(frame))
    }

    fn flush(&mut self) -> Result<Vec<Frame>> {
        // Return any buffered frames
        Ok(vec![])
    }

    fn reset(&mut self) {
        self.reference_frame = None;
    }

    fn dimensions(&self) -> (u32, u32) {
        self.config.as_ref()
            .map(|c| (c.width, c.height))
            .unwrap_or((0, 0))
    }

    fn pixel_format(&self) -> PixelFormat {
        PixelFormat::Yuv420p
    }
}

impl MyCodecDecoder {
    fn decode_key_frame(
        &self,
        reader: &mut BitReader,
        frame: &mut Frame,
    ) -> Result<()> {
        // Decode intra-coded frame
        for y in 0..frame.height() {
            for x in 0..frame.width() {
                let value = reader.read_bits(8)? as u8;
                frame.set_pixel(x, y, 0, value);
            }
        }
        Ok(())
    }

    fn decode_inter_frame(
        &self,
        reader: &mut BitReader,
        reference: &Frame,
        frame: &mut Frame,
    ) -> Result<()> {
        // Decode inter-coded frame with motion compensation
        // ... implementation
        Ok(())
    }
}
```

## Step 4: Implement Encoder

```rust
// encoder.rs

use transcode_core::{Frame, Packet};
use transcode_codecs::traits::{VideoEncoder, CodecInfo};
use crate::error::Result;

pub struct MyCodecEncoder {
    config: MyCodecConfig,
    frame_count: u64,
}

impl MyCodecEncoder {
    pub fn new(config: MyCodecConfig) -> Result<Self> {
        Ok(Self {
            config,
            frame_count: 0,
        })
    }
}

impl VideoEncoder for MyCodecEncoder {
    fn encode(&mut self, frame: &Frame) -> Result<Vec<Packet>> {
        let mut data = Vec::new();

        // Write header
        self.write_header(&mut data, frame)?;

        // Encode frame data
        if self.frame_count % 30 == 0 {
            // Key frame every 30 frames
            self.encode_key_frame(&mut data, frame)?;
        } else {
            self.encode_inter_frame(&mut data, frame)?;
        }

        self.frame_count += 1;

        Ok(vec![Packet::new(data)])
    }

    fn flush(&mut self) -> Result<Vec<Packet>> {
        Ok(vec![])
    }

    fn config(&self) -> &EncoderConfig {
        &self.config.into()
    }
}
```

## Step 5: Add SIMD Optimizations

```rust
// simd/mod.rs

#[cfg(target_arch = "x86_64")]
mod x86_64;
#[cfg(target_arch = "aarch64")]
mod aarch64;

pub fn transform_block(block: &mut [i16; 64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { x86_64::transform_block_avx2(block) }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { aarch64::transform_block_neon(block) }
            return;
        }
    }

    transform_block_scalar(block)
}

fn transform_block_scalar(block: &mut [i16; 64]) {
    // Scalar implementation
}
```

## Step 6: Export from Module

```rust
// mod.rs

//! My Custom Codec implementation.
//!
//! This module provides encode and decode support for MyCodec.

mod bitstream;
mod decoder;
mod encoder;
mod types;
mod simd;

pub use decoder::MyCodecDecoder;
pub use encoder::MyCodecEncoder;
pub use types::{MyCodecConfig, FrameType};
```

## Step 7: Register Codec

```rust
// In transcode-codecs/src/video/mod.rs

pub mod mycodec;

// In codec registry
pub fn register_codecs(registry: &mut CodecRegistry) {
    registry.register_video_decoder("mycodec", || {
        Box::new(mycodec::MyCodecDecoder::new()?)
    });
    registry.register_video_encoder("mycodec", |config| {
        Box::new(mycodec::MyCodecEncoder::new(config)?)
    });
}
```

## Step 8: Write Tests

```rust
// decoder.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_key_frame() {
        let mut decoder = MyCodecDecoder::new().unwrap();
        let packet = create_test_key_frame();

        let frame = decoder.decode(&packet).unwrap().unwrap();

        assert_eq!(frame.width(), 320);
        assert_eq!(frame.height(), 240);
    }

    #[test]
    fn test_decode_invalid_sync() {
        let mut decoder = MyCodecDecoder::new().unwrap();
        let packet = Packet::new(vec![0xFF; 100]);

        let result = decoder.decode(&packet);

        assert!(matches!(
            result,
            Err(CodecError::BitstreamCorruption { .. })
        ));
    }
}
```

## Checklist

- [ ] Module structure created
- [ ] Types defined
- [ ] Bitstream parsing implemented
- [ ] Decoder implements `VideoDecoder` trait
- [ ] Encoder implements `VideoEncoder` trait (if applicable)
- [ ] SIMD optimizations for hot paths
- [ ] Unit tests for all public functions
- [ ] Integration tests with test vectors
- [ ] Documentation with examples
- [ ] Codec registered in registry
- [ ] Feature flag added to `Cargo.toml`
