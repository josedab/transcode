---
sidebar_position: 1
title: Custom Codecs
description: Implement custom video and audio codecs
---

# Custom Codecs

This guide explains how to implement custom video and audio codecs for Transcode.

## Overview

Transcode uses a trait-based architecture for codecs. To add a new codec, implement the appropriate traits from `transcode-codecs`.

## Codec Traits

### Video Decoder

```rust
use transcode_codecs::{VideoDecoder, CodecInfo};
use transcode_core::{Frame, Packet, Result};

pub trait VideoDecoder: CodecInfo {
    /// Decode a packet into a frame
    fn decode(&mut self, packet: &Packet) -> Result<Option<Frame>>;

    /// Flush remaining frames after all packets are processed
    fn flush(&mut self) -> Result<Vec<Frame>>;

    /// Reset decoder state
    fn reset(&mut self) -> Result<()>;
}
```

### Video Encoder

```rust
use transcode_codecs::{VideoEncoder, CodecInfo};
use transcode_core::{Frame, Packet, Result};

pub trait VideoEncoder: CodecInfo {
    /// Encode a frame into packets
    fn encode(&mut self, frame: &Frame) -> Result<Vec<Packet>>;

    /// Flush remaining packets after all frames are processed
    fn flush(&mut self) -> Result<Vec<Packet>>;

    /// Reset encoder state
    fn reset(&mut self) -> Result<()>;
}
```

### Audio Decoder

```rust
use transcode_codecs::{AudioDecoder, CodecInfo};
use transcode_core::{SampleBuffer, Packet, Result};

pub trait AudioDecoder: CodecInfo {
    /// Decode a packet into audio samples
    fn decode(&mut self, packet: &Packet) -> Result<Option<SampleBuffer>>;

    /// Flush remaining samples
    fn flush(&mut self) -> Result<Vec<SampleBuffer>>;

    /// Reset decoder state
    fn reset(&mut self) -> Result<()>;
}
```

### Audio Encoder

```rust
use transcode_codecs::{AudioEncoder, CodecInfo};
use transcode_core::{SampleBuffer, Packet, Result};

pub trait AudioEncoder: CodecInfo {
    /// Encode samples into packets
    fn encode(&mut self, samples: &SampleBuffer) -> Result<Vec<Packet>>;

    /// Flush remaining packets
    fn flush(&mut self) -> Result<Vec<Packet>>;

    /// Reset encoder state
    fn reset(&mut self) -> Result<()>;
}
```

### Codec Info

```rust
pub trait CodecInfo {
    /// Codec name (e.g., "h264", "aac")
    fn name(&self) -> &str;

    /// Codec description
    fn description(&self) -> &str;

    /// Codec type (video or audio)
    fn codec_type(&self) -> CodecType;
}
```

## Example: Custom Video Decoder

Here's a complete example of implementing a custom video decoder:

```rust
use transcode_codecs::{VideoDecoder, CodecInfo, CodecType};
use transcode_core::{Frame, Packet, PixelFormat, Result};
use transcode_core::error::CodecError;

/// Configuration for the custom decoder
pub struct CustomDecoderConfig {
    pub width: u32,
    pub height: u32,
}

/// Custom video decoder implementation
pub struct CustomDecoder {
    config: CustomDecoderConfig,
    frame_count: u64,
    // Internal decoder state
    state: DecoderState,
}

struct DecoderState {
    initialized: bool,
    // Add your decoder-specific state here
}

impl CustomDecoder {
    /// Create a new decoder with the given configuration
    pub fn new(config: CustomDecoderConfig) -> Result<Self> {
        Ok(Self {
            config,
            frame_count: 0,
            state: DecoderState {
                initialized: false,
            },
        })
    }

    /// Initialize decoder from codec parameters in stream
    pub fn from_extradata(extradata: &[u8]) -> Result<Self> {
        // Parse extradata to get configuration
        let (width, height) = parse_extradata(extradata)?;

        Self::new(CustomDecoderConfig { width, height })
    }

    /// Internal decoding logic
    fn decode_internal(&mut self, data: &[u8]) -> Result<Frame> {
        // Your decoding implementation here
        // This is where you'd implement the actual codec logic

        // Example: create output frame
        let mut frame = Frame::new(
            self.config.width,
            self.config.height,
            PixelFormat::Yuv420p,
        );

        // Decode data into frame planes
        // frame.plane_mut(0) - Y plane
        // frame.plane_mut(1) - U plane
        // frame.plane_mut(2) - V plane

        self.frame_count += 1;
        frame.set_pts(Some(self.frame_count as i64));

        Ok(frame)
    }
}

impl CodecInfo for CustomDecoder {
    fn name(&self) -> &str {
        "custom"
    }

    fn description(&self) -> &str {
        "Custom Video Codec"
    }

    fn codec_type(&self) -> CodecType {
        CodecType::Video
    }
}

impl VideoDecoder for CustomDecoder {
    fn decode(&mut self, packet: &Packet) -> Result<Option<Frame>> {
        // Initialize on first packet if needed
        if !self.state.initialized {
            self.initialize()?;
            self.state.initialized = true;
        }

        // Handle empty packet (flush signal)
        if packet.data().is_empty() {
            return Ok(None);
        }

        // Decode the packet
        let frame = self.decode_internal(packet.data())?;
        Ok(Some(frame))
    }

    fn flush(&mut self) -> Result<Vec<Frame>> {
        // Return any buffered frames
        // Most decoders buffer frames for reference
        let mut frames = Vec::new();

        // Flush your internal buffers
        // while let Some(frame) = self.flush_internal()? {
        //     frames.push(frame);
        // }

        Ok(frames)
    }

    fn reset(&mut self) -> Result<()> {
        self.state.initialized = false;
        self.frame_count = 0;
        // Reset internal state
        Ok(())
    }
}

fn parse_extradata(extradata: &[u8]) -> Result<(u32, u32)> {
    // Parse your codec's extradata format
    // Return width and height
    Ok((1920, 1080))
}

impl CustomDecoder {
    fn initialize(&mut self) -> Result<()> {
        // One-time initialization
        Ok(())
    }
}
```

## Example: Custom Video Encoder

```rust
use transcode_codecs::{VideoEncoder, CodecInfo, CodecType};
use transcode_core::{Frame, Packet, Result};

pub struct CustomEncoderConfig {
    pub width: u32,
    pub height: u32,
    pub bitrate: u32,
    pub fps: f64,
    pub keyframe_interval: u32,
}

impl Default for CustomEncoderConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            bitrate: 5_000_000,
            fps: 30.0,
            keyframe_interval: 250,
        }
    }
}

pub struct CustomEncoder {
    config: CustomEncoderConfig,
    frame_count: u64,
    // Internal state
}

impl CustomEncoder {
    pub fn new(config: CustomEncoderConfig) -> Result<Self> {
        Ok(Self {
            config,
            frame_count: 0,
        })
    }

    /// Get codec extradata (SPS/PPS, etc.)
    pub fn extradata(&self) -> Vec<u8> {
        // Return codec configuration data
        // This is written to the container header
        Vec::new()
    }

    fn encode_internal(&mut self, frame: &Frame) -> Result<Vec<u8>> {
        // Your encoding implementation
        // Access frame data:
        let y_plane = frame.plane(0);
        let u_plane = frame.plane(1);
        let v_plane = frame.plane(2);

        // Encode and return compressed data
        Ok(Vec::new())
    }

    fn is_keyframe(&self) -> bool {
        self.frame_count % self.config.keyframe_interval as u64 == 0
    }
}

impl CodecInfo for CustomEncoder {
    fn name(&self) -> &str {
        "custom"
    }

    fn description(&self) -> &str {
        "Custom Video Codec Encoder"
    }

    fn codec_type(&self) -> CodecType {
        CodecType::Video
    }
}

impl VideoEncoder for CustomEncoder {
    fn encode(&mut self, frame: &Frame) -> Result<Vec<Packet>> {
        let encoded_data = self.encode_internal(frame)?;

        let mut packet = Packet::new(encoded_data);
        packet.set_pts(frame.pts());
        packet.set_dts(frame.pts()); // May differ for B-frames
        packet.set_keyframe(self.is_keyframe());

        self.frame_count += 1;

        Ok(vec![packet])
    }

    fn flush(&mut self) -> Result<Vec<Packet>> {
        // Flush any buffered frames
        // B-frame encoders typically buffer several frames
        Ok(Vec::new())
    }

    fn reset(&mut self) -> Result<()> {
        self.frame_count = 0;
        Ok(())
    }
}
```

## Example: Custom Audio Codec

```rust
use transcode_codecs::{AudioDecoder, AudioEncoder, CodecInfo, CodecType};
use transcode_core::{SampleBuffer, SampleFormat, Packet, Result};

pub struct CustomAudioDecoderConfig {
    pub sample_rate: u32,
    pub channels: u8,
}

pub struct CustomAudioDecoder {
    config: CustomAudioDecoderConfig,
}

impl CustomAudioDecoder {
    pub fn new(config: CustomAudioDecoderConfig) -> Result<Self> {
        Ok(Self { config })
    }
}

impl CodecInfo for CustomAudioDecoder {
    fn name(&self) -> &str {
        "custom_audio"
    }

    fn description(&self) -> &str {
        "Custom Audio Codec"
    }

    fn codec_type(&self) -> CodecType {
        CodecType::Audio
    }
}

impl AudioDecoder for CustomAudioDecoder {
    fn decode(&mut self, packet: &Packet) -> Result<Option<SampleBuffer>> {
        // Decode packet into samples
        let frame_size = 1024; // samples per channel

        let mut buffer = SampleBuffer::new(
            frame_size,
            self.config.channels as usize,
            SampleFormat::Float32,
        );

        // Decode into buffer
        // let samples: &mut [f32] = buffer.as_slice_mut();

        buffer.set_sample_rate(self.config.sample_rate);
        buffer.set_pts(packet.pts());

        Ok(Some(buffer))
    }

    fn flush(&mut self) -> Result<Vec<SampleBuffer>> {
        Ok(Vec::new())
    }

    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}
```

## Registering Custom Codecs

Register your codec so it can be discovered by name:

```rust
use transcode_codecs::registry::{CodecRegistry, CodecFactory};

// Create codec factory
struct CustomDecoderFactory;

impl CodecFactory<dyn VideoDecoder> for CustomDecoderFactory {
    fn create(&self, params: &CodecParams) -> Result<Box<dyn VideoDecoder>> {
        let config = CustomDecoderConfig {
            width: params.width,
            height: params.height,
        };
        Ok(Box::new(CustomDecoder::new(config)?))
    }
}

// Register with the codec registry
fn register_custom_codec() {
    let registry = CodecRegistry::global();

    registry.register_decoder(
        "custom",
        Box::new(CustomDecoderFactory),
    );
}
```

## Using Custom Codecs

### Direct Usage

```rust
use my_codec::{CustomDecoder, CustomDecoderConfig};

let config = CustomDecoderConfig {
    width: 1920,
    height: 1080,
};

let mut decoder = CustomDecoder::new(config)?;

for packet in packets {
    if let Some(frame) = decoder.decode(&packet)? {
        // Process frame
    }
}

for frame in decoder.flush()? {
    // Process remaining frames
}
```

### In Pipeline

```rust
use transcode_pipeline::PipelineBuilder;
use my_codec::CustomDecoder;

let pipeline = PipelineBuilder::new()
    .demuxer(demuxer)
    .video_decoder(CustomDecoder::new(config)?)
    .video_encoder(encoder)
    .muxer(muxer)
    .build()?;

pipeline.run()?;
```

## Bitstream Utilities

Use `transcode-core` bitstream utilities for parsing:

```rust
use transcode_core::bitstream::BitReader;

fn parse_header(data: &[u8]) -> Result<Header> {
    let mut reader = BitReader::new(data);

    let magic = reader.read_bits(32)?;
    let version = reader.read_bits(8)?;
    let flags = reader.read_bits(16)?;

    // Read variable-length fields
    let width = reader.read_ue()?;   // Exp-Golomb
    let height = reader.read_ue()?;

    // Read signed values
    let offset = reader.read_se()?;  // Signed Exp-Golomb

    Ok(Header { magic, version, flags, width, height, offset })
}
```

### BitWriter for Encoding

```rust
use transcode_core::bitstream::BitWriter;

fn write_header(header: &Header) -> Vec<u8> {
    let mut writer = BitWriter::new();

    writer.write_bits(header.magic, 32);
    writer.write_bits(header.version, 8);
    writer.write_bits(header.flags, 16);

    writer.write_ue(header.width);
    writer.write_ue(header.height);
    writer.write_se(header.offset);

    writer.finish()
}
```

## Testing Custom Codecs

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        // Create encoder and decoder
        let encoder_config = CustomEncoderConfig::default();
        let mut encoder = CustomEncoder::new(encoder_config).unwrap();

        let decoder_config = CustomDecoderConfig {
            width: 1920,
            height: 1080,
        };
        let mut decoder = CustomDecoder::new(decoder_config).unwrap();

        // Create test frame
        let input_frame = Frame::new(1920, 1080, PixelFormat::Yuv420p);
        // Fill with test pattern...

        // Encode
        let packets = encoder.encode(&input_frame).unwrap();
        assert!(!packets.is_empty());

        // Decode
        let output_frame = decoder.decode(&packets[0]).unwrap().unwrap();

        // Verify (may not be bit-exact for lossy codecs)
        assert_eq!(input_frame.width(), output_frame.width());
        assert_eq!(input_frame.height(), output_frame.height());
    }

    #[test]
    fn test_flush() {
        let mut decoder = CustomDecoder::new(config).unwrap();

        // Decode some packets
        for packet in packets {
            decoder.decode(&packet).unwrap();
        }

        // Flush should return buffered frames
        let flushed = decoder.flush().unwrap();
        assert!(flushed.len() >= expected_buffered);
    }
}
```

## Next Steps

- [SIMD Optimization](/docs/advanced/simd-optimization) - Optimize codec performance
- [Hardware Acceleration](/docs/advanced/hardware-acceleration) - GPU encoding
- [API Reference](/docs/reference/api) - Full API documentation
