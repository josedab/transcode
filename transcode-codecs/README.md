# transcode-codecs

Video and audio codec implementations for the Transcode library.

## Supported Codecs

### Video

| Codec | Decode | Encode |
|-------|--------|--------|
| H.264/AVC | Yes | Yes |
| H.265/HEVC | Planned | Planned |
| VP9 | Planned | Planned |
| AV1 | Planned | Planned |

### Audio

| Codec | Decode | Encode |
|-------|--------|--------|
| AAC | Yes | Yes |
| MP3 | Yes | Yes |
| Opus | Planned | Planned |

## Key Traits

The crate provides four core traits for codec implementations:

```rust
use transcode_codecs::traits::{VideoDecoder, VideoEncoder, AudioDecoder, AudioEncoder};

/// Video decoder trait
pub trait VideoDecoder: Send {
    fn decode(&mut self, packet: &Packet) -> Result<Vec<Frame>>;
    fn flush(&mut self) -> Result<Vec<Frame>>;
    fn reset(&mut self);
}

/// Video encoder trait
pub trait VideoEncoder: Send {
    fn encode(&mut self, frame: &Frame) -> Result<Vec<Packet>>;
    fn flush(&mut self) -> Result<Vec<Packet>>;
    fn extra_data(&self) -> Option<&[u8]>;
}

/// Audio decoder trait
pub trait AudioDecoder: Send {
    fn decode(&mut self, packet: &Packet) -> Result<Vec<Sample>>;
    fn flush(&mut self) -> Result<Vec<Sample>>;
    fn set_extra_data(&mut self, data: &[u8]) -> Result<()>;
}

/// Audio encoder trait
pub trait AudioEncoder: Send {
    fn encode(&mut self, sample: &Sample) -> Result<Vec<Packet>>;
    fn flush(&mut self) -> Result<Vec<Packet>>;
    fn extra_data(&self) -> Option<&[u8]>;
}
```

## Usage

### H.264 Decoding

```rust
use transcode_codecs::video::H264Decoder;
use transcode_codecs::traits::VideoDecoder;

let mut decoder = H264Decoder::new_default();
decoder.set_extra_data(&avcc_data)?;

let frames = decoder.decode(&packet)?;
```

### H.264 Encoding

```rust
use transcode_codecs::video::h264::{H264Encoder, H264EncoderConfig, RateControlMode};

let config = H264EncoderConfig::new(1920, 1080)
    .with_frame_rate(30)
    .with_rate_control(RateControlMode::Crf(23));

let mut encoder = H264Encoder::new(config)?;
let packets = encoder.encode(&frame)?;
```

### AAC Decoding

```rust
use transcode_codecs::audio::AacDecoder;
use transcode_codecs::traits::AudioDecoder;

let mut decoder = AacDecoder::new();
decoder.set_extra_data(&audio_specific_config)?;

let samples = decoder.decode(&packet)?;
```

### AAC Encoding

```rust
use transcode_codecs::audio::aac::{AacEncoder, AacEncoderConfig};

let config = AacEncoderConfig {
    sample_rate: 44100,
    channels: 2,
    bitrate: 128000,
    ..Default::default()
};

let mut encoder = AacEncoder::new(config)?;
let packets = encoder.encode(&sample)?;
```

## SIMD Optimization

Performance-critical operations use SIMD with runtime detection:

- **x86_64**: AVX2 when available
- **AArch64**: NEON (always available)
- **Fallback**: Scalar implementations for all platforms

Optimized operations include IDCT, motion compensation, Hadamard transform, SAD calculation, deblocking filters, and MDCT.

## Documentation

See the main [Transcode documentation](../README.md) for the complete library guide.
