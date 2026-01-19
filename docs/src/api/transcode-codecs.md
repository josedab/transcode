# transcode-codecs

Codec implementations for video and audio encoding/decoding.

## Overview

`transcode-codecs` provides pure Rust implementations of common codecs:

- **Video**: H.264, H.265/HEVC, VP8, VP9, AV1
- **Audio**: AAC, MP3, Opus, FLAC, Vorbis

## Crate Features

```toml
[dependencies]
transcode-codecs = { version = "1.0", features = ["h264", "aac"] }
```

| Feature | Description |
|---------|-------------|
| `h264` | H.264/AVC codec |
| `hevc` | H.265/HEVC codec |
| `vp9` | VP9 codec |
| `aac` | AAC codec |
| `mp3` | MP3 decoder |
| `opus` | Opus codec |
| `flac` | FLAC codec |
| `all-video` | All video codecs |
| `all-audio` | All audio codecs |
| `default` | Common codecs |

## Video Codecs

### H.264 Decoder

```rust
use transcode_codecs::video::h264::H264Decoder;

let mut decoder = H264Decoder::new()?;

// Decode packets
for packet in packets {
    if let Some(frame) = decoder.decode(&packet)? {
        process_frame(&frame);
    }
}

// Flush remaining frames
for frame in decoder.flush()? {
    process_frame(&frame);
}
```

### H.264 Encoder

```rust
use transcode_codecs::video::h264::{H264Encoder, H264Config, Profile};

let config = H264Config::builder()
    .width(1920)
    .height(1080)
    .framerate(30, 1)
    .bitrate(5_000_000)
    .profile(Profile::High)
    .level(Level::L4_1)
    .build()?;

let mut encoder = H264Encoder::new(config)?;

// Encode frames
for frame in frames {
    let packets = encoder.encode(&frame)?;
    for packet in packets {
        write_packet(&packet);
    }
}

// Flush remaining packets
for packet in encoder.flush()? {
    write_packet(&packet);
}
```

### Configuration Options

```rust
// Quality-based encoding (CRF)
let config = H264Config::builder()
    .crf(23)                    // 0-51, lower = better
    .preset(Preset::Medium)
    .build()?;

// Bitrate-based encoding
let config = H264Config::builder()
    .bitrate(5_000_000)         // Target bitrate
    .max_bitrate(7_000_000)     // Maximum bitrate
    .buffer_size(10_000_000)    // VBV buffer size
    .build()?;

// Low-latency streaming
let config = H264Config::builder()
    .preset(Preset::Ultrafast)
    .tune(Tune::ZeroLatency)
    .keyint(30)                 // Keyframe interval
    .build()?;
```

## Audio Codecs

### AAC Decoder

```rust
use transcode_codecs::audio::aac::AacDecoder;

let mut decoder = AacDecoder::new()?;

for packet in packets {
    if let Some(samples) = decoder.decode(&packet)? {
        process_audio(&samples);
    }
}
```

### AAC Encoder

```rust
use transcode_codecs::audio::aac::{AacEncoder, AacConfig, AacProfile};

let config = AacConfig::builder()
    .profile(AacProfile::Lc)
    .sample_rate(44100)
    .channels(2)
    .bitrate(128_000)
    .build()?;

let mut encoder = AacEncoder::new(config)?;

for samples in audio_samples {
    let packets = encoder.encode(&samples)?;
    for packet in packets {
        write_packet(&packet);
    }
}
```

## Codec Traits

All codecs implement common traits:

### CodecInfo

```rust
pub trait CodecInfo {
    fn codec_id(&self) -> &str;
    fn name(&self) -> &str;
    fn supported_formats(&self) -> &[Format];
    fn simd_features(&self) -> SimdFeatures;
}

// Usage
let decoder = H264Decoder::new()?;
println!("Codec: {}", decoder.name());        // "H.264/AVC"
println!("ID: {}", decoder.codec_id());       // "h264"
```

### VideoDecoder

```rust
pub trait VideoDecoder: CodecInfo {
    fn decode(&mut self, packet: &Packet) -> Result<Option<Frame>>;
    fn flush(&mut self) -> Result<Vec<Frame>>;
    fn reset(&mut self);
    fn dimensions(&self) -> (u32, u32);
    fn pixel_format(&self) -> PixelFormat;
}
```

### VideoEncoder

```rust
pub trait VideoEncoder: CodecInfo {
    fn encode(&mut self, frame: &Frame) -> Result<Vec<Packet>>;
    fn flush(&mut self) -> Result<Vec<Packet>>;
    fn config(&self) -> &EncoderConfig;
}
```

## SIMD Detection

```rust
use transcode_codecs::detect_simd;

let caps = detect_simd();
println!("AVX2: {}", caps.avx2);
println!("NEON: {}", caps.neon);
println!("Best level: {}", caps.best_level());
```

## Codec Registry

Dynamic codec lookup:

```rust
use transcode_codecs::CodecRegistry;

// Get decoder by ID
let decoder = CodecRegistry::video_decoder("h264")?;

// List available codecs
for codec in CodecRegistry::list_video_decoders() {
    println!("{}: {}", codec.codec_id(), codec.name());
}
```

## Full API Reference

See the [rustdoc documentation](https://docs.rs/transcode-codecs) for complete API details.
