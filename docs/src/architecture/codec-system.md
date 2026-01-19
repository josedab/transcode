# Codec System Architecture

This document describes the architecture of Transcode's codec system.

## Overview

The codec system is built on traits that define common interfaces for all codecs:

```
┌─────────────────────────────────────────────────────┐
│                    CodecInfo                         │
│  (capabilities, supported formats, SIMD features)    │
└─────────────────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│VideoDecoder │  │VideoEncoder │  │AudioDecoder │ ...
└─────────────┘  └─────────────┘  └─────────────┘
```

## Core Traits

### CodecInfo

Provides metadata about codec capabilities:

```rust
pub trait CodecInfo {
    /// Unique codec identifier (e.g., "h264", "aac")
    fn codec_id(&self) -> &str;

    /// Human-readable name
    fn name(&self) -> &str;

    /// Supported pixel formats (video) or sample formats (audio)
    fn supported_formats(&self) -> &[Format];

    /// SIMD features used by this codec
    fn simd_features(&self) -> SimdFeatures;
}
```

### VideoDecoder

Interface for video decoders:

```rust
pub trait VideoDecoder: CodecInfo {
    /// Decode a packet into a frame
    fn decode(&mut self, packet: &Packet) -> Result<Option<Frame>>;

    /// Flush remaining frames
    fn flush(&mut self) -> Result<Vec<Frame>>;

    /// Reset decoder state
    fn reset(&mut self);

    /// Get output frame dimensions
    fn dimensions(&self) -> (u32, u32);

    /// Get output pixel format
    fn pixel_format(&self) -> PixelFormat;
}
```

### VideoEncoder

Interface for video encoders:

```rust
pub trait VideoEncoder: CodecInfo {
    /// Encode a frame into packets
    fn encode(&mut self, frame: &Frame) -> Result<Vec<Packet>>;

    /// Flush remaining packets
    fn flush(&mut self) -> Result<Vec<Packet>>;

    /// Get encoder configuration
    fn config(&self) -> &EncoderConfig;
}
```

## Codec Registration

Codecs are registered in a global registry:

```rust
use transcode_codecs::{CodecRegistry, VideoDecoder};

// Get decoder by codec ID
let decoder: Box<dyn VideoDecoder> = CodecRegistry::video_decoder("h264")?;

// List available codecs
for codec in CodecRegistry::list_video_decoders() {
    println!("{}: {}", codec.codec_id(), codec.name());
}
```

## H.264 Decoder Architecture

The H.264 decoder is organized into several modules:

```
transcode-codecs/src/video/h264/
├── mod.rs          # Public API
├── decoder.rs      # Main decoder struct
├── nal.rs          # NAL unit parsing
├── sps.rs          # Sequence Parameter Set
├── pps.rs          # Picture Parameter Set
├── slice.rs        # Slice header parsing
├── cabac.rs        # CABAC entropy decoding
├── cavlc.rs        # CAVLC entropy decoding
├── transform.rs    # Integer transform (IDCT)
├── prediction.rs   # Intra/inter prediction
├── deblock.rs      # Deblocking filter
└── dpb.rs          # Decoded Picture Buffer
```

### Decoding Pipeline

```
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ NAL     │──▶│ Entropy │──▶│ Residual│──▶│ Recon-  │
│ Parse   │   │ Decode  │   │ + Pred  │   │ struct  │
└─────────┘   └─────────┘   └─────────┘   └─────────┘
                                               │
                                               ▼
                                         ┌─────────┐
                                         │ Deblock │
                                         │ Filter  │
                                         └─────────┘
                                               │
                                               ▼
                                         ┌─────────┐
                                         │ Output  │
                                         │ Frame   │
                                         └─────────┘
```

## AAC Decoder Architecture

The AAC decoder modules:

```
transcode-codecs/src/audio/aac/
├── mod.rs          # Public API
├── decoder.rs      # Main decoder struct
├── adts.rs         # ADTS header parsing
├── huffman.rs      # Huffman decoding
├── mdct.rs         # Modified DCT
├── tns.rs          # Temporal Noise Shaping
├── psy.rs          # Psychoacoustic model
└── window.rs       # Windowing functions
```

## SIMD Acceleration

Each codec can provide SIMD-optimized implementations:

```rust
// Runtime SIMD dispatch
pub fn idct_4x4(coeffs: &mut [i16; 16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { idct_4x4_avx2(coeffs) }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { idct_4x4_neon(coeffs) }
            return;
        }
    }

    idct_4x4_scalar(coeffs)
}
```

## Error Handling

Codecs use structured errors:

```rust
#[derive(Debug, Error)]
pub enum CodecError {
    #[error("Bitstream corruption at offset {offset}: {message}")]
    BitstreamCorruption { offset: usize, message: String },

    #[error("Unsupported profile: {profile}")]
    UnsupportedProfile { profile: String },

    #[error("Invalid parameter: {name} = {value}")]
    InvalidParameter { name: String, value: String },

    #[error("Decoder not ready: {reason}")]
    NotReady { reason: String },
}
```

## Adding a New Codec

See [Adding a Codec](../contributing/adding-a-codec.md) for a step-by-step guide.

## Performance Considerations

1. **Zero-copy where possible** - Use borrowed data for packet parsing
2. **SIMD for hot paths** - Transform, prediction, and filtering
3. **Memory pooling** - Reuse frame buffers
4. **Parallel processing** - Slice-level parallelism when spec allows
