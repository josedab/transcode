# Transcode Project Guide

This document provides context for working with the Transcode codec library.

## Project Structure

```
transcode/
├── transcode-core/         # Core types and utilities (bitstream, frames, packets, errors)
├── transcode-codecs/       # Codec implementations (H.264, AAC, MP3)
├── transcode-containers/   # Container format support (MP4)
├── transcode-pipeline/     # Transcoding pipeline orchestration
├── transcode-av1/          # AV1 codec (rav1e encoder, dav1d decoder)
├── transcode-streaming/    # HLS/DASH streaming output
├── transcode-gpu/          # GPU compute via wgpu (color conversion, scaling)
├── transcode-ai/           # AI enhancement (upscaling, denoising, interpolation)
├── transcode-quality/      # Quality metrics (PSNR, SSIM, MS-SSIM, VMAF)
├── transcode-distributed/  # Distributed transcoding (coordinator/worker)
├── transcode-intel/        # Content intelligence (scene detection, classification)
├── transcode-wasm/         # WebAssembly support with Web Workers
├── transcode/              # High-level API and CLI library
├── transcode-cli/          # Command-line interface
└── transcode-python/       # Python bindings (PyO3)
```

## Building and Testing

```bash
# Build all crates (excluding Python bindings)
cargo build --workspace --exclude transcode-python

# Run tests
cargo test --workspace --exclude transcode-python

# Run clippy with warnings as errors
cargo clippy --workspace --exclude transcode-python -- -D warnings
```

## Key Patterns

### Error Handling

Use `transcode_core::error::{Error, Result, CodecError, BitstreamError}` for error handling:

```rust
use transcode_core::error::{CodecError, Result};

fn decode_something(data: &[u8]) -> Result<Frame> {
    if data.is_empty() {
        return Err(CodecError::BitstreamCorruption {
            message: "Empty data".into()
        }.into());
    }
    // ...
}
```

### Bitstream Reading

Use `transcode_core::bitstream::BitReader` for parsing:

```rust
use transcode_core::bitstream::BitReader;

let mut reader = BitReader::new(data);
let value = reader.read_bits(8)?;
let flag = reader.read_bit()?;
let exp_golomb = reader.read_ue()?;
```

### Codec Trait Implementation

Codecs implement traits from `transcode-codecs/src/traits.rs`:
- `VideoDecoder` / `VideoEncoder`
- `AudioDecoder` / `AudioEncoder`
- `CodecInfo`

### Frame and Packet Types

- `Frame` - Decoded video frame (from `transcode_core::frame`)
- `Sample` / `SampleBuffer` - Decoded audio (from `transcode_core::sample`)
- `Packet` - Encoded/muxed data (from `transcode_core::packet`)

## Codec-Specific Notes

### H.264 Module (`transcode-codecs/src/video/h264/`)

- `nal.rs` - NAL unit parsing (Annex B and AVCC formats)
- `sps.rs`, `pps.rs` - Parameter set parsing
- `slice.rs` - Slice header parsing
- `decoder.rs`, `encoder.rs` - Main codec implementations
- `cabac.rs`, `cavlc.rs` - Entropy coding
- `transform.rs` - DCT/quantization
- `prediction.rs` - Intra/inter prediction
- `deblock.rs` - Deblocking filter

### AAC Module (`transcode-codecs/src/audio/aac/`)

- `adts.rs` - ADTS header parsing
- `decoder.rs`, `encoder.rs` - Main codec implementations
- `huffman.rs` - Huffman coding
- `mdct.rs` - MDCT transform
- `tns.rs` - Temporal Noise Shaping
- `psy.rs` - Psychoacoustic model

### MP4 Container (`transcode-containers/src/mp4/`)

- `atoms.rs` - Atom/box parsing and writing
- `demuxer.rs` - MP4 demuxing
- `muxer.rs` - MP4 muxing

## Advanced Crates

### AV1 Module (`transcode-av1/`)

- `encoder.rs` - rav1e-based AV1 encoder with configurable speed/quality presets
- `decoder.rs` - dav1d-based AV1 decoder
- `config.rs` - Encoding configuration (speed, quality, tiles)

### Streaming Module (`transcode-streaming/`)

- `hls.rs` - HLS muxer with playlist generation
- `dash.rs` - DASH muxer with MPD generation
- `segment.rs` - Segment handling (init segments, media segments)
- `manifest.rs` - Playlist/manifest types

### GPU Module (`transcode-gpu/`)

- `context.rs` - wgpu context and device management
- `pipeline.rs` - Compute pipeline orchestration
- `color.rs` - Color conversion shaders (YUV ↔ RGB)
- `scale.rs` - GPU-accelerated scaling
- `effects.rs` - Brightness, contrast, saturation
- `shaders/` - WGSL compute shaders

### AI Module (`transcode-ai/`)

- `upscaler.rs` - Neural upscaling (Lanczos, bilinear)
- `denoiser.rs` - Denoising (bilateral, NLMeans)
- `interpolator.rs` - Frame interpolation (linear, optical flow)
- `model.rs` - Model loading and inference

### Quality Module (`transcode-quality/`)

- `psnr.rs` - PSNR calculation (per-channel, weighted)
- `ssim.rs` - SSIM and MS-SSIM with Gaussian windows
- `vmaf.rs` - VMAF approximation (VIF, DLM)
- `lib.rs` - Batch assessment and quality reports

### Distributed Module (`transcode-distributed/`)

- `coordinator.rs` - Job scheduling and worker management
- `worker.rs` - Worker pool and health monitoring
- `task.rs` - Task state machine with retry logic

### Intel Module (`transcode-intel/`)

- `scene.rs` - Scene detection (histogram, content, edge, combined)
- `classify.rs` - Content classification (shot type, motion level)
- `lib.rs` - VideoAnalyzer combining detection and classification

### WASM Module (`transcode-wasm/`)

- `lib.rs` - WebAssembly bindings
- `worker.rs` - Web Worker support for parallel processing

## Clippy Allowances

Signal processing code uses several clippy allow attributes at the module level:
- `clippy::needless_range_loop` - Common in DSP loops
- `clippy::too_many_arguments` - Necessary for complex video processing
- `clippy::cast_abs_to_unsigned` - Common in codec implementations

## Common Tasks

### Adding a New Codec

1. Create module under `transcode-codecs/src/video/` or `audio/`
2. Implement appropriate trait (`VideoDecoder`, `AudioEncoder`, etc.)
3. Export from parent module's `mod.rs`
4. Add tests in the module

### Adding Container Support

1. Create module under `transcode-containers/src/`
2. Implement `Demuxer` and/or `Muxer` traits
3. Handle stream info extraction and packet routing
