# transcode-resample

High-quality audio resampling for the transcode library.

## Overview

This crate provides multiple resampling algorithms with SIMD optimization for efficient sample rate conversion.

## Features

- **Multiple Algorithms**: Linear, sinc, and polyphase filter bank resampling
- **Arbitrary Conversion**: Any sample rate to any other sample rate
- **Multi-Channel**: Full support for multi-channel audio
- **SIMD Optimization**: AVX2 (x86_64) and NEON (aarch64) acceleration
- **Low Latency**: Configurable latency/quality tradeoffs
- **Planar/Interleaved**: Support for both audio formats

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-resample = { path = "../transcode-resample" }
```

### Basic Resampling

```rust
use transcode_resample::{Resampler, ResamplerConfig, ResamplerType};

// Create a high-quality sinc resampler
let config = ResamplerConfig::new(44100, 48000)
    .with_type(ResamplerType::Sinc { window_size: 64 })
    .with_channels(2);

let mut resampler = Resampler::new(config)?;

// Process audio
let output = resampler.process(&input_buffer)?;
```

### Algorithm Selection

```rust
use transcode_resample::{LinearResampler, SincResampler, PolyphaseResampler};

// Fast linear interpolation (low quality)
let mut linear = LinearResampler::new(44100, 48000);

// High-quality sinc interpolation
let mut sinc = SincResampler::new(44100, 48000, 64)?;

// Efficient polyphase filter (fixed rate conversions)
let mut polyphase = PolyphaseResampler::new(44100, 48000, 32)?;
```

### Multi-Channel Processing

```rust
use transcode_resample::{Resampler, ResamplerConfig};

let config = ResamplerConfig::new(44100, 48000)
    .with_channels(6);  // 5.1 surround

let mut resampler = Resampler::new(config)?;

// Process interleaved 6-channel audio
let output = resampler.process_interleaved(&input, 6)?;
```

### Flush Remaining Samples

```rust
use transcode_resample::ResamplerImpl;

// Process all input
while let Some(chunk) = input_source.next() {
    let output = resampler.process(&chunk)?;
    output_sink.write(&output);
}

// Flush any remaining samples
let final_samples = resampler.flush()?;
output_sink.write(&final_samples);
```

## Algorithms

| Algorithm | Quality | Speed | Use Case |
|-----------|---------|-------|----------|
| Linear | Low | Fast | Previews, monitoring |
| Sinc | High | Medium | Master quality |
| Polyphase | High | Fast | Fixed rate conversion |

## Window Functions

Available for sinc resampling:

| Window | Description |
|--------|-------------|
| Blackman | Good sidelobe rejection |
| Kaiser | Configurable ripple/transition |
| Hann | General purpose |
| Hamming | Minimum stopband attenuation |

## Common Conversions

| From | To | Ratio |
|------|-----|-------|
| 44100 Hz | 48000 Hz | 160/147 |
| 48000 Hz | 44100 Hz | 147/160 |
| 44100 Hz | 96000 Hz | 320/147 |
| 48000 Hz | 96000 Hz | 2/1 |

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
