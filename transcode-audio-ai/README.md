# transcode-audio-ai

AI-powered audio processing for the transcode multimedia framework.

## Overview

This crate provides audio enhancement, upsampling, and content classification capabilities. It offers tools for noise reduction, loudness normalization, dynamic range compression, and intelligent audio analysis.

## Features

- **Audio Enhancement**: Noise reduction, normalization (LUFS-based), compression, and de-essing
- **Upsampling**: Multiple quality levels from fast linear interpolation to high-quality windowed sinc
- **Content Classification**: Detect speech, music, ambient audio, or silence
- **Noise Profiling**: Learn noise characteristics from silent sections for adaptive reduction

## Key Types

| Type | Description |
|------|-------------|
| `AudioBuffer` | Container for interleaved audio samples with channel/rate info |
| `AudioEnhancer` | Applies enhancement pipeline (noise reduction, normalization, etc.) |
| `AudioUpsampler` | Converts audio to higher sample rates |
| `AudioClassifier` | Classifies audio content type |
| `Compressor` | Dynamic range compression with attack/release control |
| `SpectralSubtraction` | FFT-based noise reduction processor |
| `NoiseProfile` | Learned noise characteristics for adaptive reduction |

## Usage Examples

### Audio Enhancement

```rust
use transcode_audio_ai::{AudioBuffer, AudioEnhancer, AudioEnhanceConfig};

let mut buffer = AudioBuffer::from_samples(samples, 2, 44100);

let config = AudioEnhanceConfig {
    noise_reduction: true,
    noise_reduction_strength: 0.5,
    normalize: true,
    target_lufs: -14.0,
    ..Default::default()
};

let enhancer = AudioEnhancer::new(config);
let stats = enhancer.enhance(&mut buffer)?;

println!("Loudness: {} -> {} LUFS", stats.input_lufs, stats.output_lufs);
```

### Upsampling

```rust
use transcode_audio_ai::{AudioBuffer, AudioUpsampler, UpsampleQuality};

let buffer = AudioBuffer::from_samples(samples, 2, 44100);
let upsampler = AudioUpsampler::new(UpsampleQuality::High);

let hd_audio = upsampler.upsample(&buffer, 96000)?;
```

### Content Classification

```rust
use transcode_audio_ai::{AudioBuffer, AudioClassifier, AudioContentType};

let buffer = AudioBuffer::from_samples(samples, 1, 44100);
let classifier = AudioClassifier::new();

let result = classifier.classify(&buffer)?;
match result.content_type {
    AudioContentType::Speech => println!("Detected speech"),
    AudioContentType::Music => println!("Detected music"),
    AudioContentType::Mixed => println!("Mixed content"),
    AudioContentType::Ambient => println!("Ambient audio"),
    AudioContentType::Silence => println!("Silence"),
}
```

### Noise Reduction with Profile

```rust
use transcode_audio_ai::{AudioBuffer, NoiseProfile, SpectralSubtraction};

let mut buffer = AudioBuffer::from_samples(samples, 1, 44100);

// Learn noise from first 0.5 seconds of silence
let profile = NoiseProfile::learn(&buffer, 0, 22050);

let processor = SpectralSubtraction::new(2048);
processor.process(&mut buffer, &profile)?;
```

## Upsampling Quality Levels

| Quality | Algorithm | Use Case |
|---------|-----------|----------|
| `Fast` | Linear interpolation | Real-time, low latency |
| `Medium` | Windowed sinc | Balanced quality/speed |
| `High` | Polyphase filter | High-quality offline |
| `Best` | Neural-assisted | Maximum quality |

## Dependencies

- `transcode-core` - Core types and utilities
- `rustfft` - FFT operations for spectral processing
- `ndarray` - N-dimensional arrays for signal processing

## License

See workspace `Cargo.toml` for license information.
