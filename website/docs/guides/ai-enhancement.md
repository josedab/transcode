---
sidebar_position: 4
title: AI Enhancement
description: Neural network-based upscaling, denoising, and frame interpolation
---

# AI Enhancement

Transcode includes AI-powered video enhancement features for upscaling, denoising, and frame interpolation.

## Overview

The `transcode-ai` crate provides:

- **Upscaling**: Increase resolution with neural super-resolution
- **Denoising**: Remove noise while preserving detail
- **Frame interpolation**: Generate intermediate frames for smooth motion

## Setup

```toml
[dependencies]
transcode = { version = "1.0", features = ["ai"] }
transcode-ai = "1.0"
```

## Upscaling

Increase video resolution with intelligent detail enhancement.

### Basic Upscaling

```rust
use transcode_ai::{Upscaler, UpscaleMethod};

// Create upscaler (4x scale factor)
let upscaler = Upscaler::new(UpscaleMethod::Lanczos, 4)?;

// Process frame
let hd_frame = upscaler.process(&sd_frame)?;
```

### Upscale Methods

| Method | Quality | Speed | Best For |
|--------|---------|-------|----------|
| `Nearest` | Low | Fastest | Pixel art |
| `Bilinear` | Medium | Fast | Preview |
| `Bicubic` | Good | Medium | General |
| `Lanczos` | High | Slow | Final output |

### Scale Factors

```rust
// 2x upscale (480p -> 960p)
let upscaler = Upscaler::new(UpscaleMethod::Lanczos, 2)?;

// 4x upscale (480p -> 1920p)
let upscaler = Upscaler::new(UpscaleMethod::Lanczos, 4)?;

// Custom output size
let upscaler = Upscaler::with_output_size(
    UpscaleMethod::Lanczos,
    1920, 1080
)?;
```

### In Transcoding Pipeline

```rust
use transcode::{Transcoder, TranscodeOptions};
use transcode_ai::UpscaleFilter;

let options = TranscodeOptions::new()
    .input("sd_video.mp4")      // 480p input
    .output("hd_video.mp4");    // 1080p output

let mut transcoder = Transcoder::new(options)?;

// Add AI upscaling filter
transcoder = transcoder.add_video_filter(
    UpscaleFilter::new(1920, 1080, UpscaleMethod::Lanczos)
);

transcoder.run()?;
```

## Denoising

Remove noise while preserving edges and details.

### Basic Denoising

```rust
use transcode_ai::{Denoiser, DenoiseMethod};

// Create denoiser
let denoiser = Denoiser::new(DenoiseMethod::Bilateral)?;

// Configure strength (0.0 - 1.0)
let denoiser = denoiser.with_strength(0.5);

// Process frame
let clean_frame = denoiser.process(&noisy_frame)?;
```

### Denoise Methods

| Method | Approach | Best For |
|--------|----------|----------|
| `Bilateral` | Edge-preserving blur | General noise |
| `NlMeans` | Non-local means | Film grain |
| `Temporal` | Multi-frame analysis | Video sequences |

### Temporal Denoising

Uses multiple frames for better results:

```rust
use transcode_ai::{TemporalDenoiser, TemporalConfig};

let config = TemporalConfig {
    temporal_radius: 2,      // Use 2 frames before/after
    spatial_strength: 0.5,
    temporal_strength: 0.7,
};

let denoiser = TemporalDenoiser::new(config)?;

// Feed frames in sequence
for frame in frames {
    if let Some(denoised) = denoiser.process(&frame)? {
        output_frames.push(denoised);
    }
}

// Flush remaining frames
for frame in denoiser.flush()? {
    output_frames.push(frame);
}
```

### In Transcoding Pipeline

```rust
use transcode_ai::DenoiseFilter;

let mut transcoder = Transcoder::new(options)?;

transcoder = transcoder.add_video_filter(
    DenoiseFilter::new(DenoiseMethod::Bilateral)
        .strength(0.5)
);

transcoder.run()?;
```

## Frame Interpolation

Generate intermediate frames for smooth slow-motion or frame rate conversion.

### Basic Interpolation

```rust
use transcode_ai::{FrameInterpolator, InterpolationMethod};

let interpolator = FrameInterpolator::new(InterpolationMethod::OpticalFlow)?;

// Generate frame between two existing frames
// t=0.5 means exactly halfway
let mid_frame = interpolator.interpolate(&frame1, &frame2, 0.5)?;
```

### Interpolation Methods

| Method | Quality | Speed | Notes |
|--------|---------|-------|-------|
| `Linear` | Low | Fast | Simple blending |
| `OpticalFlow` | High | Medium | Motion-aware |

### Frame Rate Conversion

Convert 30fps to 60fps:

```rust
use transcode_ai::{FrameRateConverter, InterpolationMethod};

let converter = FrameRateConverter::new(
    30.0,  // Input fps
    60.0,  // Output fps
    InterpolationMethod::OpticalFlow
)?;

for frame in input_frames {
    for output_frame in converter.process(&frame)? {
        output_frames.push(output_frame);
    }
}
```

### Slow Motion

Create smooth slow motion from normal video:

```rust
use transcode_ai::SlowMotion;

// 4x slow motion (30fps -> 120fps effective, played at 30fps)
let slow_mo = SlowMotion::new(4, InterpolationMethod::OpticalFlow)?;

for frame in input_frames {
    for output_frame in slow_mo.process(&frame)? {
        output_frames.push(output_frame);
    }
}
```

### In Transcoding Pipeline

```rust
use transcode_ai::FrameRateFilter;

let options = TranscodeOptions::new()
    .input("30fps_video.mp4")
    .output("60fps_video.mp4");

let mut transcoder = Transcoder::new(options)?;

transcoder = transcoder.add_video_filter(
    FrameRateFilter::new(60.0, InterpolationMethod::OpticalFlow)
);

transcoder.run()?;
```

## Combining Enhancements

Chain multiple AI enhancements:

```rust
use transcode::{Transcoder, TranscodeOptions};
use transcode_ai::{
    DenoiseFilter, DenoiseMethod,
    UpscaleFilter, UpscaleMethod,
    FrameRateFilter, InterpolationMethod,
};

let options = TranscodeOptions::new()
    .input("old_video.mp4")    // Noisy SD 24fps
    .output("enhanced.mp4");   // Clean HD 60fps

let mut transcoder = Transcoder::new(options)?;

// 1. First denoise (before upscaling to avoid amplifying noise)
transcoder = transcoder.add_video_filter(
    DenoiseFilter::new(DenoiseMethod::Bilateral).strength(0.6)
);

// 2. Then upscale
transcoder = transcoder.add_video_filter(
    UpscaleFilter::new(1920, 1080, UpscaleMethod::Lanczos)
);

// 3. Finally interpolate frames
transcoder = transcoder.add_video_filter(
    FrameRateFilter::new(60.0, InterpolationMethod::OpticalFlow)
);

transcoder.run()?;
```

## GPU Acceleration

AI enhancements benefit greatly from GPU acceleration:

```rust
use transcode_ai::{Upscaler, UpscaleMethod, Device};

// Use GPU if available
let device = Device::gpu().unwrap_or(Device::cpu());

let upscaler = Upscaler::new(UpscaleMethod::Lanczos, 4)?
    .with_device(device);
```

### Check Device

```rust
use transcode_ai::Device;

match Device::gpu() {
    Some(device) => {
        println!("Using GPU: {}", device.name());
    }
    None => {
        println!("No GPU available, using CPU");
    }
}
```

## Performance Considerations

### Memory Usage

AI enhancement is memory-intensive:

```rust
use transcode_ai::{Config, MemoryLimit};

let config = Config {
    // Limit memory usage
    memory_limit: MemoryLimit::Mb(2048),  // 2GB max

    // Process in tiles for large frames
    tile_size: 512,

    // Overlap tiles to avoid seams
    tile_overlap: 32,
};

let upscaler = Upscaler::with_config(config)?;
```

### Batch Processing

Process multiple frames together:

```rust
use transcode_ai::{Upscaler, BatchConfig};

let batch_config = BatchConfig {
    batch_size: 4,  // Process 4 frames at once
};

let upscaler = Upscaler::new(UpscaleMethod::Lanczos, 4)?
    .with_batch_config(batch_config);
```

### Quality vs Speed Trade-offs

| Setting | Quality | Speed |
|---------|---------|-------|
| `Method::Nearest` | Lowest | Fastest |
| `Method::Bilinear` | Low | Fast |
| `Method::Lanczos` | High | Slow |
| Small tile size | Lower | Faster |
| Large tile size | Higher | Slower |
| No GPU | N/A | 10x slower |

## Limitations

1. **Artifacts**: AI upscaling may introduce artifacts on certain content
2. **Motion blur**: Frame interpolation may struggle with fast motion
3. **Processing time**: High-quality enhancement is computationally expensive
4. **Memory**: Large frames require significant GPU/CPU memory

## Best Practices

1. **Denoise before upscaling** to avoid amplifying noise
2. **Use temporal denoising** for video sequences when possible
3. **Test on sample clips** before processing full videos
4. **Monitor quality** with [quality metrics](/docs/guides/quality-metrics)
5. **Use GPU acceleration** when available

## Next Steps

- [Quality Metrics](/docs/guides/quality-metrics) - Measure enhancement quality
- [GPU Acceleration](/docs/guides/gpu-acceleration) - Optimize GPU usage
- [Distributed Processing](/docs/guides/distributed-processing) - Scale AI processing
