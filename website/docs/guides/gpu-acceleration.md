---
sidebar_position: 3
title: GPU Acceleration
description: Use GPU compute shaders for video processing
---

# GPU Acceleration

Transcode uses wgpu for cross-platform GPU acceleration, enabling fast color conversion, scaling, and video effects.

## Overview

GPU acceleration in Transcode handles:

- **Color space conversion** (YUV ↔ RGB)
- **Scaling and resizing**
- **Video effects** (brightness, contrast, saturation)
- **Format conversion**

:::note
GPU acceleration is for video **processing**, not encoding. For hardware-accelerated encoding, see [Hardware Acceleration](/docs/advanced/hardware-acceleration).
:::

## Setup

### Enable the GPU Feature

```toml
[dependencies]
transcode = { version = "1.0", features = ["gpu"] }
transcode-gpu = "1.0"
```

### Check GPU Availability

```rust
use transcode_gpu::{GpuContext, GpuCapabilities};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Try to create GPU context
    match GpuContext::new().await {
        Ok(ctx) => {
            let caps = ctx.capabilities();
            println!("GPU: {}", caps.device_name);
            println!("Backend: {}", caps.backend);
            println!("Max texture: {}x{}", caps.max_texture_dimension, caps.max_texture_dimension);
        }
        Err(e) => {
            println!("No GPU available: {}", e);
            println!("Falling back to CPU processing");
        }
    }

    Ok(())
}
```

## Basic GPU Processing

### Initialize Context

```rust
use transcode_gpu::{GpuContext, GpuProcessor};

#[tokio::main]
async fn main() -> transcode_gpu::Result<()> {
    // Create GPU context (auto-selects best backend)
    let ctx = GpuContext::new().await?;

    // Create processor
    let processor = GpuProcessor::new(&ctx)?;

    // Process frames...

    Ok(())
}
```

### Color Conversion

Convert between YUV and RGB color spaces:

```rust
use transcode_gpu::{GpuContext, ColorConverter, ColorSpace};

let ctx = GpuContext::new().await?;
let converter = ColorConverter::new(&ctx)?;

// YUV to RGB
let rgb_frame = converter.yuv_to_rgb(&yuv_frame, ColorSpace::Bt709).await?;

// RGB to YUV
let yuv_frame = converter.rgb_to_yuv(&rgb_frame, ColorSpace::Bt709).await?;
```

### GPU Scaling

High-quality scaling on GPU:

```rust
use transcode_gpu::{GpuContext, GpuScaler, ScaleMode};

let ctx = GpuContext::new().await?;
let scaler = GpuScaler::new(&ctx, 3840, 2160)?;  // Output size

// Scale frame
let scaled = scaler.scale(&frame, ScaleMode::Lanczos).await?;
```

Scale modes:
- `Nearest` - Fast, blocky (good for pixel art)
- `Bilinear` - Fast, smooth (default)
- `Bicubic` - Higher quality
- `Lanczos` - Best quality, slower

## Video Effects

### Brightness, Contrast, Saturation

```rust
use transcode_gpu::{GpuContext, EffectsProcessor, Effects};

let ctx = GpuContext::new().await?;
let effects = EffectsProcessor::new(&ctx)?;

let effect_params = Effects {
    brightness: 0.1,    // -1.0 to 1.0 (0.0 = no change)
    contrast: 1.2,      // 0.0 to 2.0 (1.0 = no change)
    saturation: 1.1,    // 0.0 to 2.0 (1.0 = no change)
    gamma: 1.0,         // 0.1 to 3.0 (1.0 = no change)
};

let processed = effects.apply(&frame, &effect_params).await?;
```

### Sharpen

```rust
use transcode_gpu::SharpenFilter;

let sharpen = SharpenFilter::new(&ctx, 0.5)?;  // Strength 0.0-1.0
let sharpened = sharpen.process(&frame).await?;
```

### Blur

```rust
use transcode_gpu::BlurFilter;

let blur = BlurFilter::new(&ctx, 5)?;  // Radius in pixels
let blurred = blur.process(&frame).await?;
```

## Processing Pipeline

Chain multiple GPU operations:

```rust
use transcode_gpu::{
    GpuContext, GpuPipeline, GpuPipelineBuilder,
    ScaleMode, ColorSpace, Effects,
};

let ctx = GpuContext::new().await?;

let pipeline = GpuPipelineBuilder::new(&ctx)
    // Convert from YUV to RGB
    .yuv_to_rgb(ColorSpace::Bt709)
    // Scale to 1080p
    .scale(1920, 1080, ScaleMode::Lanczos)
    // Apply color correction
    .effects(Effects {
        brightness: 0.05,
        contrast: 1.1,
        saturation: 1.05,
        gamma: 1.0,
    })
    // Sharpen slightly
    .sharpen(0.3)
    // Convert back to YUV for encoding
    .rgb_to_yuv(ColorSpace::Bt709)
    .build()?;

// Process frames
for frame in frames {
    let processed = pipeline.process(&frame).await?;
    encoder.encode(&processed)?;
}
```

## Integration with Transcoding

### Using GPU Filters in Pipeline

```rust
use transcode::{Transcoder, TranscodeOptions};
use transcode_gpu::{GpuScaleFilter, GpuEffectsFilter};

let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.mp4")
    .gpu_acceleration(true);  // Enable GPU processing

let mut transcoder = Transcoder::new(options)?;

// Add GPU-accelerated filters
transcoder = transcoder.add_video_filter(
    GpuScaleFilter::new(1920, 1080)
);
transcoder = transcoder.add_video_filter(
    GpuEffectsFilter::new()
        .brightness(0.1)
        .contrast(1.1)
);

transcoder.run()?;
```

### Automatic GPU Fallback

```rust
use transcode::TranscodeOptions;

let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.mp4")
    .gpu_acceleration(true)
    .gpu_fallback_to_cpu(true);  // Use CPU if GPU unavailable
```

## Backend Selection

Transcode uses wgpu, which supports multiple backends:

| Backend | Platform | Notes |
|---------|----------|-------|
| Vulkan | Linux, Windows | Best performance |
| Metal | macOS, iOS | Apple platforms |
| DX12 | Windows | DirectX 12 |
| WebGPU | Browser | WebAssembly |

### Force Specific Backend

```rust
use transcode_gpu::{GpuContext, GpuBackend};

let ctx = GpuContext::with_backend(GpuBackend::Vulkan).await?;
```

### List Available Backends

```rust
use transcode_gpu::GpuContext;

for backend in GpuContext::available_backends() {
    println!("Available: {:?}", backend);
}
```

## Performance Optimization

### Batch Processing

Process multiple frames together:

```rust
use transcode_gpu::BatchProcessor;

let batch = BatchProcessor::new(&ctx, 8)?;  // Process 8 frames at once

let processed_frames = batch.process_batch(&frames).await?;
```

### Memory Management

Reuse GPU buffers:

```rust
use transcode_gpu::{GpuContext, GpuFramePool};

let ctx = GpuContext::new().await?;
let pool = GpuFramePool::new(&ctx, 10, 1920, 1080)?;

// Get frame from pool (reuses GPU memory)
let gpu_frame = pool.get()?;
// ... process ...
drop(gpu_frame);  // Returns to pool
```

### Async Processing

Overlap GPU and CPU work:

```rust
use futures::stream::{self, StreamExt};

let frames: Vec<Frame> = decode_frames()?;

// Process frames in parallel on GPU
let processed: Vec<Frame> = stream::iter(frames)
    .map(|frame| async {
        pipeline.process(&frame).await
    })
    .buffer_unordered(4)  // Up to 4 concurrent GPU operations
    .collect::<Vec<_>>()
    .await
    .into_iter()
    .collect::<Result<Vec<_>, _>>()?;
```

## GPU Capabilities

Check what your GPU supports:

```rust
let caps = ctx.capabilities();

println!("Device: {}", caps.device_name);
println!("Driver: {}", caps.driver_info);
println!("Backend: {}", caps.backend);
println!("Max texture: {}", caps.max_texture_dimension);
println!("Max buffer: {} MB", caps.max_buffer_size / 1_000_000);
println!("Workgroup size: {:?}", caps.max_workgroup_size);
println!("Float16 support: {}", caps.supports_f16);
```

## Troubleshooting

### GPU Not Detected

```rust
// Check for GPU errors
match GpuContext::new().await {
    Ok(_) => println!("GPU ready"),
    Err(transcode_gpu::GpuError::NoAdapter) => {
        println!("No GPU adapter found");
        println!("Ensure GPU drivers are installed");
    }
    Err(transcode_gpu::GpuError::NoDevice) => {
        println!("GPU device not available");
        println!("GPU may be in use by another process");
    }
    Err(e) => println!("GPU error: {}", e),
}
```

### Performance Issues

1. **Check GPU utilization**:
   ```bash
   # NVIDIA
   nvidia-smi

   # AMD (Linux)
   radeontop
   ```

2. **Profile GPU operations**:
   ```rust
   let ctx = GpuContext::with_profiling(true).await?;
   // ... process ...
   for timing in ctx.get_timings() {
       println!("{}: {:?}", timing.name, timing.duration);
   }
   ```

3. **Reduce memory pressure**:
   - Lower batch size
   - Use frame pools
   - Process lower resolution intermediate frames

## Next Steps

- [AI Enhancement](/docs/guides/ai-enhancement) - Neural network processing
- [Hardware Acceleration](/docs/advanced/hardware-acceleration) - Hardware encoding
- [Quality Metrics](/docs/guides/quality-metrics) - Measure processing quality
