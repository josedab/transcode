# GPU Acceleration

This guide covers GPU-accelerated video processing using the `transcode-gpu` crate.

## Overview

Transcode uses [wgpu](https://wgpu.rs/) for cross-platform GPU compute, enabling:
- Color space conversion (YUV ↔ RGB)
- Video scaling and resizing
- Visual effects (brightness, contrast, saturation)
- Frame blending and compositing

## Requirements

GPU acceleration requires a compatible graphics API:
- **Vulkan** (Linux, Windows)
- **Metal** (macOS, iOS)
- **DX12** (Windows)
- **WebGPU** (Browsers)

## Basic Usage

```rust
use transcode_gpu::{GpuContext, ColorConverter, GpuScaler};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU context
    let ctx = GpuContext::new().await?;

    println!("Using GPU: {}", ctx.adapter_info().name);

    Ok(())
}
```

## Color Conversion

Convert between color spaces on the GPU:

```rust
use transcode_gpu::{GpuContext, ColorConverter};

// Create converter
let converter = ColorConverter::new(&ctx)?;

// YUV to RGB conversion
let rgb_frame = converter.yuv_to_rgb(&yuv_frame).await?;

// RGB to YUV conversion
let yuv_frame = converter.rgb_to_yuv(&rgb_frame).await?;
```

### Supported Color Spaces

| Source | Target | Performance |
|--------|--------|-------------|
| YUV420P | RGB24 | ~0.5ms (1080p) |
| YUV420P | RGBA32 | ~0.5ms (1080p) |
| RGB24 | YUV420P | ~0.6ms (1080p) |
| NV12 | RGB24 | ~0.4ms (1080p) |

## Scaling

GPU-accelerated video scaling:

```rust
use transcode_gpu::{GpuContext, GpuScaler, ScaleMode};

// Create scaler for 4K output
let scaler = GpuScaler::new(&ctx, 3840, 2160)?;

// Scale with different algorithms
let scaled = scaler.scale(&frame, ScaleMode::Bilinear).await?;
let scaled = scaler.scale(&frame, ScaleMode::Lanczos).await?;
```

### Scale Modes

| Mode | Quality | Speed | Best For |
|------|---------|-------|----------|
| Nearest | Low | Fastest | Pixel art |
| Bilinear | Medium | Fast | Real-time |
| Bicubic | High | Medium | General use |
| Lanczos | Highest | Slower | Final output |

## Effects Pipeline

Apply visual effects:

```rust
use transcode_gpu::{GpuContext, EffectsPipeline, Effect};

let mut pipeline = EffectsPipeline::new(&ctx)?;

// Chain effects
pipeline.add(Effect::Brightness(1.1));     // +10% brightness
pipeline.add(Effect::Contrast(1.2));       // +20% contrast
pipeline.add(Effect::Saturation(0.9));     // -10% saturation

// Process frame
let result = pipeline.process(&frame).await?;
```

## Memory Management

### Buffer Pooling

Reuse GPU buffers for better performance:

```rust
use transcode_gpu::{GpuContext, BufferPool};

// Create pool for 1080p frames
let pool = BufferPool::new(&ctx, 1920, 1080, 8)?; // 8 buffers

// Get buffer from pool
let buffer = pool.acquire().await?;

// Buffer automatically returned when dropped
```

### Zero-Copy Transfers

When possible, use zero-copy transfers:

```rust
use transcode_gpu::ZeroCopyFrame;

// Map GPU memory directly
let mapped = ZeroCopyFrame::map(&ctx, &frame)?;

// Access pixel data without copying
let pixels = mapped.as_slice();
```

## Performance Tips

1. **Batch operations** - Process multiple frames together
2. **Reuse contexts** - Don't recreate GPU context per frame
3. **Use buffer pools** - Avoid allocation overhead
4. **Async processing** - Overlap CPU and GPU work

```rust
// Good: Batch processing
let futures: Vec<_> = frames.iter()
    .map(|f| converter.yuv_to_rgb(f))
    .collect();
let results = futures::future::join_all(futures).await;

// Bad: Sequential processing
for frame in &frames {
    let result = converter.yuv_to_rgb(frame).await?;
}
```

## Fallback Handling

When GPU is unavailable:

```rust
use transcode_gpu::GpuContext;

match GpuContext::new().await {
    Ok(ctx) => {
        // Use GPU acceleration
    }
    Err(_) => {
        // Fall back to CPU processing
        println!("GPU unavailable, using CPU");
    }
}
```

## Debugging

Enable GPU debugging:

```rust
let ctx = GpuContext::builder()
    .validation(true)      // Enable validation layers
    .debug_labels(true)    // Add debug labels
    .build()
    .await?;
```

## Benchmarks

Typical performance on NVIDIA RTX 3080:

| Operation | 1080p | 4K |
|-----------|-------|-----|
| YUV→RGB | 0.3ms | 1.2ms |
| Scale 2x | 0.5ms | 2.0ms |
| Full pipeline | 1.5ms | 6.0ms |
