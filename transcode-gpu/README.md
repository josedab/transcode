# transcode-gpu

GPU-accelerated video processing using wgpu compute shaders.

## Features

- **Cross-platform GPU compute** via wgpu (Vulkan, Metal, DX12, WebGPU)
- **Color space conversion** - YUV to RGB and RGB to YUV (BT.601, BT.709)
- **Scaling** - Bilinear, bicubic, and Lanczos interpolation
- **Filters** - Gaussian blur, sharpening, grayscale
- **Color adjustment** - Brightness, contrast, saturation, gamma

## Supported Pixel Formats

- RGBA8, BGRA8
- NV12, I420 (YUV 4:2:0)
- YUYV (YUV 4:2:2)
- RGBA16, RGBA32F (high bit depth)

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-gpu = { path = "../transcode-gpu" }
```

### Basic Example

```rust
use transcode_gpu::{GpuProcessor, ProcessorConfig, ScaleMode};

async fn process_frame() -> transcode_gpu::Result<()> {
    // Create processor (initializes GPU context)
    let processor = GpuProcessor::new(ProcessorConfig::default()).await?;

    // Upload RGBA data to GPU
    let input = processor.upload_rgba(&rgba_data, 1280, 720)?;

    // Scale to 1080p with bicubic interpolation
    let scaled = processor.scale(&input, 1920, 1080, ScaleMode::Bicubic)?;

    // Apply sharpening
    let sharpened = processor.sharpen(&scaled, 0.5)?;

    // Download result
    let output_data = processor.download(&sharpened)?;

    Ok(())
}
```

### Filter Operations

```rust
// Gaussian blur
let blurred = processor.blur(&input, 5, 1.5)?;  // radius=5, sigma=1.5

// Color adjustment
let adjusted = processor.color_adjust(
    &input,
    0.1,   // brightness (+10%)
    1.2,   // contrast
    1.0,   // saturation
    1.0,   // gamma
)?;

// Grayscale conversion
let gray = processor.grayscale(&input)?;
```

### Custom GPU Context

```rust
use transcode_gpu::{GpuContext, GpuContextConfig, GpuProcessor, ProcessorConfig};

// Configure for low power GPU
let config = GpuContextConfig::low_power()
    .with_backend(wgpu::Backends::METAL);

let context = GpuContext::with_config(config).await?;
let processor = GpuProcessor::with_context(context, ProcessorConfig::default())?;
```

## Scale Modes

| Mode | Description |
|------|-------------|
| `Nearest` | Fastest, blocky artifacts |
| `Bilinear` | Good balance of speed and quality |
| `Bicubic` | Higher quality, Catmull-Rom spline |
| `Lanczos` | Best quality, slowest |

## GPU Capabilities

Query device capabilities:

```rust
let caps = processor.capabilities();
println!("GPU: {} ({})", caps.device_name, caps.backend);
println!("Max texture: {}x{}", caps.max_texture_dimension, caps.max_texture_dimension);
```

## Optional Features

- `async` - Enables tokio/futures integration for async runtime support

## Documentation

See the main [transcode documentation](../README.md) for the complete transcoding framework.
