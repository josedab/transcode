# transcode-ai

AI-powered video enhancement for the transcode project. Provides neural network-based upscaling, denoising, and frame interpolation capabilities.

## Features

### Super Resolution (Upscaling)
Upscale video to higher resolutions (2x or 4x) using neural networks or traditional methods.

- **Models**: RealESRGAN, ESPCN, FSRCNN, EDSR, SwinIR, Anime4K
- **Fallback**: Lanczos interpolation when models unavailable
- **Tiled processing**: Handle large images efficiently

### Denoising
Remove noise while preserving details with configurable noise levels.

- **Models**: DnCNN, FFDNet, NAFNet, Restormer
- **Fallback**: Bilateral filter, Non-local means
- **Temporal denoising**: For video sequences

### Frame Interpolation
Generate intermediate frames for higher frame rates with scene change detection.

- **Models**: RIFE, FILM, IFRNet, CAIN
- **Modes**: Linear, motion-compensated, optical flow, neural
- **Fallback**: Block-matching motion compensation

## Key Types

| Type | Description |
|------|-------------|
| `EnhancementPipeline` | Combines multiple enhancers with configurable processing order |
| `Upscaler` / `UpscalerConfig` | Super-resolution upscaling |
| `Denoiser` / `DenoiserConfig` | Noise reduction |
| `FrameInterpolator` / `InterpolatorConfig` | Frame rate conversion |
| `Frame` | Video frame with RGB data and timestamp |
| `ModelLoader` / `ModelBackend` | Model management (CPU, CUDA, CoreML, DirectML) |

## Usage

### Upscaling

```rust
use transcode_ai::{Upscaler, UpscalerConfig, ScaleFactor, UpscaleModel};

let config = UpscalerConfig::default()
    .with_scale_factor(ScaleFactor::X2)
    .with_model(UpscaleModel::RealESRGAN);

let upscaler = Upscaler::new(config)?;
let upscaled = upscaler.process(&frame)?;
```

### Denoising

```rust
use transcode_ai::{Denoiser, DenoiserConfig, NoiseLevel};

let config = DenoiserConfig::default()
    .with_noise_level(NoiseLevel::Medium)
    .with_detail_preservation(0.3);

let denoiser = Denoiser::new(config)?;
let denoised = denoiser.process(&frame)?;
```

### Frame Interpolation

```rust
use transcode_ai::{FrameInterpolator, InterpolatorConfig};

let config = InterpolatorConfig::default()
    .with_multiplier(2)  // Double frame rate
    .with_scene_threshold(0.5);

let interpolator = FrameInterpolator::new(config)?;
let intermediate_frames = interpolator.interpolate(&frame1, &frame2, 0.5)?;
```

### Enhancement Pipeline

```rust
use transcode_ai::{EnhancementPipeline, PipelineConfig, ScaleFactor, NoiseLevel};

// Quick setup for upscaling only
let pipeline = EnhancementPipeline::upscale_only(ScaleFactor::X4)?;

// Or combine multiple enhancements
let config = PipelineConfig {
    enable_upscale: true,
    enable_denoise: true,
    ..Default::default()
};
let pipeline = EnhancementPipeline::new(config)?;
let enhanced = pipeline.process_frame(&frame)?;
```

## Feature Flags

- `onnx` - Enable ONNX Runtime for neural network inference
- `gpu` - Enable GPU acceleration via `transcode-gpu`

## Documentation

See the [main transcode documentation](../README.md) for the complete project overview.
