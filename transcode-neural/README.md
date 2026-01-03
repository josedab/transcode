# transcode-neural

Neural network-based video super-resolution for the transcode project. This crate provides AI-powered video upscaling using ONNX runtime with fallback to traditional interpolation methods.

## Features

- **Neural Super-Resolution**: ONNX-based inference for high-quality upscaling
- **Multiple Models**: Support for Real-ESRGAN, BSRGAN, SwinIR, and custom models
- **GPU Acceleration**: Optional GPU support for faster inference
- **Tiled Processing**: Handle large images via configurable tile sizes
- **Fallback Algorithms**: Bicubic, bilinear, and nearest-neighbor interpolation
- **Color Space Conversion**: YUV420 to RGB and back for video processing

## Key Types

| Type | Description |
|------|-------------|
| `NeuralUpscaler` | Main upscaler using ONNX runtime |
| `NeuralConfig` | Configuration (model, scale, GPU, tile size) |
| `NeuralFrame` | RGB frame data in HWC layout |
| `ModelType` | Enum of supported models |
| `ModelManager` | Model path resolution and caching |
| `CpuUpscaler` | CPU fallback with traditional algorithms |
| `UpscaleAlgorithm` | Nearest, Bilinear, Bicubic, Lanczos, Neural |
| `NeuralError` | Error types for neural operations |

## Supported Models

- `RealEsrgan` - General-purpose 4x upscaling
- `RealEsrganAnime` - Optimized for anime content (4x)
- `Bsrgan` - Blind super-resolution (4x)
- `SwinIR` - Transformer-based upscaling (2x)
- `Custom(path)` - User-provided ONNX models

## Usage

### Basic Upscaling

```rust
use transcode_neural::{NeuralConfig, NeuralUpscaler, NeuralFrame, ModelType};

// Configure the upscaler
let config = NeuralConfig {
    model: ModelType::RealEsrgan,
    scale: 2,
    use_gpu: true,
    device_id: 0,
    tile_size: 512,
    tile_overlap: 32,
    batch_size: 1,
};

// Create upscaler and process frame
let upscaler = NeuralUpscaler::new(config)?;
let input = NeuralFrame::new(640, 480);
let output = upscaler.upscale(&input)?;
// output is now 1280x960
```

### From YUV420 Video Data

```rust
use transcode_neural::NeuralFrame;

// Convert from YUV420 (common video format)
let frame = NeuralFrame::from_yuv420(&y_plane, &u_plane, &v_plane, 1920, 1080);

// Process the frame...

// Convert back to YUV420
let (y, u, v) = frame.to_yuv420();
```

### CPU Fallback Algorithms

```rust
use transcode_neural::{CpuUpscaler, NeuralFrame};

let frame = NeuralFrame::new(320, 240);

// Different interpolation methods
let nearest = CpuUpscaler::nearest(&frame, 640, 480)?;
let bilinear = CpuUpscaler::bilinear(&frame, 640, 480)?;
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `onnx` | Enable ONNX runtime for neural inference |

```toml
[dependencies]
transcode-neural = { version = "0.1", features = ["onnx"] }
```

## Dependencies

- `ndarray` - N-dimensional arrays for tensor operations
- `image` - Image processing utilities
- `ort` (optional) - ONNX Runtime bindings
- `transcode-core` - Core types from the transcode project

## License

See the repository root for license information.
