# ADR-0007: AI Enhancement Pipeline Architecture

## Status

Accepted

## Date

2024-05 (inferred from module structure)

## Context

Modern video enhancement increasingly relies on neural network-based processing:

1. **Super Resolution**: Upscale video to higher resolutions (2x, 4x)
2. **Denoising**: Remove noise while preserving details
3. **Frame Interpolation**: Generate intermediate frames for higher frame rates
4. **Edge Enhancement**: Sharpen and enhance edges

We need to support:

- Multiple AI models (RealESRGAN, RIFE, custom models)
- Multiple hardware accelerators (CPU, CUDA, CoreML, DirectML)
- Composable enhancement pipelines
- Efficient batch processing

The challenge is that AI inference is computationally expensive and requires careful orchestration to achieve acceptable performance.

## Decision

Use **ONNX Runtime** for neural network inference with a **chain-of-responsibility** pattern for composable enhancement pipelines.

### 1. ONNX Runtime Integration

ONNX Runtime provides portable model execution with multiple backends:

```rust
pub enum ExecutionProvider {
    Cpu,
    Cuda { device_id: i32 },
    TensorRT { device_id: i32, fp16: bool, max_workspace_size: usize },
    CoreML { use_neural_engine: bool, only_neural_engine: bool },
    DirectML { device_id: i32 },
    OpenVINO { device_type: String },
    ROCm { device_id: i32 },
}

pub struct OnnxConfig {
    pub providers: Vec<ExecutionProvider>,
    pub optimization_level: OptimizationLevel,
    pub intra_op_threads: usize,
    pub inter_op_threads: usize,
    pub enable_memory_pattern: bool,
}
```

### 2. Automatic Provider Selection

Detect and use the best available hardware:

```rust
impl ExecutionProvider {
    pub fn best_available() -> Self {
        #[cfg(target_os = "macos")]
        {
            let coreml = Self::CoreML {
                use_neural_engine: true,
                only_neural_engine: false,
            };
            if coreml.is_available() {
                return coreml;
            }
        }

        #[cfg(target_os = "windows")]
        {
            let dml = Self::DirectML { device_id: 0 };
            if dml.is_available() {
                return dml;
            }
        }

        let cuda = Self::Cuda { device_id: 0 };
        if cuda.is_available() {
            return cuda;
        }

        Self::Cpu
    }
}
```

### 3. Composable Enhancement Pipeline

Chain multiple enhancers with configurable order:

```rust
pub struct EnhancementPipeline {
    upscaler: Option<Upscaler>,
    denoiser: Option<Denoiser>,
    interpolator: Option<FrameInterpolator>,
    config: PipelineConfig,
}

pub enum ProcessingOrder {
    DenoiseUpscaleInterpolate,  // Best quality
    UpscaleDenoiseInterpolate,  // Alternative order
    Custom,
}

impl EnhancementPipeline {
    pub fn process_frame(&self, frame: &Frame) -> Result<Frame> {
        let mut result = frame.clone();

        match self.config.order {
            ProcessingOrder::DenoiseUpscaleInterpolate => {
                if let Some(ref denoiser) = self.denoiser {
                    result = denoiser.process(&result)?;
                }
                if let Some(ref upscaler) = self.upscaler {
                    result = upscaler.process(&result)?;
                }
            }
            // ...
        }

        Ok(result)
    }
}
```

### 4. Individual Enhancers

Each enhancer is independently configurable:

```rust
// Upscaling
pub struct UpscalerConfig {
    pub scale_factor: ScaleFactor,  // X2, X4
    pub model: UpscaleModel,        // Lanczos, Bilinear, RealESRGAN
}

// Denoising
pub struct DenoiserConfig {
    pub noise_level: NoiseLevel,    // Low, Medium, High
    pub algorithm: DenoiseAlgorithm, // Bilateral, NLMeans
}

// Frame Interpolation
pub struct InterpolatorConfig {
    pub mode: InterpolationMode,    // Linear, OpticalFlow
    pub target_multiplier: f32,     // 2x, 4x frame rate
}
```

### 5. Model Loading and Caching

Efficient model management:

```rust
pub struct OnnxSession {
    session: ort::Session,
    inputs: Vec<TensorInfo>,
    outputs: Vec<TensorInfo>,
    metadata: ModelMetadata,
    active_provider: ExecutionProvider,
}

impl OnnxSession {
    pub fn load(path: impl AsRef<Path>, config: OnnxConfig) -> Result<Self> {
        // Build session with fallback providers
        let mut builder = ort::Session::builder()?;

        for provider in &config.providers {
            match Self::register_provider(&mut builder, provider) {
                Ok(()) => break,
                Err(e) => warn!("Provider {} unavailable: {}", provider.name(), e),
            }
        }

        let session = builder.commit_from_file(path)?;
        // ...
    }
}
```

### Pipeline Topology

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Denoiser   │────▶│  Upscaler   │────▶│Interpolator │
│  (Optional) │     │  (Optional) │     │  (Optional) │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
   ONNX Session       ONNX Session       ONNX Session
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────┐
│              Execution Provider                      │
│  (CUDA / TensorRT / CoreML / DirectML / CPU)        │
└─────────────────────────────────────────────────────┘
```

## Consequences

### Positive

1. **Portable inference**: ONNX models run on any platform with ONNX Runtime

2. **Hardware acceleration**: Automatic use of GPU, NPU, or specialized hardware

3. **Composable pipelines**: Mix and match enhancement stages as needed

4. **Ecosystem compatibility**: Use pre-trained models from PyTorch, TensorFlow, etc.

5. **Fallback support**: CPU execution when accelerators unavailable

6. **Memory efficiency**: Batch processing and memory pattern optimization

### Negative

1. **Model size**: Neural network models can be large (100MB+)

2. **Inference latency**: Real-time processing challenging for complex models

3. **ONNX Runtime dependency**: Adds native library dependencies

4. **Model compatibility**: Not all PyTorch/TF features convert to ONNX

### Mitigations

1. **Model compression**: Support quantized (INT8) and FP16 models

```rust
pub fn with_provider(self, provider: ExecutionProvider) -> Self {
    // TensorRT with FP16 for faster inference
    ExecutionProvider::TensorRT {
        device_id: 0,
        fp16: true,
        max_workspace_size: 1 << 30,
    }
}
```

2. **Lazy loading**: Load models on first use, not at startup

3. **Batch processing**: Process multiple frames per inference call

4. **Optional feature**: AI module behind feature flag

## Alternatives Considered

### Alternative 1: PyTorch via Bindings

Use PyTorch directly through C++ bindings (libtorch).

Rejected because:
- Large dependency (~2GB)
- Complex build process
- Python ecosystem fragmentation
- No WebAssembly support

### Alternative 2: TensorFlow Lite

Use TensorFlow Lite for inference.

Rejected because:
- Limited model support
- Fewer hardware accelerators
- Less active Rust ecosystem

### Alternative 3: Custom Inference Engine

Build inference from scratch.

Rejected because:
- Enormous development effort
- Miss hardware optimizations
- Reinventing well-solved problems

### Alternative 4: External Process

Shell out to Python for AI processing.

Rejected because:
- Process overhead
- Complex data serialization
- Deployment complexity
- No WebAssembly support

## References

- [ONNX Runtime](https://onnxruntime.ai/)
- [ONNX format specification](https://onnx.ai/)
- [RealESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [RIFE frame interpolation](https://github.com/hzwer/ECCV2022-RIFE)
