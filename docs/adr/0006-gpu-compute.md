# ADR-0006: GPU Compute via wgpu

## Status

Accepted

## Date

2024-04 (inferred from module structure)

## Context

Video processing involves computationally intensive operations that can benefit from GPU acceleration:

1. **Color space conversion** (YUV ↔ RGB) - pixel-parallel operations
2. **Scaling and resizing** - interpolation algorithms
3. **Frame filtering** - blur, sharpen, denoise kernels
4. **Pixel format conversion** - data transformation

These operations are embarrassingly parallel and map well to GPU compute workloads. However, we need to support:

- **Cross-platform deployment** (Linux, Windows, macOS)
- **WebAssembly targets** for browser-based processing
- **Multiple GPU vendors** (NVIDIA, AMD, Intel, Apple)
- **Graceful fallback** when GPU is unavailable

Traditional approaches like CUDA or OpenCL have significant drawbacks:
- CUDA is NVIDIA-only
- OpenCL has inconsistent driver support
- Neither works in WebAssembly

## Decision

Use **wgpu** as the GPU abstraction layer with compute shaders written in **WGSL**.

### 1. wgpu for Cross-Platform GPU Access

wgpu provides a unified API that maps to native backends:

```rust
pub struct GpuContext {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    capabilities: GpuCapabilities,
}

impl GpuContext {
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        // ...
    }
}
```

### 2. Backend Mapping

wgpu automatically selects the best available backend:

| Platform | Backends |
|----------|----------|
| Windows | Vulkan, DX12 |
| macOS | Metal |
| Linux | Vulkan |
| WebAssembly | WebGPU |

### 3. WGSL Compute Shaders

All shaders are written in WGSL (WebGPU Shading Language):

```wgsl
@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: ColorParams;

@compute @workgroup_size(16, 16)
fn yuv_to_rgb(@builtin(global_invocation_id) id: vec3<u32>) {
    let y = textureLoad(input_texture, id.xy, 0).r;
    let u = textureLoad(input_texture, id.xy / 2, 1).r - 0.5;
    let v = textureLoad(input_texture, id.xy / 2, 2).r - 0.5;

    let r = y + 1.402 * v;
    let g = y - 0.344 * u - 0.714 * v;
    let b = y + 1.772 * u;

    textureStore(output_texture, id.xy, vec4(r, g, b, 1.0));
}
```

### 4. Capability Detection

Query GPU capabilities at runtime:

```rust
pub struct GpuCapabilities {
    pub device_name: String,
    pub backend: String,
    pub max_texture_dimension: u32,
    pub max_buffer_size: u64,
    pub max_workgroup_size: [u32; 3],
    pub supports_f16: bool,
}
```

### 5. Supported Operations

| Operation | Implementation |
|-----------|----------------|
| Color conversion | YUV↔RGB compute shaders |
| Scaling | Bilinear, Bicubic, Lanczos shaders |
| Color adjustment | Brightness, contrast, saturation |
| Format conversion | Planar ↔ packed pixel formats |

## Consequences

### Positive

1. **True cross-platform support**: Single codebase for all platforms including WebAssembly

2. **No vendor lock-in**: Works with NVIDIA, AMD, Intel, and Apple GPUs

3. **Future-proofed**: WebGPU is the emerging standard for GPU compute in browsers

4. **Memory safety**: wgpu provides a safe Rust API with validation layers

5. **Shader portability**: WGSL compiles to SPIR-V, MSL, or DXIL as needed

6. **Consistent performance**: Similar performance across backends for typical workloads

### Negative

1. **Learning curve**: WGSL is newer and less documented than GLSL/HLSL

2. **Feature limitations**: Some advanced GPU features not exposed by wgpu

3. **Async initialization**: GPU context creation requires async runtime

4. **Memory overhead**: Additional abstraction layer vs. direct GPU APIs

### Mitigations

1. **CPU fallback**: All operations have CPU implementations for when GPU is unavailable

```rust
pub trait VideoProcessor {
    fn process(&self, frame: &Frame) -> Result<Frame>;

    fn has_gpu_support(&self) -> bool;
}

impl VideoProcessor for Scaler {
    fn process(&self, frame: &Frame) -> Result<Frame> {
        if let Some(ref gpu) = self.gpu_context {
            self.scale_gpu(gpu, frame)
        } else {
            self.scale_cpu(frame)
        }
    }
}
```

2. **Shader caching**: Compile shaders once and cache pipelines

3. **Batch processing**: Process multiple frames per GPU dispatch

## Alternatives Considered

### Alternative 1: CUDA

Use NVIDIA CUDA for GPU compute.

Rejected because:
- NVIDIA-only (excludes AMD, Intel, Apple)
- No WebAssembly support
- Requires CUDA toolkit installation
- Does not align with pure Rust goal

### Alternative 2: OpenCL

Use OpenCL for cross-vendor GPU compute.

Rejected because:
- Inconsistent driver support across vendors
- Poor macOS support (deprecated by Apple)
- No WebAssembly support
- Complex memory management

### Alternative 3: Vulkan Compute Only

Use Vulkan compute shaders directly.

Rejected because:
- No macOS support (requires MoltenVK translation)
- Extremely verbose API
- No WebAssembly support
- Higher implementation complexity

### Alternative 4: No GPU Acceleration

Keep all processing on CPU.

Rejected because:
- Significantly slower for real-time processing
- Poor resource utilization on modern hardware
- Competitive disadvantage for supported operations

## References

- [wgpu documentation](https://wgpu.rs/)
- [WebGPU specification](https://www.w3.org/TR/webgpu/)
- [WGSL specification](https://www.w3.org/TR/WGSL/)
- [GPU compute for video processing](https://developer.nvidia.com/video-processing)
