//! Compute shader definitions and management.

use crate::error::{GpuError, Result};
use crate::GpuContext;
use std::collections::HashMap;

/// Shader type identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderKind {
    /// Bilinear scaling.
    ScaleBilinear,
    /// Bicubic scaling.
    ScaleBicubic,
    /// Lanczos scaling.
    ScaleLanczos,
    /// YUV to RGB conversion.
    YuvToRgb,
    /// RGB to YUV conversion.
    RgbToYuv,
    /// Gaussian blur.
    GaussianBlur,
    /// Sharpen filter.
    Sharpen,
    /// Denoise filter.
    Denoise,
    /// Color adjustment.
    ColorAdjust,
    /// Grayscale conversion.
    Grayscale,
}

/// Registry of compiled shaders.
pub struct ShaderRegistry {
    modules: HashMap<ShaderKind, wgpu::ShaderModule>,
}

impl ShaderRegistry {
    /// Create a new shader registry.
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
        }
    }

    /// Load and compile all shaders.
    pub fn load_all(context: &GpuContext) -> Result<Self> {
        let mut registry = Self::new();

        // Load each shader
        registry.load(context, ShaderKind::ScaleBilinear, SHADER_SCALE_BILINEAR)?;
        registry.load(context, ShaderKind::ScaleBicubic, SHADER_SCALE_BICUBIC)?;
        registry.load(context, ShaderKind::YuvToRgb, SHADER_YUV_TO_RGB)?;
        registry.load(context, ShaderKind::RgbToYuv, SHADER_RGB_TO_YUV)?;
        registry.load(context, ShaderKind::GaussianBlur, SHADER_GAUSSIAN_BLUR)?;
        registry.load(context, ShaderKind::Sharpen, SHADER_SHARPEN)?;
        registry.load(context, ShaderKind::ColorAdjust, SHADER_COLOR_ADJUST)?;
        registry.load(context, ShaderKind::Grayscale, SHADER_GRAYSCALE)?;

        Ok(registry)
    }

    /// Load a single shader.
    pub fn load(&mut self, context: &GpuContext, kind: ShaderKind, source: &str) -> Result<()> {
        let module = context.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{:?} Shader", kind)),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        self.modules.insert(kind, module);
        Ok(())
    }

    /// Get a compiled shader module.
    pub fn get(&self, kind: ShaderKind) -> Result<&wgpu::ShaderModule> {
        self.modules
            .get(&kind)
            .ok_or_else(|| GpuError::ShaderCompilationFailed(format!("{:?} shader not loaded", kind)))
    }
}

impl Default for ShaderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// WGSL Shader Source Code
// ============================================================================

/// Bilinear scaling shader.
pub const SHADER_SCALE_BILINEAR: &str = r#"
struct Params {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var tex_sampler: sampler;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dst_x = global_id.x;
    let dst_y = global_id.y;

    if dst_x >= params.dst_width || dst_y >= params.dst_height {
        return;
    }

    // Calculate normalized texture coordinates
    let u = (f32(dst_x) + 0.5) / f32(params.dst_width);
    let v = (f32(dst_y) + 0.5) / f32(params.dst_height);

    // Sample with bilinear filtering
    let color = textureSampleLevel(input_texture, tex_sampler, vec2<f32>(u, v), 0.0);

    textureStore(output_texture, vec2<i32>(i32(dst_x), i32(dst_y)), color);
}
"#;

/// Bicubic scaling shader.
pub const SHADER_SCALE_BICUBIC: &str = r#"
struct Params {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: Params;

fn cubic_weight(x: f32) -> f32 {
    let a = -0.5; // Catmull-Rom spline
    let ax = abs(x);
    if ax < 1.0 {
        return (a + 2.0) * ax * ax * ax - (a + 3.0) * ax * ax + 1.0;
    } else if ax < 2.0 {
        return a * ax * ax * ax - 5.0 * a * ax * ax + 8.0 * a * ax - 4.0 * a;
    }
    return 0.0;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dst_x = global_id.x;
    let dst_y = global_id.y;

    if dst_x >= params.dst_width || dst_y >= params.dst_height {
        return;
    }

    // Map destination to source coordinates
    let src_x = (f32(dst_x) + 0.5) * f32(params.src_width) / f32(params.dst_width) - 0.5;
    let src_y = (f32(dst_y) + 0.5) * f32(params.src_height) / f32(params.dst_height) - 0.5;

    let x0 = i32(floor(src_x));
    let y0 = i32(floor(src_y));
    let fx = src_x - f32(x0);
    let fy = src_y - f32(y0);

    var color = vec4<f32>(0.0);
    var total_weight = 0.0;

    // 4x4 bicubic sampling
    for (var j = -1; j <= 2; j++) {
        for (var i = -1; i <= 2; i++) {
            let px = clamp(x0 + i, 0, i32(params.src_width) - 1);
            let py = clamp(y0 + j, 0, i32(params.src_height) - 1);

            let sample = textureLoad(input_texture, vec2<i32>(px, py), 0);
            let weight = cubic_weight(f32(i) - fx) * cubic_weight(f32(j) - fy);

            color += sample * weight;
            total_weight += weight;
        }
    }

    color /= total_weight;
    textureStore(output_texture, vec2<i32>(i32(dst_x), i32(dst_y)), color);
}
"#;

/// YUV to RGB conversion shader (BT.709).
pub const SHADER_YUV_TO_RGB: &str = r#"
struct Params {
    width: u32,
    height: u32,
    color_space: u32, // 0=BT.601, 1=BT.709, 2=BT.2020
    _padding: u32,
}

@group(0) @binding(0) var y_texture: texture_2d<f32>;
@group(0) @binding(1) var uv_texture: texture_2d<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> params: Params;

// BT.709 coefficients
const KR_709: f32 = 0.2126;
const KB_709: f32 = 0.0722;

// BT.601 coefficients
const KR_601: f32 = 0.299;
const KB_601: f32 = 0.114;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if x >= params.width || y >= params.height {
        return;
    }

    // Load Y value
    let y_val = textureLoad(y_texture, vec2<i32>(i32(x), i32(y)), 0).r;

    // Load UV values (NV12 format: U and V are interleaved, half resolution)
    let uv_coord = vec2<i32>(i32(x / 2u), i32(y / 2u));
    let uv = textureLoad(uv_texture, uv_coord, 0);
    let u_val = uv.r - 0.5;
    let v_val = uv.g - 0.5;

    // YUV to RGB conversion
    var r: f32;
    var g: f32;
    var b: f32;

    if params.color_space == 1u {
        // BT.709
        r = y_val + 1.5748 * v_val;
        g = y_val - 0.1873 * u_val - 0.4681 * v_val;
        b = y_val + 1.8556 * u_val;
    } else {
        // BT.601 (default)
        r = y_val + 1.402 * v_val;
        g = y_val - 0.344136 * u_val - 0.714136 * v_val;
        b = y_val + 1.772 * u_val;
    }

    let color = vec4<f32>(
        clamp(r, 0.0, 1.0),
        clamp(g, 0.0, 1.0),
        clamp(b, 0.0, 1.0),
        1.0
    );

    textureStore(output_texture, vec2<i32>(i32(x), i32(y)), color);
}
"#;

/// RGB to YUV conversion shader (BT.709).
pub const SHADER_RGB_TO_YUV: &str = r#"
struct Params {
    width: u32,
    height: u32,
    color_space: u32,
    _padding: u32,
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var y_texture: texture_storage_2d<r8unorm, write>;
@group(0) @binding(2) var uv_texture: texture_storage_2d<rg8unorm, write>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if x >= params.width || y >= params.height {
        return;
    }

    let color = textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0);
    let r = color.r;
    let g = color.g;
    let b = color.b;

    // RGB to YUV (BT.709)
    var y_val: f32;
    var u_val: f32;
    var v_val: f32;

    if params.color_space == 1u {
        // BT.709
        y_val = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        u_val = -0.1146 * r - 0.3854 * g + 0.5 * b + 0.5;
        v_val = 0.5 * r - 0.4542 * g - 0.0458 * b + 0.5;
    } else {
        // BT.601
        y_val = 0.299 * r + 0.587 * g + 0.114 * b;
        u_val = -0.169 * r - 0.331 * g + 0.5 * b + 0.5;
        v_val = 0.5 * r - 0.419 * g - 0.081 * b + 0.5;
    }

    // Write Y plane
    textureStore(y_texture, vec2<i32>(i32(x), i32(y)), vec4<f32>(y_val, 0.0, 0.0, 1.0));

    // Write UV plane (subsampled 4:2:0)
    if x % 2u == 0u && y % 2u == 0u {
        // Average 2x2 block for UV
        var u_sum = u_val;
        var v_sum = v_val;
        var count = 1.0;

        if x + 1u < params.width {
            let c1 = textureLoad(input_texture, vec2<i32>(i32(x) + 1, i32(y)), 0);
            u_sum += -0.1146 * c1.r - 0.3854 * c1.g + 0.5 * c1.b + 0.5;
            v_sum += 0.5 * c1.r - 0.4542 * c1.g - 0.0458 * c1.b + 0.5;
            count += 1.0;
        }
        if y + 1u < params.height {
            let c2 = textureLoad(input_texture, vec2<i32>(i32(x), i32(y) + 1), 0);
            u_sum += -0.1146 * c2.r - 0.3854 * c2.g + 0.5 * c2.b + 0.5;
            v_sum += 0.5 * c2.r - 0.4542 * c2.g - 0.0458 * c2.b + 0.5;
            count += 1.0;
        }
        if x + 1u < params.width && y + 1u < params.height {
            let c3 = textureLoad(input_texture, vec2<i32>(i32(x) + 1, i32(y) + 1), 0);
            u_sum += -0.1146 * c3.r - 0.3854 * c3.g + 0.5 * c3.b + 0.5;
            v_sum += 0.5 * c3.r - 0.4542 * c3.g - 0.0458 * c3.b + 0.5;
            count += 1.0;
        }

        let uv_coord = vec2<i32>(i32(x / 2u), i32(y / 2u));
        textureStore(uv_texture, uv_coord, vec4<f32>(u_sum / count, v_sum / count, 0.0, 1.0));
    }
}
"#;

/// Gaussian blur shader.
pub const SHADER_GAUSSIAN_BLUR: &str = r#"
struct Params {
    width: u32,
    height: u32,
    radius: u32,
    sigma: f32,
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: Params;

fn gaussian(x: f32, sigma: f32) -> f32 {
    return exp(-(x * x) / (2.0 * sigma * sigma)) / (2.506628 * sigma);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if x >= params.width || y >= params.height {
        return;
    }

    var color = vec4<f32>(0.0);
    var total_weight = 0.0;

    let radius = i32(params.radius);

    for (var dy = -radius; dy <= radius; dy++) {
        for (var dx = -radius; dx <= radius; dx++) {
            let px = clamp(i32(x) + dx, 0, i32(params.width) - 1);
            let py = clamp(i32(y) + dy, 0, i32(params.height) - 1);

            let sample = textureLoad(input_texture, vec2<i32>(px, py), 0);
            let dist = sqrt(f32(dx * dx + dy * dy));
            let weight = gaussian(dist, params.sigma);

            color += sample * weight;
            total_weight += weight;
        }
    }

    color /= total_weight;
    textureStore(output_texture, vec2<i32>(i32(x), i32(y)), color);
}
"#;

/// Sharpen filter shader.
pub const SHADER_SHARPEN: &str = r#"
struct Params {
    width: u32,
    height: u32,
    strength: f32,
    _padding: u32,
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if x >= params.width || y >= params.height {
        return;
    }

    let ix = i32(x);
    let iy = i32(y);

    // Unsharp mask kernel
    let center = textureLoad(input_texture, vec2<i32>(ix, iy), 0);

    let left = textureLoad(input_texture, vec2<i32>(max(ix - 1, 0), iy), 0);
    let right = textureLoad(input_texture, vec2<i32>(min(ix + 1, i32(params.width) - 1), iy), 0);
    let top = textureLoad(input_texture, vec2<i32>(ix, max(iy - 1, 0)), 0);
    let bottom = textureLoad(input_texture, vec2<i32>(ix, min(iy + 1, i32(params.height) - 1)), 0);

    // Laplacian
    let laplacian = 4.0 * center - left - right - top - bottom;

    // Sharpen
    let sharpened = center + params.strength * laplacian;

    let color = vec4<f32>(
        clamp(sharpened.r, 0.0, 1.0),
        clamp(sharpened.g, 0.0, 1.0),
        clamp(sharpened.b, 0.0, 1.0),
        1.0
    );

    textureStore(output_texture, vec2<i32>(ix, iy), color);
}
"#;

/// Color adjustment shader.
pub const SHADER_COLOR_ADJUST: &str = r#"
struct Params {
    width: u32,
    height: u32,
    brightness: f32,
    contrast: f32,
    saturation: f32,
    gamma: f32,
    _padding: vec2<f32>,
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if x >= params.width || y >= params.height {
        return;
    }

    var color = textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0);

    // Brightness
    color = vec4<f32>(color.rgb + params.brightness, color.a);

    // Contrast
    color = vec4<f32>((color.rgb - 0.5) * params.contrast + 0.5, color.a);

    // Saturation
    let luminance = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    color = vec4<f32>(mix(vec3<f32>(luminance), color.rgb, params.saturation), color.a);

    // Gamma correction
    let inv_gamma = 1.0 / params.gamma;
    color = vec4<f32>(pow(max(color.rgb, vec3<f32>(0.0)), vec3<f32>(inv_gamma)), color.a);

    // Clamp
    color = vec4<f32>(
        clamp(color.r, 0.0, 1.0),
        clamp(color.g, 0.0, 1.0),
        clamp(color.b, 0.0, 1.0),
        1.0
    );

    textureStore(output_texture, vec2<i32>(i32(x), i32(y)), color);
}
"#;

/// Grayscale conversion shader.
pub const SHADER_GRAYSCALE: &str = r#"
struct Params {
    width: u32,
    height: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if x >= params.width || y >= params.height {
        return;
    }

    let color = textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0);

    // Luminance (BT.709)
    let gray = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));

    textureStore(output_texture, vec2<i32>(i32(x), i32(y)), vec4<f32>(gray, gray, gray, 1.0));
}
"#;
