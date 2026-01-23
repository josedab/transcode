---
slug: simd-optimization-guide
title: "SIMD Optimization in Transcode: A Deep Dive"
authors: [transcode-team]
tags: [technical, performance]
---

Modern CPUs include powerful SIMD (Single Instruction, Multiple Data) instructions that can dramatically accelerate video processing. Transcode automatically detects and uses the best available SIMD instructions on your hardware.

In this post, we'll explore how SIMD optimization works in Transcode and the performance gains you can expect.

<!-- truncate -->

## What is SIMD?

SIMD instructions process multiple data elements with a single instruction. For video processing, this means operating on multiple pixels simultaneously:

```
Scalar (1 pixel at a time):
  pixel[0] = pixel[0] + 10
  pixel[1] = pixel[1] + 10
  pixel[2] = pixel[2] + 10
  pixel[3] = pixel[3] + 10

SIMD (4 pixels at once):
  pixels[0..4] = pixels[0..4] + [10, 10, 10, 10]
```

## SIMD Instruction Sets

Transcode supports multiple SIMD instruction sets:

| Instruction Set | Width | Platform | Speedup |
|----------------|-------|----------|---------|
| SSE4.2 | 128-bit | x86_64 | ~2x |
| AVX2 | 256-bit | x86_64 | ~3x |
| AVX-512 | 512-bit | x86_64 | ~4x |
| NEON | 128-bit | ARM64 | ~2.5x |

## Runtime Detection

Transcode automatically detects available SIMD features at runtime:

```rust
use transcode_codecs::detect_simd;

let caps = detect_simd();
println!("SSE4.2: {}", caps.sse42);
println!("AVX2: {}", caps.avx2);
println!("AVX-512: {}", caps.avx512);
println!("NEON: {}", caps.neon);
println!("Best available: {}", caps.best_level());
```

CLI output:
```bash
$ transcode info --simd
SIMD Capabilities:
  SSE4.2:  yes
  AVX2:    yes
  AVX-512: no
  FMA:     yes
  NEON:    no (x86_64 platform)

Best level: AVX2
```

## How We Use SIMD

### Color Space Conversion

Converting YUV to RGB is a perfect SIMD target—each pixel is independent:

```rust
// Simplified AVX2 YUV to RGB conversion
#[cfg(target_arch = "x86_64")]
unsafe fn yuv_to_rgb_avx2(y: &[u8], u: &[u8], v: &[u8], rgb: &mut [u8]) {
    use std::arch::x86_64::*;

    // Process 32 pixels at once
    let y_vec = _mm256_loadu_si256(y.as_ptr() as *const __m256i);

    // ... conversion math using SIMD operations ...

    _mm256_storeu_si256(rgb.as_mut_ptr() as *mut __m256i, result);
}
```

### DCT Transform

The Discrete Cosine Transform (DCT) used in H.264/H.265 benefits greatly from SIMD:

```rust
// 8x8 DCT transform processes 8 values simultaneously
fn forward_dct_8x8_avx2(input: &[i16; 64], output: &mut [i16; 64]) {
    // Each row/column transform uses AVX2 for 8-way parallelism
    // ...
}
```

### Motion Compensation

Interpolating between reference frames:

```rust
// Bilinear interpolation with SIMD
fn interpolate_avx2(ref_frame: &[u8], dst: &mut [u8], mv: MotionVector) {
    // Process 16 pixels per iteration
    // ...
}
```

## Performance Impact

Real-world speedups for H.264 encoding (1080p, medium preset):

| Operation | Scalar | SSE4.2 | AVX2 | AVX-512 |
|-----------|--------|--------|------|---------|
| DCT | 28 fps | 52 fps | 78 fps | 92 fps |
| Motion Est. | 15 fps | 28 fps | 45 fps | 58 fps |
| Deblock | 40 fps | 75 fps | 110 fps | 140 fps |

## Safe SIMD Abstractions

Transcode wraps unsafe SIMD intrinsics in safe abstractions:

```rust
pub struct SimdVec256 {
    // Internal AVX2 vector
}

impl SimdVec256 {
    /// Safely load from a slice, with bounds checking
    pub fn load(data: &[u8]) -> Result<Self> {
        if data.len() < 32 {
            return Err(Error::InsufficientData);
        }
        // Safe wrapper around _mm256_loadu_si256
        Ok(unsafe { Self::load_unchecked(data) })
    }

    /// Add two vectors
    pub fn add(self, other: Self) -> Self {
        // Safe wrapper around _mm256_add_epi8
        unsafe { Self::add_unchecked(self, other) }
    }
}
```

## Fallback Behavior

When SIMD isn't available, Transcode uses optimized scalar code:

```rust
fn process_pixels(data: &mut [u8]) {
    let caps = detect_simd();

    match caps.best_level() {
        SimdLevel::Avx512 => process_avx512(data),
        SimdLevel::Avx2 => process_avx2(data),
        SimdLevel::Sse42 => process_sse42(data),
        SimdLevel::Neon => process_neon(data),
        SimdLevel::Scalar => process_scalar(data),
    }
}
```

This ensures Transcode works on any hardware while maximizing performance where possible.

## Compiling for Specific Targets

For maximum performance on known hardware:

```bash
# Compile for your specific CPU
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Or target a specific feature set
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo build --release
```

## Measuring SIMD Impact

Use the `--benchmark` flag to see SIMD impact:

```bash
$ transcode -i input.mp4 -o output.mp4 --benchmark

Encoding Performance:
  SIMD level: AVX2
  Average FPS: 78.3
  Peak FPS: 95.2

  Without SIMD (estimated): 28 fps
  SIMD speedup: 2.79x
```

## Conclusion

SIMD optimization is crucial for video processing performance. Transcode's automatic detection and safe abstractions let you benefit from SIMD without:

- Writing platform-specific code
- Dealing with unsafe intrinsics directly
- Worrying about compatibility

The result is portable, safe code that runs fast on any modern CPU.

---

*For more details on Transcode's performance, see our [benchmarks documentation](/docs/reference/benchmarks).*
