---
sidebar_position: 2
title: SIMD Optimization
description: Optimize performance with SIMD intrinsics
---

# SIMD Optimization

Transcode uses SIMD (Single Instruction, Multiple Data) instructions for high-performance video processing. This guide covers how SIMD is used and how to write SIMD-optimized code.

## Overview

SIMD enables processing multiple data elements with a single instruction:

```
// Scalar: 4 operations
a[0] + b[0] → c[0]
a[1] + b[1] → c[1]
a[2] + b[2] → c[2]
a[3] + b[3] → c[3]

// SIMD: 1 operation
[a[0], a[1], a[2], a[3]] + [b[0], b[1], b[2], b[3]] → [c[0], c[1], c[2], c[3]]
```

### Supported Architectures

| Architecture | Instruction Sets | Status |
|--------------|------------------|--------|
| x86_64 | SSE2, SSE4.1, AVX2, AVX-512 | ✅ |
| ARM64 | NEON | ✅ |
| WebAssembly | SIMD128 | ✅ |

## Using SIMD in Transcode

### Enable SIMD Feature

```toml
[dependencies]
transcode = { version = "1.0", features = ["simd"] }
```

### Runtime Detection

Transcode automatically detects and uses the best available SIMD instructions:

```rust
use transcode::simd::{SimdCapabilities, SimdLevel};

let caps = SimdCapabilities::detect();

println!("SIMD support:");
println!("  SSE2: {}", caps.has_sse2());
println!("  SSE4.1: {}", caps.has_sse41());
println!("  AVX2: {}", caps.has_avx2());
println!("  AVX-512: {}", caps.has_avx512());
println!("  NEON: {}", caps.has_neon());

// Get best available level
match caps.best_level() {
    SimdLevel::Avx512 => println!("Using AVX-512"),
    SimdLevel::Avx2 => println!("Using AVX2"),
    SimdLevel::Sse41 => println!("Using SSE4.1"),
    SimdLevel::Sse2 => println!("Using SSE2"),
    SimdLevel::Neon => println!("Using NEON"),
    SimdLevel::None => println!("Using scalar fallback"),
}
```

## SIMD in Video Processing

### Color Conversion

YUV to RGB conversion is heavily optimized with SIMD:

```rust
use transcode::simd::color::{yuv_to_rgb_simd, rgb_to_yuv_simd};

// Convert YUV frame to RGB
let rgb_data = yuv_to_rgb_simd(
    y_plane,
    u_plane,
    v_plane,
    width,
    height,
    ColorSpace::Bt709,
)?;
```

### Scaling

Image scaling uses SIMD for horizontal and vertical passes:

```rust
use transcode::simd::scale::{scale_bilinear_simd, scale_lanczos_simd};

let scaled = scale_lanczos_simd(
    input_data,
    input_width,
    input_height,
    output_width,
    output_height,
)?;
```

### DCT Transform

The Discrete Cosine Transform in H.264/HEVC uses SIMD:

```rust
use transcode::simd::transform::{dct_4x4_simd, idct_4x4_simd};

// Forward DCT
let coefficients = dct_4x4_simd(&block);

// Inverse DCT
let pixels = idct_4x4_simd(&coefficients);
```

## Writing SIMD Code

### Using std::simd (Nightly)

For the cleanest SIMD code, use Rust's portable SIMD:

```rust
#![feature(portable_simd)]
use std::simd::{f32x8, SimdFloat, SimdPartialOrd};

fn process_samples(input: &[f32], output: &mut [f32], gain: f32) {
    let gain_vec = f32x8::splat(gain);

    let chunks = input.chunks_exact(8);
    let mut out_chunks = output.chunks_exact_mut(8);

    for (inp, out) in chunks.zip(&mut out_chunks) {
        let v = f32x8::from_slice(inp);
        let result = v * gain_vec;
        result.copy_to_slice(out);
    }

    // Handle remainder
    let remainder = chunks.remainder();
    let out_remainder = out_chunks.into_remainder();
    for (i, o) in remainder.iter().zip(out_remainder.iter_mut()) {
        *o = *i * gain;
    }
}
```

### Using arch-specific Intrinsics

For stable Rust, use architecture-specific intrinsics:

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_arrays_avx2(a: &[f32], b: &[f32], c: &mut [f32]) {
    let len = a.len();
    let simd_len = len - (len % 8);

    for i in (0..simd_len).step_by(8) {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c.as_mut_ptr().add(i), vc);
    }

    // Scalar remainder
    for i in simd_len..len {
        c[i] = a[i] + b[i];
    }
}
```

### Multi-Architecture Support

Provide implementations for multiple architectures:

```rust
pub fn process_data(input: &[u8], output: &mut [u8]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { process_data_avx2(input, output) }
        return;
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]
    {
        unsafe { process_data_sse41(input, output) }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { process_data_neon(input, output) }
        return;
    }

    // Scalar fallback
    process_data_scalar(input, output)
}
```

### Runtime Dispatch

Dispatch based on runtime CPU detection:

```rust
use std::sync::OnceLock;

type ProcessFn = fn(&[u8], &mut [u8]);

static PROCESS_FN: OnceLock<ProcessFn> = OnceLock::new();

fn get_process_fn() -> ProcessFn {
    *PROCESS_FN.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return process_data_avx2_wrapper;
            }
            if is_x86_feature_detected!("sse4.1") {
                return process_data_sse41_wrapper;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON is always available on AArch64
            return process_data_neon_wrapper;
        }

        process_data_scalar
    })
}

pub fn process_data(input: &[u8], output: &mut [u8]) {
    let f = get_process_fn();
    f(input, output)
}
```

## Common SIMD Patterns

### Horizontal Sum

Sum all elements in a vector:

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
    // Sum pairs
    let sum1 = _mm256_hadd_ps(v, v);
    // Sum again
    let sum2 = _mm256_hadd_ps(sum1, sum1);
    // Extract low and high 128-bit lanes
    let low = _mm256_castps256_ps128(sum2);
    let high = _mm256_extractf128_ps(sum2, 1);
    // Add lanes
    let sum3 = _mm_add_ss(low, high);
    _mm_cvtss_f32(sum3)
}
```

### Clamp Values

Clamp values to a range:

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn clamp_avx2(v: __m256, min: __m256, max: __m256) -> __m256 {
    let clamped_low = _mm256_max_ps(v, min);
    _mm256_min_ps(clamped_low, max)
}
```

### Pack/Unpack

Convert between data types:

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn u8_to_i16_avx2(input: __m128i) -> __m256i {
    _mm256_cvtepu8_epi16(input)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn i16_to_u8_saturate_avx2(input: __m256i) -> __m128i {
    let packed = _mm256_packus_epi16(input, input);
    _mm256_castsi256_si128(_mm256_permute4x64_epi64(packed, 0xD8))
}
```

### Shuffle/Permute

Rearrange elements:

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn interleave_rgb_avx2(r: __m256i, g: __m256i, b: __m256i) -> [__m256i; 3] {
    // Interleave R, G, B into RGB RGB RGB...
    let rg_lo = _mm256_unpacklo_epi8(r, g);
    let rg_hi = _mm256_unpackhi_epi8(r, g);
    // ... complex shuffle logic
    [result0, result1, result2]
}
```

## SIMD in Audio Processing

### Sample Processing

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_gain_avx2(samples: &mut [f32], gain: f32) {
    let gain_vec = _mm256_set1_ps(gain);

    for chunk in samples.chunks_exact_mut(8) {
        let v = _mm256_loadu_ps(chunk.as_ptr());
        let result = _mm256_mul_ps(v, gain_vec);
        _mm256_storeu_ps(chunk.as_mut_ptr(), result);
    }
}
```

### Stereo to Mono

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn stereo_to_mono_avx2(stereo: &[f32], mono: &mut [f32]) {
    let half = _mm256_set1_ps(0.5);

    for (s, m) in stereo.chunks_exact(16).zip(mono.chunks_exact_mut(8)) {
        let left_right_0 = _mm256_loadu_ps(s.as_ptr());
        let left_right_1 = _mm256_loadu_ps(s.as_ptr().add(8));

        // Deinterleave and average
        let left = _mm256_shuffle_ps(left_right_0, left_right_1, 0x88);
        let right = _mm256_shuffle_ps(left_right_0, left_right_1, 0xDD);

        let sum = _mm256_add_ps(left, right);
        let avg = _mm256_mul_ps(sum, half);

        _mm256_storeu_ps(m.as_mut_ptr(), avg);
    }
}
```

## Benchmarking

### Using Criterion

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_color_conversion(c: &mut Criterion) {
    let width = 1920;
    let height = 1080;
    let y = vec![128u8; width * height];
    let u = vec![128u8; width * height / 4];
    let v = vec![128u8; width * height / 4];
    let mut rgb = vec![0u8; width * height * 3];

    c.bench_function("yuv_to_rgb_scalar", |b| {
        b.iter(|| {
            yuv_to_rgb_scalar(
                black_box(&y),
                black_box(&u),
                black_box(&v),
                black_box(&mut rgb),
                width,
                height,
            )
        })
    });

    c.bench_function("yuv_to_rgb_avx2", |b| {
        b.iter(|| unsafe {
            yuv_to_rgb_avx2(
                black_box(&y),
                black_box(&u),
                black_box(&v),
                black_box(&mut rgb),
                width,
                height,
            )
        })
    });
}

criterion_group!(benches, benchmark_color_conversion);
criterion_main!(benches);
```

### Typical Speedups

| Operation | Scalar | SSE4.1 | AVX2 | Speedup |
|-----------|--------|--------|------|---------|
| YUV→RGB (1080p) | 15ms | 4ms | 2ms | 7.5x |
| Scale (1080→4K) | 45ms | 12ms | 6ms | 7.5x |
| DCT 4x4 | 100ns | 25ns | 15ns | 6.7x |

## Best Practices

### 1. Always Provide Scalar Fallback

```rust
pub fn process(data: &mut [u8]) {
    if is_x86_feature_detected!("avx2") {
        unsafe { process_avx2(data) }
    } else {
        process_scalar(data)
    }
}
```

### 2. Align Data When Possible

```rust
#[repr(align(32))]
struct AlignedBuffer {
    data: [u8; 1024],
}
```

### 3. Process in Chunks

```rust
// Process 32 bytes at a time with AVX2
for chunk in data.chunks_exact_mut(32) {
    // SIMD processing
}

// Handle remainder with scalar code
for byte in data.chunks_exact_mut(32).into_remainder() {
    // Scalar processing
}
```

### 4. Minimize Memory Bandwidth

```rust
// Bad: Multiple passes over data
fn process_bad(data: &mut [u8]) {
    apply_filter1(data);  // Read + write all data
    apply_filter2(data);  // Read + write all data again
}

// Good: Single pass
fn process_good(data: &mut [u8]) {
    for chunk in data.chunks_exact_mut(32) {
        let v = load(chunk);
        let v = filter1(v);
        let v = filter2(v);
        store(chunk, v);
    }
}
```

### 5. Test Thoroughly

```rust
#[test]
fn test_simd_matches_scalar() {
    let input = generate_test_data();
    let mut output_scalar = vec![0u8; input.len()];
    let mut output_simd = vec![0u8; input.len()];

    process_scalar(&input, &mut output_scalar);
    unsafe { process_avx2(&input, &mut output_simd) };

    assert_eq!(output_scalar, output_simd);
}
```

## Next Steps

- [Hardware Acceleration](/docs/advanced/hardware-acceleration) - GPU encoding
- [Custom Codecs](/docs/advanced/custom-codecs) - Implement codecs
- [API Reference](/docs/reference/api) - Full API documentation
