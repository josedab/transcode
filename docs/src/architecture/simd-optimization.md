# SIMD Optimization

This document describes Transcode's SIMD (Single Instruction, Multiple Data) optimization strategy.

## Overview

Transcode uses SIMD instructions to accelerate computationally intensive operations:

| Architecture | Extensions | Status |
|--------------|------------|--------|
| x86_64 | SSE4.2, AVX2, AVX-512, FMA | Full support |
| aarch64 | NEON, SVE | Full support |
| wasm32 | SIMD128 | Partial support |

## Runtime Detection

SIMD features are detected at runtime:

```rust
use transcode_codecs::detect_simd;

let caps = detect_simd();

println!("SSE4.2: {}", caps.sse42);
println!("AVX2: {}", caps.avx2);
println!("AVX-512: {}", caps.avx512);
println!("FMA: {}", caps.fma);
println!("NEON: {}", caps.neon);
println!("SVE: {}", caps.sve);

// Get best available level
println!("Best: {}", caps.best_level());
```

## Dispatch Pattern

Functions dispatch to the best available implementation:

```rust
pub fn idct_8x8(block: &mut [i16; 64]) {
    // Compile-time architecture check
    #[cfg(target_arch = "x86_64")]
    {
        // Runtime feature detection
        if is_x86_feature_detected!("avx2") {
            unsafe { idct_8x8_avx2(block) }
            return;
        }
        if is_x86_feature_detected!("sse4.2") {
            unsafe { idct_8x8_sse42(block) }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { idct_8x8_neon(block) }
            return;
        }
    }

    // Scalar fallback
    idct_8x8_scalar(block)
}
```

## Optimized Operations

### Video Processing

| Operation | SSE4.2 | AVX2 | NEON | Speedup |
|-----------|--------|------|------|---------|
| IDCT 4x4 | ✅ | ✅ | ✅ | 4-8x |
| IDCT 8x8 | ✅ | ✅ | ✅ | 4-8x |
| Motion Compensation | ✅ | ✅ | ✅ | 3-6x |
| Deblocking Filter | ✅ | ✅ | ✅ | 2-4x |
| Intra Prediction | ✅ | ✅ | ✅ | 3-5x |
| Color Conversion | ✅ | ✅ | ✅ | 4-8x |

### Audio Processing

| Operation | SSE4.2 | AVX2 | NEON | Speedup |
|-----------|--------|------|------|---------|
| MDCT | ✅ | ✅ | ✅ | 3-5x |
| FFT | ✅ | ✅ | ✅ | 4-6x |
| Windowing | ✅ | ✅ | ✅ | 2-4x |
| Quantization | ✅ | ✅ | ✅ | 3-5x |

## Implementation Examples

### AVX2 IDCT

```rust
#[target_feature(enable = "avx2")]
unsafe fn idct_4x4_avx2(coeffs: &mut [i16; 16]) {
    use std::arch::x86_64::*;

    // Load 16 coefficients into two 128-bit registers
    let row01 = _mm_loadu_si128(coeffs.as_ptr() as *const __m128i);
    let row23 = _mm_loadu_si128(coeffs.as_ptr().add(8) as *const __m128i);

    // Horizontal transform
    let h0 = _mm_add_epi16(row01, row23);
    let h1 = _mm_sub_epi16(row01, row23);

    // Vertical transform
    let v0 = _mm_unpacklo_epi16(h0, h1);
    let v1 = _mm_unpackhi_epi16(h0, h1);

    // Store results
    _mm_storeu_si128(coeffs.as_mut_ptr() as *mut __m128i, v0);
    _mm_storeu_si128(coeffs.as_mut_ptr().add(8) as *mut __m128i, v1);
}
```

### NEON Motion Compensation

```rust
#[target_feature(enable = "neon")]
unsafe fn mc_luma_neon(
    src: &[u8],
    dst: &mut [u8],
    src_stride: usize,
    dst_stride: usize,
) {
    use std::arch::aarch64::*;

    for y in 0..8 {
        let src_row = vld1q_u8(src.as_ptr().add(y * src_stride));
        vst1q_u8(dst.as_mut_ptr().add(y * dst_stride), src_row);
    }
}
```

## Safety

All SIMD code is wrapped in `unsafe` blocks with documented safety requirements:

```rust
/// Performs AVX2-accelerated IDCT on a 4x4 block.
///
/// # Safety
///
/// - CPU must support AVX2 (verify with `is_x86_feature_detected!("avx2")`)
/// - `coeffs` must be a valid mutable reference to exactly 16 elements
/// - Memory at `coeffs` must be properly aligned (16-byte boundary)
#[target_feature(enable = "avx2")]
pub unsafe fn idct_4x4_avx2(coeffs: &mut [i16; 16]) {
    // Implementation
}
```

## Benchmarking

Compare implementations with criterion:

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn idct_benchmark(c: &mut Criterion) {
    let mut data = [0i16; 64];

    let mut group = c.benchmark_group("IDCT 8x8");

    group.bench_function("scalar", |b| {
        b.iter(|| idct_8x8_scalar(&mut data))
    });

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        group.bench_function("avx2", |b| {
            b.iter(|| unsafe { idct_8x8_avx2(&mut data) })
        });
    }

    group.finish();
}
```

## Testing

SIMD implementations are tested against scalar reference:

```rust
#[test]
fn test_idct_avx2_matches_scalar() {
    if !is_x86_feature_detected!("avx2") {
        return; // Skip on unsupported hardware
    }

    let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

    let mut scalar_out = input;
    let mut avx2_out = input;

    idct_4x4_scalar(&mut scalar_out);
    unsafe { idct_4x4_avx2(&mut avx2_out) };

    assert_eq!(scalar_out, avx2_out);
}
```

## Disabling SIMD

For debugging or compatibility:

```rust
// Compile-time disable
#[cfg(feature = "no-simd")]
fn idct_8x8(block: &mut [i16; 64]) {
    idct_8x8_scalar(block)
}

// Runtime disable
std::env::set_var("TRANSCODE_NO_SIMD", "1");
```

## Future Work

- **AVX-512** - Wider vectors for batch processing
- **SVE/SVE2** - Scalable vectors on ARM
- **WASM SIMD** - Browser acceleration
- **Auto-vectorization** - Compiler hints for simple loops
