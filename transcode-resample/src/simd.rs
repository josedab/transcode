//! SIMD-optimized filter operations.
//!
//! Provides optimized implementations for x86_64 (AVX2) and aarch64 (NEON).

/// Apply filter using AVX2 SIMD instructions.
///
/// # Safety
/// This function uses inline assembly and requires AVX2 support.
/// Check `is_x86_feature_detected!("avx2")` before calling.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn apply_filter_avx2_impl(
    input: &[f32],
    start_idx: usize,
    coeffs: &[f32],
    window_size: usize,
) -> f32 {
    use std::arch::x86_64::*;

    let end = (start_idx + window_size).min(input.len());
    let len = end.saturating_sub(start_idx).min(coeffs.len());

    if len == 0 {
        return 0.0;
    }

    let input_ptr = input.as_ptr().add(start_idx);
    let coeff_ptr = coeffs.as_ptr();

    // Process 8 elements at a time using AVX2
    let mut sum = _mm256_setzero_ps();
    let chunks = len / 8;
    let remainder = len % 8;

    for i in 0..chunks {
        let offset = i * 8;
        let inp = _mm256_loadu_ps(input_ptr.add(offset));
        let cof = _mm256_loadu_ps(coeff_ptr.add(offset));
        sum = _mm256_fmadd_ps(inp, cof, sum);
    }

    // Horizontal sum of AVX register
    // sum = [a, b, c, d, e, f, g, h]
    let high = _mm256_extractf128_ps(sum, 1);
    let low = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(low, high);

    // sum128 = [a+e, b+f, c+g, d+h]
    let high64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, high64);

    // sum64 = [a+e+c+g, b+f+d+h, ...]
    let high32 = _mm_shuffle_ps(sum64, sum64, 0x01);
    let sum32 = _mm_add_ss(sum64, high32);

    let mut result = _mm_cvtss_f32(sum32);

    // Handle remainder
    let base = chunks * 8;
    for i in 0..remainder {
        result += *input_ptr.add(base + i) * *coeff_ptr.add(base + i);
    }

    result
}

/// Safe wrapper for AVX2 filter application.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub fn apply_filter_avx2(
    input: &[f32],
    start_idx: usize,
    coeffs: &[f32],
    window_size: usize,
) -> f32 {
    // Safety: We check for AVX2 support at the call site
    unsafe { apply_filter_avx2_impl(input, start_idx, coeffs, window_size) }
}

/// Apply filter using NEON SIMD instructions.
///
/// # Safety
/// This function uses NEON intrinsics which are always available on aarch64.
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub fn apply_filter_neon(
    input: &[f32],
    start_idx: usize,
    coeffs: &[f32],
    window_size: usize,
) -> f32 {
    use std::arch::aarch64::*;

    let end = (start_idx + window_size).min(input.len());
    let len = end.saturating_sub(start_idx).min(coeffs.len());

    if len == 0 {
        return 0.0;
    }

    unsafe {
        let input_ptr = input.as_ptr().add(start_idx);
        let coeff_ptr = coeffs.as_ptr();

        // Process 4 elements at a time using NEON
        let mut sum = vdupq_n_f32(0.0);
        let chunks = len / 4;
        let remainder = len % 4;

        for i in 0..chunks {
            let offset = i * 4;
            let inp = vld1q_f32(input_ptr.add(offset));
            let cof = vld1q_f32(coeff_ptr.add(offset));
            sum = vfmaq_f32(sum, inp, cof);
        }

        // Horizontal sum of NEON register
        let sum2 = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
        let mut result = vget_lane_f32(vpadd_f32(sum2, sum2), 0);

        // Handle remainder
        let base = chunks * 4;
        for i in 0..remainder {
            result += *input_ptr.add(base + i) * *coeff_ptr.add(base + i);
        }

        result
    }
}

/// Scalar fallback implementation for filter application.
#[inline]
pub fn apply_filter_scalar(
    input: &[f32],
    start_idx: usize,
    coeffs: &[f32],
    window_size: usize,
) -> f32 {
    let end = (start_idx + window_size).min(input.len());
    let mut sum = 0.0f32;

    for i in start_idx..end {
        let coeff_idx = i - start_idx;
        if coeff_idx < coeffs.len() {
            sum += input[i] * coeffs[coeff_idx];
        }
    }

    sum
}

/// Batch apply filter for multiple phases (SIMD-optimized).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn batch_filter_avx2(
    input: &[f32],
    filter_bank: &[Vec<f32>],
    start_idx: usize,
    output: &mut [f32],
) {
    for (phase, out) in output.iter_mut().enumerate() {
        if phase < filter_bank.len() {
            *out = apply_filter_avx2_impl(input, start_idx, &filter_bank[phase], filter_bank[phase].len());
        }
    }
}

/// Linear interpolation using SIMD (for multiple samples).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn lerp_avx2(a: &[f32], b: &[f32], t: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    let len = a.len().min(b.len()).min(t.len()).min(output.len());
    let chunks = len / 8;
    let remainder = len % 8;

    let one = _mm256_set1_ps(1.0);

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
        let vt = _mm256_loadu_ps(t.as_ptr().add(offset));

        // result = a + (b - a) * t = a * (1 - t) + b * t
        let one_minus_t = _mm256_sub_ps(one, vt);
        let term_a = _mm256_mul_ps(va, one_minus_t);
        let term_b = _mm256_mul_ps(vb, vt);
        let result = _mm256_add_ps(term_a, term_b);

        _mm256_storeu_ps(output.as_mut_ptr().add(offset), result);
    }

    // Handle remainder
    let base = chunks * 8;
    for i in 0..remainder {
        let idx = base + i;
        output[idx] = a[idx] + (b[idx] - a[idx]) * t[idx];
    }
}

/// Linear interpolation using NEON (for multiple samples).
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub fn lerp_neon(a: &[f32], b: &[f32], t: &[f32], output: &mut [f32]) {
    use std::arch::aarch64::*;

    let len = a.len().min(b.len()).min(t.len()).min(output.len());
    let chunks = len / 4;
    let remainder = len % 4;

    unsafe {
        let one = vdupq_n_f32(1.0);

        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));
            let vt = vld1q_f32(t.as_ptr().add(offset));

            // result = a * (1 - t) + b * t
            let one_minus_t = vsubq_f32(one, vt);
            let term_a = vmulq_f32(va, one_minus_t);
            let result = vfmaq_f32(term_a, vb, vt);

            vst1q_f32(output.as_mut_ptr().add(offset), result);
        }

        // Handle remainder
        let base = chunks * 4;
        for i in 0..remainder {
            let idx = base + i;
            output[idx] = a[idx] + (b[idx] - a[idx]) * t[idx];
        }
    }
}

/// Detect available SIMD features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimdFeatures {
    pub avx2: bool,
    pub neon: bool,
}

impl SimdFeatures {
    /// Detect available SIMD features on the current platform.
    pub fn detect() -> Self {
        Self {
            #[cfg(target_arch = "x86_64")]
            avx2: is_x86_feature_detected!("avx2"),
            #[cfg(not(target_arch = "x86_64"))]
            avx2: false,

            #[cfg(target_arch = "aarch64")]
            neon: true, // NEON is always available on aarch64
            #[cfg(not(target_arch = "aarch64"))]
            neon: false,
        }
    }

    /// Check if any SIMD acceleration is available.
    pub fn has_simd(&self) -> bool {
        self.avx2 || self.neon
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_filter() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let coeffs = vec![0.2f32, 0.3, 0.5];

        let result = apply_filter_scalar(&input, 0, &coeffs, 3);

        // Expected: 1.0 * 0.2 + 2.0 * 0.3 + 3.0 * 0.5 = 0.2 + 0.6 + 1.5 = 2.3
        assert!((result - 2.3).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_filter_offset() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let coeffs = vec![0.5f32, 0.5];

        let result = apply_filter_scalar(&input, 2, &coeffs, 2);

        // Expected: 3.0 * 0.5 + 4.0 * 0.5 = 3.5
        assert!((result - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_simd_features() {
        let features = SimdFeatures::detect();

        // At least one of these should be meaningful based on architecture
        #[cfg(target_arch = "x86_64")]
        {
            // AVX2 detection should work
            let _ = features.avx2;
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON should always be true on aarch64
            assert!(features.neon);
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn test_avx2_filter() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let input: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let coeffs: Vec<f32> = vec![1.0; 16];

        let scalar_result = apply_filter_scalar(&input, 0, &coeffs, 16);
        let simd_result = apply_filter_avx2(&input, 0, &coeffs, 16);

        assert!((scalar_result - simd_result).abs() < 1e-4);
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[test]
    fn test_neon_filter() {
        let input: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let coeffs: Vec<f32> = vec![1.0; 16];

        let scalar_result = apply_filter_scalar(&input, 0, &coeffs, 16);
        let simd_result = apply_filter_neon(&input, 0, &coeffs, 16);

        assert!((scalar_result - simd_result).abs() < 1e-4);
    }
}
