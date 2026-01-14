//! SIMD optimizations for codec hot paths.
//!
//! This module provides SIMD-optimized implementations for performance-critical
//! codec operations with runtime CPU feature detection.
//!
//! # Implementation Status
//!
//! Not all operations have full SIMD implementations. The table below shows
//! which operations are fully accelerated vs. falling back to scalar code:
//!
//! ## Video Operations
//!
//! | Operation | AVX2 (x86_64) | NEON (aarch64) | Notes |
//! |-----------|---------------|----------------|-------|
//! | `sad16x16` | ✅ Full | ✅ Full | Sum of Absolute Differences |
//! | `quantize` | ✅ Full | ✅ Full | Transform coefficient quantization |
//! | `mc_bilinear` | ✅ Full | ✅ Full | Motion compensation with bilinear interpolation |
//! | `idct4x4` | ⚠️ Scalar fallback | ⚠️ Scalar fallback | Complex matrix transpose required |
//! | `idct8x8` | ⚠️ Scalar fallback | ⚠️ Scalar fallback | Chen DCT algorithm not yet vectorized |
//! | `hadamard4x4` | ⚠️ Scalar fallback | ⚠️ Scalar fallback | Butterfly operations need complex shuffles |
//! | `deblock_luma_h` | ⚠️ Scalar fallback | ⚠️ Scalar fallback | Highly data-dependent branching |
//!
//! ## Audio Operations
//!
//! | Operation | AVX2 (x86_64) | NEON (aarch64) | Notes |
//! |-----------|---------------|----------------|-------|
//! | `apply_gain` | ✅ Full | ✅ Full | Simple multiply |
//! | `mix_samples` | ✅ Full | ✅ Full | FMA (fused multiply-add) |
//! | `interleave_stereo` | ✅ Full | ✅ Full | Channel interleaving |
//! | `deinterleave_stereo` | ✅ Full | ✅ Full | Channel deinterleaving |
//! | `fir_filter` | ✅ Full | ✅ Full | FIR filter dot product |
//! | `calculate_rms` | ✅ Full | ✅ Full | RMS level calculation |
//! | `find_peak` | ✅ Full | ✅ Full | Peak detection |
//! | `hard_clip` | ✅ Full | ✅ Full | Hard limiting/clipping |
//! | `overlap_add` | ✅ Full | ✅ Full | MDCT windowed overlap-add |
//! | `mdct_forward` | ✅ Partial | ✅ Partial | Main loop vectorized, remainder scalar |
//! | `biquad_process` | ❌ Scalar only | ❌ Scalar only | State dependencies prevent vectorization |
//! | `soft_clip` | ❌ Scalar only | ❌ Scalar only | tanh approximation not vectorized |
//! | `remove_dc_offset` | ❌ Scalar only | ❌ Scalar only | State dependencies |
//!
//! # Performance Notes
//!
//! - Operations marked "Scalar fallback" dispatch to SIMD code but immediately
//!   call the scalar implementation internally. This is intentional for
//!   correctness while full SIMD implementations are developed.
//! - The SIMD implementations provide significant speedups (2-8x) for operations
//!   marked "Full" when processing large buffers.
//! - Minimum buffer sizes for SIMD dispatch: AVX2 requires 8+ elements for f32,
//!   16+ elements for i16/u8. NEON requires 4+ elements for f32, 8+ for i16/u8.
//!
//! # Future Work
//!
//! The following operations are candidates for full SIMD implementation:
//! - `idct4x4`/`idct8x8`: Implement matrix transpose using shuffle instructions
//! - `hadamard4x4`: Implement butterfly operations with vperm/vshuf
//! - `biquad_process`: Consider parallel multi-channel processing

// Allow common patterns in SIMD/signal processing code
#![allow(
    dead_code,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::unnecessary_cast,
    clippy::cast_abs_to_unsigned,
    clippy::manual_memcpy
)]

mod detect;
mod scalar;

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(target_arch = "aarch64")]
mod aarch64;

pub use detect::{SimdCapabilities, detect_simd};

/// SIMD-optimized operations interface.
pub struct SimdOps {
    caps: SimdCapabilities,
}

impl SimdOps {
    /// Create a new SIMD operations instance with runtime detection.
    pub fn new() -> Self {
        Self {
            caps: detect_simd(),
        }
    }

    /// Get detected SIMD capabilities.
    pub fn capabilities(&self) -> &SimdCapabilities {
        &self.caps
    }

    /// Perform 4x4 IDCT (Inverse Discrete Cosine Transform).
    #[inline]
    pub fn idct4x4(&self, input: &[i16; 16], output: &mut [i16; 16]) {
        #[cfg(target_arch = "x86_64")]
        if self.caps.avx2 {
            unsafe { x86_64::idct4x4_avx2(input, output); }
            return;
        }

        #[cfg(target_arch = "aarch64")]
        if self.caps.neon {
            unsafe { aarch64::idct4x4_neon(input, output); }
            return;
        }

        scalar::idct4x4_scalar(input, output);
    }

    /// Perform 8x8 IDCT.
    #[inline]
    pub fn idct8x8(&self, input: &[i16; 64], output: &mut [i16; 64]) {
        #[cfg(target_arch = "x86_64")]
        if self.caps.avx2 {
            unsafe { x86_64::idct8x8_avx2(input, output); }
            return;
        }

        #[cfg(target_arch = "aarch64")]
        if self.caps.neon {
            unsafe { aarch64::idct8x8_neon(input, output); }
            return;
        }

        scalar::idct8x8_scalar(input, output);
    }

    /// Bilinear interpolation for motion compensation (horizontal + vertical).
    #[inline]
    pub fn mc_bilinear(
        &self,
        dst: &mut [u8],
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        width: usize,
        height: usize,
        dx: i32,
        dy: i32,
    ) {
        #[cfg(target_arch = "x86_64")]
        if self.caps.avx2 && width >= 16 {
            unsafe {
                x86_64::mc_bilinear_avx2(dst, dst_stride, src, src_stride, width, height, dx, dy);
            }
            return;
        }

        #[cfg(target_arch = "aarch64")]
        if self.caps.neon && width >= 8 {
            unsafe {
                aarch64::mc_bilinear_neon(dst, dst_stride, src, src_stride, width, height, dx, dy);
            }
            return;
        }

        scalar::mc_bilinear_scalar(dst, dst_stride, src, src_stride, width, height, dx, dy);
    }

    /// Hadamard transform for SATD calculation.
    #[inline]
    pub fn hadamard4x4(&self, input: &[i16; 16], output: &mut [i16; 16]) {
        #[cfg(target_arch = "x86_64")]
        if self.caps.avx2 {
            unsafe { x86_64::hadamard4x4_avx2(input, output); }
            return;
        }

        #[cfg(target_arch = "aarch64")]
        if self.caps.neon {
            unsafe { aarch64::hadamard4x4_neon(input, output); }
            return;
        }

        scalar::hadamard4x4_scalar(input, output);
    }

    /// Calculate SAD (Sum of Absolute Differences) for motion estimation.
    #[inline]
    pub fn sad16x16(&self, src1: &[u8], stride1: usize, src2: &[u8], stride2: usize) -> u32 {
        #[cfg(target_arch = "x86_64")]
        if self.caps.avx2 {
            unsafe {
                return x86_64::sad16x16_avx2(src1, stride1, src2, stride2);
            }
        }

        #[cfg(target_arch = "aarch64")]
        if self.caps.neon {
            unsafe {
                return aarch64::sad16x16_neon(src1, stride1, src2, stride2);
            }
        }

        scalar::sad16x16_scalar(src1, stride1, src2, stride2)
    }

    /// Deblocking filter edge detection.
    #[inline]
    pub fn deblock_luma_h(&self, pixels: &mut [u8], stride: usize, alpha: i32, beta: i32, tc: &[i32; 4]) {
        #[cfg(target_arch = "x86_64")]
        if self.caps.avx2 {
            unsafe { x86_64::deblock_luma_h_avx2(pixels, stride, alpha, beta, tc); }
            return;
        }

        #[cfg(target_arch = "aarch64")]
        if self.caps.neon {
            unsafe { aarch64::deblock_luma_h_neon(pixels, stride, alpha, beta, tc); }
            return;
        }

        scalar::deblock_luma_h_scalar(pixels, stride, alpha, beta, tc);
    }

    /// MDCT for AAC audio encoding.
    #[inline]
    pub fn mdct_forward(&self, input: &[f32], output: &mut [f32], twiddles: &[(f32, f32)]) {
        #[cfg(target_arch = "x86_64")]
        if self.caps.avx2 {
            unsafe { x86_64::mdct_forward_avx2(input, output, twiddles); }
            return;
        }

        #[cfg(target_arch = "aarch64")]
        if self.caps.neon {
            unsafe { aarch64::mdct_forward_neon(input, output, twiddles); }
            return;
        }

        scalar::mdct_forward_scalar(input, output, twiddles);
    }

    /// Quantization for transform coefficients.
    #[inline]
    pub fn quantize(&self, coeffs: &[i16], output: &mut [i16], qp: u8, intra: bool) {
        #[cfg(target_arch = "x86_64")]
        if self.caps.avx2 && coeffs.len() >= 16 {
            unsafe { x86_64::quantize_avx2(coeffs, output, qp, intra); }
            return;
        }

        #[cfg(target_arch = "aarch64")]
        if self.caps.neon && coeffs.len() >= 8 {
            unsafe { aarch64::quantize_neon(coeffs, output, qp, intra); }
            return;
        }

        scalar::quantize_scalar(coeffs, output, qp, intra);
    }

    // ========================================================================
    // Audio processing SIMD operations
    // ========================================================================

    /// Apply gain to audio samples.
    #[inline]
    pub fn apply_gain(&self, samples: &mut [f32], gain: f32) {
        #[cfg(target_arch = "x86_64")]
        if self.caps.avx2 && samples.len() >= 8 {
            unsafe { x86_64::apply_gain_avx2(samples, gain); }
            return;
        }

        #[cfg(target_arch = "aarch64")]
        if self.caps.neon && samples.len() >= 4 {
            unsafe { aarch64::apply_gain_neon(samples, gain); }
            return;
        }

        scalar::apply_gain_scalar(samples, gain);
    }

    /// Mix source samples into destination with gain.
    #[inline]
    pub fn mix_samples(&self, dst: &mut [f32], src: &[f32], gain: f32) {
        #[cfg(target_arch = "x86_64")]
        if self.caps.avx2 && dst.len() >= 8 {
            unsafe { x86_64::mix_samples_avx2(dst, src, gain); }
            return;
        }

        #[cfg(target_arch = "aarch64")]
        if self.caps.neon && dst.len() >= 4 {
            unsafe { aarch64::mix_samples_neon(dst, src, gain); }
            return;
        }

        scalar::mix_samples_scalar(dst, src, gain);
    }

    /// Interleave stereo channels.
    #[inline]
    pub fn interleave_stereo(&self, left: &[f32], right: &[f32], output: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if self.caps.avx2 && left.len() >= 8 {
            unsafe { x86_64::interleave_stereo_avx2(left, right, output); }
            return;
        }

        #[cfg(target_arch = "aarch64")]
        if self.caps.neon && left.len() >= 4 {
            unsafe { aarch64::interleave_stereo_neon(left, right, output); }
            return;
        }

        scalar::interleave_stereo_scalar(left, right, output);
    }

    /// Deinterleave stereo channels.
    #[inline]
    pub fn deinterleave_stereo(&self, input: &[f32], left: &mut [f32], right: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if self.caps.avx2 && input.len() >= 16 {
            unsafe { x86_64::deinterleave_stereo_avx2(input, left, right); }
            return;
        }

        #[cfg(target_arch = "aarch64")]
        if self.caps.neon && input.len() >= 8 {
            unsafe { aarch64::deinterleave_stereo_neon(input, left, right); }
            return;
        }

        scalar::deinterleave_stereo_scalar(input, left, right);
    }

    /// FIR filter dot product (convolution).
    #[inline]
    pub fn fir_filter(&self, input: &[f32], coeffs: &[f32], start_idx: usize) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if self.caps.avx2 && coeffs.len() >= 8 {
            unsafe { return x86_64::fir_filter_avx2(input, coeffs, start_idx); }
        }

        #[cfg(target_arch = "aarch64")]
        if self.caps.neon && coeffs.len() >= 4 {
            unsafe { return aarch64::fir_filter_neon(input, coeffs, start_idx); }
        }

        scalar::fir_filter_scalar(input, coeffs, start_idx)
    }

    /// Calculate RMS of audio samples.
    #[inline]
    pub fn calculate_rms(&self, samples: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if self.caps.avx2 && samples.len() >= 8 {
            unsafe { return x86_64::calculate_rms_avx2(samples); }
        }

        #[cfg(target_arch = "aarch64")]
        if self.caps.neon && samples.len() >= 4 {
            unsafe { return aarch64::calculate_rms_neon(samples); }
        }

        scalar::calculate_rms_scalar(samples)
    }

    /// Find peak absolute value in samples.
    #[inline]
    pub fn find_peak(&self, samples: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if self.caps.avx2 && samples.len() >= 8 {
            unsafe { return x86_64::find_peak_avx2(samples); }
        }

        #[cfg(target_arch = "aarch64")]
        if self.caps.neon && samples.len() >= 4 {
            unsafe { return aarch64::find_peak_neon(samples); }
        }

        scalar::find_peak_scalar(samples)
    }

    /// Hard clip samples to range [min, max].
    #[inline]
    pub fn hard_clip(&self, samples: &mut [f32], min: f32, max: f32) {
        #[cfg(target_arch = "x86_64")]
        if self.caps.avx2 && samples.len() >= 8 {
            unsafe { x86_64::hard_clip_avx2(samples, min, max); }
            return;
        }

        #[cfg(target_arch = "aarch64")]
        if self.caps.neon && samples.len() >= 4 {
            unsafe { aarch64::hard_clip_neon(samples, min, max); }
            return;
        }

        scalar::hard_clip_scalar(samples, min, max);
    }

    /// Windowed overlap-add for MDCT synthesis.
    #[inline]
    pub fn overlap_add(
        &self,
        output: &mut [f32],
        prev_window: &[f32],
        curr_window: &[f32],
        window: &[f32],
    ) {
        #[cfg(target_arch = "x86_64")]
        if self.caps.avx2 && output.len() >= 8 {
            unsafe { x86_64::overlap_add_avx2(output, prev_window, curr_window, window); }
            return;
        }

        #[cfg(target_arch = "aarch64")]
        if self.caps.neon && output.len() >= 4 {
            unsafe { aarch64::overlap_add_neon(output, prev_window, curr_window, window); }
            return;
        }

        scalar::overlap_add_scalar(output, prev_window, curr_window, window);
    }

    /// Process biquad filter.
    #[inline]
    pub fn biquad_process(
        &self,
        input: &[f32],
        output: &mut [f32],
        coeffs: &scalar::BiquadCoeffs,
        state: &mut scalar::BiquadState,
    ) {
        // Biquad is inherently sequential due to state dependencies
        // SIMD can help with parallel channels, but single-channel processing is scalar
        scalar::biquad_process_scalar(input, output, coeffs, state);
    }

    /// Soft clip samples using tanh-based algorithm.
    #[inline]
    pub fn soft_clip(&self, samples: &mut [f32], threshold: f32) {
        // Tanh is expensive to vectorize without approximation
        scalar::soft_clip_scalar(samples, threshold);
    }

    /// Remove DC offset from samples.
    #[inline]
    pub fn remove_dc_offset(&self, samples: &mut [f32], state: &mut f32, alpha: f32) {
        // DC removal has state dependencies
        scalar::remove_dc_offset_scalar(samples, state, alpha);
    }
}

impl Default for SimdOps {
    fn default() -> Self {
        Self::new()
    }
}

// Re-export audio types for convenience
pub use scalar::{BiquadCoeffs, BiquadState};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_detection() {
        let caps = detect_simd();
        println!("SIMD Capabilities: {:?}", caps);
        // Should always have at least scalar fallback
    }

    #[test]
    fn test_idct4x4() {
        let ops = SimdOps::new();
        let input = [100i16; 16];
        let mut output = [0i16; 16];
        ops.idct4x4(&input, &mut output);
        assert!(output.iter().any(|&x| x != 0));
    }

    #[test]
    fn test_sad16x16() {
        let ops = SimdOps::new();
        let src1 = [100u8; 256];
        let src2 = [100u8; 256];
        let sad = ops.sad16x16(&src1, 16, &src2, 16);
        assert_eq!(sad, 0);

        let src3 = [101u8; 256];
        let sad2 = ops.sad16x16(&src1, 16, &src3, 16);
        assert_eq!(sad2, 256);
    }

    #[test]
    fn test_hadamard4x4() {
        let ops = SimdOps::new();
        let input = [1i16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut output = [0i16; 16];
        ops.hadamard4x4(&input, &mut output);
        // DC coefficient should be sum of all inputs (136/2 = 68)
        assert!(output[0] != 0);
    }

    // Audio SIMD tests
    #[test]
    fn test_apply_gain() {
        let ops = SimdOps::new();
        let mut samples: Vec<f32> = (0..32).map(|i| i as f32).collect();
        ops.apply_gain(&mut samples, 0.5);
        assert!((samples[0] - 0.0).abs() < 1e-5);
        assert!((samples[10] - 5.0).abs() < 1e-5);
        assert!((samples[20] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_mix_samples() {
        let ops = SimdOps::new();
        let mut dst: Vec<f32> = vec![1.0; 32];
        let src: Vec<f32> = vec![0.5; 32];
        ops.mix_samples(&mut dst, &src, 2.0);
        for &val in &dst {
            assert!((val - 2.0).abs() < 1e-5); // 1.0 + 0.5 * 2.0
        }
    }

    #[test]
    fn test_stereo_interleave_deinterleave() {
        let ops = SimdOps::new();
        let left: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let right: Vec<f32> = (16..32).map(|i| i as f32).collect();
        let mut interleaved = vec![0.0f32; 32];
        let mut left_out = vec![0.0f32; 16];
        let mut right_out = vec![0.0f32; 16];

        ops.interleave_stereo(&left, &right, &mut interleaved);
        ops.deinterleave_stereo(&interleaved, &mut left_out, &mut right_out);

        for i in 0..16 {
            assert!((left[i] - left_out[i]).abs() < 1e-5);
            assert!((right[i] - right_out[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_fir_filter() {
        let ops = SimdOps::new();
        let input: Vec<f32> = vec![1.0; 32];
        let coeffs: Vec<f32> = vec![0.25; 16];
        let result = ops.fir_filter(&input, &coeffs, 0);
        // 16 * 1.0 * 0.25 = 4.0
        assert!((result - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_calculate_rms() {
        let ops = SimdOps::new();
        let samples: Vec<f32> = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let rms = ops.calculate_rms(&samples);
        assert!((rms - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_find_peak() {
        let ops = SimdOps::new();
        let mut samples: Vec<f32> = vec![0.5; 32];
        samples[15] = -0.9;
        let peak = ops.find_peak(&samples);
        assert!((peak - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_hard_clip() {
        let ops = SimdOps::new();
        let mut samples: Vec<f32> = vec![-1.5, -0.5, 0.5, 1.5, -2.0, 0.0, 0.8, 2.5];
        ops.hard_clip(&mut samples, -1.0, 1.0);
        assert!((samples[0] - (-1.0)).abs() < 1e-5);
        assert!((samples[1] - (-0.5)).abs() < 1e-5);
        assert!((samples[2] - 0.5).abs() < 1e-5);
        assert!((samples[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_biquad_process() {
        let ops = SimdOps::new();
        let coeffs = BiquadCoeffs {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        };
        let mut state = BiquadState::default();
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];
        ops.biquad_process(&input, &mut output, &coeffs, &mut state);
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 2.0).abs() < 1e-5);
    }
}
