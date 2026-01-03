//! Scalar (non-SIMD) fallback implementations.
//!
//! These implementations work on all platforms and serve as the baseline
//! for correctness testing of SIMD implementations.

/// Scalar 4x4 IDCT implementation.
pub fn idct4x4_scalar(input: &[i16; 16], output: &mut [i16; 16]) {
    // Horizontal transform
    let mut temp = [0i32; 16];
    for row in 0..4 {
        let i = row * 4;
        let a = input[i] as i32 + input[i + 2] as i32;
        let b = input[i] as i32 - input[i + 2] as i32;
        let c = (input[i + 1] as i32 >> 1) - input[i + 3] as i32;
        let d = input[i + 1] as i32 + (input[i + 3] as i32 >> 1);

        temp[i] = a + d;
        temp[i + 1] = b + c;
        temp[i + 2] = b - c;
        temp[i + 3] = a - d;
    }

    // Vertical transform
    for col in 0..4 {
        let a = temp[col] + temp[col + 8];
        let b = temp[col] - temp[col + 8];
        let c = (temp[col + 4] >> 1) - temp[col + 12];
        let d = temp[col + 4] + (temp[col + 12] >> 1);

        output[col] = ((a + d + 32) >> 6) as i16;
        output[col + 4] = ((b + c + 32) >> 6) as i16;
        output[col + 8] = ((b - c + 32) >> 6) as i16;
        output[col + 12] = ((a - d + 32) >> 6) as i16;
    }
}

/// Scalar 8x8 IDCT implementation.
pub fn idct8x8_scalar(input: &[i16; 64], output: &mut [i16; 64]) {
    // Constants for 8x8 DCT
    const C1: i32 = 1004; // cos(1*pi/16) * 1024
    const C2: i32 = 946;  // cos(2*pi/16) * 1024
    const C3: i32 = 851;  // cos(3*pi/16) * 1024
    const C4: i32 = 724;  // cos(4*pi/16) * 1024 (1/sqrt(2))
    const C5: i32 = 569;  // cos(5*pi/16) * 1024
    const C6: i32 = 392;  // cos(6*pi/16) * 1024
    const C7: i32 = 200;  // cos(7*pi/16) * 1024

    let mut temp = [0i32; 64];

    // Row transform
    for row in 0..8 {
        let i = row * 8;
        let x0 = input[i] as i32;
        let x1 = input[i + 1] as i32;
        let x2 = input[i + 2] as i32;
        let x3 = input[i + 3] as i32;
        let x4 = input[i + 4] as i32;
        let x5 = input[i + 5] as i32;
        let x6 = input[i + 6] as i32;
        let x7 = input[i + 7] as i32;

        // Even part
        let s0 = (x0 + x4) * C4;
        let s1 = (x0 - x4) * C4;
        let s2 = x2 * C6 - x6 * C2;
        let s3 = x2 * C2 + x6 * C6;

        let t0 = s0 + s3;
        let t1 = s1 + s2;
        let t2 = s1 - s2;
        let t3 = s0 - s3;

        // Odd part
        let s4 = x1 * C7 - x7 * C1;
        let s5 = x1 * C1 + x7 * C7;
        let s6 = x3 * C3 - x5 * C5;
        let s7 = x3 * C5 + x5 * C3;

        let t4 = s4 + s6;
        let t5 = s5 + s7;
        let t6 = s5 - s7;
        let t7 = s4 - s6;

        temp[i] = t0 + t5;
        temp[i + 1] = t1 + t6;
        temp[i + 2] = t2 + t7;
        temp[i + 3] = t3 + t4;
        temp[i + 4] = t3 - t4;
        temp[i + 5] = t2 - t7;
        temp[i + 6] = t1 - t6;
        temp[i + 7] = t0 - t5;
    }

    // Column transform
    for col in 0..8 {
        let x0 = temp[col];
        let x1 = temp[col + 8];
        let x2 = temp[col + 16];
        let x3 = temp[col + 24];
        let x4 = temp[col + 32];
        let x5 = temp[col + 40];
        let x6 = temp[col + 48];
        let x7 = temp[col + 56];

        // Even part
        let s0 = (x0 + x4) * C4;
        let s1 = (x0 - x4) * C4;
        let s2 = x2 * C6 - x6 * C2;
        let s3 = x2 * C2 + x6 * C6;

        let t0 = s0 + s3;
        let t1 = s1 + s2;
        let t2 = s1 - s2;
        let t3 = s0 - s3;

        // Odd part
        let s4 = x1 * C7 - x7 * C1;
        let s5 = x1 * C1 + x7 * C7;
        let s6 = x3 * C3 - x5 * C5;
        let s7 = x3 * C5 + x5 * C3;

        let t4 = s4 + s6;
        let t5 = s5 + s7;
        let t6 = s5 - s7;
        let t7 = s4 - s6;

        // Scale and round
        let round = 1 << 19; // For proper rounding
        output[col] = ((t0 + t5 + round) >> 20) as i16;
        output[col + 8] = ((t1 + t6 + round) >> 20) as i16;
        output[col + 16] = ((t2 + t7 + round) >> 20) as i16;
        output[col + 24] = ((t3 + t4 + round) >> 20) as i16;
        output[col + 32] = ((t3 - t4 + round) >> 20) as i16;
        output[col + 40] = ((t2 - t7 + round) >> 20) as i16;
        output[col + 48] = ((t1 - t6 + round) >> 20) as i16;
        output[col + 56] = ((t0 - t5 + round) >> 20) as i16;
    }
}

/// Scalar bilinear motion compensation.
pub fn mc_bilinear_scalar(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    width: usize,
    height: usize,
    dx: i32,
    dy: i32,
) {
    // Quarter-pel interpolation weights
    let wx = (dx & 3) as u32;
    let wy = (dy & 3) as u32;
    let w00 = (4 - wx) * (4 - wy);
    let w01 = wx * (4 - wy);
    let w10 = (4 - wx) * wy;
    let w11 = wx * wy;

    let ox = (dx >> 2) as isize;
    let oy = (dy >> 2) as isize;

    for y in 0..height {
        let src_y = (y as isize + oy) as usize;
        let src_row0 = src_y * src_stride;
        let src_row1 = (src_y + 1).min(height - 1) * src_stride;
        let dst_row = y * dst_stride;

        for x in 0..width {
            let src_x = (x as isize + ox) as usize;
            let x1 = (src_x + 1).min(width - 1);

            let p00 = src[src_row0 + src_x] as u32;
            let p01 = src[src_row0 + x1] as u32;
            let p10 = src[src_row1 + src_x] as u32;
            let p11 = src[src_row1 + x1] as u32;

            let val = (p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11 + 8) >> 4;
            dst[dst_row + x] = val.min(255) as u8;
        }
    }
}

/// Scalar 4x4 Hadamard transform.
pub fn hadamard4x4_scalar(input: &[i16; 16], output: &mut [i16; 16]) {
    // Horizontal transform
    let mut temp = [0i32; 16];
    for row in 0..4 {
        let i = row * 4;
        let a = input[i] as i32 + input[i + 3] as i32;
        let b = input[i + 1] as i32 + input[i + 2] as i32;
        let c = input[i] as i32 - input[i + 3] as i32;
        let d = input[i + 1] as i32 - input[i + 2] as i32;

        temp[i] = a + b;
        temp[i + 1] = c + d;
        temp[i + 2] = a - b;
        temp[i + 3] = c - d;
    }

    // Vertical transform
    for col in 0..4 {
        let a = temp[col] + temp[col + 12];
        let b = temp[col + 4] + temp[col + 8];
        let c = temp[col] - temp[col + 12];
        let d = temp[col + 4] - temp[col + 8];

        output[col] = ((a + b) >> 1) as i16;
        output[col + 4] = ((c + d) >> 1) as i16;
        output[col + 8] = ((a - b) >> 1) as i16;
        output[col + 12] = ((c - d) >> 1) as i16;
    }
}

/// Scalar 16x16 SAD (Sum of Absolute Differences).
pub fn sad16x16_scalar(src1: &[u8], stride1: usize, src2: &[u8], stride2: usize) -> u32 {
    let mut sum = 0u32;
    for y in 0..16 {
        let row1 = y * stride1;
        let row2 = y * stride2;
        for x in 0..16 {
            let diff = (src1[row1 + x] as i32 - src2[row2 + x] as i32).abs() as u32;
            sum += diff;
        }
    }
    sum
}

/// Scalar horizontal luma deblocking filter.
pub fn deblock_luma_h_scalar(pixels: &mut [u8], stride: usize, alpha: i32, beta: i32, tc: &[i32; 4]) {
    for i in 0..4 {
        let idx = i * stride;

        let p2 = pixels[idx] as i32;
        let p1 = pixels[idx + 1] as i32;
        let p0 = pixels[idx + 2] as i32;
        let q0 = pixels[idx + 3] as i32;
        let q1 = pixels[idx + 4] as i32;
        let q2 = pixels[idx + 5] as i32;

        // Check filter condition
        if (p0 - q0).abs() < alpha && (p1 - p0).abs() < beta && (q1 - q0).abs() < beta {
            let tc0 = tc[i];
            if tc0 > 0 {
                let ap = (p2 - p0).abs();
                let aq = (q2 - q0).abs();

                // Strong filter condition
                let delta = ((q0 - p0) * 4 + (p1 - q1) + 4) >> 3;
                let delta = delta.clamp(-tc0, tc0);

                pixels[idx + 2] = (p0 + delta).clamp(0, 255) as u8;
                pixels[idx + 3] = (q0 - delta).clamp(0, 255) as u8;

                // Filter adjacent pixels
                if ap < beta {
                    let delta_p1 = ((p2 + ((p0 + q0 + 1) >> 1) - (p1 << 1)) >> 1).clamp(-tc0, tc0);
                    pixels[idx + 1] = (p1 + delta_p1).clamp(0, 255) as u8;
                }
                if aq < beta {
                    let delta_q1 = ((q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1).clamp(-tc0, tc0);
                    pixels[idx + 4] = (q1 + delta_q1).clamp(0, 255) as u8;
                }
            }
        }
    }
}

/// Scalar MDCT forward transform.
pub fn mdct_forward_scalar(input: &[f32], output: &mut [f32], twiddles: &[(f32, f32)]) {
    let n = output.len();
    let n2 = n / 2;
    let n4 = n / 4;

    // Pre-twiddle and butterfly
    for k in 0..n4 {
        let idx1 = 2 * k;
        let idx2 = n2 - 1 - 2 * k;
        let idx3 = n2 + 2 * k;
        let idx4 = n - 1 - 2 * k;

        if idx4 < input.len() && k < twiddles.len() {
            let (cos_tw, sin_tw) = twiddles[k];

            let a = if idx3 < input.len() { input[idx3] } else { 0.0 }
                  - if idx1 < input.len() { input[idx1] } else { 0.0 };
            let b = if idx2 < input.len() { input[idx2] } else { 0.0 }
                  + if idx4 < input.len() { input[idx4] } else { 0.0 };

            output[2 * k] = a * cos_tw + b * sin_tw;
            if 2 * k + 1 < output.len() {
                output[2 * k + 1] = b * cos_tw - a * sin_tw;
            }
        }
    }

    // Simple DFT for the complex transform (in a real implementation, use FFT)
    // This is a placeholder - real MDCT would use proper FFT
}

/// Quantization scaling factors for different QP values.
const QUANT_SCALE: [i32; 6] = [13107, 11916, 10082, 9362, 8192, 7282];

/// Scalar quantization.
pub fn quantize_scalar(coeffs: &[i16], output: &mut [i16], qp: u8, intra: bool) {
    let qp_mod = (qp % 6) as usize;
    let qp_div = qp / 6;
    let scale = QUANT_SCALE[qp_mod];
    let offset = if intra { 682 } else { 342 };

    for (i, &coeff) in coeffs.iter().enumerate() {
        if i >= output.len() {
            break;
        }
        let sign = if coeff < 0 { -1 } else { 1 };
        let abs = coeff.unsigned_abs() as i32;
        let level = (abs * scale + offset) >> (15 + qp_div);
        output[i] = (level * sign) as i16;
    }
}

// ============================================================================
// Audio-specific SIMD operations
// ============================================================================

/// Biquad filter coefficients.
#[derive(Debug, Clone, Copy)]
pub struct BiquadCoeffs {
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
}

/// Biquad filter state.
#[derive(Debug, Clone, Copy, Default)]
pub struct BiquadState {
    pub z1: f32,
    pub z2: f32,
}

/// Scalar biquad filter processing - processes samples one at a time.
/// Direct Form II Transposed implementation.
#[inline]
pub fn biquad_process_scalar(
    input: &[f32],
    output: &mut [f32],
    coeffs: &BiquadCoeffs,
    state: &mut BiquadState,
) {
    let len = input.len().min(output.len());

    for i in 0..len {
        let x = input[i];
        let y = coeffs.b0 * x + state.z1;
        state.z1 = coeffs.b1 * x - coeffs.a1 * y + state.z2;
        state.z2 = coeffs.b2 * x - coeffs.a2 * y;
        output[i] = y;
    }
}

/// Scalar sample gain application (multiply all samples by a constant).
#[inline]
pub fn apply_gain_scalar(samples: &mut [f32], gain: f32) {
    for sample in samples.iter_mut() {
        *sample *= gain;
    }
}

/// Scalar sample mixing (add src samples to dst with optional gain).
#[inline]
pub fn mix_samples_scalar(dst: &mut [f32], src: &[f32], gain: f32) {
    let len = dst.len().min(src.len());
    for i in 0..len {
        dst[i] += src[i] * gain;
    }
}

/// Scalar stereo interleave: combine L and R channels into interleaved stereo.
pub fn interleave_stereo_scalar(left: &[f32], right: &[f32], output: &mut [f32]) {
    let len = left.len().min(right.len());
    assert!(output.len() >= len * 2);

    for i in 0..len {
        output[i * 2] = left[i];
        output[i * 2 + 1] = right[i];
    }
}

/// Scalar stereo deinterleave: split interleaved stereo into L and R channels.
pub fn deinterleave_stereo_scalar(input: &[f32], left: &mut [f32], right: &mut [f32]) {
    let stereo_samples = input.len() / 2;
    let len = stereo_samples.min(left.len()).min(right.len());

    for i in 0..len {
        left[i] = input[i * 2];
        right[i] = input[i * 2 + 1];
    }
}

/// Scalar FIR filter dot product (convolution with filter coefficients).
#[inline]
pub fn fir_filter_scalar(input: &[f32], coeffs: &[f32], start_idx: usize) -> f32 {
    let mut sum = 0.0f32;
    let len = coeffs.len().min(input.len().saturating_sub(start_idx));

    for i in 0..len {
        sum += input[start_idx + i] * coeffs[i];
    }

    sum
}

/// Scalar windowed overlap-add for MDCT synthesis.
pub fn overlap_add_scalar(
    output: &mut [f32],
    prev_window: &[f32],
    curr_window: &[f32],
    window: &[f32],
) {
    let len = output.len().min(prev_window.len()).min(curr_window.len()).min(window.len());

    for i in 0..len {
        // Previous window descending, current window ascending
        let prev_coeff = window[len - 1 - i];
        let curr_coeff = window[i];
        output[i] = prev_window[i] * prev_coeff + curr_window[i] * curr_coeff;
    }
}

/// Scalar RMS (root mean square) calculation.
#[inline]
pub fn calculate_rms_scalar(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let sum_sq: f32 = samples.iter().map(|&x| x * x).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Scalar peak detection.
#[inline]
pub fn find_peak_scalar(samples: &[f32]) -> f32 {
    samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max)
}

/// Scalar soft clipping (tanh-based).
#[inline]
pub fn soft_clip_scalar(samples: &mut [f32], threshold: f32) {
    let inv_threshold = 1.0 / threshold;
    for sample in samples.iter_mut() {
        if sample.abs() > threshold {
            *sample = threshold * (*sample * inv_threshold).tanh();
        }
    }
}

/// Scalar hard clipping.
#[inline]
pub fn hard_clip_scalar(samples: &mut [f32], min: f32, max: f32) {
    for sample in samples.iter_mut() {
        *sample = sample.clamp(min, max);
    }
}

/// Scalar DC offset removal using a simple high-pass filter.
pub fn remove_dc_offset_scalar(samples: &mut [f32], state: &mut f32, alpha: f32) {
    for sample in samples.iter_mut() {
        let output = *sample - *state;
        *state = *sample - output * alpha;
        *sample = output;
    }
}

/// Scalar sample rate conversion helper: linear interpolation.
#[inline]
pub fn lerp_scalar(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Scalar cubic interpolation (Catmull-Rom spline).
#[inline]
pub fn cubic_interp_scalar(y0: f32, y1: f32, y2: f32, y3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;

    let a0 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
    let a1 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
    let a2 = -0.5 * y0 + 0.5 * y2;
    let a3 = y1;

    a0 * t3 + a1 * t2 + a2 * t + a3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sad16x16() {
        let src1 = [128u8; 256];
        let src2 = [128u8; 256];
        assert_eq!(sad16x16_scalar(&src1, 16, &src2, 16), 0);

        let src3 = [129u8; 256];
        assert_eq!(sad16x16_scalar(&src1, 16, &src3, 16), 256);
    }

    #[test]
    fn test_hadamard4x4() {
        let input = [1i16; 16];
        let mut output = [0i16; 16];
        hadamard4x4_scalar(&input, &mut output);
        // DC coefficient should be 8 (sum/2)
        assert_eq!(output[0], 8);
    }

    #[test]
    fn test_idct4x4() {
        let input = [0i16; 16];
        let mut output = [0i16; 16];
        idct4x4_scalar(&input, &mut output);
        // All zeros should produce all zeros
        assert!(output.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_quantize() {
        let coeffs = [100i16; 16];
        let mut output = [0i16; 16];
        quantize_scalar(&coeffs, &mut output, 26, true);
        // Quantization should produce non-zero output for non-zero input
        assert!(output.iter().any(|&x| x != 0));
    }

    // Audio SIMD tests
    #[test]
    fn test_apply_gain() {
        let mut samples = vec![1.0f32, 2.0, 3.0, 4.0];
        apply_gain_scalar(&mut samples, 0.5);
        assert!((samples[0] - 0.5).abs() < 1e-6);
        assert!((samples[1] - 1.0).abs() < 1e-6);
        assert!((samples[2] - 1.5).abs() < 1e-6);
        assert!((samples[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_mix_samples() {
        let mut dst = vec![1.0f32, 2.0, 3.0, 4.0];
        let src = vec![0.5f32, 0.5, 0.5, 0.5];
        mix_samples_scalar(&mut dst, &src, 2.0);
        assert!((dst[0] - 2.0).abs() < 1e-6); // 1.0 + 0.5 * 2.0
        assert!((dst[1] - 3.0).abs() < 1e-6);
        assert!((dst[2] - 4.0).abs() < 1e-6);
        assert!((dst[3] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_stereo_interleave() {
        let left = vec![1.0f32, 2.0, 3.0, 4.0];
        let right = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut output = vec![0.0f32; 8];
        interleave_stereo_scalar(&left, &right, &mut output);
        assert_eq!(output, vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]);
    }

    #[test]
    fn test_stereo_deinterleave() {
        let input = vec![1.0f32, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0];
        let mut left = vec![0.0f32; 4];
        let mut right = vec![0.0f32; 4];
        deinterleave_stereo_scalar(&input, &mut left, &mut right);
        assert_eq!(left, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(right, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_fir_filter() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let coeffs = vec![0.25f32, 0.5, 0.25];
        let result = fir_filter_scalar(&input, &coeffs, 0);
        // 1.0 * 0.25 + 2.0 * 0.5 + 3.0 * 0.25 = 0.25 + 1.0 + 0.75 = 2.0
        assert!((result - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_rms() {
        let samples = vec![1.0f32, -1.0, 1.0, -1.0];
        let rms = calculate_rms_scalar(&samples);
        assert!((rms - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_find_peak() {
        let samples = vec![0.5f32, -0.8, 0.3, -0.2];
        let peak = find_peak_scalar(&samples);
        assert!((peak - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_hard_clip() {
        let mut samples = vec![-1.5f32, -0.5, 0.5, 1.5];
        hard_clip_scalar(&mut samples, -1.0, 1.0);
        assert!((samples[0] - (-1.0)).abs() < 1e-6);
        assert!((samples[1] - (-0.5)).abs() < 1e-6);
        assert!((samples[2] - 0.5).abs() < 1e-6);
        assert!((samples[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_lerp() {
        assert!((lerp_scalar(0.0, 1.0, 0.5) - 0.5).abs() < 1e-6);
        assert!((lerp_scalar(0.0, 1.0, 0.0) - 0.0).abs() < 1e-6);
        assert!((lerp_scalar(0.0, 1.0, 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_biquad_passthrough() {
        // Unity filter (passthrough)
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
        biquad_process_scalar(&input, &mut output, &coeffs, &mut state);
        assert!((output[0] - 1.0).abs() < 1e-6);
        assert!((output[1] - 2.0).abs() < 1e-6);
        assert!((output[2] - 3.0).abs() < 1e-6);
        assert!((output[3] - 4.0).abs() < 1e-6);
    }
}
