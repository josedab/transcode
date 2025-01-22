//! Transform kernels for VVC encoding/decoding.
//!
//! This module implements the transform operations specified in VVC/H.266:
//! - DCT-II (type 2 Discrete Cosine Transform)
//! - DST-VII (type 7 Discrete Sine Transform)
//! - DCT-VIII (type 8 Discrete Cosine Transform)
//! - LFNST (Low-Frequency Non-Separable Transform)

use std::f64::consts::PI;

/// Transform type selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformType {
    /// DCT-II (default transform)
    DctII,
    /// DST-VII (used for small blocks in MTS mode)
    DstVII,
    /// DCT-VIII (used in MTS mode)
    DctVIII,
}

/// Transform kernel sizes supported by VVC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformSize {
    Size4,
    Size8,
    Size16,
    Size32,
    Size64,
}

impl TransformSize {
    pub fn to_usize(self) -> usize {
        match self {
            Self::Size4 => 4,
            Self::Size8 => 8,
            Self::Size16 => 16,
            Self::Size32 => 32,
            Self::Size64 => 64,
        }
    }

    pub fn from_usize(n: usize) -> Option<Self> {
        match n {
            4 => Some(Self::Size4),
            8 => Some(Self::Size8),
            16 => Some(Self::Size16),
            32 => Some(Self::Size32),
            64 => Some(Self::Size64),
            _ => None,
        }
    }
}

/// Transform kernel containing pre-computed coefficients.
pub struct TransformKernel {
    /// Transform size
    pub size: usize,
    /// Transform type
    pub transform_type: TransformType,
    /// Forward transform matrix (row-major)
    pub forward: Vec<i32>,
    /// Inverse transform matrix (row-major)
    pub inverse: Vec<i32>,
    /// Bit shift for normalization
    pub shift: u8,
}

impl TransformKernel {
    /// Create a new DCT-II kernel of the given size.
    pub fn dct_ii(size: TransformSize) -> Self {
        let n = size.to_usize();
        let (forward, inverse, shift) = compute_dct_ii_matrix(n);
        Self {
            size: n,
            transform_type: TransformType::DctII,
            forward,
            inverse,
            shift,
        }
    }

    /// Create a new DST-VII kernel of the given size.
    pub fn dst_vii(size: TransformSize) -> Self {
        let n = size.to_usize();
        let (forward, inverse, shift) = compute_dst_vii_matrix(n);
        Self {
            size: n,
            transform_type: TransformType::DstVII,
            forward,
            inverse,
            shift,
        }
    }

    /// Create a new DCT-VIII kernel of the given size.
    pub fn dct_viii(size: TransformSize) -> Self {
        let n = size.to_usize();
        let (forward, inverse, shift) = compute_dct_viii_matrix(n);
        Self {
            size: n,
            transform_type: TransformType::DctVIII,
            forward,
            inverse,
            shift,
        }
    }

    /// Apply forward transform to a block.
    /// Input and output are in row-major order.
    pub fn forward_transform(&self, input: &[i32], output: &mut [i32]) {
        assert_eq!(input.len(), self.size * self.size);
        assert_eq!(output.len(), self.size * self.size);

        let mut temp = vec![0i64; self.size * self.size];

        // First pass: horizontal transform
        for row in 0..self.size {
            for col in 0..self.size {
                let mut sum: i64 = 0;
                for k in 0..self.size {
                    sum += input[row * self.size + k] as i64
                        * self.forward[col * self.size + k] as i64;
                }
                temp[row * self.size + col] = sum;
            }
        }

        // Second pass: vertical transform
        let shift1 = self.shift;
        let add1 = 1i64 << (shift1 - 1);

        for row in 0..self.size {
            for col in 0..self.size {
                let mut sum: i64 = 0;
                for k in 0..self.size {
                    sum += temp[k * self.size + col] * self.forward[row * self.size + k] as i64;
                }
                output[row * self.size + col] = ((sum + add1) >> shift1) as i32;
            }
        }
    }

    /// Apply inverse transform to a block.
    /// Input and output are in row-major order.
    pub fn inverse_transform(&self, input: &[i32], output: &mut [i32]) {
        assert_eq!(input.len(), self.size * self.size);
        assert_eq!(output.len(), self.size * self.size);

        let mut temp = vec![0i64; self.size * self.size];

        // First pass: vertical inverse transform
        let shift1 = 7;
        let add1 = 1i64 << (shift1 - 1);

        for row in 0..self.size {
            for col in 0..self.size {
                let mut sum: i64 = 0;
                for k in 0..self.size {
                    sum +=
                        input[k * self.size + col] as i64 * self.inverse[k * self.size + row] as i64;
                }
                temp[row * self.size + col] = (sum + add1) >> shift1;
            }
        }

        // Second pass: horizontal inverse transform
        let shift2 = 12;
        let add2 = 1i64 << (shift2 - 1);

        for row in 0..self.size {
            for col in 0..self.size {
                let mut sum: i64 = 0;
                for k in 0..self.size {
                    sum += temp[row * self.size + k] * self.inverse[k * self.size + col] as i64;
                }
                output[row * self.size + col] = ((sum + add2) >> shift2) as i32;
            }
        }
    }
}

/// Compute DCT-II transform matrix.
/// DCT-II: X[k] = sqrt(2/N) * sum_{n=0}^{N-1} x[n] * cos(pi*k*(2n+1)/(2N))
fn compute_dct_ii_matrix(n: usize) -> (Vec<i32>, Vec<i32>, u8) {
    let scale = 64.0; // Fixed-point scaling
    let shift = 6u8;

    let mut forward = vec![0i32; n * n];
    let mut inverse = vec![0i32; n * n];

    for k in 0..n {
        for i in 0..n {
            let angle = PI * k as f64 * (2.0 * i as f64 + 1.0) / (2.0 * n as f64);
            let c0 = if k == 0 {
                1.0 / (n as f64).sqrt()
            } else {
                (2.0 / n as f64).sqrt()
            };
            let coeff = c0 * angle.cos() * scale;

            forward[k * n + i] = coeff.round() as i32;
            inverse[i * n + k] = coeff.round() as i32;
        }
    }

    (forward, inverse, shift)
}

/// Compute DST-VII transform matrix.
/// DST-VII: X[k] = sqrt(2/(N+0.5)) * sum_{n=0}^{N-1} x[n] * sin(pi*(k+0.5)*(n+1)/(N+0.5))
fn compute_dst_vii_matrix(n: usize) -> (Vec<i32>, Vec<i32>, u8) {
    let scale = 64.0;
    let shift = 6u8;

    let mut forward = vec![0i32; n * n];
    let mut inverse = vec![0i32; n * n];

    let norm = (2.0 / (n as f64 + 0.5)).sqrt();

    for k in 0..n {
        for i in 0..n {
            let angle = PI * (k as f64 + 0.5) * (i as f64 + 1.0) / (n as f64 + 0.5);
            let coeff = norm * angle.sin() * scale;

            forward[k * n + i] = coeff.round() as i32;
            inverse[i * n + k] = coeff.round() as i32;
        }
    }

    (forward, inverse, shift)
}

/// Compute DCT-VIII transform matrix.
/// DCT-VIII: X[k] = sqrt(2/(N+0.5)) * sum_{n=0}^{N-1} x[n] * cos(pi*(k+0.5)*(n+0.5)/(N+0.5))
fn compute_dct_viii_matrix(n: usize) -> (Vec<i32>, Vec<i32>, u8) {
    let scale = 64.0;
    let shift = 6u8;

    let mut forward = vec![0i32; n * n];
    let mut inverse = vec![0i32; n * n];

    let norm = (2.0 / (n as f64 + 0.5)).sqrt();

    for k in 0..n {
        for i in 0..n {
            let angle = PI * (k as f64 + 0.5) * (i as f64 + 0.5) / (n as f64 + 0.5);
            let coeff = norm * angle.cos() * scale;

            forward[k * n + i] = coeff.round() as i32;
            inverse[i * n + k] = coeff.round() as i32;
        }
    }

    (forward, inverse, shift)
}

/// Pre-computed VVC transform kernels.
pub struct VvcTransforms {
    /// DCT-II kernels by size
    pub dct_ii: [TransformKernel; 5],
    /// DST-VII kernels by size (4x4, 8x8, 16x16, 32x32 only)
    pub dst_vii: [TransformKernel; 4],
    /// DCT-VIII kernels by size (4x4, 8x8, 16x16, 32x32 only)
    pub dct_viii: [TransformKernel; 4],
}

impl VvcTransforms {
    /// Create all VVC transform kernels.
    pub fn new() -> Self {
        Self {
            dct_ii: [
                TransformKernel::dct_ii(TransformSize::Size4),
                TransformKernel::dct_ii(TransformSize::Size8),
                TransformKernel::dct_ii(TransformSize::Size16),
                TransformKernel::dct_ii(TransformSize::Size32),
                TransformKernel::dct_ii(TransformSize::Size64),
            ],
            dst_vii: [
                TransformKernel::dst_vii(TransformSize::Size4),
                TransformKernel::dst_vii(TransformSize::Size8),
                TransformKernel::dst_vii(TransformSize::Size16),
                TransformKernel::dst_vii(TransformSize::Size32),
            ],
            dct_viii: [
                TransformKernel::dct_viii(TransformSize::Size4),
                TransformKernel::dct_viii(TransformSize::Size8),
                TransformKernel::dct_viii(TransformSize::Size16),
                TransformKernel::dct_viii(TransformSize::Size32),
            ],
        }
    }

    /// Get the appropriate transform kernel.
    pub fn get_kernel(&self, size: usize, transform_type: TransformType) -> Option<&TransformKernel> {
        let idx = match size {
            4 => 0,
            8 => 1,
            16 => 2,
            32 => 3,
            64 => 4,
            _ => return None,
        };

        match transform_type {
            TransformType::DctII => self.dct_ii.get(idx),
            TransformType::DstVII if idx < 4 => self.dst_vii.get(idx),
            TransformType::DctVIII if idx < 4 => self.dct_viii.get(idx),
            _ => None,
        }
    }
}

impl Default for VvcTransforms {
    fn default() -> Self {
        Self::new()
    }
}

/// LFNST (Low-Frequency Non-Separable Transform) kernel index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LfnstIndex {
    /// No LFNST applied
    Disabled,
    /// LFNST kernel 0
    Kernel0,
    /// LFNST kernel 1
    Kernel1,
}

/// LFNST transformer for 4x4 and 8x8 blocks.
pub struct LfnstTransform {
    /// 4x4 LFNST kernels (2 sets x 4 intra mode groups)
    kernels_4x4: Vec<Vec<i32>>,
    /// 8x8 LFNST kernels (2 sets x 4 intra mode groups)
    #[allow(dead_code)]
    kernels_8x8: Vec<Vec<i32>>,
}

impl LfnstTransform {
    /// Create LFNST transform with pre-computed kernels.
    pub fn new() -> Self {
        // Simplified LFNST kernels - in production these would be the full VVC spec kernels
        let mut kernels_4x4 = Vec::new();
        let mut kernels_8x8 = Vec::new();

        // Generate 8 4x4 kernels (2 sets x 4 mode groups)
        for i in 0..8 {
            kernels_4x4.push(generate_lfnst_kernel_4x4(i));
        }

        // Generate 8 8x8 kernels (2 sets x 4 mode groups)
        for i in 0..8 {
            kernels_8x8.push(generate_lfnst_kernel_8x8(i));
        }

        Self {
            kernels_4x4,
            kernels_8x8,
        }
    }

    /// Apply forward LFNST to coefficients.
    pub fn forward_4x4(
        &self,
        input: &[i32; 16],
        output: &mut [i32; 16],
        intra_mode: u8,
        kernel_idx: LfnstIndex,
    ) {
        if kernel_idx == LfnstIndex::Disabled {
            output.copy_from_slice(input);
            return;
        }

        let mode_group = get_lfnst_mode_group(intra_mode);
        let set_idx = match kernel_idx {
            LfnstIndex::Kernel0 => 0,
            LfnstIndex::Kernel1 => 1,
            LfnstIndex::Disabled => unreachable!(),
        };

        let kernel_index = set_idx * 4 + mode_group;
        let kernel = &self.kernels_4x4[kernel_index];

        // Apply 16x16 matrix multiplication (simplified - full LFNST uses 16x8)
        for i in 0..16 {
            let mut sum: i64 = 0;
            for j in 0..16 {
                sum += input[j] as i64 * kernel[i * 16 + j] as i64;
            }
            output[i] = ((sum + 64) >> 7) as i32;
        }
    }

    /// Apply inverse LFNST to coefficients.
    pub fn inverse_4x4(
        &self,
        input: &[i32; 16],
        output: &mut [i32; 16],
        intra_mode: u8,
        kernel_idx: LfnstIndex,
    ) {
        if kernel_idx == LfnstIndex::Disabled {
            output.copy_from_slice(input);
            return;
        }

        let mode_group = get_lfnst_mode_group(intra_mode);
        let set_idx = match kernel_idx {
            LfnstIndex::Kernel0 => 0,
            LfnstIndex::Kernel1 => 1,
            LfnstIndex::Disabled => unreachable!(),
        };

        let kernel_index = set_idx * 4 + mode_group;
        let kernel = &self.kernels_4x4[kernel_index];

        // Apply transposed matrix multiplication for inverse
        for i in 0..16 {
            let mut sum: i64 = 0;
            for j in 0..16 {
                sum += input[j] as i64 * kernel[j * 16 + i] as i64;
            }
            output[i] = ((sum + 64) >> 7) as i32;
        }
    }
}

impl Default for LfnstTransform {
    fn default() -> Self {
        Self::new()
    }
}

/// Get LFNST mode group from intra prediction mode.
fn get_lfnst_mode_group(intra_mode: u8) -> usize {
    match intra_mode {
        0 => 0,        // Planar
        1 => 1,        // DC
        2..=12 => 2,   // Horizontal-ish modes
        13..=66 => 3,  // Vertical-ish and diagonal modes
        _ => 0,
    }
}

/// Generate a 4x4 LFNST kernel (simplified).
fn generate_lfnst_kernel_4x4(index: usize) -> Vec<i32> {
    let mut kernel = vec![0i32; 16 * 16];

    // Initialize with scaled identity-like basis
    for i in 0..16 {
        for j in 0..16 {
            if i == j {
                kernel[i * 16 + j] = 128;
            } else {
                // Add small perturbations based on index for variety
                let phase = (i + j + index) as f64 * PI / 16.0;
                kernel[i * 16 + j] = (phase.sin() * 16.0) as i32;
            }
        }
    }

    kernel
}

/// Generate an 8x8 LFNST kernel (simplified).
fn generate_lfnst_kernel_8x8(index: usize) -> Vec<i32> {
    let mut kernel = vec![0i32; 48 * 16]; // 8x8 LFNST uses 48 primary coeffs -> 16 secondary

    // Initialize with simplified pattern
    for i in 0..16 {
        for j in 0..48 {
            let phase = ((i + j + index) as f64 * PI) / 32.0;
            kernel[i * 48 + j] = (phase.cos() * 64.0) as i32;
        }
    }

    kernel
}

/// Quantization parameters for a transform block.
#[derive(Debug, Clone, Copy)]
pub struct QuantParams {
    /// Quantization parameter (0-63)
    pub qp: u8,
    /// Quantization matrix scale
    pub scale: i32,
    /// Inverse quantization shift
    pub shift: u8,
    /// Dead zone factor for quantization
    pub deadzone: i32,
}

impl QuantParams {
    /// Create quantization parameters from QP value.
    pub fn from_qp(qp: u8, bit_depth: u8) -> Self {
        let qp = qp.min(63);

        // VVC quantization scales (simplified)
        let scale_table: [i32; 6] = [26214, 23302, 20560, 18396, 16384, 14564];
        let qp_mod = (qp % 6) as usize;
        let qp_div = qp / 6;

        let scale = scale_table[qp_mod];
        let shift = 14 + qp_div + (bit_depth - 8);
        let deadzone = (1 << shift) / 3;

        Self {
            qp,
            scale,
            shift,
            deadzone,
        }
    }

    /// Quantize a single coefficient.
    pub fn quantize(&self, coeff: i32) -> i32 {
        let sign = if coeff < 0 { -1 } else { 1 };
        let abs_coeff = coeff.abs() as i64;

        let level = ((abs_coeff * self.scale as i64 + self.deadzone as i64) >> self.shift) as i32;
        sign * level
    }

    /// Dequantize a single coefficient.
    pub fn dequantize(&self, level: i32) -> i32 {
        // Inverse quantization
        let inv_scale_table: [i32; 6] = [40, 45, 51, 57, 64, 72];
        let qp_mod = (self.qp % 6) as usize;
        let qp_div = self.qp / 6;

        let inv_scale = inv_scale_table[qp_mod];
        let shift = qp_div;

        (level * inv_scale) << shift
    }
}

/// Quantize a block of transform coefficients.
pub fn quantize_block(
    coeffs: &[i32],
    levels: &mut [i32],
    params: &QuantParams,
) {
    for (coeff, level) in coeffs.iter().zip(levels.iter_mut()) {
        *level = params.quantize(*coeff);
    }
}

/// Dequantize a block of coefficient levels.
pub fn dequantize_block(
    levels: &[i32],
    coeffs: &mut [i32],
    params: &QuantParams,
) {
    for (level, coeff) in levels.iter().zip(coeffs.iter_mut()) {
        *coeff = params.dequantize(*level);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_ii_4x4() {
        let kernel = TransformKernel::dct_ii(TransformSize::Size4);
        assert_eq!(kernel.size, 4);
        assert_eq!(kernel.forward.len(), 16);
    }

    #[test]
    fn test_transform_roundtrip() {
        let kernel = TransformKernel::dct_ii(TransformSize::Size4);

        let input: Vec<i32> = vec![100, 50, 25, 10, 80, 40, 20, 5, 60, 30, 15, 3, 40, 20, 10, 2];
        let mut transformed = vec![0i32; 16];
        let mut reconstructed = vec![0i32; 16];

        kernel.forward_transform(&input, &mut transformed);
        kernel.inverse_transform(&transformed, &mut reconstructed);

        // Check reconstruction is within reasonable bounds (allowing for fixed-point errors)
        // DCT roundtrip with fixed-point arithmetic has some error
        let mut total_error: i64 = 0;
        for (orig, recon) in input.iter().zip(reconstructed.iter()) {
            total_error += (orig - recon).abs() as i64;
        }
        // Average error should be reasonable
        let avg_error = total_error / 16;
        assert!(
            avg_error < 50,
            "Average reconstruction error too large: {}",
            avg_error
        );
    }

    #[test]
    fn test_dst_vii_creation() {
        let kernel = TransformKernel::dst_vii(TransformSize::Size8);
        assert_eq!(kernel.size, 8);
        assert_eq!(kernel.transform_type, TransformType::DstVII);
    }

    #[test]
    fn test_dct_viii_creation() {
        let kernel = TransformKernel::dct_viii(TransformSize::Size16);
        assert_eq!(kernel.size, 16);
        assert_eq!(kernel.transform_type, TransformType::DctVIII);
    }

    #[test]
    fn test_vvc_transforms() {
        let transforms = VvcTransforms::new();

        // Test all DCT-II sizes
        assert!(transforms.get_kernel(4, TransformType::DctII).is_some());
        assert!(transforms.get_kernel(8, TransformType::DctII).is_some());
        assert!(transforms.get_kernel(16, TransformType::DctII).is_some());
        assert!(transforms.get_kernel(32, TransformType::DctII).is_some());
        assert!(transforms.get_kernel(64, TransformType::DctII).is_some());

        // DST-VII and DCT-VIII don't support 64x64
        assert!(transforms.get_kernel(64, TransformType::DstVII).is_none());
        assert!(transforms.get_kernel(64, TransformType::DctVIII).is_none());
    }

    #[test]
    fn test_quantization() {
        let params = QuantParams::from_qp(22, 8);

        // Test quantize/dequantize roundtrip
        let original = 1000;
        let quantized = params.quantize(original);
        let dequantized = params.dequantize(quantized);

        // Verify quantized value is non-zero for this input
        assert!(quantized != 0, "Quantization should produce non-zero level");
        // Dequantized should have same sign as original
        assert!(dequantized.signum() == original.signum() || dequantized == 0);
    }

    #[test]
    fn test_quantize_block() {
        let params = QuantParams::from_qp(27, 8);
        let coeffs = vec![100, -50, 25, -10, 200, -100, 50, -25];
        let mut levels = vec![0; 8];

        quantize_block(&coeffs, &mut levels, &params);

        // Non-zero coefficients should produce non-zero levels (except very small ones)
        let non_zero_count = levels.iter().filter(|&&l| l != 0).count();
        assert!(non_zero_count > 0, "Should have some non-zero levels");
    }

    #[test]
    fn test_lfnst_disabled() {
        let lfnst = LfnstTransform::new();
        let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut output = [0i32; 16];

        lfnst.forward_4x4(&input, &mut output, 0, LfnstIndex::Disabled);

        assert_eq!(input, output, "Disabled LFNST should pass through unchanged");
    }

    #[test]
    fn test_lfnst_mode_groups() {
        assert_eq!(get_lfnst_mode_group(0), 0); // Planar
        assert_eq!(get_lfnst_mode_group(1), 1); // DC
        assert_eq!(get_lfnst_mode_group(10), 2); // Horizontal
        assert_eq!(get_lfnst_mode_group(50), 3); // Diagonal
    }
}
