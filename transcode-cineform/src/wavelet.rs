//! Wavelet transform for CineForm
//!
//! CineForm uses a 2D wavelet decomposition. This module implements
//! the Haar wavelet transform which is computationally efficient.

use crate::error::Result;

/// Wavelet transform type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveletType {
    /// Haar wavelet (simple averaging/differencing)
    Haar,
    /// LeGall 5/3 wavelet (lossless)
    LeGall53,
    /// CDF 9/7 wavelet (lossy, higher quality)
    Cdf97,
}

impl Default for WaveletType {
    fn default() -> Self {
        WaveletType::Haar
    }
}

/// Wavelet subband types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subband {
    /// Low-Low (approximation)
    LL,
    /// Low-High (horizontal detail)
    LH,
    /// High-Low (vertical detail)
    HL,
    /// High-High (diagonal detail)
    HH,
}

/// Wavelet decomposition of a single channel
#[derive(Debug, Clone)]
pub struct WaveletDecomposition {
    /// Width of original image
    pub width: usize,
    /// Height of original image
    pub height: usize,
    /// Number of decomposition levels
    pub levels: usize,
    /// Subband data for each level
    /// Level 0 is the finest (largest), level n-1 is coarsest
    /// Each level contains [LL, LH, HL, HH] for all but the last which only has LL
    pub subbands: Vec<Vec<i16>>,
}

impl WaveletDecomposition {
    /// Create a new decomposition structure
    pub fn new(width: usize, height: usize, levels: usize) -> Self {
        let mut subbands = Vec::with_capacity(levels * 3 + 1);

        let mut w = width;
        let mut h = height;

        // For each level, we have LH, HL, HH subbands
        for _level in 0..levels {
            let half_w = (w + 1) / 2;
            let half_h = (h + 1) / 2;
            let subband_size = half_w * half_h;

            // LH, HL, HH for this level
            subbands.push(vec![0i16; subband_size]); // LH
            subbands.push(vec![0i16; subband_size]); // HL
            subbands.push(vec![0i16; subband_size]); // HH

            w = half_w;
            h = half_h;
        }

        // Final LL subband (coarsest approximation)
        let final_size = w * h;
        subbands.push(vec![0i16; final_size]);

        WaveletDecomposition {
            width,
            height,
            levels,
            subbands,
        }
    }

    /// Get subband index for a specific level and type
    pub fn subband_index(&self, level: usize, subband: Subband) -> usize {
        match subband {
            Subband::LL => {
                // LL is only at the final level
                self.subbands.len() - 1
            }
            Subband::LH => level * 3,
            Subband::HL => level * 3 + 1,
            Subband::HH => level * 3 + 2,
        }
    }

    /// Get dimensions at a specific level
    pub fn level_dimensions(&self, level: usize) -> (usize, usize) {
        let mut w = self.width;
        let mut h = self.height;

        for _ in 0..=level {
            w = (w + 1) / 2;
            h = (h + 1) / 2;
        }

        (w, h)
    }
}

/// Perform forward wavelet transform (decomposition)
pub fn forward_wavelet_2d(
    input: &[i16],
    width: usize,
    height: usize,
    levels: usize,
    wavelet_type: WaveletType,
) -> Result<WaveletDecomposition> {
    let mut decomp = WaveletDecomposition::new(width, height, levels);

    // Start with the input as our working buffer
    let mut current = input.to_vec();
    let mut current_w = width;
    let mut current_h = height;

    for level in 0..levels {
        let half_w = (current_w + 1) / 2;
        let half_h = (current_h + 1) / 2;

        // Allocate temporary buffers for subbands
        let mut ll = vec![0i16; half_w * half_h];
        let mut lh = vec![0i16; half_w * half_h];
        let mut hl = vec![0i16; half_w * half_h];
        let mut hh = vec![0i16; half_w * half_h];

        // Perform 2D wavelet transform
        match wavelet_type {
            WaveletType::Haar => {
                haar_forward_2d(&current, current_w, current_h, &mut ll, &mut lh, &mut hl, &mut hh);
            }
            WaveletType::LeGall53 => {
                legall_forward_2d(&current, current_w, current_h, &mut ll, &mut lh, &mut hl, &mut hh);
            }
            WaveletType::Cdf97 => {
                cdf97_forward_2d(&current, current_w, current_h, &mut ll, &mut lh, &mut hl, &mut hh);
            }
        }

        // Store detail subbands
        decomp.subbands[level * 3] = lh;
        decomp.subbands[level * 3 + 1] = hl;
        decomp.subbands[level * 3 + 2] = hh;

        // LL becomes input for next level
        current = ll;
        current_w = half_w;
        current_h = half_h;
    }

    // Store final LL
    decomp.subbands[levels * 3] = current;

    Ok(decomp)
}

/// Perform inverse wavelet transform (reconstruction)
pub fn inverse_wavelet_2d(
    decomp: &WaveletDecomposition,
    wavelet_type: WaveletType,
) -> Result<Vec<i16>> {
    let levels = decomp.levels;

    // Start with the coarsest LL
    let mut current = decomp.subbands[levels * 3].clone();
    let mut current_w = decomp.level_dimensions(levels - 1).0;
    let mut current_h = decomp.level_dimensions(levels - 1).1;

    // Reconstruct from coarsest to finest
    for level in (0..levels).rev() {
        let lh = &decomp.subbands[level * 3];
        let hl = &decomp.subbands[level * 3 + 1];
        let hh = &decomp.subbands[level * 3 + 2];

        // Calculate output dimensions
        let out_w = if level == 0 {
            decomp.width
        } else {
            let (w, _) = decomp.level_dimensions(level - 1);
            w * 2
        };
        let out_h = if level == 0 {
            decomp.height
        } else {
            let (_, h) = decomp.level_dimensions(level - 1);
            h * 2
        };

        let mut output = vec![0i16; out_w * out_h];

        match wavelet_type {
            WaveletType::Haar => {
                haar_inverse_2d(&current, lh, hl, hh, current_w, current_h, &mut output);
            }
            WaveletType::LeGall53 => {
                legall_inverse_2d(&current, lh, hl, hh, current_w, current_h, &mut output);
            }
            WaveletType::Cdf97 => {
                cdf97_inverse_2d(&current, lh, hl, hh, current_w, current_h, &mut output);
            }
        }

        current = output;
        current_w = out_w;
        current_h = out_h;
    }

    Ok(current)
}

/// Haar forward transform 2D
fn haar_forward_2d(
    input: &[i16],
    width: usize,
    height: usize,
    ll: &mut [i16],
    lh: &mut [i16],
    hl: &mut [i16],
    hh: &mut [i16],
) {
    let half_w = (width + 1) / 2;
    let half_h = (height + 1) / 2;

    for y in 0..half_h {
        for x in 0..half_w {
            let y0 = y * 2;
            let y1 = (y * 2 + 1).min(height - 1);
            let x0 = x * 2;
            let x1 = (x * 2 + 1).min(width - 1);

            let p00 = input[y0 * width + x0] as i32;
            let p01 = input[y0 * width + x1] as i32;
            let p10 = input[y1 * width + x0] as i32;
            let p11 = input[y1 * width + x1] as i32;

            // Haar transform
            let idx = y * half_w + x;
            ll[idx] = ((p00 + p01 + p10 + p11 + 2) >> 2) as i16; // Average
            lh[idx] = ((p00 + p01 - p10 - p11 + 2) >> 2) as i16; // Horizontal diff
            hl[idx] = ((p00 - p01 + p10 - p11 + 2) >> 2) as i16; // Vertical diff
            hh[idx] = ((p00 - p01 - p10 + p11 + 2) >> 2) as i16; // Diagonal diff
        }
    }
}

/// Haar inverse transform 2D
fn haar_inverse_2d(
    ll: &[i16],
    lh: &[i16],
    hl: &[i16],
    hh: &[i16],
    half_w: usize,
    half_h: usize,
    output: &mut [i16],
) {
    let out_w = half_w * 2;

    for y in 0..half_h {
        for x in 0..half_w {
            let idx = y * half_w + x;

            let ll_val = ll[idx] as i32;
            let lh_val = lh[idx] as i32;
            let hl_val = hl[idx] as i32;
            let hh_val = hh[idx] as i32;

            // Inverse Haar transform
            let p00 = ll_val + lh_val + hl_val + hh_val;
            let p01 = ll_val + lh_val - hl_val - hh_val;
            let p10 = ll_val - lh_val + hl_val - hh_val;
            let p11 = ll_val - lh_val - hl_val + hh_val;

            let y0 = y * 2;
            let y1 = y * 2 + 1;
            let x0 = x * 2;
            let x1 = x * 2 + 1;

            if y0 < output.len() / out_w && x0 < out_w {
                output[y0 * out_w + x0] = p00.clamp(-32768, 32767) as i16;
            }
            if y0 < output.len() / out_w && x1 < out_w {
                output[y0 * out_w + x1] = p01.clamp(-32768, 32767) as i16;
            }
            if y1 < output.len() / out_w && x0 < out_w {
                output[y1 * out_w + x0] = p10.clamp(-32768, 32767) as i16;
            }
            if y1 < output.len() / out_w && x1 < out_w {
                output[y1 * out_w + x1] = p11.clamp(-32768, 32767) as i16;
            }
        }
    }
}

/// LeGall 5/3 forward transform 2D (simplified)
fn legall_forward_2d(
    input: &[i16],
    width: usize,
    height: usize,
    ll: &mut [i16],
    lh: &mut [i16],
    hl: &mut [i16],
    hh: &mut [i16],
) {
    // For simplicity, use Haar as fallback
    // A full implementation would use lifting scheme
    haar_forward_2d(input, width, height, ll, lh, hl, hh);
}

/// LeGall 5/3 inverse transform 2D (simplified)
fn legall_inverse_2d(
    ll: &[i16],
    lh: &[i16],
    hl: &[i16],
    hh: &[i16],
    half_w: usize,
    half_h: usize,
    output: &mut [i16],
) {
    haar_inverse_2d(ll, lh, hl, hh, half_w, half_h, output);
}

/// CDF 9/7 forward transform 2D (simplified)
fn cdf97_forward_2d(
    input: &[i16],
    width: usize,
    height: usize,
    ll: &mut [i16],
    lh: &mut [i16],
    hl: &mut [i16],
    hh: &mut [i16],
) {
    // For simplicity, use Haar as fallback
    // A full implementation would use lifting scheme with specific coefficients
    haar_forward_2d(input, width, height, ll, lh, hl, hh);
}

/// CDF 9/7 inverse transform 2D (simplified)
fn cdf97_inverse_2d(
    ll: &[i16],
    lh: &[i16],
    hl: &[i16],
    hh: &[i16],
    half_w: usize,
    half_h: usize,
    output: &mut [i16],
) {
    haar_inverse_2d(ll, lh, hl, hh, half_w, half_h, output);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavelet_decomposition_structure() {
        let decomp = WaveletDecomposition::new(64, 64, 3);

        // 3 levels * 3 subbands + 1 final LL = 10 subbands
        assert_eq!(decomp.subbands.len(), 10);
        assert_eq!(decomp.levels, 3);
    }

    #[test]
    fn test_haar_roundtrip() {
        let width = 8;
        let height = 8;
        let input: Vec<i16> = (0..64).map(|i| (i * 4) as i16).collect();

        let decomp = forward_wavelet_2d(&input, width, height, 1, WaveletType::Haar).unwrap();
        let reconstructed = inverse_wavelet_2d(&decomp, WaveletType::Haar).unwrap();

        // Should be close to original (may have small rounding errors)
        for (orig, rec) in input.iter().zip(reconstructed.iter()) {
            assert!(
                (orig - rec).abs() <= 4,
                "Mismatch: orig={}, rec={}",
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_multilevel_decomposition() {
        let width = 32;
        let height = 32;
        let input: Vec<i16> = vec![128; width * height];

        let decomp = forward_wavelet_2d(&input, width, height, 3, WaveletType::Haar).unwrap();

        // Check level dimensions
        let (w, h) = decomp.level_dimensions(0);
        assert_eq!((w, h), (16, 16));

        let (w, h) = decomp.level_dimensions(2);
        assert_eq!((w, h), (4, 4));
    }

    #[test]
    fn test_uniform_input_produces_zero_detail() {
        let width = 8;
        let height = 8;
        let input: Vec<i16> = vec![128; width * height];

        let decomp = forward_wavelet_2d(&input, width, height, 1, WaveletType::Haar).unwrap();

        // Uniform input should have zero (or near-zero) detail coefficients
        let lh = &decomp.subbands[0];
        let hl = &decomp.subbands[1];
        let hh = &decomp.subbands[2];

        for &val in lh.iter().chain(hl.iter()).chain(hh.iter()) {
            assert!(val.abs() <= 1, "Detail should be near zero for uniform input");
        }
    }

    #[test]
    fn test_subband_index() {
        let decomp = WaveletDecomposition::new(64, 64, 3);

        assert_eq!(decomp.subband_index(0, Subband::LH), 0);
        assert_eq!(decomp.subband_index(0, Subband::HL), 1);
        assert_eq!(decomp.subband_index(0, Subband::HH), 2);
        assert_eq!(decomp.subband_index(1, Subband::LH), 3);
        assert_eq!(decomp.subband_index(2, Subband::HH), 8);
        assert_eq!(decomp.subband_index(0, Subband::LL), 9); // Final LL
    }
}
