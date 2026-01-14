//! Quantization for CineForm wavelet coefficients

use crate::types::Quality;
use crate::wavelet::{Subband, WaveletDecomposition};

/// Quantization table for wavelet subbands
#[derive(Debug, Clone)]
pub struct QuantTable {
    /// Quantization steps for each subband at each level
    /// Index: level * 4 + subband_type (LL=0, LH=1, HL=2, HH=3)
    steps: Vec<u16>,
    /// Number of levels
    levels: usize,
}

impl QuantTable {
    /// Create quantization table for given quality
    pub fn for_quality(quality: Quality, levels: usize) -> Self {
        let base_divisor = quality.quant_divisor() as u16;

        // Each level gets progressively coarser quantization
        // Detail subbands (LH, HL, HH) can be quantized more aggressively
        let mut steps = Vec::with_capacity(levels * 4 + 1);

        for level in 0..levels {
            let level_scale = 1u16 << level; // Double for each level

            // LH - horizontal detail
            steps.push((base_divisor * level_scale).max(1));
            // HL - vertical detail
            steps.push((base_divisor * level_scale).max(1));
            // HH - diagonal detail (can be most aggressive)
            steps.push((base_divisor * level_scale * 3 / 2).max(1));
        }

        // LL (final approximation) - preserve most accuracy
        steps.push((base_divisor / 2).max(1));

        QuantTable { steps, levels }
    }

    /// Get quantization step for a subband
    pub fn get_step(&self, level: usize, subband: Subband) -> u16 {
        match subband {
            Subband::LL => self.steps[self.levels * 3],
            Subband::LH => self.steps[level * 3],
            Subband::HL => self.steps[level * 3 + 1],
            Subband::HH => self.steps[level * 3 + 2],
        }
    }

    /// Get all steps
    #[allow(dead_code)]
    pub fn steps(&self) -> &[u16] {
        &self.steps
    }
}

/// Quantize wavelet decomposition
pub fn quantize_decomposition(decomp: &mut WaveletDecomposition, quant: &QuantTable) {
    let levels = decomp.levels;

    // Quantize detail subbands
    for level in 0..levels {
        // LH
        let step = quant.get_step(level, Subband::LH);
        quantize_subband(&mut decomp.subbands[level * 3], step);

        // HL
        let step = quant.get_step(level, Subband::HL);
        quantize_subband(&mut decomp.subbands[level * 3 + 1], step);

        // HH
        let step = quant.get_step(level, Subband::HH);
        quantize_subband(&mut decomp.subbands[level * 3 + 2], step);
    }

    // Quantize final LL
    let step = quant.get_step(0, Subband::LL);
    quantize_subband(&mut decomp.subbands[levels * 3], step);
}

/// Dequantize wavelet decomposition
pub fn dequantize_decomposition(decomp: &mut WaveletDecomposition, quant: &QuantTable) {
    let levels = decomp.levels;

    // Dequantize detail subbands
    for level in 0..levels {
        // LH
        let step = quant.get_step(level, Subband::LH);
        dequantize_subband(&mut decomp.subbands[level * 3], step);

        // HL
        let step = quant.get_step(level, Subband::HL);
        dequantize_subband(&mut decomp.subbands[level * 3 + 1], step);

        // HH
        let step = quant.get_step(level, Subband::HH);
        dequantize_subband(&mut decomp.subbands[level * 3 + 2], step);
    }

    // Dequantize final LL
    let step = quant.get_step(0, Subband::LL);
    dequantize_subband(&mut decomp.subbands[levels * 3], step);
}

/// Quantize a single subband
fn quantize_subband(data: &mut [i16], step: u16) {
    if step <= 1 {
        return; // No quantization needed
    }

    let step = step as i32;
    let half = step / 2;

    for val in data.iter_mut() {
        let v = *val as i32;
        *val = if v >= 0 {
            ((v + half) / step) as i16
        } else {
            ((v - half) / step) as i16
        };
    }
}

/// Dequantize a single subband
fn dequantize_subband(data: &mut [i16], step: u16) {
    if step <= 1 {
        return;
    }

    let step = step as i32;

    for val in data.iter_mut() {
        *val = ((*val as i32) * step).clamp(-32768, 32767) as i16;
    }
}

/// Dead zone quantization (zeros out small values more aggressively)
#[allow(dead_code)]
pub fn deadzone_quantize(data: &mut [i16], step: u16, deadzone: u16) {
    let step = step as i32;
    let deadzone = deadzone as i32;

    for val in data.iter_mut() {
        let v = *val as i32;
        let abs_v = v.abs();

        if abs_v <= deadzone {
            *val = 0;
        } else {
            let sign = if v >= 0 { 1 } else { -1 };
            *val = (sign * ((abs_v - deadzone) / step)) as i16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_table_creation() {
        let quant = QuantTable::for_quality(Quality::High, 3);

        // Should have 3*3 + 1 = 10 steps
        assert_eq!(quant.steps.len(), 10);

        // LL should have smallest step (highest quality)
        let ll_step = quant.get_step(0, Subband::LL);
        let hh_step = quant.get_step(0, Subband::HH);
        assert!(ll_step <= hh_step);
    }

    #[test]
    fn test_quantize_dequantize() {
        let original: Vec<i16> = vec![100, -50, 25, -12, 6, -3, 0, 200];
        let mut data = original.clone();

        let step = 8u16;
        quantize_subband(&mut data, step);

        // Values should be smaller
        for (orig, quant) in original.iter().zip(data.iter()) {
            assert!(quant.abs() <= orig.abs());
        }

        dequantize_subband(&mut data, step);

        // Should be close to original (within step size)
        for (orig, deq) in original.iter().zip(data.iter()) {
            assert!((orig - deq).abs() <= step as i16);
        }
    }

    #[test]
    fn test_deadzone_quantize() {
        let mut data = vec![5i16, -5, 10, -10, 20, -20, 3, -3];
        deadzone_quantize(&mut data, 8, 6);

        // Values within deadzone should be zero
        assert_eq!(data[0], 0); // 5 is within deadzone of 6
        assert_eq!(data[1], 0); // -5 is within deadzone
        assert_eq!(data[6], 0); // 3 is within deadzone
        assert_eq!(data[7], 0); // -3 is within deadzone
    }

    #[test]
    fn test_quality_affects_quantization() {
        let low = QuantTable::for_quality(Quality::Low, 3);
        let high = QuantTable::for_quality(Quality::FilmScan2, 3);

        // Low quality should have larger steps
        assert!(low.get_step(0, Subband::LH) > high.get_step(0, Subband::LH));
    }
}
