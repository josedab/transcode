//! Peak Signal-to-Noise Ratio (PSNR) metric.
//!
//! PSNR measures the ratio between the maximum possible power of a signal
//! and the power of corrupting noise. Higher values indicate better quality.
//!
//! Typical PSNR values:
//! - Excellent: > 40 dB
//! - Good: 30-40 dB
//! - Acceptable: 20-30 dB
//! - Poor: < 20 dB

#![allow(clippy::needless_range_loop)]

use crate::error::{QualityError, Result};
use crate::Frame;
use rayon::prelude::*;

/// PSNR result.
#[derive(Debug, Clone, Copy)]
pub struct PsnrResult {
    /// Overall PSNR (dB).
    pub psnr: f64,
    /// MSE (Mean Squared Error).
    pub mse: f64,
    /// Per-channel PSNR values [Y/R, U/G, V/B] (dB).
    pub per_channel: [f64; 3],
}

impl PsnrResult {
    /// Check if quality is excellent (> 40 dB).
    pub fn is_excellent(&self) -> bool {
        self.psnr > 40.0
    }

    /// Check if quality is good (30-40 dB).
    pub fn is_good(&self) -> bool {
        self.psnr >= 30.0 && self.psnr <= 40.0
    }

    /// Check if quality is acceptable (20-30 dB).
    pub fn is_acceptable(&self) -> bool {
        self.psnr >= 20.0 && self.psnr < 30.0
    }
}

/// PSNR calculator configuration.
#[derive(Debug, Clone, Default)]
pub struct PsnrConfig {
    /// Bit depth of the source (8, 10, or 12).
    pub bit_depth: u8,
}

impl PsnrConfig {
    /// Create config for 8-bit content.
    pub fn bit_depth_8() -> Self {
        Self { bit_depth: 8 }
    }

    /// Create config for 10-bit content.
    pub fn bit_depth_10() -> Self {
        Self { bit_depth: 10 }
    }

    /// Create config for 12-bit content.
    pub fn bit_depth_12() -> Self {
        Self { bit_depth: 12 }
    }

    /// Get maximum pixel value for the bit depth.
    fn max_value(&self) -> f64 {
        let depth = if self.bit_depth == 0 { 8 } else { self.bit_depth };
        (1u32 << depth) as f64 - 1.0
    }
}

/// PSNR calculator.
#[derive(Debug, Clone)]
pub struct Psnr {
    config: PsnrConfig,
}

impl Default for Psnr {
    fn default() -> Self {
        Self::new(PsnrConfig::default())
    }
}

impl Psnr {
    /// Create a new PSNR calculator.
    pub fn new(config: PsnrConfig) -> Self {
        Self { config }
    }

    /// Calculate PSNR between reference and distorted frames.
    pub fn calculate(&self, reference: &Frame, distorted: &Frame) -> Result<PsnrResult> {
        self.validate_frames(reference, distorted)?;

        let max_value = self.config.max_value();
        let channels = reference.channels as usize;
        let pixels_per_channel = (reference.width * reference.height) as usize;

        // Calculate MSE per channel
        let mut channel_mse = [0.0f64; 3];

        for c in 0..channels.min(3) {
            let mse: f64 = (0..pixels_per_channel)
                .into_par_iter()
                .map(|i| {
                    let idx = i * channels + c;
                    let ref_val = reference.data[idx] as f64;
                    let dist_val = distorted.data[idx] as f64;
                    let diff = ref_val - dist_val;
                    diff * diff
                })
                .sum::<f64>()
                / pixels_per_channel as f64;

            channel_mse[c] = mse;
        }

        // Calculate per-channel PSNR
        let mut per_channel = [0.0f64; 3];
        for c in 0..channels.min(3) {
            per_channel[c] = if channel_mse[c] > 0.0 {
                10.0 * (max_value * max_value / channel_mse[c]).log10()
            } else {
                f64::INFINITY
            };
        }

        // Calculate overall MSE (average of channels)
        let total_mse: f64 = channel_mse[..channels.min(3)].iter().sum::<f64>() / channels.min(3) as f64;

        // Calculate overall PSNR
        let psnr = if total_mse > 0.0 {
            10.0 * (max_value * max_value / total_mse).log10()
        } else {
            f64::INFINITY
        };

        Ok(PsnrResult {
            psnr,
            mse: total_mse,
            per_channel,
        })
    }

    /// Calculate weighted PSNR (typically used for YUV: Y has higher weight).
    pub fn calculate_weighted(
        &self,
        reference: &Frame,
        distorted: &Frame,
        weights: [f64; 3],
    ) -> Result<PsnrResult> {
        let result = self.calculate(reference, distorted)?;

        let max_value = self.config.max_value();
        let total_weight: f64 = weights.iter().sum();
        let normalized_weights: Vec<f64> = weights.iter().map(|w| w / total_weight).collect();

        // Calculate weighted MSE
        let weighted_mse: f64 = result
            .per_channel
            .iter()
            .enumerate()
            .map(|(i, &psnr)| {
                if psnr.is_infinite() {
                    0.0
                } else {
                    let mse = max_value * max_value / 10.0f64.powf(psnr / 10.0);
                    mse * normalized_weights[i]
                }
            })
            .sum();

        let weighted_psnr = if weighted_mse > 0.0 {
            10.0 * (max_value * max_value / weighted_mse).log10()
        } else {
            f64::INFINITY
        };

        Ok(PsnrResult {
            psnr: weighted_psnr,
            mse: weighted_mse,
            per_channel: result.per_channel,
        })
    }

    /// Validate that frames are compatible.
    fn validate_frames(&self, reference: &Frame, distorted: &Frame) -> Result<()> {
        if reference.width != distorted.width || reference.height != distorted.height {
            return Err(QualityError::DimensionMismatch {
                reference: format!("{}x{}", reference.width, reference.height),
                distorted: format!("{}x{}", distorted.width, distorted.height),
            });
        }

        if reference.channels != distorted.channels {
            return Err(QualityError::DimensionMismatch {
                reference: format!("{} channels", reference.channels),
                distorted: format!("{} channels", distorted.channels),
            });
        }

        if reference.data.len() != distorted.data.len() {
            return Err(QualityError::InvalidFrame(
                "Data length mismatch".to_string(),
            ));
        }

        Ok(())
    }
}

/// Calculate PSNR with default settings.
pub fn psnr(reference: &Frame, distorted: &Frame) -> Result<f64> {
    let calc = Psnr::default();
    Ok(calc.calculate(reference, distorted)?.psnr)
}

/// Calculate PSNR for YUV content with standard 6:1:1 weighting.
pub fn psnr_yuv_weighted(reference: &Frame, distorted: &Frame) -> Result<f64> {
    let calc = Psnr::default();
    Ok(calc
        .calculate_weighted(reference, distorted, [6.0, 1.0, 1.0])?
        .psnr)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(width: u32, height: u32, value: u8) -> Frame {
        let data = vec![value; (width * height * 3) as usize];
        Frame {
            data,
            width,
            height,
            channels: 3,
        }
    }

    #[test]
    fn test_identical_frames() {
        let frame = create_test_frame(64, 64, 128);
        let result = psnr(&frame, &frame).unwrap();
        assert!(result.is_infinite());
    }

    #[test]
    fn test_different_frames() {
        let ref_frame = create_test_frame(64, 64, 128);
        let dist_frame = create_test_frame(64, 64, 138);
        let result = psnr(&ref_frame, &dist_frame).unwrap();
        assert!(result > 0.0 && result < 100.0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let ref_frame = create_test_frame(64, 64, 128);
        let dist_frame = create_test_frame(32, 32, 128);
        let result = psnr(&ref_frame, &dist_frame);
        assert!(matches!(result, Err(QualityError::DimensionMismatch { .. })));
    }
}
