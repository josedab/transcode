//! Structural Similarity Index (SSIM) metric.
//!
//! SSIM measures the perceptual similarity between two images based on:
//! - Luminance comparison
//! - Contrast comparison
//! - Structure comparison
//!
//! SSIM values range from -1 to 1, where 1 indicates perfect similarity.
//! Typical interpretation:
//! - > 0.98: Nearly indistinguishable
//! - 0.95-0.98: High quality
//! - 0.90-0.95: Good quality
//! - < 0.90: Noticeable artifacts

#![allow(clippy::needless_range_loop)]

use crate::error::{QualityError, Result};
use crate::Frame;
use rayon::prelude::*;

/// SSIM result.
#[derive(Debug, Clone, Copy)]
pub struct SsimResult {
    /// Overall SSIM score (-1 to 1).
    pub ssim: f64,
    /// Per-channel SSIM values.
    pub per_channel: [f64; 3],
    /// Luminance component.
    pub luminance: f64,
    /// Contrast component.
    pub contrast: f64,
    /// Structure component.
    pub structure: f64,
}

impl SsimResult {
    /// Check if quality is excellent (> 0.98).
    pub fn is_excellent(&self) -> bool {
        self.ssim > 0.98
    }

    /// Check if quality is high (0.95-0.98).
    pub fn is_high(&self) -> bool {
        self.ssim >= 0.95 && self.ssim <= 0.98
    }

    /// Check if quality is good (0.90-0.95).
    pub fn is_good(&self) -> bool {
        self.ssim >= 0.90 && self.ssim < 0.95
    }

    /// Convert SSIM to dB scale for comparison with PSNR.
    pub fn to_db(&self) -> f64 {
        if self.ssim >= 1.0 {
            f64::INFINITY
        } else {
            -10.0 * (1.0 - self.ssim).log10()
        }
    }
}

/// SSIM calculator configuration.
#[derive(Debug, Clone)]
pub struct SsimConfig {
    /// Window size for local statistics (default: 11).
    pub window_size: usize,
    /// Gaussian sigma for window weighting (default: 1.5).
    pub sigma: f64,
    /// K1 stability constant (default: 0.01).
    pub k1: f64,
    /// K2 stability constant (default: 0.03).
    pub k2: f64,
    /// Bit depth of source (default: 8).
    pub bit_depth: u8,
}

impl Default for SsimConfig {
    fn default() -> Self {
        Self {
            window_size: 11,
            sigma: 1.5,
            k1: 0.01,
            k2: 0.03,
            bit_depth: 8,
        }
    }
}

impl SsimConfig {
    /// Create config with custom window size.
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Create config with custom bit depth.
    pub fn with_bit_depth(mut self, depth: u8) -> Self {
        self.bit_depth = depth;
        self
    }

    /// Get maximum pixel value for the bit depth.
    fn max_value(&self) -> f64 {
        let depth = if self.bit_depth == 0 { 8 } else { self.bit_depth };
        (1u32 << depth) as f64 - 1.0
    }
}

/// SSIM calculator.
#[derive(Debug, Clone)]
pub struct Ssim {
    config: SsimConfig,
    /// Precomputed Gaussian window.
    window: Vec<f64>,
}

impl Default for Ssim {
    fn default() -> Self {
        Self::new(SsimConfig::default())
    }
}

impl Ssim {
    /// Create a new SSIM calculator.
    pub fn new(config: SsimConfig) -> Self {
        let window = Self::create_gaussian_window(config.window_size, config.sigma);
        Self { config, window }
    }

    /// Create Gaussian window for weighted averaging.
    fn create_gaussian_window(size: usize, sigma: f64) -> Vec<f64> {
        let mut window = vec![0.0; size * size];
        let center = size as f64 / 2.0;
        let mut sum = 0.0;

        for y in 0..size {
            for x in 0..size {
                let dx = x as f64 - center;
                let dy = y as f64 - center;
                let value = (-((dx * dx + dy * dy) / (2.0 * sigma * sigma))).exp();
                window[y * size + x] = value;
                sum += value;
            }
        }

        // Normalize
        for v in &mut window {
            *v /= sum;
        }

        window
    }

    /// Calculate SSIM between reference and distorted frames.
    pub fn calculate(&self, reference: &Frame, distorted: &Frame) -> Result<SsimResult> {
        self.validate_frames(reference, distorted)?;

        let channels = reference.channels as usize;
        let mut per_channel = [0.0f64; 3];
        let mut total_luminance = 0.0;
        let mut total_contrast = 0.0;
        let mut total_structure = 0.0;

        for c in 0..channels.min(3) {
            let (ssim, lum, con, str_) = self.calculate_channel(reference, distorted, c)?;
            per_channel[c] = ssim;
            total_luminance += lum;
            total_contrast += con;
            total_structure += str_;
        }

        let num_channels = channels.min(3) as f64;
        let overall_ssim = per_channel[..channels.min(3)].iter().sum::<f64>() / num_channels;

        Ok(SsimResult {
            ssim: overall_ssim,
            per_channel,
            luminance: total_luminance / num_channels,
            contrast: total_contrast / num_channels,
            structure: total_structure / num_channels,
        })
    }

    /// Calculate SSIM for a single channel.
    fn calculate_channel(
        &self,
        reference: &Frame,
        distorted: &Frame,
        channel: usize,
    ) -> Result<(f64, f64, f64, f64)> {
        let width = reference.width as usize;
        let height = reference.height as usize;
        let _channels = reference.channels as usize;
        let win_size = self.config.window_size;
        let _half_win = win_size / 2;

        let max_val = self.config.max_value();
        let c1 = (self.config.k1 * max_val).powi(2);
        let c2 = (self.config.k2 * max_val).powi(2);
        let c3 = c2 / 2.0;

        // Calculate SSIM map
        let map_height = height.saturating_sub(win_size - 1);
        let map_width = width.saturating_sub(win_size - 1);

        if map_height == 0 || map_width == 0 {
            return Err(QualityError::InvalidParameter(
                "Image too small for SSIM window".to_string(),
            ));
        }

        let results: Vec<(f64, f64, f64, f64)> = (0..map_height)
            .into_par_iter()
            .flat_map(|y| {
                (0..map_width)
                    .map(|x| {
                        // Calculate local statistics
                        let (mu_ref, mu_dist, var_ref, var_dist, covar) =
                            self.compute_local_stats(reference, distorted, x, y, channel);

                        // Luminance comparison
                        let l = (2.0 * mu_ref * mu_dist + c1) / (mu_ref * mu_ref + mu_dist * mu_dist + c1);

                        // Contrast comparison
                        let sigma_ref = var_ref.sqrt();
                        let sigma_dist = var_dist.sqrt();
                        let c = (2.0 * sigma_ref * sigma_dist + c2) / (var_ref + var_dist + c2);

                        // Structure comparison
                        let s = (covar + c3) / (sigma_ref * sigma_dist + c3);

                        // SSIM
                        let ssim = l * c * s;

                        (ssim, l, c, s)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let n = results.len() as f64;
        let (ssim, lum, con, str_) = results
            .into_iter()
            .fold((0.0, 0.0, 0.0, 0.0), |acc, val| {
                (acc.0 + val.0, acc.1 + val.1, acc.2 + val.2, acc.3 + val.3)
            });

        Ok((ssim / n, lum / n, con / n, str_ / n))
    }

    /// Compute local statistics for a window.
    fn compute_local_stats(
        &self,
        reference: &Frame,
        distorted: &Frame,
        x: usize,
        y: usize,
        channel: usize,
    ) -> (f64, f64, f64, f64, f64) {
        let width = reference.width as usize;
        let channels = reference.channels as usize;
        let win_size = self.config.window_size;

        let mut mu_ref = 0.0;
        let mut mu_dist = 0.0;
        let mut sigma_ref_sq = 0.0;
        let mut sigma_dist_sq = 0.0;
        let mut sigma_ref_dist = 0.0;

        // Weighted mean
        for wy in 0..win_size {
            for wx in 0..win_size {
                let px = x + wx;
                let py = y + wy;
                let idx = (py * width + px) * channels + channel;
                let weight = self.window[wy * win_size + wx];

                let ref_val = reference.data[idx] as f64;
                let dist_val = distorted.data[idx] as f64;

                mu_ref += weight * ref_val;
                mu_dist += weight * dist_val;
            }
        }

        // Weighted variance and covariance
        for wy in 0..win_size {
            for wx in 0..win_size {
                let px = x + wx;
                let py = y + wy;
                let idx = (py * width + px) * channels + channel;
                let weight = self.window[wy * win_size + wx];

                let ref_val = reference.data[idx] as f64;
                let dist_val = distorted.data[idx] as f64;

                let ref_diff = ref_val - mu_ref;
                let dist_diff = dist_val - mu_dist;

                sigma_ref_sq += weight * ref_diff * ref_diff;
                sigma_dist_sq += weight * dist_diff * dist_diff;
                sigma_ref_dist += weight * ref_diff * dist_diff;
            }
        }

        (mu_ref, mu_dist, sigma_ref_sq, sigma_dist_sq, sigma_ref_dist)
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

        Ok(())
    }
}

/// Multi-Scale SSIM (MS-SSIM) calculator.
///
/// MS-SSIM provides better correlation with human perception by
/// evaluating SSIM at multiple scales.
#[derive(Debug, Clone)]
pub struct MsSsim {
    /// Number of scales.
    scales: usize,
    /// SSIM calculator.
    ssim: Ssim,
    /// Weights for each scale.
    weights: Vec<f64>,
}

impl Default for MsSsim {
    fn default() -> Self {
        Self::new(5, SsimConfig::default())
    }
}

impl MsSsim {
    /// Create a new MS-SSIM calculator.
    pub fn new(scales: usize, config: SsimConfig) -> Self {
        // Default weights from the original MS-SSIM paper
        let weights = match scales {
            5 => vec![0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
            4 => vec![0.0448, 0.2856, 0.3001, 0.3695],
            3 => vec![0.0448, 0.2856, 0.6696],
            2 => vec![0.3304, 0.6696],
            _ => {
                let w = 1.0 / scales as f64;
                vec![w; scales]
            }
        };

        Self {
            scales,
            ssim: Ssim::new(config),
            weights,
        }
    }

    /// Calculate MS-SSIM between reference and distorted frames.
    pub fn calculate(&self, reference: &Frame, distorted: &Frame) -> Result<f64> {
        let mut ref_frame = reference.clone();
        let mut dist_frame = distorted.clone();

        let mut ms_ssim = 1.0;

        for scale in 0..self.scales {
            let result = self.ssim.calculate(&ref_frame, &dist_frame)?;

            if scale < self.scales - 1 {
                // Use contrast and structure for intermediate scales
                ms_ssim *= result.contrast.powf(self.weights[scale]);
                ms_ssim *= result.structure.powf(self.weights[scale]);

                // Downsample by 2x
                ref_frame = Self::downsample(&ref_frame)?;
                dist_frame = Self::downsample(&dist_frame)?;
            } else {
                // Use full SSIM for the last scale
                ms_ssim *= result.ssim.powf(self.weights[scale]);
            }
        }

        Ok(ms_ssim)
    }

    /// Downsample frame by 2x using simple averaging.
    fn downsample(frame: &Frame) -> Result<Frame> {
        let new_width = frame.width / 2;
        let new_height = frame.height / 2;

        if new_width == 0 || new_height == 0 {
            return Err(QualityError::InvalidParameter(
                "Frame too small for further downsampling".to_string(),
            ));
        }

        let channels = frame.channels as usize;
        let old_width = frame.width as usize;
        let mut data = vec![0u8; (new_width * new_height * frame.channels as u32) as usize];

        for y in 0..new_height as usize {
            for x in 0..new_width as usize {
                for c in 0..channels {
                    // Average 2x2 block
                    let idx00 = ((y * 2) * old_width + (x * 2)) * channels + c;
                    let idx01 = ((y * 2) * old_width + (x * 2 + 1)) * channels + c;
                    let idx10 = ((y * 2 + 1) * old_width + (x * 2)) * channels + c;
                    let idx11 = ((y * 2 + 1) * old_width + (x * 2 + 1)) * channels + c;

                    let avg = (frame.data[idx00] as u32
                        + frame.data[idx01] as u32
                        + frame.data[idx10] as u32
                        + frame.data[idx11] as u32)
                        / 4;

                    let out_idx = (y * new_width as usize + x) * channels + c;
                    data[out_idx] = avg as u8;
                }
            }
        }

        Ok(Frame {
            data,
            width: new_width,
            height: new_height,
            channels: frame.channels,
        })
    }
}

/// Calculate SSIM with default settings.
pub fn ssim(reference: &Frame, distorted: &Frame) -> Result<f64> {
    let calc = Ssim::default();
    Ok(calc.calculate(reference, distorted)?.ssim)
}

/// Calculate MS-SSIM with default settings.
pub fn ms_ssim(reference: &Frame, distorted: &Frame) -> Result<f64> {
    let calc = MsSsim::default();
    calc.calculate(reference, distorted)
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
    fn test_identical_frames_ssim() {
        let frame = create_test_frame(64, 64, 128);
        let result = ssim(&frame, &frame).unwrap();
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_different_frames_ssim() {
        let ref_frame = create_test_frame(64, 64, 128);
        let dist_frame = create_test_frame(64, 64, 138);
        let result = ssim(&ref_frame, &dist_frame).unwrap();
        assert!(result < 1.0 && result > 0.0);
    }

    #[test]
    fn test_ms_ssim() {
        // MS-SSIM needs larger images due to 5 scales of downsampling
        let frame = create_test_frame(256, 256, 128);
        let result = ms_ssim(&frame, &frame).unwrap();
        assert!((result - 1.0).abs() < 1e-6);
    }
}
