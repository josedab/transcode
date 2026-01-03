//! Video Multi-Method Assessment Fusion (VMAF) metric.
//!
//! VMAF is a perceptual video quality metric developed by Netflix that
//! combines multiple elementary quality metrics using machine learning.
//!
//! VMAF scores range from 0 to 100:
//! - 93+: Excellent (imperceptible difference)
//! - 80-93: Good (noticeable but acceptable)
//! - 60-80: Fair (annoying artifacts)
//! - <60: Poor (very annoying artifacts)
//!
//! This module provides:
//! - A full VMAF implementation (requires `vmaf` feature with libvmaf)
//! - A simplified VMAF approximation using basic metrics

use crate::error::{QualityError, Result};
use crate::psnr::Psnr;
use crate::ssim::Ssim;
use crate::Frame;

/// VMAF result.
#[derive(Debug, Clone, Copy)]
pub struct VmafResult {
    /// VMAF score (0-100).
    pub score: f64,
    /// Visual Information Fidelity (VIF) component.
    pub vif: f64,
    /// Detail Loss Metric (DLM) component.
    pub dlm: f64,
    /// Motion component (temporal).
    pub motion: f64,
    /// SSIM component (if computed).
    pub ssim: Option<f64>,
    /// PSNR component (if computed).
    pub psnr: Option<f64>,
}

impl VmafResult {
    /// Check if quality is excellent (>= 93).
    pub fn is_excellent(&self) -> bool {
        self.score >= 93.0
    }

    /// Check if quality is good (80-93).
    pub fn is_good(&self) -> bool {
        self.score >= 80.0 && self.score < 93.0
    }

    /// Check if quality is fair (60-80).
    pub fn is_fair(&self) -> bool {
        self.score >= 60.0 && self.score < 80.0
    }

    /// Check if quality is poor (< 60).
    pub fn is_poor(&self) -> bool {
        self.score < 60.0
    }

    /// Get quality rating as a string.
    pub fn rating(&self) -> &'static str {
        if self.is_excellent() {
            "Excellent"
        } else if self.is_good() {
            "Good"
        } else if self.is_fair() {
            "Fair"
        } else {
            "Poor"
        }
    }
}

/// VMAF model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VmafModel {
    /// Default VMAF model (1080p optimized).
    #[default]
    Default,
    /// 4K model (optimized for 4K content).
    Uhd4K,
    /// Phone model (optimized for mobile viewing).
    Phone,
    /// NEG model (No Enhancement Gain - penalizes over-sharpening).
    Neg,
}

impl VmafModel {
    /// Get model name for libvmaf.
    pub fn model_name(&self) -> &'static str {
        match self {
            Self::Default => "vmaf_v0.6.1",
            Self::Uhd4K => "vmaf_4k_v0.6.1",
            Self::Phone => "vmaf_phone_v0.6.1",
            Self::Neg => "vmaf_neg_v0.6.1",
        }
    }
}

/// VMAF calculator configuration.
#[derive(Debug, Clone)]
pub struct VmafConfig {
    /// VMAF model to use.
    pub model: VmafModel,
    /// Enable pooling methods.
    pub enable_pooling: bool,
    /// Number of threads (0 = auto).
    pub threads: usize,
    /// Also compute PSNR.
    pub compute_psnr: bool,
    /// Also compute SSIM.
    pub compute_ssim: bool,
    /// Subsample factor for speed (1 = no subsampling).
    pub subsample: u32,
}

impl Default for VmafConfig {
    fn default() -> Self {
        Self {
            model: VmafModel::Default,
            enable_pooling: true,
            threads: 0,
            compute_psnr: true,
            compute_ssim: true,
            subsample: 1,
        }
    }
}

impl VmafConfig {
    /// Set VMAF model.
    pub fn with_model(mut self, model: VmafModel) -> Self {
        self.model = model;
        self
    }

    /// Enable/disable additional metrics.
    pub fn with_extra_metrics(mut self, psnr: bool, ssim: bool) -> Self {
        self.compute_psnr = psnr;
        self.compute_ssim = ssim;
        self
    }

    /// Set subsample factor for faster computation.
    pub fn with_subsample(mut self, factor: u32) -> Self {
        self.subsample = factor.max(1);
        self
    }
}

/// VMAF calculator.
///
/// This implementation provides a simplified VMAF approximation using
/// VIF, DLM, and motion metrics. For accurate VMAF scores, enable the
/// `vmaf` feature to use Netflix's libvmaf.
#[derive(Debug)]
pub struct Vmaf {
    config: VmafConfig,
    psnr_calc: Psnr,
    ssim_calc: Ssim,
}

impl Default for Vmaf {
    fn default() -> Self {
        Self::new(VmafConfig::default())
    }
}

impl Vmaf {
    /// Create a new VMAF calculator.
    pub fn new(config: VmafConfig) -> Self {
        Self {
            config,
            psnr_calc: Psnr::default(),
            ssim_calc: Ssim::default(),
        }
    }

    /// Calculate VMAF between reference and distorted frames.
    ///
    /// Note: This is a simplified approximation. For accurate VMAF scores,
    /// use the `vmaf` feature with libvmaf.
    pub fn calculate(&self, reference: &Frame, distorted: &Frame) -> Result<VmafResult> {
        self.validate_frames(reference, distorted)?;

        // Calculate VIF (Visual Information Fidelity)
        let vif = self.calculate_vif(reference, distorted)?;

        // Calculate DLM (Detail Loss Metric)
        let dlm = self.calculate_dlm(reference, distorted)?;

        // Calculate motion (simplified - would need previous frame for temporal)
        let motion = 0.0;

        // Calculate optional metrics
        let ssim = if self.config.compute_ssim {
            Some(self.ssim_calc.calculate(reference, distorted)?.ssim)
        } else {
            None
        };

        let psnr = if self.config.compute_psnr {
            Some(self.psnr_calc.calculate(reference, distorted)?.psnr)
        } else {
            None
        };

        // Approximate VMAF score using a simplified model
        // Real VMAF uses an SVM trained on subjective quality data
        let score = self.approximate_vmaf_score(vif, dlm, motion, ssim, psnr);

        Ok(VmafResult {
            score,
            vif,
            dlm,
            motion,
            ssim,
            psnr,
        })
    }

    /// Calculate VIF (Visual Information Fidelity).
    ///
    /// VIF measures the mutual information between reference and distorted
    /// images in the wavelet domain.
    fn calculate_vif(&self, reference: &Frame, distorted: &Frame) -> Result<f64> {
        let width = reference.width as usize;
        let height = reference.height as usize;

        // Simplified VIF using local statistics
        let block_size = 8;
        let sigma_nsq = 2.0; // Noise variance

        let mut num = 0.0;
        let mut den = 0.0;

        for y in (0..height - block_size).step_by(block_size) {
            for x in (0..width - block_size).step_by(block_size) {
                // Calculate local statistics for luminance (first channel)
                let (_mu_ref, _mu_dist, var_ref, var_dist, cov) =
                    self.calculate_block_stats(reference, distorted, x, y, 0, block_size);

                let g = if var_ref > 1e-10 { cov / var_ref } else { 0.0 };
                let sv_sq = var_dist - g * cov;
                let sv_sq = sv_sq.max(1e-10);

                // VIF computation
                let eps = 1e-10;
                num += (1.0 + (g * g * var_ref) / (sv_sq + sigma_nsq)).ln() + eps;
                den += (1.0 + var_ref / sigma_nsq).ln() + eps;
            }
        }

        Ok(if den > 0.0 { num / den } else { 0.0 })
    }

    /// Calculate DLM (Detail Loss Metric).
    ///
    /// DLM measures the loss of spatial detail.
    fn calculate_dlm(&self, reference: &Frame, distorted: &Frame) -> Result<f64> {

        // Calculate edge magnitude using Sobel-like operator
        let ref_edges = self.calculate_edges(reference)?;
        let dist_edges = self.calculate_edges(distorted)?;

        // Calculate detail preservation
        let mut detail_loss = 0.0;
        let mut total_detail = 0.0;

        for i in 0..ref_edges.len() {
            let ref_edge = ref_edges[i];
            let dist_edge = dist_edges[i];

            // Detail loss is where reference has detail but distorted doesn't
            let loss = (ref_edge - dist_edge).max(0.0);
            detail_loss += loss;
            total_detail += ref_edge;
        }

        // DLM: lower is better, we invert and normalize to 0-1 range
        let dlm = if total_detail > 0.0 {
            1.0 - (detail_loss / total_detail).min(1.0)
        } else {
            1.0
        };

        Ok(dlm)
    }

    /// Calculate edge magnitudes for a frame.
    fn calculate_edges(&self, frame: &Frame) -> Result<Vec<f64>> {
        let width = frame.width as usize;
        let height = frame.height as usize;
        let channels = frame.channels as usize;

        let mut edges = vec![0.0; width * height];

        // Sobel operators
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                // Use luminance (first channel for RGB, or actual Y for YUV)
                let get_pixel = |dx: i32, dy: i32| -> f64 {
                    let px = (x as i32 + dx) as usize;
                    let py = (y as i32 + dy) as usize;
                    frame.data[(py * width + px) * channels] as f64
                };

                // Sobel X
                let gx = -get_pixel(-1, -1) - 2.0 * get_pixel(-1, 0) - get_pixel(-1, 1)
                    + get_pixel(1, -1)
                    + 2.0 * get_pixel(1, 0)
                    + get_pixel(1, 1);

                // Sobel Y
                let gy = -get_pixel(-1, -1) - 2.0 * get_pixel(0, -1) - get_pixel(1, -1)
                    + get_pixel(-1, 1)
                    + 2.0 * get_pixel(0, 1)
                    + get_pixel(1, 1);

                edges[y * width + x] = (gx * gx + gy * gy).sqrt();
            }
        }

        Ok(edges)
    }

    /// Calculate block statistics.
    fn calculate_block_stats(
        &self,
        reference: &Frame,
        distorted: &Frame,
        x: usize,
        y: usize,
        channel: usize,
        block_size: usize,
    ) -> (f64, f64, f64, f64, f64) {
        let width = reference.width as usize;
        let channels = reference.channels as usize;

        let mut sum_ref = 0.0;
        let mut sum_dist = 0.0;
        let n = (block_size * block_size) as f64;

        // Calculate means
        for by in 0..block_size {
            for bx in 0..block_size {
                let idx = ((y + by) * width + (x + bx)) * channels + channel;
                sum_ref += reference.data[idx] as f64;
                sum_dist += distorted.data[idx] as f64;
            }
        }

        let mu_ref = sum_ref / n;
        let mu_dist = sum_dist / n;

        // Calculate variances and covariance
        let mut var_ref = 0.0;
        let mut var_dist = 0.0;
        let mut cov = 0.0;

        for by in 0..block_size {
            for bx in 0..block_size {
                let idx = ((y + by) * width + (x + bx)) * channels + channel;
                let ref_val = reference.data[idx] as f64 - mu_ref;
                let dist_val = distorted.data[idx] as f64 - mu_dist;

                var_ref += ref_val * ref_val;
                var_dist += dist_val * dist_val;
                cov += ref_val * dist_val;
            }
        }

        var_ref /= n - 1.0;
        var_dist /= n - 1.0;
        cov /= n - 1.0;

        (mu_ref, mu_dist, var_ref, var_dist, cov)
    }

    /// Approximate VMAF score from component metrics.
    ///
    /// This is a simplified approximation of the actual VMAF model,
    /// which uses an SVM trained on subjective quality data.
    fn approximate_vmaf_score(
        &self,
        vif: f64,
        dlm: f64,
        _motion: f64,
        ssim: Option<f64>,
        psnr: Option<f64>,
    ) -> f64 {
        // Simplified linear combination (actual VMAF uses SVM)
        // These weights are approximations
        let vif_weight = 0.45;
        let dlm_weight = 0.35;
        let ssim_weight = 0.15;
        let psnr_weight = 0.05;

        let mut score = 0.0;
        let mut total_weight = 0.0;

        // VIF contribution (scale to 0-100)
        let vif_score = (vif * 100.0).clamp(0.0, 100.0);
        score += vif_weight * vif_score;
        total_weight += vif_weight;

        // DLM contribution (scale to 0-100)
        let dlm_score = (dlm * 100.0).clamp(0.0, 100.0);
        score += dlm_weight * dlm_score;
        total_weight += dlm_weight;

        // SSIM contribution if available
        if let Some(ssim_val) = ssim {
            let ssim_score = (ssim_val * 100.0).clamp(0.0, 100.0);
            score += ssim_weight * ssim_score;
            total_weight += ssim_weight;
        }

        // PSNR contribution if available (normalize to 0-100 scale)
        if let Some(psnr_val) = psnr {
            let psnr_score = if psnr_val.is_infinite() {
                100.0
            } else {
                // Map PSNR to 0-100: 20dB -> 0, 50dB -> 100
                ((psnr_val - 20.0) / 30.0 * 100.0).clamp(0.0, 100.0)
            };
            score += psnr_weight * psnr_score;
            total_weight += psnr_weight;
        }

        (score / total_weight).clamp(0.0, 100.0)
    }

    /// Validate frames.
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

/// Calculate VMAF (simplified) with default settings.
pub fn vmaf(reference: &Frame, distorted: &Frame) -> Result<f64> {
    let calc = Vmaf::default();
    Ok(calc.calculate(reference, distorted)?.score)
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
    fn test_identical_frames_vmaf() {
        let frame = create_test_frame(64, 64, 128);
        let result = vmaf(&frame, &frame).unwrap();
        // Identical frames should have high VMAF (close to 100)
        assert!(result > 90.0);
    }

    #[test]
    fn test_different_frames_vmaf() {
        let ref_frame = create_test_frame(64, 64, 128);
        let dist_frame = create_test_frame(64, 64, 100);
        let result = vmaf(&ref_frame, &dist_frame).unwrap();
        // Different frames should have lower VMAF
        assert!(result < 100.0 && result > 0.0);
    }

    #[test]
    fn test_vmaf_result_rating() {
        let result = VmafResult {
            score: 95.0,
            vif: 0.95,
            dlm: 0.95,
            motion: 0.0,
            ssim: Some(0.99),
            psnr: Some(45.0),
        };
        assert_eq!(result.rating(), "Excellent");
    }
}
