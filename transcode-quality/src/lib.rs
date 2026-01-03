//! Perceptual video quality metrics for transcode.
//!
//! This crate provides implementations of common video quality metrics:
//!
//! - **PSNR** (Peak Signal-to-Noise Ratio) - Simple, fast pixel-based metric
//! - **SSIM** (Structural Similarity Index) - Perceptual metric based on structure
//! - **MS-SSIM** (Multi-Scale SSIM) - Multi-scale version of SSIM
//! - **VMAF** (Video Multi-Method Assessment Fusion) - Netflix's ML-based metric
//!
//! # Example
//!
//! ```ignore
//! use transcode_quality::{psnr, ssim, vmaf, Frame};
//!
//! let reference = Frame::new(ref_data, 1920, 1080, 3);
//! let distorted = Frame::new(dist_data, 1920, 1080, 3);
//!
//! // Calculate individual metrics
//! let psnr_score = psnr(&reference, &distorted)?;
//! let ssim_score = ssim(&reference, &distorted)?;
//! let vmaf_score = vmaf(&reference, &distorted)?;
//!
//! // Or use the unified QualityAssessment
//! let qa = QualityAssessment::new(QualityConfig::default());
//! let report = qa.assess(&reference, &distorted)?;
//! println!("{}", report);
//! ```
//!
//! # Quality Metric Comparison
//!
//! | Metric | Range | Speed | Correlation with Human Perception |
//! |--------|-------|-------|----------------------------------|
//! | PSNR | 0-∞ dB | Fast | Low |
//! | SSIM | -1 to 1 | Medium | Medium |
//! | MS-SSIM | -1 to 1 | Slow | High |
//! | VMAF | 0-100 | Slow | Very High |

pub mod error;
pub mod psnr;
pub mod ssim;
pub mod vmaf;

pub use error::{QualityError, Result};
pub use psnr::{psnr, psnr_yuv_weighted, Psnr, PsnrConfig, PsnrResult};
pub use ssim::{ms_ssim, ssim, MsSsim, Ssim, SsimConfig, SsimResult};
pub use vmaf::{vmaf, Vmaf, VmafConfig, VmafModel, VmafResult};

/// A video frame for quality assessment.
///
/// This is a simple RGB/YUV frame representation compatible with
/// the quality metrics. For integration with other transcode crates,
/// convert your frames to this format.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Raw pixel data (packed RGB or YUV).
    pub data: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Number of channels (3 for RGB/YUV).
    pub channels: u8,
}

impl Frame {
    /// Create a new frame.
    pub fn new(data: Vec<u8>, width: u32, height: u32, channels: u8) -> Self {
        Self {
            data,
            width,
            height,
            channels,
        }
    }

    /// Validate frame dimensions.
    pub fn validate(&self) -> Result<()> {
        let expected_size = (self.width * self.height * self.channels as u32) as usize;
        if self.data.len() != expected_size {
            return Err(QualityError::InvalidFrame(format!(
                "Expected {} bytes, got {}",
                expected_size,
                self.data.len()
            )));
        }
        Ok(())
    }
}

/// Quality metrics to compute.
#[derive(Debug, Clone, Copy)]
pub struct QualityMetrics {
    /// Compute PSNR.
    pub psnr: bool,
    /// Compute SSIM.
    pub ssim: bool,
    /// Compute MS-SSIM.
    pub ms_ssim: bool,
    /// Compute VMAF.
    pub vmaf: bool,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            psnr: true,
            ssim: true,
            ms_ssim: false,
            vmaf: true,
        }
    }
}

impl QualityMetrics {
    /// Enable all metrics.
    pub fn all() -> Self {
        Self {
            psnr: true,
            ssim: true,
            ms_ssim: true,
            vmaf: true,
        }
    }

    /// Fast metrics only (PSNR, SSIM).
    pub fn fast() -> Self {
        Self {
            psnr: true,
            ssim: true,
            ms_ssim: false,
            vmaf: false,
        }
    }
}

/// Configuration for quality assessment.
#[derive(Debug, Clone)]
pub struct QualityConfig {
    /// Which metrics to compute.
    pub metrics: QualityMetrics,
    /// Bit depth of source content.
    pub bit_depth: u8,
    /// VMAF model to use.
    pub vmaf_model: VmafModel,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            metrics: QualityMetrics::default(),
            bit_depth: 8,
            vmaf_model: VmafModel::Default,
        }
    }
}

/// Complete quality assessment report.
#[derive(Debug, Clone)]
pub struct QualityReport {
    /// PSNR result (if computed).
    pub psnr: Option<PsnrResult>,
    /// SSIM result (if computed).
    pub ssim: Option<SsimResult>,
    /// MS-SSIM score (if computed).
    pub ms_ssim: Option<f64>,
    /// VMAF result (if computed).
    pub vmaf: Option<VmafResult>,
}

impl QualityReport {
    /// Get overall quality score (0-100 scale, based on VMAF or estimated).
    pub fn overall_score(&self) -> f64 {
        // Prefer VMAF if available
        if let Some(ref vmaf) = self.vmaf {
            return vmaf.score;
        }

        // Fall back to SSIM-based estimate
        if let Some(ref ssim) = self.ssim {
            return ssim.ssim * 100.0;
        }

        // Fall back to PSNR-based estimate
        if let Some(ref psnr) = self.psnr {
            if psnr.psnr.is_infinite() {
                return 100.0;
            }
            // Map PSNR to 0-100: 20dB -> 0, 50dB -> 100
            return ((psnr.psnr - 20.0) / 30.0 * 100.0).clamp(0.0, 100.0);
        }

        0.0
    }

    /// Get quality rating string.
    pub fn rating(&self) -> &'static str {
        let score = self.overall_score();
        if score >= 93.0 {
            "Excellent"
        } else if score >= 80.0 {
            "Good"
        } else if score >= 60.0 {
            "Fair"
        } else {
            "Poor"
        }
    }
}

impl std::fmt::Display for QualityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Quality Report")?;
        writeln!(f, "==============")?;

        if let Some(ref psnr) = self.psnr {
            writeln!(f, "PSNR: {:.2} dB", psnr.psnr)?;
        }

        if let Some(ref ssim) = self.ssim {
            writeln!(f, "SSIM: {:.4}", ssim.ssim)?;
        }

        if let Some(ms_ssim) = self.ms_ssim {
            writeln!(f, "MS-SSIM: {:.4}", ms_ssim)?;
        }

        if let Some(ref vmaf) = self.vmaf {
            writeln!(f, "VMAF: {:.2}", vmaf.score)?;
        }

        writeln!(f)?;
        writeln!(f, "Overall: {:.1} ({})", self.overall_score(), self.rating())?;

        Ok(())
    }
}

/// Unified quality assessment calculator.
#[derive(Debug)]
pub struct QualityAssessment {
    config: QualityConfig,
    psnr_calc: Psnr,
    ssim_calc: Ssim,
    ms_ssim_calc: MsSsim,
    vmaf_calc: Vmaf,
}

impl Default for QualityAssessment {
    fn default() -> Self {
        Self::new(QualityConfig::default())
    }
}

impl QualityAssessment {
    /// Create a new quality assessment calculator.
    pub fn new(config: QualityConfig) -> Self {
        let psnr_config = PsnrConfig {
            bit_depth: config.bit_depth,
        };

        let ssim_config = SsimConfig::default().with_bit_depth(config.bit_depth);

        let vmaf_config = VmafConfig::default().with_model(config.vmaf_model);

        Self {
            config,
            psnr_calc: Psnr::new(psnr_config),
            ssim_calc: Ssim::new(ssim_config.clone()),
            ms_ssim_calc: MsSsim::new(5, ssim_config),
            vmaf_calc: Vmaf::new(vmaf_config),
        }
    }

    /// Assess quality between reference and distorted frames.
    pub fn assess(&self, reference: &Frame, distorted: &Frame) -> Result<QualityReport> {
        let psnr = if self.config.metrics.psnr {
            Some(self.psnr_calc.calculate(reference, distorted)?)
        } else {
            None
        };

        let ssim = if self.config.metrics.ssim {
            Some(self.ssim_calc.calculate(reference, distorted)?)
        } else {
            None
        };

        let ms_ssim = if self.config.metrics.ms_ssim {
            Some(self.ms_ssim_calc.calculate(reference, distorted)?)
        } else {
            None
        };

        let vmaf = if self.config.metrics.vmaf {
            Some(self.vmaf_calc.calculate(reference, distorted)?)
        } else {
            None
        };

        Ok(QualityReport {
            psnr,
            ssim,
            ms_ssim,
            vmaf,
        })
    }

    /// Quick assessment with just PSNR and SSIM.
    pub fn assess_fast(&self, reference: &Frame, distorted: &Frame) -> Result<QualityReport> {
        let psnr = Some(self.psnr_calc.calculate(reference, distorted)?);
        let ssim = Some(self.ssim_calc.calculate(reference, distorted)?);

        Ok(QualityReport {
            psnr,
            ssim,
            ms_ssim: None,
            vmaf: None,
        })
    }
}

/// Batch quality assessment for video sequences.
#[derive(Debug, Default)]
pub struct BatchQualityAssessment {
    /// Individual frame scores.
    pub frame_scores: Vec<QualityReport>,
}

impl BatchQualityAssessment {
    /// Create a new batch assessment.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a frame score.
    pub fn add(&mut self, report: QualityReport) {
        self.frame_scores.push(report);
    }

    /// Get average PSNR across all frames.
    pub fn average_psnr(&self) -> Option<f64> {
        let scores: Vec<f64> = self
            .frame_scores
            .iter()
            .filter_map(|r| r.psnr.as_ref().map(|p| p.psnr))
            .filter(|p| !p.is_infinite())
            .collect();

        if scores.is_empty() {
            None
        } else {
            Some(scores.iter().sum::<f64>() / scores.len() as f64)
        }
    }

    /// Get average SSIM across all frames.
    pub fn average_ssim(&self) -> Option<f64> {
        let scores: Vec<f64> = self
            .frame_scores
            .iter()
            .filter_map(|r| r.ssim.as_ref().map(|s| s.ssim))
            .collect();

        if scores.is_empty() {
            None
        } else {
            Some(scores.iter().sum::<f64>() / scores.len() as f64)
        }
    }

    /// Get average VMAF across all frames.
    pub fn average_vmaf(&self) -> Option<f64> {
        let scores: Vec<f64> = self
            .frame_scores
            .iter()
            .filter_map(|r| r.vmaf.as_ref().map(|v| v.score))
            .collect();

        if scores.is_empty() {
            None
        } else {
            Some(scores.iter().sum::<f64>() / scores.len() as f64)
        }
    }

    /// Get minimum scores (worst quality frames).
    pub fn min_scores(&self) -> (Option<f64>, Option<f64>, Option<f64>) {
        let min_psnr = self
            .frame_scores
            .iter()
            .filter_map(|r| r.psnr.as_ref().map(|p| p.psnr))
            .filter(|p| !p.is_infinite())
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min_ssim = self
            .frame_scores
            .iter()
            .filter_map(|r| r.ssim.as_ref().map(|s| s.ssim))
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min_vmaf = self
            .frame_scores
            .iter()
            .filter_map(|r| r.vmaf.as_ref().map(|v| v.score))
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        (min_psnr, min_ssim, min_vmaf)
    }

    /// Get summary report.
    pub fn summary(&self) -> String {
        let mut result = String::new();
        result.push_str("Batch Quality Summary\n");
        result.push_str("=====================\n");
        result.push_str(&format!("Frames analyzed: {}\n\n", self.frame_scores.len()));

        if let Some(psnr) = self.average_psnr() {
            result.push_str(&format!("Average PSNR: {:.2} dB\n", psnr));
        }

        if let Some(ssim) = self.average_ssim() {
            result.push_str(&format!("Average SSIM: {:.4}\n", ssim));
        }

        if let Some(vmaf) = self.average_vmaf() {
            result.push_str(&format!("Average VMAF: {:.2}\n", vmaf));
        }

        let (min_psnr, min_ssim, min_vmaf) = self.min_scores();
        result.push_str("\nMinimum scores (worst frames):\n");

        if let Some(psnr) = min_psnr {
            result.push_str(&format!("  PSNR: {:.2} dB\n", psnr));
        }

        if let Some(ssim) = min_ssim {
            result.push_str(&format!("  SSIM: {:.4}\n", ssim));
        }

        if let Some(vmaf) = min_vmaf {
            result.push_str(&format!("  VMAF: {:.2}\n", vmaf));
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(width: u32, height: u32, value: u8) -> Frame {
        let data = vec![value; (width * height * 3) as usize];
        Frame::new(data, width, height, 3)
    }

    #[test]
    fn test_quality_assessment() {
        let reference = create_test_frame(64, 64, 128);
        let distorted = create_test_frame(64, 64, 138);

        let qa = QualityAssessment::default();
        let report = qa.assess(&reference, &distorted).unwrap();

        assert!(report.psnr.is_some());
        assert!(report.ssim.is_some());
        assert!(report.vmaf.is_some());
    }

    #[test]
    fn test_quality_report_display() {
        let reference = create_test_frame(64, 64, 128);
        let qa = QualityAssessment::default();
        let report = qa.assess(&reference, &reference).unwrap();

        let display = format!("{}", report);
        assert!(display.contains("PSNR"));
        assert!(display.contains("SSIM"));
    }
}
