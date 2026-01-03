//! Metric types and calculations

use std::time::Duration;

/// Encoding performance metrics
#[derive(Debug, Clone, Default)]
pub struct EncodingMetrics {
    /// Frames encoded
    pub frames_encoded: u64,
    /// Total encoding time
    pub total_encode_time: Duration,
    /// Fastest frame
    pub min_frame_time: Option<Duration>,
    /// Slowest frame
    pub max_frame_time: Option<Duration>,
    /// Total output bytes
    pub total_bytes: u64,
}

impl EncodingMetrics {
    /// Get average frame time
    pub fn avg_frame_time(&self) -> Duration {
        if self.frames_encoded > 0 {
            self.total_encode_time / self.frames_encoded as u32
        } else {
            Duration::ZERO
        }
    }

    /// Get average bitrate in kbps
    pub fn avg_bitrate_kbps(&self, fps: f64) -> f64 {
        if self.frames_encoded > 0 && fps > 0.0 {
            let duration_secs = self.frames_encoded as f64 / fps;
            (self.total_bytes as f64 * 8.0) / duration_secs / 1000.0
        } else {
            0.0
        }
    }

    /// Update with new frame
    pub fn update(&mut self, frame_time: Duration, frame_bytes: u64) {
        self.frames_encoded += 1;
        self.total_encode_time += frame_time;
        self.total_bytes += frame_bytes;

        match self.min_frame_time {
            Some(min) if frame_time < min => self.min_frame_time = Some(frame_time),
            None => self.min_frame_time = Some(frame_time),
            _ => {}
        }

        match self.max_frame_time {
            Some(max) if frame_time > max => self.max_frame_time = Some(frame_time),
            None => self.max_frame_time = Some(frame_time),
            _ => {}
        }
    }
}

/// Quality metrics aggregation
#[derive(Debug, Clone, Default)]
pub struct QualityMetrics {
    /// Number of samples
    pub sample_count: u64,
    /// PSNR sum
    pub psnr_sum: f64,
    /// PSNR min
    pub psnr_min: Option<f64>,
    /// PSNR max
    pub psnr_max: Option<f64>,
    /// SSIM sum
    pub ssim_sum: f64,
    /// SSIM min
    pub ssim_min: Option<f64>,
    /// SSIM max
    pub ssim_max: Option<f64>,
}

impl QualityMetrics {
    /// Add PSNR sample
    pub fn add_psnr(&mut self, psnr: f64) {
        self.sample_count += 1;
        self.psnr_sum += psnr;

        match self.psnr_min {
            Some(min) if psnr < min => self.psnr_min = Some(psnr),
            None => self.psnr_min = Some(psnr),
            _ => {}
        }

        match self.psnr_max {
            Some(max) if psnr > max => self.psnr_max = Some(psnr),
            None => self.psnr_max = Some(psnr),
            _ => {}
        }
    }

    /// Add SSIM sample
    pub fn add_ssim(&mut self, ssim: f64) {
        self.ssim_sum += ssim;

        match self.ssim_min {
            Some(min) if ssim < min => self.ssim_min = Some(ssim),
            None => self.ssim_min = Some(ssim),
            _ => {}
        }

        match self.ssim_max {
            Some(max) if ssim > max => self.ssim_max = Some(ssim),
            None => self.ssim_max = Some(ssim),
            _ => {}
        }
    }

    /// Get average PSNR
    pub fn avg_psnr(&self) -> Option<f64> {
        if self.sample_count > 0 {
            Some(self.psnr_sum / self.sample_count as f64)
        } else {
            None
        }
    }

    /// Get average SSIM
    pub fn avg_ssim(&self) -> Option<f64> {
        if self.sample_count > 0 {
            Some(self.ssim_sum / self.sample_count as f64)
        } else {
            None
        }
    }
}

/// Bitrate statistics
#[derive(Debug, Clone, Default)]
pub struct BitrateStats {
    /// Target bitrate
    pub target_kbps: Option<u32>,
    /// Current bitrate
    pub current_kbps: f64,
    /// Minimum bitrate observed
    pub min_kbps: f64,
    /// Maximum bitrate observed
    pub max_kbps: f64,
    /// Average bitrate
    pub avg_kbps: f64,
    /// Bitrate variance
    pub variance: f64,
}

impl BitrateStats {
    /// Update with new bitrate sample
    pub fn update(&mut self, bitrate_kbps: f64, sample_count: u64) {
        self.current_kbps = bitrate_kbps;

        if bitrate_kbps < self.min_kbps || self.min_kbps == 0.0 {
            self.min_kbps = bitrate_kbps;
        }
        if bitrate_kbps > self.max_kbps {
            self.max_kbps = bitrate_kbps;
        }

        // Running average
        if sample_count > 0 {
            let n = sample_count as f64;
            self.avg_kbps = self.avg_kbps * (n - 1.0) / n + bitrate_kbps / n;
        }
    }

    /// Check if meeting target
    pub fn is_on_target(&self, tolerance: f64) -> Option<bool> {
        self.target_kbps.map(|target| {
            let diff = (self.current_kbps - target as f64).abs();
            diff / target as f64 <= tolerance
        })
    }
}
