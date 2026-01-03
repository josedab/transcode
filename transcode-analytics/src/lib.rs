//! Real-time quality analytics for transcode
//!
//! This crate provides real-time monitoring and analytics for transcoding pipelines.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

mod error;
mod metrics;
mod timeseries;

pub use error::*;
pub use metrics::*;
pub use timeseries::*;

/// Result type for analytics operations
pub type Result<T> = std::result::Result<T, AnalyticsError>;

/// Analytics configuration
#[derive(Debug, Clone)]
pub struct AnalyticsConfig {
    /// Window size for moving averages (in samples)
    pub window_size: usize,
    /// Sampling interval
    pub sample_interval: Duration,
    /// Enable detailed metrics
    pub detailed_metrics: bool,
    /// Enable quality metrics
    pub quality_metrics: bool,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            sample_interval: Duration::from_millis(100),
            detailed_metrics: true,
            quality_metrics: true,
        }
    }
}

/// Frame statistics
#[derive(Debug, Clone, Default)]
pub struct FrameStats {
    /// Frame number
    pub frame_number: u64,
    /// Frame size in bytes
    pub size_bytes: u64,
    /// Encoding time
    pub encode_time: Duration,
    /// Frame type (I, P, B)
    pub frame_type: FrameType,
    /// Quantization parameter
    pub qp: Option<f32>,
    /// PSNR (if calculated)
    pub psnr: Option<f64>,
    /// SSIM (if calculated)
    pub ssim: Option<f64>,
}

/// Frame type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrameType {
    /// Intra frame
    #[default]
    I,
    /// Predicted frame
    P,
    /// Bidirectional frame
    B,
}

/// Real-time analytics
#[derive(Debug, Clone)]
pub struct RealtimeStats {
    /// Frames per second
    pub fps: f64,
    /// Current bitrate (kbps)
    pub bitrate_kbps: f64,
    /// Average encode time per frame
    pub avg_encode_time: Duration,
    /// Buffer fullness (0.0-1.0)
    pub buffer_fullness: f64,
    /// Dropped frames
    pub dropped_frames: u64,
    /// Current quality score (0-100)
    pub quality_score: f64,
}

impl Default for RealtimeStats {
    fn default() -> Self {
        Self {
            fps: 0.0,
            bitrate_kbps: 0.0,
            avg_encode_time: Duration::ZERO,
            buffer_fullness: 0.0,
            dropped_frames: 0,
            quality_score: 100.0,
        }
    }
}

/// Analytics collector
pub struct AnalyticsCollector {
    config: AnalyticsConfig,
    frame_stats: VecDeque<FrameStats>,
    start_time: Instant,
    total_frames: u64,
    total_bytes: u64,
    dropped_frames: u64,
    quality_sum: f64,
    quality_count: u64,
}

impl AnalyticsCollector {
    /// Create a new analytics collector
    pub fn new(config: AnalyticsConfig) -> Self {
        Self {
            config,
            frame_stats: VecDeque::new(),
            start_time: Instant::now(),
            total_frames: 0,
            total_bytes: 0,
            dropped_frames: 0,
            quality_sum: 0.0,
            quality_count: 0,
        }
    }

    /// Record frame statistics
    pub fn record_frame(&mut self, stats: FrameStats) {
        self.total_frames += 1;
        self.total_bytes += stats.size_bytes;

        if let Some(psnr) = stats.psnr {
            self.quality_sum += psnr;
            self.quality_count += 1;
        }

        self.frame_stats.push_back(stats);

        // Trim to window size
        while self.frame_stats.len() > self.config.window_size {
            self.frame_stats.pop_front();
        }
    }

    /// Record dropped frame
    pub fn record_dropped(&mut self) {
        self.dropped_frames += 1;
    }

    /// Get current real-time statistics
    pub fn get_realtime_stats(&self) -> RealtimeStats {
        let elapsed = self.start_time.elapsed();

        let fps = if elapsed.as_secs_f64() > 0.0 {
            self.total_frames as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        let bitrate_kbps = if elapsed.as_secs_f64() > 0.0 {
            (self.total_bytes as f64 * 8.0) / elapsed.as_secs_f64() / 1000.0
        } else {
            0.0
        };

        let avg_encode_time = if !self.frame_stats.is_empty() {
            let total: Duration = self.frame_stats.iter()
                .map(|s| s.encode_time)
                .sum();
            total / self.frame_stats.len() as u32
        } else {
            Duration::ZERO
        };

        let quality_score = if self.quality_count > 0 {
            (self.quality_sum / self.quality_count as f64).min(100.0)
        } else {
            100.0
        };

        RealtimeStats {
            fps,
            bitrate_kbps,
            avg_encode_time,
            buffer_fullness: 0.5, // Would need buffer info
            dropped_frames: self.dropped_frames,
            quality_score,
        }
    }

    /// Get frame type distribution
    pub fn get_frame_distribution(&self) -> FrameDistribution {
        let mut dist = FrameDistribution::default();

        for stats in &self.frame_stats {
            match stats.frame_type {
                FrameType::I => dist.i_frames += 1,
                FrameType::P => dist.p_frames += 1,
                FrameType::B => dist.b_frames += 1,
            }
        }

        let total = (dist.i_frames + dist.p_frames + dist.b_frames) as f64;
        if total > 0.0 {
            dist.i_ratio = dist.i_frames as f64 / total;
            dist.p_ratio = dist.p_frames as f64 / total;
            dist.b_ratio = dist.b_frames as f64 / total;
        }

        dist
    }

    /// Get bitrate over time
    pub fn get_bitrate_history(&self) -> Vec<(Duration, f64)> {
        let mut history = Vec::new();
        let mut accumulated_bytes = 0u64;
        let mut accumulated_time = Duration::ZERO;

        for stats in &self.frame_stats {
            accumulated_bytes += stats.size_bytes;
            accumulated_time += stats.encode_time;

            if accumulated_time.as_secs_f64() > 0.0 {
                let bitrate = (accumulated_bytes as f64 * 8.0) / accumulated_time.as_secs_f64() / 1000.0;
                history.push((accumulated_time, bitrate));
            }
        }

        history
    }

    /// Get quality metrics over time
    pub fn get_quality_history(&self) -> Vec<QualityPoint> {
        self.frame_stats.iter()
            .filter_map(|s| {
                s.psnr.map(|psnr| QualityPoint {
                    frame_number: s.frame_number,
                    psnr,
                    ssim: s.ssim,
                })
            })
            .collect()
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        self.frame_stats.clear();
        self.start_time = Instant::now();
        self.total_frames = 0;
        self.total_bytes = 0;
        self.dropped_frames = 0;
        self.quality_sum = 0.0;
        self.quality_count = 0;
    }

    /// Get total frames processed
    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// Get total bytes output
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Frame type distribution
#[derive(Debug, Clone, Default)]
pub struct FrameDistribution {
    pub i_frames: u64,
    pub p_frames: u64,
    pub b_frames: u64,
    pub i_ratio: f64,
    pub p_ratio: f64,
    pub b_ratio: f64,
}

/// Quality data point
#[derive(Debug, Clone)]
pub struct QualityPoint {
    pub frame_number: u64,
    pub psnr: f64,
    pub ssim: Option<f64>,
}

/// Analytics report
#[derive(Debug, Clone)]
pub struct AnalyticsReport {
    /// Duration of encoding
    pub duration: Duration,
    /// Total frames
    pub total_frames: u64,
    /// Total output size
    pub total_bytes: u64,
    /// Average FPS
    pub avg_fps: f64,
    /// Average bitrate
    pub avg_bitrate_kbps: f64,
    /// Frame distribution
    pub frame_distribution: FrameDistribution,
    /// Average PSNR
    pub avg_psnr: Option<f64>,
    /// Average SSIM
    pub avg_ssim: Option<f64>,
    /// Dropped frames
    pub dropped_frames: u64,
    /// Quality score
    pub quality_score: f64,
}

impl AnalyticsCollector {
    /// Generate final report
    pub fn generate_report(&self) -> AnalyticsReport {
        let stats = self.get_realtime_stats();
        let dist = self.get_frame_distribution();

        let avg_psnr = if self.quality_count > 0 {
            Some(self.quality_sum / self.quality_count as f64)
        } else {
            None
        };

        let avg_ssim = {
            let ssim_sum: f64 = self.frame_stats.iter()
                .filter_map(|s| s.ssim)
                .sum();
            let ssim_count = self.frame_stats.iter()
                .filter(|s| s.ssim.is_some())
                .count();
            if ssim_count > 0 {
                Some(ssim_sum / ssim_count as f64)
            } else {
                None
            }
        };

        AnalyticsReport {
            duration: self.elapsed(),
            total_frames: self.total_frames,
            total_bytes: self.total_bytes,
            avg_fps: stats.fps,
            avg_bitrate_kbps: stats.bitrate_kbps,
            frame_distribution: dist,
            avg_psnr,
            avg_ssim,
            dropped_frames: self.dropped_frames,
            quality_score: stats.quality_score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analytics_collector() {
        let config = AnalyticsConfig::default();
        let mut collector = AnalyticsCollector::new(config);

        for i in 0..100 {
            collector.record_frame(FrameStats {
                frame_number: i,
                size_bytes: 10000,
                encode_time: Duration::from_millis(10),
                frame_type: if i % 30 == 0 { FrameType::I } else { FrameType::P },
                qp: Some(25.0),
                psnr: Some(40.0),
                ssim: Some(0.95),
            });
        }

        let stats = collector.get_realtime_stats();
        assert!(stats.fps > 0.0);
        assert!(stats.bitrate_kbps > 0.0);

        let dist = collector.get_frame_distribution();
        assert!(dist.i_frames > 0);
        assert!(dist.p_frames > 0);
    }

    #[test]
    fn test_frame_distribution() {
        let config = AnalyticsConfig::default();
        let mut collector = AnalyticsCollector::new(config);

        // Add 1 I-frame and 29 P-frames (typical GOP)
        collector.record_frame(FrameStats {
            frame_type: FrameType::I,
            ..Default::default()
        });
        for _ in 0..29 {
            collector.record_frame(FrameStats {
                frame_type: FrameType::P,
                ..Default::default()
            });
        }

        let dist = collector.get_frame_distribution();
        assert_eq!(dist.i_frames, 1);
        assert_eq!(dist.p_frames, 29);
    }
}
