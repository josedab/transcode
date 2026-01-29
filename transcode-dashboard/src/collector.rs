//! Quality data collection and aggregation.

use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};

/// Type of quality metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MetricType {
    Psnr,
    Ssim,
    MsSsim,
    Vmaf,
    Bitrate,
}

/// A single quality measurement for a frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityEvent {
    pub frame: u64,
    pub metric: MetricType,
    pub value: f64,
}

impl QualityEvent {
    pub fn new(frame: u64, metric: MetricType, value: f64) -> Self {
        Self { frame, metric, value }
    }
}

/// Aggregated metrics for a single frame.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FrameMetrics {
    pub frame: u64,
    pub psnr: Option<f64>,
    pub ssim: Option<f64>,
    pub ms_ssim: Option<f64>,
    pub vmaf: Option<f64>,
    pub bitrate: Option<f64>,
}

/// Summary statistics for the entire encode.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualitySummary {
    pub total_frames: u64,
    pub avg_psnr: f64,
    pub min_psnr: f64,
    pub max_psnr: f64,
    pub avg_ssim: f64,
    pub min_ssim: f64,
    pub max_ssim: f64,
    pub avg_vmaf: f64,
    pub avg_bitrate: f64,
}

/// Collects quality events and provides aggregated views.
pub struct Dashboard {
    events: Vec<QualityEvent>,
    frames: BTreeMap<u64, FrameMetrics>,
}

impl Dashboard {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            frames: BTreeMap::new(),
        }
    }

    /// Record a quality event.
    pub fn record(&mut self, event: QualityEvent) {
        let frame_num = event.frame;
        let entry = self.frames.entry(frame_num).or_insert_with(|| FrameMetrics {
            frame: frame_num,
            ..Default::default()
        });

        match event.metric {
            MetricType::Psnr => entry.psnr = Some(event.value),
            MetricType::Ssim => entry.ssim = Some(event.value),
            MetricType::MsSsim => entry.ms_ssim = Some(event.value),
            MetricType::Vmaf => entry.vmaf = Some(event.value),
            MetricType::Bitrate => entry.bitrate = Some(event.value),
        }

        self.events.push(event);
    }

    /// Compute summary statistics.
    pub fn summary(&self) -> QualitySummary {
        let psnr_values: Vec<f64> = self.frames.values().filter_map(|f| f.psnr).collect();
        let ssim_values: Vec<f64> = self.frames.values().filter_map(|f| f.ssim).collect();
        let vmaf_values: Vec<f64> = self.frames.values().filter_map(|f| f.vmaf).collect();
        let bitrate_values: Vec<f64> = self.frames.values().filter_map(|f| f.bitrate).collect();

        QualitySummary {
            total_frames: self.frames.len() as u64,
            avg_psnr: avg(&psnr_values),
            min_psnr: psnr_values.iter().copied().fold(f64::INFINITY, f64::min),
            max_psnr: psnr_values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            avg_ssim: avg(&ssim_values),
            min_ssim: ssim_values.iter().copied().fold(f64::INFINITY, f64::min),
            max_ssim: ssim_values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            avg_vmaf: avg(&vmaf_values),
            avg_bitrate: avg(&bitrate_values),
        }
    }

    /// Get per-frame metrics.
    pub fn frame_metrics(&self) -> Vec<&FrameMetrics> {
        self.frames.values().collect()
    }

    /// Get metrics for a specific frame.
    pub fn get_frame(&self, frame: u64) -> Option<&FrameMetrics> {
        self.frames.get(&frame)
    }

    /// Total number of frames with data.
    pub fn frame_count(&self) -> u64 {
        self.frames.len() as u64
    }

    /// Export as JSON.
    pub fn export_json(&self) -> String {
        let data: Vec<&FrameMetrics> = self.frames.values().collect();
        serde_json::to_string_pretty(&data).unwrap_or_default()
    }

    /// Export as CSV.
    pub fn export_csv(&self) -> String {
        let mut out = "frame,psnr,ssim,ms_ssim,vmaf,bitrate\n".to_string();
        for fm in self.frames.values() {
            out.push_str(&format!(
                "{},{},{},{},{},{}\n",
                fm.frame,
                fm.psnr.map(|v| format!("{:.2}", v)).unwrap_or_default(),
                fm.ssim.map(|v| format!("{:.4}", v)).unwrap_or_default(),
                fm.ms_ssim.map(|v| format!("{:.4}", v)).unwrap_or_default(),
                fm.vmaf.map(|v| format!("{:.2}", v)).unwrap_or_default(),
                fm.bitrate.map(|v| format!("{:.0}", v)).unwrap_or_default(),
            ));
        }
        out
    }

    /// Export as HTML dashboard.
    pub fn export_html(&self) -> String {
        crate::html::render_dashboard(self)
    }
}

impl Default for Dashboard {
    fn default() -> Self {
        Self::new()
    }
}

fn avg(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_summary() {
        let mut dash = Dashboard::new();
        dash.record(QualityEvent::new(0, MetricType::Psnr, 40.0));
        dash.record(QualityEvent::new(0, MetricType::Ssim, 0.95));
        dash.record(QualityEvent::new(1, MetricType::Psnr, 42.0));
        dash.record(QualityEvent::new(1, MetricType::Ssim, 0.97));

        let summary = dash.summary();
        assert_eq!(summary.total_frames, 2);
        assert!((summary.avg_psnr - 41.0).abs() < 0.01);
        assert!((summary.avg_ssim - 0.96).abs() < 0.01);
    }

    #[test]
    fn test_csv_export() {
        let mut dash = Dashboard::new();
        dash.record(QualityEvent::new(0, MetricType::Psnr, 38.5));
        let csv = dash.export_csv();
        assert!(csv.contains("frame,psnr"));
        assert!(csv.contains("38.50"));
    }

    #[test]
    fn test_empty_summary() {
        let dash = Dashboard::new();
        let summary = dash.summary();
        assert_eq!(summary.total_frames, 0);
        assert_eq!(summary.avg_psnr, 0.0);
    }
}
