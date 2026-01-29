//! Embeddable quality monitoring dashboard.
//!
//! Provides a web-based dashboard for visualizing real-time quality metrics
//! (PSNR, SSIM, VMAF) during and after transcoding. Can be served from the
//! CLI or embedded as a standalone application.
//!
//! # Features
//!
//! - Real-time per-frame quality metrics display
//! - Bitrate vs quality graphs
//! - Frame-by-frame quality comparison
//! - Export to HTML/PDF reports
//! - JSON/CSV data export
//!
//! # Example
//!
//! ```
//! use transcode_dashboard::{Dashboard, QualityEvent, MetricType};
//!
//! let mut dashboard = Dashboard::new();
//!
//! // Record quality events during transcoding
//! dashboard.record(QualityEvent::new(0, MetricType::Psnr, 42.5));
//! dashboard.record(QualityEvent::new(0, MetricType::Ssim, 0.98));
//! dashboard.record(QualityEvent::new(1, MetricType::Psnr, 41.2));
//! dashboard.record(QualityEvent::new(1, MetricType::Ssim, 0.97));
//!
//! // Generate summary
//! let summary = dashboard.summary();
//! assert!(summary.avg_psnr > 0.0);
//!
//! // Export report
//! let html = dashboard.export_html();
//! assert!(!html.is_empty());
//! ```

#![allow(dead_code)]

mod error;
mod collector;
mod report;
mod html;
mod alerts;

pub use error::{Error, Result};
pub use collector::{Dashboard, QualityEvent, MetricType, QualitySummary, FrameMetrics};
pub use report::{ReportFormat, ReportConfig, QualityReport};
pub use alerts::{
    AlertEngine, AlertRule, Alert, AlertSeverity, AlertCondition, AlertMetricType,
    MetricStream, StreamUpdate, StreamMetrics, ChartData, DataPoint,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_lifecycle() {
        let mut dash = Dashboard::new();
        for i in 0..100 {
            dash.record(QualityEvent::new(i, MetricType::Psnr, 40.0 + (i as f64 * 0.01)));
            dash.record(QualityEvent::new(i, MetricType::Ssim, 0.95 + (i as f64 * 0.0003)));
        }

        let summary = dash.summary();
        assert_eq!(summary.total_frames, 100);
        assert!(summary.avg_psnr > 40.0);
        assert!(summary.avg_ssim > 0.95);
    }

    #[test]
    fn test_export_html() {
        let mut dash = Dashboard::new();
        dash.record(QualityEvent::new(0, MetricType::Psnr, 42.0));
        dash.record(QualityEvent::new(0, MetricType::Ssim, 0.98));
        let html = dash.export_html();
        assert!(html.contains("Quality Dashboard"));
        assert!(html.contains("42.0"));
    }

    #[test]
    fn test_export_json() {
        let mut dash = Dashboard::new();
        dash.record(QualityEvent::new(0, MetricType::Psnr, 38.5));
        let json = dash.export_json();
        assert!(json.contains("38.5"));
    }
}
