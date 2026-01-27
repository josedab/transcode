//! Prometheus-compatible metrics for the transcode operator.

use std::sync::atomic::{AtomicU64, Ordering};

/// Collects operational metrics for the operator.
pub struct MetricsCollector {
    jobs_submitted: AtomicU64,
    jobs_completed: AtomicU64,
    jobs_failed: AtomicU64,
    total_frames_processed: AtomicU64,
    total_processing_secs: std::sync::Mutex<f64>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            jobs_submitted: AtomicU64::new(0),
            jobs_completed: AtomicU64::new(0),
            jobs_failed: AtomicU64::new(0),
            total_frames_processed: AtomicU64::new(0),
            total_processing_secs: std::sync::Mutex::new(0.0),
        }
    }

    pub fn record_job_submitted(&self, _job_name: &str) {
        self.jobs_submitted.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_job_completed(&self, _job_name: &str, duration_secs: f64) {
        self.jobs_completed.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut total) = self.total_processing_secs.lock() {
            *total += duration_secs;
        }
    }

    pub fn record_job_failed(&self, _job_name: &str) {
        self.jobs_failed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_frames_processed(&self, frames: u64) {
        self.total_frames_processed
            .fetch_add(frames, Ordering::Relaxed);
    }

    /// Return a point-in-time snapshot of all metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let completed = self.jobs_completed.load(Ordering::Relaxed);
        let total_secs = self
            .total_processing_secs
            .lock()
            .map(|v| *v)
            .unwrap_or(0.0);
        MetricsSnapshot {
            jobs_submitted: self.jobs_submitted.load(Ordering::Relaxed),
            jobs_completed: completed,
            jobs_failed: self.jobs_failed.load(Ordering::Relaxed),
            total_frames_processed: self.total_frames_processed.load(Ordering::Relaxed),
            avg_job_duration_secs: if completed > 0 {
                total_secs / completed as f64
            } else {
                0.0
            },
        }
    }

    /// Format metrics in Prometheus exposition format.
    pub fn prometheus_exposition(&self) -> String {
        let s = self.snapshot();
        format!(
            "# HELP transcode_jobs_submitted_total Total transcoding jobs submitted\n\
             # TYPE transcode_jobs_submitted_total counter\n\
             transcode_jobs_submitted_total {}\n\
             # HELP transcode_jobs_completed_total Total transcoding jobs completed\n\
             # TYPE transcode_jobs_completed_total counter\n\
             transcode_jobs_completed_total {}\n\
             # HELP transcode_jobs_failed_total Total transcoding jobs failed\n\
             # TYPE transcode_jobs_failed_total counter\n\
             transcode_jobs_failed_total {}\n\
             # HELP transcode_frames_processed_total Total video frames processed\n\
             # TYPE transcode_frames_processed_total counter\n\
             transcode_frames_processed_total {}\n\
             # HELP transcode_avg_job_duration_seconds Average job duration\n\
             # TYPE transcode_avg_job_duration_seconds gauge\n\
             transcode_avg_job_duration_seconds {:.2}\n",
            s.jobs_submitted,
            s.jobs_completed,
            s.jobs_failed,
            s.total_frames_processed,
            s.avg_job_duration_secs,
        )
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub jobs_submitted: u64,
    pub jobs_completed: u64,
    pub jobs_failed: u64,
    pub total_frames_processed: u64,
    pub avg_job_duration_secs: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prometheus_format() {
        let collector = MetricsCollector::new();
        collector.record_job_submitted("j1");
        collector.record_job_completed("j1", 10.0);
        let output = collector.prometheus_exposition();
        assert!(output.contains("transcode_jobs_submitted_total 1"));
        assert!(output.contains("transcode_jobs_completed_total 1"));
    }

    #[test]
    fn test_avg_duration() {
        let collector = MetricsCollector::new();
        collector.record_job_completed("j1", 10.0);
        collector.record_job_completed("j2", 20.0);
        let snapshot = collector.snapshot();
        assert!((snapshot.avg_job_duration_secs - 15.0).abs() < 0.01);
    }
}
