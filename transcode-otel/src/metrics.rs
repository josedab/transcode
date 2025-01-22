//! Metrics collection for transcoding operations.

use crate::config::TelemetryConfig;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Metrics recorder for transcoding operations.
#[derive(Clone)]
pub struct MetricsRecorder {
    inner: Arc<MetricsInner>,
}

struct MetricsInner {
    counters: RwLock<HashMap<String, AtomicU64>>,
    gauges: RwLock<HashMap<String, AtomicU64>>,
    histograms: RwLock<HashMap<String, HistogramData>>,
    job_metrics: RwLock<HashMap<String, JobMetrics>>,
    enable_quality: bool,
}

impl MetricsRecorder {
    /// Create a new metrics recorder.
    pub fn new(config: &TelemetryConfig) -> Self {
        Self {
            inner: Arc::new(MetricsInner {
                counters: RwLock::new(HashMap::new()),
                gauges: RwLock::new(HashMap::new()),
                histograms: RwLock::new(HashMap::new()),
                job_metrics: RwLock::new(HashMap::new()),
                enable_quality: config.enable_quality_metrics,
            }),
        }
    }

    /// Increment a counter.
    pub fn increment(&self, name: &str, value: u64) {
        let counters = self.inner.counters.read();
        if let Some(counter) = counters.get(name) {
            counter.fetch_add(value, Ordering::Relaxed);
        } else {
            drop(counters);
            let mut counters = self.inner.counters.write();
            counters
                .entry(name.to_string())
                .or_insert_with(|| AtomicU64::new(0))
                .fetch_add(value, Ordering::Relaxed);
        }
    }

    /// Set a gauge value.
    pub fn gauge(&self, name: &str, value: u64) {
        let mut gauges = self.inner.gauges.write();
        gauges
            .entry(name.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .store(value, Ordering::Relaxed);
    }

    /// Record a histogram observation.
    pub fn histogram(&self, name: &str, value: f64) {
        let mut histograms = self.inner.histograms.write();
        histograms
            .entry(name.to_string())
            .or_insert_with(HistogramData::new)
            .observe(value);
    }

    /// Record frame metrics.
    pub fn record_frame(&self, metrics: FrameMetrics) {
        self.increment("frames_processed", 1);
        self.histogram("frame_encode_time_ms", metrics.encode_time.as_secs_f64() * 1000.0);
        self.histogram("frame_size_bytes", metrics.size_bytes as f64);

        if self.inner.enable_quality && metrics.psnr.is_some() {
            self.histogram("frame_psnr", metrics.psnr.unwrap());
        }
        if self.inner.enable_quality && metrics.ssim.is_some() {
            self.histogram("frame_ssim", metrics.ssim.unwrap());
        }
    }

    /// Record job metrics.
    pub fn record_job(&self, job_id: &str, metrics: JobMetrics) {
        self.inner.job_metrics.write().insert(job_id.to_string(), metrics.clone());

        self.increment("jobs_completed", 1);
        self.histogram("job_duration_secs", metrics.duration.as_secs_f64());
        self.histogram("job_fps", metrics.fps);
        self.histogram("job_output_bitrate_kbps", metrics.output_bitrate_kbps as f64);
    }

    /// Record quality metrics.
    pub fn record_quality(&self, metrics: QualityMetrics) {
        if self.inner.enable_quality {
            if let Some(psnr) = metrics.psnr {
                self.histogram("quality_psnr", psnr);
            }
            if let Some(ssim) = metrics.ssim {
                self.histogram("quality_ssim", ssim);
            }
            if let Some(vmaf) = metrics.vmaf {
                self.histogram("quality_vmaf", vmaf);
            }
        }
    }

    /// Record worker metrics.
    pub fn record_worker(&self, metrics: WorkerMetrics) {
        self.gauge("worker_cpu_percent", (metrics.cpu_percent * 100.0) as u64);
        self.gauge("worker_memory_mb", metrics.memory_mb);
        self.gauge("worker_active_jobs", metrics.active_jobs);
        if let Some(gpu) = metrics.gpu_percent {
            self.gauge("worker_gpu_percent", (gpu * 100.0) as u64);
        }
    }

    /// Get a counter value.
    pub fn get_counter(&self, name: &str) -> u64 {
        self.inner
            .counters
            .read()
            .get(name)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Get a gauge value.
    pub fn get_gauge(&self, name: &str) -> u64 {
        self.inner
            .gauges
            .read()
            .get(name)
            .map(|g| g.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Get histogram statistics.
    pub fn get_histogram_stats(&self, name: &str) -> Option<HistogramStats> {
        self.inner.histograms.read().get(name).map(|h| h.stats())
    }

    /// Get all transcode metrics.
    pub fn get_transcode_metrics(&self) -> TranscodeMetrics {
        TranscodeMetrics {
            frames_processed: self.get_counter("frames_processed"),
            jobs_completed: self.get_counter("jobs_completed"),
            bytes_read: self.get_counter("bytes_read"),
            bytes_written: self.get_counter("bytes_written"),
            errors: self.get_counter("errors"),
            avg_encode_time_ms: self
                .get_histogram_stats("frame_encode_time_ms")
                .map(|s| s.mean)
                .unwrap_or(0.0),
            avg_fps: self
                .get_histogram_stats("job_fps")
                .map(|s| s.mean)
                .unwrap_or(0.0),
        }
    }
}

/// Histogram data structure.
struct HistogramData {
    values: Vec<f64>,
    sum: f64,
    count: u64,
    min: f64,
    max: f64,
}

impl HistogramData {
    fn new() -> Self {
        Self {
            values: Vec::new(),
            sum: 0.0,
            count: 0,
            min: f64::MAX,
            max: f64::MIN,
        }
    }

    fn observe(&mut self, value: f64) {
        self.values.push(value);
        self.sum += value;
        self.count += 1;
        self.min = self.min.min(value);
        self.max = self.max.max(value);

        // Keep only recent values to bound memory
        if self.values.len() > 10000 {
            self.values.drain(0..5000);
        }
    }

    fn stats(&self) -> HistogramStats {
        let mean = if self.count > 0 {
            self.sum / self.count as f64
        } else {
            0.0
        };

        let p50 = self.percentile(0.5);
        let p95 = self.percentile(0.95);
        let p99 = self.percentile(0.99);

        HistogramStats {
            count: self.count,
            sum: self.sum,
            mean,
            min: if self.min == f64::MAX { 0.0 } else { self.min },
            max: if self.max == f64::MIN { 0.0 } else { self.max },
            p50,
            p95,
            p99,
        }
    }

    fn percentile(&self, p: f64) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((sorted.len() as f64 * p) as usize).min(sorted.len() - 1);
        sorted[idx]
    }
}

/// Histogram statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramStats {
    pub count: u64,
    pub sum: f64,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

/// Metrics for a single frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameMetrics {
    pub frame_number: u64,
    pub encode_time: Duration,
    pub size_bytes: u64,
    pub frame_type: String,
    pub psnr: Option<f64>,
    pub ssim: Option<f64>,
}

/// Metrics for a transcoding job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobMetrics {
    pub job_id: String,
    pub duration: Duration,
    pub frames_encoded: u64,
    pub fps: f64,
    pub input_size_bytes: u64,
    pub output_size_bytes: u64,
    pub output_bitrate_kbps: u32,
    pub codec: String,
    pub resolution: String,
}

/// Quality metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub psnr: Option<f64>,
    pub ssim: Option<f64>,
    pub vmaf: Option<f64>,
    pub ms_ssim: Option<f64>,
}

/// Worker resource metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerMetrics {
    pub worker_id: String,
    pub cpu_percent: f64,
    pub memory_mb: u64,
    pub active_jobs: u64,
    pub gpu_percent: Option<f64>,
    pub gpu_memory_mb: Option<u64>,
}

/// Overall transcode metrics summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscodeMetrics {
    pub frames_processed: u64,
    pub jobs_completed: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub errors: u64,
    pub avg_encode_time_ms: f64,
    pub avg_fps: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter() {
        let config = TelemetryConfig::default();
        let recorder = MetricsRecorder::new(&config);

        recorder.increment("test_counter", 5);
        recorder.increment("test_counter", 3);

        assert_eq!(recorder.get_counter("test_counter"), 8);
    }

    #[test]
    fn test_gauge() {
        let config = TelemetryConfig::default();
        let recorder = MetricsRecorder::new(&config);

        recorder.gauge("test_gauge", 100);
        assert_eq!(recorder.get_gauge("test_gauge"), 100);

        recorder.gauge("test_gauge", 50);
        assert_eq!(recorder.get_gauge("test_gauge"), 50);
    }

    #[test]
    fn test_histogram() {
        let config = TelemetryConfig::default();
        let recorder = MetricsRecorder::new(&config);

        for i in 0..100 {
            recorder.histogram("test_hist", i as f64);
        }

        let stats = recorder.get_histogram_stats("test_hist").unwrap();
        assert_eq!(stats.count, 100);
        assert!(stats.mean > 40.0 && stats.mean < 60.0);
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 99.0);
    }

    #[test]
    fn test_frame_metrics() {
        let config = TelemetryConfig::default();
        let recorder = MetricsRecorder::new(&config);

        let frame_metrics = FrameMetrics {
            frame_number: 1,
            encode_time: Duration::from_millis(10),
            size_bytes: 50000,
            frame_type: "I".to_string(),
            psnr: Some(45.0),
            ssim: Some(0.98),
        };

        recorder.record_frame(frame_metrics);

        assert_eq!(recorder.get_counter("frames_processed"), 1);
    }
}
