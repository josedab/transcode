//! Metrics collection for the transcode system.
//!
//! This module provides a comprehensive metrics system for monitoring transcode operations,
//! including counters, histograms, and gauges for tracking job performance, codec metrics,
//! error rates, and resource utilization.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use crate::error::{MetricsError, Result};

/// Labels for metric dimensions.
pub type Labels = HashMap<String, String>;

/// A counter metric that can only increase.
#[derive(Debug)]
pub struct Counter {
    name: String,
    description: String,
    value: AtomicU64,
    labels: RwLock<Labels>,
}

impl Counter {
    /// Create a new counter.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            value: AtomicU64::new(0),
            labels: RwLock::new(HashMap::new()),
        }
    }

    /// Create a new counter with labels.
    pub fn with_labels(
        name: impl Into<String>,
        description: impl Into<String>,
        labels: Labels,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            value: AtomicU64::new(0),
            labels: RwLock::new(labels),
        }
    }

    /// Increment the counter by 1.
    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the counter by a specific amount.
    pub fn inc_by(&self, delta: u64) {
        self.value.fetch_add(delta, Ordering::Relaxed);
    }

    /// Get the current value.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Get the counter name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the counter description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Get a copy of the labels.
    pub fn labels(&self) -> Labels {
        self.labels.read().unwrap().clone()
    }

    /// Add a label.
    pub fn add_label(&self, key: impl Into<String>, value: impl Into<String>) {
        self.labels.write().unwrap().insert(key.into(), value.into());
    }
}

/// A gauge metric that can increase or decrease.
#[derive(Debug)]
pub struct Gauge {
    name: String,
    description: String,
    value: AtomicU64, // Stored as bits of f64
    labels: RwLock<Labels>,
}

impl Gauge {
    /// Create a new gauge.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            value: AtomicU64::new(0.0_f64.to_bits()),
            labels: RwLock::new(HashMap::new()),
        }
    }

    /// Create a new gauge with labels.
    pub fn with_labels(
        name: impl Into<String>,
        description: impl Into<String>,
        labels: Labels,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            value: AtomicU64::new(0.0_f64.to_bits()),
            labels: RwLock::new(labels),
        }
    }

    /// Set the gauge value.
    pub fn set(&self, value: f64) {
        self.value.store(value.to_bits(), Ordering::Relaxed);
    }

    /// Increment the gauge by 1.
    pub fn inc(&self) {
        self.add(1.0);
    }

    /// Decrement the gauge by 1.
    pub fn dec(&self) {
        self.sub(1.0);
    }

    /// Add to the gauge value.
    pub fn add(&self, delta: f64) {
        loop {
            let current = self.value.load(Ordering::Relaxed);
            let current_f64 = f64::from_bits(current);
            let new = (current_f64 + delta).to_bits();
            if self
                .value
                .compare_exchange_weak(current, new, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }

    /// Subtract from the gauge value.
    pub fn sub(&self, delta: f64) {
        self.add(-delta);
    }

    /// Get the current value.
    pub fn get(&self) -> f64 {
        f64::from_bits(self.value.load(Ordering::Relaxed))
    }

    /// Get the gauge name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the gauge description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Get a copy of the labels.
    pub fn labels(&self) -> Labels {
        self.labels.read().unwrap().clone()
    }

    /// Add a label.
    pub fn add_label(&self, key: impl Into<String>, value: impl Into<String>) {
        self.labels.write().unwrap().insert(key.into(), value.into());
    }
}

/// A histogram metric for recording distributions of values.
#[derive(Debug)]
pub struct Histogram {
    name: String,
    description: String,
    buckets: Vec<f64>,
    bucket_counts: Vec<AtomicU64>,
    sum: AtomicU64, // Stored as bits of f64
    count: AtomicU64,
    labels: RwLock<Labels>,
}

impl Histogram {
    /// Default bucket boundaries for latency histograms (in seconds).
    pub const DEFAULT_LATENCY_BUCKETS: [f64; 11] = [
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
    ];

    /// Default bucket boundaries for size histograms (in bytes).
    pub const DEFAULT_SIZE_BUCKETS: [f64; 10] = [
        100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0, 100000000.0, 1000000000.0,
        10000000000.0, 100000000000.0,
    ];

    /// Bucket boundaries for FPS histograms.
    pub const FPS_BUCKETS: [f64; 10] = [15.0, 24.0, 25.0, 30.0, 50.0, 60.0, 90.0, 120.0, 144.0, 240.0];

    /// Create a new histogram with default latency buckets.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self::with_buckets(name, description, &Self::DEFAULT_LATENCY_BUCKETS)
    }

    /// Create a new histogram with custom buckets.
    pub fn with_buckets(
        name: impl Into<String>,
        description: impl Into<String>,
        buckets: &[f64],
    ) -> Self {
        let bucket_counts: Vec<AtomicU64> = (0..=buckets.len())
            .map(|_| AtomicU64::new(0))
            .collect();

        Self {
            name: name.into(),
            description: description.into(),
            buckets: buckets.to_vec(),
            bucket_counts,
            sum: AtomicU64::new(0.0_f64.to_bits()),
            count: AtomicU64::new(0),
            labels: RwLock::new(HashMap::new()),
        }
    }

    /// Create a new histogram with labels.
    pub fn with_labels(
        name: impl Into<String>,
        description: impl Into<String>,
        buckets: &[f64],
        labels: Labels,
    ) -> Self {
        let bucket_counts: Vec<AtomicU64> = (0..=buckets.len())
            .map(|_| AtomicU64::new(0))
            .collect();

        Self {
            name: name.into(),
            description: description.into(),
            buckets: buckets.to_vec(),
            bucket_counts,
            sum: AtomicU64::new(0.0_f64.to_bits()),
            count: AtomicU64::new(0),
            labels: RwLock::new(labels),
        }
    }

    /// Record a value in the histogram.
    pub fn observe(&self, value: f64) {
        // Find the bucket
        let bucket_idx = self
            .buckets
            .iter()
            .position(|&b| value <= b)
            .unwrap_or(self.buckets.len());

        // Increment the bucket and all buckets above it (cumulative)
        for i in bucket_idx..self.bucket_counts.len() {
            self.bucket_counts[i].fetch_add(1, Ordering::Relaxed);
        }

        // Update sum
        loop {
            let current = self.sum.load(Ordering::Relaxed);
            let current_f64 = f64::from_bits(current);
            let new = (current_f64 + value).to_bits();
            if self
                .sum
                .compare_exchange_weak(current, new, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }

        // Update count
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the bucket boundaries.
    pub fn buckets(&self) -> &[f64] {
        &self.buckets
    }

    /// Get the bucket counts.
    pub fn bucket_counts(&self) -> Vec<u64> {
        self.bucket_counts
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect()
    }

    /// Get the sum of all observed values.
    pub fn sum(&self) -> f64 {
        f64::from_bits(self.sum.load(Ordering::Relaxed))
    }

    /// Get the count of observations.
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get the histogram name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the histogram description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Get a copy of the labels.
    pub fn labels(&self) -> Labels {
        self.labels.read().unwrap().clone()
    }

    /// Add a label.
    pub fn add_label(&self, key: impl Into<String>, value: impl Into<String>) {
        self.labels.write().unwrap().insert(key.into(), value.into());
    }

    /// Calculate the mean of observed values.
    pub fn mean(&self) -> f64 {
        let count = self.count();
        if count == 0 {
            return 0.0;
        }
        self.sum() / count as f64
    }
}

/// Timer for measuring durations and recording them to a histogram.
pub struct Timer<'a> {
    histogram: &'a Histogram,
    start: Instant,
}

impl<'a> Timer<'a> {
    /// Create a new timer.
    pub fn new(histogram: &'a Histogram) -> Self {
        Self {
            histogram,
            start: Instant::now(),
        }
    }

    /// Stop the timer and record the duration.
    pub fn observe_duration(self) -> Duration {
        let duration = self.start.elapsed();
        self.histogram.observe(duration.as_secs_f64());
        duration
    }
}

impl<'a> Drop for Timer<'a> {
    fn drop(&mut self) {
        // Record duration if not already observed
        let duration = self.start.elapsed();
        self.histogram.observe(duration.as_secs_f64());
    }
}

/// Transcode-specific metrics collection.
#[derive(Debug)]
pub struct TranscodeMetrics {
    /// Jobs started counter.
    pub jobs_started: Counter,
    /// Jobs completed counter.
    pub jobs_completed: Counter,
    /// Jobs failed counter.
    pub jobs_failed: Counter,
    /// Current jobs in progress.
    pub jobs_in_progress: Gauge,
    /// Job queue depth.
    pub queue_depth: Gauge,

    /// Frames processed counter.
    pub frames_processed: Counter,
    /// Current FPS gauge.
    pub current_fps: Gauge,
    /// FPS histogram.
    pub fps_histogram: Histogram,

    /// Current bitrate gauge (bits per second).
    pub current_bitrate: Gauge,
    /// Bitrate histogram.
    pub bitrate_histogram: Histogram,

    /// Transcode progress percentage (0-100).
    pub progress_percentage: Gauge,
    /// Transcode duration histogram.
    pub transcode_duration: Histogram,

    /// Bytes read counter.
    pub bytes_read: Counter,
    /// Bytes written counter.
    pub bytes_written: Counter,
}

impl TranscodeMetrics {
    /// Create a new TranscodeMetrics instance.
    pub fn new() -> Self {
        Self {
            jobs_started: Counter::new(
                "transcode_jobs_started_total",
                "Total number of transcode jobs started",
            ),
            jobs_completed: Counter::new(
                "transcode_jobs_completed_total",
                "Total number of transcode jobs completed successfully",
            ),
            jobs_failed: Counter::new(
                "transcode_jobs_failed_total",
                "Total number of transcode jobs that failed",
            ),
            jobs_in_progress: Gauge::new(
                "transcode_jobs_in_progress",
                "Number of transcode jobs currently in progress",
            ),
            queue_depth: Gauge::new(
                "transcode_queue_depth",
                "Number of jobs waiting in the queue",
            ),
            frames_processed: Counter::new(
                "transcode_frames_processed_total",
                "Total number of frames processed",
            ),
            current_fps: Gauge::new(
                "transcode_current_fps",
                "Current frames per second being processed",
            ),
            fps_histogram: Histogram::with_buckets(
                "transcode_fps",
                "Distribution of FPS during transcoding",
                &Histogram::FPS_BUCKETS,
            ),
            current_bitrate: Gauge::new(
                "transcode_current_bitrate_bps",
                "Current bitrate in bits per second",
            ),
            bitrate_histogram: Histogram::with_buckets(
                "transcode_bitrate_bps",
                "Distribution of bitrates",
                &[
                    500_000.0,
                    1_000_000.0,
                    2_000_000.0,
                    5_000_000.0,
                    10_000_000.0,
                    20_000_000.0,
                    50_000_000.0,
                ],
            ),
            progress_percentage: Gauge::new(
                "transcode_progress_percentage",
                "Current transcode progress (0-100)",
            ),
            transcode_duration: Histogram::with_buckets(
                "transcode_duration_seconds",
                "Histogram of transcode job durations",
                &[1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0],
            ),
            bytes_read: Counter::new(
                "transcode_bytes_read_total",
                "Total bytes read from source",
            ),
            bytes_written: Counter::new(
                "transcode_bytes_written_total",
                "Total bytes written to destination",
            ),
        }
    }

    /// Record job start.
    pub fn record_job_start(&self) {
        self.jobs_started.inc();
        self.jobs_in_progress.inc();
    }

    /// Record job completion.
    pub fn record_job_complete(&self, duration_secs: f64) {
        self.jobs_completed.inc();
        self.jobs_in_progress.dec();
        self.transcode_duration.observe(duration_secs);
    }

    /// Record job failure.
    pub fn record_job_failure(&self) {
        self.jobs_failed.inc();
        self.jobs_in_progress.dec();
    }

    /// Update FPS.
    pub fn update_fps(&self, fps: f64) {
        self.current_fps.set(fps);
        self.fps_histogram.observe(fps);
    }

    /// Update bitrate.
    pub fn update_bitrate(&self, bitrate_bps: f64) {
        self.current_bitrate.set(bitrate_bps);
        self.bitrate_histogram.observe(bitrate_bps);
    }

    /// Update progress.
    pub fn update_progress(&self, percentage: f64) {
        self.progress_percentage.set(percentage.clamp(0.0, 100.0));
    }

    /// Update queue depth.
    pub fn update_queue_depth(&self, depth: i64) {
        self.queue_depth.set(depth as f64);
    }
}

impl Default for TranscodeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Codec performance metrics.
#[derive(Debug)]
pub struct CodecMetrics {
    /// Encode time histogram.
    pub encode_time: Histogram,
    /// Decode time histogram.
    pub decode_time: Histogram,
    /// Frames encoded counter.
    pub frames_encoded: Counter,
    /// Frames decoded counter.
    pub frames_decoded: Counter,
    /// GOP (Group of Pictures) size histogram.
    pub gop_size: Histogram,
    /// Keyframes generated counter.
    pub keyframes_generated: Counter,
    /// Encode errors counter.
    pub encode_errors: Counter,
    /// Decode errors counter.
    pub decode_errors: Counter,
}

impl CodecMetrics {
    /// Create new codec metrics.
    pub fn new() -> Self {
        Self {
            encode_time: Histogram::with_buckets(
                "codec_encode_time_seconds",
                "Time spent encoding frames",
                &[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            ),
            decode_time: Histogram::with_buckets(
                "codec_decode_time_seconds",
                "Time spent decoding frames",
                &[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            ),
            frames_encoded: Counter::new(
                "codec_frames_encoded_total",
                "Total frames encoded",
            ),
            frames_decoded: Counter::new(
                "codec_frames_decoded_total",
                "Total frames decoded",
            ),
            gop_size: Histogram::with_buckets(
                "codec_gop_size",
                "Size of GOPs (Group of Pictures)",
                &[1.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 250.0],
            ),
            keyframes_generated: Counter::new(
                "codec_keyframes_total",
                "Total keyframes generated",
            ),
            encode_errors: Counter::new(
                "codec_encode_errors_total",
                "Total encoding errors",
            ),
            decode_errors: Counter::new(
                "codec_decode_errors_total",
                "Total decoding errors",
            ),
        }
    }

    /// Record frame encode.
    pub fn record_encode(&self, duration_secs: f64, is_keyframe: bool) {
        self.encode_time.observe(duration_secs);
        self.frames_encoded.inc();
        if is_keyframe {
            self.keyframes_generated.inc();
        }
    }

    /// Record frame decode.
    pub fn record_decode(&self, duration_secs: f64) {
        self.decode_time.observe(duration_secs);
        self.frames_decoded.inc();
    }

    /// Record GOP.
    pub fn record_gop(&self, size: u32) {
        self.gop_size.observe(size as f64);
    }

    /// Record encode error.
    pub fn record_encode_error(&self) {
        self.encode_errors.inc();
    }

    /// Record decode error.
    pub fn record_decode_error(&self) {
        self.decode_errors.inc();
    }

    /// Start an encode timer.
    pub fn start_encode_timer(&self) -> Timer<'_> {
        Timer::new(&self.encode_time)
    }

    /// Start a decode timer.
    pub fn start_decode_timer(&self) -> Timer<'_> {
        Timer::new(&self.decode_time)
    }
}

impl Default for CodecMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Resource utilization metrics.
#[derive(Debug)]
pub struct ResourceMetrics {
    /// CPU utilization percentage.
    pub cpu_usage: Gauge,
    /// Memory usage in bytes.
    pub memory_usage: Gauge,
    /// Memory available in bytes.
    pub memory_available: Gauge,
    /// GPU utilization percentage.
    pub gpu_usage: Gauge,
    /// GPU memory usage in bytes.
    pub gpu_memory_usage: Gauge,
    /// Disk read bytes per second.
    pub disk_read_rate: Gauge,
    /// Disk write bytes per second.
    pub disk_write_rate: Gauge,
    /// Network receive bytes per second.
    pub network_rx_rate: Gauge,
    /// Network transmit bytes per second.
    pub network_tx_rate: Gauge,
    /// Number of open file descriptors.
    pub open_fds: Gauge,
    /// Number of threads.
    pub thread_count: Gauge,
}

impl ResourceMetrics {
    /// Create new resource metrics.
    pub fn new() -> Self {
        Self {
            cpu_usage: Gauge::new(
                "resource_cpu_usage_percent",
                "CPU utilization percentage",
            ),
            memory_usage: Gauge::new(
                "resource_memory_usage_bytes",
                "Memory usage in bytes",
            ),
            memory_available: Gauge::new(
                "resource_memory_available_bytes",
                "Available memory in bytes",
            ),
            gpu_usage: Gauge::new(
                "resource_gpu_usage_percent",
                "GPU utilization percentage",
            ),
            gpu_memory_usage: Gauge::new(
                "resource_gpu_memory_usage_bytes",
                "GPU memory usage in bytes",
            ),
            disk_read_rate: Gauge::new(
                "resource_disk_read_bytes_per_second",
                "Disk read rate in bytes per second",
            ),
            disk_write_rate: Gauge::new(
                "resource_disk_write_bytes_per_second",
                "Disk write rate in bytes per second",
            ),
            network_rx_rate: Gauge::new(
                "resource_network_rx_bytes_per_second",
                "Network receive rate in bytes per second",
            ),
            network_tx_rate: Gauge::new(
                "resource_network_tx_bytes_per_second",
                "Network transmit rate in bytes per second",
            ),
            open_fds: Gauge::new(
                "resource_open_file_descriptors",
                "Number of open file descriptors",
            ),
            thread_count: Gauge::new(
                "resource_thread_count",
                "Number of threads",
            ),
        }
    }

    /// Update CPU usage.
    pub fn update_cpu(&self, usage_percent: f64) {
        self.cpu_usage.set(usage_percent.clamp(0.0, 100.0));
    }

    /// Update memory metrics.
    pub fn update_memory(&self, usage_bytes: u64, available_bytes: u64) {
        self.memory_usage.set(usage_bytes as f64);
        self.memory_available.set(available_bytes as f64);
    }

    /// Update GPU metrics.
    pub fn update_gpu(&self, usage_percent: f64, memory_bytes: u64) {
        self.gpu_usage.set(usage_percent.clamp(0.0, 100.0));
        self.gpu_memory_usage.set(memory_bytes as f64);
    }

    /// Update disk I/O rates.
    pub fn update_disk_io(&self, read_rate: f64, write_rate: f64) {
        self.disk_read_rate.set(read_rate);
        self.disk_write_rate.set(write_rate);
    }

    /// Update network I/O rates.
    pub fn update_network_io(&self, rx_rate: f64, tx_rate: f64) {
        self.network_rx_rate.set(rx_rate);
        self.network_tx_rate.set(tx_rate);
    }

    /// Update thread count.
    pub fn update_thread_count(&self, count: u32) {
        self.thread_count.set(count as f64);
    }

    /// Update open file descriptors.
    pub fn update_open_fds(&self, count: u32) {
        self.open_fds.set(count as f64);
    }
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Error tracking metrics.
#[derive(Debug)]
pub struct ErrorMetrics {
    /// Total errors by type.
    errors_by_type: RwLock<HashMap<String, Counter>>,
    /// Total errors counter.
    pub total_errors: Counter,
    /// Error rate (errors per second).
    pub error_rate: Gauge,
    /// Last error timestamp.
    last_error_time: RwLock<Option<Instant>>,
}

impl ErrorMetrics {
    /// Create new error metrics.
    pub fn new() -> Self {
        Self {
            errors_by_type: RwLock::new(HashMap::new()),
            total_errors: Counter::new(
                "errors_total",
                "Total number of errors",
            ),
            error_rate: Gauge::new(
                "error_rate_per_second",
                "Current error rate per second",
            ),
            last_error_time: RwLock::new(None),
        }
    }

    /// Record an error.
    pub fn record_error(&self, error_type: &str) {
        self.total_errors.inc();

        // Update error by type
        {
            let mut errors = self.errors_by_type.write().unwrap();
            errors
                .entry(error_type.to_string())
                .or_insert_with(|| {
                    Counter::new(
                        format!("errors_{}_total", error_type),
                        format!("Total {} errors", error_type),
                    )
                })
                .inc();
        }

        // Update last error time
        *self.last_error_time.write().unwrap() = Some(Instant::now());
    }

    /// Get error count by type.
    pub fn get_error_count(&self, error_type: &str) -> u64 {
        self.errors_by_type
            .read()
            .unwrap()
            .get(error_type)
            .map(|c| c.get())
            .unwrap_or(0)
    }

    /// Get all error types and counts.
    pub fn get_all_errors(&self) -> HashMap<String, u64> {
        self.errors_by_type
            .read()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.get()))
            .collect()
    }

    /// Update error rate.
    pub fn update_error_rate(&self, rate: f64) {
        self.error_rate.set(rate);
    }

    /// Get time since last error.
    pub fn time_since_last_error(&self) -> Option<Duration> {
        self.last_error_time.read().unwrap().map(|t| t.elapsed())
    }
}

impl Default for ErrorMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Central metrics registry.
#[derive(Debug)]
pub struct MetricsRegistry {
    /// Transcode-specific metrics.
    pub transcode: Arc<TranscodeMetrics>,
    /// Codec performance metrics.
    pub codec: Arc<CodecMetrics>,
    /// Resource utilization metrics.
    pub resource: Arc<ResourceMetrics>,
    /// Error tracking metrics.
    pub errors: Arc<ErrorMetrics>,
    /// Custom counters.
    custom_counters: RwLock<HashMap<String, Arc<Counter>>>,
    /// Custom gauges.
    custom_gauges: RwLock<HashMap<String, Arc<Gauge>>>,
    /// Custom histograms.
    custom_histograms: RwLock<HashMap<String, Arc<Histogram>>>,
}

impl MetricsRegistry {
    /// Create a new metrics registry.
    pub fn new() -> Self {
        Self {
            transcode: Arc::new(TranscodeMetrics::new()),
            codec: Arc::new(CodecMetrics::new()),
            resource: Arc::new(ResourceMetrics::new()),
            errors: Arc::new(ErrorMetrics::new()),
            custom_counters: RwLock::new(HashMap::new()),
            custom_gauges: RwLock::new(HashMap::new()),
            custom_histograms: RwLock::new(HashMap::new()),
        }
    }

    /// Register a custom counter.
    pub fn register_counter(
        &self,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Result<Arc<Counter>> {
        let name = name.into();
        let mut counters = self.custom_counters.write().unwrap();

        if counters.contains_key(&name) {
            return Err(MetricsError::Creation {
                name: name.clone(),
                reason: "Counter already exists".to_string(),
            }
            .into());
        }

        let counter = Arc::new(Counter::new(&name, description));
        counters.insert(name, counter.clone());
        Ok(counter)
    }

    /// Register a custom gauge.
    pub fn register_gauge(
        &self,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Result<Arc<Gauge>> {
        let name = name.into();
        let mut gauges = self.custom_gauges.write().unwrap();

        if gauges.contains_key(&name) {
            return Err(MetricsError::Creation {
                name: name.clone(),
                reason: "Gauge already exists".to_string(),
            }
            .into());
        }

        let gauge = Arc::new(Gauge::new(&name, description));
        gauges.insert(name, gauge.clone());
        Ok(gauge)
    }

    /// Register a custom histogram.
    pub fn register_histogram(
        &self,
        name: impl Into<String>,
        description: impl Into<String>,
        buckets: &[f64],
    ) -> Result<Arc<Histogram>> {
        let name = name.into();
        let mut histograms = self.custom_histograms.write().unwrap();

        if histograms.contains_key(&name) {
            return Err(MetricsError::Creation {
                name: name.clone(),
                reason: "Histogram already exists".to_string(),
            }
            .into());
        }

        let histogram = Arc::new(Histogram::with_buckets(&name, description, buckets));
        histograms.insert(name, histogram.clone());
        Ok(histogram)
    }

    /// Get a custom counter by name.
    pub fn get_counter(&self, name: &str) -> Option<Arc<Counter>> {
        self.custom_counters.read().unwrap().get(name).cloned()
    }

    /// Get a custom gauge by name.
    pub fn get_gauge(&self, name: &str) -> Option<Arc<Gauge>> {
        self.custom_gauges.read().unwrap().get(name).cloned()
    }

    /// Get a custom histogram by name.
    pub fn get_histogram(&self, name: &str) -> Option<Arc<Histogram>> {
        self.custom_histograms.read().unwrap().get(name).cloned()
    }

    /// Get all custom counters.
    pub fn custom_counters(&self) -> Vec<Arc<Counter>> {
        self.custom_counters.read().unwrap().values().cloned().collect()
    }

    /// Get all custom gauges.
    pub fn custom_gauges(&self) -> Vec<Arc<Gauge>> {
        self.custom_gauges.read().unwrap().values().cloned().collect()
    }

    /// Get all custom histograms.
    pub fn custom_histograms(&self) -> Vec<Arc<Histogram>> {
        self.custom_histograms.read().unwrap().values().cloned().collect()
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter() {
        let counter = Counter::new("test_counter", "A test counter");
        assert_eq!(counter.get(), 0);

        counter.inc();
        assert_eq!(counter.get(), 1);

        counter.inc_by(5);
        assert_eq!(counter.get(), 6);
    }

    #[test]
    fn test_counter_with_labels() {
        let mut labels = HashMap::new();
        labels.insert("job_id".to_string(), "123".to_string());

        let counter = Counter::with_labels("test_counter", "A test counter", labels);
        counter.add_label("codec", "h264");

        let labels = counter.labels();
        assert_eq!(labels.get("job_id"), Some(&"123".to_string()));
        assert_eq!(labels.get("codec"), Some(&"h264".to_string()));
    }

    #[test]
    fn test_gauge() {
        let gauge = Gauge::new("test_gauge", "A test gauge");
        assert_eq!(gauge.get(), 0.0);

        gauge.set(42.5);
        assert_eq!(gauge.get(), 42.5);

        gauge.inc();
        assert_eq!(gauge.get(), 43.5);

        gauge.dec();
        assert_eq!(gauge.get(), 42.5);

        gauge.add(7.5);
        assert_eq!(gauge.get(), 50.0);

        gauge.sub(10.0);
        assert_eq!(gauge.get(), 40.0);
    }

    #[test]
    fn test_histogram() {
        let histogram = Histogram::with_buckets(
            "test_histogram",
            "A test histogram",
            &[1.0, 5.0, 10.0],
        );

        histogram.observe(0.5);
        histogram.observe(3.0);
        histogram.observe(7.0);
        histogram.observe(15.0);

        assert_eq!(histogram.count(), 4);
        assert!((histogram.sum() - 25.5).abs() < 0.001);
        assert!((histogram.mean() - 6.375).abs() < 0.001);

        let counts = histogram.bucket_counts();
        assert_eq!(counts[0], 1); // <= 1.0
        assert_eq!(counts[1], 2); // <= 5.0
        assert_eq!(counts[2], 3); // <= 10.0
        assert_eq!(counts[3], 4); // > 10.0 (inf bucket)
    }

    #[test]
    fn test_transcode_metrics() {
        let metrics = TranscodeMetrics::new();

        metrics.record_job_start();
        assert_eq!(metrics.jobs_started.get(), 1);
        assert_eq!(metrics.jobs_in_progress.get(), 1.0);

        metrics.update_fps(30.0);
        assert_eq!(metrics.current_fps.get(), 30.0);

        metrics.update_bitrate(5_000_000.0);
        assert_eq!(metrics.current_bitrate.get(), 5_000_000.0);

        metrics.update_progress(50.0);
        assert_eq!(metrics.progress_percentage.get(), 50.0);

        // Test progress clamping
        metrics.update_progress(150.0);
        assert_eq!(metrics.progress_percentage.get(), 100.0);

        metrics.record_job_complete(60.0);
        assert_eq!(metrics.jobs_completed.get(), 1);
        assert_eq!(metrics.jobs_in_progress.get(), 0.0);
    }

    #[test]
    fn test_codec_metrics() {
        let metrics = CodecMetrics::new();

        metrics.record_encode(0.033, true);
        assert_eq!(metrics.frames_encoded.get(), 1);
        assert_eq!(metrics.keyframes_generated.get(), 1);

        metrics.record_encode(0.016, false);
        assert_eq!(metrics.frames_encoded.get(), 2);
        assert_eq!(metrics.keyframes_generated.get(), 1);

        metrics.record_decode(0.010);
        assert_eq!(metrics.frames_decoded.get(), 1);

        metrics.record_gop(30);
        assert_eq!(metrics.gop_size.count(), 1);

        metrics.record_encode_error();
        assert_eq!(metrics.encode_errors.get(), 1);
    }

    #[test]
    fn test_resource_metrics() {
        let metrics = ResourceMetrics::new();

        metrics.update_cpu(75.5);
        assert_eq!(metrics.cpu_usage.get(), 75.5);

        // Test CPU clamping
        metrics.update_cpu(150.0);
        assert_eq!(metrics.cpu_usage.get(), 100.0);

        metrics.update_memory(4_000_000_000, 8_000_000_000);
        assert_eq!(metrics.memory_usage.get(), 4_000_000_000.0);
        assert_eq!(metrics.memory_available.get(), 8_000_000_000.0);

        metrics.update_gpu(50.0, 2_000_000_000);
        assert_eq!(metrics.gpu_usage.get(), 50.0);
        assert_eq!(metrics.gpu_memory_usage.get(), 2_000_000_000.0);
    }

    #[test]
    fn test_error_metrics() {
        let metrics = ErrorMetrics::new();

        metrics.record_error("decode");
        metrics.record_error("decode");
        metrics.record_error("encode");

        assert_eq!(metrics.total_errors.get(), 3);
        assert_eq!(metrics.get_error_count("decode"), 2);
        assert_eq!(metrics.get_error_count("encode"), 1);
        assert_eq!(metrics.get_error_count("unknown"), 0);

        let all_errors = metrics.get_all_errors();
        assert_eq!(all_errors.len(), 2);
    }

    #[test]
    fn test_metrics_registry() {
        let registry = MetricsRegistry::new();

        // Test custom counter
        let counter = registry
            .register_counter("custom_counter", "A custom counter")
            .unwrap();
        counter.inc();
        assert_eq!(counter.get(), 1);

        // Test duplicate registration
        let result = registry.register_counter("custom_counter", "Duplicate");
        assert!(result.is_err());

        // Test custom gauge
        let gauge = registry
            .register_gauge("custom_gauge", "A custom gauge")
            .unwrap();
        gauge.set(42.0);
        assert_eq!(gauge.get(), 42.0);

        // Test custom histogram
        let histogram = registry
            .register_histogram("custom_histogram", "A custom histogram", &[1.0, 5.0])
            .unwrap();
        histogram.observe(3.0);
        assert_eq!(histogram.count(), 1);

        // Test retrieval
        assert!(registry.get_counter("custom_counter").is_some());
        assert!(registry.get_gauge("custom_gauge").is_some());
        assert!(registry.get_histogram("custom_histogram").is_some());
        assert!(registry.get_counter("nonexistent").is_none());
    }

    #[test]
    fn test_histogram_default_buckets() {
        assert_eq!(Histogram::DEFAULT_LATENCY_BUCKETS.len(), 11);
        assert_eq!(Histogram::DEFAULT_SIZE_BUCKETS.len(), 10);
        assert_eq!(Histogram::FPS_BUCKETS.len(), 10);
    }
}
