//! Prometheus metrics exporter for the transcode system.
//!
//! This module provides a Prometheus-compatible `/metrics` endpoint that exports
//! all collected metrics in the Prometheus text exposition format.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

use crate::error::{PrometheusError, Result};
use crate::metrics::{
    Counter, Gauge, Histogram, MetricsRegistry, TranscodeMetrics, CodecMetrics,
    ResourceMetrics, ErrorMetrics, Labels,
};

/// Prometheus metric type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    /// Counter metric.
    Counter,
    /// Gauge metric.
    Gauge,
    /// Histogram metric.
    Histogram,
    /// Summary metric.
    Summary,
    /// Untyped metric.
    Untyped,
}

impl MetricType {
    /// Get the type name.
    pub fn as_str(&self) -> &'static str {
        match self {
            MetricType::Counter => "counter",
            MetricType::Gauge => "gauge",
            MetricType::Histogram => "histogram",
            MetricType::Summary => "summary",
            MetricType::Untyped => "untyped",
        }
    }
}

/// A Prometheus metric family.
#[derive(Debug, Clone)]
pub struct MetricFamily {
    /// Metric name.
    pub name: String,
    /// Metric help/description.
    pub help: String,
    /// Metric type.
    pub metric_type: MetricType,
    /// Metric samples.
    pub samples: Vec<MetricSample>,
}

impl MetricFamily {
    /// Create a new metric family.
    pub fn new(
        name: impl Into<String>,
        help: impl Into<String>,
        metric_type: MetricType,
    ) -> Self {
        Self {
            name: name.into(),
            help: help.into(),
            metric_type,
            samples: Vec::new(),
        }
    }

    /// Add a sample.
    pub fn add_sample(&mut self, sample: MetricSample) {
        self.samples.push(sample);
    }

    /// Format to Prometheus text format.
    pub fn to_prometheus_text(&self) -> String {
        let mut output = String::new();

        // HELP line
        output.push_str(&format!("# HELP {} {}\n", self.name, escape_help(&self.help)));

        // TYPE line
        output.push_str(&format!("# TYPE {} {}\n", self.name, self.metric_type.as_str()));

        // Sample lines
        for sample in &self.samples {
            output.push_str(&sample.to_prometheus_text(&self.name));
            output.push('\n');
        }

        output
    }
}

/// A single metric sample.
#[derive(Debug, Clone)]
pub struct MetricSample {
    /// Metric suffix (e.g., "_total", "_bucket", "_sum", "_count").
    pub suffix: String,
    /// Label key-value pairs.
    pub labels: Labels,
    /// Metric value.
    pub value: f64,
}

impl MetricSample {
    /// Create a new sample.
    pub fn new(value: f64) -> Self {
        Self {
            suffix: String::new(),
            labels: HashMap::new(),
            value,
        }
    }

    /// Set the suffix.
    pub fn suffix(mut self, suffix: impl Into<String>) -> Self {
        self.suffix = suffix.into();
        self
    }

    /// Add a label.
    pub fn label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// Add multiple labels.
    pub fn labels(mut self, labels: Labels) -> Self {
        for (k, v) in labels {
            self.labels.insert(k, v);
        }
        self
    }

    /// Format to Prometheus text format.
    pub fn to_prometheus_text(&self, name: &str) -> String {
        let metric_name = format!("{}{}", name, self.suffix);

        if self.labels.is_empty() {
            format!("{} {}", metric_name, format_value(self.value))
        } else {
            let labels_str: Vec<String> = self
                .labels
                .iter()
                .map(|(k, v)| format!("{}=\"{}\"", k, escape_label_value(v)))
                .collect();
            format!(
                "{}{{{}}} {}",
                metric_name,
                labels_str.join(","),
                format_value(self.value)
            )
        }
    }
}

/// Format a float value for Prometheus output.
fn format_value(value: f64) -> String {
    if value.is_nan() {
        "NaN".to_string()
    } else if value.is_infinite() {
        if value.is_sign_positive() {
            "+Inf".to_string()
        } else {
            "-Inf".to_string()
        }
    } else if value == value.floor() && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        format!("{}", value)
    }
}

/// Escape help text for Prometheus format.
fn escape_help(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\n', "\\n")
}

/// Escape label value for Prometheus format.
fn escape_label_value(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// Prometheus text encoder.
pub struct PrometheusEncoder;

impl PrometheusEncoder {
    /// Encode a counter to metric families.
    pub fn encode_counter(counter: &Counter) -> MetricFamily {
        let mut family = MetricFamily::new(
            counter.name(),
            counter.description(),
            MetricType::Counter,
        );

        let mut sample = MetricSample::new(counter.get() as f64);
        let labels = counter.labels();
        if !labels.is_empty() {
            sample = sample.labels(labels);
        }
        family.add_sample(sample);

        family
    }

    /// Encode a gauge to metric families.
    pub fn encode_gauge(gauge: &Gauge) -> MetricFamily {
        let mut family = MetricFamily::new(
            gauge.name(),
            gauge.description(),
            MetricType::Gauge,
        );

        let mut sample = MetricSample::new(gauge.get());
        let labels = gauge.labels();
        if !labels.is_empty() {
            sample = sample.labels(labels);
        }
        family.add_sample(sample);

        family
    }

    /// Encode a histogram to metric families.
    pub fn encode_histogram(histogram: &Histogram) -> MetricFamily {
        let mut family = MetricFamily::new(
            histogram.name(),
            histogram.description(),
            MetricType::Histogram,
        );

        let buckets = histogram.buckets();
        let bucket_counts = histogram.bucket_counts();
        let base_labels = histogram.labels();

        // Bucket samples
        for (&upper_bound, &count) in buckets.iter().zip(bucket_counts.iter()) {
            let mut sample = MetricSample::new(count as f64)
                .suffix("_bucket")
                .label("le", format_value(upper_bound));
            if !base_labels.is_empty() {
                sample = sample.labels(base_labels.clone());
            }
            family.add_sample(sample);
        }

        // +Inf bucket
        let inf_count = bucket_counts.last().copied().unwrap_or(0);
        let mut inf_sample = MetricSample::new(inf_count as f64)
            .suffix("_bucket")
            .label("le", "+Inf");
        if !base_labels.is_empty() {
            inf_sample = inf_sample.labels(base_labels.clone());
        }
        family.add_sample(inf_sample);

        // Sum sample
        let mut sum_sample = MetricSample::new(histogram.sum()).suffix("_sum");
        if !base_labels.is_empty() {
            sum_sample = sum_sample.labels(base_labels.clone());
        }
        family.add_sample(sum_sample);

        // Count sample
        let mut count_sample = MetricSample::new(histogram.count() as f64).suffix("_count");
        if !base_labels.is_empty() {
            count_sample = count_sample.labels(base_labels);
        }
        family.add_sample(count_sample);

        family
    }

    /// Encode transcode metrics to Prometheus text format.
    pub fn encode_transcode_metrics(metrics: &TranscodeMetrics) -> String {
        let mut output = String::new();

        // Job counters
        output.push_str(&Self::encode_counter(&metrics.jobs_started).to_prometheus_text());
        output.push_str(&Self::encode_counter(&metrics.jobs_completed).to_prometheus_text());
        output.push_str(&Self::encode_counter(&metrics.jobs_failed).to_prometheus_text());

        // Job gauges
        output.push_str(&Self::encode_gauge(&metrics.jobs_in_progress).to_prometheus_text());
        output.push_str(&Self::encode_gauge(&metrics.queue_depth).to_prometheus_text());

        // Frame counters
        output.push_str(&Self::encode_counter(&metrics.frames_processed).to_prometheus_text());

        // FPS metrics
        output.push_str(&Self::encode_gauge(&metrics.current_fps).to_prometheus_text());
        output.push_str(&Self::encode_histogram(&metrics.fps_histogram).to_prometheus_text());

        // Bitrate metrics
        output.push_str(&Self::encode_gauge(&metrics.current_bitrate).to_prometheus_text());
        output.push_str(&Self::encode_histogram(&metrics.bitrate_histogram).to_prometheus_text());

        // Progress
        output.push_str(&Self::encode_gauge(&metrics.progress_percentage).to_prometheus_text());

        // Duration histogram
        output.push_str(&Self::encode_histogram(&metrics.transcode_duration).to_prometheus_text());

        // Byte counters
        output.push_str(&Self::encode_counter(&metrics.bytes_read).to_prometheus_text());
        output.push_str(&Self::encode_counter(&metrics.bytes_written).to_prometheus_text());

        output
    }

    /// Encode codec metrics to Prometheus text format.
    pub fn encode_codec_metrics(metrics: &CodecMetrics) -> String {
        let mut output = String::new();

        output.push_str(&Self::encode_histogram(&metrics.encode_time).to_prometheus_text());
        output.push_str(&Self::encode_histogram(&metrics.decode_time).to_prometheus_text());
        output.push_str(&Self::encode_counter(&metrics.frames_encoded).to_prometheus_text());
        output.push_str(&Self::encode_counter(&metrics.frames_decoded).to_prometheus_text());
        output.push_str(&Self::encode_histogram(&metrics.gop_size).to_prometheus_text());
        output.push_str(&Self::encode_counter(&metrics.keyframes_generated).to_prometheus_text());
        output.push_str(&Self::encode_counter(&metrics.encode_errors).to_prometheus_text());
        output.push_str(&Self::encode_counter(&metrics.decode_errors).to_prometheus_text());

        output
    }

    /// Encode resource metrics to Prometheus text format.
    pub fn encode_resource_metrics(metrics: &ResourceMetrics) -> String {
        let mut output = String::new();

        output.push_str(&Self::encode_gauge(&metrics.cpu_usage).to_prometheus_text());
        output.push_str(&Self::encode_gauge(&metrics.memory_usage).to_prometheus_text());
        output.push_str(&Self::encode_gauge(&metrics.memory_available).to_prometheus_text());
        output.push_str(&Self::encode_gauge(&metrics.gpu_usage).to_prometheus_text());
        output.push_str(&Self::encode_gauge(&metrics.gpu_memory_usage).to_prometheus_text());
        output.push_str(&Self::encode_gauge(&metrics.disk_read_rate).to_prometheus_text());
        output.push_str(&Self::encode_gauge(&metrics.disk_write_rate).to_prometheus_text());
        output.push_str(&Self::encode_gauge(&metrics.network_rx_rate).to_prometheus_text());
        output.push_str(&Self::encode_gauge(&metrics.network_tx_rate).to_prometheus_text());
        output.push_str(&Self::encode_gauge(&metrics.open_fds).to_prometheus_text());
        output.push_str(&Self::encode_gauge(&metrics.thread_count).to_prometheus_text());

        output
    }

    /// Encode error metrics to Prometheus text format.
    pub fn encode_error_metrics(metrics: &ErrorMetrics) -> String {
        let mut output = String::new();

        output.push_str(&Self::encode_counter(&metrics.total_errors).to_prometheus_text());
        output.push_str(&Self::encode_gauge(&metrics.error_rate).to_prometheus_text());

        // Error counts by type
        let all_errors = metrics.get_all_errors();
        if !all_errors.is_empty() {
            let mut family = MetricFamily::new(
                "errors_by_type_total",
                "Error counts by type",
                MetricType::Counter,
            );

            for (error_type, count) in all_errors {
                family.add_sample(
                    MetricSample::new(count as f64).label("error_type", error_type),
                );
            }

            output.push_str(&family.to_prometheus_text());
        }

        output
    }

    /// Encode all metrics from a registry to Prometheus text format.
    pub fn encode_registry(registry: &MetricsRegistry) -> String {
        let mut output = String::new();

        // Built-in metrics
        output.push_str(&Self::encode_transcode_metrics(&registry.transcode));
        output.push_str(&Self::encode_codec_metrics(&registry.codec));
        output.push_str(&Self::encode_resource_metrics(&registry.resource));
        output.push_str(&Self::encode_error_metrics(&registry.errors));

        // Custom counters
        for counter in registry.custom_counters() {
            output.push_str(&Self::encode_counter(&counter).to_prometheus_text());
        }

        // Custom gauges
        for gauge in registry.custom_gauges() {
            output.push_str(&Self::encode_gauge(&gauge).to_prometheus_text());
        }

        // Custom histograms
        for histogram in registry.custom_histograms() {
            output.push_str(&Self::encode_histogram(&histogram).to_prometheus_text());
        }

        output
    }
}

/// Configuration for the Prometheus exporter server.
#[derive(Debug, Clone)]
pub struct PrometheusExporterConfig {
    /// Address to bind to.
    pub address: String,
    /// Port to listen on.
    pub port: u16,
    /// Metrics path.
    pub metrics_path: String,
    /// Enable gzip compression.
    pub enable_compression: bool,
}

impl PrometheusExporterConfig {
    /// Create a new configuration.
    pub fn new() -> Self {
        Self {
            address: "0.0.0.0".to_string(),
            port: 9090,
            metrics_path: "/metrics".to_string(),
            enable_compression: false,
        }
    }

    /// Set the address.
    pub fn address(mut self, address: impl Into<String>) -> Self {
        self.address = address.into();
        self
    }

    /// Set the port.
    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Set the metrics path.
    pub fn metrics_path(mut self, path: impl Into<String>) -> Self {
        self.metrics_path = path.into();
        self
    }

    /// Get the bind address.
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.address, self.port)
    }
}

impl Default for PrometheusExporterConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Prometheus exporter server.
pub struct PrometheusExporter {
    config: PrometheusExporterConfig,
    registry: Arc<MetricsRegistry>,
    running: Arc<RwLock<bool>>,
    listener: Option<TcpListener>,
}

impl PrometheusExporter {
    /// Create a new Prometheus exporter.
    pub fn new(config: PrometheusExporterConfig, registry: Arc<MetricsRegistry>) -> Self {
        Self {
            config,
            registry,
            running: Arc::new(RwLock::new(false)),
            listener: None,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &PrometheusExporterConfig {
        &self.config
    }

    /// Check if the exporter is running.
    pub fn is_running(&self) -> bool {
        *self.running.read().unwrap()
    }

    /// Start the exporter server.
    pub fn start(&mut self) -> Result<()> {
        if self.is_running() {
            return Ok(());
        }

        let addr = self.config.bind_address();
        let listener = TcpListener::bind(&addr).map_err(|e| PrometheusError::Bind {
            address: addr.clone(),
            reason: e.to_string(),
        })?;

        listener
            .set_nonblocking(true)
            .map_err(|e| PrometheusError::Server {
                reason: e.to_string(),
            })?;

        self.listener = Some(listener);
        *self.running.write().unwrap() = true;

        Ok(())
    }

    /// Stop the exporter server.
    pub fn stop(&mut self) {
        *self.running.write().unwrap() = false;
        self.listener = None;
    }

    /// Process incoming connections (non-blocking).
    pub fn poll(&self) -> Result<()> {
        if !self.is_running() {
            return Ok(());
        }

        let listener = match &self.listener {
            Some(l) => l,
            None => return Ok(()),
        };

        // Accept connections
        match listener.accept() {
            Ok((stream, _addr)) => {
                self.handle_connection(stream)?;
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // No connections waiting
            }
            Err(e) => {
                return Err(PrometheusError::Server {
                    reason: e.to_string(),
                }
                .into());
            }
        }

        Ok(())
    }

    /// Handle a client connection.
    fn handle_connection(&self, mut stream: TcpStream) -> Result<()> {
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .ok();
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .ok();

        // Read request
        let mut buffer = [0u8; 4096];
        let n = stream.read(&mut buffer).map_err(|e| PrometheusError::Server {
            reason: e.to_string(),
        })?;

        let request = String::from_utf8_lossy(&buffer[..n]);

        // Parse request line
        let request_line = request.lines().next().unwrap_or("");
        let parts: Vec<&str> = request_line.split_whitespace().collect();

        if parts.len() < 2 {
            self.send_response(&mut stream, 400, "Bad Request", "Invalid request")?;
            return Ok(());
        }

        let method = parts[0];
        let path = parts[1];

        // Handle request
        match (method, path) {
            ("GET", p) if p == self.config.metrics_path || p.starts_with(&format!("{}?", self.config.metrics_path)) => {
                let metrics = PrometheusEncoder::encode_registry(&self.registry);
                self.send_metrics_response(&mut stream, &metrics)?;
            }
            ("GET", "/") => {
                let body = format!(
                    "<html><body><h1>Transcode Metrics Exporter</h1><p><a href=\"{}\">Metrics</a></p></body></html>",
                    self.config.metrics_path
                );
                self.send_response(&mut stream, 200, "OK", &body)?;
            }
            _ => {
                self.send_response(&mut stream, 404, "Not Found", "Not Found")?;
            }
        }

        Ok(())
    }

    /// Send an HTTP response.
    fn send_response(
        &self,
        stream: &mut TcpStream,
        status: u16,
        status_text: &str,
        body: &str,
    ) -> Result<()> {
        let response = format!(
            "HTTP/1.1 {} {}\r\n\
             Content-Type: text/html; charset=utf-8\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             \r\n\
             {}",
            status,
            status_text,
            body.len(),
            body
        );

        stream
            .write_all(response.as_bytes())
            .map_err(|e| PrometheusError::Server {
                reason: e.to_string(),
            })?;

        Ok(())
    }

    /// Send a metrics response.
    fn send_metrics_response(&self, stream: &mut TcpStream, metrics: &str) -> Result<()> {
        let response = format!(
            "HTTP/1.1 200 OK\r\n\
             Content-Type: text/plain; version=0.0.4; charset=utf-8\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             \r\n\
             {}",
            metrics.len(),
            metrics
        );

        stream
            .write_all(response.as_bytes())
            .map_err(|e| PrometheusError::Server {
                reason: e.to_string(),
            })?;

        Ok(())
    }

    /// Run the server in the current thread (blocking).
    pub fn run(&mut self) -> Result<()> {
        self.start()?;

        while self.is_running() {
            self.poll()?;
            thread::sleep(Duration::from_millis(10));
        }

        Ok(())
    }

    /// Get the current metrics as Prometheus text.
    pub fn get_metrics(&self) -> String {
        PrometheusEncoder::encode_registry(&self.registry)
    }
}

/// Build a simple metrics response for use with other HTTP frameworks.
pub fn build_metrics_response(registry: &MetricsRegistry) -> (String, String) {
    let content_type = "text/plain; version=0.0.4; charset=utf-8".to_string();
    let body = PrometheusEncoder::encode_registry(registry);
    (content_type, body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_value() {
        assert_eq!(format_value(42.0), "42");
        assert_eq!(format_value(3.14), "3.14");
        assert_eq!(format_value(f64::NAN), "NaN");
        assert_eq!(format_value(f64::INFINITY), "+Inf");
        assert_eq!(format_value(f64::NEG_INFINITY), "-Inf");
    }

    #[test]
    fn test_escape_help() {
        assert_eq!(escape_help("simple help"), "simple help");
        assert_eq!(escape_help("line1\nline2"), "line1\\nline2");
        assert_eq!(escape_help("back\\slash"), "back\\\\slash");
    }

    #[test]
    fn test_escape_label_value() {
        assert_eq!(escape_label_value("simple"), "simple");
        assert_eq!(escape_label_value("with\"quote"), "with\\\"quote");
        assert_eq!(escape_label_value("with\\backslash"), "with\\\\backslash");
        assert_eq!(escape_label_value("with\nnewline"), "with\\nnewline");
    }

    #[test]
    fn test_metric_sample_simple() {
        let sample = MetricSample::new(42.0);
        assert_eq!(sample.to_prometheus_text("test_metric"), "test_metric 42");
    }

    #[test]
    fn test_metric_sample_with_labels() {
        let sample = MetricSample::new(42.0)
            .label("job", "test")
            .label("instance", "localhost");
        let text = sample.to_prometheus_text("test_metric");
        assert!(text.contains("test_metric{"));
        assert!(text.contains("job=\"test\""));
        assert!(text.contains("instance=\"localhost\""));
        assert!(text.contains("} 42"));
    }

    #[test]
    fn test_metric_sample_with_suffix() {
        let sample = MetricSample::new(100.0).suffix("_total");
        assert_eq!(sample.to_prometheus_text("requests"), "requests_total 100");
    }

    #[test]
    fn test_metric_family_counter() {
        let mut family = MetricFamily::new(
            "http_requests_total",
            "Total HTTP requests",
            MetricType::Counter,
        );
        family.add_sample(MetricSample::new(1000.0));

        let text = family.to_prometheus_text();
        assert!(text.contains("# HELP http_requests_total Total HTTP requests"));
        assert!(text.contains("# TYPE http_requests_total counter"));
        assert!(text.contains("http_requests_total 1000"));
    }

    #[test]
    fn test_encode_counter() {
        let counter = Counter::new("test_counter", "A test counter");
        counter.inc_by(42);

        let family = PrometheusEncoder::encode_counter(&counter);
        assert_eq!(family.name, "test_counter");
        assert_eq!(family.metric_type, MetricType::Counter);
        assert_eq!(family.samples.len(), 1);
        assert_eq!(family.samples[0].value, 42.0);
    }

    #[test]
    fn test_encode_gauge() {
        let gauge = Gauge::new("test_gauge", "A test gauge");
        gauge.set(3.14);

        let family = PrometheusEncoder::encode_gauge(&gauge);
        assert_eq!(family.name, "test_gauge");
        assert_eq!(family.metric_type, MetricType::Gauge);
        assert_eq!(family.samples.len(), 1);
        assert!((family.samples[0].value - 3.14).abs() < 0.001);
    }

    #[test]
    fn test_encode_histogram() {
        let histogram = Histogram::with_buckets(
            "test_histogram",
            "A test histogram",
            &[1.0, 5.0, 10.0],
        );
        histogram.observe(0.5);
        histogram.observe(3.0);
        histogram.observe(7.0);

        let family = PrometheusEncoder::encode_histogram(&histogram);
        assert_eq!(family.name, "test_histogram");
        assert_eq!(family.metric_type, MetricType::Histogram);

        let text = family.to_prometheus_text();
        assert!(text.contains("test_histogram_bucket{le=\"1\"} 1"));
        assert!(text.contains("test_histogram_bucket{le=\"5\"} 2"));
        assert!(text.contains("test_histogram_bucket{le=\"10\"} 3"));
        assert!(text.contains("test_histogram_bucket{le=\"+Inf\"} 3"));
        assert!(text.contains("test_histogram_sum"));
        assert!(text.contains("test_histogram_count 3"));
    }

    #[test]
    fn test_encode_transcode_metrics() {
        let metrics = TranscodeMetrics::new();
        metrics.record_job_start();
        metrics.update_fps(30.0);

        let text = PrometheusEncoder::encode_transcode_metrics(&metrics);
        assert!(text.contains("transcode_jobs_started_total"));
        assert!(text.contains("transcode_jobs_in_progress"));
        assert!(text.contains("transcode_current_fps"));
    }

    #[test]
    fn test_encode_codec_metrics() {
        let metrics = CodecMetrics::new();
        metrics.record_encode(0.033, true);
        metrics.record_decode(0.010);

        let text = PrometheusEncoder::encode_codec_metrics(&metrics);
        assert!(text.contains("codec_encode_time_seconds"));
        assert!(text.contains("codec_decode_time_seconds"));
        assert!(text.contains("codec_frames_encoded_total"));
        assert!(text.contains("codec_keyframes_total"));
    }

    #[test]
    fn test_encode_resource_metrics() {
        let metrics = ResourceMetrics::new();
        metrics.update_cpu(75.5);
        metrics.update_memory(4_000_000_000, 8_000_000_000);

        let text = PrometheusEncoder::encode_resource_metrics(&metrics);
        assert!(text.contains("resource_cpu_usage_percent"));
        assert!(text.contains("resource_memory_usage_bytes"));
    }

    #[test]
    fn test_encode_error_metrics() {
        let metrics = ErrorMetrics::new();
        metrics.record_error("decode");
        metrics.record_error("decode");
        metrics.record_error("encode");

        let text = PrometheusEncoder::encode_error_metrics(&metrics);
        assert!(text.contains("errors_total"));
        assert!(text.contains("errors_by_type_total"));
    }

    #[test]
    fn test_encode_registry() {
        let registry = MetricsRegistry::new();
        registry.transcode.record_job_start();
        registry.codec.record_encode(0.033, true);

        let text = PrometheusEncoder::encode_registry(&registry);
        assert!(text.contains("transcode_jobs_started_total"));
        assert!(text.contains("codec_frames_encoded_total"));
    }

    #[test]
    fn test_prometheus_exporter_config() {
        let config = PrometheusExporterConfig::new()
            .address("127.0.0.1")
            .port(9091)
            .metrics_path("/custom-metrics");

        assert_eq!(config.address, "127.0.0.1");
        assert_eq!(config.port, 9091);
        assert_eq!(config.metrics_path, "/custom-metrics");
        assert_eq!(config.bind_address(), "127.0.0.1:9091");
    }

    #[test]
    fn test_build_metrics_response() {
        let registry = MetricsRegistry::new();
        registry.transcode.record_job_start();

        let (content_type, body) = build_metrics_response(&registry);
        assert!(content_type.contains("text/plain"));
        assert!(body.contains("transcode_jobs_started_total"));
    }

    #[test]
    fn test_counter_with_labels_prometheus() {
        let mut labels = HashMap::new();
        labels.insert("codec".to_string(), "h264".to_string());
        labels.insert("resolution".to_string(), "1080p".to_string());

        let counter = Counter::with_labels("encoded_frames_total", "Total encoded frames", labels);
        counter.inc_by(100);

        let family = PrometheusEncoder::encode_counter(&counter);
        let text = family.to_prometheus_text();

        assert!(text.contains("codec=\"h264\""));
        assert!(text.contains("resolution=\"1080p\""));
    }

    #[test]
    fn test_metric_type_as_str() {
        assert_eq!(MetricType::Counter.as_str(), "counter");
        assert_eq!(MetricType::Gauge.as_str(), "gauge");
        assert_eq!(MetricType::Histogram.as_str(), "histogram");
        assert_eq!(MetricType::Summary.as_str(), "summary");
        assert_eq!(MetricType::Untyped.as_str(), "untyped");
    }
}
