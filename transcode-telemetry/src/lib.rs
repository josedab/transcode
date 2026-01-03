//! Transcode Telemetry - OpenTelemetry-based observability for the transcode system.
//!
//! This crate provides comprehensive observability features for the transcode system:
//!
//! - **Metrics**: Counters, gauges, and histograms for tracking transcode job metrics,
//!   codec performance, error rates, and resource utilization.
//! - **Tracing**: Distributed tracing with spans for tracking operations across
//!   pipeline stages.
//! - **Logging**: Structured JSON logging with trace correlation.
//! - **Prometheus**: Prometheus-compatible `/metrics` endpoint.
//! - **Health**: Liveness and readiness health check endpoints.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use transcode_telemetry::{
//!     TelemetryConfig, Telemetry,
//!     metrics::MetricsRegistry,
//!     logging::{Logger, LoggerConfig, Level},
//!     health::HealthChecker,
//! };
//! use std::sync::Arc;
//!
//! // Create telemetry configuration
//! let config = TelemetryConfig::new("transcode-service")
//!     .version("1.0.0")
//!     .environment("production")
//!     .log_level(Level::Info)
//!     .prometheus_port(9090)
//!     .health_port(8080);
//!
//! // Initialize telemetry
//! let telemetry = Telemetry::new(config);
//!
//! // Record metrics
//! telemetry.metrics().transcode.record_job_start();
//! telemetry.metrics().transcode.update_fps(30.0);
//!
//! // Log events
//! telemetry.logger().info("Transcode started");
//!
//! // Start span
//! let span = telemetry.tracer().start_span("process_frame");
//! span.set_attribute("frame", 100i64);
//!
//! // Mark as ready
//! telemetry.health().readiness().set_ready(true);
//! ```
//!
//! # Feature Overview
//!
//! ## Metrics
//!
//! The metrics module provides pre-defined metrics for transcode operations:
//!
//! - `TranscodeMetrics`: Job counters, FPS, bitrate, progress, queue depth
//! - `CodecMetrics`: Encode/decode time, GOP stats, keyframe counts
//! - `ResourceMetrics`: CPU, memory, GPU utilization
//! - `ErrorMetrics`: Error counts by type
//!
//! ## Tracing
//!
//! Distributed tracing for tracking operations:
//!
//! - Automatic trace/span ID generation
//! - W3C Trace Context propagation
//! - Pipeline stage spans
//! - Codec operation spans
//!
//! ## Logging
//!
//! Structured JSON logging:
//!
//! - Trace correlation (trace_id, span_id)
//! - Custom fields
//! - Multiple output targets
//!
//! ## Health Checks
//!
//! Kubernetes-compatible health endpoints:
//!
//! - `/health/live` - Liveness probe
//! - `/health/ready` - Readiness probe
//! - `/health` - Full health status
//!
//! ## Prometheus
//!
//! Prometheus metrics export:
//!
//! - `/metrics` endpoint
//! - All metric types (counter, gauge, histogram)
//! - Custom labels support

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod error;
pub mod health;
pub mod logging;
pub mod metrics;
pub mod prometheus;
pub mod tracing;

pub use error::{Result, TelemetryError};

use std::sync::Arc;

use health::{HealthChecker, HealthServerConfig, HealthServer};
use logging::{Level, Logger, LoggerConfig, StdoutOutput};
use metrics::MetricsRegistry;
use prometheus::{PrometheusExporter, PrometheusExporterConfig};
use tracing::Tracer;

/// Telemetry configuration.
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// Service name.
    pub service_name: String,
    /// Service version.
    pub service_version: Option<String>,
    /// Environment (e.g., "production", "development").
    pub environment: Option<String>,
    /// Minimum log level.
    pub log_level: Level,
    /// Enable JSON logging.
    pub json_logging: bool,
    /// Prometheus metrics port (0 to disable).
    pub prometheus_port: u16,
    /// Prometheus metrics path.
    pub prometheus_path: String,
    /// Health check port (0 to disable).
    pub health_port: u16,
    /// Enable console output.
    pub console_output: bool,
}

impl TelemetryConfig {
    /// Create a new telemetry configuration.
    pub fn new(service_name: impl Into<String>) -> Self {
        Self {
            service_name: service_name.into(),
            service_version: None,
            environment: None,
            log_level: Level::Info,
            json_logging: true,
            prometheus_port: 9090,
            prometheus_path: "/metrics".to_string(),
            health_port: 8080,
            console_output: true,
        }
    }

    /// Set the service version.
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.service_version = Some(version.into());
        self
    }

    /// Set the environment.
    pub fn environment(mut self, env: impl Into<String>) -> Self {
        self.environment = Some(env.into());
        self
    }

    /// Set the log level.
    pub fn log_level(mut self, level: Level) -> Self {
        self.log_level = level;
        self
    }

    /// Set JSON logging mode.
    pub fn json_logging(mut self, enabled: bool) -> Self {
        self.json_logging = enabled;
        self
    }

    /// Set the Prometheus port (0 to disable).
    pub fn prometheus_port(mut self, port: u16) -> Self {
        self.prometheus_port = port;
        self
    }

    /// Set the Prometheus path.
    pub fn prometheus_path(mut self, path: impl Into<String>) -> Self {
        self.prometheus_path = path.into();
        self
    }

    /// Set the health check port (0 to disable).
    pub fn health_port(mut self, port: u16) -> Self {
        self.health_port = port;
        self
    }

    /// Set console output mode.
    pub fn console_output(mut self, enabled: bool) -> Self {
        self.console_output = enabled;
        self
    }
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self::new("transcode")
    }
}

/// Main telemetry system.
pub struct Telemetry {
    config: TelemetryConfig,
    metrics: Arc<MetricsRegistry>,
    tracer: Arc<Tracer>,
    logger: Arc<Logger>,
    health: Arc<HealthChecker>,
    prometheus_exporter: Option<PrometheusExporter>,
    health_server: Option<HealthServer>,
}

impl Telemetry {
    /// Create a new telemetry system.
    pub fn new(config: TelemetryConfig) -> Self {
        // Create metrics registry
        let metrics = Arc::new(MetricsRegistry::new());

        // Create tracer
        let mut tracer = Tracer::new(&config.service_name);
        if let Some(ref version) = config.service_version {
            tracer = tracer.with_version(version);
        }
        let tracer = Arc::new(tracer);

        // Create logger
        let mut logger_config = LoggerConfig::new(&config.service_name)
            .level(config.log_level);

        if let Some(ref version) = config.service_version {
            logger_config = logger_config.version(version);
        }

        if let Some(ref env) = config.environment {
            logger_config = logger_config.environment(env);
        }

        let mut logger = Logger::new(logger_config);

        if config.console_output {
            let output = Arc::new(StdoutOutput::new().json(config.json_logging));
            logger.add_output(output);
        }

        let logger = Arc::new(logger);

        // Create health checker
        let health = Arc::new(HealthChecker::new());

        Self {
            config,
            metrics,
            tracer,
            logger,
            health,
            prometheus_exporter: None,
            health_server: None,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &TelemetryConfig {
        &self.config
    }

    /// Get the metrics registry.
    pub fn metrics(&self) -> &Arc<MetricsRegistry> {
        &self.metrics
    }

    /// Get the tracer.
    pub fn tracer(&self) -> &Arc<Tracer> {
        &self.tracer
    }

    /// Get the logger.
    pub fn logger(&self) -> &Arc<Logger> {
        &self.logger
    }

    /// Get the health checker.
    pub fn health(&self) -> &Arc<HealthChecker> {
        &self.health
    }

    /// Start the Prometheus exporter.
    pub fn start_prometheus(&mut self) -> Result<()> {
        if self.config.prometheus_port == 0 {
            return Ok(());
        }

        let config = PrometheusExporterConfig::new()
            .port(self.config.prometheus_port)
            .metrics_path(&self.config.prometheus_path);

        let mut exporter = PrometheusExporter::new(config, self.metrics.clone());
        exporter.start()?;

        self.prometheus_exporter = Some(exporter);
        Ok(())
    }

    /// Start the health server.
    pub fn start_health_server(&mut self) -> Result<()> {
        if self.config.health_port == 0 {
            return Ok(());
        }

        let config = HealthServerConfig::new().port(self.config.health_port);
        let mut server = HealthServer::new(config, self.health.clone());
        server.start()?;

        self.health_server = Some(server);
        Ok(())
    }

    /// Start all servers (Prometheus and health).
    pub fn start_servers(&mut self) -> Result<()> {
        self.start_prometheus()?;
        self.start_health_server()?;
        Ok(())
    }

    /// Poll servers for incoming connections (non-blocking).
    pub fn poll(&self) -> Result<()> {
        if let Some(ref exporter) = self.prometheus_exporter {
            exporter.poll()?;
        }

        if let Some(ref server) = self.health_server {
            server.poll()?;
        }

        Ok(())
    }

    /// Stop all servers.
    pub fn stop_servers(&mut self) {
        if let Some(ref mut exporter) = self.prometheus_exporter {
            exporter.stop();
        }

        if let Some(ref mut server) = self.health_server {
            server.stop();
        }
    }

    /// Mark the service as ready.
    pub fn set_ready(&self, ready: bool) {
        if ready {
            self.health.readiness().set_ready(true);
        } else {
            self.health.readiness().set_not_ready("Service not ready");
        }
    }

    /// Set the span context for log correlation.
    pub fn set_log_span_context(&self, context: Option<tracing::SpanContext>) {
        self.logger.set_span_context(context);
    }

    /// Get the current Prometheus metrics output.
    pub fn get_prometheus_metrics(&self) -> String {
        prometheus::PrometheusEncoder::encode_registry(&self.metrics)
    }
}

/// Builder for creating a Telemetry instance.
pub struct TelemetryBuilder {
    config: TelemetryConfig,
}

impl TelemetryBuilder {
    /// Create a new builder.
    pub fn new(service_name: impl Into<String>) -> Self {
        Self {
            config: TelemetryConfig::new(service_name),
        }
    }

    /// Set the service version.
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.config.service_version = Some(version.into());
        self
    }

    /// Set the environment.
    pub fn environment(mut self, env: impl Into<String>) -> Self {
        self.config.environment = Some(env.into());
        self
    }

    /// Set the log level.
    pub fn log_level(mut self, level: Level) -> Self {
        self.config.log_level = level;
        self
    }

    /// Enable/disable JSON logging.
    pub fn json_logging(mut self, enabled: bool) -> Self {
        self.config.json_logging = enabled;
        self
    }

    /// Set the Prometheus port.
    pub fn prometheus_port(mut self, port: u16) -> Self {
        self.config.prometheus_port = port;
        self
    }

    /// Set the health check port.
    pub fn health_port(mut self, port: u16) -> Self {
        self.config.health_port = port;
        self
    }

    /// Enable/disable console output.
    pub fn console_output(mut self, enabled: bool) -> Self {
        self.config.console_output = enabled;
        self
    }

    /// Build the Telemetry instance.
    pub fn build(self) -> Telemetry {
        Telemetry::new(self.config)
    }

    /// Build and start all servers.
    pub fn build_and_start(self) -> Result<Telemetry> {
        let mut telemetry = self.build();
        telemetry.start_servers()?;
        Ok(telemetry)
    }
}

/// Initialize telemetry with default configuration.
pub fn init(service_name: impl Into<String>) -> Telemetry {
    Telemetry::new(TelemetryConfig::new(service_name))
}

/// Initialize telemetry with a builder.
pub fn builder(service_name: impl Into<String>) -> TelemetryBuilder {
    TelemetryBuilder::new(service_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_config() {
        let config = TelemetryConfig::new("test-service")
            .version("1.0.0")
            .environment("test")
            .log_level(Level::Debug)
            .prometheus_port(9091)
            .health_port(8081);

        assert_eq!(config.service_name, "test-service");
        assert_eq!(config.service_version, Some("1.0.0".to_string()));
        assert_eq!(config.environment, Some("test".to_string()));
        assert_eq!(config.log_level, Level::Debug);
        assert_eq!(config.prometheus_port, 9091);
        assert_eq!(config.health_port, 8081);
    }

    #[test]
    fn test_telemetry_creation() {
        let config = TelemetryConfig::new("test-service")
            .prometheus_port(0)
            .health_port(0)
            .console_output(false);

        let telemetry = Telemetry::new(config);

        assert_eq!(telemetry.config().service_name, "test-service");
        assert!(Arc::strong_count(&telemetry.metrics) >= 1);
        assert!(Arc::strong_count(&telemetry.tracer) >= 1);
        assert!(Arc::strong_count(&telemetry.logger) >= 1);
        assert!(Arc::strong_count(&telemetry.health) >= 1);
    }

    #[test]
    fn test_telemetry_metrics() {
        let config = TelemetryConfig::new("test-service")
            .prometheus_port(0)
            .health_port(0)
            .console_output(false);

        let telemetry = Telemetry::new(config);

        telemetry.metrics().transcode.record_job_start();
        assert_eq!(telemetry.metrics().transcode.jobs_started.get(), 1);

        telemetry.metrics().transcode.update_fps(30.0);
        assert_eq!(telemetry.metrics().transcode.current_fps.get(), 30.0);
    }

    #[test]
    fn test_telemetry_tracing() {
        let config = TelemetryConfig::new("test-service")
            .prometheus_port(0)
            .health_port(0)
            .console_output(false);

        let telemetry = Telemetry::new(config);

        let span = telemetry.tracer().start_span("test_operation");
        span.set_attribute("key", "value");

        assert_eq!(span.name(), "test_operation");
        assert!(telemetry.tracer().active_spans().len() >= 1);
    }

    #[test]
    fn test_telemetry_health() {
        let config = TelemetryConfig::new("test-service")
            .prometheus_port(0)
            .health_port(0)
            .console_output(false);

        let telemetry = Telemetry::new(config);

        // Check liveness
        let result = telemetry.health().check_liveness();
        assert!(result.status.is_healthy());

        // Check readiness (initially not ready)
        let result = telemetry.health().check_readiness();
        assert!(!result.status.is_healthy());

        // Set ready
        telemetry.set_ready(true);
        let result = telemetry.health().check_readiness();
        assert!(result.status.is_healthy());
    }

    #[test]
    fn test_telemetry_builder() {
        let telemetry = builder("test-service")
            .version("2.0.0")
            .environment("staging")
            .log_level(Level::Warn)
            .prometheus_port(0)
            .health_port(0)
            .console_output(false)
            .build();

        assert_eq!(telemetry.config().service_name, "test-service");
        assert_eq!(telemetry.config().service_version, Some("2.0.0".to_string()));
        assert_eq!(telemetry.config().environment, Some("staging".to_string()));
        assert_eq!(telemetry.config().log_level, Level::Warn);
    }

    #[test]
    fn test_init() {
        let telemetry = init("simple-service");
        assert_eq!(telemetry.config().service_name, "simple-service");
    }

    #[test]
    fn test_prometheus_metrics_output() {
        let config = TelemetryConfig::new("test-service")
            .prometheus_port(0)
            .health_port(0)
            .console_output(false);

        let telemetry = Telemetry::new(config);

        telemetry.metrics().transcode.record_job_start();
        telemetry.metrics().transcode.record_job_complete(60.0);

        let output = telemetry.get_prometheus_metrics();
        assert!(output.contains("transcode_jobs_started_total"));
        assert!(output.contains("transcode_jobs_completed_total"));
    }

    #[test]
    fn test_set_log_span_context() {
        let config = TelemetryConfig::new("test-service")
            .prometheus_port(0)
            .health_port(0)
            .console_output(false);

        let telemetry = Telemetry::new(config);

        let ctx = tracing::SpanContext::new_root();
        telemetry.set_log_span_context(Some(ctx.clone()));

        assert!(telemetry.logger().span_context().is_some());

        telemetry.set_log_span_context(None);
        assert!(telemetry.logger().span_context().is_none());
    }

    #[test]
    fn test_telemetry_default_config() {
        let config = TelemetryConfig::default();
        assert_eq!(config.service_name, "transcode");
        assert_eq!(config.log_level, Level::Info);
        assert!(config.json_logging);
        assert_eq!(config.prometheus_port, 9090);
        assert_eq!(config.health_port, 8080);
    }
}
