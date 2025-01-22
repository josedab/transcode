//! Configuration for OpenTelemetry telemetry.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Telemetry configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Service name for traces.
    pub service_name: String,
    /// Service version.
    pub service_version: String,
    /// Environment (e.g., "production", "staging").
    pub environment: String,
    /// OTLP endpoint for traces.
    pub otlp_endpoint: Option<String>,
    /// OTLP endpoint for metrics (if different from traces).
    pub otlp_metrics_endpoint: Option<String>,
    /// Prometheus endpoint (if enabled).
    pub prometheus_endpoint: Option<String>,
    /// Sampling ratio for traces (0.0-1.0).
    pub trace_sampling_ratio: f64,
    /// Batch export interval.
    pub export_interval: Duration,
    /// Maximum batch size for export.
    pub max_batch_size: usize,
    /// Log level.
    pub log_level: String,
    /// Custom resource attributes.
    pub resource_attributes: HashMap<String, String>,
    /// Enable per-frame tracing (can be verbose).
    pub enable_frame_tracing: bool,
    /// Enable quality metrics.
    pub enable_quality_metrics: bool,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            service_name: "transcode".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            environment: "development".to_string(),
            otlp_endpoint: None,
            otlp_metrics_endpoint: None,
            prometheus_endpoint: None,
            trace_sampling_ratio: 1.0,
            export_interval: Duration::from_secs(5),
            max_batch_size: 512,
            log_level: "info".to_string(),
            resource_attributes: HashMap::new(),
            enable_frame_tracing: false,
            enable_quality_metrics: true,
        }
    }
}

impl TelemetryConfig {
    /// Create a new configuration with service name.
    pub fn new(service_name: impl Into<String>) -> Self {
        Self {
            service_name: service_name.into(),
            ..Default::default()
        }
    }

    /// Set the service name.
    pub fn service_name(mut self, name: impl Into<String>) -> Self {
        self.service_name = name.into();
        self
    }

    /// Set the service version.
    pub fn service_version(mut self, version: impl Into<String>) -> Self {
        self.service_version = version.into();
        self
    }

    /// Set the environment.
    pub fn environment(mut self, env: impl Into<String>) -> Self {
        self.environment = env.into();
        self
    }

    /// Set the OTLP endpoint.
    pub fn otlp_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.otlp_endpoint = Some(endpoint.into());
        self
    }

    /// Set the Prometheus endpoint.
    pub fn prometheus_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.prometheus_endpoint = Some(endpoint.into());
        self
    }

    /// Set the trace sampling ratio.
    pub fn trace_sampling_ratio(mut self, ratio: f64) -> Self {
        self.trace_sampling_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Set the log level.
    pub fn log_level(mut self, level: impl Into<String>) -> Self {
        self.log_level = level.into();
        self
    }

    /// Add a resource attribute.
    pub fn resource_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.resource_attributes.insert(key.into(), value.into());
        self
    }

    /// Enable per-frame tracing.
    pub fn enable_frame_tracing(mut self, enabled: bool) -> Self {
        self.enable_frame_tracing = enabled;
        self
    }

    /// Create a builder.
    pub fn builder() -> TelemetryConfigBuilder {
        TelemetryConfigBuilder::default()
    }
}

/// Builder for telemetry configuration.
#[derive(Debug, Default)]
pub struct TelemetryConfigBuilder {
    config: TelemetryConfig,
}

impl TelemetryConfigBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the service name.
    pub fn service_name(mut self, name: impl Into<String>) -> Self {
        self.config.service_name = name.into();
        self
    }

    /// Set the OTLP endpoint.
    pub fn otlp_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.config.otlp_endpoint = Some(endpoint.into());
        self
    }

    /// Set the environment.
    pub fn environment(mut self, env: impl Into<String>) -> Self {
        self.config.environment = env.into();
        self
    }

    /// Build the configuration.
    pub fn build(self) -> TelemetryConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TelemetryConfig::default();
        assert_eq!(config.service_name, "transcode");
        assert_eq!(config.trace_sampling_ratio, 1.0);
    }

    #[test]
    fn test_builder() {
        let config = TelemetryConfig::builder()
            .service_name("my-service")
            .otlp_endpoint("http://localhost:4317")
            .environment("production")
            .build();

        assert_eq!(config.service_name, "my-service");
        assert_eq!(config.otlp_endpoint, Some("http://localhost:4317".to_string()));
        assert_eq!(config.environment, "production");
    }
}
