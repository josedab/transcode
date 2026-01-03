//! Error types for the transcode-telemetry crate.
//!
//! This module defines all error types that can occur during telemetry operations,
//! including initialization failures, metric recording errors, and export failures.

use std::fmt;

/// Result type alias for telemetry operations.
pub type Result<T> = std::result::Result<T, TelemetryError>;

/// Main error type for telemetry operations.
#[derive(Debug)]
pub enum TelemetryError {
    /// Error during telemetry initialization.
    Initialization(InitializationError),
    /// Error recording metrics.
    Metrics(MetricsError),
    /// Error during tracing operations.
    Tracing(TracingError),
    /// Error during logging operations.
    Logging(LoggingError),
    /// Error with Prometheus exporter.
    Prometheus(PrometheusError),
    /// Error with health checks.
    Health(HealthError),
    /// Configuration error.
    Config(ConfigError),
    /// I/O error.
    Io(std::io::Error),
}

impl std::error::Error for TelemetryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TelemetryError::Initialization(e) => Some(e),
            TelemetryError::Metrics(e) => Some(e),
            TelemetryError::Tracing(e) => Some(e),
            TelemetryError::Logging(e) => Some(e),
            TelemetryError::Prometheus(e) => Some(e),
            TelemetryError::Health(e) => Some(e),
            TelemetryError::Config(e) => Some(e),
            TelemetryError::Io(e) => Some(e),
        }
    }
}

impl fmt::Display for TelemetryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TelemetryError::Initialization(e) => write!(f, "Initialization error: {}", e),
            TelemetryError::Metrics(e) => write!(f, "Metrics error: {}", e),
            TelemetryError::Tracing(e) => write!(f, "Tracing error: {}", e),
            TelemetryError::Logging(e) => write!(f, "Logging error: {}", e),
            TelemetryError::Prometheus(e) => write!(f, "Prometheus error: {}", e),
            TelemetryError::Health(e) => write!(f, "Health check error: {}", e),
            TelemetryError::Config(e) => write!(f, "Configuration error: {}", e),
            TelemetryError::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl From<std::io::Error> for TelemetryError {
    fn from(err: std::io::Error) -> Self {
        TelemetryError::Io(err)
    }
}

impl From<InitializationError> for TelemetryError {
    fn from(err: InitializationError) -> Self {
        TelemetryError::Initialization(err)
    }
}

impl From<MetricsError> for TelemetryError {
    fn from(err: MetricsError) -> Self {
        TelemetryError::Metrics(err)
    }
}

impl From<TracingError> for TelemetryError {
    fn from(err: TracingError) -> Self {
        TelemetryError::Tracing(err)
    }
}

impl From<LoggingError> for TelemetryError {
    fn from(err: LoggingError) -> Self {
        TelemetryError::Logging(err)
    }
}

impl From<PrometheusError> for TelemetryError {
    fn from(err: PrometheusError) -> Self {
        TelemetryError::Prometheus(err)
    }
}

impl From<HealthError> for TelemetryError {
    fn from(err: HealthError) -> Self {
        TelemetryError::Health(err)
    }
}

impl From<ConfigError> for TelemetryError {
    fn from(err: ConfigError) -> Self {
        TelemetryError::Config(err)
    }
}

/// Errors that can occur during telemetry initialization.
#[derive(Debug)]
pub struct InitializationError {
    /// The component that failed to initialize.
    pub component: String,
    /// Description of the error.
    pub message: String,
    /// Underlying cause, if any.
    pub source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl InitializationError {
    /// Create a new initialization error.
    pub fn new(component: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            component: component.into(),
            message: message.into(),
            source: None,
        }
    }

    /// Create a new initialization error with a source.
    pub fn with_source(
        component: impl Into<String>,
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self {
            component: component.into(),
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }
}

impl std::error::Error for InitializationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|e| e.as_ref() as &(dyn std::error::Error + 'static))
    }
}

impl fmt::Display for InitializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Failed to initialize {}: {}", self.component, self.message)
    }
}

/// Errors related to metrics collection.
#[derive(Debug)]
#[allow(missing_docs)]
pub enum MetricsError {
    /// Failed to create a metric.
    Creation { name: String, reason: String },
    /// Failed to record a metric value.
    Recording { name: String, reason: String },
    /// Invalid metric name.
    InvalidName { name: String, reason: String },
    /// Invalid label key or value.
    InvalidLabel { key: String, value: String, reason: String },
    /// Metric not found.
    NotFound { name: String },
    /// Metric type mismatch.
    TypeMismatch { name: String, expected: String, actual: String },
}

impl std::error::Error for MetricsError {}

impl fmt::Display for MetricsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetricsError::Creation { name, reason } => {
                write!(f, "Failed to create metric '{}': {}", name, reason)
            }
            MetricsError::Recording { name, reason } => {
                write!(f, "Failed to record metric '{}': {}", name, reason)
            }
            MetricsError::InvalidName { name, reason } => {
                write!(f, "Invalid metric name '{}': {}", name, reason)
            }
            MetricsError::InvalidLabel { key, value, reason } => {
                write!(f, "Invalid label {}='{}': {}", key, value, reason)
            }
            MetricsError::NotFound { name } => {
                write!(f, "Metric '{}' not found", name)
            }
            MetricsError::TypeMismatch { name, expected, actual } => {
                write!(f, "Metric '{}' type mismatch: expected {}, got {}", name, expected, actual)
            }
        }
    }
}

/// Errors related to distributed tracing.
#[derive(Debug)]
#[allow(missing_docs)]
pub enum TracingError {
    /// Failed to create a span.
    SpanCreation { name: String, reason: String },
    /// Failed to export spans.
    Export { reason: String },
    /// Invalid span context.
    InvalidContext { reason: String },
    /// Tracer provider error.
    Provider { reason: String },
}

impl std::error::Error for TracingError {}

impl fmt::Display for TracingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TracingError::SpanCreation { name, reason } => {
                write!(f, "Failed to create span '{}': {}", name, reason)
            }
            TracingError::Export { reason } => {
                write!(f, "Failed to export spans: {}", reason)
            }
            TracingError::InvalidContext { reason } => {
                write!(f, "Invalid span context: {}", reason)
            }
            TracingError::Provider { reason } => {
                write!(f, "Tracer provider error: {}", reason)
            }
        }
    }
}

/// Errors related to logging.
#[derive(Debug)]
#[allow(missing_docs)]
pub enum LoggingError {
    /// Failed to initialize the logger.
    Initialization { reason: String },
    /// Failed to write log.
    Write { reason: String },
    /// Invalid log level.
    InvalidLevel { level: String },
    /// Subscriber error.
    Subscriber { reason: String },
}

impl std::error::Error for LoggingError {}

impl fmt::Display for LoggingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoggingError::Initialization { reason } => {
                write!(f, "Failed to initialize logger: {}", reason)
            }
            LoggingError::Write { reason } => {
                write!(f, "Failed to write log: {}", reason)
            }
            LoggingError::InvalidLevel { level } => {
                write!(f, "Invalid log level: {}", level)
            }
            LoggingError::Subscriber { reason } => {
                write!(f, "Subscriber error: {}", reason)
            }
        }
    }
}

/// Errors related to the Prometheus exporter.
#[derive(Debug)]
#[allow(missing_docs)]
pub enum PrometheusError {
    /// Failed to start the exporter server.
    Server { reason: String },
    /// Failed to encode metrics.
    Encoding { reason: String },
    /// Registry error.
    Registry { reason: String },
    /// Failed to bind to address.
    Bind { address: String, reason: String },
}

impl std::error::Error for PrometheusError {}

impl fmt::Display for PrometheusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrometheusError::Server { reason } => {
                write!(f, "Prometheus server error: {}", reason)
            }
            PrometheusError::Encoding { reason } => {
                write!(f, "Failed to encode metrics: {}", reason)
            }
            PrometheusError::Registry { reason } => {
                write!(f, "Registry error: {}", reason)
            }
            PrometheusError::Bind { address, reason } => {
                write!(f, "Failed to bind to {}: {}", address, reason)
            }
        }
    }
}

/// Errors related to health checks.
#[derive(Debug)]
#[allow(missing_docs)]
pub enum HealthError {
    /// Health check failed.
    CheckFailed { check_name: String, reason: String },
    /// Health check timed out.
    Timeout { check_name: String, timeout_ms: u64 },
    /// Component unhealthy.
    Unhealthy { component: String, reason: String },
}

impl std::error::Error for HealthError {}

impl fmt::Display for HealthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HealthError::CheckFailed { check_name, reason } => {
                write!(f, "Health check '{}' failed: {}", check_name, reason)
            }
            HealthError::Timeout { check_name, timeout_ms } => {
                write!(f, "Health check '{}' timed out after {}ms", check_name, timeout_ms)
            }
            HealthError::Unhealthy { component, reason } => {
                write!(f, "Component '{}' is unhealthy: {}", component, reason)
            }
        }
    }
}

/// Configuration errors.
#[derive(Debug)]
#[allow(missing_docs)]
pub enum ConfigError {
    /// Missing required configuration.
    Missing { key: String },
    /// Invalid configuration value.
    Invalid { key: String, value: String, reason: String },
    /// Configuration parsing error.
    Parse { reason: String },
}

impl std::error::Error for ConfigError {}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Missing { key } => {
                write!(f, "Missing required configuration: {}", key)
            }
            ConfigError::Invalid { key, value, reason } => {
                write!(f, "Invalid configuration {}='{}': {}", key, value, reason)
            }
            ConfigError::Parse { reason } => {
                write!(f, "Configuration parse error: {}", reason)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialization_error_display() {
        let err = InitializationError::new("metrics", "failed to connect");
        assert_eq!(
            err.to_string(),
            "Failed to initialize metrics: failed to connect"
        );
    }

    #[test]
    fn test_metrics_error_display() {
        let err = MetricsError::Creation {
            name: "transcode_fps".to_string(),
            reason: "invalid bucket boundaries".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Failed to create metric 'transcode_fps': invalid bucket boundaries"
        );
    }

    #[test]
    fn test_telemetry_error_from_initialization() {
        let init_err = InitializationError::new("tracing", "exporter failed");
        let telemetry_err: TelemetryError = init_err.into();
        assert!(matches!(telemetry_err, TelemetryError::Initialization(_)));
    }

    #[test]
    fn test_health_error_timeout() {
        let err = HealthError::Timeout {
            check_name: "database".to_string(),
            timeout_ms: 5000,
        };
        assert_eq!(
            err.to_string(),
            "Health check 'database' timed out after 5000ms"
        );
    }

    #[test]
    fn test_config_error_missing() {
        let err = ConfigError::Missing {
            key: "OTEL_ENDPOINT".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Missing required configuration: OTEL_ENDPOINT"
        );
    }

    #[test]
    fn test_prometheus_error_bind() {
        let err = PrometheusError::Bind {
            address: "0.0.0.0:9090".to_string(),
            reason: "address already in use".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Failed to bind to 0.0.0.0:9090: address already in use"
        );
    }

    #[test]
    fn test_tracing_error_export() {
        let err = TracingError::Export {
            reason: "connection refused".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Failed to export spans: connection refused"
        );
    }

    #[test]
    fn test_logging_error_invalid_level() {
        let err = LoggingError::InvalidLevel {
            level: "VERBOSE".to_string(),
        };
        assert_eq!(err.to_string(), "Invalid log level: VERBOSE");
    }
}
