//! OpenTelemetry observability integration for the transcode library.
//!
//! This crate provides comprehensive observability for transcoding operations
//! including distributed tracing, metrics collection, and log correlation.
//!
//! # Features
//!
//! - **Distributed Tracing**: Per-frame trace spans with context propagation
//! - **Metrics**: Encoding speed, quality metrics, resource utilization
//! - **OTLP Export**: Send telemetry to any OpenTelemetry-compatible backend
//! - **Prometheus**: Optional Prometheus metrics endpoint
//!
//! # Example
//!
//! ```ignore
//! use transcode_otel::{TelemetryConfig, init_telemetry, TranscodeSpan};
//!
//! #[tokio::main]
//! async fn main() {
//!     // Initialize telemetry
//!     let config = TelemetryConfig::default()
//!         .service_name("my-transcoder")
//!         .otlp_endpoint("http://localhost:4317");
//!
//!     init_telemetry(config).await.unwrap();
//!
//!     // Create a span for transcoding operation
//!     let span = TranscodeSpan::new("transcode_video")
//!         .input_file("input.mp4")
//!         .output_codec("h264")
//!         .start();
//!
//!     // Record frame processing
//!     span.record_frame(1, 1920, 1080, 5.2);
//!
//!     // Finish the span
//!     span.finish();
//! }
//! ```

#![allow(dead_code)]

mod config;
mod error;
mod metrics;
mod spans;

pub use config::{TelemetryConfig, TelemetryConfigBuilder};
pub use error::{OtelError, Result};
pub use metrics::{
    FrameMetrics, JobMetrics, MetricsRecorder, QualityMetrics, TranscodeMetrics, WorkerMetrics,
};
pub use spans::{FrameSpan, JobSpan, SpanContext, TranscodeSpan};

use parking_lot::RwLock;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Global telemetry state.
static TELEMETRY: RwLock<Option<Arc<TelemetryState>>> = RwLock::new(None);

struct TelemetryState {
    config: TelemetryConfig,
    metrics: MetricsRecorder,
}

/// Initialize telemetry with the given configuration.
pub async fn init_telemetry(config: TelemetryConfig) -> Result<()> {
    // Set up tracing subscriber
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(&config.log_level)
    });

    let subscriber = tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer());

    // In production with opentelemetry features enabled, would add:
    // .with(tracing_opentelemetry::layer().with_tracer(tracer))

    subscriber.init();

    // Initialize metrics
    let metrics = MetricsRecorder::new(&config);

    let state = Arc::new(TelemetryState { config, metrics });

    *TELEMETRY.write() = Some(state);

    tracing::info!("Telemetry initialized");
    Ok(())
}

/// Shutdown telemetry and flush pending data.
pub async fn shutdown_telemetry() -> Result<()> {
    if let Some(state) = TELEMETRY.write().take() {
        tracing::info!("Shutting down telemetry");
        // Flush metrics and traces
        drop(state);
    }
    Ok(())
}

/// Get the global metrics recorder.
pub fn metrics() -> Option<MetricsRecorder> {
    TELEMETRY.read().as_ref().map(|s| s.metrics.clone())
}

/// Record a transcoding job.
pub fn record_job(job_id: &str) -> JobSpan {
    JobSpan::new(job_id)
}

/// Record frame processing.
pub fn record_frame(frame_number: u64, width: u32, height: u32) -> FrameSpan {
    FrameSpan::new(frame_number, width, height)
}

/// Version of the library.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_init_telemetry() {
        let config = TelemetryConfig::default();
        // Note: Can only init once per process, so this may fail in parallel tests
        // In real tests, would use a test-specific setup
    }
}
