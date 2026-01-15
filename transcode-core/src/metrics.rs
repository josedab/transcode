//! Optional metrics collection for observability.
//!
//! This module provides convenience macros and functions for recording metrics
//! when the `metrics` feature is enabled. When disabled, all operations are no-ops.
//!
//! # Feature Flag
//!
//! Enable metrics collection by adding to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! transcode-core = { version = "1.0", features = ["metrics"] }
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use transcode_core::metrics::{record_counter, record_histogram};
//! use std::time::Instant;
//!
//! // Record a counter
//! record_counter!("codec.decode.packets", 1);
//!
//! // Record timing
//! let start = Instant::now();
//! // ... do work ...
//! record_histogram!("codec.decode.duration_ns", start.elapsed().as_nanos() as f64);
//! ```
//!
//! # Metric Names
//!
//! Recommended naming conventions:
//! - `codec.encode.packets` - packets encoded
//! - `codec.encode.bytes` - bytes encoded
//! - `codec.encode.duration_ns` - encoding time in nanoseconds
//! - `codec.decode.packets` - packets decoded
//! - `codec.decode.frames` - frames decoded
//! - `codec.decode.duration_ns` - decoding time in nanoseconds
//! - `container.demux.packets` - packets demuxed
//! - `container.mux.packets` - packets muxed
//! - `pipeline.frames_processed` - total frames through pipeline

/// Record a counter metric (increments by given value).
///
/// When the `metrics` feature is disabled, this is a no-op.
///
/// # Example
///
/// ```ignore
/// use transcode_core::metrics::record_counter;
///
/// record_counter!("codec.decode.packets", 1);
/// record_counter!("codec.decode.bytes", packet.len() as u64);
/// ```
#[macro_export]
#[cfg(feature = "metrics")]
macro_rules! record_counter {
    ($name:expr, $value:expr) => {
        ::metrics::counter!($name).increment($value)
    };
    ($name:expr, $value:expr, $($label_key:expr => $label_value:expr),+ $(,)?) => {
        ::metrics::counter!($name, $($label_key => $label_value),+).increment($value)
    };
}

#[macro_export]
#[cfg(not(feature = "metrics"))]
macro_rules! record_counter {
    ($name:expr, $value:expr) => {
        let _ = ($name, $value);
    };
    ($name:expr, $value:expr, $($label_key:expr => $label_value:expr),+ $(,)?) => {
        let _ = ($name, $value, $($label_key, $label_value),+);
    };
}

/// Record a histogram/distribution metric.
///
/// When the `metrics` feature is disabled, this is a no-op.
///
/// # Example
///
/// ```ignore
/// use transcode_core::metrics::record_histogram;
/// use std::time::Instant;
///
/// let start = Instant::now();
/// // ... do work ...
/// record_histogram!("codec.decode.duration_ns", start.elapsed().as_nanos() as f64);
/// ```
#[macro_export]
#[cfg(feature = "metrics")]
macro_rules! record_histogram {
    ($name:expr, $value:expr) => {
        ::metrics::histogram!($name).record($value)
    };
    ($name:expr, $value:expr, $($label_key:expr => $label_value:expr),+ $(,)?) => {
        ::metrics::histogram!($name, $($label_key => $label_value),+).record($value)
    };
}

#[macro_export]
#[cfg(not(feature = "metrics"))]
macro_rules! record_histogram {
    ($name:expr, $value:expr) => {
        let _ = ($name, $value);
    };
    ($name:expr, $value:expr, $($label_key:expr => $label_value:expr),+ $(,)?) => {
        let _ = ($name, $value, $($label_key, $label_value),+);
    };
}

/// Record a gauge metric (absolute value).
///
/// When the `metrics` feature is disabled, this is a no-op.
///
/// # Example
///
/// ```ignore
/// use transcode_core::metrics::record_gauge;
///
/// record_gauge!("pipeline.queue_depth", queue.len() as f64);
/// ```
#[macro_export]
#[cfg(feature = "metrics")]
macro_rules! record_gauge {
    ($name:expr, $value:expr) => {
        ::metrics::gauge!($name).set($value)
    };
    ($name:expr, $value:expr, $($label_key:expr => $label_value:expr),+ $(,)?) => {
        ::metrics::gauge!($name, $($label_key => $label_value),+).set($value)
    };
}

#[macro_export]
#[cfg(not(feature = "metrics"))]
macro_rules! record_gauge {
    ($name:expr, $value:expr) => {
        let _ = ($name, $value);
    };
    ($name:expr, $value:expr, $($label_key:expr => $label_value:expr),+ $(,)?) => {
        let _ = ($name, $value, $($label_key, $label_value),+);
    };
}

// Re-export macros at module level for convenient imports
pub use record_counter;
pub use record_gauge;
pub use record_histogram;

/// Helper to time an operation and record a histogram.
///
/// Returns the result of the closure and records the duration.
///
/// # Example
///
/// ```ignore
/// use transcode_core::metrics::timed;
///
/// let result = timed("codec.decode.duration_ns", || {
///     decoder.decode(packet)
/// });
/// ```
#[inline]
pub fn timed<F, R>(metric_name: &'static str, f: F) -> R
where
    F: FnOnce() -> R,
{
    #[cfg(feature = "metrics")]
    {
        let start = std::time::Instant::now();
        let result = f();
        record_histogram!(metric_name, start.elapsed().as_nanos() as f64);
        result
    }

    #[cfg(not(feature = "metrics"))]
    {
        let _ = metric_name;
        f()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter_macro_compiles() {
        record_counter!("test.counter", 1u64);
        record_counter!("test.counter.labeled", 5u64, "codec" => "h264");
    }

    #[test]
    fn test_histogram_macro_compiles() {
        record_histogram!("test.histogram", 42.0);
        record_histogram!("test.histogram.labeled", 100.0, "operation" => "decode");
    }

    #[test]
    fn test_gauge_macro_compiles() {
        record_gauge!("test.gauge", 10.0);
        record_gauge!("test.gauge.labeled", 25.0, "queue" => "input");
    }

    #[test]
    fn test_timed_helper() {
        let result = timed("test.timed", || {
            std::thread::sleep(std::time::Duration::from_micros(100));
            42
        });
        assert_eq!(result, 42);
    }
}
