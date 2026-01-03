//! Error types for loudness processing.
//!
//! This module defines all error types that can occur during loudness
//! measurement, normalization, and true peak limiting.

use thiserror::Error;

/// Main error type for loudness processing.
#[derive(Error, Debug, Clone)]
pub enum LoudnessError {
    /// Invalid sample rate provided.
    #[error("Invalid sample rate: {rate} Hz (must be >= 8000 Hz)")]
    InvalidSampleRate {
        /// The invalid sample rate value.
        rate: u32,
    },

    /// Invalid channel count.
    #[error("Invalid channel count: {count} (must be between 1 and 8)")]
    InvalidChannelCount {
        /// The invalid channel count value.
        count: u32,
    },

    /// Invalid target loudness.
    #[error("Invalid target loudness: {lufs} LUFS (must be between -70 and 0)")]
    InvalidTargetLoudness {
        /// The invalid target loudness value in LUFS.
        lufs: f64,
    },

    /// Invalid true peak limit.
    #[error("Invalid true peak limit: {dbtp} dBTP (must be between -20 and 0)")]
    InvalidTruePeakLimit {
        /// The invalid true peak limit value in dBTP.
        dbtp: f64,
    },

    /// No audio data processed.
    #[error("No audio data processed - cannot calculate loudness")]
    NoDataProcessed,

    /// Insufficient data for measurement.
    #[error("Insufficient data for {measurement}: need at least {required_ms} ms, have {available_ms} ms")]
    InsufficientData {
        /// The type of measurement attempted.
        measurement: &'static str,
        /// Required duration in milliseconds.
        required_ms: u32,
        /// Available duration in milliseconds.
        available_ms: u32,
    },

    /// Filter initialization failed.
    #[error("Failed to initialize K-weighting filter: {reason}")]
    FilterInitError {
        /// The reason for the initialization failure.
        reason: String,
    },

    /// Sample format conversion error.
    #[error("Sample format not supported: {format}")]
    UnsupportedFormat {
        /// The unsupported format name.
        format: String,
    },

    /// Gain calculation overflow.
    #[error("Gain calculation overflow: loudness difference too large")]
    GainOverflow,

    /// Buffer size mismatch.
    #[error("Buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch {
        /// Expected buffer size.
        expected: usize,
        /// Actual buffer size provided.
        actual: usize,
    },
}

/// Result type for loudness operations.
pub type Result<T> = std::result::Result<T, LoudnessError>;

impl LoudnessError {
    /// Create an invalid sample rate error.
    pub fn invalid_sample_rate(rate: u32) -> Self {
        LoudnessError::InvalidSampleRate { rate }
    }

    /// Create an invalid channel count error.
    pub fn invalid_channel_count(count: u32) -> Self {
        LoudnessError::InvalidChannelCount { count }
    }

    /// Create an invalid target loudness error.
    pub fn invalid_target_loudness(lufs: f64) -> Self {
        LoudnessError::InvalidTargetLoudness { lufs }
    }

    /// Create an invalid true peak limit error.
    pub fn invalid_true_peak_limit(dbtp: f64) -> Self {
        LoudnessError::InvalidTruePeakLimit { dbtp }
    }

    /// Create an insufficient data error.
    pub fn insufficient_data(measurement: &'static str, required_ms: u32, available_ms: u32) -> Self {
        LoudnessError::InsufficientData {
            measurement,
            required_ms,
            available_ms,
        }
    }

    /// Create a filter initialization error.
    pub fn filter_init_error(reason: impl Into<String>) -> Self {
        LoudnessError::FilterInitError {
            reason: reason.into(),
        }
    }

    /// Create an unsupported format error.
    pub fn unsupported_format(format: impl Into<String>) -> Self {
        LoudnessError::UnsupportedFormat {
            format: format.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = LoudnessError::invalid_sample_rate(100);
        assert_eq!(
            err.to_string(),
            "Invalid sample rate: 100 Hz (must be >= 8000 Hz)"
        );

        let err = LoudnessError::invalid_channel_count(0);
        assert_eq!(
            err.to_string(),
            "Invalid channel count: 0 (must be between 1 and 8)"
        );

        let err = LoudnessError::invalid_target_loudness(-80.0);
        assert_eq!(
            err.to_string(),
            "Invalid target loudness: -80 LUFS (must be between -70 and 0)"
        );
    }

    #[test]
    fn test_insufficient_data_error() {
        let err = LoudnessError::insufficient_data("momentary loudness", 400, 200);
        assert_eq!(
            err.to_string(),
            "Insufficient data for momentary loudness: need at least 400 ms, have 200 ms"
        );
    }

    #[test]
    fn test_error_is_clone() {
        let err = LoudnessError::NoDataProcessed;
        let err2 = err.clone();
        assert_eq!(err.to_string(), err2.to_string());
    }
}
