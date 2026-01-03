//! Error types for audio resampling.

use thiserror::Error;

/// Result type for resampling operations.
pub type Result<T> = std::result::Result<T, ResampleError>;

/// Errors that can occur during resampling.
#[derive(Debug, Error)]
pub enum ResampleError {
    /// Invalid sample rate specified.
    #[error("Invalid sample rate: {rate} Hz (must be > 0)")]
    InvalidSampleRate { rate: u32 },

    /// Invalid channel count.
    #[error("Invalid channel count: {count} (must be > 0)")]
    InvalidChannelCount { count: usize },

    /// Invalid window size for sinc resampler.
    #[error("Invalid window size: {size} (must be > 0 and even)")]
    InvalidWindowSize { size: usize },

    /// Input buffer size mismatch.
    #[error("Input buffer size {actual} is not divisible by channel count {channels}")]
    BufferSizeMismatch { actual: usize, channels: usize },

    /// Resampling ratio too extreme.
    #[error("Resampling ratio {ratio} exceeds maximum supported ratio")]
    RatioTooExtreme { ratio: f64 },

    /// Internal processing error.
    #[error("Internal resampling error: {message}")]
    Internal { message: String },
}

impl ResampleError {
    /// Create an internal error with a message.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }
}
