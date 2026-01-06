//! PCM codec error types.

use thiserror::Error;

/// PCM codec errors.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum PcmError {
    /// Invalid sample rate.
    #[error("Invalid sample rate: {0}")]
    InvalidSampleRate(u32),

    /// Invalid channel count.
    #[error("Invalid channel count: {0}")]
    InvalidChannelCount(u8),

    /// Buffer size mismatch.
    #[error("Buffer size {actual} is not aligned to sample size {expected}")]
    BufferSizeMismatch {
        /// Actual buffer size.
        actual: usize,
        /// Expected alignment.
        expected: usize,
    },

    /// Invalid PCM data.
    #[error("Invalid PCM data: {0}")]
    InvalidData(String),

    /// Unsupported format.
    #[error("Unsupported PCM format: {0}")]
    UnsupportedFormat(String),

    /// Conversion error.
    #[error("Conversion error: {0}")]
    ConversionError(String),

    /// Unexpected end of data.
    #[error("Unexpected end of data")]
    UnexpectedEndOfData,
}

/// PCM codec result type.
pub type Result<T> = std::result::Result<T, PcmError>;

impl From<PcmError> for transcode_core::Error {
    fn from(err: PcmError) -> Self {
        transcode_core::Error::Codec(transcode_core::error::CodecError::Other(err.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = PcmError::InvalidSampleRate(0);
        assert!(err.to_string().contains("Invalid sample rate"));

        let err = PcmError::InvalidChannelCount(0);
        assert!(err.to_string().contains("Invalid channel count"));
    }

    #[test]
    fn test_buffer_size_mismatch() {
        let err = PcmError::BufferSizeMismatch {
            actual: 101,
            expected: 4,
        };
        assert!(err.to_string().contains("101"));
        assert!(err.to_string().contains("4"));
    }
}
