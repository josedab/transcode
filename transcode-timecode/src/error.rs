//! Error types for timecode operations.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Result type for timecode operations.
pub type Result<T> = std::result::Result<T, TimecodeError>;

/// Errors that can occur during timecode operations.
#[derive(Debug, Clone, Error, Serialize, Deserialize, PartialEq, Eq)]
pub enum TimecodeError {
    /// Invalid timecode format in string.
    #[error("Invalid timecode format: {message}")]
    InvalidFormat {
        /// Description of the format error.
        message: String,
    },

    /// Invalid timecode component value.
    #[error("Invalid timecode component: {component} = {value} (max {max})")]
    InvalidComponent {
        /// Name of the invalid component (hours, minutes, seconds, frames).
        component: String,
        /// The invalid value that was provided.
        value: u32,
        /// The maximum allowed value for this component.
        max: u32,
    },

    /// Invalid frame rate.
    #[error("Invalid frame rate: {numerator}/{denominator}")]
    InvalidFrameRate {
        /// Frame rate numerator.
        numerator: u32,
        /// Frame rate denominator.
        denominator: u32,
    },

    /// Unsupported frame rate for operation.
    #[error("Unsupported frame rate for {operation}: {frame_rate}")]
    UnsupportedFrameRate {
        /// The operation that doesn't support this frame rate.
        operation: String,
        /// String representation of the unsupported frame rate.
        frame_rate: String,
    },

    /// Overflow during timecode arithmetic.
    #[error("Timecode overflow")]
    Overflow,

    /// Underflow during timecode arithmetic.
    #[error("Timecode underflow")]
    Underflow,

    /// Drop-frame timecode error.
    #[error("Drop-frame error: {message}")]
    DropFrameError {
        /// Description of the drop-frame error.
        message: String,
    },

    /// LTC encoding/decoding error.
    #[error("LTC error: {message}")]
    LtcError {
        /// Description of the LTC error.
        message: String,
    },

    /// VITC encoding/decoding error.
    #[error("VITC error: {message}")]
    VitcError {
        /// Description of the VITC error.
        message: String,
    },

    /// Frame rate mismatch in operation.
    #[error("Frame rate mismatch: {left} vs {right}")]
    FrameRateMismatch {
        /// String representation of the left operand's frame rate.
        left: String,
        /// String representation of the right operand's frame rate.
        right: String,
    },
}

impl TimecodeError {
    /// Create an invalid format error.
    pub fn invalid_format(message: impl Into<String>) -> Self {
        Self::InvalidFormat {
            message: message.into(),
        }
    }

    /// Create an invalid component error.
    pub fn invalid_component(component: impl Into<String>, value: u32, max: u32) -> Self {
        Self::InvalidComponent {
            component: component.into(),
            value,
            max,
        }
    }

    /// Create an invalid frame rate error.
    pub fn invalid_frame_rate(numerator: u32, denominator: u32) -> Self {
        Self::InvalidFrameRate {
            numerator,
            denominator,
        }
    }

    /// Create an unsupported frame rate error.
    pub fn unsupported_frame_rate(
        operation: impl Into<String>,
        frame_rate: impl Into<String>,
    ) -> Self {
        Self::UnsupportedFrameRate {
            operation: operation.into(),
            frame_rate: frame_rate.into(),
        }
    }

    /// Create a drop-frame error.
    pub fn drop_frame(message: impl Into<String>) -> Self {
        Self::DropFrameError {
            message: message.into(),
        }
    }

    /// Create an LTC error.
    pub fn ltc(message: impl Into<String>) -> Self {
        Self::LtcError {
            message: message.into(),
        }
    }

    /// Create a VITC error.
    pub fn vitc(message: impl Into<String>) -> Self {
        Self::VitcError {
            message: message.into(),
        }
    }

    /// Create a frame rate mismatch error.
    pub fn frame_rate_mismatch(left: impl Into<String>, right: impl Into<String>) -> Self {
        Self::FrameRateMismatch {
            left: left.into(),
            right: right.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = TimecodeError::invalid_format("missing separator");
        assert_eq!(
            err.to_string(),
            "Invalid timecode format: missing separator"
        );

        let err = TimecodeError::invalid_component("hours", 25, 23);
        assert_eq!(
            err.to_string(),
            "Invalid timecode component: hours = 25 (max 23)"
        );

        let err = TimecodeError::Overflow;
        assert_eq!(err.to_string(), "Timecode overflow");
    }

    #[test]
    fn test_error_serialization() {
        let err = TimecodeError::invalid_format("test error");
        let json = serde_json::to_string(&err).unwrap();
        let decoded: TimecodeError = serde_json::from_str(&json).unwrap();
        assert_eq!(err, decoded);
    }
}
