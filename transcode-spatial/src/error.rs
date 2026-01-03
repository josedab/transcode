//! Error types for spatial audio processing.
//!
//! This module provides comprehensive error types for all spatial audio operations
//! including channel layout conversion, ambisonics processing, binaural rendering,
//! and object-based audio handling.

use thiserror::Error;

/// Main error type for spatial audio operations.
#[derive(Error, Debug)]
pub enum SpatialError {
    /// Channel layout error.
    #[error("Channel layout error: {0}")]
    ChannelLayout(#[from] ChannelLayoutError),

    /// Ambisonics processing error.
    #[error("Ambisonics error: {0}")]
    Ambisonics(#[from] AmbisonicsError),

    /// Binaural rendering error.
    #[error("Binaural error: {0}")]
    Binaural(#[from] BinauralError),

    /// Object-based audio error.
    #[error("Object audio error: {0}")]
    ObjectAudio(#[from] ObjectAudioError),

    /// Atmos processing error.
    #[error("Atmos error: {0}")]
    Atmos(#[from] AtmosError),

    /// Renderer error.
    #[error("Renderer error: {0}")]
    Renderer(#[from] RendererError),

    /// Downmix error.
    #[error("Downmix error: {0}")]
    Downmix(#[from] DownmixError),

    /// Invalid parameter provided.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Unsupported feature or format.
    #[error("Unsupported: {0}")]
    Unsupported(String),

    /// Core transcode error.
    #[error("Core error: {0}")]
    Core(#[from] transcode_core::Error),
}

/// Channel layout errors.
#[derive(Error, Debug)]
pub enum ChannelLayoutError {
    /// Invalid channel count.
    #[error("Invalid channel count: {count}, expected {expected}")]
    InvalidChannelCount {
        /// Actual channel count.
        count: u32,
        /// Expected channel count.
        expected: u32,
    },

    /// Unknown channel position.
    #[error("Unknown channel position: {0}")]
    UnknownPosition(String),

    /// Incompatible layouts for conversion.
    #[error("Cannot convert from {from} to {to}")]
    IncompatibleLayouts {
        /// Source layout.
        from: String,
        /// Target layout.
        to: String,
    },

    /// Duplicate channel position in layout.
    #[error("Duplicate channel position: {0}")]
    DuplicatePosition(String),

    /// Missing required channel.
    #[error("Missing required channel: {0}")]
    MissingChannel(String),

    /// Invalid channel order.
    #[error("Invalid channel order: {0}")]
    InvalidOrder(String),
}

/// Ambisonics processing errors.
#[derive(Error, Debug)]
pub enum AmbisonicsError {
    /// Invalid ambisonics order.
    #[error("Invalid ambisonics order: {order}, must be >= 0")]
    InvalidOrder {
        /// The invalid order value.
        order: i32,
    },

    /// Channel count mismatch for ambisonics order.
    #[error("Channel count {count} does not match order {order} (expected {expected})")]
    ChannelCountMismatch {
        /// Actual channel count.
        count: u32,
        /// Ambisonics order.
        order: u32,
        /// Expected channel count for this order.
        expected: u32,
    },

    /// Invalid normalization scheme.
    #[error("Invalid normalization scheme: {0}")]
    InvalidNormalization(String),

    /// Invalid channel ordering.
    #[error("Invalid channel ordering: {0}")]
    InvalidOrdering(String),

    /// Decoder matrix not available.
    #[error("Decoder matrix not available for {order} order to {speakers} speakers")]
    DecoderNotAvailable {
        /// Ambisonics order.
        order: u32,
        /// Number of output speakers.
        speakers: u32,
    },

    /// HRTF not loaded for binaural conversion.
    #[error("HRTF not loaded for binaural conversion")]
    HrtfNotLoaded,
}

/// Binaural rendering errors.
#[derive(Error, Debug)]
pub enum BinauralError {
    /// HRTF loading failed.
    #[error("Failed to load HRTF: {0}")]
    HrtfLoadFailed(String),

    /// Invalid HRTF format.
    #[error("Invalid HRTF format: {0}")]
    InvalidHrtfFormat(String),

    /// Sample rate mismatch.
    #[error("Sample rate mismatch: HRTF is {hrtf_rate}Hz, input is {input_rate}Hz")]
    SampleRateMismatch {
        /// HRTF sample rate in Hz.
        hrtf_rate: u32,
        /// Input audio sample rate in Hz.
        input_rate: u32,
    },

    /// Convolution buffer error.
    #[error("Convolution buffer error: {0}")]
    ConvolutionError(String),

    /// Invalid head tracking data.
    #[error("Invalid head tracking data: {0}")]
    InvalidHeadTracking(String),

    /// Near-field compensation error.
    #[error("Near-field compensation error: {0}")]
    NearFieldError(String),
}

/// Object-based audio errors.
#[derive(Error, Debug)]
pub enum ObjectAudioError {
    /// Object not found.
    #[error("Object not found: {id}")]
    ObjectNotFound {
        /// Object ID that was not found.
        id: u32,
    },

    /// Invalid object position.
    #[error("Invalid object position: ({x}, {y}, {z})")]
    InvalidPosition {
        /// X coordinate.
        x: f32,
        /// Y coordinate.
        y: f32,
        /// Z coordinate.
        z: f32,
    },

    /// Maximum objects exceeded.
    #[error("Maximum objects exceeded: {count} > {max}")]
    MaxObjectsExceeded {
        /// Current object count.
        count: u32,
        /// Maximum allowed objects.
        max: u32,
    },

    /// Invalid object metadata.
    #[error("Invalid object metadata: {0}")]
    InvalidMetadata(String),

    /// Bed channel error.
    #[error("Bed channel error: {0}")]
    BedError(String),
}

/// Dolby Atmos processing errors.
#[derive(Error, Debug)]
pub enum AtmosError {
    /// Invalid ADM data.
    #[error("Invalid ADM data: {0}")]
    InvalidAdm(String),

    /// Invalid JOC metadata.
    #[error("Invalid JOC metadata: {0}")]
    InvalidJoc(String),

    /// Unsupported Atmos version.
    #[error("Unsupported Atmos version: {0}")]
    UnsupportedVersion(String),

    /// Object trajectory error.
    #[error("Object trajectory error: {0}")]
    TrajectoryError(String),

    /// Bed configuration error.
    #[error("Bed configuration error: {0}")]
    BedConfigError(String),

    /// Downmix error.
    #[error("Atmos downmix error: {0}")]
    DownmixError(String),
}

/// Renderer errors.
#[derive(Error, Debug)]
pub enum RendererError {
    /// Speaker configuration error.
    #[error("Speaker configuration error: {0}")]
    SpeakerConfig(String),

    /// VBAP error.
    #[error("VBAP error: {0}")]
    VbapError(String),

    /// Distance attenuation error.
    #[error("Distance attenuation error: {0}")]
    DistanceError(String),

    /// Room simulation error.
    #[error("Room simulation error: {0}")]
    RoomError(String),

    /// Output format error.
    #[error("Output format error: {0}")]
    OutputFormatError(String),

    /// Buffer size mismatch.
    #[error("Buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch {
        /// Expected buffer size.
        expected: usize,
        /// Actual buffer size.
        actual: usize,
    },
}

/// Downmix errors.
#[derive(Error, Debug)]
pub enum DownmixError {
    /// Invalid downmix matrix.
    #[error("Invalid downmix matrix: {0}")]
    InvalidMatrix(String),

    /// Unsupported downmix path.
    #[error("Unsupported downmix from {from} to {to}")]
    UnsupportedPath {
        /// Source format.
        from: String,
        /// Target format.
        to: String,
    },

    /// LFE handling error.
    #[error("LFE handling error: {0}")]
    LfeError(String),

    /// Dialog normalization error.
    #[error("Dialog normalization error: {0}")]
    DialogNormError(String),

    /// Coefficient error.
    #[error("Invalid downmix coefficient: {0}")]
    CoefficientError(String),
}

/// Result type alias for spatial audio operations.
pub type Result<T> = std::result::Result<T, SpatialError>;

impl SpatialError {
    /// Create an invalid parameter error.
    pub fn invalid_param(msg: impl Into<String>) -> Self {
        SpatialError::InvalidParameter(msg.into())
    }

    /// Create an unsupported error.
    pub fn unsupported(msg: impl Into<String>) -> Self {
        SpatialError::Unsupported(msg.into())
    }

    /// Check if this error is recoverable.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            SpatialError::ObjectAudio(ObjectAudioError::ObjectNotFound { .. })
                | SpatialError::Renderer(RendererError::BufferSizeMismatch { .. })
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = SpatialError::InvalidParameter("test parameter".into());
        assert_eq!(err.to_string(), "Invalid parameter: test parameter");
    }

    #[test]
    fn test_channel_layout_error_conversion() {
        let layout_err = ChannelLayoutError::InvalidChannelCount {
            count: 4,
            expected: 6,
        };
        let err: SpatialError = layout_err.into();
        assert!(matches!(
            err,
            SpatialError::ChannelLayout(ChannelLayoutError::InvalidChannelCount { .. })
        ));
    }

    #[test]
    fn test_is_recoverable() {
        let recoverable = SpatialError::ObjectAudio(ObjectAudioError::ObjectNotFound { id: 1 });
        assert!(recoverable.is_recoverable());

        let not_recoverable = SpatialError::InvalidParameter("test".into());
        assert!(!not_recoverable.is_recoverable());
    }

    #[test]
    fn test_ambisonics_error() {
        let err = AmbisonicsError::ChannelCountMismatch {
            count: 4,
            order: 2,
            expected: 9,
        };
        assert_eq!(
            err.to_string(),
            "Channel count 4 does not match order 2 (expected 9)"
        );
    }

    #[test]
    fn test_binaural_error() {
        let err = BinauralError::SampleRateMismatch {
            hrtf_rate: 48000,
            input_rate: 44100,
        };
        assert!(err.to_string().contains("48000"));
        assert!(err.to_string().contains("44100"));
    }
}
