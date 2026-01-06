//! AV1 error types.

use thiserror::Error;

/// Errors that can occur during AV1 encoding/decoding.
#[derive(Error, Debug)]
pub enum Av1Error {
    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Encoder error.
    #[error("Encoder error: {0}")]
    EncoderError(String),

    /// Decoder error.
    #[error("Decoder error: {0}")]
    DecoderError(String),

    /// Invalid frame data.
    #[error("Invalid frame: {0}")]
    InvalidFrame(String),

    /// Rate control error.
    #[error("Rate control error: {0}")]
    RateControlError(String),

    /// Resource exhausted.
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    /// Encoding not ready.
    #[error("Encoder needs more frames")]
    NeedsMoreFrames,

    /// Decoding not ready (need more data).
    #[error("Decoder needs more data")]
    NeedsMoreData,

    /// End of stream reached.
    #[error("End of stream")]
    EndOfStream,

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<Av1Error> for transcode_core::Error {
    fn from(err: Av1Error) -> Self {
        transcode_core::Error::Codec(transcode_core::error::CodecError::Other(err.to_string()))
    }
}
