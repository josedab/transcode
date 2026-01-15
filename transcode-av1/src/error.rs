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
        use transcode_core::error::{Av1ErrorKind, CodecError};

        let kind = match err {
            Av1Error::InvalidConfig(msg) => Av1ErrorKind::InvalidConfig(msg),
            Av1Error::EncoderError(msg) => Av1ErrorKind::EncoderError(msg),
            Av1Error::DecoderError(msg) => Av1ErrorKind::DecoderError(msg),
            Av1Error::InvalidFrame(msg) => Av1ErrorKind::InvalidFrame(msg),
            Av1Error::RateControlError(msg) => Av1ErrorKind::RateControlError(msg),
            Av1Error::NeedsMoreFrames => Av1ErrorKind::NeedsMoreFrames,
            Av1Error::NeedsMoreData => Av1ErrorKind::NeedsMoreData,
            Av1Error::ResourceExhausted(msg) => {
                return transcode_core::Error::ResourceExhausted(msg);
            }
            Av1Error::EndOfStream => {
                return transcode_core::Error::EndOfStream;
            }
            Av1Error::Io(e) => {
                return transcode_core::Error::Io(e);
            }
        };

        transcode_core::Error::Codec(CodecError::Av1(kind))
    }
}
