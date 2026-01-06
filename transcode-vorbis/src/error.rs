//! Vorbis error types.

use thiserror::Error;

/// Vorbis codec errors.
#[derive(Error, Debug)]
pub enum VorbisError {
    /// Invalid Vorbis header.
    #[error("Invalid Vorbis header: {0}")]
    InvalidHeader(String),

    /// Invalid identification header.
    #[error("Invalid identification header: {0}")]
    InvalidIdHeader(String),

    /// Invalid comment header.
    #[error("Invalid comment header: {0}")]
    InvalidCommentHeader(String),

    /// Invalid setup header.
    #[error("Invalid setup header: {0}")]
    InvalidSetupHeader(String),

    /// Invalid codebook.
    #[error("Invalid codebook: {0}")]
    InvalidCodebook(String),

    /// Invalid floor configuration.
    #[error("Invalid floor configuration: {0}")]
    InvalidFloor(String),

    /// Invalid residue configuration.
    #[error("Invalid residue configuration: {0}")]
    InvalidResidue(String),

    /// Invalid mapping configuration.
    #[error("Invalid mapping configuration: {0}")]
    InvalidMapping(String),

    /// Invalid mode configuration.
    #[error("Invalid mode configuration: {0}")]
    InvalidMode(String),

    /// Unsupported sample rate.
    #[error("Unsupported sample rate: {0}")]
    UnsupportedSampleRate(u32),

    /// Unsupported channel count.
    #[error("Unsupported channel count: {0}")]
    UnsupportedChannels(u8),

    /// Bitstream error.
    #[error("Bitstream error: {0}")]
    BitstreamError(String),

    /// Decoder not initialized.
    #[error("Decoder not initialized")]
    NotInitialized,

    /// Encoder configuration error.
    #[error("Encoder configuration error: {0}")]
    ConfigError(String),

    /// Invalid audio data.
    #[error("Invalid audio data: {0}")]
    InvalidData(String),

    /// End of stream.
    #[error("End of stream")]
    EndOfStream,

    /// Need more data.
    #[error("Need more data")]
    NeedMoreData,
}

/// Vorbis result type.
pub type Result<T> = std::result::Result<T, VorbisError>;

impl From<VorbisError> for transcode_core::Error {
    fn from(err: VorbisError) -> Self {
        transcode_core::Error::Codec(transcode_core::error::CodecError::Other(err.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = VorbisError::InvalidHeader("bad magic".to_string());
        assert!(err.to_string().contains("bad magic"));
    }
}
