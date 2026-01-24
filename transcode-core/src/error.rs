//! Error types for the Transcode library.
//!
//! This module provides a comprehensive error hierarchy for all components of the library.

use thiserror::Error;

/// Main error type for the Transcode library.
#[derive(Error, Debug)]
pub enum Error {
    /// Container format errors (demuxing/muxing).
    #[error("Container error: {0}")]
    Container(#[from] ContainerError),

    /// Codec errors (encoding/decoding).
    #[error("Codec error: {0}")]
    Codec(#[from] CodecError),

    /// Bitstream parsing errors.
    #[error("Bitstream error: {0}")]
    Bitstream(#[from] BitstreamError),

    /// I/O errors.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid parameter provided.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Unsupported feature or format.
    #[error("Unsupported: {0}")]
    Unsupported(String),

    /// Resource exhausted (memory, buffers, etc.).
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    /// Operation was cancelled.
    #[error("Operation cancelled")]
    Cancelled,

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// End of stream reached.
    #[error("End of stream")]
    EndOfStream,

    /// Buffer too small for operation.
    #[error("Buffer too small: need {needed} bytes, have {available}")]
    BufferTooSmall { needed: usize, available: usize },

    /// Error with additional context.
    #[error("{context}: {source}")]
    WithContext {
        /// The underlying error.
        #[source]
        source: Box<Error>,
        /// Additional context about where/why the error occurred.
        context: String,
    },
}

/// Container format errors.
#[derive(Error, Debug)]
pub enum ContainerError {
    /// Invalid or corrupted container structure.
    #[error("Invalid container structure: {0}")]
    InvalidStructure(String),

    /// Unknown or unsupported container format.
    #[error("Unknown container format")]
    UnknownFormat,

    /// Missing required atom/box/element.
    #[error("Missing required element: {0}")]
    MissingElement(String),

    /// Invalid atom/box size.
    #[error("Invalid element size at offset {offset}: {message}")]
    InvalidSize { offset: u64, message: String },

    /// Recursion limit exceeded during parsing.
    #[error("Recursion limit exceeded at depth {depth}")]
    RecursionLimit { depth: u32 },

    /// Timeout during parsing operation.
    #[error("Parsing timeout exceeded")]
    Timeout,

    /// Stream not found in container.
    #[error("Stream {index} not found")]
    StreamNotFound { index: u32 },

    /// Seek operation failed.
    #[error("Seek failed: {0}")]
    SeekFailed(String),

    /// Track configuration error.
    #[error("Track configuration error: {0}")]
    TrackConfig(String),

    /// Generic container error message.
    #[error("{0}")]
    Other(String),
}

impl From<String> for ContainerError {
    fn from(s: String) -> Self {
        ContainerError::Other(s)
    }
}

impl From<&str> for ContainerError {
    fn from(s: &str) -> Self {
        ContainerError::Other(s.to_string())
    }
}

/// Codec errors.
#[derive(Error, Debug)]
pub enum CodecError {
    /// Unsupported codec profile.
    #[error("Unsupported profile: {0}")]
    UnsupportedProfile(String),

    /// Unsupported codec level.
    #[error("Unsupported level: {0}")]
    UnsupportedLevel(String),

    /// Bitstream corruption detected.
    #[error("Bitstream corruption at offset {offset}")]
    BitstreamCorruption { offset: u64 },

    /// Missing reference frame.
    #[error("Missing reference frame: {frame_num}")]
    MissingReference { frame_num: u32 },

    /// Decoder not initialized.
    #[error("Decoder not initialized")]
    NotInitialized,

    /// Encoder configuration error.
    #[error("Encoder configuration error: {0}")]
    EncoderConfig(String),

    /// Decoder configuration error.
    #[error("Decoder configuration error: {0}")]
    DecoderConfig(String),

    /// Invalid NAL unit.
    #[error("Invalid NAL unit: {0}")]
    InvalidNalUnit(String),

    /// Invalid parameter set.
    #[error("Invalid parameter set: {0}")]
    InvalidParameterSet(String),

    /// Slice parsing error.
    #[error("Slice error: {0}")]
    SliceError(String),

    /// Frame dimensions exceed limits.
    #[error("Frame dimensions {width}x{height} exceed maximum {max_width}x{max_height}")]
    DimensionsExceeded {
        width: u32,
        height: u32,
        max_width: u32,
        max_height: u32,
    },

    /// Resource limit exceeded (parameter sets, reference frames, etc.).
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    /// AV1-specific error.
    #[error("AV1: {0}")]
    Av1(Av1ErrorKind),

    /// Opus-specific error.
    #[error("Opus: {0}")]
    Opus(OpusErrorKind),

    /// Generic codec error message.
    #[error("{0}")]
    Other(String),
}

/// AV1-specific error variants.
///
/// These preserve type information for AV1 codec errors, enabling
/// programmatic error handling and recovery.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum Av1ErrorKind {
    /// Invalid AV1 configuration.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Encoder error during AV1 encoding.
    #[error("encoder error: {0}")]
    EncoderError(String),

    /// Decoder error during AV1 decoding.
    #[error("decoder error: {0}")]
    DecoderError(String),

    /// Invalid frame data for AV1.
    #[error("invalid frame: {0}")]
    InvalidFrame(String),

    /// AV1 rate control error.
    #[error("rate control error: {0}")]
    RateControlError(String),

    /// Encoder needs more frames before producing output.
    #[error("encoder needs more frames")]
    NeedsMoreFrames,

    /// Decoder needs more data before producing output.
    #[error("decoder needs more data")]
    NeedsMoreData,
}

/// Opus-specific error variants.
///
/// These preserve type information for Opus codec errors, enabling
/// programmatic error handling and recovery.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum OpusErrorKind {
    /// Invalid Opus packet structure.
    #[error("invalid packet: {0}")]
    InvalidPacket(String),

    /// Invalid TOC (Table of Contents) byte.
    #[error("invalid TOC byte: 0x{0:02x}")]
    InvalidToc(u8),

    /// Invalid Opus sample rate.
    #[error("invalid sample rate: {0} Hz")]
    InvalidSampleRate(u32),

    /// Invalid Opus channel count.
    #[error("invalid channel count: {0}")]
    InvalidChannels(u8),

    /// Invalid frame size for Opus.
    #[error("invalid frame size: {0} samples")]
    InvalidFrameSize(usize),

    /// Unsupported Opus configuration.
    #[error("unsupported config: {0}")]
    UnsupportedConfig(String),

    /// Range coder error in Opus bitstream.
    #[error("range coder error: {0}")]
    RangeCoder(String),

    /// SILK layer decoder error.
    #[error("SILK decoder error: {0}")]
    SilkDecoder(String),

    /// CELT layer decoder error.
    #[error("CELT decoder error: {0}")]
    CeltDecoder(String),

    /// Opus encoder configuration error.
    #[error("encoder config: {0}")]
    EncoderConfig(String),

    /// Packet loss concealment failed.
    #[error("PLC failed: {0}")]
    PlcFailed(String),

    /// Bitstream corruption detected.
    #[error("bitstream corruption at offset {0}")]
    BitstreamCorruption(usize),
}

impl From<String> for CodecError {
    fn from(s: String) -> Self {
        CodecError::Other(s)
    }
}

impl From<&str> for CodecError {
    fn from(s: &str) -> Self {
        CodecError::Other(s.to_string())
    }
}

/// Bitstream parsing errors.
#[derive(Error, Debug)]
pub enum BitstreamError {
    /// Unexpected end of bitstream.
    #[error("Unexpected end of bitstream")]
    UnexpectedEnd,

    /// Invalid start code.
    #[error("Invalid start code at offset {offset}")]
    InvalidStartCode { offset: u64 },

    /// Invalid syntax element value.
    #[error("Invalid syntax element: {element} = {value}")]
    InvalidSyntax { element: String, value: i64 },

    /// Exp-Golomb decoding error.
    #[error("Exp-Golomb decoding error: value too large")]
    ExpGolombOverflow,

    /// Bit alignment error.
    #[error("Bit alignment error")]
    AlignmentError,

    /// Generic bitstream error message.
    #[error("{0}")]
    Other(String),
}

impl From<String> for BitstreamError {
    fn from(s: String) -> Self {
        BitstreamError::Other(s)
    }
}

impl From<&str> for BitstreamError {
    fn from(s: &str) -> Self {
        BitstreamError::Other(s.to_string())
    }
}

/// Result type alias using our Error type.
pub type Result<T> = std::result::Result<T, Error>;

/// Extension trait for adding context to errors.
///
/// This trait allows adding descriptive context to errors as they propagate
/// up the call stack, making debugging easier.
///
/// # Example
///
/// ```
/// use transcode_core::error::{Result, ErrorContext};
///
/// fn read_config() -> Result<()> {
///     // ... operation that might fail ...
///     Ok(())
/// }
///
/// fn process() -> Result<()> {
///     read_config().context("reading configuration file")?;
///     Ok(())
/// }
/// ```
pub trait ErrorContext<T> {
    /// Add context to an error.
    fn context(self, msg: impl Into<String>) -> Result<T>;

    /// Add lazily-evaluated context to an error.
    fn with_context<F: FnOnce() -> String>(self, f: F) -> Result<T>;
}

impl<T> ErrorContext<T> for Result<T> {
    fn context(self, msg: impl Into<String>) -> Result<T> {
        self.map_err(|e| Error::WithContext {
            source: Box::new(e),
            context: msg.into(),
        })
    }

    fn with_context<F: FnOnce() -> String>(self, f: F) -> Result<T> {
        self.map_err(|e| Error::WithContext {
            source: Box::new(e),
            context: f(),
        })
    }
}

impl<T> ErrorContext<T> for std::result::Result<T, std::io::Error> {
    fn context(self, msg: impl Into<String>) -> Result<T> {
        self.map_err(|e| Error::WithContext {
            source: Box::new(Error::Io(e)),
            context: msg.into(),
        })
    }

    fn with_context<F: FnOnce() -> String>(self, f: F) -> Result<T> {
        self.map_err(|e| Error::WithContext {
            source: Box::new(Error::Io(e)),
            context: f(),
        })
    }
}

/// Trait for errors that can provide actionable suggestions.
///
/// This trait enables errors to provide helpful suggestions to users
/// about how to resolve the issue.
///
/// # Example
///
/// ```
/// use transcode_core::error::{Error, ErrorSuggestion};
///
/// let err = Error::Unsupported("VP9 codec not available".into());
/// if let Some(suggestion) = err.suggestion() {
///     println!("Suggestion: {}", suggestion);
/// }
/// ```
pub trait ErrorSuggestion {
    /// Get a suggestion for how to resolve this error.
    ///
    /// Returns `None` if no specific suggestion is available.
    fn suggestion(&self) -> Option<&'static str>;
}

impl ErrorSuggestion for Error {
    fn suggestion(&self) -> Option<&'static str> {
        match self {
            Error::Io(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Some("Check that the file path is correct and the file exists")
            }
            Error::Io(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                Some("Check file permissions or try running with elevated privileges")
            }
            Error::Io(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                Some("Use --overwrite flag to replace existing output file")
            }
            Error::Unsupported(msg) if msg.contains("codec") => {
                Some("Run 'transcode codecs' to see available codecs")
            }
            Error::Unsupported(msg) if msg.contains("format") => {
                Some("Check that the input file is a supported format (MP4, MKV, WebM)")
            }
            Error::InvalidParameter(msg) if msg.contains("resolution") => {
                Some("Resolution should be in format WIDTHxHEIGHT (e.g., 1920x1080)")
            }
            Error::InvalidParameter(msg) if msg.contains("bitrate") => {
                Some("Bitrate should be a positive number, optionally with k/M suffix (e.g., 5M, 2500k)")
            }
            Error::Config(msg) if msg.contains("input") => {
                Some("Specify an input file with -i <file>")
            }
            Error::Config(msg) if msg.contains("output") => {
                Some("Specify an output file with -o <file>")
            }
            Error::Container(e) => e.suggestion(),
            Error::Codec(e) => e.suggestion(),
            Error::Bitstream(e) => e.suggestion(),
            Error::ResourceExhausted(_) => {
                Some("Try reducing resolution, frame rate, or quality settings")
            }
            Error::WithContext { source, .. } => source.suggestion(),
            _ => None,
        }
    }
}

impl ErrorSuggestion for ContainerError {
    fn suggestion(&self) -> Option<&'static str> {
        match self {
            ContainerError::UnknownFormat => {
                Some("Check that the file is a valid media container (MP4, MKV, WebM)")
            }
            ContainerError::InvalidStructure(_) => {
                Some("The file may be corrupted - try re-downloading or using a different source")
            }
            ContainerError::MissingElement(elem) if elem.contains("moov") => {
                Some("MP4 file appears truncated - try running 'qt-faststart' to repair")
            }
            ContainerError::SeekFailed(_) => {
                Some("Try using a file path instead of stdin for seekable input")
            }
            _ => None,
        }
    }
}

impl ErrorSuggestion for CodecError {
    fn suggestion(&self) -> Option<&'static str> {
        match self {
            CodecError::UnsupportedProfile(p) if p.contains("High 10") => {
                Some("10-bit profiles require hardware support - try converting to 8-bit")
            }
            CodecError::UnsupportedProfile(_) => {
                Some("Try a different encoder profile (e.g., --profile baseline)")
            }
            CodecError::BitstreamCorruption { .. } => {
                Some("Input file may be corrupted - try re-encoding from source")
            }
            CodecError::NotInitialized => {
                Some("Internal error - please report this issue")
            }
            CodecError::EncoderConfig(_) => {
                Some("Run 'transcode presets' to see valid encoding configurations")
            }
            CodecError::DimensionsExceeded { .. } => {
                Some("Try scaling to a smaller resolution with --scale")
            }
            _ => None,
        }
    }
}

impl ErrorSuggestion for BitstreamError {
    fn suggestion(&self) -> Option<&'static str> {
        match self {
            BitstreamError::UnexpectedEnd => {
                Some("File appears truncated - check if download completed")
            }
            BitstreamError::InvalidStartCode { .. } => {
                Some("Input file may not be in the expected format")
            }
            _ => None,
        }
    }
}

impl Error {
    /// Create an invalid parameter error.
    pub fn invalid_param(msg: impl Into<String>) -> Self {
        Error::InvalidParameter(msg.into())
    }

    /// Create an unsupported error.
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Error::Unsupported(msg.into())
    }

    /// Check if this is an end-of-stream error.
    #[must_use]
    pub fn is_eof(&self) -> bool {
        match self {
            Error::EndOfStream => true,
            Error::WithContext { source, .. } => source.is_eof(),
            _ => false,
        }
    }

    /// Check if this error is recoverable (can continue processing).
    #[must_use]
    pub fn is_recoverable(&self) -> bool {
        match self {
            Error::Codec(CodecError::BitstreamCorruption { .. })
            | Error::Codec(CodecError::MissingReference { .. })
            | Error::Bitstream(BitstreamError::InvalidSyntax { .. }) => true,
            Error::WithContext { source, .. } => source.is_recoverable(),
            _ => false,
        }
    }

    /// Get the root cause of this error, unwrapping any context layers.
    #[must_use]
    pub fn root_cause(&self) -> &Error {
        match self {
            Error::WithContext { source, .. } => source.root_cause(),
            _ => self,
        }
    }

    /// Iterate over the chain of context messages.
    pub fn context_chain(&self) -> Vec<&str> {
        let mut contexts = Vec::new();
        self.collect_contexts(&mut contexts);
        contexts
    }

    fn collect_contexts<'a>(&'a self, contexts: &mut Vec<&'a str>) {
        if let Error::WithContext { source, context } = self {
            contexts.push(context.as_str());
            source.collect_contexts(contexts);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::InvalidParameter("test parameter".into());
        assert_eq!(err.to_string(), "Invalid parameter: test parameter");
    }

    #[test]
    fn test_container_error_conversion() {
        let container_err = ContainerError::UnknownFormat;
        let err: Error = container_err.into();
        assert!(matches!(err, Error::Container(ContainerError::UnknownFormat)));
    }

    #[test]
    fn test_is_eof() {
        assert!(Error::EndOfStream.is_eof());
        assert!(!Error::Cancelled.is_eof());
    }

    #[test]
    fn test_is_recoverable() {
        let recoverable = Error::Codec(CodecError::BitstreamCorruption { offset: 0 });
        assert!(recoverable.is_recoverable());

        let not_recoverable = Error::EndOfStream;
        assert!(!not_recoverable.is_recoverable());
    }

    #[test]
    fn test_error_context() {
        let result: Result<()> = Err(Error::EndOfStream);
        let with_context = result.context("reading from input file");

        assert!(with_context.is_err());
        let err = with_context.unwrap_err();
        assert!(err.to_string().contains("reading from input file"));
        assert!(err.to_string().contains("End of stream"));
    }

    #[test]
    fn test_error_with_context_lazy() {
        let result: Result<()> = Err(Error::EndOfStream);
        let with_context = result.with_context(|| format!("processing frame {}", 42));

        let err = with_context.unwrap_err();
        assert!(err.to_string().contains("processing frame 42"));
    }

    #[test]
    fn test_error_context_chain() {
        let base_err: Result<()> = Err(Error::EndOfStream);
        let err = base_err
            .context("inner operation")
            .context("middle layer")
            .context("outer operation")
            .unwrap_err();

        let chain = err.context_chain();
        assert_eq!(chain.len(), 3);
        assert_eq!(chain[0], "outer operation");
        assert_eq!(chain[1], "middle layer");
        assert_eq!(chain[2], "inner operation");
    }

    #[test]
    fn test_error_root_cause() {
        let base_err: Result<()> = Err(Error::EndOfStream);
        let wrapped = base_err
            .context("layer 1")
            .context("layer 2")
            .unwrap_err();

        let root = wrapped.root_cause();
        assert!(matches!(root, Error::EndOfStream));
    }

    #[test]
    fn test_is_eof_through_context() {
        let base_err: Result<()> = Err(Error::EndOfStream);
        let wrapped = base_err.context("reading packet").unwrap_err();

        assert!(wrapped.is_eof());
    }

    #[test]
    fn test_is_recoverable_through_context() {
        let base_err: Result<()> = Err(Error::Codec(CodecError::BitstreamCorruption { offset: 100 }));
        let wrapped = base_err.context("decoding frame").unwrap_err();

        assert!(wrapped.is_recoverable());
    }

    #[test]
    fn test_error_context_preserves_source() {
        let base_err = Error::Codec(CodecError::MissingReference { frame_num: 5 });
        let wrapped = Error::WithContext {
            source: Box::new(base_err),
            context: "decoding P-frame".into(),
        };

        // std::error::Error source() should return the inner error
        use std::error::Error as StdError;
        assert!(wrapped.source().is_some());
    }
}
