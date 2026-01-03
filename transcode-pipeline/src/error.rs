//! Pipeline error types.

use thiserror::Error;
use transcode_core::error::{CodecError, Error as CoreError};

/// Track type for pipeline streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineTrackType {
    Video,
    Audio,
    Subtitle,
    Data,
    Unknown,
}

/// Pipeline error type.
#[derive(Error, Debug)]
pub enum PipelineError {
    /// Core error.
    #[error("Core error: {0}")]
    Core(#[from] CoreError),

    /// Codec error.
    #[error("Codec error: {0}")]
    Codec(#[from] CodecError),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Node not found.
    #[error("Node not found: {0}")]
    NodeNotFound(String),

    /// Invalid node connection.
    #[error("Invalid node connection: {0}")]
    InvalidConnection(String),

    /// Pipeline not initialized.
    #[error("Pipeline not initialized")]
    NotInitialized,

    /// Pipeline already running.
    #[error("Pipeline already running")]
    AlreadyRunning,

    /// Stream not found.
    #[error("Stream {0} not found")]
    StreamNotFound(usize),

    /// Unsupported track type.
    #[error("Unsupported track type: {0:?}")]
    UnsupportedTrackType(PipelineTrackType),

    /// No decoder for stream.
    #[error("No decoder for stream {0}")]
    NoDecoder(usize),

    /// No encoder for stream.
    #[error("No encoder for stream {0}")]
    NoEncoder(usize),

    /// Synchronization error.
    #[error("Synchronization error: {0}")]
    Sync(String),

    /// End of stream.
    #[error("End of stream")]
    EndOfStream,

    /// Pipeline aborted.
    #[error("Pipeline aborted: {0}")]
    Aborted(String),
}

/// Pipeline result type.
pub type Result<T> = std::result::Result<T, PipelineError>;
