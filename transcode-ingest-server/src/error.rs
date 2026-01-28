use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Failed to bind to {address}: {message}")]
    Bind { address: String, message: String },

    #[error("Stream '{key}' not found")]
    StreamNotFound { key: String },

    #[error("Maximum stream limit ({max}) reached")]
    MaxStreams { max: usize },

    #[error("Protocol error ({protocol}): {message}")]
    Protocol { protocol: String, message: String },

    #[error("Transcoding pipeline error: {message}")]
    Pipeline { message: String },

    #[error("Output error: {message}")]
    Output { message: String },

    #[error("Configuration error: {message}")]
    Config { message: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Core(#[from] transcode_core::error::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
