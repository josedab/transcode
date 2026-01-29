use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Invalid time range: {start}s to {end}s")]
    InvalidTimeRange { start: f64, end: f64 },

    #[error("Clip '{name}' not found")]
    ClipNotFound { name: String },

    #[error("Timeline validation failed: {message}")]
    Validation { message: String },

    #[error("Frame {frame} is out of range (max: {max})")]
    FrameOutOfRange { frame: u64, max: u64 },

    #[error("Transition at index {index} is invalid: {message}")]
    InvalidTransition { index: usize, message: String },

    #[error("EDL parse error at line {line}: {message}")]
    EdlParse { line: usize, message: String },

    #[error("Incompatible tracks: {message}")]
    IncompatibleTracks { message: String },

    #[error(transparent)]
    Core(#[from] transcode_core::error::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
