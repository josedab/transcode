use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("No quality data recorded")]
    NoData,

    #[error("Frame {frame} out of range")]
    FrameOutOfRange { frame: u64 },

    #[error("Report generation failed: {message}")]
    Report { message: String },

    #[error("Export error: {message}")]
    Export { message: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Core(#[from] transcode_core::error::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
