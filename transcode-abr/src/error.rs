use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Invalid content profile: {message}")]
    InvalidProfile { message: String },

    #[error("Optimization failed: {message}")]
    Optimization { message: String },

    #[error("Invalid configuration: {message}")]
    Config { message: String },

    #[error("Resolution {width}x{height} exceeds source {src_width}x{src_height}")]
    ResolutionExceedsSource {
        width: u32,
        height: u32,
        src_width: u32,
        src_height: u32,
    },

    #[error(transparent)]
    Core(#[from] transcode_core::error::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
