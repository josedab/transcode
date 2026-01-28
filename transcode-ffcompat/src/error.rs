use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Missing argument value for '{flag}'")]
    MissingValue { flag: String },

    #[error("Invalid argument value for '{flag}': {message}")]
    InvalidValue { flag: String, message: String },

    #[error("No input file specified (missing -i)")]
    NoInput,

    #[error("No output file specified")]
    NoOutput,

    #[error("Unknown codec '{name}' — no transcode equivalent")]
    UnknownCodec { name: String },

    #[error("Parse error: {message}")]
    Parse { message: String },

    #[error(transparent)]
    Core(#[from] transcode_core::error::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
