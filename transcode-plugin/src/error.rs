use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Plugin '{name}' already registered")]
    AlreadyRegistered { name: String },

    #[error("Plugin '{name}' not found")]
    NotFound { name: String },

    #[error("API version mismatch: plugin has v{plugin}, expected v{expected}")]
    ApiVersionMismatch { plugin: u32, expected: u32 },

    #[error("Failed to load plugin from {path}: {message}")]
    LoadFailed { path: String, message: String },

    #[error("Plugin initialization failed: {message}")]
    InitFailed { message: String },

    #[error("Invalid plugin binary: {message}")]
    InvalidBinary { message: String },

    #[error("Sandbox violation: {message}")]
    SandboxViolation { message: String },

    #[error("Plugin '{name}' returned error: {message}")]
    PluginError { name: String, message: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Core(#[from] transcode_core::error::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
