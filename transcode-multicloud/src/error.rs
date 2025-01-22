//! Error types for multi-cloud operations.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum MultiCloudError {
    #[error("Provider error: {provider} - {message}")]
    ProviderError { provider: String, message: String },

    #[error("No providers available")]
    NoProvidersAvailable,

    #[error("Job not found: {0}")]
    JobNotFound(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Authentication error: {0}")]
    AuthError(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, MultiCloudError>;
