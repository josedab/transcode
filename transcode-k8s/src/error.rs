use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Kubernetes API error: {message}")]
    KubeApi { message: String },

    #[error("Resource not found: {kind}/{name} in namespace {namespace}")]
    NotFound {
        kind: String,
        name: String,
        namespace: String,
    },

    #[error("Reconciliation failed for {resource}: {message}")]
    Reconcile { resource: String, message: String },

    #[error("Scaling error: {message}")]
    Scaling { message: String },

    #[error("Configuration error: {message}")]
    Config { message: String },

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Distributed system error: {message}")]
    Distributed { message: String },

    #[error(transparent)]
    Core(#[from] transcode_core::error::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
