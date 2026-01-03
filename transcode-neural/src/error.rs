//! Neural processing errors

use thiserror::Error;

/// Errors from neural processing
#[derive(Error, Debug)]
pub enum NeuralError {
    /// Model loading failed
    #[error("model load error: {0}")]
    ModelLoad(String),

    /// Inference error
    #[error("inference error: {0}")]
    Inference(String),

    /// Invalid input data
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// ONNX feature not enabled
    #[error("ONNX feature not enabled")]
    OnnxNotEnabled,

    /// GPU error
    #[error("GPU error: {0}")]
    Gpu(String),
}
