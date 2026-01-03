//! Caption errors

use thiserror::Error;

/// Errors from caption processing
#[derive(Error, Debug)]
pub enum CaptionError {
    /// Model loading failed
    #[error("model load error: {0}")]
    ModelLoad(String),

    /// Model not loaded
    #[error("model not loaded")]
    ModelNotLoaded,

    /// Transcription error
    #[error("transcription error: {0}")]
    Transcription(String),

    /// Invalid audio
    #[error("invalid audio: {0}")]
    InvalidAudio(String),

    /// Whisper feature not enabled
    #[error("whisper feature not enabled")]
    WhisperNotEnabled,

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
