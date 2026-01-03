//! Per-title encoding errors

use thiserror::Error;

/// Errors from per-title encoding
#[derive(Error, Debug)]
pub enum PerTitleError {
    /// No frames provided for analysis
    #[error("no frames provided for analysis")]
    NoFrames,

    /// No valid resolutions for source
    #[error("no valid resolutions for source video")]
    NoValidResolutions,

    /// Quality metric calculation failed
    #[error("quality metric error: {0}")]
    QualityMetric(String),

    /// Content analysis failed
    #[error("content analysis error: {0}")]
    Analysis(String),

    /// Configuration error
    #[error("configuration error: {0}")]
    Config(String),

    /// Core error
    #[error("core error: {0}")]
    Core(#[from] transcode_core::error::Error),
}
