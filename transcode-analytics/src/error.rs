//! Analytics errors

use thiserror::Error;

/// Errors from analytics operations
#[derive(Error, Debug)]
pub enum AnalyticsError {
    /// No data available
    #[error("no data available")]
    NoData,

    /// Invalid configuration
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Calculation error
    #[error("calculation error: {0}")]
    Calculation(String),
}
