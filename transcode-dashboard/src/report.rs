//! Report generation in various formats.

use serde::{Deserialize, Serialize};

/// Supported report output formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    Html,
    Json,
    Csv,
}

/// Configuration for report generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportConfig {
    pub title: String,
    pub include_per_frame: bool,
    pub include_charts: bool,
    pub include_summary: bool,
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            title: "Quality Report".into(),
            include_per_frame: true,
            include_charts: true,
            include_summary: true,
        }
    }
}

/// A generated quality report.
#[derive(Debug, Clone)]
pub struct QualityReport {
    pub format: ReportFormat,
    pub content: String,
}

impl QualityReport {
    pub fn new(format: ReportFormat, content: String) -> Self {
        Self { format, content }
    }

    pub fn content_type(&self) -> &'static str {
        match self.format {
            ReportFormat::Html => "text/html",
            ReportFormat::Json => "application/json",
            ReportFormat::Csv => "text/csv",
        }
    }

    pub fn file_extension(&self) -> &'static str {
        match self.format {
            ReportFormat::Html => "html",
            ReportFormat::Json => "json",
            ReportFormat::Csv => "csv",
        }
    }
}
