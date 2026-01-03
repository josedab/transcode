//! Content analysis for per-title encoding

use crate::PerTitleConfig;

/// Advanced content analyzer
pub struct ContentAnalyzer {
    config: PerTitleConfig,
}

impl ContentAnalyzer {
    /// Create a new content analyzer
    pub fn new(config: PerTitleConfig) -> Self {
        Self { config }
    }

    /// Get configuration
    pub fn config(&self) -> &PerTitleConfig {
        &self.config
    }
}
