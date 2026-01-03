//! Bitrate ladder generation

use crate::PerTitleConfig;

/// Ladder generator for ABR streaming
pub struct LadderGenerator {
    config: PerTitleConfig,
}

impl LadderGenerator {
    /// Create a new ladder generator
    pub fn new(config: PerTitleConfig) -> Self {
        Self { config }
    }

    /// Get configuration
    pub fn config(&self) -> &PerTitleConfig {
        &self.config
    }
}
