//! Encoding optimization based on content analysis

use crate::{ContentComplexity, EncodingPreset, PerTitleConfig};

/// Encoding optimizer
pub struct EncodingOptimizer {
    _config: PerTitleConfig,
}

impl EncodingOptimizer {
    /// Create a new encoding optimizer
    pub fn new(config: PerTitleConfig) -> Self {
        Self { _config: config }
    }

    /// Recommend encoding preset based on complexity
    pub fn recommend_preset(&self, complexity: &ContentComplexity) -> EncodingPreset {
        match complexity.overall {
            c if c > 70.0 => EncodingPreset::Slow,
            c if c > 40.0 => EncodingPreset::Medium,
            _ => EncodingPreset::Fast,
        }
    }
}
