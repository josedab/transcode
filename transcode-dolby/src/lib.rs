//! Dolby Vision support for the transcode project.
//!
//! This crate provides comprehensive Dolby Vision support including:
//!
//! - **Profile support**: Profiles 4, 5, 7, 8, and 9 with level definitions
//! - **RPU parsing**: Reference Processing Unit NAL parsing and serialization
//! - **Metadata handling**: L1-L11 metadata structures and processing
//! - **Tone/gamut mapping**: Polynomial, MMR, and NLQ processing
//! - **Profile conversion**: Convert between profiles (e.g., Profile 7 to 8)
//! - **Stream extraction**: Extract RPUs and separate layers from HEVC streams
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use transcode_dolby::{DolbyConfig, DolbyVisionProfile};
//! use transcode_dolby::extractor::{DolbyVisionExtractor, extract_all_rpus};
//!
//! // Create a configuration
//! let config = DolbyConfig::new(DolbyVisionProfile::Profile8);
//!
//! // Extract RPUs from an HEVC stream
//! let rpus = extract_all_rpus(&hevc_data)?;
//!
//! // Process metadata
//! for rpu in &rpus {
//!     if let Some(l1) = &rpu.metadata.l1 {
//!         println!("Scene max: {} nits", l1.max_nits());
//!     }
//! }
//! ```
//!
//! # Profile Conversion
//!
//! ```rust,ignore
//! use transcode_dolby::converter::ProfileConverter;
//! use transcode_dolby::DolbyVisionProfile;
//!
//! // Convert Profile 7 (dual-layer) to Profile 8 (single-layer)
//! let converter = ProfileConverter::profile7_to_8();
//! let converted_rpu = converter.convert_rpu(&source_rpu)?;
//! ```
//!
//! # Tone Mapping
//!
//! ```rust,ignore
//! use transcode_dolby::mapping::ToneMapper;
//!
//! // Create a tone mapper for a 1000 nit display
//! let mapper = ToneMapper::for_1000_nits();
//! mapper.configure_from_rpu(&rpu)?;
//!
//! // Map a pixel
//! let (r, g, b) = mapper.map_pixel(r_in, g_in, b_in);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::unnecessary_lazy_evaluations)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::collapsible_if)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(clippy::vec_init_then_push)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::manual_find)]
#![allow(clippy::unnecessary_map_or)]
#![allow(unused_imports)]

pub mod converter;
pub mod error;
pub mod extractor;
pub mod mapping;
pub mod metadata;
pub mod profile;
pub mod rpu;

// Re-exports for convenience
pub use error::{DolbyError, Result};
pub use metadata::DolbyVisionMetadata;
pub use profile::{
    BaseCodec, DolbyVisionLevel, DolbyVisionProfile, EnhancementLayer, LevelConstraints,
    ProfileLevel, SubProfile, TransferFunction,
};
pub use rpu::Rpu;

/// Dolby Vision configuration.
///
/// Main configuration structure for Dolby Vision processing,
/// containing profile, level, and processing options.
#[derive(Debug, Clone)]
pub struct DolbyConfig {
    /// Target Dolby Vision profile.
    pub profile: DolbyVisionProfile,
    /// Target level (optional, auto-detected if not set).
    pub level: Option<DolbyVisionLevel>,
    /// Sub-profile for Profile 8.
    pub sub_profile: Option<SubProfile>,
    /// Target display peak luminance in nits.
    pub target_peak_nits: f64,
    /// Target display minimum luminance in nits.
    pub target_min_nits: f64,
    /// Enable gamut mapping.
    pub enable_gamut_mapping: bool,
    /// Target color primaries for gamut mapping.
    pub target_primaries: Option<metadata::ColorPrimaries>,
    /// Preserve source metadata during conversion.
    pub preserve_metadata: bool,
    /// Generate L2 trim passes if missing.
    pub generate_trim: bool,
}

impl DolbyConfig {
    /// Create a new configuration with the specified profile.
    pub fn new(profile: DolbyVisionProfile) -> Self {
        DolbyConfig {
            profile,
            level: None,
            sub_profile: None,
            target_peak_nits: 1000.0,
            target_min_nits: 0.0001,
            enable_gamut_mapping: false,
            target_primaries: None,
            preserve_metadata: true,
            generate_trim: true,
        }
    }

    /// Create a configuration for Profile 8.1 (HDR10 compatible).
    pub fn profile8_hdr10() -> Self {
        DolbyConfig {
            profile: DolbyVisionProfile::Profile8,
            level: None,
            sub_profile: Some(SubProfile::Profile8_1),
            target_peak_nits: 1000.0,
            target_min_nits: 0.0001,
            enable_gamut_mapping: false,
            target_primaries: None,
            preserve_metadata: true,
            generate_trim: true,
        }
    }

    /// Create a configuration for Profile 5 (SDR compatible).
    pub fn profile5_sdr() -> Self {
        DolbyConfig {
            profile: DolbyVisionProfile::Profile5,
            level: None,
            sub_profile: None,
            target_peak_nits: 100.0,
            target_min_nits: 0.1,
            enable_gamut_mapping: true,
            target_primaries: Some(metadata::ColorPrimaries::Bt709),
            preserve_metadata: true,
            generate_trim: true,
        }
    }

    /// Set the target level.
    pub fn with_level(mut self, level: DolbyVisionLevel) -> Self {
        self.level = Some(level);
        self
    }

    /// Set the sub-profile (for Profile 8).
    pub fn with_sub_profile(mut self, sub: SubProfile) -> Self {
        self.sub_profile = Some(sub);
        self
    }

    /// Set target display peak luminance.
    pub fn with_target_peak(mut self, nits: f64) -> Self {
        self.target_peak_nits = nits;
        self
    }

    /// Enable gamut mapping to target primaries.
    pub fn with_gamut_mapping(mut self, primaries: metadata::ColorPrimaries) -> Self {
        self.enable_gamut_mapping = true;
        self.target_primaries = Some(primaries);
        self
    }

    /// Disable metadata preservation.
    pub fn without_metadata_preservation(mut self) -> Self {
        self.preserve_metadata = false;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        // Check sub-profile is only used with Profile 8
        if self.sub_profile.is_some() && self.profile != DolbyVisionProfile::Profile8 {
            return Err(DolbyError::ConstraintViolation {
                message: "Sub-profile can only be used with Profile 8".to_string(),
            });
        }

        // Check target luminance is reasonable
        if self.target_peak_nits <= self.target_min_nits {
            return Err(DolbyError::ConstraintViolation {
                message: "Target peak must be greater than target minimum".to_string(),
            });
        }

        if self.target_peak_nits > 10000.0 {
            return Err(DolbyError::ConstraintViolation {
                message: "Target peak cannot exceed 10000 nits".to_string(),
            });
        }

        Ok(())
    }

    /// Get the profile/level combination.
    pub fn profile_level(&self) -> ProfileLevel {
        if let Some(sub) = self.sub_profile {
            ProfileLevel::with_sub_profile(
                self.profile,
                self.level.unwrap_or(DolbyVisionLevel::Level6),
                sub,
            )
        } else {
            ProfileLevel::new(
                self.profile,
                self.level.unwrap_or(DolbyVisionLevel::Level6),
            )
        }
    }

    /// Create a tone mapper from this configuration.
    pub fn create_tone_mapper(&self) -> mapping::ToneMapper {
        mapping::ToneMapper::new(self.target_peak_nits, self.target_min_nits)
    }

    /// Create a gamut mapper from this configuration.
    pub fn create_gamut_mapper(&self) -> Option<mapping::GamutMapper> {
        if !self.enable_gamut_mapping {
            return None;
        }

        match self.target_primaries {
            Some(metadata::ColorPrimaries::Bt709) => Some(mapping::GamutMapper::bt2020_to_bt709()),
            Some(metadata::ColorPrimaries::P3D65) => Some(mapping::GamutMapper::bt2020_to_p3()),
            _ => None,
        }
    }
}

impl Default for DolbyConfig {
    fn default() -> Self {
        DolbyConfig::new(DolbyVisionProfile::Profile8)
    }
}

/// Dolby Vision stream information.
#[derive(Debug, Clone)]
pub struct DolbyVisionInfo {
    /// Detected profile.
    pub profile: DolbyVisionProfile,
    /// Detected level.
    pub level: Option<DolbyVisionLevel>,
    /// Sub-profile (if applicable).
    pub sub_profile: Option<SubProfile>,
    /// Whether stream has enhancement layer.
    pub has_enhancement_layer: bool,
    /// Number of RPUs found.
    pub rpu_count: usize,
    /// Maximum content light level (from L6 or calculated).
    pub max_cll: Option<u16>,
    /// Maximum frame average light level.
    pub max_fall: Option<u16>,
}

impl DolbyVisionInfo {
    /// Analyze a stream and extract Dolby Vision information.
    pub fn from_stream(data: &[u8]) -> Result<Option<Self>> {
        let rpus = extractor::extract_all_rpus(data)?;

        if rpus.is_empty() {
            return Ok(None);
        }

        let first_rpu = &rpus[0];

        let profile = first_rpu.profile().ok_or_else(|| DolbyError::InvalidProfile {
            profile: first_rpu.header.guessed_profile,
        })?;

        let has_enhancement_layer = rpus.iter().any(|r| r.has_enhancement_layer());

        // Find max CLL/FALL from L6 metadata
        let (max_cll, max_fall) = rpus
            .iter()
            .filter_map(|r| r.metadata.l6.as_ref())
            .fold((None, None), |(cll, fall), l6| {
                (
                    Some(cll.unwrap_or(0).max(l6.max_cll)),
                    Some(fall.unwrap_or(0).max(l6.max_fall)),
                )
            });

        Ok(Some(DolbyVisionInfo {
            profile,
            level: None,
            sub_profile: None,
            has_enhancement_layer,
            rpu_count: rpus.len(),
            max_cll,
            max_fall,
        }))
    }

    /// Check if this is a dual-layer configuration.
    pub fn is_dual_layer(&self) -> bool {
        self.has_enhancement_layer
    }

    /// Get codec string for this configuration.
    pub fn codec_string(&self) -> String {
        let pl = ProfileLevel::new(
            self.profile,
            self.level.unwrap_or(DolbyVisionLevel::Level6),
        );
        pl.codec_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dolby_config_creation() {
        let config = DolbyConfig::new(DolbyVisionProfile::Profile8);
        assert_eq!(config.profile, DolbyVisionProfile::Profile8);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_dolby_config_profile8_hdr10() {
        let config = DolbyConfig::profile8_hdr10();
        assert_eq!(config.profile, DolbyVisionProfile::Profile8);
        assert_eq!(config.sub_profile, Some(SubProfile::Profile8_1));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_dolby_config_validation() {
        // Invalid: sub-profile with non-Profile 8
        let config = DolbyConfig {
            profile: DolbyVisionProfile::Profile5,
            sub_profile: Some(SubProfile::Profile8_1),
            ..Default::default()
        };
        assert!(config.validate().is_err());

        // Invalid: peak <= min
        let config = DolbyConfig {
            target_peak_nits: 0.0,
            target_min_nits: 1.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_dolby_config_builders() {
        let config = DolbyConfig::new(DolbyVisionProfile::Profile8)
            .with_level(DolbyVisionLevel::Level9)
            .with_target_peak(4000.0)
            .with_gamut_mapping(metadata::ColorPrimaries::Bt709);

        assert_eq!(config.level, Some(DolbyVisionLevel::Level9));
        assert_eq!(config.target_peak_nits, 4000.0);
        assert!(config.enable_gamut_mapping);
    }

    #[test]
    fn test_profile_level_codec_string() {
        let config = DolbyConfig::new(DolbyVisionProfile::Profile8)
            .with_level(DolbyVisionLevel::Level6);

        let pl = config.profile_level();
        assert_eq!(pl.codec_string(), "dvhe.08.06");
    }

    #[test]
    fn test_create_tone_mapper() {
        let config = DolbyConfig::new(DolbyVisionProfile::Profile8)
            .with_target_peak(1000.0);

        let mapper = config.create_tone_mapper();
        assert_eq!(mapper.target_peak_nits, 1000.0);
    }

    #[test]
    fn test_create_gamut_mapper() {
        let config = DolbyConfig::new(DolbyVisionProfile::Profile8);
        assert!(config.create_gamut_mapper().is_none());

        let config = config.with_gamut_mapping(metadata::ColorPrimaries::Bt709);
        assert!(config.create_gamut_mapper().is_some());
    }
}
