//! Dolby Vision profile and level definitions.
//!
//! This module defines all Dolby Vision profiles (4, 5, 7, 8, 9) and their
//! associated constraints, levels, and compatibility information.

use crate::error::{DolbyError, Result};

/// Dolby Vision profiles.
///
/// Each profile defines a specific combination of:
/// - Base layer codec and bit depth
/// - Enhancement layer presence and type
/// - HDR compatibility (HDR10, HLG, SDR)
/// - Backward compatibility
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DolbyVisionProfile {
    /// Profile 4: HLG cross-compatible, 12-bit.
    ///
    /// - Base layer: HEVC Main 10, 12-bit
    /// - Enhancement layer: None (single layer)
    /// - Backward compatible: HLG
    /// - Transfer function: HLG
    Profile4 = 4,

    /// Profile 5: SDR cross-compatible.
    ///
    /// - Base layer: HEVC Main 10
    /// - Enhancement layer: RPU only
    /// - Backward compatible: SDR (BT.709)
    /// - Common use: Converting SDR content with DV metadata
    Profile5 = 5,

    /// Profile 7: MEL + FEL, backward compatible.
    ///
    /// - Base layer: HEVC Main 10 (HDR10 or HLG)
    /// - Enhancement layer: FEL (Full Enhancement Layer) or MEL (Minimal Enhancement Layer)
    /// - Backward compatible: HDR10 or HLG
    /// - Dual-layer configuration
    /// - Highest quality but largest file size
    Profile7 = 7,

    /// Profile 8: HDR10 cross-compatible (most common).
    ///
    /// - Base layer: HEVC Main 10 (HDR10)
    /// - Enhancement layer: RPU only (no FEL/MEL)
    /// - Backward compatible: HDR10
    /// - Single-layer with metadata
    /// - Most widely supported profile
    ///
    /// Sub-profiles:
    /// - 8.1: HDR10 base, PQ transfer
    /// - 8.2: SDR base, converted
    /// - 8.4: HLG base, HLG transfer
    Profile8 = 8,

    /// Profile 9: AV1 based.
    ///
    /// - Base layer: AV1
    /// - Enhancement layer: RPU only
    /// - Backward compatible: AV1 HDR10
    /// - Designed for streaming with AV1 codec
    Profile9 = 9,
}

impl DolbyVisionProfile {
    /// Parse a profile from its numeric value.
    pub fn from_u8(value: u8) -> Result<Self> {
        match value {
            4 => Ok(DolbyVisionProfile::Profile4),
            5 => Ok(DolbyVisionProfile::Profile5),
            7 => Ok(DolbyVisionProfile::Profile7),
            8 => Ok(DolbyVisionProfile::Profile8),
            9 => Ok(DolbyVisionProfile::Profile9),
            _ => Err(DolbyError::InvalidProfile { profile: value }),
        }
    }

    /// Get the numeric value of the profile.
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Check if this profile uses a dual-layer configuration.
    pub fn is_dual_layer(self) -> bool {
        matches!(self, DolbyVisionProfile::Profile7)
    }

    /// Check if this profile is single-layer with RPU only.
    pub fn is_single_layer(self) -> bool {
        !self.is_dual_layer()
    }

    /// Check if this profile is backward compatible with HDR10.
    pub fn is_hdr10_compatible(self) -> bool {
        matches!(
            self,
            DolbyVisionProfile::Profile7 | DolbyVisionProfile::Profile8 | DolbyVisionProfile::Profile9
        )
    }

    /// Check if this profile is backward compatible with HLG.
    pub fn is_hlg_compatible(self) -> bool {
        matches!(self, DolbyVisionProfile::Profile4 | DolbyVisionProfile::Profile7)
    }

    /// Check if this profile is backward compatible with SDR.
    pub fn is_sdr_compatible(self) -> bool {
        matches!(self, DolbyVisionProfile::Profile5)
    }

    /// Get the base layer codec for this profile.
    pub fn base_codec(self) -> BaseCodec {
        match self {
            DolbyVisionProfile::Profile9 => BaseCodec::Av1,
            _ => BaseCodec::Hevc,
        }
    }

    /// Get the required bit depth for this profile.
    pub fn bit_depth(self) -> u8 {
        match self {
            DolbyVisionProfile::Profile4 => 12,
            _ => 10,
        }
    }

    /// Get the transfer function for this profile.
    pub fn transfer_function(self) -> TransferFunction {
        match self {
            DolbyVisionProfile::Profile4 => TransferFunction::Hlg,
            DolbyVisionProfile::Profile5 => TransferFunction::Sdr,
            _ => TransferFunction::Pq,
        }
    }

    /// Get the enhancement layer type for this profile.
    pub fn enhancement_layer(self) -> EnhancementLayer {
        match self {
            DolbyVisionProfile::Profile7 => EnhancementLayer::FelOrMel,
            DolbyVisionProfile::Profile4 => EnhancementLayer::None,
            _ => EnhancementLayer::RpuOnly,
        }
    }

    /// Check if this profile can be converted to another profile.
    pub fn can_convert_to(self, target: DolbyVisionProfile) -> bool {
        match (self, target) {
            // Profile 7 can be converted to Profile 8 (dual to single layer)
            (DolbyVisionProfile::Profile7, DolbyVisionProfile::Profile8) => true,
            // Profile 5 can be converted to Profile 8.1 with remapping
            (DolbyVisionProfile::Profile5, DolbyVisionProfile::Profile8) => true,
            // Same profile is always valid
            (a, b) if a == b => true,
            _ => false,
        }
    }

    /// Get supported sub-profiles (for Profile 8).
    pub fn sub_profiles(self) -> &'static [SubProfile] {
        match self {
            DolbyVisionProfile::Profile8 => &[
                SubProfile::Profile8_1,
                SubProfile::Profile8_2,
                SubProfile::Profile8_4,
            ],
            _ => &[],
        }
    }
}

impl std::fmt::Display for DolbyVisionProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DolbyVisionProfile::Profile4 => write!(f, "Profile 4 (HLG, 12-bit)"),
            DolbyVisionProfile::Profile5 => write!(f, "Profile 5 (SDR cross-compatible)"),
            DolbyVisionProfile::Profile7 => write!(f, "Profile 7 (MEL+FEL, BC)"),
            DolbyVisionProfile::Profile8 => write!(f, "Profile 8 (HDR10 cross-compatible)"),
            DolbyVisionProfile::Profile9 => write!(f, "Profile 9 (AV1 based)"),
        }
    }
}

/// Profile 8 sub-profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubProfile {
    /// Profile 8.1: HDR10 base layer with PQ transfer.
    Profile8_1,
    /// Profile 8.2: SDR base layer, converted.
    Profile8_2,
    /// Profile 8.4: HLG base layer with HLG transfer.
    Profile8_4,
}

impl SubProfile {
    /// Get the numeric identifier (8.x).
    pub fn version(self) -> (u8, u8) {
        match self {
            SubProfile::Profile8_1 => (8, 1),
            SubProfile::Profile8_2 => (8, 2),
            SubProfile::Profile8_4 => (8, 4),
        }
    }

    /// Get the transfer function for this sub-profile.
    pub fn transfer_function(self) -> TransferFunction {
        match self {
            SubProfile::Profile8_1 | SubProfile::Profile8_2 => TransferFunction::Pq,
            SubProfile::Profile8_4 => TransferFunction::Hlg,
        }
    }
}

/// Base layer codec type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BaseCodec {
    /// HEVC/H.265 base layer.
    Hevc,
    /// AV1 base layer.
    Av1,
}

impl std::fmt::Display for BaseCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BaseCodec::Hevc => write!(f, "HEVC"),
            BaseCodec::Av1 => write!(f, "AV1"),
        }
    }
}

/// Transfer function (EOTF).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferFunction {
    /// PQ (Perceptual Quantizer) - SMPTE ST 2084.
    Pq,
    /// HLG (Hybrid Log-Gamma) - ARIB STD-B67.
    Hlg,
    /// SDR (BT.1886 / BT.709).
    Sdr,
}

impl std::fmt::Display for TransferFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransferFunction::Pq => write!(f, "PQ (ST 2084)"),
            TransferFunction::Hlg => write!(f, "HLG"),
            TransferFunction::Sdr => write!(f, "SDR (BT.709)"),
        }
    }
}

/// Enhancement layer type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EnhancementLayer {
    /// No enhancement layer (single layer).
    None,
    /// RPU only (metadata without enhancement data).
    RpuOnly,
    /// FEL (Full Enhancement Layer) or MEL (Minimal Enhancement Layer).
    FelOrMel,
}

impl std::fmt::Display for EnhancementLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnhancementLayer::None => write!(f, "None"),
            EnhancementLayer::RpuOnly => write!(f, "RPU only"),
            EnhancementLayer::FelOrMel => write!(f, "FEL/MEL"),
        }
    }
}

/// Dolby Vision levels.
///
/// Levels define constraints on resolution, frame rate, and bitrate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum DolbyVisionLevel {
    /// Level 1: HD (1280x720 @ 24fps)
    Level1 = 1,
    /// Level 2: HD (1280x720 @ 30fps)
    Level2 = 2,
    /// Level 3: Full HD (1920x1080 @ 24fps)
    Level3 = 3,
    /// Level 4: Full HD (1920x1080 @ 30fps)
    Level4 = 4,
    /// Level 5: Full HD (1920x1080 @ 60fps)
    Level5 = 5,
    /// Level 6: UHD (3840x2160 @ 24fps)
    Level6 = 6,
    /// Level 7: UHD (3840x2160 @ 30fps)
    Level7 = 7,
    /// Level 8: UHD (3840x2160 @ 48fps)
    Level8 = 8,
    /// Level 9: UHD (3840x2160 @ 60fps)
    Level9 = 9,
    /// Level 10: UHD (3840x2160 @ 120fps)
    Level10 = 10,
    /// Level 11: 8K (7680x4320 @ 24fps)
    Level11 = 11,
    /// Level 12: 8K (7680x4320 @ 30fps)
    Level12 = 12,
    /// Level 13: 8K (7680x4320 @ 60fps)
    Level13 = 13,
}

impl DolbyVisionLevel {
    /// Parse a level from its numeric value.
    pub fn from_u8(value: u8) -> Result<Self> {
        match value {
            1 => Ok(DolbyVisionLevel::Level1),
            2 => Ok(DolbyVisionLevel::Level2),
            3 => Ok(DolbyVisionLevel::Level3),
            4 => Ok(DolbyVisionLevel::Level4),
            5 => Ok(DolbyVisionLevel::Level5),
            6 => Ok(DolbyVisionLevel::Level6),
            7 => Ok(DolbyVisionLevel::Level7),
            8 => Ok(DolbyVisionLevel::Level8),
            9 => Ok(DolbyVisionLevel::Level9),
            10 => Ok(DolbyVisionLevel::Level10),
            11 => Ok(DolbyVisionLevel::Level11),
            12 => Ok(DolbyVisionLevel::Level12),
            13 => Ok(DolbyVisionLevel::Level13),
            _ => Err(DolbyError::InvalidLevel { level: value }),
        }
    }

    /// Get the numeric value of the level.
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Get the level constraints.
    pub fn constraints(self) -> LevelConstraints {
        match self {
            DolbyVisionLevel::Level1 => LevelConstraints {
                max_width: 1280,
                max_height: 720,
                max_frame_rate: 24.0,
                max_bitrate_kbps: 20_000,
            },
            DolbyVisionLevel::Level2 => LevelConstraints {
                max_width: 1280,
                max_height: 720,
                max_frame_rate: 30.0,
                max_bitrate_kbps: 20_000,
            },
            DolbyVisionLevel::Level3 => LevelConstraints {
                max_width: 1920,
                max_height: 1080,
                max_frame_rate: 24.0,
                max_bitrate_kbps: 25_000,
            },
            DolbyVisionLevel::Level4 => LevelConstraints {
                max_width: 1920,
                max_height: 1080,
                max_frame_rate: 30.0,
                max_bitrate_kbps: 30_000,
            },
            DolbyVisionLevel::Level5 => LevelConstraints {
                max_width: 1920,
                max_height: 1080,
                max_frame_rate: 60.0,
                max_bitrate_kbps: 40_000,
            },
            DolbyVisionLevel::Level6 => LevelConstraints {
                max_width: 3840,
                max_height: 2160,
                max_frame_rate: 24.0,
                max_bitrate_kbps: 60_000,
            },
            DolbyVisionLevel::Level7 => LevelConstraints {
                max_width: 3840,
                max_height: 2160,
                max_frame_rate: 30.0,
                max_bitrate_kbps: 60_000,
            },
            DolbyVisionLevel::Level8 => LevelConstraints {
                max_width: 3840,
                max_height: 2160,
                max_frame_rate: 48.0,
                max_bitrate_kbps: 80_000,
            },
            DolbyVisionLevel::Level9 => LevelConstraints {
                max_width: 3840,
                max_height: 2160,
                max_frame_rate: 60.0,
                max_bitrate_kbps: 100_000,
            },
            DolbyVisionLevel::Level10 => LevelConstraints {
                max_width: 3840,
                max_height: 2160,
                max_frame_rate: 120.0,
                max_bitrate_kbps: 150_000,
            },
            DolbyVisionLevel::Level11 => LevelConstraints {
                max_width: 7680,
                max_height: 4320,
                max_frame_rate: 24.0,
                max_bitrate_kbps: 200_000,
            },
            DolbyVisionLevel::Level12 => LevelConstraints {
                max_width: 7680,
                max_height: 4320,
                max_frame_rate: 30.0,
                max_bitrate_kbps: 240_000,
            },
            DolbyVisionLevel::Level13 => LevelConstraints {
                max_width: 7680,
                max_height: 4320,
                max_frame_rate: 60.0,
                max_bitrate_kbps: 400_000,
            },
        }
    }
}

impl std::fmt::Display for DolbyVisionLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let c = self.constraints();
        write!(
            f,
            "Level {} ({}x{} @ {}fps)",
            self.as_u8(),
            c.max_width,
            c.max_height,
            c.max_frame_rate
        )
    }
}

/// Level constraints defining maximum supported parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LevelConstraints {
    /// Maximum width in pixels.
    pub max_width: u32,
    /// Maximum height in pixels.
    pub max_height: u32,
    /// Maximum frame rate in fps.
    pub max_frame_rate: f64,
    /// Maximum bitrate in kbps.
    pub max_bitrate_kbps: u32,
}

impl LevelConstraints {
    /// Check if the given parameters fit within these constraints.
    pub fn fits(&self, width: u32, height: u32, frame_rate: f64, bitrate_kbps: Option<u32>) -> bool {
        width <= self.max_width
            && height <= self.max_height
            && frame_rate <= self.max_frame_rate
            && bitrate_kbps.map_or(true, |br| br <= self.max_bitrate_kbps)
    }

    /// Maximum number of pixels (width * height).
    pub fn max_pixels(&self) -> u32 {
        self.max_width * self.max_height
    }
}

/// Find the minimum level that supports the given parameters.
pub fn find_minimum_level(
    width: u32,
    height: u32,
    frame_rate: f64,
    bitrate_kbps: Option<u32>,
) -> Option<DolbyVisionLevel> {
    let levels = [
        DolbyVisionLevel::Level1,
        DolbyVisionLevel::Level2,
        DolbyVisionLevel::Level3,
        DolbyVisionLevel::Level4,
        DolbyVisionLevel::Level5,
        DolbyVisionLevel::Level6,
        DolbyVisionLevel::Level7,
        DolbyVisionLevel::Level8,
        DolbyVisionLevel::Level9,
        DolbyVisionLevel::Level10,
        DolbyVisionLevel::Level11,
        DolbyVisionLevel::Level12,
        DolbyVisionLevel::Level13,
    ];

    for level in levels {
        if level.constraints().fits(width, height, frame_rate, bitrate_kbps) {
            return Some(level);
        }
    }

    None
}

/// Profile and level combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProfileLevel {
    /// The Dolby Vision profile.
    pub profile: DolbyVisionProfile,
    /// The Dolby Vision level.
    pub level: DolbyVisionLevel,
    /// Optional sub-profile (for Profile 8).
    pub sub_profile: Option<SubProfile>,
}

impl ProfileLevel {
    /// Create a new profile/level combination.
    pub fn new(profile: DolbyVisionProfile, level: DolbyVisionLevel) -> Self {
        ProfileLevel {
            profile,
            level,
            sub_profile: None,
        }
    }

    /// Create a new profile/level with sub-profile.
    pub fn with_sub_profile(
        profile: DolbyVisionProfile,
        level: DolbyVisionLevel,
        sub_profile: SubProfile,
    ) -> Self {
        ProfileLevel {
            profile,
            level,
            sub_profile: Some(sub_profile),
        }
    }

    /// Validate this profile/level combination.
    pub fn validate(&self) -> Result<()> {
        // Check sub-profile is only used with Profile 8
        if self.sub_profile.is_some() && self.profile != DolbyVisionProfile::Profile8 {
            return Err(DolbyError::ConstraintViolation {
                message: "Sub-profile can only be used with Profile 8".to_string(),
            });
        }

        Ok(())
    }

    /// Get the codec configuration string (e.g., "dvhe.08.06").
    pub fn codec_string(&self) -> String {
        let profile_str = match self.profile {
            DolbyVisionProfile::Profile4 => "dvhe.04",
            DolbyVisionProfile::Profile5 => "dvhe.05",
            DolbyVisionProfile::Profile7 => "dvhe.07",
            DolbyVisionProfile::Profile8 => {
                match self.sub_profile {
                    Some(SubProfile::Profile8_1) => "dvhe.08",
                    Some(SubProfile::Profile8_2) => "dvhe.08",
                    Some(SubProfile::Profile8_4) => "dvhe.08",
                    None => "dvhe.08",
                }
            }
            DolbyVisionProfile::Profile9 => "dav1.09",
        };

        format!("{}.{:02}", profile_str, self.level.as_u8())
    }
}

impl std::fmt::Display for ProfileLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(sub) = self.sub_profile {
            let (major, minor) = sub.version();
            write!(f, "Profile {}.{}, {}", major, minor, self.level)
        } else {
            write!(f, "{}, {}", self.profile, self.level)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_from_u8() {
        assert_eq!(
            DolbyVisionProfile::from_u8(4).unwrap(),
            DolbyVisionProfile::Profile4
        );
        assert_eq!(
            DolbyVisionProfile::from_u8(8).unwrap(),
            DolbyVisionProfile::Profile8
        );
        assert!(DolbyVisionProfile::from_u8(0).is_err());
        assert!(DolbyVisionProfile::from_u8(6).is_err());
    }

    #[test]
    fn test_profile_properties() {
        assert!(DolbyVisionProfile::Profile7.is_dual_layer());
        assert!(DolbyVisionProfile::Profile8.is_single_layer());
        assert!(DolbyVisionProfile::Profile8.is_hdr10_compatible());
        assert!(DolbyVisionProfile::Profile4.is_hlg_compatible());
        assert!(DolbyVisionProfile::Profile5.is_sdr_compatible());
    }

    #[test]
    fn test_level_constraints() {
        let level9 = DolbyVisionLevel::Level9;
        let constraints = level9.constraints();
        assert_eq!(constraints.max_width, 3840);
        assert_eq!(constraints.max_height, 2160);
        assert_eq!(constraints.max_frame_rate, 60.0);
    }

    #[test]
    fn test_find_minimum_level() {
        // 1080p24 should be Level 3
        assert_eq!(
            find_minimum_level(1920, 1080, 24.0, None),
            Some(DolbyVisionLevel::Level3)
        );

        // 4K60 should be Level 9
        assert_eq!(
            find_minimum_level(3840, 2160, 60.0, None),
            Some(DolbyVisionLevel::Level9)
        );

        // 8K120 doesn't fit any level
        assert_eq!(find_minimum_level(7680, 4320, 120.0, None), None);
    }

    #[test]
    fn test_profile_level_codec_string() {
        let pl = ProfileLevel::new(DolbyVisionProfile::Profile8, DolbyVisionLevel::Level6);
        assert_eq!(pl.codec_string(), "dvhe.08.06");

        let pl = ProfileLevel::new(DolbyVisionProfile::Profile9, DolbyVisionLevel::Level9);
        assert_eq!(pl.codec_string(), "dav1.09.09");
    }

    #[test]
    fn test_profile_conversion() {
        assert!(DolbyVisionProfile::Profile7.can_convert_to(DolbyVisionProfile::Profile8));
        assert!(DolbyVisionProfile::Profile5.can_convert_to(DolbyVisionProfile::Profile8));
        assert!(!DolbyVisionProfile::Profile8.can_convert_to(DolbyVisionProfile::Profile7));
    }
}
