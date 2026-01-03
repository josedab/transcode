//! Dolby Vision metadata structures.
//!
//! This module defines the various metadata levels (L1-L11) used in Dolby Vision
//! to convey scene-specific HDR information, trim passes, and display mappings.

use crate::error::{DolbyError, Result};

/// L1 metadata: Scene-level luminance information.
///
/// L1 metadata is mandatory and provides the minimum, maximum, and average
/// luminance values for a scene, used for dynamic tone mapping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct L1Metadata {
    /// Minimum luminance in the scene (in PQ code values, 0-4095).
    pub min_pq: u16,
    /// Maximum luminance in the scene (in PQ code values, 0-4095).
    pub max_pq: u16,
    /// Average luminance in the scene (in PQ code values, 0-4095).
    pub avg_pq: u16,
}

impl L1Metadata {
    /// Create new L1 metadata.
    pub fn new(min_pq: u16, max_pq: u16, avg_pq: u16) -> Result<Self> {
        if min_pq > 4095 || max_pq > 4095 || avg_pq > 4095 {
            return Err(DolbyError::invalid_metadata(
                1,
                "PQ values must be in range 0-4095",
            ));
        }
        if min_pq > avg_pq || avg_pq > max_pq {
            return Err(DolbyError::invalid_metadata(
                1,
                "Invalid luminance ordering: min <= avg <= max required",
            ));
        }
        Ok(L1Metadata { min_pq, max_pq, avg_pq })
    }

    /// Convert PQ code value to linear luminance in nits.
    pub fn pq_to_nits(pq: u16) -> f64 {
        let pq_normalized = pq as f64 / 4095.0;
        pq_eotf(pq_normalized) * 10000.0
    }

    /// Get minimum luminance in nits.
    pub fn min_nits(&self) -> f64 {
        Self::pq_to_nits(self.min_pq)
    }

    /// Get maximum luminance in nits.
    pub fn max_nits(&self) -> f64 {
        Self::pq_to_nits(self.max_pq)
    }

    /// Get average luminance in nits.
    pub fn avg_nits(&self) -> f64 {
        Self::pq_to_nits(self.avg_pq)
    }
}

impl Default for L1Metadata {
    fn default() -> Self {
        L1Metadata {
            min_pq: 0,
            max_pq: 4095,
            avg_pq: 2048,
        }
    }
}

/// L2 metadata: Trim pass for specific target display.
///
/// L2 metadata provides parameters for tone mapping to a specific
/// target display luminance level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct L2Metadata {
    /// Target display peak luminance identifier.
    pub target_max_pq: u16,
    /// Trim slope.
    pub trim_slope: u16,
    /// Trim offset.
    pub trim_offset: u16,
    /// Trim power.
    pub trim_power: u16,
    /// Trim chroma weight.
    pub trim_chroma_weight: u16,
    /// Trim saturation gain.
    pub trim_saturation_gain: u16,
    /// Mid-tone offset for tone curve.
    pub ms_weight: i16,
}

impl L2Metadata {
    /// Create new L2 metadata with default values.
    pub fn new(target_max_pq: u16) -> Self {
        L2Metadata {
            target_max_pq,
            trim_slope: 2048,    // 1.0 in 12-bit fixed point
            trim_offset: 2048,   // 0.0 centered at 2048
            trim_power: 2048,    // 1.0 in 12-bit fixed point
            trim_chroma_weight: 2048,
            trim_saturation_gain: 2048,
            ms_weight: 0,
        }
    }

    /// Get target display luminance in nits.
    pub fn target_nits(&self) -> f64 {
        L1Metadata::pq_to_nits(self.target_max_pq)
    }

    /// Get trim slope as a floating point value.
    pub fn slope(&self) -> f64 {
        self.trim_slope as f64 / 2048.0
    }

    /// Get trim offset as a floating point value.
    pub fn offset(&self) -> f64 {
        (self.trim_offset as f64 - 2048.0) / 2048.0
    }

    /// Get trim power as a floating point value.
    pub fn power(&self) -> f64 {
        self.trim_power as f64 / 2048.0
    }
}

impl Default for L2Metadata {
    fn default() -> Self {
        L2Metadata::new(2081) // ~100 nits
    }
}

/// L5 metadata: Active area definition.
///
/// L5 metadata defines the active picture area within the coded frame,
/// useful for letterboxed or pillarboxed content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L5Metadata {
    /// Left offset in pixels.
    pub left_offset: u16,
    /// Right offset in pixels.
    pub right_offset: u16,
    /// Top offset in pixels.
    pub top_offset: u16,
    /// Bottom offset in pixels.
    pub bottom_offset: u16,
}

impl L5Metadata {
    /// Create new L5 metadata.
    pub fn new(left: u16, right: u16, top: u16, bottom: u16) -> Self {
        L5Metadata {
            left_offset: left,
            right_offset: right,
            top_offset: top,
            bottom_offset: bottom,
        }
    }

    /// Check if there's any active area cropping.
    pub fn has_cropping(&self) -> bool {
        self.left_offset > 0
            || self.right_offset > 0
            || self.top_offset > 0
            || self.bottom_offset > 0
    }

    /// Get the active width given the frame width.
    pub fn active_width(&self, frame_width: u16) -> u16 {
        frame_width.saturating_sub(self.left_offset).saturating_sub(self.right_offset)
    }

    /// Get the active height given the frame height.
    pub fn active_height(&self, frame_height: u16) -> u16 {
        frame_height.saturating_sub(self.top_offset).saturating_sub(self.bottom_offset)
    }
}

impl Default for L5Metadata {
    fn default() -> Self {
        L5Metadata::new(0, 0, 0, 0)
    }
}

/// L6 metadata: MaxCLL and MaxFALL.
///
/// L6 metadata provides Content Light Level information compatible
/// with HDR10 static metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L6Metadata {
    /// Maximum Content Light Level in nits.
    pub max_cll: u16,
    /// Maximum Frame Average Light Level in nits.
    pub max_fall: u16,
}

impl L6Metadata {
    /// Create new L6 metadata.
    pub fn new(max_cll: u16, max_fall: u16) -> Result<Self> {
        if max_fall > max_cll {
            return Err(DolbyError::invalid_metadata(
                6,
                "MaxFALL cannot exceed MaxCLL",
            ));
        }
        Ok(L6Metadata { max_cll, max_fall })
    }

    /// Create L6 metadata without validation.
    pub fn new_unchecked(max_cll: u16, max_fall: u16) -> Self {
        L6Metadata { max_cll, max_fall }
    }
}

impl Default for L6Metadata {
    fn default() -> Self {
        L6Metadata {
            max_cll: 1000,
            max_fall: 400,
        }
    }
}

/// L8 metadata: Tone mapping curves.
///
/// L8 metadata provides detailed mapping curves for different
/// target luminance levels, including coefficients for polynomial mapping.
#[derive(Debug, Clone, PartialEq)]
pub struct L8Metadata {
    /// Target display index.
    pub target_display_index: u8,
    /// Trim slope.
    pub trim_slope: u16,
    /// Trim offset.
    pub trim_offset: u16,
    /// Trim power.
    pub trim_power: u16,
    /// Trim chroma weight.
    pub trim_chroma_weight: u16,
    /// Trim saturation gain.
    pub trim_saturation_gain: u16,
    /// Mid-tone width offset.
    pub ms_weight: i16,
    /// Target mid contrast.
    pub target_mid_contrast: u16,
    /// Clip trim.
    pub clip_trim: u16,
    /// Saturation vector field.
    pub saturation_vector_field: Vec<u8>,
    /// Hue vector field.
    pub hue_vector_field: Vec<u8>,
}

impl L8Metadata {
    /// Create new L8 metadata with default values.
    pub fn new(target_display_index: u8) -> Self {
        L8Metadata {
            target_display_index,
            trim_slope: 2048,
            trim_offset: 2048,
            trim_power: 2048,
            trim_chroma_weight: 2048,
            trim_saturation_gain: 2048,
            ms_weight: 0,
            target_mid_contrast: 2048,
            clip_trim: 2048,
            saturation_vector_field: Vec::new(),
            hue_vector_field: Vec::new(),
        }
    }
}

impl Default for L8Metadata {
    fn default() -> Self {
        L8Metadata::new(0)
    }
}

/// L9 metadata: Source display information.
///
/// L9 metadata describes the characteristics of the display
/// used during content mastering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L9Metadata {
    /// Source primary index (color gamut).
    pub source_primary_index: u8,
    /// Source display peak luminance in PQ.
    pub source_display_max_pq: u16,
    /// Source display minimum luminance in PQ.
    pub source_display_min_pq: u16,
}

impl L9Metadata {
    /// Create new L9 metadata.
    pub fn new(primary_index: u8, max_pq: u16, min_pq: u16) -> Result<Self> {
        if max_pq > 4095 || min_pq > 4095 {
            return Err(DolbyError::invalid_metadata(
                9,
                "PQ values must be in range 0-4095",
            ));
        }
        if min_pq > max_pq {
            return Err(DolbyError::invalid_metadata(
                9,
                "min_pq cannot exceed max_pq",
            ));
        }
        Ok(L9Metadata {
            source_primary_index: primary_index,
            source_display_max_pq: max_pq,
            source_display_min_pq: min_pq,
        })
    }

    /// Get source display peak luminance in nits.
    pub fn max_nits(&self) -> f64 {
        L1Metadata::pq_to_nits(self.source_display_max_pq)
    }

    /// Get source display minimum luminance in nits.
    pub fn min_nits(&self) -> f64 {
        L1Metadata::pq_to_nits(self.source_display_min_pq)
    }

    /// Get the color primaries from the index.
    pub fn primaries(&self) -> ColorPrimaries {
        ColorPrimaries::from_index(self.source_primary_index)
    }
}

impl Default for L9Metadata {
    fn default() -> Self {
        L9Metadata {
            source_primary_index: 0, // BT.2020
            source_display_max_pq: 3079, // ~1000 nits
            source_display_min_pq: 0,
        }
    }
}

/// L11 metadata: Content type information.
///
/// L11 metadata provides information about the content type
/// for adaptive processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L11Metadata {
    /// Content type.
    pub content_type: ContentType,
    /// Whitepoint adaptation mode.
    pub whitepoint: u8,
    /// Reference mode flag.
    pub reference_mode_flag: bool,
}

impl L11Metadata {
    /// Create new L11 metadata.
    pub fn new(content_type: ContentType) -> Self {
        L11Metadata {
            content_type,
            whitepoint: 0,
            reference_mode_flag: false,
        }
    }
}

impl Default for L11Metadata {
    fn default() -> Self {
        L11Metadata {
            content_type: ContentType::Unknown,
            whitepoint: 0,
            reference_mode_flag: false,
        }
    }
}

/// Content type enumeration for L11 metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContentType {
    /// Unknown or unspecified content.
    #[default]
    Unknown = 0,
    /// Film/cinema content.
    Film = 1,
    /// Video/broadcast content.
    Video = 2,
    /// Graphics/animation content.
    Graphics = 3,
    /// Photography/still images.
    Photo = 4,
    /// Gaming content.
    Gaming = 5,
}

impl ContentType {
    /// Parse content type from u8 value.
    pub fn from_u8(value: u8) -> Self {
        match value {
            1 => ContentType::Film,
            2 => ContentType::Video,
            3 => ContentType::Graphics,
            4 => ContentType::Photo,
            5 => ContentType::Gaming,
            _ => ContentType::Unknown,
        }
    }

    /// Convert to u8 value.
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Color primaries enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorPrimaries {
    /// BT.709 (sRGB, HD).
    Bt709,
    /// BT.2020 (Wide Color Gamut, UHD).
    Bt2020,
    /// P3-D65 (Digital Cinema).
    P3D65,
    /// P3-DCI (DCI-P3).
    P3Dci,
    /// Unknown/custom primaries.
    Unknown(u8),
}

impl ColorPrimaries {
    /// Get primaries from source primary index.
    pub fn from_index(index: u8) -> Self {
        match index {
            0 => ColorPrimaries::Bt2020,
            1 => ColorPrimaries::P3D65,
            2 => ColorPrimaries::Bt709,
            _ => ColorPrimaries::Unknown(index),
        }
    }

    /// Get the index for these primaries.
    pub fn to_index(self) -> u8 {
        match self {
            ColorPrimaries::Bt2020 => 0,
            ColorPrimaries::P3D65 => 1,
            ColorPrimaries::Bt709 => 2,
            ColorPrimaries::P3Dci => 3,
            ColorPrimaries::Unknown(i) => i,
        }
    }
}

/// Complete metadata block containing all metadata levels.
#[derive(Debug, Clone, Default)]
pub struct DolbyVisionMetadata {
    /// L1 metadata (scene luminance) - required.
    pub l1: Option<L1Metadata>,
    /// L2 metadata (trim passes) - one per target display.
    pub l2: Vec<L2Metadata>,
    /// L5 metadata (active area) - optional.
    pub l5: Option<L5Metadata>,
    /// L6 metadata (MaxCLL/MaxFALL) - optional.
    pub l6: Option<L6Metadata>,
    /// L8 metadata (mapping curves) - optional.
    pub l8: Vec<L8Metadata>,
    /// L9 metadata (source display) - optional.
    pub l9: Option<L9Metadata>,
    /// L11 metadata (content type) - optional.
    pub l11: Option<L11Metadata>,
}

impl DolbyVisionMetadata {
    /// Create new empty metadata.
    pub fn new() -> Self {
        DolbyVisionMetadata::default()
    }

    /// Create metadata with L1 data.
    pub fn with_l1(l1: L1Metadata) -> Self {
        DolbyVisionMetadata {
            l1: Some(l1),
            ..Default::default()
        }
    }

    /// Check if this metadata is valid (has required L1).
    pub fn is_valid(&self) -> bool {
        self.l1.is_some()
    }

    /// Add L2 trim pass.
    pub fn add_l2(&mut self, l2: L2Metadata) {
        self.l2.push(l2);
    }

    /// Get L2 for a specific target luminance.
    pub fn get_l2_for_target(&self, target_pq: u16) -> Option<&L2Metadata> {
        self.l2.iter().find(|l2| l2.target_max_pq == target_pq)
    }

    /// Add L8 mapping curve.
    pub fn add_l8(&mut self, l8: L8Metadata) {
        self.l8.push(l8);
    }

    /// Validate the metadata consistency.
    pub fn validate(&self) -> Result<()> {
        // L1 is required
        let _l1 = self.l1.as_ref().ok_or_else(|| {
            DolbyError::MissingMetadata {
                field: "L1 (scene luminance)".to_string(),
            }
        })?;

        // L6 MaxFALL <= MaxCLL
        if let Some(l6) = &self.l6 {
            if l6.max_fall > l6.max_cll {
                return Err(DolbyError::invalid_metadata(
                    6,
                    "MaxFALL cannot exceed MaxCLL",
                ));
            }
        }

        Ok(())
    }
}

/// PQ EOTF (Electro-Optical Transfer Function).
///
/// Converts PQ signal value to linear light (0-1 range).
fn pq_eotf(pq: f64) -> f64 {
    const M1: f64 = 2610.0 / 16384.0;
    const M2: f64 = 2523.0 / 4096.0 * 128.0;
    const C1: f64 = 3424.0 / 4096.0;
    const C2: f64 = 2413.0 / 4096.0 * 32.0;
    const C3: f64 = 2392.0 / 4096.0 * 32.0;

    if pq <= 0.0 {
        return 0.0;
    }

    let pq_pow = pq.powf(1.0 / M2);
    let numerator = (pq_pow - C1).max(0.0);
    let denominator = C2 - C3 * pq_pow;

    if denominator <= 0.0 {
        return 1.0;
    }

    (numerator / denominator).powf(1.0 / M1)
}

/// PQ inverse EOTF (OETF).
///
/// Converts linear light to PQ signal value.
#[allow(dead_code)]
fn pq_oetf(linear: f64) -> f64 {
    const M1: f64 = 2610.0 / 16384.0;
    const M2: f64 = 2523.0 / 4096.0 * 128.0;
    const C1: f64 = 3424.0 / 4096.0;
    const C2: f64 = 2413.0 / 4096.0 * 32.0;
    const C3: f64 = 2392.0 / 4096.0 * 32.0;

    if linear <= 0.0 {
        return 0.0;
    }

    let y_pow = linear.powf(M1);
    let numerator = C1 + C2 * y_pow;
    let denominator = 1.0 + C3 * y_pow;

    (numerator / denominator).powf(M2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l1_metadata() {
        let l1 = L1Metadata::new(0, 3079, 1500).unwrap();
        assert!(l1.max_nits() > 900.0 && l1.max_nits() < 1100.0); // ~1000 nits

        // Invalid: min > avg
        assert!(L1Metadata::new(2000, 3000, 1000).is_err());
    }

    #[test]
    fn test_l5_metadata() {
        let l5 = L5Metadata::new(100, 100, 50, 50);
        assert!(l5.has_cropping());
        assert_eq!(l5.active_width(1920), 1720);
        assert_eq!(l5.active_height(1080), 980);
    }

    #[test]
    fn test_l6_metadata() {
        assert!(L6Metadata::new(1000, 400).is_ok());
        assert!(L6Metadata::new(400, 1000).is_err()); // MaxFALL > MaxCLL
    }

    #[test]
    fn test_pq_conversion() {
        // PQ 0 should be ~0 nits
        let nits_0 = L1Metadata::pq_to_nits(0);
        assert!(nits_0 < 0.01);

        // PQ 3079 should be ~1000 nits
        let nits_1000 = L1Metadata::pq_to_nits(3079);
        assert!(nits_1000 > 900.0 && nits_1000 < 1100.0);

        // PQ 4095 should be ~10000 nits
        let nits_10000 = L1Metadata::pq_to_nits(4095);
        assert!(nits_10000 > 9000.0 && nits_10000 < 11000.0);
    }

    #[test]
    fn test_content_type() {
        assert_eq!(ContentType::from_u8(1), ContentType::Film);
        assert_eq!(ContentType::from_u8(5), ContentType::Gaming);
        assert_eq!(ContentType::from_u8(99), ContentType::Unknown);
    }

    #[test]
    fn test_dolby_vision_metadata() {
        let mut metadata = DolbyVisionMetadata::new();
        assert!(!metadata.is_valid());

        metadata.l1 = Some(L1Metadata::default());
        assert!(metadata.is_valid());
        assert!(metadata.validate().is_ok());
    }
}
