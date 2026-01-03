//! HDR metadata handling.
//!
//! This module provides support for:
//! - Static metadata (MaxCLL, MaxFALL, mastering display info)
//! - HDR10 metadata (SEI parsing and generation)
//! - HDR10+ dynamic metadata parsing
//! - Content light level information

use crate::{HdrError, Result};

// ============================================================================
// Content Light Level Info (CLL)
// ============================================================================

/// Content Light Level information.
/// Contains MaxCLL and MaxFALL values per BT.2020/ST.2086.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ContentLightLevel {
    /// Maximum Content Light Level (nits)
    /// The maximum luminance of any single pixel in the content.
    pub max_cll: u16,
    /// Maximum Frame-Average Light Level (nits)
    /// The maximum average luminance of any single frame.
    pub max_fall: u16,
}

impl ContentLightLevel {
    /// Create new content light level info.
    pub fn new(max_cll: u16, max_fall: u16) -> Self {
        Self { max_cll, max_fall }
    }

    /// Check if the content exceeds SDR range (100 nits).
    pub fn is_hdr(&self) -> bool {
        self.max_cll > 100 || self.max_fall > 100
    }

    /// Parse from raw bytes (big-endian format as in SEI).
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            return Err(HdrError::InvalidMetadata(
                "Content light level data too short".into(),
            ));
        }

        Ok(Self {
            max_cll: u16::from_be_bytes([data[0], data[1]]),
            max_fall: u16::from_be_bytes([data[2], data[3]]),
        })
    }

    /// Serialize to bytes (big-endian format).
    pub fn to_bytes(&self) -> [u8; 4] {
        let mut data = [0u8; 4];
        data[0..2].copy_from_slice(&self.max_cll.to_be_bytes());
        data[2..4].copy_from_slice(&self.max_fall.to_be_bytes());
        data
    }
}

// ============================================================================
// Mastering Display Color Volume (MDCV)
// ============================================================================

/// Mastering Display Color Volume.
/// Contains color primaries and luminance range of the mastering display.
#[derive(Debug, Clone, PartialEq)]
pub struct MasteringDisplayColorVolume {
    /// Display primaries in CIE 1931 xy chromaticity (RGB order)
    /// Values are in the range 0.0 to 1.0
    pub primaries: [[f64; 2]; 3],
    /// White point in CIE 1931 xy chromaticity
    pub white_point: [f64; 2],
    /// Maximum display luminance (nits)
    pub max_luminance: f64,
    /// Minimum display luminance (nits)
    pub min_luminance: f64,
}

impl Default for MasteringDisplayColorVolume {
    fn default() -> Self {
        // Default to BT.2020 primaries with typical HDR mastering display
        Self {
            primaries: [
                [0.708, 0.292],  // Red
                [0.170, 0.797],  // Green
                [0.131, 0.046],  // Blue
            ],
            white_point: [0.3127, 0.3290],  // D65
            max_luminance: 1000.0,
            min_luminance: 0.0001,
        }
    }
}

impl MasteringDisplayColorVolume {
    /// Create MDCV with custom values.
    pub fn new(
        primaries: [[f64; 2]; 3],
        white_point: [f64; 2],
        max_luminance: f64,
        min_luminance: f64,
    ) -> Self {
        Self {
            primaries,
            white_point,
            max_luminance,
            min_luminance,
        }
    }

    /// Create MDCV for HDR10 standard (1000 nits peak).
    pub fn hdr10_standard() -> Self {
        Self {
            max_luminance: 1000.0,
            min_luminance: 0.0001,
            ..Default::default()
        }
    }

    /// Create MDCV for HDR10 high brightness (4000 nits peak).
    pub fn hdr10_high_brightness() -> Self {
        Self {
            max_luminance: 4000.0,
            min_luminance: 0.00005,
            ..Default::default()
        }
    }

    /// Parse from SEI data.
    /// Data format per HEVC spec: G, B, R primaries, white point, luminance.
    /// Each chromaticity value is u16 with 0.00002 precision.
    /// Luminance is u32 with 0.0001 precision for max, 0.00001 for min.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 24 {
            return Err(HdrError::InvalidMetadata(
                "Mastering display metadata too short".into(),
            ));
        }

        // Parse chromaticity values (G, B, R order in SEI, we store as R, G, B)
        let parse_chromaticity = |offset: usize| -> [f64; 2] {
            let x = u16::from_be_bytes([data[offset], data[offset + 1]]) as f64 * 0.00002;
            let y = u16::from_be_bytes([data[offset + 2], data[offset + 3]]) as f64 * 0.00002;
            [x, y]
        };

        // Parse in G, B, R order and reorder to R, G, B
        let green = parse_chromaticity(0);
        let blue = parse_chromaticity(4);
        let red = parse_chromaticity(8);
        let white_point = parse_chromaticity(12);

        // Parse luminance values
        let max_luminance = u32::from_be_bytes([data[16], data[17], data[18], data[19]]) as f64 * 0.0001;
        let min_luminance = u32::from_be_bytes([data[20], data[21], data[22], data[23]]) as f64 * 0.0001;

        Ok(Self {
            primaries: [red, green, blue],
            white_point,
            max_luminance,
            min_luminance,
        })
    }

    /// Serialize to SEI data bytes.
    pub fn to_bytes(&self) -> [u8; 24] {
        let mut data = [0u8; 24];

        let encode_chromaticity = |xy: [f64; 2]| -> [u8; 4] {
            let x = ((xy[0] / 0.00002).round() as u16).to_be_bytes();
            let y = ((xy[1] / 0.00002).round() as u16).to_be_bytes();
            [x[0], x[1], y[0], y[1]]
        };

        // Encode in G, B, R order per SEI spec
        data[0..4].copy_from_slice(&encode_chromaticity(self.primaries[1]));  // Green
        data[4..8].copy_from_slice(&encode_chromaticity(self.primaries[2]));  // Blue
        data[8..12].copy_from_slice(&encode_chromaticity(self.primaries[0])); // Red
        data[12..16].copy_from_slice(&encode_chromaticity(self.white_point));

        // Encode luminance
        let max_lum = ((self.max_luminance / 0.0001).round() as u32).to_be_bytes();
        let min_lum = ((self.min_luminance / 0.0001).round() as u32).to_be_bytes();
        data[16..20].copy_from_slice(&max_lum);
        data[20..24].copy_from_slice(&min_lum);

        data
    }

    /// Get the dynamic range in stops.
    pub fn dynamic_range_stops(&self) -> f64 {
        if self.min_luminance <= 0.0 {
            return f64::INFINITY;
        }
        (self.max_luminance / self.min_luminance).log2()
    }
}

// ============================================================================
// HDR10 Static Metadata
// ============================================================================

/// Complete HDR10 static metadata.
#[derive(Debug, Clone, Default)]
pub struct Hdr10Metadata {
    /// Content light level info
    pub content_light_level: ContentLightLevel,
    /// Mastering display color volume
    pub mastering_display: MasteringDisplayColorVolume,
}

impl Hdr10Metadata {
    /// Create new HDR10 metadata.
    pub fn new(cll: ContentLightLevel, mdcv: MasteringDisplayColorVolume) -> Self {
        Self {
            content_light_level: cll,
            mastering_display: mdcv,
        }
    }

    /// Parse from combined SEI data (CLL followed by MDCV).
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 28 {
            return Err(HdrError::InvalidMetadata(
                "HDR10 metadata too short".into(),
            ));
        }

        let cll = ContentLightLevel::parse(&data[0..4])?;
        let mdcv = MasteringDisplayColorVolume::parse(&data[4..28])?;

        Ok(Self {
            content_light_level: cll,
            mastering_display: mdcv,
        })
    }
}

// ============================================================================
// HDR10+ Dynamic Metadata
// ============================================================================

/// HDR10+ processing window.
#[derive(Debug, Clone, Default)]
pub struct Hdr10PlusProcessingWindow {
    /// Upper left corner X (normalized 0-1)
    pub upper_left_x: f64,
    /// Upper left corner Y (normalized 0-1)
    pub upper_left_y: f64,
    /// Lower right corner X (normalized 0-1)
    pub lower_right_x: f64,
    /// Lower right corner Y (normalized 0-1)
    pub lower_right_y: f64,
}

impl Hdr10PlusProcessingWindow {
    /// Check if this is a full-frame window.
    pub fn is_full_frame(&self) -> bool {
        self.upper_left_x == 0.0
            && self.upper_left_y == 0.0
            && self.lower_right_x == 1.0
            && self.lower_right_y == 1.0
    }
}

/// HDR10+ distribution maxRGB percentile.
#[derive(Debug, Clone, Copy, Default)]
pub struct DistributionMaxrgb {
    /// Percentile value (0-100)
    pub percentile: u8,
    /// MaxRGB value at this percentile
    pub percentile_maxrgb: u32,
}

/// HDR10+ tone mapping curve anchor.
#[derive(Debug, Clone, Copy, Default)]
pub struct BezierCurveAnchor {
    /// Anchor value (normalized 0-1)
    pub anchor: f64,
}

/// HDR10+ dynamic metadata per frame.
#[derive(Debug, Clone, Default)]
pub struct Hdr10PlusDynamicMetadata {
    /// Processing window (usually full frame)
    pub processing_window: Hdr10PlusProcessingWindow,
    /// Targeted system display maximum luminance (nits)
    pub targeted_system_display_max_luminance: u32,
    /// Maximum of max RGB components (per processing window)
    pub maxscl: [u32; 3],
    /// Average maxRGB value
    pub average_maxrgb: u32,
    /// Distribution of maxRGB percentiles
    pub distribution_values: Vec<DistributionMaxrgb>,
    /// Fraction of selected percentile
    pub fraction_bright_pixels: f64,
    /// Knee point X (normalized 0-1)
    pub knee_point_x: f64,
    /// Knee point Y (normalized 0-1)
    pub knee_point_y: f64,
    /// Number of Bezier curve anchors
    pub num_bezier_curve_anchors: u8,
    /// Bezier curve anchors
    pub bezier_curve_anchors: Vec<BezierCurveAnchor>,
    /// Color saturation mapping flag
    pub color_saturation_mapping_flag: bool,
    /// Color saturation weight
    pub color_saturation_weight: f64,
}

impl Hdr10PlusDynamicMetadata {
    /// Get the maximum scene luminance from maxSCL values.
    pub fn max_scene_luminance(&self) -> u32 {
        *self.maxscl.iter().max().unwrap_or(&0)
    }

    /// Evaluate the tone mapping curve at a given input value.
    /// Input and output are normalized to 0-1.
    pub fn evaluate_curve(&self, input: f64) -> f64 {
        if self.bezier_curve_anchors.is_empty() {
            // No curve defined, use linear mapping
            return input;
        }

        // Simple piecewise linear interpolation between anchors
        // (Full Bezier evaluation would be more complex)
        let knee_x = self.knee_point_x;
        let knee_y = self.knee_point_y;

        if input <= knee_x {
            // Below knee point: linear
            input * knee_y / knee_x
        } else {
            // Above knee point: interpolate through anchors
            let normalized_input = (input - knee_x) / (1.0 - knee_x);

            // Find surrounding anchors
            let n = self.bezier_curve_anchors.len();
            let segment = (normalized_input * n as f64).floor() as usize;
            let segment = segment.min(n - 1);

            let start = if segment == 0 {
                knee_y
            } else {
                self.bezier_curve_anchors[segment - 1].anchor
            };

            let end = self.bezier_curve_anchors[segment].anchor;

            let t = (normalized_input * n as f64) - segment as f64;
            start + (end - start) * t
        }
    }

    /// Parse HDR10+ JSON metadata.
    /// Returns None if parsing fails (graceful degradation).
    pub fn parse_json(json_str: &str) -> Option<Self> {
        // Simplified JSON parsing - in production, use serde_json
        // This is a placeholder for the parsing logic
        let _ = json_str;
        None
    }

    /// Parse from raw HDR10+ SEI payload.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(HdrError::InvalidMetadata(
                "HDR10+ metadata too short".into(),
            ));
        }

        // Parse country code and terminal provider
        // ITU-T T.35 format
        let _country_code = data[0];
        let _terminal_provider_code = u16::from_be_bytes([data[1], data[2]]);
        let _terminal_provider_oriented_code = data[3];

        // Parse application identifier and version
        let _application_identifier = data[4];
        let application_version = data[5];

        if application_version < 1 {
            return Err(HdrError::InvalidMetadata(
                "Unsupported HDR10+ version".into(),
            ));
        }

        // Parse the rest based on version
        // This is a simplified parser - full implementation would parse all fields
        let mut metadata = Self::default();

        // Parse targeted system display max luminance (simplified)
        if data.len() >= 10 {
            metadata.targeted_system_display_max_luminance =
                u32::from_be_bytes([0, data[6], data[7], data[8]]);
        }

        Ok(metadata)
    }
}

// ============================================================================
// Dolby Vision RPU (Reference Processing Unit)
// ============================================================================

/// Dolby Vision profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DolbyVisionProfile {
    /// Profile 4: Single layer with cross-compatible HDR10 base
    Profile4,
    /// Profile 5: Single layer with IPT-PQ base
    Profile5,
    /// Profile 7: Dual layer with cross-compatible HDR10 base
    Profile7,
    /// Profile 8: Single layer with cross-compatible HDR10 base (MEL/FEL)
    Profile8,
    /// Unknown profile
    Unknown(u8),
}

impl From<u8> for DolbyVisionProfile {
    fn from(value: u8) -> Self {
        match value {
            4 => Self::Profile4,
            5 => Self::Profile5,
            7 => Self::Profile7,
            8 => Self::Profile8,
            other => Self::Unknown(other),
        }
    }
}

/// Dolby Vision RPU header.
#[derive(Debug, Clone)]
pub struct DolbyVisionRpuHeader {
    /// RPU type
    pub rpu_type: u8,
    /// RPU format
    pub rpu_format: u16,
    /// VDR RPU profile
    pub vdr_rpu_profile: DolbyVisionProfile,
    /// VDR RPU level
    pub vdr_rpu_level: u8,
    /// VDR sequence info present flag
    pub vdr_seq_info_present: bool,
    /// Chroma resampling explicit filter flag
    pub chroma_resampling_explicit_filter_flag: bool,
    /// Coefficient data type
    pub coefficient_data_type: u8,
    /// Coefficient log2 denominator
    pub coefficient_log2_denom: u8,
    /// VDR RPU normalized idc
    pub vdr_rpu_normalized_idc: u8,
    /// BL video full range flag
    pub bl_video_full_range_flag: bool,
    /// BL bit depth
    pub bl_bit_depth: u8,
    /// EL bit depth
    pub el_bit_depth: u8,
    /// VDR bit depth
    pub vdr_bit_depth: u8,
    /// Spatial resampling filter flag
    pub spatial_resampling_filter_flag: bool,
    /// EL spatial resampling filter flag
    pub el_spatial_resampling_filter_flag: bool,
    /// Disable residual flag
    pub disable_residual_flag: bool,
}

impl Default for DolbyVisionRpuHeader {
    fn default() -> Self {
        Self {
            rpu_type: 2,
            rpu_format: 0,
            vdr_rpu_profile: DolbyVisionProfile::Profile8,
            vdr_rpu_level: 0,
            vdr_seq_info_present: true,
            chroma_resampling_explicit_filter_flag: false,
            coefficient_data_type: 0,
            coefficient_log2_denom: 23,
            vdr_rpu_normalized_idc: 1,
            bl_video_full_range_flag: false,
            bl_bit_depth: 10,
            el_bit_depth: 10,
            vdr_bit_depth: 12,
            spatial_resampling_filter_flag: false,
            el_spatial_resampling_filter_flag: false,
            disable_residual_flag: false,
        }
    }
}

/// Dolby Vision RPU data.
#[derive(Debug, Clone, Default)]
pub struct DolbyVisionRpu {
    /// RPU header
    pub header: DolbyVisionRpuHeader,
    /// Raw coefficient data
    pub coefficient_data: Vec<u8>,
    /// Mapping polynomial coefficients (if parsed)
    pub mapping_coefficients: Option<DolbyVisionMappingCoefficients>,
}

/// Dolby Vision mapping coefficients for tone mapping.
#[derive(Debug, Clone, Default)]
pub struct DolbyVisionMappingCoefficients {
    /// Number of pivots in the mapping curve
    pub num_pivots: u8,
    /// Pivot values
    pub pivots: Vec<f64>,
    /// Polynomial order for each piece
    pub poly_order: Vec<u8>,
    /// Polynomial coefficients
    pub poly_coef: Vec<Vec<f64>>,
}

impl DolbyVisionRpu {
    /// Parse RPU from raw data.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            return Err(HdrError::InvalidMetadata(
                "Dolby Vision RPU too short".into(),
            ));
        }

        // Parse RPU header (simplified)
        let rpu_type = (data[0] >> 6) & 0x03;
        let rpu_format = u16::from(data[0] & 0x3F) << 8 | u16::from(data[1]);

        let header = DolbyVisionRpuHeader {
            rpu_type,
            rpu_format,
            ..Default::default()
        };

        Ok(Self {
            header,
            coefficient_data: data[2..].to_vec(),
            mapping_coefficients: None,
        })
    }

    /// Check if this is a profile 8 RPU (cross-compatible with HDR10).
    pub fn is_profile_8(&self) -> bool {
        matches!(self.header.vdr_rpu_profile, DolbyVisionProfile::Profile8)
    }

    /// Check if this is a dual-layer profile.
    pub fn is_dual_layer(&self) -> bool {
        matches!(self.header.vdr_rpu_profile, DolbyVisionProfile::Profile7)
    }
}

// ============================================================================
// Metadata Container
// ============================================================================

/// Combined HDR metadata container supporting all formats.
#[derive(Debug, Clone, Default)]
pub struct HdrMetadataContainer {
    /// HDR10 static metadata
    pub hdr10: Option<Hdr10Metadata>,
    /// HDR10+ dynamic metadata (per-frame)
    pub hdr10_plus: Option<Hdr10PlusDynamicMetadata>,
    /// Dolby Vision RPU
    pub dolby_vision: Option<DolbyVisionRpu>,
}

impl HdrMetadataContainer {
    /// Create empty container.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if any HDR metadata is present.
    pub fn has_hdr_metadata(&self) -> bool {
        self.hdr10.is_some() || self.hdr10_plus.is_some() || self.dolby_vision.is_some()
    }

    /// Get the maximum content light level from available metadata.
    pub fn max_content_light_level(&self) -> Option<u16> {
        if let Some(ref hdr10) = self.hdr10 {
            return Some(hdr10.content_light_level.max_cll);
        }
        if let Some(ref hdr10_plus) = self.hdr10_plus {
            return Some(hdr10_plus.max_scene_luminance() as u16);
        }
        None
    }

    /// Get the maximum frame-average light level from available metadata.
    pub fn max_frame_average_light_level(&self) -> Option<u16> {
        if let Some(ref hdr10) = self.hdr10 {
            return Some(hdr10.content_light_level.max_fall);
        }
        if let Some(ref hdr10_plus) = self.hdr10_plus {
            return Some(hdr10_plus.average_maxrgb as u16);
        }
        None
    }

    /// Get the mastering display peak luminance.
    pub fn mastering_display_peak(&self) -> Option<f64> {
        self.hdr10.as_ref().map(|m| m.mastering_display.max_luminance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_light_level_parse() {
        let data = [0x03, 0xE8, 0x01, 0xF4]; // MaxCLL=1000, MaxFALL=500
        let cll = ContentLightLevel::parse(&data).unwrap();
        assert_eq!(cll.max_cll, 1000);
        assert_eq!(cll.max_fall, 500);
    }

    #[test]
    fn test_content_light_level_roundtrip() {
        let cll = ContentLightLevel::new(1000, 400);
        let bytes = cll.to_bytes();
        let parsed = ContentLightLevel::parse(&bytes).unwrap();
        assert_eq!(cll.max_cll, parsed.max_cll);
        assert_eq!(cll.max_fall, parsed.max_fall);
    }

    #[test]
    fn test_content_light_level_is_hdr() {
        let sdr = ContentLightLevel::new(80, 50);
        assert!(!sdr.is_hdr());

        let hdr = ContentLightLevel::new(1000, 400);
        assert!(hdr.is_hdr());
    }

    #[test]
    fn test_mdcv_default() {
        let mdcv = MasteringDisplayColorVolume::default();
        assert_eq!(mdcv.max_luminance, 1000.0);
        assert!(mdcv.min_luminance < 0.001);
    }

    #[test]
    fn test_mdcv_dynamic_range() {
        let mdcv = MasteringDisplayColorVolume {
            max_luminance: 1000.0,
            min_luminance: 0.001,
            ..Default::default()
        };
        let dr = mdcv.dynamic_range_stops();
        assert!(dr > 19.0 && dr < 21.0); // Approximately 20 stops
    }

    #[test]
    fn test_hdr10_plus_evaluate_curve() {
        let metadata = Hdr10PlusDynamicMetadata {
            knee_point_x: 0.5,
            knee_point_y: 0.75,
            bezier_curve_anchors: vec![
                BezierCurveAnchor { anchor: 0.9 },
                BezierCurveAnchor { anchor: 1.0 },
            ],
            ..Default::default()
        };

        // Below knee should be linear-ish
        let below = metadata.evaluate_curve(0.25);
        assert!(below < 0.5);

        // At knee point
        let at_knee = metadata.evaluate_curve(0.5);
        assert!((at_knee - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_processing_window_full_frame() {
        let window = Hdr10PlusProcessingWindow {
            upper_left_x: 0.0,
            upper_left_y: 0.0,
            lower_right_x: 1.0,
            lower_right_y: 1.0,
        };
        assert!(window.is_full_frame());

        let partial = Hdr10PlusProcessingWindow {
            upper_left_x: 0.1,
            upper_left_y: 0.1,
            lower_right_x: 0.9,
            lower_right_y: 0.9,
        };
        assert!(!partial.is_full_frame());
    }

    #[test]
    fn test_dolby_vision_profile() {
        assert_eq!(DolbyVisionProfile::from(8), DolbyVisionProfile::Profile8);
        assert_eq!(DolbyVisionProfile::from(5), DolbyVisionProfile::Profile5);
        assert!(matches!(DolbyVisionProfile::from(99), DolbyVisionProfile::Unknown(99)));
    }

    #[test]
    fn test_metadata_container() {
        let mut container = HdrMetadataContainer::new();
        assert!(!container.has_hdr_metadata());

        container.hdr10 = Some(Hdr10Metadata {
            content_light_level: ContentLightLevel::new(1000, 400),
            mastering_display: MasteringDisplayColorVolume::default(),
        });

        assert!(container.has_hdr_metadata());
        assert_eq!(container.max_content_light_level(), Some(1000));
        assert_eq!(container.max_frame_average_light_level(), Some(400));
    }
}
