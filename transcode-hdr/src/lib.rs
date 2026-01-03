//! HDR pipeline for transcode
//!
//! This crate provides comprehensive HDR (High Dynamic Range) video processing support including:
//!
//! - **Transfer Functions**: PQ (ST.2084), HLG (ARIB STD-B67), SDR gamma, BT.1886
//! - **Color Spaces**: BT.709, BT.2020, DCI-P3, Display P3
//! - **Tone Mapping**: Reinhard, Hable, ACES, BT.2390, Mobius
//! - **Metadata**: HDR10, HDR10+, Dolby Vision RPU
//! - **Gamut Mapping**: Soft clipping, desaturation, ACES compression
//! - **Processing Pipeline**: HDR-to-SDR, SDR-to-HDR, HDR format conversion
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use transcode_hdr::{HdrProcessor, HdrProcessorConfig};
//!
//! // Create an HDR10 to SDR processor
//! let mut processor = HdrProcessor::hdr10_to_sdr();
//!
//! // Process a frame (RGB triplets, PQ-encoded)
//! let mut pixels = vec![0.5, 0.5, 0.5, 0.3, 0.7, 0.4];
//! processor.process_frame(&mut pixels).unwrap();
//! ```
//!
//! # Modules
//!
//! - [`colorspace`] - Color space definitions, primaries, and conversion matrices
//! - [`transfer`] - Transfer function implementations (EOTF/OETF)
//! - [`tonemapping`] - Tone mapping operators for HDR to SDR conversion
//! - [`metadata`] - HDR metadata parsing (HDR10, HDR10+, Dolby Vision)
//! - [`gamut`] - Gamut mapping for out-of-gamut color handling
//! - [`processor`] - Complete HDR processing pipeline
//! - [`error`] - Error types for HDR operations

pub mod colorspace;
pub mod error;
pub mod gamut;
pub mod metadata;
pub mod processor;
pub mod tonemapping;
pub mod transfer;

// Re-export main types from error module
pub use error::{HdrError, Result};

// Re-export colorspace types
pub use colorspace::{
    ColorPrimaries, ColorSpaceConverter, MatrixCoefficients,
    BT709_PRIMARIES, BT2020_PRIMARIES, DISPLAY_P3_PRIMARIES, DCI_P3_PRIMARIES,
    BT709_MATRIX, BT2020_MATRIX, BT601_MATRIX,
    get_primaries, get_matrix_coefficients,
    luminance_bt709, luminance_bt2020,
};

// Re-export transfer function types
pub use transfer::{
    TransferConverter,
    pq_eotf, pq_oetf, pq_eotf_nits, pq_oetf_nits,
    hlg_eotf, hlg_oetf, hlg_ootf, hlg_system_gamma,
    bt1886_eotf, bt1886_oetf,
    srgb_eotf, srgb_oetf,
    gamma_eotf, gamma_oetf,
    convert_transfer, is_hdr, reference_white,
    nits_to_pq, pq_to_nits,
};

// Re-export tonemapping types
pub use tonemapping::{
    ToneMapAlgorithm, ToneMapConfig, ToneMapper,
    reinhard_global, reinhard_extended, reinhard_local_pixel,
    hable_filmic, hable_filmic_custom,
    aces_filmic, aces_filmic_exposed,
    bt2390_eetf, bt2390_eetf_full,
    mobius, mobius_linear,
    calculate_key_value, auto_exposure, percentile_luminance,
};

// Re-export metadata types
pub use metadata::{
    ContentLightLevel, MasteringDisplayColorVolume, Hdr10Metadata,
    Hdr10PlusDynamicMetadata, Hdr10PlusProcessingWindow,
    DistributionMaxrgb, BezierCurveAnchor,
    DolbyVisionProfile, DolbyVisionRpu, DolbyVisionRpuHeader,
    DolbyVisionMappingCoefficients,
    HdrMetadataContainer,
};

// Re-export gamut types
pub use gamut::{
    GamutMappingAlgorithm, GamutMapConfig, GamutMapper,
    GamutAnalysis, analyze_gamut,
    is_in_gamut, is_in_gamut_with_tolerance, gamut_distance,
    clip_to_gamut, soft_clip,
    desaturate_to_white, desaturate_to_neutral,
    aces_gamut_compress, bt2407_gamut_map, cusp_gamut_map,
};

// Re-export processor types
pub use processor::{
    ProcessingMode, HdrProcessorConfig, HdrProcessor, BatchProcessor,
    quick,
};

/// HDR format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HdrFormat {
    /// SDR (no HDR)
    #[default]
    Sdr,
    /// HDR10 (static metadata)
    Hdr10,
    /// HDR10+ (dynamic metadata)
    Hdr10Plus,
    /// Dolby Vision
    DolbyVision,
    /// Hybrid Log-Gamma
    Hlg,
    /// PQ (Perceptual Quantizer)
    Pq,
}

impl HdrFormat {
    /// Check if this format is HDR.
    pub fn is_hdr(&self) -> bool {
        !matches!(self, Self::Sdr)
    }

    /// Check if this format uses PQ transfer function.
    pub fn uses_pq(&self) -> bool {
        matches!(self, Self::Hdr10 | Self::Hdr10Plus | Self::DolbyVision | Self::Pq)
    }

    /// Check if this format uses HLG transfer function.
    pub fn uses_hlg(&self) -> bool {
        matches!(self, Self::Hlg)
    }

    /// Check if this format has dynamic metadata.
    pub fn has_dynamic_metadata(&self) -> bool {
        matches!(self, Self::Hdr10Plus | Self::DolbyVision)
    }

    /// Get the typical color space for this format.
    pub fn typical_colorspace(&self) -> ColorSpace {
        match self {
            Self::Sdr => ColorSpace::Bt709,
            _ => ColorSpace::Bt2020,
        }
    }

    /// Get the typical transfer function for this format.
    pub fn typical_transfer(&self) -> TransferFunction {
        match self {
            Self::Sdr => TransferFunction::Gamma22,
            Self::Hlg => TransferFunction::Hlg,
            _ => TransferFunction::Pq,
        }
    }
}

/// Color space
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorSpace {
    /// BT.709 (SDR, same primaries as sRGB)
    #[default]
    Bt709,
    /// BT.2020 (HDR wide gamut)
    Bt2020,
    /// Display P3 (wide gamut with D65 white)
    DisplayP3,
    /// DCI-P3 (theatrical)
    DciP3,
    /// sRGB (same primaries as BT.709)
    Srgb,
}

impl ColorSpace {
    /// Check if this is a wide gamut color space.
    pub fn is_wide_gamut(&self) -> bool {
        matches!(self, Self::Bt2020 | Self::DisplayP3 | Self::DciP3)
    }

    /// Get the approximate gamut coverage relative to BT.709.
    pub fn gamut_coverage(&self) -> f64 {
        match self {
            Self::Bt709 | Self::Srgb => 1.0,
            Self::DisplayP3 => 1.25,
            Self::DciP3 => 1.26,
            Self::Bt2020 => 1.75,
        }
    }
}

/// Transfer function
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransferFunction {
    /// Gamma 2.2 (typical SDR)
    #[default]
    Gamma22,
    /// Gamma 2.4 (BT.1886 reference)
    Gamma24,
    /// sRGB transfer function
    Srgb,
    /// PQ (SMPTE ST 2084)
    Pq,
    /// HLG (ARIB STD-B67)
    Hlg,
    /// Linear light
    Linear,
    /// BT.1886 (SDR reference display)
    Bt1886,
}

impl TransferFunction {
    /// Check if this is an HDR transfer function.
    pub fn is_hdr(&self) -> bool {
        matches!(self, Self::Pq | Self::Hlg)
    }

    /// Get the reference peak luminance for this transfer function (nits).
    pub fn peak_luminance(&self) -> f64 {
        match self {
            Self::Pq => 10000.0,
            Self::Hlg => 1000.0,
            _ => 100.0,
        }
    }
}

/// HDR pipeline configuration
#[derive(Debug, Clone)]
pub struct HdrConfig {
    /// Target format (for conversion)
    pub target_format: HdrFormat,
    /// Target color space
    pub target_color_space: ColorSpace,
    /// Target transfer function
    pub target_transfer: TransferFunction,
    /// Target peak luminance (nits)
    pub target_peak_nits: f64,
    /// Tone mapping algorithm
    pub tone_map_algorithm: ToneMapAlgorithm,
    /// Gamut mapping algorithm
    pub gamut_map_algorithm: GamutMappingAlgorithm,
    /// Use dynamic metadata if available
    pub use_dynamic_metadata: bool,
}

impl Default for HdrConfig {
    fn default() -> Self {
        Self {
            target_format: HdrFormat::Sdr,
            target_color_space: ColorSpace::Bt709,
            target_transfer: TransferFunction::Gamma22,
            target_peak_nits: 100.0,
            tone_map_algorithm: ToneMapAlgorithm::Bt2390,
            gamut_map_algorithm: GamutMappingAlgorithm::SoftClip,
            use_dynamic_metadata: true,
        }
    }
}

impl HdrConfig {
    /// Create configuration for HDR10 to SDR conversion.
    pub fn hdr10_to_sdr() -> Self {
        Self::default()
    }

    /// Create configuration for HLG to SDR conversion.
    pub fn hlg_to_sdr() -> Self {
        Self {
            target_format: HdrFormat::Sdr,
            target_color_space: ColorSpace::Bt709,
            target_transfer: TransferFunction::Gamma22,
            target_peak_nits: 100.0,
            ..Default::default()
        }
    }

    /// Create configuration for SDR to HDR10.
    pub fn sdr_to_hdr10() -> Self {
        Self {
            target_format: HdrFormat::Hdr10,
            target_color_space: ColorSpace::Bt2020,
            target_transfer: TransferFunction::Pq,
            target_peak_nits: 1000.0,
            ..Default::default()
        }
    }
}

/// HDR metadata container (legacy compatibility)
#[derive(Debug, Clone)]
pub struct HdrMetadata {
    /// HDR format
    pub format: HdrFormat,
    /// Color space
    pub color_space: ColorSpace,
    /// Transfer function
    pub transfer_function: TransferFunction,
    /// Maximum content light level (nits)
    pub max_cll: u16,
    /// Maximum frame-average light level (nits)
    pub max_fall: u16,
    /// Master display info
    pub master_display: Option<MasterDisplay>,
}

impl Default for HdrMetadata {
    fn default() -> Self {
        Self {
            format: HdrFormat::Sdr,
            color_space: ColorSpace::Bt709,
            transfer_function: TransferFunction::Gamma22,
            max_cll: 0,
            max_fall: 0,
            master_display: None,
        }
    }
}

impl HdrMetadata {
    /// Check if this represents HDR content.
    pub fn is_hdr(&self) -> bool {
        self.format.is_hdr() || self.max_cll > 100
    }

    /// Convert to the new Hdr10Metadata type.
    pub fn to_hdr10_metadata(&self) -> Hdr10Metadata {
        Hdr10Metadata {
            content_light_level: ContentLightLevel::new(self.max_cll, self.max_fall),
            mastering_display: self.master_display.as_ref().map(|md| {
                MasteringDisplayColorVolume::new(
                    md.primaries,
                    md.white_point,
                    md.max_luminance,
                    md.min_luminance,
                )
            }).unwrap_or_default(),
        }
    }
}

/// Master display color volume info (legacy compatibility)
#[derive(Debug, Clone)]
pub struct MasterDisplay {
    /// Display primaries (x, y pairs for RGB)
    pub primaries: [[f64; 2]; 3],
    /// White point (x, y)
    pub white_point: [f64; 2],
    /// Maximum luminance (nits)
    pub max_luminance: f64,
    /// Minimum luminance (nits)
    pub min_luminance: f64,
}

impl Default for MasterDisplay {
    fn default() -> Self {
        // BT.2020 primaries
        Self {
            primaries: [
                [0.708, 0.292], // Red
                [0.170, 0.797], // Green
                [0.131, 0.046], // Blue
            ],
            white_point: [0.3127, 0.3290], // D65
            max_luminance: 1000.0,
            min_luminance: 0.0001,
        }
    }
}

/// HDR processing pipeline (legacy compatibility)
pub struct HdrPipeline {
    processor: HdrProcessor,
    config: HdrConfig,
}

impl HdrPipeline {
    /// Create a new HDR pipeline
    pub fn new(config: HdrConfig) -> Self {
        let processor_config = HdrProcessorConfig {
            mode: if config.target_format == HdrFormat::Sdr {
                ProcessingMode::HdrToSdr
            } else if config.target_format.is_hdr() {
                ProcessingMode::SdrToHdr
            } else {
                ProcessingMode::Passthrough
            },
            target_format: config.target_format,
            target_colorspace: config.target_color_space,
            target_transfer: config.target_transfer,
            target_peak_nits: config.target_peak_nits,
            tone_map_algorithm: config.tone_map_algorithm,
            gamut_map_algorithm: config.gamut_map_algorithm,
            use_dynamic_metadata: config.use_dynamic_metadata,
            ..Default::default()
        };

        Self {
            processor: HdrProcessor::new(processor_config),
            config,
        }
    }

    /// Process RGB pixel data (linear space)
    pub fn process(&self, pixels: &mut [f32], _metadata: &HdrMetadata) -> Result<()> {
        // Convert to f64, process, convert back
        let mut f64_pixels: Vec<f64> = pixels.iter().map(|&v| v as f64).collect();

        for chunk in f64_pixels.chunks_exact_mut(3) {
            let (r, g, b) = self.processor.process_pixel(chunk[0], chunk[1], chunk[2]);
            chunk[0] = r;
            chunk[1] = g;
            chunk[2] = b;
        }

        for (i, &v) in f64_pixels.iter().enumerate() {
            pixels[i] = v as f32;
        }

        Ok(())
    }

    /// Get configuration
    pub fn config(&self) -> &HdrConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdr_format() {
        assert!(!HdrFormat::Sdr.is_hdr());
        assert!(HdrFormat::Hdr10.is_hdr());
        assert!(HdrFormat::Hdr10.uses_pq());
        assert!(HdrFormat::Hlg.uses_hlg());
        assert!(HdrFormat::Hdr10Plus.has_dynamic_metadata());
    }

    #[test]
    fn test_color_space() {
        assert!(!ColorSpace::Bt709.is_wide_gamut());
        assert!(ColorSpace::Bt2020.is_wide_gamut());
        assert!(ColorSpace::Bt2020.gamut_coverage() > ColorSpace::Bt709.gamut_coverage());
    }

    #[test]
    fn test_transfer_function() {
        assert!(!TransferFunction::Gamma22.is_hdr());
        assert!(TransferFunction::Pq.is_hdr());
        assert!(TransferFunction::Hlg.is_hdr());
        assert_eq!(TransferFunction::Pq.peak_luminance(), 10000.0);
    }

    #[test]
    fn test_hdr_config() {
        let config = HdrConfig::hdr10_to_sdr();
        assert_eq!(config.target_format, HdrFormat::Sdr);
        assert_eq!(config.target_color_space, ColorSpace::Bt709);
    }

    #[test]
    fn test_hdr_metadata() {
        let meta = HdrMetadata::default();
        assert!(!meta.is_hdr());

        let hdr_meta = HdrMetadata {
            format: HdrFormat::Hdr10,
            max_cll: 1000,
            max_fall: 400,
            ..Default::default()
        };
        assert!(hdr_meta.is_hdr());
    }

    #[test]
    fn test_hdr_pipeline() {
        let config = HdrConfig::hdr10_to_sdr();
        let pipeline = HdrPipeline::new(config);

        let metadata = HdrMetadata {
            format: HdrFormat::Hdr10,
            color_space: ColorSpace::Bt2020,
            transfer_function: TransferFunction::Pq,
            max_cll: 1000,
            max_fall: 400,
            master_display: Some(MasterDisplay::default()),
        };

        let mut pixels = vec![0.5f32, 0.5, 0.5];
        pipeline.process(&mut pixels, &metadata).unwrap();

        assert!(pixels.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_pq_roundtrip() {
        for l in [0.0, 0.1, 0.5, 1.0] {
            let encoded = pq_oetf(l);
            let decoded = pq_eotf(encoded);
            assert!((l - decoded).abs() < 1e-6, "PQ roundtrip failed for {}", l);
        }
    }

    #[test]
    fn test_hlg_roundtrip() {
        for l in [0.0, 0.05, 0.1, 0.5, 1.0] {
            let encoded = hlg_oetf(l);
            let decoded = hlg_eotf(encoded);
            assert!((l - decoded).abs() < 0.01, "HLG roundtrip failed for {}", l);
        }
    }

    #[test]
    fn test_color_space_conversion() {
        let converter = ColorSpaceConverter::new(ColorSpace::Bt2020, ColorSpace::Bt709);

        // White should stay white
        let (r, g, b) = converter.convert(1.0, 1.0, 1.0);
        assert!((r - 1.0).abs() < 0.01);
        assert!((g - 1.0).abs() < 0.01);
        assert!((b - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_tone_mapping() {
        let mapper = ToneMapper::simple(ToneMapAlgorithm::Reinhard, 1000.0, 100.0);

        // Tone mapping should compress values
        let (r, g, b) = mapper.map_rgb(1.0, 1.0, 1.0);
        assert!(r < 1.0);
        assert!(g < 1.0);
        assert!(b < 1.0);
    }

    #[test]
    fn test_gamut_mapping() {
        let mapper = GamutMapper::bt2020_to_bt709();

        // Out-of-gamut color should be mapped into gamut
        let (r, g, b) = mapper.map(1.5, -0.1, 0.5);
        assert!(is_in_gamut(r, g, b));
    }
}
