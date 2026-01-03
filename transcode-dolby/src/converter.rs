//! Profile conversion utilities.
//!
//! This module provides functionality for converting between Dolby Vision profiles,
//! including Profile 7 to Profile 8 conversion, Profile 5 to Profile 8.1,
//! and dual-layer to single-layer conversion.

use crate::error::{DolbyError, Result};
use crate::metadata::{DolbyVisionMetadata, L1Metadata, L2Metadata, L6Metadata};
use crate::profile::{DolbyVisionProfile, SubProfile};
use crate::rpu::{ExtensionBlock, Rpu, RpuHeader};

/// Profile converter for Dolby Vision streams.
#[derive(Debug)]
pub struct ProfileConverter {
    /// Source profile.
    source_profile: DolbyVisionProfile,
    /// Target profile.
    target_profile: DolbyVisionProfile,
    /// Target sub-profile (for Profile 8).
    target_sub_profile: Option<SubProfile>,
    /// Conversion options.
    options: ConversionOptions,
}

impl ProfileConverter {
    /// Create a new profile converter.
    pub fn new(
        source: DolbyVisionProfile,
        target: DolbyVisionProfile,
    ) -> Result<Self> {
        if !source.can_convert_to(target) {
            return Err(DolbyError::ConversionError {
                from: source.as_u8(),
                to: target.as_u8(),
                message: "Unsupported conversion path".to_string(),
            });
        }

        Ok(ProfileConverter {
            source_profile: source,
            target_profile: target,
            target_sub_profile: None,
            options: ConversionOptions::default(),
        })
    }

    /// Create a converter from Profile 7 to Profile 8.
    pub fn profile7_to_8() -> Self {
        ProfileConverter {
            source_profile: DolbyVisionProfile::Profile7,
            target_profile: DolbyVisionProfile::Profile8,
            target_sub_profile: Some(SubProfile::Profile8_1),
            options: ConversionOptions::default(),
        }
    }

    /// Create a converter from Profile 5 to Profile 8.1.
    pub fn profile5_to_8_1() -> Self {
        ProfileConverter {
            source_profile: DolbyVisionProfile::Profile5,
            target_profile: DolbyVisionProfile::Profile8,
            target_sub_profile: Some(SubProfile::Profile8_1),
            options: ConversionOptions::default(),
        }
    }

    /// Set target sub-profile (for Profile 8 conversions).
    pub fn with_sub_profile(mut self, sub: SubProfile) -> Self {
        self.target_sub_profile = Some(sub);
        self
    }

    /// Set conversion options.
    pub fn with_options(mut self, options: ConversionOptions) -> Self {
        self.options = options;
        self
    }

    /// Convert an RPU to the target profile.
    pub fn convert_rpu(&self, rpu: &Rpu) -> Result<Rpu> {
        match (self.source_profile, self.target_profile) {
            (DolbyVisionProfile::Profile7, DolbyVisionProfile::Profile8) => {
                self.convert_7_to_8(rpu)
            }
            (DolbyVisionProfile::Profile5, DolbyVisionProfile::Profile8) => {
                self.convert_5_to_8(rpu)
            }
            (a, b) if a == b => {
                // Same profile, just clone
                Ok(rpu.clone())
            }
            _ => Err(DolbyError::ConversionError {
                from: self.source_profile.as_u8(),
                to: self.target_profile.as_u8(),
                message: "Conversion not implemented".to_string(),
            }),
        }
    }

    /// Convert Profile 7 (dual-layer) to Profile 8 (single-layer).
    fn convert_7_to_8(&self, rpu: &Rpu) -> Result<Rpu> {
        let mut new_rpu = rpu.clone();

        // Update header for Profile 8
        new_rpu.header = self.create_profile8_header(&rpu.header);

        // Remove enhancement layer data
        new_rpu.header.el_spatial_resampling_filter_flag = false;
        new_rpu.header.disable_residual_flag = Some(true);

        // Preserve VDR data but simplify
        if let Some(ref mut vdr) = new_rpu.vdr {
            // Keep polynomial coefficients for tone mapping
            // Remove NLQ data (enhancement layer specific)
            vdr.nlq_data = None;
        }

        // Preserve metadata
        self.preserve_metadata(&mut new_rpu.metadata)?;

        Ok(new_rpu)
    }

    /// Convert Profile 5 (SDR compatible) to Profile 8.
    fn convert_5_to_8(&self, rpu: &Rpu) -> Result<Rpu> {
        let mut new_rpu = rpu.clone();

        // Update header for Profile 8
        new_rpu.header = self.create_profile8_header(&rpu.header);

        // Profile 5 is already single-layer, mainly need to update transfer function
        // from SDR to HDR10

        // Remap L1 metadata if needed
        if let Some(ref mut l1) = new_rpu.metadata.l1 {
            // SDR content has limited dynamic range
            // Map SDR black/white to appropriate PQ values
            if self.options.remap_sdr_to_hdr {
                *l1 = self.remap_l1_sdr_to_hdr(l1);
            }
        }

        // Add L2 trim for SDR target if not present
        if new_rpu.metadata.l2.is_empty() && self.options.generate_l2_trim {
            new_rpu.metadata.l2.push(L2Metadata::default());
        }

        // Preserve other metadata
        self.preserve_metadata(&mut new_rpu.metadata)?;

        Ok(new_rpu)
    }

    fn create_profile8_header(&self, source: &RpuHeader) -> RpuHeader {
        RpuHeader {
            rpu_type: 2,
            rpu_format: 18, // Standard Profile 8 format
            guessed_profile: 8,
            bl_video_full_range_flag: source.bl_video_full_range_flag,
            bl_bit_depth_minus8: 2, // 10-bit
            el_bit_depth_minus8: 2,
            vdr_bit_depth_minus8: 4, // 12-bit processing
            el_spatial_resampling_filter_flag: false,
            disable_residual_flag: Some(true),
            scene_refresh_flag: source.scene_refresh_flag,
            coefficient_data_type: source.coefficient_data_type,
            mapping_idc: source.mapping_idc.clone(),
            num_pivots_minus2: source.num_pivots_minus2,
            nlq_method_idc: None,
            nlq_num_pivots_minus2: None,
        }
    }

    fn remap_l1_sdr_to_hdr(&self, l1: &L1Metadata) -> L1Metadata {
        // SDR content is typically mastered for ~100 nits peak
        // Remap to reasonable HDR values
        let sdr_peak_pq = 2081; // ~100 nits
        let sdr_black_pq = 64;  // ~0.005 nits

        // Scale the values
        let scale = sdr_peak_pq as f64 / 4095.0;

        L1Metadata {
            min_pq: ((l1.min_pq as f64 * scale) as u16).max(sdr_black_pq),
            max_pq: ((l1.max_pq as f64 * scale) as u16).min(sdr_peak_pq),
            avg_pq: (l1.avg_pq as f64 * scale) as u16,
        }
    }

    fn preserve_metadata(&self, metadata: &mut DolbyVisionMetadata) -> Result<()> {
        // Ensure L1 is present
        if metadata.l1.is_none() && self.options.generate_default_l1 {
            metadata.l1 = Some(L1Metadata::default());
        }

        // Validate L6 if present
        if let Some(ref l6) = metadata.l6 {
            if l6.max_fall > l6.max_cll {
                // Fix invalid L6
                let fixed_l6 = L6Metadata::new_unchecked(l6.max_cll, l6.max_cll.min(l6.max_fall));
                metadata.l6 = Some(fixed_l6);
            }
        }

        Ok(())
    }

    /// Get source profile.
    pub fn source_profile(&self) -> DolbyVisionProfile {
        self.source_profile
    }

    /// Get target profile.
    pub fn target_profile(&self) -> DolbyVisionProfile {
        self.target_profile
    }
}

/// Options for profile conversion.
#[derive(Debug, Clone)]
pub struct ConversionOptions {
    /// Generate default L1 metadata if missing.
    pub generate_default_l1: bool,
    /// Generate L2 trim passes if missing.
    pub generate_l2_trim: bool,
    /// Remap SDR L1 values to HDR range (for Profile 5 to 8).
    pub remap_sdr_to_hdr: bool,
    /// Preserve L5 active area metadata.
    pub preserve_active_area: bool,
    /// Preserve L6 MaxCLL/MaxFALL.
    pub preserve_static_metadata: bool,
    /// Target peak luminance for generated metadata.
    pub target_peak_nits: f64,
}

impl Default for ConversionOptions {
    fn default() -> Self {
        ConversionOptions {
            generate_default_l1: true,
            generate_l2_trim: true,
            remap_sdr_to_hdr: true,
            preserve_active_area: true,
            preserve_static_metadata: true,
            target_peak_nits: 1000.0,
        }
    }
}

/// Dual-layer to single-layer converter.
///
/// Handles conversion of dual-layer Dolby Vision streams (Profile 7)
/// to single-layer streams (Profile 8) by removing the enhancement layer
/// and preserving metadata.
#[derive(Debug)]
pub struct DualToSingleConverter {
    /// Profile converter.
    converter: ProfileConverter,
    /// Enhancement layer processing mode.
    el_mode: EnhancementLayerMode,
}

impl DualToSingleConverter {
    /// Create a new dual to single layer converter.
    pub fn new() -> Self {
        DualToSingleConverter {
            converter: ProfileConverter::profile7_to_8(),
            el_mode: EnhancementLayerMode::Discard,
        }
    }

    /// Set enhancement layer processing mode.
    pub fn with_el_mode(mut self, mode: EnhancementLayerMode) -> Self {
        self.el_mode = mode;
        self
    }

    /// Convert an RPU from dual-layer to single-layer.
    pub fn convert(&self, rpu: &Rpu) -> Result<Rpu> {
        // Verify source is dual-layer
        if !rpu.has_enhancement_layer() {
            return Err(DolbyError::DualToSingleError {
                message: "Source RPU does not have enhancement layer".to_string(),
            });
        }

        let mut result = self.converter.convert_rpu(rpu)?;

        match self.el_mode {
            EnhancementLayerMode::Discard => {
                // Already handled by converter
            }
            EnhancementLayerMode::BakeIn => {
                // Bake enhancement layer data into base layer
                // This would require actual pixel processing
                // For now, just adjust metadata
                self.adjust_metadata_for_bake_in(&mut result.metadata);
            }
        }

        Ok(result)
    }

    fn adjust_metadata_for_bake_in(&self, metadata: &mut DolbyVisionMetadata) {
        // When baking in enhancement layer, the dynamic range
        // might be better preserved in the base layer
        // Adjust L1 to reflect this
        if let Some(ref mut l1) = metadata.l1 {
            // Slightly increase the max to account for EL contribution
            l1.max_pq = (l1.max_pq as u32 + 100).min(4095) as u16;
        }
    }
}

impl Default for DualToSingleConverter {
    fn default() -> Self {
        DualToSingleConverter::new()
    }
}

/// Enhancement layer processing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnhancementLayerMode {
    /// Discard enhancement layer completely.
    Discard,
    /// Bake enhancement layer into base layer (requires pixel processing).
    BakeIn,
}

/// Metadata preservation helper.
#[derive(Debug)]
pub struct MetadataPreserver {
    /// Preserve L1 metadata.
    pub preserve_l1: bool,
    /// Preserve L2 metadata.
    pub preserve_l2: bool,
    /// Preserve L5 metadata.
    pub preserve_l5: bool,
    /// Preserve L6 metadata.
    pub preserve_l6: bool,
    /// Preserve L8 metadata.
    pub preserve_l8: bool,
    /// Preserve L9 metadata.
    pub preserve_l9: bool,
    /// Preserve L11 metadata.
    pub preserve_l11: bool,
}

impl MetadataPreserver {
    /// Create preserver that preserves all metadata.
    pub fn all() -> Self {
        MetadataPreserver {
            preserve_l1: true,
            preserve_l2: true,
            preserve_l5: true,
            preserve_l6: true,
            preserve_l8: true,
            preserve_l9: true,
            preserve_l11: true,
        }
    }

    /// Create preserver that only preserves essential metadata.
    pub fn essential() -> Self {
        MetadataPreserver {
            preserve_l1: true,
            preserve_l2: true,
            preserve_l5: false,
            preserve_l6: true,
            preserve_l8: false,
            preserve_l9: false,
            preserve_l11: false,
        }
    }

    /// Copy metadata from source to target based on preservation settings.
    pub fn copy_metadata(
        &self,
        source: &DolbyVisionMetadata,
        target: &mut DolbyVisionMetadata,
    ) {
        if self.preserve_l1 && source.l1.is_some() {
            target.l1 = source.l1;
        }

        if self.preserve_l2 {
            target.l2 = source.l2.clone();
        }

        if self.preserve_l5 {
            target.l5 = source.l5;
        }

        if self.preserve_l6 {
            target.l6 = source.l6;
        }

        if self.preserve_l8 {
            target.l8 = source.l8.clone();
        }

        if self.preserve_l9 {
            target.l9 = source.l9;
        }

        if self.preserve_l11 {
            target.l11 = source.l11;
        }
    }

    /// Create extension blocks from metadata.
    pub fn create_extension_blocks(
        &self,
        metadata: &DolbyVisionMetadata,
    ) -> Vec<ExtensionBlock> {
        let mut blocks = Vec::new();

        if self.preserve_l1 {
            if let Some(l1) = metadata.l1 {
                blocks.push(ExtensionBlock::Level1(l1));
            }
        }

        if self.preserve_l2 {
            for l2 in &metadata.l2 {
                blocks.push(ExtensionBlock::Level2(*l2));
            }
        }

        if self.preserve_l5 {
            if let Some(l5) = metadata.l5 {
                blocks.push(ExtensionBlock::Level5(l5));
            }
        }

        if self.preserve_l6 {
            if let Some(l6) = metadata.l6 {
                blocks.push(ExtensionBlock::Level6(l6));
            }
        }

        if self.preserve_l8 {
            for l8 in &metadata.l8 {
                blocks.push(ExtensionBlock::Level8(l8.clone()));
            }
        }

        if self.preserve_l9 {
            if let Some(l9) = metadata.l9 {
                blocks.push(ExtensionBlock::Level9(l9));
            }
        }

        if self.preserve_l11 {
            if let Some(l11) = metadata.l11 {
                blocks.push(ExtensionBlock::Level11(l11));
            }
        }

        blocks
    }
}

impl Default for MetadataPreserver {
    fn default() -> Self {
        MetadataPreserver::all()
    }
}

/// Batch converter for processing multiple RPUs.
#[derive(Debug)]
pub struct BatchConverter {
    /// Profile converter.
    converter: ProfileConverter,
    /// Statistics.
    stats: ConversionStats,
}

impl BatchConverter {
    /// Create a new batch converter.
    pub fn new(converter: ProfileConverter) -> Self {
        BatchConverter {
            converter,
            stats: ConversionStats::default(),
        }
    }

    /// Convert a batch of RPUs.
    pub fn convert_batch(&mut self, rpus: &[Rpu]) -> Result<Vec<Rpu>> {
        let mut results = Vec::with_capacity(rpus.len());

        for rpu in rpus {
            match self.converter.convert_rpu(rpu) {
                Ok(converted) => {
                    results.push(converted);
                    self.stats.successful += 1;
                }
                Err(e) => {
                    self.stats.failed += 1;
                    self.stats.errors.push(e.to_string());
                    // Continue with other RPUs
                }
            }
        }

        self.stats.total = rpus.len();

        Ok(results)
    }

    /// Get conversion statistics.
    pub fn stats(&self) -> &ConversionStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = ConversionStats::default();
    }
}

/// Conversion statistics.
#[derive(Debug, Clone, Default)]
pub struct ConversionStats {
    /// Total RPUs processed.
    pub total: usize,
    /// Successfully converted.
    pub successful: usize,
    /// Failed conversions.
    pub failed: usize,
    /// Error messages.
    pub errors: Vec<String>,
}

impl ConversionStats {
    /// Get success rate as percentage.
    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            100.0
        } else {
            (self.successful as f64 / self.total as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::L5Metadata;
    use crate::rpu::create_rpu_from_l1;

    #[test]
    fn test_profile_converter_creation() {
        // Valid conversion paths
        assert!(ProfileConverter::new(
            DolbyVisionProfile::Profile7,
            DolbyVisionProfile::Profile8
        )
        .is_ok());

        assert!(ProfileConverter::new(
            DolbyVisionProfile::Profile5,
            DolbyVisionProfile::Profile8
        )
        .is_ok());

        // Invalid conversion path
        assert!(ProfileConverter::new(
            DolbyVisionProfile::Profile8,
            DolbyVisionProfile::Profile7
        )
        .is_err());
    }

    #[test]
    fn test_convert_7_to_8() {
        let l1 = L1Metadata::new(0, 3079, 1500).unwrap();
        let mut rpu = create_rpu_from_l1(l1, DolbyVisionProfile::Profile7);
        rpu.header.el_spatial_resampling_filter_flag = true;
        rpu.header.disable_residual_flag = Some(false);

        let converter = ProfileConverter::profile7_to_8();
        let result = converter.convert_rpu(&rpu).unwrap();

        assert_eq!(result.profile(), Some(DolbyVisionProfile::Profile8));
        assert!(!result.has_enhancement_layer());
        assert!(result.metadata.l1.is_some());
    }

    #[test]
    fn test_convert_5_to_8() {
        let l1 = L1Metadata::new(0, 2048, 1024).unwrap();
        let rpu = create_rpu_from_l1(l1, DolbyVisionProfile::Profile5);

        let converter = ProfileConverter::profile5_to_8_1();
        let result = converter.convert_rpu(&rpu).unwrap();

        assert_eq!(result.profile(), Some(DolbyVisionProfile::Profile8));
        assert!(result.metadata.l1.is_some());
    }

    #[test]
    fn test_dual_to_single_converter() {
        let l1 = L1Metadata::new(0, 3500, 2000).unwrap();
        let mut rpu = create_rpu_from_l1(l1, DolbyVisionProfile::Profile7);
        rpu.header.el_spatial_resampling_filter_flag = true;

        let converter = DualToSingleConverter::new();
        let result = converter.convert(&rpu).unwrap();

        assert!(!result.has_enhancement_layer());
    }

    #[test]
    fn test_metadata_preserver() {
        let mut source = DolbyVisionMetadata::new();
        source.l1 = Some(L1Metadata::default());
        source.l5 = Some(L5Metadata::new(10, 10, 5, 5));
        source.l6 = Some(L6Metadata::new_unchecked(1000, 400));

        let preserver = MetadataPreserver::essential();
        let mut target = DolbyVisionMetadata::new();

        preserver.copy_metadata(&source, &mut target);

        assert!(target.l1.is_some());
        assert!(target.l5.is_none()); // Not essential
        assert!(target.l6.is_some());
    }

    #[test]
    fn test_batch_converter() {
        let rpus: Vec<Rpu> = (0..5)
            .map(|i| {
                let l1 = L1Metadata::new(0, 3000 + i * 100, 1500).unwrap();
                create_rpu_from_l1(l1, DolbyVisionProfile::Profile7)
            })
            .collect();

        let converter = ProfileConverter::profile7_to_8();
        let mut batch = BatchConverter::new(converter);

        let results = batch.convert_batch(&rpus).unwrap();

        assert_eq!(results.len(), 5);
        assert_eq!(batch.stats().successful, 5);
        assert_eq!(batch.stats().success_rate(), 100.0);
    }

    #[test]
    fn test_conversion_options() {
        let options = ConversionOptions {
            generate_default_l1: true,
            generate_l2_trim: true,
            remap_sdr_to_hdr: false,
            ..Default::default()
        };

        let converter = ProfileConverter::new(
            DolbyVisionProfile::Profile5,
            DolbyVisionProfile::Profile8,
        )
        .unwrap()
        .with_options(options);

        assert_eq!(converter.target_profile(), DolbyVisionProfile::Profile8);
    }
}
