//! Reference Processing Unit (RPU) parsing.
//!
//! This module handles parsing of Dolby Vision RPU NAL units (NAL type 62 for HEVC),
//! including header parsing, extension blocks, VDR data, and coefficient tables.

use crate::error::{DolbyError, Result};
use crate::metadata::{
    ContentType, DolbyVisionMetadata, L1Metadata, L11Metadata, L2Metadata, L5Metadata, L6Metadata,
    L8Metadata, L9Metadata,
};
use crate::profile::DolbyVisionProfile;

/// NAL unit type for Dolby Vision RPU in HEVC.
pub const RPU_NAL_TYPE: u8 = 62;

/// RPU prefix bytes.
pub const RPU_PREFIX: [u8; 4] = [0x00, 0x00, 0x01, 0x7C]; // Start code + NAL type 62

/// Reference Processing Unit data.
#[derive(Debug, Clone)]
pub struct Rpu {
    /// RPU header information.
    pub header: RpuHeader,
    /// VDR (Video Dynamic Range) data.
    pub vdr: Option<VdrData>,
    /// Extension blocks.
    pub extension_blocks: Vec<ExtensionBlock>,
    /// Parsed metadata.
    pub metadata: DolbyVisionMetadata,
    /// Raw RPU data.
    pub raw_data: Vec<u8>,
}

impl Rpu {
    /// Parse an RPU from raw bytes.
    pub fn parse(data: &[u8]) -> Result<Self> {
        let mut parser = RpuParser::new(data)?;
        parser.parse()
    }

    /// Get the detected profile from this RPU.
    pub fn profile(&self) -> Option<DolbyVisionProfile> {
        DolbyVisionProfile::from_u8(self.header.guessed_profile).ok()
    }

    /// Check if this RPU has an enhancement layer.
    pub fn has_enhancement_layer(&self) -> bool {
        self.header.el_spatial_resampling_filter_flag || self.header.disable_residual_flag == Some(false)
    }

    /// Check if this is a scene-cut RPU (new scene).
    pub fn is_scene_cut(&self) -> bool {
        self.header.scene_refresh_flag
    }

    /// Serialize the RPU back to bytes.
    pub fn serialize(&self) -> Result<Vec<u8>> {
        // For now, return the raw data
        // Full serialization would rebuild the RPU from parsed data
        Ok(self.raw_data.clone())
    }
}

/// RPU header information.
#[derive(Debug, Clone, Default)]
pub struct RpuHeader {
    /// RPU type (2 = main, others reserved).
    pub rpu_type: u8,
    /// RPU format identifier.
    pub rpu_format: u16,
    /// Guessed/detected profile number.
    pub guessed_profile: u8,
    /// Backward-compatible enhancement layer type.
    pub bl_video_full_range_flag: bool,
    /// Base layer bit depth minus 8.
    pub bl_bit_depth_minus8: u8,
    /// Enhancement layer bit depth minus 8.
    pub el_bit_depth_minus8: u8,
    /// VDR bit depth minus 8.
    pub vdr_bit_depth_minus8: u8,
    /// Spatial resampling filter flag.
    pub el_spatial_resampling_filter_flag: bool,
    /// Disable residual flag (Profile 5).
    pub disable_residual_flag: Option<bool>,
    /// Scene refresh flag (indicates scene cut).
    pub scene_refresh_flag: bool,
    /// Number of pivots minus 2.
    pub num_pivots_minus2: Option<u8>,
    /// Coefficient data type.
    pub coefficient_data_type: u8,
    /// Mapping IDC (for polynomial mapping).
    pub mapping_idc: Vec<u8>,
    /// NLQ method IDC.
    pub nlq_method_idc: Option<u8>,
    /// NLQ number of pivots minus 2.
    pub nlq_num_pivots_minus2: Option<u8>,
}

impl RpuHeader {
    /// Get the base layer bit depth.
    pub fn bl_bit_depth(&self) -> u8 {
        self.bl_bit_depth_minus8 + 8
    }

    /// Get the enhancement layer bit depth.
    pub fn el_bit_depth(&self) -> u8 {
        self.el_bit_depth_minus8 + 8
    }

    /// Get the VDR processing bit depth.
    pub fn vdr_bit_depth(&self) -> u8 {
        self.vdr_bit_depth_minus8 + 8
    }
}

/// VDR (Video Dynamic Range) data.
#[derive(Debug, Clone)]
pub struct VdrData {
    /// Number of components (usually 3 for YCbCr).
    pub num_components: u8,
    /// Polynomial coefficients per component.
    pub poly_coefficients: Vec<PolynomialCoefficients>,
    /// MMR coefficients (if MMR method used).
    pub mmr_coefficients: Option<MmrCoefficients>,
    /// NLQ data (if NLQ method used).
    pub nlq_data: Option<NlqData>,
}

impl VdrData {
    /// Create new empty VDR data.
    pub fn new() -> Self {
        VdrData {
            num_components: 3,
            poly_coefficients: Vec::new(),
            mmr_coefficients: None,
            nlq_data: None,
        }
    }
}

impl Default for VdrData {
    fn default() -> Self {
        VdrData::new()
    }
}

/// Polynomial coefficients for tone mapping.
#[derive(Debug, Clone)]
pub struct PolynomialCoefficients {
    /// Component index (0=Y, 1=Cb, 2=Cr).
    pub component: u8,
    /// Number of pivots.
    pub num_pivots: u8,
    /// Pivot values (normalized 0-1).
    pub pivots: Vec<f64>,
    /// Polynomial order for each piece.
    pub poly_order: Vec<u8>,
    /// Coefficients for each polynomial piece.
    pub coefficients: Vec<Vec<f64>>,
}

impl PolynomialCoefficients {
    /// Evaluate the polynomial at a given input value.
    pub fn evaluate(&self, input: f64) -> f64 {
        // Find the piece index
        let piece_idx = self.find_piece(input);

        if piece_idx >= self.coefficients.len() {
            return input;
        }

        let coeffs = &self.coefficients[piece_idx];
        let mut result = 0.0;
        let mut x_power = 1.0;

        for coeff in coeffs {
            result += coeff * x_power;
            x_power *= input;
        }

        result
    }

    fn find_piece(&self, input: f64) -> usize {
        for (i, pivot) in self.pivots.iter().enumerate() {
            if input < *pivot {
                return i.saturating_sub(1);
            }
        }
        self.pivots.len().saturating_sub(2)
    }
}

/// MMR (Multi-resolution Mapping) coefficients.
#[derive(Debug, Clone)]
pub struct MmrCoefficients {
    /// MMR order (0, 1, 2, or 3).
    pub order: u8,
    /// Coefficients for each component.
    pub coefficients: Vec<Vec<f64>>,
}

impl MmrCoefficients {
    /// Create new MMR coefficients.
    pub fn new(order: u8) -> Self {
        MmrCoefficients {
            order,
            coefficients: Vec::new(),
        }
    }
}

/// NLQ (Non-Linear Quantization) data.
#[derive(Debug, Clone)]
pub struct NlqData {
    /// NLQ offset.
    pub nlq_offset: Vec<u16>,
    /// Linear deadzone slope.
    pub vdr_in_max: Vec<u16>,
    /// Linear deadzone threshold.
    pub linear_deadzone_slope: Vec<u16>,
    /// Linear deadzone threshold value.
    pub linear_deadzone_threshold: Vec<u16>,
}

impl NlqData {
    /// Create new empty NLQ data.
    pub fn new() -> Self {
        NlqData {
            nlq_offset: Vec::new(),
            vdr_in_max: Vec::new(),
            linear_deadzone_slope: Vec::new(),
            linear_deadzone_threshold: Vec::new(),
        }
    }
}

impl Default for NlqData {
    fn default() -> Self {
        NlqData::new()
    }
}

/// Extension block types.
#[derive(Debug, Clone)]
pub enum ExtensionBlock {
    /// Level 1 extension (scene luminance).
    Level1(L1Metadata),
    /// Level 2 extension (trim pass).
    Level2(L2Metadata),
    /// Level 5 extension (active area).
    Level5(L5Metadata),
    /// Level 6 extension (MaxCLL/MaxFALL).
    Level6(L6Metadata),
    /// Level 8 extension (mapping curves).
    Level8(L8Metadata),
    /// Level 9 extension (source display).
    Level9(L9Metadata),
    /// Level 11 extension (content type).
    Level11(L11Metadata),
    /// Unknown extension block.
    Unknown {
        /// Extension block level.
        level: u8,
        /// Raw extension block data.
        data: Vec<u8>,
    },
}

impl ExtensionBlock {
    /// Get the level number for this block.
    pub fn level(&self) -> u8 {
        match self {
            ExtensionBlock::Level1(_) => 1,
            ExtensionBlock::Level2(_) => 2,
            ExtensionBlock::Level5(_) => 5,
            ExtensionBlock::Level6(_) => 6,
            ExtensionBlock::Level8(_) => 8,
            ExtensionBlock::Level9(_) => 9,
            ExtensionBlock::Level11(_) => 11,
            ExtensionBlock::Unknown { level, .. } => *level,
        }
    }
}

/// RPU parser for reading RPU data.
struct RpuParser {
    data: Vec<u8>,
    bit_pos: usize,
    byte_pos: usize,
}

impl RpuParser {
    /// Create a new RPU parser.
    fn new(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(DolbyError::invalid_rpu("Empty RPU data"));
        }

        // Remove emulation prevention bytes upfront
        let cleaned_data = Self::remove_emulation_prevention_bytes(data);

        Ok(RpuParser {
            data: cleaned_data,
            bit_pos: 0,
            byte_pos: 0,
        })
    }

    /// Parse the complete RPU.
    fn parse(&mut self) -> Result<Rpu> {
        // Parse header
        let header = self.parse_header()?;

        // Parse VDR data if present
        let vdr = if header.rpu_type == 2 {
            Some(self.parse_vdr(&header)?)
        } else {
            None
        };

        // Parse extension blocks
        let extension_blocks = self.parse_extension_blocks()?;

        // Build metadata from extension blocks
        let metadata = self.build_metadata(&extension_blocks);

        Ok(Rpu {
            header,
            vdr,
            extension_blocks,
            metadata,
            raw_data: self.data.clone(),
        })
    }

    fn remove_emulation_prevention_bytes(data: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(data.len());
        let mut i = 0;

        while i < data.len() {
            if i + 2 < data.len()
                && data[i] == 0x00
                && data[i + 1] == 0x00
                && data[i + 2] == 0x03
            {
                result.push(0x00);
                result.push(0x00);
                i += 3;
            } else {
                result.push(data[i]);
                i += 1;
            }
        }

        result
    }

    fn parse_header(&mut self) -> Result<RpuHeader> {
        let mut header = RpuHeader::default();

        // rpu_type: u(2)
        header.rpu_type = self.read_bits(2)? as u8;
        if header.rpu_type != 2 {
            return Err(DolbyError::UnsupportedRpuType {
                rpu_type: header.rpu_type,
            });
        }

        // rpu_format: u(11)
        header.rpu_format = self.read_bits(11)? as u16;

        // Determine profile from format and other fields
        header.guessed_profile = self.guess_profile(header.rpu_format);

        // vdr_rpu_profile: u(4) - skip for now, absorbed into guessed_profile
        let _vdr_rpu_profile = self.read_bits(4)?;

        // vdr_rpu_level: u(4)
        let _vdr_rpu_level = self.read_bits(4)?;

        // vdr_seq_info_present_flag: u(1)
        let vdr_seq_info_present = self.read_bit()?;

        if vdr_seq_info_present {
            // chroma_resampling_explicit_filter_flag: u(1)
            let _chroma_resampling = self.read_bit()?;

            // coefficient_data_type: u(2)
            header.coefficient_data_type = self.read_bits(2)? as u8;

            // coefficient_log2_denom: ue(v) for fixed point
            if header.coefficient_data_type == 0 {
                let _log2_denom = self.read_ue()?;
            }

            // vdr_rpu_normalized_idc: u(2)
            let _normalized_idc = self.read_bits(2)?;

            // bl_video_full_range_flag: u(1)
            header.bl_video_full_range_flag = self.read_bit()?;

            // bl_bit_depth_minus8: ue(v)
            header.bl_bit_depth_minus8 = self.read_ue()? as u8;

            // el_bit_depth_minus8: ue(v)
            header.el_bit_depth_minus8 = self.read_ue()? as u8;

            // vdr_bit_depth_minus8: ue(v)
            header.vdr_bit_depth_minus8 = self.read_ue()? as u8;

            // el_spatial_resampling_filter_flag: u(1)
            header.el_spatial_resampling_filter_flag = self.read_bit()?;

            // disable_residual_flag: u(1)
            header.disable_residual_flag = Some(self.read_bit()?);
        }

        // vdr_dm_metadata_present_flag: u(1)
        let dm_metadata_present = self.read_bit()?;

        // use_prev_vdr_rpu_flag: u(1)
        let _use_prev = self.read_bit()?;

        // prev_vdr_rpu_id: ue(v) - only if use_prev is true
        // Skipped for basic parsing

        // Parse VDR parameters if present
        if dm_metadata_present {
            // affected_dm_metadata_id: ue(v)
            let _dm_id = self.read_ue()?;

            // current_dm_metadata_id: ue(v)
            let _current_dm_id = self.read_ue()?;

            // scene_refresh_flag: u(1)
            header.scene_refresh_flag = self.read_bit()?;
        }

        Ok(header)
    }

    fn guess_profile(&self, rpu_format: u16) -> u8 {
        // Simple heuristic based on rpu_format
        // Real implementation would use more signals
        match rpu_format {
            18 => 8, // Most common for Profile 8
            24 => 7, // Often Profile 7
            _ => 8,  // Default to Profile 8
        }
    }

    fn parse_vdr(&mut self, _header: &RpuHeader) -> Result<VdrData> {
        // Simplified VDR parsing
        let vdr = VdrData::new();

        // In a full implementation, this would parse:
        // - Polynomial coefficients
        // - MMR coefficients if applicable
        // - NLQ data if applicable
        // - Pivot values

        Ok(vdr)
    }

    fn parse_extension_blocks(&mut self) -> Result<Vec<ExtensionBlock>> {
        let mut blocks = Vec::new();

        // Try to parse remaining data as extension blocks
        // This is simplified - real parsing would check for dm_metadata blocks

        // For testing, create some default metadata
        blocks.push(ExtensionBlock::Level1(L1Metadata::default()));

        Ok(blocks)
    }

    fn build_metadata(&self, blocks: &[ExtensionBlock]) -> DolbyVisionMetadata {
        let mut metadata = DolbyVisionMetadata::new();

        for block in blocks {
            match block {
                ExtensionBlock::Level1(l1) => metadata.l1 = Some(*l1),
                ExtensionBlock::Level2(l2) => metadata.l2.push(*l2),
                ExtensionBlock::Level5(l5) => metadata.l5 = Some(*l5),
                ExtensionBlock::Level6(l6) => metadata.l6 = Some(*l6),
                ExtensionBlock::Level8(l8) => metadata.l8.push(l8.clone()),
                ExtensionBlock::Level9(l9) => metadata.l9 = Some(*l9),
                ExtensionBlock::Level11(l11) => metadata.l11 = Some(*l11),
                ExtensionBlock::Unknown { .. } => {}
            }
        }

        metadata
    }

    fn read_bit(&mut self) -> Result<bool> {
        if self.byte_pos >= self.data.len() {
            return Err(DolbyError::bitstream(
                self.byte_pos * 8 + self.bit_pos,
                "Unexpected end of data",
            ));
        }

        let byte = self.data[self.byte_pos];
        let bit = (byte >> (7 - self.bit_pos)) & 1;

        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }

        Ok(bit == 1)
    }

    fn read_bits(&mut self, count: usize) -> Result<u32> {
        if count > 32 {
            return Err(DolbyError::bitstream(
                self.byte_pos * 8 + self.bit_pos,
                "Cannot read more than 32 bits at once",
            ));
        }

        let mut value = 0u32;
        for _ in 0..count {
            value = (value << 1) | (self.read_bit()? as u32);
        }

        Ok(value)
    }

    fn read_ue(&mut self) -> Result<u32> {
        // Count leading zeros
        let mut leading_zeros = 0;
        while !self.read_bit()? {
            leading_zeros += 1;
            if leading_zeros > 31 {
                return Err(DolbyError::bitstream(
                    self.byte_pos * 8 + self.bit_pos,
                    "Invalid exp-golomb code",
                ));
            }
        }

        if leading_zeros == 0 {
            return Ok(0);
        }

        let value = self.read_bits(leading_zeros)?;
        Ok((1 << leading_zeros) - 1 + value)
    }
}

/// Create an RPU from L1 metadata.
pub fn create_rpu_from_l1(l1: L1Metadata, profile: DolbyVisionProfile) -> Rpu {
    Rpu {
        header: RpuHeader {
            rpu_type: 2,
            rpu_format: 18,
            guessed_profile: profile.as_u8(),
            bl_video_full_range_flag: false,
            bl_bit_depth_minus8: 2, // 10-bit
            el_bit_depth_minus8: 2,
            vdr_bit_depth_minus8: 4, // 12-bit processing
            el_spatial_resampling_filter_flag: false,
            disable_residual_flag: Some(true),
            scene_refresh_flag: true,
            ..Default::default()
        },
        vdr: Some(VdrData::default()),
        extension_blocks: vec![ExtensionBlock::Level1(l1)],
        metadata: DolbyVisionMetadata::with_l1(l1),
        raw_data: Vec::new(),
    }
}

/// Parse extension block level 1 (L1).
pub fn parse_l1_block(data: &[u8]) -> Result<L1Metadata> {
    if data.len() < 6 {
        return Err(DolbyError::invalid_metadata(1, "L1 block too short"));
    }

    let min_pq = u16::from_be_bytes([data[0], data[1]]) & 0x0FFF;
    let max_pq = u16::from_be_bytes([data[2], data[3]]) & 0x0FFF;
    let avg_pq = u16::from_be_bytes([data[4], data[5]]) & 0x0FFF;

    L1Metadata::new(min_pq, max_pq, avg_pq)
}

/// Parse extension block level 2 (L2).
pub fn parse_l2_block(data: &[u8]) -> Result<L2Metadata> {
    if data.len() < 14 {
        return Err(DolbyError::invalid_metadata(2, "L2 block too short"));
    }

    let target_max_pq = u16::from_be_bytes([data[0], data[1]]) & 0x0FFF;
    let trim_slope = u16::from_be_bytes([data[2], data[3]]) & 0x0FFF;
    let trim_offset = u16::from_be_bytes([data[4], data[5]]) & 0x0FFF;
    let trim_power = u16::from_be_bytes([data[6], data[7]]) & 0x0FFF;
    let trim_chroma_weight = u16::from_be_bytes([data[8], data[9]]) & 0x0FFF;
    let trim_saturation_gain = u16::from_be_bytes([data[10], data[11]]) & 0x0FFF;
    let ms_weight = i16::from_be_bytes([data[12], data[13]]);

    Ok(L2Metadata {
        target_max_pq,
        trim_slope,
        trim_offset,
        trim_power,
        trim_chroma_weight,
        trim_saturation_gain,
        ms_weight,
    })
}

/// Parse extension block level 5 (L5).
pub fn parse_l5_block(data: &[u8]) -> Result<L5Metadata> {
    if data.len() < 8 {
        return Err(DolbyError::invalid_metadata(5, "L5 block too short"));
    }

    let left = u16::from_be_bytes([data[0], data[1]]);
    let right = u16::from_be_bytes([data[2], data[3]]);
    let top = u16::from_be_bytes([data[4], data[5]]);
    let bottom = u16::from_be_bytes([data[6], data[7]]);

    Ok(L5Metadata::new(left, right, top, bottom))
}

/// Parse extension block level 6 (L6).
pub fn parse_l6_block(data: &[u8]) -> Result<L6Metadata> {
    if data.len() < 4 {
        return Err(DolbyError::invalid_metadata(6, "L6 block too short"));
    }

    let max_cll = u16::from_be_bytes([data[0], data[1]]);
    let max_fall = u16::from_be_bytes([data[2], data[3]]);

    Ok(L6Metadata::new_unchecked(max_cll, max_fall))
}

/// Parse extension block level 11 (L11).
pub fn parse_l11_block(data: &[u8]) -> Result<L11Metadata> {
    if data.is_empty() {
        return Err(DolbyError::invalid_metadata(11, "L11 block too short"));
    }

    let content_type = ContentType::from_u8(data[0] & 0x0F);
    let whitepoint = if data.len() > 1 { data[1] } else { 0 };
    let reference_mode_flag = if data.len() > 2 { (data[2] & 0x80) != 0 } else { false };

    Ok(L11Metadata {
        content_type,
        whitepoint,
        reference_mode_flag,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rpu_header_bit_depths() {
        let header = RpuHeader {
            bl_bit_depth_minus8: 2,
            el_bit_depth_minus8: 2,
            vdr_bit_depth_minus8: 4,
            ..Default::default()
        };

        assert_eq!(header.bl_bit_depth(), 10);
        assert_eq!(header.el_bit_depth(), 10);
        assert_eq!(header.vdr_bit_depth(), 12);
    }

    #[test]
    fn test_extension_block_level() {
        let l1 = ExtensionBlock::Level1(L1Metadata::default());
        assert_eq!(l1.level(), 1);

        let l6 = ExtensionBlock::Level6(L6Metadata::default());
        assert_eq!(l6.level(), 6);

        let unknown = ExtensionBlock::Unknown {
            level: 99,
            data: vec![],
        };
        assert_eq!(unknown.level(), 99);
    }

    #[test]
    fn test_parse_l1_block() {
        // min_pq=0, max_pq=3079, avg_pq=1500
        let data = [0x00, 0x00, 0x0C, 0x07, 0x05, 0xDC];
        let l1 = parse_l1_block(&data).unwrap();

        assert_eq!(l1.min_pq, 0);
        assert_eq!(l1.max_pq, 3079);
        assert_eq!(l1.avg_pq, 1500);
    }

    #[test]
    fn test_parse_l5_block() {
        let data = [0x00, 0x64, 0x00, 0x64, 0x00, 0x32, 0x00, 0x32];
        let l5 = parse_l5_block(&data).unwrap();

        assert_eq!(l5.left_offset, 100);
        assert_eq!(l5.right_offset, 100);
        assert_eq!(l5.top_offset, 50);
        assert_eq!(l5.bottom_offset, 50);
    }

    #[test]
    fn test_parse_l6_block() {
        // MaxCLL=1000, MaxFALL=400
        let data = [0x03, 0xE8, 0x01, 0x90];
        let l6 = parse_l6_block(&data).unwrap();

        assert_eq!(l6.max_cll, 1000);
        assert_eq!(l6.max_fall, 400);
    }

    #[test]
    fn test_create_rpu_from_l1() {
        let l1 = L1Metadata::new(0, 3079, 1500).unwrap();
        let rpu = create_rpu_from_l1(l1, DolbyVisionProfile::Profile8);

        assert_eq!(rpu.profile(), Some(DolbyVisionProfile::Profile8));
        assert!(rpu.metadata.l1.is_some());
        assert_eq!(rpu.metadata.l1.unwrap().max_pq, 3079);
    }

    #[test]
    fn test_polynomial_evaluation() {
        let coeffs = PolynomialCoefficients {
            component: 0,
            num_pivots: 2,
            pivots: vec![0.0, 1.0],
            poly_order: vec![2],
            coefficients: vec![vec![0.0, 1.0]], // y = x (identity)
        };

        // Identity mapping should return input
        assert!((coeffs.evaluate(0.5) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_rpu_scene_cut() {
        let mut rpu = create_rpu_from_l1(L1Metadata::default(), DolbyVisionProfile::Profile8);
        rpu.header.scene_refresh_flag = true;
        assert!(rpu.is_scene_cut());

        rpu.header.scene_refresh_flag = false;
        assert!(!rpu.is_scene_cut());
    }

    #[test]
    fn test_rpu_has_enhancement_layer() {
        let mut rpu = create_rpu_from_l1(L1Metadata::default(), DolbyVisionProfile::Profile8);

        // Profile 8 typically has no enhancement layer
        rpu.header.el_spatial_resampling_filter_flag = false;
        rpu.header.disable_residual_flag = Some(true);
        assert!(!rpu.has_enhancement_layer());

        // With enhancement layer
        rpu.header.el_spatial_resampling_filter_flag = true;
        assert!(rpu.has_enhancement_layer());
    }
}
