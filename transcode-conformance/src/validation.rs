//! Bitstream validation for H.264 conformance testing
//!
//! Provides validation of H.264 bitstream structure and syntax elements.

use crate::{H264Profile, Result};
use std::collections::HashMap;

/// Bitstream validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the bitstream is valid
    pub is_valid: bool,
    /// List of errors found
    pub errors: Vec<ValidationError>,
    /// List of warnings
    pub warnings: Vec<ValidationWarning>,
    /// NAL unit statistics
    pub nal_stats: NalStatistics,
    /// SPS information
    pub sps_info: Option<SpsInfo>,
    /// PPS information
    pub pps_info: Vec<PpsInfo>,
}

impl ValidationResult {
    /// Create a new empty result
    pub fn new() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            nal_stats: NalStatistics::default(),
            sps_info: None,
            pps_info: Vec::new(),
        }
    }

    /// Add an error
    pub fn add_error(&mut self, error: ValidationError) {
        self.errors.push(error);
        self.is_valid = false;
    }

    /// Add a warning
    pub fn add_warning(&mut self, warning: ValidationWarning) {
        self.warnings.push(warning);
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error code
    pub code: ValidationErrorCode,
    /// Error message
    pub message: String,
    /// Byte offset where error occurred
    pub offset: Option<usize>,
    /// NAL unit index where error occurred
    pub nal_index: Option<usize>,
}

/// Validation error codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationErrorCode {
    /// Invalid start code
    InvalidStartCode,
    /// Invalid NAL unit type
    InvalidNalType,
    /// SPS parsing error
    SpsParseError,
    /// PPS parsing error
    PpsParseError,
    /// Slice header error
    SliceHeaderError,
    /// Reference picture error
    ReferencePictureError,
    /// Entropy coding error
    EntropyCodingError,
    /// Transform error
    TransformError,
    /// Profile constraint violation
    ProfileConstraintViolation,
    /// Level constraint violation
    LevelConstraintViolation,
    /// Invalid syntax element
    InvalidSyntax,
    /// Bitstream corruption
    BitstreamCorruption,
}

/// Validation warning
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// Warning code
    pub code: ValidationWarningCode,
    /// Warning message
    pub message: String,
    /// Byte offset
    pub offset: Option<usize>,
}

/// Validation warning codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationWarningCode {
    /// Unusual but valid parameter
    UnusualParameter,
    /// Deprecated feature used
    DeprecatedFeature,
    /// Performance warning
    PerformanceWarning,
    /// Compatibility warning
    CompatibilityWarning,
}

/// NAL unit statistics
#[derive(Debug, Clone, Default)]
pub struct NalStatistics {
    /// Total NAL units
    pub total_count: usize,
    /// Count by NAL unit type
    pub by_type: HashMap<u8, usize>,
    /// SPS count
    pub sps_count: usize,
    /// PPS count
    pub pps_count: usize,
    /// IDR slice count
    pub idr_count: usize,
    /// Non-IDR slice count
    pub non_idr_count: usize,
    /// SEI message count
    pub sei_count: usize,
    /// AUD count
    pub aud_count: usize,
}

/// SPS information extracted during validation
#[derive(Debug, Clone)]
pub struct SpsInfo {
    /// Profile IDC
    pub profile_idc: u8,
    /// Constraint flags
    pub constraint_flags: u8,
    /// Level IDC
    pub level_idc: u8,
    /// SPS ID
    pub sps_id: u8,
    /// Chroma format (0=mono, 1=420, 2=422, 3=444)
    pub chroma_format_idc: u8,
    /// Bit depth luma minus 8
    pub bit_depth_luma: u8,
    /// Bit depth chroma minus 8
    pub bit_depth_chroma: u8,
    /// Max frame number
    pub log2_max_frame_num: u8,
    /// POC type
    pub pic_order_cnt_type: u8,
    /// Max POC LSB
    pub log2_max_pic_order_cnt_lsb: Option<u8>,
    /// Number of reference frames
    pub num_ref_frames: u8,
    /// Picture width in MBs
    pub pic_width_in_mbs: u16,
    /// Picture height in MB units
    pub pic_height_in_map_units: u16,
    /// Frame/field coding
    pub frame_mbs_only: bool,
    /// Computed width
    pub width: u32,
    /// Computed height
    pub height: u32,
}

/// PPS information extracted during validation
#[derive(Debug, Clone)]
pub struct PpsInfo {
    /// PPS ID
    pub pps_id: u8,
    /// Referenced SPS ID
    pub sps_id: u8,
    /// Entropy coding mode (0=CAVLC, 1=CABAC)
    pub entropy_coding_mode: u8,
    /// Bottom field POC flag
    pub bottom_field_pic_order_in_frame_present: bool,
    /// Number of slice groups
    pub num_slice_groups: u8,
    /// Number of reference indices L0
    pub num_ref_idx_l0_default: u8,
    /// Number of reference indices L1
    pub num_ref_idx_l1_default: u8,
    /// Weighted prediction P
    pub weighted_pred_flag: bool,
    /// Weighted prediction B
    pub weighted_bipred_idc: u8,
    /// Initial QP
    pub pic_init_qp: i8,
    /// Initial QS
    pub pic_init_qs: i8,
    /// Chroma QP offset
    pub chroma_qp_index_offset: i8,
    /// Deblocking filter control
    pub deblocking_filter_control_present: bool,
    /// Constrained intra prediction
    pub constrained_intra_pred: bool,
    /// Redundant picture count
    pub redundant_pic_cnt_present: bool,
    /// 8x8 transform flag (High profile)
    pub transform_8x8_mode: bool,
}

/// Bitstream validator
pub struct BitstreamValidator {
    /// Expected profile (optional)
    expected_profile: Option<H264Profile>,
    /// Strict mode
    strict: bool,
}

impl Default for BitstreamValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl BitstreamValidator {
    /// Create a new validator
    pub fn new() -> Self {
        Self {
            expected_profile: None,
            strict: false,
        }
    }

    /// Set expected profile
    pub fn expect_profile(mut self, profile: H264Profile) -> Self {
        self.expected_profile = Some(profile);
        self
    }

    /// Enable strict mode
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Validate a bitstream
    pub fn validate(&self, data: &[u8]) -> Result<ValidationResult> {
        let mut result = ValidationResult::new();

        if data.is_empty() {
            result.add_error(ValidationError {
                code: ValidationErrorCode::BitstreamCorruption,
                message: "Empty bitstream".to_string(),
                offset: Some(0),
                nal_index: None,
            });
            return Ok(result);
        }

        // Find and validate NAL units
        let nal_units = self.find_nal_units(data)?;

        if nal_units.is_empty() {
            result.add_error(ValidationError {
                code: ValidationErrorCode::InvalidStartCode,
                message: "No valid NAL units found".to_string(),
                offset: Some(0),
                nal_index: None,
            });
            return Ok(result);
        }

        // Process each NAL unit
        for (index, (offset, nal_data)) in nal_units.iter().enumerate() {
            self.validate_nal_unit(&mut result, nal_data, *offset, index)?;
        }

        // Validate profile constraints if expected
        if let Some(expected) = &self.expected_profile {
            self.validate_profile_constraints(&mut result, expected)?;
        }

        Ok(result)
    }

    /// Find NAL units in bitstream (Annex B format)
    fn find_nal_units<'a>(&self, data: &'a [u8]) -> Result<Vec<(usize, &'a [u8])>> {
        let mut units = Vec::new();
        let mut i = 0;

        while i < data.len() {
            // Look for start code (0x000001 or 0x00000001)
            if i + 3 <= data.len() && data[i] == 0 && data[i + 1] == 0 {
                let (_start_code_len, nal_start) = if data[i + 2] == 1 {
                    (3, i + 3)
                } else if i + 4 <= data.len() && data[i + 2] == 0 && data[i + 3] == 1 {
                    (4, i + 4)
                } else {
                    i += 1;
                    continue;
                };

                // Find end of NAL unit (next start code or end of data)
                let mut nal_end = nal_start;
                while nal_end < data.len() {
                    if nal_end + 3 <= data.len()
                        && data[nal_end] == 0
                        && data[nal_end + 1] == 0
                        && (data[nal_end + 2] == 1
                            || (nal_end + 4 <= data.len()
                                && data[nal_end + 2] == 0
                                && data[nal_end + 3] == 1))
                    {
                        break;
                    }
                    nal_end += 1;
                }

                if nal_end > nal_start {
                    units.push((i, &data[nal_start..nal_end]));
                }

                i = nal_end;
            } else {
                i += 1;
            }
        }

        Ok(units)
    }

    /// Validate a single NAL unit
    fn validate_nal_unit(
        &self,
        result: &mut ValidationResult,
        nal_data: &[u8],
        offset: usize,
        index: usize,
    ) -> Result<()> {
        if nal_data.is_empty() {
            result.add_error(ValidationError {
                code: ValidationErrorCode::InvalidNalType,
                message: "Empty NAL unit".to_string(),
                offset: Some(offset),
                nal_index: Some(index),
            });
            return Ok(());
        }

        let nal_header = nal_data[0];
        let forbidden_bit = (nal_header >> 7) & 1;
        let _nal_ref_idc = (nal_header >> 5) & 3;
        let nal_unit_type = nal_header & 0x1f;

        // Check forbidden bit
        if forbidden_bit != 0 {
            result.add_error(ValidationError {
                code: ValidationErrorCode::BitstreamCorruption,
                message: "Forbidden bit is set".to_string(),
                offset: Some(offset),
                nal_index: Some(index),
            });
        }

        // Update statistics
        result.nal_stats.total_count += 1;
        *result.nal_stats.by_type.entry(nal_unit_type).or_insert(0) += 1;

        match nal_unit_type {
            1 => result.nal_stats.non_idr_count += 1,  // Non-IDR slice
            5 => result.nal_stats.idr_count += 1,       // IDR slice
            6 => result.nal_stats.sei_count += 1,       // SEI
            7 => {
                // SPS
                result.nal_stats.sps_count += 1;
                self.parse_sps(result, nal_data, offset, index)?;
            }
            8 => {
                // PPS
                result.nal_stats.pps_count += 1;
                self.parse_pps(result, nal_data, offset, index)?;
            }
            9 => result.nal_stats.aud_count += 1, // AUD
            0 | 10..=12 | 14..=18 => {
                // Valid but less common types
            }
            _ => {
                if nal_unit_type > 23 {
                    result.add_warning(ValidationWarning {
                        code: ValidationWarningCode::UnusualParameter,
                        message: format!("Unusual NAL unit type: {}", nal_unit_type),
                        offset: Some(offset),
                    });
                }
            }
        }

        Ok(())
    }

    /// Parse SPS (simplified)
    fn parse_sps(
        &self,
        result: &mut ValidationResult,
        nal_data: &[u8],
        offset: usize,
        index: usize,
    ) -> Result<()> {
        if nal_data.len() < 4 {
            result.add_error(ValidationError {
                code: ValidationErrorCode::SpsParseError,
                message: "SPS too short".to_string(),
                offset: Some(offset),
                nal_index: Some(index),
            });
            return Ok(());
        }

        let profile_idc = nal_data[1];
        let constraint_flags = nal_data[2];
        let level_idc = nal_data[3];

        // Validate profile_idc
        let valid_profiles = [66, 77, 88, 100, 110, 122, 244, 44, 83, 86, 118, 128];
        if !valid_profiles.contains(&profile_idc) {
            result.add_warning(ValidationWarning {
                code: ValidationWarningCode::UnusualParameter,
                message: format!("Unknown profile_idc: {}", profile_idc),
                offset: Some(offset),
            });
        }

        // Basic SPS info (would need full bitstream parsing for complete info)
        result.sps_info = Some(SpsInfo {
            profile_idc,
            constraint_flags,
            level_idc,
            sps_id: 0,
            chroma_format_idc: 1,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            log2_max_frame_num: 4,
            pic_order_cnt_type: 0,
            log2_max_pic_order_cnt_lsb: Some(4),
            num_ref_frames: 1,
            pic_width_in_mbs: 0,
            pic_height_in_map_units: 0,
            frame_mbs_only: true,
            width: 0,
            height: 0,
        });

        Ok(())
    }

    /// Parse PPS (simplified)
    fn parse_pps(
        &self,
        result: &mut ValidationResult,
        nal_data: &[u8],
        offset: usize,
        index: usize,
    ) -> Result<()> {
        if nal_data.len() < 2 {
            result.add_error(ValidationError {
                code: ValidationErrorCode::PpsParseError,
                message: "PPS too short".to_string(),
                offset: Some(offset),
                nal_index: Some(index),
            });
            return Ok(());
        }

        // Basic PPS info (would need Exp-Golomb decoding for complete parsing)
        result.pps_info.push(PpsInfo {
            pps_id: 0,
            sps_id: 0,
            entropy_coding_mode: 0,
            bottom_field_pic_order_in_frame_present: false,
            num_slice_groups: 1,
            num_ref_idx_l0_default: 1,
            num_ref_idx_l1_default: 1,
            weighted_pred_flag: false,
            weighted_bipred_idc: 0,
            pic_init_qp: 26,
            pic_init_qs: 26,
            chroma_qp_index_offset: 0,
            deblocking_filter_control_present: false,
            constrained_intra_pred: false,
            redundant_pic_cnt_present: false,
            transform_8x8_mode: false,
        });

        Ok(())
    }

    /// Validate profile constraints
    fn validate_profile_constraints(
        &self,
        result: &mut ValidationResult,
        expected: &H264Profile,
    ) -> Result<()> {
        if let Some(sps) = &result.sps_info {
            let actual_profile = sps.profile_idc;
            let expected_idc = expected.profile_idc();

            if actual_profile != expected_idc {
                result.add_error(ValidationError {
                    code: ValidationErrorCode::ProfileConstraintViolation,
                    message: format!(
                        "Expected profile {} (idc={}), found profile idc={}",
                        expected.name(),
                        expected_idc,
                        actual_profile
                    ),
                    offset: None,
                    nal_index: None,
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_result() {
        let mut result = ValidationResult::new();
        assert!(result.is_valid);

        result.add_error(ValidationError {
            code: ValidationErrorCode::InvalidSyntax,
            message: "test error".to_string(),
            offset: Some(0),
            nal_index: None,
        });

        assert!(!result.is_valid);
        assert_eq!(result.errors.len(), 1);
    }

    #[test]
    fn test_empty_bitstream() {
        let validator = BitstreamValidator::new();
        let result = validator.validate(&[]).unwrap();
        assert!(!result.is_valid);
    }

    #[test]
    fn test_no_start_code() {
        let validator = BitstreamValidator::new();
        let data = vec![0x67, 0x42, 0x00, 0x0a]; // SPS without start code
        let result = validator.validate(&data).unwrap();
        assert!(!result.is_valid);
    }

    #[test]
    fn test_valid_nal_unit() {
        let validator = BitstreamValidator::new();
        // Start code + SPS NAL unit type + minimal SPS data
        let data = vec![0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x00];
        let result = validator.validate(&data).unwrap();
        assert_eq!(result.nal_stats.sps_count, 1);
    }

    #[test]
    fn test_nal_statistics() {
        let stats = NalStatistics::default();
        assert_eq!(stats.total_count, 0);
        assert_eq!(stats.sps_count, 0);
        assert_eq!(stats.pps_count, 0);
    }
}
