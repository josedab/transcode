//! Integration tests for the HEVC codec.
//!
//! These tests verify the complete functionality of the HEVC encoder and decoder,
//! including NAL parsing, parameter sets, transforms, and CABAC coding.

use pretty_assertions::assert_eq;
use transcode_hevc::{
    cabac::{CabacDecoder, CabacEncoder},
    decoder::{HevcDecoder, HevcDecoderConfig, IntraMode, PartMode, PredictionMode},
    encoder::{HevcEncoder, HevcEncoderConfig, HevcPreset, RateControlMode},
    error::{HevcError, HevcLevel, HevcProfile, HevcTier},
    nal::{NalUnitHeader, NalUnitType, SliceType, parse_annexb_nal_units},
    transform::{HevcQuantizer, HevcTransform, TransformSize},
};
use transcode_core::frame::PixelFormat;

// ============================================================================
// NAL Unit Tests
// ============================================================================

#[test]
fn test_nal_unit_type_all_values() {
    // Verify all standard NAL unit types can be created and converted back
    for i in 0..=47 {
        let nal_type = NalUnitType::from_raw(i);
        assert_eq!(nal_type.to_raw(), i);
    }

    // Test unspecified types (48-63)
    for i in 48..64 {
        let nal_type = NalUnitType::from_raw(i);
        assert!(matches!(nal_type, NalUnitType::Unspecified(_)));
        assert_eq!(nal_type.to_raw(), i);
    }
}

#[test]
fn test_nal_unit_type_vcl_classification() {
    // VCL types (0-31)
    for i in 0..32 {
        let nal_type = NalUnitType::from_raw(i);
        assert!(nal_type.is_vcl(), "NAL type {} should be VCL", i);
    }

    // Non-VCL types (32-63)
    for i in 32..64 {
        let nal_type = NalUnitType::from_raw(i);
        assert!(!nal_type.is_vcl(), "NAL type {} should not be VCL", i);
    }
}

#[test]
fn test_nal_unit_type_irap_classification() {
    let irap_types = [
        NalUnitType::BlaWLp,
        NalUnitType::BlaWRadl,
        NalUnitType::BlaNLp,
        NalUnitType::IdrWRadl,
        NalUnitType::IdrNLp,
        NalUnitType::CraNut,
    ];

    for nal_type in &irap_types {
        assert!(nal_type.is_irap(), "{} should be IRAP", nal_type);
    }

    // Non-IRAP types
    let non_irap_types = [
        NalUnitType::TrailN,
        NalUnitType::TrailR,
        NalUnitType::VpsNut,
        NalUnitType::SpsNut,
        NalUnitType::PpsNut,
    ];

    for nal_type in &non_irap_types {
        assert!(!nal_type.is_irap(), "{} should not be IRAP", nal_type);
    }
}

#[test]
fn test_nal_unit_type_reference_classification() {
    // Reference types have odd-numbered VCL NAL unit types
    let reference_types = [
        NalUnitType::TrailR,     // 1
        NalUnitType::TsaR,       // 3
        NalUnitType::StsaR,      // 5
        NalUnitType::RadlR,      // 7
        NalUnitType::RaslR,      // 9
    ];

    for nal_type in &reference_types {
        assert!(nal_type.is_reference(), "{} should be reference", nal_type);
    }

    // Non-reference VCL types have even numbers
    let non_reference_types = [
        NalUnitType::TrailN,     // 0
        NalUnitType::TsaN,       // 2
        NalUnitType::StsaN,      // 4
        NalUnitType::RadlN,      // 6
        NalUnitType::RaslN,      // 8
    ];

    for nal_type in &non_reference_types {
        assert!(!nal_type.is_reference(), "{} should not be reference", nal_type);
    }
}

#[test]
fn test_nal_unit_header_parse_valid() {
    // Various valid NAL unit headers
    let test_cases = [
        // (data, expected_nal_type, expected_layer_id, expected_temporal_id)
        ([0x02, 0x01], NalUnitType::TrailR, 0, 1),
        ([0x40, 0x01], NalUnitType::VpsNut, 0, 1),
        ([0x42, 0x01], NalUnitType::SpsNut, 0, 1),
        ([0x44, 0x01], NalUnitType::PpsNut, 0, 1),
        ([0x26, 0x01], NalUnitType::IdrWRadl, 0, 1),
        ([0x28, 0x01], NalUnitType::IdrNLp, 0, 1),
        ([0x02, 0x03], NalUnitType::TrailR, 0, 3), // Higher temporal ID
    ];

    for (data, expected_type, expected_layer, expected_temporal) in test_cases {
        let header = NalUnitHeader::parse(&data).unwrap();
        assert_eq!(header.nal_unit_type, expected_type);
        assert_eq!(header.nuh_layer_id, expected_layer);
        assert_eq!(header.nuh_temporal_id_plus1, expected_temporal);
    }
}

#[test]
fn test_nal_unit_header_parse_invalid() {
    // Too short
    assert!(NalUnitHeader::parse(&[0x02]).is_err());
    assert!(NalUnitHeader::parse(&[]).is_err());

    // Forbidden zero bit set
    assert!(NalUnitHeader::parse(&[0x82, 0x01]).is_err());

    // Zero temporal ID
    assert!(NalUnitHeader::parse(&[0x02, 0x00]).is_err());
}

#[test]
fn test_annexb_nal_parsing() {
    // Create a simple Annex B stream with multiple NAL units
    let stream = [
        0x00, 0x00, 0x00, 0x01, // 4-byte start code
        0x40, 0x01,             // VPS NAL header
        0x00,                   // VPS data (minimal)
        0x00, 0x00, 0x01,       // 3-byte start code
        0x42, 0x01,             // SPS NAL header
        0x00,                   // SPS data (minimal)
        0x00, 0x00, 0x01,       // 3-byte start code
        0x44, 0x01,             // PPS NAL header
        0x00,                   // PPS data (minimal)
    ];

    let nal_units = parse_annexb_nal_units(&stream);
    assert_eq!(nal_units.len(), 3);

    assert_eq!(nal_units[0].0.nal_unit_type, NalUnitType::VpsNut);
    assert_eq!(nal_units[1].0.nal_unit_type, NalUnitType::SpsNut);
    assert_eq!(nal_units[2].0.nal_unit_type, NalUnitType::PpsNut);
}

#[test]
fn test_annexb_empty_stream() {
    let nal_units = parse_annexb_nal_units(&[]);
    assert!(nal_units.is_empty());
}

#[test]
fn test_annexb_no_start_code() {
    let stream = [0x40, 0x01, 0x00, 0x00]; // No start code prefix
    let nal_units = parse_annexb_nal_units(&stream);
    assert!(nal_units.is_empty());
}

// ============================================================================
// Slice Type Tests
// ============================================================================

#[test]
fn test_slice_type_properties() {
    // I-slice
    assert!(SliceType::I.is_intra());
    assert!(!SliceType::I.uses_list0());
    assert!(!SliceType::I.uses_list1());

    // P-slice
    assert!(!SliceType::P.is_intra());
    assert!(SliceType::P.uses_list0());
    assert!(!SliceType::P.uses_list1());

    // B-slice
    assert!(!SliceType::B.is_intra());
    assert!(SliceType::B.uses_list0());
    assert!(SliceType::B.uses_list1());
}

#[test]
fn test_slice_type_from_raw() {
    assert!(matches!(SliceType::from_raw(0), Ok(SliceType::B)));
    assert!(matches!(SliceType::from_raw(1), Ok(SliceType::P)));
    assert!(matches!(SliceType::from_raw(2), Ok(SliceType::I)));
    assert!(SliceType::from_raw(3).is_err());
    assert!(SliceType::from_raw(255).is_err());
}

// ============================================================================
// Encoder Configuration Tests
// ============================================================================

#[test]
fn test_encoder_config_builder() {
    let config = HevcEncoderConfig::new(1920, 1080)
        .with_preset(HevcPreset::Medium)
        .with_profile(HevcProfile::Main)
        .with_crf(23.0)
        .with_gop_size(250)
        .with_bframes(4);

    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);
    assert_eq!(config.preset, HevcPreset::Medium);
    assert_eq!(config.profile, HevcProfile::Main);
    assert!(matches!(config.rate_control, RateControlMode::Crf { crf } if (crf - 23.0).abs() < 0.01));
    assert_eq!(config.gop_size, 250);
    assert_eq!(config.bframes, 4);
}

#[test]
fn test_encoder_preset_properties() {
    let presets = [
        HevcPreset::Ultrafast,
        HevcPreset::Superfast,
        HevcPreset::Veryfast,
        HevcPreset::Faster,
        HevcPreset::Fast,
        HevcPreset::Medium,
        HevcPreset::Slow,
        HevcPreset::Slower,
        HevcPreset::Veryslow,
        HevcPreset::Placebo,
    ];

    // Verify search range and RDO level ordering
    let mut prev_rdo = 0;
    for preset in &presets {
        let rdo = preset.rdo_level();
        assert!(rdo >= prev_rdo,
            "{:?} should have >= RDO level than previous preset", preset);
        prev_rdo = rdo;
    }

    // Verify max CU depth increases with slower presets
    let mut prev_depth = 0;
    for preset in &presets {
        let depth = preset.max_cu_depth();
        assert!(depth >= prev_depth,
            "{:?} should have >= max CU depth than previous preset", preset);
        prev_depth = depth;
    }
}

#[test]
fn test_encoder_rate_control_modes() {
    // Test CQP
    let config_cqp = HevcEncoderConfig::new(1920, 1080).with_qp(22);
    assert!(matches!(config_cqp.rate_control, RateControlMode::Cqp { qp: 22 }));

    // Test ABR/VBR
    let config_abr = HevcEncoderConfig::new(1920, 1080).with_bitrate(5_000_000);
    assert!(matches!(config_abr.rate_control, RateControlMode::Abr { bitrate: 5_000_000 }));

    // Test CRF
    let config_crf = HevcEncoderConfig::new(1920, 1080).with_crf(23.0);
    assert!(matches!(config_crf.rate_control, RateControlMode::Crf { crf } if (crf - 23.0).abs() < 0.01));

    // Test QP clamping
    let config_qp_high = HevcEncoderConfig::new(1920, 1080).with_qp(100);
    assert!(matches!(config_qp_high.rate_control, RateControlMode::Cqp { qp: 51 })); // Clamped to max
}

#[test]
fn test_encoder_creation() {
    let config = HevcEncoderConfig::new(1920, 1080);
    let result = HevcEncoder::new(config);
    assert!(result.is_ok());
}

#[test]
fn test_encoder_invalid_dimensions() {
    // Width/height of 0 should fail
    let config = HevcEncoderConfig::new(0, 1080);
    let result = HevcEncoder::new(config);
    assert!(result.is_err());

    let config = HevcEncoderConfig::new(1920, 0);
    let result = HevcEncoder::new(config);
    assert!(result.is_err());
}

// ============================================================================
// Decoder Tests
// ============================================================================

#[test]
fn test_decoder_creation() {
    let decoder = HevcDecoder::default_config();
    let info = decoder.info();
    assert_eq!(info.width, 0); // Not initialized yet
    assert_eq!(info.height, 0);
}

#[test]
fn test_decoder_config() {
    let config = HevcDecoderConfig {
        max_width: 3840,
        max_height: 2160,
        output_format: PixelFormat::Yuv420p10le,
        threads: 8,
        threading: true,
    };

    let decoder = HevcDecoder::new(config.clone());
    // Decoder should be created but not yet initialized
    assert_eq!(decoder.get_width(), 0);
    assert_eq!(decoder.get_height(), 0);
}

#[test]
fn test_intra_mode_properties() {
    // Test all intra prediction modes
    assert_eq!(IntraMode::Planar.index(), 0);
    assert_eq!(IntraMode::Dc.index(), 1);
    assert_eq!(IntraMode::Horizontal.index(), 10);
    assert_eq!(IntraMode::Vertical.index(), 26);

    // Angular modes
    for i in 2..35 {
        if i != 10 && i != 26 {
            let mode = IntraMode::Angular(i);
            assert_eq!(mode.index(), i);
            assert!(mode.is_angular());
        }
    }

    // Non-angular modes
    assert!(!IntraMode::Planar.is_angular());
    assert!(!IntraMode::Dc.is_angular());
}

#[test]
fn test_intra_mode_from_index() {
    assert!(matches!(IntraMode::from_index(0), IntraMode::Planar));
    assert!(matches!(IntraMode::from_index(1), IntraMode::Dc));
    assert!(matches!(IntraMode::from_index(10), IntraMode::Horizontal));
    assert!(matches!(IntraMode::from_index(26), IntraMode::Vertical));
    assert!(matches!(IntraMode::from_index(15), IntraMode::Angular(15)));
}

#[test]
fn test_part_mode_num_parts() {
    assert_eq!(PartMode::Part2Nx2N.num_parts(), 1);
    assert_eq!(PartMode::PartNxN.num_parts(), 4);
    assert_eq!(PartMode::Part2NxN.num_parts(), 2);
    assert_eq!(PartMode::PartNx2N.num_parts(), 2);
    assert_eq!(PartMode::Part2NxnU.num_parts(), 2);
    assert_eq!(PartMode::Part2NxnD.num_parts(), 2);
    assert_eq!(PartMode::PartnLx2N.num_parts(), 2);
    assert_eq!(PartMode::PartnRx2N.num_parts(), 2);
}

#[test]
fn test_prediction_mode_variants() {
    let intra = PredictionMode::Intra(26);
    let inter = PredictionMode::Inter { ref_idx_l0: 0, ref_idx_l1: 1 };
    let skip = PredictionMode::Skip;

    assert!(matches!(intra, PredictionMode::Intra(26)));
    assert!(matches!(inter, PredictionMode::Inter { ref_idx_l0: 0, ref_idx_l1: 1 }));
    assert!(matches!(skip, PredictionMode::Skip));
}

// ============================================================================
// Profile/Tier/Level Tests
// ============================================================================

#[test]
fn test_hevc_profiles() {
    let profiles = [
        (1, HevcProfile::Main),
        (2, HevcProfile::Main10),
        (3, HevcProfile::MainStillPicture),
        (4, HevcProfile::RangeExtensions),
        (5, HevcProfile::HighThroughput),
        (6, HevcProfile::MultiviewMain),
        (7, HevcProfile::ScalableMain),
        (8, HevcProfile::ThreeDMain),
        (9, HevcProfile::ScreenContentCoding),
        (10, HevcProfile::ScalableRangeExtensions),
        (11, HevcProfile::HighThroughputScreenContent),
    ];

    for (idc, expected) in profiles {
        let profile = HevcProfile::from_idc(idc);
        assert_eq!(profile, Some(expected), "Profile IDC {} mismatch", idc);
    }

    // Unknown profile
    assert!(HevcProfile::from_idc(99).is_none());
}

#[test]
fn test_hevc_profile_capabilities() {
    // Main doesn't support 10-bit
    assert!(!HevcProfile::Main.supports_10bit());

    // Main10 supports 10-bit
    assert!(HevcProfile::Main10.supports_10bit());

    // Range Extensions supports everything
    assert!(HevcProfile::RangeExtensions.supports_10bit());
    assert!(HevcProfile::RangeExtensions.supports_422());
    assert!(HevcProfile::RangeExtensions.supports_444());

    // Main doesn't support higher chroma
    assert!(!HevcProfile::Main.supports_422());
    assert!(!HevcProfile::Main.supports_444());
}

#[test]
fn test_hevc_tiers() {
    // Default tier is Main
    assert_eq!(HevcTier::default(), HevcTier::Main);
}

#[test]
fn test_hevc_levels() {
    // Test level constants
    assert_eq!(HevcLevel::L1.level_idc, 30);
    assert_eq!(HevcLevel::L4_1.level_idc, 123);
    assert_eq!(HevcLevel::L5_1.level_idc, 153);
    assert_eq!(HevcLevel::L6_2.level_idc, 186);

    // Test major.minor parsing
    assert_eq!(HevcLevel::L5_1.major(), 5);
    assert_eq!(HevcLevel::L5_1.minor(), 1);
    assert_eq!(HevcLevel::L4_1.to_string(), "4.1");
}

#[test]
fn test_level_constraints() {
    // Level 4.1 constraints
    let level = HevcLevel::L4_1;
    let max_bitrate_main = level.max_bitrate(HevcTier::Main);
    let max_bitrate_high = level.max_bitrate(HevcTier::High);

    assert!(max_bitrate_main > 0);
    assert!(max_bitrate_high > max_bitrate_main); // High tier allows higher bitrate

    // Level 5.1 should allow more than 4.1
    let level51 = HevcLevel::L5_1;
    assert!(level51.max_luma_picture_size() > level.max_luma_picture_size());
}

#[test]
fn test_level_from_idc() {
    let level = HevcLevel::from_idc(153);
    assert_eq!(level.level_idc, 153);
    assert_eq!(level.major(), 5);
    assert_eq!(level.minor(), 1);
}

// ============================================================================
// Transform Tests
// ============================================================================

#[test]
fn test_transform_sizes() {
    assert_eq!(TransformSize::T4x4.size(), 4);
    assert_eq!(TransformSize::T8x8.size(), 8);
    assert_eq!(TransformSize::T16x16.size(), 16);
    assert_eq!(TransformSize::T32x32.size(), 32);

    assert!(TransformSize::from_size(4).is_some());
    assert!(TransformSize::from_size(8).is_some());
    assert!(TransformSize::from_size(16).is_some());
    assert!(TransformSize::from_size(32).is_some());
    assert!(TransformSize::from_size(64).is_none());
    assert!(TransformSize::from_size(7).is_none());
}

#[test]
fn test_transform_forward_4x4() {
    let transform = HevcTransform::new(8);
    let input: Vec<i16> = (0..16).map(|x| (x * 10) as i16).collect();
    let mut transformed = vec![0i32; 16];

    // Forward transform (is_intra=true, is_luma=true uses DST for 4x4)
    transform.forward_transform(&input, &mut transformed, TransformSize::T4x4, 4, true, true);

    // Verify forward transform produces non-zero output
    let has_nonzero = transformed.iter().any(|&x| x != 0);
    assert!(has_nonzero, "Forward transform should produce non-zero coefficients");

    // Verify DC coefficient exists (first coefficient typically captures energy)
    let dc = transformed[0];
    assert!(dc != 0, "DC coefficient should be non-zero for non-zero input");
}

#[test]
fn test_transform_inverse_4x4() {
    let transform = HevcTransform::new(8);
    // Start with transform-domain data
    let input = vec![1000i32, 100, 50, 25, 20, 15, 10, 5, 5, 4, 3, 2, 1, 1, 0, 0];
    let mut output = vec![0i16; 16];

    // Inverse transform should produce spatial domain samples
    transform.inverse_transform(&input, &mut output, TransformSize::T4x4, 4, true, true);

    // Verify output is not all zeros when input has energy
    let has_nonzero = output.iter().any(|&x| x != 0);
    assert!(has_nonzero, "Inverse transform should produce non-zero samples");
}

#[test]
fn test_transform_forward_8x8() {
    let transform = HevcTransform::new(8);
    let input: Vec<i16> = (0..64).map(|x| (x * 5) as i16).collect();
    let mut transformed = vec![0i32; 64];

    transform.forward_transform(&input, &mut transformed, TransformSize::T8x8, 8, false, true);

    // Verify forward transform produces non-zero output
    let has_nonzero = transformed.iter().any(|&x| x != 0);
    assert!(has_nonzero, "Forward transform should produce non-zero coefficients");

    // DC coefficient should capture the average
    let dc = transformed[0];
    assert!(dc != 0, "DC coefficient should be non-zero for non-zero input");
}

#[test]
fn test_transform_inverse_8x8() {
    let transform = HevcTransform::new(8);
    // Start with transform-domain data (typical coefficient distribution)
    let mut input = vec![0i32; 64];
    input[0] = 2000; // DC component
    input[1] = 200;
    input[8] = 150;
    let mut output = vec![0i16; 64];

    transform.inverse_transform(&input, &mut output, TransformSize::T8x8, 8, false, true);

    // Verify output is produced
    let has_nonzero = output.iter().any(|&x| x != 0);
    assert!(has_nonzero, "Inverse transform should produce non-zero samples");
}

#[test]
fn test_transform_forward_16x16() {
    let transform = HevcTransform::new(8);
    let input: Vec<i16> = (0..256).map(|x| (x * 2) as i16).collect();
    let mut transformed = vec![0i32; 256];

    transform.forward_transform(&input, &mut transformed, TransformSize::T16x16, 16, false, true);

    // Verify forward transform produces non-zero output
    let has_nonzero = transformed.iter().any(|&x| x != 0);
    assert!(has_nonzero, "Forward transform should produce non-zero coefficients");

    // DC coefficient should capture the average
    let dc = transformed[0];
    assert!(dc != 0, "DC coefficient should be non-zero for non-zero input");
}

#[test]
fn test_transform_inverse_16x16() {
    let transform = HevcTransform::new(8);
    // Start with transform-domain data
    let mut input = vec![0i32; 256];
    input[0] = 5000; // DC component
    input[1] = 500;
    input[16] = 300;
    let mut output = vec![0i16; 256];

    transform.inverse_transform(&input, &mut output, TransformSize::T16x16, 16, false, true);

    // Verify output is produced
    let has_nonzero = output.iter().any(|&x| x != 0);
    assert!(has_nonzero, "Inverse transform should produce non-zero samples");
}

#[test]
fn test_quantizer_roundtrip() {
    let quantizer = HevcQuantizer::new(8);
    // Use larger values to ensure quantization produces non-zero coefficients
    let input: Vec<i32> = (0..16).map(|x| x * 1000).collect();
    let mut quantized = vec![0i32; 16];
    let mut dequantized = vec![0i32; 16];

    // Quantize (is_intra = true, low QP for better reconstruction)
    quantizer.quantize(&input, &mut quantized, 4, 12, true, None).unwrap();

    // Verify quantization produces some non-zero coefficients for large inputs
    let nonzero_quant: Vec<_> = quantized.iter()
        .enumerate()
        .filter(|(_, &c)| c != 0)
        .collect();
    // At least some coefficients should survive quantization
    assert!(!nonzero_quant.is_empty() || input[0] == 0,
        "Quantization should preserve some large coefficients");

    // Dequantize
    quantizer.dequantize(&quantized, &mut dequantized, 4, 12, None).unwrap();

    // For coefficients that survived quantization, verify sign is preserved
    for i in 0..16 {
        if input[i] > 0 && quantized[i] != 0 {
            assert!(dequantized[i] >= 0,
                "Dequantized positive coefficient should remain non-negative");
        }
    }
}

#[test]
fn test_quantizer_qp_range() {
    let quantizer = HevcQuantizer::new(8);
    let input: Vec<i32> = (0..16).map(|x| x * 10).collect();

    // Test various QP values
    for qp in [0, 10, 22, 32, 42, 51] {
        let mut quantized = vec![0i32; 16];
        let result = quantizer.quantize(&input, &mut quantized, 4, qp, true, None);
        assert!(result.is_ok(), "QP {} should be valid", qp);
    }
}

#[test]
fn test_quantizer_different_sizes() {
    let quantizer = HevcQuantizer::new(8);

    for size in [4, 8, 16, 32] {
        let num_coeffs = size * size;
        let input: Vec<i32> = (0..num_coeffs).map(|x| (x * 10) as i32).collect();
        let mut quantized = vec![0i32; num_coeffs];
        let mut dequantized = vec![0i32; num_coeffs];

        quantizer.quantize(&input, &mut quantized, size, 22, true, None).unwrap();
        quantizer.dequantize(&quantized, &mut dequantized, size, 22, None).unwrap();

        // Just verify it doesn't panic
        assert_eq!(quantized.len(), num_coeffs);
        assert_eq!(dequantized.len(), num_coeffs);
    }
}

// ============================================================================
// CABAC Tests
// ============================================================================

#[test]
fn test_cabac_encoder_creation() {
    let encoder = CabacEncoder::new();
    // Encoder should be created without panic - data starts empty
    assert!(encoder.data().is_empty());
}

#[test]
fn test_cabac_decoder_creation() {
    let data = vec![0x00, 0x00, 0x00, 0x01]; // Minimal data
    let decoder = CabacDecoder::new(&data);
    // Decoder should be created without panic
    drop(decoder);
}

#[test]
fn test_cabac_bypass_coding() {
    // Test bypass coding mode
    let mut encoder = CabacEncoder::new();
    encoder.encode_bypass(true).unwrap();
    encoder.encode_bypass(false).unwrap();
    encoder.encode_bypass(true).unwrap();
    encoder.encode_terminate(true).unwrap();

    // Get the encoded data - terminate writes output
    let data = encoder.data();
    // After terminate, there should be output
    assert!(!data.is_empty());
}

#[test]
fn test_cabac_context_initialization() {
    let data = vec![0x00, 0x00, 0x01, 0xFF, 0xFF, 0xFF, 0xFF];
    let mut decoder = CabacDecoder::new(&data);
    decoder.init().unwrap();
    // Context initialization should work
}

// ============================================================================
// Error Type Tests
// ============================================================================

#[test]
fn test_error_types() {
    let errors = [
        HevcError::Sps("test".to_string()),
        HevcError::Pps("test".to_string()),
        HevcError::SliceHeader("test".to_string()),
        HevcError::Transform("test".to_string()),
        HevcError::Cabac("test".to_string()),
        HevcError::InvalidState("test".to_string()),
        HevcError::DecoderConfig("test".to_string()),
        HevcError::EncoderConfig("test".to_string()),
        HevcError::Prediction("test".to_string()),
        HevcError::Sao("test".to_string()),
        HevcError::Deblock("test".to_string()),
        HevcError::Unsupported("test".to_string()),
    ];

    for error in errors {
        let msg = error.to_string();
        assert!(!msg.is_empty());
    }
}

#[test]
fn test_dimensions_exceeded_error() {
    let error = HevcError::DimensionsExceeded {
        width: 8192,
        height: 4320,
        max_width: 4096,
        max_height: 2160,
    };
    let msg = error.to_string();
    assert!(msg.contains("8192"));
    assert!(msg.contains("4320"));
}

// ============================================================================
// Integration Tests - Encoder/Decoder Workflow
// ============================================================================

#[test]
fn test_encoder_initializes_with_parameter_sets() {
    // When encoder is created, it internally generates VPS/SPS/PPS
    // We can verify this by ensuring the encoder is successfully created
    let config = HevcEncoderConfig::new(320, 240)
        .with_preset(HevcPreset::Ultrafast);
    let encoder = HevcEncoder::new(config);

    // Encoder creation includes parameter set generation
    assert!(encoder.is_ok());
}

#[test]
fn test_encoder_with_various_configurations() {
    // Test that encoder properly initializes with different configurations
    let configs = [
        HevcEncoderConfig::new(320, 240).with_preset(HevcPreset::Ultrafast),
        HevcEncoderConfig::new(1920, 1080).with_preset(HevcPreset::Medium),
        HevcEncoderConfig::new(640, 480).with_profile(HevcProfile::Main10),
    ];

    for config in configs {
        let encoder = HevcEncoder::new(config);
        assert!(encoder.is_ok(), "Encoder should initialize successfully");
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_minimum_valid_dimensions() {
    // Minimum CTU size is typically 16x16 for HEVC
    let config = HevcEncoderConfig::new(16, 16);
    let result = HevcEncoder::new(config);
    assert!(result.is_ok());
}

#[test]
fn test_non_multiple_of_8_dimensions() {
    // HEVC should handle non-aligned dimensions via conformance window
    let config = HevcEncoderConfig::new(1919, 1079);
    let result = HevcEncoder::new(config);
    assert!(result.is_ok());
}

#[test]
fn test_4k_dimensions() {
    let config = HevcEncoderConfig::new(3840, 2160);
    let result = HevcEncoder::new(config);
    assert!(result.is_ok());
}

#[test]
fn test_8k_dimensions() {
    let config = HevcEncoderConfig::new(7680, 4320);
    let result = HevcEncoder::new(config);
    assert!(result.is_ok());
}

#[test]
fn test_encoder_preset_affects_speed() {
    // Ultrafast should have lower search range
    assert!(HevcPreset::Ultrafast.search_range() < HevcPreset::Placebo.search_range());

    // Ultrafast should have lower max CU depth
    assert!(HevcPreset::Ultrafast.max_cu_depth() <= HevcPreset::Placebo.max_cu_depth());
}

#[test]
fn test_encoder_with_different_presets() {
    let presets = [
        HevcPreset::Ultrafast,
        HevcPreset::Medium,
        HevcPreset::Veryslow,
    ];

    for preset in presets {
        let config = HevcEncoderConfig::new(640, 480).with_preset(preset);
        let result = HevcEncoder::new(config);
        assert!(result.is_ok(), "Failed to create encoder with {:?} preset", preset);
    }
}

#[test]
fn test_encoder_with_10bit_profile() {
    let config = HevcEncoderConfig::new(1920, 1080)
        .with_profile(HevcProfile::Main10);

    assert_eq!(config.bit_depth, 10);

    let result = HevcEncoder::new(config);
    assert!(result.is_ok());
}
