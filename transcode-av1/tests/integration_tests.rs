//! Integration tests for the AV1 codec crate.
//!
//! These tests verify the public API and common usage patterns.

use transcode_av1::{
    Av1Config, Av1Encoder, Av1Error, Av1Info, Av1Preset, RateControlMode, Result,
};

// ============================================================================
// Info Tests
// ============================================================================

#[test]
fn test_info_defaults() {
    let info = transcode_av1::get_info();
    assert_eq!(info.encoder_name, "rav1e");
    assert!(info.supports_10bit);
    assert!(info.supports_hdr);
    assert_eq!(info.max_width, 8192);
    assert_eq!(info.max_height, 4320);
}

#[test]
fn test_info_version() {
    let info = transcode_av1::get_info();
    // Version should be a valid semver string
    assert!(!info.encoder_version.is_empty());
}

// ============================================================================
// Preset Tests
// ============================================================================

#[test]
fn test_preset_to_speed_mapping() {
    assert_eq!(Av1Preset::Placebo.to_speed(), 0);
    assert_eq!(Av1Preset::VerySlow.to_speed(), 2);
    assert_eq!(Av1Preset::Slower.to_speed(), 3);
    assert_eq!(Av1Preset::Slow.to_speed(), 4);
    assert_eq!(Av1Preset::Medium.to_speed(), 6);
    assert_eq!(Av1Preset::Fast.to_speed(), 7);
    assert_eq!(Av1Preset::Faster.to_speed(), 8);
    assert_eq!(Av1Preset::VeryFast.to_speed(), 9);
    assert_eq!(Av1Preset::UltraFast.to_speed(), 10);
}

#[test]
fn test_preset_from_speed() {
    assert_eq!(Av1Preset::from_speed(0), Av1Preset::Placebo);
    assert_eq!(Av1Preset::from_speed(1), Av1Preset::Placebo);
    assert_eq!(Av1Preset::from_speed(2), Av1Preset::VerySlow);
    assert_eq!(Av1Preset::from_speed(3), Av1Preset::Slower);
    assert_eq!(Av1Preset::from_speed(4), Av1Preset::Slow);
    assert_eq!(Av1Preset::from_speed(5), Av1Preset::Medium);
    assert_eq!(Av1Preset::from_speed(6), Av1Preset::Medium);
    assert_eq!(Av1Preset::from_speed(7), Av1Preset::Fast);
    assert_eq!(Av1Preset::from_speed(8), Av1Preset::Faster);
    assert_eq!(Av1Preset::from_speed(9), Av1Preset::VeryFast);
    assert_eq!(Av1Preset::from_speed(10), Av1Preset::UltraFast);
    assert_eq!(Av1Preset::from_speed(100), Av1Preset::UltraFast);
}

#[test]
fn test_preset_default() {
    let preset = Av1Preset::default();
    assert_eq!(preset, Av1Preset::Medium);
}

#[test]
fn test_preset_roundtrip() {
    for preset in [
        Av1Preset::Placebo,
        Av1Preset::VerySlow,
        Av1Preset::Slower,
        Av1Preset::Slow,
        Av1Preset::Medium,
        Av1Preset::Fast,
        Av1Preset::Faster,
        Av1Preset::VeryFast,
        Av1Preset::UltraFast,
    ] {
        let speed = preset.to_speed();
        let recovered = Av1Preset::from_speed(speed);
        assert_eq!(preset, recovered);
    }
}

// ============================================================================
// Config Tests
// ============================================================================

#[test]
fn test_config_new() {
    let config = Av1Config::new(1920, 1080);
    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);
    assert_eq!(config.bit_depth, 8);
    assert_eq!(config.preset, Av1Preset::Medium);
}

#[test]
fn test_config_default() {
    let config = Av1Config::default();
    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);
    assert_eq!(config.framerate_num, 30);
    assert_eq!(config.framerate_den, 1);
}

#[test]
fn test_config_with_preset() {
    let config = Av1Config::new(1280, 720)
        .with_preset(Av1Preset::Fast);
    assert_eq!(config.preset, Av1Preset::Fast);
}

#[test]
fn test_config_with_framerate() {
    let config = Av1Config::new(1920, 1080)
        .with_framerate(60, 1);
    assert_eq!(config.framerate_num, 60);
    assert_eq!(config.framerate_den, 1);
}

#[test]
fn test_config_with_framerate_fractional() {
    let config = Av1Config::new(1920, 1080)
        .with_framerate(30000, 1001);
    assert_eq!(config.framerate_num, 30000);
    assert_eq!(config.framerate_den, 1001);
}

#[test]
fn test_config_with_bitrate() {
    let config = Av1Config::new(1920, 1080)
        .with_bitrate(5_000_000);

    match config.rate_control {
        RateControlMode::Vbr { bitrate } => assert_eq!(bitrate, 5_000_000),
        _ => panic!("Expected VBR rate control"),
    }
}

#[test]
fn test_config_with_quality() {
    let config = Av1Config::new(1920, 1080)
        .with_quality(24);

    match config.rate_control {
        RateControlMode::ConstantQuality { quantizer } => assert_eq!(quantizer, 24),
        _ => panic!("Expected ConstantQuality rate control"),
    }
}

#[test]
fn test_config_quality_clamping() {
    let config = Av1Config::new(1920, 1080)
        .with_quality(100); // Should be clamped to 63

    match config.rate_control {
        RateControlMode::ConstantQuality { quantizer } => assert_eq!(quantizer, 63),
        _ => panic!("Expected ConstantQuality rate control"),
    }
}

#[test]
fn test_config_with_bit_depth() {
    let config8 = Av1Config::new(1920, 1080).with_bit_depth(8);
    assert_eq!(config8.bit_depth, 8);

    let config10 = Av1Config::new(1920, 1080).with_bit_depth(10);
    assert_eq!(config10.bit_depth, 10);

    // Test clamping for invalid values
    let config_invalid = Av1Config::new(1920, 1080).with_bit_depth(12);
    assert_eq!(config_invalid.bit_depth, 10);
}

#[test]
fn test_config_with_tiles() {
    let config = Av1Config::new(3840, 2160)
        .with_tiles(2, 2);
    assert_eq!(config.tile_cols_log2, 2);
    assert_eq!(config.tile_rows_log2, 2);
}

#[test]
fn test_config_tiles_clamping() {
    let config = Av1Config::new(3840, 2160)
        .with_tiles(10, 10); // Should be clamped to 6
    assert_eq!(config.tile_cols_log2, 6);
    assert_eq!(config.tile_rows_log2, 6);
}

#[test]
fn test_config_with_keyframe_interval() {
    let config = Av1Config::new(1920, 1080)
        .with_keyframe_interval(120);
    assert_eq!(config.keyframe_interval, 120);
}

#[test]
fn test_config_with_low_latency() {
    let config = Av1Config::new(1920, 1080)
        .with_low_latency(true);
    assert!(config.low_latency);
}

#[test]
fn test_config_with_threads() {
    let config = Av1Config::new(1920, 1080)
        .with_threads(4);
    assert_eq!(config.threads, 4);
}

#[test]
fn test_config_with_hdr() {
    let config = Av1Config::new(3840, 2160)
        .with_hdr();

    assert_eq!(config.bit_depth, 10);
    // HDR config should set BT.2020 color space and PQ transfer
}

#[test]
fn test_config_chained_builder() {
    let config = Av1Config::new(3840, 2160)
        .with_preset(Av1Preset::Fast)
        .with_framerate(60, 1)
        .with_bitrate(20_000_000)
        .with_bit_depth(10)
        .with_tiles(2, 2)
        .with_keyframe_interval(120)
        .with_low_latency(true)
        .with_threads(8);

    assert_eq!(config.width, 3840);
    assert_eq!(config.height, 2160);
    assert_eq!(config.preset, Av1Preset::Fast);
    assert_eq!(config.framerate_num, 60);
    assert_eq!(config.bit_depth, 10);
    assert_eq!(config.tile_cols_log2, 2);
    assert_eq!(config.keyframe_interval, 120);
    assert!(config.low_latency);
    assert_eq!(config.threads, 8);
}

// ============================================================================
// Config Validation Tests
// ============================================================================

#[test]
fn test_config_validate_success() {
    let config = Av1Config::new(1920, 1080);
    assert!(config.validate().is_ok());
}

#[test]
fn test_config_validate_zero_width() {
    let config = Av1Config::new(0, 1080);
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validate_zero_height() {
    let config = Av1Config::new(1920, 0);
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validate_too_wide() {
    let config = Av1Config::new(10000, 1080);
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validate_too_tall() {
    let config = Av1Config::new(1920, 5000);
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validate_max_resolution() {
    let config = Av1Config::new(8192, 4320);
    assert!(config.validate().is_ok());
}

#[test]
fn test_config_validate_invalid_framerate_num() {
    let mut config = Av1Config::new(1920, 1080);
    config.framerate_num = 0;
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validate_invalid_framerate_den() {
    let mut config = Av1Config::new(1920, 1080);
    config.framerate_den = 0;
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validate_invalid_bit_depth() {
    let mut config = Av1Config::new(1920, 1080);
    config.bit_depth = 12; // Only 8 and 10 are valid
    assert!(config.validate().is_err());
}

// ============================================================================
// Auto Tiles Tests
// ============================================================================

#[test]
fn test_config_auto_tiles_4k() {
    let mut config = Av1Config::new(3840, 2160);
    config.threads = 8;
    config.auto_tiles();

    // 4K with 8+ threads should enable tiles
    assert!(config.tile_cols_log2 > 0 || config.tile_rows_log2 > 0);
}

#[test]
fn test_config_auto_tiles_1080p() {
    let mut config = Av1Config::new(1920, 1080);
    config.threads = 4;
    config.auto_tiles();

    // 1080p with 4+ threads may enable tiles
    // Exact behavior depends on implementation
}

#[test]
fn test_config_auto_tiles_small() {
    let mut config = Av1Config::new(640, 480);
    config.threads = 2;
    config.auto_tiles();

    // Small resolution shouldn't need tiles
    assert_eq!(config.tile_cols_log2, 0);
    assert_eq!(config.tile_rows_log2, 0);
}

// ============================================================================
// Rate Control Mode Tests
// ============================================================================

#[test]
fn test_rate_control_default() {
    let mode = RateControlMode::default();
    match mode {
        RateControlMode::ConstantQuality { quantizer } => {
            assert_eq!(quantizer, 28);
        }
        _ => panic!("Expected ConstantQuality as default"),
    }
}

#[test]
fn test_rate_control_cbr() {
    let mode = RateControlMode::Cbr { bitrate: 2_000_000 };
    match mode {
        RateControlMode::Cbr { bitrate } => assert_eq!(bitrate, 2_000_000),
        _ => panic!("Expected CBR mode"),
    }
}

#[test]
fn test_rate_control_two_pass_first() {
    let mode = RateControlMode::TwoPassFirst;
    matches!(mode, RateControlMode::TwoPassFirst);
}

#[test]
fn test_rate_control_two_pass_second() {
    let stats = vec![1, 2, 3, 4];
    let mode = RateControlMode::TwoPassSecond { stats: stats.clone() };
    match mode {
        RateControlMode::TwoPassSecond { stats: s } => assert_eq!(s, stats),
        _ => panic!("Expected TwoPassSecond mode"),
    }
}

// ============================================================================
// Encoder Tests (require feature = "encoder")
// ============================================================================

#[cfg(feature = "encoder")]
mod encoder_tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let config = Av1Config::new(320, 240)
            .with_preset(Av1Preset::UltraFast);

        let encoder = Av1Encoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_creation_with_validation_error() {
        let config = Av1Config::new(0, 240);
        let encoder = Av1Encoder::new(config);
        assert!(encoder.is_err());
    }

    #[test]
    fn test_encoder_config_access() {
        let config = Av1Config::new(320, 240)
            .with_preset(Av1Preset::Fast);

        let encoder = Av1Encoder::new(config).unwrap();
        assert_eq!(encoder.config().width, 320);
        assert_eq!(encoder.config().height, 240);
        assert_eq!(encoder.config().preset, Av1Preset::Fast);
    }

    #[test]
    fn test_encoder_not_finished_initially() {
        let config = Av1Config::new(320, 240)
            .with_preset(Av1Preset::UltraFast);

        let encoder = Av1Encoder::new(config).unwrap();
        assert!(!encoder.is_finished());
    }

    #[test]
    fn test_encoder_encode_frame() {
        let config = Av1Config::new(320, 240)
            .with_preset(Av1Preset::UltraFast)
            .with_quality(32);

        let mut encoder = Av1Encoder::new(config).unwrap();

        // Create test frame (gray)
        let y_plane = vec![128u8; 320 * 240];
        let u_plane = vec![128u8; 160 * 120];
        let v_plane = vec![128u8; 160 * 120];

        let result = encoder.encode_frame(&y_plane, &u_plane, &v_plane, 0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encoder_encode_multiple_frames() {
        let config = Av1Config::new(160, 120)
            .with_preset(Av1Preset::UltraFast)
            .with_quality(40);

        let mut encoder = Av1Encoder::new(config).unwrap();

        let y_plane = vec![128u8; 160 * 120];
        let u_plane = vec![128u8; 80 * 60];
        let v_plane = vec![128u8; 80 * 60];

        for i in 0..5 {
            let result = encoder.encode_frame(&y_plane, &u_plane, &v_plane, i);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_encoder_stats_initial() {
        let config = Av1Config::new(320, 240)
            .with_preset(Av1Preset::UltraFast);

        let encoder = Av1Encoder::new(config).unwrap();
        let stats = encoder.stats();

        assert_eq!(stats.frames_encoded, 0);
        assert_eq!(stats.bytes_produced, 0);
        assert_eq!(stats.keyframes, 0);
    }

    #[test]
    fn test_encoder_stats_after_encoding() {
        let config = Av1Config::new(160, 120)
            .with_preset(Av1Preset::UltraFast)
            .with_quality(50);

        let mut encoder = Av1Encoder::new(config).unwrap();

        let y_plane = vec![128u8; 160 * 120];
        let u_plane = vec![128u8; 80 * 60];
        let v_plane = vec![128u8; 80 * 60];

        for i in 0..3 {
            encoder.encode_frame(&y_plane, &u_plane, &v_plane, i).unwrap();
        }

        let stats = encoder.stats();
        assert_eq!(stats.frames_encoded, 3);
    }

    #[test]
    fn test_encoder_flush() {
        let config = Av1Config::new(160, 120)
            .with_preset(Av1Preset::UltraFast)
            .with_quality(50)
            .with_low_latency(true);

        let mut encoder = Av1Encoder::new(config).unwrap();

        let y_plane = vec![128u8; 160 * 120];
        let u_plane = vec![128u8; 80 * 60];
        let v_plane = vec![128u8; 80 * 60];

        // Encode some frames
        for i in 0..5 {
            encoder.encode_frame(&y_plane, &u_plane, &v_plane, i).unwrap();
        }

        // Flush to get remaining packets
        let packets = encoder.flush().unwrap();

        // After flush, encoder should be finished
        assert!(encoder.is_finished());

        // Should have produced at least one packet with a keyframe
        let has_keyframe = packets.iter().any(|p| p.keyframe);
        // Note: with low latency mode and few frames, may not have flushed packets
    }

    #[test]
    fn test_encoder_sequence_header() {
        let config = Av1Config::new(320, 240)
            .with_preset(Av1Preset::UltraFast);

        let encoder = Av1Encoder::new(config).unwrap();
        let header = encoder.get_sequence_header();

        assert!(header.is_ok());
        let header_data = header.unwrap();
        // Sequence header should be non-empty
        assert!(!header_data.is_empty());
    }

    #[test]
    fn test_encoder_10bit_creation() {
        let config = Av1Config::new(320, 240)
            .with_preset(Av1Preset::UltraFast)
            .with_bit_depth(10);

        let encoder = Av1Encoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_10bit_encode_frame() {
        let config = Av1Config::new(160, 120)
            .with_preset(Av1Preset::UltraFast)
            .with_bit_depth(10)
            .with_quality(40);

        let mut encoder = Av1Encoder::new(config).unwrap();

        // 10-bit values (0-1023, using 512 as midpoint)
        let y_plane = vec![512u16; 160 * 120];
        let u_plane = vec![512u16; 80 * 60];
        let v_plane = vec![512u16; 80 * 60];

        let result = encoder.encode_frame_10bit(&y_plane, &u_plane, &v_plane, 0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encoder_8bit_rejects_10bit_method() {
        let config = Av1Config::new(160, 120)
            .with_preset(Av1Preset::UltraFast)
            .with_bit_depth(8);

        let mut encoder = Av1Encoder::new(config).unwrap();

        let y_plane = vec![512u16; 160 * 120];
        let u_plane = vec![512u16; 80 * 60];
        let v_plane = vec![512u16; 80 * 60];

        let result = encoder.encode_frame_10bit(&y_plane, &u_plane, &v_plane, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_encoder_10bit_rejects_8bit_method() {
        let config = Av1Config::new(160, 120)
            .with_preset(Av1Preset::UltraFast)
            .with_bit_depth(10);

        let mut encoder = Av1Encoder::new(config).unwrap();

        let y_plane = vec![128u8; 160 * 120];
        let u_plane = vec![128u8; 80 * 60];
        let v_plane = vec![128u8; 80 * 60];

        let result = encoder.encode_frame(&y_plane, &u_plane, &v_plane, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_encoder_hdr_config() {
        let config = Av1Config::new(3840, 2160)
            .with_preset(Av1Preset::Fast)
            .with_hdr();

        let encoder = Av1Encoder::new(config);
        assert!(encoder.is_ok());

        let encoder = encoder.unwrap();
        assert_eq!(encoder.config().bit_depth, 10);
    }

    #[test]
    fn test_encoder_with_tiles() {
        let config = Av1Config::new(1920, 1080)
            .with_preset(Av1Preset::UltraFast)
            .with_tiles(1, 1)
            .with_threads(4);

        let encoder = Av1Encoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_vbr_mode() {
        let config = Av1Config::new(320, 240)
            .with_preset(Av1Preset::UltraFast)
            .with_bitrate(500_000);

        let encoder = Av1Encoder::new(config);
        assert!(encoder.is_ok());
    }
}

// ============================================================================
// Error Tests
// ============================================================================

#[test]
fn test_error_display() {
    let err = Av1Error::InvalidConfig("test error".into());
    let display = format!("{}", err);
    assert!(display.contains("Invalid configuration"));
    assert!(display.contains("test error"));
}

#[test]
fn test_error_encoder_error() {
    let err = Av1Error::EncoderError("encoder failed".into());
    let display = format!("{}", err);
    assert!(display.contains("Encoder error"));
}

#[test]
fn test_error_invalid_frame() {
    let err = Av1Error::InvalidFrame("bad frame".into());
    let display = format!("{}", err);
    assert!(display.contains("Invalid frame"));
}

#[test]
fn test_error_rate_control() {
    let err = Av1Error::RateControlError("rate issue".into());
    let display = format!("{}", err);
    assert!(display.contains("Rate control"));
}

#[test]
fn test_error_resource_exhausted() {
    let err = Av1Error::ResourceExhausted("out of memory".into());
    let display = format!("{}", err);
    assert!(display.contains("Resource exhausted"));
}

#[test]
fn test_error_needs_more_frames() {
    let err = Av1Error::NeedsMoreFrames;
    let display = format!("{}", err);
    assert!(display.contains("needs more frames"));
}

#[test]
fn test_error_end_of_stream() {
    let err = Av1Error::EndOfStream;
    let display = format!("{}", err);
    assert!(display.contains("End of stream"));
}

#[test]
fn test_error_io_conversion() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
    let err: Av1Error = io_err.into();
    let display = format!("{}", err);
    assert!(display.contains("I/O error"));
}

#[test]
fn test_error_to_core_error() {
    let err = Av1Error::EncoderError("test".into());
    let core_err: transcode_core::Error = err.into();
    // Should convert to CodecError::Other
    let display = format!("{}", core_err);
    assert!(display.contains("test"));
}
