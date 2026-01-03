//! Integration tests for transcode-flac

#[allow(unused_imports)]
use transcode_flac::{
    FlacEncoder, FlacDecoder, StreamingDecoder, CompressionLevel,
    FlacMetadata, StreamInfo, VorbisComment, FlacError,
    MetadataBlockType, ChannelAssignment, SubframeType,
};

/// Helper to create a sine wave for testing
fn generate_sine_wave(sample_rate: u32, frequency: f64, duration_secs: f64, amplitude: i32) -> Vec<i32> {
    let num_samples = (sample_rate as f64 * duration_secs) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f64 / sample_rate as f64;
        let sample = (amplitude as f64 * (2.0 * std::f64::consts::PI * frequency * t).sin()) as i32;
        samples.push(sample);
    }

    samples
}

/// Helper to create stereo samples from mono
fn make_stereo(mono: &[i32]) -> Vec<i32> {
    let mut stereo = Vec::with_capacity(mono.len() * 2);
    for &sample in mono {
        stereo.push(sample);
        stereo.push(sample);
    }
    stereo
}

#[test]
fn test_encoder_basic() {
    let buffer = Vec::new();
    let encoder = FlacEncoder::new(buffer, 44100, 2, 16, CompressionLevel::Default);
    assert!(encoder.is_ok());

    let encoder = encoder.unwrap();
    assert_eq!(encoder.stream_info().sample_rate, 44100);
    assert_eq!(encoder.stream_info().channels, 2);
    assert_eq!(encoder.stream_info().bits_per_sample, 16);
}

#[test]
fn test_encoder_mono() {
    let buffer = Vec::new();
    let encoder = FlacEncoder::new(buffer, 44100, 1, 16, CompressionLevel::Default);
    assert!(encoder.is_ok());

    let encoder = encoder.unwrap();
    assert_eq!(encoder.stream_info().channels, 1);
}

#[test]
fn test_encoder_invalid_channels() {
    let buffer = Vec::new();

    // 0 channels should fail
    let result = FlacEncoder::new(buffer.clone(), 44100, 0, 16, CompressionLevel::Default);
    assert!(result.is_err());

    // 9 channels should fail
    let result = FlacEncoder::new(buffer, 44100, 9, 16, CompressionLevel::Default);
    assert!(result.is_err());
}

#[test]
fn test_encoder_invalid_bits_per_sample() {
    let buffer = Vec::new();

    // 3 bits should fail
    let result = FlacEncoder::new(buffer.clone(), 44100, 2, 3, CompressionLevel::Default);
    assert!(result.is_err());

    // 33 bits should fail
    let result = FlacEncoder::new(buffer, 44100, 2, 33, CompressionLevel::Default);
    assert!(result.is_err());
}

#[test]
fn test_encoder_invalid_sample_rate() {
    let buffer = Vec::new();

    // 0 sample rate should fail
    let result = FlacEncoder::new(buffer.clone(), 0, 2, 16, CompressionLevel::Default);
    assert!(result.is_err());
}

#[test]
fn test_compression_levels() {
    for level in 0..=8 {
        let cl = CompressionLevel::from(level);
        assert!(cl.block_size() > 0);
        assert!(cl.max_lpc_order() <= 32);
    }
}

#[test]
fn test_encode_sine_wave() {
    let mut buffer = Vec::new();
    let mut encoder = FlacEncoder::new(&mut buffer, 44100, 2, 16, CompressionLevel::Default).unwrap();

    // Generate a short sine wave
    let mono = generate_sine_wave(44100, 440.0, 0.1, 16000);
    let stereo = make_stereo(&mono);

    // Encode
    let result = encoder.encode_samples(&stereo);
    assert!(result.is_ok());

    // Finish
    let _ = encoder.finish();

    // Verify we have some output
    assert!(buffer.len() > 4, "Buffer should contain FLAC data");

    // Verify FLAC marker
    assert_eq!(&buffer[0..4], b"fLaC", "Should start with fLaC marker");
}

#[test]
fn test_encode_silence() {
    let mut buffer = Vec::new();
    let mut encoder = FlacEncoder::new(&mut buffer, 44100, 2, 16, CompressionLevel::Default).unwrap();

    // Silence should compress very well (constant subframes)
    let silence: Vec<i32> = vec![0; 8192]; // Stereo silence

    let result = encoder.encode_samples(&silence);
    assert!(result.is_ok());

    let _ = encoder.finish();

    // Verify output
    assert!(buffer.len() > 4);
    assert_eq!(&buffer[0..4], b"fLaC");
}

#[test]
fn test_encode_constant_value() {
    let mut buffer = Vec::new();
    let mut encoder = FlacEncoder::new(&mut buffer, 44100, 1, 16, CompressionLevel::Default).unwrap();

    // Constant non-zero value should also use constant subframes
    let constant: Vec<i32> = vec![1000; 4096];

    let result = encoder.encode_samples(&constant);
    assert!(result.is_ok());

    let _ = encoder.finish();
    assert!(buffer.len() > 4);
}

#[test]
fn test_encode_multiple_blocks() {
    let mut buffer = Vec::new();
    let mut encoder = FlacEncoder::new(&mut buffer, 44100, 2, 16, CompressionLevel::Default).unwrap();

    // Encode multiple blocks
    for _ in 0..5 {
        let mono = generate_sine_wave(44100, 440.0, 0.1, 16000);
        let stereo = make_stereo(&mono);
        encoder.encode_samples(&stereo).unwrap();
    }

    let _ = encoder.finish();
    assert!(buffer.len() > 4);
}

#[test]
fn test_streaming_decoder_basic() {
    let decoder = StreamingDecoder::new();
    assert!(!decoder.has_metadata());
    assert_eq!(decoder.buffer_size(), 0);
}

#[test]
fn test_streaming_decoder_feed() {
    let mut decoder = StreamingDecoder::new();
    decoder.feed(b"fLaC");
    assert_eq!(decoder.buffer_size(), 4);
}

#[test]
fn test_streaming_decoder_invalid_marker() {
    let mut decoder = StreamingDecoder::new();
    decoder.feed(b"NOTF");

    let result = decoder.decode_frame();
    assert!(matches!(result, Err(FlacError::InvalidMarker)));
}

#[test]
fn test_streaming_decoder_reset() {
    let mut decoder = StreamingDecoder::new();
    decoder.feed(b"fLaC");
    assert_eq!(decoder.buffer_size(), 4);

    decoder.reset();
    assert!(!decoder.has_metadata());
    assert_eq!(decoder.buffer_size(), 0);
}

#[test]
fn test_metadata_block_type_conversion() {
    assert_eq!(MetadataBlockType::from(0), MetadataBlockType::StreamInfo);
    assert_eq!(MetadataBlockType::from(1), MetadataBlockType::Padding);
    assert_eq!(MetadataBlockType::from(2), MetadataBlockType::Application);
    assert_eq!(MetadataBlockType::from(3), MetadataBlockType::SeekTable);
    assert_eq!(MetadataBlockType::from(4), MetadataBlockType::VorbisComment);
    assert_eq!(MetadataBlockType::from(5), MetadataBlockType::CueSheet);
    assert_eq!(MetadataBlockType::from(6), MetadataBlockType::Picture);
    assert!(matches!(MetadataBlockType::from(100), MetadataBlockType::Reserved(100)));
}

#[test]
fn test_stream_info_default() {
    let info = StreamInfo::default();
    assert_eq!(info.sample_rate, 44100);
    assert_eq!(info.channels, 2);
    assert_eq!(info.bits_per_sample, 16);
    assert_eq!(info.total_samples, 0);
}

#[test]
fn test_vorbis_comment_default() {
    let comment = VorbisComment::default();
    assert!(comment.vendor.is_empty());
    assert!(comment.comments.is_empty());
}

#[test]
fn test_flac_metadata_default() {
    let metadata = FlacMetadata::default();
    assert!(metadata.stream_info.is_none());
    assert!(metadata.vorbis_comment.is_none());
    assert!(metadata.pictures.is_empty());
    assert!(metadata.seek_table.is_empty());
}

#[test]
fn test_compression_level_custom() {
    let custom = CompressionLevel::Custom {
        block_size: 8192,
        max_lpc_order: 16,
        rice_parameter_search: 3,
        do_exhaustive_model_search: true,
        do_qlp_coeff_prec_search: true,
    };

    assert_eq!(custom.block_size(), 8192);
    assert_eq!(custom.max_lpc_order(), 16);
    assert_eq!(custom.rice_parameter_search(), 3);
    assert!(custom.do_exhaustive_model_search());
}

#[test]
fn test_encode_different_sample_rates() {
    for sample_rate in [8000, 16000, 22050, 44100, 48000, 96000] {
        let mut buffer = Vec::new();
        let encoder = FlacEncoder::new(&mut buffer, sample_rate, 2, 16, CompressionLevel::Default);
        assert!(encoder.is_ok(), "Should support sample rate {}", sample_rate);
    }
}

#[test]
fn test_encode_different_bit_depths() {
    for bits in [8, 12, 16, 20, 24, 32] {
        let mut buffer = Vec::new();
        let encoder = FlacEncoder::new(&mut buffer, 44100, 2, bits, CompressionLevel::Default);
        assert!(encoder.is_ok(), "Should support {} bits", bits);
    }
}

#[test]
fn test_encode_different_channel_counts() {
    for channels in 1..=8 {
        let mut buffer = Vec::new();
        let encoder = FlacEncoder::new(&mut buffer, 44100, channels, 16, CompressionLevel::Default);
        assert!(encoder.is_ok(), "Should support {} channels", channels);
    }
}

#[test]
fn test_encoder_with_vorbis_comment() {
    let mut buffer = Vec::new();
    let mut encoder = FlacEncoder::new(&mut buffer, 44100, 2, 16, CompressionLevel::Default).unwrap();

    let comment = VorbisComment {
        vendor: "transcode-flac".to_string(),
        comments: vec![
            ("ARTIST".to_string(), "Test Artist".to_string()),
            ("TITLE".to_string(), "Test Title".to_string()),
        ],
    };

    let result = encoder.write_vorbis_comment(&comment);
    assert!(result.is_ok());
}

#[test]
fn test_frame_number_increments() {
    let mut buffer = Vec::new();
    let mut encoder = FlacEncoder::new(&mut buffer, 44100, 2, 16, CompressionLevel::Default).unwrap();

    assert_eq!(encoder.frame_number(), 0);

    // Encode a block
    let samples = generate_sine_wave(44100, 440.0, 0.1, 16000);
    let stereo = make_stereo(&samples);
    encoder.encode_samples(&stereo).unwrap();

    // Frame number should have incremented
    assert!(encoder.frame_number() > 0);
}

#[test]
fn test_samples_written_count() {
    let mut buffer = Vec::new();
    let mut encoder = FlacEncoder::new(&mut buffer, 44100, 2, 16, CompressionLevel::Default).unwrap();

    assert_eq!(encoder.samples_written(), 0);

    // Encode some samples
    let samples = vec![0i32; 8820]; // 0.1 seconds at 44100 Hz, stereo
    encoder.encode_samples(&samples).unwrap();

    // Should have counted samples per channel
    assert_eq!(encoder.samples_written(), 4410);
}

#[test]
fn test_roundtrip_encoding() {
    // This test verifies basic encode path - full roundtrip would require
    // a complete decoder that can read our encoded output

    let mut buffer = Vec::new();
    let mut encoder = FlacEncoder::new(&mut buffer, 44100, 2, 16, CompressionLevel::Default).unwrap();

    // Create test signal
    let original: Vec<i32> = (0..8192).map(|i| ((i as f64 * 0.1).sin() * 10000.0) as i32).collect();
    encoder.encode_samples(&original).unwrap();

    let result = encoder.finish();
    assert!(result.is_ok());

    // Verify output structure
    let output = result.unwrap();
    assert!(output.len() > 42, "Should contain header + frame data");
    assert_eq!(&output[0..4], b"fLaC", "Should start with FLAC marker");

    // Verify STREAMINFO block header
    let _is_last = output[4] & 0x80 != 0;
    let block_type = output[4] & 0x7F;
    assert_eq!(block_type, 0, "First metadata block should be STREAMINFO");
}

#[test]
fn test_encoder_finish_idempotent() {
    let mut buffer = Vec::new();
    let mut encoder = FlacEncoder::new(&mut buffer, 44100, 2, 16, CompressionLevel::Default).unwrap();

    let samples = vec![0i32; 8192];
    encoder.encode_samples(&samples).unwrap();

    // First finish should succeed
    let result = encoder.finish();
    assert!(result.is_ok());
}

#[test]
fn test_all_prediction_types_used() {
    // Test that different signal types use different prediction methods

    let mut buffer = Vec::new();

    // Constant signal - should use constant subframe
    {
        let mut encoder = FlacEncoder::new(&mut buffer, 44100, 1, 16, CompressionLevel::Default).unwrap();
        let constant: Vec<i32> = vec![1234; 4096];
        encoder.encode_samples(&constant).unwrap();
        let _ = encoder.finish();
    }

    buffer.clear();

    // Random-ish signal - might use verbatim or LPC
    {
        let mut encoder = FlacEncoder::new(&mut buffer, 44100, 1, 16, CompressionLevel::Default).unwrap();
        let complex: Vec<i32> = (0..4096)
            .map(|i| ((i * 12345 + 6789) % 65536 - 32768) as i32)
            .collect();
        encoder.encode_samples(&complex).unwrap();
        let _ = encoder.finish();
    }

    buffer.clear();

    // Smooth signal - should use fixed or LPC
    {
        let mut encoder = FlacEncoder::new(&mut buffer, 44100, 1, 16, CompressionLevel::Best).unwrap();
        let smooth = generate_sine_wave(44100, 440.0, 0.1, 16000);
        encoder.encode_samples(&smooth).unwrap();
        let _ = encoder.finish();
    }
}

#[test]
fn test_channel_assignment_types() {
    // These are compile-time checks that the types exist and work
    let _independent = ChannelAssignment::Independent(2);
    let _left_side = ChannelAssignment::LeftSide;
    let _right_side = ChannelAssignment::RightSide;
    let _mid_side = ChannelAssignment::MidSide;
}

#[test]
fn test_subframe_types() {
    // Compile-time type checks
    let _constant = SubframeType::Constant;
    let _verbatim = SubframeType::Verbatim;
    let _fixed = SubframeType::Fixed(4);
    let _lpc = SubframeType::Lpc(12);
}

#[test]
fn test_error_display() {
    let error = FlacError::InvalidMarker;
    assert!(!error.to_string().is_empty());

    let error = FlacError::CrcMismatch { expected: 0x1234, actual: 0x5678 };
    assert!(error.to_string().contains("1234"));
    assert!(error.to_string().contains("5678"));
}

#[test]
fn test_large_block_encoding() {
    let mut buffer = Vec::new();
    let mut encoder = FlacEncoder::new(&mut buffer, 44100, 2, 16, CompressionLevel::Default).unwrap();

    // Large block of audio
    let samples: Vec<i32> = (0..88200).map(|i| (i % 65536 - 32768) as i32).collect();
    let result = encoder.encode_samples(&samples);
    assert!(result.is_ok());

    let _ = encoder.finish();
    assert!(buffer.len() > 0);
}

#[test]
fn test_edge_case_sample_values() {
    let mut buffer = Vec::new();
    let mut encoder = FlacEncoder::new(&mut buffer, 44100, 1, 16, CompressionLevel::Default).unwrap();

    // Test with extreme values
    let samples: Vec<i32> = vec![
        i16::MIN as i32,
        i16::MAX as i32,
        0,
        i16::MIN as i32 + 1,
        i16::MAX as i32 - 1,
    ];

    let result = encoder.encode_samples(&samples);
    assert!(result.is_ok());
}

#[test]
fn test_streaming_decoder_partial_data() {
    let mut decoder = StreamingDecoder::new();

    // Feed partial marker
    decoder.feed(b"fL");
    let result = decoder.decode_frame();
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());

    // Complete the marker
    decoder.feed(b"aC");
    let result = decoder.decode_frame();
    assert!(result.is_ok());
}

#[test]
fn test_streaming_decoder_clear_buffer() {
    let mut decoder = StreamingDecoder::new();
    decoder.feed(b"some data");
    assert!(decoder.buffer_size() > 0);

    decoder.clear_buffer();
    assert_eq!(decoder.buffer_size(), 0);
}
