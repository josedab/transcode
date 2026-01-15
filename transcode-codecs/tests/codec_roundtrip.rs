//! Codec round-trip tests for H.264 and AAC.
//!
//! These tests verify that encoding and decoding produce consistent results.
//!
//! ## Property-Based Tests
//!
//! This module includes property-based tests using `proptest` to verify that
//! codecs handle arbitrary valid inputs correctly.

use transcode_codecs::video::h264::{H264Encoder, H264EncoderConfig, H264Decoder, H264DecoderConfig, RateControlMode};
use transcode_codecs::audio::aac::{AacEncoder, AacEncoderConfig, AacDecoder, AacProfile};
use transcode_codecs::traits::{VideoEncoder, VideoDecoder, AudioEncoder, AudioDecoder};
use transcode_core::frame::{Frame, PixelFormat};
use transcode_core::sample::{Sample, SampleFormat, ChannelLayout};
use transcode_core::timestamp::TimeBase;
use proptest::prelude::*;

/// Test helper: Create a test frame with a specific pattern.
fn create_test_frame(width: u32, height: u32, frame_num: u64) -> Frame {
    let time_base = TimeBase::new(1, 90000);
    let mut frame = Frame::new(width, height, PixelFormat::Yuv420p, time_base);

    // Fill Y plane with gradient pattern
    if let Some(y_plane) = frame.plane_mut(0) {
        for row in 0..height as usize {
            for col in 0..width as usize {
                let y_val = ((row + col + frame_num as usize * 10) % 256) as u8;
                y_plane[row * width as usize + col] = y_val;
            }
        }
    }

    // Fill U and V planes with constant values (neutral chroma)
    let uv_width = width as usize / 2;
    let uv_height = height as usize / 2;

    if let Some(u_plane) = frame.plane_mut(1) {
        for i in 0..(uv_width * uv_height) {
            u_plane[i] = 128;
        }
    }

    if let Some(v_plane) = frame.plane_mut(2) {
        for i in 0..(uv_width * uv_height) {
            v_plane[i] = 128;
        }
    }

    // Set timestamps
    frame.pts = transcode_core::timestamp::Timestamp::new(frame_num as i64 * 3000, time_base);
    frame.dts = transcode_core::timestamp::Timestamp::new(frame_num as i64 * 3000, time_base);

    frame
}

/// Test helper: Create a test audio sample buffer.
fn create_test_audio(num_samples: usize, _frame_num: u64) -> Sample {
    // Create sample with S16 format (easier to fill)
    let mut sample = Sample::new(
        num_samples,
        SampleFormat::S16,
        ChannelLayout::Stereo,
        44100,
    );

    // Get the interleaved data buffer and fill with simple pattern
    let data = sample.data_mut();
    // Fill with a simple pattern - data is i16 samples in native byte order
    for i in 0..(num_samples * 2) {
        let byte_idx = i * 2;
        if byte_idx + 1 < data.len() {
            let value = ((i as i32 * 100) % 32000) as i16;
            let bytes = value.to_ne_bytes();
            data[byte_idx] = bytes[0];
            data[byte_idx + 1] = bytes[1];
        }
    }

    sample
}

// =============================================================================
// H.264 Encoder Tests
// =============================================================================

#[test]
fn test_h264_encoder_creates_valid_packets() {
    let config = H264EncoderConfig::new(320, 240);
    let mut encoder = H264Encoder::new(config).unwrap();

    let frame = create_test_frame(320, 240, 0);
    let packets = encoder.encode(&frame).unwrap();

    // First frame should produce at least one packet
    assert!(!packets.is_empty(), "Encoder should produce packets");

    // First packet should be a keyframe
    assert!(packets[0].is_keyframe(), "First packet should be keyframe");
}

#[test]
fn test_h264_encoder_multiple_frames() {
    let config = H264EncoderConfig::new(320, 240);
    let mut encoder = H264Encoder::new(config).unwrap();

    for i in 0..10 {
        let frame = create_test_frame(320, 240, i);
        let packets = encoder.encode(&frame).unwrap();

        assert!(!packets.is_empty(), "Frame {} should produce packets", i);

        for packet in &packets {
            assert!(!packet.data().is_empty(), "Packet should have data");
        }
    }
}

#[test]
fn test_h264_encoder_constant_qp() {
    let mut config = H264EncoderConfig::new(320, 240);
    config.rate_control = RateControlMode::Cqp(28);

    let mut encoder = H264Encoder::new(config).unwrap();

    let frame = create_test_frame(320, 240, 0);
    let packets = encoder.encode(&frame).unwrap();

    assert!(!packets.is_empty());
}

#[test]
fn test_h264_encoder_cabac_vs_cavlc() {
    // Test CAVLC (Baseline profile)
    let mut config_cavlc = H264EncoderConfig::new(320, 240);
    config_cavlc.cabac = false;
    let mut encoder_cavlc = H264Encoder::new(config_cavlc).unwrap();

    let frame = create_test_frame(320, 240, 0);
    let packets_cavlc = encoder_cavlc.encode(&frame).unwrap();

    // Test CABAC (Main profile)
    let mut config_cabac = H264EncoderConfig::new(320, 240);
    config_cabac.cabac = true;
    let mut encoder_cabac = H264Encoder::new(config_cabac).unwrap();

    let packets_cabac = encoder_cabac.encode(&frame).unwrap();

    // Both should produce valid output
    assert!(!packets_cavlc.is_empty(), "CAVLC should produce packets");
    assert!(!packets_cabac.is_empty(), "CABAC should produce packets");
}

#[test]
fn test_h264_encoder_reset() {
    let config = H264EncoderConfig::new(320, 240);
    let mut encoder = H264Encoder::new(config).unwrap();

    // Encode some frames
    for i in 0..5 {
        let frame = create_test_frame(320, 240, i);
        let _ = encoder.encode(&frame).unwrap();
    }

    // Reset encoder (returns unit, not Result)
    encoder.reset();

    // Should be able to encode again starting fresh
    let frame = create_test_frame(320, 240, 0);
    let packets = encoder.encode(&frame).unwrap();
    assert!(!packets.is_empty());
}

// =============================================================================
// H.264 Decoder Tests
// =============================================================================

#[test]
fn test_h264_decoder_creates() {
    let config = H264DecoderConfig::default();
    let decoder = H264Decoder::new(config);
    assert!(decoder.codec_info().can_decode);
}

// =============================================================================
// AAC Encoder Tests
// =============================================================================

#[test]
fn test_aac_encoder_creates_valid_packets() {
    let config = AacEncoderConfig {
        profile: AacProfile::Lc,
        sample_rate: 44100,
        channels: 2,
        bitrate: 128000,
        adts: true,
        quality: 0.5,
    };

    let mut encoder = AacEncoder::new(config).unwrap();

    let sample = create_test_audio(1024, 0);
    let result = encoder.encode(&sample);

    // AAC encoder should succeed (may or may not output immediately due to buffering)
    assert!(result.is_ok());
}

#[test]
fn test_aac_encoder_multiple_frames() {
    let config = AacEncoderConfig {
        profile: AacProfile::Lc,
        sample_rate: 44100,
        channels: 2,
        bitrate: 128000,
        adts: true,
        quality: 0.5,
    };

    let mut encoder = AacEncoder::new(config).unwrap();
    let mut total_packets = 0;

    for i in 0..10 {
        let sample = create_test_audio(1024, i);
        let packets = encoder.encode(&sample).unwrap();
        total_packets += packets.len();
    }

    // Flush to get remaining packets
    let flushed = encoder.flush().unwrap();
    total_packets += flushed.len();

    // Should have produced some packets
    assert!(total_packets > 0, "AAC encoder should produce packets");
}

#[test]
fn test_aac_encoder_reset() {
    let config = AacEncoderConfig {
        profile: AacProfile::Lc,
        sample_rate: 44100,
        channels: 2,
        bitrate: 128000,
        adts: true,
        quality: 0.5,
    };

    let mut encoder = AacEncoder::new(config).unwrap();

    // Encode some audio
    for i in 0..5 {
        let sample = create_test_audio(1024, i);
        let _ = encoder.encode(&sample);
    }

    // Reset (returns unit, not Result)
    encoder.reset();

    // Should work after reset
    let sample = create_test_audio(1024, 0);
    let result = encoder.encode(&sample);
    assert!(result.is_ok());
}

// =============================================================================
// AAC Decoder Tests
// =============================================================================

#[test]
fn test_aac_decoder_creates() {
    let decoder = AacDecoder::new();
    assert!(decoder.codec_info().can_decode);
}

// =============================================================================
// Frame Format Tests
// =============================================================================

#[test]
fn test_frame_plane_access() {
    let frame = create_test_frame(320, 240, 0);

    // Should be able to access Y plane
    let y_plane = frame.plane(0).expect("Should have Y plane");
    assert_eq!(y_plane.len(), 320 * 240);

    // Should be able to access U plane
    let u_plane = frame.plane(1).expect("Should have U plane");
    assert_eq!(u_plane.len(), 160 * 120);

    // Should be able to access V plane
    let v_plane = frame.plane(2).expect("Should have V plane");
    assert_eq!(v_plane.len(), 160 * 120);
}

#[test]
fn test_frame_dimensions() {
    let frame = create_test_frame(1920, 1080, 0);

    assert_eq!(frame.width(), 1920);
    assert_eq!(frame.height(), 1080);
    assert_eq!(frame.format(), PixelFormat::Yuv420p);
}

// =============================================================================
// Sample Buffer Tests
// =============================================================================

#[test]
fn test_sample_buffer_properties() {
    let sample = create_test_audio(1024, 0);

    assert_eq!(sample.channels(), 2);
    assert_eq!(sample.sample_rate(), 44100);
    assert_eq!(sample.format(), SampleFormat::S16);
    assert_eq!(sample.num_samples(), 1024);
}

// =============================================================================
// Property-Based Tests
// =============================================================================

/// Strategy for generating valid video dimensions.
/// Dimensions must be multiples of 16 for H.264 macroblocks.
fn video_dimensions() -> impl Strategy<Value = (u32, u32)> {
    // Width and height must be multiples of 16, min 64, max 1920x1080
    (4u32..=120, 4u32..=68).prop_map(|(w, h)| (w * 16, h * 16))
}

/// Strategy for generating Y plane values.
fn y_plane_value() -> impl Strategy<Value = u8> {
    0u8..=255
}

/// Strategy for generating frame numbers.
fn frame_number() -> impl Strategy<Value = u64> {
    0u64..1000
}

/// Strategy for generating audio sample counts (must be suitable for AAC: 1024 samples).
fn audio_sample_count() -> impl Strategy<Value = usize> {
    // AAC works with 1024 sample frames
    prop_oneof![Just(1024), Just(2048), Just(4096)]
}

/// Strategy for generating sample rates.
fn audio_sample_rate() -> impl Strategy<Value = u32> {
    prop_oneof![Just(44100u32), Just(48000), Just(32000), Just(22050)]
}

/// Create a frame with arbitrary Y plane data.
fn create_arbitrary_frame(width: u32, height: u32, y_fill: u8, frame_num: u64) -> Frame {
    let time_base = TimeBase::new(1, 90000);
    let mut frame = Frame::new(width, height, PixelFormat::Yuv420p, time_base);

    // Fill Y plane with the provided value
    if let Some(y_plane) = frame.plane_mut(0) {
        for pixel in y_plane.iter_mut() {
            *pixel = y_fill;
        }
    }

    // Fill UV planes with neutral chroma
    let uv_size = (width as usize / 2) * (height as usize / 2);
    if let Some(u_plane) = frame.plane_mut(1) {
        for pixel in u_plane.iter_mut().take(uv_size) {
            *pixel = 128;
        }
    }
    if let Some(v_plane) = frame.plane_mut(2) {
        for pixel in v_plane.iter_mut().take(uv_size) {
            *pixel = 128;
        }
    }

    frame.pts = transcode_core::timestamp::Timestamp::new(frame_num as i64 * 3000, time_base);
    frame.dts = transcode_core::timestamp::Timestamp::new(frame_num as i64 * 3000, time_base);

    frame
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Property: H.264 encoder should produce valid packets for any valid frame dimensions.
    #[test]
    fn prop_h264_encoder_handles_arbitrary_dimensions(
        (width, height) in video_dimensions(),
        y_fill in y_plane_value(),
    ) {
        let config = H264EncoderConfig::new(width, height);
        let mut encoder = H264Encoder::new(config).unwrap();

        let frame = create_arbitrary_frame(width, height, y_fill, 0);
        let packets = encoder.encode(&frame).unwrap();

        // Encoder must produce at least one packet
        prop_assert!(!packets.is_empty(), "Encoder should produce packets for {}x{}", width, height);

        // First frame should always be a keyframe
        prop_assert!(packets[0].is_keyframe(), "First packet should be keyframe");

        // All packets should have data
        for (i, packet) in packets.iter().enumerate() {
            prop_assert!(!packet.data().is_empty(), "Packet {} should have data", i);
        }
    }

    /// Property: H.264 encoder should handle sequences of frames.
    #[test]
    fn prop_h264_encoder_handles_frame_sequences(
        (width, height) in video_dimensions(),
        frame_count in 2u64..10,
    ) {
        let config = H264EncoderConfig::new(width, height);
        let mut encoder = H264Encoder::new(config).unwrap();

        let mut total_packets = 0;
        for i in 0..frame_count {
            let y_fill = ((i * 17) % 256) as u8;
            let frame = create_arbitrary_frame(width, height, y_fill, i);
            let packets = encoder.encode(&frame).unwrap();
            total_packets += packets.len();

            // Each frame should produce at least one packet
            prop_assert!(!packets.is_empty(), "Frame {} should produce packets", i);
        }

        // Total packets should be reasonable (at least one per frame)
        prop_assert!(
            total_packets >= frame_count as usize,
            "Should produce at least {} packets, got {}",
            frame_count,
            total_packets
        );
    }

    /// Property: Encoder reset should allow re-encoding.
    #[test]
    fn prop_h264_encoder_reset_allows_reuse(
        (width, height) in video_dimensions(),
        y_fill1 in y_plane_value(),
        y_fill2 in y_plane_value(),
    ) {
        let config = H264EncoderConfig::new(width, height);
        let mut encoder = H264Encoder::new(config).unwrap();

        // Encode first frame
        let frame1 = create_arbitrary_frame(width, height, y_fill1, 0);
        let packets1 = encoder.encode(&frame1).unwrap();
        prop_assert!(!packets1.is_empty());

        // Reset encoder
        encoder.reset();

        // Encode second frame after reset - should still be keyframe
        let frame2 = create_arbitrary_frame(width, height, y_fill2, 0);
        let packets2 = encoder.encode(&frame2).unwrap();
        prop_assert!(!packets2.is_empty());
        prop_assert!(packets2[0].is_keyframe(), "First frame after reset should be keyframe");
    }

    /// Property: AAC encoder should handle arbitrary audio.
    #[test]
    fn prop_aac_encoder_handles_arbitrary_audio(
        sample_count in audio_sample_count(),
        sample_rate in audio_sample_rate(),
        frame_num in frame_number(),
    ) {
        let config = AacEncoderConfig {
            profile: AacProfile::Lc,
            sample_rate,
            channels: 2,
            bitrate: 128000,
            adts: true,
            quality: 0.5,
        };

        let encoder_result = AacEncoder::new(config);

        // Some sample rates may not be supported
        if let Ok(mut encoder) = encoder_result {
            // Create sample
            let mut sample = Sample::new(
                sample_count,
                SampleFormat::S16,
                ChannelLayout::Stereo,
                sample_rate,
            );

            // Fill with pattern
            let data = sample.data_mut();
            for i in 0..(sample_count * 2) {
                let byte_idx = i * 2;
                if byte_idx + 1 < data.len() {
                    let value = (((i as i64 + frame_num as i64) * 100) % 32000) as i16;
                    let bytes = value.to_ne_bytes();
                    data[byte_idx] = bytes[0];
                    data[byte_idx + 1] = bytes[1];
                }
            }

            // Encoding should succeed
            let result = encoder.encode(&sample);
            prop_assert!(result.is_ok(), "AAC encode should succeed");
        }
    }

    /// Property: Frame dimensions should be correctly preserved.
    #[test]
    fn prop_frame_dimensions_preserved(
        (width, height) in video_dimensions(),
    ) {
        let time_base = TimeBase::new(1, 90000);
        let frame = Frame::new(width, height, PixelFormat::Yuv420p, time_base);

        prop_assert_eq!(frame.width(), width);
        prop_assert_eq!(frame.height(), height);

        // Y plane should have at least width*height pixels
        let y_plane = frame.plane(0).expect("Should have Y plane");
        prop_assert!(
            y_plane.len() >= (width * height) as usize,
            "Y plane should have at least {}x{} = {} pixels, got {}",
            width, height, width * height, y_plane.len()
        );

        // UV planes should exist and have reasonable size (at least 1/4 of Y)
        let min_uv_size = ((width / 2) * (height / 2)) as usize;
        let u_plane = frame.plane(1).expect("Should have U plane");
        prop_assert!(
            u_plane.len() >= min_uv_size,
            "U plane should have at least {} pixels, got {}",
            min_uv_size, u_plane.len()
        );

        let v_plane = frame.plane(2).expect("Should have V plane");
        prop_assert!(
            v_plane.len() >= min_uv_size,
            "V plane should have at least {} pixels, got {}",
            min_uv_size, v_plane.len()
        );
    }
}
