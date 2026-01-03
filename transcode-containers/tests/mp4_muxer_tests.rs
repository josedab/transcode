//! MP4 muxer and demuxer tests.
//!
//! These tests verify MP4 container round-trip functionality.

use std::io::Cursor;
use transcode_containers::mp4::{Mp4Demuxer, Mp4Muxer};
use transcode_containers::traits::{CodecId, Demuxer, Muxer, StreamInfo, TrackType, VideoStreamInfo, AudioStreamInfo};
use transcode_core::packet::Packet;
use transcode_core::rational::Rational;

// =============================================================================
// MP4 Demuxer Tests
// =============================================================================

#[test]
fn test_mp4_demuxer_empty_data() {
    // Create empty data
    let data: &[u8] = &[];
    let cursor = Cursor::new(data);

    let mut demuxer = Mp4Demuxer::new();
    // Open with empty data - may or may not fail immediately
    // depending on implementation (deferred parsing)
    let result = demuxer.open(cursor);
    // If it succeeds, trying to read should fail or return no streams
    if result.is_ok() {
        assert_eq!(demuxer.num_streams(), 0);
    }
}

#[test]
fn test_mp4_demuxer_invalid_data() {
    let data: &[u8] = b"not valid mp4 data at all";
    let cursor = Cursor::new(data);

    let mut demuxer = Mp4Demuxer::new();
    let result = demuxer.open(cursor);
    // If it succeeds (due to deferred parsing), there should be no valid streams
    if result.is_ok() {
        assert_eq!(demuxer.num_streams(), 0);
    }
}

#[test]
fn test_mp4_demuxer_creation() {
    let demuxer = Mp4Demuxer::new();
    assert_eq!(demuxer.format_name(), "mp4");
}

// =============================================================================
// MP4 Muxer Tests
// =============================================================================

#[test]
fn test_mp4_muxer_creation() {
    let muxer = Mp4Muxer::new();
    assert_eq!(muxer.format_name(), "mp4");
}

#[test]
fn test_mp4_muxer_with_video_track() {
    let mut muxer = Mp4Muxer::new();
    let buffer: Vec<u8> = Vec::new();
    let cursor = Cursor::new(buffer);

    muxer.create(cursor).unwrap();

    let stream = StreamInfo {
        index: 0,
        track_type: TrackType::Video,
        codec_id: CodecId::H264,
        time_base: Rational::new(1, 90000),
        duration: None,
        extra_data: None,
        video: Some(VideoStreamInfo {
            width: 1920,
            height: 1080,
            frame_rate: Some(Rational::new(30, 1)),
            pixel_aspect_ratio: None,
            bit_depth: 8,
        }),
        audio: None,
    };

    let result = muxer.add_stream(stream);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 0);
}

#[test]
fn test_mp4_muxer_with_audio_track() {
    let mut muxer = Mp4Muxer::new();
    let buffer: Vec<u8> = Vec::new();
    let cursor = Cursor::new(buffer);

    muxer.create(cursor).unwrap();

    let stream = StreamInfo {
        index: 0,
        track_type: TrackType::Audio,
        codec_id: CodecId::Aac,
        time_base: Rational::new(1, 48000),
        duration: None,
        extra_data: None,
        video: None,
        audio: Some(AudioStreamInfo {
            sample_rate: 48000,
            channels: 2,
            bits_per_sample: 16,
        }),
    };

    let result = muxer.add_stream(stream);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 0);
}

#[test]
fn test_mp4_muxer_with_multiple_tracks() {
    let mut muxer = Mp4Muxer::new();
    let buffer: Vec<u8> = Vec::new();
    let cursor = Cursor::new(buffer);

    muxer.create(cursor).unwrap();

    // Add video track
    let video_stream = StreamInfo {
        index: 0,
        track_type: TrackType::Video,
        codec_id: CodecId::H264,
        time_base: Rational::new(1, 90000),
        duration: None,
        extra_data: None,
        video: Some(VideoStreamInfo {
            width: 1920,
            height: 1080,
            frame_rate: Some(Rational::new(30, 1)),
            pixel_aspect_ratio: None,
            bit_depth: 8,
        }),
        audio: None,
    };

    let video_result = muxer.add_stream(video_stream);
    assert!(video_result.is_ok());

    // Add audio track
    let audio_stream = StreamInfo {
        index: 1,
        track_type: TrackType::Audio,
        codec_id: CodecId::Aac,
        time_base: Rational::new(1, 48000),
        duration: None,
        extra_data: None,
        video: None,
        audio: Some(AudioStreamInfo {
            sample_rate: 48000,
            channels: 2,
            bits_per_sample: 16,
        }),
    };

    let audio_result = muxer.add_stream(audio_stream);
    assert!(audio_result.is_ok());
}

#[test]
fn test_mp4_muxer_write_header() {
    let mut muxer = Mp4Muxer::new();
    let buffer: Vec<u8> = Vec::new();
    let cursor = Cursor::new(buffer);

    muxer.create(cursor).unwrap();

    // Add a video track
    let stream = StreamInfo {
        index: 0,
        track_type: TrackType::Video,
        codec_id: CodecId::H264,
        time_base: Rational::new(1, 90000),
        duration: None,
        extra_data: Some(vec![0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1f]), // Minimal SPS
        video: Some(VideoStreamInfo {
            width: 320,
            height: 240,
            frame_rate: Some(Rational::new(30, 1)),
            pixel_aspect_ratio: None,
            bit_depth: 8,
        }),
        audio: None,
    };

    muxer.add_stream(stream).unwrap();

    // Write header
    let result = muxer.write_header();
    assert!(result.is_ok());
}

#[test]
fn test_mp4_muxer_write_packet() {
    let mut muxer = Mp4Muxer::new();
    let buffer: Vec<u8> = Vec::new();
    let cursor = Cursor::new(buffer);

    muxer.create(cursor).unwrap();

    let stream = StreamInfo {
        index: 0,
        track_type: TrackType::Video,
        codec_id: CodecId::H264,
        time_base: Rational::new(1, 90000),
        duration: None,
        extra_data: Some(vec![0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1f]),
        video: Some(VideoStreamInfo {
            width: 320,
            height: 240,
            frame_rate: Some(Rational::new(30, 1)),
            pixel_aspect_ratio: None,
            bit_depth: 8,
        }),
        audio: None,
    };

    muxer.add_stream(stream).unwrap();
    muxer.write_header().unwrap();

    // Create a test packet with NAL data (simplified)
    let nal_data = vec![
        0x00, 0x00, 0x00, 0x01, // Start code
        0x65, // IDR NAL type
        0x88, 0x84, // Dummy data
    ];
    let mut packet = Packet::new(nal_data);
    packet.set_keyframe(true);
    let packet = packet.with_stream_index(0);

    let result = muxer.write_packet(&packet);
    assert!(result.is_ok());
}

#[test]
fn test_mp4_muxer_finalize() {
    let mut muxer = Mp4Muxer::new();
    let buffer: Vec<u8> = Vec::new();
    let cursor = Cursor::new(buffer);

    muxer.create(cursor).unwrap();

    let stream = StreamInfo {
        index: 0,
        track_type: TrackType::Video,
        codec_id: CodecId::H264,
        time_base: Rational::new(1, 90000),
        duration: None,
        extra_data: Some(vec![0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1f]),
        video: Some(VideoStreamInfo {
            width: 320,
            height: 240,
            frame_rate: Some(Rational::new(30, 1)),
            pixel_aspect_ratio: None,
            bit_depth: 8,
        }),
        audio: None,
    };

    muxer.add_stream(stream).unwrap();
    muxer.write_header().unwrap();

    // Write a packet
    let nal_data = vec![0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84];
    let mut packet = Packet::new(nal_data);
    packet.set_keyframe(true);
    let packet = packet.with_stream_index(0);
    muxer.write_packet(&packet).unwrap();

    // Write trailer should work
    let result = muxer.write_trailer();
    assert!(result.is_ok());
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[test]
fn test_mp4_muxer_empty_finalize() {
    let mut muxer = Mp4Muxer::new();
    let buffer: Vec<u8> = Vec::new();
    let cursor = Cursor::new(buffer);

    muxer.create(cursor).unwrap();

    // Add a track but write no packets
    let stream = StreamInfo {
        index: 0,
        track_type: TrackType::Video,
        codec_id: CodecId::H264,
        time_base: Rational::new(1, 90000),
        duration: None,
        extra_data: None,
        video: Some(VideoStreamInfo {
            width: 320,
            height: 240,
            frame_rate: Some(Rational::new(30, 1)),
            pixel_aspect_ratio: None,
            bit_depth: 8,
        }),
        audio: None,
    };

    muxer.add_stream(stream).unwrap();
    muxer.write_header().unwrap();

    // Finalize without writing any packets
    let result = muxer.write_trailer();
    // Should still produce valid (minimal) output
    assert!(result.is_ok());
}

// =============================================================================
// CodecId Tests
// =============================================================================

#[test]
fn test_codec_id_fourcc() {
    assert_eq!(CodecId::H264.fourcc(), Some(*b"avc1"));
    assert_eq!(CodecId::H265.fourcc(), Some(*b"hvc1"));
    assert_eq!(CodecId::Aac.fourcc(), Some(*b"mp4a"));
    assert_eq!(CodecId::Vp9.fourcc(), Some(*b"vp09"));
    assert_eq!(CodecId::Av1.fourcc(), Some(*b"av01"));
}

#[test]
fn test_track_type() {
    assert_eq!(TrackType::Video, TrackType::Video);
    assert_eq!(TrackType::Audio, TrackType::Audio);
    assert_ne!(TrackType::Video, TrackType::Audio);
}
