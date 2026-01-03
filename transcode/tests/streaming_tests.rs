//! Streaming integration tests.
//!
//! Tests for HLS and DASH streaming output.

use std::fs;
use tempfile::tempdir;
use transcode_streaming::{
    HlsConfig, HlsWriter, MasterPlaylist, MediaPlaylist, VariantStream,
    Quality, Segment, SegmentNaming, SegmentType, LowLatencyConfig,
};

/// Test HLS configuration validation.
#[test]
fn test_hls_config_validation() {
    // Valid config
    let config = HlsConfig::new("/tmp/hls")
        .with_segment_duration(6.0)
        .with_quality(Quality::hd_720p());

    assert!(config.validate().is_ok());

    // Config with no qualities should fail
    let config_no_quality = HlsConfig {
        output_dir: "/tmp/hls".into(),
        segment_duration: 6.0,
        qualities: vec![], // No qualities
        naming: SegmentNaming::Sequential,
        low_latency: false,
        part_duration: 0.2,
        max_playlist_entries: 0,
        program_date_time: false,
        independent_segments: true,
        custom_headers: vec![],
        ll_hls_config: LowLatencyConfig::default(),
    };

    assert!(config_no_quality.validate().is_err());
}

/// Test master playlist generation.
#[test]
fn test_master_playlist_generation() {
    let mut master = MasterPlaylist::new();

    master.add_variant(VariantStream {
        bandwidth: 5_500_000,
        average_bandwidth: Some(5_000_000),
        resolution: "1920x1080".to_string(),
        frame_rate: 30.0,
        codecs: "avc1.64001f,mp4a.40.2".to_string(),
        uri: "1080p/playlist.m3u8".to_string(),
        audio_group: None,
    });

    master.add_variant(VariantStream {
        bandwidth: 2_500_000,
        average_bandwidth: Some(2_000_000),
        resolution: "1280x720".to_string(),
        frame_rate: 30.0,
        codecs: "avc1.64001e,mp4a.40.2".to_string(),
        uri: "720p/playlist.m3u8".to_string(),
        audio_group: None,
    });

    let content = master.render();

    // Check required M3U8 tags
    assert!(content.contains("#EXTM3U"));
    assert!(content.contains("#EXT-X-VERSION:7"));
    assert!(content.contains("#EXT-X-STREAM-INF"));
    assert!(content.contains("BANDWIDTH=5500000"));
    assert!(content.contains("RESOLUTION=1920x1080"));
    assert!(content.contains("1080p/playlist.m3u8"));
}

/// Test media playlist generation.
#[test]
fn test_media_playlist_generation() {
    let mut playlist = MediaPlaylist::new(6);

    playlist.add_segment(Segment::new(0, SegmentType::Media, 6.0, 0.0, "seg0.ts", "1080p"));
    playlist.add_segment(Segment::new(1, SegmentType::Media, 6.0, 6.0, "seg1.ts", "1080p"));
    playlist.add_segment(Segment::new(2, SegmentType::Media, 5.5, 12.0, "seg2.ts", "1080p"));
    playlist.end();

    let content = playlist.render();

    // Check required M3U8 tags
    assert!(content.contains("#EXTM3U"));
    assert!(content.contains("#EXT-X-TARGETDURATION:6"));
    assert!(content.contains("#EXT-X-MEDIA-SEQUENCE:0"));
    assert!(content.contains("#EXTINF:6.000000"));
    assert!(content.contains("#EXT-X-ENDLIST"));
    assert!(content.contains("seg0.ts"));
    assert!(content.contains("seg1.ts"));
}

/// Test HLS writer creates directory structure.
#[test]
fn test_hls_writer_creates_directories() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let output_path = temp_dir.path().join("hls_output");

    let config = HlsConfig::new(&output_path)
        .with_qualities(vec![Quality::hd_720p(), Quality::fhd_1080p()]);

    let _writer = HlsWriter::new(config).expect("Should create HLS writer");

    // Check directories were created
    assert!(output_path.exists());
    assert!(output_path.join("720p").exists());
    assert!(output_path.join("1080p").exists());
}

/// Test HLS writer segment writing.
#[test]
fn test_hls_writer_segment_writing() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let output_path = temp_dir.path().join("hls_output");

    let config = HlsConfig::new(&output_path)
        .with_qualities(vec![Quality::hd_720p()])
        .with_segment_duration(6.0);

    let mut writer = HlsWriter::new(config).expect("Should create HLS writer");

    // Write a segment
    let segment_data = vec![0u8; 1024];
    let segment = writer
        .write_segment(0, &segment_data, 6.0, true)
        .expect("Should write segment");

    assert_eq!(segment.sequence, 0);
    assert_eq!(segment.duration, 6.0);
    assert!(segment.keyframe);

    // Write another segment
    let segment2 = writer
        .write_segment(0, &segment_data, 6.0, false)
        .expect("Should write segment");

    assert_eq!(segment2.sequence, 1);

    // Check segment count
    assert_eq!(writer.segment_count(0), 2);
    assert_eq!(writer.total_duration(), 12.0);
}

/// Test HLS writer finalization.
#[test]
fn test_hls_writer_finalization() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let output_path = temp_dir.path().join("hls_output");

    let config = HlsConfig::new(&output_path)
        .with_qualities(vec![Quality::hd_720p()]);

    let mut writer = HlsWriter::new(config).expect("Should create HLS writer");

    // Write some segments
    let segment_data = vec![0u8; 1024];
    writer.write_segment(0, &segment_data, 6.0, true).unwrap();
    writer.write_segment(0, &segment_data, 6.0, false).unwrap();

    // Finalize
    writer.finalize().expect("Should finalize");

    // Check that master playlist was created
    let master_path = output_path.join("master.m3u8");
    assert!(master_path.exists(), "Master playlist should exist");

    // Check that media playlist was created
    let media_path = output_path.join("720p").join("playlist.m3u8");
    assert!(media_path.exists(), "Media playlist should exist");

    // Check master playlist content
    let master_content = fs::read_to_string(master_path).expect("Should read master playlist");
    assert!(master_content.contains("#EXTM3U"));
    assert!(master_content.contains("720p/playlist.m3u8"));

    // Check media playlist content
    let media_content = fs::read_to_string(media_path).expect("Should read media playlist");
    assert!(media_content.contains("#EXT-X-ENDLIST"));
}

/// Test segment naming strategies.
#[test]
fn test_segment_naming() {
    // Sequential naming
    let seq = SegmentNaming::Sequential;
    assert_eq!(seq.generate(0, "ts"), "segment_00000.ts");
    assert_eq!(seq.generate(1, "ts"), "segment_00001.ts");
    assert_eq!(seq.generate(999, "ts"), "segment_00999.ts");

    // TimeBased naming
    let tb = SegmentNaming::TimeBased;
    let name = tb.generate(100, "ts");
    assert!(name.ends_with(".ts"));
    assert_eq!(name, "segment_00000100.ts");
}

/// Test quality presets.
#[test]
fn test_quality_presets() {
    let q_480p = Quality::sd_480p();
    assert_eq!(q_480p.width, 854);
    assert_eq!(q_480p.height, 480);

    let q_720p = Quality::hd_720p();
    assert_eq!(q_720p.width, 1280);
    assert_eq!(q_720p.height, 720);

    let q_1080p = Quality::fhd_1080p();
    assert_eq!(q_1080p.width, 1920);
    assert_eq!(q_1080p.height, 1080);

    let q_4k = Quality::uhd_4k();
    assert_eq!(q_4k.width, 3840);
    assert_eq!(q_4k.height, 2160);

    // Check resolution strings
    assert_eq!(q_1080p.resolution_string(), "1920x1080");
}

/// Test low-latency HLS configuration.
#[test]
fn test_low_latency_hls_config() {
    let config = HlsConfig::new("/tmp/llhls")
        .with_low_latency(0.2)
        .with_qualities(vec![Quality::hd_720p()]);

    assert!(config.low_latency);
    assert_eq!(config.part_duration, 0.2);
    assert!(config.validate().is_ok());
}

/// Test init segment writing.
#[test]
fn test_init_segment_writing() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let output_path = temp_dir.path().join("fmp4_output");

    let config = HlsConfig::new(&output_path)
        .with_qualities(vec![Quality::hd_720p()]);

    let mut writer = HlsWriter::new(config).expect("Should create HLS writer");

    // Write init segment
    let init_data = vec![0u8; 512];
    writer.write_init_segment(0, &init_data).expect("Should write init segment");

    // Check init segment was created
    let init_path = output_path.join("720p").join("init.mp4");
    assert!(init_path.exists(), "Init segment should exist");
}
