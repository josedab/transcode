//! Container format integration tests.
//!
//! Tests for MP4, MKV, and other container formats.

use transcode_core::packet::{Packet, PacketFlags};
use transcode_core::timestamp::{TimeBase, Timestamp, Duration};

/// Test packet creation and manipulation.
#[test]
fn test_packet_creation() {
    let data = vec![0u8, 1, 2, 3, 4];
    let packet = Packet::new(data);

    assert_eq!(packet.size(), 5);
    assert_eq!(packet.data(), &[0, 1, 2, 3, 4]);
    assert!(!packet.is_keyframe());
}

/// Test packet from slice (zero-copy).
#[test]
fn test_packet_from_slice() {
    let data = [1u8, 2, 3, 4, 5];
    let packet = Packet::from_slice(&data);

    assert_eq!(packet.size(), 5);
    assert_eq!(packet.data(), &[1, 2, 3, 4, 5]);
}

/// Test packet into owned conversion.
#[test]
fn test_packet_into_owned() {
    let data = [1u8, 2, 3];
    let packet = Packet::from_slice(&data);
    let owned = packet.into_owned();

    assert_eq!(owned.data(), &[1, 2, 3]);
}

/// Test packet keyframe flag.
#[test]
fn test_packet_keyframe() {
    let mut packet = Packet::new(vec![0u8; 10]);

    assert!(!packet.is_keyframe());

    packet.set_keyframe(true);
    assert!(packet.is_keyframe());

    packet.set_keyframe(false);
    assert!(!packet.is_keyframe());
}

/// Test packet flags.
#[test]
fn test_packet_flags() {
    let packet = Packet::new(vec![0u8; 10])
        .with_flags(PacketFlags::KEYFRAME | PacketFlags::DISPOSABLE);

    assert!(packet.is_keyframe());
    assert!(packet.flags.contains(PacketFlags::DISPOSABLE));
}

/// Test packet timestamps.
#[test]
fn test_packet_timestamps() {
    let time_base = TimeBase::new(1, 30);
    let pts = Timestamp::new(100, time_base);
    let dts = Timestamp::new(90, time_base);

    let packet = Packet::new(vec![0u8; 10])
        .with_timestamps(pts, dts);

    assert_eq!(packet.pts.value, 100);
    assert_eq!(packet.dts.value, 90);
}

/// Test packet stream index.
#[test]
fn test_packet_stream_index() {
    let packet = Packet::new(vec![0u8; 10])
        .with_stream_index(2);

    assert_eq!(packet.stream_index, 2);
}

/// Test empty packet.
#[test]
fn test_empty_packet() {
    let packet = Packet::empty();

    assert!(packet.is_empty());
    assert_eq!(packet.size(), 0);
}

/// Test TimeBase operations.
#[test]
fn test_time_base() {
    let tb = TimeBase::new(1, 30);

    assert_eq!(tb.0.num, 1);
    assert_eq!(tb.0.den, 30);
}

/// Test common time bases.
#[test]
fn test_common_time_bases() {
    let mpeg = TimeBase::MPEG;
    assert_eq!(mpeg.0.num, 1);
    assert_eq!(mpeg.0.den, 90000);

    let ms = TimeBase::MILLISECONDS;
    assert_eq!(ms.0.num, 1);
    assert_eq!(ms.0.den, 1000);
}

/// Test Timestamp creation and conversion.
#[test]
fn test_timestamp() {
    let tb = TimeBase::new(1, 30);
    let ts = Timestamp::new(90, tb);

    assert_eq!(ts.value, 90);
    assert_eq!(ts.to_seconds().unwrap(), 3.0); // 90 frames at 30fps = 3 seconds
}

/// Test Timestamp rescaling.
#[test]
fn test_timestamp_rescale() {
    let source_tb = TimeBase::new(1, 30);
    let target_tb = TimeBase::new(1, 60);

    let ts = Timestamp::new(30, source_tb);
    let rescaled = ts.rescale(target_tb);

    // 30 frames at 30fps = 60 frames at 60fps
    assert_eq!(rescaled.value, 60);
}

/// Test Duration creation.
#[test]
fn test_duration() {
    let tb = TimeBase::new(1, 1000); // milliseconds
    let dur = Duration::new(1000, tb); // 1 second

    assert_eq!(dur.to_seconds(), 1.0);
}

/// Test Duration rescaling.
#[test]
fn test_duration_rescale() {
    let source_tb = TimeBase::new(1, 1000);
    let target_tb = TimeBase::new(1, 90000);

    let dur = Duration::new(1000, source_tb); // 1 second
    let rescaled = dur.rescale(target_tb);

    // 1 second = 90000 ticks at 90kHz
    assert_eq!(rescaled.value, 90000);
}

/// Test packet cloning.
#[test]
fn test_packet_clone() {
    let original = Packet::new(vec![1, 2, 3, 4, 5])
        .with_stream_index(1)
        .with_flags(PacketFlags::KEYFRAME);

    let cloned = original.clone();

    assert_eq!(cloned.data(), original.data());
    assert_eq!(cloned.stream_index, original.stream_index);
    assert_eq!(cloned.flags, original.flags);
}

/// Test packet size calculation.
#[test]
fn test_packet_sizes() {
    let small = Packet::new(vec![0u8; 100]);
    let medium = Packet::new(vec![0u8; 10_000]);
    let large = Packet::new(vec![0u8; 1_000_000]);

    assert_eq!(small.size(), 100);
    assert_eq!(medium.size(), 10_000);
    assert_eq!(large.size(), 1_000_000);
}

/// Test timestamp comparison.
#[test]
fn test_timestamp_ordering() {
    let tb = TimeBase::new(1, 30);

    let ts1 = Timestamp::new(0, tb);
    let ts2 = Timestamp::new(30, tb);
    let ts3 = Timestamp::new(60, tb);

    assert!(ts1.value < ts2.value);
    assert!(ts2.value < ts3.value);
}

/// Test packet timestamp propagation.
#[test]
fn test_packet_timestamp_propagation() {
    let tb = TimeBase::new(1, 90000);
    let pts = Timestamp::new(90000, tb);

    let packet = Packet::new(vec![0u8; 100])
        .with_timestamps(pts, pts);

    // Timestamps should be preserved
    assert_eq!(packet.pts.value, 90000);
    assert_eq!(packet.pts.to_seconds().unwrap(), 1.0);
}

/// Test TimeBase conversion.
#[test]
fn test_time_base_conversion() {
    let tb = TimeBase::new(1, 90000);

    // Convert 90000 ticks at 90kHz to seconds
    let seconds = tb.to_seconds(90000);
    assert_eq!(seconds, 1.0);

    // Convert 1 second back to ticks
    let ticks = tb.from_seconds(1.0);
    assert_eq!(ticks, 90000);
}

/// Test invalid timestamp.
#[test]
fn test_invalid_timestamp() {
    let ts = Timestamp::none();

    assert!(!ts.is_valid());
    assert!(ts.to_seconds().is_none());
}

/// Test Duration zero.
#[test]
fn test_duration_zero() {
    let dur = Duration::zero();

    assert!(dur.is_zero());
    assert_eq!(dur.to_seconds(), 0.0);
}

/// Test Duration from milliseconds.
#[test]
fn test_duration_from_millis() {
    let dur = Duration::from_millis(1500);

    assert_eq!(dur.to_seconds(), 1.5);
}

/// Test Timestamp from milliseconds.
#[test]
fn test_timestamp_from_millis() {
    let ts = Timestamp::from_millis(2500);

    assert_eq!(ts.to_seconds().unwrap(), 2.5);
}
