//! Security tests for input validation and DoS prevention
//!
//! These tests verify that the live streaming modules properly validate
//! all inputs to prevent denial-of-service attacks through resource exhaustion.

use std::net::SocketAddr;
use std::time::Duration;
use transcode_live::*;

// =============================================================================
// RTMP Chunk Size Validation Tests
// =============================================================================

#[test]
fn test_rtmp_chunk_size_at_minimum() {
    let mut stream = RtmpChunkStream::new();
    assert!(stream.set_chunk_size(RTMP_MIN_CHUNK_SIZE).is_ok());
    assert_eq!(stream.chunk_size(), RTMP_MIN_CHUNK_SIZE);
}

#[test]
fn test_rtmp_chunk_size_at_maximum() {
    let mut stream = RtmpChunkStream::new();
    assert!(stream.set_chunk_size(RTMP_MAX_CHUNK_SIZE).is_ok());
    assert_eq!(stream.chunk_size(), RTMP_MAX_CHUNK_SIZE);
}

#[test]
fn test_rtmp_chunk_size_below_minimum_rejected() {
    let mut stream = RtmpChunkStream::new();
    let result = stream.set_chunk_size(RTMP_MIN_CHUNK_SIZE - 1);
    assert!(result.is_err());
    // Verify original value unchanged
    assert_eq!(stream.chunk_size(), RTMP_DEFAULT_CHUNK_SIZE);
}

#[test]
fn test_rtmp_chunk_size_zero_rejected() {
    let mut stream = RtmpChunkStream::new();
    let result = stream.set_chunk_size(0);
    assert!(result.is_err());
}

#[test]
fn test_rtmp_chunk_size_above_maximum_rejected() {
    let mut stream = RtmpChunkStream::new();
    let result = stream.set_chunk_size(RTMP_MAX_CHUNK_SIZE + 1);
    assert!(result.is_err());
    // Verify original value unchanged
    assert_eq!(stream.chunk_size(), RTMP_DEFAULT_CHUNK_SIZE);
}

#[test]
fn test_rtmp_chunk_size_extreme_value_rejected() {
    let mut stream = RtmpChunkStream::new();
    // Try to set chunk size to max u32 - would cause huge allocations
    let result = stream.set_chunk_size(u32::MAX);
    assert!(result.is_err());
}

// =============================================================================
// RTMP Window Size Validation Tests
// =============================================================================

#[test]
fn test_rtmp_window_size_valid() {
    let mut stream = RtmpChunkStream::new();
    assert!(stream.set_window_size(1_000_000).is_ok());
    assert_eq!(stream.window_size(), 1_000_000);
}

#[test]
fn test_rtmp_window_size_at_maximum() {
    let mut stream = RtmpChunkStream::new();
    assert!(stream.set_window_size(RTMP_MAX_WINDOW_SIZE).is_ok());
    assert_eq!(stream.window_size(), RTMP_MAX_WINDOW_SIZE);
}

#[test]
fn test_rtmp_window_size_above_maximum_rejected() {
    let mut stream = RtmpChunkStream::new();
    let original = stream.window_size();
    let result = stream.set_window_size(RTMP_MAX_WINDOW_SIZE + 1);
    assert!(result.is_err());
    // Verify original value unchanged
    assert_eq!(stream.window_size(), original);
}

#[test]
fn test_rtmp_window_size_extreme_value_rejected() {
    let mut stream = RtmpChunkStream::new();
    let result = stream.set_window_size(u32::MAX);
    assert!(result.is_err());
}

// =============================================================================
// SRT Passphrase Validation Tests
// =============================================================================

#[test]
fn test_srt_passphrase_valid() {
    let config = SrtConfig {
        passphrase: Some("valid_passphrase".into()),
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_srt_passphrase_at_max_length() {
    let config = SrtConfig {
        passphrase: Some("x".repeat(SRT_MAX_PASSPHRASE_LEN)),
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_srt_passphrase_empty_rejected() {
    let config = SrtConfig {
        passphrase: Some(String::new()),
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_srt_passphrase_exceeds_max_rejected() {
    let config = SrtConfig {
        passphrase: Some("x".repeat(SRT_MAX_PASSPHRASE_LEN + 1)),
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_srt_passphrase_extreme_length_rejected() {
    let config = SrtConfig {
        passphrase: Some("x".repeat(1_000_000)),
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

// =============================================================================
// SRT Stream ID Validation Tests
// =============================================================================

#[test]
fn test_srt_stream_id_valid() {
    let config = SrtConfig {
        stream_id: Some("valid_stream_id".into()),
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_srt_stream_id_at_max_length() {
    let config = SrtConfig {
        stream_id: Some("x".repeat(SRT_MAX_STREAM_ID_LEN)),
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_srt_stream_id_exceeds_max_rejected() {
    let config = SrtConfig {
        stream_id: Some("x".repeat(SRT_MAX_STREAM_ID_LEN + 1)),
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

// =============================================================================
// SRT Bandwidth Validation Tests
// =============================================================================

#[test]
fn test_srt_bandwidth_valid() {
    let config = SrtConfig {
        max_bandwidth: 1_000_000_000, // 1 Gbps
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_srt_bandwidth_at_maximum() {
    let config = SrtConfig {
        max_bandwidth: SRT_MAX_BANDWIDTH,
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_srt_bandwidth_exceeds_max_rejected() {
    let config = SrtConfig {
        max_bandwidth: SRT_MAX_BANDWIDTH + 1,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

// =============================================================================
// SRT Socket Creation Validation Tests
// =============================================================================

#[test]
fn test_srt_socket_validates_config_on_creation() {
    // Invalid config should fail socket creation
    let invalid_config = SrtConfig {
        passphrase: Some(String::new()), // Empty passphrase is invalid
        ..Default::default()
    };
    let result = SrtSocket::new(invalid_config);
    assert!(result.is_err());
}

#[test]
fn test_srt_socket_valid_config_succeeds() {
    let valid_config = SrtConfig::default();
    let result = SrtSocket::new(valid_config);
    assert!(result.is_ok());
}

// =============================================================================
// LiveConfig Buffer Size Validation Tests
// =============================================================================

#[test]
fn test_live_config_buffer_size_at_minimum() {
    let config = LiveConfig {
        buffer_size: MIN_BUFFER_SIZE,
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_live_config_buffer_size_at_maximum() {
    let config = LiveConfig {
        buffer_size: MAX_BUFFER_SIZE,
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_live_config_buffer_size_below_minimum_rejected() {
    let config = LiveConfig {
        buffer_size: MIN_BUFFER_SIZE - 1,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_live_config_buffer_size_zero_rejected() {
    let config = LiveConfig {
        buffer_size: 0,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_live_config_buffer_size_above_maximum_rejected() {
    let config = LiveConfig {
        buffer_size: MAX_BUFFER_SIZE + 1,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_live_config_buffer_size_extreme_value_rejected() {
    let config = LiveConfig {
        buffer_size: usize::MAX,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

// =============================================================================
// LiveConfig Max Clients Validation Tests
// =============================================================================

#[test]
fn test_live_config_max_clients_valid() {
    let config = LiveConfig {
        max_clients: 100,
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_live_config_max_clients_at_maximum() {
    let config = LiveConfig {
        max_clients: MAX_CLIENTS_LIMIT,
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_live_config_max_clients_zero_rejected() {
    let config = LiveConfig {
        max_clients: 0,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_live_config_max_clients_above_maximum_rejected() {
    let config = LiveConfig {
        max_clients: MAX_CLIENTS_LIMIT + 1,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

// =============================================================================
// LiveServer Validation Integration Tests
// =============================================================================

#[tokio::test]
async fn test_live_server_validates_config_on_start() {
    let invalid_config = LiveConfig {
        buffer_size: 0, // Invalid
        ..Default::default()
    };
    let mut server = LiveServer::new(invalid_config);
    let result = server.start().await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_live_server_valid_config_allowed() {
    let valid_config = LiveConfig {
        protocol: StreamProtocol::Rtmp,
        listen_addr: "127.0.0.1:0".parse::<SocketAddr>().unwrap(),
        timeout: Duration::from_secs(30),
        max_clients: 10,
        buffer_size: 256,
    };
    let server = LiveServer::new(valid_config);
    // Note: start() would actually try to bind, but validation should pass
    // For the purpose of this test, we just validate the config
    assert!(server.config().validate().is_ok());
}

// =============================================================================
// Combined Validation Tests
// =============================================================================

#[test]
fn test_srt_multiple_invalid_fields() {
    // Config with multiple invalid fields
    let config = SrtConfig {
        passphrase: Some(String::new()), // Invalid: empty
        stream_id: Some("x".repeat(1000)), // Invalid: too long
        max_bandwidth: u64::MAX,          // Invalid: too large
        ..Default::default()
    };
    // Should fail on first validation error
    assert!(config.validate().is_err());
}

#[test]
fn test_live_config_multiple_invalid_fields() {
    let config = LiveConfig {
        buffer_size: 0,      // Invalid
        max_clients: 0,      // Invalid
        ..Default::default()
    };
    assert!(config.validate().is_err());
}
