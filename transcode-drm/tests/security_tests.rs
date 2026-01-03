//! Security tests for DRM input validation and DoS prevention
//!
//! These tests verify that the DRM modules properly validate all inputs
//! to prevent denial-of-service attacks through resource exhaustion.

use transcode_drm::cenc::{CencConfig, CencEncryptor, EncryptionScheme, MAX_SAMPLE_SIZE};
use transcode_drm::error::{DrmError, EncryptionError, PsshError};
use transcode_drm::key::{ContentKey, Iv, KeyId};
use transcode_drm::widevine::{WidevineData, MAX_PSSH_DATA_SIZE};

// =============================================================================
// Helper Functions
// =============================================================================

fn test_key_pair() -> (KeyId, ContentKey) {
    let key_id = KeyId::parse("12345678-1234-1234-1234-123456789012").unwrap();
    let key = ContentKey::from_hex("00112233445566778899aabbccddeeff").unwrap();
    (key_id, key)
}

fn test_iv() -> Iv {
    Iv::from_hex("00000000000000000000000000000000").unwrap()
}

// =============================================================================
// PSSH Data Size Validation Tests
// =============================================================================

#[test]
fn test_widevine_decode_valid_size() {
    // Create small valid data
    let key_id = KeyId::parse("12345678-1234-1234-1234-123456789012").unwrap();
    let data = WidevineData::new(key_id);
    let encoded = data.encode().unwrap();

    // Should decode successfully
    let decoded = WidevineData::decode(&encoded);
    assert!(decoded.is_ok());
}

#[test]
fn test_widevine_decode_at_max_size() {
    // Create data at exactly MAX_PSSH_DATA_SIZE
    let data = vec![0u8; MAX_PSSH_DATA_SIZE];

    // This should be allowed (though it may fail to parse as valid protobuf,
    // the size check should pass)
    let result = WidevineData::decode(&data);
    // Size check passed, may fail on content parsing
    assert!(result.is_ok() || !matches!(
        result.as_ref().err(),
        Some(DrmError::Pssh(PsshError::DataTooLarge { .. }))
    ));
}

#[test]
fn test_widevine_decode_exceeds_max_size() {
    // Create data exceeding MAX_PSSH_DATA_SIZE
    let data = vec![0u8; MAX_PSSH_DATA_SIZE + 1];

    let result = WidevineData::decode(&data);
    assert!(result.is_err());

    // Verify it's specifically a DataTooLarge error
    match result.unwrap_err() {
        DrmError::Pssh(PsshError::DataTooLarge { size, max }) => {
            assert_eq!(size, MAX_PSSH_DATA_SIZE + 1);
            assert_eq!(max, MAX_PSSH_DATA_SIZE);
        }
        other => panic!("Expected DataTooLarge error, got: {:?}", other),
    }
}

#[test]
fn test_widevine_decode_extreme_size_rejected() {
    // Try to decode extremely large data (would cause OOM if not checked)
    let data = vec![0u8; MAX_PSSH_DATA_SIZE * 10];

    let result = WidevineData::decode(&data);
    assert!(result.is_err());

    match result.unwrap_err() {
        DrmError::Pssh(PsshError::DataTooLarge { .. }) => {}
        other => panic!("Expected DataTooLarge error, got: {:?}", other),
    }
}

#[test]
fn test_widevine_encode_size_limit() {
    // The encode function also checks size limits
    let key_id = KeyId::parse("12345678-1234-1234-1234-123456789012").unwrap();
    let mut data = WidevineData::new(key_id);

    // Add a very large provider string that would exceed limits
    // MAX_PSSH_DATA_SIZE is 64KB, so we need content larger than that
    data.provider = Some("x".repeat(MAX_PSSH_DATA_SIZE + 1000));

    let result = data.encode();
    assert!(result.is_err());

    match result.unwrap_err() {
        DrmError::Pssh(PsshError::DataTooLarge { .. }) => {}
        other => panic!("Expected DataTooLarge error, got: {:?}", other),
    }
}

// =============================================================================
// CENC Sample Size Validation Tests
// =============================================================================

#[test]
fn test_cenc_encrypt_valid_sample() {
    let (key_id, key) = test_key_pair();
    let iv = test_iv();

    let config = CencConfig::new(EncryptionScheme::Cenc, key_id, key, iv);
    let mut encryptor = CencEncryptor::new(config).unwrap();

    // Small sample should work
    let mut data = vec![0u8; 1024];
    let result = encryptor.encrypt_sample(&mut data);
    assert!(result.is_ok());
}

#[test]
fn test_cenc_encrypt_at_max_sample_size() {
    let (key_id, key) = test_key_pair();
    let iv = test_iv();

    let config = CencConfig::new(EncryptionScheme::Cenc, key_id, key, iv);
    let mut encryptor = CencEncryptor::new(config).unwrap();

    // Sample at exactly MAX_SAMPLE_SIZE should work
    let mut data = vec![0u8; MAX_SAMPLE_SIZE];
    let result = encryptor.encrypt_sample(&mut data);
    assert!(result.is_ok());
}

#[test]
fn test_cenc_encrypt_exceeds_max_sample_size() {
    let (key_id, key) = test_key_pair();
    let iv = test_iv();

    let config = CencConfig::new(EncryptionScheme::Cenc, key_id, key, iv);
    let mut encryptor = CencEncryptor::new(config).unwrap();

    // Sample exceeding MAX_SAMPLE_SIZE should fail
    let mut data = vec![0u8; MAX_SAMPLE_SIZE + 1];
    let result = encryptor.encrypt_sample(&mut data);
    assert!(result.is_err());

    match result.unwrap_err() {
        DrmError::Encryption(EncryptionError::SampleTooLarge { size, max }) => {
            assert_eq!(size, MAX_SAMPLE_SIZE + 1);
            assert_eq!(max, MAX_SAMPLE_SIZE);
        }
        other => panic!("Expected SampleTooLarge error, got: {:?}", other),
    }
}

#[test]
fn test_cenc_encrypt_cbcs_sample_size_limit() {
    let (key_id, key) = test_key_pair();
    let iv = test_iv();

    // Test with CBCS scheme as well
    let config = CencConfig::new(EncryptionScheme::Cbcs, key_id, key, iv);
    let mut encryptor = CencEncryptor::new(config).unwrap();

    // Should reject oversized samples
    let mut data = vec![0u8; MAX_SAMPLE_SIZE + 1];
    let result = encryptor.encrypt_sample(&mut data);
    assert!(result.is_err());

    match result.unwrap_err() {
        DrmError::Encryption(EncryptionError::SampleTooLarge { .. }) => {}
        other => panic!("Expected SampleTooLarge error, got: {:?}", other),
    }
}

// =============================================================================
// Edge Cases and Boundary Tests
// =============================================================================

#[test]
fn test_pssh_data_size_constant_value() {
    // Verify the constant is set to 64KB
    assert_eq!(MAX_PSSH_DATA_SIZE, 64 * 1024);
}

#[test]
fn test_sample_size_constant_value() {
    // Verify the constant is set to 100MB
    assert_eq!(MAX_SAMPLE_SIZE, 100 * 1024 * 1024);
}

#[test]
fn test_empty_sample_encryption() {
    let (key_id, key) = test_key_pair();
    let iv = test_iv();

    let config = CencConfig::new(EncryptionScheme::Cenc, key_id, key, iv);
    let mut encryptor = CencEncryptor::new(config).unwrap();

    // Empty sample should work (edge case, 0 < MAX_SAMPLE_SIZE)
    let mut data: Vec<u8> = vec![];
    let result = encryptor.encrypt_sample(&mut data);
    assert!(result.is_ok());
}

#[test]
fn test_single_byte_sample_encryption() {
    let (key_id, key) = test_key_pair();
    let iv = test_iv();

    let config = CencConfig::new(EncryptionScheme::Cenc, key_id, key, iv);
    let mut encryptor = CencEncryptor::new(config).unwrap();

    // Single byte sample should work
    let mut data = vec![0xFFu8];
    let result = encryptor.encrypt_sample(&mut data);
    assert!(result.is_ok());
}

#[test]
fn test_multiple_large_samples_sequential() {
    let (key_id, key) = test_key_pair();
    let iv = test_iv();

    let config = CencConfig::new(EncryptionScheme::Cenc, key_id, key, iv);
    let mut encryptor = CencEncryptor::new(config).unwrap();

    // Encrypt multiple samples near the limit - should all succeed
    for _ in 0..5 {
        let mut data = vec![0u8; MAX_SAMPLE_SIZE];
        let result = encryptor.encrypt_sample(&mut data);
        assert!(result.is_ok());
    }

    // But an oversized one should still fail
    let mut oversized = vec![0u8; MAX_SAMPLE_SIZE + 1];
    let result = encryptor.encrypt_sample(&mut oversized);
    assert!(result.is_err());
}

// =============================================================================
// Error Message Quality Tests
// =============================================================================

#[test]
fn test_sample_too_large_error_display() {
    let err = EncryptionError::SampleTooLarge {
        size: 150_000_000,
        max: 100_000_000,
    };

    let msg = err.to_string();
    assert!(msg.contains("150000000"));
    assert!(msg.contains("100000000"));
    assert!(msg.contains("Sample too large"));
}

#[test]
fn test_pssh_data_too_large_error_display() {
    let err = PsshError::DataTooLarge {
        size: 100_000,
        max: 65536,
    };

    let msg = err.to_string();
    assert!(msg.contains("100000"));
    assert!(msg.contains("65536"));
    assert!(msg.contains("too large"));
}
