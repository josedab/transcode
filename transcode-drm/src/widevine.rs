//! Widevine PSSH box generation.
//!
//! This module implements Widevine-specific PSSH (Protection System Specific Header)
//! box generation for DRM packaging. Widevine uses a protobuf-encoded data format
//! within the PSSH box.

use crate::error::{PsshError, Result};
use crate::key::{system_ids, KeyId};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Widevine system ID.
pub const WIDEVINE_SYSTEM_ID: Uuid = system_ids::WIDEVINE;

/// Maximum size for PSSH data.
pub const MAX_PSSH_DATA_SIZE: usize = 64 * 1024; // 64KB

/// Widevine content protection algorithm.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Algorithm {
    /// Unencrypted content.
    Unencrypted = 0,
    /// AES-CTR encryption.
    #[default]
    AesCtr = 1,
    /// AES-CBC encryption (used with CBCS).
    AesCbc = 2,
}

/// Widevine protection scheme.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProtectionScheme {
    /// CENC scheme (AES-CTR).
    #[default]
    Cenc,
    /// CBC1 scheme (AES-CBC).
    Cbc1,
    /// CENS scheme (AES-CTR with pattern).
    Cens,
    /// CBCS scheme (AES-CBC with pattern).
    Cbcs,
}

impl ProtectionScheme {
    /// Get the four-character code as u32.
    pub fn as_fourcc(&self) -> u32 {
        match self {
            Self::Cenc => u32::from_be_bytes(*b"cenc"),
            Self::Cbc1 => u32::from_be_bytes(*b"cbc1"),
            Self::Cens => u32::from_be_bytes(*b"cens"),
            Self::Cbcs => u32::from_be_bytes(*b"cbcs"),
        }
    }
}

/// Widevine PSSH data (simplified protobuf encoding).
///
/// This structure represents the data portion of a Widevine PSSH box.
/// In production, this would use proper protobuf encoding.
#[derive(Clone, Debug, Default)]
pub struct WidevineData {
    /// Algorithm used for encryption.
    pub algorithm: Algorithm,
    /// Key IDs included in this PSSH.
    pub key_ids: Vec<KeyId>,
    /// Content provider name.
    pub provider: Option<String>,
    /// Content ID.
    pub content_id: Option<Vec<u8>>,
    /// Policy string.
    pub policy: Option<String>,
    /// Protection scheme.
    pub protection_scheme: ProtectionScheme,
    /// Crypto period index (for key rotation).
    pub crypto_period_index: Option<u32>,
    /// Grouped license (for multiple keys).
    pub grouped_license: Option<Vec<u8>>,
}

impl WidevineData {
    /// Create new Widevine data with a single key ID.
    pub fn new(key_id: KeyId) -> Self {
        Self {
            key_ids: vec![key_id],
            ..Default::default()
        }
    }

    /// Create new Widevine data with multiple key IDs.
    pub fn with_key_ids(key_ids: Vec<KeyId>) -> Self {
        Self {
            key_ids,
            ..Default::default()
        }
    }

    /// Set the content provider.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Set the content ID.
    pub fn with_content_id(mut self, content_id: Vec<u8>) -> Self {
        self.content_id = Some(content_id);
        self
    }

    /// Set the policy.
    pub fn with_policy(mut self, policy: impl Into<String>) -> Self {
        self.policy = Some(policy.into());
        self
    }

    /// Set the protection scheme.
    pub fn with_protection_scheme(mut self, scheme: ProtectionScheme) -> Self {
        self.protection_scheme = scheme;
        self
    }

    /// Set the algorithm.
    pub fn with_algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Add a key ID.
    pub fn add_key_id(&mut self, key_id: KeyId) {
        self.key_ids.push(key_id);
    }

    /// Encode the data as simplified protobuf.
    ///
    /// This is a simplified encoding that covers the most common fields.
    /// Full protobuf support would require a proper protobuf library.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut data = Vec::new();

        // Field 1: Algorithm (varint)
        if self.algorithm != Algorithm::Unencrypted {
            data.push(0x08); // field 1, wire type 0 (varint)
            data.push(self.algorithm as u8);
        }

        // Field 2: Key IDs (length-delimited, repeated)
        for key_id in &self.key_ids {
            data.push(0x12); // field 2, wire type 2 (length-delimited)
            data.push(16); // length = 16 bytes
            data.extend_from_slice(key_id.as_bytes());
        }

        // Field 3: Provider (length-delimited)
        if let Some(ref provider) = self.provider {
            let provider_bytes = provider.as_bytes();
            data.push(0x1a); // field 3, wire type 2
            encode_varint(&mut data, provider_bytes.len() as u64);
            data.extend_from_slice(provider_bytes);
        }

        // Field 4: Content ID (length-delimited)
        if let Some(ref content_id) = self.content_id {
            data.push(0x22); // field 4, wire type 2
            encode_varint(&mut data, content_id.len() as u64);
            data.extend_from_slice(content_id);
        }

        // Field 6: Policy (length-delimited)
        if let Some(ref policy) = self.policy {
            let policy_bytes = policy.as_bytes();
            data.push(0x32); // field 6, wire type 2
            encode_varint(&mut data, policy_bytes.len() as u64);
            data.extend_from_slice(policy_bytes);
        }

        // Field 9: Protection scheme (fixed32)
        data.push(0x4d); // field 9, wire type 5 (fixed32)
        data.extend_from_slice(&self.protection_scheme.as_fourcc().to_le_bytes());

        // Field 7: Crypto period index (varint)
        if let Some(index) = self.crypto_period_index {
            data.push(0x38); // field 7, wire type 0
            encode_varint(&mut data, index as u64);
        }

        if data.len() > MAX_PSSH_DATA_SIZE {
            return Err(PsshError::DataTooLarge {
                size: data.len(),
                max: MAX_PSSH_DATA_SIZE,
            }
            .into());
        }

        Ok(data)
    }

    /// Decode from simplified protobuf.
    ///
    /// # Errors
    ///
    /// Returns an error if the data exceeds `MAX_PSSH_DATA_SIZE`.
    pub fn decode(data: &[u8]) -> Result<Self> {
        // Validate input size before processing
        if data.len() > MAX_PSSH_DATA_SIZE {
            return Err(PsshError::DataTooLarge {
                size: data.len(),
                max: MAX_PSSH_DATA_SIZE,
            }
            .into());
        }

        let mut result = Self::default();
        let mut pos = 0;

        while pos < data.len() {
            let tag = data[pos];
            pos += 1;

            let field_number = tag >> 3;
            let wire_type = tag & 0x07;

            match (field_number, wire_type) {
                (1, 0) => {
                    // Algorithm
                    let (value, bytes_read) = decode_varint(&data[pos..])?;
                    result.algorithm = match value {
                        0 => Algorithm::Unencrypted,
                        1 => Algorithm::AesCtr,
                        2 => Algorithm::AesCbc,
                        _ => Algorithm::AesCtr,
                    };
                    pos += bytes_read;
                }
                (2, 2) => {
                    // Key ID
                    let (len, bytes_read) = decode_varint(&data[pos..])?;
                    pos += bytes_read;
                    if len == 16 && pos + 16 <= data.len() {
                        result.key_ids.push(KeyId::from_slice(&data[pos..pos + 16])?);
                        pos += 16;
                    } else {
                        pos += len as usize;
                    }
                }
                (3, 2) => {
                    // Provider
                    let (len, bytes_read) = decode_varint(&data[pos..])?;
                    pos += bytes_read;
                    let end = pos + len as usize;
                    if end <= data.len() {
                        result.provider = Some(
                            String::from_utf8_lossy(&data[pos..end]).into_owned()
                        );
                    }
                    pos = end;
                }
                (4, 2) => {
                    // Content ID
                    let (len, bytes_read) = decode_varint(&data[pos..])?;
                    pos += bytes_read;
                    let end = pos + len as usize;
                    if end <= data.len() {
                        result.content_id = Some(data[pos..end].to_vec());
                    }
                    pos = end;
                }
                (6, 2) => {
                    // Policy
                    let (len, bytes_read) = decode_varint(&data[pos..])?;
                    pos += bytes_read;
                    let end = pos + len as usize;
                    if end <= data.len() {
                        result.policy = Some(
                            String::from_utf8_lossy(&data[pos..end]).into_owned()
                        );
                    }
                    pos = end;
                }
                (7, 0) => {
                    // Crypto period index
                    let (value, bytes_read) = decode_varint(&data[pos..])?;
                    result.crypto_period_index = Some(value as u32);
                    pos += bytes_read;
                }
                (9, 5) => {
                    // Protection scheme (fixed32)
                    if pos + 4 <= data.len() {
                        let fourcc = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                        result.protection_scheme = match &fourcc.to_be_bytes() {
                            b"cenc" => ProtectionScheme::Cenc,
                            b"cbc1" => ProtectionScheme::Cbc1,
                            b"cens" => ProtectionScheme::Cens,
                            b"cbcs" => ProtectionScheme::Cbcs,
                            _ => ProtectionScheme::Cenc,
                        };
                    }
                    pos += 4;
                }
                _ => {
                    // Skip unknown fields
                    match wire_type {
                        0 => {
                            let (_, bytes_read) = decode_varint(&data[pos..])?;
                            pos += bytes_read;
                        }
                        2 => {
                            let (len, bytes_read) = decode_varint(&data[pos..])?;
                            pos += bytes_read + len as usize;
                        }
                        5 => pos += 4,
                        1 => pos += 8,
                        _ => break,
                    }
                }
            }
        }

        Ok(result)
    }
}

/// Widevine PSSH box.
#[derive(Clone, Debug)]
pub struct WidevinePssh {
    /// PSSH version (0 or 1).
    pub version: u8,
    /// PSSH flags.
    pub flags: u32,
    /// Key IDs (for version 1).
    pub key_ids: Vec<KeyId>,
    /// Widevine-specific data.
    pub data: WidevineData,
}

impl WidevinePssh {
    /// Create a new Widevine PSSH box (version 0).
    pub fn new(data: WidevineData) -> Self {
        Self {
            version: 0,
            flags: 0,
            key_ids: Vec::new(),
            data,
        }
    }

    /// Create a version 1 PSSH box with key IDs in the header.
    pub fn version1(data: WidevineData) -> Self {
        let key_ids = data.key_ids.clone();
        Self {
            version: 1,
            flags: 0,
            key_ids,
            data,
        }
    }

    /// Create from a single key ID.
    pub fn from_key_id(key_id: KeyId) -> Self {
        Self::new(WidevineData::new(key_id))
    }

    /// Create from multiple key IDs.
    pub fn from_key_ids(key_ids: Vec<KeyId>) -> Self {
        Self::new(WidevineData::with_key_ids(key_ids))
    }

    /// Serialize to a complete PSSH box.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let pssh_data = self.data.encode()?;
        let mut box_data = Vec::new();

        // Version and flags (4 bytes)
        box_data.push(self.version);
        box_data.extend_from_slice(&self.flags.to_be_bytes()[1..]);

        // System ID (16 bytes)
        box_data.extend_from_slice(WIDEVINE_SYSTEM_ID.as_bytes());

        // Version 1: Key IDs
        if self.version >= 1 {
            box_data.extend_from_slice(&(self.key_ids.len() as u32).to_be_bytes());
            for key_id in &self.key_ids {
                box_data.extend_from_slice(key_id.as_bytes());
            }
        }

        // Data size and data
        box_data.extend_from_slice(&(pssh_data.len() as u32).to_be_bytes());
        box_data.extend_from_slice(&pssh_data);

        // Create full box with header
        let box_size = 8 + box_data.len(); // 4 (size) + 4 (type) + data
        let mut full_box = Vec::with_capacity(box_size);
        full_box.extend_from_slice(&(box_size as u32).to_be_bytes());
        full_box.extend_from_slice(b"pssh");
        full_box.extend_from_slice(&box_data);

        Ok(full_box)
    }

    /// Serialize to base64 (for manifests).
    pub fn to_base64(&self) -> Result<String> {
        let bytes = self.to_bytes()?;
        Ok(BASE64.encode(&bytes))
    }

    /// Parse from a PSSH box.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 32 {
            return Err(PsshError::InvalidFormat("PSSH box too small".into()).into());
        }

        let mut pos = 0;

        // Box size
        let _box_size = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        // Box type
        if &data[pos..pos + 4] != b"pssh" {
            return Err(PsshError::InvalidFormat("Not a PSSH box".into()).into());
        }
        pos += 4;

        // Version and flags
        let version = data[pos];
        let flags = u32::from_be_bytes([0, data[pos + 1], data[pos + 2], data[pos + 3]]);
        pos += 4;

        // System ID
        let system_id = Uuid::from_bytes(data[pos..pos + 16].try_into().unwrap());
        if system_id != WIDEVINE_SYSTEM_ID {
            return Err(PsshError::UnknownSystemId(system_id.to_string()).into());
        }
        pos += 16;

        // Version 1: Key IDs
        let mut key_ids = Vec::new();
        if version >= 1 {
            let kid_count = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            for _ in 0..kid_count {
                if pos + 16 <= data.len() {
                    key_ids.push(KeyId::from_slice(&data[pos..pos + 16])?);
                    pos += 16;
                }
            }
        }

        // Data size and data
        if pos + 4 > data.len() {
            return Err(PsshError::InvalidFormat("Missing data size".into()).into());
        }
        let data_size = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        if pos + data_size > data.len() {
            return Err(PsshError::InvalidFormat("Data truncated".into()).into());
        }

        let widevine_data = WidevineData::decode(&data[pos..pos + data_size])?;

        Ok(Self {
            version,
            flags,
            key_ids,
            data: widevine_data,
        })
    }

    /// Parse from base64.
    pub fn from_base64(encoded: &str) -> Result<Self> {
        let bytes = BASE64
            .decode(encoded)
            .map_err(|e| PsshError::InvalidFormat(e.to_string()))?;
        Self::from_bytes(&bytes)
    }
}

/// Encode a varint (protobuf variable-length integer).
fn encode_varint(buf: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        buf.push((value as u8) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
}

/// Decode a varint, returning the value and number of bytes read.
fn decode_varint(data: &[u8]) -> Result<(u64, usize)> {
    let mut value: u64 = 0;
    let mut shift = 0;

    for (i, &byte) in data.iter().enumerate() {
        if shift >= 64 {
            return Err(PsshError::ProtobufError("Varint too large".into()).into());
        }

        value |= ((byte & 0x7F) as u64) << shift;

        if byte & 0x80 == 0 {
            return Ok((value, i + 1));
        }

        shift += 7;
    }

    Err(PsshError::ProtobufError("Incomplete varint".into()).into())
}

/// Generate a Widevine PSSH for a content configuration.
pub fn generate_pssh(
    key_ids: &[KeyId],
    provider: Option<&str>,
    content_id: Option<&[u8]>,
) -> Result<WidevinePssh> {
    if key_ids.is_empty() {
        return Err(PsshError::MissingField("key_ids".into()).into());
    }

    let mut data = WidevineData::with_key_ids(key_ids.to_vec());

    if let Some(provider) = provider {
        data = data.with_provider(provider);
    }

    if let Some(content_id) = content_id {
        data = data.with_content_id(content_id.to_vec());
    }

    Ok(WidevinePssh::new(data))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key_id() -> KeyId {
        KeyId::parse("12345678-1234-1234-1234-123456789012").unwrap()
    }

    #[test]
    fn test_widevine_system_id() {
        assert_eq!(
            WIDEVINE_SYSTEM_ID.to_string(),
            "edef8ba9-79d6-4ace-a3c8-27dcd51d21ed"
        );
    }

    #[test]
    fn test_widevine_data_encode_decode() {
        let key_id = test_key_id();
        let data = WidevineData::new(key_id)
            .with_provider("TestProvider")
            .with_content_id(b"test-content-123".to_vec())
            .with_protection_scheme(ProtectionScheme::Cenc);

        let encoded = data.encode().unwrap();
        let decoded = WidevineData::decode(&encoded).unwrap();

        assert_eq!(decoded.key_ids.len(), 1);
        assert_eq!(decoded.key_ids[0], key_id);
        assert_eq!(decoded.provider, Some("TestProvider".into()));
        assert_eq!(decoded.content_id, Some(b"test-content-123".to_vec()));
        assert_eq!(decoded.protection_scheme, ProtectionScheme::Cenc);
    }

    #[test]
    fn test_widevine_data_multiple_keys() {
        let key1 = KeyId::parse("11111111-1111-1111-1111-111111111111").unwrap();
        let key2 = KeyId::parse("22222222-2222-2222-2222-222222222222").unwrap();

        let data = WidevineData::with_key_ids(vec![key1, key2]);
        let encoded = data.encode().unwrap();
        let decoded = WidevineData::decode(&encoded).unwrap();

        assert_eq!(decoded.key_ids.len(), 2);
        assert_eq!(decoded.key_ids[0], key1);
        assert_eq!(decoded.key_ids[1], key2);
    }

    #[test]
    fn test_widevine_pssh_roundtrip() {
        let key_id = test_key_id();
        let pssh = WidevinePssh::from_key_id(key_id);

        let bytes = pssh.to_bytes().unwrap();
        let parsed = WidevinePssh::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.version, 0);
        assert_eq!(parsed.data.key_ids.len(), 1);
        assert_eq!(parsed.data.key_ids[0], key_id);
    }

    #[test]
    fn test_widevine_pssh_version1() {
        let key_id = test_key_id();
        let data = WidevineData::new(key_id);
        let pssh = WidevinePssh::version1(data);

        assert_eq!(pssh.version, 1);
        assert_eq!(pssh.key_ids.len(), 1);

        let bytes = pssh.to_bytes().unwrap();
        let parsed = WidevinePssh::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.key_ids.len(), 1);
        assert_eq!(parsed.key_ids[0], key_id);
    }

    #[test]
    fn test_widevine_pssh_base64() {
        let key_id = test_key_id();
        let pssh = WidevinePssh::from_key_id(key_id);

        let base64 = pssh.to_base64().unwrap();
        let parsed = WidevinePssh::from_base64(&base64).unwrap();

        assert_eq!(parsed.data.key_ids[0], key_id);
    }

    #[test]
    fn test_generate_pssh() {
        let key_id = test_key_id();
        let pssh = generate_pssh(&[key_id], Some("TestProvider"), Some(b"content123")).unwrap();

        assert_eq!(pssh.data.key_ids.len(), 1);
        assert_eq!(pssh.data.provider, Some("TestProvider".into()));
        assert_eq!(pssh.data.content_id, Some(b"content123".to_vec()));
    }

    #[test]
    fn test_generate_pssh_no_keys() {
        let result = generate_pssh(&[], None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_protection_scheme_fourcc() {
        assert_eq!(
            ProtectionScheme::Cenc.as_fourcc(),
            u32::from_be_bytes(*b"cenc")
        );
        assert_eq!(
            ProtectionScheme::Cbcs.as_fourcc(),
            u32::from_be_bytes(*b"cbcs")
        );
    }

    #[test]
    fn test_varint_encoding() {
        let mut buf = Vec::new();

        // Single byte
        encode_varint(&mut buf, 1);
        assert_eq!(buf, vec![1]);
        let (val, len) = decode_varint(&buf).unwrap();
        assert_eq!(val, 1);
        assert_eq!(len, 1);

        // Multi-byte
        buf.clear();
        encode_varint(&mut buf, 300);
        let (val, len) = decode_varint(&buf).unwrap();
        assert_eq!(val, 300);
        assert_eq!(len, 2);
    }

    #[test]
    fn test_pssh_box_structure() {
        let key_id = test_key_id();
        let pssh = WidevinePssh::from_key_id(key_id);
        let bytes = pssh.to_bytes().unwrap();

        // Check box header
        let size = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;
        assert_eq!(size, bytes.len());
        assert_eq!(&bytes[4..8], b"pssh");

        // Check system ID position
        assert_eq!(&bytes[12..28], WIDEVINE_SYSTEM_ID.as_bytes());
    }
}
