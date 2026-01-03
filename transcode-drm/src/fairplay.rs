//! FairPlay Streaming (FPS) support.
//!
//! This module implements Apple FairPlay Streaming DRM support for HLS content.
//! FairPlay uses the CBCS encryption scheme with a 1:9 pattern.

use crate::cbcs::Pattern;
use crate::cenc::EncryptionScheme;
use crate::error::{DrmError, PsshError, Result};
use crate::key::{system_ids, ContentKey, Iv, KeyId};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

/// FairPlay Streaming system ID.
pub const FAIRPLAY_SYSTEM_ID: Uuid = system_ids::FAIRPLAY;

/// FairPlay key format identifier.
pub const FAIRPLAY_KEY_FORMAT: &str = "com.apple.streamingkeydelivery";

/// FairPlay key format versions.
pub const FAIRPLAY_KEY_FORMAT_VERSION: &str = "1";

/// Default FairPlay encryption pattern - crypt blocks (1:9 for CBCS).
pub const FAIRPLAY_CRYPT_BLOCKS: u32 = 1;
/// Default FairPlay encryption pattern - skip blocks (1:9 for CBCS).
pub const FAIRPLAY_SKIP_BLOCKS: u32 = 9;

/// FairPlay streaming key request.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeyRequest {
    /// Asset ID.
    pub asset_id: String,
    /// Session ID.
    pub session_id: Option<String>,
    /// Key server URL.
    pub key_server_url: String,
    /// Certificate data (DER encoded).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub certificate: Option<Vec<u8>>,
}

impl KeyRequest {
    /// Create a new key request.
    pub fn new(asset_id: impl Into<String>, key_server_url: impl Into<String>) -> Self {
        Self {
            asset_id: asset_id.into(),
            session_id: None,
            key_server_url: key_server_url.into(),
            certificate: None,
        }
    }

    /// Set the session ID.
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Set the certificate.
    pub fn with_certificate(mut self, certificate: Vec<u8>) -> Self {
        self.certificate = Some(certificate);
        self
    }
}

/// FairPlay streaming content key context.
#[derive(Clone, Debug)]
pub struct FairPlayContext {
    /// Content key identifier.
    pub key_id: KeyId,
    /// Initialization vector.
    pub iv: Iv,
    /// Key server URL.
    pub key_server_url: String,
    /// Asset ID (derived from key ID if not specified).
    pub asset_id: Option<String>,
}

impl FairPlayContext {
    /// Create a new FairPlay context.
    pub fn new(key_id: KeyId, iv: Iv, key_server_url: impl Into<String>) -> Self {
        Self {
            key_id,
            iv,
            key_server_url: key_server_url.into(),
            asset_id: None,
        }
    }

    /// Set a custom asset ID.
    pub fn with_asset_id(mut self, asset_id: impl Into<String>) -> Self {
        self.asset_id = Some(asset_id.into());
        self
    }

    /// Get the asset ID (key ID hex if not set).
    pub fn get_asset_id(&self) -> String {
        self.asset_id
            .clone()
            .unwrap_or_else(|| self.key_id.to_hex())
    }

    /// Generate the EXT-X-KEY URI for HLS.
    pub fn key_uri(&self) -> String {
        format!(
            "skd://{}",
            self.get_asset_id()
        )
    }

    /// Generate the EXT-X-KEY tag for HLS master playlist.
    pub fn ext_x_key_tag(&self) -> String {
        format!(
            "#EXT-X-KEY:METHOD=SAMPLE-AES,URI=\"{}\",KEYFORMAT=\"{}\",KEYFORMATVERSIONS=\"{}\"",
            self.key_uri(),
            FAIRPLAY_KEY_FORMAT,
            FAIRPLAY_KEY_FORMAT_VERSION
        )
    }

    /// Generate the EXT-X-SESSION-KEY tag for HLS.
    pub fn ext_x_session_key_tag(&self) -> String {
        format!(
            "#EXT-X-SESSION-KEY:METHOD=SAMPLE-AES,URI=\"{}\",KEYFORMAT=\"{}\",KEYFORMATVERSIONS=\"{}\"",
            self.key_uri(),
            FAIRPLAY_KEY_FORMAT,
            FAIRPLAY_KEY_FORMAT_VERSION
        )
    }
}

/// FairPlay PSSH box (simplified - mainly for compatibility).
///
/// Note: FairPlay primarily uses EXT-X-KEY tags in HLS rather than PSSH boxes,
/// but a PSSH can be included for compatibility with some players.
#[derive(Clone, Debug)]
pub struct FairPlayPssh {
    /// PSSH version.
    pub version: u8,
    /// Key IDs.
    pub key_ids: Vec<KeyId>,
    /// Custom data.
    pub data: Vec<u8>,
}

impl FairPlayPssh {
    /// Create a new FairPlay PSSH.
    pub fn new(key_ids: Vec<KeyId>) -> Self {
        Self {
            version: 1,
            key_ids,
            data: Vec::new(),
        }
    }

    /// Create from a single key ID.
    pub fn from_key_id(key_id: KeyId) -> Self {
        Self::new(vec![key_id])
    }

    /// Set custom data.
    pub fn with_data(mut self, data: Vec<u8>) -> Self {
        self.data = data;
        self
    }

    /// Serialize to a complete PSSH box.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut box_data = Vec::new();

        // Version and flags
        box_data.push(self.version);
        box_data.extend_from_slice(&[0, 0, 0]); // flags

        // System ID
        box_data.extend_from_slice(FAIRPLAY_SYSTEM_ID.as_bytes());

        // Key IDs (version 1)
        if self.version >= 1 {
            box_data.extend_from_slice(&(self.key_ids.len() as u32).to_be_bytes());
            for key_id in &self.key_ids {
                box_data.extend_from_slice(key_id.as_bytes());
            }
        }

        // Data size and data
        box_data.extend_from_slice(&(self.data.len() as u32).to_be_bytes());
        box_data.extend_from_slice(&self.data);

        // Create full box
        let box_size = 8 + box_data.len();
        let mut full_box = Vec::with_capacity(box_size);
        full_box.extend_from_slice(&(box_size as u32).to_be_bytes());
        full_box.extend_from_slice(b"pssh");
        full_box.extend_from_slice(&box_data);

        Ok(full_box)
    }

    /// Serialize to base64.
    pub fn to_base64(&self) -> Result<String> {
        let bytes = self.to_bytes()?;
        Ok(BASE64.encode(&bytes))
    }

    /// Parse from PSSH box bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 32 {
            return Err(PsshError::InvalidFormat("PSSH box too small".into()).into());
        }

        let mut pos = 0;

        // Box size and type
        let _box_size = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;

        if &data[pos..pos + 4] != b"pssh" {
            return Err(PsshError::InvalidFormat("Not a PSSH box".into()).into());
        }
        pos += 4;

        // Version
        let version = data[pos];
        pos += 4; // Skip version + flags

        // System ID
        let system_id = Uuid::from_bytes(data[pos..pos + 16].try_into().unwrap());
        if system_id != FAIRPLAY_SYSTEM_ID {
            return Err(PsshError::UnknownSystemId(system_id.to_string()).into());
        }
        pos += 16;

        // Key IDs
        let mut key_ids = Vec::new();
        if version >= 1 && pos + 4 <= data.len() {
            let kid_count = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            for _ in 0..kid_count {
                if pos + 16 <= data.len() {
                    key_ids.push(KeyId::from_slice(&data[pos..pos + 16])?);
                    pos += 16;
                }
            }
        }

        // Data
        let mut pssh_data = Vec::new();
        if pos + 4 <= data.len() {
            let data_size = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + data_size <= data.len() {
                pssh_data = data[pos..pos + data_size].to_vec();
            }
        }

        Ok(Self {
            version,
            key_ids,
            data: pssh_data,
        })
    }
}

/// FairPlay encryption configuration for HLS.
#[derive(Clone, Debug)]
pub struct FairPlayConfig {
    /// Key ID.
    pub key_id: KeyId,
    /// Content key.
    pub key: ContentKey,
    /// Initialization vector.
    pub iv: Iv,
    /// Key server URL.
    pub key_server_url: String,
    /// Asset ID.
    pub asset_id: Option<String>,
}

impl FairPlayConfig {
    /// Create a new FairPlay configuration.
    pub fn new(
        key_id: KeyId,
        key: ContentKey,
        iv: Iv,
        key_server_url: impl Into<String>,
    ) -> Self {
        Self {
            key_id,
            key,
            iv,
            key_server_url: key_server_url.into(),
            asset_id: None,
        }
    }

    /// Set a custom asset ID.
    pub fn with_asset_id(mut self, asset_id: impl Into<String>) -> Self {
        self.asset_id = Some(asset_id.into());
        self
    }

    /// Get the encryption scheme.
    pub fn scheme(&self) -> EncryptionScheme {
        EncryptionScheme::Cbcs
    }

    /// Get the encryption pattern.
    pub fn pattern(&self) -> Pattern {
        Pattern {
            crypt_blocks: FAIRPLAY_CRYPT_BLOCKS,
            skip_blocks: FAIRPLAY_SKIP_BLOCKS,
        }
    }

    /// Create a FairPlay context from this config.
    pub fn to_context(&self) -> FairPlayContext {
        let mut ctx = FairPlayContext::new(
            self.key_id,
            self.iv.clone(),
            &self.key_server_url,
        );
        if let Some(ref asset_id) = self.asset_id {
            ctx = ctx.with_asset_id(asset_id);
        }
        ctx
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.key_server_url.is_empty() {
            return Err(DrmError::InvalidConfig("Key server URL is required".into()));
        }
        Ok(())
    }
}

/// Generate a FairPlay-compatible asset ID from content.
///
/// Creates a deterministic asset ID based on content metadata.
pub fn generate_asset_id(content_id: &str, variant: Option<&str>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content_id.as_bytes());
    if let Some(v) = variant {
        hasher.update(b":");
        hasher.update(v.as_bytes());
    }
    let hash = hasher.finalize();

    // Use first 16 bytes as a UUID-like identifier
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        hash[0], hash[1], hash[2], hash[3],
        hash[4], hash[5], hash[6], hash[7],
        hash[8], hash[9], hash[10], hash[11],
        hash[12], hash[13], hash[14], hash[15]
    )
}

/// Generate EXT-X-KEY tag for HLS playlist.
pub fn generate_ext_x_key(
    key_id: &KeyId,
    iv: &Iv,
    _key_server_url: &str,
) -> String {
    let uri = format!("skd://{}", key_id.to_hex());
    let iv_hex = iv.to_hex();

    format!(
        "#EXT-X-KEY:METHOD=SAMPLE-AES,URI=\"{}\",IV=0x{},KEYFORMAT=\"{}\",KEYFORMATVERSIONS=\"{}\"",
        uri, iv_hex, FAIRPLAY_KEY_FORMAT, FAIRPLAY_KEY_FORMAT_VERSION
    )
}

/// HLS playlist DRM information for FairPlay.
#[derive(Clone, Debug)]
pub struct HlsDrmInfo {
    /// EXT-X-KEY tag.
    pub ext_x_key: String,
    /// EXT-X-SESSION-KEY tag (for master playlist).
    pub ext_x_session_key: String,
    /// Key server URL.
    pub key_server_url: String,
    /// Asset ID.
    pub asset_id: String,
}

impl HlsDrmInfo {
    /// Create HLS DRM info from a FairPlay configuration.
    pub fn from_config(config: &FairPlayConfig) -> Self {
        let context = config.to_context();
        let asset_id = context.get_asset_id();
        let uri = format!("skd://{}", asset_id);
        let iv_hex = config.iv.to_hex();

        let ext_x_key = format!(
            "#EXT-X-KEY:METHOD=SAMPLE-AES,URI=\"{}\",IV=0x{},KEYFORMAT=\"{}\",KEYFORMATVERSIONS=\"{}\"",
            uri, iv_hex, FAIRPLAY_KEY_FORMAT, FAIRPLAY_KEY_FORMAT_VERSION
        );

        let ext_x_session_key = format!(
            "#EXT-X-SESSION-KEY:METHOD=SAMPLE-AES,URI=\"{}\",KEYFORMAT=\"{}\",KEYFORMATVERSIONS=\"{}\"",
            uri, FAIRPLAY_KEY_FORMAT, FAIRPLAY_KEY_FORMAT_VERSION
        );

        Self {
            ext_x_key,
            ext_x_session_key,
            key_server_url: config.key_server_url.clone(),
            asset_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key_id() -> KeyId {
        KeyId::parse("12345678-1234-1234-1234-123456789012").unwrap()
    }

    fn test_key() -> ContentKey {
        ContentKey::from_hex("00112233445566778899aabbccddeeff").unwrap()
    }

    fn test_iv() -> Iv {
        Iv::from_hex("00112233445566778899aabbccddeeff").unwrap()
    }

    #[test]
    fn test_fairplay_system_id() {
        assert_eq!(
            FAIRPLAY_SYSTEM_ID.to_string(),
            "94ce86fb-07ff-4f43-adb8-93d2fa968ca2"
        );
    }

    #[test]
    fn test_fairplay_context() {
        let key_id = test_key_id();
        let iv = test_iv();

        let context = FairPlayContext::new(key_id, iv, "https://fps.example.com/");

        assert_eq!(context.get_asset_id(), key_id.to_hex());
        assert!(context.key_uri().starts_with("skd://"));
        assert!(context.ext_x_key_tag().contains("SAMPLE-AES"));
        assert!(context.ext_x_key_tag().contains(FAIRPLAY_KEY_FORMAT));
    }

    #[test]
    fn test_fairplay_context_custom_asset_id() {
        let key_id = test_key_id();
        let iv = test_iv();

        let context = FairPlayContext::new(key_id, iv, "https://fps.example.com/")
            .with_asset_id("my-custom-asset");

        assert_eq!(context.get_asset_id(), "my-custom-asset");
        assert!(context.key_uri().contains("my-custom-asset"));
    }

    #[test]
    fn test_fairplay_pssh_roundtrip() {
        let key_id = test_key_id();
        let pssh = FairPlayPssh::from_key_id(key_id);

        let bytes = pssh.to_bytes().unwrap();
        let parsed = FairPlayPssh::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.key_ids.len(), 1);
        assert_eq!(parsed.key_ids[0], key_id);
    }

    #[test]
    fn test_fairplay_pssh_base64() {
        let key_id = test_key_id();
        let pssh = FairPlayPssh::from_key_id(key_id);

        let base64 = pssh.to_base64().unwrap();
        assert!(!base64.is_empty());
    }

    #[test]
    fn test_fairplay_config() {
        let key_id = test_key_id();
        let key = test_key();
        let iv = test_iv();

        let config = FairPlayConfig::new(key_id, key, iv, "https://fps.example.com/");

        assert_eq!(config.scheme(), EncryptionScheme::Cbcs);
        assert_eq!(config.pattern().crypt_blocks, 1);
        assert_eq!(config.pattern().skip_blocks, 9);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_fairplay_config_validation() {
        let key_id = test_key_id();
        let key = test_key();
        let iv = test_iv();

        let config = FairPlayConfig::new(key_id, key, iv, "");
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_generate_asset_id() {
        let asset_id1 = generate_asset_id("my-content", None);
        let asset_id2 = generate_asset_id("my-content", Some("720p"));
        let asset_id3 = generate_asset_id("my-content", None);

        // Same input = same output
        assert_eq!(asset_id1, asset_id3);

        // Different variant = different output
        assert_ne!(asset_id1, asset_id2);

        // Format check (UUID-like)
        assert_eq!(asset_id1.len(), 36);
        assert_eq!(asset_id1.chars().filter(|&c| c == '-').count(), 4);
    }

    #[test]
    fn test_generate_ext_x_key() {
        let key_id = test_key_id();
        let iv = test_iv();

        let tag = generate_ext_x_key(&key_id, &iv, "https://fps.example.com/");

        assert!(tag.starts_with("#EXT-X-KEY:"));
        assert!(tag.contains("METHOD=SAMPLE-AES"));
        assert!(tag.contains("URI=\"skd://"));
        assert!(tag.contains("IV=0x"));
        assert!(tag.contains(&format!("KEYFORMAT=\"{}\"", FAIRPLAY_KEY_FORMAT)));
    }

    #[test]
    fn test_hls_drm_info() {
        let key_id = test_key_id();
        let key = test_key();
        let iv = test_iv();

        let config = FairPlayConfig::new(key_id, key, iv, "https://fps.example.com/");
        let info = HlsDrmInfo::from_config(&config);

        assert!(info.ext_x_key.contains("METHOD=SAMPLE-AES"));
        assert!(info.ext_x_session_key.contains("EXT-X-SESSION-KEY"));
        assert_eq!(info.key_server_url, "https://fps.example.com/");
    }

    #[test]
    fn test_key_request() {
        let request = KeyRequest::new("asset123", "https://fps.example.com/")
            .with_session_id("session456");

        assert_eq!(request.asset_id, "asset123");
        assert_eq!(request.session_id, Some("session456".into()));
    }

    #[test]
    fn test_fairplay_pssh_with_data() {
        let key_id = test_key_id();
        let custom_data = b"custom-fairplay-data".to_vec();

        let pssh = FairPlayPssh::from_key_id(key_id)
            .with_data(custom_data.clone());

        let bytes = pssh.to_bytes().unwrap();
        let parsed = FairPlayPssh::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.data, custom_data);
    }
}
