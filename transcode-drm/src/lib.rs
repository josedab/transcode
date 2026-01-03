//! DRM and encryption support for the Transcode library.
//!
//! This crate provides comprehensive DRM (Digital Rights Management) and encryption
//! capabilities for media content protection, supporting industry-standard systems
//! including Widevine, PlayReady, and FairPlay Streaming.
//!
//! # Features
//!
//! - **Common Encryption (CENC)**: Full ISO/IEC 23001-7 implementation with all
//!   four encryption schemes (cenc, cbc1, cens, cbcs)
//! - **AES Encryption**: AES-128-CTR and AES-128-CBC encryption modes
//! - **Pattern Encryption**: CBCS pattern encryption for HLS/FairPlay
//! - **Multi-DRM Support**: Widevine, PlayReady, and FairPlay Streaming
//! - **PSSH Box Generation**: Protection System Specific Header for all DRM systems
//! - **Subsample Encryption**: NAL unit-aware encryption for video streams
//! - **Clear Lead**: Configurable unencrypted initial segments
//! - **Key Management**: Key ID and content key handling with key stores
//!
//! # Encryption Schemes
//!
//! | Scheme | Mode | Pattern | Use Case |
//! |--------|------|---------|----------|
//! | `cenc` | CTR | No | DASH (Widevine, PlayReady) |
//! | `cbc1` | CBC | No | Legacy systems |
//! | `cens` | CTR | Yes | Pattern-based CTR |
//! | `cbcs` | CBC | Yes | HLS (FairPlay) |
//!
//! # Example: Basic CENC Encryption
//!
//! ```rust
//! use transcode_drm::prelude::*;
//!
//! // Generate or provide keys
//! let key_id = KeyId::generate();
//! let key = ContentKey::generate();
//! let iv = Iv::generate();
//!
//! // Create CENC configuration for DASH
//! let config = CencConfig::for_dash(key_id, key.clone(), iv);
//!
//! // Create encryptor
//! let mut encryptor = CencEncryptor::new(config).unwrap();
//!
//! // Encrypt a sample
//! let mut sample_data = vec![0u8; 1024];
//! let info = encryptor.encrypt_sample(&mut sample_data).unwrap();
//! assert!(info.is_encrypted);
//! ```
//!
//! # Example: Multi-DRM PSSH Generation
//!
//! ```rust
//! use transcode_drm::prelude::*;
//!
//! let key_id = KeyId::generate();
//!
//! // Generate PSSH boxes for all DRM systems
//! let multi_drm = MultiDrm::new(key_id)
//!     .with_widevine(|w| w.with_provider("MyService"))
//!     .with_playready(|p| p.with_la_url("https://license.example.com/"))
//!     .with_fairplay(|f| f.with_key_server_url("https://fps.example.com/"));
//!
//! // Get PSSH boxes
//! let psshs = multi_drm.pssh_boxes().unwrap();
//! ```
//!
//! # Example: HLS with FairPlay
//!
//! ```rust
//! use transcode_drm::prelude::*;
//!
//! let key_id = KeyId::generate();
//! let key = ContentKey::generate();
//! let iv = Iv::generate();
//!
//! // Create FairPlay configuration
//! let config = FairPlayConfig::new(
//!     key_id,
//!     key,
//!     iv,
//!     "https://fps.example.com/",
//! );
//!
//! // Get HLS DRM information
//! let hls_info = HlsDrmInfo::from_config(&config);
//! println!("EXT-X-KEY: {}", hls_info.ext_x_key);
//! ```
//!
//! # Example: Subsample Encryption
//!
//! ```rust
//! use transcode_drm::prelude::*;
//!
//! let key = ContentKey::generate();
//! let iv = Iv::generate();
//!
//! // Create subsamples for NAL units
//! let nal_sizes = vec![100, 200, 150];
//! let subsamples = calculate_nal_subsamples(&nal_sizes, 1); // 1-byte NAL header
//!
//! // Encrypt with subsample encryption
//! let mut cipher = AesCtr::new(key, iv);
//! let mut data = vec![0u8; 450];
//! encrypt_subsamples(&mut cipher, &mut data, &subsamples).unwrap();
//! ```

//! Missing docs lint is warn, not deny, due to thiserror struct fields.
#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]

pub mod aes_ctr;
pub mod cbcs;
pub mod cenc;
pub mod error;
pub mod fairplay;
pub mod key;
pub mod playready;
pub mod widevine;

// Re-exports for convenience
pub use aes_ctr::{
    calculate_nal_subsamples, decrypt_subsamples, encrypt_subsamples, optimize_subsamples,
    AesCtr, SampleEncryptionInfo, SubsampleEntry,
};
pub use cbcs::{CbcsDecryptor, CbcsEncryptor, CbcsSubsample, Pattern};
pub use cenc::{
    CencConfig, CencDecryptor, CencEncryptor, EncryptionScheme, SampleEncryptionEntry,
    TrackEncryptionBox,
};
pub use error::{DrmError, EncryptionError, KeyError, PsshError, Result};
pub use fairplay::{FairPlayConfig, FairPlayContext, FairPlayPssh, HlsDrmInfo};
pub use key::{system_ids, ContentKey, Iv, KeyId, KeyPair, KeyStore};
pub use playready::{KeyRecord, PlayReadyAlgorithm, PlayReadyHeader, PlayReadyPssh};
pub use widevine::{Algorithm, ProtectionScheme, WidevineData, WidevinePssh};

/// Prelude for common imports.
pub mod prelude {
    pub use crate::aes_ctr::{
        calculate_nal_subsamples, decrypt_subsamples, encrypt_subsamples, optimize_subsamples,
        AesCtr, SampleEncryptionInfo, SubsampleEntry,
    };
    pub use crate::cbcs::{CbcsDecryptor, CbcsEncryptor, Pattern};
    pub use crate::cenc::{CencConfig, CencDecryptor, CencEncryptor, EncryptionScheme};
    pub use crate::error::{DrmError, Result};
    pub use crate::fairplay::{FairPlayConfig, FairPlayContext, HlsDrmInfo};
    pub use crate::key::{ContentKey, Iv, KeyId, KeyPair, KeyStore};
    pub use crate::playready::{PlayReadyHeader, PlayReadyPssh};
    pub use crate::widevine::{WidevineData, WidevinePssh};
    pub use crate::MultiDrm;
}

/// Multi-DRM packaging helper.
///
/// Simplifies creating PSSH boxes for multiple DRM systems.
#[derive(Clone, Debug)]
pub struct MultiDrm {
    /// Key ID.
    key_id: KeyId,
    /// Additional key IDs.
    additional_key_ids: Vec<KeyId>,
    /// Widevine configuration.
    widevine: Option<WidevineData>,
    /// PlayReady configuration.
    playready: Option<PlayReadyHeader>,
    /// FairPlay configuration.
    fairplay: Option<FairPlayContext>,
}

impl MultiDrm {
    /// Create a new multi-DRM configuration with a single key.
    pub fn new(key_id: KeyId) -> Self {
        Self {
            key_id,
            additional_key_ids: Vec::new(),
            widevine: None,
            playready: None,
            fairplay: None,
        }
    }

    /// Create with multiple keys.
    pub fn with_keys(key_ids: Vec<KeyId>) -> Result<Self> {
        if key_ids.is_empty() {
            return Err(DrmError::InvalidConfig("At least one key ID required".into()));
        }

        let key_id = key_ids[0];
        let additional_key_ids = key_ids[1..].to_vec();

        Ok(Self {
            key_id,
            additional_key_ids,
            widevine: None,
            playready: None,
            fairplay: None,
        })
    }

    /// Add an additional key ID.
    pub fn add_key_id(mut self, key_id: KeyId) -> Self {
        self.additional_key_ids.push(key_id);
        self
    }

    /// Get all key IDs.
    pub fn all_key_ids(&self) -> Vec<KeyId> {
        let mut ids = vec![self.key_id];
        ids.extend_from_slice(&self.additional_key_ids);
        ids
    }

    /// Configure Widevine DRM.
    pub fn with_widevine<F>(mut self, f: F) -> Self
    where
        F: FnOnce(WidevineData) -> WidevineData,
    {
        let data = WidevineData::with_key_ids(self.all_key_ids());
        self.widevine = Some(f(data));
        self
    }

    /// Configure PlayReady DRM.
    pub fn with_playready<F>(mut self, f: F) -> Self
    where
        F: FnOnce(PlayReadyHeader) -> PlayReadyHeader,
    {
        let key_records: Vec<KeyRecord> = self
            .all_key_ids()
            .into_iter()
            .map(|kid| KeyRecord::new(kid, PlayReadyAlgorithm::AesCtr))
            .collect();
        let header = PlayReadyHeader::with_keys(key_records);
        self.playready = Some(f(header));
        self
    }

    /// Configure FairPlay DRM.
    pub fn with_fairplay<F>(mut self, f: F) -> Self
    where
        F: FnOnce(FairPlayContextBuilder) -> FairPlayContextBuilder,
    {
        let builder = FairPlayContextBuilder::new(self.key_id);
        self.fairplay = Some(f(builder).build());
        self
    }

    /// Generate all configured PSSH boxes.
    pub fn pssh_boxes(&self) -> Result<PsshCollection> {
        let widevine = if let Some(ref data) = self.widevine {
            Some(WidevinePssh::new(data.clone()).to_bytes()?)
        } else {
            None
        };

        let playready = if let Some(ref header) = self.playready {
            Some(PlayReadyPssh::new(header.clone()).to_bytes()?)
        } else {
            None
        };

        let fairplay = if let Some(ref context) = self.fairplay {
            Some(FairPlayPssh::from_key_id(context.key_id).to_bytes()?)
        } else {
            None
        };

        Ok(PsshCollection {
            widevine,
            playready,
            fairplay,
        })
    }

    /// Generate combined PSSH data for muxing.
    pub fn combined_pssh(&self) -> Result<Vec<u8>> {
        let collection = self.pssh_boxes()?;
        let mut combined = Vec::new();

        if let Some(ref data) = collection.widevine {
            combined.extend_from_slice(data);
        }
        if let Some(ref data) = collection.playready {
            combined.extend_from_slice(data);
        }
        if let Some(ref data) = collection.fairplay {
            combined.extend_from_slice(data);
        }

        Ok(combined)
    }

    /// Check if Widevine is configured.
    pub fn has_widevine(&self) -> bool {
        self.widevine.is_some()
    }

    /// Check if PlayReady is configured.
    pub fn has_playready(&self) -> bool {
        self.playready.is_some()
    }

    /// Check if FairPlay is configured.
    pub fn has_fairplay(&self) -> bool {
        self.fairplay.is_some()
    }
}

/// Builder for FairPlay context in multi-DRM.
#[derive(Clone, Debug)]
pub struct FairPlayContextBuilder {
    key_id: KeyId,
    iv: Option<Iv>,
    key_server_url: String,
    asset_id: Option<String>,
}

impl FairPlayContextBuilder {
    /// Create a new builder.
    pub fn new(key_id: KeyId) -> Self {
        Self {
            key_id,
            iv: None,
            key_server_url: String::new(),
            asset_id: None,
        }
    }

    /// Set the IV.
    pub fn with_iv(mut self, iv: Iv) -> Self {
        self.iv = Some(iv);
        self
    }

    /// Set the key server URL.
    pub fn with_key_server_url(mut self, url: impl Into<String>) -> Self {
        self.key_server_url = url.into();
        self
    }

    /// Set the asset ID.
    pub fn with_asset_id(mut self, asset_id: impl Into<String>) -> Self {
        self.asset_id = Some(asset_id.into());
        self
    }

    /// Build the FairPlay context.
    pub fn build(self) -> FairPlayContext {
        let iv = self.iv.unwrap_or_else(Iv::zero);
        let mut ctx = FairPlayContext::new(self.key_id, iv, &self.key_server_url);
        if let Some(asset_id) = self.asset_id {
            ctx = ctx.with_asset_id(asset_id);
        }
        ctx
    }
}

/// Collection of PSSH boxes for different DRM systems.
#[derive(Clone, Debug, Default)]
pub struct PsshCollection {
    /// Widevine PSSH box bytes.
    pub widevine: Option<Vec<u8>>,
    /// PlayReady PSSH box bytes.
    pub playready: Option<Vec<u8>>,
    /// FairPlay PSSH box bytes.
    pub fairplay: Option<Vec<u8>>,
}

impl PsshCollection {
    /// Get all PSSH boxes as a combined byte vector.
    pub fn combined(&self) -> Vec<u8> {
        let mut combined = Vec::new();

        if let Some(ref data) = self.widevine {
            combined.extend_from_slice(data);
        }
        if let Some(ref data) = self.playready {
            combined.extend_from_slice(data);
        }
        if let Some(ref data) = self.fairplay {
            combined.extend_from_slice(data);
        }

        combined
    }

    /// Get the total number of PSSH boxes.
    pub fn count(&self) -> usize {
        let mut count = 0;
        if self.widevine.is_some() {
            count += 1;
        }
        if self.playready.is_some() {
            count += 1;
        }
        if self.fairplay.is_some() {
            count += 1;
        }
        count
    }

    /// Check if the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.widevine.is_none() && self.playready.is_none() && self.fairplay.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key_id() -> KeyId {
        KeyId::parse("12345678-1234-1234-1234-123456789012").unwrap()
    }

    #[test]
    fn test_multi_drm_basic() {
        let key_id = test_key_id();
        let multi = MultiDrm::new(key_id)
            .with_widevine(|w| w.with_provider("TestProvider"))
            .with_playready(|p| p.with_la_url("https://license.example.com/"));

        assert!(multi.has_widevine());
        assert!(multi.has_playready());
        assert!(!multi.has_fairplay());

        let psshs = multi.pssh_boxes().unwrap();
        assert!(psshs.widevine.is_some());
        assert!(psshs.playready.is_some());
        assert!(psshs.fairplay.is_none());
    }

    #[test]
    fn test_multi_drm_all_systems() {
        let key_id = test_key_id();
        let multi = MultiDrm::new(key_id)
            .with_widevine(|w| w)
            .with_playready(|p| p)
            .with_fairplay(|f| f.with_key_server_url("https://fps.example.com/"));

        let psshs = multi.pssh_boxes().unwrap();
        assert_eq!(psshs.count(), 3);
        assert!(!psshs.is_empty());
    }

    #[test]
    fn test_multi_drm_combined_pssh() {
        let key_id = test_key_id();
        let multi = MultiDrm::new(key_id)
            .with_widevine(|w| w)
            .with_playready(|p| p);

        let combined = multi.combined_pssh().unwrap();
        assert!(!combined.is_empty());

        // Should contain both PSSH boxes
        let psshs = multi.pssh_boxes().unwrap();
        let widevine_len = psshs.widevine.as_ref().unwrap().len();
        let playready_len = psshs.playready.as_ref().unwrap().len();
        assert_eq!(combined.len(), widevine_len + playready_len);
    }

    #[test]
    fn test_multi_drm_multiple_keys() {
        let key1 = KeyId::generate();
        let key2 = KeyId::generate();

        let multi = MultiDrm::with_keys(vec![key1, key2])
            .unwrap()
            .with_widevine(|w| w);

        assert_eq!(multi.all_key_ids().len(), 2);
    }

    #[test]
    fn test_pssh_collection() {
        let mut collection = PsshCollection::default();
        assert!(collection.is_empty());
        assert_eq!(collection.count(), 0);

        collection.widevine = Some(vec![1, 2, 3]);
        assert!(!collection.is_empty());
        assert_eq!(collection.count(), 1);

        collection.playready = Some(vec![4, 5, 6]);
        assert_eq!(collection.count(), 2);

        let combined = collection.combined();
        assert_eq!(combined, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_fairplay_builder() {
        let key_id = test_key_id();
        let iv = Iv::generate();

        let context = FairPlayContextBuilder::new(key_id)
            .with_iv(iv)
            .with_key_server_url("https://fps.example.com/")
            .with_asset_id("my-asset")
            .build();

        assert_eq!(context.key_id, key_id);
        assert_eq!(context.key_server_url, "https://fps.example.com/");
        assert_eq!(context.get_asset_id(), "my-asset");
    }

    #[test]
    fn test_prelude_imports() {
        // Verify prelude exports compile
        use crate::prelude::*;

        let _key_id = KeyId::generate();
        let _key = ContentKey::generate();
        let _iv = Iv::generate();
    }

    #[test]
    fn test_encryption_workflow() {
        let key_id = KeyId::generate();
        let key = ContentKey::generate();
        let iv = Iv::generate();

        // Create CENC configuration
        let config = CencConfig::for_dash(key_id, key.clone(), iv.clone());
        let mut encryptor = CencEncryptor::new(config).unwrap();

        // Encrypt sample
        let original = vec![0x42u8; 256];
        let mut data = original.clone();
        let info = encryptor.encrypt_sample(&mut data).unwrap();

        assert!(info.is_encrypted);
        assert_ne!(data, original);

        // Decrypt
        let mut key_store = KeyStore::new();
        key_store.add_key(key_id, key).unwrap();

        let mut decryptor = CencDecryptor::new(EncryptionScheme::Cenc, key_store);
        decryptor
            .decrypt_sample(&key_id, info.iv.as_ref().unwrap(), &mut data)
            .unwrap();

        assert_eq!(data, original);
    }
}
