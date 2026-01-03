//! Common Encryption (CENC) implementation - ISO/IEC 23001-7.
//!
//! This module provides the unified CENC API supporting all four encryption
//! schemes defined in the specification:
//! - `cenc`: AES-CTR mode, full sample or subsample encryption
//! - `cbc1`: AES-CBC mode, full sample encryption
//! - `cens`: AES-CTR mode with pattern encryption
//! - `cbcs`: AES-CBC mode with pattern encryption (for HLS/FairPlay)

use crate::aes_ctr::{AesCtr, SampleEncryptionInfo, SubsampleEntry};
use crate::cbcs::{CbcsDecryptor, CbcsEncryptor, Pattern};
use crate::error::{DrmError, EncryptionError, Result};
use crate::key::{ContentKey, Iv, KeyId, KeyStore};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Maximum sample size for encryption (100MB).
pub const MAX_SAMPLE_SIZE: usize = 100 * 1024 * 1024;

/// CENC encryption scheme.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EncryptionScheme {
    /// AES-CTR mode without pattern (ISO scheme type: 'cenc').
    Cenc,
    /// AES-CBC mode without pattern (ISO scheme type: 'cbc1').
    Cbc1,
    /// AES-CTR mode with pattern (ISO scheme type: 'cens').
    Cens,
    /// AES-CBC mode with pattern (ISO scheme type: 'cbcs').
    Cbcs,
}

impl EncryptionScheme {
    /// Get the four-character code for this scheme.
    pub fn fourcc(&self) -> [u8; 4] {
        match self {
            Self::Cenc => *b"cenc",
            Self::Cbc1 => *b"cbc1",
            Self::Cens => *b"cens",
            Self::Cbcs => *b"cbcs",
        }
    }

    /// Parse a scheme from its four-character code.
    pub fn from_fourcc(fourcc: &[u8; 4]) -> Option<Self> {
        match fourcc {
            b"cenc" => Some(Self::Cenc),
            b"cbc1" => Some(Self::Cbc1),
            b"cens" => Some(Self::Cens),
            b"cbcs" => Some(Self::Cbcs),
            _ => None,
        }
    }

    /// Check if this scheme uses CBC mode.
    pub fn uses_cbc(&self) -> bool {
        matches!(self, Self::Cbc1 | Self::Cbcs)
    }

    /// Check if this scheme uses CTR mode.
    pub fn uses_ctr(&self) -> bool {
        matches!(self, Self::Cenc | Self::Cens)
    }

    /// Check if this scheme uses pattern encryption.
    pub fn uses_pattern(&self) -> bool {
        matches!(self, Self::Cens | Self::Cbcs)
    }

    /// Get the scheme recommended for a given use case.
    pub fn for_hls() -> Self {
        Self::Cbcs
    }

    /// Get the scheme recommended for DASH.
    pub fn for_dash() -> Self {
        Self::Cenc
    }
}

impl fmt::Display for EncryptionScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cenc => write!(f, "cenc"),
            Self::Cbc1 => write!(f, "cbc1"),
            Self::Cens => write!(f, "cens"),
            Self::Cbcs => write!(f, "cbcs"),
        }
    }
}

impl std::str::FromStr for EncryptionScheme {
    type Err = DrmError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cenc" => Ok(Self::Cenc),
            "cbc1" => Ok(Self::Cbc1),
            "cens" => Ok(Self::Cens),
            "cbcs" => Ok(Self::Cbcs),
            _ => Err(DrmError::Encryption(EncryptionError::InvalidScheme(
                s.to_string(),
            ))),
        }
    }
}

/// Configuration for CENC encryption.
#[derive(Clone, Debug)]
pub struct CencConfig {
    /// Encryption scheme to use.
    pub scheme: EncryptionScheme,
    /// Key ID for the content key.
    pub key_id: KeyId,
    /// Content encryption key.
    pub key: ContentKey,
    /// Base initialization vector.
    pub iv: Iv,
    /// Pattern for pattern-based schemes (cens, cbcs).
    pub pattern: Option<Pattern>,
    /// Clear lead duration in seconds (unencrypted initial segment).
    pub clear_lead_seconds: f64,
}

impl CencConfig {
    /// Create a new CENC configuration.
    pub fn new(scheme: EncryptionScheme, key_id: KeyId, key: ContentKey, iv: Iv) -> Self {
        let pattern = if scheme.uses_pattern() {
            Some(Pattern::default_cbcs())
        } else {
            None
        };

        Self {
            scheme,
            key_id,
            key,
            iv,
            pattern,
            clear_lead_seconds: 0.0,
        }
    }

    /// Create configuration for HLS streaming (cbcs).
    pub fn for_hls(key_id: KeyId, key: ContentKey, iv: Iv) -> Self {
        Self::new(EncryptionScheme::Cbcs, key_id, key, iv)
    }

    /// Create configuration for DASH streaming (cenc).
    pub fn for_dash(key_id: KeyId, key: ContentKey, iv: Iv) -> Self {
        Self::new(EncryptionScheme::Cenc, key_id, key, iv)
    }

    /// Set the encryption pattern (for cens/cbcs schemes).
    pub fn with_pattern(mut self, pattern: Pattern) -> Self {
        self.pattern = Some(pattern);
        self
    }

    /// Set clear lead duration.
    pub fn with_clear_lead(mut self, seconds: f64) -> Self {
        self.clear_lead_seconds = seconds;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.scheme.uses_pattern() && self.pattern.is_none() {
            return Err(DrmError::InvalidConfig(format!(
                "Pattern required for {} scheme",
                self.scheme
            )));
        }

        if self.clear_lead_seconds < 0.0 {
            return Err(DrmError::Encryption(EncryptionError::InvalidClearLead(
                "Clear lead cannot be negative".into(),
            )));
        }

        Ok(())
    }
}

/// CENC sample encryptor.
///
/// Unified encryptor that supports all four CENC schemes.
pub struct CencEncryptor {
    /// Configuration.
    config: CencConfig,
    /// CTR mode encryptor (for cenc/cens).
    ctr_cipher: Option<AesCtr>,
    /// CBC mode encryptor (for cbc1/cbcs).
    cbc_encryptor: Option<CbcsEncryptor>,
    /// Sample counter for IV generation.
    sample_counter: u64,
    /// Elapsed time for clear lead tracking.
    elapsed_seconds: f64,
}

impl CencEncryptor {
    /// Create a new CENC encryptor.
    pub fn new(config: CencConfig) -> Result<Self> {
        config.validate()?;

        let ctr_cipher = if config.scheme.uses_ctr() {
            Some(AesCtr::new(config.key.clone(), config.iv.clone()))
        } else {
            None
        };

        let cbc_encryptor = if config.scheme.uses_cbc() {
            let pattern = config
                .pattern
                .unwrap_or_else(Pattern::full_encryption);
            Some(CbcsEncryptor::new(
                config.key.clone(),
                config.iv.clone(),
                pattern,
            ))
        } else {
            None
        };

        Ok(Self {
            config,
            ctr_cipher,
            cbc_encryptor,
            sample_counter: 0,
            elapsed_seconds: 0.0,
        })
    }

    /// Check if a sample should be encrypted based on clear lead.
    pub fn should_encrypt(&self) -> bool {
        self.elapsed_seconds >= self.config.clear_lead_seconds
    }

    /// Update elapsed time for clear lead tracking.
    pub fn add_sample_duration(&mut self, duration_seconds: f64) {
        self.elapsed_seconds += duration_seconds;
    }

    /// Encrypt a sample in place.
    ///
    /// # Errors
    ///
    /// Returns an error if the sample size exceeds `MAX_SAMPLE_SIZE`.
    pub fn encrypt_sample(&mut self, data: &mut [u8]) -> Result<SampleEncryptionInfo> {
        // Validate sample size before processing
        if data.len() > MAX_SAMPLE_SIZE {
            return Err(DrmError::Encryption(EncryptionError::SampleTooLarge {
                size: data.len(),
                max: MAX_SAMPLE_SIZE,
            }));
        }

        if !self.should_encrypt() {
            self.sample_counter += 1;
            return Ok(SampleEncryptionInfo::clear());
        }

        let iv = self.generate_sample_iv();

        match self.config.scheme {
            EncryptionScheme::Cenc | EncryptionScheme::Cens => {
                if let Some(ref mut cipher) = self.ctr_cipher {
                    cipher.set_iv(iv.clone());
                    cipher.encrypt(data);
                }
            }
            EncryptionScheme::Cbc1 | EncryptionScheme::Cbcs => {
                if let Some(ref encryptor) = self.cbc_encryptor {
                    // For full sample encryption, ensure block alignment
                    let aligned_len = (data.len() / 16) * 16;
                    if aligned_len > 0 {
                        let cbc_encryptor = CbcsEncryptor::new(
                            self.config.key.clone(),
                            iv.clone(),
                            encryptor.pattern(),
                        );
                        cbc_encryptor.encrypt(&mut data[..aligned_len])?;
                    }
                }
            }
        }

        self.sample_counter += 1;
        Ok(SampleEncryptionInfo::full_sample(iv))
    }

    /// Encrypt a sample with subsample encryption.
    pub fn encrypt_sample_with_subsamples(
        &mut self,
        data: &mut [u8],
        subsamples: &[SubsampleEntry],
    ) -> Result<SampleEncryptionInfo> {
        if !self.should_encrypt() {
            self.sample_counter += 1;
            return Ok(SampleEncryptionInfo::clear());
        }

        let iv = self.generate_sample_iv();

        match self.config.scheme {
            EncryptionScheme::Cenc | EncryptionScheme::Cens => {
                if let Some(ref mut cipher) = self.ctr_cipher {
                    cipher.set_iv(iv.clone());
                    crate::aes_ctr::encrypt_subsamples(cipher, data, subsamples)?;
                }
            }
            EncryptionScheme::Cbc1 | EncryptionScheme::Cbcs => {
                // For CBC subsample encryption, handle each subsample
                let mut offset = 0;
                for subsample in subsamples {
                    let clear_bytes = subsample.bytes_of_clear_data as usize;
                    let encrypted_bytes = subsample.bytes_of_encrypted_data as usize;

                    if encrypted_bytes > 0 {
                        let start = offset + clear_bytes;
                        let end = start + encrypted_bytes;
                        let aligned_len = (encrypted_bytes / 16) * 16;

                        if aligned_len > 0 && end <= data.len() {
                            let pattern = self.config.pattern.unwrap_or_else(Pattern::full_encryption);
                            let cbc_encryptor = CbcsEncryptor::new(
                                self.config.key.clone(),
                                iv.clone(),
                                pattern,
                            );
                            cbc_encryptor.encrypt(&mut data[start..start + aligned_len])?;
                        }
                    }

                    offset += clear_bytes + encrypted_bytes;
                }
            }
        }

        self.sample_counter += 1;
        Ok(SampleEncryptionInfo::with_subsamples(iv, subsamples.to_vec()))
    }

    /// Generate IV for the current sample.
    fn generate_sample_iv(&self) -> Iv {
        let mut iv = self.config.iv.clone();

        if self.config.scheme.uses_ctr() {
            // For CTR modes, the counter can be incremented per sample
            // The IV format is: 8-byte nonce + 8-byte counter
            iv.set_counter(self.sample_counter);
        }
        // For CBC modes, we use the same constant IV (as per CBCS spec)

        iv
    }

    /// Get the configuration.
    pub fn config(&self) -> &CencConfig {
        &self.config
    }

    /// Get the key ID.
    pub fn key_id(&self) -> &KeyId {
        &self.config.key_id
    }

    /// Get the current sample count.
    pub fn sample_count(&self) -> u64 {
        self.sample_counter
    }

    /// Reset the encryptor state.
    pub fn reset(&mut self) {
        self.sample_counter = 0;
        self.elapsed_seconds = 0.0;
        if let Some(ref mut cipher) = self.ctr_cipher {
            cipher.reset();
        }
    }
}

/// CENC sample decryptor.
pub struct CencDecryptor {
    /// Encryption scheme.
    scheme: EncryptionScheme,
    /// Key store for looking up keys.
    key_store: KeyStore,
    /// Pattern for pattern-based schemes.
    pattern: Option<Pattern>,
}

impl CencDecryptor {
    /// Create a new CENC decryptor.
    pub fn new(scheme: EncryptionScheme, key_store: KeyStore) -> Self {
        let pattern = if scheme.uses_pattern() {
            Some(Pattern::default_cbcs())
        } else {
            None
        };

        Self {
            scheme,
            key_store,
            pattern,
        }
    }

    /// Set the encryption pattern.
    pub fn with_pattern(mut self, pattern: Pattern) -> Self {
        self.pattern = Some(pattern);
        self
    }

    /// Decrypt a sample in place.
    pub fn decrypt_sample(
        &mut self,
        key_id: &KeyId,
        iv: &Iv,
        data: &mut [u8],
    ) -> Result<()> {
        let key = self.key_store.get_key(key_id)?;

        match self.scheme {
            EncryptionScheme::Cenc | EncryptionScheme::Cens => {
                let mut cipher = AesCtr::new(key.clone(), iv.clone());
                cipher.decrypt(data);
            }
            EncryptionScheme::Cbc1 | EncryptionScheme::Cbcs => {
                let aligned_len = (data.len() / 16) * 16;
                if aligned_len > 0 {
                    let pattern = self.pattern.unwrap_or_else(Pattern::full_encryption);
                    let decryptor = CbcsDecryptor::new(key.clone(), iv.clone(), pattern);
                    decryptor.decrypt(&mut data[..aligned_len])?;
                }
            }
        }

        Ok(())
    }

    /// Decrypt a sample with subsample encryption.
    pub fn decrypt_sample_with_subsamples(
        &mut self,
        key_id: &KeyId,
        iv: &Iv,
        data: &mut [u8],
        subsamples: &[SubsampleEntry],
    ) -> Result<()> {
        let key = self.key_store.get_key(key_id)?;

        match self.scheme {
            EncryptionScheme::Cenc | EncryptionScheme::Cens => {
                let mut cipher = AesCtr::new(key.clone(), iv.clone());
                crate::aes_ctr::decrypt_subsamples(&mut cipher, data, subsamples)?;
            }
            EncryptionScheme::Cbc1 | EncryptionScheme::Cbcs => {
                let mut offset = 0;
                for subsample in subsamples {
                    let clear_bytes = subsample.bytes_of_clear_data as usize;
                    let encrypted_bytes = subsample.bytes_of_encrypted_data as usize;

                    if encrypted_bytes > 0 {
                        let start = offset + clear_bytes;
                        let aligned_len = (encrypted_bytes / 16) * 16;

                        if aligned_len > 0 {
                            let pattern = self.pattern.unwrap_or_else(Pattern::full_encryption);
                            let decryptor = CbcsDecryptor::new(key.clone(), iv.clone(), pattern);
                            decryptor.decrypt(&mut data[start..start + aligned_len])?;
                        }
                    }

                    offset += clear_bytes + encrypted_bytes;
                }
            }
        }

        Ok(())
    }

    /// Get the encryption scheme.
    pub fn scheme(&self) -> EncryptionScheme {
        self.scheme
    }

    /// Add a key to the key store.
    pub fn add_key(&mut self, key_id: KeyId, key: ContentKey) -> Result<()> {
        self.key_store.add_key(key_id, key)
    }
}

/// Track encryption information for muxing.
#[derive(Clone, Debug)]
pub struct TrackEncryptionBox {
    /// Default "is encrypted" flag.
    pub default_is_protected: bool,
    /// Per-sample IV size (0, 8, or 16 bytes).
    pub default_per_sample_iv_size: u8,
    /// Key ID for this track.
    pub default_kid: KeyId,
    /// Default constant IV (for cbcs scheme with 0 per-sample IV size).
    pub default_constant_iv: Option<Iv>,
    /// Pattern for pattern-based encryption.
    pub default_crypt_byte_block: u8,
    /// Pattern for pattern-based encryption.
    pub default_skip_byte_block: u8,
}

impl TrackEncryptionBox {
    /// Create a track encryption box for cenc scheme.
    pub fn for_cenc(key_id: KeyId) -> Self {
        Self {
            default_is_protected: true,
            default_per_sample_iv_size: 8,
            default_kid: key_id,
            default_constant_iv: None,
            default_crypt_byte_block: 0,
            default_skip_byte_block: 0,
        }
    }

    /// Create a track encryption box for cbcs scheme.
    pub fn for_cbcs(key_id: KeyId, constant_iv: Iv) -> Self {
        Self {
            default_is_protected: true,
            default_per_sample_iv_size: 0, // Use constant IV
            default_kid: key_id,
            default_constant_iv: Some(constant_iv),
            default_crypt_byte_block: 1,
            default_skip_byte_block: 9,
        }
    }

    /// Create with custom pattern.
    pub fn with_pattern(mut self, crypt: u8, skip: u8) -> Self {
        self.default_crypt_byte_block = crypt;
        self.default_skip_byte_block = skip;
        self
    }

    /// Serialize to bytes for muxing.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(32);

        // Version and flags
        bytes.push(1); // version
        bytes.extend_from_slice(&[0, 0, 0]); // flags

        // Reserved + pattern + crypt/skip
        bytes.push(0); // reserved
        bytes.push((self.default_crypt_byte_block << 4) | self.default_skip_byte_block);
        bytes.push(if self.default_is_protected { 1 } else { 0 });
        bytes.push(self.default_per_sample_iv_size);

        // Key ID
        bytes.extend_from_slice(self.default_kid.as_bytes());

        // Constant IV (if per-sample IV size is 0)
        if self.default_per_sample_iv_size == 0 {
            if let Some(ref iv) = self.default_constant_iv {
                bytes.push(16); // constant IV size
                bytes.extend_from_slice(iv.as_bytes());
            }
        }

        bytes
    }
}

/// Sample encryption entry for 'senc' box.
#[derive(Clone, Debug)]
pub struct SampleEncryptionEntry {
    /// Per-sample IV (8 or 16 bytes, or empty if using constant IV).
    pub iv: Vec<u8>,
    /// Subsample encryption entries.
    pub subsamples: Vec<SubsampleEntry>,
}

impl SampleEncryptionEntry {
    /// Create an entry with only IV.
    pub fn new(iv: Iv, iv_size: u8) -> Self {
        let iv_bytes = match iv_size {
            8 => iv.as_bytes()[..8].to_vec(),
            16 => iv.as_bytes().to_vec(),
            _ => Vec::new(),
        };

        Self {
            iv: iv_bytes,
            subsamples: Vec::new(),
        }
    }

    /// Create an entry with IV and subsamples.
    pub fn with_subsamples(iv: Iv, iv_size: u8, subsamples: Vec<SubsampleEntry>) -> Self {
        let iv_bytes = match iv_size {
            8 => iv.as_bytes()[..8].to_vec(),
            16 => iv.as_bytes().to_vec(),
            _ => Vec::new(),
        };

        Self {
            iv: iv_bytes,
            subsamples,
        }
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self, has_subsamples: bool) -> Vec<u8> {
        let mut bytes = Vec::new();

        // IV
        bytes.extend_from_slice(&self.iv);

        // Subsamples
        if has_subsamples {
            bytes.extend_from_slice(&(self.subsamples.len() as u16).to_be_bytes());
            for subsample in &self.subsamples {
                bytes.extend_from_slice(&subsample.bytes_of_clear_data.to_be_bytes()[2..]);
                bytes.extend_from_slice(&subsample.bytes_of_encrypted_data.to_be_bytes());
            }
        }

        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key_pair() -> (KeyId, ContentKey) {
        let key_id = KeyId::parse("12345678-1234-1234-1234-123456789012").unwrap();
        let key = ContentKey::from_hex("00112233445566778899aabbccddeeff").unwrap();
        (key_id, key)
    }

    fn test_iv() -> Iv {
        Iv::from_hex("00000000000000000000000000000000").unwrap()
    }

    #[test]
    fn test_encryption_scheme_fourcc() {
        assert_eq!(EncryptionScheme::Cenc.fourcc(), *b"cenc");
        assert_eq!(EncryptionScheme::Cbcs.fourcc(), *b"cbcs");

        assert_eq!(
            EncryptionScheme::from_fourcc(b"cenc"),
            Some(EncryptionScheme::Cenc)
        );
        assert_eq!(
            EncryptionScheme::from_fourcc(b"cbcs"),
            Some(EncryptionScheme::Cbcs)
        );
        assert_eq!(EncryptionScheme::from_fourcc(b"xxxx"), None);
    }

    #[test]
    fn test_encryption_scheme_properties() {
        assert!(EncryptionScheme::Cenc.uses_ctr());
        assert!(!EncryptionScheme::Cenc.uses_cbc());
        assert!(!EncryptionScheme::Cenc.uses_pattern());

        assert!(!EncryptionScheme::Cbcs.uses_ctr());
        assert!(EncryptionScheme::Cbcs.uses_cbc());
        assert!(EncryptionScheme::Cbcs.uses_pattern());
    }

    #[test]
    fn test_cenc_config_validation() {
        let (key_id, key) = test_key_pair();
        let iv = test_iv();

        // Valid cenc config
        let config = CencConfig::new(EncryptionScheme::Cenc, key_id, key.clone(), iv.clone());
        assert!(config.validate().is_ok());

        // Valid cbcs config (pattern automatically set)
        let config = CencConfig::new(EncryptionScheme::Cbcs, key_id, key, iv);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_cenc_encryptor_cenc_scheme() {
        let (key_id, key) = test_key_pair();
        let iv = test_iv();

        let config = CencConfig::new(EncryptionScheme::Cenc, key_id, key.clone(), iv.clone());
        let mut encryptor = CencEncryptor::new(config).unwrap();

        let original = vec![0u8; 100];
        let mut data = original.clone();

        let info = encryptor.encrypt_sample(&mut data).unwrap();

        assert!(info.is_encrypted);
        assert_ne!(data, original);

        // Decrypt
        let mut key_store = KeyStore::new();
        key_store.add_key(key_id, key).unwrap();

        let mut decryptor = CencDecryptor::new(EncryptionScheme::Cenc, key_store);
        decryptor.decrypt_sample(&key_id, info.iv.as_ref().unwrap(), &mut data).unwrap();

        assert_eq!(data, original);
    }

    #[test]
    fn test_cenc_encryptor_cbcs_scheme() {
        let (key_id, key) = test_key_pair();
        let iv = test_iv();

        let config = CencConfig::new(EncryptionScheme::Cbcs, key_id, key.clone(), iv.clone());
        let mut encryptor = CencEncryptor::new(config).unwrap();

        // 160 bytes = 10 blocks
        let original: Vec<u8> = (0..160).collect();
        let mut data = original.clone();

        let info = encryptor.encrypt_sample(&mut data).unwrap();

        assert!(info.is_encrypted);
        // With 1:9 pattern, only first block is encrypted
        assert_ne!(&data[..16], &original[..16]);
        assert_eq!(&data[16..160], &original[16..160]);

        // Decrypt
        let mut key_store = KeyStore::new();
        key_store.add_key(key_id, key).unwrap();

        let mut decryptor = CencDecryptor::new(EncryptionScheme::Cbcs, key_store);
        decryptor.decrypt_sample(&key_id, info.iv.as_ref().unwrap(), &mut data).unwrap();

        assert_eq!(data, original);
    }

    #[test]
    fn test_clear_lead() {
        let (key_id, key) = test_key_pair();
        let iv = test_iv();

        let config = CencConfig::new(EncryptionScheme::Cenc, key_id, key, iv)
            .with_clear_lead(2.0);

        let mut encryptor = CencEncryptor::new(config).unwrap();

        // First sample (before clear lead ends)
        let original = vec![1u8; 100];
        let mut data = original.clone();
        encryptor.add_sample_duration(1.0);

        let info = encryptor.encrypt_sample(&mut data).unwrap();
        assert!(!info.is_encrypted);
        assert_eq!(data, original);

        // Second sample (clear lead ends)
        encryptor.add_sample_duration(1.5);

        let info = encryptor.encrypt_sample(&mut data).unwrap();
        assert!(info.is_encrypted);
        assert_ne!(data, original);
    }

    #[test]
    fn test_subsample_encryption_cenc() {
        let (key_id, key) = test_key_pair();
        let iv = test_iv();

        let config = CencConfig::new(EncryptionScheme::Cenc, key_id, key.clone(), iv);
        let mut encryptor = CencEncryptor::new(config).unwrap();

        let original: Vec<u8> = (0..100).collect();
        let mut data = original.clone();

        let subsamples = vec![
            SubsampleEntry::new(10, 40),
            SubsampleEntry::new(5, 45),
        ];

        let info = encryptor
            .encrypt_sample_with_subsamples(&mut data, &subsamples)
            .unwrap();

        assert!(info.is_encrypted);
        assert!(info.uses_subsamples());

        // Clear regions unchanged
        assert_eq!(&data[..10], &original[..10]);
        assert_eq!(&data[50..55], &original[50..55]);

        // Encrypted regions changed
        assert_ne!(&data[10..50], &original[10..50]);
        assert_ne!(&data[55..100], &original[55..100]);

        // Decrypt
        let mut key_store = KeyStore::new();
        key_store.add_key(key_id, key).unwrap();

        let mut decryptor = CencDecryptor::new(EncryptionScheme::Cenc, key_store);
        decryptor
            .decrypt_sample_with_subsamples(&key_id, info.iv.as_ref().unwrap(), &mut data, &subsamples)
            .unwrap();

        assert_eq!(data, original);
    }

    #[test]
    fn test_track_encryption_box_cenc() {
        let key_id = KeyId::parse("12345678-1234-1234-1234-123456789012").unwrap();
        let tenc = TrackEncryptionBox::for_cenc(key_id);

        assert!(tenc.default_is_protected);
        assert_eq!(tenc.default_per_sample_iv_size, 8);
        assert!(tenc.default_constant_iv.is_none());

        let bytes = tenc.to_bytes();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_track_encryption_box_cbcs() {
        let key_id = KeyId::parse("12345678-1234-1234-1234-123456789012").unwrap();
        let iv = test_iv();
        let tenc = TrackEncryptionBox::for_cbcs(key_id, iv);

        assert!(tenc.default_is_protected);
        assert_eq!(tenc.default_per_sample_iv_size, 0);
        assert!(tenc.default_constant_iv.is_some());
        assert_eq!(tenc.default_crypt_byte_block, 1);
        assert_eq!(tenc.default_skip_byte_block, 9);

        let bytes = tenc.to_bytes();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_sample_encryption_entry() {
        let iv = test_iv();
        let subsamples = vec![SubsampleEntry::new(10, 90)];

        let entry = SampleEncryptionEntry::with_subsamples(iv, 8, subsamples);

        assert_eq!(entry.iv.len(), 8);
        assert_eq!(entry.subsamples.len(), 1);

        let bytes = entry.to_bytes(true);
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_scheme_recommendations() {
        assert_eq!(EncryptionScheme::for_hls(), EncryptionScheme::Cbcs);
        assert_eq!(EncryptionScheme::for_dash(), EncryptionScheme::Cenc);
    }
}
