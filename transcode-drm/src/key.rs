//! Key management for DRM operations.
//!
//! This module provides key ID and content key handling, key derivation,
//! and key storage for multi-DRM encryption workflows.
//!
//! # Security
//!
//! All key material is zeroized on drop to prevent sensitive data from
//! remaining in memory after use.

use crate::error::{KeyError, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;
use uuid::Uuid;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Size of AES-128 key in bytes.
pub const AES_128_KEY_SIZE: usize = 16;

/// Size of initialization vector in bytes.
pub const IV_SIZE: usize = 16;

/// Size of counter for CTR mode (8 bytes).
pub const CTR_COUNTER_SIZE: usize = 8;

/// Content encryption key for DRM.
///
/// # Security
///
/// The key material is automatically zeroized when the struct is dropped
/// to prevent sensitive data from remaining in memory.
#[derive(Clone, PartialEq, Eq, Zeroize, ZeroizeOnDrop)]
pub struct ContentKey {
    /// Raw key bytes (16 bytes for AES-128).
    key: [u8; AES_128_KEY_SIZE],
}

impl ContentKey {
    /// Create a new content key from raw bytes.
    pub fn new(key: [u8; AES_128_KEY_SIZE]) -> Self {
        Self { key }
    }

    /// Create a content key from a byte slice.
    pub fn from_slice(slice: &[u8]) -> Result<Self> {
        if slice.len() != AES_128_KEY_SIZE {
            return Err(KeyError::InvalidKeyLength {
                expected: AES_128_KEY_SIZE,
                actual: slice.len(),
            }
            .into());
        }
        let mut key = [0u8; AES_128_KEY_SIZE];
        key.copy_from_slice(slice);
        Ok(Self { key })
    }

    /// Create a content key from base64-encoded string.
    pub fn from_base64(encoded: &str) -> Result<Self> {
        let decoded = BASE64
            .decode(encoded)
            .map_err(|e| KeyError::InvalidBase64(e.to_string()))?;
        Self::from_slice(&decoded)
    }

    /// Create a content key from hex-encoded string.
    pub fn from_hex(hex: &str) -> Result<Self> {
        let hex = hex.replace(['-', ' '], "");
        if hex.len() != AES_128_KEY_SIZE * 2 {
            return Err(KeyError::InvalidKeyLength {
                expected: AES_128_KEY_SIZE * 2,
                actual: hex.len(),
            }
            .into());
        }

        let mut key = [0u8; AES_128_KEY_SIZE];
        for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
            let hex_str = std::str::from_utf8(chunk)
                .map_err(|_| KeyError::InvalidBase64("Invalid hex string".into()))?;
            key[i] = u8::from_str_radix(hex_str, 16)
                .map_err(|_| KeyError::InvalidBase64("Invalid hex character".into()))?;
        }
        Ok(Self { key })
    }

    /// Generate a random content key.
    pub fn generate() -> Self {
        use rand::RngCore;
        let mut key = [0u8; AES_128_KEY_SIZE];
        rand::thread_rng().fill_bytes(&mut key);
        Self { key }
    }

    /// Derive a content key from a key ID and a master key.
    pub fn derive(key_id: &KeyId, master_key: &ContentKey) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(master_key.as_bytes());
        hasher.update(key_id.as_bytes());
        let hash = hasher.finalize();

        let mut key = [0u8; AES_128_KEY_SIZE];
        key.copy_from_slice(&hash[..AES_128_KEY_SIZE]);
        Self { key }
    }

    /// Get the raw key bytes.
    pub fn as_bytes(&self) -> &[u8; AES_128_KEY_SIZE] {
        &self.key
    }

    /// Encode the key as base64.
    pub fn to_base64(&self) -> String {
        BASE64.encode(self.key)
    }

    /// Encode the key as hexadecimal.
    pub fn to_hex(&self) -> String {
        self.key.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

impl fmt::Debug for ContentKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Redact key value for security
        write!(f, "ContentKey([REDACTED])")
    }
}

impl fmt::Display for ContentKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Show only first 4 characters of hex for identification
        write!(f, "ContentKey({}...)", &self.to_hex()[..4])
    }
}

/// Key identifier (UUID-based).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KeyId(Uuid);

impl KeyId {
    /// Create a key ID from a UUID.
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Create a key ID from raw bytes (16 bytes).
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(Uuid::from_bytes(bytes))
    }

    /// Create a key ID from a byte slice.
    pub fn from_slice(slice: &[u8]) -> Result<Self> {
        if slice.len() != 16 {
            return Err(KeyError::InvalidKeyId(format!(
                "Key ID must be 16 bytes, got {}",
                slice.len()
            ))
            .into());
        }
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(slice);
        Ok(Self::from_bytes(bytes))
    }

    /// Parse a key ID from a UUID string.
    pub fn parse(s: &str) -> Result<Self> {
        let uuid = Uuid::parse_str(s).map_err(|e| KeyError::InvalidKeyId(e.to_string()))?;
        Ok(Self(uuid))
    }

    /// Parse a key ID from hex (with or without dashes).
    pub fn from_hex(hex: &str) -> Result<Self> {
        let hex = hex.replace('-', "");
        if hex.len() != 32 {
            return Err(KeyError::InvalidKeyId(format!(
                "Key ID hex must be 32 characters, got {}",
                hex.len()
            ))
            .into());
        }

        let mut bytes = [0u8; 16];
        for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
            let hex_str = std::str::from_utf8(chunk)
                .map_err(|_| KeyError::InvalidKeyId("Invalid hex string".into()))?;
            bytes[i] = u8::from_str_radix(hex_str, 16)
                .map_err(|_| KeyError::InvalidKeyId("Invalid hex character".into()))?;
        }
        Ok(Self::from_bytes(bytes))
    }

    /// Create a key ID from base64-encoded string.
    pub fn from_base64(encoded: &str) -> Result<Self> {
        let decoded = BASE64
            .decode(encoded)
            .map_err(|e| KeyError::InvalidBase64(e.to_string()))?;
        Self::from_slice(&decoded)
    }

    /// Generate a random key ID.
    pub fn generate() -> Self {
        Self(Uuid::new_v4())
    }

    /// Get the raw bytes of the key ID.
    pub fn as_bytes(&self) -> &[u8; 16] {
        self.0.as_bytes()
    }

    /// Get the UUID.
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }

    /// Format as a standard UUID string (with dashes).
    pub fn to_uuid_string(&self) -> String {
        self.0.to_string()
    }

    /// Format as hexadecimal (without dashes).
    pub fn to_hex(&self) -> String {
        self.as_bytes().iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Encode as base64.
    pub fn to_base64(&self) -> String {
        BASE64.encode(self.as_bytes())
    }

    /// Convert to big-endian byte order (for some DRM systems).
    pub fn to_big_endian_bytes(&self) -> [u8; 16] {
        *self.as_bytes()
    }

    /// Convert to little-endian GUID format (for PlayReady).
    pub fn to_little_endian_bytes(&self) -> [u8; 16] {
        let bytes = self.as_bytes();
        [
            bytes[3], bytes[2], bytes[1], bytes[0], // Swap first 4 bytes
            bytes[5], bytes[4], // Swap next 2 bytes
            bytes[7], bytes[6], // Swap next 2 bytes
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]
    }
}

impl fmt::Debug for KeyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KeyId({})", self.0)
    }
}

impl fmt::Display for KeyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Initialization vector for encryption.
///
/// # Security
///
/// The IV bytes are automatically zeroized when the struct is dropped.
#[derive(Clone, PartialEq, Eq, Zeroize, ZeroizeOnDrop)]
pub struct Iv {
    /// IV bytes (16 bytes).
    bytes: [u8; IV_SIZE],
}

impl Iv {
    /// Create an IV from raw bytes.
    pub fn new(bytes: [u8; IV_SIZE]) -> Self {
        Self { bytes }
    }

    /// Create an IV from a byte slice.
    pub fn from_slice(slice: &[u8]) -> Result<Self> {
        if slice.len() > IV_SIZE {
            return Err(KeyError::InvalidIvLength {
                expected: IV_SIZE,
                actual: slice.len(),
            }
            .into());
        }

        let mut bytes = [0u8; IV_SIZE];
        // If shorter than 16 bytes, zero-pad on the right
        bytes[..slice.len()].copy_from_slice(slice);
        Ok(Self { bytes })
    }

    /// Create an IV from an 8-byte value (zero-padded).
    pub fn from_8_bytes(short_iv: [u8; 8]) -> Self {
        let mut bytes = [0u8; IV_SIZE];
        bytes[..8].copy_from_slice(&short_iv);
        Self { bytes }
    }

    /// Create an IV from hex string.
    pub fn from_hex(hex: &str) -> Result<Self> {
        let hex = hex.replace(['-', ' '], "");
        if hex.len() > IV_SIZE * 2 {
            return Err(KeyError::InvalidIvLength {
                expected: IV_SIZE,
                actual: hex.len() / 2,
            }
            .into());
        }

        let mut bytes = [0u8; IV_SIZE];
        for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
            let hex_str = std::str::from_utf8(chunk)
                .map_err(|_| KeyError::InvalidBase64("Invalid hex string".into()))?;
            bytes[i] = u8::from_str_radix(hex_str, 16)
                .map_err(|_| KeyError::InvalidBase64("Invalid hex character".into()))?;
        }
        Ok(Self { bytes })
    }

    /// Generate a random IV.
    pub fn generate() -> Self {
        use rand::RngCore;
        let mut bytes = [0u8; IV_SIZE];
        rand::thread_rng().fill_bytes(&mut bytes);
        Self { bytes }
    }

    /// Create a zero IV.
    pub fn zero() -> Self {
        Self {
            bytes: [0u8; IV_SIZE],
        }
    }

    /// Get the raw IV bytes.
    pub fn as_bytes(&self) -> &[u8; IV_SIZE] {
        &self.bytes
    }

    /// Get the first 8 bytes (for CTR counter).
    pub fn as_8_bytes(&self) -> [u8; 8] {
        let mut result = [0u8; 8];
        result.copy_from_slice(&self.bytes[..8]);
        result
    }

    /// Format as hexadecimal.
    pub fn to_hex(&self) -> String {
        self.bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Increment the IV counter (last 8 bytes as big-endian).
    pub fn increment(&mut self, amount: u64) {
        // SAFETY: bytes[8..16] is always exactly 8 bytes
        let counter_bytes: [u8; 8] = [
            self.bytes[8], self.bytes[9], self.bytes[10], self.bytes[11],
            self.bytes[12], self.bytes[13], self.bytes[14], self.bytes[15],
        ];
        let counter = u64::from_be_bytes(counter_bytes);
        let new_counter = counter.wrapping_add(amount);
        self.bytes[8..].copy_from_slice(&new_counter.to_be_bytes());
    }

    /// Get current counter value (last 8 bytes as big-endian u64).
    pub fn counter(&self) -> u64 {
        // SAFETY: bytes[8..16] is always exactly 8 bytes
        let counter_bytes: [u8; 8] = [
            self.bytes[8], self.bytes[9], self.bytes[10], self.bytes[11],
            self.bytes[12], self.bytes[13], self.bytes[14], self.bytes[15],
        ];
        u64::from_be_bytes(counter_bytes)
    }

    /// Set counter value (last 8 bytes).
    pub fn set_counter(&mut self, counter: u64) {
        self.bytes[8..].copy_from_slice(&counter.to_be_bytes());
    }
}

impl fmt::Debug for Iv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Iv({})", self.to_hex())
    }
}

/// Key-value pair for encryption.
///
/// # Security
///
/// The content key is automatically zeroized when the struct is dropped.
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct KeyPair {
    /// Key identifier.
    #[zeroize(skip)]
    pub key_id: KeyId,
    /// Content encryption key.
    pub key: ContentKey,
}

impl KeyPair {
    /// Create a new key pair.
    pub fn new(key_id: KeyId, key: ContentKey) -> Self {
        Self { key_id, key }
    }

    /// Generate a random key pair.
    pub fn generate() -> Self {
        Self {
            key_id: KeyId::generate(),
            key: ContentKey::generate(),
        }
    }
}

impl fmt::Debug for KeyPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KeyPair")
            .field("key_id", &self.key_id)
            .field("key", &"[REDACTED]")
            .finish()
    }
}

/// Key store for managing multiple keys.
///
/// # Security
///
/// All keys are automatically zeroized when the store is dropped or cleared.
#[derive(Default)]
pub struct KeyStore {
    /// Keys indexed by key ID.
    keys: HashMap<KeyId, ContentKey>,
    /// Master key for key derivation.
    master_key: Option<ContentKey>,
}

impl Drop for KeyStore {
    fn drop(&mut self) {
        // Clear calls zeroize on each key via ContentKey's ZeroizeOnDrop
        self.keys.clear();
        // master_key's ZeroizeOnDrop handles cleanup
    }
}

impl KeyStore {
    /// Create a new empty key store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a key store with a master key for derivation.
    pub fn with_master_key(master_key: ContentKey) -> Self {
        Self {
            keys: HashMap::new(),
            master_key: Some(master_key),
        }
    }

    /// Add a key to the store.
    pub fn add_key(&mut self, key_id: KeyId, key: ContentKey) -> Result<()> {
        if self.keys.contains_key(&key_id) {
            return Err(KeyError::DuplicateKey {
                key_id: key_id.to_string(),
            }
            .into());
        }
        self.keys.insert(key_id, key);
        Ok(())
    }

    /// Add a key pair to the store.
    pub fn add_key_pair(&mut self, pair: KeyPair) -> Result<()> {
        self.add_key(pair.key_id, pair.key.clone())
    }

    /// Get a key by its ID.
    pub fn get_key(&self, key_id: &KeyId) -> Result<&ContentKey> {
        self.keys.get(key_id).ok_or_else(|| {
            KeyError::KeyNotFound {
                key_id: key_id.to_string(),
            }
            .into()
        })
    }

    /// Get or derive a key.
    ///
    /// If the key exists in the store, returns it.
    /// If a master key is set and the key doesn't exist, derives it.
    pub fn get_or_derive(&mut self, key_id: &KeyId) -> Result<&ContentKey> {
        // Use entry API to avoid separate lookup and insert
        use std::collections::hash_map::Entry;

        match self.keys.entry(*key_id) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                if let Some(ref master_key) = self.master_key {
                    let derived = ContentKey::derive(key_id, master_key);
                    Ok(entry.insert(derived))
                } else {
                    Err(KeyError::KeyNotFound {
                        key_id: key_id.to_string(),
                    }
                    .into())
                }
            }
        }
    }

    /// Check if a key exists.
    pub fn contains_key(&self, key_id: &KeyId) -> bool {
        self.keys.contains_key(key_id)
    }

    /// Remove a key from the store.
    pub fn remove_key(&mut self, key_id: &KeyId) -> Option<ContentKey> {
        self.keys.remove(key_id)
    }

    /// Get all key IDs.
    pub fn key_ids(&self) -> impl Iterator<Item = &KeyId> {
        self.keys.keys()
    }

    /// Get the number of keys.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Check if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Clear all keys from the store.
    pub fn clear(&mut self) {
        self.keys.clear();
    }
}

impl fmt::Debug for KeyStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KeyStore")
            .field("key_count", &self.keys.len())
            .field("has_master_key", &self.master_key.is_some())
            .finish()
    }
}

/// Common DRM system IDs.
pub mod system_ids {
    use uuid::Uuid;

    /// Widevine system ID.
    pub const WIDEVINE: Uuid = Uuid::from_bytes([
        0xed, 0xef, 0x8b, 0xa9, 0x79, 0xd6, 0x4a, 0xce, 0xa3, 0xc8, 0x27, 0xdc, 0xd5, 0x1d, 0x21,
        0xed,
    ]);

    /// PlayReady system ID.
    pub const PLAYREADY: Uuid = Uuid::from_bytes([
        0x9a, 0x04, 0xf0, 0x79, 0x98, 0x40, 0x42, 0x86, 0xab, 0x92, 0xe6, 0x5b, 0xe0, 0x88, 0x5f,
        0x95,
    ]);

    /// FairPlay Streaming system ID.
    pub const FAIRPLAY: Uuid = Uuid::from_bytes([
        0x94, 0xce, 0x86, 0xfb, 0x07, 0xff, 0x4f, 0x43, 0xad, 0xb8, 0x93, 0xd2, 0xfa, 0x96, 0x8c,
        0xa2,
    ]);

    /// Common encryption (CENC) system ID.
    pub const COMMON: Uuid = Uuid::from_bytes([
        0x10, 0x77, 0xef, 0xec, 0xc0, 0xb2, 0x4d, 0x02, 0xac, 0xe3, 0x3c, 0x1e, 0x52, 0xe2, 0xfb,
        0x4b,
    ]);

    /// Get the system name from a system ID.
    pub fn name_for_id(id: &Uuid) -> Option<&'static str> {
        match *id {
            WIDEVINE => Some("Widevine"),
            PLAYREADY => Some("PlayReady"),
            FAIRPLAY => Some("FairPlay Streaming"),
            COMMON => Some("Common Encryption"),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::DrmError;

    #[test]
    fn test_content_key_from_hex() {
        let key = ContentKey::from_hex("00112233445566778899aabbccddeeff").unwrap();
        assert_eq!(
            key.as_bytes(),
            &[0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff]
        );
    }

    #[test]
    fn test_content_key_from_base64() {
        let key = ContentKey::from_base64("ABEiM0RVZneImaq7zN3u/w==").unwrap();
        assert_eq!(
            key.as_bytes(),
            &[0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff]
        );
    }

    #[test]
    fn test_content_key_invalid_length() {
        let result = ContentKey::from_slice(&[0u8; 8]);
        assert!(matches!(
            result,
            Err(DrmError::Key(KeyError::InvalidKeyLength { .. }))
        ));
    }

    #[test]
    fn test_content_key_derive() {
        let master = ContentKey::from_hex("00112233445566778899aabbccddeeff").unwrap();
        let key_id = KeyId::parse("12345678-1234-1234-1234-123456789012").unwrap();

        let derived = ContentKey::derive(&key_id, &master);
        assert_eq!(derived.as_bytes().len(), AES_128_KEY_SIZE);

        // Derivation should be deterministic
        let derived2 = ContentKey::derive(&key_id, &master);
        assert_eq!(derived.as_bytes(), derived2.as_bytes());
    }

    #[test]
    fn test_key_id_parse() {
        let key_id = KeyId::parse("12345678-1234-1234-1234-123456789012").unwrap();
        assert_eq!(key_id.to_uuid_string(), "12345678-1234-1234-1234-123456789012");
    }

    #[test]
    fn test_key_id_from_hex() {
        let key_id = KeyId::from_hex("12345678123412341234123456789012").unwrap();
        assert_eq!(key_id.to_hex(), "12345678123412341234123456789012");
    }

    #[test]
    fn test_key_id_little_endian() {
        let key_id = KeyId::from_hex("12345678abcdef0123456789abcdef01").unwrap();
        let le_bytes = key_id.to_little_endian_bytes();

        // First 4 bytes reversed
        assert_eq!(le_bytes[0], 0x78);
        assert_eq!(le_bytes[1], 0x56);
        assert_eq!(le_bytes[2], 0x34);
        assert_eq!(le_bytes[3], 0x12);
    }

    #[test]
    fn test_iv_operations() {
        let mut iv = Iv::from_hex("00112233445566778899aabbccddeeff").unwrap();
        assert_eq!(iv.counter(), 0x8899aabbccddeeff);

        iv.set_counter(0);
        assert_eq!(iv.counter(), 0);

        iv.increment(100);
        assert_eq!(iv.counter(), 100);
    }

    #[test]
    fn test_iv_from_8_bytes() {
        let iv = Iv::from_8_bytes([0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77]);
        assert_eq!(
            iv.as_bytes(),
            &[0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        );
    }

    #[test]
    fn test_key_store() {
        let mut store = KeyStore::new();
        let key_id = KeyId::generate();
        let key = ContentKey::generate();

        store.add_key(key_id, key.clone()).unwrap();
        assert!(store.contains_key(&key_id));

        let retrieved = store.get_key(&key_id).unwrap();
        assert_eq!(retrieved.as_bytes(), key.as_bytes());
    }

    #[test]
    fn test_key_store_duplicate() {
        let mut store = KeyStore::new();
        let key_id = KeyId::generate();
        let key = ContentKey::generate();

        store.add_key(key_id, key.clone()).unwrap();
        let result = store.add_key(key_id, key);
        assert!(matches!(
            result,
            Err(DrmError::Key(KeyError::DuplicateKey { .. }))
        ));
    }

    #[test]
    fn test_key_store_derive() {
        let master_key = ContentKey::generate();
        let mut store = KeyStore::with_master_key(master_key.clone());
        let key_id = KeyId::generate();

        // Key doesn't exist, but should be derived
        let derived = store.get_or_derive(&key_id).unwrap();
        let expected = ContentKey::derive(&key_id, &master_key);
        assert_eq!(derived.as_bytes(), expected.as_bytes());
    }

    #[test]
    fn test_system_ids() {
        assert_eq!(system_ids::name_for_id(&system_ids::WIDEVINE), Some("Widevine"));
        assert_eq!(system_ids::name_for_id(&system_ids::PLAYREADY), Some("PlayReady"));
        assert_eq!(system_ids::name_for_id(&system_ids::FAIRPLAY), Some("FairPlay Streaming"));
    }
}
