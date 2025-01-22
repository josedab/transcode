//! Key management for confidential transcoding.

use crate::error::{ConfidentialError, Result};
use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Content encryption key (CEK).
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct ContentKey {
    key: [u8; 32],
    key_id: String,
}

impl ContentKey {
    /// Generate a new random content key.
    pub fn generate() -> Result<Self> {
        use ring::rand::{SecureRandom, SystemRandom};
        let rng = SystemRandom::new();
        let mut key = [0u8; 32];
        rng.fill(&mut key)
            .map_err(|_| ConfidentialError::KeyError("Failed to generate key".to_string()))?;

        let key_id = uuid::Uuid::new_v4().to_string();
        Ok(Self { key, key_id })
    }

    /// Create from raw bytes.
    pub fn from_bytes(bytes: &[u8], key_id: impl Into<String>) -> Result<Self> {
        if bytes.len() != 32 {
            return Err(ConfidentialError::KeyError(
                "Key must be 32 bytes".to_string(),
            ));
        }
        let mut key = [0u8; 32];
        key.copy_from_slice(bytes);
        Ok(Self {
            key,
            key_id: key_id.into(),
        })
    }

    /// Get the key ID.
    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    /// Get the raw key bytes (use with caution).
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.key
    }

    /// Encrypt data with this key.
    pub fn encrypt(&self, plaintext: &[u8], nonce: &[u8; 12]) -> Result<Vec<u8>> {
        let cipher = Aes256Gcm::new_from_slice(&self.key)
            .map_err(|e| ConfidentialError::CryptoError(e.to_string()))?;

        cipher
            .encrypt(Nonce::from_slice(nonce), plaintext)
            .map_err(|e| ConfidentialError::CryptoError(e.to_string()))
    }

    /// Decrypt data with this key.
    pub fn decrypt(&self, ciphertext: &[u8], nonce: &[u8; 12]) -> Result<Vec<u8>> {
        let cipher = Aes256Gcm::new_from_slice(&self.key)
            .map_err(|e| ConfidentialError::CryptoError(e.to_string()))?;

        cipher
            .decrypt(Nonce::from_slice(nonce), ciphertext)
            .map_err(|e| ConfidentialError::CryptoError(e.to_string()))
    }
}

impl std::fmt::Debug for ContentKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContentKey")
            .field("key_id", &self.key_id)
            .field("key", &"[REDACTED]")
            .finish()
    }
}

/// Key wrapping key (KEK) for protecting content keys.
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct KeyWrapKey {
    key: [u8; 32],
}

impl KeyWrapKey {
    /// Generate a new random KEK.
    pub fn generate() -> Result<Self> {
        use ring::rand::{SecureRandom, SystemRandom};
        let rng = SystemRandom::new();
        let mut key = [0u8; 32];
        rng.fill(&mut key)
            .map_err(|_| ConfidentialError::KeyError("Failed to generate KEK".to_string()))?;
        Ok(Self { key })
    }

    /// Create from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 32 {
            return Err(ConfidentialError::KeyError(
                "KEK must be 32 bytes".to_string(),
            ));
        }
        let mut key = [0u8; 32];
        key.copy_from_slice(bytes);
        Ok(Self { key })
    }

    /// Wrap a content key.
    pub fn wrap(&self, content_key: &ContentKey) -> Result<WrappedKey> {
        use ring::rand::{SecureRandom, SystemRandom};
        let rng = SystemRandom::new();

        let mut nonce = [0u8; 12];
        rng.fill(&mut nonce)
            .map_err(|_| ConfidentialError::CryptoError("Failed to generate nonce".to_string()))?;

        let cipher = Aes256Gcm::new_from_slice(&self.key)
            .map_err(|e| ConfidentialError::CryptoError(e.to_string()))?;

        let wrapped = cipher
            .encrypt(Nonce::from_slice(&nonce), content_key.as_bytes().as_slice())
            .map_err(|e| ConfidentialError::CryptoError(e.to_string()))?;

        Ok(WrappedKey {
            key_id: content_key.key_id.clone(),
            wrapped_key: wrapped,
            nonce: nonce.to_vec(),
            algorithm: "AES-256-GCM-WRAP".to_string(),
        })
    }

    /// Unwrap a content key.
    pub fn unwrap(&self, wrapped: &WrappedKey) -> Result<ContentKey> {
        if wrapped.nonce.len() != 12 {
            return Err(ConfidentialError::CryptoError(
                "Invalid nonce length".to_string(),
            ));
        }

        let mut nonce = [0u8; 12];
        nonce.copy_from_slice(&wrapped.nonce);

        let cipher = Aes256Gcm::new_from_slice(&self.key)
            .map_err(|e| ConfidentialError::CryptoError(e.to_string()))?;

        let key_bytes = cipher
            .decrypt(Nonce::from_slice(&nonce), wrapped.wrapped_key.as_slice())
            .map_err(|e| ConfidentialError::CryptoError(e.to_string()))?;

        ContentKey::from_bytes(&key_bytes, &wrapped.key_id)
    }
}

impl std::fmt::Debug for KeyWrapKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeyWrapKey")
            .field("key", &"[REDACTED]")
            .finish()
    }
}

/// Wrapped (encrypted) content key for storage/transport.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WrappedKey {
    pub key_id: String,
    #[serde(with = "base64_serde")]
    pub wrapped_key: Vec<u8>,
    #[serde(with = "base64_serde")]
    pub nonce: Vec<u8>,
    pub algorithm: String,
}

/// Key derivation function for deriving keys from master secret.
pub struct KeyDerivation;

impl KeyDerivation {
    /// Derive a key from master secret and context.
    pub fn derive(master: &[u8], context: &[u8], key_len: usize) -> Vec<u8> {
        // HKDF-like derivation
        let mut hasher = Sha256::new();
        hasher.update(master);
        hasher.update(context);
        let hash = hasher.finalize();

        // If we need more bytes, iterate
        let mut result = Vec::with_capacity(key_len);
        let mut counter = 1u32;

        while result.len() < key_len {
            let mut iter_hasher = Sha256::new();
            iter_hasher.update(hash);
            iter_hasher.update(counter.to_be_bytes());
            result.extend_from_slice(&iter_hasher.finalize());
            counter += 1;
        }

        result.truncate(key_len);
        result
    }

    /// Derive a content key from master secret.
    pub fn derive_content_key(master: &[u8], key_id: &str) -> Result<ContentKey> {
        let derived = Self::derive(master, key_id.as_bytes(), 32);
        ContentKey::from_bytes(&derived, key_id)
    }
}

/// Sealed data for enclave persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SealedData {
    #[serde(with = "base64_serde")]
    pub ciphertext: Vec<u8>,
    #[serde(with = "base64_serde")]
    pub nonce: Vec<u8>,
    #[serde(with = "base64_serde")]
    pub tag: Vec<u8>,
    pub seal_policy: SealPolicy,
}

/// Policy for sealed data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SealPolicy {
    /// Sealed to specific enclave measurement.
    EnclaveMeasurement,
    /// Sealed to signer identity (allows enclave updates).
    SignerIdentity,
}

mod base64_serde {
    use base64::{engine::general_purpose::STANDARD, Engine};
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&STANDARD.encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let s = String::deserialize(d)?;
        STANDARD.decode(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_key_generation() {
        let key = ContentKey::generate().unwrap();
        assert_eq!(key.as_bytes().len(), 32);
        assert!(!key.key_id().is_empty());
    }

    #[test]
    fn test_content_key_encrypt_decrypt() {
        let key = ContentKey::generate().unwrap();
        let plaintext = b"Hello, secure world!";
        let nonce = [0u8; 12];

        let ciphertext = key.encrypt(plaintext, &nonce).unwrap();
        assert_ne!(ciphertext.as_slice(), plaintext);

        let decrypted = key.decrypt(&ciphertext, &nonce).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_key_wrap_unwrap() {
        let kek = KeyWrapKey::generate().unwrap();
        let cek = ContentKey::generate().unwrap();
        let key_id = cek.key_id().to_string();

        let wrapped = kek.wrap(&cek).unwrap();
        assert_eq!(wrapped.key_id, key_id);

        let unwrapped = kek.unwrap(&wrapped).unwrap();
        assert_eq!(unwrapped.key_id(), key_id);
        assert_eq!(unwrapped.as_bytes(), cek.as_bytes());
    }

    #[test]
    fn test_key_derivation() {
        let master = b"master_secret";
        let key1 = KeyDerivation::derive(master, b"context1", 32);
        let key2 = KeyDerivation::derive(master, b"context2", 32);

        assert_eq!(key1.len(), 32);
        assert_ne!(key1, key2);

        // Same inputs should give same output
        let key1_again = KeyDerivation::derive(master, b"context1", 32);
        assert_eq!(key1, key1_again);
    }

    #[test]
    fn test_zeroize() {
        let mut key = ContentKey::generate().unwrap();
        let key_bytes_before = *key.as_bytes();
        assert_ne!(key_bytes_before, [0u8; 32]);

        key.zeroize();
        assert_eq!(key.key, [0u8; 32]);
    }
}
