//! AES-CTR encryption for CENC.
//!
//! This module provides AES-128-CTR encryption/decryption as specified
//! in ISO/IEC 23001-7 (Common Encryption in ISO Base Media File Format).

use crate::error::{EncryptionError, Result};
use crate::key::{ContentKey, Iv};
use aes::cipher::{KeyIvInit, StreamCipher, StreamCipherSeek};
use aes::Aes128;

/// AES block size in bytes.
pub const AES_BLOCK_SIZE: usize = 16;

/// Type alias for AES-128-CTR cipher.
type Aes128Ctr = ctr::Ctr128BE<Aes128>;

/// AES-CTR encryptor/decryptor.
///
/// Implements AES-128-CTR mode encryption as used in CENC.
/// CTR mode turns a block cipher into a stream cipher.
#[derive(Clone)]
pub struct AesCtr {
    /// Content encryption key.
    key: ContentKey,
    /// Initialization vector (nonce + counter).
    iv: Iv,
    /// Current byte offset within the stream.
    byte_offset: u64,
}

impl AesCtr {
    /// Create a new AES-CTR instance.
    pub fn new(key: ContentKey, iv: Iv) -> Self {
        Self {
            key,
            iv,
            byte_offset: 0,
        }
    }

    /// Create a new instance with a specific byte offset.
    pub fn with_offset(key: ContentKey, iv: Iv, byte_offset: u64) -> Self {
        Self {
            key,
            iv,
            byte_offset,
        }
    }

    /// Create the internal cipher with the current state.
    fn create_cipher(&self) -> Aes128Ctr {
        Aes128Ctr::new(self.key.as_bytes().into(), self.iv.as_bytes().into())
    }

    /// Encrypt data in place.
    ///
    /// This method modifies the input buffer directly.
    /// CTR mode is symmetric, so encryption and decryption are the same operation.
    pub fn encrypt(&mut self, data: &mut [u8]) {
        if data.is_empty() {
            return;
        }

        let mut cipher = self.create_cipher();
        cipher.seek(self.byte_offset);
        cipher.apply_keystream(data);
        self.byte_offset += data.len() as u64;
    }

    /// Decrypt data in place.
    ///
    /// This is identical to encryption in CTR mode.
    pub fn decrypt(&mut self, data: &mut [u8]) {
        self.encrypt(data);
    }

    /// Encrypt data and return the result.
    pub fn encrypt_copy(&mut self, data: &[u8]) -> Vec<u8> {
        let mut output = data.to_vec();
        self.encrypt(&mut output);
        output
    }

    /// Decrypt data and return the result.
    pub fn decrypt_copy(&mut self, data: &[u8]) -> Vec<u8> {
        self.encrypt_copy(data)
    }

    /// Encrypt data at a specific offset without updating the internal state.
    pub fn encrypt_at_offset(&self, data: &mut [u8], offset: u64) {
        if data.is_empty() {
            return;
        }

        let mut cipher = self.create_cipher();
        cipher.seek(offset);
        cipher.apply_keystream(data);
    }

    /// Reset the byte offset to zero.
    pub fn reset(&mut self) {
        self.byte_offset = 0;
    }

    /// Set the byte offset.
    pub fn set_offset(&mut self, offset: u64) {
        self.byte_offset = offset;
    }

    /// Get the current byte offset.
    pub fn offset(&self) -> u64 {
        self.byte_offset
    }

    /// Set a new IV and reset the offset.
    pub fn set_iv(&mut self, iv: Iv) {
        self.iv = iv;
        self.byte_offset = 0;
    }

    /// Get the current IV.
    pub fn iv(&self) -> &Iv {
        &self.iv
    }

    /// Get the key.
    pub fn key(&self) -> &ContentKey {
        &self.key
    }
}

/// Subsample encryption for NAL units.
///
/// In CENC, video samples often use subsample encryption where only
/// portions of the sample (NAL unit payloads) are encrypted.
#[derive(Clone, Debug, Default)]
pub struct SubsampleEntry {
    /// Number of bytes in the clear (unencrypted).
    pub bytes_of_clear_data: u32,
    /// Number of encrypted bytes following the clear data.
    pub bytes_of_encrypted_data: u32,
}

impl SubsampleEntry {
    /// Create a new subsample entry.
    pub fn new(clear: u32, encrypted: u32) -> Self {
        Self {
            bytes_of_clear_data: clear,
            bytes_of_encrypted_data: encrypted,
        }
    }

    /// Total size of this subsample entry.
    pub fn total_size(&self) -> u32 {
        self.bytes_of_clear_data + self.bytes_of_encrypted_data
    }
}

/// Encrypt a sample with subsample encryption.
///
/// Only the encrypted portions of each subsample are processed.
pub fn encrypt_subsamples(
    cipher: &mut AesCtr,
    data: &mut [u8],
    subsamples: &[SubsampleEntry],
) -> Result<()> {
    let total_subsample_size: u64 = subsamples
        .iter()
        .map(|s| s.total_size() as u64)
        .sum();

    if total_subsample_size > data.len() as u64 {
        return Err(EncryptionError::InvalidSubsample(format!(
            "Subsample size {} exceeds data size {}",
            total_subsample_size,
            data.len()
        ))
        .into());
    }

    let mut offset = 0usize;
    for subsample in subsamples {
        // Skip clear data
        offset += subsample.bytes_of_clear_data as usize;

        // Encrypt the encrypted portion
        let encrypted_size = subsample.bytes_of_encrypted_data as usize;
        if encrypted_size > 0 {
            let end = offset + encrypted_size;
            if end > data.len() {
                return Err(EncryptionError::BufferTooSmall {
                    needed: end,
                    available: data.len(),
                }
                .into());
            }
            cipher.encrypt(&mut data[offset..end]);
        }
        offset += encrypted_size;
    }

    Ok(())
}

/// Decrypt a sample with subsample encryption.
pub fn decrypt_subsamples(
    cipher: &mut AesCtr,
    data: &mut [u8],
    subsamples: &[SubsampleEntry],
) -> Result<()> {
    // CTR mode decryption is identical to encryption
    encrypt_subsamples(cipher, data, subsamples)
}

/// Calculate subsample entries for H.264/H.265 NAL units.
///
/// This creates subsample entries that keep NAL headers in the clear
/// while encrypting the payload. Each NAL unit header is left unencrypted.
///
/// # Arguments
///
/// * `nal_sizes` - Sizes of individual NAL units in the sample
/// * `nal_header_size` - Size of NAL unit header (1 for H.264, 2 for H.265)
pub fn calculate_nal_subsamples(nal_sizes: &[u32], nal_header_size: u32) -> Vec<SubsampleEntry> {
    let mut subsamples = Vec::new();

    for &nal_size in nal_sizes {
        if nal_size <= nal_header_size {
            // NAL too small, keep entirely in clear
            subsamples.push(SubsampleEntry::new(nal_size, 0));
        } else {
            subsamples.push(SubsampleEntry::new(
                nal_header_size,
                nal_size - nal_header_size,
            ));
        }
    }

    subsamples
}

/// Merge consecutive subsample entries for efficiency.
///
/// Combines adjacent clear-only or adjacent encrypted regions.
pub fn optimize_subsamples(subsamples: &[SubsampleEntry]) -> Vec<SubsampleEntry> {
    if subsamples.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::new();
    let mut current = subsamples[0].clone();

    for entry in subsamples.iter().skip(1) {
        if entry.bytes_of_encrypted_data == 0 && current.bytes_of_encrypted_data == 0 {
            // Both are clear-only, merge
            current.bytes_of_clear_data += entry.bytes_of_clear_data;
        } else if entry.bytes_of_clear_data == 0 && current.bytes_of_clear_data == 0 {
            // Both are encrypted-only, merge
            current.bytes_of_encrypted_data += entry.bytes_of_encrypted_data;
        } else {
            result.push(current);
            current = entry.clone();
        }
    }
    result.push(current);

    result
}

/// Sample encryption information.
///
/// Contains all information needed to encrypt or decrypt a sample.
#[derive(Clone, Debug)]
pub struct SampleEncryptionInfo {
    /// Per-sample IV (if different from previous sample).
    pub iv: Option<Iv>,
    /// Subsample entries (empty if entire sample is encrypted).
    pub subsamples: Vec<SubsampleEntry>,
    /// Sample is encrypted (false for clear lead samples).
    pub is_encrypted: bool,
}

impl SampleEncryptionInfo {
    /// Create info for a fully encrypted sample.
    pub fn full_sample(iv: Iv) -> Self {
        Self {
            iv: Some(iv),
            subsamples: Vec::new(),
            is_encrypted: true,
        }
    }

    /// Create info for a sample with subsample encryption.
    pub fn with_subsamples(iv: Iv, subsamples: Vec<SubsampleEntry>) -> Self {
        Self {
            iv: Some(iv),
            subsamples,
            is_encrypted: true,
        }
    }

    /// Create info for a clear (unencrypted) sample.
    pub fn clear() -> Self {
        Self {
            iv: None,
            subsamples: Vec::new(),
            is_encrypted: false,
        }
    }

    /// Check if this sample uses subsample encryption.
    pub fn uses_subsamples(&self) -> bool {
        !self.subsamples.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> ContentKey {
        ContentKey::from_hex("00112233445566778899aabbccddeeff").unwrap()
    }

    fn test_iv() -> Iv {
        Iv::from_hex("00000000000000000000000000000000").unwrap()
    }

    #[test]
    fn test_aes_ctr_encrypt_decrypt() {
        let key = test_key();
        let iv = test_iv();

        let plaintext = b"Hello, World! This is a test message.";
        let mut data = plaintext.to_vec();

        // Encrypt
        let mut cipher = AesCtr::new(key.clone(), iv.clone());
        cipher.encrypt(&mut data);
        assert_ne!(&data[..], &plaintext[..]);

        // Decrypt
        let mut cipher = AesCtr::new(key, iv);
        cipher.decrypt(&mut data);
        assert_eq!(&data[..], &plaintext[..]);
    }

    #[test]
    fn test_aes_ctr_symmetry() {
        let key = test_key();
        let iv = test_iv();

        let plaintext = b"Test data for CTR mode";
        let mut data = plaintext.to_vec();

        // CTR mode: encrypt twice should give back original
        let mut cipher1 = AesCtr::new(key.clone(), iv.clone());
        cipher1.encrypt(&mut data);

        let mut cipher2 = AesCtr::new(key, iv);
        cipher2.encrypt(&mut data);

        assert_eq!(&data[..], &plaintext[..]);
    }

    #[test]
    fn test_aes_ctr_offset() {
        let key = test_key();
        let iv = test_iv();

        let plaintext = b"0123456789ABCDEF0123456789ABCDEF";

        // Encrypt in two parts
        let mut data1 = plaintext[..16].to_vec();
        let mut data2 = plaintext[16..].to_vec();

        let mut cipher = AesCtr::new(key.clone(), iv.clone());
        cipher.encrypt(&mut data1);
        assert_eq!(cipher.offset(), 16);
        cipher.encrypt(&mut data2);

        // Encrypt all at once
        let mut data_full = plaintext.to_vec();
        let mut cipher_full = AesCtr::new(key, iv);
        cipher_full.encrypt(&mut data_full);

        // Results should be identical
        let mut combined = data1;
        combined.extend_from_slice(&data2);
        assert_eq!(combined, data_full);
    }

    #[test]
    fn test_aes_ctr_encrypt_at_offset() {
        let key = test_key();
        let iv = test_iv();

        let plaintext = b"Test data";
        let mut data1 = plaintext.to_vec();
        let mut data2 = plaintext.to_vec();

        let cipher = AesCtr::new(key, iv);

        // Encrypt at offset 0
        cipher.encrypt_at_offset(&mut data1, 0);

        // Encrypt at offset 100 (should be different)
        cipher.encrypt_at_offset(&mut data2, 100);

        assert_ne!(data1, data2);
    }

    #[test]
    fn test_subsample_encryption() {
        let key = test_key();
        let iv = test_iv();

        // Create sample with clear and encrypted regions
        let mut data = vec![0u8; 100];
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = i as u8;
        }
        let original = data.clone();

        let subsamples = vec![
            SubsampleEntry::new(10, 40), // 10 clear, 40 encrypted
            SubsampleEntry::new(5, 45),  // 5 clear, 45 encrypted
        ];

        // Encrypt
        let mut cipher = AesCtr::new(key.clone(), iv.clone());
        encrypt_subsamples(&mut cipher, &mut data, &subsamples).unwrap();

        // Check clear regions are unchanged
        assert_eq!(&data[..10], &original[..10]);
        assert_eq!(&data[50..55], &original[50..55]);

        // Check encrypted regions are changed
        assert_ne!(&data[10..50], &original[10..50]);
        assert_ne!(&data[55..100], &original[55..100]);

        // Decrypt
        let mut cipher = AesCtr::new(key, iv);
        decrypt_subsamples(&mut cipher, &mut data, &subsamples).unwrap();

        assert_eq!(data, original);
    }

    #[test]
    fn test_calculate_nal_subsamples() {
        // H.264 NALs (1-byte header)
        let nal_sizes = vec![100, 50, 200];
        let subsamples = calculate_nal_subsamples(&nal_sizes, 1);

        assert_eq!(subsamples.len(), 3);
        assert_eq!(subsamples[0].bytes_of_clear_data, 1);
        assert_eq!(subsamples[0].bytes_of_encrypted_data, 99);
        assert_eq!(subsamples[1].bytes_of_clear_data, 1);
        assert_eq!(subsamples[1].bytes_of_encrypted_data, 49);
        assert_eq!(subsamples[2].bytes_of_clear_data, 1);
        assert_eq!(subsamples[2].bytes_of_encrypted_data, 199);
    }

    #[test]
    fn test_calculate_nal_subsamples_hevc() {
        // H.265 NALs (2-byte header)
        let nal_sizes = vec![100, 1]; // Second NAL too small
        let subsamples = calculate_nal_subsamples(&nal_sizes, 2);

        assert_eq!(subsamples.len(), 2);
        assert_eq!(subsamples[0].bytes_of_clear_data, 2);
        assert_eq!(subsamples[0].bytes_of_encrypted_data, 98);
        assert_eq!(subsamples[1].bytes_of_clear_data, 1); // Entire NAL in clear
        assert_eq!(subsamples[1].bytes_of_encrypted_data, 0);
    }

    #[test]
    fn test_optimize_subsamples() {
        let subsamples = vec![
            SubsampleEntry::new(10, 0), // Clear only
            SubsampleEntry::new(5, 0),  // Clear only (should merge)
            SubsampleEntry::new(0, 20), // Encrypted only
            SubsampleEntry::new(0, 30), // Encrypted only (should merge)
            SubsampleEntry::new(5, 10), // Mixed
        ];

        let optimized = optimize_subsamples(&subsamples);

        assert_eq!(optimized.len(), 3);
        assert_eq!(optimized[0].bytes_of_clear_data, 15);
        assert_eq!(optimized[0].bytes_of_encrypted_data, 0);
        assert_eq!(optimized[1].bytes_of_clear_data, 0);
        assert_eq!(optimized[1].bytes_of_encrypted_data, 50);
        assert_eq!(optimized[2].bytes_of_clear_data, 5);
        assert_eq!(optimized[2].bytes_of_encrypted_data, 10);
    }

    #[test]
    fn test_sample_encryption_info() {
        let iv = test_iv();

        let full = SampleEncryptionInfo::full_sample(iv.clone());
        assert!(full.is_encrypted);
        assert!(!full.uses_subsamples());

        let subsampled = SampleEncryptionInfo::with_subsamples(
            iv,
            vec![SubsampleEntry::new(10, 90)],
        );
        assert!(subsampled.is_encrypted);
        assert!(subsampled.uses_subsamples());

        let clear = SampleEncryptionInfo::clear();
        assert!(!clear.is_encrypted);
    }

    #[test]
    fn test_subsample_error_handling() {
        let key = test_key();
        let iv = test_iv();
        let mut data = vec![0u8; 50];

        // Subsamples larger than data
        let subsamples = vec![SubsampleEntry::new(30, 30)]; // Total 60 > 50

        let mut cipher = AesCtr::new(key, iv);
        let result = encrypt_subsamples(&mut cipher, &mut data, &subsamples);

        assert!(result.is_err());
    }
}
