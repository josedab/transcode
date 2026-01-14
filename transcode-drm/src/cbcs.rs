//! CBCS pattern encryption (AES-CBC with pattern).
//!
//! This module implements CBCS (cipher block chaining with pattern) encryption
//! as specified in ISO/IEC 23001-7 (Common Encryption). CBCS is the preferred
//! encryption mode for HLS and is required for FairPlay.

use crate::error::{EncryptionError, Result};
use crate::key::{ContentKey, Iv};
use aes::cipher::{BlockDecrypt, BlockEncrypt, KeyInit};
use aes::Aes128;

/// AES block size in bytes.
pub const AES_BLOCK_SIZE: usize = 16;

/// Default crypt blocks for CBCS pattern (1 of 10).
pub const DEFAULT_CRYPT_BLOCKS: u32 = 1;

/// Default skip blocks for CBCS pattern (9 of 10).
pub const DEFAULT_SKIP_BLOCKS: u32 = 9;

/// CBCS encryption pattern.
///
/// Defines the pattern of encrypted and clear blocks in CBCS mode.
/// The pattern encrypts `crypt_blocks` followed by skipping `skip_blocks`,
/// repeating for the entire sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pattern {
    /// Number of blocks to encrypt.
    pub crypt_blocks: u32,
    /// Number of blocks to skip (leave in clear).
    pub skip_blocks: u32,
}

impl Pattern {
    /// Create a new encryption pattern.
    pub fn new(crypt_blocks: u32, skip_blocks: u32) -> Result<Self> {
        if crypt_blocks == 0 && skip_blocks == 0 {
            return Err(EncryptionError::InvalidPattern {
                crypt_blocks,
                skip_blocks,
            }
            .into());
        }
        Ok(Self {
            crypt_blocks,
            skip_blocks,
        })
    }

    /// Default CBCS pattern (1:9 - encrypt 1 block, skip 9).
    pub fn default_cbcs() -> Self {
        Self {
            crypt_blocks: DEFAULT_CRYPT_BLOCKS,
            skip_blocks: DEFAULT_SKIP_BLOCKS,
        }
    }

    /// Full encryption pattern (encrypt all blocks).
    pub fn full_encryption() -> Self {
        Self {
            crypt_blocks: 1,
            skip_blocks: 0,
        }
    }

    /// Total pattern length in blocks.
    pub fn total_blocks(&self) -> u32 {
        self.crypt_blocks + self.skip_blocks
    }

    /// Calculate number of encrypted blocks for a given number of blocks.
    pub fn encrypted_block_count(&self, total: u32) -> u32 {
        if self.skip_blocks == 0 {
            // Full encryption
            return total;
        }

        let pattern_len = self.total_blocks();
        let full_patterns = total / pattern_len;
        let remaining = total % pattern_len;
        let remaining_encrypted = remaining.min(self.crypt_blocks);

        full_patterns * self.crypt_blocks + remaining_encrypted
    }

    /// Check if a block at a given index should be encrypted.
    pub fn should_encrypt_block(&self, block_index: u32) -> bool {
        if self.skip_blocks == 0 {
            return true;
        }
        let position_in_pattern = block_index % self.total_blocks();
        position_in_pattern < self.crypt_blocks
    }
}

impl Default for Pattern {
    fn default() -> Self {
        Self::default_cbcs()
    }
}

/// CBCS encryptor using AES-128-CBC with pattern encryption.
///
/// In CBCS mode, each encrypted block uses a constant IV (no chaining
/// across patterns). The IV is reset at the start of each pattern.
pub struct CbcsEncryptor {
    /// AES cipher for block operations.
    cipher: Aes128,
    /// Content encryption key.
    key: ContentKey,
    /// Constant IV used for each encrypted block.
    iv: Iv,
    /// Encryption pattern.
    pattern: Pattern,
}

impl CbcsEncryptor {
    /// Create a new CBCS encryptor.
    pub fn new(key: ContentKey, iv: Iv, pattern: Pattern) -> Self {
        let cipher = Aes128::new(key.as_bytes().into());
        Self {
            cipher,
            key,
            iv,
            pattern,
        }
    }

    /// Create with the default CBCS pattern (1:9).
    pub fn with_default_pattern(key: ContentKey, iv: Iv) -> Self {
        Self::new(key, iv, Pattern::default_cbcs())
    }

    /// Encrypt data in place using CBCS pattern.
    ///
    /// Data must be block-aligned (multiple of 16 bytes).
    pub fn encrypt(&self, data: &mut [u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        if !data.len().is_multiple_of(AES_BLOCK_SIZE) {
            return Err(EncryptionError::BlockAlignment {
                size: data.len(),
                block_size: AES_BLOCK_SIZE,
            }
            .into());
        }

        let num_blocks = data.len() / AES_BLOCK_SIZE;
        let iv_bytes = self.iv.as_bytes();

        for block_idx in 0..num_blocks {
            if self.pattern.should_encrypt_block(block_idx as u32) {
                let start = block_idx * AES_BLOCK_SIZE;
                let end = start + AES_BLOCK_SIZE;
                let block = &mut data[start..end];

                // XOR with IV
                for (byte, iv_byte) in block.iter_mut().zip(iv_bytes.iter()) {
                    *byte ^= iv_byte;
                }

                // Encrypt block - slice is guaranteed to be AES_BLOCK_SIZE from slicing above
                let block_len = block.len();
                let block_array: &mut [u8; AES_BLOCK_SIZE] = block.try_into().map_err(|_| {
                    EncryptionError::BlockAlignment {
                        size: block_len,
                        block_size: AES_BLOCK_SIZE,
                    }
                })?;
                self.cipher.encrypt_block(block_array.into());
            }
        }

        Ok(())
    }

    /// Encrypt the encrypted portion of a subsample.
    ///
    /// This handles the case where only part of the sample is encrypted,
    /// with a potential partial block at the end.
    pub fn encrypt_subsample(
        &self,
        data: &mut [u8],
        clear_bytes: usize,
    ) -> Result<()> {
        if clear_bytes >= data.len() {
            return Ok(()); // All clear, nothing to encrypt
        }

        let encrypted_portion = &mut data[clear_bytes..];
        let full_blocks = encrypted_portion.len() / AES_BLOCK_SIZE;
        let partial_block_size = encrypted_portion.len() % AES_BLOCK_SIZE;

        // Encrypt full blocks with pattern
        if full_blocks > 0 {
            let full_block_data = &mut encrypted_portion[..full_blocks * AES_BLOCK_SIZE];
            self.encrypt(full_block_data)?;
        }

        // Partial block at end is left in clear (CBCS spec)
        // This is intentional - partial blocks are not encrypted
        let _ = partial_block_size;

        Ok(())
    }

    /// Get the encryption pattern.
    pub fn pattern(&self) -> Pattern {
        self.pattern
    }

    /// Get the IV.
    pub fn iv(&self) -> &Iv {
        &self.iv
    }

    /// Get the key.
    pub fn key(&self) -> &ContentKey {
        &self.key
    }
}

/// CBCS decryptor using AES-128-CBC with pattern decryption.
pub struct CbcsDecryptor {
    /// AES cipher for block operations.
    cipher: Aes128,
    /// Constant IV used for each encrypted block.
    iv: Iv,
    /// Encryption pattern.
    pattern: Pattern,
}

impl CbcsDecryptor {
    /// Create a new CBCS decryptor.
    pub fn new(key: ContentKey, iv: Iv, pattern: Pattern) -> Self {
        let cipher = Aes128::new(key.as_bytes().into());
        Self { cipher, iv, pattern }
    }

    /// Create with the default CBCS pattern (1:9).
    pub fn with_default_pattern(key: ContentKey, iv: Iv) -> Self {
        Self::new(key, iv, Pattern::default_cbcs())
    }

    /// Decrypt data in place using CBCS pattern.
    pub fn decrypt(&self, data: &mut [u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        if !data.len().is_multiple_of(AES_BLOCK_SIZE) {
            return Err(EncryptionError::BlockAlignment {
                size: data.len(),
                block_size: AES_BLOCK_SIZE,
            }
            .into());
        }

        let num_blocks = data.len() / AES_BLOCK_SIZE;
        let iv_bytes = self.iv.as_bytes();

        for block_idx in 0..num_blocks {
            if self.pattern.should_encrypt_block(block_idx as u32) {
                let start = block_idx * AES_BLOCK_SIZE;
                let end = start + AES_BLOCK_SIZE;
                let block = &mut data[start..end];

                // Decrypt block - slice is guaranteed to be AES_BLOCK_SIZE from slicing above
                let block_len = block.len();
                let block_array: &mut [u8; AES_BLOCK_SIZE] = block.try_into().map_err(|_| {
                    EncryptionError::BlockAlignment {
                        size: block_len,
                        block_size: AES_BLOCK_SIZE,
                    }
                })?;
                self.cipher.decrypt_block(block_array.into());

                // XOR with IV
                for (byte, iv_byte) in block.iter_mut().zip(iv_bytes.iter()) {
                    *byte ^= iv_byte;
                }
            }
        }

        Ok(())
    }

    /// Decrypt the encrypted portion of a subsample.
    pub fn decrypt_subsample(
        &self,
        data: &mut [u8],
        clear_bytes: usize,
    ) -> Result<()> {
        if clear_bytes >= data.len() {
            return Ok(());
        }

        let encrypted_portion = &mut data[clear_bytes..];
        let full_blocks = encrypted_portion.len() / AES_BLOCK_SIZE;

        if full_blocks > 0 {
            let full_block_data = &mut encrypted_portion[..full_blocks * AES_BLOCK_SIZE];
            self.decrypt(full_block_data)?;
        }

        Ok(())
    }

    /// Get the encryption pattern.
    pub fn pattern(&self) -> Pattern {
        self.pattern
    }

    /// Get the IV.
    pub fn iv(&self) -> &Iv {
        &self.iv
    }
}

/// Calculate the number of encrypted bytes in a sample using CBCS.
///
/// Accounts for pattern encryption and partial blocks.
pub fn calculate_encrypted_bytes(sample_size: usize, clear_bytes: usize, pattern: Pattern) -> usize {
    if clear_bytes >= sample_size {
        return 0;
    }

    let encrypted_portion = sample_size - clear_bytes;
    let full_blocks = encrypted_portion / AES_BLOCK_SIZE;
    let encrypted_blocks = pattern.encrypted_block_count(full_blocks as u32) as usize;

    encrypted_blocks * AES_BLOCK_SIZE
}

/// CBCS subsample entry with pattern information.
#[derive(Clone, Debug)]
pub struct CbcsSubsample {
    /// Number of bytes in the clear.
    pub clear_bytes: usize,
    /// Number of bytes in the protected region.
    pub protected_bytes: usize,
    /// Encryption pattern to use.
    pub pattern: Pattern,
}

impl CbcsSubsample {
    /// Create a new CBCS subsample entry.
    pub fn new(clear_bytes: usize, protected_bytes: usize, pattern: Pattern) -> Self {
        Self {
            clear_bytes,
            protected_bytes,
            pattern,
        }
    }

    /// Create with default CBCS pattern.
    pub fn with_default_pattern(clear_bytes: usize, protected_bytes: usize) -> Self {
        Self::new(clear_bytes, protected_bytes, Pattern::default_cbcs())
    }

    /// Total size of this subsample.
    pub fn total_size(&self) -> usize {
        self.clear_bytes + self.protected_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> ContentKey {
        ContentKey::from_hex("00112233445566778899aabbccddeeff").unwrap()
    }

    fn test_iv() -> Iv {
        Iv::from_hex("0102030405060708090a0b0c0d0e0f10").unwrap()
    }

    #[test]
    fn test_pattern_creation() {
        let pattern = Pattern::new(1, 9).unwrap();
        assert_eq!(pattern.crypt_blocks, 1);
        assert_eq!(pattern.skip_blocks, 9);
        assert_eq!(pattern.total_blocks(), 10);
    }

    #[test]
    fn test_pattern_invalid() {
        let result = Pattern::new(0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_pattern_should_encrypt() {
        let pattern = Pattern::new(2, 3).unwrap();

        // First pattern cycle: blocks 0,1 encrypted, 2,3,4 clear
        assert!(pattern.should_encrypt_block(0));
        assert!(pattern.should_encrypt_block(1));
        assert!(!pattern.should_encrypt_block(2));
        assert!(!pattern.should_encrypt_block(3));
        assert!(!pattern.should_encrypt_block(4));

        // Second pattern cycle: blocks 5,6 encrypted, 7,8,9 clear
        assert!(pattern.should_encrypt_block(5));
        assert!(pattern.should_encrypt_block(6));
        assert!(!pattern.should_encrypt_block(7));
    }

    #[test]
    fn test_pattern_encrypted_count() {
        let pattern = Pattern::new(1, 9).unwrap();

        assert_eq!(pattern.encrypted_block_count(1), 1);
        assert_eq!(pattern.encrypted_block_count(5), 1);
        assert_eq!(pattern.encrypted_block_count(10), 1);
        assert_eq!(pattern.encrypted_block_count(11), 2);
        assert_eq!(pattern.encrypted_block_count(20), 2);
    }

    #[test]
    fn test_pattern_full_encryption() {
        let pattern = Pattern::full_encryption();

        assert!(pattern.should_encrypt_block(0));
        assert!(pattern.should_encrypt_block(100));
        assert_eq!(pattern.encrypted_block_count(50), 50);
    }

    #[test]
    fn test_cbcs_encrypt_decrypt() {
        let key = test_key();
        let iv = test_iv();
        let pattern = Pattern::new(1, 9).unwrap();

        // Create 160 bytes (10 blocks)
        let original: Vec<u8> = (0..160).collect();
        let mut data = original.clone();

        // Encrypt
        let encryptor = CbcsEncryptor::new(key.clone(), iv.clone(), pattern);
        encryptor.encrypt(&mut data).unwrap();

        // First block should be encrypted (changed)
        assert_ne!(&data[..16], &original[..16]);
        // Blocks 1-9 should be clear (unchanged)
        assert_eq!(&data[16..160], &original[16..160]);

        // Decrypt
        let decryptor = CbcsDecryptor::new(key, iv, pattern);
        decryptor.decrypt(&mut data).unwrap();

        assert_eq!(data, original);
    }

    #[test]
    fn test_cbcs_multiple_patterns() {
        let key = test_key();
        let iv = test_iv();
        let pattern = Pattern::new(1, 9).unwrap();

        // Create 320 bytes (20 blocks = 2 full patterns)
        let original: Vec<u8> = (0u8..=255).cycle().take(320).collect();
        let mut data = original.clone();

        // Encrypt
        let encryptor = CbcsEncryptor::new(key.clone(), iv.clone(), pattern);
        encryptor.encrypt(&mut data).unwrap();

        // Block 0 and 10 should be encrypted
        assert_ne!(&data[0..16], &original[0..16]);
        assert_ne!(&data[160..176], &original[160..176]);

        // Other blocks should be clear
        assert_eq!(&data[16..160], &original[16..160]);
        assert_eq!(&data[176..320], &original[176..320]);

        // Decrypt
        let decryptor = CbcsDecryptor::new(key, iv, pattern);
        decryptor.decrypt(&mut data).unwrap();

        assert_eq!(data, original);
    }

    #[test]
    fn test_cbcs_full_encryption() {
        let key = test_key();
        let iv = test_iv();
        let pattern = Pattern::full_encryption();

        let original: Vec<u8> = (0..48).collect();
        let mut data = original.clone();

        let encryptor = CbcsEncryptor::new(key.clone(), iv.clone(), pattern);
        encryptor.encrypt(&mut data).unwrap();

        // All blocks should be encrypted
        assert_ne!(data, original);

        let decryptor = CbcsDecryptor::new(key, iv, pattern);
        decryptor.decrypt(&mut data).unwrap();

        assert_eq!(data, original);
    }

    #[test]
    fn test_cbcs_alignment_error() {
        let key = test_key();
        let iv = test_iv();

        let mut data = vec![0u8; 17]; // Not block aligned

        let encryptor = CbcsEncryptor::with_default_pattern(key, iv);
        let result = encryptor.encrypt(&mut data);

        assert!(matches!(
            result,
            Err(crate::error::DrmError::Encryption(EncryptionError::BlockAlignment { .. }))
        ));
    }

    #[test]
    fn test_cbcs_subsample_encryption() {
        let key = test_key();
        let iv = test_iv();

        // Sample: 5 bytes clear, 32 bytes protected (2 blocks)
        let original: Vec<u8> = (0..37).collect();
        let mut data = original.clone();

        let encryptor = CbcsEncryptor::with_default_pattern(key.clone(), iv.clone());
        encryptor.encrypt_subsample(&mut data, 5).unwrap();

        // Clear portion unchanged
        assert_eq!(&data[..5], &original[..5]);

        // First protected block (bytes 5-20) encrypted with 1:9 pattern
        // Only block 0 of protected region is encrypted
        assert_ne!(&data[5..21], &original[5..21]);

        // Second protected block (bytes 21-36) is clear (skip block)
        assert_eq!(&data[21..37], &original[21..37]);

        // Decrypt
        let decryptor = CbcsDecryptor::with_default_pattern(key, iv);
        decryptor.decrypt_subsample(&mut data, 5).unwrap();

        assert_eq!(data, original);
    }

    #[test]
    fn test_calculate_encrypted_bytes() {
        let pattern = Pattern::new(1, 9).unwrap();

        // 160 bytes = 10 blocks with 1:9 pattern = 1 encrypted block = 16 bytes
        assert_eq!(calculate_encrypted_bytes(160, 0, pattern), 16);

        // 320 bytes = 20 blocks = 2 encrypted blocks = 32 bytes
        assert_eq!(calculate_encrypted_bytes(320, 0, pattern), 32);

        // 10 bytes clear, 150 bytes protected = 9 full blocks = 1 encrypted
        assert_eq!(calculate_encrypted_bytes(160, 10, pattern), 16);
    }

    #[test]
    fn test_cbcs_subsample_struct() {
        let subsample = CbcsSubsample::with_default_pattern(10, 150);
        assert_eq!(subsample.total_size(), 160);
        assert_eq!(subsample.pattern.crypt_blocks, 1);
        assert_eq!(subsample.pattern.skip_blocks, 9);
    }

    #[test]
    fn test_default_pattern() {
        let pattern = Pattern::default();
        assert_eq!(pattern.crypt_blocks, DEFAULT_CRYPT_BLOCKS);
        assert_eq!(pattern.skip_blocks, DEFAULT_SKIP_BLOCKS);
    }
}
