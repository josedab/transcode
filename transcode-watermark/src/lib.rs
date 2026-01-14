//! Forensic watermarking for transcode
//!
//! This crate provides invisible watermarking for piracy tracking.

#![allow(clippy::needless_range_loop)]

use sha2::{Sha256, Digest};

mod error;
mod embed;
mod extract;

pub use error::*;
pub use embed::*;
pub use extract::*;

/// Result type for watermark operations
pub type Result<T> = std::result::Result<T, WatermarkError>;

/// Watermark configuration
#[derive(Debug, Clone)]
pub struct WatermarkConfig {
    /// Watermark strength (0.0-1.0)
    pub strength: f32,
    /// Use DCT domain embedding
    pub use_dct: bool,
    /// Block size for DCT
    pub block_size: usize,
    /// Redundancy factor (repeat watermark)
    pub redundancy: usize,
    /// Error correction level
    pub error_correction: ErrorCorrectionLevel,
}

impl Default for WatermarkConfig {
    fn default() -> Self {
        Self {
            strength: 0.1,
            use_dct: true,
            block_size: 8,
            redundancy: 3,
            error_correction: ErrorCorrectionLevel::Medium,
        }
    }
}

/// Error correction level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCorrectionLevel {
    /// No error correction
    None,
    /// Low redundancy
    Low,
    /// Medium redundancy
    Medium,
    /// High redundancy
    High,
}

/// Watermark payload
#[derive(Debug, Clone)]
pub struct WatermarkPayload {
    /// Unique identifier
    pub id: String,
    /// Timestamp (Unix epoch)
    pub timestamp: u64,
    /// Custom data
    pub data: Vec<u8>,
    /// Payload hash
    pub hash: [u8; 32],
}

impl WatermarkPayload {
    /// Create a new watermark payload
    pub fn new(id: &str, data: &[u8]) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let mut payload = Self {
            id: id.to_string(),
            timestamp,
            data: data.to_vec(),
            hash: [0u8; 32],
        };

        payload.compute_hash();
        payload
    }

    fn compute_hash(&mut self) {
        let mut hasher = Sha256::new();
        hasher.update(self.id.as_bytes());
        hasher.update(self.timestamp.to_le_bytes());
        hasher.update(&self.data);
        self.hash = hasher.finalize().into();
    }

    /// Serialize payload to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // ID length and data
        let id_bytes = self.id.as_bytes();
        bytes.extend_from_slice(&(id_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(id_bytes);

        // Timestamp
        bytes.extend_from_slice(&self.timestamp.to_le_bytes());

        // Data length and data
        bytes.extend_from_slice(&(self.data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.data);

        // Hash
        bytes.extend_from_slice(&self.hash);

        bytes
    }

    /// Deserialize payload from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 48 {
            return Err(WatermarkError::InvalidPayload("too short".into()));
        }

        let mut pos = 0;

        // ID length
        if pos + 4 > bytes.len() {
            return Err(WatermarkError::InvalidPayload("truncated id length".into()));
        }
        let id_len = u32::from_le_bytes(
            bytes[pos..pos + 4]
                .try_into()
                .map_err(|_| WatermarkError::InvalidPayload("invalid id length bytes".into()))?,
        ) as usize;
        pos += 4;

        if pos + id_len > bytes.len() {
            return Err(WatermarkError::InvalidPayload("invalid id length".into()));
        }

        let id = String::from_utf8(bytes[pos..pos + id_len].to_vec())
            .map_err(|_| WatermarkError::InvalidPayload("invalid id".into()))?;
        pos += id_len;

        // Timestamp
        if pos + 8 > bytes.len() {
            return Err(WatermarkError::InvalidPayload("truncated timestamp".into()));
        }
        let timestamp = u64::from_le_bytes(
            bytes[pos..pos + 8]
                .try_into()
                .map_err(|_| WatermarkError::InvalidPayload("invalid timestamp bytes".into()))?,
        );
        pos += 8;

        // Data length
        if pos + 4 > bytes.len() {
            return Err(WatermarkError::InvalidPayload("truncated data length".into()));
        }
        let data_len = u32::from_le_bytes(
            bytes[pos..pos + 4]
                .try_into()
                .map_err(|_| WatermarkError::InvalidPayload("invalid data length bytes".into()))?,
        ) as usize;
        pos += 4;

        if pos + data_len + 32 > bytes.len() {
            return Err(WatermarkError::InvalidPayload("invalid data length".into()));
        }

        let data = bytes[pos..pos + data_len].to_vec();
        pos += data_len;

        // Hash
        let hash: [u8; 32] = bytes[pos..pos + 32]
            .try_into()
            .map_err(|_| WatermarkError::InvalidPayload("invalid hash bytes".into()))?;

        let mut payload = Self {
            id,
            timestamp,
            data,
            hash: [0u8; 32],
        };

        payload.compute_hash();

        // Verify hash
        if payload.hash != hash {
            return Err(WatermarkError::InvalidPayload("hash mismatch".into()));
        }

        payload.hash = hash;
        Ok(payload)
    }
}

/// Forensic watermarker
pub struct Watermarker {
    config: WatermarkConfig,
    key: [u8; 32],
}

impl Watermarker {
    /// Create a new watermarker with random key
    pub fn new(config: WatermarkConfig) -> Self {
        use rand::RngCore;
        let mut key = [0u8; 32];
        rand::rng().fill_bytes(&mut key);
        Self { config, key }
    }

    /// Create with specific key
    pub fn with_key(config: WatermarkConfig, key: [u8; 32]) -> Self {
        Self { config, key }
    }

    /// Embed watermark in frame data (Y plane)
    pub fn embed(&self, frame_data: &mut [u8], width: usize, height: usize, payload: &WatermarkPayload) -> Result<()> {
        let payload_bits = self.payload_to_bits(payload);

        if self.config.use_dct {
            self.embed_dct(frame_data, width, height, &payload_bits)
        } else {
            self.embed_spatial(frame_data, width, height, &payload_bits)
        }
    }

    /// Extract watermark from frame data
    pub fn extract(&self, frame_data: &[u8], width: usize, height: usize) -> Result<WatermarkPayload> {
        let bits = if self.config.use_dct {
            self.extract_dct(frame_data, width, height)?
        } else {
            self.extract_spatial(frame_data, width, height)?
        };

        self.bits_to_payload(&bits)
    }

    fn payload_to_bits(&self, payload: &WatermarkPayload) -> Vec<bool> {
        let bytes = payload.to_bytes();
        let mut bits = Vec::with_capacity(bytes.len() * 8 * self.config.redundancy);

        for _ in 0..self.config.redundancy {
            for byte in &bytes {
                for i in 0..8 {
                    bits.push((byte >> i) & 1 == 1);
                }
            }
        }

        bits
    }

    fn bits_to_payload(&self, bits: &[bool]) -> Result<WatermarkPayload> {
        let bits_per_copy = bits.len() / self.config.redundancy;
        let bytes_count = bits_per_copy / 8;

        let mut bytes = vec![0u8; bytes_count];

        // Majority voting across redundant copies
        for byte_idx in 0..bytes_count {
            for bit_idx in 0..8 {
                let mut ones = 0;
                let pos = byte_idx * 8 + bit_idx;

                for copy in 0..self.config.redundancy {
                    if bits.get(copy * bits_per_copy + pos).copied().unwrap_or(false) {
                        ones += 1;
                    }
                }

                if ones > self.config.redundancy / 2 {
                    bytes[byte_idx] |= 1 << bit_idx;
                }
            }
        }

        WatermarkPayload::from_bytes(&bytes)
    }

    fn embed_spatial(&self, frame_data: &mut [u8], _width: usize, _height: usize, bits: &[bool]) -> Result<()> {
        let strength = (self.config.strength * 4.0) as i16;

        for (i, &bit) in bits.iter().enumerate() {
            let pos = self.get_embed_position(i, frame_data.len());
            let current = frame_data[pos] as i16;

            let new_val = if bit {
                (current + strength).clamp(0, 255)
            } else {
                (current - strength).clamp(0, 255)
            };

            frame_data[pos] = new_val as u8;
        }

        Ok(())
    }

    fn extract_spatial(&self, frame_data: &[u8], _width: usize, _height: usize) -> Result<Vec<bool>> {
        // Simplified extraction - real implementation would use correlation
        let estimated_bits = frame_data.len() / 64;
        let mut bits = Vec::with_capacity(estimated_bits);

        for i in 0..estimated_bits {
            let pos = self.get_embed_position(i, frame_data.len());
            bits.push(frame_data[pos] & 1 == 1);
        }

        Ok(bits)
    }

    fn embed_dct(&self, frame_data: &mut [u8], width: usize, height: usize, bits: &[bool]) -> Result<()> {
        let block_size = self.config.block_size;
        let blocks_x = width / block_size;
        let blocks_y = height / block_size;

        let mut bit_idx = 0;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                if bit_idx >= bits.len() {
                    break;
                }

                // Embed in middle-frequency DCT coefficient (simplified)
                let x = bx * block_size + block_size / 2;
                let y = by * block_size + block_size / 2;
                let pos = y * width + x;

                if pos < frame_data.len() {
                    let strength = (self.config.strength * 8.0) as i16;
                    let current = frame_data[pos] as i16;

                    let new_val = if bits[bit_idx] {
                        (current + strength).clamp(0, 255)
                    } else {
                        (current - strength).clamp(0, 255)
                    };

                    frame_data[pos] = new_val as u8;
                }

                bit_idx += 1;
            }
        }

        Ok(())
    }

    fn extract_dct(&self, frame_data: &[u8], width: usize, height: usize) -> Result<Vec<bool>> {
        let block_size = self.config.block_size;
        let blocks_x = width / block_size;
        let blocks_y = height / block_size;

        let mut bits = Vec::with_capacity(blocks_x * blocks_y);

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let x = bx * block_size + block_size / 2;
                let y = by * block_size + block_size / 2;
                let pos = y * width + x;

                if pos < frame_data.len() {
                    // Simplified extraction using local comparison
                    let center = frame_data[pos] as i16;
                    let neighbor = frame_data.get(pos + 1).copied().unwrap_or(128) as i16;
                    bits.push(center > neighbor);
                }
            }
        }

        Ok(bits)
    }

    fn get_embed_position(&self, index: usize, max_len: usize) -> usize {
        // Pseudo-random position based on key
        let mut hasher = Sha256::new();
        hasher.update(self.key);
        hasher.update((index as u64).to_le_bytes());
        let hash = hasher.finalize();

        let offset = u64::from_le_bytes(hash[0..8].try_into().unwrap());
        (offset as usize) % max_len
    }

    /// Get configuration
    pub fn config(&self) -> &WatermarkConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_payload_roundtrip() {
        let payload = WatermarkPayload::new("test-user-123", b"custom data");
        let bytes = payload.to_bytes();
        let restored = WatermarkPayload::from_bytes(&bytes).unwrap();

        assert_eq!(restored.id, "test-user-123");
        assert_eq!(restored.data, b"custom data");
    }

    #[test]
    fn test_watermarker_basic() {
        let mut config = WatermarkConfig::default();
        config.strength = 0.5; // Higher strength so changes are visible
        let watermarker = Watermarker::new(config);

        let mut frame = vec![128u8; 1920 * 1080];
        let payload = WatermarkPayload::new("user-1", b"data");

        watermarker.embed(&mut frame, 1920, 1080, &payload).unwrap();

        // Frame should be modified but still valid
        assert!(frame.iter().any(|&v| v != 128));
    }
}
