//! OpenEXR compression methods

use crate::error::{ExrError, Result};

/// OpenEXR compression types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    /// No compression
    None,
    /// Run-length encoding
    Rle,
    /// ZIPS (per-scanline ZIP)
    ZipS,
    /// ZIP (multi-scanline, default 16 lines)
    Zip,
    /// PIZ (wavelet-based lossless)
    Piz,
    /// PXR24 (24-bit float precision)
    Pxr24,
    /// B44 (lossy 4x4 block)
    B44,
    /// B44A (B44 with alpha handling)
    B44a,
    /// DWAA (lossy DWA with alpha)
    Dwaa,
    /// DWAB (lossy DWA in blocks)
    Dwab,
}

impl Default for Compression {
    fn default() -> Self {
        Compression::Zip
    }
}

impl Compression {
    /// Create from u8 value
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Compression::None),
            1 => Some(Compression::Rle),
            2 => Some(Compression::ZipS),
            3 => Some(Compression::Zip),
            4 => Some(Compression::Piz),
            5 => Some(Compression::Pxr24),
            6 => Some(Compression::B44),
            7 => Some(Compression::B44a),
            8 => Some(Compression::Dwaa),
            9 => Some(Compression::Dwab),
            _ => None,
        }
    }

    /// Convert to u8
    pub fn to_u8(self) -> u8 {
        match self {
            Compression::None => 0,
            Compression::Rle => 1,
            Compression::ZipS => 2,
            Compression::Zip => 3,
            Compression::Piz => 4,
            Compression::Pxr24 => 5,
            Compression::B44 => 6,
            Compression::B44a => 7,
            Compression::Dwaa => 8,
            Compression::Dwab => 9,
        }
    }

    /// Human-readable name
    pub fn name(self) -> &'static str {
        match self {
            Compression::None => "NONE",
            Compression::Rle => "RLE",
            Compression::ZipS => "ZIPS",
            Compression::Zip => "ZIP",
            Compression::Piz => "PIZ",
            Compression::Pxr24 => "PXR24",
            Compression::B44 => "B44",
            Compression::B44a => "B44A",
            Compression::Dwaa => "DWAA",
            Compression::Dwab => "DWAB",
        }
    }

    /// Number of scanlines per chunk for this compression
    pub fn scanlines_per_chunk(self) -> usize {
        match self {
            Compression::None | Compression::Rle | Compression::ZipS => 1,
            Compression::Zip | Compression::Piz | Compression::Pxr24 => 16,
            Compression::B44 | Compression::B44a | Compression::Dwaa => 32,
            Compression::Dwab => 256,
        }
    }

    /// Is this compression lossless?
    pub fn is_lossless(self) -> bool {
        match self {
            Compression::None
            | Compression::Rle
            | Compression::ZipS
            | Compression::Zip
            | Compression::Piz => true,
            Compression::Pxr24 | Compression::B44 | Compression::B44a => false,
            Compression::Dwaa | Compression::Dwab => false,
        }
    }
}

/// RLE compression
pub fn compress_rle(data: &[u8]) -> Result<Vec<u8>> {
    let mut output = Vec::with_capacity(data.len());
    let mut i = 0;

    while i < data.len() {
        let mut run_start = i;
        let current = data[i];

        // Count run length
        while i < data.len() && i - run_start < 127 && data[i] == current {
            i += 1;
        }

        let run_len = i - run_start;

        if run_len >= 3 {
            // Emit run: count (negative), value
            output.push((-(run_len as i8)) as u8);
            output.push(current);
        } else {
            // Look for literal run
            let literal_start = run_start;
            i = run_start;

            while i < data.len() {
                if i + 2 < data.len()
                    && data[i] == data[i + 1]
                    && data[i] == data[i + 2]
                {
                    break;
                }
                i += 1;
                if i - literal_start >= 127 {
                    break;
                }
            }

            let literal_len = i - literal_start;
            if literal_len > 0 {
                // Emit literal: count (positive), data
                output.push((literal_len - 1) as u8);
                output.extend_from_slice(&data[literal_start..i]);
            }
        }
    }

    Ok(output)
}

/// RLE decompression
pub fn decompress_rle(data: &[u8], output_size: usize) -> Result<Vec<u8>> {
    let mut output = Vec::with_capacity(output_size);
    let mut i = 0;

    while i < data.len() && output.len() < output_size {
        let count = data[i] as i8;
        i += 1;

        if count < 0 {
            // Run of repeated bytes
            let run_len = (-count) as usize;
            if i >= data.len() {
                return Err(ExrError::DecompressionError("RLE: unexpected end".into()));
            }
            let value = data[i];
            i += 1;

            for _ in 0..run_len {
                if output.len() >= output_size {
                    break;
                }
                output.push(value);
            }
        } else {
            // Literal run
            let literal_len = (count as usize) + 1;
            if i + literal_len > data.len() {
                return Err(ExrError::DecompressionError(
                    "RLE: not enough literal data".into(),
                ));
            }

            for j in 0..literal_len {
                if output.len() >= output_size {
                    break;
                }
                output.push(data[i + j]);
            }
            i += literal_len;
        }
    }

    if output.len() < output_size {
        output.resize(output_size, 0);
    }

    Ok(output)
}

/// Predict and reorder data for better compression (used by ZIP/PIZ)
pub fn reorder_for_compression(data: &[u8], channels: usize) -> Vec<u8> {
    if channels == 0 || data.is_empty() {
        return data.to_vec();
    }

    let bytes_per_channel = data.len() / channels;
    let mut output = vec![0u8; data.len()];

    // Separate channels and apply delta encoding
    for c in 0..channels {
        let src_offset = c * bytes_per_channel;
        let dst_offset = c;

        let mut prev = 0u8;
        for i in 0..bytes_per_channel {
            let current = data[src_offset + i];
            let delta = current.wrapping_sub(prev);
            output[dst_offset + i * channels] = delta;
            prev = current;
        }
    }

    output
}

/// Reverse reorder and predict (inverse of reorder_for_compression)
pub fn reorder_from_compression(data: &[u8], channels: usize) -> Vec<u8> {
    if channels == 0 || data.is_empty() {
        return data.to_vec();
    }

    let bytes_per_channel = data.len() / channels;
    let mut output = vec![0u8; data.len()];

    // Reverse delta encoding and deinterleave channels
    for c in 0..channels {
        let dst_offset = c * bytes_per_channel;
        let src_offset = c;

        let mut prev = 0u8;
        for i in 0..bytes_per_channel {
            let delta = data[src_offset + i * channels];
            let current = delta.wrapping_add(prev);
            output[dst_offset + i] = current;
            prev = current;
        }
    }

    output
}

/// Predictor for PIZ compression (XOR-based)
pub struct PizPredictor {
    tmp: Vec<u16>,
}

impl PizPredictor {
    pub fn new(size: usize) -> Self {
        PizPredictor { tmp: vec![0; size] }
    }

    /// Apply forward prediction (for compression)
    pub fn forward(&mut self, data: &mut [u16]) {
        if data.len() < 2 {
            return;
        }

        // Store original values
        self.tmp.clear();
        self.tmp.extend_from_slice(data);

        // XOR prediction
        let mut prev = self.tmp[0];
        data[0] = prev;

        for i in 1..self.tmp.len() {
            let current = self.tmp[i];
            data[i] = current ^ prev;
            prev = current;
        }
    }

    /// Apply reverse prediction (for decompression)
    pub fn reverse(&mut self, data: &mut [u16]) {
        if data.len() < 2 {
            return;
        }

        let mut prev = data[0];
        for i in 1..data.len() {
            prev ^= data[i];
            data[i] = prev;
        }
    }
}

/// Wavelet transform for PIZ compression (simplified)
pub struct Wavelet {
    tmp: Vec<u16>,
}

impl Wavelet {
    pub fn new(size: usize) -> Self {
        Wavelet { tmp: vec![0; size] }
    }

    /// Forward wavelet transform (Haar-like)
    pub fn forward(&mut self, data: &mut [u16], width: usize, height: usize) {
        if width < 2 || height < 2 {
            return;
        }

        self.tmp.resize(data.len(), 0);

        // Horizontal pass
        for y in 0..height {
            let row_start = y * width;
            for x in (0..width).step_by(2) {
                if x + 1 < width {
                    let a = data[row_start + x] as i32;
                    let b = data[row_start + x + 1] as i32;
                    let avg = ((a + b) >> 1) as u16;
                    let diff = ((a - b) & 0xFFFF) as u16;
                    self.tmp[row_start + x / 2] = avg;
                    self.tmp[row_start + width / 2 + x / 2] = diff;
                }
            }
        }

        // Vertical pass
        for x in 0..width {
            for y in (0..height).step_by(2) {
                if y + 1 < height {
                    let a = self.tmp[y * width + x] as i32;
                    let b = self.tmp[(y + 1) * width + x] as i32;
                    let avg = ((a + b) >> 1) as u16;
                    let diff = ((a - b) & 0xFFFF) as u16;
                    data[(y / 2) * width + x] = avg;
                    data[(height / 2 + y / 2) * width + x] = diff;
                }
            }
        }
    }

    /// Inverse wavelet transform
    pub fn inverse(&mut self, data: &mut [u16], width: usize, height: usize) {
        if width < 2 || height < 2 {
            return;
        }

        self.tmp.resize(data.len(), 0);

        // Inverse vertical pass
        for x in 0..width {
            for y in (0..height).step_by(2) {
                if y + 1 < height {
                    let avg = data[(y / 2) * width + x] as i32;
                    let diff = data[(height / 2 + y / 2) * width + x] as i32;
                    let a = avg + ((diff + 1) >> 1);
                    let b = avg - (diff >> 1);
                    self.tmp[y * width + x] = (a & 0xFFFF) as u16;
                    self.tmp[(y + 1) * width + x] = (b & 0xFFFF) as u16;
                }
            }
        }

        // Inverse horizontal pass
        for y in 0..height {
            let row_start = y * width;
            for x in (0..width).step_by(2) {
                if x + 1 < width {
                    let avg = self.tmp[row_start + x / 2] as i32;
                    let diff = self.tmp[row_start + width / 2 + x / 2] as i32;
                    let a = avg + ((diff + 1) >> 1);
                    let b = avg - (diff >> 1);
                    data[row_start + x] = (a & 0xFFFF) as u16;
                    data[row_start + x + 1] = (b & 0xFFFF) as u16;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_types() {
        assert_eq!(Compression::from_u8(0), Some(Compression::None));
        assert_eq!(Compression::from_u8(3), Some(Compression::Zip));
        assert_eq!(Compression::from_u8(4), Some(Compression::Piz));
        assert_eq!(Compression::from_u8(99), None);

        assert_eq!(Compression::Zip.to_u8(), 3);
        assert_eq!(Compression::Piz.name(), "PIZ");
    }

    #[test]
    fn test_scanlines_per_chunk() {
        assert_eq!(Compression::None.scanlines_per_chunk(), 1);
        assert_eq!(Compression::Rle.scanlines_per_chunk(), 1);
        assert_eq!(Compression::Zip.scanlines_per_chunk(), 16);
        assert_eq!(Compression::Piz.scanlines_per_chunk(), 16);
    }

    #[test]
    fn test_lossless() {
        assert!(Compression::None.is_lossless());
        assert!(Compression::Zip.is_lossless());
        assert!(Compression::Piz.is_lossless());
        assert!(!Compression::Pxr24.is_lossless());
        assert!(!Compression::B44.is_lossless());
    }

    #[test]
    fn test_rle_roundtrip() {
        let original = vec![1, 1, 1, 1, 1, 2, 3, 4, 5, 5, 5, 5];
        let compressed = compress_rle(&original).unwrap();
        let decompressed = decompress_rle(&compressed, original.len()).unwrap();
        assert_eq!(original, decompressed);
    }

    #[test]
    fn test_rle_literals() {
        let original = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let compressed = compress_rle(&original).unwrap();
        let decompressed = decompress_rle(&compressed, original.len()).unwrap();
        assert_eq!(original, decompressed);
    }

    #[test]
    fn test_reorder_roundtrip() {
        let original = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let channels = 3;
        let reordered = reorder_for_compression(&original, channels);
        let restored = reorder_from_compression(&reordered, channels);
        assert_eq!(original, restored);
    }

    #[test]
    fn test_piz_predictor() {
        let mut data = vec![100, 102, 105, 110, 108, 112];
        let original = data.clone();
        let mut pred = PizPredictor::new(data.len());

        pred.forward(&mut data);
        pred.reverse(&mut data);

        assert_eq!(data, original);
    }
}
