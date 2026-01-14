//! Vorbis codebook implementation.
//!
//! Codebooks are used for Huffman decoding and vector quantization in Vorbis.

use crate::error::{VorbisError, Result};

/// Vorbis codebook entry.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CodebookEntry {
    /// Codeword length in bits.
    pub length: u8,
    /// Codeword value.
    pub codeword: u32,
    /// Quantized values for VQ (if applicable).
    pub values: Vec<f32>,
}

/// Vorbis codebook.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Codebook {
    /// Codebook dimensions (for VQ).
    dimensions: u32,
    /// Number of entries.
    entries: u32,
    /// Codebook entries.
    entry_list: Vec<CodebookEntry>,
    /// Lookup table for fast decoding.
    lookup_table: Vec<i32>,
    /// Lookup type (0 = no lookup, 1 = implicit, 2 = explicit).
    lookup_type: u8,
    /// Minimum value for VQ.
    minimum_value: f32,
    /// Delta value for VQ.
    delta_value: f32,
    /// Value bits for VQ.
    value_bits: u8,
    /// Sequence P flag.
    sequence_p: bool,
}

#[allow(dead_code)]
impl Codebook {
    /// Create an empty codebook.
    pub fn new() -> Self {
        Self {
            dimensions: 0,
            entries: 0,
            entry_list: Vec::new(),
            lookup_table: Vec::new(),
            lookup_type: 0,
            minimum_value: 0.0,
            delta_value: 0.0,
            value_bits: 0,
            sequence_p: false,
        }
    }

    /// Parse a codebook from bitstream.
    pub fn parse(data: &[u8], offset: &mut usize) -> Result<Self> {
        let mut codebook = Self::new();

        // Read sync pattern (0x564342 = "BCV")
        if *offset + 3 > data.len() {
            return Err(VorbisError::InvalidCodebook("Unexpected end of data".into()));
        }

        let sync = (data[*offset] as u32)
            | ((data[*offset + 1] as u32) << 8)
            | ((data[*offset + 2] as u32) << 16);
        *offset += 3;

        if sync != 0x564342 {
            return Err(VorbisError::InvalidCodebook(format!(
                "Invalid sync pattern: {:#x}",
                sync
            )));
        }

        // For now, use a simplified parsing that creates a basic codebook
        // A full implementation would parse all the codebook fields
        codebook.dimensions = 1;
        codebook.entries = 256;
        codebook.entry_list = (0..256)
            .map(|i| CodebookEntry {
                length: 8,
                codeword: i as u32,
                values: vec![i as f32 / 255.0],
            })
            .collect();

        Ok(codebook)
    }

    /// Create a codebook from configuration.
    pub fn from_config(
        dimensions: u32,
        entries: u32,
        lengths: &[u8],
        lookup_type: u8,
        min_val: f32,
        delta: f32,
    ) -> Result<Self> {
        if lengths.len() != entries as usize {
            return Err(VorbisError::InvalidCodebook(
                "Length mismatch".into(),
            ));
        }

        let mut entry_list = Vec::with_capacity(entries as usize);
        let mut codeword = 0u32;

        for (i, &length) in lengths.iter().enumerate() {
            if length == 0 {
                continue;
            }

            let values = Self::compute_vq_values(
                i as u32,
                dimensions,
                min_val,
                delta,
                lookup_type,
            );

            entry_list.push(CodebookEntry {
                length,
                codeword,
                values,
            });

            // Increment codeword (simplified canonical Huffman)
            codeword += 1;
        }

        Ok(Self {
            dimensions,
            entries,
            entry_list,
            lookup_table: Vec::new(),
            lookup_type,
            minimum_value: min_val,
            delta_value: delta,
            value_bits: 8,
            sequence_p: false,
        })
    }

    /// Compute VQ values for an entry.
    fn compute_vq_values(
        entry: u32,
        dimensions: u32,
        min_val: f32,
        delta: f32,
        _lookup_type: u8,
    ) -> Vec<f32> {
        let mut values = Vec::with_capacity(dimensions as usize);
        let mut idx = entry;

        for _ in 0..dimensions {
            let val = min_val + delta * (idx % 256) as f32;
            values.push(val);
            idx /= 256;
        }

        values
    }

    /// Decode a value using this codebook.
    pub fn decode(&self, bits: u32, bit_count: u8) -> Option<&CodebookEntry> {
        // Linear search (a real implementation would use lookup tables)
        self.entry_list
            .iter()
            .find(|entry| entry.length == bit_count && entry.codeword == bits)
    }

    /// Get codebook dimensions.
    pub fn dimensions(&self) -> u32 {
        self.dimensions
    }

    /// Get number of entries.
    pub fn entries(&self) -> u32 {
        self.entries
    }

    /// Get lookup type.
    pub fn lookup_type(&self) -> u8 {
        self.lookup_type
    }

    /// Build lookup table for faster decoding.
    pub fn build_lookup_table(&mut self, max_bits: u8) {
        let table_size = 1 << max_bits;
        self.lookup_table = vec![-1; table_size];

        for (i, entry) in self.entry_list.iter().enumerate() {
            if entry.length <= max_bits && entry.length > 0 {
                let base = entry.codeword << (max_bits - entry.length);
                let count = 1 << (max_bits - entry.length);
                for j in 0..count {
                    self.lookup_table[(base + j) as usize] = i as i32;
                }
            }
        }
    }

    /// Fast decode using lookup table.
    pub fn lookup_decode(&self, bits: u32) -> Option<(usize, u8)> {
        if self.lookup_table.is_empty() {
            return None;
        }

        let idx = self.lookup_table.get(bits as usize)?;
        if *idx < 0 {
            return None;
        }

        let entry = &self.entry_list[*idx as usize];
        Some((*idx as usize, entry.length))
    }
}

impl Default for Codebook {
    fn default() -> Self {
        Self::new()
    }
}

/// Codebook configuration for encoding.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CodebookConfig {
    /// Number of dimensions.
    pub dimensions: u32,
    /// Number of entries.
    pub entries: u32,
    /// Lookup type.
    pub lookup_type: u8,
    /// Minimum value.
    pub min_value: f32,
    /// Delta value.
    pub delta_value: f32,
}

#[allow(dead_code)]
impl CodebookConfig {
    /// Create a new codebook configuration.
    pub fn new(dimensions: u32, entries: u32) -> Self {
        Self {
            dimensions,
            entries,
            lookup_type: 1,
            min_value: -1.0,
            delta_value: 2.0 / entries as f32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_creation() {
        let codebook = Codebook::new();
        assert_eq!(codebook.dimensions(), 0);
        assert_eq!(codebook.entries(), 0);
    }

    #[test]
    fn test_codebook_from_config() {
        let lengths: Vec<u8> = vec![8; 256];
        let codebook = Codebook::from_config(
            1,
            256,
            &lengths,
            1,
            -1.0,
            2.0 / 256.0,
        ).unwrap();

        assert_eq!(codebook.dimensions(), 1);
        assert_eq!(codebook.entries(), 256);
    }

    #[test]
    fn test_lookup_table() {
        let lengths: Vec<u8> = (0..16).map(|_| 4).collect();
        let mut codebook = Codebook::from_config(
            1,
            16,
            &lengths,
            0,
            0.0,
            1.0,
        ).unwrap();

        codebook.build_lookup_table(8);
        assert!(!codebook.lookup_table.is_empty());
    }
}
