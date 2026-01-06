//! Huffman coding for JPEG.

use crate::error::{ImageError, Result};

/// Huffman table.
#[derive(Debug, Clone)]
pub struct HuffmanTable {
    /// Code lengths for each symbol.
    pub bits: [u8; 17],
    /// Symbol values.
    pub huffval: Vec<u8>,
    /// Minimum codes for each length.
    pub mincode: [i32; 17],
    /// Maximum codes for each length.
    pub maxcode: [i32; 17],
    /// Value offset for each length.
    pub valptr: [i32; 17],
    /// Table class (0 = DC, 1 = AC).
    pub class: u8,
    /// Table ID (0-3).
    pub id: u8,
}

impl HuffmanTable {
    /// Create an empty Huffman table.
    pub fn new(class: u8, id: u8) -> Self {
        Self {
            bits: [0; 17],
            huffval: Vec::new(),
            mincode: [0; 17],
            maxcode: [-1; 17],
            valptr: [0; 17],
            class,
            id,
        }
    }

    /// Build lookup tables from bits and huffval.
    pub fn build_lookup(&mut self) {
        let mut code: i32 = 0;
        let mut huffsize = Vec::new();
        let mut huffcode = Vec::new();

        // Generate size table
        for i in 1..=16 {
            for _ in 0..self.bits[i] {
                huffsize.push(i as u8);
            }
        }

        // Generate code table
        for &size in &huffsize {
            while huffcode.len() > 0 && huffsize[huffcode.len() - 1] != size {
                code <<= 1;
            }
            huffcode.push(code);
            code += 1;
        }

        // Generate decoding tables
        let mut j = 0;
        for i in 1..=16 {
            if self.bits[i] == 0 {
                self.maxcode[i] = -1;
            } else {
                self.valptr[i] = j as i32;
                self.mincode[i] = huffcode.get(j).copied().unwrap_or(0);
                j += self.bits[i] as usize;
                self.maxcode[i] = huffcode.get(j - 1).copied().unwrap_or(0);
            }
        }
    }

    /// Decode a symbol using this table.
    pub fn decode(&self, get_bit: &mut impl FnMut() -> Result<u8>) -> Result<u8> {
        let mut code: i32 = get_bit()? as i32;
        let mut size = 1;

        while code > self.maxcode[size] && size <= 16 {
            code = (code << 1) | get_bit()? as i32;
            size += 1;
        }

        if size > 16 {
            return Err(ImageError::DecoderError("Invalid Huffman code".into()));
        }

        let idx = (self.valptr[size] + code - self.mincode[size]) as usize;
        if idx >= self.huffval.len() {
            return Err(ImageError::DecoderError("Huffman value out of range".into()));
        }

        Ok(self.huffval[idx])
    }
}

/// Huffman encoder.
#[derive(Debug, Clone)]
pub struct HuffmanEncoder {
    /// Encoding table: (code, length) for each symbol.
    pub codes: [(u16, u8); 256],
}

impl HuffmanEncoder {
    /// Create encoder from table.
    pub fn from_table(table: &HuffmanTable) -> Self {
        let mut codes = [(0u16, 0u8); 256];
        let mut code: u16 = 0;
        let mut idx = 0;

        for i in 1..=16 {
            for _ in 0..table.bits[i] {
                if idx < table.huffval.len() {
                    let sym = table.huffval[idx] as usize;
                    codes[sym] = (code, i as u8);
                    idx += 1;
                }
                code += 1;
            }
            code <<= 1;
        }

        Self { codes }
    }

    /// Encode a symbol.
    pub fn encode(&self, symbol: u8) -> (u16, u8) {
        self.codes[symbol as usize]
    }
}

/// Standard DC luminance Huffman table.
pub fn dc_luminance_table() -> HuffmanTable {
    let mut table = HuffmanTable::new(0, 0);
    table.bits = [0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
    table.huffval = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    table.build_lookup();
    table
}

/// Standard DC chrominance Huffman table.
pub fn dc_chrominance_table() -> HuffmanTable {
    let mut table = HuffmanTable::new(0, 1);
    table.bits = [0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
    table.huffval = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    table.build_lookup();
    table
}

/// Standard AC luminance Huffman table.
pub fn ac_luminance_table() -> HuffmanTable {
    let mut table = HuffmanTable::new(1, 0);
    table.bits = [0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125];
    table.huffval = vec![
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
        0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
        0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
        0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
        0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16,
        0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
        0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
        0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
        0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
        0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
        0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
        0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
        0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
        0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4,
        0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
        0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
        0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
        0xF9, 0xFA,
    ];
    table.build_lookup();
    table
}

/// Standard AC chrominance Huffman table.
pub fn ac_chrominance_table() -> HuffmanTable {
    let mut table = HuffmanTable::new(1, 1);
    table.bits = [0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119];
    table.huffval = vec![
        0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
        0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
        0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
        0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
        0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34,
        0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
        0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38,
        0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
        0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
        0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
        0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
        0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
        0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96,
        0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
        0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4,
        0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
        0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2,
        0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
        0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9,
        0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
        0xF9, 0xFA,
    ];
    table.build_lookup();
    table
}

/// Category and bits needed for a value.
pub fn category_encode(value: i16) -> (u8, u16) {
    let abs_value = value.unsigned_abs();
    let category = if abs_value == 0 {
        0
    } else {
        16 - abs_value.leading_zeros() as u8
    };

    let bits = if value < 0 {
        (abs_value - 1) ^ ((1 << category) - 1)
    } else {
        abs_value
    };

    (category, bits)
}

/// Decode value from category and bits.
pub fn category_decode(category: u8, bits: u16) -> i16 {
    if category == 0 {
        return 0;
    }

    let threshold = 1 << (category - 1);
    if bits as i32 >= threshold {
        bits as i16
    } else {
        bits as i16 - (1 << category) + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huffman_table_creation() {
        let table = dc_luminance_table();
        assert_eq!(table.class, 0);
        assert_eq!(table.id, 0);
        assert!(!table.huffval.is_empty());
    }

    #[test]
    fn test_category_encode() {
        assert_eq!(category_encode(0), (0, 0));
        assert_eq!(category_encode(1), (1, 1));
        assert_eq!(category_encode(-1), (1, 0));
        assert_eq!(category_encode(127), (7, 127));
        assert_eq!(category_encode(-127), (7, 0));
    }

    #[test]
    fn test_category_roundtrip() {
        for val in -255..=255 {
            let (cat, bits) = category_encode(val);
            let decoded = category_decode(cat, bits);
            assert_eq!(val, decoded, "Failed for value {}", val);
        }
    }

    #[test]
    fn test_encoder_creation() {
        let table = dc_luminance_table();
        let encoder = HuffmanEncoder::from_table(&table);

        // Check that symbol 0 has a valid encoding
        let (code, len) = encoder.encode(0);
        assert!(len > 0);
    }
}
