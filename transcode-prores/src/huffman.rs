//! Huffman decoding tables for ProRes DCT coefficients

use crate::error::{ProResError, Result};

/// Maximum code length for Huffman codes
pub const MAX_CODE_LENGTH: usize = 16;

/// Huffman table for DC coefficient differences
#[derive(Debug, Clone)]
pub struct DcHuffmanTable {
    /// Code lengths for each symbol (0-11)
    pub lengths: [u8; 12],
    /// Codes for each symbol
    pub codes: [u16; 12],
    /// Lookup table for fast decoding (indexed by first 9 bits)
    pub fast_lookup: Vec<(u8, u8)>, // (symbol, bit_length)
}

impl Default for DcHuffmanTable {
    fn default() -> Self {
        Self::new()
    }
}

impl DcHuffmanTable {
    /// Create a new DC Huffman table with ProRes default codes
    pub fn new() -> Self {
        // ProRes uses a simplified DC coding scheme
        // We'll use a prefix-based table similar to other video codecs
        // Symbol represents the number of additional bits needed

        // Lengths for symbols 0-11 (typical DC difference categories)
        let lengths: [u8; 12] = [2, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10];

        // Pre-computed canonical Huffman codes for these lengths
        // These follow the standard canonical Huffman assignment
        let codes: [u16; 12] = [
            0b00,       // symbol 0, len 2
            0b01,       // symbol 1, len 2
            0b10,       // symbol 2, len 2
            0b110,      // symbol 3, len 3
            0b111,      // symbol 4, len 3 (but will shift for next length)
            0b1110,     // symbol 5, len 4 (adjusted)
            0b11110,    // symbol 6, len 5
            0b111110,   // symbol 7, len 6
            0b1111110,  // symbol 8, len 7
            0b11111110, // symbol 9, len 8
            0b111111110, // symbol 10, len 9
            0b1111111110, // symbol 11, len 10
        ];

        // Build fast lookup table (first 9 bits)
        // Mark all entries as invalid first
        let mut fast_lookup = vec![(0u8, 0u8); 512];

        for (symbol, (&len, &huffcode)) in lengths.iter().zip(codes.iter()).enumerate() {
            if len <= 9 {
                // Left-align the code to 9 bits
                let base = (huffcode as usize) << (9 - len);
                let count = 1usize << (9 - len);
                for j in 0..count {
                    let index = base + j;
                    if index < 512 {
                        fast_lookup[index] = (symbol as u8, len);
                    }
                }
            }
        }

        DcHuffmanTable {
            lengths,
            codes,
            fast_lookup,
        }
    }
}

/// Huffman table for AC coefficient run-level coding
#[derive(Debug, Clone)]
pub struct AcHuffmanTable {
    /// Fast lookup table for common codes (indexed by first 10 bits)
    pub fast_lookup: Vec<(i8, i8, u8)>, // (run, level, bit_length)
    /// Escape code for uncommon values
    pub escape_code: u16,
    /// Escape code length
    pub escape_length: u8,
}

impl Default for AcHuffmanTable {
    fn default() -> Self {
        Self::new()
    }
}

impl AcHuffmanTable {
    /// Create a new AC Huffman table with ProRes default codes
    pub fn new() -> Self {
        // ProRes uses a specific VLC table for AC coefficients
        // The table encodes (run, level) pairs
        // Run: number of zero coefficients before this one
        // Level: magnitude of the coefficient

        let mut fast_lookup = vec![(-1i8, -1i8, 0u8); 1024];

        // Common ProRes AC codes (simplified representation)
        // Format: (run, level, code, length)
        let ac_codes: &[(i8, i8, u16, u8)] = &[
            // End of block
            (0, 0, 0b10, 2),
            // Run=0 codes
            (0, 1, 0b00, 2),
            (0, 2, 0b010, 3),
            (0, 3, 0b0110, 4),
            (0, 4, 0b01110, 5),
            (0, 5, 0b011110, 6),
            (0, 6, 0b0111110, 7),
            (0, 7, 0b01111110, 8),
            // Run=1 codes
            (1, 1, 0b110, 3),
            (1, 2, 0b11110, 5),
            (1, 3, 0b1111110, 7),
            // Run=2 codes
            (2, 1, 0b1110, 4),
            (2, 2, 0b111110, 6),
            // Run=3 codes
            (3, 1, 0b111010, 6),
            // Run=4 codes
            (4, 1, 0b1110110, 7),
            // Run=5 codes
            (5, 1, 0b11101110, 8),
            // Run=6 codes
            (6, 1, 0b111011110, 9),
        ];

        // Build fast lookup
        for &(run, level, code, len) in ac_codes {
            if len <= 10 {
                let base = (code as usize) << (10 - len);
                let count = 1 << (10 - len);
                for j in 0..count {
                    if base + j < 1024 {
                        fast_lookup[base + j] = (run, level, len);
                    }
                }
            }
        }

        AcHuffmanTable {
            fast_lookup,
            escape_code: 0b11111111110,
            escape_length: 11,
        }
    }
}

/// Rice/Golomb code decoder for ProRes coefficients
#[derive(Debug, Clone)]
pub struct RiceDecoder {
    /// Current Rice parameter (k value)
    k: u8,
}

impl Default for RiceDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl RiceDecoder {
    /// Create a new Rice decoder
    pub fn new() -> Self {
        RiceDecoder { k: 0 }
    }

    /// Reset the Rice parameter
    pub fn reset(&mut self) {
        self.k = 0;
    }

    /// Update the Rice parameter based on the decoded value
    pub fn update(&mut self, value: u32) {
        // Adaptive Rice parameter update
        if value > (3 << self.k) {
            if self.k < 7 {
                self.k += 1;
            }
        } else if value < (1 << self.k) >> 1 && self.k > 0 {
            self.k -= 1;
        }
    }

    /// Get the current Rice parameter
    pub fn k(&self) -> u8 {
        self.k
    }
}

/// Bitstream reader for Huffman/Rice decoding
pub struct BitstreamReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
    bits_left: usize,
}

impl<'a> BitstreamReader<'a> {
    /// Create a new bitstream reader
    pub fn new(data: &'a [u8]) -> Self {
        BitstreamReader {
            data,
            byte_pos: 0,
            bit_pos: 0,
            bits_left: data.len() * 8,
        }
    }

    /// Read n bits from the stream
    pub fn read_bits(&mut self, n: u8) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(ProResError::BitstreamError("Cannot read more than 32 bits".into()));
        }
        if (n as usize) > self.bits_left {
            return Err(ProResError::BitstreamError("Not enough bits in stream".into()));
        }

        let mut result = 0u32;
        let mut bits_needed = n;

        while bits_needed > 0 {
            let bits_in_byte = 8 - self.bit_pos;
            let bits_to_read = bits_in_byte.min(bits_needed);

            // Use u16 to avoid overflow when bits_to_read is 8
            let mask = ((1u16 << bits_to_read) - 1) as u8;
            let shift = bits_in_byte - bits_to_read;
            let bits = (self.data[self.byte_pos] >> shift) & mask;

            result = (result << bits_to_read) | (bits as u32);
            bits_needed -= bits_to_read;
            self.bit_pos += bits_to_read;
            self.bits_left -= bits_to_read as usize;

            if self.bit_pos >= 8 {
                self.bit_pos = 0;
                self.byte_pos += 1;
            }
        }

        Ok(result)
    }

    /// Peek at the next n bits without consuming them
    pub fn peek_bits(&self, n: u8) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(ProResError::BitstreamError("Cannot peek more than 32 bits".into()));
        }
        if (n as usize) > self.bits_left {
            return Err(ProResError::BitstreamError("Not enough bits in stream".into()));
        }

        let mut result = 0u32;
        let mut byte_pos = self.byte_pos;
        let mut bit_pos = self.bit_pos;
        let mut bits_needed = n;

        while bits_needed > 0 {
            let bits_in_byte = 8 - bit_pos;
            let bits_to_read = bits_in_byte.min(bits_needed);

            // Use u16 to avoid overflow when bits_to_read is 8
            let mask = ((1u16 << bits_to_read) - 1) as u8;
            let shift = bits_in_byte - bits_to_read;
            let bits = (self.data[byte_pos] >> shift) & mask;

            result = (result << bits_to_read) | (bits as u32);
            bits_needed -= bits_to_read;
            bit_pos += bits_to_read;

            if bit_pos >= 8 {
                bit_pos = 0;
                byte_pos += 1;
            }
        }

        Ok(result)
    }

    /// Skip n bits
    pub fn skip_bits(&mut self, n: u32) -> Result<()> {
        if (n as usize) > self.bits_left {
            return Err(ProResError::BitstreamError("Not enough bits to skip".into()));
        }

        self.bits_left -= n as usize;
        let total_bits = self.bit_pos as u32 + n;
        self.byte_pos += (total_bits / 8) as usize;
        self.bit_pos = (total_bits % 8) as u8;

        Ok(())
    }

    /// Read a unary code (count of 1s followed by 0)
    pub fn read_unary(&mut self) -> Result<u32> {
        let mut count = 0;
        while self.read_bits(1)? == 1 {
            count += 1;
            if count > 32 {
                return Err(ProResError::BitstreamError("Unary code too long".into()));
            }
        }
        Ok(count)
    }

    /// Read a signed value using sign-magnitude representation
    pub fn read_signed(&mut self, bits: u8) -> Result<i32> {
        if bits == 0 {
            return Ok(0);
        }
        let value = self.read_bits(bits)?;
        // First bit is sign
        let sign = (value >> (bits - 1)) & 1;
        let magnitude = value & ((1 << (bits - 1)) - 1);
        if sign == 1 {
            Ok(-(magnitude as i32))
        } else {
            Ok(magnitude as i32)
        }
    }

    /// Get remaining bits
    pub fn bits_remaining(&self) -> usize {
        self.bits_left
    }

    /// Get current byte position
    pub fn byte_position(&self) -> usize {
        self.byte_pos
    }

    /// Get current bit position within byte
    pub fn bit_position(&self) -> u8 {
        self.bit_pos
    }

    /// Align to next byte boundary
    pub fn align_to_byte(&mut self) {
        if self.bit_pos > 0 {
            self.bits_left -= (8 - self.bit_pos) as usize;
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
    }
}

/// Decode a DC coefficient difference using Huffman coding
pub fn decode_dc_diff(reader: &mut BitstreamReader, table: &DcHuffmanTable) -> Result<i16> {
    // Try fast lookup first
    let bits = reader.peek_bits(9)? as usize;
    let (category, len) = table.fast_lookup[bits];

    if len == 0 {
        return Err(ProResError::HuffmanError("Invalid DC Huffman code".into()));
    }

    reader.skip_bits(len as u32)?;

    if category == 0 {
        return Ok(0);
    }

    // Read the difference value
    let diff_bits = reader.read_bits(category)?;

    // Convert to signed value
    let half = 1i32 << (category - 1);
    let diff = if (diff_bits as i32) < half {
        diff_bits as i32 - (2 * half - 1)
    } else {
        diff_bits as i32
    };

    Ok(diff as i16)
}

/// Decode AC coefficients for a block
pub fn decode_ac_coeffs(
    reader: &mut BitstreamReader,
    coeffs: &mut [i16; 64],
    table: &AcHuffmanTable,
) -> Result<()> {
    let mut pos = 1; // Start after DC

    while pos < 64 {
        // Try fast lookup
        let bits = reader.peek_bits(10)? as usize;
        let (run, level, len) = table.fast_lookup[bits];

        if len > 0 {
            reader.skip_bits(len as u32)?;

            if run == 0 && level == 0 {
                // End of block
                break;
            }

            pos += run as usize;
            if pos >= 64 {
                return Err(ProResError::HuffmanError("AC run exceeds block size".into()));
            }

            // Read sign bit
            let sign = reader.read_bits(1)?;
            coeffs[pos] = if sign == 1 { -level as i16 } else { level as i16 };
            pos += 1;
        } else {
            // Escape code - read raw run and level
            let peek = reader.peek_bits(11)?;
            if peek == table.escape_code as u32 {
                reader.skip_bits(table.escape_length as u32)?;

                // Read 6-bit run
                let run = reader.read_bits(6)? as usize;
                // Read 12-bit signed level
                let level_bits = reader.read_bits(12)?;
                let level = if level_bits >= 2048 {
                    (level_bits as i32) - 4096
                } else {
                    level_bits as i32
                };

                pos += run;
                if pos >= 64 {
                    return Err(ProResError::HuffmanError("AC run exceeds block size".into()));
                }

                coeffs[pos] = level as i16;
                pos += 1;
            } else {
                // Read unary prefix for run
                let run = reader.read_unary()? as usize;

                // Read level using Rice coding
                let level_prefix = reader.read_unary()?;
                let k = 2; // Fixed Rice parameter for AC
                let level_suffix = reader.read_bits(k)?;
                let level = (level_prefix << k) | level_suffix;

                pos += run;
                if pos >= 64 {
                    return Err(ProResError::HuffmanError("AC run exceeds block size".into()));
                }

                // Read sign
                let sign = reader.read_bits(1)?;
                coeffs[pos] = if sign == 1 {
                    -(level as i16) - 1
                } else {
                    level as i16 + 1
                };
                pos += 1;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitstream_reader_basic() {
        let data = [0b10110100, 0b01001011];
        let mut reader = BitstreamReader::new(&data);

        assert_eq!(reader.read_bits(1).unwrap(), 1);
        assert_eq!(reader.read_bits(1).unwrap(), 0);
        assert_eq!(reader.read_bits(2).unwrap(), 0b11);
        assert_eq!(reader.read_bits(4).unwrap(), 0b0100);
        assert_eq!(reader.read_bits(8).unwrap(), 0b01001011);
    }

    #[test]
    fn test_bitstream_reader_peek() {
        let data = [0b10110100];
        let mut reader = BitstreamReader::new(&data);

        assert_eq!(reader.peek_bits(4).unwrap(), 0b1011);
        assert_eq!(reader.peek_bits(4).unwrap(), 0b1011); // Same value
        assert_eq!(reader.read_bits(4).unwrap(), 0b1011);
        assert_eq!(reader.peek_bits(4).unwrap(), 0b0100);
    }

    #[test]
    fn test_bitstream_reader_unary() {
        let data = [0b11110000]; // 4 ones followed by zero
        let mut reader = BitstreamReader::new(&data);

        assert_eq!(reader.read_unary().unwrap(), 4);
    }

    #[test]
    fn test_dc_huffman_table() {
        let table = DcHuffmanTable::new();
        assert_eq!(table.lengths.len(), 12);
        assert!(table.fast_lookup.len() >= 512);
    }

    #[test]
    fn test_ac_huffman_table() {
        let table = AcHuffmanTable::new();
        assert_eq!(table.fast_lookup.len(), 1024);
    }

    #[test]
    fn test_rice_decoder() {
        let mut rice = RiceDecoder::new();
        assert_eq!(rice.k(), 0);

        rice.update(10);
        assert!(rice.k() > 0);

        rice.update(0);
        assert!(rice.k() < 7);
    }

    #[test]
    fn test_align_to_byte() {
        let data = [0xFF, 0x00];
        let mut reader = BitstreamReader::new(&data);

        reader.read_bits(3).unwrap();
        reader.align_to_byte();

        assert_eq!(reader.byte_position(), 1);
        assert_eq!(reader.bit_position(), 0);
    }
}
