//! Huffman coding for DNxHD

#![allow(dead_code)]

use crate::error::{DnxError, Result};

/// Huffman code entry
#[derive(Debug, Clone, Copy)]
pub struct HuffmanCode {
    /// Code bits
    pub code: u32,
    /// Number of bits in code
    pub bits: u8,
}

impl HuffmanCode {
    /// Create a new Huffman code
    pub const fn new(code: u32, bits: u8) -> Self {
        HuffmanCode { code, bits }
    }
}

/// DC coefficient Huffman table
pub struct DcHuffmanTable {
    /// Codes indexed by size category (0-12)
    codes: [HuffmanCode; 13],
}

impl Default for DcHuffmanTable {
    fn default() -> Self {
        Self::new()
    }
}

impl DcHuffmanTable {
    /// Create standard DC Huffman table
    pub fn new() -> Self {
        // DNxHD uses a simplified DC coding similar to JPEG
        // Size category codes (similar to JPEG DC table)
        let codes = [
            HuffmanCode::new(0b00, 2),          // 0
            HuffmanCode::new(0b010, 3),         // 1
            HuffmanCode::new(0b011, 3),         // 2
            HuffmanCode::new(0b100, 3),         // 3
            HuffmanCode::new(0b101, 3),         // 4
            HuffmanCode::new(0b110, 3),         // 5
            HuffmanCode::new(0b1110, 4),        // 6
            HuffmanCode::new(0b11110, 5),       // 7
            HuffmanCode::new(0b111110, 6),      // 8
            HuffmanCode::new(0b1111110, 7),     // 9
            HuffmanCode::new(0b11111110, 8),    // 10
            HuffmanCode::new(0b111111110, 9),   // 11
            HuffmanCode::new(0b1111111110, 10), // 12
        ];
        DcHuffmanTable { codes }
    }

    /// Get code for a DC difference value
    pub fn encode(&self, diff: i16) -> (HuffmanCode, i16, u8) {
        let (size, amplitude) = if diff == 0 {
            (0, 0)
        } else {
            let abs_diff = diff.unsigned_abs();
            let size = 16 - abs_diff.leading_zeros() as u8;
            let amplitude = if diff < 0 {
                diff + (1 << size) - 1
            } else {
                diff
            };
            (size, amplitude)
        };

        (self.codes[size as usize], amplitude, size)
    }

    /// Decode DC coefficient from bitstream
    pub fn decode(&self, reader: &mut BitReader) -> Result<i16> {
        // Read size category using prefix code
        let size = self.read_size(reader)?;

        if size == 0 {
            return Ok(0);
        }

        // Read amplitude
        let amplitude = reader.read_bits(size as usize)?;

        // Convert to signed value
        let half = 1 << (size - 1);
        let value = if amplitude < half {
            amplitude as i16 - ((1 << size) - 1)
        } else {
            amplitude as i16
        };

        Ok(value)
    }

    fn read_size(&self, reader: &mut BitReader) -> Result<u8> {
        // Read prefix to determine size
        if reader.read_bits(2)? == 0 {
            return Ok(0);
        }
        reader.unread_bits(2);

        let prefix = reader.read_bits(3)?;
        match prefix {
            0b010 => Ok(1),
            0b011 => Ok(2),
            0b100 => Ok(3),
            0b101 => Ok(4),
            0b110 => Ok(5),
            0b111 => {
                // Need more bits
                if reader.read_bit()? == 0 {
                    Ok(6)
                } else if reader.read_bit()? == 0 {
                    Ok(7)
                } else if reader.read_bit()? == 0 {
                    Ok(8)
                } else if reader.read_bit()? == 0 {
                    Ok(9)
                } else if reader.read_bit()? == 0 {
                    Ok(10)
                } else if reader.read_bit()? == 0 {
                    Ok(11)
                } else {
                    Ok(12)
                }
            }
            _ => {
                // Back up and try 2-bit code
                reader.unread_bits(1);
                Ok(0)
            }
        }
    }
}

/// AC coefficient run-level Huffman table
pub struct AcHuffmanTable {
    /// Run-level codes
    run_level_codes: Vec<(u8, u8, HuffmanCode)>,
    /// End of block code
    eob_code: HuffmanCode,
}

impl Default for AcHuffmanTable {
    fn default() -> Self {
        Self::new()
    }
}

impl AcHuffmanTable {
    /// Create standard AC Huffman table
    pub fn new() -> Self {
        // DNxHD AC coding uses run-level pairs
        // This is a simplified table - real DNxHD uses more complex tables
        let mut run_level_codes = Vec::new();

        // EOB: 0b10 (2 bits)
        let eob_code = HuffmanCode::new(0b10, 2);

        // Common run-level pairs
        // (run=0, level=1): 0b11 (2 bits)
        run_level_codes.push((0, 1, HuffmanCode::new(0b11, 2)));
        // (run=0, level=2): 0b0100 (4 bits)
        run_level_codes.push((0, 2, HuffmanCode::new(0b0100, 4)));
        // (run=1, level=1): 0b0101 (4 bits)
        run_level_codes.push((1, 1, HuffmanCode::new(0b0101, 4)));
        // (run=0, level=3): 0b00110 (5 bits)
        run_level_codes.push((0, 3, HuffmanCode::new(0b00110, 5)));
        // (run=2, level=1): 0b00111 (5 bits)
        run_level_codes.push((2, 1, HuffmanCode::new(0b00111, 5)));
        // (run=0, level=4): 0b001010 (6 bits)
        run_level_codes.push((0, 4, HuffmanCode::new(0b001010, 6)));
        // (run=3, level=1): 0b001011 (6 bits)
        run_level_codes.push((3, 1, HuffmanCode::new(0b001011, 6)));
        // (run=1, level=2): 0b001000 (6 bits)
        run_level_codes.push((1, 2, HuffmanCode::new(0b001000, 6)));

        AcHuffmanTable {
            run_level_codes,
            eob_code,
        }
    }

    /// Encode a run-level pair
    pub fn encode_run_level(&self, run: u8, level: i16) -> Option<(HuffmanCode, bool)> {
        let abs_level = level.unsigned_abs() as u8;
        let sign = level < 0;

        for &(r, l, code) in &self.run_level_codes {
            if r == run && l == abs_level {
                return Some((code, sign));
            }
        }

        None // Use escape code
    }

    /// Get end of block code
    pub fn eob(&self) -> HuffmanCode {
        self.eob_code
    }

    /// Decode run-level pair from bitstream
    pub fn decode_run_level(&self, reader: &mut BitReader) -> Result<Option<(u8, i16)>> {
        // Check for EOB
        if reader.peek_bits(2)? == 0b10 {
            reader.skip_bits(2)?;
            return Ok(None);
        }

        // Try to match codes
        for &(run, level, code) in &self.run_level_codes {
            if reader.peek_bits(code.bits as usize)? == code.code {
                reader.skip_bits(code.bits as usize)?;
                let sign = reader.read_bit()?;
                let signed_level = if sign == 1 {
                    -(level as i16)
                } else {
                    level as i16
                };
                return Ok(Some((run, signed_level)));
            }
        }

        // Escape code - read raw run and level
        Err(DnxError::HuffmanError("Unknown AC code".into()))
    }
}

/// Bitstream reader for Huffman decoding
pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
    /// Bits that were "unread"
    unread_buffer: u32,
    unread_count: u8,
}

impl<'a> BitReader<'a> {
    /// Create a new bit reader
    pub fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            byte_pos: 0,
            bit_pos: 0,
            unread_buffer: 0,
            unread_count: 0,
        }
    }

    /// Read a single bit
    pub fn read_bit(&mut self) -> Result<u32> {
        self.read_bits(1)
    }

    /// Read multiple bits
    pub fn read_bits(&mut self, count: usize) -> Result<u32> {
        if count == 0 {
            return Ok(0);
        }
        if count > 32 {
            return Err(DnxError::BitstreamError("Too many bits requested".into()));
        }

        // First use any unread bits
        if self.unread_count > 0 && count <= self.unread_count as usize {
            let shift = self.unread_count - count as u8;
            let result = (self.unread_buffer >> shift) & ((1 << count) - 1);
            self.unread_count -= count as u8;
            self.unread_buffer &= (1 << self.unread_count) - 1;
            return Ok(result);
        }

        let mut result = 0u32;
        let mut bits_remaining = count;

        // Use unread bits first
        if self.unread_count > 0 {
            result = self.unread_buffer;
            bits_remaining -= self.unread_count as usize;
            self.unread_count = 0;
            self.unread_buffer = 0;
        }

        while bits_remaining > 0 {
            if self.byte_pos >= self.data.len() {
                return Err(DnxError::InsufficientData {
                    needed: 1,
                    available: 0,
                });
            }

            let byte = self.data[self.byte_pos];
            let bits_in_byte = 8 - self.bit_pos as usize;

            if bits_remaining >= bits_in_byte {
                // Take remaining bits from this byte
                let mask = (1 << bits_in_byte) - 1;
                result = (result << bits_in_byte) | (byte & mask) as u32;
                bits_remaining -= bits_in_byte;
                self.byte_pos += 1;
                self.bit_pos = 0;
            } else {
                // Take some bits from this byte
                let shift = bits_in_byte - bits_remaining;
                let mask = (1 << bits_remaining) - 1;
                result = (result << bits_remaining) | ((byte >> shift) & mask) as u32;
                self.bit_pos += bits_remaining as u8;
                bits_remaining = 0;
            }
        }

        Ok(result)
    }

    /// Peek at bits without advancing
    pub fn peek_bits(&self, count: usize) -> Result<u32> {
        let mut copy = BitReader {
            data: self.data,
            byte_pos: self.byte_pos,
            bit_pos: self.bit_pos,
            unread_buffer: self.unread_buffer,
            unread_count: self.unread_count,
        };
        copy.read_bits(count)
    }

    /// Skip bits
    pub fn skip_bits(&mut self, count: usize) -> Result<()> {
        self.read_bits(count)?;
        Ok(())
    }

    /// Unread bits (push back for re-reading)
    pub fn unread_bits(&mut self, count: usize) {
        // This is a simplified implementation
        // In practice, we'd need to track what was read
        self.unread_count += count as u8;
    }

    /// Get current byte position
    pub fn position(&self) -> usize {
        self.byte_pos
    }

    /// Check if at end of data
    pub fn is_eof(&self) -> bool {
        self.byte_pos >= self.data.len()
    }

    /// Align to next byte boundary
    pub fn align_byte(&mut self) {
        if self.bit_pos > 0 {
            self.byte_pos += 1;
            self.bit_pos = 0;
        }
    }
}

/// Bitstream writer for Huffman encoding
pub struct BitWriter {
    data: Vec<u8>,
    current_byte: u8,
    bit_count: u8,
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter {
    /// Create a new bit writer
    pub fn new() -> Self {
        BitWriter {
            data: Vec::new(),
            current_byte: 0,
            bit_count: 0,
        }
    }

    /// Write a single bit
    pub fn write_bit(&mut self, bit: bool) {
        self.current_byte = (self.current_byte << 1) | (bit as u8);
        self.bit_count += 1;

        if self.bit_count == 8 {
            self.data.push(self.current_byte);
            self.current_byte = 0;
            self.bit_count = 0;
        }
    }

    /// Write multiple bits
    pub fn write_bits(&mut self, value: u32, count: u8) {
        for i in (0..count).rev() {
            let bit = ((value >> i) & 1) != 0;
            self.write_bit(bit);
        }
    }

    /// Write a Huffman code
    pub fn write_code(&mut self, code: &HuffmanCode) {
        self.write_bits(code.code, code.bits);
    }

    /// Flush remaining bits (pad with zeros)
    pub fn flush(&mut self) {
        if self.bit_count > 0 {
            self.current_byte <<= 8 - self.bit_count;
            self.data.push(self.current_byte);
            self.current_byte = 0;
            self.bit_count = 0;
        }
    }

    /// Get the written data
    pub fn into_bytes(mut self) -> Vec<u8> {
        self.flush();
        self.data
    }

    /// Get current length in bytes (including partial)
    pub fn len(&self) -> usize {
        self.data.len() + if self.bit_count > 0 { 1 } else { 0 }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty() && self.bit_count == 0
    }

    /// Get current bit position
    pub fn bit_position(&self) -> usize {
        self.data.len() * 8 + self.bit_count as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_writer_read() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b10110, 5);
        writer.write_bits(0b001, 3);
        let data = writer.into_bytes();

        assert_eq!(data, vec![0b10110001]);
    }

    #[test]
    fn test_bit_reader() {
        let data = vec![0b10110001, 0b11110000];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.read_bits(5).unwrap(), 0b10110);
        assert_eq!(reader.read_bits(3).unwrap(), 0b001);
        assert_eq!(reader.read_bits(4).unwrap(), 0b1111);
    }

    #[test]
    fn test_dc_huffman_encode_decode() {
        let table = DcHuffmanTable::new();

        // Test encoding
        let (code, amp, size) = table.encode(0);
        assert_eq!(size, 0);
        assert_eq!(code.bits, 2);

        let (code, _amp, size) = table.encode(5);
        assert_eq!(size, 3);
        assert_eq!(code.bits, 3);
    }

    #[test]
    fn test_ac_huffman_eob() {
        let table = AcHuffmanTable::new();
        let eob = table.eob();
        assert_eq!(eob.code, 0b10);
        assert_eq!(eob.bits, 2);
    }

    #[test]
    fn test_bit_writer_code() {
        let mut writer = BitWriter::new();
        let code = HuffmanCode::new(0b1101, 4);
        writer.write_code(&code);
        writer.flush();

        let data = writer.into_bytes();
        assert_eq!(data[0] >> 4, 0b1101);
    }
}
