//! Bitstream reading and writing utilities.
//!
//! This module provides efficient bit-level access to byte streams, essential for
//! parsing coded video and audio bitstreams.

use crate::error::{BitstreamError, Result};

/// A bitstream reader for parsing coded data.
///
/// Supports reading individual bits, multi-bit values, and Exp-Golomb coded values
/// commonly used in video codecs like H.264 and H.265.
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    /// Create a new bit reader from a byte slice.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Get the total number of bits in the stream.
    pub fn total_bits(&self) -> usize {
        self.data.len() * 8
    }

    /// Get the current bit position in the stream.
    pub fn position(&self) -> usize {
        self.byte_pos * 8 + self.bit_pos as usize
    }

    /// Get the number of remaining bits.
    pub fn remaining_bits(&self) -> usize {
        self.total_bits().saturating_sub(self.position())
    }

    /// Check if we've reached the end of the stream.
    pub fn is_eof(&self) -> bool {
        self.byte_pos >= self.data.len()
    }

    /// Check if the stream is byte-aligned.
    pub fn is_byte_aligned(&self) -> bool {
        self.bit_pos == 0
    }

    /// Skip to the next byte boundary.
    pub fn align_to_byte(&mut self) {
        if self.bit_pos != 0 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
    }

    /// Read a single bit.
    pub fn read_bit(&mut self) -> Result<bool> {
        if self.byte_pos >= self.data.len() {
            return Err(BitstreamError::UnexpectedEnd.into());
        }

        let bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }

        Ok(bit != 0)
    }

    /// Read up to 32 bits as an unsigned integer.
    pub fn read_bits(&mut self, n: u8) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(crate::error::Error::InvalidParameter(
                "Cannot read more than 32 bits at once".into(),
            ));
        }
        if self.remaining_bits() < n as usize {
            return Err(BitstreamError::UnexpectedEnd.into());
        }

        let mut value: u32 = 0;
        for _ in 0..n {
            value = (value << 1) | (self.read_bit()? as u32);
        }

        Ok(value)
    }

    /// Read up to 64 bits as an unsigned integer.
    pub fn read_bits_u64(&mut self, n: u8) -> Result<u64> {
        if n == 0 {
            return Ok(0);
        }
        if n > 64 {
            return Err(crate::error::Error::InvalidParameter(
                "Cannot read more than 64 bits at once".into(),
            ));
        }
        if self.remaining_bits() < n as usize {
            return Err(BitstreamError::UnexpectedEnd.into());
        }

        let mut value: u64 = 0;
        for _ in 0..n {
            value = (value << 1) | (self.read_bit()? as u64);
        }

        Ok(value)
    }

    /// Read an unsigned Exp-Golomb coded value (ue(v)).
    ///
    /// Used extensively in H.264/H.265 for compact integer representation.
    pub fn read_ue(&mut self) -> Result<u32> {
        let mut leading_zeros = 0u8;
        while !self.read_bit()? {
            leading_zeros += 1;
            if leading_zeros > 31 {
                return Err(BitstreamError::ExpGolombOverflow.into());
            }
        }

        if leading_zeros == 0 {
            return Ok(0);
        }

        let suffix = self.read_bits(leading_zeros)?;
        Ok((1u32 << leading_zeros) - 1 + suffix)
    }

    /// Read a signed Exp-Golomb coded value (se(v)).
    pub fn read_se(&mut self) -> Result<i32> {
        let ue = self.read_ue()?;
        let value = ue.div_ceil(2) as i32;
        if ue % 2 == 0 {
            Ok(-value)
        } else {
            Ok(value)
        }
    }

    /// Read a byte-aligned unsigned 8-bit value.
    pub fn read_u8(&mut self) -> Result<u8> {
        self.read_bits(8).map(|v| v as u8)
    }

    /// Read a byte-aligned unsigned 16-bit value (big-endian).
    pub fn read_u16(&mut self) -> Result<u16> {
        self.read_bits(16).map(|v| v as u16)
    }

    /// Read a byte-aligned unsigned 32-bit value (big-endian).
    pub fn read_u32(&mut self) -> Result<u32> {
        self.read_bits(32)
    }

    /// Skip a number of bits.
    pub fn skip(&mut self, n: usize) -> Result<()> {
        if self.remaining_bits() < n {
            return Err(BitstreamError::UnexpectedEnd.into());
        }

        let new_pos = self.position() + n;
        self.byte_pos = new_pos / 8;
        self.bit_pos = (new_pos % 8) as u8;

        Ok(())
    }

    /// Peek at the next bit without consuming it.
    pub fn peek_bit(&self) -> Result<bool> {
        if self.byte_pos >= self.data.len() {
            return Err(BitstreamError::UnexpectedEnd.into());
        }

        let bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1;
        Ok(bit != 0)
    }

    /// Peek at the next n bits without consuming them.
    pub fn peek_bits(&self, n: u8) -> Result<u32> {
        let mut clone = self.clone();
        clone.read_bits(n)
    }

    /// Get the underlying byte slice from the current position.
    pub fn remaining_bytes(&self) -> &'a [u8] {
        if self.bit_pos == 0 {
            &self.data[self.byte_pos..]
        } else {
            &self.data[(self.byte_pos + 1).min(self.data.len())..]
        }
    }

    /// Check for more RBSP data (used in H.264/H.265).
    pub fn more_rbsp_data(&self) -> bool {
        if self.byte_pos >= self.data.len() {
            return false;
        }

        // Check if we're at the last byte
        if self.byte_pos == self.data.len() - 1 {
            // Check for rbsp_trailing_bits
            let remaining = 8 - self.bit_pos;
            let mask = (1u8 << remaining) - 1;
            let bits = self.data[self.byte_pos] & mask;
            // Should be 1 followed by zeros for trailing bits
            bits != (1 << (remaining - 1))
        } else {
            true
        }
    }
}

/// A bitstream writer for generating coded data.
#[derive(Debug, Clone)]
pub struct BitWriter {
    data: Vec<u8>,
    bit_pos: u8,
}

impl BitWriter {
    /// Create a new bit writer.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            bit_pos: 0,
        }
    }

    /// Create a new bit writer with capacity.
    pub fn with_capacity(bytes: usize) -> Self {
        Self {
            data: Vec::with_capacity(bytes),
            bit_pos: 0,
        }
    }

    /// Get the current bit position.
    pub fn position(&self) -> usize {
        self.data.len() * 8 - (8 - self.bit_pos as usize) % 8
    }

    /// Check if the writer is byte-aligned.
    pub fn is_byte_aligned(&self) -> bool {
        self.bit_pos == 0
    }

    /// Write a single bit.
    pub fn write_bit(&mut self, bit: bool) -> Result<()> {
        if self.bit_pos == 0 {
            self.data.push(0);
        }

        if bit {
            let idx = self.data.len() - 1;
            self.data[idx] |= 1 << (7 - self.bit_pos);
        }

        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
        }
        Ok(())
    }

    /// Write up to 32 bits from an unsigned integer.
    pub fn write_bits(&mut self, value: u32, n: u8) -> Result<()> {
        for i in (0..n).rev() {
            self.write_bit((value >> i) & 1 != 0)?;
        }
        Ok(())
    }

    /// Write up to 64 bits from an unsigned integer.
    pub fn write_bits_u64(&mut self, value: u64, n: u8) -> Result<()> {
        for i in (0..n).rev() {
            self.write_bit((value >> i) & 1 != 0)?;
        }
        Ok(())
    }

    /// Write an unsigned Exp-Golomb coded value.
    pub fn write_ue(&mut self, value: u32) -> Result<()> {
        if value == 0 {
            self.write_bit(true)?;
            return Ok(());
        }

        let value_plus_1 = value + 1;
        let leading_zeros = 31 - value_plus_1.leading_zeros();

        // Write leading zeros
        for _ in 0..leading_zeros {
            self.write_bit(false)?;
        }

        // Write the value + 1
        self.write_bits(value_plus_1, leading_zeros as u8 + 1)?;
        Ok(())
    }

    /// Write a signed Exp-Golomb coded value.
    pub fn write_se(&mut self, value: i32) -> Result<()> {
        let ue = if value <= 0 {
            (-2 * value) as u32
        } else {
            (2 * value - 1) as u32
        };
        self.write_ue(ue)
    }

    /// Align to byte boundary by writing zero bits.
    pub fn align_to_byte(&mut self) -> Result<()> {
        while self.bit_pos != 0 {
            self.write_bit(false)?;
        }
        Ok(())
    }

    /// Alias for align_to_byte.
    pub fn byte_align(&mut self) -> Result<()> {
        self.align_to_byte()
    }

    /// Alias for write_se.
    pub fn write_signed_exp_golomb(&mut self, value: i32) -> Result<()> {
        self.write_se(value)
    }

    /// Write RBSP trailing bits (1 followed by zeros to byte alignment).
    pub fn write_rbsp_trailing_bits(&mut self) -> Result<()> {
        self.write_bit(true)?;
        self.align_to_byte()
    }

    /// Get the written data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Take the written data, consuming the writer.
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Find the next start code in a byte slice.
///
/// Start codes are 0x000001 or 0x00000001 sequences used in H.264/H.265.
pub fn find_start_code(data: &[u8]) -> Option<(usize, usize)> {
    let len = data.len();
    if len < 3 {
        return None;
    }

    for i in 0..len - 2 {
        if data[i] == 0 && data[i + 1] == 0 {
            if data[i + 2] == 1 {
                return Some((i, 3));
            } else if i + 3 < len && data[i + 2] == 0 && data[i + 3] == 1 {
                return Some((i, 4));
            }
        }
    }

    None
}

/// Remove emulation prevention bytes (0x03) from RBSP data.
///
/// In H.264/H.265, 0x000003 sequences are used to prevent start code emulation.
pub fn remove_emulation_prevention(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(data.len());
    let mut i = 0;

    while i < data.len() {
        if i + 2 < data.len() && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 3 {
            result.push(0);
            result.push(0);
            i += 3; // Skip the 0x03 byte
        } else {
            result.push(data[i]);
            i += 1;
        }
    }

    result
}

/// Add emulation prevention bytes to RBSP data.
pub fn add_emulation_prevention(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(data.len() + data.len() / 100);
    let mut zeros = 0;

    for &byte in data {
        if zeros == 2 && byte <= 3 {
            result.push(3); // Add emulation prevention byte
            zeros = 0;
        }

        result.push(byte);

        if byte == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_bits() {
        let data = [0b10110100, 0b11001010];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.read_bits(4).unwrap(), 0b1011);
        assert_eq!(reader.read_bits(4).unwrap(), 0b0100);
        assert_eq!(reader.read_bits(8).unwrap(), 0b11001010);
    }

    #[test]
    fn test_read_single_bits() {
        let data = [0b10110100];
        let mut reader = BitReader::new(&data);

        assert!(reader.read_bit().unwrap());
        assert!(!reader.read_bit().unwrap());
        assert!(reader.read_bit().unwrap());
        assert!(reader.read_bit().unwrap());
    }

    #[test]
    fn test_exp_golomb() {
        // 0 -> 1 (1 bit: "1")
        let data = [0b10000000];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_ue().unwrap(), 0);

        // 1 -> 010 (3 bits)
        let data = [0b01000000];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_ue().unwrap(), 1);

        // 2 -> 011 (3 bits)
        let data = [0b01100000];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_ue().unwrap(), 2);

        // 3 -> 00100 (5 bits)
        let data = [0b00100000];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_ue().unwrap(), 3);
    }

    #[test]
    fn test_signed_exp_golomb() {
        // 0 -> 1
        let data = [0b10000000];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_se().unwrap(), 0);

        // 1 -> 010 -> +1
        let data = [0b01000000];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_se().unwrap(), 1);

        // 2 -> 011 -> -1
        let data = [0b01100000];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_se().unwrap(), -1);
    }

    #[test]
    fn test_write_bits() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b1011, 4);
        writer.write_bits(0b0100, 4);
        assert_eq!(writer.data(), &[0b10110100]);
    }

    #[test]
    fn test_write_exp_golomb() {
        let mut writer = BitWriter::new();
        writer.write_ue(0);
        writer.align_to_byte();
        assert_eq!(writer.data(), &[0b10000000]);

        let mut writer = BitWriter::new();
        writer.write_ue(1);
        writer.align_to_byte();
        assert_eq!(writer.data(), &[0b01000000]);
    }

    #[test]
    fn test_find_start_code() {
        let data = [0x00, 0x00, 0x01, 0x65];
        assert_eq!(find_start_code(&data), Some((0, 3)));

        let data = [0x00, 0x00, 0x00, 0x01, 0x65];
        assert_eq!(find_start_code(&data), Some((0, 4)));

        let data = [0xFF, 0x00, 0x00, 0x01, 0x65];
        assert_eq!(find_start_code(&data), Some((1, 3)));
    }

    #[test]
    fn test_emulation_prevention() {
        let data = [0x00, 0x00, 0x03, 0x01];
        let clean = remove_emulation_prevention(&data);
        assert_eq!(clean, vec![0x00, 0x00, 0x01]);

        let data = [0x00, 0x00, 0x01];
        let escaped = add_emulation_prevention(&data);
        assert_eq!(escaped, vec![0x00, 0x00, 0x03, 0x01]);
    }
}
