//! Bit-level reader for VP8/VP8L decoding

#![allow(dead_code)]

use crate::error::{WebPError, Result};

/// A bit-level reader that reads bits from a byte stream
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_pos: u32,
    // For VP8 boolean decoder
    range: u32,
    value: u32,
    bits_left: i32,
}

impl<'a> BitReader<'a> {
    /// Create a new bit reader
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bit_pos: 0,
            range: 255,
            value: 0,
            bits_left: 0,
        }
    }

    /// Get the current byte position
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Get remaining bytes
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    /// Read a single bit (LSB first, for VP8L)
    pub fn read_bit(&mut self) -> Result<u32> {
        if self.pos >= self.data.len() {
            return Err(WebPError::UnexpectedEof);
        }

        let bit = (self.data[self.pos] >> self.bit_pos) & 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.pos += 1;
        }

        Ok(u32::from(bit))
    }

    /// Read n bits (LSB first, for VP8L)
    pub fn read_bits(&mut self, n: u32) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(WebPError::InvalidVp8l("Too many bits requested".into()));
        }

        let mut value = 0u32;
        for i in 0..n {
            let bit = self.read_bit()?;
            value |= bit << i;
        }

        Ok(value)
    }

    /// Read a signed value using n bits
    pub fn read_signed_bits(&mut self, n: u32) -> Result<i32> {
        let value = self.read_bits(n)? as i32;
        let sign = self.read_bit()?;
        if sign != 0 {
            Ok(-value)
        } else {
            Ok(value)
        }
    }

    /// Initialize the VP8 boolean decoder
    #[allow(clippy::precedence)]
    pub fn init_bool_decoder(&mut self) -> Result<()> {
        if self.data.len() < 2 {
            return Err(WebPError::UnexpectedEof);
        }

        self.value = u32::from(self.data[self.pos]) << 8 | u32::from(self.data[self.pos + 1]);
        self.pos += 2;
        self.range = 255;
        self.bits_left = 16;

        Ok(())
    }

    /// Read a boolean using the VP8 arithmetic decoder
    pub fn read_bool(&mut self, prob: u8) -> Result<bool> {
        let split = 1 + (((self.range - 1) * u32::from(prob)) >> 8);

        if self.value < (split << 8) {
            self.range = split;
            self.normalize()?;
            Ok(false)
        } else {
            self.value -= split << 8;
            self.range -= split;
            self.normalize()?;
            Ok(true)
        }
    }

    /// Read a literal value using the VP8 arithmetic decoder
    pub fn read_literal(&mut self, n: usize) -> Result<u32> {
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | if self.read_bool(128)? { 1 } else { 0 };
        }
        Ok(value)
    }

    /// Normalize the boolean decoder state
    fn normalize(&mut self) -> Result<()> {
        while self.range < 128 {
            self.range <<= 1;
            self.value <<= 1;
            self.bits_left -= 1;

            if self.bits_left <= 0 {
                if self.pos >= self.data.len() {
                    // Pad with zeros at end of stream
                    self.bits_left = 8;
                } else {
                    self.value |= u32::from(self.data[self.pos]);
                    self.pos += 1;
                    self.bits_left = 8;
                }
            }
        }
        Ok(())
    }

    /// Skip to the next byte boundary
    pub fn align_to_byte(&mut self) {
        if self.bit_pos > 0 {
            self.bit_pos = 0;
            self.pos += 1;
        }
    }

    /// Read a raw byte
    pub fn read_byte(&mut self) -> Result<u8> {
        if self.pos >= self.data.len() {
            return Err(WebPError::UnexpectedEof);
        }
        let byte = self.data[self.pos];
        self.pos += 1;
        Ok(byte)
    }

    /// Read raw bytes
    pub fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            return Err(WebPError::UnexpectedEof);
        }
        let bytes = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(bytes)
    }

    /// Peek at the next byte without consuming it
    pub fn peek_byte(&self) -> Option<u8> {
        self.data.get(self.pos).copied()
    }

    /// Check if we've reached the end
    pub fn is_empty(&self) -> bool {
        self.pos >= self.data.len() && self.bit_pos == 0
    }
}

/// Bit reader that reads in MSB-first order (used for some VP8L operations)
#[allow(dead_code)]
pub struct MsbBitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_buffer: u64,
    bits_in_buffer: u32,
}

impl<'a> MsbBitReader<'a> {
    /// Create a new MSB-first bit reader
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bit_buffer: 0,
            bits_in_buffer: 0,
        }
    }

    /// Fill the bit buffer
    fn fill_buffer(&mut self) {
        while self.bits_in_buffer <= 56 && self.pos < self.data.len() {
            self.bit_buffer |= u64::from(self.data[self.pos]) << (56 - self.bits_in_buffer);
            self.bits_in_buffer += 8;
            self.pos += 1;
        }
    }

    /// Read n bits (MSB first)
    pub fn read_bits(&mut self, n: u32) -> Result<u32> {
        if n > 32 {
            return Err(WebPError::InvalidVp8l("Too many bits requested".into()));
        }

        self.fill_buffer();

        if self.bits_in_buffer < n {
            return Err(WebPError::UnexpectedEof);
        }

        let value = (self.bit_buffer >> (64 - n)) as u32;
        self.bit_buffer <<= n;
        self.bits_in_buffer -= n;

        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_bits() {
        let data = [0b10110100, 0b11001010];
        let mut reader = BitReader::new(&data);

        // Read LSB first
        assert_eq!(reader.read_bits(4).unwrap(), 0b0100);
        assert_eq!(reader.read_bits(4).unwrap(), 0b1011);
        assert_eq!(reader.read_bits(4).unwrap(), 0b1010);
        assert_eq!(reader.read_bits(4).unwrap(), 0b1100);
    }

    #[test]
    fn test_read_single_bits() {
        let data = [0b10110100];
        let mut reader = BitReader::new(&data);

        // LSB first: 0, 0, 1, 0, 1, 1, 0, 1
        assert_eq!(reader.read_bit().unwrap(), 0);
        assert_eq!(reader.read_bit().unwrap(), 0);
        assert_eq!(reader.read_bit().unwrap(), 1);
        assert_eq!(reader.read_bit().unwrap(), 0);
        assert_eq!(reader.read_bit().unwrap(), 1);
        assert_eq!(reader.read_bit().unwrap(), 1);
        assert_eq!(reader.read_bit().unwrap(), 0);
        assert_eq!(reader.read_bit().unwrap(), 1);
    }

    #[test]
    fn test_eof() {
        let data = [0x42];
        let mut reader = BitReader::new(&data);
        reader.read_bits(8).unwrap();
        assert!(reader.read_bit().is_err());
    }
}
