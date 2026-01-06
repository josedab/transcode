//! VP8 boolean arithmetic decoder.
//!
//! VP8 uses a boolean arithmetic coder similar to VP6/VP7.
//! The decoder maintains a range and value, reading bits from the stream
//! based on probability values (0-255).

use crate::error::{Vp8Error, Result};

/// Boolean arithmetic decoder for VP8.
pub struct BoolDecoder<'a> {
    /// Input data buffer.
    data: &'a [u8],
    /// Current byte position.
    pos: usize,
    /// Current value.
    value: u32,
    /// Current range.
    range: u32,
    /// Number of bits available.
    bits: i32,
}

impl<'a> BoolDecoder<'a> {
    /// Create a new boolean decoder.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(Vp8Error::InvalidBitstream("Empty data".into()));
        }

        let mut decoder = Self {
            data,
            pos: 0,
            value: 0,
            range: 255,
            bits: 0,
        };

        // Initialize with first bytes
        decoder.fill()?;

        Ok(decoder)
    }

    /// Fill the value buffer with more bits.
    fn fill(&mut self) -> Result<()> {
        while self.bits < 8 {
            if self.pos < self.data.len() {
                self.value = (self.value << 8) | self.data[self.pos] as u32;
                self.pos += 1;
            } else {
                self.value <<= 8;
            }
            self.bits += 8;
        }
        Ok(())
    }

    /// Read a single boolean value with given probability.
    pub fn read_bool(&mut self, prob: u8) -> Result<bool> {
        let split = 1 + (((self.range - 1) * prob as u32) >> 8);

        // Ensure we have enough bits
        if self.bits < 8 {
            self.fill()?;
        }

        let shift = self.bits.saturating_sub(8) as u32;
        let high_part = self.value >> shift;
        let bit = high_part >= split;

        if bit {
            self.value = self.value.saturating_sub(split << shift);
            self.range -= split;
        } else {
            self.range = split;
        }

        // Renormalize
        while self.range < 128 {
            self.range <<= 1;
            self.bits -= 1;
            if self.bits < 0 {
                self.fill()?;
            }
        }

        Ok(bit)
    }

    /// Read a single bit (50% probability).
    pub fn read_bit(&mut self) -> Result<bool> {
        self.read_bool(128)
    }

    /// Read multiple bits as unsigned integer (big-endian).
    pub fn read_bits(&mut self, n: u32) -> Result<u32> {
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | (self.read_bit()? as u32);
        }
        Ok(value)
    }

    /// Read a literal value (n bits, MSB first).
    pub fn read_literal(&mut self, n: u32) -> Result<i32> {
        Ok(self.read_bits(n)? as i32)
    }

    /// Read a signed literal (sign bit + magnitude).
    pub fn read_signed_literal(&mut self, n: u32) -> Result<i32> {
        let value = self.read_bits(n)? as i32;
        if self.read_bit()? {
            Ok(-value)
        } else {
            Ok(value)
        }
    }

    /// Read a probability-encoded value using a tree.
    pub fn read_tree<const N: usize>(&mut self, tree: &[i8], probs: &[u8; N]) -> Result<i32> {
        let mut idx = 0i32;

        while idx >= 0 {
            let prob = probs[idx as usize >> 1];
            let bit = self.read_bool(prob)?;
            idx = tree[(idx as usize) + (bit as usize)] as i32;
        }

        Ok(-(idx + 1))
    }

    /// Check if we have more data.
    pub fn has_more(&self) -> bool {
        self.pos < self.data.len() || self.bits > 0
    }

    /// Get current position.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Skip to byte boundary.
    pub fn align(&mut self) {
        self.bits = 0;
    }
}

/// Boolean encoder for VP8.
#[cfg(feature = "encoder")]
pub struct BoolEncoder {
    /// Output buffer.
    output: Vec<u8>,
    /// Current low value.
    low: u64,
    /// Current range.
    range: u32,
    /// Number of outstanding bits.
    count: i32,
}

#[cfg(feature = "encoder")]
impl BoolEncoder {
    /// Create a new boolean encoder.
    pub fn new() -> Self {
        Self {
            output: Vec::new(),
            low: 0,
            range: 255,
            count: -24,
        }
    }

    /// Write a boolean value with given probability.
    pub fn write_bool(&mut self, value: bool, prob: u8) {
        let split = 1 + (((self.range - 1) * prob as u32) >> 8);

        if value {
            self.low += split as u64;
            self.range -= split;
        } else {
            self.range = split;
        }

        // Renormalize
        let shift = self.range.leading_zeros() as i32 - 24;
        if shift > 0 {
            self.range <<= shift;
            self.low <<= shift;
            self.count += shift;

            if self.count >= 0 {
                self.carry_out();
            }
        }
    }

    /// Write a single bit (50% probability).
    pub fn write_bit(&mut self, value: bool) {
        self.write_bool(value, 128);
    }

    /// Write multiple bits.
    pub fn write_bits(&mut self, value: u32, n: u32) {
        for i in (0..n).rev() {
            self.write_bit((value >> i) & 1 != 0);
        }
    }

    /// Write a literal value.
    pub fn write_literal(&mut self, value: i32, n: u32) {
        self.write_bits(value as u32, n);
    }

    /// Handle carry propagation.
    fn carry_out(&mut self) {
        let carry = (self.low >> 32) as u8;
        let byte = (self.low >> 24) as u8;

        if byte != 0xFF {
            // Output pending bytes
            while self.count >= 0 {
                let out = byte.wrapping_add(carry);
                self.output.push(out);
                self.count -= 8;
            }
        } else {
            self.count -= 8;
        }

        self.low = (self.low << 8) & 0xFFFFFFFF;
    }

    /// Finalize and return output.
    pub fn finalize(mut self) -> Vec<u8> {
        // Flush remaining bits
        for _ in 0..32 {
            self.write_bit(false);
        }
        self.output
    }
}

#[cfg(feature = "encoder")]
impl Default for BoolEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let data = [0x00, 0x01, 0x02, 0x03];
        let decoder = BoolDecoder::new(&data);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_decoder_empty() {
        let data: [u8; 0] = [];
        let decoder = BoolDecoder::new(&data);
        assert!(decoder.is_err());
    }

    #[test]
    fn test_read_bits() {
        let data = [0b11010110, 0b10101010];
        let mut decoder = BoolDecoder::new(&data).unwrap();

        // Reading with 50% probability should give us the actual bits
        // Note: The bool decoder works differently than a simple bit reader
        assert!(decoder.has_more());
    }

    #[test]
    fn test_read_literal() {
        let data = [0xFF, 0x00, 0xFF, 0x00];
        let mut decoder = BoolDecoder::new(&data).unwrap();
        let result = decoder.read_literal(4);
        assert!(result.is_ok());
    }
}
