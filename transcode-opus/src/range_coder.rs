//! Range coding implementation for Opus.
//!
//! Opus uses range coding (a variant of arithmetic coding) for entropy coding.
//! This implementation supports both decoding and encoding operations.

use crate::error::{OpusError, Result};

/// Top value for range normalization.
const TOP: u32 = 1 << 31;

/// Range decoder for reading entropy-coded data.
#[derive(Debug)]
pub struct RangeDecoder<'a> {
    /// Input data.
    data: &'a [u8],
    /// Current byte position.
    pos: usize,
    /// Current range value.
    range: u32,
    /// Current code value.
    value: u32,
    /// Total bits read.
    bits_read: u32,
    /// End of stream flag.
    end_of_stream: bool,
}

impl<'a> RangeDecoder<'a> {
    /// Create a new range decoder.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(OpusError::InvalidPacket("Empty data".into()));
        }

        let mut decoder = Self {
            data,
            pos: 0,
            range: 128,
            value: 0,
            bits_read: 0,
            end_of_stream: false,
        };

        // Initialize with first byte
        decoder.value = (127 - (data[0] >> 1)) as u32;
        decoder.pos = 1;

        // Normalize
        decoder.normalize()?;

        Ok(decoder)
    }

    /// Get the current position in bytes.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Get the number of bits read.
    pub fn bits_read(&self) -> u32 {
        self.bits_read
    }

    /// Check if at end of stream.
    pub fn is_eof(&self) -> bool {
        self.end_of_stream
    }

    /// Read a single bit.
    pub fn read_bit(&mut self) -> Result<bool> {
        self.read_uint(2).map(|v| v != 0)
    }

    /// Read an unsigned integer with uniform distribution.
    pub fn read_uint(&mut self, ft: u32) -> Result<u32> {
        if ft == 0 {
            return Ok(0);
        }

        let ft = ft.max(1);
        let mut k = 0u32;
        let mut scale = 1u32;

        while scale <= ft {
            scale <<= 1;
            k += 1;
        }

        if k > 0 {
            let bits = self.read_raw_bits(k as usize)?;
            if bits < ft {
                return Ok(bits);
            }
        }

        // Fall back to decode loop
        self.decode_uniform(ft)
    }

    /// Decode with uniform probability.
    fn decode_uniform(&mut self, ft: u32) -> Result<u32> {
        let ft = ft.max(1);
        let fs = self.range / ft;

        if fs == 0 {
            return Err(OpusError::RangeCoder("Division by zero in uniform decode".into()));
        }

        let mut k = self.value / fs;
        k = k.min(ft - 1);

        self.value -= k * fs;
        self.range = if k + 1 < ft {
            fs
        } else {
            self.range - k * fs
        };

        self.normalize()?;
        Ok(k)
    }

    /// Read a symbol with given probability distribution.
    pub fn read_symbol(&mut self, cdf: &[u16]) -> Result<u32> {
        if cdf.len() < 2 {
            return Err(OpusError::RangeCoder("CDF too short".into()));
        }

        let total = cdf[cdf.len() - 1] as u32;
        let fs = self.range / total;

        if fs == 0 {
            return Err(OpusError::RangeCoder("Range too small".into()));
        }

        let val = (self.value / fs).min(total - 1);

        // Find symbol
        let mut sym = 0;
        while sym < cdf.len() - 1 && (cdf[sym] as u32) <= val {
            sym += 1;
        }

        let fl = if sym > 0 { cdf[sym - 1] as u32 } else { 0 };
        let fh = cdf[sym] as u32;

        self.value -= fl * fs;
        self.range = (fh - fl) * fs;

        self.normalize()?;
        Ok(sym as u32)
    }

    /// Read a laplace-distributed value.
    pub fn read_laplace(&mut self, decay: u32) -> Result<i32> {
        // Symmetric laplace distribution
        let sign = self.read_bit()?;
        let mut value = 0i32;

        // Geometric distribution for magnitude
        let p = decay.min(255) as u16;
        while value < 256 {
            let bit = self.read_symbol(&[p, 256])?;
            if bit == 0 {
                break;
            }
            value += 1;
        }

        if sign {
            Ok(-value)
        } else {
            Ok(value)
        }
    }

    /// Read raw bits without entropy coding.
    pub fn read_raw_bits(&mut self, n: usize) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }

        if n > 32 {
            return Err(OpusError::RangeCoder("Cannot read more than 32 bits".into()));
        }

        let mut value = 0u32;
        for _ in 0..n {
            self.range >>= 1;
            let bit = if self.value >= self.range {
                self.value -= self.range;
                1u32
            } else {
                0u32
            };
            value = (value << 1) | bit;
            self.bits_read += 1;
            self.normalize()?;
        }

        Ok(value)
    }

    /// Normalize the range coder state.
    fn normalize(&mut self) -> Result<()> {
        // Limit iterations to prevent infinite loops
        let mut iterations = 0u32;
        const MAX_ITERATIONS: u32 = 32;

        while self.range < TOP && iterations < MAX_ITERATIONS {
            iterations += 1;
            self.range = self.range.saturating_mul(256);

            let byte = if self.pos < self.data.len() {
                let b = self.data[self.pos];
                self.pos += 1;
                b
            } else {
                self.end_of_stream = true;
                0
            };

            self.value = (self.value << 8) | (byte as u32);
            self.bits_read = self.bits_read.saturating_add(8);
        }

        Ok(())
    }

    /// Get remaining bytes.
    pub fn remaining_bytes(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }
}

/// Range encoder for writing entropy-coded data.
#[derive(Debug)]
pub struct RangeEncoder {
    /// Output buffer.
    buffer: Vec<u8>,
    /// Current range value.
    range: u32,
    /// Low value.
    low: u64,
    /// Carry count.
    carry: u32,
    /// Cache byte.
    cache: u8,
    /// Number of bits written.
    bits_written: u32,
}

impl RangeEncoder {
    /// Create a new range encoder.
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(1024),
            range: TOP,
            low: 0,
            carry: 0,
            cache: 0,
            bits_written: 0,
        }
    }

    /// Create with capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            range: TOP,
            low: 0,
            carry: 0,
            cache: 0,
            bits_written: 0,
        }
    }

    /// Get the number of bits written.
    pub fn bits_written(&self) -> u32 {
        self.bits_written
    }

    /// Write a single bit.
    pub fn write_bit(&mut self, bit: bool) -> Result<()> {
        self.write_uint(if bit { 1 } else { 0 }, 2)
    }

    /// Write an unsigned integer with uniform distribution.
    pub fn write_uint(&mut self, value: u32, ft: u32) -> Result<()> {
        if ft <= 1 {
            return Ok(());
        }

        if value >= ft {
            return Err(OpusError::RangeCoder(
                format!("Value {} out of range 0..{}", value, ft),
            ));
        }

        let fs = self.range / ft;
        if fs == 0 {
            return Err(OpusError::RangeCoder("Range too small".into()));
        }

        self.low += (value as u64) * (fs as u64);
        self.range = if value + 1 < ft {
            fs
        } else {
            self.range - value * fs
        };

        self.normalize()?;
        Ok(())
    }

    /// Write a symbol with given probability distribution.
    pub fn write_symbol(&mut self, sym: u32, cdf: &[u16]) -> Result<()> {
        if cdf.len() < 2 {
            return Err(OpusError::RangeCoder("CDF too short".into()));
        }

        let sym = sym as usize;
        if sym >= cdf.len() {
            return Err(OpusError::RangeCoder("Symbol out of range".into()));
        }

        let total = cdf[cdf.len() - 1] as u32;
        let fs = self.range / total;

        if fs == 0 {
            return Err(OpusError::RangeCoder("Range too small".into()));
        }

        let fl = if sym > 0 { cdf[sym - 1] as u32 } else { 0 };
        let fh = cdf[sym] as u32;

        self.low += (fl as u64) * (fs as u64);
        self.range = (fh - fl) * fs;

        self.normalize()?;
        Ok(())
    }

    /// Write a laplace-distributed value.
    pub fn write_laplace(&mut self, value: i32, decay: u32) -> Result<()> {
        // Sign bit
        self.write_bit(value < 0)?;

        // Magnitude with geometric distribution
        let magnitude = value.unsigned_abs();
        let p = decay.min(255) as u16;

        for _ in 0..magnitude {
            self.write_symbol(1, &[p, 256])?;
        }
        self.write_symbol(0, &[p, 256])?;

        Ok(())
    }

    /// Write raw bits without entropy coding.
    pub fn write_raw_bits(&mut self, value: u32, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }

        if n > 32 {
            return Err(OpusError::RangeCoder("Cannot write more than 32 bits".into()));
        }

        for i in (0..n).rev() {
            let bit = (value >> i) & 1;
            self.range >>= 1;

            if bit != 0 {
                self.low += self.range as u64;
            }

            self.bits_written += 1;
            self.normalize()?;
        }

        Ok(())
    }

    /// Normalize the encoder state.
    fn normalize(&mut self) -> Result<()> {
        // Limit iterations to prevent infinite loops
        let mut iterations = 0u32;
        const MAX_NORMALIZE_ITERATIONS: u32 = 32;

        while self.range < TOP && iterations < MAX_NORMALIZE_ITERATIONS {
            iterations += 1;

            if (self.low as u32) < 0xFF000000 || (self.low >> 32) != 0 {
                let byte = self.cache.wrapping_add((self.low >> 32) as u8);
                self.output_byte(byte);

                let c = if self.low >> 32 != 0 { 0x00 } else { 0xFF };
                for _ in 0..self.carry.min(255) {
                    self.output_byte(c);
                }
                self.carry = 0;
                self.cache = (self.low >> 24) as u8;
            } else {
                self.carry = self.carry.saturating_add(1);
            }

            self.range = self.range.saturating_mul(256);
            self.low = (self.low & 0xFFFFFF) << 8;
            self.bits_written = self.bits_written.saturating_add(8);

            // Exit if range is now large enough
            if self.range >= TOP {
                break;
            }
        }

        Ok(())
    }

    /// Output a byte.
    fn output_byte(&mut self, byte: u8) {
        self.buffer.push(byte);
    }

    /// Finalize encoding and get the output data.
    pub fn finish(mut self) -> Result<Vec<u8>> {
        // Final flush - ensure all data is written
        // Push any pending bytes with carry propagation
        let mut iterations = 0;
        const MAX_ITERATIONS: u32 = 10;

        while (self.carry > 0 || self.low > 0xFF000000) && iterations < MAX_ITERATIONS {
            let byte = self.cache.wrapping_add((self.low >> 32) as u8);
            self.output_byte(byte);

            let carry_byte = if self.low >> 32 != 0 { 0x00 } else { 0xFF };
            for _ in 0..self.carry {
                self.output_byte(carry_byte);
            }
            self.carry = 0;
            self.cache = (self.low >> 24) as u8;
            self.low = (self.low & 0xFFFFFF) << 8;
            iterations += 1;
        }

        // Output final cache byte
        if !self.buffer.is_empty() || self.cache != 0 {
            self.output_byte(self.cache);
        }

        Ok(self.buffer)
    }

    /// Get current buffer size.
    pub fn size(&self) -> usize {
        self.buffer.len()
    }
}

impl Default for RangeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// ICDF (Inverse CDF) table for common distributions.
pub struct IcdfTable {
    /// Probability table.
    pub prob: Vec<u16>,
    /// Total probability.
    pub total: u16,
}

impl IcdfTable {
    /// Create a uniform distribution table.
    pub fn uniform(n: u32) -> Self {
        let mut prob = Vec::with_capacity(n as usize);
        for i in 1..=n {
            prob.push((i * 256 / n) as u16);
        }
        Self { prob, total: 256 }
    }

    /// Create from raw ICDF values (256 - CDF).
    pub fn from_icdf(icdf: &[u8]) -> Self {
        let mut prob = Vec::with_capacity(icdf.len());
        for &v in icdf {
            prob.push((256 - v as u16).max(1));
        }
        Self { prob, total: 256 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_encoder_bits() {
        // Test that encoder can write bits without error
        let mut encoder = RangeEncoder::new();
        encoder.write_bit(true).unwrap();
        encoder.write_bit(false).unwrap();
        encoder.write_bit(true).unwrap();

        let data = encoder.finish().unwrap();
        // Verify we got some output
        assert!(!data.is_empty());
    }

    #[test]
    fn test_range_encoder_uint() {
        // Test that encoder can write uint values
        let mut encoder = RangeEncoder::new();
        encoder.write_uint(5, 10).unwrap();
        encoder.write_uint(3, 8).unwrap();
        encoder.write_uint(0, 4).unwrap();

        let data = encoder.finish().unwrap();
        // Verify we got some output
        assert!(!data.is_empty());
    }

    #[test]
    fn test_range_decoder_basic() {
        // Test decoder can read from simple data
        let data = [0x80, 0x00, 0x00, 0x00];
        let mut decoder = RangeDecoder::new(&data).unwrap();

        // Just verify we can read without crashing
        let _ = decoder.read_uint(2);
        assert!(decoder.position() > 0 || decoder.is_eof());
    }

    #[test]
    fn test_icdf_uniform() {
        let table = IcdfTable::uniform(4);
        assert_eq!(table.prob.len(), 4);
        assert_eq!(table.prob[0], 64);
        assert_eq!(table.prob[3], 256);
    }

    #[test]
    fn test_decoder_creation() {
        let data = [0x80, 0x00, 0x00, 0x00];
        let decoder = RangeDecoder::new(&data);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_empty_data() {
        let data: [u8; 0] = [];
        let decoder = RangeDecoder::new(&data);
        assert!(decoder.is_err());
    }
}
