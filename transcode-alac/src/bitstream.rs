//! ALAC bitstream reader and writer.

use crate::error::{AlacError, Result};

/// Bitstream reader for ALAC decoding.
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_pos: u32,
    cache: u64,
    cache_bits: u32,
}

impl<'a> BitReader<'a> {
    /// Create a new bit reader.
    pub fn new(data: &'a [u8]) -> Self {
        let mut reader = Self {
            data,
            pos: 0,
            bit_pos: 0,
            cache: 0,
            cache_bits: 0,
        };
        reader.refill();
        reader
    }

    /// Refill the cache.
    fn refill(&mut self) {
        while self.cache_bits <= 56 && self.pos < self.data.len() {
            self.cache = (self.cache << 8) | self.data[self.pos] as u64;
            self.pos += 1;
            self.cache_bits += 8;
        }
    }

    /// Read n bits (up to 32).
    pub fn read_bits(&mut self, n: u32) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(AlacError::BitstreamError("Too many bits requested".into()));
        }

        if self.cache_bits < n {
            self.refill();
            if self.cache_bits < n {
                return Err(AlacError::EndOfStream);
            }
        }

        let shift = self.cache_bits - n;
        let value = (self.cache >> shift) as u32 & ((1u64 << n) - 1) as u32;
        self.cache_bits -= n;
        self.bit_pos += n;

        Ok(value)
    }

    /// Read a single bit.
    pub fn read_bit(&mut self) -> Result<bool> {
        Ok(self.read_bits(1)? != 0)
    }

    /// Read a signed value (sign-magnitude).
    pub fn read_signed(&mut self, n: u32) -> Result<i32> {
        let value = self.read_bits(n)? as i32;
        let sign_bit = 1 << (n - 1);
        if value & sign_bit != 0 {
            Ok(value | !((sign_bit << 1) - 1))
        } else {
            Ok(value)
        }
    }

    /// Read unary coded value.
    pub fn read_unary(&mut self) -> Result<u32> {
        let mut count = 0u32;
        while !self.read_bit()? {
            count += 1;
            if count > 65535 {
                return Err(AlacError::BitstreamError("Unary overflow".into()));
            }
        }
        Ok(count)
    }

    /// Get current bit position.
    pub fn position(&self) -> usize {
        (self.pos * 8) - self.cache_bits as usize
    }

    /// Check if at end of stream.
    pub fn is_empty(&self) -> bool {
        self.cache_bits == 0 && self.pos >= self.data.len()
    }

    /// Align to byte boundary.
    pub fn align(&mut self) {
        let bits_to_skip = self.bit_pos & 7;
        if bits_to_skip > 0 {
            let _ = self.read_bits(8 - bits_to_skip);
        }
    }

    /// Read Rice-encoded value.
    pub fn read_rice(&mut self, k: u32, _kb: u32, _mb: u32) -> Result<i32> {
        // Read unary part (MSBs)
        let msbs = self.read_unary()?;

        // Read binary part (LSBs)
        let mut lsbs = 0u32;
        if k > 0 {
            lsbs = self.read_bits(k)?;
        }

        // Combine parts
        let unsigned_val = (msbs << k) | lsbs;

        // Convert to signed (fold encoding)
        let signed_val = if unsigned_val & 1 != 0 {
            -(((unsigned_val + 1) >> 1) as i32)
        } else {
            (unsigned_val >> 1) as i32
        };

        Ok(signed_val)
    }
}

/// Bitstream writer for ALAC encoding.
#[cfg(feature = "encoder")]
pub struct BitWriter {
    data: Vec<u8>,
    cache: u64,
    cache_bits: u32,
}

#[cfg(feature = "encoder")]
impl BitWriter {
    /// Create a new bit writer.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            cache: 0,
            cache_bits: 0,
        }
    }

    /// Create with capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            cache: 0,
            cache_bits: 0,
        }
    }

    /// Flush cache to output.
    fn flush_cache(&mut self) {
        while self.cache_bits >= 8 {
            self.cache_bits -= 8;
            self.data.push((self.cache >> self.cache_bits) as u8);
        }
    }

    /// Write n bits.
    pub fn write_bits(&mut self, value: u32, n: u32) {
        if n == 0 {
            return;
        }

        let masked = value & ((1u64 << n) - 1) as u32;
        self.cache = (self.cache << n) | masked as u64;
        self.cache_bits += n;

        if self.cache_bits >= 32 {
            self.flush_cache();
        }
    }

    /// Write a single bit.
    pub fn write_bit(&mut self, value: bool) {
        self.write_bits(value as u32, 1);
    }

    /// Write unary coded value.
    pub fn write_unary(&mut self, value: u32) {
        // Write 'value' zeros followed by a one
        for _ in 0..value {
            self.write_bit(false);
        }
        self.write_bit(true);
    }

    /// Write Rice-encoded value.
    pub fn write_rice(&mut self, value: i32, k: u32) {
        // Convert to unsigned (fold encoding)
        let unsigned_val = if value < 0 {
            ((-value as u32) << 1) - 1
        } else {
            (value as u32) << 1
        };

        // Split into MSBs (unary) and LSBs (binary)
        let msbs = unsigned_val >> k;
        let lsbs = unsigned_val & ((1 << k) - 1);

        self.write_unary(msbs);
        if k > 0 {
            self.write_bits(lsbs, k);
        }
    }

    /// Align to byte boundary.
    pub fn align(&mut self) {
        if self.cache_bits & 7 != 0 {
            let padding = 8 - (self.cache_bits & 7);
            self.write_bits(0, padding);
        }
    }

    /// Finalize and return output.
    pub fn finalize(mut self) -> Vec<u8> {
        // Flush remaining bits
        if self.cache_bits > 0 {
            self.align();
            self.flush_cache();
        }
        self.data
    }

    /// Get current bit position.
    pub fn position(&self) -> usize {
        self.data.len() * 8 + self.cache_bits as usize
    }
}

#[cfg(feature = "encoder")]
impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_reader_basic() {
        let data = [0b10110100, 0b01010101];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.read_bits(4).unwrap(), 0b1011);
        assert_eq!(reader.read_bits(4).unwrap(), 0b0100);
        assert_eq!(reader.read_bits(8).unwrap(), 0b01010101);
    }

    #[test]
    fn test_bit_reader_single_bits() {
        let data = [0b10110100];
        let mut reader = BitReader::new(&data);

        assert!(reader.read_bit().unwrap());
        assert!(!reader.read_bit().unwrap());
        assert!(reader.read_bit().unwrap());
        assert!(reader.read_bit().unwrap());
    }

    #[test]
    fn test_bit_reader_unary() {
        // 00001 = unary 4
        let data = [0b00001000];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.read_unary().unwrap(), 4);
    }

    #[cfg(feature = "encoder")]
    #[test]
    fn test_bit_writer_basic() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b1011, 4);
        writer.write_bits(0b0100, 4);
        writer.write_bits(0b01010101, 8);

        let data = writer.finalize();
        assert_eq!(data, vec![0b10110100, 0b01010101]);
    }

    #[cfg(feature = "encoder")]
    #[test]
    fn test_bit_writer_unary() {
        let mut writer = BitWriter::new();
        writer.write_unary(4); // 00001
        writer.align();

        let data = writer.finalize();
        assert_eq!(data[0] & 0b11111000, 0b00001000);
    }
}
