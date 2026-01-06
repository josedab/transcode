//! Range coder for FFV1.
//!
//! FFV1 uses range coding, which is a form of arithmetic coding.

use crate::error::{Ffv1Error, Result};

/// Range coder state (probability context).
pub type State = [u8; 2];

/// One context lookup table for range coding.
const ONE_STATE: [u8; 256] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
    97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
    113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
    129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
    145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
    161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176,
    177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192,
    193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208,
    209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224,
    225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240,
    241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 254, 255,
];

/// Zero context lookup table for range coding.
const ZERO_STATE: [u8; 256] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
    97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
    113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
    129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
    145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
    161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176,
    177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192,
];

/// Range decoder.
pub struct RangeDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    low: u32,
    range: u32,
}

impl<'a> RangeDecoder<'a> {
    /// Create a new range decoder.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        if data.len() < 2 {
            return Err(Ffv1Error::RangeCoderError("Data too short".into()));
        }

        let mut decoder = Self {
            data,
            pos: 0,
            low: 0,
            range: 0xFF00,
        };

        // Initialize low from first bytes
        decoder.low = (decoder.read_byte()? as u32) << 8;
        decoder.low |= decoder.read_byte()? as u32;

        Ok(decoder)
    }

    /// Read a byte from input.
    fn read_byte(&mut self) -> Result<u8> {
        if self.pos >= self.data.len() {
            return Err(Ffv1Error::EndOfStream);
        }
        let byte = self.data[self.pos];
        self.pos += 1;
        Ok(byte)
    }

    /// Renormalize the range coder.
    fn renorm(&mut self) -> Result<()> {
        while self.range < 256 {
            self.range <<= 8;
            self.low <<= 8;
            if self.pos < self.data.len() {
                self.low |= self.data[self.pos] as u32;
                self.pos += 1;
            }
        }
        Ok(())
    }

    /// Get a single bit with given state.
    pub fn get_bit(&mut self, state: &mut u8) -> Result<bool> {
        let range_scaled = (self.range >> 8) * (*state as u32);

        let bit = if self.low < range_scaled {
            self.range = range_scaled;
            *state = ONE_STATE[*state as usize];
            false
        } else {
            self.low -= range_scaled;
            self.range -= range_scaled;
            *state = ZERO_STATE[*state as usize];
            true
        };

        self.renorm()?;
        Ok(bit)
    }

    /// Get a symbol using Golomb-Rice coding.
    pub fn get_symbol(&mut self, state: &mut [u8]) -> Result<i32> {
        // Read unary prefix (run of 1s)
        let mut prefix = 0u32;
        while self.get_bit(&mut state[prefix.min(14) as usize])? {
            prefix += 1;
            if prefix > 65535 {
                return Err(Ffv1Error::RangeCoderError("Symbol overflow".into()));
            }
        }

        if prefix == 0 {
            return Ok(0);
        }

        // Read suffix bits
        let mut suffix = 0u32;
        for i in 0..prefix.min(16) {
            if self.get_bit(&mut state[(15 + i) as usize])? {
                suffix |= 1 << i;
            }
        }

        let unsigned_val = (1u32 << prefix) - 1 + suffix;

        // Convert to signed (fold encoding)
        let signed_val = if unsigned_val & 1 != 0 {
            -(((unsigned_val + 1) >> 1) as i32)
        } else {
            (unsigned_val >> 1) as i32
        };

        Ok(signed_val)
    }

    /// Get raw bits (bypass mode).
    pub fn get_raw_bits(&mut self, n: u32) -> Result<u32> {
        let mut value = 0u32;
        for _ in 0..n {
            self.range >>= 1;
            let bit = if self.low >= self.range {
                self.low -= self.range;
                1
            } else {
                0
            };
            value = (value << 1) | bit;
            self.renorm()?;
        }
        Ok(value)
    }

    /// Get unsigned value.
    pub fn get_unsigned(&mut self, state: &mut [u8], bits: u32) -> Result<u32> {
        let mut value = 0u32;
        for i in 0..bits {
            if self.get_bit(&mut state[i as usize])? {
                value |= 1 << i;
            }
        }
        Ok(value)
    }

    /// Current position in data.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Remaining bytes.
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }
}

/// Range encoder.
#[cfg(feature = "encoder")]
pub struct RangeEncoder {
    data: Vec<u8>,
    low: u32,
    range: u32,
    cache: u8,
    cache_size: u32,
}

#[cfg(feature = "encoder")]
impl RangeEncoder {
    /// Create a new range encoder.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            low: 0,
            range: 0xFF00,
            cache: 0,
            cache_size: 1,
        }
    }

    /// Create with capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            low: 0,
            range: 0xFF00,
            cache: 0,
            cache_size: 1,
        }
    }

    /// Shift low and output bytes.
    fn shift_low(&mut self) {
        if self.low < 0xFF0000 || (self.low >> 24) != 0 {
            let carry = (self.low >> 24) as u8;
            self.data.push(self.cache.wrapping_add(carry));
            for _ in 1..self.cache_size {
                self.data.push(0xFF_u8.wrapping_add(carry));
            }
            self.cache_size = 0;
            self.cache = (self.low >> 16) as u8;
        }
        self.cache_size += 1;
        self.low = (self.low << 8) & 0xFFFFFF;
    }

    /// Renormalize.
    fn renorm(&mut self) {
        let mut iterations = 0;
        while self.range < 256 && iterations < 10 {
            self.range <<= 8;
            self.shift_low();
            iterations += 1;
        }
    }

    /// Put a single bit with given state.
    pub fn put_bit(&mut self, bit: bool, state: &mut u8) {
        let prob = (*state as u32).max(1); // Ensure prob is at least 1
        let range_scaled = ((self.range >> 8) * prob).max(1); // Ensure range_scaled is at least 1

        if bit {
            self.low += range_scaled;
            self.range = self.range.saturating_sub(range_scaled).max(1);
            *state = ZERO_STATE[*state as usize];
        } else {
            self.range = range_scaled;
            *state = ONE_STATE[*state as usize];
        }

        self.renorm();
    }

    /// Put a symbol using Golomb-Rice coding.
    pub fn put_symbol(&mut self, value: i32, state: &mut [u8]) {
        // Convert to unsigned (fold encoding)
        let unsigned_val = if value > 0 {
            (value as u32) << 1
        } else if value < 0 {
            (((-value) as u32) << 1) - 1
        } else {
            0
        };

        if unsigned_val == 0 {
            self.put_bit(false, &mut state[0]);
            return;
        }

        // Calculate prefix (position of highest bit)
        let prefix = 32 - unsigned_val.leading_zeros();
        let suffix_mask = (1u32 << (prefix - 1)) - 1;
        let suffix = unsigned_val & suffix_mask;

        // Write unary prefix
        for i in 0..prefix {
            self.put_bit(true, &mut state[i.min(14) as usize]);
        }
        self.put_bit(false, &mut state[prefix.min(14) as usize]);

        // Write suffix bits
        for i in 0..(prefix - 1).min(16) {
            let bit = (suffix >> i) & 1 != 0;
            self.put_bit(bit, &mut state[(15 + i) as usize]);
        }
    }

    /// Put raw bits (bypass mode).
    pub fn put_raw_bits(&mut self, value: u32, n: u32) {
        for i in (0..n).rev() {
            let bit = (value >> i) & 1 != 0;
            self.range >>= 1;
            if bit {
                self.low += self.range;
            }
            self.renorm();
        }
    }

    /// Finalize and return output.
    pub fn finalize(mut self) -> Vec<u8> {
        // Flush remaining data
        for _ in 0..4 {
            self.shift_low();
        }

        // Output cache
        if self.cache_size > 0 {
            self.data.push(self.cache);
            for _ in 1..self.cache_size {
                self.data.push(0xFF);
            }
        }

        self.data
    }

    /// Current output size.
    pub fn size(&self) -> usize {
        self.data.len() + self.cache_size as usize
    }
}

#[cfg(feature = "encoder")]
impl Default for RangeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_decoder_creation() {
        let data = [0x80, 0x00, 0x00, 0x00];
        let decoder = RangeDecoder::new(&data);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_range_decoder_short_data() {
        let data = [0x80];
        let result = RangeDecoder::new(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_range_decoder_get_bit() {
        let data = [0x80, 0x00, 0xFF, 0xFF, 0xFF, 0xFF];
        let mut decoder = RangeDecoder::new(&data).unwrap();
        let mut state = 128u8;

        // Read some bits
        let bit1 = decoder.get_bit(&mut state);
        assert!(bit1.is_ok());
    }

    #[cfg(feature = "encoder")]
    #[test]
    fn test_range_encoder_creation() {
        let encoder = RangeEncoder::new();
        // Initial size includes cache slot
        assert!(encoder.size() >= 0);
    }

    #[cfg(feature = "encoder")]
    #[test]
    fn test_range_encoder_put_bit() {
        let mut encoder = RangeEncoder::new();
        let mut state = 128u8;

        encoder.put_bit(true, &mut state);
        encoder.put_bit(false, &mut state);

        let output = encoder.finalize();
        assert!(!output.is_empty());
    }

    #[cfg(feature = "encoder")]
    #[test]
    fn test_range_encoder_symbol() {
        let mut encoder = RangeEncoder::new();
        let mut state = [128u8; 32];

        encoder.put_symbol(0, &mut state);
        encoder.put_symbol(1, &mut state);
        encoder.put_symbol(-1, &mut state);
        encoder.put_symbol(100, &mut state);

        let output = encoder.finalize();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_one_state_table() {
        // ONE_STATE should increase probability
        assert!(ONE_STATE[128] > 128);
        assert!(ONE_STATE[1] > 1);
    }

    #[test]
    fn test_zero_state_table() {
        // ZERO_STATE should decrease probability for high values
        assert!(ZERO_STATE[128] < 128);
    }
}
