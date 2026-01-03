//! Vertical Interval Timecode (VITC) encoding and decoding.
//!
//! VITC is a form of SMPTE timecode embedded in the vertical blanking interval
//! of a video signal. Unlike LTC, VITC can be read while the tape is paused
//! or in slow motion.
//!
//! VITC consists of 90 bits per line:
//! - 2 sync bits at start
//! - 8 groups of 10 bits each (8 data + 2 sync)
//! - CRC bits
//!
//! VITC is typically recorded on two lines for redundancy.

#![allow(clippy::needless_range_loop)]

use crate::error::{Result, TimecodeError};
use crate::smpte::{FrameRate, Timecode};
use serde::{Deserialize, Serialize};

/// Number of bits in a VITC line.
pub const VITC_LINE_BITS: usize = 90;

/// VITC sync pattern (2 bits: 11).
pub const VITC_SYNC_PATTERN: u8 = 0b11;

/// VITC field/frame flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct VitcFlags {
    /// Drop-frame flag
    pub drop_frame: bool,
    /// Color frame flag
    pub color_frame: bool,
    /// Field mark (0 = field 1, 1 = field 2)
    pub field_mark: bool,
    /// Binary group flags (2 bits)
    pub binary_group_flags: u8,
}

/// VITC data structure.
///
/// Contains the decoded VITC data including timecode, user bits, and flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VitcData {
    /// Frame units (0-9)
    pub frame_units: u8,
    /// Frame tens (0-2)
    pub frame_tens: u8,
    /// Seconds units (0-9)
    pub second_units: u8,
    /// Seconds tens (0-5)
    pub second_tens: u8,
    /// Minutes units (0-9)
    pub minute_units: u8,
    /// Minutes tens (0-5)
    pub minute_tens: u8,
    /// Hours units (0-9)
    pub hour_units: u8,
    /// Hours tens (0-2)
    pub hour_tens: u8,
    /// User bits (8 groups of 4 bits)
    pub user_bits: [u8; 8],
    /// Flags
    pub flags: VitcFlags,
    /// CRC value
    pub crc: u8,
}

impl VitcData {
    /// Create VITC data from timecode.
    #[must_use]
    pub fn from_timecode(tc: &Timecode) -> Self {
        let mut data = Self {
            frame_units: tc.frames % 10,
            frame_tens: tc.frames / 10,
            second_units: tc.seconds % 10,
            second_tens: tc.seconds / 10,
            minute_units: tc.minutes % 10,
            minute_tens: tc.minutes / 10,
            hour_units: tc.hours % 10,
            hour_tens: tc.hours / 10,
            user_bits: [0; 8],
            flags: VitcFlags {
                drop_frame: tc.drop_frame,
                ..Default::default()
            },
            crc: 0,
        };
        data.crc = data.calculate_crc();
        data
    }

    /// Convert to timecode.
    #[must_use]
    pub fn to_timecode(&self, frame_rate: FrameRate) -> Timecode {
        Timecode {
            hours: self.hour_tens * 10 + self.hour_units,
            minutes: self.minute_tens * 10 + self.minute_units,
            seconds: self.second_tens * 10 + self.second_units,
            frames: self.frame_tens * 10 + self.frame_units,
            frame_rate,
            drop_frame: self.flags.drop_frame,
        }
    }

    /// Set user bits from a 32-bit value.
    pub fn set_user_bits_u32(&mut self, value: u32) {
        for i in 0..8 {
            self.user_bits[i] = ((value >> (i * 4)) & 0x0F) as u8;
        }
    }

    /// Get user bits as a 32-bit value.
    #[must_use]
    pub fn get_user_bits_u32(&self) -> u32 {
        let mut value = 0u32;
        for i in 0..8 {
            value |= (self.user_bits[i] as u32 & 0x0F) << (i * 4);
        }
        value
    }

    /// Calculate CRC for the VITC data.
    #[must_use]
    pub fn calculate_crc(&self) -> u8 {
        // Simple CRC-8 calculation
        let mut crc = 0u8;

        // XOR all data nibbles
        crc ^= self.frame_units;
        crc ^= self.frame_tens;
        crc ^= self.second_units;
        crc ^= self.second_tens;
        crc ^= self.minute_units;
        crc ^= self.minute_tens;
        crc ^= self.hour_units;
        crc ^= self.hour_tens;

        for &ub in &self.user_bits {
            crc ^= ub;
        }

        // Include flags
        if self.flags.drop_frame {
            crc ^= 0x01;
        }
        if self.flags.color_frame {
            crc ^= 0x02;
        }
        if self.flags.field_mark {
            crc ^= 0x04;
        }
        crc ^= self.flags.binary_group_flags;

        crc
    }

    /// Verify the CRC.
    #[must_use]
    pub fn verify_crc(&self) -> bool {
        self.calculate_crc() == self.crc
    }

    /// Encode to VITC line bits.
    ///
    /// Returns 90 bits packed into a 12-byte array.
    #[must_use]
    pub fn encode(&self) -> [u8; 12] {
        let mut bits = [0u8; 12];

        // Helper to set bits with sync
        let mut bit_pos = 0;

        let set_group = |bits: &mut [u8], pos: &mut usize, data: u8, sync: u8| {
            // 8 data bits + 2 sync bits
            for i in 0..8 {
                let byte_idx = *pos / 8;
                let bit_idx = *pos % 8;
                if (data >> i) & 1 == 1 {
                    bits[byte_idx] |= 1 << bit_idx;
                }
                *pos += 1;
            }
            // Sync bits
            for i in 0..2 {
                let byte_idx = *pos / 8;
                let bit_idx = *pos % 8;
                if (sync >> i) & 1 == 1 {
                    bits[byte_idx] |= 1 << bit_idx;
                }
                *pos += 1;
            }
        };

        // Bit layout per SMPTE 12M:
        // Group 1: Frame units (4) + user bits 1 (4) + sync (2)
        // Group 2: Frame tens (2) + drop frame + color frame + user bits 2 (4) + sync (2)
        // Group 3: Seconds units (4) + user bits 3 (4) + sync (2)
        // Group 4: Seconds tens (3) + field mark + user bits 4 (4) + sync (2)
        // Group 5: Minutes units (4) + user bits 5 (4) + sync (2)
        // Group 6: Minutes tens (3) + binary group flag 1 + user bits 6 (4) + sync (2)
        // Group 7: Hours units (4) + user bits 7 (4) + sync (2)
        // Group 8: Hours tens (2) + reserved + binary group flag 2 + user bits 8 (4) + sync (2)
        // CRC: 8 bits + sync (2)

        // Group 1
        let g1 = self.frame_units | (self.user_bits[0] << 4);
        set_group(&mut bits, &mut bit_pos, g1, VITC_SYNC_PATTERN);

        // Group 2
        let g2 = self.frame_tens
            | (if self.flags.drop_frame { 0x04 } else { 0 })
            | (if self.flags.color_frame { 0x08 } else { 0 })
            | (self.user_bits[1] << 4);
        set_group(&mut bits, &mut bit_pos, g2, VITC_SYNC_PATTERN);

        // Group 3
        let g3 = self.second_units | (self.user_bits[2] << 4);
        set_group(&mut bits, &mut bit_pos, g3, VITC_SYNC_PATTERN);

        // Group 4
        let g4 = self.second_tens
            | (if self.flags.field_mark { 0x08 } else { 0 })
            | (self.user_bits[3] << 4);
        set_group(&mut bits, &mut bit_pos, g4, VITC_SYNC_PATTERN);

        // Group 5
        let g5 = self.minute_units | (self.user_bits[4] << 4);
        set_group(&mut bits, &mut bit_pos, g5, VITC_SYNC_PATTERN);

        // Group 6
        let g6 = self.minute_tens
            | ((self.flags.binary_group_flags & 0x01) << 3)
            | (self.user_bits[5] << 4);
        set_group(&mut bits, &mut bit_pos, g6, VITC_SYNC_PATTERN);

        // Group 7
        let g7 = self.hour_units | (self.user_bits[6] << 4);
        set_group(&mut bits, &mut bit_pos, g7, VITC_SYNC_PATTERN);

        // Group 8
        let g8 = self.hour_tens
            | ((self.flags.binary_group_flags & 0x02) << 2)
            | (self.user_bits[7] << 4);
        set_group(&mut bits, &mut bit_pos, g8, VITC_SYNC_PATTERN);

        // CRC group
        set_group(&mut bits, &mut bit_pos, self.crc, VITC_SYNC_PATTERN);

        bits
    }

    /// Decode from VITC line bits.
    pub fn decode(bits: &[u8; 12]) -> Result<Self> {
        let mut bit_pos = 0;

        let get_group = |bits: &[u8], pos: &mut usize| -> Result<(u8, u8)> {
            let mut data = 0u8;
            let mut sync = 0u8;

            // 8 data bits
            for i in 0..8 {
                let byte_idx = *pos / 8;
                let bit_idx = *pos % 8;
                if (bits[byte_idx] >> bit_idx) & 1 == 1 {
                    data |= 1 << i;
                }
                *pos += 1;
            }
            // 2 sync bits
            for i in 0..2 {
                let byte_idx = *pos / 8;
                let bit_idx = *pos % 8;
                if (bits[byte_idx] >> bit_idx) & 1 == 1 {
                    sync |= 1 << i;
                }
                *pos += 1;
            }

            // Verify sync pattern
            if sync != VITC_SYNC_PATTERN {
                return Err(TimecodeError::vitc(format!(
                    "Invalid sync pattern at bit {}: expected {:02b}, got {:02b}",
                    *pos - 2,
                    VITC_SYNC_PATTERN,
                    sync
                )));
            }

            Ok((data, sync))
        };

        // Decode groups
        let (g1, _) = get_group(bits, &mut bit_pos)?;
        let (g2, _) = get_group(bits, &mut bit_pos)?;
        let (g3, _) = get_group(bits, &mut bit_pos)?;
        let (g4, _) = get_group(bits, &mut bit_pos)?;
        let (g5, _) = get_group(bits, &mut bit_pos)?;
        let (g6, _) = get_group(bits, &mut bit_pos)?;
        let (g7, _) = get_group(bits, &mut bit_pos)?;
        let (g8, _) = get_group(bits, &mut bit_pos)?;
        let (crc, _) = get_group(bits, &mut bit_pos)?;

        let data = Self {
            frame_units: g1 & 0x0F,
            frame_tens: g2 & 0x03,
            second_units: g3 & 0x0F,
            second_tens: g4 & 0x07,
            minute_units: g5 & 0x0F,
            minute_tens: g6 & 0x07,
            hour_units: g7 & 0x0F,
            hour_tens: g8 & 0x03,
            user_bits: [
                (g1 >> 4) & 0x0F,
                (g2 >> 4) & 0x0F,
                (g3 >> 4) & 0x0F,
                (g4 >> 4) & 0x0F,
                (g5 >> 4) & 0x0F,
                (g6 >> 4) & 0x0F,
                (g7 >> 4) & 0x0F,
                (g8 >> 4) & 0x0F,
            ],
            flags: VitcFlags {
                drop_frame: (g2 & 0x04) != 0,
                color_frame: (g2 & 0x08) != 0,
                field_mark: (g4 & 0x08) != 0,
                binary_group_flags: ((g6 >> 3) & 0x01) | (((g8 >> 2) & 0x02) & 0x02),
            },
            crc,
        };

        Ok(data)
    }
}

impl Default for VitcData {
    fn default() -> Self {
        let mut data = Self {
            frame_units: 0,
            frame_tens: 0,
            second_units: 0,
            second_tens: 0,
            minute_units: 0,
            minute_tens: 0,
            hour_units: 0,
            hour_tens: 0,
            user_bits: [0; 8],
            flags: VitcFlags::default(),
            crc: 0,
        };
        data.crc = data.calculate_crc();
        data
    }
}

/// VITC line encoder.
///
/// Generates pixel data for embedding VITC in video lines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VitcEncoder {
    /// Line width in pixels
    line_width: u32,
    /// Bits per pixel (samples per bit)
    samples_per_bit: f64,
    /// High level value
    high_level: u8,
    /// Low level value
    low_level: u8,
}

impl VitcEncoder {
    /// Create a new VITC encoder.
    ///
    /// # Arguments
    /// * `line_width` - Width of the video line in pixels
    pub fn new(line_width: u32) -> Self {
        Self {
            line_width,
            samples_per_bit: line_width as f64 / VITC_LINE_BITS as f64,
            high_level: 200,
            low_level: 16,
        }
    }

    /// Set the signal levels.
    pub fn set_levels(&mut self, low: u8, high: u8) {
        self.low_level = low;
        self.high_level = high;
    }

    /// Encode VITC data to a video line.
    ///
    /// Returns pixel values for one line.
    #[must_use]
    pub fn encode(&self, data: &VitcData) -> Vec<u8> {
        let bits = data.encode();
        let mut pixels = Vec::with_capacity(self.line_width as usize);

        for i in 0..self.line_width {
            // Determine which bit this pixel corresponds to
            let bit_pos = (i as f64 / self.samples_per_bit) as usize;

            if bit_pos < VITC_LINE_BITS {
                let byte_idx = bit_pos / 8;
                let bit_idx = bit_pos % 8;
                let bit_value = (bits[byte_idx] >> bit_idx) & 1;

                pixels.push(if bit_value == 1 {
                    self.high_level
                } else {
                    self.low_level
                });
            } else {
                pixels.push(self.low_level);
            }
        }

        pixels
    }
}

/// VITC line decoder.
///
/// Extracts VITC data from video line pixels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VitcDecoder {
    /// Line width in pixels
    line_width: u32,
    /// Samples per bit
    samples_per_bit: f64,
    /// Threshold for bit detection
    threshold: u8,
}

impl VitcDecoder {
    /// Create a new VITC decoder.
    pub fn new(line_width: u32) -> Self {
        Self {
            line_width,
            samples_per_bit: line_width as f64 / VITC_LINE_BITS as f64,
            threshold: 100,
        }
    }

    /// Set the detection threshold.
    pub fn set_threshold(&mut self, threshold: u8) {
        self.threshold = threshold;
    }

    /// Decode VITC data from a video line.
    pub fn decode(&self, pixels: &[u8]) -> Result<VitcData> {
        if pixels.len() < self.line_width as usize {
            return Err(TimecodeError::vitc(format!(
                "Line too short: {} pixels, expected {}",
                pixels.len(),
                self.line_width
            )));
        }

        let mut bits = [0u8; 12];

        for bit_pos in 0..VITC_LINE_BITS {
            // Sample multiple pixels for this bit and average
            let start_pixel = (bit_pos as f64 * self.samples_per_bit) as usize;
            let end_pixel = ((bit_pos + 1) as f64 * self.samples_per_bit) as usize;
            let end_pixel = end_pixel.min(pixels.len());

            let mut sum = 0u32;
            let count = (end_pixel - start_pixel).max(1);

            for i in start_pixel..end_pixel {
                sum += pixels[i] as u32;
            }

            let average = (sum / count as u32) as u8;
            let bit_value = if average > self.threshold { 1 } else { 0 };

            let byte_idx = bit_pos / 8;
            let bit_idx = bit_pos % 8;
            bits[byte_idx] |= bit_value << bit_idx;
        }

        VitcData::decode(&bits)
    }
}

/// Standard VITC line numbers for different video standards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VitcLineNumbers {
    /// NTSC: Lines 14 and 16 (field 1), 277 and 279 (field 2)
    Ntsc {
        /// Field 1 line numbers (primary, backup).
        field1: (u16, u16),
        /// Field 2 line numbers (primary, backup).
        field2: (u16, u16),
    },
    /// PAL: Lines 19 and 21 (field 1), 332 and 334 (field 2)
    Pal {
        /// Field 1 line numbers (primary, backup).
        field1: (u16, u16),
        /// Field 2 line numbers (primary, backup).
        field2: (u16, u16),
    },
}

impl VitcLineNumbers {
    /// Get NTSC line numbers.
    #[must_use]
    pub const fn ntsc() -> Self {
        Self::Ntsc {
            field1: (14, 16),
            field2: (277, 279),
        }
    }

    /// Get PAL line numbers.
    #[must_use]
    pub const fn pal() -> Self {
        Self::Pal {
            field1: (19, 21),
            field2: (332, 334),
        }
    }

    /// Get the primary line number for field 1.
    #[must_use]
    pub fn field1_primary(&self) -> u16 {
        match self {
            Self::Ntsc { field1, .. } => field1.0,
            Self::Pal { field1, .. } => field1.0,
        }
    }

    /// Get the backup line number for field 1.
    #[must_use]
    pub fn field1_backup(&self) -> u16 {
        match self {
            Self::Ntsc { field1, .. } => field1.1,
            Self::Pal { field1, .. } => field1.1,
        }
    }

    /// Get the primary line number for field 2.
    #[must_use]
    pub fn field2_primary(&self) -> u16 {
        match self {
            Self::Ntsc { field2, .. } => field2.0,
            Self::Pal { field2, .. } => field2.0,
        }
    }

    /// Get the backup line number for field 2.
    #[must_use]
    pub fn field2_backup(&self) -> u16 {
        match self {
            Self::Ntsc { field2, .. } => field2.1,
            Self::Pal { field2, .. } => field2.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_vitc_from_timecode() {
        let tc = Timecode::new(1, 30, 45, 12, FrameRate::Fps24).unwrap();
        let vitc = VitcData::from_timecode(&tc);

        assert_eq!(vitc.hour_tens, 0);
        assert_eq!(vitc.hour_units, 1);
        assert_eq!(vitc.minute_tens, 3);
        assert_eq!(vitc.minute_units, 0);
        assert_eq!(vitc.second_tens, 4);
        assert_eq!(vitc.second_units, 5);
        assert_eq!(vitc.frame_tens, 1);
        assert_eq!(vitc.frame_units, 2);
    }

    #[test]
    fn test_vitc_to_timecode() {
        let vitc = VitcData {
            hour_tens: 0,
            hour_units: 1,
            minute_tens: 3,
            minute_units: 0,
            second_tens: 4,
            second_units: 5,
            frame_tens: 1,
            frame_units: 2,
            ..Default::default()
        };

        let tc = vitc.to_timecode(FrameRate::Fps24);
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 30);
        assert_eq!(tc.seconds, 45);
        assert_eq!(tc.frames, 12);
    }

    #[test]
    fn test_vitc_encode_decode() {
        let tc = Timecode::new(12, 34, 56, 7, FrameRate::Fps24).unwrap();
        let original = VitcData::from_timecode(&tc);

        let encoded = original.encode();
        let decoded = VitcData::decode(&encoded).unwrap();

        assert_eq!(decoded.hour_tens, original.hour_tens);
        assert_eq!(decoded.hour_units, original.hour_units);
        assert_eq!(decoded.minute_tens, original.minute_tens);
        assert_eq!(decoded.minute_units, original.minute_units);
        assert_eq!(decoded.second_tens, original.second_tens);
        assert_eq!(decoded.second_units, original.second_units);
        assert_eq!(decoded.frame_tens, original.frame_tens);
        assert_eq!(decoded.frame_units, original.frame_units);
    }

    #[test]
    fn test_vitc_user_bits() {
        let mut vitc = VitcData::default();
        vitc.set_user_bits_u32(0xABCD1234);
        assert_eq!(vitc.get_user_bits_u32(), 0xABCD1234);
    }

    #[test]
    fn test_vitc_crc() {
        let vitc = VitcData::from_timecode(&Timecode::new(1, 2, 3, 4, FrameRate::Fps24).unwrap());
        assert!(vitc.verify_crc());

        let mut bad_vitc = vitc;
        bad_vitc.crc = !bad_vitc.crc;
        assert!(!bad_vitc.verify_crc());
    }

    #[test]
    fn test_vitc_flags() {
        let tc = Timecode::new_drop_frame(1, 2, 3, 4, FrameRate::Fps29_97).unwrap();
        let vitc = VitcData::from_timecode(&tc);

        assert!(vitc.flags.drop_frame);
        assert!(!vitc.flags.color_frame);
    }

    #[test]
    fn test_vitc_encoder() {
        let encoder = VitcEncoder::new(720);
        let vitc = VitcData::default();

        let pixels = encoder.encode(&vitc);
        assert_eq!(pixels.len(), 720);

        // Check that pixels are either high or low level
        for &p in &pixels {
            assert!(p == encoder.high_level || p == encoder.low_level);
        }
    }

    #[test]
    fn test_vitc_encoder_decoder_roundtrip() {
        let encoder = VitcEncoder::new(720);
        let decoder = VitcDecoder::new(720);

        let tc = Timecode::new(10, 20, 30, 15, FrameRate::Fps24).unwrap();
        let original = VitcData::from_timecode(&tc);

        let pixels = encoder.encode(&original);
        let decoded = decoder.decode(&pixels).unwrap();

        assert_eq!(decoded.hour_tens, original.hour_tens);
        assert_eq!(decoded.hour_units, original.hour_units);
        assert_eq!(decoded.minute_tens, original.minute_tens);
        assert_eq!(decoded.minute_units, original.minute_units);
        assert_eq!(decoded.second_tens, original.second_tens);
        assert_eq!(decoded.second_units, original.second_units);
        assert_eq!(decoded.frame_tens, original.frame_tens);
        assert_eq!(decoded.frame_units, original.frame_units);
    }

    #[test]
    fn test_vitc_line_numbers() {
        let ntsc = VitcLineNumbers::ntsc();
        assert_eq!(ntsc.field1_primary(), 14);
        assert_eq!(ntsc.field1_backup(), 16);
        assert_eq!(ntsc.field2_primary(), 277);

        let pal = VitcLineNumbers::pal();
        assert_eq!(pal.field1_primary(), 19);
        assert_eq!(pal.field2_primary(), 332);
    }

    #[test]
    fn test_vitc_serialization() {
        let vitc =
            VitcData::from_timecode(&Timecode::new(1, 30, 45, 12, FrameRate::Fps24).unwrap());
        let json = serde_json::to_string(&vitc).unwrap();
        let decoded: VitcData = serde_json::from_str(&json).unwrap();
        assert_eq!(vitc, decoded);
    }

    #[test]
    fn test_vitc_flags_serialization() {
        let flags = VitcFlags {
            drop_frame: true,
            color_frame: false,
            field_mark: true,
            binary_group_flags: 2,
        };
        let json = serde_json::to_string(&flags).unwrap();
        let decoded: VitcFlags = serde_json::from_str(&json).unwrap();
        assert_eq!(flags, decoded);
    }
}
