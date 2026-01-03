//! Linear Timecode (LTC) audio encoding and decoding.
//!
//! LTC is a form of SMPTE timecode that is encoded as an audio signal.
//! It can be recorded on a track of a tape recorder or transmitted over
//! audio cables.
//!
//! The LTC signal uses bi-phase modulation (Manchester encoding) where:
//! - A '0' bit has one transition in the middle of the bit period
//! - A '1' bit has two transitions (at start and middle)
//!
//! Each frame of LTC contains 80 bits organized as:
//! - User bits (32 bits across 8 groups)
//! - Timecode (24 bits for HH:MM:SS:FF)
//! - Sync word (16 bits: 0x3FFD)
//! - Various flags

use crate::error::{Result, TimecodeError};
use crate::smpte::{FrameRate, Timecode};
use serde::{Deserialize, Serialize};

/// Size of an LTC frame in bits.
pub const LTC_FRAME_BITS: usize = 80;

/// LTC sync word (last 16 bits of frame).
pub const LTC_SYNC_WORD: u16 = 0x3FFD;

/// LTC frame structure.
///
/// Contains the 80-bit LTC frame data including timecode,
/// user bits, and flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct LtcFrame {
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
    /// Drop-frame flag
    pub drop_frame: bool,
    /// Color frame flag
    pub color_frame: bool,
    /// Binary group flags (2 bits)
    pub binary_group_flags: u8,
    /// User bits (8 groups of 4 bits each)
    pub user_bits: [u8; 8],
    /// Polarity correction bit
    pub polarity_correction: bool,
}

impl LtcFrame {
    /// Create a new LTC frame from timecode.
    #[must_use]
    pub fn from_timecode(tc: &Timecode) -> Self {
        Self {
            frame_units: tc.frames % 10,
            frame_tens: tc.frames / 10,
            second_units: tc.seconds % 10,
            second_tens: tc.seconds / 10,
            minute_units: tc.minutes % 10,
            minute_tens: tc.minutes / 10,
            hour_units: tc.hours % 10,
            hour_tens: tc.hours / 10,
            drop_frame: tc.drop_frame,
            color_frame: false,
            binary_group_flags: 0,
            user_bits: [0; 8],
            polarity_correction: false,
        }
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
            drop_frame: self.drop_frame,
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

    /// Encode the frame to 80 bits.
    ///
    /// Returns the 80 bits packed into a 10-byte array, LSB first.
    #[must_use]
    pub fn encode(&self) -> [u8; 10] {
        let mut bits = [0u8; 10];

        // Bit layout (as per SMPTE 12M):
        // 0-3: Frame units
        // 4-7: User bits 1
        // 8-9: Frame tens
        // 10: Drop frame flag
        // 11: Color frame flag
        // 12-15: User bits 2
        // 16-19: Seconds units
        // 20-23: User bits 3
        // 24-26: Seconds tens
        // 27: Binary group flag 1
        // 28-31: User bits 4
        // 32-35: Minutes units
        // 36-39: User bits 5
        // 40-42: Minutes tens
        // 43: Binary group flag 2
        // 44-47: User bits 6
        // 48-51: Hours units
        // 52-55: User bits 7
        // 56-57: Hours tens
        // 58: Reserved
        // 59: Polarity correction
        // 60-63: User bits 8
        // 64-79: Sync word (0x3FFD)

        // Helper to set bits
        let set_bits = |bits: &mut [u8], start: usize, len: usize, value: u8| {
            for i in 0..len {
                let bit = (value >> i) & 1;
                let byte_idx = (start + i) / 8;
                let bit_idx = (start + i) % 8;
                if bit == 1 {
                    bits[byte_idx] |= 1 << bit_idx;
                }
            }
        };

        set_bits(&mut bits, 0, 4, self.frame_units);
        set_bits(&mut bits, 4, 4, self.user_bits[0]);
        set_bits(&mut bits, 8, 2, self.frame_tens);
        set_bits(&mut bits, 10, 1, self.drop_frame as u8);
        set_bits(&mut bits, 11, 1, self.color_frame as u8);
        set_bits(&mut bits, 12, 4, self.user_bits[1]);
        set_bits(&mut bits, 16, 4, self.second_units);
        set_bits(&mut bits, 20, 4, self.user_bits[2]);
        set_bits(&mut bits, 24, 3, self.second_tens);
        set_bits(&mut bits, 27, 1, self.binary_group_flags & 1);
        set_bits(&mut bits, 28, 4, self.user_bits[3]);
        set_bits(&mut bits, 32, 4, self.minute_units);
        set_bits(&mut bits, 36, 4, self.user_bits[4]);
        set_bits(&mut bits, 40, 3, self.minute_tens);
        set_bits(&mut bits, 43, 1, (self.binary_group_flags >> 1) & 1);
        set_bits(&mut bits, 44, 4, self.user_bits[5]);
        set_bits(&mut bits, 48, 4, self.hour_units);
        set_bits(&mut bits, 52, 4, self.user_bits[6]);
        set_bits(&mut bits, 56, 2, self.hour_tens);
        set_bits(&mut bits, 58, 1, 0); // Reserved
        set_bits(&mut bits, 59, 1, self.polarity_correction as u8);
        set_bits(&mut bits, 60, 4, self.user_bits[7]);

        // Sync word: 0x3FFD = 0011 1111 1111 1101
        bits[8] |= 0xFD; // bits 64-71
        bits[9] = 0x3F; // bits 72-79

        bits
    }

    /// Decode a frame from 80 bits.
    pub fn decode(bits: &[u8; 10]) -> Result<Self> {
        // Helper to get bits
        let get_bits = |bits: &[u8], start: usize, len: usize| -> u8 {
            let mut value = 0u8;
            for i in 0..len {
                let byte_idx = (start + i) / 8;
                let bit_idx = (start + i) % 8;
                if (bits[byte_idx] >> bit_idx) & 1 == 1 {
                    value |= 1 << i;
                }
            }
            value
        };

        // Verify sync word
        let sync_low = bits[8];
        let sync_high = bits[9];
        if sync_low != 0xFD || sync_high != 0x3F {
            return Err(TimecodeError::ltc(format!(
                "Invalid sync word: {:02X}{:02X}",
                sync_high, sync_low
            )));
        }

        Ok(Self {
            frame_units: get_bits(bits, 0, 4),
            frame_tens: get_bits(bits, 8, 2),
            second_units: get_bits(bits, 16, 4),
            second_tens: get_bits(bits, 24, 3),
            minute_units: get_bits(bits, 32, 4),
            minute_tens: get_bits(bits, 40, 3),
            hour_units: get_bits(bits, 48, 4),
            hour_tens: get_bits(bits, 56, 2),
            drop_frame: get_bits(bits, 10, 1) == 1,
            color_frame: get_bits(bits, 11, 1) == 1,
            binary_group_flags: get_bits(bits, 27, 1) | (get_bits(bits, 43, 1) << 1),
            user_bits: [
                get_bits(bits, 4, 4),
                get_bits(bits, 12, 4),
                get_bits(bits, 20, 4),
                get_bits(bits, 28, 4),
                get_bits(bits, 36, 4),
                get_bits(bits, 44, 4),
                get_bits(bits, 52, 4),
                get_bits(bits, 60, 4),
            ],
            polarity_correction: get_bits(bits, 59, 1) == 1,
        })
    }
}


/// LTC audio encoder.
///
/// Generates audio samples for LTC signals using bi-phase modulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtcEncoder {
    /// Sample rate in Hz
    sample_rate: u32,
    /// Frame rate
    frame_rate: FrameRate,
    /// Samples per bit
    samples_per_bit: f64,
    /// Current phase (for continuous encoding)
    phase: bool,
    /// Current sample position within bit
    sample_position: f64,
    /// Current bit position within frame
    bit_position: usize,
    /// Current frame data
    current_frame: [u8; 10],
    /// Amplitude (0.0 to 1.0)
    amplitude: f32,
}

impl LtcEncoder {
    /// Create a new LTC encoder.
    pub fn new(sample_rate: u32, frame_rate: FrameRate) -> Self {
        let fps = frame_rate.as_f64();
        let bits_per_second = fps * LTC_FRAME_BITS as f64;
        let samples_per_bit = sample_rate as f64 / bits_per_second;

        Self {
            sample_rate,
            frame_rate,
            samples_per_bit,
            phase: false,
            sample_position: 0.0,
            bit_position: 0,
            current_frame: [0; 10],
            amplitude: 0.5,
        }
    }

    /// Set the amplitude (0.0 to 1.0).
    pub fn set_amplitude(&mut self, amplitude: f32) {
        self.amplitude = amplitude.clamp(0.0, 1.0);
    }

    /// Encode a frame and return audio samples.
    ///
    /// Returns f32 samples in the range [-amplitude, +amplitude].
    pub fn encode_frame(&mut self, frame: &LtcFrame) -> Vec<f32> {
        self.current_frame = frame.encode();
        self.bit_position = 0;
        self.sample_position = 0.0;

        let samples_per_frame = (self.samples_per_bit * LTC_FRAME_BITS as f64).ceil() as usize;
        let mut samples = Vec::with_capacity(samples_per_frame);

        for _ in 0..samples_per_frame {
            samples.push(self.next_sample());
        }

        samples
    }

    /// Generate the next audio sample.
    fn next_sample(&mut self) -> f32 {
        if self.bit_position >= LTC_FRAME_BITS {
            return if self.phase {
                self.amplitude
            } else {
                -self.amplitude
            };
        }

        // Get current bit value
        let byte_idx = self.bit_position / 8;
        let bit_idx = self.bit_position % 8;
        let bit_value = (self.current_frame[byte_idx] >> bit_idx) & 1;

        // Bi-phase modulation:
        // - Always transition at start of bit period
        // - For '1' bit, also transition at middle
        let half_bit = self.samples_per_bit / 2.0;

        if self.sample_position < 0.5 {
            // Start of bit - always transition
            self.phase = !self.phase;
        }

        // Middle of bit - transition only for '1'
        if bit_value == 1
            && self.sample_position >= half_bit
            && self.sample_position < half_bit + 1.0
        {
            self.phase = !self.phase;
        }

        self.sample_position += 1.0;

        // Move to next bit
        if self.sample_position >= self.samples_per_bit {
            self.sample_position -= self.samples_per_bit;
            self.bit_position += 1;
        }

        if self.phase {
            self.amplitude
        } else {
            -self.amplitude
        }
    }

    /// Get the sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get the frame rate.
    #[must_use]
    pub fn frame_rate(&self) -> FrameRate {
        self.frame_rate
    }
}

/// LTC audio decoder.
///
/// Decodes LTC signals from audio samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtcDecoder {
    /// Sample rate in Hz
    sample_rate: u32,
    /// Frame rate (for validation)
    frame_rate: FrameRate,
    /// Samples per bit (approximate)
    samples_per_bit: f64,
    /// Sample buffer for edge detection
    sample_buffer: Vec<f32>,
    /// Detected edges (sample positions)
    edges: Vec<usize>,
    /// Bit buffer
    bit_buffer: Vec<u8>,
    /// Threshold for edge detection
    threshold: f32,
    /// Last sample sign
    last_sign: bool,
}

impl LtcDecoder {
    /// Create a new LTC decoder.
    pub fn new(sample_rate: u32, frame_rate: FrameRate) -> Self {
        let fps = frame_rate.as_f64();
        let bits_per_second = fps * LTC_FRAME_BITS as f64;
        let samples_per_bit = sample_rate as f64 / bits_per_second;

        Self {
            sample_rate,
            frame_rate,
            samples_per_bit,
            sample_buffer: Vec::new(),
            edges: Vec::new(),
            bit_buffer: Vec::new(),
            threshold: 0.1,
            last_sign: false,
        }
    }

    /// Set the detection threshold.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.abs();
    }

    /// Process audio samples and detect frames.
    ///
    /// Returns any complete frames detected.
    pub fn decode(&mut self, samples: &[f32]) -> Vec<LtcFrame> {
        let mut frames = Vec::new();

        for (i, &sample) in samples.iter().enumerate() {
            let sign = sample > 0.0;

            // Detect zero crossings (edges)
            if sign != self.last_sign && sample.abs() > self.threshold {
                self.edges.push(self.sample_buffer.len() + i);
            }
            self.last_sign = sign;
        }

        self.sample_buffer.extend_from_slice(samples);

        // Try to decode bits from edges
        self.decode_bits();

        // Try to find complete frames
        while let Some(frame) = self.try_extract_frame() {
            frames.push(frame);
        }

        // Trim old samples to prevent unbounded growth
        if self.sample_buffer.len() > self.sample_rate as usize * 2 {
            let trim_amount = self.sample_rate as usize;
            self.sample_buffer.drain(0..trim_amount);
            for edge in &mut self.edges {
                *edge = edge.saturating_sub(trim_amount);
            }
            self.edges.retain(|&e| e > 0);
        }

        frames
    }

    /// Decode bits from detected edges.
    fn decode_bits(&mut self) {
        if self.edges.len() < 2 {
            return;
        }

        let mut i = 0;
        while i + 1 < self.edges.len() {
            let interval = self.edges[i + 1] - self.edges[i];
            let expected_half_bit = self.samples_per_bit / 2.0;
            let expected_full_bit = self.samples_per_bit;

            // Determine if this is a '0' or '1' bit
            if (interval as f64) < expected_half_bit * 1.5 {
                // Short interval - this is half a bit, look for next edge
                if i + 2 < self.edges.len() {
                    let next_interval = self.edges[i + 2] - self.edges[i + 1];
                    if (next_interval as f64) < expected_half_bit * 1.5 {
                        // Two short intervals = '1' bit
                        self.bit_buffer.push(1);
                        i += 2;
                    } else {
                        i += 1;
                    }
                } else {
                    break;
                }
            } else if (interval as f64) < expected_full_bit * 1.5 {
                // Full bit interval = '0' bit
                self.bit_buffer.push(0);
                i += 1;
            } else {
                // Invalid interval, skip
                i += 1;
            }
        }

        // Remove processed edges
        if i > 0 && i < self.edges.len() {
            self.edges.drain(0..i);
        }
    }

    /// Try to extract a complete frame from the bit buffer.
    fn try_extract_frame(&mut self) -> Option<LtcFrame> {
        if self.bit_buffer.len() < LTC_FRAME_BITS {
            return None;
        }

        // Look for sync word pattern
        for start in 0..=(self.bit_buffer.len() - LTC_FRAME_BITS) {
            // Check for sync word at bits 64-79
            let sync_start = start + 64;
            if sync_start + 16 > self.bit_buffer.len() {
                continue;
            }

            // Pack bits 64-79 into sync word
            let mut sync_word = 0u16;
            for i in 0..16 {
                if self.bit_buffer[sync_start + i] == 1 {
                    sync_word |= 1 << i;
                }
            }

            if sync_word == LTC_SYNC_WORD {
                // Found sync word, extract frame
                let mut frame_bits = [0u8; 10];
                for i in 0..80 {
                    if self.bit_buffer[start + i] == 1 {
                        let byte_idx = i / 8;
                        let bit_idx = i % 8;
                        frame_bits[byte_idx] |= 1 << bit_idx;
                    }
                }

                // Remove used bits
                self.bit_buffer.drain(0..(start + LTC_FRAME_BITS));

                return LtcFrame::decode(&frame_bits).ok();
            }
        }

        // No frame found, trim buffer if too long
        if self.bit_buffer.len() > LTC_FRAME_BITS * 3 {
            self.bit_buffer.drain(0..LTC_FRAME_BITS);
        }

        None
    }

    /// Reset the decoder state.
    pub fn reset(&mut self) {
        self.sample_buffer.clear();
        self.edges.clear();
        self.bit_buffer.clear();
        self.last_sign = false;
    }

    /// Get the sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get the frame rate.
    #[must_use]
    pub fn frame_rate(&self) -> FrameRate {
        self.frame_rate
    }
}

/// Generate a test LTC signal for a range of timecodes.
pub fn generate_ltc_signal(start_tc: &Timecode, num_frames: u32, sample_rate: u32) -> Vec<f32> {
    let mut encoder = LtcEncoder::new(sample_rate, start_tc.frame_rate);
    let mut samples = Vec::new();

    let mut tc = *start_tc;
    for _ in 0..num_frames {
        let frame = LtcFrame::from_timecode(&tc);
        samples.extend(encoder.encode_frame(&frame));
        tc = tc.add_frames(1).unwrap_or(tc);
    }

    samples
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_ltc_frame_from_timecode() {
        let tc = Timecode::new(1, 30, 45, 12, FrameRate::Fps24).unwrap();
        let frame = LtcFrame::from_timecode(&tc);

        assert_eq!(frame.hour_tens, 0);
        assert_eq!(frame.hour_units, 1);
        assert_eq!(frame.minute_tens, 3);
        assert_eq!(frame.minute_units, 0);
        assert_eq!(frame.second_tens, 4);
        assert_eq!(frame.second_units, 5);
        assert_eq!(frame.frame_tens, 1);
        assert_eq!(frame.frame_units, 2);
    }

    #[test]
    fn test_ltc_frame_to_timecode() {
        let frame = LtcFrame {
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

        let tc = frame.to_timecode(FrameRate::Fps24);
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 30);
        assert_eq!(tc.seconds, 45);
        assert_eq!(tc.frames, 12);
    }

    #[test]
    fn test_ltc_frame_encode_decode() {
        let original = LtcFrame {
            hour_tens: 1,
            hour_units: 2,
            minute_tens: 3,
            minute_units: 4,
            second_tens: 5,
            second_units: 6,
            frame_tens: 0,
            frame_units: 7,
            drop_frame: true,
            color_frame: false,
            binary_group_flags: 2,
            user_bits: [1, 2, 3, 4, 5, 6, 7, 8],
            polarity_correction: true,
        };

        let encoded = original.encode();
        let decoded = LtcFrame::decode(&encoded).unwrap();

        assert_eq!(decoded.hour_tens, original.hour_tens);
        assert_eq!(decoded.hour_units, original.hour_units);
        assert_eq!(decoded.minute_tens, original.minute_tens);
        assert_eq!(decoded.minute_units, original.minute_units);
        assert_eq!(decoded.second_tens, original.second_tens);
        assert_eq!(decoded.second_units, original.second_units);
        assert_eq!(decoded.frame_tens, original.frame_tens);
        assert_eq!(decoded.frame_units, original.frame_units);
        assert_eq!(decoded.drop_frame, original.drop_frame);
        assert_eq!(decoded.user_bits, original.user_bits);
    }

    #[test]
    fn test_ltc_frame_user_bits() {
        let mut frame = LtcFrame::default();
        frame.set_user_bits_u32(0x12345678);
        assert_eq!(frame.get_user_bits_u32(), 0x12345678);
    }

    #[test]
    fn test_ltc_encoder_creates_samples() {
        let mut encoder = LtcEncoder::new(48000, FrameRate::Fps24);
        let frame = LtcFrame::default();

        let samples = encoder.encode_frame(&frame);

        // Should have approximately 48000 / 24 = 2000 samples per frame
        assert!(samples.len() > 1800);
        assert!(samples.len() < 2200);

        // All samples should be within amplitude range
        for &s in &samples {
            assert!(s.abs() <= encoder.amplitude);
        }
    }

    #[test]
    fn test_ltc_signal_generation() {
        let tc = Timecode::new(0, 0, 0, 0, FrameRate::Fps24).unwrap();
        let samples = generate_ltc_signal(&tc, 3, 48000);

        // Should have approximately 3 frames worth of samples
        let expected = (48000.0 / 24.0 * 3.0) as usize;
        assert!(samples.len() >= expected - 100);
        assert!(samples.len() <= expected + 100);
    }

    #[test]
    fn test_ltc_frame_serialization() {
        let frame =
            LtcFrame::from_timecode(&Timecode::new(1, 30, 45, 12, FrameRate::Fps24).unwrap());
        let json = serde_json::to_string(&frame).unwrap();
        let decoded: LtcFrame = serde_json::from_str(&json).unwrap();
        assert_eq!(frame, decoded);
    }

    #[test]
    fn test_ltc_sync_word_validation() {
        let mut bad_frame = [0u8; 10];
        // Wrong sync word
        bad_frame[8] = 0x00;
        bad_frame[9] = 0x00;

        assert!(LtcFrame::decode(&bad_frame).is_err());
    }

    #[test]
    fn test_ltc_decoder_new() {
        let decoder = LtcDecoder::new(48000, FrameRate::Fps24);
        assert_eq!(decoder.sample_rate(), 48000);
        assert_eq!(decoder.frame_rate(), FrameRate::Fps24);
    }
}
