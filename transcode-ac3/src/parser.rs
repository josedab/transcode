//! AC-3 and E-AC-3 bitstream parsers.
//!
//! Provides sync frame detection and header parsing for AC-3 and E-AC-3 bitstreams.

use crate::types::*;
use crate::{Ac3Error, Result, AC3_SYNC_WORD};

/// AC-3 sample rate table (indexed by fscod).
const AC3_SAMPLE_RATES: [u32; 3] = [48000, 44100, 32000];

/// AC-3 bitrate table (indexed by frmsizecod / 2).
const AC3_BITRATES: [u32; 19] = [
    32000, 40000, 48000, 56000, 64000, 80000, 96000, 112000, 128000, 160000, 192000, 224000,
    256000, 320000, 384000, 448000, 512000, 576000, 640000,
];

/// AC-3 frame size table (48kHz, indexed by frmsizecod / 2).
const AC3_FRAME_SIZES_48: [u16; 19] = [
    64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1152, 1280,
];

/// AC-3 frame size table (44.1kHz, indexed by frmsizecod / 2).
const AC3_FRAME_SIZES_44: [u16; 19] = [
    69, 87, 104, 121, 139, 174, 208, 243, 278, 348, 417, 487, 557, 696, 835, 975, 1114, 1253, 1393,
];

/// AC-3 frame size table (32kHz, indexed by frmsizecod / 2).
const AC3_FRAME_SIZES_32: [u16; 19] = [
    96, 120, 144, 168, 192, 240, 288, 336, 384, 480, 576, 672, 768, 960, 1152, 1344, 1536, 1728,
    1920,
];

/// E-AC-3 sample rates for fscod2.
const EAC3_SAMPLE_RATES_2: [u32; 3] = [24000, 22050, 16000];

/// AC-3 bitstream parser.
#[derive(Debug, Clone)]
pub struct Ac3Parser {
    /// Buffer for partial sync frame.
    buffer: Vec<u8>,
    /// Current parsing position.
    position: usize,
}

impl Ac3Parser {
    /// Create a new AC-3 parser.
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            position: 0,
        }
    }

    /// Parse a sync frame from the given data.
    ///
    /// Returns the parsed sync frame if successful, or None if more data is needed.
    pub fn parse_sync_frame(&mut self, data: &[u8]) -> Option<Ac3SyncFrame> {
        self.buffer.extend_from_slice(data);

        // Need at least 7 bytes for header (sync + BSI)
        if self.buffer.len() < 7 {
            return None;
        }

        // Find sync word
        while self.position + 2 <= self.buffer.len() {
            let sync = u16::from_be_bytes([self.buffer[self.position], self.buffer[self.position + 1]]);
            if sync == AC3_SYNC_WORD {
                break;
            }
            self.position += 1;
        }

        if self.position + 7 > self.buffer.len() {
            return None;
        }

        match self.parse_header() {
            Ok(frame) => {
                // Consume parsed data
                let consumed = self.position + frame.frame_size;
                if consumed <= self.buffer.len() {
                    self.buffer.drain(..consumed);
                    self.position = 0;
                    Some(frame)
                } else {
                    None
                }
            }
            Err(_) => {
                // Skip invalid sync word
                self.position += 1;
                None
            }
        }
    }

    /// Parse the sync info and BSI headers.
    fn parse_header(&self) -> Result<Ac3SyncFrame> {
        let data = &self.buffer[self.position..];

        if data.len() < 7 {
            return Err(Ac3Error::InsufficientData {
                needed: 7,
                available: data.len(),
            });
        }

        // Verify sync word
        let sync_word = u16::from_be_bytes([data[0], data[1]]);
        if sync_word != AC3_SYNC_WORD {
            return Err(Ac3Error::InvalidSyncWord(sync_word));
        }

        // CRC1
        let crc1 = u16::from_be_bytes([data[2], data[3]]);

        // Sample rate code (fscod) and frame size code (frmsizecod)
        let fscod = (data[4] >> 6) & 0x03;
        let frmsizecod = data[4] & 0x3F;

        // Validate fscod
        if fscod >= 3 {
            return Err(Ac3Error::UnsupportedSampleRate(fscod));
        }

        // Validate frmsizecod
        if frmsizecod >= 38 {
            return Err(Ac3Error::UnsupportedFrameSize(frmsizecod));
        }

        let sample_rate = AC3_SAMPLE_RATES[fscod as usize];
        let bitrate_index = (frmsizecod / 2) as usize;
        let bitrate = AC3_BITRATES[bitrate_index];

        // Frame size depends on sample rate
        let frame_size = match fscod {
            0 => AC3_FRAME_SIZES_48[bitrate_index] as usize * 2,
            1 => {
                let base = AC3_FRAME_SIZES_44[bitrate_index] as usize * 2;
                if frmsizecod & 1 == 1 {
                    base + 2
                } else {
                    base
                }
            }
            2 => AC3_FRAME_SIZES_32[bitrate_index] as usize * 2,
            _ => unreachable!(),
        };

        // BSI - Bitstream Information
        let bsid = (data[5] >> 3) & 0x1F;
        let bsmod = data[5] & 0x07;

        // Validate bsid (8 for AC-3, 6-8 for older versions)
        if bsid > 8 && bsid < 16 {
            return Err(Ac3Error::InvalidBsid(bsid));
        }

        let acmod = (data[6] >> 5) & 0x07;
        let acmod_enum = AudioCodingMode::from_value(acmod).ok_or(Ac3Error::InvalidAcmod(acmod))?;

        // Determine if LFE and other fields follow
        // This requires bit-level parsing
        let (lfe_on, dialnorm) = self.parse_bsi_details(data, acmod_enum)?;

        let channels = acmod_enum.num_channels() + if lfe_on { 1 } else { 0 };

        Ok(Ac3SyncFrame {
            frame_size,
            sample_rate,
            bitrate,
            bsid,
            acmod: acmod_enum,
            lfe_on,
            channels,
            dialnorm,
            compre: false,
            compr: None,
            langcode: false,
            langcod: None,
            audprodie: false,
            mixlevel: None,
            roomtyp: None,
            fscod,
            frmsizecod,
            bsmod: BitstreamMode::from_value(bsmod).unwrap_or(BitstreamMode::CompleteMain),
            crc1,
            crc2: 0,
        })
    }

    /// Parse additional BSI details (LFE, dialnorm, etc.).
    fn parse_bsi_details(&self, data: &[u8], acmod: AudioCodingMode) -> Result<(bool, i8)> {
        if data.len() < 8 {
            return Ok((false, -31));
        }

        // Start after acmod (bit 21)
        let mut bit_pos = 24; // After sync, crc1, fscod/frmsizecod, bsid/bsmod
        bit_pos += 3; // acmod

        // Center mix level (if center channel exists)
        if acmod.has_center() {
            bit_pos += 2;
        }

        // Surround mix level (if surround channels exist)
        if acmod.has_surround() {
            bit_pos += 2;
        }

        // Dolby surround mode (for stereo)
        if acmod == AudioCodingMode::Stereo {
            bit_pos += 2;
        }

        // LFE on flag
        let byte_idx = bit_pos / 8;
        let bit_idx = 7 - (bit_pos % 8);

        if byte_idx >= data.len() {
            return Ok((false, -31));
        }

        let lfe_on = (data[byte_idx] >> bit_idx) & 1 == 1;
        bit_pos += 1;

        // Dialnorm (5 bits)
        let dialnorm_byte_idx = bit_pos / 8;
        let dialnorm_bit_offset = bit_pos % 8;

        if dialnorm_byte_idx + 1 >= data.len() {
            return Ok((lfe_on, -31));
        }

        let combined =
            ((data[dialnorm_byte_idx] as u16) << 8) | (data[dialnorm_byte_idx + 1] as u16);
        let shift = 16 - dialnorm_bit_offset - 5;
        let dialnorm_raw = ((combined >> shift) & 0x1F) as i8;
        let dialnorm = if dialnorm_raw == 0 {
            -31
        } else {
            -dialnorm_raw
        };

        Ok((lfe_on, dialnorm))
    }

    /// Reset the parser state.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.position = 0;
    }

    /// Find the next sync frame in the data.
    pub fn find_sync(&self, data: &[u8]) -> Option<usize> {
        (0..data.len().saturating_sub(1))
            .find(|&i| data[i] == 0x0B && data[i + 1] == 0x77)
    }
}

impl Default for Ac3Parser {
    fn default() -> Self {
        Self::new()
    }
}

/// E-AC-3 bitstream parser.
#[derive(Debug, Clone)]
pub struct Eac3Parser {
    /// Buffer for partial sync frame.
    buffer: Vec<u8>,
    /// Current parsing position.
    position: usize,
}

impl Eac3Parser {
    /// Create a new E-AC-3 parser.
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            position: 0,
        }
    }

    /// Parse a sync frame from the given data.
    pub fn parse_sync_frame(&mut self, data: &[u8]) -> Option<Eac3SyncFrame> {
        self.buffer.extend_from_slice(data);

        // Need at least 6 bytes for header
        if self.buffer.len() < 6 {
            return None;
        }

        // Find sync word
        while self.position + 2 <= self.buffer.len() {
            let sync = u16::from_be_bytes([self.buffer[self.position], self.buffer[self.position + 1]]);
            if sync == AC3_SYNC_WORD {
                break;
            }
            self.position += 1;
        }

        if self.position + 6 > self.buffer.len() {
            return None;
        }

        match self.parse_header() {
            Ok(frame) => {
                let consumed = self.position + frame.frame_size;
                if consumed <= self.buffer.len() {
                    self.buffer.drain(..consumed);
                    self.position = 0;
                    Some(frame)
                } else {
                    None
                }
            }
            Err(_) => {
                self.position += 1;
                None
            }
        }
    }

    /// Parse the E-AC-3 header.
    fn parse_header(&self) -> Result<Eac3SyncFrame> {
        let data = &self.buffer[self.position..];

        if data.len() < 6 {
            return Err(Ac3Error::InsufficientData {
                needed: 6,
                available: data.len(),
            });
        }

        // Verify sync word
        let sync_word = u16::from_be_bytes([data[0], data[1]]);
        if sync_word != AC3_SYNC_WORD {
            return Err(Ac3Error::InvalidSyncWord(sync_word));
        }

        // Parse stream type and substreamid
        let strmtyp = (data[2] >> 6) & 0x03;
        let substreamid = (data[2] >> 3) & 0x07;

        // Frame size (11 bits)
        let frmsiz = (((data[2] & 0x07) as u16) << 8) | (data[3] as u16);
        let frame_size = (frmsiz + 1) * 2;

        // Sample rate code
        let fscod = (data[4] >> 6) & 0x03;
        let (sample_rate, num_blocks, fscod2) = if fscod == 3 {
            // Reduced sample rate
            let fscod2 = (data[4] >> 4) & 0x03;
            if fscod2 >= 3 {
                return Err(Ac3Error::UnsupportedSampleRate(fscod2));
            }
            (EAC3_SAMPLE_RATES_2[fscod2 as usize], 6u8, Some(fscod2))
        } else {
            let numblkscod = (data[4] >> 4) & 0x03;
            let blocks = match numblkscod {
                0 => 1,
                1 => 2,
                2 => 3,
                3 => 6,
                _ => 6,
            };
            (AC3_SAMPLE_RATES[fscod as usize], blocks, None)
        };

        // Audio coding mode and LFE
        let acmod = (data[4] >> 1) & 0x07;
        let lfeon = data[4] & 0x01;

        let acmod_enum = AudioCodingMode::from_value(acmod).ok_or(Ac3Error::InvalidAcmod(acmod))?;
        let lfe_on = lfeon == 1;
        let channels = acmod_enum.num_channels() + if lfe_on { 1 } else { 0 };

        // Bitstream ID
        let bsid = (data[5] >> 3) & 0x1F;
        if !(11..=16).contains(&bsid) {
            return Err(Ac3Error::InvalidBsid(bsid));
        }

        // Calculate approximate bitrate
        let samples = num_blocks as u32 * 256;
        let bitrate = if samples > 0 {
            (frame_size as u32 * 8 * sample_rate) / samples
        } else {
            0
        };

        Ok(Eac3SyncFrame {
            frame_size: frame_size as usize,
            sample_rate,
            num_blocks,
            bsid,
            acmod: acmod_enum,
            lfe_on,
            channels,
            stream_type: Eac3StreamType::from_value(strmtyp).unwrap_or(Eac3StreamType::Independent),
            substreamid,
            dialnorm: -31, // Would need deeper parsing
            fscod,
            fscod2,
            bsmod: BitstreamMode::CompleteMain, // Would need deeper parsing
            chanmape: false,
            num_ind_sub: 1,
            num_dep_sub: None,
            chan_loc: None,
            bitrate,
        })
    }

    /// Reset the parser state.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.position = 0;
    }

    /// Detect if stream is E-AC-3 (vs AC-3).
    pub fn is_eac3(data: &[u8]) -> bool {
        if data.len() < 6 {
            return false;
        }

        // Check sync word
        if data[0] != 0x0B || data[1] != 0x77 {
            return false;
        }

        // Check bsid (should be 11-16 for E-AC-3)
        let bsid = (data[5] >> 3) & 0x1F;
        (11..=16).contains(&bsid)
    }
}

impl Default for Eac3Parser {
    fn default() -> Self {
        Self::new()
    }
}

/// Detect AC-3 vs E-AC-3 format from sync frame.
pub fn detect_format(data: &[u8]) -> Option<Ac3Format> {
    if data.len() < 6 {
        return None;
    }

    // Check sync word
    if data[0] != 0x0B || data[1] != 0x77 {
        return None;
    }

    let bsid = (data[5] >> 3) & 0x1F;
    if bsid <= 8 {
        Some(Ac3Format::Ac3)
    } else if (11..=16).contains(&bsid) {
        Some(Ac3Format::Eac3)
    } else {
        None
    }
}

/// AC-3 format type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ac3Format {
    /// Standard AC-3 (Dolby Digital).
    Ac3,
    /// Enhanced AC-3 (Dolby Digital Plus).
    Eac3,
}

impl std::fmt::Display for Ac3Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ac3Format::Ac3 => write!(f, "AC-3"),
            Ac3Format::Eac3 => write!(f, "E-AC-3"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ac3_parser_new() {
        let parser = Ac3Parser::new();
        assert!(parser.buffer.is_empty());
    }

    #[test]
    fn test_eac3_parser_new() {
        let parser = Eac3Parser::new();
        assert!(parser.buffer.is_empty());
    }

    #[test]
    fn test_find_sync() {
        let parser = Ac3Parser::new();

        // No sync word
        assert_eq!(parser.find_sync(&[0x00, 0x00, 0x00, 0x00]), None);

        // Sync at start
        assert_eq!(parser.find_sync(&[0x0B, 0x77, 0x00, 0x00]), Some(0));

        // Sync in middle
        assert_eq!(parser.find_sync(&[0x00, 0x00, 0x0B, 0x77]), Some(2));
    }

    #[test]
    fn test_detect_format() {
        // Not enough data
        assert_eq!(detect_format(&[0x0B, 0x77]), None);

        // Invalid sync
        assert_eq!(detect_format(&[0x00, 0x00, 0x00, 0x00, 0x00, 0x00]), None);

        // AC-3 (bsid = 8)
        let ac3_header = [0x0B, 0x77, 0x00, 0x00, 0x00, 0x40]; // bsid = 8
        assert_eq!(detect_format(&ac3_header), Some(Ac3Format::Ac3));

        // E-AC-3 (bsid = 16)
        let eac3_header = [0x0B, 0x77, 0x00, 0x00, 0x00, 0x80]; // bsid = 16
        assert_eq!(detect_format(&eac3_header), Some(Ac3Format::Eac3));
    }

    #[test]
    fn test_is_eac3() {
        // AC-3 frame
        let ac3_header = [0x0B, 0x77, 0x00, 0x00, 0x00, 0x40];
        assert!(!Eac3Parser::is_eac3(&ac3_header));

        // E-AC-3 frame
        let eac3_header = [0x0B, 0x77, 0x00, 0x00, 0x00, 0x80];
        assert!(Eac3Parser::is_eac3(&eac3_header));
    }

    #[test]
    fn test_ac3_sample_rates() {
        assert_eq!(AC3_SAMPLE_RATES[0], 48000);
        assert_eq!(AC3_SAMPLE_RATES[1], 44100);
        assert_eq!(AC3_SAMPLE_RATES[2], 32000);
    }

    #[test]
    fn test_ac3_bitrates() {
        assert_eq!(AC3_BITRATES[0], 32000);
        assert_eq!(AC3_BITRATES[14], 384000);
        assert_eq!(AC3_BITRATES[18], 640000);
    }

    #[test]
    fn test_parser_reset() {
        let mut parser = Ac3Parser::new();
        parser.buffer.push(0x0B);
        parser.buffer.push(0x77);
        parser.position = 1;

        parser.reset();

        assert!(parser.buffer.is_empty());
        assert_eq!(parser.position, 0);
    }
}
