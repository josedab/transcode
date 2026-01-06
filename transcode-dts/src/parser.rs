//! DTS and TrueHD bitstream parsers.
//!
//! Parses DTS core streams, DTS-HD extensions, and TrueHD audio frames.

use crate::types::*;
use crate::{DTS_SYNC_WORD_BE, DTS_HD_SYNC, TRUEHD_SYNC, MLP_SYNC};

/// DTS sample rate lookup table.
const DTS_SAMPLE_RATES: [u32; 16] = [
    0,     // Invalid
    8000,  // 1
    16000, // 2
    32000, // 3
    0,     // 4 - Invalid
    0,     // 5 - Invalid
    11025, // 6
    22050, // 7
    44100, // 8
    0,     // 9 - Invalid
    0,     // 10 - Invalid
    12000, // 11
    24000, // 12
    48000, // 13
    96000, // 14 - DTS-HD
    192000, // 15 - DTS-HD
];

/// DTS bit rate lookup table (in kbps).
const DTS_BIT_RATES: [u32; 32] = [
    32, 56, 64, 96, 112, 128, 192, 224,
    256, 320, 384, 448, 512, 576, 640, 768,
    896, 1024, 1152, 1280, 1344, 1408, 1411, 1472,
    1536, 1920, 2048, 3072, 3840, 0, 0, 0, // Last three: open, variable, lossless
];

/// DTS audio parser.
///
/// Parses DTS core and DTS-HD sync frames and extracts audio parameters.
#[derive(Debug)]
pub struct DtsParser {
    /// Last parsed sync frame.
    last_frame: Option<DtsSyncFrame>,
    /// Last parsed HD extension.
    last_hd_frame: Option<DtsHdFrame>,
    /// Total bytes parsed.
    bytes_parsed: u64,
    /// Frame count.
    frame_count: u64,
}

impl DtsParser {
    /// Create a new DTS parser.
    pub fn new() -> Self {
        Self {
            last_frame: None,
            last_hd_frame: None,
            bytes_parsed: 0,
            frame_count: 0,
        }
    }

    /// Reset the parser state.
    pub fn reset(&mut self) {
        self.last_frame = None;
        self.last_hd_frame = None;
    }

    /// Find DTS sync word in data.
    pub fn find_sync(&self, data: &[u8]) -> Option<usize> {
        if data.len() < 4 {
            return None;
        }

        (0..data.len().saturating_sub(3)).find(|&i| {
            let sync = u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
            sync == DTS_SYNC_WORD_BE || sync == DTS_HD_SYNC
        })
    }

    /// Parse a DTS core sync frame.
    pub fn parse_sync_frame(&mut self, data: &[u8]) -> Option<DtsSyncFrame> {
        if data.len() < 12 {
            return None;
        }

        // Check sync word (big endian 14-bit format)
        let sync = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        if sync != DTS_SYNC_WORD_BE {
            return None;
        }

        let mut reader = BitReader::new(&data[4..]);

        // FTYPE - Frame type (1 bit)
        let ftype = reader.read_bit()?;
        if ftype != 1 {
            // Must be 1 for normal frame
            return None;
        }

        // SHORT - Deficit sample count (5 bits)
        let _short = reader.read_bits(5)?;

        // CPF - CRC present flag (1 bit)
        let crc_present = reader.read_bit()? == 1;

        // NBLKS - Number of PCM sample blocks (7 bits)
        let nblks = reader.read_bits(7)? as u8;
        let samples_per_channel = ((nblks as usize) + 1) * 32;

        // FSIZE - Primary frame byte size (14 bits)
        let fsize = reader.read_bits(14)? as u16;
        let frame_size = (fsize as usize) + 1;

        // AMODE - Audio channel arrangement (6 bits)
        let amode_code = reader.read_bits(6)? as u8;
        let amode = AudioMode::from_code(amode_code);
        let channels = amode.channel_count();

        // SFREQ - Core audio sampling frequency (4 bits)
        let sfreq = reader.read_bits(4)? as u8;
        let sample_rate = DTS_SAMPLE_RATES.get(sfreq as usize).copied().unwrap_or(48000);

        // RATE - Transmission bit rate (5 bits)
        let rate = reader.read_bits(5)? as u8;
        let bit_rate = DTS_BIT_RATES.get(rate as usize).copied().unwrap_or(768);

        // Skip some less important fields and read key ones
        let _ = reader.read_bit(); // DYNF
        let _ = reader.read_bit(); // TIMEF
        let _ = reader.read_bit(); // AUXF
        let _ = reader.read_bit(); // HDCD
        let ext_audio_id_code = reader.read_bits(3)? as u8;
        let ext_audio = reader.read_bit()? == 1;
        let _ = reader.read_bit(); // ASPF

        // LFF - Low frequency effects flag (2 bits)
        let lff = reader.read_bits(2)? as u8;
        let lfe = lff == 1 || lff == 2;

        // Skip remaining header fields
        let _ = reader.read_bit(); // HFLAG
        let _ = reader.read_bits(2); // CRC type
        let _ = reader.read_bit(); // FILTS
        let _ = reader.read_bits(4); // VERNUM
        let _ = reader.read_bits(2); // CHIST

        // PCM resolution
        let pcm_res = reader.read_bits(3)? as u8;
        let pcm_resolution = match pcm_res {
            0 | 1 => 16,
            2 | 3 => 20,
            4 | 5 => 24,
            _ => 16,
        };

        let ext_audio_id = ExtensionAudioId::from_code(ext_audio_id_code);

        let frame = DtsSyncFrame {
            frame_size,
            amode,
            sample_rate,
            bit_rate,
            samples_per_channel,
            channels,
            lfe,
            pcm_resolution,
            crc_present,
            nblks,
            fsize,
            sfreq,
            rate,
            dynf: false,
            timef: false,
            auxf: false,
            hdcd: false,
            ext_audio_id,
            ext_audio,
            aspf: false,
            lff,
            hflag: false,
            crc_type: 0,
            filts: false,
            vernum: 0,
            chist: 0,
            dialog_norm: 0,
        };

        self.last_frame = Some(frame.clone());
        self.bytes_parsed += frame_size as u64;
        self.frame_count += 1;

        Some(frame)
    }

    /// Parse DTS-HD extension header.
    pub fn parse_hd_extension(&mut self, data: &[u8]) -> Option<DtsHdFrame> {
        if data.len() < 8 {
            return None;
        }

        // Check DTS-HD sync word
        let sync = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        if sync != DTS_HD_SYNC {
            return None;
        }

        let mut reader = BitReader::new(&data[4..]);

        // Parse DTS-HD header (simplified)
        let _user_defined = reader.read_bits(8)?;
        let num_assets = reader.read_bits(3)? as u8 + 1;
        let num_presentations = reader.read_bits(2)? as u8 + 1;

        // Frame byte size
        let size_field = reader.read_bits(16)? as usize;
        let frame_size = size_field + 1;

        // Determine profile from extension data (simplified heuristic)
        let lossless = true; // Would need deeper parsing for accurate detection
        let profile = if lossless {
            DtsHdProfile::MasterAudio
        } else {
            DtsHdProfile::HighResolution
        };

        let hd_frame = DtsHdFrame {
            extension_type: DtsHdExtension::HdMasterAudio,
            frame_size,
            num_assets,
            num_presentations,
            sample_rate: 48000, // Would need core frame to determine
            bit_depth: 24,
            channels: 6,
            lossless,
            profile,
            dtsx: false,
            num_samples: 4096,
        };

        self.last_hd_frame = Some(hd_frame.clone());

        Some(hd_frame)
    }

    /// Get the last parsed sync frame.
    pub fn last_frame(&self) -> Option<&DtsSyncFrame> {
        self.last_frame.as_ref()
    }

    /// Get the last parsed HD frame.
    pub fn last_hd_frame(&self) -> Option<&DtsHdFrame> {
        self.last_hd_frame.as_ref()
    }

    /// Get total bytes parsed.
    pub fn bytes_parsed(&self) -> u64 {
        self.bytes_parsed
    }

    /// Get frame count.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

impl Default for DtsParser {
    fn default() -> Self {
        Self::new()
    }
}

/// TrueHD audio parser.
///
/// Parses TrueHD (MLP) sync frames and extracts audio parameters.
#[derive(Debug)]
pub struct TrueHdParser {
    /// Last parsed sync frame.
    last_frame: Option<TrueHdSyncFrame>,
    /// Total bytes parsed.
    bytes_parsed: u64,
    /// Frame count.
    frame_count: u64,
}

impl TrueHdParser {
    /// Create a new TrueHD parser.
    pub fn new() -> Self {
        Self {
            last_frame: None,
            bytes_parsed: 0,
            frame_count: 0,
        }
    }

    /// Reset the parser state.
    pub fn reset(&mut self) {
        self.last_frame = None;
    }

    /// Find TrueHD sync word in data.
    pub fn find_sync(&self, data: &[u8]) -> Option<usize> {
        if data.len() < 4 {
            return None;
        }

        (0..data.len().saturating_sub(3)).find(|&i| {
            let sync = u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
            sync == TRUEHD_SYNC || sync == MLP_SYNC
        })
    }

    /// Parse a TrueHD major sync frame.
    pub fn parse_sync_frame(&mut self, data: &[u8]) -> Option<TrueHdSyncFrame> {
        if data.len() < 28 {
            return None;
        }

        // TrueHD frames start with access unit header, then major sync
        // The major sync word appears at a specific offset
        let sync_offset = self.find_sync(data)?;
        let sync_data = &data[sync_offset..];

        if sync_data.len() < 24 {
            return None;
        }

        // Skip sync word
        let mut reader = BitReader::new(&sync_data[4..]);

        // Parse major sync header (simplified)
        let format_info = reader.read_bits(8)?;
        let _signature = reader.read_bits(16)?;
        let flags = reader.read_bits(16)?;

        // Extract sample rate from format_info
        let sample_rate_idx = (format_info >> 4) & 0x0F;
        let sample_rate = match sample_rate_idx {
            0 => 48000,
            1 => 96000,
            2 => 192000,
            8 => 44100,
            9 => 88200,
            10 => 176400,
            _ => 48000,
        };

        // Extract bit depth
        let bit_depth_idx = format_info & 0x0F;
        let bit_depth = match bit_depth_idx {
            0 => 16,
            1 => 20,
            2 => 24,
            _ => 24,
        };

        // Channel count from flags (simplified)
        let channel_idx = (flags >> 8) & 0x1F;
        let channels: u8 = match channel_idx {
            0 => 1,
            1 => 2,
            2 => 3,
            3 => 4,
            4 => 5,
            5 => 6,
            6 => 7,
            7 => 8,
            _ => (channel_idx as u8 + 1).min(16),
        };

        // Determine if Atmos
        let atmos = channels >= 8 && (flags & 0x0001) != 0;

        let channel_assignment = if atmos {
            TrueHdChannelAssignment::Atmos
        } else {
            match channels {
                1 => TrueHdChannelAssignment::Mono,
                2 => TrueHdChannelAssignment::Stereo,
                3 => TrueHdChannelAssignment::Layout3_0,
                4 => TrueHdChannelAssignment::Layout4_0,
                5 => TrueHdChannelAssignment::Layout5_0,
                6 => TrueHdChannelAssignment::Layout5_1,
                8 => TrueHdChannelAssignment::Layout7_1,
                _ => TrueHdChannelAssignment::Custom(channels),
            }
        };

        // Frame size calculation (simplified)
        let access_unit_size = reader.read_bits(16)? as u16;
        let frame_size = (access_unit_size as usize * 2).max(24);

        let frame = TrueHdSyncFrame {
            frame_size,
            sample_rate,
            channels,
            bit_depth,
            samples_per_channel: 40, // TrueHD uses 40 samples per frame at 48kHz
            group_size: 1,
            access_unit_size,
            peak_bitrate: 0,
            channel_assignment,
            atmos,
            substream_count: 1,
            crc_present: true,
            major_sync_interval: 0,
        };

        self.last_frame = Some(frame.clone());
        self.bytes_parsed += frame_size as u64;
        self.frame_count += 1;

        Some(frame)
    }

    /// Get the last parsed sync frame.
    pub fn last_frame(&self) -> Option<&TrueHdSyncFrame> {
        self.last_frame.as_ref()
    }

    /// Get total bytes parsed.
    pub fn bytes_parsed(&self) -> u64 {
        self.bytes_parsed
    }

    /// Get frame count.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

impl Default for TrueHdParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Detect audio format from data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    /// DTS core stream.
    DtsCore,
    /// DTS-HD stream.
    DtsHd,
    /// TrueHD/MLP stream.
    TrueHd,
    /// Unknown format.
    Unknown,
}

/// Detect the audio format from data.
pub fn detect_format(data: &[u8]) -> AudioFormat {
    if data.len() < 4 {
        return AudioFormat::Unknown;
    }

    let sync = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);

    match sync {
        DTS_SYNC_WORD_BE => AudioFormat::DtsCore,
        DTS_HD_SYNC => AudioFormat::DtsHd,
        TRUEHD_SYNC | MLP_SYNC => AudioFormat::TrueHd,
        _ => {
            // Search for sync words in first 1KB
            let parser = DtsParser::new();
            let truehd_parser = TrueHdParser::new();

            if parser.find_sync(data).is_some() {
                AudioFormat::DtsCore
            } else if truehd_parser.find_sync(data).is_some() {
                AudioFormat::TrueHd
            } else {
                AudioFormat::Unknown
            }
        }
    }
}

/// Simple bit reader for parsing.
#[derive(Debug)]
struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    fn read_bit(&mut self) -> Option<u8> {
        if self.byte_pos >= self.data.len() {
            return None;
        }

        let bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;
        if self.bit_pos >= 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }

        Some(bit)
    }

    fn read_bits(&mut self, count: u8) -> Option<u32> {
        let mut value = 0u32;
        for _ in 0..count {
            value = (value << 1) | self.read_bit()? as u32;
        }
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dts_parser_new() {
        let parser = DtsParser::new();
        assert!(parser.last_frame().is_none());
        assert_eq!(parser.bytes_parsed(), 0);
    }

    #[test]
    fn test_truehd_parser_new() {
        let parser = TrueHdParser::new();
        assert!(parser.last_frame().is_none());
        assert_eq!(parser.bytes_parsed(), 0);
    }

    #[test]
    fn test_dts_find_sync() {
        let parser = DtsParser::new();

        // Valid DTS sync word (big endian)
        let data = [0x7F, 0xFE, 0x80, 0x01, 0x00, 0x00];
        assert_eq!(parser.find_sync(&data), Some(0));

        // Sync word not at beginning
        let data = [0x00, 0x7F, 0xFE, 0x80, 0x01, 0x00];
        assert_eq!(parser.find_sync(&data), Some(1));

        // No sync word
        let data = [0x00, 0x00, 0x00, 0x00];
        assert_eq!(parser.find_sync(&data), None);
    }

    #[test]
    fn test_truehd_find_sync() {
        let parser = TrueHdParser::new();

        // Valid TrueHD sync word
        let data = [0xF8, 0x72, 0x6F, 0xBA, 0x00, 0x00];
        assert_eq!(parser.find_sync(&data), Some(0));

        // MLP sync word
        let data = [0xF8, 0x72, 0x6F, 0xBB, 0x00, 0x00];
        assert_eq!(parser.find_sync(&data), Some(0));
    }

    #[test]
    fn test_detect_format() {
        // DTS core
        let data = [0x7F, 0xFE, 0x80, 0x01];
        assert_eq!(detect_format(&data), AudioFormat::DtsCore);

        // TrueHD
        let data = [0xF8, 0x72, 0x6F, 0xBA];
        assert_eq!(detect_format(&data), AudioFormat::TrueHd);

        // DTS-HD
        let data = [0x64, 0x58, 0x20, 0x25];
        assert_eq!(detect_format(&data), AudioFormat::DtsHd);

        // Unknown
        let data = [0x00, 0x00, 0x00, 0x00];
        assert_eq!(detect_format(&data), AudioFormat::Unknown);
    }

    #[test]
    fn test_sample_rate_table() {
        assert_eq!(DTS_SAMPLE_RATES[1], 8000);
        assert_eq!(DTS_SAMPLE_RATES[13], 48000);
        assert_eq!(DTS_SAMPLE_RATES[14], 96000);
    }

    #[test]
    fn test_bit_rate_table() {
        assert_eq!(DTS_BIT_RATES[0], 32);
        assert_eq!(DTS_BIT_RATES[15], 768);
        assert_eq!(DTS_BIT_RATES[16], 896);
    }

    #[test]
    fn test_parser_reset() {
        let mut parser = DtsParser::new();
        parser.reset();
        assert!(parser.last_frame().is_none());
        assert!(parser.last_hd_frame().is_none());
    }
}
