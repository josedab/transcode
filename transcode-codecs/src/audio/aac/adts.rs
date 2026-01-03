//! ADTS (Audio Data Transport Stream) header parsing.

use transcode_core::error::{Error, Result};

/// ADTS header (7 or 9 bytes).
#[derive(Debug, Clone)]
pub struct AdtsHeader {
    /// MPEG version (0 = MPEG-4, 1 = MPEG-2).
    pub mpeg_version: u8,
    /// Layer (always 0 for AAC).
    pub layer: u8,
    /// CRC protection absent.
    pub protection_absent: bool,
    /// Profile (0 = Main, 1 = LC, 2 = SSR, 3 = LTP).
    pub profile: u8,
    /// Sample rate index.
    pub sample_rate_index: u8,
    /// Private bit.
    pub private_bit: bool,
    /// Channel configuration.
    pub channel_config: u8,
    /// Originality flag.
    pub original: bool,
    /// Home flag.
    pub home: bool,
    /// Copyright ID bit.
    pub copyright_id_bit: bool,
    /// Copyright ID start.
    pub copyright_id_start: bool,
    /// Frame length (including header).
    pub frame_length: u16,
    /// Buffer fullness.
    pub buffer_fullness: u16,
    /// Number of raw data blocks in frame.
    pub num_raw_data_blocks: u8,
    /// CRC (if protection_absent is false).
    pub crc: Option<u16>,
}

impl AdtsHeader {
    /// ADTS sync word.
    pub const SYNC_WORD: u16 = 0xFFF;

    /// Parse ADTS header from bytes.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 7 {
            return Err(Error::Bitstream("ADTS header too short".into()));
        }

        // Check sync word (12 bits = 0xFFF)
        if data[0] != 0xFF || (data[1] & 0xF0) != 0xF0 {
            return Err(Error::Bitstream("Invalid ADTS sync word".into()));
        }

        let mpeg_version = (data[1] >> 3) & 1;
        let layer = (data[1] >> 1) & 3;
        let protection_absent = (data[1] & 1) == 1;

        let profile = (data[2] >> 6) & 3;
        let sample_rate_index = (data[2] >> 2) & 0xF;
        let private_bit = (data[2] >> 1) & 1 == 1;
        let channel_config = ((data[2] & 1) << 2) | ((data[3] >> 6) & 3);

        let original = (data[3] >> 5) & 1 == 1;
        let home = (data[3] >> 4) & 1 == 1;

        let copyright_id_bit = (data[3] >> 3) & 1 == 1;
        let copyright_id_start = (data[3] >> 2) & 1 == 1;

        let frame_length = ((data[3] as u16 & 3) << 11) | ((data[4] as u16) << 3) | ((data[5] as u16) >> 5);

        let buffer_fullness = ((data[5] as u16 & 0x1F) << 6) | ((data[6] as u16) >> 2);
        let num_raw_data_blocks = data[6] & 3;

        let crc = if !protection_absent {
            if data.len() < 9 {
                return Err(Error::Bitstream("ADTS header with CRC too short".into()));
            }
            Some(((data[7] as u16) << 8) | data[8] as u16)
        } else {
            None
        };

        Ok(Self {
            mpeg_version,
            layer,
            protection_absent,
            profile,
            sample_rate_index,
            private_bit,
            channel_config,
            original,
            home,
            copyright_id_bit,
            copyright_id_start,
            frame_length,
            buffer_fullness,
            num_raw_data_blocks,
            crc,
        })
    }

    /// Get header size in bytes.
    pub fn header_size(&self) -> usize {
        if self.protection_absent { 7 } else { 9 }
    }

    /// Get payload size (frame length - header size).
    pub fn payload_size(&self) -> usize {
        self.frame_length as usize - self.header_size()
    }

    /// Get sample rate.
    pub fn sample_rate(&self) -> u32 {
        match self.sample_rate_index {
            0 => 96000,
            1 => 88200,
            2 => 64000,
            3 => 48000,
            4 => 44100,
            5 => 32000,
            6 => 24000,
            7 => 22050,
            8 => 16000,
            9 => 12000,
            10 => 11025,
            11 => 8000,
            12 => 7350,
            _ => 44100,
        }
    }

    /// Get number of channels.
    pub fn channels(&self) -> u8 {
        match self.channel_config {
            1 => 1,
            2 => 2,
            3 => 3,
            4 => 4,
            5 => 5,
            6 => 6,
            7 => 8,
            _ => 2,
        }
    }

    /// Encode ADTS header.
    pub fn encode(&self) -> Vec<u8> {
        let mut data = vec![0u8; if self.protection_absent { 7 } else { 9 }];

        // Sync word (12 bits)
        data[0] = 0xFF;
        data[1] = 0xF0;

        // MPEG version, layer, protection absent
        data[1] |= (self.mpeg_version & 1) << 3;
        data[1] |= (self.layer & 3) << 1;
        data[1] |= self.protection_absent as u8;

        // Profile, sample rate index, private bit, channel config (3 bits)
        data[2] = (self.profile & 3) << 6;
        data[2] |= (self.sample_rate_index & 0xF) << 2;
        data[2] |= (self.private_bit as u8) << 1;
        data[2] |= (self.channel_config >> 2) & 1;

        // Channel config (2 bits), flags, frame length (2 bits)
        data[3] = (self.channel_config & 3) << 6;
        data[3] |= (self.original as u8) << 5;
        data[3] |= (self.home as u8) << 4;
        data[3] |= (self.copyright_id_bit as u8) << 3;
        data[3] |= (self.copyright_id_start as u8) << 2;
        data[3] |= ((self.frame_length >> 11) & 3) as u8;

        // Frame length (8 bits)
        data[4] = ((self.frame_length >> 3) & 0xFF) as u8;

        // Frame length (3 bits), buffer fullness (5 bits)
        data[5] = ((self.frame_length & 7) << 5) as u8;
        data[5] |= ((self.buffer_fullness >> 6) & 0x1F) as u8;

        // Buffer fullness (6 bits), raw data blocks (2 bits)
        data[6] = ((self.buffer_fullness & 0x3F) << 2) as u8;
        data[6] |= self.num_raw_data_blocks & 3;

        // CRC
        if let Some(crc) = self.crc {
            data[7] = (crc >> 8) as u8;
            data[8] = (crc & 0xFF) as u8;
        }

        data
    }
}

/// Find next ADTS sync in a byte stream.
pub fn find_adts_sync(data: &[u8]) -> Option<usize> {
    for i in 0..data.len().saturating_sub(1) {
        if data[i] == 0xFF && (data[i + 1] & 0xF0) == 0xF0 {
            return Some(i);
        }
    }
    None
}

/// Iterator over ADTS frames.
pub struct AdtsIterator<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> AdtsIterator<'a> {
    /// Create a new ADTS iterator.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }
}

impl<'a> Iterator for AdtsIterator<'a> {
    type Item = Result<(&'a [u8], AdtsHeader)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.data.len() {
            return None;
        }

        let remaining = &self.data[self.offset..];

        // Find sync
        let sync_offset = find_adts_sync(remaining)?;
        let frame_data = &remaining[sync_offset..];

        // Parse header
        let header = match AdtsHeader::parse(frame_data) {
            Ok(h) => h,
            Err(e) => return Some(Err(e)),
        };

        let frame_len = header.frame_length as usize;
        if frame_data.len() < frame_len {
            return Some(Err(Error::Bitstream("Incomplete ADTS frame".into())));
        }

        let header_size = header.header_size();
        let payload = &frame_data[header_size..frame_len];

        self.offset += sync_offset + frame_len;

        Some(Ok((payload, header)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adts_parse() {
        // Valid ADTS header for AAC-LC, 44100 Hz, stereo
        let data = [
            0xFF, 0xF1, // Sync, MPEG-4, no CRC
            0x50,       // LC profile, 44100 Hz
            0x80,       // Stereo
            0x03, 0x00, // Frame length = 24
            0xFC,       // Buffer fullness
        ];

        let header = AdtsHeader::parse(&data).unwrap();
        assert_eq!(header.profile, 1); // LC
        assert_eq!(header.sample_rate(), 44100);
        assert_eq!(header.channels(), 2);
        assert!(header.protection_absent);
    }

    #[test]
    fn test_adts_parse_too_short() {
        let data = [0xFF, 0xF1, 0x50];
        assert!(AdtsHeader::parse(&data).is_err());
    }

    #[test]
    fn test_adts_parse_invalid_sync() {
        let data = [0xFF, 0x00, 0x50, 0x80, 0x03, 0x00, 0xFC];
        assert!(AdtsHeader::parse(&data).is_err());
    }

    #[test]
    fn test_adts_sample_rates() {
        // Test sample rate index mapping using encode/parse roundtrip
        let sample_rates: [(u8, u32); 12] = [
            (0, 96000),
            (1, 88200),
            (2, 64000),
            (3, 48000),
            (4, 44100),
            (5, 32000),
            (6, 24000),
            (7, 22050),
            (8, 16000),
            (9, 12000),
            (10, 11025),
            (11, 8000),
        ];

        for (index, expected_rate) in sample_rates {
            let header = AdtsHeader {
                mpeg_version: 0,
                layer: 0,
                protection_absent: true,
                profile: 1,
                sample_rate_index: index,
                private_bit: false,
                channel_config: 2,
                original: false,
                home: false,
                copyright_id_bit: false,
                copyright_id_start: false,
                frame_length: 100,
                buffer_fullness: 0x7FF,
                num_raw_data_blocks: 0,
                crc: None,
            };

            let encoded = header.encode();
            let parsed = AdtsHeader::parse(&encoded).unwrap();

            assert_eq!(
                parsed.sample_rate(),
                expected_rate,
                "Failed for index {}",
                index
            );
        }
    }

    #[test]
    fn test_adts_header_size() {
        // With CRC (protection_absent = false)
        let data = [
            0xFF, 0xF0, // Sync, MPEG-4, with CRC
            0x50, 0x80, 0x03, 0x80, 0xFC, 0x00, 0x00,
        ];
        let header = AdtsHeader::parse(&data).unwrap();
        assert_eq!(header.header_size(), 9);

        // Without CRC (protection_absent = true)
        let data = [
            0xFF, 0xF1, // Sync, MPEG-4, no CRC
            0x50, 0x80, 0x03, 0x00, 0xFC,
        ];
        let header = AdtsHeader::parse(&data).unwrap();
        assert_eq!(header.header_size(), 7);
    }

    #[test]
    fn test_adts_encode_roundtrip() {
        // Create a header
        let original = AdtsHeader {
            mpeg_version: 0,
            layer: 0,
            protection_absent: true,
            profile: 1, // LC
            sample_rate_index: 4, // 44100 Hz
            private_bit: false,
            channel_config: 2, // Stereo
            original: false,
            home: false,
            copyright_id_bit: false,
            copyright_id_start: false,
            frame_length: 107, // 7 + 100
            buffer_fullness: 0x7FF,
            num_raw_data_blocks: 0,
            crc: None,
        };

        // Encode and parse back
        let encoded = original.encode();
        let parsed = AdtsHeader::parse(&encoded).unwrap();

        assert_eq!(parsed.profile, original.profile);
        assert_eq!(parsed.sample_rate_index, original.sample_rate_index);
        assert_eq!(parsed.channel_config, original.channel_config);
        assert_eq!(parsed.frame_length, original.frame_length);
        assert_eq!(parsed.protection_absent, original.protection_absent);
    }

    #[test]
    fn test_adts_channels() {
        // Mono
        let data = [0xFF, 0xF1, 0x50, 0x40, 0x03, 0x00, 0xFC];
        let header = AdtsHeader::parse(&data).unwrap();
        assert_eq!(header.channels(), 1);

        // Stereo
        let data = [0xFF, 0xF1, 0x50, 0x80, 0x03, 0x00, 0xFC];
        let header = AdtsHeader::parse(&data).unwrap();
        assert_eq!(header.channels(), 2);
    }

    #[test]
    fn test_find_adts_sync() {
        let data = [0x00, 0x00, 0xFF, 0xF1, 0x50];
        assert_eq!(find_adts_sync(&data), Some(2));

        let data = [0xFF, 0xF1, 0x50];
        assert_eq!(find_adts_sync(&data), Some(0));

        let data = [0x00, 0x00, 0x00];
        assert_eq!(find_adts_sync(&data), None);
    }

    #[test]
    fn test_adts_payload_size() {
        let data = [0xFF, 0xF1, 0x50, 0x80, 0x00, 0xE0, 0xFC];
        let header = AdtsHeader::parse(&data).unwrap();
        let frame_len = header.frame_length as usize;
        let header_size = header.header_size();
        assert_eq!(header.payload_size(), frame_len - header_size);
    }
}
