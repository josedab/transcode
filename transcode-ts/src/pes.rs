//! PES (Packetized Elementary Stream) packet handling.
//!
//! This module provides parsing and generation of PES packets which carry
//! elementary stream data (video, audio) within MPEG-TS.

use crate::error::{Result, TsError};

/// PES start code prefix.
pub const PES_START_CODE_PREFIX: [u8; 3] = [0x00, 0x00, 0x01];

/// PES stream IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum StreamId {
    /// Program stream map.
    ProgramStreamMap = 0xBC,
    /// Private stream 1.
    PrivateStream1 = 0xBD,
    /// Padding stream.
    PaddingStream = 0xBE,
    /// Private stream 2.
    PrivateStream2 = 0xBF,
    /// ECM stream.
    EcmStream = 0xF0,
    /// EMM stream.
    EmmStream = 0xF1,
    /// DSMCC stream.
    DsmccStream = 0xF2,
    /// H.222.1 type E.
    H2221TypeE = 0xF8,
    /// Program stream directory.
    ProgramStreamDirectory = 0xFF,
}

impl StreamId {
    /// Audio stream base ID (0xC0 - 0xDF).
    pub const AUDIO_BASE: u8 = 0xC0;
    /// Video stream base ID (0xE0 - 0xEF).
    pub const VIDEO_BASE: u8 = 0xE0;

    /// Check if stream ID is audio.
    pub fn is_audio(id: u8) -> bool {
        (0xC0..=0xDF).contains(&id)
    }

    /// Check if stream ID is video.
    pub fn is_video(id: u8) -> bool {
        (0xE0..=0xEF).contains(&id)
    }

    /// Check if stream ID has PTS/DTS fields.
    pub fn has_pts_dts(id: u8) -> bool {
        // Most stream types except program_stream_map, padding, private_stream_2,
        // ECM, EMM, program_stream_directory, DSMCC, and H.222.1 type E
        !matches!(
            id,
            0xBC | 0xBE | 0xBF | 0xF0 | 0xF1 | 0xF2 | 0xF8 | 0xFF
        )
    }
}

/// Parsed PTS or DTS timestamp.
///
/// PTS/DTS are 33-bit values encoded in 5 bytes with marker bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PesTimestamp {
    /// 33-bit timestamp value (90 kHz clock).
    pub value: u64,
}

impl PesTimestamp {
    /// Maximum value for 33-bit timestamp.
    pub const MAX_VALUE: u64 = (1u64 << 33) - 1;

    /// Clock rate (90 kHz).
    pub const CLOCK_RATE: u64 = 90_000;

    /// Create a new PES timestamp.
    pub fn new(value: u64) -> Self {
        Self {
            value: value & Self::MAX_VALUE,
        }
    }

    /// Create from seconds.
    pub fn from_seconds(seconds: f64) -> Self {
        let value = (seconds * Self::CLOCK_RATE as f64) as u64;
        Self::new(value)
    }

    /// Convert to seconds.
    pub fn to_seconds(&self) -> f64 {
        self.value as f64 / Self::CLOCK_RATE as f64
    }

    /// Parse PTS/DTS from 5 bytes.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 5 {
            return Err(TsError::invalid_pes("Timestamp requires 5 bytes"));
        }

        // Format: '0010' or '0011' or '0001' (4 bits) + ts[32..30] + marker + ts[29..15] + marker + ts[14..0] + marker
        // Byte 0: pppp_ttt1 where pppp is prefix, ttt is ts[32:30]
        // Byte 1: tttt_tttt where t is ts[29:22]
        // Byte 2: tttt_ttt1 where t is ts[21:15]
        // Byte 3: tttt_tttt where t is ts[14:7]
        // Byte 4: tttt_ttt1 where t is ts[6:0]

        let value = ((data[0] as u64 & 0x0E) << 29)
            | ((data[1] as u64) << 22)
            | ((data[2] as u64 & 0xFE) << 14)
            | ((data[3] as u64) << 7)
            | ((data[4] as u64) >> 1);

        Ok(Self::new(value))
    }

    /// Write PTS/DTS to 5 bytes with prefix.
    ///
    /// `prefix` should be:
    /// - 0x20 for PTS only
    /// - 0x30 for PTS when DTS also present
    /// - 0x10 for DTS
    pub fn write(&self, data: &mut [u8], prefix: u8) -> Result<()> {
        if data.len() < 5 {
            return Err(TsError::BufferOverflow("Need 5 bytes for timestamp".to_string()));
        }

        data[0] = prefix | ((((self.value >> 30) as u8) & 0x07) << 1) | 0x01;
        data[1] = ((self.value >> 22) & 0xFF) as u8;
        data[2] = (((self.value >> 15) & 0x7F) << 1) as u8 | 0x01;
        data[3] = ((self.value >> 7) & 0xFF) as u8;
        data[4] = (((self.value) & 0x7F) << 1) as u8 | 0x01;

        Ok(())
    }
}

/// PES optional header flags.
#[derive(Debug, Clone, Copy, Default)]
pub struct PesFlags {
    /// PES scrambling control.
    pub scrambling_control: u8,
    /// PES priority.
    pub priority: bool,
    /// Data alignment indicator.
    pub data_alignment: bool,
    /// Copyright.
    pub copyright: bool,
    /// Original or copy.
    pub original: bool,
    /// PTS/DTS flags (00=none, 10=PTS only, 11=PTS+DTS).
    pub pts_dts_flags: u8,
    /// ESCR flag.
    pub escr_flag: bool,
    /// ES rate flag.
    pub es_rate_flag: bool,
    /// DSM trick mode flag.
    pub dsm_trick_mode_flag: bool,
    /// Additional copy info flag.
    pub additional_copy_info_flag: bool,
    /// PES CRC flag.
    pub pes_crc_flag: bool,
    /// PES extension flag.
    pub pes_extension_flag: bool,
    /// PES header data length.
    pub header_data_length: u8,
}

impl PesFlags {
    /// Parse from 2 bytes + header data length byte.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 3 {
            return Err(TsError::invalid_pes("Need at least 3 bytes for PES optional header"));
        }

        let byte0 = data[0];
        let byte1 = data[1];

        Ok(Self {
            scrambling_control: (byte0 >> 4) & 0x03,
            priority: (byte0 & 0x08) != 0,
            data_alignment: (byte0 & 0x04) != 0,
            copyright: (byte0 & 0x02) != 0,
            original: (byte0 & 0x01) != 0,
            pts_dts_flags: (byte1 >> 6) & 0x03,
            escr_flag: (byte1 & 0x20) != 0,
            es_rate_flag: (byte1 & 0x10) != 0,
            dsm_trick_mode_flag: (byte1 & 0x08) != 0,
            additional_copy_info_flag: (byte1 & 0x04) != 0,
            pes_crc_flag: (byte1 & 0x02) != 0,
            pes_extension_flag: (byte1 & 0x01) != 0,
            header_data_length: data[2],
        })
    }

    /// Write to buffer.
    pub fn write(&self, data: &mut [u8]) -> Result<()> {
        if data.len() < 3 {
            return Err(TsError::BufferOverflow("Need 3 bytes for PES flags".to_string()));
        }

        data[0] = 0x80 // marker bits '10'
            | ((self.scrambling_control & 0x03) << 4)
            | ((self.priority as u8) << 3)
            | ((self.data_alignment as u8) << 2)
            | ((self.copyright as u8) << 1)
            | (self.original as u8);

        data[1] = ((self.pts_dts_flags & 0x03) << 6)
            | ((self.escr_flag as u8) << 5)
            | ((self.es_rate_flag as u8) << 4)
            | ((self.dsm_trick_mode_flag as u8) << 3)
            | ((self.additional_copy_info_flag as u8) << 2)
            | ((self.pes_crc_flag as u8) << 1)
            | (self.pes_extension_flag as u8);

        data[2] = self.header_data_length;

        Ok(())
    }

    /// Check if PTS is present.
    pub fn has_pts(&self) -> bool {
        self.pts_dts_flags & 0x02 != 0
    }

    /// Check if DTS is present.
    pub fn has_dts(&self) -> bool {
        self.pts_dts_flags == 0x03
    }
}

/// Parsed PES packet header.
#[derive(Debug, Clone)]
pub struct PesHeader {
    /// Stream ID.
    pub stream_id: u8,
    /// PES packet length (0 for unbounded video).
    pub packet_length: u16,
    /// Optional header flags.
    pub flags: Option<PesFlags>,
    /// Presentation timestamp.
    pub pts: Option<PesTimestamp>,
    /// Decode timestamp.
    pub dts: Option<PesTimestamp>,
    /// Total header size (including start code).
    pub header_size: usize,
}

impl PesHeader {
    /// Minimum PES header size (start code + stream_id + length).
    pub const MIN_SIZE: usize = 6;

    /// Parse PES header from data.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < Self::MIN_SIZE {
            return Err(TsError::invalid_pes("Data too short for PES header"));
        }

        // Check start code prefix
        if data[0..3] != PES_START_CODE_PREFIX {
            return Err(TsError::invalid_pes("Invalid PES start code prefix"));
        }

        let stream_id = data[3];
        let packet_length = ((data[4] as u16) << 8) | (data[5] as u16);

        // Check if this stream type has optional header
        if !StreamId::has_pts_dts(stream_id) {
            return Ok(Self {
                stream_id,
                packet_length,
                flags: None,
                pts: None,
                dts: None,
                header_size: 6,
            });
        }

        // Parse optional header
        if data.len() < 9 {
            return Err(TsError::invalid_pes("Data too short for PES optional header"));
        }

        // Check marker bits
        if (data[6] & 0xC0) != 0x80 {
            return Err(TsError::invalid_pes("Invalid PES optional header marker bits"));
        }

        let flags = PesFlags::parse(&data[6..9])?;
        let header_data_end = 9 + flags.header_data_length as usize;

        if data.len() < header_data_end {
            return Err(TsError::invalid_pes("Header data length exceeds available data"));
        }

        let mut offset = 9;
        let mut pts = None;
        let mut dts = None;

        // Parse PTS
        if flags.has_pts() {
            if offset + 5 > data.len() {
                return Err(TsError::invalid_pes("Truncated PTS"));
            }
            pts = Some(PesTimestamp::parse(&data[offset..offset + 5])?);
            offset += 5;

            // Parse DTS if present
            if flags.has_dts() {
                if offset + 5 > data.len() {
                    return Err(TsError::invalid_pes("Truncated DTS"));
                }
                dts = Some(PesTimestamp::parse(&data[offset..offset + 5])?);
                // offset += 5; // Not needed as this is the last field we parse
            }
        }

        Ok(Self {
            stream_id,
            packet_length,
            flags: Some(flags),
            pts,
            dts,
            header_size: header_data_end,
        })
    }

    /// Check if this is a video stream.
    pub fn is_video(&self) -> bool {
        StreamId::is_video(self.stream_id)
    }

    /// Check if this is an audio stream.
    pub fn is_audio(&self) -> bool {
        StreamId::is_audio(self.stream_id)
    }

    /// Get the payload offset within the PES packet.
    pub fn payload_offset(&self) -> usize {
        self.header_size
    }
}

/// PES packet builder for muxing.
#[derive(Debug)]
pub struct PesPacketBuilder {
    /// Stream ID.
    stream_id: u8,
    /// Presentation timestamp.
    pts: Option<PesTimestamp>,
    /// Decode timestamp.
    dts: Option<PesTimestamp>,
    /// Data alignment indicator.
    data_alignment: bool,
    /// Random access indicator (keyframe).
    random_access: bool,
}

impl PesPacketBuilder {
    /// Create a new PES packet builder for video.
    pub fn video() -> Self {
        Self {
            stream_id: StreamId::VIDEO_BASE,
            pts: None,
            dts: None,
            data_alignment: true,
            random_access: false,
        }
    }

    /// Create a new PES packet builder for audio.
    pub fn audio() -> Self {
        Self {
            stream_id: StreamId::AUDIO_BASE,
            pts: None,
            dts: None,
            data_alignment: true,
            random_access: false,
        }
    }

    /// Create with specific stream ID.
    pub fn with_stream_id(stream_id: u8) -> Self {
        Self {
            stream_id,
            pts: None,
            dts: None,
            data_alignment: true,
            random_access: false,
        }
    }

    /// Set PTS.
    pub fn pts(mut self, pts: PesTimestamp) -> Self {
        self.pts = Some(pts);
        self
    }

    /// Set PTS and DTS.
    pub fn pts_dts(mut self, pts: PesTimestamp, dts: PesTimestamp) -> Self {
        self.pts = Some(pts);
        self.dts = Some(dts);
        self
    }

    /// Set random access indicator.
    pub fn random_access(mut self, is_keyframe: bool) -> Self {
        self.random_access = is_keyframe;
        self
    }

    /// Set data alignment indicator.
    pub fn data_alignment(mut self, aligned: bool) -> Self {
        self.data_alignment = aligned;
        self
    }

    /// Calculate the header size.
    pub fn header_size(&self) -> usize {
        let mut size = 9; // start code (3) + stream_id (1) + length (2) + optional header (3)

        if self.pts.is_some() {
            size += 5;
        }
        if self.dts.is_some() {
            size += 5;
        }

        size
    }

    /// Build the PES packet header.
    pub fn build_header(&self, payload_length: usize) -> Result<Vec<u8>> {
        let header_data_length = if self.pts.is_some() {
            if self.dts.is_some() {
                10
            } else {
                5
            }
        } else {
            0
        };

        let pes_packet_length = if payload_length == 0 || self.is_unbounded_video() {
            0 // Unbounded for video
        } else {
            (3 + header_data_length + payload_length).min(0xFFFF) as u16
        };

        let mut header = Vec::with_capacity(self.header_size());

        // Start code prefix
        header.extend_from_slice(&PES_START_CODE_PREFIX);

        // Stream ID
        header.push(self.stream_id);

        // PES packet length
        header.push((pes_packet_length >> 8) as u8);
        header.push((pes_packet_length & 0xFF) as u8);

        // Optional header
        let pts_dts_flags = match (self.pts.is_some(), self.dts.is_some()) {
            (true, true) => 0x03,
            (true, false) => 0x02,
            _ => 0x00,
        };

        let flags = PesFlags {
            scrambling_control: 0,
            priority: false,
            data_alignment: self.data_alignment,
            copyright: false,
            original: true,
            pts_dts_flags,
            escr_flag: false,
            es_rate_flag: false,
            dsm_trick_mode_flag: false,
            additional_copy_info_flag: false,
            pes_crc_flag: false,
            pes_extension_flag: false,
            header_data_length: header_data_length as u8,
        };

        let mut flags_bytes = [0u8; 3];
        flags.write(&mut flags_bytes)?;
        header.extend_from_slice(&flags_bytes);

        // PTS
        if let Some(pts) = &self.pts {
            let mut pts_bytes = [0u8; 5];
            let prefix = if self.dts.is_some() { 0x30 } else { 0x20 };
            pts.write(&mut pts_bytes, prefix)?;
            header.extend_from_slice(&pts_bytes);
        }

        // DTS
        if let Some(dts) = &self.dts {
            let mut dts_bytes = [0u8; 5];
            dts.write(&mut dts_bytes, 0x10)?;
            header.extend_from_slice(&dts_bytes);
        }

        Ok(header)
    }

    /// Check if this is unbounded video.
    fn is_unbounded_video(&self) -> bool {
        StreamId::is_video(self.stream_id)
    }
}

/// PES packet assembler for demuxing.
///
/// Assembles PES packets from multiple TS packets.
#[derive(Debug)]
pub struct PesAssembler {
    /// PID being assembled.
    pid: u16,
    /// Accumulated data.
    buffer: Vec<u8>,
    /// Whether we have started receiving data.
    started: bool,
    /// Parsed header (if available).
    header: Option<PesHeader>,
}

impl PesAssembler {
    /// Create a new PES assembler.
    pub fn new(pid: u16) -> Self {
        Self {
            pid,
            buffer: Vec::with_capacity(65536),
            started: false,
            header: None,
        }
    }

    /// Reset the assembler.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.started = false;
        self.header = None;
    }

    /// Get the PID.
    pub fn pid(&self) -> u16 {
        self.pid
    }

    /// Add TS packet payload to the assembler.
    ///
    /// Returns the complete PES packet data if available.
    pub fn add(&mut self, payload: &[u8], pusi: bool) -> Result<Option<Vec<u8>>> {
        if pusi {
            // Start of new PES packet
            // If we have previous data, return it first
            let previous = if self.started && !self.buffer.is_empty() {
                let data = std::mem::take(&mut self.buffer);
                Some(data)
            } else {
                None
            };

            self.buffer.clear();
            self.started = true;
            self.header = None;

            // Add new data
            self.buffer.extend_from_slice(payload);

            // Try to parse header
            if self.buffer.len() >= PesHeader::MIN_SIZE {
                if let Ok(header) = PesHeader::parse(&self.buffer) {
                    self.header = Some(header);
                }
            }

            return Ok(previous);
        }

        // Continuation of current PES packet
        if self.started {
            self.buffer.extend_from_slice(payload);
        }

        // Check if we have a complete packet
        if let Some(ref header) = self.header {
            if header.packet_length > 0 {
                let expected_size = 6 + header.packet_length as usize;
                if self.buffer.len() >= expected_size {
                    let data = self.buffer[..expected_size].to_vec();
                    self.buffer.drain(..expected_size);
                    if self.buffer.is_empty() {
                        self.started = false;
                        self.header = None;
                    }
                    return Ok(Some(data));
                }
            }
        }

        Ok(None)
    }

    /// Flush any remaining data.
    pub fn flush(&mut self) -> Option<Vec<u8>> {
        if self.started && !self.buffer.is_empty() {
            self.started = false;
            self.header = None;
            Some(std::mem::take(&mut self.buffer))
        } else {
            None
        }
    }

    /// Get the parsed header if available.
    pub fn header(&self) -> Option<&PesHeader> {
        self.header.as_ref()
    }

    /// Check if assembler has started receiving data.
    pub fn is_started(&self) -> bool {
        self.started
    }

    /// Get current buffer size.
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pes_timestamp_parse_write() {
        let original = PesTimestamp::new(12345678);

        let mut data = [0u8; 5];
        original.write(&mut data, 0x20).unwrap();

        let parsed = PesTimestamp::parse(&data).unwrap();
        assert_eq!(parsed.value, original.value);
    }

    #[test]
    fn test_pes_timestamp_seconds() {
        let ts = PesTimestamp::from_seconds(1.0);
        assert_eq!(ts.value, 90_000);

        let back = ts.to_seconds();
        assert!((back - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_pes_header_parse() {
        // Create a simple PES header with PTS
        let mut data = vec![0x00, 0x00, 0x01]; // Start code prefix
        data.push(0xE0); // Video stream ID
        data.push(0x00); // Packet length high
        data.push(0x10); // Packet length low (16 bytes)
        data.push(0x80); // Marker bits
        data.push(0x80); // PTS only
        data.push(0x05); // Header data length

        // Add PTS (5 bytes)
        let pts = PesTimestamp::new(90000);
        let mut pts_bytes = [0u8; 5];
        pts.write(&mut pts_bytes, 0x20).unwrap();
        data.extend_from_slice(&pts_bytes);

        let header = PesHeader::parse(&data).unwrap();
        assert_eq!(header.stream_id, 0xE0);
        assert!(header.is_video());
        assert!(header.pts.is_some());
        assert_eq!(header.pts.unwrap().value, 90000);
        assert!(header.dts.is_none());
    }

    #[test]
    fn test_pes_packet_builder_video() {
        let pts = PesTimestamp::new(90000);
        let dts = PesTimestamp::new(87000);

        let builder = PesPacketBuilder::video()
            .pts_dts(pts, dts)
            .random_access(true);

        let header = builder.build_header(1000).unwrap();

        // Parse it back
        let parsed = PesHeader::parse(&header).unwrap();
        assert!(parsed.is_video());
        assert_eq!(parsed.pts.unwrap().value, 90000);
        assert_eq!(parsed.dts.unwrap().value, 87000);
    }

    #[test]
    fn test_pes_packet_builder_audio() {
        let pts = PesTimestamp::new(45000);

        let builder = PesPacketBuilder::audio().pts(pts);

        let header = builder.build_header(500).unwrap();

        let parsed = PesHeader::parse(&header).unwrap();
        assert!(parsed.is_audio());
        assert_eq!(parsed.pts.unwrap().value, 45000);
        assert!(parsed.dts.is_none());
    }

    #[test]
    fn test_stream_id() {
        assert!(StreamId::is_video(0xE0));
        assert!(StreamId::is_video(0xEF));
        assert!(!StreamId::is_video(0xC0));

        assert!(StreamId::is_audio(0xC0));
        assert!(StreamId::is_audio(0xDF));
        assert!(!StreamId::is_audio(0xE0));

        assert!(StreamId::has_pts_dts(0xE0));
        assert!(StreamId::has_pts_dts(0xC0));
        assert!(!StreamId::has_pts_dts(0xBE)); // Padding
    }

    #[test]
    fn test_pes_assembler() {
        let mut assembler = PesAssembler::new(256);

        // Create a PES packet
        let builder = PesPacketBuilder::video().pts(PesTimestamp::new(90000));
        let header = builder.build_header(100).unwrap();

        let mut pes_data = header.clone();
        // Add 100 bytes of payload
        for i in 0..100 {
            pes_data.push(i as u8);
        }

        // First packet with PUSI (send first half)
        let mid = pes_data.len() / 2;
        let result = assembler.add(&pes_data[..mid], true).unwrap();
        assert!(result.is_none()); // No previous packet

        // Continuation (send rest)
        let result = assembler.add(&pes_data[mid..], false).unwrap();
        assert!(result.is_none()); // Not complete yet (unbounded video)

        // Start new packet - should return previous
        let result = assembler.add(&pes_data[..10], true).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_pes_flags() {
        let flags = PesFlags {
            scrambling_control: 0,
            priority: false,
            data_alignment: true,
            copyright: false,
            original: true,
            pts_dts_flags: 0x03,
            escr_flag: false,
            es_rate_flag: false,
            dsm_trick_mode_flag: false,
            additional_copy_info_flag: false,
            pes_crc_flag: false,
            pes_extension_flag: false,
            header_data_length: 10,
        };

        let mut data = [0u8; 3];
        flags.write(&mut data).unwrap();

        let parsed = PesFlags::parse(&data).unwrap();
        assert!(parsed.has_pts());
        assert!(parsed.has_dts());
        assert!(parsed.data_alignment);
        assert!(parsed.original);
        assert_eq!(parsed.header_data_length, 10);
    }
}
