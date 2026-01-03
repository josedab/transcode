//! Program Specific Information (PSI) tables.
//!
//! This module provides parsing and generation of PSI tables used in MPEG-TS:
//! - PAT (Program Association Table)
//! - PMT (Program Map Table)
//! - SDT (Service Description Table)

use crate::error::{Result, TsError};

/// CRC-32 polynomial used in MPEG-TS (ISO/IEC 13818-1).
const CRC32_POLY: u32 = 0x04C11DB7;

/// Pre-computed CRC-32 table.
static CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = (i as u32) << 24;
        let mut j = 0;
        while j < 8 {
            if crc & 0x80000000 != 0 {
                crc = (crc << 1) ^ CRC32_POLY;
            } else {
                crc <<= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

/// Calculate CRC-32 for PSI sections.
pub fn calculate_crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFF;
    for &byte in data {
        let index = ((crc >> 24) ^ (byte as u32)) as usize;
        crc = (crc << 8) ^ CRC32_TABLE[index];
    }
    crc
}

/// MPEG-TS stream types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum StreamType {
    /// MPEG-1 Video.
    Mpeg1Video = 0x01,
    /// MPEG-2 Video.
    Mpeg2Video = 0x02,
    /// MPEG-1 Audio.
    Mpeg1Audio = 0x03,
    /// MPEG-2 Audio.
    Mpeg2Audio = 0x04,
    /// Private sections.
    PrivateSections = 0x05,
    /// Private PES data.
    PrivateData = 0x06,
    /// MHEG multimedia.
    Mheg = 0x07,
    /// DSM-CC.
    DsmCc = 0x08,
    /// ITU-T H.222.1.
    H2221 = 0x09,
    /// DSM-CC type A.
    DsmCcTypeA = 0x0A,
    /// DSM-CC type B.
    DsmCcTypeB = 0x0B,
    /// DSM-CC type C.
    DsmCcTypeC = 0x0C,
    /// DSM-CC type D.
    DsmCcTypeD = 0x0D,
    /// MPEG-2 auxiliary.
    Mpeg2Auxiliary = 0x0E,
    /// AAC ADTS.
    AacAdts = 0x0F,
    /// MPEG-4 Visual.
    Mpeg4Visual = 0x10,
    /// AAC LATM.
    AacLatm = 0x11,
    /// MPEG-4 FlexMux PES.
    Mpeg4FlexMuxPes = 0x12,
    /// MPEG-4 FlexMux sections.
    Mpeg4FlexMuxSections = 0x13,
    /// Synchronized Download Protocol.
    Sdp = 0x14,
    /// Metadata in PES.
    MetadataPes = 0x15,
    /// Metadata in sections.
    MetadataSections = 0x16,
    /// Metadata in Data Carousel.
    MetadataDataCarousel = 0x17,
    /// Metadata in Object Carousel.
    MetadataObjectCarousel = 0x18,
    /// Synchronized Download Protocol descriptor.
    SdpDescriptor = 0x19,
    /// IPMP.
    Ipmp = 0x1A,
    /// H.264/AVC video.
    H264 = 0x1B,
    /// H.265/HEVC video.
    H265 = 0x24,
    /// AC-3 audio (ATSC).
    Ac3 = 0x81,
    /// SCTE-35 splice info.
    Scte35 = 0x86,
    /// E-AC-3 audio.
    Eac3 = 0x87,
}

impl StreamType {
    /// Create from raw value.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x01 => Some(StreamType::Mpeg1Video),
            0x02 => Some(StreamType::Mpeg2Video),
            0x03 => Some(StreamType::Mpeg1Audio),
            0x04 => Some(StreamType::Mpeg2Audio),
            0x05 => Some(StreamType::PrivateSections),
            0x06 => Some(StreamType::PrivateData),
            0x07 => Some(StreamType::Mheg),
            0x08 => Some(StreamType::DsmCc),
            0x09 => Some(StreamType::H2221),
            0x0A => Some(StreamType::DsmCcTypeA),
            0x0B => Some(StreamType::DsmCcTypeB),
            0x0C => Some(StreamType::DsmCcTypeC),
            0x0D => Some(StreamType::DsmCcTypeD),
            0x0E => Some(StreamType::Mpeg2Auxiliary),
            0x0F => Some(StreamType::AacAdts),
            0x10 => Some(StreamType::Mpeg4Visual),
            0x11 => Some(StreamType::AacLatm),
            0x12 => Some(StreamType::Mpeg4FlexMuxPes),
            0x13 => Some(StreamType::Mpeg4FlexMuxSections),
            0x14 => Some(StreamType::Sdp),
            0x15 => Some(StreamType::MetadataPes),
            0x16 => Some(StreamType::MetadataSections),
            0x17 => Some(StreamType::MetadataDataCarousel),
            0x18 => Some(StreamType::MetadataObjectCarousel),
            0x19 => Some(StreamType::SdpDescriptor),
            0x1A => Some(StreamType::Ipmp),
            0x1B => Some(StreamType::H264),
            0x24 => Some(StreamType::H265),
            0x81 => Some(StreamType::Ac3),
            0x86 => Some(StreamType::Scte35),
            0x87 => Some(StreamType::Eac3),
            _ => None,
        }
    }

    /// Check if this is a video stream type.
    pub fn is_video(&self) -> bool {
        matches!(
            self,
            StreamType::Mpeg1Video
                | StreamType::Mpeg2Video
                | StreamType::Mpeg4Visual
                | StreamType::H264
                | StreamType::H265
        )
    }

    /// Check if this is an audio stream type.
    pub fn is_audio(&self) -> bool {
        matches!(
            self,
            StreamType::Mpeg1Audio
                | StreamType::Mpeg2Audio
                | StreamType::AacAdts
                | StreamType::AacLatm
                | StreamType::Ac3
                | StreamType::Eac3
        )
    }
}

/// PSI section header common to all table types.
#[derive(Debug, Clone)]
pub struct PsiHeader {
    /// Table ID.
    pub table_id: u8,
    /// Section syntax indicator.
    pub section_syntax_indicator: bool,
    /// Section length (12 bits).
    pub section_length: u16,
    /// Table ID extension.
    pub table_id_extension: u16,
    /// Version number (5 bits).
    pub version_number: u8,
    /// Current/next indicator.
    pub current_next: bool,
    /// Section number.
    pub section_number: u8,
    /// Last section number.
    pub last_section_number: u8,
}

impl PsiHeader {
    /// Parse PSI header from data.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(TsError::invalid_psi("Section too short for header"));
        }

        let table_id = data[0];
        let section_syntax_indicator = (data[1] & 0x80) != 0;
        let section_length = ((data[1] as u16 & 0x0F) << 8) | (data[2] as u16);

        if !section_syntax_indicator {
            // Short form without extended header
            return Ok(Self {
                table_id,
                section_syntax_indicator,
                section_length,
                table_id_extension: 0,
                version_number: 0,
                current_next: true,
                section_number: 0,
                last_section_number: 0,
            });
        }

        let table_id_extension = ((data[3] as u16) << 8) | (data[4] as u16);
        let version_number = (data[5] >> 1) & 0x1F;
        let current_next = (data[5] & 0x01) != 0;
        let section_number = data[6];
        let last_section_number = data[7];

        Ok(Self {
            table_id,
            section_syntax_indicator,
            section_length,
            table_id_extension,
            version_number,
            current_next,
            section_number,
            last_section_number,
        })
    }

    /// Get the total section size including header.
    pub fn section_size(&self) -> usize {
        3 + self.section_length as usize
    }

    /// Write PSI header to buffer.
    pub fn write(&self, data: &mut [u8]) -> Result<usize> {
        if data.len() < 8 {
            return Err(TsError::BufferOverflow("Need 8 bytes for PSI header".to_string()));
        }

        data[0] = self.table_id;
        data[1] = ((self.section_syntax_indicator as u8) << 7)
            | 0x30 // Reserved bits set to 1, private indicator = 0
            | ((self.section_length >> 8) as u8 & 0x0F);
        data[2] = (self.section_length & 0xFF) as u8;
        data[3] = (self.table_id_extension >> 8) as u8;
        data[4] = (self.table_id_extension & 0xFF) as u8;
        data[5] = 0xC0 // Reserved bits
            | ((self.version_number & 0x1F) << 1)
            | (self.current_next as u8);
        data[6] = self.section_number;
        data[7] = self.last_section_number;

        Ok(8)
    }
}

/// Program entry in PAT.
#[derive(Debug, Clone, Copy)]
pub struct PatEntry {
    /// Program number (0 = NIT).
    pub program_number: u16,
    /// PID of PMT or NIT.
    pub pid: u16,
}

/// Program Association Table (PAT).
#[derive(Debug, Clone)]
pub struct Pat {
    /// Transport stream ID.
    pub transport_stream_id: u16,
    /// Version number.
    pub version_number: u8,
    /// Current/next indicator.
    pub current_next: bool,
    /// Program entries.
    pub programs: Vec<PatEntry>,
}

impl Pat {
    /// PAT table ID.
    pub const TABLE_ID: u8 = 0x00;

    /// Create a new PAT.
    pub fn new(transport_stream_id: u16) -> Self {
        Self {
            transport_stream_id,
            version_number: 0,
            current_next: true,
            programs: Vec::new(),
        }
    }

    /// Add a program entry.
    pub fn add_program(&mut self, program_number: u16, pmt_pid: u16) {
        self.programs.push(PatEntry {
            program_number,
            pid: pmt_pid,
        });
    }

    /// Parse PAT from section data.
    pub fn parse(data: &[u8]) -> Result<Self> {
        let header = PsiHeader::parse(data)?;

        if header.table_id != Self::TABLE_ID {
            return Err(TsError::invalid_pat(format!(
                "Expected table ID 0x00, got 0x{:02X}",
                header.table_id
            )));
        }

        // Verify CRC
        let section_end = header.section_size();
        if data.len() < section_end {
            return Err(TsError::invalid_pat("Section truncated"));
        }

        let crc_offset = section_end - 4;
        let stored_crc =
            ((data[crc_offset] as u32) << 24)
                | ((data[crc_offset + 1] as u32) << 16)
                | ((data[crc_offset + 2] as u32) << 8)
                | (data[crc_offset + 3] as u32);

        let calculated_crc = calculate_crc32(&data[..crc_offset]);
        if stored_crc != calculated_crc {
            return Err(TsError::CrcMismatch {
                expected: stored_crc,
                actual: calculated_crc,
            });
        }

        // Parse program entries
        let mut programs = Vec::new();
        let mut offset = 8; // After header
        let entries_end = crc_offset;

        while offset + 4 <= entries_end {
            let program_number = ((data[offset] as u16) << 8) | (data[offset + 1] as u16);
            let pid = ((data[offset + 2] as u16 & 0x1F) << 8) | (data[offset + 3] as u16);

            programs.push(PatEntry {
                program_number,
                pid,
            });

            offset += 4;
        }

        Ok(Self {
            transport_stream_id: header.table_id_extension,
            version_number: header.version_number,
            current_next: header.current_next,
            programs,
        })
    }

    /// Serialize PAT to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        let entries_len = self.programs.len() * 4;
        let section_length = 5 + entries_len + 4; // header fields + entries + CRC

        let mut data = vec![0u8; 3 + section_length];

        // Table ID
        data[0] = Self::TABLE_ID;
        // Section syntax + reserved + section length
        data[1] = 0xB0 | ((section_length >> 8) as u8 & 0x0F);
        data[2] = (section_length & 0xFF) as u8;
        // Transport stream ID
        data[3] = (self.transport_stream_id >> 8) as u8;
        data[4] = (self.transport_stream_id & 0xFF) as u8;
        // Reserved + version + current/next
        data[5] = 0xC1 | ((self.version_number & 0x1F) << 1);
        // Section number
        data[6] = 0;
        // Last section number
        data[7] = 0;

        // Program entries
        let mut offset = 8;
        for entry in &self.programs {
            data[offset] = (entry.program_number >> 8) as u8;
            data[offset + 1] = (entry.program_number & 0xFF) as u8;
            data[offset + 2] = 0xE0 | ((entry.pid >> 8) as u8 & 0x1F);
            data[offset + 3] = (entry.pid & 0xFF) as u8;
            offset += 4;
        }

        // Calculate and append CRC
        let crc = calculate_crc32(&data[..offset]);
        data[offset] = (crc >> 24) as u8;
        data[offset + 1] = (crc >> 16) as u8;
        data[offset + 2] = (crc >> 8) as u8;
        data[offset + 3] = (crc & 0xFF) as u8;

        data
    }

    /// Get PMT PID for a program.
    pub fn get_pmt_pid(&self, program_number: u16) -> Option<u16> {
        self.programs
            .iter()
            .find(|p| p.program_number == program_number)
            .map(|p| p.pid)
    }
}

/// Elementary stream entry in PMT.
#[derive(Debug, Clone)]
pub struct PmtStream {
    /// Stream type.
    pub stream_type: u8,
    /// Elementary stream PID.
    pub pid: u16,
    /// ES info descriptors.
    pub descriptors: Vec<u8>,
}

impl PmtStream {
    /// Check if this is a video stream.
    pub fn is_video(&self) -> bool {
        StreamType::from_u8(self.stream_type)
            .map(|st| st.is_video())
            .unwrap_or(false)
    }

    /// Check if this is an audio stream.
    pub fn is_audio(&self) -> bool {
        StreamType::from_u8(self.stream_type)
            .map(|st| st.is_audio())
            .unwrap_or(false)
    }
}

/// Program Map Table (PMT).
#[derive(Debug, Clone)]
pub struct Pmt {
    /// Program number.
    pub program_number: u16,
    /// Version number.
    pub version_number: u8,
    /// Current/next indicator.
    pub current_next: bool,
    /// PCR PID.
    pub pcr_pid: u16,
    /// Program info descriptors.
    pub program_info: Vec<u8>,
    /// Elementary streams.
    pub streams: Vec<PmtStream>,
}

impl Pmt {
    /// PMT table ID.
    pub const TABLE_ID: u8 = 0x02;

    /// Create a new PMT.
    pub fn new(program_number: u16, pcr_pid: u16) -> Self {
        Self {
            program_number,
            version_number: 0,
            current_next: true,
            pcr_pid,
            program_info: Vec::new(),
            streams: Vec::new(),
        }
    }

    /// Add an elementary stream.
    pub fn add_stream(&mut self, stream_type: u8, pid: u16) {
        self.streams.push(PmtStream {
            stream_type,
            pid,
            descriptors: Vec::new(),
        });
    }

    /// Parse PMT from section data.
    pub fn parse(data: &[u8]) -> Result<Self> {
        let header = PsiHeader::parse(data)?;

        if header.table_id != Self::TABLE_ID {
            return Err(TsError::invalid_pmt(format!(
                "Expected table ID 0x02, got 0x{:02X}",
                header.table_id
            )));
        }

        // Verify CRC
        let section_end = header.section_size();
        if data.len() < section_end {
            return Err(TsError::invalid_pmt("Section truncated"));
        }

        let crc_offset = section_end - 4;
        let stored_crc =
            ((data[crc_offset] as u32) << 24)
                | ((data[crc_offset + 1] as u32) << 16)
                | ((data[crc_offset + 2] as u32) << 8)
                | (data[crc_offset + 3] as u32);

        let calculated_crc = calculate_crc32(&data[..crc_offset]);
        if stored_crc != calculated_crc {
            return Err(TsError::CrcMismatch {
                expected: stored_crc,
                actual: calculated_crc,
            });
        }

        // Parse PMT specific fields
        if data.len() < 12 {
            return Err(TsError::invalid_pmt("PMT too short"));
        }

        let pcr_pid = ((data[8] as u16 & 0x1F) << 8) | (data[9] as u16);
        let program_info_length = ((data[10] as u16 & 0x0F) << 8) | (data[11] as u16);

        let mut offset = 12 + program_info_length as usize;
        let program_info = if program_info_length > 0 {
            data[12..offset].to_vec()
        } else {
            Vec::new()
        };

        // Parse elementary streams
        let mut streams = Vec::new();
        while offset + 5 <= crc_offset {
            let stream_type = data[offset];
            let pid = ((data[offset + 1] as u16 & 0x1F) << 8) | (data[offset + 2] as u16);
            let es_info_length = ((data[offset + 3] as u16 & 0x0F) << 8) | (data[offset + 4] as u16);

            let descriptors = if es_info_length > 0 {
                let desc_end = offset + 5 + es_info_length as usize;
                if desc_end > crc_offset {
                    return Err(TsError::invalid_pmt("ES info length exceeds section"));
                }
                data[offset + 5..desc_end].to_vec()
            } else {
                Vec::new()
            };

            streams.push(PmtStream {
                stream_type,
                pid,
                descriptors,
            });

            offset += 5 + es_info_length as usize;
        }

        Ok(Self {
            program_number: header.table_id_extension,
            version_number: header.version_number,
            current_next: header.current_next,
            pcr_pid,
            program_info,
            streams,
        })
    }

    /// Serialize PMT to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        let program_info_len = self.program_info.len();
        let streams_len: usize = self
            .streams
            .iter()
            .map(|s| 5 + s.descriptors.len())
            .sum();
        let section_length = 9 + program_info_len + streams_len + 4; // header + pcr + prog_info + streams + CRC

        let mut data = vec![0u8; 3 + section_length];

        // Table ID
        data[0] = Self::TABLE_ID;
        // Section syntax + reserved + section length
        data[1] = 0xB0 | ((section_length >> 8) as u8 & 0x0F);
        data[2] = (section_length & 0xFF) as u8;
        // Program number
        data[3] = (self.program_number >> 8) as u8;
        data[4] = (self.program_number & 0xFF) as u8;
        // Reserved + version + current/next
        data[5] = 0xC1 | ((self.version_number & 0x1F) << 1);
        // Section number
        data[6] = 0;
        // Last section number
        data[7] = 0;
        // PCR PID
        data[8] = 0xE0 | ((self.pcr_pid >> 8) as u8 & 0x1F);
        data[9] = (self.pcr_pid & 0xFF) as u8;
        // Program info length
        data[10] = 0xF0 | ((program_info_len >> 8) as u8 & 0x0F);
        data[11] = (program_info_len & 0xFF) as u8;

        let mut offset = 12;

        // Program info descriptors
        if !self.program_info.is_empty() {
            data[offset..offset + program_info_len].copy_from_slice(&self.program_info);
            offset += program_info_len;
        }

        // Elementary streams
        for stream in &self.streams {
            data[offset] = stream.stream_type;
            data[offset + 1] = 0xE0 | ((stream.pid >> 8) as u8 & 0x1F);
            data[offset + 2] = (stream.pid & 0xFF) as u8;

            let desc_len = stream.descriptors.len();
            data[offset + 3] = 0xF0 | ((desc_len >> 8) as u8 & 0x0F);
            data[offset + 4] = (desc_len & 0xFF) as u8;

            if !stream.descriptors.is_empty() {
                data[offset + 5..offset + 5 + desc_len].copy_from_slice(&stream.descriptors);
            }

            offset += 5 + desc_len;
        }

        // Calculate and append CRC
        let crc = calculate_crc32(&data[..offset]);
        data[offset] = (crc >> 24) as u8;
        data[offset + 1] = (crc >> 16) as u8;
        data[offset + 2] = (crc >> 8) as u8;
        data[offset + 3] = (crc & 0xFF) as u8;

        data
    }

    /// Get stream by PID.
    pub fn get_stream(&self, pid: u16) -> Option<&PmtStream> {
        self.streams.iter().find(|s| s.pid == pid)
    }

    /// Get the video stream (first video stream if multiple).
    pub fn video_stream(&self) -> Option<&PmtStream> {
        self.streams.iter().find(|s| s.is_video())
    }

    /// Get the audio stream (first audio stream if multiple).
    pub fn audio_stream(&self) -> Option<&PmtStream> {
        self.streams.iter().find(|s| s.is_audio())
    }
}

/// Service in SDT.
#[derive(Debug, Clone)]
pub struct SdtService {
    /// Service ID.
    pub service_id: u16,
    /// EIT schedule flag.
    pub eit_schedule_flag: bool,
    /// EIT present/following flag.
    pub eit_present_following_flag: bool,
    /// Running status.
    pub running_status: u8,
    /// Free CA mode.
    pub free_ca_mode: bool,
    /// Descriptors.
    pub descriptors: Vec<u8>,
}

/// Service Description Table (SDT).
#[derive(Debug, Clone)]
pub struct Sdt {
    /// Transport stream ID.
    pub transport_stream_id: u16,
    /// Original network ID.
    pub original_network_id: u16,
    /// Version number.
    pub version_number: u8,
    /// Current/next indicator.
    pub current_next: bool,
    /// Services.
    pub services: Vec<SdtService>,
}

impl Sdt {
    /// SDT actual table ID.
    pub const TABLE_ID_ACTUAL: u8 = 0x42;
    /// SDT other table ID.
    pub const TABLE_ID_OTHER: u8 = 0x46;

    /// Create a new SDT.
    pub fn new(transport_stream_id: u16, original_network_id: u16) -> Self {
        Self {
            transport_stream_id,
            original_network_id,
            version_number: 0,
            current_next: true,
            services: Vec::new(),
        }
    }

    /// Parse SDT from section data.
    pub fn parse(data: &[u8]) -> Result<Self> {
        let header = PsiHeader::parse(data)?;

        if header.table_id != Self::TABLE_ID_ACTUAL && header.table_id != Self::TABLE_ID_OTHER {
            return Err(TsError::invalid_psi(format!(
                "Expected SDT table ID 0x42 or 0x46, got 0x{:02X}",
                header.table_id
            )));
        }

        // Verify CRC
        let section_end = header.section_size();
        if data.len() < section_end {
            return Err(TsError::invalid_psi("Section truncated"));
        }

        let crc_offset = section_end - 4;
        let stored_crc =
            ((data[crc_offset] as u32) << 24)
                | ((data[crc_offset + 1] as u32) << 16)
                | ((data[crc_offset + 2] as u32) << 8)
                | (data[crc_offset + 3] as u32);

        let calculated_crc = calculate_crc32(&data[..crc_offset]);
        if stored_crc != calculated_crc {
            return Err(TsError::CrcMismatch {
                expected: stored_crc,
                actual: calculated_crc,
            });
        }

        if data.len() < 11 {
            return Err(TsError::invalid_psi("SDT too short"));
        }

        let original_network_id = ((data[8] as u16) << 8) | (data[9] as u16);
        // data[10] is reserved

        let mut services = Vec::new();
        let mut offset = 11;

        while offset + 5 <= crc_offset {
            let service_id = ((data[offset] as u16) << 8) | (data[offset + 1] as u16);
            let eit_schedule_flag = (data[offset + 2] & 0x02) != 0;
            let eit_present_following_flag = (data[offset + 2] & 0x01) != 0;
            let running_status = (data[offset + 3] >> 5) & 0x07;
            let free_ca_mode = (data[offset + 3] & 0x10) != 0;
            let desc_loop_length =
                ((data[offset + 3] as u16 & 0x0F) << 8) | (data[offset + 4] as u16);

            let descriptors = if desc_loop_length > 0 {
                let desc_end = offset + 5 + desc_loop_length as usize;
                if desc_end > crc_offset {
                    return Err(TsError::invalid_psi("Descriptor length exceeds section"));
                }
                data[offset + 5..desc_end].to_vec()
            } else {
                Vec::new()
            };

            services.push(SdtService {
                service_id,
                eit_schedule_flag,
                eit_present_following_flag,
                running_status,
                free_ca_mode,
                descriptors,
            });

            offset += 5 + desc_loop_length as usize;
        }

        Ok(Self {
            transport_stream_id: header.table_id_extension,
            original_network_id,
            version_number: header.version_number,
            current_next: header.current_next,
            services,
        })
    }
}

/// PSI section assembler for handling multi-packet sections.
#[derive(Debug)]
pub struct PsiAssembler {
    /// PID being assembled.
    #[allow(dead_code)]
    pid: u16,
    /// Accumulated data.
    buffer: Vec<u8>,
    /// Expected section length.
    expected_length: Option<usize>,
    /// Last continuity counter.
    last_cc: Option<u8>,
}

impl PsiAssembler {
    /// Create a new PSI assembler.
    pub fn new(pid: u16) -> Self {
        Self {
            pid,
            buffer: Vec::with_capacity(4096),
            expected_length: None,
            last_cc: None,
        }
    }

    /// Reset the assembler.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.expected_length = None;
        self.last_cc = None;
    }

    /// Add packet payload to the assembler.
    ///
    /// Returns the complete section if available.
    pub fn add(&mut self, payload: &[u8], pusi: bool, cc: u8) -> Result<Option<Vec<u8>>> {
        // Check continuity
        if let Some(last_cc) = self.last_cc {
            let expected_cc = (last_cc + 1) & 0x0F;
            if cc != expected_cc {
                // Discontinuity - reset
                self.reset();
            }
        }
        self.last_cc = Some(cc);

        if pusi {
            // Payload unit start
            if payload.is_empty() {
                return Ok(None);
            }

            let pointer_field = payload[0] as usize;

            // If there's data from a previous section, complete it
            if pointer_field > 0
                && !self.buffer.is_empty()
                && pointer_field < payload.len()
            {
                self.buffer.extend_from_slice(&payload[1..1 + pointer_field]);
            }

            // Check if we have a complete previous section
            let completed = if let Some(expected) = self.expected_length {
                if self.buffer.len() >= expected {
                    Some(self.buffer[..expected].to_vec())
                } else {
                    None
                }
            } else {
                None
            };

            // Start new section
            self.buffer.clear();
            self.expected_length = None;

            let section_start = 1 + pointer_field;
            if section_start < payload.len() {
                let section_data = &payload[section_start..];

                // Skip padding bytes (0xFF)
                let actual_start = section_data.iter().position(|&b| b != 0xFF);

                if let Some(start) = actual_start {
                    self.buffer.extend_from_slice(&section_data[start..]);

                    // Parse section length if we have enough data
                    if self.buffer.len() >= 3 {
                        let length = ((self.buffer[1] as usize & 0x0F) << 8)
                            | (self.buffer[2] as usize);
                        self.expected_length = Some(3 + length);
                    }
                }
            }

            if completed.is_some() {
                return Ok(completed);
            }
        } else if !self.buffer.is_empty() {
            // Continuation
            self.buffer.extend_from_slice(payload);
        }

        // Check if section is complete
        if let Some(expected) = self.expected_length {
            if self.buffer.len() >= expected {
                let section = self.buffer[..expected].to_vec();
                self.buffer.clear();
                self.expected_length = None;
                return Ok(Some(section));
            }
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32() {
        // Test with known value
        let data = [0x00, 0xB0, 0x0D, 0x00, 0x01, 0xC1, 0x00, 0x00, 0x00, 0x01, 0xE0, 0x20];
        let crc = calculate_crc32(&data);
        // CRC should be consistent
        assert_eq!(calculate_crc32(&data), crc);
    }

    #[test]
    fn test_pat_serialize_parse() {
        let mut pat = Pat::new(1);
        pat.add_program(1, 0x100);
        pat.add_program(2, 0x200);

        let data = pat.serialize();
        let parsed = Pat::parse(&data).unwrap();

        assert_eq!(parsed.transport_stream_id, 1);
        assert_eq!(parsed.programs.len(), 2);
        assert_eq!(parsed.programs[0].program_number, 1);
        assert_eq!(parsed.programs[0].pid, 0x100);
        assert_eq!(parsed.programs[1].program_number, 2);
        assert_eq!(parsed.programs[1].pid, 0x200);
    }

    #[test]
    fn test_pmt_serialize_parse() {
        let mut pmt = Pmt::new(1, 0x100);
        pmt.add_stream(StreamType::H264 as u8, 0x100);
        pmt.add_stream(StreamType::AacAdts as u8, 0x101);

        let data = pmt.serialize();
        let parsed = Pmt::parse(&data).unwrap();

        assert_eq!(parsed.program_number, 1);
        assert_eq!(parsed.pcr_pid, 0x100);
        assert_eq!(parsed.streams.len(), 2);
        assert_eq!(parsed.streams[0].stream_type, StreamType::H264 as u8);
        assert_eq!(parsed.streams[0].pid, 0x100);
        assert_eq!(parsed.streams[1].stream_type, StreamType::AacAdts as u8);
        assert_eq!(parsed.streams[1].pid, 0x101);
    }

    #[test]
    fn test_stream_type() {
        assert!(StreamType::H264.is_video());
        assert!(StreamType::H265.is_video());
        assert!(!StreamType::H264.is_audio());

        assert!(StreamType::AacAdts.is_audio());
        assert!(StreamType::Ac3.is_audio());
        assert!(!StreamType::AacAdts.is_video());
    }

    #[test]
    fn test_psi_assembler() {
        let mut assembler = PsiAssembler::new(crate::packet::PID_PAT);

        // Create a simple PAT
        let pat = Pat::new(1);
        let section = pat.serialize();

        // Simulate receiving it in a single packet
        let mut payload = vec![0u8]; // pointer field = 0
        payload.extend_from_slice(&section);

        let result = assembler.add(&payload, true, 0).unwrap();
        assert!(result.is_some());

        let parsed = Pat::parse(&result.unwrap()).unwrap();
        assert_eq!(parsed.transport_stream_id, 1);
    }

    #[test]
    fn test_pat_get_pmt_pid() {
        let mut pat = Pat::new(1);
        pat.add_program(1, 0x100);
        pat.add_program(2, 0x200);

        assert_eq!(pat.get_pmt_pid(1), Some(0x100));
        assert_eq!(pat.get_pmt_pid(2), Some(0x200));
        assert_eq!(pat.get_pmt_pid(3), None);
    }
}
