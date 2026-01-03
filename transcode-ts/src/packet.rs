//! MPEG Transport Stream packet implementation.
//!
//! This module provides types for working with 188-byte MPEG-TS packets,
//! including parsing, creation, and manipulation of packet headers and
//! adaptation fields.

use crate::error::{Result, TsError};

/// MPEG-TS packet size in bytes.
pub const TS_PACKET_SIZE: usize = 188;

/// MPEG-TS sync byte value.
pub const SYNC_BYTE: u8 = 0x47;

/// Null packet PID.
pub const PID_NULL: u16 = 0x1FFF;

/// PAT (Program Association Table) PID.
pub const PID_PAT: u16 = 0x0000;

/// CAT (Conditional Access Table) PID.
pub const PID_CAT: u16 = 0x0001;

/// TSDT (Transport Stream Description Table) PID.
pub const PID_TSDT: u16 = 0x0002;

/// SDT/BAT (Service Description Table) PID.
pub const PID_SDT: u16 = 0x0011;

/// EIT (Event Information Table) PID.
pub const PID_EIT: u16 = 0x0012;

/// Maximum valid PID value.
pub const PID_MAX: u16 = 0x1FFF;

/// Adaptation field control values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptationFieldControl {
    /// Reserved for future use.
    Reserved,
    /// Payload only.
    PayloadOnly,
    /// Adaptation field only.
    AdaptationFieldOnly,
    /// Adaptation field followed by payload.
    AdaptationFieldAndPayload,
}

impl AdaptationFieldControl {
    /// Parse from 2-bit value.
    pub fn from_bits(bits: u8) -> Self {
        match bits & 0x03 {
            0 => AdaptationFieldControl::Reserved,
            1 => AdaptationFieldControl::PayloadOnly,
            2 => AdaptationFieldControl::AdaptationFieldOnly,
            3 => AdaptationFieldControl::AdaptationFieldAndPayload,
            _ => unreachable!(),
        }
    }

    /// Convert to 2-bit value.
    pub fn to_bits(self) -> u8 {
        match self {
            AdaptationFieldControl::Reserved => 0,
            AdaptationFieldControl::PayloadOnly => 1,
            AdaptationFieldControl::AdaptationFieldOnly => 2,
            AdaptationFieldControl::AdaptationFieldAndPayload => 3,
        }
    }

    /// Check if packet has adaptation field.
    pub fn has_adaptation_field(self) -> bool {
        matches!(
            self,
            AdaptationFieldControl::AdaptationFieldOnly
                | AdaptationFieldControl::AdaptationFieldAndPayload
        )
    }

    /// Check if packet has payload.
    pub fn has_payload(self) -> bool {
        matches!(
            self,
            AdaptationFieldControl::PayloadOnly | AdaptationFieldControl::AdaptationFieldAndPayload
        )
    }
}

/// Scrambling control values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScramblingControl {
    /// Not scrambled.
    #[default]
    NotScrambled,
    /// User defined (even key).
    UserDefinedEven,
    /// User defined (odd key).
    UserDefinedOdd,
    /// Reserved.
    Reserved,
}

impl ScramblingControl {
    /// Parse from 2-bit value.
    pub fn from_bits(bits: u8) -> Self {
        match bits & 0x03 {
            0 => ScramblingControl::NotScrambled,
            1 => ScramblingControl::Reserved,
            2 => ScramblingControl::UserDefinedEven,
            3 => ScramblingControl::UserDefinedOdd,
            _ => unreachable!(),
        }
    }

    /// Convert to 2-bit value.
    pub fn to_bits(self) -> u8 {
        match self {
            ScramblingControl::NotScrambled => 0,
            ScramblingControl::Reserved => 1,
            ScramblingControl::UserDefinedEven => 2,
            ScramblingControl::UserDefinedOdd => 3,
        }
    }
}

/// Program Clock Reference (PCR).
///
/// PCR is a 42-bit value (33-bit base + 9-bit extension) encoded in 6 bytes.
/// It represents a 27 MHz clock value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pcr {
    /// 33-bit base value (90 kHz clock).
    pub base: u64,
    /// 9-bit extension (27 MHz subdivisions).
    pub extension: u16,
}

impl Pcr {
    /// PCR clock frequency (27 MHz).
    pub const CLOCK_RATE: u64 = 27_000_000;

    /// PCR base clock frequency (90 kHz).
    pub const BASE_CLOCK_RATE: u64 = 90_000;

    /// Create a new PCR from base and extension.
    pub fn new(base: u64, extension: u16) -> Self {
        Self {
            base: base & 0x1_FFFF_FFFF, // 33 bits
            extension: extension & 0x1FF, // 9 bits
        }
    }

    /// Create a PCR from a 27 MHz timestamp.
    pub fn from_27mhz(value: u64) -> Self {
        let base = value / 300;
        let extension = (value % 300) as u16;
        Self::new(base, extension)
    }

    /// Convert to a 27 MHz timestamp.
    pub fn to_27mhz(&self) -> u64 {
        self.base * 300 + self.extension as u64
    }

    /// Convert to seconds as f64.
    pub fn to_seconds(&self) -> f64 {
        self.to_27mhz() as f64 / Self::CLOCK_RATE as f64
    }

    /// Create from seconds.
    pub fn from_seconds(seconds: f64) -> Self {
        Self::from_27mhz((seconds * Self::CLOCK_RATE as f64) as u64)
    }

    /// Parse PCR from 6 bytes.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 6 {
            return Err(TsError::InvalidAdaptationField(
                "PCR requires 6 bytes".to_string(),
            ));
        }

        // PCR format:
        // - bytes 0-3: base[32:1]
        // - byte 4: base[0], reserved(6), extension[8]
        // - byte 5: extension[7:0]
        let base = ((data[0] as u64) << 25)
            | ((data[1] as u64) << 17)
            | ((data[2] as u64) << 9)
            | ((data[3] as u64) << 1)
            | ((data[4] as u64) >> 7);

        let extension = (((data[4] & 0x01) as u16) << 8) | (data[5] as u16);

        Ok(Self::new(base, extension))
    }

    /// Write PCR to 6 bytes.
    pub fn write(&self, data: &mut [u8]) -> Result<()> {
        if data.len() < 6 {
            return Err(TsError::BufferOverflow(
                "PCR requires 6 bytes".to_string(),
            ));
        }

        data[0] = (self.base >> 25) as u8;
        data[1] = (self.base >> 17) as u8;
        data[2] = (self.base >> 9) as u8;
        data[3] = (self.base >> 1) as u8;
        data[4] = ((self.base & 0x01) << 7) as u8 | 0x7E | ((self.extension >> 8) & 0x01) as u8;
        data[5] = (self.extension & 0xFF) as u8;

        Ok(())
    }
}

/// Adaptation field parsed from a TS packet.
#[derive(Debug, Clone, Default)]
pub struct AdaptationField {
    /// Adaptation field length (excluding length byte).
    pub length: u8,
    /// Discontinuity indicator.
    pub discontinuity: bool,
    /// Random access indicator (keyframe).
    pub random_access: bool,
    /// Elementary stream priority indicator.
    pub es_priority: bool,
    /// PCR flag.
    pub pcr_flag: bool,
    /// OPCR flag.
    pub opcr_flag: bool,
    /// Splicing point flag.
    pub splicing_point_flag: bool,
    /// Transport private data flag.
    pub transport_private_data_flag: bool,
    /// Adaptation field extension flag.
    pub adaptation_extension_flag: bool,
    /// Program Clock Reference (if present).
    pub pcr: Option<Pcr>,
    /// Original Program Clock Reference (if present).
    pub opcr: Option<Pcr>,
    /// Splice countdown (if present).
    pub splice_countdown: Option<i8>,
    /// Stuffing bytes count.
    pub stuffing_bytes: usize,
}

impl AdaptationField {
    /// Parse adaptation field from packet data.
    ///
    /// The `data` slice should start at the adaptation field length byte.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(TsError::InvalidAdaptationField(
                "Empty adaptation field".to_string(),
            ));
        }

        let length = data[0];

        if length == 0 {
            return Ok(Self {
                length: 0,
                ..Default::default()
            });
        }

        if data.len() < length as usize + 1 {
            return Err(TsError::InvalidAdaptationField(format!(
                "Adaptation field length {} exceeds available data {}",
                length,
                data.len() - 1
            )));
        }

        let flags = data[1];
        let discontinuity = (flags & 0x80) != 0;
        let random_access = (flags & 0x40) != 0;
        let es_priority = (flags & 0x20) != 0;
        let pcr_flag = (flags & 0x10) != 0;
        let opcr_flag = (flags & 0x08) != 0;
        let splicing_point_flag = (flags & 0x04) != 0;
        let transport_private_data_flag = (flags & 0x02) != 0;
        let adaptation_extension_flag = (flags & 0x01) != 0;

        let mut offset = 2;

        let pcr = if pcr_flag {
            if offset + 6 > data.len() {
                return Err(TsError::InvalidAdaptationField(
                    "Truncated PCR".to_string(),
                ));
            }
            let pcr = Pcr::parse(&data[offset..offset + 6])?;
            offset += 6;
            Some(pcr)
        } else {
            None
        };

        let opcr = if opcr_flag {
            if offset + 6 > data.len() {
                return Err(TsError::InvalidAdaptationField(
                    "Truncated OPCR".to_string(),
                ));
            }
            let opcr = Pcr::parse(&data[offset..offset + 6])?;
            offset += 6;
            Some(opcr)
        } else {
            None
        };

        let splice_countdown = if splicing_point_flag {
            if offset >= data.len() {
                return Err(TsError::InvalidAdaptationField(
                    "Truncated splice countdown".to_string(),
                ));
            }
            let countdown = data[offset] as i8;
            offset += 1;
            Some(countdown)
        } else {
            None
        };

        // Skip transport private data if present
        if transport_private_data_flag && offset < data.len() {
            let private_data_length = data[offset] as usize;
            offset += 1 + private_data_length;
        }

        // Skip adaptation extension if present
        if adaptation_extension_flag && offset < data.len() {
            let extension_length = data[offset] as usize;
            offset += 1 + extension_length;
        }

        // Remaining bytes are stuffing
        let stuffing_bytes = (length as usize + 1).saturating_sub(offset);

        Ok(Self {
            length,
            discontinuity,
            random_access,
            es_priority,
            pcr_flag,
            opcr_flag,
            splicing_point_flag,
            transport_private_data_flag,
            adaptation_extension_flag,
            pcr,
            opcr,
            splice_countdown,
            stuffing_bytes,
        })
    }

    /// Calculate the total size of the adaptation field including length byte.
    pub fn total_size(&self) -> usize {
        self.length as usize + 1
    }

    /// Create a minimal adaptation field for stuffing.
    pub fn stuffing(size: usize) -> Self {
        Self {
            length: (size.saturating_sub(1)) as u8,
            stuffing_bytes: size.saturating_sub(2),
            ..Default::default()
        }
    }

    /// Create an adaptation field with PCR.
    pub fn with_pcr(pcr: Pcr) -> Self {
        Self {
            length: 7, // 1 byte flags + 6 bytes PCR
            pcr_flag: true,
            pcr: Some(pcr),
            ..Default::default()
        }
    }

    /// Write adaptation field to buffer.
    pub fn write(&self, data: &mut [u8]) -> Result<usize> {
        if data.is_empty() {
            return Err(TsError::BufferOverflow(
                "No space for adaptation field".to_string(),
            ));
        }

        data[0] = self.length;

        if self.length == 0 {
            return Ok(1);
        }

        if data.len() < self.total_size() {
            return Err(TsError::BufferOverflow(
                "Insufficient space for adaptation field".to_string(),
            ));
        }

        let mut flags = 0u8;
        if self.discontinuity {
            flags |= 0x80;
        }
        if self.random_access {
            flags |= 0x40;
        }
        if self.es_priority {
            flags |= 0x20;
        }
        if self.pcr_flag {
            flags |= 0x10;
        }
        if self.opcr_flag {
            flags |= 0x08;
        }
        if self.splicing_point_flag {
            flags |= 0x04;
        }
        if self.transport_private_data_flag {
            flags |= 0x02;
        }
        if self.adaptation_extension_flag {
            flags |= 0x01;
        }

        data[1] = flags;
        let mut offset = 2;

        if let Some(ref pcr) = self.pcr {
            pcr.write(&mut data[offset..offset + 6])?;
            offset += 6;
        }

        if let Some(ref opcr) = self.opcr {
            opcr.write(&mut data[offset..offset + 6])?;
            offset += 6;
        }

        if let Some(countdown) = self.splice_countdown {
            data[offset] = countdown as u8;
            offset += 1;
        }

        // Fill stuffing bytes with 0xFF
        let total_size = self.total_size();
        for byte in data[offset..total_size].iter_mut() {
            *byte = 0xFF;
        }

        Ok(total_size)
    }
}

/// MPEG Transport Stream packet header.
#[derive(Debug, Clone)]
pub struct TsHeader {
    /// Transport error indicator.
    pub transport_error: bool,
    /// Payload unit start indicator.
    pub payload_unit_start: bool,
    /// Transport priority.
    pub transport_priority: bool,
    /// Packet Identifier (13 bits).
    pub pid: u16,
    /// Scrambling control.
    pub scrambling_control: ScramblingControl,
    /// Adaptation field control.
    pub adaptation_field_control: AdaptationFieldControl,
    /// Continuity counter (4 bits).
    pub continuity_counter: u8,
}

impl TsHeader {
    /// Header size in bytes.
    pub const SIZE: usize = 4;

    /// Create a new header with default values.
    pub fn new(pid: u16) -> Self {
        Self {
            transport_error: false,
            payload_unit_start: false,
            transport_priority: false,
            pid: pid & PID_MAX,
            scrambling_control: ScramblingControl::NotScrambled,
            adaptation_field_control: AdaptationFieldControl::PayloadOnly,
            continuity_counter: 0,
        }
    }

    /// Parse header from 4 bytes.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            return Err(TsError::PacketTooShort(data.len()));
        }

        if data[0] != SYNC_BYTE {
            return Err(TsError::InvalidSyncByte(data[0]));
        }

        let transport_error = (data[1] & 0x80) != 0;
        let payload_unit_start = (data[1] & 0x40) != 0;
        let transport_priority = (data[1] & 0x20) != 0;
        let pid = ((data[1] as u16 & 0x1F) << 8) | (data[2] as u16);
        let scrambling_control = ScramblingControl::from_bits(data[3] >> 6);
        let adaptation_field_control = AdaptationFieldControl::from_bits(data[3] >> 4);
        let continuity_counter = data[3] & 0x0F;

        Ok(Self {
            transport_error,
            payload_unit_start,
            transport_priority,
            pid,
            scrambling_control,
            adaptation_field_control,
            continuity_counter,
        })
    }

    /// Write header to 4 bytes.
    pub fn write(&self, data: &mut [u8]) -> Result<()> {
        if data.len() < 4 {
            return Err(TsError::BufferOverflow(
                "Need 4 bytes for header".to_string(),
            ));
        }

        data[0] = SYNC_BYTE;
        data[1] = ((self.transport_error as u8) << 7)
            | ((self.payload_unit_start as u8) << 6)
            | ((self.transport_priority as u8) << 5)
            | ((self.pid >> 8) as u8 & 0x1F);
        data[2] = (self.pid & 0xFF) as u8;
        data[3] = (self.scrambling_control.to_bits() << 6)
            | (self.adaptation_field_control.to_bits() << 4)
            | (self.continuity_counter & 0x0F);

        Ok(())
    }
}

impl Default for TsHeader {
    fn default() -> Self {
        Self::new(PID_NULL)
    }
}

/// A complete 188-byte MPEG Transport Stream packet.
#[derive(Debug, Clone)]
pub struct TsPacket {
    /// Packet data (188 bytes).
    data: [u8; TS_PACKET_SIZE],
}

impl TsPacket {
    /// Create a new packet from raw data.
    pub fn new(data: [u8; TS_PACKET_SIZE]) -> Result<Self> {
        if data[0] != SYNC_BYTE {
            return Err(TsError::InvalidSyncByte(data[0]));
        }
        Ok(Self { data })
    }

    /// Create a packet from a slice.
    pub fn from_slice(data: &[u8]) -> Result<Self> {
        if data.len() < TS_PACKET_SIZE {
            return Err(TsError::PacketTooShort(data.len()));
        }

        let mut packet_data = [0u8; TS_PACKET_SIZE];
        packet_data.copy_from_slice(&data[..TS_PACKET_SIZE]);
        Self::new(packet_data)
    }

    /// Create a null packet.
    pub fn null_packet() -> Self {
        let mut data = [0xFF; TS_PACKET_SIZE];
        data[0] = SYNC_BYTE;
        data[1] = 0x1F; // PID high bits
        data[2] = 0xFF; // PID low bits (0x1FFF = null packet)
        data[3] = 0x10; // Payload only, CC = 0
        Self { data }
    }

    /// Create a new empty packet with header.
    pub fn with_header(header: &TsHeader) -> Result<Self> {
        let mut data = [0xFF; TS_PACKET_SIZE];
        header.write(&mut data[..4])?;
        Ok(Self { data })
    }

    /// Get the raw packet data.
    pub fn data(&self) -> &[u8; TS_PACKET_SIZE] {
        &self.data
    }

    /// Get mutable access to raw packet data.
    pub fn data_mut(&mut self) -> &mut [u8; TS_PACKET_SIZE] {
        &mut self.data
    }

    /// Parse the header.
    pub fn header(&self) -> Result<TsHeader> {
        TsHeader::parse(&self.data)
    }

    /// Get the PID.
    pub fn pid(&self) -> u16 {
        ((self.data[1] as u16 & 0x1F) << 8) | (self.data[2] as u16)
    }

    /// Check if this is a null packet.
    pub fn is_null(&self) -> bool {
        self.pid() == PID_NULL
    }

    /// Get the continuity counter.
    pub fn continuity_counter(&self) -> u8 {
        self.data[3] & 0x0F
    }

    /// Set the continuity counter.
    pub fn set_continuity_counter(&mut self, cc: u8) {
        self.data[3] = (self.data[3] & 0xF0) | (cc & 0x0F);
    }

    /// Check if payload unit start is set.
    pub fn payload_unit_start(&self) -> bool {
        (self.data[1] & 0x40) != 0
    }

    /// Get the adaptation field control.
    pub fn adaptation_field_control(&self) -> AdaptationFieldControl {
        AdaptationFieldControl::from_bits(self.data[3] >> 4)
    }

    /// Check if packet has adaptation field.
    pub fn has_adaptation_field(&self) -> bool {
        self.adaptation_field_control().has_adaptation_field()
    }

    /// Check if packet has payload.
    pub fn has_payload(&self) -> bool {
        self.adaptation_field_control().has_payload()
    }

    /// Parse the adaptation field if present.
    pub fn adaptation_field(&self) -> Result<Option<AdaptationField>> {
        if !self.has_adaptation_field() {
            return Ok(None);
        }
        Ok(Some(AdaptationField::parse(&self.data[4..])?))
    }

    /// Get the payload data.
    pub fn payload(&self) -> Option<&[u8]> {
        if !self.has_payload() {
            return None;
        }

        let payload_start = if self.has_adaptation_field() {
            let af_length = self.data[4] as usize;
            5 + af_length
        } else {
            4
        };

        if payload_start >= TS_PACKET_SIZE {
            None
        } else {
            Some(&self.data[payload_start..])
        }
    }

    /// Get mutable payload data.
    pub fn payload_mut(&mut self) -> Option<&mut [u8]> {
        if !self.has_payload() {
            return None;
        }

        let payload_start = if self.has_adaptation_field() {
            let af_length = self.data[4] as usize;
            5 + af_length
        } else {
            4
        };

        if payload_start >= TS_PACKET_SIZE {
            None
        } else {
            Some(&mut self.data[payload_start..])
        }
    }

    /// Get the payload size.
    pub fn payload_size(&self) -> usize {
        self.payload().map_or(0, |p| p.len())
    }
}

impl Default for TsPacket {
    fn default() -> Self {
        Self::null_packet()
    }
}

impl AsRef<[u8]> for TsPacket {
    fn as_ref(&self) -> &[u8] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ts_header_parse() {
        // Example TS header: sync, PID=256, payload only, CC=5
        let data = [0x47, 0x01, 0x00, 0x15];
        let header = TsHeader::parse(&data).unwrap();

        assert!(!header.transport_error);
        assert!(!header.payload_unit_start);
        assert!(!header.transport_priority);
        assert_eq!(header.pid, 256);
        assert_eq!(
            header.adaptation_field_control,
            AdaptationFieldControl::PayloadOnly
        );
        assert_eq!(header.continuity_counter, 5);
    }

    #[test]
    fn test_ts_header_write() {
        let mut header = TsHeader::new(256);
        header.continuity_counter = 5;
        header.payload_unit_start = true;

        let mut data = [0u8; 4];
        header.write(&mut data).unwrap();

        assert_eq!(data[0], SYNC_BYTE);
        assert_eq!(data[1] & 0x40, 0x40); // PUSI
        assert_eq!(((data[1] as u16 & 0x1F) << 8) | data[2] as u16, 256);
        assert_eq!(data[3] & 0x0F, 5);
    }

    #[test]
    fn test_null_packet() {
        let packet = TsPacket::null_packet();
        assert!(packet.is_null());
        assert_eq!(packet.pid(), PID_NULL);
    }

    #[test]
    fn test_pcr_parse_write() {
        let pcr = Pcr::new(12345678, 123);

        let mut data = [0u8; 6];
        pcr.write(&mut data).unwrap();

        let parsed = Pcr::parse(&data).unwrap();
        assert_eq!(parsed.base, pcr.base);
        assert_eq!(parsed.extension, pcr.extension);
    }

    #[test]
    fn test_pcr_conversion() {
        let seconds = 10.5;
        let pcr = Pcr::from_seconds(seconds);
        let back = pcr.to_seconds();
        assert!((back - seconds).abs() < 0.001);
    }

    #[test]
    fn test_adaptation_field_parse() {
        // Adaptation field with PCR
        let mut data = [0u8; 188 - 4];
        data[0] = 7; // length
        data[1] = 0x50; // random_access + PCR flag

        // Write a PCR
        let pcr = Pcr::new(1000, 50);
        pcr.write(&mut data[2..8]).unwrap();

        let af = AdaptationField::parse(&data).unwrap();
        assert_eq!(af.length, 7);
        assert!(af.random_access);
        assert!(af.pcr_flag);
        assert!(af.pcr.is_some());
    }

    #[test]
    fn test_ts_packet_payload() {
        let mut data = [0u8; TS_PACKET_SIZE];
        data[0] = SYNC_BYTE;
        data[1] = 0x41; // PUSI + PID high
        data[2] = 0x00; // PID low
        data[3] = 0x10; // Payload only

        // Fill payload with pattern
        for (i, byte) in data[4..].iter_mut().enumerate() {
            *byte = (i & 0xFF) as u8;
        }

        let packet = TsPacket::new(data).unwrap();
        assert!(packet.payload_unit_start());
        assert_eq!(packet.pid(), 256);

        let payload = packet.payload().unwrap();
        assert_eq!(payload.len(), 184);
        assert_eq!(payload[0], 0);
        assert_eq!(payload[1], 1);
    }

    #[test]
    fn test_ts_packet_with_adaptation_field() {
        let mut data = [0u8; TS_PACKET_SIZE];
        data[0] = SYNC_BYTE;
        data[1] = 0x01; // PID high
        data[2] = 0x00; // PID low
        data[3] = 0x30; // Adaptation + payload
        data[4] = 7; // AF length
        data[5] = 0x10; // PCR flag

        // Write PCR
        let pcr = Pcr::new(90000, 0);
        pcr.write(&mut data[6..12]).unwrap();

        // Fill remaining with payload
        for byte in data[12..].iter_mut() {
            *byte = 0xAB;
        }

        let packet = TsPacket::new(data).unwrap();
        assert!(packet.has_adaptation_field());
        assert!(packet.has_payload());

        let af = packet.adaptation_field().unwrap().unwrap();
        assert!(af.pcr.is_some());

        let payload = packet.payload().unwrap();
        assert_eq!(payload.len(), 176); // 188 - 4 - 8 (header + AF)
    }

    #[test]
    fn test_invalid_sync_byte() {
        let mut data = [0u8; TS_PACKET_SIZE];
        data[0] = 0x00;

        let result = TsPacket::new(data);
        assert!(matches!(result, Err(TsError::InvalidSyncByte(0x00))));
    }

    #[test]
    fn test_adaptation_field_stuffing() {
        let af = AdaptationField::stuffing(10);
        assert_eq!(af.length, 9);
        assert_eq!(af.total_size(), 10);
    }
}
