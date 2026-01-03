//! FLV tag types and parsing.
//!
//! FLV files contain tags of three types:
//! - Audio tags (type 8)
//! - Video tags (type 9)
//! - Script data tags (type 18)
//!
//! Each tag has an 11-byte header followed by data and a 4-byte previous tag size.

use crate::error::{FlvError, Result};
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Write};

/// Audio tag type.
pub const TAG_TYPE_AUDIO: u8 = 8;

/// Video tag type.
pub const TAG_TYPE_VIDEO: u8 = 9;

/// Script data tag type.
pub const TAG_TYPE_SCRIPT_DATA: u8 = 18;

/// FLV tag header size.
pub const TAG_HEADER_SIZE: usize = 11;

/// Maximum tag data size (16 MB - 1).
pub const MAX_TAG_DATA_SIZE: u32 = 0x00FF_FFFF;

/// FLV tag type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TagType {
    /// Audio data.
    Audio = TAG_TYPE_AUDIO,
    /// Video data.
    Video = TAG_TYPE_VIDEO,
    /// Script data (metadata, etc.).
    ScriptData = TAG_TYPE_SCRIPT_DATA,
}

impl TagType {
    /// Create a TagType from a raw byte value.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            TAG_TYPE_AUDIO => Some(Self::Audio),
            TAG_TYPE_VIDEO => Some(Self::Video),
            TAG_TYPE_SCRIPT_DATA => Some(Self::ScriptData),
            _ => None,
        }
    }

    /// Convert to raw byte value.
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Check if this is an audio tag.
    pub fn is_audio(self) -> bool {
        self == Self::Audio
    }

    /// Check if this is a video tag.
    pub fn is_video(self) -> bool {
        self == Self::Video
    }

    /// Check if this is a script data tag.
    pub fn is_script_data(self) -> bool {
        self == Self::ScriptData
    }
}

/// FLV tag header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagHeader {
    /// Tag type.
    pub tag_type: TagType,
    /// Data size (not including header).
    pub data_size: u32,
    /// Timestamp in milliseconds (lower 24 bits).
    pub timestamp: u32,
    /// Timestamp extended (upper 8 bits).
    pub timestamp_extended: u8,
    /// Stream ID (always 0).
    pub stream_id: u32,
}

impl TagHeader {
    /// Create a new tag header.
    pub fn new(tag_type: TagType, data_size: u32, timestamp_ms: u32) -> Self {
        Self {
            tag_type,
            data_size,
            timestamp: timestamp_ms & 0x00FF_FFFF,
            timestamp_extended: ((timestamp_ms >> 24) & 0xFF) as u8,
            stream_id: 0,
        }
    }

    /// Create an audio tag header.
    pub fn audio(data_size: u32, timestamp_ms: u32) -> Self {
        Self::new(TagType::Audio, data_size, timestamp_ms)
    }

    /// Create a video tag header.
    pub fn video(data_size: u32, timestamp_ms: u32) -> Self {
        Self::new(TagType::Video, data_size, timestamp_ms)
    }

    /// Create a script data tag header.
    pub fn script_data(data_size: u32, timestamp_ms: u32) -> Self {
        Self::new(TagType::ScriptData, data_size, timestamp_ms)
    }

    /// Get the full 32-bit timestamp.
    pub fn timestamp_ms(&self) -> u32 {
        ((self.timestamp_extended as u32) << 24) | self.timestamp
    }

    /// Set the full 32-bit timestamp.
    pub fn set_timestamp_ms(&mut self, timestamp_ms: u32) {
        self.timestamp = timestamp_ms & 0x00FF_FFFF;
        self.timestamp_extended = ((timestamp_ms >> 24) & 0xFF) as u8;
    }

    /// Get the total tag size including header and previous tag size.
    pub fn total_size(&self) -> u32 {
        TAG_HEADER_SIZE as u32 + self.data_size + 4
    }

    /// Parse a tag header from a reader.
    pub fn parse<R: Read>(reader: &mut R) -> Result<Self> {
        // Read tag type
        let tag_type_byte = reader.read_u8()?;
        let tag_type = TagType::from_u8(tag_type_byte)
            .ok_or(FlvError::InvalidTagType(tag_type_byte))?;

        // Read data size (3 bytes)
        let mut size_bytes = [0u8; 3];
        reader.read_exact(&mut size_bytes)?;
        let data_size =
            ((size_bytes[0] as u32) << 16) | ((size_bytes[1] as u32) << 8) | (size_bytes[2] as u32);

        // Read timestamp (3 bytes) and timestamp extended (1 byte)
        let mut ts_bytes = [0u8; 3];
        reader.read_exact(&mut ts_bytes)?;
        let timestamp =
            ((ts_bytes[0] as u32) << 16) | ((ts_bytes[1] as u32) << 8) | (ts_bytes[2] as u32);
        let timestamp_extended = reader.read_u8()?;

        // Read stream ID (3 bytes)
        let mut stream_id_bytes = [0u8; 3];
        reader.read_exact(&mut stream_id_bytes)?;
        let stream_id = ((stream_id_bytes[0] as u32) << 16)
            | ((stream_id_bytes[1] as u32) << 8)
            | (stream_id_bytes[2] as u32);

        Ok(Self {
            tag_type,
            data_size,
            timestamp,
            timestamp_extended,
            stream_id,
        })
    }

    /// Write the tag header to a writer.
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<usize> {
        // Write tag type
        writer.write_u8(self.tag_type.as_u8())?;

        // Write data size (3 bytes)
        writer.write_u8(((self.data_size >> 16) & 0xFF) as u8)?;
        writer.write_u8(((self.data_size >> 8) & 0xFF) as u8)?;
        writer.write_u8((self.data_size & 0xFF) as u8)?;

        // Write timestamp (3 bytes)
        writer.write_u8(((self.timestamp >> 16) & 0xFF) as u8)?;
        writer.write_u8(((self.timestamp >> 8) & 0xFF) as u8)?;
        writer.write_u8((self.timestamp & 0xFF) as u8)?;

        // Write timestamp extended
        writer.write_u8(self.timestamp_extended)?;

        // Write stream ID (3 bytes, always 0)
        writer.write_u8(0)?;
        writer.write_u8(0)?;
        writer.write_u8(0)?;

        Ok(TAG_HEADER_SIZE)
    }
}

/// A complete FLV tag including header and data.
#[derive(Debug, Clone)]
pub struct FlvTag {
    /// Tag header.
    pub header: TagHeader,
    /// Tag data.
    pub data: Vec<u8>,
}

impl FlvTag {
    /// Create a new FLV tag.
    pub fn new(tag_type: TagType, timestamp_ms: u32, data: Vec<u8>) -> Self {
        Self {
            header: TagHeader::new(tag_type, data.len() as u32, timestamp_ms),
            data,
        }
    }

    /// Create an audio tag.
    pub fn audio(timestamp_ms: u32, data: Vec<u8>) -> Self {
        Self::new(TagType::Audio, timestamp_ms, data)
    }

    /// Create a video tag.
    pub fn video(timestamp_ms: u32, data: Vec<u8>) -> Self {
        Self::new(TagType::Video, timestamp_ms, data)
    }

    /// Create a script data tag.
    pub fn script_data(timestamp_ms: u32, data: Vec<u8>) -> Self {
        Self::new(TagType::ScriptData, timestamp_ms, data)
    }

    /// Get the tag type.
    pub fn tag_type(&self) -> TagType {
        self.header.tag_type
    }

    /// Get the timestamp in milliseconds.
    pub fn timestamp_ms(&self) -> u32 {
        self.header.timestamp_ms()
    }

    /// Get the data size.
    pub fn data_size(&self) -> u32 {
        self.data.len() as u32
    }

    /// Get the total tag size including header and previous tag size.
    pub fn total_size(&self) -> u32 {
        TAG_HEADER_SIZE as u32 + self.data.len() as u32 + 4
    }

    /// Parse a complete FLV tag from a reader.
    pub fn parse<R: Read>(reader: &mut R) -> Result<Self> {
        let header = TagHeader::parse(reader)?;

        // Validate data size
        if header.data_size > MAX_TAG_DATA_SIZE {
            return Err(FlvError::InvalidTagSize {
                offset: 0,
                message: format!("Data size {} exceeds maximum {}", header.data_size, MAX_TAG_DATA_SIZE),
            });
        }

        // Read data
        let mut data = vec![0u8; header.data_size as usize];
        reader.read_exact(&mut data)?;

        Ok(Self { header, data })
    }

    /// Write the complete FLV tag to a writer.
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<usize> {
        let header_size = self.header.write(writer)?;
        writer.write_all(&self.data)?;
        Ok(header_size + self.data.len())
    }

    /// Write the previous tag size to a writer.
    pub fn write_previous_tag_size<W: Write>(&self, writer: &mut W) -> Result<()> {
        let prev_tag_size = TAG_HEADER_SIZE as u32 + self.data.len() as u32;
        writer.write_u32::<BigEndian>(prev_tag_size)?;
        Ok(())
    }
}

/// Read the previous tag size from a reader.
pub fn read_previous_tag_size<R: Read>(reader: &mut R) -> Result<u32> {
    Ok(reader.read_u32::<BigEndian>()?)
}

/// Write a previous tag size to a writer.
pub fn write_previous_tag_size<W: Write>(writer: &mut W, size: u32) -> Result<()> {
    writer.write_u32::<BigEndian>(size)?;
    Ok(())
}

/// Extended timestamp handling for timestamps > 0xFFFFFF ms (~4.66 hours).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtendedTimestamp {
    /// Base timestamp (lower 24 bits).
    pub base: u32,
    /// Extended part (upper 8 bits).
    pub extended: u8,
}

impl ExtendedTimestamp {
    /// Create from a 32-bit millisecond timestamp.
    pub fn from_ms(timestamp_ms: u32) -> Self {
        Self {
            base: timestamp_ms & 0x00FF_FFFF,
            extended: ((timestamp_ms >> 24) & 0xFF) as u8,
        }
    }

    /// Convert to a 32-bit millisecond timestamp.
    pub fn to_ms(self) -> u32 {
        ((self.extended as u32) << 24) | self.base
    }

    /// Check if this timestamp uses the extended field.
    pub fn is_extended(self) -> bool {
        self.extended != 0
    }
}

impl From<u32> for ExtendedTimestamp {
    fn from(ms: u32) -> Self {
        Self::from_ms(ms)
    }
}

impl From<ExtendedTimestamp> for u32 {
    fn from(ts: ExtendedTimestamp) -> u32 {
        ts.to_ms()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_tag_type_from_u8() {
        assert_eq!(TagType::from_u8(8), Some(TagType::Audio));
        assert_eq!(TagType::from_u8(9), Some(TagType::Video));
        assert_eq!(TagType::from_u8(18), Some(TagType::ScriptData));
        assert_eq!(TagType::from_u8(99), None);
    }

    #[test]
    fn test_tag_type_checks() {
        assert!(TagType::Audio.is_audio());
        assert!(!TagType::Audio.is_video());
        assert!(TagType::Video.is_video());
        assert!(!TagType::Video.is_audio());
        assert!(TagType::ScriptData.is_script_data());
    }

    #[test]
    fn test_tag_header_new() {
        let header = TagHeader::new(TagType::Video, 1000, 5000);
        assert_eq!(header.tag_type, TagType::Video);
        assert_eq!(header.data_size, 1000);
        assert_eq!(header.timestamp_ms(), 5000);
        assert_eq!(header.stream_id, 0);
    }

    #[test]
    fn test_tag_header_extended_timestamp() {
        // Test with timestamp requiring extended byte
        let timestamp_ms = 0x12345678;
        let header = TagHeader::new(TagType::Audio, 100, timestamp_ms);

        assert_eq!(header.timestamp, 0x345678);
        assert_eq!(header.timestamp_extended, 0x12);
        assert_eq!(header.timestamp_ms(), timestamp_ms);
    }

    #[test]
    fn test_tag_header_roundtrip() {
        let original = TagHeader::new(TagType::Video, 12345, 0x12345678);

        let mut buffer = Vec::new();
        original.write(&mut buffer).unwrap();

        assert_eq!(buffer.len(), TAG_HEADER_SIZE);

        let mut cursor = Cursor::new(&buffer);
        let parsed = TagHeader::parse(&mut cursor).unwrap();

        assert_eq!(original.tag_type, parsed.tag_type);
        assert_eq!(original.data_size, parsed.data_size);
        assert_eq!(original.timestamp_ms(), parsed.timestamp_ms());
        assert_eq!(original.stream_id, parsed.stream_id);
    }

    #[test]
    fn test_flv_tag_new() {
        let data = vec![0xAB; 100];
        let tag = FlvTag::new(TagType::Audio, 1000, data.clone());

        assert_eq!(tag.tag_type(), TagType::Audio);
        assert_eq!(tag.timestamp_ms(), 1000);
        assert_eq!(tag.data, data);
        assert_eq!(tag.data_size(), 100);
    }

    #[test]
    fn test_flv_tag_roundtrip() {
        let original_data = vec![0x01, 0x02, 0x03, 0x04, 0x05];
        let original = FlvTag::video(5000, original_data.clone());

        let mut buffer = Vec::new();
        original.write(&mut buffer).unwrap();

        let mut cursor = Cursor::new(&buffer);
        let parsed = FlvTag::parse(&mut cursor).unwrap();

        assert_eq!(parsed.tag_type(), TagType::Video);
        assert_eq!(parsed.timestamp_ms(), 5000);
        assert_eq!(parsed.data, original_data);
    }

    #[test]
    fn test_flv_tag_total_size() {
        let tag = FlvTag::audio(0, vec![0u8; 100]);
        // 11 (header) + 100 (data) + 4 (prev tag size) = 115
        assert_eq!(tag.total_size(), 115);
    }

    #[test]
    fn test_previous_tag_size_roundtrip() {
        let mut buffer = Vec::new();
        write_previous_tag_size(&mut buffer, 12345).unwrap();

        let mut cursor = Cursor::new(&buffer);
        let size = read_previous_tag_size(&mut cursor).unwrap();

        assert_eq!(size, 12345);
    }

    #[test]
    fn test_extended_timestamp() {
        let ts = ExtendedTimestamp::from_ms(0x12345678);
        assert_eq!(ts.base, 0x345678);
        assert_eq!(ts.extended, 0x12);
        assert_eq!(ts.to_ms(), 0x12345678);
        assert!(ts.is_extended());

        let ts = ExtendedTimestamp::from_ms(0x00123456);
        assert_eq!(ts.base, 0x123456);
        assert_eq!(ts.extended, 0x00);
        assert_eq!(ts.to_ms(), 0x00123456);
        assert!(!ts.is_extended());
    }

    #[test]
    fn test_tag_type_as_u8() {
        assert_eq!(TagType::Audio.as_u8(), 8);
        assert_eq!(TagType::Video.as_u8(), 9);
        assert_eq!(TagType::ScriptData.as_u8(), 18);
    }

    #[test]
    fn test_tag_header_helpers() {
        let audio = TagHeader::audio(100, 1000);
        assert_eq!(audio.tag_type, TagType::Audio);

        let video = TagHeader::video(200, 2000);
        assert_eq!(video.tag_type, TagType::Video);

        let script = TagHeader::script_data(300, 3000);
        assert_eq!(script.tag_type, TagType::ScriptData);
    }

    #[test]
    fn test_invalid_tag_type() {
        let data = [
            99, // Invalid tag type
            0, 0, 10, // Data size
            0, 0, 0, 0, // Timestamp
            0, 0, 0, // Stream ID
        ];
        let mut cursor = Cursor::new(&data);

        let result = TagHeader::parse(&mut cursor);
        assert!(matches!(result, Err(FlvError::InvalidTagType(99))));
    }
}
