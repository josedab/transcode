//! RIFF chunk parsing and writing

use crate::error::{AviError, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read, Write};

/// FourCC (Four Character Code) identifier
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FourCC(pub [u8; 4]);

impl FourCC {
    /// Create from bytes
    pub fn new(bytes: [u8; 4]) -> Self {
        FourCC(bytes)
    }

    /// Create from string (must be 4 bytes)
    pub fn from_str(s: &str) -> Option<Self> {
        if s.len() == 4 {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(s.as_bytes());
            Some(FourCC(bytes))
        } else {
            None
        }
    }

    /// Get as string
    pub fn as_str(&self) -> String {
        String::from_utf8_lossy(&self.0).to_string()
    }

    /// Get raw bytes
    pub fn as_bytes(&self) -> &[u8; 4] {
        &self.0
    }
}

impl std::fmt::Debug for FourCC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FourCC(\"{}\")", self.as_str())
    }
}

impl std::fmt::Display for FourCC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl From<[u8; 4]> for FourCC {
    fn from(bytes: [u8; 4]) -> Self {
        FourCC(bytes)
    }
}

impl From<&[u8; 4]> for FourCC {
    fn from(bytes: &[u8; 4]) -> Self {
        FourCC(*bytes)
    }
}

/// Well-known chunk IDs
pub mod chunk_ids {
    use super::FourCC;

    pub const RIFF: FourCC = FourCC(*b"RIFF");
    pub const AVI: FourCC = FourCC(*b"AVI ");
    pub const AVIX: FourCC = FourCC(*b"AVIX");
    pub const LIST: FourCC = FourCC(*b"LIST");
    pub const HDRL: FourCC = FourCC(*b"hdrl");
    pub const AVIH: FourCC = FourCC(*b"avih");
    pub const STRL: FourCC = FourCC(*b"strl");
    pub const STRH: FourCC = FourCC(*b"strh");
    pub const STRF: FourCC = FourCC(*b"strf");
    pub const STRN: FourCC = FourCC(*b"strn");
    pub const STRD: FourCC = FourCC(*b"strd");
    pub const INDX: FourCC = FourCC(*b"indx");
    pub const MOVI: FourCC = FourCC(*b"movi");
    pub const IDX1: FourCC = FourCC(*b"idx1");
    pub const JUNK: FourCC = FourCC(*b"JUNK");
    pub const INFO: FourCC = FourCC(*b"INFO");
    pub const ISFT: FourCC = FourCC(*b"ISFT");
    pub const INAM: FourCC = FourCC(*b"INAM");
    pub const ICOP: FourCC = FourCC(*b"ICOP");
    pub const ODML: FourCC = FourCC(*b"odml");
    pub const DMLH: FourCC = FourCC(*b"dmlh");
}

/// Chunk identifier with stream number
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkId {
    /// Raw FourCC
    pub fourcc: FourCC,
    /// Stream number (if applicable)
    pub stream_number: Option<u16>,
    /// Chunk type
    pub chunk_type: ChunkType,
}

impl ChunkId {
    /// Parse chunk ID from FourCC
    pub fn parse(fourcc: FourCC) -> Self {
        let bytes = fourcc.as_bytes();

        // Check for stream-specific chunks (e.g., "00dc", "01wb")
        if bytes[0].is_ascii_digit() && bytes[1].is_ascii_digit() {
            let stream_num =
                ((bytes[0] - b'0') as u16) * 10 + ((bytes[1] - b'0') as u16);

            let chunk_type = match &bytes[2..4] {
                b"dc" | b"DC" => ChunkType::VideoCompressed,
                b"db" | b"DB" => ChunkType::VideoUncompressed,
                b"wb" | b"WB" => ChunkType::Audio,
                b"tx" | b"TX" => ChunkType::Text,
                b"ix" | b"IX" => ChunkType::Index,
                b"pc" | b"PC" => ChunkType::PaletteChange,
                _ => ChunkType::Unknown,
            };

            ChunkId {
                fourcc,
                stream_number: Some(stream_num),
                chunk_type,
            }
        } else {
            ChunkId {
                fourcc,
                stream_number: None,
                chunk_type: ChunkType::Unknown,
            }
        }
    }

    /// Create a stream chunk ID
    pub fn stream_chunk(stream_num: u16, chunk_type: ChunkType) -> Self {
        let suffix = match chunk_type {
            ChunkType::VideoCompressed => *b"dc",
            ChunkType::VideoUncompressed => *b"db",
            ChunkType::Audio => *b"wb",
            ChunkType::Text => *b"tx",
            ChunkType::Index => *b"ix",
            ChunkType::PaletteChange => *b"pc",
            ChunkType::Unknown => *b"??",
        };

        let fourcc = FourCC([
            b'0' + (stream_num / 10) as u8,
            b'0' + (stream_num % 10) as u8,
            suffix[0],
            suffix[1],
        ]);

        ChunkId {
            fourcc,
            stream_number: Some(stream_num),
            chunk_type,
        }
    }
}

/// Chunk type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkType {
    /// Compressed video frame
    VideoCompressed,
    /// Uncompressed video frame
    VideoUncompressed,
    /// Audio data
    Audio,
    /// Text/subtitle
    Text,
    /// Index chunk
    Index,
    /// Palette change
    PaletteChange,
    /// Unknown type
    Unknown,
}

/// RIFF chunk
#[derive(Debug, Clone)]
pub struct RiffChunk {
    /// Chunk ID
    pub id: FourCC,
    /// Chunk size (not including header)
    pub size: u32,
    /// Chunk data
    pub data: Vec<u8>,
}

impl RiffChunk {
    /// Create new chunk
    pub fn new(id: FourCC, data: Vec<u8>) -> Self {
        RiffChunk {
            id,
            size: data.len() as u32,
            data,
        }
    }

    /// Read chunk from data
    pub fn read(data: &[u8], offset: usize) -> Result<(Self, usize)> {
        if offset + 8 > data.len() {
            return Err(AviError::InsufficientData {
                needed: 8,
                available: data.len().saturating_sub(offset),
            });
        }

        let mut cursor = Cursor::new(&data[offset..]);
        let mut id_bytes = [0u8; 4];
        cursor.read_exact(&mut id_bytes)?;
        let id = FourCC(id_bytes);

        let size = cursor.read_u32::<LittleEndian>()?;

        // Calculate padded size (RIFF chunks are word-aligned)
        let padded_size = ((size as usize) + 1) & !1;

        if offset + 8 + padded_size > data.len() {
            return Err(AviError::InsufficientData {
                needed: padded_size,
                available: data.len().saturating_sub(offset + 8),
            });
        }

        let chunk_data = data[offset + 8..offset + 8 + size as usize].to_vec();

        Ok((
            RiffChunk {
                id,
                size,
                data: chunk_data,
            },
            offset + 8 + padded_size,
        ))
    }

    /// Write chunk to writer
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(self.id.as_bytes())?;
        writer.write_u32::<LittleEndian>(self.data.len() as u32)?;
        writer.write_all(&self.data)?;

        // Pad to word boundary
        if self.data.len() % 2 != 0 {
            writer.write_all(&[0])?;
        }

        Ok(())
    }

    /// Total size including header and padding
    pub fn total_size(&self) -> usize {
        8 + ((self.data.len() + 1) & !1)
    }
}

/// AVI chunk with parsed type information
#[derive(Debug, Clone)]
pub struct AviChunk {
    /// Chunk identifier
    pub chunk_id: ChunkId,
    /// Raw chunk
    pub raw: RiffChunk,
}

impl AviChunk {
    /// Create from raw chunk
    pub fn from_raw(raw: RiffChunk) -> Self {
        let chunk_id = ChunkId::parse(raw.id);
        AviChunk { chunk_id, raw }
    }

    /// Check if this is a video chunk
    pub fn is_video(&self) -> bool {
        matches!(
            self.chunk_id.chunk_type,
            ChunkType::VideoCompressed | ChunkType::VideoUncompressed
        )
    }

    /// Check if this is an audio chunk
    pub fn is_audio(&self) -> bool {
        self.chunk_id.chunk_type == ChunkType::Audio
    }

    /// Check if this is a keyframe (based on data or flags)
    pub fn is_keyframe(&self) -> bool {
        // For AVI, we typically need external index info
        // But some codecs start keyframes with specific patterns
        if self.is_video() && !self.raw.data.is_empty() {
            // Check for I-frame indicators in common codecs
            // This is a heuristic - proper detection needs codec info
            true // Default to keyframe for safety
        } else {
            false
        }
    }
}

/// LIST chunk (container for other chunks)
#[derive(Debug, Clone)]
pub struct ListChunk {
    /// List type
    pub list_type: FourCC,
    /// Child chunks
    pub chunks: Vec<RiffChunk>,
}

impl ListChunk {
    /// Create new list chunk
    pub fn new(list_type: FourCC) -> Self {
        ListChunk {
            list_type,
            chunks: Vec::new(),
        }
    }

    /// Add a child chunk
    pub fn add_chunk(&mut self, chunk: RiffChunk) {
        self.chunks.push(chunk);
    }

    /// Parse LIST chunk from data
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            return Err(AviError::InsufficientData {
                needed: 4,
                available: data.len(),
            });
        }

        let mut list_type = [0u8; 4];
        list_type.copy_from_slice(&data[0..4]);

        let mut chunks = Vec::new();
        let mut offset = 4;

        while offset + 8 <= data.len() {
            match RiffChunk::read(data, offset) {
                Ok((chunk, next_offset)) => {
                    chunks.push(chunk);
                    offset = next_offset;
                }
                Err(_) => break,
            }
        }

        Ok(ListChunk {
            list_type: FourCC(list_type),
            chunks,
        })
    }

    /// Write LIST chunk
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Calculate total size
        let content_size: usize = 4 + self.chunks.iter().map(|c| c.total_size()).sum::<usize>();

        writer.write_all(b"LIST")?;
        writer.write_u32::<LittleEndian>(content_size as u32)?;
        writer.write_all(self.list_type.as_bytes())?;

        for chunk in &self.chunks {
            chunk.write(writer)?;
        }

        // Pad to word boundary if needed
        if content_size % 2 != 0 {
            writer.write_all(&[0])?;
        }

        Ok(())
    }

    /// Find chunk by ID
    pub fn find_chunk(&self, id: FourCC) -> Option<&RiffChunk> {
        self.chunks.iter().find(|c| c.id == id)
    }

    /// Find all chunks by ID
    pub fn find_chunks(&self, id: FourCC) -> Vec<&RiffChunk> {
        self.chunks.iter().filter(|c| c.id == id).collect()
    }
}

/// AVI index entry (idx1 format)
#[derive(Debug, Clone, Copy)]
pub struct IndexEntry {
    /// Chunk ID
    pub chunk_id: FourCC,
    /// Flags
    pub flags: u32,
    /// Offset from movi list
    pub offset: u32,
    /// Size of chunk data
    pub size: u32,
}

impl IndexEntry {
    /// Index flags
    pub const KEYFRAME: u32 = 0x10;

    /// Read from data
    pub fn read(data: &[u8]) -> Result<Self> {
        if data.len() < 16 {
            return Err(AviError::InsufficientData {
                needed: 16,
                available: data.len(),
            });
        }

        let mut cursor = Cursor::new(data);
        let mut id_bytes = [0u8; 4];
        cursor.read_exact(&mut id_bytes)?;

        Ok(IndexEntry {
            chunk_id: FourCC(id_bytes),
            flags: cursor.read_u32::<LittleEndian>()?,
            offset: cursor.read_u32::<LittleEndian>()?,
            size: cursor.read_u32::<LittleEndian>()?,
        })
    }

    /// Write to writer
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(self.chunk_id.as_bytes())?;
        writer.write_u32::<LittleEndian>(self.flags)?;
        writer.write_u32::<LittleEndian>(self.offset)?;
        writer.write_u32::<LittleEndian>(self.size)?;
        Ok(())
    }

    /// Check if this is a keyframe
    pub fn is_keyframe(&self) -> bool {
        (self.flags & Self::KEYFRAME) != 0
    }
}

/// Parse idx1 index
pub fn parse_index(data: &[u8]) -> Vec<IndexEntry> {
    let mut entries = Vec::new();
    let mut offset = 0;

    while offset + 16 <= data.len() {
        if let Ok(entry) = IndexEntry::read(&data[offset..]) {
            entries.push(entry);
        }
        offset += 16;
    }

    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fourcc() {
        let fourcc = FourCC::new(*b"RIFF");
        assert_eq!(fourcc.as_str(), "RIFF");
        assert_eq!(fourcc.as_bytes(), b"RIFF");

        let fourcc2 = FourCC::from_str("AVI ").unwrap();
        assert_eq!(fourcc2.as_str(), "AVI ");
    }

    #[test]
    fn test_chunk_id_parse() {
        let video = ChunkId::parse(FourCC(*b"00dc"));
        assert_eq!(video.stream_number, Some(0));
        assert_eq!(video.chunk_type, ChunkType::VideoCompressed);

        let audio = ChunkId::parse(FourCC(*b"01wb"));
        assert_eq!(audio.stream_number, Some(1));
        assert_eq!(audio.chunk_type, ChunkType::Audio);

        let avih = ChunkId::parse(FourCC(*b"avih"));
        assert_eq!(avih.stream_number, None);
    }

    #[test]
    fn test_chunk_id_create() {
        let video = ChunkId::stream_chunk(0, ChunkType::VideoCompressed);
        assert_eq!(video.fourcc.as_str(), "00dc");

        let audio = ChunkId::stream_chunk(5, ChunkType::Audio);
        assert_eq!(audio.fourcc.as_str(), "05wb");
    }

    #[test]
    fn test_riff_chunk_read_write() {
        let original = RiffChunk::new(FourCC(*b"test"), vec![1, 2, 3, 4, 5]);

        let mut buffer = Vec::new();
        original.write(&mut buffer).unwrap();

        // Should be padded to even length
        assert_eq!(buffer.len(), 8 + 6); // header + data + padding

        let (parsed, next) = RiffChunk::read(&buffer, 0).unwrap();
        assert_eq!(parsed.id, original.id);
        assert_eq!(parsed.data, original.data);
        assert_eq!(next, buffer.len());
    }

    #[test]
    fn test_list_chunk() {
        let mut list = ListChunk::new(FourCC(*b"hdrl"));
        list.add_chunk(RiffChunk::new(FourCC(*b"avih"), vec![0; 56]));
        list.add_chunk(RiffChunk::new(FourCC(*b"test"), vec![1, 2, 3]));

        assert_eq!(list.chunks.len(), 2);
        assert!(list.find_chunk(FourCC(*b"avih")).is_some());
        assert!(list.find_chunk(FourCC(*b"none")).is_none());
    }

    #[test]
    fn test_index_entry() {
        let entry = IndexEntry {
            chunk_id: FourCC(*b"00dc"),
            flags: IndexEntry::KEYFRAME,
            offset: 1000,
            size: 5000,
        };

        assert!(entry.is_keyframe());

        let mut buffer = Vec::new();
        entry.write(&mut buffer).unwrap();
        assert_eq!(buffer.len(), 16);

        let parsed = IndexEntry::read(&buffer).unwrap();
        assert_eq!(parsed.chunk_id, entry.chunk_id);
        assert_eq!(parsed.flags, entry.flags);
        assert_eq!(parsed.offset, entry.offset);
        assert_eq!(parsed.size, entry.size);
    }

    #[test]
    fn test_parse_index() {
        let mut data = Vec::new();

        // Write two index entries
        IndexEntry {
            chunk_id: FourCC(*b"00dc"),
            flags: IndexEntry::KEYFRAME,
            offset: 0,
            size: 1000,
        }
        .write(&mut data)
        .unwrap();

        IndexEntry {
            chunk_id: FourCC(*b"01wb"),
            flags: 0,
            offset: 1000,
            size: 500,
        }
        .write(&mut data)
        .unwrap();

        let entries = parse_index(&data);
        assert_eq!(entries.len(), 2);
        assert!(entries[0].is_keyframe());
        assert!(!entries[1].is_keyframe());
    }
}
