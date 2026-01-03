//! RIFF container parsing for WebP files
//!
//! WebP uses the RIFF container format with the following structure:
//! - RIFF header (4 bytes: "RIFF")
//! - File size (4 bytes, little-endian)
//! - WEBP signature (4 bytes: "WEBP")
//! - Chunks (variable)

use std::io::{Read, Seek, SeekFrom};
use byteorder::{LittleEndian, ReadBytesExt};
use crate::error::{WebPError, Result};

/// RIFF file signature
const RIFF_SIGNATURE: &[u8; 4] = b"RIFF";
/// WebP format signature
const WEBP_SIGNATURE: &[u8; 4] = b"WEBP";

/// Chunk types in WebP files
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChunkType {
    /// VP8 lossy bitstream
    VP8,
    /// VP8L lossless bitstream
    VP8L,
    /// Extended format header
    VP8X,
    /// Alpha channel data
    ALPH,
    /// Animation global parameters
    ANIM,
    /// Animation frame
    ANMF,
    /// ICC color profile
    ICCP,
    /// EXIF metadata
    EXIF,
    /// XMP metadata
    XMP,
    /// Unknown chunk type
    Unknown([u8; 4]),
}

impl ChunkType {
    /// Parse chunk type from 4-byte identifier
    pub fn from_fourcc(fourcc: [u8; 4]) -> Self {
        match &fourcc {
            b"VP8 " => ChunkType::VP8,
            b"VP8L" => ChunkType::VP8L,
            b"VP8X" => ChunkType::VP8X,
            b"ALPH" => ChunkType::ALPH,
            b"ANIM" => ChunkType::ANIM,
            b"ANMF" => ChunkType::ANMF,
            b"ICCP" => ChunkType::ICCP,
            b"EXIF" => ChunkType::EXIF,
            b"XMP " => ChunkType::XMP,
            _ => ChunkType::Unknown(fourcc),
        }
    }

    /// Get the 4-byte identifier for this chunk type
    pub fn to_fourcc(&self) -> [u8; 4] {
        match self {
            ChunkType::VP8 => *b"VP8 ",
            ChunkType::VP8L => *b"VP8L",
            ChunkType::VP8X => *b"VP8X",
            ChunkType::ALPH => *b"ALPH",
            ChunkType::ANIM => *b"ANIM",
            ChunkType::ANMF => *b"ANMF",
            ChunkType::ICCP => *b"ICCP",
            ChunkType::EXIF => *b"EXIF",
            ChunkType::XMP => *b"XMP ",
            ChunkType::Unknown(fourcc) => *fourcc,
        }
    }
}

/// A chunk in the RIFF container
#[derive(Debug, Clone)]
pub struct WebPChunk {
    /// Type of this chunk
    pub chunk_type: ChunkType,
    /// Chunk data (not including header)
    pub data: Vec<u8>,
    /// Offset in the file where this chunk's data begins
    pub offset: u64,
}

impl WebPChunk {
    /// Get the total size of this chunk including header and padding
    pub fn total_size(&self) -> usize {
        // 4 bytes type + 4 bytes size + data + padding
        8 + self.data.len() + (self.data.len() % 2)
    }
}

/// Parsed RIFF container for WebP
#[derive(Debug, Clone)]
pub struct RiffContainer {
    /// Total file size (from RIFF header)
    pub file_size: u32,
    /// All chunks in the container
    pub chunks: Vec<WebPChunk>,
    /// VP8X flags (if present)
    pub vp8x_flags: Option<Vp8xFlags>,
    /// Canvas width (from VP8X or image data)
    pub width: Option<u32>,
    /// Canvas height (from VP8X or image data)
    pub height: Option<u32>,
}

impl RiffContainer {
    /// Find a chunk by type
    pub fn find_chunk(&self, chunk_type: ChunkType) -> Option<&WebPChunk> {
        self.chunks.iter().find(|c| c.chunk_type == chunk_type)
    }

    /// Find all chunks of a given type
    pub fn find_chunks(&self, chunk_type: ChunkType) -> Vec<&WebPChunk> {
        self.chunks.iter().filter(|c| c.chunk_type == chunk_type).collect()
    }

    /// Check if this is an animated WebP
    pub fn is_animated(&self) -> bool {
        self.vp8x_flags.is_some_and(|f| f.animation)
    }

    /// Check if this WebP has alpha
    pub fn has_alpha(&self) -> bool {
        self.vp8x_flags.is_some_and(|f| f.alpha)
    }

    /// Check if this WebP has EXIF metadata
    pub fn has_exif(&self) -> bool {
        self.vp8x_flags.is_some_and(|f| f.exif)
    }

    /// Check if this WebP has XMP metadata
    pub fn has_xmp(&self) -> bool {
        self.vp8x_flags.is_some_and(|f| f.xmp)
    }

    /// Check if this WebP has ICC profile
    pub fn has_icc(&self) -> bool {
        self.vp8x_flags.is_some_and(|f| f.icc)
    }

    /// Get image dimensions
    pub fn dimensions(&self) -> Result<(u32, u32)> {
        match (self.width, self.height) {
            (Some(w), Some(h)) => Ok((w, h)),
            _ => Err(WebPError::InvalidFormat("Dimensions not available".into())),
        }
    }
}

/// VP8X extended header flags
#[derive(Debug, Clone, Copy, Default)]
pub struct Vp8xFlags {
    /// Has ICC profile
    pub icc: bool,
    /// Has alpha channel
    pub alpha: bool,
    /// Has EXIF metadata
    pub exif: bool,
    /// Has XMP metadata
    pub xmp: bool,
    /// Is animated
    pub animation: bool,
}

impl Vp8xFlags {
    /// Parse VP8X flags from the first byte
    pub fn from_byte(byte: u8) -> Self {
        Self {
            icc: (byte & 0x20) != 0,
            alpha: (byte & 0x10) != 0,
            exif: (byte & 0x08) != 0,
            xmp: (byte & 0x04) != 0,
            animation: (byte & 0x02) != 0,
        }
    }
}

/// Parse a RIFF container from a reader
pub fn parse_riff<R: Read + Seek>(reader: &mut R) -> Result<RiffContainer> {
    // Read and verify RIFF signature
    let mut signature = [0u8; 4];
    reader.read_exact(&mut signature)?;
    if &signature != RIFF_SIGNATURE {
        return Err(WebPError::InvalidRiff(format!(
            "Invalid RIFF signature: {:?}",
            signature
        )));
    }

    // Read file size (total size minus 8 bytes for RIFF header)
    let file_size = reader.read_u32::<LittleEndian>()?;

    // Read and verify WEBP signature
    reader.read_exact(&mut signature)?;
    if &signature != WEBP_SIGNATURE {
        return Err(WebPError::InvalidRiff(format!(
            "Invalid WEBP signature: {:?}",
            signature
        )));
    }

    // Parse chunks
    let mut chunks = Vec::new();
    let mut vp8x_flags = None;
    let mut width = None;
    let mut height = None;

    // Calculate end position (file_size includes WEBP signature but not RIFF header)
    let end_pos = 12 + file_size as u64 - 4;

    while reader.stream_position()? < end_pos {
        let chunk = parse_chunk(reader)?;

        // Extract VP8X info
        if chunk.chunk_type == ChunkType::VP8X && chunk.data.len() >= 10 {
            vp8x_flags = Some(Vp8xFlags::from_byte(chunk.data[0]));
            // Width and height are stored as 24-bit values minus 1
            let w = u32::from(chunk.data[4])
                | (u32::from(chunk.data[5]) << 8)
                | (u32::from(chunk.data[6]) << 16);
            let h = u32::from(chunk.data[7])
                | (u32::from(chunk.data[8]) << 8)
                | (u32::from(chunk.data[9]) << 16);
            width = Some(w + 1);
            height = Some(h + 1);
        }

        // Extract dimensions from VP8 if not in VP8X
        if width.is_none() && chunk.chunk_type == ChunkType::VP8 && chunk.data.len() >= 10 {
            if let Some((w, h)) = parse_vp8_dimensions(&chunk.data) {
                width = Some(w);
                height = Some(h);
            }
        }

        // Extract dimensions from VP8L if not already set
        if width.is_none() && chunk.chunk_type == ChunkType::VP8L && chunk.data.len() >= 5 {
            if let Some((w, h)) = parse_vp8l_dimensions(&chunk.data) {
                width = Some(w);
                height = Some(h);
            }
        }

        chunks.push(chunk);
    }

    Ok(RiffContainer {
        file_size,
        chunks,
        vp8x_flags,
        width,
        height,
    })
}

/// Parse a single chunk from the reader
fn parse_chunk<R: Read + Seek>(reader: &mut R) -> Result<WebPChunk> {
    // Read chunk type
    let mut fourcc = [0u8; 4];
    reader.read_exact(&mut fourcc)?;
    let chunk_type = ChunkType::from_fourcc(fourcc);

    // Read chunk size
    let size = reader.read_u32::<LittleEndian>()? as usize;

    // Record offset
    let offset = reader.stream_position()?;

    // Read chunk data
    let mut data = vec![0u8; size];
    reader.read_exact(&mut data)?;

    // Skip padding byte if size is odd
    if !size.is_multiple_of(2) {
        reader.seek(SeekFrom::Current(1))?;
    }

    Ok(WebPChunk {
        chunk_type,
        data,
        offset,
    })
}

/// Parse dimensions from VP8 bitstream header
fn parse_vp8_dimensions(data: &[u8]) -> Option<(u32, u32)> {
    // VP8 frame header starts with a 3-byte frame tag
    if data.len() < 10 {
        return None;
    }

    // Check for key frame (bit 0 of first byte should be 0)
    let frame_tag = u32::from(data[0])
        | (u32::from(data[1]) << 8)
        | (u32::from(data[2]) << 16);

    let is_keyframe = (frame_tag & 1) == 0;
    if !is_keyframe {
        return None;
    }

    // Key frame starts with signature 0x9d 0x01 0x2a
    if data[3] != 0x9d || data[4] != 0x01 || data[5] != 0x2a {
        return None;
    }

    // Width and height follow (16-bit each, little-endian)
    // Only lower 14 bits are the actual dimension
    let width = u32::from(data[6]) | (u32::from(data[7]) << 8);
    let height = u32::from(data[8]) | (u32::from(data[9]) << 8);

    Some((width & 0x3FFF, height & 0x3FFF))
}

/// Parse dimensions from VP8L bitstream header
fn parse_vp8l_dimensions(data: &[u8]) -> Option<(u32, u32)> {
    if data.len() < 5 {
        return None;
    }

    // VP8L signature
    if data[0] != 0x2f {
        return None;
    }

    // Width and height are packed in 28 bits
    let bits = u32::from(data[1])
        | (u32::from(data[2]) << 8)
        | (u32::from(data[3]) << 16)
        | (u32::from(data[4]) << 24);

    let width = (bits & 0x3FFF) + 1;
    let height = ((bits >> 14) & 0x3FFF) + 1;

    Some((width, height))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_chunk_type_parsing() {
        assert_eq!(ChunkType::from_fourcc(*b"VP8 "), ChunkType::VP8);
        assert_eq!(ChunkType::from_fourcc(*b"VP8L"), ChunkType::VP8L);
        assert_eq!(ChunkType::from_fourcc(*b"VP8X"), ChunkType::VP8X);
        assert_eq!(ChunkType::from_fourcc(*b"ALPH"), ChunkType::ALPH);
        assert_eq!(ChunkType::from_fourcc(*b"ANIM"), ChunkType::ANIM);
        assert_eq!(ChunkType::from_fourcc(*b"ANMF"), ChunkType::ANMF);
        assert_eq!(ChunkType::from_fourcc(*b"EXIF"), ChunkType::EXIF);
        assert_eq!(ChunkType::from_fourcc(*b"XMP "), ChunkType::XMP);
    }

    #[test]
    fn test_vp8x_flags() {
        let flags = Vp8xFlags::from_byte(0x3E);
        assert!(flags.icc);
        assert!(flags.alpha);
        assert!(flags.exif);
        assert!(flags.xmp);
        assert!(flags.animation);

        let flags = Vp8xFlags::from_byte(0x00);
        assert!(!flags.icc);
        assert!(!flags.alpha);
        assert!(!flags.exif);
        assert!(!flags.xmp);
        assert!(!flags.animation);
    }

    #[test]
    fn test_invalid_riff() {
        let data = b"NOTARIFF";
        let mut cursor = Cursor::new(&data[..]);
        let result = parse_riff(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn test_vp8l_dimensions() {
        // VP8L signature + packed dimensions for 100x100
        let mut data = vec![0x2f];
        // Width = 100 (stored as 99), Height = 100 (stored as 99)
        // bits = (99) | ((99) << 14)
        let bits: u32 = 99 | (99 << 14);
        data.push((bits & 0xFF) as u8);
        data.push(((bits >> 8) & 0xFF) as u8);
        data.push(((bits >> 16) & 0xFF) as u8);
        data.push(((bits >> 24) & 0xFF) as u8);

        let dims = parse_vp8l_dimensions(&data);
        assert_eq!(dims, Some((100, 100)));
    }
}
