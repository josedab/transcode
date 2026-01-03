//! FLV file header parsing and writing.
//!
//! The FLV header is the first 9 bytes of an FLV file and contains:
//! - Signature: "FLV" (3 bytes)
//! - Version: 1 (1 byte)
//! - Flags: audio/video presence (1 byte)
//! - Header size: 9 (4 bytes, big-endian)

use crate::error::{FlvError, Result};
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Write};

/// FLV file signature.
pub const FLV_SIGNATURE: &[u8; 3] = b"FLV";

/// Current FLV version.
pub const FLV_VERSION: u8 = 1;

/// Standard FLV header size.
pub const FLV_HEADER_SIZE: u32 = 9;

/// Flag indicating audio is present.
pub const FLV_FLAG_AUDIO: u8 = 0x04;

/// Flag indicating video is present.
pub const FLV_FLAG_VIDEO: u8 = 0x01;

/// FLV file header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlvHeader {
    /// FLV version (should be 1).
    pub version: u8,
    /// Whether the file contains audio.
    pub has_audio: bool,
    /// Whether the file contains video.
    pub has_video: bool,
    /// Header size (should be 9).
    pub header_size: u32,
}

impl Default for FlvHeader {
    fn default() -> Self {
        Self {
            version: FLV_VERSION,
            has_audio: false,
            has_video: false,
            header_size: FLV_HEADER_SIZE,
        }
    }
}

impl FlvHeader {
    /// Create a new FLV header with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a header with audio only.
    pub fn audio_only() -> Self {
        Self {
            has_audio: true,
            has_video: false,
            ..Self::default()
        }
    }

    /// Create a header with video only.
    pub fn video_only() -> Self {
        Self {
            has_audio: false,
            has_video: true,
            ..Self::default()
        }
    }

    /// Create a header with both audio and video.
    pub fn audio_video() -> Self {
        Self {
            has_audio: true,
            has_video: true,
            ..Self::default()
        }
    }

    /// Set whether audio is present.
    pub fn with_audio(mut self, has_audio: bool) -> Self {
        self.has_audio = has_audio;
        self
    }

    /// Set whether video is present.
    pub fn with_video(mut self, has_video: bool) -> Self {
        self.has_video = has_video;
        self
    }

    /// Parse an FLV header from a reader.
    pub fn parse<R: Read>(reader: &mut R) -> Result<Self> {
        // Read signature
        let mut signature = [0u8; 3];
        reader.read_exact(&mut signature)?;

        if &signature != FLV_SIGNATURE {
            return Err(FlvError::InvalidSignature(
                String::from_utf8_lossy(&signature).to_string(),
            ));
        }

        // Read version
        let version = reader.read_u8()?;
        if version != FLV_VERSION {
            return Err(FlvError::InvalidVersion(version));
        }

        // Read flags
        let flags = reader.read_u8()?;
        let has_audio = (flags & FLV_FLAG_AUDIO) != 0;
        let has_video = (flags & FLV_FLAG_VIDEO) != 0;

        // Read header size
        let header_size = reader.read_u32::<BigEndian>()?;

        // Validate header size (must be at least 9)
        if header_size < FLV_HEADER_SIZE {
            return Err(FlvError::InvalidTagSize {
                offset: 5,
                message: format!(
                    "Header size {} is less than minimum {}",
                    header_size, FLV_HEADER_SIZE
                ),
            });
        }

        Ok(Self {
            version,
            has_audio,
            has_video,
            header_size,
        })
    }

    /// Write the FLV header to a writer.
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<usize> {
        // Write signature
        writer.write_all(FLV_SIGNATURE)?;

        // Write version
        writer.write_u8(self.version)?;

        // Write flags
        let mut flags = 0u8;
        if self.has_audio {
            flags |= FLV_FLAG_AUDIO;
        }
        if self.has_video {
            flags |= FLV_FLAG_VIDEO;
        }
        writer.write_u8(flags)?;

        // Write header size
        writer.write_u32::<BigEndian>(self.header_size)?;

        Ok(FLV_HEADER_SIZE as usize)
    }

    /// Get the flags byte.
    pub fn flags(&self) -> u8 {
        let mut flags = 0u8;
        if self.has_audio {
            flags |= FLV_FLAG_AUDIO;
        }
        if self.has_video {
            flags |= FLV_FLAG_VIDEO;
        }
        flags
    }

    /// Check if the header is valid.
    pub fn is_valid(&self) -> bool {
        self.version == FLV_VERSION && self.header_size >= FLV_HEADER_SIZE
    }
}

/// Check if data starts with a valid FLV signature.
pub fn is_flv_signature(data: &[u8]) -> bool {
    data.len() >= 3 && &data[0..3] == FLV_SIGNATURE
}

/// Quickly check if a reader contains an FLV file.
pub fn probe_flv<R: Read>(reader: &mut R) -> Result<bool> {
    let mut signature = [0u8; 3];
    match reader.read_exact(&mut signature) {
        Ok(()) => Ok(&signature == FLV_SIGNATURE),
        Err(_) => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_header_default() {
        let header = FlvHeader::default();
        assert_eq!(header.version, FLV_VERSION);
        assert!(!header.has_audio);
        assert!(!header.has_video);
        assert_eq!(header.header_size, FLV_HEADER_SIZE);
    }

    #[test]
    fn test_header_audio_video() {
        let header = FlvHeader::audio_video();
        assert!(header.has_audio);
        assert!(header.has_video);
    }

    #[test]
    fn test_header_audio_only() {
        let header = FlvHeader::audio_only();
        assert!(header.has_audio);
        assert!(!header.has_video);
    }

    #[test]
    fn test_header_video_only() {
        let header = FlvHeader::video_only();
        assert!(!header.has_audio);
        assert!(header.has_video);
    }

    #[test]
    fn test_header_roundtrip() {
        let original = FlvHeader::audio_video();

        let mut buffer = Vec::new();
        original.write(&mut buffer).unwrap();

        assert_eq!(buffer.len(), FLV_HEADER_SIZE as usize);

        let mut cursor = Cursor::new(&buffer);
        let parsed = FlvHeader::parse(&mut cursor).unwrap();

        assert_eq!(original, parsed);
    }

    #[test]
    fn test_header_write_bytes() {
        let header = FlvHeader::audio_video();

        let mut buffer = Vec::new();
        header.write(&mut buffer).unwrap();

        // FLV signature
        assert_eq!(&buffer[0..3], b"FLV");
        // Version
        assert_eq!(buffer[3], 1);
        // Flags (audio + video = 0x05)
        assert_eq!(buffer[4], 0x05);
        // Header size (9 as big-endian u32)
        assert_eq!(&buffer[5..9], &[0, 0, 0, 9]);
    }

    #[test]
    fn test_header_parse_invalid_signature() {
        let data = [b'A', b'B', b'C', 1, 0x05, 0, 0, 0, 9];
        let mut cursor = Cursor::new(&data);

        let result = FlvHeader::parse(&mut cursor);
        assert!(matches!(result, Err(FlvError::InvalidSignature(_))));
    }

    #[test]
    fn test_header_parse_invalid_version() {
        let data = [b'F', b'L', b'V', 2, 0x05, 0, 0, 0, 9];
        let mut cursor = Cursor::new(&data);

        let result = FlvHeader::parse(&mut cursor);
        assert!(matches!(result, Err(FlvError::InvalidVersion(2))));
    }

    #[test]
    fn test_header_flags() {
        let header = FlvHeader::audio_video();
        assert_eq!(header.flags(), 0x05);

        let header = FlvHeader::audio_only();
        assert_eq!(header.flags(), 0x04);

        let header = FlvHeader::video_only();
        assert_eq!(header.flags(), 0x01);

        let header = FlvHeader::new();
        assert_eq!(header.flags(), 0x00);
    }

    #[test]
    fn test_is_flv_signature() {
        assert!(is_flv_signature(b"FLV"));
        assert!(is_flv_signature(b"FLV\x01\x05"));
        assert!(!is_flv_signature(b"FL"));
        assert!(!is_flv_signature(b"ABC"));
        assert!(!is_flv_signature(b""));
    }

    #[test]
    fn test_is_valid() {
        let header = FlvHeader::default();
        assert!(header.is_valid());

        let invalid_header = FlvHeader {
            version: 2,
            ..FlvHeader::default()
        };
        assert!(!invalid_header.is_valid());
    }

    #[test]
    fn test_builder_pattern() {
        let header = FlvHeader::new().with_audio(true).with_video(true);

        assert!(header.has_audio);
        assert!(header.has_video);
    }

    #[test]
    fn test_probe_flv() {
        let valid_data = b"FLV\x01\x05\x00\x00\x00\x09";
        let mut cursor = Cursor::new(&valid_data[..]);
        assert!(probe_flv(&mut cursor).unwrap());

        let invalid_data = b"ABC";
        let mut cursor = Cursor::new(&invalid_data[..]);
        assert!(!probe_flv(&mut cursor).unwrap());
    }
}
