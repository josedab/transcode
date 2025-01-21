//! RTMP message parsing.
//!
//! This module handles parsing of RTMP chunks and messages.

use crate::{LiveError, Result, StreamPacket, PacketType};

/// RTMP chunk header format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkFormat {
    /// Type 0: Full header (11 bytes)
    Type0,
    /// Type 1: Same stream (7 bytes)
    Type1,
    /// Type 2: Same stream and length (3 bytes)
    Type2,
    /// Type 3: Continuation (0 bytes)
    Type3,
}

impl ChunkFormat {
    /// Parse from the first two bits of basic header.
    pub fn from_bits(bits: u8) -> Self {
        match bits >> 6 {
            0 => ChunkFormat::Type0,
            1 => ChunkFormat::Type1,
            2 => ChunkFormat::Type2,
            _ => ChunkFormat::Type3,
        }
    }

    /// Get the header size in bytes (excluding basic header).
    pub fn header_size(self) -> usize {
        match self {
            ChunkFormat::Type0 => 11,
            ChunkFormat::Type1 => 7,
            ChunkFormat::Type2 => 3,
            ChunkFormat::Type3 => 0,
        }
    }
}

/// RTMP chunk basic header.
#[derive(Debug, Clone, Copy)]
pub struct ChunkBasicHeader {
    /// Chunk format (0-3)
    pub format: ChunkFormat,
    /// Chunk stream ID (2-65599)
    pub chunk_stream_id: u32,
}

impl ChunkBasicHeader {
    /// Parse from bytes, returns header and bytes consumed.
    pub fn parse(data: &[u8]) -> Result<(Self, usize)> {
        if data.is_empty() {
            return Err(LiveError::Protocol("empty chunk data".into()));
        }

        let first_byte = data[0];
        let format = ChunkFormat::from_bits(first_byte);
        let csid_base = first_byte & 0x3F;

        let (chunk_stream_id, consumed) = match csid_base {
            0 => {
                // 2-byte form: csid = 64 + second byte
                if data.len() < 2 {
                    return Err(LiveError::Protocol("incomplete chunk header".into()));
                }
                (64 + data[1] as u32, 2)
            }
            1 => {
                // 3-byte form: csid = 64 + second byte + third byte * 256
                if data.len() < 3 {
                    return Err(LiveError::Protocol("incomplete chunk header".into()));
                }
                (64 + data[1] as u32 + (data[2] as u32) * 256, 3)
            }
            _ => {
                // 1-byte form: csid = value (2-63)
                (csid_base as u32, 1)
            }
        };

        Ok((Self { format, chunk_stream_id }, consumed))
    }
}

/// RTMP message header.
#[derive(Debug, Clone)]
pub struct MessageHeader {
    /// Timestamp (milliseconds)
    pub timestamp: u32,
    /// Message length in bytes
    pub message_length: u32,
    /// Message type ID
    pub message_type_id: u8,
    /// Message stream ID
    pub message_stream_id: u32,
}

impl MessageHeader {
    /// Parse type 0 header (full header).
    pub fn parse_type0(data: &[u8]) -> Result<Self> {
        if data.len() < 11 {
            return Err(LiveError::Protocol("incomplete type 0 header".into()));
        }

        let timestamp = u32::from_be_bytes([0, data[0], data[1], data[2]]);
        let message_length = u32::from_be_bytes([0, data[3], data[4], data[5]]);
        let message_type_id = data[6];
        // Message stream ID is little-endian
        let message_stream_id = u32::from_le_bytes([data[7], data[8], data[9], data[10]]);

        Ok(Self {
            timestamp,
            message_length,
            message_type_id,
            message_stream_id,
        })
    }

    /// Parse type 1 header (same stream).
    pub fn parse_type1(data: &[u8], prev: &MessageHeader) -> Result<Self> {
        if data.len() < 7 {
            return Err(LiveError::Protocol("incomplete type 1 header".into()));
        }

        let timestamp_delta = u32::from_be_bytes([0, data[0], data[1], data[2]]);
        let message_length = u32::from_be_bytes([0, data[3], data[4], data[5]]);
        let message_type_id = data[6];

        Ok(Self {
            timestamp: prev.timestamp.wrapping_add(timestamp_delta),
            message_length,
            message_type_id,
            message_stream_id: prev.message_stream_id,
        })
    }

    /// Parse type 2 header (same stream and length).
    pub fn parse_type2(data: &[u8], prev: &MessageHeader) -> Result<Self> {
        if data.len() < 3 {
            return Err(LiveError::Protocol("incomplete type 2 header".into()));
        }

        let timestamp_delta = u32::from_be_bytes([0, data[0], data[1], data[2]]);

        Ok(Self {
            timestamp: prev.timestamp.wrapping_add(timestamp_delta),
            message_length: prev.message_length,
            message_type_id: prev.message_type_id,
            message_stream_id: prev.message_stream_id,
        })
    }

    /// Create type 3 header (continuation) from previous.
    pub fn from_type3(prev: &MessageHeader) -> Self {
        prev.clone()
    }
}

/// RTMP message type constants.
pub mod message_type {
    pub const SET_CHUNK_SIZE: u8 = 1;
    pub const ABORT: u8 = 2;
    pub const ACKNOWLEDGEMENT: u8 = 3;
    pub const USER_CONTROL: u8 = 4;
    pub const WINDOW_ACK_SIZE: u8 = 5;
    pub const SET_PEER_BANDWIDTH: u8 = 6;
    pub const AUDIO: u8 = 8;
    pub const VIDEO: u8 = 9;
    pub const AMF3_DATA: u8 = 15;
    pub const AMF3_SHARED_OBJECT: u8 = 16;
    pub const AMF3_COMMAND: u8 = 17;
    pub const AMF0_DATA: u8 = 18;
    pub const AMF0_SHARED_OBJECT: u8 = 19;
    pub const AMF0_COMMAND: u8 = 20;
    pub const AGGREGATE: u8 = 22;
}

/// RTMP chunk parser state.
pub struct ChunkParser {
    chunk_size: u32,
    prev_headers: std::collections::HashMap<u32, MessageHeader>,
    pending_messages: std::collections::HashMap<u32, Vec<u8>>,
}

impl ChunkParser {
    /// Create a new chunk parser.
    pub fn new() -> Self {
        Self {
            chunk_size: 128,
            prev_headers: std::collections::HashMap::new(),
            pending_messages: std::collections::HashMap::new(),
        }
    }

    /// Set the chunk size.
    pub fn set_chunk_size(&mut self, size: u32) {
        self.chunk_size = size;
    }

    /// Get the current chunk size.
    pub fn chunk_size(&self) -> u32 {
        self.chunk_size
    }

    /// Parse chunks from a data buffer.
    /// Returns parsed messages and remaining bytes.
    pub fn parse(&mut self, data: &[u8]) -> Result<(Vec<RtmpMessage>, usize)> {
        let mut messages = Vec::new();
        let mut offset = 0;

        while offset < data.len() {
            // Parse basic header
            let remaining = &data[offset..];
            let (basic, basic_len) = match ChunkBasicHeader::parse(remaining) {
                Ok(r) => r,
                Err(_) => break, // Need more data
            };

            let header_size = basic.format.header_size();
            if remaining.len() < basic_len + header_size {
                break; // Need more data
            }

            // Parse message header
            let header_data = &remaining[basic_len..basic_len + header_size];
            let prev = self.prev_headers.get(&basic.chunk_stream_id);

            let header = match basic.format {
                ChunkFormat::Type0 => MessageHeader::parse_type0(header_data)?,
                ChunkFormat::Type1 => {
                    let prev = prev.ok_or_else(|| {
                        LiveError::Protocol("type 1 chunk without prior header".into())
                    })?;
                    MessageHeader::parse_type1(header_data, prev)?
                }
                ChunkFormat::Type2 => {
                    let prev = prev.ok_or_else(|| {
                        LiveError::Protocol("type 2 chunk without prior header".into())
                    })?;
                    MessageHeader::parse_type2(header_data, prev)?
                }
                ChunkFormat::Type3 => {
                    let prev = prev.ok_or_else(|| {
                        LiveError::Protocol("type 3 chunk without prior header".into())
                    })?;
                    MessageHeader::from_type3(prev)
                }
            };

            // Calculate chunk data size
            let pending = self
                .pending_messages
                .get(&basic.chunk_stream_id)
                .map(|v| v.len())
                .unwrap_or(0);
            let remaining_msg_size = header.message_length as usize - pending;
            let chunk_data_size = remaining_msg_size.min(self.chunk_size as usize);

            let payload_start = basic_len + header_size;
            if remaining.len() < payload_start + chunk_data_size {
                break; // Need more data
            }

            // Collect chunk payload
            let payload = &remaining[payload_start..payload_start + chunk_data_size];
            let buffer = self
                .pending_messages
                .entry(basic.chunk_stream_id)
                .or_default();
            buffer.extend_from_slice(payload);

            // Check if message is complete
            if buffer.len() >= header.message_length as usize {
                let data = self.pending_messages.remove(&basic.chunk_stream_id).unwrap();
                messages.push(RtmpMessage {
                    header: header.clone(),
                    data,
                });
            }

            // Store header for next chunk
            self.prev_headers.insert(basic.chunk_stream_id, header);

            offset += payload_start + chunk_data_size;
        }

        Ok((messages, offset))
    }
}

impl Default for ChunkParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Parsed RTMP message.
#[derive(Debug, Clone)]
pub struct RtmpMessage {
    /// Message header
    pub header: MessageHeader,
    /// Message data
    pub data: Vec<u8>,
}

impl RtmpMessage {
    /// Convert to stream packet if this is audio/video.
    pub fn to_stream_packet(&self) -> Option<StreamPacket> {
        let packet_type = match self.header.message_type_id {
            message_type::AUDIO => PacketType::Audio,
            message_type::VIDEO => PacketType::Video,
            _ => return None,
        };

        // Check for keyframe (first byte of FLV video tag)
        let is_keyframe = if packet_type == PacketType::Video && !self.data.is_empty() {
            (self.data[0] >> 4) == 1 // Frame type 1 = keyframe
        } else {
            false
        };

        Some(StreamPacket {
            packet_type,
            timestamp: self.header.timestamp as u64,
            data: self.data.clone(),
            is_keyframe,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_format() {
        assert_eq!(ChunkFormat::from_bits(0x00), ChunkFormat::Type0);
        assert_eq!(ChunkFormat::from_bits(0x40), ChunkFormat::Type1);
        assert_eq!(ChunkFormat::from_bits(0x80), ChunkFormat::Type2);
        assert_eq!(ChunkFormat::from_bits(0xC0), ChunkFormat::Type3);
    }

    #[test]
    fn test_basic_header_1byte() {
        let data = [0x03]; // Type 0, csid 3
        let (header, len) = ChunkBasicHeader::parse(&data).unwrap();

        assert_eq!(header.format, ChunkFormat::Type0);
        assert_eq!(header.chunk_stream_id, 3);
        assert_eq!(len, 1);
    }

    #[test]
    fn test_basic_header_2byte() {
        let data = [0x00, 0x10]; // Type 0, csid 64+16=80
        let (header, len) = ChunkBasicHeader::parse(&data).unwrap();

        assert_eq!(header.format, ChunkFormat::Type0);
        assert_eq!(header.chunk_stream_id, 80);
        assert_eq!(len, 2);
    }

    #[test]
    fn test_message_header_type0() {
        let data = [
            0x00, 0x00, 0x64, // timestamp = 100
            0x00, 0x00, 0x0A, // length = 10
            0x09,             // type = video
            0x01, 0x00, 0x00, 0x00, // stream id = 1 (little endian)
        ];

        let header = MessageHeader::parse_type0(&data).unwrap();

        assert_eq!(header.timestamp, 100);
        assert_eq!(header.message_length, 10);
        assert_eq!(header.message_type_id, message_type::VIDEO);
        assert_eq!(header.message_stream_id, 1);
    }

    #[test]
    fn test_chunk_parser() {
        let mut parser = ChunkParser::new();

        assert_eq!(parser.chunk_size(), 128);

        parser.set_chunk_size(4096);
        assert_eq!(parser.chunk_size(), 4096);
    }
}
