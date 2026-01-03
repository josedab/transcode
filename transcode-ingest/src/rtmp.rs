//! RTMP (Real-Time Messaging Protocol) implementation
//!
//! This module provides RTMP server and client functionality including:
//! - Full handshake support (C0/S0, C1/S1, C2/S2)
//! - AMF0 message parsing
//! - Audio/video chunk handling
//! - Publish/play commands

use async_trait::async_trait;
use bytes::{Buf, BufMut, Bytes, BytesMut};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::{
    AudioCodec, ConnectionState, IngestConfig, IngestError, IngestResult, IngestServer,
    IngestSource, MediaData, MetadataValue, StreamInfo, VideoCodec,
};

// RTMP Constants
const RTMP_VERSION: u8 = 3;
const RTMP_HANDSHAKE_SIZE: usize = 1536;
const RTMP_DEFAULT_CHUNK_SIZE: u32 = 128;
const RTMP_MAX_CHUNK_SIZE: u32 = 65536;

// Message type IDs
const MSG_SET_CHUNK_SIZE: u8 = 1;
const MSG_ABORT: u8 = 2;
const MSG_ACK: u8 = 3;
const MSG_USER_CONTROL: u8 = 4;
const MSG_WINDOW_ACK_SIZE: u8 = 5;
const MSG_SET_PEER_BANDWIDTH: u8 = 6;
const MSG_AUDIO: u8 = 8;
const MSG_VIDEO: u8 = 9;
const MSG_DATA_AMF3: u8 = 15;
#[allow(dead_code)]
const MSG_SHARED_OBJECT_AMF3: u8 = 16;
const MSG_COMMAND_AMF3: u8 = 17;
const MSG_DATA_AMF0: u8 = 18;
#[allow(dead_code)]
const MSG_SHARED_OBJECT_AMF0: u8 = 19;
const MSG_COMMAND_AMF0: u8 = 20;
#[allow(dead_code)]
const MSG_AGGREGATE: u8 = 22;

// AMF0 type markers
const AMF0_NUMBER: u8 = 0x00;
const AMF0_BOOLEAN: u8 = 0x01;
const AMF0_STRING: u8 = 0x02;
const AMF0_OBJECT: u8 = 0x03;
const AMF0_NULL: u8 = 0x05;
const AMF0_UNDEFINED: u8 = 0x06;
const AMF0_ECMA_ARRAY: u8 = 0x08;
const AMF0_OBJECT_END: u8 = 0x09;
const AMF0_STRICT_ARRAY: u8 = 0x0A;
const AMF0_LONG_STRING: u8 = 0x0C;

/// AMF0 value representation
#[derive(Debug, Clone)]
pub enum Amf0Value {
    Number(f64),
    Boolean(bool),
    String(String),
    Object(HashMap<String, Amf0Value>),
    Null,
    Undefined,
    EcmaArray(HashMap<String, Amf0Value>),
    StrictArray(Vec<Amf0Value>),
}

impl Amf0Value {
    /// Parse an AMF0 value from bytes
    pub fn parse(data: &mut BytesMut) -> IngestResult<Self> {
        if data.is_empty() {
            return Err(IngestError::InvalidMessage("Empty AMF0 data".into()));
        }

        let marker = data.get_u8();
        match marker {
            AMF0_NUMBER => {
                if data.remaining() < 8 {
                    return Err(IngestError::InvalidMessage("Incomplete AMF0 number".into()));
                }
                Ok(Amf0Value::Number(data.get_f64()))
            }
            AMF0_BOOLEAN => {
                if data.is_empty() {
                    return Err(IngestError::InvalidMessage("Incomplete AMF0 boolean".into()));
                }
                Ok(Amf0Value::Boolean(data.get_u8() != 0))
            }
            AMF0_STRING => Self::parse_string(data),
            AMF0_OBJECT => Self::parse_object(data),
            AMF0_NULL => Ok(Amf0Value::Null),
            AMF0_UNDEFINED => Ok(Amf0Value::Undefined),
            AMF0_ECMA_ARRAY => {
                if data.remaining() < 4 {
                    return Err(IngestError::InvalidMessage(
                        "Incomplete AMF0 ECMA array".into(),
                    ));
                }
                let _count = data.get_u32();
                Self::parse_object_properties(data).map(Amf0Value::EcmaArray)
            }
            AMF0_STRICT_ARRAY => {
                if data.remaining() < 4 {
                    return Err(IngestError::InvalidMessage(
                        "Incomplete AMF0 strict array".into(),
                    ));
                }
                let count = data.get_u32() as usize;
                let mut items = Vec::with_capacity(count);
                for _ in 0..count {
                    items.push(Self::parse(data)?);
                }
                Ok(Amf0Value::StrictArray(items))
            }
            AMF0_LONG_STRING => Self::parse_long_string(data),
            _ => Err(IngestError::InvalidMessage(format!(
                "Unknown AMF0 marker: {:#x}",
                marker
            ))),
        }
    }

    fn parse_string(data: &mut BytesMut) -> IngestResult<Self> {
        if data.remaining() < 2 {
            return Err(IngestError::InvalidMessage(
                "Incomplete AMF0 string length".into(),
            ));
        }
        let len = data.get_u16() as usize;
        if data.remaining() < len {
            return Err(IngestError::InvalidMessage("Incomplete AMF0 string".into()));
        }
        let bytes = data.split_to(len);
        let s = String::from_utf8_lossy(&bytes).into_owned();
        Ok(Amf0Value::String(s))
    }

    fn parse_long_string(data: &mut BytesMut) -> IngestResult<Self> {
        if data.remaining() < 4 {
            return Err(IngestError::InvalidMessage(
                "Incomplete AMF0 long string length".into(),
            ));
        }
        let len = data.get_u32() as usize;
        if data.remaining() < len {
            return Err(IngestError::InvalidMessage(
                "Incomplete AMF0 long string".into(),
            ));
        }
        let bytes = data.split_to(len);
        let s = String::from_utf8_lossy(&bytes).into_owned();
        Ok(Amf0Value::String(s))
    }

    fn parse_object(data: &mut BytesMut) -> IngestResult<Self> {
        Self::parse_object_properties(data).map(Amf0Value::Object)
    }

    fn parse_object_properties(data: &mut BytesMut) -> IngestResult<HashMap<String, Amf0Value>> {
        let mut properties = HashMap::new();

        loop {
            if data.remaining() < 2 {
                return Err(IngestError::InvalidMessage(
                    "Incomplete AMF0 object property".into(),
                ));
            }

            let key_len = data.get_u16() as usize;

            if key_len == 0 {
                if data.is_empty() {
                    return Err(IngestError::InvalidMessage(
                        "Missing AMF0 object end marker".into(),
                    ));
                }
                let end_marker = data.get_u8();
                if end_marker == AMF0_OBJECT_END {
                    break;
                } else {
                    return Err(IngestError::InvalidMessage(format!(
                        "Expected object end marker, got {:#x}",
                        end_marker
                    )));
                }
            }

            if data.remaining() < key_len {
                return Err(IngestError::InvalidMessage(
                    "Incomplete AMF0 property key".into(),
                ));
            }

            let key_bytes = data.split_to(key_len);
            let key = String::from_utf8_lossy(&key_bytes).into_owned();
            let value = Self::parse(data)?;
            properties.insert(key, value);
        }

        Ok(properties)
    }

    /// Encode an AMF0 value to bytes
    pub fn encode(&self, buf: &mut BytesMut) {
        match self {
            Amf0Value::Number(n) => {
                buf.put_u8(AMF0_NUMBER);
                buf.put_f64(*n);
            }
            Amf0Value::Boolean(b) => {
                buf.put_u8(AMF0_BOOLEAN);
                buf.put_u8(if *b { 1 } else { 0 });
            }
            Amf0Value::String(s) => {
                if s.len() > 65535 {
                    buf.put_u8(AMF0_LONG_STRING);
                    buf.put_u32(s.len() as u32);
                } else {
                    buf.put_u8(AMF0_STRING);
                    buf.put_u16(s.len() as u16);
                }
                buf.put_slice(s.as_bytes());
            }
            Amf0Value::Object(props) => {
                buf.put_u8(AMF0_OBJECT);
                Self::encode_properties(props, buf);
            }
            Amf0Value::Null => {
                buf.put_u8(AMF0_NULL);
            }
            Amf0Value::Undefined => {
                buf.put_u8(AMF0_UNDEFINED);
            }
            Amf0Value::EcmaArray(props) => {
                buf.put_u8(AMF0_ECMA_ARRAY);
                buf.put_u32(props.len() as u32);
                Self::encode_properties(props, buf);
            }
            Amf0Value::StrictArray(items) => {
                buf.put_u8(AMF0_STRICT_ARRAY);
                buf.put_u32(items.len() as u32);
                for item in items {
                    item.encode(buf);
                }
            }
        }
    }

    fn encode_properties(props: &HashMap<String, Amf0Value>, buf: &mut BytesMut) {
        for (key, value) in props {
            buf.put_u16(key.len() as u16);
            buf.put_slice(key.as_bytes());
            value.encode(buf);
        }
        // Object end marker
        buf.put_u16(0);
        buf.put_u8(AMF0_OBJECT_END);
    }

    /// Convert to string if possible
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Amf0Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Convert to number if possible
    pub fn as_number(&self) -> Option<f64> {
        match self {
            Amf0Value::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Convert to object if possible
    pub fn as_object(&self) -> Option<&HashMap<String, Amf0Value>> {
        match self {
            Amf0Value::Object(o) | Amf0Value::EcmaArray(o) => Some(o),
            _ => None,
        }
    }
}

/// RTMP chunk header
#[derive(Debug, Clone)]
pub struct ChunkHeader {
    /// Chunk stream ID (2-65599)
    pub chunk_stream_id: u32,
    /// Timestamp (or timestamp delta)
    pub timestamp: u32,
    /// Message length
    pub message_length: u32,
    /// Message type ID
    pub message_type_id: u8,
    /// Message stream ID
    pub message_stream_id: u32,
    /// Whether timestamp is a delta
    pub timestamp_delta: bool,
}

/// RTMP message
#[derive(Debug, Clone)]
pub struct RtmpMessage {
    /// Message header
    pub header: ChunkHeader,
    /// Message payload
    pub payload: Bytes,
}

/// RTMP chunk stream state
#[derive(Debug, Clone, Default)]
struct ChunkStreamState {
    last_header: Option<ChunkHeader>,
    incomplete_message: Option<BytesMut>,
}

/// RTMP connection handler
pub struct RtmpConnection {
    stream: TcpStream,
    peer_addr: SocketAddr,
    state: ConnectionState,
    stream_info: StreamInfo,
    #[allow(dead_code)]
    config: IngestConfig,

    // Chunk handling
    read_chunk_size: u32,
    write_chunk_size: u32,
    chunk_streams: HashMap<u32, ChunkStreamState>,

    // Protocol state
    window_ack_size: u32,
    bytes_received: u64,
    last_ack_sent: u64,

    // Read buffer
    read_buffer: BytesMut,
}

impl RtmpConnection {
    /// Create a new RTMP connection from a TCP stream
    pub fn new(stream: TcpStream, peer_addr: SocketAddr, config: IngestConfig) -> Self {
        Self {
            stream,
            peer_addr,
            state: ConnectionState::Disconnected,
            stream_info: StreamInfo {
                name: String::new(),
                app: String::new(),
                peer_addr,
                connected_at: Instant::now(),
            },
            config,
            read_chunk_size: RTMP_DEFAULT_CHUNK_SIZE,
            write_chunk_size: RTMP_DEFAULT_CHUNK_SIZE,
            chunk_streams: HashMap::new(),
            window_ack_size: 2500000,
            bytes_received: 0,
            last_ack_sent: 0,
            read_buffer: BytesMut::with_capacity(65536),
        }
    }

    /// Perform RTMP handshake as server
    pub async fn perform_server_handshake(&mut self) -> IngestResult<()> {
        self.state = ConnectionState::Handshaking;

        // Read C0
        let mut c0 = [0u8; 1];
        self.stream.read_exact(&mut c0).await?;
        if c0[0] != RTMP_VERSION {
            return Err(IngestError::HandshakeFailed(format!(
                "Unsupported RTMP version: {}",
                c0[0]
            )));
        }

        // Read C1
        let mut c1 = vec![0u8; RTMP_HANDSHAKE_SIZE];
        self.stream.read_exact(&mut c1).await?;

        // Send S0 + S1 + S2
        let s0 = [RTMP_VERSION];
        let s1 = Self::generate_handshake_data();
        let s2 = c1.clone(); // Echo C1 as S2

        self.stream.write_all(&s0).await?;
        self.stream.write_all(&s1).await?;
        self.stream.write_all(&s2).await?;
        self.stream.flush().await?;

        // Read C2
        let mut c2 = vec![0u8; RTMP_HANDSHAKE_SIZE];
        self.stream.read_exact(&mut c2).await?;

        // Verify C2 matches S1 (simplified verification)
        if c2[..4] != s1[..4] {
            tracing::warn!("C2 timestamp mismatch, continuing anyway");
        }

        self.state = ConnectionState::Connected;
        tracing::info!("RTMP handshake completed with {}", self.peer_addr);

        Ok(())
    }

    /// Perform RTMP handshake as client
    pub async fn perform_client_handshake(&mut self) -> IngestResult<()> {
        self.state = ConnectionState::Handshaking;

        // Send C0 + C1
        let c0 = [RTMP_VERSION];
        let c1 = Self::generate_handshake_data();

        self.stream.write_all(&c0).await?;
        self.stream.write_all(&c1).await?;
        self.stream.flush().await?;

        // Read S0
        let mut s0 = [0u8; 1];
        self.stream.read_exact(&mut s0).await?;
        if s0[0] != RTMP_VERSION {
            return Err(IngestError::HandshakeFailed(format!(
                "Unsupported RTMP version: {}",
                s0[0]
            )));
        }

        // Read S1
        let mut s1 = vec![0u8; RTMP_HANDSHAKE_SIZE];
        self.stream.read_exact(&mut s1).await?;

        // Read S2
        let mut s2 = vec![0u8; RTMP_HANDSHAKE_SIZE];
        self.stream.read_exact(&mut s2).await?;

        // Verify S2 matches C1
        if s2[..4] != c1[..4] {
            tracing::warn!("S2 timestamp mismatch, continuing anyway");
        }

        // Send C2 (echo S1)
        self.stream.write_all(&s1).await?;
        self.stream.flush().await?;

        self.state = ConnectionState::Connected;
        tracing::info!("RTMP client handshake completed");

        Ok(())
    }

    /// Generate handshake random data
    fn generate_handshake_data() -> Vec<u8> {
        let mut data = vec![0u8; RTMP_HANDSHAKE_SIZE];

        // First 4 bytes: timestamp
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u32;
        data[0..4].copy_from_slice(&timestamp.to_be_bytes());

        // Next 4 bytes: zero (for simple handshake)
        data[4..8].copy_from_slice(&[0, 0, 0, 0]);

        // Remaining bytes: random data
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.fill(&mut data[8..]);

        data
    }

    /// Read a single RTMP message
    pub async fn read_message(&mut self) -> IngestResult<Option<RtmpMessage>> {
        loop {
            // Try to parse a complete message from buffer
            if let Some(msg) = self.try_parse_message()? {
                return Ok(Some(msg));
            }

            // Need more data
            let mut buf = [0u8; 4096];
            let n = self.stream.read(&mut buf).await?;
            if n == 0 {
                if self.read_buffer.is_empty() {
                    return Ok(None);
                } else {
                    return Err(IngestError::ConnectionClosed);
                }
            }

            self.read_buffer.extend_from_slice(&buf[..n]);
            self.bytes_received += n as u64;

            // Send acknowledgement if needed
            if self.bytes_received - self.last_ack_sent >= self.window_ack_size as u64 {
                self.send_acknowledgement().await?;
            }
        }
    }

    /// Try to parse a complete message from the buffer
    fn try_parse_message(&mut self) -> IngestResult<Option<RtmpMessage>> {
        if self.read_buffer.is_empty() {
            return Ok(None);
        }

        // Parse chunk basic header
        let first_byte = self.read_buffer[0];
        let fmt = (first_byte >> 6) & 0x03;
        let cs_id_basic = first_byte & 0x3F;

        let (chunk_stream_id, basic_header_len) = match cs_id_basic {
            0 => {
                if self.read_buffer.len() < 2 {
                    return Ok(None);
                }
                ((self.read_buffer[1] as u32) + 64, 2)
            }
            1 => {
                if self.read_buffer.len() < 3 {
                    return Ok(None);
                }
                let id = ((self.read_buffer[2] as u32) << 8) + (self.read_buffer[1] as u32) + 64;
                (id, 3)
            }
            _ => (cs_id_basic as u32, 1),
        };

        // Determine message header size based on format
        let msg_header_len = match fmt {
            0 => 11, // Full header
            1 => 7,  // Without message stream ID
            2 => 3,  // Only timestamp delta
            3 => 0,  // No header (use previous)
            _ => unreachable!(),
        };

        let total_header_len = basic_header_len + msg_header_len;
        if self.read_buffer.len() < total_header_len {
            return Ok(None);
        }

        // Get or create chunk stream state
        let prev_state = self.chunk_streams.get(&chunk_stream_id).cloned();

        // Parse message header
        let header = self.parse_chunk_header(
            fmt,
            chunk_stream_id,
            basic_header_len,
            prev_state.as_ref().and_then(|s| s.last_header.as_ref()),
        )?;

        // Check for extended timestamp
        let mut extended_timestamp_len = 0;
        if header.timestamp >= 0xFFFFFF {
            if self.read_buffer.len() < total_header_len + 4 {
                return Ok(None);
            }
            extended_timestamp_len = 4;
        }

        let full_header_len = total_header_len + extended_timestamp_len;

        // Calculate how much payload we need/have
        let state = self
            .chunk_streams
            .entry(chunk_stream_id)
            .or_default();

        let payload_needed = if let Some(ref incomplete) = state.incomplete_message {
            header.message_length as usize - incomplete.len()
        } else {
            header.message_length as usize
        };

        let chunk_payload_len = std::cmp::min(payload_needed, self.read_chunk_size as usize);

        if self.read_buffer.len() < full_header_len + chunk_payload_len {
            return Ok(None);
        }

        // Consume header
        self.read_buffer.advance(full_header_len);

        // Read payload chunk
        let payload_chunk = self.read_buffer.split_to(chunk_payload_len);

        // Accumulate payload
        let state = self.chunk_streams.get_mut(&chunk_stream_id).unwrap();
        if state.incomplete_message.is_none() {
            state.incomplete_message = Some(BytesMut::with_capacity(header.message_length as usize));
        }

        state
            .incomplete_message
            .as_mut()
            .unwrap()
            .extend_from_slice(&payload_chunk);

        // Check if message is complete
        if state.incomplete_message.as_ref().unwrap().len() >= header.message_length as usize {
            let payload = state.incomplete_message.take().unwrap().freeze();
            state.last_header = Some(header.clone());

            return Ok(Some(RtmpMessage { header, payload }));
        }

        // Message incomplete, need more chunks
        state.last_header = Some(header);
        Ok(None)
    }

    /// Parse chunk header based on format type
    fn parse_chunk_header(
        &self,
        fmt: u8,
        chunk_stream_id: u32,
        basic_header_len: usize,
        prev_header: Option<&ChunkHeader>,
    ) -> IngestResult<ChunkHeader> {
        let data = &self.read_buffer[basic_header_len..];

        match fmt {
            0 => {
                // Type 0: Full header (11 bytes)
                let timestamp =
                    ((data[0] as u32) << 16) | ((data[1] as u32) << 8) | (data[2] as u32);
                let message_length =
                    ((data[3] as u32) << 16) | ((data[4] as u32) << 8) | (data[5] as u32);
                let message_type_id = data[6];
                let message_stream_id = u32::from_le_bytes([data[7], data[8], data[9], data[10]]);

                Ok(ChunkHeader {
                    chunk_stream_id,
                    timestamp,
                    message_length,
                    message_type_id,
                    message_stream_id,
                    timestamp_delta: false,
                })
            }
            1 => {
                // Type 1: Without stream ID (7 bytes)
                let timestamp_delta =
                    ((data[0] as u32) << 16) | ((data[1] as u32) << 8) | (data[2] as u32);
                let message_length =
                    ((data[3] as u32) << 16) | ((data[4] as u32) << 8) | (data[5] as u32);
                let message_type_id = data[6];

                let prev = prev_header.ok_or_else(|| {
                    IngestError::ProtocolError("Type 1 chunk without previous header".into())
                })?;

                Ok(ChunkHeader {
                    chunk_stream_id,
                    timestamp: prev.timestamp + timestamp_delta,
                    message_length,
                    message_type_id,
                    message_stream_id: prev.message_stream_id,
                    timestamp_delta: true,
                })
            }
            2 => {
                // Type 2: Only timestamp delta (3 bytes)
                let timestamp_delta =
                    ((data[0] as u32) << 16) | ((data[1] as u32) << 8) | (data[2] as u32);

                let prev = prev_header.ok_or_else(|| {
                    IngestError::ProtocolError("Type 2 chunk without previous header".into())
                })?;

                Ok(ChunkHeader {
                    chunk_stream_id,
                    timestamp: prev.timestamp + timestamp_delta,
                    message_length: prev.message_length,
                    message_type_id: prev.message_type_id,
                    message_stream_id: prev.message_stream_id,
                    timestamp_delta: true,
                })
            }
            3 => {
                // Type 3: No header, use previous
                let prev = prev_header.ok_or_else(|| {
                    IngestError::ProtocolError("Type 3 chunk without previous header".into())
                })?;

                Ok(prev.clone())
            }
            _ => Err(IngestError::ProtocolError(format!(
                "Invalid chunk format: {}",
                fmt
            ))),
        }
    }

    /// Send an RTMP message
    pub async fn send_message(&mut self, msg: &RtmpMessage) -> IngestResult<()> {
        let mut buf = BytesMut::new();

        // Write chunk basic header (simplified - always use format 0)
        if msg.header.chunk_stream_id < 64 {
            buf.put_u8(msg.header.chunk_stream_id as u8);
        } else if msg.header.chunk_stream_id < 320 {
            buf.put_u8(0);
            buf.put_u8((msg.header.chunk_stream_id - 64) as u8);
        } else {
            buf.put_u8(1);
            let id = msg.header.chunk_stream_id - 64;
            buf.put_u8((id & 0xFF) as u8);
            buf.put_u8(((id >> 8) & 0xFF) as u8);
        }

        // Write message header (format 0)
        let timestamp = if msg.header.timestamp >= 0xFFFFFF {
            0xFFFFFF
        } else {
            msg.header.timestamp
        };

        buf.put_u8(((timestamp >> 16) & 0xFF) as u8);
        buf.put_u8(((timestamp >> 8) & 0xFF) as u8);
        buf.put_u8((timestamp & 0xFF) as u8);

        buf.put_u8(((msg.header.message_length >> 16) & 0xFF) as u8);
        buf.put_u8(((msg.header.message_length >> 8) & 0xFF) as u8);
        buf.put_u8((msg.header.message_length & 0xFF) as u8);

        buf.put_u8(msg.header.message_type_id);
        buf.put_u32_le(msg.header.message_stream_id);

        // Extended timestamp if needed
        if msg.header.timestamp >= 0xFFFFFF {
            buf.put_u32(msg.header.timestamp);
        }

        // Write payload in chunks
        let mut remaining = &msg.payload[..];
        let mut first_chunk = true;

        while !remaining.is_empty() {
            if !first_chunk {
                // Write type 3 chunk header
                if msg.header.chunk_stream_id < 64 {
                    buf.put_u8(0xC0 | msg.header.chunk_stream_id as u8);
                } else if msg.header.chunk_stream_id < 320 {
                    buf.put_u8(0xC0);
                    buf.put_u8((msg.header.chunk_stream_id - 64) as u8);
                } else {
                    buf.put_u8(0xC1);
                    let id = msg.header.chunk_stream_id - 64;
                    buf.put_u8((id & 0xFF) as u8);
                    buf.put_u8(((id >> 8) & 0xFF) as u8);
                }
            }

            let chunk_size = std::cmp::min(remaining.len(), self.write_chunk_size as usize);
            buf.put_slice(&remaining[..chunk_size]);
            remaining = &remaining[chunk_size..];
            first_chunk = false;
        }

        self.stream.write_all(&buf).await?;
        self.stream.flush().await?;

        Ok(())
    }

    /// Send acknowledgement
    async fn send_acknowledgement(&mut self) -> IngestResult<()> {
        let mut payload = BytesMut::new();
        payload.put_u32(self.bytes_received as u32);

        let msg = RtmpMessage {
            header: ChunkHeader {
                chunk_stream_id: 2,
                timestamp: 0,
                message_length: 4,
                message_type_id: MSG_ACK,
                message_stream_id: 0,
                timestamp_delta: false,
            },
            payload: payload.freeze(),
        };

        self.send_message(&msg).await?;
        self.last_ack_sent = self.bytes_received;

        Ok(())
    }

    /// Set chunk size
    pub async fn set_chunk_size(&mut self, size: u32) -> IngestResult<()> {
        let size = size.min(RTMP_MAX_CHUNK_SIZE);

        let mut payload = BytesMut::new();
        payload.put_u32(size);

        let msg = RtmpMessage {
            header: ChunkHeader {
                chunk_stream_id: 2,
                timestamp: 0,
                message_length: 4,
                message_type_id: MSG_SET_CHUNK_SIZE,
                message_stream_id: 0,
                timestamp_delta: false,
            },
            payload: payload.freeze(),
        };

        self.send_message(&msg).await?;
        self.write_chunk_size = size;

        Ok(())
    }

    /// Handle a command message
    pub async fn handle_command(&mut self, payload: &[u8]) -> IngestResult<Option<RtmpCommand>> {
        let mut data = BytesMut::from(payload);
        let command_name = Amf0Value::parse(&mut data)?;
        let transaction_id = Amf0Value::parse(&mut data)?;

        let command_name = command_name
            .as_string()
            .ok_or_else(|| IngestError::InvalidMessage("Command name must be string".into()))?
            .to_string();

        let transaction_id = transaction_id.as_number().unwrap_or(0.0);

        // Parse command object (may be null)
        let command_object = if !data.is_empty() {
            Some(Amf0Value::parse(&mut data)?)
        } else {
            None
        };

        // Parse additional arguments
        let mut args = Vec::new();
        while !data.is_empty() {
            args.push(Amf0Value::parse(&mut data)?);
        }

        Ok(Some(RtmpCommand {
            name: command_name,
            transaction_id,
            command_object,
            args,
        }))
    }

    /// Send a command response
    pub async fn send_command_response(
        &mut self,
        command: &str,
        transaction_id: f64,
        properties: Option<HashMap<String, Amf0Value>>,
        info: Option<HashMap<String, Amf0Value>>,
    ) -> IngestResult<()> {
        let mut payload = BytesMut::new();

        Amf0Value::String(command.to_string()).encode(&mut payload);
        Amf0Value::Number(transaction_id).encode(&mut payload);

        if let Some(props) = properties {
            Amf0Value::Object(props).encode(&mut payload);
        } else {
            Amf0Value::Null.encode(&mut payload);
        }

        if let Some(info) = info {
            Amf0Value::Object(info).encode(&mut payload);
        }

        let msg = RtmpMessage {
            header: ChunkHeader {
                chunk_stream_id: 3,
                timestamp: 0,
                message_length: payload.len() as u32,
                message_type_id: MSG_COMMAND_AMF0,
                message_stream_id: 0,
                timestamp_delta: false,
            },
            payload: payload.freeze(),
        };

        self.send_message(&msg).await
    }

    /// Process incoming messages and handle protocol
    pub async fn process_messages(&mut self) -> IngestResult<Option<MediaData>> {
        loop {
            let msg = match self.read_message().await? {
                Some(msg) => msg,
                None => return Ok(None),
            };

            match msg.header.message_type_id {
                MSG_SET_CHUNK_SIZE => {
                    if msg.payload.len() >= 4 {
                        let mut data = &msg.payload[..];
                        self.read_chunk_size = data.get_u32();
                        tracing::debug!("Chunk size set to {}", self.read_chunk_size);
                    }
                }
                MSG_ABORT => {
                    if msg.payload.len() >= 4 {
                        let mut data = &msg.payload[..];
                        let cs_id = data.get_u32();
                        self.chunk_streams.remove(&cs_id);
                    }
                }
                MSG_ACK => {
                    // Acknowledgement received, no action needed
                }
                MSG_WINDOW_ACK_SIZE => {
                    if msg.payload.len() >= 4 {
                        let mut data = &msg.payload[..];
                        self.window_ack_size = data.get_u32();
                    }
                }
                MSG_SET_PEER_BANDWIDTH => {
                    // Peer bandwidth set, send window ack size response
                    let mut payload = BytesMut::new();
                    payload.put_u32(self.window_ack_size);

                    let response = RtmpMessage {
                        header: ChunkHeader {
                            chunk_stream_id: 2,
                            timestamp: 0,
                            message_length: 4,
                            message_type_id: MSG_WINDOW_ACK_SIZE,
                            message_stream_id: 0,
                            timestamp_delta: false,
                        },
                        payload: payload.freeze(),
                    };
                    self.send_message(&response).await?;
                }
                MSG_AUDIO => {
                    return Ok(Some(self.parse_audio_data(&msg)?));
                }
                MSG_VIDEO => {
                    return Ok(Some(self.parse_video_data(&msg)?));
                }
                MSG_DATA_AMF0 | MSG_DATA_AMF3 => {
                    if let Some(metadata) = self.parse_metadata(&msg)? {
                        return Ok(Some(metadata));
                    }
                }
                MSG_COMMAND_AMF0 | MSG_COMMAND_AMF3 => {
                    if let Some(cmd) = self.handle_command(&msg.payload).await? {
                        self.process_command(cmd).await?;
                    }
                }
                MSG_USER_CONTROL => {
                    // Handle user control messages
                    self.handle_user_control(&msg)?;
                }
                _ => {
                    tracing::debug!("Unhandled message type: {}", msg.header.message_type_id);
                }
            }
        }
    }

    /// Parse audio data from RTMP message
    fn parse_audio_data(&self, msg: &RtmpMessage) -> IngestResult<MediaData> {
        if msg.payload.is_empty() {
            return Err(IngestError::InvalidMessage("Empty audio data".into()));
        }

        let first_byte = msg.payload[0];
        let codec_id = (first_byte >> 4) & 0x0F;
        let sample_rate_idx = (first_byte >> 2) & 0x03;
        let _sample_size = (first_byte >> 1) & 0x01;
        let channels = (first_byte & 0x01) + 1;

        let codec = match codec_id {
            10 => AudioCodec::Aac,
            2 => AudioCodec::Mp3,
            0 | 3 => AudioCodec::Pcm,
            11 => AudioCodec::Speex,
            _ => AudioCodec::Unknown(codec_id),
        };

        let sample_rate = match sample_rate_idx {
            0 => 5500,
            1 => 11025,
            2 => 22050,
            3 => 44100,
            _ => 44100,
        };

        Ok(MediaData::Audio {
            data: msg.payload.slice(1..),
            timestamp: msg.header.timestamp,
            codec,
            sample_rate,
            channels,
        })
    }

    /// Parse video data from RTMP message
    fn parse_video_data(&self, msg: &RtmpMessage) -> IngestResult<MediaData> {
        if msg.payload.is_empty() {
            return Err(IngestError::InvalidMessage("Empty video data".into()));
        }

        let first_byte = msg.payload[0];
        let frame_type = (first_byte >> 4) & 0x0F;
        let codec_id = first_byte & 0x0F;

        let codec = match codec_id {
            7 => VideoCodec::H264,
            12 => VideoCodec::H265,
            _ => VideoCodec::Unknown(codec_id),
        };

        let is_keyframe = frame_type == 1;

        let composition_time = if codec == VideoCodec::H264 && msg.payload.len() >= 5 {
            let b = &msg.payload[2..5];
            ((b[0] as i32) << 16) | ((b[1] as i32) << 8) | (b[2] as i32)
        } else {
            0
        };

        Ok(MediaData::Video {
            data: msg.payload.slice(1..),
            timestamp: msg.header.timestamp,
            codec,
            is_keyframe,
            composition_time,
        })
    }

    /// Parse metadata from data message
    fn parse_metadata(&self, msg: &RtmpMessage) -> IngestResult<Option<MediaData>> {
        let mut data = BytesMut::from(&msg.payload[..]);

        // Skip AMF3 marker if present
        if msg.header.message_type_id == MSG_DATA_AMF3 && !data.is_empty() {
            data.advance(1);
        }

        // Parse metadata name
        let name = Amf0Value::parse(&mut data)?;
        let name = name.as_string().unwrap_or("");

        if name != "@setDataFrame" && name != "onMetaData" {
            return Ok(None);
        }

        // Skip "@setDataFrame" marker if present
        if name == "@setDataFrame" {
            let _ = Amf0Value::parse(&mut data)?;
        }

        // Parse metadata object
        let metadata_value = Amf0Value::parse(&mut data)?;

        let properties = if let Some(obj) = metadata_value.as_object() {
            obj.iter()
                .map(|(k, v)| {
                    let value = match v {
                        Amf0Value::Number(n) => MetadataValue::Number(*n),
                        Amf0Value::Boolean(b) => MetadataValue::Boolean(*b),
                        Amf0Value::String(s) => MetadataValue::String(s.clone()),
                        Amf0Value::Null => MetadataValue::Null,
                        _ => MetadataValue::String(format!("{:?}", v)),
                    };
                    (k.clone(), value)
                })
                .collect()
        } else {
            Vec::new()
        };

        Ok(Some(MediaData::Metadata { properties }))
    }

    /// Process RTMP command
    async fn process_command(&mut self, cmd: RtmpCommand) -> IngestResult<()> {
        tracing::debug!("Processing command: {}", cmd.name);

        match cmd.name.as_str() {
            "connect" => {
                self.handle_connect(cmd).await?;
            }
            "releaseStream" | "FCPublish" | "FCUnpublish" => {
                // These are optional commands, just acknowledge them
                self.send_command_response("_result", cmd.transaction_id, None, None)
                    .await?;
            }
            "createStream" => {
                self.handle_create_stream(cmd).await?;
            }
            "publish" => {
                self.handle_publish(cmd).await?;
            }
            "play" => {
                self.handle_play(cmd).await?;
            }
            "deleteStream" => {
                self.state = ConnectionState::Closing;
            }
            _ => {
                tracing::debug!("Unhandled command: {}", cmd.name);
            }
        }

        Ok(())
    }

    /// Handle connect command
    async fn handle_connect(&mut self, cmd: RtmpCommand) -> IngestResult<()> {
        // Extract app name from command object
        if let Some(obj) = cmd.command_object.as_ref().and_then(|v| v.as_object()) {
            if let Some(app) = obj.get("app").and_then(|v| v.as_string()) {
                self.stream_info.app = app.to_string();
            }
        }

        // Send window ack size
        let mut payload = BytesMut::new();
        payload.put_u32(self.window_ack_size);
        let msg = RtmpMessage {
            header: ChunkHeader {
                chunk_stream_id: 2,
                timestamp: 0,
                message_length: 4,
                message_type_id: MSG_WINDOW_ACK_SIZE,
                message_stream_id: 0,
                timestamp_delta: false,
            },
            payload: payload.freeze(),
        };
        self.send_message(&msg).await?;

        // Send set peer bandwidth
        let mut payload = BytesMut::new();
        payload.put_u32(self.window_ack_size);
        payload.put_u8(2); // Dynamic
        let msg = RtmpMessage {
            header: ChunkHeader {
                chunk_stream_id: 2,
                timestamp: 0,
                message_length: 5,
                message_type_id: MSG_SET_PEER_BANDWIDTH,
                message_stream_id: 0,
                timestamp_delta: false,
            },
            payload: payload.freeze(),
        };
        self.send_message(&msg).await?;

        // Send set chunk size
        self.set_chunk_size(4096).await?;

        // Send connect result
        let mut props = HashMap::new();
        props.insert("fmsVer".to_string(), Amf0Value::String("FMS/3,0,1,123".to_string()));
        props.insert("capabilities".to_string(), Amf0Value::Number(31.0));

        let mut info = HashMap::new();
        info.insert("level".to_string(), Amf0Value::String("status".to_string()));
        info.insert("code".to_string(), Amf0Value::String("NetConnection.Connect.Success".to_string()));
        info.insert("description".to_string(), Amf0Value::String("Connection succeeded.".to_string()));
        info.insert("objectEncoding".to_string(), Amf0Value::Number(0.0));

        self.send_command_response("_result", cmd.transaction_id, Some(props), Some(info))
            .await?;

        Ok(())
    }

    /// Handle createStream command
    async fn handle_create_stream(&mut self, cmd: RtmpCommand) -> IngestResult<()> {
        let mut payload = BytesMut::new();
        Amf0Value::String("_result".to_string()).encode(&mut payload);
        Amf0Value::Number(cmd.transaction_id).encode(&mut payload);
        Amf0Value::Null.encode(&mut payload);
        Amf0Value::Number(1.0).encode(&mut payload); // Stream ID

        let msg = RtmpMessage {
            header: ChunkHeader {
                chunk_stream_id: 3,
                timestamp: 0,
                message_length: payload.len() as u32,
                message_type_id: MSG_COMMAND_AMF0,
                message_stream_id: 0,
                timestamp_delta: false,
            },
            payload: payload.freeze(),
        };

        self.send_message(&msg).await
    }

    /// Handle publish command
    async fn handle_publish(&mut self, cmd: RtmpCommand) -> IngestResult<()> {
        // Extract stream name from arguments
        if let Some(name_val) = cmd.args.first() {
            if let Some(name) = name_val.as_string() {
                self.stream_info.name = name.to_string();
            }
        }

        self.state = ConnectionState::Publishing;

        // Send onStatus
        let mut info = HashMap::new();
        info.insert("level".to_string(), Amf0Value::String("status".to_string()));
        info.insert("code".to_string(), Amf0Value::String("NetStream.Publish.Start".to_string()));
        info.insert("description".to_string(), Amf0Value::String("Start publishing".to_string()));

        self.send_command_response("onStatus", 0.0, None, Some(info))
            .await?;

        tracing::info!(
            "Stream '{}' started publishing on app '{}'",
            self.stream_info.name,
            self.stream_info.app
        );

        Ok(())
    }

    /// Handle play command
    async fn handle_play(&mut self, cmd: RtmpCommand) -> IngestResult<()> {
        // Extract stream name from arguments
        if let Some(name_val) = cmd.args.first() {
            if let Some(name) = name_val.as_string() {
                self.stream_info.name = name.to_string();
            }
        }

        self.state = ConnectionState::Playing;

        // Send stream begin user control message
        let mut payload = BytesMut::new();
        payload.put_u16(0); // StreamBegin
        payload.put_u32(1); // Stream ID

        let msg = RtmpMessage {
            header: ChunkHeader {
                chunk_stream_id: 2,
                timestamp: 0,
                message_length: 6,
                message_type_id: MSG_USER_CONTROL,
                message_stream_id: 0,
                timestamp_delta: false,
            },
            payload: payload.freeze(),
        };
        self.send_message(&msg).await?;

        // Send onStatus - Reset
        let mut info = HashMap::new();
        info.insert("level".to_string(), Amf0Value::String("status".to_string()));
        info.insert("code".to_string(), Amf0Value::String("NetStream.Play.Reset".to_string()));
        info.insert("description".to_string(), Amf0Value::String("Playing and resetting".to_string()));

        self.send_command_response("onStatus", 0.0, None, Some(info))
            .await?;

        // Send onStatus - Start
        let mut info = HashMap::new();
        info.insert("level".to_string(), Amf0Value::String("status".to_string()));
        info.insert("code".to_string(), Amf0Value::String("NetStream.Play.Start".to_string()));
        info.insert("description".to_string(), Amf0Value::String("Started playing".to_string()));

        self.send_command_response("onStatus", 0.0, None, Some(info))
            .await?;

        tracing::info!(
            "Playing stream '{}' on app '{}'",
            self.stream_info.name,
            self.stream_info.app
        );

        Ok(())
    }

    /// Handle user control message
    fn handle_user_control(&self, msg: &RtmpMessage) -> IngestResult<()> {
        if msg.payload.len() < 2 {
            return Ok(());
        }

        let event_type = ((msg.payload[0] as u16) << 8) | (msg.payload[1] as u16);

        match event_type {
            0 => tracing::debug!("Stream Begin"),
            1 => tracing::debug!("Stream EOF"),
            2 => tracing::debug!("Stream Dry"),
            3 => tracing::debug!("SetBuffer Length"),
            4 => tracing::debug!("Stream Is Recorded"),
            6 => tracing::debug!("Ping Request"),
            7 => tracing::debug!("Ping Response"),
            _ => tracing::debug!("Unknown user control event: {}", event_type),
        }

        Ok(())
    }
}

/// RTMP command representation
#[derive(Debug, Clone)]
pub struct RtmpCommand {
    pub name: String,
    pub transaction_id: f64,
    pub command_object: Option<Amf0Value>,
    pub args: Vec<Amf0Value>,
}

#[async_trait]
impl IngestSource for RtmpConnection {
    fn stream_info(&self) -> &StreamInfo {
        &self.stream_info
    }

    fn state(&self) -> ConnectionState {
        self.state
    }

    async fn read_media(&mut self) -> IngestResult<Option<MediaData>> {
        self.process_messages().await
    }

    async fn write_media(&mut self, data: MediaData) -> IngestResult<()> {
        let (msg_type, payload, timestamp) = match data {
            MediaData::Audio {
                data,
                timestamp,
                codec,
                ..
            } => {
                let mut buf = BytesMut::new();
                let codec_id = match codec {
                    AudioCodec::Aac => 10,
                    AudioCodec::Mp3 => 2,
                    AudioCodec::Pcm => 0,
                    AudioCodec::Speex => 11,
                    AudioCodec::Opus => 13,
                    AudioCodec::Unknown(id) => id,
                };
                buf.put_u8((codec_id << 4) | 0x0F); // codec + stereo 44100
                buf.put_slice(&data);
                (MSG_AUDIO, buf.freeze(), timestamp)
            }
            MediaData::Video {
                data,
                timestamp,
                codec,
                is_keyframe,
                ..
            } => {
                let mut buf = BytesMut::new();
                let frame_type = if is_keyframe { 1 } else { 2 };
                let codec_id = match codec {
                    VideoCodec::H264 => 7,
                    VideoCodec::H265 => 12,
                    VideoCodec::Vp8 => 8,
                    VideoCodec::Vp9 => 9,
                    VideoCodec::Av1 => 13,
                    VideoCodec::Unknown(id) => id,
                };
                buf.put_u8((frame_type << 4) | codec_id);
                buf.put_slice(&data);
                (MSG_VIDEO, buf.freeze(), timestamp)
            }
            MediaData::Metadata { properties } => {
                let mut buf = BytesMut::new();
                Amf0Value::String("@setDataFrame".to_string()).encode(&mut buf);
                Amf0Value::String("onMetaData".to_string()).encode(&mut buf);

                let mut obj = HashMap::new();
                for (key, value) in properties {
                    let amf_value = match value {
                        MetadataValue::Number(n) => Amf0Value::Number(n),
                        MetadataValue::Boolean(b) => Amf0Value::Boolean(b),
                        MetadataValue::String(s) => Amf0Value::String(s),
                        MetadataValue::Null => Amf0Value::Null,
                    };
                    obj.insert(key, amf_value);
                }
                Amf0Value::Object(obj).encode(&mut buf);

                (MSG_DATA_AMF0, buf.freeze(), 0)
            }
        };

        let msg = RtmpMessage {
            header: ChunkHeader {
                chunk_stream_id: if msg_type == MSG_AUDIO { 4 } else { 6 },
                timestamp,
                message_length: payload.len() as u32,
                message_type_id: msg_type,
                message_stream_id: 1,
                timestamp_delta: false,
            },
            payload,
        };

        self.send_message(&msg).await
    }

    async fn close(&mut self) -> IngestResult<()> {
        self.state = ConnectionState::Closing;
        self.stream.shutdown().await?;
        self.state = ConnectionState::Disconnected;
        Ok(())
    }
}

/// RTMP Server
pub struct RtmpServer {
    listener: TcpListener,
    config: IngestConfig,
}

impl RtmpServer {
    /// Bind to an address and create a new RTMP server
    pub async fn bind(addr: SocketAddr) -> IngestResult<Self> {
        Self::bind_with_config(addr, IngestConfig::default()).await
    }

    /// Bind with custom configuration
    pub async fn bind_with_config(addr: SocketAddr, config: IngestConfig) -> IngestResult<Self> {
        let listener = TcpListener::bind(addr).await?;
        tracing::info!("RTMP server listening on {}", addr);
        Ok(Self { listener, config })
    }
}

#[async_trait]
impl IngestServer for RtmpServer {
    type Connection = RtmpConnection;

    async fn accept(&mut self) -> IngestResult<Self::Connection> {
        let (stream, peer_addr) = self.listener.accept().await?;
        tracing::info!("New RTMP connection from {}", peer_addr);

        let mut conn = RtmpConnection::new(stream, peer_addr, self.config.clone());
        conn.perform_server_handshake().await?;

        Ok(conn)
    }

    fn local_addr(&self) -> IngestResult<SocketAddr> {
        Ok(self.listener.local_addr()?)
    }

    async fn shutdown(&mut self) -> IngestResult<()> {
        // TcpListener doesn't have an explicit shutdown, just drop
        Ok(())
    }
}

/// RTMP Client
pub struct RtmpClient;

impl RtmpClient {
    /// Connect to an RTMP server
    pub async fn connect(addr: SocketAddr) -> IngestResult<RtmpConnection> {
        Self::connect_with_config(addr, IngestConfig::default()).await
    }

    /// Connect with custom configuration
    pub async fn connect_with_config(
        addr: SocketAddr,
        config: IngestConfig,
    ) -> IngestResult<RtmpConnection> {
        let stream = tokio::time::timeout(config.connect_timeout, TcpStream::connect(addr))
            .await
            .map_err(|_| IngestError::Timeout("Connection timeout".into()))??;

        let mut conn = RtmpConnection::new(stream, addr, config);
        conn.perform_client_handshake().await?;

        Ok(conn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amf0_number() {
        let mut buf = BytesMut::new();
        Amf0Value::Number(42.5).encode(&mut buf);

        let parsed = Amf0Value::parse(&mut buf).unwrap();
        assert!(matches!(parsed, Amf0Value::Number(n) if (n - 42.5).abs() < f64::EPSILON));
    }

    #[test]
    fn test_amf0_string() {
        let mut buf = BytesMut::new();
        Amf0Value::String("hello".to_string()).encode(&mut buf);

        let parsed = Amf0Value::parse(&mut buf).unwrap();
        assert!(matches!(parsed, Amf0Value::String(s) if s == "hello"));
    }

    #[test]
    fn test_amf0_boolean() {
        let mut buf = BytesMut::new();
        Amf0Value::Boolean(true).encode(&mut buf);

        let parsed = Amf0Value::parse(&mut buf).unwrap();
        assert!(matches!(parsed, Amf0Value::Boolean(true)));
    }

    #[test]
    fn test_amf0_null() {
        let mut buf = BytesMut::new();
        Amf0Value::Null.encode(&mut buf);

        let parsed = Amf0Value::parse(&mut buf).unwrap();
        assert!(matches!(parsed, Amf0Value::Null));
    }

    #[test]
    fn test_amf0_object() {
        let mut obj = HashMap::new();
        obj.insert("key".to_string(), Amf0Value::Number(123.0));
        obj.insert("name".to_string(), Amf0Value::String("test".to_string()));

        let mut buf = BytesMut::new();
        Amf0Value::Object(obj).encode(&mut buf);

        let parsed = Amf0Value::parse(&mut buf).unwrap();
        if let Amf0Value::Object(parsed_obj) = parsed {
            assert!(parsed_obj.contains_key("key"));
            assert!(parsed_obj.contains_key("name"));
        } else {
            panic!("Expected object");
        }
    }

    #[test]
    fn test_handshake_data_generation() {
        let data = RtmpConnection::generate_handshake_data();
        assert_eq!(data.len(), RTMP_HANDSHAKE_SIZE);
        // First 4 bytes should be timestamp
        // Bytes 4-7 should be zero
        assert_eq!(&data[4..8], &[0, 0, 0, 0]);
    }
}
