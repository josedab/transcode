//! RTMP protocol support
//!
//! This module provides RTMP server and client functionality for live streaming.
//! Supports RTMP handshake, chunk processing, and AMF message handling.

use crate::{LiveError, Result, StreamMetadata, StreamPacket};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, mpsc, RwLock};

/// Minimum chunk size per RTMP spec
pub const RTMP_MIN_CHUNK_SIZE: u32 = 128;
/// Maximum chunk size (64KB reasonable limit)
pub const RTMP_MAX_CHUNK_SIZE: u32 = 65536;
/// Default chunk size
pub const RTMP_DEFAULT_CHUNK_SIZE: u32 = 128;
/// Maximum window size (10MB reasonable limit)
pub const RTMP_MAX_WINDOW_SIZE: u32 = 10_000_000;

/// RTMP handshake state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtmpHandshake {
    /// Waiting for C0/C1
    WaitingC0C1,
    /// Waiting for C2
    WaitingC2,
    /// Handshake complete
    Complete,
}

/// RTMP message type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtmpMessageType {
    /// Set chunk size
    SetChunkSize,
    /// Abort message
    Abort,
    /// Acknowledgement
    Acknowledgement,
    /// Window acknowledgement size
    WindowAckSize,
    /// Set peer bandwidth
    SetPeerBandwidth,
    /// Audio data
    Audio,
    /// Video data
    Video,
    /// AMF3 command
    Amf3Command,
    /// AMF0 command
    Amf0Command,
    /// AMF0 data
    Amf0Data,
}

/// RTMP chunk stream
pub struct RtmpChunkStream {
    chunk_size: u32,
    window_size: u32,
    _bytes_received: u64,
}

impl Default for RtmpChunkStream {
    fn default() -> Self {
        Self {
            chunk_size: 128,
            window_size: 2500000,
            _bytes_received: 0,
        }
    }
}

impl RtmpChunkStream {
    /// Create new chunk stream
    pub fn new() -> Self {
        Self::default()
    }

    /// Set chunk size with validation
    ///
    /// # Errors
    ///
    /// Returns an error if the chunk size is outside the valid range
    /// `[RTMP_MIN_CHUNK_SIZE, RTMP_MAX_CHUNK_SIZE]`.
    pub fn set_chunk_size(&mut self, size: u32) -> Result<()> {
        if !(RTMP_MIN_CHUNK_SIZE..=RTMP_MAX_CHUNK_SIZE).contains(&size) {
            return Err(LiveError::Configuration(format!(
                "chunk size {} out of valid range [{}, {}]",
                size, RTMP_MIN_CHUNK_SIZE, RTMP_MAX_CHUNK_SIZE
            )));
        }
        self.chunk_size = size;
        Ok(())
    }

    /// Get chunk size
    pub fn chunk_size(&self) -> u32 {
        self.chunk_size
    }

    /// Set window size with validation
    ///
    /// # Errors
    ///
    /// Returns an error if the window size exceeds `RTMP_MAX_WINDOW_SIZE`.
    pub fn set_window_size(&mut self, size: u32) -> Result<()> {
        if size > RTMP_MAX_WINDOW_SIZE {
            return Err(LiveError::Configuration(format!(
                "window size {} exceeds maximum {}",
                size, RTMP_MAX_WINDOW_SIZE
            )));
        }
        self.window_size = size;
        Ok(())
    }

    /// Get window size
    pub fn window_size(&self) -> u32 {
        self.window_size
    }
}

/// RTMP session
pub struct RtmpSession {
    handshake: RtmpHandshake,
    _chunk_stream: RtmpChunkStream,
    metadata: Option<StreamMetadata>,
}

impl RtmpSession {
    /// Create new RTMP session
    pub fn new() -> Self {
        Self {
            handshake: RtmpHandshake::WaitingC0C1,
            _chunk_stream: RtmpChunkStream::new(),
            metadata: None,
        }
    }

    /// Process handshake data
    pub fn process_handshake(&mut self, data: &[u8]) -> Result<Vec<u8>> {
        match self.handshake {
            RtmpHandshake::WaitingC0C1 => {
                if data.len() < 1537 {
                    return Err(LiveError::Protocol("incomplete C0C1".into()));
                }

                // Generate S0S1S2 response
                let mut response = vec![0u8; 3073];
                response[0] = 3; // RTMP version

                self.handshake = RtmpHandshake::WaitingC2;
                Ok(response)
            }
            RtmpHandshake::WaitingC2 => {
                if data.len() < 1536 {
                    return Err(LiveError::Protocol("incomplete C2".into()));
                }

                self.handshake = RtmpHandshake::Complete;
                Ok(Vec::new())
            }
            RtmpHandshake::Complete => Ok(Vec::new()),
        }
    }

    /// Check if handshake is complete
    pub fn is_handshake_complete(&self) -> bool {
        self.handshake == RtmpHandshake::Complete
    }

    /// Get stream metadata
    pub fn metadata(&self) -> Option<&StreamMetadata> {
        self.metadata.as_ref()
    }
}

impl Default for RtmpSession {
    fn default() -> Self {
        Self::new()
    }
}

/// RTMP server for receiving live streams
pub struct RtmpServer {
    /// Listen address
    addr: std::net::SocketAddr,
    /// Active streams indexed by stream key
    streams: Arc<RwLock<HashMap<String, RtmpStreamState>>>,
    /// Packet broadcaster for subscribers
    packet_tx: broadcast::Sender<StreamPacket>,
    /// Server running flag
    running: Arc<RwLock<bool>>,
}

/// State for an active RTMP stream
#[derive(Debug)]
#[allow(dead_code)]
struct RtmpStreamState {
    /// Stream metadata
    metadata: StreamMetadata,
    /// Number of connected viewers
    viewer_count: usize,
    /// Stream start timestamp
    start_time: std::time::Instant,
}

impl RtmpServer {
    /// Create a new RTMP server
    pub fn new(addr: std::net::SocketAddr) -> Self {
        let (packet_tx, _) = broadcast::channel(1024);
        Self {
            addr,
            streams: Arc::new(RwLock::new(HashMap::new())),
            packet_tx,
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Start the RTMP server
    pub async fn start(&self) -> Result<()> {
        let listener = TcpListener::bind(self.addr)
            .await
            .map_err(|e| LiveError::Connection(e.to_string()))?;

        tracing::info!("RTMP server listening on {}", self.addr);

        {
            let mut running = self.running.write().await;
            *running = true;
        }

        let streams = Arc::clone(&self.streams);
        let packet_tx = self.packet_tx.clone();
        let running = Arc::clone(&self.running);

        tokio::spawn(async move {
            loop {
                {
                    let is_running = running.read().await;
                    if !*is_running {
                        break;
                    }
                }

                match listener.accept().await {
                    Ok((socket, addr)) => {
                        tracing::debug!("New RTMP connection from {}", addr);
                        let streams = Arc::clone(&streams);
                        let packet_tx = packet_tx.clone();

                        tokio::spawn(async move {
                            if let Err(e) =
                                Self::handle_connection(socket, streams, packet_tx).await
                            {
                                tracing::warn!("RTMP connection error from {}: {}", addr, e);
                            }
                        });
                    }
                    Err(e) => {
                        tracing::error!("Failed to accept connection: {}", e);
                    }
                }
            }
        });

        Ok(())
    }

    /// Stop the RTMP server
    pub async fn stop(&self) -> Result<()> {
        let mut running = self.running.write().await;
        *running = false;
        tracing::info!("RTMP server stopped");
        Ok(())
    }

    /// Get a receiver for stream packets
    pub fn subscribe(&self) -> broadcast::Receiver<StreamPacket> {
        self.packet_tx.subscribe()
    }

    /// Get list of active streams
    pub async fn active_streams(&self) -> Vec<String> {
        let streams = self.streams.read().await;
        streams.keys().cloned().collect()
    }

    /// Handle a single RTMP connection
    async fn handle_connection(
        mut socket: TcpStream,
        streams: Arc<RwLock<HashMap<String, RtmpStreamState>>>,
        packet_tx: broadcast::Sender<StreamPacket>,
    ) -> Result<()> {
        let mut session = RtmpSession::new();
        let mut buffer = vec![0u8; 4096];
        let mut read_buffer = Vec::new();

        // Handshake phase
        loop {
            let n = socket
                .read(&mut buffer)
                .await
                .map_err(|e| LiveError::Connection(e.to_string()))?;

            if n == 0 {
                return Err(LiveError::Connection("connection closed".into()));
            }

            read_buffer.extend_from_slice(&buffer[..n]);

            match session.process_handshake(&read_buffer) {
                Ok(response) => {
                    if !response.is_empty() {
                        socket
                            .write_all(&response)
                            .await
                            .map_err(|e| LiveError::Connection(e.to_string()))?;
                    }

                    if session.is_handshake_complete() {
                        read_buffer.clear();
                        break;
                    }
                }
                Err(LiveError::Protocol(_)) => {
                    // Need more data
                    continue;
                }
                Err(e) => return Err(e),
            }
        }

        tracing::debug!("RTMP handshake complete");

        // Message processing phase
        let mut chunk_parser = crate::rtmp_parser::ChunkParser::new();
        let mut stream_key: Option<String> = None;

        loop {
            let n = socket
                .read(&mut buffer)
                .await
                .map_err(|e| LiveError::Connection(e.to_string()))?;

            if n == 0 {
                break;
            }

            read_buffer.extend_from_slice(&buffer[..n]);

            // Parse RTMP messages
            let (messages, consumed) = chunk_parser.parse(&read_buffer)?;
            read_buffer.drain(..consumed);

            for msg in messages {
                match msg.header.message_type_id {
                    crate::rtmp_parser::message_type::SET_CHUNK_SIZE => {
                        if msg.data.len() >= 4 {
                            let size = u32::from_be_bytes([
                                msg.data[0],
                                msg.data[1],
                                msg.data[2],
                                msg.data[3],
                            ]);
                            chunk_parser.set_chunk_size(size);
                        }
                    }
                    crate::rtmp_parser::message_type::AMF0_COMMAND => {
                        // Parse AMF command for connect/publish
                        if let Some(key) = Self::parse_publish_command(&msg.data) {
                            stream_key = Some(key.clone());

                            let mut streams = streams.write().await;
                            streams.insert(
                                key.clone(),
                                RtmpStreamState {
                                    metadata: StreamMetadata {
                                        stream_id: key,
                                        ..Default::default()
                                    },
                                    viewer_count: 0,
                                    start_time: std::time::Instant::now(),
                                },
                            );
                        }
                    }
                    crate::rtmp_parser::message_type::AUDIO
                    | crate::rtmp_parser::message_type::VIDEO => {
                        if let Some(packet) = msg.to_stream_packet() {
                            let _ = packet_tx.send(packet);
                        }
                    }
                    _ => {}
                }
            }
        }

        // Cleanup on disconnect
        if let Some(key) = stream_key {
            let mut streams = streams.write().await;
            streams.remove(&key);
            tracing::info!("Stream {} ended", key);
        }

        Ok(())
    }

    /// Parse a publish command from AMF0 data
    fn parse_publish_command(data: &[u8]) -> Option<String> {
        // Simple AMF0 parsing - look for "publish" command
        // AMF0 string marker is 0x02
        if data.len() < 10 {
            return None;
        }

        // Check for AMF0 string marker
        if data[0] != 0x02 {
            return None;
        }

        // Read string length (big-endian u16)
        let cmd_len = u16::from_be_bytes([data[1], data[2]]) as usize;
        if data.len() < 3 + cmd_len {
            return None;
        }

        let cmd = std::str::from_utf8(&data[3..3 + cmd_len]).ok()?;

        if cmd == "publish" {
            // Skip to stream key (after command name, transaction ID, and null)
            // This is a simplified parser - real implementation needs full AMF0 parsing
            let mut offset = 3 + cmd_len;

            // Skip transaction ID (AMF0 number = 0x00 + 8 bytes)
            if data.len() > offset && data[offset] == 0x00 {
                offset += 9;
            }

            // Skip null (0x05)
            if data.len() > offset && data[offset] == 0x05 {
                offset += 1;
            }

            // Read stream key string
            if data.len() > offset + 3 && data[offset] == 0x02 {
                let key_len = u16::from_be_bytes([data[offset + 1], data[offset + 2]]) as usize;
                if data.len() >= offset + 3 + key_len {
                    return std::str::from_utf8(&data[offset + 3..offset + 3 + key_len])
                        .ok()
                        .map(|s| s.to_string());
                }
            }
        }

        None
    }
}

/// RTMP client for publishing streams
pub struct RtmpClient {
    /// Server URL
    url: url::Url,
    /// Packet sender
    packet_tx: Option<mpsc::Sender<StreamPacket>>,
}

impl RtmpClient {
    /// Create a new RTMP client
    pub fn new(url: &str) -> Result<Self> {
        let url = url::Url::parse(url).map_err(|e| LiveError::InvalidUrl(e.to_string()))?;

        if url.scheme() != "rtmp" {
            return Err(LiveError::InvalidUrl("URL must use rtmp:// scheme".into()));
        }

        Ok(Self {
            url,
            packet_tx: None,
        })
    }

    /// Connect to the RTMP server
    pub async fn connect(&mut self) -> Result<mpsc::Receiver<StreamPacket>> {
        let host = self
            .url
            .host_str()
            .ok_or_else(|| LiveError::InvalidUrl("no host in URL".into()))?;
        let port = self.url.port().unwrap_or(1935);
        let addr = format!("{}:{}", host, port);

        let mut socket = TcpStream::connect(&addr)
            .await
            .map_err(|e| LiveError::Connection(e.to_string()))?;

        // Perform RTMP handshake
        Self::perform_handshake(&mut socket).await?;

        // Create packet channel
        let (tx, rx) = mpsc::channel(1024);
        self.packet_tx = Some(tx);

        // Start background reader
        let packet_tx = self.packet_tx.clone().unwrap();
        tokio::spawn(async move {
            if let Err(e) = Self::read_packets(socket, packet_tx).await {
                tracing::warn!("RTMP client read error: {}", e);
            }
        });

        Ok(rx)
    }

    /// Perform RTMP handshake as client
    async fn perform_handshake(socket: &mut TcpStream) -> Result<()> {
        // Send C0 + C1
        let mut c0c1 = vec![0u8; 1537];
        c0c1[0] = 3; // RTMP version
        socket
            .write_all(&c0c1)
            .await
            .map_err(|e| LiveError::Connection(e.to_string()))?;

        // Read S0 + S1 + S2
        let mut s0s1s2 = vec![0u8; 3073];
        socket
            .read_exact(&mut s0s1s2)
            .await
            .map_err(|e| LiveError::Connection(e.to_string()))?;

        // Send C2 (echo S1)
        socket
            .write_all(&s0s1s2[1..1537])
            .await
            .map_err(|e| LiveError::Connection(e.to_string()))?;

        Ok(())
    }

    /// Read packets from the server
    async fn read_packets(
        mut socket: TcpStream,
        packet_tx: mpsc::Sender<StreamPacket>,
    ) -> Result<()> {
        let mut chunk_parser = crate::rtmp_parser::ChunkParser::new();
        let mut buffer = vec![0u8; 4096];
        let mut read_buffer = Vec::new();

        loop {
            let n = socket
                .read(&mut buffer)
                .await
                .map_err(|e| LiveError::Connection(e.to_string()))?;

            if n == 0 {
                break;
            }

            read_buffer.extend_from_slice(&buffer[..n]);

            let (messages, consumed) = chunk_parser.parse(&read_buffer)?;
            read_buffer.drain(..consumed);

            for msg in messages {
                if let Some(packet) = msg.to_stream_packet() {
                    if packet_tx.send(packet).await.is_err() {
                        return Ok(());
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rtmp_server_creation() {
        let addr: std::net::SocketAddr = "127.0.0.1:1935".parse().unwrap();
        let server = RtmpServer::new(addr);
        assert_eq!(server.addr, addr);
    }

    #[test]
    fn test_rtmp_client_url_validation() {
        assert!(RtmpClient::new("rtmp://localhost/live/stream").is_ok());
        assert!(RtmpClient::new("http://localhost/live").is_err());
        assert!(RtmpClient::new("invalid").is_err());
    }

    #[test]
    fn test_parse_publish_command() {
        // Valid AMF0 publish command
        let data = [
            0x02, // String marker
            0x00, 0x07, // Length = 7
            b'p', b'u', b'b', b'l', b'i', b's', b'h', // "publish"
            0x00, // Number marker
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Transaction ID
            0x05, // Null marker
            0x02, // String marker
            0x00, 0x04, // Length = 4
            b't', b'e', b's', b't', // "test"
        ];

        let key = RtmpServer::parse_publish_command(&data);
        assert_eq!(key, Some("test".to_string()));
    }
}
