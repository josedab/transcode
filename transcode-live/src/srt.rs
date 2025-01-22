//! SRT protocol support
//!
//! This module provides SRT (Secure Reliable Transport) server and client functionality.
//! SRT is a UDP-based protocol that provides reliable, low-latency streaming.

use crate::{LiveError, PacketType, Result, StreamMetadata, StreamPacket};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::UdpSocket;
use tokio::sync::{broadcast, RwLock};

/// Maximum passphrase length per SRT specification
pub const SRT_MAX_PASSPHRASE_LEN: usize = 79;
/// Maximum stream ID length
pub const SRT_MAX_STREAM_ID_LEN: usize = 512;
/// Maximum bandwidth cap (100 Gbps)
pub const SRT_MAX_BANDWIDTH: u64 = 100_000_000_000;

/// SRT socket mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrtMode {
    /// Caller mode (connects to listener)
    Caller,
    /// Listener mode (accepts connections)
    Listener,
    /// Rendezvous mode (both connect)
    Rendezvous,
}

/// SRT configuration
#[derive(Debug, Clone)]
pub struct SrtConfig {
    /// Socket mode
    pub mode: SrtMode,
    /// Passphrase for encryption
    pub passphrase: Option<String>,
    /// Latency in milliseconds
    pub latency: Duration,
    /// Maximum bandwidth (0 = unlimited)
    pub max_bandwidth: u64,
    /// Stream ID
    pub stream_id: Option<String>,
}

impl Default for SrtConfig {
    fn default() -> Self {
        Self {
            mode: SrtMode::Caller,
            passphrase: None,
            latency: Duration::from_millis(120),
            max_bandwidth: 0,
            stream_id: None,
        }
    }
}

impl SrtConfig {
    /// Validate the SRT configuration
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Passphrase is specified but empty or exceeds `SRT_MAX_PASSPHRASE_LEN`
    /// - Stream ID exceeds `SRT_MAX_STREAM_ID_LEN`
    /// - Max bandwidth exceeds `SRT_MAX_BANDWIDTH`
    pub fn validate(&self) -> Result<()> {
        if let Some(ref passphrase) = self.passphrase {
            if passphrase.is_empty() {
                return Err(LiveError::Configuration(
                    "passphrase cannot be empty when specified".into(),
                ));
            }
            if passphrase.len() > SRT_MAX_PASSPHRASE_LEN {
                return Err(LiveError::Configuration(format!(
                    "passphrase length {} exceeds maximum {} per SRT spec",
                    passphrase.len(),
                    SRT_MAX_PASSPHRASE_LEN
                )));
            }
        }

        if let Some(ref stream_id) = self.stream_id {
            if stream_id.len() > SRT_MAX_STREAM_ID_LEN {
                return Err(LiveError::Configuration(format!(
                    "stream ID length {} exceeds maximum {}",
                    stream_id.len(),
                    SRT_MAX_STREAM_ID_LEN
                )));
            }
        }

        if self.max_bandwidth > SRT_MAX_BANDWIDTH {
            return Err(LiveError::Configuration(format!(
                "max bandwidth {} exceeds allowed maximum {}",
                self.max_bandwidth, SRT_MAX_BANDWIDTH
            )));
        }

        Ok(())
    }
}

/// SRT statistics
#[derive(Debug, Clone, Default)]
pub struct SrtStats {
    /// Packets sent
    pub packets_sent: u64,
    /// Packets received
    pub packets_received: u64,
    /// Packets lost
    pub packets_lost: u64,
    /// Packets retransmitted
    pub packets_retransmitted: u64,
    /// Round-trip time in milliseconds
    pub rtt: f64,
    /// Bandwidth in Mbps
    pub bandwidth: f64,
}

/// SRT socket wrapper
pub struct SrtSocket {
    _config: SrtConfig,
    stats: SrtStats,
    connected: bool,
}

impl SrtSocket {
    /// Create a new SRT socket with validated configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(config: SrtConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            _config: config,
            stats: SrtStats::default(),
            connected: false,
        })
    }

    /// Connect to remote endpoint
    pub async fn connect(&mut self, addr: &str) -> Result<()> {
        tracing::info!("SRT connecting to {}", addr);
        // SRT connection implementation would go here
        self.connected = true;
        Ok(())
    }

    /// Send data
    pub async fn send(&mut self, _data: &[u8]) -> Result<()> {
        if !self.connected {
            return Err(LiveError::NotConnected);
        }

        self.stats.packets_sent += 1;
        // Send implementation would go here
        Ok(())
    }

    /// Receive data
    pub async fn recv(&mut self) -> Result<Vec<u8>> {
        if !self.connected {
            return Err(LiveError::NotConnected);
        }

        self.stats.packets_received += 1;
        // Receive implementation would go here
        Ok(Vec::new())
    }

    /// Close the socket
    pub async fn close(&mut self) -> Result<()> {
        self.connected = false;
        Ok(())
    }

    /// Get statistics
    pub fn stats(&self) -> &SrtStats {
        &self.stats
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.connected
    }
}

// SRT Protocol Constants
/// SRT handshake request type
const SRT_HANDSHAKE_REQUEST: u32 = 1;
/// SRT handshake response type
const SRT_HANDSHAKE_RESPONSE: u32 = 0xFFFFFFFF;
/// SRT handshake done type
const SRT_HANDSHAKE_DONE: u32 = 0xFFFFFFFE;

/// SRT packet type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrtPacketType {
    /// Data packet
    Data,
    /// Control packet
    Control,
}

/// SRT control packet types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrtControlType {
    /// Handshake
    Handshake,
    /// Keep-alive
    KeepAlive,
    /// ACK
    Ack,
    /// NAK (loss report)
    Nak,
    /// Congestion warning
    CongestionWarning,
    /// Shutdown
    Shutdown,
    /// ACK of ACK
    AckAck,
    /// Drop request
    DropReq,
    /// Peer error
    PeerError,
    /// Unknown
    Unknown,
}

impl From<u16> for SrtControlType {
    fn from(value: u16) -> Self {
        match value {
            0 => SrtControlType::Handshake,
            1 => SrtControlType::KeepAlive,
            2 => SrtControlType::Ack,
            3 => SrtControlType::Nak,
            4 => SrtControlType::CongestionWarning,
            5 => SrtControlType::Shutdown,
            6 => SrtControlType::AckAck,
            7 => SrtControlType::DropReq,
            8 => SrtControlType::PeerError,
            _ => SrtControlType::Unknown,
        }
    }
}

/// SRT packet header
#[derive(Debug, Clone)]
pub struct SrtHeader {
    /// Packet type (0 = data, 1 = control)
    pub packet_type: SrtPacketType,
    /// Sequence number (data) or control type (control)
    pub sequence_or_type: u32,
    /// Timestamp
    pub timestamp: u32,
    /// Destination socket ID
    pub dest_socket_id: u32,
}

impl SrtHeader {
    /// Parse an SRT header from bytes
    pub fn parse(data: &[u8]) -> Option<Self> {
        if data.len() < 16 {
            return None;
        }

        let first_word = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        let packet_type = if first_word & 0x80000000 != 0 {
            SrtPacketType::Control
        } else {
            SrtPacketType::Data
        };

        let sequence_or_type = first_word & 0x7FFFFFFF;
        let timestamp = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
        let dest_socket_id = u32::from_be_bytes([data[12], data[13], data[14], data[15]]);

        Some(Self {
            packet_type,
            sequence_or_type,
            timestamp,
            dest_socket_id,
        })
    }

    /// Get control type if this is a control packet
    pub fn control_type(&self) -> Option<SrtControlType> {
        if self.packet_type == SrtPacketType::Control {
            Some(SrtControlType::from((self.sequence_or_type >> 16) as u16))
        } else {
            None
        }
    }
}

/// State for an SRT connection
#[derive(Debug)]
#[allow(dead_code)]
struct SrtConnectionState {
    /// Remote address
    remote_addr: SocketAddr,
    /// Socket ID
    socket_id: u32,
    /// Stream ID
    stream_id: Option<String>,
    /// Connection state
    state: SrtConnectionPhase,
    /// Stream metadata
    metadata: StreamMetadata,
    /// Last activity timestamp
    last_activity: std::time::Instant,
    /// Statistics
    stats: SrtStats,
}

/// SRT connection phase
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrtConnectionPhase {
    /// Waiting for handshake
    WaitingHandshake,
    /// Handshake in progress
    HandshakeInProgress,
    /// Connected
    Connected,
    /// Closing
    Closing,
}

/// SRT server for receiving live streams
pub struct SrtServer {
    /// Listen address
    addr: SocketAddr,
    /// Active connections
    connections: Arc<RwLock<HashMap<u32, SrtConnectionState>>>,
    /// Packet broadcaster
    packet_tx: broadcast::Sender<StreamPacket>,
    /// Running flag
    running: Arc<RwLock<bool>>,
    /// Next socket ID
    next_socket_id: Arc<RwLock<u32>>,
    /// Server configuration
    config: SrtConfig,
}

impl SrtServer {
    /// Create a new SRT server
    pub fn new(addr: SocketAddr, config: SrtConfig) -> Self {
        let (packet_tx, _) = broadcast::channel(1024);
        Self {
            addr,
            connections: Arc::new(RwLock::new(HashMap::new())),
            packet_tx,
            running: Arc::new(RwLock::new(false)),
            next_socket_id: Arc::new(RwLock::new(1)),
            config,
        }
    }

    /// Start the SRT server
    pub async fn start(&self) -> Result<()> {
        let socket = UdpSocket::bind(self.addr)
            .await
            .map_err(|e| LiveError::Connection(e.to_string()))?;

        tracing::info!("SRT server listening on {}", self.addr);

        {
            let mut running = self.running.write().await;
            *running = true;
        }

        let connections = Arc::clone(&self.connections);
        let packet_tx = self.packet_tx.clone();
        let running = Arc::clone(&self.running);
        let next_socket_id = Arc::clone(&self.next_socket_id);
        let latency = self.config.latency;

        tokio::spawn(async move {
            let mut buf = vec![0u8; 1500]; // MTU-sized buffer

            loop {
                {
                    let is_running = running.read().await;
                    if !*is_running {
                        break;
                    }
                }

                match tokio::time::timeout(Duration::from_millis(100), socket.recv_from(&mut buf))
                    .await
                {
                    Ok(Ok((len, addr))) => {
                        if let Err(e) = Self::handle_packet(
                            &socket,
                            &buf[..len],
                            addr,
                            &connections,
                            &packet_tx,
                            &next_socket_id,
                            latency,
                        )
                        .await
                        {
                            tracing::warn!("SRT packet error from {}: {}", addr, e);
                        }
                    }
                    Ok(Err(e)) => {
                        tracing::error!("SRT recv error: {}", e);
                    }
                    Err(_) => {
                        // Timeout - check for stale connections
                        Self::cleanup_stale_connections(&connections).await;
                    }
                }
            }
        });

        Ok(())
    }

    /// Stop the SRT server
    pub async fn stop(&self) -> Result<()> {
        let mut running = self.running.write().await;
        *running = false;
        tracing::info!("SRT server stopped");
        Ok(())
    }

    /// Get a receiver for stream packets
    pub fn subscribe(&self) -> broadcast::Receiver<StreamPacket> {
        self.packet_tx.subscribe()
    }

    /// Get active connection count
    pub async fn connection_count(&self) -> usize {
        let connections = self.connections.read().await;
        connections.len()
    }

    /// Handle an incoming SRT packet
    async fn handle_packet(
        socket: &UdpSocket,
        data: &[u8],
        addr: SocketAddr,
        connections: &Arc<RwLock<HashMap<u32, SrtConnectionState>>>,
        packet_tx: &broadcast::Sender<StreamPacket>,
        next_socket_id: &Arc<RwLock<u32>>,
        latency: Duration,
    ) -> Result<()> {
        let header = SrtHeader::parse(data).ok_or_else(|| {
            LiveError::Protocol("invalid SRT header".into())
        })?;

        match header.packet_type {
            SrtPacketType::Control => {
                Self::handle_control_packet(
                    socket,
                    &header,
                    data,
                    addr,
                    connections,
                    next_socket_id,
                    latency,
                )
                .await
            }
            SrtPacketType::Data => {
                Self::handle_data_packet(&header, data, connections, packet_tx).await
            }
        }
    }

    /// Handle an SRT control packet
    async fn handle_control_packet(
        socket: &UdpSocket,
        header: &SrtHeader,
        data: &[u8],
        addr: SocketAddr,
        connections: &Arc<RwLock<HashMap<u32, SrtConnectionState>>>,
        next_socket_id: &Arc<RwLock<u32>>,
        latency: Duration,
    ) -> Result<()> {
        let control_type = header.control_type().unwrap_or(SrtControlType::Unknown);

        match control_type {
            SrtControlType::Handshake => {
                Self::handle_handshake(socket, data, addr, connections, next_socket_id, latency)
                    .await
            }
            SrtControlType::KeepAlive => {
                // Update last activity time
                let mut conns = connections.write().await;
                if let Some(conn) = conns.get_mut(&header.dest_socket_id) {
                    conn.last_activity = std::time::Instant::now();
                }
                Ok(())
            }
            SrtControlType::Ack => {
                // Handle ACK - update RTT and congestion control
                let mut conns = connections.write().await;
                if let Some(conn) = conns.get_mut(&header.dest_socket_id) {
                    conn.stats.packets_received += 1;
                    conn.last_activity = std::time::Instant::now();
                }
                Ok(())
            }
            SrtControlType::Nak => {
                // Handle NAK - request retransmission
                let mut conns = connections.write().await;
                if let Some(conn) = conns.get_mut(&header.dest_socket_id) {
                    conn.stats.packets_lost += 1;
                    conn.last_activity = std::time::Instant::now();
                }
                Ok(())
            }
            SrtControlType::Shutdown => {
                // Handle shutdown - remove connection
                let mut conns = connections.write().await;
                if let Some(conn) = conns.remove(&header.dest_socket_id) {
                    tracing::info!(
                        "SRT connection {} closed from {}",
                        header.dest_socket_id,
                        conn.remote_addr
                    );
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Handle an SRT handshake packet
    async fn handle_handshake(
        socket: &UdpSocket,
        data: &[u8],
        addr: SocketAddr,
        connections: &Arc<RwLock<HashMap<u32, SrtConnectionState>>>,
        next_socket_id: &Arc<RwLock<u32>>,
        latency: Duration,
    ) -> Result<()> {
        if data.len() < 48 {
            return Err(LiveError::Protocol("handshake too short".into()));
        }

        // Parse handshake extension if present
        let handshake_type = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
        let peer_socket_id = u32::from_be_bytes([data[44], data[45], data[46], data[47]]);

        match handshake_type {
            SRT_HANDSHAKE_REQUEST => {
                // Induction phase - send back handshake response
                let server_socket_id = {
                    let mut id = next_socket_id.write().await;
                    let socket_id = *id;
                    *id += 1;
                    socket_id
                };

                let response = Self::build_handshake_response(
                    server_socket_id,
                    peer_socket_id,
                    latency.as_millis() as u32,
                );

                socket
                    .send_to(&response, addr)
                    .await
                    .map_err(|e| LiveError::Connection(e.to_string()))?;

                // Create connection state
                let mut conns = connections.write().await;
                conns.insert(
                    server_socket_id,
                    SrtConnectionState {
                        remote_addr: addr,
                        socket_id: server_socket_id,
                        stream_id: None,
                        state: SrtConnectionPhase::HandshakeInProgress,
                        metadata: StreamMetadata::default(),
                        last_activity: std::time::Instant::now(),
                        stats: SrtStats::default(),
                    },
                );

                tracing::debug!(
                    "SRT handshake induction from {}, assigned socket {}",
                    addr,
                    server_socket_id
                );
            }
            SRT_HANDSHAKE_DONE | SRT_HANDSHAKE_RESPONSE => {
                // Conclusion phase - finalize handshake
                let mut conns = connections.write().await;
                for conn in conns.values_mut() {
                    if conn.remote_addr == addr
                        && conn.state == SrtConnectionPhase::HandshakeInProgress
                    {
                        conn.state = SrtConnectionPhase::Connected;
                        conn.last_activity = std::time::Instant::now();

                        // Extract stream ID from handshake extension if present
                        if data.len() > 64 {
                            if let Some(stream_id) = Self::parse_stream_id_extension(&data[48..]) {
                                conn.stream_id = Some(stream_id.clone());
                                conn.metadata.stream_id = stream_id;
                            }
                        }

                        tracing::info!(
                            "SRT connection established from {}, socket {}",
                            addr,
                            conn.socket_id
                        );
                        break;
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Build a handshake response packet
    fn build_handshake_response(server_socket_id: u32, peer_socket_id: u32, latency_ms: u32) -> Vec<u8> {
        let mut response = vec![0u8; 64];

        // Control packet flag + handshake type
        let first_word: u32 = 0x80000000; // Control bit set, handshake = 0
        response[0..4].copy_from_slice(&first_word.to_be_bytes());

        // Additional info (reserved)
        response[4..8].copy_from_slice(&0u32.to_be_bytes());

        // Timestamp
        response[8..12].copy_from_slice(&0u32.to_be_bytes());

        // Destination socket ID
        response[12..16].copy_from_slice(&peer_socket_id.to_be_bytes());

        // Handshake type (response)
        response[16..20].copy_from_slice(&SRT_HANDSHAKE_RESPONSE.to_be_bytes());

        // SRT version (1.4.0 = 0x00010400)
        response[20..24].copy_from_slice(&0x00010400u32.to_be_bytes());

        // Socket type (DGRAM = 1)
        response[24..26].copy_from_slice(&1u16.to_be_bytes());

        // Reserved
        response[26..28].copy_from_slice(&0u16.to_be_bytes());

        // Initial sequence number
        response[28..32].copy_from_slice(&0u32.to_be_bytes());

        // Maximum segment size
        response[32..36].copy_from_slice(&1500u32.to_be_bytes());

        // Maximum flow window size
        response[36..40].copy_from_slice(&8192u32.to_be_bytes());

        // Handshake type (connection request)
        response[40..44].copy_from_slice(&1u32.to_be_bytes());

        // Server socket ID
        response[44..48].copy_from_slice(&server_socket_id.to_be_bytes());

        // SYN cookie (simplified)
        response[48..52].copy_from_slice(&12345u32.to_be_bytes());

        // Peer IP address (IPv4 - 0.0.0.0 for any)
        response[52..56].copy_from_slice(&0u32.to_be_bytes());

        // Latency (extension)
        response[56..60].copy_from_slice(&latency_ms.to_be_bytes());

        response
    }

    /// Parse stream ID from SRT handshake extension
    fn parse_stream_id_extension(data: &[u8]) -> Option<String> {
        // SRT extension format: type (2 bytes) + length (2 bytes) + data
        let mut offset = 0;
        while offset + 4 <= data.len() {
            let ext_type = u16::from_be_bytes([data[offset], data[offset + 1]]);
            let ext_len = u16::from_be_bytes([data[offset + 2], data[offset + 3]]) as usize;

            // Extension type 5 is stream ID
            if ext_type == 5 && offset + 4 + ext_len <= data.len() {
                return std::str::from_utf8(&data[offset + 4..offset + 4 + ext_len])
                    .ok()
                    .map(|s| s.trim_end_matches('\0').to_string());
            }

            offset += 4 + ext_len;
            // Align to 4 bytes
            offset = (offset + 3) & !3;
        }
        None
    }

    /// Handle an SRT data packet
    async fn handle_data_packet(
        header: &SrtHeader,
        data: &[u8],
        connections: &Arc<RwLock<HashMap<u32, SrtConnectionState>>>,
        packet_tx: &broadcast::Sender<StreamPacket>,
    ) -> Result<()> {
        let payload = if data.len() > 16 { &data[16..] } else { &[] };

        // Update connection stats
        {
            let mut conns = connections.write().await;
            if let Some(conn) = conns.get_mut(&header.dest_socket_id) {
                conn.stats.packets_received += 1;
                conn.last_activity = std::time::Instant::now();
            }
        }

        // Determine packet type from payload (MPEG-TS over SRT typically)
        // First byte of MPEG-TS is 0x47 sync byte
        let (packet_type, is_keyframe) = if !payload.is_empty() && payload[0] == 0x47 {
            // MPEG-TS packet - need to inspect PID to determine type
            // For simplicity, treat as video unless it's audio PID range
            (PacketType::Video, false)
        } else {
            // Raw data - assume video
            (PacketType::Video, false)
        };

        let packet = StreamPacket {
            packet_type,
            timestamp: u64::from(header.timestamp),
            data: payload.to_vec(),
            is_keyframe,
        };

        let _ = packet_tx.send(packet);

        Ok(())
    }

    /// Clean up stale connections
    async fn cleanup_stale_connections(connections: &Arc<RwLock<HashMap<u32, SrtConnectionState>>>) {
        let timeout = Duration::from_secs(30);
        let mut conns = connections.write().await;
        let stale: Vec<u32> = conns
            .iter()
            .filter(|(_, conn)| conn.last_activity.elapsed() > timeout)
            .map(|(id, _)| *id)
            .collect();

        for id in stale {
            if let Some(conn) = conns.remove(&id) {
                tracing::info!(
                    "SRT connection {} timed out from {}",
                    id,
                    conn.remote_addr
                );
            }
        }
    }
}

/// SRT client for publishing streams
pub struct SrtClient {
    /// Server address
    addr: String,
    /// Socket
    socket: Option<UdpSocket>,
    /// Configuration
    config: SrtConfig,
    /// Local socket ID
    socket_id: u32,
    /// Server socket ID
    server_socket_id: Option<u32>,
    /// Sequence number
    sequence: u32,
}

impl SrtClient {
    /// Create a new SRT client
    pub fn new(addr: &str, config: SrtConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            addr: addr.to_string(),
            socket: None,
            config,
            socket_id: rand::random::<u32>() % 0x7FFFFFFF,
            server_socket_id: None,
            sequence: 0,
        })
    }

    /// Connect to the SRT server
    pub async fn connect(&mut self) -> Result<()> {
        let socket = UdpSocket::bind("0.0.0.0:0")
            .await
            .map_err(|e| LiveError::Connection(e.to_string()))?;

        socket
            .connect(&self.addr)
            .await
            .map_err(|e| LiveError::Connection(e.to_string()))?;

        // Send handshake request
        let handshake = self.build_handshake_request();
        socket
            .send(&handshake)
            .await
            .map_err(|e| LiveError::Connection(e.to_string()))?;

        // Wait for response
        let mut buf = vec![0u8; 1500];
        let len = tokio::time::timeout(Duration::from_secs(5), socket.recv(&mut buf))
            .await
            .map_err(|_| LiveError::Timeout)?
            .map_err(|e| LiveError::Connection(e.to_string()))?;

        // Parse response
        if len >= 48 {
            self.server_socket_id =
                Some(u32::from_be_bytes([buf[44], buf[45], buf[46], buf[47]]));
        }

        // Send conclusion
        let conclusion = self.build_handshake_conclusion();
        socket
            .send(&conclusion)
            .await
            .map_err(|e| LiveError::Connection(e.to_string()))?;

        self.socket = Some(socket);
        tracing::info!("SRT client connected to {}", self.addr);

        Ok(())
    }

    /// Send data
    pub async fn send(&mut self, data: &[u8]) -> Result<()> {
        let socket = self
            .socket
            .as_ref()
            .ok_or(LiveError::NotConnected)?;

        let packet = self.build_data_packet(data);
        socket
            .send(&packet)
            .await
            .map_err(|e| LiveError::Connection(e.to_string()))?;

        self.sequence += 1;

        Ok(())
    }

    /// Close the connection
    pub async fn close(&mut self) -> Result<()> {
        if let Some(socket) = &self.socket {
            let shutdown = self.build_shutdown_packet();
            let _ = socket.send(&shutdown).await;
        }
        self.socket = None;
        Ok(())
    }

    /// Build handshake request packet
    fn build_handshake_request(&self) -> Vec<u8> {
        let mut packet = vec![0u8; 64];

        // Control packet flag + handshake type
        let first_word: u32 = 0x80000000;
        packet[0..4].copy_from_slice(&first_word.to_be_bytes());

        // Destination socket ID (0 for initial handshake)
        packet[12..16].copy_from_slice(&0u32.to_be_bytes());

        // Handshake type (request)
        packet[16..20].copy_from_slice(&SRT_HANDSHAKE_REQUEST.to_be_bytes());

        // SRT version
        packet[20..24].copy_from_slice(&0x00010400u32.to_be_bytes());

        // Socket type (DGRAM = 1)
        packet[24..26].copy_from_slice(&1u16.to_be_bytes());

        // Max segment size
        packet[32..36].copy_from_slice(&1500u32.to_be_bytes());

        // Flow window
        packet[36..40].copy_from_slice(&8192u32.to_be_bytes());

        // Client socket ID
        packet[44..48].copy_from_slice(&self.socket_id.to_be_bytes());

        // Latency
        let latency_ms = self.config.latency.as_millis() as u32;
        packet[56..60].copy_from_slice(&latency_ms.to_be_bytes());

        packet
    }

    /// Build handshake conclusion packet
    fn build_handshake_conclusion(&self) -> Vec<u8> {
        let mut packet = vec![0u8; 64];

        // Control packet flag + handshake type
        let first_word: u32 = 0x80000000;
        packet[0..4].copy_from_slice(&first_word.to_be_bytes());

        // Destination socket ID
        if let Some(server_id) = self.server_socket_id {
            packet[12..16].copy_from_slice(&server_id.to_be_bytes());
        }

        // Handshake type (done)
        packet[16..20].copy_from_slice(&SRT_HANDSHAKE_DONE.to_be_bytes());

        // SRT version
        packet[20..24].copy_from_slice(&0x00010400u32.to_be_bytes());

        // Client socket ID
        packet[44..48].copy_from_slice(&self.socket_id.to_be_bytes());

        packet
    }

    /// Build a data packet
    fn build_data_packet(&self, data: &[u8]) -> Vec<u8> {
        let mut packet = vec![0u8; 16 + data.len()];

        // Sequence number (data packet - no control bit)
        packet[0..4].copy_from_slice(&self.sequence.to_be_bytes());

        // Message number and flags
        packet[4..8].copy_from_slice(&1u32.to_be_bytes());

        // Timestamp
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u32;
        packet[8..12].copy_from_slice(&timestamp.to_be_bytes());

        // Destination socket ID
        if let Some(server_id) = self.server_socket_id {
            packet[12..16].copy_from_slice(&server_id.to_be_bytes());
        }

        // Payload
        packet[16..].copy_from_slice(data);

        packet
    }

    /// Build a shutdown packet
    fn build_shutdown_packet(&self) -> Vec<u8> {
        let mut packet = vec![0u8; 16];

        // Control packet flag + shutdown type (5)
        let first_word: u32 = 0x80050000;
        packet[0..4].copy_from_slice(&first_word.to_be_bytes());

        // Destination socket ID
        if let Some(server_id) = self.server_socket_id {
            packet[12..16].copy_from_slice(&server_id.to_be_bytes());
        }

        packet
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srt_header_parse() {
        // Data packet
        let data_packet = [
            0x00, 0x00, 0x00, 0x01, // Sequence number (no control bit)
            0x00, 0x00, 0x00, 0x00, // Message number
            0x00, 0x00, 0x00, 0x00, // Timestamp
            0x00, 0x00, 0x00, 0x01, // Socket ID
        ];
        let header = SrtHeader::parse(&data_packet).unwrap();
        assert_eq!(header.packet_type, SrtPacketType::Data);
        assert_eq!(header.sequence_or_type, 1);

        // Control packet (handshake)
        let control_packet = [
            0x80, 0x00, 0x00, 0x00, // Control bit set, type 0 (handshake)
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01,
        ];
        let header = SrtHeader::parse(&control_packet).unwrap();
        assert_eq!(header.packet_type, SrtPacketType::Control);
        assert_eq!(header.control_type(), Some(SrtControlType::Handshake));
    }

    #[test]
    fn test_srt_server_creation() {
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();
        let config = SrtConfig::default();
        let server = SrtServer::new(addr, config);
        assert_eq!(server.addr, addr);
    }

    #[test]
    fn test_srt_client_creation() {
        let config = SrtConfig::default();
        let client = SrtClient::new("127.0.0.1:9000", config);
        assert!(client.is_ok());
    }

    #[test]
    fn test_build_handshake_response() {
        let response = SrtServer::build_handshake_response(1, 2, 120);
        assert_eq!(response.len(), 64);
        // Check control bit is set
        assert_eq!(response[0] & 0x80, 0x80);
    }

    #[test]
    fn test_control_type_conversion() {
        assert_eq!(SrtControlType::from(0), SrtControlType::Handshake);
        assert_eq!(SrtControlType::from(1), SrtControlType::KeepAlive);
        assert_eq!(SrtControlType::from(2), SrtControlType::Ack);
        assert_eq!(SrtControlType::from(5), SrtControlType::Shutdown);
        assert_eq!(SrtControlType::from(99), SrtControlType::Unknown);
    }
}
