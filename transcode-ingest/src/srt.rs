//! SRT (Secure Reliable Transport) protocol implementation
//!
//! This module provides SRT server and client functionality including:
//! - Connection handshake
//! - Reliable UDP transport
//! - Encryption support placeholder
//! - Congestion control

// Protocol constants and incomplete implementations for future expansion
#![allow(dead_code)]

use async_trait::async_trait;
use bytes::{Buf, BufMut, Bytes, BytesMut};
use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::UdpSocket;
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio::time::interval;

use crate::{
    AudioCodec, ConnectionState, IngestConfig, IngestError, IngestResult, IngestServer,
    IngestSource, MediaData, StreamInfo, VideoCodec,
};

// SRT Protocol Constants
const SRT_VERSION: u32 = 0x00010401; // 1.4.1
#[allow(dead_code)]
const SRT_MAGIC_CODE: u16 = 0x4154; // "AT" - handshake magic
const SRT_HDR_SIZE: usize = 16;
const SRT_MAX_PAYLOAD_SIZE: usize = 1316; // Standard MPEG-TS aligned payload

// SRT Packet Types (in control bit)
#[allow(dead_code)]
const SRT_DATA_PACKET: u32 = 0;
const SRT_CONTROL_PACKET: u32 = 1 << 31;

// Control Packet Types
const SRT_CTRL_HANDSHAKE: u16 = 0x0000;
const SRT_CTRL_KEEPALIVE: u16 = 0x0001;
const SRT_CTRL_ACK: u16 = 0x0002;
const SRT_CTRL_NAK: u16 = 0x0003;
const SRT_CTRL_SHUTDOWN: u16 = 0x0005;
const SRT_CTRL_ACKACK: u16 = 0x0006;
const SRT_CTRL_DROPREQ: u16 = 0x0007;

// Handshake Types
const SRT_HS_TYPE_DONE: u32 = 0xFFFFFFFD;
const SRT_HS_TYPE_AGREEMENT: u32 = 0xFFFFFFFE;
const SRT_HS_TYPE_CONCLUSION: u32 = 0xFFFFFFFF;
const SRT_HS_TYPE_INDUCTION: u32 = 0x00000001;

// Socket Types
const SRT_SOCKET_STREAM: u32 = 1;
const SRT_SOCKET_DGRAM: u32 = 2;

// Handshake Extension Types
const SRT_HS_EXT_HSREQ: u16 = 1;
const SRT_HS_EXT_KMREQ: u16 = 3;
const SRT_HS_EXT_CONFIG: u16 = 5;

// Encryption Key Length
const SRT_PBKEYLEN_AES128: u32 = 16;
const SRT_PBKEYLEN_AES192: u32 = 24;
const SRT_PBKEYLEN_AES256: u32 = 32;

/// SRT handshake packet
#[derive(Debug, Clone)]
pub struct SrtHandshake {
    /// UDT version (4)
    pub version: u32,
    /// Socket type (1=STREAM, 2=DGRAM)
    pub socket_type: u32,
    /// Initial sequence number
    pub initial_sequence: u32,
    /// Maximum segment size
    pub max_segment_size: u32,
    /// Maximum flow window size
    pub max_flow_window: u32,
    /// Handshake type
    pub handshake_type: u32,
    /// Socket ID
    pub socket_id: u32,
    /// SYN cookie
    pub syn_cookie: u32,
    /// Peer IP address (for NAT traversal)
    pub peer_ip: [u8; 16],
    /// SRT extension fields
    pub extensions: Vec<SrtExtension>,
}

/// SRT handshake extension
#[derive(Debug, Clone)]
pub struct SrtExtension {
    pub ext_type: u16,
    pub ext_len: u16,
    pub ext_data: Bytes,
}

impl SrtHandshake {
    /// Parse handshake from bytes
    pub fn parse(data: &[u8]) -> IngestResult<Self> {
        if data.len() < 48 {
            return Err(IngestError::InvalidMessage(
                "Handshake packet too short".into(),
            ));
        }

        let mut cursor = data;

        let version = cursor.get_u32();
        let socket_type = cursor.get_u32();
        let initial_sequence = cursor.get_u32();
        let max_segment_size = cursor.get_u32();
        let max_flow_window = cursor.get_u32();
        let handshake_type = cursor.get_u32();
        let socket_id = cursor.get_u32();
        let syn_cookie = cursor.get_u32();

        let mut peer_ip = [0u8; 16];
        peer_ip.copy_from_slice(&cursor[..16]);
        cursor.advance(16);

        // Parse extensions
        let mut extensions = Vec::new();
        while cursor.remaining() >= 4 {
            let ext_type = cursor.get_u16();
            let ext_len = cursor.get_u16();

            if ext_type == 0 {
                break;
            }

            let ext_size = (ext_len as usize) * 4;
            if cursor.remaining() < ext_size {
                break;
            }

            let ext_data = Bytes::copy_from_slice(&cursor[..ext_size]);
            cursor.advance(ext_size);

            extensions.push(SrtExtension {
                ext_type,
                ext_len,
                ext_data,
            });
        }

        Ok(Self {
            version,
            socket_type,
            initial_sequence,
            max_segment_size,
            max_flow_window,
            handshake_type,
            socket_id,
            syn_cookie,
            peer_ip,
            extensions,
        })
    }

    /// Encode handshake to bytes
    pub fn encode(&self) -> BytesMut {
        let mut buf = BytesMut::with_capacity(128);

        buf.put_u32(self.version);
        buf.put_u32(self.socket_type);
        buf.put_u32(self.initial_sequence);
        buf.put_u32(self.max_segment_size);
        buf.put_u32(self.max_flow_window);
        buf.put_u32(self.handshake_type);
        buf.put_u32(self.socket_id);
        buf.put_u32(self.syn_cookie);
        buf.put_slice(&self.peer_ip);

        for ext in &self.extensions {
            buf.put_u16(ext.ext_type);
            buf.put_u16(ext.ext_len);
            buf.put_slice(&ext.ext_data);
        }

        buf
    }

    /// Create HSREQ extension
    pub fn create_hsreq_extension(
        srt_version: u32,
        srt_flags: u32,
        recv_tsbpd_delay: u16,
        send_tsbpd_delay: u16,
    ) -> SrtExtension {
        let mut data = BytesMut::with_capacity(12);
        data.put_u32(srt_version);
        data.put_u32(srt_flags);
        data.put_u16(recv_tsbpd_delay);
        data.put_u16(send_tsbpd_delay);

        SrtExtension {
            ext_type: SRT_HS_EXT_HSREQ,
            ext_len: 3,
            ext_data: data.freeze(),
        }
    }
}

/// SRT packet header
#[derive(Debug, Clone)]
pub struct SrtPacketHeader {
    /// Control flag (1 for control, 0 for data)
    pub is_control: bool,
    /// Sequence number (for data) or packet type (for control)
    pub sequence_or_type: u32,
    /// Timestamp
    pub timestamp: u32,
    /// Destination socket ID
    pub dest_socket_id: u32,
}

impl SrtPacketHeader {
    /// Parse header from bytes
    pub fn parse(data: &[u8]) -> IngestResult<Self> {
        if data.len() < SRT_HDR_SIZE {
            return Err(IngestError::InvalidMessage("Packet header too short".into()));
        }

        let mut cursor = data;
        let first_word = cursor.get_u32();
        let is_control = (first_word & SRT_CONTROL_PACKET) != 0;
        let sequence_or_type = first_word & 0x7FFFFFFF;

        // Skip additional info field for now
        cursor.advance(4);

        let timestamp = cursor.get_u32();
        let dest_socket_id = cursor.get_u32();

        Ok(Self {
            is_control,
            sequence_or_type,
            timestamp,
            dest_socket_id,
        })
    }

    /// Encode header to bytes
    pub fn encode(&self, buf: &mut BytesMut) {
        let first_word = if self.is_control {
            SRT_CONTROL_PACKET | self.sequence_or_type
        } else {
            self.sequence_or_type
        };

        buf.put_u32(first_word);
        buf.put_u32(0); // Additional info
        buf.put_u32(self.timestamp);
        buf.put_u32(self.dest_socket_id);
    }
}

/// SRT ACK packet
#[derive(Debug, Clone)]
pub struct SrtAck {
    /// Last acknowledged sequence number
    pub last_ack_seq: u32,
    /// RTT (round-trip time) in microseconds
    pub rtt: u32,
    /// RTT variance
    pub rtt_var: u32,
    /// Available buffer size
    pub buffer_size: u32,
    /// Packets receiving rate (packets/second)
    pub packet_recv_rate: u32,
    /// Estimated link capacity (packets/second)
    pub link_capacity: u32,
}

impl SrtAck {
    pub fn encode(&self, buf: &mut BytesMut) {
        buf.put_u32(self.last_ack_seq);
        buf.put_u32(self.rtt);
        buf.put_u32(self.rtt_var);
        buf.put_u32(self.buffer_size);
        buf.put_u32(self.packet_recv_rate);
        buf.put_u32(self.link_capacity);
    }

    pub fn parse(data: &[u8]) -> IngestResult<Self> {
        if data.len() < 24 {
            return Err(IngestError::InvalidMessage("ACK packet too short".into()));
        }

        let mut cursor = data;
        Ok(Self {
            last_ack_seq: cursor.get_u32(),
            rtt: cursor.get_u32(),
            rtt_var: cursor.get_u32(),
            buffer_size: cursor.get_u32(),
            packet_recv_rate: cursor.get_u32(),
            link_capacity: cursor.get_u32(),
        })
    }
}

/// SRT NAK (Negative Acknowledgement) packet
#[derive(Debug, Clone)]
pub struct SrtNak {
    /// List of lost sequence numbers or ranges
    pub loss_list: Vec<(u32, u32)>, // (start, end) for ranges
}

impl SrtNak {
    pub fn encode(&self, buf: &mut BytesMut) {
        for (start, end) in &self.loss_list {
            if start == end {
                buf.put_u32(*start);
            } else {
                buf.put_u32(*start | 0x80000000); // Range marker
                buf.put_u32(*end);
            }
        }
    }

    pub fn parse(data: &[u8]) -> IngestResult<Self> {
        let mut loss_list = Vec::new();
        let mut cursor = data;

        while cursor.remaining() >= 4 {
            let first = cursor.get_u32();
            if (first & 0x80000000) != 0 {
                // Range
                if cursor.remaining() < 4 {
                    break;
                }
                let end = cursor.get_u32();
                loss_list.push((first & 0x7FFFFFFF, end));
            } else {
                // Single
                loss_list.push((first, first));
            }
        }

        Ok(Self { loss_list })
    }
}

/// SRT encryption configuration
#[derive(Debug, Clone)]
pub struct SrtEncryption {
    /// Passphrase for key derivation
    pub passphrase: Option<String>,
    /// Key length (16, 24, or 32 bytes)
    pub key_length: u32,
    /// Whether encryption is enabled
    pub enabled: bool,
}

impl Default for SrtEncryption {
    fn default() -> Self {
        Self {
            passphrase: None,
            key_length: SRT_PBKEYLEN_AES128,
            enabled: false,
        }
    }
}

impl SrtEncryption {
    /// Create new encryption config with passphrase
    #[cfg(feature = "srt-encryption")]
    pub fn with_passphrase(passphrase: String, key_length: u32) -> Self {
        Self {
            passphrase: Some(passphrase),
            key_length,
            enabled: true,
        }
    }

    /// Placeholder for key derivation (actual implementation requires crypto)
    pub fn derive_key(&self) -> IngestResult<Option<Vec<u8>>> {
        if !self.enabled {
            return Ok(None);
        }

        #[cfg(feature = "srt-encryption")]
        {
            // In a real implementation, this would use PBKDF2 or similar
            // to derive the encryption key from the passphrase
            Err(IngestError::Unsupported(
                "Encryption requires 'srt-encryption' feature and crypto implementation".into(),
            ))
        }

        #[cfg(not(feature = "srt-encryption"))]
        {
            Err(IngestError::Unsupported(
                "Encryption requires 'srt-encryption' feature".into(),
            ))
        }
    }
}

/// SRT congestion control state
#[derive(Debug)]
struct CongestionControl {
    /// Current sending rate (packets/second)
    send_rate: u32,
    /// Congestion window size
    cwnd: u32,
    /// RTT estimate (microseconds)
    rtt: u32,
    /// RTT variance
    rtt_var: u32,
    /// Last packet send time
    last_send_time: Instant,
    /// Inter-packet interval
    send_interval: Duration,
}

impl Default for CongestionControl {
    fn default() -> Self {
        Self {
            send_rate: 10000,
            cwnd: 16,
            rtt: 100000, // 100ms initial
            rtt_var: 50000,
            last_send_time: Instant::now(),
            send_interval: Duration::from_micros(100),
        }
    }
}

impl CongestionControl {
    fn update_rtt(&mut self, new_rtt: u32) {
        // Exponential moving average
        self.rtt = (self.rtt * 7 + new_rtt) / 8;
        let diff = new_rtt.abs_diff(self.rtt);
        self.rtt_var = (self.rtt_var * 3 + diff) / 4;
    }

    fn on_ack(&mut self) {
        // Increase congestion window on successful ack
        if self.cwnd < 1000 {
            self.cwnd += 1;
        }
    }

    fn on_loss(&mut self) {
        // Reduce sending rate on loss
        self.cwnd = std::cmp::max(2, self.cwnd / 2);
    }
}

/// Receive buffer for reordering and loss recovery
struct ReceiveBuffer {
    /// Buffered packets by sequence number
    packets: HashMap<u32, Bytes>,
    /// Next expected sequence number
    next_seq: u32,
    /// Highest received sequence number
    highest_seq: u32,
    /// Lost packets pending retransmission request
    lost_packets: VecDeque<u32>,
}

impl ReceiveBuffer {
    fn new(initial_seq: u32) -> Self {
        Self {
            packets: HashMap::new(),
            next_seq: initial_seq,
            highest_seq: initial_seq,
            lost_packets: VecDeque::new(),
        }
    }

    fn insert(&mut self, seq: u32, data: Bytes) {
        if seq >= self.next_seq {
            self.packets.insert(seq, data);

            // Track highest sequence
            if seq > self.highest_seq {
                // Mark any gaps as lost
                for missing in (self.highest_seq + 1)..seq {
                    if !self.packets.contains_key(&missing) {
                        self.lost_packets.push_back(missing);
                    }
                }
                self.highest_seq = seq;
            }
        }
    }

    fn get_next(&mut self) -> Option<Bytes> {
        if let Some(data) = self.packets.remove(&self.next_seq) {
            self.next_seq = self.next_seq.wrapping_add(1);
            // Remove from lost list if it was recovered
            self.lost_packets.retain(|&s| s != self.next_seq - 1);
            Some(data)
        } else {
            None
        }
    }

    fn get_loss_list(&mut self) -> Vec<(u32, u32)> {
        let mut ranges = Vec::new();
        let mut current_start: Option<u32> = None;
        let mut current_end: u32 = 0;

        for &seq in &self.lost_packets {
            match current_start {
                None => {
                    current_start = Some(seq);
                    current_end = seq;
                }
                Some(start) => {
                    if seq == current_end + 1 {
                        current_end = seq;
                    } else {
                        ranges.push((start, current_end));
                        current_start = Some(seq);
                        current_end = seq;
                    }
                }
            }
        }

        if let Some(start) = current_start {
            ranges.push((start, current_end));
        }

        // Clear the loss list after returning
        self.lost_packets.clear();

        ranges
    }
}

/// SRT Connection
pub struct SrtConnection {
    socket: Arc<UdpSocket>,
    peer_addr: SocketAddr,
    state: ConnectionState,
    stream_info: StreamInfo,
    config: IngestConfig,

    // Connection identifiers
    local_socket_id: u32,
    peer_socket_id: u32,

    // Sequence numbers
    send_seq: AtomicU32,
    initial_recv_seq: u32,

    // Buffers
    receive_buffer: Mutex<ReceiveBuffer>,
    send_buffer: Mutex<HashMap<u32, (Bytes, Instant)>>,

    // Congestion control
    cc: Mutex<CongestionControl>,

    // Encryption
    encryption: SrtEncryption,

    // Timing
    start_time: Instant,

    // Shutdown flag
    shutdown: AtomicBool,

    // Media data channel
    media_rx: Mutex<Option<mpsc::Receiver<MediaData>>>,
    media_tx: mpsc::Sender<MediaData>,
}

impl SrtConnection {
    /// Create a new SRT connection
    fn new(
        socket: Arc<UdpSocket>,
        peer_addr: SocketAddr,
        local_socket_id: u32,
        peer_socket_id: u32,
        initial_recv_seq: u32,
        config: IngestConfig,
    ) -> Self {
        let (media_tx, media_rx) = mpsc::channel(1000);

        Self {
            socket,
            peer_addr,
            state: ConnectionState::Connected,
            stream_info: StreamInfo {
                name: String::new(),
                app: "srt".to_string(),
                peer_addr,
                connected_at: Instant::now(),
            },
            config,
            local_socket_id,
            peer_socket_id,
            send_seq: AtomicU32::new(0),
            initial_recv_seq,
            receive_buffer: Mutex::new(ReceiveBuffer::new(initial_recv_seq)),
            send_buffer: Mutex::new(HashMap::new()),
            cc: Mutex::new(CongestionControl::default()),
            encryption: SrtEncryption::default(),
            start_time: Instant::now(),
            shutdown: AtomicBool::new(false),
            media_rx: Mutex::new(Some(media_rx)),
            media_tx,
        }
    }

    /// Get timestamp for packets
    fn get_timestamp(&self) -> u32 {
        self.start_time.elapsed().as_micros() as u32
    }

    /// Send a control packet
    async fn send_control(
        &self,
        ctrl_type: u16,
        additional_info: u32,
        payload: &[u8],
    ) -> IngestResult<()> {
        let mut buf = BytesMut::with_capacity(SRT_HDR_SIZE + payload.len());

        // Control packet header
        buf.put_u32(SRT_CONTROL_PACKET | (ctrl_type as u32) << 16);
        buf.put_u32(additional_info);
        buf.put_u32(self.get_timestamp());
        buf.put_u32(self.peer_socket_id);
        buf.put_slice(payload);

        self.socket.send_to(&buf, self.peer_addr).await?;
        Ok(())
    }

    /// Send a data packet
    async fn send_data(&self, data: Bytes) -> IngestResult<u32> {
        let seq = self.send_seq.fetch_add(1, Ordering::SeqCst);

        let mut buf = BytesMut::with_capacity(SRT_HDR_SIZE + data.len());

        // Data packet header
        buf.put_u32(seq & 0x7FFFFFFF); // Clear control bit
        buf.put_u32(0); // Message number / PP / O / KK / R
        buf.put_u32(self.get_timestamp());
        buf.put_u32(self.peer_socket_id);
        buf.put_slice(&data);

        // Store in send buffer for potential retransmission
        {
            let mut send_buf = self.send_buffer.lock().await;
            send_buf.insert(seq, (data, Instant::now()));
        }

        self.socket.send_to(&buf, self.peer_addr).await?;
        Ok(seq)
    }

    /// Send ACK
    async fn send_ack(&self, ack_seq: u32) -> IngestResult<()> {
        let cc = self.cc.lock().await;
        let ack = SrtAck {
            last_ack_seq: ack_seq,
            rtt: cc.rtt,
            rtt_var: cc.rtt_var,
            buffer_size: 8192,
            packet_recv_rate: 10000,
            link_capacity: 100000,
        };
        drop(cc);

        let mut payload = BytesMut::new();
        ack.encode(&mut payload);

        self.send_control(SRT_CTRL_ACK, ack_seq, &payload).await
    }

    /// Send NAK for lost packets
    async fn send_nak(&self, loss_list: Vec<(u32, u32)>) -> IngestResult<()> {
        if loss_list.is_empty() {
            return Ok(());
        }

        let nak = SrtNak { loss_list };
        let mut payload = BytesMut::new();
        nak.encode(&mut payload);

        self.send_control(SRT_CTRL_NAK, 0, &payload).await
    }

    /// Send keepalive
    async fn send_keepalive(&self) -> IngestResult<()> {
        self.send_control(SRT_CTRL_KEEPALIVE, 0, &[]).await
    }

    /// Send shutdown
    async fn send_shutdown(&self) -> IngestResult<()> {
        self.send_control(SRT_CTRL_SHUTDOWN, 0, &[]).await
    }

    /// Handle incoming packet
    async fn handle_packet(&self, data: &[u8]) -> IngestResult<Option<Bytes>> {
        if data.len() < SRT_HDR_SIZE {
            return Ok(None);
        }

        let header = SrtPacketHeader::parse(data)?;

        if header.is_control {
            let ctrl_type = ((header.sequence_or_type >> 16) & 0x7FFF) as u16;
            let payload = &data[SRT_HDR_SIZE..];

            match ctrl_type {
                SRT_CTRL_ACK => {
                    let ack = SrtAck::parse(payload)?;
                    let mut cc = self.cc.lock().await;
                    cc.update_rtt(ack.rtt);
                    cc.on_ack();

                    // Remove acknowledged packets from send buffer
                    let mut send_buf = self.send_buffer.lock().await;
                    send_buf.retain(|&seq, _| seq > ack.last_ack_seq);

                    // Send ACKACK
                    drop(cc);
                    drop(send_buf);
                    self.send_control(SRT_CTRL_ACKACK, ack.last_ack_seq, &[])
                        .await?;
                }
                SRT_CTRL_NAK => {
                    let nak = SrtNak::parse(payload)?;
                    let mut cc = self.cc.lock().await;
                    cc.on_loss();
                    drop(cc);

                    // Retransmit lost packets
                    let send_buf = self.send_buffer.lock().await;
                    for (start, end) in nak.loss_list {
                        for seq in start..=end {
                            if let Some((data, _)) = send_buf.get(&seq) {
                                let _ = self.send_data(data.clone()).await;
                            }
                        }
                    }
                }
                SRT_CTRL_KEEPALIVE => {
                    // Respond to keepalive
                    self.send_keepalive().await?;
                }
                SRT_CTRL_SHUTDOWN => {
                    self.shutdown.store(true, Ordering::SeqCst);
                    return Err(IngestError::ConnectionClosed);
                }
                SRT_CTRL_ACKACK => {
                    // ACKACK received, update RTT
                }
                _ => {
                    tracing::debug!("Unhandled control packet type: {}", ctrl_type);
                }
            }

            Ok(None)
        } else {
            // Data packet
            let seq = header.sequence_or_type;
            let payload = Bytes::copy_from_slice(&data[SRT_HDR_SIZE..]);

            let mut recv_buf = self.receive_buffer.lock().await;
            recv_buf.insert(seq, payload);

            // Get next in-order packet
            let result = recv_buf.get_next();

            // Check for losses and send NAK
            let loss_list = recv_buf.get_loss_list();
            if !loss_list.is_empty() {
                drop(recv_buf);
                self.send_nak(loss_list).await?;
            }

            Ok(result)
        }
    }

    /// Start background tasks for connection maintenance
    fn start_background_tasks(self: &Arc<Self>) {
        let conn = Arc::clone(self);

        // Keepalive task
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(1));
            while !conn.shutdown.load(Ordering::SeqCst) {
                interval.tick().await;
                if conn.send_keepalive().await.is_err() {
                    break;
                }
            }
        });
    }

    /// Parse MPEG-TS data to extract media
    fn parse_ts_data(&self, data: &Bytes) -> Option<MediaData> {
        // Simplified TS parsing - in production would use a proper demuxer
        if data.len() < 188 || data[0] != 0x47 {
            return None;
        }

        // Extract PID
        let pid = (((data[1] & 0x1F) as u16) << 8) | (data[2] as u16);

        // Common PIDs (simplified)
        let is_video = pid == 256 || pid == 4096;
        let is_audio = pid == 257 || pid == 4097;

        let timestamp = self.start_time.elapsed().as_millis() as u32;

        if is_video {
            Some(MediaData::Video {
                data: data.clone(),
                timestamp,
                codec: VideoCodec::H264,
                is_keyframe: false, // Would need to parse NAL units
                composition_time: 0,
            })
        } else if is_audio {
            Some(MediaData::Audio {
                data: data.clone(),
                timestamp,
                codec: AudioCodec::Aac,
                sample_rate: 48000,
                channels: 2,
            })
        } else {
            None
        }
    }
}

#[async_trait]
impl IngestSource for SrtConnection {
    fn stream_info(&self) -> &StreamInfo {
        &self.stream_info
    }

    fn state(&self) -> ConnectionState {
        if self.shutdown.load(Ordering::SeqCst) {
            ConnectionState::Disconnected
        } else {
            self.state
        }
    }

    async fn read_media(&mut self) -> IngestResult<Option<MediaData>> {
        let mut buf = [0u8; 2048];

        loop {
            if self.shutdown.load(Ordering::SeqCst) {
                return Ok(None);
            }

            let timeout = tokio::time::timeout(
                self.config.read_timeout,
                self.socket.recv_from(&mut buf),
            );

            match timeout.await {
                Ok(Ok((n, addr))) => {
                    if addr != self.peer_addr {
                        continue; // Ignore packets from other sources
                    }

                    if let Some(data) = self.handle_packet(&buf[..n]).await? {
                        if let Some(media) = self.parse_ts_data(&data) {
                            return Ok(Some(media));
                        }
                    }
                }
                Ok(Err(e)) => return Err(e.into()),
                Err(_) => {
                    // Timeout - send keepalive
                    self.send_keepalive().await?;
                }
            }
        }
    }

    async fn write_media(&mut self, data: MediaData) -> IngestResult<()> {
        let payload = match data {
            MediaData::Video { data, .. } | MediaData::Audio { data, .. } => data,
            MediaData::Metadata { .. } => return Ok(()), // SRT doesn't have metadata messages
        };

        // Split into SRT-sized packets if needed
        for chunk in payload.chunks(SRT_MAX_PAYLOAD_SIZE) {
            self.send_data(Bytes::copy_from_slice(chunk)).await?;
        }

        Ok(())
    }

    async fn close(&mut self) -> IngestResult<()> {
        self.shutdown.store(true, Ordering::SeqCst);
        self.send_shutdown().await?;
        Ok(())
    }
}

/// SRT Server
pub struct SrtServer {
    socket: Arc<UdpSocket>,
    config: IngestConfig,
    encryption: SrtEncryption,
    local_addr: SocketAddr,
    next_socket_id: AtomicU32,
    pending_connections: Arc<RwLock<HashMap<SocketAddr, PendingConnection>>>,
}

#[derive(Clone)]
struct PendingConnection {
    socket_id: u32,
    syn_cookie: u32,
    initial_seq: u32,
    state: HandshakeState,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum HandshakeState {
    Induction,
    Conclusion,
    Done,
}

impl SrtServer {
    /// Bind to an address and create a new SRT server
    pub async fn bind(addr: SocketAddr) -> IngestResult<Self> {
        Self::bind_with_config(addr, IngestConfig::default()).await
    }

    /// Bind with custom configuration
    pub async fn bind_with_config(addr: SocketAddr, config: IngestConfig) -> IngestResult<Self> {
        let socket = UdpSocket::bind(addr).await?;
        let local_addr = socket.local_addr()?;

        tracing::info!("SRT server listening on {}", local_addr);

        Ok(Self {
            socket: Arc::new(socket),
            config,
            encryption: SrtEncryption::default(),
            local_addr,
            next_socket_id: AtomicU32::new(1),
            pending_connections: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Set encryption configuration
    pub fn set_encryption(&mut self, encryption: SrtEncryption) {
        self.encryption = encryption;
    }

    /// Generate SYN cookie for connection
    fn generate_syn_cookie(addr: &SocketAddr) -> u32 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        addr.hash(&mut hasher);
        Instant::now().elapsed().as_nanos().hash(&mut hasher);
        hasher.finish() as u32
    }

    /// Handle induction phase
    async fn handle_induction(
        &self,
        addr: SocketAddr,
        _hs: &SrtHandshake,
    ) -> IngestResult<BytesMut> {
        let socket_id = self.next_socket_id.fetch_add(1, Ordering::SeqCst);
        let syn_cookie = Self::generate_syn_cookie(&addr);
        let initial_seq = rand::random::<u32>() & 0x7FFFFFFF;

        // Store pending connection
        {
            let mut pending = self.pending_connections.write().await;
            pending.insert(
                addr,
                PendingConnection {
                    socket_id,
                    syn_cookie,
                    initial_seq,
                    state: HandshakeState::Induction,
                },
            );
        }

        // Build induction response
        let mut response = SrtHandshake {
            version: 5, // UDT5/SRT
            socket_type: SRT_SOCKET_DGRAM,
            initial_sequence: initial_seq,
            max_segment_size: 1500,
            max_flow_window: 8192,
            handshake_type: SRT_HS_TYPE_INDUCTION,
            socket_id,
            syn_cookie,
            peer_ip: [0; 16],
            extensions: Vec::new(),
        };

        // Copy peer IP (IPv4 mapped to IPv6)
        match addr.ip() {
            std::net::IpAddr::V4(ip) => {
                response.peer_ip[12..16].copy_from_slice(&ip.octets());
            }
            std::net::IpAddr::V6(ip) => {
                response.peer_ip.copy_from_slice(&ip.octets());
            }
        }

        // Add HSRSP extension (response to HSREQ)
        response.extensions.push(SrtHandshake::create_hsreq_extension(
            SRT_VERSION,
            0x000000BF, // Standard flags
            120,        // 120ms TSBPD delay
            120,
        ));

        Ok(response.encode())
    }

    /// Handle conclusion phase
    async fn handle_conclusion(
        &self,
        addr: SocketAddr,
        hs: &SrtHandshake,
    ) -> IngestResult<(BytesMut, u32, u32, u32)> {
        let pending = {
            let pending_guard = self.pending_connections.read().await;
            pending_guard.get(&addr).cloned()
        };

        let pending = pending
            .ok_or_else(|| IngestError::HandshakeFailed("No pending connection".into()))?;

        // Verify SYN cookie
        if hs.syn_cookie != pending.syn_cookie {
            return Err(IngestError::HandshakeFailed("Invalid SYN cookie".into()));
        }

        // Build conclusion response
        let response = SrtHandshake {
            version: 5,
            socket_type: SRT_SOCKET_DGRAM,
            initial_sequence: pending.initial_seq,
            max_segment_size: 1500,
            max_flow_window: 8192,
            handshake_type: SRT_HS_TYPE_DONE,
            socket_id: pending.socket_id,
            syn_cookie: 0,
            peer_ip: hs.peer_ip,
            extensions: Vec::new(),
        };

        // Mark as done
        {
            let mut pending_guard = self.pending_connections.write().await;
            if let Some(p) = pending_guard.get_mut(&addr) {
                p.state = HandshakeState::Done;
            }
        }

        Ok((
            response.encode(),
            pending.socket_id,
            hs.socket_id,
            hs.initial_sequence,
        ))
    }

    /// Send handshake response
    async fn send_handshake_response(
        &self,
        addr: SocketAddr,
        response: BytesMut,
        dest_socket_id: u32,
    ) -> IngestResult<()> {
        let mut buf = BytesMut::with_capacity(SRT_HDR_SIZE + response.len());

        // Control packet header for handshake
        buf.put_u32(SRT_CONTROL_PACKET | ((SRT_CTRL_HANDSHAKE as u32) << 16));
        buf.put_u32(0); // Additional info
        buf.put_u32(0); // Timestamp
        buf.put_u32(dest_socket_id);
        buf.put_slice(&response);

        self.socket.send_to(&buf, addr).await?;
        Ok(())
    }
}

#[async_trait]
impl IngestServer for SrtServer {
    type Connection = SrtConnection;

    async fn accept(&mut self) -> IngestResult<Self::Connection> {
        let mut buf = [0u8; 2048];

        loop {
            let (n, addr) = self.socket.recv_from(&mut buf).await?;

            if n < SRT_HDR_SIZE {
                continue;
            }

            let header = SrtPacketHeader::parse(&buf[..n])?;

            if !header.is_control {
                continue; // Skip data packets during handshake
            }

            let ctrl_type = ((header.sequence_or_type >> 16) & 0x7FFF) as u16;

            if ctrl_type != SRT_CTRL_HANDSHAKE {
                continue; // Only handle handshake during accept
            }

            let hs = SrtHandshake::parse(&buf[SRT_HDR_SIZE..n])?;

            match hs.handshake_type {
                SRT_HS_TYPE_INDUCTION => {
                    tracing::debug!("SRT induction from {}", addr);
                    let response = self.handle_induction(addr, &hs).await?;
                    self.send_handshake_response(addr, response, hs.socket_id)
                        .await?;
                }
                SRT_HS_TYPE_CONCLUSION => {
                    tracing::debug!("SRT conclusion from {}", addr);
                    let (response, local_socket_id, peer_socket_id, initial_seq) =
                        self.handle_conclusion(addr, &hs).await?;
                    self.send_handshake_response(addr, response, peer_socket_id)
                        .await?;

                    // Remove from pending
                    {
                        let mut pending = self.pending_connections.write().await;
                        pending.remove(&addr);
                    }

                    tracing::info!("SRT connection established with {}", addr);

                    // Create connection
                    let conn = SrtConnection::new(
                        Arc::clone(&self.socket),
                        addr,
                        local_socket_id,
                        peer_socket_id,
                        initial_seq,
                        self.config.clone(),
                    );

                    return Ok(conn);
                }
                _ => {
                    tracing::debug!("Unknown handshake type: {:#x}", hs.handshake_type);
                }
            }
        }
    }

    fn local_addr(&self) -> IngestResult<SocketAddr> {
        Ok(self.local_addr)
    }

    async fn shutdown(&mut self) -> IngestResult<()> {
        // UDP socket doesn't need explicit shutdown
        Ok(())
    }
}

/// SRT Client
pub struct SrtClient;

impl SrtClient {
    /// Connect to an SRT server
    pub async fn connect(addr: SocketAddr) -> IngestResult<SrtConnection> {
        Self::connect_with_config(addr, IngestConfig::default()).await
    }

    /// Connect with custom configuration
    pub async fn connect_with_config(
        addr: SocketAddr,
        config: IngestConfig,
    ) -> IngestResult<SrtConnection> {
        // Bind to any local port
        let socket = UdpSocket::bind("0.0.0.0:0").await?;
        let socket = Arc::new(socket);

        let local_socket_id = rand::random::<u32>();
        let initial_seq = rand::random::<u32>() & 0x7FFFFFFF;

        // Send induction handshake
        let hs = SrtHandshake {
            version: 5,
            socket_type: SRT_SOCKET_DGRAM,
            initial_sequence: initial_seq,
            max_segment_size: 1500,
            max_flow_window: 8192,
            handshake_type: SRT_HS_TYPE_INDUCTION,
            socket_id: local_socket_id,
            syn_cookie: 0,
            peer_ip: [0; 16],
            extensions: vec![SrtHandshake::create_hsreq_extension(
                SRT_VERSION,
                0x000000BF,
                120,
                120,
            )],
        };

        let mut buf = BytesMut::with_capacity(256);
        buf.put_u32(SRT_CONTROL_PACKET | ((SRT_CTRL_HANDSHAKE as u32) << 16));
        buf.put_u32(0);
        buf.put_u32(0);
        buf.put_u32(0); // Server socket ID unknown
        buf.put_slice(&hs.encode());

        socket.send_to(&buf, addr).await?;

        // Wait for induction response
        let mut recv_buf = [0u8; 2048];
        let timeout = tokio::time::timeout(config.connect_timeout, socket.recv_from(&mut recv_buf));

        let (n, _) = timeout
            .await
            .map_err(|_| IngestError::Timeout("Handshake timeout".into()))??;

        if n < SRT_HDR_SIZE {
            return Err(IngestError::HandshakeFailed(
                "Response too short".into(),
            ));
        }

        let response_hs = SrtHandshake::parse(&recv_buf[SRT_HDR_SIZE..n])?;
        let syn_cookie = response_hs.syn_cookie;
        let server_socket_id = response_hs.socket_id;
        let server_initial_seq = response_hs.initial_sequence;

        // Send conclusion handshake
        let conclusion_hs = SrtHandshake {
            version: 5,
            socket_type: SRT_SOCKET_DGRAM,
            initial_sequence: initial_seq,
            max_segment_size: 1500,
            max_flow_window: 8192,
            handshake_type: SRT_HS_TYPE_CONCLUSION,
            socket_id: local_socket_id,
            syn_cookie,
            peer_ip: [0; 16],
            extensions: vec![SrtHandshake::create_hsreq_extension(
                SRT_VERSION,
                0x000000BF,
                120,
                120,
            )],
        };

        let mut buf = BytesMut::with_capacity(256);
        buf.put_u32(SRT_CONTROL_PACKET | ((SRT_CTRL_HANDSHAKE as u32) << 16));
        buf.put_u32(0);
        buf.put_u32(0);
        buf.put_u32(server_socket_id);
        buf.put_slice(&conclusion_hs.encode());

        socket.send_to(&buf, addr).await?;

        // Wait for final response
        let timeout = tokio::time::timeout(config.connect_timeout, socket.recv_from(&mut recv_buf));

        let (n, _) = timeout
            .await
            .map_err(|_| IngestError::Timeout("Handshake timeout".into()))??;

        if n < SRT_HDR_SIZE {
            return Err(IngestError::HandshakeFailed(
                "Final response too short".into(),
            ));
        }

        let final_hs = SrtHandshake::parse(&recv_buf[SRT_HDR_SIZE..n])?;

        if final_hs.handshake_type != SRT_HS_TYPE_DONE {
            return Err(IngestError::HandshakeFailed(format!(
                "Unexpected handshake type: {:#x}",
                final_hs.handshake_type
            )));
        }

        tracing::info!("SRT client connected to {}", addr);

        let conn = SrtConnection::new(
            socket,
            addr,
            local_socket_id,
            server_socket_id,
            server_initial_seq,
            config,
        );

        Ok(conn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handshake_encode_decode() {
        let hs = SrtHandshake {
            version: 5,
            socket_type: SRT_SOCKET_DGRAM,
            initial_sequence: 12345,
            max_segment_size: 1500,
            max_flow_window: 8192,
            handshake_type: SRT_HS_TYPE_INDUCTION,
            socket_id: 1,
            syn_cookie: 0,
            peer_ip: [0; 16],
            extensions: Vec::new(),
        };

        let encoded = hs.encode();
        let decoded = SrtHandshake::parse(&encoded).unwrap();

        assert_eq!(decoded.version, hs.version);
        assert_eq!(decoded.socket_type, hs.socket_type);
        assert_eq!(decoded.initial_sequence, hs.initial_sequence);
        assert_eq!(decoded.handshake_type, hs.handshake_type);
    }

    #[test]
    fn test_packet_header() {
        let mut buf = BytesMut::new();
        let header = SrtPacketHeader {
            is_control: false,
            sequence_or_type: 12345,
            timestamp: 1000,
            dest_socket_id: 1,
        };
        header.encode(&mut buf);

        let parsed = SrtPacketHeader::parse(&buf).unwrap();
        assert!(!parsed.is_control);
        assert_eq!(parsed.sequence_or_type, 12345);
        assert_eq!(parsed.timestamp, 1000);
    }

    #[test]
    fn test_control_packet_header() {
        let mut buf = BytesMut::new();
        let header = SrtPacketHeader {
            is_control: true,
            sequence_or_type: (SRT_CTRL_ACK as u32) << 16,
            timestamp: 2000,
            dest_socket_id: 2,
        };
        header.encode(&mut buf);

        let parsed = SrtPacketHeader::parse(&buf).unwrap();
        assert!(parsed.is_control);
    }

    #[test]
    fn test_ack_encode_decode() {
        let ack = SrtAck {
            last_ack_seq: 100,
            rtt: 50000,
            rtt_var: 10000,
            buffer_size: 8192,
            packet_recv_rate: 10000,
            link_capacity: 100000,
        };

        let mut buf = BytesMut::new();
        ack.encode(&mut buf);

        let parsed = SrtAck::parse(&buf).unwrap();
        assert_eq!(parsed.last_ack_seq, 100);
        assert_eq!(parsed.rtt, 50000);
    }

    #[test]
    fn test_nak_encode_decode() {
        let nak = SrtNak {
            loss_list: vec![(10, 10), (20, 25)],
        };

        let mut buf = BytesMut::new();
        nak.encode(&mut buf);

        let parsed = SrtNak::parse(&buf).unwrap();
        assert_eq!(parsed.loss_list.len(), 2);
    }

    #[test]
    fn test_receive_buffer() {
        let mut buf = ReceiveBuffer::new(0);

        buf.insert(0, Bytes::from_static(b"first"));
        buf.insert(2, Bytes::from_static(b"third")); // Gap at 1
        buf.insert(1, Bytes::from_static(b"second"));

        assert_eq!(buf.get_next(), Some(Bytes::from_static(b"first")));
        assert_eq!(buf.get_next(), Some(Bytes::from_static(b"second")));
        assert_eq!(buf.get_next(), Some(Bytes::from_static(b"third")));
        assert_eq!(buf.get_next(), None);
    }

    #[test]
    fn test_encryption_default() {
        let enc = SrtEncryption::default();
        assert!(!enc.enabled);
        assert!(enc.passphrase.is_none());
    }

    #[test]
    fn test_congestion_control() {
        let mut cc = CongestionControl::default();

        cc.update_rtt(80000);
        assert!(cc.rtt < 100000); // Should decrease toward new value

        cc.on_ack();
        assert!(cc.cwnd > 16);

        cc.on_loss();
        assert!(cc.cwnd < 17); // Should be halved
    }
}
