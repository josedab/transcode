//! WebRTC protocol support
//!
//! This module provides WebRTC peer connection management with:
//! - ICE (Interactive Connectivity Establishment) candidate gathering and processing
//! - DTLS (Datagram Transport Layer Security) fingerprint handling
//! - SDP (Session Description Protocol) generation and parsing
//! - Connection state management
//! - WebRTC signaling server for WebSocket-based signaling

use crate::{LiveError, Result, StreamMetadata, StreamPacket};
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, RwLock};

/// ICE candidate types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IceCandidateType {
    /// Host candidate (local IP)
    Host,
    /// Server reflexive candidate (STUN)
    Srflx,
    /// Peer reflexive candidate
    Prflx,
    /// Relay candidate (TURN)
    Relay,
}

impl IceCandidateType {
    /// Get priority modifier for this candidate type
    fn type_preference(&self) -> u32 {
        match self {
            IceCandidateType::Host => 126,
            IceCandidateType::Prflx => 110,
            IceCandidateType::Srflx => 100,
            IceCandidateType::Relay => 0,
        }
    }
}

/// ICE candidate
#[derive(Debug, Clone)]
pub struct IceCandidate {
    /// Candidate string
    pub candidate: String,
    /// SDP mid
    pub sdp_mid: Option<String>,
    /// SDP mline index
    pub sdp_mline_index: Option<u32>,
    /// Foundation
    pub foundation: String,
    /// Component ID (1 = RTP, 2 = RTCP)
    pub component: u8,
    /// Transport protocol
    pub transport: IceTransport,
    /// Priority
    pub priority: u32,
    /// Candidate address
    pub address: Option<SocketAddr>,
    /// Candidate type
    pub candidate_type: IceCandidateType,
    /// Related address (for srflx/relay)
    pub related_address: Option<SocketAddr>,
}

impl IceCandidate {
    /// Parse an ICE candidate from SDP attribute format
    pub fn parse(candidate_str: &str) -> Result<Self> {
        let parts: Vec<&str> = candidate_str.split_whitespace().collect();
        if parts.len() < 8 {
            return Err(LiveError::Protocol("Invalid ICE candidate format".into()));
        }

        let foundation = parts[0].trim_start_matches("candidate:").to_string();
        let component = parts[1].parse().unwrap_or(1);
        let transport = match parts[2].to_uppercase().as_str() {
            "UDP" => IceTransport::Udp,
            "TCP" => IceTransport::Tcp,
            _ => IceTransport::Udp,
        };
        let priority = parts[3].parse().unwrap_or(0);

        let ip: IpAddr = parts[4].parse()
            .map_err(|_| LiveError::Protocol("Invalid candidate IP".into()))?;
        let port: u16 = parts[5].parse()
            .map_err(|_| LiveError::Protocol("Invalid candidate port".into()))?;

        let address = Some(SocketAddr::new(ip, port));

        let candidate_type = if parts.len() > 7 && parts[6] == "typ" {
            match parts[7] {
                "host" => IceCandidateType::Host,
                "srflx" => IceCandidateType::Srflx,
                "prflx" => IceCandidateType::Prflx,
                "relay" => IceCandidateType::Relay,
                _ => IceCandidateType::Host,
            }
        } else {
            IceCandidateType::Host
        };

        // Parse related address if present
        let related_address = if parts.len() > 11 && parts[8] == "raddr" && parts[10] == "rport" {
            if let (Ok(rip), Ok(rport)) = (parts[9].parse::<IpAddr>(), parts[11].parse::<u16>()) {
                Some(SocketAddr::new(rip, rport))
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            candidate: candidate_str.to_string(),
            sdp_mid: None,
            sdp_mline_index: None,
            foundation,
            component,
            transport,
            priority,
            address,
            candidate_type,
            related_address,
        })
    }

    /// Generate candidate string for SDP
    pub fn to_sdp_attribute(&self) -> String {
        let addr = self.address.map(|a| format!("{} {}", a.ip(), a.port()))
            .unwrap_or_else(|| "0.0.0.0 9".to_string());

        let typ = match self.candidate_type {
            IceCandidateType::Host => "host",
            IceCandidateType::Srflx => "srflx",
            IceCandidateType::Prflx => "prflx",
            IceCandidateType::Relay => "relay",
        };

        let transport = match self.transport {
            IceTransport::Udp => "UDP",
            IceTransport::Tcp => "TCP",
        };

        let mut result = format!(
            "candidate:{} {} {} {} {} typ {}",
            self.foundation, self.component, transport, self.priority, addr, typ
        );

        if let Some(raddr) = self.related_address {
            result.push_str(&format!(" raddr {} rport {}", raddr.ip(), raddr.port()));
        }

        result
    }

    /// Calculate priority based on RFC 8445
    pub fn calculate_priority(type_pref: u32, local_pref: u32, component: u8) -> u32 {
        (type_pref << 24) | (local_pref << 8) | (256 - component as u32)
    }
}

/// ICE transport protocol
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IceTransport {
    /// UDP transport
    Udp,
    /// TCP transport
    Tcp,
}

/// ICE gathering state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IceGatheringState {
    /// Not started gathering
    New,
    /// Gathering in progress
    Gathering,
    /// Gathering complete
    Complete,
}

/// ICE connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IceConnectionState {
    /// Checking connectivity
    New,
    /// ICE checks in progress
    Checking,
    /// At least one working candidate pair
    Connected,
    /// All checks completed, connectivity established
    Completed,
    /// All checks failed
    Failed,
    /// Connection lost
    Disconnected,
    /// Connection closed
    Closed,
}

/// DTLS role
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtlsRole {
    /// Client initiates handshake
    Client,
    /// Server waits for handshake
    Server,
    /// Role determined by offer/answer
    Auto,
}

/// DTLS fingerprint
#[derive(Debug, Clone)]
pub struct DtlsFingerprint {
    /// Hash algorithm (e.g., "sha-256")
    pub algorithm: String,
    /// Fingerprint value (hex-encoded)
    pub value: String,
}

impl DtlsFingerprint {
    /// Create a new fingerprint
    pub fn new(algorithm: &str, value: &str) -> Self {
        Self {
            algorithm: algorithm.to_string(),
            value: value.to_string(),
        }
    }

    /// Generate a dummy fingerprint for testing
    pub fn generate_dummy() -> Self {
        // In production, this would compute SHA-256 of the DTLS certificate
        Self {
            algorithm: "sha-256".to_string(),
            value: "00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:\
                   00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF".to_string(),
        }
    }

    /// Format for SDP
    pub fn to_sdp_attribute(&self) -> String {
        format!("fingerprint:{} {}", self.algorithm, self.value)
    }
}

/// DTLS parameters
#[derive(Debug, Clone)]
pub struct DtlsParameters {
    /// DTLS role
    pub role: DtlsRole,
    /// Fingerprints
    pub fingerprints: Vec<DtlsFingerprint>,
}

impl Default for DtlsParameters {
    fn default() -> Self {
        Self {
            role: DtlsRole::Auto,
            fingerprints: vec![DtlsFingerprint::generate_dummy()],
        }
    }
}

/// ICE candidate pair
#[derive(Debug, Clone)]
pub struct CandidatePair {
    /// Local candidate
    pub local: IceCandidate,
    /// Remote candidate
    pub remote: IceCandidate,
    /// Pair priority
    pub priority: u64,
    /// Pair state
    pub state: CandidatePairState,
    /// Nominated flag
    pub nominated: bool,
    /// Last check time
    pub last_check: Option<Instant>,
}

/// Candidate pair state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidatePairState {
    /// Waiting to be checked
    Waiting,
    /// Check in progress
    InProgress,
    /// Check succeeded
    Succeeded,
    /// Check failed
    Failed,
    /// Frozen (waiting for another pair)
    Frozen,
}

impl CandidatePair {
    /// Calculate pair priority per RFC 8445
    pub fn calculate_priority(controlling: bool, local_prio: u32, remote_prio: u32) -> u64 {
        let (g, d) = if controlling {
            (local_prio as u64, remote_prio as u64)
        } else {
            (remote_prio as u64, local_prio as u64)
        };
        (1 << 32) * g.min(d) + 2 * g.max(d) + if g > d { 1 } else { 0 }
    }
}

/// Session description
#[derive(Debug, Clone)]
pub struct SessionDescription {
    /// SDP type
    pub sdp_type: SdpType,
    /// SDP string
    pub sdp: String,
}

/// SDP type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdpType {
    /// Offer
    Offer,
    /// Answer
    Answer,
    /// Provisional answer
    Pranswer,
    /// Rollback
    Rollback,
}

/// WebRTC peer connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerConnectionState {
    /// New connection
    New,
    /// Connecting
    Connecting,
    /// Connected
    Connected,
    /// Disconnected
    Disconnected,
    /// Failed
    Failed,
    /// Closed
    Closed,
}

/// WebRTC configuration
#[derive(Debug, Clone)]
pub struct WebRtcConfig {
    /// STUN servers
    pub stun_servers: Vec<String>,
    /// TURN servers
    pub turn_servers: Vec<TurnServer>,
    /// ICE transport policy
    pub ice_transport_policy: IceTransportPolicy,
    /// Bundle policy
    pub bundle_policy: BundlePolicy,
    /// RTCP mux policy
    pub rtcp_mux_policy: RtcpMuxPolicy,
    /// ICE candidate pool size
    pub ice_candidate_pool_size: u8,
}

/// ICE transport policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IceTransportPolicy {
    /// Use all candidates
    All,
    /// Use only relay candidates
    Relay,
}

/// Bundle policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundlePolicy {
    /// Bundle all media
    MaxBundle,
    /// Balance bundling
    Balanced,
    /// Separate transports
    MaxCompat,
}

/// RTCP mux policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtcpMuxPolicy {
    /// Require RTCP mux
    Require,
    /// Negotiate RTCP mux
    Negotiate,
}

impl Default for WebRtcConfig {
    fn default() -> Self {
        Self {
            stun_servers: vec!["stun:stun.l.google.com:19302".into()],
            turn_servers: Vec::new(),
            ice_transport_policy: IceTransportPolicy::All,
            bundle_policy: BundlePolicy::MaxBundle,
            rtcp_mux_policy: RtcpMuxPolicy::Require,
            ice_candidate_pool_size: 0,
        }
    }
}

/// TURN server configuration
#[derive(Debug, Clone)]
pub struct TurnServer {
    /// Server URL
    pub url: String,
    /// Username
    pub username: String,
    /// Credential
    pub credential: String,
}

/// ICE agent for managing ICE connectivity
#[allow(dead_code)]
pub struct IceAgent {
    /// Local candidates
    local_candidates: Vec<IceCandidate>,
    /// Remote candidates
    remote_candidates: Vec<IceCandidate>,
    /// Candidate pairs
    candidate_pairs: Vec<CandidatePair>,
    /// Selected pair
    selected_pair: Option<usize>,
    /// Gathering state
    gathering_state: IceGatheringState,
    /// Connection state
    connection_state: IceConnectionState,
    /// Controlling role
    controlling: bool,
    /// Tie breaker for role conflicts
    tie_breaker: u64,
    /// Local ICE credentials
    local_ufrag: String,
    local_pwd: String,
    /// Remote ICE credentials
    remote_ufrag: Option<String>,
    remote_pwd: Option<String>,
    /// Check list frozen pairs per foundation
    foundations: HashMap<String, bool>,
}

impl IceAgent {
    /// Create a new ICE agent
    pub fn new(controlling: bool) -> Self {
        Self {
            local_candidates: Vec::new(),
            remote_candidates: Vec::new(),
            candidate_pairs: Vec::new(),
            selected_pair: None,
            gathering_state: IceGatheringState::New,
            connection_state: IceConnectionState::New,
            controlling,
            tie_breaker: rand_u64(),
            local_ufrag: generate_ice_ufrag(),
            local_pwd: generate_ice_pwd(),
            remote_ufrag: None,
            remote_pwd: None,
            foundations: HashMap::new(),
        }
    }

    /// Get local ICE ufrag
    pub fn local_ufrag(&self) -> &str {
        &self.local_ufrag
    }

    /// Get local ICE pwd
    pub fn local_pwd(&self) -> &str {
        &self.local_pwd
    }

    /// Set remote ICE credentials
    pub fn set_remote_credentials(&mut self, ufrag: &str, pwd: &str) {
        self.remote_ufrag = Some(ufrag.to_string());
        self.remote_pwd = Some(pwd.to_string());
    }

    /// Start gathering candidates
    pub fn gather_candidates(&mut self, local_addresses: &[IpAddr]) {
        self.gathering_state = IceGatheringState::Gathering;
        self.local_candidates.clear();

        // Generate host candidates for each local address
        for (idx, addr) in local_addresses.iter().enumerate() {
            let foundation = format!("host{}", idx);
            let priority = IceCandidate::calculate_priority(
                IceCandidateType::Host.type_preference(),
                65535 - idx as u32,
                1,
            );

            // RTP candidate
            let candidate = IceCandidate {
                candidate: String::new(),
                sdp_mid: Some("0".to_string()),
                sdp_mline_index: Some(0),
                foundation: foundation.clone(),
                component: 1,
                transport: IceTransport::Udp,
                priority,
                address: Some(SocketAddr::new(*addr, 9)), // Port assigned later
                candidate_type: IceCandidateType::Host,
                related_address: None,
            };

            self.local_candidates.push(candidate);
        }

        self.gathering_state = IceGatheringState::Complete;
    }

    /// Add a remote candidate
    pub fn add_remote_candidate(&mut self, candidate: IceCandidate) {
        self.remote_candidates.push(candidate.clone());

        // Form pairs with all local candidates
        for local in &self.local_candidates {
            if local.component == candidate.component {
                let priority = CandidatePair::calculate_priority(
                    self.controlling,
                    local.priority,
                    candidate.priority,
                );

                let pair = CandidatePair {
                    local: local.clone(),
                    remote: candidate.clone(),
                    priority,
                    state: CandidatePairState::Frozen,
                    nominated: false,
                    last_check: None,
                };

                self.candidate_pairs.push(pair);
            }
        }

        // Sort pairs by priority
        self.candidate_pairs.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Unfreeze first pair per foundation
        self.unfreeze_pairs();
    }

    /// Unfreeze pairs based on foundation
    fn unfreeze_pairs(&mut self) {
        self.foundations.clear();

        for pair in &mut self.candidate_pairs {
            let foundation = format!("{}:{}", pair.local.foundation, pair.remote.foundation);

            if let std::collections::hash_map::Entry::Vacant(e) = self.foundations.entry(foundation)
            {
                if pair.state == CandidatePairState::Frozen {
                    pair.state = CandidatePairState::Waiting;
                }
                e.insert(true);
            }
        }
    }

    /// Process connectivity check
    pub fn check_connectivity(&mut self) -> Option<&CandidatePair> {
        self.connection_state = IceConnectionState::Checking;

        // Find next pair to check
        for (idx, pair) in self.candidate_pairs.iter_mut().enumerate() {
            if pair.state == CandidatePairState::Waiting {
                pair.state = CandidatePairState::InProgress;
                pair.last_check = Some(Instant::now());

                // Simulate successful check for demonstration
                pair.state = CandidatePairState::Succeeded;

                if self.selected_pair.is_none() {
                    self.selected_pair = Some(idx);
                    self.connection_state = IceConnectionState::Connected;
                }

                return self.selected_pair.map(|i| &self.candidate_pairs[i]);
            }
        }

        // All pairs checked
        if self.selected_pair.is_some() {
            self.connection_state = IceConnectionState::Completed;
            self.selected_pair.map(|i| &self.candidate_pairs[i])
        } else {
            self.connection_state = IceConnectionState::Failed;
            None
        }
    }

    /// Get selected candidate pair
    pub fn selected_pair(&self) -> Option<&CandidatePair> {
        self.selected_pair.map(|i| &self.candidate_pairs[i])
    }

    /// Get gathering state
    pub fn gathering_state(&self) -> IceGatheringState {
        self.gathering_state
    }

    /// Get connection state
    pub fn connection_state(&self) -> IceConnectionState {
        self.connection_state
    }
}

/// WebRTC peer connection
#[allow(dead_code)]
pub struct WebRtcPeer {
    config: WebRtcConfig,
    state: PeerConnectionState,
    local_description: Option<SessionDescription>,
    remote_description: Option<SessionDescription>,
    ice_agent: IceAgent,
    dtls_params: DtlsParameters,
    /// Media stream tracks
    tracks: Vec<MediaTrack>,
}

/// Media track type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    /// Audio track
    Audio,
    /// Video track
    Video,
}

/// Media track
#[derive(Debug, Clone)]
pub struct MediaTrack {
    /// Track ID
    pub id: String,
    /// Media kind
    pub kind: MediaKind,
    /// Track label
    pub label: String,
    /// Enabled state
    pub enabled: bool,
}

impl WebRtcPeer {
    /// Create a new WebRTC peer
    pub fn new(config: WebRtcConfig) -> Self {
        Self {
            config,
            state: PeerConnectionState::New,
            local_description: None,
            remote_description: None,
            ice_agent: IceAgent::new(true), // Start as controlling
            dtls_params: DtlsParameters::default(),
            tracks: Vec::new(),
        }
    }

    /// Add a media track
    pub fn add_track(&mut self, track: MediaTrack) {
        self.tracks.push(track);
    }

    /// Create an offer
    pub async fn create_offer(&mut self) -> Result<SessionDescription> {
        // Gather local candidates
        let local_addrs = get_local_addresses();
        self.ice_agent.gather_candidates(&local_addrs);

        // Generate SDP offer
        let sdp = self.generate_sdp(true);
        let desc = SessionDescription {
            sdp_type: SdpType::Offer,
            sdp,
        };
        self.local_description = Some(desc.clone());
        Ok(desc)
    }

    /// Create an answer
    pub async fn create_answer(&mut self) -> Result<SessionDescription> {
        if self.remote_description.is_none() {
            return Err(LiveError::Protocol("no remote description".into()));
        }

        // Gather local candidates
        let local_addrs = get_local_addresses();
        self.ice_agent.gather_candidates(&local_addrs);

        // Generate SDP answer
        self.ice_agent = IceAgent::new(false); // Answerer is controlled
        let sdp = self.generate_sdp(false);
        let desc = SessionDescription {
            sdp_type: SdpType::Answer,
            sdp,
        };
        self.local_description = Some(desc.clone());
        Ok(desc)
    }

    /// Set local description
    pub async fn set_local_description(&mut self, desc: SessionDescription) -> Result<()> {
        self.local_description = Some(desc);
        self.state = PeerConnectionState::Connecting;

        // Start ICE connectivity checks
        self.ice_agent.check_connectivity();

        Ok(())
    }

    /// Set remote description
    pub async fn set_remote_description(&mut self, desc: SessionDescription) -> Result<()> {
        // Parse remote ICE credentials from SDP
        if let Some((ufrag, pwd)) = parse_ice_credentials(&desc.sdp) {
            self.ice_agent.set_remote_credentials(&ufrag, &pwd);
        }

        // Parse remote candidates from SDP
        for candidate in parse_ice_candidates(&desc.sdp) {
            self.ice_agent.add_remote_candidate(candidate);
        }

        self.remote_description = Some(desc);
        self.state = PeerConnectionState::Connecting;

        // Check connectivity
        if self.ice_agent.check_connectivity().is_some() {
            self.state = PeerConnectionState::Connected;
        }

        Ok(())
    }

    /// Add ICE candidate
    pub fn add_ice_candidate(&mut self, candidate: IceCandidate) -> Result<()> {
        self.ice_agent.add_remote_candidate(candidate);

        // Re-check connectivity with new candidate
        if self.ice_agent.check_connectivity().is_some() {
            self.state = PeerConnectionState::Connected;
        }

        Ok(())
    }

    /// Get connection state
    pub fn state(&self) -> PeerConnectionState {
        self.state
    }

    /// Get ICE connection state
    pub fn ice_connection_state(&self) -> IceConnectionState {
        self.ice_agent.connection_state()
    }

    /// Get ICE gathering state
    pub fn ice_gathering_state(&self) -> IceGatheringState {
        self.ice_agent.gathering_state()
    }

    /// Get local candidates
    pub fn local_candidates(&self) -> &[IceCandidate] {
        &self.ice_agent.local_candidates
    }

    /// Get selected candidate pair
    pub fn selected_candidate_pair(&self) -> Option<&CandidatePair> {
        self.ice_agent.selected_pair()
    }

    /// Close the connection
    pub async fn close(&mut self) -> Result<()> {
        self.state = PeerConnectionState::Closed;
        Ok(())
    }

    fn generate_sdp(&self, is_offer: bool) -> String {
        let mut sdp = String::new();

        // Session-level attributes
        sdp.push_str("v=0\r\n");
        sdp.push_str(&format!("o=- {} 2 IN IP4 127.0.0.1\r\n", rand_u64()));
        sdp.push_str("s=-\r\n");
        sdp.push_str("t=0 0\r\n");

        // Bundle group
        let mids: Vec<&str> = self.tracks.iter()
            .enumerate()
            .map(|(i, _)| if i == 0 { "0" } else { "1" })
            .collect();
        if !mids.is_empty() {
            sdp.push_str(&format!("a=group:BUNDLE {}\r\n", mids.join(" ")));
        }

        // ICE options
        sdp.push_str("a=ice-options:trickle\r\n");

        // Media sections
        for (idx, track) in self.tracks.iter().enumerate() {
            let (media_type, payload_type, codec) = match track.kind {
                MediaKind::Video => ("video", 96, "H264/90000"),
                MediaKind::Audio => ("audio", 111, "opus/48000/2"),
            };

            sdp.push_str(&format!(
                "m={} 9 UDP/TLS/RTP/SAVPF {}\r\n",
                media_type, payload_type
            ));
            sdp.push_str("c=IN IP4 0.0.0.0\r\n");
            sdp.push_str(&format!("a=mid:{}\r\n", idx));

            // ICE credentials
            sdp.push_str(&format!("a=ice-ufrag:{}\r\n", self.ice_agent.local_ufrag()));
            sdp.push_str(&format!("a=ice-pwd:{}\r\n", self.ice_agent.local_pwd()));

            // DTLS fingerprint
            for fp in &self.dtls_params.fingerprints {
                sdp.push_str(&format!("a={}\r\n", fp.to_sdp_attribute()));
            }

            // DTLS setup
            let setup = if is_offer { "actpass" } else { "active" };
            sdp.push_str(&format!("a=setup:{}\r\n", setup));

            // RTP/RTCP
            sdp.push_str(&format!("a=rtpmap:{} {}\r\n", payload_type, codec));
            sdp.push_str("a=rtcp-mux\r\n");

            // Direction
            let direction = if is_offer { "sendrecv" } else { "recvonly" };
            sdp.push_str(&format!("a={}\r\n", direction));

            // ICE candidates
            for candidate in &self.ice_agent.local_candidates {
                sdp.push_str(&format!("a={}\r\n", candidate.to_sdp_attribute()));
            }
        }

        // Add default video track if none specified
        if self.tracks.is_empty() {
            sdp.push_str("m=video 9 UDP/TLS/RTP/SAVPF 96\r\n");
            sdp.push_str("c=IN IP4 0.0.0.0\r\n");
            sdp.push_str("a=mid:0\r\n");
            sdp.push_str(&format!("a=ice-ufrag:{}\r\n", self.ice_agent.local_ufrag()));
            sdp.push_str(&format!("a=ice-pwd:{}\r\n", self.ice_agent.local_pwd()));
            for fp in &self.dtls_params.fingerprints {
                sdp.push_str(&format!("a={}\r\n", fp.to_sdp_attribute()));
            }
            let setup = if is_offer { "actpass" } else { "active" };
            sdp.push_str(&format!("a=setup:{}\r\n", setup));
            sdp.push_str("a=rtpmap:96 H264/90000\r\n");
            sdp.push_str("a=rtcp-mux\r\n");
            let direction = if is_offer { "sendrecv" } else { "recvonly" };
            sdp.push_str(&format!("a={}\r\n", direction));
        }

        sdp
    }
}

impl Default for WebRtcPeer {
    fn default() -> Self {
        Self::new(WebRtcConfig::default())
    }
}

// Helper functions

/// Generate random ICE ufrag (4-256 characters)
fn generate_ice_ufrag() -> String {
    // Use a simple deterministic pattern for testing
    let chars: Vec<char> = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        .chars().collect();
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);

    (0..8)
        .map(|i| {
            let idx = ((seed.wrapping_mul(1103515245).wrapping_add(12345 + i)) % 62) as usize;
            chars[idx]
        })
        .collect()
}

/// Generate random ICE pwd (22-256 characters)
fn generate_ice_pwd() -> String {
    let chars: Vec<char> = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/"
        .chars().collect();
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);

    (0..24)
        .map(|i| {
            let idx = ((seed.wrapping_mul(1103515245).wrapping_add(12345 + i * 7)) % 64) as usize;
            chars[idx]
        })
        .collect()
}

/// Generate random u64 for tie-breaker
fn rand_u64() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

/// Get local network addresses
fn get_local_addresses() -> Vec<IpAddr> {
    // Return localhost for now - in production would enumerate interfaces
    vec![
        "127.0.0.1".parse().unwrap(),
    ]
}

/// Parse ICE credentials from SDP
fn parse_ice_credentials(sdp: &str) -> Option<(String, String)> {
    let mut ufrag = None;
    let mut pwd = None;

    for line in sdp.lines() {
        if line.starts_with("a=ice-ufrag:") {
            ufrag = Some(line.trim_start_matches("a=ice-ufrag:").to_string());
        } else if line.starts_with("a=ice-pwd:") {
            pwd = Some(line.trim_start_matches("a=ice-pwd:").to_string());
        }
    }

    ufrag.and_then(|u| pwd.map(|p| (u, p)))
}

/// Parse ICE candidates from SDP
fn parse_ice_candidates(sdp: &str) -> Vec<IceCandidate> {
    let mut candidates = Vec::new();

    for line in sdp.lines() {
        if line.starts_with("a=candidate:") {
            let candidate_str = line.trim_start_matches("a=");
            if let Ok(candidate) = IceCandidate::parse(candidate_str) {
                candidates.push(candidate);
            }
        }
    }

    candidates
}

/// Signaling message types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignalingMessageType {
    /// SDP offer
    Offer,
    /// SDP answer
    Answer,
    /// ICE candidate
    Candidate,
    /// Join room
    Join,
    /// Leave room
    Leave,
    /// Error
    Error,
}

/// Signaling message
#[derive(Debug, Clone)]
pub struct SignalingMessage {
    /// Message type
    pub msg_type: SignalingMessageType,
    /// Room ID
    pub room_id: Option<String>,
    /// Peer ID
    pub peer_id: Option<String>,
    /// SDP payload
    pub sdp: Option<String>,
    /// ICE candidate payload
    pub candidate: Option<String>,
    /// Error message
    pub error: Option<String>,
}

impl SignalingMessage {
    /// Parse a signaling message from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        // Simple JSON parsing without external dependencies
        let msg_type = if json.contains("\"type\":\"offer\"") {
            SignalingMessageType::Offer
        } else if json.contains("\"type\":\"answer\"") {
            SignalingMessageType::Answer
        } else if json.contains("\"type\":\"candidate\"") {
            SignalingMessageType::Candidate
        } else if json.contains("\"type\":\"join\"") {
            SignalingMessageType::Join
        } else if json.contains("\"type\":\"leave\"") {
            SignalingMessageType::Leave
        } else if json.contains("\"type\":\"error\"") {
            SignalingMessageType::Error
        } else {
            return Err(LiveError::Protocol("unknown message type".into()));
        };

        let room_id = extract_json_string(json, "roomId");
        let peer_id = extract_json_string(json, "peerId");
        let sdp = extract_json_string(json, "sdp");
        let candidate = extract_json_string(json, "candidate");
        let error = extract_json_string(json, "error");

        Ok(Self {
            msg_type,
            room_id,
            peer_id,
            sdp,
            candidate,
            error,
        })
    }

    /// Convert to JSON string
    pub fn to_json(&self) -> String {
        let type_str = match self.msg_type {
            SignalingMessageType::Offer => "offer",
            SignalingMessageType::Answer => "answer",
            SignalingMessageType::Candidate => "candidate",
            SignalingMessageType::Join => "join",
            SignalingMessageType::Leave => "leave",
            SignalingMessageType::Error => "error",
        };

        let mut parts = vec![format!("\"type\":\"{}\"", type_str)];

        if let Some(ref room_id) = self.room_id {
            parts.push(format!("\"roomId\":\"{}\"", room_id));
        }
        if let Some(ref peer_id) = self.peer_id {
            parts.push(format!("\"peerId\":\"{}\"", peer_id));
        }
        if let Some(ref sdp) = self.sdp {
            parts.push(format!("\"sdp\":\"{}\"", escape_json(sdp)));
        }
        if let Some(ref candidate) = self.candidate {
            parts.push(format!("\"candidate\":\"{}\"", escape_json(candidate)));
        }
        if let Some(ref error) = self.error {
            parts.push(format!("\"error\":\"{}\"", escape_json(error)));
        }

        format!("{{{}}}", parts.join(","))
    }
}

/// Extract a string value from JSON
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\":\"", key);
    if let Some(start) = json.find(&pattern) {
        let value_start = start + pattern.len();
        let remaining = &json[value_start..];

        // Find the closing quote, accounting for escapes
        let mut end = 0;
        let mut escaped = false;
        for (i, c) in remaining.char_indices() {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                end = i;
                break;
            }
        }

        if end > 0 {
            return Some(unescape_json(&remaining[..end]));
        }
    }
    None
}

/// Escape special characters for JSON
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Unescape JSON string
fn unescape_json(s: &str) -> String {
    s.replace("\\\\", "\x00")
        .replace("\\\"", "\"")
        .replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
        .replace('\x00', "\\")
}

/// State for a connected WebRTC peer
#[derive(Debug)]
#[allow(dead_code)]
struct WebRtcPeerState {
    /// Peer ID
    peer_id: String,
    /// Room ID
    room_id: Option<String>,
    /// Connection state
    state: PeerConnectionState,
    /// Stream metadata
    metadata: StreamMetadata,
    /// Last activity timestamp
    last_activity: Instant,
}

/// State for a signaling room
#[derive(Debug)]
#[allow(dead_code)]
struct SignalingRoom {
    /// Room ID
    room_id: String,
    /// Peer IDs in this room
    peers: Vec<String>,
    /// Room creation time
    created_at: Instant,
}

/// WebRTC signaling server
///
/// Provides WebSocket-like signaling over HTTP for WebRTC peer connections.
/// Handles offer/answer exchange and ICE candidate trickle.
pub struct WebRtcServer {
    /// Listen address
    addr: SocketAddr,
    /// Connected peers
    peers: Arc<RwLock<HashMap<String, WebRtcPeerState>>>,
    /// Signaling rooms
    rooms: Arc<RwLock<HashMap<String, SignalingRoom>>>,
    /// Packet broadcaster
    packet_tx: broadcast::Sender<StreamPacket>,
    /// Running flag
    running: Arc<RwLock<bool>>,
    /// WebRTC configuration
    config: WebRtcConfig,
    /// Next peer ID counter
    next_peer_id: Arc<RwLock<u64>>,
}

impl WebRtcServer {
    /// Create a new WebRTC signaling server
    pub fn new(addr: SocketAddr, config: WebRtcConfig) -> Self {
        let (packet_tx, _) = broadcast::channel(1024);
        Self {
            addr,
            peers: Arc::new(RwLock::new(HashMap::new())),
            rooms: Arc::new(RwLock::new(HashMap::new())),
            packet_tx,
            running: Arc::new(RwLock::new(false)),
            config,
            next_peer_id: Arc::new(RwLock::new(1)),
        }
    }

    /// Start the WebRTC signaling server
    pub async fn start(&self) -> Result<()> {
        let listener = TcpListener::bind(self.addr)
            .await
            .map_err(|e| LiveError::Connection(e.to_string()))?;

        tracing::info!("WebRTC signaling server listening on {}", self.addr);

        {
            let mut running = self.running.write().await;
            *running = true;
        }

        let peers = Arc::clone(&self.peers);
        let rooms = Arc::clone(&self.rooms);
        let packet_tx = self.packet_tx.clone();
        let running = Arc::clone(&self.running);
        let next_peer_id = Arc::clone(&self.next_peer_id);
        let config = self.config.clone();

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
                        tracing::debug!("New WebRTC signaling connection from {}", addr);

                        let peers = Arc::clone(&peers);
                        let rooms = Arc::clone(&rooms);
                        let packet_tx = packet_tx.clone();
                        let next_peer_id = Arc::clone(&next_peer_id);
                        let config = config.clone();

                        tokio::spawn(async move {
                            if let Err(e) = Self::handle_connection(
                                socket,
                                peers,
                                rooms,
                                packet_tx,
                                next_peer_id,
                                config,
                            )
                            .await
                            {
                                tracing::warn!(
                                    "WebRTC signaling connection error from {}: {}",
                                    addr,
                                    e
                                );
                            }
                        });
                    }
                    Err(e) => {
                        tracing::error!("Failed to accept WebRTC connection: {}", e);
                    }
                }
            }
        });

        Ok(())
    }

    /// Stop the WebRTC signaling server
    pub async fn stop(&self) -> Result<()> {
        let mut running = self.running.write().await;
        *running = false;
        tracing::info!("WebRTC signaling server stopped");
        Ok(())
    }

    /// Get a receiver for stream packets
    pub fn subscribe(&self) -> broadcast::Receiver<StreamPacket> {
        self.packet_tx.subscribe()
    }

    /// Get active peer count
    pub async fn peer_count(&self) -> usize {
        let peers = self.peers.read().await;
        peers.len()
    }

    /// Get room count
    pub async fn room_count(&self) -> usize {
        let rooms = self.rooms.read().await;
        rooms.len()
    }

    /// Handle a signaling connection
    async fn handle_connection(
        socket: TcpStream,
        peers: Arc<RwLock<HashMap<String, WebRtcPeerState>>>,
        rooms: Arc<RwLock<HashMap<String, SignalingRoom>>>,
        packet_tx: broadcast::Sender<StreamPacket>,
        next_peer_id: Arc<RwLock<u64>>,
        _config: WebRtcConfig,
    ) -> Result<()> {
        let (reader, mut writer) = socket.into_split();
        let mut reader = BufReader::new(reader);

        // Assign peer ID
        let peer_id = {
            let mut id = next_peer_id.write().await;
            let peer_id = format!("peer_{}", *id);
            *id += 1;
            peer_id
        };

        // Register peer
        {
            let mut peers_guard = peers.write().await;
            peers_guard.insert(
                peer_id.clone(),
                WebRtcPeerState {
                    peer_id: peer_id.clone(),
                    room_id: None,
                    state: PeerConnectionState::New,
                    metadata: StreamMetadata::default(),
                    last_activity: Instant::now(),
                },
            );
        }

        tracing::info!("WebRTC peer {} connected", peer_id);

        // Simple HTTP-style signaling protocol
        // Read lines and parse as JSON messages
        let mut line = String::new();
        let mut current_room: Option<String> = None;

        loop {
            line.clear();

            match tokio::time::timeout(Duration::from_secs(60), reader.read_line(&mut line)).await {
                Ok(Ok(0)) => break, // Connection closed
                Ok(Ok(_)) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    // Handle HTTP upgrade request (simplified WebSocket handshake)
                    if trimmed.starts_with("GET ") || trimmed.starts_with("POST ") {
                        // Skip HTTP headers until empty line
                        loop {
                            line.clear();
                            if reader.read_line(&mut line).await.is_err() {
                                break;
                            }
                            if line.trim().is_empty() {
                                break;
                            }
                        }

                        // Send a simple response
                        let response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{{\"peerId\":\"{}\"}}\n",
                            peer_id
                        );
                        writer
                            .write_all(response.as_bytes())
                            .await
                            .map_err(|e| LiveError::Connection(e.to_string()))?;
                        continue;
                    }

                    // Parse signaling message
                    match SignalingMessage::from_json(trimmed) {
                        Ok(msg) => {
                            let response = Self::process_message(
                                &msg,
                                &peer_id,
                                &mut current_room,
                                &peers,
                                &rooms,
                                &packet_tx,
                            )
                            .await?;

                            if let Some(resp_msg) = response {
                                let json = resp_msg.to_json();
                                writer
                                    .write_all(format!("{}\n", json).as_bytes())
                                    .await
                                    .map_err(|e| LiveError::Connection(e.to_string()))?;
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Invalid signaling message from {}: {}", peer_id, e);
                        }
                    }
                }
                Ok(Err(e)) => {
                    tracing::warn!("Read error from {}: {}", peer_id, e);
                    break;
                }
                Err(_) => {
                    // Timeout - send keepalive or check connection
                    continue;
                }
            }
        }

        // Cleanup on disconnect
        Self::cleanup_peer(&peer_id, &current_room, &peers, &rooms).await;
        tracing::info!("WebRTC peer {} disconnected", peer_id);

        Ok(())
    }

    /// Process a signaling message
    async fn process_message(
        msg: &SignalingMessage,
        peer_id: &str,
        current_room: &mut Option<String>,
        peers: &Arc<RwLock<HashMap<String, WebRtcPeerState>>>,
        rooms: &Arc<RwLock<HashMap<String, SignalingRoom>>>,
        _packet_tx: &broadcast::Sender<StreamPacket>,
    ) -> Result<Option<SignalingMessage>> {
        match msg.msg_type {
            SignalingMessageType::Join => {
                let room_id = msg
                    .room_id
                    .clone()
                    .unwrap_or_else(|| format!("room_{}", rand_u64() % 10000));

                // Join or create room
                {
                    let mut rooms_guard = rooms.write().await;
                    let room = rooms_guard.entry(room_id.clone()).or_insert_with(|| {
                        SignalingRoom {
                            room_id: room_id.clone(),
                            peers: Vec::new(),
                            created_at: Instant::now(),
                        }
                    });
                    room.peers.push(peer_id.to_string());
                }

                // Update peer state
                {
                    let mut peers_guard = peers.write().await;
                    if let Some(peer) = peers_guard.get_mut(peer_id) {
                        peer.room_id = Some(room_id.clone());
                        peer.last_activity = Instant::now();
                    }
                }

                *current_room = Some(room_id.clone());
                tracing::info!("Peer {} joined room {}", peer_id, room_id);

                Ok(Some(SignalingMessage {
                    msg_type: SignalingMessageType::Join,
                    room_id: Some(room_id),
                    peer_id: Some(peer_id.to_string()),
                    sdp: None,
                    candidate: None,
                    error: None,
                }))
            }
            SignalingMessageType::Leave => {
                if let Some(ref room_id) = current_room {
                    Self::leave_room(peer_id, room_id, rooms).await;
                    *current_room = None;
                }

                Ok(Some(SignalingMessage {
                    msg_type: SignalingMessageType::Leave,
                    room_id: None,
                    peer_id: Some(peer_id.to_string()),
                    sdp: None,
                    candidate: None,
                    error: None,
                }))
            }
            SignalingMessageType::Offer | SignalingMessageType::Answer => {
                // Update peer state
                {
                    let mut peers_guard = peers.write().await;
                    if let Some(peer) = peers_guard.get_mut(peer_id) {
                        peer.state = PeerConnectionState::Connecting;
                        peer.last_activity = Instant::now();
                    }
                }

                // Forward to other peers in room
                // In a real implementation, this would be sent to the target peer
                tracing::debug!(
                    "Received {:?} from {} for room {:?}",
                    msg.msg_type,
                    peer_id,
                    current_room
                );

                Ok(None)
            }
            SignalingMessageType::Candidate => {
                // Process ICE candidate
                if let Some(ref candidate_str) = msg.candidate {
                    // Parse and forward candidate
                    tracing::debug!(
                        "Received ICE candidate from {}: {}",
                        peer_id,
                        candidate_str
                    );

                    // In a complete implementation, forward to target peer
                }

                Ok(None)
            }
            SignalingMessageType::Error => {
                tracing::warn!(
                    "Received error from {}: {:?}",
                    peer_id,
                    msg.error
                );
                Ok(None)
            }
        }
    }

    /// Leave a room
    async fn leave_room(
        peer_id: &str,
        room_id: &str,
        rooms: &Arc<RwLock<HashMap<String, SignalingRoom>>>,
    ) {
        let mut rooms_guard = rooms.write().await;
        if let Some(room) = rooms_guard.get_mut(room_id) {
            room.peers.retain(|p| p != peer_id);

            // Remove empty rooms
            if room.peers.is_empty() {
                rooms_guard.remove(room_id);
                tracing::debug!("Room {} removed (empty)", room_id);
            }
        }
    }

    /// Cleanup a disconnected peer
    async fn cleanup_peer(
        peer_id: &str,
        current_room: &Option<String>,
        peers: &Arc<RwLock<HashMap<String, WebRtcPeerState>>>,
        rooms: &Arc<RwLock<HashMap<String, SignalingRoom>>>,
    ) {
        // Leave room if in one
        if let Some(ref room_id) = current_room {
            Self::leave_room(peer_id, room_id, rooms).await;
        }

        // Remove peer
        let mut peers_guard = peers.write().await;
        peers_guard.remove(peer_id);
    }
}

/// WebRTC client for connecting to signaling servers
pub struct WebRtcClient {
    /// Server address
    addr: String,
    /// Peer connection
    peer: WebRtcPeer,
    /// Peer ID assigned by server
    peer_id: Option<String>,
    /// Current room
    room_id: Option<String>,
}

impl WebRtcClient {
    /// Create a new WebRTC client
    pub fn new(addr: &str, config: WebRtcConfig) -> Self {
        Self {
            addr: addr.to_string(),
            peer: WebRtcPeer::new(config),
            peer_id: None,
            room_id: None,
        }
    }

    /// Connect to the signaling server
    pub async fn connect(&mut self) -> Result<()> {
        let mut socket = TcpStream::connect(&self.addr)
            .await
            .map_err(|e| LiveError::Connection(e.to_string()))?;

        // Send HTTP request to get peer ID
        let request = "GET /signaling HTTP/1.1\r\nHost: localhost\r\n\r\n";
        socket
            .write_all(request.as_bytes())
            .await
            .map_err(|e| LiveError::Connection(e.to_string()))?;

        // Read response
        let mut buf = vec![0u8; 1024];
        let n = tokio::time::timeout(Duration::from_secs(5), socket.read(&mut buf))
            .await
            .map_err(|_| LiveError::Timeout)?
            .map_err(|e| LiveError::Connection(e.to_string()))?;

        // Parse peer ID from response
        let response = String::from_utf8_lossy(&buf[..n]);
        if let Some(peer_id) = extract_json_string(&response, "peerId") {
            self.peer_id = Some(peer_id);
        }

        tracing::info!(
            "WebRTC client connected to {}, peer ID: {:?}",
            self.addr,
            self.peer_id
        );

        Ok(())
    }

    /// Join a room
    pub async fn join_room(&mut self, room_id: &str) -> Result<()> {
        self.room_id = Some(room_id.to_string());
        // In a real implementation, send join message to server
        Ok(())
    }

    /// Create an offer
    pub async fn create_offer(&mut self) -> Result<SessionDescription> {
        self.peer.create_offer().await
    }

    /// Set remote description
    pub async fn set_remote_description(&mut self, desc: SessionDescription) -> Result<()> {
        self.peer.set_remote_description(desc).await
    }

    /// Add ICE candidate
    pub fn add_ice_candidate(&mut self, candidate: IceCandidate) -> Result<()> {
        self.peer.add_ice_candidate(candidate)
    }

    /// Get connection state
    pub fn state(&self) -> PeerConnectionState {
        self.peer.state()
    }

    /// Get peer ID
    pub fn peer_id(&self) -> Option<&str> {
        self.peer_id.as_deref()
    }
}

use tokio::io::AsyncReadExt;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ice_candidate_parse() {
        let candidate_str = "candidate:0 1 UDP 2122252543 192.168.1.100 54321 typ host";
        let candidate = IceCandidate::parse(candidate_str).unwrap();

        assert_eq!(candidate.foundation, "0");
        assert_eq!(candidate.component, 1);
        assert_eq!(candidate.transport, IceTransport::Udp);
        assert_eq!(candidate.candidate_type, IceCandidateType::Host);
        assert_eq!(candidate.address.unwrap().port(), 54321);
    }

    #[test]
    fn test_ice_candidate_priority() {
        let priority = IceCandidate::calculate_priority(126, 65535, 1);
        assert!(priority > 0);

        let lower_priority = IceCandidate::calculate_priority(100, 65535, 1);
        assert!(priority > lower_priority);
    }

    #[test]
    fn test_candidate_pair_priority() {
        let p1 = CandidatePair::calculate_priority(true, 100, 50);
        let p2 = CandidatePair::calculate_priority(false, 100, 50);
        // Both should produce valid priorities
        assert!(p1 > 0);
        assert!(p2 > 0);
    }

    #[test]
    fn test_ice_agent_creation() {
        let agent = IceAgent::new(true);
        assert_eq!(agent.gathering_state(), IceGatheringState::New);
        assert_eq!(agent.connection_state(), IceConnectionState::New);
    }

    #[test]
    fn test_dtls_fingerprint() {
        let fp = DtlsFingerprint::generate_dummy();
        assert_eq!(fp.algorithm, "sha-256");
        let sdp = fp.to_sdp_attribute();
        assert!(sdp.starts_with("fingerprint:sha-256"));
    }

    #[test]
    fn test_webrtc_peer_creation() {
        let peer = WebRtcPeer::default();
        assert_eq!(peer.state(), PeerConnectionState::New);
        assert_eq!(peer.ice_gathering_state(), IceGatheringState::New);
    }

    #[tokio::test]
    async fn test_create_offer() {
        let mut peer = WebRtcPeer::default();
        let offer = peer.create_offer().await.unwrap();

        assert_eq!(offer.sdp_type, SdpType::Offer);
        assert!(offer.sdp.contains("v=0"));
        assert!(offer.sdp.contains("a=ice-ufrag:"));
        assert!(offer.sdp.contains("a=fingerprint:"));
    }

    #[test]
    fn test_parse_ice_credentials() {
        let sdp = "v=0\r\na=ice-ufrag:test123\r\na=ice-pwd:password456\r\n";
        let (ufrag, pwd) = parse_ice_credentials(sdp).unwrap();
        assert_eq!(ufrag, "test123");
        assert_eq!(pwd, "password456");
    }

    #[test]
    fn test_signaling_message_parse() {
        let json = r#"{"type":"join","roomId":"room_123","peerId":"peer_1"}"#;
        let msg = SignalingMessage::from_json(json).unwrap();

        assert_eq!(msg.msg_type, SignalingMessageType::Join);
        assert_eq!(msg.room_id, Some("room_123".to_string()));
        assert_eq!(msg.peer_id, Some("peer_1".to_string()));
    }

    #[test]
    fn test_signaling_message_to_json() {
        let msg = SignalingMessage {
            msg_type: SignalingMessageType::Offer,
            room_id: Some("room_456".to_string()),
            peer_id: Some("peer_2".to_string()),
            sdp: None,
            candidate: None,
            error: None,
        };

        let json = msg.to_json();
        assert!(json.contains("\"type\":\"offer\""));
        assert!(json.contains("\"roomId\":\"room_456\""));
        assert!(json.contains("\"peerId\":\"peer_2\""));
    }

    #[test]
    fn test_webrtc_server_creation() {
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let config = WebRtcConfig::default();
        let server = WebRtcServer::new(addr, config);
        assert_eq!(server.addr, addr);
    }

    #[test]
    fn test_webrtc_client_creation() {
        let config = WebRtcConfig::default();
        let client = WebRtcClient::new("127.0.0.1:8080", config);
        assert_eq!(client.state(), PeerConnectionState::New);
        assert!(client.peer_id().is_none());
    }

    #[test]
    fn test_json_escape_unescape() {
        let original = "line1\nline2\ttab\"quote";
        let escaped = escape_json(original);
        let unescaped = unescape_json(&escaped);
        assert_eq!(original, unescaped);
    }
}
