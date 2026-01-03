//! WebRTC protocol support

use crate::{LiveError, Result};

/// ICE candidate
#[derive(Debug, Clone)]
pub struct IceCandidate {
    /// Candidate string
    pub candidate: String,
    /// SDP mid
    pub sdp_mid: Option<String>,
    /// SDP mline index
    pub sdp_mline_index: Option<u32>,
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
}

impl Default for WebRtcConfig {
    fn default() -> Self {
        Self {
            stun_servers: vec!["stun:stun.l.google.com:19302".into()],
            turn_servers: Vec::new(),
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

/// WebRTC peer connection
pub struct WebRtcPeer {
    _config: WebRtcConfig,
    state: PeerConnectionState,
    local_description: Option<SessionDescription>,
    remote_description: Option<SessionDescription>,
    ice_candidates: Vec<IceCandidate>,
}

impl WebRtcPeer {
    /// Create a new WebRTC peer
    pub fn new(config: WebRtcConfig) -> Self {
        Self {
            _config: config,
            state: PeerConnectionState::New,
            local_description: None,
            remote_description: None,
            ice_candidates: Vec::new(),
        }
    }

    /// Create an offer
    pub async fn create_offer(&mut self) -> Result<SessionDescription> {
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

        // Generate SDP answer
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
        Ok(())
    }

    /// Set remote description
    pub async fn set_remote_description(&mut self, desc: SessionDescription) -> Result<()> {
        self.remote_description = Some(desc);
        self.state = PeerConnectionState::Connecting;
        Ok(())
    }

    /// Add ICE candidate
    pub fn add_ice_candidate(&mut self, candidate: IceCandidate) {
        self.ice_candidates.push(candidate);
    }

    /// Get connection state
    pub fn state(&self) -> PeerConnectionState {
        self.state
    }

    /// Close the connection
    pub async fn close(&mut self) -> Result<()> {
        self.state = PeerConnectionState::Closed;
        Ok(())
    }

    fn generate_sdp(&self, is_offer: bool) -> String {
        // Minimal SDP for demonstration
        format!(
            "v=0\r\n\
             o=- 0 0 IN IP4 127.0.0.1\r\n\
             s=-\r\n\
             t=0 0\r\n\
             a=group:BUNDLE 0\r\n\
             m=video 9 UDP/TLS/RTP/SAVPF 96\r\n\
             c=IN IP4 0.0.0.0\r\n\
             a=rtpmap:96 H264/90000\r\n\
             a={}\r\n",
            if is_offer { "sendrecv" } else { "recvonly" }
        )
    }
}

impl Default for WebRtcPeer {
    fn default() -> Self {
        Self::new(WebRtcConfig::default())
    }
}
