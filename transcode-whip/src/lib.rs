//! WHIP/WHEP WebRTC-based live ingest protocol support.
//!
//! This crate implements the WebRTC-HTTP Ingest Protocol (WHIP) and
//! WebRTC-HTTP Egress Protocol (WHEP) for live streaming ingest and
//! playback with sub-second latency.
//!
//! # Features
//!
//! - **WHIP**: Ingest live streams from WebRTC clients (browsers, OBS, etc.)
//! - **WHEP**: Serve live streams to WebRTC viewers
//! - **SDP Negotiation**: Full SDP offer/answer handling
//! - **ICE Trickle**: Support for incremental ICE candidate exchange
//! - **Codec Support**: H.264, VP8, VP9, AV1, Opus
//! - **Session Management**: Track and manage active streams
//!
//! # Example
//!
//! ```no_run
//! use transcode_whip::{WhipServer, ServerConfig};
//!
//! #[tokio::main]
//! async fn main() -> transcode_whip::Result<()> {
//!     let config = ServerConfig {
//!         bind_address: "0.0.0.0:8080".to_string(),
//!         max_sessions: 100,
//!         ..Default::default()
//!     };
//!
//!     let server = WhipServer::new(config);
//!
//!     // Subscribe to events
//!     let mut events = server.subscribe();
//!     tokio::spawn(async move {
//!         while let Ok(event) = events.recv().await {
//!             println!("Event: {:?}", event);
//!         }
//!     });
//!
//!     server.run().await
//! }
//! ```
//!
//! # Protocol Overview
//!
//! ## WHIP (Ingest)
//!
//! WHIP allows publishers to send live media to the server:
//!
//! 1. Client sends POST to `/whip` with SDP offer
//! 2. Server responds with 201 Created and SDP answer
//! 3. WebRTC connection established
//! 4. Client streams media to server
//! 5. Client sends DELETE to end session
//!
//! ## WHEP (Egress)
//!
//! WHEP allows viewers to receive live media from the server:
//!
//! 1. Client sends POST to `/whep` with SDP offer
//! 2. Server responds with 201 Created and SDP answer
//! 3. WebRTC connection established
//! 4. Server streams media to client
//! 5. Client sends DELETE to end session
//!
//! # Integration with Transcode Pipeline
//!
//! The WHIP server can be integrated with the transcode pipeline for
//! live transcoding:
//!
//! ```ignore
//! // WHIP -> Decode -> Transcode -> Encode -> HLS/DASH
//! let whip = WhipServer::new(config);
//! let pipeline = Pipeline::new()
//!     .source(whip.as_source())
//!     .transcode(h264_to_av1())
//!     .output(HlsOutput::new("live"));
//! ```

#![allow(dead_code)]

mod error;
mod sdp;
mod server;
mod session;

pub use error::{Result, WhipError};
pub use sdp::{
    Codec, IceCandidate, MediaDescription, MediaDirection, MediaType, SessionDescription,
};
pub use server::{IceServer, ServerConfig, ServerEvent, ServerState, WhipServer};
pub use session::{
    MediaTrack, Session, SessionManager, SessionManagerHandle, SessionState, SessionStats,
    SessionType,
};

/// Protocol version.
pub const PROTOCOL_VERSION: &str = "draft-ietf-wish-whip-08";

/// WHEP protocol version.
pub const WHEP_VERSION: &str = "draft-ietf-wish-whep-01";

/// Default WHIP content type.
pub const WHIP_CONTENT_TYPE: &str = "application/sdp";

/// ICE trickle content type.
pub const TRICKLE_CONTENT_TYPE: &str = "application/trickle-ice-sdpfrag";

/// Builder for creating a WHIP server with custom configuration.
#[derive(Debug, Clone)]
pub struct WhipServerBuilder {
    config: ServerConfig,
}

impl Default for WhipServerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl WhipServerBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: ServerConfig::default(),
        }
    }

    /// Set the bind address.
    pub fn bind(mut self, address: impl Into<String>) -> Self {
        self.config.bind_address = address.into();
        self
    }

    /// Set the maximum number of sessions.
    pub fn max_sessions(mut self, max: usize) -> Self {
        self.config.max_sessions = max;
        self
    }

    /// Set the WHIP endpoint path.
    pub fn whip_path(mut self, path: impl Into<String>) -> Self {
        self.config.whip_path = path.into();
        self
    }

    /// Set the WHEP endpoint path.
    pub fn whep_path(mut self, path: impl Into<String>) -> Self {
        self.config.whep_path = path.into();
        self
    }

    /// Add an ICE server.
    pub fn ice_server(mut self, url: impl Into<String>) -> Self {
        self.config.ice_servers.push(IceServer {
            urls: vec![url.into()],
            username: None,
            credential: None,
        });
        self
    }

    /// Add a TURN server with credentials.
    pub fn turn_server(
        mut self,
        url: impl Into<String>,
        username: impl Into<String>,
        credential: impl Into<String>,
    ) -> Self {
        self.config.ice_servers.push(IceServer {
            urls: vec![url.into()],
            username: Some(username.into()),
            credential: Some(credential.into()),
        });
        self
    }

    /// Enable or disable CORS.
    pub fn cors(mut self, enabled: bool) -> Self {
        self.config.enable_cors = enabled;
        self
    }

    /// Build the server.
    pub fn build(self) -> WhipServer {
        WhipServer::new(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_versions() {
        assert!(PROTOCOL_VERSION.contains("whip"));
        assert!(WHEP_VERSION.contains("whep"));
    }

    #[test]
    fn test_builder() {
        let server = WhipServerBuilder::new()
            .bind("127.0.0.1:9090")
            .max_sessions(50)
            .whip_path("/ingest")
            .whep_path("/play")
            .ice_server("stun:stun.example.com:3478")
            .cors(true)
            .build();

        assert_eq!(server.config().bind_address, "127.0.0.1:9090");
        assert_eq!(server.config().max_sessions, 50);
        assert_eq!(server.config().whip_path, "/ingest");
        assert_eq!(server.config().whep_path, "/play");
    }

    #[test]
    fn test_session_type_variants() {
        let whip = SessionType::Whip;
        let whep = SessionType::Whep;
        assert_ne!(whip, whep);
    }
}
