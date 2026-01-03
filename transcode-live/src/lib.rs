//! Live streaming input for transcode
//!
//! This crate provides support for RTMP, SRT, and WebRTC input streams.

use std::net::SocketAddr;
use std::time::Duration;

/// Minimum buffer size in packets
pub const MIN_BUFFER_SIZE: usize = 16;
/// Maximum buffer size in packets (64K)
pub const MAX_BUFFER_SIZE: usize = 65536;
/// Maximum number of clients
pub const MAX_CLIENTS_LIMIT: usize = 10000;

mod error;
mod rtmp;
mod srt;
mod webrtc;

pub use error::*;
pub use rtmp::*;
pub use srt::*;
pub use webrtc::*;

/// Result type for live streaming operations
pub type Result<T> = std::result::Result<T, LiveError>;

/// Live stream configuration
#[derive(Debug, Clone)]
pub struct LiveConfig {
    /// Protocol to use
    pub protocol: StreamProtocol,
    /// Listen address
    pub listen_addr: SocketAddr,
    /// Connection timeout
    pub timeout: Duration,
    /// Maximum clients
    pub max_clients: usize,
    /// Buffer size in packets
    pub buffer_size: usize,
}

impl Default for LiveConfig {
    fn default() -> Self {
        Self {
            protocol: StreamProtocol::Rtmp,
            listen_addr: "0.0.0.0:1935".parse().unwrap(),
            timeout: Duration::from_secs(30),
            max_clients: 100,
            buffer_size: 1024,
        }
    }
}

impl LiveConfig {
    /// Validate the live configuration
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Buffer size is outside `[MIN_BUFFER_SIZE, MAX_BUFFER_SIZE]`
    /// - Max clients is zero or exceeds `MAX_CLIENTS_LIMIT`
    pub fn validate(&self) -> Result<()> {
        if self.buffer_size < MIN_BUFFER_SIZE || self.buffer_size > MAX_BUFFER_SIZE {
            return Err(LiveError::Configuration(format!(
                "buffer size {} out of valid range [{}, {}]",
                self.buffer_size, MIN_BUFFER_SIZE, MAX_BUFFER_SIZE
            )));
        }

        if self.max_clients == 0 || self.max_clients > MAX_CLIENTS_LIMIT {
            return Err(LiveError::Configuration(format!(
                "max clients {} out of valid range [1, {}]",
                self.max_clients, MAX_CLIENTS_LIMIT
            )));
        }

        Ok(())
    }
}

/// Streaming protocol
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamProtocol {
    /// RTMP protocol
    Rtmp,
    /// SRT protocol
    Srt,
    /// WebRTC protocol
    WebRtc,
}

/// Stream metadata
#[derive(Debug, Clone, Default)]
pub struct StreamMetadata {
    /// Stream ID/key
    pub stream_id: String,
    /// Video codec
    pub video_codec: Option<String>,
    /// Audio codec
    pub audio_codec: Option<String>,
    /// Video width
    pub width: Option<u32>,
    /// Video height
    pub height: Option<u32>,
    /// Frame rate
    pub frame_rate: Option<f64>,
    /// Video bitrate
    pub video_bitrate: Option<u32>,
    /// Audio bitrate
    pub audio_bitrate: Option<u32>,
    /// Audio sample rate
    pub sample_rate: Option<u32>,
    /// Audio channels
    pub channels: Option<u32>,
}

/// Stream packet
#[derive(Debug, Clone)]
pub struct StreamPacket {
    /// Packet type
    pub packet_type: PacketType,
    /// Timestamp in milliseconds
    pub timestamp: u64,
    /// Packet data
    pub data: Vec<u8>,
    /// Is keyframe
    pub is_keyframe: bool,
}

/// Type of stream packet
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacketType {
    /// Video packet
    Video,
    /// Audio packet
    Audio,
    /// Metadata packet
    Metadata,
}

/// Stream state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamState {
    /// Waiting for connection
    Waiting,
    /// Connecting
    Connecting,
    /// Connected and streaming
    Streaming,
    /// Disconnected
    Disconnected,
    /// Error state
    Error,
}

/// Live stream server
pub struct LiveServer {
    config: LiveConfig,
    state: StreamState,
}

impl LiveServer {
    /// Create a new live server
    pub fn new(config: LiveConfig) -> Self {
        Self {
            config,
            state: StreamState::Waiting,
        }
    }

    /// Start the server
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub async fn start(&mut self) -> Result<()> {
        // Validate configuration before starting
        self.config.validate()?;

        tracing::info!(
            "Starting live server on {} ({:?})",
            self.config.listen_addr,
            self.config.protocol
        );

        match self.config.protocol {
            StreamProtocol::Rtmp => self.start_rtmp().await,
            StreamProtocol::Srt => self.start_srt().await,
            StreamProtocol::WebRtc => self.start_webrtc().await,
        }
    }

    async fn start_rtmp(&mut self) -> Result<()> {
        self.state = StreamState::Connecting;
        // RTMP server implementation would go here
        tracing::info!("RTMP server started");
        self.state = StreamState::Streaming;
        Ok(())
    }

    async fn start_srt(&mut self) -> Result<()> {
        self.state = StreamState::Connecting;
        // SRT server implementation would go here
        tracing::info!("SRT server started");
        self.state = StreamState::Streaming;
        Ok(())
    }

    async fn start_webrtc(&mut self) -> Result<()> {
        self.state = StreamState::Connecting;
        // WebRTC server implementation would go here
        tracing::info!("WebRTC server started");
        self.state = StreamState::Streaming;
        Ok(())
    }

    /// Stop the server
    pub async fn stop(&mut self) -> Result<()> {
        tracing::info!("Stopping live server");
        self.state = StreamState::Disconnected;
        Ok(())
    }

    /// Get server state
    pub fn state(&self) -> StreamState {
        self.state
    }

    /// Get configuration
    pub fn config(&self) -> &LiveConfig {
        &self.config
    }
}

/// Live stream client for connecting to external servers
pub struct LiveClient {
    url: url::Url,
    state: StreamState,
}

impl LiveClient {
    /// Create a new live client
    pub fn new(url: &str) -> Result<Self> {
        let url = url::Url::parse(url)
            .map_err(|e| LiveError::InvalidUrl(e.to_string()))?;
        Ok(Self {
            url,
            state: StreamState::Disconnected,
        })
    }

    /// Connect to the server
    pub async fn connect(&mut self) -> Result<StreamMetadata> {
        tracing::info!("Connecting to {}", self.url);
        self.state = StreamState::Connecting;

        // Connection implementation would go here
        self.state = StreamState::Streaming;

        Ok(StreamMetadata::default())
    }

    /// Read next packet
    pub async fn read_packet(&mut self) -> Result<Option<StreamPacket>> {
        if self.state != StreamState::Streaming {
            return Err(LiveError::NotConnected);
        }

        // Packet reading implementation would go here
        Ok(None)
    }

    /// Disconnect from server
    pub async fn disconnect(&mut self) -> Result<()> {
        tracing::info!("Disconnecting from {}", self.url);
        self.state = StreamState::Disconnected;
        Ok(())
    }

    /// Get client state
    pub fn state(&self) -> StreamState {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LiveConfig::default();
        assert_eq!(config.protocol, StreamProtocol::Rtmp);
        assert_eq!(config.max_clients, 100);
    }

    #[test]
    fn test_stream_metadata() {
        let meta = StreamMetadata {
            stream_id: "test".into(),
            video_codec: Some("h264".into()),
            width: Some(1920),
            height: Some(1080),
            ..Default::default()
        };

        assert_eq!(meta.stream_id, "test");
        assert_eq!(meta.width, Some(1920));
    }

    #[test]
    fn test_client_creation() {
        let client = LiveClient::new("rtmp://localhost/live/stream");
        assert!(client.is_ok());

        let client = LiveClient::new("invalid-url");
        assert!(client.is_err());
    }
}
