//! # Transcode Ingest
//!
//! A library for handling live media ingest protocols including RTMP and SRT.
//!
//! ## Features
//!
//! - RTMP server/client with full handshake support
//! - SRT protocol with reliable UDP transport
//! - Common `IngestSource` trait for protocol abstraction
//! - Async tokio-based implementation
//!
//! ## Example
//!
//! ```rust,no_run
//! use transcode_ingest::{IngestSource, RtmpServer, SrtServer};
//! use std::net::SocketAddr;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let addr: SocketAddr = "0.0.0.0:1935".parse()?;
//!     let rtmp = RtmpServer::bind(addr).await?;
//!     // Handle incoming connections...
//!     Ok(())
//! }
//! ```

pub mod rtmp;
pub mod srt;

use async_trait::async_trait;
use bytes::Bytes;
use std::net::SocketAddr;
use std::time::Duration;
use thiserror::Error;

// Re-exports
pub use rtmp::{RtmpClient, RtmpConnection, RtmpServer};
pub use srt::{SrtClient, SrtConnection, SrtServer};

/// Errors that can occur during ingest operations
#[derive(Error, Debug)]
pub enum IngestError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Connection closed")]
    ConnectionClosed,

    #[error("Handshake failed: {0}")]
    HandshakeFailed(String),

    #[error("Protocol error: {0}")]
    ProtocolError(String),

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("Invalid message: {0}")]
    InvalidMessage(String),

    #[error("Unsupported feature: {0}")]
    Unsupported(String),

    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),

    #[error("Stream not found: {0}")]
    StreamNotFound(String),

    #[error("Encryption error: {0}")]
    EncryptionError(String),
}

/// Result type for ingest operations
pub type IngestResult<T> = Result<T, IngestError>;

/// Media data types that can be received from an ingest source
#[derive(Debug, Clone)]
pub enum MediaData {
    /// Audio data with codec information
    Audio {
        /// Raw audio bytes
        data: Bytes,
        /// Timestamp in milliseconds
        timestamp: u32,
        /// Audio codec identifier
        codec: AudioCodec,
        /// Sample rate in Hz
        sample_rate: u32,
        /// Number of channels
        channels: u8,
    },
    /// Video data with codec information
    Video {
        /// Raw video bytes
        data: Bytes,
        /// Timestamp in milliseconds
        timestamp: u32,
        /// Video codec identifier
        codec: VideoCodec,
        /// Whether this is a keyframe
        is_keyframe: bool,
        /// Composition time offset (for B-frames)
        composition_time: i32,
    },
    /// Metadata (e.g., stream info)
    Metadata {
        /// Key-value pairs of metadata
        properties: Vec<(String, MetadataValue)>,
    },
}

/// Supported audio codecs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioCodec {
    Aac,
    Mp3,
    Pcm,
    Speex,
    Opus,
    Unknown(u8),
}

/// Supported video codecs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    H264,
    H265,
    Vp8,
    Vp9,
    Av1,
    Unknown(u8),
}

/// Metadata value types
#[derive(Debug, Clone)]
pub enum MetadataValue {
    Number(f64),
    Boolean(bool),
    String(String),
    Null,
}

/// Stream information
#[derive(Debug, Clone)]
pub struct StreamInfo {
    /// Stream name/key
    pub name: String,
    /// Application name (for RTMP)
    pub app: String,
    /// Remote peer address
    pub peer_addr: SocketAddr,
    /// Connection start time
    pub connected_at: std::time::Instant,
}

/// Ingest connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Initial state, not connected
    Disconnected,
    /// Handshake in progress
    Handshaking,
    /// Connected and ready
    Connected,
    /// Publishing media data
    Publishing,
    /// Playing/receiving media data
    Playing,
    /// Connection closing
    Closing,
}

/// Configuration for ingest connections
#[derive(Debug, Clone)]
pub struct IngestConfig {
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Read timeout
    pub read_timeout: Duration,
    /// Write timeout
    pub write_timeout: Duration,
    /// Maximum chunk size
    pub max_chunk_size: usize,
    /// Buffer size for incoming data
    pub buffer_size: usize,
}

impl Default for IngestConfig {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_secs(10),
            read_timeout: Duration::from_secs(30),
            write_timeout: Duration::from_secs(30),
            max_chunk_size: 4096,
            buffer_size: 65536,
        }
    }
}

/// Common trait for all ingest sources
///
/// This trait provides a unified interface for both RTMP and SRT protocols,
/// allowing applications to handle different ingest sources uniformly.
#[async_trait]
pub trait IngestSource: Send + Sync {
    /// Returns the stream information for this connection
    fn stream_info(&self) -> &StreamInfo;

    /// Returns the current connection state
    fn state(&self) -> ConnectionState;

    /// Reads the next media data from the source
    ///
    /// Returns `None` when the stream ends normally.
    async fn read_media(&mut self) -> IngestResult<Option<MediaData>>;

    /// Sends media data to the source (for playback/relay)
    async fn write_media(&mut self, data: MediaData) -> IngestResult<()>;

    /// Closes the connection gracefully
    async fn close(&mut self) -> IngestResult<()>;

    /// Returns the remote peer address
    fn peer_addr(&self) -> SocketAddr {
        self.stream_info().peer_addr
    }

    /// Returns the stream name
    fn stream_name(&self) -> &str {
        &self.stream_info().name
    }
}

/// Trait for ingest servers that accept connections
#[async_trait]
pub trait IngestServer: Send + Sync {
    /// The connection type this server produces
    type Connection: IngestSource;

    /// Accepts an incoming connection
    async fn accept(&mut self) -> IngestResult<Self::Connection>;

    /// Returns the local address the server is bound to
    fn local_addr(&self) -> IngestResult<SocketAddr>;

    /// Shuts down the server
    async fn shutdown(&mut self) -> IngestResult<()>;
}

/// Trait for ingest clients that connect to servers
#[async_trait]
pub trait IngestClient: Send + Sync {
    /// The connection type this client produces
    type Connection: IngestSource;

    /// Connects to a remote server
    async fn connect(addr: SocketAddr, config: IngestConfig) -> IngestResult<Self::Connection>
    where
        Self: Sized;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = IngestConfig::default();
        assert_eq!(config.connect_timeout, Duration::from_secs(10));
        assert_eq!(config.max_chunk_size, 4096);
        assert_eq!(config.buffer_size, 65536);
    }

    #[test]
    fn test_audio_codec_equality() {
        assert_eq!(AudioCodec::Aac, AudioCodec::Aac);
        assert_ne!(AudioCodec::Aac, AudioCodec::Mp3);
        assert_eq!(AudioCodec::Unknown(1), AudioCodec::Unknown(1));
    }

    #[test]
    fn test_video_codec_equality() {
        assert_eq!(VideoCodec::H264, VideoCodec::H264);
        assert_ne!(VideoCodec::H264, VideoCodec::H265);
    }

    #[test]
    fn test_connection_state() {
        let state = ConnectionState::Disconnected;
        assert_eq!(state, ConnectionState::Disconnected);
        assert_ne!(state, ConnectionState::Connected);
    }
}
