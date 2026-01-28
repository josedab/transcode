//! Standalone live streaming ingest server.
//!
//! This crate provides a single-binary live ingest server that accepts streams
//! via RTMP, SRT, and WHIP protocols, transcodes in real-time, and outputs to
//! HLS/DASH or forwards via WHEP.
//!
//! # Architecture
//!
//! ```text
//!   ┌──────────┐     ┌──────────────┐     ┌──────────────┐
//!   │  RTMP    │────▶│              │────▶│  HLS Output  │
//!   │  Client  │     │   Ingest     │     └──────────────┘
//!   └──────────┘     │   Server     │     ┌──────────────┐
//!   ┌──────────┐     │              │────▶│ DASH Output  │
//!   │  SRT     │────▶│  Transcode   │     └──────────────┘
//!   │  Client  │     │  Pipeline    │     ┌──────────────┐
//!   └──────────┘     │              │────▶│ WHEP Output  │
//!   ┌──────────┐     │              │     └──────────────┘
//!   │  WHIP    │────▶│              │
//!   │  Client  │     └──────────────┘
//!   └──────────┘
//! ```
//!
//! # Example
//!
//! ```no_run
//! use transcode_ingest_server::{IngestServer, ServerConfig, OutputConfig, OutputFormat};
//!
//! #[tokio::main]
//! async fn main() -> transcode_ingest_server::Result<()> {
//!     let config = ServerConfig {
//!         rtmp_port: 1935,
//!         srt_port: Some(9000),
//!         whip_port: Some(8080),
//!         max_streams: 10,
//!         output: OutputConfig {
//!             format: OutputFormat::Hls,
//!             directory: "/var/hls".into(),
//!             segment_duration_secs: 6.0,
//!         },
//!         ..Default::default()
//!     };
//!
//!     let server = IngestServer::new(config)?;
//!     server.run().await
//! }
//! ```

#![allow(dead_code)]

mod error;
mod server;
mod stream;
mod output;

pub use error::{Error, Result};
pub use server::{IngestServer, ServerConfig};
pub use stream::{StreamSession, StreamStatus, StreamInfo, ProtocolType};
pub use output::{OutputConfig, OutputFormat, OutputManager};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config_defaults() {
        let config = ServerConfig::default();
        assert_eq!(config.rtmp_port, 1935);
        assert_eq!(config.max_streams, 100);
    }

    #[test]
    fn test_stream_session_creation() {
        let session = StreamSession::new(
            "test-stream",
            ProtocolType::Rtmp,
            "192.168.1.1:12345",
        );
        assert_eq!(session.info().stream_key, "test-stream");
        assert_eq!(session.info().protocol, ProtocolType::Rtmp);
        assert!(matches!(session.status(), StreamStatus::Connecting));
    }

    #[test]
    fn test_output_format() {
        assert_eq!(OutputFormat::Hls.extension(), "m3u8");
        assert_eq!(OutputFormat::Dash.extension(), "mpd");
    }

    #[test]
    fn test_server_creation() {
        let config = ServerConfig::default();
        let server = IngestServer::new(config);
        assert!(server.is_ok());
    }
}
