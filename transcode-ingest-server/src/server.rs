//! Ingest server core.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::error::{Error, Result};
use crate::output::OutputConfig;
use crate::stream::{StreamSession, StreamStatus, ProtocolType};

/// Server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub rtmp_port: u16,
    pub srt_port: Option<u16>,
    pub whip_port: Option<u16>,
    pub max_streams: usize,
    pub output: OutputConfig,
    pub bind_address: String,
    /// Authentication token (None = no auth).
    pub auth_token: Option<String>,
    /// Maximum input bitrate per stream (bytes/sec, 0 = unlimited).
    pub max_input_bitrate: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            rtmp_port: 1935,
            srt_port: Some(9000),
            whip_port: Some(8080),
            max_streams: 100,
            output: OutputConfig::default(),
            bind_address: "0.0.0.0".into(),
            auth_token: None,
            max_input_bitrate: 0,
        }
    }
}

/// The live streaming ingest server.
pub struct IngestServer {
    config: ServerConfig,
    streams: Arc<RwLock<HashMap<String, StreamSession>>>,
}

impl IngestServer {
    pub fn new(config: ServerConfig) -> Result<Self> {
        if config.max_streams == 0 {
            return Err(Error::Config {
                message: "max_streams must be > 0".into(),
            });
        }

        Ok(Self {
            config,
            streams: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Run the server (blocks until shutdown).
    pub async fn run(&self) -> Result<()> {
        info!(
            rtmp_port = self.config.rtmp_port,
            srt_port = ?self.config.srt_port,
            whip_port = ?self.config.whip_port,
            "Starting ingest server"
        );

        // In a real implementation, this would spawn listeners for each protocol.
        // For now, we provide the structure.
        tokio::signal::ctrl_c().await.map_err(Error::Io)?;
        info!("Shutting down ingest server");
        Ok(())
    }

    /// Register a new incoming stream.
    pub fn register_stream(
        &self,
        stream_key: &str,
        protocol: ProtocolType,
        remote_addr: &str,
    ) -> Result<()> {
        let mut streams = self.streams.write();
        if streams.len() >= self.config.max_streams {
            return Err(Error::MaxStreams {
                max: self.config.max_streams,
            });
        }

        let session = StreamSession::new(stream_key, protocol, remote_addr);
        info!(
            key = stream_key,
            protocol = ?protocol,
            remote = remote_addr,
            "Stream registered"
        );
        streams.insert(stream_key.to_string(), session);
        Ok(())
    }

    /// Remove a stream.
    pub fn remove_stream(&self, stream_key: &str) -> Result<()> {
        let mut streams = self.streams.write();
        streams
            .remove(stream_key)
            .ok_or_else(|| Error::StreamNotFound {
                key: stream_key.into(),
            })?;
        info!(key = stream_key, "Stream removed");
        Ok(())
    }

    /// Get active stream count.
    pub fn active_streams(&self) -> usize {
        self.streams
            .read()
            .values()
            .filter(|s| matches!(s.status(), StreamStatus::Live))
            .count()
    }

    /// Get total stream count (including connecting/idle).
    pub fn total_streams(&self) -> usize {
        self.streams.read().len()
    }

    /// List all stream keys.
    pub fn stream_keys(&self) -> Vec<String> {
        self.streams.read().keys().cloned().collect()
    }

    pub fn config(&self) -> &ServerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_remove_stream() {
        let server = IngestServer::new(ServerConfig::default()).unwrap();
        server
            .register_stream("test-key", ProtocolType::Rtmp, "127.0.0.1:1234")
            .unwrap();
        assert_eq!(server.total_streams(), 1);

        server.remove_stream("test-key").unwrap();
        assert_eq!(server.total_streams(), 0);
    }

    #[test]
    fn test_max_streams_limit() {
        let config = ServerConfig {
            max_streams: 2,
            ..Default::default()
        };
        let server = IngestServer::new(config).unwrap();
        server.register_stream("s1", ProtocolType::Rtmp, "1").unwrap();
        server.register_stream("s2", ProtocolType::Srt, "2").unwrap();
        assert!(server.register_stream("s3", ProtocolType::Whip, "3").is_err());
    }

    #[test]
    fn test_remove_nonexistent_stream() {
        let server = IngestServer::new(ServerConfig::default()).unwrap();
        assert!(server.remove_stream("nope").is_err());
    }
}
