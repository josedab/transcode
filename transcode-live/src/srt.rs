//! SRT protocol support

use crate::{LiveError, Result};
use std::time::Duration;

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
