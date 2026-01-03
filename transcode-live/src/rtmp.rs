//! RTMP protocol support

use crate::{LiveError, Result, StreamMetadata};

/// Minimum chunk size per RTMP spec
pub const RTMP_MIN_CHUNK_SIZE: u32 = 128;
/// Maximum chunk size (64KB reasonable limit)
pub const RTMP_MAX_CHUNK_SIZE: u32 = 65536;
/// Default chunk size
pub const RTMP_DEFAULT_CHUNK_SIZE: u32 = 128;
/// Maximum window size (10MB reasonable limit)
pub const RTMP_MAX_WINDOW_SIZE: u32 = 10_000_000;

/// RTMP handshake state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtmpHandshake {
    /// Waiting for C0/C1
    WaitingC0C1,
    /// Waiting for C2
    WaitingC2,
    /// Handshake complete
    Complete,
}

/// RTMP message type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtmpMessageType {
    /// Set chunk size
    SetChunkSize,
    /// Abort message
    Abort,
    /// Acknowledgement
    Acknowledgement,
    /// Window acknowledgement size
    WindowAckSize,
    /// Set peer bandwidth
    SetPeerBandwidth,
    /// Audio data
    Audio,
    /// Video data
    Video,
    /// AMF3 command
    Amf3Command,
    /// AMF0 command
    Amf0Command,
    /// AMF0 data
    Amf0Data,
}

/// RTMP chunk stream
pub struct RtmpChunkStream {
    chunk_size: u32,
    window_size: u32,
    _bytes_received: u64,
}

impl Default for RtmpChunkStream {
    fn default() -> Self {
        Self {
            chunk_size: 128,
            window_size: 2500000,
            _bytes_received: 0,
        }
    }
}

impl RtmpChunkStream {
    /// Create new chunk stream
    pub fn new() -> Self {
        Self::default()
    }

    /// Set chunk size with validation
    ///
    /// # Errors
    ///
    /// Returns an error if the chunk size is outside the valid range
    /// `[RTMP_MIN_CHUNK_SIZE, RTMP_MAX_CHUNK_SIZE]`.
    pub fn set_chunk_size(&mut self, size: u32) -> Result<()> {
        if !(RTMP_MIN_CHUNK_SIZE..=RTMP_MAX_CHUNK_SIZE).contains(&size) {
            return Err(LiveError::Configuration(format!(
                "chunk size {} out of valid range [{}, {}]",
                size, RTMP_MIN_CHUNK_SIZE, RTMP_MAX_CHUNK_SIZE
            )));
        }
        self.chunk_size = size;
        Ok(())
    }

    /// Get chunk size
    pub fn chunk_size(&self) -> u32 {
        self.chunk_size
    }

    /// Set window size with validation
    ///
    /// # Errors
    ///
    /// Returns an error if the window size exceeds `RTMP_MAX_WINDOW_SIZE`.
    pub fn set_window_size(&mut self, size: u32) -> Result<()> {
        if size > RTMP_MAX_WINDOW_SIZE {
            return Err(LiveError::Configuration(format!(
                "window size {} exceeds maximum {}",
                size, RTMP_MAX_WINDOW_SIZE
            )));
        }
        self.window_size = size;
        Ok(())
    }

    /// Get window size
    pub fn window_size(&self) -> u32 {
        self.window_size
    }
}

/// RTMP session
pub struct RtmpSession {
    handshake: RtmpHandshake,
    _chunk_stream: RtmpChunkStream,
    metadata: Option<StreamMetadata>,
}

impl RtmpSession {
    /// Create new RTMP session
    pub fn new() -> Self {
        Self {
            handshake: RtmpHandshake::WaitingC0C1,
            _chunk_stream: RtmpChunkStream::new(),
            metadata: None,
        }
    }

    /// Process handshake data
    pub fn process_handshake(&mut self, data: &[u8]) -> Result<Vec<u8>> {
        match self.handshake {
            RtmpHandshake::WaitingC0C1 => {
                if data.len() < 1537 {
                    return Err(LiveError::Protocol("incomplete C0C1".into()));
                }

                // Generate S0S1S2 response
                let mut response = vec![0u8; 3073];
                response[0] = 3; // RTMP version

                self.handshake = RtmpHandshake::WaitingC2;
                Ok(response)
            }
            RtmpHandshake::WaitingC2 => {
                if data.len() < 1536 {
                    return Err(LiveError::Protocol("incomplete C2".into()));
                }

                self.handshake = RtmpHandshake::Complete;
                Ok(Vec::new())
            }
            RtmpHandshake::Complete => Ok(Vec::new()),
        }
    }

    /// Check if handshake is complete
    pub fn is_handshake_complete(&self) -> bool {
        self.handshake == RtmpHandshake::Complete
    }

    /// Get stream metadata
    pub fn metadata(&self) -> Option<&StreamMetadata> {
        self.metadata.as_ref()
    }
}

impl Default for RtmpSession {
    fn default() -> Self {
        Self::new()
    }
}
