//! Media relay for forwarding streams from WHIP publishers to WHEP viewers.

use crate::error::{Result, WhipError};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::broadcast;

/// A media packet that can be relayed between sessions.
#[derive(Debug, Clone)]
pub struct RelayPacket {
    /// Stream ID this packet belongs to.
    pub stream_id: String,
    /// Track identifier (e.g., "video-0", "audio-0").
    pub track_id: String,
    /// RTP payload type.
    pub payload_type: u8,
    /// Sequence number.
    pub sequence: u32,
    /// Timestamp (RTP).
    pub timestamp: u32,
    /// Is this a keyframe?
    pub keyframe: bool,
    /// Raw media data.
    pub data: Vec<u8>,
}

/// A single stream's relay channel.
#[derive(Debug)]
struct StreamRelay {
    publisher_session_id: String,
    tx: broadcast::Sender<RelayPacket>,
    viewer_count: usize,
}

/// Media relay that routes packets from WHIP publishers to WHEP viewers.
#[derive(Debug)]
pub struct MediaRelay {
    streams: RwLock<HashMap<String, StreamRelay>>,
    channel_capacity: usize,
}

impl Default for MediaRelay {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl MediaRelay {
    /// Create a new media relay with the specified channel capacity per stream.
    pub fn new(channel_capacity: usize) -> Self {
        Self {
            streams: RwLock::new(HashMap::new()),
            channel_capacity,
        }
    }

    /// Register a publisher for a stream. Returns an error if the stream already has a publisher.
    pub fn register_publisher(&self, stream_id: &str, session_id: &str) -> Result<()> {
        let mut streams = self.streams.write();
        if streams.contains_key(stream_id) {
            return Err(WhipError::SessionExists(format!(
                "Stream '{}' already has a publisher",
                stream_id
            )));
        }

        let (tx, _rx) = broadcast::channel(self.channel_capacity);
        streams.insert(
            stream_id.to_string(),
            StreamRelay {
                publisher_session_id: session_id.to_string(),
                tx,
                viewer_count: 0,
            },
        );

        tracing::info!(stream_id = %stream_id, session_id = %session_id, "Publisher registered");
        Ok(())
    }

    /// Unregister a publisher, removing the stream.
    pub fn unregister_publisher(&self, stream_id: &str) -> Result<()> {
        let mut streams = self.streams.write();
        streams.remove(stream_id).ok_or_else(|| {
            WhipError::SessionNotFound(format!("Stream '{}' not found", stream_id))
        })?;
        tracing::info!(stream_id = %stream_id, "Publisher unregistered");
        Ok(())
    }

    /// Subscribe a WHEP viewer to a stream. Returns a receiver for media packets.
    pub fn subscribe(&self, stream_id: &str) -> Result<broadcast::Receiver<RelayPacket>> {
        let mut streams = self.streams.write();
        let relay = streams.get_mut(stream_id).ok_or_else(|| {
            WhipError::SessionNotFound(format!("Stream '{}' not found", stream_id))
        })?;

        relay.viewer_count += 1;
        let rx = relay.tx.subscribe();
        tracing::info!(stream_id = %stream_id, viewers = relay.viewer_count, "Viewer subscribed");
        Ok(rx)
    }

    /// Unsubscribe a viewer from a stream.
    pub fn unsubscribe(&self, stream_id: &str) {
        let mut streams = self.streams.write();
        if let Some(relay) = streams.get_mut(stream_id) {
            relay.viewer_count = relay.viewer_count.saturating_sub(1);
            tracing::info!(stream_id = %stream_id, viewers = relay.viewer_count, "Viewer unsubscribed");
        }
    }

    /// Publish a media packet to all viewers of a stream.
    pub fn publish(&self, packet: RelayPacket) -> Result<usize> {
        let streams = self.streams.read();
        let relay = streams.get(&packet.stream_id).ok_or_else(|| {
            WhipError::SessionNotFound(format!("Stream '{}' not found", packet.stream_id))
        })?;

        match relay.tx.send(packet) {
            Ok(n) => Ok(n),
            Err(_) => Ok(0), // No active receivers
        }
    }

    /// Get the number of active streams.
    pub fn stream_count(&self) -> usize {
        self.streams.read().len()
    }

    /// Get the viewer count for a stream.
    pub fn viewer_count(&self, stream_id: &str) -> Option<usize> {
        self.streams.read().get(stream_id).map(|r| r.viewer_count)
    }

    /// List all active stream IDs.
    pub fn list_streams(&self) -> Vec<String> {
        self.streams.read().keys().cloned().collect()
    }

    /// Get relay statistics.
    pub fn stats(&self) -> RelayStats {
        let streams = self.streams.read();
        let total_viewers: usize = streams.values().map(|r| r.viewer_count).sum();
        RelayStats {
            active_streams: streams.len(),
            total_viewers,
        }
    }
}

/// Relay statistics.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct RelayStats {
    pub active_streams: usize,
    pub total_viewers: usize,
}

/// Handle for sharing the media relay across async tasks.
pub type MediaRelayHandle = Arc<MediaRelay>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_publisher() {
        let relay = MediaRelay::new(100);
        relay.register_publisher("stream-1", "session-abc").unwrap();
        assert_eq!(relay.stream_count(), 1);
    }

    #[test]
    fn test_duplicate_publisher_rejected() {
        let relay = MediaRelay::new(100);
        relay
            .register_publisher("stream-1", "session-abc")
            .unwrap();
        let result = relay.register_publisher("stream-1", "session-def");
        assert!(result.is_err());
    }

    #[test]
    fn test_subscribe_and_publish() {
        let relay = MediaRelay::new(100);
        relay
            .register_publisher("stream-1", "session-abc")
            .unwrap();

        let mut rx = relay.subscribe("stream-1").unwrap();
        assert_eq!(relay.viewer_count("stream-1"), Some(1));

        let packet = RelayPacket {
            stream_id: "stream-1".to_string(),
            track_id: "video-0".to_string(),
            payload_type: 96,
            sequence: 1,
            timestamp: 0,
            keyframe: true,
            data: vec![0u8; 1024],
        };

        let receivers = relay.publish(packet).unwrap();
        assert_eq!(receivers, 1);

        let received = rx.try_recv().unwrap();
        assert_eq!(received.track_id, "video-0");
        assert!(received.keyframe);
    }

    #[test]
    fn test_unregister_publisher() {
        let relay = MediaRelay::new(100);
        relay
            .register_publisher("stream-1", "session-abc")
            .unwrap();
        relay.unregister_publisher("stream-1").unwrap();
        assert_eq!(relay.stream_count(), 0);
    }

    #[test]
    fn test_subscribe_nonexistent_stream() {
        let relay = MediaRelay::new(100);
        let result = relay.subscribe("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_relay_stats() {
        let relay = MediaRelay::new(100);
        relay.register_publisher("s1", "sess-1").unwrap();
        relay.register_publisher("s2", "sess-2").unwrap();
        let _rx1 = relay.subscribe("s1").unwrap();
        let _rx2 = relay.subscribe("s1").unwrap();
        let _rx3 = relay.subscribe("s2").unwrap();

        let stats = relay.stats();
        assert_eq!(stats.active_streams, 2);
        assert_eq!(stats.total_viewers, 3);
    }

    #[test]
    fn test_list_streams() {
        let relay = MediaRelay::new(100);
        relay.register_publisher("alpha", "s1").unwrap();
        relay.register_publisher("beta", "s2").unwrap();
        let streams = relay.list_streams();
        assert_eq!(streams.len(), 2);
    }
}
