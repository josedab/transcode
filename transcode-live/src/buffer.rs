//! Packet buffering for live streaming.
//!
//! This module provides jitter buffer and packet reordering for live streams.

use std::collections::VecDeque;
use std::time::Instant;

use crate::StreamPacket;

/// Jitter buffer configuration.
#[derive(Debug, Clone)]
pub struct JitterBufferConfig {
    /// Minimum buffer depth in milliseconds
    pub min_depth_ms: u32,
    /// Maximum buffer depth in milliseconds
    pub max_depth_ms: u32,
    /// Target buffer depth in milliseconds
    pub target_depth_ms: u32,
    /// Maximum packet count
    pub max_packets: usize,
}

impl Default for JitterBufferConfig {
    fn default() -> Self {
        Self {
            min_depth_ms: 20,
            max_depth_ms: 500,
            target_depth_ms: 100,
            max_packets: 1024,
        }
    }
}

/// Entry in the jitter buffer.
#[derive(Debug)]
struct BufferEntry {
    packet: StreamPacket,
    #[allow(dead_code)]
    arrival_time: Instant,
}

/// Jitter buffer for smoothing packet arrival timing.
pub struct JitterBuffer {
    config: JitterBufferConfig,
    buffer: VecDeque<BufferEntry>,
    last_output_timestamp: Option<u64>,
    start_time: Option<Instant>,
}

impl JitterBuffer {
    /// Create a new jitter buffer with the given configuration.
    pub fn new(config: JitterBufferConfig) -> Self {
        Self {
            config,
            buffer: VecDeque::new(),
            last_output_timestamp: None,
            start_time: None,
        }
    }

    /// Push a packet into the buffer.
    pub fn push(&mut self, packet: StreamPacket) {
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        }

        // Insert in timestamp order
        let entry = BufferEntry {
            packet,
            arrival_time: Instant::now(),
        };

        let pos = self
            .buffer
            .iter()
            .position(|e| e.packet.timestamp > entry.packet.timestamp)
            .unwrap_or(self.buffer.len());

        self.buffer.insert(pos, entry);

        // Enforce max packets limit
        while self.buffer.len() > self.config.max_packets {
            self.buffer.pop_front();
        }
    }

    /// Pop a packet from the buffer if ready.
    pub fn pop(&mut self) -> Option<StreamPacket> {
        // Check if we have enough buffered data
        let current_depth = self.current_depth_ms();
        if current_depth < self.config.min_depth_ms {
            return None;
        }

        if let Some(entry) = self.buffer.pop_front() {
            self.last_output_timestamp = Some(entry.packet.timestamp);
            Some(entry.packet)
        } else {
            None
        }
    }

    /// Get the current buffer depth in milliseconds.
    pub fn current_depth_ms(&self) -> u32 {
        if self.buffer.len() < 2 {
            return 0;
        }

        let first_ts = self.buffer.front().map(|e| e.packet.timestamp).unwrap_or(0);
        let last_ts = self.buffer.back().map(|e| e.packet.timestamp).unwrap_or(0);

        (last_ts.saturating_sub(first_ts)) as u32
    }

    /// Get the number of packets in the buffer.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.last_output_timestamp = None;
        self.start_time = None;
    }
}

impl Default for JitterBuffer {
    fn default() -> Self {
        Self::new(JitterBufferConfig::default())
    }
}

/// Packet reordering buffer.
pub struct ReorderBuffer {
    max_reorder: u32,
    buffer: Vec<(u64, StreamPacket)>,
    next_expected_seq: u64,
}

impl ReorderBuffer {
    /// Create a new reorder buffer.
    pub fn new(max_reorder: u32) -> Self {
        Self {
            max_reorder,
            buffer: Vec::new(),
            next_expected_seq: 0,
        }
    }

    /// Push a packet with sequence number.
    pub fn push(&mut self, seq: u64, packet: StreamPacket) -> Vec<StreamPacket> {
        use std::cmp::Ordering;

        let mut output = Vec::new();

        match seq.cmp(&self.next_expected_seq) {
            Ordering::Equal => {
                // This is the next expected packet, output it immediately
                output.push(packet);
                self.next_expected_seq += 1;

                // Check if we can output buffered packets
                self.buffer.sort_by_key(|(s, _)| *s);
                while let Some(idx) = self
                    .buffer
                    .iter()
                    .position(|(s, _)| *s == self.next_expected_seq)
                {
                    let (_, pkt) = self.buffer.remove(idx);
                    output.push(pkt);
                    self.next_expected_seq += 1;
                }
            }
            Ordering::Greater => {
                // Future packet, buffer it
                if self.buffer.len() < self.max_reorder as usize {
                    self.buffer.push((seq, packet));
                }

                // Check if gap is too large, skip ahead
                if seq > self.next_expected_seq + self.max_reorder as u64 {
                    self.next_expected_seq = seq + 1;
                    // Output all buffered packets
                    self.buffer.sort_by_key(|(s, _)| *s);
                    for (_, pkt) in self.buffer.drain(..) {
                        output.push(pkt);
                    }
                }
            }
            Ordering::Less => {
                // Old packets (seq < next_expected) are discarded
            }
        }

        output
    }

    /// Get the next expected sequence number.
    pub fn next_expected(&self) -> u64 {
        self.next_expected_seq
    }

    /// Get buffered packet count.
    pub fn buffered_count(&self) -> usize {
        self.buffer.len()
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.next_expected_seq = 0;
    }
}

impl Default for ReorderBuffer {
    fn default() -> Self {
        Self::new(16)
    }
}

/// Statistics for buffer monitoring.
#[derive(Debug, Clone, Default)]
pub struct BufferStats {
    /// Packets received
    pub packets_received: u64,
    /// Packets dropped due to buffer overflow
    pub packets_dropped: u64,
    /// Packets reordered
    pub packets_reordered: u64,
    /// Late packets (arrived after output deadline)
    pub packets_late: u64,
    /// Current buffer depth in milliseconds
    pub current_depth_ms: u32,
    /// Average buffer depth in milliseconds
    pub avg_depth_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PacketType;

    fn make_packet(timestamp: u64) -> StreamPacket {
        StreamPacket {
            packet_type: PacketType::Video,
            timestamp,
            data: vec![0u8; 100],
            is_keyframe: false,
        }
    }

    #[test]
    fn test_jitter_buffer_ordering() {
        let mut buffer = JitterBuffer::new(JitterBufferConfig {
            min_depth_ms: 0,
            ..Default::default()
        });

        // Insert out of order
        buffer.push(make_packet(300));
        buffer.push(make_packet(100));
        buffer.push(make_packet(200));

        // Should come out in order
        assert_eq!(buffer.pop().map(|p| p.timestamp), Some(100));
        assert_eq!(buffer.pop().map(|p| p.timestamp), Some(200));
        assert_eq!(buffer.pop().map(|p| p.timestamp), Some(300));
    }

    #[test]
    fn test_reorder_buffer() {
        let mut buffer = ReorderBuffer::new(8);

        // Send packets out of order
        let out1 = buffer.push(0, make_packet(0));
        assert_eq!(out1.len(), 1);

        let out2 = buffer.push(2, make_packet(200));
        assert_eq!(out2.len(), 0); // Waiting for seq 1

        let out3 = buffer.push(1, make_packet(100));
        assert_eq!(out3.len(), 2); // Both 1 and 2 output
    }

    #[test]
    fn test_buffer_depth() {
        let mut buffer = JitterBuffer::default();

        buffer.push(make_packet(0));
        buffer.push(make_packet(100));
        buffer.push(make_packet(200));

        assert_eq!(buffer.current_depth_ms(), 200);
    }
}
