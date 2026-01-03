//! Packet abstractions for encoded media data.
//!
//! Packets contain compressed/encoded data before decoding or after encoding.

use crate::timestamp::{Duration, TimeBase, Timestamp};
use bitflags::bitflags;
use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;

bitflags! {
    /// Flags for packet properties.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct PacketFlags: u32 {
        /// This packet contains a keyframe.
        const KEYFRAME = 0x0001;
        /// Packet data is corrupted.
        const CORRUPT = 0x0002;
        /// Packet should be discarded.
        const DISCARD = 0x0004;
        /// Packet contains a disposable frame (can be dropped).
        const DISPOSABLE = 0x0008;
    }
}

/// An encoded media packet.
///
/// Packets can own their data or reference external data (zero-copy).
#[derive(Clone)]
pub struct Packet<'a> {
    /// The packet data.
    data: Cow<'a, [u8]>,
    /// Presentation timestamp.
    pub pts: Timestamp,
    /// Decode timestamp.
    pub dts: Timestamp,
    /// Duration of the packet.
    pub duration: Duration,
    /// Stream index this packet belongs to.
    pub stream_index: u32,
    /// Packet flags.
    pub flags: PacketFlags,
    /// Position in the input stream (bytes).
    pub pos: Option<u64>,
    /// Codec-specific side data.
    side_data: Vec<SideData>,
}

impl<'a> Packet<'a> {
    /// Create a new packet with owned data.
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data: Cow::Owned(data),
            pts: Timestamp::none(),
            dts: Timestamp::none(),
            duration: Duration::zero(),
            stream_index: 0,
            flags: PacketFlags::empty(),
            pos: None,
            side_data: Vec::new(),
        }
    }

    /// Create a new packet referencing external data.
    pub fn from_slice(data: &'a [u8]) -> Self {
        Self {
            data: Cow::Borrowed(data),
            pts: Timestamp::none(),
            dts: Timestamp::none(),
            duration: Duration::zero(),
            stream_index: 0,
            flags: PacketFlags::empty(),
            pos: None,
            side_data: Vec::new(),
        }
    }

    /// Create an empty packet.
    pub fn empty() -> Self {
        Self::new(Vec::new())
    }

    /// Get the packet data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get the size of the packet data.
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Check if this packet is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Check if this is a keyframe packet.
    pub fn is_keyframe(&self) -> bool {
        self.flags.contains(PacketFlags::KEYFRAME)
    }

    /// Set the keyframe flag.
    pub fn set_keyframe(&mut self, keyframe: bool) {
        if keyframe {
            self.flags.insert(PacketFlags::KEYFRAME);
        } else {
            self.flags.remove(PacketFlags::KEYFRAME);
        }
    }

    /// Make the packet own its data.
    pub fn into_owned(self) -> Packet<'static> {
        Packet {
            data: Cow::Owned(self.data.into_owned()),
            pts: self.pts,
            dts: self.dts,
            duration: self.duration,
            stream_index: self.stream_index,
            flags: self.flags,
            pos: self.pos,
            side_data: self.side_data,
        }
    }

    /// Add side data to the packet.
    pub fn add_side_data(&mut self, data: SideData) {
        self.side_data.push(data);
    }

    /// Get side data of a specific type.
    pub fn get_side_data(&self, data_type: SideDataType) -> Option<&SideData> {
        self.side_data.iter().find(|sd| sd.data_type == data_type)
    }

    /// Rescale timestamps to a new time base.
    pub fn rescale(&mut self, target: TimeBase) {
        self.pts = self.pts.rescale(target);
        self.dts = self.dts.rescale(target);
        self.duration = self.duration.rescale(target);
    }

    /// Create a new packet with the specified timestamps.
    pub fn with_timestamps(mut self, pts: Timestamp, dts: Timestamp) -> Self {
        self.pts = pts;
        self.dts = dts;
        self
    }

    /// Create a new packet with the specified stream index.
    pub fn with_stream_index(mut self, index: u32) -> Self {
        self.stream_index = index;
        self
    }

    /// Create a new packet with the specified flags.
    pub fn with_flags(mut self, flags: PacketFlags) -> Self {
        self.flags = flags;
        self
    }
}

impl<'a> fmt::Debug for Packet<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Packet")
            .field("size", &self.size())
            .field("pts", &self.pts)
            .field("dts", &self.dts)
            .field("stream_index", &self.stream_index)
            .field("flags", &self.flags)
            .finish()
    }
}

impl<'a> Default for Packet<'a> {
    fn default() -> Self {
        Self::empty()
    }
}

/// Types of side data that can be attached to packets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SideDataType {
    /// H.264/H.265 parameter sets (SPS, PPS).
    ParameterSets,
    /// Display matrix (rotation/flip).
    DisplayMatrix,
    /// Content light level (HDR).
    ContentLightLevel,
    /// Mastering display metadata (HDR).
    MasteringDisplayMetadata,
    /// Encryption info.
    EncryptionInfo,
    /// Skip samples (for gapless playback).
    SkipSamples,
    /// A53 closed captions.
    A53ClosedCaptions,
    /// Custom/unknown.
    Custom(u32),
}

/// Side data attached to a packet.
#[derive(Debug, Clone)]
pub struct SideData {
    /// Type of side data.
    pub data_type: SideDataType,
    /// The side data payload.
    pub data: Vec<u8>,
}

impl SideData {
    /// Create new side data.
    pub fn new(data_type: SideDataType, data: Vec<u8>) -> Self {
        Self { data_type, data }
    }
}

/// An owned packet suitable for storage and async operations.
pub type OwnedPacket = Packet<'static>;

/// A reference-counted packet for efficient sharing.
pub type SharedPacket = Arc<OwnedPacket>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_creation() {
        let data = vec![0u8; 100];
        let packet = Packet::new(data);
        assert_eq!(packet.size(), 100);
        assert!(!packet.is_empty());
    }

    #[test]
    fn test_packet_from_slice() {
        let data = [1u8, 2, 3, 4, 5];
        let packet = Packet::from_slice(&data);
        assert_eq!(packet.data(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_packet_keyframe() {
        let mut packet = Packet::empty();
        assert!(!packet.is_keyframe());
        packet.set_keyframe(true);
        assert!(packet.is_keyframe());
    }

    #[test]
    fn test_packet_into_owned() {
        let data = [1u8, 2, 3];
        let packet = Packet::from_slice(&data);
        let owned: Packet<'static> = packet.into_owned();
        assert_eq!(owned.data(), &[1, 2, 3]);
    }

    #[test]
    fn test_side_data() {
        let mut packet = Packet::empty();
        packet.add_side_data(SideData::new(SideDataType::ParameterSets, vec![1, 2, 3]));
        assert!(packet.get_side_data(SideDataType::ParameterSets).is_some());
        assert!(packet.get_side_data(SideDataType::DisplayMatrix).is_none());
    }
}
