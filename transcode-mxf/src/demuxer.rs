//! MXF demuxer

use crate::error::{MxfError, Result};
use crate::klv::KlvReader;
use crate::metadata::ContentPackage;
use crate::partition::{Partition, PartitionKind, RandomIndexPack};
use crate::types::{EditRate, EssenceCoding, Rational, TrackKind};
use crate::ul::{labels, UniversalLabel};

/// Track information
#[derive(Debug, Clone)]
pub struct TrackInfo {
    /// Track index
    pub index: u32,
    /// Track ID from MXF
    pub track_id: u32,
    /// Track kind
    pub kind: TrackKind,
    /// Edit rate
    pub edit_rate: EditRate,
    /// Duration in edit units
    pub duration: u64,
    /// Essence coding
    pub coding: EssenceCoding,
    /// Track name
    pub name: Option<String>,
}

/// MXF packet
#[derive(Debug, Clone)]
pub struct MxfPacket {
    /// Track index
    pub track_index: u32,
    /// Position in edit units
    pub position: u64,
    /// Duration in edit units
    pub duration: u32,
    /// Packet data
    pub data: Vec<u8>,
    /// Is random access point (keyframe)
    pub is_random_access: bool,
}

/// MXF demuxer
pub struct MxfDemuxer<'a> {
    /// Source data
    data: &'a [u8],
    /// KLV reader
    reader: KlvReader<'a>,
    /// Partitions
    partitions: Vec<Partition>,
    /// Track information
    tracks: Vec<TrackInfo>,
    /// Content packages (material/source)
    #[allow(dead_code)]
    packages: Vec<ContentPackage>,
    /// Current read position
    current_position: u64,
    /// Essence start offset
    essence_offset: u64,
    /// Random index pack
    rip: Option<RandomIndexPack>,
}

impl<'a> MxfDemuxer<'a> {
    /// Create new demuxer from data
    pub fn new(data: &'a [u8]) -> Result<Self> {
        let mut demuxer = MxfDemuxer {
            data,
            reader: KlvReader::new(data),
            partitions: Vec::new(),
            tracks: Vec::new(),
            packages: Vec::new(),
            current_position: 0,
            essence_offset: 0,
            rip: None,
        };

        demuxer.parse()?;
        Ok(demuxer)
    }

    /// Parse MXF structure
    fn parse(&mut self) -> Result<()> {
        // Check for partition pack at start
        if self.data.len() < 16 {
            return Err(MxfError::InsufficientData {
                needed: 16,
                available: self.data.len(),
            });
        }

        let first_key = UniversalLabel::new([
            self.data[0],
            self.data[1],
            self.data[2],
            self.data[3],
            self.data[4],
            self.data[5],
            self.data[6],
            self.data[7],
            self.data[8],
            self.data[9],
            self.data[10],
            self.data[11],
            self.data[12],
            self.data[13],
            self.data[14],
            self.data[15],
        ]);

        if !first_key.is_partition_pack() {
            return Err(MxfError::InvalidMxf(
                "File does not start with partition pack".into(),
            ));
        }

        // Try to read RIP from end of file
        self.try_read_rip();

        // Parse all partitions
        self.parse_partitions()?;

        // Build track information from metadata
        self.build_track_info();

        Ok(())
    }

    /// Try to read Random Index Pack from end of file
    fn try_read_rip(&mut self) {
        if self.data.len() < 20 {
            return;
        }

        // RIP ends with overall length (4 bytes)
        let last_4 = &self.data[self.data.len() - 4..];
        let overall_length = u32::from_be_bytes([last_4[0], last_4[1], last_4[2], last_4[3]]);

        if overall_length as usize > self.data.len() || overall_length < 20 {
            return;
        }

        let rip_start = self.data.len() - overall_length as usize;

        // Check for RIP key
        if self.data.len() >= rip_start + 16 {
            let key = UniversalLabel::new([
                self.data[rip_start],
                self.data[rip_start + 1],
                self.data[rip_start + 2],
                self.data[rip_start + 3],
                self.data[rip_start + 4],
                self.data[rip_start + 5],
                self.data[rip_start + 6],
                self.data[rip_start + 7],
                self.data[rip_start + 8],
                self.data[rip_start + 9],
                self.data[rip_start + 10],
                self.data[rip_start + 11],
                self.data[rip_start + 12],
                self.data[rip_start + 13],
                self.data[rip_start + 14],
                self.data[rip_start + 15],
            ]);

            if key.as_bytes() == &labels::RANDOM_INDEX_PACK {
                if let Ok(rip) = RandomIndexPack::parse(&self.data[rip_start..]) {
                    self.rip = Some(rip);
                    log::debug!("Found RIP with {} entries", self.rip.as_ref().unwrap().entries.len());
                }
            }
        }
    }

    /// Parse all partitions
    fn parse_partitions(&mut self) -> Result<()> {
        let mut offset = 0;

        while offset + 16 <= self.data.len() {
            let key = UniversalLabel::new([
                self.data[offset],
                self.data[offset + 1],
                self.data[offset + 2],
                self.data[offset + 3],
                self.data[offset + 4],
                self.data[offset + 5],
                self.data[offset + 6],
                self.data[offset + 7],
                self.data[offset + 8],
                self.data[offset + 9],
                self.data[offset + 10],
                self.data[offset + 11],
                self.data[offset + 12],
                self.data[offset + 13],
                self.data[offset + 14],
                self.data[offset + 15],
            ]);

            if key.is_partition_pack() {
                match Partition::parse(&self.data[offset..]) {
                    Ok(partition) => {
                        log::debug!(
                            "Found {:?} partition at offset {}",
                            partition.kind,
                            offset
                        );

                        // Track essence start
                        if partition.body_sid > 0 && self.essence_offset == 0 {
                            // Essence starts after this partition's metadata
                            self.essence_offset = offset as u64 + partition.size() as u64;
                        }

                        self.partitions.push(partition);
                    }
                    Err(e) => {
                        log::warn!("Failed to parse partition at {}: {}", offset, e);
                    }
                }
            }

            // Move to next KLV
            self.reader.seek(offset);
            match self.reader.skip_klv() {
                Ok(Some(_)) => {
                    offset = self.reader.position();
                }
                _ => break,
            }
        }

        if self.partitions.is_empty() {
            return Err(MxfError::InvalidMxf("No partitions found".into()));
        }

        Ok(())
    }

    /// Build track information from parsed metadata
    fn build_track_info(&mut self) {
        // Create synthetic track info based on partition essence containers
        let mut track_index = 0u32;

        for partition in &self.partitions {
            for essence_ul in &partition.essence_containers {
                let coding = self.detect_essence_coding(essence_ul);
                let kind = match coding {
                    EssenceCoding::Uncompressed
                    | EssenceCoding::Mpeg2
                    | EssenceCoding::Avc
                    | EssenceCoding::Hevc
                    | EssenceCoding::Jpeg2000
                    | EssenceCoding::ProRes
                    | EssenceCoding::DnxHd => TrackKind::Video,
                    EssenceCoding::Pcm => TrackKind::Audio,
                    EssenceCoding::Unknown => TrackKind::Unknown,
                };

                // Avoid duplicates
                if !self.tracks.iter().any(|t| t.coding == coding && t.kind == kind) {
                    self.tracks.push(TrackInfo {
                        index: track_index,
                        track_id: track_index + 1,
                        kind,
                        edit_rate: Rational::fps_25(),
                        duration: 0,
                        coding,
                        name: None,
                    });
                    track_index += 1;
                }
            }
        }

        // If no tracks found from containers, add defaults
        if self.tracks.is_empty() && !self.partitions.is_empty() {
            self.tracks.push(TrackInfo {
                index: 0,
                track_id: 1,
                kind: TrackKind::Video,
                edit_rate: Rational::fps_25(),
                duration: 0,
                coding: EssenceCoding::Unknown,
                name: None,
            });
        }
    }

    /// Detect essence coding from UL
    fn detect_essence_coding(&self, ul: &[u8; 16]) -> EssenceCoding {
        // Check bytes 12-14 for essence type
        if ul[0..8] == labels::ESSENCE_MPEG2[0..8] {
            return EssenceCoding::Mpeg2;
        }
        if ul[0..8] == labels::ESSENCE_AVC[0..8] {
            return EssenceCoding::Avc;
        }
        if ul[0..8] == labels::ESSENCE_JPEG2000[0..8] {
            return EssenceCoding::Jpeg2000;
        }
        if ul[0..8] == labels::ESSENCE_UNCOMPRESSED[0..8] {
            return EssenceCoding::Uncompressed;
        }
        if ul[0..8] == labels::ESSENCE_PCM[0..8] {
            return EssenceCoding::Pcm;
        }

        EssenceCoding::Unknown
    }

    /// Get track count
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Get track info
    pub fn track(&self, index: usize) -> Option<&TrackInfo> {
        self.tracks.get(index)
    }

    /// Get all tracks
    pub fn tracks(&self) -> &[TrackInfo] {
        &self.tracks
    }

    /// Get video track
    pub fn video_track(&self) -> Option<&TrackInfo> {
        self.tracks.iter().find(|t| t.kind == TrackKind::Video)
    }

    /// Get audio tracks
    pub fn audio_tracks(&self) -> Vec<&TrackInfo> {
        self.tracks.iter().filter(|t| t.kind == TrackKind::Audio).collect()
    }

    /// Get partitions
    pub fn partitions(&self) -> &[Partition] {
        &self.partitions
    }

    /// Get header partition
    pub fn header_partition(&self) -> Option<&Partition> {
        self.partitions.iter().find(|p| p.kind == PartitionKind::Header)
    }

    /// Get footer partition
    pub fn footer_partition(&self) -> Option<&Partition> {
        self.partitions.iter().find(|p| p.kind == PartitionKind::Footer)
    }

    /// Read next packet
    pub fn read_packet(&mut self) -> Result<Option<MxfPacket>> {
        // Start from essence offset
        if self.current_position < self.essence_offset {
            self.current_position = self.essence_offset;
        }

        self.reader.seek(self.current_position as usize);

        loop {
            match self.reader.read_klv() {
                Ok(Some(klv)) => {
                    self.current_position = self.reader.position() as u64;

                    // Check if this is essence data
                    if klv.key.is_essence() {
                        // Determine track from essence element key
                        let track_index = self.essence_key_to_track(&klv.key);

                        return Ok(Some(MxfPacket {
                            track_index,
                            position: 0, // Would need index table for accurate position
                            duration: 1,
                            data: klv.value,
                            is_random_access: true, // Would need index for accurate info
                        }));
                    }

                    // Skip non-essence KLVs (metadata, fill, etc.)
                    if klv.key.is_partition_pack() {
                        // Hit another partition, might be footer
                        continue;
                    }

                    if klv.key.is_fill_item() {
                        continue;
                    }

                    // Skip other metadata
                    if klv.key.is_metadata() {
                        continue;
                    }
                }
                Ok(None) => return Ok(None),
                Err(e) => return Err(e),
            }
        }
    }

    /// Map essence element key to track index
    fn essence_key_to_track(&self, key: &UniversalLabel) -> u32 {
        // Essence element keys have track info in bytes 13-15
        // For now, use a simple heuristic
        let bytes = key.as_bytes();

        // Check if video or audio based on category byte
        let is_audio = bytes[12] >= 0x10 && bytes[12] < 0x20;

        if is_audio {
            // Find first audio track
            self.tracks
                .iter()
                .find(|t| t.kind == TrackKind::Audio)
                .map(|t| t.index)
                .unwrap_or(0)
        } else {
            // Assume video (default to first track)
            0
        }
    }

    /// Seek to position
    pub fn seek(&mut self, _track_index: u32, _position: u64) -> Result<()> {
        // Without proper index table parsing, seek is limited
        // For now, reset to essence start
        self.current_position = self.essence_offset;
        Ok(())
    }

    /// Reset to beginning
    pub fn reset(&mut self) {
        self.current_position = self.essence_offset;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_minimal_mxf() -> Vec<u8> {
        let mut data = Vec::new();

        // Header partition pack
        let mut partition = Partition::header();
        partition.add_essence_container(labels::ESSENCE_UNCOMPRESSED);
        partition.write(&mut data).unwrap();

        // Add fill to make it look valid
        data.extend_from_slice(&labels::FILL_ITEM);
        data.push(4); // BER length
        data.extend_from_slice(&[0, 0, 0, 0]); // Fill data

        data
    }

    #[test]
    fn test_demuxer_parse() {
        let data = create_minimal_mxf();
        let demuxer = MxfDemuxer::new(&data).unwrap();

        assert!(!demuxer.partitions().is_empty());
        assert!(demuxer.header_partition().is_some());
    }

    #[test]
    fn test_demuxer_tracks() {
        let data = create_minimal_mxf();
        let demuxer = MxfDemuxer::new(&data).unwrap();

        // Should have detected uncompressed video
        assert!(demuxer.track_count() > 0);
    }

    #[test]
    fn test_invalid_mxf() {
        let data = b"NOT_AN_MXF_FILE";
        let result = MxfDemuxer::new(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_demuxer_reset() {
        let data = create_minimal_mxf();
        let mut demuxer = MxfDemuxer::new(&data).unwrap();

        demuxer.current_position = 9999;
        demuxer.reset();

        assert_eq!(demuxer.current_position, demuxer.essence_offset);
    }
}
