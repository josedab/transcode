//! MXF muxer

use crate::error::Result;
use crate::klv::{encode_ber_length, Klv};
use crate::metadata::{ContentPackage, PrimerPack, Track};
use crate::partition::{Partition, PartitionCompleteness, PartitionKind, PartitionStatus, RandomIndexPack};
use crate::types::{EditRate, EssenceCoding, FrameSize, Rational, TrackKind, Umid};
use crate::ul::labels;
use byteorder::{BigEndian, WriteBytesExt};
use std::io::{Seek, SeekFrom, Write};

/// Track configuration for muxer
#[derive(Debug, Clone)]
pub struct TrackConfig {
    /// Track kind
    pub kind: TrackKind,
    /// Edit rate
    pub edit_rate: EditRate,
    /// Essence coding
    pub coding: EssenceCoding,
    /// Video frame size (for video tracks)
    pub frame_size: Option<FrameSize>,
    /// Audio sample rate (for audio tracks)
    pub sample_rate: Option<u32>,
    /// Audio channels (for audio tracks)
    pub channels: Option<u32>,
    /// Bits per sample (for audio tracks)
    pub bits_per_sample: Option<u32>,
}

impl TrackConfig {
    /// Create video track config
    pub fn video(width: u32, height: u32, fps: f64, coding: EssenceCoding) -> Self {
        let (num, den) = if (fps - 29.97).abs() < 0.01 {
            (30000, 1001)
        } else if (fps - 23.976).abs() < 0.01 {
            (24000, 1001)
        } else if (fps - 59.94).abs() < 0.01 {
            (60000, 1001)
        } else {
            (fps.round() as i32, 1)
        };

        TrackConfig {
            kind: TrackKind::Video,
            edit_rate: Rational::new(num, den),
            coding,
            frame_size: Some(FrameSize::new(width, height)),
            sample_rate: None,
            channels: None,
            bits_per_sample: None,
        }
    }

    /// Create audio track config
    pub fn audio(sample_rate: u32, channels: u32, bits: u32) -> Self {
        TrackConfig {
            kind: TrackKind::Audio,
            edit_rate: Rational::new(sample_rate as i32, 1),
            coding: EssenceCoding::Pcm,
            frame_size: None,
            sample_rate: Some(sample_rate),
            channels: Some(channels),
            bits_per_sample: Some(bits),
        }
    }
}

/// Muxer configuration
#[derive(Debug, Clone)]
pub struct MuxerConfig {
    /// Operational pattern (Op1a is default)
    pub operational_pattern: [u8; 16],
    /// KAG size (key alignment grid)
    pub kag_size: u32,
    /// Include index table
    pub write_index: bool,
}

impl Default for MuxerConfig {
    fn default() -> Self {
        MuxerConfig {
            operational_pattern: crate::partition::OP1A_UL,
            kag_size: 1,
            write_index: false,
        }
    }
}

/// Stream state
struct StreamState {
    config: TrackConfig,
    track_id: u32,
    frame_count: u64,
    total_bytes: u64,
}

/// MXF muxer
pub struct MxfMuxer<W: Write + Seek> {
    writer: W,
    config: MuxerConfig,
    streams: Vec<StreamState>,
    header_written: bool,
    finalized: bool,
    header_partition_offset: u64,
    body_partition_offset: u64,
    essence_start_offset: u64,
    package_uid: Umid,
}

impl<W: Write + Seek> MxfMuxer<W> {
    /// Create new muxer
    pub fn new(writer: W, config: MuxerConfig) -> Self {
        MxfMuxer {
            writer,
            config,
            streams: Vec::new(),
            header_written: false,
            finalized: false,
            header_partition_offset: 0,
            body_partition_offset: 0,
            essence_start_offset: 0,
            package_uid: Umid::generate(),
        }
    }

    /// Add a stream
    pub fn add_stream(&mut self, config: TrackConfig) -> Result<u32> {
        if self.header_written {
            log::warn!("Cannot add stream after header is written");
            return Ok(self.streams.len() as u32);
        }

        let track_id = (self.streams.len() + 1) as u32;
        self.streams.push(StreamState {
            config,
            track_id,
            frame_count: 0,
            total_bytes: 0,
        });

        Ok(track_id - 1)
    }

    /// Write header
    pub fn write_header(&mut self) -> Result<()> {
        if self.header_written {
            return Ok(());
        }

        self.header_partition_offset = self.writer.stream_position()?;

        // Create header partition
        let mut header = Partition::header();
        header.status = PartitionStatus::Open;
        header.completeness = PartitionCompleteness::Incomplete;
        header.this_partition = self.header_partition_offset;
        header.operational_pattern = self.config.operational_pattern;
        header.kag_size = self.config.kag_size;

        // Add essence containers based on tracks
        for stream in &self.streams {
            let essence_ul = self.coding_to_essence_ul(stream.config.coding);
            header.add_essence_container(essence_ul);
        }

        header.write(&mut self.writer)?;

        // Write primer pack
        self.write_primer_pack()?;

        // Write preface and other metadata
        self.write_header_metadata()?;

        // Record body partition offset
        self.body_partition_offset = self.writer.stream_position()?;

        // Write body partition (for essence)
        let mut body = Partition::body();
        body.this_partition = self.body_partition_offset;
        body.previous_partition = self.header_partition_offset;
        body.body_sid = 1;
        body.operational_pattern = self.config.operational_pattern;

        for stream in &self.streams {
            let essence_ul = self.coding_to_essence_ul(stream.config.coding);
            body.add_essence_container(essence_ul);
        }

        body.write(&mut self.writer)?;

        self.essence_start_offset = self.writer.stream_position()?;
        self.header_written = true;

        log::debug!("Header written, essence starts at {}", self.essence_start_offset);

        Ok(())
    }

    /// Write primer pack
    fn write_primer_pack(&mut self) -> Result<()> {
        let primer = PrimerPack::new();
        let mut value = Vec::new();

        // Write batch header
        value.write_u32::<BigEndian>(primer.mappings.len() as u32)?;
        value.write_u32::<BigEndian>(18)?; // Item size: 2 byte tag + 16 byte UL

        // Write mappings
        for (tag, ul) in &primer.mappings {
            value.write_u16::<BigEndian>(*tag)?;
            value.extend_from_slice(ul);
        }

        let klv = Klv::new(labels::PRIMER_PACK, value);
        klv.write(&mut self.writer)?;

        Ok(())
    }

    /// Write header metadata
    fn write_header_metadata(&mut self) -> Result<()> {
        // Write minimal preface
        self.write_preface()?;

        // Write content storage
        self.write_content_storage()?;

        // Write material package
        self.write_material_package()?;

        // Write source package
        self.write_source_package()?;

        Ok(())
    }

    /// Write preface
    fn write_preface(&mut self) -> Result<()> {
        let mut value = Vec::new();

        // Instance UID (tag 0x3C0A)
        value.write_u16::<BigEndian>(0x3C0A)?;
        value.write_u16::<BigEndian>(16)?;
        let instance_uid = uuid::Uuid::new_v4();
        value.extend_from_slice(instance_uid.as_bytes());

        // Generation UID (tag 0x0102)
        value.write_u16::<BigEndian>(0x0102)?;
        value.write_u16::<BigEndian>(16)?;
        let gen_uid = uuid::Uuid::new_v4();
        value.extend_from_slice(gen_uid.as_bytes());

        // Last Modified Date (tag 0x3B02)
        value.write_u16::<BigEndian>(0x3B02)?;
        value.write_u16::<BigEndian>(8)?;
        value.extend_from_slice(&[0x07, 0xE8, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00]);

        // Version (tag 0x3B05)
        value.write_u16::<BigEndian>(0x3B05)?;
        value.write_u16::<BigEndian>(2)?;
        value.write_u16::<BigEndian>(0x0103)?; // MXF 1.3

        // Operational Pattern (tag 0x3B09)
        value.write_u16::<BigEndian>(0x3B09)?;
        value.write_u16::<BigEndian>(16)?;
        value.extend_from_slice(&self.config.operational_pattern);

        let klv = Klv::new(labels::PREFACE, value);
        klv.write(&mut self.writer)?;

        Ok(())
    }

    /// Write content storage
    fn write_content_storage(&mut self) -> Result<()> {
        let mut value = Vec::new();

        // Instance UID
        value.write_u16::<BigEndian>(0x3C0A)?;
        value.write_u16::<BigEndian>(16)?;
        let instance_uid = uuid::Uuid::new_v4();
        value.extend_from_slice(instance_uid.as_bytes());

        let klv = Klv::new(labels::CONTENT_STORAGE, value);
        klv.write(&mut self.writer)?;

        Ok(())
    }

    /// Write material package
    fn write_material_package(&mut self) -> Result<()> {
        let mut value = Vec::new();

        // Instance UID
        value.write_u16::<BigEndian>(0x3C0A)?;
        value.write_u16::<BigEndian>(16)?;
        let instance_uid = uuid::Uuid::new_v4();
        value.extend_from_slice(instance_uid.as_bytes());

        // Package UID (UMID)
        value.write_u16::<BigEndian>(0x4401)?;
        value.write_u16::<BigEndian>(32)?;
        value.extend_from_slice(&self.package_uid.0);

        let klv = Klv::new(labels::MATERIAL_PACKAGE, value);
        klv.write(&mut self.writer)?;

        Ok(())
    }

    /// Write source package
    fn write_source_package(&mut self) -> Result<()> {
        let mut value = Vec::new();

        // Instance UID
        value.write_u16::<BigEndian>(0x3C0A)?;
        value.write_u16::<BigEndian>(16)?;
        let instance_uid = uuid::Uuid::new_v4();
        value.extend_from_slice(instance_uid.as_bytes());

        // Package UID (UMID) - different from material
        value.write_u16::<BigEndian>(0x4401)?;
        value.write_u16::<BigEndian>(32)?;
        let source_uid = Umid::generate();
        value.extend_from_slice(&source_uid.0);

        let klv = Klv::new(labels::SOURCE_PACKAGE, value);
        klv.write(&mut self.writer)?;

        Ok(())
    }

    /// Map essence coding to UL
    fn coding_to_essence_ul(&self, coding: EssenceCoding) -> [u8; 16] {
        match coding {
            EssenceCoding::Uncompressed => labels::ESSENCE_UNCOMPRESSED,
            EssenceCoding::Mpeg2 => labels::ESSENCE_MPEG2,
            EssenceCoding::Avc => labels::ESSENCE_AVC,
            EssenceCoding::Jpeg2000 => labels::ESSENCE_JPEG2000,
            EssenceCoding::Pcm => labels::ESSENCE_PCM,
            _ => labels::ESSENCE_UNCOMPRESSED,
        }
    }

    /// Write a packet
    pub fn write_packet(
        &mut self,
        track_index: u32,
        data: &[u8],
        _is_key_frame: bool,
    ) -> Result<()> {
        if !self.header_written {
            self.write_header()?;
        }

        if self.finalized {
            return Ok(());
        }

        let stream_idx = track_index as usize;
        if stream_idx >= self.streams.len() {
            return Ok(());
        }

        // Create essence element key
        let essence_key = self.create_essence_key(track_index);

        // Write KLV
        self.writer.write_all(&essence_key)?;
        let len_bytes = encode_ber_length(data.len());
        self.writer.write_all(&len_bytes)?;
        self.writer.write_all(data)?;

        // Update stream state
        let stream = &mut self.streams[stream_idx];
        stream.frame_count += 1;
        stream.total_bytes += data.len() as u64;

        Ok(())
    }

    /// Create essence element key for a track
    fn create_essence_key(&self, track_index: u32) -> [u8; 16] {
        let stream = &self.streams[track_index as usize];

        // Base essence element key
        let mut key = [
            0x06, 0x0E, 0x2B, 0x34, // SMPTE prefix
            0x01, 0x02, 0x01, 0x01, // Category/registry/structure/version
            0x0D, 0x01, 0x03, 0x01, // MXF GC
            0x00, 0x00, 0x00, 0x00, // Item type/element count/element type/element number
        ];

        match stream.config.kind {
            TrackKind::Video => {
                key[12] = 0x15; // Picture item
                key[13] = 0x01; // Element count
                key[14] = match stream.config.coding {
                    EssenceCoding::Mpeg2 => 0x05,
                    EssenceCoding::Avc => 0x10,
                    EssenceCoding::Hevc => 0x15,
                    _ => 0x01,
                };
                key[15] = (track_index + 1) as u8;
            }
            TrackKind::Audio => {
                key[12] = 0x16; // Sound item
                key[13] = 0x01;
                key[14] = 0x01; // PCM
                key[15] = (track_index + 1) as u8;
            }
            _ => {
                key[12] = 0x17; // Data item
                key[13] = 0x01;
                key[14] = 0x01;
                key[15] = (track_index + 1) as u8;
            }
        }

        key
    }

    /// Finalize the file
    pub fn finalize(&mut self) -> Result<()> {
        if self.finalized {
            return Ok(());
        }

        if !self.header_written {
            self.write_header()?;
        }

        let footer_offset = self.writer.stream_position()?;

        // Write footer partition
        let mut footer = Partition::footer();
        footer.this_partition = footer_offset;
        footer.previous_partition = self.body_partition_offset;
        footer.footer_partition = footer_offset;
        footer.operational_pattern = self.config.operational_pattern;

        for stream in &self.streams {
            let essence_ul = self.coding_to_essence_ul(stream.config.coding);
            footer.add_essence_container(essence_ul);
        }

        footer.write(&mut self.writer)?;

        // Write random index pack
        let mut rip = RandomIndexPack::new();
        rip.add_entry(0, self.header_partition_offset);
        rip.add_entry(1, self.body_partition_offset);
        rip.add_entry(0, footer_offset);
        rip.write(&mut self.writer)?;

        // Update header partition with footer offset
        self.writer.seek(SeekFrom::Start(self.header_partition_offset + 16 + 4 + 4 + 4 + 8 + 8))?;
        self.writer.write_u64::<BigEndian>(footer_offset)?;

        self.finalized = true;
        log::debug!("MXF finalized, footer at {}", footer_offset);

        Ok(())
    }

    /// Get frame count for track
    pub fn frame_count(&self, track_index: u32) -> u64 {
        self.streams
            .get(track_index as usize)
            .map(|s| s.frame_count)
            .unwrap_or(0)
    }

    /// Get total bytes written for track
    pub fn bytes_written(&self, track_index: u32) -> u64 {
        self.streams
            .get(track_index as usize)
            .map(|s| s.total_bytes)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_track_config_video() {
        let config = TrackConfig::video(1920, 1080, 25.0, EssenceCoding::Avc);
        assert_eq!(config.kind, TrackKind::Video);
        assert_eq!(config.frame_size.unwrap().width, 1920);
        assert_eq!(config.coding, EssenceCoding::Avc);
    }

    #[test]
    fn test_track_config_audio() {
        let config = TrackConfig::audio(48000, 2, 24);
        assert_eq!(config.kind, TrackKind::Audio);
        assert_eq!(config.sample_rate, Some(48000));
        assert_eq!(config.channels, Some(2));
    }

    #[test]
    fn test_muxer_create() {
        let buffer = Cursor::new(Vec::new());
        let config = MuxerConfig::default();
        let muxer: MxfMuxer<Cursor<Vec<u8>>> = MxfMuxer::new(buffer, config);

        assert!(!muxer.header_written);
        assert!(!muxer.finalized);
    }

    #[test]
    fn test_muxer_add_stream() {
        let buffer = Cursor::new(Vec::new());
        let config = MuxerConfig::default();
        let mut muxer = MxfMuxer::new(buffer, config);

        let video = TrackConfig::video(1920, 1080, 25.0, EssenceCoding::Avc);
        let track_id = muxer.add_stream(video).unwrap();

        assert_eq!(track_id, 0);
        assert_eq!(muxer.streams.len(), 1);
    }

    #[test]
    fn test_muxer_write_header() {
        let buffer = Cursor::new(Vec::new());
        let config = MuxerConfig::default();
        let mut muxer = MxfMuxer::new(buffer, config);

        let video = TrackConfig::video(1920, 1080, 25.0, EssenceCoding::Avc);
        muxer.add_stream(video).unwrap();

        muxer.write_header().unwrap();
        assert!(muxer.header_written);
    }

    #[test]
    fn test_muxer_write_packet() {
        let buffer = Cursor::new(Vec::new());
        let config = MuxerConfig::default();
        let mut muxer = MxfMuxer::new(buffer, config);

        let video = TrackConfig::video(1920, 1080, 25.0, EssenceCoding::Avc);
        muxer.add_stream(video).unwrap();

        let frame_data = vec![0u8; 1000];
        muxer.write_packet(0, &frame_data, true).unwrap();

        assert_eq!(muxer.frame_count(0), 1);
        assert_eq!(muxer.bytes_written(0), 1000);
    }

    #[test]
    fn test_muxer_finalize() {
        let buffer = Cursor::new(Vec::new());
        let config = MuxerConfig::default();
        let mut muxer = MxfMuxer::new(buffer, config);

        let video = TrackConfig::video(1920, 1080, 25.0, EssenceCoding::Avc);
        muxer.add_stream(video).unwrap();

        let frame_data = vec![0u8; 1000];
        muxer.write_packet(0, &frame_data, true).unwrap();

        muxer.finalize().unwrap();
        assert!(muxer.finalized);

        // Check output has SMPTE prefix at start
        let output = muxer.writer.into_inner();
        assert!(output.len() > 100);
        assert_eq!(&output[0..4], &[0x06, 0x0E, 0x2B, 0x34]);
    }
}
