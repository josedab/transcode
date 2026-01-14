//! MXF metadata structures

use crate::types::{
    AspectRatio, ColorPrimaries, EditRate, EssenceCoding, FrameSize, Rational, TrackKind,
    TransferCharacteristic, Umid,
};
use crate::ul::UL;

/// Content package (essence and metadata)
#[derive(Debug, Clone)]
pub struct ContentPackage {
    /// Package UID (UMID)
    pub package_uid: Umid,
    /// Package name
    pub name: Option<String>,
    /// Tracks in this package
    pub tracks: Vec<Track>,
}

impl Default for ContentPackage {
    fn default() -> Self {
        ContentPackage {
            package_uid: Umid::generate(),
            name: None,
            tracks: Vec::new(),
        }
    }
}

impl ContentPackage {
    /// Create new content package
    pub fn new() -> Self {
        Self::default()
    }

    /// Add track
    pub fn add_track(&mut self, track: Track) {
        self.tracks.push(track);
    }

    /// Find video track
    pub fn video_track(&self) -> Option<&Track> {
        self.tracks.iter().find(|t| t.kind == TrackKind::Video)
    }

    /// Find audio tracks
    pub fn audio_tracks(&self) -> Vec<&Track> {
        self.tracks.iter().filter(|t| t.kind == TrackKind::Audio).collect()
    }
}

/// Track in a package
#[derive(Debug, Clone)]
pub struct Track {
    /// Track ID
    pub track_id: u32,
    /// Track number (for essence mapping)
    pub track_number: u32,
    /// Track kind
    pub kind: TrackKind,
    /// Edit rate
    pub edit_rate: EditRate,
    /// Origin (start time)
    pub origin: i64,
    /// Duration in edit units
    pub duration: u64,
    /// Track name
    pub name: Option<String>,
}

impl Default for Track {
    fn default() -> Self {
        Track {
            track_id: 1,
            track_number: 0,
            kind: TrackKind::Unknown,
            edit_rate: Rational::fps_25(),
            origin: 0,
            duration: 0,
            name: None,
        }
    }
}

/// Essence descriptor
#[derive(Debug, Clone)]
pub struct EssenceDescriptor {
    /// Linked track ID
    pub linked_track_id: u32,
    /// Essence container UL
    pub essence_container: UL,
    /// Codec UL
    pub codec: UL,
    /// Sample rate
    pub sample_rate: Rational,
    /// Container duration
    pub container_duration: Option<u64>,
    /// Essence coding type
    pub coding: EssenceCoding,
    /// Video-specific properties
    pub video: Option<VideoDescriptor>,
    /// Audio-specific properties
    pub audio: Option<AudioDescriptor>,
}

impl Default for EssenceDescriptor {
    fn default() -> Self {
        EssenceDescriptor {
            linked_track_id: 0,
            essence_container: [0; 16],
            codec: [0; 16],
            sample_rate: Rational::fps_25(),
            container_duration: None,
            coding: EssenceCoding::Unknown,
            video: None,
            audio: None,
        }
    }
}

/// Video-specific descriptor
#[derive(Debug, Clone)]
pub struct VideoDescriptor {
    /// Frame size
    pub frame_size: FrameSize,
    /// Display size (may differ from frame size)
    pub display_size: FrameSize,
    /// Stored size (including padding)
    pub stored_size: FrameSize,
    /// Aspect ratio
    pub aspect_ratio: AspectRatio,
    /// Video line map
    pub video_line_map: Vec<i32>,
    /// Field dominance
    pub field_dominance: FieldDominance,
    /// Component depth (bits per component)
    pub component_depth: u32,
    /// Horizontal subsampling
    pub horizontal_subsampling: u32,
    /// Vertical subsampling
    pub vertical_subsampling: u32,
    /// Color primaries
    pub color_primaries: ColorPrimaries,
    /// Transfer characteristic
    pub transfer_characteristic: TransferCharacteristic,
    /// Picture compression
    pub picture_compression: UL,
}

impl Default for VideoDescriptor {
    fn default() -> Self {
        VideoDescriptor {
            frame_size: FrameSize::hd(),
            display_size: FrameSize::hd(),
            stored_size: FrameSize::hd(),
            aspect_ratio: AspectRatio::widescreen(),
            video_line_map: vec![21, 584], // HD default
            field_dominance: FieldDominance::Progressive,
            component_depth: 10,
            horizontal_subsampling: 2, // 4:2:2
            vertical_subsampling: 1,
            color_primaries: ColorPrimaries::BT709,
            transfer_characteristic: TransferCharacteristic::BT709,
            picture_compression: [0; 16],
        }
    }
}

/// Audio-specific descriptor
#[derive(Debug, Clone)]
pub struct AudioDescriptor {
    /// Audio sampling rate
    pub audio_sampling_rate: Rational,
    /// Channel count
    pub channel_count: u32,
    /// Quantization bits
    pub quantization_bits: u32,
    /// Locked (audio is locked to video)
    pub locked: bool,
    /// Audio reference level
    pub audio_ref_level: i8,
    /// Dial norm
    pub dial_norm: i8,
    /// Sound compression
    pub sound_compression: UL,
}

impl Default for AudioDescriptor {
    fn default() -> Self {
        AudioDescriptor {
            audio_sampling_rate: Rational::new(48000, 1),
            channel_count: 2,
            quantization_bits: 24,
            locked: true,
            audio_ref_level: 0,
            dial_norm: 0,
            sound_compression: [0; 16],
        }
    }
}

/// Field dominance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FieldDominance {
    /// Progressive (non-interlaced)
    #[default]
    Progressive,
    /// Field 1 dominant (top field first)
    Field1,
    /// Field 2 dominant (bottom field first)
    Field2,
}

/// Index table segment
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct IndexTableSegment {
    /// Instance ID
    pub instance_id: [u8; 16],
    /// Index edit rate
    pub index_edit_rate: Rational,
    /// Index start position
    pub index_start_position: i64,
    /// Index duration
    pub index_duration: i64,
    /// Edit unit byte count (0 for variable)
    pub edit_unit_byte_count: u32,
    /// Index SID
    pub index_sid: u32,
    /// Body SID
    pub body_sid: u32,
    /// Slice count (for interleaved essence)
    pub slice_count: u8,
    /// Delta entries
    pub delta_entries: Vec<DeltaEntry>,
    /// Index entries
    pub index_entries: Vec<IndexEntry>,
}

impl Default for IndexTableSegment {
    fn default() -> Self {
        IndexTableSegment {
            instance_id: [0; 16],
            index_edit_rate: Rational::fps_25(),
            index_start_position: 0,
            index_duration: 0,
            edit_unit_byte_count: 0,
            index_sid: 1,
            body_sid: 1,
            slice_count: 0,
            delta_entries: Vec::new(),
            index_entries: Vec::new(),
        }
    }
}

/// Delta entry (for multi-track essence)
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct DeltaEntry {
    /// Position offset
    pub pos_table_index: i8,
    /// Slice number
    pub slice: u8,
    /// Element delta (bytes from start)
    pub element_delta: u32,
}

/// Index entry
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct IndexEntry {
    /// Temporal offset
    pub temporal_offset: i8,
    /// Key frame offset
    pub key_frame_offset: i8,
    /// Flags
    pub flags: u8,
    /// Stream offset
    pub stream_offset: u64,
    /// Slice offsets (for interleaved)
    pub slice_offsets: Vec<u32>,
    /// Position table (for delta entries)
    pub pos_table: Vec<Rational>,
}

#[allow(dead_code)]
impl IndexEntry {
    /// Index entry flags
    pub const FLAG_RANDOM_ACCESS: u8 = 0x80;
    pub const FLAG_SEQUENCE_HEADER: u8 = 0x40;
    pub const FLAG_FORWARD_PREDICTION: u8 = 0x20;
    pub const FLAG_BACKWARD_PREDICTION: u8 = 0x10;

    /// Check if this is a random access point
    pub fn is_random_access(&self) -> bool {
        (self.flags & Self::FLAG_RANDOM_ACCESS) != 0
    }
}

/// Primer pack for local tag mapping
#[derive(Debug, Clone)]
pub struct PrimerPack {
    /// Local tag to UL mapping
    pub mappings: Vec<(u16, UL)>,
}

impl Default for PrimerPack {
    fn default() -> Self {
        PrimerPack::new()
    }
}

#[allow(dead_code)]
impl PrimerPack {
    /// Create new primer pack with standard mappings
    pub fn new() -> Self {
        PrimerPack {
            mappings: vec![
                // Common metadata tags
                (0x3C0A, *b"\x06\x0E\x2B\x34\x01\x01\x01\x01\x01\x01\x15\x02\x00\x00\x00\x00"), // Instance UID
                (0x0102, *b"\x06\x0E\x2B\x34\x01\x01\x01\x02\x05\x20\x07\x01\x05\x01\x00\x00"), // Generation UID
            ],
        }
    }

    /// Add mapping
    pub fn add(&mut self, tag: u16, ul: UL) {
        if !self.mappings.iter().any(|(t, _)| *t == tag) {
            self.mappings.push((tag, ul));
        }
    }

    /// Lookup UL by tag
    pub fn lookup(&self, tag: u16) -> Option<&UL> {
        self.mappings.iter().find(|(t, _)| *t == tag).map(|(_, ul)| ul)
    }

    /// Lookup tag by UL
    pub fn reverse_lookup(&self, ul: &UL) -> Option<u16> {
        self.mappings.iter().find(|(_, u)| u == ul).map(|(t, _)| *t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_package() {
        let mut pkg = ContentPackage::new();
        assert!(!pkg.package_uid.is_zero());

        pkg.add_track(Track {
            track_id: 1,
            kind: TrackKind::Video,
            ..Default::default()
        });

        pkg.add_track(Track {
            track_id: 2,
            kind: TrackKind::Audio,
            ..Default::default()
        });

        assert!(pkg.video_track().is_some());
        assert_eq!(pkg.audio_tracks().len(), 1);
    }

    #[test]
    fn test_essence_descriptor() {
        let desc = EssenceDescriptor::default();
        assert!(desc.video.is_none());
        assert!(desc.audio.is_none());
    }

    #[test]
    fn test_video_descriptor() {
        let desc = VideoDescriptor::default();
        assert_eq!(desc.frame_size.width, 1920);
        assert_eq!(desc.frame_size.height, 1080);
        assert_eq!(desc.component_depth, 10);
    }

    #[test]
    fn test_audio_descriptor() {
        let desc = AudioDescriptor::default();
        assert_eq!(desc.audio_sampling_rate.numerator, 48000);
        assert_eq!(desc.channel_count, 2);
        assert_eq!(desc.quantization_bits, 24);
    }

    #[test]
    fn test_primer_pack() {
        let mut primer = PrimerPack::new();
        primer.add(0x1234, [0; 16]);

        assert!(primer.lookup(0x1234).is_some());
        assert!(primer.lookup(0x9999).is_none());
    }

    #[test]
    fn test_index_entry_flags() {
        let entry = IndexEntry {
            temporal_offset: 0,
            key_frame_offset: 0,
            flags: IndexEntry::FLAG_RANDOM_ACCESS,
            stream_offset: 0,
            slice_offsets: Vec::new(),
            pos_table: Vec::new(),
        };

        assert!(entry.is_random_access());
    }
}
