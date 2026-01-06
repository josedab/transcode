//! Universal Label (UL) types for MXF
//!
//! Universal Labels are 16-byte identifiers defined by SMPTE for
//! identifying all elements in MXF files.

use std::fmt;

/// A 16-byte Universal Label
pub type UL = [u8; 16];

/// Universal Label wrapper with helper methods
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct UniversalLabel(pub UL);

impl UniversalLabel {
    /// Create from raw bytes
    pub fn new(bytes: UL) -> Self {
        UniversalLabel(bytes)
    }

    /// Get raw bytes
    pub fn as_bytes(&self) -> &UL {
        &self.0
    }

    /// Check if this is a SMPTE-registered label (starts with 06 0E 2B 34)
    pub fn is_smpte(&self) -> bool {
        self.0[0] == 0x06
            && self.0[1] == 0x0E
            && self.0[2] == 0x2B
            && self.0[3] == 0x34
    }

    /// Get the category code (byte 5)
    pub fn category(&self) -> u8 {
        self.0[4]
    }

    /// Get the registry designator (byte 6)
    pub fn registry(&self) -> u8 {
        self.0[5]
    }

    /// Get the structure designator (byte 7)
    pub fn structure(&self) -> u8 {
        self.0[6]
    }

    /// Get version (byte 8)
    pub fn version(&self) -> u8 {
        self.0[7]
    }

    /// Check if this matches a pattern (ignoring version byte)
    pub fn matches_base(&self, pattern: &UL) -> bool {
        // Compare all bytes except version (byte 7) and kind-specific bytes (13-15)
        self.0[0..7] == pattern[0..7] && self.0[8..13] == pattern[8..13]
    }

    /// Check if this is a partition pack
    pub fn is_partition_pack(&self) -> bool {
        // Match first 13 bytes AND byte 13 must be 0x02 (header), 0x03 (body), or 0x04 (footer)
        // Primer pack has byte 13 = 0x05, so exclude it
        self.0[0..13] == labels::PARTITION_PACK_BASE[0..13]
            && (self.0[13] >= 0x02 && self.0[13] <= 0x04)
    }

    /// Check if this is a primer pack
    pub fn is_primer_pack(&self) -> bool {
        self.0 == labels::PRIMER_PACK
    }

    /// Check if this is a fill item
    pub fn is_fill_item(&self) -> bool {
        self.0[0..13] == labels::FILL_ITEM[0..13]
    }

    /// Check if this is essence data
    pub fn is_essence(&self) -> bool {
        // Essence elements have category 01
        self.is_smpte() && self.0[4] == 0x01
    }

    /// Check if this is a metadata set
    pub fn is_metadata(&self) -> bool {
        // Local sets have category 02
        self.is_smpte() && self.0[4] == 0x02
    }

    /// Get kind description
    pub fn kind(&self) -> LabelKind {
        if self.is_partition_pack() {
            LabelKind::PartitionPack
        } else if self.is_primer_pack() {
            LabelKind::PrimerPack
        } else if self.is_fill_item() {
            LabelKind::FillItem
        } else if self.is_essence() {
            LabelKind::EssenceElement
        } else if self.is_metadata() {
            LabelKind::MetadataSet
        } else {
            LabelKind::Unknown
        }
    }
}

impl fmt::Debug for UniversalLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "UL({:02x}.{:02x}.{:02x}.{:02x}.{:02x}.{:02x}.{:02x}.{:02x}.\
             {:02x}.{:02x}.{:02x}.{:02x}.{:02x}.{:02x}.{:02x}.{:02x})",
            self.0[0],
            self.0[1],
            self.0[2],
            self.0[3],
            self.0[4],
            self.0[5],
            self.0[6],
            self.0[7],
            self.0[8],
            self.0[9],
            self.0[10],
            self.0[11],
            self.0[12],
            self.0[13],
            self.0[14],
            self.0[15]
        )
    }
}

impl fmt::Display for UniversalLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = labels::lookup_name(&self.0);
        write!(f, "{}", name)
    }
}

impl From<UL> for UniversalLabel {
    fn from(bytes: UL) -> Self {
        UniversalLabel(bytes)
    }
}

impl From<&[u8; 16]> for UniversalLabel {
    fn from(bytes: &[u8; 16]) -> Self {
        UniversalLabel(*bytes)
    }
}

/// Kind of Universal Label
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelKind {
    /// Partition pack
    PartitionPack,
    /// Primer pack
    PrimerPack,
    /// Fill item (KLV fill)
    FillItem,
    /// Essence element
    EssenceElement,
    /// Metadata set
    MetadataSet,
    /// Index table
    IndexTable,
    /// Unknown
    Unknown,
}

/// Well-known Universal Labels
pub mod labels {
    use super::UL;

    /// SMPTE Label prefix
    pub const SMPTE_PREFIX: [u8; 4] = [0x06, 0x0E, 0x2B, 0x34];

    /// Partition pack base (last 4 bytes vary)
    pub const PARTITION_PACK_BASE: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x05, 0x01, 0x01, 0x0D, 0x01, 0x02, 0x01, 0x01, 0x00, 0x00,
        0x00,
    ];

    /// Header partition - open incomplete
    pub const HEADER_PARTITION_OPEN_INCOMPLETE: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x05, 0x01, 0x01, 0x0D, 0x01, 0x02, 0x01, 0x01, 0x02, 0x01,
        0x00,
    ];

    /// Header partition - closed incomplete
    pub const HEADER_PARTITION_CLOSED_INCOMPLETE: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x05, 0x01, 0x01, 0x0D, 0x01, 0x02, 0x01, 0x01, 0x02, 0x02,
        0x00,
    ];

    /// Header partition - open complete
    pub const HEADER_PARTITION_OPEN_COMPLETE: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x05, 0x01, 0x01, 0x0D, 0x01, 0x02, 0x01, 0x01, 0x02, 0x03,
        0x00,
    ];

    /// Header partition - closed complete
    pub const HEADER_PARTITION_CLOSED_COMPLETE: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x05, 0x01, 0x01, 0x0D, 0x01, 0x02, 0x01, 0x01, 0x02, 0x04,
        0x00,
    ];

    /// Body partition - open incomplete
    pub const BODY_PARTITION_OPEN_INCOMPLETE: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x05, 0x01, 0x01, 0x0D, 0x01, 0x02, 0x01, 0x01, 0x03, 0x01,
        0x00,
    ];

    /// Body partition - closed complete
    pub const BODY_PARTITION_CLOSED_COMPLETE: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x05, 0x01, 0x01, 0x0D, 0x01, 0x02, 0x01, 0x01, 0x03, 0x04,
        0x00,
    ];

    /// Footer partition
    pub const FOOTER_PARTITION: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x05, 0x01, 0x01, 0x0D, 0x01, 0x02, 0x01, 0x01, 0x04, 0x04,
        0x00,
    ];

    /// Primer pack
    pub const PRIMER_PACK: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x05, 0x01, 0x01, 0x0D, 0x01, 0x02, 0x01, 0x01, 0x05, 0x01,
        0x00,
    ];

    /// Fill item
    pub const FILL_ITEM: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x01, 0x01, 0x01, 0x02, 0x03, 0x01, 0x02, 0x10, 0x01, 0x00, 0x00,
        0x00,
    ];

    /// Index table segment
    pub const INDEX_TABLE_SEGMENT: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x53, 0x01, 0x01, 0x0D, 0x01, 0x02, 0x01, 0x01, 0x10, 0x01,
        0x00,
    ];

    /// Random index pack
    pub const RANDOM_INDEX_PACK: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x05, 0x01, 0x01, 0x0D, 0x01, 0x02, 0x01, 0x01, 0x11, 0x01,
        0x00,
    ];

    // Metadata sets

    /// Preface (root of metadata tree)
    pub const PREFACE: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x53, 0x01, 0x01, 0x0D, 0x01, 0x01, 0x01, 0x01, 0x01, 0x2F,
        0x00,
    ];

    /// Content storage
    pub const CONTENT_STORAGE: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x53, 0x01, 0x01, 0x0D, 0x01, 0x01, 0x01, 0x01, 0x01, 0x18,
        0x00,
    ];

    /// Material package
    pub const MATERIAL_PACKAGE: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x53, 0x01, 0x01, 0x0D, 0x01, 0x01, 0x01, 0x01, 0x01, 0x36,
        0x00,
    ];

    /// Source package
    pub const SOURCE_PACKAGE: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x53, 0x01, 0x01, 0x0D, 0x01, 0x01, 0x01, 0x01, 0x01, 0x37,
        0x00,
    ];

    /// Timeline track
    pub const TIMELINE_TRACK: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x53, 0x01, 0x01, 0x0D, 0x01, 0x01, 0x01, 0x01, 0x01, 0x3B,
        0x00,
    ];

    /// Sequence
    pub const SEQUENCE: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x53, 0x01, 0x01, 0x0D, 0x01, 0x01, 0x01, 0x01, 0x01, 0x0F,
        0x00,
    ];

    /// Source clip
    pub const SOURCE_CLIP: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x53, 0x01, 0x01, 0x0D, 0x01, 0x01, 0x01, 0x01, 0x01, 0x11,
        0x00,
    ];

    // Essence descriptors

    /// CDCI descriptor (component video)
    pub const CDCI_DESCRIPTOR: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x53, 0x01, 0x01, 0x0D, 0x01, 0x01, 0x01, 0x01, 0x01, 0x28,
        0x00,
    ];

    /// RGBA descriptor
    pub const RGBA_DESCRIPTOR: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x53, 0x01, 0x01, 0x0D, 0x01, 0x01, 0x01, 0x01, 0x01, 0x29,
        0x00,
    ];

    /// Wave audio descriptor
    pub const WAVE_AUDIO_DESCRIPTOR: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x53, 0x01, 0x01, 0x0D, 0x01, 0x01, 0x01, 0x01, 0x01, 0x48,
        0x00,
    ];

    /// AES3 audio descriptor
    pub const AES3_AUDIO_DESCRIPTOR: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x53, 0x01, 0x01, 0x0D, 0x01, 0x01, 0x01, 0x01, 0x01, 0x47,
        0x00,
    ];

    // Essence container labels

    /// MPEG-2 video
    pub const ESSENCE_MPEG2: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x04, 0x01, 0x01, 0x03, 0x04, 0x01, 0x02, 0x02, 0x01, 0x00, 0x00,
        0x00,
    ];

    /// AVC (H.264) video
    pub const ESSENCE_AVC: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x04, 0x01, 0x01, 0x0A, 0x04, 0x01, 0x02, 0x02, 0x01, 0x32, 0x00,
        0x00,
    ];

    /// JPEG 2000
    pub const ESSENCE_JPEG2000: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x04, 0x01, 0x01, 0x07, 0x04, 0x01, 0x02, 0x02, 0x03, 0x01, 0x00,
        0x00,
    ];

    /// Uncompressed video
    pub const ESSENCE_UNCOMPRESSED: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x04, 0x01, 0x01, 0x01, 0x04, 0x01, 0x02, 0x01, 0x01, 0x00, 0x00,
        0x00,
    ];

    /// PCM audio
    pub const ESSENCE_PCM: UL = [
        0x06, 0x0E, 0x2B, 0x34, 0x04, 0x01, 0x01, 0x01, 0x04, 0x02, 0x02, 0x01, 0x00, 0x00, 0x00,
        0x00,
    ];

    /// Lookup human-readable name for a UL
    pub fn lookup_name(ul: &UL) -> &'static str {
        // Check primer pack first (before partition packs, as they share bytes 0..13)
        if *ul == PRIMER_PACK {
            return "Primer Pack";
        }

        // Check partition packs
        if ul[0..13] == PARTITION_PACK_BASE[0..13] && (ul[13] >= 0x02 && ul[13] <= 0x04) {
            return match ul[13..15] {
                [0x02, 0x01] => "Header Partition (Open Incomplete)",
                [0x02, 0x02] => "Header Partition (Closed Incomplete)",
                [0x02, 0x03] => "Header Partition (Open Complete)",
                [0x02, 0x04] => "Header Partition (Closed Complete)",
                [0x03, 0x01] => "Body Partition (Open Incomplete)",
                [0x03, 0x04] => "Body Partition (Closed Complete)",
                [0x04, _] => "Footer Partition",
                _ => "Partition Pack (Unknown)",
            };
        }

        if ul[0..13] == FILL_ITEM[0..13] {
            return "Fill Item";
        }
        if *ul == INDEX_TABLE_SEGMENT {
            return "Index Table Segment";
        }
        if *ul == RANDOM_INDEX_PACK {
            return "Random Index Pack";
        }
        if *ul == PREFACE {
            return "Preface";
        }
        if *ul == CONTENT_STORAGE {
            return "Content Storage";
        }
        if *ul == MATERIAL_PACKAGE {
            return "Material Package";
        }
        if *ul == SOURCE_PACKAGE {
            return "Source Package";
        }
        if *ul == TIMELINE_TRACK {
            return "Timeline Track";
        }
        if *ul == SEQUENCE {
            return "Sequence";
        }
        if *ul == SOURCE_CLIP {
            return "Source Clip";
        }
        if *ul == CDCI_DESCRIPTOR {
            return "CDCI Descriptor";
        }
        if *ul == RGBA_DESCRIPTOR {
            return "RGBA Descriptor";
        }
        if *ul == WAVE_AUDIO_DESCRIPTOR {
            return "Wave Audio Descriptor";
        }
        if *ul == AES3_AUDIO_DESCRIPTOR {
            return "AES3 Audio Descriptor";
        }

        // Check for essence elements
        if ul[0..4] == SMPTE_PREFIX && ul[4] == 0x01 {
            return "Essence Element";
        }

        // Check for metadata sets
        if ul[0..4] == SMPTE_PREFIX && ul[4] == 0x02 {
            return "Metadata Set";
        }

        "Unknown"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_universal_label() {
        let ul = UniversalLabel::new(labels::PRIMER_PACK);
        assert!(ul.is_smpte());
        assert!(ul.is_primer_pack());
        assert!(!ul.is_partition_pack());
    }

    #[test]
    fn test_partition_detection() {
        let ul = UniversalLabel::new(labels::HEADER_PARTITION_CLOSED_COMPLETE);
        assert!(ul.is_partition_pack());
        assert!(!ul.is_primer_pack());
    }

    #[test]
    fn test_label_kind() {
        let partition = UniversalLabel::new(labels::HEADER_PARTITION_OPEN_INCOMPLETE);
        assert_eq!(partition.kind(), LabelKind::PartitionPack);

        let primer = UniversalLabel::new(labels::PRIMER_PACK);
        assert_eq!(primer.kind(), LabelKind::PrimerPack);

        let fill = UniversalLabel::new(labels::FILL_ITEM);
        assert_eq!(fill.kind(), LabelKind::FillItem);
    }

    #[test]
    fn test_label_name_lookup() {
        assert_eq!(
            labels::lookup_name(&labels::PRIMER_PACK),
            "Primer Pack"
        );
        assert_eq!(
            labels::lookup_name(&labels::PREFACE),
            "Preface"
        );
    }
}
