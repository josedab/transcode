//! MXF partition handling
//!
//! MXF files are divided into partitions:
//! - Header partition (required)
//! - Body partitions (optional)
//! - Footer partition (required for closed/complete files)

use crate::error::{MxfError, Result};
use crate::ul::{labels, UniversalLabel};
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read, Write};

/// Partition kind
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionKind {
    /// Header partition
    Header,
    /// Body partition
    Body,
    /// Footer partition
    Footer,
}

/// Partition status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionStatus {
    /// Open (not yet finalized)
    Open,
    /// Closed (finalized)
    Closed,
}

/// Partition completeness
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionCompleteness {
    /// Incomplete (missing some metadata)
    Incomplete,
    /// Complete
    Complete,
}

/// MXF partition pack
#[derive(Debug, Clone)]
pub struct Partition {
    /// Partition kind
    pub kind: PartitionKind,
    /// Status (open/closed)
    pub status: PartitionStatus,
    /// Completeness
    pub completeness: PartitionCompleteness,
    /// Major version (should be 1)
    pub major_version: u16,
    /// Minor version (should be 2 or 3)
    pub minor_version: u16,
    /// KAG size (key alignment grid)
    pub kag_size: u32,
    /// This partition offset in file
    pub this_partition: u64,
    /// Previous partition offset
    pub previous_partition: u64,
    /// Footer partition offset
    pub footer_partition: u64,
    /// Header byte count (bytes after partition pack)
    pub header_byte_count: u64,
    /// Index byte count
    pub index_byte_count: u64,
    /// Index SID (stream ID for index)
    pub index_sid: u32,
    /// Body offset (for body partitions)
    pub body_offset: u64,
    /// Body SID (stream ID for essence)
    pub body_sid: u32,
    /// Operational pattern
    pub operational_pattern: [u8; 16],
    /// Essence containers
    pub essence_containers: Vec<[u8; 16]>,
}

impl Default for Partition {
    fn default() -> Self {
        Partition {
            kind: PartitionKind::Header,
            status: PartitionStatus::Closed,
            completeness: PartitionCompleteness::Complete,
            major_version: 1,
            minor_version: 3,
            kag_size: 1,
            this_partition: 0,
            previous_partition: 0,
            footer_partition: 0,
            header_byte_count: 0,
            index_byte_count: 0,
            index_sid: 0,
            body_offset: 0,
            body_sid: 0,
            operational_pattern: OP1A_UL,
            essence_containers: Vec::new(),
        }
    }
}

/// Op1a operational pattern
pub const OP1A_UL: [u8; 16] = [
    0x06, 0x0E, 0x2B, 0x34, 0x04, 0x01, 0x01, 0x01, 0x0D, 0x01, 0x02, 0x01, 0x01, 0x01, 0x01, 0x00,
];

impl Partition {
    /// Create header partition
    pub fn header() -> Self {
        Partition {
            kind: PartitionKind::Header,
            ..Default::default()
        }
    }

    /// Create body partition
    pub fn body() -> Self {
        Partition {
            kind: PartitionKind::Body,
            ..Default::default()
        }
    }

    /// Create footer partition
    pub fn footer() -> Self {
        Partition {
            kind: PartitionKind::Footer,
            ..Default::default()
        }
    }

    /// Get the partition pack UL based on kind and status
    pub fn pack_ul(&self) -> [u8; 16] {
        match (self.kind, self.status, self.completeness) {
            (PartitionKind::Header, PartitionStatus::Open, PartitionCompleteness::Incomplete) => {
                labels::HEADER_PARTITION_OPEN_INCOMPLETE
            }
            (PartitionKind::Header, PartitionStatus::Closed, PartitionCompleteness::Incomplete) => {
                labels::HEADER_PARTITION_CLOSED_INCOMPLETE
            }
            (PartitionKind::Header, PartitionStatus::Open, PartitionCompleteness::Complete) => {
                labels::HEADER_PARTITION_OPEN_COMPLETE
            }
            (PartitionKind::Header, PartitionStatus::Closed, PartitionCompleteness::Complete) => {
                labels::HEADER_PARTITION_CLOSED_COMPLETE
            }
            (PartitionKind::Body, PartitionStatus::Open, _) => {
                labels::BODY_PARTITION_OPEN_INCOMPLETE
            }
            (PartitionKind::Body, PartitionStatus::Closed, _) => {
                labels::BODY_PARTITION_CLOSED_COMPLETE
            }
            (PartitionKind::Footer, _, _) => labels::FOOTER_PARTITION,
        }
    }

    /// Parse partition from data
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 88 {
            return Err(MxfError::InsufficientData {
                needed: 88,
                available: data.len(),
            });
        }

        let ul = UniversalLabel::new([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8],
            data[9], data[10], data[11], data[12], data[13], data[14], data[15],
        ]);

        if !ul.is_partition_pack() {
            return Err(MxfError::InvalidPartition(
                "Not a partition pack".into(),
            ));
        }

        // Determine kind from UL
        let kind = match data[13] {
            0x02 => PartitionKind::Header,
            0x03 => PartitionKind::Body,
            0x04 => PartitionKind::Footer,
            _ => {
                return Err(MxfError::InvalidPartition(format!(
                    "Unknown partition type: {:02x}",
                    data[13]
                )))
            }
        };

        let status = match data[14] {
            0x01 | 0x03 => PartitionStatus::Open,
            0x02 | 0x04 => PartitionStatus::Closed,
            _ => PartitionStatus::Closed,
        };

        let completeness = match data[14] {
            0x01 | 0x02 => PartitionCompleteness::Incomplete,
            0x03 | 0x04 => PartitionCompleteness::Complete,
            _ => PartitionCompleteness::Complete,
        };

        // Skip key (16) and length (BER) - assume fixed position for now
        let mut cursor = Cursor::new(&data[16..]);

        // Read BER length
        let first_len_byte = cursor.read_u8()?;
        if first_len_byte >= 0x80 {
            let num_bytes = (first_len_byte & 0x7F) as usize;
            cursor.set_position(cursor.position() + num_bytes as u64);
        }

        let major_version = cursor.read_u16::<BigEndian>()?;
        let minor_version = cursor.read_u16::<BigEndian>()?;
        let kag_size = cursor.read_u32::<BigEndian>()?;
        let this_partition = cursor.read_u64::<BigEndian>()?;
        let previous_partition = cursor.read_u64::<BigEndian>()?;
        let footer_partition = cursor.read_u64::<BigEndian>()?;
        let header_byte_count = cursor.read_u64::<BigEndian>()?;
        let index_byte_count = cursor.read_u64::<BigEndian>()?;
        let index_sid = cursor.read_u32::<BigEndian>()?;
        let body_offset = cursor.read_u64::<BigEndian>()?;
        let body_sid = cursor.read_u32::<BigEndian>()?;

        let mut operational_pattern = [0u8; 16];
        cursor.read_exact(&mut operational_pattern)?;

        // Read essence container batch
        let batch_count = cursor.read_u32::<BigEndian>()?;
        let batch_item_size = cursor.read_u32::<BigEndian>()?;

        let mut essence_containers = Vec::new();
        for _ in 0..batch_count {
            if batch_item_size == 16 {
                let mut ul = [0u8; 16];
                cursor.read_exact(&mut ul)?;
                essence_containers.push(ul);
            } else {
                // Skip unknown item size
                cursor.set_position(cursor.position() + batch_item_size as u64);
            }
        }

        Ok(Partition {
            kind,
            status,
            completeness,
            major_version,
            minor_version,
            kag_size,
            this_partition,
            previous_partition,
            footer_partition,
            header_byte_count,
            index_byte_count,
            index_sid,
            body_offset,
            body_sid,
            operational_pattern,
            essence_containers,
        })
    }

    /// Write partition pack
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<usize> {
        // Write key
        writer.write_all(&self.pack_ul())?;

        // Calculate value size
        let essence_size = self.essence_containers.len() * 16;
        let value_size = 88 + essence_size;

        // Write length (BER)
        let len_bytes = crate::klv::encode_ber_length(value_size);
        writer.write_all(&len_bytes)?;

        // Write value
        writer.write_u16::<BigEndian>(self.major_version)?;
        writer.write_u16::<BigEndian>(self.minor_version)?;
        writer.write_u32::<BigEndian>(self.kag_size)?;
        writer.write_u64::<BigEndian>(self.this_partition)?;
        writer.write_u64::<BigEndian>(self.previous_partition)?;
        writer.write_u64::<BigEndian>(self.footer_partition)?;
        writer.write_u64::<BigEndian>(self.header_byte_count)?;
        writer.write_u64::<BigEndian>(self.index_byte_count)?;
        writer.write_u32::<BigEndian>(self.index_sid)?;
        writer.write_u64::<BigEndian>(self.body_offset)?;
        writer.write_u32::<BigEndian>(self.body_sid)?;
        writer.write_all(&self.operational_pattern)?;

        // Write essence container batch
        writer.write_u32::<BigEndian>(self.essence_containers.len() as u32)?;
        writer.write_u32::<BigEndian>(16)?; // item size
        for ec in &self.essence_containers {
            writer.write_all(ec)?;
        }

        Ok(16 + len_bytes.len() + value_size)
    }

    /// Calculate total size
    pub fn size(&self) -> usize {
        let essence_size = self.essence_containers.len() * 16;
        let value_size = 88 + essence_size;
        16 + crate::klv::ber_length_size(value_size) + value_size
    }

    /// Add essence container
    pub fn add_essence_container(&mut self, ul: [u8; 16]) {
        if !self.essence_containers.contains(&ul) {
            self.essence_containers.push(ul);
        }
    }
}

/// Random Index Pack entry
#[derive(Debug, Clone, Copy)]
pub struct RipEntry {
    /// Body SID (0 for header/footer)
    pub body_sid: u32,
    /// Byte offset of partition
    pub byte_offset: u64,
}

/// Random Index Pack
#[derive(Debug, Clone)]
pub struct RandomIndexPack {
    pub entries: Vec<RipEntry>,
}

impl RandomIndexPack {
    /// Create new RIP
    pub fn new() -> Self {
        RandomIndexPack {
            entries: Vec::new(),
        }
    }

    /// Add entry
    pub fn add_entry(&mut self, body_sid: u32, byte_offset: u64) {
        self.entries.push(RipEntry { body_sid, byte_offset });
    }

    /// Parse RIP from data
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            return Err(MxfError::InsufficientData {
                needed: 4,
                available: data.len(),
            });
        }

        // Last 4 bytes are the overall length
        let overall_length = u32::from_be_bytes([
            data[data.len() - 4],
            data[data.len() - 3],
            data[data.len() - 2],
            data[data.len() - 1],
        ]) as usize;

        // Each entry is 12 bytes (4 byte SID + 8 byte offset)
        let entry_count = (overall_length - 16 - 4 - 4) / 12; // minus key, BER, and overall length

        let mut entries = Vec::new();
        let mut offset = 17; // After key and typical BER length

        for _ in 0..entry_count {
            if offset + 12 > data.len() - 4 {
                break;
            }

            let mut cursor = Cursor::new(&data[offset..]);
            let body_sid = cursor.read_u32::<BigEndian>()?;
            let byte_offset = cursor.read_u64::<BigEndian>()?;

            entries.push(RipEntry { body_sid, byte_offset });
            offset += 12;
        }

        Ok(RandomIndexPack { entries })
    }

    /// Write RIP
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<usize> {
        let value_size = self.entries.len() * 12 + 4; // entries + overall length

        // Write key
        writer.write_all(&labels::RANDOM_INDEX_PACK)?;

        // Write length
        let len_bytes = crate::klv::encode_ber_length(value_size);
        writer.write_all(&len_bytes)?;

        // Write entries
        for entry in &self.entries {
            writer.write_u32::<BigEndian>(entry.body_sid)?;
            writer.write_u64::<BigEndian>(entry.byte_offset)?;
        }

        // Write overall length
        let overall_length = (16 + len_bytes.len() + value_size) as u32;
        writer.write_u32::<BigEndian>(overall_length)?;

        Ok(overall_length as usize)
    }
}

impl Default for RandomIndexPack {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_default() {
        let part = Partition::default();
        assert_eq!(part.kind, PartitionKind::Header);
        assert_eq!(part.major_version, 1);
        assert_eq!(part.minor_version, 3);
    }

    #[test]
    fn test_partition_kinds() {
        let header = Partition::header();
        assert_eq!(header.kind, PartitionKind::Header);

        let body = Partition::body();
        assert_eq!(body.kind, PartitionKind::Body);

        let footer = Partition::footer();
        assert_eq!(footer.kind, PartitionKind::Footer);
    }

    #[test]
    fn test_partition_write_read() {
        let mut part = Partition::header();
        part.kag_size = 512;
        part.body_sid = 1;

        let mut buffer = Vec::new();
        let size = part.write(&mut buffer).unwrap();

        assert!(size > 100);
        assert_eq!(&buffer[0..4], &[0x06, 0x0E, 0x2B, 0x34]);
    }

    #[test]
    fn test_rip() {
        let mut rip = RandomIndexPack::new();
        rip.add_entry(0, 0);
        rip.add_entry(1, 1000);
        rip.add_entry(0, 5000);

        let mut buffer = Vec::new();
        let size = rip.write(&mut buffer).unwrap();

        // Should have written key + length + 3 entries + overall length
        assert!(size > 16 + 36 + 4);
    }
}
