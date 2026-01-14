//! MXF (Material eXchange Format) Container
//!
//! This crate provides support for the MXF container format, which is the
//! standard file format for professional video exchange, particularly in
//! broadcast, post-production, and archival workflows.
//!
//! # MXF Structure
//!
//! MXF files are organized as a sequence of KLV (Key-Length-Value) triplets:
//!
//! ```text
//! ┌──────────────────┐
//! │  Header Partition │ ◄── File metadata, edit rates, track info
//! ├──────────────────┤
//! │  Header Metadata  │ ◄── SMPTE metadata sets
//! ├──────────────────┤
//! │  Body Partitions  │ ◄── Essence data (video, audio)
//! │  (Content Package)│
//! ├──────────────────┤
//! │  Footer Partition │ ◄── Index tables, random access
//! └──────────────────┘
//! ```
//!
//! # Features
//!
//! - MXF file parsing (demuxing)
//! - MXF file writing (muxing)
//! - Support for Op1a (single-item, single-package) operational pattern
//! - KLV (Key-Length-Value) triplet handling
//! - SMPTE Universal Labels (UL) parsing
//! - Basic metadata extraction
//!
//! # Example
//!
//! ```no_run
//! use transcode_mxf::{MxfDemuxer, MxfMuxer};
//!
//! // Demux an MXF file
//! let data = std::fs::read("video.mxf").unwrap();
//! let mut demuxer = MxfDemuxer::new(&data).unwrap();
//!
//! println!("Tracks: {}", demuxer.track_count());
//! ```
//!
//! # Public Types
//!
//! - [`MxfDemuxer`] - MXF file reading and essence extraction
//! - [`MxfMuxer`] - MXF file writing and essence packaging
//! - [`Klv`], [`KlvReader`] - KLV triplet parsing
//! - [`UniversalLabel`], [`UL`] - SMPTE Universal Label handling
//! - [`Partition`], [`PartitionKind`] - MXF partition structures
//! - [`ContentPackage`], [`EssenceDescriptor`] - Essence descriptors and content packages

mod demuxer;
mod error;
mod klv;
mod metadata;
mod muxer;
mod partition;
mod types;
mod ul;

pub use demuxer::{MxfDemuxer, MxfPacket, TrackInfo};
pub use error::{MxfError, Result};
pub use klv::{Klv, KlvReader};
pub use metadata::{ContentPackage, EssenceDescriptor};
pub use muxer::{MuxerConfig, MxfMuxer, TrackConfig};
pub use partition::{Partition, PartitionKind};
pub use types::{EditRate, MxfTimestamp, Rational, Umid};
pub use ul::{UniversalLabel, UL};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exports() {
        // Verify core types are exported
        let _: fn() -> Result<()> = || Ok(());
    }

    #[test]
    fn test_edit_rate() {
        let rate = EditRate { numerator: 24, denominator: 1 };
        assert_eq!(rate.numerator, 24);
        assert_eq!(rate.denominator, 1);
    }

    #[test]
    fn test_rational() {
        let r = Rational { numerator: 30000, denominator: 1001 };
        assert_eq!(r.numerator, 30000);
        assert_eq!(r.denominator, 1001);
    }
}
