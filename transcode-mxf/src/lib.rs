//! MXF (Material eXchange Format) Container
//!
//! This crate provides support for the MXF container format, which is the
//! standard file format for professional video exchange, particularly in
//! broadcast, post-production, and archival workflows.
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
