//! Concat demuxer for sequential file demuxing.
//!
//! This module provides a demuxer that can read multiple input files
//! and demux them as a single continuous stream, handling timestamp
//! adjustment automatically between files.
//!
//! Supports two input modes:
//! - Direct file paths
//! - FFmpeg-style concat file format

mod demuxer;

pub use demuxer::{ConcatConfig, ConcatDemuxer, ConcatError, ConcatInputSource, FileEntry};
