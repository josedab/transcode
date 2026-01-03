//! Container format implementations for demuxing and muxing.
//!
//! This crate provides support for reading and writing media container formats
//! such as MP4, MKV, WebM, and MPEG-TS.
//!
//! Additionally, this crate provides a concat demuxer for reading multiple files
//! as a single continuous stream.

pub mod chapters;
pub mod concat;
pub mod mp4;
pub mod traits;

pub use chapters::{Chapter, ChapterList, ChapterTrackRef, Mp4ChapterReader, Mp4ChapterWriter};
pub use concat::{ConcatConfig, ConcatDemuxer, ConcatError, ConcatInputSource, FileEntry};
pub use mp4::{Mp4Demuxer, Mp4Muxer};
pub use traits::{Demuxer, Muxer, SeekMode, SeekResult, SeekTarget, StreamInfo, TrackType};
