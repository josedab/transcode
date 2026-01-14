//! AVI Container Format
//!
//! This crate provides support for the AVI (Audio Video Interleave) container
//! format. AVI is based on the RIFF (Resource Interchange File Format) structure.
//!
//! # Features
//!
//! - Full RIFF chunk parsing
//! - AVI demuxing (reading)
//! - AVI muxing (writing)
//! - Support for multiple audio/video streams
//! - OpenDML extensions for large files (AVI 2.0)
//!
//! # Example
//!
//! ```no_run
//! use transcode_avi::{AviDemuxer, AviMuxer};
//!
//! // Demux an AVI file
//! let data = std::fs::read("video.avi").unwrap();
//! let mut demuxer = AviDemuxer::new(&data).unwrap();
//!
//! println!("Duration: {} frames", demuxer.header().total_frames);
//! ```

mod chunks;
mod demuxer;
mod error;
mod muxer;
mod types;

pub use chunks::{AviChunk, ChunkId, FourCC, RiffChunk};
pub use demuxer::{AviDemuxer, AviPacket, StreamInfo};
pub use error::{AviError, Result};
pub use muxer::{AviMuxer, MuxerConfig, StreamConfig};
pub use types::{AviHeader, StreamHeader, VideoFormat, AudioFormat};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exports() {
        // Verify core types are exported
        let _: fn() -> Result<()> = || Ok(());
    }

    #[test]
    fn test_fourcc() {
        let fourcc = FourCC::new(*b"RIFF");
        assert_eq!(fourcc.as_str(), "RIFF");
    }

    #[test]
    fn test_chunk_ids() {
        use chunks::chunk_ids;
        assert_eq!(chunk_ids::RIFF.as_str(), "RIFF");
        assert_eq!(chunk_ids::AVI.as_str(), "AVI ");
        assert_eq!(chunk_ids::MOVI.as_str(), "movi");
    }
}
