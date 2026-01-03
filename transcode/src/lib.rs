//! # Transcode
//!
//! A memory-safe, high-performance universal codec library written in Rust.
//!
//! This crate provides a unified API for transcoding media files, with support for:
//! - Video codecs: H.264/AVC, H.265/HEVC, VP9, AV1
//! - Audio codecs: AAC, MP3, Opus, FLAC
//! - Containers: MP4/MOV, MKV/WebM, MPEG-TS
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use transcode::{Transcoder, TranscodeOptions};
//!
//! fn main() -> transcode::Result<()> {
//!     let options = TranscodeOptions::new()
//!         .input("input.mp4")
//!         .output("output.mp4")
//!         .video_codec("h264")
//!         .audio_codec("aac");
//!
//!     let mut transcoder = Transcoder::new(options)?;
//!     transcoder.run()?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Architecture
//!
//! The library is organized into several crates:
//! - `transcode-core`: Core types and utilities
//! - `transcode-codecs`: Video and audio codec implementations
//! - `transcode-containers`: Container format demuxers and muxers
//! - `transcode-pipeline`: Transcoding pipeline infrastructure
//!
//! This crate re-exports the most commonly used types and provides a
//! high-level API for simple use cases.

mod options;
mod presets;
pub mod thumbnail;
mod transcoder;

// Re-export core types
pub use transcode_core::{
    error::{BitstreamError, CodecError, ContainerError, Error, Result},
    frame::{ColorRange, ColorSpace, Frame, FrameBuffer, PixelFormat},
    packet::{Packet, PacketFlags},
    rational::Rational,
    sample::{ChannelLayout, Sample, SampleBuffer, SampleFormat},
    timestamp::{Duration, TimeBase, Timestamp},
    ContainerFormat, VideoCodec, AudioCodec,
};

// Re-export codec types
pub use transcode_codecs::{
    traits::{AudioDecoder, AudioEncoder, VideoDecoder, VideoEncoder, CodecInfo},
};

// Re-export container types
pub use transcode_containers::{
    traits::{Demuxer, Muxer, StreamInfo, TrackType, AudioStreamInfo, VideoStreamInfo, CodecId},
    mp4::{Mp4Demuxer, Mp4Muxer},
};

// Re-export pipeline types
pub use transcode_pipeline::{
    Pipeline, PipelineBuilder, PipelineConfig, PipelineState,
    PipelineTrackType, PipelineStreamInfo,
    Filter, VideoFilter, AudioFilter, FilterChain,
    ScaleFilter, VolumeFilter,
    SyncConfig, SyncMode, Synchronizer,
    DemuxerWrapper, MuxerWrapper, VideoDecoderWrapper, AudioDecoderWrapper,
    VideoEncoderWrapper, AudioEncoderWrapper,
    DemuxerNode, MuxerNode, DecoderNode, EncoderNode, Node, NodeId, NodeOutput,
};

// High-level API
pub use options::{TranscodeOptions, InputConfig, OutputConfig, VideoConfig, AudioConfig};
pub use presets::{Preset, Quality, Format};
pub use transcoder::{Transcoder, TranscodeStats};

/// Version information.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Get library version string.
pub fn version() -> &'static str {
    VERSION
}

/// Get build information.
pub fn build_info() -> BuildInfo {
    BuildInfo {
        version: VERSION,
        target: std::env::consts::ARCH,
        os: std::env::consts::OS,
        debug: cfg!(debug_assertions),
    }
}

/// Build information.
#[derive(Debug, Clone)]
pub struct BuildInfo {
    /// Library version.
    pub version: &'static str,
    /// Target architecture.
    pub target: &'static str,
    /// Operating system.
    pub os: &'static str,
    /// Debug build.
    pub debug: bool,
}
