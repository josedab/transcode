//! Transcoding pipeline for the Transcode codec library.
//!
//! Provides a high-level API for building transcoding pipelines
//! that connect demuxers, decoders, encoders, and muxers.

mod error;
mod filter;
mod node;
mod pipeline;
mod sync;

pub use error::{PipelineError, PipelineTrackType, Result};
pub use filter::{AudioFilter, Filter, FilterChain, NullAudioFilter, NullVideoFilter, ScaleFilter, VideoFilter, VolumeFilter};
pub use node::{
    AudioDecoderWrapper, AudioEncoderWrapper, DecoderNode, DemuxerNode, DemuxerWrapper,
    EncoderNode, MuxerNode, MuxerWrapper, Node, NodeId, NodeOutput, PipelineStreamInfo,
    VideoDecoderWrapper, VideoEncoderWrapper,
};
pub use pipeline::{Pipeline, PipelineBuilder, PipelineConfig, PipelineState};
pub use sync::{ReorderBuffer, SyncConfig, SyncMode, Synchronizer};
