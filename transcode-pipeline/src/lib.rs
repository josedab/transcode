//! Transcoding pipeline for the Transcode codec library.
//!
//! This crate provides a high-level API for building transcoding pipelines
//! that connect demuxers, decoders, encoders, and muxers.
//!
//! # Architecture
//!
//! The pipeline follows a node-based architecture where data flows through
//! a series of processing stages:
//!
//! ```text
//! ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌───────┐
//! │ Demuxer │───▶│ Decoder │───▶│ Filters │───▶│ Encoder │───▶│ Muxer │
//! └─────────┘    └─────────┘    └─────────┘    └─────────┘    └───────┘
//!                     │              │
//!                     ▼              ▼
//!              Video/Audio     Scale/Volume
//!                Frames         Adjustments
//! ```
//!
//! # Key Components
//!
//! - [`Pipeline`] - The main transcoding orchestrator
//! - [`PipelineBuilder`] - Builder pattern for pipeline construction
//! - [`Node`] - Trait for pipeline processing nodes
//! - [`FilterChain`] - Composable video/audio filter chains
//! - [`Synchronizer`] - Audio/video synchronization
//!
//! # Usage
//!
//! ```ignore
//! use transcode_pipeline::{Pipeline, PipelineBuilder, PipelineConfig};
//!
//! let pipeline = PipelineBuilder::new()
//!     .config(PipelineConfig::default())
//!     .demuxer(demuxer_node)
//!     .decoder(0, video_decoder)
//!     .encoder(0, h264_encoder)
//!     .muxer(mp4_muxer)
//!     .build()?;
//!
//! pipeline.run()?;
//! ```
//!
//! # Modules
//!
//! - [`error`] - Pipeline error types
//! - [`filter`] - Video and audio filter implementations
//! - [`node`] - Processing node types (demuxer, decoder, encoder, muxer)
//! - [`pipeline`] - Main pipeline implementation
//! - [`sync`] - Audio/video synchronization utilities

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
