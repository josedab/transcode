//! Prelude module for convenient imports.
//!
//! This module re-exports the most commonly used types and traits
//! from the transcode library for convenient glob imports.
//!
//! # Usage
//!
//! ```rust
//! use transcode::prelude::*;
//! ```
//!
//! This is equivalent to importing:
//! - Core error and result types
//! - Frame and packet types
//! - Timestamp and time base types
//! - Codec traits
//! - High-level API types

// Core error types
pub use crate::{Error, Result};

// Frame and buffer types
pub use crate::{Frame, FrameBuffer, PixelFormat, ColorSpace, ColorRange};

// Packet types
pub use crate::{Packet, PacketFlags};

// Sample/audio types
pub use crate::{Sample, SampleBuffer, SampleFormat, ChannelLayout};

// Timestamp types
pub use crate::{Timestamp, TimeBase, Duration};

// Codec format types
pub use crate::{VideoCodec, AudioCodec, ContainerFormat};

// Codec traits
pub use crate::{VideoDecoder, VideoEncoder, AudioDecoder, AudioEncoder, CodecInfo};

// Container traits
pub use crate::{Demuxer, Muxer, StreamInfo, TrackType};

// Pipeline types
pub use crate::{Pipeline, PipelineBuilder, PipelineConfig};

// Filter types
pub use crate::{Filter, VideoFilter, AudioFilter, FilterChain};

// High-level API
pub use crate::{Transcoder, TranscodeOptions, TranscodeStats};
pub use crate::{InputConfig, OutputConfig, VideoConfig, AudioConfig};
pub use crate::{Preset, Quality, Format};

// Utility types
pub use crate::Rational;
