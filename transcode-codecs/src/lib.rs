//! # Transcode Codecs
//!
//! Video and audio codec implementations for the Transcode library.
//!
//! ## Video Codecs
//! - H.264/AVC decoder and encoder
//! - H.265/HEVC decoder and encoder (planned)
//! - VP9 decoder and encoder (planned)
//! - AV1 decoder and encoder (planned)
//!
//! ## Audio Codecs
//! - AAC decoder and encoder
//! - MP3 decoder
//! - Opus decoder and encoder (planned)
//!
//! ## Trait System
//!
//! All codecs implement a common trait interface for uniform access:
//!
//! - [`Decoder`] - Common interface for all decoders
//! - [`Encoder`] - Common interface for all encoders
//! - [`CodecInfo`] - Metadata about codec capabilities
//!
//! This allows pipeline code to work generically with any codec without
//! knowing the specific implementation details.
//!
//! ## Parallel Processing
//!
//! The [`parallel`] module provides multi-threaded encoding utilities:
//!
//! - [`ParallelMotionEstimator`] - Parallel motion estimation across blocks
//! - [`ParallelRowEncoder`] - Row-level parallelism for encoding
//! - [`LookaheadBuffer`] - Frame lookahead for rate control decisions
//! - [`FrameBatchProcessor`] - Batch processing of video frames
//!
//! ## SIMD Optimization
//! - AVX2 for x86_64 with runtime detection
//! - NEON for AArch64 (always available)
//! - Scalar fallback for all platforms

pub mod video;
pub mod audio;
pub mod traits;
pub mod simd;
pub mod parallel;

pub use traits::{Decoder, Encoder, CodecInfo};
pub use simd::{SimdOps, SimdCapabilities, detect_simd};
pub use parallel::{
    ThreadConfig, FrameType, MotionVector, MotionResult, ReferenceFrame,
    LookaheadFrame, LookaheadBuffer, ParallelMotionEstimator, ParallelRowEncoder,
    ThreadSafeEncoderState, RateControlState, FrameBatchProcessor,
};
