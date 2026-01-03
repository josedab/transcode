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
