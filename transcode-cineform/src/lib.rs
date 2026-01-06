// Codec implementations often use patterns that trigger clippy warnings
#![allow(clippy::too_many_arguments)]
#![allow(clippy::similar_names)]

//! CineForm Video Codec
//!
//! This crate provides a pure Rust implementation of the GoPro CineForm
//! intermediate video codec. CineForm is a wavelet-based codec that provides
//! high-quality, editing-friendly video compression.
//!
//! # Features
//!
//! - Wavelet-based compression (Haar wavelets)
//! - Multiple quality levels (low to film scan)
//! - Support for various pixel formats
//! - RGBA/BGRA and YUV formats
//! - Variable bitrate encoding
//!
//! # Quality Levels
//!
//! - Low (Proxy): Smallest files, suitable for offline editing
//! - Medium: Good balance of quality and size
//! - High: High quality for finishing
//! - Film Scan 1: Very high quality
//! - Film Scan 2: Maximum quality
//!
//! # Example
//!
//! ```no_run
//! use transcode_cineform::{CineformDecoder, CineformFrame};
//!
//! let data = std::fs::read("video.cfhd").unwrap();
//! let mut decoder = CineformDecoder::new();
//! let frame = decoder.decode_frame(&data).unwrap();
//!
//! println!("Frame: {}x{}", frame.width, frame.height);
//! ```

mod decoder;
mod encoder;
mod error;
mod frame;
mod quantize;
mod tables;
mod types;
mod wavelet;

pub use decoder::{decode_cineform, get_dimensions, probe_cineform, CineformDecoder, DecoderConfig};
pub use encoder::{encode_cineform, CineformEncoder, EncoderConfig};
pub use error::{CineformError, Result};
pub use frame::{CineformFrame, FrameHeader};
pub use types::{BitDepth, PixelFormat, Quality};
