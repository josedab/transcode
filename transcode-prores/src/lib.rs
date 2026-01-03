// Codec implementations often use patterns that trigger clippy warnings
#![allow(clippy::too_many_arguments)]
#![allow(clippy::similar_names)]

//! ProRes Video Decoder
//!
//! This crate provides a pure Rust implementation of the Apple ProRes video codec decoder.
//! It supports all ProRes profiles including:
//!
//! - ProRes 422 Proxy
//! - ProRes 422 LT
//! - ProRes 422 Standard
//! - ProRes 422 HQ
//! - ProRes 4444
//! - ProRes 4444 XQ
//!
//! # Features
//!
//! - Full frame header parsing
//! - Slice-based decoding
//! - DCT coefficient decoding with Huffman tables
//! - 10-bit and 12-bit sample support
//! - Alpha channel support for 4444 profiles
//!
//! # Example
//!
//! ```no_run
//! use transcode_prores::{ProResDecoder, ProResFrame};
//!
//! let data = std::fs::read("video.prores").unwrap();
//! let mut decoder = ProResDecoder::new();
//! let frame = decoder.decode_frame(&data).unwrap();
//!
//! println!("Frame: {}x{}, profile: {:?}", frame.width, frame.height, frame.profile);
//! ```

mod decoder;
mod error;
mod frame;
mod slice;
mod tables;
mod types;

/// Huffman decoding module (public for testing)
pub mod huffman;

pub use decoder::{get_dimensions, get_profile, probe_prores, DecoderConfig, ProResDecoder};
pub use error::{ProResError, Result};
pub use frame::{FrameHeader, ProResFrame};
pub use slice::SliceHeader;
pub use tables::{dezigzag, zigzag, ZIGZAG_SCAN};
pub use types::{BitDepth, ChromaFormat, ColorPrimaries, InterlaceMode, MatrixCoefficients, ProResProfile, TransferCharacteristic};
