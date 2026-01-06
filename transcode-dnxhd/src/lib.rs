// Codec implementations often use patterns that trigger clippy warnings
#![allow(clippy::too_many_arguments)]
#![allow(clippy::similar_names)]

//! DNxHD/DNxHR Video Codec
//!
//! This crate provides a pure Rust implementation of the Avid DNxHD and DNxHR
//! video codecs. These are professional-grade intra-frame codecs widely used
//! in video editing workflows.
//!
//! # Features
//!
//! - Full frame header parsing
//! - DCT-based encoding and decoding
//! - Multiple profile support (DNxHD and DNxHR)
//! - 8-bit and 10-bit sample support
//! - Huffman entropy coding
//!
//! # Profiles
//!
//! ## DNxHD Profiles (HD resolution)
//! - DNxHD 36/45: Low bandwidth for offline editing
//! - DNxHD 90/90x: Medium quality
//! - DNxHD 120/145: High quality for broadcast
//! - DNxHD 175/175x: High quality for finishing
//! - DNxHD 220/220x: Highest quality
//!
//! ## DNxHR Profiles (Higher resolutions)
//! - DNxHR LB (Low Bandwidth): Offline editing
//! - DNxHR SQ (Standard Quality): Production
//! - DNxHR HQ (High Quality): Finishing
//! - DNxHR HQX (High Quality 10-bit): High-end finishing
//! - DNxHR 444: Full chroma (4:4:4)
//!
//! # Example
//!
//! ```no_run
//! use transcode_dnxhd::{DnxDecoder, DnxFrame};
//!
//! let data = std::fs::read("video.dnxhd").unwrap();
//! let mut decoder = DnxDecoder::new();
//! let frame = decoder.decode_frame(&data).unwrap();
//!
//! println!("Frame: {}x{}, profile: {:?}", frame.width, frame.height, frame.profile);
//! ```

mod decoder;
mod encoder;
mod error;
mod frame;
mod huffman;
mod profile;
mod tables;
mod types;

pub use decoder::{DnxDecoder, DecoderConfig, probe_dnxhd, get_profile, get_dimensions};
pub use encoder::{DnxEncoder, EncoderConfig, encode_dnxhd};
pub use error::{DnxError, Result};
pub use frame::{FrameHeader, DnxFrame};
pub use profile::{DnxProfile, ProfileInfo};
pub use types::{BitDepth, ChromaFormat, Colorimetry};
