//! # Transcode Core
//!
//! Core types and utilities for the Transcode codec library.
//!
//! This crate provides the fundamental building blocks used across all Transcode components:
//! - Error handling types
//! - Bitstream reading/writing utilities
//! - Frame and sample buffer abstractions
//! - Packet and timestamp management
//! - Memory pool implementations

pub mod error;
pub mod bitstream;
pub mod frame;
pub mod sample;
pub mod packet;
pub mod timestamp;
pub mod pool;
pub mod format;
pub mod rational;

pub use error::{Error, Result};
pub use frame::{Frame, FrameBuffer, PixelFormat, ColorSpace, ColorRange};
pub use sample::{Sample, SampleBuffer, SampleFormat};
pub use packet::{Packet, PacketFlags};
pub use timestamp::{Timestamp, TimeBase, Duration};
pub use format::{ContainerFormat, VideoCodec, AudioCodec};
pub use rational::Rational;
