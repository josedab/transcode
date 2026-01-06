//! AV1 codec support for the transcode library.
//!
//! This crate provides AV1 video encoding and decoding capabilities:
//! - **Encoding**: Using rav1e, the fastest and safest AV1 encoder written in Rust
//! - **Decoding**: Using dav1d, the fast and efficient AV1 decoder
//!
//! # Features
//!
//! - High-quality AV1 encoding with configurable speed/quality tradeoffs
//! - Fast AV1 decoding with multi-threaded support
//! - Support for 8-bit and 10-bit content
//! - Multiple rate control modes (CQP, VBR, CBR)
//! - Film grain synthesis support in decoder
//! - Tile-based parallelism for faster encoding
//!
//! # Encoding Example
//!
//! ```no_run
//! use transcode_av1::{Av1Encoder, Av1Config, Av1Preset};
//!
//! let config = Av1Config::new(1920, 1080)
//!     .with_preset(Av1Preset::Medium)
//!     .with_bitrate(2_000_000);
//!
//! let mut encoder = Av1Encoder::new(config)?;
//!
//! // Encode frames...
//! # Ok::<(), transcode_av1::Av1Error>(())
//! ```
//!
//! # Decoding Example
//!
//! ```no_run
//! use transcode_av1::{Av1Decoder, Av1DecoderConfig};
//!
//! let config = Av1DecoderConfig::new()
//!     .with_frame_threads(4)
//!     .with_film_grain(true);
//!
//! let mut decoder = Av1Decoder::with_config(config)?;
//!
//! // Send AV1 OBU data
//! decoder.send_data(&av1_data, pts, duration)?;
//!
//! // Get decoded frames
//! while let Some(frame) = decoder.get_frame()? {
//!     // Process decoded frame...
//! }
//! # let av1_data: Vec<u8> = vec![];
//! # let pts: i64 = 0;
//! # let duration: i64 = 0;
//! # Ok::<(), transcode_av1::Av1Error>(())
//! ```

#![allow(dead_code)]

mod config;
mod encoder;
mod decoder;
mod error;

pub use config::{Av1Config, Av1Preset, RateControlMode};
pub use encoder::{Av1Encoder, Av1Packet, Av1FrameType, Av1EncoderStats};
pub use decoder::{Av1Decoder, Av1DecoderConfig, Av1DecodedFrame, Av1DecoderStats};
pub use error::Av1Error;

/// Result type for AV1 operations.
pub type Result<T> = std::result::Result<T, Av1Error>;

/// AV1 codec information.
pub struct Av1Info {
    /// Encoder name.
    pub encoder_name: &'static str,
    /// Encoder version.
    pub encoder_version: &'static str,
    /// Whether 10-bit encoding is supported.
    pub supports_10bit: bool,
    /// Whether HDR is supported.
    pub supports_hdr: bool,
    /// Maximum supported resolution width.
    pub max_width: u32,
    /// Maximum supported resolution height.
    pub max_height: u32,
}

impl Default for Av1Info {
    fn default() -> Self {
        Self {
            encoder_name: "rav1e",
            encoder_version: env!("CARGO_PKG_VERSION"),
            supports_10bit: true,
            supports_hdr: true,
            max_width: 8192,
            max_height: 4320,
        }
    }
}

/// Get AV1 codec information.
pub fn get_info() -> Av1Info {
    Av1Info::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_info() {
        let info = get_info();
        assert_eq!(info.encoder_name, "rav1e");
        assert!(info.supports_10bit);
    }
}
