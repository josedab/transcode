//! SMPTE Timecode Library for Transcode
//!
//! This crate provides comprehensive SMPTE 12M timecode support including:
//!
//! - **SMPTE Timecode**: Standard HH:MM:SS:FF format with support for common frame rates
//! - **Drop-Frame Timecode**: Accurate wall-clock time for 29.97/59.94 fps
//! - **LTC (Linear Timecode)**: Audio encoding/decoding of timecode
//! - **VITC (Vertical Interval Timecode)**: Video line embedding of timecode
//!
//! # Quick Start
//!
//! ```rust
//! use transcode_timecode::{Timecode, FrameRate};
//!
//! // Create a timecode
//! let tc = Timecode::new(1, 30, 45, 12, FrameRate::Fps24).unwrap();
//! println!("Timecode: {}", tc); // Output: 01:30:45:12
//!
//! // Parse from string
//! let tc2: Timecode = "01:30:45:12".parse().unwrap();
//!
//! // Convert to frame number
//! let frame_number = tc.to_frame_number();
//!
//! // Timecode arithmetic
//! let tc3 = tc.add_frames(100).unwrap();
//! ```
//!
//! # Drop-Frame Timecode
//!
//! For 29.97 fps content, drop-frame timecode maintains synchronization with
//! real wall-clock time:
//!
//! ```rust
//! use transcode_timecode::{Timecode, FrameRate};
//!
//! // Create drop-frame timecode (note the semicolon separator)
//! let tc = Timecode::new_drop_frame(1, 0, 0, 2, FrameRate::Fps29_97).unwrap();
//! println!("Drop-frame: {}", tc); // Output: 01:00:00;02
//!
//! // Parse drop-frame from string
//! let tc2: Timecode = "01:00:00;02".parse().unwrap();
//! assert!(tc2.drop_frame);
//! ```
//!
//! # LTC Audio Encoding
//!
//! Generate audio signals containing timecode:
//!
//! ```rust
//! use transcode_timecode::{Timecode, FrameRate, ltc::{LtcEncoder, LtcFrame}};
//!
//! let tc = Timecode::new(0, 0, 0, 0, FrameRate::Fps24).unwrap();
//! let mut encoder = LtcEncoder::new(48000, FrameRate::Fps24);
//!
//! let frame = LtcFrame::from_timecode(&tc);
//! let samples = encoder.encode_frame(&frame);
//! // samples contains f32 audio data
//! ```
//!
//! # VITC Video Embedding
//!
//! Embed timecode in video lines:
//!
//! ```rust
//! use transcode_timecode::{Timecode, FrameRate, vitc::{VitcEncoder, VitcData}};
//!
//! let tc = Timecode::new(0, 0, 0, 0, FrameRate::Fps24).unwrap();
//! let encoder = VitcEncoder::new(720);
//!
//! let vitc = VitcData::from_timecode(&tc);
//! let pixels = encoder.encode(&vitc);
//! // pixels contains line data for VBI
//! ```

#![deny(missing_docs)]
#![deny(unsafe_code)]
#![warn(clippy::all)]

pub mod dropframe;
pub mod error;
pub mod ltc;
pub mod smpte;
pub mod vitc;

// Re-export main types
pub use error::{Result, TimecodeError};
pub use smpte::{parse_timecode, FrameRate, Timecode};

// Re-export drop-frame utilities
pub use dropframe::{
    drop_frame_timecode_to_frame, drop_frame_to_wall_time, frame_to_drop_frame_timecode,
    is_dropped_frame, validate_drop_frame_timecode, wall_time_to_drop_frame, DropFrameConfig,
};

/// The version of SMPTE standard this library implements.
pub const SMPTE_VERSION: &str = "SMPTE 12M-2008";

/// Maximum hours value in timecode (23).
pub const MAX_HOURS: u8 = 23;

/// Maximum minutes value in timecode (59).
pub const MAX_MINUTES: u8 = 59;

/// Maximum seconds value in timecode (59).
pub const MAX_SECONDS: u8 = 59;

/// Create a timecode from hours, minutes, seconds, and frames.
///
/// This is a convenience function that creates a non-drop-frame timecode.
///
/// # Arguments
/// * `hours` - Hours (0-23)
/// * `minutes` - Minutes (0-59)
/// * `seconds` - Seconds (0-59)
/// * `frames` - Frames (0 to fps-1)
/// * `frame_rate` - The frame rate
///
/// # Example
/// ```rust
/// use transcode_timecode::{timecode, FrameRate};
///
/// let tc = timecode(1, 30, 45, 12, FrameRate::Fps24).unwrap();
/// assert_eq!(tc.to_string(), "01:30:45:12");
/// ```
pub fn timecode(
    hours: u8,
    minutes: u8,
    seconds: u8,
    frames: u8,
    frame_rate: FrameRate,
) -> Result<Timecode> {
    Timecode::new(hours, minutes, seconds, frames, frame_rate)
}

/// Create a drop-frame timecode from hours, minutes, seconds, and frames.
///
/// # Arguments
/// * `hours` - Hours (0-23)
/// * `minutes` - Minutes (0-59)
/// * `seconds` - Seconds (0-59)
/// * `frames` - Frames (0 to fps-1, excluding dropped frames)
/// * `frame_rate` - The frame rate (must be 29.97 or 59.94)
///
/// # Example
/// ```rust
/// use transcode_timecode::{timecode_df, FrameRate};
///
/// let tc = timecode_df(1, 0, 0, 2, FrameRate::Fps29_97).unwrap();
/// assert_eq!(tc.to_string(), "01:00:00;02");
/// ```
pub fn timecode_df(
    hours: u8,
    minutes: u8,
    seconds: u8,
    frames: u8,
    frame_rate: FrameRate,
) -> Result<Timecode> {
    Timecode::new_drop_frame(hours, minutes, seconds, frames, frame_rate)
}

/// Calculate the duration between two timecodes in seconds.
///
/// # Arguments
/// * `start` - Start timecode
/// * `end` - End timecode
///
/// # Returns
/// Duration in seconds (can be negative if end is before start)
#[must_use]
pub fn duration_seconds(start: &Timecode, end: &Timecode) -> f64 {
    end.to_seconds() - start.to_seconds()
}

/// Calculate the duration between two timecodes in frames.
///
/// Note: This only makes sense if both timecodes have the same frame rate.
///
/// # Arguments
/// * `start` - Start timecode
/// * `end` - End timecode
///
/// # Returns
/// Duration in frames (can be negative if end is before start)
#[must_use]
pub fn duration_frames(start: &Timecode, end: &Timecode) -> i64 {
    end.to_frame_number() as i64 - start.to_frame_number() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_timecode_convenience() {
        let tc = timecode(1, 30, 45, 12, FrameRate::Fps24).unwrap();
        assert_eq!(tc.to_string(), "01:30:45:12");
    }

    #[test]
    fn test_timecode_df_convenience() {
        let tc = timecode_df(1, 0, 0, 2, FrameRate::Fps29_97).unwrap();
        assert_eq!(tc.to_string(), "01:00:00;02");
        assert!(tc.drop_frame);
    }

    #[test]
    fn test_duration_seconds() {
        let start = timecode(0, 0, 0, 0, FrameRate::Fps24).unwrap();
        let end = timecode(0, 1, 0, 0, FrameRate::Fps24).unwrap();

        let duration = duration_seconds(&start, &end);
        assert!((duration - 60.0).abs() < 0.001);
    }

    #[test]
    fn test_duration_frames() {
        let start = timecode(0, 0, 0, 0, FrameRate::Fps24).unwrap();
        let end = timecode(0, 0, 1, 0, FrameRate::Fps24).unwrap();

        let duration = duration_frames(&start, &end);
        assert_eq!(duration, 24);
    }

    #[test]
    fn test_negative_duration() {
        let start = timecode(0, 1, 0, 0, FrameRate::Fps24).unwrap();
        let end = timecode(0, 0, 0, 0, FrameRate::Fps24).unwrap();

        let duration = duration_seconds(&start, &end);
        assert!((duration + 60.0).abs() < 0.001);

        let frame_duration = duration_frames(&start, &end);
        assert_eq!(frame_duration, -1440); // -60 seconds * 24 fps
    }

    #[test]
    fn test_constants() {
        assert_eq!(MAX_HOURS, 23);
        assert_eq!(MAX_MINUTES, 59);
        assert_eq!(MAX_SECONDS, 59);
        assert_eq!(SMPTE_VERSION, "SMPTE 12M-2008");
    }

    #[test]
    fn test_parse_and_format_roundtrip() {
        let original = "12:34:56:07";
        let tc = parse_timecode(original, FrameRate::Fps24).unwrap();
        let formatted = tc.to_string();
        assert_eq!(original, formatted);
    }

    #[test]
    fn test_drop_frame_parse_roundtrip() {
        let original = "12:34:56;07";
        let tc = parse_timecode(original, FrameRate::Fps29_97).unwrap();
        assert!(tc.drop_frame);
        let formatted = tc.to_string();
        assert_eq!(original, formatted);
    }

    #[test]
    fn test_frame_rate_conversions() {
        // Test all standard frame rates
        let frame_rates = [
            FrameRate::Fps24,
            FrameRate::Fps23_976,
            FrameRate::Fps25,
            FrameRate::Fps29_97,
            FrameRate::Fps30,
            FrameRate::Fps48,
            FrameRate::Fps50,
            FrameRate::Fps59_94,
            FrameRate::Fps60,
        ];

        for fps in frame_rates {
            let tc = timecode(0, 1, 0, 0, fps).unwrap();
            let seconds = tc.to_seconds();
            // Should be approximately 60 seconds
            assert!(
                (seconds - 60.0).abs() < 0.1,
                "Frame rate {} gave {} seconds",
                fps,
                seconds
            );
        }
    }

    #[test]
    fn test_timecode_comparison() {
        let tc1 = timecode(0, 0, 0, 0, FrameRate::Fps24).unwrap();
        let tc2 = timecode(0, 0, 0, 1, FrameRate::Fps24).unwrap();
        let tc3 = timecode(0, 0, 1, 0, FrameRate::Fps24).unwrap();

        assert!(tc1 < tc2);
        assert!(tc2 < tc3);
        assert!(tc1 < tc3);
    }

    #[test]
    fn test_frame_number_roundtrip() {
        for frame in [0, 1, 24, 100, 1000, 86400, 100000] {
            let tc = Timecode::from_frame_number(frame, FrameRate::Fps24, false);
            let back = tc.to_frame_number();
            assert_eq!(frame, back, "Frame {} roundtrip failed", frame);
        }
    }

    #[test]
    fn test_drop_frame_roundtrip() {
        for frame in [0, 1, 29, 30, 1799, 1800, 1801, 17982] {
            let tc = Timecode::from_frame_number(frame, FrameRate::Fps29_97, true);
            let back = tc.to_frame_number();
            assert_eq!(
                frame, back,
                "Drop-frame {} roundtrip failed via {}",
                frame, tc
            );
        }
    }
}
