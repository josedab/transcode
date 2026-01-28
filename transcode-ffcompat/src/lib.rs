//! FFmpeg-compatible CLI argument translation for drop-in replacement.
//!
//! This crate translates FFmpeg-style command-line arguments into Transcode
//! native options, enabling users to test Transcode with existing scripts
//! and workflows without rewriting.
//!
//! # Supported Arguments
//!
//! ## Input/Output
//! - `-i <file>` → Input file
//! - `-y` → Overwrite output
//! - `-n` → Never overwrite
//!
//! ## Video
//! - `-c:v <codec>` / `-vcodec <codec>` → Video codec (libx264→h264, libx265→hevc, etc.)
//! - `-b:v <rate>` → Video bitrate
//! - `-r <fps>` → Frame rate
//! - `-s <WxH>` → Resolution
//! - `-crf <value>` → Constant Rate Factor
//! - `-preset <name>` → Encoding preset
//! - `-g <size>` → GOP size
//!
//! ## Audio
//! - `-c:a <codec>` / `-acodec <codec>` → Audio codec
//! - `-b:a <rate>` → Audio bitrate
//! - `-ar <rate>` → Audio sample rate
//! - `-ac <channels>` → Audio channels
//! - `-an` → Disable audio
//!
//! ## General
//! - `-t <duration>` → Duration limit
//! - `-ss <time>` → Start time seek
//! - `-to <time>` → End time
//! - `-threads <n>` → Thread count
//! - `-f <format>` → Force output format
//!
//! # Example
//!
//! ```
//! use transcode_ffcompat::{FfmpegArgs, TranslationResult};
//!
//! let args = vec![
//!     "-i", "input.mp4",
//!     "-c:v", "libx264",
//!     "-crf", "23",
//!     "-preset", "medium",
//!     "-c:a", "aac",
//!     "-b:a", "128k",
//!     "output.mp4",
//! ];
//!
//! let result = FfmpegArgs::parse(&args).unwrap();
//! let native = result.to_native_args();
//!
//! assert_eq!(native.video_codec, Some("h264".into()));
//! assert_eq!(native.crf, Some(23));
//! ```

#![allow(dead_code)]

mod error;
pub mod filter;
mod parser;
mod translator;
mod report;

pub use error::{Error, Result};
pub use filter::{FilterGraph, FilterChain, ParsedFilter, NativeFilter};
pub use parser::{FfmpegArgs, FfmpegArg};
pub use translator::{NativeArgs, TranslationResult, TranslationWarning};
pub use report::CompatibilityReport;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_translation() {
        let args = vec![
            "-i", "input.mp4", "-c:v", "libx264", "-crf", "23", "output.mp4",
        ];
        let parsed = FfmpegArgs::parse(&args).unwrap();
        let result = parsed.translate();

        assert_eq!(result.native.input, Some("input.mp4".into()));
        assert_eq!(result.native.output, Some("output.mp4".into()));
        assert_eq!(result.native.video_codec, Some("h264".into()));
        assert_eq!(result.native.crf, Some(23));
    }

    #[test]
    fn test_unsupported_args_generate_warnings() {
        let args = vec![
            "-i", "input.mp4", "-filter_complex", "[0:v]split[a][b]",
            "output.mp4",
        ];
        let parsed = FfmpegArgs::parse(&args).unwrap();
        let result = parsed.translate();
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn test_compatibility_report() {
        let args = vec![
            "-i", "in.mp4", "-c:v", "libx264", "-b:v", "5M",
            "-c:a", "aac", "-b:a", "128k", "out.mp4",
        ];
        let parsed = FfmpegArgs::parse(&args).unwrap();
        let result = parsed.translate();
        let report = CompatibilityReport::from_result(&result);
        assert!(report.fully_supported());
    }
}
