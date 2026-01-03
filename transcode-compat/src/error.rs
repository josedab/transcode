//! Error types for FFmpeg compatibility layer.
//!
//! This module defines errors that can occur when parsing FFmpeg-style
//! command-line options, filter graphs, and other FFmpeg syntax.

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// Result type for FFmpeg compatibility operations.
pub type Result<T> = std::result::Result<T, CompatError>;

/// Error type for FFmpeg compatibility operations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum CompatError {
    /// Invalid option syntax.
    #[error("Invalid option syntax: {0}")]
    InvalidOption(String),

    /// Unknown option.
    #[error("Unknown option: {0}")]
    UnknownOption(String),

    /// Missing option value.
    #[error("Option '{0}' requires a value")]
    MissingValue(String),

    /// Invalid value for option.
    #[error("Invalid value '{value}' for option '{option}': {reason}")]
    InvalidValue {
        /// The option name.
        option: String,
        /// The invalid value.
        value: String,
        /// The reason the value is invalid.
        reason: String,
    },

    /// Invalid stream specifier.
    #[error("Invalid stream specifier: {0}")]
    InvalidStreamSpecifier(String),

    /// Invalid filter syntax.
    #[error("Invalid filter syntax: {0}")]
    InvalidFilter(String),

    /// Invalid filter graph.
    #[error("Invalid filter graph: {0}")]
    InvalidFilterGraph(String),

    /// Unknown codec name.
    #[error("Unknown codec: {0}")]
    UnknownCodec(String),

    /// Unknown format name.
    #[error("Unknown format: {0}")]
    UnknownFormat(String),

    /// Unknown preset name.
    #[error("Unknown preset: {0}")]
    UnknownPreset(String),

    /// Invalid bitrate specification.
    #[error("Invalid bitrate: {0}")]
    InvalidBitrate(String),

    /// Invalid time specification.
    #[error("Invalid time: {0}")]
    InvalidTime(String),

    /// Invalid aspect ratio.
    #[error("Invalid aspect ratio: {0}")]
    InvalidAspectRatio(String),

    /// Invalid resolution.
    #[error("Invalid resolution: {0}")]
    InvalidResolution(String),

    /// Missing input file.
    #[error("No input file specified")]
    MissingInput,

    /// Missing output file.
    #[error("No output file specified")]
    MissingOutput,

    /// Conflicting options.
    #[error("Conflicting options: {0}")]
    ConflictingOptions(String),

    /// Unsupported feature.
    #[error("Unsupported feature: {0}")]
    Unsupported(String),

    /// Unsupported filter.
    #[error("Unsupported filter: {0}")]
    UnsupportedFilter(String),

    /// Filter format mismatch.
    #[error("Filter format mismatch: {0}")]
    FilterFormatMismatch(String),

    /// Parse error with position information.
    #[error("Parse error at position {position}: {message}")]
    ParseError {
        /// Position in input where the error occurred.
        position: usize,
        /// Error message.
        message: String,
    },
}

impl CompatError {
    /// Create an invalid value error.
    pub fn invalid_value(
        option: impl Into<String>,
        value: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self::InvalidValue {
            option: option.into(),
            value: value.into(),
            reason: reason.into(),
        }
    }

    /// Create a parse error.
    pub fn parse_error(position: usize, message: impl Into<String>) -> Self {
        Self::ParseError {
            position,
            message: message.into(),
        }
    }

    /// Check if this is a recoverable error (can continue parsing).
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::UnknownOption(_) | Self::Unsupported(_)
        )
    }
}

/// Specifies which stream(s) an option applies to.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StreamSpecifier {
    /// All streams.
    All,
    /// All video streams.
    Video,
    /// All audio streams.
    Audio,
    /// All subtitle streams.
    Subtitle,
    /// All data streams.
    Data,
    /// Specific stream by global index.
    Index(u32),
    /// Specific stream by type and index within type.
    TypeIndex {
        /// The stream type (video, audio, etc.).
        stream_type: StreamType,
        /// Index within the stream type.
        index: u32,
    },
    /// Specific stream from specific input file.
    FileStream {
        /// Input file index.
        file_index: u32,
        /// Stream index within the file.
        stream_index: u32,
    },
    /// Specific stream by type from specific input file.
    FileTypeStream {
        /// Input file index.
        file_index: u32,
        /// The stream type (video, audio, etc.).
        stream_type: StreamType,
        /// Index within the stream type.
        type_index: u32,
    },
}

/// Stream type for specifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StreamType {
    /// Video stream.
    Video,
    /// Audio stream.
    Audio,
    /// Subtitle stream.
    Subtitle,
    /// Data stream.
    Data,
}

impl fmt::Display for StreamType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Video => write!(f, "v"),
            Self::Audio => write!(f, "a"),
            Self::Subtitle => write!(f, "s"),
            Self::Data => write!(f, "d"),
        }
    }
}

impl StreamSpecifier {
    /// Parse an FFmpeg stream specifier string.
    ///
    /// Supports formats like:
    /// - `0` - stream index
    /// - `v` - all video streams
    /// - `a` - all audio streams
    /// - `v:0` - first video stream
    /// - `0:v` - video streams from first input
    /// - `0:v:0` - first video stream from first input
    /// - `0:1` - stream 1 from first input
    pub fn parse(spec: &str) -> Result<Self> {
        let spec = spec.trim();

        if spec.is_empty() {
            return Ok(Self::All);
        }

        // Single character type specifier
        if spec.len() == 1 {
            match spec {
                "v" | "V" => return Ok(Self::Video),
                "a" | "A" => return Ok(Self::Audio),
                "s" | "S" => return Ok(Self::Subtitle),
                "d" | "D" => return Ok(Self::Data),
                _ => {}
            }
            // Single digit is a stream index
            if let Ok(idx) = spec.parse::<u32>() {
                return Ok(Self::Index(idx));
            }
        }

        let parts: Vec<&str> = spec.split(':').collect();

        match parts.as_slice() {
            // Two-part specifiers need special handling to distinguish:
            // - "v:0" (type:index)
            // - "0:v" (file:type)
            // - "0:1" (file:stream)
            [first, second] => {
                // Try to parse both as numbers first (file:stream case)
                if let (Ok(file_index), Ok(stream_index)) =
                    (first.parse::<u32>(), second.parse::<u32>())
                {
                    return Ok(Self::FileStream {
                        file_index,
                        stream_index,
                    });
                }

                // Try type:index (e.g., "v:0")
                if first.len() == 1 {
                    if let Ok(stream_type) = parse_stream_type(first) {
                        if let Ok(index) = second.parse::<u32>() {
                            return Ok(Self::TypeIndex { stream_type, index });
                        }
                    }
                }

                // Try file:type (e.g., "0:v")
                if second.len() == 1 {
                    if let Ok(file_index) = first.parse::<u32>() {
                        if let Ok(stream_type) = parse_stream_type(second) {
                            return Ok(Self::FileTypeStream {
                                file_index,
                                stream_type,
                                type_index: 0,
                            });
                        }
                    }
                }

                Err(CompatError::InvalidStreamSpecifier(spec.to_string()))
            }
            // "0:v:0" - file, type, and index
            [file_idx, type_char, type_idx] => {
                let file_index = file_idx
                    .parse::<u32>()
                    .map_err(|_| CompatError::InvalidStreamSpecifier(spec.to_string()))?;
                let stream_type = parse_stream_type(type_char)?;
                let type_index = type_idx
                    .parse::<u32>()
                    .map_err(|_| CompatError::InvalidStreamSpecifier(spec.to_string()))?;
                Ok(Self::FileTypeStream {
                    file_index,
                    stream_type,
                    type_index,
                })
            }
            // Just a number
            [idx] => {
                if let Ok(index) = idx.parse::<u32>() {
                    Ok(Self::Index(index))
                } else {
                    Err(CompatError::InvalidStreamSpecifier(spec.to_string()))
                }
            }
            _ => Err(CompatError::InvalidStreamSpecifier(spec.to_string())),
        }
    }

    /// Check if this specifier matches a given stream.
    pub fn matches(&self, file_index: u32, stream_index: u32, stream_type: StreamType) -> bool {
        match self {
            Self::All => true,
            Self::Video => stream_type == StreamType::Video,
            Self::Audio => stream_type == StreamType::Audio,
            Self::Subtitle => stream_type == StreamType::Subtitle,
            Self::Data => stream_type == StreamType::Data,
            Self::Index(idx) => *idx == stream_index,
            Self::TypeIndex {
                stream_type: st,
                index,
            } => *st == stream_type && *index == stream_index,
            Self::FileStream {
                file_index: fi,
                stream_index: si,
            } => *fi == file_index && *si == stream_index,
            Self::FileTypeStream {
                file_index: fi,
                stream_type: st,
                type_index: ti,
            } => *fi == file_index && *st == stream_type && *ti == stream_index,
        }
    }
}

impl fmt::Display for StreamSpecifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::All => write!(f, ""),
            Self::Video => write!(f, "v"),
            Self::Audio => write!(f, "a"),
            Self::Subtitle => write!(f, "s"),
            Self::Data => write!(f, "d"),
            Self::Index(idx) => write!(f, "{}", idx),
            Self::TypeIndex { stream_type, index } => write!(f, "{}:{}", stream_type, index),
            Self::FileStream {
                file_index,
                stream_index,
            } => write!(f, "{}:{}", file_index, stream_index),
            Self::FileTypeStream {
                file_index,
                stream_type,
                type_index,
            } => write!(f, "{}:{}:{}", file_index, stream_type, type_index),
        }
    }
}

fn parse_stream_type(s: &str) -> Result<StreamType> {
    match s.to_lowercase().as_str() {
        "v" | "video" => Ok(StreamType::Video),
        "a" | "audio" => Ok(StreamType::Audio),
        "s" | "subtitle" | "sub" => Ok(StreamType::Subtitle),
        "d" | "data" => Ok(StreamType::Data),
        _ => Err(CompatError::InvalidStreamSpecifier(format!(
            "unknown stream type: {}",
            s
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_specifier_parse_type() {
        assert_eq!(StreamSpecifier::parse("v").unwrap(), StreamSpecifier::Video);
        assert_eq!(StreamSpecifier::parse("a").unwrap(), StreamSpecifier::Audio);
        assert_eq!(
            StreamSpecifier::parse("s").unwrap(),
            StreamSpecifier::Subtitle
        );
    }

    #[test]
    fn test_stream_specifier_parse_index() {
        assert_eq!(
            StreamSpecifier::parse("0").unwrap(),
            StreamSpecifier::Index(0)
        );
        assert_eq!(
            StreamSpecifier::parse("1").unwrap(),
            StreamSpecifier::Index(1)
        );
    }

    #[test]
    fn test_stream_specifier_parse_type_index() {
        assert_eq!(
            StreamSpecifier::parse("v:0").unwrap(),
            StreamSpecifier::TypeIndex {
                stream_type: StreamType::Video,
                index: 0
            }
        );
        assert_eq!(
            StreamSpecifier::parse("a:1").unwrap(),
            StreamSpecifier::TypeIndex {
                stream_type: StreamType::Audio,
                index: 1
            }
        );
    }

    #[test]
    fn test_stream_specifier_parse_file_stream() {
        assert_eq!(
            StreamSpecifier::parse("0:1").unwrap(),
            StreamSpecifier::FileStream {
                file_index: 0,
                stream_index: 1
            }
        );
    }

    #[test]
    fn test_stream_specifier_parse_file_type_stream() {
        assert_eq!(
            StreamSpecifier::parse("0:v:0").unwrap(),
            StreamSpecifier::FileTypeStream {
                file_index: 0,
                stream_type: StreamType::Video,
                type_index: 0
            }
        );
        assert_eq!(
            StreamSpecifier::parse("1:a:2").unwrap(),
            StreamSpecifier::FileTypeStream {
                file_index: 1,
                stream_type: StreamType::Audio,
                type_index: 2
            }
        );
    }

    #[test]
    fn test_stream_specifier_matches() {
        let spec = StreamSpecifier::Video;
        assert!(spec.matches(0, 0, StreamType::Video));
        assert!(!spec.matches(0, 0, StreamType::Audio));

        let spec = StreamSpecifier::FileTypeStream {
            file_index: 0,
            stream_type: StreamType::Audio,
            type_index: 1,
        };
        assert!(spec.matches(0, 1, StreamType::Audio));
        assert!(!spec.matches(0, 0, StreamType::Audio));
        assert!(!spec.matches(1, 1, StreamType::Audio));
    }

    #[test]
    fn test_error_messages() {
        let err = CompatError::invalid_value("-b:v", "5X", "invalid suffix");
        assert!(err.to_string().contains("5X"));
        assert!(err.to_string().contains("-b:v"));
    }
}
