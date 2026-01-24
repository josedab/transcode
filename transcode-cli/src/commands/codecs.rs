//! List available codecs command.

use clap::Args;
use console::style;
use serde::Serialize;

/// Information about a codec.
#[derive(Debug, Clone, Serialize)]
pub struct CodecEntry {
    /// Codec name/identifier.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Codec type (video/audio).
    #[serde(rename = "type")]
    pub codec_type: String,
    /// Whether encoding is supported.
    pub encode: bool,
    /// Whether decoding is supported.
    pub decode: bool,
}

/// List available codecs.
#[derive(Args, Debug)]
pub struct CmdCodecs {
    /// Output in JSON format.
    #[arg(long)]
    pub json: bool,

    /// Filter by codec type (video, audio).
    #[arg(long)]
    pub filter: Option<String>,
}

impl CmdCodecs {
    /// Execute the codecs command.
    pub fn run(&self) -> anyhow::Result<()> {
        let codecs = get_available_codecs();

        // Filter if requested
        let filtered: Vec<_> = if let Some(ref filter) = self.filter {
            codecs
                .into_iter()
                .filter(|c| c.codec_type.eq_ignore_ascii_case(filter))
                .collect()
        } else {
            codecs
        };

        if self.json {
            let output = serde_json::json!({
                "codecs": filtered
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        } else {
            println!();
            println!("{}", style("Available Codecs").cyan().bold());
            println!();

            // Video codecs
            println!("{}", style("Video Codecs:").white().bold());
            for codec in filtered.iter().filter(|c| c.codec_type == "video") {
                let encode_status = if codec.encode {
                    style("E").green()
                } else {
                    style("-").dim()
                };
                let decode_status = if codec.decode {
                    style("D").green()
                } else {
                    style("-").dim()
                };
                println!(
                    "  {} [{}{}] {}",
                    style(&codec.name).yellow(),
                    encode_status,
                    decode_status,
                    codec.description
                );
            }

            println!();
            println!("{}", style("Audio Codecs:").white().bold());
            for codec in filtered.iter().filter(|c| c.codec_type == "audio") {
                let encode_status = if codec.encode {
                    style("E").green()
                } else {
                    style("-").dim()
                };
                let decode_status = if codec.decode {
                    style("D").green()
                } else {
                    style("-").dim()
                };
                println!(
                    "  {} [{}{}] {}",
                    style(&codec.name).yellow(),
                    encode_status,
                    decode_status,
                    codec.description
                );
            }

            println!();
            println!(
                "{}: {} = Encode, {} = Decode",
                style("Legend").dim(),
                style("E").green(),
                style("D").green()
            );
        }

        Ok(())
    }
}

/// Get list of available codecs.
fn get_available_codecs() -> Vec<CodecEntry> {
    vec![
        // Video codecs
        CodecEntry {
            name: "h264".to_string(),
            description: "H.264/AVC - Advanced Video Coding".to_string(),
            codec_type: "video".to_string(),
            encode: true,
            decode: true,
        },
        CodecEntry {
            name: "h265".to_string(),
            description: "H.265/HEVC - High Efficiency Video Coding".to_string(),
            codec_type: "video".to_string(),
            encode: true,
            decode: true,
        },
        CodecEntry {
            name: "av1".to_string(),
            description: "AV1 - AOMedia Video 1".to_string(),
            codec_type: "video".to_string(),
            encode: true,
            decode: true,
        },
        CodecEntry {
            name: "vp9".to_string(),
            description: "VP9 - Google VP9".to_string(),
            codec_type: "video".to_string(),
            encode: false,
            decode: true,
        },
        // Audio codecs
        CodecEntry {
            name: "aac".to_string(),
            description: "AAC - Advanced Audio Coding".to_string(),
            codec_type: "audio".to_string(),
            encode: true,
            decode: true,
        },
        CodecEntry {
            name: "mp3".to_string(),
            description: "MP3 - MPEG Audio Layer III".to_string(),
            codec_type: "audio".to_string(),
            encode: false,
            decode: true,
        },
        CodecEntry {
            name: "opus".to_string(),
            description: "Opus - Interactive Audio Codec".to_string(),
            codec_type: "audio".to_string(),
            encode: true,
            decode: true,
        },
        CodecEntry {
            name: "flac".to_string(),
            description: "FLAC - Free Lossless Audio Codec".to_string(),
            codec_type: "audio".to_string(),
            encode: false,
            decode: true,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_available_codecs() {
        let codecs = get_available_codecs();
        assert!(!codecs.is_empty());

        // Check video codecs exist
        assert!(codecs.iter().any(|c| c.name == "h264"));
        assert!(codecs.iter().any(|c| c.name == "h265"));

        // Check audio codecs exist
        assert!(codecs.iter().any(|c| c.name == "aac"));
        assert!(codecs.iter().any(|c| c.name == "mp3"));
    }

    #[test]
    fn test_codec_entry_serialization() {
        let codec = CodecEntry {
            name: "h264".to_string(),
            description: "Test".to_string(),
            codec_type: "video".to_string(),
            encode: true,
            decode: true,
        };
        let json = serde_json::to_string(&codec).unwrap();
        assert!(json.contains("\"name\":\"h264\""));
        assert!(json.contains("\"type\":\"video\""));
    }
}
