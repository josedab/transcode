//! Show encoding presets command.

use clap::Args;
use console::style;
use serde::Serialize;

/// Preset information for display.
#[derive(Debug, Clone, Serialize)]
pub struct PresetInfo {
    /// Preset name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Output format.
    pub format: String,
    /// Video codec (if applicable).
    pub video_codec: Option<String>,
    /// Audio codec.
    pub audio_codec: String,
    /// Quality level.
    pub quality: String,
    /// Encoding speed preset.
    pub speed: String,
}

/// Show encoding presets.
#[derive(Args, Debug)]
pub struct CmdPresets {
    /// Show detailed info for a specific preset.
    #[arg(long)]
    pub show: Option<String>,

    /// Output in JSON format.
    #[arg(long)]
    pub json: bool,
}

impl CmdPresets {
    /// Execute the presets command.
    pub fn run(&self) -> anyhow::Result<()> {
        let presets = get_available_presets();

        if let Some(ref name) = self.show {
            // Show specific preset
            if let Some(preset) = presets.iter().find(|p| p.name.eq_ignore_ascii_case(name)) {
                if self.json {
                    println!("{}", serde_json::to_string_pretty(preset)?);
                } else {
                    print_preset_details(preset);
                }
            } else {
                anyhow::bail!("Unknown preset: {}. Run 'transcode presets' to see available presets.", name);
            }
        } else if self.json {
            let output = serde_json::json!({
                "presets": presets
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        } else {
            println!();
            println!("{}", style("Encoding Presets").cyan().bold());
            println!();
            println!(
                "{:<16} {:<12} {:<8} {}",
                style("NAME").white().bold(),
                style("FORMAT").white().bold(),
                style("QUALITY").white().bold(),
                style("DESCRIPTION").white().bold()
            );
            println!("{}", style("-".repeat(70)).dim());

            for preset in &presets {
                println!(
                    "{:<16} {:<12} {:<8} {}",
                    style(&preset.name).yellow(),
                    preset.format,
                    preset.quality,
                    preset.description
                );
            }

            println!();
            println!(
                "Use {} to see details for a specific preset.",
                style("--show <preset>").cyan()
            );
        }

        Ok(())
    }
}

fn print_preset_details(preset: &PresetInfo) {
    println!();
    println!(
        "{}: {}",
        style("Preset").cyan().bold(),
        style(&preset.name).yellow()
    );
    println!();
    println!("  {:<16} {}", style("Description:").white(), preset.description);
    println!("  {:<16} {}", style("Format:").white(), preset.format);
    println!("  {:<16} {}", style("Quality:").white(), preset.quality);
    println!("  {:<16} {}", style("Speed:").white(), preset.speed);

    if let Some(ref video) = preset.video_codec {
        println!("  {:<16} {}", style("Video Codec:").white(), video);
    }
    println!("  {:<16} {}", style("Audio Codec:").white(), preset.audio_codec);
    println!();
}

/// Get available encoding presets.
fn get_available_presets() -> Vec<PresetInfo> {
    vec![
        PresetInfo {
            name: "web_streaming".to_string(),
            description: "Optimized for web streaming".to_string(),
            format: "mp4".to_string(),
            video_codec: Some("h264".to_string()),
            audio_codec: "aac".to_string(),
            quality: "medium".to_string(),
            speed: "fast".to_string(),
        },
        PresetInfo {
            name: "archive".to_string(),
            description: "High quality archival".to_string(),
            format: "mkv".to_string(),
            video_codec: Some("h265".to_string()),
            audio_codec: "flac".to_string(),
            quality: "veryhigh".to_string(),
            speed: "slow".to_string(),
        },
        PresetInfo {
            name: "mobile".to_string(),
            description: "Optimized for mobile devices".to_string(),
            format: "mp4".to_string(),
            video_codec: Some("h264".to_string()),
            audio_codec: "aac".to_string(),
            quality: "medium".to_string(),
            speed: "veryfast".to_string(),
        },
        PresetInfo {
            name: "podcast".to_string(),
            description: "Voice/podcast content".to_string(),
            format: "mp3".to_string(),
            video_codec: None,
            audio_codec: "mp3".to_string(),
            quality: "medium".to_string(),
            speed: "medium".to_string(),
        },
        PresetInfo {
            name: "music".to_string(),
            description: "Music streaming".to_string(),
            format: "opus".to_string(),
            video_codec: None,
            audio_codec: "opus".to_string(),
            quality: "high".to_string(),
            speed: "medium".to_string(),
        },
        PresetInfo {
            name: "4k_uhd".to_string(),
            description: "4K Ultra HD content".to_string(),
            format: "mp4".to_string(),
            video_codec: Some("h265".to_string()),
            audio_codec: "aac".to_string(),
            quality: "veryhigh".to_string(),
            speed: "medium".to_string(),
        },
        PresetInfo {
            name: "social".to_string(),
            description: "Social media sharing".to_string(),
            format: "mp4".to_string(),
            video_codec: Some("h264".to_string()),
            audio_codec: "aac".to_string(),
            quality: "medium".to_string(),
            speed: "faster".to_string(),
        },
        PresetInfo {
            name: "hls".to_string(),
            description: "HLS adaptive streaming".to_string(),
            format: "ts".to_string(),
            video_codec: Some("h264".to_string()),
            audio_codec: "aac".to_string(),
            quality: "medium".to_string(),
            speed: "fast".to_string(),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_available_presets() {
        let presets = get_available_presets();
        assert!(!presets.is_empty());
        assert!(presets.iter().any(|p| p.name == "web_streaming"));
        assert!(presets.iter().any(|p| p.name == "archive"));
    }

    #[test]
    fn test_preset_serialization() {
        let preset = PresetInfo {
            name: "test".to_string(),
            description: "Test preset".to_string(),
            format: "mp4".to_string(),
            video_codec: Some("h264".to_string()),
            audio_codec: "aac".to_string(),
            quality: "medium".to_string(),
            speed: "fast".to_string(),
        };
        let json = serde_json::to_string(&preset).unwrap();
        assert!(json.contains("\"name\":\"test\""));
    }
}
