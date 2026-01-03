//! Subtitle format support

use crate::Transcription;
use serde::{Deserialize, Serialize};

/// Subtitle format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleFormat {
    /// SubRip (.srt)
    Srt,
    /// WebVTT (.vtt)
    Vtt,
    /// SubStation Alpha (.ass/.ssa)
    Ass,
    /// TTML/IMSC1 (.ttml)
    Ttml,
    /// JSON format
    Json,
}

impl SubtitleFormat {
    /// Get file extension
    pub fn extension(&self) -> &str {
        match self {
            SubtitleFormat::Srt => "srt",
            SubtitleFormat::Vtt => "vtt",
            SubtitleFormat::Ass => "ass",
            SubtitleFormat::Ttml => "ttml",
            SubtitleFormat::Json => "json",
        }
    }

    /// Get MIME type
    pub fn mime_type(&self) -> &str {
        match self {
            SubtitleFormat::Srt => "text/plain",
            SubtitleFormat::Vtt => "text/vtt",
            SubtitleFormat::Ass => "text/plain",
            SubtitleFormat::Ttml => "application/ttml+xml",
            SubtitleFormat::Json => "application/json",
        }
    }
}

/// Subtitle styling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtitleStyle {
    /// Font name
    pub font_name: String,
    /// Font size
    pub font_size: u32,
    /// Primary color (RGBA)
    pub primary_color: [u8; 4],
    /// Background color (RGBA)
    pub background_color: [u8; 4],
    /// Bold
    pub bold: bool,
    /// Italic
    pub italic: bool,
    /// Outline width
    pub outline: u32,
    /// Shadow offset
    pub shadow: u32,
}

impl Default for SubtitleStyle {
    fn default() -> Self {
        Self {
            font_name: "Arial".into(),
            font_size: 24,
            primary_color: [255, 255, 255, 255],
            background_color: [0, 0, 0, 128],
            bold: false,
            italic: false,
            outline: 2,
            shadow: 1,
        }
    }
}

/// Subtitle exporter
pub struct SubtitleExporter {
    format: SubtitleFormat,
    style: SubtitleStyle,
}

impl SubtitleExporter {
    /// Create a new exporter
    pub fn new(format: SubtitleFormat) -> Self {
        Self {
            format,
            style: SubtitleStyle::default(),
        }
    }

    /// Set styling
    pub fn with_style(mut self, style: SubtitleStyle) -> Self {
        self.style = style;
        self
    }

    /// Export transcription
    pub fn export(&self, transcription: &Transcription) -> String {
        match self.format {
            SubtitleFormat::Srt => transcription.to_srt(),
            SubtitleFormat::Vtt => transcription.to_vtt(),
            SubtitleFormat::Ass => self.export_ass(transcription),
            SubtitleFormat::Ttml => self.export_ttml(transcription),
            SubtitleFormat::Json => self.export_json(transcription),
        }
    }

    fn export_ass(&self, transcription: &Transcription) -> String {
        let mut ass = String::new();

        // Script info
        ass.push_str("[Script Info]\n");
        ass.push_str("ScriptType: v4.00+\n");
        ass.push_str("PlayResX: 1920\n");
        ass.push_str("PlayResY: 1080\n\n");

        // Styles
        ass.push_str("[V4+ Styles]\n");
        ass.push_str("Format: Name, Fontname, Fontsize, PrimaryColour, BackColour, Bold, Italic, Outline, Shadow, Alignment\n");
        ass.push_str(&format!(
            "Style: Default,{},{},&H00FFFFFF,&H80000000,{},{},{},{},2\n\n",
            self.style.font_name,
            self.style.font_size,
            self.style.bold as u32,
            self.style.italic as u32,
            self.style.outline,
            self.style.shadow,
        ));

        // Events
        ass.push_str("[Events]\n");
        ass.push_str("Format: Layer, Start, End, Style, Text\n");

        for segment in &transcription.segments {
            ass.push_str(&format!(
                "Dialogue: 0,{},{},Default,{}\n",
                format_ass_time(segment.start_ms),
                format_ass_time(segment.end_ms),
                segment.text,
            ));
        }

        ass
    }

    fn export_ttml(&self, transcription: &Transcription) -> String {
        let mut ttml = String::new();

        ttml.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
        ttml.push('\n');
        ttml.push_str(r#"<tt xmlns="http://www.w3.org/ns/ttml">"#);
        ttml.push('\n');
        ttml.push_str("  <body>\n");
        ttml.push_str("    <div>\n");

        for segment in &transcription.segments {
            ttml.push_str(&format!(
                "      <p begin=\"{}\" end=\"{}\">{}</p>\n",
                format_ttml_time(segment.start_ms),
                format_ttml_time(segment.end_ms),
                escape_xml(&segment.text),
            ));
        }

        ttml.push_str("    </div>\n");
        ttml.push_str("  </body>\n");
        ttml.push_str("</tt>\n");

        ttml
    }

    fn export_json(&self, transcription: &Transcription) -> String {
        serde_json::to_string_pretty(transcription).unwrap_or_default()
    }
}

fn format_ass_time(ms: u64) -> String {
    let hours = ms / 3600000;
    let minutes = (ms % 3600000) / 60000;
    let seconds = (ms % 60000) / 1000;
    let centis = (ms % 1000) / 10;

    format!("{}:{:02}:{:02}.{:02}", hours, minutes, seconds, centis)
}

fn format_ttml_time(ms: u64) -> String {
    let hours = ms / 3600000;
    let minutes = (ms % 3600000) / 60000;
    let seconds = (ms % 60000) / 1000;
    let millis = ms % 1000;

    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
