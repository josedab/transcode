//! Caption and subtitle handling for compliance.

use crate::{ComplianceError, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Caption format types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CaptionFormat {
    /// WebVTT format.
    WebVtt,
    /// SRT (SubRip) format.
    Srt,
    /// CEA-608 closed captions.
    Cea608,
    /// CEA-708 closed captions.
    Cea708,
    /// TTML (Timed Text Markup Language).
    Ttml,
    /// EBU-TT (European Broadcasting Union).
    EbuTt,
    /// IMSC (Internet Media Subtitles and Captions).
    Imsc,
    /// SSA/ASS (SubStation Alpha).
    Ssa,
}

impl CaptionFormat {
    /// Get file extension for format.
    pub fn extension(&self) -> &'static str {
        match self {
            CaptionFormat::WebVtt => "vtt",
            CaptionFormat::Srt => "srt",
            CaptionFormat::Cea608 => "scc",
            CaptionFormat::Cea708 => "scc",
            CaptionFormat::Ttml => "ttml",
            CaptionFormat::EbuTt => "xml",
            CaptionFormat::Imsc => "xml",
            CaptionFormat::Ssa => "ass",
        }
    }

    /// Check if format supports styling.
    pub fn supports_styling(&self) -> bool {
        matches!(
            self,
            CaptionFormat::WebVtt
                | CaptionFormat::Ttml
                | CaptionFormat::EbuTt
                | CaptionFormat::Imsc
                | CaptionFormat::Ssa
        )
    }

    /// Check if format supports positioning.
    pub fn supports_positioning(&self) -> bool {
        matches!(
            self,
            CaptionFormat::WebVtt
                | CaptionFormat::Cea608
                | CaptionFormat::Cea708
                | CaptionFormat::Ttml
                | CaptionFormat::EbuTt
                | CaptionFormat::Imsc
                | CaptionFormat::Ssa
        )
    }
}

/// A single caption cue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptionCue {
    /// Cue identifier.
    pub id: Option<String>,
    /// Start time.
    pub start: Duration,
    /// End time.
    pub end: Duration,
    /// Caption text.
    pub text: String,
    /// Speaker identification.
    pub speaker: Option<String>,
    /// Style settings.
    pub style: CaptionStyle,
    /// Position settings.
    pub position: CaptionPosition,
}

impl CaptionCue {
    /// Create a new caption cue.
    pub fn new(start: Duration, end: Duration, text: impl Into<String>) -> Self {
        Self {
            id: None,
            start,
            end,
            text: text.into(),
            speaker: None,
            style: CaptionStyle::default(),
            position: CaptionPosition::default(),
        }
    }

    /// Set cue ID.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set speaker.
    pub fn with_speaker(mut self, speaker: impl Into<String>) -> Self {
        self.speaker = Some(speaker.into());
        self
    }

    /// Set style.
    pub fn with_style(mut self, style: CaptionStyle) -> Self {
        self.style = style;
        self
    }

    /// Set position.
    pub fn with_position(mut self, position: CaptionPosition) -> Self {
        self.position = position;
        self
    }

    /// Get duration of this cue.
    pub fn duration(&self) -> Duration {
        self.end.saturating_sub(self.start)
    }

    /// Check if cue overlaps with a time range.
    pub fn overlaps(&self, start: Duration, end: Duration) -> bool {
        self.start < end && self.end > start
    }

    /// Calculate words per minute for this cue.
    pub fn words_per_minute(&self) -> f64 {
        let word_count = self.text.split_whitespace().count() as f64;
        let duration_mins = self.duration().as_secs_f64() / 60.0;
        if duration_mins > 0.0 {
            word_count / duration_mins
        } else {
            0.0
        }
    }

    /// Calculate characters per second.
    pub fn chars_per_second(&self) -> f64 {
        let char_count = self.text.chars().count() as f64;
        let duration_secs = self.duration().as_secs_f64();
        if duration_secs > 0.0 {
            char_count / duration_secs
        } else {
            0.0
        }
    }
}

/// Caption style settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CaptionStyle {
    /// Font family.
    pub font_family: Option<String>,
    /// Font size (as percentage).
    pub font_size: Option<f64>,
    /// Font color (CSS color string).
    pub color: Option<String>,
    /// Background color.
    pub background_color: Option<String>,
    /// Bold text.
    pub bold: bool,
    /// Italic text.
    pub italic: bool,
    /// Underlined text.
    pub underline: bool,
}

/// Caption position settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptionPosition {
    /// Vertical position (0.0 = top, 100.0 = bottom).
    pub vertical: f64,
    /// Horizontal alignment.
    pub align: CaptionAlign,
    /// Text direction.
    pub direction: TextDirection,
}

impl Default for CaptionPosition {
    fn default() -> Self {
        Self {
            vertical: 90.0, // Near bottom
            align: CaptionAlign::Center,
            direction: TextDirection::Ltr,
        }
    }
}

/// Caption horizontal alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CaptionAlign {
    Left,
    #[default]
    Center,
    Right,
}

/// Text direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TextDirection {
    #[default]
    Ltr,
    Rtl,
}

/// A complete caption track.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptionTrack {
    /// Track identifier.
    pub id: String,
    /// Language code (BCP 47).
    pub language: String,
    /// Track label.
    pub label: Option<String>,
    /// Caption format.
    pub format: CaptionFormat,
    /// Whether this is the default track.
    pub is_default: bool,
    /// Caption type.
    pub caption_type: CaptionType,
    /// All cues in this track.
    pub cues: Vec<CaptionCue>,
}

impl CaptionTrack {
    /// Create a new caption track.
    pub fn new(language: impl Into<String>, format: CaptionFormat) -> Self {
        let language = language.into();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            language: language.clone(),
            label: None,
            format,
            is_default: false,
            caption_type: CaptionType::Captions,
            cues: Vec::new(),
        }
    }

    /// Set track label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set as default track.
    pub fn as_default(mut self) -> Self {
        self.is_default = true;
        self
    }

    /// Set caption type.
    pub fn with_type(mut self, caption_type: CaptionType) -> Self {
        self.caption_type = caption_type;
        self
    }

    /// Add a cue.
    pub fn add_cue(&mut self, cue: CaptionCue) {
        self.cues.push(cue);
    }

    /// Add multiple cues.
    pub fn add_cues(&mut self, cues: impl IntoIterator<Item = CaptionCue>) {
        self.cues.extend(cues);
    }

    /// Sort cues by start time.
    pub fn sort_cues(&mut self) {
        self.cues.sort_by_key(|c| c.start);
    }

    /// Get total duration.
    pub fn duration(&self) -> Duration {
        self.cues
            .iter()
            .map(|c| c.end)
            .max()
            .unwrap_or(Duration::ZERO)
    }

    /// Get cue at a specific time.
    pub fn cue_at(&self, time: Duration) -> Option<&CaptionCue> {
        self.cues.iter().find(|c| c.start <= time && c.end > time)
    }

    /// Get all cues in a time range.
    pub fn cues_in_range(&self, start: Duration, end: Duration) -> Vec<&CaptionCue> {
        self.cues.iter().filter(|c| c.overlaps(start, end)).collect()
    }
}

/// Type of caption track.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CaptionType {
    /// Standard captions (for deaf/hard of hearing).
    #[default]
    Captions,
    /// Subtitles (translation).
    Subtitles,
    /// Descriptions (for blind/visually impaired).
    Descriptions,
    /// Chapters/navigation.
    Chapters,
    /// Metadata track.
    Metadata,
}

/// Caption converter for format conversion.
pub struct CaptionConverter;

impl CaptionConverter {
    /// Convert a caption track to a different format.
    pub fn convert(track: &CaptionTrack, target_format: CaptionFormat) -> Result<String> {
        match target_format {
            CaptionFormat::WebVtt => Self::to_webvtt(track),
            CaptionFormat::Srt => Self::to_srt(track),
            CaptionFormat::Ttml => Self::to_ttml(track),
            _ => Err(ComplianceError::UnsupportedFormat(format!(
                "Conversion to {:?} not yet implemented",
                target_format
            ))),
        }
    }

    /// Convert to WebVTT format.
    fn to_webvtt(track: &CaptionTrack) -> Result<String> {
        let mut output = String::from("WEBVTT\n");

        if let Some(label) = &track.label {
            output.push_str(&format!("NOTE {}\n", label));
        }
        output.push('\n');

        for cue in &track.cues {
            if let Some(id) = &cue.id {
                output.push_str(id);
                output.push('\n');
            }

            output.push_str(&format!(
                "{} --> {}\n",
                format_webvtt_time(cue.start),
                format_webvtt_time(cue.end)
            ));

            output.push_str(&cue.text);
            output.push_str("\n\n");
        }

        Ok(output)
    }

    /// Convert to SRT format.
    fn to_srt(track: &CaptionTrack) -> Result<String> {
        let mut output = String::new();

        for (i, cue) in track.cues.iter().enumerate() {
            output.push_str(&format!("{}\n", i + 1));
            output.push_str(&format!(
                "{} --> {}\n",
                format_srt_time(cue.start),
                format_srt_time(cue.end)
            ));
            output.push_str(&cue.text);
            output.push_str("\n\n");
        }

        Ok(output)
    }

    /// Convert to TTML format.
    fn to_ttml(track: &CaptionTrack) -> Result<String> {
        let mut output = String::from(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling" xml:lang=""#,
        );
        output.push_str(&track.language);
        output.push_str(
            r#"">
  <head>
    <styling>
      <style xml:id="defaultStyle" tts:fontFamily="proportionalSansSerif" tts:fontSize="100%" tts:textAlign="center"/>
    </styling>
  </head>
  <body>
    <div>
"#,
        );

        for cue in &track.cues {
            output.push_str(&format!(
                r#"      <p begin="{}" end="{}">{}</p>
"#,
                format_ttml_time(cue.start),
                format_ttml_time(cue.end),
                escape_xml(&cue.text)
            ));
        }

        output.push_str(
            r#"    </div>
  </body>
</tt>
"#,
        );

        Ok(output)
    }

    /// Parse WebVTT format.
    pub fn parse_webvtt(content: &str) -> Result<CaptionTrack> {
        let mut track = CaptionTrack::new("en", CaptionFormat::WebVtt);
        let lines: Vec<&str> = content.lines().collect();

        if lines.is_empty() || !lines[0].starts_with("WEBVTT") {
            return Err(ComplianceError::InvalidCaptionFormat(
                "Missing WEBVTT header".into(),
            ));
        }

        let mut i = 1;
        while i < lines.len() {
            // Skip empty lines and notes
            if lines[i].is_empty() || lines[i].starts_with("NOTE") {
                i += 1;
                continue;
            }

            // Check for timing line
            let timing_idx = if lines[i].contains("-->") {
                i
            } else if i + 1 < lines.len() && lines[i + 1].contains("-->") {
                i + 1
            } else {
                i += 1;
                continue;
            };

            // Parse timing
            let cue_id = if timing_idx > i {
                Some(lines[i].to_string())
            } else {
                None
            };

            let timing_parts: Vec<&str> = lines[timing_idx].split("-->").collect();
            if timing_parts.len() != 2 {
                i += 1;
                continue;
            }

            let start = parse_webvtt_time(timing_parts[0].trim())?;
            let end_part = timing_parts[1].split_whitespace().next().unwrap_or("");
            let end = parse_webvtt_time(end_part)?;

            // Collect text lines
            let mut text_lines = Vec::new();
            i = timing_idx + 1;
            while i < lines.len() && !lines[i].is_empty() {
                text_lines.push(lines[i]);
                i += 1;
            }

            let mut cue = CaptionCue::new(start, end, text_lines.join("\n"));
            if let Some(id) = cue_id {
                cue = cue.with_id(id);
            }
            track.add_cue(cue);
        }

        Ok(track)
    }

    /// Parse SRT format.
    pub fn parse_srt(content: &str) -> Result<CaptionTrack> {
        let mut track = CaptionTrack::new("en", CaptionFormat::Srt);
        let blocks: Vec<&str> = content.split("\n\n").collect();

        for block in blocks {
            let lines: Vec<&str> = block.lines().collect();
            if lines.len() < 3 {
                continue;
            }

            // Skip sequence number (lines[0])
            let timing_parts: Vec<&str> = lines[1].split("-->").collect();
            if timing_parts.len() != 2 {
                continue;
            }

            let start = parse_srt_time(timing_parts[0].trim())?;
            let end = parse_srt_time(timing_parts[1].trim())?;
            let text = lines[2..].join("\n");

            track.add_cue(CaptionCue::new(start, end, text));
        }

        Ok(track)
    }
}

fn format_webvtt_time(d: Duration) -> String {
    let total_secs = d.as_secs();
    let hours = total_secs / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;
    let millis = d.subsec_millis();

    format!("{:02}:{:02}:{:02}.{:03}", hours, mins, secs, millis)
}

fn format_srt_time(d: Duration) -> String {
    let total_secs = d.as_secs();
    let hours = total_secs / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;
    let millis = d.subsec_millis();

    format!("{:02}:{:02}:{:02},{:03}", hours, mins, secs, millis)
}

fn format_ttml_time(d: Duration) -> String {
    let total_secs = d.as_secs();
    let hours = total_secs / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;
    let millis = d.subsec_millis();

    format!("{:02}:{:02}:{:02}.{:03}", hours, mins, secs, millis)
}

fn parse_webvtt_time(s: &str) -> Result<Duration> {
    parse_time_string(s, '.')
}

fn parse_srt_time(s: &str) -> Result<Duration> {
    parse_time_string(s, ',')
}

fn parse_time_string(s: &str, ms_sep: char) -> Result<Duration> {
    let parts: Vec<&str> = s.split(ms_sep).collect();
    if parts.len() != 2 {
        return Err(ComplianceError::ParseError(format!(
            "Invalid time format: {}",
            s
        )));
    }

    let time_parts: Vec<&str> = parts[0].split(':').collect();
    if time_parts.len() < 2 {
        return Err(ComplianceError::ParseError(format!(
            "Invalid time format: {}",
            s
        )));
    }

    let (hours, mins, secs) = if time_parts.len() == 3 {
        (
            time_parts[0]
                .parse::<u64>()
                .map_err(|_| ComplianceError::ParseError("Invalid hours".into()))?,
            time_parts[1]
                .parse::<u64>()
                .map_err(|_| ComplianceError::ParseError("Invalid minutes".into()))?,
            time_parts[2]
                .parse::<u64>()
                .map_err(|_| ComplianceError::ParseError("Invalid seconds".into()))?,
        )
    } else {
        (
            0,
            time_parts[0]
                .parse::<u64>()
                .map_err(|_| ComplianceError::ParseError("Invalid minutes".into()))?,
            time_parts[1]
                .parse::<u64>()
                .map_err(|_| ComplianceError::ParseError("Invalid seconds".into()))?,
        )
    };

    let millis = parts[1]
        .parse::<u64>()
        .map_err(|_| ComplianceError::ParseError("Invalid milliseconds".into()))?;

    Ok(Duration::from_secs(hours * 3600 + mins * 60 + secs) + Duration::from_millis(millis))
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_caption_cue() {
        let cue = CaptionCue::new(
            Duration::from_secs(1),
            Duration::from_secs(3),
            "Hello, world!",
        );

        assert_eq!(cue.duration(), Duration::from_secs(2));
        assert!(cue.overlaps(Duration::from_secs(2), Duration::from_secs(4)));
        assert!(!cue.overlaps(Duration::from_secs(5), Duration::from_secs(6)));
    }

    #[test]
    fn test_webvtt_conversion() {
        let mut track = CaptionTrack::new("en", CaptionFormat::WebVtt);
        track.add_cue(CaptionCue::new(
            Duration::from_secs(0),
            Duration::from_secs(2),
            "Hello",
        ));
        track.add_cue(CaptionCue::new(
            Duration::from_secs(2),
            Duration::from_secs(4),
            "World",
        ));

        let vtt = CaptionConverter::to_webvtt(&track).unwrap();
        assert!(vtt.starts_with("WEBVTT"));
        assert!(vtt.contains("00:00:00.000 --> 00:00:02.000"));
        assert!(vtt.contains("Hello"));
    }

    #[test]
    fn test_srt_conversion() {
        let mut track = CaptionTrack::new("en", CaptionFormat::Srt);
        track.add_cue(CaptionCue::new(
            Duration::from_secs(0),
            Duration::from_secs(2),
            "Hello",
        ));

        let srt = CaptionConverter::to_srt(&track).unwrap();
        assert!(srt.contains("00:00:00,000 --> 00:00:02,000"));
    }

    #[test]
    fn test_parse_webvtt() {
        let vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nHello\n\n00:00:02.000 --> 00:00:04.000\nWorld";
        let track = CaptionConverter::parse_webvtt(vtt).unwrap();
        assert_eq!(track.cues.len(), 2);
        assert_eq!(track.cues[0].text, "Hello");
    }

    #[test]
    fn test_wpm_calculation() {
        let cue = CaptionCue::new(
            Duration::from_secs(0),
            Duration::from_secs(60),
            "one two three four five",
        );
        assert_eq!(cue.words_per_minute(), 5.0);
    }
}
