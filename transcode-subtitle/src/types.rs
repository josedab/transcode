//! Common subtitle types and structures.
//!
//! This module defines the core types used across all subtitle formats,
//! including timing, styling, and event structures.

use std::fmt;
use thiserror::Error;

/// Errors that can occur when working with subtitle types.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum SubtitleError {
    #[error("Invalid timestamp format: {0}")]
    InvalidTimestamp(String),

    #[error("Invalid color format: {0}")]
    InvalidColor(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    #[error("IO error: {0}")]
    IoError(String),
}

/// Result type for subtitle operations.
pub type SubtitleResult<T> = Result<T, SubtitleError>;

/// Represents a timestamp with millisecond precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Timestamp {
    /// Total milliseconds from the start.
    pub milliseconds: u64,
}

impl Timestamp {
    /// Creates a new timestamp from hours, minutes, seconds, and milliseconds.
    pub fn new(hours: u64, minutes: u64, seconds: u64, milliseconds: u64) -> Self {
        Self {
            milliseconds: hours * 3_600_000 + minutes * 60_000 + seconds * 1000 + milliseconds,
        }
    }

    /// Creates a new timestamp from total milliseconds.
    pub fn from_millis(milliseconds: u64) -> Self {
        Self { milliseconds }
    }

    /// Returns the hours component.
    pub fn hours(&self) -> u64 {
        self.milliseconds / 3_600_000
    }

    /// Returns the minutes component (0-59).
    pub fn minutes(&self) -> u64 {
        (self.milliseconds % 3_600_000) / 60_000
    }

    /// Returns the seconds component (0-59).
    pub fn seconds(&self) -> u64 {
        (self.milliseconds % 60_000) / 1000
    }

    /// Returns the milliseconds component (0-999).
    pub fn millis(&self) -> u64 {
        self.milliseconds % 1000
    }

    /// Returns the total duration in seconds as a float.
    pub fn as_seconds_f64(&self) -> f64 {
        self.milliseconds as f64 / 1000.0
    }

    /// Parses a timestamp in SRT format (HH:MM:SS,mmm).
    pub fn parse_srt(s: &str) -> SubtitleResult<Self> {
        let s = s.trim();
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 3 {
            return Err(SubtitleError::InvalidTimestamp(format!(
                "Expected HH:MM:SS,mmm format, got: {}",
                s
            )));
        }

        let hours: u64 = parts[0]
            .parse()
            .map_err(|_| SubtitleError::InvalidTimestamp(format!("Invalid hours: {}", parts[0])))?;
        let minutes: u64 = parts[1]
            .parse()
            .map_err(|_| SubtitleError::InvalidTimestamp(format!("Invalid minutes: {}", parts[1])))?;

        // Handle seconds and milliseconds (separated by comma or period)
        let sec_parts: Vec<&str> = parts[2].split([',', '.']).collect();
        if sec_parts.len() != 2 {
            return Err(SubtitleError::InvalidTimestamp(format!(
                "Invalid seconds format: {}",
                parts[2]
            )));
        }

        let seconds: u64 = sec_parts[0].parse().map_err(|_| {
            SubtitleError::InvalidTimestamp(format!("Invalid seconds: {}", sec_parts[0]))
        })?;
        let millis: u64 = sec_parts[1].parse().map_err(|_| {
            SubtitleError::InvalidTimestamp(format!("Invalid milliseconds: {}", sec_parts[1]))
        })?;

        Ok(Self::new(hours, minutes, seconds, millis))
    }

    /// Formats the timestamp in SRT format (HH:MM:SS,mmm).
    pub fn to_srt_string(&self) -> String {
        format!(
            "{:02}:{:02}:{:02},{:03}",
            self.hours(),
            self.minutes(),
            self.seconds(),
            self.millis()
        )
    }

    /// Parses a timestamp in WebVTT format (HH:MM:SS.mmm or MM:SS.mmm).
    pub fn parse_vtt(s: &str) -> SubtitleResult<Self> {
        let s = s.trim();
        let parts: Vec<&str> = s.split(':').collect();

        match parts.len() {
            2 => {
                // MM:SS.mmm format
                let minutes: u64 = parts[0].parse().map_err(|_| {
                    SubtitleError::InvalidTimestamp(format!("Invalid minutes: {}", parts[0]))
                })?;

                let sec_parts: Vec<&str> = parts[1].split('.').collect();
                if sec_parts.len() != 2 {
                    return Err(SubtitleError::InvalidTimestamp(format!(
                        "Invalid seconds format: {}",
                        parts[1]
                    )));
                }

                let seconds: u64 = sec_parts[0].parse().map_err(|_| {
                    SubtitleError::InvalidTimestamp(format!("Invalid seconds: {}", sec_parts[0]))
                })?;
                let millis: u64 = sec_parts[1].parse().map_err(|_| {
                    SubtitleError::InvalidTimestamp(format!("Invalid milliseconds: {}", sec_parts[1]))
                })?;

                Ok(Self::new(0, minutes, seconds, millis))
            }
            3 => {
                // HH:MM:SS.mmm format
                let hours: u64 = parts[0].parse().map_err(|_| {
                    SubtitleError::InvalidTimestamp(format!("Invalid hours: {}", parts[0]))
                })?;
                let minutes: u64 = parts[1].parse().map_err(|_| {
                    SubtitleError::InvalidTimestamp(format!("Invalid minutes: {}", parts[1]))
                })?;

                let sec_parts: Vec<&str> = parts[2].split('.').collect();
                if sec_parts.len() != 2 {
                    return Err(SubtitleError::InvalidTimestamp(format!(
                        "Invalid seconds format: {}",
                        parts[2]
                    )));
                }

                let seconds: u64 = sec_parts[0].parse().map_err(|_| {
                    SubtitleError::InvalidTimestamp(format!("Invalid seconds: {}", sec_parts[0]))
                })?;
                let millis: u64 = sec_parts[1].parse().map_err(|_| {
                    SubtitleError::InvalidTimestamp(format!("Invalid milliseconds: {}", sec_parts[1]))
                })?;

                Ok(Self::new(hours, minutes, seconds, millis))
            }
            _ => Err(SubtitleError::InvalidTimestamp(format!(
                "Expected MM:SS.mmm or HH:MM:SS.mmm format, got: {}",
                s
            ))),
        }
    }

    /// Formats the timestamp in WebVTT format (HH:MM:SS.mmm).
    pub fn to_vtt_string(&self) -> String {
        format!(
            "{:02}:{:02}:{:02}.{:03}",
            self.hours(),
            self.minutes(),
            self.seconds(),
            self.millis()
        )
    }

    /// Parses a timestamp in ASS format (H:MM:SS.cc where cc is centiseconds).
    pub fn parse_ass(s: &str) -> SubtitleResult<Self> {
        let s = s.trim();
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 3 {
            return Err(SubtitleError::InvalidTimestamp(format!(
                "Expected H:MM:SS.cc format, got: {}",
                s
            )));
        }

        let hours: u64 = parts[0]
            .parse()
            .map_err(|_| SubtitleError::InvalidTimestamp(format!("Invalid hours: {}", parts[0])))?;
        let minutes: u64 = parts[1]
            .parse()
            .map_err(|_| SubtitleError::InvalidTimestamp(format!("Invalid minutes: {}", parts[1])))?;

        let sec_parts: Vec<&str> = parts[2].split('.').collect();
        if sec_parts.len() != 2 {
            return Err(SubtitleError::InvalidTimestamp(format!(
                "Invalid seconds format: {}",
                parts[2]
            )));
        }

        let seconds: u64 = sec_parts[0].parse().map_err(|_| {
            SubtitleError::InvalidTimestamp(format!("Invalid seconds: {}", sec_parts[0]))
        })?;
        // ASS uses centiseconds (1/100th of a second)
        let centiseconds: u64 = sec_parts[1].parse().map_err(|_| {
            SubtitleError::InvalidTimestamp(format!("Invalid centiseconds: {}", sec_parts[1]))
        })?;

        Ok(Self::new(hours, minutes, seconds, centiseconds * 10))
    }

    /// Formats the timestamp in ASS format (H:MM:SS.cc).
    pub fn to_ass_string(&self) -> String {
        format!(
            "{}:{:02}:{:02}.{:02}",
            self.hours(),
            self.minutes(),
            self.seconds(),
            self.millis() / 10
        )
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_srt_string())
    }
}

/// Represents an RGBA color.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    /// Creates a new color with full opacity.
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// Creates a new color with specified alpha.
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Predefined white color.
    pub const WHITE: Color = Color::rgb(255, 255, 255);

    /// Predefined black color.
    pub const BLACK: Color = Color::rgb(0, 0, 0);

    /// Predefined red color.
    pub const RED: Color = Color::rgb(255, 0, 0);

    /// Predefined green color.
    pub const GREEN: Color = Color::rgb(0, 255, 0);

    /// Predefined blue color.
    pub const BLUE: Color = Color::rgb(0, 0, 255);

    /// Predefined yellow color.
    pub const YELLOW: Color = Color::rgb(255, 255, 0);

    /// Predefined cyan color.
    pub const CYAN: Color = Color::rgb(0, 255, 255);

    /// Predefined magenta color.
    pub const MAGENTA: Color = Color::rgb(255, 0, 255);

    /// Parses a hex color string (#RGB, #RRGGBB, or #RRGGBBAA).
    pub fn from_hex(s: &str) -> SubtitleResult<Self> {
        let s = s.trim().trim_start_matches('#');

        match s.len() {
            3 => {
                // #RGB format
                let r = u8::from_str_radix(&s[0..1], 16)
                    .map_err(|_| SubtitleError::InvalidColor(s.to_string()))?;
                let g = u8::from_str_radix(&s[1..2], 16)
                    .map_err(|_| SubtitleError::InvalidColor(s.to_string()))?;
                let b = u8::from_str_radix(&s[2..3], 16)
                    .map_err(|_| SubtitleError::InvalidColor(s.to_string()))?;
                Ok(Self::rgb(r * 17, g * 17, b * 17))
            }
            6 => {
                // #RRGGBB format
                let r = u8::from_str_radix(&s[0..2], 16)
                    .map_err(|_| SubtitleError::InvalidColor(s.to_string()))?;
                let g = u8::from_str_radix(&s[2..4], 16)
                    .map_err(|_| SubtitleError::InvalidColor(s.to_string()))?;
                let b = u8::from_str_radix(&s[4..6], 16)
                    .map_err(|_| SubtitleError::InvalidColor(s.to_string()))?;
                Ok(Self::rgb(r, g, b))
            }
            8 => {
                // #RRGGBBAA format
                let r = u8::from_str_radix(&s[0..2], 16)
                    .map_err(|_| SubtitleError::InvalidColor(s.to_string()))?;
                let g = u8::from_str_radix(&s[2..4], 16)
                    .map_err(|_| SubtitleError::InvalidColor(s.to_string()))?;
                let b = u8::from_str_radix(&s[4..6], 16)
                    .map_err(|_| SubtitleError::InvalidColor(s.to_string()))?;
                let a = u8::from_str_radix(&s[6..8], 16)
                    .map_err(|_| SubtitleError::InvalidColor(s.to_string()))?;
                Ok(Self::rgba(r, g, b, a))
            }
            _ => Err(SubtitleError::InvalidColor(format!(
                "Invalid hex color format: {}",
                s
            ))),
        }
    }

    /// Converts the color to a hex string (#RRGGBB or #RRGGBBAA if alpha < 255).
    pub fn to_hex(&self) -> String {
        if self.a == 255 {
            format!("#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
        } else {
            format!("#{:02X}{:02X}{:02X}{:02X}", self.r, self.g, self.b, self.a)
        }
    }

    /// Parses ASS color format (&HAABBGGRR or &HBBGGRR).
    pub fn from_ass(s: &str) -> SubtitleResult<Self> {
        let s = s.trim().trim_start_matches("&H").trim_start_matches("&h");
        let s = s.trim_end_matches('&');

        match s.len() {
            6 => {
                // &HBBGGRR format (no alpha, assume fully opaque)
                let b = u8::from_str_radix(&s[0..2], 16)
                    .map_err(|_| SubtitleError::InvalidColor(s.to_string()))?;
                let g = u8::from_str_radix(&s[2..4], 16)
                    .map_err(|_| SubtitleError::InvalidColor(s.to_string()))?;
                let r = u8::from_str_radix(&s[4..6], 16)
                    .map_err(|_| SubtitleError::InvalidColor(s.to_string()))?;
                Ok(Self::rgb(r, g, b))
            }
            8 => {
                // &HAABBGGRR format
                let a = u8::from_str_radix(&s[0..2], 16)
                    .map_err(|_| SubtitleError::InvalidColor(s.to_string()))?;
                let b = u8::from_str_radix(&s[2..4], 16)
                    .map_err(|_| SubtitleError::InvalidColor(s.to_string()))?;
                let g = u8::from_str_radix(&s[4..6], 16)
                    .map_err(|_| SubtitleError::InvalidColor(s.to_string()))?;
                let r = u8::from_str_radix(&s[6..8], 16)
                    .map_err(|_| SubtitleError::InvalidColor(s.to_string()))?;
                // ASS alpha is inverted (0 = opaque, 255 = transparent)
                Ok(Self::rgba(r, g, b, 255 - a))
            }
            _ => Err(SubtitleError::InvalidColor(format!(
                "Invalid ASS color format: {}",
                s
            ))),
        }
    }

    /// Converts the color to ASS format (&HAABBGGRR).
    pub fn to_ass(&self) -> String {
        // ASS alpha is inverted (0 = opaque, 255 = transparent)
        format!(
            "&H{:02X}{:02X}{:02X}{:02X}",
            255 - self.a,
            self.b,
            self.g,
            self.r
        )
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::WHITE
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// Text styling options.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TextStyle {
    /// Bold text.
    pub bold: bool,
    /// Italic text.
    pub italic: bool,
    /// Underline text.
    pub underline: bool,
    /// Strikethrough text.
    pub strikethrough: bool,
    /// Text color.
    pub color: Option<Color>,
    /// Background color.
    pub background_color: Option<Color>,
    /// Font name.
    pub font_name: Option<String>,
    /// Font size.
    pub font_size: Option<f32>,
}

impl TextStyle {
    /// Creates a new default text style.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets bold styling.
    pub fn with_bold(mut self, bold: bool) -> Self {
        self.bold = bold;
        self
    }

    /// Sets italic styling.
    pub fn with_italic(mut self, italic: bool) -> Self {
        self.italic = italic;
        self
    }

    /// Sets underline styling.
    pub fn with_underline(mut self, underline: bool) -> Self {
        self.underline = underline;
        self
    }

    /// Sets strikethrough styling.
    pub fn with_strikethrough(mut self, strikethrough: bool) -> Self {
        self.strikethrough = strikethrough;
        self
    }

    /// Sets the text color.
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }

    /// Sets the background color.
    pub fn with_background_color(mut self, color: Color) -> Self {
        self.background_color = Some(color);
        self
    }

    /// Sets the font name.
    pub fn with_font_name(mut self, name: impl Into<String>) -> Self {
        self.font_name = Some(name.into());
        self
    }

    /// Sets the font size.
    pub fn with_font_size(mut self, size: f32) -> Self {
        self.font_size = Some(size);
        self
    }

    /// Checks if any styling is applied.
    pub fn has_styling(&self) -> bool {
        self.bold
            || self.italic
            || self.underline
            || self.strikethrough
            || self.color.is_some()
            || self.background_color.is_some()
            || self.font_name.is_some()
            || self.font_size.is_some()
    }
}

/// A segment of styled text.
#[derive(Debug, Clone, PartialEq)]
pub struct StyledText {
    /// The text content.
    pub text: String,
    /// The style applied to this text.
    pub style: TextStyle,
}

impl StyledText {
    /// Creates a new styled text segment.
    pub fn new(text: impl Into<String>, style: TextStyle) -> Self {
        Self {
            text: text.into(),
            style,
        }
    }

    /// Creates a plain text segment with no styling.
    pub fn plain(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            style: TextStyle::default(),
        }
    }
}

/// A single subtitle event (one displayed subtitle).
#[derive(Debug, Clone, PartialEq)]
pub struct SubtitleEvent {
    /// Start time of the subtitle.
    pub start: Timestamp,
    /// End time of the subtitle.
    pub end: Timestamp,
    /// The text content with styling information.
    pub text: Vec<StyledText>,
    /// Optional layer/depth for overlapping subtitles.
    pub layer: i32,
    /// Optional style name reference (for ASS/SSA).
    pub style_name: Option<String>,
    /// Optional speaker/actor name.
    pub speaker: Option<String>,
    /// Optional positioning information.
    pub position: Option<Position>,
}

impl SubtitleEvent {
    /// Creates a new subtitle event with plain text.
    pub fn new(start: Timestamp, end: Timestamp, text: impl Into<String>) -> Self {
        Self {
            start,
            end,
            text: vec![StyledText::plain(text)],
            layer: 0,
            style_name: None,
            speaker: None,
            position: None,
        }
    }

    /// Creates a new subtitle event with styled text segments.
    pub fn with_styled_text(start: Timestamp, end: Timestamp, text: Vec<StyledText>) -> Self {
        Self {
            start,
            end,
            text,
            layer: 0,
            style_name: None,
            speaker: None,
            position: None,
        }
    }

    /// Returns the plain text content without styling.
    pub fn plain_text(&self) -> String {
        self.text.iter().map(|s| s.text.as_str()).collect()
    }

    /// Returns the duration of the subtitle.
    pub fn duration(&self) -> Timestamp {
        Timestamp::from_millis(self.end.milliseconds.saturating_sub(self.start.milliseconds))
    }

    /// Sets the layer.
    pub fn with_layer(mut self, layer: i32) -> Self {
        self.layer = layer;
        self
    }

    /// Sets the style name.
    pub fn with_style_name(mut self, name: impl Into<String>) -> Self {
        self.style_name = Some(name.into());
        self
    }

    /// Sets the speaker name.
    pub fn with_speaker(mut self, speaker: impl Into<String>) -> Self {
        self.speaker = Some(speaker.into());
        self
    }

    /// Sets the position.
    pub fn with_position(mut self, position: Position) -> Self {
        self.position = Some(position);
        self
    }
}

/// Position information for a subtitle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position {
    /// Horizontal position (0-100% or pixel value).
    pub x: f32,
    /// Vertical position (0-100% or pixel value).
    pub y: f32,
    /// Alignment.
    pub alignment: Alignment,
}

impl Position {
    /// Creates a new position.
    pub fn new(x: f32, y: f32, alignment: Alignment) -> Self {
        Self { x, y, alignment }
    }
}

/// Text alignment options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Alignment {
    /// Bottom left.
    BottomLeft,
    /// Bottom center (default).
    #[default]
    BottomCenter,
    /// Bottom right.
    BottomRight,
    /// Middle left.
    MiddleLeft,
    /// Middle center.
    MiddleCenter,
    /// Middle right.
    MiddleRight,
    /// Top left.
    TopLeft,
    /// Top center.
    TopCenter,
    /// Top right.
    TopRight,
}

impl Alignment {
    /// Converts from ASS alignment number (1-9, numpad layout).
    pub fn from_ass_alignment(value: u8) -> Self {
        match value {
            1 => Alignment::BottomLeft,
            2 => Alignment::BottomCenter,
            3 => Alignment::BottomRight,
            4 => Alignment::MiddleLeft,
            5 => Alignment::MiddleCenter,
            6 => Alignment::MiddleRight,
            7 => Alignment::TopLeft,
            8 => Alignment::TopCenter,
            9 => Alignment::TopRight,
            _ => Alignment::BottomCenter,
        }
    }

    /// Converts to ASS alignment number (1-9, numpad layout).
    pub fn to_ass_alignment(&self) -> u8 {
        match self {
            Alignment::BottomLeft => 1,
            Alignment::BottomCenter => 2,
            Alignment::BottomRight => 3,
            Alignment::MiddleLeft => 4,
            Alignment::MiddleCenter => 5,
            Alignment::MiddleRight => 6,
            Alignment::TopLeft => 7,
            Alignment::TopCenter => 8,
            Alignment::TopRight => 9,
        }
    }
}

/// A complete subtitle track containing multiple events.
#[derive(Debug, Clone, PartialEq)]
pub struct SubtitleTrack {
    /// Title of the track.
    pub title: Option<String>,
    /// Language code (e.g., "en", "es").
    pub language: Option<String>,
    /// The subtitle events.
    pub events: Vec<SubtitleEvent>,
    /// Named styles (for ASS/SSA).
    pub styles: Vec<NamedStyle>,
    /// Additional metadata.
    pub metadata: TrackMetadata,
}

impl SubtitleTrack {
    /// Creates a new empty subtitle track.
    pub fn new() -> Self {
        Self {
            title: None,
            language: None,
            events: Vec::new(),
            styles: Vec::new(),
            metadata: TrackMetadata::default(),
        }
    }

    /// Creates a new subtitle track with events.
    pub fn with_events(events: Vec<SubtitleEvent>) -> Self {
        Self {
            title: None,
            language: None,
            events,
            styles: Vec::new(),
            metadata: TrackMetadata::default(),
        }
    }

    /// Adds a subtitle event.
    pub fn add_event(&mut self, event: SubtitleEvent) {
        self.events.push(event);
    }

    /// Adds a named style.
    pub fn add_style(&mut self, style: NamedStyle) {
        self.styles.push(style);
    }

    /// Sorts events by start time.
    pub fn sort_by_time(&mut self) {
        self.events.sort_by_key(|e| e.start.milliseconds);
    }

    /// Returns the total duration of the track.
    pub fn duration(&self) -> Timestamp {
        self.events
            .iter()
            .map(|e| e.end.milliseconds)
            .max()
            .map(Timestamp::from_millis)
            .unwrap_or_default()
    }

    /// Sets the title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Sets the language.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }
}

impl Default for SubtitleTrack {
    fn default() -> Self {
        Self::new()
    }
}

/// A named style definition (primarily for ASS/SSA).
#[derive(Debug, Clone, PartialEq)]
pub struct NamedStyle {
    /// Name of the style.
    pub name: String,
    /// The text style properties.
    pub style: TextStyle,
    /// Border style.
    pub border_style: BorderStyle,
    /// Outline thickness.
    pub outline: f32,
    /// Shadow depth.
    pub shadow: f32,
    /// Margins (left, right, vertical).
    pub margins: (i32, i32, i32),
    /// Alignment.
    pub alignment: Alignment,
}

impl NamedStyle {
    /// Creates a new named style with defaults.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            style: TextStyle::default(),
            border_style: BorderStyle::default(),
            outline: 2.0,
            shadow: 2.0,
            margins: (10, 10, 10),
            alignment: Alignment::BottomCenter,
        }
    }
}

/// Border style options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BorderStyle {
    /// Outline with shadow.
    #[default]
    OutlineAndShadow,
    /// Opaque box behind text.
    OpaqueBox,
}

/// Additional metadata for a subtitle track.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TrackMetadata {
    /// Original script info.
    pub script_info: Option<String>,
    /// Video resolution width.
    pub play_res_x: Option<u32>,
    /// Video resolution height.
    pub play_res_y: Option<u32>,
    /// Timer speed percentage.
    pub timer: Option<f64>,
    /// Wrap style.
    pub wrap_style: Option<u8>,
    /// Collisions mode.
    pub collisions: Option<String>,
}

/// Subtitle format types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubtitleFormat {
    /// SubRip format (.srt).
    Srt,
    /// Advanced SubStation Alpha format (.ass).
    Ass,
    /// SubStation Alpha format (.ssa).
    Ssa,
    /// WebVTT format (.vtt).
    WebVtt,
    /// CEA-608 closed captions (Line 21).
    Cea608,
    /// CEA-708 closed captions (DTVCC).
    Cea708,
}

impl SubtitleFormat {
    /// Returns the typical file extension for this format.
    pub fn extension(&self) -> &'static str {
        match self {
            SubtitleFormat::Srt => "srt",
            SubtitleFormat::Ass => "ass",
            SubtitleFormat::Ssa => "ssa",
            SubtitleFormat::WebVtt => "vtt",
            SubtitleFormat::Cea608 => "cc",
            SubtitleFormat::Cea708 => "cc",
        }
    }

    /// Returns the MIME type for this format.
    pub fn mime_type(&self) -> &'static str {
        match self {
            SubtitleFormat::Srt => "application/x-subrip",
            SubtitleFormat::Ass | SubtitleFormat::Ssa => "text/x-ssa",
            SubtitleFormat::WebVtt => "text/vtt",
            SubtitleFormat::Cea608 => "application/x-cea-608",
            SubtitleFormat::Cea708 => "application/x-cea-708",
        }
    }

    /// Attempts to detect the format from a file extension.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "srt" => Some(SubtitleFormat::Srt),
            "ass" => Some(SubtitleFormat::Ass),
            "ssa" => Some(SubtitleFormat::Ssa),
            "vtt" | "webvtt" => Some(SubtitleFormat::WebVtt),
            _ => None,
        }
    }
}

impl fmt::Display for SubtitleFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SubtitleFormat::Srt => write!(f, "SRT"),
            SubtitleFormat::Ass => write!(f, "ASS"),
            SubtitleFormat::Ssa => write!(f, "SSA"),
            SubtitleFormat::WebVtt => write!(f, "WebVTT"),
            SubtitleFormat::Cea608 => write!(f, "CEA-608"),
            SubtitleFormat::Cea708 => write!(f, "CEA-708"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp_creation() {
        let ts = Timestamp::new(1, 30, 45, 500);
        assert_eq!(ts.hours(), 1);
        assert_eq!(ts.minutes(), 30);
        assert_eq!(ts.seconds(), 45);
        assert_eq!(ts.millis(), 500);
        assert_eq!(ts.milliseconds, 5445500);
    }

    #[test]
    fn test_timestamp_srt_format() {
        let ts = Timestamp::new(0, 1, 23, 456);
        assert_eq!(ts.to_srt_string(), "00:01:23,456");

        let parsed = Timestamp::parse_srt("00:01:23,456").unwrap();
        assert_eq!(parsed, ts);
    }

    #[test]
    fn test_timestamp_vtt_format() {
        let ts = Timestamp::new(0, 1, 23, 456);
        assert_eq!(ts.to_vtt_string(), "00:01:23.456");

        let parsed = Timestamp::parse_vtt("00:01:23.456").unwrap();
        assert_eq!(parsed, ts);

        // Test short format
        let parsed_short = Timestamp::parse_vtt("01:23.456").unwrap();
        assert_eq!(parsed_short, ts);
    }

    #[test]
    fn test_timestamp_ass_format() {
        let ts = Timestamp::new(1, 23, 45, 670);
        assert_eq!(ts.to_ass_string(), "1:23:45.67");

        let parsed = Timestamp::parse_ass("1:23:45.67").unwrap();
        assert_eq!(parsed, ts);
    }

    #[test]
    fn test_color_hex() {
        let color = Color::rgb(255, 128, 64);
        assert_eq!(color.to_hex(), "#FF8040");

        let parsed = Color::from_hex("#FF8040").unwrap();
        assert_eq!(parsed, color);

        // Test with alpha
        let color_alpha = Color::rgba(255, 128, 64, 128);
        assert_eq!(color_alpha.to_hex(), "#FF804080");
    }

    #[test]
    fn test_color_ass() {
        let color = Color::rgb(255, 128, 64);
        let ass = color.to_ass();
        assert_eq!(ass, "&H004080FF");

        let parsed = Color::from_ass("&H004080FF").unwrap();
        assert_eq!(parsed, color);
    }

    #[test]
    fn test_text_style_builder() {
        let style = TextStyle::new()
            .with_bold(true)
            .with_italic(true)
            .with_color(Color::RED);

        assert!(style.bold);
        assert!(style.italic);
        assert!(!style.underline);
        assert_eq!(style.color, Some(Color::RED));
        assert!(style.has_styling());
    }

    #[test]
    fn test_subtitle_event() {
        let event = SubtitleEvent::new(
            Timestamp::new(0, 0, 1, 0),
            Timestamp::new(0, 0, 3, 0),
            "Hello, world!",
        );

        assert_eq!(event.plain_text(), "Hello, world!");
        assert_eq!(event.duration().milliseconds, 2000);
    }

    #[test]
    fn test_alignment_conversion() {
        assert_eq!(Alignment::from_ass_alignment(2), Alignment::BottomCenter);
        assert_eq!(Alignment::BottomCenter.to_ass_alignment(), 2);

        assert_eq!(Alignment::from_ass_alignment(5), Alignment::MiddleCenter);
        assert_eq!(Alignment::MiddleCenter.to_ass_alignment(), 5);
    }

    #[test]
    fn test_subtitle_format() {
        assert_eq!(SubtitleFormat::from_extension("srt"), Some(SubtitleFormat::Srt));
        assert_eq!(SubtitleFormat::from_extension("VTT"), Some(SubtitleFormat::WebVtt));
        assert_eq!(SubtitleFormat::Srt.extension(), "srt");
        assert_eq!(SubtitleFormat::WebVtt.mime_type(), "text/vtt");
    }
}
