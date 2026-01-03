//! # transcode-subtitle
//!
//! A comprehensive subtitle parsing, writing, and conversion library supporting
//! multiple subtitle formats including SRT, ASS/SSA, and WebVTT.
//!
//! ## Features
//!
//! - Parse and write SRT (SubRip) subtitles
//! - Parse and write ASS/SSA (Advanced SubStation Alpha) subtitles
//! - Parse and write WebVTT (Web Video Text Tracks) subtitles
//! - Convert between different subtitle formats
//! - Support for text styling (bold, italic, underline, color)
//! - Preserve timing and formatting during conversion
//!
//! ## Quick Start
//!
//! ### Parsing Subtitles
//!
//! ```rust
//! use transcode_subtitle::{srt, vtt, ass};
//!
//! // Parse SRT
//! let srt_content = r#"1
//! 00:00:01,000 --> 00:00:04,000
//! Hello, world!
//!
//! "#;
//! let track = srt::parse(srt_content).unwrap();
//! assert_eq!(track.events.len(), 1);
//!
//! // Parse WebVTT
//! let vtt_content = r#"WEBVTT
//!
//! 00:00:01.000 --> 00:00:04.000
//! Hello, world!
//! "#;
//! let track = vtt::parse(vtt_content).unwrap();
//! ```
//!
//! ### Writing Subtitles
//!
//! ```rust
//! use transcode_subtitle::{srt, SubtitleTrack, SubtitleEvent, Timestamp};
//!
//! let mut track = SubtitleTrack::new();
//! track.add_event(SubtitleEvent::new(
//!     Timestamp::new(0, 0, 1, 0),
//!     Timestamp::new(0, 0, 4, 0),
//!     "Hello, world!",
//! ));
//!
//! let output = srt::write(&track);
//! assert!(output.contains("Hello, world!"));
//! ```
//!
//! ### Converting Between Formats
//!
//! ```rust
//! use transcode_subtitle::{convert, SubtitleFormat};
//!
//! let srt_content = r#"1
//! 00:00:01,000 --> 00:00:04,000
//! Hello, world!
//!
//! "#;
//!
//! // Auto-detect source format and convert to WebVTT
//! let vtt = convert::convert_auto(srt_content, SubtitleFormat::WebVtt).unwrap();
//! assert!(vtt.starts_with("WEBVTT"));
//!
//! // Or use the conversion builder for more control
//! let vtt = convert::ConversionBuilder::new(srt_content)
//!     .from_format(SubtitleFormat::Srt)
//!     .to_vtt()
//!     .unwrap();
//! ```
//!
//! ### Working with Styled Text
//!
//! ```rust
//! use transcode_subtitle::{srt, SubtitleTrack, SubtitleEvent, Timestamp, StyledText, TextStyle, Color};
//!
//! let mut track = SubtitleTrack::new();
//!
//! // Create styled text segments
//! let styled_text = vec![
//!     StyledText::new("Bold text", TextStyle::new().with_bold(true)),
//!     StyledText::plain(" and "),
//!     StyledText::new("red text", TextStyle::new().with_color(Color::RED)),
//! ];
//!
//! track.add_event(SubtitleEvent::with_styled_text(
//!     Timestamp::new(0, 0, 1, 0),
//!     Timestamp::new(0, 0, 4, 0),
//!     styled_text,
//! ));
//!
//! let output = srt::write(&track);
//! assert!(output.contains("<b>Bold text</b>"));
//! ```

pub mod ass;
pub mod convert;
pub mod srt;
pub mod types;
pub mod vtt;

// Re-export commonly used types at the crate root
pub use types::{
    Alignment, BorderStyle, Color, NamedStyle, Position, StyledText, SubtitleError,
    SubtitleEvent, SubtitleFormat, SubtitleResult, SubtitleTrack, TextStyle, Timestamp,
    TrackMetadata,
};

/// Prelude module for convenient imports.
///
/// ```rust
/// use transcode_subtitle::prelude::*;
/// ```
pub mod prelude {
    pub use crate::ass;
    pub use crate::convert::{self, ConversionBuilder};
    pub use crate::srt;
    pub use crate::types::{
        Alignment, BorderStyle, Color, NamedStyle, Position, StyledText, SubtitleError,
        SubtitleEvent, SubtitleFormat, SubtitleResult, SubtitleTrack, TextStyle, Timestamp,
        TrackMetadata,
    };
    pub use crate::vtt;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srt_roundtrip() {
        let original = r#"1
00:00:01,000 --> 00:00:04,000
Hello, world!

2
00:00:05,000 --> 00:00:08,500
Second line.

"#;

        let track = srt::parse(original).unwrap();
        let output = srt::write(&track);
        let reparsed = srt::parse(&output).unwrap();

        assert_eq!(track.events.len(), reparsed.events.len());
        assert_eq!(track.events[0].plain_text(), reparsed.events[0].plain_text());
    }

    #[test]
    fn test_vtt_roundtrip() {
        let original = r#"WEBVTT

00:00:01.000 --> 00:00:04.000
Hello, world!

00:00:05.000 --> 00:00:08.500
Second line.

"#;

        let track = vtt::parse(original).unwrap();
        let output = vtt::write(&track);
        let reparsed = vtt::parse(&output).unwrap();

        assert_eq!(track.events.len(), reparsed.events.len());
        assert_eq!(track.events[0].plain_text(), reparsed.events[0].plain_text());
    }

    #[test]
    fn test_ass_roundtrip() {
        let original = r#"[Script Info]
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,Hello, world!
"#;

        let track = ass::parse(original).unwrap();
        let output = ass::write(&track);
        let reparsed = ass::parse(&output).unwrap();

        assert_eq!(track.events.len(), reparsed.events.len());
        assert_eq!(track.events[0].plain_text(), reparsed.events[0].plain_text());
    }

    #[test]
    fn test_cross_format_conversion() {
        let srt = r#"1
00:00:01,000 --> 00:00:04,000
Hello, world!

"#;

        // SRT -> VTT
        let track = srt::parse(srt).unwrap();
        let vtt_output = vtt::write(&track);
        assert!(vtt_output.contains("WEBVTT"));
        assert!(vtt_output.contains("Hello, world!"));

        // VTT -> ASS
        let track = vtt::parse(&vtt_output).unwrap();
        let ass_output = ass::write(&track);
        assert!(ass_output.contains("[Script Info]"));
        assert!(ass_output.contains("Hello, world!"));

        // ASS -> SRT
        let track = ass::parse(&ass_output).unwrap();
        let srt_output = srt::write(&track);
        assert!(srt_output.contains("Hello, world!"));
    }

    #[test]
    fn test_styled_text_preservation() {
        let styled_srt = r#"1
00:00:01,000 --> 00:00:04,000
<b>Bold</b> and <i>italic</i>

"#;

        let track = srt::parse(styled_srt).unwrap();

        // Check that styling was parsed
        let has_bold = track.events[0]
            .text
            .iter()
            .any(|s| s.style.bold && s.text.contains("Bold"));
        let has_italic = track.events[0]
            .text
            .iter()
            .any(|s| s.style.italic && s.text.contains("italic"));

        assert!(has_bold);
        assert!(has_italic);

        // Convert to VTT and back
        let vtt_output = vtt::write(&track);
        assert!(vtt_output.contains("<b>Bold</b>"));
        assert!(vtt_output.contains("<i>italic</i>"));
    }

    #[test]
    fn test_color_preservation() {
        let colored_srt = r##"1
00:00:01,000 --> 00:00:04,000
<font color="#FF0000">Red text</font>

"##;

        let track = srt::parse(colored_srt).unwrap();

        // Check that color was parsed
        let has_red = track.events[0].text.iter().any(|s| {
            s.style.color == Some(Color::RED) && s.text.contains("Red text")
        });

        assert!(has_red);
    }

    #[test]
    fn test_multiline_subtitles() {
        let multiline_srt = r#"1
00:00:01,000 --> 00:00:04,000
Line one
Line two
Line three

"#;

        let track = srt::parse(multiline_srt).unwrap();
        let text = track.events[0].plain_text();

        assert!(text.contains("Line one"));
        assert!(text.contains("Line two"));
        assert!(text.contains("Line three"));
        assert!(text.contains('\n'));
    }

    #[test]
    fn test_timestamp_accuracy() {
        let srt = r#"1
01:23:45,678 --> 02:34:56,789
Test

"#;

        let track = srt::parse(srt).unwrap();
        let event = &track.events[0];

        assert_eq!(event.start.hours(), 1);
        assert_eq!(event.start.minutes(), 23);
        assert_eq!(event.start.seconds(), 45);
        assert_eq!(event.start.millis(), 678);

        assert_eq!(event.end.hours(), 2);
        assert_eq!(event.end.minutes(), 34);
        assert_eq!(event.end.seconds(), 56);
        assert_eq!(event.end.millis(), 789);
    }

    #[test]
    fn test_format_detection() {
        let srt = "1\n00:00:01,000 --> 00:00:02,000\nTest\n";
        let vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nTest\n";
        let ass = "[Script Info]\nScriptType: v4.00+\n";

        assert_eq!(convert::detect_format(srt).unwrap(), SubtitleFormat::Srt);
        assert_eq!(convert::detect_format(vtt).unwrap(), SubtitleFormat::WebVtt);
        assert_eq!(convert::detect_format(ass).unwrap(), SubtitleFormat::Ass);
    }
}
