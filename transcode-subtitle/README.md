# transcode-subtitle

Subtitle parsing, writing, and conversion for the transcode library.

## Overview

This crate provides comprehensive subtitle support including parsing, writing, and format conversion for SRT, ASS/SSA, WebVTT, and closed caption formats.

## Features

- **SRT Support**: Parse and write SubRip subtitles
- **ASS/SSA Support**: Advanced SubStation Alpha with full styling
- **WebVTT Support**: Web Video Text Tracks for HTML5
- **CEA-608/708**: Closed caption support
- **Format Conversion**: Convert between any supported formats
- **Text Styling**: Bold, italic, underline, color preservation
- **Timing Accuracy**: Millisecond-precision timestamps

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-subtitle = { path = "../transcode-subtitle" }
```

### Parsing Subtitles

```rust
use transcode_subtitle::{srt, vtt, ass};

// Parse SRT
let srt_content = r#"1
00:00:01,000 --> 00:00:04,000
Hello, world!

"#;
let track = srt::parse(srt_content)?;

// Parse WebVTT
let vtt_content = r#"WEBVTT

00:00:01.000 --> 00:00:04.000
Hello, world!
"#;
let track = vtt::parse(vtt_content)?;
```

### Writing Subtitles

```rust
use transcode_subtitle::{srt, SubtitleTrack, SubtitleEvent, Timestamp};

let mut track = SubtitleTrack::new();
track.add_event(SubtitleEvent::new(
    Timestamp::new(0, 0, 1, 0),   // 00:00:01.000
    Timestamp::new(0, 0, 4, 0),   // 00:00:04.000
    "Hello, world!",
));

let output = srt::write(&track);
```

### Format Conversion

```rust
use transcode_subtitle::{convert, SubtitleFormat};

// Auto-detect source format and convert to WebVTT
let vtt = convert::convert_auto(srt_content, SubtitleFormat::WebVtt)?;

// Or use the conversion builder
let vtt = convert::ConversionBuilder::new(srt_content)
    .from_format(SubtitleFormat::Srt)
    .to_vtt()?;
```

### Styled Text

```rust
use transcode_subtitle::{StyledText, TextStyle, Color, SubtitleEvent, Timestamp};

let styled_text = vec![
    StyledText::new("Bold text", TextStyle::new().with_bold(true)),
    StyledText::plain(" and "),
    StyledText::new("red text", TextStyle::new().with_color(Color::RED)),
];

let event = SubtitleEvent::with_styled_text(
    Timestamp::new(0, 0, 1, 0),
    Timestamp::new(0, 0, 4, 0),
    styled_text,
);
```

### Format Detection

```rust
use transcode_subtitle::convert::detect_format;

let format = detect_format(content)?;
match format {
    SubtitleFormat::Srt => println!("SRT format"),
    SubtitleFormat::WebVtt => println!("WebVTT format"),
    SubtitleFormat::Ass => println!("ASS/SSA format"),
    _ => println!("Unknown"),
}
```

## Supported Formats

| Format | Parse | Write | Styling |
|--------|-------|-------|---------|
| SRT | Yes | Yes | Basic |
| WebVTT | Yes | Yes | Full |
| ASS/SSA | Yes | Yes | Full |
| CEA-608 | Yes | No | Limited |
| CEA-708 | Yes | No | Full |

## Text Styles

| Style | SRT | WebVTT | ASS |
|-------|-----|--------|-----|
| Bold | Yes | Yes | Yes |
| Italic | Yes | Yes | Yes |
| Underline | Yes | Yes | Yes |
| Color | Yes | Yes | Yes |
| Font | No | Yes | Yes |
| Position | No | Yes | Yes |

## Timestamp Format

```rust
use transcode_subtitle::Timestamp;

// Create from components
let ts = Timestamp::new(1, 23, 45, 678);  // 01:23:45.678

// Access components
assert_eq!(ts.hours(), 1);
assert_eq!(ts.minutes(), 23);
assert_eq!(ts.seconds(), 45);
assert_eq!(ts.millis(), 678);
```

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
