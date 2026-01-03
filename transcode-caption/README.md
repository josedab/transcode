# transcode-caption

Automatic captioning and subtitle extraction for the transcode media processing library.

## Overview

This crate provides speech-to-text captioning with Whisper integration and multi-format subtitle export capabilities.

## Features

- **Speech-to-Text**: Whisper-based transcription with configurable model sizes (Tiny to Large)
- **Language Support**: Auto-detection or explicit language selection with translation to English
- **Word-Level Timestamps**: Optional word-level timing for precise synchronization
- **Multiple Subtitle Formats**: Export to SRT, VTT, ASS, TTML, and JSON
- **Timing Utilities**: Shift, scale, split, merge, and frame-snap subtitle timing

## Key Types

### Transcription Types

| Type | Description |
|------|-------------|
| `Captioner` | Main transcriber using Whisper models |
| `CaptionConfig` | Configuration for model size, language, and timestamps |
| `Transcription` | Complete transcription result with segments |
| `Segment` | Timed text segment with confidence and word timestamps |
| `Word` | Individual word with timing and confidence |
| `ModelSize` | Whisper model variants: Tiny, Base, Small, Medium, Large |

### Subtitle Types

| Type | Description |
|------|-------------|
| `SubtitleFormat` | Output format enum (Srt, Vtt, Ass, Ttml, Json) |
| `SubtitleStyle` | Styling options for font, colors, outline, shadow |
| `SubtitleExporter` | Format-specific subtitle exporter |
| `TimingAdjuster` | Utilities for timing manipulation |

## Usage

### Basic Transcription

```rust
use transcode_caption::{Captioner, CaptionConfig, ModelSize};

let config = CaptionConfig {
    model_size: ModelSize::Base,
    language: Some("en".into()),
    word_timestamps: true,
    ..Default::default()
};

let captioner = Captioner::new(config);
let transcription = captioner.transcribe(&audio_samples)?;
```

### Export to Subtitle Formats

```rust
use transcode_caption::{SubtitleExporter, SubtitleFormat};

// Export to SRT
let srt = transcription.to_srt();

// Export to VTT
let vtt = transcription.to_vtt();

// Export with custom styling (ASS format)
let exporter = SubtitleExporter::new(SubtitleFormat::Ass);
let ass = exporter.export(&transcription);
```

### Timing Adjustments

```rust
use transcode_caption::TimingAdjuster;

// Shift subtitles 500ms later
TimingAdjuster::shift(&mut segments, 500);

// Scale timing by 1.05x (for speed changes)
TimingAdjuster::scale(&mut segments, 1.05);

// Split long segments (max 5s or 40 chars)
TimingAdjuster::split_long(&mut segments, 5000, 40);

// Merge short segments (min 1s, max 200ms gap)
TimingAdjuster::merge_short(&mut segments, 1000, 200);

// Snap to 24fps frame boundaries
TimingAdjuster::snap_to_frames(&mut segments, 24.0);
```

## Model Sizes

| Model | Parameters | Size (MB) |
|-------|------------|-----------|
| Tiny | ~39M | 75 |
| Base | ~74M | 142 |
| Small | ~244M | 466 |
| Medium | ~769M | 1533 |
| Large | ~1.5B | 2952 |

## Feature Flags

- `whisper` - Enable Whisper integration (requires external whisper.cpp build)

## License

See the workspace root for license information.
