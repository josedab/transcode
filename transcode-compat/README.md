# transcode-compat

FFmpeg compatibility layer for the Transcode library. Enables easy migration from FFmpeg-based workflows by providing familiar command-line parsing and syntax support.

## Features

- **Command-line Parsing**: Parse FFmpeg-style arguments into structured commands
- **Stream Specifiers**: Full support for `-c:v`, `-b:a:0`, `0:v:0` syntax
- **Filter Graphs**: Parse simple chains (`scale=1920:1080,fps=30`) and complex graphs
- **Codec Mapping**: Convert FFmpeg codec names (libx264, libopus) to internal types
- **Preset Mapping**: Map x264/x265/VP9/AV1 presets to encoder settings
- **Value Parsing**: Handle bitrates (`5M`), times (`1:30:00`), resolutions (`1080p`)

## Quick Start

```rust
use transcode_compat::{FfmpegCommand, CommandBuilder};

// Parse FFmpeg-style command-line arguments
let args = ["-i", "input.mp4", "-c:v", "libx264", "-b:v", "5M", "output.mp4"];
let cmd = FfmpegCommand::parse(&args).unwrap();

// Use the builder for convenient access
let builder = CommandBuilder::new(cmd);
println!("Video codec: {:?}", builder.video_codec());
println!("Video bitrate: {:?}", builder.video_bitrate());
```

## Key Types

| Type | Description |
|------|-------------|
| `FfmpegCommand` | Parsed command with inputs, outputs, and global options |
| `CommandBuilder` | Convenience wrapper for accessing parsed settings |
| `FilterChain` / `FilterGraph` | Parsed video/audio filter expressions |
| `VideoCodecName` / `AudioCodecName` | FFmpeg codec name mappings |
| `Preset` / `Tune` | Encoder preset and tuning options |
| `Bitrate` / `TimeValue` / `Resolution` | Parsed value types |
| `StreamSpecifier` | Stream selection (v:0, a:1, 0:v:0) |

## Value Parsing

```rust
use transcode_compat::options::{Bitrate, TimeValue, Resolution};

let bitrate = Bitrate::parse("5M").unwrap();       // 5,000,000 bps
let time = TimeValue::parse("1:30:00").unwrap();   // 5400 seconds
let res = Resolution::parse("1080p").unwrap();     // 1920x1080
```

## Filter Graphs

```rust
use transcode_compat::filter::{FilterChain, FilterGraph, video, audio};

// Parse filter chains
let chain = FilterChain::parse("scale=1920:1080,fps=30").unwrap();

// Parse complex filter graphs
let graph = FilterGraph::parse("[0:v]scale=1280:720[v];[0:a]volume=0.5[a]").unwrap();

// Build filters programmatically
let scale = video::scale(1920, 1080);
let volume = audio::volume(0.5);
```

## Codec Mapping

```rust
use transcode_compat::formats::{VideoCodecName, AudioCodecName, ContainerName};

let vcodec = VideoCodecName::parse("libx264").unwrap();  // -> H264
let acodec = AudioCodecName::parse("libopus").unwrap();  // -> Opus
let format = ContainerName::from_extension("mkv");       // -> Mkv
```

## Presets

```rust
use transcode_compat::preset::{Preset, EncoderSettings};

let preset = Preset::parse("slow").unwrap();
println!("Speed: {}, Quality: {}", preset.speed_value(), preset.quality_value());

let settings = EncoderSettings::new()
    .with_preset(Preset::Medium)
    .with_crf(23.0);
```

## Stream Specifiers

```rust
use transcode_compat::error::{StreamSpecifier, StreamType};

let spec = StreamSpecifier::parse("v:0").unwrap();    // First video stream
let spec = StreamSpecifier::parse("0:a:1").unwrap();  // Second audio from first input

assert!(spec.matches(0, 1, StreamType::Audio));
```

## Supported FFmpeg Options

**Video**: `-c:v`, `-b:v`, `-r`, `-s`, `-aspect`, `-vf`, `-preset`, `-tune`, `-crf`, `-profile:v`

**Audio**: `-c:a`, `-b:a`, `-ar`, `-ac`, `-af`

**General**: `-i`, `-f`, `-ss`, `-t`, `-to`, `-map`, `-y`, `-threads`, `-filter_complex`

## License

See the workspace root for license information.
