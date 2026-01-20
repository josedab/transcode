# ADR-0016: FFmpeg Compatibility Layer

## Status

Accepted

## Date

2024-07

## Context

FFmpeg is the de facto standard for video processing. Operators, scripts, and workflows worldwide use FFmpeg command-line syntax:

```bash
ffmpeg -i input.mp4 -c:v libx264 -b:v 5M -c:a aac -b:a 128k output.mp4
```

Adopting Transcode requires either:
1. Learning entirely new syntax and conventions
2. Rewriting existing automation scripts
3. Maintaining parallel FFmpeg installations

This friction slows adoption, especially in organizations with established FFmpeg workflows.

## Decision

Implement an **FFmpeg compatibility layer** in `transcode-compat` that:

### 1. Parses FFmpeg Command-Line Syntax

```rust
pub struct FfmpegCommand {
    pub inputs: Vec<InputSpec>,
    pub outputs: Vec<OutputSpec>,
    pub global_options: GlobalOptions,
}

impl FfmpegCommand {
    pub fn parse(args: &[&str]) -> Result<Self>;
}

// Usage
let cmd = FfmpegCommand::parse(&[
    "-i", "input.mp4",
    "-c:v", "libx264",
    "-b:v", "5M",
    "output.mp4"
])?;
```

### 2. Supports Stream Specifiers

```rust
pub enum StreamSpecifier {
    All,                    // No specifier
    Type(StreamType),       // :v, :a, :s
    Index(usize),           // :0, :1
    TypeIndex(StreamType, usize), // :v:0, :a:1
    Tag(String),            // :m:language:eng
}

// Parse "-c:v:0 libx264" -> video stream 0, codec x264
```

### 3. Parses Filter Graphs

```rust
pub struct FilterGraph {
    pub chains: Vec<FilterChain>,
}

pub struct FilterChain {
    pub input_labels: Vec<String>,
    pub filters: Vec<Filter>,
    pub output_labels: Vec<String>,
}

// Parse "[0:v]scale=1920:1080,fps=30[out]"
let graph = FilterGraph::parse("[0:v]scale=1920:1080,fps=30[out]")?;
```

### 4. Maps Codec Names

```rust
pub enum VideoCodecName {
    Libx264,    // -> H264
    Libx265,    // -> Hevc
    Libvpx,     // -> Vp8
    LibvpxVp9,  // -> Vp9
    Libaom,     // -> Av1
    // ... etc
}

impl VideoCodecName {
    pub fn parse(s: &str) -> Option<Self>;
    pub fn to_video_codec(&self) -> Option<VideoCodec>;
}
```

### 5. Parses Value Formats

```rust
// Bitrate: "5M", "128k", "5000000"
pub struct Bitrate(u64);
impl Bitrate {
    pub fn parse(s: &str) -> Result<Self>;
    pub fn bps(&self) -> u64;
}

// Time: "1:30:00", "90.5", "00:01:30.500"
pub struct TimeValue(f64);

// Resolution: "1080p", "1920x1080", "hd"
pub struct Resolution { pub width: u32, pub height: u32 }

// Aspect ratio: "16:9", "1.777"
pub struct AspectRatio { pub num: u32, pub den: u32 }
```

## Consequences

### Positive

1. **Zero learning curve**: Existing FFmpeg users can switch immediately

2. **Script compatibility**: Many existing scripts work with minimal changes
   ```bash
   # Before
   ffmpeg -i input.mp4 -c:v libx264 output.mp4

   # After
   transcode -i input.mp4 -c:v libx264 output.mp4
   ```

3. **Documentation leverage**: FFmpeg's extensive docs apply to common operations

4. **Gradual migration**: Users can adopt Transcode-specific features incrementally

5. **Filter graph interop**: Complex filter expressions work as expected

### Negative

1. **Incomplete coverage**: Not every FFmpeg option can be supported

2. **Behavioral differences**: Some options may have subtly different effects

3. **Maintenance burden**: Must track FFmpeg option changes

4. **Confusion risk**: Users may expect 100% compatibility

### Mitigations

1. **Clear documentation**: List supported vs unsupported options

2. **Validation mode**: `--validate-compat` warns about unsupported options

3. **Native alternative docs**: Show Transcode-native equivalents

## Implementation Details

### Command Parser

```rust
impl FfmpegCommand {
    pub fn parse(args: &[&str]) -> Result<Self> {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut current_output = OutputSpec::default();
        let mut iter = args.iter().peekable();

        while let Some(arg) = iter.next() {
            match *arg {
                "-i" => {
                    let path = iter.next().ok_or(ParseError::MissingValue)?;
                    inputs.push(InputSpec::new(path));
                }
                "-c:v" | "-vcodec" => {
                    let codec = iter.next().ok_or(ParseError::MissingValue)?;
                    current_output.video_codec = VideoCodecName::parse(codec);
                }
                "-b:v" => {
                    let rate = iter.next().ok_or(ParseError::MissingValue)?;
                    current_output.video_bitrate = Some(Bitrate::parse(rate)?);
                }
                // ... many more options
                arg if !arg.starts_with('-') => {
                    current_output.path = arg.to_string();
                    outputs.push(std::mem::take(&mut current_output));
                }
                _ => {}
            }
        }

        Ok(Self { inputs, outputs, global_options: GlobalOptions::default() })
    }
}
```

### Filter Parser

Uses a simple recursive descent parser:

```rust
// Grammar:
// graph      = chain (";" chain)*
// chain      = [labels] filters [labels]
// labels     = "[" name "]" ("[" name "]")*
// filters    = filter ("," filter)*
// filter     = name ("=" params)?
// params     = param (":" param)*

impl FilterGraph {
    pub fn parse(input: &str) -> Result<Self> {
        let mut parser = FilterParser::new(input);
        parser.parse_graph()
    }
}
```

### Codec Mapping Table

```rust
const VIDEO_CODEC_MAP: &[(&str, VideoCodec)] = &[
    ("libx264", VideoCodec::H264),
    ("h264", VideoCodec::H264),
    ("libx265", VideoCodec::Hevc),
    ("hevc", VideoCodec::Hevc),
    ("libvpx-vp9", VideoCodec::Vp9),
    ("libaom-av1", VideoCodec::Av1),
    ("av1", VideoCodec::Av1),
    // ...
];

const AUDIO_CODEC_MAP: &[(&str, AudioCodec)] = &[
    ("aac", AudioCodec::Aac),
    ("libopus", AudioCodec::Opus),
    ("libvorbis", AudioCodec::Vorbis),
    ("flac", AudioCodec::Flac),
    // ...
];
```

### Preset Mapping

```rust
pub fn map_x264_preset(preset: &str) -> EncoderPreset {
    match preset {
        "ultrafast" => EncoderPreset::new().speed(10).subme(0),
        "superfast" => EncoderPreset::new().speed(9).subme(1),
        "veryfast" => EncoderPreset::new().speed(8).subme(2),
        "faster" => EncoderPreset::new().speed(7).subme(4),
        "fast" => EncoderPreset::new().speed(6).subme(6),
        "medium" => EncoderPreset::new().speed(5).subme(7),
        "slow" => EncoderPreset::new().speed(4).subme(8),
        "slower" => EncoderPreset::new().speed(3).subme(9),
        "veryslow" => EncoderPreset::new().speed(2).subme(10),
        "placebo" => EncoderPreset::new().speed(1).subme(11),
        _ => EncoderPreset::default(),
    }
}
```

## Alternatives Considered

### Alternative 1: No Compatibility Layer

Force users to learn native Transcode syntax.

Rejected because:
- High adoption friction
- Wastes existing knowledge
- Scripts need complete rewrites

### Alternative 2: FFmpeg Wrapper

Shell out to actual FFmpeg for unsupported features.

Rejected because:
- Defeats purpose of pure-Rust library
- Requires FFmpeg installation
- Complex error handling across process boundary

### Alternative 3: Full FFmpeg CLI Clone

Implement every FFmpeg option.

Rejected because:
- Enormous scope (FFmpeg has hundreds of options)
- Many options don't map to Transcode concepts
- Maintenance nightmare

## Supported Options

### Input/Output

| Option | Supported | Notes |
|--------|-----------|-------|
| `-i` | Yes | Input file |
| `-y` | Yes | Overwrite output |
| `-n` | Yes | Never overwrite |
| `-f` | Yes | Force format |

### Video

| Option | Supported | Notes |
|--------|-----------|-------|
| `-c:v` | Yes | Video codec |
| `-b:v` | Yes | Video bitrate |
| `-r` | Yes | Frame rate |
| `-s` | Yes | Resolution |
| `-aspect` | Yes | Aspect ratio |
| `-vf` | Partial | Common filters |
| `-preset` | Yes | Encoder preset |
| `-crf` | Yes | Quality factor |

### Audio

| Option | Supported | Notes |
|--------|-----------|-------|
| `-c:a` | Yes | Audio codec |
| `-b:a` | Yes | Audio bitrate |
| `-ar` | Yes | Sample rate |
| `-ac` | Yes | Channel count |
| `-af` | Partial | Common filters |

## References

- [FFmpeg Documentation](https://ffmpeg.org/ffmpeg.html)
- [FFmpeg Filters](https://ffmpeg.org/ffmpeg-filters.html)
- [FFmpeg Codec Options](https://ffmpeg.org/ffmpeg-codecs.html)
