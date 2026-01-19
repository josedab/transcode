# transcode-containers

Container format support for demuxing and muxing.

## Overview

`transcode-containers` provides support for common container formats:

- **MP4/MOV** - MPEG-4 Part 14
- **MKV** - Matroska
- **WebM** - WebM
- **MPEG-TS** - Transport Stream
- **AVI** - Audio Video Interleave
- **FLV** - Flash Video
- **MXF** - Material Exchange Format

## Crate Features

```toml
[dependencies]
transcode-containers = { version = "1.0", features = ["mp4", "mkv"] }
```

| Feature | Description |
|---------|-------------|
| `mp4` | MP4/MOV support |
| `mkv` | Matroska support |
| `webm` | WebM support |
| `ts` | MPEG-TS support |
| `avi` | AVI support |
| `flv` | FLV support |
| `mxf` | MXF support |
| `all` | All formats |

## Demuxing

### MP4 Demuxer

```rust
use transcode_containers::mp4::Mp4Demuxer;

let mut demuxer = Mp4Demuxer::open("input.mp4")?;

// Get stream information
for stream in demuxer.streams() {
    println!("Stream {}: {:?}", stream.index, stream.codec);
}

// Read packets
while let Some(packet) = demuxer.read_packet()? {
    println!("Packet: stream={}, pts={}", packet.stream_index, packet.pts);
}
```

### Generic Demuxer

```rust
use transcode_containers::{Demuxer, detect_format};

// Auto-detect format
let format = detect_format("input.mkv")?;
let mut demuxer = Demuxer::open("input.mkv", format)?;

// Or use extension-based detection
let mut demuxer = Demuxer::open_auto("input.mkv")?;
```

### Seeking

```rust
// Seek to timestamp (in stream timebase)
demuxer.seek(90000, 0)?;  // Seek video stream to 1 second (90000/90000)

// Seek to keyframe
demuxer.seek_keyframe(180000, 0)?;
```

## Muxing

### MP4 Muxer

```rust
use transcode_containers::mp4::{Mp4Muxer, Mp4Config};

let config = Mp4Config::builder()
    .faststart(true)           // Move moov to start
    .brand("mp42")             // File brand
    .build();

let mut muxer = Mp4Muxer::create("output.mp4", config)?;

// Add streams
let video_stream = muxer.add_video_stream(VideoParams {
    codec: Codec::H264,
    width: 1920,
    height: 1080,
    framerate: Rational::new(30, 1),
})?;

let audio_stream = muxer.add_audio_stream(AudioParams {
    codec: Codec::Aac,
    sample_rate: 48000,
    channels: 2,
})?;

// Write header
muxer.write_header()?;

// Write packets
for packet in packets {
    muxer.write_packet(&packet)?;
}

// Finalize
muxer.write_trailer()?;
```

### Generic Muxer

```rust
use transcode_containers::{Muxer, Format};

let mut muxer = Muxer::create("output.mkv", Format::Matroska)?;
```

## Stream Information

### StreamInfo

```rust
pub struct StreamInfo {
    pub index: usize,
    pub codec: Codec,
    pub codec_params: CodecParams,
    pub timebase: Rational,
    pub duration: Option<i64>,
    pub metadata: HashMap<String, String>,
}

// Video-specific
pub struct VideoParams {
    pub width: u32,
    pub height: u32,
    pub framerate: Rational,
    pub pixel_format: PixelFormat,
}

// Audio-specific
pub struct AudioParams {
    pub sample_rate: u32,
    pub channels: u32,
    pub sample_format: SampleFormat,
}
```

### Accessing Stream Info

```rust
let demuxer = Mp4Demuxer::open("input.mp4")?;

for stream in demuxer.streams() {
    match &stream.codec_params {
        CodecParams::Video(params) => {
            println!("Video: {}x{} @ {} fps",
                params.width,
                params.height,
                params.framerate.to_f64()
            );
        }
        CodecParams::Audio(params) => {
            println!("Audio: {} Hz, {} channels",
                params.sample_rate,
                params.channels
            );
        }
    }
}
```

## Metadata

### Reading Metadata

```rust
let demuxer = Mp4Demuxer::open("input.mp4")?;

// Container-level metadata
if let Some(title) = demuxer.metadata().get("title") {
    println!("Title: {}", title);
}

// Stream-level metadata
for stream in demuxer.streams() {
    if let Some(lang) = stream.metadata.get("language") {
        println!("Stream {} language: {}", stream.index, lang);
    }
}
```

### Writing Metadata

```rust
let mut muxer = Mp4Muxer::create("output.mp4", config)?;

muxer.set_metadata("title", "My Video")?;
muxer.set_metadata("artist", "Author Name")?;
```

## Chapters

### Reading Chapters

```rust
let demuxer = Mp4Demuxer::open("input.mp4")?;

for chapter in demuxer.chapters() {
    println!("Chapter: {} ({} - {})",
        chapter.title,
        format_time(chapter.start),
        format_time(chapter.end)
    );
}
```

### Writing Chapters

```rust
muxer.add_chapter(Chapter {
    start: 0,
    end: 60_000,
    title: "Introduction".into(),
})?;
```

## Format Detection

```rust
use transcode_containers::{detect_format, Format};

// From file extension
let format = Format::from_extension("mp4");  // Some(Format::Mp4)

// From file header
let format = detect_format("input.mp4")?;    // Format::Mp4

// Check support
println!("Can demux: {}", format.can_demux());
println!("Can mux: {}", format.can_mux());
```

## Full API Reference

See the [rustdoc documentation](https://docs.rs/transcode-containers) for complete API details.
