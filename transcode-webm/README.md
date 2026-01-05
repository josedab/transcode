# transcode-webm

WebM container format support for the transcode library.

## Overview

WebM is a subset of the Matroska container format specifically designed for web video. It provides efficient, royalty-free media delivery optimized for streaming.

## Features

- **EBML Parsing**: Full EBML (Extensible Binary Meta Language) support
- **Muxing**: Cluster management and keyframe-based segmentation
- **Demuxing**: Seeking via cues, track parsing
- **Laced Frames**: Xiph, EBML, and fixed-size lacing support
- **Codec Private Data**: VP8, VP9, AV1, Vorbis, Opus support
- **Cue Generation**: Automatic seeking index creation

## Supported Codecs

| Type | Codecs |
|------|--------|
| Video | VP8, VP9, AV1 |
| Audio | Vorbis, Opus |

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-webm = { path = "../transcode-webm" }
```

### Muxing

```rust
use transcode_webm::{WebmMuxer, VideoTrackConfig, AudioTrackConfig};
use transcode_core::{VideoCodec, AudioCodec, Packet, PacketFlags};
use std::io::Cursor;

// Create a muxer
let buffer = Cursor::new(Vec::new());
let mut muxer = WebmMuxer::new(buffer);

// Add tracks
muxer.add_video_track(
    VideoTrackConfig::new(1, VideoCodec::Vp9, 1920, 1080)
        .with_frame_rate(30.0)
)?;

muxer.add_audio_track(
    AudioTrackConfig::new(2, AudioCodec::Opus, 48000.0, 2)
)?;

// Write header
muxer.write_header()?;

// Write packets
muxer.write_packet(&packet)?;

// Finalize
muxer.finalize()?;
```

### Demuxing

```rust
use transcode_webm::WebmDemuxer;
use std::fs::File;

let file = File::open("video.webm")?;
let mut demuxer = WebmDemuxer::new(file);

// Read header and tracks
demuxer.read_segment_info()?;

// Print track info
for (num, track) in &demuxer.tracks {
    println!("Track {}: {:?} - {}", num, track.track_type, track.codec_id);
}

// Read packets
while let Some(packet) = demuxer.read_packet()? {
    println!("Packet: track={}, pts={}, size={}",
        packet.stream_index, packet.pts.value, packet.data().len());
}
```

### Configuration

```rust
use transcode_webm::WebmConfig;

let config = WebmConfig::new()
    .with_timecode_scale(1_000_000)      // 1ms precision
    .with_max_cluster_duration(5000)      // 5 second clusters
    .with_cues(true)                      // Generate seeking index
    .with_writing_app("my-app")
    .with_title("My Video");
```

### Codec Private Data

```rust
use transcode_webm::{Vp9CodecPrivate, OpusCodecPrivate};

// VP9 codec private
let vp9_private = Vp9CodecPrivate::new()
    .with_profile(0)
    .with_level(31)
    .with_bit_depth(8)
    .build();

// Opus codec private
let opus_private = OpusCodecPrivate::new(2, 48000)
    .with_pre_skip(312)
    .build();
```

## WebM vs Matroska

| Feature | WebM | Matroska |
|---------|------|----------|
| Video Codecs | VP8, VP9, AV1 | Any |
| Audio Codecs | Vorbis, Opus | Any |
| Subtitles | WebVTT | Any |
| Container | Subset | Full |
| Purpose | Web streaming | Universal |

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| timecode_scale | 1,000,000 | Nanoseconds per timecode unit |
| max_cluster_duration | 5000ms | Maximum cluster duration |
| generate_cues | true | Create seeking index |

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
