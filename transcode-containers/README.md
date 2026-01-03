# transcode-containers

Container format demuxers and muxers for the Transcode codec library.

## Overview

This crate provides support for reading (demuxing) and writing (muxing) media container formats. It handles the extraction of encoded streams from container files and the packaging of encoded data into standard container formats.

## Supported Formats

| Format | Demux | Mux | Notes |
|--------|-------|-----|-------|
| MP4/MOV | Yes | Yes | Full ISOBMFF support with H.264, H.265, AAC |
| MKV | Probe | - | Format detection only |
| WebM | Probe | - | Format detection only |
| MPEG-TS | Probe | - | Format detection only |

## Key Types

### Traits

- **`Demuxer`** - Trait for reading container formats. Provides methods for opening files, reading packets, seeking, and accessing stream metadata.
- **`Muxer`** - Trait for writing container formats. Provides methods for creating files, adding streams, writing packets, and finalizing output.

### Stream Information

- **`StreamInfo`** - Metadata about a stream (codec, time base, duration, dimensions)
- **`TrackType`** - Video, Audio, Subtitle, or Data
- **`CodecId`** - H264, H265, VP9, AV1, AAC, MP3, Opus, FLAC

### Implementations

- **`Mp4Demuxer`** - Demuxer for MP4/MOV/ISOBMFF files
- **`Mp4Muxer`** - Muxer for creating MP4 files

## Usage

### Demuxing (Reading)

```rust
use transcode_containers::{Mp4Demuxer, Demuxer};
use std::fs::File;

let file = File::open("input.mp4")?;
let mut demuxer = Mp4Demuxer::new();
demuxer.open(file)?;

// Get stream info
for i in 0..demuxer.num_streams() {
    if let Some(info) = demuxer.stream_info(i) {
        println!("Stream {}: {:?}", i, info.codec_id);
    }
}

// Read packets
while let Some(packet) = demuxer.read_packet()? {
    println!("Packet: stream={}, pts={:?}", packet.stream_index, packet.pts);
}
```

### Muxing (Writing)

```rust
use transcode_containers::{Mp4Muxer, Muxer, StreamInfo, TrackType, CodecId};
use transcode_core::rational::Rational;
use std::fs::File;

let file = File::create("output.mp4")?;
let mut muxer = Mp4Muxer::new();
muxer.create(file)?;

// Add a video stream
let stream_info = StreamInfo {
    index: 0,
    track_type: TrackType::Video,
    codec_id: CodecId::H264,
    time_base: Rational::new(1, 90000),
    duration: None,
    extra_data: Some(avc_config),
    video: Some(video_info),
    audio: None,
};
muxer.add_stream(stream_info)?;
muxer.write_header()?;

// Write encoded packets
muxer.write_packet(&packet)?;

// Finalize
muxer.write_trailer()?;
```

### Format Detection

```rust
use transcode_containers::traits::probe;
use std::fs::File;

let mut file = File::open("unknown_file")?;
if let Some(result) = probe(&mut file, 4096)? {
    println!("Detected format: {} (confidence: {})", result.format_name, result.score);
}
```

## Features

- `async` - Enables async support with tokio and futures

## Documentation

For complete documentation of the Transcode library, see the [main transcode crate](../transcode/).
