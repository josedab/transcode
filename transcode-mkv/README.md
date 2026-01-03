# transcode-mkv

MKV/Matroska and WebM container support for the transcode library.

## Overview

This crate provides demuxing and muxing capabilities for:
- **Matroska (.mkv)** - Full container support with all codecs
- **WebM (.webm)** - Subset with VP8/VP9/AV1 video and Vorbis/Opus audio

## Features

- EBML (Extensible Binary Meta Language) parsing and writing
- Variable-length integer (VINT) encoding/decoding
- Multiple tracks (video, audio, subtitle)
- Seeking with cues
- Chapters and tags metadata
- WebM codec validation

## Codec Support

| Type | Codec ID | Description | WebM |
|------|----------|-------------|------|
| Video | V_VP8 | VP8 | Yes |
| Video | V_VP9 | VP9 | Yes |
| Video | V_AV1 | AV1 | Yes |
| Video | V_MPEG4/ISO/AVC | H.264 | No |
| Video | V_MPEGH/ISO/HEVC | H.265 | No |
| Audio | A_OPUS | Opus | Yes |
| Audio | A_VORBIS | Vorbis | Yes |
| Audio | A_AAC | AAC | No |
| Audio | A_FLAC | FLAC | No |

## Key Types

- `MkvDemuxer` - Demuxes MKV/WebM files into packets
- `MkvMuxer` - Muxes packets into MKV files
- `WebmDemuxer` / `WebmMuxer` - WebM-specific wrappers with codec validation
- `VideoTrackConfig` / `AudioTrackConfig` - Track configuration for muxing
- `TrackInfo` - Track metadata from demuxed files
- `OpusCodecPrivate` / `Vp9CodecPrivate` - Codec private data builders

## Usage Examples

### Demuxing an MKV file

```rust
use std::fs::File;
use std::io::BufReader;
use transcode_mkv::MkvDemuxer;

let file = File::open("video.mkv").unwrap();
let mut demuxer = MkvDemuxer::new(BufReader::new(file));

// Read header and segment info
demuxer.read_segment_info().unwrap();

// Print track information
for (num, track) in &demuxer.tracks {
    println!("Track {}: {:?} - {}", num, track.track_type, track.codec_id);
}

// Read packets
while let Some(packet) = demuxer.read_packet().unwrap() {
    println!("stream={}, pts={:?}, size={}",
             packet.stream_index, packet.pts.to_millis(), packet.size());
}
```

### Muxing a WebM file

```rust
use std::fs::File;
use std::io::BufWriter;
use transcode_mkv::{WebmMuxer, VideoTrackConfig, AudioTrackConfig};
use transcode_core::{VideoCodec, AudioCodec};

let file = File::create("output.webm").unwrap();
let mut muxer = WebmMuxer::new(BufWriter::new(file));

// Add VP9 video track
let video = VideoTrackConfig::new(1, VideoCodec::Vp9, 1920, 1080);
muxer.add_video_track(video).unwrap();

// Add Opus audio track
let audio = AudioTrackConfig::new(2, AudioCodec::Opus, 48000.0, 2);
muxer.add_audio_track(audio).unwrap();

// Write header and packets
muxer.write_header().unwrap();
// muxer.write_packet(&packet).unwrap();
muxer.finalize().unwrap();
```

### Building Opus codec private data

```rust
use transcode_mkv::OpusCodecPrivate;

let opus = OpusCodecPrivate::new(2, 48000)
    .with_pre_skip(312);
let codec_private = opus.build();
```

## Matroska Element Structure

```
EBML Header
Segment
+-- SeekHead (index to other elements)
+-- Info (segment information)
+-- Tracks (track definitions)
|   +-- TrackEntry
|       +-- Video / Audio
+-- Chapters (chapter markers)
+-- Cues (seeking index)
+-- Tags (metadata)
+-- Cluster (media data)
    +-- Timestamp
    +-- SimpleBlock / BlockGroup
```

## License

See the workspace root for license information.
