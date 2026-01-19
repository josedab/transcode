# Container Formats

This guide covers the container formats (file formats) supported by Transcode.

## What is a Container?

A container is a file format that holds:
- Video streams (one or more)
- Audio streams (one or more)
- Subtitle tracks
- Metadata (title, chapters, etc.)

The container doesn't define compression—that's the codec's job.

## Supported Containers

| Format | Demux | Mux | Extension | Use Case |
|--------|-------|-----|-----------|----------|
| MP4/MOV | ✅ | ✅ | .mp4, .m4v, .mov | Streaming, web |
| MKV | ✅ | ✅ | .mkv | Archival, flexibility |
| WebM | ✅ | ✅ | .webm | Web, open source |
| MPEG-TS | ✅ | ✅ | .ts, .mts | Broadcast, HLS |
| AVI | ✅ | ✅ | .avi | Legacy |
| FLV | ✅ | ✅ | .flv | Legacy streaming |
| MXF | ✅ | ✅ | .mxf | Professional broadcast |
| HLS | ❌ | ✅ | .m3u8 | Adaptive streaming |
| DASH | ❌ | ✅ | .mpd | Adaptive streaming |

## MP4/MOV

The most common container for video delivery.

### Features
- Wide device support
- Supports H.264, H.265, AAC
- Chapter markers
- Fast start (moov atom placement)

```rust
use transcode::containers::mp4::Mp4Muxer;

let options = TranscodeOptions::new()
    .output("output.mp4")
    .faststart(true)  // Move moov to start for streaming
    .brand("mp42");   // Compatibility brand
```

### Faststart

For web streaming, move metadata to the beginning:

```rust
let options = TranscodeOptions::new()
    .faststart(true);
```

This allows playback to start before the entire file downloads.

## MKV (Matroska)

Flexible container supporting almost any codec.

### Features
- Any codec combination
- Multiple audio/subtitle tracks
- Chapters and attachments
- Soft subtitles (SRT, ASS)

```rust
use transcode::containers::mkv::MkvMuxer;

let options = TranscodeOptions::new()
    .output("output.mkv")
    .add_subtitle("english.srt", "eng")
    .add_subtitle("spanish.srt", "spa");
```

## WebM

Open format for web video.

### Features
- Royalty-free
- VP8/VP9/AV1 video
- Opus/Vorbis audio
- Native browser support

```rust
let options = TranscodeOptions::new()
    .output("output.webm")
    .video_codec(Codec::Vp9)
    .audio_codec(Codec::Opus);
```

## MPEG-TS

Transport stream for broadcast and streaming.

### Features
- Error resilience
- No seeking required
- Used by HLS
- Live streaming support

```rust
use transcode::containers::ts::TsMuxer;

let options = TranscodeOptions::new()
    .output("output.ts")
    .pat_period(100)  // PAT every 100ms
    .pcr_period(40);  // PCR every 40ms
```

## MXF

Professional broadcast container.

### Features
- Frame-accurate editing
- Timecode support
- Multiple essence types
- Industry standard for broadcast

```rust
use transcode::containers::mxf::MxfMuxer;

let muxer = MxfMuxer::new()
    .operational_pattern(OpPattern::Op1a)
    .timecode_track(true)
    .build()?;
```

## Choosing a Container

```
Need web streaming?         → MP4 (faststart)
Need maximum flexibility?   → MKV
Need open/royalty-free?     → WebM
Need broadcast delivery?    → MXF or MPEG-TS
Need adaptive streaming?    → HLS or DASH
Need legacy support?        → AVI
```

## Codec Compatibility

| Container | H.264 | H.265 | AV1 | VP9 | AAC | Opus |
|-----------|-------|-------|-----|-----|-----|------|
| MP4 | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| MKV | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| WebM | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ |
| TS | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |

## Remuxing

Change container without re-encoding:

```rust
// MKV to MP4 (no re-encoding)
let options = TranscodeOptions::new()
    .input("input.mkv")
    .output("output.mp4")
    .video_codec(Codec::Copy)
    .audio_codec(Codec::Copy);
```

This is very fast since frames aren't decoded/encoded.
