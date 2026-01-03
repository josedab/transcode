# transcode

A memory-safe, high-performance universal codec library written in Rust.

## Overview

This crate provides a unified, high-level API for transcoding media files. It supports:

- **Video codecs**: H.264/AVC, H.265/HEVC, VP9, AV1
- **Audio codecs**: AAC, MP3, Opus, FLAC
- **Containers**: MP4/MOV, MKV/WebM, MPEG-TS

## Quick Start

```rust
use transcode::{Transcoder, TranscodeOptions};

fn main() -> transcode::Result<()> {
    let options = TranscodeOptions::new()
        .input("input.mp4")
        .output("output.mp4")
        .video_codec("h264")
        .audio_codec("aac");

    let mut transcoder = Transcoder::new(options)?;
    transcoder.run()?;

    Ok(())
}
```

## Key Types

### `TranscodeOptions`

Builder for configuring transcoding jobs. Supports fluent API:

```rust
let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.webm")
    .video_codec("vp9")
    .video_resolution(1920, 1080)
    .video_bitrate(5_000_000)
    .audio_codec("opus")
    .audio_bitrate(128_000)
    .threads(4)
    .hardware_acceleration(true)
    .overwrite(true);
```

Configuration structs: `InputConfig`, `OutputConfig`, `VideoConfig`, `AudioConfig`.

### `Transcoder`

Orchestrates the transcoding process with progress tracking:

```rust
let mut transcoder = Transcoder::new(options)?
    .on_progress(|percent, packets| {
        println!("Progress: {:.1}% ({} packets)", percent, packets);
    });

transcoder.run()?;

let stats = transcoder.stats();
println!("Compression ratio: {:.2}x", stats.compression_ratio());
```

### Presets

Pre-configured profiles for common use cases:

```rust
use transcode::presets::EncodingProfile;

// Web streaming (H.264 + AAC, broad compatibility)
let profile = EncodingProfile::web_streaming();

// High-quality archive (H.265 + FLAC)
let profile = EncodingProfile::archive();

// Mobile-optimized (720p, fast encoding)
let profile = EncodingProfile::mobile();

// Audio-only profiles
let profile = EncodingProfile::podcast();  // MP3, mono
let profile = EncodingProfile::music();    // Opus, stereo
```

Quality and speed presets:

```rust
use transcode::presets::{Quality, Preset};

let crf = Quality::High.crf();  // Returns 18
let bitrate = Quality::Medium.video_bitrate_1080p();  // Returns 5 Mbps
```

## Features

- `async` (default): Enables async/await support via tokio
- Default features provide synchronous API

## Architecture

This crate re-exports types from the underlying workspace crates:

- `transcode-core`: Core types (frames, packets, timestamps, errors)
- `transcode-codecs`: Codec implementations and traits
- `transcode-containers`: Container demuxers/muxers
- `transcode-pipeline`: Transcoding pipeline infrastructure

## License

See repository root for license information.
