---
sidebar_position: 3
title: Your First Transcode
description: A detailed walkthrough of your first transcoding operation
---

# Your First Transcode

This guide walks through a complete transcoding workflow, explaining each step along the way.

## What We'll Build

We'll create a Rust application that:

1. Opens an input video file
2. Configures encoding parameters
3. Transcodes with progress reporting
4. Displays statistics

## Prerequisites

- Rust installed ([installation guide](/docs/getting-started/installation))
- A test video file (any MP4, MKV, or MOV)

## Step 1: Create a New Project

```bash
cargo new my-transcoder
cd my-transcoder
```

Add Transcode to `Cargo.toml`:

```toml
[dependencies]
transcode = "1.0"
```

## Step 2: Basic Transcoding

Replace `src/main.rs` with:

```rust
use transcode::{Transcoder, TranscodeOptions, Result};

fn main() -> Result<()> {
    // Configure the transcoding operation
    let options = TranscodeOptions::new()
        .input("input.mp4")
        .output("output.mp4")
        .video_bitrate(5_000_000)   // 5 Mbps
        .audio_bitrate(128_000)     // 128 kbps
        .overwrite(true);           // Overwrite if exists

    // Create and run the transcoder
    let mut transcoder = Transcoder::new(options)?;
    transcoder.run()?;

    // Print results
    let stats = transcoder.stats();
    println!("Transcoding complete!");
    println!("  Frames decoded:  {}", stats.frames_decoded);
    println!("  Frames encoded:  {}", stats.frames_encoded);
    println!("  Input size:      {:.2} MB", stats.input_size as f64 / 1_000_000.0);
    println!("  Output size:     {:.2} MB", stats.output_size as f64 / 1_000_000.0);
    println!("  Compression:     {:.2}x", stats.compression_ratio());

    Ok(())
}
```

Run it:

```bash
cargo run
```

## Step 3: Add Progress Reporting

Track progress during transcoding:

```rust
use transcode::{Transcoder, TranscodeOptions, Result};
use std::io::{self, Write};

fn main() -> Result<()> {
    let options = TranscodeOptions::new()
        .input("input.mp4")
        .output("output.mp4")
        .video_bitrate(5_000_000)
        .audio_bitrate(128_000)
        .overwrite(true);

    let mut transcoder = Transcoder::new(options)?;

    // Add progress callback
    transcoder = transcoder.on_progress(|progress, frames| {
        print!("\rProgress: {:.1}% ({} frames)", progress, frames);
        io::stdout().flush().unwrap();
    });

    transcoder.run()?;
    println!("\nDone!");

    Ok(())
}
```

## Step 4: Specify Codecs

Control the output format explicitly:

```rust
let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.mp4")
    .video_codec("h264")        // H.264/AVC
    .audio_codec("aac")         // AAC audio
    .video_bitrate(5_000_000)
    .audio_bitrate(128_000)
    .overwrite(true);
```

### Available Codecs

**Video:**
- `h264` - H.264/AVC (most compatible)
- `h265` - H.265/HEVC (better compression)
- `av1` - AV1 (best compression, slower)
- `vp9` - VP9 (good for WebM)

**Audio:**
- `aac` - AAC (most compatible)
- `opus` - Opus (best quality/size)
- `flac` - FLAC (lossless)

## Step 5: Apply Video Filters

Resize, adjust framerate, or apply effects:

```rust
use transcode::{Transcoder, TranscodeOptions, ScaleFilter, Result};

fn main() -> Result<()> {
    let options = TranscodeOptions::new()
        .input("4k-video.mp4")
        .output("1080p-video.mp4")
        .video_codec("h264")
        .video_bitrate(8_000_000)
        .overwrite(true);

    let mut transcoder = Transcoder::new(options)?;

    // Scale to 1080p
    transcoder = transcoder.add_video_filter(
        ScaleFilter::new(1920, 1080)
    );

    transcoder.run()?;

    Ok(())
}
```

### Common Filters

```rust
// Scale with aspect ratio preservation
ScaleFilter::new(1920, -1)  // Width 1920, auto height

// Change framerate
FpsFilter::new(30.0)

// Adjust volume
VolumeFilter::new(0.8)  // 80% volume
```

## Step 6: Error Handling

Handle common errors gracefully:

```rust
use transcode::{Transcoder, TranscodeOptions, Result, Error};

fn main() {
    match transcode_video() {
        Ok(()) => println!("Success!"),
        Err(e) => handle_error(e),
    }
}

fn transcode_video() -> Result<()> {
    let options = TranscodeOptions::new()
        .input("input.mp4")
        .output("output.mp4");

    let mut transcoder = Transcoder::new(options)?;
    transcoder.run()?;
    Ok(())
}

fn handle_error(error: Error) {
    match error {
        Error::Io(e) => {
            eprintln!("File error: {}", e);
            eprintln!("Check that the input file exists and is readable.");
        }
        Error::Codec(e) => {
            eprintln!("Codec error: {}", e);
            eprintln!("The file may be corrupted or use an unsupported codec.");
        }
        Error::Container(e) => {
            eprintln!("Container error: {}", e);
            eprintln!("The container format may not be supported.");
        }
        _ => eprintln!("Error: {}", error),
    }
}
```

## Step 7: Complete Example

Here's a production-ready example with all features:

```rust
use transcode::{
    Transcoder, TranscodeOptions, Result,
    ScaleFilter, VideoCodec, AudioCodec,
};
use std::env;
use std::io::{self, Write};
use std::time::Instant;

fn main() -> Result<()> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <input> <output>", args[0]);
        std::process::exit(1);
    }

    let input = &args[1];
    let output = &args[2];

    println!("Transcoding: {} -> {}", input, output);

    // Configure transcoding
    let options = TranscodeOptions::new()
        .input(input)
        .output(output)
        .video_codec("h264")
        .audio_codec("aac")
        .video_bitrate(5_000_000)
        .audio_bitrate(128_000)
        .overwrite(true);

    // Create transcoder
    let mut transcoder = Transcoder::new(options)?;

    // Add progress callback
    let start = Instant::now();
    transcoder = transcoder.on_progress(move |progress, frames| {
        let elapsed = start.elapsed().as_secs_f64();
        let fps = if elapsed > 0.0 { frames as f64 / elapsed } else { 0.0 };
        print!("\rProgress: {:5.1}% | {:6} frames | {:5.1} fps",
               progress, frames, fps);
        io::stdout().flush().unwrap();
    });

    // Run transcoding
    transcoder.run()?;

    // Print final statistics
    let stats = transcoder.stats();
    let elapsed = start.elapsed();

    println!("\n");
    println!("Transcoding complete!");
    println!("  Time:        {:.2}s", elapsed.as_secs_f64());
    println!("  Frames:      {}", stats.frames_encoded);
    println!("  Avg FPS:     {:.1}", stats.frames_encoded as f64 / elapsed.as_secs_f64());
    println!("  Input:       {:.2} MB", stats.input_size as f64 / 1_000_000.0);
    println!("  Output:      {:.2} MB", stats.output_size as f64 / 1_000_000.0);
    println!("  Compression: {:.2}x", stats.compression_ratio());

    Ok(())
}
```

Run with:

```bash
cargo run -- input.mp4 output.mp4
```

## What's Next?

Now that you understand the basics, explore:

- [Basic Transcoding Patterns](/docs/guides/basic-transcoding) - Common workflows
- [Streaming Output](/docs/guides/streaming-output) - Generate HLS/DASH
- [GPU Acceleration](/docs/guides/gpu-acceleration) - Speed up processing
- [Quality Metrics](/docs/guides/quality-metrics) - Measure output quality
