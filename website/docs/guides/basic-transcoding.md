---
sidebar_position: 1
title: Basic Transcoding
description: Common transcoding patterns and use cases
---

# Basic Transcoding

This guide covers common transcoding patterns you'll use in everyday workflows.

## Simple Transcode

The most basic operation: convert a video from one format to another.

```rust
use transcode::{Transcoder, TranscodeOptions};

let options = TranscodeOptions::new()
    .input("input.mov")
    .output("output.mp4")
    .video_codec("h264")
    .audio_codec("aac");

let mut transcoder = Transcoder::new(options)?;
transcoder.run()?;
```

CLI equivalent:
```bash
transcode -i input.mov -o output.mp4 --video-codec h264 --audio-codec aac
```

## Control Quality with Bitrate

### Constant Bitrate (CBR)

Predictable file size, variable quality:

```rust
let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.mp4")
    .video_bitrate(5_000_000)     // 5 Mbps
    .audio_bitrate(128_000);      // 128 kbps
```

### Variable Bitrate (VBR)

Consistent quality, variable file size:

```rust
let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.mp4")
    .video_crf(23)                // Quality level (18-28 typical)
    .audio_bitrate(128_000);
```

**CRF Guidelines:**
- 18-20: Visually lossless
- 21-23: High quality (recommended)
- 24-26: Good quality
- 27-28: Lower quality, smaller files

## Change Resolution

### Scale to Specific Size

```rust
use transcode::{Transcoder, TranscodeOptions, ScaleFilter};

let options = TranscodeOptions::new()
    .input("4k_video.mp4")
    .output("1080p_video.mp4");

let mut transcoder = Transcoder::new(options)?;
transcoder = transcoder.add_video_filter(ScaleFilter::new(1920, 1080));
transcoder.run()?;
```

CLI:
```bash
transcode -i 4k_video.mp4 -o 1080p_video.mp4 -F "scale=1920:1080"
```

### Preserve Aspect Ratio

```rust
// Scale width to 1280, calculate height automatically
let filter = ScaleFilter::new(1280, -1);

// Scale to fit within 1920x1080, preserving aspect ratio
let filter = ScaleFilter::fit(1920, 1080);
```

### Common Resolutions

| Name | Resolution | Aspect | Use Case |
|------|------------|--------|----------|
| 4K UHD | 3840x2160 | 16:9 | High-end streaming |
| 1080p | 1920x1080 | 16:9 | Standard HD |
| 720p | 1280x720 | 16:9 | Web delivery |
| 480p | 854x480 | 16:9 | Mobile, low bandwidth |

## Change Frame Rate

```rust
use transcode::FpsFilter;

let options = TranscodeOptions::new()
    .input("60fps_video.mp4")
    .output("30fps_video.mp4");

let mut transcoder = Transcoder::new(options)?;
transcoder = transcoder.add_video_filter(FpsFilter::new(30.0));
transcoder.run()?;
```

CLI:
```bash
transcode -i 60fps_video.mp4 -o 30fps_video.mp4 -F "fps=30"
```

## Extract Audio

Convert video to audio-only:

```rust
let options = TranscodeOptions::new()
    .input("video.mp4")
    .output("audio.aac")
    .video_codec("none")          // Disable video
    .audio_codec("aac")
    .audio_bitrate(256_000);
```

Or extract to MP3:
```bash
transcode -i video.mp4 -o audio.mp3 --video-codec none --audio-codec mp3
```

## Extract Video (Remove Audio)

```rust
let options = TranscodeOptions::new()
    .input("video.mp4")
    .output("video_only.mp4")
    .audio_codec("none");         // Disable audio
```

## Trim Video

Extract a portion of a video:

```rust
use transcode::Duration;

let options = TranscodeOptions::new()
    .input("long_video.mp4")
    .output("clip.mp4")
    .start_time(Duration::from_secs(30))   // Start at 30s
    .duration(Duration::from_secs(60));     // 60 seconds long
```

CLI:
```bash
transcode -i long_video.mp4 -o clip.mp4 --start 30 --duration 60
```

## Concatenate Videos

Join multiple videos:

```rust
use transcode::Concatenator;

let concat = Concatenator::new()
    .add("part1.mp4")
    .add("part2.mp4")
    .add("part3.mp4")
    .output("combined.mp4");

concat.run()?;
```

## Batch Processing

Process multiple files:

```rust
use std::fs;
use transcode::{Transcoder, TranscodeOptions};

let input_dir = "input/";
let output_dir = "output/";

for entry in fs::read_dir(input_dir)? {
    let entry = entry?;
    let path = entry.path();

    if path.extension().map_or(false, |e| e == "mov") {
        let input = path.to_str().unwrap();
        let output = format!(
            "{}/{}.mp4",
            output_dir,
            path.file_stem().unwrap().to_str().unwrap()
        );

        let options = TranscodeOptions::new()
            .input(input)
            .output(&output)
            .video_codec("h264")
            .video_bitrate(5_000_000);

        let mut transcoder = Transcoder::new(options)?;
        transcoder.run()?;

        println!("Converted: {} -> {}", input, output);
    }
}
```

CLI batch mode:
```bash
transcode --batch "*.mov" --batch-output-dir output/
```

## Two-Pass Encoding

Better quality for target file size:

```rust
let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.mp4")
    .video_bitrate(5_000_000)
    .two_pass(true);
```

## Preset-Based Encoding

Use presets for common scenarios:

```rust
use transcode::{Preset, Quality, Format};

// Web-optimized preset
let options = TranscodeOptions::preset(Preset::Web)
    .input("input.mp4")
    .output("web.mp4");

// High quality archive
let options = TranscodeOptions::preset(Preset::Archive)
    .quality(Quality::High)
    .input("input.mp4")
    .output("archive.mp4");

// Social media optimized
let options = TranscodeOptions::preset(Preset::Social)
    .format(Format::Instagram)
    .input("input.mp4")
    .output("instagram.mp4");
```

### Available Presets

| Preset | Video | Audio | Use Case |
|--------|-------|-------|----------|
| `Web` | H.264, CRF 23 | AAC 128k | General web delivery |
| `Mobile` | H.264, CRF 26 | AAC 96k | Mobile apps |
| `Archive` | H.265, CRF 18 | FLAC | Long-term storage |
| `Social` | H.264, optimized | AAC 128k | Social platforms |
| `Broadcast` | H.264, 50Mbps | PCM | TV broadcast |

## Watermarking

Add a watermark to video:

```rust
use transcode::WatermarkFilter;

let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("watermarked.mp4");

let watermark = WatermarkFilter::new("logo.png")
    .position(Position::BottomRight)
    .opacity(0.7)
    .margin(20);

let mut transcoder = Transcoder::new(options)?;
transcoder = transcoder.add_video_filter(watermark);
transcoder.run()?;
```

## Error Handling

Handle common issues:

```rust
use transcode::{Transcoder, TranscodeOptions, Error};

fn transcode_with_retry(input: &str, output: &str) -> Result<(), Error> {
    let options = TranscodeOptions::new()
        .input(input)
        .output(output);

    match Transcoder::new(options) {
        Ok(mut transcoder) => {
            transcoder.run()?;
            Ok(())
        }
        Err(Error::Codec(e)) => {
            // Try with different codec
            println!("Codec error: {}, trying fallback...", e);

            let options = TranscodeOptions::new()
                .input(input)
                .output(output)
                .video_codec("h264")  // Force H.264
                .audio_codec("aac");  // Force AAC

            let mut transcoder = Transcoder::new(options)?;
            transcoder.run()?;
            Ok(())
        }
        Err(e) => Err(e),
    }
}
```

## Performance Tips

1. **Use hardware acceleration** when available:
   ```rust
   let options = TranscodeOptions::new()
       .hardware_acceleration(true);
   ```

2. **Set appropriate thread count**:
   ```rust
   let options = TranscodeOptions::new()
       .threads(num_cpus::get());
   ```

3. **Use frame pools** for memory efficiency:
   ```rust
   let options = TranscodeOptions::new()
       .use_frame_pool(true);
   ```

4. **Profile before optimizing**:
   ```rust
   let stats = transcoder.stats();
   println!("Decode time: {:?}", stats.decode_time);
   println!("Encode time: {:?}", stats.encode_time);
   ```

## Next Steps

- [Streaming Output](/docs/guides/streaming-output) - Generate HLS/DASH
- [GPU Acceleration](/docs/guides/gpu-acceleration) - Use GPU for processing
- [Quality Metrics](/docs/guides/quality-metrics) - Measure output quality
