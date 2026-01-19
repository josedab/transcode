# Quick Start

This guide will get you transcoding in under 5 minutes.

## Basic Transcoding

### Using the CLI

```bash
# Simple transcode (auto-detects codecs)
transcode -i input.mp4 -o output.mp4

# Specify video codec and bitrate
transcode -i input.mp4 -o output.mp4 --video-codec h264 --video-bitrate 5000000

# Convert to different format
transcode -i input.mkv -o output.mp4

# Show progress
transcode -i input.mp4 -o output.mp4 --progress
```

### Using the Rust Library

```rust
use transcode::{Transcoder, TranscodeOptions, Result};

fn main() -> Result<()> {
    // Configure transcoding options
    let options = TranscodeOptions::new()
        .input("input.mp4")
        .output("output.mp4")
        .video_bitrate(5_000_000)  // 5 Mbps
        .audio_bitrate(128_000)    // 128 kbps
        .overwrite(true);

    // Create and run transcoder
    let mut transcoder = Transcoder::new(options)?;
    transcoder.run()?;

    // Get statistics
    let stats = transcoder.stats();
    println!("Processed {} frames", stats.frames_encoded);
    println!("Compression ratio: {:.2}x", stats.compression_ratio());

    Ok(())
}
```

### Using Python

```python
import transcode_py

# Simple transcoding
stats = transcode_py.transcode('input.mp4', 'output.mp4')
print(f"Processed {stats.frames_encoded} frames")

# With options
options = transcode_py.TranscodeOptions()
options = options.input('input.mp4')
options = options.output('output.mp4')
options = options.video_bitrate(5_000_000)
options = options.overwrite(True)

transcoder = transcode_py.Transcoder(options)
stats = transcoder.run()
```

## Common Tasks

### Change Resolution

```rust
let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.mp4")
    .width(1920)
    .height(1080);
```

### Extract Audio

```bash
transcode -i video.mp4 -o audio.aac --no-video
```

### Change Container Format

```bash
# MKV to MP4 (copy streams, no re-encoding)
transcode -i input.mkv -o output.mp4 --video-codec copy --audio-codec copy
```

### Two-Pass Encoding

```rust
let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.mp4")
    .video_bitrate(5_000_000)
    .two_pass(true);
```

## Quality Presets

Transcode provides quality presets for common use cases:

```rust
use transcode::presets;

// Web streaming (smaller file, good quality)
let options = presets::web_streaming("input.mp4", "output.mp4");

// Archive (high quality, larger file)
let options = presets::archive("input.mp4", "output.mp4");

// Mobile (optimized for phones)
let options = presets::mobile("input.mp4", "output.mp4");
```

## Error Handling

```rust
use transcode::{Transcoder, TranscodeOptions, Result, Error};

fn main() -> Result<()> {
    let options = TranscodeOptions::new()
        .input("input.mp4")
        .output("output.mp4");

    match Transcoder::new(options) {
        Ok(mut transcoder) => {
            transcoder.run()?;
            Ok(())
        }
        Err(Error::FileNotFound(path)) => {
            eprintln!("Input file not found: {}", path);
            Err(Error::FileNotFound(path))
        }
        Err(Error::UnsupportedCodec(codec)) => {
            eprintln!("Codec not supported: {}", codec);
            Err(Error::UnsupportedCodec(codec))
        }
        Err(e) => Err(e),
    }
}
```

## Next Steps

- [CLI Usage](./cli-usage.md) - Full command-line reference
- [Video Codecs](../guides/video-codecs.md) - Codec-specific options
- [Quality Metrics](../guides/quality-metrics.md) - Measure output quality
