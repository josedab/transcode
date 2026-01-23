---
sidebar_position: 2
title: Quick Start
description: Transcode your first video in under 2 minutes
---

# Quick Start

Get a working transcoding pipeline in under 2 minutes.

## CLI Quick Start

The fastest way to transcode a video:

```bash
# Install the CLI
cargo install transcode-cli

# Transcode a video
transcode -i input.mp4 -o output.mp4
```

That's it. Transcode will automatically detect codecs and apply sensible defaults.

### With Options

```bash
# Specify codec and bitrate
transcode -i input.mp4 -o output.mp4 \
  --video-codec h264 \
  --video-bitrate 5000 \
  --audio-codec aac \
  --audio-bitrate 128
```

## Rust Quick Start

```rust
use transcode::{Transcoder, TranscodeOptions};

fn main() -> transcode::Result<()> {
    let options = TranscodeOptions::new()
        .input("input.mp4")
        .output("output.mp4");

    let mut transcoder = Transcoder::new(options)?;
    transcoder.run()?;

    println!("Done! Encoded {} frames", transcoder.stats().frames_encoded);
    Ok(())
}
```

Run it:

```bash
cargo run
```

## Python Quick Start

```python
import transcode_py

# One-liner transcoding
stats = transcode_py.transcode('input.mp4', 'output.mp4')
print(f"Encoded {stats.frames_encoded} frames")
```

### With Options

```python
import transcode_py

options = transcode_py.TranscodeOptions()
options = options.input('input.mp4')
options = options.output('output.mp4')
options = options.video_bitrate(5_000_000)
options = options.audio_bitrate(128_000)

transcoder = transcode_py.Transcoder(options)
stats = transcoder.run()

print(f"Compression ratio: {stats.compression_ratio:.2f}x")
```

## Node.js Quick Start

```javascript
const transcode = require('transcode');

async function main() {
  const stats = await transcode.transcode('input.mp4', 'output.mp4');
  console.log(`Encoded ${stats.framesEncoded} frames`);
}

main();
```

## Docker Quick Start

```bash
# Pull the image
docker pull transcode/transcode

# Run transcoding
docker run -v $(pwd):/data transcode/transcode \
  -i /data/input.mp4 \
  -o /data/output.mp4
```

## What Just Happened?

When you run a transcode operation, the library:

1. **Demuxes** the input container (MP4, MKV, etc.)
2. **Decodes** video and audio streams
3. **Encodes** to the target codecs
4. **Muxes** into the output container

```
┌─────────┐    ┌────────┐    ┌────────┐    ┌─────────┐
│ Demuxer │ -> │Decoder │ -> │Encoder │ -> │  Muxer  │
│  (MP4)  │    │(H.264) │    │(H.264) │    │  (MP4)  │
└─────────┘    └────────┘    └────────┘    └─────────┘
```

## Common Use Cases

### Convert to H.264 for Web

```bash
transcode -i input.mov -o web.mp4 \
  --video-codec h264 \
  --video-bitrate 2500
```

### Extract Audio

```bash
transcode -i video.mp4 -o audio.aac \
  --video-codec none
```

### Change Resolution

```bash
transcode -i 4k.mp4 -o 1080p.mp4 \
  -F "scale=1920:1080"
```

### Generate HLS Stream

```bash
transcode -i input.mp4 -o stream/playlist.m3u8 \
  --format hls
```

## Check SIMD Capabilities

Transcode auto-detects CPU features for optimal performance:

```rust
use transcode_codecs::detect_simd;

let caps = detect_simd();
println!("AVX2: {}", caps.avx2);
println!("NEON: {}", caps.neon);
println!("Best level: {}", caps.best_level());
```

Output on an M1 Mac:
```
AVX2: false
NEON: true
Best level: neon
```

## Next Steps

- [First Transcode](/docs/getting-started/first-transcode) - Detailed walkthrough with explanations
- [Basic Transcoding Guide](/docs/guides/basic-transcoding) - Common transcoding patterns
- [Architecture](/docs/core-concepts/architecture) - Understanding how Transcode works
