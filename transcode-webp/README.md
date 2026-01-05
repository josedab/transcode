# transcode-webp

WebP image codec support for the transcode library.

## Overview

WebP is a modern image format developed by Google that provides both lossy and lossless compression. It offers smaller file sizes compared to JPEG and PNG while maintaining quality.

## Features

- **RIFF Container**: Full RIFF container parsing and writing
- **VP8 Lossy**: Lossy compression based on VP8 video codec
- **VP8L Lossless**: Lossless compression with predictive coding
- **Alpha Channel**: Separate alpha plane support
- **Animation**: Animated WebP with ANIM/ANMF chunks
- **Metadata**: EXIF and XMP metadata extraction

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-webp = { path = "../transcode-webp" }
```

### Decoding

```rust
use transcode_webp::WebPDecoder;
use std::fs::File;
use std::io::BufReader;

let file = File::open("image.webp")?;
let reader = BufReader::new(file);
let decoder = WebPDecoder::new(reader)?;

// Get dimensions without full decode
let (width, height) = decoder.dimensions()?;

// Full decode
let image = decoder.decode()?;
```

### Encoding

```rust
use transcode_webp::{WebPEncoder, EncodingMode};

// RGBA image data (4 bytes per pixel)
let rgba_data = vec![255u8; 100 * 100 * 4];

// Lossless encoding
let encoder = WebPEncoder::new().mode(EncodingMode::Lossless);
let frame = encoder.encode_rgba(&rgba_data, 100, 100)?;

// Write to RIFF container
let mut output = Vec::new();
encoder.write_to_riff(&frame, &mut output)?;
```

### Lossy Encoding with Quality

```rust
use transcode_webp::{WebPEncoder, EncodingMode, WebPEncoderConfig};

let config = WebPEncoderConfig::new()
    .mode(EncodingMode::Lossy)
    .quality(80);

let encoder = WebPEncoder::with_config(config);
let output = encoder.encode_rgba(&rgba_data, width, height)?;
```

### Animation Decoding

```rust
use transcode_webp::WebPDecoder;

let decoder = WebPDecoder::new(reader)?;

if decoder.is_animated()? {
    let frames = decoder.decode_animation()?;
    for frame in frames {
        println!("Frame: {}x{} @ {}ms",
                 frame.width, frame.height, frame.timestamp);
    }
}
```

### Metadata Extraction

```rust
use transcode_webp::WebPDecoder;

let mut decoder = WebPDecoder::new(reader)?;
let metadata = decoder.metadata()?;

if let Some(exif) = metadata.exif {
    println!("EXIF data: {} bytes", exif.data.len());
}

if let Some(xmp) = metadata.xmp {
    println!("XMP data: {} bytes", xmp.data.len());
}
```

## Encoding Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| Lossy | VP8-based compression | Photos, general images |
| Lossless | VP8L compression | Graphics, screenshots |
| Auto | Automatic selection | Best of both |

## Chunk Types

| Chunk | Description |
|-------|-------------|
| VP8 | Lossy image data |
| VP8L | Lossless image data |
| VP8X | Extended format header |
| ALPH | Alpha channel data |
| ANIM | Animation parameters |
| ANMF | Animation frame |
| EXIF | EXIF metadata |
| XMP | XMP metadata |

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
