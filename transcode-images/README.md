# transcode-images

Image codec support for the transcode library.

## Overview

This crate provides encoding and decoding for common image formats with a unified API for image manipulation and format conversion.

## Features

- **JPEG**: Baseline and progressive JPEG encoding/decoding
- **PNG**: Full PNG support with alpha channel
- **GIF**: Animated GIF support
- **Format Detection**: Automatic format detection from magic bytes
- **Color Spaces**: RGB, RGBA, Grayscale support
- **Metadata**: Basic EXIF/metadata handling

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-images = { path = "../transcode-images" }
```

### Decoding

```rust
use transcode_images::{JpegDecoder, PngDecoder, Image};

// Decode JPEG
let mut decoder = JpegDecoder::new();
let image = decoder.decode(&jpeg_data)?;
println!("Image: {}x{}", image.width(), image.height());

// Decode PNG
let mut decoder = PngDecoder::new();
let image = decoder.decode(&png_data)?;
```

### Encoding

```rust
use transcode_images::{JpegEncoder, PngEncoder, JpegConfig};

// Encode to JPEG with quality setting
let config = JpegConfig::new().with_quality(85);
let mut encoder = JpegEncoder::with_config(config);
let jpeg_data = encoder.encode(&image)?;

// Encode to PNG
let mut encoder = PngEncoder::new();
let png_data = encoder.encode(&image)?;
```

### Format Detection

```rust
use transcode_images::{detect_format, ImageFormat};

let format = detect_format(&data);
match format {
    Some(ImageFormat::Jpeg) => println!("JPEG image"),
    Some(ImageFormat::Png) => println!("PNG image"),
    Some(ImageFormat::Gif) => println!("GIF image"),
    _ => println!("Unknown format"),
}
```

### Format Conversion

```rust
use transcode_images::{JpegDecoder, PngEncoder};

// Convert JPEG to PNG
let mut decoder = JpegDecoder::new();
let image = decoder.decode(&jpeg_data)?;

let mut encoder = PngEncoder::new();
let png_data = encoder.encode(&image)?;
```

## Supported Formats

| Format | Decode | Encode | Alpha | Animation |
|--------|--------|--------|-------|-----------|
| JPEG | Yes | Yes | No | No |
| PNG | Yes | Yes | Yes | Yes (APNG) |
| GIF | Yes | Yes | Yes | Yes |
| WebP | Ref | Ref | Yes | Yes |
| BMP | Yes | No | No | No |

## Feature Flags

- `jpeg` - JPEG encoder/decoder (default)
- `png` - PNG encoder/decoder (default)
- `gif` - GIF encoder/decoder

## Pixel Formats

| Format | Description |
|--------|-------------|
| RGB8 | 8-bit RGB |
| RGBA8 | 8-bit RGBA |
| Gray8 | 8-bit Grayscale |
| GrayA8 | 8-bit Grayscale with Alpha |

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
