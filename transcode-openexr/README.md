# transcode-openexr

OpenEXR image codec implementation for the transcode library.

## Overview

OpenEXR is a high dynamic range (HDR) image format developed by Industrial Light & Magic. It's widely used in the VFX, animation, and film industry for its ability to store high-precision floating-point image data.

## Features

- **High Dynamic Range**: Half-float (16-bit) and float (32-bit) pixel data
- **Multiple Compression**: PIZ, ZIP, ZIPS, RLE, PXR24, B44, DWAA/DWAB
- **Multi-Channel**: RGBA, deep images, arbitrary channels
- **Tiled Storage**: Tiled and scanline storage modes
- **Multi-Part**: Multi-part file support
- **Deep Data**: Deep image support for VFX compositing

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-openexr = { path = "../transcode-openexr" }
```

### Decoding

```rust
use transcode_openexr::{ExrDecoder, Header};

let decoder = ExrDecoder::new()?;
let image = decoder.decode(&exr_data)?;

// Access header information
println!("Channels: {:?}", image.header.channels);
println!("Data window: {:?}", image.header.data_window);
```

### Encoding

```rust
use transcode_openexr::{ExrEncoder, Compression, Header};

let mut header = Header::new(1920, 1080);
header.compression = Compression::Piz;

let encoder = ExrEncoder::new(header)?;
let output = encoder.encode(&pixel_data)?;
```

### Channel Access

```rust
use transcode_openexr::{Channel, PixelType};

// Define channels
let channels = vec![
    Channel::new("R", PixelType::Half),
    Channel::new("G", PixelType::Half),
    Channel::new("B", PixelType::Half),
    Channel::new("A", PixelType::Half),
];
```

## Compression Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| None | No compression | Maximum speed |
| RLE | Run-length encoding | Fast, moderate compression |
| ZIPS | ZIP per scanline | Good for small tiles |
| ZIP | ZIP 16 scanlines | General purpose |
| PIZ | Wavelet-based | Best for noisy images |
| PXR24 | Lossy 24-bit | VFX workflows |
| B44 | Fixed-rate lossy | Real-time playback |
| DWAA/DWAB | DCT-based lossy | High compression |

## Pixel Types

| Type | Bits | Range |
|------|------|-------|
| Half | 16 | Float16 (-65504 to +65504) |
| Float | 32 | Float32 (full IEEE 754) |
| Uint | 32 | Unsigned integer (0 to 4294967295) |

## Version Flags

| Flag | Description |
|------|-------------|
| TILED | Tiled image |
| LONG_NAMES | Names >31 characters |
| NON_IMAGE | Deep data |
| MULTI_PART | Multi-part file |

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
