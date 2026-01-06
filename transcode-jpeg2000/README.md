# transcode-jpeg2000

JPEG2000 codec support for DCP/cinema workflows in the Transcode library.

## Overview

This crate provides JPEG2000 (J2K/JP2) codec support with native codestream parsing and optional FFI integration via OpenJPEG for full encoding/decoding capabilities.

> **Note**: JPEG2000 is an open standard (ISO/IEC 15444). While the specification is royalty-free, some implementations may have patent considerations. This crate provides native bitstream parsing; full encoding/decoding requires the `ffi-openjpeg` feature.

## Features

- **Native Parsing**: Parse J2K codestreams and JP2 file format without external dependencies
- **Metadata Extraction**: Extract image dimensions, bit depth, component info, and tile structure
- **Format Detection**: Automatic detection of codestream vs. file format, profile identification
- **Cinema Profiles**: Support for DCI Cinema 2K/4K, IMF, and Broadcast profiles
- **FFI Support**: Optional full decode/encode via OpenJPEG (requires `ffi-openjpeg` feature)

## JPEG2000 Format Overview

JPEG2000 uses discrete wavelet transform (DWT) for compression:

| Transform | Type | Filter | Use Case |
|-----------|------|--------|----------|
| 9/7 Irreversible | Lossy | CDF 9/7 | Cinema, broadcast |
| 5/3 Reversible | Lossless | Le Gall 5/3 | Archival, medical |

### Key Features

- **Tile-based encoding**: Large images split into independent tiles
- **Resolution levels**: Progressive decoding by resolution (precincts)
- **Region of interest**: Priority coding for specific image regions
- **Progressive modes**: By quality, resolution, position, or component

## Supported Profiles

| Profile | Max Resolution | Max Bitrate | Use Case |
|---------|----------------|-------------|----------|
| Cinema 2K | 2048×1080 | 250 Mbps | Digital cinema (2K DCP) |
| Cinema 4K | 4096×2160 | 500 Mbps | Digital cinema (4K DCP) |
| Broadcast | 1920×1080 | 200 Mbps | Broadcast contribution |
| IMF | 4096×2160 | Variable | Interoperable Master Format |
| Lossless | Unlimited | N/A | Archival, medical imaging |

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-jpeg2000 = "0.1"

# For full encoding/decoding (requires OpenJPEG)
# transcode-jpeg2000 = { version = "0.1", features = ["ffi-openjpeg"] }
```

## Usage

### Parse Codestream Header

```rust
use transcode_jpeg2000::{Jpeg2000Parser, CodestreamInfo};

let data = include_bytes!("image.j2k");
let mut parser = Jpeg2000Parser::new();

if let Ok(info) = parser.parse_header(data) {
    println!("Image: {}x{}", info.width, info.height);
    println!("Components: {}", info.num_components);
    println!("Bit depth: {} bits", info.bit_depth);
    println!("Tiles: {}x{}", info.tiles_x, info.tiles_y);

    if info.is_lossless() {
        println!("Lossless compression (5/3 wavelet)");
    }
}
```

### Parse JP2 File Format

```rust
use transcode_jpeg2000::{Jp2Parser, Jpeg2000Parser};

let data = include_bytes!("image.jp2");
let mut parser = Jp2Parser::new();

// Parse JP2 file boxes
if let Ok(boxes) = parser.parse(data) {
    println!("Found {} JP2 boxes", boxes.len());

    // Extract codestream for further parsing
    if let Some(codestream) = parser.extract_codestream(data) {
        let mut j2k_parser = Jpeg2000Parser::new();
        let info = j2k_parser.parse_header(&codestream)?;
    }
}
```

### Streaming Decoder

```rust
use transcode_jpeg2000::{StreamingDecoder, DecodedImage};

let mut decoder = StreamingDecoder::new()?;

// Feed data incrementally
for chunk in data_chunks {
    decoder.feed(&chunk)?;

    // Check if we can decode tiles
    while let Some(tile) = decoder.next_tile()? {
        println!("Decoded tile {}: {}x{}",
            tile.index, tile.width, tile.height);
    }
}
```

### Encoding (requires `ffi-openjpeg`)

```rust
use transcode_jpeg2000::{Jpeg2000Encoder, Jpeg2000EncoderConfig, Jpeg2000Profile};

// Configure for Cinema 2K DCP
let config = Jpeg2000EncoderConfig::cinema_2k()
    .with_frame_rate(24.0);

let mut encoder = Jpeg2000Encoder::new(config)?;

// Encode RGB image data
let image_data: Vec<u8> = load_image();
let encoded = encoder.encode(&image_data, 2048, 1080, 3)?;

println!("Encoded {} bytes", encoded.len());
```

### Cinema 4K Encoding

```rust
use transcode_jpeg2000::{Jpeg2000Encoder, Jpeg2000EncoderConfig};

let config = Jpeg2000EncoderConfig::cinema_4k()
    .with_frame_rate(24.0);

let mut encoder = Jpeg2000Encoder::new(config)?;
let encoded = encoder.encode(&image_data, 4096, 2160, 3)?;
```

### Lossless Encoding

```rust
use transcode_jpeg2000::{Jpeg2000Encoder, Jpeg2000EncoderConfig};

let config = Jpeg2000EncoderConfig::lossless();
let mut encoder = Jpeg2000Encoder::new(config)?;
let encoded = encoder.encode(&image_data, width, height, components)?;
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `CodestreamInfo` | Parsed codestream metadata |
| `SizMarker` | Image and tile size parameters |
| `CodMarker` | Coding style default |
| `QcdMarker` | Quantization parameters |
| `DecodedImage` | Decoded image buffer with metadata |
| `EncodedPacket` | Encoded JPEG2000 packet |

### Parsers

| Type | Description |
|------|-------------|
| `Jpeg2000Parser` | Parse J2K codestream headers |
| `Jp2Parser` | Parse JP2 file format boxes |

### Decoders/Encoders

| Type | Description | FFI Required |
|------|-------------|--------------|
| `Jpeg2000Decoder` | Full image decoder | Yes* |
| `StreamingDecoder` | Incremental tile decoder | No (header only) |
| `TileDecoder` | Individual tile decoder | Yes* |
| `Jpeg2000Encoder` | JPEG2000 encoder | Yes* |

*Full decode/encode requires `ffi-openjpeg` feature. Without FFI, provides metadata extraction only.

### Marker Types

```rust
use transcode_jpeg2000::MarkerType;

// Start/End markers
MarkerType::Soc  // Start of codestream (0xFF4F)
MarkerType::Eoc  // End of codestream (0xFFD9)

// Header markers
MarkerType::Siz  // Image and tile size (0xFF51)
MarkerType::Cod  // Coding style default (0xFF52)
MarkerType::Qcd  // Quantization default (0xFF5C)
MarkerType::Com  // Comment (0xFF64)

// Tile markers
MarkerType::Sot  // Start of tile-part (0xFF90)
MarkerType::Sod  // Start of data (0xFF93)
```

### Profiles and Configuration

```rust
use transcode_jpeg2000::{Jpeg2000Profile, WaveletTransform, ProgressionOrder};

// Profiles
Jpeg2000Profile::Cinema2k    // DCI 2K digital cinema
Jpeg2000Profile::Cinema4k    // DCI 4K digital cinema
Jpeg2000Profile::Broadcast   // Broadcast contribution
Jpeg2000Profile::Imf         // Interoperable Master Format
Jpeg2000Profile::Lossless    // Lossless archival

// Wavelet transforms
WaveletTransform::Reversible_5_3   // Lossless
WaveletTransform::Irreversible_9_7 // Lossy

// Progression orders
ProgressionOrder::Lrcp  // Layer-Resolution-Component-Position
ProgressionOrder::Rlcp  // Resolution-Layer-Component-Position
ProgressionOrder::Rpcl  // Resolution-Position-Component-Layer
ProgressionOrder::Pcrl  // Position-Component-Resolution-Layer
ProgressionOrder::Cprl  // Component-Position-Resolution-Layer
```

## Error Handling

```rust
use transcode_jpeg2000::{Jpeg2000Error, Result};

fn process_image(data: &[u8]) -> Result<()> {
    match parse_codestream(data) {
        Ok(info) => { /* process */ }
        Err(Jpeg2000Error::InvalidCodestream(msg)) => {
            eprintln!("Invalid codestream: {}", msg);
        }
        Err(Jpeg2000Error::MissingMarker { marker }) => {
            eprintln!("Missing required marker: {:?}", marker);
        }
        Err(Jpeg2000Error::InvalidDimensions { width, height }) => {
            eprintln!("Invalid dimensions: {}x{}", width, height);
        }
        Err(Jpeg2000Error::FfiNotAvailable) => {
            eprintln!("Enable ffi-openjpeg feature for full decoding");
        }
        Err(e) => return Err(e),
    }
    Ok(())
}
```

## DCP Workflow Example

```rust
use transcode_jpeg2000::{Jpeg2000Encoder, Jpeg2000EncoderConfig};

/// Encode a frame sequence for Digital Cinema Package
fn encode_dcp_sequence(
    frames: &[Vec<u8>],
    width: u32,
    height: u32,
    frame_rate: f64,
) -> Result<Vec<Vec<u8>>> {
    // Select profile based on resolution
    let config = if width <= 2048 {
        Jpeg2000EncoderConfig::cinema_2k()
    } else {
        Jpeg2000EncoderConfig::cinema_4k()
    }
    .with_frame_rate(frame_rate);

    let mut encoder = Jpeg2000Encoder::new(config)?;
    let mut encoded_frames = Vec::new();

    for frame in frames {
        let encoded = encoder.encode(frame, width, height, 3)?;
        encoded_frames.push(encoded);
    }

    Ok(encoded_frames)
}
```

## Technical Details

### Codestream Structure

```
+--------+--------+--------+--------+--------+--------+
|  SOC   |  SIZ   |  COD   |  QCD   |  SOT   |  SOD   | ... | EOC |
+--------+--------+--------+--------+--------+--------+
   ↑        ↑        ↑        ↑        ↑        ↑           ↑
   |        |        |        |        |        |           |
Start   Size    Coding  Quant   Tile   Data       End
        Info    Style   Params  Start  Start
```

### Bitrate Calculation

For cinema profiles, maximum bitrate per frame:
- **Cinema 2K @ 24fps**: 250 Mbps ÷ 24 = ~1.3 MB/frame
- **Cinema 4K @ 24fps**: 500 Mbps ÷ 24 = ~2.6 MB/frame

### Color Space Support

| Color Space | Components | Typical Use |
|-------------|------------|-------------|
| Grayscale | 1 | Medical, archival |
| sRGB | 3 | General purpose |
| YCbCr | 3 | Video content |
| XYZ | 3 | Digital cinema |
| CMYK | 4 | Print preparation |

## See Also

- [transcode-images](../transcode-images) - JPEG, PNG, GIF codecs
- [transcode-openexr](../transcode-openexr) - OpenEXR HDR images
- [transcode-mxf](../transcode-mxf) - MXF container (used with DCP)
- [ISO/IEC 15444-1](https://www.iso.org/standard/78321.html) - JPEG2000 Core
- [DCI Specification](https://www.dcimovies.com/) - Digital Cinema Initiatives

## License

This crate is dual-licensed under MIT or Apache-2.0.
