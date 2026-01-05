# transcode-mpeg2

MPEG-2 video codec support for the transcode library.

## Patent Notice

**MPEG-2 was patent-encumbered by MPEG-LA. Most patents have expired (2018-2019), but some may still apply in certain jurisdictions.**

## Features

- **Bitstream Parsing**: Native MPEG-2 video elementary stream parsing
- **Sequence Header Detection**: Resolution, frame rate, aspect ratio extraction
- **GOP Structure**: I/P/B frame detection and analysis
- **Profile Support**: Main Profile at various levels
- **FFI Integration**: Optional FFmpeg-based full decoding via `ffi-ffmpeg` feature

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-mpeg2 = { path = "../transcode-mpeg2" }
```

### Parsing MPEG-2 Video

```rust
use transcode_mpeg2::{Mpeg2Parser, SequenceHeader};

let data = vec![0x00, 0x00, 0x01, 0xB3, /* ... sequence header ... */];
let mut parser = Mpeg2Parser::new();

if let Some(seq) = parser.parse_sequence_header(&data) {
    println!("Resolution: {}x{}", seq.horizontal_size, seq.vertical_size);
    println!("Frame rate: {:.3} fps", seq.frame_rate());
    println!("Aspect ratio: {:?}", seq.aspect_ratio);
}
```

### Decoding (with FFI)

```rust
use transcode_mpeg2::Mpeg2Decoder;

let mut decoder = Mpeg2Decoder::new()?;
let frame = decoder.decode(&encoded_data)?;
```

### Encoding (with FFI)

```rust
use transcode_mpeg2::{Mpeg2Encoder, Mpeg2EncoderConfig};

let config = Mpeg2EncoderConfig::new(720, 480)
    .with_bitrate(8_000_000)
    .with_gop_size(15);

let mut encoder = Mpeg2Encoder::new(config)?;
let packet = encoder.encode(&frame)?;
```

## Start Codes

| Code | Description |
|------|-------------|
| 0x000001B3 | Sequence Header |
| 0x000001B5 | Extension |
| 0x000001B8 | GOP |
| 0x00000100 | Picture |
| 0x01-0xAF | Slices |

## Feature Flags

- `ffi-ffmpeg`: Enable FFmpeg-based encoding/decoding

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
