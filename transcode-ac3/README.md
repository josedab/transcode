# transcode-ac3

AC-3 (Dolby Digital) and E-AC-3 (Enhanced AC-3/Dolby Digital Plus) audio codec support for the transcode library.

## Important Notice

**AC-3 and E-AC-3 are patent-encumbered codecs owned by Dolby Laboratories.** This crate provides native bitstream parsing; full encoding/decoding requires the `ffi-ffmpeg` feature and proper licensing.

## Features

- **Bitstream Parsing**: Native AC-3/E-AC-3 sync frame detection and header parsing
- **Metadata Extraction**: Sample rate, channels, bitrate, dialnorm, compression profiles
- **E-AC-3 Support**: Enhanced AC-3 (Dolby Digital Plus) with up to 16 channels
- **FFI Integration**: Optional FFmpeg-based encoding/decoding via `ffi-ffmpeg` feature
- **CRC Validation**: Frame integrity checking

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-ac3 = { path = "../transcode-ac3" }
```

### Parsing AC-3 Bitstream

```rust
use transcode_ac3::{Ac3Parser, Ac3SyncFrame};

let data = vec![0x0B, 0x77, /* ... AC-3 frame data ... */];
let mut parser = Ac3Parser::new();

if let Some(frame) = parser.parse_sync_frame(&data) {
    println!("Sample rate: {} Hz", frame.sample_rate);
    println!("Channels: {}", frame.channels);
    println!("Bitrate: {} kbps", frame.bitrate / 1000);
}
```

### E-AC-3 (Dolby Digital Plus)

```rust
use transcode_ac3::{Eac3Parser, Eac3SyncFrame};

let data = vec![0x0B, 0x77, /* ... E-AC-3 frame data ... */];
let mut parser = Eac3Parser::new();

if let Some(frame) = parser.parse_sync_frame(&data) {
    println!("Sample rate: {} Hz", frame.sample_rate);
    println!("Stream type: {:?}", frame.stream_type);
}
```

## Specifications

| Parameter | AC-3 | E-AC-3 |
|-----------|------|--------|
| Max Channels | 6 (5.1) | 16 (7.1.4) |
| Sample Rates | 32, 44.1, 48 kHz | 32, 44.1, 48 kHz |
| Max Bitrate | 640 kbps | 6.144 Mbps |
| Samples/Frame | 1536 | 256-1536 |

## Feature Flags

- `ffi-ffmpeg`: Enable FFmpeg-based encoding/decoding (requires FFmpeg libraries)

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
