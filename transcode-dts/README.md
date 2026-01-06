# transcode-dts

DTS and Dolby TrueHD audio codec support for the Transcode library.

## Overview

This crate provides parsing, decoding, and encoding support for DTS and Dolby TrueHD audio formats commonly found in Blu-ray discs, streaming services, and home theater systems.

> **Important**: DTS and TrueHD are patent-encumbered codecs with licensing requirements.
> - DTS formats are licensed by DTS, Inc.
> - TrueHD is licensed by Dolby Laboratories.
>
> This crate provides native bitstream parsing. Full encoding/decoding requires the `ffi-ffmpeg` feature and appropriate licenses.

## Features

- **Native Parsing**: Parse DTS and TrueHD sync frames without external dependencies
- **Metadata Extraction**: Extract sample rate, channels, bit depth, and profile information
- **Format Detection**: Automatic detection of DTS Core, DTS-HD, TrueHD, and Atmos streams
- **FFI Support**: Optional full decode/encode via FFmpeg (requires `ffi-ffmpeg` feature)

## Supported Formats

| Format | Description | Max Channels | Sample Rates |
|--------|-------------|--------------|--------------|
| DTS Core | Original DTS Digital Surround | 5.1 | 48kHz |
| DTS-HD HR | High Resolution lossy extension | 7.1 | 96kHz |
| DTS-HD MA | Master Audio lossless | 7.1 | 192kHz |
| DTS:X | Object-based immersive audio | 11.1+ | 48kHz |
| TrueHD | Dolby's lossless format | 7.1 | 192kHz |
| Atmos via TrueHD | Object-based audio layer | 7.1.4+ | 48kHz |

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-dts = "0.1"

# For full encoding/decoding (requires FFmpeg and licenses)
# transcode-dts = { version = "0.1", features = ["ffi-ffmpeg"] }
```

## Usage

### Parse DTS Sync Frame

```rust
use transcode_dts::{DtsParser, DtsSyncFrame};

let data = include_bytes!("audio.dts");
let mut parser = DtsParser::new();

if let Some(frame) = parser.parse_sync_frame(data) {
    println!("Sample rate: {} Hz", frame.sample_rate);
    println!("Channels: {} ({})", frame.channels, frame.amode);
    println!("Bit rate: {} kbps", frame.bit_rate);
    println!("Frame size: {} bytes", frame.frame_size);

    if frame.is_hd() {
        println!("DTS-HD extension detected");
    }
}
```

### Parse TrueHD Stream

```rust
use transcode_dts::{TrueHdParser, TrueHdSyncFrame};

let data = include_bytes!("audio.thd");
let mut parser = TrueHdParser::new();

if let Some(frame) = parser.parse_sync_frame(data) {
    println!("Sample rate: {} Hz", frame.sample_rate);
    println!("Channels: {}", frame.channel_assignment);
    println!("Bit depth: {} bits", frame.bit_depth);

    if frame.is_atmos() {
        println!("Dolby Atmos metadata present");
    }
}
```

### Streaming Decoder

```rust
use transcode_dts::{DtsDecoder, StreamingDecoder};

let mut decoder = StreamingDecoder::new()?;

// Feed data incrementally
for chunk in audio_chunks {
    decoder.feed(&chunk)?;

    while let Some(frame_info) = decoder.next_frame()? {
        println!("Decoded frame: {} samples", frame_info.samples_per_channel);
    }
}
```

### Encoding (requires `ffi-ffmpeg`)

```rust
use transcode_dts::{DtsEncoder, DtsEncoderConfig, DtsProfile};

let config = DtsEncoderConfig::surround_5_1(48000)
    .with_bitrate(1509)
    .with_profile(DtsProfile::Core);

let mut encoder = DtsEncoder::new(config)?;

// Encode PCM samples
let pcm_samples: Vec<f32> = get_audio_samples();
let encoded = encoder.encode_frame(&pcm_samples)?;
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `DtsSyncFrame` | Parsed DTS core frame header |
| `DtsHdFrame` | DTS-HD extension frame information |
| `TrueHdSyncFrame` | Parsed TrueHD/MLP frame header |
| `AudioMode` | DTS channel configuration enum |
| `ChannelLayout` | Detailed channel layout descriptor |
| `DecodedAudio` | Decoded audio buffer with metadata |

### Parsers

| Type | Description |
|------|-------------|
| `DtsParser` | Parse DTS core and HD sync frames |
| `TrueHdParser` | Parse TrueHD/MLP sync frames |

### Decoders/Encoders

| Type | Description | FFI Required |
|------|-------------|--------------|
| `DtsDecoder` | DTS audio decoder | Yes* |
| `TrueHdDecoder` | TrueHD audio decoder | Yes* |
| `DtsEncoder` | DTS audio encoder | Yes* |
| `StreamingDecoder` | Incremental frame decoder | No (header only) |

*Full decode/encode requires `ffi-ffmpeg` feature. Without FFI, decoders provide metadata extraction only.

### Constants

```rust
// Sync words for format detection
pub const DTS_SYNC_WORD_BE: u32 = 0x7FFE8001;
pub const DTS_HD_SYNC: u32 = 0x64582025;
pub const TRUEHD_SYNC: u32 = 0xF8726FBA;

// Frame parameters
pub const DTS_SAMPLES_PER_FRAME: usize = 512;
pub const DTS_HD_SAMPLES_PER_FRAME: usize = 4096;
pub const TRUEHD_SAMPLES_PER_FRAME: usize = 40;
```

## Error Handling

```rust
use transcode_dts::{DtsError, Result};

fn process_audio(data: &[u8]) -> Result<()> {
    match parse_frame(data) {
        Ok(frame) => { /* process */ }
        Err(DtsError::InvalidSyncWord(word)) => {
            eprintln!("Not a valid DTS stream: {:#x}", word);
        }
        Err(DtsError::InsufficientData { needed, available }) => {
            eprintln!("Need {} more bytes", needed - available);
        }
        Err(DtsError::FfiNotAvailable) => {
            eprintln!("Enable ffi-ffmpeg feature for full decoding");
        }
        Err(e) => return Err(e),
    }
    Ok(())
}
```

## Licensing Considerations

### For Development/Testing
- Parsing and metadata extraction work without any licenses
- You can analyze DTS/TrueHD streams freely

### For Production Use
- Contact [DTS Licensing](https://www.dts.com/licensing) for DTS codec licenses
- Contact [Dolby Licensing](https://www.dolby.com/about/licensing) for TrueHD/Atmos licenses
- Ensure your product/service has appropriate patent licenses before deploying

## See Also

- [transcode-ac3](../transcode-ac3) - AC-3/E-AC-3 (Dolby Digital) codec
- [transcode-spatial](../transcode-spatial) - Spatial audio processing
- [DTS Specification](https://www.etsi.org/deliver/etsi_ts/102100_102199/102114/) - ETSI TS 102 114

## License

This crate is dual-licensed under MIT or Apache-2.0. Note that this covers the code only; using DTS or TrueHD codecs may require additional patent licenses.
