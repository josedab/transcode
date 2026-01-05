# transcode-flac

A pure Rust FLAC (Free Lossless Audio Codec) implementation with streaming support.

## Features

- **Full Decoder**: Metadata parsing, all prediction types, CRC validation
- **Full Encoder**: Configurable compression levels (0-8)
- **Streaming Support**: Real-time decoding for live applications
- **Metadata Support**: Vorbis comments, pictures, seek tables
- **Pure Rust**: No external dependencies for codec functionality

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-flac = { path = "../transcode-flac" }
```

### Decoding

```rust
use transcode_flac::FlacDecoder;
use std::fs::File;
use std::io::BufReader;

let file = File::open("audio.flac")?;
let mut decoder = FlacDecoder::new(BufReader::new(file))?;

// Access metadata
if let Some(info) = &decoder.metadata().stream_info {
    println!("Sample rate: {} Hz", info.sample_rate);
    println!("Channels: {}", info.channels);
    println!("Bit depth: {}", info.bits_per_sample);
}

// Decode frames
while let Some(frame) = decoder.next_frame()? {
    // Process decoded samples (interleaved i32)
    process_samples(&frame.samples);
}
```

### Encoding

```rust
use transcode_flac::{FlacEncoder, CompressionLevel};
use std::fs::File;

let output = File::create("output.flac")?;
let mut encoder = FlacEncoder::new(
    output,
    44100,  // sample rate
    2,      // channels
    16,     // bits per sample
    CompressionLevel::Default
)?;

encoder.encode_samples(&samples)?;
encoder.finish()?;
```

### Streaming Decoder

```rust
use transcode_flac::StreamingDecoder;

let mut decoder = StreamingDecoder::new();

// Feed data incrementally
decoder.feed(&chunk1);
decoder.feed(&chunk2);

// Decode available frames
while let Some(frame) = decoder.decode_frame()? {
    process_samples(&frame.samples);
}
```

## Compression Levels

| Level | Speed | Compression | Use Case |
|-------|-------|-------------|----------|
| 0 | Fastest | Lowest | Real-time encoding |
| 5 | Default | Balanced | General purpose |
| 8 | Slowest | Highest | Archival |

## Metadata Support

- **StreamInfo**: Sample rate, channels, bit depth, total samples
- **VorbisComment**: Tags (artist, title, album, etc.)
- **Picture**: Album art and other images
- **SeekTable**: Fast random access

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
