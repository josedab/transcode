# transcode-vorbis

A pure Rust Vorbis audio codec implementation for the transcode library.

## Features

- **Full Decoder**: Complete Vorbis I specification support
- **Full Encoder**: VBR and ABR encoding modes
- **Quality Range**: -2 to 10 quality levels
- **Multi-channel**: Up to 8 channels with proper coupling
- **Sample Rates**: 8 kHz to 192 kHz
- **Pure Rust**: No external dependencies

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-vorbis = { path = "../transcode-vorbis" }
```

### Decoding

```rust
use transcode_vorbis::VorbisDecoder;

let mut decoder = VorbisDecoder::new();

// Feed identification, comment, and setup headers
decoder.init(&id_header, &comment_header, &setup_header)?;

// Decode audio packets
let samples = decoder.decode(&audio_packet)?;
```

### Encoding

```rust
use transcode_vorbis::{VorbisEncoder, VorbisConfig};

// Create encoder with quality 5 (good quality, ~160 kbps for stereo)
let config = VorbisConfig::new(44100, 2).with_quality(5.0);
let mut encoder = VorbisEncoder::new(config)?;

// Get header packets for container
let headers = encoder.headers();

// Encode audio samples
let packet = encoder.encode(&samples)?;

// Flush remaining data
let final_packets = encoder.flush()?;
```

## Quality Levels

| Quality | Approximate Bitrate | Use Case |
|---------|---------------------|----------|
| -2 | ~32 kbps | Speech, low bandwidth |
| 0 | ~64 kbps | Acceptable quality |
| 3 | ~112 kbps | Good quality |
| 5 | ~160 kbps | High quality (default) |
| 7 | ~224 kbps | Very high quality |
| 10 | ~500 kbps | Transparent quality |

## Technical Details

Vorbis uses:
- Modified Discrete Cosine Transform (MDCT)
- Huffman and vector quantization codebooks
- Floor curves for spectral envelope
- Residue coding for spectral details
- Channel coupling for stereo efficiency

## Container Support

Vorbis is typically used with:
- **OGG**: Native container (use with transcode-containers)
- **WebM**: Web video container
- **MKV**: Matroska container

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
