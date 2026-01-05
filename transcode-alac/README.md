# transcode-alac

Apple Lossless Audio Codec (ALAC) implementation for the transcode library.

## Features

- **Lossless Compression**: Bit-perfect audio reproduction
- **Multi-channel Support**: 1-8 channels (mono to 7.1 surround)
- **Multiple Bit Depths**: 16, 20, 24, and 32-bit support
- **High Sample Rates**: Up to 384 kHz
- **Adaptive Linear Prediction**: Efficient entropy coding

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-alac = { path = "../transcode-alac" }
```

### Decoding

```rust
use transcode_alac::{AlacDecoder, AlacConfig};

// Create config from magic cookie (from MP4/M4A container)
let config = AlacConfig::from_magic_cookie(&cookie)?;
let mut decoder = AlacDecoder::new(config)?;

// Decode compressed packets
let samples = decoder.decode(&compressed_packet)?;
```

### Configuration

```rust
use transcode_alac::AlacConfig;

// Create config for 44.1kHz stereo 16-bit
let config = AlacConfig::new(44100, 2, 16)?;

// Create config for 48kHz 5.1 surround 24-bit
let config = AlacConfig::new(48000, 6, 24)?;
```

### Encoding (with `encoder` feature)

```rust
use transcode_alac::{AlacEncoder, AlacEncoderConfig};

let config = AlacEncoderConfig::new(44100, 2, 16);
let mut encoder = AlacEncoder::new(config)?;
let packet = encoder.encode(&samples)?;
```

## Channel Layouts

| Layout | Channels | Description |
|--------|----------|-------------|
| Mono | 1 | Single channel |
| Stereo | 2 | L, R |
| 3.0 | 3 | L, R, C |
| 4.0 | 4 | L, R, Ls, Rs |
| 5.0 | 5 | L, R, C, Ls, Rs |
| 5.1 | 6 | L, R, C, LFE, Ls, Rs |
| 6.1 | 7 | L, R, C, LFE, Ls, Rs, Cs |
| 7.1 | 8 | L, R, C, LFE, Ls, Rs, Lrs, Rrs |

## Feature Flags

- `encoder` (default): Enable ALAC encoding support

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
