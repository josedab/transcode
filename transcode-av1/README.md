# transcode-av1

AV1 codec support for the transcode library, providing high-quality video encoding using [rav1e](https://github.com/xiph/rav1e).

## Features

- **AV1 Encoding**: Pure Rust encoder via rav1e (fastest and safest AV1 encoder)
- **8-bit and 10-bit support**: Full support for both bit depths
- **HDR content**: BT.2020 color primaries and PQ/HLG transfer functions
- **Multiple rate control modes**: Constant quality (CQP), VBR, and CBR
- **Tile-based parallelism**: Configurable tile rows/columns for faster encoding
- **Speed presets**: From Placebo (best quality) to UltraFast (fastest)

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-av1 = { path = "../transcode-av1" }
```

### Basic Encoding

```rust
use transcode_av1::{Av1Encoder, Av1Config, Av1Preset};

// Create encoder configuration
let config = Av1Config::new(1920, 1080)
    .with_preset(Av1Preset::Medium)
    .with_bitrate(2_000_000);  // 2 Mbps VBR

let mut encoder = Av1Encoder::new(config)?;

// Encode YUV420 frames
let packet = encoder.encode_frame(&y_plane, &u_plane, &v_plane, pts)?;

// Flush remaining packets when done
let remaining = encoder.flush()?;
```

### Configuration Options

```rust
let config = Av1Config::new(3840, 2160)
    .with_preset(Av1Preset::Fast)       // Speed preset (0-10)
    .with_quality(28)                    // Constant quality (0-63, lower = better)
    .with_bitrate(10_000_000)            // Target bitrate in bps
    .with_bit_depth(10)                  // 8 or 10 bit
    .with_tiles(2, 2)                    // Tile columns/rows (log2)
    .with_framerate(60, 1)               // 60 fps
    .with_keyframe_interval(240)         // Keyframe every 240 frames
    .with_low_latency(true)              // Low latency mode
    .with_threads(8)                     // Thread count (0 = auto)
    .with_hdr()                          // Enable HDR (10-bit, BT.2020, PQ)
    .with_content_type(ContentType::Film);
```

### Speed Presets

| Preset     | Speed | Use Case                    |
|------------|-------|-----------------------------|
| Placebo    | 0     | Maximum quality, very slow  |
| VerySlow   | 2     | Archival encoding           |
| Slower     | 3     | High quality distribution   |
| Slow       | 4     | Quality-focused             |
| Medium     | 6     | Balanced (default)          |
| Fast       | 7     | Faster encoding             |
| Faster     | 8     | Real-time capable           |
| VeryFast   | 9     | Streaming                   |
| UltraFast  | 10    | Preview/testing             |

### Rate Control

```rust
use transcode_av1::RateControlMode;

// Constant quality (recommended for quality-focused encoding)
RateControlMode::ConstantQuality { quantizer: 28 }

// Variable bitrate
RateControlMode::Vbr { bitrate: 5_000_000 }

// Constant bitrate (for streaming)
RateControlMode::Cbr { bitrate: 4_000_000 }
```

## Feature Flags

- `encoder` (default): Enables rav1e-based AV1 encoding

## Resolution Limits

- Maximum: 8192x4320 (8K UHD)
- Minimum: Non-zero dimensions required

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
