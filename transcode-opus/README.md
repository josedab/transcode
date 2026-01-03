# transcode-opus

Opus audio codec implementation for the Transcode project.

Opus is a versatile audio codec combining two coding modes:
- **SILK**: Optimized for speech (6-40 kbps)
- **CELT**: Optimized for music (12-256 kbps)
- **Hybrid**: Combines both for high-quality voice (16-128 kbps)

## Features

- Variable bitrate (VBR), constrained VBR, and CBR modes
- Sample rates: 8, 12, 16, 24, 48 kHz
- Mono and stereo channels
- Frame sizes: 2.5, 5, 10, 20, 40, 60 ms
- Packet loss concealment (PLC)
- Low latency (down to 2.5 ms)
- Automatic speech/music detection
- In-band forward error correction (FEC)

## Key Types

| Type | Description |
|------|-------------|
| `OpusDecoder` | Decodes Opus packets to PCM samples |
| `OpusEncoder` | Encodes PCM samples to Opus packets |
| `OpusEncoderConfig` | Encoder configuration (bitrate, application, etc.) |
| `OpusMode` | Codec mode: `Silk`, `Celt`, or `Hybrid` |
| `Bandwidth` | Audio bandwidth: `Narrowband` to `Fullband` |
| `Application` | Optimization hint: `Voip`, `Audio`, or `LowDelay` |
| `OpusPacketInfo` | Parsed packet metadata from TOC byte |

## Usage

### Decoding

```rust
use transcode_opus::OpusDecoder;

let mut decoder = OpusDecoder::new(48000, 2)?;
let samples = decoder.decode_packet(&opus_packet)?;

// Handle packet loss
let concealed = decoder.conceal_packet_loss()?;
```

### Encoding

```rust
use transcode_opus::{OpusEncoder, OpusEncoderConfig};

// Default config (48kHz stereo, 64kbps)
let config = OpusEncoderConfig::default();
let mut encoder = OpusEncoder::new(config)?;
let packet = encoder.encode(&samples)?;

// Voice-optimized config
let voice_config = OpusEncoderConfig::for_voice(16000, 1);

// Music-optimized config
let music_config = OpusEncoderConfig::for_music(48000, 2);

// Low-latency config
let low_delay_config = OpusEncoderConfig::for_low_latency(48000, 2);
```

### Encoder Configuration

```rust
let mut encoder = OpusEncoder::new(OpusEncoderConfig::default())?;

// Adjust settings at runtime
encoder.set_bitrate(128000)?;
encoder.set_complexity(5)?;
encoder.set_signal_type(SignalType::Music);
encoder.set_fec(true);
encoder.set_packet_loss_perc(10);
```

### Packet Inspection

```rust
use transcode_opus::OpusPacketInfo;

if let Some(info) = OpusPacketInfo::parse(&packet) {
    println!("Mode: {}", info.mode);
    println!("Bandwidth: {}", info.bandwidth);
    println!("Duration: {} ms", info.duration_ms());
    println!("Stereo: {}", info.stereo);
}
```

## Codec Modes

The encoder automatically selects the optimal mode based on:
- Signal content (voice vs. music)
- Target bitrate
- Application type hint

Use `SignalType::Voice` or `SignalType::Music` to override automatic detection.

## License

See the repository root for license information.
