# Audio Codecs

This guide covers the audio codecs supported by Transcode.

## Codec Overview

| Codec | Encode | Decode | Type | Best For |
|-------|--------|--------|------|----------|
| AAC | ✅ | ✅ | Lossy | Streaming, compatibility |
| Opus | ✅ | ✅ | Lossy | VoIP, web, low bitrate |
| MP3 | ❌ | ✅ | Lossy | Legacy compatibility |
| FLAC | ✅ | ✅ | Lossless | Archival |
| AC-3 | ✅ | ✅ | Lossy | Surround, broadcast |
| DTS | ✅ | ✅ | Lossy | Cinema, Blu-ray |
| ALAC | ✅ | ✅ | Lossless | Apple ecosystem |
| Vorbis | ✅ | ✅ | Lossy | WebM, games |
| PCM | ✅ | ✅ | Uncompressed | Editing, archival |

## AAC (Advanced Audio Coding)

The most widely supported lossy audio codec.

### Profiles

| Profile | Features | Use Case |
|---------|----------|----------|
| LC (Low Complexity) | Standard | Most applications |
| HE (High Efficiency) | SBR | Low bitrate streaming |
| HE v2 | SBR + PS | Very low bitrate |

```rust
use transcode::codecs::aac::{AacEncoder, AacProfile};

let encoder = AacEncoder::new()
    .profile(AacProfile::Lc)
    .bitrate(128_000)
    .sample_rate(44100)
    .build()?;
```

### Recommended Bitrates

| Quality | Stereo | 5.1 |
|---------|--------|-----|
| Low | 96 kbps | 256 kbps |
| Standard | 128 kbps | 384 kbps |
| High | 192 kbps | 512 kbps |
| Transparent | 256 kbps | 640 kbps |

## Opus

Modern, royalty-free codec. Excellent at all bitrates.

```rust
use transcode::codecs::opus::{OpusEncoder, Application};

let encoder = OpusEncoder::new()
    .application(Application::Audio)  // or VoIP, LowDelay
    .bitrate(128_000)
    .build()?;
```

### Use Cases

| Application | Bitrate | Latency |
|-------------|---------|---------|
| VoIP | 16-32 kbps | 20ms |
| Music streaming | 96-128 kbps | 60ms |
| High quality | 160-256 kbps | 60ms |

## FLAC (Free Lossless Audio Codec)

Lossless compression, typically 50-70% of original size.

```rust
use transcode::codecs::flac::{FlacEncoder, CompressionLevel};

let encoder = FlacEncoder::new()
    .compression_level(CompressionLevel::Five)  // 0-8
    .build()?;
```

### Compression Levels

| Level | Speed | Size |
|-------|-------|------|
| 0 | Fastest | Largest |
| 5 | Balanced | Medium |
| 8 | Slowest | Smallest |

## AC-3 (Dolby Digital)

Standard for surround sound in broadcast and DVDs.

```rust
use transcode::codecs::ac3::{Ac3Encoder, ChannelLayout};

let encoder = Ac3Encoder::new()
    .channel_layout(ChannelLayout::FiveOne)
    .bitrate(448_000)
    .build()?;
```

### Channel Configurations

| Layout | Channels | Typical Bitrate |
|--------|----------|-----------------|
| Stereo | 2.0 | 192 kbps |
| Surround | 5.1 | 384-448 kbps |
| Atmos | 7.1+ | 640+ kbps |

## Choosing a Codec

```
Need maximum compatibility?      → AAC-LC
Need low bitrate efficiency?     → Opus
Need lossless archival?          → FLAC
Need surround for broadcast?     → AC-3 or E-AC-3
Need Apple ecosystem?            → AAC or ALAC
Need WebM compatibility?         → Opus or Vorbis
```

## Sample Rate Considerations

| Content | Sample Rate |
|---------|-------------|
| Speech/Podcast | 44.1 kHz |
| Music | 44.1 or 48 kHz |
| Video sync | 48 kHz |
| Hi-res audio | 96 or 192 kHz |

```rust
let options = TranscodeOptions::new()
    .audio_codec(Codec::Aac)
    .sample_rate(48000)  // Match video
    .audio_bitrate(128_000);
```

## Channel Downmixing

Convert surround to stereo:

```rust
let options = TranscodeOptions::new()
    .audio_channels(2)  // Downmix to stereo
    .downmix_algorithm(Downmix::DplII);  // Dolby Pro Logic II
```
