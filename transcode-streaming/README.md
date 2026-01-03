# transcode-streaming

HLS and DASH adaptive bitrate streaming output for the transcode library.

## Features

- **HLS**: Master/media playlist generation, LL-HLS support, EXT-X-PROGRAM-DATE-TIME
- **DASH**: MPD manifest generation, ISO on-demand/live profiles, low-latency DASH
- **Adaptive Bitrate**: Multiple quality levels with configurable resolutions and bitrates
- **Segment Management**: Sequential, time-based, or UUID naming strategies
- **DRM Integration**: Built-in DRM configuration support

## Key Types

| Type | Description |
|------|-------------|
| `HlsConfig` / `HlsWriter` | HLS configuration and segment writer |
| `DashConfig` / `DashWriter` | DASH configuration and segment writer |
| `Quality` | Video quality level (resolution, bitrate, codecs) |
| `Segment` | Media segment metadata |
| `MasterPlaylist` / `MediaPlaylist` | HLS playlist types |
| `MpdManifest` | DASH MPD manifest |

## Usage

### HLS Output

```rust
use transcode_streaming::{HlsConfig, HlsWriter, Quality};

let config = HlsConfig::new("output/hls")
    .with_segment_duration(6.0)
    .with_quality(Quality::fhd_1080p())
    .with_quality(Quality::hd_720p())
    .with_quality(Quality::sd_480p());

let mut writer = HlsWriter::new(config)?;

// Write segments for each quality level
writer.write_segment(0, &segment_data, 6.0, true)?;

// Finalize playlists
writer.finalize()?;
```

### DASH Output

```rust
use transcode_streaming::{DashConfig, DashWriter, Quality};

let config = DashConfig::new("output/dash")
    .with_segment_duration(4.0)
    .with_quality(Quality::fhd_1080p())
    .with_quality(Quality::hd_720p());

let mut writer = DashWriter::new(config)?;

// Write init segment
writer.write_init_segment(0, &init_data)?;

// Write media segments
writer.write_segment(0, &segment_data, 4.0, true)?;

// Finalize MPD manifest
writer.finalize()?;
```

### Quality Presets

```rust
use transcode_streaming::Quality;

Quality::uhd_4k();    // 3840x2160 @ 15 Mbps
Quality::fhd_1080p(); // 1920x1080 @ 5 Mbps
Quality::hd_720p();   // 1280x720  @ 2.5 Mbps
Quality::sd_480p();   // 854x480   @ 1 Mbps
Quality::low_360p();  // 640x360   @ 500 Kbps

// Custom quality
Quality::new(1280, 720, 3_000_000)
    .with_video_codec("hvc1.1.6.L93.B0")
    .with_framerate(60.0);
```

### Low-Latency Streaming

```rust
// LL-HLS
let config = HlsConfig::new("output")
    .with_low_latency(0.2); // 200ms parts

// Low-latency DASH
let config = DashConfig::new("output")
    .with_low_latency();
```

## Feature Flags

- `hls` - HLS support (enabled by default)
- `dash` - DASH support (enabled by default)

## Documentation

See the main [transcode documentation](../README.md) for complete library usage.
