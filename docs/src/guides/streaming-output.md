# Streaming Output

This guide covers HLS and DASH streaming output using the `transcode-streaming` crate.

## Overview

Transcode supports two major adaptive streaming formats:

| Format | Standard | Playlist | Segments |
|--------|----------|----------|----------|
| HLS | Apple | .m3u8 | .ts or .fmp4 |
| DASH | MPEG | .mpd | .m4s |

## HLS (HTTP Live Streaming)

### Basic Usage

```rust
use transcode_streaming::{HlsMuxer, HlsConfig};

let config = HlsConfig::default()
    .segment_duration(6.0)
    .playlist_type(PlaylistType::Vod);

let mut hls = HlsMuxer::new("output/", config)?;

// Write segments
for segment in segments {
    hls.write_segment(&segment)?;
}

// Finalize and write playlist
hls.finish()?;
```

### Multi-Bitrate

```rust
use transcode_streaming::{HlsMuxer, HlsVariant};

let variants = vec![
    HlsVariant::new(1920, 1080, 5_000_000),  // 1080p @ 5 Mbps
    HlsVariant::new(1280, 720, 2_500_000),   // 720p @ 2.5 Mbps
    HlsVariant::new(854, 480, 1_000_000),    // 480p @ 1 Mbps
];

let mut hls = HlsMuxer::builder()
    .output_dir("output/")
    .variants(variants)
    .build()?;
```

### Configuration Options

```rust
let config = HlsConfig::default()
    .segment_duration(6.0)           // Target segment duration
    .playlist_type(PlaylistType::Vod) // VOD or Event or Live
    .version(7)                       // HLS version
    .independent_segments(true)       // Each segment independently decodable
    .program_date_time(true)          // Include timestamps
    .discontinuity_sequence(0);       // Starting sequence
```

### Segment Types

```rust
use transcode_streaming::SegmentFormat;

// MPEG-TS segments (traditional)
let config = HlsConfig::default()
    .segment_format(SegmentFormat::Ts);

// fMP4 segments (modern, required for HEVC/AV1)
let config = HlsConfig::default()
    .segment_format(SegmentFormat::Fmp4);
```

## DASH (Dynamic Adaptive Streaming over HTTP)

### Basic Usage

```rust
use transcode_streaming::{DashMuxer, DashConfig};

let config = DashConfig::default()
    .segment_duration(4.0)
    .profile(DashProfile::OnDemand);

let mut dash = DashMuxer::new("output/", config)?;

// Write segments
for segment in segments {
    dash.write_segment(&segment)?;
}

// Generate MPD
dash.finish()?;
```

### Multi-Bitrate

```rust
use transcode_streaming::{DashMuxer, DashRepresentation};

let representations = vec![
    DashRepresentation::video(1920, 1080, 5_000_000),
    DashRepresentation::video(1280, 720, 2_500_000),
    DashRepresentation::video(854, 480, 1_000_000),
    DashRepresentation::audio(128_000, "en"),
    DashRepresentation::audio(64_000, "en"),
];

let mut dash = DashMuxer::builder()
    .output_dir("output/")
    .representations(representations)
    .build()?;
```

### Configuration Options

```rust
let config = DashConfig::default()
    .segment_duration(4.0)               // Target duration
    .profile(DashProfile::OnDemand)       // Profile
    .min_buffer_time(2.0)                 // Minimum buffer
    .suggested_presentation_delay(10.0);  // Live edge delay
```

## Segment Management

### Init Segments

```rust
use transcode_streaming::InitSegment;

// Create initialization segment
let init = InitSegment::new(&codec_params)?;

// Write init segment
hls.write_init_segment(&init)?;
```

### Media Segments

```rust
use transcode_streaming::MediaSegment;

let segment = MediaSegment::builder()
    .sequence(0)
    .duration(6.0)
    .pts(0)
    .data(encoded_data)
    .keyframe(true)
    .build()?;

hls.write_segment(&segment)?;
```

## Live Streaming

### HLS Live

```rust
let config = HlsConfig::default()
    .playlist_type(PlaylistType::Live)
    .target_duration(6)
    .playlist_length(5);  // Keep 5 segments in playlist

let mut hls = HlsMuxer::new("output/", config)?;

// Continuously write segments
loop {
    let segment = encode_next_segment()?;
    hls.write_segment(&segment)?;

    // Older segments automatically removed from playlist
}
```

### DASH Live

```rust
let config = DashConfig::default()
    .profile(DashProfile::Live)
    .availability_start_time(Utc::now())
    .time_shift_buffer_depth(30.0);  // 30 second DVR window

let mut dash = DashMuxer::new("output/", config)?;
```

## Encryption

### HLS with AES-128

```rust
use transcode_streaming::{HlsConfig, Encryption};

let config = HlsConfig::default()
    .encryption(Encryption::Aes128 {
        key_uri: "https://example.com/key",
        iv: Some([0u8; 16]),
    });
```

### DASH with Widevine

```rust
use transcode_streaming::{DashConfig, DrmConfig};

let config = DashConfig::default()
    .drm(DrmConfig::Widevine {
        license_url: "https://example.com/license",
        pssh: pssh_box,
    });
```

## Complete Example

```rust
use transcode::{Transcoder, TranscodeOptions};
use transcode_streaming::{HlsMuxer, HlsConfig, HlsVariant};

fn create_hls_output(input: &str) -> Result<()> {
    let variants = vec![
        HlsVariant::new(1920, 1080, 5_000_000),
        HlsVariant::new(1280, 720, 2_500_000),
        HlsVariant::new(854, 480, 1_000_000),
    ];

    let config = HlsConfig::default()
        .segment_duration(6.0)
        .playlist_type(PlaylistType::Vod);

    let mut hls = HlsMuxer::builder()
        .output_dir("output/")
        .variants(variants)
        .config(config)
        .build()?;

    // Transcode each variant
    for (i, variant) in hls.variants().enumerate() {
        let options = TranscodeOptions::new()
            .input(input)
            .output(variant.segment_pattern())
            .video_bitrate(variant.bitrate)
            .resolution(variant.width, variant.height);

        let mut transcoder = Transcoder::new(options)?;
        transcoder.run()?;
    }

    hls.finish()?;
    Ok(())
}
```

## Output Structure

### HLS

```
output/
├── master.m3u8           # Master playlist
├── 1080p/
│   ├── playlist.m3u8     # Variant playlist
│   ├── init.mp4          # Init segment
│   ├── segment0.m4s
│   └── segment1.m4s
├── 720p/
│   └── ...
└── 480p/
    └── ...
```

### DASH

```
output/
├── manifest.mpd          # MPD manifest
├── init-v1.m4s           # Video init
├── init-a1.m4s           # Audio init
├── video-1080p/
│   ├── seg-0.m4s
│   └── seg-1.m4s
└── audio-128k/
    ├── seg-0.m4s
    └── seg-1.m4s
```
