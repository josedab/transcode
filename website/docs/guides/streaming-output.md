---
sidebar_position: 2
title: Streaming Output
description: Generate HLS and DASH streaming content
---

# Streaming Output

Generate adaptive bitrate streaming content with HLS (HTTP Live Streaming) and DASH (Dynamic Adaptive Streaming over HTTP).

## HLS Streaming

HLS is Apple's streaming protocol, supported on all Apple devices and most modern browsers.

### Basic HLS Output

```rust
use transcode_streaming::{HlsMuxer, HlsConfig, PlaylistType};

let config = HlsConfig {
    segment_duration: 6.0,        // 6 seconds per segment
    playlist_type: PlaylistType::Vod,
    ..Default::default()
};

let muxer = HlsMuxer::new("output/", config)?;

// Feed encoded packets
muxer.write_packet(&video_packet)?;
muxer.write_packet(&audio_packet)?;

// Finalize
muxer.finish()?;
```

Output structure:
```
output/
├── playlist.m3u8          # Master playlist
├── stream_0/
│   ├── playlist.m3u8      # Variant playlist
│   ├── segment_000.ts
│   ├── segment_001.ts
│   └── segment_002.ts
```

### Multi-Bitrate HLS

Create adaptive streams with multiple quality levels:

```rust
use transcode_streaming::{HlsMuxer, HlsConfig, HlsVariant};

let variants = vec![
    HlsVariant {
        name: "1080p".to_string(),
        bandwidth: 5_000_000,
        resolution: Some((1920, 1080)),
        codecs: "avc1.640028,mp4a.40.2".to_string(),
    },
    HlsVariant {
        name: "720p".to_string(),
        bandwidth: 2_500_000,
        resolution: Some((1280, 720)),
        codecs: "avc1.64001f,mp4a.40.2".to_string(),
    },
    HlsVariant {
        name: "480p".to_string(),
        bandwidth: 1_000_000,
        resolution: Some((854, 480)),
        codecs: "avc1.64001e,mp4a.40.2".to_string(),
    },
];

let config = HlsConfig {
    segment_duration: 6.0,
    variants,
    ..Default::default()
};

let muxer = HlsMuxer::new("output/", config)?;
```

### Using the High-Level API

```rust
use transcode::{Transcoder, TranscodeOptions, StreamingFormat};

let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("stream/playlist.m3u8")
    .format(StreamingFormat::Hls)
    .segment_duration(6.0)
    .variants(vec![
        ("1080p", 5_000_000, 1920, 1080),
        ("720p", 2_500_000, 1280, 720),
        ("480p", 1_000_000, 854, 480),
    ]);

let mut transcoder = Transcoder::new(options)?;
transcoder.run()?;
```

CLI:
```bash
transcode -i input.mp4 -o stream/playlist.m3u8 \
  --format hls \
  --segment-duration 6 \
  --variants "1080p:5000:1920x1080,720p:2500:1280x720,480p:1000:854x480"
```

## DASH Streaming

DASH is the industry standard for adaptive streaming, supported by most platforms except Safari.

### Basic DASH Output

```rust
use transcode_streaming::{DashMuxer, DashConfig};

let config = DashConfig {
    segment_duration: 4.0,
    min_buffer_time: 2.0,
    ..Default::default()
};

let muxer = DashMuxer::new("output/", config)?;
muxer.write_packet(&video_packet)?;
muxer.finish()?;
```

Output structure:
```
output/
├── manifest.mpd           # DASH manifest
├── init_video.mp4         # Video initialization segment
├── init_audio.mp4         # Audio initialization segment
├── segment_video_001.m4s  # Video segments
├── segment_video_002.m4s
├── segment_audio_001.m4s  # Audio segments
└── segment_audio_002.m4s
```

### Multi-Bitrate DASH

```rust
use transcode_streaming::{DashMuxer, DashConfig, DashRepresentation};

let representations = vec![
    DashRepresentation {
        id: "v1080".to_string(),
        bandwidth: 5_000_000,
        width: 1920,
        height: 1080,
        ..Default::default()
    },
    DashRepresentation {
        id: "v720".to_string(),
        bandwidth: 2_500_000,
        width: 1280,
        height: 720,
        ..Default::default()
    },
];

let config = DashConfig {
    segment_duration: 4.0,
    representations,
    ..Default::default()
};
```

## Live Streaming

For live content, use event or live playlist types.

### HLS Live

```rust
let config = HlsConfig {
    segment_duration: 2.0,           // Shorter segments for lower latency
    playlist_type: PlaylistType::Event,  // Growing playlist
    max_segments: 10,                // Keep last 10 segments in playlist
    ..Default::default()
};
```

### DASH Live

```rust
let config = DashConfig {
    segment_duration: 2.0,
    min_buffer_time: 4.0,
    suggested_presentation_delay: 6.0,
    live: true,
    ..Default::default()
};
```

## Low-Latency HLS (LL-HLS)

For near-real-time streaming:

```rust
let config = HlsConfig {
    segment_duration: 4.0,
    part_duration: 0.5,              // 500ms partial segments
    playlist_type: PlaylistType::Event,
    low_latency: true,
    ..Default::default()
};
```

## Encryption (DRM)

### AES-128 Encryption for HLS

```rust
use transcode_streaming::{HlsConfig, HlsEncryption};

let encryption = HlsEncryption::Aes128 {
    key_uri: "https://example.com/key".to_string(),
    key: [0u8; 16],  // 128-bit key
    iv: None,        // Auto-generate IV
};

let config = HlsConfig {
    encryption: Some(encryption),
    ..Default::default()
};
```

### CENC for DASH (Widevine, PlayReady)

```rust
use transcode_streaming::{DashConfig, CencConfig};

let cenc = CencConfig {
    key_id: "...",
    key: "...",
    pssh: vec![
        PsshBox::widevine(...),
        PsshBox::playready(...),
    ],
};

let config = DashConfig {
    cenc: Some(cenc),
    ..Default::default()
};
```

## Codec Selection for Streaming

### HLS Compatible Codecs

| Video | Audio | Notes |
|-------|-------|-------|
| H.264 | AAC | Universal support |
| H.265 | AAC | Newer devices only |
| (no AV1) | (no Opus) | Limited HLS support |

### DASH Compatible Codecs

| Video | Audio | Notes |
|-------|-------|-------|
| H.264 | AAC | Universal |
| H.265 | AAC | Wide support |
| VP9 | Opus | Chrome, Firefox |
| AV1 | Opus | Modern browsers |

## Complete Streaming Pipeline

```rust
use transcode::{Transcoder, TranscodeOptions, StreamingFormat};
use transcode_streaming::{HlsConfig, HlsVariant};

fn create_streaming_content(input: &str, output_dir: &str) -> transcode::Result<()> {
    // Define quality levels
    let variants = vec![
        ("1080p", 5_000_000, 1920, 1080),
        ("720p", 2_500_000, 1280, 720),
        ("480p", 1_000_000, 854, 480),
        ("360p", 500_000, 640, 360),
    ];

    // Create HLS output
    let hls_options = TranscodeOptions::new()
        .input(input)
        .output(&format!("{}/hls/playlist.m3u8", output_dir))
        .format(StreamingFormat::Hls)
        .segment_duration(6.0)
        .video_codec("h264")
        .audio_codec("aac")
        .variants(variants.clone());

    let mut hls_transcoder = Transcoder::new(hls_options)?;
    hls_transcoder.run()?;

    // Create DASH output
    let dash_options = TranscodeOptions::new()
        .input(input)
        .output(&format!("{}/dash/manifest.mpd", output_dir))
        .format(StreamingFormat::Dash)
        .segment_duration(4.0)
        .video_codec("h264")
        .audio_codec("aac")
        .variants(variants);

    let mut dash_transcoder = Transcoder::new(dash_options)?;
    dash_transcoder.run()?;

    Ok(())
}
```

## Thumbnail Generation

Generate thumbnails for video preview:

```rust
use transcode::thumbnail::{ThumbnailGenerator, ThumbnailConfig};

let config = ThumbnailConfig {
    interval: 10.0,              // Every 10 seconds
    width: 320,
    height: 180,
    format: ImageFormat::Jpeg,
    quality: 80,
};

let generator = ThumbnailGenerator::new(config);
generator.generate("input.mp4", "thumbnails/")?;
```

Output:
```
thumbnails/
├── thumb_0000.jpg    # 0:00
├── thumb_0010.jpg    # 0:10
├── thumb_0020.jpg    # 0:20
└── ...
```

### WebVTT Thumbnails

Generate thumbnail sprites with WebVTT metadata:

```rust
let config = ThumbnailConfig {
    sprite: true,
    columns: 10,
    rows: 10,
    ..Default::default()
};

generator.generate_webvtt("input.mp4", "thumbnails/")?;
```

Output:
```
thumbnails/
├── sprite_0.jpg      # Sprite image
├── sprite_1.jpg
└── thumbnails.vtt    # WebVTT timing file
```

## Best Practices

### Segment Duration

| Use Case | HLS | DASH |
|----------|-----|------|
| VOD | 6-10s | 4-6s |
| Live | 2-4s | 2-4s |
| Low-latency | 1-2s | 1-2s |

### GOP Size

Align GOP (Group of Pictures) with segment duration:

```rust
let options = TranscodeOptions::new()
    .segment_duration(6.0)
    .gop_size(180);  // 6 seconds at 30fps
```

### Bandwidth Ladder

Recommended quality levels:

| Resolution | Bitrate | Frame Rate |
|------------|---------|------------|
| 1080p | 4-6 Mbps | 30/60 fps |
| 720p | 2-3 Mbps | 30 fps |
| 480p | 1-1.5 Mbps | 30 fps |
| 360p | 500-800 kbps | 30 fps |
| 240p | 300-500 kbps | 30 fps |

## Next Steps

- [GPU Acceleration](/docs/guides/gpu-acceleration) - Speed up encoding
- [Distributed Processing](/docs/guides/distributed-processing) - Scale streaming workflows
- [Quality Metrics](/docs/guides/quality-metrics) - Ensure output quality
