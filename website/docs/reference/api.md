---
sidebar_position: 2
title: API Reference
description: Rust API documentation overview
---

# API Reference

This page provides an overview of the Transcode Rust API. For complete API documentation, visit [docs.rs/transcode](https://docs.rs/transcode).

## Crate Structure

```
transcode                    # Main high-level API
├── transcode-core          # Core types and utilities
├── transcode-codecs        # Codec implementations
├── transcode-containers    # Container formats
├── transcode-pipeline      # Processing pipeline
├── transcode-av1           # AV1 codec
├── transcode-streaming     # HLS/DASH output
├── transcode-gpu           # GPU acceleration
├── transcode-ai            # AI enhancement
├── transcode-quality       # Quality metrics
├── transcode-distributed   # Distributed processing
├── transcode-intel         # Content intelligence
└── transcode-wasm          # WebAssembly support
```

## Quick Links

| Crate | docs.rs |
|-------|---------|
| `transcode` | [docs.rs/transcode](https://docs.rs/transcode) |
| `transcode-core` | [docs.rs/transcode-core](https://docs.rs/transcode-core) |
| `transcode-codecs` | [docs.rs/transcode-codecs](https://docs.rs/transcode-codecs) |
| `transcode-pipeline` | [docs.rs/transcode-pipeline](https://docs.rs/transcode-pipeline) |

## Main Entry Points

### Transcoder

The primary interface for transcoding operations.

```rust
use transcode::{Transcoder, TranscodeOptions};

let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.mp4")
    .video_bitrate(5_000_000)
    .video_codec(VideoCodec::H264);

let mut transcoder = Transcoder::new(options)?;
let result = transcoder.run()?;
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `new(options)` | Create a new transcoder |
| `run()` | Run transcoding (blocking) |
| `run_async()` | Run transcoding (async) |
| `add_video_filter(filter)` | Add a video filter |
| `add_audio_filter(filter)` | Add an audio filter |
| `on_progress(callback)` | Set progress callback |
| `cancel()` | Cancel transcoding |

### TranscodeOptions

Configuration for transcoding operations.

```rust
use transcode::{TranscodeOptions, VideoCodec, AudioCodec, Preset};

let options = TranscodeOptions::new()
    // Input/Output
    .input("input.mp4")
    .output("output.mp4")

    // Video settings
    .video_codec(VideoCodec::H264)
    .video_bitrate(5_000_000)
    .resolution(1920, 1080)
    .fps(30.0)
    .preset(Preset::Medium)
    .crf(23)

    // Audio settings
    .audio_codec(AudioCodec::Aac)
    .audio_bitrate(128_000)
    .sample_rate(48000)
    .channels(2)

    // Time range
    .start_time(Duration::from_secs(10))
    .duration(Duration::from_secs(60));
```

### MediaInfo

Retrieve information about media files.

```rust
use transcode::MediaInfo;

let info = MediaInfo::read("video.mp4")?;

println!("Duration: {:?}", info.duration());
println!("Format: {}", info.format());

for stream in info.video_streams() {
    println!("Video: {}x{} @ {} fps",
        stream.width(),
        stream.height(),
        stream.frame_rate());
}

for stream in info.audio_streams() {
    println!("Audio: {} Hz, {} channels",
        stream.sample_rate(),
        stream.channels());
}
```

## Core Types

### Frame

Represents a decoded video frame.

```rust
use transcode_core::Frame;

// Create frame
let frame = Frame::new(1920, 1080, PixelFormat::Yuv420p);

// Access frame data
let y_plane = frame.plane(0);
let u_plane = frame.plane(1);
let v_plane = frame.plane(2);

// Frame properties
println!("Size: {}x{}", frame.width(), frame.height());
println!("PTS: {:?}", frame.pts());
println!("Format: {:?}", frame.pixel_format());
```

### Packet

Represents encoded data.

```rust
use transcode_core::Packet;

// Packet properties
let data = packet.data();
let pts = packet.pts();
let dts = packet.dts();
let duration = packet.duration();
let is_keyframe = packet.is_keyframe();
```

### Sample / SampleBuffer

Audio sample data.

```rust
use transcode_core::{Sample, SampleBuffer, SampleFormat};

// Create sample buffer
let buffer = SampleBuffer::new(
    1024,                    // frames
    2,                       // channels
    SampleFormat::Float32,   // format
);

// Access samples
let samples: &[f32] = buffer.as_slice();
```

## Codecs

### VideoDecoder / VideoEncoder Traits

```rust
use transcode_codecs::{VideoDecoder, VideoEncoder};

// Decoder
pub trait VideoDecoder {
    fn decode(&mut self, packet: &Packet) -> Result<Option<Frame>>;
    fn flush(&mut self) -> Result<Vec<Frame>>;
}

// Encoder
pub trait VideoEncoder {
    fn encode(&mut self, frame: &Frame) -> Result<Vec<Packet>>;
    fn flush(&mut self) -> Result<Vec<Packet>>;
}
```

### H.264 Decoder

```rust
use transcode_codecs::h264::H264Decoder;

let mut decoder = H264Decoder::new()?;

// Decode packets
for packet in packets {
    if let Some(frame) = decoder.decode(&packet)? {
        // Process frame
    }
}

// Flush remaining frames
for frame in decoder.flush()? {
    // Process frame
}
```

### H.264 Encoder

```rust
use transcode_codecs::h264::{H264Encoder, H264EncoderConfig};

let config = H264EncoderConfig {
    width: 1920,
    height: 1080,
    bitrate: 5_000_000,
    fps: 30.0,
    preset: Preset::Medium,
    profile: Profile::High,
    ..Default::default()
};

let mut encoder = H264Encoder::new(config)?;

// Encode frames
for frame in frames {
    for packet in encoder.encode(&frame)? {
        // Write packet
    }
}

// Flush remaining packets
for packet in encoder.flush()? {
    // Write packet
}
```

### AV1 Codec

```rust
use transcode_av1::{Av1Encoder, Av1EncoderConfig, Av1Decoder};

// Encoder
let config = Av1EncoderConfig {
    width: 1920,
    height: 1080,
    bitrate: 4_000_000,
    speed: 6,  // 0-10, higher = faster
    ..Default::default()
};

let mut encoder = Av1Encoder::new(config)?;

// Decoder
let mut decoder = Av1Decoder::new()?;
```

## Containers

### MP4 Demuxer

```rust
use transcode_containers::mp4::Mp4Demuxer;

let mut demuxer = Mp4Demuxer::open("input.mp4")?;

// Get streams
for stream in demuxer.streams() {
    println!("Stream {}: {:?}", stream.index(), stream.codec_type());
}

// Read packets
while let Some(packet) = demuxer.read_packet()? {
    // Process packet
}
```

### MP4 Muxer

```rust
use transcode_containers::mp4::{Mp4Muxer, Mp4MuxerConfig};

let config = Mp4MuxerConfig {
    faststart: true,  // Move moov atom to beginning
    ..Default::default()
};

let mut muxer = Mp4Muxer::create("output.mp4", config)?;

// Add streams
let video_stream = muxer.add_video_stream(video_config)?;
let audio_stream = muxer.add_audio_stream(audio_config)?;

// Write header
muxer.write_header()?;

// Write packets
for packet in packets {
    muxer.write_packet(&packet)?;
}

// Finalize
muxer.write_trailer()?;
```

## Pipeline

### Building a Pipeline

```rust
use transcode_pipeline::{Pipeline, PipelineBuilder};

let pipeline = PipelineBuilder::new()
    .demuxer(Mp4Demuxer::open("input.mp4")?)
    .video_decoder(H264Decoder::new()?)
    .video_filter(ScaleFilter::new(1920, 1080))
    .video_encoder(H264Encoder::new(config)?)
    .audio_decoder(AacDecoder::new()?)
    .audio_encoder(AacEncoder::new(audio_config)?)
    .muxer(Mp4Muxer::create("output.mp4", mux_config)?)
    .build()?;

pipeline.run()?;
```

### Custom Filters

```rust
use transcode_pipeline::{VideoFilter, Frame};

struct BrightnessFilter {
    adjustment: f32,
}

impl VideoFilter for BrightnessFilter {
    fn process(&mut self, frame: &mut Frame) -> Result<()> {
        // Apply brightness adjustment to frame
        for pixel in frame.data_mut() {
            *pixel = (*pixel as f32 * (1.0 + self.adjustment))
                .clamp(0.0, 255.0) as u8;
        }
        Ok(())
    }
}
```

## Streaming

### HLS Output

```rust
use transcode_streaming::hls::{HlsMuxer, HlsConfig, HlsVariant};

let config = HlsConfig {
    segment_duration: Duration::from_secs(6),
    playlist_type: PlaylistType::Vod,
    variants: vec![
        HlsVariant {
            resolution: (1920, 1080),
            bitrate: 5_000_000,
            ..Default::default()
        },
        HlsVariant {
            resolution: (1280, 720),
            bitrate: 2_500_000,
            ..Default::default()
        },
    ],
    ..Default::default()
};

let mut muxer = HlsMuxer::create("output/hls", config)?;
```

### DASH Output

```rust
use transcode_streaming::dash::{DashMuxer, DashConfig};

let config = DashConfig {
    segment_duration: Duration::from_secs(4),
    ..Default::default()
};

let mut muxer = DashMuxer::create("output/dash", config)?;
```

## GPU Processing

```rust
use transcode_gpu::{GpuContext, GpuPipelineBuilder, ScaleMode};

// Initialize GPU
let ctx = GpuContext::new().await?;

// Create processing pipeline
let pipeline = GpuPipelineBuilder::new(&ctx)
    .yuv_to_rgb(ColorSpace::Bt709)
    .scale(1920, 1080, ScaleMode::Lanczos)
    .effects(Effects {
        brightness: 0.1,
        contrast: 1.1,
        saturation: 1.05,
        gamma: 1.0,
    })
    .rgb_to_yuv(ColorSpace::Bt709)
    .build()?;

// Process frames
for frame in frames {
    let processed = pipeline.process(&frame).await?;
}
```

## Quality Metrics

```rust
use transcode_quality::{QualityAssessment, Metric};

let qa = QualityAssessment::new()
    .metrics(&[Metric::Psnr, Metric::Ssim, Metric::Vmaf]);

let report = qa.assess_video("reference.mp4", "compressed.mp4")?;

println!("PSNR: {:.2} dB", report.psnr.average);
println!("SSIM: {:.4}", report.ssim.average);
println!("VMAF: {:.2}", report.vmaf.average);
```

## Distributed Processing

```rust
use transcode_distributed::{Coordinator, Worker, Job};

// Start coordinator
let coordinator = Coordinator::new(config).await?;
coordinator.run().await?;

// Submit job
let client = Client::connect("http://coordinator:8080").await?;
let job = Job::new("transcode")
    .input("s3://bucket/input.mp4")
    .output("s3://bucket/output.mp4")
    .params(params);

let job_id = client.submit(job).await?;

// Monitor progress
let status = client.get_status(&job_id).await?;
```

## Error Handling

```rust
use transcode_core::error::{Error, Result, CodecError};

fn decode_video(data: &[u8]) -> Result<Frame> {
    if data.is_empty() {
        return Err(CodecError::BitstreamCorruption {
            message: "Empty data".into(),
        }.into());
    }
    // ...
}

// Match on specific errors
match decode_video(data) {
    Ok(frame) => { /* success */ }
    Err(Error::Codec(CodecError::BitstreamCorruption { message })) => {
        eprintln!("Corrupted data: {}", message);
    }
    Err(e) => {
        eprintln!("Other error: {}", e);
    }
}
```

## Feature Flags

```toml
[dependencies]
transcode = { version = "1.0", features = ["full"] }
```

| Feature | Description |
|---------|-------------|
| `default` | Core functionality |
| `full` | All features |
| `gpu` | GPU acceleration |
| `ai` | AI enhancement |
| `quality` | Quality metrics |
| `distributed` | Distributed processing |
| `streaming` | HLS/DASH output |
| `av1` | AV1 codec |
| `simd` | SIMD optimizations |

## Next Steps

- [Codecs Matrix](/docs/reference/codecs-matrix) - Supported codecs
- [Configuration](/docs/reference/configuration) - Config options
- [docs.rs/transcode](https://docs.rs/transcode) - Full API docs
