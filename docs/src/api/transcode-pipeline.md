# transcode-pipeline

Pipeline orchestration for transcoding workflows.

## Overview

`transcode-pipeline` provides high-level APIs for building transcoding pipelines:

- Declarative pipeline construction
- Automatic codec negotiation
- Filter chain support
- Progress monitoring
- Async processing

## Crate Features

```toml
[dependencies]
transcode-pipeline = { version = "1.0", features = ["async"] }
```

| Feature | Description |
|---------|-------------|
| `async` | Async/await support |
| `parallel` | Multi-threaded processing |
| `filters` | Built-in filter chain |

## Basic Usage

### Simple Transcoding

```rust
use transcode_pipeline::{Pipeline, PipelineConfig};

let pipeline = Pipeline::builder()
    .input("input.mp4")
    .output("output.mp4")
    .video_codec(Codec::H264)
    .video_bitrate(5_000_000)
    .audio_codec(Codec::Aac)
    .audio_bitrate(128_000)
    .build()?;

let stats = pipeline.run()?;

println!("Encoded {} frames in {:?}", stats.frames, stats.duration);
```

### With Progress Callback

```rust
let pipeline = Pipeline::builder()
    .input("input.mp4")
    .output("output.mp4")
    .on_progress(|progress| {
        println!("Progress: {:.1}%", progress * 100.0);
    })
    .build()?;

pipeline.run()?;
```

## Pipeline Builder

### Video Options

```rust
let pipeline = Pipeline::builder()
    .input("input.mp4")
    .output("output.mp4")
    // Codec selection
    .video_codec(Codec::Hevc)
    // Quality settings
    .crf(23)                          // Quality-based
    .video_bitrate(5_000_000)         // Bitrate-based
    // Encoding speed
    .preset(Preset::Medium)
    // Resolution
    .resolution(1920, 1080)
    // Framerate
    .framerate(30.0)
    .build()?;
```

### Audio Options

```rust
let pipeline = Pipeline::builder()
    .input("input.mp4")
    .output("output.mp4")
    .audio_codec(Codec::Opus)
    .audio_bitrate(128_000)
    .audio_channels(2)
    .sample_rate(48000)
    .build()?;
```

### Multiple Outputs

```rust
let pipeline = Pipeline::builder()
    .input("input.mp4")
    .output("output_1080p.mp4")
        .resolution(1920, 1080)
        .video_bitrate(5_000_000)
    .output("output_720p.mp4")
        .resolution(1280, 720)
        .video_bitrate(2_500_000)
    .output("output_480p.mp4")
        .resolution(854, 480)
        .video_bitrate(1_000_000)
    .build()?;
```

## Filter Chain

### Built-in Filters

```rust
use transcode_pipeline::filters::*;

let pipeline = Pipeline::builder()
    .input("input.mp4")
    .video_filter(Scale::new(1920, 1080))
    .video_filter(Deinterlace::new())
    .video_filter(Denoise::new(DenoiseStrength::Medium))
    .video_filter(Crop::new(100, 100, 1720, 880))
    .output("output.mp4")
    .build()?;
```

### Filter Chain Builder

```rust
use transcode_pipeline::FilterChain;

let chain = FilterChain::new()
    .scale(1920, 1080)
    .fps(30.0)
    .colorspace(ColorSpace::Bt709);

let pipeline = Pipeline::builder()
    .input("input.mp4")
    .video_filters(chain)
    .output("output.mp4")
    .build()?;
```

### Custom Filters

```rust
use transcode_pipeline::Filter;

struct GrayscaleFilter;

impl Filter for GrayscaleFilter {
    fn process(&mut self, frame: Frame) -> Result<Frame> {
        let mut output = frame.clone();
        // Set U and V planes to 128 (neutral)
        output.plane_mut(1).fill(128);
        output.plane_mut(2).fill(128);
        Ok(output)
    }
}

let pipeline = Pipeline::builder()
    .input("input.mp4")
    .video_filter(GrayscaleFilter)
    .output("output.mp4")
    .build()?;
```

## Async Pipeline

```rust
use transcode_pipeline::AsyncPipeline;

#[tokio::main]
async fn main() -> Result<()> {
    let pipeline = AsyncPipeline::builder()
        .input("input.mp4")
        .output("output.mp4")
        .build()?;

    // Run with event stream
    let mut events = pipeline.run_stream();

    while let Some(event) = events.next().await {
        match event {
            PipelineEvent::Progress(p) => {
                println!("Progress: {:.1}%", p * 100.0);
            }
            PipelineEvent::FrameEncoded(n) => {
                println!("Encoded frame {}", n);
            }
            PipelineEvent::Complete(stats) => {
                println!("Done: {:?}", stats);
                break;
            }
            PipelineEvent::Error(e) => {
                return Err(e);
            }
        }
    }

    Ok(())
}
```

## Configuration

### PipelineConfig

```rust
use transcode_pipeline::PipelineConfig;

let config = PipelineConfig {
    // Threading
    decode_threads: 4,
    encode_threads: 4,

    // Buffering
    frame_buffer_size: 16,

    // Performance
    hw_accel: HwAccel::Auto,

    // Error handling
    error_handling: ErrorHandling::Stop,
};

let pipeline = Pipeline::builder()
    .config(config)
    .input("input.mp4")
    .output("output.mp4")
    .build()?;
```

## Statistics

### TranscodeStats

```rust
let stats = pipeline.run()?;

println!("Input duration: {:?}", stats.input_duration);
println!("Output duration: {:?}", stats.output_duration);
println!("Frames encoded: {}", stats.frames_encoded);
println!("Encoding time: {:?}", stats.encoding_time);
println!("Speed: {:.2}x realtime", stats.speed());
println!("Input size: {} bytes", stats.input_size);
println!("Output size: {} bytes", stats.output_size);
println!("Compression: {:.2}x", stats.compression_ratio());
```

## Error Handling

```rust
use transcode_pipeline::{Pipeline, PipelineError};

match pipeline.run() {
    Ok(stats) => println!("Success: {} frames", stats.frames_encoded),
    Err(PipelineError::InputNotFound(path)) => {
        eprintln!("Input file not found: {}", path);
    }
    Err(PipelineError::CodecNotSupported(codec)) => {
        eprintln!("Codec not supported: {:?}", codec);
    }
    Err(PipelineError::EncodingFailed(e)) => {
        eprintln!("Encoding failed: {}", e);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Cancellation

```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

let cancel = Arc::new(AtomicBool::new(false));
let cancel_clone = cancel.clone();

// In another thread
std::thread::spawn(move || {
    std::thread::sleep(Duration::from_secs(10));
    cancel_clone.store(true, Ordering::SeqCst);
});

let pipeline = Pipeline::builder()
    .input("input.mp4")
    .output("output.mp4")
    .cancel_token(cancel)
    .build()?;

match pipeline.run() {
    Err(PipelineError::Cancelled) => println!("Cancelled"),
    result => handle_result(result),
}
```

## Full API Reference

See the [rustdoc documentation](https://docs.rs/transcode-pipeline) for complete API details.
