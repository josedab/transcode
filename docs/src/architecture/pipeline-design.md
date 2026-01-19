# Pipeline Design

This document describes the transcoding pipeline architecture.

## Overview

The pipeline orchestrates data flow between components:

```
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ Demuxer │──▶│ Decoder │──▶│ Filters │──▶│ Encoder │──▶│  Muxer  │
└─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
```

## Pipeline Stages

### 1. Demuxer

Reads container format and extracts packets:

```rust
pub trait Demuxer {
    /// Read next packet from container
    fn read_packet(&mut self) -> Result<Option<Packet>>;

    /// Get stream information
    fn streams(&self) -> &[StreamInfo];

    /// Seek to timestamp
    fn seek(&mut self, timestamp: i64, stream_index: usize) -> Result<()>;
}
```

### 2. Decoder

Converts compressed packets to raw frames:

```rust
pub trait Decoder {
    type Output;

    fn decode(&mut self, packet: &Packet) -> Result<Option<Self::Output>>;
    fn flush(&mut self) -> Result<Vec<Self::Output>>;
}
```

### 3. Filter Chain

Processes raw frames:

```rust
pub trait Filter {
    fn process(&mut self, frame: Frame) -> Result<Frame>;
}

// Chain multiple filters
let chain = FilterChain::new()
    .add(ScaleFilter::new(1920, 1080))
    .add(DeinterlaceFilter::new())
    .add(ColorConvertFilter::new(PixelFormat::Yuv420p));
```

### 4. Encoder

Compresses raw frames to packets:

```rust
pub trait Encoder {
    type Input;

    fn encode(&mut self, input: &Self::Input) -> Result<Vec<Packet>>;
    fn flush(&mut self) -> Result<Vec<Packet>>;
}
```

### 5. Muxer

Writes packets to container format:

```rust
pub trait Muxer {
    fn write_header(&mut self) -> Result<()>;
    fn write_packet(&mut self, packet: &Packet) -> Result<()>;
    fn write_trailer(&mut self) -> Result<()>;
}
```

## Pipeline Builder

```rust
use transcode_pipeline::{Pipeline, PipelineBuilder};

let pipeline = PipelineBuilder::new()
    .input("input.mp4")
    .video_decoder(VideoCodec::H264)
    .video_filter(ScaleFilter::new(1280, 720))
    .video_encoder(VideoCodec::Hevc)
    .audio_decoder(AudioCodec::Aac)
    .audio_encoder(AudioCodec::Opus)
    .output("output.mkv")
    .build()?;

pipeline.run()?;
```

## Async Pipeline

For non-blocking operation:

```rust
use transcode_pipeline::AsyncPipeline;

let pipeline = AsyncPipeline::builder()
    .input("input.mp4")
    .output("output.mp4")
    .build()?;

// Run with progress updates
let mut stream = pipeline.run_stream();

while let Some(event) = stream.next().await {
    match event {
        PipelineEvent::Progress(p) => println!("Progress: {:.1}%", p * 100.0),
        PipelineEvent::Frame(f) => println!("Processed frame {}", f),
        PipelineEvent::Complete(stats) => break,
        PipelineEvent::Error(e) => return Err(e),
    }
}
```

## Parallel Processing

### Frame-Level Parallelism

```
Frame 0 ──▶ [Decode] ──▶ [Filter] ──▶ [Encode] ──▶
Frame 1 ──▶ [Decode] ──▶ [Filter] ──▶ [Encode] ──▶
Frame 2 ──▶ [Decode] ──▶ [Filter] ──▶ [Encode] ──▶
```

### Segment-Level Parallelism

For distributed processing:

```rust
use transcode_pipeline::SegmentedPipeline;

let pipeline = SegmentedPipeline::builder()
    .input("input.mp4")
    .segment_duration(Duration::from_secs(10))
    .parallel_segments(4)
    .build()?;
```

## Buffer Management

### Ring Buffer

Frames flow through a ring buffer between stages:

```rust
pub struct RingBuffer<T> {
    buffer: Vec<Option<T>>,
    read_pos: usize,
    write_pos: usize,
    capacity: usize,
}

impl<T> RingBuffer<T> {
    pub fn push(&mut self, item: T) -> Result<()>;
    pub fn pop(&mut self) -> Option<T>;
    pub fn is_full(&self) -> bool;
    pub fn is_empty(&self) -> bool;
}
```

### Frame Pool

Reuse frame buffers to reduce allocations:

```rust
pub struct FramePool {
    pool: Vec<Frame>,
    in_use: HashSet<usize>,
}

impl FramePool {
    pub fn acquire(&mut self) -> PooledFrame;
    pub fn release(&mut self, frame: PooledFrame);
}
```

## Synchronization

### A/V Sync

Audio and video streams are synchronized by PTS:

```rust
pub struct AvSync {
    video_pts: i64,
    audio_pts: i64,
    timebase: Rational,
}

impl AvSync {
    pub fn sync(&mut self, packet: &Packet) -> SyncAction {
        match packet.stream_type {
            StreamType::Video => {
                if self.video_pts < self.audio_pts {
                    SyncAction::Process
                } else {
                    SyncAction::Wait
                }
            }
            StreamType::Audio => {
                // Similar logic for audio
            }
        }
    }
}
```

## Error Recovery

### Packet Loss Recovery

```rust
impl Pipeline {
    fn handle_decode_error(&mut self, error: CodecError) -> Result<()> {
        match error {
            CodecError::BitstreamCorruption { .. } => {
                // Skip to next keyframe
                self.decoder.reset();
                self.seek_to_next_keyframe()?;
            }
            _ => return Err(error.into()),
        }
        Ok(())
    }
}
```

## Metrics and Monitoring

```rust
pub struct PipelineMetrics {
    pub frames_decoded: u64,
    pub frames_encoded: u64,
    pub frames_dropped: u64,
    pub decode_time: Duration,
    pub encode_time: Duration,
    pub filter_time: Duration,
}

// Access during processing
let metrics = pipeline.metrics();
println!("FPS: {:.1}", metrics.current_fps());
```

## Configuration

```rust
pub struct PipelineConfig {
    /// Number of decode threads
    pub decode_threads: usize,

    /// Number of encode threads
    pub encode_threads: usize,

    /// Frame buffer size between stages
    pub buffer_size: usize,

    /// Enable frame dropping under load
    pub allow_frame_drop: bool,

    /// Target latency for live streaming
    pub target_latency: Option<Duration>,
}
```
