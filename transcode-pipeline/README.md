# transcode-pipeline

Transcoding pipeline orchestration for the Transcode codec library. Provides a high-level API for building transcoding pipelines that connect demuxers, decoders, encoders, and muxers.

## Pipeline Architecture

```
┌──────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐    ┌───────┐
│ Demuxer  │───▶│ Decoder  │───▶│ Filters │───▶│ Encoder  │───▶│ Muxer │
└──────────┘    └──────────┘    └─────────┘    └──────────┘    └───────┘
                                     │
                              ┌──────┴──────┐
                              │Synchronizer │
                              └─────────────┘
```

The pipeline processes media in a pull-based model:
1. **DemuxerNode** reads packets from container formats
2. **DecoderNode** decodes packets into raw frames/samples
3. **FilterChain** applies video/audio filters
4. **EncoderNode** encodes frames/samples into packets
5. **MuxerNode** writes packets to output container
6. **Synchronizer** manages A/V timing alignment

## Key Components

| Component | Description |
|-----------|-------------|
| `Pipeline` | Main orchestrator connecting all nodes |
| `PipelineBuilder` | Fluent API for constructing pipelines |
| `DemuxerNode` | Wraps container demuxers (MP4, etc.) |
| `DecoderNode` | Wraps video/audio decoders |
| `EncoderNode` | Wraps video/audio encoders |
| `MuxerNode` | Wraps container muxers |
| `FilterChain` | Chains video or audio filters |
| `Synchronizer` | A/V sync with multiple modes |
| `ReorderBuffer` | Handles out-of-order packets |

### Filters

- `ScaleFilter` - Video resizing (nearest-neighbor)
- `VolumeFilter` - Audio gain adjustment (dB)
- `NullVideoFilter` / `NullAudioFilter` - Pass-through

### Sync Modes

- `AudioMaster` - Video syncs to audio (default)
- `VideoMaster` - Audio syncs to video
- `ExternalClock` - Sync to external time source
- `None` - No synchronization

## Usage

### Basic Pipeline

```rust
use transcode_pipeline::{Pipeline, PipelineBuilder, PipelineConfig};

let pipeline = PipelineBuilder::new()
    .config(PipelineConfig::default())
    .demuxer(demuxer_node)
    .decoder(0, video_decoder_node)
    .decoder(1, audio_decoder_node)
    .encoder(0, video_encoder_node)
    .encoder(1, audio_encoder_node)
    .muxer(muxer_node)
    .build()?;

pipeline.run()?;
```

### With Filters

```rust
use transcode_pipeline::{PipelineBuilder, ScaleFilter, VolumeFilter};

let pipeline = PipelineBuilder::new()
    .demuxer(demuxer_node)
    .decoder(0, video_decoder)
    .video_filter(Box::new(ScaleFilter::new(1280, 720)))
    .audio_filter(Box::new(VolumeFilter::new(3.0))) // +3dB
    .encoder(0, video_encoder)
    .muxer(muxer_node)
    .build()?;
```

### Step-by-Step Processing

```rust
let mut pipeline = Pipeline::new(PipelineConfig::default());
pipeline.set_demuxer(demuxer);
pipeline.set_muxer(muxer);
pipeline.add_decoder(0, decoder);
pipeline.add_encoder(0, encoder);
pipeline.initialize()?;

while pipeline.step()? {
    println!("Processed {} packets", pipeline.packets_processed());
}
```

### Custom Sync Configuration

```rust
use transcode_pipeline::{SyncConfig, SyncMode};

let config = PipelineConfig {
    sync: SyncConfig {
        mode: SyncMode::VideoMaster,
        max_audio_drift_us: 50_000,
        max_video_drift_us: 100_000,
        ..Default::default()
    },
    max_buffer_size: 64,
    report_progress: true,
    progress_interval: 100,
};
```

## Documentation

See the main [Transcode documentation](../README.md) for complete API reference and additional examples.
