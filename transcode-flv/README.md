# transcode-flv

FLV (Flash Video) container support for the transcode library.

## Features

- **FLV Header Parsing**: Signature, version, and stream flags
- **Tag Parsing**: Audio, video, and script data tags
- **Audio Codecs**: AAC (with AudioSpecificConfig), MP3
- **Video Codecs**: H.264/AVC, H.265/HEVC (Enhanced FLV)
- **AMF0 Metadata**: onMetaData parsing and generation
- **Timestamp Handling**: Extended timestamps, wraparound detection
- **Seeking Support**: Basic seeking (without index)
- **Streaming Ready**: Commonly used for RTMP and HTTP streaming

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-flv = { path = "../transcode-flv" }
```

### Demuxing

```rust
use std::fs::File;
use std::io::BufReader;
use transcode_flv::FlvDemuxer;

let file = File::open("input.flv")?;
let reader = BufReader::new(file);
let mut demuxer = FlvDemuxer::new(reader)?;

// Print stream information
if let Some(video) = demuxer.video_info() {
    println!("Video: {} {}x{} @ {}fps",
             video.codec.name(), video.width, video.height, video.frame_rate);
}

if let Some(audio) = demuxer.audio_info() {
    println!("Audio: {} {} Hz, {} channels",
             audio.format.name(), audio.sample_rate, audio.channels);
}

// Read packets
while let Ok(Some(packet)) = demuxer.read_packet() {
    if packet.is_video() && !packet.is_sequence_header {
        println!("Video: {}ms, keyframe={}, size={}",
                 packet.timestamp_ms, packet.is_keyframe, packet.data.len());
    }
}
```

### Muxing

```rust
use std::fs::File;
use std::io::BufWriter;
use transcode_flv::{FlvMuxer, MuxerConfig, VideoStreamConfig, AudioStreamConfig};

let file = File::create("output.flv")?;
let writer = BufWriter::new(file);

let config = MuxerConfig::default();
let mut muxer = FlvMuxer::new(writer, config);

// Configure streams
muxer.set_video_config(VideoStreamConfig::avc(1920, 1080, 30.0));
muxer.set_audio_config(AudioStreamConfig::aac(48000, 2));

// Write header (includes metadata)
muxer.write_header()?;

// Write sequence headers (from codec configuration)
muxer.write_video_sequence_header(&avc_config)?;
muxer.write_audio_sequence_header(&aac_config)?;

// Write packets
muxer.write_video_packet(timestamp_ms, &data, is_keyframe, cts)?;
muxer.write_audio_packet(timestamp_ms, &data)?;

muxer.finalize()?;
```

### Configuration

```rust
use transcode_flv::{FlvConfig, VideoCodec, SoundFormat};

let config = FlvConfig::new()
    .with_video(VideoCodec::Avc)
    .with_audio(SoundFormat::Aac)
    .with_metadata(true);

// Or use presets
let config = FlvConfig::avc_aac();      // H.264 + AAC
let config = FlvConfig::hevc_aac();     // H.265 + AAC (Enhanced FLV)
```

## Codec Support

### Video Codecs

| CodecID | Codec | Notes |
|---------|-------|-------|
| 7 | H.264/AVC | Standard FLV |
| 12 | H.265/HEVC | Enhanced FLV |
| 2 | Sorenson H.263 | Legacy |
| 4-5 | VP6 | Legacy |

### Audio Codecs

| SoundFormat | Codec | Notes |
|-------------|-------|-------|
| 10 | AAC | With AudioSpecificConfig |
| 2 | MP3 | Standard |
| 14 | MP3 8kHz | Low sample rate |
| 0-1 | PCM | Raw audio |

## FLV File Structure

```
FLV File
├── Header (9 bytes)
│   ├── Signature: "FLV"
│   ├── Version: 1
│   ├── Flags: has audio, has video
│   └── Header size: 9
├── Previous Tag Size 0 (4 bytes, always 0)
└── Tags (repeating)
    ├── Tag Header (11 bytes)
    │   ├── Tag type (8=audio, 9=video, 18=script)
    │   ├── Data size (3 bytes)
    │   ├── Timestamp (3 bytes + 1 extended)
    │   └── Stream ID (3 bytes, always 0)
    ├── Tag Data
    └── Previous Tag Size (4 bytes)
```

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
