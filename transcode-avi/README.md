# transcode-avi

AVI (Audio Video Interleave) container format support for the transcode library.

## Features

- **RIFF Chunk Parsing**: Full RIFF (Resource Interchange File Format) structure support
- **Demuxing**: Read and parse AVI files
- **Muxing**: Create AVI files with multiple streams
- **Multiple Streams**: Support for multiple audio and video tracks
- **OpenDML Extensions**: AVI 2.0 support for large files (>2GB)
- **FourCC Handling**: Standard FourCC codec identification

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-avi = { path = "../transcode-avi" }
```

### Demuxing

```rust
use transcode_avi::{AviDemuxer, AviPacket};

let data = std::fs::read("video.avi")?;
let mut demuxer = AviDemuxer::new(&data)?;

// Read header information
let header = demuxer.header();
println!("Duration: {} frames", header.total_frames);
println!("Resolution: {}x{}", header.width, header.height);

// Read packets
while let Some(packet) = demuxer.read_packet()? {
    println!("Stream {}: {} bytes", packet.stream_index, packet.data.len());
}
```

### Muxing

```rust
use transcode_avi::{AviMuxer, MuxerConfig, StreamConfig, VideoFormat};

let config = MuxerConfig::new()
    .with_video(StreamConfig::video(
        VideoFormat::h264(),
        1920, 1080,
        30, 1,  // framerate
    ));

let mut muxer = AviMuxer::new(config)?;
muxer.write_header()?;

// Write video frames
muxer.write_frame(0, &frame_data, is_keyframe)?;

muxer.finalize()?;
```

### Stream Information

```rust
use transcode_avi::AviDemuxer;

let mut demuxer = AviDemuxer::new(&data)?;

for stream in demuxer.streams() {
    match &stream.format {
        StreamFormat::Video(fmt) => {
            println!("Video: {} {}x{}", fmt.fourcc, fmt.width, fmt.height);
        }
        StreamFormat::Audio(fmt) => {
            println!("Audio: {} Hz, {} channels", fmt.sample_rate, fmt.channels);
        }
    }
}
```

## AVI Structure

| Chunk | Description |
|-------|-------------|
| RIFF 'AVI ' | Main container |
| LIST 'hdrl' | Header list |
| avih | Main AVI header |
| LIST 'strl' | Stream list (per stream) |
| strh | Stream header |
| strf | Stream format |
| LIST 'movi' | Movie data |
| idx1 | Legacy index |
| indx | OpenDML index |

## Limitations

- AVI is a legacy format with limited codec support
- No native support for modern codecs (VP9, AV1)
- 2GB file size limit without OpenDML extensions
- No built-in streaming support

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
