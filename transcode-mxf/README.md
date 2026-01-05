# transcode-mxf

MXF (Material eXchange Format) container support for the transcode library.

## Overview

MXF is the standard file format for professional video exchange, particularly in broadcast, post-production, and archival workflows. It is defined by SMPTE standards and used extensively in the television and film industry.

## Features

- **MXF Parsing**: Full demuxing support for MXF files
- **MXF Writing**: Muxing support for creating MXF files
- **Op1a Support**: Single-item, single-package operational pattern
- **KLV Handling**: Key-Length-Value triplet parsing and writing
- **Universal Labels**: SMPTE Universal Label (UL) support
- **Metadata Extraction**: Content package and essence descriptor parsing
- **Partition Support**: Header, body, and footer partition handling

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-mxf = { path = "../transcode-mxf" }
```

### Demuxing

```rust
use transcode_mxf::{MxfDemuxer, MxfPacket};

let data = std::fs::read("video.mxf")?;
let mut demuxer = MxfDemuxer::new(&data)?;

// Get track information
println!("Tracks: {}", demuxer.track_count());

for track in demuxer.tracks() {
    println!("Track {}: {:?}", track.track_id, track.essence_type);
}

// Read packets
while let Some(packet) = demuxer.read_packet()? {
    println!("Track {}: {} bytes at edit unit {}",
             packet.track_id, packet.data.len(), packet.edit_unit);
}
```

### Muxing

```rust
use transcode_mxf::{MxfMuxer, MuxerConfig, TrackConfig, EditRate};

let config = MuxerConfig::new()
    .with_edit_rate(EditRate::from_fps(25))
    .with_video_track(TrackConfig::video(1920, 1080));

let mut muxer = MxfMuxer::new(config)?;

// Write header partition
muxer.write_header()?;

// Write essence data
muxer.write_frame(0, &frame_data)?;

// Write footer and finalize
muxer.finalize()?;
```

### Metadata Access

```rust
use transcode_mxf::MxfDemuxer;

let mut demuxer = MxfDemuxer::new(&data)?;

// Access content package metadata
if let Some(content) = demuxer.content_package() {
    println!("Package UID: {:?}", content.package_uid);
}

// Access essence descriptors
for descriptor in demuxer.essence_descriptors() {
    println!("Essence: {:?}", descriptor);
}
```

## MXF Structure

| Element | Description |
|---------|-------------|
| Header Partition | File header, metadata |
| Body Partitions | Essence data |
| Footer Partition | Index tables, RIP |
| KLV Triplets | Key-Length-Value encoded data |
| Universal Labels | 16-byte SMPTE identifiers |

## Operational Patterns

| Pattern | Description |
|---------|-------------|
| Op1a | Single item, single package |
| Op1b | Single item, ganged packages |
| Op2a | Playlist items |
| Op3a | Edit items |
| OpAtom | Single track, atom file |

## Common Use Cases

- **Broadcast Delivery**: Standard format for TV content exchange
- **Post-Production**: Inter-application media transfer
- **Archival**: Long-term preservation with rich metadata
- **IMF**: Interoperable Master Format packages

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
