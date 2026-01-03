# transcode-ts

MPEG Transport Stream container support for the Transcode library.

This crate provides demuxing and muxing capabilities for MPEG-TS files, commonly used for broadcast television and streaming media.

## Features

- **188-byte packets**: Standard MPEG-TS packet handling with sync byte validation
- **PID filtering**: Filter and route packets by Program ID
- **PAT/PMT parsing**: Full Program Association Table and Program Map Table support
- **PES packetization**: Packetized Elementary Stream handling with PTS/DTS timestamps
- **PCR handling**: Program Clock Reference for stream synchronization
- **Continuity counter**: Packet sequence verification
- **Adaptation field**: Random access indicators, PCR, stuffing
- **Stream types**: Video (H.264, H.265, MPEG-2), Audio (AAC, AC-3, MP2)

## Key Types

| Type | Description |
|------|-------------|
| `TsDemuxer` | Reads MPEG-TS files and extracts elementary streams |
| `TsMuxer` | Creates MPEG-TS files from elementary stream data |
| `TsPacket` | 188-byte transport stream packet |
| `TsHeader` | 4-byte packet header (sync, PID, flags, CC) |
| `AdaptationField` | Optional field with PCR, random access indicators |
| `Pcr` | 42-bit Program Clock Reference (27 MHz) |
| `Pat` / `Pmt` | Program Association and Program Map Tables |
| `StreamConfig` | Stream configuration for muxing |
| `MuxerConfig` | Muxer settings (PSI intervals, PCR rate) |

## Usage

### Demuxing

```rust
use transcode_ts::TsDemuxer;
use std::fs::File;
use std::io::BufReader;

let file = File::open("input.ts").unwrap();
let reader = BufReader::new(file);
let mut demuxer = TsDemuxer::new(reader).unwrap();

// Print stream info
for i in 0..demuxer.num_streams() {
    let stream = demuxer.stream(i).unwrap();
    println!("Stream {}: {} (PID: {})", i, stream.codec_name(), stream.pid);
}

// Read packets
while let Ok(Some(packet)) = demuxer.read_packet() {
    println!("Packet: stream={}, size={}", packet.stream_index, packet.data().len());
}
```

### Muxing

```rust
use transcode_ts::{TsMuxer, MuxerConfig, StreamConfig};
use transcode_core::packet::Packet;
use transcode_core::timestamp::{Timestamp, TimeBase};
use std::fs::File;
use std::io::BufWriter;

let file = File::create("output.ts").unwrap();
let writer = BufWriter::new(file);

let config = MuxerConfig::default();
let mut muxer = TsMuxer::new(writer, config);

// Add streams
muxer.add_stream(StreamConfig::video(0x101));
muxer.add_stream(StreamConfig::aac(0x102));

// Write header
muxer.write_header().unwrap();

// Write packets
let data = vec![0u8; 1000];
let mut packet = Packet::new(data);
packet.stream_index = 0;
packet.pts = Timestamp::new(90000, TimeBase::MPEG);
muxer.write_packet(&packet).unwrap();

// Finalize
muxer.write_trailer().unwrap();
```

## Well-Known PIDs

| PID | Description |
|-----|-------------|
| `0x0000` | PAT (Program Association Table) |
| `0x0001` | CAT (Conditional Access Table) |
| `0x0011` | SDT (Service Description Table) |
| `0x0012` | EIT (Event Information Table) |
| `0x1FFF` | Null packets |

## License

See the repository root for license information.
