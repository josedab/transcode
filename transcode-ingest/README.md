# transcode-ingest

Live media ingest protocol support for the transcode library.

## Overview

This crate provides handling for live media ingest protocols including RTMP and SRT, enabling live streaming and contribution workflows.

## Features

- **RTMP Server/Client**: Full handshake support, AMF metadata
- **SRT Protocol**: Reliable UDP transport with encryption
- **Protocol Abstraction**: Common `IngestSource` trait for uniform handling
- **Async Support**: Tokio-based asynchronous implementation
- **Media Types**: Audio, video, and metadata handling

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-ingest = { path = "../transcode-ingest" }
```

### RTMP Server

```rust
use transcode_ingest::{RtmpServer, IngestSource, IngestServer};
use std::net::SocketAddr;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr: SocketAddr = "0.0.0.0:1935".parse()?;
    let mut server = RtmpServer::bind(addr).await?;

    println!("RTMP server listening on {}", server.local_addr()?);

    while let Ok(conn) = server.accept().await {
        tokio::spawn(async move {
            handle_connection(conn).await;
        });
    }
    Ok(())
}

async fn handle_connection(mut conn: impl IngestSource) {
    println!("New connection from {}", conn.peer_addr());

    while let Ok(Some(media)) = conn.read_media().await {
        match media {
            MediaData::Video { timestamp, is_keyframe, .. } => {
                println!("Video: {}ms, keyframe={}", timestamp, is_keyframe);
            }
            MediaData::Audio { timestamp, .. } => {
                println!("Audio: {}ms", timestamp);
            }
            _ => {}
        }
    }
}
```

### SRT Server

```rust
use transcode_ingest::{SrtServer, IngestServer};
use std::net::SocketAddr;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr: SocketAddr = "0.0.0.0:9000".parse()?;
    let mut server = SrtServer::bind(addr).await?;

    while let Ok(conn) = server.accept().await {
        println!("SRT connection: {}", conn.stream_info().name);
    }
    Ok(())
}
```

### Client Connection

```rust
use transcode_ingest::{RtmpClient, IngestClient, IngestConfig};
use std::net::SocketAddr;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr: SocketAddr = "server:1935".parse()?;
    let config = IngestConfig::default();

    let conn = RtmpClient::connect(addr, config).await?;
    println!("Connected to {}", conn.peer_addr());

    Ok(())
}
```

## Configuration

```rust
use transcode_ingest::IngestConfig;
use std::time::Duration;

let config = IngestConfig {
    connect_timeout: Duration::from_secs(10),
    read_timeout: Duration::from_secs(30),
    write_timeout: Duration::from_secs(30),
    max_chunk_size: 4096,
    buffer_size: 65536,
};
```

## Media Data Types

| Type | Fields |
|------|--------|
| Audio | data, timestamp, codec, sample_rate, channels |
| Video | data, timestamp, codec, is_keyframe, composition_time |
| Metadata | properties (key-value pairs) |

## Supported Codecs

### Audio
- AAC, MP3, PCM, Speex, Opus

### Video
- H.264, H.265, VP8, VP9, AV1

## Connection States

| State | Description |
|-------|-------------|
| Disconnected | Not connected |
| Handshaking | Handshake in progress |
| Connected | Ready for operations |
| Publishing | Sending media data |
| Playing | Receiving media data |
| Closing | Connection closing |

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
