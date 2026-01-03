# transcode-live

Live streaming protocol support for the transcode project. This crate provides input stream handling for RTMP, SRT, and WebRTC protocols.

## Features

- **RTMP Protocol**: Full handshake support, chunk streaming, and AMF message handling
- **SRT Protocol**: Secure Reliable Transport with encryption, configurable latency, and statistics
- **WebRTC Protocol**: ICE candidates, SDP offer/answer, STUN/TURN server support
- **Unified API**: Common `LiveServer` and `LiveClient` abstractions across all protocols
- **Async/Await**: Built on Tokio for non-blocking I/O

## Key Types

### Core Types

| Type | Description |
|------|-------------|
| `LiveServer` | Server that accepts incoming streams |
| `LiveClient` | Client for connecting to external servers |
| `LiveConfig` | Server configuration (protocol, address, timeout, max clients) |
| `StreamProtocol` | Protocol enum: `Rtmp`, `Srt`, `WebRtc` |
| `StreamPacket` | Packet with type, timestamp, data, and keyframe flag |
| `StreamMetadata` | Stream info (codecs, resolution, bitrate, sample rate) |
| `StreamState` | Connection state: `Waiting`, `Connecting`, `Streaming`, `Disconnected`, `Error` |

### Protocol-Specific Types

- **RTMP**: `RtmpSession`, `RtmpChunkStream`, `RtmpHandshake`, `RtmpMessageType`
- **SRT**: `SrtSocket`, `SrtConfig`, `SrtStats`, `SrtMode`
- **WebRTC**: `WebRtcPeer`, `WebRtcConfig`, `IceCandidate`, `SessionDescription`, `TurnServer`

## Usage

### Starting a Live Server

```rust
use transcode_live::{LiveServer, LiveConfig, StreamProtocol};
use std::net::SocketAddr;

#[tokio::main]
async fn main() -> transcode_live::Result<()> {
    let config = LiveConfig {
        protocol: StreamProtocol::Rtmp,
        listen_addr: "0.0.0.0:1935".parse().unwrap(),
        max_clients: 100,
        ..Default::default()
    };

    let mut server = LiveServer::new(config);
    server.start().await?;
    Ok(())
}
```

### Connecting as a Client

```rust
use transcode_live::LiveClient;

#[tokio::main]
async fn main() -> transcode_live::Result<()> {
    let mut client = LiveClient::new("rtmp://localhost/live/stream")?;
    let metadata = client.connect().await?;

    while let Some(packet) = client.read_packet().await? {
        println!("Received {:?} packet, {} bytes", packet.packet_type, packet.data.len());
    }

    client.disconnect().await?;
    Ok(())
}
```

### SRT with Encryption

```rust
use transcode_live::{SrtSocket, SrtConfig, SrtMode};
use std::time::Duration;

let config = SrtConfig {
    mode: SrtMode::Caller,
    passphrase: Some("secret".into()),
    latency: Duration::from_millis(120),
    ..Default::default()
};

let mut socket = SrtSocket::new(config);
socket.connect("srt://server:9000").await?;
```

### WebRTC Peer Connection

```rust
use transcode_live::{WebRtcPeer, WebRtcConfig};

let mut peer = WebRtcPeer::new(WebRtcConfig::default());
let offer = peer.create_offer().await?;
peer.set_local_description(offer).await?;
```

## Error Handling

All operations return `Result<T, LiveError>`. Error variants include:

- `InvalidUrl` - Malformed URL
- `Connection` - Connection failure
- `NotConnected` - Operation requires active connection
- `Timeout` - Operation timed out
- `Protocol` - Protocol-level error
- `Io` - Underlying I/O error

## License

See the workspace root for license information.
