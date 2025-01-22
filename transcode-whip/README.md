# transcode-whip

WHIP/WHEP WebRTC-based live ingest protocol support for the transcode library.

## Overview

This crate implements the WebRTC-HTTP Ingest Protocol (WHIP) and WebRTC-HTTP Egress Protocol (WHEP) for live streaming with sub-second latency.

## Features

- **WHIP**: Ingest live streams from WebRTC clients (browsers, OBS, etc.)
- **WHEP**: Serve live streams to WebRTC viewers
- **SDP Negotiation**: Full SDP offer/answer handling
- **ICE Trickle**: Support for incremental ICE candidate exchange
- **Codec Support**: H.264, VP8, VP9, AV1, Opus
- **Session Management**: Track and manage active streams

## Quick Start

```rust
use transcode_whip::{WhipServerBuilder, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let server = WhipServerBuilder::new()
        .bind("0.0.0.0:8080")
        .max_sessions(100)
        .ice_server("stun:stun.l.google.com:19302")
        .build();

    // Subscribe to events
    let mut events = server.subscribe();
    tokio::spawn(async move {
        while let Ok(event) = events.recv().await {
            println!("Event: {:?}", event);
        }
    });

    server.run().await
}
```

## Endpoints

### WHIP (Ingest)

```bash
# Publish a stream
curl -X POST http://localhost:8080/whip \
  -H "Content-Type: application/sdp" \
  -d @offer.sdp

# Response: 201 Created with SDP answer
# Location: /whip/{session-id}
```

### WHEP (Playback)

```bash
# Subscribe to a stream
curl -X POST http://localhost:8080/whep \
  -H "Content-Type: application/sdp" \
  -d @offer.sdp

# Response: 201 Created with SDP answer
# Location: /whep/{session-id}
```

### ICE Trickle

```bash
# Add ICE candidate
curl -X PATCH http://localhost:8080/whip/{session-id} \
  -H "Content-Type: application/trickle-ice-sdpfrag" \
  -H "If-Match: {etag}" \
  -d "a=candidate:..."
```

### Session Management

```bash
# List sessions
curl http://localhost:8080/sessions

# Get session details
curl http://localhost:8080/sessions/{session-id}

# End session
curl -X DELETE http://localhost:8080/whip/{session-id}
```

## Configuration

```rust
use transcode_whip::ServerConfig;

let config = ServerConfig {
    bind_address: "0.0.0.0:8080".to_string(),
    max_sessions: 1000,
    whip_path: "/whip".to_string(),
    whep_path: "/whep".to_string(),
    ice_servers: vec![/* STUN/TURN servers */],
    enable_cors: true,
    session_timeout_secs: 30,
};
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
