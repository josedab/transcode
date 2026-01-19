# ADR-0005: Async Pipeline Design

## Status

Accepted

## Date

2024-03

## Context

A transcoding pipeline has multiple stages:
1. **Demuxing**: Reading from input file/stream
2. **Decoding**: CPU-intensive decompression
3. **Filtering**: Processing (scale, crop, color convert)
4. **Encoding**: CPU-intensive compression
5. **Muxing**: Writing to output file/stream

These stages have different characteristics:
- Demuxing/Muxing: I/O bound
- Decoding/Encoding: CPU bound
- Filtering: May be GPU bound

We want to:
1. Maximize throughput via parallelism
2. Handle backpressure gracefully
3. Support streaming (live) inputs
4. Enable distributed processing

## Decision

Use an **async pipeline architecture** with the following components:

### 1. Tokio Runtime

Use Tokio as the async runtime for I/O operations.

```rust
#[tokio::main]
async fn main() -> Result<()> {
    let pipeline = Pipeline::new(config).await?;
    pipeline.run().await?;
    Ok(())
}
```

### 2. Channel-Based Stage Communication

Stages communicate via bounded async channels:

```rust
pub struct Pipeline {
    demux_tx: mpsc::Sender<Packet>,
    decode_rx: mpsc::Receiver<Packet>,
    decode_tx: mpsc::Sender<Frame>,
    // ...
}
```

### 3. Thread Pool for CPU Work

CPU-intensive work (decode/encode) uses a dedicated thread pool:

```rust
let frame = tokio::task::spawn_blocking(move || {
    decoder.decode(&packet)
}).await??;
```

### 4. Backpressure via Bounded Channels

```rust
// If encoder is slow, decoder blocks when channel is full
let (tx, rx) = mpsc::channel(BUFFER_SIZE);
```

### 5. Graceful Shutdown

```rust
impl Pipeline {
    pub async fn stop_graceful(&self, timeout: Duration) -> usize {
        // 1. Stop accepting new input
        self.running.store(false, Ordering::SeqCst);

        // 2. Wait for in-flight work
        let deadline = Instant::now() + timeout;
        while self.has_pending_work() && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // 3. Cancel remainder
        self.cancel_pending().await
    }
}
```

### Pipeline Topology

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Demux   │───▶│  Decode  │───▶│  Filter  │───▶│  Encode  │───▶│   Mux    │
│  (I/O)   │    │  (CPU)   │    │(CPU/GPU) │    │  (CPU)   │    │  (I/O)   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
     │               │               │               │               │
     │               ▼               ▼               ▼               │
     │          Thread Pool     Thread Pool     Thread Pool          │
     │                                                               │
     └─────────────────── Tokio Runtime ────────────────────────────┘
```

## Consequences

### Positive

1. **High throughput**: Stages run in parallel

2. **Backpressure**: Slow stages naturally throttle fast ones

3. **Streaming support**: Works with unbounded inputs

4. **Resource control**: Bounded buffers limit memory usage

5. **Graceful shutdown**: Clean handling of interrupts

6. **Distributed-ready**: Channel abstraction works across network

### Negative

1. **Complexity**: Async code is harder to debug

2. **Overhead**: Channel operations have cost

3. **Ordering**: Must maintain frame order across async boundaries

4. **Dependencies**: Requires Tokio runtime

### Mitigations

1. **Tracing**: Use `tracing` crate for async-aware logging

2. **Sequence numbers**: Frames carry sequence IDs for reordering

3. **Sync API**: Provide blocking wrapper for simple use cases
   ```rust
   // Async
   pipeline.run().await?;

   // Sync wrapper
   pipeline.run_blocking()?;
   ```

## Buffer Sizing

| Stage | Buffer Size | Rationale |
|-------|-------------|-----------|
| Demux → Decode | 32 packets | ~1 second of video |
| Decode → Filter | 8 frames | Memory limit (~100MB) |
| Filter → Encode | 8 frames | Match decode buffer |
| Encode → Mux | 32 packets | Match demux buffer |

## Alternatives Considered

### Alternative 1: Synchronous Pipeline

Simple loop: read → decode → filter → encode → write.

Rejected because:
- No parallelism
- Can't handle live streams
- Poor resource utilization

### Alternative 2: Actor Model (Actix)

Each stage as an actor with message passing.

Rejected because:
- More complex than needed
- Actix adds significant dependencies
- Harder to reason about ordering

### Alternative 3: Rayon for Parallelism

Use Rayon's parallel iterators.

Rejected because:
- Designed for batch processing, not streaming
- No backpressure mechanism
- Doesn't handle async I/O

## References

- [Tokio documentation](https://tokio.rs/)
- [Backpressure explained](https://mechanical-sympathy.blogspot.com/2012/05/apply-back-pressure-when-overloaded.html)
- [GStreamer pipeline model](https://gstreamer.freedesktop.org/documentation/application-development/introduction/basics.html)
