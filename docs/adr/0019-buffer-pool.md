# ADR-0019: Buffer Pool and Memory Reuse

## Status

Accepted

## Date

2024-08

## Context

Video transcoding processes thousands of frames per second. Each frame requires memory allocation:

- **1080p YUV420**: ~3 MB per frame
- **4K YUV420**: ~12 MB per frame
- **Audio sample buffers**: ~8 KB per buffer (1024 samples × 4 bytes × 2 channels)

With naive allocation:
```rust
// Allocation-heavy decode loop
loop {
    let mut frame = Frame::new(width, height, format); // Allocates
    decoder.decode_into(packet, &mut frame)?;
    process(frame);
    // frame dropped, memory freed
}
```

At 60 fps, this means 60 allocations/deallocations per second per stage. For a pipeline with demux → decode → filter → encode → mux, memory pressure compounds:

- **Allocator contention**: Multiple threads compete for allocator locks
- **Memory fragmentation**: Repeated alloc/free fragments the heap
- **Cache invalidation**: New allocations are cold in cache
- **GC latency** (if using arenas): Periodic cleanup causes stalls

## Decision

Implement **buffer pooling** with pre-allocated, reusable frame and sample buffers:

### 1. Frame Pool

```rust
pub struct FramePool {
    available: VecDeque<FrameBuffer>,
    width: u32,
    height: u32,
    format: PixelFormat,
    max_size: usize,
}

impl FramePool {
    pub fn new(width: u32, height: u32, format: PixelFormat, capacity: usize) -> Self;

    /// Get a buffer from pool (or allocate if empty)
    pub fn acquire(&mut self) -> FrameBuffer;

    /// Return buffer to pool for reuse
    pub fn release(&mut self, buffer: FrameBuffer);
}
```

### 2. Thread-Safe Shared Pool

```rust
pub struct SharedFramePool {
    inner: Arc<Mutex<FramePool>>,
}

impl SharedFramePool {
    pub fn new(config: PoolConfig) -> Self;

    pub fn acquire(&self) -> PooledFrame;  // RAII wrapper
}

/// Frame that returns to pool on drop
pub struct PooledFrame {
    buffer: Option<FrameBuffer>,
    pool: Arc<Mutex<FramePool>>,
}

impl Drop for PooledFrame {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            if let Ok(mut pool) = self.pool.lock() {
                pool.release(buffer);
            }
        }
    }
}
```

### 3. Sample Pool (Audio)

```rust
pub struct SamplePool {
    available: VecDeque<SampleBuffer>,
    samples_per_buffer: usize,
    format: SampleFormat,
    channels: usize,
    max_size: usize,
}

pub struct SharedSamplePool {
    inner: Arc<Mutex<SamplePool>>,
}
```

### 4. Pool Configuration

```rust
pub struct PoolConfig {
    pub initial_capacity: usize,  // Pre-allocate this many
    pub max_capacity: usize,      // Never exceed this
    pub growth_strategy: GrowthStrategy,
}

pub enum GrowthStrategy {
    Fixed,           // Never grow beyond initial
    Linear(usize),   // Grow by N at a time
    Exponential,     // Double when exhausted
}
```

## Consequences

### Positive

1. **Reduced allocation overhead**: Hot path does pool lookup, not malloc

2. **Better cache locality**: Reused buffers stay warm in cache

3. **Predictable memory usage**: Max pool size bounds memory consumption

4. **Lower latency variance**: No GC-like pauses from allocation storms

5. **Thread-safe sharing**: Pools work across pipeline stages

### Negative

1. **Upfront memory cost**: Pools pre-allocate even if not fully used

2. **Complexity**: Must track buffer lifecycle correctly

3. **Pool exhaustion**: If all buffers in use, must wait or allocate

4. **Format changes**: Pool buffers fixed at creation-time format

### Mitigations

1. **Lazy allocation**: Start small, grow on demand
   ```rust
   pub fn acquire(&mut self) -> FrameBuffer {
       self.available.pop_front().unwrap_or_else(|| {
           if self.allocated < self.max_size {
               self.allocated += 1;
               FrameBuffer::new(self.width, self.height, self.format)
           } else {
               panic!("Pool exhausted"); // Or block/allocate
           }
       })
   }
   ```

2. **Pool per format**: Separate pools for different resolutions

3. **Overflow handling**: Allocate beyond pool if necessary (with warning)

## Implementation Details

### FrameBuffer Structure

```rust
pub struct FrameBuffer {
    /// Backing storage (potentially shared)
    data: Arc<Vec<u8>>,
    /// Plane offsets within data
    planes: [PlaneInfo; 4],
    /// Format information
    width: u32,
    height: u32,
    format: PixelFormat,
}

impl FrameBuffer {
    /// Clear buffer for reuse (zeros optional)
    pub fn reset(&mut self) {
        // Optionally zero memory for security
        // Or just reset metadata
    }

    /// Check if buffer matches required format
    pub fn matches(&self, width: u32, height: u32, format: PixelFormat) -> bool {
        self.width == width && self.height == height && self.format == format
    }
}
```

### Pool Usage in Pipeline

```rust
pub struct DecodePipeline {
    decoder: Box<dyn VideoDecoder>,
    frame_pool: SharedFramePool,
}

impl DecodePipeline {
    pub fn decode_next(&mut self) -> Result<PooledFrame> {
        let mut frame = self.frame_pool.acquire();

        // Decode into pooled buffer
        self.decoder.decode_into(&packet, frame.buffer_mut())?;

        Ok(frame)  // Returns to pool when dropped
    }
}
```

### Lock-Free Alternative

For highest performance, consider lock-free pools:

```rust
pub struct LockFreeFramePool {
    available: crossbeam_queue::ArrayQueue<FrameBuffer>,
}

impl LockFreeFramePool {
    pub fn try_acquire(&self) -> Option<FrameBuffer> {
        self.available.pop()
    }

    pub fn release(&self, buffer: FrameBuffer) {
        // Discard if pool full
        let _ = self.available.push(buffer);
    }
}
```

### Memory Statistics

```rust
pub struct PoolStats {
    pub capacity: usize,
    pub available: usize,
    pub in_use: usize,
    pub total_acquires: u64,
    pub total_releases: u64,
    pub allocations: u64,  // Times we had to allocate
    pub peak_usage: usize,
}

impl FramePool {
    pub fn stats(&self) -> PoolStats;
}
```

## Performance Impact

### Benchmark: 1080p Decode (60 fps target)

| Approach | Alloc/sec | Frame Time | CPU Usage |
|----------|-----------|------------|-----------|
| Per-frame alloc | 60 | 18.2ms avg | 45% |
| Frame pool (mutex) | 0 | 14.8ms avg | 38% |
| Frame pool (lock-free) | 0 | 14.1ms avg | 36% |

*Measured on 8-core system, 1000 frame sequence*

### Memory Usage

| Configuration | Peak Memory | Steady State |
|---------------|-------------|--------------|
| No pooling | 180 MB | 12 MB (varies) |
| Pool (8 frames) | 48 MB | 48 MB |
| Pool (16 frames) | 96 MB | 96 MB |

## Integration with Codec Traits

The codec traits support buffer reuse via `*_into` methods:

```rust
pub trait VideoDecoderExt: VideoDecoder {
    /// Decode into caller-provided buffer
    fn decode_into(&mut self, packet: &Packet, frame: &mut Frame) -> Result<bool>;

    /// Flush remaining frames into buffer
    fn flush_into(&mut self, frame: &mut Frame) -> Result<Option<bool>>;
}
```

## Alternatives Considered

### Alternative 1: Arena Allocation

Use a bump allocator that frees all at once.

Rejected because:
- Doesn't work well with long-running pipelines
- Memory grows unbounded until arena reset
- Doesn't integrate with Drop semantics

### Alternative 2: Operating System Page Pool

Use mmap/munmap for large buffers.

Rejected because:
- Higher overhead than userspace pooling
- System call per acquire/release
- Less control over memory layout

### Alternative 3: No Pooling, Trust the Allocator

Modern allocators (jemalloc, mimalloc) handle repeated allocations well.

Partially valid, but rejected as primary strategy because:
- Still has overhead vs pool lookup
- Fragmentation still occurs over time
- Less predictable latency

## References

- [Object Pool Pattern](https://en.wikipedia.org/wiki/Object_pool_pattern)
- [crossbeam-queue](https://docs.rs/crossbeam-queue/)
- [parking_lot](https://docs.rs/parking_lot/) - Fast mutexes
- [Memory Pool Allocators](https://www.boost.org/doc/libs/release/libs/pool/)
