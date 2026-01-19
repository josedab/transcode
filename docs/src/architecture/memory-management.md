# Memory Management

This document describes Transcode's memory management strategies.

## Overview

Video processing is memory-intensive. Transcode uses several strategies to manage memory efficiently:

- **Frame pooling** - Reuse allocated buffers
- **Zero-copy parsing** - Borrow instead of clone
- **Arena allocation** - Batch allocations for related data
- **Memory mapping** - Efficient file access

## Frame Memory Layout

Video frames use planar layout for cache efficiency:

```
┌─────────────────────────────────────┐
│            Y Plane                   │
│         (width × height)             │
├─────────────────────────────────────┤
│      U Plane (width/2 × height/2)    │
├─────────────────────────────────────┤
│      V Plane (width/2 × height/2)    │
└─────────────────────────────────────┘
```

```rust
pub struct Frame {
    /// Raw pixel data
    data: Vec<u8>,

    /// Plane offsets within data
    planes: [usize; 4],

    /// Bytes per row for each plane
    strides: [usize; 4],

    /// Frame dimensions
    width: u32,
    height: u32,

    /// Pixel format
    format: PixelFormat,
}

impl Frame {
    /// Get plane data by index
    pub fn plane(&self, index: usize) -> &[u8] {
        let start = self.planes[index];
        let end = self.planes.get(index + 1).copied().unwrap_or(self.data.len());
        &self.data[start..end]
    }
}
```

## Frame Pool

Reuse frame buffers to avoid allocation overhead:

```rust
use transcode_core::FramePool;

// Create pool for 1080p YUV420 frames
let pool = FramePool::new(1920, 1080, PixelFormat::Yuv420p, 8);

// Acquire frame from pool
let mut frame = pool.acquire();

// Use frame...
decode_into(&mut frame)?;

// Frame automatically returned to pool when dropped
drop(frame);
```

### Pool Configuration

```rust
pub struct FramePoolConfig {
    /// Maximum frames to keep in pool
    pub capacity: usize,

    /// Pre-allocate frames on creation
    pub preallocate: bool,

    /// Clear frame data on release
    pub clear_on_release: bool,
}

let pool = FramePool::with_config(config);
```

## Zero-Copy Parsing

Use borrowed data where possible:

```rust
// Good: borrow packet data
pub struct Packet<'a> {
    data: &'a [u8],
    pts: i64,
    dts: i64,
}

// Parse NAL unit without copying
pub fn parse_nal_unit(data: &[u8]) -> Result<NalUnit<'_>> {
    Ok(NalUnit {
        nal_type: data[0] & 0x1F,
        data: &data[1..],  // Borrowed, not copied
    })
}
```

### When to Copy

Copy is necessary when:
- Data must outlive the source
- Parallel processing requires independent ownership
- Data needs modification

```rust
// Copy when needed for parallel processing
let packets: Vec<OwnedPacket> = raw_packets
    .iter()
    .map(|p| p.to_owned())
    .collect();

// Process in parallel
packets.par_iter().for_each(|p| process(p));
```

## Arena Allocation

Group related allocations for batch deallocation:

```rust
use bumpalo::Bump;

pub struct DecoderArena {
    arena: Bump,
}

impl DecoderArena {
    /// Allocate coefficients for a macroblock
    pub fn alloc_coeffs(&self) -> &mut [i16; 256] {
        self.arena.alloc([0i16; 256])
    }

    /// Reset arena for next frame
    pub fn reset(&mut self) {
        self.arena.reset();
    }
}
```

## Memory-Mapped Files

For large files, use memory mapping:

```rust
use memmap2::Mmap;

pub struct MappedInput {
    mmap: Mmap,
}

impl MappedInput {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { mmap })
    }

    pub fn data(&self) -> &[u8] {
        &self.mmap
    }
}
```

## Buffer Strategies

### Decoder Buffers

```rust
pub struct DecoderBuffers {
    /// Reference frame buffer
    ref_frames: Vec<Frame>,

    /// Decoded picture buffer (DPB)
    dpb: RingBuffer<Frame>,

    /// Temporary coefficient storage
    coeffs: Vec<i16>,
}
```

### Encoder Buffers

```rust
pub struct EncoderBuffers {
    /// Lookahead buffer for B-frame decisions
    lookahead: VecDeque<Frame>,

    /// Rate control statistics
    rc_stats: Vec<FrameStats>,

    /// Output packet buffer
    output: Vec<u8>,
}
```

## Memory Limits

Configure maximum memory usage:

```rust
pub struct MemoryConfig {
    /// Maximum total memory (bytes)
    pub max_memory: usize,

    /// Maximum frame pool size
    pub max_pool_frames: usize,

    /// Maximum decoder buffer size
    pub max_dpb_frames: usize,
}

let config = MemoryConfig {
    max_memory: 2 * 1024 * 1024 * 1024,  // 2 GB
    max_pool_frames: 16,
    max_dpb_frames: 16,
};

let transcoder = Transcoder::with_memory_config(config)?;
```

## Monitoring

Track memory usage:

```rust
pub struct MemoryStats {
    pub allocated: usize,
    pub pool_size: usize,
    pub pool_available: usize,
    pub peak_usage: usize,
}

let stats = transcoder.memory_stats();
println!("Allocated: {} MB", stats.allocated / 1_000_000);
println!("Peak: {} MB", stats.peak_usage / 1_000_000);
```

## Best Practices

1. **Reuse buffers** - Use pools for frames and packets
2. **Borrow over clone** - Use references when lifetime allows
3. **Pre-allocate** - Avoid allocation in hot paths
4. **Batch deallocate** - Use arenas for temporary data
5. **Memory map large files** - Avoid loading entire file into memory

## Debugging

### Memory Profiling

```bash
# Using heaptrack
heaptrack ./transcode input.mp4 output.mp4

# Using valgrind
valgrind --tool=massif ./transcode input.mp4 output.mp4
```

### Tracking Allocations

```rust
#[cfg(feature = "track-allocations")]
pub fn track_allocation(size: usize, location: &str) {
    ALLOCATIONS.lock().push(AllocationInfo {
        size,
        location: location.to_string(),
        timestamp: Instant::now(),
    });
}
```
