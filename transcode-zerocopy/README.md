# transcode-zerocopy

Zero-copy I/O optimizations for the transcode media processing library.

## Overview

This crate provides memory-mapped I/O and aligned buffer utilities for efficient data transfer in video/audio transcoding pipelines. It minimizes memory copies by mapping files directly into the process address space and using properly aligned buffers for DMA operations.

## Features

- **Memory-mapped I/O**: Direct file access via `mmap` without intermediate copying
- **Aligned buffers**: Cache-line and page-aligned allocations for SIMD and DMA
- **Buffer pooling**: Reusable buffer pools to reduce allocation overhead
- **Chunked reading**: Stream large files without mapping the entire file
- **io_uring support**: Linux async I/O (optional feature)

## Key Types

| Type | Description |
|------|-------------|
| `MappedReader` | Memory-mapped file reader with seek support |
| `MappedWriter` | Memory-mapped file writer with sync/async flush |
| `AlignedBuffer` | Buffer with configurable alignment (SIMD, cache-line, page) |
| `BufferPool` | Pool of pre-allocated aligned buffers |
| `AdvancedMmap` | Memory mapping with madvise hints |
| `ChunkedMmapReader` | Chunked reader for very large files |
| `AlignedRingBuffer` | Ring buffer with aligned memory |

## Usage

### Memory-Mapped Reading

```rust
use transcode_zerocopy::MappedReader;
use std::path::Path;

let reader = MappedReader::open(Path::new("input.mp4"))?;

// Access entire file as a slice (zero-copy)
let data = reader.as_slice();

// Or read incrementally
let mut reader = MappedReader::open(Path::new("input.mp4"))?;
let chunk = reader.read(4096)?;
```

### Memory-Mapped Writing

```rust
use transcode_zerocopy::MappedWriter;
use std::path::Path;

let mut writer = MappedWriter::create(Path::new("output.bin"), 1024 * 1024)?;
writer.write(b"encoded data")?;
writer.flush()?; // or flush_async() for non-blocking
```

### Aligned Buffers

```rust
use transcode_zerocopy::AlignedBuffer;

// Cache-line aligned (64 bytes) for optimal CPU access
let buffer = AlignedBuffer::new(4096, 64);

// Page aligned (4096 bytes) for DMA operations
let dma_buffer = AlignedBuffer::from_slice(&data, 4096);
```

### Buffer Pool

```rust
use transcode_zerocopy::BufferPool;

// Create pool of 8 buffers, 64KB each, page-aligned
let mut pool = BufferPool::new(8, 65536, 4096);

let buf = pool.get().expect("buffer available");
// ... use buffer ...
pool.put(buf); // return to pool
```

### Chunked Reading for Large Files

```rust
use transcode_zerocopy::ChunkedMmapReader;
use std::path::Path;

// Map 64MB chunks at a time
let mut reader = ChunkedMmapReader::open(Path::new("large.mp4"), 64 * 1024 * 1024)?;
let chunk = reader.chunk_at(0)?;
```

## Configuration

```rust
use transcode_zerocopy::ZeroCopyConfig;

let config = ZeroCopyConfig {
    use_mmap: true,
    use_io_uring: false, // Linux only, requires feature
    alignment: 4096,
    prefetch_size: 1024 * 1024, // 1MB
    read_ahead_pages: 16,
};
```

## Optional Features

- `io-uring` - Enable Linux io_uring support for async I/O

```toml
[dependencies]
transcode-zerocopy = { version = "0.1", features = ["io-uring"] }
```

## License

See the workspace root for license information.
