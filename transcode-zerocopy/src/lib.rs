//! Zero-copy I/O optimizations for transcode
//!
//! This crate provides memory-mapped I/O and io_uring support for efficient data transfer.

use std::fs::File;
use std::path::Path;

mod error;
mod mmap;
mod aligned;

pub use error::*;
pub use mmap::*;
pub use aligned::*;

#[cfg(all(target_os = "linux", feature = "io-uring"))]
mod uring;
#[cfg(all(target_os = "linux", feature = "io-uring"))]
pub use uring::*;

/// Result type for zero-copy operations
pub type Result<T> = std::result::Result<T, ZeroCopyError>;

/// Zero-copy configuration
#[derive(Debug, Clone)]
pub struct ZeroCopyConfig {
    /// Use memory-mapped I/O
    pub use_mmap: bool,
    /// Use io_uring (Linux only)
    pub use_io_uring: bool,
    /// Alignment for buffers
    pub alignment: usize,
    /// Prefetch size
    pub prefetch_size: usize,
    /// Read-ahead pages
    pub read_ahead_pages: usize,
}

impl Default for ZeroCopyConfig {
    fn default() -> Self {
        Self {
            use_mmap: true,
            use_io_uring: cfg!(all(target_os = "linux", feature = "io-uring")),
            alignment: 4096,
            prefetch_size: 1024 * 1024, // 1MB
            read_ahead_pages: 16,
        }
    }
}

/// Memory-mapped file reader
pub struct MappedReader {
    mmap: memmap2::Mmap,
    position: usize,
}

impl MappedReader {
    /// Open a file for memory-mapped reading
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        Ok(Self {
            mmap,
            position: 0,
        })
    }

    /// Get the entire file as a byte slice
    pub fn as_slice(&self) -> &[u8] {
        &self.mmap
    }

    /// Get remaining data from current position
    pub fn remaining(&self) -> &[u8] {
        &self.mmap[self.position..]
    }

    /// Get file size
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// Get current position
    pub fn position(&self) -> usize {
        self.position
    }

    /// Seek to position
    pub fn seek(&mut self, position: usize) -> Result<()> {
        if position > self.mmap.len() {
            return Err(ZeroCopyError::OutOfBounds);
        }
        self.position = position;
        Ok(())
    }

    /// Read bytes at current position
    pub fn read(&mut self, len: usize) -> Result<&[u8]> {
        let end = self.position.saturating_add(len).min(self.mmap.len());
        let data = &self.mmap[self.position..end];
        self.position = end;
        Ok(data)
    }

    /// Advise kernel about access pattern
    pub fn advise_sequential(&self) -> Result<()> {
        // madvise could be used here for further optimization
        Ok(())
    }
}

/// Memory-mapped file writer
pub struct MappedWriter {
    mmap: memmap2::MmapMut,
    position: usize,
    capacity: usize,
}

impl MappedWriter {
    /// Create a new memory-mapped file for writing
    pub fn create(path: &Path, size: usize) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        file.set_len(size as u64)?;

        let mmap = unsafe { memmap2::MmapMut::map_mut(&file)? };

        Ok(Self {
            mmap,
            position: 0,
            capacity: size,
        })
    }

    /// Write data at current position
    pub fn write(&mut self, data: &[u8]) -> Result<usize> {
        let available = self.capacity.saturating_sub(self.position);
        let to_write = data.len().min(available);

        self.mmap[self.position..self.position + to_write]
            .copy_from_slice(&data[..to_write]);

        self.position += to_write;
        Ok(to_write)
    }

    /// Get current position
    pub fn position(&self) -> usize {
        self.position
    }

    /// Seek to position
    pub fn seek(&mut self, position: usize) -> Result<()> {
        if position > self.capacity {
            return Err(ZeroCopyError::OutOfBounds);
        }
        self.position = position;
        Ok(())
    }

    /// Flush changes to disk
    pub fn flush(&self) -> Result<()> {
        self.mmap.flush()?;
        Ok(())
    }

    /// Async flush
    pub fn flush_async(&self) -> Result<()> {
        self.mmap.flush_async()?;
        Ok(())
    }
}

/// Aligned buffer for DMA operations
pub struct AlignedBuffer {
    data: aligned_vec::AVec<u8>,
}

impl AlignedBuffer {
    /// Create a new aligned buffer
    pub fn new(size: usize, alignment: usize) -> Self {
        let data = aligned_vec::AVec::with_capacity(alignment, size);
        Self { data }
    }

    /// Create from existing data
    pub fn from_slice(data: &[u8], alignment: usize) -> Self {
        let mut buffer = Self::new(data.len(), alignment);
        buffer.data.extend_from_slice(data);
        buffer
    }

    /// Get as slice
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get alignment
    pub fn alignment(&self) -> usize {
        self.data.alignment()
    }

    /// Resize buffer
    pub fn resize(&mut self, new_size: usize, value: u8) {
        self.data.resize(new_size, value);
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

/// Buffer pool for zero-copy operations
pub struct BufferPool {
    buffers: Vec<AlignedBuffer>,
    _buffer_size: usize,
    _alignment: usize,
}

impl BufferPool {
    /// Create a new buffer pool
    pub fn new(count: usize, buffer_size: usize, alignment: usize) -> Self {
        let buffers = (0..count)
            .map(|_| AlignedBuffer::new(buffer_size, alignment))
            .collect();

        Self {
            buffers,
            _buffer_size: buffer_size,
            _alignment: alignment,
        }
    }

    /// Get a buffer from the pool
    pub fn get(&mut self) -> Option<AlignedBuffer> {
        self.buffers.pop()
    }

    /// Return a buffer to the pool
    pub fn put(&mut self, mut buffer: AlignedBuffer) {
        buffer.clear();
        self.buffers.push(buffer);
    }

    /// Get pool size
    pub fn available(&self) -> usize {
        self.buffers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_aligned_buffer() {
        let buffer = AlignedBuffer::new(1024, 64); // Cache-line alignment
        assert!(buffer.alignment() >= 64);
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_buffer_pool() {
        let mut pool = BufferPool::new(4, 1024, 64); // Cache-line alignment
        assert_eq!(pool.available(), 4);

        let buf = pool.get().unwrap();
        assert_eq!(pool.available(), 3);

        pool.put(buf);
        assert_eq!(pool.available(), 4);
    }

    #[test]
    fn test_mapped_reader() {
        // Create temp file
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.bin");

        let mut file = File::create(&path).unwrap();
        file.write_all(b"Hello, World!").unwrap();
        drop(file);

        // Read with mmap
        let reader = MappedReader::open(&path).unwrap();
        assert_eq!(reader.as_slice(), b"Hello, World!");
    }

    #[test]
    fn test_mapped_writer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("output.bin");

        {
            let mut writer = MappedWriter::create(&path, 1024).unwrap();
            writer.write(b"Test data").unwrap();
            writer.flush().unwrap();
        }

        // Verify
        let data = std::fs::read(&path).unwrap();
        assert!(data.starts_with(b"Test data"));
    }
}
