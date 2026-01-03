//! Aligned memory allocation utilities

#![allow(clippy::needless_range_loop)]

/// Calculate aligned size
pub fn align_up(size: usize, alignment: usize) -> usize {
    (size + alignment - 1) & !(alignment - 1)
}

/// Check if pointer is aligned
pub fn is_aligned(ptr: *const u8, alignment: usize) -> bool {
    (ptr as usize).is_multiple_of(alignment)
}

/// Alignment requirements for different use cases
#[derive(Debug, Clone, Copy)]
pub enum AlignmentRequirement {
    /// Standard alignment (8 bytes)
    Standard,
    /// SIMD alignment (16 or 32 bytes)
    Simd,
    /// Cache line alignment (64 bytes)
    CacheLine,
    /// Page alignment (4096 bytes)
    Page,
    /// Huge page alignment (2MB)
    HugePage,
    /// Custom alignment
    Custom(usize),
}

impl AlignmentRequirement {
    /// Get alignment in bytes
    pub fn bytes(&self) -> usize {
        match self {
            AlignmentRequirement::Standard => 8,
            AlignmentRequirement::Simd => 32,
            AlignmentRequirement::CacheLine => 64,
            AlignmentRequirement::Page => 4096,
            AlignmentRequirement::HugePage => 2 * 1024 * 1024,
            AlignmentRequirement::Custom(n) => *n,
        }
    }
}

/// Aligned slice wrapper
pub struct AlignedSlice<'a> {
    data: &'a [u8],
    alignment: usize,
}

impl<'a> AlignedSlice<'a> {
    /// Create from slice if properly aligned
    pub fn new(data: &'a [u8], alignment: usize) -> Option<Self> {
        if is_aligned(data.as_ptr(), alignment) {
            Some(Self { data, alignment })
        } else {
            None
        }
    }

    /// Get inner slice
    pub fn as_slice(&self) -> &[u8] {
        self.data
    }

    /// Get alignment
    pub fn alignment(&self) -> usize {
        self.alignment
    }
}

/// Ring buffer with aligned segments
pub struct AlignedRingBuffer {
    data: aligned_vec::AVec<u8>,
    read_pos: usize,
    write_pos: usize,
    capacity: usize,
}

impl AlignedRingBuffer {
    /// Create new aligned ring buffer
    pub fn new(capacity: usize, alignment: usize) -> Self {
        let mut data = aligned_vec::AVec::with_capacity(alignment, capacity);
        data.resize(capacity, 0);

        Self {
            data,
            read_pos: 0,
            write_pos: 0,
            capacity,
        }
    }

    /// Get available space for writing
    pub fn write_available(&self) -> usize {
        if self.write_pos >= self.read_pos {
            self.capacity - self.write_pos + self.read_pos - 1
        } else {
            self.read_pos - self.write_pos - 1
        }
    }

    /// Get available data for reading
    pub fn read_available(&self) -> usize {
        if self.write_pos >= self.read_pos {
            self.write_pos - self.read_pos
        } else {
            self.capacity - self.read_pos + self.write_pos
        }
    }

    /// Write data to buffer
    pub fn write(&mut self, data: &[u8]) -> usize {
        let available = self.write_available();
        let to_write = data.len().min(available);

        for i in 0..to_write {
            self.data[self.write_pos] = data[i];
            self.write_pos = (self.write_pos + 1) % self.capacity;
        }

        to_write
    }

    /// Read data from buffer
    pub fn read(&mut self, buf: &mut [u8]) -> usize {
        let available = self.read_available();
        let to_read = buf.len().min(available);

        for i in 0..to_read {
            buf[i] = self.data[self.read_pos];
            self.read_pos = (self.read_pos + 1) % self.capacity;
        }

        to_read
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.read_pos = 0;
        self.write_pos = 0;
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.read_pos == self.write_pos
    }

    /// Check if full
    pub fn is_full(&self) -> bool {
        (self.write_pos + 1) % self.capacity == self.read_pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(1, 8), 8);
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(9, 8), 16);
        assert_eq!(align_up(4095, 4096), 4096);
        assert_eq!(align_up(4096, 4096), 4096);
    }

    #[test]
    fn test_ring_buffer() {
        let mut ring = AlignedRingBuffer::new(16, 64);

        assert!(ring.is_empty());
        assert_eq!(ring.write(b"Hello"), 5);
        assert!(!ring.is_empty());

        let mut buf = [0u8; 10];
        assert_eq!(ring.read(&mut buf), 5);
        assert_eq!(&buf[..5], b"Hello");
        assert!(ring.is_empty());
    }
}
