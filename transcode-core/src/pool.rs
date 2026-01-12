//! Memory pool implementations for efficient buffer reuse.
//!
//! Provides frame and sample buffer pools to reduce allocation overhead
//! in high-throughput transcoding pipelines.

use crate::frame::{Frame, FrameBuffer, PixelFormat};
use crate::sample::{ChannelLayout, SampleBuffer, SampleFormat};
use crate::timestamp::TimeBase;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;

/// A pool of reusable frame buffers.
pub struct FramePool {
    /// Available buffers.
    available: VecDeque<FrameBuffer>,
    /// Frame dimensions.
    width: u32,
    height: u32,
    /// Pixel format.
    format: PixelFormat,
    /// Maximum number of frames to keep in the pool.
    max_size: usize,
    /// Total frames allocated (for statistics).
    total_allocated: usize,
}

impl FramePool {
    /// Create a new frame pool.
    pub fn new(width: u32, height: u32, format: PixelFormat, max_size: usize) -> Self {
        Self {
            available: VecDeque::with_capacity(max_size),
            width,
            height,
            format,
            max_size,
            total_allocated: 0,
        }
    }

    /// Acquire a frame buffer from the pool.
    ///
    /// Returns an existing buffer if available, otherwise allocates a new one.
    pub fn acquire(&mut self) -> FrameBuffer {
        if let Some(buffer) = self.available.pop_front() {
            buffer
        } else {
            self.total_allocated += 1;
            FrameBuffer::new(self.width, self.height, self.format)
        }
    }

    /// Acquire a frame from the pool.
    pub fn acquire_frame(&mut self, _time_base: TimeBase) -> Frame {
        let buffer = self.acquire();
        Frame::from_buffer(buffer)
    }

    /// Release a frame buffer back to the pool.
    pub fn release(&mut self, buffer: FrameBuffer) {
        if self.available.len() < self.max_size
            && buffer.width == self.width
            && buffer.height == self.height
            && buffer.format == self.format
        {
            self.available.push_back(buffer);
        }
        // Otherwise, the buffer is dropped
    }

    /// Get the number of available buffers.
    pub fn available(&self) -> usize {
        self.available.len()
    }

    /// Get the total number of allocated buffers.
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Clear all pooled buffers.
    pub fn clear(&mut self) {
        self.available.clear();
    }
}

/// A thread-safe frame pool.
pub struct SharedFramePool {
    inner: Arc<Mutex<FramePool>>,
}

impl SharedFramePool {
    /// Create a new shared frame pool.
    pub fn new(width: u32, height: u32, format: PixelFormat, max_size: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(FramePool::new(width, height, format, max_size))),
        }
    }

    /// Acquire a frame buffer from the pool.
    pub fn acquire(&self) -> FrameBuffer {
        self.inner.lock().acquire()
    }

    /// Acquire a frame from the pool.
    pub fn acquire_frame(&self, time_base: TimeBase) -> Frame {
        self.inner.lock().acquire_frame(time_base)
    }

    /// Release a frame buffer back to the pool.
    pub fn release(&self, buffer: FrameBuffer) {
        self.inner.lock().release(buffer);
    }

    /// Get the number of available buffers.
    pub fn available(&self) -> usize {
        self.inner.lock().available()
    }
}

impl Clone for SharedFramePool {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// A pool of reusable sample buffers.
pub struct SamplePool {
    /// Available buffers.
    available: VecDeque<SampleBuffer>,
    /// Number of samples per buffer.
    num_samples: usize,
    /// Sample format.
    format: SampleFormat,
    /// Channel layout.
    layout: ChannelLayout,
    /// Sample rate.
    sample_rate: u32,
    /// Maximum number of buffers to keep.
    max_size: usize,
    /// Total buffers allocated.
    total_allocated: usize,
}

impl SamplePool {
    /// Create a new sample pool.
    pub fn new(
        num_samples: usize,
        format: SampleFormat,
        layout: ChannelLayout,
        sample_rate: u32,
        max_size: usize,
    ) -> Self {
        Self {
            available: VecDeque::with_capacity(max_size),
            num_samples,
            format,
            layout,
            sample_rate,
            max_size,
            total_allocated: 0,
        }
    }

    /// Acquire a sample buffer from the pool.
    pub fn acquire(&mut self) -> SampleBuffer {
        if let Some(buffer) = self.available.pop_front() {
            buffer
        } else {
            self.total_allocated += 1;
            SampleBuffer::new(self.num_samples, self.format, self.layout, self.sample_rate)
        }
    }

    /// Release a sample buffer back to the pool.
    pub fn release(&mut self, buffer: SampleBuffer) {
        if self.available.len() < self.max_size
            && buffer.num_samples == self.num_samples
            && buffer.format == self.format
            && buffer.layout == self.layout
        {
            self.available.push_back(buffer);
        }
    }

    /// Get the number of available buffers.
    pub fn available(&self) -> usize {
        self.available.len()
    }

    /// Clear all pooled buffers.
    pub fn clear(&mut self) {
        self.available.clear();
    }
}

/// A thread-safe sample pool.
pub struct SharedSamplePool {
    inner: Arc<Mutex<SamplePool>>,
}

impl SharedSamplePool {
    /// Create a new shared sample pool.
    pub fn new(
        num_samples: usize,
        format: SampleFormat,
        layout: ChannelLayout,
        sample_rate: u32,
        max_size: usize,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(SamplePool::new(
                num_samples,
                format,
                layout,
                sample_rate,
                max_size,
            ))),
        }
    }

    /// Acquire a sample buffer from the pool.
    pub fn acquire(&self) -> SampleBuffer {
        self.inner.lock().acquire()
    }

    /// Release a sample buffer back to the pool.
    pub fn release(&self, buffer: SampleBuffer) {
        self.inner.lock().release(buffer);
    }

    /// Get the number of available buffers.
    pub fn available(&self) -> usize {
        self.inner.lock().available()
    }
}

impl Clone for SharedSamplePool {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_pool() {
        let mut pool = FramePool::new(1920, 1080, PixelFormat::Yuv420p, 4);

        // Acquire first buffer
        let buf1 = pool.acquire();
        assert_eq!(pool.total_allocated(), 1);
        assert_eq!(pool.available(), 0);

        // Release and reacquire
        pool.release(buf1);
        assert_eq!(pool.available(), 1);

        let _buf2 = pool.acquire();
        assert_eq!(pool.total_allocated(), 1); // Reused
        assert_eq!(pool.available(), 0);
    }

    #[test]
    fn test_frame_pool_max_size() {
        let mut pool = FramePool::new(1920, 1080, PixelFormat::Yuv420p, 2);

        // Acquire and release 3 buffers
        let buf1 = pool.acquire();
        let buf2 = pool.acquire();
        let buf3 = pool.acquire();

        pool.release(buf1);
        pool.release(buf2);
        pool.release(buf3); // This one should be dropped

        assert_eq!(pool.available(), 2); // Max size is 2
    }

    #[test]
    fn test_sample_pool() {
        let mut pool = SamplePool::new(
            1024,
            SampleFormat::F32,
            ChannelLayout::Stereo,
            48000,
            4,
        );

        let buf = pool.acquire();
        assert_eq!(pool.total_allocated, 1);

        pool.release(buf);
        assert_eq!(pool.available(), 1);
    }

    #[test]
    fn test_shared_frame_pool() {
        let pool = SharedFramePool::new(1920, 1080, PixelFormat::Yuv420p, 4);
        let pool2 = pool.clone();

        let buf = pool.acquire();
        assert_eq!(pool2.available(), 0);

        pool.release(buf);
        assert_eq!(pool2.available(), 1);
    }
}
