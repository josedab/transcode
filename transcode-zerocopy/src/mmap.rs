//! Memory mapping utilities

use crate::Result;
use std::path::Path;

/// Memory map advice
#[derive(Debug, Clone, Copy)]
pub enum MmapAdvice {
    /// Normal access pattern
    Normal,
    /// Sequential access
    Sequential,
    /// Random access
    Random,
    /// Will need this soon
    WillNeed,
    /// Don't need this anymore
    DontNeed,
}

/// Memory mapping options
#[derive(Debug, Clone)]
pub struct MmapOptions {
    /// Populate page tables (prefault)
    pub populate: bool,
    /// Use huge pages
    pub huge_pages: bool,
    /// Lock pages in memory
    pub locked: bool,
    /// Access advice
    pub advice: MmapAdvice,
}

impl Default for MmapOptions {
    fn default() -> Self {
        Self {
            populate: false,
            huge_pages: false,
            locked: false,
            advice: MmapAdvice::Normal,
        }
    }
}

/// Advanced memory-mapped file
pub struct AdvancedMmap {
    inner: memmap2::Mmap,
    len: usize,
}

impl AdvancedMmap {
    /// Create with options
    pub fn open(path: &Path, options: MmapOptions) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let len = file.metadata()?.len() as usize;

        let mut builder = memmap2::MmapOptions::new();

        if options.populate {
            builder.populate();
        }

        let inner = unsafe { builder.map(&file)? };

        let mmap = Self { inner, len };

        // Apply advice
        mmap.advise(options.advice)?;

        Ok(mmap)
    }

    /// Apply memory advice
    pub fn advise(&self, advice: MmapAdvice) -> Result<()> {
        #[cfg(unix)]
        {
            use memmap2::Advice;
            let advice = match advice {
                MmapAdvice::Normal => Advice::Normal,
                MmapAdvice::Sequential => Advice::Sequential,
                MmapAdvice::Random => Advice::Random,
                MmapAdvice::WillNeed => Advice::WillNeed,
                // DontNeed uses Normal as fallback since it's not in all versions
                MmapAdvice::DontNeed => Advice::Normal,
            };
            self.inner.advise(advice)?;
        }
        Ok(())
    }

    /// Get data slice
    pub fn as_slice(&self) -> &[u8] {
        &self.inner
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Chunked memory-mapped reader for very large files
pub struct ChunkedMmapReader {
    file: std::fs::File,
    file_size: u64,
    chunk_size: usize,
    current_chunk: Option<memmap2::Mmap>,
    current_offset: u64,
}

impl ChunkedMmapReader {
    /// Open a file for chunked reading
    pub fn open(path: &Path, chunk_size: usize) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let file_size = file.metadata()?.len();

        Ok(Self {
            file,
            file_size,
            chunk_size,
            current_chunk: None,
            current_offset: 0,
        })
    }

    /// Get chunk at offset
    pub fn chunk_at(&mut self, offset: u64) -> Result<&[u8]> {
        // Check if current chunk covers the offset
        let chunk_start = (offset / self.chunk_size as u64) * self.chunk_size as u64;

        if self.current_offset != chunk_start || self.current_chunk.is_none() {
            // Map new chunk
            let remaining = self.file_size.saturating_sub(chunk_start);
            let len = (remaining as usize).min(self.chunk_size);

            let mmap = unsafe {
                memmap2::MmapOptions::new()
                    .offset(chunk_start)
                    .len(len)
                    .map(&self.file)?
            };

            self.current_chunk = Some(mmap);
            self.current_offset = chunk_start;
        }

        Ok(self.current_chunk.as_ref().unwrap().as_ref())
    }

    /// Get total file size
    pub fn file_size(&self) -> u64 {
        self.file_size
    }
}
