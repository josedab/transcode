//! Cache management for conformance test streams
//!
//! Manages local caching of downloaded test streams to avoid repeated downloads.

use crate::{ConformanceError, Result, TestStream};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Cache manager for test streams
pub struct StreamCache {
    cache_dir: PathBuf,
}

impl StreamCache {
    /// Create a new cache manager
    pub fn new(cache_dir: PathBuf) -> Result<Self> {
        if !cache_dir.exists() {
            fs::create_dir_all(&cache_dir)?;
            info!("Created cache directory: {:?}", cache_dir);
        }
        Ok(Self { cache_dir })
    }

    /// Get the cache directory
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Check if a stream is cached
    pub fn is_cached(&self, stream: &TestStream) -> bool {
        self.get_cached_path(stream)
            .map(|p| p.exists())
            .unwrap_or(false)
    }

    /// Get the cached path for a stream
    pub fn get_cached_path(&self, stream: &TestStream) -> Option<PathBuf> {
        let path = self.cache_dir.join(&stream.id).with_extension("264");
        if path.exists() {
            Some(path)
        } else {
            None
        }
    }

    /// Get the expected cache path for a stream (may not exist yet)
    pub fn expected_path(&self, stream: &TestStream) -> PathBuf {
        self.cache_dir.join(&stream.id).with_extension("264")
    }

    /// Store data in cache
    pub fn store(&self, stream: &TestStream, data: &[u8]) -> Result<PathBuf> {
        let path = self.expected_path(stream);

        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent)?;
            }
        }

        fs::write(&path, data)?;
        debug!("Cached stream {} to {:?}", stream.id, path);

        // Also store metadata
        self.store_metadata(stream)?;

        Ok(path)
    }

    /// Store stream metadata
    fn store_metadata(&self, stream: &TestStream) -> Result<()> {
        let meta_path = self.cache_dir.join(&stream.id).with_extension("json");
        let json = serde_json::to_string_pretty(stream).map_err(|e| {
            ConformanceError::Configuration {
                message: format!("Failed to serialize metadata: {}", e),
            }
        })?;
        fs::write(&meta_path, json)?;
        Ok(())
    }

    /// Load stream metadata
    pub fn load_metadata(&self, stream_id: &str) -> Result<Option<TestStream>> {
        let meta_path = self.cache_dir.join(stream_id).with_extension("json");
        if !meta_path.exists() {
            return Ok(None);
        }

        let json = fs::read_to_string(&meta_path)?;
        let stream: TestStream = serde_json::from_str(&json).map_err(|e| {
            ConformanceError::Configuration {
                message: format!("Failed to deserialize metadata: {}", e),
            }
        })?;
        Ok(Some(stream))
    }

    /// Clear cache for a specific stream
    pub fn clear(&self, stream: &TestStream) -> Result<()> {
        let stream_path = self.expected_path(stream);
        let meta_path = self.cache_dir.join(&stream.id).with_extension("json");

        if stream_path.exists() {
            fs::remove_file(&stream_path)?;
            debug!("Removed cached stream: {:?}", stream_path);
        }

        if meta_path.exists() {
            fs::remove_file(&meta_path)?;
        }

        Ok(())
    }

    /// Clear entire cache
    pub fn clear_all(&self) -> Result<()> {
        if self.cache_dir.exists() {
            fs::remove_dir_all(&self.cache_dir)?;
            fs::create_dir_all(&self.cache_dir)?;
            info!("Cleared cache directory: {:?}", self.cache_dir);
        }
        Ok(())
    }

    /// Get cache size in bytes
    pub fn size(&self) -> Result<u64> {
        let mut total = 0u64;
        if self.cache_dir.exists() {
            for entry in fs::read_dir(&self.cache_dir)? {
                let entry = entry?;
                let metadata = entry.metadata()?;
                if metadata.is_file() {
                    total += metadata.len();
                }
            }
        }
        Ok(total)
    }

    /// List cached streams
    pub fn list_cached(&self) -> Result<Vec<String>> {
        let mut streams = Vec::new();
        if self.cache_dir.exists() {
            for entry in fs::read_dir(&self.cache_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().map(|e| e == "264").unwrap_or(false) {
                    if let Some(stem) = path.file_stem() {
                        streams.push(stem.to_string_lossy().to_string());
                    }
                }
            }
        }
        streams.sort();
        Ok(streams)
    }

    /// Verify cache integrity using checksums
    pub fn verify(&self, stream: &TestStream) -> Result<bool> {
        let path = match self.get_cached_path(stream) {
            Some(p) => p,
            None => return Ok(false),
        };

        let expected = match &stream.bitstream_checksum {
            Some(c) => c,
            None => {
                warn!(
                    "No checksum available for stream {}, skipping verification",
                    stream.id
                );
                return Ok(true);
            }
        };

        let data = fs::read(&path)?;
        let actual = crate::checksum::compute_md5(&data);

        if actual != *expected {
            warn!(
                "Cache integrity check failed for {}: expected {}, got {}",
                stream.id, expected, actual
            );
            Ok(false)
        } else {
            debug!("Cache integrity verified for {}", stream.id);
            Ok(true)
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cached streams
    pub stream_count: usize,
    /// Total size in bytes
    pub total_size: u64,
    /// Cache directory
    pub cache_dir: PathBuf,
}

impl CacheStats {
    /// Human-readable size
    pub fn size_human(&self) -> String {
        const KB: u64 = 1024;
        const MB: u64 = KB * 1024;
        const GB: u64 = MB * 1024;

        if self.total_size >= GB {
            format!("{:.2} GB", self.total_size as f64 / GB as f64)
        } else if self.total_size >= MB {
            format!("{:.2} MB", self.total_size as f64 / MB as f64)
        } else if self.total_size >= KB {
            format!("{:.2} KB", self.total_size as f64 / KB as f64)
        } else {
            format!("{} bytes", self.total_size)
        }
    }
}

impl StreamCache {
    /// Get cache statistics
    pub fn stats(&self) -> Result<CacheStats> {
        let streams = self.list_cached()?;
        let size = self.size()?;
        Ok(CacheStats {
            stream_count: streams.len(),
            total_size: size,
            cache_dir: self.cache_dir.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::H264Profile;
    use tempfile::tempdir;

    #[test]
    fn test_cache_operations() {
        let dir = tempdir().unwrap();
        let cache = StreamCache::new(dir.path().to_path_buf()).unwrap();

        let stream = TestStream::builder("test_stream")
            .name("Test Stream")
            .profile(H264Profile::Baseline)
            .build();

        assert!(!cache.is_cached(&stream));

        let data = b"fake bitstream data";
        let path = cache.store(&stream, data).unwrap();
        assert!(path.exists());
        assert!(cache.is_cached(&stream));

        let cached_streams = cache.list_cached().unwrap();
        assert_eq!(cached_streams.len(), 1);
        assert_eq!(cached_streams[0], "test_stream");

        cache.clear(&stream).unwrap();
        assert!(!cache.is_cached(&stream));
    }

    #[test]
    fn test_cache_stats() {
        let dir = tempdir().unwrap();
        let cache = StreamCache::new(dir.path().to_path_buf()).unwrap();

        let stream = TestStream::builder("test_stream")
            .profile(H264Profile::Baseline)
            .build();

        let data = vec![0u8; 1024];
        cache.store(&stream, &data).unwrap();

        let stats = cache.stats().unwrap();
        assert_eq!(stats.stream_count, 1);
        assert!(stats.total_size >= 1024);
    }
}
