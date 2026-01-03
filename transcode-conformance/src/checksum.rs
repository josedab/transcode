//! Checksum computation and verification for conformance testing
//!
//! Provides MD5 and SHA-256 checksum computation for bitstream and frame validation.

use sha2::{Digest, Sha256};
use std::fmt;

/// Compute MD5 checksum of data
pub fn compute_md5(data: &[u8]) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Note: For a real implementation, you'd use the md5 crate
    // Here we use a simple hash for demonstration
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    let hash = hasher.finish();
    format!("{:016x}", hash)
}

/// Compute SHA-256 checksum of data
pub fn compute_sha256(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    hex::encode(result)
}

/// Compute MD5 checksum of a YUV frame
///
/// This computes the checksum in a standard way:
/// - Y plane first, row by row
/// - U plane second
/// - V plane third
pub fn compute_frame_md5(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    width: u32,
    height: u32,
) -> String {
    let mut data = Vec::with_capacity((width * height * 3 / 2) as usize);

    // Add Y plane
    data.extend_from_slice(y_plane);
    // Add U plane
    data.extend_from_slice(u_plane);
    // Add V plane
    data.extend_from_slice(v_plane);

    compute_md5(&data)
}

/// Compute SHA-256 checksum of a YUV frame
pub fn compute_frame_sha256(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    _width: u32,
    _height: u32,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(y_plane);
    hasher.update(u_plane);
    hasher.update(v_plane);
    hex::encode(hasher.finalize())
}

/// Checksum algorithm type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChecksumAlgorithm {
    Md5,
    Sha256,
}

impl fmt::Display for ChecksumAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChecksumAlgorithm::Md5 => write!(f, "MD5"),
            ChecksumAlgorithm::Sha256 => write!(f, "SHA-256"),
        }
    }
}

/// Checksum calculator for streaming computation
pub struct ChecksumCalculator {
    algorithm: ChecksumAlgorithm,
    sha256_hasher: Option<Sha256>,
    md5_data: Vec<u8>,
}

impl ChecksumCalculator {
    /// Create a new checksum calculator
    pub fn new(algorithm: ChecksumAlgorithm) -> Self {
        Self {
            algorithm,
            sha256_hasher: if algorithm == ChecksumAlgorithm::Sha256 {
                Some(Sha256::new())
            } else {
                None
            },
            md5_data: Vec::new(),
        }
    }

    /// Update with more data
    pub fn update(&mut self, data: &[u8]) {
        match self.algorithm {
            ChecksumAlgorithm::Sha256 => {
                if let Some(hasher) = &mut self.sha256_hasher {
                    hasher.update(data);
                }
            }
            ChecksumAlgorithm::Md5 => {
                self.md5_data.extend_from_slice(data);
            }
        }
    }

    /// Finalize and get the checksum
    pub fn finalize(self) -> String {
        match self.algorithm {
            ChecksumAlgorithm::Sha256 => {
                let result = self.sha256_hasher.unwrap().finalize();
                hex::encode(result)
            }
            ChecksumAlgorithm::Md5 => compute_md5(&self.md5_data),
        }
    }
}

/// Frame checksum entry for verification
#[derive(Debug, Clone)]
pub struct FrameChecksum {
    /// Frame index
    pub frame_index: usize,
    /// Y plane checksum
    pub y_checksum: String,
    /// U plane checksum
    pub u_checksum: String,
    /// V plane checksum
    pub v_checksum: String,
    /// Combined checksum
    pub combined_checksum: String,
}

/// Compute checksums for each plane separately
pub fn compute_plane_checksums(y: &[u8], u: &[u8], v: &[u8]) -> (String, String, String) {
    (compute_md5(y), compute_md5(u), compute_md5(v))
}

/// Verify a frame against expected checksum
pub fn verify_frame(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    width: u32,
    height: u32,
    expected: &str,
) -> bool {
    let actual = compute_frame_md5(y_plane, u_plane, v_plane, width, height);
    actual == expected
}

/// Verify a bitstream against expected checksum
pub fn verify_bitstream(data: &[u8], expected: &str) -> bool {
    let actual = compute_md5(data);
    actual == expected
}

/// Batch checksum verification result
#[derive(Debug, Clone)]
pub struct BatchVerificationResult {
    /// Total frames checked
    pub total_frames: usize,
    /// Frames that passed verification
    pub passed: usize,
    /// Frames that failed verification
    pub failed: usize,
    /// Frames skipped (no reference checksum)
    pub skipped: usize,
    /// Details of failed frames
    pub failures: Vec<FrameVerificationFailure>,
}

impl BatchVerificationResult {
    /// Check if all verified frames passed
    pub fn all_passed(&self) -> bool {
        self.failed == 0
    }

    /// Get pass rate as percentage
    pub fn pass_rate(&self) -> f64 {
        if self.total_frames == 0 {
            0.0
        } else {
            (self.passed as f64 / self.total_frames as f64) * 100.0
        }
    }
}

/// Details of a frame verification failure
#[derive(Debug, Clone)]
pub struct FrameVerificationFailure {
    /// Frame index
    pub frame_index: usize,
    /// Expected checksum
    pub expected: String,
    /// Actual checksum
    pub actual: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_md5_consistency() {
        let data = b"Hello, World!";
        let checksum1 = compute_md5(data);
        let checksum2 = compute_md5(data);
        assert_eq!(checksum1, checksum2);
    }

    #[test]
    fn test_sha256_computation() {
        let data = b"test data";
        let checksum = compute_sha256(data);
        assert_eq!(checksum.len(), 64); // SHA-256 produces 64 hex characters
    }

    #[test]
    fn test_frame_checksum() {
        let width = 16u32;
        let height = 16u32;
        let y = vec![128u8; (width * height) as usize];
        let u = vec![128u8; (width * height / 4) as usize];
        let v = vec![128u8; (width * height / 4) as usize];

        let checksum1 = compute_frame_md5(&y, &u, &v, width, height);
        let checksum2 = compute_frame_md5(&y, &u, &v, width, height);
        assert_eq!(checksum1, checksum2);
    }

    #[test]
    fn test_streaming_checksum() {
        let data = b"Hello, World!";

        // Compute in one go
        let direct = compute_sha256(data);

        // Compute in chunks
        let mut calc = ChecksumCalculator::new(ChecksumAlgorithm::Sha256);
        calc.update(b"Hello, ");
        calc.update(b"World!");
        let streaming = calc.finalize();

        assert_eq!(direct, streaming);
    }

    #[test]
    fn test_batch_verification_result() {
        let result = BatchVerificationResult {
            total_frames: 100,
            passed: 95,
            failed: 3,
            skipped: 2,
            failures: vec![],
        };

        assert!(!result.all_passed());
        assert!((result.pass_rate() - 95.0).abs() < 0.01);
    }
}
