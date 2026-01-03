//! Watermark extraction algorithms

use crate::{Result, WatermarkError};

/// Watermark detector
pub struct WatermarkDetector {
    threshold: f64,
}

impl WatermarkDetector {
    /// Create a new detector
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Detect if watermark is present
    pub fn detect(&self, correlation: f64) -> bool {
        correlation.abs() > self.threshold
    }

    /// Decode bit from correlation
    pub fn decode_bit(&self, correlation: f64) -> Option<bool> {
        if correlation > self.threshold {
            Some(true)
        } else if correlation < -self.threshold {
            Some(false)
        } else {
            None
        }
    }
}

/// Multi-frame extractor for improved robustness
pub struct MultiFrameExtractor {
    frame_bits: Vec<Vec<Option<bool>>>,
    num_frames: usize,
}

impl MultiFrameExtractor {
    /// Create a new multi-frame extractor
    pub fn new() -> Self {
        Self {
            frame_bits: Vec::new(),
            num_frames: 0,
        }
    }

    /// Add bits from a frame
    pub fn add_frame(&mut self, bits: Vec<Option<bool>>) {
        self.frame_bits.push(bits);
        self.num_frames += 1;
    }

    /// Get consensus bits using majority voting
    pub fn get_consensus(&self) -> Result<Vec<bool>> {
        if self.frame_bits.is_empty() {
            return Err(WatermarkError::NotFound);
        }

        let bit_count = self.frame_bits[0].len();
        let mut result = Vec::with_capacity(bit_count);

        for i in 0..bit_count {
            let mut ones = 0;
            let mut zeros = 0;

            for frame in &self.frame_bits {
                if let Some(Some(bit)) = frame.get(i) {
                    if *bit {
                        ones += 1;
                    } else {
                        zeros += 1;
                    }
                }
            }

            if ones + zeros == 0 {
                return Err(WatermarkError::Corrupted);
            }

            result.push(ones > zeros);
        }

        Ok(result)
    }

    /// Get detection confidence
    pub fn confidence(&self) -> f64 {
        if self.frame_bits.is_empty() {
            return 0.0;
        }

        let bit_count = self.frame_bits[0].len();
        let mut total_confidence = 0.0;

        for i in 0..bit_count {
            let mut ones = 0;
            let mut zeros = 0;

            for frame in &self.frame_bits {
                if let Some(Some(bit)) = frame.get(i) {
                    if *bit {
                        ones += 1;
                    } else {
                        zeros += 1;
                    }
                }
            }

            let total = ones + zeros;
            if total > 0 {
                let max_votes = ones.max(zeros);
                total_confidence += max_votes as f64 / total as f64;
            }
        }

        total_confidence / bit_count as f64
    }
}

impl Default for MultiFrameExtractor {
    fn default() -> Self {
        Self::new()
    }
}
