//! Watermark embedding algorithms

/// Embedding position selector
pub struct PositionSelector {
    key: [u8; 32],
    seed: u64,
}

impl PositionSelector {
    /// Create a new position selector
    pub fn new(key: [u8; 32]) -> Self {
        Self { key, seed: 0 }
    }

    /// Get next position
    pub fn next(&mut self, max: usize) -> usize {
        use sha2::{Sha256, Digest};

        let mut hasher = Sha256::new();
        hasher.update(self.key);
        hasher.update(self.seed.to_le_bytes());
        let hash = hasher.finalize();

        self.seed = self.seed.wrapping_add(1);

        let value = u64::from_le_bytes(hash[0..8].try_into().unwrap());
        (value as usize) % max
    }

    /// Reset sequence
    pub fn reset(&mut self) {
        self.seed = 0;
    }
}

/// Spread spectrum embedding
pub struct SpreadSpectrumEmbedder {
    chip_length: usize,
}

impl SpreadSpectrumEmbedder {
    /// Create new spread spectrum embedder
    pub fn new(chip_length: usize) -> Self {
        Self { chip_length }
    }

    /// Generate spreading sequence for a bit
    pub fn generate_sequence(&self, bit: bool, key: &[u8; 32]) -> Vec<i8> {
        use sha2::{Sha256, Digest};

        let mut sequence = Vec::with_capacity(self.chip_length);
        let sign = if bit { 1i8 } else { -1i8 };

        for i in 0..self.chip_length {
            let mut hasher = Sha256::new();
            hasher.update(key);
            hasher.update((i as u64).to_le_bytes());
            let hash = hasher.finalize();

            let chip = if hash[0] & 1 == 1 { 1i8 } else { -1i8 };
            sequence.push(chip * sign);
        }

        sequence
    }

    /// Correlate signal with sequence
    pub fn correlate(&self, signal: &[i8], sequence: &[i8]) -> f64 {
        let len = signal.len().min(sequence.len());
        let sum: i64 = signal.iter()
            .zip(sequence.iter())
            .take(len)
            .map(|(&s, &seq)| s as i64 * seq as i64)
            .sum();

        sum as f64 / len as f64
    }
}
