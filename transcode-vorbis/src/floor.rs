//! Vorbis floor implementation.
//!
//! Floors represent the spectral envelope in Vorbis.
//! Floor type 0 uses LSP coefficients, floor type 1 uses piecewise linear curves.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::ptr_arg)]

use crate::codebook::Codebook;
use crate::error::Result;

/// Floor type 0 configuration (LSP-based).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Floor0Config {
    /// Order of the floor polynomial.
    pub order: u8,
    /// Bark map size.
    pub rate: u16,
    /// Bark map size.
    pub bark_map_size: u16,
    /// Amplitude bits.
    pub amplitude_bits: u8,
    /// Amplitude offset.
    pub amplitude_offset: u8,
    /// Number of books.
    pub number_of_books: u8,
    /// Book list.
    pub book_list: Vec<u8>,
}

/// Floor type 1 configuration (piecewise linear).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Floor1Config {
    /// Number of partitions.
    pub partitions: u8,
    /// Partition class list.
    pub partition_classes: Vec<u8>,
    /// Class dimensions.
    pub class_dimensions: Vec<u8>,
    /// Class subclasses.
    pub class_subclasses: Vec<u8>,
    /// Class masterbooks.
    pub class_masterbooks: Vec<u8>,
    /// Class subclass books.
    pub subclass_books: Vec<Vec<i8>>,
    /// Multiplier (1-4).
    pub multiplier: u8,
    /// X list (positions).
    pub x_list: Vec<u16>,
}

/// Floor configuration.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum FloorConfig {
    /// Floor type 0 (LSP).
    Type0(Floor0Config),
    /// Floor type 1 (piecewise linear).
    Type1(Floor1Config),
}

/// Floor decoder/encoder.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Floor {
    config: FloorConfig,
}

#[allow(dead_code)]
impl Floor {
    /// Create a floor type 1 with default configuration.
    pub fn new_type1(block_size: usize) -> Self {
        // Default floor 1 configuration
        let x_list = Self::generate_default_x_list(block_size);

        Self {
            config: FloorConfig::Type1(Floor1Config {
                partitions: 2,
                partition_classes: vec![0, 1],
                class_dimensions: vec![2, 3],
                class_subclasses: vec![0, 1],
                class_masterbooks: vec![0, 0],
                subclass_books: vec![vec![0, 1], vec![2, 3]],
                multiplier: 2,
                x_list,
            }),
        }
    }

    /// Generate default X list for floor 1.
    fn generate_default_x_list(block_size: usize) -> Vec<u16> {
        let n = block_size / 2;
        let mut x_list = vec![0, n as u16];

        // Add intermediate points (simplified)
        let num_points = 8.min(n);
        for i in 1..num_points {
            let pos = (n * i / num_points) as u16;
            if !x_list.contains(&pos) {
                x_list.push(pos);
            }
        }

        x_list.sort();
        x_list
    }

    /// Decode floor data.
    pub fn decode(
        &self,
        _codebooks: &[Codebook],
        _data: &[u8],
        _offset: &mut usize,
        output: &mut [f32],
    ) -> Result<bool> {
        match &self.config {
            FloorConfig::Type0(config) => self.decode_type0(config, output),
            FloorConfig::Type1(config) => self.decode_type1(config, output),
        }
    }

    /// Decode floor type 0 (LSP).
    fn decode_type0(&self, _config: &Floor0Config, output: &mut [f32]) -> Result<bool> {
        // Simplified: just fill with 1.0 (no floor)
        for sample in output.iter_mut() {
            *sample = 1.0;
        }
        Ok(true)
    }

    /// Decode floor type 1 (piecewise linear).
    fn decode_type1(&self, config: &Floor1Config, output: &mut [f32]) -> Result<bool> {
        let n = output.len();

        // Simplified floor 1 decoding - linear interpolation between points
        // A full implementation would decode Y values from the bitstream
        let y_values: Vec<f32> = config.x_list.iter()
            .map(|&x| 1.0 - (x as f32 / n as f32) * 0.5) // Simple slope
            .collect();

        // Linear interpolation
        for i in 0..n {
            let x = i as f32;

            // Find surrounding X points
            let mut j = 0;
            while j < config.x_list.len() - 1 && config.x_list[j + 1] as f32 <= x {
                j += 1;
            }

            if j >= config.x_list.len() - 1 {
                output[i] = *y_values.last().unwrap_or(&1.0);
            } else {
                let x0 = config.x_list[j] as f32;
                let x1 = config.x_list[j + 1] as f32;
                let y0 = y_values[j];
                let y1 = y_values[j + 1];

                let t = (x - x0) / (x1 - x0);
                output[i] = y0 + t * (y1 - y0);
            }
        }

        Ok(true)
    }

    /// Encode floor data.
    pub fn encode(
        &self,
        spectrum: &[f32],
        output: &mut Vec<u8>,
    ) -> Result<Vec<f32>> {
        match &self.config {
            FloorConfig::Type0(config) => self.encode_type0(config, spectrum, output),
            FloorConfig::Type1(config) => self.encode_type1(config, spectrum, output),
        }
    }

    /// Encode floor type 0.
    fn encode_type0(
        &self,
        _config: &Floor0Config,
        spectrum: &[f32],
        _output: &mut Vec<u8>,
    ) -> Result<Vec<f32>> {
        // Return flat floor for simplicity
        Ok(vec![1.0; spectrum.len()])
    }

    /// Encode floor type 1.
    fn encode_type1(
        &self,
        config: &Floor1Config,
        spectrum: &[f32],
        _output: &mut Vec<u8>,
    ) -> Result<Vec<f32>> {
        let n = spectrum.len();
        let mut floor_curve = vec![1.0f32; n];

        // Compute floor by finding spectral peaks at X positions
        for (i, &x) in config.x_list.iter().enumerate() {
            let x = x as usize;
            if x < n {
                // Use local average as floor value
                let start = x.saturating_sub(4);
                let end = (x + 4).min(n);
                let avg: f32 = spectrum[start..end]
                    .iter()
                    .map(|s| s.abs())
                    .sum::<f32>() / (end - start) as f32;

                // Interpolate to neighbors
                let prev_x = if i > 0 { config.x_list[i - 1] as usize } else { 0 };
                for j in prev_x..=x {
                    if j < n {
                        floor_curve[j] = avg.max(0.001);
                    }
                }
            }
        }

        Ok(floor_curve)
    }

    /// Get the floor configuration.
    pub fn config(&self) -> &FloorConfig {
        &self.config
    }
}

/// Render floor curve to linear domain.
#[allow(dead_code)]
pub fn render_floor(curve: &[f32], output: &mut [f32]) {
    assert_eq!(curve.len(), output.len());

    for (i, &val) in curve.iter().enumerate() {
        // Floor values are typically stored as dB, convert to linear
        output[i] = if val > 0.0 {
            (10.0f32).powf(val / 20.0)
        } else {
            0.0
        };
    }
}

/// Apply floor curve to residue.
#[allow(dead_code)]
pub fn apply_floor(floor: &[f32], residue: &mut [f32]) {
    for (f, r) in floor.iter().zip(residue.iter_mut()) {
        *r *= *f;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_floor_creation() {
        let floor = Floor::new_type1(256);
        match floor.config() {
            FloorConfig::Type1(config) => {
                assert!(!config.x_list.is_empty());
                assert!(config.x_list[0] == 0);
            }
            _ => panic!("Expected Type1 floor"),
        }
    }

    #[test]
    fn test_floor_decode() {
        let floor = Floor::new_type1(256);
        let codebooks = vec![];
        let data = vec![];
        let mut offset = 0;
        let mut output = vec![0.0f32; 128];

        let result = floor.decode(&codebooks, &data, &mut offset, &mut output);
        assert!(result.is_ok());
    }

    #[test]
    fn test_floor_encode() {
        let floor = Floor::new_type1(256);
        let spectrum: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin().abs()).collect();
        let mut output = Vec::new();

        let curve = floor.encode(&spectrum, &mut output).unwrap();
        assert_eq!(curve.len(), 128);
    }

    #[test]
    fn test_apply_floor() {
        let floor = vec![2.0f32; 10];
        let mut residue = vec![1.0f32; 10];

        apply_floor(&floor, &mut residue);

        for &r in &residue {
            assert!((r - 2.0).abs() < 0.0001);
        }
    }
}
