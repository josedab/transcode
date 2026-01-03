//! Temporal Noise Shaping (TNS) for AAC.
//!
//! TNS applies an LPC filter in the frequency domain to improve
//! the temporal fine structure of the decoded signal.

use transcode_core::bitstream::BitReader;
use transcode_core::error::Result;

/// Maximum TNS filter order.
pub const TNS_MAX_ORDER: usize = 20;

/// Maximum number of TNS filters.
pub const TNS_MAX_FILTERS: usize = 3;

/// TNS filter data.
#[derive(Debug, Clone, Default)]
pub struct TnsFilter {
    /// Filter length (number of bands).
    pub length: u8,
    /// Filter order.
    pub order: u8,
    /// Filter direction (0 = forward, 1 = backward).
    pub direction: bool,
    /// Coefficient resolution (3 or 4 bits).
    pub coef_resolution: u8,
    /// Coefficient compression.
    pub coef_compress: bool,
    /// Filter coefficients.
    pub coeffs: [i8; TNS_MAX_ORDER],
}

/// TNS data for a channel.
#[derive(Debug, Clone, Default)]
pub struct TnsData {
    /// Number of filters per window.
    pub n_filt: [u8; 8],
    /// Filters for each window.
    pub filters: [[TnsFilter; TNS_MAX_FILTERS]; 8],
}

impl TnsData {
    /// Parse TNS data from bitstream.
    pub fn parse(reader: &mut BitReader<'_>, window_sequence: u8) -> Result<Self> {
        let mut tns = TnsData::default();

        let n_filt_bits = if window_sequence == 2 { 1 } else { 2 };
        let length_bits = if window_sequence == 2 { 4 } else { 6 };
        let order_bits = if window_sequence == 2 { 3 } else { 5 };

        let num_windows = if window_sequence == 2 { 8 } else { 1 };

        for w in 0..num_windows {
            tns.n_filt[w] = reader.read_bits(n_filt_bits)? as u8;

            if tns.n_filt[w] > 0 {
                let coef_res = reader.read_bit()? as u8;

                for f in 0..tns.n_filt[w] as usize {
                    let filter = &mut tns.filters[w][f];

                    filter.length = reader.read_bits(length_bits)? as u8;
                    filter.order = reader.read_bits(order_bits)? as u8;

                    if filter.order > 0 {
                        filter.direction = reader.read_bit()?;
                        filter.coef_compress = reader.read_bit()?;

                        filter.coef_resolution = coef_res + 3 - filter.coef_compress as u8;

                        // Read coefficients
                        for i in 0..filter.order as usize {
                            filter.coeffs[i] =
                                reader.read_bits(filter.coef_resolution)? as i8;
                        }
                    }
                }
            }
        }

        Ok(tns)
    }

    /// Check if TNS is present.
    pub fn is_present(&self) -> bool {
        self.n_filt.iter().any(|&n| n > 0)
    }
}

/// TNS processor.
pub struct TnsProcessor {
    /// LPC coefficients (converted from quantized).
    lpc_coeffs: [f32; TNS_MAX_ORDER],
    /// Filter state.
    state: [f32; TNS_MAX_ORDER],
}

impl TnsProcessor {
    /// Create a new TNS processor.
    pub fn new() -> Self {
        Self {
            lpc_coeffs: [0.0; TNS_MAX_ORDER],
            state: [0.0; TNS_MAX_ORDER],
        }
    }

    /// Apply TNS filtering to spectral coefficients.
    pub fn apply(
        &mut self,
        coeffs: &mut [f32],
        tns_data: &TnsData,
        window_group: usize,
        sfb_offsets: &[u16],
    ) {
        if !tns_data.is_present() {
            return;
        }

        for f in 0..tns_data.n_filt[window_group] as usize {
            let filter = &tns_data.filters[window_group][f];

            if filter.order == 0 {
                continue;
            }

            // Convert quantized coefficients to LPC
            self.convert_lpc_coeffs(filter);

            // Determine filter region
            let start = sfb_offsets[0] as usize;
            let end = sfb_offsets[filter.length as usize].min(coeffs.len() as u16) as usize;

            // Apply filter
            if filter.direction {
                self.apply_filter_backward(coeffs, start, end, filter.order as usize);
            } else {
                self.apply_filter_forward(coeffs, start, end, filter.order as usize);
            }
        }
    }

    /// Convert quantized TNS coefficients to LPC.
    fn convert_lpc_coeffs(&mut self, filter: &TnsFilter) {
        // Coefficient tables for different resolutions
        const COEF_TABLE_3: [f32; 8] = [
            0.0, 0.433, -0.781, -0.975, -0.434, -0.782, 0.974, 0.0,
        ];
        const COEF_TABLE_4: [f32; 16] = [
            0.0, 0.207, 0.413, 0.609, 0.793, 0.924, 0.991, 1.0,
            -0.207, -0.413, -0.609, -0.793, -0.924, -0.991, -1.0, 0.0,
        ];

        let mut tmp = [0.0f32; TNS_MAX_ORDER];

        for i in 0..filter.order as usize {
            let coef_idx = filter.coeffs[i] as usize;
            let coef = if filter.coef_resolution == 3 {
                COEF_TABLE_3[coef_idx & 7]
            } else {
                COEF_TABLE_4[coef_idx & 15]
            };

            // Levinson-Durbin recursion
            self.lpc_coeffs[i] = coef;
            tmp[..i].copy_from_slice(&self.lpc_coeffs[..i]);
            for j in 0..i {
                self.lpc_coeffs[j] += coef * tmp[i - 1 - j];
            }
        }
    }

    /// Apply TNS filter in forward direction.
    fn apply_filter_forward(&mut self, coeffs: &mut [f32], start: usize, end: usize, order: usize) {
        self.state.fill(0.0);

        for i in start..end {
            let mut y = coeffs[i];
            for j in 0..order {
                y -= self.lpc_coeffs[j] * self.state[j];
            }

            // Shift state
            for j in (1..order).rev() {
                self.state[j] = self.state[j - 1];
            }
            if order > 0 {
                self.state[0] = coeffs[i];
            }

            coeffs[i] = y;
        }
    }

    /// Apply TNS filter in backward direction.
    fn apply_filter_backward(
        &mut self,
        coeffs: &mut [f32],
        start: usize,
        end: usize,
        order: usize,
    ) {
        self.state.fill(0.0);

        for i in (start..end).rev() {
            let mut y = coeffs[i];
            for j in 0..order {
                y -= self.lpc_coeffs[j] * self.state[j];
            }

            // Shift state
            for j in (1..order).rev() {
                self.state[j] = self.state[j - 1];
            }
            if order > 0 {
                self.state[0] = coeffs[i];
            }

            coeffs[i] = y;
        }
    }

    /// Reset processor state.
    pub fn reset(&mut self) {
        self.lpc_coeffs.fill(0.0);
        self.state.fill(0.0);
    }
}

impl Default for TnsProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tns_filter_default() {
        let filter = TnsFilter::default();
        assert_eq!(filter.length, 0);
        assert_eq!(filter.order, 0);
        assert!(!filter.direction);
        assert_eq!(filter.coef_resolution, 0);
    }

    #[test]
    fn test_tns_data_default() {
        let data = TnsData::default();
        assert!(data.n_filt.iter().all(|&n| n == 0));
        assert!(!data.is_present());
    }

    #[test]
    fn test_tns_data_is_present() {
        let mut data = TnsData::default();
        assert!(!data.is_present());

        data.n_filt[0] = 1;
        assert!(data.is_present());
    }

    #[test]
    fn test_tns_processor_new() {
        let processor = TnsProcessor::new();
        assert!(processor.lpc_coeffs.iter().all(|&c| c == 0.0));
        assert!(processor.state.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_tns_processor_reset() {
        let mut processor = TnsProcessor::new();
        processor.lpc_coeffs[0] = 1.0;
        processor.state[0] = 1.0;

        processor.reset();
        assert!(processor.lpc_coeffs.iter().all(|&c| c == 0.0));
        assert!(processor.state.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_tns_processor_apply_empty() {
        let mut processor = TnsProcessor::new();
        let tns_data = TnsData::default();
        let mut coeffs = vec![1.0f32; 1024];
        let sfb_offsets = vec![0u16, 100, 200, 300, 400];

        // Should not modify coefficients when no filters present
        processor.apply(&mut coeffs, &tns_data, 0, &sfb_offsets);
        assert!(coeffs.iter().all(|&c| c == 1.0));
    }
}
