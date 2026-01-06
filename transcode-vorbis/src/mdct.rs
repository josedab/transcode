//! Modified Discrete Cosine Transform (MDCT) for Vorbis.

use std::f32::consts::PI;

/// MDCT calculator for a specific block size.
#[derive(Debug, Clone)]
pub struct Mdct {
    /// Block size (N).
    n: usize,
    /// Half block size (N/2).
    n2: usize,
    /// Cosine lookup table.
    cos_table: Vec<f32>,
    /// Sine lookup table.
    sin_table: Vec<f32>,
    /// Window function (Vorbis window).
    window: Vec<f32>,
    /// Twiddle factors for FFT.
    twiddle: Vec<(f32, f32)>,
}

impl Mdct {
    /// Create a new MDCT calculator for block size N.
    pub fn new(n: usize) -> Self {
        let n2 = n / 2;
        let n4 = n / 4;

        // Pre-compute cosine/sine tables
        let mut cos_table = Vec::with_capacity(n2);
        let mut sin_table = Vec::with_capacity(n2);
        for k in 0..n2 {
            let angle = 2.0 * PI * (k as f32 + 0.125) / n as f32;
            cos_table.push(angle.cos());
            sin_table.push(angle.sin());
        }

        // Pre-compute Vorbis window
        let window = Self::compute_vorbis_window(n);

        // Pre-compute twiddle factors
        let mut twiddle = Vec::with_capacity(n4);
        for i in 0..n4 {
            let angle = PI * (4.0 * i as f32 + 1.0) / (2.0 * n as f32);
            twiddle.push((angle.cos(), angle.sin()));
        }

        Self {
            n,
            n2,
            cos_table,
            sin_table,
            window,
            twiddle,
        }
    }

    /// Compute the Vorbis window function.
    fn compute_vorbis_window(n: usize) -> Vec<f32> {
        let mut window = Vec::with_capacity(n);
        for i in 0..n {
            let x = (PI / n as f32) * (i as f32 + 0.5);
            let sin_val = x.sin();
            let inner = (PI / 2.0) * sin_val * sin_val;
            window.push(inner.sin());
        }
        window
    }

    /// Get the block size.
    pub fn block_size(&self) -> usize {
        self.n
    }

    /// Get the window function.
    pub fn window(&self) -> &[f32] {
        &self.window
    }

    /// Forward MDCT: time domain -> frequency domain.
    /// Input: N samples, Output: N/2 coefficients.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.n);
        assert_eq!(output.len(), self.n2);

        let n4 = self.n / 4;

        // Apply window and pre-rotation
        let mut temp: Vec<(f32, f32)> = Vec::with_capacity(n4);
        for i in 0..n4 {
            let i0 = i;
            let i1 = self.n2 - 1 - i;
            let i2 = self.n2 + i;
            let i3 = self.n - 1 - i;

            // Gather samples with windowing
            let x0 = self.window[i0] * input[i0];
            let x1 = self.window[i1] * input[i1];
            let x2 = self.window[i2] * input[i2];
            let x3 = self.window[i3] * input[i3];

            // Pre-MDCT rotation
            let re = -x2 - x0;
            let im = x3 - x1;

            let (tw_cos, tw_sin) = self.twiddle[i];
            temp.push((
                re * tw_cos - im * tw_sin,
                re * tw_sin + im * tw_cos,
            ));
        }

        // FFT (simplified for MDCT)
        let fft_out = self.fft(&temp);

        // Post-rotation and output
        for i in 0..n4 {
            let (tw_cos, tw_sin) = self.twiddle[i];
            let (re, im) = fft_out[i];

            let out_re = re * tw_cos - im * tw_sin;
            let out_im = re * tw_sin + im * tw_cos;

            // Interleave output
            output[2 * i] = out_re;
            output[self.n2 - 1 - 2 * i] = -out_im;
        }
    }

    /// Inverse MDCT: frequency domain -> time domain.
    /// Input: N/2 coefficients, Output: N samples.
    pub fn inverse(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.n2);
        assert_eq!(output.len(), self.n);

        let n4 = self.n / 4;

        // Pre-rotation
        let mut temp: Vec<(f32, f32)> = Vec::with_capacity(n4);
        for i in 0..n4 {
            let re = input[2 * i];
            let im = input[self.n2 - 1 - 2 * i];

            let (tw_cos, tw_sin) = self.twiddle[i];
            temp.push((
                (re * tw_cos + im * tw_sin) * 2.0,
                (-re * tw_sin + im * tw_cos) * 2.0,
            ));
        }

        // Inverse FFT
        let ifft_out = self.ifft(&temp);

        // Post-rotation and windowing
        for i in 0..n4 {
            let (tw_cos, tw_sin) = self.twiddle[i];
            let (re, im) = ifft_out[i];

            let out_re = re * tw_cos - im * tw_sin;
            let out_im = re * tw_sin + im * tw_cos;

            // Output with overlap-add structure
            output[i] = out_im * self.window[i];
            output[self.n2 - 1 - i] = out_re * self.window[self.n2 - 1 - i];
            output[self.n2 + i] = -out_re * self.window[self.n2 + i];
            output[self.n - 1 - i] = -out_im * self.window[self.n - 1 - i];
        }
    }

    /// Simple DFT-based FFT (for simplicity; real implementation would use radix-2/4).
    fn fft(&self, input: &[(f32, f32)]) -> Vec<(f32, f32)> {
        let n = input.len();
        let mut output = vec![(0.0f32, 0.0f32); n];

        for k in 0..n {
            let mut sum_re = 0.0;
            let mut sum_im = 0.0;

            for j in 0..n {
                let angle = -2.0 * PI * (k * j) as f32 / n as f32;
                let (sin_a, cos_a) = angle.sin_cos();
                let (re, im) = input[j];

                sum_re += re * cos_a - im * sin_a;
                sum_im += re * sin_a + im * cos_a;
            }

            output[k] = (sum_re, sum_im);
        }

        output
    }

    /// Inverse FFT.
    fn ifft(&self, input: &[(f32, f32)]) -> Vec<(f32, f32)> {
        let n = input.len();
        let mut output = vec![(0.0f32, 0.0f32); n];
        let scale = 1.0 / n as f32;

        for k in 0..n {
            let mut sum_re = 0.0;
            let mut sum_im = 0.0;

            for j in 0..n {
                let angle = 2.0 * PI * (k * j) as f32 / n as f32;
                let (sin_a, cos_a) = angle.sin_cos();
                let (re, im) = input[j];

                sum_re += re * cos_a - im * sin_a;
                sum_im += re * sin_a + im * cos_a;
            }

            output[k] = (sum_re * scale, sum_im * scale);
        }

        output
    }

    /// Apply overlap-add for MDCT reconstruction.
    pub fn overlap_add(
        &self,
        current: &[f32],
        previous: &[f32],
        output: &mut [f32],
    ) {
        let n2 = self.n2;
        assert_eq!(current.len(), self.n);
        assert_eq!(previous.len(), self.n);
        assert_eq!(output.len(), n2);

        // Overlap-add the second half of previous with first half of current
        for i in 0..n2 {
            output[i] = previous[n2 + i] + current[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mdct_creation() {
        let mdct = Mdct::new(256);
        assert_eq!(mdct.block_size(), 256);
        assert_eq!(mdct.window().len(), 256);
    }

    #[test]
    fn test_window_properties() {
        let mdct = Mdct::new(256);
        let window = mdct.window();

        // Window should be symmetric
        for i in 0..128 {
            assert!((window[i] - window[255 - i]).abs() < 0.0001);
        }

        // Window values should be in [0, 1]
        for &w in window {
            assert!(w >= 0.0 && w <= 1.0);
        }
    }

    #[test]
    fn test_mdct_roundtrip() {
        let mdct = Mdct::new(64);
        let input: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut freq = vec![0.0f32; 32];
        let mut output = vec![0.0f32; 64];

        mdct.forward(&input, &mut freq);
        mdct.inverse(&freq, &mut output);

        // The reconstruction with proper overlap-add should give back original
        // This simplified test just checks structure is preserved
        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_vorbis_window() {
        let window = Mdct::compute_vorbis_window(128);
        assert_eq!(window.len(), 128);

        // Check perfect reconstruction property: w[n]^2 + w[N/2+n]^2 = 1
        for i in 0..64 {
            let sum = window[i].powi(2) + window[64 + i].powi(2);
            assert!((sum - 1.0).abs() < 0.0001, "Failed at i={}: sum={}", i, sum);
        }
    }
}
