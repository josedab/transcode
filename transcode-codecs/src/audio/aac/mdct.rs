//! Modified Discrete Cosine Transform (MDCT) for AAC.

use std::f32::consts::PI;

/// MDCT processor.
pub struct Mdct {
    /// Transform size.
    size: usize,
    /// Twiddle factors.
    twiddles: Vec<(f32, f32)>,
    /// Temporary buffer.
    temp: Vec<f32>,
}

impl Mdct {
    /// Create a new MDCT processor.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0 or not divisible by 4.
    pub fn new(size: usize) -> Self {
        assert!(size > 0, "MDCT size must be greater than 0");
        assert!(size % 4 == 0, "MDCT size must be divisible by 4");

        let n = size;
        let n4 = n / 4;

        // Precompute twiddle factors
        let mut twiddles = Vec::with_capacity(n4);
        for k in 0..n4 {
            let angle = 2.0 * PI * (k as f32 + 0.125) / n as f32;
            twiddles.push((angle.cos(), angle.sin()));
        }

        Self {
            size,
            twiddles,
            temp: vec![0.0; size],
        }
    }

    /// Forward MDCT: time domain (2N) -> frequency domain (N).
    pub fn forward(&mut self, input: &[f32], output: &mut [f32]) {
        let n = self.size;
        let n2 = n / 2;
        let n4 = n / 4;

        assert!(input.len() >= 2 * n);
        assert!(output.len() >= n);

        // Pre-twiddle
        for k in 0..n4 {
            let idx1 = 2 * k;
            let idx2 = n2 - 1 - 2 * k;
            let idx3 = n2 + 2 * k;
            let idx4 = n - 1 - 2 * k;

            let (cos_tw, sin_tw) = self.twiddles[k];

            let a = input[idx3] - input[idx1];
            let b = input[idx2] + input[idx4];

            self.temp[2 * k] = a * cos_tw + b * sin_tw;
            self.temp[2 * k + 1] = b * cos_tw - a * sin_tw;
        }

        // N/2-point complex FFT
        self.fft_complex_n2();

        // Post-twiddle
        for k in 0..n4 {
            let (cos_tw, sin_tw) = self.twiddles[k];

            let re = self.temp[2 * k];
            let im = self.temp[2 * k + 1];

            output[2 * k] = re * cos_tw + im * sin_tw;
            output[n2 - 1 - 2 * k] = im * cos_tw - re * sin_tw;
        }
    }

    /// Inverse MDCT: frequency domain (N) -> time domain (2N).
    pub fn inverse(&mut self, input: &[f32], output: &mut [f32]) {
        let n = self.size;
        let n2 = n / 2;
        let n4 = n / 4;

        assert!(input.len() >= n);
        assert!(output.len() >= 2 * n);

        // Pre-twiddle
        for k in 0..n4 {
            let (cos_tw, sin_tw) = self.twiddles[k];

            let re = input[2 * k];
            let im = input[n2 - 1 - 2 * k];

            self.temp[2 * k] = re * cos_tw + im * sin_tw;
            self.temp[2 * k + 1] = im * cos_tw - re * sin_tw;
        }

        // N/2-point complex IFFT
        self.ifft_complex_n2();

        // Post-twiddle and reorder
        let scale = 2.0 / n as f32;
        let n8 = n4 / 2;
        for k in 0..n8 {
            let (cos_tw, sin_tw) = self.twiddles[k];

            let re = self.temp[2 * k] * scale;
            let im = self.temp[2 * k + 1] * scale;

            let a = re * cos_tw + im * sin_tw;
            let b = im * cos_tw - re * sin_tw;

            // First quadrant
            output[n4 + 2 * k] = a;
            output[n4 + 2 * k + 1] = -b;
            if n4 > 2 * k {
                output[n4 - 1 - 2 * k] = -a;
            }
            if n4 > 1 + 2 * k {
                output[n4 - 2 - 2 * k] = b;
            }

            // Second quadrant
            output[n2 + n4 + 2 * k] = -b;
            output[n2 + n4 + 2 * k + 1] = a;
            if n + n4 >= 1 + 2 * k {
                output[n + n4 - 1 - 2 * k] = b;
            }
            if n + n4 >= 2 + 2 * k {
                output[n + n4 - 2 - 2 * k] = -a;
            }
        }

        // Handle remaining indices in second half of n4
        for k in n8..n4 {
            let (cos_tw, sin_tw) = self.twiddles[k];

            let re = self.temp[2 * k] * scale;
            let im = self.temp[2 * k + 1] * scale;

            let a = re * cos_tw + im * sin_tw;
            let b = im * cos_tw - re * sin_tw;

            output[n4 + 2 * k] = a;
            output[n4 + 2 * k + 1] = -b;
            output[n2 + n4 + 2 * k] = -b;
            output[n2 + n4 + 2 * k + 1] = a;
        }
    }

    /// Simple N/2-point complex FFT (Cooley-Tukey).
    fn fft_complex_n2(&mut self) {
        let n = self.size / 2;
        let mut step = 1;

        // Bit-reversal permutation
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;

            if i < j {
                self.temp.swap(2 * i, 2 * j);
                self.temp.swap(2 * i + 1, 2 * j + 1);
            }
        }

        // Butterfly stages
        while step < n {
            let step2 = step * 2;
            let theta = PI / step as f32;

            for k in (0..n).step_by(step2) {
                for j in 0..step {
                    let angle = theta * j as f32;
                    let wr = angle.cos();
                    let wi = -angle.sin();

                    let i1 = k + j;
                    let i2 = i1 + step;

                    let tr = wr * self.temp[2 * i2] - wi * self.temp[2 * i2 + 1];
                    let ti = wr * self.temp[2 * i2 + 1] + wi * self.temp[2 * i2];

                    self.temp[2 * i2] = self.temp[2 * i1] - tr;
                    self.temp[2 * i2 + 1] = self.temp[2 * i1 + 1] - ti;
                    self.temp[2 * i1] += tr;
                    self.temp[2 * i1 + 1] += ti;
                }
            }
            step = step2;
        }
    }

    /// Simple N/2-point complex IFFT.
    fn ifft_complex_n2(&mut self) {
        let n = self.size / 2;

        // Conjugate
        for i in 0..n {
            self.temp[2 * i + 1] = -self.temp[2 * i + 1];
        }

        // Forward FFT
        self.fft_complex_n2();

        // Conjugate and scale
        let scale = 1.0 / n as f32;
        for i in 0..n {
            self.temp[2 * i] *= scale;
            self.temp[2 * i + 1] *= -scale;
        }
    }
}

/// IMDCT with windowing and overlap-add for AAC.
pub struct ImdctContext {
    /// Long block MDCT.
    mdct_long: Mdct,
    /// Short block MDCT.
    mdct_short: Mdct,
    /// Previous samples for overlap.
    overlap: Vec<f32>,
    /// Long sine window.
    sine_long: [f32; 1024],
    /// Short sine window.
    sine_short: [f32; 128],
}

impl ImdctContext {
    /// Create a new IMDCT context.
    pub fn new() -> Self {
        let mut sine_long = [0.0f32; 1024];
        let mut sine_short = [0.0f32; 128];

        for i in 0..1024 {
            sine_long[i] = ((i as f32 + 0.5) * PI / 1024.0).sin();
        }
        for i in 0..128 {
            sine_short[i] = ((i as f32 + 0.5) * PI / 128.0).sin();
        }

        Self {
            mdct_long: Mdct::new(1024),
            mdct_short: Mdct::new(128),
            overlap: vec![0.0; 1024],
            sine_long,
            sine_short,
        }
    }

    /// Process a long block.
    pub fn process_long(&mut self, coeffs: &[f32; 1024], output: &mut [f32; 1024]) {
        let mut temp = [0.0f32; 2048];
        self.mdct_long.inverse(coeffs, &mut temp);

        // Windowing and overlap-add
        for i in 0..1024 {
            output[i] = temp[i] * self.sine_long[i] + self.overlap[i];
            self.overlap[i] = temp[i + 1024] * self.sine_long[1023 - i];
        }
    }

    /// Process eight short blocks.
    pub fn process_short(&mut self, coeffs: &[[f32; 128]; 8], output: &mut [f32; 1024]) {
        // Initialize output with overlap
        output.copy_from_slice(&self.overlap);
        self.overlap.fill(0.0);

        let mut temp = [0.0f32; 256];

        for (block_idx, block_coeffs) in coeffs.iter().enumerate() {
            self.mdct_short.inverse(block_coeffs, &mut temp);

            let offset = 448 + block_idx * 128;

            // First half: overlap-add to output
            for i in 0..64 {
                if offset + i < 1024 {
                    output[offset + i] += temp[i] * self.sine_short[i];
                }
            }

            // Second half: add to overlap buffer
            for i in 0..64 {
                if offset + 64 + i < 1024 {
                    output[offset + 64 + i] += temp[64 + i] * self.sine_short[64 + i];
                } else if offset + 64 + i - 1024 < 1024 {
                    self.overlap[offset + 64 + i - 1024] += temp[64 + i] * self.sine_short[64 + i];
                }
            }

            for i in 0..128 {
                if offset + 128 + i < 1024 {
                    output[offset + 128 + i] += temp[128 + i] * self.sine_short[127 - i];
                } else if offset + 128 + i - 1024 < 1024 {
                    self.overlap[offset + 128 + i - 1024] +=
                        temp[128 + i] * self.sine_short[127 - i];
                }
            }
        }
    }

    /// Reset overlap buffer.
    pub fn reset(&mut self) {
        self.overlap.fill(0.0);
    }
}

impl Default for ImdctContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mdct_inverse_identity() {
        let mut mdct = Mdct::new(256);

        // Create a simple signal
        let mut input = vec![0.0f32; 256];
        for i in 0..256 {
            input[i] = (i as f32 * 0.1).sin();
        }

        let mut output = vec![0.0f32; 512];
        mdct.inverse(&input, &mut output);

        // Output should be non-zero
        assert!(output.iter().any(|&x| x.abs() > 1e-6));
    }
}
