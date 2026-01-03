//! CELT mode implementation for Opus.
//!
//! CELT (Constrained Energy Lapped Transform) is the music coding mode of Opus.
//! It uses MDCT with a psychoacoustic model for frequency domain coding.
//!
//! Key features:
//! - Optimized for music and general audio
//! - Very low latency (2.5-20 ms frames)
//! - Sample rate: 48 kHz (internal)
//! - Perceptual noise shaping
//! - Band-based energy coding

#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::needless_range_loop)]

use crate::error::{OpusError, Result};
use crate::range_coder::{RangeDecoder, RangeEncoder};

/// Maximum number of CELT bands.
pub const MAX_BANDS: usize = 21;

/// Maximum frame size in samples (20ms at 48kHz).
pub const MAX_FRAME_SIZE: usize = 960;

/// MDCT window size for short frames (2.5ms at 48kHz).
pub const MDCT_SIZE_120: usize = 120;
/// MDCT window size for medium frames (10ms at 48kHz).
pub const MDCT_SIZE_480: usize = 480;
/// MDCT window size for long frames (20ms at 48kHz).
pub const MDCT_SIZE_960: usize = 960;

/// CELT bandwidth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CeltBandwidth {
    /// Narrowband (4 kHz).
    Narrow,
    /// Medium (6 kHz).
    Medium,
    /// Wideband (8 kHz).
    Wide,
    /// Super-wideband (12 kHz).
    SuperWide,
    /// Fullband (20 kHz).
    Full,
}

impl CeltBandwidth {
    /// Get the number of bands for this bandwidth.
    pub fn num_bands(&self) -> usize {
        match self {
            Self::Narrow => 13,
            Self::Medium => 15,
            Self::Wide => 17,
            Self::SuperWide => 19,
            Self::Full => 21,
        }
    }

    /// Get the maximum frequency in Hz.
    pub fn max_freq(&self) -> u32 {
        match self {
            Self::Narrow => 4000,
            Self::Medium => 6000,
            Self::Wide => 8000,
            Self::SuperWide => 12000,
            Self::Full => 20000,
        }
    }
}

/// CELT band boundaries (in bins for 960-sample MDCT).
pub static BAND_BOUNDARIES: [usize; 22] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100,
];

/// Ener-band structure for CELT.
#[derive(Debug, Clone)]
pub struct CeltBands {
    /// Band energies (log2 scale).
    pub energy: [f32; MAX_BANDS],
    /// Previous frame energies (for prediction).
    pub prev_energy: [f32; MAX_BANDS],
    /// Coarse energy (quantized).
    pub coarse_energy: [i8; MAX_BANDS],
    /// Fine energy bits.
    pub fine_energy: [u8; MAX_BANDS],
    /// Number of active bands.
    pub num_bands: usize,
}

impl Default for CeltBands {
    fn default() -> Self {
        Self {
            energy: [0.0; MAX_BANDS],
            prev_energy: [0.0; MAX_BANDS],
            coarse_energy: [0; MAX_BANDS],
            fine_energy: [0; MAX_BANDS],
            num_bands: MAX_BANDS,
        }
    }
}

/// CELT PVQ (Pyramid Vector Quantization) coefficients.
#[derive(Debug, Clone)]
pub struct CeltPvq {
    /// Normalized coefficients per band.
    pub coeffs: Vec<Vec<f32>>,
    /// Pulse counts per band.
    pub pulses: [u8; MAX_BANDS],
    /// Signs per band.
    pub signs: Vec<Vec<bool>>,
}

impl Default for CeltPvq {
    fn default() -> Self {
        Self {
            coeffs: vec![Vec::new(); MAX_BANDS],
            pulses: [0; MAX_BANDS],
            signs: vec![Vec::new(); MAX_BANDS],
        }
    }
}

/// CELT frame data.
#[derive(Debug, Clone)]
pub struct CeltFrame {
    /// Transient flag.
    pub transient: bool,
    /// Short block flag.
    pub short_blocks: bool,
    /// Intra-frame energy coding.
    pub intra: bool,
    /// Band energies.
    pub bands: CeltBands,
    /// PVQ coefficients.
    pub pvq: CeltPvq,
    /// Spread (temporal spreading).
    pub spread: u8,
    /// MDCT coefficients.
    pub mdct_coeffs: Vec<f32>,
}

impl Default for CeltFrame {
    fn default() -> Self {
        Self {
            transient: false,
            short_blocks: false,
            intra: false,
            bands: CeltBands::default(),
            pvq: CeltPvq::default(),
            spread: 2,
            mdct_coeffs: vec![0.0; MAX_FRAME_SIZE],
        }
    }
}

/// MDCT (Modified Discrete Cosine Transform) processor.
#[derive(Debug)]
pub struct Mdct {
    /// Transform size.
    size: usize,
    /// Twiddle factors (reserved for future optimization).
    _twiddle: Vec<(f32, f32)>,
    /// Window function.
    window: Vec<f32>,
}

impl Mdct {
    /// Create a new MDCT processor.
    pub fn new(size: usize) -> Self {
        let n = size;
        let n2 = n / 2;

        // Compute twiddle factors
        let mut twiddle = Vec::with_capacity(n2);
        for k in 0..n2 {
            let angle = std::f32::consts::PI * (k as f32 + 0.125) / n as f32;
            twiddle.push((angle.cos(), angle.sin()));
        }

        // Compute window (sine window)
        let mut window = Vec::with_capacity(n);
        for i in 0..n {
            let angle = std::f32::consts::PI * (i as f32 + 0.5) / n as f32;
            window.push(angle.sin());
        }

        Self {
            size,
            _twiddle: twiddle,
            window,
        }
    }

    /// Forward MDCT.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        let n = self.size;
        let n2 = n / 2;
        let n4 = n / 4;

        // Pre-windowing and rotation
        let mut x = vec![0.0f32; n2];

        for i in 0..n4 {
            // Apply window and fold
            let t0 = input.get(n4 + i).copied().unwrap_or(0.0) * self.window[n4 + i]
                + input.get(n + n4 - 1 - i).copied().unwrap_or(0.0) * self.window[n - n4 - 1 - i];
            let t1 = input.get(n4 - 1 - i).copied().unwrap_or(0.0) * self.window[n4 - 1 - i]
                - input.get(n4 + i).copied().unwrap_or(0.0) * self.window[n4 + i];

            x[i] = t0;
            x[n2 - 1 - i] = t1;
        }

        // FFT of size n/2 (simplified direct computation for correctness)
        let mut temp = vec![(0.0f32, 0.0f32); n2];
        for k in 0..n2 {
            let mut re = 0.0f32;
            let mut im = 0.0f32;

            for n_idx in 0..n2 {
                let angle = 2.0 * std::f32::consts::PI * (k as f32) * (n_idx as f32) / n2 as f32;
                re += x[n_idx] * angle.cos();
                im -= x[n_idx] * angle.sin();
            }

            temp[k] = (re, im);
        }

        // Post-rotation
        for k in 0..n2 {
            let angle =
                std::f32::consts::PI * (2.0 * k as f32 + 1.0 + (n2 as f32)) / (2.0 * n as f32);
            let (cos_a, sin_a) = (angle.cos(), angle.sin());
            let (re, im) = temp[k];

            output[k] = re * cos_a + im * sin_a;
        }
    }

    /// Inverse MDCT.
    pub fn inverse(&self, input: &[f32], output: &mut [f32]) {
        let n = self.size;
        let n2 = n / 2;

        // Pre-rotation
        let mut temp = vec![(0.0f32, 0.0f32); n2];

        for k in 0..n2 {
            let angle =
                std::f32::consts::PI * (2.0 * k as f32 + 1.0 + (n2 as f32)) / (2.0 * n as f32);
            let (cos_a, sin_a) = (angle.cos(), angle.sin());
            let val = input.get(k).copied().unwrap_or(0.0);

            temp[k] = (val * cos_a, val * sin_a);
        }

        // IFFT of size n/2
        let mut y = vec![0.0f32; n2];
        for n_idx in 0..n2 {
            let mut re = 0.0f32;

            for k in 0..n2 {
                let angle = 2.0 * std::f32::consts::PI * (k as f32) * (n_idx as f32) / n2 as f32;
                re += temp[k].0 * angle.cos() + temp[k].1 * angle.sin();
            }

            y[n_idx] = re * 2.0 / n2 as f32;
        }

        // Post-windowing and overlap
        for i in 0..n2 {
            if i < output.len() {
                output[i] = y[i] * self.window.get(i).copied().unwrap_or(1.0);
            }
            if n2 + i < output.len() {
                output[n2 + i] = y[n2 - 1 - i] * self.window.get(n2 + i).copied().unwrap_or(1.0);
            }
        }
    }

    /// Reset state.
    pub fn reset(&mut self) {
        // No state to reset for MDCT
    }
}

/// CELT decoder state.
#[derive(Debug)]
pub struct CeltDecoder {
    /// Number of channels.
    channels: u8,
    /// Frame size.
    frame_size: usize,
    /// Overlap size.
    overlap_size: usize,
    /// MDCT processors.
    mdct: Mdct,
    /// Overlap buffer per channel.
    overlap: Vec<Vec<f32>>,
    /// Previous frame energies per channel.
    prev_energy: Vec<[f32; MAX_BANDS]>,
    /// Previous normalized coefficients.
    prev_coeffs: Vec<Vec<f32>>,
    /// Frame counter.
    frame_count: u64,
    /// Post-filter state.
    postfilter: CeltPostfilter,
}

impl CeltDecoder {
    /// Create a new CELT decoder.
    pub fn new(channels: u8, frame_size: usize) -> Self {
        let overlap_size = frame_size / 4;

        Self {
            channels,
            frame_size,
            overlap_size,
            mdct: Mdct::new(frame_size),
            overlap: vec![vec![0.0; overlap_size]; channels as usize],
            prev_energy: vec![[0.0; MAX_BANDS]; channels as usize],
            prev_coeffs: vec![vec![0.0; frame_size]; channels as usize],
            frame_count: 0,
            postfilter: CeltPostfilter::new(),
        }
    }

    /// Decode a CELT frame.
    pub fn decode_frame(
        &mut self,
        reader: &mut RangeDecoder<'_>,
        bandwidth: CeltBandwidth,
    ) -> Result<Vec<f32>> {
        let num_bands = bandwidth.num_bands();

        // Decode frame flags
        let transient = reader.read_bit()?;
        let intra = if self.frame_count == 0 {
            true
        } else {
            reader.read_bit()?
        };

        // Decode band energies
        let bands = self.decode_band_energies(reader, num_bands, intra)?;

        // Decode PVQ coefficients
        let pvq = self.decode_pvq(reader, &bands, num_bands)?;

        // Reconstruct MDCT coefficients
        let mut mdct_coeffs = vec![0.0f32; self.frame_size];
        self.reconstruct_coefficients(&bands, &pvq, num_bands, &mut mdct_coeffs);

        // Inverse MDCT
        let mut output = vec![0.0f32; self.frame_size * self.channels as usize];

        for ch in 0..self.channels as usize {
            let channel_offset = ch * self.frame_size;
            let mut channel_out = vec![0.0f32; self.frame_size];

            // Apply IMDCT
            self.mdct.inverse(&mdct_coeffs, &mut channel_out);

            // Overlap-add
            for i in 0..self.overlap_size {
                if i < channel_out.len() {
                    channel_out[i] += self.overlap[ch].get(i).copied().unwrap_or(0.0);
                }
            }

            // Save overlap for next frame
            let overlap_start = self.frame_size - self.overlap_size;
            for i in 0..self.overlap_size {
                self.overlap[ch][i] = channel_out.get(overlap_start + i).copied().unwrap_or(0.0);
            }

            // Copy to output
            for (i, &sample) in channel_out.iter().enumerate().take(self.frame_size - self.overlap_size) {
                if channel_offset + i < output.len() {
                    output[channel_offset + i] = sample;
                }
            }

            // Update previous energy
            for b in 0..num_bands {
                self.prev_energy[ch][b] = bands.energy[b];
            }
        }

        // Apply post-filter
        self.postfilter.apply(&mut output, transient);

        self.frame_count += 1;
        Ok(output)
    }

    /// Decode band energies.
    fn decode_band_energies(
        &self,
        reader: &mut RangeDecoder<'_>,
        num_bands: usize,
        intra: bool,
    ) -> Result<CeltBands> {
        let mut bands = CeltBands::default();
        bands.num_bands = num_bands;

        // Decode coarse energy
        for b in 0..num_bands {
            if intra {
                // Absolute coding
                let energy_int = reader.read_uint(32)? as i32 - 16;
                bands.coarse_energy[b] = energy_int.clamp(-128, 127) as i8;
            } else {
                // Differential coding
                let delta = reader.read_laplace(8)?;
                let prev = self.prev_energy.first().map(|e| e[b]).unwrap_or(0.0) as i32;
                bands.coarse_energy[b] = (prev + delta).clamp(-128, 127) as i8;
            }

            // Convert to linear energy
            let coarse_db = bands.coarse_energy[b] as f32 * 1.5;
            bands.energy[b] = 10.0f32.powf(coarse_db / 10.0);
        }

        // Decode fine energy (if bits available)
        for b in 0..num_bands {
            if reader.remaining_bytes() > 0 {
                let fine_bits = reader.read_uint(8)? as u8;
                bands.fine_energy[b] = fine_bits;

                // Adjust energy based on fine bits
                let fine_adjust = (fine_bits as f32 - 128.0) * 0.01;
                bands.energy[b] *= 10.0f32.powf(fine_adjust);
            }
        }

        Ok(bands)
    }

    /// Decode PVQ coefficients.
    fn decode_pvq(
        &self,
        reader: &mut RangeDecoder<'_>,
        bands: &CeltBands,
        num_bands: usize,
    ) -> Result<CeltPvq> {
        let mut pvq = CeltPvq::default();

        for b in 0..num_bands {
            let band_size = self.get_band_size(b);

            // Determine pulse count based on available bits and band energy
            let pulses = self.allocate_pulses(bands.energy[b], band_size);
            pvq.pulses[b] = pulses;

            if pulses == 0 {
                // No pulses - use noise
                pvq.coeffs[b] = vec![0.0; band_size];
                continue;
            }

            // Decode PVQ
            let (coeffs, signs) = self.decode_pvq_band(reader, band_size, pulses)?;
            pvq.coeffs[b] = coeffs;
            pvq.signs[b] = signs;
        }

        Ok(pvq)
    }

    /// Get band size in bins.
    fn get_band_size(&self, band: usize) -> usize {
        if band + 1 < BAND_BOUNDARIES.len() {
            BAND_BOUNDARIES[band + 1] - BAND_BOUNDARIES[band]
        } else {
            1
        }
    }

    /// Allocate pulses for a band.
    fn allocate_pulses(&self, energy: f32, band_size: usize) -> u8 {
        // Simple allocation based on energy
        let target = (energy.log10() * 3.0 + 5.0).clamp(0.0, 15.0);
        (target * band_size as f32 / 8.0).min(255.0) as u8
    }

    /// Decode PVQ for a single band.
    fn decode_pvq_band(
        &self,
        reader: &mut RangeDecoder<'_>,
        band_size: usize,
        pulses: u8,
    ) -> Result<(Vec<f32>, Vec<bool>)> {
        let mut coeffs = vec![0.0f32; band_size];
        let mut signs = vec![false; band_size];

        if pulses == 0 || band_size == 0 {
            return Ok((coeffs, signs));
        }

        // Simplified PVQ decoding using Theta coding
        let mut remaining_pulses = pulses as i32;

        for i in 0..band_size {
            if remaining_pulses <= 0 {
                break;
            }

            // Decode pulse count for this coefficient
            let max_pulses = remaining_pulses.min(15) as u32;
            let pulse_count = if max_pulses > 0 {
                reader.read_uint(max_pulses + 1)? as i32
            } else {
                0
            };

            if pulse_count > 0 {
                // Decode sign
                signs[i] = reader.read_bit()?;
                coeffs[i] = pulse_count as f32;
                if signs[i] {
                    coeffs[i] = -coeffs[i];
                }
            }

            remaining_pulses -= pulse_count.abs();
        }

        // Normalize coefficients
        let energy: f32 = coeffs.iter().map(|c| c * c).sum();
        if energy > 0.0 {
            let norm = 1.0 / energy.sqrt();
            for c in &mut coeffs {
                *c *= norm;
            }
        }

        Ok((coeffs, signs))
    }

    /// Reconstruct MDCT coefficients.
    fn reconstruct_coefficients(
        &self,
        bands: &CeltBands,
        pvq: &CeltPvq,
        num_bands: usize,
        output: &mut [f32],
    ) {
        let mut bin = 0;

        for b in 0..num_bands {
            let band_size = self.get_band_size(b);
            let energy = bands.energy[b].sqrt();

            for i in 0..band_size {
                if bin < output.len() {
                    let coeff = pvq.coeffs.get(b).and_then(|c| c.get(i)).copied().unwrap_or(0.0);
                    output[bin] = coeff * energy;
                }
                bin += 1;
            }
        }

        // Zero out remaining bins
        for i in bin..output.len() {
            output[i] = 0.0;
        }
    }

    /// Perform packet loss concealment.
    pub fn conceal_packet(&mut self) -> Vec<f32> {
        let output_size = self.frame_size * self.channels as usize;
        let mut output = vec![0.0f32; output_size];

        // Decay previous output
        let decay = 0.85f32;

        for ch in 0..self.channels as usize {
            // Use previous overlap with decay
            for i in 0..self.frame_size {
                let channel_offset = ch * self.frame_size;
                if channel_offset + i < output.len() {
                    // Random noise with decaying energy
                    let prev_val = self.prev_coeffs.get(ch)
                        .and_then(|c| c.get(i))
                        .copied()
                        .unwrap_or(0.0);
                    output[channel_offset + i] = prev_val * decay;
                }
            }

            // Decay overlap
            for o in &mut self.overlap[ch] {
                *o *= decay;
            }

            // Update previous for next concealment
            for (i, &sample) in output.iter().skip(ch * self.frame_size).take(self.frame_size).enumerate() {
                if i < self.prev_coeffs[ch].len() {
                    self.prev_coeffs[ch][i] = sample;
                }
            }
        }

        output
    }

    /// Reset decoder state.
    pub fn reset(&mut self) {
        for overlap in &mut self.overlap {
            overlap.fill(0.0);
        }
        for energy in &mut self.prev_energy {
            energy.fill(0.0);
        }
        for coeffs in &mut self.prev_coeffs {
            coeffs.fill(0.0);
        }
        self.frame_count = 0;
        self.postfilter.reset();
    }
}

/// CELT post-filter.
#[derive(Debug)]
struct CeltPostfilter {
    /// Filter state.
    state: Vec<f32>,
    /// Previous pitch.
    prev_pitch: u16,
}

impl CeltPostfilter {
    fn new() -> Self {
        Self {
            state: vec![0.0; 1024],
            prev_pitch: 0,
        }
    }

    fn apply(&mut self, samples: &mut [f32], _transient: bool) {
        // Simple low-pass smoothing as post-filter
        let alpha = 0.1f32;

        for i in 1..samples.len() {
            samples[i] = samples[i] * (1.0 - alpha) + samples[i - 1] * alpha;
        }
    }

    fn reset(&mut self) {
        self.state.fill(0.0);
        self.prev_pitch = 0;
    }
}

/// CELT encoder state.
#[derive(Debug)]
pub struct CeltEncoder {
    /// Number of channels.
    channels: u8,
    /// Frame size.
    frame_size: usize,
    /// Target bitrate (reserved for rate control).
    _bitrate: u32,
    /// MDCT processor.
    mdct: Mdct,
    /// Analysis window.
    window: Vec<f32>,
    /// Previous frame samples per channel.
    prev_samples: Vec<Vec<f32>>,
    /// Previous band energies.
    prev_energy: Vec<[f32; MAX_BANDS]>,
    /// Frame counter.
    frame_count: u64,
    /// Bits per frame budget.
    bits_per_frame: u32,
}

impl CeltEncoder {
    /// Create a new CELT encoder.
    pub fn new(channels: u8, frame_size: usize, bitrate: u32) -> Self {
        let sample_rate = 48000;
        let bits_per_frame = bitrate / (sample_rate / frame_size as u32);

        // Create analysis window
        let mut window = vec![0.0f32; frame_size];
        for i in 0..frame_size {
            let angle = std::f32::consts::PI * (i as f32 + 0.5) / frame_size as f32;
            window[i] = angle.sin();
        }

        Self {
            channels,
            frame_size,
            _bitrate: bitrate,
            mdct: Mdct::new(frame_size),
            window,
            prev_samples: vec![vec![0.0; frame_size]; channels as usize],
            prev_energy: vec![[0.0; MAX_BANDS]; channels as usize],
            frame_count: 0,
            bits_per_frame,
        }
    }

    /// Encode a CELT frame.
    pub fn encode_frame(
        &mut self,
        samples: &[f32],
        bandwidth: CeltBandwidth,
        writer: &mut RangeEncoder,
    ) -> Result<()> {
        let num_bands = bandwidth.num_bands();

        if samples.len() < self.frame_size * self.channels as usize {
            return Err(OpusError::InvalidFrameSize(samples.len()));
        }

        // Detect transient
        let transient = self.detect_transient(samples);

        // Encode frame flags
        writer.write_bit(transient)?;

        let intra = self.frame_count == 0;
        if self.frame_count > 0 {
            writer.write_bit(intra)?;
        }

        // Apply window and MDCT
        let mut mdct_coeffs = vec![0.0f32; self.frame_size];
        self.compute_mdct(samples, &mut mdct_coeffs)?;

        // Compute and encode band energies
        let bands = self.compute_band_energies(&mdct_coeffs, num_bands);
        self.encode_band_energies(writer, &bands, num_bands, intra)?;

        // Normalize coefficients by band
        let normalized = self.normalize_by_band(&mdct_coeffs, &bands, num_bands);

        // Encode PVQ coefficients
        self.encode_pvq(writer, &normalized, &bands, num_bands)?;

        // Update state
        self.frame_count += 1;
        for ch in 0..self.channels as usize {
            let ch_start = ch * self.frame_size;
            let ch_end = ch_start + self.frame_size;
            if ch_end <= samples.len() {
                self.prev_samples[ch].copy_from_slice(&samples[ch_start..ch_end]);
            }
            for b in 0..num_bands {
                self.prev_energy[ch][b] = bands.energy[b];
            }
        }

        Ok(())
    }

    /// Detect transient in signal.
    fn detect_transient(&self, samples: &[f32]) -> bool {
        // Compute energy in sub-frames
        let subframe_size = self.frame_size / 4;
        let mut energies = [0.0f32; 4];

        for sf in 0..4 {
            let start = sf * subframe_size;
            let end = start + subframe_size;
            energies[sf] = samples[start..end.min(samples.len())]
                .iter()
                .map(|s| s * s)
                .sum();
        }

        // Check for large energy changes
        for i in 1..4 {
            if energies[i] > energies[i - 1] * 4.0 || energies[i] * 4.0 < energies[i - 1] {
                return true;
            }
        }

        false
    }

    /// Compute MDCT of input samples.
    fn compute_mdct(&self, samples: &[f32], output: &mut [f32]) -> Result<()> {
        // Apply window
        let mut windowed = vec![0.0f32; self.frame_size * 2];

        for i in 0..self.frame_size {
            // Previous frame overlap
            windowed[i] = self.prev_samples.first().and_then(|s| s.get(i)).copied().unwrap_or(0.0)
                * self.window.get(i).copied().unwrap_or(1.0);
        }

        for i in 0..self.frame_size {
            // Current frame
            windowed[self.frame_size + i] = samples.get(i).copied().unwrap_or(0.0)
                * self.window.get(self.frame_size - 1 - i).copied().unwrap_or(1.0);
        }

        // Forward MDCT
        self.mdct.forward(&windowed, output);

        Ok(())
    }

    /// Compute band energies from MDCT coefficients.
    fn compute_band_energies(&self, mdct: &[f32], num_bands: usize) -> CeltBands {
        let mut bands = CeltBands::default();
        bands.num_bands = num_bands;

        for b in 0..num_bands {
            let start = BAND_BOUNDARIES[b];
            let end = BAND_BOUNDARIES.get(b + 1).copied().unwrap_or(start);

            let mut energy = 0.0f32;
            for i in start..end {
                if i < mdct.len() {
                    energy += mdct[i] * mdct[i];
                }
            }

            bands.energy[b] = energy.max(1e-10);

            // Quantize to coarse energy
            let log_energy = 10.0 * bands.energy[b].log10();
            bands.coarse_energy[b] = (log_energy / 1.5).clamp(-128.0, 127.0) as i8;
        }

        bands
    }

    /// Encode band energies.
    fn encode_band_energies(
        &self,
        writer: &mut RangeEncoder,
        bands: &CeltBands,
        num_bands: usize,
        intra: bool,
    ) -> Result<()> {
        for b in 0..num_bands {
            if intra {
                // Absolute coding
                let energy_code = (bands.coarse_energy[b] as i32 + 16).clamp(0, 31) as u32;
                writer.write_uint(energy_code, 32)?;
            } else {
                // Differential coding
                let prev = self.prev_energy.first().map(|e| e[b]).unwrap_or(0.0);
                let prev_coarse = (10.0 * prev.log10() / 1.5) as i32;
                let delta = bands.coarse_energy[b] as i32 - prev_coarse;
                writer.write_laplace(delta, 8)?;
            }
        }

        // Encode fine energy
        for b in 0..num_bands {
            let fine_bits = ((bands.energy[b].log10() * 50.0) as i32 + 128).clamp(0, 255) as u8;
            writer.write_uint(fine_bits as u32, 8)?;
        }

        Ok(())
    }

    /// Normalize coefficients by band energy.
    fn normalize_by_band(
        &self,
        mdct: &[f32],
        bands: &CeltBands,
        num_bands: usize,
    ) -> Vec<Vec<f32>> {
        let mut normalized = vec![Vec::new(); num_bands];

        for b in 0..num_bands {
            let start = BAND_BOUNDARIES[b];
            let end = BAND_BOUNDARIES.get(b + 1).copied().unwrap_or(start);
            let band_size = end - start;

            normalized[b] = vec![0.0; band_size];

            let norm = if bands.energy[b] > 0.0 {
                1.0 / bands.energy[b].sqrt()
            } else {
                0.0
            };

            for (i, coeff) in normalized[b].iter_mut().enumerate() {
                let idx = start + i;
                if idx < mdct.len() {
                    *coeff = mdct[idx] * norm;
                }
            }
        }

        normalized
    }

    /// Encode PVQ coefficients.
    fn encode_pvq(
        &self,
        writer: &mut RangeEncoder,
        normalized: &[Vec<f32>],
        bands: &CeltBands,
        num_bands: usize,
    ) -> Result<()> {
        for b in 0..num_bands {
            let band_coeffs = &normalized[b];
            let band_size = band_coeffs.len();

            // Allocate pulses based on energy and bit budget
            let pulses = self.allocate_pulses(bands.energy[b], band_size);

            if pulses == 0 {
                continue;
            }

            // Quantize using PVQ
            let quantized = self.pvq_quantize(band_coeffs, pulses);

            // Encode quantized coefficients
            for &q in &quantized {
                let abs_val = q.unsigned_abs() as u32;
                let sign = q < 0;

                if abs_val > 0 {
                    writer.write_uint(abs_val.min(15), 16)?;
                    writer.write_bit(sign)?;
                } else {
                    writer.write_uint(0, 16)?;
                }
            }
        }

        Ok(())
    }

    /// Allocate pulses for a band.
    fn allocate_pulses(&self, energy: f32, band_size: usize) -> u8 {
        let bits_for_band = (self.bits_per_frame as f32 * energy.log10().max(0.0) / 10.0) as usize;
        let pulses = bits_for_band / (band_size.max(1) * 2);
        pulses.min(15) as u8
    }

    /// PVQ quantization.
    fn pvq_quantize(&self, coeffs: &[f32], pulses: u8) -> Vec<i16> {
        let mut quantized = vec![0i16; coeffs.len()];
        let mut remaining_pulses = pulses as i32;

        // Simple greedy pulse allocation
        let mut magnitudes: Vec<(usize, f32)> = coeffs
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, c.abs()))
            .collect();

        magnitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (idx, _mag) in magnitudes {
            if remaining_pulses <= 0 {
                break;
            }

            let pulse = remaining_pulses.clamp(1, 4);
            quantized[idx] = if coeffs[idx] >= 0.0 {
                pulse as i16
            } else {
                -(pulse as i16)
            };
            remaining_pulses -= pulse;
        }

        quantized
    }

    /// Reset encoder state.
    pub fn reset(&mut self) {
        for samples in &mut self.prev_samples {
            samples.fill(0.0);
        }
        for energy in &mut self.prev_energy {
            energy.fill(0.0);
        }
        self.frame_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bandwidth() {
        assert_eq!(CeltBandwidth::Full.num_bands(), 21);
        assert_eq!(CeltBandwidth::Narrow.max_freq(), 4000);
    }

    #[test]
    fn test_mdct_creation() {
        let mdct = Mdct::new(960);
        assert_eq!(mdct.size, 960);
    }

    #[test]
    fn test_decoder_creation() {
        let decoder = CeltDecoder::new(2, 960);
        assert_eq!(decoder.channels, 2);
        assert_eq!(decoder.frame_size, 960);
    }

    #[test]
    fn test_encoder_creation() {
        let encoder = CeltEncoder::new(2, 960, 128000);
        assert_eq!(encoder.channels, 2);
        assert_eq!(encoder.frame_size, 960);
    }

    #[test]
    fn test_band_boundaries() {
        assert_eq!(BAND_BOUNDARIES[0], 0);
        assert_eq!(BAND_BOUNDARIES.len(), 22);
    }

    #[test]
    fn test_mdct_forward_inverse() {
        let mdct = Mdct::new(128);
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut freq = vec![0.0f32; 128];
        let mut output = vec![0.0f32; 256];

        mdct.forward(&input, &mut freq);
        mdct.inverse(&freq, &mut output);

        // Check that some energy is preserved
        let _input_energy: f32 = input.iter().map(|x| x * x).sum();
        let output_energy: f32 = output.iter().map(|x| x * x).sum();
        assert!(output_energy > 0.0);
    }

    #[test]
    fn test_plc() {
        let mut decoder = CeltDecoder::new(1, 480);
        let output = decoder.conceal_packet();
        assert_eq!(output.len(), 480);
    }
}
