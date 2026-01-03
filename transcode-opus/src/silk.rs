//! SILK mode implementation for Opus.
//!
//! SILK is the speech coding mode of Opus, originally developed by Skype.
//! It uses linear prediction (LP) coding with adaptive codebook for excitation.
//!
//! Key features:
//! - Optimized for speech signals
//! - Variable bitrate from 6-40 kbps
//! - Sample rates: 8, 12, 16, 24 kHz (internal)
//! - Frame sizes: 10, 20, 40, 60 ms
//! - Linear prediction order: 10-16

#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::needless_range_loop)]

use crate::error::{OpusError, Result};
use crate::range_coder::{RangeDecoder, RangeEncoder};

/// SILK internal sample rates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SilkSampleRate {
    /// Narrowband: 8 kHz.
    Nb8000 = 8000,
    /// Medium-band: 12 kHz.
    Mb12000 = 12000,
    /// Wideband: 16 kHz.
    Wb16000 = 16000,
    /// Super-wideband: 24 kHz.
    Swb24000 = 24000,
}

impl SilkSampleRate {
    /// Get sample rate from value.
    pub fn from_hz(hz: u32) -> Option<Self> {
        match hz {
            8000 => Some(Self::Nb8000),
            12000 => Some(Self::Mb12000),
            16000 => Some(Self::Wb16000),
            24000 => Some(Self::Swb24000),
            _ => None,
        }
    }

    /// Get the LP order for this sample rate.
    pub fn lp_order(&self) -> usize {
        match self {
            Self::Nb8000 => 10,
            Self::Mb12000 => 12,
            Self::Wb16000 => 14,
            Self::Swb24000 => 16,
        }
    }

    /// Get the frame size in samples for given milliseconds.
    pub fn frame_samples(&self, ms: u32) -> usize {
        (*self as u32 * ms / 1000) as usize
    }
}

/// SILK frame header information.
#[derive(Debug, Clone)]
pub struct SilkFrameHeader {
    /// Voice Activity Detection flag.
    pub vad_flag: bool,
    /// LBRR (Low Bitrate Redundancy) flag.
    pub lbrr_flag: bool,
    /// Signal type (0 = inactive, 1 = unvoiced, 2 = voiced).
    pub signal_type: u8,
    /// Quantization offset type.
    pub quant_offset_type: u8,
    /// Gain indices.
    pub gain_indices: [u8; 4],
    /// Number of subframes.
    pub num_subframes: u8,
}

impl Default for SilkFrameHeader {
    fn default() -> Self {
        Self {
            vad_flag: false,
            lbrr_flag: false,
            signal_type: 0,
            quant_offset_type: 0,
            gain_indices: [0; 4],
            num_subframes: 4,
        }
    }
}

/// SILK LP (Linear Prediction) coefficients.
#[derive(Debug, Clone)]
pub struct SilkLpc {
    /// Quantized LSF (Line Spectral Frequency) coefficients.
    pub lsf: [i16; 16],
    /// Interpolated LSF for each subframe.
    pub lsf_interp: [[i16; 16]; 4],
    /// LP filter coefficients.
    pub a_q: [[i16; 16]; 4],
    /// LP order.
    pub order: usize,
}

impl Default for SilkLpc {
    fn default() -> Self {
        Self {
            lsf: [0; 16],
            lsf_interp: [[0; 16]; 4],
            a_q: [[0; 16]; 4],
            order: 16,
        }
    }
}

/// SILK excitation parameters.
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct SilkExcitation {
    /// Pitch lags for each subframe.
    pub pitch_lags: [u16; 4],
    /// LTP (Long-Term Prediction) coefficients.
    pub ltp_coefs: [[i8; 5]; 4],
    /// LTP scaling.
    pub ltp_scale: i16,
    /// Excitation signal (quantized).
    pub excitation: Vec<i16>,
}


/// SILK decoder state.
#[derive(Debug)]
pub struct SilkDecoder {
    /// Sample rate.
    sample_rate: SilkSampleRate,
    /// Number of channels (reserved for stereo support).
    _channels: u8,
    /// Previous LP coefficients (for interpolation).
    prev_lpc: SilkLpc,
    /// Previous excitation (for LTP).
    prev_excitation: Vec<i16>,
    /// Previous output (for overlap-add).
    prev_output: Vec<f32>,
    /// LPC synthesis filter state.
    lpc_state: Vec<i32>,
    /// Previous gains.
    prev_gains: [f32; 4],
    /// Frame counter.
    frame_count: u64,
    /// Packet loss flag (for PLC).
    packet_lost: bool,
    /// Number of consecutive losses.
    consecutive_losses: u32,
}

impl SilkDecoder {
    /// Create a new SILK decoder.
    pub fn new(sample_rate: SilkSampleRate, channels: u8) -> Self {
        let frame_size = sample_rate.frame_samples(20);
        let lp_order = sample_rate.lp_order();

        Self {
            sample_rate,
            _channels: channels,
            prev_lpc: SilkLpc::default(),
            prev_excitation: vec![0; frame_size + 128],
            prev_output: vec![0.0; frame_size],
            lpc_state: vec![0; lp_order],
            prev_gains: [1.0; 4],
            frame_count: 0,
            packet_lost: false,
            consecutive_losses: 0,
        }
    }

    /// Decode a SILK frame.
    pub fn decode_frame(
        &mut self,
        reader: &mut RangeDecoder<'_>,
        frame_size_ms: u32,
    ) -> Result<Vec<f32>> {
        let frame_samples = self.sample_rate.frame_samples(frame_size_ms);
        let subframe_samples = frame_samples / 4;
        let num_subframes = (frame_size_ms / 5).max(1) as u8;

        // Decode frame header
        let header = self.decode_frame_header(reader, num_subframes)?;

        // Decode LP coefficients
        let lpc = self.decode_lpc(reader)?;

        // Decode excitation
        let excitation = if header.signal_type == 2 {
            // Voiced: decode pitch and LTP
            self.decode_voiced_excitation(reader, frame_samples, num_subframes)?
        } else {
            // Unvoiced: decode noise excitation
            self.decode_unvoiced_excitation(reader, frame_samples)?
        };

        // Decode gains
        let gains = self.decode_gains(reader, &header, num_subframes)?;

        // Synthesize output
        let mut output = vec![0.0f32; frame_samples];

        for sf in 0..num_subframes as usize {
            let start = sf * subframe_samples;
            let end = start + subframe_samples;

            // Get excitation for this subframe
            let exc_slice = &excitation.excitation[start..end.min(excitation.excitation.len())];

            // Apply gain
            let gain = gains[sf];

            // LPC synthesis filter
            self.lpc_synthesis(
                exc_slice,
                &lpc.a_q[sf],
                lpc.order,
                gain,
                &mut output[start..end],
            );
        }

        // Update state
        self.prev_lpc = lpc;
        self.prev_gains = gains;
        self.prev_output = output.clone();
        self.frame_count += 1;
        self.consecutive_losses = 0;

        Ok(output)
    }

    /// Decode frame header.
    fn decode_frame_header(
        &self,
        reader: &mut RangeDecoder<'_>,
        num_subframes: u8,
    ) -> Result<SilkFrameHeader> {
        let mut header = SilkFrameHeader {
            num_subframes,
            ..Default::default()
        };

        // VAD flag
        header.vad_flag = reader.read_bit()?;

        // LBRR flag (only for certain configurations)
        header.lbrr_flag = reader.read_bit()?;

        // Signal type (2 bits)
        header.signal_type = reader.read_uint(4)? as u8;

        // Quantization offset type
        header.quant_offset_type = reader.read_uint(2)? as u8;

        Ok(header)
    }

    /// Decode LP coefficients.
    fn decode_lpc(&self, reader: &mut RangeDecoder<'_>) -> Result<SilkLpc> {
        let mut lpc = SilkLpc::default();
        lpc.order = self.sample_rate.lp_order();

        // Decode LSF stage 1 (codebook index)
        let stage1_index = reader.read_uint(32)? as usize;

        // Decode LSF stage 2 (residual)
        for i in 0..lpc.order {
            let residual = reader.read_laplace(16)?;
            lpc.lsf[i] = self.get_lsf_codebook_value(stage1_index, i) + (residual as i16);
        }

        // Ensure LSF ordering and stability
        self.stabilize_lsf(&mut lpc.lsf, lpc.order);

        // Interpolate LSF for each subframe
        for sf in 0..4 {
            let interp_factor = (sf as f32 + 1.0) / 4.0;
            for i in 0..lpc.order {
                lpc.lsf_interp[sf][i] = ((1.0 - interp_factor) * self.prev_lpc.lsf[i] as f32
                    + interp_factor * lpc.lsf[i] as f32) as i16;
            }

            // Convert LSF to LP coefficients
            self.lsf_to_lpc(&lpc.lsf_interp[sf], &mut lpc.a_q[sf], lpc.order);
        }

        Ok(lpc)
    }

    /// Get LSF codebook value.
    fn get_lsf_codebook_value(&self, _index: usize, coef_index: usize) -> i16 {
        // Simplified codebook - real implementation uses trained codebooks
        // Values are in Q15 format representing normalized frequencies
        let base_values: [i16; 16] = [
            1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192,
            9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384,
        ];
        base_values[coef_index.min(15)]
    }

    /// Ensure LSF coefficients are ordered and stable.
    fn stabilize_lsf(&self, lsf: &mut [i16; 16], order: usize) {
        const MIN_DELTA: i16 = 100; // Minimum spacing

        // Sort LSFs
        let mut sorted: Vec<i16> = lsf[..order].to_vec();
        sorted.sort();

        // Ensure minimum spacing
        for i in 1..order {
            if sorted[i] - sorted[i - 1] < MIN_DELTA {
                sorted[i] = sorted[i - 1] + MIN_DELTA;
            }
        }

        // Clamp to valid range
        for (i, val) in sorted.iter().enumerate() {
            lsf[i] = (*val).clamp(0, 32767);
        }
    }

    /// Convert LSF to LP coefficients.
    fn lsf_to_lpc(&self, lsf: &[i16; 16], lpc: &mut [i16; 16], order: usize) {
        // Simplified LSF to LPC conversion
        // Uses the standard cos(omega) representation

        let mut p = vec![0.0f64; order / 2 + 1];
        let mut q = vec![0.0f64; order / 2 + 1];

        p[0] = 1.0;
        q[0] = 1.0;

        for i in 0..order {
            let omega = (lsf[i] as f64) * std::f64::consts::PI / 32768.0;
            let cos_omega = omega.cos();

            if i % 2 == 0 {
                // P polynomial
                let half = i / 2;
                for j in (1..=half + 1).rev() {
                    if j <= order / 2 {
                        p[j] = p[j] - 2.0 * cos_omega * p[j - 1] + if j >= 2 { p[j - 2] } else { 0.0 };
                    }
                }
            } else {
                // Q polynomial
                let half = i / 2;
                for j in (1..=half + 1).rev() {
                    if j <= order / 2 {
                        q[j] = q[j] - 2.0 * cos_omega * q[j - 1] + if j >= 2 { q[j - 2] } else { 0.0 };
                    }
                }
            }
        }

        // Compute LP coefficients from P and Q
        for i in 0..order {
            let a = if i < order / 2 {
                0.5 * (p[i + 1] + q[i + 1] + p[i] - q[i])
            } else {
                0.5 * (p[order - i] + q[order - i] - p[order - i - 1] + q[order - i - 1])
            };
            lpc[i] = (a * 32768.0).clamp(-32768.0, 32767.0) as i16;
        }
    }

    /// Decode voiced excitation (pitch and LTP).
    fn decode_voiced_excitation(
        &self,
        reader: &mut RangeDecoder<'_>,
        frame_samples: usize,
        num_subframes: u8,
    ) -> Result<SilkExcitation> {
        let mut exc = SilkExcitation::default();

        // Decode pitch lag for first subframe
        let pitch_lag_base = reader.read_uint(256)? as u16 + 16; // 16-271
        exc.pitch_lags[0] = pitch_lag_base;

        // Decode pitch lag deltas for other subframes
        for sf in 1..num_subframes as usize {
            let delta = reader.read_laplace(8)? as i16;
            exc.pitch_lags[sf] = (pitch_lag_base as i16 + delta).max(16) as u16;
        }

        // Decode LTP coefficients
        for sf in 0..num_subframes as usize {
            let ltp_index = reader.read_uint(32)? as usize;
            exc.ltp_coefs[sf] = self.get_ltp_codebook(ltp_index);
        }

        // Decode LTP scaling
        exc.ltp_scale = reader.read_uint(4)? as i16;

        // Decode innovation (residual)
        exc.excitation = vec![0; frame_samples];
        for i in 0..frame_samples {
            exc.excitation[i] = reader.read_laplace(32)? as i16;
        }

        Ok(exc)
    }

    /// Decode unvoiced excitation (noise).
    fn decode_unvoiced_excitation(
        &self,
        reader: &mut RangeDecoder<'_>,
        frame_samples: usize,
    ) -> Result<SilkExcitation> {
        let mut exc = SilkExcitation::default();

        // For unvoiced, just decode the noise excitation
        exc.excitation = vec![0; frame_samples];
        for i in 0..frame_samples {
            exc.excitation[i] = reader.read_laplace(64)? as i16;
        }

        Ok(exc)
    }

    /// Get LTP codebook entry.
    fn get_ltp_codebook(&self, _index: usize) -> [i8; 5] {
        // Simplified LTP codebook
        // Real implementation has trained codebooks
        [0, 32, 64, 32, 0]
    }

    /// Decode gain values.
    fn decode_gains(
        &self,
        reader: &mut RangeDecoder<'_>,
        _header: &SilkFrameHeader,
        num_subframes: u8,
    ) -> Result<[f32; 4]> {
        let mut gains = [0.0f32; 4];

        // Decode first gain (absolute)
        let gain_index = reader.read_uint(64)? as usize;
        gains[0] = self.gain_index_to_linear(gain_index);

        // Decode delta gains
        for sf in 1..num_subframes as usize {
            let delta_index = reader.read_laplace(8)?;
            let new_index = ((gain_index as i32 + delta_index).max(0) as usize).min(63);
            gains[sf] = self.gain_index_to_linear(new_index);
        }

        Ok(gains)
    }

    /// Convert gain index to linear gain.
    fn gain_index_to_linear(&self, index: usize) -> f32 {
        // Gain quantization table (dB scale)
        let db = -40.0 + (index as f32) * 1.5;
        10.0f32.powf(db / 20.0)
    }

    /// LPC synthesis filter.
    fn lpc_synthesis(
        &mut self,
        excitation: &[i16],
        lpc: &[i16; 16],
        order: usize,
        gain: f32,
        output: &mut [f32],
    ) {
        for (i, out_sample) in output.iter_mut().enumerate() {
            // Accumulate LPC prediction
            let mut prediction: i64 = 0;

            for j in 0..order {
                let state_idx = (self.lpc_state.len() + i).wrapping_sub(j + 1);
                if state_idx < self.lpc_state.len() {
                    prediction += (lpc[j] as i64) * (self.lpc_state[state_idx] as i64);
                }
            }

            // Scale prediction (Q15 arithmetic)
            prediction >>= 15;

            // Add excitation with gain
            let exc = if i < excitation.len() {
                (excitation[i] as f32) * gain
            } else {
                0.0
            };

            let sample = exc + (prediction as f32);
            *out_sample = sample;

            // Update filter state
            if i < self.lpc_state.len() {
                self.lpc_state[i] = sample as i32;
            }
        }

        // Shift state for next frame
        let frame_samples = output.len();
        let state_len = self.lpc_state.len();
        if frame_samples < state_len {
            for i in 0..state_len - frame_samples {
                self.lpc_state[i] = self.lpc_state[i + frame_samples];
            }
            for (i, &out) in output.iter().enumerate() {
                if state_len - frame_samples + i < state_len {
                    self.lpc_state[state_len - frame_samples + i] = out as i32;
                }
            }
        }
    }

    /// Perform packet loss concealment.
    pub fn conceal_packet(&mut self, frame_size_ms: u32) -> Vec<f32> {
        let frame_samples = self.sample_rate.frame_samples(frame_size_ms);

        self.consecutive_losses += 1;
        self.packet_lost = true;

        // Decay factor based on consecutive losses
        let decay = 0.9f32.powi(self.consecutive_losses as i32);

        // Generate concealment output
        let mut output = vec![0.0f32; frame_samples];

        // Use previous output with decay
        for i in 0..frame_samples {
            if i < self.prev_output.len() {
                output[i] = self.prev_output[i] * decay;
            }
        }

        // Apply random noise for unvoiced segments
        if self.consecutive_losses > 2 {
            for sample in &mut output {
                let noise = (rand_simple() * 2.0 - 1.0) * 0.001 * decay;
                *sample += noise;
            }
        }

        self.prev_output = output.clone();
        output
    }

    /// Reset decoder state.
    pub fn reset(&mut self) {
        self.prev_lpc = SilkLpc::default();
        self.prev_excitation.fill(0);
        self.prev_output.fill(0.0);
        self.lpc_state.fill(0);
        self.prev_gains = [1.0; 4];
        self.frame_count = 0;
        self.packet_lost = false;
        self.consecutive_losses = 0;
    }
}

/// Simple pseudo-random number generator.
fn rand_simple() -> f32 {
    use std::cell::Cell;

    thread_local! {
        static SEED: Cell<u32> = const { Cell::new(12345) };
    }

    SEED.with(|seed| {
        let mut s = seed.get();
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        seed.set(s);
        (s >> 16) as f32 / 32768.0
    })
}

/// SILK encoder state.
#[derive(Debug)]
pub struct SilkEncoder {
    /// Sample rate.
    sample_rate: SilkSampleRate,
    /// Number of channels (reserved for stereo support).
    _channels: u8,
    /// Target bitrate (reserved for rate control).
    _bitrate: u32,
    /// Previous LP coefficients.
    prev_lpc: SilkLpc,
    /// Previous samples (for analysis).
    prev_samples: Vec<f32>,
    /// LPC analysis order.
    lp_order: usize,
    /// Frame counter.
    frame_count: u64,
    /// Previous pitch lag.
    prev_pitch_lag: u16,
}

impl SilkEncoder {
    /// Create a new SILK encoder.
    pub fn new(sample_rate: SilkSampleRate, channels: u8, bitrate: u32) -> Self {
        let frame_size = sample_rate.frame_samples(20);
        let lp_order = sample_rate.lp_order();

        Self {
            sample_rate,
            _channels: channels,
            _bitrate: bitrate,
            prev_lpc: SilkLpc::default(),
            prev_samples: vec![0.0; frame_size * 2],
            lp_order,
            frame_count: 0,
            prev_pitch_lag: 80,
        }
    }

    /// Encode a SILK frame.
    pub fn encode_frame(
        &mut self,
        samples: &[f32],
        frame_size_ms: u32,
        writer: &mut RangeEncoder,
    ) -> Result<()> {
        let frame_samples = self.sample_rate.frame_samples(frame_size_ms);
        let subframe_samples = frame_samples / 4;
        let num_subframes = (frame_size_ms / 5).max(1) as u8;

        if samples.len() < frame_samples {
            return Err(OpusError::InvalidFrameSize(samples.len()));
        }

        // Analyze signal type (voiced/unvoiced)
        let signal_type = self.analyze_signal_type(&samples[..frame_samples]);

        // Encode frame header
        self.encode_frame_header(writer, signal_type, num_subframes)?;

        // LPC analysis
        let lpc = self.analyze_lpc(&samples[..frame_samples])?;

        // Encode LP coefficients
        self.encode_lpc(writer, &lpc)?;

        // Calculate residual/excitation
        let excitation = self.calculate_excitation(&samples[..frame_samples], &lpc);

        if signal_type == 2 {
            // Voiced: encode pitch and LTP
            self.encode_voiced_excitation(writer, &excitation, num_subframes)?;
        } else {
            // Unvoiced: encode noise excitation
            self.encode_unvoiced_excitation(writer, &excitation)?;
        }

        // Encode gains
        let gains = self.analyze_gains(&samples[..frame_samples], subframe_samples);
        self.encode_gains(writer, &gains, num_subframes)?;

        // Update state
        self.prev_lpc = lpc;
        self.prev_samples[..frame_samples].copy_from_slice(&samples[..frame_samples]);
        self.frame_count += 1;

        Ok(())
    }

    /// Analyze signal type.
    fn analyze_signal_type(&self, samples: &[f32]) -> u8 {
        // Simple pitch detection to determine voiced/unvoiced
        let energy: f32 = samples.iter().map(|s| s * s).sum();
        let avg_energy = energy / samples.len() as f32;

        if avg_energy < 0.0001 {
            0 // Inactive
        } else {
            // Compute zero crossing rate
            let mut crossings = 0;
            for i in 1..samples.len() {
                if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                    crossings += 1;
                }
            }

            let zcr = crossings as f32 / samples.len() as f32;

            if zcr > 0.2 {
                1 // Unvoiced (high zero crossing rate)
            } else {
                2 // Voiced
            }
        }
    }

    /// Encode frame header.
    fn encode_frame_header(
        &self,
        writer: &mut RangeEncoder,
        signal_type: u8,
        _num_subframes: u8,
    ) -> Result<()> {
        // VAD flag (always on for now)
        writer.write_bit(true)?;

        // LBRR flag
        writer.write_bit(false)?;

        // Signal type
        writer.write_uint(signal_type as u32, 4)?;

        // Quantization offset type
        writer.write_uint(1, 2)?;

        Ok(())
    }

    /// Analyze LPC coefficients.
    fn analyze_lpc(&self, samples: &[f32]) -> Result<SilkLpc> {
        let mut lpc = SilkLpc::default();
        lpc.order = self.lp_order;

        // Compute autocorrelation
        let mut r = vec![0.0f64; self.lp_order + 1];
        for lag in 0..=self.lp_order {
            for i in lag..samples.len() {
                r[lag] += (samples[i] as f64) * (samples[i - lag] as f64);
            }
        }

        // Levinson-Durbin recursion
        let mut a = vec![0.0f64; self.lp_order];
        let mut e = r[0];

        if e > 0.0 {
            for i in 0..self.lp_order {
                let mut lambda = r[i + 1];
                for j in 0..i {
                    lambda -= a[j] * r[i - j];
                }

                if e.abs() < 1e-10 {
                    break;
                }

                let k = lambda / e;
                a[i] = k;

                // Update coefficients
                let a_prev = a.clone();
                for j in 0..i {
                    a[j] = a_prev[j] - k * a_prev[i - 1 - j];
                }

                e *= 1.0 - k * k;
            }
        }

        // Convert to LP coefficients and LSF
        for (i, coef) in a.iter().enumerate() {
            let lpc_val = (*coef * 32768.0).clamp(-32768.0, 32767.0) as i16;
            lpc.a_q[0][i] = lpc_val;
            lpc.a_q[1][i] = lpc_val;
            lpc.a_q[2][i] = lpc_val;
            lpc.a_q[3][i] = lpc_val;
        }

        // Convert LP to LSF for quantization
        self.lpc_to_lsf(&lpc.a_q[0], &mut lpc.lsf, self.lp_order);

        Ok(lpc)
    }

    /// Convert LP coefficients to LSF.
    fn lpc_to_lsf(&self, lpc: &[i16; 16], lsf: &mut [i16; 16], order: usize) {
        // Simplified LP to LSF conversion using root finding
        // Real implementation uses Chebyshev polynomials

        for i in 0..order {
            // Approximate LSF positions
            let base_freq = (i + 1) as f64 * std::f64::consts::PI / (order + 1) as f64;

            // Adjust based on LP coefficient
            let adjust = if i < order {
                (lpc[i] as f64) / 32768.0 * 0.1
            } else {
                0.0
            };

            let omega = (base_freq + adjust).clamp(0.01, std::f64::consts::PI - 0.01);
            lsf[i] = ((omega / std::f64::consts::PI) * 32768.0) as i16;
        }
    }

    /// Encode LP coefficients.
    fn encode_lpc(&self, writer: &mut RangeEncoder, lpc: &SilkLpc) -> Result<()> {
        // Encode LSF stage 1 (codebook index)
        let stage1_index = self.find_lsf_codebook_index(&lpc.lsf, lpc.order);
        writer.write_uint(stage1_index, 32)?;

        // Encode LSF stage 2 (residuals)
        for i in 0..lpc.order {
            let codebook_val = self.get_lsf_codebook_value(stage1_index as usize, i);
            let residual = lpc.lsf[i] - codebook_val;
            writer.write_laplace(residual as i32, 16)?;
        }

        Ok(())
    }

    /// Find best LSF codebook index.
    fn find_lsf_codebook_index(&self, _lsf: &[i16; 16], _order: usize) -> u32 {
        // Simplified: return index 0
        // Real implementation searches codebook for best match
        0
    }

    /// Get LSF codebook value (same as decoder).
    fn get_lsf_codebook_value(&self, _index: usize, coef_index: usize) -> i16 {
        let base_values: [i16; 16] = [
            1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192,
            9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384,
        ];
        base_values[coef_index.min(15)]
    }

    /// Calculate excitation/residual.
    fn calculate_excitation(&self, samples: &[f32], lpc: &SilkLpc) -> SilkExcitation {
        let mut exc = SilkExcitation::default();
        exc.excitation = vec![0; samples.len()];

        // Calculate LPC residual
        for i in 0..samples.len() {
            let mut prediction = 0.0f64;

            for j in 0..lpc.order {
                if i > j {
                    prediction += (lpc.a_q[0][j] as f64 / 32768.0) * (samples[i - j - 1] as f64);
                }
            }

            let residual = samples[i] as f64 - prediction;
            exc.excitation[i] = (residual * 256.0).clamp(-32768.0, 32767.0) as i16;
        }

        exc
    }

    /// Encode voiced excitation.
    fn encode_voiced_excitation(
        &mut self,
        writer: &mut RangeEncoder,
        exc: &SilkExcitation,
        num_subframes: u8,
    ) -> Result<()> {
        // Encode pitch lag
        let pitch_lag = self.estimate_pitch_lag(&exc.excitation);
        writer.write_uint((pitch_lag - 16) as u32, 256)?;

        // Encode pitch lag deltas
        for _ in 1..num_subframes {
            writer.write_laplace(0, 8)?;
        }

        // Encode LTP coefficients
        for _ in 0..num_subframes as usize {
            writer.write_uint(0, 32)?; // Simplified: use index 0
        }

        // Encode LTP scale
        writer.write_uint(8, 4)?;

        // Encode innovation
        for &e in &exc.excitation {
            writer.write_laplace(e as i32, 32)?;
        }

        self.prev_pitch_lag = pitch_lag;
        Ok(())
    }

    /// Estimate pitch lag.
    fn estimate_pitch_lag(&self, excitation: &[i16]) -> u16 {
        // Simple autocorrelation-based pitch detection
        let mut best_lag = 80u16;
        let mut best_corr = 0.0f64;

        for lag in 16..256 {
            let mut corr = 0.0f64;
            let mut energy = 0.0f64;

            for i in lag..excitation.len() {
                corr += (excitation[i] as f64) * (excitation[i - lag] as f64);
                energy += (excitation[i - lag] as f64).powi(2);
            }

            if energy > 0.0 {
                let normalized = corr / energy.sqrt();
                if normalized > best_corr {
                    best_corr = normalized;
                    best_lag = lag as u16;
                }
            }
        }

        best_lag
    }

    /// Encode unvoiced excitation.
    fn encode_unvoiced_excitation(
        &self,
        writer: &mut RangeEncoder,
        exc: &SilkExcitation,
    ) -> Result<()> {
        for &e in &exc.excitation {
            writer.write_laplace(e as i32, 64)?;
        }
        Ok(())
    }

    /// Analyze gains.
    fn analyze_gains(&self, samples: &[f32], subframe_samples: usize) -> [f32; 4] {
        let mut gains = [0.0f32; 4];

        for sf in 0..4 {
            let start = sf * subframe_samples;
            let end = (start + subframe_samples).min(samples.len());

            if start < end {
                let energy: f32 = samples[start..end].iter().map(|s| s * s).sum();
                let rms = (energy / (end - start) as f32).sqrt();
                gains[sf] = rms.max(0.001);
            } else {
                gains[sf] = 0.001;
            }
        }

        gains
    }

    /// Encode gains.
    fn encode_gains(
        &self,
        writer: &mut RangeEncoder,
        gains: &[f32; 4],
        num_subframes: u8,
    ) -> Result<()> {
        // Convert first gain to index
        let gain_index = self.linear_to_gain_index(gains[0]);
        writer.write_uint(gain_index as u32, 64)?;

        // Encode delta gains
        for sf in 1..num_subframes as usize {
            let this_index = self.linear_to_gain_index(gains[sf]);
            let delta = this_index as i32 - gain_index as i32;
            writer.write_laplace(delta, 8)?;
        }

        Ok(())
    }

    /// Convert linear gain to index.
    fn linear_to_gain_index(&self, gain: f32) -> usize {
        let db = 20.0 * gain.max(0.0001).log10();
        ((db + 40.0) / 1.5).clamp(0.0, 63.0) as usize
    }

    /// Reset encoder state.
    pub fn reset(&mut self) {
        self.prev_lpc = SilkLpc::default();
        self.prev_samples.fill(0.0);
        self.frame_count = 0;
        self.prev_pitch_lag = 80;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silk_sample_rates() {
        assert_eq!(SilkSampleRate::from_hz(8000), Some(SilkSampleRate::Nb8000));
        assert_eq!(SilkSampleRate::from_hz(16000), Some(SilkSampleRate::Wb16000));
        assert_eq!(SilkSampleRate::from_hz(44100), None);
    }

    #[test]
    fn test_frame_samples() {
        let rate = SilkSampleRate::Wb16000;
        assert_eq!(rate.frame_samples(10), 160);
        assert_eq!(rate.frame_samples(20), 320);
    }

    #[test]
    fn test_decoder_creation() {
        let decoder = SilkDecoder::new(SilkSampleRate::Wb16000, 1);
        assert_eq!(decoder.sample_rate, SilkSampleRate::Wb16000);
    }

    #[test]
    fn test_encoder_creation() {
        let encoder = SilkEncoder::new(SilkSampleRate::Wb16000, 1, 20000);
        assert_eq!(encoder.sample_rate, SilkSampleRate::Wb16000);
        assert_eq!(encoder.lp_order, 14);
    }

    #[test]
    fn test_plc() {
        let mut decoder = SilkDecoder::new(SilkSampleRate::Wb16000, 1);
        let output = decoder.conceal_packet(20);
        assert_eq!(output.len(), 320);
    }
}
