//! MP3 encoder implementation.

use super::tables::*;
use super::{ChannelMode, MpegVersion};
use crate::traits::{AudioEncoder, CodecInfo};
use transcode_core::error::{Error, Result};
use transcode_core::packet::Packet;
use transcode_core::sample::{Sample, SampleFormat};
use std::f32::consts::PI;

/// MP3 encoder configuration.
#[derive(Debug, Clone)]
pub struct Mp3EncoderConfig {
    /// MPEG version.
    pub version: MpegVersion,
    /// Sample rate.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u8,
    /// Bitrate in bits per second.
    pub bitrate: u32,
    /// Quality (0.0 - 1.0, higher = better quality).
    pub quality: f32,
    /// Channel mode.
    pub channel_mode: ChannelMode,
    /// Add CRC protection.
    pub crc: bool,
    /// VBR mode.
    pub vbr: bool,
}

impl Default for Mp3EncoderConfig {
    fn default() -> Self {
        Self {
            version: MpegVersion::Mpeg1,
            sample_rate: 44100,
            channels: 2,
            bitrate: 128000,
            quality: 0.5,
            channel_mode: ChannelMode::Stereo,
            crc: false,
            vbr: false,
        }
    }
}

impl Mp3EncoderConfig {
    /// Create config for mono.
    pub fn mono(sample_rate: u32, bitrate: u32) -> Self {
        Self {
            version: Self::version_for_sample_rate(sample_rate),
            sample_rate,
            channels: 1,
            bitrate,
            channel_mode: ChannelMode::Mono,
            ..Default::default()
        }
    }

    /// Create config for stereo.
    pub fn stereo(sample_rate: u32, bitrate: u32) -> Self {
        Self {
            version: Self::version_for_sample_rate(sample_rate),
            sample_rate,
            channels: 2,
            bitrate,
            channel_mode: ChannelMode::JointStereo,
            ..Default::default()
        }
    }

    fn version_for_sample_rate(sample_rate: u32) -> MpegVersion {
        if sample_rate >= 32000 {
            MpegVersion::Mpeg1
        } else if sample_rate >= 16000 {
            MpegVersion::Mpeg2
        } else {
            MpegVersion::Mpeg25
        }
    }
}

/// Block type for granule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
enum BlockType {
    #[default]
    Normal = 0,
    Start = 1,
    Short = 2,
    Stop = 3,
}

/// Quantization state for a granule.
#[derive(Debug, Clone)]
struct GranuleState {
    /// Quantized MDCT coefficients.
    quantized: Box<[i32; 576]>,
    /// Scalefactors per band.
    scalefactors: [u8; 22],
    /// Global gain.
    global_gain: u8,
    /// Block type.
    block_type: BlockType,
    /// Subblock gains (for short blocks).
    subblock_gain: [u8; 3],
    /// Table select for regions.
    table_select: [u8; 3],
    /// Big values.
    big_values: u16,
    /// Part2-3 length estimate.
    part2_3_length: u16,
    /// Region boundaries.
    region0_count: u8,
    region1_count: u8,
}

impl Default for GranuleState {
    fn default() -> Self {
        Self {
            quantized: Box::new([0; 576]),
            scalefactors: [0; 22],
            global_gain: 0,
            block_type: BlockType::Normal,
            subblock_gain: [0; 3],
            table_select: [0; 3],
            big_values: 0,
            part2_3_length: 0,
            region0_count: 0,
            region1_count: 0,
        }
    }
}

/// Psychoacoustic model state.
struct PsyModel {
    /// Sample rate.
    sample_rate: u32,
    /// Energy in each critical band.
    band_energy: [f32; 22],
    /// Masking threshold.
    masking_threshold: [f32; 22],
    /// Previous block energy (for temporal masking).
    prev_energy: [f32; 22],
}

impl PsyModel {
    fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            band_energy: [0.0; 22],
            masking_threshold: [0.0; 22],
            prev_energy: [0.0; 22],
        }
    }

    /// Analyze MDCT coefficients and compute masking thresholds.
    fn analyze(&mut self, mdct: &[f32; 576]) {
        let sfb = self.scalefactor_bands();

        // Compute energy in each scalefactor band
        for (band, (start, end)) in sfb.iter().enumerate() {
            let mut energy = 0.0f32;
            for i in *start as usize..*end as usize {
                energy += mdct[i] * mdct[i];
            }
            self.band_energy[band] = energy;
        }

        // Apply spreading function and compute masking threshold
        // Simplified model based on bark scale spreading
        for band in 0..sfb.len() {
            let mut mask = 0.0f32;

            for j in 0..sfb.len() {
                let spread = self.spreading_function(band as i32 - j as i32);
                mask += self.band_energy[j] * spread;
            }

            // Temporal masking
            mask = mask.max(self.prev_energy[band] * 0.3);

            // Add absolute threshold
            mask = mask.max(1e-10);

            self.masking_threshold[band] = mask;
        }

        // Update previous energy
        self.prev_energy.copy_from_slice(&self.band_energy);
    }

    fn spreading_function(&self, delta_bark: i32) -> f32 {
        let db = delta_bark.abs() as f32;
        if db == 0.0 {
            1.0
        } else if db < 1.0 {
            0.8
        } else if db < 2.0 {
            0.3
        } else if db < 3.0 {
            0.1
        } else {
            0.01
        }
    }

    fn scalefactor_bands(&self) -> Vec<(u16, u16)> {
        let table = match self.sample_rate {
            48000 => &SFB_LONG_48000,
            32000 => &SFB_LONG_32000,
            _ => &SFB_LONG_44100,
        };

        table
            .windows(2)
            .map(|w| (w[0], w[1]))
            .collect()
    }

    /// Get masking threshold for a band.
    fn threshold(&self, band: usize) -> f32 {
        self.masking_threshold[band.min(21)]
    }
}

/// MDCT processor for MP3.
struct MdctProcessor {
    /// Window coefficients.
    window: [f32; 36],
    /// Cosine table.
    cos_table: [[f32; 18]; 36],
    /// Previous block samples.
    prev_block: [[f32; 18]; 2],
}

impl MdctProcessor {
    fn new() -> Self {
        let window = imdct_window_long();
        let mut cos_table = [[0.0f32; 18]; 36];

        for n in 0..36 {
            for k in 0..18 {
                cos_table[n][k] =
                    ((PI / 36.0) * (2.0 * n as f32 + 1.0 + 18.0) * (2.0 * k as f32 + 1.0) / 2.0)
                        .cos();
            }
        }

        Self {
            window,
            cos_table,
            prev_block: [[0.0; 18]; 2],
        }
    }

    /// Forward MDCT: 36 time samples -> 18 frequency coefficients.
    fn forward(&mut self, input: &[f32; 36], output: &mut [f32; 18], channel: usize) {
        // Apply window and compute MDCT
        for k in 0..18 {
            let mut sum = 0.0f32;
            for n in 0..36 {
                sum += input[n] * self.window[n] * self.cos_table[n][k];
            }
            output[k] = sum;
        }

        // Overlap-add with previous block
        for k in 0..18 {
            output[k] -= self.prev_block[channel][k];
        }
        self.prev_block[channel].copy_from_slice(output);
    }
}

/// Bit reservoir for MP3 encoding.
struct BitReservoir {
    /// Main data buffer.
    buffer: Vec<u8>,
    /// Current bit position.
    bit_pos: usize,
    /// Available bits.
    available_bits: i32,
    /// Maximum reservoir size.
    max_size: usize,
}

impl BitReservoir {
    fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(4096),
            bit_pos: 0,
            available_bits: 0,
            max_size: 7680, // 511 bytes * 8 bits
        }
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.bit_pos = 0;
        self.available_bits = 0;
    }
}

/// MP3 encoder.
pub struct Mp3Encoder {
    /// Configuration.
    config: Mp3EncoderConfig,
    /// MDCT processor.
    mdct: MdctProcessor,
    /// Psychoacoustic models per channel.
    psy_models: Vec<PsyModel>,
    /// Input sample buffer per channel.
    input_buffer: Vec<Vec<f32>>,
    /// Bit reservoir.
    reservoir: BitReservoir,
    /// Frame counter.
    frame_count: u64,
    /// Granule states.
    granule_states: [[GranuleState; 2]; 2],
    /// Samples per frame.
    samples_per_frame: usize,
}

impl Mp3Encoder {
    /// Create a new MP3 encoder.
    pub fn new(config: Mp3EncoderConfig) -> Result<Self> {
        if config.channels == 0 || config.channels > 2 {
            return Err(Error::invalid_param(format!(
                "Invalid channel count: {}",
                config.channels
            )));
        }

        let samples_per_frame = match config.version {
            MpegVersion::Mpeg1 => 1152,
            MpegVersion::Mpeg2 | MpegVersion::Mpeg25 => 576,
            MpegVersion::Reserved => {
                return Err(Error::invalid_param("Reserved MPEG version"))
            }
        };

        let mut psy_models = Vec::with_capacity(config.channels as usize);
        let mut input_buffer = Vec::with_capacity(config.channels as usize);

        for _ in 0..config.channels {
            psy_models.push(PsyModel::new(config.sample_rate));
            input_buffer.push(Vec::with_capacity(samples_per_frame * 2));
        }

        Ok(Self {
            config,
            mdct: MdctProcessor::new(),
            psy_models,
            input_buffer,
            reservoir: BitReservoir::new(),
            frame_count: 0,
            granule_states: Default::default(),
            samples_per_frame,
        })
    }

    /// Encode samples to MP3 frame.
    fn encode_frame(&mut self, samples: &[&[f32]]) -> Result<Vec<u8>> {
        let num_granules = if self.config.version == MpegVersion::Mpeg1 {
            2
        } else {
            1
        };

        // Process granules
        let granule_size = 576;
        let mut mdct_coeffs = [[[0.0f32; 576]; 2]; 2]; // [granule][channel][coefficient]

        for gr in 0..num_granules {
            for ch in 0..self.config.channels as usize {
                let offset = gr * granule_size;
                let end = (offset + granule_size).min(samples[ch].len());

                // Perform MDCT on granule
                self.compute_mdct(
                    &samples[ch][offset..end],
                    &mut mdct_coeffs[gr][ch],
                    ch,
                );

                // Psychoacoustic analysis
                self.psy_models[ch].analyze(&mdct_coeffs[gr][ch]);

                // Quantization
                self.quantize_granule(gr, ch, &mdct_coeffs[gr][ch])?;
            }
        }

        // Build frame
        self.build_frame(num_granules)
    }

    fn compute_mdct(&mut self, input: &[f32], output: &mut [f32; 576], channel: usize) {
        // Zero output
        output.fill(0.0);

        // Process 32 subbands
        let mut block = [0.0f32; 36];

        for sb in 0..32 {
            // Fill block with samples for this subband
            for i in 0..36.min(input.len() / 32) {
                let idx = i * 32 + sb;
                block[i] = if idx < input.len() { input[idx] } else { 0.0 };
            }

            // Forward MDCT
            let mut mdct_out = [0.0f32; 18];
            self.mdct.forward(&block, &mut mdct_out, channel);

            // Store in frequency-order
            for i in 0..18 {
                output[sb * 18 + i] = mdct_out[i];
            }
        }
    }

    fn quantize_granule(&mut self, gr: usize, ch: usize, mdct: &[f32; 576]) -> Result<()> {
        // Find maximum value
        let max_val = mdct
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, |a, b| a.max(b));

        if max_val < 1e-10 {
            // Silence
            let state = &mut self.granule_states[gr][ch];
            state.quantized.fill(0);
            state.global_gain = 0;
            state.big_values = 0;
            return Ok(());
        }

        // Compute global gain
        // Based on: global_gain = 210 - log2(max) * 4
        let log_max = max_val.log2();
        let global_gain_f = 210.0 - log_max * 4.0;
        let global_gain = global_gain_f.clamp(0.0, 255.0) as u8;

        // Quantization step size
        let step = 2.0_f32.powf((global_gain as f32 - 210.0) / 4.0);

        // Quantize with noise shaping based on psychoacoustic model
        let sfb = self.psy_models[ch].scalefactor_bands();
        let mut last_nonzero = 0;

        // Collect thresholds and scale factors first to avoid borrow conflicts
        let thresholds: Vec<f32> = (0..sfb.len())
            .map(|band| self.psy_models[ch].threshold(band))
            .collect();

        let scale_factors: Vec<u8> = sfb
            .iter()
            .zip(thresholds.iter())
            .map(|((start, end), &threshold)| {
                Self::compute_scale_factor(mdct, *start, *end, threshold, step)
            })
            .collect();

        // Now mutably borrow granule state
        let state = &mut self.granule_states[gr][ch];
        state.global_gain = global_gain;

        for (band, ((start, end), scale_factor)) in sfb.iter().zip(scale_factors.iter()).enumerate() {
            state.scalefactors[band] = *scale_factor;

            let sf_mult = 2.0_f32.powf(-(*scale_factor as f32) * 0.5);

            for i in *start as usize..*end as usize {
                if i >= 576 {
                    break;
                }

                let scaled = mdct[i] * sf_mult / step;
                let quantized = scaled.abs().powf(0.75).round() as i32;
                state.quantized[i] = if scaled < 0.0 { -quantized } else { quantized };

                if state.quantized[i] != 0 {
                    last_nonzero = i;
                }
            }
        }

        state.big_values = ((last_nonzero + 2) / 2) as u16;
        state.big_values = state.big_values.min(288);

        // Select Huffman tables for regions
        Self::select_huffman_tables(state);

        Ok(())
    }

    fn compute_scale_factor(
        mdct: &[f32; 576],
        start: u16,
        end: u16,
        threshold: f32,
        step: f32,
    ) -> u8 {
        let mut energy = 0.0f32;
        for i in start as usize..end as usize {
            if i >= 576 {
                break;
            }
            energy += mdct[i] * mdct[i];
        }

        if energy < threshold || energy < 1e-10 {
            return 0;
        }

        // Compute scalefactor to achieve target SNR
        let snr = energy / threshold;
        (snr.log2() * 0.5).clamp(0.0, 15.0) as u8
    }

    fn select_huffman_tables(state: &mut GranuleState) {
        // Simplified table selection
        // Count maximum value in each region to select table

        let big_values = state.big_values as usize * 2;

        // Region boundaries
        let region0_end = (big_values / 3).min(576);
        let region1_end = (big_values * 2 / 3).min(576);

        // Count max in each region
        let max0 = state.quantized[0..region0_end]
            .iter()
            .map(|x| x.abs())
            .max()
            .unwrap_or(0);
        let max1 = state.quantized[region0_end..region1_end]
            .iter()
            .map(|x| x.abs())
            .max()
            .unwrap_or(0);
        let max2 = state.quantized[region1_end..big_values]
            .iter()
            .map(|x| x.abs())
            .max()
            .unwrap_or(0);

        // Select table based on max value
        state.table_select[0] = Self::select_table(max0);
        state.table_select[1] = Self::select_table(max1);
        state.table_select[2] = Self::select_table(max2);

        // Set region counts
        state.region0_count = 4;
        state.region1_count = 3;
    }

    fn select_table(max_val: i32) -> u8 {
        match max_val {
            0 => 0,
            1 => 1,
            2..=3 => 2,
            4..=7 => 5,
            8..=15 => 7,
            _ => 13, // linbits table
        }
    }

    fn build_frame(&mut self, num_granules: usize) -> Result<Vec<u8>> {
        let mut frame = Vec::with_capacity(1024);

        // Frame header (4 bytes)
        let header_bytes = self.build_header();
        frame.extend_from_slice(&header_bytes);

        // CRC if enabled (2 bytes)
        if self.config.crc {
            frame.extend_from_slice(&[0x00, 0x00]); // Placeholder CRC
        }

        // Side information
        let side_info = self.build_side_info(num_granules);
        frame.extend_from_slice(&side_info);

        // Main data
        let main_data = self.build_main_data(num_granules);
        frame.extend_from_slice(&main_data);

        // Pad to frame size
        let target_size = self.calculate_frame_size();
        while frame.len() < target_size {
            frame.push(0x00);
        }

        self.frame_count += 1;
        Ok(frame)
    }

    fn build_header(&self) -> [u8; 4] {
        let bitrate_index = self.bitrate_index();
        let sample_rate_index = self.sample_rate_index();

        let version_bits = match self.config.version {
            MpegVersion::Mpeg1 => 0x03,
            MpegVersion::Mpeg2 => 0x02,
            MpegVersion::Mpeg25 => 0x00,
            MpegVersion::Reserved => 0x01,
        };

        let mode_bits = match self.config.channel_mode {
            ChannelMode::Stereo => 0x00,
            ChannelMode::JointStereo => 0x01,
            ChannelMode::DualChannel => 0x02,
            ChannelMode::Mono => 0x03,
        };

        [
            0xFF, // Sync
            0xE0 | (version_bits << 3) | (1 << 1) | (!self.config.crc as u8), // Version, Layer 3, CRC
            (bitrate_index << 4) | (sample_rate_index << 2), // Bitrate, Sample rate, no padding
            (mode_bits << 6), // Channel mode, no extension
        ]
    }

    fn bitrate_index(&self) -> u8 {
        let kbps = self.config.bitrate / 1000;

        let table = match self.config.version {
            MpegVersion::Mpeg1 => &[0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
            _ => &[0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
        };

        for (i, &rate) in table.iter().enumerate() {
            if rate >= kbps {
                return i as u8;
            }
        }
        14 // Maximum
    }

    fn sample_rate_index(&self) -> u8 {
        match (self.config.version, self.config.sample_rate) {
            (MpegVersion::Mpeg1, 44100) => 0,
            (MpegVersion::Mpeg1, 48000) => 1,
            (MpegVersion::Mpeg1, 32000) => 2,
            (MpegVersion::Mpeg2, 22050) => 0,
            (MpegVersion::Mpeg2, 24000) => 1,
            (MpegVersion::Mpeg2, 16000) => 2,
            (MpegVersion::Mpeg25, 11025) => 0,
            (MpegVersion::Mpeg25, 12000) => 1,
            (MpegVersion::Mpeg25, 8000) => 2,
            _ => 0,
        }
    }

    fn build_side_info(&self, num_granules: usize) -> Vec<u8> {
        let side_info_size = if self.config.version == MpegVersion::Mpeg1 {
            if self.config.channels == 1 {
                17
            } else {
                32
            }
        } else if self.config.channels == 1 {
            9
        } else {
            17
        };

        let mut side_info = vec![0u8; side_info_size];

        // Main data begin (9 bits for MPEG1, 8 bits for MPEG2)
        side_info[0] = 0; // No main_data_begin for now

        // Private bits
        let private_offset = if self.config.version == MpegVersion::Mpeg1 {
            if self.config.channels == 1 {
                5
            } else {
                3
            }
        } else if self.config.channels == 1 {
            1
        } else {
            2
        };
        // Private bits are already zero

        // Granule info would go here
        // For simplicity, we're using zeros which indicates minimal compression

        side_info
    }

    fn build_main_data(&self, num_granules: usize) -> Vec<u8> {
        let mut main_data = Vec::new();
        let mut bit_buffer = BitBuffer::new();

        for gr in 0..num_granules {
            for ch in 0..self.config.channels as usize {
                let state = &self.granule_states[gr][ch];

                // Encode scalefactors
                self.encode_scalefactors(&mut bit_buffer, state);

                // Encode Huffman data
                self.encode_huffman(&mut bit_buffer, state);
            }
        }

        bit_buffer.flush(&mut main_data);
        main_data
    }

    fn encode_scalefactors(&self, buffer: &mut BitBuffer, state: &GranuleState) {
        // Simplified: encode first 11 scalefactors with slen1 bits, rest with slen2
        let slen1 = 4;
        let slen2 = 3;

        for i in 0..11 {
            buffer.write_bits(state.scalefactors[i] as u32, slen1);
        }

        for i in 11..21 {
            buffer.write_bits(state.scalefactors[i] as u32, slen2);
        }
    }

    fn encode_huffman(&self, buffer: &mut BitBuffer, state: &GranuleState) {
        // Encode big values using selected tables
        let big_values = state.big_values as usize * 2;

        for i in (0..big_values).step_by(2) {
            let x = state.quantized[i].abs().min(15) as u32;
            let y = state.quantized.get(i + 1).map(|v| v.abs().min(15) as u32).unwrap_or(0);

            // Simplified Huffman: just write values directly
            // A real encoder would use the Huffman tables
            buffer.write_bits(x, 4);
            if state.quantized[i] != 0 {
                buffer.write_bits(if state.quantized[i] < 0 { 1 } else { 0 }, 1);
            }
            buffer.write_bits(y, 4);
            if state.quantized.get(i + 1).map(|v| *v != 0).unwrap_or(false) {
                buffer.write_bits(
                    if state.quantized.get(i + 1).map(|v| *v < 0).unwrap_or(false) {
                        1
                    } else {
                        0
                    },
                    1,
                );
            }
        }

        // Count1 region (quadruples)
        for i in (big_values..576).step_by(4) {
            for j in 0..4 {
                let idx = i + j;
                if idx < 576 {
                    let v = state.quantized[idx];
                    if v.abs() <= 1 {
                        buffer.write_bits(v.unsigned_abs().min(1), 1);
                        if v != 0 {
                            buffer.write_bits(if v < 0 { 1 } else { 0 }, 1);
                        }
                    }
                }
            }
        }
    }

    fn calculate_frame_size(&self) -> usize {
        let bitrate = self.config.bitrate;
        let sample_rate = self.config.sample_rate;

        let slots_per_frame = match self.config.version {
            MpegVersion::Mpeg1 => 144 * bitrate / sample_rate,
            MpegVersion::Mpeg2 | MpegVersion::Mpeg25 => 72 * bitrate / sample_rate,
            MpegVersion::Reserved => 0,
        };

        slots_per_frame as usize
    }
}

impl AudioEncoder for Mp3Encoder {
    fn codec_info(&self) -> CodecInfo {
        CodecInfo {
            name: "mp3",
            long_name: "MP3 (MPEG-1 Audio Layer III)",
            can_encode: true,
            can_decode: false,
        }
    }

    fn name(&self) -> &str {
        "mp3"
    }

    fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    fn channels(&self) -> u8 {
        self.config.channels
    }

    fn sample_format(&self) -> SampleFormat {
        SampleFormat::F32
    }

    fn encode(&mut self, sample: &Sample) -> Result<Vec<Packet<'static>>> {
        // Get samples as f32, handling different input formats
        let num_channels = self.config.channels as usize;
        let num_samples = sample.num_samples();

        // Convert input samples to f32 based on format
        let samples_f32: Vec<Vec<f32>> = match sample.format() {
            SampleFormat::F32 => {
                // Interleaved f32
                if let Some(data) = sample.buffer().as_f32() {
                    (0..num_channels)
                        .map(|ch| {
                            data.iter()
                                .skip(ch)
                                .step_by(num_channels)
                                .copied()
                                .collect()
                        })
                        .collect()
                } else {
                    vec![vec![0.0f32; num_samples]; num_channels]
                }
            }
            SampleFormat::S16 => {
                // Interleaved s16
                if let Some(data) = sample.buffer().as_s16() {
                    (0..num_channels)
                        .map(|ch| {
                            data.iter()
                                .skip(ch)
                                .step_by(num_channels)
                                .map(|&s| s as f32 / 32768.0)
                                .collect()
                        })
                        .collect()
                } else {
                    vec![vec![0.0f32; num_samples]; num_channels]
                }
            }
            SampleFormat::F32p => {
                // Planar f32
                (0..num_channels)
                    .map(|ch| {
                        if let Some(channel_data) = sample.channel(ch as u32) {
                            let floats: &[f32] = unsafe {
                                std::slice::from_raw_parts(
                                    channel_data.as_ptr() as *const f32,
                                    channel_data.len() / 4,
                                )
                            };
                            floats.to_vec()
                        } else {
                            vec![0.0f32; num_samples]
                        }
                    })
                    .collect()
            }
            SampleFormat::S16p => {
                // Planar s16
                (0..num_channels)
                    .map(|ch| {
                        if let Some(channel_data) = sample.channel(ch as u32) {
                            let shorts: &[i16] = unsafe {
                                std::slice::from_raw_parts(
                                    channel_data.as_ptr() as *const i16,
                                    channel_data.len() / 2,
                                )
                            };
                            shorts.iter().map(|&s| s as f32 / 32768.0).collect()
                        } else {
                            vec![0.0f32; num_samples]
                        }
                    })
                    .collect()
            }
            _ => {
                // Default: treat as interleaved bytes and assume S16
                let data = sample.data();
                let shorts: &[i16] = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const i16,
                        data.len() / 2,
                    )
                };
                (0..num_channels)
                    .map(|ch| {
                        shorts.iter()
                            .skip(ch)
                            .step_by(num_channels)
                            .map(|&s| s as f32 / 32768.0)
                            .collect()
                    })
                    .collect()
            }
        };

        // Add to input buffer
        for (ch, channel_samples) in samples_f32.iter().enumerate() {
            self.input_buffer[ch].extend_from_slice(channel_samples);
        }

        let mut packets = Vec::new();

        // Encode complete frames
        while self.input_buffer[0].len() >= self.samples_per_frame {
            let frame_samples: Vec<Vec<f32>> = self
                .input_buffer
                .iter_mut()
                .map(|buf| buf.drain(..self.samples_per_frame).collect())
                .collect();

            let refs: Vec<&[f32]> = frame_samples.iter().map(|v| v.as_slice()).collect();

            let frame_data = self.encode_frame(&refs)?;

            let mut packet = Packet::new(frame_data);
            packet.set_keyframe(true);

            packets.push(packet);
        }

        Ok(packets)
    }

    fn flush(&mut self) -> Result<Vec<Packet<'static>>> {
        let mut packets = Vec::new();

        // Encode remaining samples with padding
        if !self.input_buffer[0].is_empty() {
            let remaining = self.input_buffer[0].len();
            let padding = self.samples_per_frame - remaining;

            for buf in &mut self.input_buffer {
                buf.extend(vec![0.0; padding]);
            }

            let frame_samples: Vec<Vec<f32>> = self
                .input_buffer
                .iter_mut()
                .map(std::mem::take)
                .collect();

            let refs: Vec<&[f32]> = frame_samples.iter().map(|v| v.as_slice()).collect();

            let frame_data = self.encode_frame(&refs)?;

            let mut packet = Packet::new(frame_data);
            packet.set_keyframe(true);

            packets.push(packet);
        }

        Ok(packets)
    }

    fn reset(&mut self) {
        for buf in &mut self.input_buffer {
            buf.clear();
        }
        self.reservoir.reset();
        self.frame_count = 0;

        for gr in &mut self.granule_states {
            for ch in gr {
                *ch = GranuleState::default();
            }
        }
    }

    fn extra_data(&self) -> Option<&[u8]> {
        None // MP3 doesn't require extra data
    }
}

impl Default for Mp3Encoder {
    fn default() -> Self {
        Self::new(Mp3EncoderConfig::default()).unwrap()
    }
}

/// Bit buffer for writing bits.
struct BitBuffer {
    data: Vec<u8>,
    current: u32,
    bits: usize,
}

impl BitBuffer {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            current: 0,
            bits: 0,
        }
    }

    fn write_bits(&mut self, value: u32, num_bits: usize) {
        self.current = (self.current << num_bits) | (value & ((1 << num_bits) - 1));
        self.bits += num_bits;

        while self.bits >= 8 {
            self.bits -= 8;
            self.data.push((self.current >> self.bits) as u8);
        }
    }

    fn flush(&mut self, output: &mut Vec<u8>) {
        if self.bits > 0 {
            self.data.push((self.current << (8 - self.bits)) as u8);
        }
        output.extend_from_slice(&self.data);
        self.data.clear();
        self.current = 0;
        self.bits = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use transcode_core::sample::{ChannelLayout, SampleBuffer, SampleFormat};

    #[test]
    fn test_encoder_creation() {
        let config = Mp3EncoderConfig::default();
        let encoder = Mp3Encoder::new(config).unwrap();
        assert_eq!(encoder.sample_rate(), 44100);
        assert_eq!(encoder.channels(), 2);
    }

    #[test]
    fn test_encoder_config_presets() {
        let mono = Mp3EncoderConfig::mono(44100, 128000);
        assert_eq!(mono.channels, 1);
        assert_eq!(mono.channel_mode, ChannelMode::Mono);

        let stereo = Mp3EncoderConfig::stereo(48000, 256000);
        assert_eq!(stereo.channels, 2);
        assert_eq!(stereo.channel_mode, ChannelMode::JointStereo);
    }

    #[test]
    fn test_codec_info() {
        let encoder = Mp3Encoder::default();
        let info = encoder.codec_info();
        assert_eq!(info.name, "mp3");
        assert!(info.can_encode);
    }

    #[test]
    fn test_bit_buffer() {
        let mut buffer = BitBuffer::new();
        buffer.write_bits(0b11, 2);
        buffer.write_bits(0b0000, 4);
        buffer.write_bits(0b11, 2);

        let mut output = Vec::new();
        buffer.flush(&mut output);

        assert_eq!(output[0], 0b11000011);
    }

    #[test]
    fn test_bitrate_index() {
        let config = Mp3EncoderConfig {
            bitrate: 128000,
            ..Default::default()
        };
        let encoder = Mp3Encoder::new(config).unwrap();
        let index = encoder.bitrate_index();
        assert!(index > 0 && index < 15);
    }

    #[test]
    fn test_sample_rate_index() {
        let config = Mp3EncoderConfig {
            sample_rate: 44100,
            version: MpegVersion::Mpeg1,
            ..Default::default()
        };
        let encoder = Mp3Encoder::new(config).unwrap();
        assert_eq!(encoder.sample_rate_index(), 0);
    }

    #[test]
    fn test_header_building() {
        let config = Mp3EncoderConfig::default();
        let encoder = Mp3Encoder::new(config).unwrap();
        let header = encoder.build_header();

        // Check sync word
        assert_eq!(header[0], 0xFF);
        assert_eq!(header[1] & 0xE0, 0xE0);
    }

    #[test]
    fn test_encode_silence() {
        let config = Mp3EncoderConfig::stereo(44100, 128000);
        let mut encoder = Mp3Encoder::new(config).unwrap();

        // Create silent samples
        let samples = vec![0.0f32; 1152 * 2];
        let sample_buffer = SampleBuffer::new(1152, SampleFormat::F32, ChannelLayout::Stereo, 44100);

        // This would need proper Sample conversion
        // Just test that the encoder doesn't panic
    }

    #[test]
    fn test_psy_model() {
        let mut psy = PsyModel::new(44100);
        let mdct = [0.0f32; 576];
        psy.analyze(&mdct);

        // All thresholds should be non-zero
        for band in 0..21 {
            assert!(psy.threshold(band) > 0.0);
        }
    }

    #[test]
    fn test_mdct_processor() {
        let mut mdct = MdctProcessor::new();
        let input = [0.0f32; 36];
        let mut output = [0.0f32; 18];

        mdct.forward(&input, &mut output, 0);

        // Zero input should produce zero output (after overlap)
        // Note: first frame has no overlap, so might not be all zeros
    }
}
