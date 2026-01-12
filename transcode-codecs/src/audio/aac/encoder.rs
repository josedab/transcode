//! AAC encoder implementation.

use super::huffman::HuffmanEncoder;
use super::mdct::Mdct;
use super::psy::{PsyModel, PsyModelConfig};
use super::tables::{sine_window_1024, sine_window_128, WindowSequence};
use super::{AdtsHeader, AacProfile, AudioSpecificConfig, SampleRateIndex};
use crate::traits::{AudioEncoder, CodecInfo};
use transcode_core::error::{Error, Result};
use transcode_core::packet::Packet;
use transcode_core::sample::{Sample, SampleBuffer, SampleFormat};

/// AAC encoder configuration.
#[derive(Debug, Clone)]
pub struct AacEncoderConfig {
    /// Profile.
    pub profile: AacProfile,
    /// Sample rate.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u8,
    /// Bitrate in bits per second.
    pub bitrate: u32,
    /// Add ADTS headers.
    pub adts: bool,
    /// Quality (0.0 - 1.0, higher = better quality, more bits).
    pub quality: f32,
}

impl Default for AacEncoderConfig {
    fn default() -> Self {
        Self {
            profile: AacProfile::Lc,
            sample_rate: 44100,
            channels: 2,
            bitrate: 128000,
            adts: true,
            quality: 0.5,
        }
    }
}

/// Quantization state for a channel.
struct QuantState {
    /// Scalefactors per band.
    scalefactors: [u8; 64],
    /// Quantized coefficients.
    quantized: [i16; 1024],
    /// Window sequence.
    window_sequence: WindowSequence,
}

impl Default for QuantState {
    fn default() -> Self {
        Self {
            scalefactors: [0; 64],
            quantized: [0; 1024],
            window_sequence: WindowSequence::OnlyLong,
        }
    }
}

/// AAC encoder.
pub struct AacEncoder {
    /// Configuration.
    config: AacEncoderConfig,
    /// MDCT processors.
    mdct_long: Mdct,
    mdct_short: Mdct,
    /// Psychoacoustic model per channel.
    psy_models: Vec<PsyModel>,
    /// Input buffer per channel.
    input_buffer: Vec<Vec<f32>>,
    /// Previous input for overlap.
    prev_input: Vec<Vec<f32>>,
    /// Long window.
    window_long: [f32; 1024],
    /// Short window.
    window_short: [f32; 128],
    /// Frame counter.
    frame_count: u64,
    /// Bits per frame target.
    bits_per_frame: u32,
    /// Cached extra data (AudioSpecificConfig).
    extra_data_cache: Vec<u8>,
}

impl AacEncoder {
    /// Create a new AAC encoder.
    pub fn new(config: AacEncoderConfig) -> Result<Self> {
        // Validate sample rate to prevent division by zero
        if config.sample_rate == 0 {
            return Err(Error::InvalidParameter(
                "sample_rate must be greater than 0".into(),
            ));
        }
        if config.channels == 0 {
            return Err(Error::InvalidParameter(
                "channels must be greater than 0".into(),
            ));
        }

        let channels = config.channels as usize;

        let psy_config = PsyModelConfig {
            sample_rate: config.sample_rate,
            channels: config.channels,
            bitrate: config.bitrate,
            quality: config.quality,
        };

        let mut psy_models = Vec::with_capacity(channels);
        let mut input_buffer = Vec::with_capacity(channels);
        let mut prev_input = Vec::with_capacity(channels);

        for _ in 0..channels {
            psy_models.push(PsyModel::new(psy_config.clone()));
            input_buffer.push(vec![0.0; 2048]);
            prev_input.push(vec![0.0; 1024]);
        }

        // Calculate bits per frame, avoiding division by zero for low sample rates
        let frames_per_second = config.sample_rate.max(1024) / 1024;
        let bits_per_frame = config.bitrate / frames_per_second.max(1);

        // Compute extra data (AudioSpecificConfig)
        let sample_rate_index = SampleRateIndex::from_sample_rate(config.sample_rate) as u8;
        let asc = AudioSpecificConfig {
            object_type: config.profile.object_type(),
            sample_rate_index,
            sample_rate: config.sample_rate,
            channel_config: config.channels,
            frame_length: 1024,
            depends_on_core_coder: false,
            extension_object_type: None,
            extension_sample_rate: None,
            sbr_present: false,
            ps_present: false,
        };
        let extra_data_cache = asc.encode();

        Ok(Self {
            config,
            mdct_long: Mdct::new(1024),
            mdct_short: Mdct::new(128),
            psy_models,
            input_buffer,
            prev_input,
            window_long: sine_window_1024(),
            window_short: sine_window_128(),
            frame_count: 0,
            bits_per_frame,
            extra_data_cache,
        })
    }

    /// Get Audio Specific Config.
    pub fn audio_specific_config(&self) -> AudioSpecificConfig {
        let sample_rate_index = SampleRateIndex::from_sample_rate(self.config.sample_rate) as u8;

        AudioSpecificConfig {
            object_type: self.config.profile.object_type(),
            sample_rate_index,
            sample_rate: self.config.sample_rate,
            channel_config: self.config.channels,
            frame_length: 1024,
            depends_on_core_coder: false,
            extension_object_type: None,
            extension_sample_rate: None,
            sbr_present: false,
            ps_present: false,
        }
    }

    /// Encode a frame of audio samples.
    pub fn encode_frame(&mut self, samples: &SampleBuffer) -> Result<Packet<'static>> {
        if samples.layout.channels() != self.config.channels as u32 {
            return Err(Error::Codec(
                format!(
                    "Expected {} channels, got {}",
                    self.config.channels,
                    samples.layout.channels()
                )
                .into(),
            ));
        }

        let mut encoded_data = Vec::with_capacity(2048);

        // Process each channel
        let mut quant_states = Vec::with_capacity(self.config.channels as usize);

        for ch in 0..self.config.channels as usize {
            // Get input samples as f32
            let channel_samples = self.get_channel_samples(samples, ch);

            // Fill input buffer (overlap-add with previous frame)
            self.input_buffer[ch][..1024].copy_from_slice(&self.prev_input[ch]);
            self.input_buffer[ch][1024..].copy_from_slice(&channel_samples[..1024]);

            // Update previous input
            self.prev_input[ch].copy_from_slice(&channel_samples[..1024]);

            // Psychoacoustic analysis
            let mut spectrum = [0.0f32; 1024];
            self.compute_spectrum(ch, &mut spectrum);
            let psy_analysis = self.psy_models[ch].analyze(&spectrum);

            // Decide window sequence
            let window_sequence = if psy_analysis.use_short_windows {
                WindowSequence::EightShort
            } else {
                WindowSequence::OnlyLong
            };

            // Apply window and MDCT
            let mdct_coeffs = self.apply_mdct(ch, window_sequence);

            // Quantization
            let quant_state = self.quantize(
                &mdct_coeffs,
                &psy_analysis.thresholds,
                window_sequence,
            );

            quant_states.push(quant_state);
        }

        // Write bitstream
        self.write_frame(&quant_states, &mut encoded_data)?;

        // Add ADTS header if configured
        let final_data = if self.config.adts {
            let header = AdtsHeader {
                mpeg_version: 0, // MPEG-4
                layer: 0,
                protection_absent: true,
                profile: (self.config.profile.object_type() - 1) & 3,
                sample_rate_index: SampleRateIndex::from_sample_rate(self.config.sample_rate) as u8,
                private_bit: false,
                channel_config: self.config.channels,
                original: false,
                home: false,
                copyright_id_bit: false,
                copyright_id_start: false,
                frame_length: (7 + encoded_data.len()) as u16,
                buffer_fullness: 0x7FF, // VBR
                num_raw_data_blocks: 0,
                crc: None,
            };

            let mut adts_data = header.encode();
            adts_data.extend_from_slice(&encoded_data);
            adts_data
        } else {
            encoded_data
        };

        let mut packet = Packet::new(final_data);
        let time_base = transcode_core::TimeBase::new(1, self.config.sample_rate as i64);
        packet.pts = transcode_core::Timestamp::new((self.frame_count * 1024) as i64, time_base);
        packet.duration = transcode_core::Duration::new(1024, time_base);

        self.frame_count += 1;

        Ok(packet)
    }

    /// Get samples from channel as f32.
    fn get_channel_samples(&self, samples: &SampleBuffer, ch: usize) -> Vec<f32> {
        let channel_data = samples.channel(ch as u32).unwrap_or(&[]);
        let num_samples = samples.num_samples;
        let mut result = vec![0.0f32; num_samples];

        // Assume F32 format
        for i in 0..num_samples {
            if (i + 1) * 4 <= channel_data.len() {
                let bytes = &channel_data[i * 4..(i + 1) * 4];
                result[i] = f32::from_le_bytes(bytes.try_into().unwrap_or([0; 4]));
            }
        }

        result
    }

    /// Compute FFT spectrum for psychoacoustic analysis.
    fn compute_spectrum(&self, ch: usize, output: &mut [f32; 1024]) {
        // Simple magnitude spectrum computation
        let input = &self.input_buffer[ch];

        for k in 0..512 {
            let mut re = 0.0f32;
            let mut im = 0.0f32;

            for n in 0..1024 {
                let angle = -2.0 * std::f32::consts::PI * (k as f32) * (n as f32) / 1024.0;
                re += input[n] * angle.cos();
                im += input[n] * angle.sin();
            }

            output[k] = (re * re + im * im).sqrt();
        }

        // Mirror
        for k in 512..1024 {
            output[k] = output[1023 - k];
        }
    }

    /// Apply window and MDCT.
    fn apply_mdct(&mut self, ch: usize, window_sequence: WindowSequence) -> Vec<f32> {
        let mut coeffs = vec![0.0f32; 1024];

        match window_sequence {
            WindowSequence::OnlyLong | WindowSequence::LongStart | WindowSequence::LongStop => {
                // Apply window
                let mut windowed = vec![0.0f32; 2048];
                for i in 0..2048 {
                    let window_val = if i < 1024 {
                        self.window_long[i]
                    } else {
                        self.window_long[2047 - i]
                    };
                    windowed[i] = self.input_buffer[ch][i] * window_val;
                }

                // MDCT
                self.mdct_long.forward(&windowed, &mut coeffs);
            }
            WindowSequence::EightShort => {
                // 8 short blocks
                for b in 0..8 {
                    let offset = 448 + b * 128;
                    let mut windowed = vec![0.0f32; 256];

                    for i in 0..256 {
                        let window_val = if i < 128 {
                            self.window_short[i]
                        } else {
                            self.window_short[255 - i]
                        };
                        if offset + i < 2048 {
                            windowed[i] = self.input_buffer[ch][offset + i] * window_val;
                        }
                    }

                    let mut block_coeffs = vec![0.0f32; 128];
                    self.mdct_short.forward(&windowed, &mut block_coeffs);

                    coeffs[b * 128..(b + 1) * 128].copy_from_slice(&block_coeffs);
                }
            }
        }

        coeffs
    }

    /// Quantize MDCT coefficients.
    fn quantize(
        &self,
        coeffs: &[f32],
        thresholds: &[f32],
        window_sequence: WindowSequence,
    ) -> QuantState {
        let mut state = QuantState {
            window_sequence,
            ..Default::default()
        };

        // Number of bands
        let num_bands = 49;

        // Quantization loop
        for band in 0..num_bands {
            let start = band * 20; // Simplified band boundaries
            let end = ((band + 1) * 20).min(1024);

            if start >= coeffs.len() {
                break;
            }

            // Find optimal scalefactor for this band
            let threshold = thresholds.get(band).copied().unwrap_or(1.0);
            let band_coeffs = &coeffs[start..end.min(coeffs.len())];

            let max_abs = band_coeffs
                .iter()
                .map(|c| c.abs())
                .fold(0.0f32, f32::max);

            if max_abs < threshold {
                // Below masking threshold, zero out
                state.scalefactors[band] = 0;
                for i in start..end.min(1024) {
                    state.quantized[i] = 0;
                }
            } else {
                // Find scalefactor that quantizes without too much distortion
                let sf = self.find_scalefactor(max_abs, threshold);
                state.scalefactors[band] = sf;

                let gain = 2.0f32.powf(0.25 * (sf as f32 - 100.0));

                for i in start..end.min(1024) {
                    let scaled = coeffs[i] / gain;
                    // Quantize: sign(x) * floor(|x|^0.75 + 0.4054)
                    let sign = if scaled < 0.0 { -1.0 } else { 1.0 };
                    let quant = (scaled.abs().powf(0.75) + 0.4054).floor();
                    state.quantized[i] = (sign * quant) as i16;
                }
            }
        }

        state
    }

    /// Find optimal scalefactor for a band.
    fn find_scalefactor(&self, max_coeff: f32, _threshold: f32) -> u8 {
        // Binary search for best scalefactor
        let mut sf = 100u8;

        // Target: quantize such that reconstruction error is below threshold
        let target_gain = max_coeff / 8191.0; // Max quantized value is ~8191

        // sf = 4 * log2(gain) + 100
        if target_gain > 0.0 {
            sf = ((4.0 * target_gain.log2() + 100.0) as u8).clamp(0, 255);
        }

        sf
    }

    /// Write encoded frame to bitstream.
    fn write_frame(&self, quant_states: &[QuantState], output: &mut Vec<u8>) -> Result<()> {
        use transcode_core::bitstream::BitWriter;

        let mut writer = BitWriter::new();

        // Write syntactic elements
        for (ch, state) in quant_states.iter().enumerate() {
            // id_syn_ele (3 bits): SCE=0x1, CPE=0x2
            if ch == 0 {
                writer.write_bits(0x1, 3)?; // SCE for first channel
            }

            // element_instance_tag (4 bits)
            writer.write_bits(ch as u32, 4)?;

            // ICS info
            self.write_ics_info(&mut writer, state)?;

            // Section data
            self.write_section_data(&mut writer, state)?;

            // Scalefactors
            self.write_scalefactors(&mut writer, state)?;

            // Spectral data
            self.write_spectral_data(&mut writer, state)?;
        }

        // END element
        writer.write_bits(0x7, 3)?; // id_syn_ele = END

        // Byte-align
        writer.byte_align()?;

        output.extend_from_slice(writer.data());
        Ok(())
    }

    fn write_ics_info(&self, writer: &mut transcode_core::bitstream::BitWriter, state: &QuantState) -> Result<()> {
        writer.write_bit(false)?; // ics_reserved_bit
        writer.write_bits(state.window_sequence as u32, 2)?;
        writer.write_bit(false)?; // window_shape (sine)

        if state.window_sequence == WindowSequence::EightShort {
            writer.write_bits(14, 4)?; // max_sfb for short
            writer.write_bits(0, 7)?; // scale_factor_grouping (all in one group)
        } else {
            writer.write_bits(49, 6)?; // max_sfb for long
            writer.write_bit(false)?; // predictor_data_present
        }

        Ok(())
    }

    fn write_section_data(&self, writer: &mut transcode_core::bitstream::BitWriter, state: &QuantState) -> Result<()> {
        // Simplified: one section per band with selected codebook
        let num_bands = if state.window_sequence == WindowSequence::EightShort {
            14
        } else {
            49
        };

        let sect_bits = if state.window_sequence == WindowSequence::EightShort { 3 } else { 5 };

        let mut band = 0;
        while band < num_bands {
            // Find run of bands with same codebook
            let cb = HuffmanEncoder::select_codebook(&state.quantized[band * 20..((band + 1) * 20).min(1024)]);
            let mut run = 1;

            while band + run < num_bands {
                let next_cb = HuffmanEncoder::select_codebook(
                    &state.quantized[(band + run) * 20..((band + run + 1) * 20).min(1024)],
                );
                if next_cb != cb {
                    break;
                }
                run += 1;
            }

            // Write section
            writer.write_bits(cb as u32, 4)?;

            // Section length encoding
            let sect_esc: u32 = (1u32 << sect_bits) - 1;
            let mut remaining = run as u32;
            while remaining >= sect_esc {
                writer.write_bits(sect_esc, sect_bits)?;
                remaining -= sect_esc;
            }
            writer.write_bits(remaining as u32, sect_bits)?;

            band += run;
        }

        Ok(())
    }

    fn write_scalefactors(&self, writer: &mut transcode_core::bitstream::BitWriter, state: &QuantState) -> Result<()> {
        let num_bands = if state.window_sequence == WindowSequence::EightShort { 14 } else { 49 };

        // Global gain (8 bits)
        let global_gain = state.scalefactors[0];
        writer.write_bits(global_gain as u32, 8)?;

        // Differential coding for remaining scalefactors
        let mut prev_sf = global_gain as i16;
        for band in 1..num_bands {
            let sf = state.scalefactors[band] as i16;
            let delta = sf - prev_sf;

            // Write as signed Exp-Golomb
            writer.write_signed_exp_golomb(delta as i32)?;

            prev_sf = sf;
        }

        Ok(())
    }

    fn write_spectral_data(&self, writer: &mut transcode_core::bitstream::BitWriter, state: &QuantState) -> Result<()> {
        // Simplified: write quantized values directly
        // Real implementation would use Huffman coding

        let num_bands = if state.window_sequence == WindowSequence::EightShort { 14 } else { 49 };

        for band in 0..num_bands {
            let start = band * 20;
            let end = ((band + 1) * 20).min(1024);

            for i in start..end {
                let val = state.quantized[i];
                if val != 0 {
                    // Sign bit
                    writer.write_bit(val < 0)?;
                    // Magnitude (simplified)
                    let abs_val = val.abs() as u32;
                    writer.write_bits(abs_val.min(15), 4)?;
                }
            }
        }

        Ok(())
    }
}

impl AudioEncoder for AacEncoder {
    fn codec_info(&self) -> CodecInfo {
        CodecInfo {
            name: "aac",
            long_name: "AAC (Advanced Audio Coding)",
            can_encode: true,
            can_decode: true,
        }
    }

    fn name(&self) -> &str {
        "AAC"
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

    fn extra_data(&self) -> Option<&[u8]> {
        Some(&self.extra_data_cache)
    }

    fn encode(&mut self, sample: &Sample) -> Result<Vec<Packet<'static>>> {
        let packet = self.encode_frame(sample.buffer())?;
        Ok(vec![packet.into_owned()])
    }

    fn flush(&mut self) -> Result<Vec<Packet<'static>>> {
        // Reset state
        for ch in 0..self.config.channels as usize {
            self.prev_input[ch].fill(0.0);
            self.psy_models[ch].reset();
        }

        Ok(Vec::new())
    }

    fn reset(&mut self) {
        for ch in 0..self.config.channels as usize {
            self.prev_input[ch].fill(0.0);
            self.psy_models[ch].reset();
        }
        self.frame_count = 0;
    }
}
