//! MP3 decoder implementation.

use super::huffman::{calculate_regions, Mp3Huffman};
use super::tables::*;
use super::{ChannelMode, GranuleInfo, Mp3FrameHeader, MpegVersion, SideInfo};
use crate::traits::{AudioDecoder, CodecInfo};
use transcode_core::bitstream::BitReader;
use transcode_core::error::{Error, Result};
use transcode_core::packet::Packet;
use transcode_core::sample::{ChannelLayout, Sample, SampleBuffer, SampleFormat};
use std::f32::consts::PI;

/// MP3 decoder state.
pub struct Mp3Decoder {
    /// Sample rate.
    sample_rate: u32,
    /// Number of channels.
    channels: u8,
    /// Bit reservoir.
    bit_reservoir: Vec<u8>,
    /// Synthesis filterbank state per channel.
    synth_state: [[f32; 1024]; 2],
    /// Previous granule output for overlap-add.
    prev_samples: [[f32; 576]; 2],
    /// IMDCT windows.
    window_long: [f32; 36],
    window_short: [f32; 12],
    window_start: [f32; 36],
    window_stop: [f32; 36],
    /// Synthesis window.
    synth_window: [f32; 512],
    /// Synthesis cosine table.
    synth_cos: [[f32; 32]; 64],
    /// Gain table.
    gain_table: [f32; 256],
    /// Frame count.
    frame_count: u64,
}

impl Mp3Decoder {
    /// Create a new MP3 decoder.
    pub fn new() -> Self {
        Self {
            sample_rate: 44100,
            channels: 2,
            bit_reservoir: Vec::with_capacity(4096),
            synth_state: [[0.0; 1024]; 2],
            prev_samples: [[0.0; 576]; 2],
            window_long: imdct_window_long(),
            window_short: imdct_window_short(),
            window_start: imdct_window_start(),
            window_stop: imdct_window_stop(),
            synth_window: synthesis_window(),
            synth_cos: synthesis_cos_table(),
            gain_table: gain_table(),
            frame_count: 0,
        }
    }

    /// Decode an MP3 frame.
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<SampleBuffer> {
        // Parse frame header
        let header = Mp3FrameHeader::parse(data)?;

        self.sample_rate = header.sample_rate();
        self.channels = header.channel_mode.channels();

        let samples_per_frame = header.samples_per_frame();
        let layout = if self.channels == 1 {
            ChannelLayout::Mono
        } else {
            ChannelLayout::Stereo
        };

        let mut output = SampleBuffer::new(
            samples_per_frame,
            SampleFormat::F32,
            layout,
            self.sample_rate,
        );

        // Skip header and CRC
        let header_size = header.header_size();
        let main_data = &data[header_size..];

        // Parse side information
        let side_info = self.parse_side_info(main_data, &header)?;

        // Update bit reservoir
        let side_info_size = header.side_info_size();
        let frame_main_data = &main_data[side_info_size..];

        // Add to reservoir
        self.bit_reservoir.extend_from_slice(frame_main_data);

        // Get main data from reservoir
        let main_data_begin = side_info.main_data_begin as usize;
        if main_data_begin > self.bit_reservoir.len() {
            return Err(Error::Bitstream("Invalid main_data_begin".into()));
        }

        let reservoir_offset = self.bit_reservoir.len() - main_data_begin - frame_main_data.len();
        let main_data_start = reservoir_offset.max(0);

        // Number of granules
        let num_granules = if header.version == MpegVersion::Mpeg1 { 2 } else { 1 };

        // Decode each granule
        let mut pcm_samples = vec![[0.0f32; 576]; 2];

        for gr in 0..num_granules {
            for ch in 0..self.channels as usize {
                let granule = &side_info.granules[gr][ch];

                // Decode Huffman data
                let mut spectrum = [0i32; 576];
                self.decode_huffman(&self.bit_reservoir[main_data_start..], granule, &mut spectrum)?;

                // Requantize
                let mut requant = [0.0f32; 576];
                self.requantize(granule, &spectrum, &mut requant);

                // Reorder short blocks if needed
                if granule.window_switching && granule.block_type == 2 {
                    self.reorder_short_blocks(&mut requant);
                }

                // Stereo processing (joint stereo)
                if ch == 1 && header.channel_mode == ChannelMode::JointStereo {
                    self.apply_joint_stereo(&mut pcm_samples[0], &mut requant, &header);
                }

                // Alias reduction (long blocks only)
                if !granule.window_switching || granule.block_type != 2 {
                    self.alias_reduction(&mut requant);
                }

                // IMDCT
                let mut time_samples = [0.0f32; 576];
                self.apply_imdct(granule, &requant, &mut time_samples, ch);

                // Frequency inversion for odd subbands
                self.frequency_inversion(&mut time_samples);

                // Store for synthesis
                pcm_samples[ch] = time_samples;
            }

            // Synthesis filterbank for each channel
            let gr_offset = gr * 576;
            for ch in 0..self.channels as usize {
                let mut synth_output = [0.0f32; 576];
                self.synthesis_filterbank(ch, &pcm_samples[ch], &mut synth_output);

                // Copy to output
                if let Some(channel_ptr) = output.channel_mut(ch as u32) {
                    for (i, &sample) in synth_output.iter().enumerate() {
                        let out_idx = gr_offset + i;
                        if out_idx < samples_per_frame {
                            let byte_offset = out_idx * 4;
                            if byte_offset + 4 <= channel_ptr.len() {
                                channel_ptr[byte_offset..byte_offset + 4].copy_from_slice(&sample.to_le_bytes());
                            }
                        }
                    }
                }
            }
        }

        // Trim reservoir
        if self.bit_reservoir.len() > 2048 {
            self.bit_reservoir.drain(0..self.bit_reservoir.len() - 2048);
        }

        self.frame_count += 1;
        Ok(output)
    }

    /// Parse side information.
    fn parse_side_info(&self, data: &[u8], header: &Mp3FrameHeader) -> Result<SideInfo> {
        let mut reader = BitReader::new(data);
        let mut side = SideInfo::default();

        let channels = header.channel_mode.channels() as usize;
        let num_granules = if header.version == MpegVersion::Mpeg1 { 2 } else { 1 };

        // Main data begin pointer
        if header.version == MpegVersion::Mpeg1 {
            side.main_data_begin = reader.read_bits(9)? as u16;
            side.private_bits = reader.read_bits(if channels == 1 { 5 } else { 3 })? as u8;

            // SCFSI
            for ch in 0..channels {
                for band in 0..4 {
                    side.scfsi[ch][band] = reader.read_bit()?;
                }
            }
        } else {
            side.main_data_begin = reader.read_bits(8)? as u16;
            side.private_bits = reader.read_bits(if channels == 1 { 1 } else { 2 })? as u8;
        }

        // Granule info
        for gr in 0..num_granules {
            for ch in 0..channels {
                let g = &mut side.granules[gr][ch];

                g.part2_3_length = reader.read_bits(12)? as u16;
                g.big_values = reader.read_bits(9)? as u16;
                g.global_gain = reader.read_bits(8)? as u8;

                if header.version == MpegVersion::Mpeg1 {
                    g.scalefac_compress = reader.read_bits(4)? as u8;
                } else {
                    g.scalefac_compress = reader.read_bits(9)? as u8;
                }

                g.window_switching = reader.read_bit()?;

                if g.window_switching {
                    g.block_type = reader.read_bits(2)? as u8;
                    g.mixed_block = reader.read_bit()?;

                    for i in 0..2 {
                        g.table_select[i] = reader.read_bits(5)? as u8;
                    }

                    for i in 0..3 {
                        g.subblock_gain[i] = reader.read_bits(3)? as u8;
                    }

                    // Set region counts for short blocks
                    g.region0_count = 8;
                    g.region1_count = 0;
                } else {
                    g.block_type = 0;

                    for i in 0..3 {
                        g.table_select[i] = reader.read_bits(5)? as u8;
                    }

                    g.region0_count = reader.read_bits(4)? as u8;
                    g.region1_count = reader.read_bits(3)? as u8;
                }

                if header.version == MpegVersion::Mpeg1 {
                    g.preflag = reader.read_bit()?;
                }
                g.scalefac_scale = reader.read_bit()?;
                g.count1_table = reader.read_bit()?;
            }
        }

        Ok(side)
    }

    /// Decode Huffman-coded spectral data.
    fn decode_huffman(
        &self,
        data: &[u8],
        granule: &GranuleInfo,
        output: &mut [i32; 576],
    ) -> Result<()> {
        let mut reader = BitReader::new(data);

        // Get region boundaries
        let (region1_start, region2_start, big_values_end) = calculate_regions(
            granule.big_values,
            granule.region0_count,
            granule.region1_count,
            granule.block_type,
            &SFB_LONG_44100,
        );

        // Decode big_values regions
        if region1_start > 0 {
            let region0 = &mut output[..region1_start];
            Mp3Huffman::decode_big_values(
                &mut reader,
                granule.table_select[0],
                region1_start / 2,
                region0,
            )?;
        }

        if region2_start > region1_start {
            let region1 = &mut output[region1_start..region2_start];
            Mp3Huffman::decode_big_values(
                &mut reader,
                granule.table_select[1],
                (region2_start - region1_start) / 2,
                region1,
            )?;
        }

        if big_values_end > region2_start {
            let region2 = &mut output[region2_start..big_values_end];
            Mp3Huffman::decode_big_values(
                &mut reader,
                granule.table_select[2],
                (big_values_end - region2_start) / 2,
                region2,
            )?;
        }

        // Decode count1 region
        let count1_start = big_values_end;
        let count1_end = 576;
        let count1_count = (count1_end - count1_start) / 4;

        let count1_region = &mut output[count1_start..];
        Mp3Huffman::decode_count1(&mut reader, granule.count1_table, count1_count, count1_region)?;

        Ok(())
    }

    /// Requantize spectral values.
    fn requantize(&self, granule: &GranuleInfo, input: &[i32; 576], output: &mut [f32; 576]) {
        let global_gain = self.gain_table[granule.global_gain as usize];
        let _scalefac_scale = if granule.scalefac_scale { 1.0 } else { 0.5 };

        for (&val, out) in input.iter().zip(output.iter_mut()) {
            if val == 0 {
                *out = 0.0;
            } else {
                let sign = if val < 0 { -1.0 } else { 1.0 };
                let abs_val = val.abs() as f32;

                // Requantization: sign * |val|^(4/3) * 2^(gain/4)
                *out = sign * abs_val.powf(4.0 / 3.0) * global_gain;
            }
        }
    }

    /// Reorder short block samples.
    fn reorder_short_blocks(&self, samples: &mut [f32; 576]) {
        let mut temp = [0.0f32; 576];

        // Reorder from window-interleaved to subband order
        for sfb in 0..13 {
            let start = SFB_SHORT_44100[sfb] as usize;
            let end = SFB_SHORT_44100[sfb + 1] as usize;
            let width = end - start;

            for win in 0..3 {
                for i in 0..width {
                    let src = start * 3 + win * width + i;
                    let dst = start + win * width + i;
                    if src < 576 && dst < 576 {
                        temp[dst] = samples[src];
                    }
                }
            }
        }

        samples.copy_from_slice(&temp);
    }

    /// Apply joint stereo processing.
    fn apply_joint_stereo(
        &self,
        left: &mut [f32; 576],
        right: &mut [f32; 576],
        header: &Mp3FrameHeader,
    ) {
        let ms_stereo = (header.mode_extension & 2) != 0;
        let intensity_stereo = (header.mode_extension & 1) != 0;

        if ms_stereo {
            // M/S stereo: L = (M + S) / sqrt(2), R = (M - S) / sqrt(2)
            let scale = 1.0 / 2.0f32.sqrt();
            for i in 0..576 {
                let m = left[i];
                let s = right[i];
                left[i] = (m + s) * scale;
                right[i] = (m - s) * scale;
            }
        }

        if intensity_stereo {
            // Intensity stereo processing would go here
            // (using scalefactor band positions)
        }
    }

    /// Apply alias reduction between subbands.
    fn alias_reduction(&self, samples: &mut [f32; 576]) {
        for sb in 1..32 {
            let base = sb * 18;

            for i in 0..8 {
                let upper = base - 1 - i;
                let lower = base + i;

                if upper < 576 && lower < 576 {
                    let a = samples[upper];
                    let b = samples[lower];

                    samples[upper] = a * ALIAS_CS[i] - b * ALIAS_CA[i];
                    samples[lower] = b * ALIAS_CS[i] + a * ALIAS_CA[i];
                }
            }
        }
    }

    /// Apply IMDCT to transform frequency to time domain.
    fn apply_imdct(
        &mut self,
        granule: &GranuleInfo,
        input: &[f32; 576],
        output: &mut [f32; 576],
        ch: usize,
    ) {
        if granule.window_switching && granule.block_type == 2 {
            // Short blocks
            for sb in 0..32 {
                let in_offset = sb * 18;
                let mut temp = [0.0f32; 36];

                // 3 short blocks per subband
                for block in 0..3 {
                    let block_offset = block * 6;
                    let mut block_out = [0.0f32; 12];

                    // 12-point IMDCT
                    for i in 0..12 {
                        let mut sum = 0.0f32;
                        for k in 0..6 {
                            let angle = PI / 12.0 * (2.0 * i as f32 + 1.0 + 6.0) * (2.0 * k as f32 + 1.0) / 2.0;
                            sum += input[in_offset + block * 6 + k] * angle.cos();
                        }
                        block_out[i] = sum;
                    }

                    // Apply window and overlap
                    for i in 0..12 {
                        temp[block_offset + i] += block_out[i] * self.window_short[i];
                    }
                }

                // Overlap-add with previous frame
                for i in 0..18 {
                    output[sb * 18 + i] = temp[i] + self.prev_samples[ch][sb * 18 + i];
                }
                self.prev_samples[ch][sb * 18..sb * 18 + 18].copy_from_slice(&temp[18..36]);
            }
        } else {
            // Long blocks
            let window = match granule.block_type {
                1 => &self.window_start,
                3 => &self.window_stop,
                _ => &self.window_long,
            };

            for sb in 0..32 {
                let in_offset = sb * 18;
                let mut temp = [0.0f32; 36];

                // 36-point IMDCT
                for i in 0..36 {
                    let mut sum = 0.0f32;
                    for k in 0..18 {
                        let angle = PI / 36.0 * (2.0 * i as f32 + 1.0 + 18.0) * (2.0 * k as f32 + 1.0) / 2.0;
                        sum += input[in_offset + k] * angle.cos();
                    }
                    temp[i] = sum * window[i];
                }

                // Overlap-add with previous frame
                for i in 0..18 {
                    output[sb * 18 + i] = temp[i] + self.prev_samples[ch][sb * 18 + i];
                }
                self.prev_samples[ch][sb * 18..sb * 18 + 18].copy_from_slice(&temp[18..36]);
            }
        }
    }

    /// Apply frequency inversion for odd subbands.
    fn frequency_inversion(&self, samples: &mut [f32; 576]) {
        for sb in (1..32).step_by(2) {
            for i in (1..18).step_by(2) {
                samples[sb * 18 + i] = -samples[sb * 18 + i];
            }
        }
    }

    /// Apply synthesis filterbank.
    fn synthesis_filterbank(&mut self, ch: usize, input: &[f32; 576], output: &mut [f32; 576]) {
        // Process 18 samples at a time (one subband)
        for block in 0..18 {
            // Shift state buffer
            for i in (64..1024).rev() {
                self.synth_state[ch][i] = self.synth_state[ch][i - 64];
            }

            // Matrixing: transform 32 subbands to 64 intermediate values
            for i in 0..64 {
                let mut sum = 0.0f32;
                for k in 0..32 {
                    sum += input[k * 18 + block] * self.synth_cos[i][k];
                }
                self.synth_state[ch][i] = sum;
            }

            // Build 32 output samples
            for i in 0..32 {
                let mut sum = 0.0f32;

                // 16 taps
                for j in 0..16 {
                    let idx = i + 32 * j;
                    if idx < 512 {
                        let state_idx = (j * 64 + i) % 1024;
                        sum += self.synth_window[idx] * self.synth_state[ch][state_idx];
                    }
                }

                output[block * 32 + i] = sum;
            }
        }
    }
}

impl Default for Mp3Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioDecoder for Mp3Decoder {
    fn codec_info(&self) -> CodecInfo {
        CodecInfo {
            name: "mp3",
            long_name: "MP3 (MPEG Audio Layer III)",
            can_encode: false,
            can_decode: true,
        }
    }

    fn name(&self) -> &str {
        "MP3"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> u8 {
        self.channels
    }

    fn sample_format(&self) -> SampleFormat {
        SampleFormat::F32
    }

    fn set_extra_data(&mut self, _data: &[u8]) -> Result<()> {
        // MP3 doesn't require extra data
        Ok(())
    }

    fn decode(&mut self, packet: &Packet<'_>) -> Result<Vec<Sample>> {
        let buffer = self.decode_frame(packet.data())?;
        let sample = Sample::from_buffer(buffer);
        Ok(vec![sample])
    }

    fn flush(&mut self) -> Result<Vec<Sample>> {
        self.bit_reservoir.clear();
        self.synth_state = [[0.0; 1024]; 2];
        self.prev_samples = [[0.0; 576]; 2];
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.bit_reservoir.clear();
        self.synth_state = [[0.0; 1024]; 2];
        self.prev_samples = [[0.0; 576]; 2];
    }
}
