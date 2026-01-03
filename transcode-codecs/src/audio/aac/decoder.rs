//! AAC decoder implementation.

use super::huffman::HuffmanDecoder;
use super::mdct::ImdctContext;
use super::tables::{scalefactor_gain, WindowSequence};
use super::tns::{TnsData, TnsProcessor};
use super::AudioSpecificConfig;
use crate::traits::{AudioDecoder, CodecInfo};
use transcode_core::bitstream::BitReader;
use transcode_core::error::{Error, Result};
use transcode_core::packet::Packet;
use transcode_core::sample::{ChannelLayout, Sample, SampleBuffer, SampleFormat};

/// Individual Channel Stream data.
#[derive(Debug, Default)]
struct IcsInfo {
    /// Window sequence.
    window_sequence: WindowSequence,
    /// Window shape.
    window_shape: u8,
    /// Max scalefactor band (short blocks).
    max_sfb: u8,
    /// Scale factor grouping.
    scale_factor_grouping: u8,
    /// Predictor data present.
    predictor_data_present: bool,
    /// Number of window groups.
    num_window_groups: u8,
    /// Windows per group.
    window_group_len: [u8; 8],
    /// Total number of scalefactor bands.
    num_swb: u8,
}

impl IcsInfo {
    fn parse(reader: &mut BitReader<'_>) -> Result<Self> {
        let mut info = IcsInfo::default();

        let _reserved = reader.read_bit()?;
        let window_sequence = reader.read_bits(2)?;
        info.window_sequence = match window_sequence {
            0 => WindowSequence::OnlyLong,
            1 => WindowSequence::LongStart,
            2 => WindowSequence::EightShort,
            3 => WindowSequence::LongStop,
            _ => WindowSequence::OnlyLong,
        };
        info.window_shape = reader.read_bit()? as u8;

        if info.window_sequence == WindowSequence::EightShort {
            info.max_sfb = reader.read_bits(4)? as u8;
            info.scale_factor_grouping = reader.read_bits(7)? as u8;

            // Compute window groups
            info.num_window_groups = 1;
            info.window_group_len[0] = 1;

            for i in 0..7 {
                if (info.scale_factor_grouping >> (6 - i)) & 1 == 0 {
                    info.num_window_groups += 1;
                    info.window_group_len[info.num_window_groups as usize - 1] = 1;
                } else {
                    info.window_group_len[info.num_window_groups as usize - 1] += 1;
                }
            }

            info.num_swb = 14; // Typical for short windows
        } else {
            info.max_sfb = reader.read_bits(6)? as u8;

            info.predictor_data_present = reader.read_bit()?;
            if info.predictor_data_present {
                // Parse predictor data (simplified)
                let _predictor_reset = reader.read_bit()?;
            }

            info.num_window_groups = 1;
            info.window_group_len[0] = 1;
            info.num_swb = 49; // Typical for long windows
        }

        Ok(info)
    }
}

/// Section data for codebook assignment.
#[derive(Debug)]
struct SectionData {
    /// Codebook for each section.
    sect_cb: [[u8; 64]; 8],
    /// Section start band.
    sect_start: [[u8; 64]; 8],
    /// Section end band.
    sect_end: [[u8; 64]; 8],
    /// Number of sections per group.
    num_sect: [u8; 8],
}

impl Default for SectionData {
    fn default() -> Self {
        Self {
            sect_cb: [[0u8; 64]; 8],
            sect_start: [[0u8; 64]; 8],
            sect_end: [[0u8; 64]; 8],
            num_sect: [0u8; 8],
        }
    }
}

impl SectionData {
    fn parse(
        reader: &mut BitReader<'_>,
        ics_info: &IcsInfo,
    ) -> Result<Self> {
        let mut data = SectionData::default();

        let sect_bits = if ics_info.window_sequence == WindowSequence::EightShort {
            3
        } else {
            5
        };
        let sect_esc = (1 << sect_bits) - 1;

        for g in 0..ics_info.num_window_groups as usize {
            let mut k = 0u8;
            let mut i = 0usize;

            while k < ics_info.max_sfb {
                data.sect_cb[g][i] = reader.read_bits(4)? as u8;
                data.sect_start[g][i] = k;

                let mut sect_len = 0u8;
                loop {
                    let incr = reader.read_bits(sect_bits)? as u8;
                    sect_len += incr;
                    if incr != sect_esc {
                        break;
                    }
                }

                data.sect_end[g][i] = k + sect_len;
                k += sect_len;
                i += 1;
            }

            data.num_sect[g] = i as u8;
        }

        Ok(data)
    }
}

/// Channel data.
struct ChannelData {
    /// ICS info.
    ics_info: IcsInfo,
    /// Section data.
    section_data: SectionData,
    /// Scalefactors.
    scalefactors: [[u8; 64]; 8],
    /// Spectral coefficients (quantized).
    quant_spec: [i16; 1024],
    /// TNS data.
    tns_data: Option<TnsData>,
    /// Gain control present.
    gain_control_present: bool,
}

impl Default for ChannelData {
    fn default() -> Self {
        Self {
            ics_info: IcsInfo::default(),
            section_data: SectionData::default(),
            scalefactors: [[0; 64]; 8],
            quant_spec: [0; 1024],
            tns_data: None,
            gain_control_present: false,
        }
    }
}

/// AAC decoder.
pub struct AacDecoder {
    /// Audio specific config.
    config: Option<AudioSpecificConfig>,
    /// IMDCT context per channel.
    imdct: Vec<ImdctContext>,
    /// TNS processor per channel.
    tns: Vec<TnsProcessor>,
    /// Sample rate.
    sample_rate: u32,
    /// Number of channels.
    channels: u8,
    /// Initialized flag.
    initialized: bool,
}

impl AacDecoder {
    /// Create a new AAC decoder.
    pub fn new() -> Self {
        Self {
            config: None,
            imdct: Vec::new(),
            tns: Vec::new(),
            sample_rate: 44100,
            channels: 2,
            initialized: false,
        }
    }

    /// Initialize with Audio Specific Config.
    pub fn init(&mut self, asc: AudioSpecificConfig) -> Result<()> {
        self.sample_rate = asc.get_sample_rate();
        self.channels = asc.get_channels();

        self.imdct.clear();
        self.tns.clear();

        for _ in 0..self.channels {
            self.imdct.push(ImdctContext::new());
            self.tns.push(TnsProcessor::new());
        }

        self.config = Some(asc);
        self.initialized = true;

        Ok(())
    }

    /// Decode a raw AAC frame.
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<SampleBuffer> {
        if !self.initialized {
            return Err(Error::Codec("Decoder not initialized".into()));
        }

        let mut reader = BitReader::new(data);

        // Allocate output buffer
        let samples_per_channel = 1024;
        let layout = match self.channels {
            1 => ChannelLayout::Mono,
            2 => ChannelLayout::Stereo,
            6 => ChannelLayout::Surround51,
            _ => ChannelLayout::Stereo,
        };

        let mut output = SampleBuffer::new(
            samples_per_channel,
            SampleFormat::F32,
            layout,
            self.sample_rate,
        );

        // Decode each channel
        for ch in 0..self.channels as usize {
            let channel_data = self.decode_channel(&mut reader)?;
            let samples = self.process_channel(ch, &channel_data)?;

            // Copy to output
            if let Some(channel_ptr) = output.channel_mut(ch as u32) {
                for (i, &sample) in samples.iter().enumerate() {
                    let start = i * std::mem::size_of::<f32>();
                    let end = (i + 1) * std::mem::size_of::<f32>();
                    if end <= channel_ptr.len() {
                        channel_ptr[start..end].copy_from_slice(&sample.to_le_bytes());
                    }
                }
            }
        }

        Ok(output)
    }

    /// Decode a single channel.
    fn decode_channel(&self, reader: &mut BitReader<'_>) -> Result<ChannelData> {
        let mut channel = ChannelData::default();

        // Parse ICS info
        channel.ics_info = IcsInfo::parse(reader)?;

        // Parse section data
        channel.section_data = SectionData::parse(reader, &channel.ics_info)?;

        // Parse scalefactors
        self.parse_scalefactors(reader, &mut channel)?;

        // Pulse data (skip for now)
        let pulse_present = reader.read_bit()?;
        if pulse_present {
            self.skip_pulse_data(reader)?;
        }

        // TNS data
        let tns_present = reader.read_bit()?;
        if tns_present {
            channel.tns_data = Some(TnsData::parse(
                reader,
                channel.ics_info.window_sequence as u8,
            )?);
        }

        // Gain control
        channel.gain_control_present = reader.read_bit()?;
        if channel.gain_control_present {
            // Skip gain control data
            return Err(Error::Codec("Gain control not supported".into()));
        }

        // Decode spectral data
        self.decode_spectral(reader, &mut channel)?;

        Ok(channel)
    }

    /// Parse scalefactors.
    fn parse_scalefactors(
        &self,
        reader: &mut BitReader<'_>,
        channel: &mut ChannelData,
    ) -> Result<()> {
        let mut global_gain = reader.read_bits(8)? as i16;

        for g in 0..channel.ics_info.num_window_groups as usize {
            for sfb in 0..channel.ics_info.max_sfb as usize {
                // Get codebook for this band
                let cb = channel.section_data.sect_cb[g][sfb];

                if cb != 0 && cb != 13 && cb != 14 && cb != 15 {
                    // Read scalefactor delta
                    let delta = reader.read_se()? as i16;
                    global_gain += delta;
                    channel.scalefactors[g][sfb] = global_gain.clamp(0, 255) as u8;
                }
            }
        }

        Ok(())
    }

    /// Skip pulse data.
    fn skip_pulse_data(&self, reader: &mut BitReader<'_>) -> Result<()> {
        let num_pulse = reader.read_bits(2)?;
        let _pulse_start_sfb = reader.read_bits(6)?;

        for _ in 0..=num_pulse {
            let _pulse_offset = reader.read_bits(5)?;
            let _pulse_amp = reader.read_bits(4)?;
        }

        Ok(())
    }

    /// Decode spectral data.
    fn decode_spectral(&self, reader: &mut BitReader<'_>, channel: &mut ChannelData) -> Result<()> {
        let ics = &channel.ics_info;

        for g in 0..ics.num_window_groups as usize {
            for s in 0..channel.section_data.num_sect[g] as usize {
                let start = channel.section_data.sect_start[g][s] as usize;
                let end = channel.section_data.sect_end[g][s] as usize;
                let cb = channel.section_data.sect_cb[g][s];

                // Calculate spectral offset
                let spec_start = start * 4; // Each band has 4 coefficients (simplified)
                let spec_end = (end * 4).min(1024);

                let output = &mut channel.quant_spec[spec_start..spec_end];
                HuffmanDecoder::decode_spectral(reader, cb, output)?;
            }
        }

        Ok(())
    }

    /// Process decoded channel data to samples.
    fn process_channel(&mut self, ch: usize, channel: &ChannelData) -> Result<[f32; 1024]> {
        let mut spec = [0.0f32; 1024];

        // Inverse quantization and scalefactor application
        for g in 0..channel.ics_info.num_window_groups as usize {
            for sfb in 0..channel.ics_info.max_sfb as usize {
                let sf = channel.scalefactors[g][sfb];
                let gain = scalefactor_gain(sf);

                // Apply gain to spectral coefficients
                let start = sfb * 4; // Simplified
                let end = ((sfb + 1) * 4).min(1024);

                for i in start..end {
                    let quant = channel.quant_spec[i];
                    // Inverse quantization: sign(q) * |q|^(4/3) * gain
                    let sign = if quant < 0 { -1.0 } else { 1.0 };
                    spec[i] = sign * (quant.abs() as f32).powf(4.0 / 3.0) * gain;
                }
            }
        }

        // Apply TNS
        if let Some(ref tns_data) = channel.tns_data {
            // Simplified SFB offsets for 44.1kHz
            let sfb_offsets: Vec<u16> = (0..=50).map(|i| (i * 20).min(1024)).collect();
            self.tns[ch].apply(&mut spec, tns_data, 0, &sfb_offsets);
        }

        // IMDCT
        let mut output = [0.0f32; 1024];
        if channel.ics_info.window_sequence == WindowSequence::EightShort {
            // 8 short blocks
            let mut short_specs = [[0.0f32; 128]; 8];
            for (i, block) in short_specs.iter_mut().enumerate() {
                block.copy_from_slice(&spec[i * 128..(i + 1) * 128]);
            }
            self.imdct[ch].process_short(&short_specs, &mut output);
        } else {
            // Long block
            let mut spec_1024 = [0.0f32; 1024];
            spec_1024.copy_from_slice(&spec);
            self.imdct[ch].process_long(&spec_1024, &mut output);
        }

        Ok(output)
    }
}

impl Default for AacDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioDecoder for AacDecoder {
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
        self.sample_rate
    }

    fn channels(&self) -> u8 {
        self.channels
    }

    fn sample_format(&self) -> SampleFormat {
        SampleFormat::F32
    }

    fn set_extra_data(&mut self, data: &[u8]) -> Result<()> {
        if let Some(asc) = AudioSpecificConfig::parse(data) {
            self.init(asc)
        } else {
            Err(Error::Codec("Invalid AudioSpecificConfig".into()))
        }
    }

    fn decode(&mut self, packet: &Packet<'_>) -> Result<Vec<Sample>> {
        let buffer = self.decode_frame(packet.data())?;
        let sample = Sample::from_buffer(buffer);
        Ok(vec![sample])
    }

    fn flush(&mut self) -> Result<Vec<Sample>> {
        for imdct in &mut self.imdct {
            imdct.reset();
        }
        for tns in &mut self.tns {
            tns.reset();
        }
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        for imdct in &mut self.imdct {
            imdct.reset();
        }
        for tns in &mut self.tns {
            tns.reset();
        }
    }
}
