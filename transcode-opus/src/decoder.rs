//! Opus decoder implementation.
//!
//! This module provides the main Opus decoder that can decode SILK, CELT, or hybrid mode.

use crate::celt::{CeltBandwidth, CeltDecoder};
use crate::error::{OpusError, Result};
use crate::range_coder::RangeDecoder;
use crate::silk::{SilkDecoder, SilkSampleRate};
use crate::{Bandwidth, OpusConfig, OpusMode, MAX_PACKET_SIZE, SAMPLE_RATES};
use transcode_core::sample::{ChannelLayout, SampleBuffer, SampleFormat};

/// Opus packet Table of Contents (TOC) byte parser.
#[derive(Debug, Clone, Copy)]
pub struct OpusToc {
    /// Configuration number (0-31).
    pub config: u8,
    /// Stereo flag.
    pub stereo: bool,
    /// Frame count code (0-3).
    pub frame_count_code: u8,
}

impl OpusToc {
    /// Parse TOC byte from packet.
    pub fn parse(byte: u8) -> Self {
        Self {
            config: (byte >> 3) & 0x1F,
            stereo: (byte >> 2) & 1 != 0,
            frame_count_code: byte & 0x03,
        }
    }

    /// Get the mode from config.
    pub fn mode(&self) -> OpusMode {
        match self.config {
            0..=11 => OpusMode::Silk,
            12..=15 => OpusMode::Hybrid,
            16..=31 => OpusMode::Celt,
            _ => OpusMode::Celt,
        }
    }

    /// Get bandwidth from config.
    pub fn bandwidth(&self) -> Bandwidth {
        match self.config {
            0..=3 => Bandwidth::Narrowband,
            4..=7 => Bandwidth::Mediumband,
            8..=11 => Bandwidth::Wideband,
            12..=13 => Bandwidth::SuperWideband,
            14..=15 => Bandwidth::Fullband,
            16..=19 => Bandwidth::Narrowband,
            20..=23 => Bandwidth::Wideband,
            24..=27 => Bandwidth::SuperWideband,
            28..=31 => Bandwidth::Fullband,
            _ => Bandwidth::Fullband,
        }
    }

    /// Get frame size in samples at 48kHz.
    pub fn frame_size_48k(&self) -> usize {
        let frame_size_code = match self.config {
            0..=11 => self.config % 4,
            12..=15 => (self.config - 12) % 2 + 2, // 10ms or 20ms
            16..=31 => (self.config - 16) % 4,
            _ => 3,
        };

        match frame_size_code {
            0 => 120,  // 2.5ms
            1 => 240,  // 5ms
            2 => 480,  // 10ms
            3 => 960,  // 20ms
            _ => 960,
        }
    }

    /// Get number of frames from code.
    pub fn frame_count(&self, data: &[u8]) -> (usize, usize) {
        match self.frame_count_code {
            0 => (1, 1), // 1 frame, TOC only
            1 => (2, 1), // 2 frames, equal size
            2 => (2, 2), // 2 frames, different sizes
            3 => {
                // Arbitrary number
                if data.len() >= 2 {
                    let count = (data[1] & 0x3F) as usize;
                    let padding = if data[1] & 0x40 != 0 { 1 } else { 0 };
                    (count, 2 + padding)
                } else {
                    (0, 1)
                }
            }
            _ => (1, 1),
        }
    }

    /// Get SILK sample rate for this config.
    pub fn silk_sample_rate(&self) -> Option<SilkSampleRate> {
        match self.bandwidth() {
            Bandwidth::Narrowband => Some(SilkSampleRate::Nb8000),
            Bandwidth::Mediumband => Some(SilkSampleRate::Mb12000),
            Bandwidth::Wideband => Some(SilkSampleRate::Wb16000),
            Bandwidth::SuperWideband => Some(SilkSampleRate::Swb24000),
            Bandwidth::Fullband => None, // CELT only for fullband
        }
    }

    /// Get CELT bandwidth.
    pub fn celt_bandwidth(&self) -> CeltBandwidth {
        match self.bandwidth() {
            Bandwidth::Narrowband => CeltBandwidth::Narrow,
            Bandwidth::Mediumband => CeltBandwidth::Medium,
            Bandwidth::Wideband => CeltBandwidth::Wide,
            Bandwidth::SuperWideband => CeltBandwidth::SuperWide,
            Bandwidth::Fullband => CeltBandwidth::Full,
        }
    }
}

/// Opus decoder.
pub struct OpusDecoder {
    /// Configuration (reserved for future use).
    _config: OpusConfig,
    /// SILK decoder.
    silk_decoder: Option<SilkDecoder>,
    /// CELT decoder.
    celt_decoder: CeltDecoder,
    /// Previous mode (for mode switching).
    prev_mode: OpusMode,
    /// Output sample rate.
    output_sample_rate: u32,
    /// Number of output channels.
    channels: u8,
    /// Frame size at 48kHz.
    frame_size_48k: usize,
    /// Resampler state.
    resampler_state: ResamplerState,
    /// Consecutive packet losses.
    consecutive_losses: u32,
    /// Frame counter.
    frame_count: u64,
    /// Initialized flag.
    initialized: bool,
}

impl OpusDecoder {
    /// Create a new Opus decoder.
    pub fn new(sample_rate: u32, channels: u8) -> Result<Self> {
        // Validate sample rate
        if !SAMPLE_RATES.contains(&sample_rate) {
            return Err(OpusError::InvalidSampleRate(sample_rate));
        }

        // Validate channels
        if channels != 1 && channels != 2 {
            return Err(OpusError::InvalidChannels(channels));
        }

        let config = OpusConfig {
            sample_rate,
            channels,
            ..Default::default()
        };

        // Create CELT decoder (always needed)
        let celt_decoder = CeltDecoder::new(channels, 960);

        Ok(Self {
            _config: config,
            silk_decoder: None,
            celt_decoder,
            prev_mode: OpusMode::Celt,
            output_sample_rate: sample_rate,
            channels,
            frame_size_48k: 960,
            resampler_state: ResamplerState::new(48000, sample_rate),
            consecutive_losses: 0,
            frame_count: 0,
            initialized: true,
        })
    }

    /// Decode a single Opus packet.
    pub fn decode_packet(&mut self, data: &[u8]) -> Result<SampleBuffer> {
        if !self.initialized {
            return Err(OpusError::NotInitialized);
        }

        if data.is_empty() {
            return Err(OpusError::InvalidPacket("Empty packet".into()));
        }

        if data.len() > MAX_PACKET_SIZE {
            return Err(OpusError::InvalidPacket(format!(
                "Packet too large: {} > {}",
                data.len(),
                MAX_PACKET_SIZE
            )));
        }

        // Parse TOC byte
        let toc = OpusToc::parse(data[0]);
        let mode = toc.mode();
        let _bandwidth = toc.bandwidth();

        // Get frame count and header size
        let (frame_count, header_size) = toc.frame_count(data);

        if header_size >= data.len() {
            return Err(OpusError::InvalidPacket("Packet too short".into()));
        }

        // Get frame size
        self.frame_size_48k = toc.frame_size_48k();

        // Calculate output samples
        let output_samples = self.frame_size_48k * frame_count;
        let output_samples_resampled =
            (output_samples as u64 * self.output_sample_rate as u64 / 48000) as usize;

        // Create output buffer
        let layout = if self.channels == 1 {
            ChannelLayout::Mono
        } else {
            ChannelLayout::Stereo
        };

        let mut output = SampleBuffer::new(
            output_samples_resampled,
            SampleFormat::F32,
            layout,
            self.output_sample_rate,
        );

        // Decode frames
        let payload = &data[header_size..];
        let mut decoded_samples = Vec::with_capacity(output_samples * self.channels as usize);

        for frame_idx in 0..frame_count {
            // Get frame data
            let frame_data = self.get_frame_data(payload, frame_idx, frame_count, toc.frame_count_code)?;

            // Decode based on mode
            let frame_samples = match mode {
                OpusMode::Silk => self.decode_silk_frame(frame_data, &toc)?,
                OpusMode::Celt => self.decode_celt_frame(frame_data, &toc)?,
                OpusMode::Hybrid => self.decode_hybrid_frame(frame_data, &toc)?,
            };

            decoded_samples.extend(frame_samples);
        }

        // Resample if needed
        let final_samples = if self.output_sample_rate != 48000 {
            self.resampler_state.resample(&decoded_samples, self.channels)
        } else {
            decoded_samples
        };

        // Copy to output buffer
        self.copy_to_buffer(&final_samples, &mut output);

        self.prev_mode = mode;
        self.consecutive_losses = 0;
        self.frame_count += frame_count as u64;

        Ok(output)
    }

    /// Get frame data for a specific frame index.
    fn get_frame_data<'a>(
        &self,
        payload: &'a [u8],
        frame_idx: usize,
        frame_count: usize,
        frame_count_code: u8,
    ) -> Result<&'a [u8]> {
        if frame_count == 0 {
            return Err(OpusError::InvalidPacket("No frames".into()));
        }

        match frame_count_code {
            0 => Ok(payload), // Single frame
            1 => {
                // 2 equal frames
                let frame_size = payload.len() / 2;
                let start = frame_idx * frame_size;
                let end = start + frame_size;
                if end <= payload.len() {
                    Ok(&payload[start..end])
                } else {
                    Err(OpusError::InvalidPacket("Frame bounds exceeded".into()))
                }
            }
            2 => {
                // 2 frames, first size coded
                if payload.is_empty() {
                    return Err(OpusError::InvalidPacket("Missing frame size".into()));
                }

                let first_size = self.decode_frame_size(payload)?;
                if frame_idx == 0 {
                    Ok(&payload[1..1 + first_size])
                } else {
                    Ok(&payload[1 + first_size..])
                }
            }
            3 => {
                // Multiple frames with sizes
                self.get_vbr_frame_data(payload, frame_idx, frame_count)
            }
            _ => Err(OpusError::InvalidPacket("Invalid frame count code".into())),
        }
    }

    /// Decode frame size from VBR header.
    fn decode_frame_size(&self, data: &[u8]) -> Result<usize> {
        if data.is_empty() {
            return Err(OpusError::InvalidPacket("Missing size byte".into()));
        }

        let first_byte = data[0];
        if first_byte < 252 {
            Ok(first_byte as usize)
        } else if data.len() >= 2 {
            Ok(252 + (first_byte as usize - 252) + (data[1] as usize) * 4)
        } else {
            Err(OpusError::InvalidPacket("Incomplete size encoding".into()))
        }
    }

    /// Get frame data for VBR (code 3) packets.
    fn get_vbr_frame_data<'a>(
        &self,
        payload: &'a [u8],
        frame_idx: usize,
        frame_count: usize,
    ) -> Result<&'a [u8]> {
        if payload.is_empty() {
            return Err(OpusError::InvalidPacket("Empty VBR payload".into()));
        }

        // Skip count byte
        let mut pos = 1;
        let mut frame_sizes = Vec::with_capacity(frame_count);

        // Read frame sizes
        for _ in 0..frame_count - 1 {
            if pos >= payload.len() {
                return Err(OpusError::InvalidPacket("Missing frame size".into()));
            }

            let size = self.decode_frame_size(&payload[pos..])?;
            let size_bytes = if payload[pos] < 252 { 1 } else { 2 };
            pos += size_bytes;
            frame_sizes.push(size);
        }

        // Last frame gets remaining bytes
        let last_size = payload.len().saturating_sub(pos)
            - frame_sizes.iter().sum::<usize>();
        frame_sizes.push(last_size);

        // Get requested frame
        let mut start = pos;
        for (idx, &size) in frame_sizes.iter().enumerate() {
            if idx == frame_idx {
                return Ok(&payload[start..start + size]);
            }
            start += size;
        }

        Err(OpusError::InvalidPacket("Frame index out of range".into()))
    }

    /// Decode a SILK frame.
    fn decode_silk_frame(&mut self, data: &[u8], toc: &OpusToc) -> Result<Vec<f32>> {
        let silk_rate = toc.silk_sample_rate().ok_or_else(|| {
            OpusError::UnsupportedConfig("Invalid SILK sample rate".into())
        })?;

        // Create SILK decoder if needed
        if self.silk_decoder.is_none() {
            self.silk_decoder = Some(SilkDecoder::new(silk_rate, self.channels));
        }

        let silk = self.silk_decoder.as_mut().unwrap();

        // Create range decoder
        let mut reader = RangeDecoder::new(data)?;

        // Decode SILK frame
        let frame_size_ms = match self.frame_size_48k {
            120 => 2,  // 2.5ms
            240 => 5,
            480 => 10,
            960 => 20,
            1920 => 40,
            2880 => 60,
            _ => 20,
        };

        let samples = silk.decode_frame(&mut reader, frame_size_ms)?;

        // Upsample to 48kHz if needed
        let upsampled = self.upsample_silk(&samples, silk_rate);

        Ok(upsampled)
    }

    /// Upsample SILK output to 48kHz.
    fn upsample_silk(&self, samples: &[f32], rate: SilkSampleRate) -> Vec<f32> {
        let ratio = 48000 / (rate as u32);
        if ratio == 1 {
            return samples.to_vec();
        }

        let output_len = samples.len() * ratio as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..samples.len() - 1 {
            let s0 = samples[i];
            let s1 = samples[i + 1];

            for j in 0..ratio as usize {
                let t = j as f32 / ratio as f32;
                output.push(s0 * (1.0 - t) + s1 * t);
            }
        }

        // Last sample
        for _ in 0..ratio as usize {
            output.push(*samples.last().unwrap_or(&0.0));
        }

        output
    }

    /// Decode a CELT frame.
    fn decode_celt_frame(&mut self, data: &[u8], toc: &OpusToc) -> Result<Vec<f32>> {
        let bandwidth = toc.celt_bandwidth();

        // Create range decoder
        let mut reader = RangeDecoder::new(data)?;

        // Decode CELT frame
        self.celt_decoder.decode_frame(&mut reader, bandwidth)
    }

    /// Decode a hybrid frame (SILK + CELT).
    fn decode_hybrid_frame(&mut self, data: &[u8], toc: &OpusToc) -> Result<Vec<f32>> {
        // In hybrid mode, SILK handles lower frequencies and CELT handles higher
        // The data is split between them

        if data.is_empty() {
            return Err(OpusError::InvalidPacket("Empty hybrid frame".into()));
        }

        // Decode SILK portion (first part of data)
        let silk_samples = self.decode_silk_frame(&data[..data.len() / 2], toc)?;

        // Decode CELT portion (second part of data)
        let celt_samples = self.decode_celt_frame(&data[data.len() / 2..], toc)?;

        // Combine SILK (low frequencies) with CELT (high frequencies)
        let mut output = Vec::with_capacity(self.frame_size_48k * self.channels as usize);

        // Simple combination: SILK provides base, CELT adds high frequencies
        for i in 0..self.frame_size_48k {
            for ch in 0..self.channels as usize {
                let idx = i * self.channels as usize + ch;

                let silk_val = silk_samples.get(idx).copied().unwrap_or(0.0);
                let celt_val = celt_samples.get(idx).copied().unwrap_or(0.0);

                // Combine with crossover around 8kHz
                output.push(silk_val * 0.6 + celt_val * 0.4);
            }
        }

        Ok(output)
    }

    /// Perform packet loss concealment.
    pub fn conceal_packet_loss(&mut self) -> Result<SampleBuffer> {
        self.consecutive_losses += 1;

        let output_samples = self.frame_size_48k;
        let output_samples_resampled =
            (output_samples as u64 * self.output_sample_rate as u64 / 48000) as usize;

        let layout = if self.channels == 1 {
            ChannelLayout::Mono
        } else {
            ChannelLayout::Stereo
        };

        let mut output = SampleBuffer::new(
            output_samples_resampled,
            SampleFormat::F32,
            layout,
            self.output_sample_rate,
        );

        // Use previous mode for concealment
        let concealed = match self.prev_mode {
            OpusMode::Silk => {
                if let Some(ref mut silk) = self.silk_decoder {
                    let frame_ms = (self.frame_size_48k / 48) as u32;
                    silk.conceal_packet(frame_ms)
                } else {
                    vec![0.0; output_samples * self.channels as usize]
                }
            }
            OpusMode::Celt | OpusMode::Hybrid => self.celt_decoder.conceal_packet(),
        };

        // Resample if needed
        let final_samples = if self.output_sample_rate != 48000 {
            self.resampler_state.resample(&concealed, self.channels)
        } else {
            concealed
        };

        self.copy_to_buffer(&final_samples, &mut output);

        self.frame_count += 1;
        Ok(output)
    }

    /// Copy samples to output buffer.
    fn copy_to_buffer(&self, samples: &[f32], buffer: &mut SampleBuffer) {
        let data = buffer.data_mut();
        let bytes: &mut [f32] = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f32, data.len() / 4)
        };

        for (i, &sample) in samples.iter().take(bytes.len()).enumerate() {
            bytes[i] = sample;
        }
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.output_sample_rate
    }

    /// Get the number of channels.
    pub fn channels(&self) -> u8 {
        self.channels
    }

    /// Get the last frame size in samples.
    pub fn last_frame_size(&self) -> usize {
        (self.frame_size_48k as u64 * self.output_sample_rate as u64 / 48000) as usize
    }

    /// Reset the decoder state.
    pub fn reset(&mut self) {
        if let Some(ref mut silk) = self.silk_decoder {
            silk.reset();
        }
        self.celt_decoder.reset();
        self.resampler_state.reset();
        self.consecutive_losses = 0;
        self.frame_count = 0;
        self.prev_mode = OpusMode::Celt;
    }
}

/// Simple resampler state.
#[derive(Debug)]
struct ResamplerState {
    /// Input sample rate.
    input_rate: u32,
    /// Output sample rate.
    output_rate: u32,
    /// Previous samples for interpolation.
    prev_samples: Vec<f32>,
    /// Fractional position.
    frac_pos: f64,
}

impl ResamplerState {
    fn new(input_rate: u32, output_rate: u32) -> Self {
        Self {
            input_rate,
            output_rate,
            prev_samples: vec![0.0; 8],
            frac_pos: 0.0,
        }
    }

    fn resample(&mut self, input: &[f32], channels: u8) -> Vec<f32> {
        if self.input_rate == self.output_rate {
            return input.to_vec();
        }

        let ratio = self.input_rate as f64 / self.output_rate as f64;
        let input_samples = input.len() / channels as usize;
        let output_samples = (input_samples as f64 / ratio).ceil() as usize;

        let mut output = Vec::with_capacity(output_samples * channels as usize);

        for out_idx in 0..output_samples {
            let in_pos = out_idx as f64 * ratio + self.frac_pos;
            let in_idx = in_pos.floor() as usize;
            let frac = in_pos - in_idx as f64;

            for ch in 0..channels as usize {
                let idx0 = (in_idx * channels as usize + ch).min(input.len().saturating_sub(1));
                let idx1 = ((in_idx + 1) * channels as usize + ch).min(input.len().saturating_sub(1));

                let s0 = input.get(idx0).copied().unwrap_or(0.0);
                let s1 = input.get(idx1).copied().unwrap_or(s0);

                // Linear interpolation
                output.push(s0 * (1.0 - frac as f32) + s1 * frac as f32);
            }
        }

        // Update fractional position
        self.frac_pos = (self.frac_pos + input_samples as f64) % 1.0;

        output
    }

    fn reset(&mut self) {
        self.prev_samples.fill(0.0);
        self.frac_pos = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toc_parsing() {
        // Config 28 (fullband CELT), stereo, 1 frame
        let toc = OpusToc::parse(0b11100100);
        assert_eq!(toc.config, 28);
        assert!(toc.stereo);
        assert_eq!(toc.frame_count_code, 0);
        assert_eq!(toc.mode(), OpusMode::Celt);
    }

    #[test]
    fn test_toc_bandwidth() {
        // SILK narrowband
        let toc = OpusToc::parse(0b00000000);
        assert_eq!(toc.bandwidth(), Bandwidth::Narrowband);
        assert_eq!(toc.mode(), OpusMode::Silk);

        // CELT fullband
        let toc = OpusToc::parse(0b11111000);
        assert_eq!(toc.bandwidth(), Bandwidth::Fullband);
        assert_eq!(toc.mode(), OpusMode::Celt);
    }

    #[test]
    fn test_frame_size() {
        // 20ms frame at 48kHz = 960 samples
        let toc = OpusToc::parse(0b00011000); // Config 3
        assert_eq!(toc.frame_size_48k(), 960);
    }

    #[test]
    fn test_decoder_creation() {
        let decoder = OpusDecoder::new(48000, 2);
        assert!(decoder.is_ok());

        let decoder = decoder.unwrap();
        assert_eq!(decoder.sample_rate(), 48000);
        assert_eq!(decoder.channels(), 2);
    }

    #[test]
    fn test_invalid_sample_rate() {
        let decoder = OpusDecoder::new(44100, 2);
        assert!(decoder.is_err());
    }

    #[test]
    fn test_invalid_channels() {
        let decoder = OpusDecoder::new(48000, 5);
        assert!(decoder.is_err());
    }

    #[test]
    fn test_resampler() {
        let mut resampler = ResamplerState::new(48000, 24000);
        let input: Vec<f32> = (0..480).map(|i| (i as f32 * 0.1).sin()).collect();
        let output = resampler.resample(&input, 1);
        assert!(output.len() > 200 && output.len() < 300);
    }

    #[test]
    fn test_plc() {
        let mut decoder = OpusDecoder::new(48000, 1).unwrap();
        let result = decoder.conceal_packet_loss();
        assert!(result.is_ok());
    }
}
