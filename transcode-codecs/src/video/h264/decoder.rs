//! H.264 decoder implementation.

use std::collections::HashMap;
use transcode_core::{Frame, PixelFormat, Packet, Result, TimeBase, Timestamp};
use transcode_core::error::CodecError;
use crate::traits::{VideoDecoder, CodecInfo};
use super::nal::{NalUnit, NalUnitType, NalIterator, parse_avcc};
use super::sps::SequenceParameterSet;
use super::pps::PictureParameterSet;
use super::dpb::DecodedPictureBuffer;

/// Maximum number of SPS entries allowed per stream (H.264 spec allows 0-31).
const MAX_SPS_COUNT: usize = 32;

/// Maximum number of PPS entries allowed per stream (H.264 spec allows 0-255).
const MAX_PPS_COUNT: usize = 256;

/// H.264 decoder configuration.
#[derive(Debug, Clone)]
pub struct H264DecoderConfig {
    /// Maximum number of reference frames.
    pub max_ref_frames: u8,
    /// Enable error concealment.
    pub error_concealment: bool,
    /// Output frame buffer pool size.
    pub output_pool_size: usize,
}

impl Default for H264DecoderConfig {
    fn default() -> Self {
        Self {
            max_ref_frames: 16,
            error_concealment: true,
            output_pool_size: 8,
        }
    }
}

/// H.264 decoder.
pub struct H264Decoder {
    config: H264DecoderConfig,
    /// Parsed SPS map (indexed by sps_id).
    sps_map: HashMap<u8, SequenceParameterSet>,
    /// Parsed PPS map (indexed by pps_id).
    pps_map: HashMap<u8, PictureParameterSet>,
    /// Active SPS.
    active_sps: Option<u8>,
    /// Active PPS.
    active_pps: Option<u8>,
    /// Decoded picture buffer.
    dpb: DecodedPictureBuffer,
    /// Current picture being decoded.
    current_frame: Option<Frame>,
    /// Frame number counter.
    frame_num: u32,
    /// Picture order count.
    poc: i32,
    /// AVCC length size (0 for Annex B).
    avcc_length_size: usize,
    /// Extra data (codec configuration).
    extra_data: Vec<u8>,
    /// Time base for timestamps.
    time_base: TimeBase,
    /// Initialized flag.
    initialized: bool,
}

impl H264Decoder {
    /// Create a new H.264 decoder.
    pub fn new(config: H264DecoderConfig) -> Self {
        Self {
            config,
            sps_map: HashMap::new(),
            pps_map: HashMap::new(),
            active_sps: None,
            active_pps: None,
            dpb: DecodedPictureBuffer::new(16),
            current_frame: None,
            frame_num: 0,
            poc: 0,
            avcc_length_size: 0,
            extra_data: Vec::new(),
            time_base: TimeBase::MPEG,
            initialized: false,
        }
    }

    /// Create with default configuration.
    pub fn new_default() -> Self {
        Self::new(H264DecoderConfig::default())
    }

    /// Set the extra data (AVCC configuration).
    pub fn set_extra_data(&mut self, data: &[u8]) -> Result<()> {
        if data.len() < 7 {
            return Err(CodecError::DecoderConfig("Invalid AVCC data".into()).into());
        }

        // Parse AVCC configuration
        if data[0] != 1 {
            return Err(CodecError::DecoderConfig("Invalid AVCC version".into()).into());
        }

        self.avcc_length_size = ((data[4] & 0x03) + 1) as usize;

        // Parse SPS
        let num_sps = data[5] & 0x1F;
        let mut offset = 6;

        for _ in 0..num_sps {
            if offset + 2 > data.len() {
                return Err(CodecError::DecoderConfig("Truncated AVCC data".into()).into());
            }
            let sps_len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2;

            if offset + sps_len > data.len() {
                return Err(CodecError::DecoderConfig("Truncated SPS in AVCC".into()).into());
            }

            self.parse_nal(&data[offset..offset + sps_len])?;
            offset += sps_len;
        }

        // Parse PPS
        if offset >= data.len() {
            return Err(CodecError::DecoderConfig("Missing PPS count in AVCC".into()).into());
        }
        let num_pps = data[offset];
        offset += 1;

        for _ in 0..num_pps {
            if offset + 2 > data.len() {
                return Err(CodecError::DecoderConfig("Truncated AVCC data".into()).into());
            }
            let pps_len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2;

            if offset + pps_len > data.len() {
                return Err(CodecError::DecoderConfig("Truncated PPS in AVCC".into()).into());
            }

            self.parse_nal(&data[offset..offset + pps_len])?;
            offset += pps_len;
        }

        self.extra_data = data.to_vec();
        self.initialized = true;

        Ok(())
    }

    /// Set time base for output timestamps.
    pub fn set_time_base(&mut self, time_base: TimeBase) {
        self.time_base = time_base;
    }

    /// Parse a single NAL unit.
    fn parse_nal(&mut self, data: &[u8]) -> Result<Option<Frame>> {
        let nal = NalUnit::parse(data)?;

        match nal.nal_type {
            NalUnitType::Sps => {
                let sps = SequenceParameterSet::parse(&nal.data)?;
                let sps_id = sps.sps_id;
                // Enforce SPS count limit to prevent DoS via parameter set flooding
                if !self.sps_map.contains_key(&sps_id) && self.sps_map.len() >= MAX_SPS_COUNT {
                    return Err(CodecError::ResourceExhausted("SPS count limit exceeded".into()).into());
                }
                tracing::debug!(
                    sps_id = sps_id,
                    width = sps.width(),
                    height = sps.height(),
                    "Parsed SPS"
                );
                self.sps_map.insert(sps_id, sps);
                self.active_sps = Some(sps_id);
                Ok(None)
            }
            NalUnitType::Pps => {
                let pps = PictureParameterSet::parse(&nal.data)?;
                let pps_id = pps.pps_id;
                // Enforce PPS count limit to prevent DoS via parameter set flooding
                if !self.pps_map.contains_key(&pps_id) && self.pps_map.len() >= MAX_PPS_COUNT {
                    return Err(CodecError::ResourceExhausted("PPS count limit exceeded".into()).into());
                }
                tracing::debug!(pps_id = pps_id, "Parsed PPS");
                self.pps_map.insert(pps_id, pps);
                self.active_pps = Some(pps_id);
                Ok(None)
            }
            NalUnitType::IdrSlice | NalUnitType::Slice => {
                self.decode_slice(&nal)
            }
            NalUnitType::Sei => {
                // Parse SEI for timing info, etc.
                Ok(None)
            }
            NalUnitType::Aud => {
                // Access unit delimiter - start of new frame
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    /// Decode a slice.
    fn decode_slice(&mut self, nal: &NalUnit) -> Result<Option<Frame>> {
        let sps = self.active_sps
            .and_then(|id| self.sps_map.get(&id))
            .ok_or(CodecError::NotInitialized)?;

        let _pps = self.active_pps
            .and_then(|id| self.pps_map.get(&id))
            .ok_or(CodecError::NotInitialized)?;

        let width = sps.width();
        let height = sps.height();

        // For now, create a placeholder frame
        // Full implementation would decode the slice data
        let mut frame = Frame::new(
            width,
            height,
            PixelFormat::Yuv420p,
            self.time_base,
        );

        frame.pts = Timestamp::new(self.poc as i64, self.time_base);

        if nal.nal_type == NalUnitType::IdrSlice {
            frame.flags.insert(transcode_core::frame::FrameFlags::KEYFRAME);
            self.frame_num = 0;
            self.poc = 0;
        }

        self.poc += 2; // Assuming frame-based POC
        self.frame_num += 1;

        Ok(Some(frame))
    }

    /// Get the current width.
    pub fn width(&self) -> Option<u32> {
        self.active_sps
            .and_then(|id| self.sps_map.get(&id))
            .map(|sps| sps.width())
    }

    /// Get the current height.
    pub fn height(&self) -> Option<u32> {
        self.active_sps
            .and_then(|id| self.sps_map.get(&id))
            .map(|sps| sps.height())
    }
}

impl VideoDecoder for H264Decoder {
    fn codec_info(&self) -> CodecInfo {
        CodecInfo {
            name: "h264",
            long_name: "H.264 / AVC / MPEG-4 AVC",
            can_encode: false,
            can_decode: true,
        }
    }

    #[tracing::instrument(
        level = "trace",
        skip(self, packet),
        fields(
            codec = "h264",
            packet_size = packet.data().len(),
            pts = ?packet.pts,
        )
    )]
    fn decode(&mut self, packet: &Packet<'_>) -> Result<Vec<Frame>> {
        let mut frames = Vec::new();

        // Parse NAL units based on format
        let nals = if self.avcc_length_size > 0 {
            parse_avcc(packet.data(), self.avcc_length_size)
        } else {
            NalIterator::new(packet.data()).collect()
        };

        for nal_result in nals {
            let nal = nal_result?;

            // Process parameter sets and slices
            match nal.nal_type {
                NalUnitType::Sps => {
                    let sps = SequenceParameterSet::parse(&nal.data)?;
                    let sps_id = sps.sps_id;
                    // Enforce SPS count limit to prevent DoS via parameter set flooding
                    if !self.sps_map.contains_key(&sps_id) && self.sps_map.len() >= MAX_SPS_COUNT {
                        return Err(CodecError::ResourceExhausted("SPS count limit exceeded".into()).into());
                    }
                    self.sps_map.insert(sps_id, sps);
                    self.active_sps = Some(sps_id);
                }
                NalUnitType::Pps => {
                    let pps = PictureParameterSet::parse(&nal.data)?;
                    let pps_id = pps.pps_id;
                    // Enforce PPS count limit to prevent DoS via parameter set flooding
                    if !self.pps_map.contains_key(&pps_id) && self.pps_map.len() >= MAX_PPS_COUNT {
                        return Err(CodecError::ResourceExhausted("PPS count limit exceeded".into()).into());
                    }
                    self.pps_map.insert(pps_id, pps);
                    self.active_pps = Some(pps_id);
                }
                NalUnitType::IdrSlice | NalUnitType::Slice => {
                    if let Some(frame) = self.decode_slice(&nal)? {
                        frames.push(frame);
                    }
                }
                _ => {}
            }
        }

        Ok(frames)
    }

    fn flush(&mut self) -> Result<Vec<Frame>> {
        // Return any buffered frames from DPB
        let frames = self.dpb.flush();
        Ok(frames)
    }

    fn reset(&mut self) {
        self.dpb.clear();
        self.current_frame = None;
        self.frame_num = 0;
        self.poc = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let decoder = H264Decoder::new_default();
        assert!(decoder.width().is_none());
        assert!(decoder.height().is_none());
    }
}
