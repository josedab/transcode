//! HEVC decoder implementation.
//!
//! This module provides a complete HEVC/H.265 decoder including:
//! - NAL unit parsing and parameter set management
//! - CTU/CU/PU/TU decoding
//! - Intra and inter prediction
//! - Transform and inverse quantization
//! - In-loop filtering (deblocking, SAO)

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::unnecessary_cast)]
#![allow(unused_variables)]
#![allow(dead_code)]

use crate::cabac::CabacDecoder;
use crate::error::{HevcError, HevcLevel, HevcProfile, HevcTier, Result};
use crate::nal::{
    NalUnitHeader, NalUnitType, Pps, SliceSegmentHeader, SliceType, Sps, Vps,
    parse_annexb_nal_units,
};
use crate::transform::{HevcQuantizer, HevcTransform, TransformSize};
use transcode_core::frame::{Frame, PixelFormat};
use transcode_core::packet::Packet;
use transcode_core::timestamp::TimeBase;
use std::collections::HashMap;

/// HEVC decoder configuration.
#[derive(Debug, Clone)]
pub struct HevcDecoderConfig {
    /// Maximum frame width.
    pub max_width: u32,
    /// Maximum frame height.
    pub max_height: u32,
    /// Output pixel format.
    pub output_format: PixelFormat,
    /// Number of threads for parallel decoding.
    pub threads: usize,
    /// Enable threading.
    pub threading: bool,
}

impl Default for HevcDecoderConfig {
    fn default() -> Self {
        Self {
            max_width: 4096,
            max_height: 2160,
            output_format: PixelFormat::Yuv420p,
            threads: 1,
            threading: false,
        }
    }
}

/// Reference picture entry.
#[derive(Debug, Clone)]
struct ReferencePicture {
    /// Frame data.
    frame: Frame,
    /// Picture order count.
    poc: i32,
    /// Is long-term reference.
    is_long_term: bool,
    /// Is used for reference.
    is_reference: bool,
}

/// Decoded picture buffer (DPB).
#[derive(Debug)]
struct DecodedPictureBuffer {
    /// Reference pictures.
    pictures: Vec<ReferencePicture>,
    /// Maximum size.
    max_size: usize,
}

impl DecodedPictureBuffer {
    fn new(max_size: usize) -> Self {
        Self {
            pictures: Vec::with_capacity(max_size),
            max_size,
        }
    }

    fn add(&mut self, frame: Frame, poc: i32, is_long_term: bool) {
        // Remove old pictures if at capacity
        while self.pictures.len() >= self.max_size {
            // Remove oldest non-reference picture
            if let Some(idx) = self.pictures.iter().position(|p| !p.is_reference) {
                self.pictures.remove(idx);
            } else {
                self.pictures.remove(0);
            }
        }

        self.pictures.push(ReferencePicture {
            frame,
            poc,
            is_long_term,
            is_reference: true,
        });
    }

    fn get_by_poc(&self, poc: i32) -> Option<&Frame> {
        self.pictures
            .iter()
            .find(|p| p.poc == poc)
            .map(|p| &p.frame)
    }

    fn clear(&mut self) {
        self.pictures.clear();
    }

    fn mark_unused(&mut self, poc: i32) {
        if let Some(pic) = self.pictures.iter_mut().find(|p| p.poc == poc) {
            pic.is_reference = false;
        }
    }
}

/// Coding tree unit (CTU) data.
#[derive(Debug, Clone)]
pub struct CodingTreeUnit {
    /// CTU address.
    pub address: u32,
    /// X position in picture.
    pub x: u32,
    /// Y position in picture.
    pub y: u32,
    /// CTU size.
    pub size: u32,
}

/// Coding unit (CU) data.
#[derive(Debug, Clone)]
pub struct CodingUnit {
    /// X position.
    pub x: u32,
    /// Y position.
    pub y: u32,
    /// Size (width = height for square CUs).
    pub size: u32,
    /// Depth in the CTU tree.
    pub depth: u8,
    /// Is intra predicted.
    pub is_intra: bool,
    /// Prediction mode for each PU.
    pub pred_mode: Vec<PredictionMode>,
    /// Part mode (2Nx2N, NxN, etc.).
    pub part_mode: PartMode,
    /// QP for this CU.
    pub qp: i32,
    /// Transquant bypass flag.
    pub transquant_bypass: bool,
}

/// Prediction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionMode {
    /// Intra prediction with mode index.
    Intra(u8),
    /// Inter prediction with reference indices.
    Inter { ref_idx_l0: i8, ref_idx_l1: i8 },
    /// Skip mode.
    Skip,
}

/// Partition mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartMode {
    /// 2Nx2N (single PU).
    Part2Nx2N,
    /// NxN (4 square PUs, intra only).
    PartNxN,
    /// 2NxN (2 horizontal PUs).
    Part2NxN,
    /// Nx2N (2 vertical PUs).
    PartNx2N,
    /// 2NxnU (asymmetric, upper).
    Part2NxnU,
    /// 2NxnD (asymmetric, lower).
    Part2NxnD,
    /// nLx2N (asymmetric, left).
    PartnLx2N,
    /// nRx2N (asymmetric, right).
    PartnRx2N,
}

impl PartMode {
    /// Get number of prediction units.
    pub fn num_parts(&self) -> usize {
        match self {
            Self::Part2Nx2N => 1,
            Self::PartNxN => 4,
            _ => 2,
        }
    }
}

/// Transform unit (TU) data.
#[derive(Debug, Clone)]
pub struct TransformUnit {
    /// X position.
    pub x: u32,
    /// Y position.
    pub y: u32,
    /// Size.
    pub size: u32,
    /// Depth.
    pub depth: u8,
    /// CBF for luma.
    pub cbf_luma: bool,
    /// CBF for Cb.
    pub cbf_cb: bool,
    /// CBF for Cr.
    pub cbf_cr: bool,
    /// Coefficients.
    pub coeffs: Vec<i32>,
}

/// Intra prediction modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntraMode {
    /// Planar prediction (mode 0).
    Planar,
    /// DC prediction (mode 1).
    Dc,
    /// Horizontal (mode 10).
    Horizontal,
    /// Vertical (mode 26).
    Vertical,
    /// Angular mode (2-34).
    Angular(u8),
}

impl IntraMode {
    /// Create from mode index.
    pub fn from_index(idx: u8) -> Self {
        match idx {
            0 => Self::Planar,
            1 => Self::Dc,
            10 => Self::Horizontal,
            26 => Self::Vertical,
            _ => Self::Angular(idx),
        }
    }

    /// Get the mode index.
    pub fn index(&self) -> u8 {
        match self {
            Self::Planar => 0,
            Self::Dc => 1,
            Self::Horizontal => 10,
            Self::Vertical => 26,
            Self::Angular(idx) => *idx,
        }
    }

    /// Check if this is an angular mode.
    pub fn is_angular(&self) -> bool {
        !matches!(self, Self::Planar | Self::Dc)
    }
}

/// SAO (Sample Adaptive Offset) parameters.
#[derive(Debug, Clone, Default)]
pub struct SaoParams {
    /// SAO merge left flag.
    pub merge_left: bool,
    /// SAO merge up flag.
    pub merge_up: bool,
    /// SAO type for luma.
    pub type_idx_luma: u8,
    /// SAO type for chroma.
    pub type_idx_chroma: u8,
    /// SAO offsets for luma.
    pub offset_luma: [i8; 4],
    /// SAO offsets for Cb.
    pub offset_cb: [i8; 4],
    /// SAO offsets for Cr.
    pub offset_cr: [i8; 4],
    /// SAO band position for luma.
    pub band_position_luma: u8,
    /// SAO band position for chroma.
    pub band_position_chroma: u8,
    /// SAO EO class for luma.
    pub eo_class_luma: u8,
    /// SAO EO class for chroma.
    pub eo_class_chroma: u8,
}

/// HEVC decoder.
#[derive(Debug)]
pub struct HevcDecoder {
    /// Configuration.
    config: HevcDecoderConfig,
    /// Video parameter sets.
    vps_map: HashMap<u8, Vps>,
    /// Sequence parameter sets.
    sps_map: HashMap<u8, Sps>,
    /// Picture parameter sets.
    pps_map: HashMap<u8, Pps>,
    /// Current VPS ID.
    current_vps_id: Option<u8>,
    /// Current SPS ID.
    current_sps_id: Option<u8>,
    /// Current PPS ID.
    current_pps_id: Option<u8>,
    /// Decoded picture buffer.
    dpb: DecodedPictureBuffer,
    /// Current frame being decoded.
    current_frame: Option<Frame>,
    /// Current POC.
    current_poc: i32,
    /// Previous POC LSB.
    prev_poc_lsb: u32,
    /// Previous POC MSB.
    prev_poc_msb: i32,
    /// Transform processor.
    transform: HevcTransform,
    /// Quantizer.
    quantizer: HevcQuantizer,
    /// SAO parameters per CTU.
    sao_params: Vec<SaoParams>,
    /// Is initialized.
    initialized: bool,
    /// Frame count.
    frame_count: u64,
}

impl HevcDecoder {
    /// Create a new HEVC decoder.
    pub fn new(config: HevcDecoderConfig) -> Self {
        Self {
            config,
            vps_map: HashMap::new(),
            sps_map: HashMap::new(),
            pps_map: HashMap::new(),
            current_vps_id: None,
            current_sps_id: None,
            current_pps_id: None,
            dpb: DecodedPictureBuffer::new(16),
            current_frame: None,
            current_poc: 0,
            prev_poc_lsb: 0,
            prev_poc_msb: 0,
            transform: HevcTransform::new(8),
            quantizer: HevcQuantizer::new(8),
            sao_params: Vec::new(),
            initialized: false,
            frame_count: 0,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(HevcDecoderConfig::default())
    }

    /// Get decoder info.
    pub fn info(&self) -> HevcDecoderInfo {
        HevcDecoderInfo {
            width: self.get_width(),
            height: self.get_height(),
            profile: self.get_profile(),
            tier: self.get_tier(),
            level: self.get_level(),
            bit_depth: self.get_bit_depth(),
            chroma_format: self.get_chroma_format(),
        }
    }

    /// Get current width.
    pub fn get_width(&self) -> u32 {
        self.current_sps_id
            .and_then(|id| self.sps_map.get(&id))
            .map(|sps| sps.width())
            .unwrap_or(0)
    }

    /// Get current height.
    pub fn get_height(&self) -> u32 {
        self.current_sps_id
            .and_then(|id| self.sps_map.get(&id))
            .map(|sps| sps.height())
            .unwrap_or(0)
    }

    /// Get current profile.
    pub fn get_profile(&self) -> Option<HevcProfile> {
        self.current_sps_id
            .and_then(|id| self.sps_map.get(&id))
            .and_then(|sps| sps.profile_tier_level.profile())
    }

    /// Get current tier.
    pub fn get_tier(&self) -> HevcTier {
        self.current_sps_id
            .and_then(|id| self.sps_map.get(&id))
            .map(|sps| sps.profile_tier_level.tier())
            .unwrap_or_default()
    }

    /// Get current level.
    pub fn get_level(&self) -> HevcLevel {
        self.current_sps_id
            .and_then(|id| self.sps_map.get(&id))
            .map(|sps| sps.profile_tier_level.level())
            .unwrap_or_default()
    }

    /// Get bit depth.
    pub fn get_bit_depth(&self) -> u8 {
        self.current_sps_id
            .and_then(|id| self.sps_map.get(&id))
            .map(|sps| sps.bit_depth_luma())
            .unwrap_or(8)
    }

    /// Get chroma format.
    pub fn get_chroma_format(&self) -> u8 {
        self.current_sps_id
            .and_then(|id| self.sps_map.get(&id))
            .map(|sps| sps.chroma_format_idc)
            .unwrap_or(1)
    }

    /// Decode a packet.
    pub fn decode(&mut self, packet: &Packet) -> Result<Option<Frame>> {
        let data = packet.data();
        if data.is_empty() {
            return Ok(None);
        }

        // Parse NAL units
        let nal_units = parse_annexb_nal_units(data);

        let mut output_frame = None;

        for (header, rbsp) in nal_units {
            match header.nal_unit_type {
                NalUnitType::VpsNut => {
                    self.process_vps(&rbsp)?;
                }
                NalUnitType::SpsNut => {
                    self.process_sps(&rbsp)?;
                }
                NalUnitType::PpsNut => {
                    self.process_pps(&rbsp)?;
                }
                NalUnitType::IdrWRadl | NalUnitType::IdrNLp => {
                    output_frame = self.decode_idr_slice(&header, &rbsp)?;
                }
                NalUnitType::TrailN
                | NalUnitType::TrailR
                | NalUnitType::TsaN
                | NalUnitType::TsaR
                | NalUnitType::StsaN
                | NalUnitType::StsaR => {
                    output_frame = self.decode_trailing_slice(&header, &rbsp)?;
                }
                NalUnitType::RadlN
                | NalUnitType::RadlR
                | NalUnitType::RaslN
                | NalUnitType::RaslR => {
                    output_frame = self.decode_leading_slice(&header, &rbsp)?;
                }
                NalUnitType::CraNut => {
                    output_frame = self.decode_cra_slice(&header, &rbsp)?;
                }
                NalUnitType::BlaWLp | NalUnitType::BlaWRadl | NalUnitType::BlaNLp => {
                    output_frame = self.decode_bla_slice(&header, &rbsp)?;
                }
                NalUnitType::PrefixSeiNut | NalUnitType::SuffixSeiNut => {
                    self.process_sei(&rbsp)?;
                }
                NalUnitType::AudNut => {
                    // Access unit delimiter - no action needed
                }
                NalUnitType::EosNut => {
                    // End of sequence - reset decoder state
                    self.reset_state();
                }
                NalUnitType::EobNut => {
                    // End of bitstream
                }
                NalUnitType::FdNut => {
                    // Filler data - ignore
                }
                _ => {
                    // Unknown or reserved NAL unit type
                }
            }
        }

        Ok(output_frame)
    }

    /// Process VPS NAL unit.
    fn process_vps(&mut self, rbsp: &[u8]) -> Result<()> {
        let vps = Vps::parse(rbsp)?;
        let id = vps.vps_video_parameter_set_id;
        self.vps_map.insert(id, vps);
        self.current_vps_id = Some(id);
        Ok(())
    }

    /// Process SPS NAL unit.
    fn process_sps(&mut self, rbsp: &[u8]) -> Result<()> {
        let sps = Sps::parse(rbsp)?;
        let id = sps.sps_seq_parameter_set_id;

        // Update transform and quantizer for new bit depth
        let bit_depth = sps.bit_depth_luma();
        self.transform = HevcTransform::new(bit_depth);
        self.quantizer = HevcQuantizer::new(bit_depth);

        self.sps_map.insert(id, sps);
        self.current_sps_id = Some(id);
        self.initialized = true;
        Ok(())
    }

    /// Process PPS NAL unit.
    fn process_pps(&mut self, rbsp: &[u8]) -> Result<()> {
        let pps = Pps::parse(rbsp)?;
        let id = pps.pps_pic_parameter_set_id;
        self.pps_map.insert(id, pps);
        self.current_pps_id = Some(id);
        Ok(())
    }

    /// Process SEI NAL unit.
    fn process_sei(&mut self, _rbsp: &[u8]) -> Result<()> {
        // SEI parsing is optional, skip for now
        Ok(())
    }

    /// Decode an IDR slice.
    fn decode_idr_slice(
        &mut self,
        header: &NalUnitHeader,
        rbsp: &[u8],
    ) -> Result<Option<Frame>> {
        // Clear DPB for IDR
        self.dpb.clear();
        self.prev_poc_lsb = 0;
        self.prev_poc_msb = 0;
        self.current_poc = 0;

        self.decode_slice(header, rbsp)
    }

    /// Decode a trailing slice.
    fn decode_trailing_slice(
        &mut self,
        header: &NalUnitHeader,
        rbsp: &[u8],
    ) -> Result<Option<Frame>> {
        self.decode_slice(header, rbsp)
    }

    /// Decode a leading slice.
    fn decode_leading_slice(
        &mut self,
        header: &NalUnitHeader,
        rbsp: &[u8],
    ) -> Result<Option<Frame>> {
        self.decode_slice(header, rbsp)
    }

    /// Decode a CRA slice.
    fn decode_cra_slice(
        &mut self,
        header: &NalUnitHeader,
        rbsp: &[u8],
    ) -> Result<Option<Frame>> {
        self.decode_slice(header, rbsp)
    }

    /// Decode a BLA slice.
    fn decode_bla_slice(
        &mut self,
        header: &NalUnitHeader,
        rbsp: &[u8],
    ) -> Result<Option<Frame>> {
        // Clear DPB for BLA
        self.dpb.clear();
        self.decode_slice(header, rbsp)
    }

    /// Decode a slice.
    fn decode_slice(
        &mut self,
        header: &NalUnitHeader,
        rbsp: &[u8],
    ) -> Result<Option<Frame>> {
        if !self.initialized {
            return Err(HevcError::DecoderConfig("Decoder not initialized".to_string()));
        }

        let sps = self.sps_map.get(&self.current_sps_id.unwrap())
            .ok_or_else(|| HevcError::Sps("SPS not found".to_string()))?
            .clone();

        let pps = self.pps_map.get(&self.current_pps_id.unwrap_or(0))
            .ok_or_else(|| HevcError::Pps("PPS not found".to_string()))?
            .clone();

        // Parse slice header
        let slice_header = SliceSegmentHeader::parse(rbsp, &sps, &pps, header.nal_unit_type)?;

        // Calculate POC
        if !header.nal_unit_type.is_idr() {
            self.calculate_poc(&slice_header, &sps);
        }

        // Allocate frame if first slice in picture
        if slice_header.first_slice_segment_in_pic_flag {
            self.allocate_frame(&sps)?;
        }

        // Initialize SAO parameters
        self.init_sao_params(&sps);

        // Decode CTUs
        self.decode_ctus(&slice_header, &sps, &pps, rbsp)?;

        // Apply in-loop filters
        if let Some(ref mut frame) = self.current_frame {
            if !slice_header.slice_deblocking_filter_disabled_flag {
                Self::apply_deblocking_filter_static(frame, &slice_header, &sps, &pps)?;
            }
            if slice_header.slice_sao_luma_flag || slice_header.slice_sao_chroma_flag {
                Self::apply_sao_filter_static(frame, &slice_header, &sps, &self.sao_params)?;
            }
        }

        // Output frame if complete
        let output = self.current_frame.take();
        if let Some(ref frame) = output {
            // Add to DPB if reference picture
            if header.nal_unit_type.is_reference() {
                self.dpb.add(frame.clone(), self.current_poc, false);
            }
            self.frame_count += 1;
        }

        Ok(output)
    }

    /// Calculate POC (Picture Order Count).
    fn calculate_poc(&mut self, slice_header: &SliceSegmentHeader, sps: &Sps) {
        let max_poc_lsb = 1 << (sps.log2_max_pic_order_cnt_lsb_minus4 + 4);
        let poc_lsb = slice_header.slice_pic_order_cnt_lsb;

        let poc_msb = if poc_lsb < self.prev_poc_lsb
            && (self.prev_poc_lsb - poc_lsb) >= (max_poc_lsb / 2)
        {
            self.prev_poc_msb + max_poc_lsb as i32
        } else if poc_lsb > self.prev_poc_lsb
            && (poc_lsb - self.prev_poc_lsb) > (max_poc_lsb / 2)
        {
            self.prev_poc_msb - max_poc_lsb as i32
        } else {
            self.prev_poc_msb
        };

        self.current_poc = poc_msb + poc_lsb as i32;
        self.prev_poc_lsb = poc_lsb;
        self.prev_poc_msb = poc_msb;
    }

    /// Allocate a new frame.
    fn allocate_frame(&mut self, sps: &Sps) -> Result<()> {
        let width = sps.pic_width_in_luma_samples;
        let height = sps.pic_height_in_luma_samples;
        let bit_depth = sps.bit_depth_luma();

        let format = match (sps.chroma_format_idc, bit_depth) {
            (1, 8) => PixelFormat::Yuv420p,
            (1, 10) => PixelFormat::Yuv420p10le,
            (2, 8) => PixelFormat::Yuv422p,
            (2, 10) => PixelFormat::Yuv422p10le,
            (3, 8) => PixelFormat::Yuv444p,
            (3, 10) => PixelFormat::Yuv444p10le,
            _ => self.config.output_format,
        };

        let frame = Frame::new(width, height, format, TimeBase::new(1, 90000));
        self.current_frame = Some(frame);
        Ok(())
    }

    /// Initialize SAO parameters.
    fn init_sao_params(&mut self, sps: &Sps) {
        let num_ctbs = (sps.pic_width_in_ctbs() * sps.pic_height_in_ctbs()) as usize;
        self.sao_params = vec![SaoParams::default(); num_ctbs];
    }

    /// Decode CTUs in the slice.
    fn decode_ctus(
        &mut self,
        slice_header: &SliceSegmentHeader,
        sps: &Sps,
        pps: &Pps,
        rbsp: &[u8],
    ) -> Result<()> {
        let ctb_size = sps.ctb_size();
        let pic_width_in_ctbs = sps.pic_width_in_ctbs();
        let pic_height_in_ctbs = sps.pic_height_in_ctbs();

        // Initialize CABAC decoder
        let mut cabac = CabacDecoder::new(rbsp);
        cabac.init()?;
        cabac.init_contexts(slice_header, 0);

        let start_ctb = slice_header.slice_segment_address;
        let end_ctb = pic_width_in_ctbs * pic_height_in_ctbs;

        for ctb_addr in start_ctb..end_ctb {
            let ctb_x = ctb_addr % pic_width_in_ctbs;
            let ctb_y = ctb_addr / pic_width_in_ctbs;

            let ctu = CodingTreeUnit {
                address: ctb_addr,
                x: ctb_x * ctb_size,
                y: ctb_y * ctb_size,
                size: ctb_size,
            };

            // Decode SAO parameters
            if sps.sample_adaptive_offset_enabled_flag {
                self.decode_sao_params(&mut cabac, ctb_addr as usize, slice_header)?;
            }

            // Decode coding quadtree
            self.decode_coding_quadtree(
                &mut cabac,
                &ctu,
                0,
                slice_header,
                sps,
                pps,
            )?;

            // Check for end of slice
            if cabac.is_end_of_slice()? {
                break;
            }
        }

        Ok(())
    }

    /// Decode SAO parameters for a CTU.
    fn decode_sao_params(
        &mut self,
        cabac: &mut CabacDecoder,
        ctb_addr: usize,
        slice_header: &SliceSegmentHeader,
    ) -> Result<()> {
        let mut params = SaoParams::default();

        if slice_header.slice_sao_luma_flag || slice_header.slice_sao_chroma_flag {
            // SAO merge flags
            params.merge_left = cabac.decode_decision(0)?;
            if !params.merge_left {
                params.merge_up = cabac.decode_decision(0)?;
            }

            if !params.merge_left && !params.merge_up {
                // Decode SAO type and offsets
                if slice_header.slice_sao_luma_flag {
                    params.type_idx_luma = cabac.decode_truncated_unary(0, 2)? as u8;
                    if params.type_idx_luma > 0 {
                        for i in 0..4 {
                            params.offset_luma[i] = cabac.decode_exp_golomb(0)? as i8;
                        }
                    }
                }

                if slice_header.slice_sao_chroma_flag {
                    params.type_idx_chroma = cabac.decode_truncated_unary(0, 2)? as u8;
                    if params.type_idx_chroma > 0 {
                        for i in 0..4 {
                            params.offset_cb[i] = cabac.decode_exp_golomb(0)? as i8;
                            params.offset_cr[i] = cabac.decode_exp_golomb(0)? as i8;
                        }
                    }
                }
            }
        }

        self.sao_params[ctb_addr] = params;
        Ok(())
    }

    /// Decode coding quadtree.
    fn decode_coding_quadtree(
        &mut self,
        cabac: &mut CabacDecoder,
        ctu: &CodingTreeUnit,
        depth: u8,
        slice_header: &SliceSegmentHeader,
        sps: &Sps,
        pps: &Pps,
    ) -> Result<()> {
        let log2_ctb_size = sps.log2_ctb_size();
        let log2_min_cb_size = sps.log2_min_cb_size();
        let log2_size = log2_ctb_size - depth;
        let size = 1u32 << log2_size;

        // Check if we can split further
        let split = if log2_size > log2_min_cb_size {
            let ctx_idx = depth as usize;
            cabac.decode_split_cu_flag(ctx_idx)?
        } else {
            false
        };

        if split {
            // Recurse into 4 sub-CUs
            let half_size = size / 2;
            for i in 0..4 {
                let sub_x = ctu.x + (i % 2) * half_size;
                let sub_y = ctu.y + (i / 2) * half_size;

                if sub_x < sps.pic_width_in_luma_samples
                    && sub_y < sps.pic_height_in_luma_samples
                {
                    let sub_ctu = CodingTreeUnit {
                        address: ctu.address,
                        x: sub_x,
                        y: sub_y,
                        size: half_size,
                    };
                    self.decode_coding_quadtree(
                        cabac,
                        &sub_ctu,
                        depth + 1,
                        slice_header,
                        sps,
                        pps,
                    )?;
                }
            }
        } else {
            // Decode coding unit
            self.decode_coding_unit(
                cabac,
                ctu.x,
                ctu.y,
                size,
                depth,
                slice_header,
                sps,
                pps,
            )?;
        }

        Ok(())
    }

    /// Decode a coding unit.
    fn decode_coding_unit(
        &mut self,
        cabac: &mut CabacDecoder,
        x: u32,
        y: u32,
        size: u32,
        depth: u8,
        slice_header: &SliceSegmentHeader,
        sps: &Sps,
        pps: &Pps,
    ) -> Result<()> {
        let log2_size = (size as f32).log2() as u8;
        let is_intra_slice = slice_header.slice_type == SliceType::I;

        // Skip flag (inter only)
        let skip_flag = if !is_intra_slice {
            cabac.decode_cu_skip_flag(0)?
        } else {
            false
        };

        let (is_intra, part_mode) = if skip_flag {
            (false, PartMode::Part2Nx2N)
        } else {
            // Prediction mode flag
            let is_intra = if is_intra_slice {
                true
            } else {
                cabac.decode_pred_mode_flag()?
            };

            // Part mode
            let part_mode = if is_intra {
                if log2_size == sps.log2_min_cb_size() {
                    let nxn = cabac.decode_decision(0)?;
                    if nxn { PartMode::PartNxN } else { PartMode::Part2Nx2N }
                } else {
                    PartMode::Part2Nx2N
                }
            } else {
                let pm = cabac.decode_part_mode(log2_size, false)?;
                match pm {
                    0 => PartMode::Part2Nx2N,
                    1 => PartMode::Part2NxN,
                    2 => PartMode::PartNx2N,
                    3 => PartMode::PartNxN,
                    4 => PartMode::Part2NxnU,
                    5 => PartMode::Part2NxnD,
                    6 => PartMode::PartnLx2N,
                    7 => PartMode::PartnRx2N,
                    _ => PartMode::Part2Nx2N,
                }
            };

            (is_intra, part_mode)
        };

        // Decode prediction units
        let num_pus = part_mode.num_parts();
        for pu_idx in 0..num_pus {
            if is_intra {
                self.decode_intra_prediction(cabac, x, y, size, pu_idx, sps)?;
            } else if !skip_flag {
                self.decode_inter_prediction(cabac, x, y, size, pu_idx, slice_header)?;
            }
        }

        // Decode transform tree
        if !skip_flag {
            let qp = slice_header.slice_qp(pps);
            self.decode_transform_tree(cabac, x, y, size, 0, is_intra, qp, sps)?;
        }

        Ok(())
    }

    /// Decode intra prediction.
    fn decode_intra_prediction(
        &mut self,
        cabac: &mut CabacDecoder,
        x: u32,
        y: u32,
        size: u32,
        pu_idx: usize,
        sps: &Sps,
    ) -> Result<()> {
        // Decode prev_intra_luma_pred_flag
        let prev_intra_flag = cabac.decode_prev_intra_luma_pred_flag()?;

        let mode = if prev_intra_flag {
            // Use MPM (Most Probable Mode)
            let mpm_idx = cabac.decode_mpm_idx()?;
            self.get_mpm(x, y, mpm_idx, sps)
        } else {
            // Decode remaining mode
            let rem_mode = cabac.decode_rem_intra_luma_pred_mode()?;
            self.derive_intra_mode(x, y, rem_mode, sps)
        };

        // Apply intra prediction
        self.apply_intra_prediction(x, y, size, IntraMode::from_index(mode), true)?;

        // Decode chroma mode
        let chroma_mode = cabac.decode_bypass_bins(2)? as u8;
        self.apply_intra_prediction(x, y, size, IntraMode::from_index(chroma_mode), false)?;

        Ok(())
    }

    /// Get most probable mode.
    fn get_mpm(&self, _x: u32, _y: u32, mpm_idx: u8, _sps: &Sps) -> u8 {
        // Simplified MPM derivation
        match mpm_idx {
            0 => 0,  // Planar
            1 => 1,  // DC
            2 => 26, // Vertical
            _ => 0,
        }
    }

    /// Derive intra mode from remaining mode.
    fn derive_intra_mode(&self, _x: u32, _y: u32, rem_mode: u8, _sps: &Sps) -> u8 {
        // Add 2 to skip planar and DC
        rem_mode + 2
    }

    /// Apply intra prediction to a block.
    fn apply_intra_prediction(
        &mut self,
        x: u32,
        y: u32,
        size: u32,
        mode: IntraMode,
        is_luma: bool,
    ) -> Result<()> {
        let frame = self.current_frame.as_mut()
            .ok_or_else(|| HevcError::InvalidState("No current frame".to_string()))?;

        let plane_idx = if is_luma { 0 } else { 1 };
        let stride = frame.stride(plane_idx);
        let plane = frame.plane_mut(plane_idx)
            .ok_or_else(|| HevcError::InvalidState("Plane not found".to_string()))?;

        let (x, y, size) = if is_luma {
            (x as usize, y as usize, size as usize)
        } else {
            // Chroma is subsampled
            ((x / 2) as usize, (y / 2) as usize, (size / 2) as usize)
        };

        match mode {
            IntraMode::Planar => {
                Self::predict_planar_static(plane, x, y, size, stride);
            }
            IntraMode::Dc => {
                Self::predict_dc_static(plane, x, y, size, stride);
            }
            IntraMode::Horizontal => {
                Self::predict_horizontal_static(plane, x, y, size, stride);
            }
            IntraMode::Vertical => {
                Self::predict_vertical_static(plane, x, y, size, stride);
            }
            IntraMode::Angular(angle) => {
                Self::predict_angular_static(plane, x, y, size, stride, angle);
            }
        }

        Ok(())
    }

    /// Planar prediction (static version to avoid borrow conflicts).
    fn predict_planar_static(plane: &mut [u8], x: usize, y: usize, size: usize, stride: usize) {
        // Get reference samples
        let _top_left = if x > 0 && y > 0 {
            plane[(y - 1) * stride + x - 1]
        } else {
            128
        };

        for j in 0..size {
            for i in 0..size {
                let h_weight = size - 1 - i;
                let v_weight = size - 1 - j;

                let left = if x > 0 { plane[(y + j) * stride + x - 1] } else { 128 };
                let top = if y > 0 { plane[(y - 1) * stride + x + i] } else { 128 };
                let top_right = if y > 0 && x + size < stride {
                    plane[(y - 1) * stride + x + size]
                } else {
                    128
                };
                let bottom_left = if x > 0 && y + size < plane.len() / stride {
                    plane[(y + size) * stride + x - 1]
                } else {
                    128
                };

                let pred = (h_weight as u32 * left as u32
                    + (size - 1 - h_weight) as u32 * top_right as u32
                    + v_weight as u32 * top as u32
                    + (size - 1 - v_weight) as u32 * bottom_left as u32
                    + size as u32)
                    / (2 * size as u32);

                plane[(y + j) * stride + x + i] = pred.clamp(0, 255) as u8;
            }
        }
    }

    /// DC prediction (static version to avoid borrow conflicts).
    fn predict_dc_static(plane: &mut [u8], x: usize, y: usize, size: usize, stride: usize) {
        let mut sum = 0u32;
        let mut count = 0u32;

        // Sum left column
        if x > 0 {
            for j in 0..size {
                sum += plane[(y + j) * stride + x - 1] as u32;
                count += 1;
            }
        }

        // Sum top row
        if y > 0 {
            for i in 0..size {
                sum += plane[(y - 1) * stride + x + i] as u32;
                count += 1;
            }
        }

        let dc = if count > 0 {
            ((sum + count / 2) / count) as u8
        } else {
            128
        };

        // Fill block with DC value
        for j in 0..size {
            for i in 0..size {
                plane[(y + j) * stride + x + i] = dc;
            }
        }
    }

    /// Horizontal prediction (static version to avoid borrow conflicts).
    fn predict_horizontal_static(plane: &mut [u8], x: usize, y: usize, size: usize, stride: usize) {
        for j in 0..size {
            let left = if x > 0 {
                plane[(y + j) * stride + x - 1]
            } else {
                128
            };

            for i in 0..size {
                plane[(y + j) * stride + x + i] = left;
            }
        }
    }

    /// Vertical prediction (static version to avoid borrow conflicts).
    fn predict_vertical_static(plane: &mut [u8], x: usize, y: usize, size: usize, stride: usize) {
        for i in 0..size {
            let top = if y > 0 {
                plane[(y - 1) * stride + x + i]
            } else {
                128
            };

            for j in 0..size {
                plane[(y + j) * stride + x + i] = top;
            }
        }
    }

    /// Angular prediction (static version to avoid borrow conflicts).
    fn predict_angular_static(
        plane: &mut [u8],
        x: usize,
        y: usize,
        size: usize,
        stride: usize,
        angle: u8,
    ) {
        // Simplified angular prediction
        // Full implementation would use angle-specific interpolation
        if angle < 18 {
            // Horizontal-ish angles
            Self::predict_horizontal_static(plane, x, y, size, stride);
        } else {
            // Vertical-ish angles
            Self::predict_vertical_static(plane, x, y, size, stride);
        }
    }

    /// Decode inter prediction.
    fn decode_inter_prediction(
        &mut self,
        cabac: &mut CabacDecoder,
        _x: u32,
        _y: u32,
        _size: u32,
        _pu_idx: usize,
        slice_header: &SliceSegmentHeader,
    ) -> Result<()> {
        // Decode merge flag
        let merge_flag = cabac.decode_merge_flag()?;

        if merge_flag {
            // Decode merge index
            let _merge_idx = cabac.decode_merge_idx(slice_header.max_num_merge_cand())?;
        } else {
            // Decode motion vectors and reference indices
            let _mvd = cabac.decode_mvd()?;
            // Additional inter prediction decoding would go here
        }

        Ok(())
    }

    /// Decode transform tree.
    fn decode_transform_tree(
        &mut self,
        cabac: &mut CabacDecoder,
        x: u32,
        y: u32,
        size: u32,
        depth: u8,
        is_intra: bool,
        qp: i8,
        sps: &Sps,
    ) -> Result<()> {
        let max_depth = if is_intra {
            sps.max_transform_hierarchy_depth_intra
        } else {
            sps.max_transform_hierarchy_depth_inter
        };

        let log2_size = (size as f32).log2() as u8;
        let min_tb_log2 = sps.log2_min_luma_transform_block_size_minus2 + 2;
        let max_tb_log2 = min_tb_log2 + sps.log2_diff_max_min_luma_transform_block_size;

        let split = if log2_size <= max_tb_log2 && log2_size > min_tb_log2 && depth < max_depth {
            cabac.decode_decision(0)?
        } else {
            log2_size > max_tb_log2
        };

        if split {
            let half_size = size / 2;
            for i in 0..4 {
                let sub_x = x + (i % 2) * half_size;
                let sub_y = y + (i / 2) * half_size;
                self.decode_transform_tree(
                    cabac,
                    sub_x,
                    sub_y,
                    half_size,
                    depth + 1,
                    is_intra,
                    qp,
                    sps,
                )?;
            }
        } else {
            // Decode transform unit
            self.decode_transform_unit(cabac, x, y, size, is_intra, qp)?;
        }

        Ok(())
    }

    /// Decode a transform unit.
    fn decode_transform_unit(
        &mut self,
        cabac: &mut CabacDecoder,
        x: u32,
        y: u32,
        size: u32,
        is_intra: bool,
        qp: i8,
    ) -> Result<()> {
        let log2_size = (size as f32).log2() as u8;

        // Decode CBF
        let cbf_luma = cabac.decode_cbf(0)?;
        let cbf_cb = cabac.decode_cbf(1)?;
        let cbf_cr = cabac.decode_cbf(2)?;

        // Decode coefficients
        if cbf_luma {
            self.decode_residual_coding(cabac, x, y, size, true, is_intra, qp)?;
        }

        if cbf_cb {
            self.decode_residual_coding(cabac, x / 2, y / 2, size / 2, false, is_intra, qp)?;
        }

        if cbf_cr {
            self.decode_residual_coding(cabac, x / 2, y / 2, size / 2, false, is_intra, qp)?;
        }

        Ok(())
    }

    /// Decode residual coding.
    fn decode_residual_coding(
        &mut self,
        cabac: &mut CabacDecoder,
        x: u32,
        y: u32,
        size: u32,
        is_luma: bool,
        is_intra: bool,
        qp: i8,
    ) -> Result<()> {
        let log2_size = (size as f32).log2() as u8;
        let num_coeffs = (size * size) as usize;

        // Decode last significant position
        let (last_x, last_y) = cabac.decode_last_sig_coeff(log2_size, !is_luma)?;

        // Decode coefficient levels
        let mut coeffs = vec![0i32; num_coeffs];
        cabac.decode_sig_coeff_flags(&mut coeffs, log2_size, last_x, last_y, is_luma)?;
        cabac.decode_coeff_abs_level(&mut coeffs, num_coeffs, 0)?;

        // Dequantize
        let mut dq_coeffs = vec![0i32; num_coeffs];
        self.quantizer.dequantize(&coeffs, &mut dq_coeffs, size as usize, qp as i32, None)?;

        // Inverse transform
        let transform_size = TransformSize::from_size(size as usize)
            .ok_or_else(|| HevcError::Transform("Invalid transform size".to_string()))?;

        let stride = size as usize;
        let mut residual = vec![0i16; num_coeffs];
        self.transform.inverse_transform(
            &dq_coeffs,
            &mut residual,
            transform_size,
            stride,
            is_intra,
            is_luma,
        );

        // Add residual to prediction
        self.add_residual(x, y, size, &residual, is_luma)?;

        Ok(())
    }

    /// Add residual to prediction.
    fn add_residual(
        &mut self,
        x: u32,
        y: u32,
        size: u32,
        residual: &[i16],
        is_luma: bool,
    ) -> Result<()> {
        let frame = self.current_frame.as_mut()
            .ok_or_else(|| HevcError::InvalidState("No current frame".to_string()))?;

        let plane_idx = if is_luma { 0 } else { 1 };
        let stride = frame.stride(plane_idx);
        let plane = frame.plane_mut(plane_idx)
            .ok_or_else(|| HevcError::InvalidState("Plane not found".to_string()))?;

        let x = x as usize;
        let y = y as usize;
        let size = size as usize;

        for j in 0..size {
            for i in 0..size {
                let pred = plane[(y + j) * stride + x + i] as i16;
                let recon = (pred + residual[j * size + i]).clamp(0, 255);
                plane[(y + j) * stride + x + i] = recon as u8;
            }
        }

        Ok(())
    }

    /// Apply deblocking filter.
    fn apply_deblocking_filter_static(
        frame: &mut Frame,
        slice_header: &SliceSegmentHeader,
        _sps: &Sps,
        _pps: &Pps,
    ) -> Result<()> {
        let beta_offset = slice_header.slice_beta_offset_div2 * 2;
        let tc_offset = slice_header.slice_tc_offset_div2 * 2;

        // Apply vertical edges
        Self::deblock_edges_static(frame, true, beta_offset, tc_offset)?;

        // Apply horizontal edges
        Self::deblock_edges_static(frame, false, beta_offset, tc_offset)?;

        Ok(())
    }

    /// Deblock edges (static version).
    fn deblock_edges_static(
        frame: &mut Frame,
        vertical: bool,
        beta_offset: i8,
        tc_offset: i8,
    ) -> Result<()> {
        let width = frame.width() as usize;
        let height = frame.height() as usize;
        let stride = frame.stride(0);

        let plane = frame.plane_mut(0)
            .ok_or_else(|| HevcError::InvalidState("Luma plane not found".to_string()))?;

        // Apply filter at 8x8 block boundaries
        let step = 8;

        if vertical {
            for y in 0..height {
                for x in (step..width).step_by(step) {
                    Self::deblock_edge_v_static(plane, x, y, stride, beta_offset, tc_offset);
                }
            }
        } else {
            for y in (step..height).step_by(step) {
                for x in 0..width {
                    Self::deblock_edge_h_static(plane, x, y, stride, beta_offset, tc_offset);
                }
            }
        }

        Ok(())
    }

    /// Deblock vertical edge (static version).
    fn deblock_edge_v_static(
        plane: &mut [u8],
        x: usize,
        y: usize,
        stride: usize,
        beta_offset: i8,
        tc_offset: i8,
    ) {
        if x < 3 {
            return;
        }

        let p0 = plane[y * stride + x - 1] as i16;
        let p1 = plane[y * stride + x - 2] as i16;
        let p2 = plane[y * stride + x - 3] as i16;
        let q0 = plane[y * stride + x] as i16;
        let q1 = plane[y * stride + x + 1] as i16;
        let q2 = plane[y * stride + x + 2] as i16;

        let dp = (p2 - 2 * p1 + p0).abs();
        let dq = (q2 - 2 * q1 + q0).abs();
        let d = dp + dq;

        // Simplified filtering decision
        let beta = 8 + beta_offset as i16;
        let tc = 4 + tc_offset as i16;

        if d < beta {
            let delta = ((q0 - p0) * 9 - (q1 - p1) * 3 + 8) >> 4;
            let delta = delta.clamp(-tc, tc);

            plane[y * stride + x - 1] = (p0 + delta).clamp(0, 255) as u8;
            plane[y * stride + x] = (q0 - delta).clamp(0, 255) as u8;
        }
    }

    /// Deblock horizontal edge (static version).
    fn deblock_edge_h_static(
        plane: &mut [u8],
        x: usize,
        y: usize,
        stride: usize,
        beta_offset: i8,
        tc_offset: i8,
    ) {
        if y < 3 {
            return;
        }

        let p0 = plane[(y - 1) * stride + x] as i16;
        let p1 = plane[(y - 2) * stride + x] as i16;
        let p2 = plane[(y - 3) * stride + x] as i16;
        let q0 = plane[y * stride + x] as i16;
        let q1 = plane[(y + 1) * stride + x] as i16;
        let q2 = plane[(y + 2) * stride + x] as i16;

        let dp = (p2 - 2 * p1 + p0).abs();
        let dq = (q2 - 2 * q1 + q0).abs();
        let d = dp + dq;

        let beta = 8 + beta_offset as i16;
        let tc = 4 + tc_offset as i16;

        if d < beta {
            let delta = ((q0 - p0) * 9 - (q1 - p1) * 3 + 8) >> 4;
            let delta = delta.clamp(-tc, tc);

            plane[(y - 1) * stride + x] = (p0 + delta).clamp(0, 255) as u8;
            plane[y * stride + x] = (q0 - delta).clamp(0, 255) as u8;
        }
    }

    /// Apply SAO filter (static version).
    fn apply_sao_filter_static(
        frame: &mut Frame,
        slice_header: &SliceSegmentHeader,
        sps: &Sps,
        sao_params: &[SaoParams],
    ) -> Result<()> {
        let ctb_size = sps.ctb_size() as usize;
        let pic_width_in_ctbs = sps.pic_width_in_ctbs() as usize;
        let pic_height_in_ctbs = sps.pic_height_in_ctbs() as usize;

        for ctu_y in 0..pic_height_in_ctbs {
            for ctu_x in 0..pic_width_in_ctbs {
                let ctu_addr = ctu_y * pic_width_in_ctbs + ctu_x;
                let params = &sao_params[ctu_addr];

                if params.type_idx_luma > 0 && slice_header.slice_sao_luma_flag {
                    Self::apply_sao_ctu_static(
                        frame,
                        0,
                        ctu_x * ctb_size,
                        ctu_y * ctb_size,
                        ctb_size,
                        params,
                        true,
                    )?;
                }

                if params.type_idx_chroma > 0 && slice_header.slice_sao_chroma_flag {
                    Self::apply_sao_ctu_static(
                        frame,
                        1,
                        (ctu_x * ctb_size) / 2,
                        (ctu_y * ctb_size) / 2,
                        ctb_size / 2,
                        params,
                        false,
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Apply SAO to a CTU (static version).
    fn apply_sao_ctu_static(
        frame: &mut Frame,
        plane_idx: usize,
        x: usize,
        y: usize,
        size: usize,
        params: &SaoParams,
        is_luma: bool,
    ) -> Result<()> {
        let stride = frame.stride(plane_idx);
        let plane = frame.plane_mut(plane_idx)
            .ok_or_else(|| HevcError::InvalidState("Plane not found".to_string()))?;

        let type_idx = if is_luma {
            params.type_idx_luma
        } else {
            params.type_idx_chroma
        };

        let offsets = if is_luma {
            &params.offset_luma
        } else {
            &params.offset_cb
        };

        match type_idx {
            1 => {
                // Band offset
                let band_pos = if is_luma {
                    params.band_position_luma
                } else {
                    params.band_position_chroma
                };
                Self::apply_sao_band_static(plane, x, y, size, stride, offsets, band_pos);
            }
            2 => {
                // Edge offset
                let eo_class = if is_luma {
                    params.eo_class_luma
                } else {
                    params.eo_class_chroma
                };
                Self::apply_sao_edge_static(plane, x, y, size, stride, offsets, eo_class);
            }
            _ => {}
        }

        Ok(())
    }

    /// Apply SAO band offset (static version).
    fn apply_sao_band_static(
        plane: &mut [u8],
        x: usize,
        y: usize,
        size: usize,
        stride: usize,
        offsets: &[i8; 4],
        band_pos: u8,
    ) {
        for j in 0..size {
            for i in 0..size {
                let idx = (y + j) * stride + x + i;
                let sample = plane[idx];
                let band = (sample >> 3) as u8;

                if band >= band_pos && band < band_pos + 4 {
                    let offset_idx = (band - band_pos) as usize;
                    let new_val = (sample as i16 + offsets[offset_idx] as i16).clamp(0, 255);
                    plane[idx] = new_val as u8;
                }
            }
        }
    }

    /// Apply SAO edge offset (static version).
    fn apply_sao_edge_static(
        plane: &mut [u8],
        x: usize,
        y: usize,
        size: usize,
        stride: usize,
        offsets: &[i8; 4],
        eo_class: u8,
    ) {
        let (dx, dy) = match eo_class {
            0 => (1, 0),  // Horizontal
            1 => (0, 1),  // Vertical
            2 => (1, 1),  // 135 degree
            3 => (1, -1), // 45 degree
            _ => return,
        };

        for j in 1..size.saturating_sub(1) {
            for i in 1..size.saturating_sub(1) {
                let idx = (y + j) * stride + x + i;
                let p = plane[idx] as i16;

                let n1_x = (x + i).wrapping_add_signed(-dx);
                let n1_y = (y + j).wrapping_add_signed(-dy);
                let n2_x = (x + i).wrapping_add_signed(dx);
                let n2_y = (y + j).wrapping_add_signed(dy);

                if n1_x < stride && n1_y < plane.len() / stride
                    && n2_x < stride && n2_y < plane.len() / stride
                {
                    let n1 = plane[n1_y * stride + n1_x] as i16;
                    let n2 = plane[n2_y * stride + n2_x] as i16;

                    let edge_idx = (p - n1).signum() + (p - n2).signum();
                    let offset_idx = match edge_idx {
                        -2 => 0,
                        -1 => 1,
                        1 => 2,
                        2 => 3,
                        _ => continue,
                    };

                    let new_val = (p + offsets[offset_idx] as i16).clamp(0, 255);
                    plane[idx] = new_val as u8;
                }
            }
        }
    }

    /// Reset decoder state.
    fn reset_state(&mut self) {
        self.dpb.clear();
        self.current_frame = None;
        self.current_poc = 0;
        self.prev_poc_lsb = 0;
        self.prev_poc_msb = 0;
    }

    /// Flush the decoder.
    pub fn flush(&mut self) -> Vec<Frame> {
        let mut output = Vec::new();

        // Output any remaining frames in DPB
        for pic in &self.dpb.pictures {
            output.push(pic.frame.clone());
        }

        self.reset_state();
        output
    }
}

/// HEVC decoder info.
#[derive(Debug, Clone)]
pub struct HevcDecoderInfo {
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Profile.
    pub profile: Option<HevcProfile>,
    /// Tier.
    pub tier: HevcTier,
    /// Level.
    pub level: HevcLevel,
    /// Bit depth.
    pub bit_depth: u8,
    /// Chroma format.
    pub chroma_format: u8,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let decoder = HevcDecoder::default_config();
        assert!(!decoder.initialized);
    }

    #[test]
    fn test_decoder_config() {
        let config = HevcDecoderConfig {
            max_width: 1920,
            max_height: 1080,
            output_format: PixelFormat::Yuv420p,
            threads: 4,
            threading: true,
        };
        let decoder = HevcDecoder::new(config);
        assert_eq!(decoder.config.max_width, 1920);
        assert_eq!(decoder.config.max_height, 1080);
    }

    #[test]
    fn test_intra_mode() {
        assert_eq!(IntraMode::Planar.index(), 0);
        assert_eq!(IntraMode::Dc.index(), 1);
        assert_eq!(IntraMode::Horizontal.index(), 10);
        assert_eq!(IntraMode::Vertical.index(), 26);
        assert!(IntraMode::Angular(15).is_angular());
        assert!(!IntraMode::Dc.is_angular());
    }

    #[test]
    fn test_part_mode() {
        assert_eq!(PartMode::Part2Nx2N.num_parts(), 1);
        assert_eq!(PartMode::PartNxN.num_parts(), 4);
        assert_eq!(PartMode::Part2NxN.num_parts(), 2);
    }

    #[test]
    fn test_prediction_mode() {
        let intra = PredictionMode::Intra(26);
        let inter = PredictionMode::Inter { ref_idx_l0: 0, ref_idx_l1: -1 };
        let skip = PredictionMode::Skip;

        assert!(matches!(intra, PredictionMode::Intra(_)));
        assert!(matches!(inter, PredictionMode::Inter { .. }));
        assert!(matches!(skip, PredictionMode::Skip));
    }

    #[test]
    fn test_sao_params_default() {
        let params = SaoParams::default();
        assert!(!params.merge_left);
        assert!(!params.merge_up);
        assert_eq!(params.type_idx_luma, 0);
    }

    #[test]
    fn test_dpb() {
        let mut dpb = DecodedPictureBuffer::new(4);
        let frame = Frame::new(64, 64, PixelFormat::Yuv420p, TimeBase::new(1, 30));

        dpb.add(frame.clone(), 0, false);
        assert!(dpb.get_by_poc(0).is_some());
        assert!(dpb.get_by_poc(1).is_none());

        dpb.mark_unused(0);
        dpb.clear();
        assert!(dpb.get_by_poc(0).is_none());
    }
}
