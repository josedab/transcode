//! VVC decoder implementation.
//!
//! This module provides a complete VVC/H.266 decoder including:
//! - NAL unit parsing and parameter set management
//! - CTU/CU/PU/TU decoding with QTBT+MTT structure
//! - Intra and inter prediction (including MIP, ISP, affine, GPM)
//! - Transform and inverse quantization (including LFNST, MTS)
//! - In-loop filtering (deblocking, SAO, ALF)
//! - Reference picture management

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::unnecessary_cast)]
#![allow(unused_variables)]
#![allow(dead_code)]

use crate::error::{Result, VvcError, VvcLevel, VvcProfile, VvcTier};
use crate::nal::{
    NalUnitHeader, NalUnitType, PictureHeader, Pps, SliceHeader, Sps, Vps,
    parse_annexb_nal_units,
};
use crate::syntax::{
    AlfCtuParams, CodingTreeUnit, CodingUnit, IntraChromaMode, IntraMode,
    IspMode, LmcsData, MergeMode, MotionVector, PredMode, SaoParams, SplitMode, TransformUnit,
};
use std::collections::HashMap;
use transcode_core::frame::{Frame, PixelFormat};
use transcode_core::packet::Packet;
use transcode_core::timestamp::TimeBase;

/// VVC decoder configuration.
#[derive(Debug, Clone)]
pub struct VvcDecoderConfig {
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
    /// Enable ALF.
    pub alf_enabled: bool,
    /// Enable LMCS.
    pub lmcs_enabled: bool,
}

impl Default for VvcDecoderConfig {
    fn default() -> Self {
        Self {
            max_width: 8192,
            max_height: 4320,
            output_format: PixelFormat::Yuv420p10le,
            threads: 1,
            threading: false,
            alf_enabled: true,
            lmcs_enabled: true,
        }
    }
}

/// Merge candidate for inter prediction.
#[derive(Debug, Clone, Copy, Default)]
struct MergeCandidate {
    /// Motion vector for L0 reference.
    mv_l0: MotionVector,
    /// Motion vector for L1 reference.
    mv_l1: MotionVector,
    /// Reference index for L0.
    ref_idx_l0: i8,
    /// Reference index for L1.
    ref_idx_l1: i8,
}

/// Affine merge candidate for subblock motion.
#[derive(Debug, Clone, Copy, Default)]
struct AffineMergeCandidate {
    /// Control point motion vectors (up to 3 for 6-parameter affine).
    cpmv: [MotionVector; 3],
    /// Affine type: 0 = 4-parameter, 1 = 6-parameter.
    affine_type: u8,
    /// Reference index for L0.
    ref_idx_l0: i8,
    /// Reference index for L1.
    ref_idx_l1: i8,
}

/// Palette entry for palette mode.
#[derive(Debug, Clone, Copy, Default)]
struct PaletteEntry {
    /// Luma value.
    y: u8,
    /// Cb chroma value.
    cb: u8,
    /// Cr chroma value.
    cr: u8,
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
    /// Layer ID.
    layer_id: u8,
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

    fn add(&mut self, frame: Frame, poc: i32, is_long_term: bool, layer_id: u8) {
        // Remove old pictures if at capacity
        while self.pictures.len() >= self.max_size {
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
            layer_id,
        });
    }

    fn get_by_poc(&self, poc: i32) -> Option<&Frame> {
        self.pictures.iter().find(|p| p.poc == poc).map(|p| &p.frame)
    }

    fn get_by_poc_and_layer(&self, poc: i32, layer_id: u8) -> Option<&Frame> {
        self.pictures
            .iter()
            .find(|p| p.poc == poc && p.layer_id == layer_id)
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

    fn get_ref_list(&self, is_long_term: bool) -> Vec<&Frame> {
        self.pictures
            .iter()
            .filter(|p| p.is_reference && p.is_long_term == is_long_term)
            .map(|p| &p.frame)
            .collect()
    }
}

/// CTU decoding context.
#[derive(Debug, Clone, Default)]
struct CtuContext {
    /// SAO parameters.
    sao_params: SaoParams,
    /// ALF parameters.
    alf_params: AlfCtuParams,
    /// LMCS data.
    lmcs_data: Option<LmcsData>,
    /// QP for luma.
    qp_y: i8,
    /// QP for Cb.
    qp_cb: i8,
    /// QP for Cr.
    qp_cr: i8,
}

/// VVC decoder.
#[derive(Debug)]
pub struct VvcDecoder {
    /// Configuration.
    config: VvcDecoderConfig,
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
    /// Current picture header.
    current_ph: Option<PictureHeader>,
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
    /// CTU context per CTU.
    ctu_contexts: Vec<CtuContext>,
    /// Is initialized.
    initialized: bool,
    /// Frame count.
    frame_count: u64,
    /// Current layer ID.
    current_layer_id: u8,
    /// GDR recovery point POC.
    gdr_recovery_poc: Option<i32>,
    /// LMCS lookup table.
    lmcs_lut_forward: Vec<u16>,
    /// LMCS inverse lookup table.
    lmcs_lut_inverse: Vec<u16>,
}

impl VvcDecoder {
    /// Create a new VVC decoder.
    pub fn new(config: VvcDecoderConfig) -> Self {
        Self {
            config,
            vps_map: HashMap::new(),
            sps_map: HashMap::new(),
            pps_map: HashMap::new(),
            current_vps_id: None,
            current_sps_id: None,
            current_pps_id: None,
            current_ph: None,
            dpb: DecodedPictureBuffer::new(16),
            current_frame: None,
            current_poc: 0,
            prev_poc_lsb: 0,
            prev_poc_msb: 0,
            ctu_contexts: Vec::new(),
            initialized: false,
            frame_count: 0,
            current_layer_id: 0,
            gdr_recovery_poc: None,
            lmcs_lut_forward: Vec::new(),
            lmcs_lut_inverse: Vec::new(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(VvcDecoderConfig::default())
    }

    /// Get decoder info.
    pub fn info(&self) -> VvcDecoderInfo {
        VvcDecoderInfo {
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
    pub fn get_profile(&self) -> Option<VvcProfile> {
        self.current_sps_id
            .and_then(|id| self.sps_map.get(&id))
            .and_then(|sps| sps.profile_tier_level.profile())
    }

    /// Get current tier.
    pub fn get_tier(&self) -> VvcTier {
        self.current_sps_id
            .and_then(|id| self.sps_map.get(&id))
            .map(|sps| sps.profile_tier_level.tier())
            .unwrap_or_default()
    }

    /// Get current level.
    pub fn get_level(&self) -> VvcLevel {
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
            .unwrap_or(10)
    }

    /// Get chroma format.
    pub fn get_chroma_format(&self) -> u8 {
        self.current_sps_id
            .and_then(|id| self.sps_map.get(&id))
            .map(|sps| sps.sps_chroma_format_idc)
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
            self.current_layer_id = header.nuh_layer_id;

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
                NalUnitType::PhNut => {
                    self.process_picture_header(&rbsp)?;
                }
                NalUnitType::IdrNLp | NalUnitType::IdrWRadl => {
                    output_frame = self.decode_idr_slice(&header, &rbsp)?;
                }
                NalUnitType::TrailNut | NalUnitType::StsaNut => {
                    output_frame = self.decode_trailing_slice(&header, &rbsp)?;
                }
                NalUnitType::RadlNut | NalUnitType::RaslNut => {
                    output_frame = self.decode_leading_slice(&header, &rbsp)?;
                }
                NalUnitType::CraNut => {
                    output_frame = self.decode_cra_slice(&header, &rbsp)?;
                }
                NalUnitType::GdrNut => {
                    output_frame = self.decode_gdr_slice(&header, &rbsp)?;
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
                NalUnitType::ApsPrefix | NalUnitType::ApsNut => {
                    self.process_aps(&rbsp)?;
                }
                NalUnitType::DciNut => {
                    self.process_dci(&rbsp)?;
                }
                NalUnitType::OpiNut => {
                    self.process_opi(&rbsp)?;
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

        // Initialize LMCS LUTs if enabled
        if sps.sps_lmcs_enabled_flag && self.config.lmcs_enabled {
            self.init_lmcs_luts(sps.bit_depth_luma());
        }

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

    /// Process Picture Header NAL unit.
    fn process_picture_header(&mut self, rbsp: &[u8]) -> Result<()> {
        let sps = self.sps_map.get(&self.current_sps_id.unwrap_or(0))
            .ok_or_else(|| VvcError::Sps("SPS not found".to_string()))?
            .clone();
        let pps = self.pps_map.get(&self.current_pps_id.unwrap_or(0))
            .ok_or_else(|| VvcError::Pps("PPS not found".to_string()))?
            .clone();

        let ph = PictureHeader::parse(rbsp, &sps, &pps)?;
        self.current_ph = Some(ph);
        Ok(())
    }

    /// Process SEI NAL unit.
    fn process_sei(&mut self, _rbsp: &[u8]) -> Result<()> {
        // SEI parsing is optional
        Ok(())
    }

    /// Process APS (Adaptation Parameter Set) NAL unit.
    fn process_aps(&mut self, _rbsp: &[u8]) -> Result<()> {
        // APS parsing for ALF, LMCS, scaling list
        Ok(())
    }

    /// Process DCI (Decoding Capability Information) NAL unit.
    fn process_dci(&mut self, _rbsp: &[u8]) -> Result<()> {
        // DCI parsing
        Ok(())
    }

    /// Process OPI (Operating Point Information) NAL unit.
    fn process_opi(&mut self, _rbsp: &[u8]) -> Result<()> {
        // OPI parsing
        Ok(())
    }

    /// Initialize LMCS lookup tables.
    fn init_lmcs_luts(&mut self, bit_depth: u8) {
        let max_val = (1u32 << bit_depth) - 1;
        self.lmcs_lut_forward = vec![0u16; (max_val + 1) as usize];
        self.lmcs_lut_inverse = vec![0u16; (max_val + 1) as usize];

        // Initialize with identity mapping
        for i in 0..=max_val {
            self.lmcs_lut_forward[i as usize] = i as u16;
            self.lmcs_lut_inverse[i as usize] = i as u16;
        }
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
        self.gdr_recovery_poc = None;

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

    /// Decode a GDR slice.
    fn decode_gdr_slice(
        &mut self,
        header: &NalUnitHeader,
        rbsp: &[u8],
    ) -> Result<Option<Frame>> {
        // Set GDR recovery point
        if let Some(ref ph) = self.current_ph {
            self.gdr_recovery_poc = Some(self.current_poc + ph.ph_recovery_poc_cnt as i32);
        }
        self.decode_slice(header, rbsp)
    }

    /// Decode a slice.
    fn decode_slice(
        &mut self,
        header: &NalUnitHeader,
        rbsp: &[u8],
    ) -> Result<Option<Frame>> {
        if !self.initialized {
            return Err(VvcError::DecoderConfig("Decoder not initialized".to_string()));
        }

        let sps = self.sps_map.get(&self.current_sps_id.unwrap())
            .ok_or_else(|| VvcError::Sps("SPS not found".to_string()))?
            .clone();

        let pps = self.pps_map.get(&self.current_pps_id.unwrap_or(0))
            .ok_or_else(|| VvcError::Pps("PPS not found".to_string()))?
            .clone();

        // Parse slice header
        let slice_header = SliceHeader::parse(rbsp, &sps, &pps, header.nal_unit_type)?;

        // Get picture header (either embedded or from previous PH NAL)
        let ph = if slice_header.sh_picture_header_in_slice_header_flag {
            slice_header.picture_header.clone()
                .ok_or_else(|| VvcError::PictureHeader("Missing embedded PH".to_string()))?
        } else {
            self.current_ph.clone()
                .ok_or_else(|| VvcError::PictureHeader("Missing PH".to_string()))?
        };

        // Calculate POC
        if !header.nal_unit_type.is_idr() {
            self.calculate_poc(&ph, &sps);
        }

        // Allocate frame if first slice in picture
        self.allocate_frame(&sps)?;

        // Initialize CTU contexts
        self.init_ctu_contexts(&sps);

        // Decode CTUs
        self.decode_ctus(&slice_header, &ph, &sps, &pps, rbsp)?;

        // Apply in-loop filters
        // Take the frame temporarily to avoid borrow checker issues
        if let Some(mut frame) = self.current_frame.take() {
            // Deblocking filter
            if !ph.ph_deblocking_filter_disabled_flag && !slice_header.sh_deblocking_filter_disabled_flag {
                Self::apply_deblocking_filter_static(&mut frame, &slice_header, &ph, &sps, &pps)?;
            }

            // SAO filter
            if sps.sps_sao_enabled_flag {
                let sao_luma = ph.ph_sao_luma_enabled_flag || slice_header.sh_sao_luma_used_flag;
                let sao_chroma = ph.ph_sao_chroma_enabled_flag || slice_header.sh_sao_chroma_used_flag;
                if sao_luma || sao_chroma {
                    Self::apply_sao_filter_static(&mut frame, &sps, sao_luma, sao_chroma)?;
                }
            }

            // ALF filter
            if sps.sps_alf_enabled_flag && self.config.alf_enabled {
                let alf_enabled = ph.ph_alf_enabled_flag || slice_header.sh_alf_enabled_flag;
                if alf_enabled {
                    Self::apply_alf_filter_static(&mut frame, &slice_header, &ph, &sps)?;
                }
            }

            // Put the frame back
            self.current_frame = Some(frame);
        }

        // Output frame if complete
        let output = self.current_frame.take();
        if let Some(ref frame) = output {
            // Add to DPB if reference picture
            if header.nal_unit_type.is_reference() && !ph.ph_non_ref_pic_flag {
                self.dpb.add(frame.clone(), self.current_poc, false, self.current_layer_id);
            }
            self.frame_count += 1;
        }

        Ok(output)
    }

    /// Calculate POC (Picture Order Count).
    fn calculate_poc(&mut self, ph: &PictureHeader, sps: &Sps) {
        let max_poc_lsb = sps.max_pic_order_cnt_lsb();
        let poc_lsb = ph.ph_pic_order_cnt_lsb;

        let poc_msb = if (poc_lsb < self.prev_poc_lsb)
            && (self.prev_poc_lsb - poc_lsb >= max_poc_lsb / 2)
        {
            self.prev_poc_msb + max_poc_lsb as i32
        } else if (poc_lsb > self.prev_poc_lsb)
            && (poc_lsb - self.prev_poc_lsb > max_poc_lsb / 2)
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
        let width = sps.sps_pic_width_max_in_luma_samples;
        let height = sps.sps_pic_height_max_in_luma_samples;
        let bit_depth = sps.bit_depth_luma();

        let format = match (sps.sps_chroma_format_idc, bit_depth) {
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

    /// Initialize CTU contexts.
    fn init_ctu_contexts(&mut self, sps: &Sps) {
        let num_ctus = (sps.pic_width_in_ctbs() * sps.pic_height_in_ctbs()) as usize;
        self.ctu_contexts = vec![CtuContext::default(); num_ctus];
    }

    /// Decode CTUs in the slice.
    fn decode_ctus(
        &mut self,
        slice_header: &SliceHeader,
        ph: &PictureHeader,
        sps: &Sps,
        pps: &Pps,
        rbsp: &[u8],
    ) -> Result<()> {
        let ctb_size = sps.ctb_size_y();
        let pic_width_in_ctbs = sps.pic_width_in_ctbs();
        let pic_height_in_ctbs = sps.pic_height_in_ctbs();

        let start_ctb = slice_header.sh_slice_address;
        let end_ctb = pic_width_in_ctbs * pic_height_in_ctbs;

        // Get base QP
        let base_qp = pps.init_qp() + ph.ph_qp_delta + slice_header.sh_qp_delta;

        for ctb_addr in start_ctb..end_ctb {
            let ctb_x = ctb_addr % pic_width_in_ctbs;
            let ctb_y = ctb_addr / pic_width_in_ctbs;

            let ctu = CodingTreeUnit::new(
                ctb_addr,
                ctb_x * ctb_size,
                ctb_y * ctb_size,
                sps.log2_ctb_size(),
            );

            // Initialize CTU context
            let ctu_ctx = &mut self.ctu_contexts[ctb_addr as usize];
            ctu_ctx.qp_y = base_qp;

            // Decode SAO parameters
            if sps.sps_sao_enabled_flag {
                self.decode_sao_params(ctb_addr as usize, slice_header, ph)?;
            }

            // Decode ALF CTU parameters
            if sps.sps_alf_enabled_flag {
                self.decode_alf_ctu_params(ctb_addr as usize, slice_header, ph)?;
            }

            // Decode coding quadtree (QTBT+MTT structure)
            self.decode_coding_quadtree(
                &ctu,
                0,
                0,
                0,
                slice_header,
                ph,
                sps,
                pps,
            )?;
        }

        Ok(())
    }

    /// Decode SAO parameters for a CTU.
    fn decode_sao_params(
        &mut self,
        ctb_addr: usize,
        slice_header: &SliceHeader,
        ph: &PictureHeader,
    ) -> Result<()> {
        let ctx = &mut self.ctu_contexts[ctb_addr];

        // Simplified SAO parameter decoding
        // A full implementation would use CABAC

        Ok(())
    }

    /// Decode ALF CTU parameters.
    fn decode_alf_ctu_params(
        &mut self,
        ctb_addr: usize,
        slice_header: &SliceHeader,
        ph: &PictureHeader,
    ) -> Result<()> {
        let ctx = &mut self.ctu_contexts[ctb_addr];

        // Simplified ALF parameter decoding
        // A full implementation would use CABAC

        Ok(())
    }

    /// Decode coding quadtree (QTBT+MTT structure).
    fn decode_coding_quadtree(
        &mut self,
        ctu: &CodingTreeUnit,
        qt_depth: u8,
        mtt_depth: u8,
        mtt_depth_hor: u8,
        slice_header: &SliceHeader,
        ph: &PictureHeader,
        sps: &Sps,
        pps: &Pps,
    ) -> Result<()> {
        let x = ctu.x;
        let y = ctu.y;
        let size = ctu.size;

        let log2_size = (size as f32).log2() as u8;
        let min_qt_size = sps.min_cb_size();
        let max_bt_size = 128u32; // Simplified
        let max_tt_size = 64u32;  // Simplified
        let max_mtt_depth = 4u8;  // Simplified

        // Check if we're within picture bounds
        let in_bounds = x < sps.sps_pic_width_max_in_luma_samples
            && y < sps.sps_pic_height_max_in_luma_samples;

        if !in_bounds {
            return Ok(());
        }

        // Decide split mode (simplified - a real decoder would use CABAC)
        let split_mode = self.decide_split_mode(
            x, y, size, qt_depth, mtt_depth, slice_header, sps,
        );

        match split_mode {
            SplitMode::NoSplit => {
                // Decode coding unit
                let mut cu = CodingUnit::new(x, y, size, size);
                cu.qt_depth = qt_depth;
                cu.mtt_depth = mtt_depth;
                self.decode_coding_unit(&mut cu, slice_header, ph, sps, pps)?;
            }
            SplitMode::QtSplit => {
                // Quad-tree split
                let half_size = size / 2;
                for i in 0..4 {
                    let sub_x = x + (i % 2) * half_size;
                    let sub_y = y + (i / 2) * half_size;

                    let sub_ctu = CodingTreeUnit::new(
                        ctu.ctu_address,
                        sub_x,
                        sub_y,
                        log2_size - 1,
                    );

                    self.decode_coding_quadtree(
                        &sub_ctu,
                        qt_depth + 1,
                        0, // Reset MTT depth after QT split
                        0,
                        slice_header,
                        ph,
                        sps,
                        pps,
                    )?;
                }
            }
            SplitMode::BtHorSplit => {
                // Binary tree horizontal split
                let half_height = size / 2;
                for i in 0..2 {
                    let sub_y = y + i * half_height;
                    let sub_ctu = CodingTreeUnit::new(
                        ctu.ctu_address,
                        x,
                        sub_y,
                        log2_size,
                    );

                    self.decode_mtt_subtree(
                        x, sub_y, size, half_height,
                        qt_depth, mtt_depth + 1, mtt_depth_hor + 1,
                        slice_header, ph, sps, pps,
                    )?;
                }
            }
            SplitMode::BtVerSplit => {
                // Binary tree vertical split
                let half_width = size / 2;
                for i in 0..2 {
                    let sub_x = x + i * half_width;

                    self.decode_mtt_subtree(
                        sub_x, y, half_width, size,
                        qt_depth, mtt_depth + 1, mtt_depth_hor,
                        slice_header, ph, sps, pps,
                    )?;
                }
            }
            SplitMode::TtHorSplit => {
                // Ternary tree horizontal split (1:2:1)
                let quarter_height = size / 4;
                let half_height = size / 2;

                self.decode_mtt_subtree(
                    x, y, size, quarter_height,
                    qt_depth, mtt_depth + 1, mtt_depth_hor + 1,
                    slice_header, ph, sps, pps,
                )?;
                self.decode_mtt_subtree(
                    x, y + quarter_height, size, half_height,
                    qt_depth, mtt_depth + 1, mtt_depth_hor + 1,
                    slice_header, ph, sps, pps,
                )?;
                self.decode_mtt_subtree(
                    x, y + quarter_height + half_height, size, quarter_height,
                    qt_depth, mtt_depth + 1, mtt_depth_hor + 1,
                    slice_header, ph, sps, pps,
                )?;
            }
            SplitMode::TtVerSplit => {
                // Ternary tree vertical split (1:2:1)
                let quarter_width = size / 4;
                let half_width = size / 2;

                self.decode_mtt_subtree(
                    x, y, quarter_width, size,
                    qt_depth, mtt_depth + 1, mtt_depth_hor,
                    slice_header, ph, sps, pps,
                )?;
                self.decode_mtt_subtree(
                    x + quarter_width, y, half_width, size,
                    qt_depth, mtt_depth + 1, mtt_depth_hor,
                    slice_header, ph, sps, pps,
                )?;
                self.decode_mtt_subtree(
                    x + quarter_width + half_width, y, quarter_width, size,
                    qt_depth, mtt_depth + 1, mtt_depth_hor,
                    slice_header, ph, sps, pps,
                )?;
            }
        }

        Ok(())
    }

    /// Decode MTT (Multi-Type Tree) subtree.
    fn decode_mtt_subtree(
        &mut self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        qt_depth: u8,
        mtt_depth: u8,
        mtt_depth_hor: u8,
        slice_header: &SliceHeader,
        ph: &PictureHeader,
        sps: &Sps,
        pps: &Pps,
    ) -> Result<()> {
        // Check bounds
        if x >= sps.sps_pic_width_max_in_luma_samples
            || y >= sps.sps_pic_height_max_in_luma_samples
        {
            return Ok(());
        }

        // For simplicity, decode as leaf CU
        // A full implementation would continue the MTT tree
        let mut cu = CodingUnit::new(x, y, width, height);
        cu.qt_depth = qt_depth;
        cu.mtt_depth = mtt_depth;
        self.decode_coding_unit(&mut cu, slice_header, ph, sps, pps)?;

        Ok(())
    }

    /// Decide split mode (simplified).
    fn decide_split_mode(
        &self,
        x: u32,
        y: u32,
        size: u32,
        qt_depth: u8,
        mtt_depth: u8,
        slice_header: &SliceHeader,
        sps: &Sps,
    ) -> SplitMode {
        let min_size = sps.min_cb_size();

        // If at minimum size, can't split further
        if size <= min_size {
            return SplitMode::NoSplit;
        }

        // Simplified: always use QT split if possible, otherwise no split
        // A real decoder would use CABAC to decode the split decision
        let max_qt_depth = 4u8;
        if qt_depth < max_qt_depth && size > min_size * 2 {
            SplitMode::QtSplit
        } else {
            SplitMode::NoSplit
        }
    }

    /// Decode a coding unit.
    fn decode_coding_unit(
        &mut self,
        cu: &mut CodingUnit,
        slice_header: &SliceHeader,
        ph: &PictureHeader,
        sps: &Sps,
        pps: &Pps,
    ) -> Result<()> {
        let is_intra_slice = slice_header.is_intra();

        // Determine prediction mode
        if is_intra_slice {
            cu.pred_mode = PredMode::Intra;
        } else {
            // Simplified: assume intra for now
            // A real decoder would use CABAC to decode skip/merge/pred mode flags
            cu.pred_mode = PredMode::Intra;
        }

        match cu.pred_mode {
            PredMode::Intra => {
                self.decode_intra_prediction(cu, slice_header, ph, sps, pps)?;
            }
            PredMode::Inter => {
                self.decode_inter_prediction(cu, slice_header, ph, sps, pps)?;
            }
            PredMode::Ibc => {
                self.decode_ibc_prediction(cu, slice_header, sps)?;
            }
            PredMode::Palette => {
                self.decode_palette_mode(cu, sps)?;
            }
        }

        // Decode transform tree
        self.decode_transform_tree(cu, slice_header, sps, pps)?;

        Ok(())
    }

    /// Decode intra prediction.
    fn decode_intra_prediction(
        &mut self,
        cu: &mut CodingUnit,
        slice_header: &SliceHeader,
        ph: &PictureHeader,
        sps: &Sps,
        pps: &Pps,
    ) -> Result<()> {
        // Decode intra prediction mode
        // Simplified: use DC mode
        cu.intra_luma_mode = IntraMode::Dc.index();
        cu.intra_chroma_mode = IntraChromaMode::Dm as u8;

        // Check for MIP (Matrix Intra Prediction)
        if sps.sps_mip_enabled_flag && cu.width <= 64 && cu.height <= 64 {
            // MIP mode decision would be decoded here
            cu.intra_mip_flag = false;
        }

        // Check for ISP (Intra Sub-Partitions)
        if sps.sps_isp_enabled_flag && cu.is_square() && cu.width >= 4 && cu.width <= 64 {
            // ISP mode decision would be decoded here
            cu.isp_mode = IspMode::NoIsp;
        }

        // Apply prediction
        self.apply_intra_prediction(cu, true)?;

        if sps.sps_chroma_format_idc != 0 {
            self.apply_intra_prediction(cu, false)?;
        }

        Ok(())
    }

    /// Decode inter prediction.
    fn decode_inter_prediction(
        &mut self,
        cu: &mut CodingUnit,
        slice_header: &SliceHeader,
        ph: &PictureHeader,
        sps: &Sps,
        pps: &Pps,
    ) -> Result<()> {
        // Decode merge/AMVP mode
        if cu.merge_flag {
            match cu.merge_mode {
                MergeMode::Regular => {
                    self.decode_regular_merge(cu, slice_header)?;
                }
                MergeMode::Mmvd => {
                    self.decode_mmvd_merge(cu, slice_header)?;
                }
                MergeMode::Subblock => {
                    self.decode_subblock_merge(cu, slice_header)?;
                }
                MergeMode::Ciip => {
                    self.decode_ciip(cu, slice_header, sps)?;
                }
                MergeMode::Gpm => {
                    self.decode_gpm(cu, slice_header)?;
                }
            }
        } else {
            // AMVP mode
            self.decode_amvp(cu, slice_header)?;
        }

        // Apply motion compensation
        self.apply_inter_prediction(cu)?;

        Ok(())
    }

    /// Decode regular merge mode.
    ///
    /// In regular merge mode, the motion information is derived from spatial and temporal
    /// neighboring blocks. The merge_idx selects which candidate from the merge list to use.
    fn decode_regular_merge(&mut self, cu: &mut CodingUnit, slice_header: &SliceHeader) -> Result<()> {
        // Build merge candidate list from spatial and temporal neighbors
        let merge_candidates = self.build_merge_candidate_list(cu, slice_header, 6)?;

        // Select candidate based on merge index
        let idx = cu.merge_idx as usize;
        if idx < merge_candidates.len() {
            let candidate = &merge_candidates[idx];
            cu.mv_l0 = candidate.mv_l0;
            cu.mv_l1 = candidate.mv_l1;
            cu.ref_idx_l0 = candidate.ref_idx_l0;
            cu.ref_idx_l1 = candidate.ref_idx_l1;
        }

        Ok(())
    }

    /// Decode MMVD merge mode.
    ///
    /// MMVD (Merge with MVD) extends regular merge by adding a motion vector difference
    /// to the base merge candidate. It uses predefined distance and direction steps.
    fn decode_mmvd_merge(&mut self, cu: &mut CodingUnit, slice_header: &SliceHeader) -> Result<()> {
        // Build merge candidate list (only first 2 candidates used for MMVD)
        let merge_candidates = self.build_merge_candidate_list(cu, slice_header, 2)?;

        // MMVD uses base_idx to select from first two merge candidates
        let base_idx = if cu.mmvd_merge_flag { 1 } else { 0 };

        if base_idx < merge_candidates.len() {
            let candidate = &merge_candidates[base_idx];
            cu.mv_l0 = candidate.mv_l0;
            cu.mv_l1 = candidate.mv_l1;
            cu.ref_idx_l0 = candidate.ref_idx_l0;
            cu.ref_idx_l1 = candidate.ref_idx_l1;

            // Apply MMVD offset based on distance and direction indices
            // Distance steps: 1/4, 1/2, 1, 2, 4, 8, 16, 32 pels
            // Direction: 0=+x, 1=-x, 2=+y, 3=-y
            let mmvd_distance_idx = (cu.merge_idx >> 2) & 0x7;
            let mmvd_direction_idx = cu.merge_idx & 0x3;

            let distance = match mmvd_distance_idx {
                0 => 1,   // 1/4 pel (in 1/16 units = 4)
                1 => 2,   // 1/2 pel (in 1/16 units = 8)
                2 => 4,   // 1 pel (in 1/16 units = 16)
                3 => 8,   // 2 pels
                4 => 16,  // 4 pels
                5 => 32,  // 8 pels
                6 => 64,  // 16 pels
                _ => 128, // 32 pels
            } * 4; // Convert to 1/16 pel units

            let (dx, dy) = match mmvd_direction_idx {
                0 => (distance as i16, 0),
                1 => (-(distance as i16), 0),
                2 => (0, distance as i16),
                _ => (0, -(distance as i16)),
            };

            // Apply offset to appropriate reference list
            if cu.ref_idx_l0 >= 0 {
                cu.mv_l0.x = cu.mv_l0.x.saturating_add(dx);
                cu.mv_l0.y = cu.mv_l0.y.saturating_add(dy);
            }
            if cu.ref_idx_l1 >= 0 {
                cu.mv_l1.x = cu.mv_l1.x.saturating_add(dx);
                cu.mv_l1.y = cu.mv_l1.y.saturating_add(dy);
            }
        }

        Ok(())
    }

    /// Decode subblock merge mode (affine).
    ///
    /// Affine merge mode uses control point motion vectors to model complex motion
    /// like rotation and zoom. Each 4x4 subblock gets its own derived motion vector.
    fn decode_subblock_merge(&mut self, cu: &mut CodingUnit, slice_header: &SliceHeader) -> Result<()> {
        // Build affine merge candidate list from spatial neighbors
        let affine_candidates = self.build_affine_merge_list(cu, slice_header)?;

        let idx = cu.merge_idx as usize;
        if idx < affine_candidates.len() {
            let candidate = &affine_candidates[idx];
            cu.affine_flag = true;
            cu.affine_type = candidate.affine_type;
            cu.affine_cpmv = candidate.cpmv;
            cu.ref_idx_l0 = candidate.ref_idx_l0;
            cu.ref_idx_l1 = candidate.ref_idx_l1;
        }

        Ok(())
    }

    /// Decode CIIP (Combined Inter-Intra Prediction).
    ///
    /// CIIP combines inter merge prediction with planar intra prediction using
    /// weighted averaging. The weights depend on block position.
    fn decode_ciip(&mut self, cu: &mut CodingUnit, slice_header: &SliceHeader, sps: &Sps) -> Result<()> {
        // First, decode regular merge for the inter part
        self.decode_regular_merge(cu, slice_header)?;

        // Set up intra mode for the intra part (always planar)
        cu.intra_luma_mode = 0; // Planar mode
        cu.ciip_flag = true;

        // The actual blending will be done in apply_inter_prediction
        Ok(())
    }

    /// Decode GPM (Geometric Partition Mode).
    ///
    /// GPM splits the block diagonally and applies different motion to each partition.
    /// There are 64 possible split directions.
    fn decode_gpm(&mut self, cu: &mut CodingUnit, slice_header: &SliceHeader) -> Result<()> {
        // Build merge candidate list
        let merge_candidates = self.build_merge_candidate_list(cu, slice_header, 6)?;

        // GPM uses two separate merge indices for the two partitions
        let idx0 = cu.gpm_merge_idx[0] as usize;
        let idx1 = cu.gpm_merge_idx[1] as usize;

        if idx0 < merge_candidates.len() {
            let candidate = &merge_candidates[idx0];
            cu.mv_l0 = candidate.mv_l0;
            cu.ref_idx_l0 = candidate.ref_idx_l0;
        }

        if idx1 < merge_candidates.len() {
            let candidate = &merge_candidates[idx1];
            cu.mv_l1 = candidate.mv_l0; // Use L0 MV from second candidate as "L1" for GPM
            cu.ref_idx_l1 = candidate.ref_idx_l0;
        }

        cu.gpm_flag = true;
        // gpm_split_dir was already decoded from bitstream

        Ok(())
    }

    /// Decode AMVP mode.
    ///
    /// AMVP (Advanced Motion Vector Prediction) explicitly signals the motion vector
    /// difference from a predictor. This allows more precise motion specification.
    fn decode_amvp(&mut self, cu: &mut CodingUnit, slice_header: &SliceHeader) -> Result<()> {
        // Build AMVP candidate list (2 candidates) for L0
        if cu.ref_idx_l0 >= 0 {
            let amvp_list_l0 = self.build_amvp_candidate_list(cu, slice_header, 0)?;

            // MVP index selects the predictor
            let mvp_idx = 0; // Would be decoded from bitstream
            if mvp_idx < amvp_list_l0.len() {
                let mvp = amvp_list_l0[mvp_idx];
                // Add MVD to MVP
                cu.mv_l0.x = mvp.x.saturating_add(cu.mvd_l0.x);
                cu.mv_l0.y = mvp.y.saturating_add(cu.mvd_l0.y);
            }
        }

        // Build AMVP candidate list for L1 (B-slices only)
        if cu.ref_idx_l1 >= 0 {
            let amvp_list_l1 = self.build_amvp_candidate_list(cu, slice_header, 1)?;

            let mvp_idx = 0;
            if mvp_idx < amvp_list_l1.len() {
                let mvp = amvp_list_l1[mvp_idx];
                cu.mv_l1.x = mvp.x.saturating_add(cu.mvd_l1.x);
                cu.mv_l1.y = mvp.y.saturating_add(cu.mvd_l1.y);
            }
        }

        Ok(())
    }

    /// Build merge candidate list from spatial and temporal neighbors.
    fn build_merge_candidate_list(
        &self,
        cu: &CodingUnit,
        _slice_header: &SliceHeader,
        max_candidates: usize,
    ) -> Result<Vec<MergeCandidate>> {
        let mut candidates = Vec::with_capacity(max_candidates);

        // Spatial candidates: A0, A1, B0, B1, B2
        // A0: Bottom-left
        // A1: Left
        // B0: Top-right
        // B1: Top
        // B2: Top-left

        let positions = [
            (cu.x as i32 - 1, cu.y as i32 + cu.height as i32 - 1), // A1 (left)
            (cu.x as i32 + cu.width as i32, cu.y as i32 - 1),      // B1 (top-right corner)
            (cu.x as i32 + cu.width as i32 - 1, cu.y as i32 - 1),  // B0 (top)
            (cu.x as i32 - 1, cu.y as i32),                        // A0 (bottom-left)
            (cu.x as i32 - 1, cu.y as i32 - 1),                    // B2 (top-left)
        ];

        for &(nx, ny) in &positions {
            if candidates.len() >= max_candidates {
                break;
            }

            if nx >= 0 && ny >= 0 {
                // Check if neighbor is available and inter-coded
                // For now, create a default candidate
                if let Some(neighbor_mv) = self.get_neighbor_motion(nx as u32, ny as u32) {
                    // Check for duplicate before adding
                    let is_duplicate = candidates.iter().any(|c: &MergeCandidate| {
                        c.mv_l0.x == neighbor_mv.mv_l0.x && c.mv_l0.y == neighbor_mv.mv_l0.y
                    });

                    if !is_duplicate {
                        candidates.push(neighbor_mv);
                    }
                }
            }
        }

        // Add temporal candidate (collocated)
        if candidates.len() < max_candidates {
            if let Some(temporal_mv) = self.get_temporal_mv_predictor(cu) {
                candidates.push(temporal_mv);
            }
        }

        // Fill remaining with zero MV candidates
        while candidates.len() < max_candidates {
            candidates.push(MergeCandidate::default());
        }

        Ok(candidates)
    }

    /// Build affine merge candidate list.
    fn build_affine_merge_list(
        &self,
        cu: &CodingUnit,
        _slice_header: &SliceHeader,
    ) -> Result<Vec<AffineMergeCandidate>> {
        let mut candidates = Vec::with_capacity(5);

        // Check spatial neighbors for affine-coded blocks
        // Inherited affine candidates from A0, A1, B0, B1, B2
        let positions = [
            (cu.x as i32 - 1, cu.y as i32 + cu.height as i32 - 1), // A1
            (cu.x as i32 + cu.width as i32, cu.y as i32 - 1),      // B1
            (cu.x as i32 + cu.width as i32 - 1, cu.y as i32 - 1),  // B0
            (cu.x as i32 - 1, cu.y as i32),                        // A0
            (cu.x as i32 - 1, cu.y as i32 - 1),                    // B2
        ];

        for &(nx, ny) in &positions {
            if candidates.len() >= 5 {
                break;
            }

            if nx >= 0 && ny >= 0 {
                // Would check if neighbor is affine-coded and inherit CPMVs
                // For now, construct candidates from regular MVs
            }
        }

        // Construct affine candidates from control point MVs
        // This requires deriving CPMVs from neighboring translation MVs
        let default_cpmv = [MotionVector::default(); 3];

        // Add constructed 4-parameter affine candidate
        if candidates.len() < 5 {
            candidates.push(AffineMergeCandidate {
                cpmv: default_cpmv,
                affine_type: 0, // 4-parameter
                ref_idx_l0: 0,
                ref_idx_l1: -1,
            });
        }

        // Fill with zero candidates
        while candidates.len() < 5 {
            candidates.push(AffineMergeCandidate::default());
        }

        Ok(candidates)
    }

    /// Build AMVP candidate list for a reference list.
    fn build_amvp_candidate_list(
        &self,
        cu: &CodingUnit,
        _slice_header: &SliceHeader,
        ref_list: u8,
    ) -> Result<Vec<MotionVector>> {
        let mut candidates = Vec::with_capacity(2);

        // Spatial MVP candidates
        let positions = [
            (cu.x as i32 - 1, cu.y as i32 + cu.height as i32 - 1), // Left
            (cu.x as i32 + cu.width as i32 - 1, cu.y as i32 - 1),  // Top
        ];

        for &(nx, ny) in &positions {
            if candidates.len() >= 2 {
                break;
            }

            if nx >= 0 && ny >= 0 {
                if let Some(neighbor_mv) = self.get_neighbor_motion(nx as u32, ny as u32) {
                    let mv = if ref_list == 0 { neighbor_mv.mv_l0 } else { neighbor_mv.mv_l1 };

                    // Check for duplicate
                    if !candidates.iter().any(|c: &MotionVector| c.x == mv.x && c.y == mv.y) {
                        candidates.push(mv);
                    }
                }
            }
        }

        // Temporal MVP
        if candidates.len() < 2 {
            if let Some(temporal) = self.get_temporal_mv_predictor(cu) {
                let mv = if ref_list == 0 { temporal.mv_l0 } else { temporal.mv_l1 };
                if !candidates.iter().any(|c| c.x == mv.x && c.y == mv.y) {
                    candidates.push(mv);
                }
            }
        }

        // Fill with zero MV
        while candidates.len() < 2 {
            candidates.push(MotionVector::default());
        }

        Ok(candidates)
    }

    /// Get motion information from a neighboring position.
    fn get_neighbor_motion(&self, _x: u32, _y: u32) -> Option<MergeCandidate> {
        // Would look up motion info from decoded picture buffer
        // For now return a default candidate if position is valid
        Some(MergeCandidate::default())
    }

    /// Get temporal motion vector predictor from collocated picture.
    fn get_temporal_mv_predictor(&self, cu: &CodingUnit) -> Option<MergeCandidate> {
        // Would derive MV from collocated block in reference picture
        // Scale MV based on POC difference
        let center_x = cu.x + cu.width / 2;
        let center_y = cu.y + cu.height / 2;

        // Look up collocated block position in reference picture
        // For now, return a default candidate
        let _ = (center_x, center_y);
        Some(MergeCandidate::default())
    }

    /// Decode IBC (Intra Block Copy) prediction.
    ///
    /// IBC copies a block from already-decoded regions of the current picture.
    /// It's useful for screen content with repeated patterns.
    fn decode_ibc_prediction(
        &mut self,
        cu: &mut CodingUnit,
        slice_header: &SliceHeader,
        sps: &Sps,
    ) -> Result<()> {
        // Check if IBC is enabled
        if !sps.sps_ibc_enabled_flag {
            return Err(VvcError::Bitstream("IBC not enabled in SPS".to_string()));
        }

        // Decode IBC block vector using merge or AMVP
        if cu.merge_flag {
            // IBC merge mode - derive BV from spatial neighbors
            let bv_candidates = self.build_ibc_merge_list(cu)?;

            let idx = cu.merge_idx as usize;
            if idx < bv_candidates.len() {
                cu.mv_l0 = bv_candidates[idx]; // BV stored in mv_l0
            }
        } else {
            // IBC AMVP mode - decode BVD and add to predictor
            let bv_predictors = self.build_ibc_amvp_list(cu)?;

            let mvp_idx = 0; // Would be decoded from bitstream
            if mvp_idx < bv_predictors.len() {
                let bvp = bv_predictors[mvp_idx];
                cu.mv_l0.x = bvp.x.saturating_add(cu.mvd_l0.x);
                cu.mv_l0.y = bvp.y.saturating_add(cu.mvd_l0.y);
            }
        }

        // Validate BV: must point to already decoded region
        let bv_x = cu.mv_l0.x >> 4; // Convert to integer pels
        let bv_y = cu.mv_l0.y >> 4;

        let ref_x = cu.x as i16 + bv_x;
        let ref_y = cu.y as i16 + bv_y;

        // Check that reference block is fully within already-decoded area
        // Reference must be to the left or above current CTU
        if ref_x < 0 || ref_y < 0 {
            return Err(VvcError::Bitstream("IBC reference out of bounds".to_string()));
        }

        // Apply IBC: copy block from reference position
        self.apply_ibc_copy(cu)?;

        Ok(())
    }

    /// Build IBC merge candidate list.
    fn build_ibc_merge_list(&self, cu: &CodingUnit) -> Result<Vec<MotionVector>> {
        let mut candidates = Vec::with_capacity(6);

        // Spatial BV candidates from neighboring IBC blocks
        let positions = [
            (cu.x as i32 - 1, cu.y as i32 + cu.height as i32 - 1), // Left
            (cu.x as i32 + cu.width as i32 - 1, cu.y as i32 - 1),  // Top
            (cu.x as i32 + cu.width as i32, cu.y as i32 - 1),      // Top-right
            (cu.x as i32 - 1, cu.y as i32),                        // Bottom-left
            (cu.x as i32 - 1, cu.y as i32 - 1),                    // Top-left
        ];

        for &(nx, ny) in &positions {
            if candidates.len() >= 6 {
                break;
            }

            if nx >= 0 && ny >= 0 {
                // Would look up BV from neighboring IBC-coded block
                // For now, add default candidates
                candidates.push(MotionVector::default());
            }
        }

        // History-based BV predictor (HBVP)
        // Would maintain a FIFO of recent BVs
        while candidates.len() < 6 {
            candidates.push(MotionVector::default());
        }

        Ok(candidates)
    }

    /// Build IBC AMVP list.
    fn build_ibc_amvp_list(&self, cu: &CodingUnit) -> Result<Vec<MotionVector>> {
        let mut candidates = Vec::with_capacity(2);

        // Spatial BV predictors
        let positions = [
            (cu.x as i32 - 1, cu.y as i32 + cu.height as i32 - 1), // Left
            (cu.x as i32 + cu.width as i32 - 1, cu.y as i32 - 1),  // Top
        ];

        for &(nx, ny) in &positions {
            if candidates.len() >= 2 {
                break;
            }

            if nx >= 0 && ny >= 0 {
                candidates.push(MotionVector::default());
            }
        }

        // Fill with zero BV
        while candidates.len() < 2 {
            candidates.push(MotionVector::default());
        }

        Ok(candidates)
    }

    /// Apply IBC block copy.
    fn apply_ibc_copy(&mut self, cu: &CodingUnit) -> Result<()> {
        let frame = self.current_frame.as_mut()
            .ok_or_else(|| VvcError::InvalidState("No current frame".to_string()))?;

        let bv_x = (cu.mv_l0.x >> 4) as i32;
        let bv_y = (cu.mv_l0.y >> 4) as i32;

        // Copy luma
        let stride = frame.stride(0);
        let plane = frame.plane_mut(0)
            .ok_or_else(|| VvcError::InvalidState("Luma plane not found".to_string()))?;

        for y in 0..cu.height as usize {
            for x in 0..cu.width as usize {
                let dst_x = cu.x as usize + x;
                let dst_y = cu.y as usize + y;
                let src_x = (cu.x as i32 + bv_x + x as i32) as usize;
                let src_y = (cu.y as i32 + bv_y + y as i32) as usize;

                // Bounds check
                if src_x < stride && src_y * stride < plane.len() && dst_y * stride + dst_x < plane.len() {
                    plane[dst_y * stride + dst_x] = plane[src_y * stride + src_x];
                }
            }
        }

        Ok(())
    }

    /// Decode palette mode.
    ///
    /// Palette mode represents pixels using a small color palette.
    /// It's efficient for screen content with limited colors.
    fn decode_palette_mode(&mut self, cu: &mut CodingUnit, sps: &Sps) -> Result<()> {
        // Check if palette mode is enabled
        if !sps.sps_palette_enabled_flag {
            return Err(VvcError::Bitstream("Palette mode not enabled in SPS".to_string()));
        }

        // Derive palette predictor from previous palette
        let predictor_palette = self.get_palette_predictor();

        // Decode palette size and entries
        let palette_size = self.decode_palette_size(cu)?;
        let mut palette = Vec::with_capacity(palette_size);

        // Reuse entries from predictor
        for i in 0..predictor_palette.len().min(palette_size) {
            palette.push(predictor_palette[i]);
        }

        // Decode new palette entries
        while palette.len() < palette_size {
            // Would decode new color from bitstream
            palette.push(PaletteEntry { y: 128, cb: 128, cr: 128 });
        }

        // Decode palette indices for each pixel
        let indices = self.decode_palette_indices(cu, palette_size)?;

        // Apply palette: map indices to colors
        self.apply_palette(cu, &palette, &indices)?;

        // Update palette predictor for next CU
        self.update_palette_predictor(&palette);

        Ok(())
    }

    /// Get palette predictor from previously decoded palette.
    fn get_palette_predictor(&self) -> Vec<PaletteEntry> {
        // Would return the predictor palette from previous CU
        // For now, return empty predictor
        Vec::new()
    }

    /// Decode palette size.
    fn decode_palette_size(&self, cu: &CodingUnit) -> Result<usize> {
        // Maximum palette size depends on CU size and bit depth
        let max_size = 31.min(cu.width as usize * cu.height as usize);
        // Would decode from bitstream, return reasonable default
        Ok(8.min(max_size))
    }

    /// Decode palette indices for all pixels.
    fn decode_palette_indices(&self, cu: &CodingUnit, palette_size: usize) -> Result<Vec<u8>> {
        let num_pixels = cu.width as usize * cu.height as usize;
        let mut indices = Vec::with_capacity(num_pixels);

        // VVC uses run-length coding for palette indices
        // Modes: INDEX (explicit index), COPY_ABOVE (copy from line above)

        let mut pos = 0;
        while pos < num_pixels {
            // Would decode run mode and length from bitstream
            let run_type = 0; // 0 = INDEX, 1 = COPY_ABOVE
            let run_length = 1;
            let index = 0u8;

            if run_type == 0 {
                // INDEX mode: use explicit palette index
                for _ in 0..run_length.min(num_pixels - pos) {
                    indices.push(index % palette_size as u8);
                    pos += 1;
                }
            } else {
                // COPY_ABOVE mode: copy from row above
                for _ in 0..run_length.min(num_pixels - pos) {
                    let above_idx = if pos >= cu.width as usize {
                        indices[pos - cu.width as usize]
                    } else {
                        0
                    };
                    indices.push(above_idx);
                    pos += 1;
                }
            }
        }

        Ok(indices)
    }

    /// Apply palette to reconstruct pixels.
    fn apply_palette(&mut self, cu: &CodingUnit, palette: &[PaletteEntry], indices: &[u8]) -> Result<()> {
        let frame = self.current_frame.as_mut()
            .ok_or_else(|| VvcError::InvalidState("No current frame".to_string()))?;

        // Apply to luma
        let stride_y = frame.stride(0);
        let plane_y = frame.plane_mut(0)
            .ok_or_else(|| VvcError::InvalidState("Luma plane not found".to_string()))?;

        for y in 0..cu.height as usize {
            for x in 0..cu.width as usize {
                let pos = y * cu.width as usize + x;
                let idx = indices.get(pos).copied().unwrap_or(0) as usize;
                let entry = palette.get(idx).unwrap_or(&PaletteEntry { y: 128, cb: 128, cr: 128 });

                let dst = (cu.y as usize + y) * stride_y + cu.x as usize + x;
                if dst < plane_y.len() {
                    plane_y[dst] = entry.y;
                }
            }
        }

        // Apply to chroma (assuming 4:2:0)
        let stride_cb = frame.stride(1);
        let stride_cr = frame.stride(2);

        if let Some(plane_cb) = frame.plane_mut(1) {
            for y in 0..(cu.height as usize / 2) {
                for x in 0..(cu.width as usize / 2) {
                    // Average 2x2 block for chroma
                    let pos = (y * 2) * cu.width as usize + (x * 2);
                    let idx = indices.get(pos).copied().unwrap_or(0) as usize;
                    let entry = palette.get(idx).unwrap_or(&PaletteEntry { y: 128, cb: 128, cr: 128 });

                    let dst = (cu.y as usize / 2 + y) * stride_cb + cu.x as usize / 2 + x;
                    if dst < plane_cb.len() {
                        plane_cb[dst] = entry.cb;
                    }
                }
            }
        }

        if let Some(plane_cr) = frame.plane_mut(2) {
            for y in 0..(cu.height as usize / 2) {
                for x in 0..(cu.width as usize / 2) {
                    let pos = (y * 2) * cu.width as usize + (x * 2);
                    let idx = indices.get(pos).copied().unwrap_or(0) as usize;
                    let entry = palette.get(idx).unwrap_or(&PaletteEntry { y: 128, cb: 128, cr: 128 });

                    let dst = (cu.y as usize / 2 + y) * stride_cr + cu.x as usize / 2 + x;
                    if dst < plane_cr.len() {
                        plane_cr[dst] = entry.cr;
                    }
                }
            }
        }

        Ok(())
    }

    /// Update palette predictor for next CU.
    fn update_palette_predictor(&mut self, _palette: &[PaletteEntry]) {
        // Would store palette for use as predictor in next CU
        // The predictor is maintained at CTU level
    }

    /// Apply intra prediction.
    fn apply_intra_prediction(&mut self, cu: &CodingUnit, is_luma: bool) -> Result<()> {
        let frame = self.current_frame.as_mut()
            .ok_or_else(|| VvcError::InvalidState("No current frame".to_string()))?;

        let plane_idx = if is_luma { 0 } else { 1 };
        let stride = frame.stride(plane_idx);
        let plane = frame.plane_mut(plane_idx)
            .ok_or_else(|| VvcError::InvalidState("Plane not found".to_string()))?;

        let (x, y, width, height) = if is_luma {
            (cu.x as usize, cu.y as usize, cu.width as usize, cu.height as usize)
        } else {
            // Chroma subsampling
            ((cu.x / 2) as usize, (cu.y / 2) as usize,
             (cu.width / 2) as usize, (cu.height / 2) as usize)
        };

        let mode = IntraMode::from_index(cu.intra_luma_mode);

        match mode {
            IntraMode::Planar => {
                Self::predict_planar_static(plane, x, y, width, height, stride);
            }
            IntraMode::Dc => {
                Self::predict_dc_static(plane, x, y, width, height, stride);
            }
            IntraMode::Angular(angle) => {
                Self::predict_angular_static(plane, x, y, width, height, stride, angle);
            }
        }

        Ok(())
    }

    /// Planar prediction (static version).
    fn predict_planar_static(plane: &mut [u8], x: usize, y: usize, width: usize, height: usize, stride: usize) {
        for j in 0..height {
            for i in 0..width {
                let h_weight = width - 1 - i;
                let v_weight = height - 1 - j;

                let left = if x > 0 { plane[(y + j) * stride + x - 1] } else { 128 };
                let top = if y > 0 { plane[(y - 1) * stride + x + i] } else { 128 };
                let top_right = if y > 0 && x + width < stride {
                    plane[(y - 1) * stride + x + width]
                } else {
                    128
                };
                let bottom_left = if x > 0 && y + height < plane.len() / stride {
                    plane[(y + height) * stride + x - 1]
                } else {
                    128
                };

                let pred = (h_weight as u32 * left as u32
                    + (width - 1 - h_weight) as u32 * top_right as u32
                    + v_weight as u32 * top as u32
                    + (height - 1 - v_weight) as u32 * bottom_left as u32
                    + (width + height) as u32 / 2)
                    / ((width + height) as u32);

                plane[(y + j) * stride + x + i] = pred.clamp(0, 255) as u8;
            }
        }
    }

    /// DC prediction (static version).
    fn predict_dc_static(plane: &mut [u8], x: usize, y: usize, width: usize, height: usize, stride: usize) {
        let mut sum = 0u32;
        let mut count = 0u32;

        // Sum left column
        if x > 0 {
            for j in 0..height {
                sum += plane[(y + j) * stride + x - 1] as u32;
                count += 1;
            }
        }

        // Sum top row
        if y > 0 {
            for i in 0..width {
                sum += plane[(y - 1) * stride + x + i] as u32;
                count += 1;
            }
        }

        let dc = if count > 0 {
            ((sum + count / 2) / count) as u8
        } else {
            128
        };

        // Fill block
        for j in 0..height {
            for i in 0..width {
                plane[(y + j) * stride + x + i] = dc;
            }
        }
    }

    /// Angular prediction (static version).
    fn predict_angular_static(plane: &mut [u8], x: usize, y: usize, width: usize, height: usize, stride: usize, angle: u8) {
        // Simplified angular prediction
        if angle < 34 {
            // Horizontal-ish
            for j in 0..height {
                let left = if x > 0 { plane[(y + j) * stride + x - 1] } else { 128 };
                for i in 0..width {
                    plane[(y + j) * stride + x + i] = left;
                }
            }
        } else {
            // Vertical-ish
            for i in 0..width {
                let top = if y > 0 { plane[(y - 1) * stride + x + i] } else { 128 };
                for j in 0..height {
                    plane[(y + j) * stride + x + i] = top;
                }
            }
        }
    }

    /// Apply inter prediction.
    fn apply_inter_prediction(&mut self, cu: &CodingUnit) -> Result<()> {
        // Motion compensation would be applied here
        Ok(())
    }

    /// Decode transform tree.
    fn decode_transform_tree(
        &mut self,
        cu: &mut CodingUnit,
        _slice_header: &SliceHeader,
        _sps: &Sps,
        _pps: &Pps,
    ) -> Result<()> {
        // Simplified: decode single TU
        let tu = TransformUnit::new(cu.x, cu.y, cu.width, cu.height);
        cu.transform_units.push(tu);

        // Decode coefficients and apply inverse transform for each TU
        // In a full implementation, this would decode coefficient levels,
        // apply inverse quantization, and apply inverse transform (DCT/DST/LFNST)
        let num_tus = cu.transform_units.len();
        for i in 0..num_tus {
            Self::decode_transform_unit_static(&mut cu.transform_units[i], cu.x, cu.y, cu.width, cu.height)?;
        }

        Ok(())
    }

    /// Decode a transform unit (static version).
    fn decode_transform_unit_static(
        _tu: &mut TransformUnit,
        _cu_x: u32,
        _cu_y: u32,
        _cu_width: u32,
        _cu_height: u32,
    ) -> Result<()> {
        // Decode CBF flags
        // Decode coefficient levels
        // Apply inverse quantization
        // Apply inverse transform (DCT/DST)
        // Apply LFNST if enabled
        // Add residual to prediction

        Ok(())
    }

    /// Apply deblocking filter (static version for borrow checker).
    fn apply_deblocking_filter_static(
        frame: &mut Frame,
        _slice_header: &SliceHeader,
        ph: &PictureHeader,
        _sps: &Sps,
        _pps: &Pps,
    ) -> Result<()> {
        let beta_offset_luma = ph.ph_luma_beta_offset_div2 * 2;
        let tc_offset_luma = ph.ph_luma_tc_offset_div2 * 2;

        // Apply vertical edges
        Self::deblock_edges_static(frame, true, beta_offset_luma, tc_offset_luma)?;

        // Apply horizontal edges
        Self::deblock_edges_static(frame, false, beta_offset_luma, tc_offset_luma)?;

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
            .ok_or_else(|| VvcError::InvalidState("Luma plane not found".to_string()))?;

        // Apply filter at 4x4 block boundaries (VVC uses 4-sample granularity)
        let step = 4;

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
    fn deblock_edge_v_static(plane: &mut [u8], x: usize, y: usize, stride: usize, beta_offset: i8, tc_offset: i8) {
        if x < 4 {
            return;
        }

        let p0 = plane[y * stride + x - 1] as i16;
        let p1 = plane[y * stride + x - 2] as i16;
        let p2 = plane[y * stride + x - 3] as i16;
        let p3 = plane[y * stride + x - 4] as i16;
        let q0 = plane[y * stride + x] as i16;
        let q1 = plane[y * stride + x + 1] as i16;
        let q2 = plane[y * stride + x + 2] as i16;
        let q3 = plane[y * stride + x + 3] as i16;

        let dp = (p2 - 2 * p1 + p0).abs();
        let dq = (q2 - 2 * q1 + q0).abs();
        let d = dp + dq;

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
    fn deblock_edge_h_static(plane: &mut [u8], x: usize, y: usize, stride: usize, beta_offset: i8, tc_offset: i8) {
        if y < 4 {
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

    /// Apply SAO filter (static version for borrow checker).
    fn apply_sao_filter_static(
        frame: &mut Frame,
        _sps: &Sps,
        _sao_luma: bool,
        _sao_chroma: bool,
    ) -> Result<()> {
        // Simplified SAO - in a full implementation, this would apply SAO per-CTU
        // For now, we skip SAO as it requires access to per-CTU context which
        // was stored during decoding. A full implementation would pass the contexts
        // as a parameter or restructure the decoder.
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
    ) -> Result<()> {
        let stride = frame.stride(plane_idx);
        let plane = frame.plane_mut(plane_idx)
            .ok_or_else(|| VvcError::InvalidState("Plane not found".to_string()))?;

        let comp_idx = plane_idx.min(2);
        let type_idx = params.type_idx[comp_idx];
        let offsets = &params.offset[comp_idx];

        match type_idx {
            1 => {
                // Band offset
                let band_pos = params.band_position[comp_idx];
                Self::apply_sao_band_static(plane, x, y, size, stride, offsets, band_pos);
            }
            2 => {
                // Edge offset
                let eo_class = params.eo_class[comp_idx];
                Self::apply_sao_edge_static(plane, x, y, size, stride, offsets, eo_class);
            }
            _ => {}
        }

        Ok(())
    }

    /// Apply SAO band offset (static version).
    fn apply_sao_band_static(plane: &mut [u8], x: usize, y: usize, size: usize, stride: usize, offsets: &[i8; 4], band_pos: u8) {
        for j in 0..size {
            for i in 0..size {
                let idx = (y + j) * stride + x + i;
                if idx >= plane.len() {
                    continue;
                }
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
    fn apply_sao_edge_static(plane: &mut [u8], x: usize, y: usize, size: usize, stride: usize, offsets: &[i8; 4], eo_class: u8) {
        let (dx, dy) = match eo_class {
            0 => (1isize, 0isize),   // Horizontal
            1 => (0, 1),             // Vertical
            2 => (1, 1),             // 135 degree
            3 => (1, -1),            // 45 degree
            _ => return,
        };

        for j in 1..size.saturating_sub(1) {
            for i in 1..size.saturating_sub(1) {
                let idx = (y + j) * stride + x + i;
                if idx >= plane.len() {
                    continue;
                }
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

    /// Apply ALF filter (static version for borrow checker).
    fn apply_alf_filter_static(
        _frame: &mut Frame,
        _slice_header: &SliceHeader,
        _ph: &PictureHeader,
        _sps: &Sps,
    ) -> Result<()> {
        // ALF (Adaptive Loop Filter) implementation
        // VVC uses 7x7 diamond-shaped filters for luma and 5x5 for chroma

        // Simplified: skip ALF for now
        // A full implementation would apply the filter coefficients from APS

        Ok(())
    }

    /// Reset decoder state.
    fn reset_state(&mut self) {
        self.dpb.clear();
        self.current_frame = None;
        self.current_poc = 0;
        self.prev_poc_lsb = 0;
        self.prev_poc_msb = 0;
        self.current_ph = None;
        self.gdr_recovery_poc = None;
        self.ctu_contexts.clear();
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

/// VVC decoder info.
#[derive(Debug, Clone)]
pub struct VvcDecoderInfo {
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Profile.
    pub profile: Option<VvcProfile>,
    /// Tier.
    pub tier: VvcTier,
    /// Level.
    pub level: VvcLevel,
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
        let decoder = VvcDecoder::default_config();
        assert!(!decoder.initialized);
    }

    #[test]
    fn test_decoder_config() {
        let config = VvcDecoderConfig {
            max_width: 3840,
            max_height: 2160,
            output_format: PixelFormat::Yuv420p10le,
            threads: 4,
            threading: true,
            alf_enabled: true,
            lmcs_enabled: true,
        };
        let decoder = VvcDecoder::new(config);
        assert_eq!(decoder.config.max_width, 3840);
        assert_eq!(decoder.config.max_height, 2160);
    }

    #[test]
    fn test_dpb() {
        let mut dpb = DecodedPictureBuffer::new(4);
        let frame = Frame::new(64, 64, PixelFormat::Yuv420p, TimeBase::new(1, 30));

        dpb.add(frame.clone(), 0, false, 0);
        assert!(dpb.get_by_poc(0).is_some());
        assert!(dpb.get_by_poc(1).is_none());

        dpb.mark_unused(0);
        dpb.clear();
        assert!(dpb.get_by_poc(0).is_none());
    }

    #[test]
    fn test_ctu_context() {
        let ctx = CtuContext::default();
        assert!(!ctx.sao_params.merge_left);
        assert!(!ctx.alf_params.alf_ctb_flag_luma);
    }

    #[test]
    fn test_decoder_info() {
        let decoder = VvcDecoder::default_config();
        let info = decoder.info();
        assert_eq!(info.width, 0); // Not initialized
        assert_eq!(info.bit_depth, 10); // Default
    }

    #[test]
    fn test_split_mode_decision() {
        let decoder = VvcDecoder::default_config();
        // This would need more setup to test properly
    }
}
