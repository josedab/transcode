//! VVC NAL (Network Abstraction Layer) unit types and parsing.
//!
//! This module defines VVC NAL unit types including:
//! - VPS_NUT, SPS_NUT, PPS_NUT, PH_NUT
//! - Slice types (I, P, B)
//! - SEI messages
//! - Parameter set structures

#![allow(clippy::too_many_arguments)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::unnecessary_cast)]
#![allow(dead_code)]

use crate::error::{Result, VvcError, VvcLevel, VvcProfile, VvcTier};
use transcode_core::bitstream::BitReader;

/// VVC NAL unit types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum NalUnitType {
    /// Coded slice of a trailing picture (non-reference).
    TrailNut = 0,
    /// Coded slice of a STSA picture (non-reference).
    StsaNut = 1,
    /// Coded slice of a RADL picture.
    RadlNut = 2,
    /// Coded slice of a RASL picture.
    RaslNut = 3,
    /// Reserved non-IRAP VCL NAL unit types.
    RsvVcl4 = 4,
    RsvVcl5 = 5,
    RsvVcl6 = 6,
    /// Coded slice of an IDR picture (N_LP).
    IdrNLp = 7,
    /// Coded slice of an IDR picture (W_RADL).
    IdrWRadl = 8,
    /// Coded slice of a CRA picture.
    CraNut = 9,
    /// Coded slice of a GDR picture.
    GdrNut = 10,
    /// Reserved IRAP VCL NAL unit types.
    RsvIrap11 = 11,
    /// Operating point information.
    OpiNut = 12,
    /// Decoding capability information.
    DciNut = 13,
    /// Video parameter set.
    VpsNut = 14,
    /// Sequence parameter set.
    SpsNut = 15,
    /// Picture parameter set.
    PpsNut = 16,
    /// Adaptation parameter set.
    ApsPrefix = 17,
    ApsNut = 18,
    /// Picture header.
    PhNut = 19,
    /// Access unit delimiter.
    AudNut = 20,
    /// End of sequence.
    EosNut = 21,
    /// End of bitstream.
    EobNut = 22,
    /// Prefix SEI message.
    PrefixSeiNut = 23,
    /// Suffix SEI message.
    SuffixSeiNut = 24,
    /// Filler data.
    FdNut = 25,
    /// Reserved non-VCL NAL unit types.
    RsvNvcl26 = 26,
    RsvNvcl27 = 27,
    /// Unspecified non-VCL NAL unit types.
    Unspec28 = 28,
    Unspec29 = 29,
    Unspec30 = 30,
    Unspec31 = 31,
}

impl NalUnitType {
    /// Create from raw NAL unit type value.
    pub fn from_raw(val: u8) -> Self {
        Self::from_u8(val).unwrap_or(Self::Unspec31)
    }

    /// Convert to raw NAL unit type value.
    pub fn to_raw(&self) -> u8 {
        *self as u8
    }

    /// Create from NAL unit type value.
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::TrailNut),
            1 => Some(Self::StsaNut),
            2 => Some(Self::RadlNut),
            3 => Some(Self::RaslNut),
            4 => Some(Self::RsvVcl4),
            5 => Some(Self::RsvVcl5),
            6 => Some(Self::RsvVcl6),
            7 => Some(Self::IdrNLp),
            8 => Some(Self::IdrWRadl),
            9 => Some(Self::CraNut),
            10 => Some(Self::GdrNut),
            11 => Some(Self::RsvIrap11),
            12 => Some(Self::OpiNut),
            13 => Some(Self::DciNut),
            14 => Some(Self::VpsNut),
            15 => Some(Self::SpsNut),
            16 => Some(Self::PpsNut),
            17 => Some(Self::ApsPrefix),
            18 => Some(Self::ApsNut),
            19 => Some(Self::PhNut),
            20 => Some(Self::AudNut),
            21 => Some(Self::EosNut),
            22 => Some(Self::EobNut),
            23 => Some(Self::PrefixSeiNut),
            24 => Some(Self::SuffixSeiNut),
            25 => Some(Self::FdNut),
            26 => Some(Self::RsvNvcl26),
            27 => Some(Self::RsvNvcl27),
            28 => Some(Self::Unspec28),
            29 => Some(Self::Unspec29),
            30 => Some(Self::Unspec30),
            31 => Some(Self::Unspec31),
            _ => None,
        }
    }

    /// Check if this is an IDR picture.
    pub fn is_idr(&self) -> bool {
        matches!(self, Self::IdrNLp | Self::IdrWRadl)
    }

    /// Check if this is an IRAP (Intra Random Access Point) picture.
    pub fn is_irap(&self) -> bool {
        matches!(self, Self::IdrNLp | Self::IdrWRadl | Self::CraNut | Self::GdrNut)
    }

    /// Check if this is a VCL (Video Coding Layer) NAL unit.
    pub fn is_vcl(&self) -> bool {
        (*self as u8) <= 12
    }

    /// Check if this is a reference picture.
    pub fn is_reference(&self) -> bool {
        self.is_irap() || matches!(self, Self::TrailNut | Self::StsaNut)
    }

    /// Check if this is a leading picture.
    pub fn is_leading(&self) -> bool {
        matches!(self, Self::RadlNut | Self::RaslNut)
    }

    /// Check if this is a trailing picture.
    pub fn is_trailing(&self) -> bool {
        matches!(self, Self::TrailNut | Self::StsaNut)
    }

    /// Check if this is a GDR (Gradual Decoder Refresh) picture.
    pub fn is_gdr(&self) -> bool {
        matches!(self, Self::GdrNut)
    }
}

/// VVC NAL unit header.
#[derive(Debug, Clone, Copy)]
pub struct NalUnitHeader {
    /// NAL unit type.
    pub nal_unit_type: NalUnitType,
    /// Layer ID.
    pub nuh_layer_id: u8,
    /// Temporal ID + 1.
    pub nuh_temporal_id_plus1: u8,
}

impl NalUnitHeader {
    /// Parse NAL unit header from bitstream.
    pub fn parse(reader: &mut BitReader) -> Result<Self> {
        // forbidden_zero_bit
        let forbidden = reader.read_bit()?;
        if forbidden {
            return Err(VvcError::NalUnit("Forbidden zero bit is set".to_string()));
        }

        // nuh_reserved_zero_bit
        let reserved = reader.read_bit()?;
        if reserved {
            return Err(VvcError::NalUnit("Reserved zero bit is set".to_string()));
        }

        // nuh_layer_id (6 bits)
        let nuh_layer_id = reader.read_bits(6)? as u8;

        // nal_unit_type (5 bits)
        let nal_type_val = reader.read_bits(5)? as u8;
        let nal_unit_type = NalUnitType::from_u8(nal_type_val)
            .ok_or_else(|| VvcError::NalUnit(format!("Unknown NAL type: {}", nal_type_val)))?;

        // nuh_temporal_id_plus1 (3 bits)
        let nuh_temporal_id_plus1 = reader.read_bits(3)? as u8;

        if nuh_temporal_id_plus1 == 0 {
            return Err(VvcError::NalUnit("Temporal ID cannot be 0".to_string()));
        }

        Ok(Self {
            nal_unit_type,
            nuh_layer_id,
            nuh_temporal_id_plus1,
        })
    }

    /// Write NAL unit header to bitstream.
    pub fn write(&self, writer: &mut transcode_core::bitstream::BitWriter) -> Result<()> {
        writer.write_bit(false)?; // forbidden_zero_bit
        writer.write_bit(false)?; // nuh_reserved_zero_bit
        writer.write_bits(self.nuh_layer_id as u32, 6)?;
        writer.write_bits(self.nal_unit_type as u32, 5)?;
        writer.write_bits(self.nuh_temporal_id_plus1 as u32, 3)?;
        Ok(())
    }

    /// Get temporal ID (0-based).
    pub fn temporal_id(&self) -> u8 {
        self.nuh_temporal_id_plus1 - 1
    }
}

/// VVC slice type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SliceType {
    /// B slice (bi-predictive).
    B = 0,
    /// P slice (predictive).
    P = 1,
    /// I slice (intra).
    #[default]
    I = 2,
}

impl SliceType {
    /// Create from slice type value.
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::B),
            1 => Some(Self::P),
            2 => Some(Self::I),
            _ => None,
        }
    }

    /// Check if this is an intra slice.
    pub fn is_intra(&self) -> bool {
        matches!(self, Self::I)
    }

    /// Check if this slice uses inter prediction.
    pub fn is_inter(&self) -> bool {
        !self.is_intra()
    }

    /// Check if this is a B slice.
    pub fn is_b(&self) -> bool {
        matches!(self, Self::B)
    }
}

/// Profile tier level structure.
#[derive(Debug, Clone, Default)]
pub struct ProfileTierLevel {
    /// General profile IDC.
    pub general_profile_idc: u8,
    /// General tier flag.
    pub general_tier_flag: bool,
    /// General level IDC.
    pub general_level_idc: u8,
    /// PTL frame only constraint flag.
    pub ptl_frame_only_constraint_flag: bool,
    /// PTL multilayer enabled flag.
    pub ptl_multilayer_enabled_flag: bool,
    /// General constraint info.
    pub general_constraint_info: Vec<u8>,
    /// Sub-layer level present flags.
    pub ptl_sublayer_level_present_flag: Vec<bool>,
    /// Sub-layer level IDC values.
    pub sublayer_level_idc: Vec<u8>,
    /// PTL number of sub-profiles.
    pub ptl_num_sub_profiles: u8,
    /// General sub-profile IDC values.
    pub general_sub_profile_idc: Vec<u32>,
}

impl ProfileTierLevel {
    /// Parse profile tier level from bitstream.
    pub fn parse(
        reader: &mut BitReader,
        profile_tier_present_flag: bool,
        max_num_sub_layers_minus1: u8,
    ) -> Result<Self> {
        let mut ptl = Self::default();

        if profile_tier_present_flag {
            ptl.general_profile_idc = reader.read_bits(7)? as u8;
            ptl.general_tier_flag = reader.read_bit()?;
        }

        ptl.general_level_idc = reader.read_bits(8)? as u8;

        ptl.ptl_frame_only_constraint_flag = reader.read_bit()?;
        ptl.ptl_multilayer_enabled_flag = reader.read_bit()?;

        if profile_tier_present_flag {
            // Parse general constraint info
            let gci_present_flag = reader.read_bit()?;
            if gci_present_flag {
                // General constraint info (simplified - would be 81 bits)
                for _ in 0..11 {
                    ptl.general_constraint_info.push(reader.read_bits(8)? as u8);
                }
            }

            // Byte alignment
            while !reader.is_byte_aligned() {
                reader.read_bit()?;
            }
        }

        // Sub-layer level info
        for _ in 0..max_num_sub_layers_minus1 {
            ptl.ptl_sublayer_level_present_flag.push(reader.read_bit()?);
        }

        // Byte alignment
        while !reader.is_byte_aligned() {
            reader.read_bit()?;
        }

        for i in 0..max_num_sub_layers_minus1 as usize {
            if ptl.ptl_sublayer_level_present_flag.get(i).copied().unwrap_or(false) {
                ptl.sublayer_level_idc.push(reader.read_bits(8)? as u8);
            }
        }

        if profile_tier_present_flag {
            ptl.ptl_num_sub_profiles = reader.read_bits(8)? as u8;
            for _ in 0..ptl.ptl_num_sub_profiles {
                ptl.general_sub_profile_idc.push(reader.read_bits(32)?);
            }
        }

        Ok(ptl)
    }

    /// Get the profile.
    pub fn profile(&self) -> Option<VvcProfile> {
        VvcProfile::from_idc(self.general_profile_idc)
    }

    /// Get the tier.
    pub fn tier(&self) -> VvcTier {
        VvcTier::from_flag(self.general_tier_flag)
    }

    /// Get the level.
    pub fn level(&self) -> VvcLevel {
        VvcLevel::from_idc(self.general_level_idc).unwrap_or_default()
    }
}

/// Video Parameter Set (VPS).
#[derive(Debug, Clone, Default)]
pub struct Vps {
    /// VPS ID.
    pub vps_video_parameter_set_id: u8,
    /// Maximum layers minus 1.
    pub vps_max_layers_minus1: u8,
    /// Maximum sub-layers minus 1.
    pub vps_max_sublayers_minus1: u8,
    /// Default PTL DPB HRD max TID flag.
    pub vps_default_ptl_dpb_hrd_max_tid_flag: bool,
    /// All independent layers flag.
    pub vps_all_independent_layers_flag: bool,
    /// Layer IDs.
    pub vps_layer_id: Vec<u8>,
    /// Independent layer flags.
    pub vps_independent_layer_flag: Vec<bool>,
    /// Max TID ref present flag.
    pub vps_max_tid_ref_present_flag: Vec<Vec<bool>>,
    /// Direct ref layer flags.
    pub vps_direct_ref_layer_flag: Vec<Vec<bool>>,
    /// Each layer is an OLS flag.
    pub vps_each_layer_is_an_ols_flag: bool,
    /// OLS mode IDC.
    pub vps_ols_mode_idc: u8,
    /// Number of output layer sets minus 2.
    pub vps_num_output_layer_sets_minus2: u16,
    /// PTL max TID.
    pub vps_ptl_max_tid: Vec<u8>,
    /// Profile tier level.
    pub profile_tier_level: Vec<ProfileTierLevel>,
    /// Extension flag.
    pub vps_extension_flag: bool,
}

impl Vps {
    /// Parse VPS from RBSP data.
    pub fn parse(data: &[u8]) -> Result<Self> {
        let mut reader = BitReader::new(data);
        let mut vps = Self::default();

        vps.vps_video_parameter_set_id = reader.read_bits(4)? as u8;
        vps.vps_max_layers_minus1 = reader.read_bits(6)? as u8;
        vps.vps_max_sublayers_minus1 = reader.read_bits(3)? as u8;

        if vps.vps_max_layers_minus1 > 0 && vps.vps_max_sublayers_minus1 > 0 {
            vps.vps_default_ptl_dpb_hrd_max_tid_flag = reader.read_bit()?;
        }

        if vps.vps_max_layers_minus1 > 0 {
            vps.vps_all_independent_layers_flag = reader.read_bit()?;
        } else {
            vps.vps_all_independent_layers_flag = true;
        }

        // Layer info
        for i in 0..=vps.vps_max_layers_minus1 {
            vps.vps_layer_id.push(reader.read_bits(6)? as u8);

            if i > 0 && !vps.vps_all_independent_layers_flag {
                let independent = reader.read_bit()?;
                vps.vps_independent_layer_flag.push(independent);

                if !independent {
                    let mut max_tid_ref = Vec::new();
                    let mut direct_ref = Vec::new();
                    for _j in 0..i {
                        direct_ref.push(reader.read_bit()?);
                        if vps.vps_max_sublayers_minus1 > 0 && *direct_ref.last().unwrap() {
                            max_tid_ref.push(reader.read_bit()?);
                        }
                    }
                    vps.vps_max_tid_ref_present_flag.push(max_tid_ref);
                    vps.vps_direct_ref_layer_flag.push(direct_ref);
                }
            }
        }

        // Additional VPS syntax (simplified)
        if vps.vps_max_layers_minus1 > 0 {
            if vps.vps_all_independent_layers_flag {
                vps.vps_each_layer_is_an_ols_flag = reader.read_bit()?;
            }
            if !vps.vps_each_layer_is_an_ols_flag {
                if !vps.vps_all_independent_layers_flag {
                    vps.vps_ols_mode_idc = reader.read_bits(2)? as u8;
                }
            }
        }

        vps.vps_extension_flag = reader.read_bit()?;

        Ok(vps)
    }

    /// Get the maximum number of layers.
    pub fn max_layers(&self) -> u8 {
        self.vps_max_layers_minus1 + 1
    }

    /// Get the maximum number of sub-layers.
    pub fn max_sublayers(&self) -> u8 {
        self.vps_max_sublayers_minus1 + 1
    }
}

/// Sequence Parameter Set (SPS).
#[derive(Debug, Clone, Default)]
pub struct Sps {
    /// SPS ID.
    pub sps_seq_parameter_set_id: u8,
    /// VPS ID.
    pub sps_video_parameter_set_id: u8,
    /// Maximum sub-layers minus 1.
    pub sps_max_sublayers_minus1: u8,
    /// Chroma format IDC.
    pub sps_chroma_format_idc: u8,
    /// Log2 CTU size minus 5.
    pub sps_log2_ctu_size_minus5: u8,
    /// PTL DPB HRD params present flag.
    pub sps_ptl_dpb_hrd_params_present_flag: bool,
    /// Profile tier level.
    pub profile_tier_level: ProfileTierLevel,
    /// GDR enabled flag.
    pub sps_gdr_enabled_flag: bool,
    /// Reference picture resampling enabled flag.
    pub sps_ref_pic_resampling_enabled_flag: bool,
    /// Resource present in SPS flag.
    pub sps_res_change_in_clvs_allowed_flag: bool,
    /// Picture width max in luma samples.
    pub sps_pic_width_max_in_luma_samples: u32,
    /// Picture height max in luma samples.
    pub sps_pic_height_max_in_luma_samples: u32,
    /// Conformance window flag.
    pub sps_conformance_window_flag: bool,
    /// Conformance window left offset.
    pub sps_conf_win_left_offset: u32,
    /// Conformance window right offset.
    pub sps_conf_win_right_offset: u32,
    /// Conformance window top offset.
    pub sps_conf_win_top_offset: u32,
    /// Conformance window bottom offset.
    pub sps_conf_win_bottom_offset: u32,
    /// Subpic info present flag.
    pub sps_subpic_info_present_flag: bool,
    /// Number of subpics minus 1.
    pub sps_num_subpics_minus1: u16,
    /// Independent subpics flag.
    pub sps_independent_subpics_flag: bool,
    /// Subpic same size flag.
    pub sps_subpic_same_size_flag: bool,
    /// Bit depth minus 8.
    pub sps_bitdepth_minus8: u8,
    /// Entropy coding sync enabled flag.
    pub sps_entropy_coding_sync_enabled_flag: bool,
    /// Entry point offsets present flag.
    pub sps_entry_point_offsets_present_flag: bool,
    /// Log2 max picture order count LSB minus 4.
    pub sps_log2_max_pic_order_cnt_lsb_minus4: u8,
    /// POC MSB cycle flag.
    pub sps_poc_msb_cycle_flag: bool,
    /// POC MSB cycle len minus 1.
    pub sps_poc_msb_cycle_len_minus1: u8,
    /// Number of extra PH bits.
    pub sps_num_extra_ph_bytes: u8,
    /// Number of extra SH bits.
    pub sps_num_extra_sh_bytes: u8,
    /// Extra PH bit present flag.
    pub sps_extra_ph_bit_present_flag: Vec<bool>,
    /// Extra SH bit present flag.
    pub sps_extra_sh_bit_present_flag: Vec<bool>,
    /// Sub-layer DPB params flag.
    pub sps_sublayer_dpb_params_flag: bool,
    /// Log2 min luma CB size minus 2.
    pub sps_log2_min_luma_coding_block_size_minus2: u8,
    /// Partition constraints override enabled flag.
    pub sps_partition_constraints_override_enabled_flag: bool,
    /// Log2 diff min QT min CB intra slice luma.
    pub sps_log2_diff_min_qt_min_cb_intra_slice_luma: u8,
    /// Max MTT hierarchy depth intra slice luma.
    pub sps_max_mtt_hierarchy_depth_intra_slice_luma: u8,
    /// Log2 diff max BT min QT intra slice luma.
    pub sps_log2_diff_max_bt_min_qt_intra_slice_luma: u8,
    /// Log2 diff max TT min QT intra slice luma.
    pub sps_log2_diff_max_tt_min_qt_intra_slice_luma: u8,
    /// QT/BT/TT dual tree intra flag.
    pub sps_qtbtt_dual_tree_intra_flag: bool,
    /// Log2 diff min QT min CB inter slice.
    pub sps_log2_diff_min_qt_min_cb_inter_slice: u8,
    /// Max MTT hierarchy depth inter slice.
    pub sps_max_mtt_hierarchy_depth_inter_slice: u8,
    /// Log2 diff max BT min QT inter slice.
    pub sps_log2_diff_max_bt_min_qt_inter_slice: u8,
    /// Log2 diff max TT min QT inter slice.
    pub sps_log2_diff_max_tt_min_qt_inter_slice: u8,
    /// Max luma transform size 64 flag.
    pub sps_max_luma_transform_size_64_flag: bool,
    /// Transform skip enabled flag.
    pub sps_transform_skip_enabled_flag: bool,
    /// Log2 transform skip max size minus 2.
    pub sps_log2_transform_skip_max_size_minus2: u8,
    /// BDPCM enabled flag.
    pub sps_bdpcm_enabled_flag: bool,
    /// MTS enabled flag.
    pub sps_mts_enabled_flag: bool,
    /// Explicit MTS intra enabled flag.
    pub sps_explicit_mts_intra_enabled_flag: bool,
    /// Explicit MTS inter enabled flag.
    pub sps_explicit_mts_inter_enabled_flag: bool,
    /// LFNST enabled flag.
    pub sps_lfnst_enabled_flag: bool,
    /// Joint CbCr enabled flag.
    pub sps_joint_cbcr_enabled_flag: bool,
    /// Same QP table for chroma flag.
    pub sps_same_qp_table_for_chroma_flag: bool,
    /// QP table start minus 26.
    pub sps_qp_table_start_minus26: Vec<i8>,
    /// Number delta POC ref RPS minus 1.
    pub sps_num_points_in_qp_table_minus1: Vec<u8>,
    /// Delta QP in val minus 1.
    pub sps_delta_qp_in_val_minus1: Vec<Vec<u8>>,
    /// Delta QP diff val.
    pub sps_delta_qp_diff_val: Vec<Vec<u8>>,
    /// SAO enabled flag.
    pub sps_sao_enabled_flag: bool,
    /// ALF enabled flag.
    pub sps_alf_enabled_flag: bool,
    /// CCALF enabled flag.
    pub sps_ccalf_enabled_flag: bool,
    /// LMCS enabled flag.
    pub sps_lmcs_enabled_flag: bool,
    /// Weighted pred flag.
    pub sps_weighted_pred_flag: bool,
    /// Weighted bipred flag.
    pub sps_weighted_bipred_flag: bool,
    /// Long term ref pics flag.
    pub sps_long_term_ref_pics_flag: bool,
    /// Inter layer prediction enabled flag.
    pub sps_inter_layer_prediction_enabled_flag: bool,
    /// IDR RPL present flag.
    pub sps_idr_rpl_present_flag: bool,
    /// RPL1 same as RPL0 flag.
    pub sps_rpl1_same_as_rpl0_flag: bool,
    /// Number of ref pic lists in SPS.
    pub sps_num_ref_pic_lists: [u8; 2],
    /// Ref wraparound enabled flag.
    pub sps_ref_wraparound_enabled_flag: bool,
    /// Temporal MVP enabled flag.
    pub sps_temporal_mvp_enabled_flag: bool,
    /// SbTMVP enabled flag.
    pub sps_sbtmvp_enabled_flag: bool,
    /// AMVR enabled flag.
    pub sps_amvr_enabled_flag: bool,
    /// BDOF enabled flag.
    pub sps_bdof_enabled_flag: bool,
    /// BDOF control present in PH flag.
    pub sps_bdof_control_present_in_ph_flag: bool,
    /// SMVD enabled flag.
    pub sps_smvd_enabled_flag: bool,
    /// DMVR enabled flag.
    pub sps_dmvr_enabled_flag: bool,
    /// DMVR control present in PH flag.
    pub sps_dmvr_control_present_in_ph_flag: bool,
    /// MMVD enabled flag.
    pub sps_mmvd_enabled_flag: bool,
    /// MMVD fullpel only enabled flag.
    pub sps_mmvd_fullpel_only_enabled_flag: bool,
    /// Six minus max num merge cand.
    pub sps_six_minus_max_num_merge_cand: u8,
    /// SBT enabled flag.
    pub sps_sbt_enabled_flag: bool,
    /// Affine enabled flag.
    pub sps_affine_enabled_flag: bool,
    /// Five minus max num subblock merge cand.
    pub sps_five_minus_max_num_subblock_merge_cand: u8,
    /// Six param affine enabled flag.
    pub sps_6param_affine_enabled_flag: bool,
    /// Affine amvr enabled flag.
    pub sps_affine_amvr_enabled_flag: bool,
    /// Affine prof enabled flag.
    pub sps_affine_prof_enabled_flag: bool,
    /// Prof control present in PH flag.
    pub sps_prof_control_present_in_ph_flag: bool,
    /// BCW enabled flag.
    pub sps_bcw_enabled_flag: bool,
    /// CIIP enabled flag.
    pub sps_ciip_enabled_flag: bool,
    /// GPM enabled flag.
    pub sps_gpm_enabled_flag: bool,
    /// Max num merge cand minus max num GPM cand.
    pub sps_max_num_merge_cand_minus_max_num_gpm_cand: u8,
    /// Log2 parallel merge level minus 2.
    pub sps_log2_parallel_merge_level_minus2: u8,
    /// ISP enabled flag.
    pub sps_isp_enabled_flag: bool,
    /// MRL enabled flag.
    pub sps_mrl_enabled_flag: bool,
    /// MIP enabled flag.
    pub sps_mip_enabled_flag: bool,
    /// CCLM enabled flag.
    pub sps_cclm_enabled_flag: bool,
    /// Chroma horizontal collocated flag.
    pub sps_chroma_horizontal_collocated_flag: bool,
    /// Chroma vertical collocated flag.
    pub sps_chroma_vertical_collocated_flag: bool,
    /// Palette enabled flag.
    pub sps_palette_enabled_flag: bool,
    /// ACT enabled flag.
    pub sps_act_enabled_flag: bool,
    /// Min QP prime TS minus 4.
    pub sps_min_qp_prime_ts_minus4: u8,
    /// IBC enabled flag.
    pub sps_ibc_enabled_flag: bool,
    /// Six minus max num IBC merge cand.
    pub sps_six_minus_max_num_ibc_merge_cand: u8,
    /// LADF enabled flag.
    pub sps_ladf_enabled_flag: bool,
    /// Num LADF intervals minus 2.
    pub sps_num_ladf_intervals_minus2: u8,
    /// LADF lowest interval QP offset.
    pub sps_ladf_lowest_interval_qp_offset: i8,
    /// LADF QP offset.
    pub sps_ladf_qp_offset: Vec<i8>,
    /// LADF delta threshold minus 1.
    pub sps_ladf_delta_threshold_minus1: Vec<u16>,
    /// Explicit scaling list enabled flag.
    pub sps_explicit_scaling_list_enabled_flag: bool,
    /// Scaling matrix for LFNST disabled flag.
    pub sps_scaling_matrix_for_lfnst_disabled_flag: bool,
    /// Scaling matrix for alternative colour space disabled flag.
    pub sps_scaling_matrix_for_alternative_colour_space_disabled_flag: bool,
    /// Scaling matrix designated colour space flag.
    pub sps_scaling_matrix_designated_colour_space_flag: bool,
    /// Dep quant enabled flag.
    pub sps_dep_quant_enabled_flag: bool,
    /// Sign data hiding enabled flag.
    pub sps_sign_data_hiding_enabled_flag: bool,
    /// Virtual boundaries enabled flag.
    pub sps_virtual_boundaries_enabled_flag: bool,
    /// Virtual boundaries present flag.
    pub sps_virtual_boundaries_present_flag: bool,
    /// Num ver virtual boundaries.
    pub sps_num_ver_virtual_boundaries: u8,
    /// Num hor virtual boundaries.
    pub sps_num_hor_virtual_boundaries: u8,
    /// Virtual boundary pos X minus 1.
    pub sps_virtual_boundary_pos_x_minus1: Vec<u16>,
    /// Virtual boundary pos Y minus 1.
    pub sps_virtual_boundary_pos_y_minus1: Vec<u16>,
    /// Timing HRD params present flag.
    pub sps_timing_hrd_params_present_flag: bool,
    /// Sub-layer CPB params present flag.
    pub sps_sublayer_cpb_params_present_flag: bool,
    /// Field seq flag.
    pub sps_field_seq_flag: bool,
    /// VUI parameters present flag.
    pub sps_vui_parameters_present_flag: bool,
    /// VUI payload size minus 1.
    pub sps_vui_payload_size_minus1: u16,
    /// Extension flag.
    pub sps_extension_flag: bool,
}

impl Sps {
    /// Parse SPS from RBSP data.
    pub fn parse(data: &[u8]) -> Result<Self> {
        let mut reader = BitReader::new(data);
        let mut sps = Self::default();

        sps.sps_seq_parameter_set_id = reader.read_bits(4)? as u8;
        sps.sps_video_parameter_set_id = reader.read_bits(4)? as u8;
        sps.sps_max_sublayers_minus1 = reader.read_bits(3)? as u8;
        sps.sps_chroma_format_idc = reader.read_bits(2)? as u8;
        sps.sps_log2_ctu_size_minus5 = reader.read_bits(2)? as u8;

        sps.sps_ptl_dpb_hrd_params_present_flag = reader.read_bit()?;

        if sps.sps_ptl_dpb_hrd_params_present_flag {
            sps.profile_tier_level = ProfileTierLevel::parse(
                &mut reader,
                true,
                sps.sps_max_sublayers_minus1,
            )?;
        }

        sps.sps_gdr_enabled_flag = reader.read_bit()?;
        sps.sps_ref_pic_resampling_enabled_flag = reader.read_bit()?;

        if sps.sps_ref_pic_resampling_enabled_flag {
            sps.sps_res_change_in_clvs_allowed_flag = reader.read_bit()?;
        }

        sps.sps_pic_width_max_in_luma_samples = reader.read_ue()?;
        sps.sps_pic_height_max_in_luma_samples = reader.read_ue()?;

        sps.sps_conformance_window_flag = reader.read_bit()?;
        if sps.sps_conformance_window_flag {
            sps.sps_conf_win_left_offset = reader.read_ue()?;
            sps.sps_conf_win_right_offset = reader.read_ue()?;
            sps.sps_conf_win_top_offset = reader.read_ue()?;
            sps.sps_conf_win_bottom_offset = reader.read_ue()?;
        }

        sps.sps_subpic_info_present_flag = reader.read_bit()?;
        if sps.sps_subpic_info_present_flag {
            sps.sps_num_subpics_minus1 = reader.read_ue()? as u16;
            if sps.sps_num_subpics_minus1 > 0 {
                sps.sps_independent_subpics_flag = reader.read_bit()?;
                sps.sps_subpic_same_size_flag = reader.read_bit()?;
            }
            // Additional subpic parsing would go here
        }

        sps.sps_bitdepth_minus8 = reader.read_ue()? as u8;
        sps.sps_entropy_coding_sync_enabled_flag = reader.read_bit()?;
        sps.sps_entry_point_offsets_present_flag = reader.read_bit()?;
        sps.sps_log2_max_pic_order_cnt_lsb_minus4 = reader.read_bits(4)? as u8;
        sps.sps_poc_msb_cycle_flag = reader.read_bit()?;

        if sps.sps_poc_msb_cycle_flag {
            sps.sps_poc_msb_cycle_len_minus1 = reader.read_ue()? as u8;
        }

        // Extra PH/SH bytes
        sps.sps_num_extra_ph_bytes = reader.read_bits(2)? as u8;
        for _ in 0..sps.sps_num_extra_ph_bytes * 8 {
            sps.sps_extra_ph_bit_present_flag.push(reader.read_bit()?);
        }

        sps.sps_num_extra_sh_bytes = reader.read_bits(2)? as u8;
        for _ in 0..sps.sps_num_extra_sh_bytes * 8 {
            sps.sps_extra_sh_bit_present_flag.push(reader.read_bit()?);
        }

        // DPB params
        if sps.sps_ptl_dpb_hrd_params_present_flag {
            if sps.sps_max_sublayers_minus1 > 0 {
                sps.sps_sublayer_dpb_params_flag = reader.read_bit()?;
            }
            // Parse DPB params (simplified)
        }

        // Coding block size and partition constraints
        sps.sps_log2_min_luma_coding_block_size_minus2 = reader.read_ue()? as u8;
        sps.sps_partition_constraints_override_enabled_flag = reader.read_bit()?;
        sps.sps_log2_diff_min_qt_min_cb_intra_slice_luma = reader.read_ue()? as u8;
        sps.sps_max_mtt_hierarchy_depth_intra_slice_luma = reader.read_ue()? as u8;

        if sps.sps_max_mtt_hierarchy_depth_intra_slice_luma != 0 {
            sps.sps_log2_diff_max_bt_min_qt_intra_slice_luma = reader.read_ue()? as u8;
            sps.sps_log2_diff_max_tt_min_qt_intra_slice_luma = reader.read_ue()? as u8;
        }

        if sps.sps_chroma_format_idc != 0 {
            sps.sps_qtbtt_dual_tree_intra_flag = reader.read_bit()?;
        }

        sps.sps_log2_diff_min_qt_min_cb_inter_slice = reader.read_ue()? as u8;
        sps.sps_max_mtt_hierarchy_depth_inter_slice = reader.read_ue()? as u8;

        if sps.sps_max_mtt_hierarchy_depth_inter_slice != 0 {
            sps.sps_log2_diff_max_bt_min_qt_inter_slice = reader.read_ue()? as u8;
            sps.sps_log2_diff_max_tt_min_qt_inter_slice = reader.read_ue()? as u8;
        }

        // Max transform size
        if sps.ctb_size_y() > 32 {
            sps.sps_max_luma_transform_size_64_flag = reader.read_bit()?;
        }

        // Transform and tools
        sps.sps_transform_skip_enabled_flag = reader.read_bit()?;
        if sps.sps_transform_skip_enabled_flag {
            sps.sps_log2_transform_skip_max_size_minus2 = reader.read_ue()? as u8;
            sps.sps_bdpcm_enabled_flag = reader.read_bit()?;
        }

        sps.sps_mts_enabled_flag = reader.read_bit()?;
        if sps.sps_mts_enabled_flag {
            sps.sps_explicit_mts_intra_enabled_flag = reader.read_bit()?;
            sps.sps_explicit_mts_inter_enabled_flag = reader.read_bit()?;
        }

        sps.sps_lfnst_enabled_flag = reader.read_bit()?;

        if sps.sps_chroma_format_idc != 0 {
            sps.sps_joint_cbcr_enabled_flag = reader.read_bit()?;
            sps.sps_same_qp_table_for_chroma_flag = reader.read_bit()?;
            // QP table parsing would go here
        }

        sps.sps_sao_enabled_flag = reader.read_bit()?;
        sps.sps_alf_enabled_flag = reader.read_bit()?;
        if sps.sps_alf_enabled_flag && sps.sps_chroma_format_idc != 0 {
            sps.sps_ccalf_enabled_flag = reader.read_bit()?;
        }

        sps.sps_lmcs_enabled_flag = reader.read_bit()?;
        sps.sps_weighted_pred_flag = reader.read_bit()?;
        sps.sps_weighted_bipred_flag = reader.read_bit()?;
        sps.sps_long_term_ref_pics_flag = reader.read_bit()?;

        if sps.sps_video_parameter_set_id > 0 {
            sps.sps_inter_layer_prediction_enabled_flag = reader.read_bit()?;
        }

        sps.sps_idr_rpl_present_flag = reader.read_bit()?;
        sps.sps_rpl1_same_as_rpl0_flag = reader.read_bit()?;

        // Remaining SPS syntax would continue here...
        // This is a simplified implementation

        sps.sps_extension_flag = reader.read_bit().unwrap_or(false);

        Ok(sps)
    }

    /// Get picture width.
    pub fn width(&self) -> u32 {
        if self.sps_conformance_window_flag {
            let sub_width_c = if self.sps_chroma_format_idc == 1 || self.sps_chroma_format_idc == 2 {
                2
            } else {
                1
            };
            self.sps_pic_width_max_in_luma_samples
                - sub_width_c * (self.sps_conf_win_left_offset + self.sps_conf_win_right_offset)
        } else {
            self.sps_pic_width_max_in_luma_samples
        }
    }

    /// Get picture height.
    pub fn height(&self) -> u32 {
        if self.sps_conformance_window_flag {
            let sub_height_c = if self.sps_chroma_format_idc == 1 { 2 } else { 1 };
            self.sps_pic_height_max_in_luma_samples
                - sub_height_c * (self.sps_conf_win_top_offset + self.sps_conf_win_bottom_offset)
        } else {
            self.sps_pic_height_max_in_luma_samples
        }
    }

    /// Get CTU size.
    pub fn ctb_size_y(&self) -> u32 {
        1 << (self.sps_log2_ctu_size_minus5 + 5)
    }

    /// Get log2 of CTU size.
    pub fn log2_ctb_size(&self) -> u8 {
        self.sps_log2_ctu_size_minus5 + 5
    }

    /// Get minimum coding block size.
    pub fn min_cb_size(&self) -> u32 {
        1 << (self.sps_log2_min_luma_coding_block_size_minus2 + 2)
    }

    /// Get log2 of minimum coding block size.
    pub fn log2_min_cb_size(&self) -> u8 {
        self.sps_log2_min_luma_coding_block_size_minus2 + 2
    }

    /// Get picture width in CTUs.
    pub fn pic_width_in_ctbs(&self) -> u32 {
        (self.sps_pic_width_max_in_luma_samples + self.ctb_size_y() - 1) / self.ctb_size_y()
    }

    /// Get picture height in CTUs.
    pub fn pic_height_in_ctbs(&self) -> u32 {
        (self.sps_pic_height_max_in_luma_samples + self.ctb_size_y() - 1) / self.ctb_size_y()
    }

    /// Get bit depth for luma.
    pub fn bit_depth_luma(&self) -> u8 {
        self.sps_bitdepth_minus8 + 8
    }

    /// Get bit depth for chroma.
    pub fn bit_depth_chroma(&self) -> u8 {
        self.sps_bitdepth_minus8 + 8
    }

    /// Get maximum picture order count LSB.
    pub fn max_pic_order_cnt_lsb(&self) -> u32 {
        1 << (self.sps_log2_max_pic_order_cnt_lsb_minus4 + 4)
    }
}

/// Picture Parameter Set (PPS).
#[derive(Debug, Clone, Default)]
pub struct Pps {
    /// PPS ID.
    pub pps_pic_parameter_set_id: u8,
    /// SPS ID.
    pub pps_seq_parameter_set_id: u8,
    /// Mixed NAL unit types in picture flag.
    pub pps_mixed_nalu_types_in_pic_flag: bool,
    /// Picture width in luma samples.
    pub pps_pic_width_in_luma_samples: u32,
    /// Picture height in luma samples.
    pub pps_pic_height_in_luma_samples: u32,
    /// Conformance window flag.
    pub pps_conformance_window_flag: bool,
    /// Conf win left offset.
    pub pps_conf_win_left_offset: u32,
    /// Conf win right offset.
    pub pps_conf_win_right_offset: u32,
    /// Conf win top offset.
    pub pps_conf_win_top_offset: u32,
    /// Conf win bottom offset.
    pub pps_conf_win_bottom_offset: u32,
    /// Scaling window explicit signalling flag.
    pub pps_scaling_window_explicit_signalling_flag: bool,
    /// Output flag present flag.
    pub pps_output_flag_present_flag: bool,
    /// No pic partition flag.
    pub pps_no_pic_partition_flag: bool,
    /// Subpic ID mapping present flag.
    pub pps_subpic_id_mapping_present_flag: bool,
    /// Log2 CTU size minus 5.
    pub pps_log2_ctu_size_minus5: u8,
    /// Number exp tile columns minus 1.
    pub pps_num_exp_tile_columns_minus1: u16,
    /// Number exp tile rows minus 1.
    pub pps_num_exp_tile_rows_minus1: u16,
    /// Tile column width minus 1.
    pub pps_tile_column_width_minus1: Vec<u16>,
    /// Tile row height minus 1.
    pub pps_tile_row_height_minus1: Vec<u16>,
    /// Loop filter across tiles enabled flag.
    pub pps_loop_filter_across_tiles_enabled_flag: bool,
    /// Rect slice flag.
    pub pps_rect_slice_flag: bool,
    /// Single slice per subpic flag.
    pub pps_single_slice_per_subpic_flag: bool,
    /// Number slices in pic minus 1.
    pub pps_num_slices_in_pic_minus1: u16,
    /// Tile IDX delta present flag.
    pub pps_tile_idx_delta_present_flag: bool,
    /// Loop filter across slices enabled flag.
    pub pps_loop_filter_across_slices_enabled_flag: bool,
    /// CABAC init present flag.
    pub pps_cabac_init_present_flag: bool,
    /// Num ref IDX default active minus 1.
    pub pps_num_ref_idx_default_active_minus1: [u8; 2],
    /// RPL1 IDX present flag.
    pub pps_rpl1_idx_present_flag: bool,
    /// Weighted pred flag.
    pub pps_weighted_pred_flag: bool,
    /// Weighted bipred flag.
    pub pps_weighted_bipred_flag: bool,
    /// Ref wraparound enabled flag.
    pub pps_ref_wraparound_enabled_flag: bool,
    /// Pic width minus wraparound offset.
    pub pps_pic_width_minus_wraparound_offset: u16,
    /// Init QP minus 26.
    pub pps_init_qp_minus26: i8,
    /// CU QP delta enabled flag.
    pub pps_cu_qp_delta_enabled_flag: bool,
    /// Chroma tool offsets present flag.
    pub pps_chroma_tool_offsets_present_flag: bool,
    /// CB QP offset.
    pub pps_cb_qp_offset: i8,
    /// CR QP offset.
    pub pps_cr_qp_offset: i8,
    /// Joint CbCr QP offset present flag.
    pub pps_joint_cbcr_qp_offset_present_flag: bool,
    /// Joint CbCr QP offset value.
    pub pps_joint_cbcr_qp_offset_value: i8,
    /// Slice chroma QP offsets present flag.
    pub pps_slice_chroma_qp_offsets_present_flag: bool,
    /// CU chroma QP offset list enabled flag.
    pub pps_cu_chroma_qp_offset_list_enabled_flag: bool,
    /// Chroma QP offset list len minus 1.
    pub pps_chroma_qp_offset_list_len_minus1: u8,
    /// CB QP offset list.
    pub pps_cb_qp_offset_list: Vec<i8>,
    /// CR QP offset list.
    pub pps_cr_qp_offset_list: Vec<i8>,
    /// Joint CbCr QP offset list.
    pub pps_joint_cbcr_qp_offset_list: Vec<i8>,
    /// Deblocking filter control present flag.
    pub pps_deblocking_filter_control_present_flag: bool,
    /// Deblocking filter override enabled flag.
    pub pps_deblocking_filter_override_enabled_flag: bool,
    /// Deblocking filter disabled flag.
    pub pps_deblocking_filter_disabled_flag: bool,
    /// DBF info in PH flag.
    pub pps_dbf_info_in_ph_flag: bool,
    /// Luma beta offset div2.
    pub pps_luma_beta_offset_div2: i8,
    /// Luma TC offset div2.
    pub pps_luma_tc_offset_div2: i8,
    /// CB beta offset div2.
    pub pps_cb_beta_offset_div2: i8,
    /// CB TC offset div2.
    pub pps_cb_tc_offset_div2: i8,
    /// CR beta offset div2.
    pub pps_cr_beta_offset_div2: i8,
    /// CR TC offset div2.
    pub pps_cr_tc_offset_div2: i8,
    /// RPL info in PH flag.
    pub pps_rpl_info_in_ph_flag: bool,
    /// SAO info in PH flag.
    pub pps_sao_info_in_ph_flag: bool,
    /// ALF info in PH flag.
    pub pps_alf_info_in_ph_flag: bool,
    /// WP info in PH flag.
    pub pps_wp_info_in_ph_flag: bool,
    /// QP delta info in PH flag.
    pub pps_qp_delta_info_in_ph_flag: bool,
    /// Picture header extension present flag.
    pub pps_picture_header_extension_present_flag: bool,
    /// Slice header extension present flag.
    pub pps_slice_header_extension_present_flag: bool,
    /// Extension flag.
    pub pps_extension_flag: bool,
}

impl Pps {
    /// Parse PPS from RBSP data.
    pub fn parse(data: &[u8]) -> Result<Self> {
        let mut reader = BitReader::new(data);
        let mut pps = Self::default();

        pps.pps_pic_parameter_set_id = reader.read_bits(6)? as u8;
        pps.pps_seq_parameter_set_id = reader.read_bits(4)? as u8;
        pps.pps_mixed_nalu_types_in_pic_flag = reader.read_bit()?;
        pps.pps_pic_width_in_luma_samples = reader.read_ue()?;
        pps.pps_pic_height_in_luma_samples = reader.read_ue()?;

        pps.pps_conformance_window_flag = reader.read_bit()?;
        if pps.pps_conformance_window_flag {
            pps.pps_conf_win_left_offset = reader.read_ue()?;
            pps.pps_conf_win_right_offset = reader.read_ue()?;
            pps.pps_conf_win_top_offset = reader.read_ue()?;
            pps.pps_conf_win_bottom_offset = reader.read_ue()?;
        }

        pps.pps_scaling_window_explicit_signalling_flag = reader.read_bit()?;
        // Scaling window params would go here if enabled

        pps.pps_output_flag_present_flag = reader.read_bit()?;
        pps.pps_no_pic_partition_flag = reader.read_bit()?;
        pps.pps_subpic_id_mapping_present_flag = reader.read_bit()?;

        if !pps.pps_no_pic_partition_flag {
            pps.pps_log2_ctu_size_minus5 = reader.read_bits(2)? as u8;
            pps.pps_num_exp_tile_columns_minus1 = reader.read_ue()? as u16;
            pps.pps_num_exp_tile_rows_minus1 = reader.read_ue()? as u16;

            for _ in 0..=pps.pps_num_exp_tile_columns_minus1 {
                pps.pps_tile_column_width_minus1.push(reader.read_ue()? as u16);
            }
            for _ in 0..=pps.pps_num_exp_tile_rows_minus1 {
                pps.pps_tile_row_height_minus1.push(reader.read_ue()? as u16);
            }

            // Additional tile/slice syntax
            pps.pps_loop_filter_across_tiles_enabled_flag = reader.read_bit()?;
            pps.pps_rect_slice_flag = reader.read_bit()?;

            if pps.pps_rect_slice_flag {
                pps.pps_single_slice_per_subpic_flag = reader.read_bit()?;
            }

            if pps.pps_rect_slice_flag && !pps.pps_single_slice_per_subpic_flag {
                pps.pps_num_slices_in_pic_minus1 = reader.read_ue()? as u16;
                // Slice syntax would continue here
            }

            if !pps.pps_rect_slice_flag || pps.pps_single_slice_per_subpic_flag || pps.pps_num_slices_in_pic_minus1 > 0 {
                pps.pps_tile_idx_delta_present_flag = reader.read_bit()?;
            }
        }

        pps.pps_loop_filter_across_slices_enabled_flag = reader.read_bit()?;
        pps.pps_cabac_init_present_flag = reader.read_bit()?;

        pps.pps_num_ref_idx_default_active_minus1[0] = reader.read_ue()? as u8;
        pps.pps_num_ref_idx_default_active_minus1[1] = reader.read_ue()? as u8;

        pps.pps_rpl1_idx_present_flag = reader.read_bit()?;
        pps.pps_weighted_pred_flag = reader.read_bit()?;
        pps.pps_weighted_bipred_flag = reader.read_bit()?;
        pps.pps_ref_wraparound_enabled_flag = reader.read_bit()?;

        if pps.pps_ref_wraparound_enabled_flag {
            pps.pps_pic_width_minus_wraparound_offset = reader.read_ue()? as u16;
        }

        pps.pps_init_qp_minus26 = reader.read_se()? as i8;
        pps.pps_cu_qp_delta_enabled_flag = reader.read_bit()?;
        pps.pps_chroma_tool_offsets_present_flag = reader.read_bit()?;

        if pps.pps_chroma_tool_offsets_present_flag {
            pps.pps_cb_qp_offset = reader.read_se()? as i8;
            pps.pps_cr_qp_offset = reader.read_se()? as i8;
            pps.pps_joint_cbcr_qp_offset_present_flag = reader.read_bit()?;
            if pps.pps_joint_cbcr_qp_offset_present_flag {
                pps.pps_joint_cbcr_qp_offset_value = reader.read_se()? as i8;
            }
            pps.pps_slice_chroma_qp_offsets_present_flag = reader.read_bit()?;
            pps.pps_cu_chroma_qp_offset_list_enabled_flag = reader.read_bit()?;
        }

        pps.pps_deblocking_filter_control_present_flag = reader.read_bit()?;
        if pps.pps_deblocking_filter_control_present_flag {
            pps.pps_deblocking_filter_override_enabled_flag = reader.read_bit()?;
            pps.pps_deblocking_filter_disabled_flag = reader.read_bit()?;

            if !pps.pps_no_pic_partition_flag && pps.pps_deblocking_filter_override_enabled_flag {
                pps.pps_dbf_info_in_ph_flag = reader.read_bit()?;
            }

            if !pps.pps_deblocking_filter_disabled_flag {
                pps.pps_luma_beta_offset_div2 = reader.read_se()? as i8;
                pps.pps_luma_tc_offset_div2 = reader.read_se()? as i8;
                pps.pps_cb_beta_offset_div2 = reader.read_se()? as i8;
                pps.pps_cb_tc_offset_div2 = reader.read_se()? as i8;
                pps.pps_cr_beta_offset_div2 = reader.read_se()? as i8;
                pps.pps_cr_tc_offset_div2 = reader.read_se()? as i8;
            }
        }

        if !pps.pps_no_pic_partition_flag {
            pps.pps_rpl_info_in_ph_flag = reader.read_bit()?;
            pps.pps_sao_info_in_ph_flag = reader.read_bit()?;
            pps.pps_alf_info_in_ph_flag = reader.read_bit()?;

            if (pps.pps_weighted_pred_flag || pps.pps_weighted_bipred_flag) && pps.pps_rpl_info_in_ph_flag {
                pps.pps_wp_info_in_ph_flag = reader.read_bit()?;
            }

            pps.pps_qp_delta_info_in_ph_flag = reader.read_bit()?;
        }

        pps.pps_picture_header_extension_present_flag = reader.read_bit()?;
        pps.pps_slice_header_extension_present_flag = reader.read_bit()?;
        pps.pps_extension_flag = reader.read_bit()?;

        Ok(pps)
    }

    /// Get initial QP.
    pub fn init_qp(&self) -> i8 {
        26 + self.pps_init_qp_minus26
    }

    /// Get picture width.
    pub fn width(&self) -> u32 {
        self.pps_pic_width_in_luma_samples
    }

    /// Get picture height.
    pub fn height(&self) -> u32 {
        self.pps_pic_height_in_luma_samples
    }
}

/// Picture Header (PH).
#[derive(Debug, Clone, Default)]
pub struct PictureHeader {
    /// GDR or IRAP picture flag.
    pub ph_gdr_or_irap_pic_flag: bool,
    /// Non-reference picture flag.
    pub ph_non_ref_pic_flag: bool,
    /// GDR picture flag.
    pub ph_gdr_pic_flag: bool,
    /// Inter slice allowed flag.
    pub ph_inter_slice_allowed_flag: bool,
    /// Intra slice allowed flag.
    pub ph_intra_slice_allowed_flag: bool,
    /// PPS ID.
    pub ph_pic_parameter_set_id: u8,
    /// Picture order count LSB.
    pub ph_pic_order_cnt_lsb: u32,
    /// Recovery POC count.
    pub ph_recovery_poc_cnt: u32,
    /// Extra bit.
    pub ph_extra_bit: Vec<bool>,
    /// POC MSB cycle present flag.
    pub ph_poc_msb_cycle_present_flag: bool,
    /// POC MSB cycle val.
    pub ph_poc_msb_cycle_val: u32,
    /// ALF enabled flag.
    pub ph_alf_enabled_flag: bool,
    /// Num ALF APS IDs luma.
    pub ph_num_alf_aps_ids_luma: u8,
    /// ALF APS ID luma.
    pub ph_alf_aps_id_luma: Vec<u8>,
    /// ALF CB enabled flag.
    pub ph_alf_cb_enabled_flag: bool,
    /// ALF CR enabled flag.
    pub ph_alf_cr_enabled_flag: bool,
    /// ALF APS ID chroma.
    pub ph_alf_aps_id_chroma: u8,
    /// ALF CC CB enabled flag.
    pub ph_alf_cc_cb_enabled_flag: bool,
    /// ALF CC CB APS ID.
    pub ph_alf_cc_cb_aps_id: u8,
    /// ALF CC CR enabled flag.
    pub ph_alf_cc_cr_enabled_flag: bool,
    /// ALF CC CR APS ID.
    pub ph_alf_cc_cr_aps_id: u8,
    /// LMCS enabled flag.
    pub ph_lmcs_enabled_flag: bool,
    /// LMCS APS ID.
    pub ph_lmcs_aps_id: u8,
    /// Chroma residual scale flag.
    pub ph_chroma_residual_scale_flag: bool,
    /// Explicit scaling list enabled flag.
    pub ph_explicit_scaling_list_enabled_flag: bool,
    /// Scaling list APS ID.
    pub ph_scaling_list_aps_id: u8,
    /// Virtual boundaries present flag.
    pub ph_virtual_boundaries_present_flag: bool,
    /// Pic output flag.
    pub ph_pic_output_flag: bool,
    /// Collocated from L0 flag.
    pub ph_collocated_from_l0_flag: bool,
    /// Collocated ref IDX.
    pub ph_collocated_ref_idx: u8,
    /// MMVD fullpel only flag.
    pub ph_mmvd_fullpel_only_flag: bool,
    /// MVD L1 zero flag.
    pub ph_mvd_l1_zero_flag: bool,
    /// BDOF disabled flag.
    pub ph_bdof_disabled_flag: bool,
    /// DMVR disabled flag.
    pub ph_dmvr_disabled_flag: bool,
    /// PROF disabled flag.
    pub ph_prof_disabled_flag: bool,
    /// QP delta.
    pub ph_qp_delta: i8,
    /// Joint CbCr sign flag.
    pub ph_joint_cbcr_sign_flag: bool,
    /// SAO luma enabled flag.
    pub ph_sao_luma_enabled_flag: bool,
    /// SAO chroma enabled flag.
    pub ph_sao_chroma_enabled_flag: bool,
    /// Deblocking params present flag.
    pub ph_deblocking_params_present_flag: bool,
    /// Deblocking filter disabled flag.
    pub ph_deblocking_filter_disabled_flag: bool,
    /// Luma beta offset div2.
    pub ph_luma_beta_offset_div2: i8,
    /// Luma TC offset div2.
    pub ph_luma_tc_offset_div2: i8,
    /// CB beta offset div2.
    pub ph_cb_beta_offset_div2: i8,
    /// CB TC offset div2.
    pub ph_cb_tc_offset_div2: i8,
    /// CR beta offset div2.
    pub ph_cr_beta_offset_div2: i8,
    /// CR TC offset div2.
    pub ph_cr_tc_offset_div2: i8,
    /// Extension length.
    pub ph_extension_length: u16,
    /// Extension data byte.
    pub ph_extension_data_byte: Vec<u8>,
}

impl PictureHeader {
    /// Parse picture header (simplified).
    pub fn parse(data: &[u8], sps: &Sps, pps: &Pps) -> Result<Self> {
        let mut reader = BitReader::new(data);
        let mut ph = Self::default();

        ph.ph_gdr_or_irap_pic_flag = reader.read_bit()?;
        ph.ph_non_ref_pic_flag = reader.read_bit()?;

        if ph.ph_gdr_or_irap_pic_flag {
            ph.ph_gdr_pic_flag = reader.read_bit()?;
        }

        ph.ph_inter_slice_allowed_flag = reader.read_bit()?;
        if ph.ph_inter_slice_allowed_flag {
            ph.ph_intra_slice_allowed_flag = reader.read_bit()?;
        } else {
            ph.ph_intra_slice_allowed_flag = true;
        }

        ph.ph_pic_parameter_set_id = reader.read_ue()? as u8;
        ph.ph_pic_order_cnt_lsb = reader.read_bits((sps.sps_log2_max_pic_order_cnt_lsb_minus4 + 4) as u8)?;

        if ph.ph_gdr_pic_flag {
            ph.ph_recovery_poc_cnt = reader.read_ue()?;
        }

        // Extra PH bits
        for _ in 0..sps.sps_num_extra_ph_bytes * 8 {
            ph.ph_extra_bit.push(reader.read_bit()?);
        }

        // POC MSB
        if sps.sps_poc_msb_cycle_flag {
            ph.ph_poc_msb_cycle_present_flag = reader.read_bit()?;
            if ph.ph_poc_msb_cycle_present_flag {
                ph.ph_poc_msb_cycle_val = reader.read_bits((sps.sps_poc_msb_cycle_len_minus1 + 1) as u8)?;
            }
        }

        // ALF
        if sps.sps_alf_enabled_flag && pps.pps_alf_info_in_ph_flag {
            ph.ph_alf_enabled_flag = reader.read_bit()?;
            if ph.ph_alf_enabled_flag {
                ph.ph_num_alf_aps_ids_luma = reader.read_bits(3)? as u8;
                for _ in 0..ph.ph_num_alf_aps_ids_luma {
                    ph.ph_alf_aps_id_luma.push(reader.read_bits(3)? as u8);
                }
                if sps.sps_chroma_format_idc != 0 {
                    ph.ph_alf_cb_enabled_flag = reader.read_bit()?;
                    ph.ph_alf_cr_enabled_flag = reader.read_bit()?;
                }
                if ph.ph_alf_cb_enabled_flag || ph.ph_alf_cr_enabled_flag {
                    ph.ph_alf_aps_id_chroma = reader.read_bits(3)? as u8;
                }
                if sps.sps_ccalf_enabled_flag {
                    ph.ph_alf_cc_cb_enabled_flag = reader.read_bit()?;
                    if ph.ph_alf_cc_cb_enabled_flag {
                        ph.ph_alf_cc_cb_aps_id = reader.read_bits(3)? as u8;
                    }
                    ph.ph_alf_cc_cr_enabled_flag = reader.read_bit()?;
                    if ph.ph_alf_cc_cr_enabled_flag {
                        ph.ph_alf_cc_cr_aps_id = reader.read_bits(3)? as u8;
                    }
                }
            }
        }

        // LMCS
        if sps.sps_lmcs_enabled_flag {
            ph.ph_lmcs_enabled_flag = reader.read_bit()?;
            if ph.ph_lmcs_enabled_flag {
                ph.ph_lmcs_aps_id = reader.read_bits(2)? as u8;
                if sps.sps_chroma_format_idc != 0 {
                    ph.ph_chroma_residual_scale_flag = reader.read_bit()?;
                }
            }
        }

        // Scaling list
        if sps.sps_explicit_scaling_list_enabled_flag {
            ph.ph_explicit_scaling_list_enabled_flag = reader.read_bit()?;
            if ph.ph_explicit_scaling_list_enabled_flag {
                ph.ph_scaling_list_aps_id = reader.read_bits(3)? as u8;
            }
        }

        // Virtual boundaries
        if sps.sps_virtual_boundaries_enabled_flag && !sps.sps_virtual_boundaries_present_flag {
            ph.ph_virtual_boundaries_present_flag = reader.read_bit()?;
        }

        // Output flag
        if pps.pps_output_flag_present_flag {
            ph.ph_pic_output_flag = reader.read_bit()?;
        } else {
            ph.ph_pic_output_flag = true;
        }

        // SAO
        if pps.pps_sao_info_in_ph_flag {
            if sps.sps_sao_enabled_flag {
                ph.ph_sao_luma_enabled_flag = reader.read_bit()?;
                if sps.sps_chroma_format_idc != 0 {
                    ph.ph_sao_chroma_enabled_flag = reader.read_bit()?;
                }
            }
        }

        // Deblocking
        if pps.pps_dbf_info_in_ph_flag {
            ph.ph_deblocking_params_present_flag = reader.read_bit()?;
            if ph.ph_deblocking_params_present_flag {
                if !pps.pps_deblocking_filter_disabled_flag {
                    ph.ph_deblocking_filter_disabled_flag = reader.read_bit()?;
                }
                if !ph.ph_deblocking_filter_disabled_flag {
                    ph.ph_luma_beta_offset_div2 = reader.read_se()? as i8;
                    ph.ph_luma_tc_offset_div2 = reader.read_se()? as i8;
                    if sps.sps_chroma_format_idc != 0 {
                        ph.ph_cb_beta_offset_div2 = reader.read_se()? as i8;
                        ph.ph_cb_tc_offset_div2 = reader.read_se()? as i8;
                        ph.ph_cr_beta_offset_div2 = reader.read_se()? as i8;
                        ph.ph_cr_tc_offset_div2 = reader.read_se()? as i8;
                    }
                }
            }
        }

        // QP delta
        if pps.pps_qp_delta_info_in_ph_flag {
            ph.ph_qp_delta = reader.read_se()? as i8;
        }

        Ok(ph)
    }

    /// Get the POC (Picture Order Count).
    pub fn poc(&self) -> u32 {
        self.ph_pic_order_cnt_lsb
    }
}

/// Slice header structure.
#[derive(Debug, Clone, Default)]
pub struct SliceHeader {
    /// Picture header in slice header flag.
    pub sh_picture_header_in_slice_header_flag: bool,
    /// Embedded picture header.
    pub picture_header: Option<PictureHeader>,
    /// Subpic ID.
    pub sh_subpic_id: u16,
    /// Slice address.
    pub sh_slice_address: u32,
    /// Extra SH bit.
    pub sh_extra_bit: Vec<bool>,
    /// Number of tiles in slice minus 1.
    pub sh_num_tiles_in_slice_minus1: u16,
    /// Slice type.
    pub sh_slice_type: SliceType,
    /// No output of prior pics flag.
    pub sh_no_output_of_prior_pics_flag: bool,
    /// ALF enabled flag.
    pub sh_alf_enabled_flag: bool,
    /// Num ALF APS IDs luma.
    pub sh_num_alf_aps_ids_luma: u8,
    /// ALF APS ID luma.
    pub sh_alf_aps_id_luma: Vec<u8>,
    /// ALF CB enabled flag.
    pub sh_alf_cb_enabled_flag: bool,
    /// ALF CR enabled flag.
    pub sh_alf_cr_enabled_flag: bool,
    /// ALF APS ID chroma.
    pub sh_alf_aps_id_chroma: u8,
    /// ALF CC CB enabled flag.
    pub sh_alf_cc_cb_enabled_flag: bool,
    /// ALF CC CB APS ID.
    pub sh_alf_cc_cb_aps_id: u8,
    /// ALF CC CR enabled flag.
    pub sh_alf_cc_cr_enabled_flag: bool,
    /// ALF CC CR APS ID.
    pub sh_alf_cc_cr_aps_id: u8,
    /// LMCS used flag.
    pub sh_lmcs_used_flag: bool,
    /// Explicit scaling list used flag.
    pub sh_explicit_scaling_list_used_flag: bool,
    /// Num ref IDX active override flag.
    pub sh_num_ref_idx_active_override_flag: bool,
    /// Num ref IDX active minus 1.
    pub sh_num_ref_idx_active_minus1: [u8; 2],
    /// CABAC init flag.
    pub sh_cabac_init_flag: bool,
    /// Collocated from L0 flag.
    pub sh_collocated_from_l0_flag: bool,
    /// Collocated ref IDX.
    pub sh_collocated_ref_idx: u8,
    /// QP delta.
    pub sh_qp_delta: i8,
    /// CB QP offset.
    pub sh_cb_qp_offset: i8,
    /// CR QP offset.
    pub sh_cr_qp_offset: i8,
    /// Joint CbCr QP offset.
    pub sh_joint_cbcr_qp_offset: i8,
    /// CU chroma QP offset enabled flag.
    pub sh_cu_chroma_qp_offset_enabled_flag: bool,
    /// SAO luma used flag.
    pub sh_sao_luma_used_flag: bool,
    /// SAO chroma used flag.
    pub sh_sao_chroma_used_flag: bool,
    /// Deblocking params present flag.
    pub sh_deblocking_params_present_flag: bool,
    /// Deblocking filter disabled flag.
    pub sh_deblocking_filter_disabled_flag: bool,
    /// Luma beta offset div2.
    pub sh_luma_beta_offset_div2: i8,
    /// Luma TC offset div2.
    pub sh_luma_tc_offset_div2: i8,
    /// CB beta offset div2.
    pub sh_cb_beta_offset_div2: i8,
    /// CB TC offset div2.
    pub sh_cb_tc_offset_div2: i8,
    /// CR beta offset div2.
    pub sh_cr_beta_offset_div2: i8,
    /// CR TC offset div2.
    pub sh_cr_tc_offset_div2: i8,
    /// Dep quant used flag.
    pub sh_dep_quant_used_flag: bool,
    /// Sign data hiding used flag.
    pub sh_sign_data_hiding_used_flag: bool,
    /// TS residual coding disabled flag.
    pub sh_ts_residual_coding_disabled_flag: bool,
    /// Slice header extension length.
    pub sh_slice_header_extension_length: u16,
    /// Slice header extension data byte.
    pub sh_slice_header_extension_data_byte: Vec<u8>,
    /// Entry offset len minus 1.
    pub sh_entry_offset_len_minus1: u8,
    /// Entry point offset minus 1.
    pub sh_entry_point_offset_minus1: Vec<u32>,
}

impl SliceHeader {
    /// Parse slice header (simplified).
    pub fn parse(
        data: &[u8],
        sps: &Sps,
        pps: &Pps,
        nal_unit_type: NalUnitType,
    ) -> Result<Self> {
        let mut reader = BitReader::new(data);
        let mut sh = Self::default();

        sh.sh_picture_header_in_slice_header_flag = reader.read_bit()?;

        if sh.sh_picture_header_in_slice_header_flag {
            // Parse embedded picture header
            sh.picture_header = Some(PictureHeader::parse(data, sps, pps)?);
        }

        // Subpic ID
        if sps.sps_subpic_info_present_flag {
            let bits = (sps.sps_num_subpics_minus1.max(1).ilog2() + 1) as u8;
            sh.sh_subpic_id = reader.read_bits(bits)? as u16;
        }

        // Slice address
        if !pps.pps_rect_slice_flag && pps.pps_num_slices_in_pic_minus1 > 0 {
            let bits = ((pps.pps_num_slices_in_pic_minus1 + 1).ilog2() + 1) as u8;
            sh.sh_slice_address = reader.read_bits(bits)?;
        }

        // Extra SH bits
        for _ in 0..sps.sps_num_extra_sh_bytes * 8 {
            sh.sh_extra_bit.push(reader.read_bit()?);
        }

        // Tiles in slice
        if !pps.pps_rect_slice_flag {
            sh.sh_num_tiles_in_slice_minus1 = reader.read_ue()? as u16;
        }

        // Slice type
        let picture_header = sh.picture_header.as_ref();
        let inter_allowed = picture_header.map(|ph| ph.ph_inter_slice_allowed_flag).unwrap_or(true);

        if inter_allowed {
            sh.sh_slice_type = SliceType::from_u8(reader.read_ue()? as u8)
                .ok_or_else(|| VvcError::SliceHeader("Invalid slice type".to_string()))?;
        } else {
            sh.sh_slice_type = SliceType::I;
        }

        // IDR-specific
        if nal_unit_type.is_idr() {
            sh.sh_no_output_of_prior_pics_flag = reader.read_bit()?;
        }

        // ALF
        if sps.sps_alf_enabled_flag && !pps.pps_alf_info_in_ph_flag {
            sh.sh_alf_enabled_flag = reader.read_bit()?;
            if sh.sh_alf_enabled_flag {
                sh.sh_num_alf_aps_ids_luma = reader.read_bits(3)? as u8;
                for _ in 0..sh.sh_num_alf_aps_ids_luma {
                    sh.sh_alf_aps_id_luma.push(reader.read_bits(3)? as u8);
                }
                if sps.sps_chroma_format_idc != 0 {
                    sh.sh_alf_cb_enabled_flag = reader.read_bit()?;
                    sh.sh_alf_cr_enabled_flag = reader.read_bit()?;
                }
                if sh.sh_alf_cb_enabled_flag || sh.sh_alf_cr_enabled_flag {
                    sh.sh_alf_aps_id_chroma = reader.read_bits(3)? as u8;
                }
            }
        }

        // QP delta
        if !pps.pps_qp_delta_info_in_ph_flag {
            sh.sh_qp_delta = reader.read_se()? as i8;
        }

        // SAO
        if !pps.pps_sao_info_in_ph_flag {
            if sps.sps_sao_enabled_flag {
                sh.sh_sao_luma_used_flag = reader.read_bit()?;
                if sps.sps_chroma_format_idc != 0 {
                    sh.sh_sao_chroma_used_flag = reader.read_bit()?;
                }
            }
        }

        // Deblocking
        if pps.pps_deblocking_filter_override_enabled_flag && !pps.pps_dbf_info_in_ph_flag {
            sh.sh_deblocking_params_present_flag = reader.read_bit()?;
            if sh.sh_deblocking_params_present_flag {
                if !pps.pps_deblocking_filter_disabled_flag {
                    sh.sh_deblocking_filter_disabled_flag = reader.read_bit()?;
                }
                if !sh.sh_deblocking_filter_disabled_flag {
                    sh.sh_luma_beta_offset_div2 = reader.read_se()? as i8;
                    sh.sh_luma_tc_offset_div2 = reader.read_se()? as i8;
                    if sps.sps_chroma_format_idc != 0 {
                        sh.sh_cb_beta_offset_div2 = reader.read_se()? as i8;
                        sh.sh_cb_tc_offset_div2 = reader.read_se()? as i8;
                        sh.sh_cr_beta_offset_div2 = reader.read_se()? as i8;
                        sh.sh_cr_tc_offset_div2 = reader.read_se()? as i8;
                    }
                }
            }
        }

        // Dep quant
        if sps.sps_dep_quant_enabled_flag {
            sh.sh_dep_quant_used_flag = reader.read_bit()?;
        }

        // Sign data hiding
        if sps.sps_sign_data_hiding_enabled_flag && !sh.sh_dep_quant_used_flag {
            sh.sh_sign_data_hiding_used_flag = reader.read_bit()?;
        }

        // TS residual coding
        if sps.sps_transform_skip_enabled_flag
            && !sh.sh_dep_quant_used_flag
            && !sh.sh_sign_data_hiding_used_flag
        {
            sh.sh_ts_residual_coding_disabled_flag = reader.read_bit()?;
        }

        // Slice header extension
        if pps.pps_slice_header_extension_present_flag {
            sh.sh_slice_header_extension_length = reader.read_ue()? as u16;
            for _ in 0..sh.sh_slice_header_extension_length {
                sh.sh_slice_header_extension_data_byte.push(reader.read_bits(8)? as u8);
            }
        }

        // Entry point offsets
        if sps.sps_entry_point_offsets_present_flag {
            // Calculate number of entry points based on tiles/CTU rows
            let num_entry_points = 0u32; // Simplified - would need tile/CTU info
            if num_entry_points > 0 {
                sh.sh_entry_offset_len_minus1 = reader.read_ue()? as u8;
                for _ in 0..num_entry_points {
                    sh.sh_entry_point_offset_minus1.push(
                        reader.read_bits((sh.sh_entry_offset_len_minus1 + 1) as u8)?
                    );
                }
            }
        }

        Ok(sh)
    }

    /// Get slice QP.
    pub fn slice_qp(&self, pps: &Pps, ph: Option<&PictureHeader>) -> i8 {
        let base_qp = pps.init_qp();
        let ph_delta = ph.map(|h| h.ph_qp_delta).unwrap_or(0);
        base_qp + ph_delta + self.sh_qp_delta
    }

    /// Check if this is an intra slice.
    pub fn is_intra(&self) -> bool {
        self.sh_slice_type.is_intra()
    }
}

/// SEI (Supplemental Enhancement Information) message type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeiType {
    /// Buffering period.
    BufferingPeriod = 0,
    /// Picture timing.
    PicTiming = 1,
    /// Pan scan rect.
    PanScanRect = 2,
    /// Filler payload.
    FillerPayload = 3,
    /// User data registered ITU-T T.35.
    UserDataRegisteredItuTT35 = 4,
    /// User data unregistered.
    UserDataUnregistered = 5,
    /// Recovery point.
    RecoveryPoint = 6,
    /// Decoded picture hash.
    DecodedPictureHash = 132,
    /// Scalable nesting.
    ScalableNesting = 133,
    /// Mastering display colour volume.
    MasteringDisplayColourVolume = 137,
    /// Content light level info.
    ContentLightLevelInfo = 144,
    /// Alternative transfer characteristics.
    AlternativeTransferCharacteristics = 147,
    /// Ambient viewing environment.
    AmbientViewingEnvironment = 148,
}

impl SeiType {
    /// Create from type value.
    pub fn from_u32(val: u32) -> Option<Self> {
        match val {
            0 => Some(Self::BufferingPeriod),
            1 => Some(Self::PicTiming),
            2 => Some(Self::PanScanRect),
            3 => Some(Self::FillerPayload),
            4 => Some(Self::UserDataRegisteredItuTT35),
            5 => Some(Self::UserDataUnregistered),
            6 => Some(Self::RecoveryPoint),
            132 => Some(Self::DecodedPictureHash),
            133 => Some(Self::ScalableNesting),
            137 => Some(Self::MasteringDisplayColourVolume),
            144 => Some(Self::ContentLightLevelInfo),
            147 => Some(Self::AlternativeTransferCharacteristics),
            148 => Some(Self::AmbientViewingEnvironment),
            _ => None,
        }
    }
}

/// SEI message.
#[derive(Debug, Clone)]
pub struct SeiMessage {
    /// SEI type.
    pub sei_type: u32,
    /// Payload size.
    pub payload_size: u32,
    /// Payload data.
    pub payload: Vec<u8>,
}

impl SeiMessage {
    /// Parse SEI message.
    pub fn parse(reader: &mut BitReader) -> Result<Self> {
        // Read payload type
        let mut sei_type = 0u32;
        loop {
            let byte = reader.read_bits(8)? as u8;
            sei_type += byte as u32;
            if byte != 0xFF {
                break;
            }
        }

        // Read payload size
        let mut payload_size = 0u32;
        loop {
            let byte = reader.read_bits(8)? as u8;
            payload_size += byte as u32;
            if byte != 0xFF {
                break;
            }
        }

        // Read payload
        let mut payload = Vec::with_capacity(payload_size as usize);
        for _ in 0..payload_size {
            payload.push(reader.read_bits(8)? as u8);
        }

        Ok(Self {
            sei_type,
            payload_size,
            payload,
        })
    }

    /// Get typed SEI type.
    pub fn typed_sei_type(&self) -> Option<SeiType> {
        SeiType::from_u32(self.sei_type)
    }
}

/// Parse Annex B NAL units from bitstream data.
pub fn parse_annexb_nal_units(data: &[u8]) -> Vec<(NalUnitHeader, Vec<u8>)> {
    let mut result = Vec::new();
    let mut i = 0;

    while i < data.len() {
        // Find start code
        let start_code_len = if i + 3 < data.len() && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 0 && data[i + 3] == 1 {
            4
        } else if i + 2 < data.len() && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            3
        } else {
            i += 1;
            continue;
        };

        let nal_start = i + start_code_len;

        // Find next start code or end
        let mut nal_end = data.len();
        for j in nal_start..data.len().saturating_sub(2) {
            if data[j] == 0 && data[j + 1] == 0 && (data[j + 2] == 1 || (j + 3 < data.len() && data[j + 2] == 0 && data[j + 3] == 1)) {
                nal_end = j;
                break;
            }
        }

        if nal_start < nal_end {
            let nal_data = &data[nal_start..nal_end];

            // Remove emulation prevention bytes
            let rbsp = remove_emulation_prevention(nal_data);

            if rbsp.len() >= 2 {
                let mut reader = BitReader::new(&rbsp);
                if let Ok(header) = NalUnitHeader::parse(&mut reader) {
                    result.push((header, rbsp[2..].to_vec()));
                }
            }
        }

        i = nal_end;
    }

    result
}

/// Remove emulation prevention bytes from NAL unit data.
fn remove_emulation_prevention(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(data.len());
    let mut i = 0;

    while i < data.len() {
        if i + 2 < data.len() && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 3 {
            result.push(0);
            result.push(0);
            i += 3;
        } else {
            result.push(data[i]);
            i += 1;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nal_unit_type() {
        assert!(NalUnitType::IdrWRadl.is_idr());
        assert!(NalUnitType::IdrNLp.is_idr());
        assert!(!NalUnitType::TrailNut.is_idr());

        assert!(NalUnitType::IdrWRadl.is_irap());
        assert!(NalUnitType::CraNut.is_irap());
        assert!(NalUnitType::GdrNut.is_irap());

        assert!(NalUnitType::TrailNut.is_vcl());
        assert!(!NalUnitType::SpsNut.is_vcl());

        assert!(NalUnitType::TrailNut.is_trailing());
        assert!(NalUnitType::RadlNut.is_leading());
    }

    #[test]
    fn test_slice_type() {
        assert!(SliceType::I.is_intra());
        assert!(!SliceType::P.is_intra());
        assert!(!SliceType::B.is_intra());

        assert!(SliceType::P.is_inter());
        assert!(SliceType::B.is_inter());
        assert!(SliceType::B.is_b());
    }

    #[test]
    fn test_sei_type() {
        assert_eq!(SeiType::from_u32(0), Some(SeiType::BufferingPeriod));
        assert_eq!(SeiType::from_u32(132), Some(SeiType::DecodedPictureHash));
        assert_eq!(SeiType::from_u32(999), None);
    }

    #[test]
    fn test_remove_emulation_prevention() {
        let data = vec![0x00, 0x00, 0x03, 0x01, 0x00, 0x00, 0x03, 0x02];
        let result = remove_emulation_prevention(&data);
        assert_eq!(result, vec![0x00, 0x00, 0x01, 0x00, 0x00, 0x02]);
    }

    #[test]
    fn test_nal_header_temporal_id() {
        let header = NalUnitHeader {
            nal_unit_type: NalUnitType::TrailNut,
            nuh_layer_id: 0,
            nuh_temporal_id_plus1: 3,
        };
        assert_eq!(header.temporal_id(), 2);
    }
}
