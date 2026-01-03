//! HEVC NAL unit parsing.
//!
//! This module provides comprehensive NAL (Network Abstraction Layer) unit parsing
//! for HEVC/H.265, including VPS, SPS, PPS, and slice headers.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::manual_div_ceil)]
#![allow(unused_variables)]

use crate::error::{HevcError, HevcLevel, HevcProfile, HevcTier, NalError, Result};
use transcode_core::bitstream::{BitReader, BitWriter, remove_emulation_prevention};
use std::fmt;

/// HEVC NAL unit types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum NalUnitType {
    /// Trailing picture, non-reference.
    TrailN = 0,
    /// Trailing picture, reference.
    TrailR = 1,
    /// Temporal sub-layer access, non-reference.
    TsaN = 2,
    /// Temporal sub-layer access, reference.
    TsaR = 3,
    /// Stepwise temporal sub-layer access, non-reference.
    StsaN = 4,
    /// Stepwise temporal sub-layer access, reference.
    StsaR = 5,
    /// Random access decodable leading, non-reference.
    RadlN = 6,
    /// Random access decodable leading, reference.
    RadlR = 7,
    /// Random access skipped leading, non-reference.
    RaslN = 8,
    /// Random access skipped leading, reference.
    RaslR = 9,
    /// Reserved VCL N10.
    RsvVclN10 = 10,
    /// Reserved VCL R11.
    RsvVclR11 = 11,
    /// Reserved VCL N12.
    RsvVclN12 = 12,
    /// Reserved VCL R13.
    RsvVclR13 = 13,
    /// Reserved VCL N14.
    RsvVclN14 = 14,
    /// Reserved VCL R15.
    RsvVclR15 = 15,
    /// Broken link access, W leading picture.
    BlaWLp = 16,
    /// Broken link access, W random access decodable leading.
    BlaWRadl = 17,
    /// Broken link access, N leading picture.
    BlaNLp = 18,
    /// Instantaneous decoder refresh, W random access decodable leading.
    IdrWRadl = 19,
    /// Instantaneous decoder refresh, N leading picture.
    IdrNLp = 20,
    /// Clean random access.
    CraNut = 21,
    /// Reserved IRAP VCL 22.
    RsvIrapVcl22 = 22,
    /// Reserved IRAP VCL 23.
    RsvIrapVcl23 = 23,
    /// Reserved VCL 24.
    RsvVcl24 = 24,
    /// Reserved VCL 25.
    RsvVcl25 = 25,
    /// Reserved VCL 26.
    RsvVcl26 = 26,
    /// Reserved VCL 27.
    RsvVcl27 = 27,
    /// Reserved VCL 28.
    RsvVcl28 = 28,
    /// Reserved VCL 29.
    RsvVcl29 = 29,
    /// Reserved VCL 30.
    RsvVcl30 = 30,
    /// Reserved VCL 31.
    RsvVcl31 = 31,
    /// Video Parameter Set.
    VpsNut = 32,
    /// Sequence Parameter Set.
    SpsNut = 33,
    /// Picture Parameter Set.
    PpsNut = 34,
    /// Access Unit Delimiter.
    AudNut = 35,
    /// End of Sequence.
    EosNut = 36,
    /// End of Bitstream.
    EobNut = 37,
    /// Filler Data.
    FdNut = 38,
    /// Prefix Supplemental Enhancement Information.
    PrefixSeiNut = 39,
    /// Suffix Supplemental Enhancement Information.
    SuffixSeiNut = 40,
    /// Reserved NVCL 41.
    RsvNvcl41 = 41,
    /// Reserved NVCL 42.
    RsvNvcl42 = 42,
    /// Reserved NVCL 43.
    RsvNvcl43 = 43,
    /// Reserved NVCL 44.
    RsvNvcl44 = 44,
    /// Reserved NVCL 45.
    RsvNvcl45 = 45,
    /// Reserved NVCL 46.
    RsvNvcl46 = 46,
    /// Reserved NVCL 47.
    RsvNvcl47 = 47,
    /// Unspecified.
    Unspecified(u8),
}

impl NalUnitType {
    /// Create from raw value.
    pub fn from_raw(value: u8) -> Self {
        match value {
            0 => Self::TrailN,
            1 => Self::TrailR,
            2 => Self::TsaN,
            3 => Self::TsaR,
            4 => Self::StsaN,
            5 => Self::StsaR,
            6 => Self::RadlN,
            7 => Self::RadlR,
            8 => Self::RaslN,
            9 => Self::RaslR,
            10 => Self::RsvVclN10,
            11 => Self::RsvVclR11,
            12 => Self::RsvVclN12,
            13 => Self::RsvVclR13,
            14 => Self::RsvVclN14,
            15 => Self::RsvVclR15,
            16 => Self::BlaWLp,
            17 => Self::BlaWRadl,
            18 => Self::BlaNLp,
            19 => Self::IdrWRadl,
            20 => Self::IdrNLp,
            21 => Self::CraNut,
            22 => Self::RsvIrapVcl22,
            23 => Self::RsvIrapVcl23,
            24 => Self::RsvVcl24,
            25 => Self::RsvVcl25,
            26 => Self::RsvVcl26,
            27 => Self::RsvVcl27,
            28 => Self::RsvVcl28,
            29 => Self::RsvVcl29,
            30 => Self::RsvVcl30,
            31 => Self::RsvVcl31,
            32 => Self::VpsNut,
            33 => Self::SpsNut,
            34 => Self::PpsNut,
            35 => Self::AudNut,
            36 => Self::EosNut,
            37 => Self::EobNut,
            38 => Self::FdNut,
            39 => Self::PrefixSeiNut,
            40 => Self::SuffixSeiNut,
            41 => Self::RsvNvcl41,
            42 => Self::RsvNvcl42,
            43 => Self::RsvNvcl43,
            44 => Self::RsvNvcl44,
            45 => Self::RsvNvcl45,
            46 => Self::RsvNvcl46,
            47 => Self::RsvNvcl47,
            v => Self::Unspecified(v),
        }
    }

    /// Get the raw value.
    pub fn to_raw(&self) -> u8 {
        match self {
            Self::TrailN => 0,
            Self::TrailR => 1,
            Self::TsaN => 2,
            Self::TsaR => 3,
            Self::StsaN => 4,
            Self::StsaR => 5,
            Self::RadlN => 6,
            Self::RadlR => 7,
            Self::RaslN => 8,
            Self::RaslR => 9,
            Self::RsvVclN10 => 10,
            Self::RsvVclR11 => 11,
            Self::RsvVclN12 => 12,
            Self::RsvVclR13 => 13,
            Self::RsvVclN14 => 14,
            Self::RsvVclR15 => 15,
            Self::BlaWLp => 16,
            Self::BlaWRadl => 17,
            Self::BlaNLp => 18,
            Self::IdrWRadl => 19,
            Self::IdrNLp => 20,
            Self::CraNut => 21,
            Self::RsvIrapVcl22 => 22,
            Self::RsvIrapVcl23 => 23,
            Self::RsvVcl24 => 24,
            Self::RsvVcl25 => 25,
            Self::RsvVcl26 => 26,
            Self::RsvVcl27 => 27,
            Self::RsvVcl28 => 28,
            Self::RsvVcl29 => 29,
            Self::RsvVcl30 => 30,
            Self::RsvVcl31 => 31,
            Self::VpsNut => 32,
            Self::SpsNut => 33,
            Self::PpsNut => 34,
            Self::AudNut => 35,
            Self::EosNut => 36,
            Self::EobNut => 37,
            Self::FdNut => 38,
            Self::PrefixSeiNut => 39,
            Self::SuffixSeiNut => 40,
            Self::RsvNvcl41 => 41,
            Self::RsvNvcl42 => 42,
            Self::RsvNvcl43 => 43,
            Self::RsvNvcl44 => 44,
            Self::RsvNvcl45 => 45,
            Self::RsvNvcl46 => 46,
            Self::RsvNvcl47 => 47,
            Self::Unspecified(v) => *v,
        }
    }

    /// Check if this is a VCL (Video Coding Layer) NAL unit.
    pub fn is_vcl(&self) -> bool {
        self.to_raw() < 32
    }

    /// Check if this is an IRAP (Intra Random Access Point) picture.
    pub fn is_irap(&self) -> bool {
        matches!(
            self,
            Self::BlaWLp
                | Self::BlaWRadl
                | Self::BlaNLp
                | Self::IdrWRadl
                | Self::IdrNLp
                | Self::CraNut
        )
    }

    /// Check if this is an IDR picture.
    pub fn is_idr(&self) -> bool {
        matches!(self, Self::IdrWRadl | Self::IdrNLp)
    }

    /// Check if this is a BLA (Broken Link Access) picture.
    pub fn is_bla(&self) -> bool {
        matches!(self, Self::BlaWLp | Self::BlaWRadl | Self::BlaNLp)
    }

    /// Check if this is a CRA (Clean Random Access) picture.
    pub fn is_cra(&self) -> bool {
        matches!(self, Self::CraNut)
    }

    /// Check if this is a leading picture.
    pub fn is_leading(&self) -> bool {
        matches!(
            self,
            Self::RadlN | Self::RadlR | Self::RaslN | Self::RaslR
        )
    }

    /// Check if this is a trailing picture.
    pub fn is_trailing(&self) -> bool {
        matches!(self, Self::TrailN | Self::TrailR)
    }

    /// Check if this is a reference picture.
    pub fn is_reference(&self) -> bool {
        let raw = self.to_raw();
        // Odd-numbered VCL types are reference pictures
        raw < 32 && (raw % 2 == 1)
    }
}

impl fmt::Display for NalUnitType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TrailN => write!(f, "TRAIL_N"),
            Self::TrailR => write!(f, "TRAIL_R"),
            Self::TsaN => write!(f, "TSA_N"),
            Self::TsaR => write!(f, "TSA_R"),
            Self::StsaN => write!(f, "STSA_N"),
            Self::StsaR => write!(f, "STSA_R"),
            Self::RadlN => write!(f, "RADL_N"),
            Self::RadlR => write!(f, "RADL_R"),
            Self::RaslN => write!(f, "RASL_N"),
            Self::RaslR => write!(f, "RASL_R"),
            Self::BlaWLp => write!(f, "BLA_W_LP"),
            Self::BlaWRadl => write!(f, "BLA_W_RADL"),
            Self::BlaNLp => write!(f, "BLA_N_LP"),
            Self::IdrWRadl => write!(f, "IDR_W_RADL"),
            Self::IdrNLp => write!(f, "IDR_N_LP"),
            Self::CraNut => write!(f, "CRA_NUT"),
            Self::VpsNut => write!(f, "VPS_NUT"),
            Self::SpsNut => write!(f, "SPS_NUT"),
            Self::PpsNut => write!(f, "PPS_NUT"),
            Self::AudNut => write!(f, "AUD_NUT"),
            Self::EosNut => write!(f, "EOS_NUT"),
            Self::EobNut => write!(f, "EOB_NUT"),
            Self::FdNut => write!(f, "FD_NUT"),
            Self::PrefixSeiNut => write!(f, "PREFIX_SEI_NUT"),
            Self::SuffixSeiNut => write!(f, "SUFFIX_SEI_NUT"),
            Self::Unspecified(v) => write!(f, "UNSPECIFIED({})", v),
            _ => write!(f, "RESERVED({})", self.to_raw()),
        }
    }
}

/// HEVC NAL unit header.
#[derive(Debug, Clone, Copy)]
pub struct NalUnitHeader {
    /// NAL unit type.
    pub nal_unit_type: NalUnitType,
    /// Layer ID (for scalable/multiview extensions).
    pub nuh_layer_id: u8,
    /// Temporal ID plus 1.
    pub nuh_temporal_id_plus1: u8,
}

impl NalUnitHeader {
    /// Parse NAL unit header from data.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 2 {
            return Err(NalError::Truncated {
                expected: 2,
                got: data.len(),
            }
            .into());
        }

        // forbidden_zero_bit (1 bit) must be 0
        if (data[0] & 0x80) != 0 {
            return Err(NalError::InvalidHeader.into());
        }

        // nal_unit_type (6 bits)
        let nal_unit_type = NalUnitType::from_raw((data[0] >> 1) & 0x3F);

        // nuh_layer_id (6 bits)
        let nuh_layer_id = ((data[0] & 0x01) << 5) | ((data[1] >> 3) & 0x1F);

        // nuh_temporal_id_plus1 (3 bits)
        let nuh_temporal_id_plus1 = data[1] & 0x07;

        if nuh_temporal_id_plus1 == 0 {
            return Err(NalError::InvalidHeader.into());
        }

        Ok(Self {
            nal_unit_type,
            nuh_layer_id,
            nuh_temporal_id_plus1,
        })
    }

    /// Write NAL unit header to a writer.
    pub fn write(&self, writer: &mut BitWriter) -> Result<()> {
        // forbidden_zero_bit (1 bit)
        writer.write_bit(false)?;
        // nal_unit_type (6 bits)
        writer.write_bits(self.nal_unit_type.to_raw() as u32, 6)?;
        // nuh_layer_id (6 bits)
        writer.write_bits(self.nuh_layer_id as u32, 6)?;
        // nuh_temporal_id_plus1 (3 bits)
        writer.write_bits(self.nuh_temporal_id_plus1 as u32, 3)?;
        Ok(())
    }

    /// Get the temporal ID (0-based).
    pub fn temporal_id(&self) -> u8 {
        self.nuh_temporal_id_plus1 - 1
    }
}

/// Profile, tier, and level information.
#[derive(Debug, Clone)]
pub struct ProfileTierLevel {
    /// General profile space.
    pub general_profile_space: u8,
    /// General tier flag.
    pub general_tier_flag: bool,
    /// General profile IDC.
    pub general_profile_idc: u8,
    /// General profile compatibility flags.
    pub general_profile_compatibility_flags: [bool; 32],
    /// General progressive source flag.
    pub general_progressive_source_flag: bool,
    /// General interlaced source flag.
    pub general_interlaced_source_flag: bool,
    /// General non-packed constraint flag.
    pub general_non_packed_constraint_flag: bool,
    /// General frame only constraint flag.
    pub general_frame_only_constraint_flag: bool,
    /// General level IDC.
    pub general_level_idc: u8,
    /// Sub-layer profile present flags.
    pub sub_layer_profile_present_flag: Vec<bool>,
    /// Sub-layer level present flags.
    pub sub_layer_level_present_flag: Vec<bool>,
}

impl ProfileTierLevel {
    /// Parse profile_tier_level syntax.
    pub fn parse(
        reader: &mut BitReader,
        profile_present_flag: bool,
        max_num_sub_layers_minus1: u8,
    ) -> Result<Self> {
        let mut general_profile_space = 0;
        let mut general_tier_flag = false;
        let mut general_profile_idc = 0;
        let mut general_profile_compatibility_flags = [false; 32];
        let mut general_progressive_source_flag = false;
        let mut general_interlaced_source_flag = false;
        let mut general_non_packed_constraint_flag = false;
        let mut general_frame_only_constraint_flag = false;

        if profile_present_flag {
            general_profile_space = reader.read_bits(2)? as u8;
            general_tier_flag = reader.read_bit()?;
            general_profile_idc = reader.read_bits(5)? as u8;

            for i in 0..32 {
                general_profile_compatibility_flags[i] = reader.read_bit()?;
            }

            general_progressive_source_flag = reader.read_bit()?;
            general_interlaced_source_flag = reader.read_bit()?;
            general_non_packed_constraint_flag = reader.read_bit()?;
            general_frame_only_constraint_flag = reader.read_bit()?;

            // Skip reserved bits (44 bits)
            reader.skip(44)?;
        }

        let general_level_idc = reader.read_bits(8)? as u8;

        let mut sub_layer_profile_present_flag = Vec::with_capacity(max_num_sub_layers_minus1 as usize);
        let mut sub_layer_level_present_flag = Vec::with_capacity(max_num_sub_layers_minus1 as usize);

        for _ in 0..max_num_sub_layers_minus1 {
            sub_layer_profile_present_flag.push(reader.read_bit()?);
            sub_layer_level_present_flag.push(reader.read_bit()?);
        }

        if max_num_sub_layers_minus1 > 0 {
            for _ in max_num_sub_layers_minus1..8 {
                reader.skip(2)?; // reserved_zero_2bits
            }
        }

        // Skip sub-layer profile/level parsing for now
        for i in 0..max_num_sub_layers_minus1 as usize {
            if sub_layer_profile_present_flag[i] {
                reader.skip(88)?; // sub_layer profile data
            }
            if sub_layer_level_present_flag[i] {
                reader.skip(8)?; // sub_layer_level_idc
            }
        }

        Ok(Self {
            general_profile_space,
            general_tier_flag,
            general_profile_idc,
            general_profile_compatibility_flags,
            general_progressive_source_flag,
            general_interlaced_source_flag,
            general_non_packed_constraint_flag,
            general_frame_only_constraint_flag,
            general_level_idc,
            sub_layer_profile_present_flag,
            sub_layer_level_present_flag,
        })
    }

    /// Get the profile.
    pub fn profile(&self) -> Option<HevcProfile> {
        HevcProfile::from_idc(self.general_profile_idc)
    }

    /// Get the tier.
    pub fn tier(&self) -> HevcTier {
        if self.general_tier_flag {
            HevcTier::High
        } else {
            HevcTier::Main
        }
    }

    /// Get the level.
    pub fn level(&self) -> HevcLevel {
        HevcLevel::from_idc(self.general_level_idc)
    }
}

/// Video Parameter Set (VPS).
#[derive(Debug, Clone)]
pub struct Vps {
    /// VPS ID (0-15).
    pub vps_video_parameter_set_id: u8,
    /// Base layer internal flag.
    pub vps_base_layer_internal_flag: bool,
    /// Base layer available flag.
    pub vps_base_layer_available_flag: bool,
    /// Maximum layers minus 1.
    pub vps_max_layers_minus1: u8,
    /// Maximum sub-layers minus 1.
    pub vps_max_sub_layers_minus1: u8,
    /// Temporal ID nesting flag.
    pub vps_temporal_id_nesting_flag: bool,
    /// Profile, tier, level info.
    pub profile_tier_level: ProfileTierLevel,
    /// Sub-layer ordering info present flag.
    pub vps_sub_layer_ordering_info_present_flag: bool,
    /// Maximum decoded picture buffer size.
    pub vps_max_dec_pic_buffering_minus1: Vec<u32>,
    /// Maximum number of reorder pictures.
    pub vps_max_num_reorder_pics: Vec<u32>,
    /// Maximum latency increase.
    pub vps_max_latency_increase_plus1: Vec<u32>,
    /// Maximum layer ID.
    pub vps_max_layer_id: u8,
    /// Number of layer sets minus 1.
    pub vps_num_layer_sets_minus1: u32,
    /// Timing info present flag.
    pub vps_timing_info_present_flag: bool,
    /// Number of units in tick.
    pub vps_num_units_in_tick: u32,
    /// Time scale.
    pub vps_time_scale: u32,
}

impl Vps {
    /// Parse VPS from RBSP data.
    pub fn parse(rbsp: &[u8]) -> Result<Self> {
        let mut reader = BitReader::new(rbsp);

        let vps_video_parameter_set_id = reader.read_bits(4)? as u8;
        let vps_base_layer_internal_flag = reader.read_bit()?;
        let vps_base_layer_available_flag = reader.read_bit()?;
        let vps_max_layers_minus1 = reader.read_bits(6)? as u8;
        let vps_max_sub_layers_minus1 = reader.read_bits(3)? as u8;
        let vps_temporal_id_nesting_flag = reader.read_bit()?;

        // reserved_0xffff_16bits
        reader.skip(16)?;

        let profile_tier_level =
            ProfileTierLevel::parse(&mut reader, true, vps_max_sub_layers_minus1)?;

        let vps_sub_layer_ordering_info_present_flag = reader.read_bit()?;

        let start = if vps_sub_layer_ordering_info_present_flag {
            0
        } else {
            vps_max_sub_layers_minus1
        };

        let mut vps_max_dec_pic_buffering_minus1 = vec![0; (vps_max_sub_layers_minus1 + 1) as usize];
        let mut vps_max_num_reorder_pics = vec![0; (vps_max_sub_layers_minus1 + 1) as usize];
        let mut vps_max_latency_increase_plus1 = vec![0; (vps_max_sub_layers_minus1 + 1) as usize];

        for i in start..=vps_max_sub_layers_minus1 {
            vps_max_dec_pic_buffering_minus1[i as usize] = reader.read_ue()?;
            vps_max_num_reorder_pics[i as usize] = reader.read_ue()?;
            vps_max_latency_increase_plus1[i as usize] = reader.read_ue()?;
        }

        let vps_max_layer_id = reader.read_bits(6)? as u8;
        let vps_num_layer_sets_minus1 = reader.read_ue()?;

        // Skip layer_id_included_flag
        for _ in 1..=vps_num_layer_sets_minus1 {
            for _ in 0..=vps_max_layer_id {
                reader.skip(1)?;
            }
        }

        let vps_timing_info_present_flag = reader.read_bit()?;

        let mut vps_num_units_in_tick = 0;
        let mut vps_time_scale = 0;

        if vps_timing_info_present_flag {
            vps_num_units_in_tick = reader.read_bits(32)?;
            vps_time_scale = reader.read_bits(32)?;
        }

        Ok(Self {
            vps_video_parameter_set_id,
            vps_base_layer_internal_flag,
            vps_base_layer_available_flag,
            vps_max_layers_minus1,
            vps_max_sub_layers_minus1,
            vps_temporal_id_nesting_flag,
            profile_tier_level,
            vps_sub_layer_ordering_info_present_flag,
            vps_max_dec_pic_buffering_minus1,
            vps_max_num_reorder_pics,
            vps_max_latency_increase_plus1,
            vps_max_layer_id,
            vps_num_layer_sets_minus1,
            vps_timing_info_present_flag,
            vps_num_units_in_tick,
            vps_time_scale,
        })
    }

    /// Get the frame rate if timing info is present.
    pub fn frame_rate(&self) -> Option<f64> {
        if self.vps_timing_info_present_flag && self.vps_num_units_in_tick > 0 {
            Some(self.vps_time_scale as f64 / self.vps_num_units_in_tick as f64)
        } else {
            None
        }
    }
}

/// Short-term reference picture set.
#[derive(Debug, Clone, Default)]
pub struct ShortTermRefPicSet {
    /// Number of negative pictures.
    pub num_negative_pics: u32,
    /// Number of positive pictures.
    pub num_positive_pics: u32,
    /// Delta POC S0.
    pub delta_poc_s0: Vec<i32>,
    /// Used by current picture S0.
    pub used_by_curr_pic_s0: Vec<bool>,
    /// Delta POC S1.
    pub delta_poc_s1: Vec<i32>,
    /// Used by current picture S1.
    pub used_by_curr_pic_s1: Vec<bool>,
}

impl ShortTermRefPicSet {
    /// Parse short-term reference picture set.
    pub fn parse(
        reader: &mut BitReader,
        st_rps_idx: usize,
        num_short_term_ref_pic_sets: usize,
        st_ref_pic_sets: &[ShortTermRefPicSet],
    ) -> Result<Self> {
        let inter_ref_pic_set_prediction_flag = if st_rps_idx != 0 {
            reader.read_bit()?
        } else {
            false
        };

        if inter_ref_pic_set_prediction_flag {
            // Prediction from another set
            let delta_idx_minus1 = if st_rps_idx == num_short_term_ref_pic_sets {
                reader.read_ue()?
            } else {
                0
            };
            let _delta_rps_sign = reader.read_bit()?;
            let _abs_delta_rps_minus1 = reader.read_ue()?;

            let ref_rps_idx = st_rps_idx - (delta_idx_minus1 as usize) - 1;
            let ref_rps = &st_ref_pic_sets[ref_rps_idx];
            let num_delta_pocs = ref_rps.num_negative_pics + ref_rps.num_positive_pics;

            for _ in 0..=num_delta_pocs {
                let used_by_curr_pic_flag = reader.read_bit()?;
                if !used_by_curr_pic_flag {
                    reader.read_bit()?; // use_delta_flag
                }
            }

            // For now, return empty set (full prediction would require more complex logic)
            Ok(Self::default())
        } else {
            let num_negative_pics = reader.read_ue()?;
            let num_positive_pics = reader.read_ue()?;

            let mut delta_poc_s0 = Vec::with_capacity(num_negative_pics as usize);
            let mut used_by_curr_pic_s0 = Vec::with_capacity(num_negative_pics as usize);

            for _ in 0..num_negative_pics {
                let delta_poc_s0_minus1 = reader.read_ue()?;
                let used = reader.read_bit()?;
                delta_poc_s0.push(-(delta_poc_s0_minus1 as i32 + 1));
                used_by_curr_pic_s0.push(used);
            }

            let mut delta_poc_s1 = Vec::with_capacity(num_positive_pics as usize);
            let mut used_by_curr_pic_s1 = Vec::with_capacity(num_positive_pics as usize);

            for _ in 0..num_positive_pics {
                let delta_poc_s1_minus1 = reader.read_ue()?;
                let used = reader.read_bit()?;
                delta_poc_s1.push(delta_poc_s1_minus1 as i32 + 1);
                used_by_curr_pic_s1.push(used);
            }

            Ok(Self {
                num_negative_pics,
                num_positive_pics,
                delta_poc_s0,
                used_by_curr_pic_s0,
                delta_poc_s1,
                used_by_curr_pic_s1,
            })
        }
    }

    /// Get total number of pictures in this set.
    pub fn num_pics(&self) -> u32 {
        self.num_negative_pics + self.num_positive_pics
    }
}

/// Sequence Parameter Set (SPS).
#[derive(Debug, Clone)]
pub struct Sps {
    /// VPS ID.
    pub sps_video_parameter_set_id: u8,
    /// Maximum sub-layers minus 1.
    pub sps_max_sub_layers_minus1: u8,
    /// Temporal ID nesting flag.
    pub sps_temporal_id_nesting_flag: bool,
    /// Profile, tier, level info.
    pub profile_tier_level: ProfileTierLevel,
    /// SPS ID.
    pub sps_seq_parameter_set_id: u8,
    /// Chroma format IDC.
    pub chroma_format_idc: u8,
    /// Separate color plane flag.
    pub separate_colour_plane_flag: bool,
    /// Picture width in luma samples.
    pub pic_width_in_luma_samples: u32,
    /// Picture height in luma samples.
    pub pic_height_in_luma_samples: u32,
    /// Conformance window flag.
    pub conformance_window_flag: bool,
    /// Conformance window left offset.
    pub conf_win_left_offset: u32,
    /// Conformance window right offset.
    pub conf_win_right_offset: u32,
    /// Conformance window top offset.
    pub conf_win_top_offset: u32,
    /// Conformance window bottom offset.
    pub conf_win_bottom_offset: u32,
    /// Bit depth luma minus 8.
    pub bit_depth_luma_minus8: u8,
    /// Bit depth chroma minus 8.
    pub bit_depth_chroma_minus8: u8,
    /// Log2 max POC LSB minus 4.
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    /// Sub-layer ordering info present flag.
    pub sps_sub_layer_ordering_info_present_flag: bool,
    /// Log2 minimum luma coding block size minus 3.
    pub log2_min_luma_coding_block_size_minus3: u8,
    /// Log2 difference between max and min luma coding block size.
    pub log2_diff_max_min_luma_coding_block_size: u8,
    /// Log2 minimum luma transform block size minus 2.
    pub log2_min_luma_transform_block_size_minus2: u8,
    /// Log2 difference between max and min luma transform block size.
    pub log2_diff_max_min_luma_transform_block_size: u8,
    /// Maximum transform hierarchy depth for inter prediction.
    pub max_transform_hierarchy_depth_inter: u8,
    /// Maximum transform hierarchy depth for intra prediction.
    pub max_transform_hierarchy_depth_intra: u8,
    /// Scaling list enabled flag.
    pub scaling_list_enabled_flag: bool,
    /// AMP enabled flag.
    pub amp_enabled_flag: bool,
    /// SAO enabled flag.
    pub sample_adaptive_offset_enabled_flag: bool,
    /// PCM enabled flag.
    pub pcm_enabled_flag: bool,
    /// Number of short-term reference picture sets.
    pub num_short_term_ref_pic_sets: u8,
    /// Short-term reference picture sets.
    pub st_ref_pic_sets: Vec<ShortTermRefPicSet>,
    /// Long-term reference pictures present flag.
    pub long_term_ref_pics_present_flag: bool,
    /// Temporal MVP enabled flag.
    pub sps_temporal_mvp_enabled_flag: bool,
    /// Strong intra smoothing enabled flag.
    pub strong_intra_smoothing_enabled_flag: bool,
    /// VUI parameters present flag.
    pub vui_parameters_present_flag: bool,
}

impl Sps {
    /// Parse SPS from RBSP data.
    pub fn parse(rbsp: &[u8]) -> Result<Self> {
        let mut reader = BitReader::new(rbsp);

        let sps_video_parameter_set_id = reader.read_bits(4)? as u8;
        let sps_max_sub_layers_minus1 = reader.read_bits(3)? as u8;
        let sps_temporal_id_nesting_flag = reader.read_bit()?;

        let profile_tier_level =
            ProfileTierLevel::parse(&mut reader, true, sps_max_sub_layers_minus1)?;

        let sps_seq_parameter_set_id = reader.read_ue()? as u8;

        let chroma_format_idc = reader.read_ue()? as u8;
        let separate_colour_plane_flag = if chroma_format_idc == 3 {
            reader.read_bit()?
        } else {
            false
        };

        let pic_width_in_luma_samples = reader.read_ue()?;
        let pic_height_in_luma_samples = reader.read_ue()?;

        let conformance_window_flag = reader.read_bit()?;
        let mut conf_win_left_offset = 0;
        let mut conf_win_right_offset = 0;
        let mut conf_win_top_offset = 0;
        let mut conf_win_bottom_offset = 0;

        if conformance_window_flag {
            conf_win_left_offset = reader.read_ue()?;
            conf_win_right_offset = reader.read_ue()?;
            conf_win_top_offset = reader.read_ue()?;
            conf_win_bottom_offset = reader.read_ue()?;
        }

        let bit_depth_luma_minus8 = reader.read_ue()? as u8;
        let bit_depth_chroma_minus8 = reader.read_ue()? as u8;
        let log2_max_pic_order_cnt_lsb_minus4 = reader.read_ue()? as u8;

        let sps_sub_layer_ordering_info_present_flag = reader.read_bit()?;

        let start = if sps_sub_layer_ordering_info_present_flag {
            0
        } else {
            sps_max_sub_layers_minus1
        };

        for _ in start..=sps_max_sub_layers_minus1 {
            reader.read_ue()?; // sps_max_dec_pic_buffering_minus1
            reader.read_ue()?; // sps_max_num_reorder_pics
            reader.read_ue()?; // sps_max_latency_increase_plus1
        }

        let log2_min_luma_coding_block_size_minus3 = reader.read_ue()? as u8;
        let log2_diff_max_min_luma_coding_block_size = reader.read_ue()? as u8;
        let log2_min_luma_transform_block_size_minus2 = reader.read_ue()? as u8;
        let log2_diff_max_min_luma_transform_block_size = reader.read_ue()? as u8;
        let max_transform_hierarchy_depth_inter = reader.read_ue()? as u8;
        let max_transform_hierarchy_depth_intra = reader.read_ue()? as u8;

        let scaling_list_enabled_flag = reader.read_bit()?;
        if scaling_list_enabled_flag {
            let sps_scaling_list_data_present_flag = reader.read_bit()?;
            if sps_scaling_list_data_present_flag {
                // Skip scaling list data for now
                Self::skip_scaling_list_data(&mut reader)?;
            }
        }

        let amp_enabled_flag = reader.read_bit()?;
        let sample_adaptive_offset_enabled_flag = reader.read_bit()?;

        let pcm_enabled_flag = reader.read_bit()?;
        if pcm_enabled_flag {
            reader.skip(4)?; // pcm_sample_bit_depth_luma_minus1
            reader.skip(4)?; // pcm_sample_bit_depth_chroma_minus1
            reader.read_ue()?; // log2_min_pcm_luma_coding_block_size_minus3
            reader.read_ue()?; // log2_diff_max_min_pcm_luma_coding_block_size
            reader.skip(1)?; // pcm_loop_filter_disabled_flag
        }

        let num_short_term_ref_pic_sets = reader.read_ue()? as u8;
        let mut st_ref_pic_sets = Vec::with_capacity(num_short_term_ref_pic_sets as usize);

        for i in 0..num_short_term_ref_pic_sets as usize {
            let st_rps = ShortTermRefPicSet::parse(
                &mut reader,
                i,
                num_short_term_ref_pic_sets as usize,
                &st_ref_pic_sets,
            )?;
            st_ref_pic_sets.push(st_rps);
        }

        let long_term_ref_pics_present_flag = reader.read_bit()?;
        if long_term_ref_pics_present_flag {
            let num_long_term_ref_pics_sps = reader.read_ue()?;
            for _ in 0..num_long_term_ref_pics_sps {
                reader.skip(log2_max_pic_order_cnt_lsb_minus4 as usize + 4)?;
                reader.skip(1)?; // used_by_curr_pic_lt_sps_flag
            }
        }

        let sps_temporal_mvp_enabled_flag = reader.read_bit()?;
        let strong_intra_smoothing_enabled_flag = reader.read_bit()?;
        let vui_parameters_present_flag = reader.read_bit()?;

        Ok(Self {
            sps_video_parameter_set_id,
            sps_max_sub_layers_minus1,
            sps_temporal_id_nesting_flag,
            profile_tier_level,
            sps_seq_parameter_set_id,
            chroma_format_idc,
            separate_colour_plane_flag,
            pic_width_in_luma_samples,
            pic_height_in_luma_samples,
            conformance_window_flag,
            conf_win_left_offset,
            conf_win_right_offset,
            conf_win_top_offset,
            conf_win_bottom_offset,
            bit_depth_luma_minus8,
            bit_depth_chroma_minus8,
            log2_max_pic_order_cnt_lsb_minus4,
            sps_sub_layer_ordering_info_present_flag,
            log2_min_luma_coding_block_size_minus3,
            log2_diff_max_min_luma_coding_block_size,
            log2_min_luma_transform_block_size_minus2,
            log2_diff_max_min_luma_transform_block_size,
            max_transform_hierarchy_depth_inter,
            max_transform_hierarchy_depth_intra,
            scaling_list_enabled_flag,
            amp_enabled_flag,
            sample_adaptive_offset_enabled_flag,
            pcm_enabled_flag,
            num_short_term_ref_pic_sets,
            st_ref_pic_sets,
            long_term_ref_pics_present_flag,
            sps_temporal_mvp_enabled_flag,
            strong_intra_smoothing_enabled_flag,
            vui_parameters_present_flag,
        })
    }

    fn skip_scaling_list_data(reader: &mut BitReader) -> Result<()> {
        for size_id in 0..4 {
            let num_matrices = if size_id == 3 { 2 } else { 6 };
            for matrix_id in 0..num_matrices {
                let scaling_list_pred_mode_flag = reader.read_bit()?;
                if !scaling_list_pred_mode_flag {
                    reader.read_ue()?; // scaling_list_pred_matrix_id_delta
                } else {
                    let coef_num = std::cmp::min(64, 1 << (4 + (size_id << 1)));
                    if size_id > 1 {
                        reader.read_se()?; // scaling_list_dc_coef_minus8
                    }
                    for _ in 0..coef_num {
                        reader.read_se()?; // scaling_list_delta_coef
                    }
                }
            }
        }
        Ok(())
    }

    /// Get the actual picture width after conformance window cropping.
    pub fn width(&self) -> u32 {
        let sub_width_c = self.sub_width_c();
        self.pic_width_in_luma_samples
            - sub_width_c * (self.conf_win_left_offset + self.conf_win_right_offset)
    }

    /// Get the actual picture height after conformance window cropping.
    pub fn height(&self) -> u32 {
        let sub_height_c = self.sub_height_c();
        self.pic_height_in_luma_samples
            - sub_height_c * (self.conf_win_top_offset + self.conf_win_bottom_offset)
    }

    /// Get chroma subsampling width factor.
    pub fn sub_width_c(&self) -> u32 {
        match self.chroma_format_idc {
            1 | 2 => 2,
            _ => 1,
        }
    }

    /// Get chroma subsampling height factor.
    pub fn sub_height_c(&self) -> u32 {
        match self.chroma_format_idc {
            1 => 2,
            _ => 1,
        }
    }

    /// Get bit depth for luma.
    pub fn bit_depth_luma(&self) -> u8 {
        8 + self.bit_depth_luma_minus8
    }

    /// Get bit depth for chroma.
    pub fn bit_depth_chroma(&self) -> u8 {
        8 + self.bit_depth_chroma_minus8
    }

    /// Get the log2 of the minimum CTU size.
    pub fn log2_min_cb_size(&self) -> u8 {
        self.log2_min_luma_coding_block_size_minus3 + 3
    }

    /// Get the log2 of the maximum CTU size.
    pub fn log2_ctb_size(&self) -> u8 {
        self.log2_min_cb_size() + self.log2_diff_max_min_luma_coding_block_size
    }

    /// Get the CTU size (CTB size).
    pub fn ctb_size(&self) -> u32 {
        1 << self.log2_ctb_size()
    }

    /// Get the minimum CB size.
    pub fn min_cb_size(&self) -> u32 {
        1 << self.log2_min_cb_size()
    }

    /// Get the picture width in CTUs.
    pub fn pic_width_in_ctbs(&self) -> u32 {
        (self.pic_width_in_luma_samples + self.ctb_size() - 1) / self.ctb_size()
    }

    /// Get the picture height in CTUs.
    pub fn pic_height_in_ctbs(&self) -> u32 {
        (self.pic_height_in_luma_samples + self.ctb_size() - 1) / self.ctb_size()
    }

    /// Get the minimum TU size.
    pub fn min_tb_size(&self) -> u32 {
        1 << (self.log2_min_luma_transform_block_size_minus2 + 2)
    }

    /// Get the maximum TU size.
    pub fn max_tb_size(&self) -> u32 {
        self.min_tb_size() << self.log2_diff_max_min_luma_transform_block_size
    }
}

/// Picture Parameter Set (PPS).
#[derive(Debug, Clone)]
pub struct Pps {
    /// PPS ID.
    pub pps_pic_parameter_set_id: u8,
    /// SPS ID.
    pub pps_seq_parameter_set_id: u8,
    /// Dependent slice segments enabled flag.
    pub dependent_slice_segments_enabled_flag: bool,
    /// Output flag present flag.
    pub output_flag_present_flag: bool,
    /// Number of extra slice header bits.
    pub num_extra_slice_header_bits: u8,
    /// Sign data hiding enabled flag.
    pub sign_data_hiding_enabled_flag: bool,
    /// CABAC init present flag.
    pub cabac_init_present_flag: bool,
    /// Number of reference index L0 default active minus 1.
    pub num_ref_idx_l0_default_active_minus1: u8,
    /// Number of reference index L1 default active minus 1.
    pub num_ref_idx_l1_default_active_minus1: u8,
    /// Initial QP minus 26.
    pub init_qp_minus26: i8,
    /// Constrained intra prediction flag.
    pub constrained_intra_pred_flag: bool,
    /// Transform skip enabled flag.
    pub transform_skip_enabled_flag: bool,
    /// CU QP delta enabled flag.
    pub cu_qp_delta_enabled_flag: bool,
    /// Diff CU QP delta depth.
    pub diff_cu_qp_delta_depth: u8,
    /// Cb QP offset.
    pub pps_cb_qp_offset: i8,
    /// Cr QP offset.
    pub pps_cr_qp_offset: i8,
    /// Slice chroma QP offsets present flag.
    pub pps_slice_chroma_qp_offsets_present_flag: bool,
    /// Weighted prediction flag.
    pub weighted_pred_flag: bool,
    /// Weighted biprediction flag.
    pub weighted_bipred_flag: bool,
    /// Transquant bypass enabled flag.
    pub transquant_bypass_enabled_flag: bool,
    /// Tiles enabled flag.
    pub tiles_enabled_flag: bool,
    /// Entropy coding sync enabled flag.
    pub entropy_coding_sync_enabled_flag: bool,
    /// Number of tile columns minus 1.
    pub num_tile_columns_minus1: u32,
    /// Number of tile rows minus 1.
    pub num_tile_rows_minus1: u32,
    /// Uniform spacing flag.
    pub uniform_spacing_flag: bool,
    /// Loop filter across tiles enabled flag.
    pub loop_filter_across_tiles_enabled_flag: bool,
    /// Loop filter across slices enabled flag.
    pub pps_loop_filter_across_slices_enabled_flag: bool,
    /// Deblocking filter control present flag.
    pub deblocking_filter_control_present_flag: bool,
    /// Deblocking filter override enabled flag.
    pub deblocking_filter_override_enabled_flag: bool,
    /// Deblocking filter disabled flag.
    pub pps_deblocking_filter_disabled_flag: bool,
    /// Beta offset div 2.
    pub pps_beta_offset_div2: i8,
    /// Tc offset div 2.
    pub pps_tc_offset_div2: i8,
    /// Scaling list data present flag.
    pub pps_scaling_list_data_present_flag: bool,
    /// Lists modification present flag.
    pub lists_modification_present_flag: bool,
    /// Log2 parallel merge level minus 2.
    pub log2_parallel_merge_level_minus2: u8,
    /// Slice segment header extension present flag.
    pub slice_segment_header_extension_present_flag: bool,
}

impl Pps {
    /// Parse PPS from RBSP data.
    pub fn parse(rbsp: &[u8]) -> Result<Self> {
        let mut reader = BitReader::new(rbsp);

        let pps_pic_parameter_set_id = reader.read_ue()? as u8;
        let pps_seq_parameter_set_id = reader.read_ue()? as u8;
        let dependent_slice_segments_enabled_flag = reader.read_bit()?;
        let output_flag_present_flag = reader.read_bit()?;
        let num_extra_slice_header_bits = reader.read_bits(3)? as u8;
        let sign_data_hiding_enabled_flag = reader.read_bit()?;
        let cabac_init_present_flag = reader.read_bit()?;
        let num_ref_idx_l0_default_active_minus1 = reader.read_ue()? as u8;
        let num_ref_idx_l1_default_active_minus1 = reader.read_ue()? as u8;
        let init_qp_minus26 = reader.read_se()? as i8;
        let constrained_intra_pred_flag = reader.read_bit()?;
        let transform_skip_enabled_flag = reader.read_bit()?;

        let cu_qp_delta_enabled_flag = reader.read_bit()?;
        let diff_cu_qp_delta_depth = if cu_qp_delta_enabled_flag {
            reader.read_ue()? as u8
        } else {
            0
        };

        let pps_cb_qp_offset = reader.read_se()? as i8;
        let pps_cr_qp_offset = reader.read_se()? as i8;
        let pps_slice_chroma_qp_offsets_present_flag = reader.read_bit()?;
        let weighted_pred_flag = reader.read_bit()?;
        let weighted_bipred_flag = reader.read_bit()?;
        let transquant_bypass_enabled_flag = reader.read_bit()?;
        let tiles_enabled_flag = reader.read_bit()?;
        let entropy_coding_sync_enabled_flag = reader.read_bit()?;

        let mut num_tile_columns_minus1 = 0;
        let mut num_tile_rows_minus1 = 0;
        let mut uniform_spacing_flag = true;
        let mut loop_filter_across_tiles_enabled_flag = true;

        if tiles_enabled_flag {
            num_tile_columns_minus1 = reader.read_ue()?;
            num_tile_rows_minus1 = reader.read_ue()?;
            uniform_spacing_flag = reader.read_bit()?;

            if !uniform_spacing_flag {
                for _ in 0..num_tile_columns_minus1 {
                    reader.read_ue()?; // column_width_minus1
                }
                for _ in 0..num_tile_rows_minus1 {
                    reader.read_ue()?; // row_height_minus1
                }
            }

            loop_filter_across_tiles_enabled_flag = reader.read_bit()?;
        }

        let pps_loop_filter_across_slices_enabled_flag = reader.read_bit()?;
        let deblocking_filter_control_present_flag = reader.read_bit()?;

        let mut deblocking_filter_override_enabled_flag = false;
        let mut pps_deblocking_filter_disabled_flag = false;
        let mut pps_beta_offset_div2 = 0;
        let mut pps_tc_offset_div2 = 0;

        if deblocking_filter_control_present_flag {
            deblocking_filter_override_enabled_flag = reader.read_bit()?;
            pps_deblocking_filter_disabled_flag = reader.read_bit()?;

            if !pps_deblocking_filter_disabled_flag {
                pps_beta_offset_div2 = reader.read_se()? as i8;
                pps_tc_offset_div2 = reader.read_se()? as i8;
            }
        }

        let pps_scaling_list_data_present_flag = reader.read_bit()?;
        if pps_scaling_list_data_present_flag {
            // Skip scaling list data
            Sps::skip_scaling_list_data(&mut reader)?;
        }

        let lists_modification_present_flag = reader.read_bit()?;
        let log2_parallel_merge_level_minus2 = reader.read_ue()? as u8;
        let slice_segment_header_extension_present_flag = reader.read_bit()?;

        Ok(Self {
            pps_pic_parameter_set_id,
            pps_seq_parameter_set_id,
            dependent_slice_segments_enabled_flag,
            output_flag_present_flag,
            num_extra_slice_header_bits,
            sign_data_hiding_enabled_flag,
            cabac_init_present_flag,
            num_ref_idx_l0_default_active_minus1,
            num_ref_idx_l1_default_active_minus1,
            init_qp_minus26,
            constrained_intra_pred_flag,
            transform_skip_enabled_flag,
            cu_qp_delta_enabled_flag,
            diff_cu_qp_delta_depth,
            pps_cb_qp_offset,
            pps_cr_qp_offset,
            pps_slice_chroma_qp_offsets_present_flag,
            weighted_pred_flag,
            weighted_bipred_flag,
            transquant_bypass_enabled_flag,
            tiles_enabled_flag,
            entropy_coding_sync_enabled_flag,
            num_tile_columns_minus1,
            num_tile_rows_minus1,
            uniform_spacing_flag,
            loop_filter_across_tiles_enabled_flag,
            pps_loop_filter_across_slices_enabled_flag,
            deblocking_filter_control_present_flag,
            deblocking_filter_override_enabled_flag,
            pps_deblocking_filter_disabled_flag,
            pps_beta_offset_div2,
            pps_tc_offset_div2,
            pps_scaling_list_data_present_flag,
            lists_modification_present_flag,
            log2_parallel_merge_level_minus2,
            slice_segment_header_extension_present_flag,
        })
    }

    /// Get the initial QP value.
    pub fn init_qp(&self) -> i8 {
        26 + self.init_qp_minus26
    }
}

/// Slice type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SliceType {
    /// B-slice (bidirectional prediction).
    B = 0,
    /// P-slice (forward prediction).
    P = 1,
    /// I-slice (intra only).
    I = 2,
}

impl SliceType {
    /// Create from raw value.
    pub fn from_raw(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::B),
            1 => Ok(Self::P),
            2 => Ok(Self::I),
            _ => Err(HevcError::SliceHeader(format!("Invalid slice type: {}", value))),
        }
    }

    /// Check if this is an intra slice.
    pub fn is_intra(&self) -> bool {
        matches!(self, Self::I)
    }

    /// Check if this uses forward prediction.
    pub fn uses_list0(&self) -> bool {
        !matches!(self, Self::I)
    }

    /// Check if this uses backward prediction.
    pub fn uses_list1(&self) -> bool {
        matches!(self, Self::B)
    }
}

impl fmt::Display for SliceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::B => write!(f, "B"),
            Self::P => write!(f, "P"),
            Self::I => write!(f, "I"),
        }
    }
}

/// Slice segment header.
#[derive(Debug, Clone)]
pub struct SliceSegmentHeader {
    /// First slice segment in picture flag.
    pub first_slice_segment_in_pic_flag: bool,
    /// No output of prior pics flag (for IRAP).
    pub no_output_of_prior_pics_flag: bool,
    /// PPS ID.
    pub slice_pic_parameter_set_id: u8,
    /// Dependent slice segment flag.
    pub dependent_slice_segment_flag: bool,
    /// Slice segment address.
    pub slice_segment_address: u32,
    /// Slice type.
    pub slice_type: SliceType,
    /// Picture output flag.
    pub pic_output_flag: bool,
    /// Color plane ID (for separate color planes).
    pub colour_plane_id: u8,
    /// Slice picture order count LSB.
    pub slice_pic_order_cnt_lsb: u32,
    /// Short-term reference picture set SPS flag.
    pub short_term_ref_pic_set_sps_flag: bool,
    /// Short-term reference picture set index.
    pub short_term_ref_pic_set_idx: u8,
    /// Number of long-term SPS pictures.
    pub num_long_term_sps: u32,
    /// Number of long-term pictures.
    pub num_long_term_pics: u32,
    /// Slice temporal MVP enabled flag.
    pub slice_temporal_mvp_enabled_flag: bool,
    /// Slice SAO luma flag.
    pub slice_sao_luma_flag: bool,
    /// Slice SAO chroma flag.
    pub slice_sao_chroma_flag: bool,
    /// Number of reference pictures in L0.
    pub num_ref_idx_l0_active_minus1: u8,
    /// Number of reference pictures in L1.
    pub num_ref_idx_l1_active_minus1: u8,
    /// CABAC init flag.
    pub cabac_init_flag: bool,
    /// Collocated from L0 flag.
    pub collocated_from_l0_flag: bool,
    /// Five minus max number of merge candidates.
    pub five_minus_max_num_merge_cand: u8,
    /// Slice QP delta.
    pub slice_qp_delta: i8,
    /// Slice Cb QP offset.
    pub slice_cb_qp_offset: i8,
    /// Slice Cr QP offset.
    pub slice_cr_qp_offset: i8,
    /// Deblocking filter override flag.
    pub deblocking_filter_override_flag: bool,
    /// Slice deblocking filter disabled flag.
    pub slice_deblocking_filter_disabled_flag: bool,
    /// Slice beta offset div 2.
    pub slice_beta_offset_div2: i8,
    /// Slice tc offset div 2.
    pub slice_tc_offset_div2: i8,
    /// Slice loop filter across slices enabled flag.
    pub slice_loop_filter_across_slices_enabled_flag: bool,
}

impl SliceSegmentHeader {
    /// Parse slice segment header.
    pub fn parse(rbsp: &[u8], sps: &Sps, pps: &Pps, nal_unit_type: NalUnitType) -> Result<Self> {
        let mut reader = BitReader::new(rbsp);

        let first_slice_segment_in_pic_flag = reader.read_bit()?;

        let no_output_of_prior_pics_flag = if nal_unit_type.is_irap() {
            reader.read_bit()?
        } else {
            false
        };

        let slice_pic_parameter_set_id = reader.read_ue()? as u8;

        let dependent_slice_segment_flag = if !first_slice_segment_in_pic_flag
            && pps.dependent_slice_segments_enabled_flag
        {
            reader.read_bit()?
        } else {
            false
        };

        let slice_segment_address = if !first_slice_segment_in_pic_flag {
            let num_ctbs = sps.pic_width_in_ctbs() * sps.pic_height_in_ctbs();
            let bits = (32 - num_ctbs.leading_zeros()) as u8;
            reader.read_bits(bits)?
        } else {
            0
        };

        // Skip extra slice header bits
        for _ in 0..pps.num_extra_slice_header_bits {
            reader.skip(1)?;
        }

        let slice_type = SliceType::from_raw(reader.read_ue()?)?;

        let pic_output_flag = if pps.output_flag_present_flag {
            reader.read_bit()?
        } else {
            true
        };

        let colour_plane_id = if sps.separate_colour_plane_flag {
            reader.read_bits(2)? as u8
        } else {
            0
        };

        let mut slice_pic_order_cnt_lsb = 0;
        let mut short_term_ref_pic_set_sps_flag = false;
        let mut short_term_ref_pic_set_idx = 0;
        let mut num_long_term_sps = 0;
        let mut num_long_term_pics = 0;
        let mut slice_temporal_mvp_enabled_flag = false;

        if !nal_unit_type.is_idr() {
            let poc_bits = sps.log2_max_pic_order_cnt_lsb_minus4 + 4;
            slice_pic_order_cnt_lsb = reader.read_bits(poc_bits)?;

            short_term_ref_pic_set_sps_flag = reader.read_bit()?;

            if !short_term_ref_pic_set_sps_flag {
                // Parse short-term ref pic set in slice header
                let _st_rps = ShortTermRefPicSet::parse(
                    &mut reader,
                    sps.num_short_term_ref_pic_sets as usize,
                    sps.num_short_term_ref_pic_sets as usize,
                    &sps.st_ref_pic_sets,
                )?;
            } else if sps.num_short_term_ref_pic_sets > 1 {
                let bits = (32 - (sps.num_short_term_ref_pic_sets as u32).leading_zeros()) as u8;
                short_term_ref_pic_set_idx = reader.read_bits(bits)? as u8;
            }

            if sps.long_term_ref_pics_present_flag {
                // Parse long-term ref pics
                num_long_term_sps = reader.read_ue()?;
                num_long_term_pics = reader.read_ue()?;

                for _ in 0..(num_long_term_sps + num_long_term_pics) {
                    // Skip long-term ref pic parsing
                    reader.read_ue()?;
                    reader.skip(sps.log2_max_pic_order_cnt_lsb_minus4 as usize + 4)?;
                }
            }

            if sps.sps_temporal_mvp_enabled_flag {
                slice_temporal_mvp_enabled_flag = reader.read_bit()?;
            }
        }

        let slice_sao_luma_flag = if sps.sample_adaptive_offset_enabled_flag {
            reader.read_bit()?
        } else {
            false
        };

        let slice_sao_chroma_flag = if sps.sample_adaptive_offset_enabled_flag && sps.chroma_format_idc != 0 {
            reader.read_bit()?
        } else {
            false
        };

        let mut num_ref_idx_l0_active_minus1 = pps.num_ref_idx_l0_default_active_minus1;
        let mut num_ref_idx_l1_active_minus1 = pps.num_ref_idx_l1_default_active_minus1;
        let mut cabac_init_flag = false;
        let mut collocated_from_l0_flag = true;
        let mut five_minus_max_num_merge_cand = 0;

        if !slice_type.is_intra() {
            let num_ref_idx_active_override_flag = reader.read_bit()?;
            if num_ref_idx_active_override_flag {
                num_ref_idx_l0_active_minus1 = reader.read_ue()? as u8;
                if slice_type.uses_list1() {
                    num_ref_idx_l1_active_minus1 = reader.read_ue()? as u8;
                }
            }

            if pps.lists_modification_present_flag {
                // Skip ref pic list modification
            }

            if slice_type.uses_list1() {
                reader.skip(1)?; // mvd_l1_zero_flag
            }

            if pps.cabac_init_present_flag {
                cabac_init_flag = reader.read_bit()?;
            }

            if slice_temporal_mvp_enabled_flag {
                if slice_type.uses_list1() {
                    collocated_from_l0_flag = reader.read_bit()?;
                }
                // Skip collocated_ref_idx
            }

            if (pps.weighted_pred_flag && slice_type == SliceType::P)
                || (pps.weighted_bipred_flag && slice_type == SliceType::B)
            {
                // Skip pred_weight_table
            }

            five_minus_max_num_merge_cand = reader.read_ue()? as u8;
        }

        let slice_qp_delta = reader.read_se()? as i8;

        let slice_cb_qp_offset = if pps.pps_slice_chroma_qp_offsets_present_flag {
            reader.read_se()? as i8
        } else {
            0
        };

        let slice_cr_qp_offset = if pps.pps_slice_chroma_qp_offsets_present_flag {
            reader.read_se()? as i8
        } else {
            0
        };

        let deblocking_filter_override_flag = if pps.deblocking_filter_override_enabled_flag {
            reader.read_bit()?
        } else {
            false
        };

        let mut slice_deblocking_filter_disabled_flag = pps.pps_deblocking_filter_disabled_flag;
        let mut slice_beta_offset_div2 = pps.pps_beta_offset_div2;
        let mut slice_tc_offset_div2 = pps.pps_tc_offset_div2;

        if deblocking_filter_override_flag {
            slice_deblocking_filter_disabled_flag = reader.read_bit()?;
            if !slice_deblocking_filter_disabled_flag {
                slice_beta_offset_div2 = reader.read_se()? as i8;
                slice_tc_offset_div2 = reader.read_se()? as i8;
            }
        }

        let slice_loop_filter_across_slices_enabled_flag =
            if pps.pps_loop_filter_across_slices_enabled_flag
                && (slice_sao_luma_flag
                    || slice_sao_chroma_flag
                    || !slice_deblocking_filter_disabled_flag)
            {
                reader.read_bit()?
            } else {
                false
            };

        Ok(Self {
            first_slice_segment_in_pic_flag,
            no_output_of_prior_pics_flag,
            slice_pic_parameter_set_id,
            dependent_slice_segment_flag,
            slice_segment_address,
            slice_type,
            pic_output_flag,
            colour_plane_id,
            slice_pic_order_cnt_lsb,
            short_term_ref_pic_set_sps_flag,
            short_term_ref_pic_set_idx,
            num_long_term_sps,
            num_long_term_pics,
            slice_temporal_mvp_enabled_flag,
            slice_sao_luma_flag,
            slice_sao_chroma_flag,
            num_ref_idx_l0_active_minus1,
            num_ref_idx_l1_active_minus1,
            cabac_init_flag,
            collocated_from_l0_flag,
            five_minus_max_num_merge_cand,
            slice_qp_delta,
            slice_cb_qp_offset,
            slice_cr_qp_offset,
            deblocking_filter_override_flag,
            slice_deblocking_filter_disabled_flag,
            slice_beta_offset_div2,
            slice_tc_offset_div2,
            slice_loop_filter_across_slices_enabled_flag,
        })
    }

    /// Get the slice QP.
    pub fn slice_qp(&self, pps: &Pps) -> i8 {
        pps.init_qp() + self.slice_qp_delta
    }

    /// Get maximum number of merge candidates.
    pub fn max_num_merge_cand(&self) -> u8 {
        5 - self.five_minus_max_num_merge_cand
    }
}

/// Parse NAL units from Annex B byte stream.
pub fn parse_annexb_nal_units(data: &[u8]) -> Vec<(NalUnitHeader, Vec<u8>)> {
    let mut result = Vec::new();
    let mut pos = 0;

    while pos < data.len() {
        // Find start code
        let start_code_pos = find_start_code(&data[pos..]);
        if start_code_pos.is_none() {
            break;
        }

        let (sc_offset, sc_len) = start_code_pos.unwrap();
        let nal_start = pos + sc_offset + sc_len;

        if nal_start >= data.len() {
            break;
        }

        // Find next start code
        let next_sc = find_start_code(&data[nal_start..]);
        let nal_end = if let Some((offset, _)) = next_sc {
            nal_start + offset
        } else {
            data.len()
        };

        // Parse NAL header
        let nal_data = &data[nal_start..nal_end];
        if let Ok(header) = NalUnitHeader::parse(nal_data) {
            let rbsp = remove_emulation_prevention(&nal_data[2..]);
            result.push((header, rbsp));
        }

        pos = nal_end;
    }

    result
}

/// Find start code in data.
fn find_start_code(data: &[u8]) -> Option<(usize, usize)> {
    if data.len() < 3 {
        return None;
    }

    for i in 0..data.len() - 2 {
        if data[i] == 0 && data[i + 1] == 0 {
            if data[i + 2] == 1 {
                return Some((i, 3));
            } else if i + 3 < data.len() && data[i + 2] == 0 && data[i + 3] == 1 {
                return Some((i, 4));
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nal_unit_type_classification() {
        assert!(NalUnitType::IdrWRadl.is_idr());
        assert!(NalUnitType::IdrNLp.is_idr());
        assert!(!NalUnitType::TrailR.is_idr());

        assert!(NalUnitType::IdrWRadl.is_irap());
        assert!(NalUnitType::CraNut.is_irap());
        assert!(!NalUnitType::TrailR.is_irap());

        assert!(NalUnitType::TrailR.is_vcl());
        assert!(!NalUnitType::VpsNut.is_vcl());
        assert!(!NalUnitType::SpsNut.is_vcl());
        assert!(!NalUnitType::PpsNut.is_vcl());

        assert!(NalUnitType::TrailR.is_reference());
        assert!(!NalUnitType::TrailN.is_reference());
    }

    #[test]
    fn test_nal_unit_header_parse() {
        // forbidden_zero_bit=0, nal_unit_type=1 (TRAIL_R), layer_id=0, temporal_id=1
        let data = [0x02, 0x01];
        let header = NalUnitHeader::parse(&data).unwrap();
        assert_eq!(header.nal_unit_type, NalUnitType::TrailR);
        assert_eq!(header.nuh_layer_id, 0);
        assert_eq!(header.nuh_temporal_id_plus1, 1);
    }

    #[test]
    fn test_nal_unit_header_forbidden_bit() {
        // forbidden_zero_bit=1 should fail
        let data = [0x82, 0x01];
        assert!(NalUnitHeader::parse(&data).is_err());
    }

    #[test]
    fn test_nal_unit_header_zero_temporal_id() {
        // temporal_id_plus1=0 should fail
        let data = [0x02, 0x00];
        assert!(NalUnitHeader::parse(&data).is_err());
    }

    #[test]
    fn test_slice_type() {
        assert!(SliceType::I.is_intra());
        assert!(!SliceType::P.is_intra());
        assert!(!SliceType::B.is_intra());

        assert!(SliceType::B.uses_list1());
        assert!(!SliceType::P.uses_list1());
        assert!(!SliceType::I.uses_list1());

        assert!(SliceType::P.uses_list0());
        assert!(SliceType::B.uses_list0());
        assert!(!SliceType::I.uses_list0());
    }

    #[test]
    fn test_find_start_code() {
        let data = [0x00, 0x00, 0x01, 0x40];
        assert_eq!(find_start_code(&data), Some((0, 3)));

        let data = [0x00, 0x00, 0x00, 0x01, 0x40];
        assert_eq!(find_start_code(&data), Some((0, 4)));

        let data = [0xFF, 0x00, 0x00, 0x01, 0x40];
        assert_eq!(find_start_code(&data), Some((1, 3)));

        let data = [0xFF, 0xFF, 0xFF];
        assert_eq!(find_start_code(&data), None);
    }

    #[test]
    fn test_nal_unit_type_display() {
        assert_eq!(NalUnitType::IdrWRadl.to_string(), "IDR_W_RADL");
        assert_eq!(NalUnitType::VpsNut.to_string(), "VPS_NUT");
        assert_eq!(NalUnitType::SpsNut.to_string(), "SPS_NUT");
        assert_eq!(NalUnitType::PpsNut.to_string(), "PPS_NUT");
    }

    #[test]
    fn test_nal_unit_type_roundtrip() {
        for i in 0..64 {
            let nal_type = NalUnitType::from_raw(i);
            assert_eq!(nal_type.to_raw(), i);
        }
    }
}
