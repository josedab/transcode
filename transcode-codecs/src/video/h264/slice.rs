//! Slice header parsing.

use transcode_core::bitstream::BitReader;
use transcode_core::error::Result;
use super::sps::SequenceParameterSet;
use super::pps::PictureParameterSet;
use super::nal::NalUnitType;

/// Slice type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceType {
    /// P slice (predictive).
    P = 0,
    /// B slice (bidirectional).
    B = 1,
    /// I slice (intra).
    I = 2,
    /// SP slice (switching P).
    Sp = 3,
    /// SI slice (switching I).
    Si = 4,
}

impl SliceType {
    /// Create from slice_type value.
    pub fn from_value(value: u32) -> Option<Self> {
        match value % 5 {
            0 => Some(Self::P),
            1 => Some(Self::B),
            2 => Some(Self::I),
            3 => Some(Self::Sp),
            4 => Some(Self::Si),
            _ => None,
        }
    }

    /// Check if all macroblocks in this slice type are intra.
    pub fn is_intra(&self) -> bool {
        matches!(self, Self::I | Self::Si)
    }

    /// Check if this slice type uses bidirectional prediction.
    pub fn is_bidirectional(&self) -> bool {
        matches!(self, Self::B)
    }
}

/// Slice header.
#[derive(Debug, Clone)]
pub struct SliceHeader {
    /// First macroblock address in slice.
    pub first_mb_in_slice: u32,
    /// Slice type.
    pub slice_type: SliceType,
    /// PPS ID.
    pub pps_id: u8,
    /// Color plane ID (for separate color plane).
    pub colour_plane_id: u8,
    /// Frame number.
    pub frame_num: u32,
    /// Field pic flag.
    pub field_pic_flag: bool,
    /// Bottom field flag.
    pub bottom_field_flag: bool,
    /// IDR picture ID.
    pub idr_pic_id: u32,
    /// Picture order count LSB.
    pub pic_order_cnt_lsb: u32,
    /// Delta POC bottom.
    pub delta_pic_order_cnt_bottom: i32,
    /// Delta POC [0].
    pub delta_pic_order_cnt_0: i32,
    /// Delta POC [1].
    pub delta_pic_order_cnt_1: i32,
    /// Redundant picture count.
    pub redundant_pic_cnt: u32,
    /// Direct spatial MV pred flag.
    pub direct_spatial_mv_pred_flag: bool,
    /// Number of reference indices for L0.
    pub num_ref_idx_l0_active_minus1: u8,
    /// Number of reference indices for L1.
    pub num_ref_idx_l1_active_minus1: u8,
    /// CABAC init IDC.
    pub cabac_init_idc: u8,
    /// Slice QP delta.
    pub slice_qp_delta: i8,
    /// SP for switch flag.
    pub sp_for_switch_flag: bool,
    /// Slice QS delta.
    pub slice_qs_delta: i8,
    /// Disable deblocking filter IDC.
    pub disable_deblocking_filter_idc: u8,
    /// Slice alpha C0 offset div 2.
    pub slice_alpha_c0_offset_div2: i8,
    /// Slice beta offset div 2.
    pub slice_beta_offset_div2: i8,
}

impl SliceHeader {
    /// Parse a slice header.
    pub fn parse(
        reader: &mut BitReader<'_>,
        nal_type: NalUnitType,
        sps: &SequenceParameterSet,
        pps: &PictureParameterSet,
    ) -> Result<Self> {
        let first_mb_in_slice = reader.read_ue()?;
        let slice_type_value = reader.read_ue()?;
        let slice_type = SliceType::from_value(slice_type_value)
            .ok_or_else(|| transcode_core::error::CodecError::SliceError(
                format!("Invalid slice type: {}", slice_type_value)
            ))?;

        let pps_id = reader.read_ue()? as u8;

        let colour_plane_id = if sps.separate_colour_plane_flag {
            reader.read_bits(2)? as u8
        } else {
            0
        };

        let frame_num = reader.read_bits(sps.log2_max_frame_num)?;

        let (field_pic_flag, bottom_field_flag) = if !sps.frame_mbs_only_flag {
            let fpf = reader.read_bit()?;
            let bff = if fpf { reader.read_bit()? } else { false };
            (fpf, bff)
        } else {
            (false, false)
        };

        let idr_pic_id = if nal_type == NalUnitType::IdrSlice {
            reader.read_ue()?
        } else {
            0
        };

        let (pic_order_cnt_lsb, delta_pic_order_cnt_bottom, delta_pic_order_cnt_0, delta_pic_order_cnt_1) =
            match sps.pic_order_cnt_type {
                0 => {
                    let lsb = reader.read_bits(sps.log2_max_pic_order_cnt_lsb)?;
                    let bottom = if pps.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
                        reader.read_se()?
                    } else {
                        0
                    };
                    (lsb, bottom, 0, 0)
                }
                1 if !sps.delta_pic_order_always_zero_flag => {
                    let d0 = reader.read_se()?;
                    let d1 = if pps.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
                        reader.read_se()?
                    } else {
                        0
                    };
                    (0, 0, d0, d1)
                }
                _ => (0, 0, 0, 0),
            };

        let redundant_pic_cnt = if pps.redundant_pic_cnt_present_flag {
            reader.read_ue()?
        } else {
            0
        };

        let direct_spatial_mv_pred_flag = if slice_type == SliceType::B {
            reader.read_bit()?
        } else {
            false
        };

        let (num_ref_idx_l0_active_minus1, num_ref_idx_l1_active_minus1) =
            if slice_type == SliceType::P || slice_type == SliceType::Sp || slice_type == SliceType::B {
                let override_flag = reader.read_bit()?;
                if override_flag {
                    let l0 = reader.read_ue()? as u8;
                    let l1 = if slice_type == SliceType::B {
                        reader.read_ue()? as u8
                    } else {
                        pps.num_ref_idx_l1_default_active_minus1
                    };
                    (l0, l1)
                } else {
                    (pps.num_ref_idx_l0_default_active_minus1, pps.num_ref_idx_l1_default_active_minus1)
                }
            } else {
                (0, 0)
            };

        // Skip ref_pic_list_modification
        if !slice_type.is_intra() {
            skip_ref_pic_list_modification(reader, slice_type)?;
        }

        // Skip pred_weight_table
        if (pps.weighted_pred_flag && (slice_type == SliceType::P || slice_type == SliceType::Sp))
            || (pps.weighted_bipred_idc == 1 && slice_type == SliceType::B)
        {
            skip_pred_weight_table(reader, sps, slice_type, num_ref_idx_l0_active_minus1, num_ref_idx_l1_active_minus1)?;
        }

        // Skip dec_ref_pic_marking
        if nal_type == NalUnitType::IdrSlice || (nal_type.is_reference() && slice_type.is_intra()) {
            skip_dec_ref_pic_marking(reader, nal_type)?;
        }

        let cabac_init_idc = if pps.entropy_coding_mode_flag && !slice_type.is_intra() {
            reader.read_ue()? as u8
        } else {
            0
        };

        let slice_qp_delta = reader.read_se()? as i8;

        let (sp_for_switch_flag, slice_qs_delta) = if slice_type == SliceType::Sp || slice_type == SliceType::Si {
            let spf = if slice_type == SliceType::Sp { reader.read_bit()? } else { false };
            let qs = reader.read_se()? as i8;
            (spf, qs)
        } else {
            (false, 0)
        };

        let (disable_deblocking_filter_idc, slice_alpha_c0_offset_div2, slice_beta_offset_div2) =
            if pps.deblocking_filter_control_present_flag {
                let idc = reader.read_ue()? as u8;
                if idc != 1 {
                    let alpha = reader.read_se()? as i8;
                    let beta = reader.read_se()? as i8;
                    (idc, alpha, beta)
                } else {
                    (idc, 0, 0)
                }
            } else {
                (0, 0, 0)
            };

        Ok(Self {
            first_mb_in_slice,
            slice_type,
            pps_id,
            colour_plane_id,
            frame_num,
            field_pic_flag,
            bottom_field_flag,
            idr_pic_id,
            pic_order_cnt_lsb,
            delta_pic_order_cnt_bottom,
            delta_pic_order_cnt_0,
            delta_pic_order_cnt_1,
            redundant_pic_cnt,
            direct_spatial_mv_pred_flag,
            num_ref_idx_l0_active_minus1,
            num_ref_idx_l1_active_minus1,
            cabac_init_idc,
            slice_qp_delta,
            sp_for_switch_flag,
            slice_qs_delta,
            disable_deblocking_filter_idc,
            slice_alpha_c0_offset_div2,
            slice_beta_offset_div2,
        })
    }

    /// Calculate the slice QP.
    pub fn qp(&self, pps: &PictureParameterSet) -> i8 {
        26 + pps.pic_init_qp_minus26 + self.slice_qp_delta
    }
}

fn skip_ref_pic_list_modification(reader: &mut BitReader<'_>, slice_type: SliceType) -> Result<()> {
    // ref_pic_list_modification_flag_l0
    if slice_type != SliceType::I && slice_type != SliceType::Si && reader.read_bit()? {
        loop {
            let modification_of_pic_nums_idc = reader.read_ue()?;
            if modification_of_pic_nums_idc == 3 {
                break;
            }
            let _ = reader.read_ue()?;
        }
    }

    // ref_pic_list_modification_flag_l1
    if slice_type == SliceType::B && reader.read_bit()? {
        loop {
            let modification_of_pic_nums_idc = reader.read_ue()?;
            if modification_of_pic_nums_idc == 3 {
                break;
            }
            let _ = reader.read_ue()?;
        }
    }

    Ok(())
}

fn skip_pred_weight_table(
    reader: &mut BitReader<'_>,
    sps: &SequenceParameterSet,
    slice_type: SliceType,
    num_ref_l0: u8,
    num_ref_l1: u8,
) -> Result<()> {
    let _ = reader.read_ue()?; // luma_log2_weight_denom

    if sps.chroma_format_idc != 0 {
        let _ = reader.read_ue()?; // chroma_log2_weight_denom
    }

    for _ in 0..=num_ref_l0 {
        if reader.read_bit()? { // luma_weight_flag
            let _ = reader.read_se()?;
            let _ = reader.read_se()?;
        }
        if sps.chroma_format_idc != 0 && reader.read_bit()? { // chroma_weight_flag
            for _ in 0..2 {
                let _ = reader.read_se()?;
                let _ = reader.read_se()?;
            }
        }
    }

    if slice_type == SliceType::B {
        for _ in 0..=num_ref_l1 {
            if reader.read_bit()? {
                let _ = reader.read_se()?;
                let _ = reader.read_se()?;
            }
            if sps.chroma_format_idc != 0 && reader.read_bit()? {
                for _ in 0..2 {
                    let _ = reader.read_se()?;
                    let _ = reader.read_se()?;
                }
            }
        }
    }

    Ok(())
}

fn skip_dec_ref_pic_marking(reader: &mut BitReader<'_>, nal_type: NalUnitType) -> Result<()> {
    if nal_type == NalUnitType::IdrSlice {
        let _ = reader.read_bit()?; // no_output_of_prior_pics_flag
        let _ = reader.read_bit()?; // long_term_reference_flag
    } else if reader.read_bit()? { // adaptive_ref_pic_marking_mode_flag
        loop {
            let mmco = reader.read_ue()?;
            if mmco == 0 {
                break;
            }
            match mmco {
                1 | 3 => { let _ = reader.read_ue()?; }
                2 => { let _ = reader.read_ue()?; }
                4 | 5 => {}
                6 => { let _ = reader.read_ue()?; let _ = reader.read_ue()?; }
                _ => break,
            }
        }
    }
    Ok(())
}
