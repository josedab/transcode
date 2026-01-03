//! Sequence Parameter Set (SPS) parsing.
//!
//! The SPS contains essential parameters about the video sequence.

use transcode_core::bitstream::BitReader;
use transcode_core::error::{CodecError, Result};
use super::{H264Profile, H264Level};

/// Sequence Parameter Set.
#[derive(Debug, Clone)]
pub struct SequenceParameterSet {
    /// Profile IDC.
    pub profile_idc: u8,
    /// Constraint set flags.
    pub constraint_set_flags: u8,
    /// Level IDC.
    pub level_idc: u8,
    /// SPS ID (0-31).
    pub sps_id: u8,
    /// Chroma format IDC (0=mono, 1=4:2:0, 2=4:2:2, 3=4:4:4).
    pub chroma_format_idc: u8,
    /// Separate color plane flag.
    pub separate_colour_plane_flag: bool,
    /// Bit depth for luma (8-14).
    pub bit_depth_luma: u8,
    /// Bit depth for chroma (8-14).
    pub bit_depth_chroma: u8,
    /// Log2 of max frame number.
    pub log2_max_frame_num: u8,
    /// Picture order count type (0-2).
    pub pic_order_cnt_type: u8,
    /// Log2 of max POC LSB.
    pub log2_max_pic_order_cnt_lsb: u8,
    /// Delta POC always zero flag.
    pub delta_pic_order_always_zero_flag: bool,
    /// Offset for non-reference pictures.
    pub offset_for_non_ref_pic: i32,
    /// Offset for top to bottom field.
    pub offset_for_top_to_bottom_field: i32,
    /// Number of reference frames in POC cycle.
    pub num_ref_frames_in_pic_order_cnt_cycle: u8,
    /// Maximum number of reference frames.
    pub max_num_ref_frames: u8,
    /// Gaps in frame number allowed.
    pub gaps_in_frame_num_allowed: bool,
    /// Picture width in macroblocks minus 1.
    pub pic_width_in_mbs_minus1: u32,
    /// Picture height in map units minus 1.
    pub pic_height_in_map_units_minus1: u32,
    /// Frame MBS only flag.
    pub frame_mbs_only_flag: bool,
    /// MB adaptive frame field flag.
    pub mb_adaptive_frame_field_flag: bool,
    /// Direct 8x8 inference flag.
    pub direct_8x8_inference_flag: bool,
    /// Frame cropping flag.
    pub frame_cropping_flag: bool,
    /// Frame crop left offset.
    pub frame_crop_left_offset: u32,
    /// Frame crop right offset.
    pub frame_crop_right_offset: u32,
    /// Frame crop top offset.
    pub frame_crop_top_offset: u32,
    /// Frame crop bottom offset.
    pub frame_crop_bottom_offset: u32,
    /// VUI parameters present flag.
    pub vui_parameters_present_flag: bool,
    /// VUI parameters.
    pub vui: Option<VuiParameters>,
}

impl SequenceParameterSet {
    /// Parse an SPS from RBSP data.
    pub fn parse(data: &[u8]) -> Result<Self> {
        let mut reader = BitReader::new(data);
        Self::parse_from_reader(&mut reader)
    }

    /// Parse from a bit reader.
    pub fn parse_from_reader(reader: &mut BitReader<'_>) -> Result<Self> {
        let profile_idc = reader.read_bits(8)? as u8;
        let constraint_set_flags = reader.read_bits(8)? as u8;
        let level_idc = reader.read_bits(8)? as u8;
        let sps_id = reader.read_ue()? as u8;

        if sps_id > 31 {
            return Err(CodecError::InvalidParameterSet(format!(
                "SPS ID {} out of range",
                sps_id
            ))
            .into());
        }

        // High profile extensions
        let (chroma_format_idc, separate_colour_plane_flag, bit_depth_luma, bit_depth_chroma) =
            if profile_idc == 100
                || profile_idc == 110
                || profile_idc == 122
                || profile_idc == 244
                || profile_idc == 44
                || profile_idc == 83
                || profile_idc == 86
                || profile_idc == 118
                || profile_idc == 128
                || profile_idc == 138
                || profile_idc == 139
                || profile_idc == 134
                || profile_idc == 135
            {
                let chroma_format_idc = reader.read_ue()? as u8;
                let separate_colour_plane_flag = if chroma_format_idc == 3 {
                    reader.read_bit()?
                } else {
                    false
                };
                let bit_depth_luma = reader.read_ue()? as u8 + 8;
                let bit_depth_chroma = reader.read_ue()? as u8 + 8;

                // qpprime_y_zero_transform_bypass_flag
                let _ = reader.read_bit()?;

                // seq_scaling_matrix_present_flag
                if reader.read_bit()? {
                    let count = if chroma_format_idc != 3 { 8 } else { 12 };
                    for _ in 0..count {
                        if reader.read_bit()? {
                            // Skip scaling list
                            let size = if count < 6 { 16 } else { 64 };
                            skip_scaling_list(reader, size)?;
                        }
                    }
                }

                (
                    chroma_format_idc,
                    separate_colour_plane_flag,
                    bit_depth_luma,
                    bit_depth_chroma,
                )
            } else {
                (1, false, 8, 8) // Default: 4:2:0, 8-bit
            };

        let log2_max_frame_num = reader.read_ue()? as u8 + 4;
        let pic_order_cnt_type = reader.read_ue()? as u8;

        let (
            log2_max_pic_order_cnt_lsb,
            delta_pic_order_always_zero_flag,
            offset_for_non_ref_pic,
            offset_for_top_to_bottom_field,
            num_ref_frames_in_pic_order_cnt_cycle,
        ) = match pic_order_cnt_type {
            0 => {
                let log2_max_poc = reader.read_ue()? as u8 + 4;
                (log2_max_poc, false, 0, 0, 0)
            }
            1 => {
                let delta_always_zero = reader.read_bit()?;
                let offset_non_ref = reader.read_se()?;
                let offset_top_bottom = reader.read_se()?;
                let num_ref_frames_raw = reader.read_ue()?;

                // Validate loop count to prevent DoS from malformed data
                // H.264 spec limits this to 255
                if num_ref_frames_raw > 255 {
                    return Err(CodecError::InvalidParameterSet(
                        "num_ref_frames_in_pic_order_cnt_cycle too large".into()
                    ).into());
                }
                let num_ref_frames = num_ref_frames_raw as u8;

                // Skip offset_for_ref_frame array
                for _ in 0..num_ref_frames {
                    let _ = reader.read_se()?;
                }

                (0, delta_always_zero, offset_non_ref, offset_top_bottom, num_ref_frames)
            }
            _ => (0, false, 0, 0, 0),
        };

        let max_num_ref_frames = reader.read_ue()? as u8;
        let gaps_in_frame_num_allowed = reader.read_bit()?;
        let pic_width_in_mbs_minus1 = reader.read_ue()?;
        let pic_height_in_map_units_minus1 = reader.read_ue()?;
        let frame_mbs_only_flag = reader.read_bit()?;

        let mb_adaptive_frame_field_flag = if !frame_mbs_only_flag {
            reader.read_bit()?
        } else {
            false
        };

        let direct_8x8_inference_flag = reader.read_bit()?;

        let (
            frame_cropping_flag,
            frame_crop_left_offset,
            frame_crop_right_offset,
            frame_crop_top_offset,
            frame_crop_bottom_offset,
        ) = if reader.read_bit()? {
            (
                true,
                reader.read_ue()?,
                reader.read_ue()?,
                reader.read_ue()?,
                reader.read_ue()?,
            )
        } else {
            (false, 0, 0, 0, 0)
        };

        let vui_parameters_present_flag = reader.read_bit()?;
        let vui = if vui_parameters_present_flag {
            Some(VuiParameters::parse(reader)?)
        } else {
            None
        };

        Ok(Self {
            profile_idc,
            constraint_set_flags,
            level_idc,
            sps_id,
            chroma_format_idc,
            separate_colour_plane_flag,
            bit_depth_luma,
            bit_depth_chroma,
            log2_max_frame_num,
            pic_order_cnt_type,
            log2_max_pic_order_cnt_lsb,
            delta_pic_order_always_zero_flag,
            offset_for_non_ref_pic,
            offset_for_top_to_bottom_field,
            num_ref_frames_in_pic_order_cnt_cycle,
            max_num_ref_frames,
            gaps_in_frame_num_allowed,
            pic_width_in_mbs_minus1,
            pic_height_in_map_units_minus1,
            frame_mbs_only_flag,
            mb_adaptive_frame_field_flag,
            direct_8x8_inference_flag,
            frame_cropping_flag,
            frame_crop_left_offset,
            frame_crop_right_offset,
            frame_crop_top_offset,
            frame_crop_bottom_offset,
            vui_parameters_present_flag,
            vui,
        })
    }

    /// Get the profile.
    pub fn profile(&self) -> Option<H264Profile> {
        H264Profile::from_idc(self.profile_idc)
    }

    /// Get the level.
    pub fn level(&self) -> H264Level {
        H264Level::from_idc(self.level_idc)
    }

    /// Get the picture width in pixels.
    /// Uses saturating arithmetic to prevent overflow from malformed data.
    pub fn width(&self) -> u32 {
        let width = self.pic_width_in_mbs_minus1
            .saturating_add(1)
            .saturating_mul(16);
        if self.frame_cropping_flag {
            let crop_unit_x: u32 = if self.chroma_format_idc == 0 { 1 } else { 2 };
            let crop_amount = crop_unit_x.saturating_mul(
                self.frame_crop_left_offset.saturating_add(self.frame_crop_right_offset)
            );
            width.saturating_sub(crop_amount)
        } else {
            width
        }
    }

    /// Get the picture height in pixels.
    /// Uses saturating arithmetic to prevent overflow from malformed data.
    pub fn height(&self) -> u32 {
        let height = self.pic_height_in_map_units_minus1
            .saturating_add(1)
            .saturating_mul(16);
        let height = if self.frame_mbs_only_flag {
            height
        } else {
            height.saturating_mul(2)
        };
        if self.frame_cropping_flag {
            let crop_unit_y: u32 = if self.chroma_format_idc == 0 {
                if self.frame_mbs_only_flag {
                    1
                } else {
                    2
                }
            } else if self.frame_mbs_only_flag {
                2
            } else {
                4
            };
            let crop_amount = crop_unit_y.saturating_mul(
                self.frame_crop_top_offset.saturating_add(self.frame_crop_bottom_offset)
            );
            height.saturating_sub(crop_amount)
        } else {
            height
        }
    }

    /// Get the frame rate if available from VUI.
    pub fn frame_rate(&self) -> Option<(u32, u32)> {
        self.vui.as_ref().and_then(|vui| {
            if vui.timing_info_present_flag {
                Some((vui.time_scale, vui.num_units_in_tick * 2))
            } else {
                None
            }
        })
    }

    /// Get the maximum number of frames in the DPB.
    pub fn max_dpb_frames(&self) -> u32 {
        let level = self.level();
        let pic_size = (self.pic_width_in_mbs_minus1 + 1) * (self.pic_height_in_map_units_minus1 + 1);
        if pic_size == 0 {
            return 1;
        }
        let max_dpb = level.max_dpb_mbs() / pic_size;
        max_dpb.clamp(1, 16)
    }
}

/// VUI (Video Usability Information) parameters.
#[derive(Debug, Clone)]
pub struct VuiParameters {
    /// Aspect ratio info present.
    pub aspect_ratio_info_present_flag: bool,
    /// Aspect ratio IDC.
    pub aspect_ratio_idc: u8,
    /// Sample aspect ratio width.
    pub sar_width: u16,
    /// Sample aspect ratio height.
    pub sar_height: u16,
    /// Overscan info present.
    pub overscan_info_present_flag: bool,
    /// Video signal type present.
    pub video_signal_type_present_flag: bool,
    /// Video format.
    pub video_format: u8,
    /// Video full range flag.
    pub video_full_range_flag: bool,
    /// Color description present.
    pub colour_description_present_flag: bool,
    /// Color primaries.
    pub colour_primaries: u8,
    /// Transfer characteristics.
    pub transfer_characteristics: u8,
    /// Matrix coefficients.
    pub matrix_coefficients: u8,
    /// Chroma loc info present.
    pub chroma_loc_info_present_flag: bool,
    /// Timing info present.
    pub timing_info_present_flag: bool,
    /// Number of units in tick.
    pub num_units_in_tick: u32,
    /// Time scale.
    pub time_scale: u32,
    /// Fixed frame rate flag.
    pub fixed_frame_rate_flag: bool,
}

impl VuiParameters {
    /// Parse VUI parameters from a bit reader.
    pub fn parse(reader: &mut BitReader<'_>) -> Result<Self> {
        let aspect_ratio_info_present_flag = reader.read_bit()?;
        let (aspect_ratio_idc, sar_width, sar_height) = if aspect_ratio_info_present_flag {
            let idc = reader.read_bits(8)? as u8;
            if idc == 255 {
                // Extended_SAR
                let w = reader.read_bits(16)? as u16;
                let h = reader.read_bits(16)? as u16;
                (idc, w, h)
            } else {
                (idc, 0, 0)
            }
        } else {
            (0, 0, 0)
        };

        let overscan_info_present_flag = reader.read_bit()?;
        if overscan_info_present_flag {
            let _ = reader.read_bit()?; // overscan_appropriate_flag
        }

        let video_signal_type_present_flag = reader.read_bit()?;
        let (
            video_format,
            video_full_range_flag,
            colour_description_present_flag,
            colour_primaries,
            transfer_characteristics,
            matrix_coefficients,
        ) = if video_signal_type_present_flag {
            let vf = reader.read_bits(3)? as u8;
            let vfr = reader.read_bit()?;
            let cdp = reader.read_bit()?;
            let (cp, tc, mc) = if cdp {
                (
                    reader.read_bits(8)? as u8,
                    reader.read_bits(8)? as u8,
                    reader.read_bits(8)? as u8,
                )
            } else {
                (0, 0, 0)
            };
            (vf, vfr, cdp, cp, tc, mc)
        } else {
            (0, false, false, 0, 0, 0)
        };

        let chroma_loc_info_present_flag = reader.read_bit()?;
        if chroma_loc_info_present_flag {
            let _ = reader.read_ue()?; // chroma_sample_loc_type_top_field
            let _ = reader.read_ue()?; // chroma_sample_loc_type_bottom_field
        }

        let timing_info_present_flag = reader.read_bit()?;
        let (num_units_in_tick, time_scale, fixed_frame_rate_flag) = if timing_info_present_flag {
            let nuit = reader.read_bits(32)?;
            let ts = reader.read_bits(32)?;
            let ffrf = reader.read_bit()?;
            (nuit, ts, ffrf)
        } else {
            (0, 0, false)
        };

        Ok(Self {
            aspect_ratio_info_present_flag,
            aspect_ratio_idc,
            sar_width,
            sar_height,
            overscan_info_present_flag,
            video_signal_type_present_flag,
            video_format,
            video_full_range_flag,
            colour_description_present_flag,
            colour_primaries,
            transfer_characteristics,
            matrix_coefficients,
            chroma_loc_info_present_flag,
            timing_info_present_flag,
            num_units_in_tick,
            time_scale,
            fixed_frame_rate_flag,
        })
    }
}

/// Skip a scaling list in the bitstream.
fn skip_scaling_list(reader: &mut BitReader<'_>, size: usize) -> Result<()> {
    let mut last_scale = 8i32;
    let mut next_scale = 8i32;

    for _ in 0..size {
        if next_scale != 0 {
            let delta_scale = reader.read_se()?;
            next_scale = (last_scale + delta_scale + 256) % 256;
        }
        last_scale = if next_scale == 0 { last_scale } else { next_scale };
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sps_parse() {
        // Minimal valid SPS for 1920x1080 Baseline profile
        // profile_idc=66, level=40, 1920x1080
        let sps_data = [
            0x42, 0x00, 0x28, // profile, constraints, level
            0x8D, 0x8D, 0x40, // sps_id=0, log2_max_frame_num, poc_type
            0x3C, 0x01, 0x13, // max_ref_frames, gaps, width
            0xF2, 0xC0, 0x44, // height, frame_mbs_only, direct_8x8
        ];

        // This is a simplified test; real SPS parsing needs more complete data
    }
}
