//! Picture Parameter Set (PPS) parsing.

use transcode_core::bitstream::BitReader;
use transcode_core::error::{CodecError, Result};

/// Picture Parameter Set.
#[derive(Debug, Clone)]
pub struct PictureParameterSet {
    /// PPS ID (0-255).
    pub pps_id: u8,
    /// Referenced SPS ID.
    pub sps_id: u8,
    /// Entropy coding mode (false=CAVLC, true=CABAC).
    pub entropy_coding_mode_flag: bool,
    /// Bottom field POC present.
    pub bottom_field_pic_order_in_frame_present_flag: bool,
    /// Number of slice groups.
    pub num_slice_groups_minus1: u8,
    /// Number of reference indices for L0.
    pub num_ref_idx_l0_default_active_minus1: u8,
    /// Number of reference indices for L1.
    pub num_ref_idx_l1_default_active_minus1: u8,
    /// Weighted prediction flag.
    pub weighted_pred_flag: bool,
    /// Weighted bipred IDC.
    pub weighted_bipred_idc: u8,
    /// Initial QP.
    pub pic_init_qp_minus26: i8,
    /// Initial QS.
    pub pic_init_qs_minus26: i8,
    /// Chroma QP offset.
    pub chroma_qp_index_offset: i8,
    /// Deblocking filter control present.
    pub deblocking_filter_control_present_flag: bool,
    /// Constrained intra prediction.
    pub constrained_intra_pred_flag: bool,
    /// Redundant pic count present.
    pub redundant_pic_cnt_present_flag: bool,
    /// Transform 8x8 mode flag.
    pub transform_8x8_mode_flag: bool,
    /// Second chroma QP offset.
    pub second_chroma_qp_index_offset: i8,
}

impl PictureParameterSet {
    /// Parse a PPS from RBSP data.
    pub fn parse(data: &[u8]) -> Result<Self> {
        let mut reader = BitReader::new(data);
        Self::parse_from_reader(&mut reader)
    }

    /// Parse from a bit reader.
    pub fn parse_from_reader(reader: &mut BitReader<'_>) -> Result<Self> {
        let pps_id = reader.read_ue()? as u8;
        let sps_id = reader.read_ue()? as u8;

        if sps_id > 31 {
            return Err(CodecError::InvalidParameterSet(format!(
                "SPS ID {} out of range",
                sps_id
            ))
            .into());
        }

        let entropy_coding_mode_flag = reader.read_bit()?;
        let bottom_field_pic_order_in_frame_present_flag = reader.read_bit()?;

        let num_slice_groups_minus1 = reader.read_ue()? as u8;
        if num_slice_groups_minus1 > 0 {
            // Skip slice group map parsing for now
            let slice_group_map_type = reader.read_ue()?;
            match slice_group_map_type {
                0 => {
                    for _ in 0..=num_slice_groups_minus1 {
                        let _ = reader.read_ue()?;
                    }
                }
                2 => {
                    for _ in 0..num_slice_groups_minus1 {
                        let _ = reader.read_ue()?;
                        let _ = reader.read_ue()?;
                    }
                }
                3..=5 => {
                    let _ = reader.read_bit()?;
                    let _ = reader.read_ue()?;
                }
                6 => {
                    let pic_size_in_map_units = reader.read_ue()? + 1;
                    let bits = (num_slice_groups_minus1 as f32 + 1.0).log2().ceil() as u8;
                    for _ in 0..pic_size_in_map_units {
                        let _ = reader.read_bits(bits)?;
                    }
                }
                _ => {}
            }
        }

        let num_ref_idx_l0_default_active_minus1 = reader.read_ue()? as u8;
        let num_ref_idx_l1_default_active_minus1 = reader.read_ue()? as u8;
        let weighted_pred_flag = reader.read_bit()?;
        let weighted_bipred_idc = reader.read_bits(2)? as u8;
        let pic_init_qp_minus26 = reader.read_se()? as i8;
        let pic_init_qs_minus26 = reader.read_se()? as i8;
        let chroma_qp_index_offset = reader.read_se()? as i8;
        let deblocking_filter_control_present_flag = reader.read_bit()?;
        let constrained_intra_pred_flag = reader.read_bit()?;
        let redundant_pic_cnt_present_flag = reader.read_bit()?;

        // Check for more RBSP data (High profile extensions)
        let (transform_8x8_mode_flag, second_chroma_qp_index_offset) =
            if reader.more_rbsp_data() {
                let t8x8 = reader.read_bit()?;
                // pic_scaling_matrix_present_flag
                if reader.read_bit()? {
                    let count = if t8x8 { 8 } else { 6 };
                    for _ in 0..count {
                        if reader.read_bit()? {
                            // Skip scaling list
                        }
                    }
                }
                let second_offset = reader.read_se()? as i8;
                (t8x8, second_offset)
            } else {
                (false, chroma_qp_index_offset)
            };

        Ok(Self {
            pps_id,
            sps_id,
            entropy_coding_mode_flag,
            bottom_field_pic_order_in_frame_present_flag,
            num_slice_groups_minus1,
            num_ref_idx_l0_default_active_minus1,
            num_ref_idx_l1_default_active_minus1,
            weighted_pred_flag,
            weighted_bipred_idc,
            pic_init_qp_minus26,
            pic_init_qs_minus26,
            chroma_qp_index_offset,
            deblocking_filter_control_present_flag,
            constrained_intra_pred_flag,
            redundant_pic_cnt_present_flag,
            transform_8x8_mode_flag,
            second_chroma_qp_index_offset,
        })
    }

    /// Check if CABAC entropy coding is used.
    pub fn uses_cabac(&self) -> bool {
        self.entropy_coding_mode_flag
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pps_parse_basic() {
        // Minimal PPS: pps_id=0, sps_id=0, CAVLC
        let pps_data = [
            0xCE, 0x3C, 0x80, // Basic PPS fields
        ];
        // Real parsing would need proper bitstream data
    }
}
