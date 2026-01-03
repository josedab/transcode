#![no_main]

//! Fuzz target for H.264 Picture Parameter Set (PPS) parsing.
//!
//! Tests PPS parsing with arbitrary RBSP data, including slice groups,
//! weighted prediction tables, and High profile extensions.

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;

#[derive(Arbitrary, Debug)]
struct PpsInput {
    /// Raw PPS data (RBSP without NAL header)
    data: Vec<u8>,
    /// Whether to parse via BitReader or directly
    parse_mode: ParseMode,
}

#[derive(Arbitrary, Debug)]
enum ParseMode {
    /// Parse using PPS::parse()
    Direct,
    /// Parse from full NAL unit including header
    WithNalHeader,
}

fuzz_target!(|input: PpsInput| {
    // Limit input size to prevent DoS
    if input.data.len() > 10 * 1024 {
        return;
    }

    match input.parse_mode {
        ParseMode::Direct => {
            // Parse PPS directly from RBSP data
            let _ = transcode_codecs::video::h264::PictureParameterSet::parse(&input.data);
        }
        ParseMode::WithNalHeader => {
            // First parse as NAL unit, then parse PPS
            if let Ok(nal) = transcode_codecs::video::h264::NalUnit::parse(&input.data) {
                if nal.nal_type == transcode_codecs::video::h264::NalUnitType::Pps {
                    let _ = transcode_codecs::video::h264::PictureParameterSet::parse(&nal.data);
                }
            }
        }
    }

    // Test derived values if parsing succeeds
    if let Ok(pps) = transcode_codecs::video::h264::PictureParameterSet::parse(&input.data) {
        // These should never panic
        let _ = pps.uses_cabac();

        // Access all fields
        let _ = pps.pps_id;
        let _ = pps.sps_id;
        let _ = pps.entropy_coding_mode_flag;
        let _ = pps.bottom_field_pic_order_in_frame_present_flag;
        let _ = pps.num_slice_groups_minus1;
        let _ = pps.num_ref_idx_l0_default_active_minus1;
        let _ = pps.num_ref_idx_l1_default_active_minus1;
        let _ = pps.weighted_pred_flag;
        let _ = pps.weighted_bipred_idc;
        let _ = pps.pic_init_qp_minus26;
        let _ = pps.deblocking_filter_control_present_flag;
        let _ = pps.transform_8x8_mode_flag;
    }
});
