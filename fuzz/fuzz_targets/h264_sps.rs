#![no_main]

//! Fuzz target for H.264 Sequence Parameter Set (SPS) parsing.
//!
//! Tests SPS parsing with arbitrary RBSP data, including edge cases
//! for various profiles and extended syntax elements.

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;

#[derive(Arbitrary, Debug)]
struct SpsInput {
    /// Raw SPS data (RBSP without NAL header)
    data: Vec<u8>,
    /// Whether to parse via BitReader or directly
    parse_mode: ParseMode,
}

#[derive(Arbitrary, Debug)]
enum ParseMode {
    /// Parse using SPS::parse()
    Direct,
    /// Parse from full NAL unit including header
    WithNalHeader,
}

fuzz_target!(|input: SpsInput| {
    // Limit input size to prevent DoS from pathological inputs
    if input.data.len() > 10 * 1024 {
        return;
    }

    match input.parse_mode {
        ParseMode::Direct => {
            // Parse SPS directly from RBSP data
            let _ = transcode_codecs::video::h264::SequenceParameterSet::parse(&input.data);
        }
        ParseMode::WithNalHeader => {
            // First parse as NAL unit, then parse SPS
            if let Ok(nal) = transcode_codecs::video::h264::NalUnit::parse(&input.data) {
                if nal.nal_type == transcode_codecs::video::h264::NalUnitType::Sps {
                    let _ = transcode_codecs::video::h264::SequenceParameterSet::parse(&nal.data);
                }
            }
        }
    }

    // Also test dimension calculations if parsing succeeds
    if let Ok(sps) = transcode_codecs::video::h264::SequenceParameterSet::parse(&input.data) {
        // These should never panic even with malformed data due to saturating arithmetic
        let _ = sps.width();
        let _ = sps.height();
        let _ = sps.max_dpb_frames();
        let _ = sps.profile();
        let _ = sps.level();
        let _ = sps.frame_rate();

        // Access additional fields
        let _ = sps.profile_idc;
        let _ = sps.level_idc;
        let _ = sps.chroma_format_idc;
        let _ = sps.bit_depth_luma;
        let _ = sps.bit_depth_chroma;
        let _ = sps.log2_max_frame_num;
        let _ = sps.pic_order_cnt_type;
        let _ = sps.frame_mbs_only_flag;
    }
});
