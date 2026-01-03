#![no_main]

//! Fuzz target for H.264 slice header parsing.
//!
//! Tests slice header parsing with pre-generated SPS/PPS context.
//! Uses arbitrary data for both the slice header and the parameter sets.

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use transcode_core::bitstream::BitReader;

#[derive(Arbitrary, Debug)]
struct SliceInput {
    /// SPS data for context
    sps_data: Vec<u8>,
    /// PPS data for context
    pps_data: Vec<u8>,
    /// Slice header data
    slice_data: Vec<u8>,
    /// NAL unit type (IDR or non-IDR)
    is_idr: bool,
}

fuzz_target!(|input: SliceInput| {
    // Limit input sizes
    if input.sps_data.len() > 1024
        || input.pps_data.len() > 1024
        || input.slice_data.len() > 10 * 1024
    {
        return;
    }

    // Try to parse SPS first - if it fails, use default values
    let sps = match transcode_codecs::video::h264::SequenceParameterSet::parse(&input.sps_data) {
        Ok(s) => s,
        Err(_) => return, // Need valid SPS for slice parsing
    };

    // Try to parse PPS
    let pps = match transcode_codecs::video::h264::PictureParameterSet::parse(&input.pps_data) {
        Ok(p) => p,
        Err(_) => return, // Need valid PPS for slice parsing
    };

    // Determine NAL type
    let nal_type = if input.is_idr {
        transcode_codecs::video::h264::NalUnitType::IdrSlice
    } else {
        transcode_codecs::video::h264::NalUnitType::Slice
    };

    // Create BitReader for slice data
    let mut reader = BitReader::new(&input.slice_data);

    // Try to parse slice header - should never panic
    let result = transcode_codecs::video::h264::SliceHeader::parse(
        &mut reader,
        nal_type,
        &sps,
        &pps,
    );

    // If parsing succeeds, test derived values
    if let Ok(slice) = result {
        // These should never panic
        let _ = slice.qp(&pps);

        // Access all fields to ensure no panics
        let _ = slice.first_mb_in_slice;
        let _ = slice.slice_type;
        let _ = slice.pps_id;
        let _ = slice.frame_num;
        let _ = slice.field_pic_flag;
        let _ = slice.idr_pic_id;
        let _ = slice.pic_order_cnt_lsb;
        let _ = slice.num_ref_idx_l0_active_minus1;
        let _ = slice.num_ref_idx_l1_active_minus1;
        let _ = slice.slice_qp_delta;
        let _ = slice.disable_deblocking_filter_idc;

        // Test slice type methods
        let _ = slice.slice_type.is_intra();
        let _ = slice.slice_type.is_bidirectional();
    }
});
