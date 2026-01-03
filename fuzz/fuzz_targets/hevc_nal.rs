#![no_main]

//! Fuzz target for HEVC/H.265 NAL unit parsing.
//!
//! Tests HEVC NAL header, VPS, SPS, PPS, and slice header parsing.

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;

#[derive(Arbitrary, Debug)]
struct HevcNalInput {
    data: Vec<u8>,
    parse_type: HevcParseType,
}

#[derive(Arbitrary, Debug)]
enum HevcParseType {
    /// Parse NAL unit header only
    NalHeader,
    /// Parse as VPS
    Vps,
    /// Parse as SPS
    Sps,
    /// Parse as PPS
    Pps,
    /// Parse Annex B stream
    AnnexB,
}

fuzz_target!(|input: HevcNalInput| {
    // Limit input size
    if input.data.len() > 1024 * 1024 {
        return;
    }

    match input.parse_type {
        HevcParseType::NalHeader => {
            // Parse just the NAL header (2 bytes)
            if let Ok(header) = transcode_hevc::nal::NalUnitHeader::parse(&input.data) {
                // Access fields - should never panic
                let _ = header.nal_unit_type.is_vcl();
                let _ = header.nal_unit_type.is_irap();
                let _ = header.nal_unit_type.is_idr();
                let _ = header.nal_unit_type.is_bla();
                let _ = header.nal_unit_type.is_cra();
                let _ = header.nal_unit_type.is_leading();
                let _ = header.nal_unit_type.is_trailing();
                let _ = header.nal_unit_type.is_reference();
                let _ = header.temporal_id();
                let _ = header.nuh_layer_id;
            }
        }
        HevcParseType::Vps => {
            // Parse as VPS
            if let Ok(vps) = transcode_hevc::nal::Vps::parse(&input.data) {
                // Test derived values - should never panic
                let _ = vps.frame_rate();
                let _ = vps.vps_video_parameter_set_id;
                let _ = vps.vps_max_sub_layers_minus1;
                let _ = vps.vps_temporal_id_nesting_flag;
                let _ = vps.profile_tier_level.profile();
                let _ = vps.profile_tier_level.tier();
                let _ = vps.profile_tier_level.level();
            }
        }
        HevcParseType::Sps => {
            // Parse as SPS
            if let Ok(sps) = transcode_hevc::nal::Sps::parse(&input.data) {
                // Test derived values - should never panic
                let _ = sps.width();
                let _ = sps.height();
                let _ = sps.bit_depth_luma();
                let _ = sps.bit_depth_chroma();
                let _ = sps.ctb_size();
                let _ = sps.min_cb_size();
                let _ = sps.pic_width_in_ctbs();
                let _ = sps.pic_height_in_ctbs();
                let _ = sps.log2_min_cb_size();
                let _ = sps.log2_ctb_size();
                let _ = sps.min_tb_size();
                let _ = sps.max_tb_size();
                let _ = sps.sub_width_c();
                let _ = sps.sub_height_c();

                // Access raw fields
                let _ = sps.chroma_format_idc;
                let _ = sps.pic_width_in_luma_samples;
                let _ = sps.pic_height_in_luma_samples;
            }
        }
        HevcParseType::Pps => {
            // Parse as PPS
            if let Ok(pps) = transcode_hevc::nal::Pps::parse(&input.data) {
                // Test derived values
                let _ = pps.init_qp();
                let _ = pps.pps_pic_parameter_set_id;
                let _ = pps.pps_seq_parameter_set_id;
                let _ = pps.tiles_enabled_flag;
                let _ = pps.entropy_coding_sync_enabled_flag;
            }
        }
        HevcParseType::AnnexB => {
            // Parse Annex B stream
            let units = transcode_hevc::nal::parse_annexb_nal_units(&input.data);
            for (header, rbsp) in units {
                // Try to parse based on NAL type
                match header.nal_unit_type {
                    transcode_hevc::nal::NalUnitType::VpsNut => {
                        let _ = transcode_hevc::nal::Vps::parse(&rbsp);
                    }
                    transcode_hevc::nal::NalUnitType::SpsNut => {
                        let _ = transcode_hevc::nal::Sps::parse(&rbsp);
                    }
                    transcode_hevc::nal::NalUnitType::PpsNut => {
                        let _ = transcode_hevc::nal::Pps::parse(&rbsp);
                    }
                    _ => {
                        // Just access type info for other NAL types
                        let _ = header.nal_unit_type.is_vcl();
                    }
                }
            }
        }
    }
});
