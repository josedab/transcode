#![no_main]

//! Fuzz target for H.264 NAL unit parsing.
//!
//! Tests both Annex B and AVCC format NAL parsing with arbitrary input.

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;

#[derive(Arbitrary, Debug)]
struct NalInput {
    data: Vec<u8>,
    format: NalFormat,
}

#[derive(Arbitrary, Debug)]
enum NalFormat {
    /// Annex B format (start codes)
    AnnexB,
    /// AVCC format with 1-byte length prefix
    Avcc1,
    /// AVCC format with 2-byte length prefix
    Avcc2,
    /// AVCC format with 4-byte length prefix
    Avcc4,
    /// Raw NAL unit (no start code or length prefix)
    Raw,
}

fuzz_target!(|input: NalInput| {
    // Limit input size to prevent excessive memory allocation
    if input.data.len() > 1024 * 1024 {
        return;
    }

    match input.format {
        NalFormat::AnnexB => {
            // Parse as Annex B stream - iterator approach
            let iterator = transcode_codecs::video::h264::NalIterator::new(&input.data);
            for nal_result in iterator {
                // Access result but don't care about errors
                if let Ok(nal) = nal_result {
                    // Access various fields - should never panic
                    let _ = nal.nal_type.is_vcl();
                    let _ = nal.nal_type.starts_access_unit();
                    let _ = nal.nal_type.is_reference();
                    let _ = nal.is_idr();
                    let _ = nal.is_slice();
                    let _ = nal.bitstream();
                }
            }
        }
        NalFormat::Avcc1 => {
            // Parse with 1-byte length prefix
            let nals = transcode_codecs::video::h264::parse_avcc(&input.data, 1);
            for nal_result in nals {
                if let Ok(nal) = nal_result {
                    let _ = nal.nal_type;
                    let _ = nal.nal_ref_idc;
                }
            }
        }
        NalFormat::Avcc2 => {
            // Parse with 2-byte length prefix
            let nals = transcode_codecs::video::h264::parse_avcc(&input.data, 2);
            for nal_result in nals {
                if let Ok(nal) = nal_result {
                    let _ = nal.nal_type;
                }
            }
        }
        NalFormat::Avcc4 => {
            // Parse with 4-byte length prefix
            let nals = transcode_codecs::video::h264::parse_avcc(&input.data, 4);
            for nal_result in nals {
                if let Ok(nal) = nal_result {
                    let _ = nal.nal_type;
                }
            }
        }
        NalFormat::Raw => {
            // Parse single raw NAL unit
            if let Ok(nal) = transcode_codecs::video::h264::NalUnit::parse(&input.data) {
                // Access all fields
                let _ = nal.nal_type.to_u8();
                let _ = nal.nal_ref_idc;
                let _ = nal.data.len();
            }
        }
    }
});
