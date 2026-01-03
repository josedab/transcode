#![no_main]

//! Fuzz target for MPEG-TS PES (Packetized Elementary Stream) parsing.
//!
//! Tests PES header parsing, timestamp decoding, and packet assembly
//! with arbitrary input to find parsing bugs and panics.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct PesInput {
    data: Vec<u8>,
    operation: PesOperation,
}

#[derive(Arbitrary, Debug)]
enum PesOperation {
    /// Parse a PES header
    ParseHeader,
    /// Parse PES flags
    ParseFlags,
    /// Parse PES timestamp (PTS or DTS)
    ParseTimestamp,
    /// Round-trip timestamp encode/decode
    TimestampRoundtrip { value: u64 },
    /// Assemble PES from multiple fragments
    AssembleFragments,
    /// Build a PES packet
    BuildPacket {
        stream_id: u8,
        pts_value: u64,
        dts_value: u64,
    },
    /// Parse stream ID
    CheckStreamId,
}

fuzz_target!(|input: PesInput| {
    // Limit input size
    if input.data.len() > 64 * 1024 {
        return;
    }

    match input.operation {
        PesOperation::ParseHeader => {
            // Need at least 6 bytes for minimal PES header
            if input.data.len() >= 6 {
                // Should not panic on any input
                if let Ok(header) = transcode_ts::PesHeader::parse(&input.data) {
                    // Access all header fields
                    let _ = header.stream_id;
                    let _ = header.packet_length;
                    let _ = header.flags;
                    let _ = header.pts;
                    let _ = header.dts;
                    let _ = header.header_size;
                    let _ = header.payload_offset();
                    let _ = header.is_video();
                    let _ = header.is_audio();
                }
            }
        }

        PesOperation::ParseFlags => {
            if input.data.len() >= 3 {
                // Should not panic on any input - needs 3 bytes for PES flags
                if let Ok(flags) = transcode_ts::PesFlags::parse(&input.data[..3]) {
                    // Access all flag fields
                    let _ = flags.scrambling_control;
                    let _ = flags.priority;
                    let _ = flags.data_alignment;
                    let _ = flags.copyright;
                    let _ = flags.original;
                    let _ = flags.pts_dts_flags;
                    let _ = flags.escr_flag;
                    let _ = flags.es_rate_flag;
                    let _ = flags.dsm_trick_mode_flag;
                    let _ = flags.additional_copy_info_flag;
                    let _ = flags.pes_crc_flag;
                    let _ = flags.pes_extension_flag;
                    let _ = flags.header_data_length;
                    let _ = flags.has_pts();
                    let _ = flags.has_dts();
                }
            }
        }

        PesOperation::ParseTimestamp => {
            if input.data.len() >= 5 {
                // Should not panic on any input
                if let Ok(ts) = transcode_ts::PesTimestamp::parse(&input.data[..5]) {
                    // Access timestamp value and conversions
                    let _ = ts.value;
                    let _ = ts.to_seconds();
                }
            }
        }

        PesOperation::TimestampRoundtrip { value } => {
            // Constrain to valid 33-bit timestamp range
            let value = value & 0x1_FFFF_FFFF;

            let ts = transcode_ts::PesTimestamp { value };

            let mut buffer = [0u8; 5];
            // Use PTS marker (0x20)
            if ts.write(&mut buffer, 0x20).is_ok() {
                if let Ok(parsed) = transcode_ts::PesTimestamp::parse(&buffer) {
                    assert_eq!(value, parsed.value, "Timestamp value mismatch");
                }
            }
        }

        PesOperation::AssembleFragments => {
            // Test PES assembler with fragmented data
            let mut assembler = transcode_ts::PesAssembler::new(0x101); // Use test PID

            // Split input into chunks and feed to assembler
            let chunk_size = 184; // TS packet payload size
            let mut is_first = true;
            for chunk in input.data.chunks(chunk_size) {
                // First chunk starts a new PES packet (PUSI=true)
                let _ = assembler.add(chunk, is_first);
                is_first = false;
            }

            // Access assembler state
            let _ = assembler.pid();

            // Reset should not panic
            assembler.reset();
        }

        PesOperation::BuildPacket {
            stream_id,
            pts_value,
            dts_value,
        } => {
            // Constrain timestamp values to valid 33-bit range
            let pts_value = pts_value & 0x1_FFFF_FFFF;
            let dts_value = dts_value & 0x1_FFFF_FFFF;

            let pts = transcode_ts::PesTimestamp::new(pts_value);
            let dts = transcode_ts::PesTimestamp::new(dts_value);

            // Build a PES packet
            let payload = &input.data[..input.data.len().min(1000)];

            let builder = transcode_ts::PesPacketBuilder::with_stream_id(stream_id)
                .pts(pts)
                .pts_dts(pts, dts);

            // Build header and verify it doesn't panic
            let _ = builder.header_size();
            if let Ok(header_data) = builder.build_header(payload.len()) {
                // Verify we can parse what we built
                if let Ok(header) = transcode_ts::PesHeader::parse(&header_data) {
                    assert_eq!(stream_id, header.stream_id, "Stream ID mismatch");

                    // PTS should match if we set it and stream type supports it
                    if let Some(parsed_pts) = header.pts {
                        assert_eq!(pts_value, parsed_pts.value, "PTS mismatch");
                    }
                }
            }
        }

        PesOperation::CheckStreamId => {
            if !input.data.is_empty() {
                let stream_id = input.data[0];

                // Check stream ID classification using static methods - should never panic
                let _ = transcode_ts::StreamId::is_video(stream_id);
                let _ = transcode_ts::StreamId::is_audio(stream_id);
                let _ = transcode_ts::StreamId::has_pts_dts(stream_id);
            }
        }
    }
});
