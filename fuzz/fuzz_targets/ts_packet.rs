#![no_main]

//! Fuzz target for MPEG-TS packet parsing.
//!
//! Tests TS packet header parsing, adaptation field handling, and PCR decoding
//! with arbitrary input to find parsing bugs and panics.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct TsPacketInput {
    data: Vec<u8>,
    operation: TsOperation,
}

#[derive(Arbitrary, Debug)]
enum TsOperation {
    /// Parse a complete TS packet (188 bytes)
    ParsePacket,
    /// Parse just the TS header (4 bytes)
    ParseHeader,
    /// Parse adaptation field
    ParseAdaptationField,
    /// Parse PCR value
    ParsePcr,
    /// Round-trip header write/read
    HeaderRoundtrip {
        pid: u16,
        continuity_counter: u8,
        payload_unit_start: bool,
    },
    /// Round-trip PCR write/read
    PcrRoundtrip { base: u64, extension: u16 },
    /// Parse multiple consecutive packets
    ParseMultiplePackets,
}

fuzz_target!(|input: TsPacketInput| {
    // Limit input size
    if input.data.len() > 64 * 1024 {
        return;
    }

    match input.operation {
        TsOperation::ParsePacket => {
            // Need at least 188 bytes for a TS packet
            if input.data.len() >= transcode_ts::TS_PACKET_SIZE {
                let mut packet_data = [0u8; transcode_ts::TS_PACKET_SIZE];
                packet_data.copy_from_slice(&input.data[..transcode_ts::TS_PACKET_SIZE]);

                // Should not panic on any input
                if let Ok(packet) = transcode_ts::TsPacket::new(packet_data) {
                    // Access various fields - should never panic
                    let _ = packet.pid();
                    let _ = packet.is_null();
                    let _ = packet.header();
                    let _ = packet.payload();
                    let _ = packet.adaptation_field();
                }
            }
        }

        TsOperation::ParseHeader => {
            if input.data.len() >= 4 {
                // Should not panic on any input
                if let Ok(header) = transcode_ts::TsHeader::parse(&input.data[..4]) {
                    // Access all header fields
                    let _ = header.pid;
                    let _ = header.payload_unit_start;
                    let _ = header.transport_priority;
                    let _ = header.continuity_counter;
                    let _ = header.adaptation_field_control;
                    let _ = header.scrambling_control;
                    let _ = header.transport_error;
                }
            }
        }

        TsOperation::ParseAdaptationField => {
            if !input.data.is_empty() {
                // Should not panic on any input
                if let Ok(af) = transcode_ts::AdaptationField::parse(&input.data) {
                    // Access all adaptation field properties
                    let _ = af.length;
                    let _ = af.discontinuity;
                    let _ = af.random_access;
                    let _ = af.es_priority;
                    let _ = af.pcr_flag;
                    let _ = af.opcr_flag;
                    let _ = af.splicing_point_flag;
                    let _ = af.pcr;
                    let _ = af.opcr;
                    let _ = af.splice_countdown;
                    let _ = af.stuffing_bytes;
                }
            }
        }

        TsOperation::ParsePcr => {
            if input.data.len() >= 6 {
                // Should not panic on any input
                if let Ok(pcr) = transcode_ts::Pcr::parse(&input.data[..6]) {
                    // Access PCR values
                    let _ = pcr.base;
                    let _ = pcr.extension;
                    let _ = pcr.to_27mhz();
                    let _ = pcr.to_seconds();
                }
            }
        }

        TsOperation::HeaderRoundtrip {
            pid,
            continuity_counter,
            payload_unit_start,
        } => {
            // Constrain to valid ranges
            let pid = pid & transcode_ts::PID_MAX;
            let continuity_counter = continuity_counter & 0x0F;

            let mut header = transcode_ts::TsHeader::new(pid);
            header.continuity_counter = continuity_counter;
            header.payload_unit_start = payload_unit_start;

            let mut buffer = [0u8; 4];
            if header.write(&mut buffer).is_ok() {
                if let Ok(parsed) = transcode_ts::TsHeader::parse(&buffer) {
                    assert_eq!(pid, parsed.pid, "PID mismatch");
                    assert_eq!(continuity_counter, parsed.continuity_counter, "CC mismatch");
                    assert_eq!(
                        payload_unit_start, parsed.payload_unit_start,
                        "PUSI mismatch"
                    );
                }
            }
        }

        TsOperation::PcrRoundtrip { base, extension } => {
            // Constrain to valid ranges (33-bit base, 9-bit extension)
            let base = base & 0x1_FFFF_FFFF;
            let extension = extension & 0x1FF;

            let pcr = transcode_ts::Pcr { base, extension };

            let mut buffer = [0u8; 6];
            if pcr.write(&mut buffer).is_ok() {
                if let Ok(parsed) = transcode_ts::Pcr::parse(&buffer) {
                    assert_eq!(base, parsed.base, "PCR base mismatch");
                    assert_eq!(extension, parsed.extension, "PCR extension mismatch");
                }
            }
        }

        TsOperation::ParseMultiplePackets => {
            // Parse consecutive TS packets
            let mut offset = 0;
            while offset + transcode_ts::TS_PACKET_SIZE <= input.data.len() {
                let mut packet_data = [0u8; transcode_ts::TS_PACKET_SIZE];
                packet_data.copy_from_slice(&input.data[offset..offset + transcode_ts::TS_PACKET_SIZE]);

                // Try to parse - may fail if sync byte is wrong
                let _ = transcode_ts::TsPacket::new(packet_data);

                offset += transcode_ts::TS_PACKET_SIZE;
            }
        }
    }
});
