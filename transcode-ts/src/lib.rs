//! # Transcode TS
//!
//! MPEG Transport Stream container support for the Transcode library.
//!
//! This crate provides demuxing and muxing capabilities for MPEG-TS files,
//! commonly used for broadcast television and streaming media.
//!
//! ## Features
//!
//! - **188-byte packets**: Standard MPEG-TS packet handling with sync byte validation
//! - **PID filtering**: Filter and route packets by Program ID
//! - **PAT/PMT parsing**: Full Program Association Table and Program Map Table support
//! - **PES packetization**: Packetized Elementary Stream handling with PTS/DTS timestamps
//! - **PCR handling**: Program Clock Reference for stream synchronization
//! - **Continuity counter**: Packet sequence verification
//! - **Adaptation field**: Random access indicators, PCR, stuffing
//! - **Stream types**: Video (H.264, H.265, MPEG-2), Audio (AAC, AC-3, MP2)
//!
//! ## Example: Demuxing
//!
//! ```no_run
//! use transcode_ts::TsDemuxer;
//! use std::fs::File;
//! use std::io::BufReader;
//!
//! let file = File::open("input.ts").unwrap();
//! let reader = BufReader::new(file);
//! let mut demuxer = TsDemuxer::new(reader).unwrap();
//!
//! // Print stream info
//! for i in 0..demuxer.num_streams() {
//!     let stream = demuxer.stream(i).unwrap();
//!     println!("Stream {}: {} (PID: {})", i, stream.codec_name(), stream.pid);
//! }
//!
//! // Read packets
//! while let Ok(Some(packet)) = demuxer.read_packet() {
//!     println!("Packet: stream={}, size={}", packet.stream_index, packet.data().len());
//! }
//! ```
//!
//! ## Example: Muxing
//!
//! ```no_run
//! use transcode_ts::{TsMuxer, MuxerConfig, StreamConfig};
//! use transcode_core::packet::Packet;
//! use transcode_core::timestamp::{Timestamp, TimeBase};
//! use std::fs::File;
//! use std::io::BufWriter;
//!
//! let file = File::create("output.ts").unwrap();
//! let writer = BufWriter::new(file);
//!
//! let config = MuxerConfig::default();
//! let mut muxer = TsMuxer::new(writer, config);
//!
//! // Add streams
//! muxer.add_stream(StreamConfig::video(0x101));
//! muxer.add_stream(StreamConfig::aac(0x102));
//!
//! // Write header
//! muxer.write_header().unwrap();
//!
//! // Write packets
//! let data = vec![0u8; 1000];
//! let mut packet = Packet::new(data);
//! packet.stream_index = 0;
//! packet.pts = Timestamp::new(90000, TimeBase::MPEG);
//! muxer.write_packet(&packet).unwrap();
//!
//! // Write trailer
//! muxer.write_trailer().unwrap();
//! ```
//!
//! ## MPEG-TS Structure
//!
//! An MPEG Transport Stream consists of fixed 188-byte packets:
//!
//! ```text
//! +------+------+------+------+----------------------------+
//! | Sync | TEI  | PUSI | TP   | Adaptation | Payload       |
//! | 0x47 | PID  |      | AFC  | Field      |               |
//! +------+------+------+------+----------------------------+
//!    1      2 bytes      1       0-183        0-184 bytes
//! ```
//!
//! - **Sync byte**: Always 0x47
//! - **PID**: 13-bit Program ID identifying the stream
//! - **PUSI**: Payload Unit Start Indicator
//! - **AFC**: Adaptation Field Control
//! - **Adaptation Field**: Optional, contains PCR, flags
//! - **Payload**: PSI tables or PES data
//!
//! ## Well-Known PIDs
//!
//! | PID    | Description                    |
//! |--------|--------------------------------|
//! | 0x0000 | PAT (Program Association Table)|
//! | 0x0001 | CAT (Conditional Access Table) |
//! | 0x0011 | SDT (Service Description Table)|
//! | 0x0012 | EIT (Event Information Table)  |
//! | 0x1FFF | Null packets                   |

pub mod error;
pub mod packet;
pub mod pes;
pub mod psi;
pub mod demuxer;
pub mod muxer;

// Re-export main types
pub use error::{Result, TsError};
pub use packet::{
    AdaptationField, AdaptationFieldControl, Pcr, ScramblingControl, TsHeader, TsPacket,
    TS_PACKET_SIZE, SYNC_BYTE, PID_NULL, PID_PAT, PID_CAT, PID_TSDT, PID_SDT, PID_EIT, PID_MAX,
};
pub use pes::{
    PesAssembler, PesFlags, PesHeader, PesPacketBuilder, PesTimestamp, StreamId,
    PES_START_CODE_PREFIX,
};
pub use psi::{
    calculate_crc32, Pat, PatEntry, Pmt, PmtStream, PsiAssembler, PsiHeader, Sdt, SdtService,
    StreamType,
};
pub use demuxer::{ProgramInfo, StreamInfo, TsDemuxer};
pub use muxer::{MuxerConfig, StreamConfig, TsMuxer};

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_roundtrip_pat_pmt() {
        // Create PAT
        let mut pat = Pat::new(1);
        pat.add_program(1, 0x100);

        // Create PMT
        let mut pmt = Pmt::new(1, 0x101);
        pmt.add_stream(StreamType::H264 as u8, 0x101);
        pmt.add_stream(StreamType::AacAdts as u8, 0x102);

        // Serialize
        let pat_data = pat.serialize();
        let pmt_data = pmt.serialize();

        // Parse back
        let parsed_pat = Pat::parse(&pat_data).unwrap();
        let parsed_pmt = Pmt::parse(&pmt_data).unwrap();

        assert_eq!(parsed_pat.transport_stream_id, 1);
        assert_eq!(parsed_pat.programs.len(), 1);
        assert_eq!(parsed_pat.programs[0].program_number, 1);
        assert_eq!(parsed_pat.programs[0].pid, 0x100);

        assert_eq!(parsed_pmt.program_number, 1);
        assert_eq!(parsed_pmt.pcr_pid, 0x101);
        assert_eq!(parsed_pmt.streams.len(), 2);
    }

    #[test]
    fn test_ts_packet_roundtrip() {
        let mut data = [0xFFu8; TS_PACKET_SIZE];

        // Create header
        let mut header = TsHeader::new(0x101);
        header.payload_unit_start = true;
        header.continuity_counter = 5;
        header.write(&mut data[..4]).unwrap();

        // Fill payload with pattern
        for (i, byte) in data[4..].iter_mut().enumerate() {
            *byte = (i & 0xFF) as u8;
        }

        // Parse
        let packet = TsPacket::new(data).unwrap();
        let parsed_header = packet.header().unwrap();

        assert_eq!(parsed_header.pid, 0x101);
        assert!(parsed_header.payload_unit_start);
        assert_eq!(parsed_header.continuity_counter, 5);
    }

    #[test]
    fn test_pcr_roundtrip() {
        let original = Pcr::from_seconds(10.5);

        let mut data = [0u8; 6];
        original.write(&mut data).unwrap();

        let parsed = Pcr::parse(&data).unwrap();

        // Should be within 1/300th of the original (extension precision)
        let diff = (original.to_27mhz() as i64 - parsed.to_27mhz() as i64).abs();
        assert!(diff < 300);
    }

    #[test]
    fn test_pes_timestamp_roundtrip() {
        let original = PesTimestamp::from_seconds(5.0);

        let mut data = [0u8; 5];
        original.write(&mut data, 0x20).unwrap();

        let parsed = PesTimestamp::parse(&data).unwrap();
        assert_eq!(parsed.value, original.value);
    }

    #[test]
    fn test_muxer_demuxer_roundtrip() {
        use transcode_core::packet::Packet;
        use transcode_core::timestamp::{TimeBase, Timestamp};

        // Create TS data using muxer
        let mut buffer = Vec::new();
        {
            let config = MuxerConfig::default();
            let mut muxer = TsMuxer::new(&mut buffer, config);

            muxer.add_stream(StreamConfig::video(0x101));
            muxer.write_header().unwrap();

            // Write a video packet
            let data = vec![0xABu8; 100];
            let mut packet = Packet::new(data);
            packet.stream_index = 0;
            packet.pts = Timestamp::new(90000, TimeBase::MPEG);
            packet.dts = Timestamp::new(90000, TimeBase::MPEG);

            muxer.write_packet(&packet).unwrap();
            muxer.write_trailer().unwrap();
        }

        // Verify it's valid TS
        assert!(!buffer.is_empty());
        assert_eq!(buffer.len() % TS_PACKET_SIZE, 0);

        // Parse with demuxer
        let cursor = Cursor::new(buffer);
        let demuxer = TsDemuxer::new(cursor).unwrap();

        // Check PAT was parsed
        assert!(demuxer.pat().is_some());
        assert_eq!(demuxer.pat().unwrap().programs.len(), 1);

        // Check PMT was parsed
        assert!(demuxer.pmt(1).is_some());
        assert_eq!(demuxer.pmt(1).unwrap().streams.len(), 1);

        // Check stream info
        assert_eq!(demuxer.num_streams(), 1);
        let stream = demuxer.stream(0).unwrap();
        assert!(stream.is_video);
        assert_eq!(stream.pid, 0x101);
    }

    #[test]
    fn test_stream_types() {
        assert!(StreamType::H264.is_video());
        assert!(StreamType::H265.is_video());
        assert!(StreamType::Mpeg2Video.is_video());
        assert!(!StreamType::H264.is_audio());

        assert!(StreamType::AacAdts.is_audio());
        assert!(StreamType::Ac3.is_audio());
        assert!(StreamType::Eac3.is_audio());
        assert!(!StreamType::AacAdts.is_video());
    }

    #[test]
    fn test_well_known_pids() {
        assert_eq!(PID_PAT, 0x0000);
        assert_eq!(PID_CAT, 0x0001);
        assert_eq!(PID_SDT, 0x0011);
        assert_eq!(PID_EIT, 0x0012);
        assert_eq!(PID_NULL, 0x1FFF);
    }

    #[test]
    fn test_null_packet() {
        let null_packet = TsPacket::null_packet();
        assert!(null_packet.is_null());
        assert_eq!(null_packet.pid(), PID_NULL);
        assert_eq!(null_packet.data()[0], SYNC_BYTE);
    }
}
