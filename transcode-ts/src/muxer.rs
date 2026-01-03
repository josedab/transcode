//! MPEG Transport Stream muxer.
//!
//! This module provides a muxer for creating MPEG-TS files from elementary
//! stream data.

use crate::error::{Result, TsError};
use crate::packet::{
    AdaptationField, AdaptationFieldControl, Pcr, TsHeader, TsPacket, TS_PACKET_SIZE, PID_PAT,
};
use crate::pes::{PesPacketBuilder, PesTimestamp};
use crate::psi::{Pat, Pmt, StreamType};

use std::io::Write;

use transcode_core::packet::Packet;
use transcode_core::timestamp::{TimeBase, Timestamp};

/// Stream configuration for muxing.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Stream PID.
    pub pid: u16,
    /// Stream type.
    pub stream_type: u8,
    /// Whether to use this stream for PCR.
    pub use_for_pcr: bool,
}

impl StreamConfig {
    /// Create a video stream config.
    pub fn video(pid: u16) -> Self {
        Self {
            pid,
            stream_type: StreamType::H264 as u8,
            use_for_pcr: true, // Video typically carries PCR
        }
    }

    /// Create an H.265 video stream config.
    pub fn h265(pid: u16) -> Self {
        Self {
            pid,
            stream_type: StreamType::H265 as u8,
            use_for_pcr: true,
        }
    }

    /// Create an AAC audio stream config.
    pub fn aac(pid: u16) -> Self {
        Self {
            pid,
            stream_type: StreamType::AacAdts as u8,
            use_for_pcr: false,
        }
    }

    /// Create an AC-3 audio stream config.
    pub fn ac3(pid: u16) -> Self {
        Self {
            pid,
            stream_type: StreamType::Ac3 as u8,
            use_for_pcr: false,
        }
    }

    /// Create a custom stream config.
    pub fn custom(pid: u16, stream_type: u8, use_for_pcr: bool) -> Self {
        Self {
            pid,
            stream_type,
            use_for_pcr,
        }
    }
}

/// Muxer configuration.
#[derive(Debug, Clone)]
pub struct MuxerConfig {
    /// Transport stream ID.
    pub transport_stream_id: u16,
    /// Program number.
    pub program_number: u16,
    /// PMT PID.
    pub pmt_pid: u16,
    /// PAT interval in packets.
    pub pat_interval: u32,
    /// PMT interval in packets.
    pub pmt_interval: u32,
    /// PCR interval in 27MHz units.
    pub pcr_interval: u64,
}

impl Default for MuxerConfig {
    fn default() -> Self {
        Self {
            transport_stream_id: 1,
            program_number: 1,
            pmt_pid: 0x100,
            pat_interval: 500,    // Every 500 packets
            pmt_interval: 500,    // Every 500 packets
            pcr_interval: 27_000, // Every 1ms at 27MHz
        }
    }
}

/// Stream state during muxing.
#[derive(Debug)]
struct StreamState {
    /// Stream configuration.
    config: StreamConfig,
    /// Continuity counter.
    continuity_counter: u8,
    /// Last DTS (90kHz).
    #[allow(dead_code)]
    last_dts: Option<u64>,
    /// Stream index in packet.
    #[allow(dead_code)]
    stream_index: u32,
}

impl StreamState {
    fn new(config: StreamConfig, stream_index: u32) -> Self {
        Self {
            config,
            continuity_counter: 0,
            last_dts: None,
            stream_index,
        }
    }

    fn next_cc(&mut self) -> u8 {
        let cc = self.continuity_counter;
        self.continuity_counter = (self.continuity_counter + 1) & 0x0F;
        cc
    }
}

/// MPEG Transport Stream muxer.
pub struct TsMuxer<W: Write> {
    /// Output writer.
    writer: W,
    /// Muxer configuration.
    config: MuxerConfig,
    /// Streams.
    streams: Vec<StreamState>,
    /// PCR PID.
    pcr_pid: Option<u16>,
    /// PAT.
    pat: Pat,
    /// PMT.
    pmt: Pmt,
    /// PAT continuity counter.
    pat_cc: u8,
    /// PMT continuity counter.
    pmt_cc: u8,
    /// Packets written.
    packets_written: u64,
    /// Last PAT packet number.
    last_pat_packet: u64,
    /// Last PMT packet number.
    last_pmt_packet: u64,
    /// Last PCR value (27MHz).
    last_pcr: u64,
    /// Header written flag.
    header_written: bool,
}

impl<W: Write> TsMuxer<W> {
    /// Create a new TS muxer.
    pub fn new(writer: W, config: MuxerConfig) -> Self {
        let pat = Pat::new(config.transport_stream_id);
        let pmt = Pmt::new(config.program_number, 0x1FFF); // Will be updated when streams added

        Self {
            writer,
            config,
            streams: Vec::new(),
            pcr_pid: None,
            pat,
            pmt,
            pat_cc: 0,
            pmt_cc: 0,
            packets_written: 0,
            last_pat_packet: 0,
            last_pmt_packet: 0,
            last_pcr: 0,
            header_written: false,
        }
    }

    /// Add a stream to the muxer.
    pub fn add_stream(&mut self, config: StreamConfig) -> u32 {
        let stream_index = self.streams.len() as u32;

        // Set PCR PID if this stream should carry PCR
        if config.use_for_pcr && self.pcr_pid.is_none() {
            self.pcr_pid = Some(config.pid);
            self.pmt.pcr_pid = config.pid;
        }

        // Add to PMT
        self.pmt.add_stream(config.stream_type, config.pid);

        // Add stream state
        self.streams.push(StreamState::new(config, stream_index));

        stream_index
    }

    /// Write the header (PAT and PMT).
    pub fn write_header(&mut self) -> Result<()> {
        if self.header_written {
            return Ok(());
        }

        // Update PAT with PMT PID
        self.pat.add_program(self.config.program_number, self.config.pmt_pid);

        // Write PAT
        self.write_pat()?;

        // Write PMT
        self.write_pmt()?;

        self.header_written = true;
        Ok(())
    }

    /// Write PAT.
    fn write_pat(&mut self) -> Result<()> {
        let section = self.pat.serialize();
        self.write_psi(PID_PAT, &section, &mut self.pat_cc.clone())?;
        self.pat_cc = (self.pat_cc + 1) & 0x0F;
        self.last_pat_packet = self.packets_written;
        Ok(())
    }

    /// Write PMT.
    fn write_pmt(&mut self) -> Result<()> {
        let section = self.pmt.serialize();
        self.write_psi(self.config.pmt_pid, &section, &mut self.pmt_cc.clone())?;
        self.pmt_cc = (self.pmt_cc + 1) & 0x0F;
        self.last_pmt_packet = self.packets_written;
        Ok(())
    }

    /// Write a PSI section.
    fn write_psi(&mut self, pid: u16, section: &[u8], cc: &mut u8) -> Result<()> {
        let mut remaining = section;
        let mut first = true;

        while !remaining.is_empty() || first {
            let mut packet_data = [0xFFu8; TS_PACKET_SIZE];

            let mut header = TsHeader::new(pid);
            header.payload_unit_start = first;
            header.continuity_counter = *cc;
            header.adaptation_field_control = AdaptationFieldControl::PayloadOnly;
            header.write(&mut packet_data[..4])?;

            let payload_start = 4;
            let available = TS_PACKET_SIZE - payload_start;

            if first {
                // Add pointer field
                packet_data[payload_start] = 0;
                let section_len = remaining.len().min(available - 1);
                packet_data[payload_start + 1..payload_start + 1 + section_len]
                    .copy_from_slice(&remaining[..section_len]);
                remaining = &remaining[section_len..];
            } else {
                let section_len = remaining.len().min(available);
                packet_data[payload_start..payload_start + section_len]
                    .copy_from_slice(&remaining[..section_len]);
                remaining = &remaining[section_len..];
            }

            self.writer
                .write_all(&packet_data)
                .map_err(|e| TsError::BufferOverflow(format!("Write error: {}", e)))?;

            self.packets_written += 1;
            *cc = (*cc + 1) & 0x0F;
            first = false;
        }

        Ok(())
    }

    /// Write a packet.
    pub fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        // Ensure header is written
        if !self.header_written {
            self.write_header()?;
        }

        // Check if we need to write PSI
        self.maybe_write_psi()?;

        // Find stream
        let stream_index = packet.stream_index as usize;
        if stream_index >= self.streams.len() {
            return Err(TsError::StreamNotFound(packet.stream_index as u16));
        }

        // Get timestamps
        let pts = self.timestamp_to_90khz(packet.pts);
        let dts = self.timestamp_to_90khz(packet.dts);

        // Build PES packet
        let stream = &self.streams[stream_index];
        let is_video = StreamType::from_u8(stream.config.stream_type)
            .map(|st| st.is_video())
            .unwrap_or(false);

        let builder = if is_video {
            PesPacketBuilder::video().random_access(packet.is_keyframe())
        } else {
            PesPacketBuilder::audio()
        };

        let builder = if let (Some(pts_val), Some(dts_val)) = (pts, dts) {
            if pts_val != dts_val {
                builder.pts_dts(PesTimestamp::new(pts_val), PesTimestamp::new(dts_val))
            } else {
                builder.pts(PesTimestamp::new(pts_val))
            }
        } else if let Some(pts_val) = pts {
            builder.pts(PesTimestamp::new(pts_val))
        } else {
            builder
        };

        let pes_header = builder.build_header(packet.data().len())?;

        // Combine PES header and data
        let mut pes_data = pes_header;
        pes_data.extend_from_slice(packet.data());

        // Packetize into TS packets
        self.write_pes(stream_index, &pes_data, packet.is_keyframe(), dts)?;

        // Update stream state
        if let Some(dts_val) = dts {
            self.streams[stream_index].last_dts = Some(dts_val);
        }

        Ok(())
    }

    /// Write PES data as TS packets.
    fn write_pes(
        &mut self,
        stream_index: usize,
        pes_data: &[u8],
        is_keyframe: bool,
        dts: Option<u64>,
    ) -> Result<()> {
        // Get stream properties before entering the loop
        let pid = self.streams[stream_index].config.pid;
        let is_pcr_stream = Some(pid) == self.pcr_pid;

        let mut remaining = pes_data;
        let mut first = true;

        while !remaining.is_empty() {
            let mut packet_data = [0xFFu8; TS_PACKET_SIZE];

            // Determine if we need adaptation field
            let need_pcr = first && is_pcr_stream && self.should_write_pcr(dts);
            let need_rai = first && is_keyframe;

            let (af_size, payload_start) = if need_pcr || need_rai {
                // Build adaptation field
                let mut af = if need_pcr {
                    let pcr_value = dts
                        .map(|d| d * 300)
                        .unwrap_or(self.last_pcr + self.config.pcr_interval);
                    self.last_pcr = pcr_value;
                    AdaptationField::with_pcr(Pcr::from_27mhz(pcr_value))
                } else {
                    AdaptationField::stuffing(2)
                };

                if need_rai {
                    af.random_access = true;
                }

                let af_bytes = af.write(&mut packet_data[4..])?;
                (af_bytes, 4 + af_bytes)
            } else {
                (0, 4)
            };

            let available = TS_PACKET_SIZE - payload_start;
            let payload_len = remaining.len().min(available);

            // Calculate stuffing needed
            let stuffing_needed = if payload_len < available && remaining.len() <= available {
                available - payload_len
            } else {
                0
            };

            // Build header
            let mut header = TsHeader::new(pid);
            header.payload_unit_start = first;
            header.continuity_counter = self.streams[stream_index].next_cc();

            if af_size > 0 || stuffing_needed > 0 {
                header.adaptation_field_control = if payload_len > 0 {
                    AdaptationFieldControl::AdaptationFieldAndPayload
                } else {
                    AdaptationFieldControl::AdaptationFieldOnly
                };
            } else {
                header.adaptation_field_control = AdaptationFieldControl::PayloadOnly;
            }

            header.write(&mut packet_data[..4])?;

            // Add stuffing if needed and we don't already have adaptation field
            let final_payload_start = if stuffing_needed > 0 && af_size == 0 {
                let af = AdaptationField::stuffing(stuffing_needed);
                af.write(&mut packet_data[4..])?;

                // Update header to indicate adaptation field
                header.adaptation_field_control = AdaptationFieldControl::AdaptationFieldAndPayload;
                header.write(&mut packet_data[..4])?;

                4 + stuffing_needed
            } else {
                payload_start
            };

            // Copy payload
            packet_data[final_payload_start..final_payload_start + payload_len]
                .copy_from_slice(&remaining[..payload_len]);

            remaining = &remaining[payload_len..];

            self.writer
                .write_all(&packet_data)
                .map_err(|e| TsError::BufferOverflow(format!("Write error: {}", e)))?;

            self.packets_written += 1;
            first = false;
        }

        Ok(())
    }

    /// Check if we should write PSI tables.
    fn maybe_write_psi(&mut self) -> Result<()> {
        if self.packets_written - self.last_pat_packet >= self.config.pat_interval as u64 {
            self.write_pat()?;
        }

        if self.packets_written - self.last_pmt_packet >= self.config.pmt_interval as u64 {
            self.write_pmt()?;
        }

        Ok(())
    }

    /// Check if we should write PCR.
    fn should_write_pcr(&self, dts: Option<u64>) -> bool {
        if let Some(dts_val) = dts {
            let pcr_value = dts_val * 300;
            pcr_value.saturating_sub(self.last_pcr) >= self.config.pcr_interval
        } else {
            true
        }
    }

    /// Convert timestamp to 90kHz.
    fn timestamp_to_90khz(&self, ts: Timestamp) -> Option<u64> {
        if !ts.is_valid() {
            return None;
        }

        let mpeg_ts = ts.rescale(TimeBase::MPEG);
        Some(mpeg_ts.value as u64)
    }

    /// Write the trailer.
    pub fn write_trailer(&mut self) -> Result<()> {
        // Write final PSI tables
        self.write_pat()?;
        self.write_pmt()?;

        // Flush writer
        self.writer
            .flush()
            .map_err(|e| TsError::BufferOverflow(format!("Flush error: {}", e)))?;

        Ok(())
    }

    /// Write a null packet.
    pub fn write_null_packet(&mut self) -> Result<()> {
        let null_packet = TsPacket::null_packet();
        self.writer
            .write_all(null_packet.data())
            .map_err(|e| TsError::BufferOverflow(format!("Write error: {}", e)))?;
        self.packets_written += 1;
        Ok(())
    }

    /// Get the number of packets written.
    pub fn packets_written(&self) -> u64 {
        self.packets_written
    }

    /// Get the number of bytes written.
    pub fn bytes_written(&self) -> u64 {
        self.packets_written * TS_PACKET_SIZE as u64
    }

    /// Get the underlying writer.
    pub fn into_inner(self) -> W {
        self.writer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use transcode_core::timestamp::Timestamp;

    #[test]
    fn test_muxer_creation() {
        let buffer = Vec::new();
        let config = MuxerConfig::default();
        let muxer = TsMuxer::new(buffer, config);

        assert_eq!(muxer.packets_written(), 0);
    }

    #[test]
    fn test_muxer_add_stream() {
        let buffer = Vec::new();
        let config = MuxerConfig::default();
        let mut muxer = TsMuxer::new(buffer, config);

        let video_idx = muxer.add_stream(StreamConfig::video(0x101));
        let audio_idx = muxer.add_stream(StreamConfig::aac(0x102));

        assert_eq!(video_idx, 0);
        assert_eq!(audio_idx, 1);
    }

    #[test]
    fn test_muxer_write_header() {
        let buffer = Vec::new();
        let config = MuxerConfig::default();
        let mut muxer = TsMuxer::new(buffer, config);

        muxer.add_stream(StreamConfig::video(0x101));
        muxer.write_header().unwrap();

        assert!(muxer.header_written);
        assert!(muxer.packets_written() >= 2); // At least PAT and PMT
    }

    #[test]
    fn test_muxer_write_packet() {
        let buffer = Vec::new();
        let config = MuxerConfig::default();
        let mut muxer = TsMuxer::new(buffer, config);

        muxer.add_stream(StreamConfig::video(0x101));
        muxer.write_header().unwrap();

        // Create a test packet
        let data = vec![0u8; 1000];
        let mut packet = Packet::new(data);
        packet.stream_index = 0;
        packet.pts = Timestamp::new(90000, TimeBase::MPEG);
        packet.dts = Timestamp::new(90000, TimeBase::MPEG);

        muxer.write_packet(&packet).unwrap();

        // Should have written multiple TS packets
        assert!(muxer.bytes_written() > TS_PACKET_SIZE as u64);
    }

    #[test]
    fn test_muxer_output_valid() {
        let mut buffer = Vec::new();
        {
            let config = MuxerConfig::default();
            let mut muxer = TsMuxer::new(&mut buffer, config);

            muxer.add_stream(StreamConfig::video(0x101));
            muxer.write_header().unwrap();

            let data = vec![0xABu8; 500];
            let mut packet = Packet::new(data);
            packet.stream_index = 0;
            packet.pts = Timestamp::new(90000, TimeBase::MPEG);
            packet.dts = Timestamp::new(90000, TimeBase::MPEG);

            muxer.write_packet(&packet).unwrap();
            muxer.write_trailer().unwrap();
        }

        // Verify output is valid TS packets
        assert!(buffer.len() >= TS_PACKET_SIZE);
        assert_eq!(buffer.len() % TS_PACKET_SIZE, 0);

        // Check sync bytes
        for i in (0..buffer.len()).step_by(TS_PACKET_SIZE) {
            assert_eq!(buffer[i], 0x47, "Invalid sync byte at offset {}", i);
        }
    }

    #[test]
    fn test_stream_config() {
        let video = StreamConfig::video(0x101);
        assert_eq!(video.stream_type, StreamType::H264 as u8);
        assert!(video.use_for_pcr);

        let h265 = StreamConfig::h265(0x102);
        assert_eq!(h265.stream_type, StreamType::H265 as u8);

        let aac = StreamConfig::aac(0x103);
        assert_eq!(aac.stream_type, StreamType::AacAdts as u8);
        assert!(!aac.use_for_pcr);

        let ac3 = StreamConfig::ac3(0x104);
        assert_eq!(ac3.stream_type, StreamType::Ac3 as u8);
    }

    #[test]
    fn test_muxer_config() {
        let config = MuxerConfig {
            transport_stream_id: 42,
            program_number: 3,
            pmt_pid: 0x200,
            pat_interval: 1000,
            pmt_interval: 1000,
            pcr_interval: 54000,
        };

        assert_eq!(config.transport_stream_id, 42);
        assert_eq!(config.program_number, 3);
        assert_eq!(config.pmt_pid, 0x200);
    }
}
