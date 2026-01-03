//! MPEG Transport Stream demuxer.
//!
//! This module provides a demuxer for reading MPEG-TS files and extracting
//! elementary stream data.

use crate::error::{Result, TsError};
use crate::packet::{Pcr, TsPacket, TS_PACKET_SIZE, PID_PAT, PID_NULL};
use crate::pes::{PesAssembler, PesHeader, PesTimestamp};
use crate::psi::{Pat, Pmt, PsiAssembler, StreamType};

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use transcode_core::error::Error as CoreError;
use transcode_core::packet::Packet;
use transcode_core::timestamp::{TimeBase, Timestamp};

/// Stream information extracted from PMT.
#[derive(Debug, Clone)]
pub struct StreamInfo {
    /// Elementary stream PID.
    pub pid: u16,
    /// Stream type.
    pub stream_type: u8,
    /// Whether this is a video stream.
    pub is_video: bool,
    /// Whether this is an audio stream.
    pub is_audio: bool,
    /// Stream descriptors.
    pub descriptors: Vec<u8>,
}

impl StreamInfo {
    /// Get the codec name for this stream type.
    pub fn codec_name(&self) -> &'static str {
        match StreamType::from_u8(self.stream_type) {
            Some(StreamType::H264) => "H.264",
            Some(StreamType::H265) => "H.265",
            Some(StreamType::Mpeg2Video) => "MPEG-2",
            Some(StreamType::Mpeg1Video) => "MPEG-1",
            Some(StreamType::AacAdts) => "AAC",
            Some(StreamType::AacLatm) => "AAC-LATM",
            Some(StreamType::Mpeg1Audio) => "MP1",
            Some(StreamType::Mpeg2Audio) => "MP2",
            Some(StreamType::Ac3) => "AC-3",
            Some(StreamType::Eac3) => "E-AC-3",
            _ => "Unknown",
        }
    }
}

/// Program information.
#[derive(Debug, Clone)]
pub struct ProgramInfo {
    /// Program number.
    pub program_number: u16,
    /// PMT PID.
    pub pmt_pid: u16,
    /// PCR PID.
    pub pcr_pid: u16,
    /// Elementary streams.
    pub streams: Vec<StreamInfo>,
}

/// State for tracking a PID.
#[derive(Debug)]
struct PidState {
    /// PES assembler for elementary streams.
    pes_assembler: Option<PesAssembler>,
    /// PSI assembler for tables.
    psi_assembler: Option<PsiAssembler>,
    /// Last continuity counter.
    last_cc: Option<u8>,
    /// Last PCR value.
    last_pcr: Option<Pcr>,
}

impl PidState {
    fn new_pes(pid: u16) -> Self {
        Self {
            pes_assembler: Some(PesAssembler::new(pid)),
            psi_assembler: None,
            last_cc: None,
            last_pcr: None,
        }
    }

    fn new_psi(pid: u16) -> Self {
        Self {
            pes_assembler: None,
            psi_assembler: Some(PsiAssembler::new(pid)),
            last_cc: None,
            last_pcr: None,
        }
    }
}

/// MPEG Transport Stream demuxer.
pub struct TsDemuxer<R: Read + Seek> {
    /// Input reader.
    reader: R,
    /// Read buffer.
    buffer: [u8; TS_PACKET_SIZE],
    /// PAT.
    pat: Option<Pat>,
    /// PMTs by program number.
    pmts: HashMap<u16, Pmt>,
    /// Programs.
    programs: Vec<ProgramInfo>,
    /// PID states.
    pid_states: HashMap<u16, PidState>,
    /// PID to program mapping.
    pid_to_program: HashMap<u16, u16>,
    /// PID to stream index mapping.
    pid_to_stream_index: HashMap<u16, u32>,
    /// Packets read.
    packets_read: u64,
    /// Current position.
    position: u64,
    /// File size (if known).
    file_size: Option<u64>,
    /// First PCR value.
    first_pcr: Option<Pcr>,
    /// Last PCR value.
    last_pcr: Option<Pcr>,
    /// Duration in seconds.
    duration: Option<f64>,
}

impl<R: Read + Seek> TsDemuxer<R> {
    /// Create a new TS demuxer.
    pub fn new(mut reader: R) -> Result<Self> {
        // Try to get file size
        let file_size = reader.seek(SeekFrom::End(0)).ok();
        reader.seek(SeekFrom::Start(0)).map_err(|e| {
            TsError::MissingData(format!("Failed to seek to start: {}", e))
        })?;

        let mut demuxer = Self {
            reader,
            buffer: [0u8; TS_PACKET_SIZE],
            pat: None,
            pmts: HashMap::new(),
            programs: Vec::new(),
            pid_states: HashMap::new(),
            pid_to_program: HashMap::new(),
            pid_to_stream_index: HashMap::new(),
            packets_read: 0,
            position: 0,
            file_size,
            first_pcr: None,
            last_pcr: None,
            duration: None,
        };

        // Initialize PAT assembler
        demuxer.pid_states.insert(PID_PAT, PidState::new_psi(PID_PAT));

        // Probe the stream
        demuxer.probe()?;

        Ok(demuxer)
    }

    /// Probe the stream to find programs and streams.
    fn probe(&mut self) -> Result<()> {
        // Read until we have PAT and at least one PMT
        let mut found_pmt = false;
        let mut probe_packets = 0;
        const MAX_PROBE_PACKETS: u64 = 10000;

        while probe_packets < MAX_PROBE_PACKETS && (!found_pmt || self.pat.is_none()) {
            match self.read_ts_packet() {
                Ok(Some(packet)) => {
                    self.process_psi_packet(&packet)?;
                    found_pmt = !self.pmts.is_empty();
                }
                Ok(None) => break,
                Err(_) => continue, // Skip errors during probing
            }
            probe_packets += 1;
        }

        // Build program info
        self.build_program_info();

        // Estimate duration
        self.estimate_duration()?;

        // Seek back to start
        self.reader.seek(SeekFrom::Start(0)).map_err(|e| {
            TsError::MissingData(format!("Failed to seek to start: {}", e))
        })?;
        self.position = 0;
        self.packets_read = 0;

        // Reset assemblers
        for state in self.pid_states.values_mut() {
            if let Some(ref mut pes) = state.pes_assembler {
                pes.reset();
            }
            if let Some(ref mut psi) = state.psi_assembler {
                psi.reset();
            }
            state.last_cc = None;
        }

        Ok(())
    }

    /// Build program info from PAT and PMTs.
    fn build_program_info(&mut self) {
        self.programs.clear();

        if let Some(ref pat) = self.pat {
            for entry in &pat.programs {
                if entry.program_number == 0 {
                    continue; // Skip NIT
                }

                if let Some(pmt) = self.pmts.get(&entry.program_number) {
                    let streams: Vec<StreamInfo> = pmt
                        .streams
                        .iter()
                        .map(|s| {
                            let stream_type = StreamType::from_u8(s.stream_type);
                            StreamInfo {
                                pid: s.pid,
                                stream_type: s.stream_type,
                                is_video: stream_type.map(|t| t.is_video()).unwrap_or(false),
                                is_audio: stream_type.map(|t| t.is_audio()).unwrap_or(false),
                                descriptors: s.descriptors.clone(),
                            }
                        })
                        .collect();

                    self.programs.push(ProgramInfo {
                        program_number: entry.program_number,
                        pmt_pid: entry.pid,
                        pcr_pid: pmt.pcr_pid,
                        streams,
                    });
                }
            }
        }

        // Build PID to stream index mapping
        let mut stream_index = 0u32;
        for program in &self.programs {
            for stream in &program.streams {
                self.pid_to_stream_index.insert(stream.pid, stream_index);
                self.pid_to_program.insert(stream.pid, program.program_number);

                // Create PES assembler for this PID
                self.pid_states
                    .entry(stream.pid)
                    .or_insert_with(|| PidState::new_pes(stream.pid));

                stream_index += 1;
            }
        }
    }

    /// Estimate duration from first and last PCR.
    fn estimate_duration(&mut self) -> Result<()> {
        if let Some(file_size) = self.file_size {
            // Read some packets from the end
            let end_offset = file_size.saturating_sub(TS_PACKET_SIZE as u64 * 1000);
            self.reader.seek(SeekFrom::Start(end_offset)).ok();

            let mut last_pcr = None;
            for _ in 0..1000 {
                if let Ok(Some(packet)) = self.read_ts_packet() {
                    if let Ok(Some(af)) = packet.adaptation_field() {
                        if let Some(pcr) = af.pcr {
                            last_pcr = Some(pcr);
                        }
                    }
                }
            }

            self.last_pcr = last_pcr;

            // Calculate duration
            if let (Some(first), Some(last)) = (self.first_pcr, self.last_pcr) {
                let duration = (last.to_27mhz().wrapping_sub(first.to_27mhz())) as f64
                    / Pcr::CLOCK_RATE as f64;
                if duration > 0.0 {
                    self.duration = Some(duration);
                }
            }
        }

        Ok(())
    }

    /// Process PSI packet (PAT, PMT).
    fn process_psi_packet(&mut self, packet: &TsPacket) -> Result<()> {
        let pid = packet.pid();
        let cc = packet.continuity_counter();
        let pusi = packet.payload_unit_start();

        // Handle PAT
        if pid == PID_PAT {
            if let Some(payload) = packet.payload() {
                let state = self.pid_states.get_mut(&PID_PAT).unwrap();
                if let Some(ref mut assembler) = state.psi_assembler {
                    if let Some(section) = assembler.add(payload, pusi, cc)? {
                        if let Ok(pat) = Pat::parse(&section) {
                            // Register PMT PIDs
                            for entry in &pat.programs {
                                if entry.program_number != 0 {
                                    self.pid_states
                                        .entry(entry.pid)
                                        .or_insert_with(|| PidState::new_psi(entry.pid));
                                }
                            }
                            self.pat = Some(pat);
                        }
                    }
                }
            }
            return Ok(());
        }

        // Check if this is a PMT PID
        let is_pmt_pid = self
            .pat
            .as_ref()
            .map(|pat| pat.programs.iter().any(|p| p.pid == pid && p.program_number != 0))
            .unwrap_or(false);

        if is_pmt_pid {
            if let Some(payload) = packet.payload() {
                if let Some(state) = self.pid_states.get_mut(&pid) {
                    if let Some(ref mut assembler) = state.psi_assembler {
                        if let Some(section) = assembler.add(payload, pusi, cc)? {
                            if let Ok(pmt) = Pmt::parse(&section) {
                                self.pmts.insert(pmt.program_number, pmt);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Read a single TS packet.
    fn read_ts_packet(&mut self) -> Result<Option<TsPacket>> {
        // Sync to packet boundary if needed
        loop {
            match self.reader.read_exact(&mut self.buffer) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
                Err(e) => return Err(TsError::MissingData(format!("Read error: {}", e))),
            }

            if self.buffer[0] == 0x47 {
                break;
            }

            // Resync
            let mut found = false;
            for i in 1..TS_PACKET_SIZE {
                if self.buffer[i] == 0x47 {
                    // Found potential sync
                    let remaining = TS_PACKET_SIZE - i;
                    self.buffer.copy_within(i.., 0);
                    if self.reader.read_exact(&mut self.buffer[remaining..]).is_ok() {
                        found = true;
                        break;
                    }
                }
            }

            if !found {
                continue;
            }
        }

        self.position += TS_PACKET_SIZE as u64;
        self.packets_read += 1;

        // Track first PCR
        let packet = TsPacket::from_slice(&self.buffer)?;
        if self.first_pcr.is_none() {
            if let Ok(Some(af)) = packet.adaptation_field() {
                if let Some(pcr) = af.pcr {
                    self.first_pcr = Some(pcr);
                }
            }
        }

        Ok(Some(packet))
    }

    /// Read the next elementary stream packet.
    pub fn read_packet(&mut self) -> std::result::Result<Option<Packet<'static>>, CoreError> {
        loop {
            let ts_packet = match self.read_ts_packet() {
                Ok(Some(p)) => p,
                Ok(None) => {
                    // EOF - flush any remaining PES data
                    for (pid, state) in &mut self.pid_states {
                        if let Some(ref mut pes) = state.pes_assembler {
                            if let Some(data) = pes.flush() {
                                if let Some(&stream_index) = self.pid_to_stream_index.get(pid) {
                                    return Ok(Some(self.make_packet(data, stream_index, None, None)?));
                                }
                            }
                        }
                    }
                    return Ok(None);
                }
                Err(e) => return Err(e.into()),
            };

            let pid = ts_packet.pid();

            // Skip null packets
            if pid == PID_NULL {
                continue;
            }

            // Track PCR
            if let Ok(Some(af)) = ts_packet.adaptation_field() {
                if let Some(pcr) = af.pcr {
                    if let Some(state) = self.pid_states.get_mut(&pid) {
                        state.last_pcr = Some(pcr);
                    }
                    self.last_pcr = Some(pcr);
                }
            }

            // Process PSI packets
            if pid == PID_PAT
                || self
                    .pat
                    .as_ref()
                    .map(|pat| pat.programs.iter().any(|p| p.pid == pid))
                    .unwrap_or(false)
            {
                self.process_psi_packet(&ts_packet).ok();
                continue;
            }

            // Check if this is an elementary stream we're tracking
            let stream_index = match self.pid_to_stream_index.get(&pid) {
                Some(&idx) => idx,
                None => continue,
            };

            // Get payload
            let payload = match ts_packet.payload() {
                Some(p) => p,
                None => continue,
            };

            let pusi = ts_packet.payload_unit_start();
            let is_keyframe = ts_packet
                .adaptation_field()
                .ok()
                .flatten()
                .map(|af| af.random_access)
                .unwrap_or(false);

            // Assemble PES packet
            if let Some(state) = self.pid_states.get_mut(&pid) {
                if let Some(ref mut pes) = state.pes_assembler {
                    match pes.add(payload, pusi) {
                        Ok(Some(pes_data)) => {
                            // Parse PES header for timestamps
                            let (pts, dts) = if let Ok(header) = PesHeader::parse(&pes_data) {
                                (
                                    header.pts.map(|ts| ts.value),
                                    header.dts.map(|ts| ts.value),
                                )
                            } else {
                                (None, None)
                            };

                            let mut packet = self.make_packet(pes_data, stream_index, pts, dts)?;

                            // Check for keyframe
                            if is_keyframe {
                                packet.set_keyframe(true);
                            }

                            return Ok(Some(packet));
                        }
                        Ok(None) => continue,
                        Err(_) => continue,
                    }
                }
            }
        }
    }

    /// Create a packet from PES data.
    fn make_packet(
        &self,
        pes_data: Vec<u8>,
        stream_index: u32,
        pts: Option<u64>,
        dts: Option<u64>,
    ) -> std::result::Result<Packet<'static>, CoreError> {
        // Parse PES header to get payload
        let (_header_size, payload_data) = if let Ok(header) = PesHeader::parse(&pes_data) {
            (header.header_size, pes_data[header.header_size..].to_vec())
        } else {
            // No valid PES header, use raw data
            (0, pes_data)
        };

        let mut packet = Packet::new(payload_data);
        packet.stream_index = stream_index;

        // Set timestamps (90kHz time base)
        let time_base = TimeBase::MPEG;
        if let Some(pts_val) = pts {
            packet.pts = Timestamp::new(pts_val as i64, time_base);
        }
        if let Some(dts_val) = dts {
            packet.dts = Timestamp::new(dts_val as i64, time_base);
        } else if let Some(pts_val) = pts {
            // Use PTS as DTS if DTS not present
            packet.dts = Timestamp::new(pts_val as i64, time_base);
        }

        Ok(packet)
    }

    /// Get the number of programs.
    pub fn num_programs(&self) -> usize {
        self.programs.len()
    }

    /// Get program info.
    pub fn program(&self, index: usize) -> Option<&ProgramInfo> {
        self.programs.get(index)
    }

    /// Get all programs.
    pub fn programs(&self) -> &[ProgramInfo] {
        &self.programs
    }

    /// Get total number of streams across all programs.
    pub fn num_streams(&self) -> usize {
        self.programs.iter().map(|p| p.streams.len()).sum()
    }

    /// Get stream info by index.
    pub fn stream(&self, index: usize) -> Option<&StreamInfo> {
        let mut current = 0;
        for program in &self.programs {
            if index < current + program.streams.len() {
                return Some(&program.streams[index - current]);
            }
            current += program.streams.len();
        }
        None
    }

    /// Get duration in seconds.
    pub fn duration(&self) -> Option<f64> {
        self.duration
    }

    /// Get the PAT.
    pub fn pat(&self) -> Option<&Pat> {
        self.pat.as_ref()
    }

    /// Get PMT for a program.
    pub fn pmt(&self, program_number: u16) -> Option<&Pmt> {
        self.pmts.get(&program_number)
    }

    /// Get current position in bytes.
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Get number of packets read.
    pub fn packets_read(&self) -> u64 {
        self.packets_read
    }

    /// Seek to a position in the stream.
    pub fn seek(&mut self, position_bytes: u64) -> Result<()> {
        // Align to packet boundary
        let aligned_position = (position_bytes / TS_PACKET_SIZE as u64) * TS_PACKET_SIZE as u64;

        self.reader
            .seek(SeekFrom::Start(aligned_position))
            .map_err(|e| TsError::MissingData(format!("Seek error: {}", e)))?;

        self.position = aligned_position;

        // Reset assemblers
        for state in self.pid_states.values_mut() {
            if let Some(ref mut pes) = state.pes_assembler {
                pes.reset();
            }
            if let Some(ref mut psi) = state.psi_assembler {
                psi.reset();
            }
            state.last_cc = None;
        }

        Ok(())
    }

    /// Seek to a timestamp (in 90kHz units).
    pub fn seek_to_timestamp(&mut self, timestamp: u64) -> Result<()> {
        // Estimate position based on file size and duration
        if let (Some(file_size), Some(duration)) = (self.file_size, self.duration) {
            let time_seconds = timestamp as f64 / PesTimestamp::CLOCK_RATE as f64;
            let fraction = time_seconds / duration;
            let position = (file_size as f64 * fraction) as u64;

            self.seek(position)?;

            // Read packets until we find one at or after the target timestamp
            // This is a simple approach; a more sophisticated implementation would
            // do binary search
            loop {
                match self.read_ts_packet() {
                    Ok(Some(packet)) => {
                        if let Ok(Some(af)) = packet.adaptation_field() {
                            if let Some(pcr) = af.pcr {
                                if pcr.base >= timestamp / 300 {
                                    // Found a packet at or after target
                                    // Seek back a bit to ensure we don't miss keyframes
                                    if self.position > TS_PACKET_SIZE as u64 * 100 {
                                        self.seek(self.position - TS_PACKET_SIZE as u64 * 100)?;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    Ok(None) => break,
                    Err(_) => continue,
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn create_test_ts_data() -> Vec<u8> {
        // Create minimal TS data with PAT and PMT
        let mut data = Vec::new();

        // Create PAT
        let mut pat = Pat::new(1);
        pat.add_program(1, 0x100);
        let pat_section = pat.serialize();

        // Create PAT packet
        let mut pat_packet = [0xFFu8; TS_PACKET_SIZE];
        pat_packet[0] = 0x47; // Sync byte
        pat_packet[1] = 0x40; // PUSI + PID high (0)
        pat_packet[2] = 0x00; // PID low (0 = PAT)
        pat_packet[3] = 0x10; // Payload only

        // Pointer field + section
        pat_packet[4] = 0x00; // Pointer field
        let pat_len = pat_section.len().min(183);
        pat_packet[5..5 + pat_len].copy_from_slice(&pat_section[..pat_len]);

        data.extend_from_slice(&pat_packet);

        // Create PMT
        let mut pmt = Pmt::new(1, 0x101);
        pmt.add_stream(StreamType::H264 as u8, 0x101);
        pmt.add_stream(StreamType::AacAdts as u8, 0x102);
        let pmt_section = pmt.serialize();

        // Create PMT packet
        let mut pmt_packet = [0xFFu8; TS_PACKET_SIZE];
        pmt_packet[0] = 0x47;
        pmt_packet[1] = 0x41; // PUSI + PID high
        pmt_packet[2] = 0x00; // PID low (0x100)
        pmt_packet[3] = 0x10; // Payload only

        pmt_packet[4] = 0x00; // Pointer field
        let pmt_len = pmt_section.len().min(183);
        pmt_packet[5..5 + pmt_len].copy_from_slice(&pmt_section[..pmt_len]);

        data.extend_from_slice(&pmt_packet);

        // Add some null packets
        for _ in 0..10 {
            let null_packet = TsPacket::null_packet();
            data.extend_from_slice(null_packet.data());
        }

        data
    }

    #[test]
    fn test_demuxer_creation() {
        let data = create_test_ts_data();
        let cursor = Cursor::new(data);
        let demuxer = TsDemuxer::new(cursor);

        assert!(demuxer.is_ok());
    }

    #[test]
    fn test_demuxer_pat_pmt_parsing() {
        let data = create_test_ts_data();
        let cursor = Cursor::new(data);
        let demuxer = TsDemuxer::new(cursor).unwrap();

        // Check PAT was parsed
        assert!(demuxer.pat().is_some());
        let pat = demuxer.pat().unwrap();
        assert_eq!(pat.transport_stream_id, 1);
        assert_eq!(pat.programs.len(), 1);
        assert_eq!(pat.programs[0].program_number, 1);

        // Check PMT was parsed
        assert!(demuxer.pmt(1).is_some());
        let pmt = demuxer.pmt(1).unwrap();
        assert_eq!(pmt.program_number, 1);
        assert_eq!(pmt.streams.len(), 2);

        // Check programs
        assert_eq!(demuxer.num_programs(), 1);
        let program = demuxer.program(0).unwrap();
        assert_eq!(program.streams.len(), 2);
    }

    #[test]
    fn test_stream_info() {
        let data = create_test_ts_data();
        let cursor = Cursor::new(data);
        let demuxer = TsDemuxer::new(cursor).unwrap();

        assert_eq!(demuxer.num_streams(), 2);

        let video_stream = demuxer.stream(0).unwrap();
        assert!(video_stream.is_video);
        assert!(!video_stream.is_audio);
        assert_eq!(video_stream.codec_name(), "H.264");

        let audio_stream = demuxer.stream(1).unwrap();
        assert!(!audio_stream.is_video);
        assert!(audio_stream.is_audio);
        assert_eq!(audio_stream.codec_name(), "AAC");
    }
}
