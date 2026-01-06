//! AVI demuxer

use crate::chunks::{chunk_ids, parse_index, AviChunk, ChunkId, IndexEntry, ListChunk, RiffChunk};
use crate::error::{AviError, Result};
use crate::types::{
    AudioFormat, AviFlags, AviHeader, Rect, StreamHeader, StreamType, VideoFormat,
};
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{Cursor, Read};

/// Stream information
#[derive(Debug, Clone)]
pub struct StreamInfo {
    /// Stream index
    pub index: u32,
    /// Stream header
    pub header: StreamHeader,
    /// Video format (if video stream)
    pub video_format: Option<VideoFormat>,
    /// Audio format (if audio stream)
    pub audio_format: Option<AudioFormat>,
    /// Stream name (if available)
    pub name: Option<String>,
}

impl StreamInfo {
    /// Check if this is a video stream
    pub fn is_video(&self) -> bool {
        self.header.stream_type == StreamType::Video
    }

    /// Check if this is an audio stream
    pub fn is_audio(&self) -> bool {
        self.header.stream_type == StreamType::Audio
    }

    /// Get frame rate for video streams
    pub fn frame_rate(&self) -> f64 {
        if self.header.scale > 0 {
            self.header.rate as f64 / self.header.scale as f64
        } else {
            0.0
        }
    }

    /// Get duration in seconds
    pub fn duration(&self) -> f64 {
        if self.header.rate > 0 && self.header.scale > 0 {
            (self.header.length as f64 * self.header.scale as f64) / self.header.rate as f64
        } else {
            0.0
        }
    }
}

/// AVI packet (decoded chunk)
#[derive(Debug, Clone)]
pub struct AviPacket {
    /// Stream index
    pub stream_index: u32,
    /// Packet data
    pub data: Vec<u8>,
    /// Timestamp in stream timebase
    pub timestamp: u64,
    /// Duration in stream timebase
    pub duration: u32,
    /// Is keyframe
    pub keyframe: bool,
}

/// AVI demuxer
pub struct AviDemuxer<'a> {
    /// Source data
    data: &'a [u8],
    /// AVI header
    header: AviHeader,
    /// Stream information
    streams: Vec<StreamInfo>,
    /// Index entries
    index: Vec<IndexEntry>,
    /// Current read position in movi
    current_offset: usize,
    /// Movi list offset
    movi_offset: usize,
    /// Movi list size
    movi_size: usize,
    /// Frame counters per stream
    frame_counters: Vec<u64>,
    /// OpenDML extended header
    odml_total_frames: Option<u64>,
}

impl<'a> AviDemuxer<'a> {
    /// Create new demuxer from data
    pub fn new(data: &'a [u8]) -> Result<Self> {
        let mut demuxer = AviDemuxer {
            data,
            header: AviHeader::default(),
            streams: Vec::new(),
            index: Vec::new(),
            current_offset: 0,
            movi_offset: 0,
            movi_size: 0,
            frame_counters: Vec::new(),
            odml_total_frames: None,
        };

        demuxer.parse()?;
        Ok(demuxer)
    }

    /// Parse AVI structure
    fn parse(&mut self) -> Result<()> {
        // Check RIFF header
        if self.data.len() < 12 {
            return Err(AviError::InsufficientData {
                needed: 12,
                available: self.data.len(),
            });
        }

        if &self.data[0..4] != b"RIFF" {
            return Err(AviError::InvalidRiff);
        }

        let file_size = u32::from_le_bytes([
            self.data[4],
            self.data[5],
            self.data[6],
            self.data[7],
        ]) as usize;

        if &self.data[8..12] != b"AVI " && &self.data[8..12] != b"AVIX" {
            return Err(AviError::InvalidAvi);
        }

        log::debug!("Parsing AVI file, size: {}", file_size);

        // Parse chunks
        let mut offset = 12;

        while offset + 8 <= self.data.len() {
            let (chunk, next_offset) = match RiffChunk::read(self.data, offset) {
                Ok(r) => r,
                Err(_) => break,
            };

            match chunk.id {
                id if id == chunk_ids::LIST => {
                    let list = ListChunk::parse(&chunk.data)?;
                    self.parse_list(&list, offset + 12)?;
                }
                id if id == chunk_ids::IDX1 => {
                    self.index = parse_index(&chunk.data);
                    log::debug!("Parsed {} index entries", self.index.len());
                }
                id if id == chunk_ids::JUNK => {
                    // Skip junk chunks
                }
                _ => {
                    log::debug!("Skipping chunk: {}", chunk.id);
                }
            }

            offset = next_offset;
        }

        // Initialize frame counters
        self.frame_counters = vec![0; self.streams.len()];

        // Set current offset to start of movi data
        self.current_offset = self.movi_offset;

        Ok(())
    }

    /// Parse LIST chunk
    fn parse_list(&mut self, list: &ListChunk, base_offset: usize) -> Result<()> {
        match list.list_type {
            id if id == chunk_ids::HDRL => {
                self.parse_hdrl(list)?;
            }
            id if id == chunk_ids::STRL => {
                self.parse_strl(list)?;
            }
            id if id == chunk_ids::MOVI => {
                // Calculate actual start of movi data (after LIST header and type)
                self.movi_offset = base_offset;
                self.movi_size = list.chunks.iter().map(|c| c.total_size()).sum();
                log::debug!(
                    "Found movi list at offset {}, size {}",
                    self.movi_offset,
                    self.movi_size
                );
            }
            id if id == chunk_ids::ODML => {
                self.parse_odml(list)?;
            }
            _ => {
                log::debug!("Skipping list: {}", list.list_type);
            }
        }

        // Recursively parse nested lists
        for chunk in &list.chunks {
            if chunk.id == chunk_ids::LIST {
                let nested = ListChunk::parse(&chunk.data)?;
                self.parse_list(&nested, 0)?;
            }
        }

        Ok(())
    }

    /// Parse hdrl (header list)
    fn parse_hdrl(&mut self, list: &ListChunk) -> Result<()> {
        // Find and parse avih chunk
        if let Some(avih) = list.find_chunk(chunk_ids::AVIH) {
            self.parse_avih(&avih.data)?;
        } else {
            return Err(AviError::MissingChunk("avih"));
        }

        Ok(())
    }

    /// Parse avih (main AVI header)
    fn parse_avih(&mut self, data: &[u8]) -> Result<()> {
        if data.len() < 56 {
            return Err(AviError::InvalidChunk {
                id: *b"avih",
                message: "Header too short".into(),
            });
        }

        let mut cursor = Cursor::new(data);

        self.header = AviHeader {
            microseconds_per_frame: cursor.read_u32::<LittleEndian>()?,
            max_bytes_per_sec: cursor.read_u32::<LittleEndian>()?,
            padding_granularity: cursor.read_u32::<LittleEndian>()?,
            flags: AviFlags::from_u32(cursor.read_u32::<LittleEndian>()?),
            total_frames: cursor.read_u32::<LittleEndian>()?,
            initial_frames: cursor.read_u32::<LittleEndian>()?,
            streams: cursor.read_u32::<LittleEndian>()?,
            suggested_buffer_size: cursor.read_u32::<LittleEndian>()?,
            width: cursor.read_u32::<LittleEndian>()?,
            height: cursor.read_u32::<LittleEndian>()?,
        };

        log::debug!(
            "AVI header: {}x{}, {} frames, {:.2} fps",
            self.header.width,
            self.header.height,
            self.header.total_frames,
            self.header.frame_rate()
        );

        Ok(())
    }

    /// Parse strl (stream list)
    fn parse_strl(&mut self, list: &ListChunk) -> Result<()> {
        let stream_index = self.streams.len() as u32;

        // Parse strh (stream header)
        let strh = list
            .find_chunk(chunk_ids::STRH)
            .ok_or(AviError::MissingChunk("strh"))?;
        let stream_header = self.parse_strh(&strh.data)?;

        // Parse strf (stream format)
        let strf = list
            .find_chunk(chunk_ids::STRF)
            .ok_or(AviError::MissingChunk("strf"))?;

        let (video_format, audio_format) = match stream_header.stream_type {
            StreamType::Video => (Some(self.parse_video_format(&strf.data)?), None),
            StreamType::Audio => (None, Some(self.parse_audio_format(&strf.data)?)),
            _ => (None, None),
        };

        // Parse strn (stream name) if present
        let name = list.find_chunk(chunk_ids::STRN).map(|chunk| {
            String::from_utf8_lossy(&chunk.data)
                .trim_end_matches('\0')
                .to_string()
        });

        self.streams.push(StreamInfo {
            index: stream_index,
            header: stream_header,
            video_format,
            audio_format,
            name,
        });

        log::debug!("Added stream {}: {:?}", stream_index, self.streams.last());

        Ok(())
    }

    /// Parse strh (stream header)
    fn parse_strh(&self, data: &[u8]) -> Result<StreamHeader> {
        if data.len() < 56 {
            return Err(AviError::InvalidChunk {
                id: *b"strh",
                message: "Stream header too short".into(),
            });
        }

        let mut cursor = Cursor::new(data);

        let mut type_bytes = [0u8; 4];
        cursor.read_exact(&mut type_bytes)?;

        let mut handler = [0u8; 4];
        cursor.read_exact(&mut handler)?;

        Ok(StreamHeader {
            stream_type: StreamType::from_fourcc(&type_bytes),
            handler,
            flags: cursor.read_u32::<LittleEndian>()?,
            priority: cursor.read_u16::<LittleEndian>()?,
            language: cursor.read_u16::<LittleEndian>()?,
            initial_frames: cursor.read_u32::<LittleEndian>()?,
            scale: cursor.read_u32::<LittleEndian>()?,
            rate: cursor.read_u32::<LittleEndian>()?,
            start: cursor.read_u32::<LittleEndian>()?,
            length: cursor.read_u32::<LittleEndian>()?,
            suggested_buffer_size: cursor.read_u32::<LittleEndian>()?,
            quality: cursor.read_u32::<LittleEndian>()?,
            sample_size: cursor.read_u32::<LittleEndian>()?,
            frame: Rect {
                left: cursor.read_i16::<LittleEndian>()?,
                top: cursor.read_i16::<LittleEndian>()?,
                right: cursor.read_i16::<LittleEndian>()?,
                bottom: cursor.read_i16::<LittleEndian>()?,
            },
        })
    }

    /// Parse video format (BITMAPINFOHEADER)
    fn parse_video_format(&self, data: &[u8]) -> Result<VideoFormat> {
        if data.len() < 40 {
            return Err(AviError::InvalidChunk {
                id: *b"strf",
                message: "Video format too short".into(),
            });
        }

        let mut cursor = Cursor::new(data);

        let size = cursor.read_u32::<LittleEndian>()?;
        let width = cursor.read_i32::<LittleEndian>()?;
        let height = cursor.read_i32::<LittleEndian>()?;
        let planes = cursor.read_u16::<LittleEndian>()?;
        let bit_count = cursor.read_u16::<LittleEndian>()?;

        let mut compression = [0u8; 4];
        cursor.read_exact(&mut compression)?;

        Ok(VideoFormat {
            size,
            width,
            height,
            planes,
            bit_count,
            compression,
            image_size: cursor.read_u32::<LittleEndian>()?,
            x_pels_per_meter: cursor.read_i32::<LittleEndian>()?,
            y_pels_per_meter: cursor.read_i32::<LittleEndian>()?,
            colors_used: cursor.read_u32::<LittleEndian>()?,
            colors_important: cursor.read_u32::<LittleEndian>()?,
        })
    }

    /// Parse audio format (WAVEFORMATEX)
    fn parse_audio_format(&self, data: &[u8]) -> Result<AudioFormat> {
        if data.len() < 16 {
            return Err(AviError::InvalidChunk {
                id: *b"strf",
                message: "Audio format too short".into(),
            });
        }

        let mut cursor = Cursor::new(data);

        let format_tag = cursor.read_u16::<LittleEndian>()?;
        let channels = cursor.read_u16::<LittleEndian>()?;
        let samples_per_sec = cursor.read_u32::<LittleEndian>()?;
        let avg_bytes_per_sec = cursor.read_u32::<LittleEndian>()?;
        let block_align = cursor.read_u16::<LittleEndian>()?;
        let bits_per_sample = cursor.read_u16::<LittleEndian>()?;

        let (extra_size, extra_data) = if data.len() >= 18 {
            let size = cursor.read_u16::<LittleEndian>()?;
            let mut extra = vec![0u8; size as usize];
            let _ = cursor.read(&mut extra);
            (size, extra)
        } else {
            (0, Vec::new())
        };

        Ok(AudioFormat {
            format_tag,
            channels,
            samples_per_sec,
            avg_bytes_per_sec,
            block_align,
            bits_per_sample,
            extra_size,
            extra_data,
        })
    }

    /// Parse ODML extension
    fn parse_odml(&mut self, list: &ListChunk) -> Result<()> {
        if let Some(dmlh) = list.find_chunk(chunk_ids::DMLH) {
            if dmlh.data.len() >= 4 {
                let total = u32::from_le_bytes([
                    dmlh.data[0],
                    dmlh.data[1],
                    dmlh.data[2],
                    dmlh.data[3],
                ]);
                self.odml_total_frames = Some(total as u64);
                log::debug!("ODML total frames: {}", total);
            }
        }
        Ok(())
    }

    /// Get AVI header
    pub fn header(&self) -> &AviHeader {
        &self.header
    }

    /// Get stream count
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    /// Get stream info
    pub fn stream(&self, index: usize) -> Option<&StreamInfo> {
        self.streams.get(index)
    }

    /// Get all streams
    pub fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    /// Get video stream (first one)
    pub fn video_stream(&self) -> Option<&StreamInfo> {
        self.streams.iter().find(|s| s.is_video())
    }

    /// Get audio stream (first one)
    pub fn audio_stream(&self) -> Option<&StreamInfo> {
        self.streams.iter().find(|s| s.is_audio())
    }

    /// Get total frames (including ODML extension)
    pub fn total_frames(&self) -> u64 {
        self.odml_total_frames
            .unwrap_or(self.header.total_frames as u64)
    }

    /// Get duration in seconds
    pub fn duration(&self) -> f64 {
        self.header.duration()
    }

    /// Get index entries
    pub fn index(&self) -> &[IndexEntry] {
        &self.index
    }

    /// Read next packet
    pub fn read_packet(&mut self) -> Result<Option<AviPacket>> {
        // Try reading from index first if available
        if !self.index.is_empty() {
            return self.read_packet_indexed();
        }

        // Sequential read from movi
        self.read_packet_sequential()
    }

    /// Read packet using index
    fn read_packet_indexed(&mut self) -> Result<Option<AviPacket>> {
        let total_packets: usize = self.frame_counters.iter().map(|&c| c as usize).sum();

        if total_packets >= self.index.len() {
            return Ok(None);
        }

        let entry = &self.index[total_packets];
        let chunk_id = ChunkId::parse(entry.chunk_id);

        if let Some(stream_num) = chunk_id.stream_number {
            let stream_idx = stream_num as usize;
            if stream_idx >= self.streams.len() {
                return Err(AviError::InvalidStream(stream_num as u32));
            }

            // Calculate actual offset (idx1 offsets are relative to movi)
            let actual_offset = self.movi_offset + entry.offset as usize + 8;

            if actual_offset + entry.size as usize > self.data.len() {
                return Err(AviError::InsufficientData {
                    needed: entry.size as usize,
                    available: self.data.len().saturating_sub(actual_offset),
                });
            }

            let data = self.data[actual_offset..actual_offset + entry.size as usize].to_vec();

            let timestamp = self.frame_counters[stream_idx];
            self.frame_counters[stream_idx] += 1;

            return Ok(Some(AviPacket {
                stream_index: stream_num as u32,
                data,
                timestamp,
                duration: 1,
                keyframe: entry.is_keyframe(),
            }));
        }

        // Skip non-stream entries
        Ok(None)
    }

    /// Read packet sequentially
    fn read_packet_sequential(&mut self) -> Result<Option<AviPacket>> {
        // Find movi boundaries
        if self.movi_offset == 0 {
            return Ok(None);
        }

        loop {
            if self.current_offset + 8 > self.data.len() {
                return Ok(None);
            }

            let (chunk, next_offset) = match RiffChunk::read(self.data, self.current_offset) {
                Ok(r) => r,
                Err(_) => return Ok(None),
            };

            self.current_offset = next_offset;

            // Check if it's a stream chunk
            let avi_chunk = AviChunk::from_raw(chunk);
            if let Some(stream_num) = avi_chunk.chunk_id.stream_number {
                let stream_idx = stream_num as usize;
                if stream_idx >= self.streams.len() {
                    continue;
                }

                let timestamp = self.frame_counters[stream_idx];
                self.frame_counters[stream_idx] += 1;

                // Check keyframe before consuming the chunk
                let keyframe = avi_chunk.is_keyframe();

                return Ok(Some(AviPacket {
                    stream_index: stream_num as u32,
                    data: avi_chunk.raw.data,
                    timestamp,
                    duration: 1,
                    keyframe,
                }));
            }

            // Check for end of movi
            if avi_chunk.raw.id == chunk_ids::IDX1 {
                return Ok(None);
            }
        }
    }

    /// Seek to timestamp
    pub fn seek(&mut self, stream_index: u32, timestamp: u64) -> Result<()> {
        if !self.index.is_empty() {
            // Find closest keyframe before timestamp
            let mut best_idx = 0;
            let mut frame_count = 0u64;

            for (idx, entry) in self.index.iter().enumerate() {
                let chunk_id = ChunkId::parse(entry.chunk_id);
                if chunk_id.stream_number == Some(stream_index as u16) {
                    if frame_count <= timestamp && entry.is_keyframe() {
                        best_idx = idx;
                    }
                    frame_count += 1;
                    if frame_count > timestamp {
                        break;
                    }
                }
            }

            // Reset counters and position
            for counter in &mut self.frame_counters {
                *counter = 0;
            }

            // Count frames up to best_idx
            for entry in &self.index[..best_idx] {
                let chunk_id = ChunkId::parse(entry.chunk_id);
                if let Some(stream_num) = chunk_id.stream_number {
                    if (stream_num as usize) < self.frame_counters.len() {
                        self.frame_counters[stream_num as usize] += 1;
                    }
                }
            }
        }

        Ok(())
    }

    /// Reset to beginning
    pub fn reset(&mut self) {
        self.current_offset = self.movi_offset;
        for counter in &mut self.frame_counters {
            *counter = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_minimal_avi() -> Vec<u8> {
        let mut data = Vec::new();

        // RIFF header
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&[0u8; 4]); // Size placeholder
        data.extend_from_slice(b"AVI ");

        // hdrl LIST
        let mut hdrl = Vec::new();
        hdrl.extend_from_slice(b"hdrl");

        // avih chunk (56 bytes)
        let mut avih = Vec::new();
        avih.extend_from_slice(&33333u32.to_le_bytes()); // microseconds per frame
        avih.extend_from_slice(&10_000_000u32.to_le_bytes()); // max bytes per sec
        avih.extend_from_slice(&0u32.to_le_bytes()); // padding
        avih.extend_from_slice(&0x110u32.to_le_bytes()); // flags
        avih.extend_from_slice(&100u32.to_le_bytes()); // total frames
        avih.extend_from_slice(&0u32.to_le_bytes()); // initial frames
        avih.extend_from_slice(&1u32.to_le_bytes()); // streams
        avih.extend_from_slice(&1_000_000u32.to_le_bytes()); // suggested buffer
        avih.extend_from_slice(&640u32.to_le_bytes()); // width
        avih.extend_from_slice(&480u32.to_le_bytes()); // height
        avih.extend_from_slice(&[0u8; 16]); // reserved

        hdrl.extend_from_slice(b"avih");
        hdrl.extend_from_slice(&(avih.len() as u32).to_le_bytes());
        hdrl.extend_from_slice(&avih);

        // strl LIST
        let mut strl = Vec::new();
        strl.extend_from_slice(b"strl");

        // strh chunk (56 bytes)
        let mut strh = Vec::new();
        strh.extend_from_slice(b"vids"); // stream type
        strh.extend_from_slice(b"H264"); // handler
        strh.extend_from_slice(&0u32.to_le_bytes()); // flags
        strh.extend_from_slice(&0u16.to_le_bytes()); // priority
        strh.extend_from_slice(&0u16.to_le_bytes()); // language
        strh.extend_from_slice(&0u32.to_le_bytes()); // initial frames
        strh.extend_from_slice(&1u32.to_le_bytes()); // scale
        strh.extend_from_slice(&30u32.to_le_bytes()); // rate
        strh.extend_from_slice(&0u32.to_le_bytes()); // start
        strh.extend_from_slice(&100u32.to_le_bytes()); // length
        strh.extend_from_slice(&1_000_000u32.to_le_bytes()); // suggested buffer
        strh.extend_from_slice(&0u32.to_le_bytes()); // quality
        strh.extend_from_slice(&0u32.to_le_bytes()); // sample size
        strh.extend_from_slice(&[0u8; 8]); // frame rect

        strl.extend_from_slice(b"strh");
        strl.extend_from_slice(&(strh.len() as u32).to_le_bytes());
        strl.extend_from_slice(&strh);

        // strf chunk (40 bytes - BITMAPINFOHEADER)
        let mut strf = Vec::new();
        strf.extend_from_slice(&40u32.to_le_bytes()); // size
        strf.extend_from_slice(&640i32.to_le_bytes()); // width
        strf.extend_from_slice(&480i32.to_le_bytes()); // height
        strf.extend_from_slice(&1u16.to_le_bytes()); // planes
        strf.extend_from_slice(&24u16.to_le_bytes()); // bit count
        strf.extend_from_slice(b"H264"); // compression
        strf.extend_from_slice(&0u32.to_le_bytes()); // image size
        strf.extend_from_slice(&0i32.to_le_bytes()); // x pels per meter
        strf.extend_from_slice(&0i32.to_le_bytes()); // y pels per meter
        strf.extend_from_slice(&0u32.to_le_bytes()); // colors used
        strf.extend_from_slice(&0u32.to_le_bytes()); // colors important

        strl.extend_from_slice(b"strf");
        strl.extend_from_slice(&(strf.len() as u32).to_le_bytes());
        strl.extend_from_slice(&strf);

        // Add strl LIST to hdrl
        hdrl.extend_from_slice(b"LIST");
        hdrl.extend_from_slice(&(strl.len() as u32).to_le_bytes());
        hdrl.extend_from_slice(&strl);

        // Add hdrl LIST to data
        data.extend_from_slice(b"LIST");
        data.extend_from_slice(&(hdrl.len() as u32).to_le_bytes());
        data.extend_from_slice(&hdrl);

        // movi LIST
        let mut movi = Vec::new();
        movi.extend_from_slice(b"movi");

        // Add a video chunk
        let video_data = vec![0u8; 100];
        movi.extend_from_slice(b"00dc");
        movi.extend_from_slice(&(video_data.len() as u32).to_le_bytes());
        movi.extend_from_slice(&video_data);

        data.extend_from_slice(b"LIST");
        data.extend_from_slice(&(movi.len() as u32).to_le_bytes());
        data.extend_from_slice(&movi);

        // Update RIFF size
        let riff_size = (data.len() - 8) as u32;
        data[4..8].copy_from_slice(&riff_size.to_le_bytes());

        data
    }

    #[test]
    fn test_demuxer_parse() {
        let data = create_minimal_avi();
        let demuxer = AviDemuxer::new(&data).unwrap();

        assert_eq!(demuxer.header().width, 640);
        assert_eq!(demuxer.header().height, 480);
        assert_eq!(demuxer.header().total_frames, 100);
        assert_eq!(demuxer.stream_count(), 1);
    }

    #[test]
    fn test_stream_info() {
        let data = create_minimal_avi();
        let demuxer = AviDemuxer::new(&data).unwrap();

        let stream = demuxer.video_stream().unwrap();
        assert!(stream.is_video());
        assert!(!stream.is_audio());
        assert_eq!(stream.frame_rate(), 30.0);
    }

    #[test]
    fn test_invalid_riff() {
        // Data must be at least 12 bytes to trigger InvalidRiff (not InsufficientData)
        let data = b"NOT_RIFF____";
        let result = AviDemuxer::new(data);
        assert!(matches!(result, Err(AviError::InvalidRiff)));
    }

    #[test]
    fn test_invalid_avi() {
        let mut data = Vec::new();
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&100u32.to_le_bytes());
        data.extend_from_slice(b"WAVE");

        let result = AviDemuxer::new(&data);
        assert!(matches!(result, Err(AviError::InvalidAvi)));
    }

    #[test]
    fn test_read_packet() {
        let data = create_minimal_avi();
        let mut demuxer = AviDemuxer::new(&data).unwrap();

        // Read the video packet
        let packet = demuxer.read_packet().unwrap();
        assert!(packet.is_some());

        let pkt = packet.unwrap();
        assert_eq!(pkt.stream_index, 0);
        assert_eq!(pkt.data.len(), 100);
    }

    #[test]
    fn test_demuxer_reset() {
        let data = create_minimal_avi();
        let mut demuxer = AviDemuxer::new(&data).unwrap();

        // Read packet
        let _ = demuxer.read_packet();

        // Reset
        demuxer.reset();

        // Should be able to read again
        let packet = demuxer.read_packet().unwrap();
        assert!(packet.is_some());
    }
}
