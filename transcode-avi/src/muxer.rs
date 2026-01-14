//! AVI muxer

use crate::chunks::{chunk_ids, FourCC, IndexEntry, RiffChunk};
use crate::error::Result;
use crate::types::{AudioFormat, AviFlags, StreamType, VideoFormat};
use byteorder::{LittleEndian, WriteBytesExt};
use std::io::{Cursor, Seek, SeekFrom, Write};

/// Stream configuration for muxer
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Stream type
    pub stream_type: StreamType,
    /// Video format (required for video streams)
    pub video_format: Option<VideoFormat>,
    /// Audio format (required for audio streams)
    pub audio_format: Option<AudioFormat>,
    /// Codec FourCC
    pub codec: [u8; 4],
    /// Time base numerator (scale)
    pub time_base_num: u32,
    /// Time base denominator (rate)
    pub time_base_den: u32,
}

impl StreamConfig {
    /// Create video stream config
    pub fn video(width: u32, height: u32, fps: f64, codec: [u8; 4]) -> Self {
        StreamConfig {
            stream_type: StreamType::Video,
            video_format: Some(VideoFormat {
                size: 40,
                width: width as i32,
                height: height as i32,
                planes: 1,
                bit_count: 24,
                compression: codec,
                image_size: 0,
                x_pels_per_meter: 0,
                y_pels_per_meter: 0,
                colors_used: 0,
                colors_important: 0,
            }),
            audio_format: None,
            codec,
            time_base_num: 1000,
            time_base_den: (fps * 1000.0) as u32,
        }
    }

    /// Create audio stream config
    pub fn audio(
        channels: u16,
        sample_rate: u32,
        bits_per_sample: u16,
        codec: [u8; 4],
    ) -> Self {
        let block_align = channels * (bits_per_sample / 8);
        let avg_bytes_per_sec = sample_rate * block_align as u32;

        StreamConfig {
            stream_type: StreamType::Audio,
            video_format: None,
            audio_format: Some(AudioFormat {
                format_tag: 1, // PCM
                channels,
                samples_per_sec: sample_rate,
                avg_bytes_per_sec,
                block_align,
                bits_per_sample,
                extra_size: 0,
                extra_data: Vec::new(),
            }),
            codec,
            time_base_num: 1,
            time_base_den: sample_rate,
        }
    }
}

/// Muxer configuration
#[derive(Debug, Clone)]
pub struct MuxerConfig {
    /// Whether to write index
    pub write_index: bool,
    /// Maximum file size before AVIX extension
    pub max_riff_size: u64,
    /// Suggested buffer size
    pub suggested_buffer_size: u32,
}

impl Default for MuxerConfig {
    fn default() -> Self {
        MuxerConfig {
            write_index: true,
            max_riff_size: 1024 * 1024 * 1024, // 1GB
            suggested_buffer_size: 1_000_000,
        }
    }
}

/// Internal stream state
struct StreamState {
    config: StreamConfig,
    frame_count: u64,
    total_bytes: u64,
}

/// AVI muxer
pub struct AviMuxer<W: Write + Seek> {
    writer: W,
    config: MuxerConfig,
    streams: Vec<StreamState>,
    index_entries: Vec<IndexEntry>,
    movi_offset: u64,
    movi_size: u64,
    header_written: bool,
    finalized: bool,
}

impl<W: Write + Seek> AviMuxer<W> {
    /// Create new muxer
    pub fn new(writer: W, config: MuxerConfig) -> Self {
        AviMuxer {
            writer,
            config,
            streams: Vec::new(),
            index_entries: Vec::new(),
            movi_offset: 0,
            movi_size: 0,
            header_written: false,
            finalized: false,
        }
    }

    /// Add a stream
    pub fn add_stream(&mut self, config: StreamConfig) -> Result<u32> {
        if self.header_written {
            log::warn!("Cannot add stream after header is written");
            return Ok(self.streams.len() as u32);
        }

        let stream_id = self.streams.len() as u32;
        self.streams.push(StreamState {
            config,
            frame_count: 0,
            total_bytes: 0,
        });

        Ok(stream_id)
    }

    /// Write header
    pub fn write_header(&mut self) -> Result<()> {
        if self.header_written {
            return Ok(());
        }

        // Calculate header size for proper offset (unused but kept for documentation)
        let _header_size = self.calculate_header_size();

        // Write RIFF header with placeholder size
        self.writer.write_all(b"RIFF")?;
        self.writer.write_u32::<LittleEndian>(0)?; // Size placeholder
        self.writer.write_all(b"AVI ")?;

        // Write hdrl LIST
        self.write_hdrl_list()?;

        // Write JUNK chunk for padding (align to 2KB boundary)
        let current_pos = self.writer.stream_position()?;
        let target_pos = current_pos.div_ceil(2048) * 2048;
        let junk_size = target_pos - current_pos - 8;
        if junk_size > 0 {
            self.writer.write_all(b"JUNK")?;
            self.writer.write_u32::<LittleEndian>(junk_size as u32)?;
            self.writer.write_all(&vec![0u8; junk_size as usize])?;
        }

        // Start movi LIST
        self.movi_offset = self.writer.stream_position()?;
        self.writer.write_all(b"LIST")?;
        self.writer.write_u32::<LittleEndian>(0)?; // Size placeholder
        self.writer.write_all(b"movi")?;

        self.header_written = true;
        log::debug!("Header written, movi starts at {}", self.movi_offset);

        Ok(())
    }

    /// Calculate header size
    fn calculate_header_size(&self) -> u64 {
        // RIFF header: 12 bytes
        // hdrl LIST overhead: 12 bytes
        // avih: 8 + 56 = 64 bytes
        // Per stream: strl LIST (12) + strh (64) + strf (48-52) ≈ 128 bytes
        12 + 12 + 64 + (self.streams.len() as u64 * 128)
    }

    /// Write hdrl LIST
    fn write_hdrl_list(&mut self) -> Result<()> {
        let hdrl_start = self.writer.stream_position()?;

        // LIST header with placeholder
        self.writer.write_all(b"LIST")?;
        self.writer.write_u32::<LittleEndian>(0)?; // Size placeholder
        self.writer.write_all(b"hdrl")?;

        // Write avih chunk
        self.write_avih()?;

        // Write strl LIST for each stream
        for i in 0..self.streams.len() {
            self.write_strl(i)?;
        }

        // Update LIST size
        let hdrl_end = self.writer.stream_position()?;
        let hdrl_size = hdrl_end - hdrl_start - 8;
        self.writer.seek(SeekFrom::Start(hdrl_start + 4))?;
        self.writer.write_u32::<LittleEndian>(hdrl_size as u32)?;
        self.writer.seek(SeekFrom::Start(hdrl_end))?;

        Ok(())
    }

    /// Write avih chunk
    fn write_avih(&mut self) -> Result<()> {
        let mut avih_data = Cursor::new(Vec::new());

        // Calculate frame rate from first video stream
        let (microseconds_per_frame, width, height) = self
            .streams
            .iter()
            .find(|s| s.config.stream_type == StreamType::Video)
            .map(|s| {
                let fps = s.config.time_base_den as f64 / s.config.time_base_num as f64;
                let usec = if fps > 0.0 {
                    (1_000_000.0 / fps) as u32
                } else {
                    33333
                };
                let (w, h) = s
                    .config
                    .video_format
                    .as_ref()
                    .map(|vf| (vf.width.unsigned_abs(), vf.height.unsigned_abs()))
                    .unwrap_or((0, 0));
                (usec, w, h)
            })
            .unwrap_or((33333, 0, 0));

        avih_data.write_u32::<LittleEndian>(microseconds_per_frame)?;
        avih_data.write_u32::<LittleEndian>(10_000_000)?; // max bytes per sec
        avih_data.write_u32::<LittleEndian>(0)?; // padding granularity

        let flags = AviFlags {
            has_index: self.config.write_index,
            is_interleaved: true,
            ..Default::default()
        };
        avih_data.write_u32::<LittleEndian>(flags.to_u32())?;

        avih_data.write_u32::<LittleEndian>(0)?; // total frames (updated later)
        avih_data.write_u32::<LittleEndian>(0)?; // initial frames
        avih_data.write_u32::<LittleEndian>(self.streams.len() as u32)?;
        avih_data.write_u32::<LittleEndian>(self.config.suggested_buffer_size)?;
        avih_data.write_u32::<LittleEndian>(width)?;
        avih_data.write_u32::<LittleEndian>(height)?;
        avih_data.write_all(&[0u8; 16])?; // reserved

        let avih_chunk = RiffChunk::new(chunk_ids::AVIH, avih_data.into_inner());
        avih_chunk.write(&mut self.writer)?;

        Ok(())
    }

    /// Write strl LIST for a stream
    fn write_strl(&mut self, stream_index: usize) -> Result<()> {
        let strl_start = self.writer.stream_position()?;

        self.writer.write_all(b"LIST")?;
        self.writer.write_u32::<LittleEndian>(0)?; // Size placeholder
        self.writer.write_all(b"strl")?;

        // Write strh - clone the config to avoid borrow issues
        let config = self.streams[stream_index].config.clone();
        self.write_strh(&config)?;

        // Write strf
        self.write_strf(&config)?;

        // Update LIST size
        let strl_end = self.writer.stream_position()?;
        let strl_size = strl_end - strl_start - 8;
        self.writer.seek(SeekFrom::Start(strl_start + 4))?;
        self.writer.write_u32::<LittleEndian>(strl_size as u32)?;
        self.writer.seek(SeekFrom::Start(strl_end))?;

        Ok(())
    }

    /// Write strh chunk
    fn write_strh(&mut self, config: &StreamConfig) -> Result<()> {
        let mut strh_data = Cursor::new(Vec::new());

        strh_data.write_all(&config.stream_type.to_fourcc())?;
        strh_data.write_all(&config.codec)?;
        strh_data.write_u32::<LittleEndian>(0)?; // flags
        strh_data.write_u16::<LittleEndian>(0)?; // priority
        strh_data.write_u16::<LittleEndian>(0)?; // language
        strh_data.write_u32::<LittleEndian>(0)?; // initial frames
        strh_data.write_u32::<LittleEndian>(config.time_base_num)?;
        strh_data.write_u32::<LittleEndian>(config.time_base_den)?;
        strh_data.write_u32::<LittleEndian>(0)?; // start
        strh_data.write_u32::<LittleEndian>(0)?; // length (updated later)
        strh_data.write_u32::<LittleEndian>(self.config.suggested_buffer_size)?;
        strh_data.write_u32::<LittleEndian>(0)?; // quality
        strh_data.write_u32::<LittleEndian>(0)?; // sample size
        strh_data.write_all(&[0u8; 8])?; // frame rect

        let strh_chunk = RiffChunk::new(chunk_ids::STRH, strh_data.into_inner());
        strh_chunk.write(&mut self.writer)?;

        Ok(())
    }

    /// Write strf chunk
    fn write_strf(&mut self, config: &StreamConfig) -> Result<()> {
        let strf_data = match config.stream_type {
            StreamType::Video => self.create_video_format(config)?,
            StreamType::Audio => self.create_audio_format(config)?,
            _ => Vec::new(),
        };

        let strf_chunk = RiffChunk::new(chunk_ids::STRF, strf_data);
        strf_chunk.write(&mut self.writer)?;

        Ok(())
    }

    /// Create video format data
    fn create_video_format(&self, config: &StreamConfig) -> Result<Vec<u8>> {
        let vf = config.video_format.as_ref().unwrap();
        let mut data = Cursor::new(Vec::new());

        data.write_u32::<LittleEndian>(vf.size)?;
        data.write_i32::<LittleEndian>(vf.width)?;
        data.write_i32::<LittleEndian>(vf.height)?;
        data.write_u16::<LittleEndian>(vf.planes)?;
        data.write_u16::<LittleEndian>(vf.bit_count)?;
        data.write_all(&vf.compression)?;
        data.write_u32::<LittleEndian>(vf.image_size)?;
        data.write_i32::<LittleEndian>(vf.x_pels_per_meter)?;
        data.write_i32::<LittleEndian>(vf.y_pels_per_meter)?;
        data.write_u32::<LittleEndian>(vf.colors_used)?;
        data.write_u32::<LittleEndian>(vf.colors_important)?;

        Ok(data.into_inner())
    }

    /// Create audio format data
    fn create_audio_format(&self, config: &StreamConfig) -> Result<Vec<u8>> {
        let af = config.audio_format.as_ref().unwrap();
        let mut data = Cursor::new(Vec::new());

        data.write_u16::<LittleEndian>(af.format_tag)?;
        data.write_u16::<LittleEndian>(af.channels)?;
        data.write_u32::<LittleEndian>(af.samples_per_sec)?;
        data.write_u32::<LittleEndian>(af.avg_bytes_per_sec)?;
        data.write_u16::<LittleEndian>(af.block_align)?;
        data.write_u16::<LittleEndian>(af.bits_per_sample)?;

        if !af.extra_data.is_empty() {
            data.write_u16::<LittleEndian>(af.extra_size)?;
            data.write_all(&af.extra_data)?;
        }

        Ok(data.into_inner())
    }

    /// Write a packet
    pub fn write_packet(
        &mut self,
        stream_index: u32,
        data: &[u8],
        keyframe: bool,
    ) -> Result<()> {
        if !self.header_written {
            self.write_header()?;
        }

        if self.finalized {
            return Ok(());
        }

        let stream_idx = stream_index as usize;
        if stream_idx >= self.streams.len() {
            return Ok(());
        }

        // Calculate chunk ID
        let chunk_type = match self.streams[stream_idx].config.stream_type {
            StreamType::Video => b"dc",
            StreamType::Audio => b"wb",
            _ => b"??",
        };

        let chunk_id = FourCC([
            b'0' + (stream_index / 10) as u8,
            b'0' + (stream_index % 10) as u8,
            chunk_type[0],
            chunk_type[1],
        ]);

        // Calculate offset relative to movi
        let current_pos = self.writer.stream_position()?;
        let offset = (current_pos - self.movi_offset - 8) as u32;

        // Write chunk
        let chunk = RiffChunk::new(chunk_id, data.to_vec());
        chunk.write(&mut self.writer)?;

        // Add index entry
        if self.config.write_index {
            self.index_entries.push(IndexEntry {
                chunk_id,
                flags: if keyframe { IndexEntry::KEYFRAME } else { 0 },
                offset,
                size: data.len() as u32,
            });
        }

        // Update stream state
        let stream = &mut self.streams[stream_idx];
        stream.frame_count += 1;
        stream.total_bytes += data.len() as u64;
        self.movi_size += chunk.total_size() as u64;

        Ok(())
    }

    /// Write video frame
    pub fn write_video_frame(
        &mut self,
        stream_index: u32,
        data: &[u8],
        keyframe: bool,
    ) -> Result<()> {
        self.write_packet(stream_index, data, keyframe)
    }

    /// Write audio samples
    pub fn write_audio_samples(&mut self, stream_index: u32, data: &[u8]) -> Result<()> {
        self.write_packet(stream_index, data, true)
    }

    /// Finalize the file
    pub fn finalize(&mut self) -> Result<()> {
        if self.finalized {
            return Ok(());
        }

        if !self.header_written {
            self.write_header()?;
        }

        // Update movi LIST size
        let movi_end = self.writer.stream_position()?;
        let movi_content_size = movi_end - self.movi_offset - 8;
        self.writer.seek(SeekFrom::Start(self.movi_offset + 4))?;
        self.writer.write_u32::<LittleEndian>(movi_content_size as u32)?;
        self.writer.seek(SeekFrom::Start(movi_end))?;

        // Write idx1 index
        if self.config.write_index && !self.index_entries.is_empty() {
            self.write_index()?;
        }

        // Update RIFF size
        let file_end = self.writer.stream_position()?;
        let riff_size = file_end - 8;
        self.writer.seek(SeekFrom::Start(4))?;
        self.writer.write_u32::<LittleEndian>(riff_size as u32)?;

        // Update avih total frames
        self.update_avih()?;

        self.finalized = true;
        log::debug!("AVI finalized, total size: {}", file_end);

        Ok(())
    }

    /// Write index
    fn write_index(&mut self) -> Result<()> {
        self.writer.write_all(b"idx1")?;
        let idx_size = self.index_entries.len() * 16;
        self.writer.write_u32::<LittleEndian>(idx_size as u32)?;

        for entry in &self.index_entries {
            entry.write(&mut self.writer)?;
        }

        Ok(())
    }

    /// Update avih chunk with final values
    fn update_avih(&mut self) -> Result<()> {
        // Find total video frames
        let total_frames: u64 = self
            .streams
            .iter()
            .filter(|s| s.config.stream_type == StreamType::Video)
            .map(|s| s.frame_count)
            .sum();

        // avih chunk is at offset 12 (RIFF header) + 12 (LIST hdrl header) + 8 (avih chunk header)
        // Total frames is at offset 16 within avih data
        let avih_total_frames_offset = 12 + 12 + 8 + 16;

        let current = self.writer.stream_position()?;
        self.writer.seek(SeekFrom::Start(avih_total_frames_offset))?;
        self.writer.write_u32::<LittleEndian>(total_frames as u32)?;
        self.writer.seek(SeekFrom::Start(current))?;

        Ok(())
    }

    /// Get frame count for stream
    pub fn frame_count(&self, stream_index: u32) -> u64 {
        self.streams
            .get(stream_index as usize)
            .map(|s| s.frame_count)
            .unwrap_or(0)
    }

    /// Get total bytes written
    pub fn total_bytes(&self) -> u64 {
        self.streams.iter().map(|s| s.total_bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_stream_config_video() {
        let config = StreamConfig::video(1920, 1080, 30.0, *b"H264");

        assert_eq!(config.stream_type, StreamType::Video);
        assert!(config.video_format.is_some());
        assert!(config.audio_format.is_none());

        let vf = config.video_format.unwrap();
        assert_eq!(vf.width, 1920);
        assert_eq!(vf.height, 1080);
    }

    #[test]
    fn test_stream_config_audio() {
        let config = StreamConfig::audio(2, 44100, 16, [1, 0, 0, 0]);

        assert_eq!(config.stream_type, StreamType::Audio);
        assert!(config.video_format.is_none());
        assert!(config.audio_format.is_some());

        let af = config.audio_format.unwrap();
        assert_eq!(af.channels, 2);
        assert_eq!(af.samples_per_sec, 44100);
        assert_eq!(af.bits_per_sample, 16);
    }

    #[test]
    fn test_muxer_create() {
        let buffer = Cursor::new(Vec::new());
        let config = MuxerConfig::default();
        let muxer: AviMuxer<Cursor<Vec<u8>>> = AviMuxer::new(buffer, config);

        assert_eq!(muxer.streams.len(), 0);
        assert!(!muxer.header_written);
        assert!(!muxer.finalized);
    }

    #[test]
    fn test_muxer_add_stream() {
        let buffer = Cursor::new(Vec::new());
        let config = MuxerConfig::default();
        let mut muxer = AviMuxer::new(buffer, config);

        let video_config = StreamConfig::video(640, 480, 30.0, *b"H264");
        let stream_id = muxer.add_stream(video_config).unwrap();

        assert_eq!(stream_id, 0);
        assert_eq!(muxer.streams.len(), 1);
    }

    #[test]
    fn test_muxer_write_header() {
        let buffer = Cursor::new(Vec::new());
        let config = MuxerConfig::default();
        let mut muxer = AviMuxer::new(buffer, config);

        let video_config = StreamConfig::video(640, 480, 30.0, *b"H264");
        muxer.add_stream(video_config).unwrap();

        muxer.write_header().unwrap();

        assert!(muxer.header_written);
    }

    #[test]
    fn test_muxer_write_packet() {
        let buffer = Cursor::new(Vec::new());
        let config = MuxerConfig::default();
        let mut muxer = AviMuxer::new(buffer, config);

        let video_config = StreamConfig::video(640, 480, 30.0, *b"H264");
        muxer.add_stream(video_config).unwrap();

        let frame_data = vec![0u8; 1000];
        muxer.write_packet(0, &frame_data, true).unwrap();

        assert_eq!(muxer.frame_count(0), 1);
        assert_eq!(muxer.total_bytes(), 1000);
    }

    #[test]
    fn test_muxer_finalize() {
        let buffer = Cursor::new(Vec::new());
        let config = MuxerConfig::default();
        let mut muxer = AviMuxer::new(buffer, config);

        let video_config = StreamConfig::video(640, 480, 30.0, *b"H264");
        muxer.add_stream(video_config).unwrap();

        let frame_data = vec![0u8; 1000];
        muxer.write_packet(0, &frame_data, true).unwrap();

        muxer.finalize().unwrap();

        assert!(muxer.finalized);
    }

    #[test]
    fn test_muxer_full_workflow() {
        let buffer = Cursor::new(Vec::new());
        let config = MuxerConfig::default();
        let mut muxer = AviMuxer::new(buffer, config);

        // Add video stream
        let video_config = StreamConfig::video(320, 240, 25.0, *b"XVID");
        muxer.add_stream(video_config).unwrap();

        // Add audio stream
        let audio_config = StreamConfig::audio(2, 44100, 16, [1, 0, 0, 0]);
        muxer.add_stream(audio_config).unwrap();

        // Write some frames
        for i in 0..10 {
            let video_data = vec![i as u8; 500];
            muxer.write_video_frame(0, &video_data, i == 0).unwrap();

            let audio_data = vec![i as u8; 100];
            muxer.write_audio_samples(1, &audio_data).unwrap();
        }

        // Finalize
        muxer.finalize().unwrap();

        assert_eq!(muxer.frame_count(0), 10);
        assert_eq!(muxer.frame_count(1), 10);

        // Verify output is valid RIFF/AVI
        let output = muxer.writer.into_inner();
        assert!(output.len() > 100);
        assert_eq!(&output[0..4], b"RIFF");
        assert_eq!(&output[8..12], b"AVI ");
    }

    #[test]
    fn test_muxer_index_entries() {
        let buffer = Cursor::new(Vec::new());
        let config = MuxerConfig {
            write_index: true,
            ..Default::default()
        };
        let mut muxer = AviMuxer::new(buffer, config);

        let video_config = StreamConfig::video(640, 480, 30.0, *b"H264");
        muxer.add_stream(video_config).unwrap();

        // Write keyframe and non-keyframes
        muxer.write_packet(0, &[0; 100], true).unwrap();
        muxer.write_packet(0, &[0; 100], false).unwrap();
        muxer.write_packet(0, &[0; 100], false).unwrap();

        assert_eq!(muxer.index_entries.len(), 3);
        assert!(muxer.index_entries[0].is_keyframe());
        assert!(!muxer.index_entries[1].is_keyframe());
        assert!(!muxer.index_entries[2].is_keyframe());
    }
}
