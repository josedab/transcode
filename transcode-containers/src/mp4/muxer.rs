//! MP4 muxer implementation.

use super::{write_u32_be, write_u64_be, Mp4Brand};
use crate::chapters::{ChapterList, Mp4ChapterWriter};
use crate::traits::{CodecId, Muxer, StreamInfo, TrackType};
use transcode_core::error::{Error, Result};
use transcode_core::packet::Packet;
use std::io::{Seek, SeekFrom, Write};

/// Sample metadata.
#[derive(Debug, Clone)]
struct SampleInfo {
    /// Size in bytes.
    size: u32,
    /// Duration in timescale units.
    duration: u32,
    /// Composition time offset.
    cts_offset: i32,
    /// Is keyframe.
    keyframe: bool,
}

/// Chunk metadata.
#[derive(Debug, Clone)]
struct ChunkInfo {
    /// Offset in file.
    offset: u64,
    /// First sample index in chunk.
    #[allow(dead_code)]
    first_sample: usize,
    /// Number of samples.
    sample_count: usize,
}

/// Track muxing state.
struct TrackState {
    /// Stream info.
    stream: StreamInfo,
    /// Samples.
    samples: Vec<SampleInfo>,
    /// Chunks.
    chunks: Vec<ChunkInfo>,
    /// Current chunk samples.
    current_chunk_samples: Vec<u32>,
    /// Current chunk offset.
    current_chunk_offset: u64,
    /// Total duration in timescale units.
    duration: u64,
    /// Timescale.
    timescale: u32,
}

impl TrackState {
    fn new(stream: StreamInfo) -> Self {
        let timescale = if stream.track_type == TrackType::Video {
            // For video, use a high timescale for accurate frame times
            90000
        } else {
            // For audio, use the sample rate
            stream.audio.as_ref().map(|a| a.sample_rate).unwrap_or(48000)
        };

        Self {
            stream,
            samples: Vec::new(),
            chunks: Vec::new(),
            current_chunk_samples: Vec::new(),
            current_chunk_offset: 0,
            duration: 0,
            timescale,
        }
    }
}

/// MP4 muxer.
pub struct Mp4Muxer {
    /// Writer.
    writer: Option<Box<dyn WriteSeek>>,
    /// Tracks.
    tracks: Vec<TrackState>,
    /// mdat start offset.
    mdat_start: u64,
    /// mdat size.
    mdat_size: u64,
    /// Header written.
    header_written: bool,
    /// Major brand.
    major_brand: Mp4Brand,
    /// Chapters to embed (Nero format in udta).
    chapters: Option<ChapterList>,
}

trait WriteSeek: Write + Seek + Send {}
impl<T: Write + Seek + Send> WriteSeek for T {}

impl Mp4Muxer {
    /// Create a new MP4 muxer.
    pub fn new() -> Self {
        Self {
            writer: None,
            tracks: Vec::new(),
            mdat_start: 0,
            mdat_size: 0,
            header_written: false,
            major_brand: Mp4Brand::Isom,
            chapters: None,
        }
    }

    /// Set the major brand.
    pub fn set_brand(&mut self, brand: Mp4Brand) {
        self.major_brand = brand;
    }

    /// Set chapters to embed in the output file.
    ///
    /// Chapters will be written as Nero chapters (chpl atom in udta).
    /// This must be called before `write_trailer`.
    pub fn set_chapters(&mut self, chapters: ChapterList) {
        self.chapters = Some(chapters);
    }

    /// Get a reference to the chapters that will be written.
    pub fn chapters(&self) -> Option<&ChapterList> {
        self.chapters.as_ref()
    }

    /// Take ownership of the chapters.
    pub fn take_chapters(&mut self) -> Option<ChapterList> {
        self.chapters.take()
    }

    /// Write ftyp atom.
    fn write_ftyp(&mut self) -> Result<()> {
        let writer = self.writer.as_mut().ok_or(Error::Container("No writer".into()))?;

        let mut compatible = vec![
            Mp4Brand::Isom.to_bytes(),
            Mp4Brand::Iso2.to_bytes(),
            Mp4Brand::Mp41.to_bytes(),
        ];

        // Add codec-specific brands
        for track in &self.tracks {
            if track.stream.codec_id == CodecId::H264
                && !compatible.contains(&Mp4Brand::Avc1.to_bytes())
            {
                compatible.push(Mp4Brand::Avc1.to_bytes());
            }
        }

        let size = 8 + 4 + 4 + compatible.len() * 4;
        writer.write_all(&write_u32_be(size as u32))?;
        writer.write_all(b"ftyp")?;
        writer.write_all(&self.major_brand.to_bytes())?;
        writer.write_all(&write_u32_be(0x200))?; // minor version

        for brand in compatible {
            writer.write_all(&brand)?;
        }

        Ok(())
    }

    /// Start mdat atom.
    fn start_mdat(&mut self) -> Result<()> {
        let writer = self.writer.as_mut().ok_or(Error::Container("No writer".into()))?;

        self.mdat_start = writer.stream_position()?;

        // Write placeholder for 64-bit size
        writer.write_all(&[0, 0, 0, 1])?; // size = 1 means extended size
        writer.write_all(b"mdat")?;
        writer.write_all(&[0u8; 8])?; // placeholder for extended size

        Ok(())
    }

    /// Finish mdat atom.
    fn finish_mdat(&mut self) -> Result<()> {
        let writer = self.writer.as_mut().ok_or(Error::Container("No writer".into()))?;

        let current = writer.stream_position()?;
        self.mdat_size = current - self.mdat_start;

        // Update mdat size
        writer.seek(SeekFrom::Start(self.mdat_start + 8))?;
        writer.write_all(&write_u64_be(self.mdat_size))?;
        writer.seek(SeekFrom::Start(current))?;

        Ok(())
    }

    /// Write moov atom.
    fn write_moov(&mut self) -> Result<()> {
        let moov_data = self.build_moov()?;

        let writer = self.writer.as_mut().ok_or(Error::Container("No writer".into()))?;
        writer.write_all(&moov_data)?;

        Ok(())
    }

    /// Build moov atom data.
    fn build_moov(&self) -> Result<Vec<u8>> {
        let mut moov = Vec::new();

        // Calculate total duration
        let movie_timescale = 1000u32;
        let max_duration = self.tracks.iter()
            .map(|t| t.duration * movie_timescale as u64 / t.timescale as u64)
            .max()
            .unwrap_or(0);

        // mvhd
        let mvhd = self.build_mvhd(movie_timescale, max_duration);
        moov.extend_from_slice(&mvhd);

        // trak atoms
        for (i, track) in self.tracks.iter().enumerate() {
            let trak = self.build_trak(track, i as u32 + 1, movie_timescale)?;
            moov.extend_from_slice(&trak);
        }

        // udta with chapters (if present)
        if let Some(ref chapters) = self.chapters {
            if !chapters.is_empty() {
                let udta = Mp4ChapterWriter::build_udta_with_chapters(chapters);
                moov.extend_from_slice(&udta);
            }
        }

        // Wrap in moov box
        let mut result = Vec::with_capacity(moov.len() + 8);
        result.extend_from_slice(&write_u32_be((moov.len() + 8) as u32));
        result.extend_from_slice(b"moov");
        result.extend_from_slice(&moov);

        Ok(result)
    }

    /// Build mvhd atom.
    fn build_mvhd(&self, timescale: u32, duration: u64) -> Vec<u8> {
        let mut data = Vec::with_capacity(120);

        // Size and type (will be prepended)
        let version = if duration > u32::MAX as u64 { 1u8 } else { 0u8 };

        // Version and flags
        data.push(version);
        data.extend_from_slice(&[0, 0, 0]); // flags

        if version == 1 {
            data.extend_from_slice(&write_u64_be(0)); // creation time
            data.extend_from_slice(&write_u64_be(0)); // modification time
            data.extend_from_slice(&write_u32_be(timescale));
            data.extend_from_slice(&write_u64_be(duration));
        } else {
            data.extend_from_slice(&write_u32_be(0)); // creation time
            data.extend_from_slice(&write_u32_be(0)); // modification time
            data.extend_from_slice(&write_u32_be(timescale));
            data.extend_from_slice(&write_u32_be(duration as u32));
        }

        // Rate (1.0 = 0x00010000)
        data.extend_from_slice(&write_u32_be(0x00010000));
        // Volume (1.0 = 0x0100)
        data.extend_from_slice(&[0x01, 0x00]);
        // Reserved
        data.extend_from_slice(&[0u8; 10]);
        // Matrix (identity)
        data.extend_from_slice(&write_u32_be(0x00010000));
        data.extend_from_slice(&[0u8; 4]);
        data.extend_from_slice(&[0u8; 4]);
        data.extend_from_slice(&[0u8; 4]);
        data.extend_from_slice(&write_u32_be(0x00010000));
        data.extend_from_slice(&[0u8; 4]);
        data.extend_from_slice(&[0u8; 4]);
        data.extend_from_slice(&[0u8; 4]);
        data.extend_from_slice(&write_u32_be(0x40000000));
        // Pre-defined
        data.extend_from_slice(&[0u8; 24]);
        // Next track ID
        data.extend_from_slice(&write_u32_be(self.tracks.len() as u32 + 1));

        // Wrap in box
        let mut result = Vec::with_capacity(data.len() + 8);
        result.extend_from_slice(&write_u32_be((data.len() + 8) as u32));
        result.extend_from_slice(b"mvhd");
        result.extend_from_slice(&data);

        result
    }

    /// Build trak atom.
    fn build_trak(&self, track: &TrackState, track_id: u32, movie_timescale: u32) -> Result<Vec<u8>> {
        let mut trak = Vec::new();

        // tkhd
        let tkhd = self.build_tkhd(track, track_id, movie_timescale);
        trak.extend_from_slice(&tkhd);

        // mdia
        let mdia = self.build_mdia(track)?;
        trak.extend_from_slice(&mdia);

        // Wrap in trak box
        let mut result = Vec::with_capacity(trak.len() + 8);
        result.extend_from_slice(&write_u32_be((trak.len() + 8) as u32));
        result.extend_from_slice(b"trak");
        result.extend_from_slice(&trak);

        Ok(result)
    }

    /// Build tkhd atom.
    fn build_tkhd(&self, track: &TrackState, track_id: u32, movie_timescale: u32) -> Vec<u8> {
        let duration = track.duration * movie_timescale as u64 / track.timescale as u64;
        let version = if duration > u32::MAX as u64 { 1u8 } else { 0u8 };

        let mut data = Vec::new();
        data.push(version);
        data.extend_from_slice(&[0, 0, 0x03]); // flags: enabled, in movie

        if version == 1 {
            data.extend_from_slice(&write_u64_be(0)); // creation time
            data.extend_from_slice(&write_u64_be(0)); // modification time
            data.extend_from_slice(&write_u32_be(track_id));
            data.extend_from_slice(&[0u8; 4]); // reserved
            data.extend_from_slice(&write_u64_be(duration));
        } else {
            data.extend_from_slice(&write_u32_be(0)); // creation time
            data.extend_from_slice(&write_u32_be(0)); // modification time
            data.extend_from_slice(&write_u32_be(track_id));
            data.extend_from_slice(&[0u8; 4]); // reserved
            data.extend_from_slice(&write_u32_be(duration as u32));
        }

        // Reserved
        data.extend_from_slice(&[0u8; 8]);
        // Layer and alternate group
        data.extend_from_slice(&[0u8; 4]);
        // Volume (1.0 for audio)
        if track.stream.track_type == TrackType::Audio {
            data.extend_from_slice(&[0x01, 0x00]);
        } else {
            data.extend_from_slice(&[0, 0]);
        }
        // Reserved
        data.extend_from_slice(&[0u8; 2]);
        // Matrix (identity)
        data.extend_from_slice(&write_u32_be(0x00010000));
        data.extend_from_slice(&[0u8; 4]);
        data.extend_from_slice(&[0u8; 4]);
        data.extend_from_slice(&[0u8; 4]);
        data.extend_from_slice(&write_u32_be(0x00010000));
        data.extend_from_slice(&[0u8; 4]);
        data.extend_from_slice(&[0u8; 4]);
        data.extend_from_slice(&[0u8; 4]);
        data.extend_from_slice(&write_u32_be(0x40000000));

        // Width and height (fixed-point 16.16)
        if let Some(ref video) = track.stream.video {
            data.extend_from_slice(&write_u32_be(video.width << 16));
            data.extend_from_slice(&write_u32_be(video.height << 16));
        } else {
            data.extend_from_slice(&[0u8; 8]);
        }

        // Wrap in box
        let mut result = Vec::with_capacity(data.len() + 8);
        result.extend_from_slice(&write_u32_be((data.len() + 8) as u32));
        result.extend_from_slice(b"tkhd");
        result.extend_from_slice(&data);

        result
    }

    /// Build mdia atom.
    fn build_mdia(&self, track: &TrackState) -> Result<Vec<u8>> {
        let mut mdia = Vec::new();

        // mdhd
        let mdhd = self.build_mdhd(track);
        mdia.extend_from_slice(&mdhd);

        // hdlr
        let hdlr = self.build_hdlr(track);
        mdia.extend_from_slice(&hdlr);

        // minf
        let minf = self.build_minf(track)?;
        mdia.extend_from_slice(&minf);

        // Wrap in mdia box
        let mut result = Vec::with_capacity(mdia.len() + 8);
        result.extend_from_slice(&write_u32_be((mdia.len() + 8) as u32));
        result.extend_from_slice(b"mdia");
        result.extend_from_slice(&mdia);

        Ok(result)
    }

    /// Build mdhd atom.
    fn build_mdhd(&self, track: &TrackState) -> Vec<u8> {
        let version = if track.duration > u32::MAX as u64 { 1u8 } else { 0u8 };

        let mut data = Vec::new();
        data.push(version);
        data.extend_from_slice(&[0, 0, 0]); // flags

        if version == 1 {
            data.extend_from_slice(&write_u64_be(0)); // creation time
            data.extend_from_slice(&write_u64_be(0)); // modification time
            data.extend_from_slice(&write_u32_be(track.timescale));
            data.extend_from_slice(&write_u64_be(track.duration));
        } else {
            data.extend_from_slice(&write_u32_be(0)); // creation time
            data.extend_from_slice(&write_u32_be(0)); // modification time
            data.extend_from_slice(&write_u32_be(track.timescale));
            data.extend_from_slice(&write_u32_be(track.duration as u32));
        }

        // Language (undetermined)
        data.extend_from_slice(&[0x55, 0xC4]); // "und"
        // Pre-defined
        data.extend_from_slice(&[0, 0]);

        // Wrap in box
        let mut result = Vec::with_capacity(data.len() + 8);
        result.extend_from_slice(&write_u32_be((data.len() + 8) as u32));
        result.extend_from_slice(b"mdhd");
        result.extend_from_slice(&data);

        result
    }

    /// Build hdlr atom.
    fn build_hdlr(&self, track: &TrackState) -> Vec<u8> {
        let handler_type = match track.stream.track_type {
            TrackType::Video => b"vide",
            TrackType::Audio => b"soun",
            _ => b"hint",
        };

        let name = match track.stream.track_type {
            TrackType::Video => "VideoHandler",
            TrackType::Audio => "SoundHandler",
            _ => "Handler",
        };

        let mut data = Vec::new();
        data.extend_from_slice(&[0, 0, 0, 0]); // version and flags
        data.extend_from_slice(&[0u8; 4]); // pre-defined
        data.extend_from_slice(handler_type);
        data.extend_from_slice(&[0u8; 12]); // reserved
        data.extend_from_slice(name.as_bytes());
        data.push(0); // null terminator

        // Wrap in box
        let mut result = Vec::with_capacity(data.len() + 8);
        result.extend_from_slice(&write_u32_be((data.len() + 8) as u32));
        result.extend_from_slice(b"hdlr");
        result.extend_from_slice(&data);

        result
    }

    /// Build minf atom.
    fn build_minf(&self, track: &TrackState) -> Result<Vec<u8>> {
        let mut minf = Vec::new();

        // vmhd or smhd
        if track.stream.track_type == TrackType::Video {
            let vmhd = self.build_vmhd();
            minf.extend_from_slice(&vmhd);
        } else {
            let smhd = self.build_smhd();
            minf.extend_from_slice(&smhd);
        }

        // dinf
        let dinf = self.build_dinf();
        minf.extend_from_slice(&dinf);

        // stbl
        let stbl = self.build_stbl(track)?;
        minf.extend_from_slice(&stbl);

        // Wrap in minf box
        let mut result = Vec::with_capacity(minf.len() + 8);
        result.extend_from_slice(&write_u32_be((minf.len() + 8) as u32));
        result.extend_from_slice(b"minf");
        result.extend_from_slice(&minf);

        Ok(result)
    }

    /// Build vmhd atom.
    fn build_vmhd(&self) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&[0, 0, 0, 1]); // version and flags
        data.extend_from_slice(&[0u8; 8]); // graphics mode and opcolor

        let mut result = Vec::with_capacity(data.len() + 8);
        result.extend_from_slice(&write_u32_be((data.len() + 8) as u32));
        result.extend_from_slice(b"vmhd");
        result.extend_from_slice(&data);

        result
    }

    /// Build smhd atom.
    fn build_smhd(&self) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&[0, 0, 0, 0]); // version and flags
        data.extend_from_slice(&[0u8; 4]); // balance and reserved

        let mut result = Vec::with_capacity(data.len() + 8);
        result.extend_from_slice(&write_u32_be((data.len() + 8) as u32));
        result.extend_from_slice(b"smhd");
        result.extend_from_slice(&data);

        result
    }

    /// Build dinf atom.
    fn build_dinf(&self) -> Vec<u8> {
        // Build dref with url entry
        let mut dref = Vec::new();
        dref.extend_from_slice(&[0, 0, 0, 0]); // version and flags
        dref.extend_from_slice(&write_u32_be(1)); // entry count

        // url entry (self-contained)
        dref.extend_from_slice(&write_u32_be(12)); // size
        dref.extend_from_slice(b"url ");
        dref.extend_from_slice(&[0, 0, 0, 1]); // flags: self-contained

        // Wrap dref in box
        let mut dref_box = Vec::with_capacity(dref.len() + 8);
        dref_box.extend_from_slice(&write_u32_be((dref.len() + 8) as u32));
        dref_box.extend_from_slice(b"dref");
        dref_box.extend_from_slice(&dref);

        // Wrap in dinf box
        let mut result = Vec::with_capacity(dref_box.len() + 8);
        result.extend_from_slice(&write_u32_be((dref_box.len() + 8) as u32));
        result.extend_from_slice(b"dinf");
        result.extend_from_slice(&dref_box);

        result
    }

    /// Build stbl atom.
    fn build_stbl(&self, track: &TrackState) -> Result<Vec<u8>> {
        let mut stbl = Vec::new();

        // stsd
        let stsd = self.build_stsd(track)?;
        stbl.extend_from_slice(&stsd);

        // stts
        let stts = self.build_stts(track);
        stbl.extend_from_slice(&stts);

        // ctts (if needed)
        if track.samples.iter().any(|s| s.cts_offset != 0) {
            let ctts = self.build_ctts(track);
            stbl.extend_from_slice(&ctts);
        }

        // stss (keyframes)
        if track.stream.track_type == TrackType::Video {
            let stss = self.build_stss(track);
            stbl.extend_from_slice(&stss);
        }

        // stsc
        let stsc = self.build_stsc(track);
        stbl.extend_from_slice(&stsc);

        // stsz
        let stsz = self.build_stsz(track);
        stbl.extend_from_slice(&stsz);

        // stco or co64
        let stco = self.build_stco(track);
        stbl.extend_from_slice(&stco);

        // Wrap in stbl box
        let mut result = Vec::with_capacity(stbl.len() + 8);
        result.extend_from_slice(&write_u32_be((stbl.len() + 8) as u32));
        result.extend_from_slice(b"stbl");
        result.extend_from_slice(&stbl);

        Ok(result)
    }

    /// Build stsd atom.
    fn build_stsd(&self, track: &TrackState) -> Result<Vec<u8>> {
        let mut data = Vec::new();
        data.extend_from_slice(&[0, 0, 0, 0]); // version and flags
        data.extend_from_slice(&write_u32_be(1)); // entry count

        // Build sample entry
        let entry = self.build_sample_entry(track)?;
        data.extend_from_slice(&entry);

        // Wrap in stsd box
        let mut result = Vec::with_capacity(data.len() + 8);
        result.extend_from_slice(&write_u32_be((data.len() + 8) as u32));
        result.extend_from_slice(b"stsd");
        result.extend_from_slice(&data);

        Ok(result)
    }

    /// Build sample entry.
    fn build_sample_entry(&self, track: &TrackState) -> Result<Vec<u8>> {
        match track.stream.track_type {
            TrackType::Video => self.build_video_sample_entry(track),
            TrackType::Audio => self.build_audio_sample_entry(track),
            _ => Err(Error::Container("Unsupported track type".into())),
        }
    }

    /// Build video sample entry.
    fn build_video_sample_entry(&self, track: &TrackState) -> Result<Vec<u8>> {
        let video = track.stream.video.as_ref()
            .ok_or_else(|| Error::Container("Missing video info".into()))?;

        let fourcc = match track.stream.codec_id {
            CodecId::H264 => b"avc1",
            CodecId::H265 => b"hvc1",
            _ => return Err(Error::Container("Unsupported video codec".into())),
        };

        let mut data = Vec::new();
        data.extend_from_slice(&[0u8; 6]); // reserved
        data.extend_from_slice(&[0, 1]); // data reference index
        data.extend_from_slice(&[0u8; 16]); // pre-defined and reserved
        data.extend_from_slice(&(video.width as u16).to_be_bytes());
        data.extend_from_slice(&(video.height as u16).to_be_bytes());
        data.extend_from_slice(&write_u32_be(0x00480000)); // horiz resolution
        data.extend_from_slice(&write_u32_be(0x00480000)); // vert resolution
        data.extend_from_slice(&[0u8; 4]); // reserved
        data.extend_from_slice(&[0, 1]); // frame count
        data.extend_from_slice(&[0u8; 32]); // compressor name
        data.extend_from_slice(&[0, 0x18]); // depth (24-bit)
        data.extend_from_slice(&[0xFF, 0xFF]); // pre-defined

        // Add codec-specific box (avcC, hvcC, etc.)
        if let Some(ref extra) = track.stream.extra_data {
            let config_box_type = match track.stream.codec_id {
                CodecId::H264 => b"avcC",
                CodecId::H265 => b"hvcC",
                _ => return Err(Error::Container("Unsupported codec".into())),
            };

            data.extend_from_slice(&write_u32_be((extra.len() + 8) as u32));
            data.extend_from_slice(config_box_type);
            data.extend_from_slice(extra);
        }

        // Wrap in entry box
        let mut result = Vec::with_capacity(data.len() + 8);
        result.extend_from_slice(&write_u32_be((data.len() + 8) as u32));
        result.extend_from_slice(fourcc);
        result.extend_from_slice(&data);

        Ok(result)
    }

    /// Build audio sample entry.
    fn build_audio_sample_entry(&self, track: &TrackState) -> Result<Vec<u8>> {
        let audio = track.stream.audio.as_ref()
            .ok_or_else(|| Error::Container("Missing audio info".into()))?;

        let mut data = Vec::new();
        data.extend_from_slice(&[0u8; 6]); // reserved
        data.extend_from_slice(&[0, 1]); // data reference index
        data.extend_from_slice(&[0u8; 8]); // reserved
        data.extend_from_slice(&(audio.channels as u16).to_be_bytes());
        data.extend_from_slice(&(audio.bits_per_sample as u16).to_be_bytes());
        data.extend_from_slice(&[0u8; 4]); // pre-defined and reserved
        data.extend_from_slice(&write_u32_be(audio.sample_rate << 16)); // sample rate (16.16)

        // Add esds box for AAC
        if let CodecId::Aac = track.stream.codec_id {
            let esds = self.build_esds(track)?;
            data.extend_from_slice(&esds);
        }

        // Wrap in entry box
        let mut result = Vec::with_capacity(data.len() + 8);
        result.extend_from_slice(&write_u32_be((data.len() + 8) as u32));
        result.extend_from_slice(b"mp4a");
        result.extend_from_slice(&data);

        Ok(result)
    }

    /// Build esds box for AAC.
    fn build_esds(&self, track: &TrackState) -> Result<Vec<u8>> {
        let empty_vec = Vec::new();
        let extra = track.stream.extra_data.as_ref().unwrap_or(&empty_vec);

        let mut data = Vec::new();
        data.extend_from_slice(&[0, 0, 0, 0]); // version and flags

        // ES descriptor
        data.push(0x03); // ES_DescrTag
        let es_desc_len = 23 + extra.len();
        data.push(es_desc_len as u8);
        data.extend_from_slice(&[0, 1]); // ES_ID
        data.push(0); // flags

        // DecoderConfigDescriptor
        data.push(0x04); // DecoderConfigDescrTag
        let dec_config_len = 15 + extra.len();
        data.push(dec_config_len as u8);
        data.push(0x40); // objectTypeIndication (AAC)
        data.push(0x15); // streamType (audio)
        data.extend_from_slice(&[0, 0, 0]); // buffer size
        data.extend_from_slice(&write_u32_be(128000)); // max bitrate
        data.extend_from_slice(&write_u32_be(128000)); // avg bitrate

        // DecoderSpecificInfo
        data.push(0x05); // DecSpecificInfoTag
        data.push(extra.len() as u8);
        data.extend_from_slice(extra);

        // SLConfigDescriptor
        data.push(0x06); // SLConfigDescrTag
        data.push(1);
        data.push(0x02);

        // Wrap in esds box
        let mut result = Vec::with_capacity(data.len() + 8);
        result.extend_from_slice(&write_u32_be((data.len() + 8) as u32));
        result.extend_from_slice(b"esds");
        result.extend_from_slice(&data);

        Ok(result)
    }

    /// Build stts atom.
    fn build_stts(&self, track: &TrackState) -> Vec<u8> {
        // Compress runs of same duration
        let mut entries: Vec<(u32, u32)> = Vec::new();

        for sample in &track.samples {
            if let Some(last) = entries.last_mut() {
                if last.1 == sample.duration {
                    last.0 += 1;
                    continue;
                }
            }
            entries.push((1, sample.duration));
        }

        let mut data = Vec::new();
        data.extend_from_slice(&[0, 0, 0, 0]); // version and flags
        data.extend_from_slice(&write_u32_be(entries.len() as u32));

        for (count, delta) in entries {
            data.extend_from_slice(&write_u32_be(count));
            data.extend_from_slice(&write_u32_be(delta));
        }

        let mut result = Vec::with_capacity(data.len() + 8);
        result.extend_from_slice(&write_u32_be((data.len() + 8) as u32));
        result.extend_from_slice(b"stts");
        result.extend_from_slice(&data);

        result
    }

    /// Build ctts atom.
    fn build_ctts(&self, track: &TrackState) -> Vec<u8> {
        let mut entries: Vec<(u32, i32)> = Vec::new();

        for sample in &track.samples {
            if let Some(last) = entries.last_mut() {
                if last.1 == sample.cts_offset {
                    last.0 += 1;
                    continue;
                }
            }
            entries.push((1, sample.cts_offset));
        }

        let mut data = Vec::new();
        data.extend_from_slice(&[1, 0, 0, 0]); // version 1 and flags
        data.extend_from_slice(&write_u32_be(entries.len() as u32));

        for (count, offset) in entries {
            data.extend_from_slice(&write_u32_be(count));
            data.extend_from_slice(&(offset as u32).to_be_bytes());
        }

        let mut result = Vec::with_capacity(data.len() + 8);
        result.extend_from_slice(&write_u32_be((data.len() + 8) as u32));
        result.extend_from_slice(b"ctts");
        result.extend_from_slice(&data);

        result
    }

    /// Build stss atom.
    fn build_stss(&self, track: &TrackState) -> Vec<u8> {
        let keyframes: Vec<u32> = track.samples
            .iter()
            .enumerate()
            .filter(|(_, s)| s.keyframe)
            .map(|(i, _)| i as u32 + 1)
            .collect();

        let mut data = Vec::new();
        data.extend_from_slice(&[0, 0, 0, 0]); // version and flags
        data.extend_from_slice(&write_u32_be(keyframes.len() as u32));

        for sample_num in keyframes {
            data.extend_from_slice(&write_u32_be(sample_num));
        }

        let mut result = Vec::with_capacity(data.len() + 8);
        result.extend_from_slice(&write_u32_be((data.len() + 8) as u32));
        result.extend_from_slice(b"stss");
        result.extend_from_slice(&data);

        result
    }

    /// Build stsc atom.
    fn build_stsc(&self, track: &TrackState) -> Vec<u8> {
        let mut entries: Vec<(u32, u32, u32)> = Vec::new();

        for (i, chunk) in track.chunks.iter().enumerate() {
            let samples_per_chunk = chunk.sample_count as u32;

            if entries.is_empty() || entries.last().unwrap().1 != samples_per_chunk {
                entries.push((i as u32 + 1, samples_per_chunk, 1));
            }
        }

        let mut data = Vec::new();
        data.extend_from_slice(&[0, 0, 0, 0]); // version and flags
        data.extend_from_slice(&write_u32_be(entries.len() as u32));

        for (first_chunk, samples, desc_idx) in entries {
            data.extend_from_slice(&write_u32_be(first_chunk));
            data.extend_from_slice(&write_u32_be(samples));
            data.extend_from_slice(&write_u32_be(desc_idx));
        }

        let mut result = Vec::with_capacity(data.len() + 8);
        result.extend_from_slice(&write_u32_be((data.len() + 8) as u32));
        result.extend_from_slice(b"stsc");
        result.extend_from_slice(&data);

        result
    }

    /// Build stsz atom.
    fn build_stsz(&self, track: &TrackState) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&[0, 0, 0, 0]); // version and flags
        data.extend_from_slice(&write_u32_be(0)); // sample size (variable)
        data.extend_from_slice(&write_u32_be(track.samples.len() as u32));

        for sample in &track.samples {
            data.extend_from_slice(&write_u32_be(sample.size));
        }

        let mut result = Vec::with_capacity(data.len() + 8);
        result.extend_from_slice(&write_u32_be((data.len() + 8) as u32));
        result.extend_from_slice(b"stsz");
        result.extend_from_slice(&data);

        result
    }

    /// Build stco or co64 atom.
    fn build_stco(&self, track: &TrackState) -> Vec<u8> {
        let use_64bit = track.chunks.iter().any(|c| c.offset > u32::MAX as u64);

        let mut data = Vec::new();
        data.extend_from_slice(&[0, 0, 0, 0]); // version and flags
        data.extend_from_slice(&write_u32_be(track.chunks.len() as u32));

        for chunk in &track.chunks {
            if use_64bit {
                data.extend_from_slice(&write_u64_be(chunk.offset));
            } else {
                data.extend_from_slice(&write_u32_be(chunk.offset as u32));
            }
        }

        let box_type = if use_64bit { b"co64" } else { b"stco" };

        let mut result = Vec::with_capacity(data.len() + 8);
        result.extend_from_slice(&write_u32_be((data.len() + 8) as u32));
        result.extend_from_slice(box_type);
        result.extend_from_slice(&data);

        result
    }
}

impl Default for Mp4Muxer {
    fn default() -> Self {
        Self::new()
    }
}

impl Muxer for Mp4Muxer {
    fn create<W: Write + Seek + Send + 'static>(&mut self, writer: W) -> Result<()> {
        self.writer = Some(Box::new(writer));
        Ok(())
    }

    fn format_name(&self) -> &str {
        "mp4"
    }

    fn add_stream(&mut self, info: StreamInfo) -> Result<usize> {
        let index = self.tracks.len();
        self.tracks.push(TrackState::new(info));
        Ok(index)
    }

    fn write_header(&mut self) -> Result<()> {
        self.write_ftyp()?;
        self.start_mdat()?;
        self.header_written = true;
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.header_written {
            return Err(Error::Container("Header not written".into()));
        }

        let track = self.tracks.get_mut(packet.stream_index as usize)
            .ok_or_else(|| Error::Container("Invalid stream index".into()))?;

        let writer = self.writer.as_mut().ok_or(Error::Container("No writer".into()))?;
        let offset = writer.stream_position()?;

        // Write sample data
        writer.write_all(packet.data())?;

        // Calculate duration (use value directly, default to 1 if zero)
        let duration = if packet.duration.is_zero() { 1 } else { packet.duration.value as u32 };

        // Calculate CTS offset using Timestamp values
        let cts_offset = if packet.pts.is_valid() && packet.dts.is_valid() {
            (packet.pts.value - packet.dts.value) as i32
        } else {
            0
        };

        // Add sample
        track.samples.push(SampleInfo {
            size: packet.data().len() as u32,
            duration,
            cts_offset,
            keyframe: packet.is_keyframe(),
        });

        // Add to current chunk or start new chunk
        if track.chunks.is_empty() || track.current_chunk_samples.len() >= 10 {
            // Start new chunk
            if !track.current_chunk_samples.is_empty() {
                track.chunks.push(ChunkInfo {
                    offset: track.current_chunk_offset,
                    first_sample: track.samples.len() - track.current_chunk_samples.len() - 1,
                    sample_count: track.current_chunk_samples.len(),
                });
                track.current_chunk_samples.clear();
            }
            track.current_chunk_offset = offset;
        }

        track.current_chunk_samples.push(packet.data().len() as u32);
        track.duration += duration as u64;

        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        // Flush remaining chunks
        for track in &mut self.tracks {
            if !track.current_chunk_samples.is_empty() {
                track.chunks.push(ChunkInfo {
                    offset: track.current_chunk_offset,
                    first_sample: track.samples.len() - track.current_chunk_samples.len(),
                    sample_count: track.current_chunk_samples.len(),
                });
                track.current_chunk_samples.clear();
            }
        }

        self.finish_mdat()?;
        self.write_moov()?;

        Ok(())
    }

    fn close(&mut self) {
        self.writer = None;
        self.tracks.clear();
    }
}
